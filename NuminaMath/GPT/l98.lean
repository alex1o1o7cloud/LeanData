import Mathlib

namespace solve_inequality_l98_9825

theorem solve_inequality :
  { x : ℝ // 10 * x^2 - 2 * x - 3 < 0 } =
  { x : ℝ // (1 - Real.sqrt 31) / 10 < x ∧ x < (1 + Real.sqrt 31) / 10 } :=
by
  sorry

end solve_inequality_l98_9825


namespace larger_number_is_1629_l98_9872

theorem larger_number_is_1629 (x y : ℕ) (h1 : y - x = 1360) (h2 : y = 6 * x + 15) : y = 1629 := 
by 
  sorry

end larger_number_is_1629_l98_9872


namespace sequence_sixth_term_l98_9841

theorem sequence_sixth_term (a b c d : ℚ) : 
  (a = 1/4 * (5 + b)) →
  (b = 1/4 * (a + 45)) →
  (45 = 1/4 * (b + c)) →
  (c = 1/4 * (45 + d)) →
  d = 1877 / 3 :=
by
  sorry

end sequence_sixth_term_l98_9841


namespace find_s_l98_9854

theorem find_s (c d n r s : ℝ) 
(h1 : c * d = 3)
(h2 : ∃ p q : ℝ, (p + q = r) ∧ (p * q = s) ∧ (p = c + 1/d ∧ q = d + 1/c)) :
s = 16 / 3 :=
by
  sorry

end find_s_l98_9854


namespace number_of_classes_min_wins_for_class2101_l98_9887

-- Proof Problem for Q1
theorem number_of_classes (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 := sorry

-- Proof Problem for Q2
theorem min_wins_for_class2101 (y : ℕ) (h : y + (9 - y) = 9 ∧ 2 * y + (9 - y) >= 14) : y >= 5 := sorry

end number_of_classes_min_wins_for_class2101_l98_9887


namespace calculate_star_operation_l98_9888

def operation (a b : ℚ) : ℚ := 2 * a - b + 1

theorem calculate_star_operation :
  operation 1 (operation 3 (-2)) = -6 :=
by
  sorry

end calculate_star_operation_l98_9888


namespace student_average_comparison_l98_9812

theorem student_average_comparison (x y w : ℤ) (hxw : x < w) (hwy : w < y) : 
  (B : ℤ) > (A : ℤ) :=
  let A := (x + y + w) / 3
  let B := ((x + w) / 2 + y) / 2
  sorry

end student_average_comparison_l98_9812


namespace fraction_of_married_men_l98_9898

theorem fraction_of_married_men (total_women married_women : ℕ) 
    (h1 : total_women = 7)
    (h2 : married_women = 4)
    (single_women_probability : ℚ)
    (h3 : single_women_probability = 3 / 7) : 
    (4 / 11 : ℚ) = (married_women / (total_women + married_women)) := 
sorry

end fraction_of_married_men_l98_9898


namespace find_f_of_f_neg2_l98_9848

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f_of_f_neg2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by
  sorry

end find_f_of_f_neg2_l98_9848


namespace right_triangle_leg_length_l98_9853

theorem right_triangle_leg_length
  (a : ℕ) (c : ℕ) (h₁ : a = 8) (h₂ : c = 17) :
  ∃ b : ℕ, a^2 + b^2 = c^2 ∧ b = 15 :=
by
  sorry

end right_triangle_leg_length_l98_9853


namespace find_dividend_l98_9826

theorem find_dividend 
  (R : ℤ) 
  (Q : ℤ) 
  (D : ℤ) 
  (h1 : R = 8) 
  (h2 : D = 3 * Q) 
  (h3 : D = 3 * R + 3) : 
  (D * Q + R = 251) :=
by {
  -- The proof would follow, but for now, we'll use sorry.
  sorry
}

end find_dividend_l98_9826


namespace fewer_buses_than_cars_l98_9816

theorem fewer_buses_than_cars
  (bus_to_car_ratio : ℕ := 1)
  (cars_on_river_road : ℕ := 65)
  (cars_per_bus : ℕ := 13) :
  cars_on_river_road - (cars_on_river_road / cars_per_bus) = 60 :=
by
  sorry

end fewer_buses_than_cars_l98_9816


namespace travis_discount_percentage_l98_9808

theorem travis_discount_percentage (P D : ℕ) (hP : P = 2000) (hD : D = 1400) :
  ((P - D) / P * 100) = 30 := by
  -- sorry to skip the proof
  sorry

end travis_discount_percentage_l98_9808


namespace sum_possible_rs_l98_9813

theorem sum_possible_rs (r s : ℤ) (h1 : r ≠ s) (h2 : r + s = 24) : 
  ∃ sum : ℤ, sum = 1232 := 
sorry

end sum_possible_rs_l98_9813


namespace arrow_estimate_closest_to_9_l98_9880

theorem arrow_estimate_closest_to_9 
  (a b : ℝ) (h₁ : a = 8.75) (h₂ : b = 9.0)
  (h : 8.75 < 9.0) :
  ∃ x ∈ Set.Icc a b, x = 9.0 :=
by
  sorry

end arrow_estimate_closest_to_9_l98_9880


namespace blue_length_of_pencil_l98_9847

theorem blue_length_of_pencil (total_length purple_length black_length blue_length : ℝ)
  (h1 : total_length = 6)
  (h2 : purple_length = 3)
  (h3 : black_length = 2)
  (h4 : total_length = purple_length + black_length + blue_length)
  : blue_length = 1 :=
by
  sorry

end blue_length_of_pencil_l98_9847


namespace probability_X_equals_3_l98_9862

def total_score (a b : ℕ) : ℕ :=
  a + b

def prob_event_A_draws_yellow_B_draws_white : ℚ :=
  (2 / 5) * (3 / 4)

def prob_event_A_draws_white_B_draws_yellow : ℚ :=
  (3 / 5) * (2 / 4)

def prob_X_equals_3 : ℚ :=
  prob_event_A_draws_yellow_B_draws_white + prob_event_A_draws_white_B_draws_yellow

theorem probability_X_equals_3 :
  prob_X_equals_3 = 3 / 5 :=
by
  sorry

end probability_X_equals_3_l98_9862


namespace probability_of_b_in_rabbit_l98_9885

theorem probability_of_b_in_rabbit : 
  let word := "rabbit"
  let total_letters := 6
  let num_b_letters := 2
  (num_b_letters : ℚ) / total_letters = 1 / 3 :=
by
  sorry

end probability_of_b_in_rabbit_l98_9885


namespace angle_equality_iff_l98_9863

variables {A A' B B' C C' G : Point}

-- Define the angles as given in conditions
def angle_A'AC (A' A C : Point) : ℝ := sorry
def angle_ABB' (A B B' : Point) : ℝ := sorry
def angle_AC'C (A C C' : Point) : ℝ := sorry
def angle_AA'B (A A' B : Point) : ℝ := sorry

-- Main theorem statement
theorem angle_equality_iff :
  angle_A'AC A' A C = angle_ABB' A B B' ↔ angle_AC'C A C C' = angle_AA'B A A' B :=
sorry

end angle_equality_iff_l98_9863


namespace fractions_are_integers_l98_9868

theorem fractions_are_integers (a b : ℕ) (h1 : 1 < a) (h2 : 1 < b) 
    (h3 : abs ((a : ℚ) / b - (a - 1) / (b - 1)) = 1) : 
    ∃ m n : ℤ, (a : ℚ) / b = m ∧ (a - 1) / (b - 1) = n := 
sorry

end fractions_are_integers_l98_9868


namespace product_of_integers_l98_9844

theorem product_of_integers (A B C D : ℕ) 
  (h1 : A + B + C + D = 100) 
  (h2 : 2^A = B - 6) 
  (h3 : C + 6 = D)
  (h4 : B + C = D + 10) : 
  A * B * C * D = 33280 := 
by
  sorry

end product_of_integers_l98_9844


namespace average_score_of_class_l98_9876

variable (students_total : ℕ) (group1_students : ℕ) (group2_students : ℕ)
variable (group1_avg : ℝ) (group2_avg : ℝ)

theorem average_score_of_class :
  students_total = 20 → 
  group1_students = 10 → 
  group2_students = 10 → 
  group1_avg = 80 → 
  group2_avg = 60 → 
  (group1_students * group1_avg + group2_students * group2_avg) / students_total = 70 := 
by
  intros students_total_eq group1_students_eq group2_students_eq group1_avg_eq group2_avg_eq
  rw [students_total_eq, group1_students_eq, group2_students_eq, group1_avg_eq, group2_avg_eq]
  simp
  sorry

end average_score_of_class_l98_9876


namespace john_total_amount_l98_9832

-- Given conditions from a)
def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount

-- Problem statement
theorem john_total_amount : grandpa_amount + grandma_amount = 120 :=
by
  sorry

end john_total_amount_l98_9832


namespace negation_of_universal_l98_9874

theorem negation_of_universal : 
  (¬ (∀ x : ℝ, 2 * x^2 - x + 1 ≥ 0)) ↔ (∃ x : ℝ, 2 * x^2 - x + 1 < 0) :=
by
  sorry

end negation_of_universal_l98_9874


namespace ratio_of_a_b_l98_9822

theorem ratio_of_a_b (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a ≠ 0 ∧ b ≠ 0) : a / b = 3 / 2 :=
by sorry

end ratio_of_a_b_l98_9822


namespace tan_double_angle_l98_9897

open Real

-- Given condition
def condition (x : ℝ) : Prop := tan x - 1 / tan x = 3 / 2

-- Main theorem to prove
theorem tan_double_angle (x : ℝ) (h : condition x) : tan (2 * x) = -4 / 3 := by
  sorry

end tan_double_angle_l98_9897


namespace perpendicular_lines_l98_9824

def line_l1 (m x y : ℝ) : Prop := m * x - y + 1 = 0
def line_l2 (m x y : ℝ) : Prop := 2 * x - (m - 1) * y + 1 = 0

theorem perpendicular_lines (m : ℝ): (∃ x y : ℝ, line_l1 m x y) ∧ (∃ x y : ℝ, line_l2 m x y) ∧ (∀ x y : ℝ, line_l1 m x y → line_l2 m x y → m * (2 / (m - 1)) = -1) → m = 1 / 3 := by
  sorry

end perpendicular_lines_l98_9824


namespace harmonic_mean_of_x_and_y_l98_9865

noncomputable def x : ℝ := 88 + (40 / 100) * 88
noncomputable def y : ℝ := x - (25 / 100) * x
noncomputable def harmonic_mean (a b : ℝ) : ℝ := 2 / ((1 / a) + (1 / b))

theorem harmonic_mean_of_x_and_y :
  harmonic_mean x y = 105.6 :=
by
  sorry

end harmonic_mean_of_x_and_y_l98_9865


namespace sufficient_not_necessary_condition_l98_9804

noncomputable section

def is_hyperbola_point (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

def foci_distance_condition (P F1 F2 : ℝ × ℝ) : Prop :=
  |(P.1 - F1.1)^2 + (P.2 - F1.2)^2 - (P.1 - F2.1)^2 + (P.2 - F2.2)^2| = 6

theorem sufficient_not_necessary_condition 
  (x y F1_1 F1_2 F2_1 F2_2 : ℝ) (P : ℝ × ℝ)
  (P_hyp: is_hyperbola_point x y)
  (cond : foci_distance_condition P (F1_1, F1_2) (F2_1, F2_2)) :
  ∃ x y, is_hyperbola_point x y ∧ foci_distance_condition P (F1_1, F1_2) (F2_1, F2_2) :=
  sorry

end sufficient_not_necessary_condition_l98_9804


namespace merchant_markup_l98_9823

theorem merchant_markup (x : ℝ) : 
  let CP := 100
  let MP := CP + (x / 100) * CP
  let SP_discount := MP - 0.1 * MP 
  let SP_profit := CP + 57.5
  SP_discount = SP_profit → x = 75 :=
by
  intros
  let CP := (100 : ℝ)
  let MP := CP + (x / 100) * CP
  let SP_discount := MP - 0.1 * MP 
  let SP_profit := CP + 57.5
  have h : SP_discount = SP_profit := sorry
  sorry

end merchant_markup_l98_9823


namespace joe_collected_cards_l98_9894

theorem joe_collected_cards (boxes : ℕ) (cards_per_box : ℕ) (filled_boxes : boxes = 11) (max_cards_per_box : cards_per_box = 8) : boxes * cards_per_box = 88 := by
  sorry

end joe_collected_cards_l98_9894


namespace rectangle_dimensions_l98_9845

theorem rectangle_dimensions (w l : ℝ) 
  (h1 : l = 2 * w)
  (h2 : 2 * l + 2 * w = 3 * (l * w)) : 
  w = 1 ∧ l = 2 :=
by 
  sorry

end rectangle_dimensions_l98_9845


namespace parabola_vertex_l98_9810

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x y : ℝ, y^2 + 8*y + 4*x + 9 = 0 → x = -1/4 * (y + 4)^2 + 7/4)
  := 
  ⟨7/4, -4, sorry⟩

end parabola_vertex_l98_9810


namespace find_value_l98_9893

theorem find_value (x : ℝ) (h : x^2 - x - 1 = 0) : 2 * x^2 - 2 * x + 2021 = 2023 := 
by 
  sorry -- Proof needs to be provided

end find_value_l98_9893


namespace range_of_b_l98_9840

-- Definitions
def polynomial_inequality (b : ℝ) (x : ℝ) : Prop := x^2 + b * x - b - 3/4 > 0

-- The main statement
theorem range_of_b (b : ℝ) : (∀ x : ℝ, polynomial_inequality b x) ↔ -3 < b ∧ b < -1 :=
by {
    sorry -- proof goes here
}

end range_of_b_l98_9840


namespace solution_exists_l98_9851

namespace EquationSystem
-- Given the conditions of the equation system:
def eq1 (a b c d : ℝ) := a * b + a * c = 3 * b + 3 * c
def eq2 (a b c d : ℝ) := b * c + b * d = 5 * c + 5 * d
def eq3 (a b c d : ℝ) := a * c + c * d = 7 * a + 7 * d
def eq4 (a b c d : ℝ) := a * d + b * d = 9 * a + 9 * b

-- We need to prove that the solutions are as described:
theorem solution_exists (a b c d : ℝ) :
  eq1 a b c d → eq2 a b c d → eq3 a b c d → eq4 a b c d →
  (a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9) ∨ ∃ t : ℝ, a = t ∧ b = -t ∧ c = t ∧ d = -t :=
  by
    sorry
end EquationSystem

end solution_exists_l98_9851


namespace kaleb_books_count_l98_9834

/-- Kaleb's initial number of books. -/
def initial_books : ℕ := 34

/-- Number of books Kaleb sold. -/
def sold_books : ℕ := 17

/-- Number of new books Kaleb bought. -/
def new_books : ℕ := 7

/-- Prove the number of books Kaleb has now. -/
theorem kaleb_books_count : initial_books - sold_books + new_books = 24 := by
  sorry

end kaleb_books_count_l98_9834


namespace last_digit_of_two_exp_sum_l98_9838

theorem last_digit_of_two_exp_sum (m : ℕ) (h : 0 < m) : 
  ((2 ^ (m + 2007) + 2 ^ (m + 1)) % 10) = 0 :=
by
  -- proof will go here
  sorry

end last_digit_of_two_exp_sum_l98_9838


namespace wizard_concoction_valid_combinations_l98_9815

structure WizardConcoction :=
(herbs : Nat)
(crystals : Nat)
(single_incompatible : Nat)
(double_incompatible : Nat)

def valid_combinations (concoction : WizardConcoction) : Nat :=
  concoction.herbs * concoction.crystals - (concoction.single_incompatible + concoction.double_incompatible)

theorem wizard_concoction_valid_combinations (c : WizardConcoction)
  (h_herbs : c.herbs = 4)
  (h_crystals : c.crystals = 6)
  (h_single_incompatible : c.single_incompatible = 1)
  (h_double_incompatible : c.double_incompatible = 2) :
  valid_combinations c = 21 :=
by
  sorry

end wizard_concoction_valid_combinations_l98_9815


namespace length_of_square_side_l98_9850

theorem length_of_square_side (length_of_string : ℝ) (num_sides : ℕ) (total_side_length : ℝ) 
  (h1 : length_of_string = 32) (h2 : num_sides = 4) (h3 : total_side_length = length_of_string) : 
  total_side_length / num_sides = 8 :=
by
  sorry

end length_of_square_side_l98_9850


namespace evaluate_9_x_minus_1_l98_9861

theorem evaluate_9_x_minus_1 (x : ℝ) (h : (3 : ℝ)^(2 * x) = 16) : (9 : ℝ)^(x - 1) = 16 / 9 := by
  sorry

end evaluate_9_x_minus_1_l98_9861


namespace no_integer_roots_if_coefficients_are_odd_l98_9882

theorem no_integer_roots_if_coefficients_are_odd (a b c x : ℤ) 
  (h1 : Odd a) (h2 : Odd b) (h3 : Odd c) (h4 : a * x^2 + b * x + c = 0) : False := 
by
  sorry

end no_integer_roots_if_coefficients_are_odd_l98_9882


namespace simplify_fraction_l98_9877

variable {x y : ℝ}

theorem simplify_fraction (hx : x = 3) (hy : y = 4) : (12 * x * y^3) / (9 * x^3 * y^2) = 16 / 27 := by
  sorry

end simplify_fraction_l98_9877


namespace diamonds_in_G_20_equals_840_l98_9891

def diamonds_in_G (n : ℕ) : ℕ :=
  if n < 3 then 1 else 2 * n * (n + 1)

theorem diamonds_in_G_20_equals_840 : diamonds_in_G 20 = 840 :=
by
  sorry

end diamonds_in_G_20_equals_840_l98_9891


namespace negation_example_l98_9895

theorem negation_example : ¬(∀ x : ℝ, x^2 + |x| ≥ 0) ↔ ∃ x : ℝ, x^2 + |x| < 0 :=
by
  sorry

end negation_example_l98_9895


namespace value_of_x_l98_9842

theorem value_of_x (x : ℤ) (h : 3 * x / 7 = 21) : x = 49 :=
sorry

end value_of_x_l98_9842


namespace log_expression_value_l98_9859

theorem log_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) : 
  ((Real.log b / Real.log a) * (Real.log a / Real.log b))^2 = 1 := 
by 
  sorry

end log_expression_value_l98_9859


namespace product_of_integers_prime_at_most_one_prime_l98_9827

open Nat

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

theorem product_of_integers_prime_at_most_one_prime (a b p : ℤ) (hp : is_prime (Int.natAbs p)) (hprod : a * b = p) :
  (is_prime (Int.natAbs a) ∧ ¬is_prime (Int.natAbs b)) ∨ (¬is_prime (Int.natAbs a) ∧ is_prime (Int.natAbs b)) ∨ ¬is_prime (Int.natAbs a) ∧ ¬is_prime (Int.natAbs b) :=
sorry

end product_of_integers_prime_at_most_one_prime_l98_9827


namespace find_y_l98_9896

theorem find_y (x : ℝ) (h1 : x = 1.3333333333333333) (h2 : (x * y) / 3 = x^2) : y = 4 :=
by 
  sorry

end find_y_l98_9896


namespace determine_base_solution_l98_9803

theorem determine_base_solution :
  ∃ (h : ℕ), 
  h > 8 ∧ 
  (8 * h^3 + 6 * h^2 + 7 * h + 4) + (4 * h^3 + 3 * h^2 + 2 * h + 9) = 1 * h^4 + 3 * h^3 + 0 * h^2 + 0 * h + 3 ∧
  (9 + 4) = 13 ∧
  1 * h + 3 = 13 ∧
  (7 + 2 + 1) = 10 ∧
  1 * h + 0 = 10 ∧
  (6 + 3 + 1) = 10 ∧
  1 * h + 0 = 10 ∧
  (8 + 4 + 1) = 13 ∧
  1 * h + 3 = 13 ∧
  h = 10 :=
by
  sorry

end determine_base_solution_l98_9803


namespace total_spent_on_video_games_l98_9871

theorem total_spent_on_video_games (cost_basketball cost_racing : ℝ) (h_ball : cost_basketball = 5.20) (h_race : cost_racing = 4.23) : 
  cost_basketball + cost_racing = 9.43 :=
by
  sorry

end total_spent_on_video_games_l98_9871


namespace sequence_eighth_term_is_sixteen_l98_9866

-- Define the sequence based on given patterns
def oddPositionTerm (n : ℕ) : ℕ :=
  1 + 2 * (n - 1)

def evenPositionTerm (n : ℕ) : ℕ :=
  4 + 4 * (n - 1)

-- Formalize the proof problem
theorem sequence_eighth_term_is_sixteen : evenPositionTerm 4 = 16 :=
by 
  unfold evenPositionTerm
  sorry

end sequence_eighth_term_is_sixteen_l98_9866


namespace isosceles_triangles_height_ratio_l98_9837

theorem isosceles_triangles_height_ratio
  (b1 b2 h1 h2 : ℝ)
  (h1_ne_zero : h1 ≠ 0) 
  (h2_ne_zero : h2 ≠ 0)
  (equal_vertical_angles : ∀ (a1 a2 : ℝ), true) -- Placeholder for equal angles since it's not used directly
  (areas_ratio : (b1 * h1) / (b2 * h2) = 16 / 36)
  (similar_triangles : b1 / b2 = h1 / h2) :
  h1 / h2 = 2 / 3 :=
by
  sorry

end isosceles_triangles_height_ratio_l98_9837


namespace probability_green_cube_l98_9846

/-- A box contains 36 pink, 18 blue, 9 green, 6 red, and 3 purple cubes that are identical in size.
    Prove that the probability that a randomly selected cube is green is 1/8. -/
theorem probability_green_cube :
  let pink_cubes := 36
  let blue_cubes := 18
  let green_cubes := 9
  let red_cubes := 6
  let purple_cubes := 3
  let total_cubes := pink_cubes + blue_cubes + green_cubes + red_cubes + purple_cubes
  let probability := (green_cubes : ℚ) / total_cubes
  probability = 1 / 8 := 
by
  sorry

end probability_green_cube_l98_9846


namespace manny_received_fraction_l98_9805

-- Conditions
def total_marbles : ℕ := 400
def marbles_per_pack : ℕ := 10
def leo_kept_packs : ℕ := 25
def neil_received_fraction : ℚ := 1 / 8

-- Definition of total packs
def total_packs : ℕ := total_marbles / marbles_per_pack

-- Proof problem: What fraction of the total packs did Manny receive?
theorem manny_received_fraction :
  (total_packs - leo_kept_packs - neil_received_fraction * total_packs) / total_packs = 1 / 4 :=
by sorry

end manny_received_fraction_l98_9805


namespace not_sum_of_squares_or_cubes_in_ap_l98_9884

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * a + b * b = n

def is_sum_of_two_cubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a * a * a + b * b * b = n

def arithmetic_progression (a d k : ℕ) : ℕ :=
  a + d * k

theorem not_sum_of_squares_or_cubes_in_ap :
  ∀ k : ℕ, ¬ is_sum_of_two_squares (arithmetic_progression 31 36 k) ∧
           ¬ is_sum_of_two_cubes (arithmetic_progression 31 36 k) := by
  sorry

end not_sum_of_squares_or_cubes_in_ap_l98_9884


namespace traveled_distance_is_9_l98_9875

-- Let x be the usual speed in mph
variable (x : ℝ)
-- Let t be the usual time in hours
variable (t : ℝ)

-- Conditions
axiom condition1 : x * t = (x + 0.5) * (3 / 4 * t)
axiom condition2 : x * t = (x - 0.5) * (t + 3)

-- The journey distance d in miles
def distance_in_miles : ℝ := x * t

-- We can now state the theorem to prove that the distance he traveled is 9 miles
theorem traveled_distance_is_9 : distance_in_miles x t = 9 := by
  sorry

end traveled_distance_is_9_l98_9875


namespace first_day_speed_l98_9800

open Real

-- Define conditions
variables (v : ℝ) (t : ℝ)
axiom distance_home_school : 1.5 = v * (t - 7/60)
axiom second_day_condition : 1.5 = 6 * (t - 8/60)

theorem first_day_speed :
  v = 10 :=
by
  -- The proof will be provided here
  sorry

end first_day_speed_l98_9800


namespace lending_rate_l98_9889

noncomputable def principal: ℝ := 5000
noncomputable def rate_borrowed: ℝ := 4
noncomputable def time_years: ℝ := 2
noncomputable def gain_per_year: ℝ := 100

theorem lending_rate :
  ∃ (rate_lent: ℝ), 
  (principal * rate_lent * time_years / 100) - (principal * rate_borrowed * time_years / 100) / time_years = gain_per_year ∧
  rate_lent = 6 :=
by
  sorry

end lending_rate_l98_9889


namespace correct_statement_is_D_l98_9821

-- Define each statement as a proposition
def statement_A (a b c : ℕ) : Prop := c ≠ 0 → (a * c = b * c → a = b)
def statement_B : Prop := 30.15 = 30 + 15/60
def statement_C : Prop := ∀ (radius : ℕ), (radius ≠ 0) → (360 * (2 / (2 + 3 + 4)) = 90)
def statement_D : Prop := 9 * 30 + 40/2 = 50

-- Define the theorem to state the correct statement (D)
theorem correct_statement_is_D : statement_D :=
sorry

end correct_statement_is_D_l98_9821


namespace mod_problem_l98_9852

theorem mod_problem (x : ℤ) (h1 : 5 * x ≡ 10 [ZMOD 25]) (h2 : 4 * x ≡ 21 [ZMOD 25]) : (x^2 ≡ 21 [ZMOD 25]) :=
sorry

end mod_problem_l98_9852


namespace original_numerator_l98_9881

theorem original_numerator (n : ℕ) (hn : (n + 3) / (9 + 3) = 2 / 3) : n = 5 :=
by
  sorry

end original_numerator_l98_9881


namespace prism_surface_area_equals_three_times_volume_l98_9839

noncomputable def log_base (a x : ℝ) := Real.log x / Real.log a

theorem prism_surface_area_equals_three_times_volume (x : ℝ) 
  (h : 2 * (log_base 5 x * log_base 6 x + log_base 5 x * log_base 10 x + log_base 6 x * log_base 10 x) 
        = 3 * (log_base 5 x * log_base 6 x * log_base 10 x)) :
  x = Real.exp ((2 / 3) * Real.log 300) :=
sorry

end prism_surface_area_equals_three_times_volume_l98_9839


namespace sin_double_angle_value_l98_9819

theorem sin_double_angle_value 
  (h1 : Real.pi / 2 < α ∧ α < β ∧ β < 3 * Real.pi / 4)
  (h2 : Real.cos (α - β) = 12 / 13)
  (h3 : Real.sin (α + β) = -3 / 5) :
  Real.sin (2 * α) = -16 / 65 :=
by
  sorry

end sin_double_angle_value_l98_9819


namespace game_is_unfair_swap_to_make_fair_l98_9811

-- Part 1: Prove the game is unfair
theorem game_is_unfair (y b r : ℕ) (hb : y = 5) (bb : b = 13) (rb : r = 22) :
  ¬((b : ℚ) / (y + b + r) = (y : ℚ) / (y + b + r)) :=
by
  -- The proof is omitted as per the instructions.
  sorry

-- Part 2: Prove that swapping 4 black balls with 4 yellow balls makes the game fair.
theorem swap_to_make_fair (y b r : ℕ) (hb : y = 5) (bb : b = 13) (rb : r = 22) (x: ℕ) :
  x = 4 →
  (b - x : ℚ) / (y + b + r) = (y + x : ℚ) / (y + b + r) :=
by
  -- The proof is omitted as per the instructions.
  sorry

end game_is_unfair_swap_to_make_fair_l98_9811


namespace symmetric_circle_equation_l98_9806

theorem symmetric_circle_equation (x y : ℝ) :
  (x^2 + y^2 - 4 * x = 0) ↔ (-x ^ 2 + y^2 + 4 * x = 0) :=
sorry

end symmetric_circle_equation_l98_9806


namespace quadratic_translation_transformed_l98_9886

-- The original function is defined as follows:
def original_func (x : ℝ) : ℝ := 2 * x^2

-- Translated function left by 3 units
def translate_left (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x + a)

-- Translated function down by 2 units
def translate_down (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := f x - b

-- Combine both translations: left by 3 units and down by 2 units
def translated_func (x : ℝ) : ℝ := translate_down (translate_left original_func 3) 2 x

-- The theorem we want to prove
theorem quadratic_translation_transformed :
  translated_func x = 2 * (x + 3)^2 - 2 := 
by
  sorry

end quadratic_translation_transformed_l98_9886


namespace quadratic_inequality_solution_l98_9849

theorem quadratic_inequality_solution {a b : ℝ} 
  (h1 : (∀ x : ℝ, ax^2 - bx - 1 ≥ 0 ↔ (x = 1/3 ∨ x = 1/2))) : 
  ∃ a b : ℝ, (∀ x : ℝ, x^2 - b * x - a < 0 ↔ (-3 < x ∧ x < -2)) :=
by
  sorry

end quadratic_inequality_solution_l98_9849


namespace euler_totient_problem_l98_9879

open Nat

def is_odd (n : ℕ) := n % 2 = 1

def is_power_of_2 (m : ℕ) := ∃ k : ℕ, m = 2^k

theorem euler_totient_problem (n : ℕ) (h1 : n > 0) (h2 : is_odd n) (h3 : is_power_of_2 (φ n)) (h4 : is_power_of_2 (φ (n + 1))) :
  is_power_of_2 (n + 1) ∨ n = 5 := 
sorry

end euler_totient_problem_l98_9879


namespace maximum_number_of_workers_l98_9883

theorem maximum_number_of_workers :
  ∀ (n : ℕ), n ≤ 5 → 2 * n + 6 ≤ 16 :=
by
  intro n h
  have hn : n ≤ 5 := h
  linarith

end maximum_number_of_workers_l98_9883


namespace value_of_k_l98_9878

open Nat

theorem value_of_k (k : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n : ℕ, 0 < n → S n = k * (n : ℝ) ^ 2 + (n : ℝ))
  (h_a : ∀ n : ℕ, 1 < n → a n = S n - S (n-1))
  (h_geom : ∀ m : ℕ, 0 < m → (a m) ≠ 0 → (a (2*m))^2 = a m * a (4*m)) :
  k = 0 ∨ k = 1 :=
sorry

end value_of_k_l98_9878


namespace tom_bike_rental_hours_calculation_l98_9860

variable (h : ℕ)
variable (base_cost : ℕ := 17)
variable (hourly_rate : ℕ := 7)
variable (total_paid : ℕ := 80)

theorem tom_bike_rental_hours_calculation (h : ℕ) 
  (base_cost : ℕ := 17) (hourly_rate : ℕ := 7) (total_paid : ℕ := 80) 
  (hours_eq : total_paid = base_cost + hourly_rate * h) : 
  h = 9 := 
by
  -- The proof is omitted.
  sorry

end tom_bike_rental_hours_calculation_l98_9860


namespace point_on_line_l98_9833

theorem point_on_line (k : ℝ) (x y : ℝ) (h : x = -1/3 ∧ y = 4) (line_eq : 1 + 3 * k * x = -4 * y) : k = 17 :=
by
  rcases h with ⟨hx, hy⟩
  sorry

end point_on_line_l98_9833


namespace largest_k_dividing_A_l98_9828

def A : ℤ := 1990^(1991^1992) + 1991^(1990^1992) + 1992^(1991^1990)

theorem largest_k_dividing_A :
  1991^(1991) ∣ A := sorry

end largest_k_dividing_A_l98_9828


namespace arithmetic_sequence_problem_l98_9892

variable {a : ℕ → ℕ} -- Assuming a_n is a function from natural numbers to natural numbers

theorem arithmetic_sequence_problem (h1 : a 1 + a 2 = 10) (h2 : a 4 = a 3 + 2) :
  a 3 + a 4 = 18 :=
sorry

end arithmetic_sequence_problem_l98_9892


namespace scarves_sold_at_new_price_l98_9836

theorem scarves_sold_at_new_price :
  ∃ (p : ℕ), (∃ (c k : ℕ), (k = p * c) ∧ (p = 30) ∧ (c = 10)) ∧
  (∃ (new_c : ℕ), new_c = 165 / 10 ∧ k = new_p * new_c) ∧
  new_p = 18
:=
sorry

end scarves_sold_at_new_price_l98_9836


namespace roots_of_quadratic_l98_9809

theorem roots_of_quadratic (x : ℝ) : (x * (x - 2) = 2 - x) ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end roots_of_quadratic_l98_9809


namespace eq_1_solution_eq_2_solution_eq_3_solution_eq_4_solution_l98_9899

-- Equation (1): 2x^2 + 2x - 1 = 0
theorem eq_1_solution (x : ℝ) :
  2 * x^2 + 2 * x - 1 = 0 ↔ (x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2) := by
  sorry

-- Equation (2): x(x-1) = 2(x-1)
theorem eq_2_solution (x : ℝ) :
  x * (x - 1) = 2 * (x - 1) ↔ (x = 1 ∨ x = 2) := by
  sorry

-- Equation (3): 4(x-2)^2 = 9(2x+1)^2
theorem eq_3_solution (x : ℝ) :
  4 * (x - 2)^2 = 9 * (2 * x + 1)^2 ↔ (x = -7 / 4 ∨ x = 1 / 8) := by
  sorry

-- Equation (4): (2x-1)^2 - 3(2x-1) = 4
theorem eq_4_solution (x : ℝ) :
  (2 * x - 1)^2 - 3 * (2 * x - 1) = 4 ↔ (x = 5 / 2 ∨ x = 0) := by
  sorry

end eq_1_solution_eq_2_solution_eq_3_solution_eq_4_solution_l98_9899


namespace marble_287_is_blue_l98_9820

def marble_color (n : ℕ) : String :=
  if n % 15 < 6 then "blue"
  else if n % 15 < 11 then "green"
  else "red"

theorem marble_287_is_blue : marble_color 287 = "blue" :=
by
  sorry

end marble_287_is_blue_l98_9820


namespace actual_distance_traveled_l98_9855

theorem actual_distance_traveled 
  (D : ℝ) (t : ℝ)
  (h1 : 8 * t = D)
  (h2 : 12 * t = D + 20) : 
  D = 40 :=
by
  sorry

end actual_distance_traveled_l98_9855


namespace point_translation_l98_9830

variable (P Q : (ℝ × ℝ))
variable (dx : ℝ) (dy : ℝ)

theorem point_translation (hP : P = (-1, 2)) (hdx : dx = 2) (hdy : dy = 3) :
  Q = (P.1 + dx, P.2 - dy) → Q = (1, -1) := by
  sorry

end point_translation_l98_9830


namespace average_of_data_is_six_l98_9801

def data : List ℕ := [4, 6, 5, 8, 7, 6]

theorem average_of_data_is_six : 
  (data.sum / data.length : ℚ) = 6 := 
by sorry

end average_of_data_is_six_l98_9801


namespace simplify_expression_l98_9857

theorem simplify_expression : 
  18 * (8 / 15) * (3 / 4) = 12 / 5 := 
by 
  sorry

end simplify_expression_l98_9857


namespace infinite_solutions_imply_values_l98_9869

theorem infinite_solutions_imply_values (a b : ℝ) :
  (∀ x : ℝ, a * (2 * x + b) = 12 * x + 5) ↔ (a = 6 ∧ b = 5 / 6) :=
by
  sorry

end infinite_solutions_imply_values_l98_9869


namespace base7_arithmetic_l98_9818

theorem base7_arithmetic : 
  let b1000 := 343  -- corresponding to 1000_7 in decimal
  let b666 := 342   -- corresponding to 666_7 in decimal
  let b1234 := 466  -- corresponding to 1234_7 in decimal
  let s := b1000 + b666  -- sum in decimal
  let s_base7 := 1421    -- sum back in base7 (1421 corresponds to 685 in decimal)
  let r_base7 := 254     -- result from subtraction in base7 (254 corresponds to 172 in decimal)
  (1000 * 7^0 + 0 * 7^1 + 0 * 7^2 + 1 * 7^3) + (6 * 7^0 + 6 * 7^1 + 6 * 7^2) - (4 * 7^0 + 3 * 7^1 + 2 * 7^2 + 1 * 7^3) = (4 * 7^0 + 5 * 7^1 + 2 * 7^2)
  :=
sorry

end base7_arithmetic_l98_9818


namespace largest_n_digit_number_divisible_by_89_l98_9802

theorem largest_n_digit_number_divisible_by_89 (n : ℕ) (h1 : n % 2 = 1) (h2 : 3 ≤ n ∧ n ≤ 7) :
  ∃ x, x = 9999951 ∧ (x % 89 = 0 ∧ (10 ^ (n-1) ≤ x ∧ x < 10 ^ n)) :=
by
  sorry

end largest_n_digit_number_divisible_by_89_l98_9802


namespace last_three_digits_of_7_pow_210_l98_9807

theorem last_three_digits_of_7_pow_210 : (7^210) % 1000 = 599 := by
  sorry

end last_three_digits_of_7_pow_210_l98_9807


namespace arithmetic_sequence_sum_l98_9831

theorem arithmetic_sequence_sum :
  ∀(a_n : ℕ → ℕ) (S : ℕ → ℕ) (a_1 d : ℕ),
    (∀ n, a_n n = a_1 + (n - 1) * d) →
    (∀ n, S n = n * (a_1 + (n - 1) * d) / 2) →
    a_1 = 2 →
    S 4 = 20 →
    S 6 = 42 :=
by
  sorry

end arithmetic_sequence_sum_l98_9831


namespace percent_of_x_eq_21_percent_l98_9856

theorem percent_of_x_eq_21_percent (x : Real) : (0.21 * x = 0.30 * 0.70 * x) := by
  sorry

end percent_of_x_eq_21_percent_l98_9856


namespace hands_opposite_22_times_in_day_l98_9835

def clock_hands_opposite_in_day : ℕ := 22

def minute_hand_speed := 12
def opposite_line_minutes := 30

theorem hands_opposite_22_times_in_day (minute_hand_speed: ℕ) (opposite_line_minutes : ℕ) : 
  minute_hand_speed = 12 →
  opposite_line_minutes = 30 →
  clock_hands_opposite_in_day = 22 :=
by
  intros h1 h2
  sorry

end hands_opposite_22_times_in_day_l98_9835


namespace fraction_length_EF_of_GH_l98_9870

theorem fraction_length_EF_of_GH (GH GE EH GF FH EF : ℝ)
  (h1 : GE = 3 * EH)
  (h2 : GF = 4 * FH)
  (h3 : GE + EH = GH)
  (h4 : GF + FH = GH) :
  EF / GH = 1 / 20 := by 
  sorry

end fraction_length_EF_of_GH_l98_9870


namespace remainder_sum_59_l98_9817

theorem remainder_sum_59 (x y z : ℕ) (h1 : x % 59 = 30) (h2 : y % 59 = 27) (h3 : z % 59 = 4) :
  (x + y + z) % 59 = 2 := 
sorry

end remainder_sum_59_l98_9817


namespace largest_angle_in_triangle_l98_9873

theorem largest_angle_in_triangle (A B C : ℝ) (h₁ : A + B = 126) (h₂ : A = B + 20) (h₃ : A + B + C = 180) :
  max A (max B C) = 73 := sorry

end largest_angle_in_triangle_l98_9873


namespace arnold_danny_age_l98_9814

theorem arnold_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 15) : x = 7 :=
sorry

end arnold_danny_age_l98_9814


namespace union_area_of_reflected_triangles_l98_9890

open Real

noncomputable def pointReflected (P : ℝ × ℝ) (line_y : ℝ) : ℝ × ℝ :=
  (P.1, 2 * line_y - P.2)

def areaOfTriangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem union_area_of_reflected_triangles :
  let A := (2, 6)
  let B := (5, -2)
  let C := (7, 3)
  let line_y := 2
  let A' := pointReflected A line_y
  let B' := pointReflected B line_y
  let C' := pointReflected C line_y
  areaOfTriangle A B C + areaOfTriangle A' B' C' = 29 := sorry

end union_area_of_reflected_triangles_l98_9890


namespace no_positive_integral_solutions_l98_9858

theorem no_positive_integral_solutions (x y : ℕ) (h : x > 0) (k : y > 0) :
  x^4 * y^4 - 8 * x^2 * y^2 + 12 ≠ 0 :=
by
  sorry

end no_positive_integral_solutions_l98_9858


namespace proof_2720000_scientific_l98_9829

def scientific_notation (n : ℕ) : ℝ := 
  2.72 * 10^6 

theorem proof_2720000_scientific :
  scientific_notation 2720000 = 2.72 * 10^6 := by
  sorry

end proof_2720000_scientific_l98_9829


namespace find_number_l98_9867

-- Define the certain number x
variable (x : ℤ)

-- Define the conditions as given in part a)
def conditions : Prop :=
  x + 10 - 2 = 44

-- State the theorem that we need to prove
theorem find_number (h : conditions x) : x = 36 :=
by sorry

end find_number_l98_9867


namespace water_bottle_size_l98_9864

-- Define conditions
def glasses_per_day : ℕ := 4
def ounces_per_glass : ℕ := 5
def fills_per_week : ℕ := 4
def days_per_week : ℕ := 7

-- Theorem statement
theorem water_bottle_size :
  (glasses_per_day * ounces_per_glass * days_per_week) / fills_per_week = 35 :=
by
  sorry

end water_bottle_size_l98_9864


namespace shaded_rectangle_ratio_l98_9843

/-- Define conditions involved in the problem -/
def side_length_large_square : ℕ := 50
def num_rows_cols_grid : ℕ := 5
def rows_spanned_rect : ℕ := 2
def cols_spanned_rect : ℕ := 3

/-- Calculate the side length of a small square in the grid -/
def side_length_small_square := side_length_large_square / num_rows_cols_grid

/-- Calculate the area of the large square -/
def area_large_square := side_length_large_square * side_length_large_square

/-- Calculate the area of the shaded rectangle -/
def area_shaded_rectangle :=
  (rows_spanned_rect * side_length_small_square) *
  (cols_spanned_rect * side_length_small_square)

/-- Prove the ratio of the shaded rectangle's area to the large square's area -/
theorem shaded_rectangle_ratio : 
  (area_shaded_rectangle : ℚ) / area_large_square = 6/25 := by
  sorry

end shaded_rectangle_ratio_l98_9843
