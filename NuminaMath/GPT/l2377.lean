import Mathlib

namespace range_of_m_l2377_237726

theorem range_of_m (m : ℝ) : 
  (¬(-2 ≤ 1 - (x - 1) / 3 ∧ (1 - (x - 1) / 3 ≤ 2)) → (∀ x, m > 0 → x^2 - 2*x + 1 - m^2 > 0)) → 
  (40 ≤ m ∧ m < 50) :=
by
  sorry

end range_of_m_l2377_237726


namespace extreme_points_sum_gt_two_l2377_237713

noncomputable def f (x : ℝ) (b : ℝ) := x^2 / 2 + b * Real.exp x
noncomputable def f_prime (x : ℝ) (b : ℝ) := x + b * Real.exp x

theorem extreme_points_sum_gt_two
  (b : ℝ)
  (h_b : -1 / Real.exp 1 < b ∧ b < 0)
  (x₁ x₂ : ℝ)
  (h_x₁ : f_prime x₁ b = 0)
  (h_x₂ : f_prime x₂ b = 0)
  (h_x₁_lt_x₂ : x₁ < x₂) :
  x₁ + x₂ > 2 := by
  sorry

end extreme_points_sum_gt_two_l2377_237713


namespace problem1_problem2_l2377_237771

variables (x y : ℝ)

-- Given Conditions
def given_conditions :=
  (x = 2 + Real.sqrt 3) ∧ (y = 2 - Real.sqrt 3)

-- Problem 1
theorem problem1 (h : given_conditions x y) : x^2 + y^2 = 14 :=
sorry

-- Problem 2
theorem problem2 (h : given_conditions x y) : (x / y) - (y / x) = 8 * Real.sqrt 3 :=
sorry

end problem1_problem2_l2377_237771


namespace rectangle_length_width_l2377_237781

theorem rectangle_length_width (x y : ℝ) (h1 : 2 * (x + y) = 26) (h2 : x * y = 42) : 
  (x = 7 ∧ y = 6) ∨ (x = 6 ∧ y = 7) :=
by
  sorry

end rectangle_length_width_l2377_237781


namespace sum_of_ages_is_18_l2377_237735

-- Define the conditions
def product_of_ages (kiana twin : ℕ) := kiana * twin^2 = 128

-- Define the proof problem statement
theorem sum_of_ages_is_18 : ∃ (kiana twin : ℕ), product_of_ages kiana twin ∧ twin > kiana ∧ kiana + twin + twin = 18 :=
by
  sorry

end sum_of_ages_is_18_l2377_237735


namespace not_mutually_exclusive_option_D_l2377_237714

-- Definitions for mutually exclusive events
def mutually_exclusive (event1 event2 : Prop) : Prop := ¬ (event1 ∧ event2)

-- Conditions as given in the problem
def eventA1 : Prop := True -- Placeholder for "score is greater than 8"
def eventA2 : Prop := True -- Placeholder for "score is less than 6"

def eventB1 : Prop := True -- Placeholder for "90 seeds germinate"
def eventB2 : Prop := True -- Placeholder for "80 seeds germinate"

def eventC1 : Prop := True -- Placeholder for "pass rate is higher than 70%"
def eventC2 : Prop := True -- Placeholder for "pass rate is 70%"

def eventD1 : Prop := True -- Placeholder for "average score is not lower than 90"
def eventD2 : Prop := True -- Placeholder for "average score is not higher than 120"

-- Lean proof statement
theorem not_mutually_exclusive_option_D :
  mutually_exclusive eventA1 eventA2 ∧
  mutually_exclusive eventB1 eventB2 ∧
  mutually_exclusive eventC1 eventC2 ∧
  ¬ mutually_exclusive eventD1 eventD2 :=
sorry

end not_mutually_exclusive_option_D_l2377_237714


namespace prob_at_least_one_solves_l2377_237790

theorem prob_at_least_one_solves (p1 p2 : ℝ) (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (1 : ℝ) - (1 - p1) * (1 - p2) = 1 - ((1 - p1) * (1 - p2)) :=
by sorry

end prob_at_least_one_solves_l2377_237790


namespace number_of_whole_numbers_between_roots_l2377_237758

theorem number_of_whole_numbers_between_roots :
  let sqrt_18 := Real.sqrt 18
  let sqrt_98 := Real.sqrt 98
  Nat.card { x : ℕ | sqrt_18 < x ∧ x < sqrt_98 } = 5 := 
by
  sorry

end number_of_whole_numbers_between_roots_l2377_237758


namespace unique_10_tuple_solution_l2377_237766

noncomputable def condition (x : Fin 10 → ℝ) : Prop :=
  (1 - x 0)^2 +
  (x 0 - x 1)^2 + 
  (x 1 - x 2)^2 + 
  (x 2 - x 3)^2 + 
  (x 3 - x 4)^2 + 
  (x 4 - x 5)^2 + 
  (x 5 - x 6)^2 + 
  (x 6 - x 7)^2 + 
  (x 7 - x 8)^2 + 
  (x 8 - x 9)^2 + 
  x 9^2 + 
  (1/2) * (x 9 - x 0)^2 = 1/10

theorem unique_10_tuple_solution : 
  ∃! (x : Fin 10 → ℝ), condition x := 
sorry

end unique_10_tuple_solution_l2377_237766


namespace units_digit_of_6_to_the_6_l2377_237716

theorem units_digit_of_6_to_the_6 : (6^6) % 10 = 6 := by
  sorry

end units_digit_of_6_to_the_6_l2377_237716


namespace interval_between_segments_systematic_sampling_l2377_237722

theorem interval_between_segments_systematic_sampling 
  (total_students : ℕ) (sample_size : ℕ) 
  (h_total_students : total_students = 1000) 
  (h_sample_size : sample_size = 40):
  total_students / sample_size = 25 :=
by
  sorry

end interval_between_segments_systematic_sampling_l2377_237722


namespace teams_face_each_other_l2377_237746

theorem teams_face_each_other (n : ℕ) (total_games : ℕ) (k : ℕ)
  (h1 : n = 20)
  (h2 : total_games = 760)
  (h3 : total_games = n * (n - 1) * k / 2) :
  k = 4 :=
by
  sorry

end teams_face_each_other_l2377_237746


namespace combined_yearly_return_percentage_l2377_237754

-- Given conditions
def investment1 : ℝ := 500
def return_rate1 : ℝ := 0.07
def investment2 : ℝ := 1500
def return_rate2 : ℝ := 0.15

-- Question to prove
theorem combined_yearly_return_percentage :
  let yearly_return1 := investment1 * return_rate1
  let yearly_return2 := investment2 * return_rate2
  let total_yearly_return := yearly_return1 + yearly_return2
  let total_investment := investment1 + investment2
  ((total_yearly_return / total_investment) * 100) = 13 :=
by
  -- skipping the proof
  sorry

end combined_yearly_return_percentage_l2377_237754


namespace people_in_room_l2377_237708

variable (total_chairs occupied_chairs people_present : ℕ)
variable (h1 : total_chairs = 28)
variable (h2 : occupied_chairs = 14)
variable (h3 : (2 / 3 : ℚ) * people_present = 14)
variable (h4 : total_chairs = 2 * occupied_chairs)

theorem people_in_room : people_present = 21 := 
by 
  --proof will be here
  sorry

end people_in_room_l2377_237708


namespace three_numbers_equal_l2377_237734

theorem three_numbers_equal {a b c d : ℕ} 
  (h : ∀ {x y z w : ℕ}, (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
                  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ (w = a ∨ w = b ∨ w = c ∨ w = d) → x * y + z * w = x * z + y * w) :
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end three_numbers_equal_l2377_237734


namespace initial_percentage_decrease_l2377_237799

theorem initial_percentage_decrease (P x : ℝ) (h1 : 0 < P) (h2 : 0 ≤ x) (h3 : x ≤ 100) :
  ((P - (x / 100) * P) * 1.50 = P * 1.20) → x = 20 :=
by
  sorry

end initial_percentage_decrease_l2377_237799


namespace find_number_l2377_237796

-- Define the given conditions and statement as Lean types
theorem find_number (x : ℝ) :
  (0.3 * x > 0.6 * 50 + 30) -> x = 200 :=
by
  -- Proof here
  sorry

end find_number_l2377_237796


namespace chemist_target_temperature_fahrenheit_l2377_237793

noncomputable def kelvinToCelsius (K : ℝ) : ℝ := K - 273.15
noncomputable def celsiusToFahrenheit (C : ℝ) : ℝ := (C * 9 / 5) + 32

theorem chemist_target_temperature_fahrenheit :
  celsiusToFahrenheit (kelvinToCelsius (373.15 - 40)) = 140 :=
by
  sorry

end chemist_target_temperature_fahrenheit_l2377_237793


namespace part_a_l2377_237728

theorem part_a (cities : Finset (ℝ × ℝ)) (h_cities : cities.card = 100) 
  (distances : Finset (ℝ × ℝ → ℝ)) (h_distances : distances.card = 4950) :
  ∃ (erased_distance : ℝ × ℝ → ℝ), ¬ ∃ (restored_distance : ℝ × ℝ → ℝ), 
    restored_distance = erased_distance :=
sorry

end part_a_l2377_237728


namespace radii_touching_circles_l2377_237711

noncomputable def radius_of_circles_touching_unit_circles 
  (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
  (centerA centerB centerC : A) 
  (unit_radius : ℝ) (h1 : dist centerA centerB = 2 * unit_radius) 
  (h2 : dist centerB centerC = 2 * unit_radius) (h3 : dist centerC centerA = 2 * unit_radius) 
  : Prop :=
  ∃ r₁ r₂ : ℝ, r₁ = 1/3 ∧ r₂ = 7/3

theorem radii_touching_circles (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (centerA centerB centerC : A)
  (unit_radius : ℝ) (h1 : dist centerA centerB = 2 * unit_radius)
  (h2 : dist centerB centerC = 2 * unit_radius) (h3 : dist centerC centerA = 2 * unit_radius)
  : radius_of_circles_touching_unit_circles A B C centerA centerB centerC unit_radius h1 h2 h3 :=
sorry

end radii_touching_circles_l2377_237711


namespace olivia_earning_l2377_237751

theorem olivia_earning
  (cost_per_bar : ℝ)
  (total_bars : ℕ)
  (unsold_bars : ℕ)
  (sold_bars : ℕ := total_bars - unsold_bars)
  (earnings : ℝ := sold_bars * cost_per_bar) :
  cost_per_bar = 3 → total_bars = 7 → unsold_bars = 4 → earnings = 9 :=
by
  sorry

end olivia_earning_l2377_237751


namespace packages_of_noodles_tom_needs_l2377_237701

def beef_weight : ℕ := 10
def noodles_needed_factor : ℕ := 2
def noodles_available : ℕ := 4
def noodle_package_weight : ℕ := 2

theorem packages_of_noodles_tom_needs :
  (beef_weight * noodles_needed_factor - noodles_available) / noodle_package_weight = 8 :=
by
  sorry

end packages_of_noodles_tom_needs_l2377_237701


namespace constant_seq_arith_geo_l2377_237723

def is_arithmetic_sequence (s : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, s (n + 1) = s n + d

def is_geometric_sequence (s : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, s (n + 1) = s n * r

theorem constant_seq_arith_geo (s : ℕ → ℝ) (d r : ℝ) :
  is_arithmetic_sequence s d →
  is_geometric_sequence s r →
  (∃ c : ℝ, ∀ n : ℕ, s n = c) ∧ r = 1 :=
by
  sorry

end constant_seq_arith_geo_l2377_237723


namespace union_complement_A_when_a_eq_1_A_cap_B_eq_A_range_of_a_l2377_237738

def setA (a : ℝ) : Set ℝ := { x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3 }
def setB : Set ℝ := { x | -1 / 2 < x ∧ x < 2 }
def complementB : Set ℝ := { x | x ≤ -1 / 2 ∨ x ≥ 2 }

theorem union_complement_A_when_a_eq_1 :
  (complementB ∪ setA 1) = { x | x ≤ 1 ∨ x ≥ 2 } :=
by
  sorry

theorem A_cap_B_eq_A_range_of_a (a : ℝ) :
  (setA a ∩ setB = setA a) → (-1 < a ∧ a ≤ 1) :=
by
  sorry

end union_complement_A_when_a_eq_1_A_cap_B_eq_A_range_of_a_l2377_237738


namespace sum_of_roots_l2377_237756

theorem sum_of_roots (x : ℝ) (h : x^2 = 10 * x + 16) : x = 10 :=
by 
  -- Rearrange the equation to standard form: x^2 - 10x - 16 = 0
  have eqn : x^2 - 10 * x - 16 = 0 := by sorry
  -- Use the formula for the sum of the roots of a quadratic equation
  -- Prove the sum of the roots is 10
  sorry

end sum_of_roots_l2377_237756


namespace dog_weight_ratio_l2377_237755

theorem dog_weight_ratio
  (w7 : ℕ) (r : ℕ) (w13 : ℕ) (w21 : ℕ) (w52 : ℕ):
  (w7 = 6) →
  (w13 = 12 * r) →
  (w21 = 2 * w13) →
  (w52 = w21 + 30) →
  (w52 = 78) →
  r = 2 :=
by 
  sorry

end dog_weight_ratio_l2377_237755


namespace julian_younger_than_frederick_by_20_l2377_237772

noncomputable def Kyle: ℕ := 25
noncomputable def Tyson: ℕ := 20
noncomputable def Julian : ℕ := Kyle - 5
noncomputable def Frederick : ℕ := 2 * Tyson

theorem julian_younger_than_frederick_by_20 : Frederick - Julian = 20 :=
by
  sorry

end julian_younger_than_frederick_by_20_l2377_237772


namespace smallest_consecutive_integer_sum_l2377_237782

-- Definitions based on conditions
def consecutive_integer_sum (n : ℕ) := 20 * n + 190

-- Theorem statement
theorem smallest_consecutive_integer_sum : 
  ∃ (n k : ℕ), (consecutive_integer_sum n = k^3) ∧ (∀ m l : ℕ, (consecutive_integer_sum m = l^3) → k^3 ≤ l^3) :=
sorry

end smallest_consecutive_integer_sum_l2377_237782


namespace probability_same_color_ball_draw_l2377_237700

theorem probability_same_color_ball_draw (red white : ℕ) 
    (h_red : red = 2) (h_white : white = 2) : 
    let total_outcomes := (red + white) * (red + white)
    let same_color_outcomes := 2 * (red * red + white * white)
    same_color_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end probability_same_color_ball_draw_l2377_237700


namespace apples_to_grapes_equivalent_l2377_237791

-- Definitions based on the problem conditions
def apples := ℝ
def grapes := ℝ

-- Given conditions
def given_condition : Prop := (3 / 4) * 12 = 9

-- Question to prove
def question : Prop := (1 / 2) * 6 = 3

-- The theorem statement combining given conditions to prove the question
theorem apples_to_grapes_equivalent : given_condition → question := 
by
    intros
    sorry

end apples_to_grapes_equivalent_l2377_237791


namespace rectangle_area_increase_l2377_237720

theorem rectangle_area_increase (b : ℕ) (h1 : 2 * b = 40) (h2 : b = 20) : 
  let l := 2 * b
  let A_original := l * b
  let l_new := l - 5
  let b_new := b + 5
  let A_new := l_new * b_new
  A_new - A_original = 75 := 
by
  sorry

end rectangle_area_increase_l2377_237720


namespace triangle_area_l2377_237705

theorem triangle_area (P : ℝ) (r : ℝ) (s : ℝ) (A : ℝ) :
  P = 42 → r = 5 → s = P / 2 → A = r * s → A = 105 :=
by
  intro hP hr hs hA
  sorry

end triangle_area_l2377_237705


namespace find_a_plus_b_l2377_237727

variables {a b : ℝ}

theorem find_a_plus_b (h1 : a - b = -3) (h2 : a * b = 2) : a + b = Real.sqrt 17 ∨ a + b = -Real.sqrt 17 := by
  -- Proof can be filled in here
  sorry

end find_a_plus_b_l2377_237727


namespace f_correct_l2377_237718

noncomputable def f (n : ℕ) : ℕ :=
  if h : n ≥ 15 then (n - 1) / 2
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 2
  else if n = 6 then 4
  else if 7 ≤ n ∧ n ≤ 15 then 7
  else 0

theorem f_correct (n : ℕ) (hn : n ≥ 3) : 
  f n = if n ≥ 15 then (n - 1) / 2
        else if n = 3 then 1
        else if n = 4 then 1
        else if n = 5 then 2
        else if n = 6 then 4
        else if 7 ≤ n ∧ n ≤ 15 then 7
        else 0 := sorry

end f_correct_l2377_237718


namespace green_face_probability_l2377_237776

def probability_of_green_face (total_faces green_faces : Nat) : ℚ :=
  green_faces / total_faces

theorem green_face_probability :
  let total_faces := 10
  let green_faces := 3
  let blue_faces := 5
  let red_faces := 2
  probability_of_green_face total_faces green_faces = 3/10 :=
by
  sorry

end green_face_probability_l2377_237776


namespace solve_x_l2377_237780

noncomputable def op (a b : ℝ) : ℝ := (1 / b) - (1 / a)

theorem solve_x (x : ℝ) (h : op (x - 1) 2 = 1) : x = -1 := 
by {
  -- proof outline here...
  sorry
}

end solve_x_l2377_237780


namespace victors_friend_decks_l2377_237760

theorem victors_friend_decks:
  ∀ (deck_cost : ℕ) (victor_decks : ℕ) (total_spent : ℕ)
  (friend_decks : ℕ),
  deck_cost = 8 →
  victor_decks = 6 →
  total_spent = 64 →
  (victor_decks * deck_cost + friend_decks * deck_cost = total_spent) →
  friend_decks = 2 :=
by
  intros deck_cost victor_decks total_spent friend_decks hc hv ht heq
  sorry

end victors_friend_decks_l2377_237760


namespace common_ratio_geometric_sequence_l2377_237785

variable (a_n : ℕ → ℝ) (a1 : ℝ) (d : ℝ)

noncomputable def is_arithmetic_sequence (a_n : ℕ → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a_n n = a1 + n * d

noncomputable def forms_geometric_sequence (a_n : ℕ → ℝ) : Prop :=
(a_n 4) / (a_n 0) = (a_n 16) / (a_n 4)

theorem common_ratio_geometric_sequence :
  d ≠ 0 → 
  forms_geometric_sequence (a_n : ℕ → ℝ) →
  is_arithmetic_sequence a_n a1 d →
  ((a_n 4) / (a1) = 9) :=
by
  sorry

end common_ratio_geometric_sequence_l2377_237785


namespace perpendicular_line_sufficient_condition_l2377_237773

theorem perpendicular_line_sufficient_condition (a : ℝ) :
  (-a) * ((a + 2) / 3) = -1 ↔ (a = -3 ∨ a = 1) :=
by {
  sorry
}

#print perpendicular_line_sufficient_condition

end perpendicular_line_sufficient_condition_l2377_237773


namespace polynomial_quotient_l2377_237795

open Polynomial

noncomputable def dividend : ℤ[X] := 5 * X^4 - 9 * X^3 + 3 * X^2 + 7 * X - 6
noncomputable def divisor : ℤ[X] := X - 1

theorem polynomial_quotient :
  dividend /ₘ divisor = 5 * X^3 - 4 * X^2 + 7 * X + 7 :=
by
  sorry

end polynomial_quotient_l2377_237795


namespace count_complex_numbers_l2377_237725

theorem count_complex_numbers (a b : ℕ) (h_pos : a > 0 ∧ b > 0) (h_sum : a + b ≤ 5) : 
  ∃ n : ℕ, n = 10 :=
by
  sorry

end count_complex_numbers_l2377_237725


namespace find_sum_of_integers_l2377_237737

theorem find_sum_of_integers (x y : ℕ) (h_diff : x - y = 8) (h_prod : x * y = 180) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : x + y = 28 :=
by
  sorry

end find_sum_of_integers_l2377_237737


namespace cost_of_first_shirt_l2377_237707

theorem cost_of_first_shirt (x : ℝ) (h1 : x + (x + 6) = 24) : x + 6 = 15 :=
by
  sorry

end cost_of_first_shirt_l2377_237707


namespace line_equation_midpoint_ellipse_l2377_237743

theorem line_equation_midpoint_ellipse (x1 y1 x2 y2 : ℝ) 
  (h_midpoint_x : x1 + x2 = 4) (h_midpoint_y : y1 + y2 = 2)
  (h_ellipse_x1_y1 : (x1^2) / 12 + (y1^2) / 4 = 1) (h_ellipse_x2_y2 : (x2^2) / 12 + (y2^2) / 4 = 1) :
  2 * (x1 - x2) + 3 * (y1 - y2) = 0 :=
sorry

end line_equation_midpoint_ellipse_l2377_237743


namespace remainder_polynomial_l2377_237733

theorem remainder_polynomial (n : ℕ) (hn : n ≥ 2) : 
  ∃ Q R : Polynomial ℤ, (R.degree < 2) ∧ (X^n = Q * (X^2 - 4 * X + 3) + R) ∧ 
                       (R = (Polynomial.C ((3^n - 1) / 2) * X + Polynomial.C ((3 - 3^n) / 2))) :=
by
  sorry

end remainder_polynomial_l2377_237733


namespace interest_difference_l2377_237736

theorem interest_difference (P R T: ℝ) (hP: P = 2500) (hR: R = 8) (hT: T = 8) :
  let I := P * R * T / 100
  (P - I = 900) :=
by
  -- definition of I
  let I := P * R * T / 100
  -- proof goal
  sorry

end interest_difference_l2377_237736


namespace surface_area_increase_96_percent_l2377_237748

variable (s : ℝ)

def original_surface_area : ℝ := 6 * s^2
def new_edge_length : ℝ := 1.4 * s
def new_surface_area : ℝ := 6 * (new_edge_length s)^2

theorem surface_area_increase_96_percent :
  (new_surface_area s - original_surface_area s) / (original_surface_area s) * 100 = 96 :=
by
  simp [original_surface_area, new_edge_length, new_surface_area]
  sorry

end surface_area_increase_96_percent_l2377_237748


namespace find_digit_D_l2377_237703

theorem find_digit_D (A B C D : ℕ) (h1 : A + B = A + 10 * (B / 10)) (h2 : D + 10 * (A / 10) = A + C)
  (h3 : A + 10 * (B / 10) - C = A) (h4 : 0 ≤ A) (h5 : A ≤ 9) (h6 : 0 ≤ B) (h7 : B ≤ 9)
  (h8 : 0 ≤ C) (h9 : C ≤ 9) (h10 : 0 ≤ D) (h11 : D ≤ 9) : D = 9 := 
sorry

end find_digit_D_l2377_237703


namespace midpoint_coordinates_l2377_237783

theorem midpoint_coordinates (A B M : ℝ × ℝ) (hx : A = (2, -4)) (hy : B = (-6, 2)) (hm : M = (-2, -1)) :
  let (x1, y1) := A
  let (x2, y2) := B
  M = ((x1 + x2) / 2, (y1 + y2) / 2) :=
  sorry

end midpoint_coordinates_l2377_237783


namespace factor_polynomial_l2377_237715

theorem factor_polynomial (x : ℝ) :
  3 * x^2 * (x - 5) + 5 * (x - 5) = (3 * x^2 + 5) * (x - 5) :=
by
  sorry

end factor_polynomial_l2377_237715


namespace larger_number_is_588_l2377_237784

theorem larger_number_is_588
  (A B hcf : ℕ)
  (lcm_factors : ℕ × ℕ)
  (hcf_condition : hcf = 42)
  (lcm_factors_condition : lcm_factors = (12, 14))
  (hcf_prop : Nat.gcd A B = hcf)
  (lcm_prop : Nat.lcm A B = hcf * lcm_factors.1 * lcm_factors.2) :
  max (A) (B) = 588 :=
by
  sorry

end larger_number_is_588_l2377_237784


namespace value_of_c_l2377_237740

variables (a b c : ℝ)

theorem value_of_c :
  a + b = 3 ∧
  a * c + b = 18 ∧
  b * c + a = 6 →
  c = 7 :=
by
  intro h
  sorry

end value_of_c_l2377_237740


namespace maximum_volume_of_pyramid_l2377_237753

theorem maximum_volume_of_pyramid (a b : ℝ) (hb : b > 0) (ha : a > 0):
  ∃ V_max : ℝ, V_max = (a * (4 * b ^ 2 - a ^ 2)) / 12 := 
sorry

end maximum_volume_of_pyramid_l2377_237753


namespace symmetric_points_tangent_line_l2377_237757

theorem symmetric_points_tangent_line (k : ℝ) (hk : 0 < k) :
  (∃ P Q : ℝ × ℝ, P.2 = Real.exp P.1 ∧ ∃ x₀ : ℝ, 
    Q.2 = k * Q.1 ∧ Q = (P.2, P.1) ∧ 
    Q.1 = x₀ ∧ k = 1 / x₀ ∧ x₀ = Real.exp 1) → k = 1 / Real.exp 1 := 
by 
  sorry

end symmetric_points_tangent_line_l2377_237757


namespace circle_equation_unique_l2377_237747

theorem circle_equation_unique {F D E : ℝ} : 
  (∀ (x y : ℝ), (x = 0 ∧ y = 0) → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
  (∀ (x y : ℝ), (x = 1 ∧ y = 1) → x^2 + y^2 + D * x + E * y + F = 0) ∧ 
  (∀ (x y : ℝ), (x = 4 ∧ y = 2) → x^2 + y^2 + D * x + E * y + F = 0) → 
  (x^2 + y^2 - 8 * x + 6 * y = 0) :=
by 
  sorry

end circle_equation_unique_l2377_237747


namespace range_of_a_l2377_237721

variable (f : ℝ → ℝ)

noncomputable def is_decreasing (f : ℝ → ℝ) : Prop := 
  ∀ x y, x < y → f y < f x

theorem range_of_a 
  (decreasing_f : is_decreasing f)
  (hfdef : ∀ x, -1 ≤ x ∧ x ≤ 1 → f (2 * x - 3) < f (x - 2)) :
  ∃ a : ℝ, 1 < a ∧ a ≤ 2  :=
by 
  sorry

end range_of_a_l2377_237721


namespace second_polygon_sides_l2377_237775

theorem second_polygon_sides (s : ℝ) (P : ℝ) (n : ℕ) : 
  (50 * 3 * s = P) ∧ (n * s = P) → n = 150 := 
by {
  sorry
}

end second_polygon_sides_l2377_237775


namespace molecular_weights_correct_l2377_237797

-- Define atomic weights
def atomic_weight_Al : Float := 26.98
def atomic_weight_Cl : Float := 35.45
def atomic_weight_K : Float := 39.10

-- Define molecular weight calculations
def molecular_weight_AlCl3 : Float :=
  atomic_weight_Al + 3 * atomic_weight_Cl

def molecular_weight_KCl : Float :=
  atomic_weight_K + atomic_weight_Cl

-- Theorem statement to prove
theorem molecular_weights_correct :
  molecular_weight_AlCl3 = 133.33 ∧ molecular_weight_KCl = 74.55 :=
by
  -- This is where we would normally prove the equivalence
  sorry

end molecular_weights_correct_l2377_237797


namespace sin_cos_identity_l2377_237769

theorem sin_cos_identity (a : ℝ) (h : Real.sin (π - a) = -2 * Real.sin (π / 2 + a)) : 
  Real.sin a * Real.cos a = -2 / 5 :=
by
  sorry

end sin_cos_identity_l2377_237769


namespace dress_designs_possible_l2377_237739

theorem dress_designs_possible (colors patterns fabric_types : Nat) (color_choices : colors = 5) (pattern_choices : patterns = 6) (fabric_type_choices : fabric_types = 2) : 
  colors * patterns * fabric_types = 60 := by 
  sorry

end dress_designs_possible_l2377_237739


namespace num_of_B_sets_l2377_237724

def A : Set ℕ := {1, 2}

theorem num_of_B_sets (S : Set ℕ) (A : Set ℕ) (h : A = {1, 2}) (h1 : ∀ B : Set ℕ, A ∪ B = S) : 
  ∃ n : ℕ, n = 4 ∧ (∀ B : Set ℕ, B ⊆ {1, 2} → S = {1, 2}) :=
by {
  sorry
}

end num_of_B_sets_l2377_237724


namespace initial_apps_count_l2377_237706

theorem initial_apps_count (x A : ℕ) 
  (h₁ : A - 18 + x = 5) : A = 23 - x :=
by
  sorry

end initial_apps_count_l2377_237706


namespace units_digit_of_expression_l2377_237765

noncomputable def units_digit (n : ℕ) : ℕ :=
  n % 10

def expr : ℕ := 2 * (1 + 3 + 3^2 + 3^3 + 3^4 + 3^5 + 3^6 + 3^7 + 3^8 + 3^9)

theorem units_digit_of_expression : units_digit expr = 6 :=
by
  sorry

end units_digit_of_expression_l2377_237765


namespace complex_number_calculation_l2377_237768

theorem complex_number_calculation (i : ℂ) (h : i * i = -1) : i^7 - 2/i = i := 
by 
  sorry

end complex_number_calculation_l2377_237768


namespace profit_percentage_l2377_237702

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 60) (h_selling : selling_price = 78) :
  ((selling_price - cost_price) / cost_price) * 100 = 30 :=
by
  sorry

end profit_percentage_l2377_237702


namespace minimum_omega_l2377_237794

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)
noncomputable def h (ω : ℝ) (x : ℝ) : ℝ := f ω x + g ω x

theorem minimum_omega (ω : ℝ) (m : ℝ) 
  (h1 : 0 < ω)
  (h2 : ∀ x : ℝ, h ω m ≤ h ω x ∧ h ω x ≤ h ω (m + 1)) :
  ω = π :=
by
  sorry

end minimum_omega_l2377_237794


namespace john_pre_lunch_drive_l2377_237786

def drive_before_lunch (h : ℕ) : Prop :=
  45 * h + 45 * 3 = 225

theorem john_pre_lunch_drive : ∃ h : ℕ, drive_before_lunch h ∧ h = 2 :=
by
  sorry

end john_pre_lunch_drive_l2377_237786


namespace calculate_total_parts_l2377_237779

theorem calculate_total_parts (sample_size : ℕ) (draw_probability : ℚ) (N : ℕ) 
  (h_sample_size : sample_size = 30) 
  (h_draw_probability : draw_probability = 0.25) 
  (h_relation : sample_size = N * draw_probability) : 
  N = 120 :=
by
  rw [h_sample_size, h_draw_probability] at h_relation
  sorry

end calculate_total_parts_l2377_237779


namespace hyperbola_focus_coordinates_l2377_237752

theorem hyperbola_focus_coordinates:
  ∀ (x y : ℝ), 
    (x - 5)^2 / 7^2 - (y - 12)^2 / 10^2 = 1 → 
      ∃ (c : ℝ), c = 5 + Real.sqrt 149 ∧ (x, y) = (c, 12) :=
by
  intros x y h
  -- prove the coordinates of the focus with the larger x-coordinate are (5 + sqrt 149, 12)
  sorry

end hyperbola_focus_coordinates_l2377_237752


namespace kelly_baking_powder_difference_l2377_237774

theorem kelly_baking_powder_difference :
  let amount_yesterday := 0.4
  let amount_now := 0.3
  amount_yesterday - amount_now = 0.1 :=
by
  -- Definitions for amounts 
  let amount_yesterday := 0.4
  let amount_now := 0.3
  
  -- Applying definitions in the computation
  show amount_yesterday - amount_now = 0.1
  sorry

end kelly_baking_powder_difference_l2377_237774


namespace area_of_each_triangle_is_half_l2377_237764

structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def area (t : Triangle) : ℝ :=
  0.5 * |t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y)|

def A : Point := { x := 0, y := 0 }
def B : Point := { x := 1, y := 0 }
def C : Point := { x := 1, y := 1 }
def D : Point := { x := 0, y := 1 }
def K : Point := { x := 0.5, y := 1 }
def L : Point := { x := 0, y := 0.5 }
def M : Point := { x := 0.5, y := 0 }
def N : Point := { x := 1, y := 0.5 }

def AKB : Triangle := { p1 := A, p2 := K, p3 := B }
def BLC : Triangle := { p1 := B, p2 := L, p3 := C }
def CMD : Triangle := { p1 := C, p2 := M, p3 := D }
def DNA : Triangle := { p1 := D, p2 := N, p3 := A }

theorem area_of_each_triangle_is_half :
  area AKB = 0.5 ∧ area BLC = 0.5 ∧ area CMD = 0.5 ∧ area DNA = 0.5 := by sorry

end area_of_each_triangle_is_half_l2377_237764


namespace find_fraction_l2377_237745

theorem find_fraction {a b : ℕ} 
  (h1 : 32016 + (a / b) = 2016 * 3 + (a / b)) 
  (ha : a = 2016) 
  (hb : b = 2016^3 - 1) : 
  (b + 1) / a^2 = 2016 := 
by 
  sorry

end find_fraction_l2377_237745


namespace inequality_solution_set_l2377_237763

theorem inequality_solution_set (x : ℝ) :
  x^2 * (x^2 + 2*x + 1) > 2*x * (x^2 + 2*x + 1) ↔
  ((x < -1) ∨ (-1 < x ∧ x < 0) ∨ (2 < x)) :=
sorry

end inequality_solution_set_l2377_237763


namespace functional_eq_solution_l2377_237710

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x := 
sorry

end functional_eq_solution_l2377_237710


namespace find_value_of_expression_l2377_237770

theorem find_value_of_expression (x y z : ℝ)
  (h1 : 12 * x - 9 * y^2 = 7)
  (h2 : 6 * y - 9 * z^2 = -2)
  (h3 : 12 * z - 9 * x^2 = 4) : 
  6 * x^2 + 9 * y^2 + 12 * z^2 = 9 :=
  sorry

end find_value_of_expression_l2377_237770


namespace workers_count_l2377_237741

noncomputable def numberOfWorkers (W: ℕ) : Prop :=
  let old_supervisor_salary := 870
  let new_supervisor_salary := 690
  let avg_old := 430
  let avg_new := 410
  let total_after_old := (W + 1) * avg_old
  let total_after_new := 9 * avg_new
  total_after_old - old_supervisor_salary = total_after_new - new_supervisor_salary

theorem workers_count : numberOfWorkers 8 :=
by
  sorry

end workers_count_l2377_237741


namespace percentage_of_50_l2377_237712

theorem percentage_of_50 (P : ℝ) :
  (0.10 * 30) + (P / 100 * 50) = 10.5 → P = 15 := by
  sorry

end percentage_of_50_l2377_237712


namespace sequence_n_value_l2377_237704

theorem sequence_n_value (n : ℤ) : (2 * n^2 - 3 = 125) → (n = 8) := 
by {
    sorry
}

end sequence_n_value_l2377_237704


namespace second_smallest_N_prevent_Bananastasia_win_l2377_237761

-- Definition of the set S, as positive integers not divisible by any p^4.
def S : Set ℕ := {n | ∀ p : ℕ, Prime p → ¬ (p ^ 4 ∣ n)}

-- Definition of the game rules and the condition for Anastasia to prevent Bananastasia from winning.
-- N is a value such that for all a in S, it is not possible for Bananastasia to directly win.

theorem second_smallest_N_prevent_Bananastasia_win :
  ∃ N : ℕ, N = 625 ∧ (∀ a ∈ S, N - a ≠ 0 ∧ N - a ≠ 1) :=
by
  sorry

end second_smallest_N_prevent_Bananastasia_win_l2377_237761


namespace solve_for_y_l2377_237744

theorem solve_for_y (y : ℝ) (hy : y ≠ -2) : 
  (6 * y / (y + 2) - 2 / (y + 2) = 5 / (y + 2)) ↔ y = 7 / 6 :=
by sorry

end solve_for_y_l2377_237744


namespace fraction_division_l2377_237788

theorem fraction_division :
  (5 : ℚ) / ((13 : ℚ) / 7) = 35 / 13 :=
by
  sorry

end fraction_division_l2377_237788


namespace sample_size_drawn_l2377_237762

theorem sample_size_drawn (sample_size : ℕ) (probability : ℚ) (N : ℚ) 
  (h1 : sample_size = 30) 
  (h2 : probability = 0.25) 
  (h3 : probability = sample_size / N) : 
  N = 120 := by
  sorry

end sample_size_drawn_l2377_237762


namespace barefoot_kids_l2377_237709

theorem barefoot_kids (total_kids kids_socks kids_shoes kids_both : ℕ) 
  (h1 : total_kids = 22) 
  (h2 : kids_socks = 12) 
  (h3 : kids_shoes = 8) 
  (h4 : kids_both = 6) : 
  (total_kids - (kids_socks - kids_both + kids_shoes - kids_both + kids_both) = 8) :=
by
  -- following sorry to skip proof.
  sorry

end barefoot_kids_l2377_237709


namespace seq_sum_eq_314_l2377_237732

theorem seq_sum_eq_314 (d r : ℕ) (k : ℕ) (a_n b_n c_n : ℕ → ℕ)
  (h1 : ∀ n, a_n n = 1 + (n - 1) * d)
  (h2 : ∀ n, b_n n = r ^ (n - 1))
  (h3 : ∀ n, c_n n = a_n n + b_n n)
  (hk1 : c_n (k - 1) = 150)
  (hk2 : c_n (k + 1) = 900) :
  c_n k = 314 := by
  sorry

end seq_sum_eq_314_l2377_237732


namespace min_value_expression_l2377_237731

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    let a := 2
    let b := 3
    let term1 := 2*x + 1/(3*y)
    let term2 := 3*y + 1/(2*x)
    (term1 * (term1 - 2023) + term2 * (term2 - 2023)) = -2050529.5 :=
sorry

end min_value_expression_l2377_237731


namespace bulb_cheaper_than_lamp_by_4_l2377_237730

/-- Jim bought a $7 lamp and a bulb. The bulb cost a certain amount less than the lamp. 
    He bought 2 lamps and 6 bulbs and paid $32 in all. 
    The amount by which the bulb is cheaper than the lamp is $4. -/
theorem bulb_cheaper_than_lamp_by_4
  (lamp_price bulb_price : ℝ)
  (h1 : lamp_price = 7)
  (h2 : bulb_price = 7 - 4)
  (h3 : 2 * lamp_price + 6 * bulb_price = 32) :
  (7 - bulb_price = 4) :=
by {
  sorry
}

end bulb_cheaper_than_lamp_by_4_l2377_237730


namespace numbering_tube_contacts_l2377_237777

theorem numbering_tube_contacts {n : ℕ} (hn : n = 7) :
  ∃ (f g : ℕ → ℕ), (∀ k : ℕ, f k = k % n) ∧ (∀ k : ℕ, g k = (n - k) % n) ∧ 
  (∀ m : ℕ, ∃ k : ℕ, f (k + m) % n = g k % n) :=
by
  sorry

end numbering_tube_contacts_l2377_237777


namespace furniture_store_revenue_increase_l2377_237792

noncomputable def percentage_increase_in_gross (P R : ℕ) : ℚ :=
  ((0.80 * P) * (1.70 * R) - (P * R)) / (P * R) * 100

theorem furniture_store_revenue_increase (P R : ℕ) :
  percentage_increase_in_gross P R = 36 := 
by
  -- We include the conditions directly in the proof.
  -- Follow theorem from the given solution.
  sorry

end furniture_store_revenue_increase_l2377_237792


namespace avg_growth_rate_proof_l2377_237759

noncomputable def avg_growth_rate_correct_eqn (x : ℝ) : Prop :=
  40 * (1 + x)^2 = 48.4

theorem avg_growth_rate_proof (x : ℝ) 
  (h1 : 40 = avg_working_hours_first_week)
  (h2 : 48.4 = avg_working_hours_third_week) :
  avg_growth_rate_correct_eqn x :=
by 
  sorry

/- Defining the known conditions -/
def avg_working_hours_first_week : ℝ := 40
def avg_working_hours_third_week : ℝ := 48.4

end avg_growth_rate_proof_l2377_237759


namespace inequality_holds_l2377_237789

theorem inequality_holds (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 + 1 / x) * (1 + 1 / y) ≥ 9 :=
by sorry

end inequality_holds_l2377_237789


namespace ball_radius_l2377_237778

noncomputable def radius_of_ball (d h : ℝ) : ℝ :=
  let r := d / 2
  (325 / 20 : ℝ)

theorem ball_radius (d h : ℝ) (hd : d = 30) (hh : h = 10) :
  radius_of_ball d h = 16.25 := by
  sorry

end ball_radius_l2377_237778


namespace find_b_l2377_237717

def f (x : ℝ) : ℝ := 5 * x + 3

theorem find_b : ∃ b : ℝ, f b = -2 ∧ b = -1 := by
  have h : 5 * (-1 : ℝ) + 3 = -2 := by norm_num
  use -1
  simp [f, h]
  sorry

end find_b_l2377_237717


namespace password_count_l2377_237749

theorem password_count : ∃ s : Finset ℕ, s.card = 4 ∧ s.sum id = 27 ∧ 
  (s = {9, 8, 7, 3} ∨ s = {9, 8, 6, 4} ∨ s = {9, 7, 6, 5}) ∧ 
  (s.toList.permutations.length = 72) := sorry

end password_count_l2377_237749


namespace problem_solution_l2377_237767

variables {m n : ℝ}

theorem problem_solution (h1 : m^2 - n^2 = m * n) (h2 : m ≠ 0) (h3 : n ≠ 0) :
  (n / m) - (m / n) = -1 :=
sorry

end problem_solution_l2377_237767


namespace find_A_l2377_237750

def is_divisible (n : ℕ) (d : ℕ) : Prop := d ∣ n

noncomputable def valid_digit (A : ℕ) : Prop :=
  A < 10

noncomputable def digit_7_number := 653802 * 10

theorem find_A (A : ℕ) (h : valid_digit A) :
  is_divisible (digit_7_number + A) 2 ∧
  is_divisible (digit_7_number + A) 3 ∧
  is_divisible (digit_7_number + A) 4 ∧
  is_divisible (digit_7_number + A) 6 ∧
  is_divisible (digit_7_number + A) 8 ∧
  is_divisible (digit_7_number + A) 9 ∧
  is_divisible (digit_7_number + A) 25 →
  A = 0 :=
sorry

end find_A_l2377_237750


namespace solution_set_inequality_l2377_237742

theorem solution_set_inequality :
  {x : ℝ | (x^2 + 4) / (x - 4)^2 ≥ 0} = {x | x < 4} ∪ {x | x > 4} :=
by
  sorry

end solution_set_inequality_l2377_237742


namespace number_of_ways_to_adjust_items_l2377_237798

theorem number_of_ways_to_adjust_items :
  let items_on_upper_shelf := 4
  let items_on_lower_shelf := 8
  let move_items := 2
  let total_ways := Nat.choose items_on_lower_shelf move_items
  total_ways = 840 :=
by
  sorry

end number_of_ways_to_adjust_items_l2377_237798


namespace length_of_rooms_l2377_237729

-- Definitions based on conditions
def width : ℕ := 18
def num_rooms : ℕ := 20
def total_area : ℕ := 6840

-- Theorem stating the length of the rooms
theorem length_of_rooms : (total_area / num_rooms) / width = 19 := by
  sorry

end length_of_rooms_l2377_237729


namespace discount_percentage_correct_l2377_237787

-- Definitions corresponding to the conditions
def number_of_toys : ℕ := 5
def cost_per_toy : ℕ := 3
def total_price_paid : ℕ := 12
def original_price : ℕ := number_of_toys * cost_per_toy
def discount_amount : ℕ := original_price - total_price_paid
def discount_percentage : ℕ := (discount_amount * 100) / original_price

-- Statement of the problem
theorem discount_percentage_correct :
  discount_percentage = 20 := 
  sorry

end discount_percentage_correct_l2377_237787


namespace all_initial_rectangles_are_squares_l2377_237719

theorem all_initial_rectangles_are_squares (n : ℕ) (total_squares : ℕ) (h_prime : Nat.Prime total_squares) 
  (cut_rect_into_squares : ℕ → ℕ → ℕ → Prop) :
  ∀ (a b : ℕ), (∀ i, i < n → cut_rect_into_squares a b total_squares) → a = b :=
by 
  sorry

end all_initial_rectangles_are_squares_l2377_237719
