import Mathlib

namespace travel_time_equation_l99_9957

theorem travel_time_equation
 (d : ℝ) (x t_saved factor : ℝ) 
 (h : d = 202) 
 (h1 : t_saved = 1.8) 
 (h2 : factor = 1.6)
 : (d / x) * factor = d / (x - t_saved) := sorry

end travel_time_equation_l99_9957


namespace min_power_for_84_to_divide_336_l99_9901

theorem min_power_for_84_to_divide_336 : 
  ∃ n : ℕ, (∀ m : ℕ, 84^m % 336 = 0 → m ≥ n) ∧ n = 2 := 
sorry

end min_power_for_84_to_divide_336_l99_9901


namespace crayons_lost_or_given_away_correct_l99_9948

def initial_crayons : ℕ := 606
def remaining_crayons : ℕ := 291
def crayons_lost_or_given_away : ℕ := initial_crayons - remaining_crayons

theorem crayons_lost_or_given_away_correct :
  crayons_lost_or_given_away = 315 :=
by
  sorry

end crayons_lost_or_given_away_correct_l99_9948


namespace f_at_three_bounds_l99_9973

theorem f_at_three_bounds (a c : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = a * x^2 - c)
  (h2 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h3 : -1 ≤ f 2 ∧ f 2 ≤ 5) : -1 ≤ f 3 ∧ f 3 ≤ 20 :=
sorry

end f_at_three_bounds_l99_9973


namespace triangular_square_l99_9904

def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_square (m n : ℕ) (h1 : 1 ≤ m) (h2 : 1 ≤ n) (h3 : 2 * triangular m = triangular n) :
  ∃ k : ℕ, triangular (2 * m - n) = k * k :=
by
  sorry

end triangular_square_l99_9904


namespace sum_first_n_terms_arithmetic_sequence_l99_9936

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n m : ℕ, a (m + 1) - a m = d

theorem sum_first_n_terms_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (a12 : a 12 = -8) (S9 : S 9 = -9) (h_arith : is_arithmetic_sequence a) :
  S 16 = -72 :=
sorry

end sum_first_n_terms_arithmetic_sequence_l99_9936


namespace complex_number_location_second_quadrant_l99_9956

theorem complex_number_location_second_quadrant (z : ℂ) (h : z / (1 + I) = I) : z.re < 0 ∧ z.im > 0 :=
by sorry

end complex_number_location_second_quadrant_l99_9956


namespace total_swordfish_caught_l99_9971

theorem total_swordfish_caught (fishing_trips : ℕ) (shelly_each_trip : ℕ) (sam_each_trip : ℕ) : 
  shelly_each_trip = 3 → 
  sam_each_trip = 2 → 
  fishing_trips = 5 → 
  (shelly_each_trip + sam_each_trip) * fishing_trips = 25 :=
by
  sorry

end total_swordfish_caught_l99_9971


namespace find_width_of_river_l99_9941

theorem find_width_of_river
    (total_distance : ℕ)
    (river_width : ℕ)
    (prob_find_item : ℚ)
    (h1 : total_distance = 500)
    (h2 : prob_find_item = 4/5)
    : river_width = 100 :=
by
    sorry

end find_width_of_river_l99_9941


namespace total_time_is_12_years_l99_9921

noncomputable def total_time_spent (shape_years climb_years_per_summit dive_months cave_years : ℕ) : ℕ :=
  shape_years + (2 * shape_years) + (7 * climb_years_per_summit) / 12 + ((7 * climb_years_per_summit) % 12) / 12 + (dive_months + 12) / 12 + cave_years

theorem total_time_is_12_years :
  total_time_spent 2 5 13 2 = 12 :=
by
  sorry

end total_time_is_12_years_l99_9921


namespace parabola_x_intercept_y_intercept_point_l99_9975

theorem parabola_x_intercept_y_intercept_point (a b w : ℝ) 
  (h1 : a = -1) 
  (h2 : b = 4) 
  (h3 : ∀ x : ℝ, x = 0 → w = 8): 
  ∃ (w : ℝ), w = 8 := 
by
  sorry

end parabola_x_intercept_y_intercept_point_l99_9975


namespace no_valid_sequence_of_integers_from_1_to_2004_l99_9991

theorem no_valid_sequence_of_integers_from_1_to_2004 :
  ¬ ∃ (a : ℕ → ℕ), 
    (∀ i, 1 ≤ a i ∧ a i ≤ 2004) ∧ 
    (∀ i j, i ≠ j → a i ≠ a j) ∧ 
    (∀ k, 1 ≤ k ∧ k + 9 ≤ 2004 → 
      (a k + a (k + 1) + a (k + 2) + a (k + 3) + a (k + 4) + a (k + 5) + 
       a (k + 6) + a (k + 7) + a (k + 8) + a (k + 9)) % 10 = 0) :=
  sorry

end no_valid_sequence_of_integers_from_1_to_2004_l99_9991


namespace angles_equal_l99_9972

theorem angles_equal {α β γ α1 β1 γ1 : ℝ} (h1 : α + β + γ = 180) (h2 : α1 + β1 + γ1 = 180) 
  (h_eq_or_sum_to_180 : (α = α1 ∨ α + α1 = 180) ∧ (β = β1 ∨ β + β1 = 180) ∧ (γ = γ1 ∨ γ + γ1 = 180)) :
  α = α1 ∧ β = β1 ∧ γ = γ1 := 
by 
  sorry

end angles_equal_l99_9972


namespace quadratic_has_distinct_real_roots_l99_9961

theorem quadratic_has_distinct_real_roots (m : ℝ) (hm : m ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (m * x1^2 - 2 * x1 + 3 = 0) ∧ (m * x2^2 - 2 * x2 + 3 = 0) ↔ 0 < m ∧ m < (1 / 3) :=
by
  sorry

end quadratic_has_distinct_real_roots_l99_9961


namespace number_of_pairs_l99_9987

open Nat

theorem number_of_pairs :
  ∃ n, n = 9 ∧
    (∃ x y : ℕ,
      x > 0 ∧ y > 0 ∧
      x + y = 150 ∧
      x % 3 = 0 ∧
      y % 5 = 0 ∧
      (∃! (x y : ℕ), x + y = 150 ∧ x % 3 = 0 ∧ y % 5 = 0 ∧ x > 0 ∧ y > 0)) := sorry

end number_of_pairs_l99_9987


namespace A_inter_B_domain_l99_9983

def A_domain : Set ℝ := {x : ℝ | x^2 + x - 2 >= 0}
def B_domain : Set ℝ := {x : ℝ | (2*x + 6)/(3 - x) >= 0 ∧ x ≠ -2}

theorem A_inter_B_domain :
  (A_domain ∩ B_domain) = {x : ℝ | (1 <= x ∧ x < 3) ∨ (-3 <= x ∧ x < -2)} :=
by
  sorry

end A_inter_B_domain_l99_9983


namespace lcm_18_60_is_180_l99_9912

theorem lcm_18_60_is_180 : Nat.lcm 18 60 = 180 := 
  sorry

end lcm_18_60_is_180_l99_9912


namespace total_commission_l99_9959

-- Define the commission rate
def commission_rate : ℝ := 0.02

-- Define the sale prices of the three houses
def sale_price1 : ℝ := 157000
def sale_price2 : ℝ := 499000
def sale_price3 : ℝ := 125000

-- Total commission calculation
theorem total_commission :
  (commission_rate * sale_price1 + commission_rate * sale_price2 + commission_rate * sale_price3) = 15620 := 
by
  sorry

end total_commission_l99_9959


namespace distance_from_tee_to_hole_l99_9968

-- Define the constants based on the problem conditions
def s1 : ℕ := 180
def s2 : ℕ := (1 / 2 * s1 + 20 - 20)

-- Define the total distance calculation
def total_distance := s1 + s2

-- State the ultimate theorem that needs to be proved
theorem distance_from_tee_to_hole : total_distance = 270 := by
  sorry

end distance_from_tee_to_hole_l99_9968


namespace simplify_expression_l99_9902

theorem simplify_expression (x : ℝ) :
  ( ( ((x + 1) ^ 3 * (x ^ 2 - x + 1) ^ 3) / (x ^ 3 + 1) ^ 3 ) ^ 2 *
    ( ((x - 1) ^ 3 * (x ^ 2 + x + 1) ^ 3) / (x ^ 3 - 1) ^ 3 ) ^ 2 ) = 1 :=
by
  sorry

end simplify_expression_l99_9902


namespace incorrect_statement_b_l99_9960

-- Defining the equation of the circle
def is_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 25

-- Defining the point not on the circle
def is_not_on_circle (x y : ℝ) : Prop :=
  x^2 + y^2 ≠ 25

-- The proposition to be proved
theorem incorrect_statement_b : ¬ ∀ p : ℝ × ℝ, is_not_on_circle p.1 p.2 → ¬ is_on_circle p.1 p.2 :=
by
  -- Here we should provide the proof, but this is not required based on the instructions.
  sorry

end incorrect_statement_b_l99_9960


namespace samantha_lost_pieces_l99_9954

theorem samantha_lost_pieces (total_pieces_on_board : ℕ) (arianna_lost : ℕ) (initial_pieces_per_player : ℕ) :
  total_pieces_on_board = 20 →
  arianna_lost = 3 →
  initial_pieces_per_player = 16 →
  (initial_pieces_per_player - (total_pieces_on_board - (initial_pieces_per_player - arianna_lost))) = 9 :=
by
  intros h1 h2 h3
  sorry

end samantha_lost_pieces_l99_9954


namespace points_of_third_l99_9992

noncomputable def points_of_first : ℕ := 11
noncomputable def points_of_second : ℕ := 7
noncomputable def points_of_fourth : ℕ := 2
noncomputable def johns_total_points : ℕ := 38500

theorem points_of_third :
  ∃ x : ℕ, (points_of_first * points_of_second * x * points_of_fourth ∣ johns_total_points) ∧
    (johns_total_points / (points_of_first * points_of_second * points_of_fourth)) = x := 
sorry

end points_of_third_l99_9992


namespace count_sum_or_diff_squares_at_least_1500_l99_9945

theorem count_sum_or_diff_squares_at_least_1500 : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 2000 ∧ (∃ (x y : ℕ), n = x^2 + y^2 ∨ n = x^2 - y^2)) → 
  1500 ≤ 2000 :=
by
  sorry

end count_sum_or_diff_squares_at_least_1500_l99_9945


namespace same_asymptotes_hyperbolas_l99_9969

theorem same_asymptotes_hyperbolas (M : ℝ) :
  (∀ x y : ℝ, ((x^2 / 9) - (y^2 / 16) = 1) ↔ ((y^2 / 32) - (x^2 / M) = 1)) →
  M = 18 :=
by
  sorry

end same_asymptotes_hyperbolas_l99_9969


namespace compute_expression_l99_9915

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 :=
by sorry

end compute_expression_l99_9915


namespace notebook_cost_3_dollars_l99_9996

def cost_of_notebook (total_spent backpack_cost pen_cost pencil_cost num_notebooks : ℕ) : ℕ := 
  (total_spent - (backpack_cost + pen_cost + pencil_cost)) / num_notebooks

theorem notebook_cost_3_dollars 
  (total_spent : ℕ := 32) 
  (backpack_cost : ℕ := 15) 
  (pen_cost : ℕ := 1) 
  (pencil_cost : ℕ := 1) 
  (num_notebooks : ℕ := 5) 
  : cost_of_notebook total_spent backpack_cost pen_cost pencil_cost num_notebooks = 3 :=
by
  sorry

end notebook_cost_3_dollars_l99_9996


namespace range_of_a_l99_9903

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x → x^2 + 2 * a * x + 1 ≥ 0) ↔ a ≥ -1 := 
by
  sorry

end range_of_a_l99_9903


namespace student_B_most_stable_l99_9928

variable (S_A S_B S_C : ℝ)
variables (hA : S_A^2 = 2.6) (hB : S_B^2 = 1.7) (hC : S_C^2 = 3.5)

/-- Student B has the most stable performance among students A, B, and C based on their variances.
    Given the conditions:
    - S_A^2 = 2.6
    - S_B^2 = 1.7
    - S_C^2 = 3.5
    we prove that student B has the most stable performance.
-/
theorem student_B_most_stable : S_B^2 < S_A^2 ∧ S_B^2 < S_C^2 :=
by
  -- Proof goes here
  sorry

end student_B_most_stable_l99_9928


namespace flight_time_sum_l99_9926

theorem flight_time_sum (h m : ℕ)
  (Hdep : true)   -- Placeholder condition for the departure time being 3:45 PM
  (Hlay : 25 = 25)   -- Placeholder condition for the layover being 25 minutes
  (Harr : true)   -- Placeholder condition for the arrival time being 8:02 PM
  (HsameTZ : true)   -- Placeholder condition for the same time zone
  (H0m : 0 < m) 
  (Hm60 : m < 60)
  (Hfinal_time : (h, m) = (3, 52)) : 
  h + m = 55 := 
by {
  sorry
}

end flight_time_sum_l99_9926


namespace lattice_points_on_sphere_at_distance_5_with_x_1_l99_9937

theorem lattice_points_on_sphere_at_distance_5_with_x_1 :
  let points := [(1, 0, 4), (1, 0, -4), (1, 4, 0), (1, -4, 0),
                 (1, 2, 4), (1, 2, -4), (1, -2, 4), (1, -2, -4),
                 (1, 4, 2), (1, 4, -2), (1, -4, 2), (1, -4, -2),
                 (1, 2, 2), (1, 2, -2), (1, -2, 2), (1, -2, -2)]
  (hs : ∀ y z, (1, y, z) ∈ points → 1^2 + y^2 + z^2 = 25) →
  24 = points.length :=
sorry

end lattice_points_on_sphere_at_distance_5_with_x_1_l99_9937


namespace geometric_sequence_nec_not_suff_l99_9963

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n ≠ 0 → (a (n + 1) / a n) = (a (n + 2) / a (n + 1))

noncomputable def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 3) = a (n + 1) * a (n + 2)

theorem geometric_sequence_nec_not_suff (a : ℕ → ℝ) (hn : ∀ n : ℕ, a n ≠ 0) : 
  (is_geometric_sequence a → satisfies_condition a) ∧ ¬(satisfies_condition a → is_geometric_sequence a) :=
by
  sorry

end geometric_sequence_nec_not_suff_l99_9963


namespace divisor_of_635_l99_9944

theorem divisor_of_635 (p : ℕ) (h1 : Nat.Prime p) (k : ℕ) (h2 : 635 = 7 * k * p + 11) : p = 89 :=
sorry

end divisor_of_635_l99_9944


namespace arithmetic_sequence_is_a_l99_9913

theorem arithmetic_sequence_is_a
  (a : ℚ) (d : ℚ)
  (h1 : 140 + d = a)
  (h2 : a + d = 45 / 28)
  (h3 : a > 0) :
  a = 3965 / 56 :=
by
  sorry

end arithmetic_sequence_is_a_l99_9913


namespace shifted_linear_function_correct_l99_9934

def original_function (x : ℝ) : ℝ := 5 * x - 8
def shifted_function (x : ℝ) : ℝ := original_function x + 4

theorem shifted_linear_function_correct (x : ℝ) :
  shifted_function x = 5 * x - 4 :=
by
  sorry

end shifted_linear_function_correct_l99_9934


namespace jaime_can_buy_five_apples_l99_9909

theorem jaime_can_buy_five_apples :
  ∀ (L M : ℝ),
  (L = M / 2 + 1 / 2) →
  (M / 3 = L / 4 + 1 / 2) →
  (15 / M = 5) :=
by
  intros L M h1 h2
  sorry

end jaime_can_buy_five_apples_l99_9909


namespace candy_bar_reduction_l99_9989

variable (W P x : ℝ)
noncomputable def percent_reduction := (x / W) * 100

theorem candy_bar_reduction (h_weight_reduced : W > 0) 
                            (h_price_same : P > 0) 
                            (h_price_increase : P / (W - x) = (5 / 3) * (P / W)) :
    percent_reduction W x = 40 := 
sorry

end candy_bar_reduction_l99_9989


namespace input_statement_is_INPUT_l99_9953

namespace ProgrammingStatements

-- Definitions of each type of statement
def PRINT_is_output : Prop := True
def INPUT_is_input : Prop := True
def THEN_is_conditional : Prop := True
def END_is_termination : Prop := True

-- The proof problem
theorem input_statement_is_INPUT :
  INPUT_is_input := by
  sorry

end ProgrammingStatements

end input_statement_is_INPUT_l99_9953


namespace contrapositive_proposition_l99_9924

theorem contrapositive_proposition
  (a b c d : ℝ) 
  (h : a + c ≠ b + d) : a ≠ b ∨ c ≠ d :=
sorry

end contrapositive_proposition_l99_9924


namespace sum_of_geometric_ratios_l99_9905

theorem sum_of_geometric_ratios (k p r : ℝ) (a2 a3 b2 b3 : ℝ)
  (hk : k ≠ 0) (hp : p ≠ r)
  (ha2 : a2 = k * p) (ha3 : a3 = k * p * p)
  (hb2 : b2 = k * r) (hb3 : b3 = k * r * r)
  (h : a3 - b3 = 3 * (a2 - b2)) :
  p + r = 3 :=
by sorry

end sum_of_geometric_ratios_l99_9905


namespace cannot_form_right_triangle_l99_9967

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

theorem cannot_form_right_triangle : ¬ is_right_triangle 40 50 60 := 
by
  sorry

end cannot_form_right_triangle_l99_9967


namespace find_x_l99_9906

theorem find_x (x : ℤ) (h : (1 + 2 + 4 + 5 + 6 + 9 + 9 + 10 + 12 + x) / 10 = 7) : x = 12 :=
by
  sorry

end find_x_l99_9906


namespace minimum_value_of_16b_over_ac_l99_9942

noncomputable def minimum_16b_over_ac (a b c : ℝ) (A B C : ℝ) : ℝ :=
  if (0 < B) ∧ (B < Real.pi / 2) ∧
     (Real.cos B ^ 2 + (1 / 2) * Real.sin (2 * B) = 1) ∧
     ((Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) = 3)) then
    16 * b / (a * c)
  else 0

theorem minimum_value_of_16b_over_ac (a b c : ℝ) (A B C : ℝ)
  (h1 : 0 < B)
  (h2 : B < Real.pi / 2)
  (h3 : Real.cos B ^ 2 + (1 / 2) * Real.sin (2 * B) = 1)
  (h4 : Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) = 3) :
  minimum_16b_over_ac a b c A B C = 16 * (2 - Real.sqrt 2) / 3 := 
sorry

end minimum_value_of_16b_over_ac_l99_9942


namespace arithmetic_geometric_sequence_S6_l99_9977

variables (S : ℕ → ℕ)

-- Definitions of conditions from a)
def S2 := S 2 = 3
def S4 := S 4 = 15

-- Main proof statement
theorem arithmetic_geometric_sequence_S6 (S : ℕ → ℕ) (h1 : S 2 = 3) (h2 : S 4 = 15) :
  S 6 = 63 :=
sorry

end arithmetic_geometric_sequence_S6_l99_9977


namespace mod_sum_correct_l99_9900

theorem mod_sum_correct (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
    (h1 : a * b * c ≡ 1 [MOD 7])
    (h2 : 5 * c ≡ 2 [MOD 7])
    (h3 : 6 * b ≡ 3 + b [MOD 7]) :
    (a + b + c) % 7 = 4 := sorry

end mod_sum_correct_l99_9900


namespace exponent_problem_l99_9947

theorem exponent_problem : (5 ^ 6 * 5 ^ 9 * 5) / 5 ^ 3 = 5 ^ 13 := 
by
  sorry

end exponent_problem_l99_9947


namespace smallest_m_n_sum_l99_9930

noncomputable def f (m n : ℕ) (x : ℝ) : ℝ := Real.arcsin (Real.log (n * x) / Real.log m)

theorem smallest_m_n_sum 
  (m n : ℕ) 
  (h_m1 : 1 < m) 
  (h_mn_closure : ∀ x, -1 ≤ Real.log (n * x) / Real.log m ∧ Real.log (n * x) / Real.log m ≤ 1) 
  (h_length : (m ^ 2 - 1) / (m * n) = 1 / 2021) : 
  m + n = 86259 := by
sorry

end smallest_m_n_sum_l99_9930


namespace deposit_percentage_is_10_l99_9958

-- Define the deposit and remaining amount
def deposit := 120
def remaining := 1080

-- Define total cost
def total_cost := deposit + remaining

-- Define deposit percentage calculation
def deposit_percentage := (deposit / total_cost) * 100

-- Theorem to prove the deposit percentage is 10%
theorem deposit_percentage_is_10 : deposit_percentage = 10 := by
  -- Since deposit, remaining and total_cost are defined explicitly,
  -- the proof verification of final result is straightforward.
  sorry

end deposit_percentage_is_10_l99_9958


namespace professor_oscar_review_questions_l99_9931

-- Define the problem conditions.
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10
def number_of_classes : ℕ := 5

-- Define the number of questions that must be reviewed.
def total_questions_to_review : ℕ := 1750

-- The theorem to be proved.
theorem professor_oscar_review_questions :
  students_per_class * questions_per_exam * number_of_classes = total_questions_to_review :=
by
  -- Here we write 'sorry' since we are not providing the full proof.
  sorry

end professor_oscar_review_questions_l99_9931


namespace quadratic_y1_gt_y2_l99_9999

theorem quadratic_y1_gt_y2 {a b c y1 y2 : ℝ} (ha : a > 0) (hy1 : y1 = a * (-1)^2 + b * (-1) + c) (hy2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
  sorry

end quadratic_y1_gt_y2_l99_9999


namespace solution_to_fractional_equation_l99_9938

theorem solution_to_fractional_equation (x : ℝ) (h₁ : 2 / (x - 3) = 1 / x) (h₂ : x ≠ 3) (h₃ : x ≠ 0) : x = -3 :=
sorry

end solution_to_fractional_equation_l99_9938


namespace product_of_remaining_numbers_is_12_l99_9925

noncomputable def final_numbers_product : ℕ := 
  12

theorem product_of_remaining_numbers_is_12 :
  ∀ (initial_ones initial_twos initial_threes initial_fours : ℕ)
  (erase_add_op : Π (a b c : ℕ), Prop),
  initial_ones = 11 ∧ initial_twos = 22 ∧ initial_threes = 33 ∧ initial_fours = 44 ∧
  (∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c → erase_add_op a b c) →
  (∃ (final1 final2 final3 : ℕ), erase_add_op 11 22 33 → final1 * final2 * final3 = final_numbers_product) :=
sorry

end product_of_remaining_numbers_is_12_l99_9925


namespace average_halfway_l99_9978

theorem average_halfway (a b : ℚ) (h_a : a = 1/8) (h_b : b = 1/3) : (a + b) / 2 = 11 / 48 := by
  sorry

end average_halfway_l99_9978


namespace visitors_not_ill_l99_9918

theorem visitors_not_ill (total_visitors : ℕ) (percent_ill : ℕ) (H1 : total_visitors = 500) (H2 : percent_ill = 40) : 
  total_visitors * (100 - percent_ill) / 100 = 300 := 
by 
  sorry

end visitors_not_ill_l99_9918


namespace find_f_expression_l99_9939

theorem find_f_expression (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) : 
  ∀ x, f x = x^2 + 2 * x + 1 :=
by
  sorry

end find_f_expression_l99_9939


namespace quadratic_expression_positive_l99_9911

theorem quadratic_expression_positive
  (a b c : ℝ) (x : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 > 0 :=
sorry

end quadratic_expression_positive_l99_9911


namespace find_z_value_l99_9950

variables {BD FC GC FE : Prop}
variables {a b c d e f g z : ℝ}

-- Assume all given conditions
axiom BD_is_straight : BD
axiom FC_is_straight : FC
axiom GC_is_straight : GC
axiom FE_is_straight : FE
axiom sum_is_z : z = a + b + c + d + e + f + g

-- Goal to prove
theorem find_z_value : z = 540 :=
by
  sorry

end find_z_value_l99_9950


namespace train_length_eq_l99_9914

theorem train_length_eq (L : ℝ) (time_tree time_platform length_platform : ℝ)
  (h_tree : time_tree = 60) (h_platform : time_platform = 105) (h_length_platform : length_platform = 450)
  (h_speed_eq : L / time_tree = (L + length_platform) / time_platform) :
  L = 600 :=
by
  sorry

end train_length_eq_l99_9914


namespace arithmetic_sequence_tenth_term_l99_9923

theorem arithmetic_sequence_tenth_term (a d : ℤ) (h₁ : a + 3 * d = 23) (h₂ : a + 8 * d = 38) : a + 9 * d = 41 := by
  sorry

end arithmetic_sequence_tenth_term_l99_9923


namespace custom_op_eval_l99_9966

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b - a^2 * b

theorem custom_op_eval :
  custom_op 3 4 = -4 :=
by
  sorry

end custom_op_eval_l99_9966


namespace hyperbola_asymptote_l99_9981

theorem hyperbola_asymptote (m : ℝ) : 
  (∀ x y : ℝ, (y^2 / 16 - x^2 / 9 = 1) ↔ (y = m * x ∨ y = -m * x)) → 
  (m = 4 / 3) :=
by
  sorry

end hyperbola_asymptote_l99_9981


namespace tan_theta_solution_l99_9986

theorem tan_theta_solution (θ : ℝ)
  (h : 2 * Real.sin (θ + Real.pi / 3) = 3 * Real.sin (Real.pi / 3 - θ)) :
  Real.tan θ = Real.sqrt 3 / 5 := sorry

end tan_theta_solution_l99_9986


namespace distance_against_current_l99_9980

theorem distance_against_current (V_b V_c : ℝ) (h1 : V_b + V_c = 2) (h2 : V_b = 1.5) : 
  (V_b - V_c) * 3 = 3 := by
  sorry

end distance_against_current_l99_9980


namespace car_distance_after_y_begins_l99_9998

theorem car_distance_after_y_begins (v_x v_y : ℝ) (t_y_start t_x_after_y : ℝ) (d_x_before_y : ℝ) :
  v_x = 35 → v_y = 50 → t_y_start = 1.2 → d_x_before_y = v_x * t_y_start → t_x_after_y = 2.8 →
  (d_x_before_y + v_x * t_x_after_y = 98) :=
by
  intros h_vx h_vy h_ty_start h_dxbefore h_txafter
  simp [h_vx, h_vy, h_ty_start, h_dxbefore, h_txafter]
  sorry

end car_distance_after_y_begins_l99_9998


namespace time_difference_l99_9951

theorem time_difference (speed_Xanthia speed_Molly book_pages : ℕ) (minutes_in_hour : ℕ) :
  speed_Xanthia = 120 ∧ speed_Molly = 40 ∧ book_pages = 360 ∧ minutes_in_hour = 60 →
  (book_pages / speed_Molly - book_pages / speed_Xanthia) * minutes_in_hour = 360 := by
  sorry

end time_difference_l99_9951


namespace proof1_proof2_l99_9993

open Real

noncomputable def problem1 (a b c : ℝ) (A : ℝ) (S : ℝ) :=
  ∃ (a b : ℝ), A = π / 3 ∧ c = 2 ∧ S = sqrt 3 / 2 ∧ S = 1/2 * b * 2 * sin (π / 3) ∧
  a^2 = b^2 + c^2 - 2 * b * c * cos (π / 3) ∧ b = 1 ∧ a = sqrt 3

noncomputable def problem2 (a b c : ℝ) (A B : ℝ) :=
  c = a * cos B ∧ (a + b + c) * (a + b - c) = (2 + sqrt 2) * a * b ∧ 
  B = π / 4 ∧ A = π / 2 → 
  ∃ C, C = π / 4 ∧ C = B

theorem proof1 : problem1 (sqrt 3) 1 2 (π / 3) (sqrt 3 / 2) :=
by
  sorry

theorem proof2 : problem2 (sqrt 3) 1 2 (π / 2) (π / 4) :=
by
  sorry

end proof1_proof2_l99_9993


namespace expected_value_of_10_sided_die_l99_9974

-- Definition of the conditions
def num_faces : ℕ := 10
def face_values : List ℕ := List.range' 2 num_faces

-- Theorem statement: The expected value of a roll of this die is 6.5
theorem expected_value_of_10_sided_die : 
  (List.sum face_values : ℚ) / num_faces = 6.5 := 
sorry

end expected_value_of_10_sided_die_l99_9974


namespace right_triangle_hypotenuse_consecutive_even_l99_9979

theorem right_triangle_hypotenuse_consecutive_even (x : ℕ) (h : x ≠ 0) :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ ((a, b, c) = (x - 2, x, x + 2) ∨ (a, b, c) = (x, x - 2, x + 2) ∨ (a, b, c) = (x + 2, x, x - 2)) ∧ c = 10 := 
by
  sorry

end right_triangle_hypotenuse_consecutive_even_l99_9979


namespace graph_is_finite_distinct_points_l99_9917

def cost (n : ℕ) : ℕ := 18 * n + 3

theorem graph_is_finite_distinct_points : 
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 20 → 
  ∀ (m : ℕ), 1 ≤ m ∧ m ≤ 20 → 
  (cost n = cost m → n = m) ∧
  ∀ x : ℕ, ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ cost n = x :=
by
  sorry

end graph_is_finite_distinct_points_l99_9917


namespace solution_l99_9995

def p (x : ℝ) : Prop := x^2 + 2 * x - 3 < 0
def q (x : ℝ) : Prop := x ∈ Set.univ

theorem solution (x : ℝ) (hx : p x ∧ q x) : x = -2 ∨ x = -1 ∨ x = 0 := 
by
  sorry

end solution_l99_9995


namespace marching_band_max_l99_9920

-- Define the conditions
variables (m k n : ℕ)

-- Lean statement of the problem
theorem marching_band_max (H1 : m = k^2 + 9) (H2 : m = n * (n + 5)) : m = 234 :=
sorry

end marching_band_max_l99_9920


namespace area_of_BEIH_l99_9919

structure Point where
  x : ℚ
  y : ℚ

def B : Point := ⟨0, 0⟩
def A : Point := ⟨0, 2⟩
def D : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩
def E : Point := ⟨0, 1⟩
def F : Point := ⟨1, 0⟩
def I : Point := ⟨2/5, 6/5⟩
def H : Point := ⟨2/3, 2/3⟩

def quadrilateral_area (p1 p2 p3 p4 : Point) : ℚ :=
  (1/2 : ℚ) * 
  ((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) - 
   (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x))

theorem area_of_BEIH : quadrilateral_area B E I H = 7 / 15 := sorry

end area_of_BEIH_l99_9919


namespace angle_in_third_quadrant_l99_9910

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2014) : 180 < θ % 360 ∧ θ % 360 < 270 :=
by
  sorry

end angle_in_third_quadrant_l99_9910


namespace swimming_lane_length_l99_9949

theorem swimming_lane_length (round_trips : ℕ) (total_distance : ℕ) (lane_length : ℕ) 
  (h1 : round_trips = 4) (h2 : total_distance = 800) 
  (h3 : total_distance = lane_length * (round_trips * 2)) : 
  lane_length = 100 := 
by
  sorry

end swimming_lane_length_l99_9949


namespace prove_county_growth_condition_l99_9982

variable (x : ℝ)
variable (investment2014 : ℝ) (investment2016 : ℝ)

def county_growth_condition
  (h1 : investment2014 = 2500)
  (h2 : investment2016 = 3500) : Prop :=
  investment2014 * (1 + x)^2 = investment2016

theorem prove_county_growth_condition
  (x : ℝ)
  (h1 : investment2014 = 2500)
  (h2 : investment2016 = 3500) : county_growth_condition x investment2014 investment2016 h1 h2 :=
by
  sorry

end prove_county_growth_condition_l99_9982


namespace butterfly_development_time_l99_9916

theorem butterfly_development_time :
  ∀ (larva_time cocoon_time : ℕ), 
  (larva_time = 3 * cocoon_time) → 
  (cocoon_time = 30) → 
  (larva_time + cocoon_time = 120) :=
by 
  intros larva_time cocoon_time h1 h2
  sorry

end butterfly_development_time_l99_9916


namespace remainder_of_sum_mod_18_l99_9940

theorem remainder_of_sum_mod_18 :
  let nums := [85, 86, 87, 88, 89, 90, 91, 92, 93]
  let sum_nums := nums.sum
  let product := 90 * sum_nums
  product % 18 = 10 :=
by
  sorry

end remainder_of_sum_mod_18_l99_9940


namespace general_formula_sum_first_n_terms_l99_9907

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

axiom a_initial : a 1 = 1
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n * (1 + 1 / n)

theorem general_formula : ∀ n : ℕ, n > 0 → a n = n * 3^(n - 1) :=
by
  sorry

theorem sum_first_n_terms : ∀ n : ℕ, S n = (2 * n - 1) * 3^n + 1 / 4 :=
by
  sorry

end general_formula_sum_first_n_terms_l99_9907


namespace probability_of_matching_pair_l99_9984

/-!
# Probability of Selecting a Matching Pair of Shoes

Given:
- 12 pairs of sneakers, each with a 4% probability of being chosen.
- 15 pairs of boots, each with a 3% probability of being chosen.
- 18 pairs of dress shoes, each with a 2% probability of being chosen.

If two shoes are selected from the warehouse without replacement, prove that the probability 
of selecting a matching pair of shoes is 52.26%.
-/

namespace ShoeWarehouse

def prob_sneakers_first : ℝ := 0.48
def prob_sneakers_second : ℝ := 0.44
def prob_boots_first : ℝ := 0.45
def prob_boots_second : ℝ := 0.42
def prob_dress_first : ℝ := 0.36
def prob_dress_second : ℝ := 0.34

theorem probability_of_matching_pair :
  (prob_sneakers_first * prob_sneakers_second) +
  (prob_boots_first * prob_boots_second) +
  (prob_dress_first * prob_dress_second) = 0.5226 :=
sorry

end ShoeWarehouse

end probability_of_matching_pair_l99_9984


namespace distance_between_foci_correct_l99_9922

/-- Define the given conditions for the ellipse -/
def ellipse_center : ℝ × ℝ := (3, -2)
def semi_major_axis : ℝ := 7
def semi_minor_axis : ℝ := 3

/-- Define the distance between the foci of the ellipse -/
noncomputable def distance_between_foci : ℝ :=
  2 * Real.sqrt (semi_major_axis ^ 2 - semi_minor_axis ^ 2)

theorem distance_between_foci_correct :
  distance_between_foci = 4 * Real.sqrt 10 := by
  sorry

end distance_between_foci_correct_l99_9922


namespace Danny_finishes_first_l99_9955

-- Definitions based on the conditions
variables (E D F : ℝ)    -- Garden areas for Emily, Danny, Fiona
variables (e d f : ℝ)    -- Mowing rates for Emily, Danny, Fiona
variables (start_time : ℝ)

-- Condition definitions
def emily_garden_size := E = 3 * D
def emily_garden_size_fiona := E = 5 * F
def fiona_mower_speed_danny := f = (1/4) * d
def fiona_mower_speed_emily := f = (1/5) * e

-- Prove Danny finishes first
theorem Danny_finishes_first 
  (h1 : emily_garden_size E D)
  (h2 : emily_garden_size_fiona E F)
  (h3 : fiona_mower_speed_danny f d)
  (h4 : fiona_mower_speed_emily f e) : 
  (start_time ≤ (5/12) * (start_time + E/d) ∧ start_time ≤ (E/f)) -> (start_time + E/d < start_time + E/e) -> 
  true := 
sorry -- proof is omitted

end Danny_finishes_first_l99_9955


namespace initial_weasels_count_l99_9962

theorem initial_weasels_count (initial_rabbits : ℕ) (foxes : ℕ) (weasels_per_fox : ℕ) (rabbits_per_fox : ℕ) 
                              (weeks : ℕ) (remaining_rabbits_weasels : ℕ) (initial_weasels : ℕ) 
                              (total_rabbits_weasels : ℕ) : 
    initial_rabbits = 50 → foxes = 3 → weasels_per_fox = 4 → rabbits_per_fox = 2 → weeks = 3 → 
    remaining_rabbits_weasels = 96 → total_rabbits_weasels = initial_rabbits + initial_weasels → initial_weasels = 100 :=
by
  sorry

end initial_weasels_count_l99_9962


namespace arun_weight_lower_limit_l99_9946

variable {W B : ℝ}

theorem arun_weight_lower_limit
  (h1 : 64 < W ∧ W < 72)
  (h2 : B < W ∧ W < 70)
  (h3 : W ≤ 67)
  (h4 : (64 + 67) / 2 = 66) :
  64 < B :=
by sorry

end arun_weight_lower_limit_l99_9946


namespace total_marks_math_physics_l99_9932

variables (M P C : ℕ)
axiom condition1 : C = P + 20
axiom condition2 : (M + C) / 2 = 45

theorem total_marks_math_physics : M + P = 70 :=
by sorry

end total_marks_math_physics_l99_9932


namespace moses_income_l99_9908

theorem moses_income (investment : ℝ) (percentage : ℝ) (dividend_rate : ℝ) (income : ℝ)
  (h1 : investment = 3000) (h2 : percentage = 0.72) (h3 : dividend_rate = 0.0504) :
  income = 210 :=
sorry

end moses_income_l99_9908


namespace unripe_oranges_zero_l99_9935

def oranges_per_day (harvest_duration : ℕ) (ripe_oranges_per_day : ℕ) : ℕ :=
  harvest_duration * ripe_oranges_per_day

theorem unripe_oranges_zero
  (harvest_duration : ℕ)
  (ripe_oranges_per_day : ℕ)
  (total_ripe_oranges : ℕ)
  (h1 : harvest_duration = 25)
  (h2 : ripe_oranges_per_day = 82)
  (h3 : total_ripe_oranges = 2050)
  (h4 : oranges_per_day harvest_duration ripe_oranges_per_day = total_ripe_oranges) :
  ∀ unripe_oranges_per_day, unripe_oranges_per_day = 0 :=
by
  sorry

end unripe_oranges_zero_l99_9935


namespace solution_set_of_inequality_l99_9927

theorem solution_set_of_inequality :
  { x : ℝ | x^2 - 5 * x + 6 ≤ 0 } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l99_9927


namespace extra_people_got_on_the_train_l99_9965

-- Definitions corresponding to the conditions
def initial_people_on_train : ℕ := 78
def people_got_off : ℕ := 27
def current_people_on_train : ℕ := 63

-- The mathematical equivalent proof problem
theorem extra_people_got_on_the_train :
  (initial_people_on_train - people_got_off + extra_people = current_people_on_train) → (extra_people = 12) :=
by
  sorry

end extra_people_got_on_the_train_l99_9965


namespace train_cross_time_l99_9990

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := 20
noncomputable def platform_length : ℝ := 320
noncomputable def time_cross_platform : ℝ := 34
noncomputable def train_length : ℝ := 360

theorem train_cross_time (v_kmph : ℝ) (v_mps : ℝ) (p_len : ℝ) (t_cross : ℝ) (t_len : ℝ) :
  v_kmph = 72 ∧ v_mps = 20 ∧ p_len = 320 ∧ t_cross = 34 ∧ t_len = 360 →
  (t_len / v_mps) = 18 :=
by
  intros
  sorry

end train_cross_time_l99_9990


namespace initial_marbles_count_l99_9976

-- Leo's initial conditions and quantities
def initial_packs := 40
def marbles_per_pack := 10
def given_Manny (P: ℕ) := P / 4
def given_Neil (P: ℕ) := P / 8
def kept_by_Leo := 25

-- The equivalent proof problem stated in Lean
theorem initial_marbles_count (P: ℕ) (Manny_packs: ℕ) (Neil_packs: ℕ) (kept_packs: ℕ) :
  Manny_packs = given_Manny P → Neil_packs = given_Neil P → kept_packs = kept_by_Leo → P = initial_packs → P * marbles_per_pack = 400 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_marbles_count_l99_9976


namespace find_n_l99_9988

variable (a r : ℚ) (n : ℕ)

def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Given conditions
axiom seq_first_term : a = 1 / 3
axiom seq_common_ratio : r = 1 / 3
axiom sum_of_first_n_terms_eq : geom_sum a r n = 80 / 243

-- Prove that n = 5
theorem find_n : n = 5 := by
  sorry

end find_n_l99_9988


namespace gravitational_force_at_384000km_l99_9952

theorem gravitational_force_at_384000km
  (d1 d2 : ℝ)
  (f1 f2 : ℝ)
  (k : ℝ)
  (h1 : d1 = 6400)
  (h2 : d2 = 384000)
  (h3 : f1 = 800)
  (h4 : f1 * d1^2 = k)
  (h5 : f2 * d2^2 = k) :
  f2 = 2 / 9 :=
by
  sorry

end gravitational_force_at_384000km_l99_9952


namespace find_g5_l99_9964

-- Define the function g on integers
def g : ℤ → ℤ := sorry

-- Define the conditions
axiom cond1 : g 1 > 1
axiom cond2 : ∀ (x y : ℤ), g (x + y) + x * g y + y * g x = g x * g y + x + y + x * y
axiom cond3 : ∀ (x : ℤ), 3 * g x = g (x + 1) + 2 * x - 1

-- The statement we need to prove
theorem find_g5 : g 5 = 248 := by
  sorry

end find_g5_l99_9964


namespace total_ages_l99_9970

def Kate_age : ℕ := 19
def Maggie_age : ℕ := 17
def Sue_age : ℕ := 12

theorem total_ages : Kate_age + Maggie_age + Sue_age = 48 := sorry

end total_ages_l99_9970


namespace calculate_expression_l99_9997

theorem calculate_expression :
  427 / 2.68 * 16 * 26.8 / 42.7 * 16 = 25600 :=
sorry

end calculate_expression_l99_9997


namespace C_share_of_profit_l99_9985

def A_investment : ℕ := 12000
def B_investment : ℕ := 16000
def C_investment : ℕ := 20000
def total_profit : ℕ := 86400

theorem C_share_of_profit: 
  (C_investment / (A_investment + B_investment + C_investment) * total_profit) = 36000 :=
by
  sorry

end C_share_of_profit_l99_9985


namespace triangle_side_length_l99_9994

variables {BC AC : ℝ} {α β γ : ℝ}

theorem triangle_side_length :
  α = 45 ∧ β = 75 ∧ AC = 6 ∧ α + β + γ = 180 →
  BC = 6 * (Real.sqrt 3 - 1) :=
by
  intros h
  sorry

end triangle_side_length_l99_9994


namespace find_general_term_l99_9929

-- Definition of sequence sum condition
def seq_sum_condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (2/3) * a n + 1/3

-- Statement of the proof problem
theorem find_general_term (a S : ℕ → ℝ) 
  (h : seq_sum_condition a S) : 
  ∀ n, a n = (-2)^(n-1) := 
by
  sorry

end find_general_term_l99_9929


namespace geometric_sequence_sum_l99_9933

theorem geometric_sequence_sum (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 1/2)
  (h2 : a 3 + a 4 = 1)
  (h_geom : ∀ n, a n + a (n+1) = (a 1 + a 2) * 2^(n-1)) :
  a 7 + a 8 + a 9 + a 10 = 12 := 
sorry

end geometric_sequence_sum_l99_9933


namespace simplify_fraction_l99_9943

theorem simplify_fraction : (2^5 + 2^3) / (2^4 - 2^2) = 10 / 3 := 
by 
  sorry

end simplify_fraction_l99_9943
