import Mathlib

namespace NUMINAMATH_GPT_spider_legs_total_l2179_217977

def num_spiders : ℕ := 4
def legs_per_spider : ℕ := 8
def total_legs : ℕ := num_spiders * legs_per_spider

theorem spider_legs_total : total_legs = 32 := by
  sorry -- proof is skipped with 'sorry'

end NUMINAMATH_GPT_spider_legs_total_l2179_217977


namespace NUMINAMATH_GPT_inverse_proportional_x_y_l2179_217907

theorem inverse_proportional_x_y (x y k : ℝ) (h_inverse : x * y = k) (h_given : 40 * 5 = k) : x = 20 :=
by 
  sorry

end NUMINAMATH_GPT_inverse_proportional_x_y_l2179_217907


namespace NUMINAMATH_GPT_derivative_of_f_eq_f_deriv_l2179_217919

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.cos a) ^ x - (Real.sin a) ^ x

noncomputable def f_deriv (a x : ℝ) : ℝ :=
  (Real.cos a) ^ x * Real.log (Real.cos a) - (Real.sin a) ^ x * Real.log (Real.sin a)

theorem derivative_of_f_eq_f_deriv (a : ℝ) (h : 0 < a ∧ a < Real.pi / 2) :
  (deriv (f a)) = f_deriv a :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_f_eq_f_deriv_l2179_217919


namespace NUMINAMATH_GPT_inequality_C_l2179_217985

variable (a b : ℝ)
variable (h : a > b)
variable (h' : b > 0)

theorem inequality_C : a + b > 2 * b := by
  sorry

end NUMINAMATH_GPT_inequality_C_l2179_217985


namespace NUMINAMATH_GPT_chord_PQ_eqn_l2179_217965

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9
def midpoint_PQ (M : ℝ × ℝ) : Prop := M = (1, 2)
def line_PQ_eq (x y : ℝ) : Prop := x + 2 * y - 5 = 0

theorem chord_PQ_eqn : 
  (∃ P Q : ℝ × ℝ, circle_eq P.1 P.2 ∧ circle_eq Q.1 Q.2 ∧ midpoint_PQ ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →
  ∃ x y : ℝ, line_PQ_eq x y := 
sorry

end NUMINAMATH_GPT_chord_PQ_eqn_l2179_217965


namespace NUMINAMATH_GPT_triangle_value_l2179_217925

variable (triangle p : ℝ)

theorem triangle_value : (triangle + p = 75 ∧ 3 * (triangle + p) - p = 198) → triangle = 48 :=
by
  sorry

end NUMINAMATH_GPT_triangle_value_l2179_217925


namespace NUMINAMATH_GPT_shaded_area_l2179_217957

theorem shaded_area (R : ℝ) (r : ℝ) (hR : R = 10) (hr : r = R / 2) : 
  π * R^2 - 2 * (π * r^2) = 50 * π :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l2179_217957


namespace NUMINAMATH_GPT_A_visits_all_seats_iff_even_l2179_217989

def move_distance_unique (n : ℕ) : Prop := 
  ∀ k l : ℕ, (1 ≤ k ∧ k < n) → (1 ≤ l ∧ l < n) → k ≠ l → (k ≠ l % n)

def visits_all_seats (n : ℕ) : Prop := 
  ∃ A : ℕ → ℕ, 
  (∀ (k : ℕ), 0 ≤ k ∧ k < n → (0 ≤ A k ∧ A k < n)) ∧ 
  (∀ (k : ℕ), 0 ≤ k ∧ k < n → ∃ (m : ℕ), m ≠ n ∧ A k ≠ (A m % n))

theorem A_visits_all_seats_iff_even (n : ℕ) :
  (move_distance_unique n ∧ visits_all_seats n) ↔ (n % 2 = 0) := 
sorry

end NUMINAMATH_GPT_A_visits_all_seats_iff_even_l2179_217989


namespace NUMINAMATH_GPT_ranking_of_ABC_l2179_217962

-- Define the ranking type
inductive Rank
| first
| second
| third

-- Define types for people
inductive Person
| A
| B
| C

open Rank Person

-- Alias for ranking of each person
def ranking := Person → Rank

-- Define the conditions
def A_statement (r : ranking) : Prop := r A ≠ first
def B_statement (r : ranking) : Prop := A_statement r ≠ false
def C_statement (r : ranking) : Prop := r C ≠ third

def B_lied : Prop := true
def C_told_truth : Prop := true

-- The equivalent problem, asked to prove the final result
theorem ranking_of_ABC (r : ranking) : 
  (B_lied ∧ C_told_truth ∧ B_statement r = false ∧ C_statement r = true) → 
  (r A = first ∧ r B = third ∧ r C = second) :=
sorry

end NUMINAMATH_GPT_ranking_of_ABC_l2179_217962


namespace NUMINAMATH_GPT_solve_for_y_l2179_217916

noncomputable def roots := [(-126 + Real.sqrt 13540) / 8, (-126 - Real.sqrt 13540) / 8]

theorem solve_for_y (y : ℝ) :
  (8*y^2 + 176*y + 2) / (3*y + 74) = 4*y + 2 →
  y = roots[0] ∨ y = roots[1] :=
by
  intros
  sorry

end NUMINAMATH_GPT_solve_for_y_l2179_217916


namespace NUMINAMATH_GPT_driving_time_per_trip_l2179_217939

-- Define the conditions
def filling_time_per_trip : ℕ := 15
def number_of_trips : ℕ := 6
def total_moving_hours : ℕ := 7
def total_moving_time : ℕ := total_moving_hours * 60

-- Define the problem
theorem driving_time_per_trip :
  (total_moving_time - (filling_time_per_trip * number_of_trips)) / number_of_trips = 55 :=
by
  sorry

end NUMINAMATH_GPT_driving_time_per_trip_l2179_217939


namespace NUMINAMATH_GPT_quadractic_inequality_solution_l2179_217937

theorem quadractic_inequality_solution (a b : ℝ) (h₁ : ∀ x : ℝ, -4 ≤ x ∧ x ≤ 3 → x^2 - (a+1) * x + b ≤ 0) : a + b = -14 :=
by 
  -- Proof construction is omitted
  sorry

end NUMINAMATH_GPT_quadractic_inequality_solution_l2179_217937


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_length_l2179_217971

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_length_l2179_217971


namespace NUMINAMATH_GPT_DebateClubOfficerSelection_l2179_217973

-- Definitions based on the conditions
def members : Finset ℕ := Finset.range 25 -- Members are indexed from 0 to 24
def Simon := 0
def Rachel := 1
def John := 2

-- Conditions regarding the officers
def is_officer (x : ℕ) (pres sec tre : ℕ) : Prop := 
  x = pres ∨ x = sec ∨ x = tre

def Simon_condition (pres sec tre : ℕ) : Prop :=
  (is_officer Simon pres sec tre) → (is_officer Rachel pres sec tre)

def Rachel_condition (pres sec tre : ℕ) : Prop :=
  (is_officer Rachel pres sec tre) → (is_officer Simon pres sec tre) ∨ (is_officer John pres sec tre)

-- Statement of the problem in Lean
theorem DebateClubOfficerSelection : ∃ (pres sec tre : ℕ), 
  pres ≠ sec ∧ sec ≠ tre ∧ pres ≠ tre ∧ 
  pres ∈ members ∧ sec ∈ members ∧ tre ∈ members ∧ 
  Simon_condition pres sec tre ∧
  Rachel_condition pres sec tre :=
sorry

end NUMINAMATH_GPT_DebateClubOfficerSelection_l2179_217973


namespace NUMINAMATH_GPT_probability_A_selected_l2179_217949

def n : ℕ := 5
def k : ℕ := 2

def total_ways : ℕ := Nat.choose n k  -- C(n, k)

def favorable_ways : ℕ := Nat.choose (n - 1) (k - 1)  -- C(n-1, k-1)

theorem probability_A_selected : (favorable_ways : ℚ) / (total_ways : ℚ) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_A_selected_l2179_217949


namespace NUMINAMATH_GPT_slope_to_y_intercept_ratio_l2179_217987

theorem slope_to_y_intercept_ratio (m b : ℝ) (c : ℝ) (h1 : m = c * b) (h2 : 2 * m + b = 0) : c = -1 / 2 :=
by sorry

end NUMINAMATH_GPT_slope_to_y_intercept_ratio_l2179_217987


namespace NUMINAMATH_GPT_max_positive_integer_difference_l2179_217910

theorem max_positive_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) : ∃ d : ℕ, d = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_positive_integer_difference_l2179_217910


namespace NUMINAMATH_GPT_describe_graph_l2179_217927

theorem describe_graph :
  ∀ (x y : ℝ), ((x + y) ^ 2 = x ^ 2 + y ^ 2 + 4 * x) ↔ (x = 0 ∨ y = 2) := 
by
  sorry

end NUMINAMATH_GPT_describe_graph_l2179_217927


namespace NUMINAMATH_GPT_riverside_high_badges_l2179_217972

/-- Given the conditions on the sums of consecutive prime badge numbers of the debate team members,
prove that Giselle's badge number is 1014, given that the current year is 2025.
-/
theorem riverside_high_badges (p1 p2 p3 p4 : ℕ) (hp1 : Prime p1) (hp2 : Prime p2) (hp3 : Prime p3) (hp4 : Prime p4)
    (hconsec : p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 = p3 + 6)
    (h1 : ∃ x, p1 + p3 = x) (h2 : ∃ y, p1 + p2 = y) (h3 : ∃ z, p2 + p3 = z ∧ z ≤ 31) 
    (h4 : p3 + p4 = 2025) : p4 = 1014 :=
by sorry

end NUMINAMATH_GPT_riverside_high_badges_l2179_217972


namespace NUMINAMATH_GPT_Brad_pumpkin_weight_l2179_217922

theorem Brad_pumpkin_weight (B : ℝ)
  (h1 : ∃ J : ℝ, J = B / 2)
  (h2 : ∃ Be : ℝ, Be = 4 * (B / 2))
  (h3 : ∃ Be J : ℝ, Be - J = 81) : B = 54 := by
  obtain ⟨J, hJ⟩ := h1
  obtain ⟨Be, hBe⟩ := h2
  obtain ⟨_, hBeJ⟩ := h3
  sorry

end NUMINAMATH_GPT_Brad_pumpkin_weight_l2179_217922


namespace NUMINAMATH_GPT_coffee_blend_l2179_217940

variable (pA pB : ℝ) (cA cB : ℝ) (total_cost : ℝ) 

theorem coffee_blend (hA : pA = 4.60) 
                     (hB : pB = 5.95) 
                     (h_ratio : cB = 2 * cA) 
                     (h_total : 4.60 * cA + 5.95 * cB = 511.50) : 
                     cA = 31 := 
by
  sorry

end NUMINAMATH_GPT_coffee_blend_l2179_217940


namespace NUMINAMATH_GPT_fixed_monthly_fee_l2179_217917

def FebruaryBill (x y : ℝ) : Prop := x + y = 18.72
def MarchBill (x y : ℝ) : Prop := x + 3 * y = 28.08

theorem fixed_monthly_fee (x y : ℝ) (h1 : FebruaryBill x y) (h2 : MarchBill x y) : x = 14.04 :=
by 
  sorry

end NUMINAMATH_GPT_fixed_monthly_fee_l2179_217917


namespace NUMINAMATH_GPT_no_solution_abs_eq_quadratic_l2179_217961

theorem no_solution_abs_eq_quadratic (x : ℝ) : ¬ (|x - 4| = x^2 + 6 * x + 8) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_abs_eq_quadratic_l2179_217961


namespace NUMINAMATH_GPT_largest_common_multiple_3_5_l2179_217904

theorem largest_common_multiple_3_5 (n : ℕ) :
  (n < 10000) ∧ (n ≥ 1000) ∧ (n % 3 = 0) ∧ (n % 5 = 0) → n ≤ 9990 :=
sorry

end NUMINAMATH_GPT_largest_common_multiple_3_5_l2179_217904


namespace NUMINAMATH_GPT_number_of_n_l2179_217984

theorem number_of_n (h1: n > 0) (h2: n ≤ 2000) (h3: ∃ m, 10 * n = m^2) : n = 14 :=
by sorry

end NUMINAMATH_GPT_number_of_n_l2179_217984


namespace NUMINAMATH_GPT_ratio_of_numbers_l2179_217983

theorem ratio_of_numbers (a b : ℕ) (ha : a = 45) (hb : b = 60) (lcm_ab : Nat.lcm a b = 180) : (a : ℚ) / b = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l2179_217983


namespace NUMINAMATH_GPT_hyperbola_aux_lines_l2179_217988

theorem hyperbola_aux_lines (a : ℝ) (h_a_positive : a > 0)
  (h_hyperbola_eqn : ∀ x y, (x^2 / a^2) - (y^2 / 16) = 1)
  (h_asymptotes : ∀ x y, y = 4/3 * x ∨ y = -4/3 * x) : 
  ∀ x, (x = 9/5 ∨ x = -9/5) := sorry

end NUMINAMATH_GPT_hyperbola_aux_lines_l2179_217988


namespace NUMINAMATH_GPT_min_games_to_predict_l2179_217920

theorem min_games_to_predict (W B : ℕ) (total_games : ℕ) (n : ℕ) : 
  W = 15 → B = 20 → total_games = W * B → n = 280 → 
  (∃ x, x ∈ {i | ∃ j, i < W ∧ j < B}) :=
by
  intros hW hB htotal hn
  sorry

end NUMINAMATH_GPT_min_games_to_predict_l2179_217920


namespace NUMINAMATH_GPT_no_valid_solution_l2179_217963

theorem no_valid_solution (x y z : ℤ) (h1 : x = 11 * y + 4) 
  (h2 : 2 * x = 24 * y + 3) (h3 : x + z = 34 * y + 5) : 
  ¬ ∃ (y : ℤ), 13 * y - x + 7 * z = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_valid_solution_l2179_217963


namespace NUMINAMATH_GPT_range_of_alpha_minus_beta_l2179_217914

theorem range_of_alpha_minus_beta (α β : ℝ) (h1 : -180 < α) (h2 : α < β) (h3 : β < 180) :
  -360 < α - β ∧ α - β < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_alpha_minus_beta_l2179_217914


namespace NUMINAMATH_GPT_determine_x_l2179_217901

variable (a b c d x : ℝ)
variable (h1 : (a^2 + x)/(b^2 + x) = c/d)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : d ≠ c) -- added condition from solution step

theorem determine_x : x = (a^2 * d - b^2 * c) / (c - d) := by
  sorry

end NUMINAMATH_GPT_determine_x_l2179_217901


namespace NUMINAMATH_GPT_remainder_3_101_add_5_mod_11_l2179_217955

theorem remainder_3_101_add_5_mod_11 : (3 ^ 101 + 5) % 11 = 8 := 
by sorry

end NUMINAMATH_GPT_remainder_3_101_add_5_mod_11_l2179_217955


namespace NUMINAMATH_GPT_proof_problem_l2179_217956

-- Given conditions
variables {a b : Type}  -- Two non-coincident lines
variables {α β : Type}  -- Two non-coincident planes

-- Definitions of the relationships
def is_parallel_to (x y : Type) : Prop := sorry  -- Parallel relationship
def is_perpendicular_to (x y : Type) : Prop := sorry  -- Perpendicular relationship

-- Statements to verify
def statement1 (a α b : Type) : Prop := 
  (is_parallel_to a α ∧ is_parallel_to b α) → is_parallel_to a b

def statement2 (a α β : Type) : Prop :=
  (is_perpendicular_to a α ∧ is_perpendicular_to a β) → is_parallel_to α β

def statement3 (α β : Type) : Prop :=
  is_perpendicular_to α β → ∃ l : Type, is_perpendicular_to l α ∧ is_parallel_to l β

def statement4 (α β : Type) : Prop :=
  is_perpendicular_to α β → ∃ γ : Type, is_perpendicular_to γ α ∧ is_perpendicular_to γ β

-- Proof problem: verifying which statements are true.
theorem proof_problem :
  ¬ (statement1 a α b) ∧ statement2 a α β ∧ statement3 α β ∧ statement4 α β :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2179_217956


namespace NUMINAMATH_GPT_number_of_cipher_keys_l2179_217945

theorem number_of_cipher_keys (n : ℕ) (h : n % 2 = 0) : 
  ∃ K : ℕ, K = 4^(n^2 / 4) :=
by 
  sorry

end NUMINAMATH_GPT_number_of_cipher_keys_l2179_217945


namespace NUMINAMATH_GPT_trip_correct_graph_l2179_217923

-- Define a structure representing the trip
structure Trip :=
  (initial_city_traffic_duration : ℕ)
  (highway_duration_to_mall : ℕ)
  (shopping_duration : ℕ)
  (highway_duration_from_mall : ℕ)
  (return_city_traffic_duration : ℕ)

-- Define the conditions about the trip
def conditions (t : Trip) : Prop :=
  t.shopping_duration = 1 ∧ -- Shopping for one hour
  t.initial_city_traffic_duration < t.highway_duration_to_mall ∧ -- Travel more rapidly on the highway
  t.return_city_traffic_duration < t.highway_duration_from_mall -- Return more rapidly on the highway

-- Define the graph representation of the trip
inductive Graph
| A | B | C | D | E

-- Define the property that graph B correctly represents the trip
def correct_graph (t : Trip) (g : Graph) : Prop :=
  g = Graph.B

-- The theorem stating that given the conditions, the correct graph is B
theorem trip_correct_graph (t : Trip) (h : conditions t) : correct_graph t Graph.B :=
by
  sorry

end NUMINAMATH_GPT_trip_correct_graph_l2179_217923


namespace NUMINAMATH_GPT_cost_of_paving_l2179_217968

theorem cost_of_paving (L W R : ℝ) (hL : L = 6.5) (hW : W = 2.75) (hR : R = 600) : 
  L * W * R = 10725 := by
  rw [hL, hW, hR]
  -- To solve the theorem successively
  -- we would need to verify the product of the values
  -- given by the conditions.
  sorry

end NUMINAMATH_GPT_cost_of_paving_l2179_217968


namespace NUMINAMATH_GPT_fewest_coach_handshakes_l2179_217921

noncomputable def binom (n : ℕ) := n * (n - 1) / 2

theorem fewest_coach_handshakes : 
  ∃ (k1 k2 k3 : ℕ), binom 43 + k1 + k2 + k3 = 903 ∧ k1 + k2 + k3 = 0 := 
by
  use 0, 0, 0
  sorry

end NUMINAMATH_GPT_fewest_coach_handshakes_l2179_217921


namespace NUMINAMATH_GPT_incorrect_option_D_l2179_217946

-- Definitions based on conditions
def cumulative_progress (days : ℕ) : ℕ :=
  30 * days

-- The Lean statement representing the mathematically equivalent proof problem
theorem incorrect_option_D : cumulative_progress 11 = 330 ∧ ¬ (cumulative_progress 10 = 330) :=
by {
  sorry
}

end NUMINAMATH_GPT_incorrect_option_D_l2179_217946


namespace NUMINAMATH_GPT_angle_A_minimum_a_l2179_217969

variable {α : Type} [LinearOrderedField α]

-- Part 1: Prove A = π / 3 given the specific equation in triangle ABC
theorem angle_A (a b c : α) (cos : α → α)
  (h : b^2 * c * cos c + c^2 * b * cos b = a * b^2 + a * c^2 - a^3) :
  ∃ A : α, A = π / 3 :=
sorry

-- Part 2: Prove the minimum value of a is 1 when b + c = 2
theorem minimum_a (a b c : α) (h : b + c = 2) :
  ∃ a : α, a = 1 :=
sorry

end NUMINAMATH_GPT_angle_A_minimum_a_l2179_217969


namespace NUMINAMATH_GPT_paper_plates_cost_l2179_217966

theorem paper_plates_cost (P C x : ℝ) 
(h1 : 100 * P + 200 * C = 6.00) 
(h2 : x * P + 40 * C = 1.20) : 
x = 20 := 
sorry

end NUMINAMATH_GPT_paper_plates_cost_l2179_217966


namespace NUMINAMATH_GPT_correct_sum_of_satisfying_values_l2179_217933

def g (x : Nat) : Nat :=
  match x with
  | 0 => 0
  | 1 => 2
  | 2 => 1
  | _ => 0  -- This handles the out-of-bounds case, though it's not needed here

def f (x : Nat) : Nat :=
  match x with
  | 0 => 2
  | 1 => 1
  | 2 => 0
  | _ => 0  -- This handles the out-of-bounds case, though it's not needed here

def satisfies_condition (x : Nat) : Bool :=
  f (g x) > g (f x)

def sum_of_satisfying_values : Nat :=
  List.sum (List.filter satisfies_condition [0, 1, 2])

theorem correct_sum_of_satisfying_values : sum_of_satisfying_values = 2 :=
  sorry

end NUMINAMATH_GPT_correct_sum_of_satisfying_values_l2179_217933


namespace NUMINAMATH_GPT_train_length_l2179_217995

theorem train_length :
  let speed_kmph := 63
  let time_seconds := 16
  let speed_mps := (speed_kmph * 1000) / 3600
  let length_meters := speed_mps * time_seconds
  length_meters = 280 := 
by
  sorry

end NUMINAMATH_GPT_train_length_l2179_217995


namespace NUMINAMATH_GPT_circumscribed_circle_radius_l2179_217934

theorem circumscribed_circle_radius (r : ℝ) (π : ℝ)
  (isosceles_right_triangle : Type) 
  (perimeter : isosceles_right_triangle → ℝ )
  (area : ℝ → ℝ)
  (h : ∀ (t : isosceles_right_triangle), perimeter t = area r) :
  r = (1 + Real.sqrt 2) / π :=
sorry

end NUMINAMATH_GPT_circumscribed_circle_radius_l2179_217934


namespace NUMINAMATH_GPT_total_reactions_eq_100_l2179_217953

variable (x : ℕ) -- Total number of reactions.
variable (thumbs_up : ℕ) -- Number of "thumbs up" reactions.
variable (thumbs_down : ℕ) -- Number of "thumbs down" reactions.
variable (S : ℕ) -- Net Score.

-- Conditions
axiom thumbs_up_eq_75percent_reactions : thumbs_up = 3 * x / 4
axiom thumbs_down_eq_25percent_reactions : thumbs_down = x / 4
axiom score_definition : S = thumbs_up - thumbs_down
axiom initial_score : S = 50

theorem total_reactions_eq_100 : x = 100 :=
by 
  sorry

end NUMINAMATH_GPT_total_reactions_eq_100_l2179_217953


namespace NUMINAMATH_GPT_James_total_area_l2179_217931

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase_dimension : ℕ := 2
def number_of_new_rooms : ℕ := 4
def larger_room_multiplier : ℕ := 2

noncomputable def new_length : ℕ := initial_length + increase_dimension
noncomputable def new_width : ℕ := initial_width + increase_dimension
noncomputable def area_of_one_new_room : ℕ := new_length * new_width
noncomputable def total_area_of_4_rooms : ℕ := area_of_one_new_room * number_of_new_rooms
noncomputable def area_of_larger_room : ℕ := area_of_one_new_room * larger_room_multiplier
noncomputable def total_area : ℕ := total_area_of_4_rooms + area_of_larger_room

theorem James_total_area : total_area = 1800 := 
by
  sorry

end NUMINAMATH_GPT_James_total_area_l2179_217931


namespace NUMINAMATH_GPT_algebra_expression_eq_l2179_217994

theorem algebra_expression_eq (x : ℝ) (h : x = Real.sqrt 2 + 1) : x^2 - 2 * x + 2 = 3 := by
  sorry

end NUMINAMATH_GPT_algebra_expression_eq_l2179_217994


namespace NUMINAMATH_GPT_smallest_integer_l2179_217909

-- Given positive integer M such that
def satisfies_conditions (M : ℕ) : Prop :=
  M % 6 = 5 ∧
  M % 7 = 6 ∧
  M % 8 = 7 ∧
  M % 9 = 8 ∧
  M % 10 = 9 ∧
  M % 11 = 10 ∧
  M % 13 = 12

-- The main theorem to prove
theorem smallest_integer (M : ℕ) (h : satisfies_conditions M) : M = 360359 :=
  sorry

end NUMINAMATH_GPT_smallest_integer_l2179_217909


namespace NUMINAMATH_GPT_maximum_notebooks_maria_can_buy_l2179_217997

def price_single : ℕ := 1
def price_pack_4 : ℕ := 3
def price_pack_7 : ℕ := 5
def total_budget : ℕ := 10

def max_notebooks (budget : ℕ) : ℕ :=
  if budget < price_single then 0
  else if budget < price_pack_4 then budget / price_single
  else if budget < price_pack_7 then max (budget / price_single) (4 * (budget / price_pack_4))
  else max (budget / price_single) (7 * (budget / price_pack_7))

theorem maximum_notebooks_maria_can_buy :
  max_notebooks total_budget = 14 := by
  sorry

end NUMINAMATH_GPT_maximum_notebooks_maria_can_buy_l2179_217997


namespace NUMINAMATH_GPT_prob_equal_even_odd_dice_l2179_217954

def even_number_probability : ℚ := 1 / 2
def odd_number_probability : ℚ := 1 / 2

def probability_equal_even_odd (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k) * (even_number_probability) ^ n

theorem prob_equal_even_odd_dice : 
  probability_equal_even_odd 8 4 = 35 / 128 :=
by
  sorry

end NUMINAMATH_GPT_prob_equal_even_odd_dice_l2179_217954


namespace NUMINAMATH_GPT_ab_non_positive_l2179_217950

theorem ab_non_positive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 :=
sorry

end NUMINAMATH_GPT_ab_non_positive_l2179_217950


namespace NUMINAMATH_GPT_sin_2gamma_proof_l2179_217928

-- Assume necessary definitions and conditions
variables {A B C D P : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P]
variables (a b c d: ℝ)
variables (α β γ: ℝ)

-- Assume points A, B, C, D, P lie on a circle in that order and AB = BC = CD
axiom points_on_circle : a = b ∧ b = c ∧ c = d
axiom cos_apc : Real.cos α = 3/5
axiom cos_bpd : Real.cos β = 1/5

noncomputable def sin_2gamma : ℝ :=
  2 * Real.sin γ * Real.cos γ

-- Statement to prove sin(2 * γ) given the conditions
theorem sin_2gamma_proof : sin_2gamma γ = 8 * Real.sqrt 5 / 25 :=
sorry

end NUMINAMATH_GPT_sin_2gamma_proof_l2179_217928


namespace NUMINAMATH_GPT_average_of_original_set_l2179_217929

theorem average_of_original_set (A : ℝ) (h1 : 7 * A = 125 * 7 / 5) : A = 25 := 
sorry

end NUMINAMATH_GPT_average_of_original_set_l2179_217929


namespace NUMINAMATH_GPT_union_of_A_and_B_l2179_217902

open Set

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {3, 5, 7, 8}

theorem union_of_A_and_B : A ∪ B = {3, 4, 5, 6, 7, 8} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2179_217902


namespace NUMINAMATH_GPT_volume_of_cube_l2179_217912

theorem volume_of_cube (SA : ℝ) (h : SA = 486) : ∃ V : ℝ, V = 729 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cube_l2179_217912


namespace NUMINAMATH_GPT_prime_of_form_4k_plus_1_as_sum_of_two_squares_prime_of_form_8k_plus_3_as_sum_of_three_squares_l2179_217998

theorem prime_of_form_4k_plus_1_as_sum_of_two_squares (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hk : p = 4 * k + 1) :
  ∃ a b : ℤ, p = a^2 + b^2 :=
sorry

theorem prime_of_form_8k_plus_3_as_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (k : ℕ) (hk : p = 8 * k + 3) :
  ∃ a b c : ℤ, p = a^2 + b^2 + c^2 :=
sorry

end NUMINAMATH_GPT_prime_of_form_4k_plus_1_as_sum_of_two_squares_prime_of_form_8k_plus_3_as_sum_of_three_squares_l2179_217998


namespace NUMINAMATH_GPT_sum_of_1984_consecutive_integers_not_square_l2179_217980

theorem sum_of_1984_consecutive_integers_not_square :
  ∀ n : ℕ, ¬ ∃ k : ℕ, 992 * (2 * n + 1985) = k * k := by
  sorry

end NUMINAMATH_GPT_sum_of_1984_consecutive_integers_not_square_l2179_217980


namespace NUMINAMATH_GPT_smaller_angle_at_7_30_is_45_degrees_l2179_217991

noncomputable def calculateAngle (hour minute : Nat) : Real :=
  let minuteAngle := (minute * 6 : Real)
  let hourAngle := (hour % 12 * 30 : Real) + (minute / 60 * 30 : Real)
  let diff := abs (hourAngle - minuteAngle)
  if diff > 180 then 360 - diff else diff

theorem smaller_angle_at_7_30_is_45_degrees :
  calculateAngle 7 30 = 45 := 
sorry

end NUMINAMATH_GPT_smaller_angle_at_7_30_is_45_degrees_l2179_217991


namespace NUMINAMATH_GPT_percentage_passed_in_both_l2179_217936

def percentage_of_students_failing_hindi : ℝ := 30
def percentage_of_students_failing_english : ℝ := 42
def percentage_of_students_failing_both : ℝ := 28

theorem percentage_passed_in_both (P_H_E: percentage_of_students_failing_hindi + percentage_of_students_failing_english - percentage_of_students_failing_both = 44) : 
  100 - (percentage_of_students_failing_hindi + percentage_of_students_failing_english - percentage_of_students_failing_both) = 56 := by
  sorry

end NUMINAMATH_GPT_percentage_passed_in_both_l2179_217936


namespace NUMINAMATH_GPT_probability_all_co_captains_l2179_217908

-- Define the number of students in each team
def students_team1 : ℕ := 4
def students_team2 : ℕ := 6
def students_team3 : ℕ := 7
def students_team4 : ℕ := 9

-- Define the probability of selecting each team
def prob_selecting_team : ℚ := 1 / 4

-- Define the probability of selecting three co-captains from each team
def prob_team1 : ℚ := 1 / Nat.choose students_team1 3
def prob_team2 : ℚ := 1 / Nat.choose students_team2 3
def prob_team3 : ℚ := 1 / Nat.choose students_team3 3
def prob_team4 : ℚ := 1 / Nat.choose students_team4 3

-- Define the total probability
def total_prob : ℚ :=
  prob_selecting_team * (prob_team1 + prob_team2 + prob_team3 + prob_team4)

theorem probability_all_co_captains :
  total_prob = 59 / 1680 := by
  sorry

end NUMINAMATH_GPT_probability_all_co_captains_l2179_217908


namespace NUMINAMATH_GPT_fraction_meaningfulness_l2179_217900

def fraction_is_meaningful (x : ℝ) : Prop :=
  x ≠ 3 / 2

theorem fraction_meaningfulness (x : ℝ) : 
  (2 * x - 3) ≠ 0 ↔ fraction_is_meaningful x :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningfulness_l2179_217900


namespace NUMINAMATH_GPT_reporters_not_covering_politics_l2179_217975

-- Definitions of basic quantities
variables (R P : ℝ) (percentage_local : ℝ) (percentage_no_local : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  R = 100 ∧
  percentage_local = 10 ∧
  percentage_no_local = 30 ∧
  percentage_local = 0.7 * P

-- Theorem statement for the problem
theorem reporters_not_covering_politics (h : conditions R P percentage_local percentage_no_local) :
  100 - P = 85.71 :=
by sorry

end NUMINAMATH_GPT_reporters_not_covering_politics_l2179_217975


namespace NUMINAMATH_GPT_inequality_solution_l2179_217943

theorem inequality_solution (x : ℝ) :
  (x * (x + 2) > x * (3 - x) + 1) ↔ (x < -1/2 ∨ x > 1) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l2179_217943


namespace NUMINAMATH_GPT_kilometers_driven_equal_l2179_217918

theorem kilometers_driven_equal (x : ℝ) :
  (20 + 0.25 * x = 24 + 0.16 * x) → x = 44 := by
  sorry

end NUMINAMATH_GPT_kilometers_driven_equal_l2179_217918


namespace NUMINAMATH_GPT_triangle_type_l2179_217913

-- Let's define what it means for a triangle to be acute, obtuse, and right in terms of angle
def is_acute_triangle (a b c : ℝ) : Prop := (a < 90) ∧ (b < 90) ∧ (c < 90)
def is_obtuse_triangle (a b c : ℝ) : Prop := (a > 90) ∨ (b > 90) ∨ (c > 90)
def is_right_triangle (a b c : ℝ) : Prop := (a = 90) ∨ (b = 90) ∨ (c = 90)

-- The problem statement
theorem triangle_type (A B C : ℝ) (h : A = 100) : is_obtuse_triangle A B C :=
by {
  -- Sorry is used to indicate a placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_triangle_type_l2179_217913


namespace NUMINAMATH_GPT_ratio_sum_l2179_217938

theorem ratio_sum {x y : ℚ} (h : x / y = 4 / 7) : (x + y) / y = 11 / 7 :=
sorry

end NUMINAMATH_GPT_ratio_sum_l2179_217938


namespace NUMINAMATH_GPT_ellipse_foci_distance_2sqrt21_l2179_217952

noncomputable def ellipse_foci_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance_2sqrt21 :
  let center : ℝ × ℝ := (5, 2)
  let a := 5
  let b := 2
  ellipse_foci_distance a b = 2 * Real.sqrt 21 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_distance_2sqrt21_l2179_217952


namespace NUMINAMATH_GPT_comparison_of_a_b_c_l2179_217996

noncomputable def a : ℝ := 2018 ^ (1 / 2018)
noncomputable def b : ℝ := Real.logb 2017 (Real.sqrt 2018)
noncomputable def c : ℝ := Real.logb 2018 (Real.sqrt 2017)

theorem comparison_of_a_b_c :
  a > b ∧ b > c :=
by
  -- Definitions
  have def_a : a = 2018 ^ (1 / 2018) := rfl
  have def_b : b = Real.logb 2017 (Real.sqrt 2018) := rfl
  have def_c : c = Real.logb 2018 (Real.sqrt 2017) := rfl

  -- Sorry is added to skip the proof
  sorry

end NUMINAMATH_GPT_comparison_of_a_b_c_l2179_217996


namespace NUMINAMATH_GPT_probability_of_selecting_storybook_l2179_217942

theorem probability_of_selecting_storybook (reference_books storybooks picture_books : ℕ) 
  (h1 : reference_books = 5) (h2 : storybooks = 3) (h3 : picture_books = 2) :
  (storybooks : ℚ) / (reference_books + storybooks + picture_books) = 3 / 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_of_selecting_storybook_l2179_217942


namespace NUMINAMATH_GPT_value_of_m_l2179_217906

-- Defining the quadratic equation condition
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 3 * x + m^2 - 4

-- Defining the condition where the constant term in the quadratic equation is 0
def constant_term_zero (m : ℝ) : Prop := m^2 - 4 = 0

-- Stating the proof problem: given the conditions, prove that m = -2
theorem value_of_m (m : ℝ) (h1 : constant_term_zero m) (h2 : m ≠ 2) : m = -2 :=
by {
  sorry -- Proof to be developed
}

end NUMINAMATH_GPT_value_of_m_l2179_217906


namespace NUMINAMATH_GPT_compression_strength_value_l2179_217990

def compression_strength (T H : ℕ) : ℚ :=
  (15 * T^5) / (H^3)

theorem compression_strength_value : 
  compression_strength 3 6 = 55 / 13 := by
  sorry

end NUMINAMATH_GPT_compression_strength_value_l2179_217990


namespace NUMINAMATH_GPT_integer_a_conditions_l2179_217941

theorem integer_a_conditions (a : ℤ) :
  (∃ (x y : ℕ), x ≠ y ∧ (a * x * y + 1) ∣ (a * x^2 + 1) ^ 2) → a ≥ -1 :=
sorry

end NUMINAMATH_GPT_integer_a_conditions_l2179_217941


namespace NUMINAMATH_GPT_sector_area_is_80pi_l2179_217986

noncomputable def sectorArea (θ r : ℝ) : ℝ := 
  1 / 2 * θ * r^2

theorem sector_area_is_80pi :
  sectorArea (2 * Real.pi / 5) 20 = 80 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sector_area_is_80pi_l2179_217986


namespace NUMINAMATH_GPT_candies_eaten_l2179_217905

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end NUMINAMATH_GPT_candies_eaten_l2179_217905


namespace NUMINAMATH_GPT_tom_is_15_years_younger_l2179_217981

/-- 
Alice is now 30 years old.
Ten years ago, Alice was 4 times as old as Tom was then.
Prove that Tom is 15 years younger than Alice.
-/
theorem tom_is_15_years_younger (A T : ℕ) (h1 : A = 30) (h2 : A - 10 = 4 * (T - 10)) : A - T = 15 :=
by
  sorry

end NUMINAMATH_GPT_tom_is_15_years_younger_l2179_217981


namespace NUMINAMATH_GPT_remainder_sum_div_l2179_217958

theorem remainder_sum_div (n : ℤ) : ((9 - n) + (n + 5)) % 9 = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_sum_div_l2179_217958


namespace NUMINAMATH_GPT_project_completion_time_l2179_217992

theorem project_completion_time 
    (w₁ w₂ : ℕ) 
    (d₁ d₂ : ℕ) 
    (fraction₁ fraction₂ : ℝ)
    (h_work_fraction : fraction₁ = 1/2)
    (h_work_time : d₁ = 6)
    (h_first_workforce : w₁ = 90)
    (h_second_workforce : w₂ = 60)
    (h_fraction_done_by_first_team : w₁ * d₁ * (1 / 1080) = fraction₁)
    (h_fraction_done_by_second_team : w₂ * d₂ * (1 / 1080) = fraction₂)
    (h_total_fraction : fraction₂ = 1 - fraction₁) :
    d₂ = 9 :=
by 
  sorry

end NUMINAMATH_GPT_project_completion_time_l2179_217992


namespace NUMINAMATH_GPT_square_side_length_l2179_217974

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
by
sorry

end NUMINAMATH_GPT_square_side_length_l2179_217974


namespace NUMINAMATH_GPT_locus_of_points_line_or_point_l2179_217948

theorem locus_of_points_line_or_point {n : ℕ} (A B : ℕ → ℝ) (k : ℝ) (h : ∀ i, 1 ≤ i ∧ i < n → (A (i + 1) - A i) / (B (i + 1) - B i) = k) :
  ∃ l : ℝ, ∀ i, 1 ≤ i ∧ i ≤ n → (A i + l*B i) = A 1 + l*B 1 :=
by
  sorry

end NUMINAMATH_GPT_locus_of_points_line_or_point_l2179_217948


namespace NUMINAMATH_GPT_total_seeds_l2179_217947

theorem total_seeds (A B C : ℕ) (h₁ : A = B + 10) (h₂ : B = 30) (h₃ : C = 30) : A + B + C = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_seeds_l2179_217947


namespace NUMINAMATH_GPT_intersection_A_B_l2179_217903

def A : Set ℝ := {x | 2*x - 1 ≤ 0}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 1/2} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l2179_217903


namespace NUMINAMATH_GPT_sum_of_common_ratios_l2179_217935

theorem sum_of_common_ratios (k p r : ℝ) (h₁ : k ≠ 0) (h₂ : p ≠ r) (h₃ : (k * (p ^ 2)) - (k * (r ^ 2)) = 4 * (k * p - k * r)) : 
  p + r = 4 :=
by
  -- Using the conditions provided, we can prove the sum of the common ratios is 4.
  sorry

end NUMINAMATH_GPT_sum_of_common_ratios_l2179_217935


namespace NUMINAMATH_GPT_Janice_time_left_l2179_217993

-- Define the conditions as variables and parameters
def homework_time := 30
def cleaning_time := homework_time / 2
def dog_walking_time := homework_time + 5
def trash_time := homework_time / 6
def total_time_before_movie := 2 * 60

-- Calculation of total time required for all tasks
def total_time_required_for_tasks : Nat :=
  homework_time + cleaning_time + dog_walking_time + trash_time

-- Time left before the movie starts after completing all tasks
def time_left_before_movie : Nat :=
  total_time_before_movie - total_time_required_for_tasks

-- The final statement to prove
theorem Janice_time_left : time_left_before_movie = 35 :=
  by
    -- This will execute automatically to verify the theorem
    sorry

end NUMINAMATH_GPT_Janice_time_left_l2179_217993


namespace NUMINAMATH_GPT_lines_are_parallel_l2179_217951

-- Definitions of the conditions
variable (θ a p : Real)
def line1 := θ = a
def line2 := p * Real.sin (θ - a) = 1

-- The proof problem: Prove the two lines are parallel
theorem lines_are_parallel (h1 : line1 θ a) (h2 : line2 θ a p) : False :=
by
  sorry

end NUMINAMATH_GPT_lines_are_parallel_l2179_217951


namespace NUMINAMATH_GPT_percentage_of_whole_l2179_217976

theorem percentage_of_whole (part whole percent : ℕ) (h1 : part = 120) (h2 : whole = 80) (h3 : percent = 150) : 
  part = (percent / 100) * whole :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_whole_l2179_217976


namespace NUMINAMATH_GPT_total_insect_legs_l2179_217964

/--
This Lean statement defines the conditions and question,
proving that given 5 insects in the laboratory and each insect
having 6 legs, the total number of insect legs is 30.
-/
theorem total_insect_legs (n_insects : Nat) (legs_per_insect : Nat) (h1 : n_insects = 5) (h2 : legs_per_insect = 6) : (n_insects * legs_per_insect) = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_insect_legs_l2179_217964


namespace NUMINAMATH_GPT_diamond_eq_l2179_217915

noncomputable def diamond_op (a b : ℝ) (k : ℝ) : ℝ := sorry

theorem diamond_eq (x : ℝ) :
  let k := 2
  let a := 2023
  let b := 7
  let c := x
  (diamond_op a (diamond_op b c k) k = 150) ∧ 
  (∀ a b c, diamond_op a (diamond_op b c k) k = k * (diamond_op a b k) * c) ∧
  (∀ a, diamond_op a a k = k) →
  x = 150 / 2023 :=
sorry

end NUMINAMATH_GPT_diamond_eq_l2179_217915


namespace NUMINAMATH_GPT_smallest_number_conditions_l2179_217926

theorem smallest_number_conditions :
  ∃ n : ℤ, (n > 0) ∧
           (n % 2 = 1) ∧
           (n % 3 = 1) ∧
           (n % 4 = 1) ∧
           (n % 5 = 1) ∧
           (n % 6 = 1) ∧
           (n % 11 = 0) ∧
           (∀ m : ℤ, (m > 0) → 
             (m % 2 = 1) ∧
             (m % 3 = 1) ∧
             (m % 4 = 1) ∧
             (m % 5 = 1) ∧
             (m % 6 = 1) ∧
             (m % 11 = 0) → 
             (n ≤ m)) :=
sorry

end NUMINAMATH_GPT_smallest_number_conditions_l2179_217926


namespace NUMINAMATH_GPT_pencil_total_length_l2179_217967

-- Definitions of the colored sections
def purple_length : ℝ := 3.5
def black_length : ℝ := 2.8
def blue_length : ℝ := 1.6
def green_length : ℝ := 0.9
def yellow_length : ℝ := 1.2

-- The theorem stating the total length of the pencil
theorem pencil_total_length : purple_length + black_length + blue_length + green_length + yellow_length = 10 := 
by
  sorry

end NUMINAMATH_GPT_pencil_total_length_l2179_217967


namespace NUMINAMATH_GPT_repeating_decimal_as_fraction_l2179_217932

theorem repeating_decimal_as_fraction :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ Int.natAbs (Int.gcd a b) = 1 ∧ a + b = 15 ∧ (a : ℚ) / b = 0.3636363636363636 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_as_fraction_l2179_217932


namespace NUMINAMATH_GPT_smallest_solution_to_equation_l2179_217970

noncomputable def smallest_solution := (11 - Real.sqrt 445) / 6

theorem smallest_solution_to_equation:
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x = smallest_solution) :=
sorry

end NUMINAMATH_GPT_smallest_solution_to_equation_l2179_217970


namespace NUMINAMATH_GPT_total_books_l2179_217999

-- Defining the conditions
def darla_books := 6
def katie_books := darla_books / 2
def combined_books := darla_books + katie_books
def gary_books := 5 * combined_books

-- Statement to prove
theorem total_books : darla_books + katie_books + gary_books = 54 := by
  sorry

end NUMINAMATH_GPT_total_books_l2179_217999


namespace NUMINAMATH_GPT_an_values_and_formula_is_geometric_sequence_l2179_217924

-- Definitions based on the conditions
def Sn (n : ℕ) : ℝ := sorry  -- S_n to be defined in the context or problem details
def a (n : ℕ) : ℝ := 2 - Sn n

-- Prove the specific values and general formula given the condition a_n = 2 - S_n
theorem an_values_and_formula (Sn : ℕ → ℝ) :
  a 1 = 1 ∧ a 2 = 1 / 2 ∧ a 3 = 1 / 4 ∧ a 4 = 1 / 8 ∧ (∀ n, a n = (1 / 2)^(n-1)) :=
sorry

-- Prove the sequence is geometric
theorem is_geometric_sequence (Sn : ℕ → ℝ) :
  (∀ n, a n = (1 / 2)^(n-1)) → ∀ n, a (n + 1) / a n = 1 / 2 :=
sorry

end NUMINAMATH_GPT_an_values_and_formula_is_geometric_sequence_l2179_217924


namespace NUMINAMATH_GPT_find_remainder_division_l2179_217911

/--
Given:
1. A dividend of 100.
2. A quotient of 9.
3. A divisor of 11.

Prove: The remainder \( r \) when dividing 100 by 11 is 1.
-/
theorem find_remainder_division :
  ∀ (q d r : Nat), q = 9 → d = 11 → 100 = (d * q + r) → r = 1 :=
by
  intros q d r hq hd hdiv
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_find_remainder_division_l2179_217911


namespace NUMINAMATH_GPT_max_brownies_l2179_217944

theorem max_brownies (m n : ℕ) (h1 : (m-2)*(n-2) = 2*(2*m + 2*n - 4)) : m * n ≤ 294 :=
by sorry

end NUMINAMATH_GPT_max_brownies_l2179_217944


namespace NUMINAMATH_GPT_simplify_expression_l2179_217960

theorem simplify_expression
  (a b c : ℝ) 
  (hnz_a : a ≠ 0) 
  (hnz_b : b ≠ 0) 
  (hnz_c : c ≠ 0) 
  (h_sum : a + b + c = 0) :
  (1 / (b^3 + c^3 - a^3)) + (1 / (a^3 + c^3 - b^3)) + (1 / (a^3 + b^3 - c^3)) = 1 / (a * b * c) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2179_217960


namespace NUMINAMATH_GPT_temperature_difference_l2179_217979

-- Definitions based on the conditions
def refrigeration_compartment_temperature : ℤ := 5
def freezer_compartment_temperature : ℤ := -2

-- Mathematically equivalent proof problem statement
theorem temperature_difference : refrigeration_compartment_temperature - freezer_compartment_temperature = 7 := by
  sorry

end NUMINAMATH_GPT_temperature_difference_l2179_217979


namespace NUMINAMATH_GPT_triangle_is_isosceles_l2179_217959

theorem triangle_is_isosceles 
  (A B C : ℝ) 
  (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) 
  (h₀ : A + B + C = π) :
  (A = B) := 
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l2179_217959


namespace NUMINAMATH_GPT_a3_plus_a4_l2179_217978

def sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 3^(n + 1)

theorem a3_plus_a4 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : sum_of_sequence S a) :
  a 3 + a 4 = 216 :=
sorry

end NUMINAMATH_GPT_a3_plus_a4_l2179_217978


namespace NUMINAMATH_GPT_complex_multiplication_l2179_217930

def imaginary_unit := Complex.I

theorem complex_multiplication (h : imaginary_unit^2 = -1) : (3 + 2 * imaginary_unit) * imaginary_unit = -2 + 3 * imaginary_unit :=
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l2179_217930


namespace NUMINAMATH_GPT_problem_solution_l2179_217982

theorem problem_solution 
  (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) :
  4 * x^4 + 17 * x^2 * y + 4 * y^2 < (m / 4) * (x^4 + 2 * x^2 * y + y^2) ↔ 25 < m :=
sorry

end NUMINAMATH_GPT_problem_solution_l2179_217982
