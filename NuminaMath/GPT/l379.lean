import Mathlib

namespace product_of_real_roots_l379_37925

theorem product_of_real_roots (x : ℝ) (h : x^5 = 100) : x = 10^(2/5) := by
  sorry

end product_of_real_roots_l379_37925


namespace sum_of_reciprocals_l379_37990

variable (x y : ℝ)

theorem sum_of_reciprocals (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  (1 / x) + (1 / y) = 3 := 
sorry

end sum_of_reciprocals_l379_37990


namespace possible_values_on_Saras_card_l379_37918

theorem possible_values_on_Saras_card :
  ∀ (y : ℝ), (0 < y ∧ y < π / 2) →
  let sin_y := Real.sin y
  let cos_y := Real.cos y
  let tan_y := Real.tan y
  (∃ (s l k : ℝ), s = sin_y ∧ l = cos_y ∧ k = tan_y ∧
  (s = l ∨ s = k ∨ l = k) ∧ (s = l ∧ l ≠ k) ∧ s = l ∧ s = 1) :=
sorry

end possible_values_on_Saras_card_l379_37918


namespace arithmetic_sequence_sum_l379_37945

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h : ∀ n, a n = a 1 + (n - 1) * d) (h_6 : a 6 = 1) :
  a 2 + a 10 = 2 := 
sorry

end arithmetic_sequence_sum_l379_37945


namespace total_hamburgers_for_lunch_l379_37970

theorem total_hamburgers_for_lunch 
  (initial_hamburgers: ℕ) 
  (additional_hamburgers: ℕ)
  (h1: initial_hamburgers = 9)
  (h2: additional_hamburgers = 3)
  : initial_hamburgers + additional_hamburgers = 12 := 
by
  sorry

end total_hamburgers_for_lunch_l379_37970


namespace find_x_l379_37980

theorem find_x :
  ∃ X : ℝ, 0.25 * X + 0.20 * 40 = 23 ∧ X = 60 :=
by
  sorry

end find_x_l379_37980


namespace movie_ticket_cost_l379_37902

-- Definitions from conditions
def total_spending : ℝ := 36
def combo_meal_cost : ℝ := 11
def candy_cost : ℝ := 2.5
def total_food_cost : ℝ := combo_meal_cost + 2 * candy_cost
def total_ticket_cost (x : ℝ) : ℝ := 2 * x

-- The theorem stating the proof problem
theorem movie_ticket_cost :
  ∃ (x : ℝ), total_ticket_cost x + total_food_cost = total_spending ∧ x = 10 :=
by
  sorry

end movie_ticket_cost_l379_37902


namespace quadratic_coefficients_l379_37976

theorem quadratic_coefficients :
  ∀ x : ℝ, 3 * x^2 = 5 * x - 1 → (∃ a b c : ℝ, a = 3 ∧ b = -5 ∧ a * x^2 + b * x + c = 0) :=
by
  intro x h
  use 3, -5, 1
  sorry

end quadratic_coefficients_l379_37976


namespace factorize_expression_l379_37950

theorem factorize_expression (x : ℝ) : x^3 - 2 * x^2 + x = x * (x - 1)^2 :=
by sorry

end factorize_expression_l379_37950


namespace length_of_AB_l379_37916

noncomputable def AB_CD_sum_240 (AB CD : ℝ) (h : ℝ) : Prop :=
  AB + CD = 240

noncomputable def ratio_of_areas (AB CD : ℝ) : Prop :=
  AB / CD = 5 / 3

theorem length_of_AB (AB CD : ℝ) (h : ℝ) (h_ratio : ratio_of_areas AB CD) (h_sum : AB_CD_sum_240 AB CD h) : AB = 150 :=
by
  unfold ratio_of_areas at h_ratio
  unfold AB_CD_sum_240 at h_sum
  sorry

end length_of_AB_l379_37916


namespace number_of_girls_l379_37989

theorem number_of_girls
  (B : ℕ) (k : ℕ) (G : ℕ)
  (hB : B = 10) 
  (hk : k = 5)
  (h1 : B / k = 2)
  (h2 : G % k = 0) :
  G = 5 := 
sorry

end number_of_girls_l379_37989


namespace transformed_triangle_area_l379_37972

-- Define the function g and its properties
variable {R : Type*} [LinearOrderedField R]
variable (g : R → R)
variable (a b c : R)
variable (area_original : R)

-- Given conditions
-- The function g is defined such that the area of the triangle formed by 
-- points (a, g(a)), (b, g(b)), and (c, g(c)) is 24
axiom h₀ : {x | x = a ∨ x = b ∨ x = c} ⊆ Set.univ
axiom h₁ : area_original = 24

-- Define a function that computes the area of a triangle given three points
noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : R) : R := 
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem transformed_triangle_area (h₀ : {x | x = a ∨ x = b ∨ x = c} ⊆ Set.univ)
  (h₁ : area_triangle a (g a) b (g b) c (g c) = 24) :
  area_triangle (a / 3) (3 * g a) (b / 3) (3 * g b) (c / 3) (3 * g c) = 24 :=
sorry

end transformed_triangle_area_l379_37972


namespace inequality_proof_l379_37929

open Real

theorem inequality_proof
  (a b c x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hx_cond : 1 / x + 1 / y + 1 / z = 1) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a ^ x + b ^ y + c ^ z ≥ (4 * a * b * c * x * y * z) / (x + y + z - 3) ^ 2 :=
by
  sorry

end inequality_proof_l379_37929


namespace distance_from_negative_two_is_three_l379_37911

theorem distance_from_negative_two_is_three (x : ℝ) : abs (x + 2) = 3 → (x = -5) ∨ (x = 1) :=
  sorry

end distance_from_negative_two_is_three_l379_37911


namespace fourth_child_sweets_l379_37965

theorem fourth_child_sweets (total_sweets : ℕ) (mother_sweets : ℕ) (child_sweets : ℕ) 
  (Y E T F: ℕ) (h1 : total_sweets = 120) (h2 : mother_sweets = total_sweets / 4) 
  (h3 : child_sweets = total_sweets - mother_sweets) 
  (h4 : E = 2 * Y) (h5 : T = F - 8) 
  (h6 : Y = (8 * (T + 6)) / 10) 
  (h7 : Y + E + (T + 6) + (F - 8) + F = child_sweets) : 
  F = 24 :=
by
  sorry

end fourth_child_sweets_l379_37965


namespace range_of_a_squared_plus_b_l379_37961

variable (a b : ℝ)

theorem range_of_a_squared_plus_b (h1 : a < -2) (h2 : b > 4) : ∃ y, y = a^2 + b ∧ 8 < y :=
by
  sorry

end range_of_a_squared_plus_b_l379_37961


namespace avg_weight_b_c_43_l379_37987

noncomputable def weights_are_correct (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧ (A + B) / 2 = 40 ∧ B = 31

theorem avg_weight_b_c_43 (A B C : ℝ) (h : weights_are_correct A B C) : (B + C) / 2 = 43 :=
by sorry

end avg_weight_b_c_43_l379_37987


namespace value_of_f_at_neg1_l379_37917

def f (x : ℤ) : ℤ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

theorem value_of_f_at_neg1 : f (-1) = 6 :=
by
  sorry

end value_of_f_at_neg1_l379_37917


namespace n_is_900_l379_37926

theorem n_is_900 
  (m n : ℕ) 
  (h1 : ∃ x y : ℤ, m = x^2 ∧ n = y^2) 
  (h2 : Prime (m - n)) : n = 900 := 
sorry

end n_is_900_l379_37926


namespace evaluate_exponent_l379_37920

theorem evaluate_exponent : (3^2)^4 = 6561 := sorry

end evaluate_exponent_l379_37920


namespace shelves_needed_l379_37944

def books_in_stock : Nat := 27
def books_sold : Nat := 6
def books_per_shelf : Nat := 7

theorem shelves_needed :
  let remaining_books := books_in_stock - books_sold
  let shelves := remaining_books / books_per_shelf
  shelves = 3 :=
by
  sorry

end shelves_needed_l379_37944


namespace extreme_values_f_a4_no_zeros_f_on_1e_l379_37956

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - (a + 2) * Real.log x - 2 / x + 2

theorem extreme_values_f_a4 :
  f 4 (1 / 2) = 6 * Real.log 2 ∧ f 4 1 = 4 := sorry

theorem no_zeros_f_on_1e (a : ℝ) :
  (a ≤ 0 ∨ a ≥ 2 / (Real.exp 1 * (Real.exp 1 - 1))) →
  ∀ x, 1 < x → x < Real.exp 1 → f a x ≠ 0 := sorry

end extreme_values_f_a4_no_zeros_f_on_1e_l379_37956


namespace interval_of_x_l379_37923

theorem interval_of_x (x : ℝ) : (4 * x > 2) ∧ (4 * x < 5) ∧ (5 * x > 2) ∧ (5 * x < 5) ↔ (x > 1/2) ∧ (x < 1) := 
by 
  sorry

end interval_of_x_l379_37923


namespace number_of_committees_correct_l379_37988

noncomputable def number_of_committees (teams members host_selection non_host_selection : ℕ) : ℕ :=
  have ways_to_choose_host := teams
  have ways_to_choose_four_from_seven := Nat.choose members host_selection
  have ways_to_choose_two_from_seven := Nat.choose members non_host_selection
  have total_non_host_combinations := ways_to_choose_two_from_seven ^ (teams - 1)
  ways_to_choose_host * ways_to_choose_four_from_seven * total_non_host_combinations

theorem number_of_committees_correct :
  number_of_committees 5 7 4 2 = 34134175 := by
  sorry

end number_of_committees_correct_l379_37988


namespace quadratic_inequality_solution_l379_37974

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + x - 12 > 0) → (x > 3 ∨ x < -4) :=
by
  sorry

end quadratic_inequality_solution_l379_37974


namespace gauss_polynomial_reciprocal_l379_37901

def gauss_polynomial (k l : ℤ) (x : ℝ) : ℝ := sorry -- Placeholder for actual polynomial definition

theorem gauss_polynomial_reciprocal (k l : ℤ) (x : ℝ) : 
  x^(k * l) * gauss_polynomial k l (1 / x) = gauss_polynomial k l x :=
sorry

end gauss_polynomial_reciprocal_l379_37901


namespace algebraic_expression_1_algebraic_expression_2_l379_37900

-- Problem 1
theorem algebraic_expression_1 (a : ℚ) (h : a = 4 / 5) : -24.7 * a + 1.3 * a - (33 / 5) * a = -24 := 
by 
  sorry

-- Problem 2
theorem algebraic_expression_2 (a b : ℕ) (ha : a = 899) (hb : b = 101) : a^2 + 2 * a * b + b^2 = 1000000 := 
by 
  sorry

end algebraic_expression_1_algebraic_expression_2_l379_37900


namespace number_of_boys_in_biology_class_l379_37991

variable (B G : ℕ) (PhysicsClass BiologyClass : ℕ)

theorem number_of_boys_in_biology_class
  (h1 : G = 3 * B)
  (h2 : PhysicsClass = 200)
  (h3 : BiologyClass = PhysicsClass / 2)
  (h4 : BiologyClass = B + G) :
  B = 25 := by
  sorry

end number_of_boys_in_biology_class_l379_37991


namespace parabola_coefficients_sum_l379_37973

theorem parabola_coefficients_sum (a b c : ℝ)
  (h_eqn : ∀ y, (-1) = a * y^2 + b * y + c)
  (h_vertex : (-1, -10) = (-a/(2*a), (4*a*c - b^2)/(4*a)))
  (h_pass_point : 0 = a * (-9)^2 + b * (-9) + c) 
  : a + b + c = 120 := 
sorry

end parabola_coefficients_sum_l379_37973


namespace perfect_square_solutions_l379_37959

theorem perfect_square_solutions :
  {n : ℕ | ∃ m : ℕ, n^2 + 77 * n = m^2} = {4, 99, 175, 1444} :=
by
  sorry

end perfect_square_solutions_l379_37959


namespace marbles_given_l379_37966

theorem marbles_given (initial remaining given : ℕ) (h_initial : initial = 143) (h_remaining : remaining = 70) :
    given = initial - remaining → given = 73 :=
by
  intros
  sorry

end marbles_given_l379_37966


namespace factorization_example_l379_37977

theorem factorization_example (a b : ℕ) : (a - 2*b)^2 = a^2 - 4*a*b + 4*b^2 := 
by sorry

end factorization_example_l379_37977


namespace alexis_shirt_expense_l379_37993

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

end alexis_shirt_expense_l379_37993


namespace max_k_pos_l379_37969

-- Define the sequences {a_n} and {b_n}
def sequence_a (n k : ℤ) : ℤ := 2 * n + k - 1
def sequence_b (n : ℤ) : ℤ := 3 * n + 2

-- Conditions and given values
def S (n k : ℤ) : ℤ := n + k
def sum_first_9_b : ℤ := 153
def b_3 : ℤ := 11

-- Given the sequence {c_n}
def sequence_c (n k : ℤ) : ℤ := sequence_a n k - k * sequence_b n

-- Define the sum of the first n terms of the sequence {c_n}
def T (n k : ℤ) : ℤ := (n * (2 * sequence_c 1 k + (n - 1) * (2 - 3 * k))) / 2

-- Proof problem statement
theorem max_k_pos (k : ℤ) : (∀ n : ℤ, n > 0 → T n k > 0) → k ≤ 1 :=
sorry

end max_k_pos_l379_37969


namespace sequence_number_theorem_l379_37940

def seq_count (n k : ℕ) : ℕ :=
  -- Sequence count function definition given the conditions.
  sorry -- placeholder, as we are only defining the statement, not the function itself.

theorem sequence_number_theorem (n k : ℕ) : seq_count n k = Nat.choose (n-1) k :=
by
  -- This is where the proof would go, currently omitted.
  sorry

end sequence_number_theorem_l379_37940


namespace sandro_children_ratio_l379_37907

theorem sandro_children_ratio (d : ℕ) (h1 : d + 3 = 21) : d / 3 = 6 :=
by
  sorry

end sandro_children_ratio_l379_37907


namespace minute_hour_hands_opposite_l379_37905

theorem minute_hour_hands_opposite (x : ℝ) (h1 : 10 * 60 ≤ x) (h2 : x ≤ 11 * 60) : 
  (5.5 * x = 442.5) :=
sorry

end minute_hour_hands_opposite_l379_37905


namespace find_smaller_number_l379_37932

theorem find_smaller_number (a b : ℤ) (h1 : a + b = 18) (h2 : a - b = 24) : b = -3 :=
by
  sorry

end find_smaller_number_l379_37932


namespace num_valid_lists_l379_37979

-- Define a predicate for a list to satisfy the given constraints
def valid_list (l : List ℕ) : Prop :=
  l = List.range' 1 12 ∧ ∀ i, 1 < i ∧ i ≤ 12 → (l.indexOf (l.get! (i - 1) + 1) < i - 1 ∨ l.indexOf (l.get! (i - 1) - 1) < i - 1) ∧ ¬(l.indexOf (l.get! (i - 1) + 1) < i - 1 ∧ l.indexOf (l.get! (i - 1) - 1) < i - 1)

-- Prove that there is exactly one valid list of such nature
theorem num_valid_lists : ∃! l : List ℕ, valid_list l :=
  sorry

end num_valid_lists_l379_37979


namespace op_five_two_is_twentyfour_l379_37985

def op (x y : Int) : Int :=
  (x + y + 1) * (x - y)

theorem op_five_two_is_twentyfour : op 5 2 = 24 := by
  unfold op
  sorry

end op_five_two_is_twentyfour_l379_37985


namespace positive_solution_unique_m_l379_37915

theorem positive_solution_unique_m (m : ℝ) : ¬ (4 < m ∧ m < 2) :=
by
  sorry

end positive_solution_unique_m_l379_37915


namespace g_neg6_eq_neg28_l379_37914

-- Define the given function g
def g (x : ℝ) : ℝ := 2 * x^7 - 3 * x^3 + 4 * x - 8

-- State the main theorem to prove g(-6) = -28 under the given conditions
theorem g_neg6_eq_neg28 (h1 : g 6 = 12) : g (-6) = -28 :=
by
  sorry

end g_neg6_eq_neg28_l379_37914


namespace toys_produced_per_week_l379_37943

theorem toys_produced_per_week (daily_production : ℕ) (work_days_per_week : ℕ) (total_production : ℕ) :
  daily_production = 680 ∧ work_days_per_week = 5 → total_production = 3400 := by
  sorry

end toys_produced_per_week_l379_37943


namespace total_trash_cans_paid_for_l379_37939

-- Definitions based on conditions
def trash_cans_on_streets : ℕ := 14
def trash_cans_back_of_stores : ℕ := 2 * trash_cans_on_streets

-- Theorem to prove
theorem total_trash_cans_paid_for : trash_cans_on_streets + trash_cans_back_of_stores = 42 := 
by
  -- proof would go here, but we use sorry since proof is not required
  sorry

end total_trash_cans_paid_for_l379_37939


namespace negation_of_proposition_is_false_l379_37935

theorem negation_of_proposition_is_false :
  (¬ ∀ (x : ℝ), x < 0 → x^2 > 0) = true :=
by
  sorry

end negation_of_proposition_is_false_l379_37935


namespace peter_pizza_fraction_l379_37904

def pizza_slices : ℕ := 16
def peter_initial_slices : ℕ := 2
def shared_slices : ℕ := 2
def shared_with_paul : ℕ := shared_slices / 2
def total_slices_peter_ate := peter_initial_slices + shared_with_paul
def fraction_peter_ate : ℚ := total_slices_peter_ate / pizza_slices

theorem peter_pizza_fraction :
  fraction_peter_ate = 3 / 16 :=
by
  -- Leave space for the proof, which is not required.
  sorry

end peter_pizza_fraction_l379_37904


namespace parabola_focus_coordinates_l379_37999

theorem parabola_focus_coordinates :
  ∀ x y : ℝ, y^2 = -8 * x → (x, y) = (-2, 0) := by
  sorry

end parabola_focus_coordinates_l379_37999


namespace train_length_l379_37933

noncomputable def relative_speed_kmh (vA vB : ℝ) : ℝ :=
  vA - vB

noncomputable def relative_speed_mps (relative_speed_kmh : ℝ) : ℝ :=
  relative_speed_kmh * (5 / 18)

noncomputable def distance_covered (relative_speed_mps : ℝ) (time_s : ℝ) : ℝ :=
  relative_speed_mps * time_s

theorem train_length (vA_kmh : ℝ) (vB_kmh : ℝ) (time_s : ℝ) (L : ℝ) 
  (h1 : vA_kmh = 42) (h2 : vB_kmh = 36) (h3 : time_s = 36) 
  (h4 : distance_covered (relative_speed_mps (relative_speed_kmh vA_kmh vB_kmh)) time_s = 2 * L) :
  L = 30 :=
by
  sorry

end train_length_l379_37933


namespace inverse_proportion_inequality_l379_37963

theorem inverse_proportion_inequality {x1 x2 : ℝ} (h1 : x1 > x2) (h2 : x2 > 0) : 
    -3 / x1 > -3 / x2 := 
by 
  sorry

end inverse_proportion_inequality_l379_37963


namespace g_difference_l379_37930

def g (n : ℕ) : ℚ :=
  1/4 * n * (n + 1) * (n + 2) * (n + 3)

theorem g_difference (r : ℕ) : g r - g (r - 1) = r * (r + 1) * (r + 2) :=
  sorry

end g_difference_l379_37930


namespace candy_pebbles_l379_37934

theorem candy_pebbles (C L : ℕ) 
  (h1 : L = 3 * C)
  (h2 : L = C + 8) :
  C = 4 :=
by
  sorry

end candy_pebbles_l379_37934


namespace smallest_xym_sum_l379_37995

def is_two_digit_integer (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

def reversed_digits (x y : ℤ) : Prop :=
  ∃ a b : ℤ, x = 10 * a + b ∧ y = 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9

def odd_multiple_of_9 (n : ℤ) : Prop :=
  ∃ k : ℤ, k % 2 = 1 ∧ n = 9 * k

theorem smallest_xym_sum :
  ∃ (x y m : ℤ), is_two_digit_integer x ∧ is_two_digit_integer y ∧ reversed_digits x y ∧ x^2 + y^2 = m^2 ∧ odd_multiple_of_9 (x + y) ∧ x + y + m = 169 :=
by
  sorry

end smallest_xym_sum_l379_37995


namespace one_circle_equiv_three_squares_l379_37951

-- Define the weights of circles and squares symbolically
variables {w_circle w_square : ℝ}

-- Equations based on the conditions in the problem
-- 3 circles balance 5 squares
axiom eq1 : 3 * w_circle = 5 * w_square

-- 2 circles balance 3 squares and 1 circle
axiom eq2 : 2 * w_circle = 3 * w_square + w_circle

-- We need to prove that 1 circle is equivalent to 3 squares
theorem one_circle_equiv_three_squares : w_circle = 3 * w_square := 
by sorry

end one_circle_equiv_three_squares_l379_37951


namespace treasures_first_level_is_4_l379_37921

-- Definitions based on conditions
def points_per_treasure : ℕ := 5
def treasures_second_level : ℕ := 3
def score_second_level : ℕ := treasures_second_level * points_per_treasure
def total_score : ℕ := 35
def points_first_level : ℕ := total_score - score_second_level

-- Main statement to prove
theorem treasures_first_level_is_4 : points_first_level / points_per_treasure = 4 := 
by
  -- We are skipping the proof here and using sorry.
  sorry

end treasures_first_level_is_4_l379_37921


namespace closure_property_of_A_l379_37962

theorem closure_property_of_A 
  (a b c d k1 k2 : ℤ) 
  (x y : ℤ) 
  (Hx : x = a^2 + k1 * a * b + b^2) 
  (Hy : y = c^2 + k2 * c * d + d^2) : 
  ∃ m k : ℤ, x * y = m * (a^2 + k * a * b + b^2) := 
  by 
    -- this is where the proof would go
    sorry

end closure_property_of_A_l379_37962


namespace solve_max_eq_l379_37957

theorem solve_max_eq (x : ℚ) (h : max x (-x) = 2 * x + 1) : x = -1 / 3 := by
  sorry

end solve_max_eq_l379_37957


namespace trevor_spending_proof_l379_37908

def trevor_spends (T R Q : ℕ) : Prop :=
  T = R + 20 ∧ R = 2 * Q ∧ 4 * T + 4 * R + 2 * Q = 680

theorem trevor_spending_proof (T R Q : ℕ) (h : trevor_spends T R Q) : T = 80 :=
by sorry

end trevor_spending_proof_l379_37908


namespace scientific_notation_of_909_000_000_000_l379_37953

theorem scientific_notation_of_909_000_000_000 :
    ∃ (a : ℝ) (n : ℤ), 909000000000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 9.09 ∧ n = 11 := 
sorry

end scientific_notation_of_909_000_000_000_l379_37953


namespace xyz_range_l379_37948

theorem xyz_range (x y z : ℝ) (h1 : x + y + z = 1) (h2 : x^2 + y^2 + z^2 = 3) : 
  -1 ≤ x * y * z ∧ x * y * z ≤ 5 / 27 := 
sorry

end xyz_range_l379_37948


namespace intersection_A_B_l379_37983

def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x < 5 / 2}

theorem intersection_A_B :
  A ∩ B = {0, 1, 2} := by
  sorry

end intersection_A_B_l379_37983


namespace meteorological_forecast_l379_37986

theorem meteorological_forecast (prob_rain : ℝ) (h1 : prob_rain = 0.7) :
  (prob_rain = 0.7) → "There is a high probability of needing to carry rain gear when going out tomorrow." = "Correct" :=
by
  intro h
  sorry

end meteorological_forecast_l379_37986


namespace find_value_of_15b_minus_2a_l379_37998

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if 1 ≤ x ∧ x < 2 then x + a / x
else if 2 ≤ x ∧ x ≤ 3 then b * x - 3
else 0

theorem find_value_of_15b_minus_2a (a b : ℝ)
  (h_periodic : ∀ x : ℝ, f x a b = f (x + 2) a b)
  (h_condition : f (7 / 2) a b = f (-7 / 2) a b) :
  15 * b - 2 * a = 41 :=
sorry

end find_value_of_15b_minus_2a_l379_37998


namespace interchange_digits_product_l379_37996

-- Definition of the proof problem
theorem interchange_digits_product (n a b k : ℤ) (h1 : n = 10 * a + b) (h2 : n = (k + 1) * (a + b)) :
  ∃ x : ℤ, (10 * b + a) = x * (a + b) ∧ x = 10 - k :=
by
  existsi (10 - k)
  sorry

end interchange_digits_product_l379_37996


namespace sum_eq_24_of_greatest_power_l379_37952

theorem sum_eq_24_of_greatest_power (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_b_gt_1 : b > 1) (h_a_pow_b_lt_500 : a^b < 500)
  (h_greatest : ∀ (x y : ℕ), (0 < x) → (0 < y) → (y > 1) → (x^y < 500) → (x^y ≤ a^b)) : a + b = 24 :=
  sorry

end sum_eq_24_of_greatest_power_l379_37952


namespace area_of_circle_l379_37958

theorem area_of_circle:
  (∃ (r : ℝ) (θ : ℝ), r = 3 * Real.cos θ - 4 * Real.sin θ) → ∃ area: ℝ, area = (25/4) * Real.pi :=
sorry

end area_of_circle_l379_37958


namespace first_team_engineers_l379_37971

theorem first_team_engineers (E : ℕ) 
  (teamQ_engineers : ℕ := 16) 
  (work_days_teamQ : ℕ := 30) 
  (work_days_first_team : ℕ := 32) 
  (working_capacity_ratio : ℚ := 3 / 2) :
  E * work_days_first_team * 3 = teamQ_engineers * work_days_teamQ * 2 → 
  E = 10 :=
by
  sorry

end first_team_engineers_l379_37971


namespace paul_spending_l379_37937

theorem paul_spending :
  let cost_of_dress_shirts := 4 * 15
  let cost_of_pants := 2 * 40
  let cost_of_suit := 150
  let cost_of_sweaters := 2 * 30
  let total_cost := cost_of_dress_shirts + cost_of_pants + cost_of_suit + cost_of_sweaters
  let store_discount := 0.2 * total_cost
  let after_store_discount := total_cost - store_discount
  let coupon_discount := 0.1 * after_store_discount
  let final_amount := after_store_discount - coupon_discount
  final_amount = 252 :=
by
  -- Mathematically equivalent proof problem.
  sorry

end paul_spending_l379_37937


namespace eval_expression_l379_37954

theorem eval_expression : (-2 ^ 3) ^ (1/3 : ℝ) - (-1 : ℝ) ^ 0 = -3 := by 
  sorry

end eval_expression_l379_37954


namespace total_messages_three_days_l379_37924

theorem total_messages_three_days :
  ∀ (A1 A2 A3 L1 L2 L3 : ℕ),
  A1 = L1 - 20 →
  L1 = 120 →
  L2 = (1 / 3 : ℚ) * L1 →
  A2 = 2 * A1 →
  A1 + L1 = A3 + L3 →
  (A1 + L1 + A2 + L2 + A3 + L3 = 680) := by
  intros A1 A2 A3 L1 L2 L3 h1 h2 h3 h4 h5
  sorry

end total_messages_three_days_l379_37924


namespace equation_of_line_AB_l379_37955

noncomputable def center_of_circle : (ℝ × ℝ) := (-4, -1)

noncomputable def point_P : (ℝ × ℝ) := (2, 3)

noncomputable def slope_OP : ℝ :=
  let (x₁, y₁) := center_of_circle
  let (x₂, y₂) := point_P
  (y₂ - y₁) / (x₂ - x₁)

noncomputable def slope_AB : ℝ :=
  -1 / slope_OP

theorem equation_of_line_AB : (6 * x + 4 * y + 19 = 0) :=
  sorry

end equation_of_line_AB_l379_37955


namespace remaining_load_after_three_deliveries_l379_37909

def initial_load : ℝ := 50000
def unload_first_store (load : ℝ) : ℝ := load - 0.10 * load
def unload_second_store (load : ℝ) : ℝ := load - 0.20 * load
def unload_third_store (load : ℝ) : ℝ := load - 0.15 * load

theorem remaining_load_after_three_deliveries : 
  unload_third_store (unload_second_store (unload_first_store initial_load)) = 30600 := 
by
  sorry

end remaining_load_after_three_deliveries_l379_37909


namespace molecular_weight_3_moles_ascorbic_acid_l379_37975

def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

def molecular_formula_ascorbic_acid : List (ℝ × ℕ) :=
  [(atomic_weight_C, 6), (atomic_weight_H, 8), (atomic_weight_O, 6)]

def molecular_weight (formula : List (ℝ × ℕ)) : ℝ :=
  formula.foldl (λ acc (aw, count) => acc + aw * count) 0.0

def weight_of_moles (mw : ℝ) (moles : ℕ) : ℝ :=
  mw * moles

theorem molecular_weight_3_moles_ascorbic_acid :
  weight_of_moles (molecular_weight molecular_formula_ascorbic_acid) 3 = 528.372 :=
by
  sorry

end molecular_weight_3_moles_ascorbic_acid_l379_37975


namespace number_of_tins_per_day_for_rest_of_week_l379_37919
-- Import necessary library

-- Define conditions as Lean definitions
def d1 : ℕ := 50
def d2 : ℕ := 3 * d1
def d3 : ℕ := d2 - 50
def total_target : ℕ := 500

-- Define what we need to prove
theorem number_of_tins_per_day_for_rest_of_week :
  ∃ (dr : ℕ), d1 + d2 + d3 + 4 * dr = total_target ∧ dr = 50 :=
by
  sorry

end number_of_tins_per_day_for_rest_of_week_l379_37919


namespace angle_BPE_l379_37913

-- Define the conditions given in the problem
def triangle_ABC (A B C : ℝ) : Prop := A = 60 ∧ 
  (∃ (B₁ B₂ B₃ : ℝ), B₁ = B / 3 ∧ B₂ = B / 3 ∧ B₃ = B / 3) ∧ 
  (∃ (C₁ C₂ C₃ : ℝ), C₁ = C / 3 ∧ C₂ = C / 3 ∧ C₃ = C / 3) ∧ 
  (B + C = 120)

-- State the theorem to proof
theorem angle_BPE (A B C x : ℝ) (h : triangle_ABC A B C) : x = 50 := by
  sorry

end angle_BPE_l379_37913


namespace apple_distribution_l379_37964

theorem apple_distribution (x : ℕ) (h₁ : 1430 % x = 0) (h₂ : 1430 % (x + 45) = 0) (h₃ : 1430 / x - 1430 / (x + 45) = 9) : 
  1430 / x = 22 :=
by
  sorry

end apple_distribution_l379_37964


namespace integral_fx_l379_37903

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x

theorem integral_fx :
  ∫ x in -Real.pi..0, f x = -2 - (1/2) * Real.pi ^ 2 :=
by
  sorry

end integral_fx_l379_37903


namespace exists_special_number_l379_37906

theorem exists_special_number :
  ∃ N : ℕ, (∀ k : ℕ, (1 ≤ k ∧ k ≤ 149 → k ∣ N) ∨ (k + 1 ∣ N) = false) :=
sorry

end exists_special_number_l379_37906


namespace radius_B_eq_8_div_9_l379_37981

-- Define the circles and their properties
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Given conditions
variable (A B C D : Circle)
variable (h1 : A.radius = 1)
variable (h2 : A.radius + A.radius = D.radius)
variable (h3 : B.radius = C.radius)
variable (h4 : (A.center.1 - B.center.1)^2 + (A.center.2 - B.center.2)^2 = (A.radius + B.radius)^2)
variable (h5 : (A.center.1 - C.center.1)^2 + (A.center.2 - C.center.2)^2 = (A.radius + C.radius)^2)
variable (h6 : (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 = (B.radius + C.radius)^2)
variable (h7 : (D.center.1 - A.center.1)^2 + (D.center.2 - A.center.2)^2 = D.radius^2)

-- Prove the radius of circle B is 8/9
theorem radius_B_eq_8_div_9 : B.radius = 8 / 9 := 
by
  sorry

end radius_B_eq_8_div_9_l379_37981


namespace slower_train_speed_l379_37982

-- Conditions
variables (L : ℕ) -- Length of each train (in meters)
variables (v_f : ℕ) -- Speed of the faster train (in km/hr)
variables (t : ℕ) -- Time taken by the faster train to pass the slower one (in seconds)
variables (v_s : ℕ) -- Speed of the slower train (in km/hr)

-- Assumptions based on conditions of the problem
axiom length_eq : L = 30
axiom fast_speed : v_f = 42
axiom passing_time : t = 36

-- Conversion for km/hr to m/s
def km_per_hr_to_m_per_s (v : ℕ) : ℕ := (v * 5) / 18

-- Problem statement
theorem slower_train_speed : v_s = 36 :=
by
  let rel_speed := km_per_hr_to_m_per_s (v_f - v_s)
  have rel_speed_def : rel_speed = (42 - v_s) * 5 / 18 := by sorry
  have distance : 60 = rel_speed * t := by sorry
  have equation : 60 = (42 - v_s) * 10 := by sorry
  have solve_v_s : v_s = 36 := by sorry
  exact solve_v_s

end slower_train_speed_l379_37982


namespace range_of_k_for_real_roots_l379_37922

theorem range_of_k_for_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by 
  sorry

end range_of_k_for_real_roots_l379_37922


namespace sales_tax_difference_l379_37994

theorem sales_tax_difference (P : ℝ) (d t1 t2 : ℝ) :
  let discounted_price := P * (1 - d)
  let total_cost1 := discounted_price * (1 + t1)
  let total_cost2 := discounted_price * (1 + t2)
  t1 = 0.08 ∧ t2 = 0.075 ∧ P = 50 ∧ d = 0.05 →
  abs ((total_cost1 - total_cost2) - 0.24) < 0.01 :=
by
  sorry

end sales_tax_difference_l379_37994


namespace paco_initial_cookies_l379_37947

theorem paco_initial_cookies :
  ∀ (total_cookies initially_ate initially_gave : ℕ),
    initially_ate = 14 →
    initially_gave = 13 →
    initially_ate = initially_gave + 1 →
    total_cookies = initially_ate + initially_gave →
    total_cookies = 27 :=
by
  intros total_cookies initially_ate initially_gave h_ate h_gave h_diff h_sum
  sorry

end paco_initial_cookies_l379_37947


namespace find_PO_l379_37942

variables {P : ℝ × ℝ} {O F : ℝ × ℝ}

def on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1
def origin (O : ℝ × ℝ) : Prop := O = (0, 0)
def focus (F : ℝ × ℝ) : Prop := F = (1, 0)
def isosceles_triangle (O P F : ℝ × ℝ) : Prop :=
  dist O P = dist O F ∨ dist O P = dist P F

theorem find_PO
  (P : ℝ × ℝ) (O : ℝ × ℝ) (F : ℝ × ℝ)
  (hO : origin O) (hF : focus F) (hP : on_parabola P) (h_iso : isosceles_triangle O P F) :
  dist O P = 1 ∨ dist O P = 3 / 2 :=
sorry

end find_PO_l379_37942


namespace books_per_shelf_l379_37984

theorem books_per_shelf 
  (initial_books : ℕ) 
  (sold_books : ℕ) 
  (num_shelves : ℕ) 
  (remaining_books : ℕ := initial_books - sold_books) :
  initial_books = 40 → sold_books = 20 → num_shelves = 5 → remaining_books / num_shelves = 4 :=
by
  sorry

end books_per_shelf_l379_37984


namespace valid_outfits_count_l379_37992

noncomputable def number_of_valid_outfits (shirt_count: ℕ) (pant_colors: List String) (hat_count: ℕ) : ℕ :=
  let total_combinations := shirt_count * (pant_colors.length) * hat_count
  let matching_outfits := List.length (List.filter (λ c => c ∈ pant_colors) ["tan", "black", "blue", "gray"])
  total_combinations - matching_outfits

theorem valid_outfits_count :
    number_of_valid_outfits 8 ["tan", "black", "blue", "gray"] 8 = 252 := by
  sorry

end valid_outfits_count_l379_37992


namespace cos_arcsin_l379_37912

theorem cos_arcsin (h3: ℝ) (h5: ℝ) (h_op: h3 = 3) (h_hyp: h5 = 5) : 
  Real.cos (Real.arcsin (3 / 5)) = 4 / 5 := 
sorry

end cos_arcsin_l379_37912


namespace find_p_l379_37960

theorem find_p (m n p : ℝ) 
  (h1 : m = (n / 2) - (2 / 5)) 
  (h2 : m + p = ((n + 4) / 2) - (2 / 5)) :
  p = 2 :=
sorry

end find_p_l379_37960


namespace mean_of_five_numbers_l379_37927

theorem mean_of_five_numbers (x1 x2 x3 x4 x5 : ℚ) (h_sum : x1 + x2 + x3 + x4 + x5 = 1/3) : 
  (x1 + x2 + x3 + x4 + x5) / 5 = 1/15 :=
by 
  sorry

end mean_of_five_numbers_l379_37927


namespace pear_juice_percentage_l379_37941

/--
Miki has a dozen oranges and pears. She extracts juice as follows:
5 pears -> 10 ounces of pear juice
3 oranges -> 12 ounces of orange juice
She uses 10 pears and 10 oranges to make a blend.
Prove that the percent of the blend that is pear juice is 33.33%.
-/
theorem pear_juice_percentage :
  let pear_juice_per_pear := 10 / 5
  let orange_juice_per_orange := 12 / 3
  let total_pear_juice := 10 * pear_juice_per_pear
  let total_orange_juice := 10 * orange_juice_per_orange
  let total_juice := total_pear_juice + total_orange_juice
  total_pear_juice / total_juice = 1 / 3 :=
by
  sorry

end pear_juice_percentage_l379_37941


namespace y_is_defined_iff_x_not_equal_to_10_l379_37946

def range_of_independent_variable (x : ℝ) : Prop :=
  x ≠ 10

theorem y_is_defined_iff_x_not_equal_to_10 (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 10)) ↔ range_of_independent_variable x :=
by sorry

end y_is_defined_iff_x_not_equal_to_10_l379_37946


namespace rachel_math_homework_pages_l379_37931

-- Define the number of pages of math homework and reading homework
def pagesReadingHomework : ℕ := 4

theorem rachel_math_homework_pages (M : ℕ) (h1 : M + 1 = pagesReadingHomework) : M = 3 :=
by
  sorry

end rachel_math_homework_pages_l379_37931


namespace find_result_of_adding_8_l379_37936

theorem find_result_of_adding_8 (x : ℕ) (h : 6 * x = 72) : x + 8 = 20 :=
sorry

end find_result_of_adding_8_l379_37936


namespace car_speed_in_kmh_l379_37910

theorem car_speed_in_kmh (rev_per_min : ℕ) (circumference : ℕ) (speed : ℕ) 
  (h1 : rev_per_min = 400) (h2 : circumference = 4) : speed = 96 :=
  sorry

end car_speed_in_kmh_l379_37910


namespace part1_quantity_of_vegetables_part2_functional_relationship_part3_min_vegetable_a_l379_37968

/-- Part 1: Quantities of vegetables A and B wholesaled. -/
theorem part1_quantity_of_vegetables (x y : ℝ) 
  (h1 : x + y = 40) 
  (h2 : 4.8 * x + 4 * y = 180) : 
  x = 25 ∧ y = 15 :=
sorry

/-- Part 2: Functional relationship between m and n. -/
theorem part2_functional_relationship (n m : ℝ) 
  (h : n ≤ 80) 
  (h2 : m = 4.8 * n + 4 * (80 - n)) : 
  m = 0.8 * n + 320 :=
sorry

/-- Part 3: Minimum amount of vegetable A to ensure profit of at least 176 yuan -/
theorem part3_min_vegetable_a (n : ℝ) 
  (h : 0.8 * n + 128 ≥ 176) : 
  n ≥ 60 :=
sorry

end part1_quantity_of_vegetables_part2_functional_relationship_part3_min_vegetable_a_l379_37968


namespace solve_quadratic_l379_37978

theorem solve_quadratic {x : ℝ} : x^2 = 2 * x ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_quadratic_l379_37978


namespace number_of_zeros_f_l379_37949

noncomputable def f (x : ℝ) : ℝ := (x - 2) * Real.log x

theorem number_of_zeros_f : ∃! n : ℕ, n = 2 ∧ ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 := by
  sorry

end number_of_zeros_f_l379_37949


namespace price_of_wheat_flour_l379_37928

theorem price_of_wheat_flour
  (initial_amount : ℕ)
  (price_rice : ℕ)
  (num_rice : ℕ)
  (price_soda : ℕ)
  (num_soda : ℕ)
  (num_wheat_flour : ℕ)
  (remaining_balance : ℕ)
  (total_spent : ℕ)
  (amount_spent_on_rice_and_soda : ℕ)
  (amount_spent_on_wheat_flour : ℕ)
  (price_per_packet_wheat_flour : ℕ) 
  (h_initial_amount : initial_amount = 500)
  (h_price_rice : price_rice = 20)
  (h_num_rice : num_rice = 2)
  (h_price_soda : price_soda = 150)
  (h_num_soda : num_soda = 1)
  (h_num_wheat_flour : num_wheat_flour = 3)
  (h_remaining_balance : remaining_balance = 235)
  (h_total_spent : total_spent = initial_amount - remaining_balance)
  (h_amount_spent_on_rice_and_soda : amount_spent_on_rice_and_soda = price_rice * num_rice + price_soda * num_soda)
  (h_amount_spent_on_wheat_flour : amount_spent_on_wheat_flour = total_spent - amount_spent_on_rice_and_soda)
  (h_price_per_packet_wheat_flour : price_per_packet_wheat_flour = amount_spent_on_wheat_flour / num_wheat_flour) :
  price_per_packet_wheat_flour = 25 :=
by 
  sorry

end price_of_wheat_flour_l379_37928


namespace solve_for_x_l379_37997

theorem solve_for_x (x y : ℝ) (h1 : 9^y = x^12) (h2 : y = 6) : x = 3 :=
by
  sorry

end solve_for_x_l379_37997


namespace minimum_familiar_pairs_l379_37967

theorem minimum_familiar_pairs (n : ℕ) (students : Finset (Fin n)) 
  (familiar : Finset (Fin n × Fin n))
  (h_n : n = 175)
  (h_condition : ∀ (s : Finset (Fin n)), s.card = 6 → 
    ∃ (s1 s2 : Finset (Fin n)), s1 ∪ s2 = s ∧ s1.card = 3 ∧ s2.card = 3 ∧ 
    ∀ x ∈ s1, ∀ y ∈ s1, (x ≠ y → (x, y) ∈ familiar) ∧
    ∀ x ∈ s2, ∀ y ∈ s2, (x ≠ y → (x, y) ∈ familiar)) :
  ∃ m : ℕ, m = 15050 ∧ ∀ p : ℕ, (∃ g : Finset (Fin n × Fin n), g.card = p) → p ≥ m := 
sorry

end minimum_familiar_pairs_l379_37967


namespace bananas_per_friend_l379_37938

-- Define the conditions
def total_bananas : ℕ := 40
def number_of_friends : ℕ := 40

-- Define the theorem to be proved
theorem bananas_per_friend : 
  (total_bananas / number_of_friends) = 1 :=
by
  sorry

end bananas_per_friend_l379_37938
