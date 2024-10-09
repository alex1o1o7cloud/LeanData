import Mathlib

namespace ratio_c_d_l1977_197761

theorem ratio_c_d (a b c d : ℝ) (h_eq : ∀ x, a * x^3 + b * x^2 + c * x + d = 0) 
    (h_roots : ∀ r, r = 2 ∨ r = 4 ∨ r = 5 ↔ (a * r^3 + b * r^2 + c * r + d = 0)) :
    c / d = 19 / 20 :=
by
  sorry

end ratio_c_d_l1977_197761


namespace polynomial_is_quadratic_l1977_197738

theorem polynomial_is_quadratic (m : ℤ) (h : (m - 2 ≠ 0) ∧ (|m| = 2)) : m = -2 :=
by sorry

end polynomial_is_quadratic_l1977_197738


namespace nonneg_int_solutions_to_ineq_system_l1977_197703

open Set

theorem nonneg_int_solutions_to_ineq_system :
  {x : ℤ | (5 * x - 6 ≤ 2 * (x + 3)) ∧ ((x / 4 : ℚ) - 1 < (x - 2) / 3)} = {0, 1, 2, 3, 4} :=
by
  sorry

end nonneg_int_solutions_to_ineq_system_l1977_197703


namespace geometric_series_sum_l1977_197745

theorem geometric_series_sum : 
  let a := 1 
  let r := 2 
  let n := 11 
  let S_n := (a * (1 - r^n)) / (1 - r)
  S_n = 2047 := by
  -- The proof steps would normally go here.
  sorry

end geometric_series_sum_l1977_197745


namespace length_of_box_l1977_197739

theorem length_of_box (v : ℝ) (w : ℝ) (h : ℝ) (l : ℝ) (conversion_factor : ℝ) (v_gallons : ℝ)
  (h_inch : ℝ) (conversion_inches_feet : ℝ) :
  v_gallons / conversion_factor = v → 
  h_inch / conversion_inches_feet = h →
  v = l * w * h →
  w = 25 →
  v_gallons = 4687.5 →
  conversion_factor = 7.5 →
  h_inch = 6 →
  conversion_inches_feet = 12 →
  l = 50 :=
by
  sorry

end length_of_box_l1977_197739


namespace find_l_find_C3_l1977_197746

-- Circle definitions
def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x - 2*y + 7 = 0

-- Given line passes through common points of C1 and C2
theorem find_l (x y : ℝ) (h1 : C1 x y) (h2 : C2 x y) : x = 1 := by
  sorry

-- Circle C3 passes through intersection points of C1 and C2, and its center lies on y = x
def C3 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1
def on_line_y_eq_x (x y : ℝ) : Prop := y = x

theorem find_C3 (x y : ℝ) (hx : C3 x y) (hy : on_line_y_eq_x x y) : (x - 1)^2 + (y - 1)^2 = 1 := by
  sorry

end find_l_find_C3_l1977_197746


namespace compute_xy_l1977_197723

variable (x y : ℝ)
variable (h1 : x - y = 6)
variable (h2 : x^3 - y^3 = 108)

theorem compute_xy : x * y = 0 := by
  sorry

end compute_xy_l1977_197723


namespace black_ball_on_second_draw_given_white_ball_on_first_draw_l1977_197770

def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 5
def total_balls : ℕ := num_white_balls + num_black_balls

def P_A : ℚ := num_white_balls / total_balls
def P_AB : ℚ := (num_white_balls * num_black_balls) / (total_balls * (total_balls - 1))
def P_B_given_A : ℚ := P_AB / P_A

theorem black_ball_on_second_draw_given_white_ball_on_first_draw : P_B_given_A = 5 / 8 :=
by
  sorry

end black_ball_on_second_draw_given_white_ball_on_first_draw_l1977_197770


namespace f_inequality_l1977_197755

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem f_inequality (x : ℝ) : f (3^x) ≥ f (2^x) := 
by 
  sorry

end f_inequality_l1977_197755


namespace focus_of_hyperbola_l1977_197757

theorem focus_of_hyperbola (m : ℝ) :
  let focus_parabola := (0, 4)
  let focus_hyperbola_upper := (0, 4)
  ∃ focus_parabola, ∃ focus_hyperbola_upper, 
    (focus_parabola = (0, 4)) ∧ (focus_hyperbola_upper = (0, 4)) ∧ 
    (3 + m = 16) → m = 13 :=
by
  sorry

end focus_of_hyperbola_l1977_197757


namespace xy_condition_l1977_197749

variable (x y : ℝ) -- This depends on the problem context specifying real numbers.

theorem xy_condition (h : x ≠ 0 ∧ y ≠ 0) : (x + y = 0 ↔ y / x + x / y = -2) :=
  sorry

end xy_condition_l1977_197749


namespace chess_games_l1977_197753

theorem chess_games (n : ℕ) (total_games : ℕ) (players : ℕ) (games_per_player : ℕ)
  (h1 : players = 9)
  (h2 : total_games = 36)
  (h3 : ∀ i : ℕ, i < players → games_per_player = players - 1)
  (h4 : 2 * total_games = players * games_per_player) :
  games_per_player = 1 :=
by
  rw [h1, h2] at h4
  sorry

end chess_games_l1977_197753


namespace largest_angle_in_pentagon_l1977_197748

-- Define the angles and sum condition
variables (x : ℝ) {P Q R S T : ℝ}

-- Conditions
def angle_P : P = 90 := sorry
def angle_Q : Q = 70 := sorry
def angle_R : R = x := sorry
def angle_S : S = x := sorry
def angle_T : T = 2*x + 20 := sorry
def sum_of_angles : P + Q + R + S + T = 540 := sorry

-- Prove the largest angle
theorem largest_angle_in_pentagon (hP : P = 90) (hQ : Q = 70)
    (hR : R = x) (hS : S = x) (hT : T = 2*x + 20) 
    (h_sum : P + Q + R + S + T = 540) : T = 200 :=
by
  sorry

end largest_angle_in_pentagon_l1977_197748


namespace region_area_l1977_197778

theorem region_area (x y : ℝ) : (x^2 + y^2 + 6*x - 4*y - 11 = 0) → (∃ (A : ℝ), A = 24 * Real.pi) :=
by
  sorry

end region_area_l1977_197778


namespace largest_prime_number_largest_composite_number_l1977_197730

-- Definitions of prime and composite
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

-- Largest prime and composite numbers less than 20
def largest_prime_less_than_20 := 19
def largest_composite_less_than_20 := 18

theorem largest_prime_number : 
  largest_prime_less_than_20 = 19 ∧ is_prime 19 ∧ 
  (∀ n : ℕ, n < 20 → is_prime n → n < 19) := 
by sorry

theorem largest_composite_number : 
  largest_composite_less_than_20 = 18 ∧ is_composite 18 ∧ 
  (∀ n : ℕ, n < 20 → is_composite n → n < 18) := 
by sorry

end largest_prime_number_largest_composite_number_l1977_197730


namespace smallest_b_for_quadratic_inequality_l1977_197714

theorem smallest_b_for_quadratic_inequality : 
  ∃ b : ℝ, (b^2 - 16 * b + 63 ≤ 0) ∧ ∀ b' : ℝ, (b'^2 - 16 * b' + 63 ≤ 0) → b ≤ b' := sorry

end smallest_b_for_quadratic_inequality_l1977_197714


namespace circumference_of_smaller_circle_l1977_197779

theorem circumference_of_smaller_circle (r R : ℝ)
  (h1 : 4 * R^2 = 784) 
  (h2 : R = (7/3) * r) :
  2 * Real.pi * r = 12 * Real.pi := 
by {
  sorry
}

end circumference_of_smaller_circle_l1977_197779


namespace side_length_of_largest_square_l1977_197759

theorem side_length_of_largest_square (S : ℝ) 
  (h1 : 2 * (S / 2)^2 + 2 * (S / 4)^2 = 810) : S = 36 :=
by
  -- proof steps go here
  sorry

end side_length_of_largest_square_l1977_197759


namespace find_largest_n_l1977_197744

theorem find_largest_n : ∃ n x y z : ℕ, n > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0 
  ∧ n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 3*x + 3*y + 3*z - 6
  ∧ (∀ m x' y' z' : ℕ, m > n → x' > 0 → y' > 0 → z' > 0 
  → m^2 ≠ x'^2 + y'^2 + z'^2 + 2*x'*y' + 2*y'*z' + 2*z'*x' + 3*x' + 3*y' + 3*z' - 6) :=
sorry

end find_largest_n_l1977_197744


namespace horner_multiplications_additions_l1977_197702

-- Define the polynomial
def f (x : ℤ) : ℤ := x^7 + 2 * x^5 + 3 * x^4 + 4 * x^3 + 5 * x^2 + 6 * x + 7

-- Define the number of multiplications and additions required by Horner's method
def horner_method_mults (n : ℕ) : ℕ := n
def horner_method_adds (n : ℕ) : ℕ := n - 1

-- Define the value of x
def x : ℤ := 3

-- Define the degree of the polynomial
def degree_of_polynomial : ℕ := 7

-- Define the statements for the proof
theorem horner_multiplications_additions :
  horner_method_mults degree_of_polynomial = 7 ∧
  horner_method_adds degree_of_polynomial = 6 :=
by
  sorry

end horner_multiplications_additions_l1977_197702


namespace shaded_area_concentric_circles_l1977_197799

theorem shaded_area_concentric_circles (R : ℝ) (r : ℝ) (hR : π * R^2 = 100 * π) (hr : r = R / 2) :
  (1 / 2) * π * R^2 + (1 / 2) * π * r^2 = 62.5 * π :=
by
  -- Given conditions
  have R10 : R = 10 := sorry  -- Derived from hR
  have r5 : r = 5 := sorry    -- Derived from hr and R10
  -- Proof steps likely skipped
  sorry

end shaded_area_concentric_circles_l1977_197799


namespace negation_example_l1977_197722

theorem negation_example :
  (¬ (∀ x : ℝ, x^2 - 2 * x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) :=
sorry

end negation_example_l1977_197722


namespace find_sale_month_4_l1977_197780

-- Definitions based on the given conditions
def avg_sale_per_month : ℕ := 6500
def num_months : ℕ := 6
def sale_month_1 : ℕ := 6435
def sale_month_2 : ℕ := 6927
def sale_month_3 : ℕ := 6855
def sale_month_5 : ℕ := 6562
def sale_month_6 : ℕ := 4991

theorem find_sale_month_4 : 
  (avg_sale_per_month * num_months) - (sale_month_1 + sale_month_2 + sale_month_3 + sale_month_5 + sale_month_6) = 7230 :=
by
  -- The proof will be provided below
  sorry

end find_sale_month_4_l1977_197780


namespace flavoring_ratio_comparison_l1977_197740

theorem flavoring_ratio_comparison (f_st cs_st w_st : ℕ) (f_sp cs_sp w_sp : ℕ) :
  f_st = 1 → cs_st = 12 → w_st = 30 →
  w_sp = 75 → cs_sp = 5 →
  f_sp / w_sp = f_st / (2 * w_st) →
  (f_st / cs_st) * 3 = f_sp / cs_sp :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end flavoring_ratio_comparison_l1977_197740


namespace math_club_partition_l1977_197704

def is_played (team : Finset ℕ) (A B C : ℕ) : Bool :=
(A ∈ team ∧ B ∉ team ∧ C ∉ team) ∨ 
(A ∉ team ∧ B ∈ team ∧ C ∉ team) ∨ 
(A ∉ team ∧ B ∉ team ∧ C ∈ team) ∨ 
(A ∈ team ∧ B ∈ team ∧ C ∈ team)

theorem math_club_partition 
  (students : Finset ℕ) (A B C : ℕ) 
  (h_size : students.card = 24)
  (teams : List (Finset ℕ))
  (h_teams : teams.length = 4)
  (h_team_size : ∀ t ∈ teams, t.card = 6)
  (h_partition : ∀ t ∈ teams, t ⊆ students) :
  ∃ (teams_played : List (Finset ℕ)), teams_played.length = 1 ∨ teams_played.length = 3 :=
sorry

end math_club_partition_l1977_197704


namespace smallest_positive_multiple_of_45_l1977_197742

def is_positive_multiple_of (n m : ℕ) : Prop :=
  ∃ x : ℕ+, m = n * x

theorem smallest_positive_multiple_of_45 : ∃ n : ℕ+, is_positive_multiple_of 45 n ∧ (∀ m : ℕ+, is_positive_multiple_of 45 m → n ≤ m) ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l1977_197742


namespace solve_eq_l1977_197709

theorem solve_eq : ∀ x : ℝ, -2 * (x - 1) = 4 → x = -1 := 
by
  intro x
  intro h
  sorry

end solve_eq_l1977_197709


namespace total_red_cards_l1977_197792

def num_standard_decks : ℕ := 3
def num_special_decks : ℕ := 2
def num_custom_decks : ℕ := 2
def red_cards_standard_deck : ℕ := 26
def red_cards_special_deck : ℕ := 30
def red_cards_custom_deck : ℕ := 20

theorem total_red_cards : num_standard_decks * red_cards_standard_deck +
                          num_special_decks * red_cards_special_deck +
                          num_custom_decks * red_cards_custom_deck = 178 :=
by
  -- Calculation omitted
  sorry

end total_red_cards_l1977_197792


namespace sum_of_angles_l1977_197787

theorem sum_of_angles (p q r s t u v w x y : ℝ)
  (H1 : p + r + t + v + x = 360)
  (H2 : q + s + u + w + y = 360) :
  p + q + r + s + t + u + v + w + x + y = 720 := 
by sorry

end sum_of_angles_l1977_197787


namespace valid_reasonings_l1977_197737

-- Define the conditions as hypotheses
def analogical_reasoning (R1 : Prop) : Prop := R1
def inductive_reasoning (R2 R4 : Prop) : Prop := R2 ∧ R4
def invalid_generalization (R3 : Prop) : Prop := ¬R3

-- Given the conditions, prove that the valid reasonings are (1), (2), and (4)
theorem valid_reasonings
  (R1 : Prop) (R2 : Prop) (R3 : Prop) (R4 : Prop)
  (h1 : analogical_reasoning R1) 
  (h2 : inductive_reasoning R2 R4) 
  (h3 : invalid_generalization R3) : 
  R1 ∧ R2 ∧ R4 :=
by 
  sorry

end valid_reasonings_l1977_197737


namespace curve_crosses_itself_at_point_l1977_197768

theorem curve_crosses_itself_at_point :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
  (2 * t₁^2 + 1 = 2 * t₂^2 + 1) ∧ 
  (2 * t₁^3 - 6 * t₁^2 + 8 = 2 * t₂^3 - 6 * t₂^2 + 8) ∧ 
  2 * t₁^2 + 1 = 1 ∧ 2 * t₁^3 - 6 * t₁^2 + 8 = 8 :=
by
  sorry

end curve_crosses_itself_at_point_l1977_197768


namespace sum_of_two_integers_l1977_197734

theorem sum_of_two_integers (a b : ℕ) (h1 : a * b + a + b = 113) (h2 : Nat.gcd a b = 1) (h3 : a < 25) (h4 : b < 25) : a + b = 23 := by
  sorry

end sum_of_two_integers_l1977_197734


namespace probability_is_stable_frequency_l1977_197793

/-- Definition of probability: the stable theoretical value reflecting the likelihood of event occurrence. -/
def probability (event : Type) : ℝ := sorry 

/-- Definition of frequency: the empirical count of how often an event occurs in repeated experiments. -/
def frequency (event : Type) (trials : ℕ) : ℝ := sorry 

/-- The statement that "probability is the stable value of frequency" is correct. -/
theorem probability_is_stable_frequency (event : Type) (trials : ℕ) :
  probability event = sorry ↔ true := 
by 
  -- This is where the proof would go, but is replaced with sorry for now. 
  sorry

end probability_is_stable_frequency_l1977_197793


namespace total_number_of_squares_up_to_50th_ring_l1977_197733

def number_of_squares_up_to_50th_ring : Nat :=
  let central_square := 1
  let sum_rings := (50 * (50 + 1)) * 4  -- Using the formula for arithmetic series sum where a = 8 and d = 8 and n = 50
  central_square + sum_rings

theorem total_number_of_squares_up_to_50th_ring : number_of_squares_up_to_50th_ring = 10201 :=
  by  -- This statement means we believe the theorem is true and will be proven.
    sorry                                                      -- Proof omitted, will need to fill this in later

end total_number_of_squares_up_to_50th_ring_l1977_197733


namespace triangle_area_l1977_197798

theorem triangle_area (a b c : ℝ) (C : ℝ) (h1 : c^2 = (a - b)^2 + 6) (h2 : C = π / 3) :
    abs ((1 / 2) * a * b * Real.sin C) = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end triangle_area_l1977_197798


namespace current_selling_price_is_correct_profit_per_unit_is_correct_l1977_197766

variable (a : ℝ)

def original_selling_price (a : ℝ) : ℝ :=
  a * 1.22

def current_selling_price (a : ℝ) : ℝ :=
  original_selling_price a * 0.85

def profit_per_unit (a : ℝ) : ℝ :=
  current_selling_price a - a

theorem current_selling_price_is_correct : current_selling_price a = 1.037 * a :=
by
  unfold current_selling_price original_selling_price
  sorry

theorem profit_per_unit_is_correct : profit_per_unit a = 0.037 * a :=
by
  unfold profit_per_unit current_selling_price original_selling_price
  sorry

end current_selling_price_is_correct_profit_per_unit_is_correct_l1977_197766


namespace factor_of_polynomial_l1977_197762

theorem factor_of_polynomial :
  (x^4 + 4 * x^2 + 16) % (x^2 + 4) = 0 :=
sorry

end factor_of_polynomial_l1977_197762


namespace problem1_problem2_l1977_197712

-- Problem 1
theorem problem1 : (2 * Real.sqrt 12 - 3 * Real.sqrt (1 / 3)) * Real.sqrt 6 = 9 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h1 : x / (2 * x - 1) = 2 - 3 / (1 - 2 * x)) : x = -1 / 3 := by
  sorry

end problem1_problem2_l1977_197712


namespace quotient_of_division_l1977_197708

theorem quotient_of_division (dividend divisor remainder quotient : ℕ) 
  (h_dividend : dividend = 271) (h_divisor : divisor = 30) 
  (h_remainder : remainder = 1) (h_division : dividend = divisor * quotient + remainder) : 
  quotient = 9 := 
by 
  sorry

end quotient_of_division_l1977_197708


namespace lower_bound_for_expression_l1977_197705

theorem lower_bound_for_expression :
  ∃ L: ℤ, (∀ n: ℤ, L < 4 * n + 7 ∧ 4 * n + 7 < 120) → L = 5 :=
sorry

end lower_bound_for_expression_l1977_197705


namespace bamboo_node_volume_5_l1977_197795

theorem bamboo_node_volume_5 {a_1 d : ℚ} :
  (a_1 + (a_1 + d) + (a_1 + 2 * d) + (a_1 + 3 * d) = 3) →
  ((a_1 + 6 * d) + (a_1 + 7 * d) + (a_1 + 8 * d) = 4) →
  (a_1 + 4 * d = 67 / 66) :=
by sorry

end bamboo_node_volume_5_l1977_197795


namespace quadratic_inequality_l1977_197719

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_l1977_197719


namespace correct_statements_l1977_197760

-- Define the statements
def statement_1 := true
def statement_2 := false
def statement_3 := true
def statement_4 := true

-- Define a function to count the number of true statements
def num_correct_statements (s1 s2 s3 s4 : Bool) : Nat :=
  [s1, s2, s3, s4].countP id

-- Define the theorem to prove that the number of correct statements is 3
theorem correct_statements :
  num_correct_statements statement_1 statement_2 statement_3 statement_4 = 3 :=
by
  -- You can use sorry to skip the proof
  sorry

end correct_statements_l1977_197760


namespace max_students_in_auditorium_l1977_197715

def increment (i : ℕ) : ℕ :=
  (i * (i + 1)) / 2

def seats_in_row (i : ℕ) : ℕ :=
  10 + increment i

def max_students_in_row (n : ℕ) : ℕ :=
  (n + 1) / 2

def total_max_students_up_to_row (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => max_students_in_row (seats_in_row (i + 1)))

theorem max_students_in_auditorium : total_max_students_up_to_row 20 = 335 := 
sorry

end max_students_in_auditorium_l1977_197715


namespace convex_functions_exist_l1977_197783

noncomputable def exponential_function (x : ℝ) : ℝ :=
  4 - 5 * (1 / 2) ^ x

noncomputable def inverse_tangent_function (x : ℝ) : ℝ :=
  (10 / Real.pi) * Real.arctan x - 1

theorem convex_functions_exist :
  ∃ (f1 f2 : ℝ → ℝ),
    (∀ x, 0 < x → f1 x = exponential_function x) ∧
    (∀ x, 0 < x → f2 x = inverse_tangent_function x) ∧
    (∀ x, 0 < x → f1 x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ x, 0 < x → f2 x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 →
      f1 x1 + f1 x2 < 2 * f1 ((x1 + x2) / 2)) ∧
    (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 →
      f2 x1 + f2 x2 < 2 * f2 ((x1 + x2) / 2)) :=
sorry

end convex_functions_exist_l1977_197783


namespace segment_length_l1977_197773

theorem segment_length (x y : ℝ) (A B : ℝ × ℝ) 
  (h1 : A.2^2 = 4 * A.1) 
  (h2 : B.2^2 = 4 * B.1) 
  (h3 : A.2 = 2 * A.1 - 2)
  (h4 : B.2 = 2 * B.1 - 2)
  (h5 : A ≠ B) :
  dist A B = 5 :=
sorry

end segment_length_l1977_197773


namespace no_solution_l1977_197797

theorem no_solution : ∀ x : ℝ, ¬ (3 * x + 2 < (x + 2)^2 ∧ (x + 2)^2 < 5 * x + 1) :=
by
  intro x
  -- Solve each part of the inequality
  have h1 : ¬ (3 * x + 2 < (x + 2)^2) ↔ x^2 + x + 2 ≤ 0 := by sorry
  have h2 : ¬ ((x + 2)^2 < 5 * x + 1) ↔ x^2 - x + 3 ≥ 0 := by sorry
  -- Combine the results
  exact sorry

end no_solution_l1977_197797


namespace extremal_values_d_l1977_197794

theorem extremal_values_d (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ → Prop)
  (hC : ∀ (x y : ℝ), C (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 1)
  (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : ∃ (x y : ℝ), C (x, y)) :
  ∃ (max_d min_d : ℝ), max_d = 14 ∧ min_d = 10 :=
by
  -- Necessary assumptions
  have h₁ : ∀ (x y : ℝ), C (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 1 := hC
  have h₂ : A = (-1, 0) := hA
  have h₃ : B = (1, 0) := hB
  have h₄ : ∃ (x y : ℝ), C (x, y) := hP
  sorry

end extremal_values_d_l1977_197794


namespace num_factors_34848_l1977_197769

/-- Define the number 34848 and its prime factorization -/
def n : ℕ := 34848
def p_factors : List (ℕ × ℕ) := [(2, 5), (3, 2), (11, 2)]

/-- Helper function to calculate the number of divisors from prime factors -/
def num_divisors (factors : List (ℕ × ℕ)) : ℕ := 
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.2 + 1)) 1

/-- Formal statement of the problem -/
theorem num_factors_34848 : num_divisors p_factors = 54 :=
by
  -- Proof that 34848 has the prime factorization 3^2 * 2^5 * 11^2 
  -- and that the number of factors is 54 would go here.
  sorry

end num_factors_34848_l1977_197769


namespace geometric_sequence_properties_l1977_197741

theorem geometric_sequence_properties (a : ℕ → ℝ) (n : ℕ) (q : ℝ) 
  (h_geom : ∀ (m k : ℕ), a (m + k) = a m * q ^ k) 
  (h_sum : a 1 + a n = 66) 
  (h_prod : a 3 * a (n - 2) = 128) 
  (h_s_n : (a 1 * (1 - q ^ n)) / (1 - q) = 126) : 
  n = 6 ∧ (q = 2 ∨ q = 1/2) :=
sorry

end geometric_sequence_properties_l1977_197741


namespace simplifies_to_minus_18_point_5_l1977_197772

theorem simplifies_to_minus_18_point_5 (x y : ℝ) (h_x : x = 1/2) (h_y : y = -2) :
  ((2 * x + y)^2 - (2 * x - y) * (x + y) - 2 * (x - 2 * y) * (x + 2 * y)) / y = -18.5 :=
by
  -- Let's replace x and y with their values
  -- Expand and simplify the expression
  -- Divide the expression by y
  -- Prove the final result is equal to -18.5
  sorry

end simplifies_to_minus_18_point_5_l1977_197772


namespace rachels_milk_consumption_l1977_197796

theorem rachels_milk_consumption :
  let bottle1 := (3 / 8 : ℚ)
  let bottle2 := (1 / 4 : ℚ)
  let total_milk := bottle1 + bottle2
  let rachel_ratio := (3 / 4 : ℚ)
  rachel_ratio * total_milk = (15 / 32 : ℚ) :=
by
  let bottle1 := (3 / 8 : ℚ)
  let bottle2 := (1 / 4 : ℚ)
  let total_milk := bottle1 + bottle2
  let rachel_ratio := (3 / 4 : ℚ)
  -- proof placeholder
  sorry

end rachels_milk_consumption_l1977_197796


namespace snow_volume_l1977_197710

theorem snow_volume
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (h_length : length = 15)
  (h_width : width = 3)
  (h_depth : depth = 0.6) :
  length * width * depth = 27 := 
by
  -- placeholder for proof
  sorry

end snow_volume_l1977_197710


namespace selling_price_40_percent_profit_l1977_197736

variable (C L : ℝ)

-- Condition: the profit earned by selling at $832 is equal to the loss incurred when selling at some price "L".
axiom eq_profit_loss : 832 - C = C - L

-- Condition: the desired profit price for a 40% profit on the cost price is $896.
axiom forty_percent_profit : 1.40 * C = 896

-- Theorem: the selling price for making a 40% profit is $896.
theorem selling_price_40_percent_profit : 1.40 * C = 896 :=
by
  sorry

end selling_price_40_percent_profit_l1977_197736


namespace tricycle_wheels_l1977_197750

theorem tricycle_wheels (T : ℕ) 
  (h1 : 3 * 2 = 6) 
  (h2 : 7 * 1 = 7) 
  (h3 : 6 + 7 + 4 * T = 25) : T = 3 :=
sorry

end tricycle_wheels_l1977_197750


namespace charles_housesitting_hours_l1977_197731

theorem charles_housesitting_hours :
  ∀ (earnings_per_hour_housesitting earnings_per_hour_walking_dog number_of_dogs_walked total_earnings : ℕ),
  earnings_per_hour_housesitting = 15 →
  earnings_per_hour_walking_dog = 22 →
  number_of_dogs_walked = 3 →
  total_earnings = 216 →
  ∃ h : ℕ, 15 * h + 22 * 3 = 216 ∧ h = 10 :=
by
  intros
  sorry

end charles_housesitting_hours_l1977_197731


namespace solve_fra_eq_l1977_197786

theorem solve_fra_eq : ∀ x : ℝ, (x - 2) / (x + 2) + 4 / (x^2 - 4) = 1 → x = 3 :=
by 
  -- Proof steps go here
  sorry

end solve_fra_eq_l1977_197786


namespace total_balloons_after_gift_l1977_197763

-- Definitions for conditions
def initial_balloons := 26
def additional_balloons := 34

-- Proposition for the total number of balloons
theorem total_balloons_after_gift : initial_balloons + additional_balloons = 60 := 
by
  -- Proof omitted, adding sorry
  sorry

end total_balloons_after_gift_l1977_197763


namespace n_cubed_plus_5_div_by_6_l1977_197720

theorem n_cubed_plus_5_div_by_6  (n : ℤ) : 6 ∣ n * (n^2 + 5) :=
sorry

end n_cubed_plus_5_div_by_6_l1977_197720


namespace sqrt_domain_l1977_197713

theorem sqrt_domain (x : ℝ) : (∃ y, y * y = x - 2) ↔ (x ≥ 2) :=
by sorry

end sqrt_domain_l1977_197713


namespace profit_ratio_l1977_197743

def praveen_initial_capital : ℝ := 3500
def hari_initial_capital : ℝ := 9000.000000000002
def total_months : ℕ := 12
def months_hari_invested : ℕ := total_months - 5

def effective_capital (initial_capital : ℝ) (months : ℕ) : ℝ :=
  initial_capital * months

theorem profit_ratio :
  effective_capital praveen_initial_capital total_months / effective_capital hari_initial_capital months_hari_invested 
  = 2 / 3 :=
by
  sorry

end profit_ratio_l1977_197743


namespace right_triangle_hypotenuse_l1977_197721

def is_nat (n : ℕ) : Prop := n > 0

theorem right_triangle_hypotenuse (x : ℕ) (x_pos : is_nat x) (consec : x + 1 > x) (h : 11^2 + x^2 = (x + 1)^2) : x + 1 = 61 :=
by
  sorry

end right_triangle_hypotenuse_l1977_197721


namespace sum_of_fractions_bounds_l1977_197717

theorem sum_of_fractions_bounds (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum_numerators : a + c = 1000) (h_sum_denominators : b + d = 1000) :
  (999 / 969 + 1 / 31) ≤ (a / b + c / d) ∧ (a / b + c / d) ≤ (999 + 1 / 999) :=
by
  sorry

end sum_of_fractions_bounds_l1977_197717


namespace product_of_reverse_numbers_l1977_197711

def reverse (n : Nat) : Nat :=
  Nat.ofDigits 10 (List.reverse (Nat.digits 10 n))

theorem product_of_reverse_numbers : 
  ∃ (a b : ℕ), a * b = 92565 ∧ b = reverse a ∧ ((a = 165 ∧ b = 561) ∨ (a = 561 ∧ b = 165)) :=
by
  sorry

end product_of_reverse_numbers_l1977_197711


namespace find_function_solution_l1977_197725

noncomputable def function_solution (f : ℤ → ℤ) : Prop :=
∀ x y : ℤ, x ≠ 0 → x * f (2 * f y - x) + y^2 * f (2 * x - f y) = f x ^ 2 / x + f (y * f y)

theorem find_function_solution : 
  ∀ f : ℤ → ℤ, function_solution f → (∀ x : ℤ, f x = 0) ∨ (∀ x : ℤ, f x = x^2) :=
sorry

end find_function_solution_l1977_197725


namespace possible_division_l1977_197706

theorem possible_division (side_length : ℕ) (areas : Fin 5 → Set (Fin side_length × Fin side_length))
  (h1 : side_length = 5)
  (h2 : ∀ i, ∃ cells : Finset (Fin side_length × Fin side_length), areas i = cells ∧ Finset.card cells = 5)
  (h3 : ∀ i j, i ≠ j → Disjoint (areas i) (areas j))
  (total_cut_length : ℕ)
  (h4 : total_cut_length ≤ 16) :
  
  ∃ cuts : Finset (Fin side_length × Fin side_length) × Finset (Fin side_length × Fin side_length),
    total_cut_length = (cuts.1.card + cuts.2.card) :=
sorry

end possible_division_l1977_197706


namespace exists_nat_expressed_as_sum_of_powers_l1977_197774

theorem exists_nat_expressed_as_sum_of_powers 
  (P : Finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) :
  ∃ x : ℕ, (∀ p ∈ P, ∃ a b : ℕ, x = a^p + b^p) ∧ (∀ p : ℕ, Nat.Prime p → p ∉ P → ¬∃ a b : ℕ, x = a^p + b^p) :=
by
  let x := 2^(P.val.prod + 1)
  use x
  sorry

end exists_nat_expressed_as_sum_of_powers_l1977_197774


namespace compute_expression_l1977_197784

theorem compute_expression :
  ( ((15 ^ 15) / (15 ^ 10)) ^ 3 * 5 ^ 6 ) / (25 ^ 2) = 3 ^ 15 * 5 ^ 17 :=
by
  -- We'll use sorry here as proof is not required
  sorry

end compute_expression_l1977_197784


namespace correct_inequality_l1977_197735

theorem correct_inequality :
  (1 / 2)^(2 / 3) < (1 / 2)^(1 / 3) ∧ (1 / 2)^(1 / 3) < 1 :=
by sorry

end correct_inequality_l1977_197735


namespace union_complement_B_A_equals_a_values_l1977_197782

namespace ProofProblem

-- Define the universal set R as real numbers
def R := Set ℝ

-- Define set A and set B as per the conditions
def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

-- Complement of B in R
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}

-- Union of complement of B with A
def union_complement_B_A : Set ℝ := complement_B ∪ A

-- The first statement to be proven
theorem union_complement_B_A_equals : 
  union_complement_B_A = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
by
  sorry

-- Define set C as per the conditions
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- The second statement to be proven
theorem a_values (a : ℝ) (h : C a ⊆ B) : 
  2 ≤ a ∧ a ≤ 8 :=
by
  sorry

end ProofProblem

end union_complement_B_A_equals_a_values_l1977_197782


namespace inequality_positive_reals_l1977_197767

theorem inequality_positive_reals (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧ 
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ (3 * Real.sqrt 2 / 2) :=
sorry

end inequality_positive_reals_l1977_197767


namespace increasing_exponential_function_range_l1977_197756

theorem increasing_exponential_function_range (a : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ (x : ℝ), f x = a ^ x) 
    (h2 : a > 0)
    (h3 : a ≠ 1)
    (h4 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) : a > 1 := 
sorry

end increasing_exponential_function_range_l1977_197756


namespace part1_even_function_part2_min_value_l1977_197724

variable {a x : ℝ}

def f (x a : ℝ) : ℝ := x^2 + |x - a| + 1

theorem part1_even_function (h : a = 0) : 
  ∀ x : ℝ, f x 0 = f (-x) 0 :=
by
  -- This statement needs to be proved to show that f(x) is even when a = 0
  sorry

theorem part2_min_value (h : true) : 
  (a > (1/2) → ∃ x : ℝ, f x a = a + (3/4)) ∧
  (a ≤ -(1/2) → ∃ x : ℝ, f x a = -a + (3/4)) ∧
  ((- (1/2) < a ∧ a ≤ (1/2)) → ∃ x : ℝ, f x a = a^2 + 1) :=
by
  -- This statement needs to be proved to show the different minimum values of the function
  sorry

end part1_even_function_part2_min_value_l1977_197724


namespace mary_rental_hours_l1977_197747

def ocean_bike_fixed_fee := 17
def ocean_bike_hourly_rate := 7
def total_paid := 80

def calculate_hours (fixed_fee : Nat) (hourly_rate : Nat) (total_amount : Nat) : Nat :=
  (total_amount - fixed_fee) / hourly_rate

theorem mary_rental_hours :
  calculate_hours ocean_bike_fixed_fee ocean_bike_hourly_rate total_paid = 9 :=
by
  sorry

end mary_rental_hours_l1977_197747


namespace find_f2_l1977_197788

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry
noncomputable def a : ℝ := sorry

axiom odd_f : ∀ x, f (-x) = -f x
axiom even_g : ∀ x, g (-x) = g x
axiom fg_eq : ∀ x, f x + g x = a^x - a^(-x) + 2
axiom g2_a : g 2 = a
axiom a_pos : a > 0
axiom a_ne1 : a ≠ 1

theorem find_f2 : f 2 = 15 / 4 := 
by sorry

end find_f2_l1977_197788


namespace ratio_of_expenditures_l1977_197726

theorem ratio_of_expenditures 
  (income_Uma : ℕ) (income_Bala : ℕ) (expenditure_Uma : ℕ) (expenditure_Bala : ℕ)
  (h_ratio_incomes : income_Uma / income_Bala = 4 / 3)
  (h_savings_Uma : income_Uma - expenditure_Uma = 5000)
  (h_savings_Bala : income_Bala - expenditure_Bala = 5000)
  (h_income_Uma : income_Uma = 20000) :
  expenditure_Uma / expenditure_Bala = 3 / 2 :=
sorry

end ratio_of_expenditures_l1977_197726


namespace inequality_solution_l1977_197727

theorem inequality_solution (x : ℝ) : 
  (x / (x + 5) ≥ 0) ↔ (x ∈ (Set.Iio (-5)).union (Set.Ici 0)) :=
by
  sorry

end inequality_solution_l1977_197727


namespace rehabilitation_centers_total_l1977_197777

noncomputable def jane_visits (han_visits : ℕ) : ℕ := 2 * han_visits + 6
noncomputable def han_visits (jude_visits : ℕ) : ℕ := 2 * jude_visits - 2
noncomputable def jude_visits (lisa_visits : ℕ) : ℕ := lisa_visits / 2
def lisa_visits : ℕ := 6

def total_visits (jane_visits han_visits jude_visits lisa_visits : ℕ) : ℕ :=
  jane_visits + han_visits + jude_visits + lisa_visits

theorem rehabilitation_centers_total :
  total_visits (jane_visits (han_visits (jude_visits lisa_visits))) 
               (han_visits (jude_visits lisa_visits))
               (jude_visits lisa_visits) 
               lisa_visits = 27 :=
by
  sorry

end rehabilitation_centers_total_l1977_197777


namespace carbonate_weight_l1977_197758

namespace MolecularWeight

def molecular_weight_Al2_CO3_3 : ℝ := 234
def molecular_weight_Al : ℝ := 26.98
def num_Al_atoms : ℕ := 2

theorem carbonate_weight :
  molecular_weight_Al2_CO3_3 - (num_Al_atoms * molecular_weight_Al) = 180.04 :=
sorry

end MolecularWeight

end carbonate_weight_l1977_197758


namespace gcd_lcm_problem_l1977_197707

theorem gcd_lcm_problem (b : ℤ) (x : ℕ) (hx_pos : 0 < x) (hx : x = 12) :
  gcd 30 b = x + 3 ∧ lcm 30 b = x * (x + 3) → b = 90 := 
by
  sorry

end gcd_lcm_problem_l1977_197707


namespace triangles_congruence_l1977_197775

theorem triangles_congruence (A_1 B_1 C_1 A_2 B_2 C_2 : ℝ)
  (angle_A1 angle_B1 angle_C1 angle_A2 angle_B2 angle_C2 : ℝ)
  (h_side1 : A_1 = A_2) 
  (h_side2 : B_1 = B_2)
  (h_angle1 : angle_A1 = angle_A2)
  (h_angle2 : angle_B1 = angle_B2)
  (h_angle3 : angle_C1 = angle_C2) : 
  ¬((A_1 = C_1) ∧ (B_1 = C_2) ∧ (angle_A1 = angle_B2) ∧ (angle_B1 = angle_A2) ∧ (angle_C1 = angle_B2) → 
     (A_1 = A_2) ∧ (B_1 = B_2) ∧ (C_1 = C_2)) :=
by {
  sorry
}

end triangles_congruence_l1977_197775


namespace find_n_l1977_197752

noncomputable def cube_probability_solid_color (num_cubes edge_length num_corner num_edge num_face_center num_center : ℕ)
  (corner_prob edge_prob face_center_prob center_prob : ℚ) : ℚ :=
  have total_corner_prob := corner_prob ^ num_corner
  have total_edge_prob := edge_prob ^ num_edge
  have total_face_center_prob := face_center_prob ^ num_face_center
  have total_center_prob := center_prob ^ num_center
  2 * (total_corner_prob * total_edge_prob * total_face_center_prob * total_center_prob)

theorem find_n : ∃ n : ℕ, cube_probability_solid_color 27 3 8 12 6 1
  (1/8) (1/4) (1/2) 1 = (1 / (2 : ℚ) ^ n) ∧ n = 53 := by
  use 53
  simp only [cube_probability_solid_color]
  sorry

end find_n_l1977_197752


namespace solution_pairs_l1977_197781

theorem solution_pairs (x y : ℝ) : 
  (2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2) ↔ (y = -x - 2 ∨ y = -2 * x + 1) := 
by 
  sorry

end solution_pairs_l1977_197781


namespace kiera_fruit_cups_l1977_197771

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def total_cost : ℕ := 17

theorem kiera_fruit_cups : ∃ kiera_fruit_cups : ℕ, muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cups = total_cost - (muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups) :=
by
  let francis_cost := muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  let remaining_cost := total_cost - francis_cost
  let kiera_fruit_cups := remaining_cost / fruit_cup_cost
  exact ⟨kiera_fruit_cups, by sorry⟩

end kiera_fruit_cups_l1977_197771


namespace find_r_l1977_197764

theorem find_r (r : ℝ) (h_curve : r = -2 * r^2 + 5 * r - 2) : r = 1 :=
sorry

end find_r_l1977_197764


namespace find_B_current_age_l1977_197789

variable {A B C : ℕ}

theorem find_B_current_age (h1 : A + 10 = 2 * (B - 10))
                          (h2 : A = B + 7)
                          (h3 : C = (A + B) / 2) :
                          B = 37 := by
  sorry

end find_B_current_age_l1977_197789


namespace range_of_m_l1977_197765

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - m| > 4) ↔ m > 3 ∨ m < -5 := 
sorry

end range_of_m_l1977_197765


namespace inappropriate_survey_method_l1977_197718

def survey_method_appropriate (method : String) : Bool :=
  method = "sampling" -- only sampling is considered appropriate in this toy model

def survey_approps : Bool :=
  let A := survey_method_appropriate "sampling"
  let B := survey_method_appropriate "sampling"
  let C := ¬ survey_method_appropriate "census"
  let D := survey_method_appropriate "census"
  C

theorem inappropriate_survey_method :
  survey_approps = true :=
by
  sorry

end inappropriate_survey_method_l1977_197718


namespace rachels_game_final_configurations_l1977_197701

-- Define the number of cells in the grid
def n : ℕ := 2011

-- Define the number of moves needed
def moves_needed : ℕ := n - 3

-- Define a function that counts the number of distinct final configurations
-- based on the number of fights (f) possible in the given moves.
def final_configurations : ℕ := moves_needed + 1

theorem rachels_game_final_configurations : final_configurations = 2009 :=
by
  -- Calculation shows that moves_needed = 2008 and therefore final_configurations = 2008 + 1 = 2009.
  sorry

end rachels_game_final_configurations_l1977_197701


namespace simplify_expression_l1977_197700

-- Definitions derived from the problem statement
variable (x : ℝ)

-- Theorem statement
theorem simplify_expression : 1 - (1 + (1 - (1 + (1 - x)))) = 1 - x :=
sorry

end simplify_expression_l1977_197700


namespace triangle_possible_side_lengths_l1977_197751

theorem triangle_possible_side_lengths (x : ℕ) (hx : x > 0) (h1 : x^2 + 9 > 12) (h2 : x^2 + 12 > 9) (h3 : 9 + 12 > x^2) : x = 2 ∨ x = 3 ∨ x = 4 :=
by
  sorry

end triangle_possible_side_lengths_l1977_197751


namespace quadratic_one_real_root_l1977_197728

theorem quadratic_one_real_root (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : ∀ x : ℝ, x^2 + 6*m*x - n = 0 → x * x = 0) : n = 9*m^2 := 
by 
  sorry

end quadratic_one_real_root_l1977_197728


namespace find_a4_l1977_197732

variable {a_n : ℕ → ℝ}
variable (S_n : ℕ → ℝ)

noncomputable def Sn := 1/2 * 5 * (a_n 1 + a_n 5)

axiom h1 : S_n 5 = 25
axiom h2 : a_n 2 = 3

theorem find_a4 : a_n 4 = 5 := sorry

end find_a4_l1977_197732


namespace tangent_line_slope_l1977_197754

theorem tangent_line_slope (x₀ y₀ k : ℝ)
    (h_tangent_point : y₀ = x₀ + Real.exp (-x₀))
    (h_tangent_line : y₀ = k * x₀) :
    k = 1 - Real.exp 1 := 
sorry

end tangent_line_slope_l1977_197754


namespace arithmetic_sequence_a1a6_eq_l1977_197785

noncomputable def a_1 : ℤ := 2
noncomputable def d : ℤ := 1
noncomputable def a_n (n : ℕ) : ℤ := a_1 + (n - 1) * d

theorem arithmetic_sequence_a1a6_eq :
  (a_1 * a_n 6) = 14 := by 
  sorry

end arithmetic_sequence_a1a6_eq_l1977_197785


namespace compare_decimal_fraction_l1977_197790

theorem compare_decimal_fraction : 0.8 - (1 / 2) = 0.3 := by
  sorry

end compare_decimal_fraction_l1977_197790


namespace wire_cut_circle_square_area_eq_l1977_197791

theorem wire_cut_circle_square_area_eq (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : (a^2 / (4 * π)) = ((b^2) / 16)) : 
  a / b = 2 / Real.sqrt π :=
by
  sorry

end wire_cut_circle_square_area_eq_l1977_197791


namespace percentage_increase_school_B_l1977_197716

theorem percentage_increase_school_B (A B Q_A Q_B : ℝ) 
  (h1 : Q_A = 0.7 * A) 
  (h2 : Q_B = 1.5 * Q_A) 
  (h3 : Q_B = 0.875 * B) :
  (B - A) / A * 100 = 20 :=
by
  sorry

end percentage_increase_school_B_l1977_197716


namespace positive_difference_solutions_of_abs_eq_l1977_197729

theorem positive_difference_solutions_of_abs_eq (x1 x2 : ℝ) (h1 : 2 * x1 - 3 = 15) (h2 : 2 * x2 - 3 = -15) : |x1 - x2| = 15 := by
  sorry

end positive_difference_solutions_of_abs_eq_l1977_197729


namespace monster_ratio_l1977_197776

theorem monster_ratio (r : ℝ) :
  (121 + 121 * r + 121 * r^2 = 847) → r = 2 :=
by
  intros h
  sorry

end monster_ratio_l1977_197776
