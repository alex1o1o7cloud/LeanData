import Mathlib

namespace perpendicular_lines_slope_product_l2279_227991

theorem perpendicular_lines_slope_product (a : ℝ) (x y : ℝ) :
  let l1 := ax + y + 2 = 0
  let l2 := x + y = 0
  ( -a * -1 = -1 ) -> a = -1 :=
sorry

end perpendicular_lines_slope_product_l2279_227991


namespace Hadley_walked_to_grocery_store_in_2_miles_l2279_227922

-- Define the variables and conditions
def distance_to_grocery_store (x : ℕ) : Prop :=
  x + (x - 1) + 3 = 6

-- Stating the main proposition to prove
theorem Hadley_walked_to_grocery_store_in_2_miles : ∃ x : ℕ, distance_to_grocery_store x ∧ x = 2 := 
by sorry

end Hadley_walked_to_grocery_store_in_2_miles_l2279_227922


namespace calculate_fraction_l2279_227913

theorem calculate_fraction : 
  ∃ f : ℝ, (14.500000000000002 ^ 2) * f = 126.15 ∧ f = 0.6 :=
by
  sorry

end calculate_fraction_l2279_227913


namespace mn_equals_neg3_l2279_227914

noncomputable def function_with_extreme_value (m n : ℝ) : Prop :=
  let f := λ x : ℝ => m * x^3 + n * x
  let f' := λ x : ℝ => 3 * m * x^2 + n
  f' (1 / m) = 0

theorem mn_equals_neg3 (m n : ℝ) (h : function_with_extreme_value m n) : m * n = -3 :=
sorry

end mn_equals_neg3_l2279_227914


namespace number_of_standing_demons_l2279_227947

variable (N : ℕ)
variable (initial_knocked_down : ℕ)
variable (initial_standing : ℕ)
variable (current_knocked_down : ℕ)
variable (current_standing : ℕ)

axiom initial_condition : initial_knocked_down = (3 * initial_standing) / 2
axiom condition_after_changes : current_knocked_down = initial_knocked_down + 2
axiom condition_after_changes_2 : current_standing = initial_standing - 10
axiom final_condition : current_standing = (5 * current_knocked_down) / 4

theorem number_of_standing_demons : current_standing = 35 :=
sorry

end number_of_standing_demons_l2279_227947


namespace function_relationship_l2279_227933

-- Definitions of the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := ∀ {x y}, x ∈ s → y ∈ s → x < y → f y ≤ f x

-- The main statement we want to prove
theorem function_relationship (f : ℝ → ℝ) 
  (hf_even : even_function f)
  (hf_decreasing : decreasing_on f (Set.Ici 0)) :
  f 1 > f (-10) :=
by sorry

end function_relationship_l2279_227933


namespace range_of_n_l2279_227907

def hyperbola_equation (m n : ℝ) : Prop :=
  (m^2 + n) * (3 * m^2 - n) > 0

def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

theorem range_of_n (m n : ℝ) :
  hyperbola_equation m n ∧ foci_distance m n →
  -1 < n ∧ n < 3 :=
by
  intro h
  have hyperbola_condition := h.1
  have distance_condition := h.2
  sorry

end range_of_n_l2279_227907


namespace alice_two_turns_probability_l2279_227979

def alice_to_alice_first_turn : ℚ := 2 / 3
def alice_to_bob_first_turn : ℚ := 1 / 3
def bob_to_alice_second_turn : ℚ := 1 / 4
def bob_keeps_second_turn : ℚ := 3 / 4
def alice_keeps_second_turn : ℚ := 2 / 3

def probability_alice_keeps_twice : ℚ := alice_to_alice_first_turn * alice_keeps_second_turn
def probability_alice_bob_alice : ℚ := alice_to_bob_first_turn * bob_to_alice_second_turn

theorem alice_two_turns_probability : 
  probability_alice_keeps_twice + probability_alice_bob_alice = 37 / 108 := 
by
  sorry

end alice_two_turns_probability_l2279_227979


namespace weather_condition_l2279_227918

theorem weather_condition (T : ℝ) (windy : Prop) (kites_will_fly : Prop) 
  (h1 : (T > 25 ∧ windy) → kites_will_fly) 
  (h2 : ¬ kites_will_fly) : T ≤ 25 ∨ ¬ windy :=
by 
  sorry

end weather_condition_l2279_227918


namespace directrix_of_parabola_l2279_227981

-- Define the given conditions
def parabola_eqn (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 5

-- The problem is to show that the directrix of this parabola has the equation y = 23/12
theorem directrix_of_parabola : 
  (∃ y : ℝ, ∀ x : ℝ, parabola_eqn x = y) →

  ∃ y : ℝ, y = 23 / 12 :=
sorry

end directrix_of_parabola_l2279_227981


namespace jamal_total_cost_l2279_227999

-- Definitions based on conditions
def dozen := 12
def half_dozen := dozen / 2
def crayons_bought := 4 * half_dozen
def cost_per_crayon := 2
def total_cost := crayons_bought * cost_per_crayon

-- Proof statement (the question translated to a Lean theorem)
theorem jamal_total_cost : total_cost = 48 := by
  -- Proof skipped
  sorry

end jamal_total_cost_l2279_227999


namespace cheaper_rock_cost_per_ton_l2279_227943

theorem cheaper_rock_cost_per_ton (x : ℝ) 
    (h1 : 24 * 1 = 24) 
    (h2 : 800 = 16 * x + 8 * 40) : 
    x = 30 :=
sorry

end cheaper_rock_cost_per_ton_l2279_227943


namespace least_multiple_of_36_with_digit_product_multiple_of_9_l2279_227932

def is_multiple_of_36 (n : ℕ) : Prop :=
  n % 36 = 0

def product_of_digits_multiple_of_9 (n : ℕ) : Prop :=
  ∃ d : List ℕ, (n = List.foldl (λ x y => x * 10 + y) 0 d) ∧ (List.foldl (λ x y => x * y) 1 d) % 9 = 0

theorem least_multiple_of_36_with_digit_product_multiple_of_9 : ∃ n : ℕ, is_multiple_of_36 n ∧ product_of_digits_multiple_of_9 n ∧ n = 36 :=
by
  sorry

end least_multiple_of_36_with_digit_product_multiple_of_9_l2279_227932


namespace count_valid_N_l2279_227993

theorem count_valid_N : ∃ (count : ℕ), count = 10 ∧ 
    (∀ N : ℕ, (10 ≤ N ∧ N < 100) → 
        (∃ a b c d : ℕ, 
            a < 3 ∧ b < 3 ∧ c < 3 ∧ d < 4 ∧
            N = 3 * a + b ∧ N = 4 * c + d ∧
            2 * N % 50 = ((9 * a + b) + (8 * c + d)) % 50)) :=
sorry

end count_valid_N_l2279_227993


namespace number_of_integers_in_sequence_l2279_227951

theorem number_of_integers_in_sequence 
  (a_0 : ℕ) 
  (h_0 : a_0 = 8820) 
  (seq : ℕ → ℕ) 
  (h_seq : ∀ n : ℕ, seq (n + 1) = seq n / 3) :
  ∃ n : ℕ, seq n = 980 ∧ n + 1 = 3 :=
by
  sorry

end number_of_integers_in_sequence_l2279_227951


namespace probability_solved_l2279_227945

theorem probability_solved (pA pB pA_and_B : ℚ) :
  pA = 2 / 3 → pB = 3 / 4 → pA_and_B = (2 / 3) * (3 / 4) →
  pA + pB - pA_and_B = 11 / 12 :=
by
  intros hA hB hA_and_B
  rw [hA, hB, hA_and_B]
  sorry

end probability_solved_l2279_227945


namespace arithmetic_seq_problem_l2279_227923

variable (a : ℕ → ℕ)

def arithmetic_seq (a₁ d : ℕ) : ℕ → ℕ :=
  λ n => a₁ + n * d

theorem arithmetic_seq_problem (a₁ d : ℕ)
  (h_cond : (arithmetic_seq a₁ d 1) + 2 * (arithmetic_seq a₁ d 5) + (arithmetic_seq a₁ d 9) = 120)
  : (arithmetic_seq a₁ d 2) + (arithmetic_seq a₁ d 8) = 60 := 
sorry

end arithmetic_seq_problem_l2279_227923


namespace fruit_difference_l2279_227949

/-- Mr. Connell harvested 60 apples and 3 times as many peaches. The difference 
    between the number of peaches and apples is 120. -/
theorem fruit_difference (apples peaches : ℕ) (h1 : apples = 60) (h2 : peaches = 3 * apples) :
  peaches - apples = 120 :=
sorry

end fruit_difference_l2279_227949


namespace probability_of_stopping_on_H_l2279_227968

theorem probability_of_stopping_on_H (y : ℚ)
  (h1 : (1 / 5) + (1 / 4) + y + y + (1 / 10) = 1)
  : y = 9 / 40 :=
sorry

end probability_of_stopping_on_H_l2279_227968


namespace maximum_side_length_l2279_227998

theorem maximum_side_length 
    (D E F : ℝ) 
    (a b c : ℝ) 
    (h_cos : Real.cos (3 * D) + Real.cos (3 * E) + Real.cos (3 * F) = 1)
    (h_a : a = 12)
    (h_perimeter : a + b + c = 40) : 
    ∃ max_side : ℝ, max_side = 7 + Real.sqrt 23 / 2 :=
by
  sorry

end maximum_side_length_l2279_227998


namespace pipe_network_renovation_l2279_227970

theorem pipe_network_renovation 
  (total_length : Real)
  (efficiency_increase : Real)
  (days_ahead_of_schedule : Nat)
  (days_completed : Nat)
  (total_period : Nat)
  (original_daily_renovation : Real)
  (additional_renovation : Real)
  (h1 : total_length = 3600)
  (h2 : efficiency_increase = 20 / 100)
  (h3 : days_ahead_of_schedule = 10)
  (h4 : days_completed = 20)
  (h5 : total_period = 40)
  (h6 : (3600 / original_daily_renovation) - (3600 / (1.2 * original_daily_renovation)) = 10)
  (h7 : 20 * (72 + additional_renovation) >= 3600 - 1440) :
  (1.2 * original_daily_renovation = 72) ∧ (additional_renovation >= 36) :=
by
  sorry

end pipe_network_renovation_l2279_227970


namespace k_greater_than_half_l2279_227990

-- Definition of the problem conditions
variables {a b c k : ℝ}

-- Assume a, b, c are the sides of a triangle
axiom triangle_inequality : a + b > c

-- Given condition
axiom sides_condition : a^2 + b^2 = k * c^2

-- The theorem to prove k > 0.5
theorem k_greater_than_half (h1 : a + b > c) (h2 : a^2 + b^2 = k * c^2) : k > 0.5 :=
by
  sorry

end k_greater_than_half_l2279_227990


namespace max_value_f_on_interval_l2279_227929

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_f_on_interval : 
  ∃ x ∈ Set.Icc (0 : ℝ) 1, ∀ y ∈ Set.Icc (0 : ℝ) 1, f y ≤ f x ∧ f x = Real.exp 1 - 1 := sorry

end max_value_f_on_interval_l2279_227929


namespace anita_apples_l2279_227997

theorem anita_apples (num_students : ℕ) (apples_per_student : ℕ) (total_apples : ℕ) 
  (h1 : num_students = 60) 
  (h2 : apples_per_student = 6) 
  (h3 : total_apples = num_students * apples_per_student) : 
  total_apples = 360 := 
by
  sorry

end anita_apples_l2279_227997


namespace equivalent_xy_xxyy_not_equivalent_xyty_txy_not_equivalent_xy_xt_l2279_227950

-- Define a transformation predicate for words
inductive transform : List Char -> List Char -> Prop
| xy_to_yyx : ∀ (l1 l2 : List Char), transform (l1 ++ ['x', 'y'] ++ l2) (l1 ++ ['y', 'y', 'x'] ++ l2)
| yyx_to_xy : ∀ (l1 l2 : List Char), transform (l1 ++ ['y', 'y', 'x'] ++ l2) (l1 ++ ['x', 'y'] ++ l2)
| xt_to_ttx : ∀ (l1 l2 : List Char), transform (l1 ++ ['x', 't'] ++ l2) (l1 ++ ['t', 't', 'x'] ++ l2)
| ttx_to_xt : ∀ (l1 l2 : List Char), transform (l1 ++ ['t', 't', 'x'] ++ l2) (l1 ++ ['x', 't'] ++ l2)
| yt_to_ty : ∀ (l1 l2 : List Char), transform (l1 ++ ['y', 't'] ++ l2) (l1 ++ ['t', 'y'] ++ l2)
| ty_to_yt : ∀ (l1 l2 : List Char), transform (l1 ++ ['t', 'y'] ++ l2) (l1 ++ ['y', 't'] ++ l2)

-- Reflexive and transitive closure of transform
inductive transforms : List Char -> List Char -> Prop
| base : ∀ l, transforms l l
| step : ∀ l m n, transform l m → transforms m n → transforms l n

-- Definitions for the words and their information
def word1 := ['x', 'x', 'y', 'y']
def word2 := ['x', 'y', 'y', 'y', 'y', 'x']
def word3 := ['x', 'y', 't', 'x']
def word4 := ['t', 'x', 'y', 't']
def word5 := ['x', 'y']
def word6 := ['x', 't']

-- Proof statements
theorem equivalent_xy_xxyy : transforms word1 word2 :=
by sorry

theorem not_equivalent_xyty_txy : ¬ transforms word3 word4 :=
by sorry

theorem not_equivalent_xy_xt : ¬ transforms word5 word6 :=
by sorry

end equivalent_xy_xxyy_not_equivalent_xyty_txy_not_equivalent_xy_xt_l2279_227950


namespace second_rice_price_l2279_227915

theorem second_rice_price (P : ℝ) 
  (price_first : ℝ := 3.10) 
  (price_mixture : ℝ := 3.25) 
  (ratio_first_to_second : ℝ := 3 / 7) :
  (3 * price_first + 7 * P) / 10 = price_mixture → 
  P = 3.3142857142857145 :=
by
  sorry

end second_rice_price_l2279_227915


namespace distance_between_parallel_lines_l2279_227942

theorem distance_between_parallel_lines :
  ∀ {x y : ℝ}, 
  (3 * x - 4 * y + 1 = 0) → (3 * x - 4 * y + 7 = 0) → 
  ∃ d, d = (6 : ℝ) / 5 :=
by 
  sorry

end distance_between_parallel_lines_l2279_227942


namespace earnings_difference_is_200_l2279_227905

noncomputable def difference_in_earnings : ℕ :=
  let asking_price := 5200
  let maintenance_cost := asking_price / 10
  let first_offer_earnings := asking_price - maintenance_cost
  let headlight_cost := 80
  let tire_cost := 3 * headlight_cost
  let total_repair_cost := headlight_cost + tire_cost
  let second_offer_earnings := asking_price - total_repair_cost
  second_offer_earnings - first_offer_earnings

theorem earnings_difference_is_200 : difference_in_earnings = 200 := by
  sorry

end earnings_difference_is_200_l2279_227905


namespace values_of_d_l2279_227977

theorem values_of_d (a b c d : ℕ) 
  (h : (ad - 1) / (a + 1) + (bd - 1) / (b + 1) + (cd - 1) / (c + 1) = d) : 
  d = 1 ∨ d = 2 ∨ d = 3 := 
sorry

end values_of_d_l2279_227977


namespace solve_for_c_l2279_227964

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem solve_for_c (a b c d : ℝ) 
    (h : ∀ x : ℝ, quadratic_function a b c x ≥ d) : c = d + b^2 / (4 * a) :=
by
  sorry

end solve_for_c_l2279_227964


namespace probability_three_even_l2279_227959

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Definition of the probability of exactly three dice showing an even number
noncomputable def prob_exactly_three_even (n : ℕ) (k : ℕ) (p : ℚ) : ℚ := 
  (binomial n k : ℚ) * (p^k) * ((1 - p)^(n - k))

-- The main theorem stating the desired probability
theorem probability_three_even (n : ℕ) (p : ℚ) (k : ℕ) (h₁ : n = 6) (h₂ : p = 1/2) (h₃ : k = 3) :
  prob_exactly_three_even n k p = 5 / 16 := by
  sorry

-- Include required definitions and expected values for the theorem
#check binomial
#check prob_exactly_three_even
#check probability_three_even

end probability_three_even_l2279_227959


namespace ellipse_focus_and_axes_l2279_227903

theorem ellipse_focus_and_axes (m : ℝ) :
  (∃ a b : ℝ, (a > b) ∧ (mx^2 + y^2 = 1) ∧ (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2 * a = 3 * 2 * b)) → 
  m = 4 / 9 :=
by
  intro h
  rcases h with ⟨a, b, hab, h_eq, ha, hb, ha_b_eq⟩
  sorry

end ellipse_focus_and_axes_l2279_227903


namespace find_larger_number_l2279_227965

theorem find_larger_number (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) (h3 : x * y = 375) (hx : x > y) : x = 25 :=
sorry

end find_larger_number_l2279_227965


namespace f_odd_and_periodic_l2279_227921

open Function

-- Define the function f : ℝ → ℝ satisfying the given conditions
variables (f : ℝ → ℝ)

-- Conditions
axiom f_condition1 : ∀ x : ℝ, f (10 + x) = f (10 - x)
axiom f_condition2 : ∀ x : ℝ, f (20 - x) = -f (20 + x)

-- Theorem statement
theorem f_odd_and_periodic : Odd f ∧ Periodic f 40 :=
by
  -- Proof will be filled here
  sorry

end f_odd_and_periodic_l2279_227921


namespace martha_blue_butterflies_l2279_227983

variables (B Y : Nat)

theorem martha_blue_butterflies (h_total : B + Y + 5 = 11) (h_twice : B = 2 * Y) : B = 4 :=
by
  sorry

end martha_blue_butterflies_l2279_227983


namespace find_y_when_x_is_6_l2279_227900

variable (x y : ℝ)
variable (h₁ : x > 0)
variable (h₂ : y > 0)
variable (k : ℝ)

axiom inverse_proportional : 3 * x^2 * y = k
axiom initial_condition : 3 * 3^2 * 30 = k

theorem find_y_when_x_is_6 (h : x = 6) : y = 7.5 :=
by
  sorry

end find_y_when_x_is_6_l2279_227900


namespace car_mpg_l2279_227958

open Nat

theorem car_mpg (x : ℕ) (h1 : ∀ (m : ℕ), m = 4 * (3 * x) -> x = 27) 
                (h2 : ∀ (d1 d2 : ℕ), d2 = (4 * d1) / 3 - d1 -> d2 = 126) 
                (h3 : ∀ g : ℕ, g = 14)
                : x = 27 := 
by
  sorry

end car_mpg_l2279_227958


namespace total_earnings_proof_l2279_227946

noncomputable def total_earnings (x y : ℝ) : ℝ :=
  let earnings_a := (18 * x * y) / 100
  let earnings_b := (20 * x * y) / 100
  let earnings_c := (20 * x * y) / 100
  earnings_a + earnings_b + earnings_c

theorem total_earnings_proof (x y : ℝ) (h : 2 * x * y = 15000) :
  total_earnings x y = 4350 := by
  sorry

end total_earnings_proof_l2279_227946


namespace tileability_condition_l2279_227994

theorem tileability_condition (a b k m n : ℕ) (h₁ : k ∣ a) (h₂ : k ∣ b) (h₃ : ∃ (t : Nat), t * (a * b) = m * n) : 
  2 * k ∣ m ∨ 2 * k ∣ n := 
sorry

end tileability_condition_l2279_227994


namespace triangle_inequality_l2279_227941

noncomputable def p (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def r (a b c : ℝ) : ℝ := 
  let p := p a b c
  let x := p - a
  let y := p - b
  let z := p - c
  Real.sqrt ((x * y * z) / (x + y + z))

noncomputable def x (a b c : ℝ) : ℝ := p a b c - a
noncomputable def y (a b c : ℝ) : ℝ := p a b c - b
noncomputable def z (a b c : ℝ) : ℝ := p a b c - c

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b > c ∧ a + c > b ∧ b + c > a) :
  1 / (x a b c)^2 + 1 / (y a b c)^2 + 1 / (z a b c)^2 ≥ (x a b c + y a b c + z a b c) / ((x a b c) * (y a b c) * (z a b c)) := by
    sorry

end triangle_inequality_l2279_227941


namespace sufficient_but_not_necessary_condition_l2279_227988

variable {a : Type} {M : Type} (line : a → Prop) (plane : M → Prop)

-- Assume the definitions of perpendicularity
def perp_to_plane (a : a) (M : M) : Prop := sorry -- define perpendicular to plane
def perp_to_lines_in_plane (a : a) (M : M) : Prop := sorry -- define perpendicular to countless lines

-- Mathematical statement
theorem sufficient_but_not_necessary_condition (a : a) (M : M) :
  (perp_to_plane a M → perp_to_lines_in_plane a M) ∧ ¬(perp_to_lines_in_plane a M → perp_to_plane a M) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2279_227988


namespace vector_linear_combination_l2279_227909

open Matrix

theorem vector_linear_combination :
  let v1 := ![3, -9]
  let v2 := ![2, -8]
  let v3 := ![1, -6]
  4 • v1 - 3 • v2 + 2 • v3 = ![8, -24] :=
by sorry

end vector_linear_combination_l2279_227909


namespace correct_option_l2279_227989

noncomputable def M : Set ℝ := {x | x > -2}

theorem correct_option : {0} ⊆ M := 
by 
  intros x hx
  simp at hx
  simp [M]
  show x > -2
  linarith

end correct_option_l2279_227989


namespace total_arrangements_l2279_227976

theorem total_arrangements :
  let students := 6
  let venueA := 1
  let venueB := 2
  let venueC := 3
  (students.choose venueA) * ((students - venueA).choose venueB) = 60 :=
by
  -- placeholder for the proof
  sorry

end total_arrangements_l2279_227976


namespace solution_set_of_inequality_l2279_227982

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) - 3

theorem solution_set_of_inequality :
  { x : ℝ | f x < 0 } = { x : ℝ | x < Real.log 3 / Real.log 2 } :=
by
  sorry

end solution_set_of_inequality_l2279_227982


namespace intersection_of_A_and_B_l2279_227939

def setA : Set ℝ := { x : ℝ | x > -1 }
def setB : Set ℝ := { y : ℝ | 0 ≤ y ∧ y < 1 }

theorem intersection_of_A_and_B :
  (setA ∩ setB) = { z : ℝ | 0 ≤ z ∧ z < 1 } :=
by
  sorry

end intersection_of_A_and_B_l2279_227939


namespace monotonic_increasing_interval_l2279_227960

def f (x : ℝ) : ℝ := x^2 - 2

theorem monotonic_increasing_interval :
  ∀ x y: ℝ, 0 <= x -> x <= y -> f x <= f y := 
by
  -- proof would be here
  sorry

end monotonic_increasing_interval_l2279_227960


namespace base_k_number_to_decimal_l2279_227916

theorem base_k_number_to_decimal (k : ℕ) (h : 4 ≤ k) : 1 * k^2 + 3 * k + 2 = 30 ↔ k = 4 := by
  sorry

end base_k_number_to_decimal_l2279_227916


namespace work_days_l2279_227912

theorem work_days (x : ℕ) (hx : 0 < x) :
  (1 / (x : ℚ) + 1 / 20) = 1 / 15 → x = 60 := by
sorry

end work_days_l2279_227912


namespace area_of_triangle_l2279_227902

theorem area_of_triangle (S_x S_y S_z S : ℝ)
  (hx : S_x = Real.sqrt 7) (hy : S_y = Real.sqrt 6)
  (hz : ∃ k : ℕ, S_z = k) (hs : ∃ n : ℕ, S = n)
  : S = 7 := by
  sorry

end area_of_triangle_l2279_227902


namespace maximize_sum_of_sides_l2279_227952

theorem maximize_sum_of_sides (a b c : ℝ) (A B C : ℝ) 
  (h_b : b = 2) (h_B : B = (Real.pi / 3)) (h_law_of_cosines : b^2 = a^2 + c^2 - 2*a*c*(Real.cos B)) :
  a + c ≤ 4 :=
by
  sorry

end maximize_sum_of_sides_l2279_227952


namespace find_prime_triplet_l2279_227911

def is_geometric_sequence (x y z : ℕ) : Prop :=
  (y^2 = x * z)

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_prime_triplet :
  ∃ (a b c : ℕ), is_prime a ∧ is_prime b ∧ is_prime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  is_geometric_sequence (a + 1) (b + 1) (c + 1) ∧
  (a = 17 ∧ b = 23 ∧ c = 31) :=
by
  sorry

end find_prime_triplet_l2279_227911


namespace average_of_remaining_two_numbers_l2279_227978

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 6.40)
  (h2 : (a + b) / 2 = 6.2)
  (h3 : (c + d) / 2 = 6.1) :
  ((e + f) / 2) = 6.9 :=
by
  sorry

end average_of_remaining_two_numbers_l2279_227978


namespace largest_inscribed_square_l2279_227901

-- Define the problem data
noncomputable def s : ℝ := 15
noncomputable def h : ℝ := s * (Real.sqrt 3) / 2
noncomputable def y : ℝ := s - h

-- Statement to prove
theorem largest_inscribed_square :
  y = (30 - 15 * Real.sqrt 3) / 2 := by
  sorry

end largest_inscribed_square_l2279_227901


namespace intersection_complement_eq_l2279_227986

-- Definitions of the sets M and N
def M : Set ℝ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 > 0}

-- Complements with respect to the reals
def complement_R (A : Set ℝ) : Set ℝ := {x | x ∉ A}

-- Target goal to prove
theorem intersection_complement_eq :
  M ∩ (complement_R N) = {1, 2} :=
by
  sorry

end intersection_complement_eq_l2279_227986


namespace train_length_is_120_l2279_227936

-- Definitions based on conditions
def bridge_length : ℕ := 600
def total_time : ℕ := 30
def on_bridge_time : ℕ := 20

-- Proof statement
theorem train_length_is_120 (x : ℕ) (speed1 speed2 : ℕ) :
  (speed1 = (bridge_length + x) / total_time) ∧
  (speed2 = bridge_length / on_bridge_time) ∧
  (speed1 = speed2) →
  x = 120 :=
by
  sorry

end train_length_is_120_l2279_227936


namespace fraction_expression_simplifies_to_313_l2279_227973

theorem fraction_expression_simplifies_to_313 :
  (12^4 + 324) * (24^4 + 324) * (36^4 + 324) * (48^4 + 324) * (60^4 + 324) * (72^4 + 324) /
  (6^4 + 324) * (18^4 + 324) * (30^4 + 324) * (42^4 + 324) * (54^4 + 324) * (66^4 + 324) = 313 :=
by
  sorry

end fraction_expression_simplifies_to_313_l2279_227973


namespace miles_driven_l2279_227924

def rental_fee : ℝ := 20.99
def charge_per_mile : ℝ := 0.25
def total_amount_paid : ℝ := 95.74

theorem miles_driven (miles_driven: ℝ) : 
  (total_amount_paid - rental_fee) / charge_per_mile = miles_driven → miles_driven = 299 := by
  intros
  sorry

end miles_driven_l2279_227924


namespace profit_percentage_l2279_227908

theorem profit_percentage (C S : ℝ) (hC : C = 800) (hS : S = 1080) :
  ((S - C) / C) * 100 = 35 := 
by
  sorry

end profit_percentage_l2279_227908


namespace problem_solution_l2279_227935

noncomputable def S : ℝ :=
  1 / (5 - Real.sqrt 19) - 1 / (Real.sqrt 19 - Real.sqrt 18) + 
  1 / (Real.sqrt 18 - Real.sqrt 17) - 1 / (Real.sqrt 17 - 3) + 
  1 / (3 - Real.sqrt 2)

theorem problem_solution : S = 5 + Real.sqrt 2 := by
  sorry

end problem_solution_l2279_227935


namespace samantha_erased_length_l2279_227953

/--
Samantha drew a line that was originally 1 meter (100 cm) long, and then it was erased until the length was 90 cm.
This theorem proves that the amount erased was 10 cm.
-/
theorem samantha_erased_length : 
  let original_length := 100 -- original length in cm
  let final_length := 90 -- final length in cm
  original_length - final_length = 10 := 
by
  sorry

end samantha_erased_length_l2279_227953


namespace negation_of_every_student_is_punctual_l2279_227938

variable (Student : Type) (student punctual : Student → Prop)

theorem negation_of_every_student_is_punctual :
  ¬ (∀ x, student x → punctual x) ↔ ∃ x, student x ∧ ¬ punctual x := by
sorry

end negation_of_every_student_is_punctual_l2279_227938


namespace sum_of_products_of_roots_eq_neg3_l2279_227966

theorem sum_of_products_of_roots_eq_neg3 {p q r s : ℂ} 
  (h : ∀ {x : ℂ}, 4 * x^4 - 8 * x^3 + 12 * x^2 - 16 * x + 9 = 0 → (x = p ∨ x = q ∨ x = r ∨ x = s)) : 
  p * q + p * r + p * s + q * r + q * s + r * s = -3 := 
sorry

end sum_of_products_of_roots_eq_neg3_l2279_227966


namespace lever_equilibrium_min_force_l2279_227917

noncomputable def lever_minimum_force (F L : ℝ) : Prop :=
  (F * L = 49 + 2 * (L^2))

theorem lever_equilibrium_min_force : ∃ F : ℝ, ∃ L : ℝ, L = 7 → lever_minimum_force F L :=
by
  sorry

end lever_equilibrium_min_force_l2279_227917


namespace exhaust_pipe_leak_time_l2279_227985

theorem exhaust_pipe_leak_time : 
  (∃ T : Real, T > 0 ∧ 
                (1 / 10 - 1 / T) = 1 / 59.999999999999964 ∧ 
                T = 12) :=
by
  sorry

end exhaust_pipe_leak_time_l2279_227985


namespace green_more_than_blue_l2279_227927

variable (B Y G : ℕ)

theorem green_more_than_blue
  (h_sum : B + Y + G = 126)
  (h_ratio : ∃ k : ℕ, B = 3 * k ∧ Y = 7 * k ∧ G = 8 * k) :
  G - B = 35 := by
  sorry

end green_more_than_blue_l2279_227927


namespace sum_of_digits_of_n_l2279_227969

theorem sum_of_digits_of_n : 
  ∃ n : ℕ, n > 1500 ∧ 
    (Nat.gcd 40 (n + 105) = 10) ∧ 
    (Nat.gcd (n + 40) 105 = 35) ∧ 
    (Nat.digits 10 n).sum = 8 :=
by 
  sorry

end sum_of_digits_of_n_l2279_227969


namespace mirror_area_correct_l2279_227937

noncomputable def width_of_mirror (frame_width : ℕ) (side_width : ℕ) : ℕ :=
  frame_width - 2 * side_width

noncomputable def height_of_mirror (frame_height : ℕ) (side_width : ℕ) : ℕ :=
  frame_height - 2 * side_width

noncomputable def area_of_mirror (frame_width : ℕ) (frame_height : ℕ) (side_width : ℕ) : ℕ :=
  width_of_mirror frame_width side_width * height_of_mirror frame_height side_width

theorem mirror_area_correct :
  area_of_mirror 50 70 7 = 2016 :=
by
  sorry

end mirror_area_correct_l2279_227937


namespace circle_through_points_l2279_227987

theorem circle_through_points :
  ∃ (D E F : ℝ), F = 0 ∧ D = -4 ∧ E = -6 ∧ ∀ (x y : ℝ),
  (x = 0 ∧ y = 0 ∨ x = 4 ∧ y = 0 ∨ x = -1 ∧ y = 1) →
  x^2 + y^2 + D * x + E * y + F = 0 :=
by
  sorry

end circle_through_points_l2279_227987


namespace photocopy_distribution_l2279_227971

-- Define the problem setting
variables {n k : ℕ}

-- Define the theorem stating the problem
theorem photocopy_distribution :
  ∀ n k : ℕ, (n > 0) → 
  (k + n).choose (n - 1) = (k + n - 1).choose (n - 1) :=
by sorry

end photocopy_distribution_l2279_227971


namespace sum_of_squares_of_roots_of_quadratic_l2279_227948

theorem sum_of_squares_of_roots_of_quadratic :
  ( ∃ x1 x2 : ℝ, x1^2 - 3 * x1 - 1 = 0 ∧ x2^2 - 3 * x2 - 1 = 0 ∧ x1 ≠ x2) →
  x1^2 + x2^2 = 11 :=
by
  /- Proof goes here -/
  sorry

end sum_of_squares_of_roots_of_quadratic_l2279_227948


namespace quadratic_roots_l2279_227974

noncomputable def roots_quadratic : Prop :=
  ∀ (a b : ℝ), (a + b = 7) ∧ (a * b = 7) → (a^2 + b^2 = 35)

theorem quadratic_roots (a b : ℝ) (h : a + b = 7 ∧ a * b = 7) : a^2 + b^2 = 35 :=
by
  sorry

end quadratic_roots_l2279_227974


namespace length_of_GH_l2279_227926

-- Define the lengths of the segments as given in the conditions
def AB : ℕ := 11
def FE : ℕ := 13
def CD : ℕ := 5

-- Define what we need to prove: the length of GH is 29
theorem length_of_GH (AB FE CD : ℕ) : AB = 11 → FE = 13 → CD = 5 → (AB + CD + FE = 29) :=
by
  sorry

end length_of_GH_l2279_227926


namespace find_n_l2279_227910

theorem find_n (n : ℕ) (hn_pos : 0 < n) (hn_greater_30 : 30 < n) 
  (divides : (4 * n - 1) ∣ 2002 * n) : n = 36 := 
by
  sorry

end find_n_l2279_227910


namespace remainder_squared_mod_five_l2279_227980

theorem remainder_squared_mod_five (n k : ℤ) (h : n = 5 * k + 3) : ((n - 1) ^ 2) % 5 = 4 :=
by
  sorry

end remainder_squared_mod_five_l2279_227980


namespace calculate_X_value_l2279_227931

theorem calculate_X_value : 
  let M := (2025 : ℝ) / 3
  let N := M / 4
  let X := M - N
  X = 506.25 :=
by 
  sorry

end calculate_X_value_l2279_227931


namespace current_short_trees_l2279_227934

theorem current_short_trees (S : ℕ) (S_planted : ℕ) (S_total : ℕ) 
  (H1 : S_planted = 105) 
  (H2 : S_total = 217) 
  (H3 : S + S_planted = S_total) :
  S = 112 :=
by
  sorry

end current_short_trees_l2279_227934


namespace error_estimate_alternating_series_l2279_227972

theorem error_estimate_alternating_series :
  let S := (1:ℝ) - (1 / 2) + (1 / 3) - (1 / 4) + (-(1 / 5)) 
  let S₄ := (1:ℝ) - (1 / 2) + (1 / 3) - (1 / 4)
  ∃ ΔS : ℝ, ΔS = |-(1 / 5)| ∧ ΔS < 0.2 := by
  sorry

end error_estimate_alternating_series_l2279_227972


namespace symmetry_center_of_tangent_l2279_227962

noncomputable def tangentFunction (x : ℝ) : ℝ := Real.tan (2 * x - (Real.pi / 3))

theorem symmetry_center_of_tangent :
  (∃ k : ℤ, (Real.pi / 6) + (k * Real.pi / 4) = 5 * Real.pi / 12 ∧ tangentFunction ((5 * Real.pi) / 12) = 0 ) :=
sorry

end symmetry_center_of_tangent_l2279_227962


namespace find_transform_l2279_227995

structure Vector3D (α : Type) := (x y z : α)

def T (u : Vector3D ℝ) : Vector3D ℝ := sorry

axiom linearity (a b : ℝ) (u v : Vector3D ℝ) : T (Vector3D.mk (a * u.x + b * v.x) (a * u.y + b * v.y) (a * u.z + b * v.z)) = 
                      Vector3D.mk (a * (T u).x + b * (T v).x) (a * (T u).y + b * (T v).y) (a * (T u).z + b * (T v).z)

axiom cross_product (u v : Vector3D ℝ) : T (Vector3D.mk (u.y * v.z - u.z * v.y) (u.z * v.x - u.x * v.z) (u.x * v.y - u.y * v.x)) = 
                    (Vector3D.mk ((T u).y * (T v).z - (T u).z * (T v).y) ((T u).z * (T v).x - (T u).x * (T v).z) ((T u).x * (T v).y - (T u).y * (T v).x))

axiom transform1 : T (Vector3D.mk 3 3 7) = Vector3D.mk 2 (-4) 5
axiom transform2 : T (Vector3D.mk (-2) 5 4) = Vector3D.mk 6 1 0

theorem find_transform : T (Vector3D.mk 5 15 11) = Vector3D.mk a b c := sorry

end find_transform_l2279_227995


namespace find_x_from_conditions_l2279_227955

theorem find_x_from_conditions (x y : ℝ)
  (h1 : (6 : ℝ) = (1 / 2 : ℝ) * x)
  (h2 : y = (1 / 2 :ℝ) * 10)
  (h3 : x * y = 60) : x = 12 := by
  sorry

end find_x_from_conditions_l2279_227955


namespace internal_angles_triangle_ABC_l2279_227975

theorem internal_angles_triangle_ABC (α β γ : ℕ) (h₁ : α + β + γ = 180)
  (h₂ : α + γ = 138) (h₃ : β + γ = 108) : (α = 72) ∧ (β = 42) ∧ (γ = 66) :=
by
  sorry

end internal_angles_triangle_ABC_l2279_227975


namespace largest_fraction_is_36_l2279_227919

theorem largest_fraction_is_36 : 
  let A := (1 : ℚ) / 5
  let B := (2 : ℚ) / 10
  let C := (7 : ℚ) / 15
  let D := (9 : ℚ) / 20
  let E := (3 : ℚ) / 6
  A < E ∧ B < E ∧ C < E ∧ D < E :=
by
  let A := (1 : ℚ) / 5
  let B := (2 : ℚ) / 10
  let C := (7 : ℚ) / 15
  let D := (9 : ℚ) / 20
  let E := (3 : ℚ) / 6
  sorry

end largest_fraction_is_36_l2279_227919


namespace vanya_four_times_faster_l2279_227904

-- We let d be the total distance, and define the respective speeds
variables (d : ℝ) (v_m v_v : ℝ)

-- Conditions from the problem
-- 1. Vanya starts after Masha
axiom start_after_masha : ∀ t : ℝ, t > 0

-- 2. Vanya overtakes Masha at one-third of the distance
axiom vanya_overtakes_masha : ∀ t : ℝ, (v_v * t) = d / 3

-- 3. When Vanya reaches the school, Masha still has half of the way to go
axiom masha_halfway : ∀ t : ℝ, (v_m * t) = d / 2

-- Goal to prove
theorem vanya_four_times_faster : v_v = 4 * v_m :=
sorry

end vanya_four_times_faster_l2279_227904


namespace total_items_left_in_store_l2279_227956

noncomputable def items_ordered : ℕ := 4458
noncomputable def items_sold : ℕ := 1561
noncomputable def items_in_storeroom : ℕ := 575

theorem total_items_left_in_store : 
  (items_ordered - items_sold) + items_in_storeroom = 3472 := 
by 
  sorry

end total_items_left_in_store_l2279_227956


namespace flour_per_special_crust_l2279_227992

-- Definitions of daily pie crusts and flour usage for standard crusts
def daily_pie_crusts := 50
def flour_per_standard_crust := 1 / 10
def total_daily_flour := daily_pie_crusts * flour_per_standard_crust

-- Definitions for special pie crusts today
def special_pie_crusts := 25
def total_special_flour := total_daily_flour / special_pie_crusts

-- Problem statement in Lean
theorem flour_per_special_crust :
  total_special_flour = 1 / 5 := by
  sorry

end flour_per_special_crust_l2279_227992


namespace triangle_inequality_l2279_227967

theorem triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  3 * (a * b + a * c + b * c) ≤ (a + b + c) ^ 2 ∧ (a + b + c) ^ 2 < 4 * (a * b + a * c + b * c) :=
sorry

end triangle_inequality_l2279_227967


namespace oranges_in_bin_l2279_227954

theorem oranges_in_bin (initial : ℕ) (thrown_away : ℕ) (added : ℕ) (result : ℕ)
    (h_initial : initial = 40)
    (h_thrown_away : thrown_away = 25)
    (h_added : added = 21)
    (h_result : result = 36) : initial - thrown_away + added = result :=
by
  -- skipped proof steps
  exact sorry

end oranges_in_bin_l2279_227954


namespace gumball_problem_l2279_227906

/--
A gumball machine contains 10 red, 6 white, 8 blue, and 9 green gumballs.
The least number of gumballs a person must buy to be sure of getting four gumballs of the same color is 13.
-/
theorem gumball_problem
  (red white blue green : ℕ)
  (h_red : red = 10)
  (h_white : white = 6)
  (h_blue : blue = 8)
  (h_green : green = 9) :
  ∃ n, n = 13 ∧ (∀ gumballs : ℕ, gumballs ≥ 13 → (∃ color_count : ℕ, color_count ≥ 4 ∧ (color_count = red ∨ color_count = white ∨ color_count = blue ∨ color_count = green))) :=
sorry

end gumball_problem_l2279_227906


namespace puzzle_solution_l2279_227996

theorem puzzle_solution :
  (∀ n m k : ℕ, n + m + k = 111 → 9 * (n + m + k) / 3 = 9) ∧
  (∀ n m k : ℕ, n + m + k = 444 → 12 * (n + m + k) / 12 = 12) ∧
  (∀ n m k : ℕ, n + m + k = 777 → (7 * 3 ≠ 15 → (7 * 3 - 6 = 15)) ) →
  ∀ n m k : ℕ, n + m + k = 888 → 8 * (n + m + k / 3) - 6 = 18 :=
by
  intros h n m k h1
  sorry

end puzzle_solution_l2279_227996


namespace false_statement_l2279_227963

noncomputable def f (x : ℝ) : ℝ := (Real.cos x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x

def p : Prop := ∃ x0 : ℝ, f x0 = -1
def q : Prop := ∀ x : ℝ, f (2 * Real.pi + x) = f x

theorem false_statement : ¬ (p ∧ q) := sorry

end false_statement_l2279_227963


namespace circle_diameter_percentage_l2279_227925

theorem circle_diameter_percentage (d_R d_S : ℝ) 
    (h : π * (d_R / 2)^2 = 0.04 * π * (d_S / 2)^2) : 
    d_R = 0.4 * d_S :=
by
    sorry

end circle_diameter_percentage_l2279_227925


namespace computer_sale_price_percent_l2279_227940

theorem computer_sale_price_percent (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (discount3 : ℝ) :
  original_price = 500 ∧ discount1 = 0.25 ∧ discount2 = 0.10 ∧ discount3 = 0.05 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)) / original_price * 100 = 64.13 :=
by
  intro h
  sorry

end computer_sale_price_percent_l2279_227940


namespace Malou_first_quiz_score_l2279_227928

variable (score1 score2 score3 : ℝ)

theorem Malou_first_quiz_score (h1 : score1 = 90) (h2 : score2 = 92) (h_avg : (score1 + score2 + score3) / 3 = 91) : score3 = 91 := by
  sorry

end Malou_first_quiz_score_l2279_227928


namespace daniel_paid_more_l2279_227944

noncomputable def num_slices : ℕ := 10
noncomputable def plain_cost : ℕ := 10
noncomputable def truffle_extra_cost : ℕ := 5
noncomputable def total_cost : ℕ := plain_cost + truffle_extra_cost
noncomputable def cost_per_slice : ℝ := total_cost / num_slices

noncomputable def truffle_slices_cost : ℝ := 5 * cost_per_slice
noncomputable def plain_slices_cost : ℝ := 5 * cost_per_slice

noncomputable def daniel_cost : ℝ := 5 * cost_per_slice + 2 * cost_per_slice
noncomputable def carl_cost : ℝ := 3 * cost_per_slice

noncomputable def payment_difference : ℝ := daniel_cost - carl_cost

theorem daniel_paid_more : payment_difference = 6 :=
by 
  sorry

end daniel_paid_more_l2279_227944


namespace siblings_are_Emma_and_Olivia_l2279_227961

structure Child where
  name : String
  eyeColor : String
  hairColor : String
  ageGroup : String

def Bella := Child.mk "Bella" "Green" "Red" "Older"
def Derek := Child.mk "Derek" "Gray" "Red" "Younger"
def Olivia := Child.mk "Olivia" "Green" "Brown" "Older"
def Lucas := Child.mk "Lucas" "Gray" "Brown" "Younger"
def Emma := Child.mk "Emma" "Green" "Red" "Older"
def Ryan := Child.mk "Ryan" "Gray" "Red" "Older"
def Sophia := Child.mk "Sophia" "Green" "Brown" "Younger"
def Ethan := Child.mk "Ethan" "Gray" "Brown" "Older"

def sharesCharacteristics (c1 c2 : Child) : Nat :=
  (if c1.eyeColor = c2.eyeColor then 1 else 0) +
  (if c1.hairColor = c2.hairColor then 1 else 0) +
  (if c1.ageGroup = c2.ageGroup then 1 else 0)

theorem siblings_are_Emma_and_Olivia :
  sharesCharacteristics Bella Emma ≥ 2 ∧
  sharesCharacteristics Bella Olivia ≥ 2 ∧
  (sharesCharacteristics Bella Derek < 2) ∧
  (sharesCharacteristics Bella Lucas < 2) ∧
  (sharesCharacteristics Bella Ryan < 2) ∧
  (sharesCharacteristics Bella Sophia < 2) ∧
  (sharesCharacteristics Bella Ethan < 2) :=
by
  sorry

end siblings_are_Emma_and_Olivia_l2279_227961


namespace problem1_problem2_l2279_227984

def M := { x : ℝ | 0 < x ∧ x < 1 }

theorem problem1 :
  { x : ℝ | |2 * x - 1| < 1 } = M :=
by
  simp [M]
  sorry

theorem problem2 (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a * b + 1) > (a + b) :=
by
  simp [M] at ha hb
  sorry

end problem1_problem2_l2279_227984


namespace part1_part2_l2279_227920

variables (a b c : ℝ)

theorem part1 (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  ab + bc + ac ≤ 1 / 3 := sorry

theorem part2 (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  1 / a + 1 / b + 1 / c ≥ 9 := sorry

end part1_part2_l2279_227920


namespace correct_finance_specialization_l2279_227957

-- Variables representing percentages of students specializing in different subjects
variables (students : Type) -- Type of students
           (is_specializing_finance : students → Prop) -- Predicate for finance specialization
           (is_specializing_marketing : students → Prop) -- Predicate for marketing specialization

-- Given conditions
def finance_specialization_percentage : ℝ := 0.88 -- 88% of students are taking finance specialization
def marketing_specialization_percentage : ℝ := 0.76 -- 76% of students are taking marketing specialization

-- The proof statement
theorem correct_finance_specialization (h_finance : finance_specialization_percentage = 0.88) :
  finance_specialization_percentage = 0.88 :=
by
  sorry

end correct_finance_specialization_l2279_227957


namespace integer_roots_of_polynomial_l2279_227930

theorem integer_roots_of_polynomial : 
  {x : ℤ | x^3 - 4 * x^2 - 7 * x + 10 = 0} = {1, -2, 5} :=
by
  sorry

end integer_roots_of_polynomial_l2279_227930
