import Mathlib

namespace trajectory_of_point_l384_38402

theorem trajectory_of_point (P : ℝ × ℝ) 
  (h1 : dist P (0, 3) = dist P (x1, -3)) :
  ∃ p > 0, (P.fst)^2 = 2 * p * P.snd ∧ p = 6 :=
by {
  sorry
}

end trajectory_of_point_l384_38402


namespace total_spent_l384_38430

theorem total_spent (jayda_spent : ℝ) (haitana_spent : ℝ) (jayda_spent_eq : jayda_spent = 400) (aitana_more_than_jayda : haitana_spent = jayda_spent + (2/5) * jayda_spent) :
  jayda_spent + haitana_spent = 960 :=
by
  rw [jayda_spent_eq, aitana_more_than_jayda]
  -- Proof steps go here
  sorry

end total_spent_l384_38430


namespace music_player_and_concert_tickets_l384_38451

theorem music_player_and_concert_tickets (n : ℕ) (h1 : 35 % 5 = 0) (h2 : 35 % n = 0) (h3 : ∀ m : ℕ, m < 35 → (m % 5 ≠ 0 ∨ m % n ≠ 0)) : n = 7 :=
sorry

end music_player_and_concert_tickets_l384_38451


namespace area_of_similar_rectangle_l384_38441

theorem area_of_similar_rectangle:
  ∀ (R1 : ℝ → ℝ → Prop) (R2 : ℝ → ℝ → Prop),
  (∀ a b, R1 a b → a = 3 ∧ a * b = 18) →
  (∀ a b c d, R1 a b → R2 c d → c / d = a / b) →
  (∀ a b, R2 a b → a^2 + b^2 = 400) →
  ∃ areaR2, (∀ a b, R2 a b → a * b = areaR2) ∧ areaR2 = 160 :=
by
  intros R1 R2 hR1 h_similar h_diagonal
  use 160
  sorry

end area_of_similar_rectangle_l384_38441


namespace solve_for_x_l384_38415

theorem solve_for_x : ∀ x : ℝ, 3^(2 * x) = Real.sqrt 27 → x = 3 / 4 :=
by
  intro x h
  sorry

end solve_for_x_l384_38415


namespace distance_last_pair_of_trees_l384_38470

theorem distance_last_pair_of_trees 
  (yard_length : ℝ := 1200)
  (num_trees : ℕ := 117)
  (initial_distance : ℝ := 5)
  (distance_increment : ℝ := 2) :
  let num_distances := num_trees - 1
  let last_distance := initial_distance + (num_distances - 1) * distance_increment
  last_distance = 235 := by 
  sorry

end distance_last_pair_of_trees_l384_38470


namespace intersection_A_B_l384_38421

-- Define the conditions of set A and B using the given inequalities and constraints
def set_A : Set ℤ := {x | -2 < x ∧ x < 3}
def set_B : Set ℤ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the proof problem translating conditions and question to Lean
theorem intersection_A_B : (set_A ∩ set_B) = {0, 1, 2} := by
  sorry

end intersection_A_B_l384_38421


namespace intersection_eq_l384_38408

def U : Set ℝ := {x : ℝ | True}
def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x ≥ 1}
def CU_N : Set ℝ := {x : ℝ | x < 1}

theorem intersection_eq : M ∩ CU_N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_eq_l384_38408


namespace car_win_probability_l384_38493

noncomputable def P (n : ℕ) : ℚ := 1 / n

theorem car_win_probability :
  let P_x := 1 / 7
  let P_y := 1 / 3
  let P_z := 1 / 5
  P_x + P_y + P_z = 71 / 105 :=
by
  sorry

end car_win_probability_l384_38493


namespace combined_cost_is_3490_l384_38429

-- Definitions for the quantities of gold each person has and their respective prices per gram
def Gary_gold_grams : ℕ := 30
def Gary_gold_price_per_gram : ℕ := 15

def Anna_gold_grams : ℕ := 50
def Anna_gold_price_per_gram : ℕ := 20

def Lisa_gold_grams : ℕ := 40
def Lisa_gold_price_per_gram : ℕ := 18

def John_gold_grams : ℕ := 60
def John_gold_price_per_gram : ℕ := 22

-- Combined cost
def combined_cost : ℕ :=
  Gary_gold_grams * Gary_gold_price_per_gram +
  Anna_gold_grams * Anna_gold_price_per_gram +
  Lisa_gold_grams * Lisa_gold_price_per_gram +
  John_gold_grams * John_gold_price_per_gram

-- Proof that the combined cost is equal to $3490
theorem combined_cost_is_3490 : combined_cost = 3490 :=
  by
  -- proof skipped
  sorry

end combined_cost_is_3490_l384_38429


namespace graphs_intersection_count_l384_38449

theorem graphs_intersection_count (g : ℝ → ℝ) (hg : Function.Injective g) :
  ∃ S : Finset ℝ, (∀ x ∈ S, g (x^3) = g (x^5)) ∧ S.card = 3 :=
by
  sorry

end graphs_intersection_count_l384_38449


namespace f_9_over_2_l384_38438

noncomputable def f (x : ℝ) : ℝ := sorry -- The function f(x) is to be defined later according to conditions

theorem f_9_over_2 :
  (∀ x : ℝ, f (x + 1) = -f (-x + 1)) ∧ -- f(x+1) is odd
  (∀ x : ℝ, f (x + 2) = f (-x + 2)) ∧ -- f(x+2) is even
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = -2 * x^2 + 2) ∧ -- f(x) = ax^2 + b, where a = -2 and b = 2
  (f 0 + f 3 = 6) → -- Sum f(0) and f(3)
  f (9 / 2) = 5 / 2 := 
by {
  sorry -- The proof is omitted as per the instruction
}

end f_9_over_2_l384_38438


namespace find_sqrt_abc_sum_l384_38456

theorem find_sqrt_abc_sum (a b c : ℝ)
  (h1 : b + c = 17)
  (h2 : c + a = 18)
  (h3 : a + b = 19) :
  Real.sqrt (a * b * c * (a + b + c)) = 36 * Real.sqrt 15 := by
  sorry

end find_sqrt_abc_sum_l384_38456


namespace sum_powers_is_76_l384_38483

theorem sum_powers_is_76 (m n : ℕ) (h1 : m + n = 1) (h2 : m^2 + n^2 = 3)
                         (h3 : m^3 + n^3 = 4) (h4 : m^4 + n^4 = 7)
                         (h5 : m^5 + n^5 = 11) : m^9 + n^9 = 76 :=
sorry

end sum_powers_is_76_l384_38483


namespace find_norm_b_projection_of_b_on_a_l384_38468

open Real EuclideanSpace

noncomputable def a : ℝ := 4

noncomputable def angle_ab : ℝ := π / 4  -- 45 degrees in radians

noncomputable def inner_prod_condition (b : ℝ) : ℝ := 
  (1 / 2 * a) * (2 * a) + 
  (1 / 2 * a) * (-3 * b) + 
  b * (2 * a) + 
  b * (-3 * b) - 12

theorem find_norm_b (b : ℝ) (hb : inner_prod_condition b = 0) : b = sqrt 2 :=
  sorry

theorem projection_of_b_on_a (b : ℝ) (hb : inner_prod_condition b = 0) : 
  (b * cos angle_ab) = 1 :=
  sorry

end find_norm_b_projection_of_b_on_a_l384_38468


namespace remainder_problem_l384_38457

def rem (x y : ℚ) := x - y * (⌊x / y⌋ : ℤ)

theorem remainder_problem :
  let x := (5 : ℚ) / 9
  let y := -(3 : ℚ) / 7
  rem x y = (-19 : ℚ) / 63 :=
by
  let x := (5 : ℚ) / 9
  let y := -(3 : ℚ) / 7
  sorry

end remainder_problem_l384_38457


namespace inequality_proof_l384_38495

theorem inequality_proof (x y z : ℝ) (n : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h_sum : x + y + z = 1) :
  (x^4 / (y * (1 - y^n)) + y^4 / (z * (1 - z^n)) + z^4 / (x * (1 - x^n))) ≥ (3^n) / (3^(n+2) - 9) :=
by
  sorry

end inequality_proof_l384_38495


namespace investment_time_P_l384_38484

-- Variables and conditions
variables {x : ℕ} {time_P : ℕ}

-- Conditions as seen from the mathematical problem
def investment_P (x : ℕ) := 7 * x
def investment_Q (x : ℕ) := 5 * x
def profit_ratio := 1 / 2
def time_Q := 14

-- Statement of the problem
theorem investment_time_P : 
  (profit_ratio = (investment_P x * time_P) / (investment_Q x * time_Q)) → 
  time_P = 5 := 
sorry

end investment_time_P_l384_38484


namespace find_a_plus_b_l384_38439

-- Definitions for the conditions
variables {a b : ℝ} (i : ℂ)
def imaginary_unit : Prop := i * i = -1

-- Given condition
def given_equation (a b : ℝ) (i : ℂ) : Prop := (a + 2 * i) / i = b + i

-- Theorem statement
theorem find_a_plus_b (h1 : imaginary_unit i) (h2 : given_equation a b i) : a + b = 1 := 
sorry

end find_a_plus_b_l384_38439


namespace range_of_d_largest_S_n_l384_38443

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (d a_1 : ℝ)

-- Conditions
axiom a_3_eq_12 : a_n 3 = 12
axiom S_12_pos : S_n 12 > 0
axiom S_13_neg : S_n 13 < 0
axiom arithmetic_sequence : ∀ n, a_n n = a_1 + (n - 1) * d
axiom sum_of_terms : ∀ n, S_n n = n * a_1 + (n * (n - 1)) / 2 * d

-- Problems
theorem range_of_d : -24/7 < d ∧ d < -3 := sorry

theorem largest_S_n : (∀ m, m > 0 ∧ m < 13 → (S_n 6 >= S_n m)) := sorry

end range_of_d_largest_S_n_l384_38443


namespace function_relationship_profit_1200_max_profit_l384_38472

namespace SalesProblem

-- Define the linear relationship between sales quantity y and selling price x
def sales_quantity (x : ℝ) : ℝ := -2 * x + 160

-- Define the cost per item
def cost_per_item := 30

-- Define the profit given selling price x and quantity y
def profit (x : ℝ) (y : ℝ) : ℝ := (x - cost_per_item) * y

-- The given data points and conditions
def data_point_1 : (ℝ × ℝ) := (35, 90)
def data_point_2 : (ℝ × ℝ) := (40, 80)

-- Prove the linear relationship between y and x
theorem function_relationship : 
  sales_quantity data_point_1.1 = data_point_1.2 ∧ 
  sales_quantity data_point_2.1 = data_point_2.2 := 
  by sorry

-- Given daily profit of 1200, proves selling price should be 50 yuan
theorem profit_1200 (x : ℝ) (h₁ : 30 ≤ x ∧ x ≤ 54) 
  (h₂ : profit x (sales_quantity x) = 1200) : 
  x = 50 := 
  by sorry

-- Prove the maximum daily profit and corresponding selling price
theorem max_profit : 
  ∃ x, 30 ≤ x ∧ x ≤ 54 ∧ (∀ y, 30 ≤ y ∧ y ≤ 54 → profit y (sales_quantity y) ≤ profit x (sales_quantity x)) ∧ 
  profit x (sales_quantity x) = 1248 := 
  by sorry

end SalesProblem

end function_relationship_profit_1200_max_profit_l384_38472


namespace solution_set_inequality_l384_38404

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom increasing_on_positive : ∀ {x y : ℝ}, 0 < x → x < y → f x < f y
axiom f_one : f 1 = 0

theorem solution_set_inequality :
  {x : ℝ | (f x) / x < 0} = {x : ℝ | x < -1} ∪ {x | 0 < x ∧ x < 1} := sorry

end solution_set_inequality_l384_38404


namespace train_speed_identification_l384_38452

-- Define the conditions
def train_length : ℕ := 300
def crossing_time : ℕ := 30

-- Define the speed calculation
def calculate_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The target theorem stating the speed of the train
theorem train_speed_identification : calculate_speed train_length crossing_time = 10 := 
by 
  sorry

end train_speed_identification_l384_38452


namespace p_is_sufficient_not_necessary_for_q_l384_38487

-- Definitions for conditions p and q
def p (x : ℝ) := x^2 - x - 20 > 0
def q (x : ℝ) := 1 - x^2 < 0

-- The main statement
theorem p_is_sufficient_not_necessary_for_q:
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end p_is_sufficient_not_necessary_for_q_l384_38487


namespace bucket_water_total_l384_38496

theorem bucket_water_total (initial_gallons : ℝ) (added_gallons : ℝ) (total_gallons : ℝ) : 
  initial_gallons = 3 ∧ added_gallons = 6.8 → total_gallons = 9.8 :=
by
  { sorry }

end bucket_water_total_l384_38496


namespace toys_left_l384_38490

-- Given conditions
def initial_toys := 7
def sold_toys := 3

-- Proven statement
theorem toys_left : initial_toys - sold_toys = 4 := by
  sorry

end toys_left_l384_38490


namespace surface_area_of_sphere_given_cube_volume_8_l384_38425

theorem surface_area_of_sphere_given_cube_volume_8 
  (volume_of_cube : ℝ)
  (h₁ : volume_of_cube = 8) :
  ∃ (surface_area_of_sphere : ℝ), 
  surface_area_of_sphere = 12 * Real.pi :=
by
  sorry

end surface_area_of_sphere_given_cube_volume_8_l384_38425


namespace howard_items_l384_38420

theorem howard_items (a b c : ℕ) (h1 : a + b + c = 40) (h2 : 40 * a + 300 * b + 400 * c = 5000) : a = 20 :=
by
  sorry

end howard_items_l384_38420


namespace spaces_per_tray_l384_38494

-- Conditions
def num_ice_cubes_glass : ℕ := 8
def num_ice_cubes_pitcher : ℕ := 2 * num_ice_cubes_glass
def total_ice_cubes_used : ℕ := num_ice_cubes_glass + num_ice_cubes_pitcher
def num_trays : ℕ := 2

-- Proof statement
theorem spaces_per_tray : total_ice_cubes_used / num_trays = 12 :=
by
  sorry

end spaces_per_tray_l384_38494


namespace indeterminate_equation_solution_exists_l384_38413

theorem indeterminate_equation_solution_exists
  (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a * c = b^2 + b + 1) :
  ∃ x y : ℤ, a * x^2 - (2 * b + 1) * x * y + c * y^2 = 1 := by
  sorry

end indeterminate_equation_solution_exists_l384_38413


namespace amount_given_to_beggar_l384_38407

variable (X : ℕ)
variable (pennies_initial : ℕ := 42)
variable (pennies_to_farmer : ℕ := 22)
variable (pennies_after_farmer : ℕ := 20)

def amount_to_boy (X : ℕ) : ℕ :=
  (20 - X) / 2 + 3

theorem amount_given_to_beggar : 
  (X = 12) →  (pennies_initial - pennies_to_farmer - X) / 2 + 3 + 1 = pennies_initial - pennies_to_farmer - X :=
by
  intro h
  subst h
  sorry

end amount_given_to_beggar_l384_38407


namespace count_valid_n_l384_38461

theorem count_valid_n : 
  ∃ (count : ℕ), count = 88 ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 2000 ∧ 
   (∃ (a b : ℤ), a + b = -2 ∧ a * b = -n) ↔ 
   ∃ m, 1 ≤ m ∧ m ≤ 2000 ∧ (∃ a, a * (a + 2) = m)) := 
sorry

end count_valid_n_l384_38461


namespace divide_polynomials_l384_38462

theorem divide_polynomials (n : ℕ) (h : ∃ (k : ℤ), n^2 + 3*n + 51 = 13 * k) : 
  ∃ (m : ℤ), 21*n^2 + 89*n + 44 = 169 * m := by
  sorry

end divide_polynomials_l384_38462


namespace water_jugs_problem_l384_38444

-- Definitions based on the conditions
variables (m n : ℕ) (relatively_prime_m_n : Nat.gcd m n = 1)
variables (k : ℕ) (hk : 1 ≤ k ∧ k ≤ m + n)

-- Statement of the theorem
theorem water_jugs_problem : 
    ∃ (x y z : ℕ), 
    (x = m ∨ x = n ∨ x = m + n) ∧ 
    (y = m ∨ y = n ∨ y = m + n) ∧ 
    (z = m ∨ z = n ∨ z = m + n) ∧ 
    (x ≤ m + n) ∧ 
    (y ≤ m + n) ∧ 
    (z ≤ m + n) ∧ 
    x + y + z = m + n ∧ 
    (x = k ∨ y = k ∨ z = k) :=
sorry

end water_jugs_problem_l384_38444


namespace smallest_possible_N_l384_38475

theorem smallest_possible_N (table_size N : ℕ) (h_table_size : table_size = 72) :
  (∀ seating : Finset ℕ, (seating.card = N) → (seating ⊆ Finset.range table_size) →
    ∃ i ∈ Finset.range table_size, (seating = ∅ ∨ ∃ j, (j ∈ seating) ∧ (i = (j + 1) % table_size ∨ i = (j - 1) % table_size)))
  → N = 18 :=
by sorry

end smallest_possible_N_l384_38475


namespace find_k_l384_38445

theorem find_k (k : ℝ) (r s : ℝ) (h1 : r ≠ 0) (h2 : s ≠ 0)
  (h3 : r / s = 3) (h4 : r + s = 4) (h5 : r * s = k) : k = 3 :=
sorry

end find_k_l384_38445


namespace hyperbolas_same_asymptotes_l384_38485

theorem hyperbolas_same_asymptotes :
  (∀ x y, (x^2 / 4 - y^2 / 9 = 1) → (∃ k, y = k * x)) →
  (∀ x y, (y^2 / 18 - x^2 / N = 1) → (∃ k, y = k * x)) →
  N = 8 :=
by sorry

end hyperbolas_same_asymptotes_l384_38485


namespace find_g_1_l384_38453

noncomputable def g (x : ℝ) : ℝ := sorry -- express g(x) as a 4th degree polynomial with unknown coefficients

-- Conditions given in the problem
axiom cond1 : |g (-1)| = 15
axiom cond2 : |g (0)| = 15
axiom cond3 : |g (2)| = 15
axiom cond4 : |g (3)| = 15
axiom cond5 : |g (4)| = 15

-- The statement we need to prove
theorem find_g_1 : |g 1| = 11 :=
sorry

end find_g_1_l384_38453


namespace shaded_area_is_110_l384_38401

-- Definitions based on conditions
def equilateral_triangle_area : ℕ := 10
def num_triangles_small : ℕ := 1
def num_triangles_medium : ℕ := 3
def num_triangles_large : ℕ := 7

-- Total area calculation
def total_area : ℕ := (num_triangles_small + num_triangles_medium + num_triangles_large) * equilateral_triangle_area

-- The theorem statement
theorem shaded_area_is_110 : total_area = 110 := 
by 
  sorry

end shaded_area_is_110_l384_38401


namespace exponent_problem_l384_38497

variable {a m n : ℝ}

theorem exponent_problem (h1 : a^m = 2) (h2 : a^n = 3) : a^(3*m + 2*n) = 72 := 
  sorry

end exponent_problem_l384_38497


namespace inv_eq_self_l384_38458

noncomputable def g (m x : ℝ) : ℝ := (3 * x + 4) / (m * x - 3)

theorem inv_eq_self (m : ℝ) :
  (∀ x : ℝ, g m x = g m (g m x)) ↔ m ∈ Set.Iic (-9 / 4) ∪ Set.Ici (-9 / 4) :=
by
  sorry

end inv_eq_self_l384_38458


namespace eliot_account_balance_l384_38489

theorem eliot_account_balance 
  (A E : ℝ) 
  (h1 : A > E)
  (h2 : A - E = (1 / 12) * (A + E))
  (h3 : 1.10 * A = 1.20 * E + 20) : 
  E = 200 :=
by 
  sorry

end eliot_account_balance_l384_38489


namespace minimum_digits_for_divisibility_l384_38409

theorem minimum_digits_for_divisibility :
  ∃ n : ℕ, (10 * 2013 + n) % 2520 = 0 ∧ n < 1000 :=
sorry

end minimum_digits_for_divisibility_l384_38409


namespace probability_diff_colors_l384_38410

theorem probability_diff_colors (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  total_balls = 4 ∧ white_balls = 3 ∧ black_balls = 1 ∧ drawn_balls = 2 ∧ 
  total_outcomes = Nat.choose 4 2 ∧ favorable_outcomes = Nat.choose 3 1 * Nat.choose 1 1
  → favorable_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end probability_diff_colors_l384_38410


namespace grade3_trees_count_l384_38431

-- Declare the variables and types
variables (x y : ℕ)

-- Given conditions as definitions
def students_equation := (2 * x + y = 100)
def trees_equation := (9 * x + (13 / 2) * y = 566)
def avg_trees_grade3 := 4

-- Assert the problem statement
theorem grade3_trees_count (hx : students_equation x y) (hy : trees_equation x y) : 
  (avg_trees_grade3 * x = 84) :=
sorry

end grade3_trees_count_l384_38431


namespace ways_to_make_30_cents_is_17_l384_38498

-- Define the value of each type of coin
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the function that counts the number of ways to make 30 cents
def count_ways_to_make_30_cents : ℕ :=
  let ways_with_1_quarter := (if 30 - quarter_value == 5 then 2 else 0)
  let ways_with_0_quarters :=
    let ways_with_2_dimes := (if 30 - 2 * dime_value == 10 then 3 else 0)
    let ways_with_1_dime := (if 30 - dime_value == 20 then 5 else 0)
    let ways_with_0_dimes := (if 30 == 30 then 7 else 0)
    ways_with_2_dimes + ways_with_1_dime + ways_with_0_dimes
  2 + ways_with_1_quarter + ways_with_0_quarters

-- Proof statement
theorem ways_to_make_30_cents_is_17 : count_ways_to_make_30_cents = 17 := sorry

end ways_to_make_30_cents_is_17_l384_38498


namespace klinker_daughter_age_l384_38428

-- Define the conditions in Lean
variable (D : ℕ) -- ℕ is the natural number type in Lean

-- Define the theorem statement
theorem klinker_daughter_age (h1 : 35 + 15 = 50)
    (h2 : 50 = 2 * (D + 15)) : D = 10 := by
  sorry

end klinker_daughter_age_l384_38428


namespace marcus_savings_l384_38455

theorem marcus_savings
  (running_shoes_price : ℝ)
  (running_shoes_discount : ℝ)
  (cashback : ℝ)
  (running_shoes_tax_rate : ℝ)
  (athletic_socks_price : ℝ)
  (athletic_socks_tax_rate : ℝ)
  (bogo : ℝ)
  (performance_tshirt_price : ℝ)
  (performance_tshirt_discount : ℝ)
  (performance_tshirt_tax_rate : ℝ)
  (total_budget : ℝ)
  (running_shoes_final_price : ℝ)
  (athletic_socks_final_price : ℝ)
  (performance_tshirt_final_price : ℝ) :
  running_shoes_price = 120 →
  running_shoes_discount = 30 / 100 →
  cashback = 10 →
  running_shoes_tax_rate = 8 / 100 →
  athletic_socks_price = 25 →
  athletic_socks_tax_rate = 6 / 100 →
  bogo = 2 →
  performance_tshirt_price = 55 →
  performance_tshirt_discount = 10 / 100 →
  performance_tshirt_tax_rate = 7 / 100 →
  total_budget = 250 →
  running_shoes_final_price = (running_shoes_price * (1 - running_shoes_discount) - cashback) * (1 + running_shoes_tax_rate) →
  athletic_socks_final_price = (athletic_socks_price * bogo) * (1 + athletic_socks_tax_rate) / bogo →
  performance_tshirt_final_price = (performance_tshirt_price * (1 - performance_tshirt_discount)) * (1 + performance_tshirt_tax_rate) →
  total_budget - (running_shoes_final_price + athletic_socks_final_price + performance_tshirt_final_price) = 103.86 :=
sorry

end marcus_savings_l384_38455


namespace integer_solutions_range_l384_38460

def operation (p q : ℝ) : ℝ := p + q - p * q

theorem integer_solutions_range (m : ℝ) :
  (∃ (x1 x2 : ℤ), (operation 2 x1 > 0) ∧ (operation x1 3 ≤ m) ∧ (operation 2 x2 > 0) ∧ (operation x2 3 ≤ m) ∧ (x1 ≠ x2)) ↔ (3 ≤ m ∧ m < 5) :=
by sorry

end integer_solutions_range_l384_38460


namespace solution_set_of_inequality_l384_38411

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 - x + 2 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end solution_set_of_inequality_l384_38411


namespace ellipse_equation_l384_38424

theorem ellipse_equation (a b : ℝ) (x y : ℝ) (M : ℝ × ℝ)
  (h1 : 2 * a = 4)
  (h2 : 2 * b = 2 * a / 2)
  (h3 : M = (2, 1))
  (line_eq : ∀ k : ℝ, (y = 1 + k * (x - 2))) :
  (a = 2) ∧ (b = 1) ∧ (∀ (x y : ℝ), (x^2 / a^2 + y^2 / b^2 = 1) → (x^2 / 4 + y^2 = 1)) ∧
  (∃ k : ℝ, (k = -1/2) ∧ (∀ x y : ℝ, (y - 1 = k * (x - 2)) → (x + 2*y - 4 = 0))) :=
by
  sorry

end ellipse_equation_l384_38424


namespace chess_tournament_green_teams_l384_38477

theorem chess_tournament_green_teams :
  ∀ (R G total_teams : ℕ)
  (red_team_count : ℕ → ℕ)
  (green_team_count : ℕ → ℕ)
  (mixed_team_count : ℕ → ℕ),
  R = 64 → G = 68 → total_teams = 66 →
  red_team_count R = 20 →
  (R + G = 132) →
  -- Details derived from mixed_team_count and green_team_count
  -- are inferred from the conditions provided
  mixed_team_count R + red_team_count R = 32 → 
  -- Total teams by definition including mixed teams 
  mixed_team_count G = G - (2 * red_team_count R) - green_team_count G →
  green_team_count (G - (mixed_team_count R)) = 2 → 
  2 * (green_team_count G) = 22 :=
by sorry

end chess_tournament_green_teams_l384_38477


namespace time_to_pass_jogger_l384_38478

noncomputable def jogger_speed_kmh := 9 -- in km/hr
noncomputable def train_speed_kmh := 45 -- in km/hr
noncomputable def jogger_headstart_m := 240 -- in meters
noncomputable def train_length_m := 100 -- in meters

noncomputable def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

noncomputable def jogger_speed_mps := kmh_to_mps jogger_speed_kmh
noncomputable def train_speed_mps := kmh_to_mps train_speed_kmh
noncomputable def relative_speed := train_speed_mps - jogger_speed_mps
noncomputable def distance_to_be_covered := jogger_headstart_m + train_length_m

theorem time_to_pass_jogger : distance_to_be_covered / relative_speed = 34 := by
  sorry

end time_to_pass_jogger_l384_38478


namespace students_with_all_three_pets_correct_l384_38454

noncomputable def students_with_all_three_pets (total_students dog_owners cat_owners bird_owners dog_and_cat_owners cat_and_bird_owners dog_and_bird_owners : ℕ) : ℕ :=
  total_students - (dog_owners + cat_owners + bird_owners - dog_and_cat_owners - cat_and_bird_owners - dog_and_bird_owners)

theorem students_with_all_three_pets_correct : 
  students_with_all_three_pets 50 30 35 10 8 5 3 = 7 :=
by
  rw [students_with_all_three_pets]
  norm_num
  sorry

end students_with_all_three_pets_correct_l384_38454


namespace correct_choice_of_f_l384_38474

def f1 (x : ℝ) : ℝ := (x - 1)^2 + 3 * (x - 1)
def f2 (x : ℝ) : ℝ := 2 * (x - 1)
def f3 (x : ℝ) : ℝ := 2 * (x - 1)^2
def f4 (x : ℝ) : ℝ := x - 1

theorem correct_choice_of_f (h : (deriv f1 1 = 3) ∧ (deriv f2 1 ≠ 3) ∧ (deriv f3 1 ≠ 3) ∧ (deriv f4 1 ≠ 3)) : 
  ∀ f, (f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4) → (deriv f 1 = 3 → f = f1) :=
by sorry

end correct_choice_of_f_l384_38474


namespace bamboo_sections_length_l384_38412

variable {n d : ℕ} (a : ℕ → ℕ)
variable (h_arith : ∀ k, a (k + 1) = a k + d)
variable (h_top : a 1 = 10)
variable (h_sum_last_three : a n + a (n - 1) + a (n - 2) = 114)
variable (h_geom_6 : (a 6) ^ 2 = a 1 * a n)

theorem bamboo_sections_length : n = 16 := 
by 
  sorry

end bamboo_sections_length_l384_38412


namespace projection_of_c_onto_b_l384_38426

open Real

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := sqrt (b.1^2 + b.2^2)
  let scalar := dot_product / magnitude_b
  (scalar * b.1 / magnitude_b, scalar * b.2 / magnitude_b)

theorem projection_of_c_onto_b :
  let a := (2, 3)
  let b := (-4, 7)
  let c := (-a.1, -a.2)
  vector_projection c b = (-sqrt 65 / 5, -sqrt 65 / 5) :=
by sorry

end projection_of_c_onto_b_l384_38426


namespace exists_n_le_2500_perfect_square_l384_38423

noncomputable def sum_of_squares (n : ℕ) : ℚ :=
  (n * (n + 1) * (2 * n + 1)) / 6

noncomputable def sum_of_squares_segment (n : ℕ) : ℚ :=
  ((26 * n^3 + 12 * n^2 + n) / 3)

theorem exists_n_le_2500_perfect_square :
  ∃ (n : ℕ), n ≤ 2500 ∧ ∃ (k : ℚ), k^2 = (sum_of_squares n) * (sum_of_squares_segment n) :=
sorry

end exists_n_le_2500_perfect_square_l384_38423


namespace sum_possible_x_eq_16_5_l384_38480

open Real

noncomputable def sum_of_possible_x : Real :=
  let a := 2
  let b := -33
  let c := 87
  (-b) / (2 * a)

theorem sum_possible_x_eq_16_5 : sum_of_possible_x = 16.5 :=
  by
    -- The actual proof goes here
    sorry

end sum_possible_x_eq_16_5_l384_38480


namespace strictly_increasing_not_gamma_interval_gamma_interval_within_one_inf_l384_38476

def f (x : ℝ) : ℝ := -x * abs x + 2 * x

theorem strictly_increasing : ∃ A : Set ℝ, A = (Set.Ioo 0 1) ∧ (∀ x y, x ∈ A → y ∈ A → x < y → f x < f y) :=
  sorry

theorem not_gamma_interval : ¬(Set.Icc (1/2) (3/2) ⊆ Set.Ioo 0 1 ∧ 
  (∀ x ∈ Set.Icc (1/2) (3/2), f x ∈ Set.Icc (1/(3/2)) (1/(1/2)))) :=
  sorry

theorem gamma_interval_within_one_inf : ∃ m n : ℝ, 1 ≤ m ∧ m < n ∧ 
  Set.Icc m n = Set.Icc 1 ((1 + Real.sqrt 5) / 2) ∧ 
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (1/n) (1/m)) :=
  sorry

end strictly_increasing_not_gamma_interval_gamma_interval_within_one_inf_l384_38476


namespace cream_ratio_l384_38467

theorem cream_ratio (joe_initial_coffee joann_initial_coffee : ℝ)
                    (joe_drank_ounces joann_drank_ounces joe_added_cream joann_added_cream : ℝ) :
  joe_initial_coffee = 20 →
  joann_initial_coffee = 20 →
  joe_drank_ounces = 3 →
  joann_drank_ounces = 3 →
  joe_added_cream = 4 →
  joann_added_cream = 4 →
  (4 : ℝ) / ((21 / 24) * 24 - 3) = (8 : ℝ) / 7 :=
by
  intros h_ji h_ji h_jd h_jd h_jc h_jc
  sorry

end cream_ratio_l384_38467


namespace fraction_of_female_participants_is_correct_l384_38465

-- defining conditions
def last_year_males : ℕ := 30
def male_increase_rate : ℚ := 1.1
def female_increase_rate : ℚ := 1.25
def overall_increase_rate : ℚ := 1.2

-- the statement to prove
theorem fraction_of_female_participants_is_correct :
  ∀ (y : ℕ), 
  let males_this_year := last_year_males * male_increase_rate
  let females_this_year := y * female_increase_rate
  let total_last_year := last_year_males + y
  let total_this_year := total_last_year * overall_increase_rate
  total_this_year = males_this_year + females_this_year →
  (females_this_year / total_this_year) = (25 / 36) :=
by
  intros y
  let males_this_year := last_year_males * male_increase_rate
  let females_this_year := y * female_increase_rate
  let total_last_year := last_year_males + y
  let total_this_year := total_last_year * overall_increase_rate
  intro h
  sorry

end fraction_of_female_participants_is_correct_l384_38465


namespace plums_in_basket_l384_38447

theorem plums_in_basket (initial : ℕ) (added : ℕ) (total : ℕ) (h_initial : initial = 17) (h_added : added = 4) : total = 21 := by
  sorry

end plums_in_basket_l384_38447


namespace acuteAnglesSum_l384_38416

theorem acuteAnglesSum (A B C : ℝ) (hA : 0 < A ∧ A < π / 2) (hB : 0 < B ∧ B < π / 2) 
  (hC : 0 < C ∧ C < π / 2) (h : Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 = 1) :
  π / 2 ≤ A + B + C ∧ A + B + C ≤ π :=
by
  sorry

end acuteAnglesSum_l384_38416


namespace total_money_l384_38433

variable (Sally Jolly Molly : ℕ)

-- Conditions
def condition1 (Sally : ℕ) : Prop := Sally - 20 = 80
def condition2 (Jolly : ℕ) : Prop := Jolly + 20 = 70
def condition3 (Molly : ℕ) : Prop := Molly + 30 = 100

-- The theorem to prove
theorem total_money (h1: condition1 Sally)
                    (h2: condition2 Jolly)
                    (h3: condition3 Molly) :
  Sally + Jolly + Molly = 220 :=
by
  sorry

end total_money_l384_38433


namespace arithmetic_progression_square_l384_38422

theorem arithmetic_progression_square (a b c : ℝ) (h : b - a = c - b) : a^2 + 8 * b * c = (2 * b + c)^2 := 
by
  sorry

end arithmetic_progression_square_l384_38422


namespace day_of_week_306_2003_l384_38434

-- Note: Definitions to support the conditions and the proof
def day_of_week (n : ℕ) : ℕ := n % 7

-- Theorem statement: Given conditions lead to the conclusion that the 306th day of the year 2003 falls on a Sunday
theorem day_of_week_306_2003 :
  (day_of_week (15) = 2) → (day_of_week (306) = 0) :=
by sorry

end day_of_week_306_2003_l384_38434


namespace part1_condition_represents_line_part2_slope_does_not_exist_part3_x_intercept_part4_angle_condition_l384_38466

theorem part1_condition_represents_line (m : ℝ) :
  (m^2 - 2 * m - 3 ≠ 0) ∧ (2 * m^2 + m - 1 ≠ 0) ↔ m ≠ -1 :=
sorry

theorem part2_slope_does_not_exist (m : ℝ) :
  (m = 1 / 2) ↔ (m^2 - 2 * m - 3 = 0 ∧ (2 * m^2 + m - 1 = 0) ∧ ((1 * x = (4 / 3)))) :=
sorry

theorem part3_x_intercept (m : ℝ) :
  (2 * m - 6) / (m^2 - 2 * m - 3) = -3 ↔ m = -5 / 3 :=
sorry

theorem part4_angle_condition (m : ℝ) :
  -((m^2 - 2 * m - 3) / (2 * m^2 + m - 1)) = 1 ↔ m = 4 / 3 :=
sorry

end part1_condition_represents_line_part2_slope_does_not_exist_part3_x_intercept_part4_angle_condition_l384_38466


namespace total_volume_of_five_boxes_l384_38486

-- Define the edge length of each cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_cube (s : ℕ) : ℕ := s ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 5

-- Define the total volume
def total_volume (s : ℕ) (n : ℕ) : ℕ := n * (volume_of_cube s)

-- The theorem to prove
theorem total_volume_of_five_boxes :
  total_volume edge_length number_of_cubes = 625 := 
by
  -- Proof is skipped
  sorry

end total_volume_of_five_boxes_l384_38486


namespace derivative_at_zero_l384_38450

def f (x : ℝ) : ℝ := (x + 1)^4

theorem derivative_at_zero : deriv f 0 = 4 :=
by
  sorry

end derivative_at_zero_l384_38450


namespace height_of_Joaos_salary_in_kilometers_l384_38463

def real_to_cruzados (reais: ℕ) : ℕ := reais * 2750000000

def stacks (cruzados: ℕ) : ℕ := cruzados / 100

def height_in_cm (stacks: ℕ) : ℕ := stacks * 15

noncomputable def height_in_km (height_cm: ℕ) : ℕ := height_cm / 100000

theorem height_of_Joaos_salary_in_kilometers :
  height_in_km (height_in_cm (stacks (real_to_cruzados 640))) = 264000 :=
by
  sorry

end height_of_Joaos_salary_in_kilometers_l384_38463


namespace isosceles_triangle_side_length_l384_38446

theorem isosceles_triangle_side_length (n : ℕ) : 
  (∃ a b : ℕ, a ≠ 4 ∧ b ≠ 4 ∧ (a = b ∨ a = 4 ∨ b = 4) ∧ 
  a^2 - 6*a + n = 0 ∧ b^2 - 6*b + n = 0) → 
  (n = 8 ∨ n = 9) := 
by
  sorry

end isosceles_triangle_side_length_l384_38446


namespace determine_m_even_function_l384_38464

theorem determine_m_even_function (m : ℤ) :
  (∀ x : ℤ, (x^2 + (m-1)*x) = (x^2 - (m-1)*x)) → m = 1 :=
by
    sorry

end determine_m_even_function_l384_38464


namespace odd_function_example_l384_38432

theorem odd_function_example (f : ℝ → ℝ)
    (h_odd : ∀ x, f (-x) = -f x)
    (h_neg : ∀ x, x < 0 → f x = x + 2) : f 0 + f 3 = 1 :=
by
  sorry

end odd_function_example_l384_38432


namespace chinese_horses_problem_l384_38481

variables (x y : ℕ)

theorem chinese_horses_problem (h1 : x + y = 100) (h2 : 3 * x + (y / 3) = 100) :
  (x + y = 100) ∧ (3 * x + (y / 3) = 100) :=
by
  sorry

end chinese_horses_problem_l384_38481


namespace only_solution_is_2_3_7_l384_38492

theorem only_solution_is_2_3_7 (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h4 : c ∣ (a * b + 1)) (h5 : a ∣ (b * c + 1)) (h6 : b ∣ (c * a + 1)) :
  (a = 2 ∧ b = 3 ∧ c = 7) ∨ (a = 3 ∧ b = 7 ∧ c = 2) ∨ (a = 7 ∧ b = 2 ∧ c = 3) ∨
  (a = 2 ∧ b = 7 ∧ c = 3) ∨ (a = 7 ∧ b = 3 ∧ c = 2) ∨ (a = 3 ∧ b = 2 ∧ c = 7) :=
  sorry

end only_solution_is_2_3_7_l384_38492


namespace valid_license_plates_count_l384_38491

theorem valid_license_plates_count :
  let letters := 26 * 26 * 26
  let digits := 9 * 10 * 10
  letters * digits = 15818400 :=
by
  sorry

end valid_license_plates_count_l384_38491


namespace race_distance_100_l384_38473

noncomputable def race_distance (a b c d : ℝ) :=
  (d / a = (d - 20) / b) ∧
  (d / b = (d - 10) / c) ∧
  (d / a = (d - 28) / c) 

theorem race_distance_100 (a b c d : ℝ) (h1 : d / a = (d - 20) / b) (h2 : d / b = (d - 10) / c) (h3 : d / a = (d - 28) / c) : 
  d = 100 :=
  sorry

end race_distance_100_l384_38473


namespace size_of_third_file_l384_38499

theorem size_of_third_file 
  (s : ℝ) (t : ℝ) (f1 : ℝ) (f2 : ℝ) (f3 : ℝ) 
  (h1 : s = 2) (h2 : t = 120) (h3 : f1 = 80) (h4 : f2 = 90) : 
  f3 = s * t - (f1 + f2) :=
by
  sorry

end size_of_third_file_l384_38499


namespace number_of_white_cats_l384_38459

theorem number_of_white_cats (total_cats : ℕ) (percent_black : ℤ) (grey_cats : ℕ) : 
  total_cats = 16 → 
  percent_black = 25 →
  grey_cats = 10 → 
  (total_cats - (total_cats * percent_black / 100 + grey_cats)) = 2 :=
by
  intros
  sorry

end number_of_white_cats_l384_38459


namespace positive_value_of_m_l384_38479

variable {m : ℝ}

theorem positive_value_of_m (h : ∃ x : ℝ, (3 * x^2 + m * x + 36) = 0 ∧ (∀ y : ℝ, (3 * y^2 + m * y + 36) = 0 → y = x)) :
  m = 12 * Real.sqrt 3 :=
sorry

end positive_value_of_m_l384_38479


namespace max_value_expression_l384_38418

theorem max_value_expression (a b c : ℝ) 
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  2 * a * b * Real.sqrt 3 + 2 * b * c ≤ 2 :=
sorry

end max_value_expression_l384_38418


namespace initial_number_correct_l384_38414

-- Define the relevant values
def x : ℝ := 53.33
def initial_number : ℝ := 319.98

-- Define the conditions in Lean with appropriate constraints
def conditions (n : ℝ) (x : ℝ) : Prop :=
  x = n / 2 / 3

-- Theorem stating that 319.98 divided by 2 and then by 3 results in 53.33
theorem initial_number_correct : conditions initial_number x :=
by
  unfold conditions
  sorry

end initial_number_correct_l384_38414


namespace distance_between_fourth_and_work_l384_38417

theorem distance_between_fourth_and_work (x : ℝ) (h₁ : x > 0) :
  let total_distance := x + 0.5 * x + 2 * x
  let to_fourth := (1 / 3) * total_distance
  let total_to_fourth := total_distance + to_fourth
  3 * total_to_fourth = 14 * x :=
by
  sorry

end distance_between_fourth_and_work_l384_38417


namespace problem_statement_l384_38405

theorem problem_statement (f : ℕ → ℕ) (h1 : f 1 = 4) (h2 : ∀ x y, f (x + y) = f x + f y + 7 * x * y + 4) :
  f 2 + f 5 = 125 :=
by
  sorry

end problem_statement_l384_38405


namespace least_multiple_of_seven_not_lucky_is_14_l384_38436

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n > 0 ∧ n % sum_of_digits n = 0

def is_multiple_of_seven_not_lucky (n : ℕ) : Prop :=
  n % 7 = 0 ∧ ¬ is_lucky_integer n

theorem least_multiple_of_seven_not_lucky_is_14 : 
  ∃ n : ℕ, is_multiple_of_seven_not_lucky n ∧ ∀ m, (is_multiple_of_seven_not_lucky m → n ≤ m) :=
⟨ 14, 
  by {
    -- Proof is provided here, but for now, we use "sorry"
    sorry
  }⟩

end least_multiple_of_seven_not_lucky_is_14_l384_38436


namespace chuck_total_time_on_trip_l384_38471

def distance_into_country : ℝ := 28.8
def rate_out : ℝ := 16
def rate_back : ℝ := 24

theorem chuck_total_time_on_trip : (distance_into_country / rate_out) + (distance_into_country / rate_back) = 3 := 
by sorry

end chuck_total_time_on_trip_l384_38471


namespace isosceles_triangle_perimeter_l384_38488

-- Define the lengths of the sides
def side1 := 2 -- 2 cm
def side2 := 4 -- 4 cm

-- Define the condition of being isosceles
def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (a = c) ∨ (b = c)

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Define the triangle perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the main theorem to prove
theorem isosceles_triangle_perimeter {a b : ℝ} (ha : a = side1) (hb : b = side2)
    (h1 : is_isosceles a b c) (h2 : triangle_inequality a b c) : perimeter a b c = 10 :=
sorry

end isosceles_triangle_perimeter_l384_38488


namespace problem1_problem2_l384_38442

noncomputable def vec (α : ℝ) (β : ℝ) : ℝ × ℝ × ℝ × ℝ :=
  (Real.cos α, Real.sin α, Real.cos β, -Real.sin β)

theorem problem1 (α β : ℝ) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : (Real.sqrt ((Real.cos α - Real.cos β) ^ 2 + (Real.sin α + Real.sin β) ^ 2)) = (Real.sqrt 10) / 5) :
  Real.cos (α + β) = 4 / 5 :=
by
  sorry

theorem problem2 (α β : ℝ) (h1 : 0 < α ∧ α < Real.pi / 2) (h2 : 0 < β ∧ β < Real.pi / 2)
  (h3 : Real.cos α = 3 / 5) (h4 : Real.cos (α + β) = 4 / 5) :
  Real.cos β = 24 / 25 :=
by
  sorry

end problem1_problem2_l384_38442


namespace symmetric_points_y_axis_l384_38419

theorem symmetric_points_y_axis (a b : ℝ) (h₁ : ∃ M N : ℝ × ℝ, M = (a, 3) ∧ N = (4, b) ∧ M.1 = -N.1 ∧ M.2 = N.2) :
  (a + b) ^ 2012 = 1 :=
by 
  sorry

end symmetric_points_y_axis_l384_38419


namespace fencing_required_l384_38469

theorem fencing_required (L : ℝ) (W : ℝ) (A : ℝ) (H1 : L = 20) (H2 : A = 720) 
  (H3 : A = L * W) : L + 2 * W = 92 := by 
{
  sorry
}

end fencing_required_l384_38469


namespace max_distance_circle_ellipse_l384_38427

theorem max_distance_circle_ellipse :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 2}
  let ellipse := {p : ℝ × ℝ | p.1^2 / 10 + p.2^2 = 1}
  ∀ (P Q : ℝ × ℝ), P ∈ circle → Q ∈ ellipse → 
  dist P Q ≤ 6 * Real.sqrt 2 :=
by
  intro circle ellipse P Q hP hQ
  sorry

end max_distance_circle_ellipse_l384_38427


namespace units_digit_uniform_l384_38435

-- Definitions
def domain : Finset ℕ := Finset.range 15

def pick : Type := { n // n ∈ domain }

def uniform_pick : pick := sorry

-- Statement of the theorem
theorem units_digit_uniform :
  ∀ (J1 J2 K : pick), 
  ∃ d : ℕ, d < 10 ∧ (J1.val + J2.val + K.val) % 10 = d
:= sorry

end units_digit_uniform_l384_38435


namespace compute_modulo_l384_38448

theorem compute_modulo :
    (2015 % 7) = 3 ∧ (2016 % 7) = 4 ∧ (2017 % 7) = 5 ∧ (2018 % 7) = 6 →
    (2015 * 2016 * 2017 * 2018) % 7 = 3 :=
by
  intros h
  have h1 := h.left
  have h2 := h.right.left
  have h3 := h.right.right.left
  have h4 := h.right.right.right
  sorry

end compute_modulo_l384_38448


namespace polygon_diagonals_regions_l384_38440

theorem polygon_diagonals_regions (n : ℕ) (hn : n ≥ 3) :
  let D := n * (n - 3) / 2
  let P := n * (n - 1) * (n - 2) * (n - 3) / 24
  let R := D + P + 1
  R = n * (n - 1) * (n - 2) * (n - 3) / 24 + n * (n - 3) / 2 + 1 :=
by
  sorry

end polygon_diagonals_regions_l384_38440


namespace find_x_l384_38406

theorem find_x (x y : ℝ) (h₁ : x - y = 10) (h₂ : x + y = 14) : x = 12 :=
by
  sorry

end find_x_l384_38406


namespace parabola_directrix_l384_38482

theorem parabola_directrix (x : ℝ) (y : ℝ) (h : y = -4 * x ^ 2 - 3) : y = - 49 / 16 := sorry

end parabola_directrix_l384_38482


namespace infinitely_many_a_l384_38437

theorem infinitely_many_a (n : ℕ) : ∃ (a : ℕ), ∃ (k : ℕ), ∀ n : ℕ, n^6 + 3 * (3 * n^4 * k + 9 * n^2 * k^2 + 9 * k^3) = (n^2 + 3 * k)^3 :=
by
  sorry

end infinitely_many_a_l384_38437


namespace two_digit_ab_divisible_by_11_13_l384_38403

theorem two_digit_ab_divisible_by_11_13 (a b : ℕ) (h1 : 0 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10^5 * 2 + 10^4 * 0 + 10^3 * 1 + 10^2 * a + 10 * b + 7) % 11 = 0)
  (h4 : (10^5 * 2 + 10^4 * 0 + 10^3 * 1 + 10^2 * a + 10 * b + 7) % 13 = 0) :
  10 * a + b = 48 :=
sorry

end two_digit_ab_divisible_by_11_13_l384_38403


namespace complement_U_M_l384_38400

noncomputable def U : Set ℝ := {x : ℝ | x > 0}

noncomputable def M : Set ℝ := {x : ℝ | 2 * x - x^2 > 0}

theorem complement_U_M : (U \ M) = {x : ℝ | x ≥ 2} := 
by
  sorry

end complement_U_M_l384_38400
