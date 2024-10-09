import Mathlib

namespace sufficient_but_not_necessary_condition_not_necessary_condition_l2039_203926

variable (x : ℝ)

theorem sufficient_but_not_necessary_condition (h1 : x < -1) : 2 * x ^ 2 + x - 1 > 0 :=
by sorry

theorem not_necessary_condition (h2 : 2 * x ^ 2 + x - 1 > 0) : x > 1/2 ∨ x < -1 :=
by sorry

end sufficient_but_not_necessary_condition_not_necessary_condition_l2039_203926


namespace lemonade_quarts_l2039_203929

theorem lemonade_quarts (total_parts water_parts lemon_juice_parts : ℕ) (total_gallons gallons_to_quarts : ℚ) 
  (h_ratio : water_parts = 4) (h_ratio_lemon : lemon_juice_parts = 1) (h_total_parts : total_parts = water_parts + lemon_juice_parts)
  (h_total_gallons : total_gallons = 1) (h_gallons_to_quarts : gallons_to_quarts = 4) :
  let volume_per_part := total_gallons / total_parts
  let volume_per_part_quarts := volume_per_part * gallons_to_quarts
  let water_volume := water_parts * volume_per_part_quarts
  water_volume = 16 / 5 :=
by
  sorry

end lemonade_quarts_l2039_203929


namespace sin_squared_minus_cos_squared_l2039_203984

theorem sin_squared_minus_cos_squared {α : ℝ} (h : Real.sin α = Real.sqrt 5 / 5) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 = -3 / 5 :=
by
  sorry -- Proof is omitted

end sin_squared_minus_cos_squared_l2039_203984


namespace hex_B1C_base10_l2039_203995

theorem hex_B1C_base10 : (11 * 16^2 + 1 * 16^1 + 12 * 16^0) = 2844 :=
by
  sorry

end hex_B1C_base10_l2039_203995


namespace max_value_of_expr_l2039_203960

theorem max_value_of_expr (x : ℝ) (h : x ≠ 0) : 
  (∀ y : ℝ, y = (x^2) / (x^6 - 2*x^5 - 2*x^4 + 4*x^3 + 4*x^2 + 16) → y ≤ 1/8) :=
sorry

end max_value_of_expr_l2039_203960


namespace exists_odd_k_l_m_l2039_203935

def odd_nat (n : ℕ) : Prop := n % 2 = 1

theorem exists_odd_k_l_m : 
  ∃ (k l m : ℕ), 
  odd_nat k ∧ odd_nat l ∧ odd_nat m ∧ 
  (k ≠ 0) ∧ (l ≠ 0) ∧ (m ≠ 0) ∧ 
  (1991 * (l * m + k * m + k * l) = k * l * m) :=
by
  sorry

end exists_odd_k_l_m_l2039_203935


namespace worked_days_proof_l2039_203949

theorem worked_days_proof (W N : ℕ) (hN : N = 24) (h0 : 100 * W = 25 * N) : W + N = 30 :=
by
  sorry

end worked_days_proof_l2039_203949


namespace scout_troop_profit_l2039_203940

noncomputable def candy_profit (purchase_bars purchase_rate sell_bars sell_rate donation_fraction : ℕ) : ℕ :=
  let cost_price_per_bar := purchase_rate / purchase_bars
  let total_cost := purchase_bars * cost_price_per_bar
  let effective_cost := total_cost * donation_fraction
  let sell_price_per_bar := sell_rate / sell_bars
  let total_revenue := purchase_bars * sell_price_per_bar
  total_revenue - effective_cost

theorem scout_troop_profit :
  candy_profit 1200 3 4 3 1/2 = 700 := by
  sorry

end scout_troop_profit_l2039_203940


namespace quadratic_inequality_solution_l2039_203978

theorem quadratic_inequality_solution (a b: ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 + a * x + b < 0) ∧
  (2 + 3 = -a) ∧
  (2 * 3 = b) →
  ∀ x : ℝ, (b * x^2 + a * x + 1 > 0) ↔ (x < 1/3 ∨ x > 1/2) :=
by
  sorry

end quadratic_inequality_solution_l2039_203978


namespace teams_points_l2039_203922

-- Definitions of teams and points
inductive Team
| A | B | C | D | E
deriving DecidableEq

def points : Team → ℕ
| Team.A => 6
| Team.B => 5
| Team.C => 4
| Team.D => 3
| Team.E => 2

-- Conditions
axiom no_draws_A : ∀ t : Team, t ≠ Team.A → (points Team.A ≠ points t)
axiom no_loses_B : ∀ t : Team, t ≠ Team.B → (points Team.B > points t) ∨ (points Team.B = points t)
axiom no_wins_D : ∀ t : Team, t ≠ Team.D → (points Team.D < points t)
axiom unique_scores : ∀ (t1 t2 : Team), t1 ≠ t2 → points t1 ≠ points t2

-- Theorem
theorem teams_points :
  points Team.A = 6 ∧
  points Team.B = 5 ∧
  points Team.C = 4 ∧
  points Team.D = 3 ∧
  points Team.E = 2 :=
by
  sorry

end teams_points_l2039_203922


namespace number_of_pages_in_contract_l2039_203901

theorem number_of_pages_in_contract (total_pages_copied : ℕ) (copies_per_person : ℕ) (number_of_people : ℕ)
  (h1 : total_pages_copied = 360) (h2 : copies_per_person = 2) (h3 : number_of_people = 9) :
  total_pages_copied / (copies_per_person * number_of_people) = 20 :=
by
  sorry

end number_of_pages_in_contract_l2039_203901


namespace g_eval_l2039_203932

-- Define the function g
def g (a : ℚ) (b : ℚ) (c : ℚ) : ℚ := (2 * a + b) / (c - a)

-- Theorem to prove g(2, 4, -1) = -8 / 3
theorem g_eval :
  g 2 4 (-1) = -8 / 3 := 
by
  sorry

end g_eval_l2039_203932


namespace parabola_hyperbola_focus_vertex_l2039_203941

theorem parabola_hyperbola_focus_vertex (p : ℝ) : 
  (∃ (focus_vertex : ℝ × ℝ), focus_vertex = (2, 0) 
    ∧ focus_vertex = (p / 2, 0)) → p = 4 :=
by
  sorry

end parabola_hyperbola_focus_vertex_l2039_203941


namespace distinct_numbers_in_T_l2039_203993

-- Definitions of sequences as functions
def seq1 (k: ℕ) : ℕ := 5 * k - 3
def seq2 (l: ℕ) : ℕ := 8 * l - 5

-- Definition of sets A and B
def A : Finset ℕ := Finset.image seq1 (Finset.range 3000)
def B : Finset ℕ := Finset.image seq2 (Finset.range 3000)

-- Definition of set T as the union of A and B
def T := A ∪ B

-- Proof statement
theorem distinct_numbers_in_T : T.card = 5400 := by
  sorry

end distinct_numbers_in_T_l2039_203993


namespace Taimour_paint_time_l2039_203986

theorem Taimour_paint_time (T : ℝ) (H1 : ∀ t : ℝ, t = 2 / T → t ≠ 0) (H2 : (1 / T + 2 / T) = 1 / 3) : T = 9 :=
by
  sorry

end Taimour_paint_time_l2039_203986


namespace judy_expense_correct_l2039_203966

noncomputable def judy_expense : ℝ :=
  let carrots := 5 * 1
  let milk := 3 * 3
  let pineapples := 2 * 4
  let original_flour_price := 5
  let discount := original_flour_price * 0.25
  let discounted_flour_price := original_flour_price - discount
  let flour := 2 * discounted_flour_price
  let ice_cream := 7
  let total_no_coupon := carrots + milk + pineapples + flour + ice_cream
  if total_no_coupon >= 30 then total_no_coupon - 10 else total_no_coupon

theorem judy_expense_correct : judy_expense = 26.5 := by
  sorry

end judy_expense_correct_l2039_203966


namespace length_of_each_train_l2039_203948

theorem length_of_each_train
  (L : ℝ) -- length of each train
  (speed_fast : ℝ) (speed_slow : ℝ) -- speeds of the fast and slow trains in km/hr
  (time_pass : ℝ) -- time for the slower train to pass the driver of the faster one in seconds
  (h_speed_fast : speed_fast = 45) -- speed of the faster train
  (h_speed_slow : speed_slow = 15) -- speed of the slower train
  (h_time_pass : time_pass = 60) -- time to pass
  (h_same_length : ∀ (x y : ℝ), x = y → x = L) :  
  L = 1000 :=
  by
  -- Skipping the proof as instructed
  sorry

end length_of_each_train_l2039_203948


namespace number_of_zeros_of_f_l2039_203905

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 6

theorem number_of_zeros_of_f : ∃! x : ℝ, 0 < x ∧ f x = 0 :=
sorry

end number_of_zeros_of_f_l2039_203905


namespace avg_eggs_per_nest_l2039_203942

/-- In the Caribbean, loggerhead turtles lay three million eggs in twenty thousand nests. 
On average, show that there are 150 eggs in each nest. -/

theorem avg_eggs_per_nest 
  (total_eggs : ℕ) 
  (total_nests : ℕ) 
  (h1 : total_eggs = 3000000) 
  (h2 : total_nests = 20000) :
  total_eggs / total_nests = 150 := 
by {
  sorry
}

end avg_eggs_per_nest_l2039_203942


namespace line_is_x_axis_l2039_203976

theorem line_is_x_axis (A B C : ℝ) (h : ∀ x : ℝ, A * x + B * 0 + C = 0) : A = 0 ∧ B ≠ 0 ∧ C = 0 :=
by sorry

end line_is_x_axis_l2039_203976


namespace point_on_x_axis_l2039_203908

theorem point_on_x_axis (m : ℝ) (h : m - 2 = 0) :
  (m + 3, m - 2) = (5, 0) :=
by
  sorry

end point_on_x_axis_l2039_203908


namespace sequence_a_n_l2039_203972

theorem sequence_a_n (a : ℕ → ℕ) (h₁ : a 1 = 1)
(h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = a (n / 2) + a ((n + 1) / 2)) :
∀ n : ℕ, a n = n :=
by
  -- skip the proof with sorry
  sorry

end sequence_a_n_l2039_203972


namespace difference_value_l2039_203961

theorem difference_value (N : ℝ) (h : 0.25 * N = 100) : N - (3/4) * N = 100 :=
by sorry

end difference_value_l2039_203961


namespace number_of_books_in_box_l2039_203958

theorem number_of_books_in_box :
  ∀ (total_weight : ℕ) (empty_box_weight : ℕ) (book_weight : ℕ),
  total_weight = 42 →
  empty_box_weight = 6 →
  book_weight = 3 →
  (total_weight - empty_box_weight) / book_weight = 12 :=
by
  intros total_weight empty_box_weight book_weight htwe hebe hbw
  sorry

end number_of_books_in_box_l2039_203958


namespace quadratic_one_root_iff_discriminant_zero_l2039_203965

theorem quadratic_one_root_iff_discriminant_zero (m : ℝ) : 
  (∃ x : ℝ, ∀ y : ℝ, y^2 - m*y + 1 ≤ 0 ↔ y = x) ↔ (m = 2 ∨ m = -2) :=
by 
  -- We assume the discriminant condition which implies the result
  sorry

end quadratic_one_root_iff_discriminant_zero_l2039_203965


namespace henry_wins_l2039_203936

-- Definitions of conditions
def total_games : ℕ := 14
def losses : ℕ := 2
def draws : ℕ := 10

-- Statement of the theorem
theorem henry_wins : (total_games - losses - draws) = 2 :=
by
  -- Proof goes here
  sorry

end henry_wins_l2039_203936


namespace ratio_of_ripe_mangoes_l2039_203999

theorem ratio_of_ripe_mangoes (total_mangoes : ℕ) (unripe_two_thirds : ℚ)
  (kept_unripe_mangoes : ℕ) (mangoes_per_jar : ℕ) (jars_made : ℕ)
  (h1 : total_mangoes = 54)
  (h2 : unripe_two_thirds = 2 / 3)
  (h3 : kept_unripe_mangoes = 16)
  (h4 : mangoes_per_jar = 4)
  (h5 : jars_made = 5) :
  1 / 3 = 18 / 54 :=
sorry

end ratio_of_ripe_mangoes_l2039_203999


namespace binary_to_decimal_1010101_l2039_203912

def bin_to_dec (bin : List ℕ) (len : ℕ): ℕ :=
  List.foldl (λ acc (digit, idx) => acc + digit * 2^idx) 0 (List.zip bin (List.range len))

theorem binary_to_decimal_1010101 : bin_to_dec [1, 0, 1, 0, 1, 0, 1] 7 = 85 :=
by
  simp [bin_to_dec, List.range, List.zip]
  -- Detailed computation can be omitted and sorry used here if necessary
  sorry

end binary_to_decimal_1010101_l2039_203912


namespace fg_of_3_eq_29_l2039_203979

def g (x : ℝ) : ℝ := x^2
def f (x : ℝ) : ℝ := 3 * x + 2

theorem fg_of_3_eq_29 : f (g 3) = 29 := by
  sorry

end fg_of_3_eq_29_l2039_203979


namespace b_sequence_is_constant_l2039_203928

noncomputable def b_sequence_formula (a b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → ∃ d q : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ (∀ n : ℕ, b (n + 1) = b n * q)) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) / a n = b n) ∧
  (∀ n : ℕ, n > 0 → b n = 1)

theorem b_sequence_is_constant (a b : ℕ → ℝ) (h : b_sequence_formula a b) : ∀ n : ℕ, n > 0 → b n = 1 :=
  by
    sorry

end b_sequence_is_constant_l2039_203928


namespace find_number_l2039_203921

theorem find_number : ∃ (x : ℝ), x + 0.303 + 0.432 = 5.485 ↔ x = 4.750 := 
sorry

end find_number_l2039_203921


namespace gcd_lcm_sum_l2039_203982

theorem gcd_lcm_sum :
  ∀ (a b c d : ℕ), gcd a b + lcm c d = 74 :=
by
  let a := 42
  let b := 70
  let c := 20
  let d := 15
  sorry

end gcd_lcm_sum_l2039_203982


namespace problem_1_problem_2_l2039_203994

-- Definitions for sets A and B
def A : Set ℝ := {x | x^2 - 2 * x - 3 < 0}
def B (a : ℝ) : Set ℝ := {x | abs (x - 1) < a}

-- Define the first problem statement: If A ⊂ B, then a > 2.
theorem problem_1 (a : ℝ) : (A ⊂ B a) → (2 < a) := by
  sorry

-- Define the second problem statement: If B ⊂ A, then a ≤ 0 or (0 < a < 2).
theorem problem_2 (a : ℝ) : (B a ⊂ A) → (a ≤ 0 ∨ (0 < a ∧ a < 2)) := by
  sorry

end problem_1_problem_2_l2039_203994


namespace cos_A_minus_cos_C_l2039_203974

-- Definitions representing the conditions
variables (A B C : ℝ) (a b c : ℝ)
variables (h₁ : 4 * b * Real.sin A = Real.sqrt 7 * a)
variables (h₂ : 2 * b = a + c) (h₃ : A < B) (h₄ : B < C)

-- Statement of the proof problem
theorem cos_A_minus_cos_C (A B C a b c : ℝ)
  (h₁ : 4 * b * Real.sin A = Real.sqrt 7 * a)
  (h₂ : 2 * b = a + c)
  (h₃ : A < B)
  (h₄ : B < C) :
  Real.cos A - Real.cos C = Real.sqrt 7 / 2 :=
by
  sorry

end cos_A_minus_cos_C_l2039_203974


namespace possible_values_of_m_l2039_203946

theorem possible_values_of_m (m : ℝ) :
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, m * x + 1 = 0}
  (∀ x, x ∈ B → x ∈ A) ↔ m = 0 ∨ m = -1 ∨ m = -1 / 3 :=
by
  let A := {x | x^2 - 4 * x + 3 = 0}
  let B := {x | ∃ m : ℝ, m * x + 1 = 0}
  sorry -- Proof needed

end possible_values_of_m_l2039_203946


namespace value_of_x4_plus_1_div_x4_l2039_203970

theorem value_of_x4_plus_1_div_x4 (x : ℝ) (hx : x^2 + 1 / x^2 = 2) : x^4 + 1 / x^4 = 2 := 
sorry

end value_of_x4_plus_1_div_x4_l2039_203970


namespace gcd_max_digits_l2039_203939

theorem gcd_max_digits (a b : ℕ) (h_a : a < 10^7) (h_b : b < 10^7) (h_lcm : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) : Nat.gcd a b < 10^3 :=
by
  sorry

end gcd_max_digits_l2039_203939


namespace total_height_correct_l2039_203969

-- Stack and dimensions setup
def height_of_disc_stack (top_diameter bottom_diameter disc_thickness : ℕ) : ℕ :=
  let num_discs := (top_diameter - bottom_diameter) / 2 + 1
  num_discs * disc_thickness

def total_height (top_diameter bottom_diameter disc_thickness cylinder_height : ℕ) : ℕ :=
  height_of_disc_stack top_diameter bottom_diameter disc_thickness + cylinder_height

-- Given conditions
def top_diameter := 15
def bottom_diameter := 1
def disc_thickness := 2
def cylinder_height := 10
def correct_answer := 26

-- Proof problem
theorem total_height_correct :
  total_height top_diameter bottom_diameter disc_thickness cylinder_height = correct_answer :=
by
  sorry

end total_height_correct_l2039_203969


namespace rectangular_solid_surface_area_l2039_203950

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (hneq1 : a ≠ b) (hneq2 : b ≠ c) (hneq3 : a ≠ c) (hvol : a * b * c = 770) : 2 * (a * b + b * c + c * a) = 1098 :=
by
  sorry

end rectangular_solid_surface_area_l2039_203950


namespace initial_employees_l2039_203992

theorem initial_employees (E : ℕ)
  (salary_per_employee : ℕ)
  (laid_off_fraction : ℚ)
  (total_paid_remaining : ℕ)
  (remaining_employees : ℕ) :
  salary_per_employee = 2000 →
  laid_off_fraction = 1 / 3 →
  total_paid_remaining = 600000 →
  remaining_employees = total_paid_remaining / salary_per_employee →
  (2 / 3 : ℚ) * E = remaining_employees →
  E = 450 := by
  sorry

end initial_employees_l2039_203992


namespace people_on_bus_before_stop_l2039_203998

variable (P_before P_after P_got_on : ℕ)
variable (h1 : P_got_on = 13)
variable (h2 : P_after = 17)

theorem people_on_bus_before_stop : P_before = 4 :=
by
  -- Given that P_after = 17 and P_got_on = 13
  -- We need to prove P_before = P_after - P_got_on = 4
  sorry

end people_on_bus_before_stop_l2039_203998


namespace theater_ticket_cost_l2039_203955

theorem theater_ticket_cost
  (num_persons : ℕ) 
  (num_children : ℕ) 
  (num_adults : ℕ)
  (children_ticket_cost : ℕ)
  (total_receipts_cents : ℕ)
  (A : ℕ) :
  num_persons = 280 →
  num_children = 80 →
  children_ticket_cost = 25 →
  total_receipts_cents = 14000 →
  num_adults = num_persons - num_children →
  200 * A + (num_children * children_ticket_cost) = total_receipts_cents →
  A = 60 :=
by
  intros h_num_persons h_num_children h_children_ticket_cost h_total_receipts_cents h_num_adults h_eqn
  sorry

end theater_ticket_cost_l2039_203955


namespace ratio_of_bottles_given_to_first_house_l2039_203956

theorem ratio_of_bottles_given_to_first_house 
  (total_bottles : ℕ) 
  (bottles_only_cider : ℕ) 
  (bottles_only_beer : ℕ) 
  (bottles_mixed : ℕ) 
  (first_house_bottles : ℕ) 
  (h1 : total_bottles = 180) 
  (h2 : bottles_only_cider = 40) 
  (h3 : bottles_only_beer = 80) 
  (h4 : bottles_mixed = total_bottles - bottles_only_cider - bottles_only_beer) 
  (h5 : first_house_bottles = 90) : 
  first_house_bottles / total_bottles = 1 / 2 :=
by 
  -- Proof goes here
  sorry

end ratio_of_bottles_given_to_first_house_l2039_203956


namespace value_of_f_at_2_l2039_203916

def f (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem value_of_f_at_2 : f 2 = 3 := by
  -- Definition of the function f.
  -- The goal is to prove that f(2) = 3.
  sorry

end value_of_f_at_2_l2039_203916


namespace find_value_of_b_l2039_203962

theorem find_value_of_b (x b : ℕ) 
    (h1 : 5 * (x + 8) = 5 * x + b + 33) : b = 7 :=
sorry

end find_value_of_b_l2039_203962


namespace male_students_tree_planting_l2039_203989

theorem male_students_tree_planting (average_trees : ℕ) (female_trees : ℕ) 
    (male_trees : ℕ) : 
    (average_trees = 6) →
    (female_trees = 15) → 
    (1 / male_trees + 1 / female_trees = 1 / average_trees) → 
    male_trees = 10 :=
by
  intros h_avg h_fem h_eq
  sorry

end male_students_tree_planting_l2039_203989


namespace chord_length_condition_l2039_203944

theorem chord_length_condition (c : ℝ) (h : c > 0) :
  (∃ (x1 x2 : ℝ), 
    x1 ≠ x2 ∧ 
    dist (x1, x1^2) (x2, x2^2) = 2 ∧ 
    ∃ k : ℝ, x1 * k + c = x1^2 ∧ x2 * k + c = x2^2 ) 
    ↔ c > 0 :=
sorry

end chord_length_condition_l2039_203944


namespace graph_symmetry_l2039_203996

variable (f : ℝ → ℝ)

theorem graph_symmetry :
  (∀ x y, y = f (x - 1) ↔ ∃ x', x' = 2 - x ∧ y = f (1 - x'))
  ∧ (∀ x' y', y' = f (1 - x') ↔ ∃ x, x = 2 - x' ∧ y' = f (x - 1)) :=
sorry

end graph_symmetry_l2039_203996


namespace smallest_number_diminished_by_10_l2039_203927

theorem smallest_number_diminished_by_10 (x : ℕ) (h : ∀ n, x - 10 = 24 * n) : x = 34 := 
  sorry

end smallest_number_diminished_by_10_l2039_203927


namespace kelly_games_left_l2039_203968

theorem kelly_games_left (initial_games : Nat) (given_away : Nat) (remaining_games : Nat) 
  (h1 : initial_games = 106) (h2 : given_away = 64) : remaining_games = 42 := by
  sorry

end kelly_games_left_l2039_203968


namespace shortest_distance_between_circles_l2039_203991

theorem shortest_distance_between_circles :
  let circle1 := (x^2 - 12*x + y^2 - 6*y + 9 = 0)
  let circle2 := (x^2 + 10*x + y^2 + 8*y + 34 = 0)
  -- Centers and radii from conditions above:
  let center1 := (6, 3)
  let radius1 := 3
  let center2 := (-5, -4)
  let radius2 := Real.sqrt 7
  let distance_centers := Real.sqrt ((6 - (-5))^2 + (3 - (-4))^2)
  -- Calculate shortest distance
  distance_centers - (radius1 + radius2) = Real.sqrt 170 - 3 - Real.sqrt 7 := sorry

end shortest_distance_between_circles_l2039_203991


namespace minimum_reflection_number_l2039_203975

theorem minimum_reflection_number (a b : ℕ) :
  ((a + 2) * (b + 2) = 4042) ∧ (Nat.gcd (a + 1) (b + 1) = 1) → 
  (a + b = 129) :=
sorry

end minimum_reflection_number_l2039_203975


namespace tangent_parallel_points_l2039_203934

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_parallel_points :
  ∃ (x0 y0 : ℝ), (curve x0 = y0) ∧ 
                 (deriv curve x0 = 4) ∧
                 ((x0 = 1 ∧ y0 = 0) ∨ (x0 = -1 ∧ y0 = -4)) :=
by
  sorry

end tangent_parallel_points_l2039_203934


namespace mean_of_sequence_l2039_203917

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem mean_of_sequence :
  mean [1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 7^2, 2] = 17.75 := by
sorry

end mean_of_sequence_l2039_203917


namespace area_of_triangle_ABC_l2039_203951

theorem area_of_triangle_ABC : 
  let A := (1, 1)
  let B := (4, 1)
  let C := (1, 5)
  let area := 6
  (1:ℝ) * abs (1 * (1 - 5) + 4 * (5 - 1) + 1 * (1 - 1)) / 2 = area := 
by
  sorry

end area_of_triangle_ABC_l2039_203951


namespace union_of_sets_l2039_203902

open Set

theorem union_of_sets (A B : Set ℝ) (hA : A = {x | -2 < x ∧ x < 1}) (hB : B = {x | 0 < x ∧ x < 2}) :
  A ∪ B = {x | -2 < x ∧ x < 2} :=
sorry

end union_of_sets_l2039_203902


namespace sum_of_reciprocals_l2039_203904

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 375) : 1 / x + 1 / y = 8 / 75 :=
by
  sorry

end sum_of_reciprocals_l2039_203904


namespace SomuAge_l2039_203919

theorem SomuAge (F S : ℕ) (h1 : S = F / 3) (h2 : S - 8 = (F - 8) / 5) : S = 16 :=
by 
  sorry

end SomuAge_l2039_203919


namespace original_height_of_tree_l2039_203973

theorem original_height_of_tree
  (current_height_in_inches : ℕ)
  (percent_taller : ℕ)
  (current_height_is_V := 180)
  (percent_taller_is_50 := 50) :
  (current_height_in_inches * 100) / (percent_taller + 100) / 12 = 10 := sorry

end original_height_of_tree_l2039_203973


namespace pudding_cost_l2039_203914

theorem pudding_cost (P : ℝ) (h1 : 75 = 5 * P + 65) : P = 2 :=
sorry

end pudding_cost_l2039_203914


namespace cube_volume_l2039_203971

-- Define the given condition: The surface area of the cube
def surface_area (A : ℕ) := A = 294

-- The key proposition we need to prove using the given condition
theorem cube_volume (s : ℕ) (A : ℕ) (V : ℕ) 
  (area_condition : surface_area A)
  (side_length_condition : s ^ 2 = A / 6) 
  (volume_condition : V = s ^ 3) : 
  V = 343 := by
  sorry

end cube_volume_l2039_203971


namespace number_of_licenses_l2039_203913

-- We define the conditions for the problem
def number_of_letters : ℕ := 3  -- B, C, or D
def number_of_digits : ℕ := 4   -- Four digits following the letter
def choices_per_digit : ℕ := 10 -- Each digit can range from 0 to 9

-- We define the total number of licenses that can be generated
def total_licenses : ℕ := number_of_letters * (choices_per_digit ^ number_of_digits)

-- We now state the theorem to be proved
theorem number_of_licenses : total_licenses = 30000 :=
by
  sorry

end number_of_licenses_l2039_203913


namespace f_sum_2018_2019_l2039_203925

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom even_shifted_function (x : ℝ) : f (x + 1) = f (-x + 1)
axiom f_neg1 : f (-1) = -1

theorem f_sum_2018_2019 : f 2018 + f 2019 = -1 :=
by sorry

end f_sum_2018_2019_l2039_203925


namespace campers_difference_l2039_203963

theorem campers_difference (a_morning : ℕ) (b_morning_afternoon : ℕ) (a_afternoon : ℕ) (a_afternoon_evening : ℕ) (c_evening_only : ℕ) :
  a_morning = 33 ∧ b_morning_afternoon = 11 ∧ a_afternoon = 34 ∧ a_afternoon_evening = 20 ∧ c_evening_only = 10 →
  a_afternoon - (a_afternoon_evening + c_evening_only) = 4 := 
by
  -- The actual proof would go here
  sorry

end campers_difference_l2039_203963


namespace whipped_cream_needed_l2039_203947

def total_days : ℕ := 15
def odd_days_count : ℕ := 8
def even_days_count : ℕ := 7

def pumpkin_pies_on_odd_days : ℕ := 3 * odd_days_count
def apple_pies_on_odd_days : ℕ := 2 * odd_days_count

def pumpkin_pies_on_even_days : ℕ := 2 * even_days_count
def apple_pies_on_even_days : ℕ := 4 * even_days_count

def total_pumpkin_pies_baked : ℕ := pumpkin_pies_on_odd_days + pumpkin_pies_on_even_days
def total_apple_pies_baked : ℕ := apple_pies_on_odd_days + apple_pies_on_even_days

def tiffany_pumpkin_pies_consumed : ℕ := 2
def tiffany_apple_pies_consumed : ℕ := 5

def remaining_pumpkin_pies : ℕ := total_pumpkin_pies_baked - tiffany_pumpkin_pies_consumed
def remaining_apple_pies : ℕ := total_apple_pies_baked - tiffany_apple_pies_consumed

def whipped_cream_for_pumpkin_pies : ℕ := 2 * remaining_pumpkin_pies
def whipped_cream_for_apple_pies : ℕ := remaining_apple_pies

def total_whipped_cream_needed : ℕ := whipped_cream_for_pumpkin_pies + whipped_cream_for_apple_pies

theorem whipped_cream_needed : total_whipped_cream_needed = 111 := by
  -- Proof omitted
  sorry

end whipped_cream_needed_l2039_203947


namespace complex_star_angle_sum_correct_l2039_203983

-- Definitions corresponding to the conditions
def complex_star_interior_angle_sum (n : ℕ) (h : n ≥ 7) : ℕ :=
  180 * (n - 4)

-- The theorem stating the problem
theorem complex_star_angle_sum_correct (n : ℕ) (h : n ≥ 7) :
  complex_star_interior_angle_sum n h = 180 * (n - 4) :=
sorry

end complex_star_angle_sum_correct_l2039_203983


namespace calculate_ab_l2039_203907

theorem calculate_ab {a b c : ℝ} (hc : c ≠ 0) (h1 : (a * b) / c = 4) (h2 : a * (b / c) = 12) : a * b = 12 :=
by
  sorry

end calculate_ab_l2039_203907


namespace factor_roots_l2039_203943

noncomputable def checkRoots (a b c t : ℚ) : Prop :=
  a * t^2 + b * t + c = 0

theorem factor_roots (t : ℚ) :
  checkRoots 8 17 (-10) t ↔ t = 5/8 ∨ t = -2 := by
sorry

end factor_roots_l2039_203943


namespace range_of_a_l2039_203900

theorem range_of_a (a : ℝ) :
  (¬ ( ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0 ) 
    ∨ 
   ¬ ( ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0 )) 
→ a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 :=
by
  sorry

end range_of_a_l2039_203900


namespace laura_saves_more_with_promotion_A_l2039_203938

def promotion_A_cost (pair_price : ℕ) : ℕ :=
  let second_pair_price := pair_price / 2
  pair_price + second_pair_price

def promotion_B_cost (pair_price : ℕ) : ℕ :=
  let discount := pair_price * 20 / 100
  pair_price + (pair_price - discount)

def savings (pair_price : ℕ) : ℕ :=
  promotion_B_cost pair_price - promotion_A_cost pair_price

theorem laura_saves_more_with_promotion_A :
  savings 50 = 15 :=
  by
  -- The detailed proof will be added here
  sorry

end laura_saves_more_with_promotion_A_l2039_203938


namespace pow_zero_eq_one_l2039_203953

theorem pow_zero_eq_one : (-2023)^0 = 1 :=
by
  -- The proof of this theorem will go here.
  sorry

end pow_zero_eq_one_l2039_203953


namespace g_iterated_six_times_is_2_l2039_203957

def g (x : ℝ) : ℝ := (x - 1)^2 + 1

theorem g_iterated_six_times_is_2 : g (g (g (g (g (g 2))))) = 2 := 
by 
  sorry

end g_iterated_six_times_is_2_l2039_203957


namespace height_at_end_of_2_years_l2039_203945

-- Step d): Define the conditions and state the theorem

-- Define a function modeling the height of the tree each year
def tree_height (initial_height : ℕ) (years : ℕ) : ℕ :=
  initial_height * 3^years

-- Given conditions as definitions
def year_4_height := 81 -- height at the end of 4 years

-- Theorem that we need to prove
theorem height_at_end_of_2_years (initial_height : ℕ) (h : tree_height initial_height 4 = year_4_height) :
  tree_height initial_height 2 = 9 :=
sorry

end height_at_end_of_2_years_l2039_203945


namespace factorize_ax_squared_minus_9a_l2039_203987

theorem factorize_ax_squared_minus_9a (a x : ℝ) : 
  a * x^2 - 9 * a = a * (x - 3) * (x + 3) :=
sorry

end factorize_ax_squared_minus_9a_l2039_203987


namespace eval_expression_correct_l2039_203954

noncomputable def evaluate_expression : ℝ :=
    3 + Real.sqrt 3 + (3 - Real.sqrt 3) / 6 + (1 / (Real.cos (Real.pi / 4) - 3))

theorem eval_expression_correct : 
  evaluate_expression = (3 * Real.sqrt 3 - 5 * Real.sqrt 2) / 34 :=
by
  -- Proof can be filled in later
  sorry

end eval_expression_correct_l2039_203954


namespace compute_a1d1_a2d2_a3d3_eq_1_l2039_203997

theorem compute_a1d1_a2d2_a3d3_eq_1 {a1 a2 a3 d1 d2 d3 : ℝ}
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 1 := by
  sorry

end compute_a1d1_a2d2_a3d3_eq_1_l2039_203997


namespace fraction_people_eating_pizza_l2039_203933

variable (people : ℕ) (initial_pizza : ℕ) (pieces_per_person : ℕ) (remaining_pizza : ℕ)
variable (fraction : ℚ)

theorem fraction_people_eating_pizza (h1 : people = 15)
    (h2 : initial_pizza = 50)
    (h3 : pieces_per_person = 4)
    (h4 : remaining_pizza = 14)
    (h5 : 4 * 15 * fraction = initial_pizza - remaining_pizza) :
    fraction = 3 / 5 := 
  sorry

end fraction_people_eating_pizza_l2039_203933


namespace range_of_a_l2039_203903

-- Define the sets A, B, and C
def set_A (x : ℝ) : Prop := -3 < x ∧ x ≤ 2
def set_B (x : ℝ) : Prop := -1 < x ∧ x < 3
def set_A_int_B (x : ℝ) : Prop := -1 < x ∧ x ≤ 2
def set_C (x : ℝ) (a : ℝ) : Prop := a < x ∧ x < a + 1

-- The target theorem to prove
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, set_C x a → set_A_int_B x) → 
  (-1 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l2039_203903


namespace sequence_term_l2039_203915

open Int

-- Define the sequence {S_n} as stated in the problem
def S (n : ℕ) : ℤ := 2 * n^2 - 3 * n

-- Define the sequence {a_n} as the finite difference of {S_n}
def a (n : ℕ) : ℤ := if n = 1 then -1 else S n - S (n - 1)

-- The theorem statement
theorem sequence_term (n : ℕ) (hn : n > 0) : a n = 4 * n - 5 :=
by sorry

end sequence_term_l2039_203915


namespace find_a_l2039_203990

theorem find_a (a : ℝ) :
  (∀ x : ℝ, deriv (fun x => a * x^3 - 2) x * x = 1) → a = 1 / 3 :=
by
  intro h
  have slope_at_minus_1 := h (-1)
  sorry -- here we stop as proof isn't needed

end find_a_l2039_203990


namespace solve_equation_l2039_203923

theorem solve_equation (x : ℝ) : x * (x + 1) = 12 → (x = -4 ∨ x = 3) :=
by
  sorry

end solve_equation_l2039_203923


namespace geometric_sequence_S6_l2039_203937

-- Assume we have a geometric sequence {a_n} and the sum of the first n terms is denoted as S_n
variable (S : ℕ → ℝ)

-- Conditions given in the problem
axiom S2_eq : S 2 = 2
axiom S4_eq : S 4 = 8

-- The goal is to find the value of S 6
theorem geometric_sequence_S6 : S 6 = 26 := 
by 
  sorry

end geometric_sequence_S6_l2039_203937


namespace value_of_s_l2039_203967

theorem value_of_s (s : ℝ) : (3 * (-1)^5 + 2 * (-1)^4 - (-1)^3 + (-1)^2 - 4 * (-1) + s = 0) → (s = -5) :=
by
  intro h
  sorry

end value_of_s_l2039_203967


namespace volume_ratio_of_cubes_l2039_203911

theorem volume_ratio_of_cubes :
  (4^3 / 10^3 : ℚ) = 8 / 125 := by
  sorry

end volume_ratio_of_cubes_l2039_203911


namespace product_equals_permutation_l2039_203910

-- Definitions and conditions
def perm (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Given product sequence
def product_seq (n k : ℕ) : ℕ :=
  (List.range' (n - k + 1) k).foldr (λ x y => x * y) 1

-- Problem statement: The product of numbers from 18 to 9 is equivalent to A_{18}^{10}
theorem product_equals_permutation :
  product_seq 18 10 = perm 18 10 :=
by
  sorry

end product_equals_permutation_l2039_203910


namespace problem_l2039_203906

def pair_eq (a b c d : ℝ) : Prop := (a = c) ∧ (b = d)

def op_a (a b c d : ℝ) : ℝ × ℝ := (a * c + b * d, b * c - a * d)
def op_o (a b c d : ℝ) : ℝ × ℝ := (a + c, b + d)

theorem problem (x y : ℝ) :
  op_a 3 4 x y = (11, -2) →
  op_o 3 4 x y = (4, 6) :=
sorry

end problem_l2039_203906


namespace jean_grandchildren_total_giveaway_l2039_203977

theorem jean_grandchildren_total_giveaway :
  let num_grandchildren := 3
  let cards_per_grandchild_per_year := 2
  let amount_per_card := 80
  let total_amount_per_grandchild_per_year := cards_per_grandchild_per_year * amount_per_card
  let total_amount_per_year := num_grandchildren * total_amount_per_grandchild_per_year
  total_amount_per_year = 480 :=
by
  sorry

end jean_grandchildren_total_giveaway_l2039_203977


namespace find_value_of_y_l2039_203981

theorem find_value_of_y (y : ℚ) (h : 3 * y / 7 = 14) : y = 98 / 3 := 
by
  /- Proof to be completed -/
  sorry

end find_value_of_y_l2039_203981


namespace walker_rate_l2039_203985

theorem walker_rate (W : ℝ) :
  (∀ t : ℝ, t = 5 / 60 ∧ t = 20 / 60 → 20 * t = (5 * 20 / 3) ∧ W * (1 / 3) = 5 / 3) →
  W = 5 :=
by
  sorry

end walker_rate_l2039_203985


namespace andrew_total_hours_l2039_203924

theorem andrew_total_hours (days_worked : ℕ) (hours_per_day : ℝ)
    (h1 : days_worked = 3) (h2 : hours_per_day = 2.5) : 
    days_worked * hours_per_day = 7.5 := by
  sorry

end andrew_total_hours_l2039_203924


namespace short_video_length_l2039_203909

theorem short_video_length 
  (videos_per_day : ℕ) 
  (short_videos_factor : ℕ) 
  (weekly_total_minutes : ℕ) 
  (days_in_week : ℕ) 
  (total_videos : videos_per_day = 3)
  (one_video_longer : short_videos_factor = 6)
  (total_weekly_minutes : weekly_total_minutes = 112)
  (days_a_week : days_in_week = 7) :
  ∃ x : ℕ, (videos_per_day * (short_videos_factor + 2)) * days_in_week = weekly_total_minutes ∧ 
            x = 2 := 
by 
  sorry 

end short_video_length_l2039_203909


namespace shortest_side_length_triangle_l2039_203959

noncomputable def triangle_min_angle_side_length (A B : ℝ) (c : ℝ) (tanA tanB : ℝ) (ha : tanA = 1 / 4) (hb : tanB = 3 / 5) (hc : c = Real.sqrt 17) : ℝ :=
   Real.sqrt 2

theorem shortest_side_length_triangle {A B c : ℝ} {tanA tanB : ℝ} 
  (ha : tanA = 1 / 4) (hb : tanB = 3 / 5) (hc : c = Real.sqrt 17) :
  triangle_min_angle_side_length A B c tanA tanB ha hb hc = Real.sqrt 2 :=
sorry

end shortest_side_length_triangle_l2039_203959


namespace social_logistics_turnover_scientific_notation_l2039_203931

noncomputable def total_social_logistics_turnover_2022 : ℝ := 347.6 * (10 ^ 12)

theorem social_logistics_turnover_scientific_notation :
  total_social_logistics_turnover_2022 = 3.476 * (10 ^ 14) :=
by
  sorry

end social_logistics_turnover_scientific_notation_l2039_203931


namespace accommodation_arrangements_l2039_203964

-- Given conditions
def triple_room_capacity : Nat := 3
def double_room_capacity : Nat := 2
def single_room_capacity : Nat := 1
def num_adult_men : Nat := 4
def num_little_boys : Nat := 2

-- Ensuring little boys are always accompanied by an adult and all rooms are occupied
def is_valid_arrangement (triple double single : Nat × Nat) : Prop :=
  let (triple_adults, triple_boys) := triple
  let (double_adults, double_boys) := double
  let (single_adults, single_boys) := single
  triple_adults + double_adults + single_adults = num_adult_men ∧
  triple_boys + double_boys + single_boys = num_little_boys ∧
  triple = (triple_room_capacity, num_little_boys) ∨
  (triple = (triple_room_capacity, 1) ∧ double = (double_room_capacity, 1)) ∧
  triple_adults + triple_boys = triple_room_capacity ∧
  double_adults + double_boys = double_room_capacity ∧
  single_adults + single_boys = single_room_capacity

-- Main theorem statement
theorem accommodation_arrangements : ∃ (triple double single : Nat × Nat),
  is_valid_arrangement triple double single ∧
  -- The number 36 comes from the correct answer in the solution steps part b)
  (triple.1 + double.1 + single.1 = 4 ∧ triple.2 + double.2 + single.2 = 2) :=
sorry

end accommodation_arrangements_l2039_203964


namespace no_square_sum_l2039_203920

theorem no_square_sum (x y : ℕ) (hxy_pos : 0 < x ∧ 0 < y)
  (hxy_gcd : Nat.gcd x y = 1)
  (hxy_perf : ∃ k : ℕ, x + 3 * y^2 = k^2) : ¬ ∃ z : ℕ, x^2 + 9 * y^4 = z^2 :=
by
  sorry

end no_square_sum_l2039_203920


namespace parabola_focus_l2039_203918

theorem parabola_focus (p : ℝ) (h : 4 = 2 * p * 1^2) : (0, 1 / (4 * 2 * p)) = (0, 1 / 16) :=
by
  sorry

end parabola_focus_l2039_203918


namespace original_sandbox_capacity_l2039_203952

theorem original_sandbox_capacity :
  ∃ (L W H : ℝ), 8 * (L * W * H) = 80 → L * W * H = 10 :=
by
  sorry

end original_sandbox_capacity_l2039_203952


namespace problem1_problem2_l2039_203988

-- Problem (1)
variables {p q : ℝ}

theorem problem1 (hpq : p^3 + q^3 = 2) : p + q ≤ 2 := sorry

-- Problem (2)
variables {a b : ℝ}

theorem problem2 (hab : |a| + |b| < 1) : ∀ x : ℝ, (x^2 + a * x + b = 0) → |x| < 1 := sorry

end problem1_problem2_l2039_203988


namespace expansion_coefficient_l2039_203930

theorem expansion_coefficient :
  ∀ (x : ℝ), (∃ (a₀ a₁ a₂ b : ℝ), x^6 + x^4 = a₀ + a₁ * (x + 2) + a₂ * (x + 2)^2 + b * (x + 2)^3) →
  (a₀ = 0 ∧ a₁ = 0 ∧ a₂ = 0 ∧ b = -168) :=
by
  sorry

end expansion_coefficient_l2039_203930


namespace lamp_post_ratio_l2039_203980

theorem lamp_post_ratio (x k m : ℕ) (h1 : 9 * x = k) (h2 : 99 * x = m) : m = 11 * k :=
by sorry

end lamp_post_ratio_l2039_203980
