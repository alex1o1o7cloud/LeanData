import Mathlib

namespace intersection_M_N_l724_72453

def M := {x : ℝ | -4 < x ∧ x < 2}
def N := {x : ℝ | (x - 3) * (x + 2) < 0}

theorem intersection_M_N : {x : ℝ | -2 < x ∧ x < 2} = M ∩ N :=
by
  sorry

end intersection_M_N_l724_72453


namespace problem_statement_l724_72477

-- Define the function f1 as the square of the sum of the digits of k
def f1 (k : Nat) : Nat :=
  let sum_digits := (Nat.digits 10 k).sum
  sum_digits * sum_digits

-- Define the recursive function f_{n+1}(k) = f1(f_n(k))
def fn : Nat → Nat → Nat
| 0, k => k
| n+1, k => f1 (fn n k)

theorem problem_statement : fn 1991 (2^1990) = 256 :=
sorry

end problem_statement_l724_72477


namespace algebraic_expression_value_l724_72484

-- Define given condition
def condition (x : ℝ) : Prop := 3 * x^2 - 2 * x - 1 = 2

-- Define the target expression
def target_expression (x : ℝ) : ℝ := -9 * x^2 + 6 * x - 1

-- The theorem statement
theorem algebraic_expression_value (x : ℝ) (h : condition x) : target_expression x = -10 := by
  sorry

end algebraic_expression_value_l724_72484


namespace two_to_the_n_plus_3_is_perfect_square_l724_72467

theorem two_to_the_n_plus_3_is_perfect_square (n : ℕ) (h : ∃ a : ℕ, 2^n + 3 = a^2) : n = 0 := 
sorry

end two_to_the_n_plus_3_is_perfect_square_l724_72467


namespace range_of_k_for_intersecting_circles_l724_72452

/-- Given circle \( C \) with equation \( x^2 + y^2 - 8x + 15 = 0 \) and a line \( y = kx - 2 \),
    prove that if there exists at least one point on the line such that a circle with this point
    as the center and a radius of 1 intersects with circle \( C \), then \( 0 \leq k \leq \frac{4}{3} \). -/
theorem range_of_k_for_intersecting_circles (k : ℝ) :
  (∃ (x y : ℝ), y = k * x - 2 ∧ (x - 4) ^ 2 + y ^ 2 - 1 ≤ 1) → 0 ≤ k ∧ k ≤ 4 / 3 :=
by {
  sorry
}

end range_of_k_for_intersecting_circles_l724_72452


namespace diameter_increase_l724_72437

theorem diameter_increase (h : 0.628 = π * d) : d = 0.2 := 
sorry

end diameter_increase_l724_72437


namespace ratio_of_shares_l724_72497

theorem ratio_of_shares (A B C : ℝ) (x : ℝ):
  A = 240 → 
  A + B + C = 600 →
  A = x * (B + C) →
  B = (2/3) * (A + C) →
  A / (B + C) = 2 / 3 :=
by
  intros hA hTotal hFraction hB
  sorry

end ratio_of_shares_l724_72497


namespace sector_central_angle_l724_72430

-- Defining the problem as a theorem in Lean 4
theorem sector_central_angle (r θ : ℝ) (h1 : 2 * r + r * θ = 4) (h2 : (1 / 2) * r^2 * θ = 1) : θ = 2 :=
by
  sorry

end sector_central_angle_l724_72430


namespace acute_triangle_probability_l724_72496

noncomputable def probability_acute_triangle : ℝ := sorry

theorem acute_triangle_probability :
  probability_acute_triangle = 1 / 4 := sorry

end acute_triangle_probability_l724_72496


namespace sin_transformation_l724_72406

theorem sin_transformation (α : ℝ) (h : Real.sin (3 * Real.pi / 2 + α) = 3 / 5) :
  Real.sin (Real.pi / 2 + 2 * α) = -7 / 25 :=
by
  sorry

end sin_transformation_l724_72406


namespace sum_ratio_arithmetic_sequence_l724_72461

theorem sum_ratio_arithmetic_sequence
  (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : ∀ n, S n = (n / 2) * (a 1 + a n))
  (h2 : ∀ k : ℕ, a (k + 1) - a k = a 2 - a 1)
  (h3 : a 4 / a 8 = 2 / 3) :
  S 7 / S 15 = 14 / 45 :=
sorry

end sum_ratio_arithmetic_sequence_l724_72461


namespace initial_cases_purchased_l724_72447

open Nat

-- Definitions based on conditions

def group1_children := 14
def group2_children := 16
def group3_children := 12
def group4_children := (group1_children + group2_children + group3_children) / 2
def total_children := group1_children + group2_children + group3_children + group4_children

def bottles_per_child_per_day := 3
def days := 3
def total_bottles_needed := total_children * bottles_per_child_per_day * days

def additional_bottles_needed := 255

def bottles_per_case := 24
def initial_bottles := total_bottles_needed - additional_bottles_needed

def cases_purchased := initial_bottles / bottles_per_case

-- Theorem to prove the number of cases purchased initially
theorem initial_cases_purchased : cases_purchased = 13 :=
  sorry

end initial_cases_purchased_l724_72447


namespace train_length_l724_72490

theorem train_length 
    (t : ℝ) 
    (s_kmh : ℝ) 
    (s_mps : ℝ)
    (h1 : t = 2.222044458665529) 
    (h2 : s_kmh = 162) 
    (h3 : s_mps = s_kmh * (5 / 18))
    (L : ℝ)
    (h4 : L = s_mps * t) : 
  L = 100 := 
sorry

end train_length_l724_72490


namespace result_l724_72492

def problem : Float :=
  let sum := 78.652 + 24.3981
  let diff := sum - 0.025
  Float.round (diff * 100) / 100

theorem result :
  problem = 103.03 := by
  sorry

end result_l724_72492


namespace min_training_iterations_l724_72479

/-- The model of exponentially decaying learning rate is given by L = L0 * D^(G / G0)
    where
    L  : the learning rate used in each round of optimization,
    L0 : the initial learning rate,
    D  : the decay coefficient,
    G  : the number of training iterations,
    G0 : the decay rate.

    Given:
    - the initial learning rate L0 = 0.5,
    - the decay rate G0 = 18,
    - when G = 18, L = 0.4,

    Prove: 
    The minimum number of training iterations required for the learning rate to decay to below 0.1 (excluding 0.1) is 130.
-/
theorem min_training_iterations
  (L0 : ℝ) (G0 : ℝ) (D : ℝ) (G : ℝ) (L : ℝ)
  (h1 : L0 = 0.5)
  (h2 : G0 = 18)
  (h3 : L = 0.4)
  (h4 : G = 18)
  (h5 : L0 * D^(G / G0) = 0.4)
  : ∃ G, G ≥ 130 ∧ L0 * D^(G / G0) < 0.1 := sorry

end min_training_iterations_l724_72479


namespace integer_distances_implies_vertex_l724_72450

theorem integer_distances_implies_vertex (M A B C D : ℝ × ℝ × ℝ)
  (a b c d : ℕ)
  (h_tetrahedron: 
    dist A B = 2 ∧ dist B C = 2 ∧ dist C D = 2 ∧ dist D A = 2 ∧ 
    dist A C = 2 ∧ dist B D = 2)
  (h_distances: 
    dist M A = a ∧ dist M B = b ∧ dist M C = c ∧ dist M D = d) :
  M = A ∨ M = B ∨ M = C ∨ M = D := 
  sorry

end integer_distances_implies_vertex_l724_72450


namespace solve_for_b_l724_72465

theorem solve_for_b (b : ℚ) (h : 2 * b + b / 4 = 5 / 2) : b = 10 / 9 :=
by sorry

end solve_for_b_l724_72465


namespace perfect_square_trinomial_k_l724_72426

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ a : ℤ, x^2 + k*x + 25 = (x + a)^2 ∧ a^2 = 25) → (k = 10 ∨ k = -10) :=
by
  sorry

end perfect_square_trinomial_k_l724_72426


namespace repairs_cost_correct_l724_72424

variable (C : ℝ)

def cost_of_scooter : ℝ := C
def repair_cost (C : ℝ) : ℝ := 0.10 * C
def selling_price (C : ℝ) : ℝ := 1.20 * C
def profit (C : ℝ) : ℝ := 1100
def profit_percentage (C : ℝ) : ℝ := 0.20 

theorem repairs_cost_correct (C : ℝ) (h₁ : selling_price C - cost_of_scooter C = profit C) (h₂ : profit_percentage C = 0.20) : 
  repair_cost C = 550 := by
  sorry

end repairs_cost_correct_l724_72424


namespace hyperbola_eccentricity_l724_72445

theorem hyperbola_eccentricity 
  (center_origin : ∃ x y : ℝ, x = 0 ∧ y = 0)
  (focus_on_x_axis : ∃ c : ℝ, c > 0)
  (asymptote_eq : ∀ x y : ℝ, (4 + 3 * y = 0) ∨ (4 - 3 * y = 0)) :
  ∃ e : ℝ, e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l724_72445


namespace injective_function_identity_l724_72491

theorem injective_function_identity (f : ℕ → ℕ) (h_inj : Function.Injective f)
  (h : ∀ (m n : ℕ), 0 < m → 0 < n → f (n * f m) ≤ n * m) : ∀ x : ℕ, f x = x :=
by
  sorry

end injective_function_identity_l724_72491


namespace volume_correct_l724_72493

open Set Real

-- Define the conditions: the inequality and the constraints on x, y, z
def region (x y z : ℝ) : Prop :=
  abs (z + x + y) + abs (z + x - y) ≤ 10 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0

-- Define the volume calculation
def volume_of_region : ℝ :=
  62.5

-- State the theorem
theorem volume_correct : ∀ (x y z : ℝ), region x y z → volume_of_region = 62.5 :=
by
  intro x y z h
  sorry

end volume_correct_l724_72493


namespace mean_score_of_all_students_l724_72471

-- Conditions
def M : ℝ := 90
def A : ℝ := 75
def ratio (m a : ℝ) : Prop := m / a = 2 / 3

-- Question and correct answer
theorem mean_score_of_all_students (m a : ℝ) (hm : ratio m a) : (60 * a + 75 * a) / (5 * a / 3) = 81 := by
  sorry

end mean_score_of_all_students_l724_72471


namespace find_a_l724_72470

theorem find_a {a b c : ℕ} (h₁ : a + b = c) (h₂ : b + c = 8) (h₃ : c = 4) : a = 0 := by
  sorry

end find_a_l724_72470


namespace g_eq_g_inv_solution_l724_72444

noncomputable def g (x : ℝ) : ℝ := 4 * x - 5
noncomputable def g_inv (x : ℝ) : ℝ := (x + 5) / 4

theorem g_eq_g_inv_solution (x : ℝ) : g x = g_inv x ↔ x = 5 / 3 :=
by
  sorry

end g_eq_g_inv_solution_l724_72444


namespace cricket_initial_overs_l724_72485

-- Definitions based on conditions
def run_rate_initial : ℝ := 3.2
def run_rate_remaining : ℝ := 12.5
def target_runs : ℝ := 282
def remaining_overs : ℕ := 20

-- Mathematical statement to prove
theorem cricket_initial_overs (x : ℝ) (y : ℝ)
    (h1 : y = run_rate_initial * x)
    (h2 : y + run_rate_remaining * remaining_overs = target_runs) :
    x = 10 :=
sorry

end cricket_initial_overs_l724_72485


namespace find_b_value_l724_72416

theorem find_b_value (b : ℕ) 
  (h1 : 5 ^ 5 * b = 3 * 15 ^ 5) 
  (h2 : b = 9 ^ 3) : b = 729 :=
by
  sorry

end find_b_value_l724_72416


namespace arithmetic_sum_S11_l724_72448

theorem arithmetic_sum_S11 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ)
  (h_arith : ∀ n, a (n+1) - a n = d) -- The sequence is arithmetic with common difference d
  (h_sum : S n = n * (a 1 + a n) / 2) -- Sum of the first n terms definition
  (h_condition: a 3 + a 6 + a 9 = 54) :
  S 11 = 198 := 
sorry

end arithmetic_sum_S11_l724_72448


namespace marble_ratio_l724_72429

theorem marble_ratio
  (L_b : ℕ) (J_y : ℕ) (A : ℕ)
  (A_b : ℕ) (A_y : ℕ) (R : ℕ)
  (h1 : L_b = 4)
  (h2 : J_y = 22)
  (h3 : A = 19)
  (h4 : A_y = J_y / 2)
  (h5 : A = A_b + A_y)
  (h6 : A_b = L_b * R) :
  R = 2 := by
  sorry

end marble_ratio_l724_72429


namespace max_y_coordinate_l724_72442

noncomputable def y_coordinate (θ : Real) : Real :=
  let u := Real.sin θ
  3 * u - 4 * u^3

theorem max_y_coordinate : ∃ θ, y_coordinate θ = 1 := by
  use Real.arcsin (1 / 2)
  sorry

end max_y_coordinate_l724_72442


namespace disc_thickness_l724_72409

theorem disc_thickness (r_sphere : ℝ) (r_disc : ℝ) (h : ℝ)
  (h_radius_sphere : r_sphere = 3)
  (h_radius_disc : r_disc = 10)
  (h_volume_constant : (4/3) * Real.pi * r_sphere^3 = Real.pi * r_disc^2 * h) :
  h = 9 / 25 :=
by
  sorry

end disc_thickness_l724_72409


namespace lawn_care_company_expense_l724_72422

theorem lawn_care_company_expense (cost_blade : ℕ) (num_blades : ℕ) (cost_string : ℕ) :
  cost_blade = 8 → num_blades = 4 → cost_string = 7 → 
  (num_blades * cost_blade + cost_string = 39) :=
by
  intro h1 h2 h3
  sorry

end lawn_care_company_expense_l724_72422


namespace evaluate_fg_of_2_l724_72459

def f (x : ℝ) : ℝ := x ^ 3
def g (x : ℝ) : ℝ := 4 * x + 5

theorem evaluate_fg_of_2 : f (g 2) = 2197 :=
by
  sorry

end evaluate_fg_of_2_l724_72459


namespace inequality_solution_set_l724_72436

theorem inequality_solution_set (x : ℝ) :
  (4 * x - 2 ≥ 3 * (x - 1)) ∧ ((x - 5) / 2 > x - 4) ↔ (-1 ≤ x ∧ x < 3) := 
by sorry

end inequality_solution_set_l724_72436


namespace complement_union_l724_72423

-- Definitions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3}

-- Theorem Statement
theorem complement_union (hU: U = {0, 1, 2, 3, 4}) (hA: A = {0, 1, 3}) (hB: B = {2, 3}) :
  (U \ (A ∪ B)) = {4} :=
sorry

end complement_union_l724_72423


namespace repeating_decimal_sum_l724_72434

theorem repeating_decimal_sum :
  let a := (2 : ℚ) / 3
  let b := (2 : ℚ) / 9
  let c := (4 : ℚ) / 9
  a + b - c = (4 : ℚ) / 9 :=
by
  sorry

end repeating_decimal_sum_l724_72434


namespace largest_AB_under_conditions_l724_72486

def is_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem largest_AB_under_conditions :
  ∃ A B C D : ℕ, is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (A + B) % (C + D) = 0 ∧
    is_prime (A + B) ∧ is_prime (C + D) ∧
    (A + B) = 11 :=
sorry

end largest_AB_under_conditions_l724_72486


namespace smallest_b_value_minimizes_l724_72458

noncomputable def smallest_b_value (a b : ℝ) (c : ℝ := 2) : ℝ :=
  if (1 < a) ∧ (a < b) ∧ (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) ∧ (¬ (1/b + 1/a > c ∧ 1/a + c > 1/b ∧ c + 1/b > 1/a)) then b else 0

theorem smallest_b_value_minimizes (a b : ℝ) (c : ℝ := 2) :
  (1 < a) ∧ (a < b) ∧ (¬ (c + a > b ∧ c + b > a ∧ a + b > c)) ∧ (¬ (1/b + 1/a > c ∧ 1/a + c > 1/b ∧ c + 1/b > 1/a)) →
  b = 2 :=
by sorry

end smallest_b_value_minimizes_l724_72458


namespace production_company_keeps_60_percent_l724_72469

noncomputable def openingWeekendRevenue : ℝ := 120
noncomputable def productionCost : ℝ := 60
noncomputable def profit : ℝ := 192
noncomputable def totalRevenue : ℝ := 3.5 * openingWeekendRevenue
noncomputable def amountKept : ℝ := profit + productionCost
noncomputable def percentageKept : ℝ := (amountKept / totalRevenue) * 100

theorem production_company_keeps_60_percent :
  percentageKept = 60 :=
by
  sorry

end production_company_keeps_60_percent_l724_72469


namespace book_total_pages_l724_72431

theorem book_total_pages (num_chapters pages_per_chapter : ℕ) (h1 : num_chapters = 31) (h2 : pages_per_chapter = 61) :
  num_chapters * pages_per_chapter = 1891 := sorry

end book_total_pages_l724_72431


namespace fractions_order_and_non_equality_l724_72472

theorem fractions_order_and_non_equality:
  (37 / 29 < 41 / 31) ∧ (41 / 31 < 31 / 23) ∧ 
  ((37 / 29 ≠ 4 / 3) ∧ (41 / 31 ≠ 4 / 3) ∧ (31 / 23 ≠ 4 / 3)) := by
  sorry

end fractions_order_and_non_equality_l724_72472


namespace rectangle_dimensions_l724_72414

theorem rectangle_dimensions
  (l w : ℕ)
  (h1 : 2 * l + 2 * w = l * w)
  (h2 : w = l - 3) :
  l = 6 ∧ w = 3 :=
by
  sorry

end rectangle_dimensions_l724_72414


namespace total_end_of_year_students_l724_72421

theorem total_end_of_year_students :
  let start_fourth := 33
  let start_fifth := 45
  let start_sixth := 28
  let left_fourth := 18
  let joined_fourth := 14
  let left_fifth := 12
  let joined_fifth := 20
  let left_sixth := 10
  let joined_sixth := 16

  let end_fourth := start_fourth - left_fourth + joined_fourth
  let end_fifth := start_fifth - left_fifth + joined_fifth
  let end_sixth := start_sixth - left_sixth + joined_sixth
  
  end_fourth + end_fifth + end_sixth = 116 := by
    sorry

end total_end_of_year_students_l724_72421


namespace exponent_division_l724_72400

theorem exponent_division (m n : ℕ) (h : m - n = 1) : 5 ^ m / 5 ^ n = 5 :=
by {
  sorry
}

end exponent_division_l724_72400


namespace total_net_gain_computation_l724_72482

noncomputable def house1_initial_value : ℝ := 15000
noncomputable def house2_initial_value : ℝ := 20000

noncomputable def house1_selling_price : ℝ := 1.15 * house1_initial_value
noncomputable def house2_selling_price : ℝ := 1.2 * house2_initial_value

noncomputable def house1_buy_back_price : ℝ := 0.85 * house1_selling_price
noncomputable def house2_buy_back_price : ℝ := 0.8 * house2_selling_price

noncomputable def house1_profit : ℝ := house1_selling_price - house1_buy_back_price
noncomputable def house2_profit : ℝ := house2_selling_price - house2_buy_back_price

noncomputable def total_net_gain : ℝ := house1_profit + house2_profit

theorem total_net_gain_computation : total_net_gain = 7387.5 :=
by
  sorry

end total_net_gain_computation_l724_72482


namespace area_of_octagon_l724_72468

theorem area_of_octagon (a b : ℝ) (hsquare : a ^ 2 = 16)
  (hperimeter : 4 * a = 8 * b) :
  2 * (1 + Real.sqrt 2) * b ^ 2 = 8 + 8 * Real.sqrt 2 :=
by
  sorry

end area_of_octagon_l724_72468


namespace smallest_single_discount_more_advantageous_l724_72456

theorem smallest_single_discount_more_advantageous (n : ℕ) :
  (∀ n, 0 < n -> (1 - (n:ℝ)/100) < 0.64 ∧ (1 - (n:ℝ)/100) < 0.658503 ∧ (1 - (n:ℝ)/100) < 0.63) → 
  n = 38 := 
sorry

end smallest_single_discount_more_advantageous_l724_72456


namespace original_count_l724_72455

-- Conditions
def original_count_eq (ping_pong_balls shuttlecocks : ℕ) : Prop :=
  ping_pong_balls = shuttlecocks

def removal_count (x : ℕ) : Prop :=
  5 * x - 3 * x = 16

-- Theorem to prove the original number of ping-pong balls and shuttlecocks
theorem original_count (ping_pong_balls shuttlecocks : ℕ) (x : ℕ) (h1 : original_count_eq ping_pong_balls shuttlecocks) (h2 : removal_count x) : ping_pong_balls = 40 ∧ shuttlecocks = 40 :=
  sorry

end original_count_l724_72455


namespace Brenda_bakes_20_cakes_a_day_l724_72483

-- Define the conditions
variables (x : ℕ)

-- Other necessary definitions
def cakes_baked_in_9_days (x : ℕ) : ℕ := 9 * x
def cakes_after_selling_half (total_cakes : ℕ) : ℕ := total_cakes.div2

-- Given condition that Brenda has 90 cakes after selling half
def final_cakes_after_selling : ℕ := 90

-- Mathematical statement we want to prove
theorem Brenda_bakes_20_cakes_a_day (x : ℕ) (h : cakes_after_selling_half (cakes_baked_in_9_days x) = final_cakes_after_selling) : x = 20 :=
by sorry

end Brenda_bakes_20_cakes_a_day_l724_72483


namespace domain_of_f_l724_72412

noncomputable def domain_f : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem domain_of_f : domain_f = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end domain_of_f_l724_72412


namespace worker_savings_multiple_l724_72449

theorem worker_savings_multiple 
  (P : ℝ)
  (P_gt_zero : P > 0)
  (save_fraction : ℝ := 1/3)
  (not_saved_fraction : ℝ := 2/3)
  (total_saved : ℝ := 12 * (save_fraction * P)) :
  ∃ multiple : ℝ, total_saved = multiple * (not_saved_fraction * P) ∧ multiple = 6 := 
by 
  sorry

end worker_savings_multiple_l724_72449


namespace capacity_of_other_bottle_l724_72498

theorem capacity_of_other_bottle 
  (total_milk : ℕ) (capacity_bottle_one : ℕ) (fraction_filled_other_bottle : ℚ)
  (equal_fraction : ℚ) (other_bottle_milk : ℚ) (capacity_other_bottle : ℚ) : 
  total_milk = 8 ∧ capacity_bottle_one = 4 ∧ other_bottle_milk = 16/3 ∧ 
  (equal_fraction * capacity_bottle_one + equal_fraction * capacity_other_bottle = total_milk) ∧ 
  (fraction_filled_other_bottle = 5.333333333333333) → capacity_other_bottle = 8 :=
by
  intro h
  sorry

end capacity_of_other_bottle_l724_72498


namespace minimum_value_128_l724_72407

theorem minimum_value_128 (a b c : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) (h_prod: a * b * c = 8) : 
  (2 * a + b) * (a + 3 * c) * (b * c + 2) ≥ 128 := 
by
  sorry

end minimum_value_128_l724_72407


namespace determine_constants_l724_72489

theorem determine_constants (α β : ℝ) (h_eq : ∀ x, (x - α) / (x + β) = (x^2 - 96 * x + 2210) / (x^2 + 65 * x - 3510))
  (h_num : ∀ x, x^2 - 96 * x + 2210 = (x - 34) * (x - 62))
  (h_denom : ∀ x, x^2 + 65 * x - 3510 = (x - 45) * (x + 78)) :
  α + β = 112 :=
sorry

end determine_constants_l724_72489


namespace initial_fish_count_l724_72420

variable (x : ℕ)

theorem initial_fish_count (initial_fish : ℕ) (given_fish : ℕ) (total_fish : ℕ)
  (h1 : total_fish = initial_fish + given_fish)
  (h2 : total_fish = 69)
  (h3 : given_fish = 47) :
  initial_fish = 22 :=
by
  sorry

end initial_fish_count_l724_72420


namespace stickers_total_correct_l724_72494

-- Define the conditions
def stickers_per_page : ℕ := 10
def pages_total : ℕ := 22

-- Define the total number of stickers
def total_stickers : ℕ := pages_total * stickers_per_page

-- The statement we want to prove
theorem stickers_total_correct : total_stickers = 220 :=
by {
  sorry
}

end stickers_total_correct_l724_72494


namespace mac_total_loss_l724_72499

-- Definitions based on conditions in part a)
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_quarter : ℝ := 0.25
def dimes_per_quarter : ℕ := 3
def nickels_per_quarter : ℕ := 7
def quarters_traded_dimes : ℕ := 20
def quarters_traded_nickels : ℕ := 20

-- Lean statement for the proof problem
theorem mac_total_loss : (dimes_per_quarter * value_dime * quarters_traded_dimes 
                          + nickels_per_quarter * value_nickel * quarters_traded_nickels
                          - 40 * value_quarter) = 3.00 := 
sorry

end mac_total_loss_l724_72499


namespace slope_of_l_l724_72478

def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

def parallel_lines (slope : ℝ) : Prop :=
  ∃ m : ℝ, ∀ x y : ℝ, y = slope * x + m

def intersects_ellipse (slope : ℝ) : Prop :=
  parallel_lines slope ∧ ∃ x y : ℝ, ellipse x y ∧ y = slope * x + (y - slope * x)

theorem slope_of_l {l_slope : ℝ} :
  (∃ (m : ℝ) (x y : ℝ), intersects_ellipse (1 / 4) ∧ (y - l_slope * x = m)) →
  (l_slope = -2) :=
sorry

end slope_of_l_l724_72478


namespace train_length_correct_l724_72413

-- Define the conditions
def bridge_length : ℝ := 180
def train_speed : ℝ := 15
def time_to_cross_bridge : ℝ := 20
def time_to_cross_man : ℝ := 8

-- Define the length of the train
def length_of_train : ℝ := 120

-- Proof statement
theorem train_length_correct :
  (train_speed * time_to_cross_man = length_of_train) ∧
  (train_speed * time_to_cross_bridge = length_of_train + bridge_length) :=
by
  sorry

end train_length_correct_l724_72413


namespace num_squares_figure8_perimeter_figure12_perimeter_figureC_eq_38_ratio_perimeter_figure29_figureD_l724_72415

-- Condition: Figure 1 is formed by 3 identical squares of side length 1 cm.
def squares_in_figure1 : ℕ := 3

-- Condition: Perimeter of Figure 1 is 8 cm.
def perimeter_figure1 : ℝ := 8

-- Condition: Each subsequent figure adds 2 squares.
def squares_in_figure (n : ℕ) : ℕ :=
  squares_in_figure1 + 2 * (n - 1)

-- Condition: Each subsequent figure increases perimeter by 2 cm.
def perimeter_figure (n : ℕ) : ℝ :=
  perimeter_figure1 + 2 * (n - 1)

-- Proof problem (a): Prove that the number of squares in Figure 8 is 17.
theorem num_squares_figure8 :
  squares_in_figure 8 = 17 :=
sorry

-- Proof problem (b): Prove that the perimeter of Figure 12 is 30 cm.
theorem perimeter_figure12 :
  perimeter_figure 12 = 30 :=
sorry

-- Proof problem (c): Prove that the positive integer C for which the perimeter of Figure C is 38 cm is 16.
theorem perimeter_figureC_eq_38 :
  ∃ C : ℕ, perimeter_figure C = 38 :=
sorry

-- Proof problem (d): Prove that the positive integer D for which the ratio of the perimeter of Figure 29 to the perimeter of Figure D is 4/11 is 85.
theorem ratio_perimeter_figure29_figureD :
  ∃ D : ℕ, (perimeter_figure 29 / perimeter_figure D) = (4 / 11) :=
sorry

end num_squares_figure8_perimeter_figure12_perimeter_figureC_eq_38_ratio_perimeter_figure29_figureD_l724_72415


namespace number_of_good_carrots_l724_72481

def total_carrots (nancy_picked : ℕ) (mom_picked : ℕ) : ℕ :=
  nancy_picked + mom_picked

def bad_carrots := 14

def good_carrots (total : ℕ) (bad : ℕ) : ℕ :=
  total - bad

theorem number_of_good_carrots :
  good_carrots (total_carrots 38 47) bad_carrots = 71 := by
  sorry

end number_of_good_carrots_l724_72481


namespace distance_between_A_and_C_l724_72473

theorem distance_between_A_and_C :
  ∀ (AB BC CD AD AC : ℝ),
  AB = 3 → BC = 2 → CD = 5 → AD = 6 → AC = 1 := 
by
  intros AB BC CD AD AC hAB hBC hCD hAD
  have h1 : AD = AB + BC + CD := by sorry
  have h2 : 6 = 3 + 2 + AC := by sorry
  have h3 : 6 = 5 + AC := by sorry
  have h4 : AC = 1 := by sorry
  exact h4

end distance_between_A_and_C_l724_72473


namespace exist_midpoints_l724_72439
open Classical

noncomputable def h (a b c : ℝ) := (a + b + c) / 3

theorem exist_midpoints (a b c : ℝ) (X Y Z : ℝ) (AX BY CZ : ℝ) :
  (0 < X) ∧ (X < a) ∧
  (0 < Y) ∧ (Y < b) ∧
  (0 < Z) ∧ (Z < c) ∧
  (X + (a - X) = (h a b c)) ∧
  (Y + (b - Y) = (h a b c)) ∧
  (Z + (c - Z) = (h a b c)) ∧
  (AX * BY * CZ = (a - X) * (b - Y) * (c - Z))
  → ∃ (X Y Z : ℝ), X = (a / 2) ∧ Y = (b / 2) ∧ Z = (c / 2) :=
by
  sorry

end exist_midpoints_l724_72439


namespace part1_part2_l724_72432

def A (x y : ℝ) : ℝ := 3 * x ^ 2 + 2 * x * y - 2 * x - 1
def B (x y : ℝ) : ℝ := - x ^ 2 + x * y - 1

theorem part1 (x y : ℝ) : A x y + 3 * B x y = 5 * x * y - 2 * x - 4 := by
  sorry

theorem part2 (y : ℝ) : (∀ x : ℝ, 5 * x * y - 2 * x - 4 = -4) → y = 2 / 5 := by
  sorry

end part1_part2_l724_72432


namespace combined_area_of_triangles_l724_72427

noncomputable def area_of_rectangle (length width : ℝ) : ℝ :=
  length * width

noncomputable def first_triangle_area (x : ℝ) : ℝ :=
  5 * x

noncomputable def second_triangle_area (base height : ℝ) : ℝ :=
  (base * height) / 2

theorem combined_area_of_triangles (length width x base height : ℝ)
  (h1 : area_of_rectangle length width / first_triangle_area x = 2 / 5)
  (h2 : base + height = 20)
  (h3 : second_triangle_area base height / first_triangle_area x = 3 / 5)
  (length_value : length = 6)
  (width_value : width = 4)
  (base_value : base = 8) :
  first_triangle_area x + second_triangle_area base height = 108 := 
by
  sorry

end combined_area_of_triangles_l724_72427


namespace complement_of_union_l724_72438

open Set

variable (U A B : Set ℕ)
variable (u_def : U = {0, 1, 2, 3, 4, 5, 6})
variable (a_def : A = {1, 3})
variable (b_def : B = {3, 5})

theorem complement_of_union :
  (U \ (A ∪ B)) = {0, 2, 4, 6} :=
by
  sorry

end complement_of_union_l724_72438


namespace largest_n_condition_l724_72460

theorem largest_n_condition :
  ∃ n : ℤ, (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧ ∃ k : ℤ, 2 * n + 99 = k^2 ∧ ∀ x : ℤ, 
  (∃ m' : ℤ, x^2 = (m' + 1)^3 - m'^3) ∧ ∃ k' : ℤ, 2 * x + 99 = k'^2 → x ≤ 289 :=
sorry

end largest_n_condition_l724_72460


namespace inequality_1_inequality_2_inequality_4_l724_72446

theorem inequality_1 (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a * b + b * c + c * a := sorry

theorem inequality_2 (a : ℝ) : a * (1 - a) ≤ 1 / 4 := sorry

theorem inequality_4 (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := sorry

end inequality_1_inequality_2_inequality_4_l724_72446


namespace calc1_calc2_l724_72475

variable (a b : ℝ) 

theorem calc1 : (-b)^2 * (-b)^3 * (-b)^5 = b^10 :=
by sorry

theorem calc2 : (2 * a * b^2)^3 = 8 * a^3 * b^6 :=
by sorry

end calc1_calc2_l724_72475


namespace Taehyung_walked_distance_l724_72408

variable (step_distance : ℝ) (steps_per_set : ℕ) (num_sets : ℕ)
variable (h1 : step_distance = 0.45)
variable (h2 : steps_per_set = 90)
variable (h3 : num_sets = 13)

theorem Taehyung_walked_distance :
  (steps_per_set * step_distance) * num_sets = 526.5 :=
by 
  rw [h1, h2, h3]
  sorry

end Taehyung_walked_distance_l724_72408


namespace root_relationship_l724_72462

theorem root_relationship (a x₁ x₂ : ℝ) 
  (h_eqn : x₁^2 - (2*a + 1)*x₁ + a^2 + 2 = 0)
  (h_roots : x₂ = 2*x₁)
  (h_vieta1 : x₁ + x₂ = 2*a + 1)
  (h_vieta2 : x₁ * x₂ = a^2 + 2) : 
  a = 4 := 
sorry

end root_relationship_l724_72462


namespace geom_seq_common_ratio_l724_72417

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

def is_geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geom_seq_common_ratio (h1 : a_n 0 + a_n 2 = 10)
                              (h2 : a_n 3 + a_n 5 = 5 / 4)
                              (h_geom : is_geom_seq a_n q) :
  q = 1 / 2 :=
by
  sorry

end geom_seq_common_ratio_l724_72417


namespace definite_integral_value_l724_72428

theorem definite_integral_value :
  (∫ x in (0 : ℝ)..Real.arctan (1/3), (8 + Real.tan x) / (18 * Real.sin x^2 + 2 * Real.cos x^2)) 
  = (Real.pi / 3) + (Real.log 2 / 36) :=
by
  -- Proof to be provided
  sorry

end definite_integral_value_l724_72428


namespace investor_profits_l724_72418

/-- Problem: Given the total contributions and profit sharing conditions, calculate the amount 
    each investor receives. -/

theorem investor_profits :
  ∀ (A_contribution B_contribution C_contribution D_contribution : ℝ) 
    (A_profit B_profit C_profit D_profit : ℝ) 
    (total_capital total_profit : ℝ),
    total_capital = 100000 → 
    A_contribution = B_contribution + 5000 →
    B_contribution = C_contribution + 10000 →
    C_contribution = D_contribution + 5000 →
    total_profit = 60000 →
    A_profit = (35 / 100) * total_profit * (1 + 10 / 100) →
    B_profit = (30 / 100) * total_profit * (1 + 8 / 100) →
    C_profit = (20 / 100) * total_profit * (1 + 5 / 100) → 
    D_profit = (15 / 100) * total_profit →
    (A_profit = 23100 ∧ B_profit = 19440 ∧ C_profit = 12600 ∧ D_profit = 9000) :=
by
  intros
  sorry

end investor_profits_l724_72418


namespace intersection_of_A_and_B_l724_72425

def setA : Set ℤ := {x | abs x < 4}
def setB : Set ℤ := {x | x - 1 ≥ 0}
def setIntersection : Set ℤ := {1, 2, 3}

theorem intersection_of_A_and_B : setA ∩ setB = setIntersection :=
by
  sorry

end intersection_of_A_and_B_l724_72425


namespace find_M_l724_72451

theorem find_M (A M C : ℕ) (h1 : (100 * A + 10 * M + C) * (A + M + C) = 2040)
(h2 : (A + M + C) % 2 = 0)
(h3 : A ≤ 9) (h4 : M ≤ 9) (h5 : C ≤ 9) :
  M = 7 := 
sorry

end find_M_l724_72451


namespace hostel_initial_plan_l724_72403

variable (x : ℕ) -- representing the initial number of days

-- Define the conditions
def provisions_for_250_men (x : ℕ) : ℕ := 250 * x
def provisions_for_200_men_45_days : ℕ := 200 * 45

-- Prove the statement
theorem hostel_initial_plan (x : ℕ) (h : provisions_for_250_men x = provisions_for_200_men_45_days) :
  x = 36 :=
by
  sorry

end hostel_initial_plan_l724_72403


namespace convex_polyhedron_property_l724_72419

-- Given conditions as definitions
def num_faces : ℕ := 40
def num_hexagons : ℕ := 8
def num_triangles_eq_twice_pentagons (P : ℕ) (T : ℕ) : Prop := T = 2 * P
def num_pentagons_eq_twice_hexagons (P : ℕ) (H : ℕ) : Prop := P = 2 * H

-- Main statement for the proof problem
theorem convex_polyhedron_property (P T V : ℕ) :
  num_triangles_eq_twice_pentagons P T ∧ num_pentagons_eq_twice_hexagons P num_hexagons ∧ 
  num_faces = T + P + num_hexagons ∧ V = (T * 3 + P * 5 + num_hexagons * 6) / 2 + num_faces - 2 →
  100 * P + 10 * T + V = 535 :=
by
  sorry

end convex_polyhedron_property_l724_72419


namespace total_money_spent_l724_72480

-- Assume Keanu gave dog 40 fish
def dog_fish := 40

-- Assume Keanu gave cat half as many fish as he gave to his dog
def cat_fish := dog_fish / 2

-- Assume each fish cost $4
def cost_per_fish := 4

-- Prove that total amount of money spent is $240
theorem total_money_spent : (dog_fish + cat_fish) * cost_per_fish = 240 := 
by
  sorry

end total_money_spent_l724_72480


namespace real_solutions_of_polynomial_l724_72410

theorem real_solutions_of_polynomial :
  ∀ x : ℝ, x^4 - 3 * x^3 + x^2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by
  sorry

end real_solutions_of_polynomial_l724_72410


namespace inequality_proof_l724_72435

theorem inequality_proof {x y z : ℝ} (hxy : 0 < x) (hyz : 0 < y) (hzx : 0 < z) (h : x * y + y * z + z * x = 1) :
  x * y * z * (x + y) * (y + z) * (x + z) ≥ (1 - x^2) * (1 - y^2) * (1 - z^2) :=
sorry

end inequality_proof_l724_72435


namespace matt_minus_sara_l724_72411

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25

def matt_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
def sara_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)

theorem matt_minus_sara : matt_total - sara_total = 0 :=
by
  sorry

end matt_minus_sara_l724_72411


namespace arithmetic_sequence_26th_term_eq_neg48_l724_72404

def arithmetic_sequence_term (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_26th_term_eq_neg48 : 
  arithmetic_sequence_term 2 (-2) 26 = -48 :=
by
  sorry

end arithmetic_sequence_26th_term_eq_neg48_l724_72404


namespace distinct_primes_eq_1980_l724_72487

theorem distinct_primes_eq_1980 (p q r A : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
    (hne1 : p ≠ q) (hne2 : q ≠ r) (hne3 : p ≠ r) 
    (h1 : 2 * p * q * r + 50 * p * q = A)
    (h2 : 7 * p * q * r + 55 * p * r = A)
    (h3 : 8 * p * q * r + 12 * q * r = A) : 
    A = 1980 := by {
  sorry
}

end distinct_primes_eq_1980_l724_72487


namespace number_of_people_purchased_only_book_A_l724_72443

-- Definitions based on the conditions
variable (A B x y z w : ℕ)
variable (h1 : z = 500)
variable (h2 : z = 2 * y)
variable (h3 : w = z)
variable (h4 : x + y + z + w = 2500)
variable (h5 : A = x + z)
variable (h6 : B = y + z)
variable (h7 : A = 2 * B)

-- The statement we want to prove
theorem number_of_people_purchased_only_book_A :
  x = 1000 :=
by
  -- The proof steps will be filled here
  sorry

end number_of_people_purchased_only_book_A_l724_72443


namespace sqrt_sqrt_of_81_eq_pm3_and_cube_root_self_l724_72401

theorem sqrt_sqrt_of_81_eq_pm3_and_cube_root_self (x : ℝ) : 
  (∃ y : ℝ, y^2 = 81 ∧ (x^2 = y → x = 3 ∨ x = -3)) ∧ (∀ z : ℝ, z^3 = z → (z = 1 ∨ z = -1 ∨ z = 0)) := by
  sorry

end sqrt_sqrt_of_81_eq_pm3_and_cube_root_self_l724_72401


namespace find_number_l724_72488

theorem find_number : ∃ n : ℝ, 50 + (5 * n) / (180 / 3) = 51 ∧ n = 12 := 
by
  use 12
  sorry

end find_number_l724_72488


namespace sum_in_base_b_l724_72474

noncomputable def s_in_base (b : ℕ) := 13 + 15 + 17

theorem sum_in_base_b (b : ℕ) (h : (13 * 15 * 17 : ℕ) = 4652) : s_in_base b = 51 := by
  sorry

end sum_in_base_b_l724_72474


namespace complex_number_solution_l724_72476

theorem complex_number_solution (i : ℂ) (h : i^2 = -1) : (5 / (2 - i) - i = 2) :=
  sorry

end complex_number_solution_l724_72476


namespace more_tails_than_heads_l724_72402

def total_flips : ℕ := 211
def heads_flips : ℕ := 65
def tails_flips : ℕ := total_flips - heads_flips

theorem more_tails_than_heads : tails_flips - heads_flips = 81 := by
  -- proof is unnecessary according to the instructions
  sorry

end more_tails_than_heads_l724_72402


namespace smallest_number_of_tins_needed_l724_72440

variable (A : ℤ) (C : ℚ)

-- Conditions
def wall_area_valid : Prop := 1915 ≤ A ∧ A < 1925
def coverage_per_tin_valid : Prop := 17.5 ≤ C ∧ C < 18.5
def tins_needed_to_cover_wall (A : ℤ) (C : ℚ) : ℚ := A / C
def smallest_tins_needed : ℚ := 111

-- Proof problem statement
theorem smallest_number_of_tins_needed (A : ℤ) (C : ℚ)
    (h1 : wall_area_valid A)
    (h2 : coverage_per_tin_valid C)
    (h3 : 1915 ≤ A)
    (h4 : A < 1925)
    (h5 : 17.5 ≤ C)
    (h6 : C < 18.5) : 
  tins_needed_to_cover_wall A C + 1 ≥ smallest_tins_needed := by
    sorry

end smallest_number_of_tins_needed_l724_72440


namespace plane_speed_with_tailwind_l724_72433

theorem plane_speed_with_tailwind (V : ℝ) (tailwind_speed : ℝ) (ground_speed_against_tailwind : ℝ) 
  (H1 : tailwind_speed = 75) (H2 : ground_speed_against_tailwind = 310) (H3 : V - tailwind_speed = ground_speed_against_tailwind) :
  V + tailwind_speed = 460 :=
by
  sorry

end plane_speed_with_tailwind_l724_72433


namespace quadratic_inequality_solution_l724_72495

variable (a x : ℝ)

-- Define the quadratic expression and the inequality condition
def quadratic_inequality (a x : ℝ) : Prop := 
  x^2 - (2 * a + 1) * x + a^2 + a < 0

-- Define the interval in which the inequality holds
def solution_set (a x : ℝ) : Prop :=
  a < x ∧ x < a + 1

-- The main statement to be proven
theorem quadratic_inequality_solution :
  ∀ a x, quadratic_inequality a x ↔ solution_set a x :=
sorry

end quadratic_inequality_solution_l724_72495


namespace solve_inequality_l724_72405

theorem solve_inequality (x : ℝ) : -1/3 * x + 1 ≤ -5 → x ≥ 18 := 
  sorry

end solve_inequality_l724_72405


namespace number_of_people_got_off_at_third_stop_l724_72454

-- Definitions for each stop
def initial_passengers : ℕ := 0
def passengers_after_first_stop : ℕ := initial_passengers + 7
def passengers_after_second_stop : ℕ := passengers_after_first_stop - 3 + 5
def passengers_after_third_stop (x : ℕ) : ℕ := passengers_after_second_stop - x + 4

-- Final condition stating there are 11 passengers after the third stop
def final_passengers : ℕ := 11

-- Proof goal
theorem number_of_people_got_off_at_third_stop (x : ℕ) :
  passengers_after_third_stop x = final_passengers → x = 2 :=
by
  -- proof goes here
  sorry

end number_of_people_got_off_at_third_stop_l724_72454


namespace base10_to_base4_of_255_l724_72441

theorem base10_to_base4_of_255 :
  (255 : ℕ) = 3 * 4^3 + 3 * 4^2 + 3 * 4^1 + 3 * 4^0 :=
by
  sorry

end base10_to_base4_of_255_l724_72441


namespace graphs_intersect_exactly_one_point_l724_72463

theorem graphs_intersect_exactly_one_point (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 5 * x + 4 = 2 * x - 6 → x = (7 / (2 * k))) ↔ k = (49 / 40) := 
by
  sorry

end graphs_intersect_exactly_one_point_l724_72463


namespace smallest_x_for_non_prime_expression_l724_72457

/-- The smallest positive integer x for which x^2 + x + 41 is not a prime number is 40. -/
theorem smallest_x_for_non_prime_expression : ∃ x : ℕ, x > 0 ∧ x^2 + x + 41 = 41 * 41 ∧ (∀ y : ℕ, 0 < y ∧ y < x → Prime (y^2 + y + 41)) := 
sorry

end smallest_x_for_non_prime_expression_l724_72457


namespace projection_non_ambiguity_l724_72466

theorem projection_non_ambiguity 
    (a b c : ℝ) 
    (theta : ℝ) 
    (h : a^2 = b^2 + c^2 - 2 * b * c * Real.cos theta) : 
    ∃ (c' : ℝ), c' = c * Real.cos theta ∧ a^2 = b^2 + c^2 + 2 * b * c' := 
sorry

end projection_non_ambiguity_l724_72466


namespace inequality_proof_l724_72464

variable (m : ℕ) (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1)

theorem inequality_proof :
    (m > 0) →
    (x^m / ((1 + y) * (1 + z)) + y^m / ((1 + x) * (1 + z)) + z^m / ((1 + x) * (1 + y)) >= 3/4) :=
by
  intro hm_pos
  -- Proof skipped
  sorry

end inequality_proof_l724_72464
