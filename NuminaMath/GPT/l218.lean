import Mathlib

namespace min_value_of_expr_l218_21829

theorem min_value_of_expr (a : ℝ) (h : a > 3) : ∃ m, (∀ b > 3, b + 4 / (b - 3) ≥ m) ∧ m = 7 :=
sorry

end min_value_of_expr_l218_21829


namespace sum_divisors_of_24_is_60_and_not_prime_l218_21885

def divisors (n : Nat) : List Nat :=
  List.filter (λ d => n % d = 0) (List.range (n + 1))

def sum_divisors (n : Nat) : Nat :=
  (divisors n).sum

def is_prime (n : Nat) : Bool :=
  n > 1 ∧ (List.filter (λ d => d > 1 ∧ d < n ∧ n % d = 0) (List.range (n + 1))).length = 0

theorem sum_divisors_of_24_is_60_and_not_prime :
  sum_divisors 24 = 60 ∧ ¬ is_prime 60 := 
by
  sorry

end sum_divisors_of_24_is_60_and_not_prime_l218_21885


namespace quadrilateral_area_correct_l218_21821

open Real
open Function
open Classical

noncomputable def quadrilateral_area : ℝ :=
  let A := (0, 0)
  let B := (2, 3)
  let C := (5, 0)
  let D := (3, -2)
  let vector_cross_product (u v : ℝ × ℝ) : ℝ := u.1 * v.2 - u.2 * v.1
  let area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 0.5 * abs (vector_cross_product (p2 - p1) (p3 - p1))
  area_triangle A B D + area_triangle B C D

theorem quadrilateral_area_correct : quadrilateral_area = 17 / 2 :=
  sorry

end quadrilateral_area_correct_l218_21821


namespace pizza_eaten_after_six_trips_l218_21864

theorem pizza_eaten_after_six_trips :
  (1 / 3) + (1 / 3) / 2 + (1 / 3) / 2 / 2 + (1 / 3) / 2 / 2 / 2 + (1 / 3) / 2 / 2 / 2 / 2 + (1 / 3) / 2 / 2 / 2 / 2 / 2 = 21 / 32 :=
by
  sorry

end pizza_eaten_after_six_trips_l218_21864


namespace determine_N_l218_21830

theorem determine_N (N : ℕ) : 995 + 997 + 999 + 1001 + 1003 = 5100 - N → N = 100 := by
  sorry

end determine_N_l218_21830


namespace seating_arrangement_l218_21859

theorem seating_arrangement (x y : ℕ) (h1 : x * 8 + y * 7 = 55) : x = 6 :=
by
  sorry

end seating_arrangement_l218_21859


namespace student_correct_answers_l218_21873

-- Defining the conditions as variables and equations
def correct_answers (c w : ℕ) : Prop :=
  c + w = 60 ∧ 4 * c - w = 160

-- Stating the problem: proving the number of correct answers is 44
theorem student_correct_answers (c w : ℕ) (h : correct_answers c w) : c = 44 :=
by 
  sorry

end student_correct_answers_l218_21873


namespace bench_cost_150_l218_21836

-- Define the conditions
def combined_cost (bench_cost table_cost : ℕ) : Prop := bench_cost + table_cost = 450
def table_cost_eq_twice_bench (bench_cost table_cost : ℕ) : Prop := table_cost = 2 * bench_cost

-- Define the main statement, which includes the goal of the proof.
theorem bench_cost_150 (bench_cost table_cost : ℕ) (h_combined_cost : combined_cost bench_cost table_cost)
  (h_table_cost_eq_twice_bench : table_cost_eq_twice_bench bench_cost table_cost) : bench_cost = 150 :=
by
  sorry

end bench_cost_150_l218_21836


namespace find_x_average_l218_21811

theorem find_x_average :
  ∃ x : ℝ, (x + 8 + (7 * x - 3) + (3 * x + 10) + (-x + 6)) / 4 = 5 * x - 4 ∧ x = 3.7 :=
  by
  use 3.7
  sorry

end find_x_average_l218_21811


namespace system_of_equations_solutions_l218_21802

theorem system_of_equations_solutions :
  ∃ (sol : Finset (ℝ × ℝ)), sol.card = 3 ∧
    (∀ (x y : ℝ), (x, y) ∈ sol ↔ (x + 3 * y = 3 ∧ abs (abs x - abs y) = 1)) :=
by
  sorry

end system_of_equations_solutions_l218_21802


namespace solve_inequality_case_a_lt_neg1_solve_inequality_case_a_eq_neg1_solve_inequality_case_a_gt_neg1_l218_21856

variable (a x : ℝ)

theorem solve_inequality_case_a_lt_neg1 (h : a < -1) :
  ((x - 1) * (x + a) > 0) ↔ (x < -a ∨ x > 1) := sorry

theorem solve_inequality_case_a_eq_neg1 (h : a = -1) :
  ((x - 1) * (x + a) > 0) ↔ (x ≠ 1) := sorry

theorem solve_inequality_case_a_gt_neg1 (h : a > -1) :
  ((x - 1) * (x + a) > 0) ↔ (x < -a ∨ x > 1) := sorry

end solve_inequality_case_a_lt_neg1_solve_inequality_case_a_eq_neg1_solve_inequality_case_a_gt_neg1_l218_21856


namespace meaningful_sqrt_l218_21809

theorem meaningful_sqrt (x : ℝ) (h : x - 5 ≥ 0) : x = 6 :=
sorry

end meaningful_sqrt_l218_21809


namespace sum_of_edges_of_rectangular_solid_l218_21884

theorem sum_of_edges_of_rectangular_solid 
(volume : ℝ) (surface_area : ℝ) (a b c : ℝ)
(h1 : volume = a * b * c)
(h2 : surface_area = 2 * (a * b + b * c + c * a))
(h3 : ∃ s : ℝ, s ≠ 0 ∧ a = b / s ∧ c = b * s)
(h4 : volume = 512)
(h5 : surface_area = 384) :
a + b + c = 24 := 
sorry

end sum_of_edges_of_rectangular_solid_l218_21884


namespace emily_garden_larger_l218_21892

-- Define the dimensions and conditions given in the problem
def john_length : ℕ := 30
def john_width : ℕ := 60
def emily_length : ℕ := 35
def emily_width : ℕ := 55

-- Define the effective area for John’s garden given the double space requirement
def john_usable_area : ℕ := (john_length * john_width) / 2

-- Define the total area for Emily’s garden
def emily_usable_area : ℕ := emily_length * emily_width

-- State the theorem to be proved
theorem emily_garden_larger : emily_usable_area - john_usable_area = 1025 :=
by
  sorry

end emily_garden_larger_l218_21892


namespace binom_identity_l218_21880

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_identity (n k : ℕ) : k * binom n k = n * binom (n - 1) (k - 1) := by
  sorry

end binom_identity_l218_21880


namespace original_cost_l218_21808

theorem original_cost (P : ℝ) (h : 0.85 * 0.76 * P = 988) : P = 1529.41 := by
  sorry

end original_cost_l218_21808


namespace problem1_problem2_l218_21806

noncomputable def op (a b : ℝ) := 2 * a - (3 / 2) * (a + b)

theorem problem1 (x : ℝ) (h : op x 4 = 0) : x = 12 :=
by sorry

theorem problem2 (x m : ℝ) (h : op x m = op (-2) (x + 4)) (hnn : x ≥ 0) : m ≥ 14 / 3 :=
by sorry

end problem1_problem2_l218_21806


namespace carrots_picked_by_Carol_l218_21820

theorem carrots_picked_by_Carol (total_carrots mom_carrots : ℕ) (h1 : total_carrots = 38 + 7) (h2 : mom_carrots = 16) :
  total_carrots - mom_carrots = 29 :=
by {
  sorry
}

end carrots_picked_by_Carol_l218_21820


namespace trigonometric_identity_l218_21814

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) + Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 * Real.tan (10 * Real.pi / 180) :=
by
  sorry

end trigonometric_identity_l218_21814


namespace cookies_per_pack_l218_21877

theorem cookies_per_pack
  (trays : ℕ) (cookies_per_tray : ℕ) (packs : ℕ)
  (h1 : trays = 8) (h2 : cookies_per_tray = 36) (h3 : packs = 12) :
  (trays * cookies_per_tray) / packs = 24 :=
by
  sorry

end cookies_per_pack_l218_21877


namespace letters_containing_only_dot_l218_21843

theorem letters_containing_only_dot (DS S_only : ℕ) (total : ℕ) (h1 : DS = 20) (h2 : S_only = 36) (h3 : total = 60) :
  total - (DS + S_only) = 4 :=
by
  sorry

end letters_containing_only_dot_l218_21843


namespace part_one_solution_set_part_two_range_of_m_l218_21872

noncomputable def f (x m : ℝ) : ℝ := |x + m| + |2 * x - 1|

/- Part I -/
theorem part_one_solution_set (x : ℝ) : 
  (f x (-1) <= 2) ↔ (0 <= x ∧ x <= 4 / 3) := 
sorry

/- Part II -/
theorem part_two_range_of_m (m : ℝ) : 
  (∀ x ∈ (Set.Icc 1 2), f x m <= |2 * x + 1|) ↔ (-3 <= m ∧ m <= 0) := 
sorry

end part_one_solution_set_part_two_range_of_m_l218_21872


namespace repeating_decimal_base_l218_21850

theorem repeating_decimal_base (k : ℕ) (h_pos : 0 < k) (h_repr : (9 : ℚ) / 61 = (3 * k + 4) / (k^2 - 1)) : k = 21 :=
  sorry

end repeating_decimal_base_l218_21850


namespace combined_weight_l218_21854

-- We define the variables and the conditions
variables (x y : ℝ)

-- First condition 
def condition1 : Prop := y = (16 - 4) + (30 - 6) + (x - 3)

-- Second condition
def condition2 : Prop := y = 12 + 24 + (x - 3)

-- The statement to prove
theorem combined_weight (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : y = x + 33 :=
by
  -- Skipping the proof part
  sorry

end combined_weight_l218_21854


namespace transaction_mistake_in_cents_l218_21810

theorem transaction_mistake_in_cents
  (x y : ℕ)
  (hx : 10 ≤ x ∧ x ≤ 99)
  (hy : 10 ≤ y ∧ y ≤ 99)
  (error_cents : 100 * y + x - (100 * x + y) = 5616) :
  y = x + 56 :=
by {
  sorry
}

end transaction_mistake_in_cents_l218_21810


namespace remaining_pencils_l218_21849

theorem remaining_pencils (j_pencils : ℝ) (v_pencils : ℝ)
  (j_initial : j_pencils = 300) 
  (j_donated_pct : ℝ := 0.30)
  (v_initial : v_pencils = 2 * 300) 
  (v_donated_pct : ℝ := 0.75) :
  (j_pencils - j_donated_pct * j_pencils) + (v_pencils - v_donated_pct * v_pencils) = 360 :=
by
  sorry

end remaining_pencils_l218_21849


namespace polynomial_factorization_l218_21847

theorem polynomial_factorization (a b : ℤ) (h : (x^2 + x - 6) = (x + a) * (x + b)) :
  (a + b)^2023 = 1 :=
sorry

end polynomial_factorization_l218_21847


namespace right_triangle_area_l218_21894

theorem right_triangle_area (a b c : ℝ) (h1 : a + b = 21) (h2 : c = 15) (h3 : a^2 + b^2 = c^2):
  (1/2) * a * b = 54 :=
by
  sorry

end right_triangle_area_l218_21894


namespace function_increasing_l218_21879

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem function_increasing (a b c : ℝ) (h : a^2 - 3 * b < 0) : 
  ∀ x y : ℝ, x < y → f x a b c < f y a b c := sorry

end function_increasing_l218_21879


namespace units_digit_specified_expression_l218_21823

theorem units_digit_specified_expression :
  let numerator := (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 * 10 * 11)
  let denominator := 8000
  let product := numerator * 20
  (∃ d, product / denominator = d ∧ (d % 10 = 6)) :=
by
  sorry

end units_digit_specified_expression_l218_21823


namespace round_robin_tournament_l218_21899

theorem round_robin_tournament (n k : ℕ) (h : (n-2) * (n-3) = 2 * 3^k): n = 5 :=
sorry

end round_robin_tournament_l218_21899


namespace toy_store_bears_shelves_l218_21896

theorem toy_store_bears_shelves (initial_stock shipment bears_per_shelf total_bears number_of_shelves : ℕ)
  (h1 : initial_stock = 17)
  (h2 : shipment = 10)
  (h3 : bears_per_shelf = 9)
  (h4 : total_bears = initial_stock + shipment)
  (h5 : number_of_shelves = total_bears / bears_per_shelf) :
  number_of_shelves = 3 :=
by
  sorry

end toy_store_bears_shelves_l218_21896


namespace rahim_average_price_l218_21800

/-- 
Rahim bought 40 books for Rs. 600 from one shop and 20 books for Rs. 240 from another.
What is the average price he paid per book?
-/
def books1 : ℕ := 40
def cost1 : ℕ := 600
def books2 : ℕ := 20
def cost2 : ℕ := 240
def totalBooks : ℕ := books1 + books2
def totalCost : ℕ := cost1 + cost2
def averagePricePerBook : ℕ := totalCost / totalBooks

theorem rahim_average_price :
  averagePricePerBook = 14 :=
by
  sorry

end rahim_average_price_l218_21800


namespace slower_ball_speed_l218_21881

open Real

variables (v u C : ℝ)

theorem slower_ball_speed :
  (20 * (v - u) = C) → (4 * (v + u) = C) → ((v + u) * 3 = 75) → u = 10 :=
by
  intros h1 h2 h3
  sorry

end slower_ball_speed_l218_21881


namespace smallest_of_three_consecutive_odd_numbers_l218_21868

theorem smallest_of_three_consecutive_odd_numbers (x : ℤ) 
(h_sum : x + (x+2) + (x+4) = 69) : x = 21 :=
by
  sorry

end smallest_of_three_consecutive_odd_numbers_l218_21868


namespace brown_loss_percentage_is_10_l218_21805

-- Define the initial conditions
def initialHousePrice : ℝ := 100000
def profitPercentage : ℝ := 0.10
def sellingPriceBrown : ℝ := 99000

-- Compute the price Mr. Brown bought the house
def priceBrownBought := initialHousePrice * (1 + profitPercentage)

-- Define the loss percentage as a goal to prove
theorem brown_loss_percentage_is_10 :
  ((priceBrownBought - sellingPriceBrown) / priceBrownBought) * 100 = 10 := by
  sorry

end brown_loss_percentage_is_10_l218_21805


namespace victor_percentage_80_l218_21867

def percentage_of_marks (marks_obtained : ℕ) (maximum_marks : ℕ) : ℕ :=
  (marks_obtained * 100) / maximum_marks

theorem victor_percentage_80 :
  percentage_of_marks 240 300 = 80 := by
  sorry

end victor_percentage_80_l218_21867


namespace union_A_B_l218_21883

noncomputable def A : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
noncomputable def B : Set ℝ := {x | x^2 - 1 < 0}

theorem union_A_B : A ∪ B = {x : ℝ | -1 < x} := by
  sorry

end union_A_B_l218_21883


namespace simplify_expression_l218_21807

variable (b c : ℝ)

theorem simplify_expression :
  3 * b * (3 * b ^ 3 + 2 * b) - 2 * b ^ 2 + c * (3 * b ^ 2 - c) = 9 * b ^ 4 + 4 * b ^ 2 + 3 * b ^ 2 * c - c ^ 2 :=
by
  sorry

end simplify_expression_l218_21807


namespace sufficient_and_necessary_condition_l218_21837

def A : Set ℝ := { x | x - 2 > 0 }

def B : Set ℝ := { x | x < 0 }

def C : Set ℝ := { x | x * (x - 2) > 0 }

theorem sufficient_and_necessary_condition :
  ∀ x : ℝ, x ∈ A ∪ B ↔ x ∈ C :=
sorry

end sufficient_and_necessary_condition_l218_21837


namespace sequence_general_term_l218_21815

theorem sequence_general_term (a : ℕ → ℤ) (n : ℕ) 
  (h₀ : a 0 = 1) 
  (h_rec : ∀ n, a (n + 1) = 2 * a n + n) :
  a n = 2^(n + 1) - n - 1 :=
by sorry

end sequence_general_term_l218_21815


namespace continuity_at_2_l218_21822

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 4 * x^2 + 5 else b * x + 3

theorem continuity_at_2 (b : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) → b = 9 :=
by
  sorry  

end continuity_at_2_l218_21822


namespace f_le_2x_f_not_le_1_9x_l218_21840

-- Define the function f and conditions
def f : ℝ → ℝ := sorry

axiom non_neg_f : ∀ x, 0 ≤ x → 0 ≤ f x
axiom f_at_1 : f 1 = 1
axiom f_additivity : ∀ x1 x2, 0 ≤ x1 → 0 ≤ x2 → x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2

-- Proof for part (1): f(x) ≤ 2x for all x in [0, 1]
theorem f_le_2x : ∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 2 * x := 
by
  sorry

-- Part (2): The inequality f(x) ≤ 1.9x does not hold for all x
theorem f_not_le_1_9x : ¬ (∀ x, 0 ≤ x → x ≤ 1 → f x ≤ 1.9 * x) := 
by
  sorry

end f_le_2x_f_not_le_1_9x_l218_21840


namespace cos_alpha_sub_beta_cos_alpha_l218_21869

section

variables (α β : ℝ)
variables (cos_α : ℝ) (sin_α : ℝ) (cos_β : ℝ) (sin_β : ℝ)

-- The given conditions as premises
variable (h1: cos_α = Real.cos α)
variable (h2: sin_α = Real.sin α)
variable (h3: cos_β = Real.cos β)
variable (h4: sin_β = Real.sin β)
variable (h5: 0 < α ∧ α < π / 2)
variable (h6: -π / 2 < β ∧ β < 0)
variable (h7: (cos_α - cos_β)^2 + (sin_α - sin_β)^2 = 4 / 5)

-- Part I: Prove that cos(α - β) = 3/5
theorem cos_alpha_sub_beta : Real.cos (α - β) = 3 / 5 :=
by
  sorry

-- Additional condition for Part II
variable (h8: cos_β = 12 / 13)

-- Part II: Prove that cos α = 56 / 65
theorem cos_alpha : Real.cos α = 56 / 65 :=
by
  sorry

end

end cos_alpha_sub_beta_cos_alpha_l218_21869


namespace combined_weight_l218_21889

variable (a b c d : ℕ)

theorem combined_weight :
  a + b = 260 →
  b + c = 245 →
  c + d = 270 →
  a + d = 285 :=
by
  intros hab hbc hcd
  sorry

end combined_weight_l218_21889


namespace part1_unique_zero_part2_inequality_l218_21895

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x + 1 / x

theorem part1_unique_zero : ∃! x : ℝ, x > 0 ∧ f x = 0 := by
  sorry

theorem part2_inequality (n : ℕ) (h : n > 0) : 
  Real.log ((n + 1) / n) < 1 / Real.sqrt (n^2 + n) := by
  sorry

end part1_unique_zero_part2_inequality_l218_21895


namespace rosa_bonheur_birth_day_l218_21835

/--
Given that Rosa Bonheur's 210th birthday was celebrated on a Wednesday,
prove that she was born on a Sunday.
-/
theorem rosa_bonheur_birth_day :
  let anniversary_year := 2022
  let birth_year := 1812
  let total_years := anniversary_year - birth_year
  let leap_years := (total_years / 4) - (total_years / 100) + (total_years / 400)
  let regular_years := total_years - leap_years
  let day_shifts := regular_years + 2 * leap_years
  (3 - day_shifts % 7) % 7 = 0 := 
sorry

end rosa_bonheur_birth_day_l218_21835


namespace tetrahedron_painting_l218_21818

theorem tetrahedron_painting (unique_coloring_per_face : ∀ f : Fin 4, ∃ c : Fin 4, True)
  (rotation_identity : ∀ f g : Fin 4, (f = g → unique_coloring_per_face f = unique_coloring_per_face g))
  : (number_of_distinct_paintings : ℕ) = 2 :=
sorry

end tetrahedron_painting_l218_21818


namespace dealership_sales_l218_21816

theorem dealership_sales (sports_cars : ℕ) (sedans : ℕ) (trucks : ℕ) 
  (h1 : sports_cars = 36)
  (h2 : (3 : ℤ) * sedans = 5 * sports_cars)
  (h3 : (3 : ℤ) * trucks = 4 * sports_cars) :
  sedans = 60 ∧ trucks = 48 := 
sorry

end dealership_sales_l218_21816


namespace bus_total_capacity_l218_21853

-- Definitions based on conditions in a)
def left_side_seats : ℕ := 15
def right_side_seats : ℕ := left_side_seats - 3
def seats_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 12

-- Proof statement
theorem bus_total_capacity : (left_side_seats + right_side_seats) * seats_per_seat + back_seat_capacity = 93 := by
  sorry

end bus_total_capacity_l218_21853


namespace find_prime_and_integer_l218_21891

theorem find_prime_and_integer (p x : ℕ) (hp : Nat.Prime p) 
  (hx1 : 1 ≤ x) (hx2 : x ≤ 2 * p) (hdiv : x^(p-1) ∣ (p-1)^x + 1) : 
  (p, x) = (2, 1) ∨ (p, x) = (2, 2) ∨ (p, x) = (3, 1) ∨ (p, x) = (3, 3) ∨ ((p ≥ 5) ∧ (x = 1)) :=
by
  sorry

end find_prime_and_integer_l218_21891


namespace remainder_of_98_mul_102_mod_9_l218_21828

theorem remainder_of_98_mul_102_mod_9 : (98 * 102) % 9 = 6 := 
by 
  -- Introducing the variables and arithmetic
  let x := 98 * 102 
  have h1 : x = 9996 := 
    by norm_num
  have h2 : x % 9 = 6 := 
    by norm_num
  -- Result
  exact h2

end remainder_of_98_mul_102_mod_9_l218_21828


namespace eq_y_as_x_l218_21888

theorem eq_y_as_x (y x : ℝ) : 
  (y = 2*x - 3*y) ∨ (x = 2 - 3*y) ∨ (-y = 2*x - 1) ∨ (y = x) → (y = x) :=
by
  sorry

end eq_y_as_x_l218_21888


namespace find_a_b_l218_21890

theorem find_a_b (a b : ℕ) (h1 : (a^3 - a^2 + 1) * (b^3 - b^2 + 2) = 2020) : 10 * a + b = 53 :=
by {
  -- Proof to be completed
  sorry
}

end find_a_b_l218_21890


namespace cost_of_paints_is_5_l218_21844

-- Define folders due to 6 classes
def folder_cost_per_item := 6
def num_classes := 6
def total_folder_cost : ℕ := folder_cost_per_item * num_classes

-- Define pencils due to the 6 classes and need per class
def pencil_cost_per_item := 2
def pencil_per_class := 3
def total_pencils : ℕ := pencil_per_class * num_classes
def total_pencil_cost : ℕ := pencil_cost_per_item * total_pencils

-- Define erasers needed based on pencils and their cost
def eraser_cost_per_item := 1
def pencils_per_eraser := 6
def total_erasers : ℕ := total_pencils / pencils_per_eraser
def total_eraser_cost : ℕ := eraser_cost_per_item * total_erasers

-- Total cost spent on folders, pencils, and erasers
def total_spent : ℕ := 80
def total_cost_supplies : ℕ := total_folder_cost + total_pencil_cost + total_eraser_cost

-- Cost of paints is the remaining amount when total cost is subtracted from total spent
def cost_of_paints : ℕ := total_spent - total_cost_supplies

-- The goal is to prove the cost of paints
theorem cost_of_paints_is_5 : cost_of_paints = 5 := by
  sorry

end cost_of_paints_is_5_l218_21844


namespace Shekar_marks_in_English_l218_21876

theorem Shekar_marks_in_English 
  (math_marks : ℕ) (science_marks : ℕ) (socialstudies_marks : ℕ) (biology_marks : ℕ) (average_marks : ℕ) (num_subjects : ℕ) 
  (mathscore : math_marks = 76)
  (sciencescore : science_marks = 65)
  (socialstudiesscore : socialstudies_marks = 82)
  (biologyscore : biology_marks = 85)
  (averagescore : average_marks = 74)
  (numsubjects : num_subjects = 5) :
  ∃ (english_marks : ℕ), english_marks = 62 :=
by
  sorry

end Shekar_marks_in_English_l218_21876


namespace hypotenuse_length_50_l218_21882

theorem hypotenuse_length_50 (a b : ℕ) (h₁ : a = 14) (h₂ : b = 48) :
  ∃ c : ℕ, c = 50 ∧ c = Nat.sqrt (a^2 + b^2) :=
by
  sorry

end hypotenuse_length_50_l218_21882


namespace find_simple_interest_sum_l218_21851

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r / 100) ^ n

noncomputable def simple_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * r * n / 100

theorem find_simple_interest_sum (P CIsum : ℝ)
  (simple_rate : ℝ) (simple_years : ℕ)
  (compound_rate : ℝ) (compound_years : ℕ)
  (compound_principal : ℝ)
  (hP : simple_interest P simple_rate simple_years = CIsum)
  (hCI : CIsum = (compound_interest compound_principal compound_rate compound_years - compound_principal) / 2) :
  P = 1272 :=
by
  sorry

end find_simple_interest_sum_l218_21851


namespace inequality_condition_l218_21862

theorem inequality_condition (a x : ℝ) : 
  x^3 + 13 * a^2 * x > 5 * a * x^2 + 9 * a^3 ↔ x > a := 
by
  sorry

end inequality_condition_l218_21862


namespace maximize_take_home_pay_l218_21861

def tax_collected (x : ℝ) : ℝ :=
  10 * x^2

def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - tax_collected x

theorem maximize_take_home_pay : ∃ x : ℝ, (x * 1000 = 50000) ∧ (∀ y : ℝ, take_home_pay x ≥ take_home_pay y) := 
sorry

end maximize_take_home_pay_l218_21861


namespace correct_flag_positions_l218_21841

-- Definitions for the gears and their relations
structure Gear where
  flag_position : ℝ -- position of the flag in degrees

-- Condition: Two identical gears
def identical_gears (A B : Gear) : Prop := true

-- Conditions: Initial positions and gear interaction
def initial_position_A (A : Gear) : Prop := A.flag_position = 0
def initial_position_B (B : Gear) : Prop := B.flag_position = 180
def gear_interaction (A B : Gear) (theta : ℝ) : Prop :=
  A.flag_position = -theta ∧ B.flag_position = theta

-- Definition for the final positions given a rotation angle θ
def final_position (A B : Gear) (theta : ℝ) : Prop :=
  identical_gears A B ∧ initial_position_A A ∧ initial_position_B B ∧ gear_interaction A B theta

-- Theorem stating the positions after some rotation θ
theorem correct_flag_positions (A B : Gear) (theta : ℝ) : final_position A B theta → 
  A.flag_position = -theta ∧ B.flag_position = theta :=
by
  intro h
  cases h
  sorry

end correct_flag_positions_l218_21841


namespace find_n_l218_21833

theorem find_n (n : ℕ) : (256 : ℝ)^(1/4) = (4 : ℝ)^n → n = 1 := 
by
  sorry

end find_n_l218_21833


namespace boys_in_class_l218_21855

theorem boys_in_class (total_students : ℕ) (ratio_girls : ℕ) (ratio_boys : ℕ)
    (h_ratio : ratio_girls = 3) (h_ratio_boys : ratio_boys = 4)
    (h_total_students : total_students = 35) :
    ∃ boys, boys = 20 :=
by
  let k := total_students / (ratio_girls + ratio_boys)
  have hk : k = 5 := by sorry
  let boys := ratio_boys * k
  have h_boys : boys = 20 := by sorry
  exact ⟨boys, h_boys⟩

end boys_in_class_l218_21855


namespace carlos_fraction_l218_21826

theorem carlos_fraction (f : ℝ) :
  (1 - f) ^ 4 * 64 = 4 → f = 1 / 2 :=
by
  intro h
  sorry

end carlos_fraction_l218_21826


namespace arctan_tan_sub_eq_l218_21801

noncomputable def arctan_tan_sub (a b : ℝ) : ℝ := Real.arctan (Real.tan a - 3 * Real.tan b)

theorem arctan_tan_sub_eq (a b : ℝ) (ha : a = 75) (hb : b = 15) :
  arctan_tan_sub a b = 75 :=
by
  sorry

end arctan_tan_sub_eq_l218_21801


namespace triangle_area_l218_21803

theorem triangle_area : 
  ∀ (x y: ℝ), (x / 5 + y / 2 = 1) → (x = 5) ∨ (y = 2) → ∃ A : ℝ, A = 5 :=
by
  intros x y h1 h2
  -- Definitions based on the problem conditions
  have hx : x = 5 := sorry
  have hy : y = 2 := sorry
  have base := 5
  have height := 2
  have area := 1 / 2 * base * height
  use area
  sorry

end triangle_area_l218_21803


namespace number_of_matches_in_first_set_l218_21886

theorem number_of_matches_in_first_set
  (x : ℕ)
  (h1 : (30 : ℚ) * x + 15 * 10 = 25 * (x + 10)) :
  x = 20 :=
by
  -- The proof will be filled in here
  sorry

end number_of_matches_in_first_set_l218_21886


namespace rhombus_area_l218_21858

-- Define the rhombus with given conditions
def rhombus (a d1 d2 : ℝ) : Prop :=
  a = 9 ∧ abs (d1 - d2) = 10 

-- The theorem stating the area of the rhombus
theorem rhombus_area (a d1 d2 : ℝ) (h : rhombus a d1 d2) : 
  (d1 * d2) / 2 = 72 :=
by
  sorry

#check rhombus_area

end rhombus_area_l218_21858


namespace fraction_of_repeating_decimal_l218_21863

theorem fraction_of_repeating_decimal:
  let a := (4 / 10 : ℝ)
  let r := (1 / 10 : ℝ)
  (∑' n:ℕ, a * r^n) = (4 / 9 : ℝ) := by
  sorry

end fraction_of_repeating_decimal_l218_21863


namespace no_integer_pairs_satisfy_equation_l218_21887

theorem no_integer_pairs_satisfy_equation :
  ∀ (a b : ℤ), a^3 + 3 * a^2 + 2 * a ≠ 125 * b^3 + 75 * b^2 + 15 * b + 2 :=
by
  intro a b
  sorry

end no_integer_pairs_satisfy_equation_l218_21887


namespace supermarket_sold_54_pints_l218_21865

theorem supermarket_sold_54_pints (x s : ℝ) 
  (h1 : x * s = 216)
  (h2 : x * (s + 2) = 324) : 
  x = 54 := 
by 
  sorry

end supermarket_sold_54_pints_l218_21865


namespace temperature_at_midnight_l218_21897

-- Define the variables for initial conditions and changes
def T_morning : ℤ := 7 -- Morning temperature in degrees Celsius
def ΔT_noon : ℤ := 2   -- Temperature increase at noon in degrees Celsius
def ΔT_midnight : ℤ := -10  -- Temperature drop at midnight in degrees Celsius

-- Calculate the temperatures at noon and midnight
def T_noon := T_morning + ΔT_noon
def T_midnight := T_noon + ΔT_midnight

-- State the theorem to prove the temperature at midnight
theorem temperature_at_midnight : T_midnight = -1 := by
  sorry

end temperature_at_midnight_l218_21897


namespace expression_simplification_l218_21827

def base_expr := (3 + 4) * (3^2 + 4^2) * (3^4 + 4^4) * (3^8 + 4^8) *
                (3^16 + 4^16) * (3^32 + 4^32) * (3^64 + 4^64)

theorem expression_simplification :
  base_expr = 3^128 - 4^128 := by
  sorry

end expression_simplification_l218_21827


namespace range_of_m_l218_21848

theorem range_of_m (a : ℝ) (m : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (hx : ∃ x < 0, a^x = 3 * m - 2) :
  1 < m :=
sorry

end range_of_m_l218_21848


namespace sin_cos_alpha_eq_fifth_l218_21842

variable {α : ℝ}
variable (h : Real.sin α = 2 * Real.cos α)

theorem sin_cos_alpha_eq_fifth : Real.sin α * Real.cos α = 2 / 5 := by
  sorry

end sin_cos_alpha_eq_fifth_l218_21842


namespace B_joined_amount_l218_21819

theorem B_joined_amount (T : ℝ)
  (A_investment : ℝ := 45000)
  (B_time : ℝ := 2)
  (profit_ratio : ℝ := 2 / 1)
  (investment_ratio_rule : (A_investment * T) / (B_investment_amount * B_time) = profit_ratio) :
  B_investment_amount = 22500 :=
by
  sorry

end B_joined_amount_l218_21819


namespace caleb_counted_right_angles_l218_21824

-- Definitions for conditions
def rectangular_park_angles : ℕ := 4
def square_field_angles : ℕ := 4
def total_angles (x y : ℕ) : ℕ := x + y

-- Theorem stating the problem
theorem caleb_counted_right_angles (h : total_angles rectangular_park_angles square_field_angles = 8) : 
   "type of anges Caleb counted" = "right angles" :=
sorry

end caleb_counted_right_angles_l218_21824


namespace values_for_a_l218_21804

def has_two (A : Set ℤ) : Prop :=
  2 ∈ A

def candidate_values (a : ℤ) : Set ℤ :=
  {-2, 2 * a, a * a - a}

theorem values_for_a (a : ℤ) :
  has_two (candidate_values a) ↔ a = 1 ∨ a = 2 :=
by
  sorry

end values_for_a_l218_21804


namespace container_volume_ratio_l218_21878

theorem container_volume_ratio
  (C D : ℕ)
  (h1 : (3 / 5 : ℚ) * C = (1 / 2 : ℚ) * D)
  (h2 : (1 / 3 : ℚ) * ((1 / 2 : ℚ) * D) + (3 / 5 : ℚ) * C = C) :
  (C : ℚ) / D = 5 / 6 :=
by {
  sorry
}

end container_volume_ratio_l218_21878


namespace ny_sales_tax_l218_21846

theorem ny_sales_tax {x : ℝ} 
  (h1 : 100 + x * 1 + 6/100 * (100 + x * 1) = 110) : 
  x = 3.77 :=
by
  sorry

end ny_sales_tax_l218_21846


namespace power_function_solution_l218_21812

theorem power_function_solution (m : ℤ)
  (h1 : ∃ (f : ℝ → ℝ), ∀ x : ℝ, f x = x^(-m^2 + 2 * m + 3) ∧ ∀ x, f x = f (-x))
  (h2 : ∀ x : ℝ, x > 0 → (x^(-m^2 + 2 * m + 3)) < x^(-m^2 + 2 * m + 3 + x)) :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f x = x^4 :=
by
  sorry

end power_function_solution_l218_21812


namespace range_of_m_correct_l218_21870

noncomputable def range_of_m (x : ℝ) (m : ℝ) : Prop :=
  (x + m) / (x - 2) - (2 * m) / (x - 2) = 3 ∧ x > 0 ∧ x ≠ 2

theorem range_of_m_correct (m : ℝ) : 
  (∃ x : ℝ, range_of_m x m) ↔ m < 6 ∧ m ≠ 2 :=
sorry

end range_of_m_correct_l218_21870


namespace degrees_to_radians_l218_21832

theorem degrees_to_radians (π_radians : ℝ) : 150 * π_radians / 180 = 5 * π_radians / 6 :=
by sorry

end degrees_to_radians_l218_21832


namespace least_n_questions_l218_21834

theorem least_n_questions {n : ℕ} : 
  (1/2 : ℝ)^n < 1/10 → n ≥ 4 :=
by
  sorry

end least_n_questions_l218_21834


namespace number_of_days_A_left_l218_21839

noncomputable def work_problem (W : ℝ) : Prop :=
  let A_rate := W / 45
  let B_rate := W / 40
  let days_B_alone := 23
  ∃ x : ℝ, x * (A_rate + B_rate) + days_B_alone * B_rate = W ∧ x = 9

theorem number_of_days_A_left (W : ℝ) : work_problem W :=
  sorry

end number_of_days_A_left_l218_21839


namespace hourly_wage_12_5_l218_21825

theorem hourly_wage_12_5 
  (H : ℝ)
  (work_hours : ℝ := 40)
  (widgets_per_week : ℝ := 1000)
  (widget_earnings_per_widget : ℝ := 0.16)
  (total_earnings : ℝ := 660) :
  (40 * H + 1000 * 0.16 = 660) → (H = 12.5) :=
by
  sorry

end hourly_wage_12_5_l218_21825


namespace count_valid_n_l218_21893

theorem count_valid_n :
  ∃ (S : Finset ℕ), (∀ n ∈ S, 300 < n^2 ∧ n^2 < 1200 ∧ n % 3 = 0) ∧
                     S.card = 6 := sorry

end count_valid_n_l218_21893


namespace at_least_one_equals_a_l218_21831

theorem at_least_one_equals_a (x y z a : ℝ) (hx_ne_0 : x ≠ 0) (hy_ne_0 : y ≠ 0) (hz_ne_0 : z ≠ 0) (ha_ne_0 : a ≠ 0)
  (h1 : x + y + z = a) (h2 : 1/x + 1/y + 1/z = 1/a) : x = a ∨ y = a ∨ z = a :=
  sorry

end at_least_one_equals_a_l218_21831


namespace find_x0_and_m_l218_21860

theorem find_x0_and_m (x : ℝ) (m : ℝ) (x0 : ℝ) :
  (abs (x + 3) - 2 * x - 1 < 0 ↔ x > 2) ∧ 
  (∃ x, abs (x - m) + abs (x + 1 / m) - 2 = 0) → 
  (x0 = 2 ∧ m = 1) := 
by
  sorry

end find_x0_and_m_l218_21860


namespace integer_values_in_interval_l218_21898

theorem integer_values_in_interval : (∃ n : ℕ, n = 25 ∧ ∀ x : ℤ, abs x < 4 * π ↔ -12 ≤ x ∧ x ≤ 12) :=
by
  sorry

end integer_values_in_interval_l218_21898


namespace evaluate_expr_at_neg3_l218_21857

-- Define the expression
def expr (x : ℤ) : ℤ := (5 + x * (5 + x) - 5^2) / (x - 5 + x^2)

-- Define the proposition to be proven
theorem evaluate_expr_at_neg3 : expr (-3) = -26 := by
  sorry

end evaluate_expr_at_neg3_l218_21857


namespace M_squared_is_odd_l218_21871

theorem M_squared_is_odd (a b : ℤ) (h1 : a = b + 1) (c : ℤ) (h2 : c = a * b) (M : ℤ) (h3 : M^2 = a^2 + b^2 + c^2) : M^2 % 2 = 1 := 
by
  sorry

end M_squared_is_odd_l218_21871


namespace exists_integer_solution_l218_21875

theorem exists_integer_solution (x : ℤ) (h : x - 1 < 0) : ∃ y : ℤ, y < 1 :=
by
  sorry

end exists_integer_solution_l218_21875


namespace NaOH_HCl_reaction_l218_21838

theorem NaOH_HCl_reaction (m : ℝ) (HCl : ℝ) (NaCl : ℝ) 
  (reaction_eq : NaOH + HCl = NaCl + H2O)
  (HCl_combined : HCl = 1)
  (NaCl_produced : NaCl = 1) :
  m = 1 := by
  sorry

end NaOH_HCl_reaction_l218_21838


namespace center_and_radius_of_circle_l218_21866

def circle_equation := ∀ (x y : ℝ), x^2 + y^2 - 2*x - 3 = 0

theorem center_and_radius_of_circle :
  (∃ h k r : ℝ, (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - 2*x - 3 = 0) ∧ h = 1 ∧ k = 0 ∧ r = 2) :=
sorry

end center_and_radius_of_circle_l218_21866


namespace max_4x3_y3_l218_21813

theorem max_4x3_y3 (x y : ℝ) (h1 : x ≤ 2) (h2 : y ≤ 3) (h3 : x + y = 3) (h_pos_x : 0 < x) (h_pos_y : 0 < y) : 
  4 * x^3 + y^3 ≤ 33 :=
sorry

end max_4x3_y3_l218_21813


namespace factorize_4a2_minus_9_factorize_2x2y_minus_8xy_plus_8y_l218_21817

-- Factorization of 4a^2 - 9 as (2a + 3)(2a - 3)
theorem factorize_4a2_minus_9 (a : ℝ) : 4 * a^2 - 9 = (2 * a + 3) * (2 * a - 3) :=
by 
  sorry

-- Factorization of 2x^2 y - 8xy + 8y as 2y(x-2)^2
theorem factorize_2x2y_minus_8xy_plus_8y (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2) ^ 2 :=
by 
  sorry

end factorize_4a2_minus_9_factorize_2x2y_minus_8xy_plus_8y_l218_21817


namespace cos_of_sin_given_l218_21845

theorem cos_of_sin_given (θ : ℝ) (h : Real.sin (88 * Real.pi / 180 + θ) = 2 / 3) :
  Real.cos (178 * Real.pi / 180 + θ) = - (2 / 3) :=
by
  sorry

end cos_of_sin_given_l218_21845


namespace n_squared_plus_2n_plus_3_mod_50_l218_21874

theorem n_squared_plus_2n_plus_3_mod_50 (n : ℤ) (hn : n % 50 = 49) : (n^2 + 2 * n + 3) % 50 = 2 := 
sorry

end n_squared_plus_2n_plus_3_mod_50_l218_21874


namespace max_wx_xy_yz_zt_l218_21852

theorem max_wx_xy_yz_zt {w x y z t : ℕ} (h_sum : w + x + y + z + t = 120)
  (hnn_w : 0 ≤ w) (hnn_x : 0 ≤ x) (hnn_y : 0 ≤ y) (hnn_z : 0 ≤ z) (hnn_t : 0 ≤ t) :
  wx + xy + yz + zt ≤ 3600 := 
sorry

end max_wx_xy_yz_zt_l218_21852
