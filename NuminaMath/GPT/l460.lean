import Mathlib

namespace shaded_area_percentage_correct_l460_46088

-- Define a square and the conditions provided
def square (side_length : ℕ) : ℕ := side_length ^ 2

-- Define conditions
def EFGH_side_length : ℕ := 6
def total_area : ℕ := square EFGH_side_length

def shaded_area_1 : ℕ := square 2
def shaded_area_2 : ℕ := square 4 - square 3
def shaded_area_3 : ℕ := square 6 - square 5

def total_shaded_area : ℕ := shaded_area_1 + shaded_area_2 + shaded_area_3

def shaded_percentage : ℚ := total_shaded_area / total_area * 100

-- Statement of the theorem to prove
theorem shaded_area_percentage_correct :
  shaded_percentage = 61.11 := by sorry

end shaded_area_percentage_correct_l460_46088


namespace product_of_three_numbers_l460_46094

theorem product_of_three_numbers (x y z : ℝ) 
  (h1 : x + y + z = 20) 
  (h2 : x = 4 * (y + z)) 
  (h3 : y = 7 * z) :
  x * y * z = 28 := 
by 
  sorry

end product_of_three_numbers_l460_46094


namespace horner_v3_value_correct_l460_46005

def f (x : ℕ) : ℕ :=
  x^5 + 2*x^3 + 3*x^2 + x + 1

def horner_eval (x : ℕ) : ℕ :=
  ((((x + 0) * x + 2) * x + 3) * x + 1) * x + 1

theorem horner_v3_value_correct :
  horner_eval 3 = 36 :=
sorry

end horner_v3_value_correct_l460_46005


namespace power_equality_l460_46046

theorem power_equality : (243 : ℝ) ^ (1 / 3) = (3 : ℝ) ^ (5 / 3) := 
by 
  sorry

end power_equality_l460_46046


namespace average_throws_to_lasso_l460_46058

theorem average_throws_to_lasso (p : ℝ) (h₁ : 1 - (1 - p)^3 = 0.875) : (1 / p) = 2 :=
by
  sorry

end average_throws_to_lasso_l460_46058


namespace find_x1_l460_46079

theorem find_x1 (x1 x2 x3 x4 : ℝ) 
  (h1 : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + (x3 - x4)^2 + x4^2 = 1 / 3)
  (h3 : x1 + x2 + x3 + x4 = 2) : 
  x1 = 4 / 5 :=
sorry

end find_x1_l460_46079


namespace shooting_competition_hits_l460_46096

noncomputable def a1 : ℝ := 1
noncomputable def d : ℝ := 0.5
noncomputable def S_n (n : ℝ) : ℝ := (n / 2) * (2 * a1 + (n - 1) * d)

theorem shooting_competition_hits (n : ℝ) (h : S_n n = 7) : 25 - n = 21 :=
by
  -- sequence of proof steps
  sorry

end shooting_competition_hits_l460_46096


namespace polynomial_coeff_sum_l460_46030

variable (d : ℤ)
variable (h : d ≠ 0)

theorem polynomial_coeff_sum : 
  (∃ a b c e : ℤ, (10 * d + 15 + 12 * d^2 + 2 * d^3) + (4 * d - 3 + 2 * d^2) = a * d^3 + b * d^2 + c * d + e ∧ a + b + c + e = 42) :=
by
  sorry

end polynomial_coeff_sum_l460_46030


namespace prime_remainder_30_l460_46013

theorem prime_remainder_30 (p : ℕ) (hp : Nat.Prime p) (hgt : p > 30) (hmod2 : p % 2 ≠ 0) 
(hmod3 : p % 3 ≠ 0) (hmod5 : p % 5 ≠ 0) : 
  ∃ (r : ℕ), r < 30 ∧ (p % 30 = r) ∧ (r = 1 ∨ Nat.Prime r) := 
by
  sorry

end prime_remainder_30_l460_46013


namespace hats_count_l460_46052

theorem hats_count (T M W : ℕ) (hT : T = 1800)
  (hM : M = (2 * T) / 3) (hW : W = T - M) 
  (hats_men : ℕ) (hats_women : ℕ) (m_hats_condition : hats_men = 15 * M / 100)
  (w_hats_condition : hats_women = 25 * W / 100) :
  hats_men + hats_women = 330 :=
by sorry

end hats_count_l460_46052


namespace triangle_BD_length_l460_46089

theorem triangle_BD_length 
  (A B C D : Type) 
  (hAC : AC = 8) 
  (hBC : BC = 8) 
  (hAD : AD = 6) 
  (hCD : CD = 5) : BD = 6 :=
  sorry

end triangle_BD_length_l460_46089


namespace gcd_polynomial_l460_46031

theorem gcd_polynomial (a : ℕ) (h : 270 ∣ a) : Nat.gcd (5 * a^3 + 3 * a^2 + 5 * a + 45) a = 45 :=
sorry

end gcd_polynomial_l460_46031


namespace possible_values_of_a_l460_46007

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a + b * Real.cos x + c * Real.sin x

theorem possible_values_of_a 
    (a b c : ℝ) 
    (h1 : f a b c (Real.pi / 2) = 1) 
    (h2 : f a b c Real.pi = 1) 
    (h3 : ∀ x : ℝ, |f a b c x| ≤ 2) :
    4 - 3 * Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2 :=
by
  sorry

end possible_values_of_a_l460_46007


namespace shelves_used_l460_46023

def initial_books : Nat := 87
def sold_books : Nat := 33
def books_per_shelf : Nat := 6

theorem shelves_used :
  (initial_books - sold_books) / books_per_shelf = 9 := by
  sorry

end shelves_used_l460_46023


namespace necessary_but_not_sufficient_condition_l460_46098

def isEllipse (a b : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ x y : ℝ, f x y = 1

theorem necessary_but_not_sufficient_condition (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  isEllipse a b (λ x y => a * x^2 + b * y^2) → ¬(∃ x y : ℝ, a * x^2 + b * y^2 = 1) :=
sorry

end necessary_but_not_sufficient_condition_l460_46098


namespace range_of_b_l460_46060

theorem range_of_b (M : Set (ℝ × ℝ)) (N : ℝ → ℝ → Set (ℝ × ℝ)) :
  (∀ m : ℝ, (∃ x y : ℝ, (x, y) ∈ M ∧ (x, y) ∈ (N m b))) ↔ b ∈ Set.Icc (- Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
by
  sorry

end range_of_b_l460_46060


namespace estimated_red_balls_l460_46043

-- Definitions based on conditions
def total_balls : ℕ := 15
def black_ball_frequency : ℝ := 0.6
def red_ball_frequency : ℝ := 1 - black_ball_frequency

-- Theorem stating the proof problem
theorem estimated_red_balls :
  (total_balls : ℝ) * red_ball_frequency = 6 := by
  sorry

end estimated_red_balls_l460_46043


namespace min_beans_l460_46036

theorem min_beans (r b : ℕ) (H1 : r ≥ 3 + 2 * b) (H2 : r ≤ 3 * b) : b ≥ 3 := 
sorry

end min_beans_l460_46036


namespace extreme_points_inequality_l460_46019

noncomputable def f (a x : ℝ) : ℝ := a * x - (a / x) - 2 * Real.log x
noncomputable def f' (a x : ℝ) : ℝ := (a * x^2 - 2 * x + a) / x^2

theorem extreme_points_inequality (a x1 x2 : ℝ) (h1 : a > 0) (h2 : 1 < x1) (h3 : x1 < Real.exp 1)
  (h4 : f a x1 = 0) (h5 : f a x2 = 0) (h6 : x1 ≠ x2) : |f a x1 - f a x2| < 1 :=
by
  sorry

end extreme_points_inequality_l460_46019


namespace pages_left_to_write_l460_46044

theorem pages_left_to_write : 
  let first_day := 25
  let second_day := 2 * first_day
  let third_day := 2 * second_day
  let fourth_day := 10
  let total_written := first_day + second_day + third_day + fourth_day
  let total_pages := 500
  let remaining_pages := total_pages - total_written
  remaining_pages = 315 :=
by
  let first_day := 25
  let second_day := 2 * first_day
  let third_day := 2 * second_day
  let fourth_day := 10
  let total_written := first_day + second_day + third_day + fourth_day
  let total_pages := 500
  let remaining_pages := total_pages - total_written
  show remaining_pages = 315
  sorry

end pages_left_to_write_l460_46044


namespace julies_balls_after_1729_steps_l460_46045

-- Define the process described
def increment_base_8 (n : ℕ) : List ℕ := 
by
  if n = 0 then
    exact [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 8) (n % 8 :: acc)
    exact loop n []

-- Define the total number of balls after 'steps' steps
def julies_total_balls (steps : ℕ) : ℕ :=
by 
  exact (increment_base_8 steps).sum

theorem julies_balls_after_1729_steps : julies_total_balls 1729 = 7 :=
by
  sorry

end julies_balls_after_1729_steps_l460_46045


namespace like_terms_exp_l460_46042

theorem like_terms_exp (a b : ℝ) (m n x : ℝ)
  (h₁ : 2 * a ^ x * b ^ (n + 1) = -3 * a * b ^ (2 * m))
  (h₂ : x = 1) (h₃ : n + 1 = 2 * m) : 
  (2 * m - n) ^ x = 1 := 
by
  sorry

end like_terms_exp_l460_46042


namespace angle_between_bisectors_of_trihedral_angle_l460_46047

noncomputable def angle_between_bisectors_trihedral (α β γ : ℝ) (hα : α = 90) (hβ : β = 90) (hγ : γ = 90) : ℝ :=
  60

theorem angle_between_bisectors_of_trihedral_angle (α β γ : ℝ) (hα : α = 90) (hβ : β = 90) (hγ : γ = 90) :
  angle_between_bisectors_trihedral α β γ hα hβ hγ = 60 := 
sorry

end angle_between_bisectors_of_trihedral_angle_l460_46047


namespace find_cheesecake_price_l460_46004

def price_of_cheesecake (C : ℝ) (coffee_price : ℝ) (discount_rate : ℝ) (final_price : ℝ) : Prop :=
  let original_price := coffee_price + C
  let discounted_price := discount_rate * original_price
  discounted_price = final_price

theorem find_cheesecake_price : ∃ C : ℝ,
  price_of_cheesecake C 6 0.75 12 ∧ C = 10 :=
by
  sorry

end find_cheesecake_price_l460_46004


namespace find_n_l460_46055

theorem find_n (n : ℕ) (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^28) : n = 27 :=
sorry

end find_n_l460_46055


namespace amount_tom_should_pay_l460_46053

theorem amount_tom_should_pay (original_price : ℝ) (multiplier : ℝ) 
  (h1 : original_price = 3) (h2 : multiplier = 3) : 
  original_price * multiplier = 9 :=
sorry

end amount_tom_should_pay_l460_46053


namespace B_squared_B_sixth_l460_46087

noncomputable def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![0, 3], ![2, -1]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ :=
  1

theorem B_squared :
  B * B = 3 * B - I := by
  sorry

theorem B_sixth :
  B^6 = 84 * B - 44 * I := by
  sorry

end B_squared_B_sixth_l460_46087


namespace hyperbola_representation_iff_l460_46082

theorem hyperbola_representation_iff (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (2 + m) - (y^2) / (m + 1) = 1) ↔ (m > -1 ∨ m < -2) :=
by
  sorry

end hyperbola_representation_iff_l460_46082


namespace denominator_of_fraction_l460_46085

theorem denominator_of_fraction (n : ℕ) (h1 : n = 20) (h2 : num = 35) (dec_value : ℝ) (h3 : dec_value = 2 / 10^n) : denom = 175 * 10^20 :=
by
  sorry

end denominator_of_fraction_l460_46085


namespace find_four_numbers_l460_46015

theorem find_four_numbers (a b c d : ℕ) 
  (h1 : a + b = 2024) 
  (h2 : a + c = 2026) 
  (h3 : a + d = 2030) 
  (h4 : b + c = 2028) 
  (h5 : b + d = 2032) 
  (h6 : c + d = 2036) : 
  (a = 1011 ∧ b = 1012 ∧ c = 1013 ∧ d = 1015) := 
sorry

end find_four_numbers_l460_46015


namespace part1_part2_l460_46017

namespace Problem

open Set

def A : Set ℤ := {x | -6 ≤ x ∧ x ≤ 6}
def B : Set ℤ := {1, 2, 3}
def C : Set ℤ := {3, 4, 5, 6}

-- Part (1)
theorem part1 : A ∩ (B ∩ C) = {3} := by 
  sorry

-- Part (2)
theorem part2 : A ∩ (A \ (B ∪ C)) = {-6, -5, -4, -3, -2, -1, 0} := by 
  sorry

end Problem

end part1_part2_l460_46017


namespace find_center_of_circle_l460_46022

noncomputable def center_of_circle (x y : ℝ) : Prop :=
  x^2 - 8 * x + y^2 + 4 * y = 16

theorem find_center_of_circle (x y : ℝ) (h : center_of_circle x y) : (x, y) = (4, -2) :=
by 
  sorry

end find_center_of_circle_l460_46022


namespace evaluate_power_l460_46075

theorem evaluate_power (n : ℕ) (h : 3^(2 * n) = 81) : 9^(n + 1) = 729 :=
by sorry

end evaluate_power_l460_46075


namespace hypotenuse_length_l460_46048

variables (a b c : ℝ)

-- Definitions from conditions
def right_angled_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

def sum_of_squares_is_2000 (a b c : ℝ) : Prop :=
  a^2 + b^2 + c^2 = 2000

def perimeter_is_60 (a b c : ℝ) : Prop :=
  a + b + c = 60

theorem hypotenuse_length (a b c : ℝ)
  (h1 : right_angled_triangle a b c)
  (h2 : sum_of_squares_is_2000 a b c)
  (h3 : perimeter_is_60 a b c) :
  c = 10 * Real.sqrt 10 :=
sorry

end hypotenuse_length_l460_46048


namespace product_mnp_l460_46095

theorem product_mnp (a x y z c : ℕ) (m n p : ℕ) :
  (a ^ 8 * x * y * z - a ^ 7 * y * z - a ^ 6 * x * z = a ^ 5 * (c ^ 5 - 1) ∧
   (a ^ m * x * z - a ^ n) * (a ^ p * y * z - a ^ 3) = a ^ 5 * c ^ 5) →
  m = 5 ∧ n = 4 ∧ p = 3 ∧ m * n * p = 60 :=
by
  sorry

end product_mnp_l460_46095


namespace length_of_AC_l460_46069

theorem length_of_AC (AB DC AD : ℝ) (h1 : AB = 15) (h2 : DC = 24) (h3 : AD = 7) : AC = 30.1 :=
sorry

end length_of_AC_l460_46069


namespace largest_four_digit_by_two_moves_l460_46024

def moves (n : Nat) (d1 d2 d3 d4 : Nat) : Prop :=
  ∃ x y : ℕ, d1 = x → d2 = y → n = 1405 → (x ≤ 2 ∧ y ≤ 2)

theorem largest_four_digit_by_two_moves :
  ∃ n : ℕ, moves 1405 1 4 0 5 ∧ n = 7705 :=
by
  sorry

end largest_four_digit_by_two_moves_l460_46024


namespace find_x_l460_46054

-- Define the given conditions
def constant_ratio (k : ℚ) : Prop :=
  ∀ (x y : ℚ), (3 * x - 4) / (y + 15) = k

def initial_condition (k : ℚ) : Prop :=
  (3 * 5 - 4) / (4 + 15) = k

def new_condition (k : ℚ) (x : ℚ) : Prop :=
  (3 * x - 4) / 30 = k

-- Prove that x = 406/57 given the conditions
theorem find_x (k : ℚ) (x : ℚ) :
  constant_ratio k →
  initial_condition k →
  new_condition k x →
  x = 406 / 57 :=
  sorry

end find_x_l460_46054


namespace total_cost_price_of_items_l460_46091

/-- 
  Definition of the selling prices of the items A, B, and C.
  Definition of the profit percentages of the items A, B, and C.
  The statement is the total cost price calculation.
-/
def ItemA_SP : ℝ := 800
def ItemA_Profit : ℝ := 0.25
def ItemB_SP : ℝ := 1200
def ItemB_Profit : ℝ := 0.20
def ItemC_SP : ℝ := 1500
def ItemC_Profit : ℝ := 0.30

theorem total_cost_price_of_items :
  let CP_A := ItemA_SP / (1 + ItemA_Profit)
  let CP_B := ItemB_SP / (1 + ItemB_Profit)
  let CP_C := ItemC_SP / (1 + ItemC_Profit)
  CP_A + CP_B + CP_C = 2793.85 :=
by
  sorry

end total_cost_price_of_items_l460_46091


namespace problem_statement_l460_46001

theorem problem_statement (a b c : ℕ) (h1 : a < 12) (h2 : b < 12) (h3 : c < 12) (h4 : b + c = 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b + c :=
by
  sorry

end problem_statement_l460_46001


namespace total_surface_area_is_correct_l460_46018

-- Define the problem constants and structure
def num_cubes := 20
def edge_length := 1
def bottom_layer := 9
def middle_layer := 8
def top_layer := 3
def total_painted_area : ℕ := 55

-- Define a function to calculate the exposed surface area
noncomputable def calc_exposed_area (num_bottom : ℕ) (num_middle : ℕ) (num_top : ℕ) (edge : ℕ) : ℕ := 
    let bottom_exposed := num_bottom * (edge * edge)
    let middle_corners_exposed := 4 * 3 * edge
    let middle_edges_exposed := (num_middle - 4) * (2 * edge)
    let top_exposed := num_top * (5 * edge)
    bottom_exposed + middle_corners_exposed + middle_edges_exposed + top_exposed

-- Statement to prove the total painted area
theorem total_surface_area_is_correct : calc_exposed_area bottom_layer middle_layer top_layer edge_length = total_painted_area :=
by
  -- The proof itself is omitted, focus is on the statement.
  sorry

end total_surface_area_is_correct_l460_46018


namespace savings_correct_l460_46072

-- Define the conditions
def in_store_price : ℝ := 320
def discount_rate : ℝ := 0.05
def monthly_payment : ℝ := 62
def monthly_payments : ℕ := 5
def shipping_handling : ℝ := 10

-- Prove that the savings from buying in-store is 16 dollars.
theorem savings_correct : 
  (monthly_payments * monthly_payment + shipping_handling) - (in_store_price * (1 - discount_rate)) = 16 := 
by
  sorry

end savings_correct_l460_46072


namespace tv_cost_l460_46016

-- Definitions from the problem conditions
def fraction_on_furniture : ℚ := 3 / 4
def total_savings : ℚ := 1800
def fraction_on_tv : ℚ := 1 - fraction_on_furniture  -- Fraction of savings on TV

-- The proof problem statement
theorem tv_cost : total_savings * fraction_on_tv = 450 := by
  sorry

end tv_cost_l460_46016


namespace arcs_intersection_l460_46011

theorem arcs_intersection (k : ℕ) : (1 ≤ k ∧ k ≤ 99) ∧ ¬(∃ m : ℕ, k + 1 = 8 * m) ↔ ∃ n l : ℕ, (2 * l + 1) * 100 = (k + 1) * n ∧ n = 100 ∧ k < 100 := by
  sorry

end arcs_intersection_l460_46011


namespace probability_is_one_twelfth_l460_46025

def probability_red_gt4_green_odd_blue_lt4 : ℚ :=
  let total_outcomes := 6 * 6 * 6
  let successful_outcomes := 2 * 3 * 3
  successful_outcomes / total_outcomes

theorem probability_is_one_twelfth :
  probability_red_gt4_green_odd_blue_lt4 = 1 / 12 :=
by
  -- proof here
  sorry

end probability_is_one_twelfth_l460_46025


namespace rectangle_width_is_16_l460_46068

-- Definitions based on the conditions
def length : ℝ := 24
def ratio := 6 / 5
def perimeter := 80

-- The proposition to prove
theorem rectangle_width_is_16 (W : ℝ) (h1 : length = 24) (h2 : length = ratio * W) (h3 : 2 * length + 2 * W = perimeter) :
  W = 16 :=
by
  sorry

end rectangle_width_is_16_l460_46068


namespace find_k_l460_46063

variable (m n k : ℝ)

def line (x y : ℝ) : Prop := x = 2 * y + 3
def point1_on_line : Prop := line m n
def point2_on_line : Prop := line (m + 2) (n + k)

theorem find_k (h1 : point1_on_line m n) (h2 : point2_on_line m n k) : k = 0 :=
by
  sorry

end find_k_l460_46063


namespace total_students_accommodated_l460_46099

structure BusConfig where
  columns : ℕ
  rows : ℕ
  broken_seats : ℕ

structure SplitBusConfig where
  columns : ℕ
  left_rows : ℕ
  right_rows : ℕ
  broken_seats : ℕ

structure ComplexBusConfig where
  columns : ℕ
  rows : ℕ
  special_rows_broken_seats : ℕ

def bus1 : BusConfig := { columns := 4, rows := 10, broken_seats := 2 }
def bus2 : BusConfig := { columns := 5, rows := 8, broken_seats := 4 }
def bus3 : BusConfig := { columns := 3, rows := 12, broken_seats := 3 }
def bus4 : SplitBusConfig := { columns := 4, left_rows := 6, right_rows := 8, broken_seats := 1 }
def bus5 : SplitBusConfig := { columns := 6, left_rows := 8, right_rows := 10, broken_seats := 5 }
def bus6 : ComplexBusConfig := { columns := 5, rows := 10, special_rows_broken_seats := 4 }

theorem total_students_accommodated :
  let seats_bus1 := (bus1.columns * bus1.rows) - bus1.broken_seats;
  let seats_bus2 := (bus2.columns * bus2.rows) - bus2.broken_seats;
  let seats_bus3 := (bus3.columns * bus3.rows) - bus3.broken_seats;
  let seats_bus4 := (bus4.columns * bus4.left_rows) + (bus4.columns * bus4.right_rows) - bus4.broken_seats;
  let seats_bus5 := (bus5.columns * bus5.left_rows) + (bus5.columns * bus5.right_rows) - bus5.broken_seats;
  let seats_bus6 := (bus6.columns * bus6.rows) - bus6.special_rows_broken_seats;
  seats_bus1 + seats_bus2 + seats_bus3 + seats_bus4 + seats_bus5 + seats_bus6 = 311 :=
sorry

end total_students_accommodated_l460_46099


namespace find_n_l460_46062

theorem find_n (n : ℕ) : (1/5)^35 * (1/4)^18 = 1/(n*(10)^35) → n = 2 :=
by
  sorry

end find_n_l460_46062


namespace sin_70_equals_1_minus_2a_squared_l460_46097

variable (a : ℝ)

theorem sin_70_equals_1_minus_2a_squared (h : Real.sin (10 * Real.pi / 180) = a) :
  Real.sin (70 * Real.pi / 180) = 1 - 2 * a^2 := 
sorry

end sin_70_equals_1_minus_2a_squared_l460_46097


namespace initial_percentage_filled_l460_46086

theorem initial_percentage_filled {P : ℝ} 
  (h1 : 45 + (P / 100) * 100 = (3 / 4) * 100) : 
  P = 30 := by
  sorry

end initial_percentage_filled_l460_46086


namespace translated_graph_symmetric_l460_46021

noncomputable def f (x : ℝ) : ℝ := sorry

theorem translated_graph_symmetric (f : ℝ → ℝ)
  (h_translate : ∀ x, f (x - 1) = e^x)
  (h_symmetric : ∀ x, f x = f (-x)) :
  ∀ x, f x = e^(-x - 1) :=
by
  sorry

end translated_graph_symmetric_l460_46021


namespace symmetric_point_origin_l460_46056

def Point := (ℝ × ℝ × ℝ)

def symmetric_point (P : Point) (O : Point) : Point :=
  let (x, y, z) := P
  let (ox, oy, oz) := O
  (2 * ox - x, 2 * oy - y, 2 * oz - z)

theorem symmetric_point_origin :
  symmetric_point (1, 3, 5) (0, 0, 0) = (-1, -3, -5) :=
by sorry

end symmetric_point_origin_l460_46056


namespace find_A_students_l460_46061

variables (Alan Beth Carlos Diana : Prop)
variable (num_As : ℕ)

def Alan_implies_Beth := Alan → Beth
def Beth_implies_no_Carlos_A := Beth → ¬Carlos
def Carlos_implies_Diana := Carlos → Diana
def Beth_implies_Diana := Beth → Diana

theorem find_A_students 
  (h1 : Alan_implies_Beth Alan Beth)
  (h2 : Beth_implies_no_Carlos_A Beth Carlos)
  (h3 : Carlos_implies_Diana Carlos Diana)
  (h4 : Beth_implies_Diana Beth Diana)
  (h_cond : num_As = 2) :
  (Alan ∧ Beth) ∨ (Beth ∧ Diana) ∨ (Carlos ∧ Diana) :=
by sorry

end find_A_students_l460_46061


namespace prism_height_l460_46057

theorem prism_height (a h : ℝ) 
  (base_side : a = 10) 
  (total_edge_length : 3 * a + 3 * a + 3 * h = 84) : 
  h = 8 :=
by sorry

end prism_height_l460_46057


namespace density_of_second_part_l460_46077

theorem density_of_second_part (ρ₁ : ℝ) (V₁ V : ℝ) (m₁ m : ℝ) (h₁ : ρ₁ = 2700) (h₂ : V₁ = 0.25 * V) (h₃ : m₁ = 0.4 * m) :
  (0.6 * m) / (0.75 * V) = 2160 :=
by
  --- Proof omitted
  sorry

end density_of_second_part_l460_46077


namespace triangle_d_not_right_l460_46051

noncomputable def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_d_not_right :
  ¬is_right_triangle 7 8 13 :=
by sorry

end triangle_d_not_right_l460_46051


namespace video_total_votes_l460_46002

theorem video_total_votes (x : ℕ) (L D : ℕ)
  (h1 : L + D = x)
  (h2 : L - D = 130)
  (h3 : 70 * x = 100 * L) :
  x = 325 :=
by
  sorry

end video_total_votes_l460_46002


namespace proof_of_diagonals_and_angles_l460_46080

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

def sum_of_internal_angles (n : ℕ) : ℕ := (n - 2) * 180

theorem proof_of_diagonals_and_angles :
  let p_diagonals := number_of_diagonals 5
  let o_diagonals := number_of_diagonals 8
  let total_diagonals := p_diagonals + o_diagonals
  let p_internal_angles := sum_of_internal_angles 5
  let o_internal_angles := sum_of_internal_angles 8
  let total_internal_angles := p_internal_angles + o_internal_angles
  total_diagonals = 25 ∧ total_internal_angles = 1620 :=
by
  sorry

end proof_of_diagonals_and_angles_l460_46080


namespace problem_part1_problem_part2_l460_46006

open Real

-- Part (1)
theorem problem_part1 : ∀ x > 0, log x ≤ x - 1 := 
by 
  sorry -- proof goes here


-- Part (2)
theorem problem_part2 : (∀ x > 0, log x ≤ a * x + (a - 1) / x - 1) → 1 ≤ a := 
by 
  sorry -- proof goes here

end problem_part1_problem_part2_l460_46006


namespace problem1_problem2_l460_46049

theorem problem1 : 
  ((-36) * ((1 : ℚ) / 3 - (1 : ℚ) / 2) + 16 / (-2) ^ 3) = 4 :=
sorry

theorem problem2 : 
  ((-5 + 2) * (1 : ℚ)/3 + (5 : ℚ)^2 / -5) = -6 :=
sorry

end problem1_problem2_l460_46049


namespace not_equal_fractions_l460_46000

theorem not_equal_fractions :
  ¬ ((14 / 12 = 7 / 6) ∧
     (1 + 1 / 6 = 7 / 6) ∧
     (21 / 18 = 7 / 6) ∧
     (1 + 2 / 12 = 7 / 6) ∧
     (1 + 1 / 3 = 7 / 6)) :=
by 
  sorry

end not_equal_fractions_l460_46000


namespace line_through_point_with_equal_intercepts_l460_46081

theorem line_through_point_with_equal_intercepts (x y : ℝ) :
  (∃ b : ℝ, 3 * x + y = 0) ∨ (∃ b : ℝ, x - y + 4 = 0) ∨ (∃ b : ℝ, x + y - 2 = 0) :=
  sorry

end line_through_point_with_equal_intercepts_l460_46081


namespace simplify_abs_expression_l460_46003

theorem simplify_abs_expression (a b c : ℝ) (h1 : a + c > b) (h2 : b + c > a) (h3 : a + b > c) :
  |a - b + c| - |a - b - c| = 2 * a - 2 * b :=
by
  sorry

end simplify_abs_expression_l460_46003


namespace no_rectangle_from_six_different_squares_l460_46029

theorem no_rectangle_from_six_different_squares (a1 a2 a3 a4 a5 a6 : ℝ) (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6) :
  ¬ (∃ (L W : ℝ), a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2 = L * W) :=
sorry

end no_rectangle_from_six_different_squares_l460_46029


namespace alice_arrives_earlier_l460_46032

/-
Alice and Bob are heading to a park that is 2 miles away from their home. 
They leave home at the same time. 
Alice cycles to the park at a speed of 12 miles per hour, 
while Bob jogs there at a speed of 6 miles per hour. 
Prove that Alice arrives 10 minutes earlier at the park than Bob.
-/

theorem alice_arrives_earlier 
  (d : ℕ) (a_speed : ℕ) (b_speed : ℕ) (arrival_difference_minutes : ℕ) 
  (h1 : d = 2) 
  (h2 : a_speed = 12) 
  (h3 : b_speed = 6) 
  (h4 : arrival_difference_minutes = 10) 
  : (d / a_speed * 60) + arrival_difference_minutes = d / b_speed * 60 :=
by
  sorry

end alice_arrives_earlier_l460_46032


namespace production_rate_l460_46039

theorem production_rate (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x * x * x = x) → (y * y * z) / x^2 = y^2 * z / x^2 :=
by
  intro h
  sorry

end production_rate_l460_46039


namespace find_m_l460_46067

-- Define the vectors a and b
def veca (m : ℝ) : ℝ × ℝ := (m, 4)
def vecb (m : ℝ) : ℝ × ℝ := (m + 4, 1)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the condition that the dot product of a and b is zero
def are_perpendicular (m : ℝ) : Prop :=
  dot_product (veca m) (vecb m) = 0

-- The goal is to prove that if a and b are perpendicular, then m = -2
theorem find_m (m : ℝ) (h : are_perpendicular m) : m = -2 :=
by {
  -- Proof will be filled here
  sorry
}

end find_m_l460_46067


namespace power_computation_l460_46078

theorem power_computation : (12 ^ (12 / 2)) = 2985984 := by
  sorry

end power_computation_l460_46078


namespace increasing_function_odd_function_l460_46040

noncomputable def f (a x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem increasing_function (a : ℝ) : ∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2 :=
sorry

theorem odd_function (a : ℝ) : (∀ x : ℝ, f a (-x) = - f a x) ↔ a = 1 :=
sorry

end increasing_function_odd_function_l460_46040


namespace union_sets_l460_46008

open Set

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4, 5}

theorem union_sets :
  A ∪ B = {1, 2, 3, 4, 5} :=
by
  sorry

end union_sets_l460_46008


namespace composite_sum_of_ab_l460_46033

theorem composite_sum_of_ab (a b : ℕ) (h : 31 * a = 54 * b) : ∃ k l : ℕ, k > 1 ∧ l > 1 ∧ a + b = k * l :=
sorry

end composite_sum_of_ab_l460_46033


namespace juan_distance_l460_46035

def time : ℝ := 80.0
def speed : ℝ := 10.0
def distance (t : ℝ) (s : ℝ) : ℝ := t * s

theorem juan_distance : distance time speed = 800.0 := by
  sorry

end juan_distance_l460_46035


namespace symmetric_intersection_range_l460_46027

theorem symmetric_intersection_range (k m p : ℝ)
  (intersection_symmetric : ∀ (x y : ℝ), 
    (x = k*y - 1 ∧ (x^2 + y^2 + k*x + m*y + 2*p = 0)) → 
    (y = x)) 
  : p < -3/2 := 
sorry

end symmetric_intersection_range_l460_46027


namespace sum_345_consecutive_sequences_l460_46066

theorem sum_345_consecutive_sequences :
  ∃ (n : ℕ), n = 7 ∧ (∀ (k : ℕ), n ≥ 2 →
    (n * (2 * k + n - 1) = 690 → 2 * k + n - 1 > n)) :=
sorry

end sum_345_consecutive_sequences_l460_46066


namespace percentage_less_than_l460_46064

theorem percentage_less_than (x y z : Real) (h1 : x = 1.20 * y) (h2 : x = 0.84 * z) : 
  ((z - y) / z) * 100 = 30 := 
sorry

end percentage_less_than_l460_46064


namespace servings_of_peanut_butter_l460_46012

-- Definitions from conditions
def total_peanut_butter : ℚ := 35 + 4/5
def serving_size : ℚ := 2 + 1/3

-- Theorem to be proved
theorem servings_of_peanut_butter :
  total_peanut_butter / serving_size = 15 + 17/35 := by
  sorry

end servings_of_peanut_butter_l460_46012


namespace local_extrema_l460_46074

-- Defining the function y = 1 + 3x - x^3
def y (x : ℝ) : ℝ := 1 + 3 * x - x ^ 3

-- Statement of the problem to be proved
theorem local_extrema :
  (∃ x : ℝ, x = -1 ∧ y x = -1 ∧ ∀ ε > 0, ∃ δ > 0, ∀ z, abs (z + 1) < δ → y z ≥ y (-1)) ∧
  (∃ x : ℝ, x = 1 ∧ y x = 3 ∧ ∀ ε > 0, ∃ δ > 0, ∀ z, abs (z - 1) < δ → y z ≤ y 1) :=
by sorry

end local_extrema_l460_46074


namespace sum_p_q_l460_46050

-- Define the cubic polynomial q(x)
def cubic_q (q : ℚ) (x : ℚ) := q * x * (x - 1) * (x + 1)

-- Define the linear polynomial p(x)
def linear_p (p : ℚ) (x : ℚ) := p * x

-- Prove the result for p(x) + q(x)
theorem sum_p_q : 
  (∀ p q : ℚ, linear_p p 4 = 4 → cubic_q q 3 = 3 → (∀ x : ℚ, linear_p p x + cubic_q q x = (1 / 24) * x^3 + (23 / 24) * x)) :=
by
  intros p q hp hq x
  sorry

end sum_p_q_l460_46050


namespace powers_of_i_cyclic_l460_46071

theorem powers_of_i_cyclic {i : ℂ} (h_i_squared : i^2 = -1) :
  i^(66) + i^(103) = -1 - i :=
by {
  -- Providing the proof steps as sorry.
  -- This is a placeholder for the actual proof.
  sorry
}

end powers_of_i_cyclic_l460_46071


namespace problem_statement_l460_46090

theorem problem_statement (x : ℕ) (h : 423 - x = 421) : (x * 423) + 421 = 1267 := by
  sorry

end problem_statement_l460_46090


namespace parallelogram_height_l460_46020

theorem parallelogram_height (A b h : ℝ) (hA : A = 288) (hb : b = 18) : h = 16 :=
by
  sorry

end parallelogram_height_l460_46020


namespace average_student_headcount_l460_46076

theorem average_student_headcount (h1 : ℕ := 10900) (h2 : ℕ := 10500) (h3 : ℕ := 10700) (h4 : ℕ := 11300) : 
  (h1 + h2 + h3 + h4) / 4 = 10850 := 
by 
  sorry

end average_student_headcount_l460_46076


namespace sum_of_integer_n_l460_46070

theorem sum_of_integer_n (n_values : List ℤ) (h : ∀ n ∈ n_values, ∃ k ∈ ({1, 3, 9} : Set ℤ), 2 * n - 1 = k) :
  List.sum n_values = 8 :=
by
  -- this is a placeholder to skip the actual proof
  sorry

end sum_of_integer_n_l460_46070


namespace intersection_complement_l460_46009

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

theorem intersection_complement :
  A ∩ (U \ B) = {2} :=
by {
  sorry
}

end intersection_complement_l460_46009


namespace equivalent_expression_l460_46073

theorem equivalent_expression (m n : ℕ) (P Q : ℕ) (hP : P = 3^m) (hQ : Q = 5^n) :
  15^(m + n) = P * Q :=
by
  sorry

end equivalent_expression_l460_46073


namespace problem_equiv_l460_46026

theorem problem_equiv :
  ((2001 * 2021 + 100) * (1991 * 2031 + 400)) / (2011^4) = 1 :=
by
  sorry

end problem_equiv_l460_46026


namespace certain_event_C_union_D_l460_46065

variable {Ω : Type} -- Omega, the sample space
variable {P : Set Ω → Prop} -- P as the probability function predicates the events

-- Definitions of the events
variable {A B C D : Set Ω}

-- Conditions
def mutually_exclusive (A B : Set Ω) : Prop := ∀ x, x ∈ A → x ∉ B
def complementary (A C : Set Ω) : Prop := ∀ x, x ∈ C ↔ x ∉ A

-- Given conditions
axiom A_and_B_mutually_exclusive : mutually_exclusive A B
axiom C_is_complementary_to_A : complementary A C
axiom D_is_complementary_to_B : complementary B D

-- Theorem statement
theorem certain_event_C_union_D : ∀ x, x ∈ C ∪ D := by
  sorry

end certain_event_C_union_D_l460_46065


namespace avg_goals_l460_46041

-- Let's declare the variables and conditions
def layla_goals : ℕ := 104
def games_played : ℕ := 4
def less_goals_kristin : ℕ := 24

-- Define the number of goals Kristin scored
def kristin_goals : ℕ := layla_goals - less_goals_kristin

-- Calculate the total number of goals scored by both
def total_goals : ℕ := layla_goals + kristin_goals

-- Calculate the average number of goals per game
def average_goals_per_game : ℕ := total_goals / games_played

-- The theorem statement
theorem avg_goals : average_goals_per_game = 46 := by
  -- proof skipped, assume correct by using sorry
  sorry

end avg_goals_l460_46041


namespace total_eggs_collected_by_all_four_l460_46084

def benjamin_eggs := 6
def carla_eggs := 3 * benjamin_eggs
def trisha_eggs := benjamin_eggs - 4
def david_eggs := 2 * trisha_eggs

theorem total_eggs_collected_by_all_four :
  benjamin_eggs + carla_eggs + trisha_eggs + david_eggs = 30 := by
  sorry

end total_eggs_collected_by_all_four_l460_46084


namespace angles_arithmetic_sequence_sides_l460_46034

theorem angles_arithmetic_sequence_sides (A B C a b c : ℝ)
  (h_angle_ABC : A + B + C = 180)
  (h_arithmetic_sequence : 2 * B = A + C)
  (h_cos_B : A * A + c * c - b * b = 2 * a * c)
  (angle_pos : 0 < A ∧ 0 < B ∧ 0 < C ∧ A < 180 ∧ B < 180 ∧ C < 180) :
  (1 / (a + b) + 1 / (b + c) = 3 / (a + b + c)) :=
sorry

end angles_arithmetic_sequence_sides_l460_46034


namespace probability_of_desired_roll_l460_46059

-- Definitions of six-sided dice rolls and probability results
def is_greater_than_four (n : ℕ) : Prop := n > 4
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5

-- Definitions of probabilities based on dice outcomes
def prob_greater_than_four : ℚ := 2 / 6
def prob_prime : ℚ := 3 / 6

-- Definition of joint probability for independent events
def joint_prob : ℚ := prob_greater_than_four * prob_prime

-- Theorem to prove
theorem probability_of_desired_roll : joint_prob = 1 / 6 := 
by
  sorry

end probability_of_desired_roll_l460_46059


namespace remainder_of_power_mod_l460_46092

theorem remainder_of_power_mod :
  (5^2023) % 11 = 4 :=
  by
    sorry

end remainder_of_power_mod_l460_46092


namespace water_level_decrease_l460_46028

theorem water_level_decrease (increase_notation : ℝ) (h : increase_notation = 2) :
  -increase_notation = -2 :=
by
  sorry

end water_level_decrease_l460_46028


namespace problem1_problem2_l460_46010

theorem problem1 : (-5 : ℝ) ^ 0 - (1 / 3) ^ (-2 : ℝ) + (-2 : ℝ) ^ 2 = -4 := 
by
  sorry

variable (a : ℝ)

theorem problem2 : (-3 * a ^ 3) ^ 2 * 2 * a ^ 3 - 8 * a ^ 12 / (2 * a ^ 3) = 14 * a ^ 9 :=
by
  sorry

end problem1_problem2_l460_46010


namespace probability_divisible_by_5_l460_46093

def spinner_nums : List ℕ := [1, 2, 3, 5]

def total_outcomes (spins : ℕ) : ℕ :=
  List.length spinner_nums ^ spins

def count_divisible_by_5 (spins : ℕ) : ℕ :=
  let units_digit := 1
  let rest_combinations := (List.length spinner_nums) ^ (spins - units_digit)
  rest_combinations

theorem probability_divisible_by_5 : 
  let spins := 3 
  let successful_cases := count_divisible_by_5 spins
  let all_cases := total_outcomes spins
  successful_cases / all_cases = 1 / 4 :=
by
  sorry

end probability_divisible_by_5_l460_46093


namespace sufficient_but_not_necessary_l460_46037

theorem sufficient_but_not_necessary (a b : ℝ) (hp : a > 1 ∧ b > 1) (hq : a + b > 2 ∧ a * b > 1) : 
  (a > 1 ∧ b > 1 → a + b > 2 ∧ a * b > 1) ∧ ¬(a + b > 2 ∧ a * b > 1 → a > 1 ∧ b > 1) :=
by
  sorry

end sufficient_but_not_necessary_l460_46037


namespace xiaoming_mirrored_time_l460_46083

-- Define the condition: actual time is 7:10 AM.
def actual_time : (ℕ × ℕ) := (7, 10)

-- Define a function to compute the mirrored time given an actual time.
def mirror_time (h m : ℕ) : (ℕ × ℕ) :=
  let mirrored_minute := if m = 0 then 0 else 60 - m
  let mirrored_hour := if m = 0 then if h = 12 then 12 else (12 - h) % 12
                        else if h = 12 then 11 else (11 - h) % 12
  (mirrored_hour, mirrored_minute)

-- Our goal is to verify that the mirrored time of 7:10 is 4:50.
theorem xiaoming_mirrored_time : mirror_time 7 10 = (4, 50) :=
by
  -- Proof will verify that mirror_time (7, 10) evaluates to (4, 50).
  sorry

end xiaoming_mirrored_time_l460_46083


namespace polar_to_cartesian_coordinates_l460_46038

theorem polar_to_cartesian_coordinates (ρ θ : ℝ) (hρ : ρ = 2) (hθ : θ = 5 * Real.pi / 6) :
  (ρ * Real.cos θ, ρ * Real.sin θ) = (-Real.sqrt 3, 1) :=
by
  sorry

end polar_to_cartesian_coordinates_l460_46038


namespace second_offset_length_l460_46014

-- Definitions based on the given conditions.
def diagonal : ℝ := 24
def offset1 : ℝ := 9
def area_quad : ℝ := 180

-- Statement to prove the length of the second offset.
theorem second_offset_length :
  ∃ h : ℝ, (1 / 2) * diagonal * offset1 + (1 / 2) * diagonal * h = area_quad ∧ h = 6 :=
by
  sorry

end second_offset_length_l460_46014
