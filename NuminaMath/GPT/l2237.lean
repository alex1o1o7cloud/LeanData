import Mathlib

namespace NUMINAMATH_GPT_road_greening_cost_l2237_223796

-- Define constants for the conditions
def l_total : ℕ := 1500
def cost_A : ℕ := 22
def cost_B : ℕ := 25

-- Define variables for the cost per stem
variables (x y : ℕ)

-- Define the conditions from Plan A and Plan B
def plan_A (x y : ℕ) : Prop := 2 * x + 3 * y = cost_A
def plan_B (x y : ℕ) : Prop := x + 5 * y = cost_B

-- System of equations to find x and y
def system_of_equations (x y : ℕ) : Prop := plan_A x y ∧ plan_B x y

-- Define the constraint for the length of road greened according to Plan B
def length_constraint (a : ℕ) : Prop := l_total - a ≥ 2 * a

-- Define the total cost function
def total_cost (a : ℕ) (x y : ℕ) : ℕ := 22 * a + (x + 5 * y) * (l_total - a)

-- Prove the cost per stem and the minimized cost
theorem road_greening_cost :
  (∃ x y, system_of_equations x y ∧ x = 5 ∧ y = 4) ∧
  (∃ a : ℕ, length_constraint a ∧ a = 500 ∧ total_cost a 5 4 = 36000) :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_road_greening_cost_l2237_223796


namespace NUMINAMATH_GPT_inequality_solution_l2237_223712

theorem inequality_solution (b c x : ℝ) (x1 x2 : ℝ)
  (hb_pos : b > 0) (hc_pos : c > 0) 
  (h_eq1 : x1 * x2 = 1) 
  (h_eq2 : -1 + x2 = 2 * x1) 
  (h_b : b = 5 / 2) 
  (h_c : c = 1) 
  : (1 < x ∧ x ≤ 5 / 2) ↔ (1 < x ∧ x ≤ 5 / 2) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2237_223712


namespace NUMINAMATH_GPT_u_1000_eq_2036_l2237_223707

open Nat

def sequence_term (n : ℕ) : ℕ :=
  let sum_to (k : ℕ) := k * (k + 1) / 2
  if n ≤ 0 then 0 else
  let group := (Nat.sqrt (2 * n)) + 1
  let k := n - sum_to (group - 1)
  (group * group) + 4 * (k - 1) - (group % 4)

theorem u_1000_eq_2036 : sequence_term 1000 = 2036 := sorry

end NUMINAMATH_GPT_u_1000_eq_2036_l2237_223707


namespace NUMINAMATH_GPT_find_a_value_l2237_223751

theorem find_a_value 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 2 * y - 1 = 0)
  (M N : ℝ × ℝ)
  (tangent_condition : (N.snd - M.snd) / (N.fst - M.fst) + (M.fst + N.fst - 2) / (M.snd + N.snd) = 0) : 
  a = 3 ∨ a = -2 := 
sorry

end NUMINAMATH_GPT_find_a_value_l2237_223751


namespace NUMINAMATH_GPT_jeff_bought_6_pairs_l2237_223701

theorem jeff_bought_6_pairs (price_of_shoes : ℝ) (num_of_shoes : ℕ) (price_of_jersey : ℝ)
  (h1 : price_of_jersey = (1 / 4) * price_of_shoes)
  (h2 : num_of_shoes * price_of_shoes = 480)
  (h3 : num_of_shoes * price_of_shoes + 4 * price_of_jersey = 560) :
  num_of_shoes = 6 :=
sorry

end NUMINAMATH_GPT_jeff_bought_6_pairs_l2237_223701


namespace NUMINAMATH_GPT_trig_comparison_l2237_223752

theorem trig_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) :
  a = Real.sin (3 * Real.pi / 5) → 
  b = Real.cos (2 * Real.pi / 5) → 
  c = Real.tan (2 * Real.pi / 5) → 
  b < a ∧ a < c :=
by
  intro ha hb hc
  sorry

end NUMINAMATH_GPT_trig_comparison_l2237_223752


namespace NUMINAMATH_GPT_least_n_for_distance_l2237_223723

-- Definitions ensuring our points and distances
def A_0 : (ℝ × ℝ) := (0, 0)

-- Assume we have distance function and equilateral triangles on given coordinates
def is_on_x_axis (p : ℕ → ℝ × ℝ) : Prop := ∀ n, (p n).snd = 0
def is_on_parabola (q : ℕ → ℝ × ℝ) : Prop := ∀ n, (q n).snd = (q n).fst^2
def is_equilateral (p : ℕ → ℝ × ℝ) (q : ℕ → ℝ × ℝ) (n : ℕ) : Prop :=
  let d1 := dist (p (n-1)) (q n)
  let d2 := dist (q n) (p n)
  let d3 := dist (p (n-1)) (p n)
  d1 = d2 ∧ d2 = d3

-- Define the main property we want to prove
def main_property (n : ℕ) (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) : Prop :=
  A 0 = A_0 ∧ is_on_x_axis A ∧ is_on_parabola B ∧
  (∀ k, is_equilateral A B (k+1)) ∧
  dist A_0 (A n) ≥ 200

-- Final theorem statement
theorem least_n_for_distance (A : ℕ → ℝ × ℝ) (B : ℕ → ℝ × ℝ) :
  (∃ n, main_property n A B ∧ (∀ m, main_property m A B → n ≤ m)) ↔ n = 24 := by
  sorry

end NUMINAMATH_GPT_least_n_for_distance_l2237_223723


namespace NUMINAMATH_GPT_arianna_sleeping_hours_l2237_223745

def hours_in_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_on_chores : ℕ := 5
def hours_sleeping : ℕ := hours_in_day - (hours_at_work + hours_on_chores)

theorem arianna_sleeping_hours : hours_sleeping = 13 := by
  sorry

end NUMINAMATH_GPT_arianna_sleeping_hours_l2237_223745


namespace NUMINAMATH_GPT_find_k_l2237_223731

variables {r k : ℝ}
variables {O A B C D : EuclideanSpace ℝ (Fin 3)}

-- Points A, B, C, and D lie on a sphere centered at O with radius r
variables (hA : dist O A = r) (hB : dist O B = r) (hC : dist O C = r) (hD : dist O D = r)
-- The given vector equation
variables (h_eq : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = (0 : EuclideanSpace ℝ (Fin 3)))

theorem find_k (hA : dist O A = r) (hB : dist O B = r) (hC : dist O C = r) (hD : dist O D = r)
(h_eq : 4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) = (0 : EuclideanSpace ℝ (Fin 3))) : 
k = -7 :=
sorry

end NUMINAMATH_GPT_find_k_l2237_223731


namespace NUMINAMATH_GPT_find_center_of_circle_l2237_223773

noncomputable def center_of_circle (θ ρ : ℝ) : Prop :=
  ρ = (1 : ℝ) ∧ θ = (-Real.pi / (3 : ℝ))

theorem find_center_of_circle (θ ρ : ℝ) (h : ρ = Real.cos θ - Real.sqrt 3 * Real.sin θ) :
  center_of_circle θ ρ := by
  sorry

end NUMINAMATH_GPT_find_center_of_circle_l2237_223773


namespace NUMINAMATH_GPT_completing_square_transformation_l2237_223759

theorem completing_square_transformation : ∀ x : ℝ, x^2 - 4 * x - 7 = 0 → (x - 2)^2 = 11 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_completing_square_transformation_l2237_223759


namespace NUMINAMATH_GPT_unique_solution_condition_l2237_223791

noncomputable def unique_solution_system (a b c x y z : ℝ) : Prop :=
  (a * x + b * y - b * z = c) ∧ 
  (a * y + b * x - b * z = c) ∧ 
  (a * z + b * y - b * x = c) → 
  (x = y ∧ y = z ∧ x = c / a)

theorem unique_solution_condition (a b c x y z : ℝ) 
  (h1 : a * x + b * y - b * z = c)
  (h2 : a * y + b * x - b * z = c)
  (h3 : a * z + b * y - b * x = c)
  (ha : a ≠ 0)
  (ha_b : a ≠ b)
  (ha_b' : a + b ≠ 0) :
  unique_solution_system a b c x y z :=
by 
  sorry

end NUMINAMATH_GPT_unique_solution_condition_l2237_223791


namespace NUMINAMATH_GPT_total_books_after_loss_l2237_223748

-- Define variables for the problem
def sandy_books : ℕ := 10
def tim_books : ℕ := 33
def benny_lost_books : ℕ := 24

-- Prove the final number of books together
theorem total_books_after_loss : (sandy_books + tim_books - benny_lost_books) = 19 := by
  sorry

end NUMINAMATH_GPT_total_books_after_loss_l2237_223748


namespace NUMINAMATH_GPT_third_character_has_2_lines_l2237_223706

-- Define the number of lines characters have
variables (x y z : ℕ)

-- The third character has x lines
-- Condition: The second character has 6 more than three times the number of lines the third character has
def second_character_lines : ℕ := 3 * x + 6

-- Condition: The first character has 8 more lines than the second character
def first_character_lines : ℕ := second_character_lines x + 8

-- The first character has 20 lines
def first_character_has_20_lines : Prop := first_character_lines x = 20

-- Prove that the third character has 2 lines
theorem third_character_has_2_lines (h : first_character_has_20_lines x) : x = 2 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_third_character_has_2_lines_l2237_223706


namespace NUMINAMATH_GPT_mean_values_are_two_l2237_223788

noncomputable def verify_means (a b : ℝ) : Prop :=
  (a + b) / 2 = 2 ∧ 2 / ((1 / a) + (1 / b)) = 2

theorem mean_values_are_two (a b : ℝ) (h : verify_means a b) : a = 2 ∧ b = 2 :=
  sorry

end NUMINAMATH_GPT_mean_values_are_two_l2237_223788


namespace NUMINAMATH_GPT_james_bags_l2237_223794

theorem james_bags (total_marbles : ℕ) (remaining_marbles : ℕ) (b : ℕ) (m : ℕ) 
  (h1 : total_marbles = 28) 
  (h2 : remaining_marbles = 21) 
  (h3 : m = total_marbles - remaining_marbles) 
  (h4 : b = total_marbles / m) : 
  b = 4 :=
by
  sorry

end NUMINAMATH_GPT_james_bags_l2237_223794


namespace NUMINAMATH_GPT_no_values_of_g_g_x_eq_one_l2237_223717

-- Define the function g and its properties based on the conditions
variable (g : ℝ → ℝ)
variable (h₁ : g (-4) = 1)
variable (h₂ : g (0) = 1)
variable (h₃ : g (4) = 3)
variable (h₄ : ∀ x, -4 ≤ x ∧ x ≤ 4 → g x ≥ 1)

-- Define the theorem to prove the number of values of x such that g(g(x)) = 1 is zero
theorem no_values_of_g_g_x_eq_one : ∃ n : ℕ, n = 0 ∧ (∀ x, -4 ≤ x ∧ x ≤ 4 → g (g x) = 1 → false) :=
by
  sorry -- proof to be provided later

end NUMINAMATH_GPT_no_values_of_g_g_x_eq_one_l2237_223717


namespace NUMINAMATH_GPT_range_of_a_l2237_223729

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f x ≥ a * x) ↔ (a ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2237_223729


namespace NUMINAMATH_GPT_missing_fraction_is_correct_l2237_223765

def sum_of_fractions (x : ℚ) : Prop :=
  (1/3 : ℚ) + (1/2) + (-5/6) + (1/5) + (1/4) + (-9/20) + x = (45/100 : ℚ)

theorem missing_fraction_is_correct : sum_of_fractions (27/60 : ℚ) :=
  by sorry

end NUMINAMATH_GPT_missing_fraction_is_correct_l2237_223765


namespace NUMINAMATH_GPT_quadratic_has_one_solution_l2237_223793

theorem quadratic_has_one_solution (q : ℚ) (hq : q ≠ 0) : 
  (∃ x, ∀ y, q*y^2 - 18*y + 8 = 0 → x = y) ↔ q = 81 / 8 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_one_solution_l2237_223793


namespace NUMINAMATH_GPT_not_necessarily_a_squared_lt_b_squared_l2237_223719
-- Import the necessary library

-- Define the variables and the condition
variables {a b : ℝ}
axiom h : a < b

-- The theorem statement that needs to be proved/disproved
theorem not_necessarily_a_squared_lt_b_squared (a b : ℝ) (h : a < b) : ¬ (a^2 < b^2) :=
sorry

end NUMINAMATH_GPT_not_necessarily_a_squared_lt_b_squared_l2237_223719


namespace NUMINAMATH_GPT_t_shirt_cost_l2237_223783

theorem t_shirt_cost (T : ℕ) 
  (h1 : 3 * T + 50 = 110) : T = 20 := 
by
  sorry

end NUMINAMATH_GPT_t_shirt_cost_l2237_223783


namespace NUMINAMATH_GPT_max_xyz_l2237_223722

theorem max_xyz (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
(h4 : (x * y) + 3 * z = (x + 3 * z) * (y + 3 * z)) 
: ∀ x y z, ∃ (a : ℝ), a = (x * y * z) ∧ a ≤ (1/81) :=
sorry

end NUMINAMATH_GPT_max_xyz_l2237_223722


namespace NUMINAMATH_GPT_marble_ratio_l2237_223740

-- Definitions and assumptions from the conditions
def my_marbles : ℕ := 16
def total_marbles : ℕ := 63
def transfer_amount : ℕ := 2

-- After transferring marbles to my brother
def my_marbles_after_transfer := my_marbles - transfer_amount
def brother_marbles (B : ℕ) := B + transfer_amount

-- Friend's marbles
def friend_marbles (F : ℕ) := F = 3 * my_marbles_after_transfer

-- Prove the ratio of marbles after transfer
theorem marble_ratio (B F : ℕ) (hf : F = 3 * my_marbles_after_transfer) (h_total : my_marbles + B + F = total_marbles)
  (h_multiple : ∃ M : ℕ, my_marbles_after_transfer = M * brother_marbles B) :
  (my_marbles_after_transfer : ℚ) / (brother_marbles B : ℚ) = 2 / 1 :=
by
  sorry

end NUMINAMATH_GPT_marble_ratio_l2237_223740


namespace NUMINAMATH_GPT_b_divisible_by_8_l2237_223730

theorem b_divisible_by_8 (b : ℕ) (h_even: ∃ k : ℕ, b = 2 * k) (h_square: ∃ n : ℕ, n > 1 ∧ ∃ m : ℕ, (b ^ n - 1) / (b - 1) = m ^ 2) : b % 8 = 0 := 
by
  sorry

end NUMINAMATH_GPT_b_divisible_by_8_l2237_223730


namespace NUMINAMATH_GPT_num_intersections_l2237_223713

noncomputable def polar_to_cartesian (r θ: ℝ): ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem num_intersections (θ: ℝ): 
  let c1 := polar_to_cartesian (6 * Real.cos θ) θ
  let c2 := polar_to_cartesian (10 * Real.sin θ) θ
  let (x1, y1) := c1
  let (x2, y2) := c2
  ((x1 - 3)^2 + y1^2 = 9 ∧ x2^2 + (y2 - 5)^2 = 25) →
  (x1, y1) = (x2, y2) ↔ false :=
by
  sorry

end NUMINAMATH_GPT_num_intersections_l2237_223713


namespace NUMINAMATH_GPT_prove_a_lt_zero_l2237_223708

variable (a b c : ℝ)

-- Define the quadratic function
def f (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions:
-- The polynomial has roots at -2 and 3
def has_roots : Prop := 
  a ≠ 0 ∧ (a * (-2)^2 + b * (-2) + c = 0) ∧ (a * 3^2 + b * 3 + c = 0)

-- f(-b/(2*a)) > 0
def vertex_positive : Prop := 
  f a b c (-b / (2 * a)) > 0

-- Target: Prove a < 0
theorem prove_a_lt_zero 
  (h_roots : has_roots a b c)
  (h_vertex : vertex_positive a b c) : a < 0 := 
sorry

end NUMINAMATH_GPT_prove_a_lt_zero_l2237_223708


namespace NUMINAMATH_GPT_AC_eq_200_l2237_223756

theorem AC_eq_200 (A B C : ℕ) (h1 : A + B + C = 500) (h2 : B + C = 330) (h3 : C = 30) : A + C = 200 := by
  sorry

end NUMINAMATH_GPT_AC_eq_200_l2237_223756


namespace NUMINAMATH_GPT_rate_of_stream_l2237_223739

-- Definitions from problem conditions
def rowing_speed_still_water : ℕ := 24

-- Assume v is the rate of the stream
variable (v : ℕ)

-- Time taken to row up is three times the time taken to row down
def rowing_time_condition : Prop :=
  1 / (rowing_speed_still_water - v) = 3 * (1 / (rowing_speed_still_water + v))

-- The rate of the stream (v) should be 12 kmph
theorem rate_of_stream (h : rowing_time_condition v) : v = 12 :=
  sorry

end NUMINAMATH_GPT_rate_of_stream_l2237_223739


namespace NUMINAMATH_GPT_cousin_age_result_l2237_223715

-- Let define the ages
def rick_age : ℕ := 15
def oldest_brother_age : ℕ := 2 * rick_age
def middle_brother_age : ℕ := oldest_brother_age / 3
def smallest_brother_age : ℕ := middle_brother_age / 2
def youngest_brother_age : ℕ := smallest_brother_age - 2
def cousin_age : ℕ := 5 * youngest_brother_age

-- The theorem stating the cousin's age.
theorem cousin_age_result : cousin_age = 15 := by
  sorry

end NUMINAMATH_GPT_cousin_age_result_l2237_223715


namespace NUMINAMATH_GPT_solution_l2237_223720

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, g (-y) = g y

def problem (f g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y) ∧
  (f 0 = 0) ∧
  (∃ x : ℝ, f x ≠ 0)

theorem solution (f g : ℝ → ℝ) (h : problem f g) : is_odd f ∧ is_even g :=
sorry

end NUMINAMATH_GPT_solution_l2237_223720


namespace NUMINAMATH_GPT_probability_exact_four_out_of_twelve_dice_is_approx_0_089_l2237_223789

noncomputable def dice_probability_exact_four_six : ℝ :=
  let p := (1/6 : ℝ)
  let q := (5/6 : ℝ)
  (Nat.choose 12 4) * (p ^ 4) * (q ^ 8)

theorem probability_exact_four_out_of_twelve_dice_is_approx_0_089 :
  abs (dice_probability_exact_four_six - 0.089) < 0.001 :=
sorry

end NUMINAMATH_GPT_probability_exact_four_out_of_twelve_dice_is_approx_0_089_l2237_223789


namespace NUMINAMATH_GPT_parametric_to_standard_l2237_223762

theorem parametric_to_standard (t : ℝ) : 
  (x = (2 + 3 * t) / (1 + t)) ∧ (y = (1 - 2 * t) / (1 + t)) → (3 * x + y - 7 = 0) ∧ (x ≠ 3) := 
by 
  sorry

end NUMINAMATH_GPT_parametric_to_standard_l2237_223762


namespace NUMINAMATH_GPT_even_product_implies_sum_of_squares_odd_product_implies_no_sum_of_squares_l2237_223767

theorem even_product_implies_sum_of_squares (a b : ℕ) (h : ∃ (a b : ℕ), a * b % 2 = 0 → ∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2) : 
  ∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2 :=
sorry

theorem odd_product_implies_no_sum_of_squares (a b : ℕ) (h : ∃ (a b : ℕ), a * b % 2 ≠ 0 → ¬∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2) : 
  ¬∃ (c d : ℕ), a^2 + b^2 + c^2 = d^2 :=
sorry

end NUMINAMATH_GPT_even_product_implies_sum_of_squares_odd_product_implies_no_sum_of_squares_l2237_223767


namespace NUMINAMATH_GPT_find_special_n_l2237_223784

open Nat

theorem find_special_n (m : ℕ) (hm : m ≥ 3) :
  ∃ (n : ℕ), 
    (n = m^2 - 2) ∧ (∃ (k : ℕ), 1 ≤ k ∧ k < n ∧ 2 * (Nat.choose n k) = (Nat.choose n (k - 1) + Nat.choose n (k + 1))) :=
by
  sorry

end NUMINAMATH_GPT_find_special_n_l2237_223784


namespace NUMINAMATH_GPT_a_and_c_can_complete_in_20_days_l2237_223792

-- Define the work rates for the pairs given in the conditions.
variables {A B C : ℚ}

-- a and b together can complete the work in 12 days
axiom H1 : A + B = 1 / 12

-- b and c together can complete the work in 15 days
axiom H2 : B + C = 1 / 15

-- a, b, and c together can complete the work in 10 days
axiom H3 : A + B + C = 1 / 10

-- We aim to prove that a and c together can complete the work in 20 days,
-- hence their combined work rate should be 1 / 20.
theorem a_and_c_can_complete_in_20_days : A + C = 1 / 20 :=
by
  -- sorry will be used to skip the proof
  sorry

end NUMINAMATH_GPT_a_and_c_can_complete_in_20_days_l2237_223792


namespace NUMINAMATH_GPT_find_x_l2237_223774

theorem find_x (x : ℝ) (h : 0.60 / x = 6 / 2) : x = 0.2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_l2237_223774


namespace NUMINAMATH_GPT_maximize_distance_l2237_223778

theorem maximize_distance (D_F D_R : ℕ) (x y : ℕ) (h1 : D_F = 21000) (h2 : D_R = 28000)
  (h3 : x + y ≤ D_F) (h4 : x + y ≤ D_R) :
  x + y = 24000 :=
sorry

end NUMINAMATH_GPT_maximize_distance_l2237_223778


namespace NUMINAMATH_GPT_gcd_polynomial_l2237_223718

theorem gcd_polynomial (b : ℤ) (h1 : ∃ k : ℤ, b = 7 * k ∧ k % 2 = 1) : 
  Int.gcd (3 * b ^ 2 + 34 * b + 76) (b + 16) = 7 := 
sorry

end NUMINAMATH_GPT_gcd_polynomial_l2237_223718


namespace NUMINAMATH_GPT_tallest_giraffe_height_l2237_223727

theorem tallest_giraffe_height :
  ∃ (height : ℕ), height = 96 ∧ (height = 68 + 28) := by
  sorry

end NUMINAMATH_GPT_tallest_giraffe_height_l2237_223727


namespace NUMINAMATH_GPT_arithmetic_sequence_length_l2237_223757

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ (a_1 d a_n : ℤ), a_1 = -3 ∧ d = 4 ∧ a_n = 45 → n = 13 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_length_l2237_223757


namespace NUMINAMATH_GPT_length_of_faster_train_is_correct_l2237_223761

def speed_faster_train := 54 -- kmph
def speed_slower_train := 36 -- kmph
def crossing_time := 27 -- seconds

def kmph_to_mps (s : ℕ) : ℕ :=
  s * 1000 / 3600

def relative_speed_faster_train := kmph_to_mps (speed_faster_train - speed_slower_train)

def length_faster_train := relative_speed_faster_train * crossing_time

theorem length_of_faster_train_is_correct : length_faster_train = 135 := 
  by
  sorry

end NUMINAMATH_GPT_length_of_faster_train_is_correct_l2237_223761


namespace NUMINAMATH_GPT_simplify_expression_l2237_223705

open Real

theorem simplify_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (( (x + 2) ^ 2 * (x ^ 2 - 2 * x + 2) ^ 2 / (x ^ 3 + 8) ^ 2 ) ^ 2 *
   ( (x - 2) ^ 2 * (x ^ 2 + 2 * x + 2) ^ 2 / (x ^ 3 - 8) ^ 2 ) ^ 2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2237_223705


namespace NUMINAMATH_GPT_james_collects_15_gallons_per_inch_l2237_223769

def rain_gallons_per_inch (G : ℝ) : Prop :=
  let monday_rain := 4
  let tuesday_rain := 3
  let price_per_gallon := 1.2
  let total_money := 126
  let total_rain := monday_rain + tuesday_rain
  (total_rain * G = total_money / price_per_gallon)

theorem james_collects_15_gallons_per_inch : rain_gallons_per_inch 15 :=
by
  -- This is the theorem statement; the proof is not required.
  sorry

end NUMINAMATH_GPT_james_collects_15_gallons_per_inch_l2237_223769


namespace NUMINAMATH_GPT_find_parallel_line_l2237_223786

def line1 : ℝ → ℝ → Prop := λ x y => 2 * x - 3 * y + 2 = 0
def line2 : ℝ → ℝ → Prop := λ x y => 3 * x - 4 * y + 2 = 0
def parallelLine : ℝ → ℝ → Prop := λ x y => 4 * x + y - 4 = 0

theorem find_parallel_line (x y : ℝ) (hx : line1 x y) (hy : line2 x y) : 
  ∃ c : ℝ, (λ x y => 4 * x + y + c = 0) (2:ℝ) (2:ℝ) ∧ 
          ∀ x' y', (λ x' y' => 4 * x' + y' + c = 0) x' y' ↔ 4 * x' + y' - 10 = 0 := 
sorry

end NUMINAMATH_GPT_find_parallel_line_l2237_223786


namespace NUMINAMATH_GPT_find_orange_shells_l2237_223716

theorem find_orange_shells :
  ∀ (total purple pink yellow blue : ℕ),
    total = 65 → purple = 13 → pink = 8 → yellow = 18 → blue = 12 →
    total - (purple + pink + yellow + blue) = 14 :=
by
  intros total purple pink yellow blue h_total h_purple h_pink h_yellow h_blue
  have h := h_total.symm
  rw [h_purple, h_pink, h_yellow, h_blue]
  simp only [Nat.add_assoc, Nat.add_comm, Nat.add_sub_cancel]
  sorry

end NUMINAMATH_GPT_find_orange_shells_l2237_223716


namespace NUMINAMATH_GPT_find_m_and_n_l2237_223772

theorem find_m_and_n (x y m n : ℝ) 
  (h1 : 5 * x - 2 * y = 3) 
  (h2 : m * x + 5 * y = 4) 
  (h3 : x - 4 * y = -3) 
  (h4 : 5 * x + n * y = 1) :
  m = -1 ∧ n = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_and_n_l2237_223772


namespace NUMINAMATH_GPT_necessary_condition_to_contain_circle_in_parabola_l2237_223755

def M (x y : ℝ) : Prop := y ≥ x^2
def N (x y a : ℝ) : Prop := x^2 + (y - a)^2 ≤ 1

theorem necessary_condition_to_contain_circle_in_parabola (a : ℝ) : 
  (∀ x y, N x y a → M x y) ↔ a ≥ 5 / 4 := 
sorry

end NUMINAMATH_GPT_necessary_condition_to_contain_circle_in_parabola_l2237_223755


namespace NUMINAMATH_GPT_market_survey_l2237_223746

theorem market_survey (X Y Z : ℕ) (h1 : X / Y = 3)
  (h2 : X / Z = 2 / 3) (h3 : X = 60) : X + Y + Z = 170 :=
by
  sorry

end NUMINAMATH_GPT_market_survey_l2237_223746


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l2237_223738

noncomputable def sufficient_but_not_necessary (x y : ℝ) : Prop :=
  (x > 1 ∧ y > 1) → (x + y > 2) ∧ (x + y > 2 → ¬(x > 1 ∧ y > 1))

theorem sufficient_not_necessary_condition (x y : ℝ) :
  sufficient_but_not_necessary x y :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l2237_223738


namespace NUMINAMATH_GPT_max_value_g_l2237_223798

def g (x : ℝ) : ℝ := 4 * x - x ^ 4

theorem max_value_g : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 2 ∧ ∀ y : ℝ, (0 ≤ y ∧ y ≤ 2) → g y ≤ g x) ∧ g x = 3 :=
by
  sorry

end NUMINAMATH_GPT_max_value_g_l2237_223798


namespace NUMINAMATH_GPT_mixed_doubles_teams_l2237_223732

theorem mixed_doubles_teams (males females : ℕ) (hm : males = 6) (hf : females = 7) : (males * females) = 42 :=
by
  sorry

end NUMINAMATH_GPT_mixed_doubles_teams_l2237_223732


namespace NUMINAMATH_GPT_cooking_people_count_l2237_223726

variables (P Y W : ℕ)

def people_practicing_yoga := 25
def people_studying_weaving := 8
def people_studying_only_cooking := 2
def people_studying_cooking_and_yoga := 7
def people_studying_cooking_and_weaving := 3
def people_studying_all_curriculums := 3

theorem cooking_people_count :
  P = people_studying_only_cooking + (people_studying_cooking_and_yoga - people_studying_all_curriculums)
    + (people_studying_cooking_and_weaving - people_studying_all_curriculums) + people_studying_all_curriculums →
  P = 9 :=
by
  intro h
  unfold people_studying_only_cooking people_studying_cooking_and_yoga people_studying_cooking_and_weaving people_studying_all_curriculums at h
  sorry

end NUMINAMATH_GPT_cooking_people_count_l2237_223726


namespace NUMINAMATH_GPT_range_of_f_l2237_223704

open Real

noncomputable def f (x y z w : ℝ) : ℝ :=
  x / (x + y) + y / (y + z) + z / (z + x) + w / (w + x)

theorem range_of_f (x y z w : ℝ) (h1x : 0 < x) (h1y : 0 < y) (h1z : 0 < z) (h1w : 0 < w) :
  1 < f x y z w ∧ f x y z w < 2 :=
  sorry

end NUMINAMATH_GPT_range_of_f_l2237_223704


namespace NUMINAMATH_GPT_time_for_c_l2237_223780

   variable (A B C : ℚ)

   -- Conditions
   def condition1 : Prop := (A + B = 1/6)
   def condition2 : Prop := (B + C = 1/8)
   def condition3 : Prop := (C + A = 1/12)

   -- Theorem to be proved
   theorem time_for_c (h1 : condition1 A B) (h2 : condition2 B C) (h3 : condition3 C A) :
     1 / C = 48 :=
   sorry
   
end NUMINAMATH_GPT_time_for_c_l2237_223780


namespace NUMINAMATH_GPT_sector_radius_l2237_223797

theorem sector_radius (r : ℝ) (h1 : r > 0) 
  (h2 : ∀ (l : ℝ), l = r → 
    (3 * r) / (1 / 2 * r^2) = 2) : r = 3 := 
sorry

end NUMINAMATH_GPT_sector_radius_l2237_223797


namespace NUMINAMATH_GPT_VehicleB_travel_time_l2237_223776

theorem VehicleB_travel_time 
    (v_A v_B : ℝ)
    (d : ℝ)
    (h1 : d = 3 * (v_A + v_B))
    (h2 : 3 * v_A = d / 2)
    (h3 : ∀ t ≤ 3.5 , d - t * v_B - 0.5 * v_A = 0)
    : d / v_B = 7.2 :=
by
  sorry

end NUMINAMATH_GPT_VehicleB_travel_time_l2237_223776


namespace NUMINAMATH_GPT_not_possible_values_l2237_223736

theorem not_possible_values (t h d : ℕ) (ht : 3 * t - 6 * h = 2001) (hd : t - h = d) (hh : 6 * h > 0) :
  ∃ n, n = 667 ∧ ∀ d : ℕ, d ≤ 667 → ¬ (t = h + d ∧ 3 * (h + d) - 6 * h = 2001) :=
by
  sorry

end NUMINAMATH_GPT_not_possible_values_l2237_223736


namespace NUMINAMATH_GPT_price_reduction_is_50_rubles_l2237_223735

theorem price_reduction_is_50_rubles :
  let P_Feb : ℕ := 300
  let P_Mar : ℕ := 250
  P_Feb - P_Mar = 50 :=
by
  let P_Feb : ℕ := 300
  let P_Mar : ℕ := 250
  sorry

end NUMINAMATH_GPT_price_reduction_is_50_rubles_l2237_223735


namespace NUMINAMATH_GPT_black_white_tile_ratio_l2237_223711

/-- Assume the original pattern has 12 black tiles and 25 white tiles.
    The pattern is extended by attaching a border of black tiles two tiles wide around the square.
    Prove that the ratio of black tiles to white tiles in the new extended pattern is 76/25.-/
theorem black_white_tile_ratio 
  (original_black_tiles : ℕ)
  (original_white_tiles : ℕ)
  (black_border_width : ℕ)
  (new_black_tiles : ℕ)
  (total_new_tiles : ℕ) 
  (total_old_tiles : ℕ) 
  (new_white_tiles : ℕ)
  : original_black_tiles = 12 → 
    original_white_tiles = 25 → 
    black_border_width = 2 → 
    total_old_tiles = 36 →
    total_new_tiles = 100 →
    new_black_tiles = 76 → 
    new_white_tiles = 25 → 
    (new_black_tiles : ℚ) / (new_white_tiles : ℚ) = 76 / 25 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_black_white_tile_ratio_l2237_223711


namespace NUMINAMATH_GPT_find_t_correct_l2237_223724

theorem find_t_correct : 
  ∃ t : ℝ, (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (5 * x^2 + t * x + 15) = 15 * x^4 - 47 * x^3 + 115 * x^2 - 110 * x + 75) ∧ t = -10 :=
sorry

end NUMINAMATH_GPT_find_t_correct_l2237_223724


namespace NUMINAMATH_GPT_expected_faces_rolled_six_times_l2237_223725

-- Define a random variable indicating appearance of a particular face
noncomputable def ζi (n : ℕ): ℝ := if n > 0 then 1 - (5 / 6) ^ 6 else 0

-- Define the expected number of distinct faces
noncomputable def expected_distinct_faces : ℝ := 6 * ζi 1

theorem expected_faces_rolled_six_times :
  expected_distinct_faces = (6 ^ 6 - 5 ^ 6) / 6 ^ 5 :=
by
  -- Here we would provide the proof
  sorry

end NUMINAMATH_GPT_expected_faces_rolled_six_times_l2237_223725


namespace NUMINAMATH_GPT_find_ordered_pair_l2237_223709

theorem find_ordered_pair (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (hroot : ∀ x : ℝ, 2 * x^2 + a * x + b = 0 → x = a ∨ x = b) :
  (a, b) = (1 / 2, -3 / 4) := 
  sorry

end NUMINAMATH_GPT_find_ordered_pair_l2237_223709


namespace NUMINAMATH_GPT_ratio_of_areas_of_squares_l2237_223764

theorem ratio_of_areas_of_squares (a_side b_side : ℕ) (h_a : a_side = 36) (h_b : b_side = 42) : 
  (a_side ^ 2 : ℚ) / (b_side ^ 2 : ℚ) = 36 / 49 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_squares_l2237_223764


namespace NUMINAMATH_GPT_total_number_of_people_on_bus_l2237_223777

theorem total_number_of_people_on_bus (boys girls : ℕ)
    (driver assistant teacher : ℕ) 
    (h1 : boys = 50)
    (h2 : girls = boys + (2 * boys / 5))
    (h3 : driver = 1)
    (h4 : assistant = 1)
    (h5 : teacher = 1) :
    (boys + girls + driver + assistant + teacher = 123) :=
by
    sorry

end NUMINAMATH_GPT_total_number_of_people_on_bus_l2237_223777


namespace NUMINAMATH_GPT_mask_production_l2237_223710

theorem mask_production (x : ℝ) :
  24 + 24 * (1 + x) + 24 * (1 + x)^2 = 88 :=
sorry

end NUMINAMATH_GPT_mask_production_l2237_223710


namespace NUMINAMATH_GPT_evaporation_period_l2237_223754

theorem evaporation_period
  (initial_amount : ℚ)
  (evaporation_rate : ℚ)
  (percentage_evaporated : ℚ)
  (actual_days : ℚ)
  (h_initial : initial_amount = 10)
  (h_evap_rate : evaporation_rate = 0.007)
  (h_percentage : percentage_evaporated = 3.5000000000000004)
  (h_days : actual_days = (percentage_evaporated / 100) * initial_amount / evaporation_rate):
  actual_days = 50 := by
  sorry

end NUMINAMATH_GPT_evaporation_period_l2237_223754


namespace NUMINAMATH_GPT_triangle_right_angled_and_common_difference_equals_inscribed_circle_radius_l2237_223741

noncomputable def a : ℝ := sorry
noncomputable def d : ℝ := a / 4
noncomputable def half_perimeter : ℝ := (a - d + a + (a + d)) / 2
noncomputable def r : ℝ := ((a - d) + a + (a + d)) / 2

theorem triangle_right_angled_and_common_difference_equals_inscribed_circle_radius :
  (half_perimeter > a + d) →
  ((a - d) + a + (a + d) = 2 * half_perimeter) →
  (a - d)^2 + a^2 = (a + d)^2 →
  d = r :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_triangle_right_angled_and_common_difference_equals_inscribed_circle_radius_l2237_223741


namespace NUMINAMATH_GPT_acute_angle_sum_eq_pi_div_two_l2237_223703

open Real

theorem acute_angle_sum_eq_pi_div_two (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h_eq : sin α ^ 2 + sin β ^ 2 = sin (α + β)) : 
  α + β = π / 2 :=
sorry

end NUMINAMATH_GPT_acute_angle_sum_eq_pi_div_two_l2237_223703


namespace NUMINAMATH_GPT_polynomials_equal_l2237_223760

noncomputable def P : ℝ → ℝ := sorry -- assume P is a nonconstant polynomial
noncomputable def Q : ℝ → ℝ := sorry -- assume Q is a nonconstant polynomial

axiom floor_eq_for_all_y (y : ℝ) : ⌊P y⌋ = ⌊Q y⌋

theorem polynomials_equal (x : ℝ) : P x = Q x :=
by
  sorry

end NUMINAMATH_GPT_polynomials_equal_l2237_223760


namespace NUMINAMATH_GPT_distance_between_cities_l2237_223714

def distance_thing 
  (d_A d_B : ℝ) 
  (v_A v_B : ℝ) 
  (t_diff : ℝ) : Prop :=
d_A = (3 / 5) * d_B ∧
v_A = 72 ∧
v_B = 108 ∧
t_diff = (1 / 4) ∧
(d_A + d_B) = 432

theorem distance_between_cities
  (d_A d_B : ℝ)
  (v_A v_B : ℝ)
  (t_diff : ℝ)
  (h : distance_thing d_A d_B v_A v_B t_diff)
  : d_A + d_B = 432 := by
  sorry

end NUMINAMATH_GPT_distance_between_cities_l2237_223714


namespace NUMINAMATH_GPT_unique_prime_range_start_l2237_223781

theorem unique_prime_range_start (N : ℕ) (hN : N = 220) (h1 : ∀ n, N ≥ n → n ≥ 211 → ¬Prime n) (h2 : Prime 211) : N - 8 = 212 :=
by
  sorry

end NUMINAMATH_GPT_unique_prime_range_start_l2237_223781


namespace NUMINAMATH_GPT_sum_of_six_primes_even_l2237_223770

/-- If A, B, and C are positive integers such that A, B, C, A-B, A+B, and A+B+C are all prime numbers, 
    and B is specifically the prime number 2,
    then the sum of these six primes is even. -/
theorem sum_of_six_primes_even (A B C : ℕ) (hA : Prime A) (hB : Prime B) (hC : Prime C) 
    (h1 : Prime (A - B)) (h2 : Prime (A + B)) (h3 : Prime (A + B + C)) (hB_eq_two : B = 2) : 
    Even (A + B + C + (A - B) + (A + B) + (A + B + C)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_six_primes_even_l2237_223770


namespace NUMINAMATH_GPT_quadratic_trinomial_has_two_roots_l2237_223744

theorem quadratic_trinomial_has_two_roots
  (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : (2 * (a + b))^2 - 4 * 3 * a * (b + c) > 0 := by
  sorry

end NUMINAMATH_GPT_quadratic_trinomial_has_two_roots_l2237_223744


namespace NUMINAMATH_GPT_remainder_when_divided_by_9_l2237_223763

open Nat

theorem remainder_when_divided_by_9 (A B : ℕ) (h : A = B * 9 + 13) : A % 9 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_9_l2237_223763


namespace NUMINAMATH_GPT_problem_b_problem_d_l2237_223749

variable (x y t : ℝ)

def condition_curve (t : ℝ) : Prop :=
  ∃ C : ℝ × ℝ → Prop, ∀ x y : ℝ, C (x, y) ↔ (x^2 / (5 - t) + y^2 / (t - 1) = 1)

theorem problem_b (h1 : t < 1) : condition_curve t → ∃ (C : ℝ × ℝ → Prop), (∀ x y, C (x, y) ↔ x^2 / (5 - t) + y^2 / (t - 1) = 1) → ¬(5 - t) < 0 ∧ (t - 1) < 0 := 
sorry

theorem problem_d (h1 : 3 < t) (h2 : t < 5) (h3 : condition_curve t) : ∃ (C : ℝ × ℝ → Prop), (∀ x y, C (x, y) ↔ x^2 / (5 - t) + y^2 / (t - 1) = 1) → 0 < (t - 1) ∧ (t - 1) > (5 - t) := 
sorry

end NUMINAMATH_GPT_problem_b_problem_d_l2237_223749


namespace NUMINAMATH_GPT_tan_sum_identity_l2237_223795

theorem tan_sum_identity (α : ℝ) (h : Real.tan α = 1 / 2) : Real.tan (α + π / 4) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_sum_identity_l2237_223795


namespace NUMINAMATH_GPT_total_cost_after_discount_l2237_223702

def num_children : Nat := 6
def num_adults : Nat := 10
def num_seniors : Nat := 4

def child_ticket_price : Real := 12
def adult_ticket_price : Real := 20
def senior_ticket_price : Real := 15

def group_discount_rate : Real := 0.15

theorem total_cost_after_discount :
  let total_cost_before_discount :=
    num_children * child_ticket_price +
    num_adults * adult_ticket_price +
    num_seniors * senior_ticket_price
  let discount := group_discount_rate * total_cost_before_discount
  let total_cost := total_cost_before_discount - discount
  total_cost = 282.20 := by
  sorry

end NUMINAMATH_GPT_total_cost_after_discount_l2237_223702


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_q_l2237_223787

theorem sufficient_but_not_necessary_condition_for_q (k : ℝ) :
  (∀ x : ℝ, x ≥ k → x^2 - x > 2) ∧ (∃ x : ℝ, x < k ∧ x^2 - x > 2) ↔ k > 2 :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_q_l2237_223787


namespace NUMINAMATH_GPT_value_of_f_at_6_l2237_223747

-- The condition that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- The condition that f(x + 2) = -f(x)
def periodic_sign_flip (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = -f (x)

-- The theorem statement
theorem value_of_f_at_6 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : periodic_sign_flip f) : f 6 = 0 :=
sorry

end NUMINAMATH_GPT_value_of_f_at_6_l2237_223747


namespace NUMINAMATH_GPT_monochromatic_triangle_probability_correct_l2237_223734

noncomputable def monochromatic_triangle_probability (p : ℝ) : ℝ :=
  1 - (3 * (p^2) * (1 - p) + 3 * ((1 - p)^2) * p)^20

theorem monochromatic_triangle_probability_correct :
  monochromatic_triangle_probability (1/2) = 1 - (3/4)^20 :=
by
  sorry

end NUMINAMATH_GPT_monochromatic_triangle_probability_correct_l2237_223734


namespace NUMINAMATH_GPT_hotel_charge_per_hour_morning_l2237_223733

noncomputable def charge_per_hour_morning := 2 -- The correct answer

theorem hotel_charge_per_hour_morning
  (cost_night : ℝ)
  (initial_money : ℝ)
  (hours_night : ℝ)
  (hours_morning : ℝ)
  (remaining_money : ℝ)
  (total_cost : ℝ)
  (M : ℝ)
  (H1 : cost_night = 1.50)
  (H2 : initial_money = 80)
  (H3 : hours_night = 6)
  (H4 : hours_morning = 4)
  (H5 : remaining_money = 63)
  (H6 : total_cost = initial_money - remaining_money)
  (H7 : total_cost = hours_night * cost_night + hours_morning * M) :
  M = charge_per_hour_morning :=
by
  sorry

end NUMINAMATH_GPT_hotel_charge_per_hour_morning_l2237_223733


namespace NUMINAMATH_GPT_sarah_problem_l2237_223700

theorem sarah_problem (x y : ℕ) (hx : 10 ≤ x ∧ x ≤ 99) (hy : 100 ≤ y ∧ y ≤ 999) 
  (h : 1000 * x + y = 11 * x * y) : x + y = 110 :=
sorry

end NUMINAMATH_GPT_sarah_problem_l2237_223700


namespace NUMINAMATH_GPT_unique_identity_function_l2237_223742

theorem unique_identity_function (f : ℝ → ℝ) (H : ∀ x y z : ℝ, (x^3 + f y * x + f z = 0) → (f x ^ 3 + y * f x + z = 0)) :
  f = id :=
by sorry

end NUMINAMATH_GPT_unique_identity_function_l2237_223742


namespace NUMINAMATH_GPT_set_intersection_complement_l2237_223728

variable (U : Set ℝ := Set.univ)
variable (M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1)})
variable (N : Set ℝ := {x | 0 < x ∧ x < 2})

theorem set_intersection_complement :
  N ∩ (U \ M) = {x | 0 < x ∧ x ≤ 1} :=
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l2237_223728


namespace NUMINAMATH_GPT_man_speed_in_still_water_l2237_223750

theorem man_speed_in_still_water (V_m V_s : ℝ) 
  (h1 : V_m + V_s = 8)
  (h2 : V_m - V_s = 6) : 
  V_m = 7 := 
by
  sorry

end NUMINAMATH_GPT_man_speed_in_still_water_l2237_223750


namespace NUMINAMATH_GPT_john_spending_l2237_223743

variable (initial_cost : ℕ) (sale_price : ℕ) (new_card_cost : ℕ)

theorem john_spending (h1 : initial_cost = 1200) (h2 : sale_price = 300) (h3 : new_card_cost = 500) :
  initial_cost - sale_price + new_card_cost = 1400 := 
by
  sorry

end NUMINAMATH_GPT_john_spending_l2237_223743


namespace NUMINAMATH_GPT_find_fake_coin_in_two_weighings_l2237_223779

theorem find_fake_coin_in_two_weighings (coins : Fin 8 → ℝ) (h : ∃ i : Fin 8, (∀ j ≠ i, coins i < coins j)) : 
  ∃! i : Fin 8, ∀ j ≠ i, coins i < coins j :=
by
  sorry

end NUMINAMATH_GPT_find_fake_coin_in_two_weighings_l2237_223779


namespace NUMINAMATH_GPT_equivalent_fraction_l2237_223785

theorem equivalent_fraction (b : ℕ) (h : b = 2024) :
  (b^3 - 2 * b^2 * (b + 1) + 3 * b * (b + 1)^2 - (b + 1)^3 + 4) / (b * (b + 1)) = 2022 := by
  rw [h]
  sorry

end NUMINAMATH_GPT_equivalent_fraction_l2237_223785


namespace NUMINAMATH_GPT_spent_on_books_l2237_223758

theorem spent_on_books (allowance games_fraction snacks_fraction toys_fraction : ℝ)
  (h_allowance : allowance = 50)
  (h_games : games_fraction = 1/4)
  (h_snacks : snacks_fraction = 1/5)
  (h_toys : toys_fraction = 2/5) :
  allowance - (allowance * games_fraction + allowance * snacks_fraction + allowance * toys_fraction) = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_spent_on_books_l2237_223758


namespace NUMINAMATH_GPT_lesser_solution_of_quadratic_l2237_223766

theorem lesser_solution_of_quadratic :
  (∃ x y: ℝ, x ≠ y ∧ x^2 + 10*x - 24 = 0 ∧ y^2 + 10*y - 24 = 0 ∧ min x y = -12) :=
by {
  sorry
}

end NUMINAMATH_GPT_lesser_solution_of_quadratic_l2237_223766


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l2237_223775

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_triangle : A + B + C = 180) 
  (h_isosceles : A = B) (h_given_angle : A = 40) : C = 100 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l2237_223775


namespace NUMINAMATH_GPT_problem_I5_1_l2237_223721

theorem problem_I5_1 (a : ℝ) (h : a^2 - 8^2 = 12^2 + 9^2) : a = 17 := 
sorry

end NUMINAMATH_GPT_problem_I5_1_l2237_223721


namespace NUMINAMATH_GPT_stretched_curve_l2237_223771

noncomputable def transformed_curve (x : ℝ) : ℝ :=
  2 * Real.sin (x / 3 + Real.pi / 3)

theorem stretched_curve (y x : ℝ) :
  y = 2 * Real.sin (x + Real.pi / 3) → y = transformed_curve x := by
  intro h
  sorry

end NUMINAMATH_GPT_stretched_curve_l2237_223771


namespace NUMINAMATH_GPT_correct_formula_l2237_223782

theorem correct_formula {x y : ℕ} : 
  (x = 0 ∧ y = 100) ∨
  (x = 1 ∧ y = 90) ∨
  (x = 2 ∧ y = 70) ∨
  (x = 3 ∧ y = 40) ∨
  (x = 4 ∧ y = 0) →
  y = 100 - 5 * x - 5 * x^2 :=
by
  sorry

end NUMINAMATH_GPT_correct_formula_l2237_223782


namespace NUMINAMATH_GPT_area_ratio_equilateral_triangle_extension_l2237_223768

variable (s : ℝ)

theorem area_ratio_equilateral_triangle_extension :
  (let A := (0, 0)
   let B := (s, 0)
   let C := (s / 2, s * (Real.sqrt 3 / 2))
   let A' := (0, -4 * s * (Real.sqrt 3 / 2))
   let B' := (3 * s, 0)
   let C' := (s / 2, s * (Real.sqrt 3 / 2) + 3 * s * (Real.sqrt 3 / 2))
   let area_ABC := (Real.sqrt 3 / 4) * s^2
   let area_A'B'C' := (Real.sqrt 3 / 4) * 60 * s^2
   area_A'B'C' / area_ABC = 60) :=
sorry

end NUMINAMATH_GPT_area_ratio_equilateral_triangle_extension_l2237_223768


namespace NUMINAMATH_GPT_clock_angle_at_3_20_is_160_l2237_223737

noncomputable def clock_angle_3_20 : ℚ :=
  let hour_hand_at_3 : ℚ := 90
  let minute_hand_per_minute : ℚ := 6
  let hour_hand_per_minute : ℚ := 1 / 2
  let time_passed : ℚ := 20
  let angle_change_per_minute : ℚ := minute_hand_per_minute - hour_hand_per_minute
  let total_angle_change : ℚ := time_passed * angle_change_per_minute
  let final_angle : ℚ := hour_hand_at_3 + total_angle_change
  let smaller_angle : ℚ := if final_angle > 180 then 360 - final_angle else final_angle
  smaller_angle

theorem clock_angle_at_3_20_is_160 : clock_angle_3_20 = 160 :=
by
  sorry

end NUMINAMATH_GPT_clock_angle_at_3_20_is_160_l2237_223737


namespace NUMINAMATH_GPT_max_view_angle_dist_l2237_223790

theorem max_view_angle_dist (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : ∃ (x : ℝ), x = Real.sqrt (b * (a + b)) := by
  sorry

end NUMINAMATH_GPT_max_view_angle_dist_l2237_223790


namespace NUMINAMATH_GPT_union_sets_l2237_223753

-- Given sets A and B
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5}

theorem union_sets : A ∪ B = {1, 2, 3, 4, 5} := by
  sorry

end NUMINAMATH_GPT_union_sets_l2237_223753


namespace NUMINAMATH_GPT_spencer_walked_distance_l2237_223799

/-- Define the distances involved -/
def total_distance := 0.8
def library_to_post_office := 0.1
def post_office_to_home := 0.4

/-- Define the distance from house to library as a variable to calculate -/
def house_to_library := total_distance - library_to_post_office - post_office_to_home

/-- The theorem states that Spencer walked 0.3 miles from his house to the library -/
theorem spencer_walked_distance : 
  house_to_library = 0.3 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_spencer_walked_distance_l2237_223799
