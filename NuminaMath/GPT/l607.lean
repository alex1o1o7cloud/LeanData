import Mathlib

namespace NUMINAMATH_GPT_green_marbles_l607_60761

theorem green_marbles 
  (total_marbles : ℕ)
  (red_marbles : ℕ)
  (at_least_blue_marbles : ℕ)
  (h1 : total_marbles = 63) 
  (h2 : at_least_blue_marbles ≥ total_marbles / 3) 
  (h3 : red_marbles = 38) 
  : ∃ green_marbles : ℕ, total_marbles - red_marbles - at_least_blue_marbles = green_marbles ∧ green_marbles = 4 :=
by
  sorry

end NUMINAMATH_GPT_green_marbles_l607_60761


namespace NUMINAMATH_GPT_smallest_w_factor_l607_60748

theorem smallest_w_factor (w : ℕ) (hw : w > 0) :
  (∃ w, 2^4 ∣ 1452 * w ∧ 3^3 ∣ 1452 * w ∧ 13^3 ∣ 1452 * w) ↔ w = 79092 :=
by sorry

end NUMINAMATH_GPT_smallest_w_factor_l607_60748


namespace NUMINAMATH_GPT_base_7_units_digit_l607_60778

theorem base_7_units_digit : ((156 + 97) % 7) = 1 := 
by
  sorry

end NUMINAMATH_GPT_base_7_units_digit_l607_60778


namespace NUMINAMATH_GPT_increasing_function_inv_condition_l607_60784

-- Given a strictly increasing real-valued function f on ℝ with an inverse,
-- satisfying the condition f(x) + f⁻¹(x) = 2x for all x in ℝ,
-- prove that f(x) = x + b, where b is a real constant.

theorem increasing_function_inv_condition (f : ℝ → ℝ) (hf_strict_mono : StrictMono f)
  (hf_inv : ∀ x, f (f⁻¹ x) = x ∧ f⁻¹ (f x) = x)
  (hf_condition : ∀ x, f x + f⁻¹ x = 2 * x) :
  ∃ b : ℝ, ∀ x, f x = x + b :=
sorry

end NUMINAMATH_GPT_increasing_function_inv_condition_l607_60784


namespace NUMINAMATH_GPT_salaries_proof_l607_60768

-- Define salaries as real numbers
variables (a b c d : ℝ)

-- Define assumptions
def conditions := 
  (a + b + c + d = 4000) ∧
  (0.05 * a + 0.15 * b = c) ∧ 
  (0.25 * d = 0.3 * b) ∧
  (b = 3 * c)

-- Define the solution as found
def solution :=
  (a = 2365.55) ∧
  (b = 645.15) ∧
  (c = 215.05) ∧
  (d = 774.18)

-- Prove that given the conditions, the solution holds
theorem salaries_proof : 
  (conditions a b c d) → (solution a b c d) := by
  sorry

end NUMINAMATH_GPT_salaries_proof_l607_60768


namespace NUMINAMATH_GPT_area_between_curves_eq_nine_l607_60755

def f (x : ℝ) := 2 * x - x^2 + 3
def g (x : ℝ) := x^2 - 4 * x + 3

theorem area_between_curves_eq_nine :
  ∫ x in (0 : ℝ)..(3 : ℝ), (f x - g x) = 9 := by
  sorry

end NUMINAMATH_GPT_area_between_curves_eq_nine_l607_60755


namespace NUMINAMATH_GPT_min_value_a_squared_ab_b_squared_l607_60785

theorem min_value_a_squared_ab_b_squared {a b t p : ℝ} (h1 : a + b = t) (h2 : ab = p) :
  a^2 + ab + b^2 ≥ 3 * t^2 / 4 := by
  sorry

end NUMINAMATH_GPT_min_value_a_squared_ab_b_squared_l607_60785


namespace NUMINAMATH_GPT_line_slope_intercept_l607_60750

theorem line_slope_intercept :
  (∀ (x y : ℝ), 3 * (x + 2) - 4 * (y - 8) = 0 → y = (3/4) * x + 9.5) :=
sorry

end NUMINAMATH_GPT_line_slope_intercept_l607_60750


namespace NUMINAMATH_GPT_DianasInitialSpeed_l607_60786

open Nat

theorem DianasInitialSpeed
  (total_distance : ℕ)
  (initial_time : ℕ)
  (tired_speed : ℕ)
  (total_time : ℕ)
  (distance_when_tired : ℕ)
  (initial_distance : ℕ)
  (initial_speed : ℕ)
  (initial_hours : ℕ) :
  total_distance = 10 →
  initial_time = 2 →
  tired_speed = 1 →
  total_time = 6 →
  distance_when_tired = tired_speed * (total_time - initial_time) →
  initial_distance = total_distance - distance_when_tired →
  initial_distance = initial_speed * initial_time →
  initial_speed = 3 := by
  sorry

end NUMINAMATH_GPT_DianasInitialSpeed_l607_60786


namespace NUMINAMATH_GPT_polynomial_divisibility_l607_60797

theorem polynomial_divisibility (m : ℕ) (odd_m : m % 2 = 1) (x y z : ℤ) :
    ∃ k : ℤ, (x + y + z)^m - x^m - y^m - z^m = k * ((x + y + z)^3 - x^3 - y^3 - z^3) := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l607_60797


namespace NUMINAMATH_GPT_solve_equation_l607_60709

theorem solve_equation (x : ℚ) (h : x ≠ 1) : (x^2 - 2 * x + 3) / (x - 1) = x + 4 ↔ x = 7 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l607_60709


namespace NUMINAMATH_GPT_find_x_l607_60707

theorem find_x (x : ℝ) (h : 0.5 * x = 0.05 * 500 - 20) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l607_60707


namespace NUMINAMATH_GPT_sum_first_six_terms_l607_60728

variable (a1 q : ℤ)
variable (n : ℕ)

noncomputable def geometric_sum (a1 q : ℤ) (n : ℕ) : ℤ :=
  a1 * (1 - q^n) / (1 - q)

theorem sum_first_six_terms :
  geometric_sum (-1) 2 6 = 63 :=
sorry

end NUMINAMATH_GPT_sum_first_six_terms_l607_60728


namespace NUMINAMATH_GPT_find_a_b_c_eq_32_l607_60735

variables {a b c : ℤ}

theorem find_a_b_c_eq_32
  (h1 : ∃ a b : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b))
  (h2 : ∃ b c : ℤ, x^2 - 21 * x + 108 = (x - b) * (x - c)) :
  a + b + c = 32 :=
sorry

end NUMINAMATH_GPT_find_a_b_c_eq_32_l607_60735


namespace NUMINAMATH_GPT_total_length_of_intervals_l607_60721

theorem total_length_of_intervals :
  (∀ (x : ℝ), |x| < 1 → Real.tan (Real.log x / Real.log 5) < 0) →
  ∃ (length : ℝ), length = (2 * (5 ^ (Real.pi / 2))) / (1 + (5 ^ (Real.pi / 2))) :=
sorry

end NUMINAMATH_GPT_total_length_of_intervals_l607_60721


namespace NUMINAMATH_GPT_find_sets_A_B_l607_60738

def C : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

def S : Finset ℕ := {4, 5, 9, 14, 23, 37}

theorem find_sets_A_B :
  ∃ (A B : Finset ℕ), 
  (A ∩ B = ∅) ∧ 
  (A ∪ B = C) ∧ 
  (∀ (x y : ℕ), x ≠ y → x ∈ A → y ∈ A → x + y ∉ S) ∧ 
  (∀ (x y : ℕ), x ≠ y → x ∈ B → y ∈ B → x + y ∉ S) ∧ 
  (A = {1, 2, 5, 6, 10, 11, 14, 15, 16, 19, 20}) ∧ 
  (B = {3, 4, 7, 8, 9, 12, 13, 17, 18}) :=
by
  sorry

end NUMINAMATH_GPT_find_sets_A_B_l607_60738


namespace NUMINAMATH_GPT_ladder_slides_out_l607_60762

theorem ladder_slides_out (ladder_length foot_initial_dist ladder_slip_down foot_final_dist : ℝ) 
  (h_ladder_length : ladder_length = 25)
  (h_foot_initial_dist : foot_initial_dist = 7)
  (h_ladder_slip_down : ladder_slip_down = 4)
  (h_foot_final_dist : foot_final_dist = 15) :
  foot_final_dist - foot_initial_dist = 8 :=
  by
  simp [h_ladder_length, h_foot_initial_dist, h_ladder_slip_down, h_foot_final_dist]
  sorry

end NUMINAMATH_GPT_ladder_slides_out_l607_60762


namespace NUMINAMATH_GPT_sheep_to_cow_ratio_l607_60713

theorem sheep_to_cow_ratio : 
  ∀ (cows sheep : ℕ) (cow_water sheep_water : ℕ),
  cows = 40 →
  cow_water = 80 →
  sheep_water = cow_water / 4 →
  7 * (cows * cow_water + sheep * sheep_water) = 78400 →
  sheep / cows = 10 :=
by
  intros cows sheep cow_water sheep_water hcows hcow_water hsheep_water htotal
  sorry

end NUMINAMATH_GPT_sheep_to_cow_ratio_l607_60713


namespace NUMINAMATH_GPT_x_of_x35x_div_by_18_l607_60766

theorem x_of_x35x_div_by_18 (x : ℕ) (h₁ : 18 = 2 * 9) (h₂ : (2 * x + 8) % 9 = 0) (h₃ : ∃ k : ℕ, x = 2 * k) : x = 8 :=
sorry

end NUMINAMATH_GPT_x_of_x35x_div_by_18_l607_60766


namespace NUMINAMATH_GPT_no_solutions_xyz_l607_60765

theorem no_solutions_xyz : ∀ (x y z : ℝ), x + y = 3 → xy - z^2 = 2 → false := by
  intros x y z h1 h2
  sorry

end NUMINAMATH_GPT_no_solutions_xyz_l607_60765


namespace NUMINAMATH_GPT_determine_a_l607_60758

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B (a : ℕ) : Set ℕ := {1, 2, a}

-- The proof statement
theorem determine_a (a : ℕ) (h : A ⊆ B a) : a = 3 :=
by 
  sorry

end NUMINAMATH_GPT_determine_a_l607_60758


namespace NUMINAMATH_GPT_range_of_set_l607_60700

/-- Given a set of three numbers with the following properties:
    1) The mean of the numbers is 5,
    2) The median of the numbers is 5,
    3) The smallest number in the set is 2,
    we want to show that the range of the set is 6. -/
theorem range_of_set (a b c : ℕ) (hmean : (a + b + c) / 3 = 5)
  (hmedian : b = 5) (hmin : a = 2) : 
  (max a (max b c)) - (min a (min b c)) = 6 := 
by 
  sorry

end NUMINAMATH_GPT_range_of_set_l607_60700


namespace NUMINAMATH_GPT_pamela_skittles_correct_l607_60747

def pamela_initial_skittles := 50
def pamela_gives_skittles_to_karen := 7
def pamela_receives_skittles_from_kevin := 3
def pamela_shares_percentage := 20

def pamela_final_skittles : Nat :=
  let after_giving := pamela_initial_skittles - pamela_gives_skittles_to_karen
  let after_receiving := after_giving + pamela_receives_skittles_from_kevin
  let share_amount := (after_receiving * pamela_shares_percentage) / 100
  let rounded_share := Nat.floor share_amount
  let final_count := after_receiving - rounded_share
  final_count

theorem pamela_skittles_correct :
  pamela_final_skittles = 37 := by
  sorry

end NUMINAMATH_GPT_pamela_skittles_correct_l607_60747


namespace NUMINAMATH_GPT_cost_of_large_poster_is_correct_l607_60796

/-- Problem conditions -/
def posters_per_day : ℕ := 5
def large_posters_per_day : ℕ := 2
def large_poster_sale_price : ℝ := 10
def small_posters_per_day : ℕ := 3
def small_poster_sale_price : ℝ := 6
def small_poster_cost : ℝ := 3
def weekly_profit : ℝ := 95

/-- The cost to make a large poster -/
noncomputable def large_poster_cost : ℝ := 5

/-- Prove that the cost to make a large poster is $5 given the conditions -/
theorem cost_of_large_poster_is_correct :
    large_poster_cost = 5 :=
by
  -- (Condition translation into Lean)
  let daily_profit := weekly_profit / 5
  let daily_revenue := (large_posters_per_day * large_poster_sale_price) + (small_posters_per_day * small_poster_sale_price)
  let daily_cost_small_posters := small_posters_per_day * small_poster_cost
  
  -- Express the daily profit in terms of costs, including unknown large_poster_cost
  have calc_profit : daily_profit = daily_revenue - daily_cost_small_posters - (large_posters_per_day * (large_poster_cost)) :=
    sorry
  
  -- Setting the equation to solve for large_poster_cost
  have eqn : daily_profit = 19 := by
    sorry

  -- Solve for large_poster_cost
  have solve_large_poster_cost : 19 = daily_revenue - daily_cost_small_posters - (large_posters_per_day * 5) :=
    by sorry
  
  sorry

end NUMINAMATH_GPT_cost_of_large_poster_is_correct_l607_60796


namespace NUMINAMATH_GPT_range_of_a_value_of_a_l607_60772

-- Problem 1
theorem range_of_a (a : ℝ) :
  (∃ x, (2 < x ∧ x < 4) ∧ (a < x ∧ x < 3 * a)) ↔ (4 / 3 ≤ a ∧ a < 4) :=
sorry

-- Problem 2
theorem value_of_a (a : ℝ) :
  (∀ x, (2 < x ∧ x < 4) ∨ (a < x ∧ x < 3 * a) ↔ (2 < x ∧ x < 6)) ↔ (a = 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_value_of_a_l607_60772


namespace NUMINAMATH_GPT_total_cost_l607_60749

-- Define the conditions
def dozen := 12
def cost_of_dozen_cupcakes := 10
def cost_of_dozen_cookies := 8
def cost_of_dozen_brownies := 12

def num_dozen_cupcakes := 4
def num_dozen_cookies := 3
def num_dozen_brownies := 2

-- Define the total cost for each type of treat
def total_cost_cupcakes := num_dozen_cupcakes * cost_of_dozen_cupcakes
def total_cost_cookies := num_dozen_cookies * cost_of_dozen_cookies
def total_cost_brownies := num_dozen_brownies * cost_of_dozen_brownies

-- The theorem to prove the total cost
theorem total_cost : total_cost_cupcakes + total_cost_cookies + total_cost_brownies = 88 := by
  -- Here would go the proof, but it's omitted as per the instructions
  sorry

end NUMINAMATH_GPT_total_cost_l607_60749


namespace NUMINAMATH_GPT_solution_to_fraction_l607_60701

theorem solution_to_fraction (x : ℝ) (h_fraction : (x^2 - 4) / (x + 4) = 0) (h_denom : x ≠ -4) : x = 2 ∨ x = -2 :=
sorry

end NUMINAMATH_GPT_solution_to_fraction_l607_60701


namespace NUMINAMATH_GPT_solve_for_a_and_b_l607_60741
-- Import the necessary library

open Classical

variable (a b x : ℝ)

theorem solve_for_a_and_b (h1 : 0 ≤ x) (h2 : x < 1) (h3 : x + 2 * a ≥ 4) (h4 : (2 * x - b) / 3 < 1) : a + b = 1 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_a_and_b_l607_60741


namespace NUMINAMATH_GPT_ages_of_father_and_daughter_l607_60720

variable (F D : ℕ)

-- Conditions
def condition1 : Prop := F = 4 * D
def condition2 : Prop := F + 20 = 2 * (D + 20)

-- Main statement
theorem ages_of_father_and_daughter (h1 : condition1 F D) (h2 : condition2 F D) : D = 10 ∧ F = 40 := by
  sorry

end NUMINAMATH_GPT_ages_of_father_and_daughter_l607_60720


namespace NUMINAMATH_GPT_triangle_area_l607_60792

noncomputable def area_triangle (b c angle_C : ℝ) : ℝ :=
  (1 / 2) * b * c * Real.sin angle_C

theorem triangle_area :
  let b := 1
  let c := Real.sqrt 3
  let angle_C := 2 * Real.pi / 3
  area_triangle b c (Real.sin angle_C) = Real.sqrt 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l607_60792


namespace NUMINAMATH_GPT_coefficient_of_x_in_first_equation_is_one_l607_60787

theorem coefficient_of_x_in_first_equation_is_one
  (x y z : ℝ)
  (h1 : x - 5 * y + 3 * z = 22 / 6)
  (h2 : 4 * x + 8 * y - 11 * z = 7)
  (h3 : 5 * x - 6 * y + 2 * z = 12)
  (h4 : x + y + z = 10) :
  (1 : ℝ) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_coefficient_of_x_in_first_equation_is_one_l607_60787


namespace NUMINAMATH_GPT_max_value_of_fraction_l607_60702

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end NUMINAMATH_GPT_max_value_of_fraction_l607_60702


namespace NUMINAMATH_GPT_hyperbola_k_range_l607_60756

theorem hyperbola_k_range (k : ℝ) : ((k + 2) * (6 - 2 * k) > 0) ↔ (-2 < k ∧ k < 3) := 
sorry

end NUMINAMATH_GPT_hyperbola_k_range_l607_60756


namespace NUMINAMATH_GPT_mutually_exclusive_event_3_l607_60798

def is_odd (n : ℕ) := n % 2 = 1
def is_even (n : ℕ) := n % 2 = 0

def event_1 (a b : ℕ) := 
(is_odd a ∧ is_even b) ∨ (is_even a ∧ is_odd b)

def event_2 (a b : ℕ) := 
is_odd a ∧ is_odd b

def event_3 (a b : ℕ) := 
is_odd a ∧ is_even a ∧ is_odd b ∧ is_even b

def event_4 (a b : ℕ) :=
(is_odd a ∧ is_even b) ∨ (is_even a ∧ is_odd b)

theorem mutually_exclusive_event_3 :
  ∀ a b : ℕ, event_3 a b → ¬ event_1 a b ∧ ¬ event_2 a b ∧ ¬ event_4 a b := by
sorry

end NUMINAMATH_GPT_mutually_exclusive_event_3_l607_60798


namespace NUMINAMATH_GPT_smallest_possible_e_l607_60771

-- Definitions based on given conditions
def polynomial (x : ℝ) (a b c d e : ℝ) : ℝ := a * x^4 + b * x^3 + c * x^2 + d * x + e

-- The given polynomial has roots -3, 4, 8, and -1/4, and e is positive integer
theorem smallest_possible_e :
  ∃ (a b c d e : ℤ), polynomial x a b c d e = 4*x^4 - 32*x^3 - 23*x^2 + 104*x + 96 ∧ e > 0 ∧ e = 96 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_e_l607_60771


namespace NUMINAMATH_GPT_polar_eq_circle_l607_60776

-- Definition of the problem condition in polar coordinates
def polar_eq (ρ : ℝ) : Prop := ρ = 1

-- Definition of the assertion we want to prove: that it represents a circle
def represents_circle (ρ : ℝ) (θ : ℝ) : Prop := (ρ = 1) → ∃ (x y : ℝ), (ρ = 1) ∧ (x^2 + y^2 = 1)

theorem polar_eq_circle : ∀ (ρ θ : ℝ), polar_eq ρ → represents_circle ρ θ :=
by
  intros ρ θ hρ hs
  sorry

end NUMINAMATH_GPT_polar_eq_circle_l607_60776


namespace NUMINAMATH_GPT_largest_of_options_l607_60737

theorem largest_of_options :
  max (2 + 0 + 1 + 3) (max (2 * 0 + 1 + 3) (max (2 + 0 * 1 + 3) (max (2 + 0 + 1 * 3) (2 * 0 * 1 * 3)))) = 2 + 0 + 1 + 3 := by sorry

end NUMINAMATH_GPT_largest_of_options_l607_60737


namespace NUMINAMATH_GPT_compound_interest_calculation_l607_60795

noncomputable def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  let A := P * ((1 + r / (n : ℝ)) ^ (n * t))
  A - P

theorem compound_interest_calculation :
  compoundInterest 500 0.05 1 5 = 138.14 := by
  sorry

end NUMINAMATH_GPT_compound_interest_calculation_l607_60795


namespace NUMINAMATH_GPT_problem_statement_l607_60714

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) (α : ℝ) (β : ℝ) : ℝ := 
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
  (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f 1988 a b α β = 3) : f 2013 a b α β = 5 :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l607_60714


namespace NUMINAMATH_GPT_remainder_a_cubed_l607_60770

theorem remainder_a_cubed {a n : ℤ} (hn : 0 < n) (hinv : a * a ≡ 1 [ZMOD n]) (ha : a ≡ -1 [ZMOD n]) : a^3 ≡ -1 [ZMOD n] := 
sorry

end NUMINAMATH_GPT_remainder_a_cubed_l607_60770


namespace NUMINAMATH_GPT_prove_3a_3b_3c_l607_60767

variable (a b c : ℝ)

def condition1 := b + c = 15 - 2 * a
def condition2 := a + c = -18 - 3 * b
def condition3 := a + b = 8 - 4 * c
def condition4 := a - b + c = 3

theorem prove_3a_3b_3c (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b c) (h4 : condition4 a b c) :
  3 * a + 3 * b + 3 * c = 24 / 5 :=
sorry

end NUMINAMATH_GPT_prove_3a_3b_3c_l607_60767


namespace NUMINAMATH_GPT_pizza_problem_l607_60732

theorem pizza_problem (diameter : ℝ) (sectors : ℕ) (h1 : diameter = 18) (h2 : sectors = 4) : 
  let R := diameter / 2 
  let θ := (2 * Real.pi / sectors : ℝ)
  let m := 2 * R * Real.sin (θ / 2) 
  (m^2 = 162) := by
  sorry

end NUMINAMATH_GPT_pizza_problem_l607_60732


namespace NUMINAMATH_GPT_citrus_grove_total_orchards_l607_60722

theorem citrus_grove_total_orchards (lemons_orchards oranges_orchards grapefruits_orchards limes_orchards total_orchards : ℕ) 
  (h1 : lemons_orchards = 8) 
  (h2 : oranges_orchards = lemons_orchards / 2) 
  (h3 : grapefruits_orchards = 2) 
  (h4 : limes_orchards = grapefruits_orchards) 
  (h5 : total_orchards = lemons_orchards + oranges_orchards + grapefruits_orchards + limes_orchards) : 
  total_orchards = 16 :=
by 
  sorry

end NUMINAMATH_GPT_citrus_grove_total_orchards_l607_60722


namespace NUMINAMATH_GPT_at_least_one_bigger_than_44_9_l607_60791

noncomputable def x : ℕ → ℝ := sorry
noncomputable def y : ℕ → ℝ := sorry

axiom x_positive (n : ℕ) : 0 < x n
axiom y_positive (n : ℕ) : 0 < y n
axiom recurrence_x (n : ℕ) : x (n + 1) = x n + 1 / (2 * y n)
axiom recurrence_y (n : ℕ) : y (n + 1) = y n + 1 / (2 * x n)

theorem at_least_one_bigger_than_44_9 : x 2018 > 44.9 ∨ y 2018 > 44.9 :=
sorry

end NUMINAMATH_GPT_at_least_one_bigger_than_44_9_l607_60791


namespace NUMINAMATH_GPT_min_sum_of_segments_is_305_l607_60774

noncomputable def min_sum_of_segments : ℕ := 
  let a : ℕ := 3
  let b : ℕ := 5
  100 * a + b

theorem min_sum_of_segments_is_305 : min_sum_of_segments = 305 := by
  sorry

end NUMINAMATH_GPT_min_sum_of_segments_is_305_l607_60774


namespace NUMINAMATH_GPT_least_common_multiple_1008_672_l607_60727

theorem least_common_multiple_1008_672 : Nat.lcm 1008 672 = 2016 := by
  -- Add the prime factorizations and show the LCM calculation
  have h1 : 1008 = 2^4 * 3^2 * 7 := by sorry
  have h2 : 672 = 2^5 * 3 * 7 := by sorry
  -- Utilize the factorizations to compute LCM
  have calc1 : Nat.lcm (2^4 * 3^2 * 7) (2^5 * 3 * 7) = 2^5 * 3^2 * 7 := by sorry
  -- Show the calculation of 2^5 * 3^2 * 7
  have calc2 : 2^5 * 3^2 * 7 = 2016 := by sorry
  -- Therefore, LCM of 1008 and 672 is 2016
  exact calc2

end NUMINAMATH_GPT_least_common_multiple_1008_672_l607_60727


namespace NUMINAMATH_GPT_exists_polynomial_P_l607_60725

open Int Nat

/-- Define a predicate for a value is a perfect square --/
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- Define the polynomial P(x, y, z) --/
noncomputable def P (x y z : ℕ) : ℤ := 
  (1 - 2013 * (z - 1) * (z - 2)) * 
  ((x + y - 1) * (x + y - 1) + 2 * y - 2 + z)

/-- The main theorem to prove --/
theorem exists_polynomial_P :
  ∃ (P : ℕ → ℕ → ℕ → ℤ), 
  (∀ n : ℕ, (¬ is_square n) ↔ ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ P x y z = n) := 
sorry

end NUMINAMATH_GPT_exists_polynomial_P_l607_60725


namespace NUMINAMATH_GPT_a_2018_mod_49_l607_60751

def a (n : ℕ) : ℕ := 6^n + 8^n

theorem a_2018_mod_49 : (a 2018) % 49 = 0 := by
  sorry

end NUMINAMATH_GPT_a_2018_mod_49_l607_60751


namespace NUMINAMATH_GPT_binary_to_decimal_10101_l607_60734

theorem binary_to_decimal_10101 : (1 * 2^4 + 0 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 21 :=
by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_10101_l607_60734


namespace NUMINAMATH_GPT_number_of_mappings_n_elements_l607_60724

theorem number_of_mappings_n_elements
  (A : Type) [Fintype A] [DecidableEq A] (n : ℕ) (h : 3 ≤ n) (f : A → A)
  (H1 : ∀ x : A, ∃ c : A, ∀ (i : ℕ), i ≥ n - 2 → f^[i] x = c)
  (H2 : ∃ x₁ x₂ : A, f^[n] x₁ ≠ f^[n] x₂) :
  ∃ m : ℕ, m = (2 * n - 5) * (n.factorial) / 2 :=
sorry

end NUMINAMATH_GPT_number_of_mappings_n_elements_l607_60724


namespace NUMINAMATH_GPT_positive_n_for_one_solution_l607_60763

theorem positive_n_for_one_solution :
  ∀ (n : ℝ), (4 * (0 : ℝ)) ^ 2 + n * (0) + 16 = 0 → (n^2 - 256 = 0) → n = 16 :=
by
  intro n
  intro h
  intro discriminant_eq_zero
  sorry

end NUMINAMATH_GPT_positive_n_for_one_solution_l607_60763


namespace NUMINAMATH_GPT_boolean_logic_problem_l607_60719

theorem boolean_logic_problem (p q : Prop) (h₁ : ¬(p ∧ q)) (h₂ : ¬(¬p)) : ¬q :=
by {
  sorry
}

end NUMINAMATH_GPT_boolean_logic_problem_l607_60719


namespace NUMINAMATH_GPT_age_difference_l607_60782

theorem age_difference
  (A B : ℕ)
  (hB : B = 48)
  (h_condition : A + 10 = 2 * (B - 10)) :
  A - B = 18 :=
by
  sorry

end NUMINAMATH_GPT_age_difference_l607_60782


namespace NUMINAMATH_GPT_natural_number_with_property_l607_60710

theorem natural_number_with_property :
  ∃ n a b c : ℕ, (n = 10 * a + b) ∧ (100 * a + 10 * c + b = 6 * n) ∧ (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (0 ≤ c ∧ c ≤ 9) ∧ (n = 18) :=
sorry

end NUMINAMATH_GPT_natural_number_with_property_l607_60710


namespace NUMINAMATH_GPT_largest_multiple_of_8_less_than_100_l607_60775

theorem largest_multiple_of_8_less_than_100 :
  ∃ n : ℕ, 8 * n < 100 ∧ (∀ m : ℕ, 8 * m < 100 → m ≤ n) ∧ 8 * n = 96 :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_8_less_than_100_l607_60775


namespace NUMINAMATH_GPT_sum_b_a1_a2_a3_a4_eq_60_l607_60799

def a_n (n : ℕ) : ℕ := n + 2
def b_n (n : ℕ) : ℕ := 2^(n-1)

theorem sum_b_a1_a2_a3_a4_eq_60 :
  b_n (a_n 1) + b_n (a_n 2) + b_n (a_n 3) + b_n (a_n 4) = 60 :=
by
  sorry

end NUMINAMATH_GPT_sum_b_a1_a2_a3_a4_eq_60_l607_60799


namespace NUMINAMATH_GPT_scientific_notation_of_3300000_l607_60736

theorem scientific_notation_of_3300000 : 3300000 = 3.3 * 10^6 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_3300000_l607_60736


namespace NUMINAMATH_GPT_car_mpg_city_l607_60723

theorem car_mpg_city
  (h c T : ℝ)
  (h1 : h * T = 480)
  (h2 : c * T = 336)
  (h3 : c = h - 6) :
  c = 14 :=
by
  sorry

end NUMINAMATH_GPT_car_mpg_city_l607_60723


namespace NUMINAMATH_GPT_max_value_y_interval_l607_60764

noncomputable def y (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem max_value_y_interval : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → y x ≤ 2) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y x = 2) 
:=
by
  sorry

end NUMINAMATH_GPT_max_value_y_interval_l607_60764


namespace NUMINAMATH_GPT_range_of_x_l607_60752

theorem range_of_x (x m : ℝ) (h₁ : 1 ≤ m) (h₂ : m ≤ 3) (h₃ : x + 3 * m + 5 > 0) : x > -14 := 
sorry

end NUMINAMATH_GPT_range_of_x_l607_60752


namespace NUMINAMATH_GPT_find_second_offset_l607_60726

variable (d : ℕ) (o₁ : ℕ) (A : ℕ)

theorem find_second_offset (hd : d = 20) (ho₁ : o₁ = 5) (hA : A = 90) : ∃ (o₂ : ℕ), o₂ = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_second_offset_l607_60726


namespace NUMINAMATH_GPT_Tucker_last_number_l607_60745

-- Define the sequence of numbers said by Todd, Tadd, and Tucker
def game_sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 4
  else if n = 5 then 5
  else if n = 6 then 6
  else sorry -- Define recursively for subsequent rounds

-- Condition: The game ends when they reach the number 1000.
def game_end := 1000

-- Define the function to determine the last number said by Tucker
def last_number_said_by_Tucker (end_num : ℕ) : ℕ :=
  -- Assuming this function correctly calculates the last number said by Tucker
  if end_num = game_end then 1000 else sorry

-- Problem statement to prove
theorem Tucker_last_number : last_number_said_by_Tucker game_end = 1000 := by
  sorry

end NUMINAMATH_GPT_Tucker_last_number_l607_60745


namespace NUMINAMATH_GPT_b_earns_more_than_a_l607_60730

-- Definitions for the conditions
def investments_ratio := (3, 4, 5)
def returns_ratio := (6, 5, 4)
def total_earnings := 10150

-- We need to prove the statement
theorem b_earns_more_than_a (x y : ℕ) (hx : 58 * x * y = 10150) : 2 * x * y = 350 := by
  -- Conditions based on ratios
  let earnings_a := 3 * x * 6 * y
  let earnings_b := 4 * x * 5 * y
  let difference := earnings_b - earnings_a
  
  -- To complete the proof, sorry is used
  sorry

end NUMINAMATH_GPT_b_earns_more_than_a_l607_60730


namespace NUMINAMATH_GPT_min_value_expression_l607_60743

theorem min_value_expression (a b : ℝ) (h : a > b) (h0 : b > 0) :
  ∃ m : ℝ, m = (a^2 + 1 / (a * b) + 1 / (a * (a - b))) ∧ m = 4 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l607_60743


namespace NUMINAMATH_GPT_simplify_expression_l607_60704

theorem simplify_expression (x y : ℝ) (h : x - 3 * y = 4) : (x - 3 * y) ^ 2 + 2 * x - 6 * y - 10 = 14 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l607_60704


namespace NUMINAMATH_GPT_total_weight_l607_60780

def w1 : ℝ := 9.91
def w2 : ℝ := 4.11

theorem total_weight : w1 + w2 = 14.02 := by 
  sorry

end NUMINAMATH_GPT_total_weight_l607_60780


namespace NUMINAMATH_GPT_total_cost_is_eight_times_short_cost_l607_60740

variables (x : ℝ)
def cost_shorts := x
def cost_shirt := x
def cost_boots := 4 * x
def cost_shin_guards := 2 * x

theorem total_cost_is_eight_times_short_cost
    (cond_shirt : cost_shorts x + cost_shirt x = 2*x)
    (cond_boots : cost_shorts x + cost_boots x = 5*x)
    (cond_shin_guards : cost_shorts x + cost_shin_guards x = 3*x) :
    cost_shorts x + cost_shirt x + cost_boots x + cost_shin_guards x = 8*x :=
by
  sorry

end NUMINAMATH_GPT_total_cost_is_eight_times_short_cost_l607_60740


namespace NUMINAMATH_GPT_smallest_distance_proof_l607_60706

noncomputable def smallest_distance (z w : ℂ) : ℝ :=
  Complex.abs (z - w)

theorem smallest_distance_proof (z w : ℂ) 
  (h1 : Complex.abs (z - (2 - 4*Complex.I)) = 2)
  (h2 : Complex.abs (w - (-5 + 6*Complex.I)) = 4) :
  smallest_distance z w ≥ Real.sqrt 149 - 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_distance_proof_l607_60706


namespace NUMINAMATH_GPT_tortoise_age_l607_60716

-- Definitions based on the given problem conditions
variables (a b c : ℕ)

-- The conditions as provided in the problem
def condition1 (a b : ℕ) : Prop := a / 4 = 2 * a - b
def condition2 (b c : ℕ) : Prop := b / 7 = 2 * b - c
def condition3 (a b c : ℕ) : Prop := a + b + c = 264

-- The main theorem to prove
theorem tortoise_age (a b c : ℕ) (h1 : condition1 a b) (h2 : condition2 b c) (h3 : condition3 a b c) : b = 77 :=
sorry

end NUMINAMATH_GPT_tortoise_age_l607_60716


namespace NUMINAMATH_GPT_spherical_coordinates_equivalence_l607_60742

theorem spherical_coordinates_equivalence
  (ρ θ φ : ℝ)
  (h_ρ : ρ > 0)
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h_φ : φ = 2 * Real.pi - (7 * Real.pi / 4)) :
  (ρ, θ, φ) = (4, 3 * Real.pi / 4, Real.pi / 4) :=
by 
  sorry

end NUMINAMATH_GPT_spherical_coordinates_equivalence_l607_60742


namespace NUMINAMATH_GPT_first_discount_percentage_l607_60790

theorem first_discount_percentage 
  (original_price final_price : ℝ) 
  (successive_discount1 successive_discount2 : ℝ) 
  (h1 : original_price = 10000)
  (h2 : final_price = 6840)
  (h3 : successive_discount1 = 0.10)
  (h4 : successive_discount2 = 0.05)
  : ∃ x, (1 - x / 100) * (1 - successive_discount1) * (1 - successive_discount2) * original_price = final_price ∧ x = 20 :=
by
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l607_60790


namespace NUMINAMATH_GPT_multiply_fractions_l607_60733

theorem multiply_fractions :
  (2/3) * (4/7) * (9/11) * (5/8) = 15/77 :=
by
  -- It is just a statement, no need for the proof steps here
  sorry

end NUMINAMATH_GPT_multiply_fractions_l607_60733


namespace NUMINAMATH_GPT_savings_percentage_l607_60711

variable {I S : ℝ}
variable (h1 : 1.30 * I - 2 * S + I - S = 2 * (I - S))

theorem savings_percentage (h : 1.30 * I - 2 * S + I - S = 2 * (I - S)) : S = 0.30 * I :=
  by
    sorry

end NUMINAMATH_GPT_savings_percentage_l607_60711


namespace NUMINAMATH_GPT_farmer_field_area_l607_60773

theorem farmer_field_area (m : ℝ) (h : (3 * m + 5) * (m + 1) = 104) : m = 4.56 :=
sorry

end NUMINAMATH_GPT_farmer_field_area_l607_60773


namespace NUMINAMATH_GPT_fraction_after_adding_liters_l607_60703

-- Given conditions
variables (c w : ℕ)
variables (h1 : w = c / 3)
variables (h2 : (w + 5) / c = 2 / 5)

-- The proof statement
theorem fraction_after_adding_liters (h1 : w = c / 3) (h2 : (w + 5) / c = 2 / 5) : 
  (w + 9) / c = 34 / 75 :=
sorry -- Proof omitted

end NUMINAMATH_GPT_fraction_after_adding_liters_l607_60703


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l607_60717

/- Given a quadratic function with specific roots and coefficients, prove a quadratic inequality. -/
theorem quadratic_inequality_solution_set :
  ∀ (a b : ℝ),
    (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 + a*x + b < 0) →
    a = -3 →
    b = 2 →
    ∀ x : ℝ, (x < 1/2 ∨ x > 1) ↔ (2*x^2 - 3*x + 1 > 0) :=
by
  intros a b h cond_a cond_b x
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l607_60717


namespace NUMINAMATH_GPT_total_time_to_school_and_back_l607_60777

-- Definition of the conditions
def speed_to_school : ℝ := 3 -- in km/hr
def speed_back_home : ℝ := 2 -- in km/hr
def distance : ℝ := 6 -- in km

-- Proof statement
theorem total_time_to_school_and_back : 
  (distance / speed_to_school) + (distance / speed_back_home) = 5 := 
by
  sorry

end NUMINAMATH_GPT_total_time_to_school_and_back_l607_60777


namespace NUMINAMATH_GPT_find_value_l607_60781

variable (x y z : ℕ)

-- Condition: x / 4 = y / 3 = z / 2
def ratio_condition := x / 4 = y / 3 ∧ y / 3 = z / 2

-- Theorem: Given the ratio condition, prove that (x - y + 3z) / x = 7 / 4.
theorem find_value (h : ratio_condition x y z) : (x - y + 3 * z) / x = 7 / 4 := 
  by sorry

end NUMINAMATH_GPT_find_value_l607_60781


namespace NUMINAMATH_GPT_evaluate_expression_l607_60779

theorem evaluate_expression : (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_evaluate_expression_l607_60779


namespace NUMINAMATH_GPT_two_times_sum_of_squares_l607_60715

theorem two_times_sum_of_squares (P a b : ℤ) (h : P = a^2 + b^2) : 
  ∃ x y : ℤ, 2 * P = x^2 + y^2 := 
by 
  sorry

end NUMINAMATH_GPT_two_times_sum_of_squares_l607_60715


namespace NUMINAMATH_GPT_circle_symmetry_l607_60760

theorem circle_symmetry (a : ℝ) : 
  (∀ x y : ℝ, (x^2 + y^2 - a*x + 2*y + 1 = 0 ↔ x^2 + y^2 = 1) ↔ a = 2) :=
sorry

end NUMINAMATH_GPT_circle_symmetry_l607_60760


namespace NUMINAMATH_GPT_largest_n_for_factoring_l607_60746

theorem largest_n_for_factoring :
  ∃ (n : ℤ), 
    (∀ A B : ℤ, (5 * B + A = n ∧ A * B = 60) → (5 * B + A ≤ n)) ∧
    n = 301 :=
by sorry

end NUMINAMATH_GPT_largest_n_for_factoring_l607_60746


namespace NUMINAMATH_GPT_value_range_of_a_l607_60793

variable (A B : Set ℝ)

noncomputable def A_def : Set ℝ := { x | 2 * x^2 - 3 * x + 1 ≤ 0 }
noncomputable def B_def (a : ℝ) : Set ℝ := { x | x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0 }

theorem value_range_of_a (a : ℝ) (hA : A = A_def) (hB : B = B_def a) :
    (Bᶜ ∩ A = ∅) → (0 ≤ a ∧ a ≤ 0.5) := 
sorry

end NUMINAMATH_GPT_value_range_of_a_l607_60793


namespace NUMINAMATH_GPT_price_of_each_apple_l607_60744

theorem price_of_each_apple
  (bike_cost: ℝ) (repair_cost_percent: ℝ) (remaining_percentage: ℝ)
  (total_apples_sold: ℕ) (repair_cost: ℝ) (total_money_earned: ℝ)
  (price_per_apple: ℝ) :
  bike_cost = 80 →
  repair_cost_percent = 0.25 →
  remaining_percentage = 0.2 →
  total_apples_sold = 20 →
  repair_cost = repair_cost_percent * bike_cost →
  total_money_earned = repair_cost / (1 - remaining_percentage) →
  price_per_apple = total_money_earned / total_apples_sold →
  price_per_apple = 1.25 := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_price_of_each_apple_l607_60744


namespace NUMINAMATH_GPT_ferry_round_trip_time_increases_l607_60789

variable {S V a b : ℝ}

theorem ferry_round_trip_time_increases (h1 : V > 0) (h2 : a < b) (h3 : V > a) (h4 : V > b) :
  (S / (V + b) + S / (V - b)) > (S / (V + a) + S / (V - a)) :=
by sorry

end NUMINAMATH_GPT_ferry_round_trip_time_increases_l607_60789


namespace NUMINAMATH_GPT_range_of_a_l607_60731

-- Define the conditions and the problem
def neg_p (x : ℝ) : Prop := -3 < x ∧ x < 0
def neg_q (x : ℝ) (a : ℝ) : Prop := x > a
def p (x : ℝ) : Prop := x ≤ -3 ∨ x ≥ 0
def q (x : ℝ) (a : ℝ) : Prop := x ≤ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, neg_p x → ¬ p x) ∧
  (∀ x : ℝ, neg_q x a → ¬ q x a) ∧
  (∀ x : ℝ, q x a → p x) ∧
  (∃ x : ℝ, ¬ (q x a → p x)) →
  a ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l607_60731


namespace NUMINAMATH_GPT_excluded_numbers_range_l607_60788

theorem excluded_numbers_range (S S' E : ℕ) (h1 : S = 31 * 10) (h2 : S' = 28 * 8) (h3 : E = S - S') (h4 : E > 70) :
  ∀ (x y : ℕ), x + y = E → 1 ≤ x ∧ x ≤ 85 ∧ 1 ≤ y ∧ y ≤ 85 := by
  sorry

end NUMINAMATH_GPT_excluded_numbers_range_l607_60788


namespace NUMINAMATH_GPT_exp_decreasing_iff_a_in_interval_l607_60708

theorem exp_decreasing_iff_a_in_interval (a : ℝ) : 
  (∀ x y : ℝ, x < y → (2 - a)^x > (2 - a)^y) ↔ 1 < a ∧ a < 2 :=
by 
  sorry

end NUMINAMATH_GPT_exp_decreasing_iff_a_in_interval_l607_60708


namespace NUMINAMATH_GPT_find_dividend_l607_60759

def quotient : ℝ := -427.86
def divisor : ℝ := 52.7
def remainder : ℝ := -14.5
def dividend : ℝ := (quotient * divisor) + remainder

theorem find_dividend : dividend = -22571.122 := by
  sorry

end NUMINAMATH_GPT_find_dividend_l607_60759


namespace NUMINAMATH_GPT_julian_initial_owing_l607_60794

theorem julian_initial_owing (jenny_owing_initial: ℕ) (borrow: ℕ) (total_owing: ℕ):
    borrow = 8 → total_owing = 28 → jenny_owing_initial + borrow = total_owing → jenny_owing_initial = 20 :=
by intros;
   exact sorry

end NUMINAMATH_GPT_julian_initial_owing_l607_60794


namespace NUMINAMATH_GPT_max_leap_years_l607_60769

theorem max_leap_years (years : ℕ) (leap_interval : ℕ) (total_years : ℕ) :
  leap_interval = 5 ∧ total_years = 200 → (years = total_years / leap_interval) :=
by
  sorry

end NUMINAMATH_GPT_max_leap_years_l607_60769


namespace NUMINAMATH_GPT_distance_between_homes_l607_60753

theorem distance_between_homes (Maxwell_distance : ℝ) (Maxwell_speed : ℝ) (Brad_speed : ℝ) (midpoint : ℝ) 
    (h1 : Maxwell_speed = 2) 
    (h2 : Brad_speed = 4) 
    (h3 : Maxwell_distance = 12) 
    (h4 : midpoint = Maxwell_distance * 2 * (Brad_speed / Maxwell_speed) + Maxwell_distance) :
midpoint = 36 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_homes_l607_60753


namespace NUMINAMATH_GPT_union_of_S_and_T_l607_60705

-- Declare sets S and T
def S : Set ℕ := {3, 4, 5}
def T : Set ℕ := {4, 7, 8}

-- Statement about their union
theorem union_of_S_and_T : S ∪ T = {3, 4, 5, 7, 8} :=
sorry

end NUMINAMATH_GPT_union_of_S_and_T_l607_60705


namespace NUMINAMATH_GPT_parallel_lines_distance_sum_l607_60739

theorem parallel_lines_distance_sum (b c : ℝ) 
  (h1 : ∃ k : ℝ, 6 = 3 * k ∧ b = 4 * k) 
  (h2 : (abs ((c / 2) - 5) / (Real.sqrt (3^2 + 4^2))) = 3) : 
  b + c = 48 ∨ b + c = -12 := by
  sorry

end NUMINAMATH_GPT_parallel_lines_distance_sum_l607_60739


namespace NUMINAMATH_GPT_total_votes_l607_60754

theorem total_votes (V : ℝ) (h : 0.60 * V - 0.40 * V = 1200) : V = 6000 :=
sorry

end NUMINAMATH_GPT_total_votes_l607_60754


namespace NUMINAMATH_GPT_train_length_l607_60729

theorem train_length (L : ℕ) :
  (L + 350) / 15 = (L + 500) / 20 → L = 100 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_train_length_l607_60729


namespace NUMINAMATH_GPT_minimum_vehicles_l607_60783

theorem minimum_vehicles (students adults : ℕ) (van_capacity minibus_capacity : ℕ)
    (severe_allergies_students : ℕ) (vehicle_requires_adult : Prop)
    (h_students : students = 24) (h_adults : adults = 3)
    (h_van_capacity : van_capacity = 8) (h_minibus_capacity : minibus_capacity = 14)
    (h_severe_allergies_students : severe_allergies_students = 2)
    (h_vehicle_requires_adult : vehicle_requires_adult)
    : ∃ (min_vehicles : ℕ), min_vehicles = 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_vehicles_l607_60783


namespace NUMINAMATH_GPT_dan_bought_18_stickers_l607_60712

variable (S D : ℕ)

-- Given conditions
def stickers_initially_same : Prop := S = S -- Cindy and Dan have the same number of stickers initially
def cindy_used_15_stickers : Prop := true -- Cindy used 15 of her stickers
def dan_bought_D_stickers : Prop := true -- Dan bought D stickers
def dan_has_33_more_stickers_than_cindy : Prop := (S + D) = (S - 15 + 33)

-- Question: Prove that the number of stickers Dan bought is 18
theorem dan_bought_18_stickers (h1 : stickers_initially_same S)
                               (h2 : cindy_used_15_stickers)
                               (h3 : dan_bought_D_stickers)
                               (h4 : dan_has_33_more_stickers_than_cindy S D) : D = 18 :=
sorry

end NUMINAMATH_GPT_dan_bought_18_stickers_l607_60712


namespace NUMINAMATH_GPT_dons_profit_l607_60718

-- Definitions from the conditions
def bundles_jamie_bought := 20
def bundles_jamie_sold := 15
def profit_jamie := 60

def bundles_linda_bought := 34
def bundles_linda_sold := 24
def profit_linda := 69

def bundles_don_bought := 40
def bundles_don_sold := 36

-- Variables representing the unknown prices
variables (b s : ℝ)

-- Conditions written as equalities
axiom eq_jamie : bundles_jamie_sold * s - bundles_jamie_bought * b = profit_jamie
axiom eq_linda : bundles_linda_sold * s - bundles_linda_bought * b = profit_linda

-- Statement to prove Don's profit
theorem dons_profit : bundles_don_sold * s - bundles_don_bought * b = 252 :=
by {
  sorry -- proof goes here
}

end NUMINAMATH_GPT_dons_profit_l607_60718


namespace NUMINAMATH_GPT_john_total_cost_l607_60757

-- Definitions based on given conditions
def yearly_cost_first_8_years : ℕ := 10000
def yearly_cost_next_10_years : ℕ := 20000
def university_tuition : ℕ := 250000
def years_first_phase : ℕ := 8
def years_second_phase : ℕ := 10

-- We need to prove the total cost John pays
theorem john_total_cost : 
  (years_first_phase * yearly_cost_first_8_years + years_second_phase * yearly_cost_next_10_years + university_tuition) / 2 = 265000 :=
by sorry

end NUMINAMATH_GPT_john_total_cost_l607_60757
