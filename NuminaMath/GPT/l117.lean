import Mathlib

namespace sequence_term_2023_l117_117996

theorem sequence_term_2023 (a : ℕ → ℚ) (h₁ : a 1 = 2) 
  (h₂ : ∀ n, 1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1) : 
  a 2023 = -1 / 2 := 
sorry

end sequence_term_2023_l117_117996


namespace part_one_part_two_part_three_l117_117658

open Nat

def number_boys := 5
def number_girls := 4
def total_people := 9
def A_included := 1
def B_included := 1

theorem part_one : (number_boys.choose 2 * number_girls.choose 2) = 60 := sorry

theorem part_two : (total_people.choose 4 - (total_people - A_included - B_included).choose 4) = 91 := sorry

theorem part_three : (total_people.choose 4 - number_boys.choose 4 - number_girls.choose 4) = 120 := sorry

end part_one_part_two_part_three_l117_117658


namespace plane_intersects_unit_cubes_l117_117496

def unitCubeCount (side_length : ℕ) : ℕ :=
  side_length ^ 3

def intersectionCount (num_unitCubes : ℕ) (side_length : ℕ) : ℕ :=
  if side_length = 4 then 32 else 0 -- intersection count only applies for side_length = 4

theorem plane_intersects_unit_cubes
  (side_length : ℕ)
  (num_unitCubes : ℕ)
  (cubeArrangement : num_unitCubes = unitCubeCount side_length)
  (planeCondition : True) -- the plane is perpendicular to the diagonal and bisects it
  : intersectionCount num_unitCubes side_length = 32 := by
  sorry

end plane_intersects_unit_cubes_l117_117496


namespace time_saved_by_taking_route_B_l117_117438

-- Defining the times for the routes A and B
def time_route_A_one_way : ℕ := 5
def time_route_B_one_way : ℕ := 2

-- The total round trip times
def time_route_A_round_trip : ℕ := 2 * time_route_A_one_way
def time_route_B_round_trip : ℕ := 2 * time_route_B_one_way

-- The statement to prove
theorem time_saved_by_taking_route_B :
  time_route_A_round_trip - time_route_B_round_trip = 6 :=
by
  -- Proof would go here
  sorry

end time_saved_by_taking_route_B_l117_117438


namespace warmup_puzzle_time_l117_117914

theorem warmup_puzzle_time (W : ℕ) (H : W + 3 * W + 3 * W = 70) : W = 10 :=
by
  sorry

end warmup_puzzle_time_l117_117914


namespace matrix_power_is_correct_l117_117959

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]
def A_cubed : Matrix (Fin 2) (Fin 2) ℤ := !![3, -6; 6, -3]

theorem matrix_power_is_correct : A ^ 3 = A_cubed := by 
  sorry

end matrix_power_is_correct_l117_117959


namespace compute_difference_l117_117554

def bin_op (x y : ℤ) : ℤ := x^2 * y - 3 * x

theorem compute_difference :
  (bin_op 5 3) - (bin_op 3 5) = 24 := by
  sorry

end compute_difference_l117_117554


namespace Tim_total_expenditure_l117_117140

theorem Tim_total_expenditure 
  (appetizer_price : ℝ) (main_course_price : ℝ) (dessert_price : ℝ)
  (appetizer_tip_percentage : ℝ) (main_course_tip_percentage : ℝ) (dessert_tip_percentage : ℝ) :
  appetizer_price = 12.35 →
  main_course_price = 27.50 →
  dessert_price = 9.95 →
  appetizer_tip_percentage = 0.18 →
  main_course_tip_percentage = 0.20 →
  dessert_tip_percentage = 0.15 →
  appetizer_price * (1 + appetizer_tip_percentage) + 
  main_course_price * (1 + main_course_tip_percentage) + 
  dessert_price * (1 + dessert_tip_percentage) = 12.35 * 1.18 + 27.50 * 1.20 + 9.95 * 1.15 :=
  by sorry

end Tim_total_expenditure_l117_117140


namespace socks_impossible_l117_117757

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l117_117757


namespace number_of_mixed_pairs_l117_117971

/-- Given 2n men and 2n women, the number of ways to form n mixed pairs is (2n!)^2 / (n! * 2^n) -/
theorem number_of_mixed_pairs (n : ℕ) : 
  let men_and_women_pairs := (2 * n)!
  let mixed_pairs_count := men_and_women_pairs * men_and_women_pairs / (n! * 2^n)
  mixed_pairs_count = (2 * n)! * (2 * n)! / (n! * 2 ^ n) :=
by
  sorry

end number_of_mixed_pairs_l117_117971


namespace smallest_rel_prime_greater_than_one_l117_117920

theorem smallest_rel_prime_greater_than_one (n : ℕ) (h : n > 1) (h0: ∀ (m : ℕ), m > 1 ∧ Nat.gcd m 2100 = 1 → 11 ≤ m):
  Nat.gcd n 2100 = 1 → n = 11 :=
by
  -- Proof skipped
  sorry

end smallest_rel_prime_greater_than_one_l117_117920


namespace min_value_l117_117081

theorem min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : 
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (1 / x + 4 / y) ≥ 9) :=
by
  sorry

end min_value_l117_117081


namespace proportion_correct_l117_117353

theorem proportion_correct (x y : ℝ) (h1 : 2 * y = 5 * x) (h2 : x ≠ 0 ∧ y ≠ 0) : x / y = 2 / 5 := 
sorry

end proportion_correct_l117_117353


namespace speed_in_still_water_l117_117630

variable (v_m v_s : ℝ)

def swims_downstream (v_m v_s : ℝ) : Prop :=
  54 = (v_m + v_s) * 3

def swims_upstream (v_m v_s : ℝ) : Prop :=
  18 = (v_m - v_s) * 3

theorem speed_in_still_water : swims_downstream v_m v_s ∧ swims_upstream v_m v_s → v_m = 12 :=
by
  sorry

end speed_in_still_water_l117_117630


namespace partA_l117_117931

theorem partA (a b : ℝ) : (a - b) ^ 2 ≥ 0 → (a^2 + b^2) / 2 ≥ a * b := 
by
  intro h
  sorry

end partA_l117_117931


namespace notebooks_bought_l117_117965

def dan_total_spent : ℕ := 32
def backpack_cost : ℕ := 15
def pens_cost : ℕ := 1
def pencils_cost : ℕ := 1
def notebook_cost : ℕ := 3

theorem notebooks_bought :
  ∃ x : ℕ, dan_total_spent - (backpack_cost + pens_cost + pencils_cost) = x * notebook_cost ∧ x = 5 := 
by
  sorry

end notebooks_bought_l117_117965


namespace min_value_f_in_interval_l117_117904

def f (x : ℝ) : ℝ := x^4 + 2 * x^2 - 1

theorem min_value_f_in_interval : 
  ∃ x ∈ (Set.Icc (-1 : ℝ) 1), f x = -1 :=
by
  sorry


end min_value_f_in_interval_l117_117904


namespace ball_problem_l117_117430

theorem ball_problem (a : ℕ) (h1 : 3 / a = 0.25) : a = 12 :=
by sorry

end ball_problem_l117_117430


namespace convert_deg_to_min_compare_negatives_l117_117786

theorem convert_deg_to_min : (0.3 : ℝ) * 60 = 18 :=
by sorry

theorem compare_negatives : -2 > -3 :=
by sorry

end convert_deg_to_min_compare_negatives_l117_117786


namespace tree_height_at_2_years_l117_117508

theorem tree_height_at_2_years (h₅ : ℕ) (h_four : ℕ) (h_three : ℕ) (h_two : ℕ) (h₅_value : h₅ = 243)
  (h_four_value : h_four = h₅ / 3) (h_three_value : h_three = h_four / 3) (h_two_value : h_two = h_three / 3) :
  h_two = 9 := by
  sorry

end tree_height_at_2_years_l117_117508


namespace popped_white_probability_l117_117932

theorem popped_white_probability :
  let P_white := 2 / 3
  let P_yellow := 1 / 3
  let P_pop_given_white := 1 / 2
  let P_pop_given_yellow := 2 / 3

  let P_white_and_pop := P_white * P_pop_given_white
  let P_yellow_and_pop := P_yellow * P_pop_given_yellow
  let P_pop := P_white_and_pop + P_yellow_and_pop

  let P_white_given_pop := P_white_and_pop / P_pop

  P_white_given_pop = 3 / 5 := sorry

end popped_white_probability_l117_117932


namespace Inequality_Solution_Set_Range_of_c_l117_117674

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x

noncomputable def g (x : ℝ) : ℝ := -(((-x)^2) + 2 * (-x))

theorem Inequality_Solution_Set (x : ℝ) :
  (g x ≥ f x - |x - 1|) ↔ (-1 ≤ x ∧ x ≤ 1/2) :=
by
  sorry

theorem Range_of_c (c : ℝ) :
  (∀ x : ℝ, g x + c ≤ f x - |x - 1|) ↔ (c ≤ -9/8) :=
by
  sorry

end Inequality_Solution_Set_Range_of_c_l117_117674


namespace triangle_ABCD_lengths_l117_117844

theorem triangle_ABCD_lengths (AB BC CA : ℝ) (h_AB : AB = 20) (h_BC : BC = 40) (h_CA : CA = 49) :
  ∃ DA DC : ℝ, DA = 27.88 ∧ DC = 47.88 ∧
  (AB + DC = BC + DA) ∧ 
  (((AB^2 + BC^2 - CA^2) / (2 * AB * BC)) + ((DC^2 + DA^2 - CA^2) / (2 * DC * DA)) = 0) :=
sorry

end triangle_ABCD_lengths_l117_117844


namespace total_preparation_time_l117_117556

theorem total_preparation_time
    (minutes_per_game : ℕ)
    (number_of_games : ℕ)
    (h1 : minutes_per_game = 10)
    (h2 : number_of_games = 15) :
    minutes_per_game * number_of_games = 150 :=
by
  -- Lean 4 proof goes here
  sorry

end total_preparation_time_l117_117556


namespace problems_per_page_l117_117706

theorem problems_per_page (pages_math pages_reading total_problems x : ℕ) (h1 : pages_math = 2) (h2 : pages_reading = 4) (h3 : total_problems = 30) : 
  (pages_math + pages_reading) * x = total_problems → x = 5 := by
  sorry

end problems_per_page_l117_117706


namespace valid_P_values_l117_117232

/-- 
Construct a 3x3 grid of distinct natural numbers where the product of the numbers 
in each row and each column is equal. Verify the valid values of P among the given set.
-/
theorem valid_P_values (P : ℕ) :
  (∃ (a b c d e f g h i : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ 
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ 
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ 
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ 
    g ≠ h ∧ g ≠ i ∧ 
    h ≠ i ∧ 
    a * b * c = P ∧ 
    d * e * f = P ∧ 
    g * h * i = P ∧ 
    a * d * g = P ∧ 
    b * e * h = P ∧ 
    c * f * i = P ∧ 
    P = (Nat.sqrt ((1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9)) )) ↔ P = 1998 ∨ P = 2000 :=
sorry

end valid_P_values_l117_117232


namespace polar_line_through_centers_l117_117702

-- Definition of the given circles in polar coordinates
def Circle1 (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ
def Circle2 (ρ θ : ℝ) : Prop := ρ = 2 * Real.sin θ

-- Statement of the problem
theorem polar_line_through_centers (ρ θ : ℝ) :
  (∃ c1 c2 : ℝ × ℝ, Circle1 c1.fst c1.snd ∧ Circle2 c2.fst c2.snd ∧ θ = Real.pi / 4) :=
sorry

end polar_line_through_centers_l117_117702


namespace jamie_school_distance_l117_117868

theorem jamie_school_distance
  (v : ℝ) -- usual speed in miles per hour
  (d : ℝ) -- distance to school in miles
  (h1 : (20 : ℝ) / 60 = 1 / 3) -- usual time to school in hours
  (h2 : (10 : ℝ) / 60 = 1 / 6) -- lighter traffic time in hours
  (h3 : d = v * (1 / 3)) -- distance equation for usual traffic
  (h4 : d = (v + 15) * (1 / 6)) -- distance equation for lighter traffic
  : d = 5 := by
  sorry

end jamie_school_distance_l117_117868


namespace ratio_of_sweater_vests_to_shirts_l117_117947

theorem ratio_of_sweater_vests_to_shirts (S V O : ℕ) (h1 : S = 3) (h2 : O = 18) (h3 : O = V * S) : (V : ℚ) / (S : ℚ) = 2 := 
  by
  sorry

end ratio_of_sweater_vests_to_shirts_l117_117947


namespace first_digit_of_base16_representation_l117_117600

-- Firstly we define the base conversion from base 4 to base 10 and from base 10 to base 16.
-- For simplicity, we assume that the required functions exist and skip their implementations.

-- Assume base 4 to base 10 conversion function
def base4_to_base10 (n : String) : Nat :=
  sorry

-- Assume base 10 to base 16 conversion function that gives the first digit
def first_digit_base16 (n : Nat) : Nat :=
  sorry

-- Given the base 4 number as string
def y_base4 : String := "20313320132220312031"

-- Define the final statement
theorem first_digit_of_base16_representation :
  first_digit_base16 (base4_to_base10 y_base4) = 5 :=
by
  sorry

end first_digit_of_base16_representation_l117_117600


namespace remainder_698_div_D_l117_117923

-- Defining the conditions
variables (D k1 k2 k3 R : ℤ)

-- Given conditions
axiom condition1 : 242 = k1 * D + 4
axiom condition2 : 940 = k3 * D + 7
axiom condition3 : 698 = k2 * D + R

-- The theorem to prove the remainder 
theorem remainder_698_div_D : R = 3 :=
by
  -- Here you would provide the logical deduction steps
  sorry

end remainder_698_div_D_l117_117923


namespace wand_cost_l117_117927

-- Conditions based on the problem
def initialWands := 3
def salePrice (x : ℝ) := x + 5
def totalCollected := 130
def soldWands := 2

-- Proof statement
theorem wand_cost (x : ℝ) : 
  2 * salePrice x = totalCollected → x = 60 := 
by 
  sorry

end wand_cost_l117_117927


namespace maximize_profit_l117_117491

def total_orders := 100
def max_days := 160
def time_per_A := 5 / 4 -- days
def time_per_B := 5 / 3 -- days
def profit_per_A := 0.5 -- (10,000 RMB)
def profit_per_B := 0.8 -- (10,000 RMB)

theorem maximize_profit : 
  ∃ (x : ℝ) (y : ℝ), 
    (time_per_A * x + time_per_B * (total_orders - x) ≤ max_days) ∧ 
    (y = -0.3 * x + 80) ∧ 
    (x = 16) ∧ 
    (y = 75.2) :=
by 
  sorry

end maximize_profit_l117_117491


namespace tree_height_at_2_years_l117_117506

theorem tree_height_at_2_years (h : ℕ → ℚ) (h5 : h 5 = 243) 
  (h_rec : ∀ n, h (n - 1) = h n / 3) : h 2 = 9 :=
  sorry

end tree_height_at_2_years_l117_117506


namespace relationship_among_a_b_c_l117_117973

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 5)
noncomputable def b : ℝ := Real.cos (5 * Real.pi / 6)
noncomputable def c : ℝ := Real.tan (7 * Real.pi / 5)

theorem relationship_among_a_b_c : c > a ∧ a > b := by
  sorry

end relationship_among_a_b_c_l117_117973


namespace find_f1_find_fx_find_largest_m_l117_117677

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x ^ 2 + b * x + c

axiom min_value_eq_zero (a b c : ℝ) : ∀ x : ℝ, f a b c x ≥ 0 ∨ f a b c x ≤ 0
axiom symmetry_condition (a b c : ℝ) : ∀ x : ℝ, f a b c (x - 1) = f a b c (-x - 1)
axiom inequality_condition (a b c : ℝ) : ∀ x : ℝ, 0 < x ∧ x < 5 → x ≤ f a b c x ∧ f a b c x ≤ 2 * |x - 1| + 1

theorem find_f1 (a b c : ℝ) : f a b c 1 = 1 := sorry

theorem find_fx (a b c : ℝ) : ∀ x : ℝ, f a b c x = (1 / 4) * (x + 1) ^ 2 := sorry

theorem find_largest_m (a b c : ℝ) : ∃ m : ℝ, m > 1 ∧ ∀ t x : ℝ, 1 ≤ x ∧ x ≤ m → f a b c (x + t) ≤ x := sorry

end find_f1_find_fx_find_largest_m_l117_117677


namespace original_number_l117_117163

theorem original_number (x : ℝ) (hx : 1000 * x = 9 * (1 / x)) : 
  x = 3 * (Real.sqrt 10) / 100 :=
by
  sorry

end original_number_l117_117163


namespace win_game_A_win_game_C_l117_117490

-- Define the probabilities for heads and tails
def prob_heads : ℚ := 3 / 4
def prob_tails : ℚ := 1 / 4

-- Define the probability of winning Game A
def prob_win_game_A : ℚ := (prob_heads ^ 3) + (prob_tails ^ 3)

-- Define the probability of winning Game C
def prob_win_game_C : ℚ := (prob_heads ^ 4) + (prob_tails ^ 4)

-- State the theorem for Game A
theorem win_game_A : prob_win_game_A = 7 / 16 :=
by 
  -- Lean will check this proof
  sorry

-- State the theorem for Game C
theorem win_game_C : prob_win_game_C = 41 / 128 :=
by 
  -- Lean will check this proof
  sorry

end win_game_A_win_game_C_l117_117490


namespace sin_minus_cos_l117_117368

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l117_117368


namespace min_value_of_c_l117_117416

noncomputable def isPerfectSquare (x : ℕ) : Prop :=
  ∃ m : ℕ, x = m^2

noncomputable def isPerfectCube (x : ℕ) : Prop :=
  ∃ n : ℕ, x = n^3

theorem min_value_of_c (c : ℕ) :
  (∃ a b d e : ℕ, a = c-2 ∧ b = c-1 ∧ d = c+1 ∧ e = c+2 ∧ a < b ∧ b < c ∧ c < d ∧ d < e) ∧
  isPerfectSquare (3 * c) ∧
  isPerfectCube (5 * c) →
  c = 675 :=
sorry

end min_value_of_c_l117_117416


namespace find_reggie_long_shots_l117_117858

-- Define the constants used in the problem
def layup_points : ℕ := 1
def free_throw_points : ℕ := 2
def long_shot_points : ℕ := 3

-- Define Reggie's shooting results
def reggie_layups : ℕ := 3
def reggie_free_throws : ℕ := 2
def reggie_long_shots : ℕ := sorry -- we need to find this

-- Define Reggie's brother's shooting results
def brother_long_shots : ℕ := 4

-- Given conditions
def reggie_total_points := reggie_layups * layup_points + reggie_free_throws * free_throw_points + reggie_long_shots * long_shot_points
def brother_total_points := brother_long_shots * long_shot_points

def reggie_lost_by_2_points := reggie_total_points + 2 = brother_total_points

-- The theorem we need to prove
theorem find_reggie_long_shots : reggie_long_shots = 1 :=
by
  sorry

end find_reggie_long_shots_l117_117858


namespace smallest_integer_conditions_l117_117619

-- Definition of a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Definition of having a prime factor less than a given number
def has_prime_factor_less_than (n k : ℕ) : Prop :=
  ∃ p : ℕ, is_prime p ∧ p ∣ n ∧ p < k

-- Problem statement
theorem smallest_integer_conditions :
  ∃ n : ℕ, n > 0 ∧ ¬ is_prime n ∧ ¬ is_square n ∧ ¬ has_prime_factor_less_than n 60 ∧ ∀ m : ℕ, (m > 0 ∧ ¬ is_prime m ∧ ¬ is_square m ∧ ¬ has_prime_factor_less_than m 60) → n ≤ m :=
  sorry

end smallest_integer_conditions_l117_117619


namespace polynomial_digit_sum_infinite_identical_l117_117782

noncomputable def P (a : ℕ → ℤ) (n : ℕ) (x : ℕ) : ℤ :=
  ∑ i in finset.range (n+1), a i * x^(n - i)

noncomputable def s (P_val : ℤ) : ℕ :=
  (P_val.natAbs.digits 10).sum

theorem polynomial_digit_sum_infinite_identical {a : ℕ → ℤ} {n : ℕ} :
  (∃^∞ k, s (P a n k).natAbs) → (∃^∞ k, ∃ m, s (P a n k).natAbs = m) :=
sorry

end polynomial_digit_sum_infinite_identical_l117_117782


namespace functional_equation_odd_l117_117277

   variable {R : Type*} [AddCommGroup R] [Module ℝ R]

   def isOdd (f : ℝ → ℝ) : Prop :=
     ∀ x : ℝ, f (-x) = -f x

   theorem functional_equation_odd (f : ℝ → ℝ)
       (h_fun : ∀ x y : ℝ, f (x + y) = f x + f y) : isOdd f :=
   by
     sorry
   
end functional_equation_odd_l117_117277


namespace trig_identity_l117_117360

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l117_117360


namespace domain_of_function_l117_117739

theorem domain_of_function (x : ℝ) :
  (2 - x ≥ 0) ∧ (x - 1 > 0) ↔ (1 < x ∧ x ≤ 2) :=
by
  sorry

end domain_of_function_l117_117739


namespace discount_difference_l117_117338

-- Definitions based on given conditions
def original_bill : ℝ := 8000
def single_discount_rate : ℝ := 0.30
def first_successive_discount_rate : ℝ := 0.26
def second_successive_discount_rate : ℝ := 0.05

-- Calculations based on conditions
def single_discount_final_amount := original_bill * (1 - single_discount_rate)
def first_successive_discount_final_amount := original_bill * (1 - first_successive_discount_rate)
def complete_successive_discount_final_amount := 
  first_successive_discount_final_amount * (1 - second_successive_discount_rate)

-- Proof statement
theorem discount_difference :
  single_discount_final_amount - complete_successive_discount_final_amount = 24 := 
  by
    -- Proof to be provided
    sorry

end discount_difference_l117_117338


namespace susie_rhode_island_reds_l117_117734

variable (R G B_R B_G : ℕ)

def susie_golden_comets := G = 6
def britney_rir := B_R = 2 * R
def britney_golden_comets := B_G = G / 2
def britney_more_chickens := B_R + B_G = R + G + 8

theorem susie_rhode_island_reds
  (h1 : susie_golden_comets G)
  (h2 : britney_rir R B_R)
  (h3 : britney_golden_comets G B_G)
  (h4 : britney_more_chickens R G B_R B_G) :
  R = 11 :=
by
  sorry

end susie_rhode_island_reds_l117_117734


namespace pure_imaginary_value_l117_117274

theorem pure_imaginary_value (a : ℝ) : (z = (0 : ℝ) + (a^2 + 2 * a - 3) * I) → (a = 0 ∨ a = -2) :=
by
  sorry

end pure_imaginary_value_l117_117274


namespace geometric_sequence_first_term_l117_117603

open Nat

theorem geometric_sequence_first_term : 
  ∃ (a r : ℝ), (a * r^3 = (6 : ℝ)!) ∧ (a * r^6 = (7 : ℝ)!) ∧ a = 720 / 7 :=
by
  sorry

end geometric_sequence_first_term_l117_117603


namespace max_min_values_l117_117218

noncomputable def max_value (x y z w : ℝ) : ℝ :=
  if x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2 then
    max (x + y + z + w) 3
  else
    0

noncomputable def min_value (x y z w : ℝ) : ℝ :=
  if x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2 then
    min (x + y + z + w) (-2 + 5 / 2 * Real.sqrt 2)
  else
    0

theorem max_min_values (x y z w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) (h_nonneg_w : 0 ≤ w)
  (h_eqn : x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2) :
  (x + y + z + w ≤ 3) ∧ (x + y + z + w ≥ -2 + 5 / 2 * Real.sqrt 2) :=
by
  sorry

end max_min_values_l117_117218


namespace cos_sq_alpha_cos_sq_beta_range_l117_117555

theorem cos_sq_alpha_cos_sq_beta_range
  (α β : ℝ)
  (h : 3 * (Real.sin α)^2 + 2 * (Real.sin β)^2 - 2 * Real.sin α = 0) :
  (Real.cos α)^2 + (Real.cos β)^2 ∈ Set.Icc (14 / 9) 2 :=
sorry

end cos_sq_alpha_cos_sq_beta_range_l117_117555


namespace cuboid_surface_area_cuboid_volume_not_unique_l117_117909

theorem cuboid_surface_area
    (a b c p q : ℝ)
    (h1 : a + b + c = p)
    (h2 : a^2 + b^2 + c^2 = q^2) :
    2 * (a * b + b * c + a * c) = p^2 - q^2 :=
by
  sorry

theorem cuboid_volume_not_unique
    (a b c p q v1 v2 : ℝ)
    (h1 : a + b + c = p)
    (h2 : a^2 + b^2 + c^2 = q^2)
    : ¬ (∀ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ), 
          a₁ + b₁ + c₁ = p ∧ a₁^2 + b₁^2 + c₁^2 = q^2 →
          a₂ + b₂ + c₂ = p ∧ a₂^2 + b₂^2 + c₂^2 = q^2 →
          (a₁ * b₁ * c₁ = a₂ * b₂ * c₂)) :=
by
  -- Provide counterexamples (4, 4, 7) and (3, 6, 6) for p = 15, q = 9
  sorry

end cuboid_surface_area_cuboid_volume_not_unique_l117_117909


namespace XY_sym_diff_l117_117077

-- The sets X and Y
def X : Set ℤ := {1, 3, 5, 7}
def Y : Set ℤ := { x | x < 4 ∧ x ∈ Set.univ }

-- Definition of set operation (A - B)
def set_sub (A B : Set ℤ) : Set ℤ := { x | x ∈ A ∧ x ∉ B }

-- Definition of set operation (A * B)
def set_sym_diff (A B : Set ℤ) : Set ℤ := (set_sub A B) ∪ (set_sub B A)

-- Prove that X * Y = {-3, -2, -1, 0, 2, 5, 7}
theorem XY_sym_diff : set_sym_diff X Y = {-3, -2, -1, 0, 2, 5, 7} :=
by
  sorry

end XY_sym_diff_l117_117077


namespace intersection_of_sets_l117_117680

def setA : Set ℝ := {x | -2 < x ∧ x < 3}
def setB : Set ℝ := {x | 0 < x ∧ x < 4}

theorem intersection_of_sets :
  setA ∩ setB = {x | 0 < x ∧ x < 3} :=
by
  sorry

end intersection_of_sets_l117_117680


namespace range_of_x_l117_117064

variable (x y : ℝ)

def op (x y : ℝ) := x * (1 - y)

theorem range_of_x (h : op (x - 1) (x + 2) < 0) : x < -1 ∨ 1 < x :=
by
  dsimp [op] at h
  sorry

end range_of_x_l117_117064


namespace trig_identity_l117_117358

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l117_117358


namespace negation_of_proposition_l117_117224

theorem negation_of_proposition : (¬ (∀ x : ℝ, x > 2 → x > 3)) = ∃ x > 2, x ≤ 3 := by
  sorry

end negation_of_proposition_l117_117224


namespace yazhong_point_1_yazhong_point_2_yazhong_point_3_part1_yazhong_point_3_part2_l117_117728

-- Defining "Yazhong point"
def yazhong (A B M : ℝ) : Prop := abs (M - A) = abs (M - B)

-- Problem 1
theorem yazhong_point_1 {A B M : ℝ} (hA : A = -5) (hB : B = 1) (hM : yazhong A B M) : M = -2 :=
sorry

-- Problem 2
theorem yazhong_point_2 {A B M : ℝ} (hM : M = 2) (hAB : B - A = 9) (h_order : A < B) (hY : yazhong A B M) :
  (A = -5/2) ∧ (B = 13/2) :=
sorry

-- Problem 3 Part ①
theorem yazhong_point_3_part1 (A : ℝ) (B : ℝ) (m : ℤ) 
  (hA : A = -6) (hB_range : -4 ≤ B ∧ B ≤ -2) (hM : yazhong A B m) : 
  m = -5 ∨ m = -4 :=
sorry

-- Problem 3 Part ②
theorem yazhong_point_3_part2 (C D : ℝ) (n : ℤ)
  (hC : C = -4) (hD : D = -2) (hM : yazhong (-6) (C + D + 2 * n) 0) : 
  8 ≤ n ∧ n ≤ 10 :=
sorry

end yazhong_point_1_yazhong_point_2_yazhong_point_3_part1_yazhong_point_3_part2_l117_117728


namespace Megan_deleted_files_l117_117254

theorem Megan_deleted_files (initial_files folders files_per_folder deleted_files : ℕ) 
    (h1 : initial_files = 93) 
    (h2 : folders = 9)
    (h3 : files_per_folder = 8) 
    (h4 : deleted_files = initial_files - folders * files_per_folder) : 
  deleted_files = 21 :=
by
  sorry

end Megan_deleted_files_l117_117254


namespace benito_juarez_birth_year_l117_117009

theorem benito_juarez_birth_year (x : ℕ) (h1 : 1801 ≤ x ∧ x ≤ 1850) (h2 : x*x = 1849) : x = 1806 :=
by sorry

end benito_juarez_birth_year_l117_117009


namespace divide_90_into_two_parts_l117_117202

theorem divide_90_into_two_parts (x y : ℝ) (h : x + y = 90) 
  (cond : 0.4 * x = 0.3 * y + 15) : x = 60 ∨ y = 60 := 
by
  sorry

end divide_90_into_two_parts_l117_117202


namespace koi_fish_count_l117_117266

-- Define the initial conditions as variables
variables (total_fish_initial : ℕ) (goldfish_end : ℕ) (days_in_week : ℕ)
          (weeks : ℕ) (koi_add_day : ℕ) (goldfish_add_day : ℕ)

-- Expressing the problem's constraints
def problem_conditions :=
  total_fish_initial = 280 ∧
  goldfish_end = 200 ∧
  days_in_week = 7 ∧
  weeks = 3 ∧
  koi_add_day = 2 ∧
  goldfish_add_day = 5

-- Calculating the expected results based on the constraints
def total_fish_end := total_fish_initial + weeks * days_in_week * (koi_add_day + goldfish_add_day)
def koi_fish_end := total_fish_end - goldfish_end

-- The theorem to prove the number of koi fish at the end is 227
theorem koi_fish_count : problem_conditions → koi_fish_end = 227 := by
  sorry

end koi_fish_count_l117_117266


namespace min_dot_product_value_l117_117227

noncomputable def dot_product_minimum (x : ℝ) : ℝ :=
  8 * x^2 + 4 * x

theorem min_dot_product_value :
  (∀ x, dot_product_minimum x ≥ -1 / 2) ∧ (∃ x, dot_product_minimum x = -1 / 2) :=
by
  sorry

end min_dot_product_value_l117_117227


namespace find_the_number_l117_117483

theorem find_the_number (x : ℝ) (h : 100 - x = x + 40) : x = 30 :=
sorry

end find_the_number_l117_117483


namespace hexagonal_prism_min_cut_l117_117319

-- We formulate the problem conditions and the desired proof
def minimum_edges_to_cut (total_edges : ℕ) (uncut_edges : ℕ) : ℕ :=
  total_edges - uncut_edges

theorem hexagonal_prism_min_cut :
  minimum_edges_to_cut 18 7 = 11 :=
by
  sorry

end hexagonal_prism_min_cut_l117_117319


namespace xy_identity_l117_117553

theorem xy_identity (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 24) : (x^2 + y^2) * (x + y) = 803 := by
  sorry

end xy_identity_l117_117553


namespace psychologist_charge_difference_l117_117492

variables (F A : ℝ)

theorem psychologist_charge_difference
  (h1 : F + 4 * A = 375)
  (h2 : F + A = 174) :
  (F - A) = 40 :=
by sorry

end psychologist_charge_difference_l117_117492


namespace max_omega_is_2_l117_117853

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + Real.pi / 6)

theorem max_omega_is_2 {ω : ℝ} (h₀ : ω > 0) (h₁ : MonotoneOn (f ω) (Set.Icc (-Real.pi / 6) (Real.pi / 6))) :
  ω ≤ 2 :=
sorry

end max_omega_is_2_l117_117853


namespace BC_total_750_l117_117053

theorem BC_total_750 (A B C : ℤ) 
  (h1 : A + B + C = 900) 
  (h2 : A + C = 400) 
  (h3 : C = 250) : 
  B + C = 750 := 
by 
  sorry

end BC_total_750_l117_117053


namespace probability_colored_ball_l117_117911

theorem probability_colored_ball (total_balls blue_balls green_balls white_balls : ℕ)
  (h_total : total_balls = 40)
  (h_blue : blue_balls = 15)
  (h_green : green_balls = 5)
  (h_white : white_balls = 20)
  (h_disjoint : total_balls = blue_balls + green_balls + white_balls) :
  (blue_balls + green_balls) / total_balls = 1 / 2 := by
  -- Proof skipped
  sorry

end probability_colored_ball_l117_117911


namespace ellipse_eccentricity_l117_117538

theorem ellipse_eccentricity (a : ℝ) (h : a > 0) 
  (ell_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / 5 = 1 ↔ x^2 / a^2 + y^2 / 5 = 1)
  (ecc_eq : (eccentricity : ℝ) = 2 / 3) : 
  a = 3 := 
sorry

end ellipse_eccentricity_l117_117538


namespace initial_tests_count_l117_117241

theorem initial_tests_count (n S : ℕ)
  (h1 : S = 35 * n)
  (h2 : (S - 20) / (n - 1) = 40) :
  n = 4 := 
sorry

end initial_tests_count_l117_117241


namespace angle_bounds_find_configurations_l117_117247

/-- Given four points A, B, C, D on a plane, where α1 and α2 are the two smallest angles,
    and β1 and β2 are the two largest angles formed by these points, we aim to prove:
    1. 0 ≤ α2 ≤ 45 degrees,
    2. 72 degrees ≤ β2 ≤ 180 degrees,
    and to find configurations that achieve α2 = 45 degrees and β2 = 72 degrees. -/
theorem angle_bounds {A B C D : ℝ × ℝ} (α1 α2 β1 β2 : ℝ) 
  (h_angles : α1 ≤ α2 ∧ α2 ≤ β2 ∧ β2 ≤ β1 ∧ 
              0 ≤ α2 ∧ α2 ≤ 45 ∧ 
              72 ≤ β2 ∧ β2 ≤ 180) : 
  (0 ≤ α2 ∧ α2 ≤ 45 ∧ 72 ≤ β2 ∧ β2 ≤ 180) := 
by sorry

/-- Find configurations where α2 = 45 degrees and β2 = 72 degrees. -/
theorem find_configurations {A B C D : ℝ × ℝ} (α1 α2 β1 β2 : ℝ)
  (h_angles : α1 ≤ α2 ∧ α2 = 45 ∧ β2 = 72 ∧ β2 ≤ β1) :
  (α2 = 45 ∧ β2 = 72) := 
by sorry

end angle_bounds_find_configurations_l117_117247


namespace speed_increase_impossible_l117_117059

theorem speed_increase_impossible (v : ℝ) : v = 60 → (¬ ∃ v', (1 / (v' / 60) = 0)) :=
by sorry

end speed_increase_impossible_l117_117059


namespace probability_sum_odd_is_118_div_231_l117_117765

-- Defining the problem conditions
def ball_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
def drawn_balls : Finset (Finset ℕ) := Finset.powersetLen 6 ball_numbers

-- Necessary to evaluate probability with rational numbers
noncomputable def favorable_outcomes : ℚ :=
  (Finset.filter (λ s, (s.sum % 2 = 1)) drawn_balls).card

noncomputable def total_outcomes : ℚ :=
  drawn_balls.card

noncomputable def probability_odd_sum : ℚ :=
  favorable_outcomes / total_outcomes

-- Theorem to prove
theorem probability_sum_odd_is_118_div_231 : probability_odd_sum = 118 / 231 :=
by {
  -- Statement needs to calculate the specific probability which requires combinatorial reasoning
  sorry
}

end probability_sum_odd_is_118_div_231_l117_117765


namespace tree_height_at_2_years_l117_117504

theorem tree_height_at_2_years (h : ℕ → ℚ) (h5 : h 5 = 243) 
  (h_rec : ∀ n, h (n - 1) = h n / 3) : h 2 = 9 :=
  sorry

end tree_height_at_2_years_l117_117504


namespace smallest_percent_coffee_tea_l117_117327

theorem smallest_percent_coffee_tea (C T : ℝ) (hC : C = 50) (hT : T = 60) : 
  ∃ x, x = C + T - 100 ∧ x = 10 :=
by
  sorry

end smallest_percent_coffee_tea_l117_117327


namespace angle_sum_around_point_l117_117473

theorem angle_sum_around_point (y : ℝ) (h : 170 + y + y = 360) : y = 95 := 
sorry

end angle_sum_around_point_l117_117473


namespace find_a_plus_b_l117_117663
-- Definition of the problem variables and conditions
variables (a b : ℝ)
def condition1 : Prop := a - b = 3
def condition2 : Prop := a^2 - b^2 = -12

-- Goal: Prove that a + b = -4 given the conditions
theorem find_a_plus_b (h1 : condition1 a b) (h2 : condition2 a b) : a + b = -4 :=
  sorry

end find_a_plus_b_l117_117663


namespace Sally_next_birthday_age_l117_117513

variables (a m s d : ℝ)

def Adam_older_than_Mary := a = 1.3 * m
def Mary_younger_than_Sally := m = 0.75 * s
def Sally_younger_than_Danielle := s = 0.8 * d
def Sum_ages := a + m + s + d = 60

theorem Sally_next_birthday_age (a m s d : ℝ) 
  (H1 : Adam_older_than_Mary a m)
  (H2 : Mary_younger_than_Sally m s)
  (H3 : Sally_younger_than_Danielle s d)
  (H4 : Sum_ages a m s d) : 
  s + 1 = 16 := 
by sorry

end Sally_next_birthday_age_l117_117513


namespace system_solution_unique_l117_117134

noncomputable def solve_system (n : ℕ) (x : Fin n → ℝ) : Prop :=
(∀ k : ℕ, k > 0 ∧ k ≤ n → ∑ i in Finset.univ, (x i) ^ k = n)

theorem system_solution_unique (n : ℕ) (x : Fin n → ℝ) :
  solve_system n x → (∀ i : Fin n, x i = 1) :=
by
  intro h
  -- Here would come the proof
  sorry

end system_solution_unique_l117_117134


namespace probability_two_red_balls_randomly_picked_l117_117040

theorem probability_two_red_balls_randomly_picked :
  (3/9) * (2/8) = 1/12 :=
by sorry

end probability_two_red_balls_randomly_picked_l117_117040


namespace fraction_of_As_l117_117696

-- Define the conditions
def fraction_B (T : ℕ) := 1/4 * T
def fraction_C (T : ℕ) := 1/2 * T
def remaining_D : ℕ := 20
def total_students_approx : ℕ := 400

-- State the theorem
theorem fraction_of_As 
  (F : ℚ) : 
  ∀ T : ℕ, 
  T = F * T + fraction_B T + fraction_C T + remaining_D → 
  T = total_students_approx → 
  F = 1/5 :=
by
  intros
  sorry

end fraction_of_As_l117_117696


namespace expand_binomials_l117_117650

theorem expand_binomials (x : ℝ) : (x - 3) * (4 * x + 8) = 4 * x^2 - 4 * x - 24 :=
by
  sorry

end expand_binomials_l117_117650


namespace seventh_term_arith_seq_l117_117459

/-- 
The seventh term of an arithmetic sequence given that the sum of the first five terms 
is 15 and the sixth term is 7.
-/
theorem seventh_term_arith_seq (a d : ℚ) 
  (h1 : 5 * a + 10 * d = 15) 
  (h2 : a + 5 * d = 7) : 
  a + 6 * d = 25 / 3 := 
sorry

end seventh_term_arith_seq_l117_117459


namespace cube_tangent_ratio_l117_117659

theorem cube_tangent_ratio 
  (edge_length : ℝ) 
  (midpoint K : ℝ) 
  (tangent E : ℝ) 
  (intersection F : ℝ) 
  (radius R : ℝ)
  (h1 : edge_length = 2)
  (h2 : radius = 1)
  (h3 : K = midpoint)
  (h4 : ∃ E F, tangent = E ∧ intersection = F) :
  (K - E) / (F - E) = 4 / 5 :=
sorry

end cube_tangent_ratio_l117_117659


namespace ratio_x_y_l117_117102

theorem ratio_x_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 5 / 4) : x / y = 13 / 2 :=
by
  sorry

end ratio_x_y_l117_117102


namespace solve_equation_l117_117651

theorem solve_equation(x : ℝ) :
  (real.sqrt ((3 + 2 * real.sqrt 2) ^ x) + real.sqrt ((3 - 2 * real.sqrt 2) ^ x)) = 6 → (x = 2 ∨ x = -2) :=
by
  sorry

end solve_equation_l117_117651


namespace estimated_germination_probability_l117_117608

-- This definition represents the conditions of the problem in Lean.
def germination_data : List (ℕ × ℕ × Real) :=
  [(2, 2, 1.000), (5, 4, 0.800), (10, 9, 0.900), (50, 44, 0.880), (100, 92, 0.920),
   (500, 463, 0.926), (1000, 928, 0.928), (1500, 1396, 0.931), (2000, 1866, 0.933), (3000, 2794, 0.931)]

-- The theorem states that the germination probability is approximately 0.93.
theorem estimated_germination_probability (data : List (ℕ × ℕ × Real)) (h : data = germination_data) :
  ∃ p : Real, p = 0.93 ∧ ∀ n m r, (n, m, r) ∈ data → |r - p| < 0.01 :=
by
  -- Placeholder for proof
  sorry

end estimated_germination_probability_l117_117608


namespace prime_divides_diff_of_cubes_l117_117780

theorem prime_divides_diff_of_cubes (a b c : ℕ) [Fact (Nat.Prime a)] [Fact (Nat.Prime b)]
  (h1 : c ∣ (a + b)) (h2 : c ∣ (a * b)) : c ∣ (a^3 - b^3) :=
by
  sorry

end prime_divides_diff_of_cubes_l117_117780


namespace neznaika_is_wrong_l117_117301

theorem neznaika_is_wrong (avg_december avg_january : ℝ)
  (h_avg_dec : avg_december = 10)
  (h_avg_jan : avg_january = 5) : 
  ∃ (dec_days jan_days : ℕ), 
    (avg_december = (dec_days * 10 + (31 - dec_days) * 0) / 31) ∧
    (avg_january = (jan_days * 10 + (31 - jan_days) * 0) / 31) ∧
    jan_days > dec_days :=
by 
  sorry

end neznaika_is_wrong_l117_117301


namespace part_a_part_b_part_c_l117_117019

def transformable (w1 w2 : String) : Prop :=
∀ q : String → String → Prop,
  (q "xy" "yyx") →
  (q "xt" "ttx") →
  (q "yt" "ty") →
  (q w1 w2)

theorem part_a : ¬ transformable "xy" "xt" :=
sorry

theorem part_b : ¬ transformable "xytx" "txyt" :=
sorry

theorem part_c : transformable "xtxyy" "ttxyyyyx" :=
sorry

end part_a_part_b_part_c_l117_117019


namespace candy_store_food_colouring_amount_l117_117174

theorem candy_store_food_colouring_amount :
  let lollipop_colour := 5 -- each lollipop uses 5ml of food colouring
  let hard_candy_colour := 20 -- each hard candy uses 20ml of food colouring
  let num_lollipops := 100 -- the candy store makes 100 lollipops in one day
  let num_hard_candies := 5 -- the candy store makes 5 hard candies in one day
  (num_lollipops * lollipop_colour) + (num_hard_candies * hard_candy_colour) = 600 :=
by
  let lollipop_colour := 5
  let hard_candy_colour := 20
  let num_lollipops := 100
  let num_hard_candies := 5
  show (num_lollipops * lollipop_colour) + (num_hard_candies * hard_candy_colour) = 600
  sorry

end candy_store_food_colouring_amount_l117_117174


namespace triangle_area_202_2192_pi_squared_l117_117510

noncomputable def triangle_area (a b c : ℝ) : ℝ := 
  let r := (a + b + c) / (2 * Real.pi)
  let theta := 20.0 * Real.pi / 180.0  -- converting 20 degrees to radians
  let angle1 := 5 * theta
  let angle2 := 6 * theta
  let angle3 := 7 * theta
  (1 / 2) * r * r * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem triangle_area_202_2192_pi_squared (a b c : ℝ) (h1 : a = 5) (h2 : b = 6) (h3 : c = 7) : 
  triangle_area a b c = 202.2192 / (Real.pi * Real.pi) := 
by {
  sorry
}

end triangle_area_202_2192_pi_squared_l117_117510


namespace mans_rate_in_still_water_l117_117168

theorem mans_rate_in_still_water (Vm Vs : ℝ) (h1 : Vm + Vs = 14) (h2 : Vm - Vs = 4) : Vm = 9 :=
by
  sorry

end mans_rate_in_still_water_l117_117168


namespace min_value_of_expression_min_value_achieved_at_l117_117204

theorem min_value_of_expression (x : ℝ) (hx : 0 < x) : 
  3 * Real.sqrt x + 4 / (x^2) ≥ 4 * 4^(1/5) :=
sorry

theorem min_value_achieved_at (x : ℝ) (hx : 0 < x) (h : x = 4^(2/5)) :
  3 * Real.sqrt x + 4 / (x^2) = 4 * 4^(1/5) :=
sorry

end min_value_of_expression_min_value_achieved_at_l117_117204


namespace Benjamin_skating_time_l117_117852

-- Definitions based on the conditions in the problem
def distance : ℕ := 80 -- Distance skated in kilometers
def speed : ℕ := 10 -- Speed in kilometers per hour

-- Theorem to prove that the skating time is 8 hours
theorem Benjamin_skating_time : distance / speed = 8 := by
  -- Proof goes here, we skip it with sorry
  sorry

end Benjamin_skating_time_l117_117852


namespace speed_against_current_l117_117310

theorem speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) (man_speed_against_current : ℝ) 
  (h : speed_with_current = 12) (h1 : current_speed = 2) : man_speed_against_current = 8 :=
by
  sorry

end speed_against_current_l117_117310


namespace factorize_polynomial_l117_117967

theorem factorize_polynomial (x y : ℝ) : 
  (2 * x^2 * y - 4 * x * y^2 + 2 * y^3) = 2 * y * (x - y)^2 :=
sorry

end factorize_polynomial_l117_117967


namespace sum_of_tangencies_l117_117063

noncomputable def f (x : ℝ) : ℝ := max (-7 * x - 23) (max (2 * x + 5) (5 * x + 17))

noncomputable def q (x : ℝ) : ℝ := sorry  -- since the exact form of q is not specified, we use sorry here

-- Define the tangency condition
def is_tangent (q f : ℝ → ℝ) (x : ℝ) : Prop := (q x = f x) ∧ (deriv q x = deriv f x)

-- Define the three points of tangency
variable {x₄ x₅ x₆ : ℝ}

-- q(x) is tangent to f(x) at points x₄, x₅, x₆
axiom tangent_x₄ : is_tangent q f x₄
axiom tangent_x₅ : is_tangent q f x₅
axiom tangent_x₆ : is_tangent q f x₆

-- Now state the theorem
theorem sum_of_tangencies : x₄ + x₅ + x₆ = -70 / 9 :=
sorry

end sum_of_tangencies_l117_117063


namespace roots_of_quadratic_eq_l117_117280

theorem roots_of_quadratic_eq (h : ∀ x : ℝ, x^2 - 3 * x = 0 → x = 0 ∨ x = 3) :
  ∀ x : ℝ, x^2 - 3 * x = 0 → x = 0 ∨ x = 3 :=
by sorry

end roots_of_quadratic_eq_l117_117280


namespace my_problem_l117_117593

theorem my_problem (a b c : ℝ) (h1 : a < b) (h2 : b < c) :
  a^2 * b + b^2 * c + c^2 * a < a^2 * c + b^2 * a + c^2 * b := 
sorry

end my_problem_l117_117593


namespace abs_sum_eq_two_l117_117550

theorem abs_sum_eq_two (a b c : ℤ) (h : (a - b) ^ 10 + (a - c) ^ 10 = 1) : 
  abs (a - b) + abs (b - c) + abs (c - a) = 2 := 
sorry

end abs_sum_eq_two_l117_117550


namespace arithmetic_sequence_statements_l117_117281

/-- 
Given the arithmetic sequence {a_n} with first term a_1 > 0 and the sum of the first n terms denoted as S_n, 
prove the following statements based on the condition S_8 = S_16:
  1. d > 0
  2. a_{13} < 0
  3. The maximum value of S_n is S_{12}
  4. When S_n < 0, the minimum value of n is 25
--/
theorem arithmetic_sequence_statements (a_1 d : ℤ) (S : ℕ → ℤ)
  (h1 : a_1 > 0)
  (h2 : S 8 = S 16)
  (hS8 : S 8 = 8 * a_1 + 28 * d)
  (hS16 : S 16 = 16 * a_1 + 120 * d) :
  (d > 0) ∨ 
  (a_1 + 12 * d < 0) ∨ 
  (∀ n, n ≠ 12 → S n ≤ S 12) ∨ 
  (∀ n, S n < 0 → n ≥ 25) :=
sorry

end arithmetic_sequence_statements_l117_117281


namespace chocolate_distribution_l117_117437

theorem chocolate_distribution
  (total_chocolate : ℚ)
  (num_piles : ℕ)
  (piles_given_to_shaina : ℕ)
  (weight_each_pile : ℚ)
  (weight_of_shaina_piles : ℚ)
  (h1 : total_chocolate = 72 / 7)
  (h2 : num_piles = 6)
  (h3 : piles_given_to_shaina = 2)
  (h4 : weight_each_pile = total_chocolate / num_piles)
  (h5 : weight_of_shaina_piles = piles_given_to_shaina * weight_each_pile) :
  weight_of_shaina_piles = 24 / 7 := by
  sorry

end chocolate_distribution_l117_117437


namespace total_length_of_joined_papers_l117_117616

theorem total_length_of_joined_papers :
  let length_each_sheet := 10 -- in cm
  let number_of_sheets := 20
  let overlap_length := 0.5 -- in cm
  let total_overlapping_connections := number_of_sheets - 1
  let total_length_without_overlap := length_each_sheet * number_of_sheets
  let total_overlap_length := overlap_length * total_overlapping_connections
  let total_length := total_length_without_overlap - total_overlap_length
  total_length = 190.5 :=
by {
    sorry
}

end total_length_of_joined_papers_l117_117616


namespace min_value_proof_l117_117118

noncomputable def minimum_value (a b c : ℝ) : ℝ :=
  9 / a + 16 / b + 25 / (c ^ 2)

theorem min_value_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 5) :
  minimum_value a b c ≥ 50 :=
sorry

end min_value_proof_l117_117118


namespace eccentricity_of_ellipse_l117_117342

axiom ellipse_def (a b: ℝ) (ha: a > 0) (hb: b > 0) (hab: a > b):
  ∃ x y: ℝ, (x/a) ^ 2 + (y/b)^ 2 = 1

axiom right_focus (a b: ℝ) (ha: a > 0) (hb: b > 0) (hab: a > b): 
  ∃ c: ℝ, c = (a ^ 2 - b ^ 2) ^ (1/2) ∧ c < a ∧ c > 0 

axiom line_intersect (a b c: ℝ) (ha: a > 0) (hb: b > 0) (hab: a > b) (hc: c = (a ^ 2 - b ^ 2) ^ (1/2)): 
  ∃ MN: ℝ, MN = 2 * (b ^ 2) / a ∧ MN = 3 * (a - c)

theorem eccentricity_of_ellipse (a b: ℝ) (ha: a > 0) (hb: b > 0) (hab: a > b) (MN: ℝ) (c: ℝ):
  2 * ((a ^ 2 - b ^ 2) / a ^ 2) - 3 * ((a ^ 2 - b ^ 2) ^ (1 / 2) / a) + 1 = 0 → 
  3 * (a - (a ^ 2 - b ^ 2) ^ (1 / 2)) = 2 * (b ^ 2) / a → 
  (a ^ 2 - b ^ 2) ^ (1 / 2)) / a = 1 / 2

end eccentricity_of_ellipse_l117_117342


namespace least_n_condition_l117_117574

-- Define the conditions and the question in Lean 4
def jackson_position (n : ℕ) : ℕ := sorry  -- Defining the position of Jackson after n steps

def expected_value (n : ℕ) : ℝ := sorry  -- Defining the expected value E_n

theorem least_n_condition : ∃ n : ℕ, (1 / expected_value n > 2017) ∧ (∀ m < n, 1 / expected_value m ≤ 2017) ∧ n = 13446 :=
by {
  -- Jackson starts at position 1
  -- The conditions described in the problem will be formulated here
  -- We need to show that the least n such that 1 / E_n > 2017 is 13446
  sorry
}

end least_n_condition_l117_117574


namespace probability_final_marble_red_l117_117806

theorem probability_final_marble_red :
  let P_wA := 5/8
  let P_bA := 3/8
  let P_gB_p1 := 6/10
  let P_rB_p2 := 4/10
  let P_gC_p1 := 4/7
  let P_rC_p2 := 3/7
  let P_wr_b_g := P_wA * P_gB_p1 * P_rB_p2
  let P_blk_g_red := P_bA * P_gC_p1 * P_rC_p2
  (P_wr_b_g + P_blk_g_red) = 79/980 :=
by {
  let P_wA := 5/8
  let P_bA := 3/8
  let P_gB_p1 := 6/10
  let P_rB_p2 := 4/10
  let P_gC_p1 := 4/7
  let P_rC_p2 := 3/7
  let P_wr_b_g := P_wA * P_gB_p1 * P_rB_p2
  let P_blk_g_red := P_bA * P_gC_p1 * P_rC_p2
  show (P_wr_b_g + P_blk_g_red) = 79/980
  sorry
}

end probability_final_marble_red_l117_117806


namespace bags_sold_in_afternoon_l117_117185

theorem bags_sold_in_afternoon (bags_morning : ℕ) (weight_per_bag : ℕ) (total_weight : ℕ) 
  (h1 : bags_morning = 29) (h2 : weight_per_bag = 7) (h3 : total_weight = 322) : 
  total_weight - bags_morning * weight_per_bag / weight_per_bag = 17 := 
by 
  sorry

end bags_sold_in_afternoon_l117_117185


namespace vector_magnitude_l117_117987

variables {V : Type*} [inner_product_space ℝ V]

-- Define the vectors a and b
variables (a b : V)

-- Given conditions
axiom norm_a : ∥a∥ = 1
axiom norm_b : ∥b∥ = 1
axiom dot_ab : ⟪a, b⟫ = -1 / 2

-- The theorem to be proven
theorem vector_magnitude : ∥a + 2 • b∥ = real.sqrt 3 :=
sorry

end vector_magnitude_l117_117987


namespace greta_hourly_wage_is_12_l117_117094

-- Define constants
def greta_hours : ℕ := 40
def lisa_hourly_wage : ℕ := 15
def lisa_hours : ℕ := 32

-- Define the total earnings of Greta and Lisa
def greta_earnings (G : ℕ) : ℕ := greta_hours * G
def lisa_earnings : ℕ := lisa_hours * lisa_hourly_wage

-- Main theorem statement
theorem greta_hourly_wage_is_12 (G : ℕ) (h : greta_earnings G = lisa_earnings) : G = 12 :=
by
  sorry

end greta_hourly_wage_is_12_l117_117094


namespace find_ordered_pair_l117_117445

variables {A B Q : Type} -- Points A, B, Q
variables [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q]
variables {a b q : A} -- Vectors at points A, B, Q
variables (r : ℝ) -- Ratio constant

-- Define the conditions from the original problem
def ratio_aq_qb (A B Q : Type) [AddCommGroup A] [AddCommGroup B] [AddCommGroup Q] (a b q : A) (r : ℝ) :=
  r = 7 / 2

-- Define the goal theorem using the conditions above
theorem find_ordered_pair (h : ratio_aq_qb A B Q a b q r) : 
  q = (7 / 9) • a + (2 / 9) • b :=
sorry

end find_ordered_pair_l117_117445


namespace chi_square_test_not_significant_probability_winning_prize_probability_no_C_given_no_prize_l117_117307


-- Part 1: Chi-square test
theorem chi_square_test_not_significant :
  let a := 20
  let b := 75
  let c := 10
  let d := 45
  let n := 150
  let chi_square := (n * (a * d - b * c) ^ 2) / ( (a + b) * (c + d) * (a + c) * (b + d) )
  χ2 < 6.635 :=
by {
  sorry
}

-- Part 2: Probability Calculations

-- 1. Probability of winning a prize
theorem probability_winning_prize :
  let prob_a := (2 / 5 : ℝ)
  let prob_b := (2 / 5 : ℝ)
  let prob_c := (1 / 5 : ℝ)
  let prob_win := prob_a * prob_b * prob_c * 6
  prob_win = 24 / 125 :=
by {
  sorry
}

-- 2. Given not winning a prize, calculate the probability of not getting a C card
theorem probability_no_C_given_no_prize :
  let prob_no_c_given_no_prize := (64 / 125 : ℝ) / (1 - 24 / 125)
  prob_no_c_given_no_prize = 64 / 101 :=
by {
  sorry
}

end chi_square_test_not_significant_probability_winning_prize_probability_no_C_given_no_prize_l117_117307


namespace fraction_unchanged_l117_117968

-- Define the digit rotation
def rotate (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => d  -- for completeness, though we assume d only takes {0, 1, 6, 8, 9}

-- Define the condition for a fraction to be unchanged when flipped
def unchanged_when_flipped (numerator denominator : ℕ) : Prop :=
  let rotated_numerator := rotate numerator
  let rotated_denominator := rotate denominator
  rotated_numerator * denominator = rotated_denominator * numerator

-- Define the specific fraction 6/9
def specific_fraction_6_9 : Prop :=
  unchanged_when_flipped 6 9 ∧ 6 < 9

-- Theorem stating 6/9 is unchanged when its digits are flipped and it's a valid fraction
theorem fraction_unchanged : specific_fraction_6_9 :=
by
  sorry

end fraction_unchanged_l117_117968


namespace max_bishops_correct_bishop_position_count_correct_l117_117939

-- Define the parameters and predicates
def chessboard_size : ℕ := 2015

def max_bishops (board_size : ℕ) : ℕ := 2 * board_size - 1 - 1

def bishop_position_count (board_size : ℕ) : ℕ := 2 ^ (board_size - 1) * 2 * 2

-- State the equalities to be proved
theorem max_bishops_correct : max_bishops chessboard_size = 4028 := by
  -- proof will be here
  sorry

theorem bishop_position_count_correct : bishop_position_count chessboard_size = 2 ^ 2016 := by
  -- proof will be here
  sorry

end max_bishops_correct_bishop_position_count_correct_l117_117939


namespace solve_for_x_l117_117269

theorem solve_for_x (x : ℕ) : (3 : ℝ)^(27^x) = (27 : ℝ)^(3^x) → x = 0 :=
by
  sorry

end solve_for_x_l117_117269


namespace koi_fish_after_three_weeks_l117_117262

theorem koi_fish_after_three_weeks
  (f_0 : ℕ := 280) -- initial total number of fish
  (days : ℕ := 21) -- days in 3 weeks
  (koi_added_per_day : ℕ := 2)
  (goldfish_added_per_day : ℕ := 5)
  (goldfish_after_3_weeks : ℕ := 200) :
  let total_fish_added := days * (koi_added_per_day + goldfish_added_per_day)
  let total_fish_after := f_0 + total_fish_added
  let koi_after_3_weeks := total_fish_after - goldfish_after_3_weeks
  koi_after_3_weeks = 227 :=
by
  let total_fish_added := days * (koi_added_per_day + goldfish_added_per_day)
  let total_fish_after := f_0 + total_fish_added
  let koi_after_3_weeks := total_fish_after - goldfish_after_3_weeks
  sorry

end koi_fish_after_three_weeks_l117_117262


namespace matrix_cubic_l117_117952

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![2, -1],
  ![1, 1]
]

theorem matrix_cubic :
  matrix_a ^ 3 = ![
    ![3, -6],
    ![6, -3]
  ] := by
  sorry

end matrix_cubic_l117_117952


namespace smallest_positive_value_floor_l117_117246

noncomputable def g (x : ℝ) : ℝ := Real.cos x - Real.sin x + 4 * Real.tan x

theorem smallest_positive_value_floor :
  ∃ s > 0, g s = 0 ∧ ⌊s⌋ = 3 :=
sorry

end smallest_positive_value_floor_l117_117246


namespace sufficient_but_not_necessary_l117_117087

theorem sufficient_but_not_necessary (x : ℝ) (h : 1 / x > 1) : x < 1 := by
  sorry

end sufficient_but_not_necessary_l117_117087


namespace eyes_saw_plane_l117_117766

theorem eyes_saw_plane (total_students : ℕ) (fraction_looked_up : ℚ) (students_with_eyepatches : ℕ) :
  total_students = 200 → fraction_looked_up = 3/4 → students_with_eyepatches = 20 →
  ∃ eyes_saw_plane, eyes_saw_plane = 280 :=
by
  intros h1 h2 h3
  sorry

end eyes_saw_plane_l117_117766


namespace value_of_y_l117_117673

theorem value_of_y (x y : ℝ) (h1 : x^(2 * y) = 8) (h2 : x = 2) : y = 3 / 2 :=
by
  sorry

end value_of_y_l117_117673


namespace min_people_with_all_items_in_Owlna_l117_117432

theorem min_people_with_all_items_in_Owlna (population : ℕ) 
(refrigerator_pct television_pct computer_pct air_conditioner_pct washing_machine_pct smartphone_pct : ℚ) 
(h_population : population = 5000) 
(h_refrigerator : refrigerator_pct = 0.72) 
(h_television : television_pct = 0.75)
(h_computer : computer_pct = 0.65)
(h_air_conditioner : air_conditioner_pct = 0.95)
(h_washing_machine : washing_machine_pct = 0.80)
(h_smartphone : smartphone_pct = 0.60) 
: ∃ (min_people : ℕ), min_people = 3000 :=
sorry

end min_people_with_all_items_in_Owlna_l117_117432


namespace trigonometric_identity_l117_117406

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l117_117406


namespace no_blue_socks_make_probability_half_l117_117744

theorem no_blue_socks_make_probability_half :
  ∀ (n m : ℕ), n + m = 2009 → (n - m) * (n - m) ≠ 2009 :=
begin
  intros n m h,
  by_contra H,
  sorry, -- Proof goes here
end

end no_blue_socks_make_probability_half_l117_117744


namespace count_prime_sums_6_l117_117597

open BigOperators
open Nat

def is_prime_list (l : List ℕ) := l.filter Nat.Prime

noncomputable def generate_sequence : List ℕ :=
  let primes := nat.primes.map (λ x, Nat.succ x) -- to adjust for skipped primes
  (List.range 15).map (λ i, 
    let n := primes.take (i+1).sum -- take i+1 primes and compute the sum.
    if i % 2 = 1 then n - primes[i] else n 
  )

theorem count_prime_sums_6 : is_prime_list generate_sequence.length = 6 := by
  sorry

end count_prime_sums_6_l117_117597


namespace john_shower_duration_l117_117712

variable (days_per_week : ℕ := 7)
variable (weeks : ℕ := 4)
variable (total_days : ℕ := days_per_week * weeks)
variable (shower_frequency : ℕ := 2) -- every other day
variable (number_of_showers : ℕ := total_days / shower_frequency)
variable (total_gallons_used : ℕ := 280)
variable (gallons_per_shower : ℕ := total_gallons_used / number_of_showers)
variable (gallons_per_minute : ℕ := 2)

theorem john_shower_duration 
  (h_cond : total_gallons_used = number_of_showers * gallons_per_shower)
  (h_shower_eq : total_days / shower_frequency = number_of_showers)
  : gallons_per_shower / gallons_per_minute = 10 :=
by
  sorry

end john_shower_duration_l117_117712


namespace average_student_headcount_l117_117324

theorem average_student_headcount (headcount_03_04 headcount_04_05 : ℕ) 
  (h1 : headcount_03_04 = 10500) 
  (h2 : headcount_04_05 = 10700) : 
  (headcount_03_04 + headcount_04_05) / 2 = 10600 := 
by
  sorry

end average_student_headcount_l117_117324


namespace f_of_3_l117_117678

def f (x : ℕ) : ℤ :=
  if x = 0 then sorry else 2 * (x - 1) - 1  -- Define an appropriate value for f(0) later

theorem f_of_3 : f 3 = 3 := by
  sorry

end f_of_3_l117_117678


namespace molecular_weight_of_3_moles_l117_117022

def molecular_weight_one_mole : ℝ := 176.14
def number_of_moles : ℝ := 3
def total_weight := number_of_moles * molecular_weight_one_mole

theorem molecular_weight_of_3_moles :
  total_weight = 528.42 := sorry

end molecular_weight_of_3_moles_l117_117022


namespace complement_U_A_l117_117349

-- Define the sets U and A
def U : Set Int := {-1, 0, 1, 2}
def A : Set Int := {-1, 1, 2}

-- State the theorem
theorem complement_U_A :
  U \ A = {0} :=
sorry

end complement_U_A_l117_117349


namespace tree_height_at_2_years_l117_117509

theorem tree_height_at_2_years (h₅ : ℕ) (h_four : ℕ) (h_three : ℕ) (h_two : ℕ) (h₅_value : h₅ = 243)
  (h_four_value : h_four = h₅ / 3) (h_three_value : h_three = h_four / 3) (h_two_value : h_two = h_three / 3) :
  h_two = 9 := by
  sorry

end tree_height_at_2_years_l117_117509


namespace height_of_water_in_cylindrical_tank_l117_117627

theorem height_of_water_in_cylindrical_tank :
  let r_cone := 15  -- radius of base of conical tank in cm
  let h_cone := 24  -- height of conical tank in cm
  let r_cylinder := 18  -- radius of base of cylindrical tank in cm
  let V_cone := (1 / 3 : ℝ) * Real.pi * r_cone^2 * h_cone  -- volume of conical tank
  let h_cyl := V_cone / (Real.pi * r_cylinder^2)  -- height of water in cylindrical tank
  h_cyl = 5.56 :=
by
  sorry

end height_of_water_in_cylindrical_tank_l117_117627


namespace equation_solution_l117_117132

noncomputable def solve_equation : Set ℝ := {x : ℝ | (3 * x + 2) / (x ^ 2 + 5 * x + 6) = 3 * x / (x - 1)
                                             ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 1}

theorem equation_solution (r : ℝ) (h : r ∈ solve_equation) : 3 * r ^ 3 + 12 * r ^ 2 + 19 * r + 2 = 0 :=
sorry

end equation_solution_l117_117132


namespace solve_otimes_eq_l117_117647

def otimes (a b : ℝ) : ℝ := (a - 2) * (b + 1)

theorem solve_otimes_eq : ∃ x : ℝ, otimes (-4) (x + 3) = 6 ↔ x = -5 :=
by
  use -5
  simp [otimes]
  sorry

end solve_otimes_eq_l117_117647


namespace capacity_of_new_bathtub_is_400_liters_l117_117770

-- Definitions based on conditions
def possible_capacities : Set ℕ := {4, 40, 400, 4000}  -- The possible capacities

-- Proof statement
theorem capacity_of_new_bathtub_is_400_liters (c : ℕ) 
  (h : c ∈ possible_capacities) : 
  c = 400 := 
sorry

end capacity_of_new_bathtub_is_400_liters_l117_117770


namespace sum_of_a_and_b_l117_117220

theorem sum_of_a_and_b (a b : ℤ) (h1 : a + 2 * b = 8) (h2 : 2 * a + b = 4) : a + b = 4 := by
  sorry

end sum_of_a_and_b_l117_117220


namespace combined_cost_price_correct_l117_117295

def face_value_A : ℝ := 100
def discount_A : ℝ := 0.02
def face_value_B : ℝ := 100
def premium_B : ℝ := 0.015
def brokerage : ℝ := 0.002

def purchase_price_A := face_value_A * (1 - discount_A)
def brokerage_fee_A := purchase_price_A * brokerage
def total_cost_price_A := purchase_price_A + brokerage_fee_A

def purchase_price_B := face_value_B * (1 + premium_B)
def brokerage_fee_B := purchase_price_B * brokerage
def total_cost_price_B := purchase_price_B + brokerage_fee_B

def combined_cost_price := total_cost_price_A + total_cost_price_B

theorem combined_cost_price_correct :
  combined_cost_price = 199.899 :=
by
  sorry

end combined_cost_price_correct_l117_117295


namespace cube_edge_length_l117_117177

theorem cube_edge_length (n_edges : ℕ) (total_length : ℝ) (length_one_edge : ℝ) 
  (h1: n_edges = 12) (h2: total_length = 96) : length_one_edge = 8 :=
by
  sorry

end cube_edge_length_l117_117177


namespace negation_of_proposition_l117_117039

theorem negation_of_proposition (x : ℝ) :
  ¬ (∃ x > -1, x^2 + x - 2018 > 0) ↔ ∀ x > -1, x^2 + x - 2018 ≤ 0 := sorry

end negation_of_proposition_l117_117039


namespace range_of_m_l117_117855

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, 3 * x^2 + 1 ≥ m * x * (x - 1)) : -6 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l117_117855


namespace simplify_expression_l117_117893

-- Define a variable x
variable (x : ℕ)

-- Statement of the problem
theorem simplify_expression : 120 * x - 75 * x = 45 * x := sorry

end simplify_expression_l117_117893


namespace quadratic_vertex_property_l117_117088

variable {a b c x0 y0 m n : ℝ}

-- Condition 1: (x0, y0) is a fixed point on the graph of the quadratic function y = ax^2 + bx + c
axiom fixed_point_on_graph : y0 = a * x0^2 + b * x0 + c

-- Condition 2: (m, n) is a moving point on the graph of the quadratic function
axiom moving_point_on_graph : n = a * m^2 + b * m + c

-- Condition 3: For any real number m, a(y0 - n) ≤ 0
axiom inequality_condition : ∀ m : ℝ, a * (y0 - (a * m^2 + b * m + c)) ≤ 0

-- Statement to prove
theorem quadratic_vertex_property : 2 * a * x0 + b = 0 := 
sorry

end quadratic_vertex_property_l117_117088


namespace ratio_of_votes_l117_117105

theorem ratio_of_votes (randy_votes : ℕ) (shaun_votes : ℕ) (eliot_votes : ℕ)
  (h1 : randy_votes = 16)
  (h2 : shaun_votes = 5 * randy_votes)
  (h3 : eliot_votes = 160) :
  eliot_votes / shaun_votes = 2 :=
by
  sorry

end ratio_of_votes_l117_117105


namespace find_triples_l117_117111

theorem find_triples (x y p : ℤ) (prime_p : Prime p) :
  x^2 - 3 * x * y + p^2 * y^2 = 12 * p ↔ 
  (p = 3 ∧ ( (x = 6 ∧ y = 0) ∨ (x = -6 ∧ y = 0) ∨ (x = 4 ∧ y = 2) ∨ (x = -2 ∧ y = 2) ∨ (x = 2 ∧ y = -2) ∨ (x = -4 ∧ y = -2) ) ) := 
by
  sorry

end find_triples_l117_117111


namespace simplify_and_evaluate_l117_117732

theorem simplify_and_evaluate (m n : ℤ) (h1 : m = 1) (h2 : n = -2) :
  -2 * (m * n - 3 * m^2) - (2 * m * n - 5 * (m * n - m^2)) = -1 :=
by
  sorry

end simplify_and_evaluate_l117_117732


namespace amount_spent_on_giftwrapping_and_expenses_l117_117992

theorem amount_spent_on_giftwrapping_and_expenses (total_spent : ℝ) (cost_of_gifts : ℝ) (h_total_spent : total_spent = 700) (h_cost_of_gifts : cost_of_gifts = 561) : 
  total_spent - cost_of_gifts = 139 :=
by
  rw [h_total_spent, h_cost_of_gifts]
  norm_num

end amount_spent_on_giftwrapping_and_expenses_l117_117992


namespace kamal_average_marks_l117_117713

theorem kamal_average_marks :
  let total_marks_obtained := 66 + 65 + 77 + 62 + 75 + 58
  let total_max_marks := 150 + 120 + 180 + 140 + 160 + 90
  (total_marks_obtained / total_max_marks.toFloat) * 100 = 48.0 :=
by
  sorry

end kamal_average_marks_l117_117713


namespace negation_of_forall_x_geq_1_l117_117905

theorem negation_of_forall_x_geq_1 :
  (¬ (∀ x : ℝ, x^2 + 1 ≥ 1)) ↔ (∃ x : ℝ, x^2 + 1 < 1) :=
by
  sorry

end negation_of_forall_x_geq_1_l117_117905


namespace proof_by_contradiction_conditions_l117_117474

theorem proof_by_contradiction_conditions:
  (∃ (neg_conclusion known_conditions ax_thms_defs original_conclusion : Prop),
    (neg_conclusion ∧ known_conditions ∧ ax_thms_defs) → False)
:= sorry

end proof_by_contradiction_conditions_l117_117474


namespace unique_prime_n_l117_117328

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem unique_prime_n (n : ℕ)
  (h1 : isPrime n)
  (h2 : isPrime (n^2 + 10))
  (h3 : isPrime (n^2 - 2))
  (h4 : isPrime (n^3 + 6))
  (h5 : isPrime (n^5 + 36)) : n = 7 :=
by
  sorry

end unique_prime_n_l117_117328


namespace matrix_power_is_correct_l117_117958

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]
def A_cubed : Matrix (Fin 2) (Fin 2) ℤ := !![3, -6; 6, -3]

theorem matrix_power_is_correct : A ^ 3 = A_cubed := by 
  sorry

end matrix_power_is_correct_l117_117958


namespace initial_nickels_l117_117730

theorem initial_nickels (N : ℕ) (h1 : N + 9 + 2 = 18) : N = 7 :=
by sorry

end initial_nickels_l117_117730


namespace allocation_schemes_l117_117584

theorem allocation_schemes :
  (choose 7 3) * (nat.factorial 3) + (choose 7 2) * (choose 3 2) * 2 = 336 :=
by
  -- Case 1: All three people in different labs
  have case1 : (choose 7 3) * (nat.factorial 3) = 210 := by
    rw [nat.choose_succ_succ, nat.factorial_succ, nat.factorial_succ, nat.factorial_one]
    simp
  -- Case 2: Two people in one lab, one in another
  have case2 : (choose 7 2) * (choose 3 2) * 2 = 126 := by
    rw [nat.choose_succ_succ, nat.choose_succ_succ, nat.factorial_succ, nat.factorial_one]
    simp
  -- Sum of the two cases
  calc
    (choose 7 3) * (nat.factorial 3) + (choose 7 2) * (choose 3 2) * 2
        = 210 + 126           : by rw [case1, case2]
    ... = 336                : by norm_num

end allocation_schemes_l117_117584


namespace sin_cos_product_neg_l117_117849

theorem sin_cos_product_neg (α : ℝ) (h : Real.tan α < 0) : Real.sin α * Real.cos α < 0 :=
sorry

end sin_cos_product_neg_l117_117849


namespace sin_minus_cos_eq_l117_117377

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l117_117377


namespace remainder_x5_3x3_2x2_x_2_div_x_minus_2_l117_117205

def polynomial (x : ℝ) : ℝ := x^5 + 3*x^3 + 2*x^2 + x + 2

theorem remainder_x5_3x3_2x2_x_2_div_x_minus_2 :
  polynomial 2 = 68 := 
by 
  sorry

end remainder_x5_3x3_2x2_x_2_div_x_minus_2_l117_117205


namespace cost_of_one_hockey_stick_l117_117194

theorem cost_of_one_hockey_stick (x : ℝ)
    (h1 : x * 2 + 25 = 68) : x = 21.50 :=
by
  sorry

end cost_of_one_hockey_stick_l117_117194


namespace second_acid_solution_percentage_l117_117846

-- Definitions of the problem conditions
def P : ℝ := 75
def V₁ : ℝ := 4
def C₁ : ℝ := 0.60
def V₂ : ℝ := 20
def C₂ : ℝ := 0.72

/-
Given that 4 liters of a 60% acid solution are mixed with a certain volume of another acid solution
to get 20 liters of 72% solution, prove that the percentage of the second acid solution must be 75%.
-/
theorem second_acid_solution_percentage
  (x : ℝ) -- volume of the second acid solution
  (P_percent : ℝ := P) -- percentage of the second acid solution
  (h1 : V₁ + x = V₂) -- condition on volume
  (h2 : C₁ * V₁ + (P_percent / 100) * x = C₂ * V₂) -- condition on acid content
  : P_percent = P := 
by
  -- Moving forward with proof the lean proof
  sorry

end second_acid_solution_percentage_l117_117846


namespace meaningful_expression_range_l117_117557

theorem meaningful_expression_range (a : ℝ) :
  (∃ (x : ℝ), x = (sqrt (a + 1)) / (a - 2)) ↔ a ≥ -1 ∧ a ≠ 2 := 
begin
  sorry
end

end meaningful_expression_range_l117_117557


namespace slope_parallel_l117_117297

theorem slope_parallel (x y : ℝ) (m : ℝ) : (3:ℝ) * x - (6:ℝ) * y = (9:ℝ) → m = (1:ℝ) / (2:ℝ) :=
by
  sorry

end slope_parallel_l117_117297


namespace no_half_probability_socks_l117_117763

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l117_117763


namespace unique_real_solution_floor_eq_l117_117777

theorem unique_real_solution_floor_eq (k : ℕ) (h : k > 0) :
  ∃! x : ℝ, k ≤ x ∧ x < k + 1 ∧ ⌊x⌋ * (x^2 + 1) = x^3 :=
sorry

end unique_real_solution_floor_eq_l117_117777


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117389

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117389


namespace tan_alpha_eq_neg2_complex_expression_eq_neg5_l117_117661

variables (α : ℝ)
variables (h_sin : Real.sin α = - (2 * Real.sqrt 5) / 5)
variables (h_tan_neg : Real.tan α < 0)

theorem tan_alpha_eq_neg2 :
  Real.tan α = -2 :=
sorry

theorem complex_expression_eq_neg5 :
  (2 * Real.sin (α + Real.pi) + Real.cos (2 * Real.pi - α)) /
  (Real.cos (α - Real.pi / 2) - Real.sin (3 * Real.pi / 2 + α)) = -5 :=
sorry

end tan_alpha_eq_neg2_complex_expression_eq_neg5_l117_117661


namespace sin_minus_cos_theta_l117_117396

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l117_117396


namespace simplify_and_evaluate_evaluate_at_zero_l117_117130

noncomputable def simplified_expression (x : ℤ) : ℚ :=
  (1 - 1/(x-1)) / ((x^2 - 4*x + 4) / (x^2 - 1))

theorem simplify_and_evaluate (x : ℤ) (h : x ≠ 1 ∧ x ≠ 2 ∧ x ≠ -1) : 
  simplified_expression x = (x+1)/(x-2) :=
by
  sorry

theorem evaluate_at_zero : simplified_expression 0 = -1/2 :=
by
  sorry

end simplify_and_evaluate_evaluate_at_zero_l117_117130


namespace problem_U_l117_117243

theorem problem_U :
  ( (1 : ℝ) / (4 - Real.sqrt 15) - (1 / (Real.sqrt 15 - Real.sqrt 14))
  + (1 / (Real.sqrt 14 - 3)) - (1 / (3 - Real.sqrt 12))
  + (1 / (Real.sqrt 12 - Real.sqrt 11)) ) = 10 + Real.sqrt 11 :=
by
  sorry

end problem_U_l117_117243


namespace solution_set_of_even_function_l117_117842

theorem solution_set_of_even_function (f : ℝ → ℝ) (h_even : ∀ x, f (-x) = f x) 
  (h_def : ∀ x, 0 < x → f x = x^2 - 2*x - 3) : 
  { x : ℝ | f x > 0 } = { x | x > 3 } ∪ { x | x < -3 } :=
sorry

end solution_set_of_even_function_l117_117842


namespace todd_initial_money_l117_117469

-- Definitions of the conditions
def cost_per_candy_bar : ℕ := 2
def number_of_candy_bars : ℕ := 4
def money_left : ℕ := 12
def total_money_spent := number_of_candy_bars * cost_per_candy_bar

-- The statement proving the initial amount of money Todd had
theorem todd_initial_money : 
  (total_money_spent + money_left) = 20 :=
by
  sorry

end todd_initial_money_l117_117469


namespace roots_of_unity_real_root_l117_117890

theorem roots_of_unity_real_root (n : ℕ) (h_even : n % 2 = 0) : ∃ z : ℝ, z ≠ 1 ∧ z^n = 1 :=
by
  sorry

end roots_of_unity_real_root_l117_117890


namespace undefined_values_l117_117323

-- Define the expression to check undefined values
noncomputable def is_undefined (x : ℝ) : Prop :=
  x^3 - 9 * x = 0

-- Statement: For which real values of x is the expression undefined?
theorem undefined_values (x : ℝ) : is_undefined x ↔ x = 0 ∨ x = -3 ∨ x = 3 :=
sorry

end undefined_values_l117_117323


namespace sin_minus_cos_theta_l117_117395

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l117_117395


namespace bicycle_owners_no_car_l117_117563

-- Definitions based on the conditions in (a)
def total_adults : ℕ := 500
def bicycle_owners : ℕ := 450
def car_owners : ℕ := 120
def both_owners : ℕ := bicycle_owners + car_owners - total_adults

-- Proof problem statement
theorem bicycle_owners_no_car : (bicycle_owners - both_owners = 380) :=
by
  -- Placeholder proof
  sorry

end bicycle_owners_no_car_l117_117563


namespace cost_of_apple_is_two_l117_117721

-- Define the costs and quantities
def cost_of_apple (A : ℝ) : Prop :=
  let total_cost := 12 * A + 4 * 1 + 4 * 3
  let total_pieces := 12 + 4 + 4
  let average_cost := 2
  total_cost = total_pieces * average_cost

theorem cost_of_apple_is_two : cost_of_apple 2 :=
by
  -- Skipping the proof with sorry
  sorry

end cost_of_apple_is_two_l117_117721


namespace sum_of_all_possible_distinct_values_l117_117001

   noncomputable def sum_of_squares_of_triples (p q r : ℕ) : ℕ :=
     p^2 + q^2 + r^2

   theorem sum_of_all_possible_distinct_values (p q r : ℕ) (h1 : p + q + r = 30)
     (h2 : Nat.gcd p q + Nat.gcd q r + Nat.gcd r p = 10) : 
     sum_of_squares_of_triples p q r = 584 :=
   by
     sorry
   
end sum_of_all_possible_distinct_values_l117_117001


namespace quadratic_inequality_l117_117559

theorem quadratic_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by sorry

end quadratic_inequality_l117_117559


namespace team_a_faster_than_team_t_l117_117286

-- Definitions for the conditions
def course_length : ℕ := 300
def team_t_speed : ℕ := 20
def team_t_time : ℕ := course_length / team_t_speed
def team_a_time : ℕ := team_t_time - 3
def team_a_speed : ℕ := course_length / team_a_time

-- Theorem to prove
theorem team_a_faster_than_team_t :
  team_a_speed - team_t_speed = 5 :=
by
  -- Define the necessary elements based on conditions
  let course_length := 300
  let team_t_speed := 20
  let team_t_time := course_length / team_t_speed -- 15 hours
  let team_a_time := team_t_time - 3 -- 12 hours
  let team_a_speed := course_length / team_a_time -- 25 mph
  
  -- Prove the statement
  have h : team_a_speed - team_t_speed = 5 := by sorry
  exact h

end team_a_faster_than_team_t_l117_117286


namespace mike_disk_space_l117_117255

theorem mike_disk_space (F L T : ℕ) (hF : F = 26) (hL : L = 2) : T = 28 :=
by
  have h : T = F + L := by sorry
  rw [hF, hL] at h
  assumption

end mike_disk_space_l117_117255


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117390

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117390


namespace angle_bisector_relation_l117_117692

theorem angle_bisector_relation (a b : ℝ) (h : a = -b ∨ a = -b) : a = -b :=
sorry

end angle_bisector_relation_l117_117692


namespace water_polo_team_selection_l117_117451

theorem water_polo_team_selection :
  let total_players := 20
  let team_size := 9
  let goalies := 2
  let remaining_players := total_players - goalies
  let combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  combination total_players goalies * combination remaining_players (team_size - goalies) = 6046560 :=
by
  -- Definitions and calculations to be filled here.
  sorry

end water_polo_team_selection_l117_117451


namespace largest_integer_solution_l117_117693

theorem largest_integer_solution (m : ℤ) (h : 2 * m + 7 ≤ 3) : m ≤ -2 :=
sorry

end largest_integer_solution_l117_117693


namespace socks_impossible_l117_117759

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l117_117759


namespace legs_in_room_l117_117146

def total_legs_in_room (tables4 : Nat) (sofa : Nat) (chairs4 : Nat) (tables3 : Nat) (table1 : Nat) (rocking_chair2 : Nat) : Nat :=
  (tables4 * 4) + (sofa * 4) + (chairs4 * 4) + (tables3 * 3) + (table1 * 1) + (rocking_chair2 * 2)

theorem legs_in_room :
  total_legs_in_room 4 1 2 3 1 1 = 40 :=
by
  -- Skipping proof steps
  sorry

end legs_in_room_l117_117146


namespace shorter_side_of_quilt_l117_117153

theorem shorter_side_of_quilt :
  ∀ (x : ℕ), (∃ y : ℕ, 24 * y = 144) -> x = 6 :=
by
  intros x h
  sorry

end shorter_side_of_quilt_l117_117153


namespace average_speed_home_l117_117178

theorem average_speed_home
  (s_to_retreat : ℝ)
  (d_to_retreat : ℝ)
  (total_round_trip_time : ℝ)
  (t_retreat : d_to_retreat / s_to_retreat = 6)
  (t_total : d_to_retreat / s_to_retreat + 4 = total_round_trip_time) :
  (d_to_retreat / 4 = 75) :=
by
  sorry

end average_speed_home_l117_117178


namespace annika_total_east_hike_distance_l117_117056

def annika_flat_rate : ℝ := 10 -- minutes per kilometer on flat terrain
def annika_initial_distance : ℝ := 2.75 -- kilometers already hiked east
def total_time : ℝ := 45 -- minutes
def uphill_rate : ℝ := 15 -- minutes per kilometer on uphill
def downhill_rate : ℝ := 5 -- minutes per kilometer on downhill
def uphill_distance : ℝ := 0.5 -- kilometer of uphill section
def downhill_distance : ℝ := 0.5 -- kilometer of downhill section

theorem annika_total_east_hike_distance :
  let total_uphill_time := uphill_distance * uphill_rate
  let total_downhill_time := downhill_distance * downhill_rate
  let time_for_uphill_and_downhill := total_uphill_time + total_downhill_time
  let time_available_for_outward_hike := total_time / 2
  let remaining_time_after_up_down := time_available_for_outward_hike - time_for_uphill_and_downhill
  let additional_flat_distance := remaining_time_after_up_down / annika_flat_rate
  (annika_initial_distance + additional_flat_distance) = 4 :=
by
  sorry

end annika_total_east_hike_distance_l117_117056


namespace trig_identity_l117_117361

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l117_117361


namespace min_value_seq_div_n_l117_117226

-- Definitions of the conditions
def a_seq (n : ℕ) : ℕ := 
  if n = 0 then 0 else if n = 1 then 98 else 102 + (n - 2) * (2 * n + 2)

-- The property we need to prove
theorem min_value_seq_div_n :
  (∀ n : ℕ, (n ≥ 1) → (a_seq n / n) ≥ 26) ∧ (∃ n : ℕ, (n ≥ 1) ∧ (a_seq n / n) = 26) :=
sorry

end min_value_seq_div_n_l117_117226


namespace root_of_quadratic_eq_when_C_is_3_l117_117531

-- Define the quadratic equation and the roots we are trying to prove
def quadratic_eq (C : ℝ) (x : ℝ) := 3 * x^2 - 6 * x + C = 0

-- Set the constant C to 3
def C : ℝ := 3

-- State the theorem that proves the root of the equation when C=3 is x=1
theorem root_of_quadratic_eq_when_C_is_3 : quadratic_eq C 1 :=
by
  -- Skip the detailed proof
  sorry

end root_of_quadratic_eq_when_C_is_3_l117_117531


namespace probability_chord_length_not_less_than_radius_l117_117668

theorem probability_chord_length_not_less_than_radius
  (R : ℝ) (M N : ℝ) (h_circle : N = 2 * π * R) : 
  (∃ P : ℝ, P = 2 / 3) :=
sorry

end probability_chord_length_not_less_than_radius_l117_117668


namespace analytical_expression_of_f_range_of_m_l117_117228

open Real

section Problem1

-- Define the vectors and conditions
variables (ω x : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (f : ℝ → ℝ)
def a := (sin (ω * x), √3 * cos (2 * ω * x))
def b := ((1/2) * cos (ω * x), 1/4)
def f := λ x, (sin (ω * x) * (1 / 2) * cos (ω * x)) + (√3 * cos (2 * ω * x) * (1 / 4))

-- Given the conditions, prove the analytical expression of f(x)
theorem analytical_expression_of_f (ω_pos : ω > 0) (symmetry_distance : (2 * π) / (2 * ω) = π) :
  f(x) = (1/2) * sin(2 * x + π / 3) :=
sorry

end Problem1

section Problem2

-- Given the range condition, prove the range of m for f(x) = m has exactly two solutions
theorem range_of_m (m : ℝ) :
  m ∈ set.Ico (√3 / 4) (1 / 2) →
  ∀ x, 0 ≤ x ∧ x ≤ 7 * π / 12 →
  (∃! y, (f y = m ∧ 0 ≤ y ∧ y ≤ 7 * π / 12)) :=
sorry

end Problem2

end analytical_expression_of_f_range_of_m_l117_117228


namespace simplify_fraction_l117_117038

theorem simplify_fraction : (2 / (1 - (2 / 3))) = 6 :=
by
  sorry

end simplify_fraction_l117_117038


namespace arithmetic_progression_square_l117_117260

theorem arithmetic_progression_square (a b c : ℝ) (h : b - a = c - b) : a^2 + 8 * b * c = (2 * b + c)^2 := 
by
  sorry

end arithmetic_progression_square_l117_117260


namespace monotone_increasing_interval_l117_117011

noncomputable def f (x : ℝ) : ℝ := 2 * log x - x^2

theorem monotone_increasing_interval : { x : ℝ | 0 < x ∧ x < 1 } = { x : ℝ | 0 < x } ∩ { x : ℝ | f' x > 0 } :=
by
    sorry

end monotone_increasing_interval_l117_117011


namespace find_smaller_number_l117_117930

theorem find_smaller_number (L S : ℕ) (h1 : L - S = 2468) (h2 : L = 8 * S + 27) : S = 349 :=
by
  sorry

end find_smaller_number_l117_117930


namespace watermelon_juice_percentage_l117_117792

theorem watermelon_juice_percentage :
  ∀ (total_ounces orange_juice_percent grape_juice_ounces : ℕ), 
  orange_juice_percent = 25 →
  grape_juice_ounces = 70 →
  total_ounces = 200 →
  ((total_ounces - (orange_juice_percent * total_ounces / 100 + grape_juice_ounces)) / total_ounces) * 100 = 40 :=
by
  intros total_ounces orange_juice_percent grape_juice_ounces h1 h2 h3
  sorry

end watermelon_juice_percentage_l117_117792


namespace min_value_geometric_seq_l117_117665

theorem min_value_geometric_seq (a : ℕ → ℝ) (m n : ℕ) (h_pos : ∀ k, a k > 0)
  (h1 : a 1 = 1)
  (h2 : a 7 = a 6 + 2 * a 5)
  (h3 : a m * a n = 16) :
  (1 / m + 4 / n) ≥ 3 / 2 :=
sorry

end min_value_geometric_seq_l117_117665


namespace min_value_f_l117_117823

def f (x y z : ℝ) : ℝ := 
  x^2 + 4 * x * y + 3 * y^2 + 2 * z^2 - 8 * x - 4 * y + 6 * z

theorem min_value_f : ∃ (x y z : ℝ), f x y z = -13.5 :=
  by
  use 1, 1.5, -1.5
  sorry

end min_value_f_l117_117823


namespace maria_correct_result_l117_117251

-- Definitions of the conditions
def maria_incorrect_divide_multiply (x : ℤ) : ℤ := x / 9 - 20
def maria_final_after_errors := 8

-- Definitions of the correct operations
def maria_correct_multiply_add (x : ℤ) : ℤ := x * 9 + 20

-- The final theorem to prove
theorem maria_correct_result (x : ℤ) (h : maria_incorrect_divide_multiply x = maria_final_after_errors) :
  maria_correct_multiply_add x = 2288 :=
sorry

end maria_correct_result_l117_117251


namespace fraction_to_terminating_decimal_l117_117326

theorem fraction_to_terminating_decimal : (21 : ℚ) / 40 = 0.525 := 
by
  sorry

end fraction_to_terminating_decimal_l117_117326


namespace m_minus_t_value_l117_117304

-- Define the sum of squares of the odd integers from 1 to 215
def sum_squares_odds (n : ℕ) : ℕ := n * (4 * n^2 - 1) / 3

-- Define the sum of squares of the even integers from 2 to 100
def sum_squares_evens (n : ℕ) : ℕ := 2 * n * (n + 1) * (2 * n + 1) / 3

-- Number of odd terms from 1 to 215
def odd_terms_count : ℕ := (215 - 1) / 2 + 1

-- Number of even terms from 2 to 100
def even_terms_count : ℕ := (100 - 2) / 2 + 1

-- Define m and t
def m : ℕ := sum_squares_odds odd_terms_count
def t : ℕ := sum_squares_evens even_terms_count

-- Prove that m - t = 1507880
theorem m_minus_t_value : m - t = 1507880 :=
by
  -- calculations to verify the proof will be here, but are omitted for now
  sorry

end m_minus_t_value_l117_117304


namespace value_of_h_l117_117548

theorem value_of_h (h : ℤ) : (-1)^3 + h * (-1) - 20 = 0 → h = -21 :=
by
  intro h_cond
  sorry

end value_of_h_l117_117548


namespace original_garden_side_length_l117_117351

theorem original_garden_side_length (a : ℝ) (h : (a + 3)^2 = 2 * a^2 + 9) : a = 6 :=
by
  sorry

end original_garden_side_length_l117_117351


namespace renovation_services_are_credence_goods_and_choice_arguments_l117_117035

-- Define what credence goods are and the concept of information asymmetry
structure CredenceGood where
  information_asymmetry : Prop
  unobservable_quality  : Prop

-- Define renovation service as an instance of CredenceGood
def RenovationService : CredenceGood := {
  information_asymmetry := true,
  unobservable_quality := true
}

-- Primary conditions for choosing between construction company and private repair crew
structure ChoiceArgument where
  information_availability     : Prop
  warranty_and_accountability  : Prop
  higher_costs                 : Prop
  potential_bias_in_reviews    : Prop

-- Arguments for using construction company
def ConstructionCompanyArguments : ChoiceArgument := {
  information_availability := true,
  warranty_and_accountability := true,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Arguments against using construction company
def PrivateRepairCrewArguments : ChoiceArgument := {
  information_availability := false,
  warranty_and_accountability := false,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Proof statement to show renovation services are credence goods and economically reasoned arguments for/against
theorem renovation_services_are_credence_goods_and_choice_arguments:
  RenovationService = {
    information_asymmetry := true,
    unobservable_quality := true
  } ∧
  (ConstructionCompanyArguments.information_availability = true ∧
   ConstructionCompanyArguments.warranty_and_accountability = true) ∧
  (ConstructionCompanyArguments.higher_costs = true ∧
   ConstructionCompanyArguments.potential_bias_in_reviews = true) ∧
  (PrivateRepairCrewArguments.higher_costs = true ∧
   PrivateRepairCrewArguments.potential_bias_in_reviews = true) :=
by sorry

end renovation_services_are_credence_goods_and_choice_arguments_l117_117035


namespace tg_half_product_l117_117867

open Real

variable (α β : ℝ)

theorem tg_half_product (h1 : sin α + sin β = 2 * sin (α + β))
                        (h2 : ∀ n : ℤ, α + β ≠ 2 * π * n) :
  tan (α / 2) * tan (β / 2) = 1 / 3 := by
  sorry

end tg_half_product_l117_117867


namespace find_missing_number_l117_117599

theorem find_missing_number (x : ℝ) :
  (20 + 40 + 60) / 3 = (10 + x + 35) / 3 + 5 → x = 60 :=
by
  sorry

end find_missing_number_l117_117599


namespace factorial_division_l117_117961

open Nat

theorem factorial_division : 12! / 11! = 12 := sorry

end factorial_division_l117_117961


namespace max_and_min_sum_of_vars_l117_117216

theorem max_and_min_sum_of_vars (x y z w : ℝ) (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ 0 ≤ w)
  (eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17 / 2) :
  ∃ max min : ℝ, max = 3 ∧ min = -2 + 5 / 2 * Real.sqrt 2 ∧
  (∀ (S : ℝ), S = x + y + z + w → S ≤ max ∧ S ≥ min) :=
by sorry

end max_and_min_sum_of_vars_l117_117216


namespace find_multiple_l117_117886

theorem find_multiple (x y : ℕ) (h1 : x = 11) (h2 : x + y = 55) (h3 : ∃ k m : ℕ, y = k * x + m) :
  ∃ k : ℕ, y = k * x ∧ k = 4 :=
by
  sorry

end find_multiple_l117_117886


namespace work_b_alone_l117_117031

theorem work_b_alone (a b : ℕ) (h1 : 2 * b = a) (h2 : a + b = 3) (h3 : (a + b) * 11 = 33) : 33 = 33 :=
by
  -- sorry is used here because we are skipping the actual proof
  sorry

end work_b_alone_l117_117031


namespace geometric_sequence_sixth_term_l117_117825

theorem geometric_sequence_sixth_term (a b : ℚ) (h : a = 3 ∧ b = -1/2) : 
  (a * (b / a) ^ 5) = -1/2592 :=
by
  sorry

end geometric_sequence_sixth_term_l117_117825


namespace monotonically_increasing_interval_l117_117010

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x ^ 2

theorem monotonically_increasing_interval :
  ∀ x : ℝ, (0 < x ∧ x < 1) → (f x > f 0) := 
by
  sorry

end monotonically_increasing_interval_l117_117010


namespace reflection_line_equation_l117_117299

-- Given condition 1: Original line equation
def original_line (x : ℝ) : ℝ := -2 * x + 7

-- Given condition 2: Reflection line
def reflection_line_x : ℝ := 3

-- Proving statement
theorem reflection_line_equation
  (a b : ℝ)
  (h₁ : a = -(-2))
  (h₂ : original_line 3 = 1)
  (h₃ : 1 = a * 3 + b) :
  2 * a + b = -1 :=
  sorry

end reflection_line_equation_l117_117299


namespace number_of_girls_in_basketball_club_l117_117041

-- Define the number of members in the basketball club
def total_members : ℕ := 30

-- Define the number of members who attended the practice session
def attended : ℕ := 18

-- Define the unknowns: number of boys (B) and number of girls (G)
variables (B G : ℕ)

-- Define the conditions provided in the problem
def condition1 : Prop := B + G = total_members
def condition2 : Prop := B + (1 / 3) * G = attended

-- Define the theorem to prove
theorem number_of_girls_in_basketball_club (B G : ℕ) (h1 : condition1 B G) (h2 : condition2 B G) : G = 18 :=
sorry

end number_of_girls_in_basketball_club_l117_117041


namespace total_distance_l117_117122

theorem total_distance (x y : ℝ) (h1 : x * y = 18) :
  let D2 := (y - 1) * (x + 1)
  let D3 := 15
  let D_total := 18 + D2 + D3
  D_total = y * x + y - x + 32 :=
by
  let D2 := (y - 1) * (x + 1)
  let D3 := 15
  let D_total := 18 + D2 + D3
  sorry

end total_distance_l117_117122


namespace tails_at_least_twice_but_not_more_than_three_times_l117_117791

open ProbabilityTheory

/-- The event that "tails will be the result at least twice but not more than three times" 
when a fair coin is flipped three times has a probability of 1/2. -/
theorem tails_at_least_twice_but_not_more_than_three_times (s : Finset (Fin 2)) :
  (∑ x in s.filter (λ ω, (ω = 0) + (ω = 1) + (ω = 2)), 1 / (s.card : ℝ)) = 1 / 2 :=
begin
  sorry
end

end tails_at_least_twice_but_not_more_than_three_times_l117_117791


namespace no_intersection_of_graphs_l117_117847

theorem no_intersection_of_graphs :
  ∃ x y : ℝ, y = |3 * x + 6| ∧ y = -|4 * x - 3| → false := by
  sorry

end no_intersection_of_graphs_l117_117847


namespace tangent_line_at_neg1_l117_117602

-- Define the function given in the condition.
def f (x : ℝ) : ℝ := x^2 + 4 * x + 2

-- Define the point of tangency given in the condition.
def point_of_tangency : ℝ × ℝ := (-1, f (-1))

-- Define the derivative of the function.
def derivative_f (x : ℝ) : ℝ := 2 * x + 4

-- The proof statement: the equation of the tangent line at x = -1 is y = 2x + 1
theorem tangent_line_at_neg1 :
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = f x → derivative_f (-1) = m ∧ point_of_tangency.fst = -1 ∧ y = m * (x + 1) + b) :=
sorry

end tangent_line_at_neg1_l117_117602


namespace linear_function_mask_l117_117027

theorem linear_function_mask (x : ℝ) : ∃ k, k = 0.9 ∧ ∀ x, y = k * x :=
by
  sorry

end linear_function_mask_l117_117027


namespace profit_function_maximize_profit_l117_117052

def cost_per_item : ℝ := 80
def purchase_quantity : ℝ := 1000
def selling_price_initial : ℝ := 100
def price_increase_per_item : ℝ := 1
def sales_decrease_per_yuan : ℝ := 10
def selling_price (x : ℕ) : ℝ := selling_price_initial + x
def profit (x : ℕ) : ℝ := (selling_price x - cost_per_item) * (purchase_quantity - sales_decrease_per_yuan * x)

theorem profit_function (x : ℕ) (h : 0 ≤ x ∧ x ≤ 100) : 
  profit x = -10 * (x : ℝ)^2 + 800 * (x : ℝ) + 20000 :=
by sorry

theorem maximize_profit :
  ∃ max_x, (0 ≤ max_x ∧ max_x ≤ 100) ∧ 
  (∀ x : ℕ, (0 ≤ x ∧ x ≤ 100) → profit x ≤ profit max_x) ∧ 
  max_x = 40 ∧ 
  profit max_x = 36000 :=
by sorry

end profit_function_maximize_profit_l117_117052


namespace range_of_m_l117_117347

theorem range_of_m (m : ℝ) : 
  ((m - 1) * x^2 - 4 * x + 1 = 0) → 
  ((20 - 4 * m ≥ 0) ∧ (m ≠ 1)) :=
by
  sorry

end range_of_m_l117_117347


namespace main_l117_117664

theorem main (x y : ℤ) (h1 : abs x = 5) (h2 : abs y = 3) (h3 : x * y > 0) : 
    x - y = 2 ∨ x - y = -2 := sorry

end main_l117_117664


namespace ordinary_eq_of_curve_l117_117223

theorem ordinary_eq_of_curve 
  (t : ℝ) (x : ℝ) (y : ℝ)
  (ht : t > 0) 
  (hx : x = Real.sqrt t - 1 / Real.sqrt t)
  (hy : y = 3 * (t + 1 / t)) :
  3 * x^2 - y + 6 = 0 ∧ y ≥ 6 :=
sorry

end ordinary_eq_of_curve_l117_117223


namespace discount_of_bag_l117_117787

def discounted_price (marked_price discount_rate : ℕ) : ℕ :=
  marked_price - ((discount_rate * marked_price) / 100)

theorem discount_of_bag : discounted_price 200 40 = 120 :=
by
  unfold discounted_price
  norm_num

end discount_of_bag_l117_117787


namespace compound_interest_rate_l117_117144

theorem compound_interest_rate 
  (PV FV : ℝ) (n : ℕ) (r : ℝ) 
  (hPV : PV = 200) 
  (hFV : FV = 242) 
  (hn : n = 2) 
  (h_eq : PV = FV / (1 + r) ^ n) : 
  r = 0.1 := 
by
  sorry

end compound_interest_rate_l117_117144


namespace matrix_cube_l117_117955

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by
  sorry

end matrix_cube_l117_117955


namespace compute_four_at_seven_l117_117521

def operation (a b : ℤ) : ℤ :=
  5 * a - 2 * b

theorem compute_four_at_seven : operation 4 7 = 6 :=
by
  sorry

end compute_four_at_seven_l117_117521


namespace hyperbola_foci_distance_l117_117819

theorem hyperbola_foci_distance :
  let a := Real.sqrt 25
  let b := Real.sqrt 9
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let distance := 2 * c
  distance = 2 * Real.sqrt 34 :=
by
  let a := Real.sqrt 25
  let b := Real.sqrt 9
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let distance := 2 * c
  exact sorry

end hyperbola_foci_distance_l117_117819


namespace trapezium_side_length_l117_117070

variable (length1 length2 height area : ℕ)

theorem trapezium_side_length
  (h1 : length1 = 20)
  (h2 : height = 15)
  (h3 : area = 270)
  (h4 : area = (length1 + length2) * height / 2) :
  length2 = 16 :=
by
  sorry

end trapezium_side_length_l117_117070


namespace tree_height_at_2_years_l117_117505

theorem tree_height_at_2_years (h : ℕ → ℚ) (h5 : h 5 = 243) 
  (h_rec : ∀ n, h (n - 1) = h n / 3) : h 2 = 9 :=
  sorry

end tree_height_at_2_years_l117_117505


namespace minimize_frac_inv_l117_117980

theorem minimize_frac_inv (a b : ℕ) (h1: 4 * a + b = 30) (h2: a > 0) (h3: b > 0) :
  (a, b) = (5, 10) :=
sorry

end minimize_frac_inv_l117_117980


namespace part1_part2_l117_117985

theorem part1 (a : ℝ) (h : a * (-a)^2 + (a - 1) * (-a) - 1 > 0) : a > 1 :=
sorry

noncomputable def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then {x : ℝ | x < -1}
  else if a = -1 then ∅
  else if a > 0 then {x : ℝ | x < -1} ∪ {x : ℝ | x > 1 / a}
  else if a < -1 then {x : ℝ | -1 < x ∧ x < 1 / a}
  else {x : ℝ | 1 / a < x ∧ x < -1}

theorem part2 (a : ℝ) :
  ∀ x : ℝ, (a * x^2 + (a - 1) * x - 1 > 0) ↔ (x ∈ (solution_set a)) :=
sorry

end part1_part2_l117_117985


namespace oranges_in_bin_l117_117036

theorem oranges_in_bin (initial_oranges : ℕ) (oranges_thrown_away : ℕ) (oranges_added : ℕ) 
  (h1 : initial_oranges = 50) (h2 : oranges_thrown_away = 40) (h3 : oranges_added = 24) 
  : initial_oranges - oranges_thrown_away + oranges_added = 34 := 
by
  -- Simplification and calculation here
  sorry

end oranges_in_bin_l117_117036


namespace triangle_angle_A_triangle_side_a_l117_117106

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h1 : (a - c)*(a + c)*sin C = c * (b - c) * sin B)
variable (area : ℝ)
variable (h2 : area = sqrt 3)
variable (h3 : sin B * sin C = 1/4)

theorem triangle_angle_A (h1 : (a - c)*(a + c)*sin C = c * (b - c) * sin B) : A = π / 3 := 
sorry

theorem triangle_side_a (h1 : (a - c)*(a + c)*sin C = c * (b - c) * sin B)
                        (h2 : area = sqrt 3)
                        (h3 : sin B * sin C = 1/4) : a = 2 * sqrt 3 := 
sorry

end triangle_angle_A_triangle_side_a_l117_117106


namespace no_non_trivial_solutions_l117_117203

theorem no_non_trivial_solutions (x y z : ℤ) :
  3 * x^2 + 7 * y^2 = z^4 → x = 0 ∧ y = 0 ∧ z = 0 :=
by
  intro h
  -- Proof goes here
  sorry

end no_non_trivial_solutions_l117_117203


namespace condition_a_neither_necessary_nor_sufficient_for_b_l117_117982

theorem condition_a_neither_necessary_nor_sufficient_for_b {x y : ℝ} (h : ¬(x = 1 ∧ y = 2)) (k : ¬(x + y = 3)) : ¬((x ≠ 1 ∧ y ≠ 2) ↔ (x + y ≠ 3)) :=
by
  sorry

end condition_a_neither_necessary_nor_sufficient_for_b_l117_117982


namespace fraction_inequality_l117_117832

theorem fraction_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) :
  b / (a - c) < a / (b - d) :=
sorry

end fraction_inequality_l117_117832


namespace no_possible_blue_socks_l117_117751

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l117_117751


namespace greatest_integer_difference_l117_117097

theorem greatest_integer_difference (x y : ℤ) (hx : 3 < x) (hx2 : x < 6) (hy : 6 < y) (hy2 : y < 10) :
  ∃ d : ℤ, d = y - x ∧ d = 5 :=
by
  sorry

end greatest_integer_difference_l117_117097


namespace max_and_min_sum_of_vars_l117_117215

theorem max_and_min_sum_of_vars (x y z w : ℝ) (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ 0 ≤ w)
  (eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17 / 2) :
  ∃ max min : ℝ, max = 3 ∧ min = -2 + 5 / 2 * Real.sqrt 2 ∧
  (∀ (S : ℝ), S = x + y + z + w → S ≤ max ∧ S ≥ min) :=
by sorry

end max_and_min_sum_of_vars_l117_117215


namespace problem_statement_l117_117828

noncomputable def a := Real.log 2 / Real.log 14
noncomputable def b := Real.log 2 / Real.log 7
noncomputable def c := Real.log 2 / Real.log 4

theorem problem_statement : (1 / a - 1 / b + 1 / c) = 3 := by
  sorry

end problem_statement_l117_117828


namespace students_behind_yoongi_l117_117151

theorem students_behind_yoongi (n k : ℕ) (hn : n = 30) (hk : k = 20) : n - (k + 1) = 9 := by
  sorry

end students_behind_yoongi_l117_117151


namespace exists_nat_square_starting_with_digits_l117_117129

theorem exists_nat_square_starting_with_digits (S : ℕ) : 
  ∃ (N k : ℕ), S * 10^k ≤ N^2 ∧ N^2 < (S + 1) * 10^k := 
by {
  sorry
}

end exists_nat_square_starting_with_digits_l117_117129


namespace simplify_expression_l117_117892

-- Define a variable x
variable (x : ℕ)

-- Statement of the problem
theorem simplify_expression : 120 * x - 75 * x = 45 * x := sorry

end simplify_expression_l117_117892


namespace units_digit_7_pow_l117_117921

theorem units_digit_7_pow (n : ℕ) : 
  ∃ k, 7^n % 10 = k ∧ ((7^1 % 10 = 7) ∧ (7^2 % 10 = 9) ∧ (7^3 % 10 = 3) ∧ (7^4 % 10 = 1) ∧ (7^5 % 10 = 7)) → 
  7^2010 % 10 = 9 :=
by
  sorry

end units_digit_7_pow_l117_117921


namespace arithmetic_sequence_sum_l117_117565

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + d)
    (h_a5 : a 5 = 3)
    (h_a6 : a 6 = -2) :
  a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -49 :=
by
  sorry

end arithmetic_sequence_sum_l117_117565


namespace fifth_term_of_arithmetic_sequence_is_minus_three_l117_117461

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

theorem fifth_term_of_arithmetic_sequence_is_minus_three (a d : ℤ) :
  (arithmetic_sequence a d 11 = 25) ∧ (arithmetic_sequence a d 12 = 29) →
  (arithmetic_sequence a d 4 = -3) :=
by 
  intros h
  sorry

end fifth_term_of_arithmetic_sequence_is_minus_three_l117_117461


namespace combined_cost_price_l117_117293

theorem combined_cost_price :
  let face_value_A : ℝ := 100
  let discount_A : ℝ := 2
  let purchase_price_A := face_value_A - (discount_A / 100 * face_value_A)
  let brokerage_A := 0.2 / 100 * purchase_price_A
  let total_cost_price_A := purchase_price_A + brokerage_A

  let face_value_B : ℝ := 100
  let premium_B : ℝ := 1.5
  let purchase_price_B := face_value_B + (premium_B / 100 * face_value_B)
  let brokerage_B := 0.2 / 100 * purchase_price_B
  let total_cost_price_B := purchase_price_B + brokerage_B

  let combined_cost_price := total_cost_price_A + total_cost_price_B

  combined_cost_price = 199.899 := by
  sorry

end combined_cost_price_l117_117293


namespace smallest_n_satisfying_conditions_l117_117472

theorem smallest_n_satisfying_conditions : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ x : ℕ, 3 * n = x^4) ∧ (∃ y : ℕ, 2 * n = y^5) ∧ n = 432 :=
by
  sorry

end smallest_n_satisfying_conditions_l117_117472


namespace sin_minus_cos_eq_l117_117375

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l117_117375


namespace locus_eqn_l117_117004

noncomputable def locus_of_centers (a b : ℝ) : Prop :=
  ∃ r : ℝ, 
    (a^2 + b^2 = (r + 1)^2) ∧ 
    ((a - 2)^2 + b^2 = (5 - r)^2)

theorem locus_eqn (a b : ℝ) : 
  locus_of_centers a b ↔ 3 * a^2 + b^2 + 44 * a + 121 = 0 :=
by
  -- Proof omitted
  sorry

end locus_eqn_l117_117004


namespace stack_map_A_front_view_l117_117519

def column1 : List ℕ := [3, 1]
def column2 : List ℕ := [2, 2, 1]
def column3 : List ℕ := [1, 4, 2]
def column4 : List ℕ := [5]

def tallest (l : List ℕ) : ℕ :=
  l.foldl max 0

theorem stack_map_A_front_view :
  [tallest column1, tallest column2, tallest column3, tallest column4] = [3, 2, 4, 5] := by
  sorry

end stack_map_A_front_view_l117_117519


namespace volume_of_resulting_solid_is_9_l117_117532

-- Defining the initial cube with edge length 3
def initial_cube_edge_length : ℝ := 3

-- Defining the volume of the initial cube
def initial_cube_volume : ℝ := initial_cube_edge_length^3

-- Defining the volume of the resulting solid after some parts are cut off
def resulting_solid_volume : ℝ := 9

-- Theorem stating that given the initial conditions, the volume of the resulting solid is 9
theorem volume_of_resulting_solid_is_9 : resulting_solid_volume = 9 :=
by
  sorry

end volume_of_resulting_solid_is_9_l117_117532


namespace groups_of_men_and_women_l117_117433

theorem groups_of_men_and_women :
  let men := 4 in
  let women := 5 in
  let group1 := 2 in
  let group2 := 3 in
  let group3 := 4 in
  (((nat.choose men 1) * (nat.choose women 1)) * 
   ((nat.choose (men - 1) 1) * (nat.choose (women - 1) 2)) *
   ((nat.choose (men - 2) 2) * (nat.choose (women - 3) 2))) = 360 :=
by
  sorry

end groups_of_men_and_women_l117_117433


namespace f_100_eq_11_l117_117120
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

def f (n : ℕ) : ℕ := sum_of_digits (n^2 + 1)

def f_iter : ℕ → ℕ → ℕ
| 0,    n => f n
| k+1,  n => f (f_iter k n)

theorem f_100_eq_11 (n : ℕ) (h : n = 1990) : f_iter 100 n = 11 := by
  sorry

end f_100_eq_11_l117_117120


namespace f_fixed_point_l117_117579

-- Definitions and conditions based on the problem statement
def g (n : ℕ) : ℕ := sorry
def f (n : ℕ) : ℕ := sorry

-- Helper functions for the repeated application of f
noncomputable def f_iter (n x : ℕ) : ℕ := 
    Nat.iterate f (x^2023) n

axiom g_bijective : Function.Bijective g
axiom f_repeated : ∀ x : ℕ, f_iter x x = x
axiom f_div_g : ∀ (x y : ℕ), x ∣ y → f x ∣ g y

-- Main theorem statement
theorem f_fixed_point : ∀ x : ℕ, f x = x := by
  sorry

end f_fixed_point_l117_117579


namespace how_many_bananas_l117_117253

theorem how_many_bananas (total_fruit apples oranges : ℕ) 
  (h_total : total_fruit = 12) (h_apples : apples = 3) (h_oranges : oranges = 5) :
  total_fruit - apples - oranges = 4 :=
by
  sorry

end how_many_bananas_l117_117253


namespace find_x_plus_y_l117_117975

theorem find_x_plus_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : |x - y| = y - x) :
  (x + y = -1) ∨ (x + y = -5) :=
sorry

end find_x_plus_y_l117_117975


namespace acute_angle_probability_correct_l117_117440

noncomputable def acute_angle_probability (n : ℕ) (n_ge_4 : n ≥ 4) : ℝ :=
  (n * (n - 2)) / (2 ^ (n-1))

theorem acute_angle_probability_correct (n : ℕ) (h : n ≥ 4) (P : Fin n → ℝ) -- P represents points on the circle
    (uniformly_distributed : ∀ i, P i ∈ Set.Icc (0 : ℝ) 1) : 
    acute_angle_probability n h = (n * (n - 2)) / (2 ^ (n-1)) := 
  sorry

end acute_angle_probability_correct_l117_117440


namespace total_points_after_3_perfect_games_l117_117633

def perfect_score := 21
def number_of_games := 3

theorem total_points_after_3_perfect_games : perfect_score * number_of_games = 63 := 
by
  sorry

end total_points_after_3_perfect_games_l117_117633


namespace tangent_lines_through_P_l117_117545

noncomputable def curve_eq (x : ℝ) : ℝ := 1/3 * x^3 + 4/3

theorem tangent_lines_through_P (x y : ℝ) :
  ((4 * x - y - 4 = 0 ∨ y = x + 2) ∧ (curve_eq 2 = 4)) :=
by
  sorry

end tangent_lines_through_P_l117_117545


namespace sin_minus_cos_l117_117383

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l117_117383


namespace solve_for_x_l117_117733

theorem solve_for_x (x : ℚ) (h : (x - 3) / (x + 2) + (3 * x - 9) / (x - 3) = 2) : x = 1 / 2 :=
by
  sorry

end solve_for_x_l117_117733


namespace trigonometric_identity_l117_117408

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l117_117408


namespace last_three_digits_of_11_pow_210_l117_117330

theorem last_three_digits_of_11_pow_210 : (11 ^ 210) % 1000 = 601 :=
by sorry

end last_three_digits_of_11_pow_210_l117_117330


namespace distribution_count_l117_117848

-- Making the function for counting the number of valid distributions
noncomputable def countValidDistributions : ℕ :=
  let cases1 := 4                            -- One box contains all five balls
  let cases2 := 4 * 3                        -- One box has 4 balls, another has 1
  let cases3 := 4 * 3                        -- One box has 3 balls, another has 2
  let cases4 := 6 * 2                        -- Two boxes have 2 balls, and one has 1
  let cases5 := 4 * 3                        -- One box has 3 balls, and two boxes have 1 each
  cases1 + cases2 + cases3 + cases4 + cases5 -- Sum of all cases

-- Theorem statement: the count of valid distributions equals 52
theorem distribution_count : countValidDistributions = 52 := 
  by
    sorry

end distribution_count_l117_117848


namespace even_decreasing_function_l117_117581

theorem even_decreasing_function (f : ℝ → ℝ) (x1 x2 : ℝ)
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y → x < 0 → y < 0 → f y < f x)
  (hx1_neg : x1 < 0)
  (hx1x2_pos : x1 + x2 > 0) :
  f x1 < f x2 :=
sorry

end even_decreasing_function_l117_117581


namespace arithmetic_series_sum_l117_117945

theorem arithmetic_series_sum :
  let a1 : ℚ := 22
  let d : ℚ := 3 / 7
  let an : ℚ := 73
  let n := (an - a1) / d + 1
  let S := n * (a1 + an) / 2
  S = 5700 := by
  sorry

end arithmetic_series_sum_l117_117945


namespace factorize_one_factorize_two_l117_117966

variable (x a b : ℝ)

-- Problem 1: Prove that 4x^2 - 64 = 4(x + 4)(x - 4)
theorem factorize_one : 4 * x^2 - 64 = 4 * (x + 4) * (x - 4) :=
sorry

-- Problem 2: Prove that 4ab^2 - 4a^2b - b^3 = -b(2a - b)^2
theorem factorize_two : 4 * a * b^2 - 4 * a^2 * b - b^3 = -b * (2 * a - b)^2 :=
sorry

end factorize_one_factorize_two_l117_117966


namespace simplify_fraction_l117_117894

theorem simplify_fraction (x y z : ℝ) (hx : x = 5) (hz : z = 2) : (10 * x * y * z) / (15 * x^2 * z) = (2 * y) / 15 :=
by
  sorry

end simplify_fraction_l117_117894


namespace object_speed_l117_117778

namespace problem

noncomputable def speed_in_miles_per_hour (distance_in_feet : ℕ) (time_in_seconds : ℕ) : ℝ :=
  let distance_in_miles := distance_in_feet / 5280
  let time_in_hours := time_in_seconds / 3600
  distance_in_miles / time_in_hours

theorem object_speed 
  (distance_in_feet : ℕ)
  (time_in_seconds : ℕ)
  (h : distance_in_feet = 80 ∧ time_in_seconds = 2) :
  speed_in_miles_per_hour distance_in_feet time_in_seconds = 27.27 :=
by
  sorry

end problem

end object_speed_l117_117778


namespace range_s_l117_117322

def s (n : ℕ) : ℕ :=
  if n = 1 then 0
  else primeDivisors n |>.sum

theorem range_s : {y | ∃ n : ℕ, n > 1 ∧ s n = y} = {y | y ≥ 2} :=
by
  sorry

end range_s_l117_117322


namespace total_legs_in_room_l117_117148

def count_legs : Nat :=
  let tables_4_legs := 4 * 4
  let sofas_legs := 1 * 4
  let chairs_4_legs := 2 * 4
  let tables_3_legs := 3 * 3
  let tables_1_leg := 1 * 1
  let rocking_chair_legs := 1 * 2
  tables_4_legs + sofas_legs + chairs_4_legs + tables_3_legs + tables_1_leg + rocking_chair_legs

theorem total_legs_in_room : count_legs = 40 := by
  sorry

end total_legs_in_room_l117_117148


namespace quadratic_function_relation_l117_117854

theorem quadratic_function_relation 
  (y : ℝ → ℝ) 
  (y_def : ∀ x : ℝ, y x = x^2 + x + 1) 
  (y1 y2 y3 : ℝ) 
  (hA : y (-3) = y1) 
  (hB : y 2 = y2) 
  (hC : y (1/2) = y3) : 
  y3 < y1 ∧ y1 = y2 := 
sorry

end quadratic_function_relation_l117_117854


namespace sufficient_but_not_necessary_condition_l117_117977

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3
def g (k x : ℝ) : ℝ := k * x - 1

theorem sufficient_but_not_necessary_condition (k : ℝ) :
  (∀ x : ℝ, f x ≥ g k x) ↔ (-6 ≤ k ∧ k ≤ 2) :=
sorry

end sufficient_but_not_necessary_condition_l117_117977


namespace prime_factors_and_divisors_6440_l117_117529

theorem prime_factors_and_divisors_6440 :
  ∃ (a b c d : ℕ), 6440 = 2^a * 5^b * 7^c * 23^d ∧ a = 3 ∧ b = 1 ∧ c = 1 ∧ d = 1 ∧
  (a + 1) * (b + 1) * (c + 1) * (d + 1) = 32 :=
by 
  sorry

end prime_factors_and_divisors_6440_l117_117529


namespace joe_first_lift_weight_l117_117700

theorem joe_first_lift_weight (x y : ℕ) 
  (h1 : x + y = 900)
  (h2 : 2 * x = y + 300) :
  x = 400 :=
by
  sorry

end joe_first_lift_weight_l117_117700


namespace will_net_calorie_intake_is_600_l117_117166

-- Given conditions translated into Lean definitions and assumptions
def breakfast_calories : ℕ := 900
def jogging_time_minutes : ℕ := 30
def calories_burned_per_minute : ℕ := 10

-- Proof statement in Lean
theorem will_net_calorie_intake_is_600 :
  breakfast_calories - (jogging_time_minutes * calories_burned_per_minute) = 600 :=
by
  sorry

end will_net_calorie_intake_is_600_l117_117166


namespace given_condition_implies_result_l117_117551

theorem given_condition_implies_result (a : ℝ) (h : a ^ 2 + 2 * a = 1) : 2 * a ^ 2 + 4 * a + 1 = 3 :=
sorry

end given_condition_implies_result_l117_117551


namespace smallest_integer_l117_117024

theorem smallest_integer (M : ℕ) :
  (M % 4 = 3) ∧ (M % 5 = 4) ∧ (M % 6 = 5) ∧ (M % 7 = 6) ∧
  (M % 8 = 7) ∧ (M % 9 = 8) → M = 2519 :=
by sorry

end smallest_integer_l117_117024


namespace probability_consecutive_cards_l117_117282

open Finset

theorem probability_consecutive_cards : ∑ s in (filter (λ s : Finset ℕ, (s.card = 2 ∧ ∃ a b, (a = 1 ∧ b = 2) ∨ (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 5)) (powerset (range 6))), 1) / ∑ s in (filter (λ s : Finset ℕ, s.card = 2) (powerset (range 6))), 1 = 0.4 :=
sorry

end probability_consecutive_cards_l117_117282


namespace socks_impossible_l117_117758

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l117_117758


namespace digit_58_of_fraction_l117_117291

theorem digit_58_of_fraction (n : ℕ) (h1 : n = 58) : 
  ∀ {d : ℕ}, 
    let repr := ("0588235294117647".foldr (λ c acc, (↑(c.toNat - '0'.toNat) :: acc)) []),
        digits := (List.cycle repr) in
    digits.get? (n - 1) = some d → d = 4 := 
by
  intro d repr digits hd
  sorry

end digit_58_of_fraction_l117_117291


namespace sin_minus_cos_eq_l117_117379

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l117_117379


namespace cos_identity_l117_117540

theorem cos_identity (α : ℝ) (h : Real.cos (π / 6 + α) = sqrt 3 / 3) : 
  Real.cos (5 * π / 6 - α) = - (sqrt 3 / 3) :=
by
  sorry

end cos_identity_l117_117540


namespace parabola_property_l117_117083

-- Define the conditions of the problem in Lean
variable (a b : ℝ)
variable (h1 : (a, b) ∈ {p : ℝ × ℝ | p.1^2 = 20 * p.2}) -- P lies on the parabola x^2 = 20y
variable (h2 : dist (a, b) (0, 5) = 25) -- Distance from P to focus F

theorem parabola_property : |a * b| = 400 := by
  sorry

end parabola_property_l117_117083


namespace joe_total_toy_cars_l117_117583

def initial_toy_cars : ℕ := 50
def uncle_additional_factor : ℝ := 1.5

theorem joe_total_toy_cars :
  (initial_toy_cars : ℝ) + uncle_additional_factor * initial_toy_cars = 125 := 
by
  sorry

end joe_total_toy_cars_l117_117583


namespace average_growth_rate_estimated_export_2023_l117_117884

theorem average_growth_rate (export_2020 export_2022 : ℕ) (h1 : export_2020 = 200000) (h2 : export_2022 = 450000) :
  ∃ (x : ℝ), x = 0.5 ∧ export_2022 = export_2020 * (1 + x)^2 :=
by 
-- Proof required.
sorry

theorem estimated_export_2023 (export_2022 : ℕ) (x : ℝ) (h1 : export_2022 = 450000) (h2 : x = 0.5) :
  let export_2023 := export_2022 * (1 + x) in
  export_2023 = 675000 :=
by
-- Proof required.
sorry

end average_growth_rate_estimated_export_2023_l117_117884


namespace symmetric_points_origin_l117_117838

theorem symmetric_points_origin (a b : ℝ) 
  (h1 : (-2, b) = (-a, -3)) : a - b = 5 := 
by
  -- solution steps are not included in the statement
  sorry

end symmetric_points_origin_l117_117838


namespace neznaika_is_wrong_l117_117300

theorem neznaika_is_wrong (avg_december avg_january : ℝ)
  (h_avg_dec : avg_december = 10)
  (h_avg_jan : avg_january = 5) : 
  ∃ (dec_days jan_days : ℕ), 
    (avg_december = (dec_days * 10 + (31 - dec_days) * 0) / 31) ∧
    (avg_january = (jan_days * 10 + (31 - jan_days) * 0) / 31) ∧
    jan_days > dec_days :=
by 
  sorry

end neznaika_is_wrong_l117_117300


namespace rides_on_roller_coaster_l117_117141

-- Definitions based on the conditions given.
def roller_coaster_cost : ℕ := 17
def total_tickets : ℕ := 255
def tickets_spent_on_other_activities : ℕ := 78

-- The proof statement.
theorem rides_on_roller_coaster : (total_tickets - tickets_spent_on_other_activities) / roller_coaster_cost = 10 :=
by 
  sorry

end rides_on_roller_coaster_l117_117141


namespace axis_of_symmetry_l117_117821

-- Define the condition for the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  x = -4 * y^2

-- Define the statement that needs to be proven
theorem axis_of_symmetry (x : ℝ) (y : ℝ) (h : parabola_equation x y) : x = 1 / 16 :=
  sorry

end axis_of_symmetry_l117_117821


namespace circle_diameter_of_circumscribed_square_l117_117499

theorem circle_diameter_of_circumscribed_square (r : ℝ) (s : ℝ) (h1 : s = 2 * r) (h2 : 4 * s = π * r^2) : 2 * r = 16 / π := by
  sorry

end circle_diameter_of_circumscribed_square_l117_117499


namespace abs_difference_of_numbers_l117_117610

theorem abs_difference_of_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 391) :
  |x - y| = 6 :=
sorry

end abs_difference_of_numbers_l117_117610


namespace find_num_apples_l117_117167

def num_apples (A P : ℕ) : Prop :=
  P = (3 * A) / 5 ∧ A + P = 240

theorem find_num_apples (A : ℕ) (P : ℕ) :
  num_apples A P → A = 150 :=
by
  intros h
  -- sorry for proof
  sorry

end find_num_apples_l117_117167


namespace arithmetic_sequence_fifth_term_l117_117567

theorem arithmetic_sequence_fifth_term (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 6) (h3 : a 3 = 2) (h_arith_seq : ∀ n, a (n + 1) = a n + d) : a 5 = -2 :=
sorry

end arithmetic_sequence_fifth_term_l117_117567


namespace equation_of_line_containing_chord_l117_117238

theorem equation_of_line_containing_chord (x y : ℝ) : 
  (y^2 = -8 * x) ∧ ((-1, 1) = ((x + x) / 2, (y + y) / 2)) →
  4 * x + y + 3 = 0 :=
by 
  sorry

end equation_of_line_containing_chord_l117_117238


namespace car_z_mpg_decrease_l117_117946

theorem car_z_mpg_decrease :
  let mpg_45 := 51
  let mpg_60 := 408 / 10
  let decrease := mpg_45 - mpg_60
  let percentage_decrease := (decrease / mpg_45) * 100
  percentage_decrease = 20 := by
  sorry

end car_z_mpg_decrease_l117_117946


namespace ripe_mangoes_remaining_l117_117488

theorem ripe_mangoes_remaining
  (initial_mangoes : ℕ)
  (ripe_fraction : ℚ)
  (consume_fraction : ℚ)
  (initial_total : initial_mangoes = 400)
  (ripe_ratio : ripe_fraction = 3 / 5)
  (consume_ratio : consume_fraction = 60 / 100) :
  (initial_mangoes * ripe_fraction - initial_mangoes * ripe_fraction * consume_fraction) = 96 :=
by
  sorry

end ripe_mangoes_remaining_l117_117488


namespace exp_problem_l117_117523

theorem exp_problem (a b c : ℕ) (H1 : a = 1000) (H2 : b = 1000^1000) (H3 : c = 500^1000) :
  a * b / c = 2^1001 * 500 :=
sorry

end exp_problem_l117_117523


namespace distinct_ordered_pairs_eq_49_l117_117345

open Nat

theorem distinct_ordered_pairs_eq_49 (a b : ℕ) (h1 : a + b = 50) (h2 : a > 0) (h3 : b > 0) :
  num_solutions (λ p : ℕ × ℕ, p.1 + p.2 = 50 ∧ p.1 > 0 ∧ p.2 > 0) = 49 :=
sorry

end distinct_ordered_pairs_eq_49_l117_117345


namespace nested_geometric_sum_l117_117195

theorem nested_geometric_sum :
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4))))))))) = 1398100 :=
by
  sorry

end nested_geometric_sum_l117_117195


namespace no_blue_socks_make_probability_half_l117_117746

theorem no_blue_socks_make_probability_half :
  ∀ (n m : ℕ), n + m = 2009 → (n - m) * (n - m) ≠ 2009 :=
begin
  intros n m h,
  by_contra H,
  sorry, -- Proof goes here
end

end no_blue_socks_make_probability_half_l117_117746


namespace total_distance_is_correct_l117_117933

noncomputable def boat_speed : ℝ := 20 -- boat speed in still water (km/hr)
noncomputable def current_speed_first : ℝ := 5 -- current speed for the first 6 minutes (km/hr)
noncomputable def current_speed_second : ℝ := 8 -- current speed for the next 6 minutes (km/hr)
noncomputable def current_speed_third : ℝ := 3 -- current speed for the last 6 minutes (km/hr)
noncomputable def time_in_hours : ℝ := 6 / 60 -- 6 minutes in hours (0.1 hours)

noncomputable def total_distance_downstream := 
  (boat_speed + current_speed_first) * time_in_hours +
  (boat_speed + current_speed_second) * time_in_hours +
  (boat_speed + current_speed_third) * time_in_hours

theorem total_distance_is_correct : total_distance_downstream = 7.6 :=
  by 
  sorry

end total_distance_is_correct_l117_117933


namespace mean_days_correct_l117_117427

noncomputable def mean_days (a1 a2 a3 a4 a5 d1 d2 d3 d4 d5 : ℕ) : ℚ :=
  (a1 * d1 + a2 * d2 + a3 * d3 + a4 * d4 + a5 * d5 : ℚ) / (a1 + a2 + a3 + a4 + a5)

theorem mean_days_correct : mean_days 2 4 5 7 4 1 2 4 5 6 = 4.05 := by
  sorry

end mean_days_correct_l117_117427


namespace number_of_types_of_sliced_meat_l117_117209

-- Define the constants and conditions
def varietyPackCostWithoutRush := 40.00
def rushDeliveryPercentage := 0.30
def costPerTypeWithRush := 13.00
def totalCostWithRush := varietyPackCostWithoutRush + (rushDeliveryPercentage * varietyPackCostWithoutRush)

-- Define the statement that needs to be proven
theorem number_of_types_of_sliced_meat :
  (totalCostWithRush / costPerTypeWithRush) = 4 := by
  sorry

end number_of_types_of_sliced_meat_l117_117209


namespace sin_minus_cos_l117_117371

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l117_117371


namespace abc_inequality_l117_117878

theorem abc_inequality (a b c : ℝ) (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1) (h4 : a^3 + b^3 + c^3 = 1) : 
  a + b + c + a^2 + b^2 + c^2 ≤ 4 := 
sorry

end abc_inequality_l117_117878


namespace sum_three_positive_numbers_ge_three_l117_117718

theorem sum_three_positive_numbers_ge_three 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_eq : (a + 1) * (b + 1) * (c + 1) = 8) : 
  a + b + c ≥ 3 :=
sorry

end sum_three_positive_numbers_ge_three_l117_117718


namespace legos_in_box_at_end_l117_117575

def initial_legos : ℕ := 500
def legos_used : ℕ := initial_legos / 2
def missing_legos : ℕ := 5
def remaining_legos := legos_used - missing_legos

theorem legos_in_box_at_end : remaining_legos = 245 := 
by
  sorry

end legos_in_box_at_end_l117_117575


namespace count_of_integer_values_not_satisfying_inequality_l117_117076

theorem count_of_integer_values_not_satisfying_inequality :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, (3 * x^2 + 11 * x + 10 ≤ 17) ↔ (x = -7 ∨ x = -6 ∨ x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0) :=
by sorry

end count_of_integer_values_not_satisfying_inequality_l117_117076


namespace cost_per_piece_l117_117587

variable (totalCost : ℝ) (numberOfPizzas : ℝ) (piecesPerPizza : ℝ)

theorem cost_per_piece (h1 : totalCost = 80) (h2 : numberOfPizzas = 4) (h3 : piecesPerPizza = 5) :
  totalCost / numberOfPizzas / piecesPerPizza = 4 := by
sorry

end cost_per_piece_l117_117587


namespace analytical_expression_of_f_range_of_m_l117_117229

open Real

section Problem1

-- Define the vectors and conditions
variables (ω x : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (f : ℝ → ℝ)
def a := (sin (ω * x), √3 * cos (2 * ω * x))
def b := ((1/2) * cos (ω * x), 1/4)
def f := λ x, (sin (ω * x) * (1 / 2) * cos (ω * x)) + (√3 * cos (2 * ω * x) * (1 / 4))

-- Given the conditions, prove the analytical expression of f(x)
theorem analytical_expression_of_f (ω_pos : ω > 0) (symmetry_distance : (2 * π) / (2 * ω) = π) :
  f(x) = (1/2) * sin(2 * x + π / 3) :=
sorry

end Problem1

section Problem2

-- Given the range condition, prove the range of m for f(x) = m has exactly two solutions
theorem range_of_m (m : ℝ) :
  m ∈ set.Ico (√3 / 4) (1 / 2) →
  ∀ x, 0 ≤ x ∧ x ≤ 7 * π / 12 →
  (∃! y, (f y = m ∧ 0 ≤ y ∧ y ≤ 7 * π / 12)) :=
sorry

end Problem2

end analytical_expression_of_f_range_of_m_l117_117229


namespace log_inequality_l117_117219

theorem log_inequality (x y : ℝ) :
  let log2 := Real.log 2
  let log5 := Real.log 5
  let log3 := Real.log 3
  let log2_3 := log3 / log2
  let log5_3 := log3 / log5
  (log2_3 ^ x - log5_3 ^ x ≥ log2_3 ^ (-y) - log5_3 ^ (-y)) → (x + y ≥ 0) :=
by
  intros h
  sorry

end log_inequality_l117_117219


namespace no_solutions_xyz_l117_117332

theorem no_solutions_xyz : ∀ (x y z : ℝ), x + y = 3 → xy - z^2 = 2 → false := by
  intros x y z h1 h2
  sorry

end no_solutions_xyz_l117_117332


namespace legs_in_room_l117_117147

def total_legs_in_room (tables4 : Nat) (sofa : Nat) (chairs4 : Nat) (tables3 : Nat) (table1 : Nat) (rocking_chair2 : Nat) : Nat :=
  (tables4 * 4) + (sofa * 4) + (chairs4 * 4) + (tables3 * 3) + (table1 * 1) + (rocking_chair2 * 2)

theorem legs_in_room :
  total_legs_in_room 4 1 2 3 1 1 = 40 :=
by
  -- Skipping proof steps
  sorry

end legs_in_room_l117_117147


namespace squirrel_cones_l117_117471

theorem squirrel_cones :
  ∃ (x y : ℕ), 
    x + y < 25 ∧ 
    2 * x > y + 26 ∧ 
    2 * y > x - 4 ∧
    x = 17 ∧ 
    y = 7 :=
by
  sorry

end squirrel_cones_l117_117471


namespace points_calculation_correct_l117_117860

-- Definitions
def points_per_enemy : ℕ := 9
def total_enemies : ℕ := 11
def enemies_undestroyed : ℕ := 3
def enemies_destroyed : ℕ := total_enemies - enemies_undestroyed

def points_earned : ℕ := enemies_destroyed * points_per_enemy

-- Theorem statement
theorem points_calculation_correct : points_earned = 72 := by
  sorry

end points_calculation_correct_l117_117860


namespace Triamoeba_Count_After_One_Week_l117_117317

def TriamoebaCount (n : ℕ) : ℕ :=
  3 ^ n

theorem Triamoeba_Count_After_One_Week : TriamoebaCount 7 = 2187 :=
by
  -- This is the statement to be proved
  sorry

end Triamoeba_Count_After_One_Week_l117_117317


namespace trigonometric_identity_l117_117407

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l117_117407


namespace smallest_k_for_bisectors_l117_117826

theorem smallest_k_for_bisectors (a b c l_a l_b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : l_a = (2 * b * c * Real.sqrt ((1 + (b^2 + c^2 - a^2) / (2 * b * c)) / 2)) / (b + c))
  (h5 : l_b = (2 * a * c * Real.sqrt ((1 + (a^2 + c^2 - b^2) / (2 * a * c)) / 2)) / (a + c)) :
  (l_a + l_b) / (a + b) ≤ 4 / 3 :=
by
  sorry

end smallest_k_for_bisectors_l117_117826


namespace fraction_of_money_to_give_l117_117900

variable {x : ℝ} -- the amount of money the younger brother has

-- Define elder brother's money as 1.25 times younger's
def elder_money := 1.25 * x

-- We want to prove that the fraction of elder's money to be given to younger is 0.1
noncomputable def fraction_to_give (x : ℝ) : ℝ := elder_money / (1.25 * x)

-- Theorem stating the fraction calculation
theorem fraction_of_money_to_give (hx : x > 0) : fraction_to_give x = 0.1 :=
by
  sorry

end fraction_of_money_to_give_l117_117900


namespace number_of_n_factorizable_l117_117284

theorem number_of_n_factorizable :
  ∃! n_values : Finset ℕ, (∀ n ∈ n_values, n ≤ 100 ∧ ∃ a b : ℤ, a + b = -2 ∧ a * b = -n) ∧ n_values.card = 9 := by
  sorry

end number_of_n_factorizable_l117_117284


namespace Berry_temperature_on_Sunday_l117_117807

theorem Berry_temperature_on_Sunday :
  let avg_temp := 99.0
  let days_in_week := 7
  let temp_day1 := 98.2
  let temp_day2 := 98.7
  let temp_day3 := 99.3
  let temp_day4 := 99.8
  let temp_day5 := 99.0
  let temp_day6 := 98.9
  let total_temp_week := avg_temp * days_in_week
  let total_temp_six_days := temp_day1 + temp_day2 + temp_day3 + temp_day4 + temp_day5 + temp_day6
  let temp_on_sunday := total_temp_week - total_temp_six_days
  temp_on_sunday = 98.1 :=
by
  -- Proof of the statement goes here
  sorry

end Berry_temperature_on_Sunday_l117_117807


namespace sin_minus_cos_l117_117365

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l117_117365


namespace sin_minus_cos_l117_117403

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l117_117403


namespace find_number_l117_117171

theorem find_number (x : ℝ) (h: x - (3 / 5) * x = 58) : x = 145 :=
by {
  sorry
}

end find_number_l117_117171


namespace more_plastic_pipe_l117_117312

variable (m_copper m_plastic : Nat)
variable (total_cost cost_per_meter : Nat)

-- Conditions
variable (h1 : m_copper = 10)
variable (h2 : cost_per_meter = 4)
variable (h3 : total_cost = 100)
variable (h4 : m_copper * cost_per_meter + m_plastic * cost_per_meter = total_cost)

-- Proof that the number of more meters of plastic pipe bought compared to the copper pipe is 5
theorem more_plastic_pipe :
  m_plastic - m_copper = 5 :=
by
  -- Since proof is not required, we place sorry here.
  sorry

end more_plastic_pipe_l117_117312


namespace rational_power_sum_l117_117048

theorem rational_power_sum (a b : ℚ) (ha : a = 1 / a) (hb : b = - b) : a ^ 2007 + b ^ 2007 = 1 ∨ a ^ 2007 + b ^ 2007 = -1 := by
  sorry

end rational_power_sum_l117_117048


namespace Fk_same_implies_eq_l117_117439

def Q (n: ℕ) : ℕ :=
  -- Implementation of the square part of n
  sorry

def N (n: ℕ) : ℕ :=
  -- Implementation of the non-square part of n
  sorry

def Fk (k: ℕ) (n: ℕ) : ℕ :=
  -- Implementation of Fk function calculating the smallest positive integer bigger than kn such that Fk(n) * n is a perfect square
  sorry

theorem Fk_same_implies_eq (k: ℕ) (n m: ℕ) (hk: 0 < k) : Fk k n = Fk k m → n = m :=
  sorry

end Fk_same_implies_eq_l117_117439


namespace sqrt8_same_type_as_sqrt2_l117_117803

def same_type_sqrt_2 (x : Real) : Prop := ∃ k : Real, k * Real.sqrt 2 = x

theorem sqrt8_same_type_as_sqrt2 : same_type_sqrt_2 (Real.sqrt 8) :=
  sorry

end sqrt8_same_type_as_sqrt2_l117_117803


namespace arithmetic_sequence_sum_ratio_l117_117701

theorem arithmetic_sequence_sum_ratio
  (a_n : ℕ → ℝ)
  (d a1 : ℝ)
  (S_n : ℕ → ℝ)
  (h_arithmetic : ∀ n, a_n n = a1 + (n-1) * d)
  (h_sum : ∀ n, S_n n = n / 2 * (2 * a1 + (n-1) * d))
  (h_ratio : S_n 4 / S_n 6 = -2 / 3) :
  S_n 5 / S_n 8 = 1 / 40.8 :=
sorry

end arithmetic_sequence_sum_ratio_l117_117701


namespace complex_quadrant_l117_117086

open Complex

theorem complex_quadrant
  (z1 z2 z : ℂ) (h1 : z1 = 2 + I) (h2 : z2 = 1 - I) (h3 : z = z1 / z2) :
  0 < z.re ∧ 0 < z.im :=
by
  -- sorry to skip the proof steps
  sorry

end complex_quadrant_l117_117086


namespace geometric_series_sum_eq_l117_117198

theorem geometric_series_sum_eq (a r : ℝ) 
  (h_sum : (∑' n:ℕ, a * r^n) = 20) 
  (h_odd_sum : (∑' n:ℕ, a * r^(2 * n + 1)) = 8) : 
  r = 2 / 3 := 
sorry

end geometric_series_sum_eq_l117_117198


namespace grill_burns_fifteen_coals_in_twenty_minutes_l117_117176

-- Define the problem conditions
def total_coals (bags : ℕ) (coals_per_bag : ℕ) : ℕ :=
  bags * coals_per_bag

def burning_ratio (total_coals : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / total_coals

-- Given conditions
def bags := 3
def coals_per_bag := 60
def total_minutes := 240
def fifteen_coals := 15

-- Problem statement
theorem grill_burns_fifteen_coals_in_twenty_minutes :
  total_minutes / total_coals bags coals_per_bag * fifteen_coals = 20 :=
by
  sorry

end grill_burns_fifteen_coals_in_twenty_minutes_l117_117176


namespace line_through_points_l117_117138

/-- The line passing through points A(1, 1) and B(2, 3) satisfies the equation 2x - y - 1 = 0. -/
theorem line_through_points (x y : ℝ) :
  (∃ k : ℝ, k * (y - 1) = 2 * (x - 1)) → 2 * x - y - 1 = 0 :=
by
  sorry

end line_through_points_l117_117138


namespace complex_fraction_value_l117_117741

-- Define the imaginary unit
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_value : (3 : ℂ) / ((1 - i) ^ 2) = (3 / 2) * i := by
  sorry

end complex_fraction_value_l117_117741


namespace fixed_point_min_value_l117_117605

theorem fixed_point_min_value {a m n : ℝ} (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) (hm_pos : 0 < m) (hn_pos : 0 < n)
  (h : 3 * m + n = 1) : (1 / m + 3 / n) = 12 := sorry

end fixed_point_min_value_l117_117605


namespace vec_problem_l117_117950

def vec1 : ℤ × ℤ := (3, -5)
def vec2 : ℤ × ℤ := (2, -6)
def scalar_mult (a : ℤ) (v : ℤ × ℤ) : ℤ × ℤ := (a * v.1, a * v.2)
def vec_sub (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 - v2.1, v1.2 - v2.2)
def result := (6, -2)

theorem vec_problem :
  vec_sub (scalar_mult 4 vec1) (scalar_mult 3 vec2) = result := 
by 
  sorry

end vec_problem_l117_117950


namespace value_of_N_l117_117352

theorem value_of_N (N : ℕ) (h : (20 / 100) * N = (60 / 100) * 2500) : N = 7500 :=
by {
  sorry
}

end value_of_N_l117_117352


namespace range_of_m_l117_117424

theorem range_of_m {x y : ℝ} (hx : 0 < x) (hy : 0 < y)
  (h_cond : 1/x + 4/y = 1) : 
  (∃ x y, 0 < x ∧ 0 < y ∧ 1/x + 4/y = 1 ∧ x + y/4 < m^2 + 3 * m) ↔
  (m < -4 ∨ 1 < m) := 
sorry

end range_of_m_l117_117424


namespace average_of_abc_l117_117249

theorem average_of_abc (A B C : ℚ) 
  (h1 : 2002 * C + 4004 * A = 8008) 
  (h2 : 3003 * B - 5005 * A = 7007) : 
  (A + B + C) / 3 = 22 / 9 := 
by 
  sorry

end average_of_abc_l117_117249


namespace no_equal_prob_for_same_color_socks_l117_117752

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l117_117752


namespace max_d_minus_r_proof_l117_117995

noncomputable def max_d_minus_r : ℕ := 35

theorem max_d_minus_r_proof (d r : ℕ) (h1 : 2017 % d = r) (h2 : 1029 % d = r) (h3 : 725 % d = r) :
  d - r ≤ max_d_minus_r :=
  sorry

end max_d_minus_r_proof_l117_117995


namespace find_integer_values_of_m_l117_117675

theorem find_integer_values_of_m (m : ℤ) (x : ℚ) 
  (h₁ : 5 * x - 2 * m = 3 * x - 6 * m + 1)
  (h₂ : -3 < x ∧ x ≤ 2) : m = 0 ∨ m = 1 := 
by 
  sorry

end find_integer_values_of_m_l117_117675


namespace girls_in_class_l117_117033

theorem girls_in_class :
  ∀ (x : ℕ), (12 * 84 + 92 * x = 86 * (12 + x)) → x = 4 :=
by
  sorry

end girls_in_class_l117_117033


namespace smaller_angle_clock_8_10_l117_117811

/-- The measure of the smaller angle formed by the hour and minute hands of a clock at 8:10 p.m. is 175 degrees. -/
theorem smaller_angle_clock_8_10 : 
  let full_circle := 360
  let hour_increment := 30
  let hour_angle_8 := 8 * hour_increment
  let minute_angle_increment := 6
  let hour_hand_adjustment := 10 * (hour_increment / 60)
  let hour_hand_position := hour_angle_8 + hour_hand_adjustment
  let minute_hand_position := 10 * minute_angle_increment
  let angle_difference := if hour_hand_position > minute_hand_position 
                          then hour_hand_position - minute_hand_position 
                          else minute_hand_position - hour_hand_position  
  let smaller_angle := if 2 * angle_difference > full_circle 
                       then full_circle - angle_difference 
                       else angle_difference
  smaller_angle = 175 :=
by 
  sorry

end smaller_angle_clock_8_10_l117_117811


namespace remainder_when_15_plus_y_div_31_l117_117876

theorem remainder_when_15_plus_y_div_31 (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (15 + y) % 31 = 24 :=
by
  sorry

end remainder_when_15_plus_y_div_31_l117_117876


namespace maria_trip_distance_l117_117648

theorem maria_trip_distance (D : ℝ) 
  (h1 : D / 2 + ((D / 2) / 4) + 150 = D) 
  (h2 : 150 = 3 * D / 8) : 
  D = 400 :=
by
  -- Placeholder for the actual proof
  sorry

end maria_trip_distance_l117_117648


namespace find_monday_temperature_l117_117457

theorem find_monday_temperature
  (M T W Th F : ℤ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : F = 35) :
  M = 43 :=
by
  sorry

end find_monday_temperature_l117_117457


namespace combined_cost_price_correct_l117_117296

def face_value_A : ℝ := 100
def discount_A : ℝ := 0.02
def face_value_B : ℝ := 100
def premium_B : ℝ := 0.015
def brokerage : ℝ := 0.002

def purchase_price_A := face_value_A * (1 - discount_A)
def brokerage_fee_A := purchase_price_A * brokerage
def total_cost_price_A := purchase_price_A + brokerage_fee_A

def purchase_price_B := face_value_B * (1 + premium_B)
def brokerage_fee_B := purchase_price_B * brokerage
def total_cost_price_B := purchase_price_B + brokerage_fee_B

def combined_cost_price := total_cost_price_A + total_cost_price_B

theorem combined_cost_price_correct :
  combined_cost_price = 199.899 :=
by
  sorry

end combined_cost_price_correct_l117_117296


namespace garden_area_l117_117862

theorem garden_area (length perimeter : ℝ) (length_50 : 50 * length = 1500) (perimeter_20 : 20 * perimeter = 1500) (rectangular : perimeter = 2 * length + 2 * (perimeter / 2 - length)) :
  length * (perimeter / 2 - length) = 225 := 
by
  sorry

end garden_area_l117_117862


namespace geom_seq_a11_l117_117237

variable {α : Type*} [LinearOrderedField α]

def geom_seq (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geom_seq_a11
  (a : ℕ → α)
  (q : α)
  (ha3 : a 3 = 3)
  (ha7 : a 7 = 6)
  (hgeom : geom_seq a q) :
  a 11 = 12 :=
by
  sorry

end geom_seq_a11_l117_117237


namespace pitbull_chihuahua_weight_ratio_l117_117789

theorem pitbull_chihuahua_weight_ratio
  (C P G : ℕ)
  (h1 : G = 307)
  (h2 : G = 3 * P + 10)
  (h3 : C + P + G = 439) :
  P / C = 3 :=
by {
  sorry
}

end pitbull_chihuahua_weight_ratio_l117_117789


namespace num_2_coins_l117_117989

open Real

theorem num_2_coins (x y z : ℝ) (h1 : x + y + z = 900)
                     (h2 : x + 2 * y + 5 * z = 1950)
                     (h3 : z = 0.5 * x) : y = 450 :=
by sorry

end num_2_coins_l117_117989


namespace boat_speed_in_still_water_l117_117788

theorem boat_speed_in_still_water (V_s : ℝ) (D : ℝ) (t_down : ℝ) (t_up : ℝ) (V_b : ℝ) :
  V_s = 3 → t_down = 1 → t_up = 3 / 2 →
  (V_b + V_s) * t_down = D → (V_b - V_s) * t_up = D → V_b = 15 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end boat_speed_in_still_water_l117_117788


namespace arith_seq_fraction_l117_117537

theorem arith_seq_fraction (a : ℕ → ℝ) (d : ℝ) (h1 : ∀ n, a (n + 1) - a n = d)
  (h2 : d ≠ 0) (h3 : a 3 = 2 * a 1) :
  (a 1 + a 3) / (a 2 + a 4) = 3 / 4 :=
sorry

end arith_seq_fraction_l117_117537


namespace cone_volume_l117_117935

theorem cone_volume (r h : ℝ) (π : ℝ) (V : ℝ) :
    r = 3 → h = 4 → π = Real.pi → V = (1/3) * π * r^2 * h → V = 37.68 :=
by
  sorry

end cone_volume_l117_117935


namespace caitlin_age_l117_117057

theorem caitlin_age (aunt_anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ) 
  (h1 : aunt_anna_age = 60)
  (h2 : brianna_age = aunt_anna_age / 3)
  (h3 : caitlin_age = brianna_age - 7)
  : caitlin_age = 13 :=
by
  sorry

end caitlin_age_l117_117057


namespace sequence_a_n_formula_and_sum_t_n_l117_117667

open Nat

theorem sequence_a_n_formula_and_sum_t_n (S : ℕ → ℕ) :
  (∀ n : ℕ, n > 0 → S n = 2 * (2^n) - 2) → 
  (∀ n : ℕ, ∑ i in range (n+1), S i = 2^(n+2) - 4 - 2 * n) :=
by 
  sorry

end sequence_a_n_formula_and_sum_t_n_l117_117667


namespace imaginary_unit_calculation_l117_117210

theorem imaginary_unit_calculation (i : ℂ) (h : i^2 = -1) : i * (1 + i) = -1 + i := 
by
  sorry

end imaginary_unit_calculation_l117_117210


namespace speed_of_man_in_still_water_l117_117169

variable (v_m v_s : ℝ)

-- Conditions as definitions 
def downstream_distance_eq : Prop :=
  36 = (v_m + v_s) * 3

def upstream_distance_eq : Prop :=
  18 = (v_m - v_s) * 3

theorem speed_of_man_in_still_water (h1 : downstream_distance_eq v_m v_s) (h2 : upstream_distance_eq v_m v_s) : v_m = 9 := 
  by
  sorry

end speed_of_man_in_still_water_l117_117169


namespace system_of_inequalities_solution_set_quadratic_equation_when_m_is_2_l117_117613

theorem system_of_inequalities_solution_set : 
  (∀ x : ℝ, (2 * x - 1 < 7) → (x + 1 > 2) ↔ (1 < x ∧ x < 4)) := 
by 
  sorry

theorem quadratic_equation_when_m_is_2 : 
  (∀ x : ℝ, x^2 - 2 * x - 2 = 0 ↔ (x = 1 + Real.sqrt 3 ∨ x = 1 - Real.sqrt 3)) := 
by 
  sorry

end system_of_inequalities_solution_set_quadratic_equation_when_m_is_2_l117_117613


namespace solve_for_x_l117_117161

theorem solve_for_x (x : ℝ) (h : 4 * x - 5 = 3) : x = 2 :=
by sorry

end solve_for_x_l117_117161


namespace sum_of_remainders_l117_117774

theorem sum_of_remainders 
  (a b c : ℕ) 
  (h1 : a % 53 = 37) 
  (h2 : b % 53 = 14) 
  (h3 : c % 53 = 7) : 
  (a + b + c) % 53 = 5 := 
by 
  sorry

end sum_of_remainders_l117_117774


namespace dissection_impossible_l117_117866

theorem dissection_impossible :
  ∀ (n m : ℕ), n = 1000 → m = 2016 → ¬(∃ (k l : ℕ), k * (n * m) = 1 * 2015 + l * 3) :=
by
  intros n m hn hm
  sorry

end dissection_impossible_l117_117866


namespace sin_minus_cos_l117_117366

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l117_117366


namespace arithmetic_progression_implies_equality_l117_117329

theorem arithmetic_progression_implies_equality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ((a + b) / 2) = ((Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) → a = b :=
by
  sorry

end arithmetic_progression_implies_equality_l117_117329


namespace chessboard_overlap_area_l117_117768

theorem chessboard_overlap_area :
  let n := 8
  let cell_area := 1
  let side_length := 8
  let overlap_area := 32 * (Real.sqrt 2 - 1)
  (∃ black_overlap_area : ℝ, black_overlap_area = overlap_area) :=
by
  sorry

end chessboard_overlap_area_l117_117768


namespace two_roses_more_than_three_carnations_l117_117573

variable {x y : ℝ}

theorem two_roses_more_than_three_carnations
  (h1 : 6 * x + 3 * y > 24)
  (h2 : 4 * x + 5 * y < 22) :
  2 * x > 3 * y := 
by 
  sorry

end two_roses_more_than_three_carnations_l117_117573


namespace find_m_l117_117984

theorem find_m (a b c d : ℕ) (m : ℕ) (a_n b_n c_n d_n: ℕ → ℕ)
  (ha : ∀ n, a_n n = a * n + b)
  (hb : ∀ n, b_n n = c * n + d)
  (hc : ∀ n, c_n n = a_n n * b_n n)
  (hd : ∀ n, d_n n = c_n (n + 1) - c_n n)
  (ha1b1 : m = a_n 1 * b_n 1)
  (hca2b2 : a_n 2 * b_n 2 = 4)
  (hca3b3 : a_n 3 * b_n 3 = 8)
  (hca4b4 : a_n 4 * b_n 4 = 16) :
  m = 4 := 
by sorry

end find_m_l117_117984


namespace goods_train_pass_time_l117_117046

theorem goods_train_pass_time 
  (speed_mans_train_kmph : ℝ) (speed_goods_train_kmph : ℝ) (length_goods_train_m : ℝ) :
  speed_mans_train_kmph = 20 → 
  speed_goods_train_kmph = 92 → 
  length_goods_train_m = 280 → 
  abs ((length_goods_train_m / ((speed_mans_train_kmph + speed_goods_train_kmph) * 1000 / 3600)) - 8.99) < 0.01 :=
by
  sorry

end goods_train_pass_time_l117_117046


namespace second_competitor_distance_difference_l117_117864

theorem second_competitor_distance_difference (jump1 jump2 jump3 jump4 : ℕ) : 
  jump1 = 22 → 
  jump4 = 24 → 
  jump3 = jump2 - 2 → 
  jump4 = jump3 + 3 → 
  jump2 - jump1 = 1 :=
by
  sorry

end second_competitor_distance_difference_l117_117864


namespace emily_original_salary_l117_117524

def original_salary_emily (num_employees : ℕ) (original_employee_salary new_employee_salary new_salary_emily : ℕ) : ℕ :=
  new_salary_emily + (new_employee_salary - original_employee_salary) * num_employees

theorem emily_original_salary :
  original_salary_emily 10 20000 35000 850000 = 1000000 :=
by
  sorry

end emily_original_salary_l117_117524


namespace paper_clips_collected_l117_117444

theorem paper_clips_collected (boxes paper_clips_per_box total_paper_clips : ℕ) 
  (h1 : boxes = 9) 
  (h2 : paper_clips_per_box = 9) 
  (h3 : total_paper_clips = boxes * paper_clips_per_box) : 
  total_paper_clips = 81 :=
by {
  sorry
}

end paper_clips_collected_l117_117444


namespace profit_percentage_is_correct_l117_117495

noncomputable def sellingPrice : ℝ := 850
noncomputable def profit : ℝ := 230
noncomputable def costPrice : ℝ := sellingPrice - profit

noncomputable def profitPercentage : ℝ :=
  (profit / costPrice) * 100

theorem profit_percentage_is_correct :
  profitPercentage = 37.10 :=
by
  sorry

end profit_percentage_is_correct_l117_117495


namespace sin_minus_cos_l117_117384

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l117_117384


namespace pens_sold_l117_117601

variable (C S : ℝ)
variable (n : ℕ)

-- Define conditions
def condition1 : Prop := 10 * C = n * S
def condition2 : Prop := S = 1.5 * C

-- Define the statement to be proved
theorem pens_sold (h1 : condition1 C S n) (h2 : condition2 C S) : n = 6 := by
  -- leave the proof steps to be filled in
  sorry

end pens_sold_l117_117601


namespace inequality_always_holds_l117_117981

variable {a b : ℝ}

theorem inequality_always_holds (ha : a > 0) (hb : b < 0) : 1 / a > 1 / b :=
by
  sorry

end inequality_always_holds_l117_117981


namespace find_linear_combination_l117_117411

variable (a b c : ℝ)

theorem find_linear_combination (h1 : a + 2 * b - 3 * c = 4)
                               (h2 : 5 * a - 6 * b + 7 * c = 8) :
  9 * a + 2 * b - 5 * c = 24 :=
sorry

end find_linear_combination_l117_117411


namespace incorrect_conclusion_l117_117543

noncomputable def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

theorem incorrect_conclusion (m : ℝ) (hx : m - 2 = 0) :
  ¬(∀ x : ℝ, quadratic m x = 2 ↔ x = 2) :=
by
  sorry

end incorrect_conclusion_l117_117543


namespace find_a_and_mono_l117_117974

open Real

noncomputable def f (x : ℝ) (a : ℝ) := (a * 2^x + a - 2) / (2^x + 1)

theorem find_a_and_mono :
  (∀ x : ℝ, f x a + f (-x) a = 0) →
  a = 1 ∧ f 3 1 = 7 / 9 ∧ ∀ x1 x2 : ℝ, x1 < x2 → f x1 1 < f x2 1 :=
by
  sorry

end find_a_and_mono_l117_117974


namespace value_calculation_l117_117415

-- Define the given number
def given_number : ℝ := 93.75

-- Define the percentages as ratios
def forty_percent : ℝ := 0.4
def sixteen_percent : ℝ := 0.16

-- Calculate the intermediate value for 40% of the given number
def intermediate_value := forty_percent * given_number

-- Final value calculation for 16% of the intermediate value
def final_value := sixteen_percent * intermediate_value

-- The theorem to prove
theorem value_calculation : final_value = 6 := by
  -- Expanding definitions to substitute and simplify
  unfold final_value intermediate_value forty_percent sixteen_percent given_number
  -- Proving the correctness by calculating
  sorry

end value_calculation_l117_117415


namespace find_quotient_l117_117256

def dividend : ℝ := 13787
def remainder : ℝ := 14
def divisor : ℝ := 154.75280898876406
def quotient : ℝ := 89

theorem find_quotient :
  (dividend - remainder) / divisor = quotient :=
sorry

end find_quotient_l117_117256


namespace max_value_ahn_operation_l117_117316

theorem max_value_ahn_operation :
  ∃ n : ℤ, 100 ≤ n ∧ n ≤ 999 ∧ (300 - n)^2 - 10 = 39990 :=
by
  sorry

end max_value_ahn_operation_l117_117316


namespace percentage_democrats_l117_117562

/-- In a certain city, some percent of the registered voters are Democrats and the rest are Republicans. In a mayoral race, 85 percent of the registered voters who are Democrats and 20 percent of the registered voters who are Republicans are expected to vote for candidate A. Candidate A is expected to get 59 percent of the registered voters' votes. Prove that 60 percent of the registered voters are Democrats. -/
theorem percentage_democrats (D R : ℝ) (h : D + R = 100) (h1 : 0.85 * D + 0.20 * R = 59) : 
  D = 60 :=
by
  sorry

end percentage_democrats_l117_117562


namespace trigonometric_identity_l117_117405

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l117_117405


namespace sequence_general_term_l117_117225

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 1 → a n = n * (a (n + 1) - a n)) : 
  ∀ n : ℕ, n ≥ 1 → a n = n := 
by 
  sorry

end sequence_general_term_l117_117225


namespace football_throwing_distance_l117_117452

theorem football_throwing_distance 
  (T : ℝ)
  (yards_per_throw_at_T : ℝ)
  (yards_per_throw_at_80 : ℝ)
  (throws_on_Saturday : ℕ)
  (throws_on_Sunday : ℕ)
  (saturday_distance sunday_distance : ℝ)
  (total_distance : ℝ) :
  yards_per_throw_at_T = 20 →
  yards_per_throw_at_80 = 40 →
  throws_on_Saturday = 20 →
  throws_on_Sunday = 30 →
  saturday_distance = throws_on_Saturday * yards_per_throw_at_T →
  sunday_distance = throws_on_Sunday * yards_per_throw_at_80 →
  total_distance = saturday_distance + sunday_distance →
  total_distance = 1600 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end football_throwing_distance_l117_117452


namespace chris_earnings_total_l117_117478

-- Define the conditions
variable (hours_week1 hours_week2 : ℕ) (wage_per_hour earnings_diff : ℝ)
variable (hours_week1_val : hours_week1 = 18)
variable (hours_week2_val : hours_week2 = 30)
variable (earnings_diff_val : earnings_diff = 65.40)
variable (constant_wage : wage_per_hour > 0)

-- Theorem statement
theorem chris_earnings_total (total_earnings : ℝ) :
  hours_week2 - hours_week1 = 12 →
  wage_per_hour = earnings_diff / 12 →
  total_earnings = (hours_week1 + hours_week2) * wage_per_hour →
  total_earnings = 261.60 :=
by
  intros h1 h2 h3
  sorry

end chris_earnings_total_l117_117478


namespace cubes_identity_l117_117549

theorem cubes_identity (a b c : ℝ) (h₁ : a + b + c = 15) (h₂ : ab + ac + bc = 40) : 
    a^3 + b^3 + c^3 - 3 * a * b * c = 1575 :=
by 
  sorry

end cubes_identity_l117_117549


namespace least_number_divisible_remainder_l117_117481

theorem least_number_divisible_remainder (n : ℕ) (h1 : n % 34 = 4) (h2 : n % 5 = 4) : n = 174 := 
sorry

end least_number_divisible_remainder_l117_117481


namespace maximum_value_P_l117_117717

open Classical

noncomputable def P (a b c d : ℝ) : ℝ := a * b + b * c + c * d + d * a

theorem maximum_value_P : ∀ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ a + b + c + d = 40 → P a b c d ≤ 800 :=
by
  sorry

end maximum_value_P_l117_117717


namespace smallest_positive_integer_form_l117_117772

theorem smallest_positive_integer_form (m n : ℤ) : 
  ∃ (d : ℤ), d > 0 ∧ d = 1205 * m + 27090 * n ∧ (∀ e, e > 0 → (∃ x y : ℤ, d = 1205 * x + 27090 * y) → d ≤ e) :=
sorry

end smallest_positive_integer_form_l117_117772


namespace SmartMart_science_kits_l117_117598

theorem SmartMart_science_kits (sc pz : ℕ) (h1 : pz = sc - 9) (h2 : pz = 36) : sc = 45 := by
  sorry

end SmartMart_science_kits_l117_117598


namespace tetrahedron_inequality_l117_117703

variables (S A B C : Point)
variables (SA SB SC : Real)
variables (ABC : Plane)
variables (z : Real)
variable (h1 : angle B S C = π / 2)
variable (h2 : Project (point S) ABC = Orthocenter triangle ABC)
variable (h3 : RadiusInscribedCircle triangle ABC = z)

theorem tetrahedron_inequality :
  SA^2 + SB^2 + SC^2 >= 18 * z^2 :=
sorry

end tetrahedron_inequality_l117_117703


namespace trigonometric_identity_l117_117410

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l117_117410


namespace minimal_value_of_function_l117_117143

theorem minimal_value_of_function (x : ℝ) (hx : x > 1 / 2) :
  (x = 1 → (x^2 + 1) / x = 2) ∧
  (∀ y, (∀ z, z > 1 / 2 → y ≤ (z^2 + 1) / z) → y = 2) :=
by {
  sorry
}

end minimal_value_of_function_l117_117143


namespace Carson_skipped_times_l117_117948

variable (length width total_circles actual_distance perimeter distance_skipped : ℕ)
variable (total_distance : ℕ)

def perimeter_calculation (length width : ℕ) : ℕ := 2 * (length + width)

def total_distance_calculation (total_circles perimeter : ℕ) : ℕ := total_circles * perimeter

def distance_skipped_calculation (total_distance actual_distance : ℕ) : ℕ := total_distance - actual_distance

def times_skipped_calculation (distance_skipped perimeter : ℕ) : ℕ := distance_skipped / perimeter

theorem Carson_skipped_times (h_length : length = 600) 
                             (h_width : width = 400) 
                             (h_total_circles : total_circles = 10) 
                             (h_actual_distance : actual_distance = 16000) 
                             (h_perimeter : perimeter = perimeter_calculation length width) 
                             (h_total_distance : total_distance = total_distance_calculation total_circles perimeter) 
                             (h_distance_skipped : distance_skipped = distance_skipped_calculation total_distance actual_distance) :
                             times_skipped_calculation distance_skipped perimeter = 2 := 
by
  simp [perimeter_calculation, total_distance_calculation, distance_skipped_calculation, times_skipped_calculation]
  sorry

end Carson_skipped_times_l117_117948


namespace find_missing_number_l117_117686

theorem find_missing_number (x : ℝ) (h : 0.25 / x = 2 / 6) : x = 0.75 :=
by
  sorry

end find_missing_number_l117_117686


namespace bruno_initial_books_l117_117058

theorem bruno_initial_books (X : ℝ)
  (h1 : X - 4.5 + 10.25 = 39.75) :
  X = 34 := by
  sorry

end bruno_initial_books_l117_117058


namespace parabola_equation_l117_117082

theorem parabola_equation (a : ℝ) :
  (∀ x, (x + 1) * (x - 3) = 0 ↔ x = -1 ∨ x = 3) →
  (∀ y, y = a * (0 + 1) * (0 - 3) → y = 3) →
  a = -1 → 
  (∀ x, y = a * (x + 1) * (x - 3) → y = -x^2 + 2 * x + 3) :=
by
  intros h₁ h₂ ha
  sorry

end parabola_equation_l117_117082


namespace circle_equation_l117_117425

theorem circle_equation (x y : ℝ) (h : ∀ x y : ℝ, x^2 + y^2 ≥ 64) :
  x^2 + y^2 - 64 = 0 ↔ x = 0 ∧ y = 0 :=
by
  sorry

end circle_equation_l117_117425


namespace vector_subtraction_l117_117949

def vector1 : ℝ × ℝ := (3, -5)
def vector2 : ℝ × ℝ := (2, -6)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_subtraction :
  (scalar1 • vector1 - scalar2 • vector2) = (6, -2) := by
  sorry

end vector_subtraction_l117_117949


namespace multiply_repeating_decimals_l117_117525

noncomputable def repeating_decimal_03 : ℚ := 1 / 33
noncomputable def repeating_decimal_8 : ℚ := 8 / 9

theorem multiply_repeating_decimals : repeating_decimal_03 * repeating_decimal_8 = 8 / 297 := by 
  sorry

end multiply_repeating_decimals_l117_117525


namespace knights_and_liars_l117_117000

/--
Suppose we have a set of natives, each of whom is either a liar or a knight.
Each native declares to all others: "You are all liars."
This setup implies that there must be exactly one knight among them.
-/
theorem knights_and_liars (natives : Type) (is_knight : natives → Prop) (is_liar : natives → Prop)
  (h1 : ∀ x, is_knight x ∨ is_liar x) 
  (h2 : ∀ x y, x ≠ y → (is_knight x → is_liar y) ∧ (is_liar x → is_knight y))
  : ∃! x, is_knight x :=
by
  sorry

end knights_and_liars_l117_117000


namespace proof_problem_l117_117441

noncomputable def A : Set ℝ := { x | x^2 - 4 = 0 }
noncomputable def B : Set ℝ := { y | ∃ x, y = x^2 - 4 }

theorem proof_problem :
  (A ∩ B = A) ∧ (A ∪ B = B) :=
by {
  sorry
}

end proof_problem_l117_117441


namespace total_wheels_in_garage_l117_117467

theorem total_wheels_in_garage :
  let cars := 2,
      wheels_per_car := 4,
      riding_lawnmower := 1,
      wheels_per_lawnmower := 4,
      bicycles := 3,
      wheels_per_bicycle := 2,
      tricycle := 1,
      wheels_per_tricycle := 3,
      unicycle := 1,
      wheels_per_unicycle := 1 in
  cars * wheels_per_car + riding_lawnmower * wheels_per_lawnmower + bicycles * wheels_per_bicycle + tricycle * wheels_per_tricycle + unicycle * wheels_per_unicycle = 22 := by
  sorry

end total_wheels_in_garage_l117_117467


namespace point_on_coordinate_axes_l117_117339

-- Definitions and assumptions from the problem conditions
variables {a b : ℝ}

-- The theorem statement asserts that point M(a, b) must be located on the coordinate axes given ab = 0
theorem point_on_coordinate_axes (h : a * b = 0) : 
  (a = 0) ∨ (b = 0) :=
by
  sorry

end point_on_coordinate_axes_l117_117339


namespace sin_minus_cos_l117_117381

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l117_117381


namespace marcia_minutes_worked_l117_117250

/--
If Marcia worked for 5 hours on her science project,
then she worked for 300 minutes.
-/
theorem marcia_minutes_worked (hours : ℕ) (h : hours = 5) : (hours * 60) = 300 := by
  sorry

end marcia_minutes_worked_l117_117250


namespace triangle_sides_l117_117564

noncomputable def sides (a b c : ℝ) : Prop :=
  (a = Real.sqrt (427 / 3)) ∧
  (b = Real.sqrt (427 / 3) + 3/2) ∧
  (c = Real.sqrt (427 / 3) - 3/2)

theorem triangle_sides (a b c : ℝ) (h1 : b - c = 3) (h2 : ∃ d : ℝ, d = 10)
  (h3 : ∃ BD CD : ℝ, CD - BD = 12 ∧ BD + CD = a ∧ 
    a = 2 * (BD + 12 / 2)) :
  sides a b c :=
  sorry

end triangle_sides_l117_117564


namespace hats_count_l117_117698

theorem hats_count (T M W : ℕ) (hT : T = 1800)
  (hM : M = (2 * T) / 3) (hW : W = T - M) 
  (hats_men : ℕ) (hats_women : ℕ) (m_hats_condition : hats_men = 15 * M / 100)
  (w_hats_condition : hats_women = 25 * W / 100) :
  hats_men + hats_women = 330 :=
by sorry

end hats_count_l117_117698


namespace solve_for_c_l117_117240

theorem solve_for_c (a b c : ℝ) (B : ℝ) (ha : a = 4) (hb : b = 2*Real.sqrt 7) (hB : B = Real.pi / 3) : 
  (c^2 - 4*c - 12 = 0) → c = 6 :=
by 
  intro h
  -- Details of the proof would be here
  sorry

end solve_for_c_l117_117240


namespace compare_fractions_l117_117834

variable {a b c d : ℝ}

theorem compare_fractions (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) :
  (b / (a - c)) < (a / (b - d)) := 
by
  sorry

end compare_fractions_l117_117834


namespace total_number_of_balls_is_twelve_l117_117431

noncomputable def num_total_balls (a : ℕ) : Prop :=
(3 : ℚ) / a = (25 : ℚ) / 100

theorem total_number_of_balls_is_twelve : num_total_balls 12 :=
by sorry

end total_number_of_balls_is_twelve_l117_117431


namespace sin_minus_cos_l117_117404

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l117_117404


namespace symmetric_coordinates_l117_117841

-- Define the point A as a tuple of its coordinates
def A : Prod ℤ ℤ := (-1, 2)

-- Define what it means for point A' to be symmetric to the origin
def symmetric_to_origin (p : Prod ℤ ℤ) : Prod ℤ ℤ :=
  (-p.1, -p.2)

-- The theorem we need to prove
theorem symmetric_coordinates :
  symmetric_to_origin A = (1, -2) :=
by
  sorry

end symmetric_coordinates_l117_117841


namespace trailing_zeros_50_factorial_l117_117060

def factorial_trailing_zeros (n : Nat) : Nat :=
  n / 5 + n / 25 -- Count the number of trailing zeros given the algorithm used in solution steps

theorem trailing_zeros_50_factorial : factorial_trailing_zeros 50 = 12 :=
by 
  -- Proof goes here
  sorry

end trailing_zeros_50_factorial_l117_117060


namespace solve_for_x_l117_117270

def condition (x : ℝ) : Prop := (x - 5)^3 = (1 / 27)⁻¹

theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 8 := by
  use 8
  unfold condition
  sorry

end solve_for_x_l117_117270


namespace necessary_not_sufficient_l117_117015

-- Definitions and conditions based on the problem statement
def x_ne_1 (x : ℝ) : Prop := x ≠ 1
def polynomial_ne_zero (x : ℝ) : Prop := (x^2 - 3 * x + 2) ≠ 0

-- The theorem statement
theorem necessary_not_sufficient (x : ℝ) : 
  (∀ x, polynomial_ne_zero x → x_ne_1 x) ∧ ¬ (∀ x, x_ne_1 x → polynomial_ne_zero x) :=
by 
  intros
  sorry

end necessary_not_sufficient_l117_117015


namespace total_blocks_to_ride_l117_117093

-- Constants representing the problem conditions
def rotations_per_block : ℕ := 200
def initial_rotations : ℕ := 600
def additional_rotations : ℕ := 1000

-- Main statement asserting the total number of blocks Greg wants to ride
theorem total_blocks_to_ride : 
  (initial_rotations / rotations_per_block) + (additional_rotations / rotations_per_block) = 8 := 
  by 
    sorry

end total_blocks_to_ride_l117_117093


namespace no_integer_solutions_l117_117528

theorem no_integer_solutions (x y : ℤ) : x^3 + 4 * x^2 + x ≠ 18 * y^3 + 18 * y^2 + 6 * y + 3 := 
by 
  sorry

end no_integer_solutions_l117_117528


namespace calculation_proof_l117_117642

theorem calculation_proof : 
  2 * Real.tan (Real.pi / 3) - (-2023) ^ 0 + (1 / 2) ^ (-1 : ℤ) + abs (Real.sqrt 3 - 1) = 3 * Real.sqrt 3 := 
by
  sorry

end calculation_proof_l117_117642


namespace rectangle_area_l117_117179

theorem rectangle_area (x : ℝ) :
  let large_rectangle_area := (2 * x + 14) * (2 * x + 10)
  let hole_area := (4 * x - 6) * (2 * x - 4)
  let square_area := (x + 3) * (x + 3)
  large_rectangle_area - hole_area + square_area = -3 * x^2 + 82 * x + 125 := 
by
  sorry

end rectangle_area_l117_117179


namespace min_value_expr_l117_117100

theorem min_value_expr (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_sum : a + b = 1) : 
  (b / (3 * a)) + (3 / b) ≥ 5 := 
sorry

end min_value_expr_l117_117100


namespace twentieth_number_l117_117463

-- Defining the conditions and goal
theorem twentieth_number :
  ∃ x : ℕ, x % 8 = 5 ∧ x % 3 = 2 ∧ (∃ n : ℕ, x = 5 + 24 * n) ∧ x = 461 := 
sorry

end twentieth_number_l117_117463


namespace solve_for_x_l117_117624

theorem solve_for_x (y : ℝ) : 
  ∃ x : ℝ, 19 * (x + y) + 17 = 19 * (-x + y) - 21 ∧ 
           x = -21 / 38 :=
by
  sorry

end solve_for_x_l117_117624


namespace solve_equation_l117_117454

noncomputable def equation (x : ℝ) : Prop :=
  2 / (x - 2) = (1 + x) / (x - 2) + 1

theorem solve_equation : ∀ (x : ℝ), equation x ∧ x ≠ 2 ↔ x = 3 / 2 :=
by
  intro x
  split
  sorry
  sorry

end solve_equation_l117_117454


namespace stamps_on_last_page_l117_117722

theorem stamps_on_last_page
  (B : ℕ) (P_b : ℕ) (S_p : ℕ) (S_p_star : ℕ) 
  (B_comp : ℕ) (P_last : ℕ) 
  (stamps_total : ℕ := B * P_b * S_p) 
  (pages_total : ℕ := stamps_total / S_p_star)
  (pages_comp : ℕ := B_comp * P_b)
  (pages_filled : ℕ := pages_total - pages_comp) :
  stamps_total - (pages_total - 1) * S_p_star = 8 :=
by
  -- Proof steps would follow here.
  sorry

end stamps_on_last_page_l117_117722


namespace correct_structure_l117_117477

-- Definitions for the conditions regarding flowchart structures
def loop_contains_conditional : Prop := ∀ (loop : Prop), ∃ (conditional : Prop), conditional ∧ loop
def unique_flowchart_for_boiling_water : Prop := ∀ (flowcharts : Prop), ∃! (boiling_process : Prop), flowcharts ∧ boiling_process
def conditional_does_not_contain_sequential : Prop := ∀ (conditional : Prop), ∃ (sequential : Prop), ¬ (conditional ∧ sequential)
def conditional_must_contain_loop : Prop := ∀ (conditional : Prop), ∃ (loop : Prop), conditional ∧ loop

-- The proof statement
theorem correct_structure (A B C D : Prop) (hA : A = loop_contains_conditional) 
  (hB : B = unique_flowchart_for_boiling_water) 
  (hC : C = conditional_does_not_contain_sequential) 
  (hD : D = conditional_must_contain_loop) : 
  A = loop_contains_conditional ∧ ¬ B ∧ ¬ C ∧ ¬ D :=
by {
  sorry
}

end correct_structure_l117_117477


namespace trigonometric_identity_l117_117409

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l117_117409


namespace original_number_satisfies_equation_l117_117162

theorem original_number_satisfies_equation 
  (x : ℝ) 
  (h : 1000 * x = 9 * (1 / x)) : 
  x = 3 * (√10 / 100) :=
by
  sorry

end original_number_satisfies_equation_l117_117162


namespace bouquet_combinations_l117_117493

theorem bouquet_combinations :
  ∃ n : ℕ, (∀ r c t : ℕ, 4 * r + 3 * c + 2 * t = 60 → true) ∧ n = 13 :=
sorry

end bouquet_combinations_l117_117493


namespace train_speed_l117_117801

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 3500) (h_time : time = 80) : 
  length / time = 43.75 := 
by 
  sorry

end train_speed_l117_117801


namespace sin_minus_cos_l117_117400

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l117_117400


namespace hannah_probability_12_flips_l117_117023

/-!
We need to prove that the probability of getting fewer than 4 heads when flipping 12 coins is 299/4096.
-/

def probability_fewer_than_4_heads (flips : ℕ) : ℚ :=
  let total_outcomes := 2^flips
  let favorable_outcomes := (Nat.choose flips 0) + (Nat.choose flips 1) + (Nat.choose flips 2) + (Nat.choose flips 3)
  favorable_outcomes / total_outcomes

theorem hannah_probability_12_flips : probability_fewer_than_4_heads 12 = 299 / 4096 := by
  sorry

end hannah_probability_12_flips_l117_117023


namespace jan_paid_amount_l117_117577

def number_of_roses (dozens : Nat) : Nat := dozens * 12

def total_cost (number_of_roses : Nat) (cost_per_rose : Nat) : Nat := number_of_roses * cost_per_rose

def discounted_price (total_cost : Nat) (discount_percentage : Nat) : Nat := total_cost * discount_percentage / 100

theorem jan_paid_amount :
  let dozens := 5
  let cost_per_rose := 6
  let discount_percentage := 80
  number_of_roses dozens = 60 →
  total_cost (number_of_roses dozens) cost_per_rose = 360 →
  discounted_price (total_cost (number_of_roses dozens) cost_per_rose) discount_percentage = 288 :=
by
  intros
  sorry

end jan_paid_amount_l117_117577


namespace Amith_current_age_l117_117306

variable (A D : ℕ)

theorem Amith_current_age
  (h1 : A - 5 = 3 * (D - 5))
  (h2 : A + 10 = 2 * (D + 10)) :
  A = 50 := by
  sorry

end Amith_current_age_l117_117306


namespace annual_avg_growth_rate_export_volume_2023_l117_117883

variable (V0 V2 V3 : ℕ) (r : ℝ)
variable (h1 : V0 = 200000) (h2 : V2 = 450000) (h3 : V3 = 675000)

-- Definition of the exponential growth equation
def growth_exponential (V0 Vn: ℕ) (n : ℕ) (r : ℝ) : Prop :=
  Vn = V0 * ((1 + r) ^ n)

-- The Lean statement to prove the annual average growth rate
theorem annual_avg_growth_rate (x : ℝ) (h : growth_exponential V0 V2 2 x) : 
  x = 0.5 :=
by
  sorry

-- The Lean statement to prove the export volume in 2023
theorem export_volume_2023 (h_growth : growth_exponential V2 V3 1 0.5) :
  V3 = 675000 :=
by
  sorry

end annual_avg_growth_rate_export_volume_2023_l117_117883


namespace a_1000_value_l117_117103

open Nat

theorem a_1000_value :
  ∃ (a : ℕ → ℤ), (a 1 = 1010) ∧ (a 2 = 1011) ∧ 
  (∀ n ≥ 1, a n + a (n+1) + a (n+2) = 2 * n) ∧ 
  (a 1000 = 1676) :=
sorry

end a_1000_value_l117_117103


namespace calculate_expression_l117_117944

theorem calculate_expression : 
  (-6)^6 / 6^4 - 2^5 + 9^2 = 85 := 
by sorry

end calculate_expression_l117_117944


namespace greg_experienced_less_rain_l117_117165

theorem greg_experienced_less_rain (rain_day1 rain_day2 rain_day3 rain_house : ℕ) 
  (h1 : rain_day1 = 3) 
  (h2 : rain_day2 = 6) 
  (h3 : rain_day3 = 5) 
  (h4 : rain_house = 26) :
  rain_house - (rain_day1 + rain_day2 + rain_day3) = 12 :=
by
  sorry

end greg_experienced_less_rain_l117_117165


namespace sqrt_abc_sum_eq_162sqrt2_l117_117719

theorem sqrt_abc_sum_eq_162sqrt2 (a b c : ℝ) (h1 : b + c = 15) (h2 : c + a = 18) (h3 : a + b = 21) :
    Real.sqrt (a * b * c * (a + b + c)) = 162 * Real.sqrt 2 :=
by
  sorry

end sqrt_abc_sum_eq_162sqrt2_l117_117719


namespace P_works_alone_l117_117723

theorem P_works_alone (P : ℝ) (hP : 2 * (1 / P + 1 / 15) + 0.6 * (1 / P) = 1) : P = 3 :=
by sorry

end P_works_alone_l117_117723


namespace valid_triangle_inequality_l117_117993

theorem valid_triangle_inequality (a : ℝ) 
  (h1 : 4 + 6 > a) 
  (h2 : 4 + a > 6) 
  (h3 : 6 + a > 4) : 
  a = 5 :=
sorry

end valid_triangle_inequality_l117_117993


namespace geometric_seq_a4_a7_l117_117104

variable {a : ℕ → ℝ}

def is_geometric (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

theorem geometric_seq_a4_a7
  (h_geom : is_geometric a)
  (h_roots : ∃ a_1 a_10 : ℝ, (a 1 = a_1 ∧ a 10 = a_10) ∧ (2 * a_1 ^ 2 + 5 * a_1 + 1 = 0) ∧ (2 * a_10 ^ 2 + 5 * a_10 + 1 = 0)):
  a 4 * a 7 = 1 / 2 :=
by
  sorry

end geometric_seq_a4_a7_l117_117104


namespace point_on_angle_bisector_l117_117687

theorem point_on_angle_bisector (a b : ℝ) (h : ∃ p : ℝ, (a, b) = (p, -p)) : a = -b :=
sorry

end point_on_angle_bisector_l117_117687


namespace simplify_expression_l117_117895

variable (x : ℝ)

def expr := (5*x^10 + 8*x^8 + 3*x^6) + (2*x^12 + 3*x^10 + x^8 + 4*x^6 + 2*x^2 + 7)

theorem simplify_expression : expr x = 2*x^12 + 8*x^10 + 9*x^8 + 7*x^6 + 2*x^2 + 7 :=
by
  sorry

end simplify_expression_l117_117895


namespace option_one_cost_option_two_cost_cost_effectiveness_l117_117043

-- Definition of costs based on conditions
def price_of_suit : ℕ := 500
def price_of_tie : ℕ := 60
def discount_option_one (x : ℕ) : ℕ := 60 * x + 8800
def discount_option_two (x : ℕ) : ℕ := 54 * x + 9000

-- Theorem statements
theorem option_one_cost (x : ℕ) (hx : x > 20) : discount_option_one x = 60 * x + 8800 :=
by sorry

theorem option_two_cost (x : ℕ) (hx : x > 20) : discount_option_two x = 54 * x + 9000 :=
by sorry

theorem cost_effectiveness (x : ℕ) (hx : x = 30) : discount_option_one x < discount_option_two x :=
by sorry

end option_one_cost_option_two_cost_cost_effectiveness_l117_117043


namespace arranging_balls_l117_117235

theorem arranging_balls (white_balls black_balls : ℕ) (h_white : white_balls = 7) (h_black : black_balls = 5) :
    ∃ n : ℕ, n = 56 ∧ 
    (white_balls ≥ black_balls + 1) ∧ 
    (n = (Nat.choose (white_balls + 1) black_balls)) := by
  sorry

end arranging_balls_l117_117235


namespace sin_minus_cos_l117_117374

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l117_117374


namespace range_of_t_l117_117115

noncomputable def condition (t : ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 5 / 2 ∧ (t * x^2 + 2 * x - 2 > 0)

theorem range_of_t (t : ℝ) : ¬¬ condition t → t > - 1 / 2 :=
by
  intros h
  -- The actual proof should be here
  sorry

end range_of_t_l117_117115


namespace length_of_CD_l117_117429

theorem length_of_CD
  (S : ℝ) -- Placeholder for the center of the circle (can be non-specific)
  (R : ℝ) (SX : ℝ) (AB : ℝ) (X : Bool := true) -- Assume X represents perpendicularity
  (H_radius : R = 52)
  (H_SX : SX = 25)
  (H_AB : AB = 96)
  (H_perpendicular : X = true)
  : ℝ := 100

end length_of_CD_l117_117429


namespace total_students_is_30_l117_117634

def students_per_bed : ℕ := 2 

def beds_per_room : ℕ := 2 

def students_per_couch : ℕ := 1 

def rooms_booked : ℕ := 6 

def total_students := (students_per_bed * beds_per_room + students_per_couch) * rooms_booked

theorem total_students_is_30 : total_students = 30 := by
  sorry

end total_students_is_30_l117_117634


namespace total_games_in_season_l117_117798

theorem total_games_in_season :
  let num_teams := 14
  let teams_per_division := 7
  let games_within_division_per_team := 6 * 3
  let games_against_other_division_per_team := 7
  let games_per_team := games_within_division_per_team + games_against_other_division_per_team
  let total_initial_games := games_per_team * num_teams
  let total_games := total_initial_games / 2
  total_games = 175 :=
by
  sorry

end total_games_in_season_l117_117798


namespace sqrt2_over_2_not_covered_by_rationals_l117_117963

noncomputable def rational_not_cover_sqrt2_over_2 : Prop :=
  ∀ (a b : ℤ) (h_ab : Int.gcd a b = 1) (h_b_pos : b > 0)
  (h_frac : (a : ℚ) / b ∈ Set.Ioo 0 1),
  abs ((Real.sqrt 2) / 2 - (a : ℚ) / b) > 1 / (4 * b^2)

-- Placeholder for the proof
theorem sqrt2_over_2_not_covered_by_rationals :
  rational_not_cover_sqrt2_over_2 := 
by sorry

end sqrt2_over_2_not_covered_by_rationals_l117_117963


namespace ceil_inequality_range_x_solve_eq_l117_117114

-- Definition of the mathematical ceiling function to comply with the condition a).
def ceil (a : ℚ) : ℤ := ⌈a⌉

-- Condition 1: Relationship between m and ⌈m⌉.
theorem ceil_inequality (m : ℚ) : m ≤ ceil m ∧ ceil m < m + 1 :=
sorry

-- Part 2.1: Range of x given {3x + 2} = 8.
theorem range_x (x : ℚ) (h : ceil (3 * x + 2) = 8) : 5 / 3 < x ∧ x ≤ 2 :=
sorry

-- Part 2.2: Solving {3x - 2} = 2x + 1/2
theorem solve_eq (x : ℚ) (h : ceil (3 * x - 2) = 2 * x + 1 / 2) : x = 7 / 4 ∨ x = 9 / 4 :=
sorry

end ceil_inequality_range_x_solve_eq_l117_117114


namespace sin_minus_cos_l117_117401

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l117_117401


namespace no_integer_pair_satisfies_conditions_l117_117812

theorem no_integer_pair_satisfies_conditions :
  ¬ ∃ (x y : ℤ), x = x^2 + y^2 + 1 ∧ y = 3 * x * y := 
by
  sorry

end no_integer_pair_satisfies_conditions_l117_117812


namespace commute_time_difference_l117_117321

-- Define the conditions as constants
def distance_to_work : ℝ := 1.5
def walking_speed : ℝ := 3
def train_speed : ℝ := 20
def additional_train_time_minutes : ℝ := 10.5

-- The main proof problem
theorem commute_time_difference : 
  (distance_to_work / walking_speed * 60) - 
  ((distance_to_work / train_speed * 60) + additional_train_time_minutes) = 15 :=
by
  sorry

end commute_time_difference_l117_117321


namespace function_decreasing_interval_l117_117742

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

def decreasing_interval (a b : ℝ) : Prop :=
  ∀ x : ℝ, a < x ∧ x < b → 0 > (deriv f x)

theorem function_decreasing_interval : decreasing_interval (-1) 3 :=
by 
  sorry

end function_decreasing_interval_l117_117742


namespace sin_minus_cos_l117_117382

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l117_117382


namespace close_time_for_pipe_b_l117_117779

-- Define entities and rates
def rate_fill (A_rate B_rate : ℝ) (t_fill t_empty t_fill_target t_close : ℝ) : Prop :=
  A_rate = 1 / t_fill ∧
  B_rate = 1 / t_empty ∧
  t_fill_target = 30 ∧
  A_rate * (t_close + (t_fill_target - t_close)) - B_rate * t_close = 1

-- Declare the theorem statement
theorem close_time_for_pipe_b (A_rate B_rate t_fill_target t_fill t_empty t_close: ℝ) :
   rate_fill A_rate B_rate t_fill t_empty t_fill_target t_close → t_close = 26.25 :=
by have h1 : A_rate = 1 / 15 := by sorry
   have h2 : B_rate = 1 / 24 := by sorry
   have h3 : t_fill_target = 30 := by sorry
   sorry

end close_time_for_pipe_b_l117_117779


namespace problem_m_n_l117_117448

theorem problem_m_n (m n : ℝ) (h1 : m * n = 1) (h2 : m^2 + n^2 = 3) (h3 : m^3 + n^3 = 44 + n^4) (h4 : m^5 + 5 = 11) : m^9 + n = -29 :=
sorry

end problem_m_n_l117_117448


namespace projection_matrix_solution_l117_117278

theorem projection_matrix_solution :
  ∃ a c : ℚ,
    (∀ (m n : ℚ), ⟨a, (21 : ℚ) / 76⟩ = m ∧ ⟨c, (55 : ℚ) / 76⟩ = n → 
       m * m + ⟨(21 : ℚ) / 76 * c, 0⟩ = (m : ℚ) * ⟨1, 0⟩ ∧ 
       ⟨ m * (21 : ℚ) / 76 + (21 : ℚ) / 76 * (55 : ℚ) / 76, m = (21 : ℚ) / 76 ⟩ ∧ 
       ⟨c * m + (55 : ℚ) / 76 * c, c = c⟩ ∧ 
       ⟨ c * (21 : ℚ) / 76 + (55 : ℚ) / 76 * (55 : ℚ) / 76, c = (55 : ℚ) / 76⟩ ∧ 
       m = (7 : ℚ) / 19 ∧ 
       c = (21 : ℚ) / 76
    ) 

end projection_matrix_solution_l117_117278


namespace three_digit_sum_of_factorials_eq_l117_117139

-- Define the property of a three-digit number
def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n < 1000

-- Define the function to calculate the sum of factorials of the digits of a number
def sum_of_factorials_of_digits (n : ℕ) : ℕ :=
  let x := n / 100 in
  let y := (n % 100) / 10 in
  let z := n % 10 in
  Nat.factorial x + Nat.factorial y + Nat.factorial z

-- State the theorem to prove
theorem three_digit_sum_of_factorials_eq (n : ℕ) (h : is_three_digit n) :
  sum_of_factorials_of_digits n = n ↔ n = 145 :=
sorry

end three_digit_sum_of_factorials_eq_l117_117139


namespace correct_conclusions_l117_117090

def pos_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0

def sum_of_n_terms (S a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) * S (n+1) = 9

def second_term_less_than_3 (a S : ℕ → ℝ) : Prop :=
  a 1 < 3

def is_decreasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) < a n

def exists_term_less_than_1_over_100 (a : ℕ → ℝ) : Prop :=
  ∃ n : ℕ, a n < 1/100

theorem correct_conclusions (a S : ℕ → ℝ) :
  pos_sequence a → sum_of_n_terms S a →
  second_term_less_than_3 a S ∧ (¬(∀ q : ℝ, ∃ r : ℝ, ∀ n : ℕ, a n = r * q ^ n)) ∧ is_decreasing_sequence a ∧ exists_term_less_than_1_over_100 a :=
sorry

end correct_conclusions_l117_117090


namespace Vasya_can_win_l117_117485

-- We need this library to avoid any import issues and provide necessary functionality for rational numbers

theorem Vasya_can_win :
  let a := (1 : ℚ) / 2009
  let b := (1 : ℚ) / 2008
  (∃ x : ℚ, a + x = 1) ∨ (∃ x : ℚ, b + x = 1) := sorry

end Vasya_can_win_l117_117485


namespace calc_man_dividend_l117_117309

noncomputable def calc_dividend (investment : ℝ) (face_value : ℝ) (premium : ℝ) (dividend_percent : ℝ) : ℝ :=
  let cost_per_share := face_value * (1 + premium / 100)
  let number_of_shares := investment / cost_per_share
  let dividend_per_share := dividend_percent / 100 * face_value
  let total_dividend := dividend_per_share * number_of_shares
  total_dividend

theorem calc_man_dividend :
  calc_dividend 14400 100 20 5 = 600 :=
by
  sorry

end calc_man_dividend_l117_117309


namespace average_nat_series_l117_117154

theorem average_nat_series : 
  let a := 12  -- first term
  let l := 53  -- last term
  let n := (l - a) / 1 + 1  -- number of terms
  let sum := n / 2 * (a + l)  -- sum of the arithmetic series
  let average := sum / n  -- average of the series
  average = 32.5 :=
by
  let a := 12
  let l := 53
  let n := (l - a) / 1 + 1
  let sum := n / 2 * (a + l)
  let average := sum / n
  sorry

end average_nat_series_l117_117154


namespace scientific_notation_of_0_0000021_l117_117640

theorem scientific_notation_of_0_0000021 :
  0.0000021 = 2.1 * 10 ^ (-6) :=
sorry

end scientific_notation_of_0_0000021_l117_117640


namespace pet_store_cages_l117_117934

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h_initial : initial_puppies = 13) (h_sold : sold_puppies = 7) (h_per_cage : puppies_per_cage = 2) : (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end pet_store_cages_l117_117934


namespace range_x_minus_y_compare_polynomials_l117_117486

-- Proof Problem 1: Range of x - y
theorem range_x_minus_y (x y : ℝ) (hx : -1 < x ∧ x < 4) (hy : 2 < y ∧ y < 3) : 
  -4 < x - y ∧ x - y < 2 := 
  sorry

-- Proof Problem 2: Comparison of polynomials
theorem compare_polynomials (x : ℝ) : 
  (x - 1) * (x^2 + x + 1) < (x + 1) * (x^2 - x + 1) := 
  sorry

end range_x_minus_y_compare_polynomials_l117_117486


namespace least_total_acorns_l117_117767

theorem least_total_acorns :
  ∃ a₁ a₂ a₃ : ℕ,
    (∀ k : ℕ, (∃ a₁ a₂ a₃ : ℕ,
      (2 * a₁ / 3 + a₁ % 3 / 3 + a₂ + a₃ / 9) % 6 = 4 * k ∧
      (a₁ / 6 + a₂ / 3 + a₃ / 3 + 8 * a₃ / 18) % 6 = 3 * k ∧
      (a₁ / 6 + 5 * a₂ / 6 + a₃ / 9) % 6 = 2 * k) → k = 630) ∧
    (a₁ + a₂ + a₃) = 630 :=
sorry

end least_total_acorns_l117_117767


namespace compare_trigonometric_values_l117_117580

noncomputable def a : ℝ := (1 / 2) * real.cos (real.pi / 180 * 7) + (real.sqrt 3 / 2) * real.sin (real.pi / 180 * 7)
noncomputable def b : ℝ := (2 * real.tan (real.pi / 180 * 19)) / (1 - (real.tan (real.pi / 180 * 19))^2)
noncomputable def c : ℝ := real.sqrt ((1 - real.cos (real.pi / 180 * 72)) / 2)

theorem compare_trigonometric_values : b > a ∧ a > c := 
by 
  sorry

end compare_trigonometric_values_l117_117580


namespace find_sample_size_l117_117936

theorem find_sample_size (f r : ℝ) (h1 : f = 20) (h2 : r = 0.125) (h3 : r = f / n) : n = 160 := 
by {
  sorry
}

end find_sample_size_l117_117936


namespace find_p_l117_117568

-- Define the coordinates as given in the problem
def Q : ℝ × ℝ := (0, 15)
def A : ℝ × ℝ := (3, 15)
def B : ℝ × ℝ := (15, 0)
def O : ℝ × ℝ := (0, 0)
def C (p : ℝ) : ℝ × ℝ := (0, p)

-- Defining the function to calculate area of triangle given three points
def area_of_triangle (P1 P2 P3 : ℝ × ℝ) : ℝ :=
  0.5 * abs (P1.fst * (P2.snd - P3.snd) + P2.fst * (P3.snd - P1.snd) + P3.fst * (P1.snd - P2.snd))

-- The statement we need to prove
theorem find_p :
  ∃ p : ℝ, area_of_triangle A B (C p) = 42 ∧ p = 11.75 :=
by
  sorry

end find_p_l117_117568


namespace conic_section_is_parabola_l117_117522

def isParabola (equation : String) : Prop := 
  equation = "|y - 3| = sqrt((x + 4)^2 + (y - 1)^2)"

theorem conic_section_is_parabola : isParabola "|y - 3| = sqrt((x + 4)^2 + (y - 1)^2)" :=
  by
  sorry

end conic_section_is_parabola_l117_117522


namespace classroomA_goal_is_200_l117_117184

def classroomA_fundraising_goal : ℕ :=
  let amount_from_two_families := 2 * 20
  let amount_from_eight_families := 8 * 10
  let amount_from_ten_families := 10 * 5
  let total_raised := amount_from_two_families + amount_from_eight_families + amount_from_ten_families
  let amount_needed := 30
  total_raised + amount_needed

theorem classroomA_goal_is_200 : classroomA_fundraising_goal = 200 := by
  sorry

end classroomA_goal_is_200_l117_117184


namespace car_value_decrease_per_year_l117_117283

theorem car_value_decrease_per_year 
  (initial_value : ℝ) (final_value : ℝ) (years : ℝ) (decrease_per_year : ℝ)
  (h1 : initial_value = 20000)
  (h2 : final_value = 14000)
  (h3 : years = 6)
  (h4 : initial_value - final_value = 6 * decrease_per_year) : 
  decrease_per_year = 1000 :=
sorry

end car_value_decrease_per_year_l117_117283


namespace fish_price_eq_shrimp_price_l117_117497

-- Conditions
variable (x : ℝ) -- regular price for a full pound of fish
variable (h1 : 0.6 * (x / 4) = 1.50) -- quarter-pound fish price after 60% discount
variable (shrimp_price : ℝ) -- price per pound of shrimp
variable (h2 : shrimp_price = 10) -- given shrimp price

-- Proof Statement
theorem fish_price_eq_shrimp_price (h1 : 0.6 * (x / 4) = 1.50) (h2 : shrimp_price = 10) :
  x = 10 ∧ x = shrimp_price :=
by
  sorry

end fish_price_eq_shrimp_price_l117_117497


namespace jeff_boxes_filled_l117_117709

noncomputable def jeff_donuts_per_day : ℕ := 10
noncomputable def number_of_days : ℕ := 12
noncomputable def jeff_eats_per_day : ℕ := 1
noncomputable def chris_eats : ℕ := 8
noncomputable def donuts_per_box : ℕ := 10

theorem jeff_boxes_filled :
  let total_donuts := jeff_donuts_per_day * number_of_days
  let jeff_eats_total := jeff_eats_per_day * number_of_days
  let remaining_donuts_after_jeff := total_donuts - jeff_eats_total
  let remaining_donuts_after_chris := remaining_donuts_after_jeff - chris_eats
  let boxes_filled := remaining_donuts_after_chris / donuts_per_box
  in boxes_filled = 10 :=
by {
  sorry
}

end jeff_boxes_filled_l117_117709


namespace flat_fee_first_night_l117_117628

-- Given conditions
variable (f n : ℝ)
axiom alice_cost : f + 3 * n = 245
axiom bob_cost : f + 5 * n = 350

-- Main theorem to prove
theorem flat_fee_first_night : f = 87.5 := by sorry

end flat_fee_first_night_l117_117628


namespace conversion_200_meters_to_kilometers_l117_117020

noncomputable def meters_to_kilometers (meters : ℕ) : ℝ :=
  meters / 1000

theorem conversion_200_meters_to_kilometers :
  meters_to_kilometers 200 = 0.2 :=
by
  sorry

end conversion_200_meters_to_kilometers_l117_117020


namespace sin_minus_cos_theta_l117_117398

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l117_117398


namespace sin_minus_cos_l117_117399

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l117_117399


namespace curve_left_of_line_l117_117101

theorem curve_left_of_line (x y : ℝ) : x^3 + 2*y^2 = 8 → x ≤ 2 := 
sorry

end curve_left_of_line_l117_117101


namespace cupcakes_frosted_in_10_minutes_l117_117809

-- Definitions representing the given conditions
def CagneyRate := 15 -- seconds per cupcake
def LaceyRate := 40 -- seconds per cupcake
def JessieRate := 30 -- seconds per cupcake
def initialDuration := 3 * 60 -- 3 minutes in seconds
def totalDuration := 10 * 60 -- 10 minutes in seconds
def afterJessieDuration := totalDuration - initialDuration -- 7 minutes in seconds

-- Proof statement
theorem cupcakes_frosted_in_10_minutes : 
  let combinedRateBefore := (CagneyRate * LaceyRate) / (CagneyRate + LaceyRate)
  let combinedRateAfter := (CagneyRate * LaceyRate * JessieRate) / (CagneyRate * LaceyRate + LaceyRate * JessieRate + JessieRate * CagneyRate)
  let cupcakesBefore := initialDuration / combinedRateBefore
  let cupcakesAfter := afterJessieDuration / combinedRateAfter
  cupcakesBefore + cupcakesAfter = 68 :=
by
  sorry

end cupcakes_frosted_in_10_minutes_l117_117809


namespace possible_values_of_x_l117_117341

theorem possible_values_of_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) : x = 4 ∨ x = 6 :=
by
  sorry

end possible_values_of_x_l117_117341


namespace problem_statement_l117_117119

noncomputable def a (k : ℕ) : ℝ := 2^k / (3^(2^k) + 1)
noncomputable def A : ℝ := (Finset.range 10).sum (λ k => a k)
noncomputable def B : ℝ := (Finset.range 10).prod (λ k => a k)

theorem problem_statement : A / B = (3^(2^10) - 1) / 2^47 - 1 / 2^36 := 
by
  sorry

end problem_statement_l117_117119


namespace probability_of_winning_fifth_game_championship_correct_overall_probability_of_winning_championship_correct_l117_117735

noncomputable def binomial (n k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ)

noncomputable def probability_of_winning_fifth_game_championship : ℝ :=
  binomial 4 3 * 0.6^4 * 0.4

noncomputable def overall_probability_of_winning_championship : ℝ :=
  0.6^4 +
  binomial 4 3 * 0.6^4 * 0.4 +
  binomial 5 3 * 0.6^4 * 0.4^2 +
  binomial 6 3 * 0.6^4 * 0.4^3

theorem probability_of_winning_fifth_game_championship_correct :
  probability_of_winning_fifth_game_championship = 0.20736 := by
  sorry

theorem overall_probability_of_winning_championship_correct :
  overall_probability_of_winning_championship = 0.710208 := by
  sorry

end probability_of_winning_fifth_game_championship_correct_overall_probability_of_winning_championship_correct_l117_117735


namespace student_A_final_score_l117_117799

theorem student_A_final_score (total_questions : ℕ) (correct_responses : ℕ) 
  (h1 : total_questions = 100) (h2 : correct_responses = 93) : 
  correct_responses - 2 * (total_questions - correct_responses) = 79 :=
by
  rw [h1, h2]
  -- sorry

end student_A_final_score_l117_117799


namespace solve_quadratic1_solve_quadratic2_l117_117271

open Real

-- Equation 1
theorem solve_quadratic1 (x : ℝ) : x^2 - 6 * x + 8 = 0 → x = 2 ∨ x = 4 := 
by sorry

-- Equation 2
theorem solve_quadratic2 (x : ℝ) : x^2 - 8 * x + 1 = 0 → x = 4 + sqrt 15 ∨ x = 4 - sqrt 15 := 
by sorry

end solve_quadratic1_solve_quadratic2_l117_117271


namespace range_of_b_if_solution_set_contains_1_2_3_l117_117856

theorem range_of_b_if_solution_set_contains_1_2_3 
  (b : ℝ)
  (h : ∀ x : ℝ, |3 * x - b| < 4 ↔ x = 1 ∨ x = 2 ∨ x = 3) :
  5 < b ∧ b < 7 :=
sorry

end range_of_b_if_solution_set_contains_1_2_3_l117_117856


namespace batsman_running_percentage_l117_117479

theorem batsman_running_percentage (total_runs boundary_runs six_runs : ℕ) 
  (h1 : total_runs = 120) (h2 : boundary_runs = 3 * 4) (h3 : six_runs = 8 * 6) : 
  (total_runs - (boundary_runs + six_runs)) * 100 / total_runs = 50 := 
sorry

end batsman_running_percentage_l117_117479


namespace first_grade_muffins_total_l117_117882

theorem first_grade_muffins_total :
  let muffins_brier : ℕ := 218
  let muffins_macadams : ℕ := 320
  let muffins_flannery : ℕ := 417
  let muffins_smith : ℕ := 292
  let muffins_jackson : ℕ := 389
  muffins_brier + muffins_macadams + muffins_flannery + muffins_smith + muffins_jackson = 1636 :=
by
  apply sorry

end first_grade_muffins_total_l117_117882


namespace determinant_of_triangle_angles_l117_117117

theorem determinant_of_triangle_angles (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Matrix.det ![
    ![Real.tan α, Real.sin α * Real.cos α, 1],
    ![Real.tan β, Real.sin β * Real.cos β, 1],
    ![Real.tan γ, Real.sin γ * Real.cos γ, 1]
  ] = 0 :=
by
  -- Proof statement goes here
  sorry

end determinant_of_triangle_angles_l117_117117


namespace imaginary_part_of_z_l117_117080

open Complex

-- Define the context
variables (z : ℂ) (a b : ℂ)

-- Define the condition
def condition := (1 - 2*I) * z = 5 * I

-- Lean 4 statement to prove the imaginary part of z 
theorem imaginary_part_of_z (h : condition z) : z.im = 1 :=
sorry

end imaginary_part_of_z_l117_117080


namespace next_working_day_together_l117_117095

theorem next_working_day_together : 
  let greta_days := 5
  let henry_days := 3
  let linda_days := 9
  let sam_days := 8
  ∃ n : ℕ, n = Nat.lcm (Nat.lcm (Nat.lcm greta_days henry_days) linda_days) sam_days ∧ n = 360 :=
by
  sorry

end next_working_day_together_l117_117095


namespace mall_incur_1_percent_loss_l117_117797

theorem mall_incur_1_percent_loss
  (a b x : ℝ)
  (ha : x = a * 1.1)
  (hb : x = b * 0.9) :
  (2 * x - (a + b)) / (a + b) = -0.01 :=
sorry

end mall_incur_1_percent_loss_l117_117797


namespace d_coverable_condition_no_d_coverable_condition_l117_117596

noncomputable def smallest_d_coverable (n : ℕ) : Option ℕ :=
if n = 4 ∨ Prime n then some (n - 1)
else none

theorem d_coverable_condition (n : ℕ) (hn : n > 1) :
  (∀ S : Finset (Fin n), S.nonempty → ∃ P : Polynomial ℤ, P.natDegree ≤ n - 1 ∧ ∀ x : ℤ, x % n ∈ S ↔ (P.eval x % n) ∈ S) ↔
  smallest_d_coverable n = some (n - 1) :=
sorry

theorem no_d_coverable_condition (n : ℕ) (hn : n > 1) :
  (∀ d : ℕ, ¬ (∀ S : Finset (Fin n), S.nonempty → ∃ P : Polynomial ℤ, P.natDegree ≤ d ∧ ∀ x : ℤ, x % n ∈ S ↔ (P.eval x % n) ∈ S)) ↔
  smallest_d_coverable n = none :=
sorry

end d_coverable_condition_no_d_coverable_condition_l117_117596


namespace sin_minus_cos_eq_l117_117378

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l117_117378


namespace no_half_probability_socks_l117_117760

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l117_117760


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117388

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117388


namespace cos_double_angle_l117_117535

open Real

theorem cos_double_angle (α : ℝ) (h0 : 0 < α ∧ α < π) (h1 : sin α + cos α = 1 / 2) : cos (2 * α) = -sqrt 7 / 4 :=
by
  sorry

end cos_double_angle_l117_117535


namespace sqrt_two_over_two_not_covered_l117_117964

theorem sqrt_two_over_two_not_covered 
  (a b : ℕ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≠ 0) (h4 : Nat.gcd a b = 1) :
  ¬ (real.sqrt 2 / 2 ∈ set.Icc (a / b - 1 / (4 * b ^ 2) : ℝ) (a / b + 1 / (4 * b ^ 2) : ℝ)) :=
sorry

end sqrt_two_over_two_not_covered_l117_117964


namespace sum_of_squares_divisibility_l117_117244

theorem sum_of_squares_divisibility (n : ℤ) : 
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  let S := a^2 + b^2 + c^2
  (S % 4 = 0 ∧ S % 3 ≠ 0) :=
by
  let a := 2 * n
  let b := 2 * n + 2
  let c := 2 * n + 4
  let S := a^2 + b^2 + c^2
  sorry

end sum_of_squares_divisibility_l117_117244


namespace maximum_value_of_function_l117_117337

theorem maximum_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) :
  ∃ M, (∀ y, y = x * (1 - 2 * x) → y ≤ M) ∧ M = 1/8 :=
sorry

end maximum_value_of_function_l117_117337


namespace total_legs_in_room_l117_117149

def count_legs : Nat :=
  let tables_4_legs := 4 * 4
  let sofas_legs := 1 * 4
  let chairs_4_legs := 2 * 4
  let tables_3_legs := 3 * 3
  let tables_1_leg := 1 * 1
  let rocking_chair_legs := 1 * 2
  tables_4_legs + sofas_legs + chairs_4_legs + tables_3_legs + tables_1_leg + rocking_chair_legs

theorem total_legs_in_room : count_legs = 40 := by
  sorry

end total_legs_in_room_l117_117149


namespace solve_investment_problem_l117_117731

def investment_problem
  (total_investment : ℝ) (etf_investment : ℝ) (mutual_funds_factor : ℝ) (mutual_funds_investment : ℝ) : Prop :=
  total_investment = etf_investment + mutual_funds_factor * etf_investment →
  mutual_funds_factor * etf_investment = mutual_funds_investment

theorem solve_investment_problem :
  investment_problem 210000 46666.67 3.5 163333.35 :=
by
  sorry

end solve_investment_problem_l117_117731


namespace min_value_of_x2_y2_sub_xy_l117_117213

theorem min_value_of_x2_y2_sub_xy (x y : ℝ) (h : x^2 + y^2 + x * y = 315) : 
  ∃ m : ℝ, (∀ (u v : ℝ), u^2 + v^2 + u * v = 315 → u^2 + v^2 - u * v ≥ m) ∧ m = 105 :=
sorry

end min_value_of_x2_y2_sub_xy_l117_117213


namespace abs_inequality_solution_l117_117460

theorem abs_inequality_solution (x : ℝ) : (|3 - x| < 4) ↔ (-1 < x ∧ x < 7) :=
by
  sorry

end abs_inequality_solution_l117_117460


namespace find_number_l117_117899

-- Define the problem statement
theorem find_number (n : ℕ) (h : (n + 2 * n + 3 * n + 4 * n + 5 * n) / 5 = 27) : n = 9 :=
sorry

end find_number_l117_117899


namespace solution_eq_l117_117652

theorem solution_eq :
  ∀ x : ℝ, sqrt ((3 + 2 * sqrt 2) ^ x) + sqrt ((3 - 2 * sqrt 2) ^ x) = 6 ↔ (x = 2 ∨ x = -2) := 
by
  intro x
  sorry -- Proof is not required as per instruction

end solution_eq_l117_117652


namespace find_a_2023_l117_117997

def sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, (1 / a n - 1 / a (n + 1) - 1 / (a n * a (n + 1)) = 1)

theorem find_a_2023 (a : ℕ → ℚ) (h : sequence a) : a 2023 = -1 / 2 :=
  sorry

end find_a_2023_l117_117997


namespace angle_C_exceeds_120_degrees_l117_117561

theorem angle_C_exceeds_120_degrees 
  (a b : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = Real.sqrt 3) (c : ℝ) (h_c : c > 3) :
  ∀ (C : ℝ), C = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) 
             → C > 120 :=
by
  sorry

end angle_C_exceeds_120_degrees_l117_117561


namespace divya_age_l117_117695

theorem divya_age (D N : ℝ) (h1 : N + 5 = 3 * (D + 5)) (h2 : N + D = 40) : D = 7.5 :=
by sorry

end divya_age_l117_117695


namespace number_of_marbles_l117_117030

theorem number_of_marbles (T : ℕ) (h1 : 12 ≤ T) : 
  (T - 12) * (T - 12) * 16 = 9 * T * T → T = 48 :=
by
  -- Proof omitted
  sorry

end number_of_marbles_l117_117030


namespace eight_point_shots_count_is_nine_l117_117773

def num_8_point_shots (x y z : ℕ) := 8 * x + 9 * y + 10 * z = 100 ∧
                                      x + y + z > 11 ∧ 
                                      x + y + z ≤ 12 ∧ 
                                      x > 0 ∧ 
                                      y > 0 ∧ 
                                      z > 0

theorem eight_point_shots_count_is_nine : 
  ∃ x y z : ℕ, num_8_point_shots x y z ∧ x = 9 :=
by
  sorry

end eight_point_shots_count_is_nine_l117_117773


namespace round_robin_matches_l117_117516

-- Define the number of players in the tournament
def numPlayers : ℕ := 10

-- Define a function to calculate the number of matches in a round-robin tournament
def calculateMatches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

-- Theorem statement to prove that the number of matches in a 10-person round-robin chess tournament is 45
theorem round_robin_matches : calculateMatches numPlayers = 45 := by
  sorry

end round_robin_matches_l117_117516


namespace grade_A_probability_l117_117796

theorem grade_A_probability
  (P_B : ℝ) (P_C : ℝ)
  (hB : P_B = 0.05)
  (hC : P_C = 0.03) :
  1 - P_B - P_C = 0.92 :=
by
  sorry

end grade_A_probability_l117_117796


namespace wand_cost_l117_117925

theorem wand_cost (c : ℕ) (h1 : 3 * c = 3 * c) (h2 : 2 * (c + 5) = 130) : c = 60 :=
by
  sorry

end wand_cost_l117_117925


namespace bianca_drawing_time_at_home_l117_117808

-- Define the conditions
def drawing_time_at_school : ℕ := 22
def total_drawing_time : ℕ := 41

-- Define the calculation for drawing time at home
def drawing_time_at_home : ℕ := total_drawing_time - drawing_time_at_school

-- The proof goal
theorem bianca_drawing_time_at_home : drawing_time_at_home = 19 := by
  sorry

end bianca_drawing_time_at_home_l117_117808


namespace find_M_l117_117818

theorem find_M (M : ℕ) (h1 : M > 0) (h2 : M < 10) : 
  5 ∣ (1989^M + M^1989) ↔ M = 1 ∨ M = 4 := by
  sorry

end find_M_l117_117818


namespace largest_value_after_2001_presses_l117_117173

noncomputable def max_value_after_presses (n : ℕ) : ℝ :=
if n = 0 then 1 else sorry -- Placeholder for the actual function definition

theorem largest_value_after_2001_presses :
  max_value_after_presses 2001 = 1 :=
sorry

end largest_value_after_2001_presses_l117_117173


namespace new_marketing_percentage_l117_117943

theorem new_marketing_percentage 
  (total_students : ℕ)
  (initial_finance_percentage : ℕ)
  (initial_marketing_percentage : ℕ)
  (initial_operations_management_percentage : ℕ)
  (new_finance_percentage : ℕ)
  (operations_management_percentage : ℕ)
  (total_percentage : ℕ) :
  total_students = 5000 →
  initial_finance_percentage = 85 →
  initial_marketing_percentage = 80 →
  initial_operations_management_percentage = 10 →
  new_finance_percentage = 92 →
  operations_management_percentage = 10 →
  total_percentage = 175 →
  initial_marketing_percentage - (new_finance_percentage - initial_finance_percentage) = 73 :=
by
  sorry

end new_marketing_percentage_l117_117943


namespace solve_for_x_l117_117785

theorem solve_for_x (x : ℤ) (h : x + 1 = 10) : x = 9 := 
by 
  sorry

end solve_for_x_l117_117785


namespace triangle_angle_a_value_triangle_side_a_value_l117_117107

open Real

theorem triangle_angle_a_value (a b c A B C : ℝ) 
  (h1 : (a - c) * (a + c) * sin C = c * (b - c) * sin B)
  (h2 : (1/2) * b * c * sin A = sqrt 3)
  (h3 : sin B * sin C = 1/4) :
  A = π / 3 :=
sorry

theorem triangle_side_a_value (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : (a - c) * (a + c) * sin C = c * (b - c) * sin B)
  (h2 : (1/2) * b * c * sin(A) = sqrt 3)
  (h3 : sin B * sin C = 1/4)
  (h4 : A = π / 3) :
  a = 2 * sqrt 3 :=
sorry

end triangle_angle_a_value_triangle_side_a_value_l117_117107


namespace tangent_line_at_origin_is_minus_3x_l117_117716

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 3) * x
noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 3)

theorem tangent_line_at_origin_is_minus_3x (a : ℝ) (h : ∀ x : ℝ, f_prime a x = f_prime a (-x)) : 
  (f_prime 0 0 = -3) → ∀ x : ℝ, (f a x = -3 * x) :=
by
  sorry

end tangent_line_at_origin_is_minus_3x_l117_117716


namespace at_least_one_real_root_l117_117843

theorem at_least_one_real_root (a : ℝ) :
  (4*a)^2 - 4*(-4*a + 3) ≥ 0 ∨
  ((a - 1)^2 - 4*a^2) ≥ 0 ∨
  (2*a)^2 - 4*(-2*a) ≥ 0 := sorry

end at_least_one_real_root_l117_117843


namespace sin_minus_cos_l117_117364

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l117_117364


namespace Aaron_final_cards_l117_117187

-- Definitions from conditions
def initial_cards_Aaron : Nat := 5
def found_cards_Aaron : Nat := 62

-- Theorem statement
theorem Aaron_final_cards : initial_cards_Aaron + found_cards_Aaron = 67 :=
by
  sorry

end Aaron_final_cards_l117_117187


namespace intersection_point_a_l117_117781

theorem intersection_point_a : ∃ (x y : ℝ), y = 4 * x - 32 ∧ y = -6 * x + 8 ∧ x = 4 ∧ y = -16 :=
sorry

end intersection_point_a_l117_117781


namespace find_annual_growth_rate_eq_50_perc_estimate_2023_export_l117_117885

open Real

-- Conditions
def initial_export_volume (year : ℕ) := 
  if year = 2020 then 200000 else 0 

def export_volume_2022 := 450000

-- Definitions
def annual_average_growth_rate (v0 v2 : ℝ) (x : ℝ) :=
  v0 * (1 + x)^2 = v2

-- Proof statement
theorem find_annual_growth_rate_eq_50_perc :
  ∃ x : ℝ, annual_average_growth_rate 200000 450000 x ∧ 0 <= x ∧ x = 0.5 :=
by
  use 0.5
  have h : 200000 * (1 + 0.5)^2 = 450000 := by linarith
  exact ⟨h, by linarith, rfl⟩
  sorry

-- Second theorem
theorem estimate_2023_export (v2 : ℕ) (x : ℝ) (expected : ℕ) :
  v2 = export_volume_2022 →
  x = 0.5 →
  expected = v2 * (1 + x) →
  expected = 675000 :=
by
  intros h₁ h₂ h₃
  rw h₁ at *
  rw h₂ at *
  simp at h₃
  exact h₃
  sorry

end find_annual_growth_rate_eq_50_perc_estimate_2023_export_l117_117885


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117391

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117391


namespace find_58th_digit_in_fraction_l117_117290

def decimal_representation_of_fraction : ℕ := sorry

theorem find_58th_digit_in_fraction:
  (decimal_representation_of_fraction = 4) := sorry

end find_58th_digit_in_fraction_l117_117290


namespace ratio_a_to_c_l117_117994

variable (a b c : ℕ)

theorem ratio_a_to_c (h1 : a / b = 8 / 3) (h2 : b / c = 1 / 5) : a / c = 8 / 15 := 
sorry

end ratio_a_to_c_l117_117994


namespace remainder_of_15_add_y_mod_31_l117_117875

theorem remainder_of_15_add_y_mod_31 (y : ℕ) (h : 7 * y ≡ 1 [MOD 31]) : (15 + y) % 31 = 24 :=
by
  have y_value : y = 9 := by
    -- proof details for y = 9 skipped with "sorry"
    sorry
  rw [y_value]
  norm_num

end remainder_of_15_add_y_mod_31_l117_117875


namespace sum_of_areas_of_two_squares_l117_117656

theorem sum_of_areas_of_two_squares (a b : ℕ) (h1 : a = 8) (h2 : b = 10) :
  a * a + b * b = 164 := by
  sorry

end sum_of_areas_of_two_squares_l117_117656


namespace sin_minus_cos_theta_l117_117397

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l117_117397


namespace solve_system_l117_117068

theorem solve_system (x y : ℚ) (h1 : 6 * x = -9 - 3 * y) (h2 : 4 * x = 5 * y - 34) : x = 1/2 ∧ y = -4 :=
by
  sorry

end solve_system_l117_117068


namespace arithmetic_progression_terms_even_l117_117861

variable (a d : ℝ) (n : ℕ)

open Real

theorem arithmetic_progression_terms_even {n : ℕ} (hn_even : n % 2 = 0)
  (h_sum_odd : (n / 2 : ℝ) * (2 * a + (n - 2) * d) = 32)
  (h_sum_even : (n / 2 : ℝ) * (2 * a + 2 * d + (n - 2) * d) = 40)
  (h_last_exceeds_first : (a + (n - 1) * d) - a = 8) : n = 16 :=
sorry

end arithmetic_progression_terms_even_l117_117861


namespace solution_set_of_inequality_l117_117609

theorem solution_set_of_inequality : 
  { x : ℝ | (x + 2) * (1 - x) > 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l117_117609


namespace triangle_PQR_PR_value_l117_117705

theorem triangle_PQR_PR_value (PQ QR PR : ℕ) (h1 : PQ = 7) (h2 : QR = 20) (h3 : 13 < PR) (h4 : PR < 27) : PR = 21 :=
by sorry

end triangle_PQR_PR_value_l117_117705


namespace sector_area_l117_117840

theorem sector_area (α r : ℝ) (hα : α = π / 3) (hr : r = 2) : 
  1 / 2 * α * r^2 = 2 * π / 3 := 
by 
  rw [hα, hr] 
  simp 
  sorry

end sector_area_l117_117840


namespace evaluate_g_expressions_l117_117199

def g (x : ℝ) : ℝ := 3 * x^2 - 5 * x + 7

theorem evaluate_g_expressions : 3 * g 5 + 4 * g (-2) = 287 := by
  sorry

end evaluate_g_expressions_l117_117199


namespace no_solutions_exist_l117_117334

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 2 :=
by sorry

end no_solutions_exist_l117_117334


namespace initial_students_per_class_l117_117308

theorem initial_students_per_class
  (S : ℕ) 
  (parents chaperones left_students left_chaperones : ℕ)
  (teachers remaining_individuals : ℕ)
  (h1 : parents = 5)
  (h2 : chaperones = 2)
  (h3 : left_students = 10)
  (h4 : left_chaperones = 2)
  (h5 : teachers = 2)
  (h6 : remaining_individuals = 15)
  (h7 : 2 * S + parents + teachers - left_students - left_chaperones = remaining_individuals) :
  S = 10 :=
by
  sorry

end initial_students_per_class_l117_117308


namespace length_of_platform_l117_117800

theorem length_of_platform (length_train speed_train time_crossing speed_train_mps distance_train_cross : ℝ)
  (h1 : length_train = 120)
  (h2 : speed_train = 60)
  (h3 : time_crossing = 20)
  (h4 : speed_train_mps = 16.67)
  (h5 : distance_train_cross = speed_train_mps * time_crossing):
  (distance_train_cross = length_train + 213.4) :=
by
  sorry

end length_of_platform_l117_117800


namespace find_p_q_sum_l117_117941

-- Define the equation of the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 + 9*y^2 = 9

-- Define the conditions
def is_equilateral_triangle (A B C : (ℝ × ℝ)) : Prop :=
  let (Ax, Ay) := A;
  let (Bx, By) := B;
  let (Cx, Cy) := C;
  (Ax = 0 ∧ Ay = 1) ∧
  ((Bx = -Cx ∧ By = Cy) ∧ 
   (Bx ≠ 0 ∧ Cy ≠ 1) ∧ 
   Ax^2 + 9*Ay^2 = 9 ∧
   Bx^2 + 9*By^2 = 9 ∧
   Cx^2 + 9*Cy^2 = 9 ∧
   -- Condition for the triangle to be equilateral
   ((Bx - Ax)^2 + (By - Ay)^2 = (Cx - Bx)^2 + (Cy - By)^2) ∧
   ((Cx - Ax)^2 + (Cy - Ay)^2 = (Bx - Ax)^2 + (By - Ay)^2))

-- Define that the length of the side squared is p/q
def length_squared_is_p_div_q (A B : (ℝ × ℝ)) (p q : ℕ) : Prop :=
  let (Ax, Ay) := A;
  let (Bx, By) := B;
  (Ax - Bx)^2 + (Ay - By)^2 = (p : ℝ) / (q : ℝ) ∧ Nat.gcd p q = 1 

-- The main statement to be proven 
theorem find_p_q_sum (A B C : (ℝ × ℝ)) (p q : ℕ) :
  is_equilateral_triangle A B C →
  length_squared_is_p_div_q A B p q →
  p + q = 292 :=
by
  intro h_triangle h_length
  sorry

end find_p_q_sum_l117_117941


namespace total_money_raised_l117_117180

def tickets_sold : ℕ := 25
def price_per_ticket : ℝ := 2.0
def donation_count : ℕ := 2
def donation_amount : ℝ := 15.0
def additional_donation : ℝ := 20.0

theorem total_money_raised :
  (tickets_sold * price_per_ticket) + (donation_count * donation_amount) + additional_donation = 100 :=
by
  sorry

end total_money_raised_l117_117180


namespace jason_needs_to_buy_guppies_l117_117707

theorem jason_needs_to_buy_guppies 
  (moray_eel_guppies : ℕ) 
  (num_betta_fish : ℕ) 
  (betta_fish_guppies : ℕ) 
  (h_moray_eel : moray_eel_guppies = 20)
  (h_num_betta_fish : num_betta_fish = 5)
  (h_betta_fish : betta_fish_guppies = 7)
  : moray_eel_guppies + num_betta_fish * betta_fish_guppies = 55 :=
by 
  rw [h_moray_eel, h_num_betta_fish, h_betta_fish]
  sorry

end jason_needs_to_buy_guppies_l117_117707


namespace quadratic_inequality_solution_l117_117969

theorem quadratic_inequality_solution (x : ℝ) : 
    (x^2 - 3*x - 4 > 0) ↔ (x < -1 ∨ x > 4) :=
sorry

end quadratic_inequality_solution_l117_117969


namespace enclosed_area_eq_32_over_3_l117_117069

def line (x : ℝ) : ℝ := 2 * x + 3
def parabola (x : ℝ) : ℝ := x^2

theorem enclosed_area_eq_32_over_3 :
  ∫ x in (-(1:ℝ))..(3:ℝ), (line x - parabola x) = 32 / 3 :=
by
  sorry

end enclosed_area_eq_32_over_3_l117_117069


namespace compare_fractions_l117_117835

variable {a b c d : ℝ}

theorem compare_fractions (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) :
  (b / (a - c)) < (a / (b - d)) := 
by
  sorry

end compare_fractions_l117_117835


namespace inverse_undefined_at_one_l117_117552

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 5)

theorem inverse_undefined_at_one : ∀ (x : ℝ), (x = 1) → ¬∃ y : ℝ, f y = x :=
by
  sorry

end inverse_undefined_at_one_l117_117552


namespace integers_sum_21_l117_117014

theorem integers_sum_21 : ∃ (m n : ℕ), m * n + m + n = 125 ∧ Int.gcd m n = 1 ∧ m < 30 ∧ n < 30 ∧ |m - n| ≤ 5 ∧ m + n = 21 :=
by
  sorry

end integers_sum_21_l117_117014


namespace green_hats_count_l117_117018

theorem green_hats_count : ∃ G B : ℕ, B + G = 85 ∧ 6 * B + 7 * G = 540 ∧ G = 30 :=
by
  sorry

end green_hats_count_l117_117018


namespace total_figurines_l117_117315

theorem total_figurines:
  let basswood_blocks := 25
  let butternut_blocks := 30
  let aspen_blocks := 35
  let oak_blocks := 40
  let cherry_blocks := 45
  let basswood_figs_per_block := 3
  let butternut_figs_per_block := 4
  let aspen_figs_per_block := 2 * basswood_figs_per_block
  let oak_figs_per_block := 5
  let cherry_figs_per_block := 7
  let basswood_total := basswood_blocks * basswood_figs_per_block
  let butternut_total := butternut_blocks * butternut_figs_per_block
  let aspen_total := aspen_blocks * aspen_figs_per_block
  let oak_total := oak_blocks * oak_figs_per_block
  let cherry_total := cherry_blocks * cherry_figs_per_block
  let total_figs := basswood_total + butternut_total + aspen_total + oak_total + cherry_total
  total_figs = 920 := by sorry

end total_figurines_l117_117315


namespace product_divisible_by_49_l117_117572

theorem product_divisible_by_49 (a b : ℕ) (h : (a^2 + b^2) % 7 = 0) : (a * b) % 49 = 0 :=
sorry

end product_divisible_by_49_l117_117572


namespace min_value_of_reciprocal_l117_117211

theorem min_value_of_reciprocal (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y = 2) :
  (∀ r, r = 1 / x + 1 / y → r ≥ 3 / 2 + Real.sqrt 2) :=
by
  sorry

end min_value_of_reciprocal_l117_117211


namespace range_of_m_for_inequality_l117_117026

theorem range_of_m_for_inequality (m : Real) : 
  (∀ (x : Real), 1 < x ∧ x < 2 → x^2 + m * x + 4 < 0) ↔ m ≤ -5 :=
by sorry

end range_of_m_for_inequality_l117_117026


namespace probability_two_red_cards_l117_117794

theorem probability_two_red_cards : 
  let total_cards := 100;
  let red_cards := 50;
  let black_cards := 50;
  (red_cards / total_cards : ℝ) * ((red_cards - 1) / (total_cards - 1) : ℝ) = 49 / 198 := 
by
  sorry

end probability_two_red_cards_l117_117794


namespace jane_buys_bagels_l117_117814

variable (b m : ℕ)
variable (h1 : b + m = 7)
variable (h2 : 65 * b + 40 * m % 100 = 80)
variable (h3 : 40 * b + 40 * m % 100 = 0)

theorem jane_buys_bagels : b = 4 := by sorry

end jane_buys_bagels_l117_117814


namespace prime_square_mod_12_l117_117888

theorem prime_square_mod_12 (p : ℕ) (hp_prime : Nat.Prime p) (hp_gt_three : p > 3) : 
  (p ^ 2) % 12 = 1 :=
sorry

end prime_square_mod_12_l117_117888


namespace solve_for_x_l117_117972

theorem solve_for_x (x y : ℕ) (h1 : x / y = 15 / 5) (h2 : y = 25) : x = 75 := 
by
  sorry

end solve_for_x_l117_117972


namespace hyperbola_foci_distance_l117_117820

theorem hyperbola_foci_distance :
  let a := Real.sqrt 25
  let b := Real.sqrt 9
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let distance := 2 * c
  distance = 2 * Real.sqrt 34 :=
by
  let a := Real.sqrt 25
  let b := Real.sqrt 9
  let c := Real.sqrt (a ^ 2 + b ^ 2)
  let distance := 2 * c
  exact sorry

end hyperbola_foci_distance_l117_117820


namespace neznaika_incorrect_l117_117302

-- Define the average consumption conditions
def average_consumption_december (total_consumption total_days_cons_december : ℕ) : Prop :=
  total_consumption = 10 * total_days_cons_december

def average_consumption_january (total_consumption total_days_cons_january : ℕ) : Prop :=
  total_consumption = 5 * total_days_cons_january

-- Define the claim to be disproven
def neznaika_claim (days_december_at_least_10 days_january_at_least_10 : ℕ) : Prop :=
  days_december_at_least_10 > days_january_at_least_10

-- Proof statement that the claim is incorrect
theorem neznaika_incorrect (total_days_cons_december total_days_cons_january total_consumption_dec total_consumption_jan : ℕ)
    (days_december_at_least_10 days_january_at_least_10 : ℕ)
    (h1 : average_consumption_december total_consumption_dec total_days_cons_december)
    (h2 : average_consumption_january total_consumption_jan total_days_cons_january)
    (h3 : total_days_cons_december = 31)
    (h4 : total_days_cons_january = 31)
    (h5 : days_december_at_least_10 ≤ total_days_cons_december)
    (h6 : days_january_at_least_10 ≤ total_days_cons_january)
    (h7 : days_december_at_least_10 = 1)
    (h8 : days_january_at_least_10 = 15) : 
    ¬ neznaika_claim days_december_at_least_10 days_january_at_least_10 :=
by
  sorry

end neznaika_incorrect_l117_117302


namespace number_of_chairs_l117_117660

theorem number_of_chairs (x t c b T C B: ℕ) (r1 r2 r3: ℕ)
  (h1: x = 2250) (h2: t = 18) (h3: c = 12) (h4: b = 30) 
  (h5: r1 = 2) (h6: r2 = 3) (h7: r3 = 1) 
  (h_ratio1: T / C = r1 / r2) (h_ratio2: B / C = r3 / r2) 
  (h_eq: t * T + c * C + b * B = x) : C = 66 :=
by
  sorry

end number_of_chairs_l117_117660


namespace tree_height_at_2_years_l117_117502

theorem tree_height_at_2_years (h : ℕ → ℕ) 
  (h_growth : ∀ n, h (n + 1) = 3 * h n) 
  (h_5 : h 5 = 243) : 
  h 2 = 9 := 
sorry

end tree_height_at_2_years_l117_117502


namespace problem_l117_117434

theorem problem : (1 * (2 + 3) * 4 * 5) = 100 := by
  sorry

end problem_l117_117434


namespace find_k_for_circle_radius_5_l117_117201

theorem find_k_for_circle_radius_5 (k : ℝ) :
  (∃ x y : ℝ, (x^2 + 12 * x + y^2 + 8 * y - k = 0)) → k = -27 :=
by
  sorry

end find_k_for_circle_radius_5_l117_117201


namespace max_value_of_b_over_a_squared_l117_117113

variables {a b x y : ℝ}

def triangle_is_right (a b x y : ℝ) : Prop :=
  (a - x)^2 + (b - y)^2 = a^2 + b^2

theorem max_value_of_b_over_a_squared (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≥ b)
    (h4 : ∃ x y, a^2 + y^2 = b^2 + x^2 
                 ∧ b^2 + x^2 = (a - x)^2 + (b - y)^2
                 ∧ 0 ≤ x ∧ x < a 
                 ∧ 0 ≤ y ∧ y < b 
                 ∧ triangle_is_right a b x y) 
    : (b / a)^2 = 4 / 3 :=
sorry

end max_value_of_b_over_a_squared_l117_117113


namespace no_blue_socks_make_probability_half_l117_117745

theorem no_blue_socks_make_probability_half :
  ∀ (n m : ℕ), n + m = 2009 → (n - m) * (n - m) ≠ 2009 :=
begin
  intros n m h,
  by_contra H,
  sorry, -- Proof goes here
end

end no_blue_socks_make_probability_half_l117_117745


namespace value_of_expression_l117_117159

theorem value_of_expression : 3^2 * 5 * 7^2 * 11 = 24255 := by
  have h1 : 3^2 = 9 := by norm_num
  have h2 : 7^2 = 49 := by norm_num
  calc
    3^2 * 5 * 7^2 * 11
        = 9 * 5 * 7^2 * 11 : by rw h1
    ... = 9 * 5 * 49 * 11  : by rw h2
    ... = 24255            : by norm_num

end value_of_expression_l117_117159


namespace union_sets_l117_117092

open Set

variable {α : Type*}

def A : Set ℝ := {x | -2 < x ∧ x < 2}

def B : Set ℝ := {y | ∃ x, x ∈ A ∧ y = 2^x}

theorem union_sets : A ∪ B = {z | -2 < z ∧ z < 4} :=
by sorry

end union_sets_l117_117092


namespace range_of_a_l117_117542

theorem range_of_a 
  (f : ℝ → ℝ)
  (h_even : ∀ x, -5 ≤ x ∧ x ≤ 5 → f x = f (-x))
  (h_decreasing : ∀ a b, 0 ≤ a ∧ a < b ∧ b ≤ 5 → f b < f a)
  (h_inequality : ∀ a, f (2 * a + 3) < f a) :
  ∀ a, -5 ≤ a ∧ a ≤ 5 → a ∈ (Set.Icc (-4) (-3) ∪ Set.Ioc (-1) 1) := 
by
  sorry

end range_of_a_l117_117542


namespace equivalent_lemons_l117_117897

theorem equivalent_lemons 
  (lemons_per_apple_approx : ∀ apples : ℝ, 3/4 * 14 = 9 → 1 = 9 / (3/4 * 14))
  (apples_to_lemons : ℝ) :
  5 / 7 * 7 = 30 / 7 :=
by
  sorry

end equivalent_lemons_l117_117897


namespace sum_of_angles_of_parallelepiped_diagonal_lt_pi_l117_117007

/-- In a rectangular parallelepiped, if the main diagonal forms angles α, β, and γ with the three edges meeting at a vertex, then the sum of these angles is less than π. -/
theorem sum_of_angles_of_parallelepiped_diagonal_lt_pi {α β γ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ)
  (h_sum : 2 * α + 2 * β + 2 * γ < 2 * π) :
  α + β + γ < π := by
sorry

end sum_of_angles_of_parallelepiped_diagonal_lt_pi_l117_117007


namespace b_share_220_l117_117729

theorem b_share_220 (A B C : ℝ) (h1 : A = B + 40) (h2 : C = A + 30) (h3 : A + B + C = 770) : 
  B = 220 :=
by
  sorry

end b_share_220_l117_117729


namespace find_k_l117_117355

theorem find_k (k : ℤ) (h1 : |k| = 1) (h2 : k - 1 ≠ 0) : k = -1 :=
by
  sorry

end find_k_l117_117355


namespace inequality_for_positive_real_numbers_l117_117258

theorem inequality_for_positive_real_numbers 
  (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) :
  (a / (b + 2 * c + 3 * d) + 
   b / (c + 2 * d + 3 * a) + 
   c / (d + 2 * a + 3 * b) + 
   d / (a + 2 * b + 3 * c)) ≥ (2 / 3) :=
by
  sorry

end inequality_for_positive_real_numbers_l117_117258


namespace circle_radius_l117_117279

theorem circle_radius (x y : ℝ) : 
  (x^2 + y^2 - 2 * x + 6 * y + 1 = 0) → (∃ (r : ℝ), r = 3) :=
by
  sorry

end circle_radius_l117_117279


namespace average_speed_round_trip_l117_117595

theorem average_speed_round_trip (D : ℝ) (hD : D > 0) :
  let time_uphill := D / 5
  let time_downhill := D / 100
  let total_distance := 2 * D
  let total_time := time_uphill + time_downhill
  let average_speed := total_distance / total_time
  average_speed = 200 / 21 :=
by
  sorry

end average_speed_round_trip_l117_117595


namespace find_divisor_l117_117476

theorem find_divisor (d x k j : ℤ) (h₁ : x = k * d + 5) (h₂ : 7 * x = j * d + 8) : d = 11 :=
sorry

end find_divisor_l117_117476


namespace measure_of_C_l117_117287

-- Define angles and their magnitudes
variables (A B C X : Type) [LinearOrder C]
def angle_measure (angle : Type) : ℕ := sorry
def parallel (l1 l2 : Type) : Prop := sorry
def transversal (l1 l2 l3 : Type) : Prop := sorry
def alternate_interior (angle1 angle2 : Type) : Prop := sorry
def adjacent (angle1 angle2 : Type) : Prop := sorry
def complementary (angle1 angle2 : Type) : Prop := sorry

-- The given conditions
axiom h1 : parallel A X
axiom h2 : transversal A B X
axiom h3 : angle_measure A = 85
axiom h4 : angle_measure B = 35
axiom h5 : alternate_interior C A
axiom h6 : complementary B X
axiom h7 : adjacent C X

-- Define the proof problem
theorem measure_of_C : angle_measure C = 85 :=
by {
  -- The proof goes here, skipping with sorry
  sorry
}

end measure_of_C_l117_117287


namespace arithmetic_sequence_ninth_term_eq_l117_117902

theorem arithmetic_sequence_ninth_term_eq :
  let a_1 := (3 : ℚ) / 8
  let a_17 := (2 : ℚ) / 3
  let a_9 := (a_1 + a_17) / 2
  a_9 = (25 : ℚ) / 48 := by
  let a_1 := (3 : ℚ) / 8
  let a_17 := (2 : ℚ) / 3
  let a_9 := (a_1 + a_17) / 2
  sorry

end arithmetic_sequence_ninth_term_eq_l117_117902


namespace students_neither_cs_nor_elec_l117_117234

theorem students_neither_cs_nor_elec
  (total_students : ℕ)
  (cs_students : ℕ)
  (elec_students : ℕ)
  (both_cs_and_elec : ℕ)
  (h_total : total_students = 150)
  (h_cs : cs_students = 90)
  (h_elec : elec_students = 60)
  (h_both : both_cs_and_elec = 20) :
  (total_students - (cs_students + elec_students - both_cs_and_elec) = 20) :=
by
  sorry

end students_neither_cs_nor_elec_l117_117234


namespace probability_four_dots_collinear_l117_117865

theorem probability_four_dots_collinear (dots : Finset (Fin 5 × Fin 5)) (h_dot_count : dots.card = 25) :
  let total_ways := (Finset.choose 25 4)
  let collinear_sets := 12
  let total_combinations := total_ways
  let probability := (collinear_sets : ℚ) / total_combinations
  probability = 6 / 6325 :=
by
  sorry

end probability_four_dots_collinear_l117_117865


namespace meaningful_expression_range_l117_117558

theorem meaningful_expression_range (a : ℝ) : (a + 1 ≥ 0) ∧ (a ≠ 2) ↔ (a ≥ -1) ∧ (a ≠ 2) :=
by
  sorry

end meaningful_expression_range_l117_117558


namespace suitable_comprehensive_survey_l117_117028

-- Definitions based on conditions

def heights_of_students (n : Nat) : Prop := n = 45
def disease_rate_wheat (area : Type) : Prop := True
def love_for_chrysanthemums (population : Type) : Prop := True
def food_safety_hotel (time : Type) : Prop := True

-- The theorem to prove

theorem suitable_comprehensive_survey : 
  (heights_of_students 45 → True) ∧ 
  (disease_rate_wheat ℕ → False) ∧ 
  (love_for_chrysanthemums ℕ → False) ∧ 
  (food_safety_hotel ℕ → False) →
  heights_of_students 45 :=
by
  intros
  sorry

end suitable_comprehensive_survey_l117_117028


namespace volume_of_A_is_2800_l117_117045

-- Define the dimensions of the fishbowl and water heights
def fishbowl_side_length : ℝ := 20
def height_with_A : ℝ := 16
def height_without_A : ℝ := 9

-- Compute the volume of water with and without object (A)
def volume_with_A : ℝ := fishbowl_side_length ^ 2 * height_with_A
def volume_without_A : ℝ := fishbowl_side_length ^ 2 * height_without_A

-- The volume of object (A)
def volume_A : ℝ := volume_with_A - volume_without_A

-- Prove that this volume is 2800 cubic centimeters
theorem volume_of_A_is_2800 :
  volume_A = 2800 := by
  sorry

end volume_of_A_is_2800_l117_117045


namespace wand_cost_l117_117928

-- Conditions based on the problem
def initialWands := 3
def salePrice (x : ℝ) := x + 5
def totalCollected := 130
def soldWands := 2

-- Proof statement
theorem wand_cost (x : ℝ) : 
  2 * salePrice x = totalCollected → x = 60 := 
by 
  sorry

end wand_cost_l117_117928


namespace bacteria_mass_at_4pm_l117_117190

theorem bacteria_mass_at_4pm 
  (r s t u v w : ℝ)
  (x y z : ℝ)
  (h1 : x = 10.0 * (1 + r))
  (h2 : y = 15.0 * (1 + s))
  (h3 : z = 8.0 * (1 + t))
  (h4 : 28.9 = x * (1 + u))
  (h5 : 35.5 = y * (1 + v))
  (h6 : 20.1 = z * (1 + w)) :
  x = 28.9 / (1 + u) ∧ y = 35.5 / (1 + v) ∧ z = 20.1 / (1 + w) :=
by
  sorry

end bacteria_mass_at_4pm_l117_117190


namespace equation_solutions_equiv_l117_117715

theorem equation_solutions_equiv (p : ℕ) (hp : p.Prime) :
  (∃ x s : ℤ, x^2 - x + 3 - p * s = 0) ↔ 
  (∃ y t : ℤ, y^2 - y + 25 - p * t = 0) :=
by { sorry }

end equation_solutions_equiv_l117_117715


namespace find_q_l117_117720

theorem find_q (a b m p q : ℚ) (h1 : a * b = 3) (h2 : a + b = m) 
  (h3 : (a + 1/b) * (b + 1/a) = q) : 
  q = 13 / 3 := by
  sorry

end find_q_l117_117720


namespace anna_original_money_l117_117804

theorem anna_original_money (x : ℝ) (h : (3 / 4) * x = 24) : x = 32 :=
by
  sorry

end anna_original_money_l117_117804


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117387

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117387


namespace neg_q_is_true_l117_117541

variable (p q : Prop)

theorem neg_q_is_true (hp : p) (hq : ¬ q) : ¬ q :=
by
  exact hq

end neg_q_is_true_l117_117541


namespace probability_three_consecutive_cards_l117_117881

-- Definitions of the conditions
def total_ways_to_draw_three : ℕ := Nat.choose 52 3

def sets_of_consecutive_ranks : ℕ := 10

def ways_to_choose_three_consecutive : ℕ := 64

def favorable_outcomes : ℕ := sets_of_consecutive_ranks * ways_to_choose_three_consecutive

def probability_consecutive_ranks : ℚ := favorable_outcomes / total_ways_to_draw_three

-- The main statement to prove
theorem probability_three_consecutive_cards :
  probability_consecutive_ranks = 32 / 1105 := 
sorry

end probability_three_consecutive_cards_l117_117881


namespace tree_height_at_2_years_l117_117503

theorem tree_height_at_2_years (h : ℕ → ℕ) 
  (h_growth : ∀ n, h (n + 1) = 3 * h n) 
  (h_5 : h 5 = 243) : 
  h 2 = 9 := 
sorry

end tree_height_at_2_years_l117_117503


namespace conference_problem_l117_117191

noncomputable def exists_round_table (n : ℕ) (scientists : Finset ℕ) (acquaintance : ℕ → Finset ℕ) : Prop :=
  ∃ (A B C D : ℕ), A ∈ scientists ∧ B ∈ scientists ∧ C ∈ scientists ∧ D ∈ scientists ∧
  ((A ≠ B ∧ A ≠ C ∧ A ≠ D) ∧ (B ≠ C ∧ B ≠ D) ∧ (C ≠ D)) ∧
  (B ∈ acquaintance A ∧ C ∈ acquaintance B ∧ D ∈ acquaintance C ∧ A ∈ acquaintance D)

theorem conference_problem :
  ∀ (scientists : Finset ℕ),
  ∀ (acquaintance : ℕ → Finset ℕ),
    (scientists.card = 50) →
    (∀ s ∈ scientists, (acquaintance s).card ≥ 25) →
    exists_round_table 50 scientists acquaintance :=
sorry

end conference_problem_l117_117191


namespace sin_minus_cos_theta_l117_117393

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l117_117393


namespace ellipse_product_l117_117257

noncomputable def computeProduct (a b : ℝ) : ℝ :=
  let AB := 2 * a
  let CD := 2 * b
  AB * CD

theorem ellipse_product (a b : ℝ) (h1 : a^2 - b^2 = 64) (h2 : a - b = 4) :
  computeProduct a b = 240 := by
sorry

end ellipse_product_l117_117257


namespace tylenol_interval_l117_117252

/-- Mark takes 2 Tylenol tablets of 500 mg each at certain intervals for 12 hours, and he ends up taking 3 grams of Tylenol in total. Prove that the interval in hours at which he takes the tablets is 2.4 hours. -/
theorem tylenol_interval 
    (total_dose_grams : ℝ)
    (tablet_mg : ℝ)
    (hours : ℝ)
    (tablets_taken_each_time : ℝ) 
    (total_tablets : ℝ) 
    (interval_hours : ℝ) :
    total_dose_grams = 3 → 
    tablet_mg = 500 → 
    hours = 12 → 
    tablets_taken_each_time = 2 → 
    total_tablets = (total_dose_grams * 1000) / tablet_mg → 
    interval_hours = hours / (total_tablets / tablets_taken_each_time - 1) → 
    interval_hours = 2.4 :=
by
  intros
  sorry

end tylenol_interval_l117_117252


namespace symmetric_points_origin_l117_117836

theorem symmetric_points_origin (a b : ℝ)
  (h1 : (-2 : ℝ) = -a)
  (h2 : (b : ℝ) = -3) : a - b = 5 :=
by
  sorry

end symmetric_points_origin_l117_117836


namespace evaluate_expression_x_eq_3_l117_117517

theorem evaluate_expression_x_eq_3 : (3^5 - 5 * 3 + 7 * 3^3) = 417 := by
  sorry

end evaluate_expression_x_eq_3_l117_117517


namespace prove_a5_l117_117990

-- Definition of the conditions
def expansion (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) :=
  (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x)^2 + a_3 * (1 + x)^3 + a_4 * (1 + x)^4 + 
               a_5 * (1 + x)^5 + a_6 * (1 + x)^6 + a_7 * (1 + x)^7 + a_8 * (1 + x)^8

-- Given condition
axiom condition (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : ∀ x : ℤ, expansion x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8

-- The target problem: proving a_5 = -448
theorem prove_a5 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : a_5 = -448 :=
by
  sorry

end prove_a5_l117_117990


namespace area_correct_l117_117889

open BigOperators

def Rectangle (PQ RS : ℕ) := PQ * RS

def PointOnSegment (a b : ℕ) (ratio : ℚ) : ℚ :=
ratio * (b - a)

def area_of_PTUS : ℚ :=
Rectangle 10 6 - (0.5 * 6 * (10 / 3) + 0.5 * 10 * 6)

theorem area_correct :
  area_of_PTUS = 20 := by
  sorry

end area_correct_l117_117889


namespace max_groups_l117_117618

theorem max_groups (cards : ℕ) (sum_group : ℕ) (c5 c2 c1 : ℕ) (cond1 : cards = 600) (cond2 : c5 = 200)
  (cond3 : c2 = 200) (cond4 : c1 = 200) (cond5 : sum_group = 9) :
  ∃ max_g : ℕ, max_g = 100 :=
by
  sorry

end max_groups_l117_117618


namespace socks_impossible_l117_117756

theorem socks_impossible (n m : ℕ) (h : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end socks_impossible_l117_117756


namespace sin_beta_equals_sqrt3_div_2_l117_117669

noncomputable def angles_acute (α β : ℝ) : Prop :=
0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2

theorem sin_beta_equals_sqrt3_div_2 
  (α β : ℝ) 
  (h_acute: angles_acute α β) 
  (h_sin_alpha: Real.sin α = (4/7) * Real.sqrt 3) 
  (h_cos_alpha_plus_beta: Real.cos (α + β) = -(11/14)) 
  : Real.sin β = (Real.sqrt 3) / 2 :=
sorry

end sin_beta_equals_sqrt3_div_2_l117_117669


namespace sin_minus_cos_l117_117370

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l117_117370


namespace sin_minus_cos_l117_117372

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l117_117372


namespace father_gave_8_candies_to_Billy_l117_117061

theorem father_gave_8_candies_to_Billy (candies_Billy : ℕ) (candies_Caleb : ℕ) (candies_Andy : ℕ) (candies_father : ℕ) 
  (candies_given_to_Caleb : ℕ) (candies_more_than_Caleb : ℕ) (candies_given_by_father_total : ℕ) :
  (candies_given_to_Caleb = 11) →
  (candies_Caleb = 11) →
  (candies_Andy = 9) →
  (candies_father = 36) →
  (candies_Andy = candies_Caleb + 4) →
  (candies_given_by_father_total = candies_given_to_Caleb + (candies_Andy - 9)) →
  (candies_father - candies_given_by_father_total = 8) →
  candies_Billy = 8 := 
by
  intros
  sorry

end father_gave_8_candies_to_Billy_l117_117061


namespace correct_option_is_B_l117_117164

noncomputable def correct_calculation (x : ℝ) : Prop :=
  (x ≠ 1) → (x ≠ 0) → (x ≠ -1) → (-2 / (2 * x - 2) = 1 / (1 - x))

theorem correct_option_is_B (x : ℝ) : correct_calculation x := by
  intros hx1 hx2 hx3
  sorry

end correct_option_is_B_l117_117164


namespace no_possible_blue_socks_l117_117749

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l117_117749


namespace jeff_boxes_filled_l117_117710

def donuts_each_day : ℕ := 10
def days : ℕ := 12
def jeff_eats_per_day : ℕ := 1
def chris_eats : ℕ := 8
def donuts_per_box : ℕ := 10

theorem jeff_boxes_filled : 
  (donuts_each_day * days - jeff_eats_per_day * days - chris_eats) / donuts_per_box = 10 :=
by
  sorry

end jeff_boxes_filled_l117_117710


namespace pass_rate_correct_l117_117181

variable {a b : ℝ}

-- Assumptions: defect rates are between 0 and 1
axiom h_a : 0 ≤ a ∧ a ≤ 1
axiom h_b : 0 ≤ b ∧ b ≤ 1

-- Definition: Pass rate is 1 minus the defect rate
def pass_rate (a b : ℝ) : ℝ := (1 - a) * (1 - b)

-- Theorem: Proving the pass rate is (1 - a) * (1 - b)
theorem pass_rate_correct : pass_rate a b = (1 - a) * (1 - b) := 
by
  sorry

end pass_rate_correct_l117_117181


namespace problem1_problem2_l117_117192

variable (x : ℝ)

-- Statement for the first problem
theorem problem1 : (-1 + 3 * x) * (-3 * x - 1) = 1 - 9 * x^2 := 
by
  sorry

-- Statement for the second problem
theorem problem2 : (x + 1)^2 - (1 - 3 * x) * (1 + 3 * x) = 10 * x^2 + 2 * x := 
by
  sorry

end problem1_problem2_l117_117192


namespace Jennifer_has_24_dollars_left_l117_117578

def remaining_money (initial amount: ℕ) (spent_sandwich spent_museum_ticket spent_book: ℕ) : ℕ :=
  initial - (spent_sandwich + spent_museum_ticket + spent_book)

theorem Jennifer_has_24_dollars_left :
  remaining_money 180 (1/5*180) (1/6*180) (1/2*180) = 24 :=
by
  sorry

end Jennifer_has_24_dollars_left_l117_117578


namespace alfonzo_visit_l117_117644

-- Define the number of princes (palaces) as n
variable (n : ℕ)

-- Define the type of connections (either a "Ruelle" or a "Canal")
inductive Transport
| Ruelle
| Canal

-- Define the connection between any two palaces
noncomputable def connection (i j : ℕ) : Transport := sorry

-- The theorem states that Prince Alfonzo can visit all his friends using only one type of transportation
theorem alfonzo_visit (h : ∀ i j, i ≠ j → ∃ t : Transport, ∀ k, k ≠ i → connection i k = t) :
  ∃ t : Transport, ∀ i j, i ≠ j → connection i j = t :=
sorry

end alfonzo_visit_l117_117644


namespace value_of_C_is_2_l117_117272

def isDivisibleBy3 (n : ℕ) : Prop := n % 3 = 0
def isDivisibleBy7 (n : ℕ) : Prop := n % 7 = 0

def sumOfDigitsFirstNumber (A B : ℕ) : ℕ := 6 + 5 + A + 3 + 1 + B + 4
def sumOfDigitsSecondNumber (A B C : ℕ) : ℕ := 4 + 1 + 7 + A + B + 5 + C

theorem value_of_C_is_2 (A B : ℕ) (hDiv3First : isDivisibleBy3 (sumOfDigitsFirstNumber A B))
  (hDiv7First : isDivisibleBy7 (sumOfDigitsFirstNumber A B))
  (hDiv3Second : isDivisibleBy3 (sumOfDigitsSecondNumber A B 2))
  (hDiv7Second : isDivisibleBy7 (sumOfDigitsSecondNumber A B 2)) : 
  (∃ (C : ℕ), C = 2) :=
sorry

end value_of_C_is_2_l117_117272


namespace sum_absolute_values_of_first_ten_terms_l117_117645

noncomputable def S (n : ℕ) : ℤ := n^2 - 4 * n + 2

noncomputable def a (n : ℕ) : ℤ := S n - S (n - 1)

noncomputable def absolute_sum_10 : ℤ :=
  |a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10|

theorem sum_absolute_values_of_first_ten_terms : absolute_sum_10 = 68 := by
  sorry

end sum_absolute_values_of_first_ten_terms_l117_117645


namespace cube_root_simplification_l117_117298

theorem cube_root_simplification :
  let a := 1
  let b := 30
  a + b = 31 := by
  sorry

end cube_root_simplification_l117_117298


namespace number_of_ordered_triples_l117_117073

theorem number_of_ordered_triples (x y z : ℝ) (hx : x + y = 3) (hy : xy - z^2 = 4)
  (hnn : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) : 
  ∃! (x y z : ℝ), (x + y = 3) ∧ (xy - z^2 = 4) ∧ (0 ≤ x) ∧ (0 ≤ y) ∧ (0 ≤ z) :=
sorry

end number_of_ordered_triples_l117_117073


namespace min_width_for_fence_area_least_200_l117_117128

theorem min_width_for_fence_area_least_200 (w : ℝ) (h : w * (w + 20) ≥ 200) : w ≥ 10 :=
sorry

end min_width_for_fence_area_least_200_l117_117128


namespace students_not_taking_music_nor_art_l117_117175

theorem students_not_taking_music_nor_art (total_students music_students art_students both_students neither_students : ℕ) 
  (h_total : total_students = 500) 
  (h_music : music_students = 50) 
  (h_art : art_students = 20) 
  (h_both : both_students = 10) 
  (h_neither : neither_students = total_students - (music_students + art_students - both_students)) : 
  neither_students = 440 :=
by
  sorry

end students_not_taking_music_nor_art_l117_117175


namespace tail_to_body_ratio_l117_117708

variables (B : ℝ) (tail : ℝ := 9) (total_length : ℝ := 30)
variables (head_ratio : ℝ := 1/6)

-- Condition: The overall length is 30 inches
def overall_length_eq : Prop := B + B * head_ratio + tail = total_length

-- Theorem: Ratio of tail length to body length is 1:2
theorem tail_to_body_ratio (h : overall_length_eq B) : tail / B = 1 / 2 :=
sorry

end tail_to_body_ratio_l117_117708


namespace tennis_balls_ordered_l117_117635

def original_white_balls : ℕ := sorry
def original_yellow_balls_with_error : ℕ := sorry

theorem tennis_balls_ordered 
  (W Y : ℕ)
  (h1 : W = Y)
  (h2 : Y + 70 = original_yellow_balls_with_error)
  (h3 : W = 8 / 13 * (Y + 70)):
  W + Y = 224 := sorry

end tennis_balls_ordered_l117_117635


namespace rate_of_mixed_oil_per_litre_l117_117851

theorem rate_of_mixed_oil_per_litre :
  let oil1_litres := 10
  let oil1_rate := 55
  let oil2_litres := 5
  let oil2_rate := 66
  let total_cost := oil1_litres * oil1_rate + oil2_litres * oil2_rate
  let total_volume := oil1_litres + oil2_litres
  let rate_per_litre := total_cost / total_volume
  rate_per_litre = 58.67 :=
by
  sorry

end rate_of_mixed_oil_per_litre_l117_117851


namespace remainder_of_number_divisor_l117_117047

-- Define the interesting number and the divisor
def number := 2519
def divisor := 9
def expected_remainder := 8

-- State the theorem to prove the remainder condition
theorem remainder_of_number_divisor :
  number % divisor = expected_remainder := by
  sorry

end remainder_of_number_divisor_l117_117047


namespace kid_ticket_price_l117_117913

theorem kid_ticket_price (adult_price kid_tickets tickets total_profit : ℕ) 
  (h_adult_price : adult_price = 6) 
  (h_kid_tickets : kid_tickets = 75) 
  (h_tickets : tickets = 175) 
  (h_total_profit : total_profit = 750) : 
  (total_profit - (tickets - kid_tickets) * adult_price) / kid_tickets = 2 :=
by
  sorry

end kid_ticket_price_l117_117913


namespace cos_4pi_over_3_l117_117784

theorem cos_4pi_over_3 : Real.cos (4 * Real.pi / 3) = -1 / 2 :=
by 
  sorry

end cos_4pi_over_3_l117_117784


namespace find_YJ_l117_117239

structure Triangle :=
  (XY XZ YZ : ℝ)
  (XY_pos : XY > 0)
  (XZ_pos : XZ > 0)
  (YZ_pos : YZ > 0)

noncomputable def incenter_length (T : Triangle) : ℝ := 
  let XY := T.XY
  let XZ := T.XZ
  let YZ := T.YZ
  -- calculation using the provided constraints goes here
  3 * Real.sqrt 13 -- this should be computed based on the constraints, but is directly given as the answer

theorem find_YJ
  (T : Triangle)
  (XY_eq : T.XY = 17)
  (XZ_eq : T.XZ = 19)
  (YZ_eq : T.YZ = 20) :
  incenter_length T = 3 * Real.sqrt 13 :=
by 
  sorry

end find_YJ_l117_117239


namespace total_pounds_of_peppers_l117_117683

-- Definitions based on the conditions
def greenPeppers : ℝ := 0.3333333333333333
def redPeppers : ℝ := 0.3333333333333333

-- Goal statement expressing the problem
theorem total_pounds_of_peppers :
  greenPeppers + redPeppers = 0.6666666666666666 := 
by
  sorry

end total_pounds_of_peppers_l117_117683


namespace prism_faces_eq_nine_l117_117795

-- Define the condition: a prism with 21 edges
def prism_edges (n : ℕ) := n = 21

-- Define the number of sides on each polygonal base
def num_sides (L : ℕ) := 3 * L = 21

-- Define the total number of faces
def total_faces (F : ℕ) (L : ℕ) := F = L + 2

-- The theorem we want to prove
theorem prism_faces_eq_nine (n L F : ℕ) 
  (h1 : prism_edges n)
  (h2 : num_sides L)
  (h3 : total_faces F L) :
  F = 9 := 
sorry

end prism_faces_eq_nine_l117_117795


namespace min_focal_length_hyperbola_l117_117348

theorem min_focal_length_hyperbola (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b - c = 2) : 
  2*c ≥ 4 + 4 * Real.sqrt 2 := 
sorry

end min_focal_length_hyperbola_l117_117348


namespace necessary_but_not_sufficient_l117_117419

-- Define the propositions P and Q
def P (a b : ℝ) : Prop := a^2 + b^2 > 2 * a * b
def Q (a b : ℝ) : Prop := abs (a + b) < abs a + abs b

-- Define the conditions for P and Q
def condition_for_P (a b : ℝ) : Prop := a ≠ b
def condition_for_Q (a b : ℝ) : Prop := a * b < 0

-- Define the statement
theorem necessary_but_not_sufficient (a b : ℝ) :
  (P a b → Q a b) ∧ ¬ (Q a b → P a b) :=
by
  sorry

end necessary_but_not_sufficient_l117_117419


namespace sin_minus_cos_l117_117386

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l117_117386


namespace find_hours_l117_117150

theorem find_hours (x : ℕ) (h : (14 + 10 + 13 + 9 + 12 + 11 + x) / 7 = 12) : x = 15 :=
by
  -- The proof is omitted
  sorry

end find_hours_l117_117150


namespace smallest_number_l117_117055

theorem smallest_number : Min 2 (-2.5) 0 (-3) = -3 := 
sorry

end smallest_number_l117_117055


namespace goods_train_speed_l117_117793

-- Define the given constants
def train_length : ℕ := 370 -- in meters
def platform_length : ℕ := 150 -- in meters
def crossing_time : ℕ := 26 -- in seconds
def conversion_factor : ℕ := 36 / 10 -- conversion from m/s to km/hr

-- Define the total distance covered
def total_distance : ℕ := train_length + platform_length -- in meters

-- Define the speed of the train in m/s
def speed_m_per_s : ℕ := total_distance / crossing_time

-- Define the speed of the train in km/hr
def speed_km_per_hr : ℕ := speed_m_per_s * conversion_factor

-- The proof problem statement
theorem goods_train_speed : speed_km_per_hr = 72 := 
by 
  -- Placeholder for the proof
  sorry

end goods_train_speed_l117_117793


namespace line_parallel_unique_a_l117_117418

theorem line_parallel_unique_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2*y + a + 3 = 0 → x + (a + 1)*y + 4 = 0) → a = -2 :=
  by
  sorry

end line_parallel_unique_a_l117_117418


namespace cornbread_pieces_count_l117_117871

-- Define the dimensions of the pan and the pieces of cornbread
def pan_length := 24
def pan_width := 20
def piece_length := 3
def piece_width := 2
def margin := 1

-- Define the effective width after considering the margin
def effective_width := pan_width - margin

-- Prove the number of pieces of cornbread is 72
theorem cornbread_pieces_count :
  (pan_length / piece_length) * (effective_width / piece_width) = 72 :=
by
  sorry

end cornbread_pieces_count_l117_117871


namespace intersection_A_B_l117_117085

def setA : Set ℝ := {x : ℝ | x > -1}
def setB : Set ℝ := {x : ℝ | x < 3}
def setIntersection : Set ℝ := {x : ℝ | x > -1 ∧ x < 3}

theorem intersection_A_B :
  setA ∩ setB = setIntersection :=
by sorry

end intersection_A_B_l117_117085


namespace patio_length_l117_117817

def patio (width length : ℝ) := length = 4 * width ∧ 2 * (width + length) = 100

theorem patio_length (width length : ℝ) (h : patio width length) : length = 40 :=
by
  cases h with len_eq_perim_eq
  sorry

end patio_length_l117_117817


namespace ashok_average_marks_l117_117188

variable (avg_5_subjects : ℕ) (marks_6th_subject : ℕ)
def total_marks_5_subjects := avg_5_subjects * 5
def total_marks_6_subjects := total_marks_5_subjects avg_5_subjects + marks_6th_subject
def avg_6_subjects := total_marks_6_subjects avg_5_subjects marks_6th_subject / 6

theorem ashok_average_marks (h1 : avg_5_subjects = 74) (h2 : marks_6th_subject = 50) : avg_6_subjects avg_5_subjects marks_6th_subject = 70 := by
  sorry

end ashok_average_marks_l117_117188


namespace radio_price_position_l117_117318

def price_positions (n : ℕ) (total_items : ℕ) (rank_lowest : ℕ) : Prop :=
  rank_lowest = total_items - n + 1

theorem radio_price_position :
  ∀ (n total_items rank_lowest : ℕ),
    total_items = 34 →
    rank_lowest = 21 →
    price_positions n total_items rank_lowest →
    n = 14 :=
by
  intros n total_items rank_lowest h_total h_rank h_pos
  rw [h_total, h_rank] at h_pos
  sorry

end radio_price_position_l117_117318


namespace marty_combinations_l117_117123

theorem marty_combinations : 
  ∃ n : ℕ, n = 5 * 4 ∧ n = 20 :=
by
  sorry

end marty_combinations_l117_117123


namespace probability_of_color_change_l117_117500

def traffic_light_cycle := 90
def green_duration := 45
def yellow_duration := 5
def red_duration := 40
def green_to_yellow := green_duration
def yellow_to_red := green_duration + yellow_duration
def red_to_green := traffic_light_cycle
def observation_interval := 4
def valid_intervals := [green_to_yellow - observation_interval + 1, green_to_yellow, 
                        yellow_to_red - observation_interval + 1, yellow_to_red, 
                        red_to_green - observation_interval + 1, red_to_green]
def total_valid_intervals := valid_intervals.length * observation_interval

theorem probability_of_color_change : 
  (total_valid_intervals : ℚ) / traffic_light_cycle = 2 / 15 := 
by
  sorry

end probability_of_color_change_l117_117500


namespace months_to_survive_l117_117428

theorem months_to_survive (P_survive : ℝ) (initial_population : ℕ) (expected_survivors : ℝ) (n : ℕ)
  (h1 : P_survive = 5 / 6)
  (h2 : initial_population = 200)
  (h3 : expected_survivors = 115.74)
  (h4 : initial_population * (P_survive ^ n) = expected_survivors) :
  n = 3 :=
sorry

end months_to_survive_l117_117428


namespace simplify_fraction_l117_117622

variable (c : ℝ)

theorem simplify_fraction :
  (6 + 2 * c) / 7 + 3 = (27 + 2 * c) / 7 := 
by 
  sorry

end simplify_fraction_l117_117622


namespace trader_sold_pens_l117_117636

theorem trader_sold_pens (C : ℝ) (N : ℕ) (hC : C > 0) (h_gain : N * (2 / 5) = 40) : N = 100 :=
by
  sorry

end trader_sold_pens_l117_117636


namespace sin_eq_cos_example_l117_117072

theorem sin_eq_cos_example 
  (n : ℤ) (h_range : -180 ≤ n ∧ n ≤ 180)
  (h_eq : Real.sin (n * Real.pi / 180) = Real.cos (682 * Real.pi / 180)) :
  n = 128 :=
sorry

end sin_eq_cos_example_l117_117072


namespace invertible_my_matrix_l117_117822

def my_matrix : Matrix (Fin 2) (Fin 2) ℚ := ![![4, 5], ![-2, 9]]

noncomputable def inverse_of_my_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  Matrix.det my_matrix • Matrix.adjugate my_matrix

theorem invertible_my_matrix :
  inverse_of_my_matrix = (1 / 46 : ℚ) • ![![9, -5], ![2, 4]] :=
by
  sorry

end invertible_my_matrix_l117_117822


namespace correct_calculation_l117_117621

noncomputable def option_A : Prop := (Real.sqrt 3 + Real.sqrt 2) ≠ Real.sqrt 5
noncomputable def option_B : Prop := (Real.sqrt 3 * Real.sqrt 5) = Real.sqrt 15 ∧ Real.sqrt 15 ≠ 15
noncomputable def option_C : Prop := Real.sqrt (32 / 8) = 2 ∧ (Real.sqrt (32 / 8) ≠ -2)
noncomputable def option_D : Prop := (2 * Real.sqrt 3) - Real.sqrt 3 = Real.sqrt 3

theorem correct_calculation : option_D :=
by
  sorry

end correct_calculation_l117_117621


namespace quadratic_value_range_l117_117607

theorem quadratic_value_range (y : ℝ) (h : y^3 - 6 * y^2 + 11 * y - 6 < 0) : 
  1 ≤ y^2 - 4 * y + 5 ∧ y^2 - 4 * y + 5 ≤ 2 := 
sorry

end quadratic_value_range_l117_117607


namespace online_game_months_l117_117929

theorem online_game_months (m : ℕ) (initial_cost monthly_cost total_cost : ℕ) 
  (h1 : initial_cost = 5) (h2 : monthly_cost = 8) (h3 : total_cost = 21) 
  (h_equation : initial_cost + monthly_cost * m = total_cost) : m = 2 :=
by {
  -- Placeholder for the proof, as we don't need to include it
  sorry
}

end online_game_months_l117_117929


namespace work_completed_by_a_l117_117098

theorem work_completed_by_a (a b : ℕ) (work_in_30_days : a + b = 4 * 30) (a_eq_3b : a = 3 * b) : (120 / a) = 40 :=
by
  -- Given a + b = 120 and a = 3 * b, prove that 120 / a = 40
  sorry

end work_completed_by_a_l117_117098


namespace root_in_interval_l117_117582

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem root_in_interval : 
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  intros h1 h2
  sorry

end root_in_interval_l117_117582


namespace tree_height_at_2_years_l117_117507

theorem tree_height_at_2_years (h₅ : ℕ) (h_four : ℕ) (h_three : ℕ) (h_two : ℕ) (h₅_value : h₅ = 243)
  (h_four_value : h_four = h₅ / 3) (h_three_value : h_three = h_four / 3) (h_two_value : h_two = h_three / 3) :
  h_two = 9 := by
  sorry

end tree_height_at_2_years_l117_117507


namespace find_a_squared_l117_117049

-- Defining the conditions for the problem
structure RectangleConditions :=
  (a : ℝ) 
  (side_length : ℝ := 36)
  (hinges_vertex : Bool := true)
  (hinges_midpoint : Bool := true)
  (pressed_distance : ℝ := 24)
  (hexagon_area_equiv : Bool := true)

-- Stating the theorem
theorem find_a_squared (cond : RectangleConditions) (ha : 36 * cond.a = 
  (24 * cond.a) + 2 * 15 * Real.sqrt (cond.a^2 - 36)) : 
  cond.a^2 = 720 :=
sorry

end find_a_squared_l117_117049


namespace perfect_square_trinomial_m_l117_117671

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ (a b : ℝ), (a ≠ 0) ∧ (x^2 - m * x + 25 = (a * x + b)^2)) → (m = 10 ∨ m = -10) :=
by
  -- Using the assumption that there exist constants a and b such that the trinomial is a perfect square
  intro h,
  obtain ⟨a, b, _, h_eq⟩ := h,
  -- Expanding the perfect square and comparing coefficients
  -- will yield the conclusion m = ±10
  sorry

end perfect_square_trinomial_m_l117_117671


namespace matrix_cubic_l117_117953

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![2, -1],
  ![1, 1]
]

theorem matrix_cubic :
  matrix_a ^ 3 = ![
    ![3, -6],
    ![6, -3]
  ] := by
  sorry

end matrix_cubic_l117_117953


namespace final_sale_price_l117_117938

def initial_price : ℝ := 450
def first_discount : ℝ := 0.25
def second_discount : ℝ := 0.10
def third_discount : ℝ := 0.05

def price_after_first_discount (initial : ℝ) (discount : ℝ) : ℝ :=
  initial * (1 - discount)
  
def price_after_second_discount (price_first : ℝ) (discount : ℝ) : ℝ :=
  price_first * (1 - discount)
  
def price_after_third_discount (price_second : ℝ) (discount : ℝ) : ℝ :=
  price_second * (1 - discount)

theorem final_sale_price :
  price_after_third_discount
    (price_after_second_discount
      (price_after_first_discount initial_price first_discount)
      second_discount)
    third_discount = 288.5625 := 
sorry

end final_sale_price_l117_117938


namespace coordinates_in_second_quadrant_l117_117435

section 
variable (x y : ℝ)
variable (hx : x = -7)
variable (hy : y = 4)
variable (quadrant : x < 0 ∧ y > 0)
variable (distance_x : |y| = 4)
variable (distance_y : |x| = 7)

theorem coordinates_in_second_quadrant :
  (x, y) = (-7, 4) := by
  sorry
end

end coordinates_in_second_quadrant_l117_117435


namespace number_of_crystals_in_container_l117_117450

-- Define the dimensions of the energy crystal
def length_crystal := 30
def width_crystal := 25
def height_crystal := 5

-- Define the dimensions of the cubic container
def side_container := 27

-- Volume of the cubic container
def volume_container := side_container ^ 3

-- Volume of the energy crystal
def volume_crystal := length_crystal * width_crystal * height_crystal

-- Proof statement
theorem number_of_crystals_in_container :
  volume_container / volume_crystal ≥ 5 :=
sorry

end number_of_crystals_in_container_l117_117450


namespace mirella_orange_books_read_l117_117815

-- Definitions based on the conditions in a)
def purpleBookPages : ℕ := 230
def orangeBookPages : ℕ := 510
def purpleBooksRead : ℕ := 5
def extraOrangePages : ℕ := 890

-- The total number of purple pages read
def purplePagesRead := purpleBooksRead * purpleBookPages

-- The number of orange books read
def orangeBooksRead (O : ℕ) := O * orangeBookPages

-- Statement to be proved
theorem mirella_orange_books_read (O : ℕ) :
  orangeBooksRead O = purplePagesRead + extraOrangePages → O = 4 :=
by
  sorry

end mirella_orange_books_read_l117_117815


namespace profit_amount_calc_l117_117790

-- Define the conditions as hypotheses
variables (SP : ℝ) (profit_percent : ℝ) (cost_price profit_amount : ℝ)

-- Given conditions
axiom selling_price : SP = 900
axiom profit_percentage : profit_percent = 50
axiom profit_formula : profit_amount = 0.5 * cost_price
axiom selling_price_formula : SP = cost_price + profit_amount

-- The theorem to be proven
theorem profit_amount_calc : profit_amount = 300 :=
by
  sorry

end profit_amount_calc_l117_117790


namespace symmetric_points_origin_l117_117837

theorem symmetric_points_origin (a b : ℝ)
  (h1 : (-2 : ℝ) = -a)
  (h2 : (b : ℝ) = -3) : a - b = 5 :=
by
  sorry

end symmetric_points_origin_l117_117837


namespace legos_in_box_at_end_l117_117576

def initial_legos : ℕ := 500
def legos_used : ℕ := initial_legos / 2
def missing_legos : ℕ := 5
def remaining_legos := legos_used - missing_legos

theorem legos_in_box_at_end : remaining_legos = 245 := 
by
  sorry

end legos_in_box_at_end_l117_117576


namespace youngest_child_age_l117_117697

theorem youngest_child_age (x : ℕ) (h1 : Prime x)
  (h2 : Prime (x + 2))
  (h3 : Prime (x + 6))
  (h4 : Prime (x + 8))
  (h5 : Prime (x + 12))
  (h6 : Prime (x + 14)) :
  x = 5 := 
sorry

end youngest_child_age_l117_117697


namespace area_of_midpoint_quadrilateral_l117_117725

theorem area_of_midpoint_quadrilateral (length width : ℝ) (h_length : length = 15) (h_width : width = 8) :
  let A := (0, width / 2)
  let B := (length / 2, 0)
  let C := (length, width / 2)
  let D := (length / 2, width)
  let mid_quad_area := (length / 2) * (width / 2)
  mid_quad_area = 30 :=
by
  simp [h_length, h_width]
  sorry

end area_of_midpoint_quadrilateral_l117_117725


namespace fraction_sum_in_simplest_form_l117_117907

theorem fraction_sum_in_simplest_form :
  ∃ a b : ℕ, a + b = 11407 ∧ 0.425875 = a / (b : ℝ) ∧ Nat.gcd a b = 1 :=
by
  sorry

end fraction_sum_in_simplest_form_l117_117907


namespace perp_a_beta_l117_117084

noncomputable def line : Type := sorry
noncomputable def plane : Type := sorry
noncomputable def Incident (l : line) (p : plane) : Prop := sorry
noncomputable def Perpendicular (l1 l2 : line) : Prop := sorry
noncomputable def Parallel (l1 l2 : line) : Prop := sorry

variables {α β : plane} {a AB : line}

-- Conditions extracted from the problem
axiom condition1 : Perpendicular α β
axiom condition2 : Incident AB β ∧ Incident AB α
axiom condition3 : Parallel a α
axiom condition4 : Perpendicular a AB

-- The statement that needs to be proved
theorem perp_a_beta : Perpendicular a β :=
  sorry

end perp_a_beta_l117_117084


namespace probability_of_shaded_triangle_l117_117863

def total_triangles : ℕ := 9
def shaded_triangles : ℕ := 3

theorem probability_of_shaded_triangle :
  total_triangles > 5 →
  (shaded_triangles : ℚ) / total_triangles = 1 / 3 :=
by
  intros h
  -- proof here
  sorry

end probability_of_shaded_triangle_l117_117863


namespace optionA_not_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_right_triangle_l117_117054
-- Import necessary libraries

-- Define each of the conditions as Lean definitions
def OptionA (a b c : ℝ) : Prop := a = 1.5 ∧ b = 2 ∧ c = 3
def OptionB (a b c : ℝ) : Prop := a = 7 ∧ b = 24 ∧ c = 25
def OptionC (a b c : ℝ) : Prop := ∃ k : ℕ, a = (3 : ℝ)*k ∧ b = (4 : ℝ)*k ∧ c = (5 : ℝ)*k
def OptionD (a b c : ℝ) : Prop := a = 9 ∧ b = 12 ∧ c = 15

-- Define the Pythagorean theorem predicate
def Pythagorean (a b c : ℝ) : Prop := a^2 + b^2 = c^2

-- State the theorem to prove Option A cannot form a right triangle
theorem optionA_not_right_triangle : ¬ Pythagorean 1.5 2 3 := by sorry

-- State the remaining options can form a right triangle
theorem optionB_right_triangle : Pythagorean 7 24 25 := by sorry
theorem optionC_right_triangle (k : ℕ) : Pythagorean (3 * k) (4 * k) (5 * k) := by sorry
theorem optionD_right_triangle : Pythagorean 9 12 15 := by sorry

end optionA_not_right_triangle_optionB_right_triangle_optionC_right_triangle_optionD_right_triangle_l117_117054


namespace factorial_division_l117_117960
-- Definition of factorial
def fact : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * fact n

-- Problem statement
theorem factorial_division : fact 12 / fact 11 = 12 :=
by sorry

end factorial_division_l117_117960


namespace adam_deleted_items_l117_117637

theorem adam_deleted_items (initial_items deleted_items remaining_items : ℕ)
  (h1 : initial_items = 100) (h2 : remaining_items = 20) 
  (h3 : remaining_items = initial_items - deleted_items) : 
  deleted_items = 80 :=
by
  sorry

end adam_deleted_items_l117_117637


namespace scientific_notation_of_0_0000021_l117_117639

theorem scientific_notation_of_0_0000021 :
  (0.0000021 : ℝ) = 2.1 * 10^(-6) :=
sorry

end scientific_notation_of_0_0000021_l117_117639


namespace find_k_l117_117827

noncomputable def series_sum (k : ℝ) : ℝ :=
  3 + ∑' (n : ℕ), (3 + (n + 1) * k) / 4^(n + 1)

theorem find_k : ∃ k : ℝ, series_sum k = 8 ∧ k = 9 :=
by
  use 9
  have h : series_sum 9 = 8 := sorry
  exact ⟨h, rfl⟩

end find_k_l117_117827


namespace remainder_modulus_9_l117_117824

theorem remainder_modulus_9 : (9 * 7^18 + 2^18) % 9 = 1 := 
by sorry

end remainder_modulus_9_l117_117824


namespace integer_pairs_satisfying_equation_l117_117620

theorem integer_pairs_satisfying_equation :
  {p : ℤ × ℤ | (p.1)^3 + (p.2)^3 - 3*(p.1)^2 + 6*(p.2)^2 + 3*(p.1) + 12*(p.2) + 6 = 0}
  = {(1, -1), (2, -2)} := 
sorry

end integer_pairs_satisfying_equation_l117_117620


namespace no_half_probability_socks_l117_117762

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l117_117762


namespace slope_intercept_condition_l117_117903

theorem slope_intercept_condition (m b : ℚ) (h_m : m = 1/3) (h_b : b = -3/4) : -1 < m * b ∧ m * b < 0 := by
  sorry

end slope_intercept_condition_l117_117903


namespace jerry_total_bill_l117_117711

-- Definitions for the initial bill and late fees
def initial_bill : ℝ := 250
def first_fee_rate : ℝ := 0.02
def second_fee_rate : ℝ := 0.03

-- Function to calculate the total bill after applying the fees
def total_bill (init : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let first_total := init * (1 + rate1)
  first_total * (1 + rate2)

-- Theorem statement
theorem jerry_total_bill : total_bill initial_bill first_fee_rate second_fee_rate = 262.65 := by
  sorry

end jerry_total_bill_l117_117711


namespace monomial_2023_l117_117050

def monomial (n : ℕ) : ℤ × ℕ :=
  ((-1)^n * (n + 1), n)

theorem monomial_2023 :
  monomial 2023 = (-2024, 2023) :=
by
  sorry

end monomial_2023_l117_117050


namespace angle_bisector_relation_l117_117691

theorem angle_bisector_relation (a b : ℝ) (h : a = -b ∨ a = -b) : a = -b :=
sorry

end angle_bisector_relation_l117_117691


namespace sum_of_interior_angles_l117_117738

theorem sum_of_interior_angles (n : ℕ) (h1 : 180 * (n - 2) = 1800) (h2 : n = 12) : 
  180 * ((n + 4) - 2) = 2520 := 
by 
  { sorry }

end sum_of_interior_angles_l117_117738


namespace num_ordered_pairs_l117_117344

theorem num_ordered_pairs :
  ∃ n : ℕ, n = 49 ∧ ∀ (a b : ℕ), a + b = 50 → 0 < a ∧ 0 < b → (1 ≤ a ∧ a < 50) :=
by
  sorry

end num_ordered_pairs_l117_117344


namespace find_f3_l117_117343

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_f3 
  (hf : is_odd f) 
  (hg : is_even g) 
  (h : ∀ x, f x + g x = 1 / (x - 1)) : 
  f 3 = 3 / 8 :=
by 
  sorry

end find_f3_l117_117343


namespace determine_symmetry_l117_117544

def quadratic_function_is_symmetric_about_y (y : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x : ℝ, y (-x) = y x

theorem determine_symmetry (m : ℝ) :
  quadratic_function_is_symmetric_about_y (λ x, m * x^2 + (m - 2) * x + 2) m ↔ m = 2 :=
by
  sorry

end determine_symmetry_l117_117544


namespace product_of_irwins_baskets_l117_117869

theorem product_of_irwins_baskets 
  (baskets_scored : Nat)
  (point_value : Nat)
  (total_baskets : baskets_scored = 2)
  (value_per_basket : point_value = 11) : 
  point_value * baskets_scored = 22 := 
by 
  sorry

end product_of_irwins_baskets_l117_117869


namespace games_that_didnt_work_l117_117880

variable (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (good_games : ℕ)

theorem games_that_didnt_work
  (h₁ : games_from_friend = 2)
  (h₂ : games_from_garage_sale = 2)
  (h₃ : good_games = 2) :
  (games_from_friend + games_from_garage_sale - good_games) = 2 :=
by 
  sorry

end games_that_didnt_work_l117_117880


namespace find_y_l117_117850

theorem find_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1 / y) (h2 : y = 2 + 1 / x) :
  y = 1 + Real.sqrt 2 ∨ y = 1 - Real.sqrt 2 :=
by
  sorry

end find_y_l117_117850


namespace amount_paid_to_Y_l117_117305

-- Definition of the conditions.
def total_payment (X Y : ℕ) : Prop := X + Y = 330
def payment_relation (X Y : ℕ) : Prop := X = 12 * Y / 10

-- The theorem we want to prove.
theorem amount_paid_to_Y (X Y : ℕ) (h1 : total_payment X Y) (h2 : payment_relation X Y) : Y = 150 := 
by 
  sorry

end amount_paid_to_Y_l117_117305


namespace difference_of_roots_l117_117116

noncomputable def r_and_s (r s : ℝ) : Prop :=
(∃ (r s : ℝ), (r, s) ≠ (s, r) ∧ r > s ∧ (5 * r - 15) / (r ^ 2 + 3 * r - 18) = r + 3
  ∧ (5 * s - 15) / (s ^ 2 + 3 * s - 18) = s + 3)

theorem difference_of_roots (r s : ℝ) (h : r_and_s r s) : r - s = Real.sqrt 29 := by
  sorry

end difference_of_roots_l117_117116


namespace worker_late_by_10_minutes_l117_117288

def usual_time : ℕ := 40
def speed_ratio : ℚ := 4 / 5
def time_new := (usual_time : ℚ) * (5 / 4) -- This is the equation derived from solving

theorem worker_late_by_10_minutes : 
  ((time_new : ℚ) - usual_time) = 10 :=
by
  sorry -- proof is skipped

end worker_late_by_10_minutes_l117_117288


namespace problem_l117_117726

variable {R : Type} [Field R]

def f1 (a b c d : R) : R := a + b + c + d
def f2 (a b c d : R) : R := (1 / a) + (1 / b) + (1 / c) + (1 / d)
def f3 (a b c d : R) : R := (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) + (1 / (1 - d))

theorem problem (a b c d : R) (h1 : f1 a b c d = 2) (h2 : f2 a b c d = 2) : f3 a b c d = 2 :=
by sorry

end problem_l117_117726


namespace Joey_downhill_speed_l117_117436

theorem Joey_downhill_speed
  (Route_length : ℝ) (Time_uphill : ℝ) (Speed_uphill : ℝ) (Overall_average_speed : ℝ) (Extra_time_due_to_rain : ℝ) :
  Route_length = 5 →
  Time_uphill = 1.25 →
  Speed_uphill = 4 →
  Overall_average_speed = 6 →
  Extra_time_due_to_rain = 0.25 →
  ((2 * Route_length) / Overall_average_speed - Time_uphill - Extra_time_due_to_rain) * (Route_length / (2 * Route_length / Overall_average_speed - Time_uphill - Extra_time_due_to_rain)) = 30 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end Joey_downhill_speed_l117_117436


namespace find_n_from_remainders_l117_117484

theorem find_n_from_remainders (a n : ℕ) (h1 : a^2 % n = 8) (h2 : a^3 % n = 25) : n = 113 := 
by 
  -- proof needed here
  sorry

end find_n_from_remainders_l117_117484


namespace trig_identity_l117_117362

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l117_117362


namespace find_replaced_man_weight_l117_117233

variable (n : ℕ) (new_weight old_avg_weight : ℝ) (weight_inc : ℝ) (W : ℝ)

theorem find_replaced_man_weight 
  (h1 : n = 8) 
  (h2 : new_weight = 68) 
  (h3 : weight_inc = 1) 
  (h4 : 8 * (old_avg_weight + 1) = 8 * old_avg_weight + (new_weight - W)) 
  : W = 60 :=
by
  sorry

end find_replaced_man_weight_l117_117233


namespace vector_dot_product_l117_117845

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)

-- Prove that the scalar product a · (a - 2b) equals 2
theorem vector_dot_product :
  let u := a
  let v := b
  u • (u - (2 • v)) = 2 :=
by 
  -- Placeholder for the proof
  sorry

end vector_dot_product_l117_117845


namespace area_of_triangle_BXC_l117_117915

/-
  Given:
  - AB = 15 units
  - CD = 40 units
  - The area of trapezoid ABCD = 550 square units

  To prove:
  - The area of triangle BXC = 1200 / 11 square units
-/
theorem area_of_triangle_BXC 
  (AB CD : ℝ) 
  (hAB : AB = 15) 
  (hCD : CD = 40) 
  (area_ABCD : ℝ)
  (hArea_ABCD : area_ABCD = 550) 
  : ∃ (area_BXC : ℝ), area_BXC = 1200 / 11 :=
by
  sorry

end area_of_triangle_BXC_l117_117915


namespace sin_minus_cos_l117_117402

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end sin_minus_cos_l117_117402


namespace regression_decrease_by_three_l117_117666

-- Given a regression equation \hat y = 2 - 3 \hat x
def regression_equation (x : ℝ) : ℝ :=
  2 - 3 * x

-- Prove that when x increases by one unit, \hat y decreases by 3 units
theorem regression_decrease_by_three (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -3 :=
by
  -- proof
  sorry

end regression_decrease_by_three_l117_117666


namespace missing_digit_B_l117_117017

theorem missing_digit_B (B : ℕ) (h : 0 ≤ B ∧ B ≤ 9) (h_div : (100 + 10 * B + 3) % 13 = 0) : B = 4 := 
by
  sorry

end missing_digit_B_l117_117017


namespace sum_of_coefficients_l117_117276

theorem sum_of_coefficients :
  (∃ a b c d e : ℤ, 512 * x ^ 3 + 27 = a * x * (c * x ^ 2 + d * x + e) + b * (c * x ^ 2 + d * x + e)) →
  (a = 8 ∧ b = 3 ∧ c = 64 ∧ d = -24 ∧ e = 9) →
  a + b + c + d + e = 60 :=
by
  intro h1 h2
  sorry

end sum_of_coefficients_l117_117276


namespace Q_eq_G_l117_117350

def P := {y | ∃ x, y = x^2 + 1}
def Q := {y : ℝ | ∃ x, y = x^2 + 1}
def E := {x : ℝ | ∃ y, y = x^2 + 1}
def F := {(x, y) | y = x^2 + 1}
def G := {x : ℝ | x ≥ 1}

theorem Q_eq_G : Q = G := by
  sorry

end Q_eq_G_l117_117350


namespace natural_number_square_l117_117200

theorem natural_number_square (n : ℕ) : 
  (∃ x : ℕ, n^4 + 4 * n^3 + 5 * n^2 + 6 * n = x^2) ↔ n = 1 := 
by 
  sorry

end natural_number_square_l117_117200


namespace prove_incorrect_conclusion_l117_117006

-- Define the parabola as y = ax^2 + bx + c
def parabola_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the points
def point1 (a b c : ℝ) : Prop := parabola_eq a b c (-2) = 0
def point2 (a b c : ℝ) : Prop := parabola_eq a b c (-1) = 4
def point3 (a b c : ℝ) : Prop := parabola_eq a b c 0 = 6
def point4 (a b c : ℝ) : Prop := parabola_eq a b c 1 = 6

-- Define the conditions
def conditions (a b c : ℝ) : Prop :=
  point1 a b c ∧ point2 a b c ∧ point3 a b c ∧ point4 a b c

-- Define the incorrect conclusion
def incorrect_conclusion (a b c : ℝ) : Prop :=
  ¬ (parabola_eq a b c 2 = 0)

-- The statement to be proven
theorem prove_incorrect_conclusion (a b c : ℝ) (h : conditions a b c) : incorrect_conclusion a b c :=
sorry

end prove_incorrect_conclusion_l117_117006


namespace positive_integer_sum_representation_l117_117887

theorem positive_integer_sum_representation :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → ∃ (a : Fin 2004 → ℕ), 
    (∀ i j : Fin 2004, i < j → a i < a j) ∧ 
    (∀ i : Fin 2003, a i ∣ a (i + 1)) ∧
    (n = (Finset.univ.sum a)) := 
sorry

end positive_integer_sum_representation_l117_117887


namespace survey_method_correct_l117_117776

/-- Definitions to represent the options in the survey method problem. -/
inductive SurveyMethod
| A
| B
| C
| D

/-- The function to determine the correct survey method. -/
def appropriate_survey_method : SurveyMethod :=
  SurveyMethod.C

/-- The theorem stating that the appropriate survey method is indeed option C. -/
theorem survey_method_correct : appropriate_survey_method = SurveyMethod.C :=
by
  /- The actual proof is omitted as per instruction. -/
  sorry

end survey_method_correct_l117_117776


namespace problem_1_problem_2_problem_3_l117_117079

section Problem

-- Initial conditions
variable (a : ℕ → ℝ) (t m : ℝ)
def a_1 : ℝ := 3
def a_n (n : ℕ) (h : 2 ≤ n) : ℝ := 2 * a (n - 1) + (t + 1) * 2^n + 3 * m + t

-- Problem 1:
theorem problem_1 (h : t = 0) (h' : m = 0) :
  ∃ d, ∀ n, 2 ≤ n → (a n / 2^n) = (a (n - 1) / 2^(n-1)) + d := sorry

-- Problem 2:
theorem problem_2 (h : t = -1) (h' : m = 4/3) :
  ∃ r, ∀ n, 2 ≤ n → a n + 3 = r * (a (n - 1) + 3) := sorry

-- Problem 3:
theorem problem_3 (h : t = 0) (h' : m = 1) :
  (∀ n, 1 ≤ n → a n = (n + 2) * 2^n - 3) ∧
  (∃ S : ℕ → ℝ, ∀ n, S n = (n + 1) * 2^(n + 1) - 2 - 3 * n) := sorry

end Problem

end problem_1_problem_2_problem_3_l117_117079


namespace alex_buys_15_pounds_of_corn_l117_117646

theorem alex_buys_15_pounds_of_corn:
  ∃ (c b : ℝ), c + b = 30 ∧ 1.20 * c + 0.60 * b = 27.00 ∧ c = 15.0 :=
by
  sorry

end alex_buys_15_pounds_of_corn_l117_117646


namespace find_second_number_l117_117016

theorem find_second_number (a b c : ℚ) (h1 : a + b + c = 98) (h2 : a = (2 / 3) * b) (h3 : c = (8 / 5) * b) : b = 30 :=
by sorry

end find_second_number_l117_117016


namespace difference_of_squares_is_40_l117_117910

theorem difference_of_squares_is_40 {x y : ℕ} (h1 : x + y = 20) (h2 : x * y = 99) (hx : x > y) : x^2 - y^2 = 40 :=
sorry

end difference_of_squares_is_40_l117_117910


namespace count_valid_n_l117_117074

theorem count_valid_n : 
  ∃ (count : ℕ), count = 88 ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 2000 ∧ 
   (∃ (a b : ℤ), a + b = -2 ∧ a * b = -n) ↔ 
   ∃ m, 1 ≤ m ∧ m ≤ 2000 ∧ (∃ a, a * (a + 2) = m)) := 
sorry

end count_valid_n_l117_117074


namespace find_ratio_of_b1_b2_l117_117896

variable (a b k a1 a2 b1 b2 : ℝ)
variable (h1 : a1 ≠ 0) (h2 : a2 ≠ 0) (hb1 : b1 ≠ 0) (hb2 : b2 ≠ 0)

noncomputable def inversely_proportional_condition := a1 * b1 = a2 * b2
noncomputable def ratio_condition := a1 / a2 = 3 / 4
noncomputable def difference_condition := b1 - b2 = 5

theorem find_ratio_of_b1_b2 
  (h_inv : inversely_proportional_condition a1 a2 b1 b2)
  (h_rat : ratio_condition a1 a2)
  (h_diff : difference_condition b1 b2) :
  b1 / b2 = 4 / 3 :=
sorry

end find_ratio_of_b1_b2_l117_117896


namespace justin_run_time_l117_117110

theorem justin_run_time : 
  let flat_ground_rate := 2 / 2 -- Justin runs 2 blocks in 2 minutes on flat ground
  let uphill_rate := 2 / 3 -- Justin runs 2 blocks in 3 minutes uphill
  let total_blocks := 10 -- Justin is 10 blocks from home
  let uphill_blocks := 6 -- 6 of those blocks are uphill
  let flat_ground_blocks := total_blocks - uphill_blocks -- Remainder are flat ground
  let flat_ground_time := flat_ground_blocks * flat_ground_rate
  let uphill_time := uphill_blocks * uphill_rate
  let total_time := flat_ground_time + uphill_time
  total_time = 13 := 
by 
  sorry

end justin_run_time_l117_117110


namespace sin_minus_cos_l117_117369

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l117_117369


namespace factor_expression_l117_117518

theorem factor_expression (x : ℝ) :
  (16 * x ^ 7 + 36 * x ^ 4 - 9) - (4 * x ^ 7 - 6 * x ^ 4 - 9) = 6 * x ^ 4 * (2 * x ^ 3 + 7) :=
by
  sorry

end factor_expression_l117_117518


namespace length_of_lawn_l117_117813

-- Definitions based on conditions
def area_per_bag : ℝ := 250
def width : ℝ := 36
def num_bags : ℝ := 4
def extra_area : ℝ := 208

-- Statement to prove
theorem length_of_lawn :
  (num_bags * area_per_bag + extra_area) / width = 33.56 := by
  sorry

end length_of_lawn_l117_117813


namespace patio_length_l117_117816

theorem patio_length (w l : ℝ) (h1 : l = 4 * w) (h2 : 2 * l + 2 * w = 100) : l = 40 := 
by 
  sorry

end patio_length_l117_117816


namespace age_difference_is_51_l117_117124

def Milena_age : ℕ := 7
def Grandmother_age : ℕ := 9 * Milena_age
def Grandfather_age : ℕ := Grandmother_age + 2
def Cousin_age : ℕ := 2 * Milena_age
def Age_difference : ℕ := Grandfather_age - Cousin_age

theorem age_difference_is_51 : Age_difference = 51 := by
  sorry

end age_difference_is_51_l117_117124


namespace problem_l117_117533

theorem problem (a b : ℝ) (h : a > b) : a / 3 > b / 3 :=
sorry

end problem_l117_117533


namespace greenwood_school_l117_117942

theorem greenwood_school (f s : ℕ) (h : (3 / 4) * f = (1 / 3) * s) : s = 3 * f :=
by
  sorry

end greenwood_school_l117_117942


namespace num_koi_fish_after_3_weeks_l117_117264

noncomputable def total_days : ℕ := 3 * 7

noncomputable def koi_fish_added_per_day : ℕ := 2
noncomputable def goldfish_added_per_day : ℕ := 5

noncomputable def total_fish_initial : ℕ := 280
noncomputable def goldfish_final : ℕ := 200

noncomputable def koi_fish_total_after_3_weeks : ℕ :=
  let koi_fish_added := koi_fish_added_per_day * total_days
  let goldfish_added := goldfish_added_per_day * total_days
  let goldfish_initial := goldfish_final - goldfish_added
  let koi_fish_initial := total_fish_initial - goldfish_initial
  koi_fish_initial + koi_fish_added

theorem num_koi_fish_after_3_weeks :
  koi_fish_total_after_3_weeks = 227 := by
  sorry

end num_koi_fish_after_3_weeks_l117_117264


namespace PS_length_correct_l117_117916

noncomputable def length_PS_of_trapezoid (PQ RS QR : ℝ) (angle_QRP angle_PSR : ℝ) (ratio_RS_PQ : ℝ) : ℝ :=
  (8 / 3)

-- Conditions are instantiated as follows:
variables (P Q R S : Point)
variables (PQ RS QR : ℝ)
variables (angle_QRP angle_PSR : ℝ)
variables (ratio_RS_PQ : ℝ)

axiom PQ_parallel_RS : PQ ∥ RS
axiom length_QR : QR = 2
axiom angle_QRP_30 : angle_QRP = 30
axiom angle_PSR_60 : angle_PSR = 60
axiom ratio_RS_PQ_7_3 : ratio_RS_PQ = 7 / 3

theorem PS_length_correct : length_PS_of_trapezoid PQ RS QR angle_QRP angle_PSR ratio_RS_PQ = 8 / 3 := 
by
  sorry

end PS_length_correct_l117_117916


namespace complex_number_z_value_l117_117670

open Complex

theorem complex_number_z_value :
  ∀ (i z : ℂ), i^2 = -1 ∧ z * (1 + i) = 2 * i^2018 → z = -1 + i :=
by
  intros i z h
  have h1 : i^2 = -1 := h.1
  have h2 : z * (1 + i) = 2 * i^2018 := h.2
  sorry

end complex_number_z_value_l117_117670


namespace sin_minus_cos_l117_117373

noncomputable def theta_condition (θ : ℝ) : Prop := (0 < θ) ∧ (θ < π / 2) ∧ (Real.tan θ = 1 / 3)

theorem sin_minus_cos (θ : ℝ) (hθ : theta_condition θ) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry

end sin_minus_cos_l117_117373


namespace blue_paint_needed_l117_117590

/-- 
If the ratio of blue paint to green paint is \(4:1\), and Sarah wants to make 40 cans of the mixture,
prove that the number of cans of blue paint needed is 32.
-/
theorem blue_paint_needed (r: ℕ) (total_cans: ℕ) (h_ratio: r = 4) (h_total: total_cans = 40) : 
  ∃ b: ℕ, b = 4 / 5 * total_cans ∧ b = 32 :=
by
  sorry

end blue_paint_needed_l117_117590


namespace trig_identity_l117_117359

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l117_117359


namespace problem_solution_l117_117443

theorem problem_solution
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d :=
sorry

end problem_solution_l117_117443


namespace distinct_ordered_pairs_50_l117_117346

theorem distinct_ordered_pairs_50 (a b : ℕ) (h1 : a + b = 50) (h2 : 0 < a) (h3 : 0 < b) : 
    ({p : ℕ × ℕ | p.1 + p.2 = 50 ∧ 0 < p.1 ∧ 0 < p.2}.to_list.length = 49) :=
sorry

end distinct_ordered_pairs_50_l117_117346


namespace greater_combined_area_l117_117096

noncomputable def area_of_rectangle (length : ℝ) (width : ℝ) : ℝ :=
  length * width

noncomputable def combined_area (length : ℝ) (width : ℝ) : ℝ :=
  2 * (area_of_rectangle length width)

theorem greater_combined_area 
  (length1 width1 length2 width2 : ℝ)
  (h1 : length1 = 11) (h2 : width1 = 13)
  (h3 : length2 = 6.5) (h4 : width2 = 11) :
  combined_area length1 width1 - combined_area length2 width2 = 143 :=
by
  rw [h1, h2, h3, h4]
  sorry

end greater_combined_area_l117_117096


namespace problem_1_solution_set_problem_2_range_of_T_l117_117222

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 2|

theorem problem_1_solution_set :
  {x : ℝ | f x > 2} = {x | x < -5 ∨ 1 < x} :=
by 
  -- to be proven
  sorry

theorem problem_2_range_of_T (T : ℝ) :
  (∀ x : ℝ, f x ≥ -T^2 - 2.5 * T - 1) →
  (T ≤ -3 ∨ T ≥ 0.5) :=
by
  -- to be proven
  sorry

end problem_1_solution_set_problem_2_range_of_T_l117_117222


namespace jason_spent_at_music_store_l117_117870

theorem jason_spent_at_music_store 
  (cost_flute : ℝ) (cost_music_tool : ℝ) (cost_song_book : ℝ)
  (h1 : cost_flute = 142.46)
  (h2 : cost_music_tool = 8.89)
  (h3 : cost_song_book = 7) :
  cost_flute + cost_music_tool + cost_song_book = 158.35 :=
by
  -- assumption proof
  sorry

end jason_spent_at_music_store_l117_117870


namespace total_amount_740_l117_117857

theorem total_amount_740 (x y z : ℝ) (hz : z = 200) (hy : y = 1.20 * z) (hx : x = 1.25 * y) : 
  x + y + z = 740 := by
  sorry

end total_amount_740_l117_117857


namespace area_of_isosceles_triangle_l117_117273

theorem area_of_isosceles_triangle
  (h : ℝ)
  (s : ℝ)
  (b : ℝ)
  (altitude : h = 10)
  (perimeter : s + (s - 2) + 2 * b = 40)
  (pythagoras : b^2 + h^2 = s^2) :
  (b * h) = 81.2 :=
by
  sorry

end area_of_isosceles_triangle_l117_117273


namespace henry_socks_l117_117699

theorem henry_socks : 
  ∃ a b c : ℕ, 
    a + b + c = 15 ∧ 
    2 * a + 3 * b + 5 * c = 36 ∧ 
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ 
    a = 11 :=
by
  sorry

end henry_socks_l117_117699


namespace problem_l117_117413

def f (a : ℕ) : ℕ := a + 3
def F (a b : ℕ) : ℕ := b^2 + a

theorem problem : F 4 (f 5) = 68 := by sorry

end problem_l117_117413


namespace matrix_cubic_l117_117951

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![2, -1],
  ![1, 1]
]

theorem matrix_cubic :
  matrix_a ^ 3 = ![
    ![3, -6],
    ![6, -3]
  ] := by
  sorry

end matrix_cubic_l117_117951


namespace distance_between_parallel_lines_l117_117136

theorem distance_between_parallel_lines :
  ∀ {x y : ℝ}, 
  (3 * x - 4 * y + 1 = 0) → (3 * x - 4 * y + 7 = 0) → 
  ∃ d, d = (6 : ℝ) / 5 :=
by 
  sorry

end distance_between_parallel_lines_l117_117136


namespace no_possible_blue_socks_l117_117750

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l117_117750


namespace limit_of_power_seq_l117_117259

-- Define the problem and its conditions
theorem limit_of_power_seq (a : ℝ) (h : 0 < a ∨ 1 < a) :
  (0 < a ∧ a < 1 → ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, a^n < ε) ∧ 
  (1 < a → ∀ N > 0, ∃ n : ℕ, a^n > N) :=
by
  sorry

end limit_of_power_seq_l117_117259


namespace fraction_inequality_l117_117833

theorem fraction_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) :
  b / (a - c) < a / (b - d) :=
sorry

end fraction_inequality_l117_117833


namespace ellas_quadratic_equation_l117_117802

theorem ellas_quadratic_equation (d e : ℤ) :
  (∀ x : ℤ, |x - 8| = 3 → (x = 11 ∨ x = 5)) →
  (∀ x : ℤ, (x = 11 ∨ x = 5) → x^2 + d * x + e = 0) →
  (d, e) = (-16, 55) :=
by
  intro h1 h2
  sorry

end ellas_quadratic_equation_l117_117802


namespace homothety_image_collinear_O_G_H_concyclic_A_l117_117248

open EuclideanGeometry

namespace GeometryProof

section

variables {A B C : Point}
variables {O G H : Point}
variables {A' B' C' D E F : Point}

-- Definitions of key points
def circumcenter (A B C : Point) : Point := sorry  -- circumcenter of ΔABC
def centroid (A B C : Point) : Point := sorry  -- centroid of ΔABC
def orthocenter (A B C : Point) : Point := sorry  -- orthocenter of ΔABC
def midpoint (P Q : Point) : Point := sorry -- midpoint of segment PQ

-- Conditions
axiom triangle_ABC : is_triangle A B C
axiom circumcenter_O : circumcenter A B C = O
axiom centroid_G : centroid A B C = G
axiom orthocenter_H : orthocenter A B C = H
axiom midpoint_A' : midpoint B C = A'
axiom midpoint_B' : midpoint C A = B'
axiom midpoint_C' : midpoint A B = C'
axiom foot_D : perpendicular (line_through A D) (line_through B C)
axiom foot_E : perpendicular (line_through B E) (line_through C A)
axiom foot_F : perpendicular (line_through C F) (line_through A B)

-- Questions to prove
theorem homothety_image : homothety_centered G (triangle ABC) (triangle A' B' C') with_ratio 1/2 :=
sorry

theorem collinear_O_G_H : collinear {O, G, H} :=
sorry

theorem concyclic_A'B'C'DEF : concyclic {A', B', C', D, E, F} :=
sorry

end

end GeometryProof

end homothety_image_collinear_O_G_H_concyclic_A_l117_117248


namespace movie_ticket_cost_l117_117197

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

end movie_ticket_cost_l117_117197


namespace dilation_transformation_result_l117_117214

theorem dilation_transformation_result
  (x y x' y' : ℝ)
  (h₀ : x'^2 / 4 + y'^2 / 9 = 1) 
  (h₁ : x' = 2 * x)
  (h₂ : y' = 3 * y)
  (h₃ : x^2 + y^2 = 1)
  : x'^2 / 4 + y'^2 / 9 = 1 := 
by
  sorry

end dilation_transformation_result_l117_117214


namespace cost_per_piece_l117_117586

variable (totalCost : ℝ) (numberOfPizzas : ℝ) (piecesPerPizza : ℝ)

theorem cost_per_piece (h1 : totalCost = 80) (h2 : numberOfPizzas = 4) (h3 : piecesPerPizza = 5) :
  totalCost / numberOfPizzas / piecesPerPizza = 4 := by
sorry

end cost_per_piece_l117_117586


namespace complementary_event_A_l117_117859

-- Define the events
def EventA (defective : ℕ) : Prop := defective ≥ 2

def ComplementaryEvent (defective : ℕ) : Prop := defective ≤ 1

-- Question: Prove that the complementary event of event A ("at least 2 defective products") 
-- is "at most 1 defective product" given the conditions.
theorem complementary_event_A (defective : ℕ) (total : ℕ) (h_total : total = 10) :
  EventA defective ↔ ComplementaryEvent defective :=
by sorry

end complementary_event_A_l117_117859


namespace no_equal_prob_for_same_color_socks_l117_117753

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l117_117753


namespace symmetric_points_origin_l117_117839

theorem symmetric_points_origin (a b : ℝ) 
  (h1 : (-2, b) = (-a, -3)) : a - b = 5 := 
by
  -- solution steps are not included in the statement
  sorry

end symmetric_points_origin_l117_117839


namespace no_blue_socks_make_probability_half_l117_117747

theorem no_blue_socks_make_probability_half :
  ∀ (n m : ℕ), n + m = 2009 → (n - m) * (n - m) ≠ 2009 :=
begin
  intros n m h,
  by_contra H,
  sorry, -- Proof goes here
end

end no_blue_socks_make_probability_half_l117_117747


namespace derivative_at_one_l117_117676

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_one : (deriv f 1) = 2 * Real.exp 1 := by
  sorry

end derivative_at_one_l117_117676


namespace sin_minus_cos_eq_l117_117376

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l117_117376


namespace point_on_bisector_l117_117690

theorem point_on_bisector {a b : ℝ} (h : ∃ θ, θ = atan (b / a) ∧ (θ = π / 4 ∨ θ = -(3 * π / 4))) : b = -a :=
sorry

end point_on_bisector_l117_117690


namespace boat_distance_travelled_upstream_l117_117743

theorem boat_distance_travelled_upstream (v : ℝ) (d : ℝ) :
  ∀ (boat_speed_in_still_water upstream_time downstream_time : ℝ),
  boat_speed_in_still_water = 25 →
  upstream_time = 1 →
  downstream_time = 0.25 →
  d = (boat_speed_in_still_water - v) * upstream_time →
  d = (boat_speed_in_still_water + v) * downstream_time →
  d = 10 :=
by
  intros
  sorry

end boat_distance_travelled_upstream_l117_117743


namespace number_is_7625_l117_117589

-- We define x as a real number
variable (x : ℝ)

-- The condition given in the problem
def condition : Prop := x^2 + 95 = (x - 20)^2

-- The theorem we need to prove
theorem number_is_7625 (h : condition x) : x = 7.625 :=
by
  sorry

end number_is_7625_l117_117589


namespace g_675_eq_42_l117_117121

-- Define the function g on positive integers
def g : ℕ → ℕ := sorry

-- State the conditions
axiom g_multiplicative : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g_15 : g 15 = 18
axiom g_45 : g 45 = 24

-- The theorem we want to prove
theorem g_675_eq_42 : g 675 = 42 := 
by 
  sorry

end g_675_eq_42_l117_117121


namespace find_fraction_sum_l117_117212

theorem find_fraction_sum (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -2) : (1 / x) + (1 / y) = -3 :=
by
  sorry

end find_fraction_sum_l117_117212


namespace fred_balloons_remaining_l117_117970

theorem fred_balloons_remaining 
    (initial_balloons : ℕ)         -- Fred starts with these many balloons
    (given_to_sandy : ℕ)           -- Fred gives these many balloons to Sandy
    (given_to_bob : ℕ)             -- Fred gives these many balloons to Bob
    (h1 : initial_balloons = 709) 
    (h2 : given_to_sandy = 221) 
    (h3 : given_to_bob = 153) : 
    (initial_balloons - given_to_sandy - given_to_bob = 335) :=
by
  sorry

end fred_balloons_remaining_l117_117970


namespace swim_speed_in_still_water_l117_117032

-- Definitions from conditions
def downstream_speed (v_man v_stream : ℝ) : ℝ := v_man + v_stream
def upstream_speed (v_man v_stream : ℝ) : ℝ := v_man - v_stream

-- Question formatted as a proof problem
theorem swim_speed_in_still_water (v_man v_stream : ℝ)
  (h1 : downstream_speed v_man v_stream = 6)
  (h2 : upstream_speed v_man v_stream = 10) : v_man = 8 :=
by
  -- The proof will come here
  sorry

end swim_speed_in_still_water_l117_117032


namespace sin_minus_cos_l117_117363

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l117_117363


namespace point_on_angle_bisector_l117_117688

theorem point_on_angle_bisector (a b : ℝ) (h : ∃ p : ℝ, (a, b) = (p, -p)) : a = -b :=
sorry

end point_on_angle_bisector_l117_117688


namespace count_ordered_pairs_l117_117065

theorem count_ordered_pairs (x y : ℕ) (px : 0 < x) (py : 0 < y) (h : 2310 = 2 * 3 * 5 * 7 * 11) :
  (x * y = 2310 → ∃ n : ℕ, n = 32) :=
by
  sorry

end count_ordered_pairs_l117_117065


namespace edward_original_amount_l117_117067

-- Given conditions
def spent : ℝ := 16
def remaining : ℝ := 6

-- Question: How much did Edward have before he spent his money?
-- Correct answer: 22
theorem edward_original_amount : (spent + remaining) = 22 :=
by sorry

end edward_original_amount_l117_117067


namespace jorges_total_yield_l117_117242

def total_yield (good_acres clay_acres : ℕ) (good_yield clay_yield : ℕ) : ℕ :=
  good_acres * good_yield + clay_acres * clay_yield / 2

theorem jorges_total_yield :
  let acres := 60
  let good_yield_per_acre := 400
  let clay_yield_per_acre := good_yield_per_acre / 2
  let good_acres := 2 * acres / 3
  let clay_acres := acres / 3
  total_yield good_acres clay_acres good_yield_per_acre clay_yield_per_acre = 20000 :=
by
  sorry

end jorges_total_yield_l117_117242


namespace remainder_sum_div_l117_117685

theorem remainder_sum_div (n : ℤ) : ((9 - n) + (n + 5)) % 9 = 5 := by
  sorry

end remainder_sum_div_l117_117685


namespace right_triangle_perimeter_l117_117183

def right_triangle_circumscribed_perimeter (r c : ℝ) (a b : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter : 
  ∀ (a b : ℝ),
  (4 : ℝ) * (a + b + (26 : ℝ)) = a * b ∧ a^2 + b^2 = (26 : ℝ)^2 →
  right_triangle_circumscribed_perimeter 4 26 a b = 60 := sorry

end right_triangle_perimeter_l117_117183


namespace largest_number_with_four_digits_divisible_by_72_is_9936_l117_117021

theorem largest_number_with_four_digits_divisible_by_72_is_9936 :
  ∃ n : ℕ, (n < 10000 ∧ n ≥ 1000) ∧ (72 ∣ n) ∧ (∀ m : ℕ, (m < 10000 ∧ m ≥ 1000) ∧ (72 ∣ m) → m ≤ n) :=
sorry

end largest_number_with_four_digits_divisible_by_72_is_9936_l117_117021


namespace calc_problem_l117_117783

def odot (a b : ℕ) : ℕ := a * b - (a + b)

theorem calc_problem : odot 6 (odot 5 4) = 49 :=
by
  sorry

end calc_problem_l117_117783


namespace tom_spent_on_videogames_l117_117614

theorem tom_spent_on_videogames (batman_game superman_game : ℝ) 
  (h1 : batman_game = 13.60) 
  (h2 : superman_game = 5.06) : 
  batman_game + superman_game = 18.66 :=
by 
  sorry

end tom_spent_on_videogames_l117_117614


namespace value_of_m_l117_117420

-- Defining the quadratic equation condition
def quadratic_eq (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + 3 * x + m^2 - 4

-- Defining the condition where the constant term in the quadratic equation is 0
def constant_term_zero (m : ℝ) : Prop := m^2 - 4 = 0

-- Stating the proof problem: given the conditions, prove that m = -2
theorem value_of_m (m : ℝ) (h1 : constant_term_zero m) (h2 : m ≠ 2) : m = -2 :=
by {
  sorry -- Proof to be developed
}

end value_of_m_l117_117420


namespace ball_hits_ground_time_l117_117137

theorem ball_hits_ground_time :
  ∃ t : ℚ, -20 * t^2 + 30 * t + 50 = 0 ∧ t = 5 / 2 :=
sorry

end ball_hits_ground_time_l117_117137


namespace value_of_a_l117_117417

theorem value_of_a (a b : ℤ) (h : (∀ x, x^2 - x - 1 = 0 → a * x^17 + b * x^16 + 1 = 0)) : a = 987 :=
by 
  sorry

end value_of_a_l117_117417


namespace calculate_value_of_expression_l117_117922

theorem calculate_value_of_expression :
  (2523 - 2428)^2 / 121 = 75 :=
by
  -- calculation steps here
  sorry

end calculate_value_of_expression_l117_117922


namespace laura_pants_count_l117_117714

def cost_of_pants : ℕ := 54
def cost_of_shirt : ℕ := 33
def number_of_shirts : ℕ := 4
def total_money_given : ℕ := 250
def change_received : ℕ := 10

def laura_spent : ℕ := total_money_given - change_received
def total_cost_shirts : ℕ := cost_of_shirt * number_of_shirts
def spent_on_pants : ℕ := laura_spent - total_cost_shirts
def pairs_of_pants_bought : ℕ := spent_on_pants / cost_of_pants

theorem laura_pants_count : pairs_of_pants_bought = 2 :=
by
  sorry

end laura_pants_count_l117_117714


namespace problem_l117_117534

theorem problem (a b : ℝ) (h : a > b) : a / 3 > b / 3 :=
sorry

end problem_l117_117534


namespace inequality_proof_l117_117831

variable (b c : ℝ)
variable (hb : b > 0) (hc : c > 0)

theorem inequality_proof :
  (b - c) ^ 2011 * (b + c) ^ 2011 * (c - b) ^ 2011 ≥ (b ^ 2011 - c ^ 2011) * (b ^ 2011 + c ^ 2011) * (c ^ 2011 - b ^ 2011) :=
  sorry

end inequality_proof_l117_117831


namespace anna_original_money_l117_117805

theorem anna_original_money (x : ℝ) (h : (3 / 4) * x = 24) : x = 32 :=
by
  sorry

end anna_original_money_l117_117805


namespace minimum_value_l117_117189

theorem minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) :
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) ≥ 9 / 4 :=
by sorry

end minimum_value_l117_117189


namespace sin_minus_cos_eq_l117_117380

-- Conditions
variable (θ : ℝ)
variable (hθ1 : 0 < θ ∧ θ < π / 2)
variable (hθ2 : Real.tan θ = 1 / 3)

-- Theorem stating the question and the correct answer
theorem sin_minus_cos_eq :
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 :=
sorry -- Proof goes here

end sin_minus_cos_eq_l117_117380


namespace inequality_proof_problem_l117_117662

theorem inequality_proof_problem (a b : ℝ) (h : a < b ∧ b < 0) : (1 / (a - b) ≤ 1 / a) :=
sorry

end inequality_proof_problem_l117_117662


namespace lcm_inequality_l117_117208

theorem lcm_inequality (m n : ℕ) (h : n > m) :
  Nat.lcm m n + Nat.lcm (m + 1) (n + 1) ≥ 2 * m * n / Real.sqrt (n - m) := 
sorry

end lcm_inequality_l117_117208


namespace determine_b_l117_117570

theorem determine_b (A B C : ℝ) (a b c : ℝ)
  (angle_C_eq_4A : C = 4 * A)
  (a_eq_30 : a = 30)
  (c_eq_48 : c = 48)
  (law_of_sines : ∀ x y, x / Real.sin A = y / Real.sin (4 * A))
  (cos_eq_solution : 4 * Real.cos A ^ 3 - 4 * Real.cos A = 8 / 5) :
  ∃ b : ℝ, b = 30 * (5 - 20 * (1 - Real.cos A ^ 2) + 16 * (1 - Real.cos A ^ 2) ^ 2) :=
by 
  sorry

end determine_b_l117_117570


namespace complex_mul_l117_117172

theorem complex_mul (i : ℂ) (hi : i * i = -1) : (1 - i) * (3 + i) = 4 - 2 * i :=
by
  sorry

end complex_mul_l117_117172


namespace product_value_l117_117160

-- Definitions of each term
def term (n : Nat) : Rat :=
  1 + 1 / (n^2 : ℚ)

-- Define the product of these terms
def product : Rat :=
  term 1 * term 2 * term 3 * term 4 * term 5 * term 6

-- The proof problem statement that needs to be verified
theorem product_value :
  product = 16661 / 3240 :=
sorry

end product_value_l117_117160


namespace jeffrey_fills_crossword_l117_117003

noncomputable def prob_fill_crossword : ℚ :=
  let total_clues := 10
  let prob_knowing_all_clues := (1 / 2) ^ total_clues
  let prob_case_1 := (2 ^ 5) / (2 ^ total_clues)
  let prob_case_2 := (2 ^ 5) / (2 ^ total_clues)
  let prob_case_3 := 25 / (2 ^ total_clues)
  let overcounted_case := prob_knowing_all_clues
  (prob_case_1 + prob_case_2 + prob_case_3 - overcounted_case)

theorem jeffrey_fills_crossword : prob_fill_crossword = 11 / 128 := by
  sorry

end jeffrey_fills_crossword_l117_117003


namespace polar_to_cartesian_conversion_l117_117520

noncomputable def polarToCartesian (ρ θ : ℝ) : ℝ × ℝ :=
  let x := ρ * Real.cos θ
  let y := ρ * Real.sin θ
  (x, y)

theorem polar_to_cartesian_conversion :
  polarToCartesian 4 (Real.pi / 3) = (2, 2 * Real.sqrt 3) :=
by
  sorry

end polar_to_cartesian_conversion_l117_117520


namespace no_equal_prob_for_same_color_socks_l117_117755

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l117_117755


namespace isosceles_triangle_circumradius_l117_117530

theorem isosceles_triangle_circumradius (b : ℝ) (s : ℝ) (R : ℝ) (hb : b = 6) (hs : s = 5) :
  R = 25 / 8 :=
by 
  sorry

end isosceles_triangle_circumradius_l117_117530


namespace smallest_number_is_C_l117_117449

def A : ℕ := 36
def B : ℕ := 27 + 5
def C : ℕ := 3 * 10
def D : ℕ := 40 - 3

theorem smallest_number_is_C :
  min (min A B) (min C D) = C :=
by
  -- Proof steps go here
  sorry

end smallest_number_is_C_l117_117449


namespace fraction_of_b_eq_two_thirds_l117_117029

theorem fraction_of_b_eq_two_thirds (A B : ℝ) (x : ℝ) (h1 : A + B = 1210) (h2 : B = 484)
  (h3 : (2/3) * A = x * B) : x = 2/3 :=
by
  sorry

end fraction_of_b_eq_two_thirds_l117_117029


namespace roses_cut_l117_117465

def r_before := 13
def r_after := 14

theorem roses_cut : r_after - r_before = 1 := by
  sorry

end roses_cut_l117_117465


namespace q_is_false_given_conditions_l117_117423

theorem q_is_false_given_conditions
  (h₁: ¬(p ∧ q) = true) 
  (h₂: ¬¬p = true) 
  : q = false := 
sorry

end q_is_false_given_conditions_l117_117423


namespace tree_height_at_2_years_l117_117501

theorem tree_height_at_2_years (h : ℕ → ℕ) 
  (h_growth : ∀ n, h (n + 1) = 3 * h n) 
  (h_5 : h 5 = 243) : 
  h 2 = 9 := 
sorry

end tree_height_at_2_years_l117_117501


namespace gcd_459_357_eq_51_l117_117918

theorem gcd_459_357_eq_51 :
  gcd 459 357 = 51 := 
by
  sorry

end gcd_459_357_eq_51_l117_117918


namespace concave_sequence_count_l117_117012

   theorem concave_sequence_count (m : ℕ) (h : 2 ≤ m) :
     ∀ b_0, (b_0 = 1 ∨ b_0 = 2) → 
     (∃ b : ℕ → ℕ, (∀ k, 2 ≤ k ∧ k ≤ m → b k + b (k - 2) ≤ 2 * b (k - 1)) → 
     (∃ S : ℕ, S ≤ 2^m)) :=
   by 
     sorry
   
end concave_sequence_count_l117_117012


namespace tan_triple_angle_formula_l117_117356

variable (θ : ℝ)
variable (h : Real.tan θ = 4)

theorem tan_triple_angle_formula : Real.tan (3 * θ) = 52 / 47 :=
by
  sorry  -- Proof is omitted

end tan_triple_angle_formula_l117_117356


namespace matrix_power_is_correct_l117_117957

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]
def A_cubed : Matrix (Fin 2) (Fin 2) ℤ := !![3, -6; 6, -3]

theorem matrix_power_is_correct : A ^ 3 = A_cubed := by 
  sorry

end matrix_power_is_correct_l117_117957


namespace polynomial_roots_sum_reciprocal_l117_117062

open Polynomial

theorem polynomial_roots_sum_reciprocal (a b c : ℝ) (h : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) :
    (40 * a^3 - 70 * a^2 + 32 * a - 3 = 0) ∧
    (40 * b^3 - 70 * b^2 + 32 * b - 3 = 0) ∧
    (40 * c^3 - 70 * c^2 + 32 * c - 3 = 0) →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = 3 :=
by
  sorry

end polynomial_roots_sum_reciprocal_l117_117062


namespace sum_of_primes_l117_117089

theorem sum_of_primes (p1 p2 p3 : ℕ) (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3) 
    (h : p1 * p2 * p3 = 31 * (p1 + p2 + p3)) :
    p1 + p2 + p3 = 51 := by
  sorry

end sum_of_primes_l117_117089


namespace sin_minus_cos_l117_117367

theorem sin_minus_cos (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < (Real.pi / 2)) (hθ3 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := by
  sorry

end sin_minus_cos_l117_117367


namespace binomial_remainder_mod_3_l117_117655

open BigOperators

theorem binomial_remainder_mod_3 :
  (1 + nat.choose 27 1 + nat.choose 27 2 + nat.choose 27 27) % 3 = 2 := by
sorry

end binomial_remainder_mod_3_l117_117655


namespace find_parameters_l117_117142

theorem find_parameters (s h : ℝ) :
  (∀ (x y t : ℝ), (x = s + 3 * t) ∧ (y = 2 + h * t) ∧ (y = 5 * x - 7)) → (s = 9 / 5 ∧ h = 15) :=
by
  sorry

end find_parameters_l117_117142


namespace find_z_l117_117991

noncomputable def w : ℝ := sorry
noncomputable def x : ℝ := (5 * w) / 4
noncomputable def y : ℝ := 1.40 * w

theorem find_z (z : ℝ) : x = (1 - z / 100) * y → z = 10.71 :=
by
  sorry

end find_z_l117_117991


namespace sides_of_regular_polygon_l117_117182

theorem sides_of_regular_polygon {n : ℕ} (h₁ : n ≥ 3)
  (h₂ : (n * (n - 3)) / 2 + 6 = 2 * n) : n = 4 :=
sorry

end sides_of_regular_polygon_l117_117182


namespace count_polynomials_l117_117236

def is_polynomial (expr : String) : Bool :=
  match expr with
  | "-7"            => true
  | "x"             => true
  | "m^2 + 1/m"     => false
  | "x^2*y + 5"     => true
  | "(x + y)/2"     => true
  | "-5ab^3c^2"     => true
  | "1/y"           => false
  | _               => false

theorem count_polynomials :
  let expressions := ["-7", "x", "m^2 + 1/m", "x^2*y + 5", "(x + y)/2", "-5ab^3c^2", "1/y"]
  List.filter is_polynomial expressions |>.length = 5 :=
by
  sorry

end count_polynomials_l117_117236


namespace students_at_start_of_year_l117_117515

variable (S : ℕ)

def initial_students := S
def students_left := 6
def students_new := 42
def end_year_students := 47

theorem students_at_start_of_year :
  initial_students + (students_new - students_left) = end_year_students → initial_students = 11 :=
by
  sorry

end students_at_start_of_year_l117_117515


namespace range_of_a_l117_117657

theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, x^2 + 2 * |x - a| ≥ a^2) : -1 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l117_117657


namespace div_4800_by_125_expr_13_mul_74_add_27_mul_13_sub_13_l117_117810

theorem div_4800_by_125 : 4800 / 125 = 38.4 :=
by
  sorry

theorem expr_13_mul_74_add_27_mul_13_sub_13 : 13 * 74 + 27 * 13 - 13 = 1300 :=
by
  sorry

end div_4800_by_125_expr_13_mul_74_add_27_mul_13_sub_13_l117_117810


namespace no_negative_product_l117_117013

theorem no_negative_product (x y : ℝ) (n : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) 
(h1 : x ^ (2 * n) - y ^ (2 * n) > x) (h2 : y ^ (2 * n) - x ^ (2 * n) > y) : x * y ≥ 0 :=
sorry

end no_negative_product_l117_117013


namespace negation_proposition_l117_117906

theorem negation_proposition :
  ¬(∀ x : ℝ, |x - 2| < 3) ↔ ∃ x : ℝ, |x - 2| ≥ 3 :=
by
  sorry

end negation_proposition_l117_117906


namespace neznaika_incorrect_l117_117303

-- Define the average consumption conditions
def average_consumption_december (total_consumption total_days_cons_december : ℕ) : Prop :=
  total_consumption = 10 * total_days_cons_december

def average_consumption_january (total_consumption total_days_cons_january : ℕ) : Prop :=
  total_consumption = 5 * total_days_cons_january

-- Define the claim to be disproven
def neznaika_claim (days_december_at_least_10 days_january_at_least_10 : ℕ) : Prop :=
  days_december_at_least_10 > days_january_at_least_10

-- Proof statement that the claim is incorrect
theorem neznaika_incorrect (total_days_cons_december total_days_cons_january total_consumption_dec total_consumption_jan : ℕ)
    (days_december_at_least_10 days_january_at_least_10 : ℕ)
    (h1 : average_consumption_december total_consumption_dec total_days_cons_december)
    (h2 : average_consumption_january total_consumption_jan total_days_cons_january)
    (h3 : total_days_cons_december = 31)
    (h4 : total_days_cons_january = 31)
    (h5 : days_december_at_least_10 ≤ total_days_cons_december)
    (h6 : days_january_at_least_10 ≤ total_days_cons_january)
    (h7 : days_december_at_least_10 = 1)
    (h8 : days_january_at_least_10 = 15) : 
    ¬ neznaika_claim days_december_at_least_10 days_january_at_least_10 :=
by
  sorry

end neznaika_incorrect_l117_117303


namespace train_passes_jogger_in_36_seconds_l117_117480

/-- A jogger runs at 9 km/h, 240m ahead of a train moving at 45 km/h.
The train is 120m long. Prove the train passes the jogger in 36 seconds. -/
theorem train_passes_jogger_in_36_seconds
  (distance_ahead : ℝ)
  (jogger_speed_km_hr train_speed_km_hr train_length_m : ℝ)
  (jogger_speed_m_s train_speed_m_s relative_speed_m_s distance_to_cover time_to_pass : ℝ)
  (h1 : distance_ahead = 240)
  (h2 : jogger_speed_km_hr = 9)
  (h3 : train_speed_km_hr = 45)
  (h4 : train_length_m = 120)
  (h5 : jogger_speed_m_s = jogger_speed_km_hr * 1000 / 3600)
  (h6 : train_speed_m_s = train_speed_km_hr * 1000 / 3600)
  (h7 : relative_speed_m_s = train_speed_m_s - jogger_speed_m_s)
  (h8 : distance_to_cover = distance_ahead + train_length_m)
  (h9 : time_to_pass = distance_to_cover / relative_speed_m_s) :
  time_to_pass = 36 := 
sorry

end train_passes_jogger_in_36_seconds_l117_117480


namespace range_of_a_l117_117829

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * x

theorem range_of_a (a : ℝ) (h : f (2 * a) < f (a - 1)) : a < -1 :=
by
  -- Steps of the proof would be placed here, but we're skipping them for now
  sorry

end range_of_a_l117_117829


namespace koi_fish_count_l117_117267

-- Define the initial conditions as variables
variables (total_fish_initial : ℕ) (goldfish_end : ℕ) (days_in_week : ℕ)
          (weeks : ℕ) (koi_add_day : ℕ) (goldfish_add_day : ℕ)

-- Expressing the problem's constraints
def problem_conditions :=
  total_fish_initial = 280 ∧
  goldfish_end = 200 ∧
  days_in_week = 7 ∧
  weeks = 3 ∧
  koi_add_day = 2 ∧
  goldfish_add_day = 5

-- Calculating the expected results based on the constraints
def total_fish_end := total_fish_initial + weeks * days_in_week * (koi_add_day + goldfish_add_day)
def koi_fish_end := total_fish_end - goldfish_end

-- The theorem to prove the number of koi fish at the end is 227
theorem koi_fish_count : problem_conditions → koi_fish_end = 227 := by
  sorry

end koi_fish_count_l117_117267


namespace linear_inequality_inequality_l117_117354

theorem linear_inequality_inequality (k : ℝ) (x : ℝ) : 
  (k - 1) * x ^ |k| + 3 ≥ 0 → is_linear x (k - 1) * x ^ |k| + 3 → k = -1 :=
by 
  sorry

end linear_inequality_inequality_l117_117354


namespace p_q_sum_is_19_l117_117873

open Finset Perm Nat

noncomputable def permutations_of_six := univ.perm

def set_T (p : ℕ) :=
  {σ ∈ permutations_of_six | ¬(σ 0 = 1 ∨ σ 0 = 2)}

def favorable_permutations (p : ℕ) :=
  {σ ∈ set_T p | σ 2 = 3}

def p_and_q_sum : ℕ :=
  let p := 3 in
  let q := 16 in 
  p + q

theorem p_q_sum_is_19 : p_and_q_sum = 19 := by
  sorry

end p_q_sum_is_19_l117_117873


namespace minimal_rope_cost_l117_117456

theorem minimal_rope_cost :
  let pieces_needed := 10
  let length_per_piece := 6 -- inches
  let total_length_needed := pieces_needed * length_per_piece -- inches
  let one_foot_length := 12 -- inches
  let cost_six_foot_rope := 5 -- dollars
  let cost_one_foot_rope := 1.25 -- dollars
  let six_foot_length := 6 * one_foot_length -- inches
  let one_foot_total_cost := (total_length_needed / one_foot_length) * cost_one_foot_rope
  let six_foot_total_cost := cost_six_foot_rope
  total_length_needed <= six_foot_length ∧ six_foot_total_cost < one_foot_total_cost →
  six_foot_total_cost = 5 := sorry

end minimal_rope_cost_l117_117456


namespace cars_fell_in_lot_l117_117466

theorem cars_fell_in_lot (initial_cars went_out_cars came_in_cars final_cars: ℕ) (h1 : initial_cars = 25) 
    (h2 : went_out_cars = 18) (h3 : came_in_cars = 12) (h4 : final_cars = initial_cars - went_out_cars + came_in_cars) :
    initial_cars - final_cars = 6 :=
    sorry

end cars_fell_in_lot_l117_117466


namespace proposition_induction_l117_117978

variable (P : ℕ → Prop)
variable (k : ℕ)

theorem proposition_induction (h : ∀ k : ℕ, P k → P (k + 1))
    (h9 : ¬ P 9) : ¬ P 8 :=
by
  sorry

end proposition_induction_l117_117978


namespace line_b_y_intercept_l117_117879

theorem line_b_y_intercept :
  ∃ c : ℝ, (∀ x : ℝ, (-3) * x + c = -3 * x + 7) ∧ ∃ p : ℝ × ℝ, (p = (5, -2)) → -3 * 5 + c = -2 →
  c = 13 :=
by
  sorry

end line_b_y_intercept_l117_117879


namespace range_of_m_l117_117536

open Real

noncomputable def complex_modulus_log_condition (m : ℝ) : Prop :=
  Complex.abs (Complex.log (m : ℂ) / Complex.log 2 + Complex.I * 4) ≤ 5

theorem range_of_m (m : ℝ) (h : complex_modulus_log_condition m) : 
  (1 / 8 : ℝ) ≤ m ∧ m ≤ (8 : ℝ) :=
sorry

end range_of_m_l117_117536


namespace trig_identity_l117_117357

theorem trig_identity 
  (θ : ℝ) 
  (h1 : 0 < θ) 
  (h2 : θ < (Real.pi / 2)) 
  (h3 : Real.tan θ = 1 / 3) :
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 :=
by
  sorry

end trig_identity_l117_117357


namespace point_on_line_iff_l117_117727

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given points O, A, B, and X in a vector space V, prove that X lies on the line AB if and only if
there exists a scalar t such that the position vector of X is a linear combination of the position vectors
of A and B with respect to O. -/
theorem point_on_line_iff (O A B X : V) :
  (∃ t : ℝ, X - O = t • (A - O) + (1 - t) • (B - O)) ↔ (∃ t : ℝ, ∃ (t : ℝ), X - O = (1 - t) • (A - O) + t • (B - O)) :=
sorry

end point_on_line_iff_l117_117727


namespace correct_exponent_operation_l117_117924

theorem correct_exponent_operation (a b : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧ 
  (6 * a^6 / (2 * a^2) ≠ 3 * a^3) ∧ 
  ((-a^2)^3 = -a^6) ∧ 
  ((-2 * a * b^2)^2 ≠ 2 * a^2 * b^4) :=
by
  sorry

end correct_exponent_operation_l117_117924


namespace find_xyz_l117_117539

theorem find_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 17) 
  (h3 : x^3 + y^3 + z^3 = 27) : 
  x * y * z = 32 / 3 :=
  sorry

end find_xyz_l117_117539


namespace triangle_RS_length_l117_117571

theorem triangle_RS_length (PQ QR PS QS RS : ℝ)
  (h1 : PQ = 8) (h2 : QR = 8) (h3 : PS = 10) (h4 : QS = 5) :
  RS = 3.5 :=
by
  sorry

end triangle_RS_length_l117_117571


namespace particle_speed_interval_l117_117632

noncomputable def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 5, 5 * t - 7)

theorem particle_speed_interval (k : ℝ) :
  let start_pos := particle_position k
  let end_pos := particle_position (k + 2)
  let delta_x := end_pos.1 - start_pos.1
  let delta_y := end_pos.2 - start_pos.2
  let speed := Real.sqrt (delta_x^2 + delta_y^2)
  speed = 2 * Real.sqrt 34 := by
  sorry

end particle_speed_interval_l117_117632


namespace six_digit_number_division_l117_117475

theorem six_digit_number_division :
  ∃ a b p : ℕ, 
    (111111 * a = 1111 * b * 233 + p) ∧ 
    (11111 * a = 111 * b * 233 + p - 1000) ∧
    (111111 * 7 = 777777) ∧
    (1111 * 3 = 3333) :=
by
  sorry

end six_digit_number_division_l117_117475


namespace all_positive_l117_117514

theorem all_positive (a1 a2 a3 a4 a5 a6 a7 : ℝ)
  (h1 : a1 + a2 + a3 + a4 > a5 + a6 + a7)
  (h2 : a1 + a2 + a3 + a5 > a4 + a6 + a7)
  (h3 : a1 + a2 + a3 + a6 > a4 + a5 + a7)
  (h4 : a1 + a2 + a3 + a7 > a4 + a5 + a6)
  (h5 : a1 + a2 + a4 + a5 > a3 + a6 + a7)
  (h6 : a1 + a2 + a4 + a6 > a3 + a5 + a7)
  (h7 : a1 + a2 + a4 + a7 > a3 + a5 + a6)
  (h8 : a1 + a2 + a5 + a6 > a3 + a4 + a7)
  (h9 : a1 + a2 + a5 + a7 > a3 + a4 + a6)
  (h10 : a1 + a2 + a6 + a7 > a3 + a4 + a5)
  (h11 : a1 + a3 + a4 + a5 > a2 + a6 + a7)
  (h12 : a1 + a3 + a4 + a6 > a2 + a5 + a7)
  (h13 : a1 + a3 + a4 + a7 > a2 + a5 + a6)
  (h14 : a1 + a3 + a5 + a6 > a2 + a4 + a7)
  (h15 : a1 + a3 + a5 + a7 > a2 + a4 + a6)
  (h16 : a1 + a3 + a6 + a7 > a2 + a4 + a5)
  (h17 : a1 + a4 + a5 + a6 > a2 + a3 + a7)
  (h18 : a1 + a4 + a5 + a7 > a2 + a3 + a6)
  (h19 : a1 + a4 + a6 + a7 > a2 + a3 + a5)
  (h20 : a1 + a5 + a6 + a7 > a2 + a3 + a4)
  (h21 : a2 + a3 + a4 + a5 > a1 + a6 + a7)
  (h22 : a2 + a3 + a4 + a6 > a1 + a5 + a7)
  (h23 : a2 + a3 + a4 + a7 > a1 + a5 + a6)
  (h24 : a2 + a3 + a5 + a6 > a1 + a4 + a7)
  (h25 : a2 + a3 + a5 + a7 > a1 + a4 + a6)
  (h26 : a2 + a3 + a6 + a7 > a1 + a4 + a5)
  (h27 : a2 + a4 + a5 + a6 > a1 + a3 + a7)
  (h28 : a2 + a4 + a5 + a7 > a1 + a3 + a6)
  (h29 : a2 + a4 + a6 + a7 > a1 + a3 + a5)
  (h30 : a2 + a5 + a6 + a7 > a1 + a3 + a4)
  (h31 : a3 + a4 + a5 + a6 > a1 + a2 + a7)
  (h32 : a3 + a4 + a5 + a7 > a1 + a2 + a6)
  (h33 : a3 + a4 + a6 + a7 > a1 + a2 + a5)
  (h34 : a3 + a5 + a6 + a7 > a1 + a2 + a4)
  (h35 : a4 + a5 + a6 + a7 > a1 + a2 + a3)
: a1 > 0 ∧ a2 > 0 ∧ a3 > 0 ∧ a4 > 0 ∧ a5 > 0 ∧ a6 > 0 ∧ a7 > 0 := 
sorry

end all_positive_l117_117514


namespace find_f_2_l117_117091

noncomputable def f (a b x : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 :=
by
  sorry

end find_f_2_l117_117091


namespace true_statement_for_f_l117_117245

variable (c : ℝ) (f : ℝ → ℝ)

theorem true_statement_for_f :
  (∀ x : ℝ, f x = x^2 - 2 * x + c) → (∀ x : ℝ, f x ≥ c - 1) :=
by
  sorry

end true_statement_for_f_l117_117245


namespace other_number_is_31_l117_117585

namespace LucasProblem

-- Definitions of the integers a and b and the condition on their sum
variables (a b : ℤ)
axiom h_sum : 3 * a + 4 * b = 161
axiom h_one_is_17 : a = 17 ∨ b = 17

-- The theorem we need to prove
theorem other_number_is_31 (h_one_is_17 : a = 17 ∨ b = 17) : 
  (b = 17 → a = 31) ∧ (a = 17 → false) :=
by
  sorry

end LucasProblem

end other_number_is_31_l117_117585


namespace xy_squares_l117_117414

theorem xy_squares (x y : ℤ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 := 
by 
  sorry

end xy_squares_l117_117414


namespace no_equal_prob_for_same_color_socks_l117_117754

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end no_equal_prob_for_same_color_socks_l117_117754


namespace quadratic_completing_square_t_l117_117285

theorem quadratic_completing_square_t : 
  ∀ (x k t : ℝ), (4 * x^2 + 16 * x - 400 = 0) →
  ((x + k)^2 = t) →
  t = 104 :=
by
  intros x k t h1 h2
  sorry

end quadratic_completing_square_t_l117_117285


namespace almonds_walnuts_ratio_l117_117042

-- Define the given weights and parts
def w_a : ℝ := 107.14285714285714
def w_m : ℝ := 150
def p_a : ℝ := 5

-- Now we will formulate the statement to prove the ratio of almonds to walnuts
theorem almonds_walnuts_ratio : 
  ∃ (p_w : ℝ), p_a / p_w = 5 / 2 :=
by
  -- It is given that p_a / p_w = 5 / 2, we need to find p_w
  sorry

end almonds_walnuts_ratio_l117_117042


namespace trig_eq_solutions_l117_117335

theorem trig_eq_solutions (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  3 * Real.sin x = 1 + Real.cos (2 * x) ↔ x = Real.pi / 6 ∨ x = 5 * Real.pi / 6 :=
by
  sorry

end trig_eq_solutions_l117_117335


namespace area_ratio_GHI_JKL_l117_117615

-- Given conditions
def side_lengths_GHI : ℕ × ℕ × ℕ := (6, 8, 10)
def side_lengths_JKL : ℕ × ℕ × ℕ := (9, 12, 15)

-- Function to calculate the area of a right triangle given the lengths of the legs
def right_triangle_area (a b : ℕ) : ℕ :=
  (a * b) / 2

-- Function to determine if a triangle is a right triangle given its side lengths
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Define the main theorem
theorem area_ratio_GHI_JKL :
  let (a₁, b₁, c₁) := side_lengths_GHI
  let (a₂, b₂, c₂) := side_lengths_JKL
  is_right_triangle a₁ b₁ c₁ →
  is_right_triangle a₂ b₂ c₂ →
  right_triangle_area a₁ b₁ % right_triangle_area a₂ b₂ = 4 / 9 :=
by sorry

end area_ratio_GHI_JKL_l117_117615


namespace product_of_fractions_l117_117325

theorem product_of_fractions :
  (3/4) * (4/5) * (5/6) * (6/7) = 3/7 :=
by
  sorry

end product_of_fractions_l117_117325


namespace geometric_sequence_first_term_l117_117604

noncomputable def first_term_of_geometric_sequence (a r : ℝ) : ℝ :=
  a

theorem geometric_sequence_first_term 
  (a r : ℝ)
  (h1 : a * r^3 = 720)   -- The fourth term is 6!
  (h2 : a * r^6 = 5040)  -- The seventh term is 7!
  : first_term_of_geometric_sequence a r = 720 / 7 :=
sorry

end geometric_sequence_first_term_l117_117604


namespace julia_monday_kids_l117_117109

theorem julia_monday_kids (x : ℕ) (h1 : x + 14 = 16) : x = 2 := 
by
  sorry

end julia_monday_kids_l117_117109


namespace triangle_area_is_9sqrt2_l117_117071

noncomputable def triangle_area_with_given_medians_and_angle (CM BN : ℝ) (angle_BKM : ℝ) : ℝ :=
  let centroid_division_ratio := (2.0 / 3.0)
  let BK := centroid_division_ratio * BN
  let MK := (1.0 / 3.0) * CM
  let area_BKM := (1.0 / 2.0) * BK * MK * Real.sin angle_BKM
  6.0 * area_BKM

theorem triangle_area_is_9sqrt2 :
  triangle_area_with_given_medians_and_angle 6 4.5 (Real.pi / 4) = 9 * Real.sqrt 2 :=
by
  sorry

end triangle_area_is_9sqrt2_l117_117071


namespace find_angle_C_find_area_of_triangle_l117_117694

-- Given triangle ABC with sides a, b, and c opposite to angles A, B, and C respectively
-- And given conditions: c * cos B = (2a - b) * cos C

variable (a b c : ℝ) (A B C : ℝ)
variable (h1 : c * Real.cos B = (2 * a - b) * Real.cos C)
variable (h2 : c = 2)
variable (h3 : a + b + c = 2 * Real.sqrt 3 + 2)

-- Prove that angle C = π / 3
theorem find_angle_C : C = Real.pi / 3 :=
by sorry

-- Given angle C, side c, and perimeter, prove the area of triangle ABC
theorem find_area_of_triangle (h4 : C = Real.pi / 3) : 
  1 / 2 * a * b * Real.sin C = 2 * Real.sqrt 3 / 3 :=
by sorry

end find_angle_C_find_area_of_triangle_l117_117694


namespace problem_eval_expression_l117_117649

theorem problem_eval_expression :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by
  sorry

end problem_eval_expression_l117_117649


namespace no_strategy_for_vasya_tolya_l117_117764

-- This definition encapsulates the conditions and question
def players_game (coins : ℕ) : Prop :=
  ∀ p v t : ℕ, 
    (1 ≤ p ∧ p ≤ 4) ∧ (1 ≤ v ∧ v ≤ 2) ∧ (1 ≤ t ∧ t ≤ 2) →
    (∃ (n : ℕ), coins = 5 * n)

-- Theorem formalizing the problem's conclusion
theorem no_strategy_for_vasya_tolya (n : ℕ) (h : n = 300) : 
  ¬ ∀ (v t : ℕ), 
     (1 ≤ v ∧ v ≤ 2) ∧ (1 ≤ t ∧ t ≤ 2) →
     players_game (n - v - t) :=
by
  intro h
  sorry -- Skip the proof, as it is not required

end no_strategy_for_vasya_tolya_l117_117764


namespace depth_of_river_bank_l117_117737

theorem depth_of_river_bank (top_width bottom_width area depth : ℝ) 
  (h₁ : top_width = 12)
  (h₂ : bottom_width = 8)
  (h₃ : area = 500)
  (h₄ : area = (1 / 2) * (top_width + bottom_width) * depth) :
  depth = 50 :=
sorry

end depth_of_river_bank_l117_117737


namespace ladder_rungs_count_l117_117044

theorem ladder_rungs_count :
  ∃ (n : ℕ), ∀ (start mid : ℕ),
    start = n / 2 →
    mid = ((start + 5 - 7) + 8 + 7) →
    mid = n →
    n = 27 :=
by
  sorry

end ladder_rungs_count_l117_117044


namespace equal_angles_not_necessarily_vertical_l117_117145

-- Define what it means for angles to be vertical
def is_vertical_angle (a b : ℝ) : Prop :=
∃ l1 l2 : ℝ, a = 180 - b ∧ (l1 + l2 == 180 ∨ l1 == 0 ∨ l2 == 0)

-- Define what it means for angles to be equal
def are_equal_angles (a b : ℝ) : Prop := a = b

-- Proposition to be proved
theorem equal_angles_not_necessarily_vertical (a b : ℝ) (h : are_equal_angles a b) : ¬ is_vertical_angle a b :=
by
  sorry

end equal_angles_not_necessarily_vertical_l117_117145


namespace no_possible_blue_socks_l117_117748

theorem no_possible_blue_socks : 
  ∀ (n m : ℕ), n + m = 2009 → (n - m)^2 ≠ 2009 := 
by
  intros n m h
  sorry

end no_possible_blue_socks_l117_117748


namespace ratio_of_speeds_l117_117037

theorem ratio_of_speeds (v_A v_B : ℝ)
  (h₀ : 4 * v_A = abs (600 - 4 * v_B))
  (h₁ : 9 * v_A = abs (600 - 9 * v_B)) :
  v_A / v_B = 2 / 3 :=
sorry

end ratio_of_speeds_l117_117037


namespace complement_U_A_complement_U_B_intersection_A_complement_U_B_union_complement_U_A_B_l117_117681

open Set

variable {α : Type*}

def U : Set ℝ := univ
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3 ∨ 4 < x ∧ x < 6}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 5}

theorem complement_U_A : (U \ A) = {x | x < 1 ∨ 3 < x ∧ x ≤ 4 ∨ 6 ≤ x} := sorry

theorem complement_U_B : (U \ B) = {x | x < 2 ∨ 5 ≤ x} := sorry

theorem intersection_A_complement_U_B : (A ∩ (U \ B)) = {x | 1 ≤ x ∧ x < 2 ∨ 5 ≤ x ∧ x < 6} := sorry

theorem union_complement_U_A_B : ((U \ A) ∪ B) = {x | x < 1 ∨ 2 ≤ x ∧ x < 5 ∨ 6 ≤ x} := sorry

end complement_U_A_complement_U_B_intersection_A_complement_U_B_union_complement_U_A_B_l117_117681


namespace matrix_cube_l117_117956

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by
  sorry

end matrix_cube_l117_117956


namespace meeting_success_probability_l117_117631

open Set

noncomputable def meeting_probability : ℝ :=
  let Ω := Icc (0:ℝ) 2 × Icc (0:ℝ) 2 × Icc (0:ℝ) 2 × Icc (0:ℝ) 2 in
  let valid_region := {t | ∃ x y w z: ℝ, t = (x, y, w, z) ∧ z > x ∧ z > y ∧ z > w ∧
                       |x - y| ≤ 0.5 ∧ |x - w| ≤ 0.5 ∧ |y - w| ≤ 0.5} in
  (volume valid_region) / (volume Ω)

theorem meeting_success_probability : meeting_probability = 0.25 := sorry

end meeting_success_probability_l117_117631


namespace find_number_l117_117311

theorem find_number (x : ℝ) (h : x / 5 = 70 + x / 6) : x = 2100 := by
  sorry

end find_number_l117_117311


namespace solve_equation_l117_117133

theorem solve_equation (x y : ℝ) : 
    ((16 * x^2 + 1) * (y^2 + 1) = 16 * x * y) ↔ 
        ((x = 1/4 ∧ y = 1) ∨ (x = -1/4 ∧ y = -1)) := 
by
  sorry

end solve_equation_l117_117133


namespace typist_original_salary_l117_117512

theorem typist_original_salary (S : ℝ) (h : (1.12 * 0.93 * 1.15 * 0.90 * S = 5204.21)) : S = 5504.00 :=
sorry

end typist_original_salary_l117_117512


namespace taxi_ride_cost_is_five_dollars_l117_117547

def base_fare : ℝ := 2.00
def cost_per_mile : ℝ := 0.30
def miles_traveled : ℝ := 10.0
def total_cost : ℝ := base_fare + (cost_per_mile * miles_traveled)

theorem taxi_ride_cost_is_five_dollars : total_cost = 5.00 :=
by
  -- proof omitted
  sorry

end taxi_ride_cost_is_five_dollars_l117_117547


namespace max_min_f_values_l117_117976

noncomputable def f (a b c d : ℝ) : ℝ := (Real.sqrt (5 * a + 9) + Real.sqrt (5 * b + 9) + Real.sqrt (5 * c + 9) + Real.sqrt (5 * d + 9))

theorem max_min_f_values (a b c d : ℝ) (h₀ : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) (h₁ : a + b + c + d = 32) :
  (f a b c d ≤ 28) ∧ (f a b c d ≥ 22) := by
  sorry

end max_min_f_values_l117_117976


namespace perfect_square_trinomial_coeff_l117_117672

theorem perfect_square_trinomial_coeff (m : ℝ) : (∃ a b : ℝ, (a ≠ 0) ∧ ((a * x + b)^2 = x^2 - m * x + 25)) ↔ (m = 10 ∨ m = -10) :=
by sorry

end perfect_square_trinomial_coeff_l117_117672


namespace prob_both_hit_prob_at_least_one_hits_l117_117891

variable (pA pB : ℝ)

-- Given conditions
def prob_A_hits : Prop := pA = 0.9
def prob_B_hits : Prop := pB = 0.8

-- Proof problems
theorem prob_both_hit (hA : prob_A_hits pA) (hB : prob_B_hits pB) : 
  pA * pB = 0.72 := 
  sorry

theorem prob_at_least_one_hits (hA : prob_A_hits pA) (hB : prob_B_hits pB) : 
  1 - (1 - pA) * (1 - pB) = 0.98 := 
  sorry

end prob_both_hit_prob_at_least_one_hits_l117_117891


namespace probability_of_usable_gas_pipe_l117_117314

theorem probability_of_usable_gas_pipe (x y : ℝ)
  (hx : 75 ≤ x) 
  (hy : 75 ≤ y)
  (hxy : x + y ≤ 225) :
  (∃ x y, 0 < x ∧ 0 < y ∧ x < 300 ∧ y < 300 ∧ x + y > 75 ∧ (300 - x - y) ≥ 75) → 
  ((150 * 150) / (300 * 300 / 2) = (1 / 4)) :=
by {
  sorry
}

end probability_of_usable_gas_pipe_l117_117314


namespace incorrect_transformation_D_l117_117186

theorem incorrect_transformation_D (x : ℝ) (hx1 : x + 1 ≠ 0) : 
  (2 - x) / (x + 1) ≠ (x - 2) / (1 + x) := 
by 
  sorry

end incorrect_transformation_D_l117_117186


namespace regular_price_per_can_l117_117458

variable (P : ℝ) -- Regular price per can

-- Condition: The regular price per can is discounted 15 percent when the soda is purchased in 24-can cases
def discountedPricePerCan (P : ℝ) : ℝ :=
  0.85 * P

-- Condition: The price of 72 cans purchased in 24-can cases is $18.36
def priceOf72CansInDollars : ℝ :=
  18.36

-- Predicate describing the condition that the price of 72 cans is 18.36
axiom h : (72 * discountedPricePerCan P) = priceOf72CansInDollars

theorem regular_price_per_can (P : ℝ) (h : (72 * discountedPricePerCan P) = priceOf72CansInDollars) : P = 0.30 :=
by
  sorry

end regular_price_per_can_l117_117458


namespace solve_nine_sections_bamboo_problem_l117_117135

-- Define the bamboo stick problem
noncomputable def nine_sections_bamboo_problem : Prop :=
∃ (a : ℕ → ℝ) (d : ℝ),
  (∀ n, a (n + 1) = a n + d) ∧ -- Arithmetic sequence
  (a 1 + a 2 + a 3 + a 4 = 3) ∧ -- Top 4 sections' total volume
  (a 7 + a 8 + a 9 = 4) ∧ -- Bottom 3 sections' total volume
  (a 5 = 67 / 66) -- Volume of the 5th section

theorem solve_nine_sections_bamboo_problem : nine_sections_bamboo_problem :=
sorry

end solve_nine_sections_bamboo_problem_l117_117135


namespace sheila_hourly_earnings_l117_117482

def sheila_hours_per_day (day : String) : Nat :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 8
  else if day = "Tuesday" ∨ day = "Thursday" then 6
  else 0

def sheila_weekly_hours : Nat :=
  sheila_hours_per_day "Monday" +
  sheila_hours_per_day "Tuesday" +
  sheila_hours_per_day "Wednesday" +
  sheila_hours_per_day "Thursday" +
  sheila_hours_per_day "Friday"

def sheila_weekly_earnings : Nat := 468

theorem sheila_hourly_earnings :
  sheila_weekly_earnings / sheila_weekly_hours = 13 :=
by
  sorry

end sheila_hourly_earnings_l117_117482


namespace math_problem_l117_117611

theorem math_problem :
  (- (1 / 8)) ^ 2007 * (- 8) ^ 2008 = -8 :=
by
  sorry

end math_problem_l117_117611


namespace teamAPointDifferenceTeamB_l117_117998

-- Definitions for players' scores and penalties
structure Player where
  name : String
  points : ℕ
  penalties : List ℕ

def TeamA : List Player := [
  { name := "Beth", points := 12, penalties := [1, 2] },
  { name := "Jan", points := 18, penalties := [1, 2, 3] },
  { name := "Mike", points := 5, penalties := [] },
  { name := "Kim", points := 7, penalties := [1, 2] },
  { name := "Chris", points := 6, penalties := [1] }
]

def TeamB : List Player := [
  { name := "Judy", points := 10, penalties := [1, 2] },
  { name := "Angel", points := 9, penalties := [1] },
  { name := "Nick", points := 12, penalties := [] },
  { name := "Steve", points := 8, penalties := [1, 2, 3] },
  { name := "Mary", points := 5, penalties := [1, 2] },
  { name := "Vera", points := 4, penalties := [1] }
]

-- Helper function to calculate total points for a player considering penalties
def Player.totalPoints (p : Player) : ℕ :=
  p.points - p.penalties.sum

-- Helper function to calculate total points for a team
def totalTeamPoints (team : List Player) : ℕ :=
  team.foldr (λ p acc => acc + p.totalPoints) 0

def teamAPoints : ℕ := totalTeamPoints TeamA
def teamBPoints : ℕ := totalTeamPoints TeamB

theorem teamAPointDifferenceTeamB :
  teamAPoints - teamBPoints = 1 :=
  sorry

end teamAPointDifferenceTeamB_l117_117998


namespace population_is_24000_l117_117489

theorem population_is_24000 (P : ℝ) (h : 0.96 * P = 23040) : P = 24000 := sorry

end population_is_24000_l117_117489


namespace find_a_l117_117679

noncomputable theory

def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x0 a : ℝ) (h : f x0 a - g x0 a = 3) : a = -Real.log 2 - 1 :=
sorry

end find_a_l117_117679


namespace present_age_ratio_l117_117127

theorem present_age_ratio (D J : ℕ) (h1 : Dan = 24) (h2 : James = 20) : Dan / James = 6 / 5 := by
  sorry

end present_age_ratio_l117_117127


namespace train_length_is_correct_l117_117623

-- Definitions
def speed_kmh := 48.0 -- in km/hr
def time_sec := 9.0 -- in seconds

-- Conversion function
def convert_speed (s_kmh : Float) : Float :=
  s_kmh * 1000 / 3600

-- Function to calculate length of train
def length_of_train (speed_kmh : Float) (time_sec : Float) : Float :=
  let speed_ms := convert_speed speed_kmh
  speed_ms * time_sec

-- Proof problem: Given the speed of the train and the time it takes to cross a pole, prove the length of the train
theorem train_length_is_correct : length_of_train speed_kmh time_sec = 119.97 :=
by
  sorry

end train_length_is_correct_l117_117623


namespace log_of_y_pow_x_eq_neg4_l117_117487

theorem log_of_y_pow_x_eq_neg4 (x y : ℝ) (h : |x - 8 * y| + (4 * y - 1) ^ 2 = 0) : 
  Real.logb 2 (y ^ x) = -4 :=
sorry

end log_of_y_pow_x_eq_neg4_l117_117487


namespace discount_percentage_l117_117152

theorem discount_percentage (original_price sale_price : ℕ) (h₁ : original_price = 1200) (h₂ : sale_price = 1020) : 
  ((original_price - sale_price) * 100 / original_price : ℝ) = 15 :=
by
  sorry

end discount_percentage_l117_117152


namespace recurring03_m_mult_recurring8_l117_117526

noncomputable def recurring03_fraction : ℚ :=
let x := 0.03.represented_by_recurring 100 in
(100 * x - x / 99)

noncomputable def recurring8_fraction : ℚ :=
let y := 0.8.represented_by_recurring 10 in
(10 * y - y / 9)

theorem recurring03_m_mult_recurring8 : recurring03_fraction * recurring8_fraction = 8 / 297 := 
by
  sorry

end recurring03_m_mult_recurring8_l117_117526


namespace max_elem_one_correct_max_elem_two_correct_min_range_x_correct_average_min_eq_x_correct_l117_117261

def max_elem_one (c : ℝ) : Prop :=
  max (-2) (max 3 c) = max 3 c

def max_elem_two (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : Prop :=
  max (3 * m) (max ((n + 3) * m) (-m * n)) = - m * n

def min_range_x (x : ℝ) : Prop :=
  min 2 (min (2 * x + 2) (4 - 2 * x)) = 2 → 0 ≤ x ∧ x ≤ 1

def average_min_eq_x : Prop :=
  ∀ (x : ℝ), (2 + (x + 1) + 2 * x) / 3 = min 2 (min (x + 1) (2 * x)) → x = 1

-- Lean 4 statements
theorem max_elem_one_correct (c : ℝ) : max_elem_one c := 
  sorry

theorem max_elem_two_correct {m n : ℝ} (h1 : m < 0) (h2 : n > 0) : max_elem_two m n h1 h2 :=
  sorry

theorem min_range_x_correct (h : min 2 (min (2 * x + 2) (4 - 2 * x)) = 2) : min_range_x x :=
  sorry

theorem average_min_eq_x_correct : average_min_eq_x :=
  sorry

end max_elem_one_correct_max_elem_two_correct_min_range_x_correct_average_min_eq_x_correct_l117_117261


namespace length_AB_l117_117626

-- Definitions and conditions
variables (R r a : ℝ) (hR : R > r) (BC_eq_a : BC = a) (r_eq_4 : r = 4)

-- Length of AB
theorem length_AB (AB : ℝ) : AB = a * Real.sqrt (R / (R - 4)) :=
sorry

end length_AB_l117_117626


namespace additional_savings_zero_l117_117051

noncomputable def windows_savings (purchase_price : ℕ) (free_windows : ℕ) (paid_windows : ℕ)
  (dave_needs : ℕ) (doug_needs : ℕ) : ℕ := sorry

theorem additional_savings_zero :
  windows_savings 100 2 5 12 10 = 0 := sorry

end additional_savings_zero_l117_117051


namespace compute_expression_l117_117196

theorem compute_expression : (7^2 - 2 * 5 + 2^3) = 47 :=
by
  sorry

end compute_expression_l117_117196


namespace jack_reads_books_in_a_year_l117_117684

/-- If Jack reads 9 books per day, how many books can he read in a year (365 days)? -/
theorem jack_reads_books_in_a_year (books_per_day : ℕ) (days_per_year : ℕ) (books_per_year : ℕ) (h1 : books_per_day = 9) (h2 : days_per_year = 365) : books_per_year = 3285 :=
by
  sorry

end jack_reads_books_in_a_year_l117_117684


namespace difference_five_three_numbers_specific_number_condition_l117_117336

def is_five_three_number (A : ℕ) : Prop :=
  let a := A / 1000
  let b := (A % 1000) / 100
  let c := (A % 100) / 10
  let d := A % 10
  a = 5 + c ∧ b = 3 + d

def M (A : ℕ) : ℕ :=
  let a := A / 1000
  let b := (A % 1000) / 100
  let c := (A % 100) / 10
  let d := A % 10
  a + c + 2 * (b + d)

def N (A : ℕ) : ℕ :=
  let b := (A % 1000) / 100
  b - 3

noncomputable def largest_five_three_number := 9946
noncomputable def smallest_five_three_number := 5300

theorem difference_five_three_numbers :
  largest_five_three_number - smallest_five_three_number = 4646 := by
  sorry

noncomputable def specific_five_three_number := 5401

theorem specific_number_condition {A : ℕ} (hA : is_five_three_number A) :
  (M A) % (N A) = 0 ∧ (M A) / (N A) % 5 = 0 → A = specific_five_three_number := by
  sorry

end difference_five_three_numbers_specific_number_condition_l117_117336


namespace initial_oranges_is_sum_l117_117464

-- Define the number of oranges taken by Jonathan
def oranges_taken : ℕ := 45

-- Define the number of oranges left in the box
def oranges_left : ℕ := 51

-- The theorem states that the initial number of oranges is the sum of the oranges taken and those left
theorem initial_oranges_is_sum : oranges_taken + oranges_left = 96 := 
by 
  -- This is where the proof would go
  sorry

end initial_oranges_is_sum_l117_117464


namespace no_solutions_exist_l117_117333

theorem no_solutions_exist : ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 2 :=
by sorry

end no_solutions_exist_l117_117333


namespace coat_price_reduction_l117_117625

theorem coat_price_reduction 
    (original_price : ℝ) 
    (reduction_amount : ℝ) 
    (h1 : original_price = 500) 
    (h2 : reduction_amount = 300) : 
    (reduction_amount / original_price) * 100 = 60 := 
by 
  sorry

end coat_price_reduction_l117_117625


namespace double_inputs_revenue_l117_117566

theorem double_inputs_revenue (A K L : ℝ) (α1 α2 : ℝ) (hα1 : α1 = 0.6) (hα2 : α2 = 0.5) (hα1_bound : 0 < α1 ∧ α1 < 1) (hα2_bound : 0 < α2 ∧ α2 < 1) :
  A * (2 * K) ^ α1 * (2 * L) ^ α2 > 2 * (A * K ^ α1 * L ^ α2) :=
by
  sorry

end double_inputs_revenue_l117_117566


namespace wand_cost_l117_117926

theorem wand_cost (c : ℕ) (h1 : 3 * c = 3 * c) (h2 : 2 * (c + 5) = 130) : c = 60 :=
by
  sorry

end wand_cost_l117_117926


namespace sum_of_marked_angles_l117_117158

theorem sum_of_marked_angles (sum_of_angles_around_vertex : ℕ := 360) 
    (vertices : ℕ := 7) (triangles : ℕ := 3) 
    (sum_of_interior_angles_triangle : ℕ := 180) :
    (vertices * sum_of_angles_around_vertex - triangles * sum_of_interior_angles_triangle) = 1980 :=
by
  sorry

end sum_of_marked_angles_l117_117158


namespace correct_option_is_B_l117_117775

-- Define the conditions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (m : ℝ) : Prop := (-2 * m^2)^3 = -8 * m^6
def optionC (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2
def optionD (a b : ℝ) : Prop := 2 * a * b + 3 * a^2 * b = 5 * a^3 * b^2

-- The proof problem: which option is correct
theorem correct_option_is_B (m : ℝ) : optionB m := by
  sorry

end correct_option_is_B_l117_117775


namespace find_angle_and_area_of_triangle_l117_117704

theorem find_angle_and_area_of_triangle (a b : ℝ) 
  (h_a : a = Real.sqrt 7) (h_b : b = 2)
  (angle_A : ℝ) (angle_A_eq : angle_A = Real.pi / 3)
  (angle_B : ℝ)
  (vec_m : ℝ × ℝ := (a, Real.sqrt 3 * b))
  (vec_n : ℝ × ℝ := (Real.cos angle_A, Real.sin angle_B))
  (colinear : vec_m.1 * vec_n.2 = vec_m.2 * vec_n.1)
  (sin_A : Real.sin angle_A = (Real.sqrt 3) / 2)
  (cos_A : Real.cos angle_A = 1 / 2) :
  angle_A = Real.pi / 3 ∧ 
  ∃ (c : ℝ), c = 3 ∧
  (1/2) * b * c * Real.sin angle_A = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end find_angle_and_area_of_triangle_l117_117704


namespace ratio_of_liquid_rise_l117_117769

theorem ratio_of_liquid_rise
  (h1 h2 : ℝ) (r1 r2 rm : ℝ)
  (V1 V2 Vm : ℝ)
  (H1 : r1 = 4)
  (H2 : r2 = 9)
  (H3 : V1 = (1 / 3) * π * r1^2 * h1)
  (H4 : V2 = (1 / 3) * π * r2^2 * h2)
  (H5 : V1 = V2)
  (H6 : rm = 2)
  (H7 : Vm = (4 / 3) * π * rm^3)
  (H8 : h2 = h1 * (81 / 16))
  (h1' h2' : ℝ)
  (H9 : h1' = h1 + Vm / ((1 / 3) * π * r1^2))
  (H10 : h2' = h2 + Vm / ((1 / 3) * π * r2^2)) :
  (h1' - h1) / (h2' - h2) = 81 / 16 :=
sorry

end ratio_of_liquid_rise_l117_117769


namespace least_cost_l117_117455

noncomputable def total_length : ℝ := 10 * (6 / 12)
def cost_6_foot_rope : ℝ := 5
def cost_per_foot_6_foot_rope : ℝ := cost_6_foot_rope / 6
def cost_1_foot_rope : ℝ := 1.25
def total_cost_1_foot : ℝ := 5 * cost_1_foot_rope

theorem least_cost : min cost_6_foot_rope total_cost_1_foot = 5 := 
by sorry

end least_cost_l117_117455


namespace part1_sequences_valid_part2_no_infinite_sequence_l117_117340

def valid_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, n > 0 → (∑ i in range (n + 1), a i)^2 = ∑ i in range (n + 1), (a i)^3

theorem part1_sequences_valid (a : ℕ → ℝ) (h : valid_sequence a) :
  a 0 = 1 ∧ ((a 1 = 2 ∧ (a 2 = 3 ∨ a 2 = -2)) ∨ (a 1 = -1 ∧ a 2 = 1)) :=
sorry

theorem part2_no_infinite_sequence (a : ℕ → ℝ) (h : valid_sequence a) (h2 : a 2012 = -2012) :
  false :=
sorry

end part1_sequences_valid_part2_no_infinite_sequence_l117_117340


namespace increasing_function_range_a_l117_117221

theorem increasing_function_range_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = if x > 1 then a^x else (4 - a/2)*x + 2) ∧
  (∀ x y, x < y → f x ≤ f y) →
  4 ≤ a ∧ a < 8 :=
by
  sorry

end increasing_function_range_a_l117_117221


namespace infinite_six_consecutive_epsilon_squarish_l117_117872

def is_epsilon_squarish (ε : ℝ) (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 1 < a ∧ a < b ∧ b < (1 + ε) * a ∧ n = a * b

theorem infinite_six_consecutive_epsilon_squarish (ε : ℝ) (hε : 0 < ε) : 
  ∃ (N : ℕ), ∃ (n : ℕ), N ≤ n ∧
  (is_epsilon_squarish ε n) ∧ 
  (is_epsilon_squarish ε (n + 1)) ∧ 
  (is_epsilon_squarish ε (n + 2)) ∧ 
  (is_epsilon_squarish ε (n + 3)) ∧ 
  (is_epsilon_squarish ε (n + 4)) ∧ 
  (is_epsilon_squarish ε (n + 5)) :=
  sorry

end infinite_six_consecutive_epsilon_squarish_l117_117872


namespace coordinates_of_P_l117_117983

theorem coordinates_of_P (m : ℝ) (P : ℝ × ℝ) :
  P = (2 * m, m + 8) ∧ 2 * m = 0 → P = (0, 8) := by
  intros hm
  sorry

end coordinates_of_P_l117_117983


namespace total_students_eq_seventeen_l117_117999

theorem total_students_eq_seventeen 
    (N : ℕ)
    (initial_students : N - 1 = 16)
    (avg_first_day : 77 * (N - 1) = 77 * 16)
    (avg_second_day : 78 * N = 78 * N)
    : N = 17 :=
sorry

end total_students_eq_seventeen_l117_117999


namespace find_a_l117_117078

variable (a : ℝ) (h_pos : a > 0) (h_integral : ∫ x in 0..a, (2 * x - 2) = 3)

theorem find_a : a = 3 :=
by sorry

end find_a_l117_117078


namespace Danny_shorts_washed_l117_117193

-- Define the given conditions
def Cally_white_shirts : ℕ := 10
def Cally_colored_shirts : ℕ := 5
def Cally_shorts : ℕ := 7
def Cally_pants : ℕ := 6

def Danny_white_shirts : ℕ := 6
def Danny_colored_shirts : ℕ := 8
def Danny_pants : ℕ := 6

def total_clothes_washed : ℕ := 58

-- Calculate total clothes washed by Cally
def total_cally_clothes : ℕ := 
  Cally_white_shirts + Cally_colored_shirts + Cally_shorts + Cally_pants

-- Calculate total clothes washed by Danny (excluding shorts)
def total_danny_clothes_excl_shorts : ℕ := 
  Danny_white_shirts + Danny_colored_shirts + Danny_pants

-- Define the statement to be proven
theorem Danny_shorts_washed : 
  total_clothes_washed - (total_cally_clothes + total_danny_clothes_excl_shorts) = 10 := by
  sorry

end Danny_shorts_washed_l117_117193


namespace doberman_puppies_count_l117_117207

theorem doberman_puppies_count (D : ℝ) (S : ℝ) (h1 : S = 55) (h2 : 3 * D - 5 + (D - S) = 90) : D = 37.5 :=
by
  sorry

end doberman_puppies_count_l117_117207


namespace find_x_squared_plus_y_squared_l117_117230

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 16) : x^2 + y^2 = 432 :=
sorry

end find_x_squared_plus_y_squared_l117_117230


namespace pq_identity_l117_117442

theorem pq_identity (p q : ℝ) (h1 : p * q = 20) (h2 : p + q = 10) : p^2 + q^2 = 60 :=
sorry

end pq_identity_l117_117442


namespace draw_contains_chinese_book_l117_117638

theorem draw_contains_chinese_book
  (total_books : ℕ)
  (chinese_books : ℕ)
  (math_books : ℕ)
  (drawn_books : ℕ)
  (h_total : total_books = 12)
  (h_chinese : chinese_books = 10)
  (h_math : math_books = 2)
  (h_drawn : drawn_books = 3) :
  ∃ n, n ≥ 1 ∧ n ≤ drawn_books ∧ n * (chinese_books/total_books) > 1 := 
  sorry

end draw_contains_chinese_book_l117_117638


namespace area_of_inscribed_triangle_l117_117511

noncomputable def triangle_area_inscribed (arc1 arc2 arc3 : ℝ) (r : ℝ) : ℝ :=
  let θ1 := (arc1 / (2 * π * r)) * (2 * π) in
  let θ2 := (arc2 / (2 * π * r)) * (2 * π) in
  let θ3 := (arc3 / (2 * π * r)) * (2 * π) in
  1 / 2 * r^2 * (Real.sin θ1 + Real.sin θ2 + Real.sin θ3)

theorem area_of_inscribed_triangle {arc1 arc2 arc3 : ℝ}
  (h1 : arc1 = 5) (h2 : arc2 = 6) (h3 : arc3 = 7) :
  triangle_area_inscribed arc1 arc2 arc3 (9 / π) = 101.2488 / π^2 := by
  sorry

end area_of_inscribed_triangle_l117_117511


namespace simplify_expression_l117_117268

theorem simplify_expression (x : ℝ) : 5 * x + 2 * x + 7 * x = 14 * x :=
by
  sorry

end simplify_expression_l117_117268


namespace susan_added_oranges_l117_117612

-- Conditions as definitions
def initial_oranges_in_box : ℝ := 55.0
def final_oranges_in_box : ℝ := 90.0

-- Define the quantity of oranges Susan put into the box
def susan_oranges := final_oranges_in_box - initial_oranges_in_box

-- Theorem statement to prove that the number of oranges Susan put into the box is 35.0
theorem susan_added_oranges : susan_oranges = 35.0 := by
  unfold susan_oranges
  sorry

end susan_added_oranges_l117_117612


namespace max_single_player_salary_l117_117937

theorem max_single_player_salary
    (num_players : ℕ) (min_salary : ℕ) (total_salary_cap : ℕ)
    (num_player_min_salary : ℕ) (max_salary : ℕ)
    (h1 : num_players = 18)
    (h2 : min_salary = 20000)
    (h3 : total_salary_cap = 600000)
    (h4 : num_player_min_salary = 17)
    (h5 : num_players = num_player_min_salary + 1)
    (h6 : total_salary_cap = num_player_min_salary * min_salary + max_salary) :
    max_salary = 260000 :=
by
  sorry

end max_single_player_salary_l117_117937


namespace bridget_apples_l117_117641

/-!
# Problem statement
Bridget bought a bag of apples. She gave half of the apples to Ann. She gave 5 apples to Cassie,
and 2 apples to Dan. She kept 6 apples for herself. Prove that Bridget originally bought 26 apples.
-/

theorem bridget_apples (x : ℕ) 
  (H1 : x / 2 + 2 * (x % 2) / 2 - 5 - 2 = 6) : x = 26 :=
sorry

end bridget_apples_l117_117641


namespace triangle_area_is_correct_l117_117527

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_is_correct :
  area_of_triangle (0, 3) (4, -2) (9, 6) = 16.5 :=
by
  sorry

end triangle_area_is_correct_l117_117527


namespace max_value_seq_l117_117979

theorem max_value_seq : 
  ∃ a : ℕ → ℝ, 
    a 1 = 1 ∧ 
    a 2 = 4 ∧ 
    (∀ n ≥ 2, 2 * a n = (n - 1) / n * a (n - 1) + (n + 1) / n * a (n + 1)) ∧ 
    ∀ n : ℕ, n > 0 → 
      ∃ m : ℕ, m > 0 ∧ 
        ∀ k : ℕ, k > 0 → (a k) / k ≤ 2 ∧ (a 2) / 2 = 2 :=
sorry

end max_value_seq_l117_117979


namespace point_on_x_axis_coordinates_l117_117099

theorem point_on_x_axis_coordinates (a : ℝ) (P : ℝ × ℝ) (h : P = (a - 1, a + 2)) (hx : P.2 = 0) : P = (-3, 0) :=
by
  -- Replace this with the full proof
  sorry

end point_on_x_axis_coordinates_l117_117099


namespace line_points_product_l117_117034

theorem line_points_product (x y : ℝ) (h1 : 8 = (1/4 : ℝ) * x) (h2 : y = (1/4 : ℝ) * 20) : x * y = 160 := 
by
  sorry

end line_points_product_l117_117034


namespace exists_positive_M_l117_117421

open Set

noncomputable def f (x : ℝ) : ℝ := sorry

theorem exists_positive_M 
  (h₁ : ∀ x ∈ Ioo (0 : ℝ) 1, f x > 0)
  (h₂ : ∀ x ∈ Ioo (0 : ℝ) 1, f (2 * x / (1 + x^2)) = 2 * f x) :
  ∃ M > 0, ∀ x ∈ Ioo (0 : ℝ) 1, f x ≤ M :=
sorry

end exists_positive_M_l117_117421


namespace age_in_1900_l117_117426

theorem age_in_1900 
  (x y : ℕ)
  (H1 : y = 29 * x)
  (H2 : 1901 ≤ y + x ∧ y + x ≤ 1930) :
  1900 - y = 44 := 
sorry

end age_in_1900_l117_117426


namespace shaded_area_of_square_with_circles_l117_117498

theorem shaded_area_of_square_with_circles :
  let side_length_square := 12
  let radius_quarter_circle := 6
  let radius_center_circle := 3
  let area_square := side_length_square * side_length_square
  let area_quarter_circles := 4 * (1 / 4) * Real.pi * (radius_quarter_circle ^ 2)
  let area_center_circle := Real.pi * (radius_center_circle ^ 2)
  area_square - area_quarter_circles - area_center_circle = 144 - 45 * Real.pi :=
by
  sorry

end shaded_area_of_square_with_circles_l117_117498


namespace rhombus_triangle_area_correct_l117_117617

noncomputable def rhombus_triangle_area (d1 d2 : ℝ) (x : ℝ) : ℝ :=
  (1/2) * (d1/2) * (d2/2) * sin x

theorem rhombus_triangle_area_correct (x : ℝ) : rhombus_triangle_area 15 20 x = 37.5 * sin x :=
by
  unfold rhombus_triangle_area
  ring
  norm_num
  sorry

end rhombus_triangle_area_correct_l117_117617


namespace percent_counties_l117_117275

def p1 : ℕ := 21
def p2 : ℕ := 44
def p3 : ℕ := 18

theorem percent_counties (h1 : p1 = 21) (h2 : p2 = 44) (h3 : p3 = 18) : p1 + p2 + p3 = 83 :=
by sorry

end percent_counties_l117_117275


namespace max_min_values_l117_117217

noncomputable def max_value (x y z w : ℝ) : ℝ :=
  if x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2 then
    max (x + y + z + w) 3
  else
    0

noncomputable def min_value (x y z w : ℝ) : ℝ :=
  if x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2 then
    min (x + y + z + w) (-2 + 5 / 2 * Real.sqrt 2)
  else
    0

theorem max_min_values (x y z w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) (h_nonneg_w : 0 ≤ w)
  (h_eqn : x^2 + y^2 + z^2 + w^2 + x + 2 * y + 3 * z + 4 * w = 17 / 2) :
  (x + y + z + w ≤ 3) ∧ (x + y + z + w ≥ -2 + 5 / 2 * Real.sqrt 2) :=
by
  sorry

end max_min_values_l117_117217


namespace baseball_attendance_difference_l117_117643

theorem baseball_attendance_difference:
  ∃ C D: ℝ, 
    (59500 ≤ C ∧ C ≤ 80500 ∧ 69565 ≤ D ∧ D ≤ 94118) ∧ 
    (max (D - C) (C - D) = 35000 ∧ min (D - C) (C - D) = 11000) := by
  sorry

end baseball_attendance_difference_l117_117643


namespace JohnReceivedDiamonds_l117_117912

def InitialDiamonds (Bill Sam : ℕ) (John : ℕ) : Prop :=
  Bill = 12 ∧ Sam = 12

def TheftEvents (BillBefore SamBefore JohnBefore BillAfter SamAfter JohnAfter : ℕ) : Prop :=
  BillAfter = BillBefore - 1 ∧ SamAfter = SamBefore - 1 ∧ JohnAfter = JohnBefore + 1

def AverageMassChange (Bill Sam John : ℕ) (BillMassChange SamMassChange JohnMassChange : ℤ) : Prop :=
  BillMassChange = Bill - 1 ∧ SamMassChange = Sam - 2 ∧ JohnMassChange = John + 4

def JohnInitialDiamonds (John : ℕ) : Prop :=
  Exists (fun x => 4 * x = 36)

theorem JohnReceivedDiamonds : ∃ John : ℕ, 
  InitialDiamonds 12 12 John ∧
  (∃ BillBefore SamBefore JohnBefore BillAfter SamAfter JohnAfter,
      TheftEvents BillBefore SamBefore JohnBefore BillAfter SamAfter JohnAfter ∧
      AverageMassChange 12 12 12 (-12) (-24) 36) →
  John = 9 :=
sorry

end JohnReceivedDiamonds_l117_117912


namespace student_total_marks_l117_117313

variable (M P C : ℕ)

theorem student_total_marks :
  C = P + 20 ∧ (M + C) / 2 = 25 → M + P = 30 :=
by
  sorry

end student_total_marks_l117_117313


namespace mary_sailboat_canvas_l117_117588

def rectangular_sail_area (length width : ℕ) : ℕ :=
  length * width

def triangular_sail_area (base height : ℕ) : ℕ :=
  (base * height) / 2

def total_canvas_area (length₁ width₁ base₁ height₁ base₂ height₂ : ℕ) : ℕ :=
  rectangular_sail_area length₁ width₁ +
  triangular_sail_area base₁ height₁ +
  triangular_sail_area base₂ height₂

theorem mary_sailboat_canvas :
  total_canvas_area 5 8 3 4 4 6 = 58 :=
by
  -- Begin proof (proof steps omitted, we just need the structure here)
  sorry -- end proof

end mary_sailboat_canvas_l117_117588


namespace wheels_in_garage_l117_117468

-- Definitions of the entities within the problem
def cars : Nat := 2
def car_wheels : Nat := 4

def riding_lawnmower : Nat := 1
def lawnmower_wheels : Nat := 4

def bicycles : Nat := 3
def bicycle_wheels : Nat := 2

def tricycle : Nat := 1
def tricycle_wheels : Nat := 3

def unicycle : Nat := 1
def unicycle_wheels : Nat := 1

-- The total number of wheels in the garage
def total_wheels :=
  (cars * car_wheels) +
  (riding_lawnmower * lawnmower_wheels) +
  (bicycles * bicycle_wheels) +
  (tricycle * tricycle_wheels) +
  (unicycle * unicycle_wheels)

-- The theorem we wish to prove
theorem wheels_in_garage : total_wheels = 22 := by
  sorry

end wheels_in_garage_l117_117468


namespace percentage_scientists_born_in_june_l117_117606

theorem percentage_scientists_born_in_june :
  (18 / 200 * 100) = 9 :=
by sorry

end percentage_scientists_born_in_june_l117_117606


namespace no_solutions_xyz_l117_117331

theorem no_solutions_xyz : ∀ (x y z : ℝ), x + y = 3 → xy - z^2 = 2 → false := by
  intros x y z h1 h2
  sorry

end no_solutions_xyz_l117_117331


namespace marla_adds_blue_paint_l117_117446

variable (M B : ℝ)

theorem marla_adds_blue_paint :
  (20 = 0.10 * M) ∧ (B = 0.70 * M) → B = 140 := 
by 
  sorry

end marla_adds_blue_paint_l117_117446


namespace quadratic_vertex_l117_117005

theorem quadratic_vertex (x : ℝ) :
  ∃ (h k : ℝ), (h = -3) ∧ (k = -5) ∧ (∀ y, y = -2 * (x + h) ^ 2 + k) :=
sorry

end quadratic_vertex_l117_117005


namespace smallest_prime_number_conditions_l117_117157

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |> List.sum -- Summing the digits in base 10

def is_prime (n : ℕ) : Prop := Nat.Prime n

def smallest_prime_number (n : ℕ) : Prop :=
  is_prime n ∧ sum_of_digits n = 17 ∧ n > 200 ∧
  (∀ m : ℕ, is_prime m ∧ sum_of_digits m = 17 ∧ m > 200 → n ≤ m)

theorem smallest_prime_number_conditions (p : ℕ) : 
  smallest_prime_number p ↔ p = 197 :=
by
  sorry

end smallest_prime_number_conditions_l117_117157


namespace problem_solution_l117_117231

theorem problem_solution (x y : ℕ) (hxy : x + y + x * y = 104) (hx : 0 < x) (hy : 0 < y) (hx30 : x < 30) (hy30 : y < 30) : 
  x + y = 20 := 
sorry

end problem_solution_l117_117231


namespace vectors_parallel_y_eq_minus_one_l117_117682

theorem vectors_parallel_y_eq_minus_one (y : ℝ) :
  let a := (1, 2)
  let b := (1, -2 * y)
  b.1 * a.2 - a.1 * b.2 = 0 → y = -1 :=
by
  intros a b h
  simp at h
  sorry

end vectors_parallel_y_eq_minus_one_l117_117682


namespace patty_fraction_3mph_l117_117724

noncomputable def fraction_time_at_3mph (t3 t6 : ℝ) (h : 3 * t3 + 6 * t6 = 5 * (t3 + t6)) : ℝ :=
  t3 / (t3 + t6)

theorem patty_fraction_3mph (t3 t6 : ℝ) (h : 3 * t3 + 6 * t6 = 5 * (t3 + t6)) :
  fraction_time_at_3mph t3 t6 h = 1 / 3 :=
by
  sorry

end patty_fraction_3mph_l117_117724


namespace equal_money_distribution_l117_117901

theorem equal_money_distribution (y : ℝ) : 
  ∃ z : ℝ, z = 0.1 * (1.25 * y) ∧ (1.25 * y) - z = y + z - y :=
by
  sorry

end equal_money_distribution_l117_117901


namespace distance_difference_l117_117917

-- Given conditions
def speed_train1 : ℕ := 20
def speed_train2 : ℕ := 25
def total_distance : ℕ := 675

-- Define the problem statement
theorem distance_difference : ∃ t : ℝ, (speed_train2 * t - speed_train1 * t) = 75 ∧ (speed_train1 * t + speed_train2 * t) = total_distance := by 
  sorry

end distance_difference_l117_117917


namespace equation_is_correct_l117_117591

-- Define the numbers
def n1 : ℕ := 2
def n2 : ℕ := 2
def n3 : ℕ := 11
def n4 : ℕ := 11

-- Define the mathematical expression and the target result
def expression : ℚ := (n1 + n2 / n3) * n4
def target_result : ℚ := 24

-- The proof statement
theorem equation_is_correct : expression = target_result := by
  sorry

end equation_is_correct_l117_117591


namespace koi_fish_after_three_weeks_l117_117263

theorem koi_fish_after_three_weeks
  (f_0 : ℕ := 280) -- initial total number of fish
  (days : ℕ := 21) -- days in 3 weeks
  (koi_added_per_day : ℕ := 2)
  (goldfish_added_per_day : ℕ := 5)
  (goldfish_after_3_weeks : ℕ := 200) :
  let total_fish_added := days * (koi_added_per_day + goldfish_added_per_day)
  let total_fish_after := f_0 + total_fish_added
  let koi_after_3_weeks := total_fish_after - goldfish_after_3_weeks
  koi_after_3_weeks = 227 :=
by
  let total_fish_added := days * (koi_added_per_day + goldfish_added_per_day)
  let total_fish_after := f_0 + total_fish_added
  let koi_after_3_weeks := total_fish_after - goldfish_after_3_weeks
  sorry

end koi_fish_after_three_weeks_l117_117263


namespace find_x_such_that_fraction_eq_l117_117206

theorem find_x_such_that_fraction_eq 
  (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ 5) : 
  (x^2 - 4 * x + 3) / (x^2 - 6 * x + 5) = (x^2 - 3 * x - 10) / (x^2 - 2 * x - 15) ↔ 
  x = -19 / 3 :=
sorry

end find_x_such_that_fraction_eq_l117_117206


namespace tim_balloon_count_l117_117320

theorem tim_balloon_count (Dan_balloons : ℕ) (h1 : Dan_balloons = 59) (Tim_balloons : ℕ) (h2 : Tim_balloons = 11 * Dan_balloons) : Tim_balloons = 649 :=
sorry

end tim_balloon_count_l117_117320


namespace pqr_problem_l117_117874

noncomputable def pqr_abs (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p ≠ q) (h5 : q ≠ r) (h6 : r ≠ p)
  (h7 : p + (2 / q) = q + (2 / r)) 
  (h8 : q + (2 / r) = r + (2 / p)) : ℝ :=
|p * q * r|

theorem pqr_problem (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p ≠ q) (h5 : q ≠ r) (h6 : r ≠ p)
  (h7 : p + (2 / q) = q + (2 / r)) 
  (h8 : q + (2 / r) = r + (2 / p)) : pqr_abs p q r h1 h2 h3 h4 h5 h6 h7 h8 = 2 := 
sorry

end pqr_problem_l117_117874


namespace no_half_probability_socks_l117_117761

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l117_117761


namespace find_quadratic_function_l117_117075

open Function

-- Define the quadratic function g(x) with parameters c and d
def g (c d : ℝ) (x : ℝ) : ℝ := x^2 + c * x + d

-- State the main theorem
theorem find_quadratic_function :
  ∃ (c d : ℝ), (∀ x : ℝ, (g c d (g c d x + x)) / (g c d x) = x^2 + 120 * x + 360) ∧ c = 119 ∧ d = 240 :=
by
  sorry

end find_quadratic_function_l117_117075


namespace probability_is_7_over_26_l117_117447

section VowelProbability

def num_students : Nat := 26

def is_vowel (c : Char) : Bool :=
  c = 'A' || c = 'E' || c = 'I' || c = 'O' || c = 'U' || c = 'Y' || c = 'W'

def num_vowels : Nat := 7

def probability_of_vowel_initials : Rat :=
  (num_vowels : Nat) / (num_students : Nat)

theorem probability_is_7_over_26 :
  probability_of_vowel_initials = 7 / 26 := by
  sorry

end VowelProbability

end probability_is_7_over_26_l117_117447


namespace corner_cells_different_colors_l117_117066

theorem corner_cells_different_colors 
  (colors : Fin 4 → Prop)
  (painted : (Fin 100 × Fin 100) → Fin 4)
  (h : ∀ (i j : Fin 99), 
    ∃ f g h k, 
      f ≠ g ∧ f ≠ h ∧ f ≠ k ∧
      g ≠ h ∧ g ≠ k ∧ 
      h ≠ k ∧ 
      painted (i, j) = f ∧ 
      painted (i.succ, j) = g ∧ 
      painted (i, j.succ) = h ∧ 
      painted (i.succ, j.succ) = k) :
  painted (0, 0) ≠ painted (99, 0) ∧
  painted (0, 0) ≠ painted (0, 99) ∧
  painted (0, 0) ≠ painted (99, 99) ∧
  painted (99, 0) ≠ painted (0, 99) ∧
  painted (99, 0) ≠ painted (99, 99) ∧
  painted (0, 99) ≠ painted (99, 99) :=
  sorry

end corner_cells_different_colors_l117_117066


namespace midpoint_product_l117_117919

-- Defining the endpoints of the line segment
def x1 : ℤ := 4
def y1 : ℤ := 7
def x2 : ℤ := -8
def y2 : ℤ := 9

-- Proof goal: show that the product of the coordinates of the midpoint is -16
theorem midpoint_product : ((x1 + x2) / 2) * ((y1 + y2) / 2) = -16 := 
by sorry

end midpoint_product_l117_117919


namespace three_digit_integer_one_more_than_multiple_l117_117025

theorem three_digit_integer_one_more_than_multiple :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n = 841 ∧ ∃ k : ℕ, n = 840 * k + 1 :=
by
  sorry

end three_digit_integer_one_more_than_multiple_l117_117025


namespace max_sum_is_2017_l117_117830

theorem max_sum_is_2017 (a b c : ℕ) 
  (h1 : a + b = 1014) 
  (h2 : c - b = 497) 
  (h3 : a > b) : 
  (a + b + c) ≤ 2017 := sorry

end max_sum_is_2017_l117_117830


namespace num_koi_fish_after_3_weeks_l117_117265

noncomputable def total_days : ℕ := 3 * 7

noncomputable def koi_fish_added_per_day : ℕ := 2
noncomputable def goldfish_added_per_day : ℕ := 5

noncomputable def total_fish_initial : ℕ := 280
noncomputable def goldfish_final : ℕ := 200

noncomputable def koi_fish_total_after_3_weeks : ℕ :=
  let koi_fish_added := koi_fish_added_per_day * total_days
  let goldfish_added := goldfish_added_per_day * total_days
  let goldfish_initial := goldfish_final - goldfish_added
  let koi_fish_initial := total_fish_initial - goldfish_initial
  koi_fish_initial + koi_fish_added

theorem num_koi_fish_after_3_weeks :
  koi_fish_total_after_3_weeks = 227 := by
  sorry

end num_koi_fish_after_3_weeks_l117_117265


namespace number_of_family_members_l117_117629

noncomputable def total_money : ℝ :=
  123 * 0.01 + 85 * 0.05 + 35 * 0.10 + 26 * 0.25

noncomputable def leftover_money : ℝ := 0.48

noncomputable def double_scoop_cost : ℝ := 3.0

noncomputable def amount_spent : ℝ := total_money - leftover_money

noncomputable def number_of_double_scoops : ℝ := amount_spent / double_scoop_cost

theorem number_of_family_members :
  number_of_double_scoops = 5 := by
  sorry

end number_of_family_members_l117_117629


namespace probability_not_rel_prime_50_l117_117156

theorem probability_not_rel_prime_50 : 
  let n := 50;
  let non_rel_primes_count := n - Nat.totient 50;
  let total_count := n;
  let probability := (non_rel_primes_count : ℚ) / (total_count : ℚ);
  probability = 3 / 5 :=
by
  sorry

end probability_not_rel_prime_50_l117_117156


namespace structure_of_S_l117_117112

def set_S (x y : ℝ) : Prop :=
  (5 >= x + 1 ∧ 5 >= y - 5) ∨
  (x + 1 >= 5 ∧ x + 1 >= y - 5) ∨
  (y - 5 >= 5 ∧ y - 5 >= x + 1)

theorem structure_of_S :
  ∃ (a b c : ℝ), set_S x y ↔ (y <= x + 6) ∧ (x <= 4) ∧ (y <= 10) 
:= sorry

end structure_of_S_l117_117112


namespace fibonacci_150_mod_7_l117_117002

def fibonacci_mod_7 : Nat → Nat
| 0 => 0
| 1 => 1
| n + 2 => (fibonacci_mod_7 (n + 1) + fibonacci_mod_7 n) % 7

theorem fibonacci_150_mod_7 : fibonacci_mod_7 150 = 1 := 
by sorry

end fibonacci_150_mod_7_l117_117002


namespace value_of_half_plus_five_l117_117988

theorem value_of_half_plus_five (n : ℕ) (h₁ : n = 20) : (n / 2) + 5 = 15 := 
by {
  sorry
}

end value_of_half_plus_five_l117_117988


namespace digit_58_in_decimal_of_one_seventeen_l117_117289

theorem digit_58_in_decimal_of_one_seventeen :
  let repeating_sequence := "0588235294117647"
  let cycle_length := 16
  ∃ (n : ℕ), n = 58 ∧ (n - 1) % cycle_length + 1 = 10 →
  repeating_sequence.get ((n - 1) % cycle_length + 1 - 1) = '2' :=
by
  sorry

end digit_58_in_decimal_of_one_seventeen_l117_117289


namespace repaved_before_today_l117_117494

variable (total_repaved today_repaved : ℕ)

theorem repaved_before_today (h1 : total_repaved = 4938) (h2 : today_repaved = 805) :
  total_repaved - today_repaved = 4133 :=
by 
  -- variables are integers and we are performing a subtraction
  sorry

end repaved_before_today_l117_117494


namespace sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117392

theorem sin_minus_cos_eq_neg_sqrt_10_over_5 (θ : ℝ) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - ((Real.sqrt 10) / 5) :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_10_over_5_l117_117392


namespace quadratic_distinct_real_roots_l117_117560

theorem quadratic_distinct_real_roots (m : ℝ) :
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁ ^ 2 - 2 * x₁ + m = 0 ∧ x₂ ^ 2 - 2 * x₂ + m = 0) ↔ m < 1 :=
by sorry

end quadratic_distinct_real_roots_l117_117560


namespace molecular_weight_is_62_024_l117_117771

def atomic_weight_H : ℝ := 1.008
def atomic_weight_C : ℝ := 12.011
def atomic_weight_O : ℝ := 15.999

def num_atoms_H : ℕ := 2
def num_atoms_C : ℕ := 1
def num_atoms_O : ℕ := 3

def molecular_weight_compound : ℝ :=
  num_atoms_H * atomic_weight_H + num_atoms_C * atomic_weight_C + num_atoms_O * atomic_weight_O

theorem molecular_weight_is_62_024 :
  molecular_weight_compound = 62.024 :=
by
  have h_H := num_atoms_H * atomic_weight_H
  have h_C := num_atoms_C * atomic_weight_C
  have h_O := num_atoms_O * atomic_weight_O
  have h_sum := h_H + h_C + h_O
  show molecular_weight_compound = 62.024
  sorry

end molecular_weight_is_62_024_l117_117771


namespace intersection_complement_P_Q_l117_117546

def P (x : ℝ) : Prop := x - 1 ≤ 0
def Q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

def complement_P (x : ℝ) : Prop := ¬ P x

theorem intersection_complement_P_Q :
  {x : ℝ | complement_P x} ∩ {x : ℝ | Q x} = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end intersection_complement_P_Q_l117_117546


namespace solve_sqrt_equation_l117_117654

theorem solve_sqrt_equation (x : ℝ) :
  (∃ x, sqrt ((3 + 2 * sqrt 2)^x) + sqrt ((3 - 2 * sqrt 2)^x) = 6) ↔ (x = 2 ∨ x = -2) :=
by
  sorry

end solve_sqrt_equation_l117_117654


namespace factorial_division_l117_117962

-- Definition of factorial in Lean
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Statement of the problem:
theorem factorial_division : fact 12 / fact 11 = 12 :=
by
  sorry

end factorial_division_l117_117962


namespace right_isosceles_areas_no_relations_l117_117594

theorem right_isosceles_areas_no_relations :
  let W := 1 / 2 * 5 * 5
  let X := 1 / 2 * 12 * 12
  let Y := 1 / 2 * 13 * 13
  ¬ (X + Y = 2 * W + X ∨ W + X = Y ∨ 2 * X = W + Y ∨ X + W = W ∨ W + Y = 2 * X) :=
by
  sorry

end right_isosceles_areas_no_relations_l117_117594


namespace average_seq_13_to_52_l117_117155

-- Define the sequence of natural numbers from 13 to 52
def seq : List ℕ := List.range' 13 52

-- Define the average of a list of natural numbers
def average (xs : List ℕ) : ℚ := (xs.sum : ℚ) / xs.length

-- Define the specific set of numbers and their average
theorem average_seq_13_to_52 : average seq = 32.5 := 
by 
  sorry

end average_seq_13_to_52_l117_117155


namespace monica_read_books_l117_117125

theorem monica_read_books (x : ℕ) 
    (h1 : 2 * (2 * x) + 5 = 69) : 
    x = 16 :=
by 
  sorry

end monica_read_books_l117_117125


namespace geometric_sequence_common_ratio_l117_117569

theorem geometric_sequence_common_ratio (a : ℕ → ℕ) (q : ℕ) (h2 : a 2 = 8) (h5 : a 5 = 64)
  (h_geom : ∀ n, a (n+1) = a n * q) : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l117_117569


namespace point_on_bisector_l117_117689

theorem point_on_bisector {a b : ℝ} (h : ∃ θ, θ = atan (b / a) ∧ (θ = π / 4 ∨ θ = -(3 * π / 4))) : b = -a :=
sorry

end point_on_bisector_l117_117689


namespace factor_cubic_expression_l117_117740

theorem factor_cubic_expression :
  ∃ a b c : ℕ, 
  a > b ∧ b > c ∧ 
  x^3 - 16 * x^2 + 65 * x - 80 = (x - a) * (x - b) * (x - c) ∧ 
  3 * b - c = 12 := 
sorry

end factor_cubic_expression_l117_117740


namespace fifty_eighth_digit_of_one_seventeenth_l117_117292

theorem fifty_eighth_digit_of_one_seventeenth :
  let decimal_repr := "0588235294117647" in
  let cycle_length := 16 in
  decimal_repr[(58 % cycle_length) - 1] = '4' :=
by
  -- Definitions and statements provided
  let decimal_repr := "0588235294117647"
  let cycle_length := 16
  have mod_calc : 58 % cycle_length = 10 := sorry
  show decimal_repr[10 - 1] = '4' from sorry

end fifty_eighth_digit_of_one_seventeenth_l117_117292


namespace combined_cost_price_l117_117294

theorem combined_cost_price :
  let face_value_A : ℝ := 100
  let discount_A : ℝ := 2
  let purchase_price_A := face_value_A - (discount_A / 100 * face_value_A)
  let brokerage_A := 0.2 / 100 * purchase_price_A
  let total_cost_price_A := purchase_price_A + brokerage_A

  let face_value_B : ℝ := 100
  let premium_B : ℝ := 1.5
  let purchase_price_B := face_value_B + (premium_B / 100 * face_value_B)
  let brokerage_B := 0.2 / 100 * purchase_price_B
  let total_cost_price_B := purchase_price_B + brokerage_B

  let combined_cost_price := total_cost_price_A + total_cost_price_B

  combined_cost_price = 199.899 := by
  sorry

end combined_cost_price_l117_117294


namespace Carol_max_chance_l117_117940

-- Definitions of the conditions
def Alice_random_choice (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 1
def Bob_random_choice (b : ℝ) : Prop := 0.4 ≤ b ∧ b ≤ 0.6
def Carol_wins (a b c : ℝ) : Prop := (a < c ∧ c < b) ∨ (b < c ∧ c < a)

-- Statement that Carol maximizes her chances by picking 0.5
theorem Carol_max_chance : ∃ c : ℝ, (∀ a b : ℝ, Alice_random_choice a → Bob_random_choice b → Carol_wins a b c) ∧ c = 0.5 := 
sorry

end Carol_max_chance_l117_117940


namespace values_of_n_eq_100_l117_117898

theorem values_of_n_eq_100 :
  ∃ (n_count : ℕ), n_count = 100 ∧
    ∀ (a b c : ℕ),
      a + 11 * b + 111 * c = 900 →
      (∀ (a : ℕ), a ≥ 0) →
      (∃ (n : ℕ), n = a + 2 * b + 3 * c ∧ n_count = 100) :=
sorry

end values_of_n_eq_100_l117_117898


namespace min_transport_cost_l117_117462

theorem min_transport_cost :
  let large_truck_capacity := 7
  let large_truck_cost := 600
  let small_truck_capacity := 4
  let small_truck_cost := 400
  let total_goods := 20
  ∃ (n_large n_small : ℕ),
    n_large * large_truck_capacity + n_small * small_truck_capacity ≥ total_goods ∧ 
    (n_large * large_truck_cost + n_small * small_truck_cost) = 1800 :=
sorry

end min_transport_cost_l117_117462


namespace sqrt_sum_eq_six_l117_117653

theorem sqrt_sum_eq_six (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := 
sorry

end sqrt_sum_eq_six_l117_117653


namespace sequence_arithmetic_l117_117108

theorem sequence_arithmetic (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h : ∀ n, S n = 2 * n^2 - 3 * n)
  (h₀ : S 0 = 0) 
  (h₁ : ∀ n, S (n+1) = S n + a (n+1)) :
  ∀ n, a n = 4 * n - 1 := sorry

end sequence_arithmetic_l117_117108


namespace initial_apples_l117_117736

-- Defining the conditions
def apples_handed_out := 8
def pies_made := 6
def apples_per_pie := 9
def apples_for_pies := pies_made * apples_per_pie

-- Prove the initial number of apples
theorem initial_apples : apples_handed_out + apples_for_pies = 62 :=
by
  sorry

end initial_apples_l117_117736


namespace total_shaded_area_l117_117470

theorem total_shaded_area (side_len : ℝ) (segment_len : ℝ) (h : ℝ) :
  side_len = 8 ∧ segment_len = 1 ∧ 0 ≤ h ∧ h ≤ 8 →
  (segment_len * h / 2 + segment_len * (side_len - h) / 2) = 4 := 
by
  intro h_cond
  rcases h_cond with ⟨h_side_len, h_segment_len, h_nonneg, h_le⟩
  -- Directly state the simplified computation
  sorry

end total_shaded_area_l117_117470


namespace circle_area_from_diameter_endpoints_l117_117592

theorem circle_area_from_diameter_endpoints :
  let C := (-2, 3)
  let D := (4, -1)
  let diameter := Real.sqrt ((4 - (-2))^2 + ((-1) - 3)^2)
  let radius := diameter / 2
  let area := Real.pi * radius^2
  C = (-2, 3) ∧ D = (4, -1) → area = 13 * Real.pi := by
    sorry

end circle_area_from_diameter_endpoints_l117_117592


namespace sin_minus_cos_l117_117385

variable (θ : ℝ)
hypothesis (h1 : 0 < θ ∧ θ < π / 2)
hypothesis (h2 : Real.tan θ = 1 / 3)

theorem sin_minus_cos (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = - (Real.sqrt 10) / 5 := 
by 
  sorry

end sin_minus_cos_l117_117385


namespace solve_equation_l117_117453

theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  2 / (x - 2) = (1 + x) / (x - 2) + 1 → x = 3 / 2 := by
  sorry

end solve_equation_l117_117453


namespace sin_minus_cos_theta_l117_117394

theorem sin_minus_cos_theta (θ : ℝ) (h₀ : 0 < θ ∧ θ < (π / 2)) 
  (h₁ : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -√10 / 5 :=
by
  sorry

end sin_minus_cos_theta_l117_117394


namespace inequality_must_hold_l117_117412

theorem inequality_must_hold (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 := 
sorry

end inequality_must_hold_l117_117412


namespace general_formula_seq_l117_117008

theorem general_formula_seq (a : ℕ → ℚ) :
  a 1 = 3 / 2 ∧ a 2 = 1 ∧ a 3 = 7 / 10 ∧ a 4 = 9 / 17 →
  ∀ n : ℕ, a n = (2 * n + 1) / (n * n + 1) :=
by {
  intros h,
  sorry
}

end general_formula_seq_l117_117008


namespace simplify_and_evaluate_expression_l117_117131

theorem simplify_and_evaluate_expression (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 2) :
  ( (2 * x - 3) / (x - 2) - 1 ) / ( (x^2 - 2 * x + 1) / (x - 2) ) = 1 / 2 :=
by {
  sorry
}

end simplify_and_evaluate_expression_l117_117131


namespace matrix_cube_l117_117954

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by
  sorry

end matrix_cube_l117_117954


namespace min_distance_from_C_to_circle_l117_117126

theorem min_distance_from_C_to_circle
  (R : ℝ) (AC : ℝ) (CB : ℝ) (C M : ℝ)
  (hR : R = 6) (hAC : AC = 4) (hCB : CB = 5)
  (hCM_eq : C = 12 - M) :
  C * M = 20 → (M < 6) → M = 2 := 
sorry

end min_distance_from_C_to_circle_l117_117126


namespace complement_intersection_is_correct_l117_117986

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {1, 2}
noncomputable def B : Set ℕ := {0, 2, 5}
noncomputable def complementA := (U \ A)

theorem complement_intersection_is_correct :
  complementA ∩ B = {0, 5} :=
by
  sorry

end complement_intersection_is_correct_l117_117986


namespace capsule_cost_difference_l117_117170

theorem capsule_cost_difference :
  let cost_per_capsule_r := 6.25 / 250
  let cost_per_capsule_t := 3.00 / 100
  cost_per_capsule_t - cost_per_capsule_r = 0.005 := by
  sorry

end capsule_cost_difference_l117_117170


namespace city_population_divided_l117_117908

theorem city_population_divided (total_population : ℕ) (parts : ℕ) (male_parts : ℕ) 
  (h1 : total_population = 1000) (h2 : parts = 5) (h3 : male_parts = 2) : 
  ∃ males : ℕ, males = 400 :=
by
  sorry

end city_population_divided_l117_117908


namespace monomials_like_terms_l117_117422

theorem monomials_like_terms (a b : ℕ) (h1 : 3 = a) (h2 : 4 = 2 * b) : a = 3 ∧ b = 2 :=
by
  sorry

end monomials_like_terms_l117_117422


namespace representatives_count_correct_l117_117877

noncomputable def assign_representatives_count : ℕ := 108 * (Nat.factorial 2014)

theorem representatives_count_correct:
  let S := {x | 1 ≤ x ∧ x ≤ 2014} in
  ∀ (r : (Set ℕ) → ℕ),
    (∀ T ⊆ S, T ≠ ∅ → r(T) ∈ T) →
    (∀ (A B C D : Set ℕ), 
       A ⊆ S ∧ B ⊆ S ∧ C ⊆ S ∧ D ⊆ S → 
       D = A ∪ B ∪ C ∧ A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ → 
       r(D) = r(A) ∨ r(D) = r(B) ∨ r(D) = r(C)
    ) →
  ∃! n : ℕ, n = assign_representatives_count
:= 
by {
  intros,
  -- proof would be here
  sorry
}

end representatives_count_correct_l117_117877
