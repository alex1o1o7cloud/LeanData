import Mathlib

namespace NUMINAMATH_GPT_linda_max_servings_is_13_l1379_137964

noncomputable def max_servings 
  (recipe_bananas : ℕ) (recipe_yogurt : ℕ) (recipe_honey : ℕ)
  (linda_bananas : ℕ) (linda_yogurt : ℕ) (linda_honey : ℕ)
  (servings_for_recipe : ℕ) : ℕ :=
  min 
    (linda_bananas * servings_for_recipe / recipe_bananas) 
    (min 
      (linda_yogurt * servings_for_recipe / recipe_yogurt)
      (linda_honey * servings_for_recipe / recipe_honey)
    )

theorem linda_max_servings_is_13 : 
  max_servings 3 2 1 10 9 4 4 = 13 :=
  sorry

end NUMINAMATH_GPT_linda_max_servings_is_13_l1379_137964


namespace NUMINAMATH_GPT_inequality_proof_l1379_137916

theorem inequality_proof (p : ℝ) (x y z v : ℝ) (hp : p ≥ 2) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hv : v ≥ 0) :
  (x + y) ^ p + (z + v) ^ p + (x + z) ^ p + (y + v) ^ p ≤ x ^ p + y ^ p + z ^ p + v ^ p + (x + y + z + v) ^ p := 
by sorry

end NUMINAMATH_GPT_inequality_proof_l1379_137916


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1379_137989

def U : Set ℕ := {x | 0 < x ∧ x < 9}

def S : Set ℕ := {1, 3, 5}

def T : Set ℕ := {3, 6}

theorem problem_part1 : S ∩ T = {3} := by
  sorry

theorem problem_part2 : U \ (S ∪ T) = {2, 4, 7, 8} := by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1379_137989


namespace NUMINAMATH_GPT_guest_bedroom_ratio_l1379_137907

theorem guest_bedroom_ratio 
  (lr_dr_kitchen : ℝ) (total_house : ℝ) (master_bedroom : ℝ) (guest_bedroom : ℝ) 
  (h1 : lr_dr_kitchen = 1000) 
  (h2 : total_house = 2300)
  (h3 : master_bedroom = 1040)
  (h4 : guest_bedroom = total_house - (lr_dr_kitchen + master_bedroom)) :
  guest_bedroom / master_bedroom = 1 / 4 := 
by
  sorry

end NUMINAMATH_GPT_guest_bedroom_ratio_l1379_137907


namespace NUMINAMATH_GPT_problem_statement_l1379_137941

theorem problem_statement (x : ℝ) (h : 0 < x) : x + 2016^2016 / x^2016 ≥ 2017 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1379_137941


namespace NUMINAMATH_GPT_big_eighteen_basketball_games_count_l1379_137930

def num_teams_in_division := 6
def num_teams := 18
def games_within_division := 3
def games_between_divisions := 1
def divisions := 3

theorem big_eighteen_basketball_games_count :
  (num_teams * ((num_teams_in_division - 1) * games_within_division + (num_teams - num_teams_in_division) * games_between_divisions)) / 2 = 243 :=
by
  have teams_in_other_divisions : num_teams - num_teams_in_division = 12 := rfl
  have games_per_team_within_division : (num_teams_in_division - 1) * games_within_division = 15 := rfl
  have games_per_team_between_division : 12 * games_between_divisions = 12 := rfl
  sorry

end NUMINAMATH_GPT_big_eighteen_basketball_games_count_l1379_137930


namespace NUMINAMATH_GPT_trigonometric_inequality_l1379_137972

theorem trigonometric_inequality (x : Real) (n : Int) :
  (9.286 * (Real.sin x)^3 * Real.sin ((Real.pi / 2) - 3 * x) +
   (Real.cos x)^3 * Real.cos ((Real.pi / 2) - 3 * x) > 
   3 * Real.sqrt 3 / 8) →
   (x > (Real.pi / 12) + (Real.pi * n / 2) ∧
   x < (5 * Real.pi / 12) + (Real.pi * n / 2)) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_l1379_137972


namespace NUMINAMATH_GPT_sampling_methods_used_l1379_137920

-- Definitions based on problem conditions
def TotalHouseholds : Nat := 2000
def FarmerHouseholds : Nat := 1800
def WorkerHouseholds : Nat := 100
def IntellectualHouseholds : Nat := TotalHouseholds - FarmerHouseholds - WorkerHouseholds
def SampleSize : Nat := 40

-- The statement of the proof problem
theorem sampling_methods_used
  (N : Nat := TotalHouseholds)
  (F : Nat := FarmerHouseholds)
  (W : Nat := WorkerHouseholds)
  (I : Nat := IntellectualHouseholds)
  (S : Nat := SampleSize)
:
  (1 ∈ [1, 2, 3]) ∧ (2 ∈ [1, 2, 3]) ∧ (3 ∈ [1, 2, 3]) :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_sampling_methods_used_l1379_137920


namespace NUMINAMATH_GPT_BANANA_permutations_l1379_137912

-- Define the total number of letters in BANANA
def totalLetters : Nat := 6

-- Define the counts of each letter
def countA : Nat := 3
def countN : Nat := 2
def countB : Nat := 1

-- Define factorial function
def fact : Nat → Nat
| 0 => 1
| n + 1 => (n + 1) * fact n

-- Calculate the number of distinct permutations
def numberOfPermutations : Nat :=
  fact totalLetters / (fact countA * fact countN * fact countB)

-- Statement of the theorem
theorem BANANA_permutations : numberOfPermutations = 60 := by
  sorry

end NUMINAMATH_GPT_BANANA_permutations_l1379_137912


namespace NUMINAMATH_GPT_man_speed_against_current_l1379_137908

-- Definitions for the problem conditions
def man_speed_with_current : ℝ := 21
def current_speed : ℝ := 4.3

-- Main proof statement
theorem man_speed_against_current : man_speed_with_current - 2 * current_speed = 12.4 :=
by
  sorry

end NUMINAMATH_GPT_man_speed_against_current_l1379_137908


namespace NUMINAMATH_GPT_find_K_l1379_137998

theorem find_K : ∃ K : ℕ, (64 ^ (2 / 3) * 16 ^ 2) / 4 = 2 ^ K :=
by
  use 10
  sorry

end NUMINAMATH_GPT_find_K_l1379_137998


namespace NUMINAMATH_GPT_find_counterfeit_coins_l1379_137915

structure Coins :=
  (a a₁ b b₁ c c₁ : ℝ)
  (genuine_weight : ℝ)
  (counterfeit_weight : ℝ)
  (a_is_genuine_or_counterfeit : a = genuine_weight ∨ a = counterfeit_weight)
  (a₁_is_genuine_or_counterfeit : a₁ = genuine_weight ∨ a₁ = counterfeit_weight)
  (b_is_genuine_or_counterfeit : b = genuine_weight ∨ b = counterfeit_weight)
  (b₁_is_genuine_or_counterfeit : b₁ = genuine_weight ∨ b₁ = counterfeit_weight)
  (c_is_genuine_or_counterfeit : c = genuine_weight ∨ c = counterfeit_weight)
  (c₁_is_genuine_or_counterfeit : c₁ = genuine_weight ∨ c₁ = counterfeit_weight)
  (counterfeit_pair_ends_unit_segment : (a = counterfeit_weight ∧ a₁ = counterfeit_weight) 
                                        ∨ (b = counterfeit_weight ∧ b₁ = counterfeit_weight)
                                        ∨ (c = counterfeit_weight ∧ c₁ = counterfeit_weight))

theorem find_counterfeit_coins (coins : Coins) : 
  (coins.a = coins.genuine_weight ∧ coins.b = coins.genuine_weight → coins.a₁ = coins.counterfeit_weight ∧ coins.b₁ = coins.counterfeit_weight) 
  ∧ (coins.a < coins.b → coins.a = coins.counterfeit_weight ∧ coins.b₁ = coins.counterfeit_weight) 
  ∧ (coins.b < coins.a → coins.b = coins.counterfeit_weight ∧ coins.a₁ = coins.counterfeit_weight) := 
by
  sorry

end NUMINAMATH_GPT_find_counterfeit_coins_l1379_137915


namespace NUMINAMATH_GPT_original_price_per_pound_l1379_137945

theorem original_price_per_pound (P x : ℝ)
  (h1 : 0.2 * x * P = 0.2 * x)
  (h2 : x * P = x * P)
  (h3 : 1.08 * (0.8 * x) * 1.08 = 1.08 * x * P) :
  P = 1.08 :=
sorry

end NUMINAMATH_GPT_original_price_per_pound_l1379_137945


namespace NUMINAMATH_GPT_ellipse_major_axis_focal_distance_l1379_137955

theorem ellipse_major_axis_focal_distance (m : ℝ) (h1 : 10 - m > 0) (h2 : m - 2 > 0) 
  (h3 : ∀ x y, x^2 / (10 - m) + y^2 / (m - 2) = 1) 
  (h4 : ∃ c, 2 * c = 4 ∧ c^2 = (m - 2) - (10 - m)) : m = 8 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_major_axis_focal_distance_l1379_137955


namespace NUMINAMATH_GPT_smallest_pieces_to_remove_l1379_137902

theorem smallest_pieces_to_remove 
  (total_fruit : ℕ)
  (friends : ℕ)
  (h_fruit : total_fruit = 30)
  (h_friends : friends = 4) 
  : ∃ k : ℕ, k = 2 ∧ ((total_fruit - k) % friends = 0) :=
sorry

end NUMINAMATH_GPT_smallest_pieces_to_remove_l1379_137902


namespace NUMINAMATH_GPT_smallest_value_of_n_l1379_137999

/-- Given that Casper has exactly enough money to buy either 
  18 pieces of red candy, 20 pieces of green candy, 
  25 pieces of blue candy, or n pieces of purple candy where 
  each purple candy costs 30 cents, prove that the smallest 
  possible value of n is 30.
-/
theorem smallest_value_of_n
  (r g b n : ℕ)
  (h : 18 * r = 20 * g ∧ 20 * g = 25 * b ∧ 25 * b = 30 * n) : 
  n = 30 :=
sorry

end NUMINAMATH_GPT_smallest_value_of_n_l1379_137999


namespace NUMINAMATH_GPT_complex_square_eq_l1379_137917

theorem complex_square_eq (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : (a + b * Complex.I) ^ 2 = 7 + 24 * Complex.I) : 
  a + b * Complex.I = 4 + 3 * Complex.I :=
by
  sorry

end NUMINAMATH_GPT_complex_square_eq_l1379_137917


namespace NUMINAMATH_GPT_initial_apples_l1379_137921

theorem initial_apples (picked: ℕ) (newly_grown: ℕ) (still_on_tree: ℕ) (initial: ℕ):
  (picked = 7) →
  (newly_grown = 2) →
  (still_on_tree = 6) →
  (still_on_tree + picked - newly_grown = initial) →
  initial = 11 :=
by
  intros hpicked hnewly_grown hstill_on_tree hcalculation
  sorry

end NUMINAMATH_GPT_initial_apples_l1379_137921


namespace NUMINAMATH_GPT_tina_jumps_more_than_cindy_l1379_137958

def cindy_jumps : ℕ := 12
def betsy_jumps : ℕ := cindy_jumps / 2
def tina_jumps : ℕ := betsy_jumps * 3

theorem tina_jumps_more_than_cindy : tina_jumps - cindy_jumps = 6 := by
  sorry

end NUMINAMATH_GPT_tina_jumps_more_than_cindy_l1379_137958


namespace NUMINAMATH_GPT_even_function_A_value_l1379_137997

-- Given function definition
def f (x : ℝ) (A : ℝ) : ℝ := (x + 1) * (x - A)

-- Statement to prove
theorem even_function_A_value (A : ℝ) (h : ∀ x : ℝ, f x A = f (-x) A) : A = 1 :=
by
  sorry

end NUMINAMATH_GPT_even_function_A_value_l1379_137997


namespace NUMINAMATH_GPT_cube_sum_l1379_137922

theorem cube_sum (a b : ℝ) (h1 : a + b = 13) (h2 : a * b = 41) : a^3 + b^3 = 598 :=
by
  sorry

end NUMINAMATH_GPT_cube_sum_l1379_137922


namespace NUMINAMATH_GPT_solution_set_inequality_l1379_137956

theorem solution_set_inequality {a b c : ℝ} (h₁ : a < 0)
  (h₂ : ∀ x : ℝ, (a * x^2 + b * x + c <= 0) ↔ (x <= -(1/3) ∨ 2 <= x)) :
  (∀ x : ℝ, (c * x^2 + b * x + a > 0) ↔ (x < -3 ∨ 1/2 < x)) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1379_137956


namespace NUMINAMATH_GPT_carol_rectangle_width_l1379_137969

def carol_width (lengthC : ℕ) (widthJ : ℕ) (lengthJ : ℕ) (widthC : ℕ) :=
  lengthC * widthC = lengthJ * widthJ

theorem carol_rectangle_width 
  {lengthC widthJ lengthJ : ℕ} (h1 : lengthC = 8)
  (h2 : widthJ = 30) (h3 : lengthJ = 4)
  (h4 : carol_width lengthC widthJ lengthJ 15) : 
  widthC = 15 :=
by 
  subst h1
  subst h2
  subst h3
  sorry -- proof not required

end NUMINAMATH_GPT_carol_rectangle_width_l1379_137969


namespace NUMINAMATH_GPT_range_of_m_l1379_137919

-- Definitions based on the conditions
def p (m : ℝ) : Prop := 4 - 4 * m > 0
def q (m : ℝ) : Prop := m + 2 > 0

-- Problem statement in Lean 4
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ≤ -2 ∨ m ≥ 1 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1379_137919


namespace NUMINAMATH_GPT_roots_nonpositive_if_ac_le_zero_l1379_137942

theorem roots_nonpositive_if_ac_le_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : a * c ≤ 0) :
  ¬ (∀ x : ℝ, x^2 - (b/a)*x + (c/a) = 0 → x > 0) :=
sorry

end NUMINAMATH_GPT_roots_nonpositive_if_ac_le_zero_l1379_137942


namespace NUMINAMATH_GPT_cost_of_10_apples_l1379_137931

-- Define the price for 10 apples as a variable
noncomputable def price_10_apples (P : ℝ) : ℝ := P

-- Theorem stating that the cost for 10 apples is the provided price
theorem cost_of_10_apples (P : ℝ) : price_10_apples P = P :=
  by
    sorry

end NUMINAMATH_GPT_cost_of_10_apples_l1379_137931


namespace NUMINAMATH_GPT_find_solutions_l1379_137996

def satisfies_inequality (x : ℝ) : Prop :=
  (Real.cos x)^2018 + (1 / (Real.sin x))^2019 ≤ (Real.sin x)^2018 + (1 / (Real.cos x))^2019

def in_intervals (x : ℝ) : Prop :=
  (x ∈ Set.Ico (-Real.pi / 3) 0) ∨
  (x ∈ Set.Ico (Real.pi / 4) (Real.pi / 2)) ∨
  (x ∈ Set.Ioc Real.pi (5 * Real.pi / 4)) ∨
  (x ∈ Set.Ioc (3 * Real.pi / 2) (5 * Real.pi / 3))

theorem find_solutions :
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi / 3) (5 * Real.pi / 3) →
  satisfies_inequality x ↔ in_intervals x := 
  by sorry

end NUMINAMATH_GPT_find_solutions_l1379_137996


namespace NUMINAMATH_GPT_find_y_l1379_137933

theorem find_y (x y : ℝ) (h₁ : 1.5 * x = 0.3 * y) (h₂ : x = 20) : y = 100 :=
sorry

end NUMINAMATH_GPT_find_y_l1379_137933


namespace NUMINAMATH_GPT_minimum_value_l1379_137904

variable (a b : ℝ)

-- Assume a and b are positive real numbers
variable (h₀ : 0 < a)
variable (h₁ : 0 < b)

-- Given the condition a + b = 2
variable (h₂ : a + b = 2)

theorem minimum_value : (1 / a) + (2 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_GPT_minimum_value_l1379_137904


namespace NUMINAMATH_GPT_avg_age_of_women_l1379_137962

theorem avg_age_of_women (T : ℕ) (W : ℕ) (T_avg : ℕ) (H1 : T_avg = T / 10)
  (H2 : (T_avg + 6) = ((T - 18 - 22 + W) / 10)) : (W / 2) = 50 :=
sorry

end NUMINAMATH_GPT_avg_age_of_women_l1379_137962


namespace NUMINAMATH_GPT_max_n_l1379_137936

noncomputable def prod := 160 * 170 * 180 * 190

theorem max_n : ∃ n : ℕ, n = 30499 ∧ n^2 ≤ prod := by
  sorry

end NUMINAMATH_GPT_max_n_l1379_137936


namespace NUMINAMATH_GPT_minimum_value_of_function_l1379_137948

theorem minimum_value_of_function (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  ∃ y : ℝ, (∀ z : ℝ, z = (1 / x) + (4 / (1 - x)) → y ≤ z) ∧ y = 9 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_function_l1379_137948


namespace NUMINAMATH_GPT_alyssa_puppies_l1379_137952

theorem alyssa_puppies (total_puppies : ℕ) (given_away : ℕ) (remaining_puppies : ℕ) 
  (h1 : total_puppies = 7) (h2 : given_away = 5) 
  : remaining_puppies = total_puppies - given_away → remaining_puppies = 2 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end NUMINAMATH_GPT_alyssa_puppies_l1379_137952


namespace NUMINAMATH_GPT_jim_gas_gallons_l1379_137914

theorem jim_gas_gallons (G : ℕ) (C_NC C_VA : ℕ → ℕ) 
  (h₁ : ∀ G, C_NC G = 2 * G)
  (h₂ : ∀ G, C_VA G = 3 * G)
  (h₃ : C_NC G + C_VA G = 50) :
  G = 10 := 
sorry

end NUMINAMATH_GPT_jim_gas_gallons_l1379_137914


namespace NUMINAMATH_GPT_triangle_inequality_l1379_137934

theorem triangle_inequality (a : ℝ) (h1 : a + 3 > 5) (h2 : a + 5 > 3) (h3 : 3 + 5 > a) :
  2 < a ∧ a < 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_inequality_l1379_137934


namespace NUMINAMATH_GPT_range_of_a_l1379_137974

open Real

theorem range_of_a (a x y : ℝ)
  (h1 : (x - a) ^ 2 + (y - (a + 2)) ^ 2 = 1)
  (h2 : ∃ M : ℝ × ℝ, (M.1 - a) ^ 2 + (M.2 - (a + 2)) ^ 2 = 1
                       ∧ dist M (0, 3) = 2 * dist M (0, 0)) :
  -3 ≤ a ∧ a ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1379_137974


namespace NUMINAMATH_GPT_apples_handed_out_to_students_l1379_137971

def initial_apples : ℕ := 47
def apples_per_pie : ℕ := 4
def number_of_pies : ℕ := 5
def apples_for_pies : ℕ := number_of_pies * apples_per_pie

theorem apples_handed_out_to_students : 
  initial_apples - apples_for_pies = 27 := 
by
  -- Since 20 apples are used for pies and there were initially 47 apples,
  -- it follows that 27 apples were handed out to students.
  sorry

end NUMINAMATH_GPT_apples_handed_out_to_students_l1379_137971


namespace NUMINAMATH_GPT_ellipse_foci_coordinates_l1379_137991

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ),
  (x^2 / 9 + y^2 / 5 = 1) →
  (x, y) = (2, 0) ∨ (x, y) = (-2, 0) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_foci_coordinates_l1379_137991


namespace NUMINAMATH_GPT_pastor_prayer_ratio_l1379_137928

theorem pastor_prayer_ratio 
  (R : ℚ) 
  (paul_prays_per_day : ℚ := 20)
  (paul_sunday_times : ℚ := 2 * paul_prays_per_day)
  (paul_total : ℚ := 6 * paul_prays_per_day + paul_sunday_times)
  (bruce_ratio : ℚ := R)
  (bruce_prays_per_day : ℚ := bruce_ratio * paul_prays_per_day)
  (bruce_sunday_times : ℚ := 2 * paul_sunday_times)
  (bruce_total : ℚ := 6 * bruce_prays_per_day + bruce_sunday_times)
  (condition : paul_total = bruce_total + 20) :
  R = 1/2 :=
sorry

end NUMINAMATH_GPT_pastor_prayer_ratio_l1379_137928


namespace NUMINAMATH_GPT_additional_people_needed_to_mow_lawn_l1379_137927

theorem additional_people_needed_to_mow_lawn :
  (∀ (k : ℕ), (∀ (n t : ℕ), n * t = k) → (4 * 6 = k) → (∃ (n : ℕ), n * 3 = k) → (8 - 4 = 4)) :=
by sorry

end NUMINAMATH_GPT_additional_people_needed_to_mow_lawn_l1379_137927


namespace NUMINAMATH_GPT_total_walnut_trees_in_park_l1379_137946

theorem total_walnut_trees_in_park 
  (initial_trees planted_by_first planted_by_second planted_by_third removed_trees : ℕ)
  (h_initial : initial_trees = 22)
  (h_first : planted_by_first = 12)
  (h_second : planted_by_second = 15)
  (h_third : planted_by_third = 10)
  (h_removed : removed_trees = 4) :
  initial_trees + (planted_by_first + planted_by_second + planted_by_third - removed_trees) = 55 :=
by
  sorry

end NUMINAMATH_GPT_total_walnut_trees_in_park_l1379_137946


namespace NUMINAMATH_GPT_find_divisor_l1379_137903

-- Define the conditions
def dividend : ℕ := 22
def quotient : ℕ := 7
def remainder : ℕ := 1

-- The divisor is what we need to find
def divisor : ℕ := 3

-- The proof problem: proving that the given conditions imply the divisor is 3
theorem find_divisor :
  ∃ d : ℕ, dividend = d * quotient + remainder ∧ d = divisor :=
by
  use 3
  -- Replace actual proof with sorry for now
  sorry

end NUMINAMATH_GPT_find_divisor_l1379_137903


namespace NUMINAMATH_GPT_intersection_of_lines_l1379_137960

theorem intersection_of_lines :
  ∃ (x y : ℝ), 10 * x - 5 * y = 5 ∧ 8 * x + 2 * y = 22 ∧ x = 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_GPT_intersection_of_lines_l1379_137960


namespace NUMINAMATH_GPT_angle_negative_225_in_second_quadrant_l1379_137984

def inSecondQuadrant (angle : Int) : Prop :=
  angle % 360 > -270 ∧ angle % 360 <= -180

theorem angle_negative_225_in_second_quadrant :
  inSecondQuadrant (-225) :=
by
  sorry

end NUMINAMATH_GPT_angle_negative_225_in_second_quadrant_l1379_137984


namespace NUMINAMATH_GPT_total_distance_traveled_l1379_137967

/-- Defining the distance Greg travels in each leg of his trip -/
def distance_workplace_to_market : ℕ := 30

def distance_market_to_friend : ℕ := distance_workplace_to_market + 10

def distance_friend_to_aunt : ℕ := 5

def distance_aunt_to_grocery : ℕ := 7

def distance_grocery_to_home : ℕ := 18

/-- The total distance Greg traveled during his entire trip is the sum of all individual distances -/
theorem total_distance_traveled :
  distance_workplace_to_market + distance_market_to_friend + distance_friend_to_aunt + distance_aunt_to_grocery + distance_grocery_to_home = 100 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l1379_137967


namespace NUMINAMATH_GPT_final_quantity_of_milk_l1379_137910

-- Define initial conditions
def initial_volume : ℝ := 60
def removed_volume : ℝ := 9

-- Given the initial conditions, calculate the quantity of milk left after two dilutions
theorem final_quantity_of_milk :
  let first_removal_ratio := initial_volume - removed_volume / initial_volume
  let first_milk_volume := initial_volume * (first_removal_ratio)
  let second_removal_ratio := first_milk_volume / initial_volume
  let second_milk_volume := first_milk_volume * (second_removal_ratio)
  second_milk_volume = 43.35 :=
by
  sorry

end NUMINAMATH_GPT_final_quantity_of_milk_l1379_137910


namespace NUMINAMATH_GPT_trigonometric_identity_l1379_137923

noncomputable def point_on_terminal_side (x y : ℝ) : Prop :=
    ∃ α : ℝ, x = Real.cos α ∧ y = Real.sin α

theorem trigonometric_identity (x y : ℝ) (h : point_on_terminal_side 1 3) :
    (Real.sin (π - α) - Real.sin (π / 2 + α)) / (2 * Real.cos (α - 2 * π)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1379_137923


namespace NUMINAMATH_GPT_find_values_of_a_and_c_l1379_137954

theorem find_values_of_a_and_c
  (a c : ℝ)
  (h1 : ∀ x : ℝ, (1 / 3 < x ∧ x < 1 / 2) ↔ a * x^2 + 5 * x + c > 0) :
  a = -6 ∧ c = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_a_and_c_l1379_137954


namespace NUMINAMATH_GPT_ratio_of_population_is_correct_l1379_137901

noncomputable def ratio_of_population (M W C : ℝ) : ℝ :=
  (M / (W + C)) * 100

theorem ratio_of_population_is_correct
  (M W C : ℝ) 
  (hW: W = 0.9 * M)
  (hC: C = 0.6 * (M + W)) :
  ratio_of_population M W C = 49.02 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_population_is_correct_l1379_137901


namespace NUMINAMATH_GPT_staff_meeting_doughnuts_l1379_137987

theorem staff_meeting_doughnuts (n_d n_s n_l : ℕ) (h₁ : n_d = 50) (h₂ : n_s = 19) (h₃ : n_l = 12) :
  (n_d - n_l) / n_s = 2 :=
by
  sorry

end NUMINAMATH_GPT_staff_meeting_doughnuts_l1379_137987


namespace NUMINAMATH_GPT_find_first_term_l1379_137935

noncomputable def firstTermOfGeometricSeries (S : ℝ) (r : ℝ) : ℝ :=
  S * (1 - r) / (1 - r)

theorem find_first_term
  (S : ℝ)
  (r : ℝ)
  (hS : S = 20)
  (hr : r = -3/7) :
  firstTermOfGeometricSeries S r = 200 / 7 :=
  by
    rw [hS, hr]
    sorry

end NUMINAMATH_GPT_find_first_term_l1379_137935


namespace NUMINAMATH_GPT_gcd_factorial_8_6_squared_l1379_137982

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_6_squared :
  Nat.gcd (factorial 8) ((factorial 6) ^ 2) = 7200 :=
by
  sorry

end NUMINAMATH_GPT_gcd_factorial_8_6_squared_l1379_137982


namespace NUMINAMATH_GPT_isabela_spent_2800_l1379_137949

/-- Given:
1. Isabela bought twice as many cucumbers as pencils.
2. Both cucumbers and pencils cost $20 each.
3. Isabela got a 20% discount on the pencils.
4. She bought 100 cucumbers.
Prove that the total amount Isabela spent is $2800. -/
theorem isabela_spent_2800 :
  ∀ (pencils cucumbers : ℕ) (pencil_cost cucumber_cost : ℤ) (discount rate: ℚ)
    (total_cost pencils_cost cucumbers_cost discount_amount : ℤ),
  cucumbers = 100 →
  pencils * 2 = cucumbers →
  pencil_cost = 20 →
  cucumber_cost = 20 →
  rate = 20 / 100 →
  pencils_cost = pencils * pencil_cost →
  discount_amount = pencils_cost * rate →
  total_cost = pencils_cost - discount_amount + cucumbers * cucumber_cost →
  total_cost = 2800 := by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end NUMINAMATH_GPT_isabela_spent_2800_l1379_137949


namespace NUMINAMATH_GPT_probability_x_add_y_lt_4_in_square_l1379_137951

noncomputable def square_area : ℝ := 3 * 3

noncomputable def triangle_area : ℝ := (1 / 2) * 2 * 2

noncomputable def region_area : ℝ := square_area - triangle_area

noncomputable def probability (A B : ℝ) : ℝ := A / B

theorem probability_x_add_y_lt_4_in_square :
  probability region_area square_area = 7 / 9 :=
by 
  sorry

end NUMINAMATH_GPT_probability_x_add_y_lt_4_in_square_l1379_137951


namespace NUMINAMATH_GPT_max_marks_l1379_137938

theorem max_marks (M : ℝ) (h : 0.80 * M = 240) : M = 300 :=
sorry

end NUMINAMATH_GPT_max_marks_l1379_137938


namespace NUMINAMATH_GPT_Amanda_needs_12_more_marbles_l1379_137950

theorem Amanda_needs_12_more_marbles (K A M : ℕ)
  (h1 : M = 5 * K)
  (h2 : M = 85)
  (h3 : M = A + 63) :
  A + 12 = 2 * K := 
sorry

end NUMINAMATH_GPT_Amanda_needs_12_more_marbles_l1379_137950


namespace NUMINAMATH_GPT_surface_area_of_circumscribed_sphere_of_triangular_pyramid_l1379_137939

theorem surface_area_of_circumscribed_sphere_of_triangular_pyramid
  (a : ℝ)
  (h₁ : a > 0) : 
  ∃ S, S = (27 * π / 32 * a^2) := 
by
  sorry

end NUMINAMATH_GPT_surface_area_of_circumscribed_sphere_of_triangular_pyramid_l1379_137939


namespace NUMINAMATH_GPT_digit_7_count_correct_l1379_137978

def base8ToBase10 (n : Nat) : Nat :=
  -- converting base 8 number 1000 to base 10
  1 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0

def countDigit7 (n : Nat) : Nat :=
  -- counts the number of times the digit '7' appears in numbers from 1 to n
  let digits := (List.range (n + 1)).map fun x => x.digits 10
  digits.foldl (fun acc ds => acc + ds.count 7) 0

theorem digit_7_count_correct : countDigit7 512 = 123 := by
  sorry

end NUMINAMATH_GPT_digit_7_count_correct_l1379_137978


namespace NUMINAMATH_GPT_gcd_64_144_l1379_137973

theorem gcd_64_144 : Nat.gcd 64 144 = 16 := by
  sorry

end NUMINAMATH_GPT_gcd_64_144_l1379_137973


namespace NUMINAMATH_GPT_min_value_abs_function_l1379_137926

theorem min_value_abs_function : ∀ (x : ℝ), (|x + 1| + |2 - x|) ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_abs_function_l1379_137926


namespace NUMINAMATH_GPT_Valley_Forge_High_School_winter_carnival_l1379_137983

noncomputable def number_of_girls (total_students : ℕ) (total_participants : ℕ) (fraction_girls_participating : ℚ) (fraction_boys_participating : ℚ) : ℕ := sorry

theorem Valley_Forge_High_School_winter_carnival
  (total_students : ℕ)
  (total_participants : ℕ)
  (fraction_girls_participating : ℚ)
  (fraction_boys_participating : ℚ)
  (h_total_students : total_students = 1500)
  (h_total_participants : total_participants = 900)
  (h_fraction_girls : fraction_girls_participating = 3 / 4)
  (h_fraction_boys : fraction_boys_participating = 2 / 3) :
  number_of_girls total_students total_participants fraction_girls_participating fraction_boys_participating = 900 := sorry

end NUMINAMATH_GPT_Valley_Forge_High_School_winter_carnival_l1379_137983


namespace NUMINAMATH_GPT_cost_of_blue_hat_is_six_l1379_137990

-- Given conditions
def total_hats : ℕ := 85
def green_hats : ℕ := 40
def blue_hats : ℕ := total_hats - green_hats
def cost_green_hat : ℕ := 7
def total_cost : ℕ := 550
def total_cost_green_hats : ℕ := green_hats * cost_green_hat
def total_cost_blue_hats : ℕ := total_cost - total_cost_green_hats
def cost_blue_hat : ℕ := total_cost_blue_hats / blue_hats

-- Proof statement
theorem cost_of_blue_hat_is_six : cost_blue_hat = 6 := sorry

end NUMINAMATH_GPT_cost_of_blue_hat_is_six_l1379_137990


namespace NUMINAMATH_GPT_share_of_a_l1379_137924

def shares_sum (a b c : ℝ) := a + b + c = 366
def share_a (a b c : ℝ) := a = 1/2 * (b + c)
def share_b (a b c : ℝ) := b = 2/3 * (a + c)

theorem share_of_a (a b c : ℝ) 
  (h1 : shares_sum a b c) 
  (h2 : share_a a b c) 
  (h3 : share_b a b c) : 
  a = 122 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_share_of_a_l1379_137924


namespace NUMINAMATH_GPT_find_n_sequence_sum_l1379_137977

theorem find_n_sequence_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₀ : ∀ n, a n = (2^n - 1) / 2^n)
  (h₁ : S 6 = 321 / 64) :
  ∃ n, S n = 321 / 64 ∧ n = 6 := 
by 
  sorry

end NUMINAMATH_GPT_find_n_sequence_sum_l1379_137977


namespace NUMINAMATH_GPT_initial_visual_range_is_90_l1379_137979

-- Define the initial visual range without the telescope (V).
variable (V : ℝ)

-- Define the condition that the visual range with the telescope is 150 km.
variable (condition1 : V + (2 / 3) * V = 150)

-- Define the proof problem statement.
theorem initial_visual_range_is_90 (V : ℝ) (condition1 : V + (2 / 3) * V = 150) : V = 90 :=
sorry

end NUMINAMATH_GPT_initial_visual_range_is_90_l1379_137979


namespace NUMINAMATH_GPT_complement_union_l1379_137988

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}
def complement_U_A : Set ℕ := U \ A

theorem complement_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) :
  (complement_U_A ∪ B) = {0, 2, 4} := by
  sorry

end NUMINAMATH_GPT_complement_union_l1379_137988


namespace NUMINAMATH_GPT_age_of_father_now_l1379_137986

variable (M F : ℕ)

theorem age_of_father_now :
  (M = 2 * F / 5) ∧ (M + 14 = (F + 14) / 2) → F = 70 :=
by 
sorry

end NUMINAMATH_GPT_age_of_father_now_l1379_137986


namespace NUMINAMATH_GPT_sparse_real_nums_l1379_137994

noncomputable def is_sparse (r : ℝ) : Prop :=
  ∃n > 0, ∀s : ℝ, s^n = r → s = 1 ∨ s = -1 ∨ s = 0

theorem sparse_real_nums (r : ℝ) : is_sparse r ↔ r = -1 ∨ r = 0 ∨ r = 1 := 
by
  sorry

end NUMINAMATH_GPT_sparse_real_nums_l1379_137994


namespace NUMINAMATH_GPT_decimal_sum_sqrt_l1379_137944

theorem decimal_sum_sqrt (a b : ℝ) (h₁ : a = Real.sqrt 5 - 2) (h₂ : b = Real.sqrt 13 - 3) : 
  a + b - Real.sqrt 5 = Real.sqrt 13 - 5 := by
  sorry

end NUMINAMATH_GPT_decimal_sum_sqrt_l1379_137944


namespace NUMINAMATH_GPT_train_speed_l1379_137937

theorem train_speed (train_length : ℕ) (cross_time : ℕ) (speed : ℕ) 
  (h_train_length : train_length = 300)
  (h_cross_time : cross_time = 10)
  (h_speed_eq : speed = train_length / cross_time) : 
  speed = 30 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1379_137937


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l1379_137909

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, x + (m + 1) * y = 2 - m → m * x + 2 * y = -8) ↔ m = -2 / 3 :=
by sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l1379_137909


namespace NUMINAMATH_GPT_rhombus_area_l1379_137992

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 30) (h2 : d2 = 12) : (d1 * d2) / 2 = 180 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1379_137992


namespace NUMINAMATH_GPT_find_pool_length_l1379_137929

noncomputable def pool_length : ℝ :=
  let drain_rate := 60 -- cubic feet per minute
  let width := 40 -- feet
  let depth := 10 -- feet
  let capacity_percent := 0.80
  let drain_time := 800 -- minutes
  let drained_volume := drain_rate * drain_time -- cubic feet
  let full_capacity := drained_volume / capacity_percent -- cubic feet
  let length := full_capacity / (width * depth) -- feet
  length

theorem find_pool_length : pool_length = 150 := by
  sorry

end NUMINAMATH_GPT_find_pool_length_l1379_137929


namespace NUMINAMATH_GPT_penny_remaining_money_l1379_137981

theorem penny_remaining_money (initial_money : ℤ) (socks_pairs : ℤ) (socks_cost_per_pair : ℤ) (hat_cost : ℤ) :
  initial_money = 20 → socks_pairs = 4 → socks_cost_per_pair = 2 → hat_cost = 7 → 
  initial_money - (socks_pairs * socks_cost_per_pair + hat_cost) = 5 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end NUMINAMATH_GPT_penny_remaining_money_l1379_137981


namespace NUMINAMATH_GPT_scientific_notation_of_0_0000003_l1379_137913

theorem scientific_notation_of_0_0000003 : 0.0000003 = 3 * 10^(-7) := by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_0_0000003_l1379_137913


namespace NUMINAMATH_GPT_equation_of_line_with_x_intercept_and_slope_l1379_137975

theorem equation_of_line_with_x_intercept_and_slope :
  ∃ (a b c : ℝ), a * x - b * y + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
sorry

end NUMINAMATH_GPT_equation_of_line_with_x_intercept_and_slope_l1379_137975


namespace NUMINAMATH_GPT_find_c_value_l1379_137906

theorem find_c_value 
  (b : ℝ) 
  (h1 : ∀ x : ℝ, x^2 + b * x + 3 ≥ 0) 
  (h2 : ∀ m c : ℝ, (∀ x : ℝ, x^2 + b * x + 3 < c ↔ m - 8 < x ∧ x < m)) 
  : c = 16 :=
sorry

end NUMINAMATH_GPT_find_c_value_l1379_137906


namespace NUMINAMATH_GPT_man_swim_upstream_distance_l1379_137961

theorem man_swim_upstream_distance (c d : ℝ) (h1 : 15.5 + c ≠ 0) (h2 : 15.5 - c ≠ 0) :
  (15.5 + c) * 2 = 36 ∧ (15.5 - c) * 2 = d → d = 26 := by
  sorry

end NUMINAMATH_GPT_man_swim_upstream_distance_l1379_137961


namespace NUMINAMATH_GPT_number_of_pickup_trucks_l1379_137932

theorem number_of_pickup_trucks 
  (cars : ℕ) (bicycles : ℕ) (tricycles : ℕ) (total_tires : ℕ)
  (tires_per_car : ℕ) (tires_per_bicycle : ℕ) (tires_per_tricycle : ℕ) (tires_per_pickup : ℕ) :
  cars = 15 →
  bicycles = 3 →
  tricycles = 1 →
  total_tires = 101 →
  tires_per_car = 4 →
  tires_per_bicycle = 2 →
  tires_per_tricycle = 3 →
  tires_per_pickup = 4 →
  ((total_tires - (cars * tires_per_car + bicycles * tires_per_bicycle + tricycles * tires_per_tricycle)) / tires_per_pickup) = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_pickup_trucks_l1379_137932


namespace NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1379_137968

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_multiple_of_17_l1379_137968


namespace NUMINAMATH_GPT_prime_if_and_only_if_digit_is_nine_l1379_137940

theorem prime_if_and_only_if_digit_is_nine (B : ℕ) (h : 0 ≤ B ∧ B < 10) :
  Prime (303200 + B) ↔ B = 9 := 
by
  sorry

end NUMINAMATH_GPT_prime_if_and_only_if_digit_is_nine_l1379_137940


namespace NUMINAMATH_GPT_sin_C_in_right_triangle_l1379_137959

theorem sin_C_in_right_triangle
  (A B C : ℝ)
  (sin_A : ℝ)
  (sin_B : ℝ)
  (B_right_angle : B = π / 2)
  (sin_A_value : sin_A = 3 / 5)
  (sin_B_value : sin_B = 1)
  (sin_of_C : ℝ)
  (tri_ABC : A + B + C = π ∧ A > 0 ∧ C > 0) :
    sin_of_C = 4 / 5 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_sin_C_in_right_triangle_l1379_137959


namespace NUMINAMATH_GPT_remainder_problem_l1379_137976

theorem remainder_problem (n m q1 q2 : ℤ) (h1 : n = 11 * q1 + 1) (h2 : m = 17 * q2 + 3) :
  ∃ r : ℤ, (r = (5 * n + 3 * m) % 11) ∧ (r = (7 * q2 + 3) % 11) :=
by
  sorry

end NUMINAMATH_GPT_remainder_problem_l1379_137976


namespace NUMINAMATH_GPT_even_numbers_average_18_l1379_137995

variable (n : ℕ)
variable (avg : ℕ)

theorem even_numbers_average_18 (h : avg = 18) : n = 17 := 
    sorry

end NUMINAMATH_GPT_even_numbers_average_18_l1379_137995


namespace NUMINAMATH_GPT_tangent_line_at_1_intervals_of_monotonicity_and_extrema_l1379_137957

open Real

noncomputable def f (x : ℝ) := 6 * log x + (1 / 2) * x^2 - 5 * x

theorem tangent_line_at_1 :
  let f' (x : ℝ) := (6 / x) + x - 5
  (f 1 = -9 / 2) →
  (f' 1 = 2) →
  (∀ x y : ℝ, y + 9 / 2 = 2 * (x - 1) → 4 * x - 2 * y - 13 = 0) := 
by
  sorry

theorem intervals_of_monotonicity_and_extrema :
  let f' (x : ℝ) := (x^2 - 5 * x + 6) / x
  (∀ x, 0 < x ∧ x < 2 → f' x > 0) → 
  (∀ x, 3 < x → f' x > 0) →
  (∀ x, 2 < x ∧ x < 3 → f' x < 0) →
  (f 2 = -8 + 6 * log 2) →
  (f 3 = -21 / 2 + 6 * log 3) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_1_intervals_of_monotonicity_and_extrema_l1379_137957


namespace NUMINAMATH_GPT_x_equals_eleven_l1379_137947

theorem x_equals_eleven (x : ℕ) 
  (h : (1 / 8) * 2^36 = 8^x) : x = 11 :=
sorry

end NUMINAMATH_GPT_x_equals_eleven_l1379_137947


namespace NUMINAMATH_GPT_original_number_increased_by_110_l1379_137966

-- Define the conditions and the proof statement without the solution steps
theorem original_number_increased_by_110 {x : ℝ} (h : x + 1.10 * x = 1680) : x = 800 :=
by 
  sorry

end NUMINAMATH_GPT_original_number_increased_by_110_l1379_137966


namespace NUMINAMATH_GPT_factorize_x4_plus_16_l1379_137905

theorem factorize_x4_plus_16 :
  ∀ x : ℝ, (x^4 + 16) = (x^2 - 2 * x + 2) * (x^2 + 2 * x + 2) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factorize_x4_plus_16_l1379_137905


namespace NUMINAMATH_GPT_radius_of_scrap_cookie_l1379_137943

theorem radius_of_scrap_cookie :
  ∀ (r : ℝ),
    (∃ (r_dough r_cookie : ℝ),
      r_dough = 6 ∧  -- Radius of the large dough
      r_cookie = 2 ∧  -- Radius of each cookie
      8 * (π * r_cookie^2) ≤ π * r_dough^2 ∧  -- Total area of cookies is less than or equal to area of large dough
      (π * r_dough^2) - (8 * (π * r_cookie^2)) = π * r^2  -- Area of scrap dough forms a circle of radius r
    ) → r = 2 := by
  sorry

end NUMINAMATH_GPT_radius_of_scrap_cookie_l1379_137943


namespace NUMINAMATH_GPT_simplify_expr_l1379_137900

variable (a b c : ℤ)

theorem simplify_expr :
  (15 * a + 45 * b + 20 * c) + (25 * a - 35 * b - 10 * c) - (10 * a + 55 * b + 30 * c) = 30 * a - 45 * b - 20 * c := 
by
  sorry

end NUMINAMATH_GPT_simplify_expr_l1379_137900


namespace NUMINAMATH_GPT_P2011_1_neg1_is_0_2_pow_1006_l1379_137970

def P1 (x y : ℤ) : ℤ × ℤ := (x + y, x - y)

def Pn : ℕ → ℤ → ℤ → ℤ × ℤ 
| 0, x, y => (x, y)
| (n + 1), x, y => P1 (Pn n x y).1 (Pn n x y).2

theorem P2011_1_neg1_is_0_2_pow_1006 : Pn 2011 1 (-1) = (0, 2^1006) := by
  sorry

end NUMINAMATH_GPT_P2011_1_neg1_is_0_2_pow_1006_l1379_137970


namespace NUMINAMATH_GPT_jerry_total_cost_correct_l1379_137911

theorem jerry_total_cost_correct :
  let bw_cost := 27
  let bw_discount := 0.1 * bw_cost
  let bw_discounted_price := bw_cost - bw_discount
  let color_cost := 32
  let color_discount := 0.05 * color_cost
  let color_discounted_price := color_cost - color_discount
  let total_color_discounted_price := 3 * color_discounted_price
  let total_discounted_price_before_tax := bw_discounted_price + total_color_discounted_price
  let tax_rate := 0.07
  let tax := total_discounted_price_before_tax * tax_rate
  let total_cost := total_discounted_price_before_tax + tax
  (Float.round (total_cost * 100) / 100) = 123.59 :=
sorry

end NUMINAMATH_GPT_jerry_total_cost_correct_l1379_137911


namespace NUMINAMATH_GPT_gcd_2pow_2025_minus_1_2pow_2016_minus_1_l1379_137965

theorem gcd_2pow_2025_minus_1_2pow_2016_minus_1 :
  Nat.gcd (2^2025 - 1) (2^2016 - 1) = 511 :=
by sorry

end NUMINAMATH_GPT_gcd_2pow_2025_minus_1_2pow_2016_minus_1_l1379_137965


namespace NUMINAMATH_GPT_water_added_l1379_137953

theorem water_added (x : ℝ) (salt_percent_initial : ℝ) (evaporation_fraction : ℝ) 
(salt_added : ℝ) (resulting_salt_percent : ℝ) 
(hx : x = 119.99999999999996) (h_initial_salt : salt_percent_initial = 0.20) 
(h_evap_fraction : evaporation_fraction = 1/4) (h_salt_added : salt_added = 16)
(h_resulting_salt_percent : resulting_salt_percent = 1/3) : 
∃ (water_added : ℝ), water_added = 30 :=
by
  sorry

end NUMINAMATH_GPT_water_added_l1379_137953


namespace NUMINAMATH_GPT_percentage_deficit_for_second_side_l1379_137993

theorem percentage_deficit_for_second_side
  (L W : ℝ) 
  (measured_first_side : ℝ := 1.12 * L) 
  (error_in_area : ℝ := 1.064) : 
  (∃ x : ℝ, (1.12 * L) * ((1 - 0.01 * x) * W) = 1.064 * (L * W) → x = 5) :=
by
  sorry

end NUMINAMATH_GPT_percentage_deficit_for_second_side_l1379_137993


namespace NUMINAMATH_GPT_min_value_of_expression_l1379_137980

open Real

theorem min_value_of_expression {a b c d e f : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
    (h_sum : a + b + c + d + e + f = 10) :
    (∃ x, x = 44.1 ∧ ∀ y, y = 1 / a + 4 / b + 9 / c + 16 / d + 25 / e + 36 / f → x ≤ y) :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l1379_137980


namespace NUMINAMATH_GPT_smallest_obtuse_triangles_l1379_137963

def obtuseTrianglesInTriangulation (n : Nat) : Nat :=
  if n < 3 then 0 else (n - 2) - 2

theorem smallest_obtuse_triangles (n : Nat) (h : n = 2003) :
  obtuseTrianglesInTriangulation n = 1999 := by
  sorry

end NUMINAMATH_GPT_smallest_obtuse_triangles_l1379_137963


namespace NUMINAMATH_GPT_square_side_length_l1379_137918

noncomputable def side_length_square_inscribed_in_hexagon : ℝ :=
  50 * Real.sqrt 3

theorem square_side_length (a b: ℝ) (h1 : a = 50) (h2 : b = 50 * (2 - Real.sqrt 3)) 
(s1 s2 s3 s4 s5 s6: ℝ) (ha : s1 = s2) (hb : s2 = s3) (hc : s3 = s4) 
(hd : s4 = s5) (he : s5 = s6) (hf : s6 = s1) : side_length_square_inscribed_in_hexagon = 50 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_l1379_137918


namespace NUMINAMATH_GPT_range_of_a_l1379_137985

noncomputable def equation_has_two_roots (a m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ + a * (2 * x₁ + 2 * m - 4 * Real.exp 1 * x₁) * (Real.log (x₁ + m) - Real.log x₁) = 0 ∧ 
    x₂ + a * (2 * x₂ + 2 * m - 4 * Real.exp 1 * x₂) * (Real.log (x₂ + m) - Real.log x₂) = 0

theorem range_of_a (m : ℝ) (hm : 0 < m) : 
  (∃ a, equation_has_two_roots a m) ↔ (a < 0 ∨ a > 1 / (2 * Real.exp 1)) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1379_137985


namespace NUMINAMATH_GPT_polygon_sides_l1379_137925

theorem polygon_sides (n : ℕ) :
  ((n - 2) * 180 = 3 * 360 - 180) →
  n = 7 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1379_137925
