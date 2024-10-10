import Mathlib

namespace great_eighteen_league_games_l4042_404247

/-- Calculates the number of games in a soccer league with specified structure -/
def soccer_league_games (divisions : Nat) (teams_per_division : Nat) 
  (intra_division_games : Nat) (inter_division_games : Nat) : Nat :=
  let intra_games := divisions * (teams_per_division.choose 2) * intra_division_games
  let inter_games := divisions.choose 2 * teams_per_division^2 * inter_division_games
  intra_games + inter_games

/-- The Great Eighteen Soccer League game count theorem -/
theorem great_eighteen_league_games : 
  soccer_league_games 3 6 3 2 = 351 := by
  sorry

end great_eighteen_league_games_l4042_404247


namespace equation_solution_l4042_404223

theorem equation_solution :
  ∃! x : ℚ, x ≠ -3 ∧ (x^2 + 4*x + 5) / (x + 3) = x + 6 :=
by
  use -13/5
  sorry

end equation_solution_l4042_404223


namespace train_bridge_crossing_time_l4042_404225

/-- Proves that a train with given length and speed takes the calculated time to cross a bridge of given length -/
theorem train_bridge_crossing_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (bridge_length : ℝ) : 
  train_length = 160 → 
  train_speed_kmh = 45 → 
  bridge_length = 215 → 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end train_bridge_crossing_time_l4042_404225


namespace solve_system_l4042_404215

theorem solve_system (x y : ℚ) 
  (eq1 : 3 * x - y = 7) 
  (eq2 : x + 3 * y = 2) : 
  x = 23 / 10 := by
sorry

end solve_system_l4042_404215


namespace geometric_sequence_sum_l4042_404246

def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℤ) (q : ℤ) :
  geometric_sequence a q ∧ a 1 = 1 ∧ q = -2 →
  a 1 + |a 2| + a 3 + |a 4| = 15 := by
  sorry

end geometric_sequence_sum_l4042_404246


namespace josie_checkout_wait_time_l4042_404204

def total_shopping_time : ℕ := 90
def cart_wait_time : ℕ := 3
def employee_wait_time : ℕ := 13
def stocker_wait_time : ℕ := 14
def shopping_time : ℕ := 42

theorem josie_checkout_wait_time :
  total_shopping_time - shopping_time - (cart_wait_time + employee_wait_time + stocker_wait_time) = 18 := by
  sorry

end josie_checkout_wait_time_l4042_404204


namespace plane_equation_l4042_404206

/-- Given a parametric equation of a plane, prove its Cartesian equation. -/
theorem plane_equation (s t : ℝ) :
  let u : ℝ × ℝ × ℝ := (2 + 2*s - 3*t, 1 + s, 4 - s + t)
  ∃ (A B C D : ℤ), A > 0 ∧ 
    (∀ (x y z : ℝ), (x, y, z) ∈ {p : ℝ × ℝ × ℝ | ∃ s t : ℝ, p = u} ↔ A*x + B*y + C*z + D = 0) ∧
    Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1 ∧
    A = 1 ∧ B = 4 ∧ C = 3 ∧ D = -18 := by
  sorry

end plane_equation_l4042_404206


namespace first_angle_is_55_l4042_404231

-- Define the triangle with the given conditions
def triangle (x : ℝ) : Prop :=
  let angle1 := x
  let angle2 := 2 * x
  let angle3 := x - 40
  (angle1 + angle2 + angle3 = 180) ∧ (angle1 > 0) ∧ (angle2 > 0) ∧ (angle3 > 0)

-- Theorem stating that the first angle is 55 degrees
theorem first_angle_is_55 : ∃ x, triangle x ∧ x = 55 := by
  sorry

end first_angle_is_55_l4042_404231


namespace ratio_of_sums_l4042_404271

theorem ratio_of_sums (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end ratio_of_sums_l4042_404271


namespace power_function_through_point_l4042_404272

/-- Given a power function that passes through the point (2, 8), prove its equation is x^3 -/
theorem power_function_through_point (n : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = x^n) → f 2 = 8 → (∀ x, f x = x^3) := by
  sorry

end power_function_through_point_l4042_404272


namespace sufficient_condition_for_quadratic_inequality_l4042_404264

theorem sufficient_condition_for_quadratic_inequality (a : ℝ) :
  (∀ x : ℝ, x > a → x^2 > 2*x) ∧
  (∃ x : ℝ, x^2 > 2*x ∧ x ≤ a) →
  a ≥ 2 :=
sorry

end sufficient_condition_for_quadratic_inequality_l4042_404264


namespace max_diff_slightly_unlucky_l4042_404220

/-- A natural number is slightly unlucky if the sum of its digits in decimal system is divisible by 13. -/
def SlightlyUnlucky (n : ℕ) : Prop :=
  (n.digits 10).sum % 13 = 0

/-- For any non-negative integer k, the intervals [100(k+1), 100(k+1)+39], [100k+60, 100k+99], and [100k+20, 100k+59] each contain at least one slightly unlucky number. -/
axiom slightly_unlucky_intervals (k : ℕ) :
  (∃ n : ℕ, SlightlyUnlucky n ∧ 100*(k+1) ≤ n ∧ n ≤ 100*(k+1)+39) ∧
  (∃ n : ℕ, SlightlyUnlucky n ∧ 100*k+60 ≤ n ∧ n ≤ 100*k+99) ∧
  (∃ n : ℕ, SlightlyUnlucky n ∧ 100*k+20 ≤ n ∧ n ≤ 100*k+59)

/-- The maximum difference between consecutive slightly unlucky numbers is 79. -/
theorem max_diff_slightly_unlucky :
  ∀ m n : ℕ, SlightlyUnlucky m → SlightlyUnlucky n → m < n →
  (∀ k : ℕ, SlightlyUnlucky k → m < k → k < n → False) →
  n - m ≤ 79 :=
sorry

end max_diff_slightly_unlucky_l4042_404220


namespace right_triangle_legs_l4042_404208

theorem right_triangle_legs (a b c h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a + b + c = 60 →   -- Perimeter condition
  h = 12 →           -- Altitude condition
  h = (a * b) / c →  -- Altitude formula
  (a = 15 ∧ b = 20) ∨ (a = 20 ∧ b = 15) :=
by sorry

end right_triangle_legs_l4042_404208


namespace move_down_two_units_l4042_404251

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Moving a point down in a Cartesian coordinate system -/
def moveDown (p : Point) (distance : ℝ) : Point :=
  ⟨p.x, p.y - distance⟩

/-- Theorem: Moving a point (a,b) down 2 units results in (a,b-2) -/
theorem move_down_two_units (a b : ℝ) :
  moveDown ⟨a, b⟩ 2 = ⟨a, b - 2⟩ := by
  sorry

end move_down_two_units_l4042_404251


namespace hyperbola_equation_l4042_404270

theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ (x y : ℝ), (x - 2)^2 + y^2 = 3 ∧ 
    (∃ (k : ℝ), b * x + a * y = k ∨ b * x - a * y = k)) →
  (a^2 = 1 ∧ b^2 = 3) :=
sorry

end hyperbola_equation_l4042_404270


namespace contrapositive_correct_l4042_404280

-- Define the proposition p
def p (passing_score : ℝ) (A_passes B_passes C_passes : Prop) : Prop :=
  (passing_score < 70) → (¬A_passes ∧ ¬B_passes ∧ ¬C_passes)

-- Define the contrapositive of p
def contrapositive_p (passing_score : ℝ) (A_passes B_passes C_passes : Prop) : Prop :=
  (A_passes ∨ B_passes ∨ C_passes) → (passing_score ≥ 70)

-- Theorem stating that contrapositive_p is indeed the contrapositive of p
theorem contrapositive_correct (passing_score : ℝ) (A_passes B_passes C_passes : Prop) :
  contrapositive_p passing_score A_passes B_passes C_passes ↔
  (¬p passing_score A_passes B_passes C_passes → False) → False :=
sorry

end contrapositive_correct_l4042_404280


namespace fifty_percent_of_2002_l4042_404202

theorem fifty_percent_of_2002 : (50 : ℚ) / 100 * 2002 = 1001 := by
  sorry

end fifty_percent_of_2002_l4042_404202


namespace value_of_expression_l4042_404257

theorem value_of_expression : 6 * 2017 - 2017 * 4 = 4034 := by
  sorry

end value_of_expression_l4042_404257


namespace product_square_theorem_l4042_404293

theorem product_square_theorem : (10 * 0.2 * 3 * 0.1)^2 = 9/25 := by
  sorry

end product_square_theorem_l4042_404293


namespace ice_cream_flavors_l4042_404262

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers,
    with at least one object in each container. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + k - 1) (k - 1)

/-- Theorem stating that there are 6 ways to distribute 5 scoops into 3 flavors
    with at least one scoop of each flavor. -/
theorem ice_cream_flavors :
  distribute_with_minimum 5 3 = 6 := by
  sorry

end ice_cream_flavors_l4042_404262


namespace factors_of_42_l4042_404282

/-- The number of positive factors of 42 -/
def number_of_factors_42 : ℕ :=
  (Finset.filter (· ∣ 42) (Finset.range 43)).card

/-- Theorem stating that the number of positive factors of 42 is 8 -/
theorem factors_of_42 : number_of_factors_42 = 8 := by
  sorry

end factors_of_42_l4042_404282


namespace set_equality_l4042_404227

-- Define the set A
def A : Set ℝ := {x : ℝ | 2 * x^2 + x - 3 = 0}

-- Define the set B
def B : Set ℝ := {i : ℝ | i^2 ≥ 4}

-- Define the complement of set C in real numbers
def compl_C : Set ℝ := {-1, 1, 3/2}

-- Theorem statement
theorem set_equality : A ∩ B ∪ compl_C = {-1, 1, 3/2} := by
  sorry

end set_equality_l4042_404227


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l4042_404226

-- 1
theorem problem_1 : 4 - (-28) + (-2) = 30 := by sorry

-- 2
theorem problem_2 : (-3) * ((-2/5) / (-1/4)) = -24/5 := by sorry

-- 3
theorem problem_3 : (-42) / (-7) - (-6) * 4 = 30 := by sorry

-- 4
theorem problem_4 : -3^2 / (-3)^2 + 3 * (-2) + |(-4)| = -3 := by sorry

-- 5
theorem problem_5 : (-24) * (3/4 - 5/6 + 7/12) = -12 := by sorry

-- 6
theorem problem_6 : -1^4 - (1 - 0.5) / (5/2) * (1/5) = -26/25 := by sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l4042_404226


namespace contest_result_l4042_404222

/-- The number of times Frannie jumped -/
def frannies_jumps : ℕ := 53

/-- The difference between Meg's and Frannie's jumps -/
def jump_difference : ℕ := 18

/-- Meg's number of jumps -/
def megs_jumps : ℕ := frannies_jumps + jump_difference

theorem contest_result : megs_jumps = 71 := by
  sorry

end contest_result_l4042_404222


namespace equation_solution_l4042_404240

theorem equation_solution : ∃ x : ℚ, (5*x + 9*x = 450 - 10*(x - 5)) ∧ x = 125/6 := by
  sorry

end equation_solution_l4042_404240


namespace race_time_differences_l4042_404253

def runner_A : ℕ := 60
def runner_B : ℕ := 100
def runner_C : ℕ := 80
def runner_D : ℕ := 120

def time_difference (t1 t2 : ℕ) : ℕ := 
  if t1 > t2 then t1 - t2 else t2 - t1

theorem race_time_differences : 
  (time_difference runner_A runner_B = 40) ∧
  (time_difference runner_A runner_C = 20) ∧
  (time_difference runner_A runner_D = 60) ∧
  (time_difference runner_B runner_C = 20) ∧
  (time_difference runner_B runner_D = 20) ∧
  (time_difference runner_C runner_D = 40) :=
by sorry

end race_time_differences_l4042_404253


namespace sons_age_l4042_404235

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
sorry

end sons_age_l4042_404235


namespace gold_coins_count_verify_conditions_l4042_404281

/-- The number of gold coins -/
def n : ℕ := 109

/-- The number of treasure chests -/
def c : ℕ := 13

/-- Theorem stating that the number of gold coins is 109 -/
theorem gold_coins_count : n = 109 :=
  by
  -- Condition 1: When putting 12 gold coins in each chest, 4 chests were left empty
  have h1 : n = 12 * (c - 4) := by sorry
  
  -- Condition 2: When putting 8 gold coins in each chest, 5 gold coins were left over
  have h2 : n = 8 * c + 5 := by sorry
  
  -- Prove that n equals 109
  sorry

/-- Theorem verifying the conditions -/
theorem verify_conditions :
  (n = 12 * (c - 4)) ∧ (n = 8 * c + 5) :=
  by sorry

end gold_coins_count_verify_conditions_l4042_404281


namespace rectangle_strip_problem_l4042_404258

theorem rectangle_strip_problem (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a * b + a * c + a * (b - a) + a * a + a * (c - a) = 43) :
  (a = 1 ∧ b + c = 22) ∨ (a = 22 ∧ b + c = 1) := by
  sorry

end rectangle_strip_problem_l4042_404258


namespace quadratic_function_theorem_l4042_404244

/-- A quadratic function satisfying certain conditions -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The function g defined in terms of f and m -/
def g (a b c m : ℝ) : ℝ → ℝ := fun x ↦ f a b c x + 2 * (1 - m) * x

/-- The theorem statement -/
theorem quadratic_function_theorem (a b c : ℝ) :
  (∀ x, f a b c x ≥ 0) →
  f a b c 0 = 1 →
  f a b c 1 = 0 →
  (∃ m, (∀ x ∈ Set.Icc (-2 : ℝ) 5, g a b c m x ≤ 13) ∧
        (∃ x ∈ Set.Icc (-2 : ℝ) 5, g a b c m x = 13)) →
  ((a = 1 ∧ b = -2 ∧ c = 1) ∧ (m = 13/10 ∨ m = 2)) :=
by sorry

end quadratic_function_theorem_l4042_404244


namespace floor_expression_l4042_404299

theorem floor_expression (n : ℕ) (hn : n = 101) : 
  ⌊(8 * (n^2 + 1) : ℚ) / (n^2 - 1)⌋ = 8 := by
  sorry

end floor_expression_l4042_404299


namespace recipe_butter_amount_l4042_404210

/-- The amount of butter (in ounces) required per cup of baking mix -/
def butter_per_cup : ℚ := 4/3

/-- The number of cups of baking mix the chef planned to use -/
def planned_cups : ℕ := 6

/-- The amount of coconut oil (in ounces) the chef used as a substitute for butter -/
def coconut_oil_used : ℕ := 8

/-- Theorem stating that the recipe calls for 4/3 ounces of butter per cup of baking mix -/
theorem recipe_butter_amount :
  butter_per_cup * planned_cups = coconut_oil_used := by
  sorry

end recipe_butter_amount_l4042_404210


namespace range_of_values_l4042_404238

theorem range_of_values (x y : ℝ) 
  (hx : 30 < x ∧ x < 42) 
  (hy : 16 < y ∧ y < 24) : 
  (46 < x + y ∧ x + y < 66) ∧ 
  (-18 < x - 2*y ∧ x - 2*y < 10) ∧ 
  (5/4 < x/y ∧ x/y < 21/8) := by
sorry

end range_of_values_l4042_404238


namespace men_in_first_group_l4042_404248

/-- Represents the daily work done by a boy -/
def boy_work : ℝ := 1

/-- Represents the daily work done by a man -/
def man_work : ℝ := 2 * boy_work

/-- The number of days taken by the first group to complete the work -/
def days_group1 : ℕ := 5

/-- The number of days taken by the second group to complete the work -/
def days_group2 : ℕ := 4

/-- The number of boys in the first group -/
def boys_group1 : ℕ := 16

/-- The number of men in the second group -/
def men_group2 : ℕ := 13

/-- The number of boys in the second group -/
def boys_group2 : ℕ := 24

/-- The theorem stating that the number of men in the first group is 12 -/
theorem men_in_first_group :
  ∃ (m : ℕ), 
    (days_group1 : ℝ) * (m * man_work + boys_group1 * boy_work) = 
    (days_group2 : ℝ) * (men_group2 * man_work + boys_group2 * boy_work) ∧
    m = 12 := by
  sorry

end men_in_first_group_l4042_404248


namespace darius_bucket_count_l4042_404288

/-- Represents the number of ounces in each of Darius's water buckets -/
def water_buckets : List ℕ := [11, 13, 12, 16, 10]

/-- The total amount of water in the first large bucket -/
def first_large_bucket : ℕ := 23

/-- The total amount of water in the second large bucket -/
def second_large_bucket : ℕ := 39

theorem darius_bucket_count :
  ∃ (bucket : ℕ) (remaining : List ℕ),
    bucket ∈ water_buckets ∧
    remaining = water_buckets.filter (λ x => x ≠ bucket ∧ x ≠ 10) ∧
    bucket + 10 = first_large_bucket ∧
    remaining.sum = second_large_bucket ∧
    water_buckets.length = 5 := by
  sorry

end darius_bucket_count_l4042_404288


namespace max_value_sum_l4042_404266

theorem max_value_sum (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (h_sum : a^2 + b^2 + c^2 + d^2 + e^2 = 504) :
  ∃ (N a_N b_N c_N d_N e_N : ℝ),
    (∀ x y z w v : ℝ, x > 0 → y > 0 → z > 0 → w > 0 → v > 0 → 
      x^2 + y^2 + z^2 + w^2 + v^2 = 504 → 
      x*z + 3*y*z + 4*z*w + 8*z*v ≤ N) ∧
    (a_N*c_N + 3*b_N*c_N + 4*c_N*d_N + 8*c_N*e_N = N) ∧
    (a_N^2 + b_N^2 + c_N^2 + d_N^2 + e_N^2 = 504) ∧
    (N + a_N + b_N + c_N + d_N + e_N = 32 + 756 * Real.sqrt 10 + 6 * Real.sqrt 7) :=
by sorry

end max_value_sum_l4042_404266


namespace acid_dilution_l4042_404295

theorem acid_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (final_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.4 →
  final_concentration = 0.25 →
  water_added = 30 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

end acid_dilution_l4042_404295


namespace max_p_value_l4042_404233

-- Define the equation
def equation (x p : ℝ) : Prop :=
  2 * Real.cos (2 * Real.pi - Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2)) - 3 =
  p - 2 * Real.sin (-Real.pi * x^2 / 6) * Real.cos (Real.pi / 3 * Real.sqrt (9 - x^2))

-- Define the theorem
theorem max_p_value :
  ∃ (p_max : ℝ), p_max = -2 ∧
  (∀ p : ℝ, (∃ x : ℝ, equation x p) → p ≤ p_max) ∧
  (∃ x : ℝ, equation x p_max) :=
sorry

end max_p_value_l4042_404233


namespace conditions_necessary_not_sufficient_l4042_404292

theorem conditions_necessary_not_sufficient :
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → |x| ≤ 1 ∧ |y| ≤ 1) ∧
  ¬(∀ x y : ℝ, |x| ≤ 1 ∧ |y| ≤ 1 → x^2 + y^2 ≤ 1) :=
by sorry

end conditions_necessary_not_sufficient_l4042_404292


namespace marble_product_l4042_404224

theorem marble_product (red blue : ℕ) : 
  (red - blue = 12) →
  (red + blue = red - blue + 40) →
  red * blue = 640 := by
sorry

end marble_product_l4042_404224


namespace solution_exists_l4042_404267

theorem solution_exists (R₀ : ℝ) : ∃ x₁ x₂ x₃ : ℤ,
  x₁ > ⌊R₀⌋ ∧ x₂ > ⌊R₀⌋ ∧ x₃ > ⌊R₀⌋ ∧ x₁^2 + x₂^2 + x₃^2 = x₁ * x₂ * x₃ := by
  sorry

end solution_exists_l4042_404267


namespace root_conditions_l4042_404219

theorem root_conditions (a b c : ℝ) : 
  (∀ x : ℝ, x^5 + 2*x^4 + a*x^2 + b*x = c ↔ x = -1 ∨ x = 1) ↔ 
  (a = -6 ∧ b = -1 ∧ c = -4) :=
sorry

end root_conditions_l4042_404219


namespace inequality_solution_implies_k_value_l4042_404290

theorem inequality_solution_implies_k_value (k : ℚ) :
  (∀ x : ℚ, 3 * x - (2 * k - 3) < 4 * x + 3 * k + 6 ↔ x > 1) →
  k = -4/5 := by
sorry

end inequality_solution_implies_k_value_l4042_404290


namespace jason_potato_eating_time_l4042_404287

/-- Given that Jason eats 3 potatoes in 20 minutes, prove that it takes him 3 hours to eat 27 potatoes. -/
theorem jason_potato_eating_time :
  let potatoes_per_20_min : ℚ := 3
  let total_potatoes : ℚ := 27
  let minutes_per_session : ℚ := 20
  let hours_to_eat_all : ℚ := (total_potatoes / potatoes_per_20_min) * (minutes_per_session / 60)
  hours_to_eat_all = 3 := by
sorry

end jason_potato_eating_time_l4042_404287


namespace stripe_area_on_cylinder_l4042_404241

/-- The area of a stripe wrapped around a cylinder -/
theorem stripe_area_on_cylinder 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℝ) 
  (h1 : diameter = 20) 
  (h2 : stripe_width = 2) 
  (h3 : revolutions = 3) : 
  stripe_width * revolutions * (π * diameter) = 240 * π := by
  sorry

end stripe_area_on_cylinder_l4042_404241


namespace interest_rates_equality_l4042_404239

theorem interest_rates_equality (initial_savings : ℝ) 
  (simple_interest : ℝ) (compound_interest : ℝ) : 
  initial_savings = 1000 ∧ 
  simple_interest = 100 ∧ 
  compound_interest = 105 →
  ∃ (r : ℝ), 
    simple_interest = (initial_savings / 2) * r * 2 ∧
    compound_interest = (initial_savings / 2) * ((1 + r)^2 - 1) :=
by sorry

end interest_rates_equality_l4042_404239


namespace quadrupled_base_exponent_l4042_404283

theorem quadrupled_base_exponent (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) :
  (4 * c)^(4 * d) = (c^d * y^d)^2 → y = 16 * c := by
  sorry

end quadrupled_base_exponent_l4042_404283


namespace complex_on_imaginary_axis_l4042_404211

theorem complex_on_imaginary_axis (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) * (2 * a - Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → z = -2 * Complex.I :=
by
  sorry

end complex_on_imaginary_axis_l4042_404211


namespace gcd_79625_51575_l4042_404207

theorem gcd_79625_51575 : Nat.gcd 79625 51575 = 25 := by
  sorry

end gcd_79625_51575_l4042_404207


namespace sam_distance_l4042_404274

/-- Given that Harvey runs 8 miles more than Sam and their total distance is 32 miles,
    prove that Sam runs 12 miles. -/
theorem sam_distance (sam : ℝ) (harvey : ℝ) : 
  harvey = sam + 8 → sam + harvey = 32 → sam = 12 := by
  sorry

end sam_distance_l4042_404274


namespace lcm_36_45_l4042_404285

theorem lcm_36_45 : Nat.lcm 36 45 = 180 := by
  sorry

end lcm_36_45_l4042_404285


namespace second_week_cut_percentage_sculpture_problem_l4042_404261

/-- Calculates the percentage of marble cut away in the second week of sculpting -/
theorem second_week_cut_percentage (initial_weight : ℝ) (first_week_cut : ℝ) 
  (third_week_cut : ℝ) (final_weight : ℝ) : ℝ :=
  let remaining_after_first := initial_weight * (1 - first_week_cut / 100)
  let second_week_cut := 100 * (1 - (final_weight / (remaining_after_first * (1 - third_week_cut / 100))))
  second_week_cut

/-- The percentage of marble cut away in the second week is 30% -/
theorem sculpture_problem :
  second_week_cut_percentage 300 30 15 124.95 = 30 := by
  sorry

end second_week_cut_percentage_sculpture_problem_l4042_404261


namespace fox_initial_coins_l4042_404228

/-- The number of times Fox crosses the bridge -/
def num_crossings : ℕ := 3

/-- The toll Fox pays after each crossing -/
def toll : ℚ := 50

/-- The final amount Fox wants to have -/
def final_amount : ℚ := 50

/-- The factor by which Fox's money is multiplied each crossing -/
def multiplier : ℚ := 3

theorem fox_initial_coins (x : ℚ) :
  (((x * multiplier - toll) * multiplier - toll) * multiplier - toll = final_amount) →
  (x = 700 / 27) :=
by sorry

end fox_initial_coins_l4042_404228


namespace wrong_height_calculation_l4042_404245

theorem wrong_height_calculation (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (actual_avg : ℝ) :
  n = 35 ∧ initial_avg = 184 ∧ actual_height = 106 ∧ actual_avg = 182 →
  ∃ wrong_height : ℝ, wrong_height = 176 ∧
    n * actual_avg = n * initial_avg - wrong_height + actual_height :=
by sorry

end wrong_height_calculation_l4042_404245


namespace trig_identity_l4042_404276

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 := by
  sorry

end trig_identity_l4042_404276


namespace probability_theorem_l4042_404230

/-- Represents the number of letters in each name -/
def letters_per_name : ℕ := 5

/-- Represents the total number of cards -/
def total_cards : ℕ := 15

/-- Represents the number of cards selected -/
def cards_selected : ℕ := 3

/-- Represents the number of different ways to select one letter from each name -/
def selection_arrangements : ℕ := 6

/-- Calculates the probability of selecting one letter from each of three names -/
def probability_one_from_each : ℚ :=
  selection_arrangements * (letters_per_name : ℚ) / total_cards *
  (letters_per_name : ℚ) / (total_cards - 1) *
  (letters_per_name : ℚ) / (total_cards - 2)

theorem probability_theorem :
  probability_one_from_each = 125 / 455 :=
sorry

end probability_theorem_l4042_404230


namespace intersection_of_A_and_B_l4042_404263

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 - x - 2 = 0}

theorem intersection_of_A_and_B : A ∩ B = {2} := by sorry

end intersection_of_A_and_B_l4042_404263


namespace quadratic_real_root_l4042_404234

theorem quadratic_real_root (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
  sorry

end quadratic_real_root_l4042_404234


namespace polynomial_expansion_theorem_l4042_404286

theorem polynomial_expansion_theorem (N : ℕ) : 
  (Nat.choose N 5 = 2002) ↔ (N = 17) := by sorry

#check polynomial_expansion_theorem

end polynomial_expansion_theorem_l4042_404286


namespace sum_of_possible_m_values_l4042_404269

theorem sum_of_possible_m_values (p q r m : ℂ) : 
  p ≠ q ∧ q ≠ r ∧ r ≠ p →
  p / (1 - q) = m ∧ q / (1 - r) = m ∧ r / (1 - p) = m →
  ∃ (m₁ m₂ m₃ : ℂ), 
    (m₁ = 0 ∨ m₁ = (1 + Complex.I * Real.sqrt 3) / 2 ∨ m₁ = (1 - Complex.I * Real.sqrt 3) / 2) ∧
    (m₂ = 0 ∨ m₂ = (1 + Complex.I * Real.sqrt 3) / 2 ∨ m₂ = (1 - Complex.I * Real.sqrt 3) / 2) ∧
    (m₃ = 0 ∨ m₃ = (1 + Complex.I * Real.sqrt 3) / 2 ∨ m₃ = (1 - Complex.I * Real.sqrt 3) / 2) ∧
    m₁ + m₂ + m₃ = 1 :=
by sorry

end sum_of_possible_m_values_l4042_404269


namespace three_consecutive_heads_sequences_l4042_404278

def coin_flip_sequence (n : ℕ) : ℕ := 2^n

def no_three_consecutive_heads : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | n + 3 => no_three_consecutive_heads (n + 2) + no_three_consecutive_heads (n + 1) + no_three_consecutive_heads n

theorem three_consecutive_heads_sequences (n : ℕ) (h : n = 10) :
  coin_flip_sequence n - no_three_consecutive_heads n = 520 := by
  sorry

end three_consecutive_heads_sequences_l4042_404278


namespace arithmetic_sequence_sum_l4042_404279

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 3 + a 8 = 10 → 3 * a 5 + a 7 = 20 := by
sorry

end arithmetic_sequence_sum_l4042_404279


namespace axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l4042_404260

/-- The axis of symmetry of a parabola y = ax^2 + bx + c is the line x = -b/(2a) -/
theorem axis_of_symmetry_parabola (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  ∀ x, f (x + (-b / (2 * a))) = f (-b / (2 * a) - x) :=
sorry

/-- The axis of symmetry of the parabola y = -x^2 + 2022 is the line x = 0 -/
theorem axis_of_symmetry_specific_parabola :
  let f : ℝ → ℝ := λ x ↦ -x^2 + 2022
  ∀ x, f (x + 0) = f (0 - x) :=
sorry

end axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l4042_404260


namespace sum_of_x_intercepts_is_14_l4042_404294

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

-- Define the transformed parabola
def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 1

-- Theorem statement
theorem sum_of_x_intercepts_is_14 :
  ∃ a b : ℝ, 
    transformed_parabola a = 0 ∧ 
    transformed_parabola b = 0 ∧ 
    a + b = 14 :=
sorry

end sum_of_x_intercepts_is_14_l4042_404294


namespace polynomial_simplification_l4042_404296

theorem polynomial_simplification (x : ℝ) :
  (15 * x^13 + 10 * x^12 + 7 * x^11) + (3 * x^15 + 2 * x^13 + x^11 + 4 * x^9 + 2 * x^5 + 6) =
  3 * x^15 + 17 * x^13 + 10 * x^12 + 8 * x^11 + 4 * x^9 + 2 * x^5 + 6 := by
  sorry

end polynomial_simplification_l4042_404296


namespace max_salary_cricket_team_l4042_404214

/-- Represents a cricket team -/
structure CricketTeam where
  players : ℕ
  minSalary : ℕ
  salaryCap : ℕ

/-- Calculates the maximum possible salary for the highest-paid player in a cricket team -/
def maxSalary (team : CricketTeam) : ℕ :=
  team.salaryCap - (team.players - 1) * team.minSalary

/-- Theorem: The maximum possible salary for the highest-paid player in the given cricket team is 416000 -/
theorem max_salary_cricket_team :
  ∃ (team : CricketTeam),
    team.players = 18 ∧
    team.minSalary = 12000 ∧
    team.salaryCap = 620000 ∧
    maxSalary team = 416000 := by
  sorry

end max_salary_cricket_team_l4042_404214


namespace sum_of_cubes_of_five_l4042_404212

theorem sum_of_cubes_of_five : 5^3 + 5^3 + 5^3 + 5^3 = 500 := by
  sorry

end sum_of_cubes_of_five_l4042_404212


namespace error_arrangement_probability_l4042_404201

/-- The number of letters in the word "error" -/
def word_length : Nat := 5

/-- The number of 'r's in the word "error" -/
def num_r : Nat := 3

/-- The number of ways to arrange the letters in "error" -/
def total_arrangements : Nat := 20

/-- The probability of incorrectly arranging the letters in "error" -/
def incorrect_probability : Rat := 19 / 20

/-- Theorem stating that the probability of incorrectly arranging the letters in "error" is 19/20 -/
theorem error_arrangement_probability :
  incorrect_probability = 19 / 20 :=
by sorry

end error_arrangement_probability_l4042_404201


namespace arithmetic_sequence_eighth_term_l4042_404297

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℚ)
  (h_arith : arithmetic_sequence a)
  (h_sum : sum_arithmetic a 15 = 0) :
  a 8 = 6 := by
  sorry

end arithmetic_sequence_eighth_term_l4042_404297


namespace game_result_l4042_404243

theorem game_result (a : ℝ) : ((2 * a + 6) / 2) - a = 3 := by
  sorry

end game_result_l4042_404243


namespace equal_roots_quadratic_l4042_404259

/-- 
If the quadratic equation x^2 + 6x + c = 0 has two equal real roots,
then c = 9.
-/
theorem equal_roots_quadratic (c : ℝ) : 
  (∃ x : ℝ, x^2 + 6*x + c = 0 ∧ 
   ∀ y : ℝ, y^2 + 6*y + c = 0 → y = x) → 
  c = 9 :=
by sorry

end equal_roots_quadratic_l4042_404259


namespace sons_age_l4042_404249

/-- Prove that the son's current age is 24 years given the conditions -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = 3 * son_age →
  father_age - 8 = 4 * (son_age - 8) →
  son_age = 24 := by
sorry

end sons_age_l4042_404249


namespace non_zero_coeffs_bound_l4042_404221

/-- A polynomial is non-zero if it has at least one non-zero coefficient -/
def NonZeroPoly (p : Polynomial ℝ) : Prop :=
  ∃ (i : ℕ), p.coeff i ≠ 0

/-- The number of non-zero coefficients in a polynomial -/
def NumNonZeroCoeffs (p : Polynomial ℝ) : ℕ :=
  (p.support).card

/-- The statement to be proved -/
theorem non_zero_coeffs_bound (Q : Polynomial ℝ) (n : ℕ) 
  (hQ : NonZeroPoly Q) (hn : n > 0) : 
  NumNonZeroCoeffs ((X - 1)^n * Q) ≥ n + 1 :=
sorry

end non_zero_coeffs_bound_l4042_404221


namespace smallest_square_sum_20_consecutive_l4042_404203

/-- The sum of an arithmetic sequence of 20 terms -/
def sum_20_consecutive (first : ℕ) : ℕ :=
  20 * (2 * first + 19) / 2

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem smallest_square_sum_20_consecutive :
  (∀ n : ℕ, n < 490 → ¬(is_perfect_square n ∧ ∃ k : ℕ, sum_20_consecutive k = n)) ∧
  (is_perfect_square 490 ∧ ∃ k : ℕ, sum_20_consecutive k = 490) :=
sorry

end smallest_square_sum_20_consecutive_l4042_404203


namespace min_area_triangle_min_area_is_minimum_l4042_404236

/-- The minimum area of a triangle with vertices (0,0), (30,18), and a third point with integer coordinates -/
theorem min_area_triangle : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (30, 18)
  3

/-- The area of the triangle is indeed the minimum possible -/
theorem min_area_is_minimum (p q : ℤ) : 
  let C : ℝ × ℝ := (p, q)
  let area := (1/2 : ℝ) * |18 * p - 30 * q|
  3 ≤ area := by
  sorry

#check min_area_triangle
#check min_area_is_minimum

end min_area_triangle_min_area_is_minimum_l4042_404236


namespace min_value_expression_l4042_404217

theorem min_value_expression (x y : ℝ) :
  x^2 + y^2 - 8*x + 6*y + 25 ≥ 0 ∧
  ∃ (a b : ℝ), a^2 + b^2 - 8*a + 6*b + 25 = 0 := by
  sorry

end min_value_expression_l4042_404217


namespace quadratic_one_solution_sum_l4042_404229

theorem quadratic_one_solution_sum (b₁ b₂ : ℝ) : 
  (∀ x, 3 * x^2 + b₁ * x + 12 * x + 16 = 0 → (b₁ + 12)^2 = 4 * 3 * 16) ∧
  (∀ x, 3 * x^2 + b₂ * x + 12 * x + 16 = 0 → (b₂ + 12)^2 = 4 * 3 * 16) →
  b₁ + b₂ = -24 := by
sorry

end quadratic_one_solution_sum_l4042_404229


namespace tire_cost_calculation_l4042_404200

theorem tire_cost_calculation (total_cost : ℕ) (num_tires : ℕ) (h1 : total_cost = 240) (h2 : num_tires = 4) :
  total_cost / num_tires = 60 := by
  sorry

end tire_cost_calculation_l4042_404200


namespace photo_arrangement_count_l4042_404277

/-- The number of ways to arrange 2 teachers and 4 students in a row -/
def arrangementCount (n : ℕ) (m : ℕ) (k : ℕ) : ℕ :=
  if n = 2 ∧ m = 4 ∧ k = 1 then
    Nat.factorial 2 * 2 * Nat.factorial 3
  else
    0

/-- Theorem stating the correct number of arrangements -/
theorem photo_arrangement_count :
  arrangementCount 2 4 1 = 24 :=
by sorry

end photo_arrangement_count_l4042_404277


namespace smallest_n_with_conditions_l4042_404275

theorem smallest_n_with_conditions : ∃ (m a : ℕ),
  145^2 = m^3 - (m-1)^3 + 5 ∧
  2*145 + 117 = a^2 ∧
  ∀ (n : ℕ), n > 0 → n < 145 →
    (∀ (m' a' : ℕ), n^2 ≠ m'^3 - (m'-1)^3 + 5 ∨ 2*n + 117 ≠ a'^2) :=
by sorry

end smallest_n_with_conditions_l4042_404275


namespace woman_work_days_value_l4042_404216

/-- The number of days it takes for a woman to complete the work -/
def woman_work_days (total_members : ℕ) (man_work_days : ℕ) (combined_work_days : ℕ) (num_women : ℕ) : ℚ :=
  let num_men := total_members - num_women
  let man_work_rate := 1 / man_work_days
  let total_man_work := (combined_work_days / 2 : ℚ) * man_work_rate * num_men
  let total_woman_work := 1 - total_man_work
  let woman_work_rate := (total_woman_work * 3) / (combined_work_days * num_women)
  1 / woman_work_rate

/-- Theorem stating the number of days it takes for a woman to complete the work -/
theorem woman_work_days_value :
  woman_work_days 15 120 17 3 = 5100 / 83 :=
by sorry

end woman_work_days_value_l4042_404216


namespace vacation_tents_l4042_404256

/-- Given a total number of people, house capacity, and tent capacity, 
    calculate the minimum number of tents needed. -/
def tents_needed (total_people : ℕ) (house_capacity : ℕ) (tent_capacity : ℕ) : ℕ :=
  ((total_people - house_capacity + tent_capacity - 1) / tent_capacity)

/-- Theorem stating that for 13 people, a house capacity of 4, and tents that sleep 2 each, 
    the minimum number of tents needed is 5. -/
theorem vacation_tents : tents_needed 13 4 2 = 5 := by
  sorry

end vacation_tents_l4042_404256


namespace A_intersect_B_empty_l4042_404250

-- Define set A
def A : Set ℝ := {0, 1, 2}

-- Define set B
def B : Set ℝ := {x : ℝ | (x + 1) * (x + 2) < 0}

-- Theorem statement
theorem A_intersect_B_empty : A ∩ B = ∅ := by
  sorry

end A_intersect_B_empty_l4042_404250


namespace right_triangle_sets_l4042_404255

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  ¬(is_right_triangle 5 7 10) ∧
  (is_right_triangle 3 4 5) ∧
  (is_right_triangle 6 8 10) ∧
  (is_right_triangle 1 2 (Real.sqrt 3)) :=
by sorry

end right_triangle_sets_l4042_404255


namespace profit_sharing_l4042_404273

/-- The profit sharing problem -/
theorem profit_sharing
  (invest_a invest_b invest_c : ℝ)
  (total_profit : ℝ)
  (h1 : invest_a = 3 * invest_b)
  (h2 : invest_a = 2 / 3 * invest_c)
  (h3 : total_profit = 12375) :
  (invest_c / (invest_a + invest_b + invest_c)) * total_profit = (9 / 17) * 12375 := by
sorry

#eval (9 / 17 : ℚ) * 12375

end profit_sharing_l4042_404273


namespace no_four_digit_square_palindromes_l4042_404284

/-- A function that checks if a natural number is a 4-digit number -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that checks if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that checks if a natural number is a palindrome -/
def is_palindrome (n : ℕ) : Prop := 
  let digits := n.digits 10
  digits = digits.reverse

/-- Theorem stating that there are no 4-digit square numbers that are palindromes -/
theorem no_four_digit_square_palindromes : 
  ¬∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
sorry

end no_four_digit_square_palindromes_l4042_404284


namespace equation_solution_l4042_404218

theorem equation_solution : 
  {x : ℝ | x^6 + (3 - x)^6 = 730} = {1.5 + Real.sqrt 5, 1.5 - Real.sqrt 5} :=
by sorry

end equation_solution_l4042_404218


namespace tangent_line_condition_max_ab_value_l4042_404289

noncomputable section

/-- The function f(x) = ln(ax + b) + x^2 -/
def f (a b x : ℝ) : ℝ := Real.log (a * x + b) + x^2

/-- The derivative of f with respect to x -/
def f_deriv (a b x : ℝ) : ℝ := a / (a * x + b) + 2 * x

theorem tangent_line_condition (a b : ℝ) (h1 : a ≠ 0) :
  (f_deriv a b 1 = 1 ∧ f a b 1 = 1) → (a = -1 ∧ b = 2) :=
sorry

theorem max_ab_value (a b : ℝ) (h1 : a ≠ 0) :
  (∀ x, f a b x ≤ x^2 + x) → (a * b ≤ Real.exp 1 / 2) :=
sorry

end tangent_line_condition_max_ab_value_l4042_404289


namespace expression_evaluation_l4042_404213

theorem expression_evaluation :
  let a : ℤ := -1
  let b : ℤ := 3
  2 * a * b^2 - (3 * a^2 * b - 2 * (3 * a^2 * b - a * b^2 - 1)) = 7 := by
  sorry

end expression_evaluation_l4042_404213


namespace arithmetic_progression_difference_divisibility_l4042_404291

theorem arithmetic_progression_difference_divisibility
  (p : ℕ) (a : ℕ → ℕ) (d : ℕ) 
  (h_p_prime : Nat.Prime p)
  (h_a_prime : ∀ i, i ∈ Finset.range p → Nat.Prime (a i))
  (h_arithmetic_progression : ∀ i, i ∈ Finset.range (p - 1) → a (i + 1) = a i + d)
  (h_increasing : ∀ i j, i < j → j < p → a i < a j)
  (h_a1_gt_p : a 0 > p) :
  p ∣ d :=
sorry

end arithmetic_progression_difference_divisibility_l4042_404291


namespace fifteen_factorial_base_eight_zeroes_l4042_404265

/-- The number of trailing zeroes in n! when written in base b --/
def trailingZeroes (n : ℕ) (b : ℕ) : ℕ :=
  sorry

/-- The factorial function --/
def factorial (n : ℕ) : ℕ :=
  sorry

theorem fifteen_factorial_base_eight_zeroes :
  trailingZeroes (factorial 15) 8 = 3 :=
sorry

end fifteen_factorial_base_eight_zeroes_l4042_404265


namespace james_stickers_l4042_404242

theorem james_stickers (initial_stickers new_stickers total_stickers : ℕ) 
  (h1 : new_stickers = 22)
  (h2 : total_stickers = 61)
  (h3 : total_stickers = initial_stickers + new_stickers) : 
  initial_stickers = 39 := by
  sorry

end james_stickers_l4042_404242


namespace hyperbola_eccentricity_is_two_l4042_404254

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : a > 0
  b_pos : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola a b) : ℝ := sorry

/-- The right focus of a hyperbola -/
def right_focus (h : Hyperbola a b) : ℝ × ℝ := sorry

/-- An asymptote of a hyperbola -/
def asymptote (h : Hyperbola a b) : ℝ → ℝ := sorry

/-- The symmetric point of a point with respect to a line -/
def symmetric_point (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ × ℝ := sorry

/-- Theorem: The eccentricity of a hyperbola is 2 given the specified conditions -/
theorem hyperbola_eccentricity_is_two (a b : ℝ) (h : Hyperbola a b) :
  let l₁ := asymptote h
  let l₂ := fun x => -l₁ x
  let f := right_focus h
  let s := symmetric_point f l₁
  s.2 = l₂ s.1 →
  eccentricity h = 2 := by sorry

end hyperbola_eccentricity_is_two_l4042_404254


namespace decagon_triangles_l4042_404205

/-- A regular decagon is a polygon with 10 vertices -/
def RegularDecagon : ℕ := 10

/-- No three vertices of a regular decagon are collinear -/
axiom decagon_vertices_not_collinear : True

/-- The number of triangles formed from vertices of a regular decagon -/
def num_triangles_from_decagon : ℕ := Nat.choose RegularDecagon 3

theorem decagon_triangles :
  num_triangles_from_decagon = 120 := by
  sorry

end decagon_triangles_l4042_404205


namespace subtract_from_twenty_l4042_404252

theorem subtract_from_twenty (x : ℤ) (h : x + 40 = 52) : 20 - x = 8 := by
  sorry

end subtract_from_twenty_l4042_404252


namespace modular_arithmetic_problem_l4042_404268

theorem modular_arithmetic_problem : ((367 * 373 * 379 % 53) * 383) % 47 = 0 := by
  sorry

end modular_arithmetic_problem_l4042_404268


namespace cycling_speed_l4042_404298

/-- The speed of Alice and Bob when cycling under specific conditions -/
theorem cycling_speed : ∃ (x : ℝ),
  (x^2 - 5*x - 14 = (x^2 + x - 20) / (x - 4)) ∧
  (x^2 - 5*x - 14 = 8 + 2*Real.sqrt 7) := by
  sorry

end cycling_speed_l4042_404298


namespace ellipse_equation_l4042_404237

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let C : Set (ℝ × ℝ) := {(x, y) | x^2 / a^2 + y^2 / b^2 = 1}
  let e : ℝ := 1/3
  let A₁ : ℝ × ℝ := (-a, 0)
  let A₂ : ℝ × ℝ := (a, 0)
  let B : ℝ × ℝ := (0, b)
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / a^2 + y^2 / b^2 = 1) →
  e = Real.sqrt (1 - b^2 / a^2) →
  ((B.1 - A₁.1) * (B.1 - A₂.1) + (B.2 - A₁.2) * (B.2 - A₂.2) = -1) →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 / 9 + y^2 / 8 = 1) := by
sorry

end ellipse_equation_l4042_404237


namespace product_ratio_equality_l4042_404209

theorem product_ratio_equality (a b c d e f : ℝ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 1 := by
  sorry

end product_ratio_equality_l4042_404209


namespace calculation_proof_l4042_404232

theorem calculation_proof : (-1/2)⁻¹ - 4 * Real.cos (30 * π / 180) - (π + 2013)^0 + Real.sqrt 12 = -3 := by
  sorry

end calculation_proof_l4042_404232
