import Mathlib

namespace instantaneous_velocity_at_3_seconds_l2501_250118

-- Define the displacement function
def s (t : ℝ) : ℝ := 4 - 2*t + t^2

-- Define the velocity function (derivative of s)
def v (t : ℝ) : ℝ := 2*t - 2

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 4 := by sorry

end instantaneous_velocity_at_3_seconds_l2501_250118


namespace chord_length_l2501_250134

-- Define the line L: 3x + 4y - 5 = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 5 = 0}

-- Define the circle C: x^2 + y^2 = 4
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- State the theorem
theorem chord_length : 
  A ∈ L ∧ A ∈ C ∧ B ∈ L ∧ B ∈ C → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end chord_length_l2501_250134


namespace existence_of_cube_triplet_l2501_250153

theorem existence_of_cube_triplet :
  ∃ n₀ : ℕ, ∀ m : ℕ, m ≥ n₀ →
    ∃ a b c : ℕ+,
      (m ^ 3 : ℝ) < (a : ℝ) ∧
      (a : ℝ) < (b : ℝ) ∧
      (b : ℝ) < (c : ℝ) ∧
      (c : ℝ) < ((m + 1) ^ 3 : ℝ) ∧
      ∃ k : ℕ, (a * b * c : ℕ) = k ^ 3 :=
by
  sorry

end existence_of_cube_triplet_l2501_250153


namespace g_243_equals_118_l2501_250170

/-- A function g with the property that g(a) + g(b) = m^3 when a + b = 3^m -/
def g_property (g : ℕ → ℝ) : Prop :=
  ∀ (a b m : ℕ), a > 0 → b > 0 → m > 0 → a + b = 3^m → g a + g b = (m : ℝ)^3

/-- The main theorem stating that g(243) = 118 -/
theorem g_243_equals_118 (g : ℕ → ℝ) (h : g_property g) : g 243 = 118 := by
  sorry


end g_243_equals_118_l2501_250170


namespace find_multiplier_l2501_250137

theorem find_multiplier (x : ℕ) : 72514 * x = 724777430 → x = 10001 := by
  sorry

end find_multiplier_l2501_250137


namespace furniture_markup_proof_l2501_250120

/-- Calculates the percentage markup given the selling price and cost price -/
def percentage_markup (selling_price cost_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Proves that the percentage markup is 25% for the given selling and cost prices -/
theorem furniture_markup_proof (selling_price cost_price : ℚ) 
  (h1 : selling_price = 4800)
  (h2 : cost_price = 3840) : 
  percentage_markup selling_price cost_price = 25 := by
  sorry

end furniture_markup_proof_l2501_250120


namespace prime_divisibility_l2501_250156

theorem prime_divisibility (p a b : ℤ) : 
  Prime p → 
  ∃ k : ℤ, p = 4 * k + 3 → 
  p ∣ (a^2 + b^2) → 
  p ∣ a ∧ p ∣ b := by
  sorry

end prime_divisibility_l2501_250156


namespace no_natural_squares_diff_2014_l2501_250100

theorem no_natural_squares_diff_2014 : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end no_natural_squares_diff_2014_l2501_250100


namespace range_of_a_l2501_250180

-- Define the function f(x) = |x+3| - |x-1|
def f (x : ℝ) : ℝ := |x + 3| - |x - 1|

-- Define the property that the solution set is non-empty
def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, f x ≤ a^2 - 5*a

-- Theorem statement
theorem range_of_a (a : ℝ) :
  has_solution a → (a ≥ 4 ∨ a ≤ 1) :=
by
  sorry

end range_of_a_l2501_250180


namespace function_properties_function_range_l2501_250171

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem function_properties (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, f a x ∈ Set.Icc 1 a) ∧
  (Set.range (f a) = Set.Icc 1 a) →
  a = 2 :=
sorry

theorem function_range (a : ℝ) :
  (∀ x ≤ 2, ∀ y ≤ 2, x < y → f a x > f a y) ∧
  (∀ x ∈ Set.Icc 1 (a + 1), ∀ y ∈ Set.Icc 1 (a + 1), |f a x - f a y| ≤ 4) →
  a ∈ Set.Icc 2 3 :=
sorry

end function_properties_function_range_l2501_250171


namespace eliana_steps_proof_l2501_250129

/-- The number of steps Eliana walked on the first day before adding 300 steps -/
def first_day_steps : ℕ := 200

/-- The total number of steps for all three days -/
def total_steps : ℕ := 1600

theorem eliana_steps_proof :
  first_day_steps + 300 + 2 * (first_day_steps + 300) + 100 = total_steps :=
by sorry

end eliana_steps_proof_l2501_250129


namespace exists_x_tan_eq_two_l2501_250115

theorem exists_x_tan_eq_two : ∃ x : ℝ, Real.tan x = 2 := by
  sorry

end exists_x_tan_eq_two_l2501_250115


namespace cylinder_volume_from_rectangle_l2501_250113

/-- The volume of a cylinder formed by rotating a rectangle about its shorter side. -/
theorem cylinder_volume_from_rectangle (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) :
  let r := w / (2 * Real.pi)
  (Real.pi * r^2 * h) = 1000 / Real.pi → h = 10 ∧ w = 20 := by
  sorry

end cylinder_volume_from_rectangle_l2501_250113


namespace smallest_k_for_mutual_criticism_l2501_250106

/-- Represents a group of deputies and their criticisms. -/
structure DeputyGroup where
  n : ℕ  -- Number of deputies
  k : ℕ  -- Number of deputies each deputy criticizes

/-- Defines when a DeputyGroup has mutual criticism. -/
def has_mutual_criticism (g : DeputyGroup) : Prop :=
  g.n * g.k > (g.n.choose 2)

/-- The smallest k that guarantees mutual criticism in a group of 15 deputies. -/
theorem smallest_k_for_mutual_criticism :
  ∃ k : ℕ, k = 8 ∧
  (∀ g : DeputyGroup, g.n = 15 → g.k ≥ k → has_mutual_criticism g) ∧
  (∀ k' : ℕ, k' < k → ∃ g : DeputyGroup, g.n = 15 ∧ g.k = k' ∧ ¬has_mutual_criticism g) :=
sorry

end smallest_k_for_mutual_criticism_l2501_250106


namespace grazing_area_fence_posts_l2501_250108

/-- Calculates the number of fence posts needed for a rectangular grazing area -/
def fencePostsRequired (length width postSpacing : ℕ) : ℕ :=
  let longSide := max length width
  let shortSide := min length width
  let longSidePosts := longSide / postSpacing + 1
  let shortSidePosts := (shortSide / postSpacing + 1) * 2 - 2
  longSidePosts + shortSidePosts

/-- The problem statement -/
theorem grazing_area_fence_posts :
  fencePostsRequired 70 50 10 = 18 := by
  sorry


end grazing_area_fence_posts_l2501_250108


namespace non_decreasing_integers_count_l2501_250176

/-- The number of digits in the integers we're considering -/
def n : ℕ := 11

/-- The number of possible digit values (1 to 9) -/
def k : ℕ := 9

/-- The number of 11-digit positive integers with non-decreasing digits -/
def non_decreasing_integers : ℕ := Nat.choose (n + k - 1) (k - 1)

theorem non_decreasing_integers_count : non_decreasing_integers = 75582 := by
  sorry

end non_decreasing_integers_count_l2501_250176


namespace mei_fruit_baskets_l2501_250168

theorem mei_fruit_baskets : Nat.gcd 15 (Nat.gcd 9 18) = 3 := by
  sorry

end mei_fruit_baskets_l2501_250168


namespace krishans_money_l2501_250139

theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 490 →
  krishan = 2890 := by
sorry

end krishans_money_l2501_250139


namespace draw_condition_butterfly_wins_condition_l2501_250190

/-- Represents the outcome of the spider web game -/
inductive GameOutcome
  | Draw
  | ButterflyWins

/-- Defines the spider web game structure and rules -/
structure SpiderWebGame where
  K : Nat  -- Number of rings
  R : Nat  -- Number of radii
  butterfly_moves_first : Bool
  K_ge_2 : K ≥ 2
  R_ge_3 : R ≥ 3

/-- Determines the outcome of the spider web game -/
def game_outcome (game : SpiderWebGame) : GameOutcome :=
  if game.K ≥ Nat.ceil (game.R / 2) then
    GameOutcome.Draw
  else
    GameOutcome.ButterflyWins

/-- Theorem stating the conditions for a draw in the spider web game -/
theorem draw_condition (game : SpiderWebGame) :
  game_outcome game = GameOutcome.Draw ↔ game.K ≥ Nat.ceil (game.R / 2) :=
sorry

/-- Theorem stating the conditions for butterfly winning in the spider web game -/
theorem butterfly_wins_condition (game : SpiderWebGame) :
  game_outcome game = GameOutcome.ButterflyWins ↔ game.K < Nat.ceil (game.R / 2) :=
sorry

end draw_condition_butterfly_wins_condition_l2501_250190


namespace sales_and_profit_theorem_l2501_250169

/-- Represents the monthly sales quantity as a function of selling price -/
def monthly_sales (x : ℝ) : ℝ := -30 * x + 960

/-- Represents the monthly profit as a function of selling price -/
def monthly_profit (x : ℝ) : ℝ := (x - 10) * (monthly_sales x)

theorem sales_and_profit_theorem :
  let cost_price : ℝ := 10
  let price1 : ℝ := 20
  let price2 : ℝ := 30
  let sales1 : ℝ := 360
  let sales2 : ℝ := 60
  let target_profit : ℝ := 3600
  (∀ x, monthly_sales x = -30 * x + 960) ∧
  (monthly_sales price1 = sales1) ∧
  (monthly_sales price2 = sales2) ∧
  (∃ x, monthly_profit x = target_profit) ∧
  (monthly_profit 22 = target_profit) ∧
  (monthly_profit 20 = target_profit) := by
  sorry

#check sales_and_profit_theorem

end sales_and_profit_theorem_l2501_250169


namespace sally_peaches_l2501_250152

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked : ℕ := 55

/-- The total number of peaches after picking -/
def total_peaches : ℕ := 68

/-- The initial number of peaches Sally had -/
def initial_peaches : ℕ := total_peaches - peaches_picked

theorem sally_peaches : initial_peaches + peaches_picked = total_peaches := by
  sorry

end sally_peaches_l2501_250152


namespace correct_combined_average_l2501_250191

def num_students : ℕ := 100
def math_avg : ℚ := 85
def science_avg : ℚ := 89
def num_incorrect : ℕ := 5

def incorrect_math_marks : List ℕ := [76, 80, 95, 70, 90]
def correct_math_marks : List ℕ := [86, 70, 75, 90, 100]
def incorrect_science_marks : List ℕ := [105, 60, 80, 92, 78]
def correct_science_marks : List ℕ := [95, 70, 90, 82, 88]

theorem correct_combined_average :
  let math_total := num_students * math_avg + (correct_math_marks.sum - incorrect_math_marks.sum)
  let science_total := num_students * science_avg + (correct_science_marks.sum - incorrect_science_marks.sum)
  let combined_total := math_total + science_total
  let combined_avg := combined_total / (2 * num_students)
  combined_avg = 87.1 := by sorry

end correct_combined_average_l2501_250191


namespace mcdonalds_coupon_value_l2501_250195

/-- Proves that given an original cost of $7.50, a senior citizen discount of 20%,
    and a final payment of $4, the coupon value that makes this possible is $2.50. -/
theorem mcdonalds_coupon_value :
  let original_cost : ℝ := 7.50
  let senior_discount : ℝ := 0.20
  let final_payment : ℝ := 4.00
  let coupon_value : ℝ := 2.50
  (1 - senior_discount) * (original_cost - coupon_value) = final_payment := by
sorry

end mcdonalds_coupon_value_l2501_250195


namespace smallest_solution_quartic_l2501_250177

theorem smallest_solution_quartic (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 → x ≥ -Real.sqrt 26 :=
by sorry

end smallest_solution_quartic_l2501_250177


namespace inverse_343_mod_103_l2501_250141

theorem inverse_343_mod_103 (h : (7⁻¹ : ZMod 103) = 44) : (343⁻¹ : ZMod 103) = 3 := by
  sorry

end inverse_343_mod_103_l2501_250141


namespace golden_ratio_properties_l2501_250163

theorem golden_ratio_properties :
  let a : ℝ := (Real.sqrt 5 + 1) / 2
  let b : ℝ := (Real.sqrt 5 - 1) / 2
  (b / a + a / b = 3) ∧ (a^2 + b^2 + a*b = 4) := by
  sorry

end golden_ratio_properties_l2501_250163


namespace moving_circle_trajectory_l2501_250121

-- Define the fixed point M
def M : ℝ × ℝ := (-4, 0)

-- Define the equation of the known circle N
def circle_N (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 16

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop := x^2/4 - y^2/12 = 1

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (x y : ℝ),
    (∃ (r : ℝ), 
      -- The moving circle passes through M
      (x + 4)^2 + y^2 = r^2 ∧
      -- The moving circle is tangent to N
      ∃ (x_n y_n : ℝ), circle_N x_n y_n ∧ (x - x_n)^2 + (y - y_n)^2 = r^2) →
    trajectory x y :=
sorry

end moving_circle_trajectory_l2501_250121


namespace yarn_ball_ratio_l2501_250181

/-- Given three balls of yarn, where:
    - The third ball is three times as large as the first ball
    - 27 feet of yarn was used for the third ball
    - 18 feet of yarn was used for the second ball
    Prove that the ratio of the size of the first ball to the size of the second ball is 1:2 -/
theorem yarn_ball_ratio :
  ∀ (first_ball second_ball third_ball : ℝ),
  third_ball = 3 * first_ball →
  third_ball = 27 →
  second_ball = 18 →
  first_ball / second_ball = 1 / 2 := by
sorry

end yarn_ball_ratio_l2501_250181


namespace sum_of_digits_of_seven_to_fifteen_l2501_250101

/-- The sum of the tens digit and the ones digit of (3 + 4)^15 is 7 -/
theorem sum_of_digits_of_seven_to_fifteen (n : ℕ) : n = (3 + 4)^15 → 
  (n / 10 % 10 + n % 10 = 7) := by
sorry

end sum_of_digits_of_seven_to_fifteen_l2501_250101


namespace cube_sum_equality_l2501_250104

theorem cube_sum_equality (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a^3 + 10) / a^2 = (b^3 + 10) / b^2 ∧
  (b^3 + 10) / b^2 = (c^3 + 10) / c^2 →
  a^3 + b^3 + c^3 = 1301 := by
  sorry

end cube_sum_equality_l2501_250104


namespace digit_sum_property_l2501_250142

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A proposition stating that a number is a 1962-digit number -/
def is1962DigitNumber (n : ℕ) : Prop := sorry

theorem digit_sum_property (n : ℕ) 
  (h1 : is1962DigitNumber n) 
  (h2 : n % 9 = 0) : 
  sumOfDigits (sumOfDigits (sumOfDigits n)) = 9 := by sorry

end digit_sum_property_l2501_250142


namespace det_A_squared_l2501_250123

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; 7, 2]

theorem det_A_squared : (Matrix.det A)^2 = 121 := by
  sorry

end det_A_squared_l2501_250123


namespace quadruple_batch_cans_l2501_250199

/-- Represents the number of cans for each ingredient in a normal batch of chili --/
structure NormalBatch where
  chilis : ℕ
  beans : ℕ
  tomatoes : ℕ

/-- Defines a normal batch of chili according to Carla's recipe --/
def carla_normal_batch : NormalBatch where
  chilis := 1
  beans := 2
  tomatoes := 3  -- 50% more than beans, so 2 * 1.5 = 3

/-- Calculates the total number of cans for a given batch size --/
def total_cans (batch : NormalBatch) (multiplier : ℕ) : ℕ :=
  multiplier * (batch.chilis + batch.beans + batch.tomatoes)

/-- Theorem: A quadruple batch of Carla's chili requires 24 cans of food --/
theorem quadruple_batch_cans : 
  total_cans carla_normal_batch 4 = 24 := by
  sorry

end quadruple_batch_cans_l2501_250199


namespace trigonometric_equation_solution_l2501_250158

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (3 * x) + Real.sin x - Real.sin (2 * x) = 2 * Real.cos x * (Real.cos x - 1)) ↔ 
  (∃ k : ℤ, x = π / 2 * (2 * k + 1)) ∨ 
  (∃ n : ℤ, x = 2 * π * n) ∨ 
  (∃ l : ℤ, x = π / 4 * (4 * l - 1)) := by
sorry

end trigonometric_equation_solution_l2501_250158


namespace paths_through_F_l2501_250182

/-- The number of paths on a grid from (0,0) to (a,b) -/
def gridPaths (a b : ℕ) : ℕ := Nat.choose (a + b) a

/-- The coordinates of point E -/
def E : ℕ × ℕ := (0, 0)

/-- The coordinates of point F -/
def F : ℕ × ℕ := (5, 2)

/-- The coordinates of point G -/
def G : ℕ × ℕ := (6, 5)

/-- The total number of steps from E to G -/
def totalSteps : ℕ := G.1 - E.1 + G.2 - E.2

theorem paths_through_F : 
  gridPaths (F.1 - E.1) (F.2 - E.2) * gridPaths (G.1 - F.1) (G.2 - F.2) = 84 ∧
  totalSteps = 12 := by
  sorry

end paths_through_F_l2501_250182


namespace additional_male_workers_hired_l2501_250196

theorem additional_male_workers_hired (
  initial_female_percentage : ℚ)
  (final_female_percentage : ℚ)
  (final_total_employees : ℕ)
  (h1 : initial_female_percentage = 3/5)
  (h2 : final_female_percentage = 11/20)
  (h3 : final_total_employees = 240) :
  (final_total_employees : ℚ) - (final_female_percentage * final_total_employees) / initial_female_percentage = 20 := by
  sorry

end additional_male_workers_hired_l2501_250196


namespace no_rearranged_power_of_two_l2501_250122

/-- Checks if all digits of a natural number are non-zero -/
def allDigitsNonZero (n : ℕ) : Prop := sorry

/-- Checks if two natural numbers have the same digits (possibly in different order) -/
def sameDigits (m n : ℕ) : Prop := sorry

/-- There do not exist two distinct powers of 2 with all non-zero digits that are rearrangements of each other -/
theorem no_rearranged_power_of_two : ¬∃ (a b : ℕ), a ≠ b ∧ 
  allDigitsNonZero (2^a) ∧ 
  allDigitsNonZero (2^b) ∧ 
  sameDigits (2^a) (2^b) := by
  sorry

end no_rearranged_power_of_two_l2501_250122


namespace same_color_probability_l2501_250178

theorem same_color_probability (blue_balls yellow_balls : ℕ) 
  (h_blue : blue_balls = 8) (h_yellow : yellow_balls = 5) : 
  let total_balls := blue_balls + yellow_balls
  let prob_blue := blue_balls / total_balls
  let prob_yellow := yellow_balls / total_balls
  prob_blue ^ 2 + prob_yellow ^ 2 = 89 / 169 := by
  sorry

end same_color_probability_l2501_250178


namespace lateral_surface_area_regular_triangular_prism_l2501_250164

/-- Given a regular triangular prism with height h, where a line passing through 
    the center of the upper base and the midpoint of the side of the lower base 
    is inclined at an angle 60° to the plane of the base, 
    the lateral surface area of the prism is 6h². -/
theorem lateral_surface_area_regular_triangular_prism 
  (h : ℝ) 
  (h_pos : h > 0) 
  (incline_angle : ℝ) 
  (incline_angle_eq : incline_angle = 60 * π / 180) : 
  ∃ (S : ℝ), S = 6 * h^2 ∧ S > 0 := by
  sorry

end lateral_surface_area_regular_triangular_prism_l2501_250164


namespace units_digit_power_four_l2501_250110

theorem units_digit_power_four (a : ℤ) (n : ℕ) : 
  10 ∣ (a^(n+4) - a^n) := by
  sorry

end units_digit_power_four_l2501_250110


namespace truck_sand_problem_l2501_250114

/-- The amount of sand remaining on a truck after making several stops --/
def sandRemaining (initialSand : ℕ) (sandLostAtStops : List ℕ) : ℕ :=
  initialSand - sandLostAtStops.sum

/-- Theorem: A truck with 1050 pounds of sand that loses 32, 67, 45, and 54 pounds at four stops will have 852 pounds remaining --/
theorem truck_sand_problem :
  let initialSand : ℕ := 1050
  let sandLostAtStops : List ℕ := [32, 67, 45, 54]
  sandRemaining initialSand sandLostAtStops = 852 := by
  sorry

#eval sandRemaining 1050 [32, 67, 45, 54]

end truck_sand_problem_l2501_250114


namespace solution_set_inequality_l2501_250136

theorem solution_set_inequality (x : ℝ) : 
  (Set.Ioo 1 2 : Set ℝ) = {x | (x - 1) * (2 - x) > 0} :=
sorry

end solution_set_inequality_l2501_250136


namespace ice_cream_flavors_l2501_250112

/-- The number of ways to distribute indistinguishable objects into distinguishable boxes -/
def distribute (n : ℕ) (k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of new ice cream flavors -/
def new_flavors : ℕ := distribute 6 5

theorem ice_cream_flavors : new_flavors = 210 := by sorry

end ice_cream_flavors_l2501_250112


namespace donut_selections_l2501_250172

theorem donut_selections (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end donut_selections_l2501_250172


namespace eugene_pencils_l2501_250186

/-- Calculates the total number of pencils Eugene has after receiving more. -/
def total_pencils (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Eugene's total pencils -/
theorem eugene_pencils : total_pencils 51 6 = 57 := by
  sorry

end eugene_pencils_l2501_250186


namespace children_off_bus_l2501_250179

theorem children_off_bus (initial : ℕ) (remaining : ℕ) (h1 : initial = 43) (h2 : remaining = 21) :
  initial - remaining = 22 := by
  sorry

end children_off_bus_l2501_250179


namespace probability_of_four_boys_l2501_250160

open BigOperators Finset

theorem probability_of_four_boys (total_students : ℕ) (total_boys : ℕ) (selected_students : ℕ) :
  total_students = 15 →
  total_boys = 7 →
  selected_students = 10 →
  (Nat.choose total_boys 4 * Nat.choose (total_students - total_boys) (selected_students - 4)) /
  Nat.choose total_students selected_students =
  Nat.choose 7 4 * Nat.choose 8 6 / Nat.choose 15 10 :=
by sorry

end probability_of_four_boys_l2501_250160


namespace pepsi_volume_l2501_250194

theorem pepsi_volume (maaza : ℕ) (sprite : ℕ) (total_cans : ℕ) (pepsi : ℕ) : 
  maaza = 40 →
  sprite = 368 →
  total_cans = 69 →
  (maaza + sprite + pepsi) % total_cans = 0 →
  pepsi = 75 :=
by sorry

end pepsi_volume_l2501_250194


namespace log3_20_approximation_l2501_250193

-- Define the approximations given in the problem
def log10_2_approx : ℝ := 0.301
def log10_3_approx : ℝ := 0.477

-- Define the target value
def target_value : ℝ := 2.7

-- State the theorem
theorem log3_20_approximation :
  let log3_20 := (1 + log10_2_approx) / log10_3_approx
  abs (log3_20 - target_value) < 0.05 := by sorry

end log3_20_approximation_l2501_250193


namespace election_win_probability_l2501_250162

/-- Represents the state of an election --/
structure ElectionState :=
  (total_voters : ℕ)
  (votes_a : ℕ)
  (votes_b : ℕ)

/-- Calculates the probability of candidate A winning given the current state --/
noncomputable def win_probability (state : ElectionState) : ℚ :=
  sorry

/-- The main theorem stating the probability of the initially leading candidate winning --/
theorem election_win_probability :
  let initial_state : ElectionState := ⟨2019, 2, 1⟩
  win_probability initial_state = 1513 / 2017 :=
sorry

end election_win_probability_l2501_250162


namespace complex_power_2013_l2501_250127

theorem complex_power_2013 : (((1 + Complex.I) / (1 - Complex.I)) ^ 2013 : ℂ) = Complex.I := by sorry

end complex_power_2013_l2501_250127


namespace lisa_candy_consumption_l2501_250146

/-- The number of candies Lisa has initially -/
def initial_candies : ℕ := 36

/-- The number of candies Lisa eats on Mondays and Wednesdays -/
def candies_on_mon_wed : ℕ := 2

/-- The number of candies Lisa eats on other days -/
def candies_on_other_days : ℕ := 1

/-- The number of days Lisa eats 2 candies per week -/
def days_with_two_candies : ℕ := 2

/-- The number of days Lisa eats 1 candy per week -/
def days_with_one_candy : ℕ := 5

/-- The total number of candies Lisa eats in a week -/
def candies_per_week : ℕ := 
  days_with_two_candies * candies_on_mon_wed + 
  days_with_one_candy * candies_on_other_days

/-- The number of weeks it takes for Lisa to eat all the candies -/
def weeks_to_eat_all_candies : ℕ := initial_candies / candies_per_week

theorem lisa_candy_consumption : weeks_to_eat_all_candies = 4 := by
  sorry

end lisa_candy_consumption_l2501_250146


namespace largest_angle_in_pentagon_l2501_250125

/-- In a pentagon FGHIJ, given the following conditions:
  - Angle F measures 50°
  - Angle G measures 75°
  - Angles H and I are equal
  - Angle J is 10° more than twice angle H
  Prove that the largest angle measures 212.5° -/
theorem largest_angle_in_pentagon (F G H I J : ℝ) : 
  F = 50 ∧ 
  G = 75 ∧ 
  H = I ∧ 
  J = 2 * H + 10 ∧ 
  F + G + H + I + J = 540 → 
  max F (max G (max H (max I J))) = 212.5 := by
sorry

end largest_angle_in_pentagon_l2501_250125


namespace y_value_l2501_250128

theorem y_value (x y : ℝ) (h1 : x^2 = y - 5) (h2 : x = 7) : y = 54 := by
  sorry

end y_value_l2501_250128


namespace relatively_prime_2n_plus_1_and_4n_squared_plus_1_l2501_250109

theorem relatively_prime_2n_plus_1_and_4n_squared_plus_1 (n : ℕ+) :
  Nat.gcd (2 * n.val + 1) (4 * n.val^2 + 1) = 1 := by
  sorry

end relatively_prime_2n_plus_1_and_4n_squared_plus_1_l2501_250109


namespace range_of_m_l2501_250184

def proposition_p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (2*m) - y^2 / (m-1) = 1 ∧ 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1 ∧ 
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧ c < a

def proposition_q (m : ℝ) : Prop :=
  ∃ e : ℝ, 1 < e ∧ e < 2 ∧
  ∃ x y : ℝ, y^2 / 5 - x^2 / m = 1 ∧
  e^2 = (5 + m) / 5

theorem range_of_m :
  ∀ m : ℝ, (proposition_p m ∨ proposition_q m) → 0 < m ∧ m < 15 :=
sorry

end range_of_m_l2501_250184


namespace last_divisor_problem_l2501_250155

theorem last_divisor_problem (initial : ℚ) (div1 div2 mult last_div : ℚ) (result : ℚ) : 
  initial = 377 →
  div1 = 13 →
  div2 = 29 →
  mult = 1/4 →
  result = 0.125 →
  (((initial / div1) / div2) * mult) / last_div = result →
  last_div = 2 :=
by sorry

end last_divisor_problem_l2501_250155


namespace degree_of_composed_product_l2501_250145

/-- Given polynomials f and g with degrees 3 and 6 respectively,
    the degree of f(x^2) · g(x^3) is 24. -/
theorem degree_of_composed_product (f g : Polynomial ℝ) 
  (hf : Polynomial.degree f = 3)
  (hg : Polynomial.degree g = 6) :
  Polynomial.degree (f.comp (Polynomial.X ^ 2) * g.comp (Polynomial.X ^ 3)) = 24 := by
  sorry

end degree_of_composed_product_l2501_250145


namespace number_with_specific_remainders_l2501_250102

theorem number_with_specific_remainders : ∃ n : ℕ, 
  (∀ k : ℕ, 2 ≤ k → k ≤ 10 → n % k = k - 1) ∧ n = 2519 := by
  sorry

end number_with_specific_remainders_l2501_250102


namespace max_value_implies_a_l2501_250154

/-- Given a function y = x(1-ax) where 0 < x < 1/a, if the maximum value of y is 1/12, then a = 3 -/
theorem max_value_implies_a (a : ℝ) : 
  (∃ (y : ℝ → ℝ), (∀ x : ℝ, 0 < x → x < 1/a → y x = x*(1-a*x)) ∧ 
   (∃ M : ℝ, M = 1/12 ∧ ∀ x : ℝ, 0 < x → x < 1/a → y x ≤ M)) →
  a = 3 := by
sorry

end max_value_implies_a_l2501_250154


namespace crocodile_earnings_exceed_peter_l2501_250157

theorem crocodile_earnings_exceed_peter (n : ℕ) : (∀ k < n, 2^k ≤ 64*k + 1) ∧ 2^n > 64*n + 1 → n = 10 := by
  sorry

end crocodile_earnings_exceed_peter_l2501_250157


namespace five_rooks_on_five_by_five_l2501_250165

/-- The number of ways to place n distinct rooks on an nxn chess board 
    such that each column and row contains no more than one rook -/
def rook_placements (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem: There are 120 ways to place 5 distinct rooks on a 5x5 chess board 
    such that each column and row contains no more than one rook -/
theorem five_rooks_on_five_by_five : rook_placements 5 = 120 := by
  sorry

end five_rooks_on_five_by_five_l2501_250165


namespace not_always_prime_l2501_250107

def P (n : ℤ) : ℤ := n^2 + n + 41

theorem not_always_prime : ∃ n : ℤ, ¬(Nat.Prime (Int.natAbs (P n))) := by
  sorry

end not_always_prime_l2501_250107


namespace quadratic_inequality_solution_set_l2501_250116

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 < 2*x} = Set.Ioo 0 2 := by sorry

end quadratic_inequality_solution_set_l2501_250116


namespace diplomats_speaking_french_l2501_250132

theorem diplomats_speaking_french (total : ℕ) (not_russian : ℕ) (neither : ℕ) (both : ℕ) :
  total = 100 →
  not_russian = 32 →
  neither = 20 →
  both = 10 →
  ∃ french : ℕ, french = 22 ∧ french = total - not_russian + both :=
by sorry

end diplomats_speaking_french_l2501_250132


namespace speedster_convertibles_count_l2501_250140

theorem speedster_convertibles_count (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) :
  speedsters = total / 3 →
  30 = total - speedsters →
  convertibles = (4 * speedsters) / 5 →
  convertibles = 12 := by
sorry

end speedster_convertibles_count_l2501_250140


namespace wall_bricks_count_l2501_250130

/-- The number of bricks in the wall after adjustments -/
def total_bricks : ℕ :=
  let initial_courses := 5
  let additional_courses := 7
  let bricks_per_course := 450
  let initial_bricks := initial_courses * bricks_per_course
  let added_bricks := additional_courses * bricks_per_course
  let removed_bricks := [
    bricks_per_course / 3,
    bricks_per_course / 4,
    bricks_per_course / 5,
    bricks_per_course / 6,
    bricks_per_course / 7,
    bricks_per_course / 9,
    10
  ]
  initial_bricks + added_bricks - removed_bricks.sum

/-- Theorem stating that the total number of bricks in the wall is 4848 -/
theorem wall_bricks_count : total_bricks = 4848 := by
  sorry

end wall_bricks_count_l2501_250130


namespace gold_silver_weight_problem_l2501_250126

theorem gold_silver_weight_problem (x y : ℝ) : 
  (9 * x = 11 * y) ∧ ((10 * y + x) - (8 * x + y) = 13) ↔ 
  (9 * x = 11 * y ∧ 
   ∃ (gold_bag silver_bag : ℝ),
     gold_bag = 9 * x ∧
     silver_bag = 11 * y ∧
     gold_bag = silver_bag ∧
     (silver_bag + x - y) - (gold_bag - x + y) = 13) :=
by sorry

end gold_silver_weight_problem_l2501_250126


namespace perpendicular_lines_from_quadratic_l2501_250151

/-- Two lines with slopes that are the roots of x^2 - 3x - 1 = 0 are perpendicular -/
theorem perpendicular_lines_from_quadratic (k₁ k₂ : ℝ) : 
  k₁^2 - 3*k₁ - 1 = 0 → k₂^2 - 3*k₂ - 1 = 0 → k₁ ≠ k₂ → k₁ * k₂ = -1 := by
  sorry

end perpendicular_lines_from_quadratic_l2501_250151


namespace toothpicks_15th_stage_l2501_250119

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  3 + 2 * (n - 1)

/-- Theorem stating that the 15th stage of the pattern has 31 toothpicks -/
theorem toothpicks_15th_stage :
  toothpicks 15 = 31 := by
  sorry

end toothpicks_15th_stage_l2501_250119


namespace sum_of_roots_quadratic_sum_of_roots_specific_equation_l2501_250105

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (x + y : ℝ) = -b / a :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 7*x + 2 - 11
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (x + y : ℝ) = 7 :=
sorry

end sum_of_roots_quadratic_sum_of_roots_specific_equation_l2501_250105


namespace group_size_proof_l2501_250183

theorem group_size_proof (total_paise : ℕ) (contribution : ℕ → ℕ) : 
  (total_paise = 1369) →
  (∀ n : ℕ, contribution n = n) →
  (∃ n : ℕ, n * contribution n = total_paise) →
  (∃ n : ℕ, n * n = total_paise) →
  (∃ n : ℕ, n = 37) :=
by sorry

end group_size_proof_l2501_250183


namespace stable_yield_promotion_l2501_250150

/-- Represents a type of red rice -/
structure RedRice where
  typeName : String
  averageYield : ℝ
  variance : ℝ

/-- Determines if a type of red rice is suitable for promotion based on yield stability -/
def isSuitableForPromotion (rice1 rice2 : RedRice) : Prop :=
  rice1.averageYield = rice2.averageYield ∧ 
  rice1.variance < rice2.variance

theorem stable_yield_promotion (A B : RedRice) 
  (h_yield : A.averageYield = B.averageYield)
  (h_variance : A.variance < B.variance) : 
  isSuitableForPromotion A B := by
  sorry

#check stable_yield_promotion

end stable_yield_promotion_l2501_250150


namespace ellen_dinner_calories_l2501_250133

/-- Calculates the remaining calories for dinner given a daily limit and calories consumed for breakfast, lunch, and snack. -/
def remaining_calories_for_dinner (daily_limit : ℕ) (breakfast : ℕ) (lunch : ℕ) (snack : ℕ) : ℕ :=
  daily_limit - (breakfast + lunch + snack)

/-- Proves that given the specific calorie values in the problem, the remaining calories for dinner is 832. -/
theorem ellen_dinner_calories : 
  remaining_calories_for_dinner 2200 353 885 130 = 832 := by
sorry

end ellen_dinner_calories_l2501_250133


namespace race_time_difference_l2501_250197

/-- 
Given a 1000-meter race where runner A completes the race in 192 seconds and 
is 40 meters ahead of runner B at the finish line, prove that A beats B by 7.68 seconds.
-/
theorem race_time_difference (race_distance : ℝ) (a_time : ℝ) (distance_difference : ℝ) : 
  race_distance = 1000 →
  a_time = 192 →
  distance_difference = 40 →
  (race_distance / a_time) * (distance_difference / race_distance) * a_time = 7.68 :=
by sorry

end race_time_difference_l2501_250197


namespace factorization_cubic_minus_linear_l2501_250138

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end factorization_cubic_minus_linear_l2501_250138


namespace A_minus_3B_formula_A_minus_3B_value_x_value_when_independent_of_y_l2501_250103

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := x^2 - 2 * x - y + x * y - 5

-- Theorem 1: A - 3B = 5x + 5y - 7xy + 15
theorem A_minus_3B_formula (x y : ℝ) :
  A x y - 3 * B x y = 5 * x + 5 * y - 7 * x * y + 15 := by sorry

-- Theorem 2: A - 3B = 26 when (x + y - 4/5)^2 + |xy + 1| = 0
theorem A_minus_3B_value (x y : ℝ) 
  (h : (x + y - 4/5)^2 + |x * y + 1| = 0) :
  A x y - 3 * B x y = 26 := by sorry

-- Theorem 3: x = 5/7 when the coefficient of y in A - 3B is zero
theorem x_value_when_independent_of_y (x : ℝ) 
  (h : ∀ y : ℝ, 5 - 7 * x = 0) :
  x = 5/7 := by sorry

end A_minus_3B_formula_A_minus_3B_value_x_value_when_independent_of_y_l2501_250103


namespace negation_of_inequality_statement_l2501_250131

theorem negation_of_inequality_statement :
  (¬ ∀ x : ℝ, x > 0 → x - 1 ≥ Real.log x) ↔
  (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ - 1 < Real.log x₀) :=
by sorry

end negation_of_inequality_statement_l2501_250131


namespace expression_simplification_l2501_250173

theorem expression_simplification (x y z : ℝ) : 
  ((x + z) - (y - 2*z)) - ((x - 2*z) - (y + z)) = 6*z := by sorry

end expression_simplification_l2501_250173


namespace cube_difference_l2501_250166

theorem cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := by
sorry

end cube_difference_l2501_250166


namespace store_purchase_combinations_l2501_250143

/-- The number of oreo flavors available -/
def num_oreo_flavors : ℕ := 6

/-- The number of milk flavors available -/
def num_milk_flavors : ℕ := 4

/-- The total number of items Alpha can choose from -/
def total_items : ℕ := num_oreo_flavors + num_milk_flavors

/-- The number of items they collectively buy -/
def total_purchased : ℕ := 3

/-- Represents the ways Alpha can choose items without repeats -/
def alpha_choices (k : ℕ) : ℕ := Nat.choose total_items k

/-- Represents the ways Beta can choose k oreos with possible repeats -/
def beta_choices (k : ℕ) : ℕ :=
  Nat.choose num_oreo_flavors k +  -- All different
  (if k ≥ 2 then num_oreo_flavors * (num_oreo_flavors - 1) else 0) +  -- Two same, one different (if k ≥ 2)
  (if k = 3 then num_oreo_flavors else 0)  -- All same (if k = 3)

/-- The total number of ways for Alpha and Beta to collectively buy 3 items -/
def total_ways : ℕ :=
  alpha_choices 3 +  -- Alpha buys 3, Beta 0
  alpha_choices 2 * num_oreo_flavors +  -- Alpha buys 2, Beta 1
  alpha_choices 1 * beta_choices 2 +  -- Alpha buys 1, Beta 2
  beta_choices 3  -- Alpha buys 0, Beta 3

theorem store_purchase_combinations :
  total_ways = 656 := by sorry

end store_purchase_combinations_l2501_250143


namespace floor_ceiling_sum_five_l2501_250174

theorem floor_ceiling_sum_five (x : ℝ) :
  (⌊x⌋ + ⌈x⌉ = 5) ↔ (2 < x ∧ x < 3) := by
  sorry

end floor_ceiling_sum_five_l2501_250174


namespace triangle_sum_equals_nine_l2501_250161

def triangle_operation (a b c : ℤ) : ℤ := a * b - c

theorem triangle_sum_equals_nine : 
  triangle_operation 3 4 5 + triangle_operation 1 2 4 + triangle_operation 2 5 6 = 9 := by
  sorry

end triangle_sum_equals_nine_l2501_250161


namespace smallest_whole_number_larger_than_sum_l2501_250124

def mixed_to_fraction (whole : ℤ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num : ℚ) / (denom : ℚ)

def sum_of_mixed_numbers : ℚ :=
  mixed_to_fraction 1 2 3 +
  mixed_to_fraction 2 1 4 +
  mixed_to_fraction 3 3 8 +
  mixed_to_fraction 4 1 6

theorem smallest_whole_number_larger_than_sum :
  (⌈sum_of_mixed_numbers⌉ : ℤ) = 12 := by sorry

end smallest_whole_number_larger_than_sum_l2501_250124


namespace bee_speed_l2501_250192

/-- The speed of a bee flying between flowers -/
theorem bee_speed (time_to_rose time_to_poppy : ℝ)
  (distance_difference speed_difference : ℝ)
  (h1 : time_to_rose = 10)
  (h2 : time_to_poppy = 6)
  (h3 : distance_difference = 8)
  (h4 : speed_difference = 3) :
  ∃ (speed_to_rose : ℝ),
    speed_to_rose * time_to_rose = 
    (speed_to_rose + speed_difference) * time_to_poppy + distance_difference ∧
    speed_to_rose = 6.5 := by
  sorry

end bee_speed_l2501_250192


namespace sum_x_y_value_l2501_250198

theorem sum_x_y_value (x y : ℚ) 
  (eq1 : 3 * x - 4 * y = 18) 
  (eq2 : x + 3 * y = -1) : 
  x + y = 29 / 13 := by
sorry

end sum_x_y_value_l2501_250198


namespace max_profundity_eq_fib_l2501_250167

/-- The dog dictionary consists of words made from letters A and U -/
inductive DogLetter
| A
| U

/-- A word in the dog dictionary is a list of DogLetters -/
def DogWord := List DogLetter

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

/-- The profundity of a word is the number of its subwords -/
def profundity (w : DogWord) : ℕ := sorry

/-- The maximum profundity for words of length n -/
def max_profundity (n : ℕ) : ℕ := sorry

/-- The main theorem: maximum profundity equals F_{n+3} - 3 -/
theorem max_profundity_eq_fib (n : ℕ) :
  max_profundity n = fib (n + 3) - 3 := by sorry

end max_profundity_eq_fib_l2501_250167


namespace single_elimination_tournament_games_l2501_250189

/-- 
Calculates the number of games required in a single-elimination tournament
to declare a winner, given the number of teams participating.
-/
def gamesRequired (numTeams : ℕ) : ℕ := numTeams - 1

/-- 
Theorem: In a single-elimination tournament with 25 teams and no possibility of ties,
the number of games required to declare a winner is 24.
-/
theorem single_elimination_tournament_games :
  gamesRequired 25 = 24 := by
  sorry

end single_elimination_tournament_games_l2501_250189


namespace distance_per_block_l2501_250175

/-- Proves that the distance of each block is 1/8 mile -/
theorem distance_per_block (total_time : ℚ) (total_blocks : ℕ) (speed : ℚ) :
  total_time = 10 / 60 →
  total_blocks = 16 →
  speed = 12 →
  (speed * total_time) / total_blocks = 1 / 8 := by
  sorry

#check distance_per_block

end distance_per_block_l2501_250175


namespace g_of_3_eq_125_l2501_250187

def g (x : ℝ) : ℝ := 3 * x^4 - 5 * x^3 + 4 * x^2 - 7 * x + 2

theorem g_of_3_eq_125 : g 3 = 125 := by
  sorry

end g_of_3_eq_125_l2501_250187


namespace kitty_dusting_time_l2501_250185

/-- Represents the cleaning activities and their durations in Kitty's living room --/
structure CleaningActivities where
  pickingUpToys : ℕ
  vacuuming : ℕ
  cleaningWindows : ℕ
  totalWeeks : ℕ
  totalMinutes : ℕ

/-- Calculates the time spent dusting furniture each week --/
def dustingTime (c : CleaningActivities) : ℕ :=
  let otherTasksTime := c.pickingUpToys + c.vacuuming + c.cleaningWindows
  let totalOtherTasksTime := otherTasksTime * c.totalWeeks
  let totalDustingTime := c.totalMinutes - totalOtherTasksTime
  totalDustingTime / c.totalWeeks

/-- Theorem stating that Kitty spends 10 minutes each week dusting furniture --/
theorem kitty_dusting_time :
  ∀ (c : CleaningActivities),
    c.pickingUpToys = 5 →
    c.vacuuming = 20 →
    c.cleaningWindows = 15 →
    c.totalWeeks = 4 →
    c.totalMinutes = 200 →
    dustingTime c = 10 := by
  sorry

end kitty_dusting_time_l2501_250185


namespace complex_expression_equality_l2501_250111

theorem complex_expression_equality : 
  (Real.pi - 3.14) ^ 0 + |-Real.sqrt 3| - (1/2)⁻¹ - Real.sin (π/3) = -1 + Real.sqrt 3 / 2 := by
  sorry

end complex_expression_equality_l2501_250111


namespace perpendicular_condition_l2501_250147

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the first line y = ax + 1 -/
def slope₁ (a : ℝ) : ℝ := a

/-- The slope of the second line y = (a-2)x - 1 -/
def slope₂ (a : ℝ) : ℝ := a - 2

/-- Theorem: a = 1 is a necessary and sufficient condition for the lines to be perpendicular -/
theorem perpendicular_condition (a : ℝ) : 
  perpendicular (slope₁ a) (slope₂ a) ↔ a = 1 := by
  sorry

end perpendicular_condition_l2501_250147


namespace bell_rings_count_l2501_250148

def number_of_classes : Nat := 5

def current_class : Nat := 5

def bell_rings_per_class : Nat := 2

theorem bell_rings_count (n : Nat) (c : Nat) (r : Nat) 
  (h1 : n = number_of_classes) 
  (h2 : c = current_class) 
  (h3 : r = bell_rings_per_class) 
  (h4 : c ≤ n) : 
  (c - 1) * r + 1 = 9 := by
  sorry

#check bell_rings_count

end bell_rings_count_l2501_250148


namespace overlapping_tape_length_l2501_250159

/-- 
Given three tapes of equal length attached with equal overlapping parts,
this theorem proves the length of one overlapping portion.
-/
theorem overlapping_tape_length 
  (tape_length : ℝ) 
  (attached_length : ℝ) 
  (h1 : tape_length = 217) 
  (h2 : attached_length = 627) : 
  (3 * tape_length - attached_length) / 2 = 12 := by
  sorry

#check overlapping_tape_length

end overlapping_tape_length_l2501_250159


namespace max_value_of_sum_products_l2501_250117

theorem max_value_of_sum_products (x y z : ℝ) (h : x + 2 * y + z = 6) :
  ∃ (max : ℝ), max = 6 ∧ ∀ (a b c : ℝ), a + 2 * b + c = 6 → a * b + a * c + b * c ≤ max :=
sorry

end max_value_of_sum_products_l2501_250117


namespace complex_modulus_l2501_250144

theorem complex_modulus (z : ℂ) (i : ℂ) (h : i * i = -1) (eq : z / (1 + i) = 2 * i) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
sorry

end complex_modulus_l2501_250144


namespace transform_f_to_g_l2501_250149

/-- The original function -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- The resulting function after transformation -/
def g (x : ℝ) : ℝ := (x - 5)^2 + 5

/-- Vertical shift transformation -/
def vertical_shift (h : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := h x + k

/-- Horizontal shift transformation -/
def horizontal_shift (h : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := h (x - k)

/-- Theorem stating that the transformation of f results in g -/
theorem transform_f_to_g : 
  ∀ x, horizontal_shift (vertical_shift f 3) 4 x = g x :=
sorry

end transform_f_to_g_l2501_250149


namespace inequality_solution_set_l2501_250188

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a + 2 / (2^x + 1)

theorem inequality_solution_set 
  (a : ℝ)
  (h1 : ∀ x y : ℝ, x < y → f a x > f a y)
  (h2 : ∀ x : ℝ, f a (-x) = -(f a x)) :
  {t : ℝ | f a (2*t + 1) + f a (t - 5) ≤ 0} = {t : ℝ | t ≥ 4/3} := by
sorry

end inequality_solution_set_l2501_250188


namespace words_lost_in_oz_l2501_250135

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 69

/-- The number of letters prohibited -/
def prohibited_letters : ℕ := 1

/-- The maximum word length -/
def max_word_length : ℕ := 2

/-- Calculate the number of words lost due to letter prohibition -/
def words_lost (alphabet_size : ℕ) (prohibited_letters : ℕ) (max_word_length : ℕ) : ℕ :=
  prohibited_letters + 
  (alphabet_size * prohibited_letters + alphabet_size * prohibited_letters - prohibited_letters * prohibited_letters)

theorem words_lost_in_oz : 
  words_lost alphabet_size prohibited_letters max_word_length = 138 := by
  sorry

end words_lost_in_oz_l2501_250135
