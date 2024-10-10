import Mathlib

namespace perfect_square_trinomial_n_l3942_394232

/-- 
A polynomial 4x^2 + 12x + n is a perfect square trinomial if and only if 
there exist real numbers a and b such that 4x^2 + 12x + n = (ax + b)^2 for all x
-/
def IsPerfectSquareTrinomial (n : ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, 4 * x^2 + 12 * x + n = (a * x + b)^2

/-- 
If 4x^2 + 12x + n is a perfect square trinomial, then n = 9
-/
theorem perfect_square_trinomial_n (n : ℝ) :
  IsPerfectSquareTrinomial n → n = 9 := by
  sorry


end perfect_square_trinomial_n_l3942_394232


namespace square_point_distance_probability_l3942_394216

-- Define the square
def Square := {p : ℝ × ℝ | (0 ≤ p.1 ∧ p.1 ≤ 2 ∧ (p.2 = 0 ∨ p.2 = 2)) ∨ (0 ≤ p.2 ∧ p.2 ≤ 2 ∧ (p.1 = 0 ∨ p.1 = 2))}

-- Define the probability function
noncomputable def probability : ℝ := sorry

-- Define the gcd function
def gcd (a b c : ℕ) : ℕ := sorry

-- State the theorem
theorem square_point_distance_probability :
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    gcd a b c = 1 ∧
    probability = (a - b * Real.pi) / c ∧
    a = 28 ∧ b = 1 ∧ c = 1 := by sorry

end square_point_distance_probability_l3942_394216


namespace value_of_X_l3942_394241

theorem value_of_X : ∀ M N X : ℕ,
  M = 2023 / 3 →
  N = M / 3 →
  X = M - N →
  X = 449 := by
sorry

end value_of_X_l3942_394241


namespace temperature_difference_l3942_394298

/-- Given the highest and lowest temperatures on a certain day in Xianning,
    prove that the temperature difference is 5°C. -/
theorem temperature_difference (lowest highest : ℝ) 
  (h_lowest : lowest = -3)
  (h_highest : highest = 2) :
  highest - lowest = 5 := by
  sorry

end temperature_difference_l3942_394298


namespace power_of_product_cubes_l3942_394250

theorem power_of_product_cubes : (3^3 * 4^3)^2 = 2985984 := by
  sorry

end power_of_product_cubes_l3942_394250


namespace distance_after_two_hours_l3942_394269

-- Define Aaron's speed
def aaron_speed : ℚ := 1 / 20

-- Define Mia's speed
def mia_speed : ℚ := 3 / 40

-- Define the time period in hours
def time_period : ℚ := 2

-- Define the direction multiplier (opposite directions)
def direction_multiplier : ℚ := 2

-- Theorem statement
theorem distance_after_two_hours :
  (aaron_speed * (time_period * 60) + mia_speed * (time_period * 60)) * direction_multiplier = 15 := by
  sorry

end distance_after_two_hours_l3942_394269


namespace pure_imaginary_complex_number_l3942_394217

theorem pure_imaginary_complex_number (a : ℝ) : 
  (Complex.I * (a - 2) = (a^2 - 3*a + 2) + Complex.I * (a - 2)) → a = 1 :=
by sorry

end pure_imaginary_complex_number_l3942_394217


namespace chimney_bricks_count_l3942_394295

/-- Represents the time taken by Brenda to build the chimney alone -/
def brenda_time : ℝ := 6

/-- Represents the time taken by Brandon to build the chimney alone -/
def brandon_time : ℝ := 8

/-- Represents the reduction in combined output when working together -/
def output_reduction : ℝ := 15

/-- Represents the time taken to build the chimney when working together -/
def combined_time : ℝ := 4

/-- Represents the number of bricks in the chimney -/
def chimney_bricks : ℝ := 360

theorem chimney_bricks_count :
  (combined_time * (chimney_bricks / brenda_time + chimney_bricks / brandon_time - output_reduction) = chimney_bricks) :=
sorry

end chimney_bricks_count_l3942_394295


namespace smallest_a_for_polynomial_l3942_394238

theorem smallest_a_for_polynomial (a b : ℤ) (r₁ r₂ r₃ : ℕ+) : 
  (∀ x : ℝ, x^3 - a*x^2 + b*x - 30030 = (x - r₁)*(x - r₂)*(x - r₃)) →
  a = r₁ + r₂ + r₃ →
  r₁ * r₂ * r₃ = 30030 →
  a ≥ 184 :=
sorry

end smallest_a_for_polynomial_l3942_394238


namespace electrician_wage_l3942_394277

/-- Given the following conditions:
  1. A bricklayer and an electrician worked for a total of 90 hours.
  2. The bricklayer's wage is $12 per hour.
  3. The total payment for both workers is $1350.
  4. The bricklayer worked for 67.5 hours.
Prove that the electrician's hourly wage is $24. -/
theorem electrician_wage (total_hours : ℝ) (bricklayer_wage : ℝ) (total_payment : ℝ) (bricklayer_hours : ℝ)
  (h1 : total_hours = 90)
  (h2 : bricklayer_wage = 12)
  (h3 : total_payment = 1350)
  (h4 : bricklayer_hours = 67.5) :
  (total_payment - bricklayer_wage * bricklayer_hours) / (total_hours - bricklayer_hours) = 24 := by
  sorry

end electrician_wage_l3942_394277


namespace neg_cos_double_angle_range_l3942_394285

theorem neg_cos_double_angle_range (θ : Real) : -1 ≤ -Real.cos (2 * θ) ∧ -Real.cos (2 * θ) ≤ 1 := by
  sorry

end neg_cos_double_angle_range_l3942_394285


namespace subset_with_sum_property_l3942_394203

theorem subset_with_sum_property (Y : Finset ℕ+) (n : ℕ) (hn : Y.card = n) :
  ∃ B : Finset ℕ+, B ⊆ Y ∧ B.card > n / 3 ∧
    ∀ u v : ℕ+, u ∈ B → v ∈ B → (u + v : ℕ+) ∉ B := by
  sorry

end subset_with_sum_property_l3942_394203


namespace probability_all_white_balls_l3942_394257

def total_balls : ℕ := 15
def white_balls : ℕ := 7
def black_balls : ℕ := 8
def drawn_balls : ℕ := 7

theorem probability_all_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls) = 1 / 6435 := by
  sorry

end probability_all_white_balls_l3942_394257


namespace total_distance_approx_l3942_394256

/-- The speed at which Tammy drove, in miles per hour -/
def speed : ℝ := 1.527777778

/-- The duration of Tammy's drive, in hours -/
def duration : ℝ := 36.0

/-- The total distance Tammy drove, in miles -/
def total_distance : ℝ := speed * duration

/-- Theorem stating that the total distance is approximately 55.0 miles -/
theorem total_distance_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |total_distance - 55.0| < ε :=
by
  sorry

end total_distance_approx_l3942_394256


namespace coffee_ounces_per_pot_l3942_394284

/-- Calculates the number of ounces per pot of coffee -/
def ounces_per_pot (ounces_per_donut : ℚ) (cost_per_pot : ℚ) (dozen_donuts : ℕ) (total_cost : ℚ) : ℚ :=
  let total_donuts := dozen_donuts * 12
  let total_ounces := total_donuts * ounces_per_donut
  let num_pots := total_cost / cost_per_pot
  total_ounces / num_pots

/-- Proves that the number of ounces per pot of coffee is 12 -/
theorem coffee_ounces_per_pot :
  ounces_per_pot 2 3 3 18 = 12 := by sorry

end coffee_ounces_per_pot_l3942_394284


namespace simplify_expression_l3942_394276

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^3 + b^3 = 3*(a + b)) : a/b + b/a - 3/(a*b) = 1 := by
  sorry

end simplify_expression_l3942_394276


namespace smallest_perfect_square_divisible_by_6_14_22_30_l3942_394255

theorem smallest_perfect_square_divisible_by_6_14_22_30 :
  ∃ (n : ℕ), n > 0 ∧ n = 5336100 ∧ 
  (∃ (k : ℕ), n = k^2) ∧
  6 ∣ n ∧ 14 ∣ n ∧ 22 ∣ n ∧ 30 ∣ n ∧
  (∀ (m : ℕ), m > 0 → (∃ (j : ℕ), m = j^2) → 
    6 ∣ m → 14 ∣ m → 22 ∣ m → 30 ∣ m → m ≥ n) :=
by sorry


end smallest_perfect_square_divisible_by_6_14_22_30_l3942_394255


namespace matthews_friends_l3942_394231

/-- Given that Matthew had 30 cakes and each person ate 15 cakes, 
    prove that the number of friends he shared with is 2. -/
theorem matthews_friends (total_cakes : ℕ) (cakes_per_person : ℕ) 
  (h1 : total_cakes = 30) 
  (h2 : cakes_per_person = 15) :
  total_cakes / cakes_per_person = 2 := by
  sorry

end matthews_friends_l3942_394231


namespace no_additional_salt_needed_l3942_394239

/-- Represents the problem of mixing salt to achieve a specific profit -/
def SaltMixtureProblem (initialSaltWeight : ℝ) (initialSaltCost : ℝ) (mixtureSalePrice : ℝ) (desiredProfit : ℝ) :=
  ∃ (additionalSaltWeight : ℝ),
    additionalSaltWeight ≥ 0 ∧
    let totalCost := initialSaltWeight * initialSaltCost + additionalSaltWeight * 0.5
    let totalWeight := initialSaltWeight + additionalSaltWeight
    let totalRevenue := totalWeight * mixtureSalePrice
    totalRevenue = (1 + desiredProfit) * totalCost

/-- The main theorem stating that no additional salt is needed for the given problem -/
theorem no_additional_salt_needed :
  SaltMixtureProblem 40 0.35 0.48 0.2 → ∃ (x : ℝ), x = 0 := by
  sorry

end no_additional_salt_needed_l3942_394239


namespace fresh_fruits_count_l3942_394261

/-- Calculates the number of fresh fruits left after sales and spoilage --/
def freshFruitsLeft (initialPineapples initialCoconuts soldPineapples soldCoconuts rottenPineapples spoiledCoconutPercentage : ℕ) : ℕ :=
  let remainingPineapples := initialPineapples - soldPineapples
  let freshPineapples := remainingPineapples - rottenPineapples
  let remainingCoconuts := initialCoconuts - soldCoconuts
  let spoiledCoconuts := (remainingCoconuts * spoiledCoconutPercentage + 99) / 100  -- Round up
  let freshCoconuts := remainingCoconuts - spoiledCoconuts
  freshPineapples + freshCoconuts

/-- Theorem stating that the total number of fresh pineapples and coconuts left is 92 --/
theorem fresh_fruits_count :
  freshFruitsLeft 120 75 52 38 11 5 = 92 := by
  sorry

end fresh_fruits_count_l3942_394261


namespace expand_polynomial_l3942_394243

theorem expand_polynomial (x : ℝ) : (3*x^2 + 7*x + 4) * (5*x - 2) = 15*x^3 + 29*x^2 + 6*x - 8 := by
  sorry

end expand_polynomial_l3942_394243


namespace delta_f_P0_approx_df_P0_l3942_394258

-- Define the function f
def f (x y : ℝ) : ℝ := x^2 * y

-- Define the point P0
def P0 : ℝ × ℝ := (5, 4)

-- Define Δx and Δy
def Δx : ℝ := 0.1
def Δy : ℝ := -0.2

-- Theorem for Δf(P0)
theorem delta_f_P0_approx : 
  let (x0, y0) := P0
  abs (f (x0 + Δx) (y0 + Δy) - f x0 y0 + 1.162) < 0.001 := by sorry

-- Theorem for df(P0)
theorem df_P0 : 
  let (x0, y0) := P0
  (2 * x0 * y0) * Δx + x0^2 * Δy = -1 := by sorry

end delta_f_P0_approx_df_P0_l3942_394258


namespace arithmetic_geometric_sequence_ratio_l3942_394246

/-- 
Given an arithmetic sequence with first term a₁ and non-zero common difference d,
if the 1st, 6th, and 21st terms form a geometric sequence,
then the common ratio of this geometric sequence is 3.
-/
theorem arithmetic_geometric_sequence_ratio 
  (a₁ : ℝ) (d : ℝ) (h : d ≠ 0) : 
  (a₁ + 5 * d)^2 = a₁ * (a₁ + 20 * d) → 
  (a₁ + 5 * d) / a₁ = 3 :=
by sorry

end arithmetic_geometric_sequence_ratio_l3942_394246


namespace abs_sum_minimum_l3942_394282

theorem abs_sum_minimum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 := by
  sorry

end abs_sum_minimum_l3942_394282


namespace min_distance_to_origin_l3942_394259

theorem min_distance_to_origin (x y : ℝ) : 
  3 * x + 4 * y = 24 → 
  x - 2 * y = 0 → 
  ∃ (min_dist : ℝ), 
    min_dist = Real.sqrt 28.8 ∧ 
    ∀ (x' y' : ℝ), 3 * x' + 4 * y' = 24 → x' - 2 * y' = 0 → 
      Real.sqrt (x' ^ 2 + y' ^ 2) ≥ min_dist := by
  sorry

end min_distance_to_origin_l3942_394259


namespace local_min_implies_a_range_l3942_394230

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 6*a*x + 3*a

-- Define what it means for f to have a local minimum in (0,1)
def has_local_min_in_interval (f : ℝ → ℝ) : Prop :=
  ∃ x, 0 < x ∧ x < 1 ∧ ∃ δ > 0, ∀ y, |y - x| < δ → f y ≥ f x

-- The theorem statement
theorem local_min_implies_a_range (a : ℝ) :
  has_local_min_in_interval (f a) → 0 < a ∧ a < 1/2 :=
sorry

end local_min_implies_a_range_l3942_394230


namespace annual_phone_bill_l3942_394227

def original_bill : ℚ := 50
def increase_rate : ℚ := 0.1
def months_per_year : ℕ := 12

theorem annual_phone_bill :
  (original_bill * (1 + increase_rate)) * months_per_year = 660 := by
  sorry

end annual_phone_bill_l3942_394227


namespace count_a_values_l3942_394251

theorem count_a_values : 
  ∃ (S : Finset Nat), 
    S.card = 9 ∧ 
    (∀ a ∈ S, 0 < a ∧ a < 100 ∧ (a^3 + 23) % 24 = 0) ∧
    (∀ a : Nat, 0 < a ∧ a < 100 ∧ (a^3 + 23) % 24 = 0 → a ∈ S) := by
  sorry

end count_a_values_l3942_394251


namespace baseball_team_groups_l3942_394234

theorem baseball_team_groups (new_players : ℕ) (returning_players : ℕ) (players_per_group : ℕ) :
  new_players = 48 →
  returning_players = 6 →
  players_per_group = 6 →
  (new_players + returning_players) / players_per_group = 9 :=
by sorry

end baseball_team_groups_l3942_394234


namespace orchid_to_rose_ratio_l3942_394254

/-- Proves that the ratio of orchids to roses in each centerpiece is 2:1 given the specified conditions. -/
theorem orchid_to_rose_ratio 
  (num_centerpieces : ℕ) 
  (roses_per_centerpiece : ℕ) 
  (lilies_per_centerpiece : ℕ) 
  (total_budget : ℕ) 
  (cost_per_flower : ℕ) 
  (h1 : num_centerpieces = 6)
  (h2 : roses_per_centerpiece = 8)
  (h3 : lilies_per_centerpiece = 6)
  (h4 : total_budget = 2700)
  (h5 : cost_per_flower = 15) : 
  ∃ (orchids_per_centerpiece : ℕ), 
    orchids_per_centerpiece = 2 * roses_per_centerpiece :=
by sorry

end orchid_to_rose_ratio_l3942_394254


namespace max_m_for_factorizable_quadratic_l3942_394297

/-- 
Given a quadratic expression 5x^2 + mx + 45 that can be factored as the product 
of two linear factors with integer coefficients, the maximum possible value of m is 226.
-/
theorem max_m_for_factorizable_quadratic : 
  ∀ m : ℤ, 
  (∃ A B : ℤ, 5*x^2 + m*x + 45 = (5*x + A)*(x + B)) → 
  m ≤ 226 :=
by sorry

end max_m_for_factorizable_quadratic_l3942_394297


namespace signal_count_l3942_394201

/-- Represents the three flag colors -/
inductive FlagColor
  | Red
  | Yellow
  | Blue

/-- Represents a signal as a list of flag colors -/
def Signal := List FlagColor

/-- Returns true if the signal is valid (contains 1, 2, or 3 flags) -/
def isValidSignal (s : Signal) : Bool :=
  1 ≤ s.length ∧ s.length ≤ 3

/-- Returns all possible valid signals -/
def allValidSignals : List Signal :=
  (List.map (λ c => [c]) [FlagColor.Red, FlagColor.Yellow, FlagColor.Blue]) ++
  (List.map (λ (c1, c2) => [c1, c2]) [(FlagColor.Red, FlagColor.Yellow), (FlagColor.Red, FlagColor.Blue), (FlagColor.Yellow, FlagColor.Red), (FlagColor.Yellow, FlagColor.Blue), (FlagColor.Blue, FlagColor.Red), (FlagColor.Blue, FlagColor.Yellow)]) ++
  (List.map (λ (c1, c2, c3) => [c1, c2, c3]) [(FlagColor.Red, FlagColor.Yellow, FlagColor.Blue), (FlagColor.Red, FlagColor.Blue, FlagColor.Yellow), (FlagColor.Yellow, FlagColor.Red, FlagColor.Blue), (FlagColor.Yellow, FlagColor.Blue, FlagColor.Red), (FlagColor.Blue, FlagColor.Red, FlagColor.Yellow), (FlagColor.Blue, FlagColor.Yellow, FlagColor.Red)])

theorem signal_count : (allValidSignals.filter isValidSignal).length = 15 := by
  sorry

end signal_count_l3942_394201


namespace smallest_sum_is_134_l3942_394279

def digits : List Nat := [5, 6, 7, 8]

def is_valid_arrangement (a b c d : Nat) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def sum_of_arrangement (a b c d : Nat) : Nat :=
  10 * a + b + 10 * c + d

theorem smallest_sum_is_134 :
  ∀ a b c d : Nat,
    is_valid_arrangement a b c d →
    sum_of_arrangement a b c d ≥ 134 :=
by sorry

end smallest_sum_is_134_l3942_394279


namespace greatest_integer_inequality_l3942_394289

theorem greatest_integer_inequality : ∀ y : ℤ, (5 : ℚ) / 8 > (y : ℚ) / 17 ↔ y ≤ 10 :=
by sorry

end greatest_integer_inequality_l3942_394289


namespace infinite_pairs_divisibility_property_l3942_394229

theorem infinite_pairs_divisibility_property (x : ℤ) (h : x ≥ 2) :
  let y := x * (x^8 - x^2 - 1)
  1 < x ∧ x < y ∧ (x^3 + y) ∣ (x + y^3) :=
by sorry

end infinite_pairs_divisibility_property_l3942_394229


namespace convexity_inequality_equality_conditions_l3942_394273

theorem convexity_inequality (x y a b : ℝ) 
  (h1 : a + b = 1) 
  (h2 : a ≥ 0) 
  (h3 : b ≥ 0) : 
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 := by
  sorry

theorem equality_conditions (x y a b : ℝ) 
  (h1 : a + b = 1) 
  (h2 : a ≥ 0) 
  (h3 : b ≥ 0) :
  (a * x + b * y)^2 = a * x^2 + b * y^2 ↔ (a = 0 ∨ b = 0 ∨ x = y) := by
  sorry

end convexity_inequality_equality_conditions_l3942_394273


namespace seed_germination_problem_l3942_394204

theorem seed_germination_problem (seeds_plot1 seeds_plot2 : ℕ) 
  (germination_rate_plot2 total_germination_rate : ℚ) :
  seeds_plot1 = 500 →
  seeds_plot2 = 200 →
  germination_rate_plot2 = 1/2 →
  total_germination_rate = 35714285714285715/100000000000000000 →
  (↑(seeds_plot1 * 3/10) + ↑(seeds_plot2 * germination_rate_plot2)) / 
   ↑(seeds_plot1 + seeds_plot2) = total_germination_rate :=
by sorry

end seed_germination_problem_l3942_394204


namespace circle_properties_l3942_394292

-- Define the circle C
def C : Set (ℝ × ℝ) := sorry

-- Define the endpoints of the diameter
def endpoint1 : ℝ × ℝ := (3, -2)
def endpoint2 : ℝ × ℝ := (-9, 4)

-- Define the center of the circle
def center : ℝ × ℝ := ((-3 : ℝ), (1 : ℝ))

-- Define the point to be checked
def point : ℝ × ℝ := (0, 1)

-- Theorem statement
theorem circle_properties :
  (center = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2)) ∧
  (point ∉ C) := by
  sorry

end circle_properties_l3942_394292


namespace custom_operation_solution_l3942_394283

-- Define the custom operation *
def star (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem custom_operation_solution :
  ∀ x : ℝ, star 3 x = 31 → x = 7 := by
  sorry

end custom_operation_solution_l3942_394283


namespace floor_length_calculation_l3942_394236

theorem floor_length_calculation (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  length = 3 * breadth →
  area = length * breadth →
  area = 80 →
  length = 3 * Real.sqrt (80 / 3) := by
sorry

end floor_length_calculation_l3942_394236


namespace solution_satisfies_system_l3942_394245

theorem solution_satisfies_system :
  let x : ℚ := 43 / 9
  let y : ℚ := 16 / 3
  (3 * x - 4 * y = -7) ∧ (6 * x - 5 * y = 2) := by
sorry

end solution_satisfies_system_l3942_394245


namespace number_where_one_seventh_is_five_l3942_394210

theorem number_where_one_seventh_is_five : 
  ∃ n : ℝ, (1 / 7 : ℝ) * n = 5 → n = 35 :=
by sorry

end number_where_one_seventh_is_five_l3942_394210


namespace triangle_inequality_l3942_394200

theorem triangle_inequality (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  (a = 4 ∧ b = 9) → (5 < x ∧ x < 13) ↔ (a + b > x ∧ a + x > b ∧ b + x > a) :=
sorry

end triangle_inequality_l3942_394200


namespace right_triangle_area_l3942_394220

theorem right_triangle_area (r R : ℝ) (h : r > 0) (h' : R > 0) : ∃ (A : ℝ), 
  A > 0 ∧ 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (a^2 + b^2 = c^2) ∧  -- right triangle condition
    (A = (a * b) / 2) ∧  -- area of triangle
    (r = A / ((a + b + c) / 2)) ∧  -- inradius formula
    (R = c / 2) ∧  -- circumradius formula for right triangle
    (A = r * (2 * R + r))) :=
by sorry

end right_triangle_area_l3942_394220


namespace range_of_m_l3942_394293

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x > 1 → (x^2 + 3) / (x - 1) > m^2 + 1) → 
  -Real.sqrt 5 < m ∧ m < Real.sqrt 5 :=
by sorry

end range_of_m_l3942_394293


namespace quadratic_inequality_solution_condition_l3942_394224

theorem quadratic_inequality_solution_condition (d : ℝ) :
  d > 0 ∧ (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ 0 < d ∧ d < 16 := by
  sorry

end quadratic_inequality_solution_condition_l3942_394224


namespace sum_in_base4_l3942_394264

/-- Converts a base 4 number to base 10 -/
def base4ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4^i)) 0

/-- Converts a base 10 number to base 4 -/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 4) ((m % 4) :: acc)
  aux n []

theorem sum_in_base4 :
  let a := [2, 1, 3]  -- 312₄ in reverse order
  let b := [1, 3]     -- 31₄ in reverse order
  let c := [3]        -- 3₄
  let sum := [2, 1, 0, 1]  -- 1012₄ in reverse order
  base10ToBase4 (base4ToBase10 a + base4ToBase10 b + base4ToBase10 c) = sum := by
  sorry

end sum_in_base4_l3942_394264


namespace q_div_p_equals_550_l3942_394274

def total_slips : ℕ := 60
def numbers_per_slip : ℕ := 12
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 5

def p : ℚ := (numbers_per_slip : ℚ) / (Nat.choose total_slips drawn_slips)

def q : ℚ := (Nat.choose numbers_per_slip 2 * Nat.choose slips_per_number 3 * Nat.choose slips_per_number 2 : ℚ) / (Nat.choose total_slips drawn_slips)

theorem q_div_p_equals_550 : q / p = 550 := by sorry

end q_div_p_equals_550_l3942_394274


namespace louise_boxes_needed_louise_needs_23_boxes_l3942_394262

/-- Represents the number of pencils a box can hold for each color --/
structure BoxCapacity where
  red : Nat
  blue : Nat
  yellow : Nat
  green : Nat

/-- Represents the number of pencils Louise has for each color --/
structure PencilCount where
  red : Nat
  blue : Nat
  yellow : Nat
  green : Nat

/-- Calculates the number of boxes needed for a given color --/
def boxesNeeded (pencils : Nat) (capacity : Nat) : Nat :=
  (pencils + capacity - 1) / capacity

/-- Theorem: Given the specified conditions, Louise needs 23 boxes in total --/
theorem louise_boxes_needed (capacity : BoxCapacity) (count : PencilCount) : Nat :=
  have red_boxes := boxesNeeded count.red capacity.red
  have blue_boxes := boxesNeeded count.blue capacity.blue
  have yellow_boxes := boxesNeeded count.yellow capacity.yellow
  have green_boxes := boxesNeeded count.green capacity.green
  red_boxes + blue_boxes + yellow_boxes + green_boxes

/-- Main theorem: Louise needs 23 boxes given the specific conditions --/
theorem louise_needs_23_boxes : 
  let capacity := BoxCapacity.mk 15 25 10 30
  let count := PencilCount.mk 45 (3 * 45) 80 (45 + 3 * 45)
  louise_boxes_needed capacity count = 23 := by
  sorry

end louise_boxes_needed_louise_needs_23_boxes_l3942_394262


namespace molecular_weight_ccl4_l3942_394271

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of chlorine in g/mol -/
def chlorine_weight : ℝ := 35.45

/-- The number of carbon atoms in a molecule of carbon tetrachloride -/
def carbon_atoms : ℕ := 1

/-- The number of chlorine atoms in a molecule of carbon tetrachloride -/
def chlorine_atoms : ℕ := 4

/-- The number of moles of carbon tetrachloride -/
def moles : ℕ := 9

/-- The molecular weight of carbon tetrachloride in g/mol -/
def ccl4_weight : ℝ := carbon_weight * carbon_atoms + chlorine_weight * chlorine_atoms

/-- Theorem stating the molecular weight of 9 moles of carbon tetrachloride -/
theorem molecular_weight_ccl4 : 
  (ccl4_weight * moles : ℝ) = 1384.29 := by sorry

end molecular_weight_ccl4_l3942_394271


namespace robert_photos_count_l3942_394291

/-- The number of photos taken by Claire -/
def claire_photos : ℕ := 8

/-- The additional number of photos taken by Robert compared to Claire -/
def robert_extra_photos : ℕ := 16

/-- The number of photos taken by Robert -/
def robert_photos : ℕ := claire_photos + robert_extra_photos

/-- Theorem: Robert has taken 24 photos -/
theorem robert_photos_count : robert_photos = 24 := by
  sorry

end robert_photos_count_l3942_394291


namespace initial_birds_count_l3942_394242

theorem initial_birds_count (storks : ℕ) (additional_birds : ℕ) (bird_stork_difference : ℕ) :
  storks = 5 →
  additional_birds = 4 →
  bird_stork_difference = 2 →
  ∃ initial_birds : ℕ, 
    initial_birds + additional_birds = storks + bird_stork_difference ∧
    initial_birds = 3 :=
by sorry

end initial_birds_count_l3942_394242


namespace sqrt_three_irrational_l3942_394228

theorem sqrt_three_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_three_irrational_l3942_394228


namespace exists_equal_shift_eval_l3942_394233

-- Define the type for polynomials of degree 2014
def Poly2014 := Polynomial ℝ

-- Define what it means for a polynomial to be monic of degree 2014
def is_monic_2014 (p : Poly2014) : Prop :=
  p.degree = 2014 ∧ p.leadingCoeff = 1

-- Define the theorem
theorem exists_equal_shift_eval
  (P Q : Poly2014)
  (h_monic_P : is_monic_2014 P)
  (h_monic_Q : is_monic_2014 Q)
  (h_not_equal : ∀ x : ℝ, P.eval x ≠ Q.eval x) :
  ∃ x : ℝ, P.eval (x - 1) = Q.eval (x + 1) := by
sorry

end exists_equal_shift_eval_l3942_394233


namespace exponent_problem_l3942_394208

theorem exponent_problem (a : ℝ) (m n : ℤ) (h1 : a^m = 5) (h2 : a^n = 2) :
  a^(m-2*n) = 5/4 := by sorry

end exponent_problem_l3942_394208


namespace custom_mul_four_three_l3942_394213

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) : ℝ := a^2 + a*b - b^2

/-- Theorem stating that 4 * 3 = 19 under the custom multiplication -/
theorem custom_mul_four_three : custom_mul 4 3 = 19 := by sorry

end custom_mul_four_three_l3942_394213


namespace probability_of_composite_product_l3942_394214

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 6

/-- The set of prime numbers that can appear on an 8-sided die -/
def primes_on_die : Set ℕ := {2, 3, 5, 7}

/-- The total number of possible outcomes when rolling 6 8-sided dice -/
def total_outcomes : ℕ := sides ^ num_dice

/-- The number of ways to roll all 1's -/
def all_ones : ℕ := 1

/-- The number of ways to roll five 1's and one prime on a 6-sided die -/
def five_ones_one_prime : ℕ := 4 * 6

/-- The total number of favorable outcomes (product is 1 or prime) -/
def favorable_outcomes : ℕ := all_ones + five_ones_one_prime

/-- The probability that the product is composite -/
def prob_composite : ℚ := 1 - (favorable_outcomes : ℚ) / total_outcomes

theorem probability_of_composite_product :
  prob_composite = 262119 / 262144 := by sorry

end probability_of_composite_product_l3942_394214


namespace inverse_contrapositive_equivalence_l3942_394294

theorem inverse_contrapositive_equivalence (a b c : ℝ) :
  (¬(a > b) → ¬(a + c > b + c)) ↔ (a + c ≤ b + c → a ≤ b) := by sorry

end inverse_contrapositive_equivalence_l3942_394294


namespace binomial_expansion_problem_l3942_394267

theorem binomial_expansion_problem (p q : ℝ) : 
  p > 0 → q > 0 → p + q = 1 → 
  55 * p^9 * q^2 = 165 * p^8 * q^3 → 
  p = 3/4 := by
sorry

end binomial_expansion_problem_l3942_394267


namespace third_day_sales_l3942_394218

/-- Represents the sales of a clothing store over three days -/
structure ClothingSales where
  /-- Number of pieces sold on the first day -/
  first_day : ℕ
  /-- Number of pieces sold on the second day -/
  second_day : ℕ
  /-- Number of pieces sold on the third day -/
  third_day : ℕ

/-- Theorem stating the relationship between sales on different days -/
theorem third_day_sales (a : ℕ) (sales : ClothingSales) 
  (h1 : sales.first_day = a)
  (h2 : sales.second_day = sales.first_day + 4)
  (h3 : sales.third_day = 2 * sales.second_day - 7) :
  sales.third_day = 2 * a + 1 := by
  sorry


end third_day_sales_l3942_394218


namespace nested_rectangles_l3942_394248

/-- A rectangle with integer side lengths -/
structure Rectangle where
  width : Nat
  height : Nat
  width_pos : width > 0
  height_pos : height > 0
  width_bound : width ≤ 100
  height_bound : height ≤ 100

/-- Predicate to check if one rectangle fits inside another -/
def fits_inside (r1 r2 : Rectangle) : Prop :=
  (r1.width ≤ r2.width ∧ r1.height ≤ r2.height) ∨
  (r1.width ≤ r2.height ∧ r1.height ≤ r2.width)

/-- Main theorem: Given 101 rectangles, there exist 3 that fit inside each other -/
theorem nested_rectangles (rectangles : Finset Rectangle) 
    (h : rectangles.card = 101) : 
    ∃ (A B C : Rectangle), A ∈ rectangles ∧ B ∈ rectangles ∧ C ∈ rectangles ∧
    fits_inside A B ∧ fits_inside B C := by
  sorry

end nested_rectangles_l3942_394248


namespace no_real_solutions_l3942_394280

theorem no_real_solutions : ¬∃ (x : ℝ), 3 * x^2 + 5 = |4 * x + 2| - 3 := by
  sorry

end no_real_solutions_l3942_394280


namespace complex_modulus_problem_l3942_394240

theorem complex_modulus_problem (z : ℂ) (i : ℂ) : 
  i^2 = -1 → z = (1 - i) / (1 + i) → Complex.abs z = 1 := by sorry

end complex_modulus_problem_l3942_394240


namespace inequality_relationship_l3942_394247

theorem inequality_relationship (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) :
  a^2 > -a*b ∧ -a*b > b^2 := by
  sorry

end inequality_relationship_l3942_394247


namespace nilpotent_matrices_l3942_394226

open Matrix

theorem nilpotent_matrices (n : ℕ) (A B : Matrix (Fin n) (Fin n) ℝ) 
  (t : Fin (n+1) → ℝ) (h_distinct : ∀ i j, i ≠ j → t i ≠ t j) :
  (∀ i : Fin (n+1), ∃ k : ℕ, (A + t i • B) ^ k = 0) →
  (∃ k₁ : ℕ, A ^ k₁ = 0) ∧ (∃ k₂ : ℕ, B ^ k₂ = 0) := by
  sorry

end nilpotent_matrices_l3942_394226


namespace cubic_roots_cosine_relation_l3942_394296

theorem cubic_roots_cosine_relation (p q r : ℝ) :
  (∃ α β γ : ℝ, α > 0 ∧ β > 0 ∧ γ > 0 ∧
    (∀ x : ℝ, x^3 + p*x^2 + q*x + r = 0 ↔ x = α ∨ x = β ∨ x = γ) ∧
    (∃ θ₁ θ₂ θ₃ : ℝ, θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧ θ₁ + θ₂ + θ₃ = π ∧
      α = Real.cos θ₁ ∧ β = Real.cos θ₂ ∧ γ = Real.cos θ₃)) →
  2*r + 1 = p^2 - 2*q :=
sorry

end cubic_roots_cosine_relation_l3942_394296


namespace negation_of_existence_proposition_l3942_394219

theorem negation_of_existence_proposition :
  ¬(∃ a : ℝ, ∃ x : ℝ, a * x^2 + 1 = 0) ↔
  (∀ a : ℝ, ∀ x : ℝ, a * x^2 + 1 ≠ 0) :=
by sorry

end negation_of_existence_proposition_l3942_394219


namespace committee_meeting_attendance_committee_meeting_attendance_proof_l3942_394205

/-- Proves that the total number of people present in the committee meeting is 7 -/
theorem committee_meeting_attendance : ℕ → ℕ → Prop :=
  fun (associate_profs assistant_profs : ℕ) =>
    -- Each associate professor brings 2 pencils and 1 chart
    -- Each assistant professor brings 1 pencil and 2 charts
    -- Total of 10 pencils and 11 charts brought to the meeting
    (2 * associate_profs + assistant_profs = 10) ∧
    (associate_profs + 2 * assistant_profs = 11) →
    -- The total number of people present is 7
    associate_profs + assistant_profs = 7

theorem committee_meeting_attendance_proof : ∃ (a b : ℕ), committee_meeting_attendance a b :=
  sorry

end committee_meeting_attendance_committee_meeting_attendance_proof_l3942_394205


namespace max_value_sum_of_fractions_l3942_394281

theorem max_value_sum_of_fractions (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + 2*b + 3*c = 1) : 
  (∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x + 2*y + 3*z = 1 ∧ 
    1/(2*x + y) + 1/(2*y + z) + 1/(2*z + x) > 1/(2*a + b) + 1/(2*b + c) + 1/(2*c + a)) ∨
  (1/(2*a + b) + 1/(2*b + c) + 1/(2*c + a) = 7) :=
sorry

end max_value_sum_of_fractions_l3942_394281


namespace factorial_ratio_52_50_l3942_394235

theorem factorial_ratio_52_50 : Nat.factorial 52 / Nat.factorial 50 = 2652 := by
  sorry

end factorial_ratio_52_50_l3942_394235


namespace power_five_mod_eleven_l3942_394265

theorem power_five_mod_eleven : 5^120 + 4 ≡ 5 [MOD 11] := by
  sorry

end power_five_mod_eleven_l3942_394265


namespace largest_divisor_of_expression_l3942_394223

theorem largest_divisor_of_expression (x : ℤ) (h : Odd x) :
  (40 : ℤ) = Nat.gcd 40 ((12*x + 2)*(8*x + 14)*(10*x + 10)).natAbs := by
  sorry

end largest_divisor_of_expression_l3942_394223


namespace min_voters_for_tall_to_win_l3942_394249

/-- Represents the voting structure and outcome of the giraffe beauty contest -/
structure GiraffeContest where
  total_voters : Nat
  num_districts : Nat
  sections_per_district : Nat
  voters_per_section : Nat
  winner_name : String

/-- Calculates the minimum number of voters required for a giraffe to win the contest -/
def min_voters_to_win (contest : GiraffeContest) : Nat :=
  sorry

/-- Theorem stating the minimum number of voters required for Tall to win the contest -/
theorem min_voters_for_tall_to_win (contest : GiraffeContest) 
  (h1 : contest.total_voters = 105)
  (h2 : contest.num_districts = 5)
  (h3 : contest.sections_per_district = 7)
  (h4 : contest.voters_per_section = 3)
  (h5 : contest.winner_name = "Tall") :
  min_voters_to_win contest = 24 := by
  sorry

end min_voters_for_tall_to_win_l3942_394249


namespace largest_power_dividing_factorial_l3942_394244

theorem largest_power_dividing_factorial (n : ℕ) (h : n = 7 * 17^2) :
  ∃ k : ℕ, k = 63 ∧ 
  (∀ m : ℕ, n^m ∣ n.factorial → m ≤ k) ∧
  n^k ∣ n.factorial :=
sorry

end largest_power_dividing_factorial_l3942_394244


namespace class_size_l3942_394275

theorem class_size (n : ℕ) (h1 : n > 0) :
  (∃ student_in_middle_row : ℕ, 
    student_in_middle_row > 0 ∧ 
    student_in_middle_row ≤ n ∧
    student_in_middle_row = 6 ∧ 
    n + 1 - student_in_middle_row = 7) →
  3 * n = 36 :=
by sorry

end class_size_l3942_394275


namespace distance_between_parallel_lines_l3942_394268

/-- The distance between two parallel lines -/
theorem distance_between_parallel_lines 
  (a : ℝ) -- Coefficient of x in the second line
  (h_parallel : a = 6) -- Condition for parallelism
  : (|(-24) - 11|) / Real.sqrt (3^2 + 4^2) = 7/2 := by
  sorry

end distance_between_parallel_lines_l3942_394268


namespace square_root_to_cube_l3942_394290

theorem square_root_to_cube (x : ℝ) : 
  Real.sqrt (2 * x + 4) = 4 → (2 * x + 4)^3 = 4096 := by sorry

end square_root_to_cube_l3942_394290


namespace count_perfect_square_factors_8820_l3942_394299

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem count_perfect_square_factors_8820 :
  let factorization := prime_factorization 8820
  factorization = [(2, 2), (3, 2), (5, 1), (7, 2)] →
  count_perfect_square_factors 8820 = 8 := by sorry

end count_perfect_square_factors_8820_l3942_394299


namespace valid_square_configurations_l3942_394266

/-- Represents a configuration of a 5x7 grid --/
structure GridConfiguration where
  squares : ℕ  -- number of 2x2 squares
  strips : ℕ   -- number of 1x3 strips
  corners : ℕ  -- number of three-cell corners

/-- Checks if a configuration is valid for a 5x7 grid --/
def isValidConfiguration (config : GridConfiguration) : Prop :=
  4 * config.squares + 3 * config.strips + 3 * config.corners = 35

/-- The theorem stating the valid configurations for 2x2 squares --/
theorem valid_square_configurations :
  ∀ (config : GridConfiguration),
    isValidConfiguration config →
    (config.squares = 5 ∨ config.squares = 2) :=
  sorry

end valid_square_configurations_l3942_394266


namespace thin_mints_price_l3942_394209

theorem thin_mints_price (samoas_price : ℝ) (fudge_delights_price : ℝ) (sugar_cookies_price : ℝ)
  (samoas_quantity : ℕ) (thin_mints_quantity : ℕ) (fudge_delights_quantity : ℕ) (sugar_cookies_quantity : ℕ)
  (total_earned : ℝ) :
  samoas_price = 4 →
  fudge_delights_price = 5 →
  sugar_cookies_price = 2 →
  samoas_quantity = 3 →
  thin_mints_quantity = 2 →
  fudge_delights_quantity = 1 →
  sugar_cookies_quantity = 9 →
  total_earned = 42 →
  (total_earned - (samoas_price * samoas_quantity + fudge_delights_price * fudge_delights_quantity + sugar_cookies_price * sugar_cookies_quantity)) / thin_mints_quantity = 3.5 := by
sorry

end thin_mints_price_l3942_394209


namespace quadratic_factorization_l3942_394207

theorem quadratic_factorization (m : ℝ) : m^2 - 14*m + 49 = (m - 7)^2 := by
  sorry

end quadratic_factorization_l3942_394207


namespace student_council_committees_l3942_394221

theorem student_council_committees (n : ℕ) : 
  n * (n - 1) / 2 = 28 → (n.choose 4) = 70 := by
  sorry

end student_council_committees_l3942_394221


namespace customer_flow_solution_l3942_394206

/-- Represents the customer flow in a restaurant --/
def customer_flow (x y z : ℕ) : Prop :=
  let initial_customers : ℕ := 3
  let final_customers : ℕ := 8
  x = 2 * z ∧
  y = x - 3 ∧
  initial_customers + x + y - z = final_customers

/-- Theorem stating the solution to the customer flow problem --/
theorem customer_flow_solution :
  customer_flow 6 3 3 ∧ 6 + 3 = 9 := by
  sorry

#check customer_flow_solution

end customer_flow_solution_l3942_394206


namespace adam_remaining_candy_l3942_394215

/-- The number of boxes of chocolate candy Adam initially bought -/
def initial_boxes : ℕ := 13

/-- The number of boxes Adam gave to his little brother -/
def given_boxes : ℕ := 7

/-- The number of pieces of candy in each box -/
def pieces_per_box : ℕ := 6

/-- Theorem: Adam still had 36 pieces of chocolate candy -/
theorem adam_remaining_candy : 
  (initial_boxes - given_boxes) * pieces_per_box = 36 := by
  sorry

end adam_remaining_candy_l3942_394215


namespace bookstore_comparison_l3942_394260

/-- Represents the amount to be paid at Bookstore A -/
def bookstore_A (x : ℝ) : ℝ := 0.8 * x

/-- Represents the amount to be paid at Bookstore B -/
def bookstore_B (x : ℝ) : ℝ := 0.6 * x + 40

theorem bookstore_comparison (x : ℝ) (h : x > 100) :
  (bookstore_A x < bookstore_B x ↔ x < 200) ∧
  (bookstore_A x > bookstore_B x ↔ x > 200) ∧
  (bookstore_A x = bookstore_B x ↔ x = 200) := by
  sorry

#check bookstore_comparison

end bookstore_comparison_l3942_394260


namespace calculation_proof_l3942_394202

theorem calculation_proof : 4 * 6 * 8 + 24 / 4 + 2 = 200 := by
  sorry

end calculation_proof_l3942_394202


namespace clock_equivalent_hours_l3942_394270

theorem clock_equivalent_hours : ∃ (n : ℕ), n > 3 ∧ 
  (∀ k : ℕ, k > 3 ∧ k < n → ¬(12 ∣ (k^2 - k))) ∧ 
  (12 ∣ (n^2 - n)) := by
  sorry

end clock_equivalent_hours_l3942_394270


namespace sequence_conditions_diamonds_in_G20_l3942_394222

/-- The number of diamonds in figure n of the sequence -/
def num_diamonds (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 9
  else 4 * n^2 + 4 * n - 7

/-- The sequence satisfies the given conditions -/
theorem sequence_conditions (n : ℕ) (h : n ≥ 3) :
  num_diamonds n = num_diamonds (n-1) + 8 * n :=
sorry

/-- The number of diamonds in G₂₀ is 1673 -/
theorem diamonds_in_G20 : num_diamonds 20 = 1673 :=
sorry

end sequence_conditions_diamonds_in_G20_l3942_394222


namespace negative_sqrt_three_is_quadratic_radical_l3942_394252

-- Define what a quadratic radical is
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ a : ℝ, a ≥ 0 ∧ x = Real.sqrt a ∨ x = -Real.sqrt a

-- Theorem statement
theorem negative_sqrt_three_is_quadratic_radical :
  is_quadratic_radical (-Real.sqrt 3) := by
  sorry

end negative_sqrt_three_is_quadratic_radical_l3942_394252


namespace candy_distribution_l3942_394286

theorem candy_distribution (n : ℕ) (total_candies : ℕ) : 
  total_candies = 120 →
  total_candies = 2 * n →
  n = 60 := by
sorry

end candy_distribution_l3942_394286


namespace ternary_to_decimal_l3942_394212

/-- 
Given a positive integer k, if the ternary number 10k2 is equal to 35 in decimal, 
then k is equal to 2.
-/
theorem ternary_to_decimal (k : ℕ+) : 
  (1 * 3^3 + k * 3 + 2 = 35) → k = 2 := by
  sorry

end ternary_to_decimal_l3942_394212


namespace complex_sum_theorem_l3942_394287

theorem complex_sum_theorem (A B C D : ℂ) : 
  A = 3 + 2*I → B = -5 → C = 1 - 2*I → D = 3 + 5*I → 
  A + B + C + D = 2 + 5*I :=
by sorry

end complex_sum_theorem_l3942_394287


namespace sports_competition_solution_l3942_394211

/-- Represents the number of medals distributed on day k --/
def medals_distributed (k : ℕ) (m_k : ℕ) : ℕ :=
  k + (m_k - k) / 7

/-- Represents the number of medals remaining after day k --/
def medals_remaining (k : ℕ) (m_k : ℕ) : ℕ :=
  m_k - medals_distributed k m_k

/-- The sports competition problem --/
theorem sports_competition_solution (n m : ℕ) : 
  (n > 1) →
  (∀ k, k ∈ Finset.range n → medals_distributed k (medals_remaining (k-1) m) = medals_distributed (k+1) (medals_remaining k m)) →
  (medals_distributed n (medals_remaining (n-1) m) = n) →
  (n = 6 ∧ m = 36) :=
by sorry

end sports_competition_solution_l3942_394211


namespace max_attachable_squares_l3942_394225

/-- A unit square in 2D space -/
structure UnitSquare where
  center : ℝ × ℝ

/-- Represents the configuration of unit squares attached to a central square -/
structure SquareConfiguration where
  central : UnitSquare
  attached : List UnitSquare

/-- Checks if two unit squares overlap -/
def squaresOverlap (s1 s2 : UnitSquare) : Prop := sorry

/-- Checks if a configuration is valid (no overlaps) -/
def isValidConfiguration (config : SquareConfiguration) : Prop := sorry

/-- The main theorem: maximum number of attachable squares is 8 -/
theorem max_attachable_squares (K : UnitSquare) :
  (∃ (config : SquareConfiguration),
    config.central = K ∧
    isValidConfiguration config ∧
    config.attached.length = 8) ∧
  (∀ (config : SquareConfiguration),
    config.central = K →
    isValidConfiguration config →
    config.attached.length ≤ 8) := by sorry

end max_attachable_squares_l3942_394225


namespace quadratic_perfect_square_l3942_394278

theorem quadratic_perfect_square (a : ℚ) :
  (∃ r s : ℚ, ∀ x, a * x^2 + 20 * x + 9 = (r * x + s)^2) →
  a = 100 / 9 := by
sorry

end quadratic_perfect_square_l3942_394278


namespace vitamin_d_scientific_notation_l3942_394237

theorem vitamin_d_scientific_notation : 0.0000046 = 4.6 * 10^(-6) := by
  sorry

end vitamin_d_scientific_notation_l3942_394237


namespace expression_evaluation_l3942_394263

theorem expression_evaluation : 
  (1728^2 : ℚ) / (137^3 - (137^2 - 11^2)) = 2985984 / 2552705 := by sorry

end expression_evaluation_l3942_394263


namespace percentage_reduction_l3942_394253

theorem percentage_reduction (P : ℝ) : (200 * (P / 100)) - 12 = 178 → P = 95 := by
  sorry

end percentage_reduction_l3942_394253


namespace three_config_m_separable_l3942_394272

/-- A 3-configuration of a set is m-separable if it can be partitioned into m subsets
    such that no three elements of the configuration are in the same subset. -/
def is_m_separable (A : Set α) (m : ℕ) : Prop :=
  ∃ (f : α → Fin m), ∀ (x y z : α), x ∈ A → y ∈ A → z ∈ A →
    x ≠ y → y ≠ z → x ≠ z → f x ≠ f y ∨ f y ≠ f z ∨ f x ≠ f z

/-- A 3-configuration of a set A is a subset of A with exactly 3 elements. -/
def is_3_configuration (S : Set α) (A : Set α) : Prop :=
  S ⊆ A ∧ S.ncard = 3

theorem three_config_m_separable
  (A : Set α) (n m : ℕ) (h_card : A.ncard = n) (h_m : m ≥ n / 2) :
  ∀ S : Set α, is_3_configuration S A → is_m_separable S m :=
sorry

end three_config_m_separable_l3942_394272


namespace solutions_for_x_l3942_394288

theorem solutions_for_x : ∃ (x₁ x₂ x₃ : ℝ),
  ((x₁ + 1)^2 = 36 ∨ (x₁ + 10)^3 = -27) ∧
  ((x₂ + 1)^2 = 36 ∨ (x₂ + 10)^3 = -27) ∧
  ((x₃ + 1)^2 = 36 ∨ (x₃ + 10)^3 = -27) ∧
  x₁ = 5 ∧ x₂ = -7 ∧ x₃ = -13 :=
by sorry

end solutions_for_x_l3942_394288
