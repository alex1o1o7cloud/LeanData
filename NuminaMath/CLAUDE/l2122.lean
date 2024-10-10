import Mathlib

namespace garden_area_l2122_212238

theorem garden_area (width : ℝ) (length : ℝ) (perimeter : ℝ) :
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  width * length = 243 := by
sorry

end garden_area_l2122_212238


namespace eight_chickens_ten_eggs_l2122_212290

/-- Given that 5 chickens lay 7 eggs in 4 days, this function calculates
    the number of days it takes for 8 chickens to lay 10 eggs. -/
def days_to_lay_eggs (initial_chickens : ℕ) (initial_eggs : ℕ) (initial_days : ℕ)
                     (target_chickens : ℕ) (target_eggs : ℕ) : ℚ :=
  (initial_chickens * initial_days * target_eggs : ℚ) /
  (initial_eggs * target_chickens : ℚ)

/-- Theorem stating that 8 chickens will take 50/7 days to lay 10 eggs,
    given that 5 chickens lay 7 eggs in 4 days. -/
theorem eight_chickens_ten_eggs :
  days_to_lay_eggs 5 7 4 8 10 = 50 / 7 := by
  sorry

#eval days_to_lay_eggs 5 7 4 8 10

end eight_chickens_ten_eggs_l2122_212290


namespace anne_solo_time_l2122_212261

-- Define the cleaning rates
def bruce_rate : ℝ := sorry
def anne_rate : ℝ := sorry

-- Define the conditions
axiom clean_together : bruce_rate + anne_rate = 1 / 4
axiom clean_anne_double : bruce_rate + 2 * anne_rate = 1 / 3

-- Theorem to prove
theorem anne_solo_time : 1 / anne_rate = 12 := by sorry

end anne_solo_time_l2122_212261


namespace water_displaced_squared_5ft_l2122_212228

/-- The volume of water displaced by a fully submerged cube -/
def water_displaced (cube_side : ℝ) : ℝ := cube_side ^ 3

/-- The square of the volume of water displaced by a fully submerged cube -/
def water_displaced_squared (cube_side : ℝ) : ℝ := (water_displaced cube_side) ^ 2

/-- Theorem: The square of the volume of water displaced by a fully submerged cube
    with side length 5 feet is equal to 15625 (cubic feet)^2 -/
theorem water_displaced_squared_5ft :
  water_displaced_squared 5 = 15625 := by
  sorry

end water_displaced_squared_5ft_l2122_212228


namespace river_depth_l2122_212268

/-- The depth of a river given its width, flow rate, and volume of water per minute -/
theorem river_depth (width : ℝ) (flow_rate : ℝ) (volume_per_minute : ℝ) : 
  width = 65 →
  flow_rate = 6 →
  volume_per_minute = 26000 →
  (width * (flow_rate * 1000 / 60) * 4 = volume_per_minute) := by sorry

end river_depth_l2122_212268


namespace repeating_decimal_sum_difference_l2122_212205

/-- The sum of 0.666... (repeating) and 0.222... (repeating) minus 0.444... (repeating) equals 4/9 -/
theorem repeating_decimal_sum_difference (x y z : ℚ) :
  x = 2/3 ∧ y = 2/9 ∧ z = 4/9 →
  x + y - z = 4/9 := by
  sorry

end repeating_decimal_sum_difference_l2122_212205


namespace complex_multiplication_l2122_212297

theorem complex_multiplication (i : ℂ) : i * i = -1 → i * (1 + i) = -1 + i := by
  sorry

end complex_multiplication_l2122_212297


namespace smallest_nth_root_of_unity_l2122_212272

theorem smallest_nth_root_of_unity : ∃ (n : ℕ), n > 0 ∧ 
  (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 18 := by
sorry

end smallest_nth_root_of_unity_l2122_212272


namespace solution_part1_solution_part2_l2122_212250

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1
theorem solution_part1 (a : ℝ) (h : a ≤ 2) :
  {x : ℝ | f a x ≥ 2} = {x : ℝ | x ≤ 1/2 ∨ x ≥ 5/2} :=
sorry

-- Part 2
theorem solution_part2 (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, f a x + |x - 1| ≥ 1) → a ≥ 2 :=
sorry

end solution_part1_solution_part2_l2122_212250


namespace sufficient_not_necessary_condition_l2122_212264

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 → a + b > 0) ∧
  (∃ a b : ℝ, a + b > 0 ∧ ¬(a > 0 ∧ b > 0)) := by
  sorry

end sufficient_not_necessary_condition_l2122_212264


namespace isabel_candy_count_l2122_212274

/-- Calculates the total number of candy pieces Isabel has -/
def total_candy (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem: Given Isabel's initial candy count and the additional pieces she received,
    prove that her total candy count is 93 -/
theorem isabel_candy_count :
  let initial := 68
  let additional := 25
  total_candy initial additional = 93 := by
  sorry

end isabel_candy_count_l2122_212274


namespace exists_fixed_point_with_iteration_l2122_212259

/-- A function satisfying the given conditions -/
def SatisfyingFunction (f : ℕ → ℕ) : Prop :=
  (∀ x, 1 ≤ f x - x ∧ f x - x ≤ 2019) ∧
  (∀ x, f (f x) % 2019 = x % 2019)

/-- The main theorem -/
theorem exists_fixed_point_with_iteration (f : ℕ → ℕ) (h : SatisfyingFunction f) :
  ∃ x, ∀ k, f^[k] x = x + 2019 * k :=
sorry

end exists_fixed_point_with_iteration_l2122_212259


namespace no_real_roots_quadratic_l2122_212293

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 + x - k ≠ 0) ↔ k < -1/8 := by
  sorry

end no_real_roots_quadratic_l2122_212293


namespace not_in_range_iff_a_in_interval_l2122_212239

/-- The function g(x) = x^2 + ax + 3 -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- Theorem: -3 is not in the range of g(x) if and only if a is in the open interval (-√24, √24) -/
theorem not_in_range_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, g a x ≠ -3) ↔ a > -Real.sqrt 24 ∧ a < Real.sqrt 24 := by
  sorry

end not_in_range_iff_a_in_interval_l2122_212239


namespace imaginary_part_of_z_l2122_212200

theorem imaginary_part_of_z : Complex.im ((1 + Complex.I)^2 + Complex.I^2011) = 1 := by
  sorry

end imaginary_part_of_z_l2122_212200


namespace binomial_inequality_l2122_212202

theorem binomial_inequality (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n) :
  (n^n : ℚ) / (m^m * (n-m)^(n-m)) > (n.factorial : ℚ) / (m.factorial * (n-m).factorial) ∧
  (n.factorial : ℚ) / (m.factorial * (n-m).factorial) > (n^n : ℚ) / (m^m * (n+1) * (n-m)^(n-m)) :=
by sorry

end binomial_inequality_l2122_212202


namespace min_value_fraction_l2122_212299

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 2) : 
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 2 → (x + y) / (x * y * z) ≤ (a + b) / (a * b * c)) →
  (x + y) / (x * y * z) = 4 :=
by sorry

end min_value_fraction_l2122_212299


namespace least_four_digit_solution_l2122_212251

theorem least_four_digit_solution (x : ℕ) : x = 1002 ↔ 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 10 [ZMOD 10] ∧
     3 * y + 20 ≡ 29 [ZMOD 12] ∧
     -3 * y + 2 ≡ 2 * y [ZMOD 30]) →
    x ≤ y) ∧
  (5 * x ≡ 10 [ZMOD 10]) ∧
  (3 * x + 20 ≡ 29 [ZMOD 12]) ∧
  (-3 * x + 2 ≡ 2 * x [ZMOD 30]) := by
sorry

end least_four_digit_solution_l2122_212251


namespace even_swaps_not_restore_order_l2122_212269

/-- A permutation of integers from 1 to n -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The identity permutation (ascending order) -/
def id_perm (n : ℕ) : Permutation n := fun i => i

/-- Swap two elements in a permutation -/
def swap (p : Permutation n) (i j : Fin n) : Permutation n :=
  fun k => if k = i then p j else if k = j then p i else p k

/-- Apply a sequence of swaps to a permutation -/
def apply_swaps (p : Permutation n) (swaps : List (Fin n × Fin n)) : Permutation n :=
  swaps.foldl (fun p' (i, j) => swap p' i j) p

/-- The main theorem -/
theorem even_swaps_not_restore_order (n : ℕ) (swaps : List (Fin n × Fin n)) :
  swaps.length % 2 = 0 → apply_swaps (id_perm n) swaps ≠ id_perm n :=
sorry

end even_swaps_not_restore_order_l2122_212269


namespace PB_equation_l2122_212220

-- Define the points A, B, and P
variable (A B P : ℝ × ℝ)

-- Define the conditions
axiom A_on_x_axis : A.2 = 0
axiom B_on_x_axis : B.2 = 0
axiom P_x_coord : P.1 = 2
axiom PA_PB_equal : (P.1 - A.1)^2 + (P.2 - A.2)^2 = (P.1 - B.1)^2 + (P.2 - B.2)^2
axiom PA_equation : ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1 - p.2 + 1 = 0} ↔ (x - P.1) * (A.2 - P.2) = (y - P.2) * (A.1 - P.1)

-- State the theorem
theorem PB_equation :
  ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | p.1 + p.2 - 5 = 0} ↔ (x - P.1) * (B.2 - P.2) = (y - P.2) * (B.1 - P.1) :=
sorry

end PB_equation_l2122_212220


namespace volleyball_team_starters_l2122_212254

-- Define the total number of players
def total_players : ℕ := 16

-- Define the number of quadruplets
def quadruplets : ℕ := 4

-- Define the number of starters to choose
def starters : ℕ := 6

-- Define the maximum number of quadruplets allowed in the starting lineup
def max_quadruplets_in_lineup : ℕ := 1

-- Theorem statement
theorem volleyball_team_starters :
  (Nat.choose (total_players - quadruplets) starters) +
  (quadruplets * Nat.choose (total_players - quadruplets) (starters - 1)) = 4092 := by
  sorry

end volleyball_team_starters_l2122_212254


namespace harry_sea_stars_harry_collected_34_sea_stars_l2122_212236

theorem harry_sea_stars : ℕ → Prop :=
  fun sea_stars =>
    sea_stars + 21 + 29 = 59 + 25 ∧ 
    sea_stars = 34

/-- Proof that Harry collected 34 sea stars initially -/
theorem harry_collected_34_sea_stars : ∃ (sea_stars : ℕ), harry_sea_stars sea_stars :=
by
  sorry

end harry_sea_stars_harry_collected_34_sea_stars_l2122_212236


namespace product_change_l2122_212267

theorem product_change (a b : ℝ) (h : a * b = 1620) :
  (4 * a) * (b / 2) = 3240 := by
  sorry

end product_change_l2122_212267


namespace function_periodicity_l2122_212287

def is_periodic (f : ℕ → ℕ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, f (n + p) = f n

theorem function_periodicity 
  (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, f (n + f n) = f n) 
  (h2 : Set.Finite (Set.range f)) : 
  is_periodic f := by
  sorry

end function_periodicity_l2122_212287


namespace solutions_correct_l2122_212279

-- Define the systems of equations
def system1 (x y : ℝ) : Prop :=
  3 * x + y = 4 ∧ 3 * x + 2 * y = 6

def system2 (x y : ℝ) : Prop :=
  2 * x + y = 3 ∧ 3 * x - 5 * y = 11

-- State the theorem
theorem solutions_correct :
  (∃ x y : ℝ, system1 x y ∧ x = 2/3 ∧ y = 2) ∧
  (∃ x y : ℝ, system2 x y ∧ x = 2 ∧ y = -1) :=
by sorry

end solutions_correct_l2122_212279


namespace sharp_composition_72_l2122_212278

-- Define the # operation
def sharp (N : ℝ) : ℝ := 0.5 * N + 2

-- State the theorem
theorem sharp_composition_72 : sharp (sharp (sharp 72)) = 12.5 := by
  sorry

end sharp_composition_72_l2122_212278


namespace manager_percentage_problem_l2122_212219

theorem manager_percentage_problem (total_employees : ℕ) 
  (managers_left : ℕ) (final_percentage : ℚ) :
  total_employees = 500 →
  managers_left = 250 →
  final_percentage = 98/100 →
  (total_employees - managers_left) * final_percentage = 
    total_employees - managers_left - 
    ((100 - 99)/100 * total_employees) →
  99/100 * total_employees = total_employees - 
    ((100 - 99)/100 * total_employees) :=
by sorry

end manager_percentage_problem_l2122_212219


namespace no_outliers_l2122_212263

def data_set : List ℝ := [2, 11, 23, 23, 25, 35, 41, 41, 55, 67, 85]
def Q1 : ℝ := 23
def Q2 : ℝ := 35
def Q3 : ℝ := 55

def is_outlier (x : ℝ) : Prop :=
  let IQR := Q3 - Q1
  x < Q1 - 2 * IQR ∨ x > Q3 + 2 * IQR

theorem no_outliers : ∀ x ∈ data_set, ¬(is_outlier x) := by sorry

end no_outliers_l2122_212263


namespace slope_angle_of_line_l2122_212262

theorem slope_angle_of_line (x y : ℝ) (α : ℝ) :
  x * Real.sin (2 * π / 5) + y * Real.cos (2 * π / 5) = 0 →
  α = 3 * π / 5 := by
  sorry

end slope_angle_of_line_l2122_212262


namespace scientific_notation_of_2310000_l2122_212270

/-- Proves that 2,310,000 is equal to 2.31 × 10^6 in scientific notation -/
theorem scientific_notation_of_2310000 : 
  2310000 = 2.31 * (10 : ℝ)^6 := by
  sorry

end scientific_notation_of_2310000_l2122_212270


namespace alternating_color_probability_alternating_color_probability_value_l2122_212206

/-- The probability of drawing 10 balls with alternating colors (starting and ending with the same color) from a box containing 5 white and 5 black balls. -/
theorem alternating_color_probability : ℚ :=
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let successful_sequences : ℕ := 2
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  successful_sequences / total_arrangements

/-- The probability of drawing 10 balls with alternating colors (starting and ending with the same color) from a box containing 5 white and 5 black balls is 1/126. -/
theorem alternating_color_probability_value : alternating_color_probability = 1 / 126 := by
  sorry

end alternating_color_probability_alternating_color_probability_value_l2122_212206


namespace coin_distribution_proof_l2122_212255

/-- Represents the coin distribution scheme between Charlie and Fred -/
def coin_distribution (x : ℕ) : Prop :=
  -- Charlie's coins are the sum of 1 to x
  let charlie_coins := x * (x + 1) / 2
  -- Fred's coins are x at the end
  let fred_coins := x
  -- Charlie has 5 times as many coins as Fred
  charlie_coins = 5 * fred_coins

/-- The total number of coins after distribution -/
def total_coins (x : ℕ) : ℕ := x * 6

theorem coin_distribution_proof :
  ∃ x : ℕ, coin_distribution x ∧ total_coins x = 54 :=
sorry

end coin_distribution_proof_l2122_212255


namespace first_degree_function_determination_l2122_212213

-- Define a first-degree function
def FirstDegreeFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x + b

-- State the theorem
theorem first_degree_function_determination
  (f : ℝ → ℝ)
  (h1 : FirstDegreeFunction f)
  (h2 : 2 * f 2 - 3 * f 1 = 5)
  (h3 : 2 * f 0 - f (-1) = 1) :
  ∀ x, f x = 3 * x - 2 :=
sorry

end first_degree_function_determination_l2122_212213


namespace square_ends_with_self_l2122_212218

theorem square_ends_with_self (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) → (A^2 ≡ A [ZMOD 1000]) ↔ (A = 376 ∨ A = 625) := by
  sorry

end square_ends_with_self_l2122_212218


namespace imaginary_part_of_complex_division_l2122_212208

theorem imaginary_part_of_complex_division : 
  let z : ℂ := 1 / (2 + Complex.I)
  Complex.im z = -1 / 5 := by
  sorry

end imaginary_part_of_complex_division_l2122_212208


namespace committee_selection_ways_l2122_212227

-- Define the total number of team owners
def total_owners : ℕ := 30

-- Define the number of owners who don't want to serve
def ineligible_owners : ℕ := 3

-- Define the size of the committee
def committee_size : ℕ := 5

-- Define the number of eligible owners
def eligible_owners : ℕ := total_owners - ineligible_owners

-- Define the combination function
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Theorem statement
theorem committee_selection_ways : 
  combination eligible_owners committee_size = 65780 := by
  sorry

end committee_selection_ways_l2122_212227


namespace expression_value_l2122_212224

theorem expression_value (x y : ℝ) (h : x - 2 * y^2 = 1) :
  -2 * x + 4 * y^2 + 1 = -1 := by sorry

end expression_value_l2122_212224


namespace phone_watch_sales_l2122_212209

/-- Represents the total sales amount for two months of phone watch sales -/
def total_sales (x : ℕ) : ℝ := 600 * 60 + 500 * (x - 60)

/-- States that the total sales amount is no less than $86000 -/
def sales_condition (x : ℕ) : Prop := total_sales x ≥ 86000

theorem phone_watch_sales (x : ℕ) : 
  sales_condition x ↔ 600 * 60 + 500 * (x - 60) ≥ 86000 := by sorry

end phone_watch_sales_l2122_212209


namespace inequality_proof_l2122_212276

theorem inequality_proof (a b : ℝ) (n : ℕ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : 1/a + 1/b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end inequality_proof_l2122_212276


namespace find_N_l2122_212217

theorem find_N : ∃ N : ℝ, (0.2 * N = 0.3 * 2500) ∧ (N = 3750) := by
  sorry

end find_N_l2122_212217


namespace regular_polygon_150_degrees_has_12_sides_l2122_212275

/-- Proves that a regular polygon with interior angles of 150 degrees has 12 sides -/
theorem regular_polygon_150_degrees_has_12_sides :
  ∀ n : ℕ, 
    n > 2 →
    (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
    n = 12 :=
by
  sorry

#check regular_polygon_150_degrees_has_12_sides

end regular_polygon_150_degrees_has_12_sides_l2122_212275


namespace ad_campaign_cost_l2122_212237

/-- Calculates the total cost of an ad campaign with given parameters and discount rules --/
theorem ad_campaign_cost 
  (page_width : ℝ) 
  (page_height : ℝ) 
  (full_page_rate : ℝ) 
  (half_page_rate : ℝ) 
  (quarter_page_rate : ℝ) 
  (eighth_page_rate : ℝ) 
  (half_page_count : ℕ) 
  (quarter_page_count : ℕ) 
  (eighth_page_count : ℕ) 
  (discount_rate_4_to_5 : ℝ) 
  (discount_rate_6_or_more : ℝ) : 
  page_width = 9 → 
  page_height = 12 → 
  full_page_rate = 6.5 → 
  half_page_rate = 8 → 
  quarter_page_rate = 10 → 
  eighth_page_rate = 12 → 
  half_page_count = 1 → 
  quarter_page_count = 3 → 
  eighth_page_count = 4 → 
  discount_rate_4_to_5 = 0.1 → 
  discount_rate_6_or_more = 0.15 → 
  ∃ (total_cost : ℝ), total_cost = 1606.5 := by
  sorry


end ad_campaign_cost_l2122_212237


namespace extreme_value_condition_l2122_212257

theorem extreme_value_condition (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (a * x^2 - 1) * Real.exp x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 1/3 := by
sorry

end extreme_value_condition_l2122_212257


namespace point_on_parametric_line_l2122_212245

/-- A line in 2D space defined by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given a point P(2,4) on a line defined by x = 1 + t and y = 3 - at, 
    where t is a parameter, the value of a must be -1 -/
theorem point_on_parametric_line (P : Point) (l : ParametricLine) (a : ℝ) :
  P.x = 2 ∧ P.y = 4 ∧
  (∃ t : ℝ, l.x t = 1 + t ∧ l.y t = 3 - a * t) ∧
  (∃ t : ℝ, P.x = l.x t ∧ P.y = l.y t) →
  a = -1 := by
  sorry

#check point_on_parametric_line

end point_on_parametric_line_l2122_212245


namespace sum_of_max_and_min_is_eight_l2122_212235

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (12 - x^4) + x^2) / x^3 + 4

def X : Set ℝ := {x | x ∈ Set.Icc (-1) 0 ∪ Set.Ioc 0 1}

theorem sum_of_max_and_min_is_eight :
  ∃ (A B : ℝ), (∀ x ∈ X, f x ≤ A) ∧ 
               (∃ x ∈ X, f x = A) ∧ 
               (∀ x ∈ X, B ≤ f x) ∧ 
               (∃ x ∈ X, f x = B) ∧ 
               A + B = 8 := by
  sorry

end sum_of_max_and_min_is_eight_l2122_212235


namespace sqrt_three_plus_two_range_l2122_212252

theorem sqrt_three_plus_two_range :
  ∃ (x : ℝ), x = Real.sqrt 3 ∧ Irrational x ∧ 1 < x ∧ x < 2 → 3.5 < x + 2 ∧ x + 2 < 4 := by
  sorry

end sqrt_three_plus_two_range_l2122_212252


namespace a_integer_not_multiple_of_five_l2122_212225

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 6*x + 1 = 0

-- Define the sequence aₙ
def a (n : ℕ) (x₁ x₂ : ℝ) : ℝ := x₁^n + x₂^n

-- State the theorem
theorem a_integer_not_multiple_of_five 
  (x₁ x₂ : ℝ) 
  (h₁ : quadratic_equation x₁) 
  (h₂ : quadratic_equation x₂) :
  ∀ n : ℕ, ∃ k : ℤ, (a n x₁ x₂ = k) ∧ ¬(∃ m : ℤ, k = 5 * m) :=
by sorry

end a_integer_not_multiple_of_five_l2122_212225


namespace quadratic_function_properties_l2122_212226

-- Define the quadratic function
def f (t x : ℝ) : ℝ := x^2 - 2*t*x + 3

-- State the theorem
theorem quadratic_function_properties (t : ℝ) (h_t : t > 0) :
  -- Part 1
  (f t 2 = 1 → t = 3/2) ∧
  -- Part 2
  (∃ (x_min : ℝ), 0 ≤ x_min ∧ x_min ≤ 3 ∧
    (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f t x ≥ f t x_min) ∧
    f t x_min = -2 → t = Real.sqrt 5) ∧
  -- Part 3
  (∀ (m a b : ℝ),
    f t (m - 2) = a ∧ f t 4 = b ∧ f t m = a ∧ a < b ∧ b < 3 →
    (3 < m ∧ m < 4) ∨ m > 6) :=
by sorry

end quadratic_function_properties_l2122_212226


namespace perfect_square_divisibility_l2122_212258

theorem perfect_square_divisibility (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) :
  ∃ k : ℕ, a = k^2 := by
sorry

end perfect_square_divisibility_l2122_212258


namespace min_value_of_f_l2122_212260

/-- The quadratic function we're minimizing -/
def f (x y : ℝ) : ℝ := 3*x^2 + 4*x*y + 2*y^2 - 6*x + 8*y + 9

theorem min_value_of_f :
  (∀ x y : ℝ, f x y ≥ -15) ∧ (∃ x y : ℝ, f x y = -15) := by sorry

end min_value_of_f_l2122_212260


namespace no_90_cents_possible_l2122_212294

/-- Represents the types of coins available --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents --/
def coinValue : Coin → Nat
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a selection of coins --/
structure CoinSelection :=
  (pennies : Nat)
  (nickels : Nat)
  (dimes : Nat)
  (quarters : Nat)

/-- Checks if a coin selection is valid according to the problem constraints --/
def isValidSelection (s : CoinSelection) : Prop :=
  s.pennies + s.nickels + s.dimes + s.quarters = 6 ∧
  s.pennies ≤ 4 ∧ s.nickels ≤ 4 ∧ s.dimes ≤ 4 ∧ s.quarters ≤ 4

/-- Calculates the total value of a coin selection in cents --/
def totalValue (s : CoinSelection) : Nat :=
  s.pennies * coinValue Coin.Penny +
  s.nickels * coinValue Coin.Nickel +
  s.dimes * coinValue Coin.Dime +
  s.quarters * coinValue Coin.Quarter

/-- Theorem stating that it's impossible to make 90 cents with a valid coin selection --/
theorem no_90_cents_possible :
  ¬∃ (s : CoinSelection), isValidSelection s ∧ totalValue s = 90 := by
  sorry


end no_90_cents_possible_l2122_212294


namespace farmer_apples_l2122_212249

/-- The number of apples the farmer has after giving some away -/
def remaining_apples (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem: The farmer has 4337 apples after giving away 3588 from his initial 7925 apples -/
theorem farmer_apples : remaining_apples 7925 3588 = 4337 := by
  sorry

end farmer_apples_l2122_212249


namespace f_monotonicity_and_roots_l2122_212265

noncomputable section

variable (a : ℝ)

def f (x : ℝ) : ℝ := 2 * x + 1 - Real.exp (a * x)

theorem f_monotonicity_and_roots :
  (∀ x y : ℝ, x < y → a ≤ 0 → f a x < f a y) ∧
  (a > 0 →
    (∀ x y : ℝ, x < y → x < (1/a) * Real.log (2/a) → f a x < f a y) ∧
    (∀ x y : ℝ, x < y → x > (1/a) * Real.log (2/a) → f a x > f a y)) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f a x₁ = 1 → f a x₂ = 1 → x₁ + x₂ > 2/a) :=
by sorry

end

end f_monotonicity_and_roots_l2122_212265


namespace gcd_lcm_product_l2122_212240

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 210) (h2 : b = 4620) :
  (Nat.gcd a b) * (3 * Nat.lcm a b) = 2910600 := by
  sorry

end gcd_lcm_product_l2122_212240


namespace parabola_properties_l2122_212284

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = -4*x

-- Define the focus
def Focus : ℝ × ℝ := (-1, 0)

-- Define the line passing through the focus with slope 45°
def Line (x y : ℝ) : Prop := y = x + 1

-- Define the chord length
def ChordLength : ℝ := 8

theorem parabola_properties :
  -- The parabola passes through (-2, 2√2)
  Parabola (-2) (2 * Real.sqrt 2) ∧
  -- The focus is at (-1, 0)
  Focus = (-1, 0) ∧
  -- The chord formed by the intersection of the parabola and the line has length 8
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    Parabola x₁ y₁ ∧ Parabola x₂ y₂ ∧
    Line x₁ y₁ ∧ Line x₂ y₂ ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = ChordLength :=
by sorry

end parabola_properties_l2122_212284


namespace isosceles_trapezoid_ratio_l2122_212241

/-- An isosceles trapezoid with a point inside dividing it into four triangles -/
structure IsoscelesTrapezoidWithPoint where
  -- The lengths of the parallel bases
  AB : ℝ
  CD : ℝ
  -- Areas of the four triangles formed by the point
  area_PAB : ℝ
  area_PBC : ℝ
  area_PCD : ℝ
  area_PDA : ℝ
  -- Conditions
  AB_gt_CD : AB > CD
  areas_clockwise : area_PAB = 9 ∧ area_PBC = 7 ∧ area_PCD = 3 ∧ area_PDA = 5

/-- The ratio of the parallel bases in the isosceles trapezoid is 3 -/
theorem isosceles_trapezoid_ratio 
  (T : IsoscelesTrapezoidWithPoint) : T.AB / T.CD = 3 := by
  sorry

end isosceles_trapezoid_ratio_l2122_212241


namespace isosceles_triangle_base_length_l2122_212203

/-- An isosceles triangle with given perimeter and leg length -/
structure IsoscelesTriangle where
  perimeter : ℝ
  leg_length : ℝ

/-- The base length of an isosceles triangle -/
def base_length (t : IsoscelesTriangle) : ℝ :=
  t.perimeter - 2 * t.leg_length

/-- Theorem: The base length of an isosceles triangle with perimeter 62 and leg length 25 is 12 -/
theorem isosceles_triangle_base_length :
  let t : IsoscelesTriangle := { perimeter := 62, leg_length := 25 }
  base_length t = 12 := by
  sorry

end isosceles_triangle_base_length_l2122_212203


namespace geometric_sequence_minimum_value_l2122_212231

theorem geometric_sequence_minimum_value (a b c : ℝ) : 
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- a, b, c form a geometric sequence
  (∀ x : ℝ, (x - 2) * Real.exp x ≥ b) →  -- b is the minimum value of (x-2)e^x
  a * c = Real.exp 2 := by
sorry

end geometric_sequence_minimum_value_l2122_212231


namespace female_officers_count_l2122_212295

/-- Proves the total number of female officers on a police force given certain conditions -/
theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_percent : ℚ) :
  total_on_duty = 152 →
  female_on_duty_percent = 19 / 100 →
  ∃ (total_female : ℕ),
    total_female = 400 ∧
    (total_female : ℚ) * female_on_duty_percent = total_on_duty / 2 :=
by sorry

end female_officers_count_l2122_212295


namespace geometric_sequence_common_ratio_l2122_212229

/-- A geometric sequence with first term a₁ and common ratio q. -/
def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ :=
  λ n => a₁ * q^(n - 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (h_geom : ∃ (a₁ q : ℝ), ∀ n, a n = geometric_sequence a₁ q n)
  (h_a₁ : a 1 = 2) (h_a₄ : a 4 = 16) :
  ∃ q, ∀ n, a n = geometric_sequence 2 q n ∧ q = 2 := by
sorry

end geometric_sequence_common_ratio_l2122_212229


namespace sufficient_not_necessary_l2122_212256

-- Define the set M
def M (k : ℝ) : Set ℝ := {x : ℝ | |x| > k}

-- Define the statement
theorem sufficient_not_necessary (k : ℝ) :
  (k = 2 → 2 ∈ (M k)ᶜ) ∧ (∃ k', k' ≠ 2 ∧ 2 ∈ (M k')ᶜ) := by
  sorry

end sufficient_not_necessary_l2122_212256


namespace inscribed_circle_radius_l2122_212298

/-- Given a square ABCD with side length 1, E is the midpoint of AB, 
    F is the intersection of ED and AC, and G is the intersection of EC and BD. 
    The radius r of the circle inscribed in quadrilateral EFPG is equal to |EF| - |FP|. -/
theorem inscribed_circle_radius (A B C D E F G P : ℝ × ℝ) (r : ℝ) : 
  A = (0, 1) →
  B = (1, 1) →
  C = (1, 0) →
  D = (0, 0) →
  E = (1/2, 1) →
  F = (0, 1) →
  G = (2/3, 2/3) →
  P = (1/2, 1/2) →
  r = |EF| - |FP| :=
by sorry

end inscribed_circle_radius_l2122_212298


namespace team_win_percentage_l2122_212285

theorem team_win_percentage (total_games : ℕ) (win_rate : ℚ) : 
  total_games = 75 → win_rate = 65/100 → 
  (win_rate * total_games) / total_games = 65/100 := by
  sorry

end team_win_percentage_l2122_212285


namespace set_equality_implies_sum_l2122_212296

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = ({a^2, a+b, 0} : Set ℝ) → 
  a^2019 + b^2018 = -1 := by
sorry

end set_equality_implies_sum_l2122_212296


namespace andrew_stamps_hundred_permits_l2122_212273

/-- Calculates the number of permits Andrew stamps in a day -/
def permits_stamped (num_appointments : ℕ) (appointment_duration : ℕ) (workday_hours : ℕ) (stamps_per_hour : ℕ) : ℕ :=
  let appointment_time := num_appointments * appointment_duration
  let stamping_time := workday_hours - appointment_time
  stamping_time * stamps_per_hour

/-- Proves that Andrew stamps 100 permits given the specified conditions -/
theorem andrew_stamps_hundred_permits :
  permits_stamped 2 3 8 50 = 100 := by
  sorry

end andrew_stamps_hundred_permits_l2122_212273


namespace sum_r_s_equals_48_l2122_212201

/-- Parabola P with equation y = x^2 + 4x + 4 -/
def P : ℝ → ℝ := λ x => x^2 + 4*x + 4

/-- Point Q (10, 24) -/
def Q : ℝ × ℝ := (10, 24)

/-- Line through Q with slope m -/
def line_through_Q (m : ℝ) : ℝ → ℝ := λ x => m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, P x ≠ line_through_Q m x

/-- Theorem: Sum of r and s equals 48 -/
theorem sum_r_s_equals_48 (r s : ℝ) 
  (h : ∀ m, no_intersection m ↔ r < m ∧ m < s) : 
  r + s = 48 := by sorry

end sum_r_s_equals_48_l2122_212201


namespace volleyball_lineup_count_l2122_212244

/-- Represents the number of ways to choose a starting lineup for a volleyball team -/
def volleyballLineupCount (totalPlayers : ℕ) (versatilePlayers : ℕ) (specializedPlayers : ℕ) : ℕ :=
  totalPlayers * (totalPlayers - 1) * versatilePlayers * (versatilePlayers - 1) * (versatilePlayers - 2)

/-- Theorem stating the number of ways to choose a starting lineup for a volleyball team with given conditions -/
theorem volleyball_lineup_count :
  volleyballLineupCount 10 8 2 = 30240 :=
by sorry

end volleyball_lineup_count_l2122_212244


namespace complement_of_intersection_l2122_212246

def U : Set Nat := {1,2,3,4,5}
def A : Set Nat := {1,2,3}
def B : Set Nat := {2,3,4}

theorem complement_of_intersection (U A B : Set Nat) 
  (hU : U = {1,2,3,4,5}) 
  (hA : A = {1,2,3}) 
  (hB : B = {2,3,4}) : 
  (A ∩ B)ᶜ = {1,4,5} := by
  sorry

end complement_of_intersection_l2122_212246


namespace truncated_cone_inscribed_sphere_l2122_212215

/-- Given a truncated cone with an inscribed sphere, this theorem relates the ratio of their volumes
    to the angle between the generatrix and the base of the cone, and specifies the allowable values for the ratio. -/
theorem truncated_cone_inscribed_sphere (k : ℝ) (α : ℝ) :
  k > (3/2) →
  (∃ (V_cone V_sphere : ℝ), V_cone > 0 ∧ V_sphere > 0 ∧ V_cone / V_sphere = k) →
  α = Real.arctan (2 / Real.sqrt (2 * k - 3)) ∧
  α = angle_between_generatrix_and_base :=
by sorry

/-- Defines the angle between the generatrix and the base of the truncated cone. -/
def angle_between_generatrix_and_base : ℝ :=
sorry

end truncated_cone_inscribed_sphere_l2122_212215


namespace geometric_sequence_condition_l2122_212234

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n ≤ a (n + 1)

theorem geometric_sequence_condition (a : ℕ → ℝ) (q : ℝ) :
  (geometric_sequence a q) →
  (¬(((a 1 * q > 0) → increasing_sequence a) ∧
     (increasing_sequence a → (a 1 * q > 0)))) := by
  sorry

end geometric_sequence_condition_l2122_212234


namespace system_solution_l2122_212277

theorem system_solution :
  ∃ (x y : ℚ), 
    (3 * (x + y) - 4 * (x - y) = 5) ∧
    ((x + y) / 2 + (x - y) / 6 = 0) ∧
    (x = -1/3) ∧ (y = 2/3) := by
  sorry

end system_solution_l2122_212277


namespace rope_length_for_second_post_l2122_212281

theorem rope_length_for_second_post
  (total_rope : ℕ)
  (first_post : ℕ)
  (third_post : ℕ)
  (fourth_post : ℕ)
  (h1 : total_rope = 70)
  (h2 : first_post = 24)
  (h3 : third_post = 14)
  (h4 : fourth_post = 12) :
  total_rope - (first_post + third_post + fourth_post) = 20 := by
  sorry

#check rope_length_for_second_post

end rope_length_for_second_post_l2122_212281


namespace player2_is_best_l2122_212216

structure Player where
  id : Nat
  average_time : ℝ
  variance : ℝ

def players : List Player := [
  { id := 1, average_time := 51, variance := 3.5 },
  { id := 2, average_time := 50, variance := 3.5 },
  { id := 3, average_time := 51, variance := 14.5 },
  { id := 4, average_time := 50, variance := 14.4 }
]

def is_better_performer (p1 p2 : Player) : Prop :=
  p1.average_time < p2.average_time ∨ 
  (p1.average_time = p2.average_time ∧ p1.variance < p2.variance)

theorem player2_is_best : 
  ∀ p ∈ players, p.id ≠ 2 → is_better_performer (players[1]) p :=
sorry

end player2_is_best_l2122_212216


namespace smallest_divisible_by_one_to_ten_l2122_212212

theorem smallest_divisible_by_one_to_ten : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, k ≤ 10 → k > 0 → n % k = 0) ∧ (∀ m : ℕ, m > 0 → (∀ k : ℕ, k ≤ 10 → k > 0 → m % k = 0) → m ≥ 2520) :=
by sorry

end smallest_divisible_by_one_to_ten_l2122_212212


namespace technician_round_trip_completion_l2122_212222

theorem technician_round_trip_completion (distance : ℝ) (h : distance > 0) :
  let one_way := distance
  let round_trip := 2 * distance
  let completed := distance + 0.2 * distance
  (completed / round_trip) * 100 = 60 := by
sorry

end technician_round_trip_completion_l2122_212222


namespace two_digit_number_twice_product_of_digits_l2122_212271

theorem two_digit_number_twice_product_of_digits : ∃! n : ℕ, 
  10 ≤ n ∧ n < 100 ∧ n = 2 * (n / 10) * (n % 10) :=
by
  -- The proof would go here
  sorry

end two_digit_number_twice_product_of_digits_l2122_212271


namespace expression_evaluation_l2122_212207

theorem expression_evaluation :
  let f (x : ℚ) := ((x + 1) / (x - 1) - 1) * ((x + 1) / (x - 1) + 1)
  f (-1/2) = -8/9 := by
  sorry

end expression_evaluation_l2122_212207


namespace fourth_number_ninth_row_l2122_212221

/-- Represents the lattice structure with the given pattern -/
def lattice_sequence (row : ℕ) (position : ℕ) : ℕ :=
  8 * (row - 1) + position

/-- The problem statement -/
theorem fourth_number_ninth_row :
  lattice_sequence 9 4 = 68 := by
  sorry

end fourth_number_ninth_row_l2122_212221


namespace third_number_in_sequence_l2122_212283

theorem third_number_in_sequence (x : ℕ) 
  (h : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 60) : 
  x + 2 = 12 := by
  sorry

end third_number_in_sequence_l2122_212283


namespace rational_nonzero_l2122_212289

theorem rational_nonzero (a b : ℚ) (h1 : a * b > a) (h2 : a - b > b) : a ≠ 0 ∧ b ≠ 0 := by
  sorry

end rational_nonzero_l2122_212289


namespace cos_150_degrees_l2122_212248

theorem cos_150_degrees :
  Real.cos (150 * π / 180) = -(1 / 2) := by sorry

end cos_150_degrees_l2122_212248


namespace tan_x_plus_pi_third_l2122_212282

theorem tan_x_plus_pi_third (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + π / 3) = -Real.sqrt 3 := by
  sorry

end tan_x_plus_pi_third_l2122_212282


namespace boys_in_school_after_increase_l2122_212253

/-- The number of boys in a school after an increase -/
def boys_after_increase (initial_boys : ℕ) (additional_boys : ℕ) : ℕ :=
  initial_boys + additional_boys

theorem boys_in_school_after_increase :
  boys_after_increase 214 910 = 1124 := by
  sorry

end boys_in_school_after_increase_l2122_212253


namespace sin_cos_sum_negative_sqrt_two_l2122_212291

theorem sin_cos_sum_negative_sqrt_two (x : Real) : 
  0 ≤ x → x < 2 * Real.pi → Real.sin x + Real.cos x = -Real.sqrt 2 → x = 5 * Real.pi / 4 := by
  sorry

end sin_cos_sum_negative_sqrt_two_l2122_212291


namespace sin_240_l2122_212232

-- Define the cofunction identity
axiom cofunction_identity (α : Real) : Real.sin (180 + α) = -Real.sin α

-- Define the special angle value
axiom sin_60 : Real.sin 60 = Real.sqrt 3 / 2

-- State the theorem to be proved
theorem sin_240 : Real.sin 240 = -(Real.sqrt 3 / 2) := by
  sorry

end sin_240_l2122_212232


namespace jimin_seokjin_money_sum_l2122_212247

/-- Calculates the total amount of money for a person given their coin distribution --/
def calculate_total (coins_100 : Nat) (coins_50 : Nat) (coins_10 : Nat) : Nat :=
  100 * coins_100 + 50 * coins_50 + 10 * coins_10

/-- Represents the coin distribution and total money for Jimin and Seokjin --/
theorem jimin_seokjin_money_sum :
  let jimin_total := calculate_total 5 1 0
  let seokjin_total := calculate_total 2 0 7
  jimin_total + seokjin_total = 820 := by
  sorry

#check jimin_seokjin_money_sum

end jimin_seokjin_money_sum_l2122_212247


namespace folded_rectangle_area_l2122_212292

/-- Represents a rectangle with length and width -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

theorem folded_rectangle_area :
  ∀ (r1 r2 r3 : Rectangle),
    perimeter r1 = perimeter r2 + 20 →
    perimeter r2 = perimeter r3 + 16 →
    r1.length = r2.length →
    r2.length = r3.length →
    r1.width = r2.width + 10 →
    r2.width = r3.width + 8 →
    area r1 = 504 :=
by sorry

end folded_rectangle_area_l2122_212292


namespace complex_magnitude_theorem_l2122_212223

theorem complex_magnitude_theorem (s : ℝ) (w : ℂ) 
  (h1 : |s| < 3) 
  (h2 : w + 2 / w = s) : 
  Complex.abs w = Real.sqrt 2 := by
  sorry

end complex_magnitude_theorem_l2122_212223


namespace ap_terms_count_l2122_212214

/-- An arithmetic progression with the given properties -/
structure ArithmeticProgression where
  n : ℕ
  a : ℝ
  d : ℝ
  even_n : Even n
  odd_sum : (n / 2) * (a + (a + (n - 2) * d)) = 30
  even_sum : (n / 2) * ((a + d) + (a + (n - 1) * d)) = 45
  last_first_diff : (a + (n - 1) * d) - a = 7.5

/-- The theorem stating that the number of terms in the arithmetic progression is 12 -/
theorem ap_terms_count (ap : ArithmeticProgression) : ap.n = 12 := by
  sorry

end ap_terms_count_l2122_212214


namespace eight_times_seven_divided_by_three_l2122_212288

theorem eight_times_seven_divided_by_three :
  (∃ (a b c : ℕ), a = 5 ∧ b = 6 ∧ c = 7 ∧ a * b = 30 ∧ b * c = 42 ∧ c * 8 = 56) →
  (8 * 7) / 3 = 18 ∧ (8 * 7) % 3 = 2 := by
  sorry

end eight_times_seven_divided_by_three_l2122_212288


namespace points_needed_in_next_game_l2122_212204

def last_home_game_score : ℕ := 62

def first_away_game_score : ℕ := last_home_game_score / 2

def second_away_game_score : ℕ := first_away_game_score + 18

def third_away_game_score : ℕ := second_away_game_score + 2

def cumulative_score_goal : ℕ := 4 * last_home_game_score

def current_cumulative_score : ℕ := 
  last_home_game_score + first_away_game_score + second_away_game_score + third_away_game_score

theorem points_needed_in_next_game : 
  cumulative_score_goal - current_cumulative_score = 55 := by
  sorry

end points_needed_in_next_game_l2122_212204


namespace candy_ratio_l2122_212210

theorem candy_ratio (chocolate_bars : ℕ) (m_and_ms : ℕ) (marshmallows : ℕ) :
  chocolate_bars = 5 →
  marshmallows = 6 * m_and_ms →
  chocolate_bars + m_and_ms + marshmallows = 250 →
  m_and_ms / chocolate_bars = 7 := by
  sorry

end candy_ratio_l2122_212210


namespace max_ratio_in_triangle_l2122_212230

/-- Given a triangle OAB where O is the origin, A is the point (4,3), and B is the point (x,0) with x > 0,
    this theorem states that the maximum value of the ratio x/l(x) is 5/3,
    where l(x) is the length of line segment AB. -/
theorem max_ratio_in_triangle (x : ℝ) (hx : x > 0) : 
  let A : ℝ × ℝ := (4, 3)
  let B : ℝ × ℝ := (x, 0)
  let l : ℝ → ℝ := fun x => Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  (∀ y > 0, y / l y ≤ x / l x) → x / l x = 5/3 := by
  sorry

end max_ratio_in_triangle_l2122_212230


namespace sector_central_angle_l2122_212286

theorem sector_central_angle (perimeter area : ℝ) (h1 : perimeter = 10) (h2 : area = 4) :
  ∃ (r l θ : ℝ), r > 0 ∧ l > 0 ∧ θ > 0 ∧ 
  2 * r + l = perimeter ∧ 
  1/2 * r * l = area ∧ 
  θ = l / r ∧ 
  θ = 1/2 := by
sorry

end sector_central_angle_l2122_212286


namespace line_point_x_coordinate_l2122_212211

/-- Given a line with slope -3.5 and y-intercept 1.5, 
    the x-coordinate of the point with y-coordinate 1025 is -1023.5 / 3.5 -/
theorem line_point_x_coordinate 
  (slope : ℝ) 
  (y_intercept : ℝ) 
  (y : ℝ) 
  (h1 : slope = -3.5) 
  (h2 : y_intercept = 1.5) 
  (h3 : y = 1025) : 
  (y - y_intercept) / (-slope) = -1023.5 / 3.5 := by
  sorry

#eval -1023.5 / 3.5

end line_point_x_coordinate_l2122_212211


namespace pages_per_donut_l2122_212266

/-- Given Jean's writing and eating habits, calculate the number of pages she writes per donut. -/
theorem pages_per_donut (pages_written : ℕ) (calories_per_donut : ℕ) (total_calories : ℕ)
  (h1 : pages_written = 12)
  (h2 : calories_per_donut = 150)
  (h3 : total_calories = 900) :
  pages_written / (total_calories / calories_per_donut) = 2 :=
by sorry

end pages_per_donut_l2122_212266


namespace max_mineral_value_l2122_212243

/-- Represents a type of mineral with its weight and value --/
structure Mineral where
  weight : ℕ
  value : ℕ

/-- The problem setup --/
def mineral_problem : Prop :=
  ∃ (j k l : Mineral) (x y z : ℕ),
    j.weight = 6 ∧ j.value = 17 ∧
    k.weight = 3 ∧ k.value = 9 ∧
    l.weight = 2 ∧ l.value = 5 ∧
    x * j.weight + y * k.weight + z * l.weight ≤ 20 ∧
    ∀ (a b c : ℕ),
      a * j.weight + b * k.weight + c * l.weight ≤ 20 →
      a * j.value + b * k.value + c * l.value ≤ x * j.value + y * k.value + z * l.value ∧
      x * j.value + y * k.value + z * l.value = 60

theorem max_mineral_value : mineral_problem := by sorry

end max_mineral_value_l2122_212243


namespace complex_equation_solution_l2122_212242

theorem complex_equation_solution (z : ℂ) (h : z * (1 + Complex.I) = 2 - 4 * Complex.I) :
  z = -1 - 3 * Complex.I := by
  sorry

end complex_equation_solution_l2122_212242


namespace equation_solution_existence_l2122_212233

theorem equation_solution_existence (a : ℝ) :
  (∃ x : ℝ, 3 * 4^(x - 2) + 27 = a + a * 4^(x - 2)) ↔ (3 < a ∧ a < 27) :=
by sorry

end equation_solution_existence_l2122_212233


namespace some_club_members_not_debate_team_l2122_212280

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Club : U → Prop)  -- x is a club member
variable (Punctual : U → Prop)  -- x is punctual
variable (DebateTeam : U → Prop)  -- x is a debate team member

-- Define the premises
variable (h1 : ∃ x, Club x ∧ ¬Punctual x)  -- Some club members are not punctual
variable (h2 : ∀ x, DebateTeam x → Punctual x)  -- All members of the debate team are punctual

-- State the theorem
theorem some_club_members_not_debate_team :
  ∃ x, Club x ∧ ¬DebateTeam x :=
by sorry

end some_club_members_not_debate_team_l2122_212280
