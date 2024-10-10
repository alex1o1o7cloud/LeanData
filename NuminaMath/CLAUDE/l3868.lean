import Mathlib

namespace farmer_earnings_l3868_386802

/-- Calculates the total earnings from selling potatoes and carrots -/
def total_earnings (potato_count : ℕ) (potato_bundle_size : ℕ) (potato_bundle_price : ℚ)
                   (carrot_count : ℕ) (carrot_bundle_size : ℕ) (carrot_bundle_price : ℚ) : ℚ :=
  let potato_bundles := potato_count / potato_bundle_size
  let carrot_bundles := carrot_count / carrot_bundle_size
  potato_bundles * potato_bundle_price + carrot_bundles * carrot_bundle_price

/-- The farmer's earnings from selling all harvested crops -/
theorem farmer_earnings : 
  total_earnings 250 25 1.9 320 20 2 = 51 := by
  sorry

end farmer_earnings_l3868_386802


namespace parabola_vertex_coordinates_l3868_386885

/-- The vertex of the parabola y = 1/2 * (x + 2)^2 + 1 has coordinates (-2, 1) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x ↦ (1/2) * (x + 2)^2 + 1
  ∃! (h k : ℝ), (∀ x, f x = (1/2) * (x - h)^2 + k) ∧ h = -2 ∧ k = 1 := by
  sorry

end parabola_vertex_coordinates_l3868_386885


namespace rationalize_denominator_l3868_386896

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (B < D) ∧
    (3 : ℚ) / (2 * Real.sqrt 18 + 5 * Real.sqrt 20) =
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    A = -18 ∧ B = 2 ∧ C = 30 ∧ D = 5 ∧ E = 428 :=
by sorry

end rationalize_denominator_l3868_386896


namespace vector_dot_product_symmetry_and_value_l3868_386840

/-- Given vectors a and b, and function f as defined, prove the axis of symmetry and a specific function value. -/
theorem vector_dot_product_symmetry_and_value 
  (x θ : ℝ) 
  (a : ℝ → ℝ × ℝ)
  (b : ℝ → ℝ × ℝ)
  (f : ℝ → ℝ)
  (h1 : a = λ x => (Real.sin x, 1))
  (h2 : b = λ x => (1, Real.cos x))
  (h3 : f = λ x => (a x).1 * (b x).1 + (a x).2 * (b x).2)
  (h4 : f (θ + π/4) = Real.sqrt 2 / 3)
  (h5 : 0 < θ)
  (h6 : θ < π/2) :
  (∃ k : ℤ, ∀ x, f x = f (2 * (k * π + π/4) - x)) ∧ 
  f (θ - π/4) = 4/3 := by
  sorry

end vector_dot_product_symmetry_and_value_l3868_386840


namespace arithmetic_geometric_ratio_l3868_386854

/-- An arithmetic sequence with a positive common difference -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_positive : d > 0
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence where a_2, a_6, and a_12 form a geometric sequence,
    the ratio of a_12 to a_2 is 9/4 -/
theorem arithmetic_geometric_ratio
  (seq : ArithmeticSequence)
  (h_geometric : (seq.a 6) ^ 2 = (seq.a 2) * (seq.a 12)) :
  (seq.a 12) / (seq.a 2) = 9 / 4 := by
  sorry

end arithmetic_geometric_ratio_l3868_386854


namespace jake_balloons_count_l3868_386860

/-- The number of balloons Allan brought to the park -/
def allan_balloons : ℕ := 5

/-- The additional number of balloons Jake brought compared to Allan -/
def jake_extra_balloons : ℕ := 6

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := allan_balloons + jake_extra_balloons

theorem jake_balloons_count : jake_balloons = 11 := by sorry

end jake_balloons_count_l3868_386860


namespace double_inequality_solution_l3868_386870

theorem double_inequality_solution (x : ℝ) : 
  (3 ≤ |x - 3| ∧ |x - 3| ≤ 6 ∧ x ≤ 8) ↔ ((-3 ≤ x ∧ x ≤ 3) ∨ (6 ≤ x ∧ x ≤ 8)) :=
by sorry

end double_inequality_solution_l3868_386870


namespace floor_plus_x_equals_seventeen_fourths_l3868_386843

theorem floor_plus_x_equals_seventeen_fourths :
  ∃ x : ℚ, (⌊x⌋ : ℚ) + x = 17/4 ∧ x = 9/4 := by sorry

end floor_plus_x_equals_seventeen_fourths_l3868_386843


namespace calculate_Y_l3868_386883

theorem calculate_Y : ∀ A B Y : ℚ,
  A = 3081 / 4 →
  B = A * 2 →
  Y = A - B →
  Y = -770.25 := by
sorry

end calculate_Y_l3868_386883


namespace not_enough_money_l3868_386888

/-- The cost of a new smartwatch in rubles -/
def smartwatch_cost : ℕ := 2019

/-- The amount of money Namzhil has in rubles -/
def namzhil_money : ℕ := (500^2 + 4 * 500 + 3) * 498^2 - 500^2 * 503 * 497

/-- Theorem stating that Namzhil does not have enough money to buy the smartwatch -/
theorem not_enough_money : namzhil_money < smartwatch_cost := by
  sorry

end not_enough_money_l3868_386888


namespace greatest_integer_no_substring_divisible_by_9_all_substrings_of_88888888_not_divisible_by_9_l3868_386810

/-- A function that returns all integer substrings of a given positive integer -/
def integerSubstrings (n : ℕ+) : Finset ℕ :=
  sorry

/-- A function that checks if any element in a finite set is divisible by 9 -/
def anyDivisibleBy9 (s : Finset ℕ) : Prop :=
  sorry

theorem greatest_integer_no_substring_divisible_by_9 :
  ∀ n : ℕ+, n > 88888888 → anyDivisibleBy9 (integerSubstrings n) :=
  sorry

theorem all_substrings_of_88888888_not_divisible_by_9 :
  ¬ anyDivisibleBy9 (integerSubstrings 88888888) :=
  sorry

end greatest_integer_no_substring_divisible_by_9_all_substrings_of_88888888_not_divisible_by_9_l3868_386810


namespace ellipse_focal_distance_l3868_386819

theorem ellipse_focal_distance (m : ℝ) :
  (∀ x y : ℝ, x^2/16 + y^2/m = 1) →
  (∃ c : ℝ, c > 0 ∧ c^2 = 16 - m ∧ 2*c = 2*Real.sqrt 7) →
  m = 9 :=
by sorry

end ellipse_focal_distance_l3868_386819


namespace min_value_of_f_l3868_386811

/-- The function f(x) = x^2 + 16x + 20 -/
def f (x : ℝ) : ℝ := x^2 + 16*x + 20

/-- The minimum value of f(x) is -44 -/
theorem min_value_of_f : 
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ (m = -44) := by
  sorry

end min_value_of_f_l3868_386811


namespace sum_of_equations_l3868_386878

theorem sum_of_equations (a b c d : ℤ) 
  (eq1 : a - b + c = 7)
  (eq2 : b - c + d = 8)
  (eq3 : c - d + a = 4)
  (eq4 : d - a + b = 1) :
  2*a + 2*b + 2*c + 2*d = 20 := by
sorry

end sum_of_equations_l3868_386878


namespace kelly_games_to_give_away_l3868_386887

/-- The number of games Kelly needs to give away to reach her desired number of games -/
def games_to_give_away (initial_games : ℕ) (desired_games : ℕ) : ℕ :=
  initial_games - desired_games

/-- Proof that Kelly needs to give away 15 games -/
theorem kelly_games_to_give_away :
  games_to_give_away 50 35 = 15 := by
  sorry

end kelly_games_to_give_away_l3868_386887


namespace distance_between_circle_centers_l3868_386831

theorem distance_between_circle_centers (a b c : ℝ) (h_a : a = 17) (h_b : b = 15) (h_c : c = 10) :
  let s := (a + b + c) / 2
  let K := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let r := K / s
  let AI := Real.sqrt (16 + (K / s) ^ 2)
  20 * AI = 20 * Real.sqrt (16 + 5544 / 441) :=
by sorry

end distance_between_circle_centers_l3868_386831


namespace dataset_reduction_fraction_l3868_386881

theorem dataset_reduction_fraction (initial : ℕ) (increase_percent : ℚ) (final : ℕ) : 
  initial = 200 →
  increase_percent = 1/5 →
  final = 180 →
  (initial + initial * increase_percent - final) / (initial + initial * increase_percent) = 1/4 :=
by sorry

end dataset_reduction_fraction_l3868_386881


namespace hyperbola_dot_product_nonnegative_l3868_386849

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

/-- The left vertex of the hyperbola -/
def A : ℝ × ℝ := (-2, 0)

/-- The right vertex of the hyperbola -/
def B : ℝ × ℝ := (2, 0)

/-- The dot product of vectors PA and PB -/
def dot_product (P : ℝ × ℝ) : ℝ :=
  let (m, n) := P
  ((-2 - m) * (2 - m)) + (n * n)

theorem hyperbola_dot_product_nonnegative :
  ∀ P : ℝ × ℝ, hyperbola P.1 P.2 → dot_product P ≥ 0 := by sorry

end hyperbola_dot_product_nonnegative_l3868_386849


namespace price_adjustment_l3868_386820

theorem price_adjustment (original_price : ℝ) (original_price_pos : original_price > 0) :
  let increased_price := original_price * (1 + 0.25)
  let decrease_percentage := (increased_price - original_price) / increased_price
  decrease_percentage = 0.20 := by
sorry

end price_adjustment_l3868_386820


namespace projection_result_l3868_386899

def v1 : ℝ × ℝ := (3, 2)
def v2 : ℝ × ℝ := (2, 5)

theorem projection_result (u : ℝ × ℝ) (q : ℝ × ℝ) 
  (h1 : ∃ (k1 : ℝ), q = k1 • u ∧ (v1 - q) • u = 0)
  (h2 : ∃ (k2 : ℝ), q = k2 • u ∧ (v2 - q) • u = 0) :
  q = (33/10, 11/10) :=
sorry

end projection_result_l3868_386899


namespace intersection_point_theorem_l3868_386821

theorem intersection_point_theorem (α β : ℝ) :
  (∃ x y : ℝ, 
    x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1 ∧
    x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1 ∧
    y = -x) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
  sorry

end intersection_point_theorem_l3868_386821


namespace matrix_transformation_l3868_386853

theorem matrix_transformation (N : Matrix (Fin 2) (Fin 2) ℚ) 
  (h1 : N.mulVec ![1, 2] = ![4, 1])
  (h2 : N.mulVec ![2, -3] = ![1, 4]) :
  N.mulVec ![7, -2] = ![(84:ℚ)/7, (81:ℚ)/7] := by
  sorry

end matrix_transformation_l3868_386853


namespace total_water_consumed_l3868_386886

-- Define the conversion rate from quarts to ounces
def quart_to_ounce : ℚ := 32

-- Define the amount of water in the bottle (in quarts)
def bottle_water : ℚ := 3/2

-- Define the amount of water in the can (in ounces)
def can_water : ℚ := 12

-- Theorem statement
theorem total_water_consumed :
  bottle_water * quart_to_ounce + can_water = 60 := by
  sorry

end total_water_consumed_l3868_386886


namespace inner_shape_area_ratio_l3868_386857

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- Points that trisect the sides of a hexagon -/
def trisection_points (h : RegularHexagon) : Fin 6 → ℝ × ℝ :=
  sorry

/-- The shape formed by joining the trisection points -/
def inner_shape (h : RegularHexagon) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating that the area of the inner shape is 2/3 of the original hexagon -/
theorem inner_shape_area_ratio (h : RegularHexagon) :
    area (inner_shape h) = (2 / 3) * area (Set.range h.vertices) := by
  sorry

end inner_shape_area_ratio_l3868_386857


namespace m_range_theorem_l3868_386894

/-- Proposition p: x^2 - mx + 1 = 0 has no real solutions -/
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 1 ≠ 0

/-- Proposition q: x^2/m + y^2 = 1 has its foci on the x-axis -/
def q (m : ℝ) : Prop := m > 1

/-- The range of real values for m given the conditions -/
def m_range (m : ℝ) : Prop := (-2 < m ∧ m ≤ 1) ∨ m ≥ 2

/-- Theorem stating the range of m given the conditions -/
theorem m_range_theorem (m : ℝ) (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : m_range m := by
  sorry

end m_range_theorem_l3868_386894


namespace set_union_problem_l3868_386805

theorem set_union_problem (a b l : ℝ) :
  let A : Set ℝ := {-2, a}
  let B : Set ℝ := {2015^a, b}
  A ∩ B = {l} →
  A ∪ B = {-2, 1, 2015} :=
by
  sorry

end set_union_problem_l3868_386805


namespace tetrahedron_volume_in_cube_l3868_386895

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (s : ℝ) (h : s = 8) :
  let cube_volume := s^3
  let tetrahedron_volume := cube_volume / 3
  tetrahedron_volume = 512 / 3 := by
  sorry

end tetrahedron_volume_in_cube_l3868_386895


namespace rogers_allowance_theorem_l3868_386815

/-- Roger's weekly allowance problem -/
theorem rogers_allowance_theorem (B : ℝ) (m s p : ℝ) : 
  (m = (1/4) * (B - s)) → 
  (s = (1/10) * (B - m)) → 
  (p = (1/10) * (m + s)) → 
  (m + s + p) / B = 22 / 65 := by
  sorry

end rogers_allowance_theorem_l3868_386815


namespace train_passing_time_l3868_386813

/-- Prove that a train with given length and speed will pass a fixed point in the calculated time -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 275 →
  train_speed_kmh = 90 →
  passing_time = train_length / (train_speed_kmh * 1000 / 3600) →
  passing_time = 11 := by
  sorry

end train_passing_time_l3868_386813


namespace percentage_of_chemical_b_in_solution_x_l3868_386822

-- Define the solutions and mixture
def solution_x (a b : ℝ) : Prop := a + b = 1 ∧ a = 0.1
def solution_y : Prop := 0.2 + 0.8 = 1
def mixture (x y : ℝ) : Prop := x + y = 1 ∧ x = 0.8

-- Define the chemical compositions
def chemical_a_in_mixture : ℝ := 0.12
def chemical_b_in_solution_x : ℝ := 0.9

-- State the theorem
theorem percentage_of_chemical_b_in_solution_x 
  (a b x y : ℝ) 
  (hx : solution_x a b) 
  (hy : solution_y)
  (hm : mixture x y)
  (ha : x * a + y * 0.2 = chemical_a_in_mixture) :
  b = chemical_b_in_solution_x :=
sorry

end percentage_of_chemical_b_in_solution_x_l3868_386822


namespace daisy_exchange_impossible_l3868_386823

/-- Represents the number of girls in the row -/
def n : ℕ := 33

/-- Represents the number of places each girl passes her daisy -/
def pass_distance : ℕ := 2

/-- Predicate that checks if a girl at position i receives a daisy -/
def receives_daisy (i : ℕ) : Prop :=
  ∃ j : ℕ, j ≤ n ∧ (i = j + pass_distance ∨ i = j - pass_distance)

/-- Theorem stating it's impossible for every girl to end up with exactly one daisy -/
theorem daisy_exchange_impossible : ¬(∀ i : ℕ, i ≤ n → ∃! j : ℕ, receives_daisy j ∧ i = j) :=
sorry

end daisy_exchange_impossible_l3868_386823


namespace gcd_2_powers_l3868_386893

theorem gcd_2_powers : 
  Nat.gcd (2^2025 - 1) (2^2016 - 1) = 2^9 - 1 := by
sorry

end gcd_2_powers_l3868_386893


namespace infinite_sum_floor_floor_2x_l3868_386868

/-- For any real number x, the sum of floor((x + 2^k) / 2^(k+1)) from k=0 to infinity is equal to floor(x). -/
theorem infinite_sum_floor (x : ℝ) : 
  (∑' k, ⌊(x + 2^k) / 2^(k+1)⌋) = ⌊x⌋ :=
by sorry

/-- For any real number x, floor(2x) = floor(x) + floor(x + 1/2). -/
theorem floor_2x (x : ℝ) : 
  ⌊2*x⌋ = ⌊x⌋ + ⌊x + 1/2⌋ :=
by sorry

end infinite_sum_floor_floor_2x_l3868_386868


namespace equation_solution_l3868_386872

theorem equation_solution : 
  let t : ℚ := -8
  (1 : ℚ) / (t + 2) + (2 * t) / (t + 2) - (3 : ℚ) / (t + 2) = 3 := by
  sorry

end equation_solution_l3868_386872


namespace two_rolls_probability_l3868_386834

/-- A fair six-sided die --/
def FairDie := Fin 6

/-- The probability of rolling a specific number on a fair die --/
def prob_single_roll : ℚ := 1 / 6

/-- The sum of two die rolls --/
def sum_of_rolls (a b : FairDie) : ℕ := a.val + b.val + 2

/-- Whether a number is prime --/
def is_prime (n : ℕ) : Prop := Nat.Prime n

/-- The probability that the sum of two rolls is prime --/
def prob_sum_is_prime : ℚ := 15 / 36

theorem two_rolls_probability (rolls : ℕ) : 
  (rolls = 2 ∧ prob_sum_is_prime = 0.41666666666666663) → rolls = 2 := by
  sorry

end two_rolls_probability_l3868_386834


namespace sum_primes_square_bound_l3868_386877

/-- S_n is the sum of the first n prime numbers -/
def S (n : ℕ) : ℕ := sorry

/-- The n-th prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

theorem sum_primes_square_bound :
  ∀ n : ℕ, n > 0 → ∃ m : ℕ, S n ≤ m^2 ∧ m^2 ≤ S (n + 1) :=
sorry

end sum_primes_square_bound_l3868_386877


namespace max_value_of_sum_of_roots_max_value_achieved_l3868_386817

theorem max_value_of_sum_of_roots (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 20) :
  (Real.sqrt (x + 20) + Real.sqrt (20 - x) + Real.sqrt (2 * x) + Real.sqrt (30 - x)) ≤ Real.sqrt 630 :=
by sorry

theorem max_value_achieved (x : ℝ) (h₁ : 0 ≤ x) (h₂ : x ≤ 20) :
  ∃ y, 0 ≤ y ∧ y ≤ 20 ∧
    (Real.sqrt (y + 20) + Real.sqrt (20 - y) + Real.sqrt (2 * y) + Real.sqrt (30 - y)) = Real.sqrt 630 :=
by sorry

end max_value_of_sum_of_roots_max_value_achieved_l3868_386817


namespace cubic_coefficient_sum_l3868_386866

theorem cubic_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) : 
  (∀ x : ℝ, (5*x + 4)^3 = a₀ + a₁*x + a₂*x^2 + a₃*x^3) → 
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
  sorry

end cubic_coefficient_sum_l3868_386866


namespace skateboard_ramp_speed_increase_l3868_386851

/-- Calculates the additional speed required to reach the top of a skateboard ramp -/
theorem skateboard_ramp_speed_increase 
  (ramp_height : ℝ) 
  (ramp_incline : ℝ) 
  (speed_without_wind : ℝ) 
  (trial_speeds : List ℝ) 
  (wind_resistance_min : ℝ) 
  (wind_resistance_max : ℝ) : 
  ramp_height = 50 → 
  ramp_incline = 30 → 
  speed_without_wind = 40 → 
  trial_speeds = [36, 34, 38] → 
  wind_resistance_min = 3 → 
  wind_resistance_max = 5 → 
  (List.sum trial_speeds / trial_speeds.length + 
   (wind_resistance_min + wind_resistance_max) / 2 + 
   speed_without_wind) - 
  (List.sum trial_speeds / trial_speeds.length) = 8 := by
  sorry

end skateboard_ramp_speed_increase_l3868_386851


namespace conclusion_l3868_386889

-- Define the variables
variable (p q r s u v : ℝ)

-- State the given conditions
axiom cond1 : p > q → r > s
axiom cond2 : r = s → u < v
axiom cond3 : p = q → s > r

-- State the theorem to be proved
theorem conclusion : p ≠ q → s ≠ r := by
  sorry

end conclusion_l3868_386889


namespace inequality_theorem_l3868_386874

theorem inequality_theorem (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) :
  x * (x - z)^2 + y * (y - z)^2 ≥ (x - z) * (y - z) * (x + y - z) ∧
  (x * (x - z)^2 + y * (y - z)^2 = (x - z) * (y - z) * (x + y - z) ↔ 
    (x = y ∧ y = z) ∨ (x = y ∧ z = 0)) :=
by sorry

end inequality_theorem_l3868_386874


namespace parabola_point_distance_to_origin_l3868_386871

theorem parabola_point_distance_to_origin :
  ∀ (x y : ℝ),
  y^2 = 2*x →  -- Point A is on the parabola y^2 = 2x
  (x + 1/2) / |y| = 5/4 →  -- Ratio condition
  ((x - 1/2)^2 + y^2)^(1/2) > 2 →  -- |AF| > 2
  (x^2 + y^2)^(1/2) = 2 * (2^(1/2)) :=
by
  sorry

end parabola_point_distance_to_origin_l3868_386871


namespace younger_brother_height_l3868_386829

def father_height : ℕ := 172
def height_diff_father_minkyung : ℕ := 35
def height_diff_minkyung_brother : ℕ := 28

theorem younger_brother_height :
  father_height - height_diff_father_minkyung - height_diff_minkyung_brother = 109 :=
by sorry

end younger_brother_height_l3868_386829


namespace total_out_of_pocket_is_190_50_l3868_386841

def consultation_cost : ℝ := 300
def consultation_coverage : ℝ := 0.8
def xray_cost : ℝ := 150
def xray_coverage : ℝ := 0.7
def medication_cost : ℝ := 75
def medication_coverage : ℝ := 0.5
def therapy_cost : ℝ := 120
def therapy_coverage : ℝ := 0.6

def total_out_of_pocket_cost : ℝ :=
  (1 - consultation_coverage) * consultation_cost +
  (1 - xray_coverage) * xray_cost +
  (1 - medication_coverage) * medication_cost +
  (1 - therapy_coverage) * therapy_cost

theorem total_out_of_pocket_is_190_50 :
  total_out_of_pocket_cost = 190.50 := by
  sorry

end total_out_of_pocket_is_190_50_l3868_386841


namespace trig_identity_l3868_386818

theorem trig_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) :
  (Real.sin θ + Real.cos θ) / (Real.sin θ - Real.cos θ) = 2 + Real.sqrt 3 := by
  sorry

end trig_identity_l3868_386818


namespace quadratic_root_form_l3868_386879

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 - 7*x + d = 0 ↔ x = (7 + Real.sqrt d) / 2 ∨ x = (7 - Real.sqrt d) / 2) → 
  d = 49 / 5 := by
sorry

end quadratic_root_form_l3868_386879


namespace largest_divisible_by_thirtyseven_with_decreasing_digits_l3868_386855

/-- 
A function that checks if a natural number's digits are in strictly decreasing order.
-/
def isStrictlyDecreasing (n : ℕ) : Prop :=
  sorry

/-- 
A function that finds the largest natural number less than or equal to n 
that is divisible by 37 and has strictly decreasing digits.
-/
def largestDivisibleByThirtySevenWithDecreasingDigits (n : ℕ) : ℕ :=
  sorry

theorem largest_divisible_by_thirtyseven_with_decreasing_digits :
  largestDivisibleByThirtySevenWithDecreasingDigits 9876543210 = 987654 :=
sorry

end largest_divisible_by_thirtyseven_with_decreasing_digits_l3868_386855


namespace point_coordinates_wrt_origin_l3868_386867

/-- The coordinates of a point with respect to the origin are the same as its given coordinates. -/
theorem point_coordinates_wrt_origin (x y : ℝ) :
  let M : ℝ × ℝ := (x, y)
  M = (x, y) := by sorry

end point_coordinates_wrt_origin_l3868_386867


namespace initial_bananas_per_child_l3868_386827

theorem initial_bananas_per_child (total_children : ℕ) (absent_children : ℕ) (extra_bananas : ℕ) :
  total_children = 840 →
  absent_children = 420 →
  extra_bananas = 2 →
  ∃ (initial_bananas : ℕ),
    total_children * initial_bananas = (total_children - absent_children) * (initial_bananas + extra_bananas) ∧
    initial_bananas = 2 :=
by sorry

end initial_bananas_per_child_l3868_386827


namespace pairing_probabilities_correct_probabilities_sum_to_one_l3868_386845

-- Define the total number of teams and English teams
def total_teams : ℕ := 8
def english_teams : ℕ := 4

-- Define the total number of possible pairings
def total_pairings : ℕ := 105

-- Define the probabilities for each scenario
def prob_no_english_pairs : ℚ := 24 / 105
def prob_two_english_pairs : ℚ := 18 / 105
def prob_one_english_pair : ℚ := 72 / 105

-- Theorem to prove the correctness of the probabilities
theorem pairing_probabilities_correct :
  (prob_no_english_pairs + prob_two_english_pairs + prob_one_english_pair = 1) ∧
  (prob_no_english_pairs = 24 / 105) ∧
  (prob_two_english_pairs = 18 / 105) ∧
  (prob_one_english_pair = 72 / 105) := by
  sorry

-- Theorem to prove that the probabilities sum to 1
theorem probabilities_sum_to_one :
  prob_no_english_pairs + prob_two_english_pairs + prob_one_english_pair = 1 := by
  sorry

end pairing_probabilities_correct_probabilities_sum_to_one_l3868_386845


namespace yellow_marbles_count_l3868_386890

theorem yellow_marbles_count (yellow blue : ℕ) 
  (h1 : blue = yellow - 2)
  (h2 : yellow + blue = 240) : 
  yellow = 121 := by
sorry

end yellow_marbles_count_l3868_386890


namespace multiple_reals_less_than_negative_one_l3868_386824

theorem multiple_reals_less_than_negative_one :
  ∃ (x y : ℝ), x < -1 ∧ y < -1 ∧ x ≠ y :=
sorry

end multiple_reals_less_than_negative_one_l3868_386824


namespace roots_have_unit_modulus_l3868_386869

theorem roots_have_unit_modulus (z : ℂ) :
  (11 * z^10 + 10 * Complex.I * z^9 + 10 * Complex.I * z - 11 = 0) → Complex.abs z = 1 := by
  sorry

end roots_have_unit_modulus_l3868_386869


namespace max_g_6_l3868_386825

/-- A polynomial of degree 2 with real, nonnegative coefficients -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating the maximum value of g(6) given the conditions -/
theorem max_g_6 (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) 
  (h1 : g a b c 3 = 3) (h2 : g a b c 9 = 243) :
  ∀ a' b' c', a' ≥ 0 → b' ≥ 0 → c' ≥ 0 → 
  g a' b' c' 3 = 3 → g a' b' c' 9 = 243 → 
  g a' b' c' 6 ≤ 6 :=
by sorry

#check max_g_6

end max_g_6_l3868_386825


namespace number_problem_l3868_386839

theorem number_problem (x : ℝ) : (0.40 * x = 0.80 * 5 + 2) → x = 15 := by
  sorry

end number_problem_l3868_386839


namespace jakes_weight_l3868_386800

theorem jakes_weight (jake kendra : ℕ) 
  (h1 : jake - 8 = 2 * kendra) 
  (h2 : jake + kendra = 287) : 
  jake = 194 := by sorry

end jakes_weight_l3868_386800


namespace max_sum_absolute_values_l3868_386897

theorem max_sum_absolute_values (x y : ℝ) (h : 4 * x^2 + y^2 = 4) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (a b : ℝ), 4 * a^2 + b^2 = 4 → |a| + |b| ≤ M :=
by sorry

end max_sum_absolute_values_l3868_386897


namespace x_fourth_plus_inverse_fourth_l3868_386863

theorem x_fourth_plus_inverse_fourth (x : ℝ) (h : x^2 - 15*x + 1 = 0) : 
  x^4 + 1/x^4 = 49727 := by
  sorry

end x_fourth_plus_inverse_fourth_l3868_386863


namespace odot_problem_l3868_386882

-- Define the custom operation
def odot (a b : ℚ) : ℚ := a + (5 * a) / (3 * b)

-- State the theorem
theorem odot_problem : (odot 12 9) + 3 = 155 / 9 := by
  sorry

end odot_problem_l3868_386882


namespace cubic_polynomial_solution_l3868_386847

noncomputable section

variable (a b c : ℝ)
variable (P : ℝ → ℝ)

def cubic_equation (x : ℝ) : Prop := x^3 + 2*x^2 + 4*x + 6 = 0

theorem cubic_polynomial_solution :
  cubic_equation a ∧ cubic_equation b ∧ cubic_equation c ∧
  (∀ x, ∃ p q r s, P x = p*x^3 + q*x^2 + r*x + s) ∧
  P a = b + c ∧
  P b = a + c ∧
  P c = a + b ∧
  P (a + b + c) = -18 →
  ∀ x, P x = 9/4*x^3 + 5/2*x^2 + 7*x + 15/2 :=
by sorry

end

end cubic_polynomial_solution_l3868_386847


namespace ellipse_intersection_fixed_point_l3868_386816

/-- The ellipse with equation x²/4 + y² = 1 and eccentricity √3/2 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 4) + p.2^2 = 1}

/-- The line x = ky - 1 -/
def Line (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1 = k * p.2 - 1}

/-- Point M is the reflection of A across the x-axis -/
def ReflectAcrossXAxis (A M : ℝ × ℝ) : Prop :=
  M.1 = A.1 ∧ M.2 = -A.2

/-- The line passing through two points -/
def LineThroughPoints (p q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {r | ∃ t : ℝ, r = (1 - t) • p + t • q}

theorem ellipse_intersection_fixed_point (k : ℝ) 
  (A B : ℝ × ℝ) (hA : A ∈ Ellipse ∩ Line k) (hB : B ∈ Ellipse ∩ Line k) 
  (M : ℝ × ℝ) (hM : ReflectAcrossXAxis A M) (hAB : A ≠ B) :
  ∃ P : ℝ × ℝ, P ∈ LineThroughPoints M B ∧ P.1 = -4 ∧ P.2 = 0 :=
sorry

end ellipse_intersection_fixed_point_l3868_386816


namespace santanas_brothers_birthdays_l3868_386836

/-- The number of Santana's brothers -/
def total_brothers : ℕ := 7

/-- The number of brothers with birthdays in March -/
def march_birthdays : ℕ := 3

/-- The number of brothers with birthdays in November -/
def november_birthdays : ℕ := 1

/-- The number of brothers with birthdays in December -/
def december_birthdays : ℕ := 2

/-- The difference in presents between the second and first half of the year -/
def present_difference : ℕ := 8

/-- The number of brothers with birthdays in October -/
def october_birthdays : ℕ := total_brothers - (march_birthdays + november_birthdays + december_birthdays)

theorem santanas_brothers_birthdays :
  october_birthdays = 1 :=
by sorry

end santanas_brothers_birthdays_l3868_386836


namespace ecommerce_problem_l3868_386898

theorem ecommerce_problem (total_spent : ℝ) (price_difference : ℝ) (total_items : ℕ) 
  (subsidy_rate : ℝ) (max_subsidy : ℝ) 
  (h1 : total_spent = 3000)
  (h2 : price_difference = 600)
  (h3 : total_items = 300)
  (h4 : subsidy_rate = 0.1)
  (h5 : max_subsidy = 50000) :
  ∃ (leather_price sweater_price : ℝ) (min_sweaters : ℕ),
    leather_price = 2600 ∧ 
    sweater_price = 400 ∧ 
    min_sweaters = 128 ∧
    leather_price + sweater_price = total_spent ∧
    leather_price = 5 * sweater_price + price_difference ∧
    (↑min_sweaters : ℝ) ≥ (max_subsidy / subsidy_rate - total_items * leather_price) / (sweater_price - leather_price) := by
  sorry

end ecommerce_problem_l3868_386898


namespace solution_satisfies_system_and_initial_conditions_l3868_386806

noncomputable def y₁ (x : ℝ) : ℝ := Real.exp (2 * x)
noncomputable def y₂ (x : ℝ) : ℝ := Real.exp (-x) + Real.exp (2 * x)
noncomputable def y₃ (x : ℝ) : ℝ := -Real.exp (-x) + Real.exp (2 * x)

theorem solution_satisfies_system_and_initial_conditions :
  (∀ x, (deriv y₁) x = y₂ x + y₃ x) ∧
  (∀ x, (deriv y₂) x = y₁ x + y₃ x) ∧
  (∀ x, (deriv y₃) x = y₁ x + y₂ x) ∧
  y₁ 0 = 1 ∧
  y₂ 0 = 2 ∧
  y₃ 0 = 0 := by
  sorry

end solution_satisfies_system_and_initial_conditions_l3868_386806


namespace oreo_milk_purchases_l3868_386833

/-- The number of different flavors of oreos --/
def oreo_flavors : ℕ := 6

/-- The number of different flavors of milk --/
def milk_flavors : ℕ := 4

/-- The total number of products Alpha and Beta purchased --/
def total_products : ℕ := 4

/-- The number of ways Alpha and Beta could have left the store --/
def purchase_combinations : ℕ := 2561

/-- Theorem stating the number of ways Alpha and Beta could have left the store --/
theorem oreo_milk_purchases :
  (oreo_flavors = 6) →
  (milk_flavors = 4) →
  (total_products = 4) →
  purchase_combinations = 2561 :=
by sorry

end oreo_milk_purchases_l3868_386833


namespace smallest_angle_in_special_triangle_l3868_386848

theorem smallest_angle_in_special_triangle : 
  ∀ (a b c : ℝ), 
    a + b + c = 180 →  -- Sum of angles is 180 degrees
    c = 5 * a →        -- Largest angle is 5 times the smallest
    b = 3 * a →        -- Middle angle is 3 times the smallest
    a = 20 :=          -- Smallest angle is 20 degrees
by sorry

end smallest_angle_in_special_triangle_l3868_386848


namespace imaginary_part_of_z_l3868_386838

theorem imaginary_part_of_z : 
  let z : ℂ := (1 - I) / (1 + 3*I)
  Complex.im z = -2/5 :=
by sorry

end imaginary_part_of_z_l3868_386838


namespace final_digit_independent_of_sequence_l3868_386837

/-- Represents the count of each digit on the blackboard -/
structure DigitCount where
  zeros : Nat
  ones : Nat
  twos : Nat

/-- Represents a single step of the digit replacement operation -/
def replaceDigits (count : DigitCount) : DigitCount :=
  sorry

/-- Determines if the operation can continue (more than one digit type remains) -/
def canContinue (count : DigitCount) : Bool :=
  sorry

/-- Performs the digit replacement operations until only one digit type remains -/
def performOperations (initial : DigitCount) : Nat :=
  sorry

theorem final_digit_independent_of_sequence (initial : DigitCount) :
  ∀ (seq1 seq2 : List (DigitCount → DigitCount)),
    (seq1.foldl (fun acc f => f acc) initial).zeros +
    (seq1.foldl (fun acc f => f acc) initial).ones +
    (seq1.foldl (fun acc f => f acc) initial).twos = 1 →
    (seq2.foldl (fun acc f => f acc) initial).zeros +
    (seq2.foldl (fun acc f => f acc) initial).ones +
    (seq2.foldl (fun acc f => f acc) initial).twos = 1 →
    (seq1.foldl (fun acc f => f acc) initial) = (seq2.foldl (fun acc f => f acc) initial) :=
  sorry

end final_digit_independent_of_sequence_l3868_386837


namespace hallies_net_earnings_l3868_386850

/-- Represents a day's work information -/
structure WorkDay where
  hours : ℕ
  hourlyRate : ℚ
  tips : ℚ

/-- Calculates the net earnings for the week -/
def calculateNetEarnings (week : List WorkDay) (taxRate : ℚ) (thursdayDiscountRate : ℚ) : ℚ :=
  sorry

/-- The main theorem stating Hallie's net earnings for the week -/
theorem hallies_net_earnings :
  let week : List WorkDay := [
    ⟨7, 10, 18⟩,  -- Monday
    ⟨5, 12, 12⟩,  -- Tuesday
    ⟨7, 10, 20⟩,  -- Wednesday
    ⟨8, 11, 25⟩,  -- Thursday
    ⟨6, 9, 15⟩    -- Friday
  ]
  let taxRate : ℚ := 5 / 100
  let thursdayDiscountRate : ℚ := 10 / 100
  calculateNetEarnings week taxRate thursdayDiscountRate = 406.1 :=
by sorry

end hallies_net_earnings_l3868_386850


namespace study_group_lawyers_l3868_386859

theorem study_group_lawyers (total_members : ℝ) (h1 : total_members > 0) : 
  let women_ratio : ℝ := 0.4
  let women_lawyer_prob : ℝ := 0.08
  let women_lawyer_ratio : ℝ := women_lawyer_prob / women_ratio
  women_lawyer_ratio = 0.2 := by sorry

end study_group_lawyers_l3868_386859


namespace equation_solution_l3868_386884

theorem equation_solution :
  ∀ x : ℚ, (6 * x / (x + 4) - 2 / (x + 4) = 3 / (x + 4)) ↔ (x = 5 / 6) :=
by sorry

end equation_solution_l3868_386884


namespace orchestra_females_count_l3868_386891

/-- The number of females in the orchestra -/
def females_in_orchestra : ℕ := 12

theorem orchestra_females_count :
  let males_in_orchestra : ℕ := 11
  let choir_size : ℕ := 12 + 17
  let total_musicians : ℕ := 98
  females_in_orchestra = 
    (total_musicians - choir_size - males_in_orchestra - 2 * males_in_orchestra) / 3 :=
by sorry

end orchestra_females_count_l3868_386891


namespace car_speed_proof_l3868_386803

/-- Proves that a car traveling at speed v km/h takes 2 seconds longer to travel 1 kilometer
    than it would at 225 km/h if and only if v = 200 km/h -/
theorem car_speed_proof (v : ℝ) : v > 0 →
  (1 / v * 3600 = 1 / 225 * 3600 + 2) ↔ v = 200 := by
  sorry

end car_speed_proof_l3868_386803


namespace percentage_relations_l3868_386876

theorem percentage_relations (x y z w : ℝ) 
  (h1 : x = 1.3 * y) 
  (h2 : y = 0.5 * z) 
  (h3 : w = 2 * x) : 
  x = 0.65 * z ∧ y = 0.5 * z ∧ w = 1.3 * z := by
  sorry

end percentage_relations_l3868_386876


namespace right_triangle_segment_ratio_l3868_386812

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a / b = 1 / 3 →    -- Given ratio of sides
  r * c = a^2 →      -- Geometric mean theorem for r
  s * c = b^2 →      -- Geometric mean theorem for s
  r + s = c →        -- Segments r and s form the hypotenuse
  r / s = 1 / 9 :=   -- Conclusion to prove
by sorry

end right_triangle_segment_ratio_l3868_386812


namespace perpendicular_line_through_point_l3868_386830

/-- A line in 2D space -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicularLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_through_point 
  (given_line : Line2D) 
  (point : Point2D) 
  (h1 : given_line.a = 1 ∧ given_line.b = 2 ∧ given_line.c = 1) 
  (h2 : point.x = 1 ∧ point.y = 1) : 
  ∃ (l : Line2D), 
    pointOnLine point l ∧ 
    perpendicularLines l given_line ∧ 
    l.a = 2 ∧ l.b = -1 ∧ l.c = -1 := by
  sorry

end perpendicular_line_through_point_l3868_386830


namespace isosceles_triangle_perimeter_l3868_386864

/-- An isosceles triangle with two sides of lengths 3 and 4 has a perimeter of either 10 or 11. -/
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a = b ∧ (a = 3 ∧ c = 4 ∨ a = 4 ∧ c = 3)) ∨ (a = c ∧ (a = 3 ∧ b = 4 ∨ a = 4 ∧ b = 3)) ∨ (b = c ∧ (b = 3 ∧ a = 4 ∨ b = 4 ∧ a = 3)) →
  a + b + c = 10 ∨ a + b + c = 11 :=
by
  sorry


end isosceles_triangle_perimeter_l3868_386864


namespace dan_gave_fourteen_marbles_l3868_386807

/-- The number of marbles Dan gave to Mary -/
def marbles_given (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proof that Dan gave 14 marbles to Mary -/
theorem dan_gave_fourteen_marbles :
  let initial := 64
  let remaining := 50
  marbles_given initial remaining = 14 := by
sorry

end dan_gave_fourteen_marbles_l3868_386807


namespace hyperbolas_same_asymptotes_l3868_386875

/-- Two hyperbolas have the same asymptotes if M = 18 -/
theorem hyperbolas_same_asymptotes (M : ℝ) :
  (∀ x y : ℝ, x^2/9 - y^2/16 = 1 ↔ y^2/32 - x^2/M = 1) →
  M = 18 :=
by sorry

end hyperbolas_same_asymptotes_l3868_386875


namespace binary_101111011_equals_379_l3868_386892

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary_101111011 : List Bool := [true, true, false, true, true, true, true, false, true]

theorem binary_101111011_equals_379 :
  binary_to_decimal binary_101111011 = 379 := by
  sorry

end binary_101111011_equals_379_l3868_386892


namespace syllogism_arrangement_l3868_386814

-- Define the property of being divisible by 2
def divisible_by_2 (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Define odd numbers
def odd (n : ℕ) : Prop := ¬(divisible_by_2 n)

-- State the theorem
theorem syllogism_arrangement :
  (∀ n : ℕ, odd n → ¬(divisible_by_2 n)) →  -- Statement ②
  (odd 2013) →                              -- Statement ③
  ¬(divisible_by_2 2013)                    -- Statement ①
  := by sorry

end syllogism_arrangement_l3868_386814


namespace johnny_earnings_l3868_386856

/-- Calculates the total earnings from two jobs with overtime --/
def total_earnings (
  job1_rate : ℚ)
  (job1_hours : ℕ)
  (job1_regular_hours : ℕ)
  (job1_overtime_multiplier : ℚ)
  (job2_rate : ℚ)
  (job2_hours : ℕ) : ℚ :=
  let job1_regular_pay := job1_rate * (job1_regular_hours : ℚ)
  let job1_overtime_hours := job1_hours - job1_regular_hours
  let job1_overtime_rate := job1_rate * job1_overtime_multiplier
  let job1_overtime_pay := job1_overtime_rate * (job1_overtime_hours : ℚ)
  let job1_total_pay := job1_regular_pay + job1_overtime_pay
  let job2_pay := job2_rate * (job2_hours : ℚ)
  job1_total_pay + job2_pay

/-- Johnny's total earnings from two jobs with overtime --/
theorem johnny_earnings : 
  total_earnings 3.25 8 6 1.5 4.5 5 = 58.25 := by
  sorry

end johnny_earnings_l3868_386856


namespace tape_pieces_for_cube_l3868_386852

/-- Represents a cube with side length n -/
structure Cube where
  sideLength : ℕ

/-- Represents a piece of tape with width 1 cm -/
structure Tape where
  length : ℕ

/-- Function to calculate the number of tape pieces needed to cover a cube -/
def tapePiecesNeeded (c : Cube) : ℕ :=
  2 * c.sideLength

/-- Theorem stating that the number of tape pieces needed is 2n -/
theorem tape_pieces_for_cube (c : Cube) :
  tapePiecesNeeded c = 2 * c.sideLength := by
  sorry

#check tape_pieces_for_cube

end tape_pieces_for_cube_l3868_386852


namespace divisibility_in_chosen_numbers_l3868_386828

theorem divisibility_in_chosen_numbers (n : ℕ+) :
  ∀ (S : Finset ℕ), S ⊆ Finset.range (2*n + 1) → S.card = n + 1 →
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ b % a = 0 :=
by sorry

end divisibility_in_chosen_numbers_l3868_386828


namespace cos_10_coeff_sum_l3868_386844

/-- Represents the coefficients in the expansion of cos(10α) -/
structure Cos10Coeffs where
  m : ℤ
  n : ℤ
  p : ℤ

/-- 
Given the equation for cos(10α) in the form:
cos(10α) = m*cos^10(α) - 1280*cos^8(α) + 1120*cos^6(α) + n*cos^4(α) + p*cos^2(α) - 1,
prove that m - n + p = 962
-/
theorem cos_10_coeff_sum (coeffs : Cos10Coeffs) : coeffs.m - coeffs.n + coeffs.p = 962 := by
  sorry

end cos_10_coeff_sum_l3868_386844


namespace infinite_series_sum_l3868_386842

theorem infinite_series_sum : 
  let series := fun n : ℕ => (n + 1 : ℝ) / 5^n
  ∑' n, series n = 9/16 := by sorry

end infinite_series_sum_l3868_386842


namespace monochromatic_rectangle_exists_l3868_386865

/-- A color type representing red, green, or blue -/
inductive Color
| Red
| Green
| Blue

/-- A type representing a 4 × 82 grid where each cell is colored -/
def Grid := Fin 4 → Fin 82 → Color

/-- A function to check if four points form a rectangle with the same color -/
def isMonochromaticRectangle (g : Grid) (x1 y1 x2 y2 : ℕ) : Prop :=
  x1 < x2 ∧ y1 < y2 ∧
  g ⟨x1, by sorry⟩ ⟨y1, by sorry⟩ = g ⟨x1, by sorry⟩ ⟨y2, by sorry⟩ ∧
  g ⟨x1, by sorry⟩ ⟨y1, by sorry⟩ = g ⟨x2, by sorry⟩ ⟨y1, by sorry⟩ ∧
  g ⟨x1, by sorry⟩ ⟨y1, by sorry⟩ = g ⟨x2, by sorry⟩ ⟨y2, by sorry⟩

/-- Theorem: In any 3-coloring of a 4 × 82 grid, there exists a rectangle whose vertices are all the same color -/
theorem monochromatic_rectangle_exists (g : Grid) :
  ∃ (x1 y1 x2 y2 : ℕ), isMonochromaticRectangle g x1 y1 x2 y2 :=
sorry

end monochromatic_rectangle_exists_l3868_386865


namespace banana_orange_equivalence_l3868_386826

/-- The cost relationship between bananas, apples, and oranges -/
structure FruitCosts where
  banana_apple_ratio : ℚ  -- 5 bananas cost as much as 3 apples
  apple_orange_ratio : ℚ  -- 9 apples cost as much as 6 oranges

/-- The theorem stating the cost equivalence of 30 bananas and 12 oranges -/
theorem banana_orange_equivalence (fc : FruitCosts) 
  (h1 : fc.banana_apple_ratio = 5 / 3)
  (h2 : fc.apple_orange_ratio = 9 / 6) : 
  (30 : ℚ) / fc.banana_apple_ratio * fc.apple_orange_ratio = 12 := by
  sorry

#check banana_orange_equivalence

end banana_orange_equivalence_l3868_386826


namespace line_parameterization_l3868_386858

/-- Given a line y = 5x - 7 parameterized by (x, y) = (s, 2) + t(3, m),
    prove that s = 9/5 and m = 3 -/
theorem line_parameterization (s m : ℝ) :
  (∀ t : ℝ, ∀ x y : ℝ, 
    x = s + 3*t ∧ y = 2 + m*t → y = 5*x - 7) →
  s = 9/5 ∧ m = 3 := by
sorry

end line_parameterization_l3868_386858


namespace min_value_sum_cubic_ratios_l3868_386804

theorem min_value_sum_cubic_ratios (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 9) :
  (x^3 + y^3) / (x + y) + (x^3 + z^3) / (x + z) + (y^3 + z^3) / (y + z) ≥ 27 := by
  sorry

end min_value_sum_cubic_ratios_l3868_386804


namespace division_problem_l3868_386861

theorem division_problem (x y : ℕ+) 
  (h1 : (x : ℝ) / (y : ℝ) = 96.12)
  (h2 : (x : ℝ) % (y : ℝ) = 11.52) : 
  y = 96 := by
  sorry

end division_problem_l3868_386861


namespace sqrt_76_between_8_and_9_l3868_386880

theorem sqrt_76_between_8_and_9 : 8 < Real.sqrt 76 ∧ Real.sqrt 76 < 9 := by
  sorry

end sqrt_76_between_8_and_9_l3868_386880


namespace min_value_problem_l3868_386809

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (4 / a + 9 / b) ≥ 25 := by
sorry

end min_value_problem_l3868_386809


namespace wang_parts_processed_l3868_386801

/-- Represents the number of parts processed by a worker in a given time -/
def parts_processed (rate : ℕ) (time : ℕ) : ℕ := rate * time

/-- Represents Xiao Wang's work cycle -/
def wang_cycle (total_time : ℕ) : ℕ :=
  parts_processed 15 (2 * (total_time / 3))

/-- Represents Xiao Li's work -/
def li_work (total_time : ℕ) : ℕ :=
  parts_processed 12 total_time

theorem wang_parts_processed :
  ∃ (t : ℕ), t > 0 ∧ wang_cycle t = li_work t ∧ wang_cycle t = 60 :=
sorry

end wang_parts_processed_l3868_386801


namespace textbook_reading_time_l3868_386835

/-- Calculates the total reading time in hours for a textbook with given parameters. -/
def totalReadingTime (totalChapters : ℕ) (readingTimePerChapter : ℕ) : ℚ :=
  let chaptersRead := totalChapters - (totalChapters / 3)
  (chaptersRead * readingTimePerChapter : ℚ) / 60

/-- Proves that the total reading time for the given textbook is 7 hours. -/
theorem textbook_reading_time :
  totalReadingTime 31 20 = 7 := by
  sorry

#eval totalReadingTime 31 20

end textbook_reading_time_l3868_386835


namespace cos_squared_minus_sin_squared_pi_12_l3868_386832

theorem cos_squared_minus_sin_squared_pi_12 : 
  Real.cos (π / 12) ^ 2 - Real.sin (π / 12) ^ 2 = Real.sqrt 3 / 2 := by sorry

end cos_squared_minus_sin_squared_pi_12_l3868_386832


namespace abs_inequality_solution_set_l3868_386808

theorem abs_inequality_solution_set (x : ℝ) :
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by sorry

end abs_inequality_solution_set_l3868_386808


namespace quadratic_to_alternative_form_integer_values_iff_integer_coefficients_l3868_386873

/-- Represents a quadratic expression Ax² + Bx + C -/
structure QuadraticExpression (α : Type) [Ring α] where
  A : α
  B : α
  C : α

/-- Represents the alternative form k(x(x-1)/2) + lx + m -/
structure AlternativeForm (α : Type) [Ring α] where
  k : α
  l : α
  m : α

/-- States that a quadratic expression can be written in the alternative form -/
theorem quadratic_to_alternative_form {α : Type} [Ring α] (q : QuadraticExpression α) :
  ∃ (a : AlternativeForm α), a.k = 2 * q.A ∧ a.l = q.A + q.B ∧ a.m = q.C :=
sorry

/-- States that the quadratic expression takes integer values for all integer x
    if and only if k, l, m in the alternative form are integers -/
theorem integer_values_iff_integer_coefficients (q : QuadraticExpression ℤ) :
  (∀ x : ℤ, ∃ y : ℤ, q.A * x^2 + q.B * x + q.C = y) ↔
  (∃ (a : AlternativeForm ℤ), a.k = 2 * q.A ∧ a.l = q.A + q.B ∧ a.m = q.C) :=
sorry

end quadratic_to_alternative_form_integer_values_iff_integer_coefficients_l3868_386873


namespace first_concert_attendance_l3868_386862

theorem first_concert_attendance (second_concert : ℕ) (difference : ℕ) : 
  second_concert = 66018 → difference = 119 → second_concert - difference = 65899 := by
  sorry

end first_concert_attendance_l3868_386862


namespace roots_equal_opposite_signs_l3868_386846

theorem roots_equal_opposite_signs (c d e : ℝ) :
  (∃ y₁ y₂ : ℝ, y₁ = -y₂ ∧ y₁ ≠ 0 ∧
    (y₁^2 - d*y₁) / (c*y₁ - e) = (n - 2) / (n + 2) ∧
    (y₂^2 - d*y₂) / (c*y₂ - e) = (n - 2) / (n + 2)) →
  n = -2 :=
by sorry

end roots_equal_opposite_signs_l3868_386846
