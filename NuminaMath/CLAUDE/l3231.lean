import Mathlib

namespace roots_of_equation_l3231_323116

theorem roots_of_equation (x : ℝ) : 
  x * (x - 1) = 0 ↔ x = 0 ∨ x = 1 := by sorry

end roots_of_equation_l3231_323116


namespace sum_of_multiples_is_even_l3231_323105

theorem sum_of_multiples_is_even (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 6 * k) 
  (hb : ∃ m : ℤ, b = 8 * m) : 
  ∃ n : ℤ, a + b = 2 * n :=
sorry

end sum_of_multiples_is_even_l3231_323105


namespace cos_45_degrees_l3231_323152

theorem cos_45_degrees : Real.cos (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_45_degrees_l3231_323152


namespace team_a_games_l3231_323164

theorem team_a_games (a : ℕ) (h1 : 3 * a = 4 * (a - (a / 4)))
  (h2 : 2 * (a + 16) = 3 * ((a + 16) - ((a + 16) / 3)))
  (h3 : (a + 16) - ((a + 16) / 3) = a - (a / 4) + 8)
  (h4 : ((a + 16) / 3) = (a / 4) + 8) : a = 192 := by
  sorry

end team_a_games_l3231_323164


namespace arithmetic_sequence_product_inequality_l3231_323137

/-- An arithmetic sequence of 8 terms with positive values and non-zero common difference -/
structure ArithmeticSequence8 where
  a : Fin 8 → ℝ
  positive : ∀ i, a i > 0
  is_arithmetic : ∃ d ≠ 0, ∀ i j, a j - a i = (j - i : ℝ) * d

theorem arithmetic_sequence_product_inequality (seq : ArithmeticSequence8) :
  seq.a 0 * seq.a 7 < seq.a 3 * seq.a 4 := by
  sorry

end arithmetic_sequence_product_inequality_l3231_323137


namespace preceding_binary_l3231_323121

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def decimal_to_binary (n : ℕ) : List Bool :=
  if n = 0 then
    [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then
        []
      else
        (m % 2 = 1) :: aux (m / 2)
    aux n |>.reverse

theorem preceding_binary (N : List Bool) :
  N = [true, true, true, false, false] →
  decimal_to_binary (binary_to_decimal N - 1) = [true, true, false, true, true] := by
  sorry

end preceding_binary_l3231_323121


namespace probability_is_half_l3231_323182

/-- A game board represented as a regular hexagon -/
structure HexagonalBoard :=
  (total_segments : ℕ)
  (shaded_segments : ℕ)
  (is_regular : total_segments = 6)
  (shaded_constraint : shaded_segments = 3)

/-- The probability of a spinner landing on a shaded region of a hexagonal board -/
def probability_shaded (board : HexagonalBoard) : ℚ :=
  board.shaded_segments / board.total_segments

/-- Theorem stating that the probability of landing on a shaded region is 1/2 -/
theorem probability_is_half (board : HexagonalBoard) :
  probability_shaded board = 1 / 2 := by
  sorry

end probability_is_half_l3231_323182


namespace prob_X_or_Y_or_Z_wins_l3231_323115

-- Define the probabilities
def prob_X : ℚ := 1/4
def prob_Y : ℚ := 1/8
def prob_Z : ℚ := 1/12

-- Define the total number of cars
def total_cars : ℕ := 15

-- Theorem statement
theorem prob_X_or_Y_or_Z_wins : 
  prob_X + prob_Y + prob_Z = 11/24 := by sorry

end prob_X_or_Y_or_Z_wins_l3231_323115


namespace ratio_of_sum_and_difference_l3231_323145

theorem ratio_of_sum_and_difference (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (h : a + b = 5 * (a - b)) : a / b = 3 / 2 := by
  sorry

end ratio_of_sum_and_difference_l3231_323145


namespace cone_lateral_surface_area_l3231_323143

/-- The lateral surface area of a cone with base radius 3 and height 4 is 15π. -/
theorem cone_lateral_surface_area :
  ∀ (r h : ℝ) (lateral_area : ℝ),
    r = 3 →
    h = 4 →
    lateral_area = π * r * (Real.sqrt (r^2 + h^2)) →
    lateral_area = 15 * π :=
by sorry

end cone_lateral_surface_area_l3231_323143


namespace largest_value_when_x_is_quarter_l3231_323180

theorem largest_value_when_x_is_quarter (x : ℝ) (h : x = 1/4) :
  (1/x > x) ∧ (1/x > x^2) ∧ (1/x > (1/2)*x) ∧ (1/x > Real.sqrt x) := by
  sorry

end largest_value_when_x_is_quarter_l3231_323180


namespace hex_to_binary_bits_l3231_323186

/-- The number of bits required to represent a positive integer in binary. -/
def bitsRequired (n : ℕ) : ℕ :=
  if n = 0 then 0 else Nat.log2 n + 1

/-- The decimal representation of the hexadecimal number 1A1A1. -/
def hexNumber : ℕ := 106913

theorem hex_to_binary_bits :
  bitsRequired hexNumber = 17 := by
  sorry

end hex_to_binary_bits_l3231_323186


namespace river_crossing_theorem_l3231_323148

/-- Calculates the time required for all explorers to cross a river --/
def river_crossing_time (num_explorers : ℕ) (boat_capacity : ℕ) (crossing_time : ℕ) : ℕ :=
  let first_trip := boat_capacity
  let remaining_explorers := num_explorers - first_trip
  let subsequent_trips := (remaining_explorers + 4) / 5  -- Ceiling division
  let total_crossings := 2 * subsequent_trips + 1
  total_crossings * crossing_time

theorem river_crossing_theorem :
  river_crossing_time 60 6 3 = 69 := by
  sorry

end river_crossing_theorem_l3231_323148


namespace triangle_abc_properties_l3231_323166

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π →
  -- Given conditions
  b * (Real.sin B - Real.sin C) + (c - a) * (Real.sin A + Real.sin C) = 0 →
  a = Real.sqrt 3 →
  Real.sin C = (1 + Real.sqrt 3) / 2 * Real.sin B →
  -- Conclusions
  A = π / 3 ∧
  (1 / 2) * a * b * Real.sin C = (3 + Real.sqrt 3) / 4 := by
sorry


end triangle_abc_properties_l3231_323166


namespace arithmetic_sequence_a2_l3231_323102

/-- An arithmetic sequence with a_1 = 1 and a_{n+2} - a_n = 3 has a_2 = 5/2 -/
theorem arithmetic_sequence_a2 (a : ℕ → ℚ) :
  a 1 = 1 →
  (∀ n : ℕ, a (n + 2) - a n = 3) →
  (∀ n : ℕ, ∃ d : ℚ, a (n + 1) = a n + d) →
  a 2 = 5/2 :=
by
  sorry

end arithmetic_sequence_a2_l3231_323102


namespace gaming_system_value_proof_l3231_323192

/-- The value of Tom's gaming system -/
def gaming_system_value : ℝ := 150

/-- The percentage of the gaming system's value given as store credit -/
def store_credit_percentage : ℝ := 0.80

/-- The amount Tom pays in cash -/
def cash_paid : ℝ := 80

/-- The change Tom receives -/
def change_received : ℝ := 10

/-- The value of the game Tom receives -/
def game_value : ℝ := 30

/-- The cost of the NES -/
def nes_cost : ℝ := 160

theorem gaming_system_value_proof :
  store_credit_percentage * gaming_system_value + cash_paid - change_received = nes_cost + game_value :=
by sorry

end gaming_system_value_proof_l3231_323192


namespace min_value_arithmetic_sequence_l3231_323172

theorem min_value_arithmetic_sequence (a : ℝ) (m : ℕ+) :
  (∃ (S : ℕ+ → ℝ), S m = 36 ∧ 
    (∀ n : ℕ+, S n = n * a - 4 * (n * (n - 1)) / 2)) →
  ∀ a' : ℝ, (∃ m' : ℕ+, ∃ S' : ℕ+ → ℝ, 
    S' m' = 36 ∧ 
    (∀ n : ℕ+, S' n = n * a' - 4 * (n * (n - 1)) / 2)) →
  a' ≥ 15 :=
by sorry

end min_value_arithmetic_sequence_l3231_323172


namespace largest_solution_is_25_l3231_323132

theorem largest_solution_is_25 :
  ∃ (x : ℝ), (x^2 + x - 1 + |x^2 - (x - 1)|) / 2 = 35*x - 250 ∧
  x = 25 ∧
  ∀ (y : ℝ), (y^2 + y - 1 + |y^2 - (y - 1)|) / 2 = 35*y - 250 → y ≤ 25 :=
by sorry

end largest_solution_is_25_l3231_323132


namespace equation_D_is_quadratic_l3231_323161

/-- Definition of a quadratic equation in terms of x -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 - 5x = 0 -/
def equation_D (x : ℝ) : ℝ := x^2 - 5*x

/-- Theorem: equation_D is a quadratic equation -/
theorem equation_D_is_quadratic : is_quadratic_equation equation_D := by
  sorry

end equation_D_is_quadratic_l3231_323161


namespace ratio_equation_solution_l3231_323190

theorem ratio_equation_solution (c d : ℚ) 
  (h1 : c / d = 4)
  (h2 : c = 15 - 3 * d) : 
  d = 15 / 7 := by sorry

end ratio_equation_solution_l3231_323190


namespace percentage_calculation_l3231_323111

theorem percentage_calculation (total : ℝ) (difference : ℝ) : 
  total = 6000 ∧ difference = 693 → 
  ∃ P : ℝ, (1/10 * total) - (P/100 * total) = difference ∧ P = 1.55 := by
sorry

end percentage_calculation_l3231_323111


namespace balls_total_weight_l3231_323163

/-- Represents the total weight of five colored metal balls -/
def total_weight (blue brown green red yellow : ℝ) : ℝ :=
  blue + brown + green + red + yellow

/-- Theorem stating the total weight of the balls -/
theorem balls_total_weight :
  ∃ (blue brown green red yellow : ℝ),
    blue = 6 ∧
    brown = 3.12 ∧
    green = 4.25 ∧
    red = 2 * green ∧
    yellow = red - 1.5 ∧
    total_weight blue brown green red yellow = 28.87 := by
  sorry

end balls_total_weight_l3231_323163


namespace blueberry_pies_l3231_323174

theorem blueberry_pies (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) :
  total_pies = 30 →
  apple_ratio = 2 →
  blueberry_ratio = 3 →
  cherry_ratio = 5 →
  blueberry_ratio * (total_pies / (apple_ratio + blueberry_ratio + cherry_ratio)) = 9 :=
by sorry

end blueberry_pies_l3231_323174


namespace harveys_steak_sales_l3231_323155

/-- Represents the number of steaks Harvey sold after having 12 steaks left -/
def steaks_sold_after_12_left (initial_steaks : ℕ) (steaks_left : ℕ) (total_sold : ℕ) : ℕ :=
  total_sold - (initial_steaks - steaks_left)

/-- Theorem stating that Harvey sold 4 steaks after having 12 steaks left -/
theorem harveys_steak_sales : steaks_sold_after_12_left 25 12 17 = 4 := by
  sorry

end harveys_steak_sales_l3231_323155


namespace line_plane_perpendicular_l3231_323184

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular 
  (m n : Line) (α β : Plane) :
  perpPlane m α → perpPlane n β → perp m n → perpPlanes α β :=
sorry

end line_plane_perpendicular_l3231_323184


namespace infinite_product_equals_sqrt_two_l3231_323193

/-- The nth term of the sequence in the exponent -/
def a (n : ℕ) : ℚ := (2^n - 1) / (3^n)

/-- The infinite product as a function -/
noncomputable def infiniteProduct : ℝ := Real.rpow 2 (∑' n, a n)

/-- The theorem stating that the infinite product equals √2 -/
theorem infinite_product_equals_sqrt_two : infiniteProduct = Real.sqrt 2 := by sorry

end infinite_product_equals_sqrt_two_l3231_323193


namespace cost_per_metre_l3231_323175

/-- Given that John bought 9.25 m of cloth for $416.25, prove that the cost price per metre is $45. -/
theorem cost_per_metre (total_length : ℝ) (total_cost : ℝ) 
  (h1 : total_length = 9.25)
  (h2 : total_cost = 416.25) :
  total_cost / total_length = 45 := by
  sorry

end cost_per_metre_l3231_323175


namespace fine_payment_l3231_323198

theorem fine_payment (F : ℚ) 
  (hF : F > 0)
  (hJoe : F / 4 + 3 + F / 3 - 3 + F / 2 - 4 = F) : 
  F / 2 - 4 = 5 * F / 12 := by
  sorry

end fine_payment_l3231_323198


namespace ellipse_focal_distance_l3231_323104

/-- Given an ellipse with equation x²/(8-m) + y²/(m-2) = 1, 
    where the major axis is on the y-axis and the focal distance is 4,
    prove that the value of m is 7. -/
theorem ellipse_focal_distance (m : ℝ) : 
  (∀ x y : ℝ, x^2 / (8 - m) + y^2 / (m - 2) = 1) →  -- Ellipse equation
  (8 - m < m - 2) →                                -- Major axis on y-axis
  (m - 2 - (8 - m) = 4) →                          -- Focal distance is 4
  m = 7 := by
sorry

end ellipse_focal_distance_l3231_323104


namespace range_of_negative_power_function_l3231_323188

open Set
open Function
open Real

theorem range_of_negative_power_function {m : ℝ} (hm : m < 0) :
  let g : ℝ → ℝ := fun x ↦ x ^ m
  range (g ∘ (fun x ↦ x) : Set.Ioo 0 1 → ℝ) = Set.Ioi 1 := by
  sorry

end range_of_negative_power_function_l3231_323188


namespace pet_store_combinations_l3231_323159

def num_puppies : ℕ := 12
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 8
def num_rabbits : ℕ := 5
def num_people : ℕ := 4

theorem pet_store_combinations : 
  (num_puppies * num_kittens * num_hamsters * num_rabbits) * Nat.factorial num_people = 115200 :=
by sorry

end pet_store_combinations_l3231_323159


namespace solution_set_implies_k_inequality_implies_k_range_l3231_323138

/-- The quadratic function f(x) = kx^2 - 2x + 6k --/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 6 * k

/-- Theorem 1: If f(x) < 0 has solution set (2,3), then k = 2/5 --/
theorem solution_set_implies_k (k : ℝ) :
  (∀ x, f k x < 0 ↔ 2 < x ∧ x < 3) → k = 2/5 := by sorry

/-- Theorem 2: If k > 0 and f(x) < 0 for all 2 < x < 3, then 0 < k ≤ 2/5 --/
theorem inequality_implies_k_range (k : ℝ) :
  k > 0 → (∀ x, 2 < x → x < 3 → f k x < 0) → 0 < k ∧ k ≤ 2/5 := by sorry

end solution_set_implies_k_inequality_implies_k_range_l3231_323138


namespace product_from_lcm_gcd_l3231_323110

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 24) 
  (h2 : Nat.gcd a b = 8) : 
  a * b = 192 := by
  sorry

end product_from_lcm_gcd_l3231_323110


namespace abs_sum_gt_abs_prod_plus_one_implies_prod_zero_l3231_323142

theorem abs_sum_gt_abs_prod_plus_one_implies_prod_zero (a b : ℤ) : 
  |a + b| > |1 + a * b| → a * b = 0 := by
  sorry

end abs_sum_gt_abs_prod_plus_one_implies_prod_zero_l3231_323142


namespace factorial_problem_l3231_323169

-- Define the factorial function
def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_problem : (factorial 13 - factorial 12) / factorial 11 = 144 := by
  sorry

end factorial_problem_l3231_323169


namespace log_difference_l3231_323129

theorem log_difference (a b c d : ℕ+) 
  (h1 : (Real.log b) / (Real.log a) = 3/2)
  (h2 : (Real.log d) / (Real.log c) = 5/4)
  (h3 : a - c = 9) :
  b - d = 93 := by
  sorry

end log_difference_l3231_323129


namespace mary_regular_hours_l3231_323187

/-- Represents Mary's work schedule and earnings --/
structure WorkSchedule where
  regularHours : ℝ
  overtimeHours : ℝ
  regularRate : ℝ
  overtimeRate : ℝ
  maxHours : ℝ
  maxEarnings : ℝ

/-- Calculates total earnings based on work schedule --/
def totalEarnings (w : WorkSchedule) : ℝ :=
  w.regularHours * w.regularRate + w.overtimeHours * w.overtimeRate

/-- Theorem stating that Mary works 20 hours at her regular rate --/
theorem mary_regular_hours (w : WorkSchedule) 
  (h1 : w.maxHours = 80)
  (h2 : w.regularRate = 8)
  (h3 : w.overtimeRate = w.regularRate * 1.25)
  (h4 : w.maxEarnings = 760)
  (h5 : w.regularHours + w.overtimeHours = w.maxHours)
  (h6 : totalEarnings w = w.maxEarnings) :
  w.regularHours = 20 := by
  sorry

#check mary_regular_hours

end mary_regular_hours_l3231_323187


namespace equidistant_complex_function_l3231_323154

/-- A complex function f(z) = (a+bi)z with the property that f(z) is equidistant
    from z and 3z for all complex z, and |a+bi| = 5, implies b^2 = 21 -/
theorem equidistant_complex_function (a b : ℝ) : 
  (∀ z : ℂ, ‖(a + b * Complex.I) * z - z‖ = ‖(a + b * Complex.I) * z - 3 * z‖) →
  Complex.abs (a + b * Complex.I) = 5 →
  b^2 = 21 := by
sorry

end equidistant_complex_function_l3231_323154


namespace train_crossing_time_l3231_323141

/-- Calculates the time taken for a train to cross a signal pole -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (platform_crossing_time : ℝ) 
  (h1 : train_length = 300) 
  (h2 : platform_length = 250) 
  (h3 : platform_crossing_time = 33) :
  (train_length / ((train_length + platform_length) / platform_crossing_time)) = 18 :=
by sorry

end train_crossing_time_l3231_323141


namespace quadratic_is_square_of_binomial_l3231_323150

theorem quadratic_is_square_of_binomial (x k : ℝ) : 
  (∃ a b : ℝ, x^2 - 20*x + k = (a*x + b)^2) ↔ k = 100 := by
sorry

end quadratic_is_square_of_binomial_l3231_323150


namespace subset_implies_a_leq_4_l3231_323173

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + 4 ≥ 0}

-- State the theorem
theorem subset_implies_a_leq_4 : ∀ a : ℝ, A ⊆ B a → a ≤ 4 := by
  sorry

end subset_implies_a_leq_4_l3231_323173


namespace sin_20_cos_10_minus_cos_160_cos_80_l3231_323139

theorem sin_20_cos_10_minus_cos_160_cos_80 :
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) -
  Real.cos (160 * π / 180) * Real.cos (80 * π / 180) = 1 / 2 := by
  sorry

end sin_20_cos_10_minus_cos_160_cos_80_l3231_323139


namespace abc_inequality_l3231_323100

theorem abc_inequality (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 2) (hb : 0 ≤ b ∧ b ≤ 2) (hc : 0 ≤ c ∧ c ≤ 2) :
  (a - b) * (b - c) * (a - c) ≤ 2 ∧
  ((a - b) * (b - c) * (a - c) = 2 ↔ 
    ((a = 2 ∧ b = 1 ∧ c = 0) ∨ (a = 1 ∧ b = 0 ∧ c = 2) ∨ (a = 0 ∧ b = 2 ∧ c = 1))) :=
by sorry


end abc_inequality_l3231_323100


namespace cake_division_l3231_323181

theorem cake_division (num_cakes : ℕ) (num_children : ℕ) (max_cuts : ℕ) :
  num_cakes = 9 →
  num_children = 4 →
  max_cuts = 2 →
  ∃ (whole_cakes : ℕ) (fractional_cake : ℚ),
    whole_cakes + fractional_cake = num_cakes / num_children ∧
    whole_cakes = 2 ∧
    fractional_cake = 1/4 ∧
    (∀ cake, cake ≤ max_cuts) :=
by sorry

end cake_division_l3231_323181


namespace clock_rings_seven_times_l3231_323124

/-- Calculates the number of rings for a clock with given interval and day length -/
def number_of_rings (interval : ℕ) (day_length : ℕ) : ℕ :=
  (day_length / interval) + 1

/-- Theorem: A clock ringing every 4 hours in a 24-hour day rings 7 times -/
theorem clock_rings_seven_times : number_of_rings 4 24 = 7 := by
  sorry

end clock_rings_seven_times_l3231_323124


namespace like_terms_exponents_l3231_323136

theorem like_terms_exponents (m n : ℕ) : 
  (∀ x y : ℝ, ∃ k : ℝ, 2 * x^(n+2) * y^3 = k * (-3 * x^3 * y^(2*m-1))) → 
  (m = 2 ∧ n = 1) := by
sorry

end like_terms_exponents_l3231_323136


namespace smallest_n_for_integer_S_l3231_323103

/-- Definition of S_n as the sum of reciprocals of non-zero digits from 1 to 2·10^n -/
def S (n : ℕ) : ℚ :=
  sorry

/-- Theorem stating that 32 is the smallest positive integer n for which S_n is an integer -/
theorem smallest_n_for_integer_S :
  ∀ k : ℕ, k > 0 → k < 32 → ¬ (S k).isInt ∧ (S 32).isInt := by
  sorry

end smallest_n_for_integer_S_l3231_323103


namespace elderly_employees_in_sample_l3231_323178

theorem elderly_employees_in_sample
  (total_employees : ℕ)
  (young_employees : ℕ)
  (sample_young : ℕ)
  (h1 : total_employees = 430)
  (h2 : young_employees = 160)
  (h3 : sample_young = 32)
  (h4 : ∃ n : ℕ, total_employees = young_employees + 2 * n + n) :
  ∃ m : ℕ, m = 18 ∧ (sample_young : ℚ) / young_employees = (m : ℚ) / ((total_employees - young_employees) / 3) :=
by sorry

end elderly_employees_in_sample_l3231_323178


namespace floor_plus_self_eq_29_4_l3231_323149

theorem floor_plus_self_eq_29_4 (x : ℚ) :
  (⌊x⌋ : ℚ) + x = 29/4 → x = 29/4 := by
  sorry

end floor_plus_self_eq_29_4_l3231_323149


namespace max_congruent_spherical_triangles_l3231_323167

/-- A spherical triangle on the surface of a sphere --/
structure SphericalTriangle where
  -- Add necessary fields for a spherical triangle
  is_on_sphere : Bool
  sides_are_great_circle_arcs : Bool
  sides_less_than_quarter : Bool

/-- A division of a sphere into congruent spherical triangles --/
structure SphereDivision where
  triangles : List SphericalTriangle
  are_congruent : Bool

/-- The maximum number of congruent spherical triangles that satisfy the conditions --/
def max_congruent_triangles : ℕ := 60

/-- Theorem stating that 60 is the maximum number of congruent spherical triangles --/
theorem max_congruent_spherical_triangles :
  ∀ (d : SphereDivision),
    (∀ t ∈ d.triangles, t.is_on_sphere ∧ t.sides_are_great_circle_arcs ∧ t.sides_less_than_quarter) →
    d.are_congruent →
    d.triangles.length ≤ max_congruent_triangles :=
by
  sorry

#check max_congruent_spherical_triangles

end max_congruent_spherical_triangles_l3231_323167


namespace unique_successful_arrangement_l3231_323189

/-- Represents a cell in the table -/
inductive Cell
| One
| NegOne

/-- Represents a square table -/
def Table (n : ℕ) := Fin (2^n - 1) → Fin (2^n - 1) → Cell

/-- Checks if two cells are neighbors -/
def is_neighbor (n : ℕ) (i j i' j' : Fin (2^n - 1)) : Prop :=
  (i = i' ∧ (j.val + 1 = j'.val ∨ j.val = j'.val + 1)) ∨
  (j = j' ∧ (i.val + 1 = i'.val ∨ i.val = i'.val + 1))

/-- Checks if a table is a successful arrangement -/
def is_successful (n : ℕ) (t : Table n) : Prop :=
  ∀ i j, t i j = Cell.One ↔ 
    ∀ i' j', is_neighbor n i j i' j' → t i' j' = Cell.One

/-- The main theorem -/
theorem unique_successful_arrangement (n : ℕ) :
  ∃! t : Table n, is_successful n t ∧ (∀ i j, t i j = Cell.One) :=
sorry

end unique_successful_arrangement_l3231_323189


namespace cigar_purchase_problem_l3231_323114

theorem cigar_purchase_problem :
  ∃ (x y z : ℕ),
    x + y + z = 100 ∧
    (1/2 : ℚ) * x + 3 * y + 10 * z = 100 ∧
    x = 94 ∧ y = 1 ∧ z = 5 := by
  sorry

end cigar_purchase_problem_l3231_323114


namespace equation_solution_l3231_323120

theorem equation_solution : ∃ x : ℚ, 9 - 3 / (x / 3) + 3 = 3 :=
by
  use 1
  sorry

end equation_solution_l3231_323120


namespace plane_q_satisfies_conditions_l3231_323165

/-- Plane type representing ax + by + cz + d = 0 --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Line type representing the intersection of two planes --/
structure Line where
  p1 : Plane
  p2 : Plane

/-- Point type in 3D space --/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Function to check if a plane contains a line --/
def containsLine (p : Plane) (l : Line) : Prop := sorry

/-- Function to calculate the distance between a plane and a point --/
def distancePlanePoint (p : Plane) (pt : Point) : ℝ := sorry

/-- Given planes --/
def plane1 : Plane := ⟨1, 3, 2, -4⟩
def plane2 : Plane := ⟨2, -1, 3, -6⟩

/-- Line M --/
def lineM : Line := ⟨plane1, plane2⟩

/-- Given point --/
def givenPoint : Point := ⟨4, 2, -2⟩

/-- Plane Q --/
def planeQ : Plane := ⟨1, -9, 5, -2⟩

theorem plane_q_satisfies_conditions :
  containsLine planeQ lineM ∧
  planeQ ≠ plane1 ∧
  planeQ ≠ plane2 ∧
  distancePlanePoint planeQ givenPoint = 3 / Real.sqrt 5 := by
  sorry

end plane_q_satisfies_conditions_l3231_323165


namespace athletes_game_count_l3231_323101

theorem athletes_game_count (malik_yards josiah_yards darnell_yards total_yards : ℕ) 
  (h1 : malik_yards = 18)
  (h2 : josiah_yards = 22)
  (h3 : darnell_yards = 11)
  (h4 : total_yards = 204) :
  ∃ n : ℕ, n * (malik_yards + josiah_yards + darnell_yards) = total_yards ∧ n = 4 := by
sorry

end athletes_game_count_l3231_323101


namespace solve_chips_problem_l3231_323144

def chips_problem (total father_chips brother_chips : ℕ) : Prop :=
  total = 800 ∧ father_chips = 268 ∧ brother_chips = 182 →
  total - (father_chips + brother_chips) = 350

theorem solve_chips_problem :
  ∀ (total father_chips brother_chips : ℕ),
    chips_problem total father_chips brother_chips :=
by
  sorry

end solve_chips_problem_l3231_323144


namespace cuboid_volume_calculation_l3231_323128

def cuboid_volume (length width height : ℝ) : ℝ := length * width * height

theorem cuboid_volume_calculation :
  let length : ℝ := 6
  let width : ℝ := 5
  let height : ℝ := 6
  cuboid_volume length width height = 180 := by
  sorry

end cuboid_volume_calculation_l3231_323128


namespace complex_fraction_fourth_quadrant_l3231_323185

/-- Given that (1+i)/(2-i) = a + (b+1)i where a and b are real numbers and i is the imaginary unit,
    prove that the point corresponding to z = a + bi lies in the fourth quadrant of the complex plane. -/
theorem complex_fraction_fourth_quadrant (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (1 + i) / (2 - i) = a + (b + 1) * i →
  0 < a ∧ b < 0 :=
sorry

end complex_fraction_fourth_quadrant_l3231_323185


namespace baking_dish_recipe_book_ratio_l3231_323197

/-- The cost of Liz's purchases -/
def total_cost : ℚ := 40

/-- The cost of the recipe book -/
def recipe_book_cost : ℚ := 6

/-- The cost of each ingredient -/
def ingredient_cost : ℚ := 3

/-- The number of ingredients purchased -/
def num_ingredients : ℕ := 5

/-- The additional cost of the apron compared to the recipe book -/
def apron_extra_cost : ℚ := 1

/-- The ratio of the baking dish cost to the recipe book cost -/
def baking_dish_to_recipe_book_ratio : ℚ := 2

theorem baking_dish_recipe_book_ratio :
  (total_cost - (recipe_book_cost + (recipe_book_cost + apron_extra_cost) + 
   (ingredient_cost * num_ingredients))) / recipe_book_cost = baking_dish_to_recipe_book_ratio := by
  sorry

end baking_dish_recipe_book_ratio_l3231_323197


namespace spring_sports_event_probabilities_l3231_323123

def male_volunteers : ℕ := 4
def female_volunteers : ℕ := 3
def team_size : ℕ := 3

def total_volunteers : ℕ := male_volunteers + female_volunteers

theorem spring_sports_event_probabilities :
  let p_at_least_one_female := 1 - (Nat.choose male_volunteers team_size : ℚ) / (Nat.choose total_volunteers team_size : ℚ)
  let p_all_male_given_at_least_one_male := 
    (Nat.choose male_volunteers team_size : ℚ) / 
    ((Nat.choose total_volunteers team_size : ℚ) - (Nat.choose female_volunteers team_size : ℚ))
  p_at_least_one_female = 31 / 35 ∧ 
  p_all_male_given_at_least_one_male = 2 / 17 := by
  sorry

end spring_sports_event_probabilities_l3231_323123


namespace quadratic_solution_property_l3231_323177

theorem quadratic_solution_property (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 2 * a + 4 * b = -2 := by
  sorry

end quadratic_solution_property_l3231_323177


namespace problem_statement_l3231_323135

theorem problem_statement : (-0.125)^2007 * (-8)^2008 = -8 := by
  sorry

end problem_statement_l3231_323135


namespace special_function_value_l3231_323134

/-- A function satisfying the given property -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ m n : ℝ, f (m + n^2) = f m + 2 * (f n)^2

theorem special_function_value (f : ℝ → ℝ) 
  (h1 : special_function f) 
  (h2 : f 1 ≠ 0) : 
  f 2014 = 1007 := by
  sorry

end special_function_value_l3231_323134


namespace num_finches_is_four_l3231_323107

-- Define the constants based on the problem conditions
def parakeet_consumption : ℕ := 2 -- grams per day
def parrot_consumption : ℕ := 14 -- grams per day
def finch_consumption : ℕ := parakeet_consumption / 2 -- grams per day
def num_parakeets : ℕ := 3
def num_parrots : ℕ := 2
def total_birdseed : ℕ := 266 -- grams for a week
def days_in_week : ℕ := 7

-- Theorem to prove
theorem num_finches_is_four :
  ∃ (num_finches : ℕ),
    num_finches = 4 ∧
    total_birdseed = (num_parakeets * parakeet_consumption + 
                      num_parrots * parrot_consumption + 
                      num_finches * finch_consumption) * days_in_week :=
by
  sorry


end num_finches_is_four_l3231_323107


namespace exactly_two_true_l3231_323170

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the four propositions
def Prop1 (f : ℝ → ℝ) : Prop := IsOdd f → f 0 = 0
def Prop2 (f : ℝ → ℝ) : Prop := f 0 = 0 → IsOdd f
def Prop3 (f : ℝ → ℝ) : Prop := ¬(IsOdd f) → f 0 ≠ 0
def Prop4 (f : ℝ → ℝ) : Prop := f 0 ≠ 0 → ¬(IsOdd f)

-- The main theorem
theorem exactly_two_true (f : ℝ → ℝ) : 
  IsOdd f → (Prop1 f ∧ Prop4 f ∧ ¬Prop2 f ∧ ¬Prop3 f) := by sorry

end exactly_two_true_l3231_323170


namespace binomial_expansion_theorem_l3231_323127

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) : 
  n ≥ 2 → 
  k > 0 → 
  a * b ≠ 0 → 
  a = (k + 1) * b → 
  (n.choose 1 * (k * b)^(n - 1) * (-b) + n.choose 2 * (k * b)^(n - 2) * (-b)^2 = k * b^n * k^(n - 2)) → 
  n = 2 * k + 2 := by
sorry

end binomial_expansion_theorem_l3231_323127


namespace sum_reciprocals_bound_l3231_323113

theorem sum_reciprocals_bound (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (sum : a + b = 1) :
  1/a + 1/b > 4 := by
sorry

end sum_reciprocals_bound_l3231_323113


namespace large_block_volume_l3231_323183

/-- Volume of a rectangular block -/
def volume (width depth length : ℝ) : ℝ := width * depth * length

theorem large_block_volume :
  ∀ (w d l : ℝ),
  volume w d l = 4 →
  volume (2 * w) (2 * d) (2 * l) = 32 := by
  sorry

end large_block_volume_l3231_323183


namespace greatest_common_divisor_under_60_l3231_323108

theorem greatest_common_divisor_under_60 : ∃ (n : ℕ), 
  n < 60 ∧ 
  n ∣ 546 ∧ 
  n ∣ 108 ∧ 
  (∀ m : ℕ, m < 60 → m ∣ 546 → m ∣ 108 → m ≤ n) ∧
  n = 42 := by
sorry

end greatest_common_divisor_under_60_l3231_323108


namespace inequality_solution_count_l3231_323106

theorem inequality_solution_count : 
  (∃ (S : Finset ℕ), 
    (∀ n ∈ S, (n : ℝ) + 6 * ((n : ℝ) - 1) * ((n : ℝ) - 15) < 0) ∧ 
    (∀ n : ℕ, (n : ℝ) + 6 * ((n : ℝ) - 1) * ((n : ℝ) - 15) < 0 → n ∈ S) ∧
    Finset.card S = 13) := by
  sorry

end inequality_solution_count_l3231_323106


namespace files_remaining_l3231_323191

theorem files_remaining (music_files video_files deleted_files : ℕ) 
  (h1 : music_files = 26)
  (h2 : video_files = 36)
  (h3 : deleted_files = 48) :
  music_files + video_files - deleted_files = 14 := by
  sorry

end files_remaining_l3231_323191


namespace even_increasing_function_solution_set_l3231_323146

-- Define the function f
def f (a b x : ℝ) : ℝ := (x - 2) * (a * x + b)

-- State the theorem
theorem even_increasing_function_solution_set
  (a b : ℝ)
  (h_even : ∀ x, f a b x = f a b (-x))
  (h_increasing : ∀ x y, 0 < x → x < y → f a b x < f a b y)
  : {x : ℝ | f a b (2 - x) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by sorry

end even_increasing_function_solution_set_l3231_323146


namespace intersection_of_A_and_B_l3231_323157

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {0, 2, 3, 4}

theorem intersection_of_A_and_B : A ∩ B = {2, 3} := by sorry

end intersection_of_A_and_B_l3231_323157


namespace chosen_number_proof_l3231_323199

theorem chosen_number_proof (x : ℝ) : (x / 6) - 189 = 3 → x = 1152 := by
  sorry

end chosen_number_proof_l3231_323199


namespace downstream_speed_theorem_l3231_323168

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed given the rowing speeds in still water and upstream -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

/-- Theorem stating that given the specific conditions, the downstream speed is 31 kmph -/
theorem downstream_speed_theorem (s : RowingSpeed) 
  (h1 : s.stillWater = 28) 
  (h2 : s.upstream = 25) : 
  downstreamSpeed s = 31 := by
  sorry

#check downstream_speed_theorem

end downstream_speed_theorem_l3231_323168


namespace circle_tangent_to_parallel_lines_l3231_323151

/-- A circle is tangent to two parallel lines and its center lies on a third line. -/
theorem circle_tangent_to_parallel_lines (x y : ℝ) :
  (3 * x - 4 * y = 12 ∨ 3 * x - 4 * y = -48) ∧ 
  (x - 2 * y = 0) →
  x = -18 ∧ y = -9 := by
  sorry

end circle_tangent_to_parallel_lines_l3231_323151


namespace prob_not_six_four_dice_value_l3231_323156

/-- The probability that (a-6)(b-6)(c-6)(d-6) ≠ 0 when four standard dice are tossed -/
def prob_not_six_four_dice : ℚ :=
  625 / 1296

/-- Theorem stating that the probability of (a-6)(b-6)(c-6)(d-6) ≠ 0 
    when four standard dice are tossed is equal to 625/1296 -/
theorem prob_not_six_four_dice_value : 
  prob_not_six_four_dice = 625 / 1296 := by
  sorry

end prob_not_six_four_dice_value_l3231_323156


namespace exists_perpendicular_line_l3231_323117

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define a relation for a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define a relation for perpendicularity between lines
variable (perpendicular : Line → Line → Prop)

-- Theorem statement
theorem exists_perpendicular_line (a : Line) (α : Plane) :
  ∃ l : Line, in_plane l α ∧ perpendicular l a :=
sorry

end exists_perpendicular_line_l3231_323117


namespace difference_is_64_l3231_323122

/-- Defines the sequence a_n based on the given recurrence relation -/
def a : ℕ → ℕ → ℕ
  | n, x => if n = 0 then x
            else if x % 2 = 0 then a (n-1) (x / 2)
            else a (n-1) (3 * x + 1)

/-- Returns all possible values of a_1 given a_7 = 2 -/
def possible_a1 : List ℕ :=
  (List.range 1000).filter (λ x => a 6 x = 2)

/-- Calculates the maximum sum of the first 7 terms -/
def max_sum : ℕ :=
  (possible_a1.map (λ x => List.sum (List.map (a · x) (List.range 7)))).maximum?
    |>.getD 0

/-- Calculates the sum of all possible values of a_1 -/
def sum_possible_a1 : ℕ :=
  List.sum possible_a1

/-- The main theorem to be proved -/
theorem difference_is_64 : max_sum - sum_possible_a1 = 64 := by
  sorry

end difference_is_64_l3231_323122


namespace parallelogram_contains_two_points_from_L_l3231_323125

/-- The set L of points in the coordinate plane -/
def L : Set (ℤ × ℤ) := {p | ∃ x y : ℤ, p = (41*x + 2*y, 59*x + 15*y)}

/-- A parallelogram centered at the origin -/
structure Parallelogram :=
  (a b c d : ℝ × ℝ)
  (center_origin : a + c = (0, 0) ∧ b + d = (0, 0))
  (area : ℝ)

/-- The theorem statement -/
theorem parallelogram_contains_two_points_from_L :
  ∀ P : Parallelogram, P.area = 1990 →
  ∃ p q : ℤ × ℤ, p ∈ L ∧ q ∈ L ∧ p ≠ q ∧ 
  (↑p.1, ↑p.2) ∈ {x : ℝ × ℝ | ∃ t s : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧ x = t • P.a + s • P.b} ∧
  (↑q.1, ↑q.2) ∈ {x : ℝ × ℝ | ∃ t s : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 0 ≤ s ∧ s ≤ 1 ∧ x = t • P.a + s • P.b} :=
sorry

end parallelogram_contains_two_points_from_L_l3231_323125


namespace tangent_curve_a_value_l3231_323133

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a)

noncomputable def tangent_line (x : ℝ) : ℝ := x + 2

theorem tangent_curve_a_value (a : ℝ) :
  (∃ x₀ : ℝ, curve a x₀ = tangent_line x₀ ∧
    (∀ x : ℝ, x ≠ x₀ → curve a x ≠ tangent_line x) ∧
    (deriv (curve a) x₀ = deriv tangent_line x₀)) →
  a = 3 := by sorry

end tangent_curve_a_value_l3231_323133


namespace water_added_calculation_l3231_323118

def initial_volume : ℝ := 340
def initial_water_percentage : ℝ := 0.80
def initial_kola_percentage : ℝ := 0.06
def added_sugar : ℝ := 3.2
def added_kola : ℝ := 6.8
def final_sugar_percentage : ℝ := 0.14111111111111112

theorem water_added_calculation (water_added : ℝ) : 
  let initial_sugar_percentage := 1 - initial_water_percentage - initial_kola_percentage
  let initial_sugar := initial_sugar_percentage * initial_volume
  let total_sugar := initial_sugar + added_sugar
  let final_volume := initial_volume + water_added + added_sugar + added_kola
  final_sugar_percentage * final_volume = total_sugar →
  water_added = 10 := by sorry

end water_added_calculation_l3231_323118


namespace youngest_sibling_age_l3231_323194

/-- Given 4 siblings where the ages of the older siblings are 3, 6, and 7 years more than 
    the youngest, and the average age of all siblings is 30, 
    the age of the youngest sibling is 26. -/
theorem youngest_sibling_age (y : ℕ) : 
  (y + (y + 3) + (y + 6) + (y + 7)) / 4 = 30 → y = 26 := by
  sorry

end youngest_sibling_age_l3231_323194


namespace games_given_to_friend_l3231_323131

theorem games_given_to_friend (initial_games : ℕ) (remaining_games : ℕ) 
  (h1 : initial_games = 9) 
  (h2 : remaining_games = 5) : 
  initial_games - remaining_games = 4 := by
  sorry

end games_given_to_friend_l3231_323131


namespace tens_digit_of_2035_pow_2037_minus_2039_l3231_323171

theorem tens_digit_of_2035_pow_2037_minus_2039 : ∃ n : ℕ, n < 10 ∧ n * 10 + 3 = (2035^2037 - 2039) % 100 := by
  sorry

end tens_digit_of_2035_pow_2037_minus_2039_l3231_323171


namespace sine_transformation_l3231_323158

theorem sine_transformation (x : ℝ) :
  let f (t : ℝ) := Real.sin t
  let g (t : ℝ) := f (t - (2/3) * Real.pi)
  let h (t : ℝ) := g (t / 3)
  h x = Real.sin (3 * x - (2/3) * Real.pi) := by
sorry

end sine_transformation_l3231_323158


namespace quadrant_I_solution_l3231_323112

theorem quadrant_I_solution (c : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x - 2*y = 5 ∧ c*x + 3*y = 2) ↔ -3/2 < c ∧ c < 2/5 :=
sorry

end quadrant_I_solution_l3231_323112


namespace irrational_numbers_count_l3231_323109

theorem irrational_numbers_count : ∃! (s : Finset ℝ), 
  (∀ x ∈ s, Irrational x ∧ ∃ k : ℤ, (x + 1) / (x^2 - 3*x + 3) = k) ∧ 
  Finset.card s = 2 := by
sorry

end irrational_numbers_count_l3231_323109


namespace binary_number_divisibility_l3231_323130

theorem binary_number_divisibility : ∃ k : ℕ, 2^139 + 2^105 + 2^15 + 2^13 = 136 * k := by
  sorry

end binary_number_divisibility_l3231_323130


namespace parabola_intersection_circle_l3231_323179

/-- The equation of the circle passing through the intersections of the parabola
    y = x^2 - 2x - 3 with the coordinate axes -/
theorem parabola_intersection_circle : 
  ∃ (x y : ℝ), (y = x^2 - 2*x - 3) → 
  ((x = 0 ∧ y = -3) ∨ (y = 0 ∧ (x = -1 ∨ x = 3))) →
  (x - 1)^2 + (y + 1)^2 = 5 :=
sorry

end parabola_intersection_circle_l3231_323179


namespace translated_quadratic_vertex_l3231_323153

/-- The vertex of a quadratic function translated to the right by 3 units -/
theorem translated_quadratic_vertex (f g : ℝ → ℝ) (h : ℝ) :
  (∀ x, f x = 2 * (x - 1)^2 - 3) →
  (∀ x, g x = 2 * (x - 4)^2 - 3) →
  (∀ x, g x = f (x - 3)) →
  h = 4 →
  (∀ x, g x ≥ g h) →
  g h = -3 :=
by sorry

end translated_quadratic_vertex_l3231_323153


namespace smallest_product_l3231_323140

def digits : List ℕ := [5, 6, 7, 8]

def valid_arrangement (a b c d : ℕ) : Prop :=
  a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ d ∈ digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

def product (a b c d : ℕ) : ℕ := (10 * a + b) * (10 * c + d)

theorem smallest_product :
  ∀ a b c d : ℕ, valid_arrangement a b c d →
    product a b c d ≥ 4368 :=
sorry

end smallest_product_l3231_323140


namespace sqrt_equation_solution_l3231_323147

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (9 - 2 * x) = 5 → x = -8 := by sorry

end sqrt_equation_solution_l3231_323147


namespace surface_area_of_T_l3231_323176

-- Define the cube
def cube_edge_length : ℝ := 10

-- Define points M, N, O
def point_M : ℝ × ℝ × ℝ := (3, 0, 0)
def point_N : ℝ × ℝ × ℝ := (0, 3, 0)
def point_O : ℝ × ℝ × ℝ := (0, 0, 3)

-- Define the distance from A to M, N, O
def distance_AM : ℝ := 3
def distance_AN : ℝ := 3
def distance_AO : ℝ := 3

-- Function to calculate the area of a triangle given three points in 3D space
def triangle_area (p1 p2 p3 : ℝ × ℝ × ℝ) : ℝ := sorry

-- Function to calculate the surface area of a cube given its edge length
def cube_surface_area (edge_length : ℝ) : ℝ := sorry

-- Theorem: The surface area of solid T is 600 - 27√2
theorem surface_area_of_T :
  let triangle_face_area := triangle_area point_M point_N point_O
  let cube_area := cube_surface_area cube_edge_length
  let removed_area := 3 * triangle_face_area
  cube_area - removed_area = 600 - 27 * Real.sqrt 2 := by sorry

end surface_area_of_T_l3231_323176


namespace intersections_of_related_functions_l3231_323126

/-- Given a quadratic function that intersects (0, 2) and (1, 1), 
    prove that the related linear function intersects the axes at (1/2, 0) and (0, -1) -/
theorem intersections_of_related_functions 
  (a c : ℝ) 
  (h1 : c = 2) 
  (h2 : a + c = 1) : 
  let f (x : ℝ) := c * x + a
  (f (1/2) = 0 ∧ f 0 = -1) := by
sorry

end intersections_of_related_functions_l3231_323126


namespace common_term_implies_fermat_number_l3231_323196

/-- Definition of the second-order arithmetic sequence -/
def a (n : ℕ) (k : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

/-- Definition of Fermat numbers -/
def fermat (m : ℕ) : ℕ :=
  2^(2^m) + 1

/-- Theorem stating that if k satisfies the condition, it must be a Fermat number -/
theorem common_term_implies_fermat_number (k : ℕ) (h1 : k > 2) :
  (∃ n m : ℕ, a n k = fermat m) → (∃ m : ℕ, k = fermat m) :=
sorry

end common_term_implies_fermat_number_l3231_323196


namespace transform_OAB_l3231_323162

/-- Transformation from xy-plane to uv-plane -/
def transform (x y : ℝ) : ℝ × ℝ := (x^2 - y^2, x * y)

/-- Triangle OAB in xy-plane -/
def triangle_OAB : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ x ∧ p = (x, y)}

/-- Image of triangle OAB in uv-plane -/
def image_OAB : Set (ℝ × ℝ) :=
  {q | ∃ p ∈ triangle_OAB, q = transform p.1 p.2}

theorem transform_OAB :
  (0, 0) ∈ image_OAB ∧ (1, 0) ∈ image_OAB ∧ (0, 1) ∈ image_OAB :=
sorry

end transform_OAB_l3231_323162


namespace ping_pong_balls_count_l3231_323160

/-- The number of ping-pong balls in the gym storage -/
def ping_pong_balls : ℕ :=
  let total_balls : ℕ := 240
  let baseball_boxes : ℕ := 35
  let baseballs_per_box : ℕ := 4
  let tennis_ball_boxes : ℕ := 6
  let tennis_balls_per_box : ℕ := 3
  let baseballs : ℕ := baseball_boxes * baseballs_per_box
  let tennis_balls : ℕ := tennis_ball_boxes * tennis_balls_per_box
  total_balls - (baseballs + tennis_balls)

theorem ping_pong_balls_count : ping_pong_balls = 82 := by
  sorry

end ping_pong_balls_count_l3231_323160


namespace set_inclusion_implies_a_range_l3231_323195

theorem set_inclusion_implies_a_range (a : ℝ) : 
  let A := {x : ℝ | -2 ≤ x ∧ x ≤ a}
  let B := {y : ℝ | ∃ x ∈ A, y = 2*x + 3}
  let C := {z : ℝ | ∃ x ∈ A, z = x^2}
  C ⊆ B → (1/2 : ℝ) ≤ a ∧ a ≤ 3 := by
sorry

end set_inclusion_implies_a_range_l3231_323195


namespace system_solution_l3231_323119

theorem system_solution (x y k : ℝ) : 
  x - y = k + 2 →
  x + 3*y = k →
  x + y = 2 →
  k = 1 := by
sorry

end system_solution_l3231_323119
