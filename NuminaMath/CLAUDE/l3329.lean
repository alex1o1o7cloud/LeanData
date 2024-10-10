import Mathlib

namespace volume_ratio_of_cubes_l3329_332912

-- Define the edge lengths in inches
def small_cube_edge : ℚ := 4
def large_cube_edge : ℚ := 24  -- 2 feet = 24 inches

-- Define the volumes of the cubes
def small_cube_volume : ℚ := small_cube_edge ^ 3
def large_cube_volume : ℚ := large_cube_edge ^ 3

-- Theorem statement
theorem volume_ratio_of_cubes : 
  small_cube_volume / large_cube_volume = 1 / 216 := by
  sorry

end volume_ratio_of_cubes_l3329_332912


namespace problem_solution_l3329_332954

theorem problem_solution (p_xavier p_yvonne p_zelda : ℚ)
  (h_xavier : p_xavier = 1/3)
  (h_yvonne : p_yvonne = 1/2)
  (h_zelda : p_zelda = 5/8) :
  p_xavier * p_yvonne * (1 - p_zelda) = 1/16 := by
  sorry

end problem_solution_l3329_332954


namespace fibonacci_divisibility_l3329_332955

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Divisibility relation -/
def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem fibonacci_divisibility (m n : ℕ) (h : m > 2) :
  divides (fib m) (fib n) ↔ divides m n := by
  sorry

end fibonacci_divisibility_l3329_332955


namespace ratio_calculation_l3329_332992

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) : 
  (2 * A + 3 * B) / (A + 5 * C) = 3 / 7 := by
  sorry

end ratio_calculation_l3329_332992


namespace vasya_numbers_l3329_332902

theorem vasya_numbers : ∃! (x y : ℝ), x + y = x * y ∧ x + y = x / y ∧ x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
  sorry

end vasya_numbers_l3329_332902


namespace arithmetic_sequence_problem_l3329_332956

/-- An arithmetic sequence with common difference d -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def geometric_sequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  a 2 = 4 →
  geometric_sequence (1 + a 3) (a 6) (4 + a 10) →
  d = 3 := by
sorry

end arithmetic_sequence_problem_l3329_332956


namespace other_root_of_quadratic_l3329_332979

theorem other_root_of_quadratic (k : ℝ) : 
  (1 : ℝ) ^ 2 + k * 1 - 2 = 0 → 
  ∃ (x : ℝ), x ≠ 1 ∧ x ^ 2 + k * x - 2 = 0 ∧ x = -2 :=
by sorry

end other_root_of_quadratic_l3329_332979


namespace determine_coins_in_38_bags_l3329_332908

/-- Represents a bag of coins -/
structure Bag where
  coins : ℕ
  inv : coins ≥ 1000

/-- Represents the state of all bags -/
def BagState := Fin 40 → Bag

/-- An operation that checks two bags and potentially removes a coin from one of them -/
def CheckOperation (state : BagState) (i j : Fin 40) : BagState := sorry

/-- Predicate to check if we know the exact number of coins in a bag -/
def KnowExactCoins (state : BagState) (i : Fin 40) : Prop := sorry

/-- The main theorem stating that it's possible to determine the number of coins in 38 out of 40 bags -/
theorem determine_coins_in_38_bags :
  ∃ (operations : List (Fin 40 × Fin 40)),
    operations.length ≤ 100 ∧
    ∀ (initial_state : BagState),
      let final_state := operations.foldl (fun state (i, j) => CheckOperation state i j) initial_state
      (∃ (unknown1 unknown2 : Fin 40), ∀ (i : Fin 40),
        i ≠ unknown1 → i ≠ unknown2 → KnowExactCoins final_state i) :=
sorry

end determine_coins_in_38_bags_l3329_332908


namespace jimmy_stairs_l3329_332959

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Jimmy's stair climbing problem -/
theorem jimmy_stairs : arithmetic_sum 30 10 8 = 520 := by
  sorry

end jimmy_stairs_l3329_332959


namespace f_2018_l3329_332963

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom periodic : ∀ x : ℝ, f (x + 4) = -f x
axiom symmetric : ∀ x : ℝ, f (1 - (x - 1)) = f (x - 1)
axiom f_2 : f 2 = 2

-- The theorem to prove
theorem f_2018 : f 2018 = 2 := by sorry

end f_2018_l3329_332963


namespace fourth_largest_common_divisor_l3329_332985

def is_divisor (d n : ℕ) : Prop := n % d = 0

def common_divisors (a b : ℕ) : Set ℕ :=
  {d : ℕ | is_divisor d a ∧ is_divisor d b}

theorem fourth_largest_common_divisor :
  let cd := common_divisors 72 120
  ∃ (l : List ℕ), (∀ x ∈ cd, x ∈ l) ∧
                  (∀ x ∈ l, x ∈ cd) ∧
                  l.Sorted (· > ·) ∧
                  l.get? 3 = some 6 :=
sorry

end fourth_largest_common_divisor_l3329_332985


namespace z_coordinate_is_zero_l3329_332940

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- A point on a line with a specific x-coordinate -/
def point_on_line (l : Line3D) (x : ℝ) : ℝ × ℝ × ℝ := sorry

theorem z_coordinate_is_zero : 
  let l : Line3D := { point1 := (1, 3, 2), point2 := (4, 2, -1) }
  let p := point_on_line l 3
  p.2.2 = 0 := by sorry

end z_coordinate_is_zero_l3329_332940


namespace circle_diameter_endpoint_l3329_332990

/-- Given a circle with center (5, -4) and one endpoint of a diameter at (0, -9),
    the other endpoint of the diameter is at (10, 1). -/
theorem circle_diameter_endpoint :
  ∀ (P : ℝ × ℝ) (A : ℝ × ℝ) (Q : ℝ × ℝ),
  P = (5, -4) →  -- Center of the circle
  A = (0, -9) →  -- One endpoint of the diameter
  (P.1 - A.1)^2 + (P.2 - A.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2 →  -- A and Q are equidistant from P
  P.1 - A.1 = Q.1 - P.1 ∧ P.2 - A.2 = Q.2 - P.2 →  -- A, P, and Q are collinear
  Q = (10, 1) :=
by sorry

end circle_diameter_endpoint_l3329_332990


namespace instructors_next_meeting_l3329_332941

theorem instructors_next_meeting (f g h i j : ℕ) 
  (hf : f = 5) (hg : g = 3) (hh : h = 9) (hi : i = 2) (hj : j = 8) :
  Nat.lcm f (Nat.lcm g (Nat.lcm h (Nat.lcm i j))) = 360 :=
by sorry

end instructors_next_meeting_l3329_332941


namespace arithmetic_sequence_sum_l3329_332969

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₂ = 3 and a₅ + a₇ = 10, prove that a₁ + a₁₀ = 9.5 -/
theorem arithmetic_sequence_sum (a : ℕ → ℚ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a2 : a 2 = 3) 
  (h_sum : a 5 + a 7 = 10) : 
  a 1 + a 10 = 9.5 := by
  sorry

end arithmetic_sequence_sum_l3329_332969


namespace min_value_expression_l3329_332978

theorem min_value_expression (x y z : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) (h4 : z ≤ 5) :
  (x - 2)^2 + (y/x - 1)^2 + (z/y - 1)^2 + (5/z - 1)^2 ≥ 4 * (5^(1/4) - 1)^2 := by
  sorry

end min_value_expression_l3329_332978


namespace arithmetic_and_geometric_is_nonzero_constant_l3329_332947

/-- A sequence that is both arithmetic and geometric is non-zero constant -/
theorem arithmetic_and_geometric_is_nonzero_constant (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) →  -- arithmetic sequence condition
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence condition
  (∃ c : ℝ, c ≠ 0 ∧ ∀ n : ℕ, a n = c) :=      -- non-zero constant sequence
by sorry

end arithmetic_and_geometric_is_nonzero_constant_l3329_332947


namespace root_sum_power_five_l3329_332914

theorem root_sum_power_five (ζ₁ ζ₂ ζ₃ : ℂ) : 
  (ζ₁^3 - ζ₁^2 - 2*ζ₁ - 2 = 0) →
  (ζ₂^3 - ζ₂^2 - 2*ζ₂ - 2 = 0) →
  (ζ₃^3 - ζ₃^2 - 2*ζ₃ - 2 = 0) →
  (ζ₁ + ζ₂ + ζ₃ = 1) →
  (ζ₁^2 + ζ₂^2 + ζ₃^2 = 5) →
  (ζ₁^3 + ζ₂^3 + ζ₃^3 = 11) →
  (ζ₁^5 + ζ₂^5 + ζ₃^5 = 55) :=
by sorry

end root_sum_power_five_l3329_332914


namespace inverse_composition_equals_six_l3329_332920

-- Define the function f
def f : ℕ → ℕ
| 1 => 4
| 2 => 6
| 3 => 2
| 4 => 1
| 5 => 5
| 6 => 3
| _ => 0  -- Default case for other inputs

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 1 => 4
| 2 => 3
| 3 => 6
| 4 => 1
| 5 => 5
| 6 => 2
| _ => 0  -- Default case for other inputs

-- State the theorem
theorem inverse_composition_equals_six :
  f_inv (f_inv (f_inv 6)) = 6 := by sorry

end inverse_composition_equals_six_l3329_332920


namespace min_transportation_cost_l3329_332949

/-- Transportation problem between two cities and two towns -/
structure TransportationProblem where
  cityA_goods : ℕ
  cityB_goods : ℕ
  townA_needs : ℕ
  townB_needs : ℕ
  costA_to_A : ℕ
  costA_to_B : ℕ
  costB_to_A : ℕ
  costB_to_B : ℕ

/-- Define the specific problem instance -/
def problem : TransportationProblem := {
  cityA_goods := 120
  cityB_goods := 130
  townA_needs := 140
  townB_needs := 110
  costA_to_A := 300
  costA_to_B := 150
  costB_to_A := 200
  costB_to_B := 100
}

/-- Total transportation cost function -/
def total_cost (p : TransportationProblem) (x : ℕ) : ℕ :=
  p.costA_to_A * x + p.costA_to_B * (p.cityA_goods - x) +
  p.costB_to_A * (p.townA_needs - x) + p.costB_to_B * (p.townB_needs - p.cityA_goods + x)

/-- Theorem: The minimum total transportation cost is 45500 yuan -/
theorem min_transportation_cost :
  ∃ x, x ≥ 10 ∧ x ≤ 120 ∧
  (∀ y, y ≥ 10 → y ≤ 120 → total_cost problem x ≤ total_cost problem y) ∧
  total_cost problem x = 45500 :=
sorry

end min_transportation_cost_l3329_332949


namespace romeo_profit_l3329_332932

/-- Calculates the profit for selling chocolate bars -/
def chocolate_profit (num_bars : ℕ) (cost_per_bar : ℕ) (selling_price : ℕ) (packaging_cost : ℕ) : ℕ :=
  selling_price - (num_bars * cost_per_bar + num_bars * packaging_cost)

/-- Theorem: Romeo's profit is $55 -/
theorem romeo_profit : 
  chocolate_profit 5 5 90 2 = 55 := by
  sorry

end romeo_profit_l3329_332932


namespace system_solution_pairs_l3329_332927

theorem system_solution_pairs :
  ∀ x y : ℝ, x > 0 ∧ y > 0 →
  (x - 3 * Real.sqrt (x * y) - 2 * Real.sqrt (x / y) = 0 ∧
   x^2 * y^2 + x^4 = 82) →
  ((x = 3 ∧ y = 1/3) ∨ (x = Real.rpow 66 (1/4) ∧ y = 4 / Real.rpow 66 (1/4))) := by
sorry

end system_solution_pairs_l3329_332927


namespace complement_union_M_N_l3329_332907

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def M : Set Nat := {1, 4}
def N : Set Nat := {2, 3}

theorem complement_union_M_N :
  (M ∪ N)ᶜ = {5, 6} :=
by sorry

end complement_union_M_N_l3329_332907


namespace origami_stars_per_bottle_l3329_332934

/-- Represents the problem of determining the number of origami stars per bottle -/
theorem origami_stars_per_bottle
  (total_bottles : ℕ)
  (total_stars : ℕ)
  (h1 : total_bottles = 5)
  (h2 : total_stars = 75) :
  total_stars / total_bottles = 15 := by
  sorry

end origami_stars_per_bottle_l3329_332934


namespace abs_diff_gt_cube_root_product_l3329_332930

theorem abs_diff_gt_cube_root_product (a b : ℤ) : 
  a ≠ b → (a^2 + a*b + b^2) ∣ (a*b*(a + b)) → |a - b| > (a*b : ℝ)^(1/3) := by
  sorry

end abs_diff_gt_cube_root_product_l3329_332930


namespace remaining_garlic_cloves_l3329_332922

-- Define the initial number of garlic cloves
def initial_cloves : ℕ := 93

-- Define the number of cloves used
def used_cloves : ℕ := 86

-- Theorem stating that the remaining cloves is 7
theorem remaining_garlic_cloves : initial_cloves - used_cloves = 7 := by
  sorry

end remaining_garlic_cloves_l3329_332922


namespace union_of_A_and_B_l3329_332943

-- Define set A
def A : Set ℝ := {x | 0 < 3 - x ∧ 3 - x ≤ 2}

-- Define set B
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 ≤ x ∧ x < 3} := by sorry

end union_of_A_and_B_l3329_332943


namespace inequality_solution_set_l3329_332964

theorem inequality_solution_set :
  ∀ x : ℝ, (((2*x - 1) / (x + 1) ≤ 1 ∧ x + 1 ≠ 0) ↔ x ∈ Set.Ioo (-1 : ℝ) 2) :=
by sorry

end inequality_solution_set_l3329_332964


namespace hyperbola_inequality_l3329_332962

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 = 1

-- Define the asymptotes
def asymptote (x y : ℝ) : Prop := y = x / 2 ∨ y = -x / 2

-- Define points A and B
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (2, -1)

-- Define any point P on the hyperbola
def P (a b : ℝ) : ℝ × ℝ := (2*a + 2*b, a - b)

theorem hyperbola_inequality (a b : ℝ) :
  hyperbola (P a b).1 (P a b).2 →
  |a + b| ≥ 1 := by sorry

end hyperbola_inequality_l3329_332962


namespace f_monotonic_increasing_interval_l3329_332935

-- Define the function f(x) = -x^2 + 1
def f (x : ℝ) : ℝ := -x^2 + 1

-- State the theorem
theorem f_monotonic_increasing_interval :
  ∀ x y, x < y ∧ y ≤ 0 → f x < f y :=
by
  sorry

end f_monotonic_increasing_interval_l3329_332935


namespace latest_start_time_l3329_332981

def movie_start_time : ℕ := 20 -- 8 pm in 24-hour format
def home_time : ℕ := 17 -- 5 pm in 24-hour format
def dinner_duration : ℕ := 45
def homework_duration : ℕ := 30
def clean_room_duration : ℕ := 30
def trash_duration : ℕ := 5
def dishwasher_duration : ℕ := 10

def total_task_duration : ℕ := 
  dinner_duration + homework_duration + clean_room_duration + trash_duration + dishwasher_duration

theorem latest_start_time (start_time : ℕ) :
  start_time + total_task_duration / 60 = movie_start_time →
  start_time ≥ home_time →
  start_time = 18 := by sorry

end latest_start_time_l3329_332981


namespace flag_distribution_theorem_l3329_332942

structure FlagDistribution where
  total_flags : ℕ
  blue_percentage : ℚ
  red_percentage : ℚ
  green_percentage : ℚ

def children_with_both_blue_and_red (fd : FlagDistribution) : ℚ :=
  fd.blue_percentage + fd.red_percentage - 1

theorem flag_distribution_theorem (fd : FlagDistribution) 
  (h1 : Even fd.total_flags)
  (h2 : fd.blue_percentage = 1/2)
  (h3 : fd.red_percentage = 3/5)
  (h4 : fd.green_percentage = 2/5)
  (h5 : fd.blue_percentage + fd.red_percentage + fd.green_percentage = 3/2) :
  children_with_both_blue_and_red fd = 1/10 := by
  sorry

end flag_distribution_theorem_l3329_332942


namespace white_area_is_42_l3329_332961

/-- The area of a rectangle -/
def rectangle_area (width : ℕ) (height : ℕ) : ℕ := width * height

/-- The area of the letter C -/
def c_area : ℕ := 2 * (6 * 1) + 1 * 4

/-- The area of the letter O -/
def o_area : ℕ := 2 * (6 * 1) + 2 * 4

/-- The area of the letter L -/
def l_area : ℕ := 1 * (6 * 1) + 1 * 4

/-- The total black area of the word COOL -/
def cool_area : ℕ := c_area + 2 * o_area + l_area

/-- The width of the sign -/
def sign_width : ℕ := 18

/-- The height of the sign -/
def sign_height : ℕ := 6

theorem white_area_is_42 : 
  rectangle_area sign_width sign_height - cool_area = 42 := by
  sorry

end white_area_is_42_l3329_332961


namespace f_monotonicity_and_extremum_l3329_332924

noncomputable def f (x : ℝ) := x * Real.exp (-x)

theorem f_monotonicity_and_extremum :
  (∀ x y : ℝ, x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y : ℝ, 1 < x ∧ x < y → f y < f x) ∧
  (∀ x : ℝ, x ≠ 1 → f x < f 1) ∧
  f 1 = Real.exp (-1) := by sorry

end f_monotonicity_and_extremum_l3329_332924


namespace sum_of_squares_coefficients_l3329_332929

/-- The sum of squares of coefficients in the simplified form of 6(x³-2x²+x-3)-5(x⁴-4x²+3x+2) is 990 -/
theorem sum_of_squares_coefficients : 
  let expression := fun x : ℝ => 6 * (x^3 - 2*x^2 + x - 3) - 5 * (x^4 - 4*x^2 + 3*x + 2)
  let coefficients := [-5, 6, 8, -9, -28]
  (coefficients.map (fun c => c^2)).sum = 990 := by
sorry

end sum_of_squares_coefficients_l3329_332929


namespace nina_homework_calculation_l3329_332948

/-- Nina's homework calculation -/
theorem nina_homework_calculation
  (ruby_math : ℕ) (ruby_reading : ℕ)
  (nina_math_multiplier : ℕ) (nina_reading_multiplier : ℕ)
  (h_ruby_math : ruby_math = 6)
  (h_ruby_reading : ruby_reading = 2)
  (h_nina_math : nina_math_multiplier = 4)
  (h_nina_reading : nina_reading_multiplier = 8) :
  (ruby_math * nina_math_multiplier + ruby_math) +
  (ruby_reading * nina_reading_multiplier + ruby_reading) = 48 := by
  sorry

#check nina_homework_calculation

end nina_homework_calculation_l3329_332948


namespace sum_of_roots_l3329_332916

/-- The function f(x) = x^3 - 6x^2 + 17x - 5 -/
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 17*x - 5

/-- Theorem: If f(a) = 3 and f(b) = 23, then a + b = 4 -/
theorem sum_of_roots (a b : ℝ) (ha : f a = 3) (hb : f b = 23) : a + b = 4 := by
  sorry

end sum_of_roots_l3329_332916


namespace greatest_n_no_substring_divisible_by_9_l3329_332905

-- Define a function to check if a number is divisible by 9
def divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

-- Define a function to get all integer substrings of a number
def integer_substrings (n : ℕ) : List ℕ := sorry

-- Define the property that no integer substring is divisible by 9
def no_substring_divisible_by_9 (n : ℕ) : Prop :=
  ∀ m ∈ integer_substrings n, ¬(divisible_by_9 m)

-- State the theorem
theorem greatest_n_no_substring_divisible_by_9 :
  (∀ k > 88888888, ¬(no_substring_divisible_by_9 k)) ∧
  (no_substring_divisible_by_9 88888888) :=
sorry

end greatest_n_no_substring_divisible_by_9_l3329_332905


namespace complex_fraction_simplification_l3329_332918

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (i : ℂ) / (1 - i) = -1/2 + (1/2 : ℂ) * i :=
by sorry

end complex_fraction_simplification_l3329_332918


namespace sine_function_inequality_l3329_332968

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x) * Real.cos φ + Real.cos (2 * x) * Real.sin φ

theorem sine_function_inequality 
  (φ : ℝ) 
  (h : ∀ x : ℝ, f x φ ≤ f (2 * Real.pi / 9) φ) : 
  f (2 * Real.pi / 3) φ < f (5 * Real.pi / 6) φ ∧ 
  f (5 * Real.pi / 6) φ < f (7 * Real.pi / 6) φ :=
sorry

end sine_function_inequality_l3329_332968


namespace proposition_induction_l3329_332913

theorem proposition_induction (P : ℕ → Prop) :
  (∀ k : ℕ, k > 0 → (P k → P (k + 1))) →
  ¬ P 7 →
  ¬ P 6 := by
  sorry

end proposition_induction_l3329_332913


namespace mary_bought_48_cards_l3329_332938

/-- Calculates the number of baseball cards Mary bought -/
def cards_mary_bought (initial_cards : ℕ) (torn_cards : ℕ) (cards_from_fred : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_cards - torn_cards + cards_from_fred)

/-- Proves that Mary bought 48 baseball cards -/
theorem mary_bought_48_cards : cards_mary_bought 18 8 26 84 = 48 := by
  sorry

#eval cards_mary_bought 18 8 26 84

end mary_bought_48_cards_l3329_332938


namespace exam_score_l3329_332960

theorem exam_score (total_questions : ℕ) (correct_answers : ℕ) 
  (marks_per_correct : ℕ) (marks_lost_per_wrong : ℕ) : 
  total_questions = 80 → 
  correct_answers = 40 → 
  marks_per_correct = 4 → 
  marks_lost_per_wrong = 1 → 
  (correct_answers * marks_per_correct) - 
    ((total_questions - correct_answers) * marks_lost_per_wrong) = 120 := by
  sorry

#check exam_score

end exam_score_l3329_332960


namespace f_neg_one_eq_neg_two_l3329_332946

/- Define an odd function f -/
def f (x : ℝ) : ℝ := sorry

/- State the properties of f -/
axiom f_odd : ∀ x, f (-x) = -f x
axiom f_positive : ∀ x > 0, f x = x^2 + 1/x

/- Theorem to prove -/
theorem f_neg_one_eq_neg_two : f (-1) = -2 := by sorry

end f_neg_one_eq_neg_two_l3329_332946


namespace cubic_sum_equals_27_l3329_332917

theorem cubic_sum_equals_27 (a b : ℝ) (h : a + b = 3) : a^3 + b^3 + 9*a*b = 27 := by
  sorry

end cubic_sum_equals_27_l3329_332917


namespace square_product_eq_sum_squares_solution_l3329_332984

theorem square_product_eq_sum_squares_solution (a b : ℤ) :
  a^2 * b^2 = a^2 + b^2 → a = 0 ∧ b = 0 := by
  sorry

end square_product_eq_sum_squares_solution_l3329_332984


namespace stating_rectangular_box_area_diagonal_product_l3329_332983

/-- Represents a rectangular box with dimensions a, b, and c -/
structure RectangularBox (a b c : ℝ) where
  bottom_area : ℝ := a * b
  side_area : ℝ := b * c
  front_area : ℝ := c * a
  diagonal_squared : ℝ := a^2 + b^2 + c^2

/-- 
Theorem stating that for a rectangular box, the product of its face areas 
multiplied by the square of its diagonal equals a²b²c² · (a² + b² + c²)
-/
theorem rectangular_box_area_diagonal_product 
  (a b c : ℝ) (box : RectangularBox a b c) : 
  box.bottom_area * box.side_area * box.front_area * box.diagonal_squared = 
  a^2 * b^2 * c^2 * (a^2 + b^2 + c^2) := by
  sorry

end stating_rectangular_box_area_diagonal_product_l3329_332983


namespace cubic_root_sum_cubes_l3329_332901

theorem cubic_root_sum_cubes (a b c : ℝ) : 
  (a^3 - 2*a^2 - a + 3 = 0) →
  (b^3 - 2*b^2 - b + 3 = 0) →
  (c^3 - 2*c^2 - c + 3 = 0) →
  a^3 + b^3 + c^3 = 5 := by
  sorry

end cubic_root_sum_cubes_l3329_332901


namespace kishore_rent_expenditure_l3329_332936

def monthly_salary (savings : ℕ) : ℕ := savings * 10

def total_expenses (salary : ℕ) : ℕ := (salary * 9) / 10

def other_expenses : ℕ := 1500 + 4500 + 2500 + 2000 + 3940

def rent_expenditure (total_exp other_exp : ℕ) : ℕ := total_exp - other_exp

theorem kishore_rent_expenditure (savings : ℕ) (h : savings = 2160) :
  rent_expenditure (total_expenses (monthly_salary savings)) other_expenses = 5000 := by
  sorry

end kishore_rent_expenditure_l3329_332936


namespace ellipse_intersection_right_triangle_l3329_332900

/-- Defines an ellipse with equation x²/4 + y²/2 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- Defines a line with equation y = x + m -/
def line (x y m : ℝ) : Prop := y = x + m

/-- Defines the intersection points of the ellipse and the line -/
def intersection (x₁ y₁ x₂ y₂ m : ℝ) : Prop :=
  ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ line x₁ y₁ m ∧ line x₂ y₂ m

/-- Defines a point Q on the y-axis -/
def point_on_y_axis (y : ℝ) : Prop := true

/-- Defines a right triangle formed by points Q, A, and B -/
def right_triangle (x₁ y₁ x₂ y₂ y : ℝ) : Prop :=
  (x₂ - x₁)*(0 - x₁) + (y₂ - y₁)*(y - y₁) = 0

/-- Main theorem: If there exists a point Q on the y-axis such that △QAB is a right triangle,
    where A and B are the intersection points of the ellipse x²/4 + y²/2 = 1 and the line y = x + m,
    then m = ±(3√10)/5 -/
theorem ellipse_intersection_right_triangle (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ y, intersection x₁ y₁ x₂ y₂ m ∧ point_on_y_axis y ∧ right_triangle x₁ y₁ x₂ y₂ y) →
  m = 3*Real.sqrt 10/5 ∨ m = -3*Real.sqrt 10/5 := by
  sorry

end ellipse_intersection_right_triangle_l3329_332900


namespace point_in_inequality_region_implies_B_range_l3329_332926

/-- Given a point A (1, 2) inside the plane region corresponding to the linear inequality 2x - By + 3 ≥ 0, 
    prove that the range of the real number B is B ≤ 2.5. -/
theorem point_in_inequality_region_implies_B_range (B : ℝ) : 
  (2 * 1 - B * 2 + 3 ≥ 0) → B ≤ 2.5 := by
  sorry

end point_in_inequality_region_implies_B_range_l3329_332926


namespace average_speed_proof_l3329_332989

/-- Prove that the average speed of a trip with given conditions is 40 miles per hour -/
theorem average_speed_proof (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

end average_speed_proof_l3329_332989


namespace min_n_for_sqrt_12n_integer_l3329_332991

theorem min_n_for_sqrt_12n_integer (n : ℕ) (h1 : n > 0) (h2 : ∃ k : ℕ, k > 0 ∧ k^2 = 12*n) :
  ∀ m : ℕ, m > 0 → (∃ j : ℕ, j > 0 ∧ j^2 = 12*m) → m ≥ 3 :=
by sorry

end min_n_for_sqrt_12n_integer_l3329_332991


namespace unique_coin_combination_l3329_332910

/-- Represents the number of coins of each denomination -/
structure CoinCounts where
  five : ℕ
  ten : ℕ
  twentyFive : ℕ

/-- Calculates the number of different values obtainable from a given set of coins -/
def differentValues (coins : CoinCounts) : ℕ :=
  14 + coins.ten + 4 * coins.twentyFive

/-- The main theorem -/
theorem unique_coin_combination :
  ∀ (coins : CoinCounts),
    coins.five + coins.ten + coins.twentyFive = 15 →
    differentValues coins = 21 →
    coins.twentyFive = 1 := by
  sorry

#check unique_coin_combination

end unique_coin_combination_l3329_332910


namespace total_amount_formula_total_amount_after_five_months_l3329_332945

/-- Savings account with monthly interest -/
structure SavingsAccount where
  initialDeposit : ℝ
  monthlyInterestRate : ℝ

/-- Calculate total amount after x months -/
def totalAmount (account : SavingsAccount) (months : ℝ) : ℝ :=
  account.initialDeposit + account.initialDeposit * account.monthlyInterestRate * months

/-- Theorem: Total amount after x months is 100 + 0.36x -/
theorem total_amount_formula (account : SavingsAccount) (months : ℝ) 
    (h1 : account.initialDeposit = 100)
    (h2 : account.monthlyInterestRate = 0.0036) : 
    totalAmount account months = 100 + 0.36 * months := by
  sorry

/-- Theorem: Total amount after 5 months is 101.8 -/
theorem total_amount_after_five_months (account : SavingsAccount) 
    (h1 : account.initialDeposit = 100)
    (h2 : account.monthlyInterestRate = 0.0036) : 
    totalAmount account 5 = 101.8 := by
  sorry

end total_amount_formula_total_amount_after_five_months_l3329_332945


namespace tims_sleep_schedule_l3329_332970

/-- Tim's sleep schedule and total sleep calculation -/
theorem tims_sleep_schedule (weekday_sleep : ℕ) (weekend_sleep : ℕ) (weekdays : ℕ) (weekend_days : ℕ) :
  weekday_sleep = 6 →
  weekend_sleep = 10 →
  weekdays = 5 →
  weekend_days = 2 →
  weekday_sleep * weekdays + weekend_sleep * weekend_days = 50 := by
  sorry

#check tims_sleep_schedule

end tims_sleep_schedule_l3329_332970


namespace craftsman_earnings_solution_l3329_332944

def craftsman_earnings (hours_worked : ℕ) (wage_A wage_B : ℚ) : Prop :=
  let earnings_A := hours_worked * wage_A
  let earnings_B := hours_worked * wage_B
  wage_A ≠ wage_B ∧
  (hours_worked - 1) * wage_A = 720 ∧
  (hours_worked - 5) * wage_B = 800 ∧
  (hours_worked - 1) * wage_B - (hours_worked - 5) * wage_A = 360 ∧
  earnings_A = 750 ∧
  earnings_B = 1000

theorem craftsman_earnings_solution :
  ∃ (hours_worked : ℕ) (wage_A wage_B : ℚ),
    craftsman_earnings hours_worked wage_A wage_B :=
by
  sorry

end craftsman_earnings_solution_l3329_332944


namespace minkowski_sum_convex_l3329_332906

-- Define a type for points in a 2D space
variable {α : Type*} [AddCommGroup α] [Module ℝ α]

-- Define a convex figure as a set of points
def ConvexFigure (S : Set α) : Prop :=
  ∀ (x y : α), x ∈ S → y ∈ S → ∀ (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
    (1 - t) • x + t • y ∈ S

-- Define Minkowski sum of two sets
def MinkowskiSum (S T : Set α) : Set α :=
  {z | ∃ (x y : α), x ∈ S ∧ y ∈ T ∧ z = x + y}

-- Theorem statement
theorem minkowski_sum_convex
  (Φ₁ Φ₂ : Set α) (h1 : ConvexFigure Φ₁) (h2 : ConvexFigure Φ₂) :
  ConvexFigure (MinkowskiSum Φ₁ Φ₂) :=
sorry

end minkowski_sum_convex_l3329_332906


namespace john_laptop_savings_l3329_332977

def octal_to_decimal (n : ℕ) : ℕ :=
  (n / 1000) * 512 + ((n / 100) % 10) * 64 + ((n / 10) % 10) * 8 + (n % 10)

theorem john_laptop_savings :
  octal_to_decimal 5555 - 1500 = 1425 := by
  sorry

end john_laptop_savings_l3329_332977


namespace population_size_l3329_332931

/-- Given a population with specified birth and death rates, and a net growth rate,
    prove that the initial population size is 3000. -/
theorem population_size (birth_rate death_rate net_growth_rate : ℝ) 
  (h1 : birth_rate = 52)
  (h2 : death_rate = 16)
  (h3 : net_growth_rate = 0.012)
  (h4 : birth_rate - death_rate = net_growth_rate * 100) : 
  (birth_rate - death_rate) / net_growth_rate = 3000 := by
  sorry

end population_size_l3329_332931


namespace complex_fraction_calculation_l3329_332993

theorem complex_fraction_calculation : 
  (2 + 5/8 - 2/3 * (2 + 5/14)) / ((3 + 1/12 + 4.375) / (19 + 8/9)) = 2 + 17/21 := by sorry

end complex_fraction_calculation_l3329_332993


namespace pool_fill_time_ab_l3329_332928

/-- Represents the time it takes for a valve to fill the pool individually -/
structure ValveTime where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents the conditions given in the problem -/
structure PoolFillConditions where
  vt : ValveTime
  all_valves_time : (1 / vt.a + 1 / vt.b + 1 / vt.c) = 1
  ac_time : (1 / vt.a + 1 / vt.c) * 1.5 = 1
  bc_time : (1 / vt.b + 1 / vt.c) * 2 = 1

/-- Theorem stating that given the conditions, the time to fill the pool with valves A and B is 1.2 hours -/
theorem pool_fill_time_ab (conditions : PoolFillConditions) : 
  (1 / conditions.vt.a + 1 / conditions.vt.b) * 1.2 = 1 := by
  sorry

end pool_fill_time_ab_l3329_332928


namespace nth_equation_holds_l3329_332995

theorem nth_equation_holds (n : ℕ) : 
  (n : ℚ) / (n + 1) = (n + 3 * 2 * n) / (n + 1 + 3 * 2 * (n + 1)) := by
  sorry

end nth_equation_holds_l3329_332995


namespace existence_of_square_root_of_minus_one_l3329_332923

theorem existence_of_square_root_of_minus_one (p : ℕ) (hp : Nat.Prime p) :
  (∃ a : ℤ, a^2 ≡ -1 [ZMOD p]) ↔ p ≡ 1 [MOD 4] := by sorry

end existence_of_square_root_of_minus_one_l3329_332923


namespace last_islander_is_knight_l3329_332987

/-- Represents the type of an islander: either a knight or a liar -/
inductive IslanderType
  | Knight
  | Liar

/-- Represents what an islander says about their neighbor -/
inductive Statement
  | Knight
  | Liar

/-- The number of islanders around the table -/
def numIslanders : Nat := 50

/-- Function that determines what an islander at a given position says -/
def statement (position : Nat) : Statement :=
  if position % 2 == 1 then Statement.Knight else Statement.Liar

/-- Function that determines the actual type of an islander based on their statement and the type of their neighbor -/
def actualType (position : Nat) (neighborType : IslanderType) : IslanderType :=
  match (statement position, neighborType) with
  | (Statement.Knight, IslanderType.Knight) => IslanderType.Knight
  | (Statement.Knight, IslanderType.Liar) => IslanderType.Liar
  | (Statement.Liar, IslanderType.Knight) => IslanderType.Liar
  | (Statement.Liar, IslanderType.Liar) => IslanderType.Knight

theorem last_islander_is_knight : 
  ∀ (first : IslanderType), actualType numIslanders first = IslanderType.Knight :=
by sorry

end last_islander_is_knight_l3329_332987


namespace kody_age_is_32_l3329_332999

-- Define Mohamed's current age
def mohamed_current_age : ℕ := 2 * 30

-- Define Mohamed's age four years ago
def mohamed_past_age : ℕ := mohamed_current_age - 4

-- Define Kody's age four years ago
def kody_past_age : ℕ := mohamed_past_age / 2

-- Define Kody's current age
def kody_current_age : ℕ := kody_past_age + 4

-- Theorem stating Kody's current age
theorem kody_age_is_32 : kody_current_age = 32 := by
  sorry

end kody_age_is_32_l3329_332999


namespace delta_triple_72_l3329_332973

-- Define the Δ function
def Δ (N : ℝ) : ℝ := 0.4 * N + 2

-- Theorem statement
theorem delta_triple_72 : Δ (Δ (Δ 72)) = 7.728 := by
  sorry

end delta_triple_72_l3329_332973


namespace arithmetic_geometric_progression_l3329_332958

theorem arithmetic_geometric_progression (a b : ℝ) : 
  (1 = (a + b) / 2) →  -- arithmetic progression condition
  (1 = |a * b|) →      -- geometric progression condition
  ((a = 1 ∧ b = 1) ∨ 
   (a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨ 
   (a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2)) := by
sorry

end arithmetic_geometric_progression_l3329_332958


namespace problem_statement_l3329_332982

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  (0 < a * b ∧ a * b ≤ 1) ∧ (a^2 + b^2 ≥ 2) ∧ (0 < b ∧ b < 2) := by
  sorry

end problem_statement_l3329_332982


namespace ellipse_line_intersection_l3329_332967

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The line equation -/
def line (x y m : ℝ) : Prop := y = x + m

/-- The line intersects the ellipse at two distinct points -/
def intersects_at_two_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line x₁ y₁ m ∧ line x₂ y₂ m

/-- Main theorem -/
theorem ellipse_line_intersection (m : ℝ) :
  intersects_at_two_points m ↔ m ∈ Set.Ioo (-Real.sqrt 7) (Real.sqrt 7) :=
sorry

end ellipse_line_intersection_l3329_332967


namespace cubic_root_implies_coefficients_l3329_332909

theorem cubic_root_implies_coefficients 
  (a b : ℝ) 
  (h : (2 - 3*Complex.I)^3 + a*(2 - 3*Complex.I)^2 - 2*(2 - 3*Complex.I) + b = 0) : 
  a = -1/4 ∧ b = 195/4 := by
  sorry

end cubic_root_implies_coefficients_l3329_332909


namespace factorial_ratio_equals_seven_and_half_l3329_332952

theorem factorial_ratio_equals_seven_and_half :
  (Nat.factorial 10 * Nat.factorial 7 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 8) = 15 / 2 := by
  sorry

end factorial_ratio_equals_seven_and_half_l3329_332952


namespace tree_height_when_boy_grows_l3329_332953

-- Define the problem parameters
def initial_tree_height : ℝ := 16
def initial_boy_height : ℝ := 24
def final_boy_height : ℝ := 36

-- Define the growth rate relationship
def tree_growth_rate (boy_growth : ℝ) : ℝ := 2 * boy_growth

-- Theorem statement
theorem tree_height_when_boy_grows (boy_growth : ℝ) 
  (h : final_boy_height = initial_boy_height + boy_growth) :
  initial_tree_height + tree_growth_rate boy_growth = 40 :=
by
  sorry


end tree_height_when_boy_grows_l3329_332953


namespace grunters_win_probability_l3329_332950

theorem grunters_win_probability (n : ℕ) (p : ℚ) (h1 : n = 6) (h2 : p = 3/5) :
  p ^ n = 729 / 15625 := by
  sorry

end grunters_win_probability_l3329_332950


namespace min_shapes_for_square_l3329_332957

/-- The area of each shape in square units -/
def shape_area : ℕ := 3

/-- The side length of the smallest possible square that can be formed -/
def square_side : ℕ := 6

/-- The theorem stating the minimum number of shapes required -/
theorem min_shapes_for_square :
  let total_area : ℕ := square_side * square_side
  let num_shapes : ℕ := total_area / shape_area
  (∀ n : ℕ, n < num_shapes → n * shape_area < square_side * square_side) ∧
  (num_shapes * shape_area = square_side * square_side) ∧
  (∃ (arrangement : ℕ → ℕ → ℕ),
    (∀ i j : ℕ, i < square_side ∧ j < square_side →
      ∃ k : ℕ, k < num_shapes ∧ arrangement i j = k)) :=
by sorry

end min_shapes_for_square_l3329_332957


namespace inequality_proofs_l3329_332974

theorem inequality_proofs :
  (∀ x : ℝ, |x - 1| < 1 - 2*x ↔ x ∈ Set.Ioo 0 1) ∧
  (∀ x : ℝ, |x - 1| - |x + 1| > x ↔ x ∈ Set.Ioi (-1) ∪ Set.Ico (-1) 0) :=
by sorry

end inequality_proofs_l3329_332974


namespace money_sharing_l3329_332976

theorem money_sharing (total : ℚ) (per_person : ℚ) (num_people : ℕ) : 
  total = 3.75 ∧ per_person = 1.25 → num_people = 3 ∧ total = num_people * per_person :=
by sorry

end money_sharing_l3329_332976


namespace pascals_triangle_sum_l3329_332904

theorem pascals_triangle_sum (n : ℕ) : 
  n = 51 → Nat.choose n 4 + Nat.choose n 6 = 18249360 := by
  sorry

end pascals_triangle_sum_l3329_332904


namespace simple_interest_problem_l3329_332911

/-- Calculate simple interest given principal, rate, and time -/
def simple_interest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- The problem statement -/
theorem simple_interest_problem :
  let principal : ℚ := 26775
  let rate : ℚ := 3
  let time : ℚ := 5
  simple_interest principal rate time = 803.25 := by
  sorry

end simple_interest_problem_l3329_332911


namespace ladybug_count_l3329_332966

/-- The number of ladybugs with spots -/
def ladybugs_with_spots : ℕ := 12170

/-- The number of ladybugs without spots -/
def ladybugs_without_spots : ℕ := 54912

/-- The total number of ladybugs -/
def total_ladybugs : ℕ := ladybugs_with_spots + ladybugs_without_spots

theorem ladybug_count : total_ladybugs = 67082 := by
  sorry

end ladybug_count_l3329_332966


namespace a_gt_one_sufficient_not_necessary_l3329_332939

theorem a_gt_one_sufficient_not_necessary (a : ℝ) :
  (a > 1 → 1/a < 1) ∧ ∃ b : ℝ, 1/b < 1 ∧ ¬(b > 1) :=
by sorry

end a_gt_one_sufficient_not_necessary_l3329_332939


namespace fermat_mod_large_prime_l3329_332903

theorem fermat_mod_large_prime (n : ℕ) (hn : n > 0) :
  ∃ M : ℕ, ∀ p : ℕ, p > M → Prime p →
    ∃ x y z : ℤ, (x^n + y^n) % p = z^n % p ∧ (x * y * z) % p ≠ 0 := by
  sorry

end fermat_mod_large_prime_l3329_332903


namespace max_z_value_l3329_332971

theorem max_z_value (x y z : ℕ) : 
  7 < x → x < 9 → 9 < y → y < 15 → 
  0 < z → 
  Nat.Prime x → Nat.Prime y → Nat.Prime z →
  (y - x) % z = 0 →
  z ≤ 2 :=
sorry

end max_z_value_l3329_332971


namespace smallest_positive_solution_congruence_l3329_332997

theorem smallest_positive_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (4 * x) % 37 = 17 % 37 ∧ 
  ∀ (y : ℕ), y > 0 → (4 * y) % 37 = 17 % 37 → x ≤ y :=
by sorry

end smallest_positive_solution_congruence_l3329_332997


namespace group_size_problem_l3329_332994

theorem group_size_problem (total_paise : ℕ) (h1 : total_paise = 5776) : ∃ n : ℕ, n * n = total_paise ∧ n = 76 := by
  sorry

end group_size_problem_l3329_332994


namespace special_school_student_count_l3329_332975

/-- Represents a school for deaf and blind students -/
structure School where
  deaf_students : ℕ
  blind_students : ℕ

/-- The total number of students in the school -/
def total_students (s : School) : ℕ := s.deaf_students + s.blind_students

/-- Theorem: Given a school where the deaf student population is three times 
    the size of the blind student population, and the number of deaf students 
    is 180, the total number of students is 240. -/
theorem special_school_student_count :
  ∀ (s : School),
  s.deaf_students = 180 →
  s.deaf_students = 3 * s.blind_students →
  total_students s = 240 := by
  sorry

end special_school_student_count_l3329_332975


namespace games_not_working_l3329_332925

theorem games_not_working (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (good_games : ℕ) :
  games_from_friend = 2 →
  games_from_garage_sale = 2 →
  good_games = 2 →
  games_from_friend + games_from_garage_sale - good_games = 2 :=
by
  sorry

end games_not_working_l3329_332925


namespace inequality_proof_l3329_332998

theorem inequality_proof (w x y z : ℝ) 
  (h_non_neg : w ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_sum : w * x + x * y + y * z + z * w = 1) : 
  w^3 / (x + y + z) + x^3 / (w + y + z) + y^3 / (w + x + z) + z^3 / (w + x + y) ≥ 1/3 := by
  sorry

end inequality_proof_l3329_332998


namespace bart_survey_earnings_l3329_332965

theorem bart_survey_earnings :
  let questions_per_survey : ℕ := 10
  let earnings_per_question : ℚ := 0.2
  let monday_surveys : ℕ := 3
  let tuesday_surveys : ℕ := 4
  
  let total_questions := questions_per_survey * (monday_surveys + tuesday_surveys)
  let total_earnings := (total_questions : ℚ) * earnings_per_question

  total_earnings = 14 :=
by sorry

end bart_survey_earnings_l3329_332965


namespace celebrity_match_probability_l3329_332972

/-- The number of celebrities -/
def n : ℕ := 4

/-- The probability of correctly matching all celebrities with their pictures and hobbies -/
def correct_match_probability : ℚ := 1 / (n.factorial * n.factorial)

/-- Theorem: The probability of correctly matching all celebrities with their pictures and hobbies is 1/576 -/
theorem celebrity_match_probability :
  correct_match_probability = 1 / 576 := by sorry

end celebrity_match_probability_l3329_332972


namespace inequality_solution_l3329_332921

theorem inequality_solution (x : ℝ) : 
  (3 - 2 / (3 * x + 4) < 5) ↔ (x < -5/3 ∧ x ≠ -4/3) :=
by sorry

end inequality_solution_l3329_332921


namespace smallest_integer_with_remainders_l3329_332996

theorem smallest_integer_with_remainders : ∃! M : ℕ,
  (M > 0) ∧
  (M % 7 = 6) ∧
  (M % 8 = 7) ∧
  (M % 9 = 8) ∧
  (M % 10 = 9) ∧
  (M % 11 = 10) ∧
  (M % 12 = 11) ∧
  (∀ n : ℕ, n > 0 ∧
    n % 7 = 6 ∧
    n % 8 = 7 ∧
    n % 9 = 8 ∧
    n % 10 = 9 ∧
    n % 11 = 10 ∧
    n % 12 = 11 → n ≥ M) ∧
  M = 27719 :=
by sorry

end smallest_integer_with_remainders_l3329_332996


namespace cube_volume_surface_area_l3329_332980

theorem cube_volume_surface_area (x : ℝ) :
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 := by
  sorry

end cube_volume_surface_area_l3329_332980


namespace product_of_roots_cubic_l3329_332919

theorem product_of_roots_cubic (a b c : ℂ) : 
  (3 * a^3 - 4 * a^2 + 9 * a - 18 = 0) ∧ 
  (3 * b^3 - 4 * b^2 + 9 * b - 18 = 0) ∧ 
  (3 * c^3 - 4 * c^2 + 9 * c - 18 = 0) → 
  a * b * c = 6 := by
sorry

end product_of_roots_cubic_l3329_332919


namespace sexagesimal_cubes_correct_l3329_332915

/-- Converts a sexagesimal number to decimal -/
def sexagesimal_to_decimal (whole : ℕ) (frac : ℕ) : ℕ :=
  whole * 60 + frac

/-- Checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

/-- Represents a sexagesimal number as a pair of natural numbers -/
structure Sexagesimal :=
  (whole : ℕ)
  (frac : ℕ)

/-- Theorem stating that the sexagesimal representation of cubes is correct for numbers from 1 to 32 -/
theorem sexagesimal_cubes_correct :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 32 →
    ∃ s : Sexagesimal, 
      sexagesimal_to_decimal s.whole s.frac = n^3 ∧
      is_perfect_cube (sexagesimal_to_decimal s.whole s.frac) :=
by sorry

end sexagesimal_cubes_correct_l3329_332915


namespace largest_solution_quadratic_l3329_332933

theorem largest_solution_quadratic : 
  let f : ℝ → ℝ := λ x => 6 * x^2 - 31 * x + 35
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x ∧ x = 2.5 := by
  sorry

end largest_solution_quadratic_l3329_332933


namespace ab_bc_ratio_is_two_plus_sqrt_three_l3329_332951

-- Define the quadrilateral ABCD
structure Quadrilateral (A B C D : ℝ × ℝ) : Prop where
  right_angle_B : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0
  right_angle_C : (C.1 - B.1) * (D.1 - C.1) + (C.2 - B.2) * (D.2 - C.2) = 0

-- Define similarity of triangles
def similar_triangles (A B C D E F : ℝ × ℝ) : Prop :=
  ∃ k > 0, (B.1 - A.1)^2 + (B.2 - A.2)^2 = k * ((E.1 - D.1)^2 + (E.2 - D.2)^2) ∧
            (C.1 - B.1)^2 + (C.2 - B.2)^2 = k * ((F.1 - E.1)^2 + (F.2 - E.2)^2) ∧
            (A.1 - C.1)^2 + (A.2 - C.2)^2 = k * ((D.1 - F.1)^2 + (D.2 - F.2)^2)

-- Define the area of a triangle
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Main theorem
theorem ab_bc_ratio_is_two_plus_sqrt_three
  (A B C D E : ℝ × ℝ)
  (h_quad : Quadrilateral A B C D)
  (h_sim_ABC_BCD : similar_triangles A B C B C D)
  (h_AB_gt_BC : (A.1 - B.1)^2 + (A.2 - B.2)^2 > (B.1 - C.1)^2 + (B.2 - C.2)^2)
  (h_E_interior : ∃ t u : ℝ, 0 < t ∧ t < 1 ∧ 0 < u ∧ u < 1 ∧
    E = (t * A.1 + (1 - t) * C.1, u * B.2 + (1 - u) * D.2))
  (h_sim_ABC_CEB : similar_triangles A B C C E B)
  (h_area_ratio : triangle_area A E D = 25 * triangle_area C E B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) / Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 2 + Real.sqrt 3 :=
sorry

end ab_bc_ratio_is_two_plus_sqrt_three_l3329_332951


namespace problems_left_to_grade_l3329_332986

theorem problems_left_to_grade 
  (total_worksheets : ℕ) 
  (graded_worksheets : ℕ) 
  (problems_per_worksheet : ℕ) 
  (h1 : total_worksheets = 9) 
  (h2 : graded_worksheets = 5) 
  (h3 : problems_per_worksheet = 4) : 
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 16 :=
by sorry

end problems_left_to_grade_l3329_332986


namespace S_bounds_l3329_332988

def S : Set ℝ := {y | ∃ x : ℝ, x ≥ 0 ∧ y = (3*x + 2)/(x + 1)}

theorem S_bounds : 
  ∃ (M m : ℝ), 
    (∀ y ∈ S, y ≤ M) ∧ 
    (∀ y ∈ S, y ≥ m) ∧ 
    (M ∉ S) ∧ 
    (m ∈ S) ∧
    (M = 3) ∧ 
    (m = 2) :=
by sorry

end S_bounds_l3329_332988


namespace g_properties_imply_g_50_l3329_332937

noncomputable def g (p q r s x : ℝ) : ℝ := (p * x + q) / (r * x + s)

theorem g_properties_imply_g_50 (p q r s : ℝ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  g p q r s 23 = 23 →
  g p q r s 101 = 101 →
  (∀ x : ℝ, x ≠ -s/r → g p q r s (g p q r s x) = x) →
  g p q r s 50 = -61 := by sorry

end g_properties_imply_g_50_l3329_332937
