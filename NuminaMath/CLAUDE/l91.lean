import Mathlib

namespace partition_16_into_8_pairs_eq_2027025_l91_9147

/-- The number of ways to partition 16 distinct elements into 8 unordered pairs -/
def partition_16_into_8_pairs : ℕ :=
  (Nat.factorial 16) / (Nat.pow 2 8 * Nat.factorial 8)

/-- Theorem stating that the number of ways to partition 16 distinct elements
    into 8 unordered pairs is equal to 2027025 -/
theorem partition_16_into_8_pairs_eq_2027025 :
  partition_16_into_8_pairs = 2027025 := by
  sorry

end partition_16_into_8_pairs_eq_2027025_l91_9147


namespace time_in_terms_of_angle_and_angular_velocity_l91_9104

theorem time_in_terms_of_angle_and_angular_velocity 
  (α ω ω₀ θ t : ℝ) 
  (h1 : ω = α * t + ω₀) 
  (h2 : θ = (1/2) * α * t^2 + ω₀ * t) : 
  t = 2 * θ / (ω + ω₀) := by
sorry

end time_in_terms_of_angle_and_angular_velocity_l91_9104


namespace constant_zero_sequence_l91_9106

def is_sum_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ k, S (k + 1) = S k + a (k + 1)

theorem constant_zero_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h : ∀ k, S (k + 1) + S k = a (k + 1)) :
  ∀ n, a n = 0 :=
by sorry

end constant_zero_sequence_l91_9106


namespace parabola_circle_intersection_l91_9199

/-- The value of p for which the axis of the parabola y^2 = 2px intersects
    the circle (x+1)^2 + y^2 = 4 at two points with distance 2√3 -/
theorem parabola_circle_intersection (p : ℝ) : p > 0 →
  (∃ A B : ℝ × ℝ,
    (A.1 + 1)^2 + A.2^2 = 4 ∧
    (B.1 + 1)^2 + B.2^2 = 4 ∧
    A.2^2 = 2 * p * A.1 ∧
    B.2^2 = 2 * p * B.1 ∧
    A.1 = B.1 ∧
    (A.2 - B.2)^2 = 12) →
  p = 4 := by sorry

end parabola_circle_intersection_l91_9199


namespace intersection_in_square_l91_9141

-- Define the trajectory function
def trajectory (x : ℝ) : ℝ :=
  (((x^5 - 2013)^5 - 2013)^5 - 2013)^5

-- Define the radar line function
def radar_line (x : ℝ) : ℝ :=
  x + 2013

-- Define the function for the difference between trajectory and radar line
def intersection_function (x : ℝ) : ℝ :=
  trajectory x - radar_line x

-- Theorem statement
theorem intersection_in_square :
  ∃ (x y : ℝ), 
    intersection_function x = 0 ∧ 
    4 ≤ x ∧ x < 5 ∧
    2017 ≤ y ∧ y < 2018 ∧
    y = radar_line x :=
sorry

end intersection_in_square_l91_9141


namespace no_real_solution_for_log_equation_l91_9125

theorem no_real_solution_for_log_equation :
  ¬∃ (x : ℝ), (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 5*x - 14 > 0) ∧
  (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 5*x - 14)) := by
  sorry

end no_real_solution_for_log_equation_l91_9125


namespace checkerboard_matching_sum_l91_9164

/-- Row-wise numbering function -/
def f (i j : ℕ) : ℕ := 19 * (i - 1) + j

/-- Column-wise numbering function -/
def g (i j : ℕ) : ℕ := 15 * (j - 1) + i

/-- The set of pairs (i, j) where the numbers match in both systems -/
def matching_squares : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => f p.1 p.2 = g p.1 p.2)
    (Finset.product (Finset.range 15) (Finset.range 19))

theorem checkerboard_matching_sum :
  (matching_squares.sum fun p => f p.1 p.2) = 668 := by
  sorry


end checkerboard_matching_sum_l91_9164


namespace percentage_of_hindu_boys_l91_9144

theorem percentage_of_hindu_boys (total : ℕ) (muslim_percent : ℚ) (sikh_percent : ℚ) (other : ℕ) :
  total = 300 →
  muslim_percent = 44 / 100 →
  sikh_percent = 10 / 100 →
  other = 54 →
  (total - (muslim_percent * total + sikh_percent * total + other)) / total = 28 / 100 := by
  sorry

end percentage_of_hindu_boys_l91_9144


namespace renovation_project_materials_l91_9134

theorem renovation_project_materials (sand dirt cement gravel stone : ℝ) 
  (h1 : sand = 0.17)
  (h2 : dirt = 0.33)
  (h3 : cement = 0.17)
  (h4 : gravel = 0.25)
  (h5 : stone = 0.08) :
  sand + dirt + cement + gravel + stone = 1 := by
  sorry

end renovation_project_materials_l91_9134


namespace silk_per_dress_is_five_l91_9105

/-- Calculates the amount of silk needed for each dress given the initial silk amount,
    number of friends, silk given to each friend, and number of dresses made. -/
def silk_per_dress (initial_silk : ℕ) (num_friends : ℕ) (silk_per_friend : ℕ) (num_dresses : ℕ) : ℕ :=
  (initial_silk - num_friends * silk_per_friend) / num_dresses

/-- Proves that given the specified conditions, the amount of silk needed for each dress is 5 meters. -/
theorem silk_per_dress_is_five :
  silk_per_dress 600 5 20 100 = 5 := by
  sorry

end silk_per_dress_is_five_l91_9105


namespace gcd_difference_square_l91_9198

theorem gcd_difference_square (x y z : ℕ+) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ (k : ℕ), (Nat.gcd x.val (Nat.gcd y.val z.val)) * (y.val - x.val) = k ^ 2 := by
  sorry

end gcd_difference_square_l91_9198


namespace restaurant_group_size_l91_9189

/-- Calculates the total number of people in a restaurant group given the following conditions:
  * The cost of an adult meal is $7
  * Kids eat for free
  * There are 9 kids in the group
  * The total cost for the group is $28
-/
theorem restaurant_group_size :
  let adult_meal_cost : ℕ := 7
  let kids_count : ℕ := 9
  let total_cost : ℕ := 28
  let adult_count : ℕ := total_cost / adult_meal_cost
  let total_people : ℕ := adult_count + kids_count
  total_people = 13 := by
  sorry

end restaurant_group_size_l91_9189


namespace geometric_sequence_S_3_range_l91_9178

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the sum of the first three terms
def S_3 (a : ℕ → ℝ) : ℝ := a 1 + a 2 + a 3

-- Theorem statement
theorem geometric_sequence_S_3_range
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_a2 : a 2 = 1) :
  ∃ y : ℝ, S_3 a = y ↔ y ∈ Set.Ici 3 ∪ Set.Iic (-1) :=
sorry

end geometric_sequence_S_3_range_l91_9178


namespace circle_diameter_l91_9171

theorem circle_diameter (C : ℝ) (h : C = 100) : C / π = 100 / π := by
  sorry

end circle_diameter_l91_9171


namespace simplify_fraction_simplify_harmonic_root1_simplify_harmonic_root2_calculate_expression_l91_9148

-- 1. Simplify fraction with square root
theorem simplify_fraction : (2 : ℝ) / (Real.sqrt 3 - 1) = Real.sqrt 3 + 1 := by sorry

-- 2. Simplify harmonic quadratic root (case 1)
theorem simplify_harmonic_root1 : Real.sqrt (4 + 2 * Real.sqrt 3) = Real.sqrt 3 + 1 := by sorry

-- 3. Simplify harmonic quadratic root (case 2)
theorem simplify_harmonic_root2 : Real.sqrt (6 - 2 * Real.sqrt 5) = Real.sqrt 5 - 1 := by sorry

-- 4. Calculate expression with harmonic quadratic roots
theorem calculate_expression (m n : ℝ) 
  (hm : m = 1 / Real.sqrt (5 + 2 * Real.sqrt 6))
  (hn : n = 1 / Real.sqrt (5 - 2 * Real.sqrt 6)) :
  (m - n) / (m + n) = -(Real.sqrt 6) / 3 := by sorry

end simplify_fraction_simplify_harmonic_root1_simplify_harmonic_root2_calculate_expression_l91_9148


namespace sugar_sacks_weight_l91_9129

theorem sugar_sacks_weight (x y : ℝ) 
  (h1 : y - x = 8)
  (h2 : x - 1 = 0.6 * (y + 1)) : 
  x + y = 40 := by
sorry

end sugar_sacks_weight_l91_9129


namespace triangle_theorem_l91_9160

noncomputable section

variables {a b c : ℝ} {A B C : ℝ}

def triangle_area (a b c : ℝ) : ℝ := (1/4) * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c))

theorem triangle_theorem 
  (h1 : b^2 + c^2 - a^2 = a*c*(Real.cos C) + c^2*(Real.cos A))
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  (h6 : a*Real.sin B = b*Real.sin A)
  (h7 : b*Real.sin C = c*Real.sin B)
  (h8 : triangle_area a b c = 25*(Real.sqrt 3)/4)
  (h9 : a = 5) :
  A = π/3 ∧ Real.sin B + Real.sin C = Real.sqrt 3 := by
  sorry

end triangle_theorem_l91_9160


namespace geometric_arithmetic_sequence_ratio_l91_9112

theorem geometric_arithmetic_sequence_ratio 
  (x y z r : ℝ) 
  (h1 : y = r * x) 
  (h2 : z = r * y) 
  (h3 : x ≠ y) 
  (h4 : 2 * (2 * y) = x + 3 * z) : 
  r = 1/3 := by
  sorry

end geometric_arithmetic_sequence_ratio_l91_9112


namespace inscribed_square_area_ratio_l91_9184

/-- A circle with a square inscribed in it, where the square's vertices touch the circle
    and the side of the square intersects the circle such that each intersection segment
    equals twice the radius of the circle. -/
structure InscribedSquare where
  r : ℝ  -- radius of the circle
  s : ℝ  -- side length of the square
  h1 : s = r * Real.sqrt 2  -- relationship between side length and radius
  h2 : s * Real.sqrt 2 = 2 * r  -- diagonal of square equals diameter of circle

/-- The ratio of the area of the inscribed square to the area of the circle is 2/π. -/
theorem inscribed_square_area_ratio (square : InscribedSquare) :
  (square.s ^ 2) / (Real.pi * square.r ^ 2) = 2 / Real.pi :=
by sorry

end inscribed_square_area_ratio_l91_9184


namespace max_value_problem_l91_9123

theorem max_value_problem (a₁ a₂ a₃ a₄ : ℝ) 
  (h_pos₁ : 0 < a₁) (h_pos₂ : 0 < a₂) (h_pos₃ : 0 < a₃) (h_pos₄ : 0 < a₄)
  (h₁ : a₁ ≥ a₂ * a₃^2) (h₂ : a₂ ≥ a₃ * a₄^2) 
  (h₃ : a₃ ≥ a₄ * a₁^2) (h₄ : a₄ ≥ a₁ * a₂^2) : 
  a₁ * a₂ * a₃ * a₄ * (a₁ - a₂ * a₃^2) * (a₂ - a₃ * a₄^2) * 
  (a₃ - a₄ * a₁^2) * (a₄ - a₁ * a₂^2) ≤ 1 / 256 := by
  sorry

end max_value_problem_l91_9123


namespace highest_power_of_three_dividing_N_l91_9181

def N : ℕ := sorry  -- Definition of N as concatenation of integers from 34 to 76

theorem highest_power_of_three_dividing_N :
  ∃ k : ℕ, (3^k ∣ N) ∧ ¬(3^(k+1) ∣ N) ∧ k = 1 := by
  sorry

end highest_power_of_three_dividing_N_l91_9181


namespace sequence_position_l91_9110

/-- The general term of the sequence -/
def sequenceTerm (n : ℕ) : ℚ := (n + 3) / (n + 1)

/-- The position we want to prove -/
def position : ℕ := 14

/-- The fraction we're looking for -/
def targetFraction : ℚ := 17 / 15

theorem sequence_position :
  sequenceTerm position = targetFraction := by sorry

end sequence_position_l91_9110


namespace sixth_power_of_z_l91_9137

theorem sixth_power_of_z (z : ℂ) : z = (Real.sqrt 3 - Complex.I) / 2 → z^6 = -1 := by
  sorry

end sixth_power_of_z_l91_9137


namespace quadratic_roots_greater_than_one_implies_s_positive_l91_9100

theorem quadratic_roots_greater_than_one_implies_s_positive
  (b c : ℝ)
  (h1 : ∃ x y : ℝ, x > 1 ∧ y > 1 ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0)
  : b + c + 1 > 0 :=
by sorry

end quadratic_roots_greater_than_one_implies_s_positive_l91_9100


namespace total_pencils_l91_9161

theorem total_pencils (drawer : Real) (desk_initial : Real) (pencil_case : Real) (dan_added : Real)
  (h1 : drawer = 43.5)
  (h2 : desk_initial = 19.25)
  (h3 : pencil_case = 8.75)
  (h4 : dan_added = 16) :
  drawer + desk_initial + pencil_case + dan_added = 87.5 := by
  sorry

end total_pencils_l91_9161


namespace sum_of_three_integers_l91_9126

theorem sum_of_three_integers (large medium small : ℕ+) 
  (sum_large_medium : large + medium = 2003)
  (diff_medium_small : medium - small = 1000) :
  large + medium + small = 2004 := by
  sorry

end sum_of_three_integers_l91_9126


namespace equilateral_triangle_area_increase_l91_9132

theorem equilateral_triangle_area_increase :
  ∀ s : ℝ,
  s > 0 →
  (s^2 * Real.sqrt 3) / 4 = 36 * Real.sqrt 3 →
  let new_s := s + 2
  let new_area := (new_s^2 * Real.sqrt 3) / 4
  let original_area := (s^2 * Real.sqrt 3) / 4
  new_area - original_area = 13 * Real.sqrt 3 := by
sorry

end equilateral_triangle_area_increase_l91_9132


namespace u_2023_equals_4_l91_9154

-- Define the function f
def f : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 2
| 4 => 1
| 5 => 4
| _ => 0  -- Default case for completeness

-- Define the sequence u
def u : ℕ → ℕ
| 0 => 5  -- u₀ = 5
| n + 1 => f (u n)  -- uₙ₊₁ = f(uₙ) for n ≥ 0

-- Theorem statement
theorem u_2023_equals_4 : u 2023 = 4 := by
  sorry

end u_2023_equals_4_l91_9154


namespace identify_all_pairs_in_75_attempts_l91_9155

/-- Represents a door-key system with 100 doors and keys -/
structure DoorKeySystem :=
  (doors : Fin 100 → Nat)
  (keys : Fin 100 → Nat)
  (key_matches : ∀ i : Fin 100, (keys i = doors i) ∨ (keys i = doors i + 1) ∨ (keys i + 1 = doors i))

/-- Represents an attempt to match a key to a door -/
def Attempt := Fin 100 × Fin 100

/-- A function that determines if all key-door pairs can be identified within a given number of attempts -/
def can_identify_all_pairs (system : DoorKeySystem) (max_attempts : Nat) : Prop :=
  ∃ (attempts : List Attempt), 
    attempts.length ≤ max_attempts ∧ 
    (∀ i : Fin 100, ∃ j : Fin 100, (i, j) ∈ attempts ∨ (j, i) ∈ attempts) ∧
    (∀ i j : Fin 100, system.keys i = system.doors j → (i, j) ∈ attempts ∨ (j, i) ∈ attempts)

/-- Theorem stating that all key-door pairs can be identified within 75 attempts -/
theorem identify_all_pairs_in_75_attempts :
  ∀ system : DoorKeySystem, can_identify_all_pairs system 75 :=
sorry

end identify_all_pairs_in_75_attempts_l91_9155


namespace cube_edge_length_l91_9173

theorem cube_edge_length 
  (paint_cost : ℝ) 
  (coverage_per_quart : ℝ) 
  (total_cost : ℝ) 
  (h1 : paint_cost = 3.20)
  (h2 : coverage_per_quart = 120)
  (h3 : total_cost = 16) : 
  ∃ (edge_length : ℝ), edge_length = 10 ∧ 
  6 * edge_length^2 = (total_cost / paint_cost) * coverage_per_quart :=
by
  sorry

end cube_edge_length_l91_9173


namespace chinese_chess_draw_probability_l91_9191

/-- Given the probabilities of winning and not losing for player A in Chinese chess,
    calculate the probability of a draw between player A and player B. -/
theorem chinese_chess_draw_probability
  (prob_win : ℝ) (prob_not_lose : ℝ)
  (h_win : prob_win = 0.4)
  (h_not_lose : prob_not_lose = 0.9) :
  prob_not_lose - prob_win = 0.5 := by
  sorry

end chinese_chess_draw_probability_l91_9191


namespace unique_solution_implies_a_less_than_neg_one_l91_9192

-- Define the function f(x) = ax + 1
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 1

-- State the theorem
theorem unique_solution_implies_a_less_than_neg_one :
  ∀ a : ℝ, (∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f a x = 0) → a < -1 := by
  sorry

end unique_solution_implies_a_less_than_neg_one_l91_9192


namespace fractional_part_inequality_l91_9146

theorem fractional_part_inequality (α : ℝ) (h_α : 0 < α ∧ α < 1) :
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ ∀ n : ℕ+, α^(n : ℝ) < (n : ℝ) * x - ⌊(n : ℝ) * x⌋ := by
  sorry

end fractional_part_inequality_l91_9146


namespace inequality_system_solution_l91_9152

theorem inequality_system_solution (x : ℝ) :
  (1 / x < 1 ∧ |4 * x - 1| > 2) ↔ (x < -1/4 ∨ x > 1) :=
by sorry

end inequality_system_solution_l91_9152


namespace function_and_inequality_l91_9107

/-- Given a function f(x) = (ax+b)/(x-2) where f(x) - x + 12 = 0 has roots 3 and 4,
    prove the form of f(x) and the solution set of f(x) < k for k > 1 -/
theorem function_and_inequality (a b : ℝ) (h1 : ∀ x : ℝ, x ≠ 2 → (a * x + b) / (x - 2) - x + 12 = 0) 
    (h2 : (a * 3 + b) / (3 - 2) - 3 + 12 = 0) (h3 : (a * 4 + b) / (4 - 2) - 4 + 12 = 0) :
  (∀ x : ℝ, x ≠ 2 → (a * x + b) / (x - 2) = (-x + 2) / (x - 2)) ∧
  (∀ k : ℝ, k > 1 →
    (1 < k ∧ k < 2 → {x : ℝ | (-x + 2) / (x - 2) < k} = {x : ℝ | 1 < x ∧ x < k} ∪ {x : ℝ | x > 2}) ∧
    (k = 2 → {x : ℝ | (-x + 2) / (x - 2) < k} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | x > 2}) ∧
    (k > 2 → {x : ℝ | (-x + 2) / (x - 2) < k} = {x : ℝ | 1 < x ∧ x < 2} ∪ {x : ℝ | x > k})) :=
by sorry

end function_and_inequality_l91_9107


namespace delaware_cell_phones_count_l91_9139

/-- The number of cell phones in Delaware -/
def delaware_cell_phones (population : ℕ) (phones_per_thousand : ℕ) : ℕ :=
  (population / 1000) * phones_per_thousand

/-- Proof that the number of cell phones in Delaware is 655,502 -/
theorem delaware_cell_phones_count :
  delaware_cell_phones 974000 673 = 655502 := by
  sorry

end delaware_cell_phones_count_l91_9139


namespace rectangular_garden_length_l91_9150

/-- The length of a rectangular garden with perimeter 900 m and breadth 190 m is 260 m. -/
theorem rectangular_garden_length : 
  ∀ (length breadth : ℝ),
  breadth = 190 →
  2 * (length + breadth) = 900 →
  length = 260 := by
sorry

end rectangular_garden_length_l91_9150


namespace backyard_length_is_20_l91_9108

-- Define the backyard and shed dimensions
def backyard_width : ℝ := 13
def shed_length : ℝ := 3
def shed_width : ℝ := 5
def sod_area : ℝ := 245

-- Theorem statement
theorem backyard_length_is_20 :
  ∃ (L : ℝ), L * backyard_width - shed_length * shed_width = sod_area ∧ L = 20 := by
  sorry

end backyard_length_is_20_l91_9108


namespace curve_to_line_equation_l91_9170

/-- Given a curve parameterized by (x, y) = (3t + 6, 5t - 7), where t is a real number,
    prove that the equation of the line in the form y = mx + b is y = (5/3)x - 17. -/
theorem curve_to_line_equation :
  ∀ (t x y : ℝ), x = 3 * t + 6 ∧ y = 5 * t - 7 →
  ∃ (m b : ℝ), m = 5 / 3 ∧ b = -17 ∧ y = m * x + b :=
by sorry

end curve_to_line_equation_l91_9170


namespace berries_to_buy_l91_9156

def total_needed : Nat := 21
def strawberries : Nat := 4
def blueberries : Nat := 8

theorem berries_to_buy (total_needed strawberries blueberries : Nat) : 
  total_needed - (strawberries + blueberries) = 9 :=
by sorry

end berries_to_buy_l91_9156


namespace sarahs_bowling_score_l91_9127

/-- Sarah's bowling score problem -/
theorem sarahs_bowling_score :
  ∀ (sarah greg : ℕ),
  sarah = greg + 60 →
  sarah + greg = 260 →
  sarah = 160 := by
sorry

end sarahs_bowling_score_l91_9127


namespace inscribed_cube_surface_area_l91_9103

theorem inscribed_cube_surface_area (V : ℝ) (h : V = 256 * Real.pi / 3) :
  let R := (3 * V / (4 * Real.pi)) ^ (1/3)
  let a := 2 * R / Real.sqrt 3
  6 * a^2 = 128 := by
  sorry

end inscribed_cube_surface_area_l91_9103


namespace circle_circumference_when_equal_to_area_l91_9151

/-- 
For a circle where the circumference and area are numerically equal,
if the diameter is 4, then the circumference is 4π.
-/
theorem circle_circumference_when_equal_to_area (d : ℝ) (C : ℝ) (A : ℝ) : 
  C = A →  -- Circumference equals area
  d = 4 →  -- Diameter is 4
  C = π * d →  -- Definition of circumference
  A = π * (d/2)^2 →  -- Definition of area
  C = 4 * π := by
sorry

end circle_circumference_when_equal_to_area_l91_9151


namespace equation_solutions_l91_9153

/-- Given an equation a · b^x · c^(2x) = ∛(d)^(1/x) · ∜(e)^(1/x), 
    this theorem states that:
    1. When a = 2, b = 3, c = 5, d = 7, e = 11, there exist two real solutions.
    2. When a = 5, b = 3, c = 2, d = 1/7, e = 1/11, there are no real solutions. -/
theorem equation_solutions (a b c d e : ℝ) : 
  (a = 2 ∧ b = 3 ∧ c = 5 ∧ d = 7 ∧ e = 11 → 
    ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * b^x₁ * c^(2*x₁) = (d^(1/3))^(1/x₁) * (e^(1/4))^(1/x₁) ∧
                   a * b^x₂ * c^(2*x₂) = (d^(1/3))^(1/x₂) * (e^(1/4))^(1/x₂)) ∧
  (a = 5 ∧ b = 3 ∧ c = 2 ∧ d = 1/7 ∧ e = 1/11 → 
    ¬∃ x : ℝ, a * b^x * c^(2*x) = (d^(1/3))^(1/x) * (e^(1/4))^(1/x)) :=
by sorry


end equation_solutions_l91_9153


namespace ellipse_from_hyperbola_vertices_l91_9113

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, 
    the equation of the ellipse whose foci are the vertices of the hyperbola 
    is x²/16 + y²/12 = 1 -/
theorem ellipse_from_hyperbola_vertices (x y : ℝ) :
  let hyperbola := (x^2 / 4 - y^2 / 12 = 1)
  let ellipse := (x^2 / 16 + y^2 / 12 = 1)
  let hyperbola_vertex := 2
  let hyperbola_focus := 4
  hyperbola → ellipse := by sorry

end ellipse_from_hyperbola_vertices_l91_9113


namespace logo_scaling_l91_9166

theorem logo_scaling (w h W : ℝ) (hw : w > 0) (hh : h > 0) (hW : W > 0) :
  let scale := W / w
  let H := scale * h
  (W / w = H / h) ∧ (H = (W / w) * h) := by sorry

end logo_scaling_l91_9166


namespace watch_cost_price_l91_9133

/-- The cost price of a watch satisfying certain conditions -/
theorem watch_cost_price : ∃ (cp : ℚ), 
  cp > 0 ∧ 
  (0.9 * cp = cp - 0.1 * cp) ∧ 
  (1.04 * cp = cp + 0.04 * cp) ∧ 
  (1.04 * cp - 0.9 * cp = 168) ∧ 
  cp = 1200 := by
sorry

end watch_cost_price_l91_9133


namespace perpendicular_vectors_x_value_l91_9158

theorem perpendicular_vectors_x_value (a b : ℝ × ℝ) (x : ℝ) :
  a = (1, 2) →
  b = (-1, x) →
  a.1 * b.1 + a.2 * b.2 = 0 →
  x = 1/2 := by
sorry

end perpendicular_vectors_x_value_l91_9158


namespace binomial_coefficient_21_15_l91_9193

theorem binomial_coefficient_21_15 :
  (Nat.choose 20 13 = 77520) →
  (Nat.choose 20 14 = 38760) →
  (Nat.choose 22 15 = 170544) →
  (Nat.choose 21 15 = 54264) :=
by
  sorry

end binomial_coefficient_21_15_l91_9193


namespace field_width_calculation_l91_9180

/-- A rectangular football field with given dimensions and running conditions. -/
structure FootballField where
  length : ℝ
  width : ℝ
  laps : ℕ
  total_distance : ℝ

/-- The width of a football field given specific conditions. -/
def field_width (f : FootballField) : ℝ :=
  f.width

/-- Theorem stating the width of the field under given conditions. -/
theorem field_width_calculation (f : FootballField)
  (h1 : f.length = 100)
  (h2 : f.laps = 6)
  (h3 : f.total_distance = 1800)
  (h4 : f.total_distance = f.laps * (2 * f.length + 2 * f.width)) :
  field_width f = 50 := by
  sorry

#check field_width_calculation

end field_width_calculation_l91_9180


namespace emma_numbers_l91_9162

theorem emma_numbers (x y : ℤ) : 
  4 * x + 3 * y = 140 → (x = 20 ∨ y = 20) → x = 20 ∧ y = 20 := by
sorry

end emma_numbers_l91_9162


namespace unique_last_digit_for_divisibility_by_6_l91_9142

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def last_digit (n : ℕ) : ℕ := n % 10

def replace_last_digit (n : ℕ) (d : ℕ) : ℕ := (n / 10) * 10 + d

theorem unique_last_digit_for_divisibility_by_6 :
  ∀ d : ℕ, d < 10 →
    (is_divisible_by_6 (replace_last_digit 314270 d) ↔ d = last_digit 314274) :=
sorry

end unique_last_digit_for_divisibility_by_6_l91_9142


namespace jake_weight_proof_l91_9138

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 108

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 48

/-- The combined weight of Jake and his sister in pounds -/
def combined_weight : ℝ := 156

theorem jake_weight_proof :
  (jake_weight - 12 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = combined_weight) →
  jake_weight = 108 :=
by sorry

end jake_weight_proof_l91_9138


namespace fraction_simplification_l91_9101

theorem fraction_simplification : (5 * 8) / 10 = 4 := by
  sorry

end fraction_simplification_l91_9101


namespace sin_two_phi_l91_9119

theorem sin_two_phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) :
  Real.sin (2 * φ) = 120 / 169 := by
  sorry

end sin_two_phi_l91_9119


namespace geometric_mean_of_1_and_9_l91_9167

theorem geometric_mean_of_1_and_9 : 
  ∃ (x : ℝ), x^2 = 1 * 9 ∧ (x = 3 ∨ x = -3) := by
  sorry

end geometric_mean_of_1_and_9_l91_9167


namespace box_dimensions_l91_9124

theorem box_dimensions (a b c : ℝ) 
  (h1 : a + c = 17) 
  (h2 : a + b = 13) 
  (h3 : b + c = 20) 
  (h4 : a < b) 
  (h5 : b < c) : 
  a = 5 ∧ b = 8 ∧ c = 12 := by
sorry

end box_dimensions_l91_9124


namespace sets_theorem_l91_9187

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

-- Define the theorem
theorem sets_theorem :
  -- Part 1
  (A (1/2) ∩ (Set.univ \ B (1/2)) = {x | 9/4 ≤ x ∧ x < 5/2}) ∧
  -- Part 2
  (∀ a : ℝ, Set.Subset (A a) (B a) ↔ -1/2 ≤ a ∧ a ≤ (3 - Real.sqrt 5) / 2) :=
sorry

end sets_theorem_l91_9187


namespace power_quotient_square_l91_9188

theorem power_quotient_square : (19^12 / 19^8)^2 = 130321 := by sorry

end power_quotient_square_l91_9188


namespace fraction_addition_l91_9186

theorem fraction_addition (d : ℝ) : (6 + 5*d) / 9 + 3 = (33 + 5*d) / 9 := by
  sorry

end fraction_addition_l91_9186


namespace translation_theorem_l91_9140

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem (A B C D : Point) (t : Translation) :
  A.x = -1 ∧ A.y = 4 ∧
  C.x = 4 ∧ C.y = 7 ∧
  B.x = -4 ∧ B.y = -1 ∧
  C = applyTranslation t A ∧
  D = applyTranslation t B →
  D.x = 1 ∧ D.y = 2 := by
  sorry

end translation_theorem_l91_9140


namespace passenger_trips_scientific_notation_l91_9135

/-- The number of operating passenger trips in millions -/
def passenger_trips : ℝ := 56.99

/-- The scientific notation representation of the passenger trips -/
def scientific_notation : ℝ := 5.699 * (10^7)

/-- Theorem stating that the number of passenger trips in millions 
    is equal to its scientific notation representation -/
theorem passenger_trips_scientific_notation : 
  passenger_trips * 10^6 = scientific_notation := by sorry

end passenger_trips_scientific_notation_l91_9135


namespace continued_proportionality_and_linear_combination_l91_9118

theorem continued_proportionality_and_linear_combination :
  -- Part (1)
  (∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
    x / (2*y + z) = y / (2*z + x) ∧ y / (2*z + x) = z / (2*x + y) →
    x / (2*y + z) = 1/3) ∧
  -- Part (2)
  (∀ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a →
    (a + b) / (a - b) = (b + c) / (2*(b - c)) ∧
    (b + c) / (2*(b - c)) = (c + a) / (3*(c - a)) →
    8*a + 9*b + 5*c = 0) := by
  sorry

end continued_proportionality_and_linear_combination_l91_9118


namespace time_difference_1200_miles_l91_9122

/-- Calculates the time difference for a 1200-mile trip between two given speeds -/
theorem time_difference_1200_miles (speed1 speed2 : ℝ) (h1 : speed1 > 0) (h2 : speed2 > 0) :
  (1200 / speed1 - 1200 / speed2) = 4 ↔ speed1 = 60 ∧ speed2 = 50 := by sorry

end time_difference_1200_miles_l91_9122


namespace ellipse_equivalence_l91_9114

/-- Given ellipse equation -/
def given_ellipse (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 36

/-- New ellipse equation -/
def new_ellipse (x y : ℝ) : Prop := x^2 / 15 + y^2 / 10 = 1

/-- Foci of an ellipse -/
def has_same_foci (e1 e2 : (ℝ → ℝ → Prop)) : Prop := sorry

theorem ellipse_equivalence :
  has_same_foci given_ellipse new_ellipse ∧ new_ellipse (-3) 2 := by sorry

end ellipse_equivalence_l91_9114


namespace set_intersection_example_l91_9157

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {1, 2, 4}

theorem set_intersection_example : A ∩ B = {1, 2} := by
  sorry

end set_intersection_example_l91_9157


namespace equation_solution_l91_9115

theorem equation_solution :
  ∃ x : ℝ, (x + 1 = 5) ∧ (x = 4) := by
  sorry

end equation_solution_l91_9115


namespace solve_for_p_l91_9175

theorem solve_for_p (p q : ℚ) 
  (eq1 : 5 * p - 2 * q = 14) 
  (eq2 : 6 * p + q = 31) : 
  p = 76 / 17 := by
  sorry

end solve_for_p_l91_9175


namespace amc_distinct_scores_l91_9196

/-- Represents the scoring system for an exam -/
structure ScoringSystem where
  totalQuestions : Nat
  correctPoints : Nat
  incorrectPoints : Nat
  unansweredPoints : Nat

/-- Calculates the number of distinct possible scores for a given scoring system -/
def distinctScores (s : ScoringSystem) : Nat :=
  sorry

/-- The AMC exam scoring system -/
def amcScoring : ScoringSystem :=
  { totalQuestions := 30
  , correctPoints := 5
  , incorrectPoints := 0
  , unansweredPoints := 2 }

/-- Theorem stating that the number of distinct possible scores for the AMC exam is 145 -/
theorem amc_distinct_scores : distinctScores amcScoring = 145 := by
  sorry

end amc_distinct_scores_l91_9196


namespace water_bottles_problem_l91_9190

theorem water_bottles_problem (initial_bottles : ℕ) : 
  (3 * (initial_bottles - 3) = 21) → initial_bottles = 10 := by
  sorry

end water_bottles_problem_l91_9190


namespace no_solution_system_l91_9111

theorem no_solution_system :
  ¬∃ (x y : ℝ), 
    (x^3 + x + y + 1 = 0) ∧ 
    (y*x^2 + x + y = 0) ∧ 
    (y^2 + y - x^2 + 1 = 0) := by
  sorry

end no_solution_system_l91_9111


namespace prob_six_diff_tens_digits_l91_9177

/-- The probability of selecting 6 different integers between 10 and 99 (inclusive) 
    with different tens digits -/
def prob_diff_tens_digits : ℚ :=
  8000 / 5895

/-- The number of integers between 10 and 99, inclusive -/
def total_integers : ℕ := 90

/-- The number of possible tens digits -/
def num_tens_digits : ℕ := 9

/-- The number of integers to be selected -/
def num_selected : ℕ := 6

/-- The number of integers for each tens digit -/
def integers_per_tens : ℕ := 10

theorem prob_six_diff_tens_digits :
  prob_diff_tens_digits = 
    (Nat.choose num_tens_digits num_selected * integers_per_tens ^ num_selected) / 
    Nat.choose total_integers num_selected :=
sorry

end prob_six_diff_tens_digits_l91_9177


namespace ellipse_range_and_logical_conditions_l91_9197

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x y : ℝ, (x^2 / (m + 1) + y^2 / (3 - m) = 1) → 
  (∃ a b : ℝ, a > b ∧ a^2 - b^2 = 3 - m - (m + 1) ∧ 
  ∀ t : ℝ, x^2 / (m + 1) + y^2 / (3 - m) = 1 → 
  (x = 0 → y^2 ≤ a^2) ∧ (y = 0 → x^2 ≤ b^2))

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 2*m + 3 ≠ 0

theorem ellipse_range_and_logical_conditions (m : ℝ) :
  (p m ↔ -1 < m ∧ m < 1) ∧
  ((¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ 1 ≤ m ∧ m < 3) :=
sorry

end ellipse_range_and_logical_conditions_l91_9197


namespace remainder_eight_n_mod_seven_l91_9117

theorem remainder_eight_n_mod_seven (n : ℤ) (h : n % 4 = 3) : (8 * n) % 7 = 3 := by
  sorry

end remainder_eight_n_mod_seven_l91_9117


namespace cinema_selection_is_systematic_sampling_l91_9163

/-- Represents a cinema with a specific number of rows and seats per row. -/
structure Cinema where
  rows : Nat
  seatsPerRow : Nat

/-- Represents a sampling method. -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic
  | WithReplacement

/-- Represents the selection of seats in a cinema. -/
structure SeatSelection where
  cinema : Cinema
  seatNumber : Nat

/-- Determines if a sampling method is systematic based on the seat selection. -/
def isSystematicSampling (selection : SeatSelection) : Prop :=
  selection.cinema.rows > 0 ∧
  selection.cinema.seatsPerRow > 0 ∧
  selection.seatNumber < selection.cinema.seatsPerRow

/-- Theorem stating that the given seat selection is an example of systematic sampling. -/
theorem cinema_selection_is_systematic_sampling 
  (cinema : Cinema)
  (selection : SeatSelection)
  (h1 : cinema.rows = 50)
  (h2 : cinema.seatsPerRow = 60)
  (h3 : selection.seatNumber = 18)
  (h4 : selection.cinema = cinema) :
  isSystematicSampling selection ∧ 
  SamplingMethod.Systematic = SamplingMethod.Systematic :=
sorry


end cinema_selection_is_systematic_sampling_l91_9163


namespace bob_initial_pennies_l91_9176

theorem bob_initial_pennies :
  ∀ (a b : ℕ),
  (b + 2 = 4 * (a - 2)) →
  (b - 2 = 3 * (a + 2)) →
  b = 62 := by
  sorry

end bob_initial_pennies_l91_9176


namespace abc_is_246_l91_9120

/-- Represents a base-8 number with two digits --/
def BaseEight (a b : ℕ) : ℕ := 8 * a + b

/-- Converts a three-digit number to its decimal representation --/
def ToDecimal (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem abc_is_246 (A B C : ℕ) 
  (h1 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
  (h2 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h3 : A < 8 ∧ B < 8)
  (h4 : C < 6)
  (h5 : BaseEight A B + C = BaseEight C 2)
  (h6 : BaseEight A B + BaseEight B A = BaseEight C C) :
  ToDecimal A B C = 246 := by
  sorry

end abc_is_246_l91_9120


namespace number_of_office_workers_l91_9159

/-- Proves the number of office workers in company J --/
theorem number_of_office_workers :
  let factory_workers : ℕ := 15
  let factory_payroll : ℕ := 30000
  let office_payroll : ℕ := 75000
  let salary_difference : ℕ := 500
  let factory_avg_salary : ℕ := factory_payroll / factory_workers
  let office_avg_salary : ℕ := factory_avg_salary + salary_difference
  office_payroll / office_avg_salary = 30 := by
  sorry

end number_of_office_workers_l91_9159


namespace divisor_cube_eq_four_n_l91_9130

/-- The number of positive divisors of a positive integer -/
def d (n : ℕ+) : ℕ := sorry

/-- The theorem stating that the only positive integers n that satisfy d(n)^3 = 4n are 2, 128, and 2000 -/
theorem divisor_cube_eq_four_n : 
  ∀ n : ℕ+, d n ^ 3 = 4 * n ↔ n = 2 ∨ n = 128 ∨ n = 2000 := by sorry

end divisor_cube_eq_four_n_l91_9130


namespace binomial_square_constant_l91_9145

theorem binomial_square_constant (x : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 150*x + 5625 = (x + a)^2) := by
  sorry

end binomial_square_constant_l91_9145


namespace magnet_cost_is_three_l91_9128

/-- The cost of the magnet at the garage sale -/
def magnet_cost (stuffed_animal_cost : ℚ) : ℚ :=
  (2 * stuffed_animal_cost) / 4

/-- Theorem stating that the magnet cost $3 -/
theorem magnet_cost_is_three :
  magnet_cost 6 = 3 := by
  sorry

end magnet_cost_is_three_l91_9128


namespace multiply_and_simplify_l91_9185

theorem multiply_and_simplify (x : ℝ) : 
  (x^4 + 49*x^2 + 2401) * (x^2 - 49) = x^6 - 117649 := by
sorry

end multiply_and_simplify_l91_9185


namespace sum_of_squares_modulo_prime_sum_of_squares_zero_modulo_prime_1mod4_sum_of_squares_nonzero_modulo_prime_3mod4_l91_9169

theorem sum_of_squares_modulo_prime (p n : ℤ) (hp : Prime p) (hp5 : p > 5) :
  ∃ x y : ℤ, x % p ≠ 0 ∧ y % p ≠ 0 ∧ (x^2 + y^2) % p = n % p :=
sorry

theorem sum_of_squares_zero_modulo_prime_1mod4 (p : ℤ) (hp : Prime p) (hp1 : p % 4 = 1) :
  ∃ x y : ℤ, x % p ≠ 0 ∧ y % p ≠ 0 ∧ (x^2 + y^2) % p = 0 :=
sorry

theorem sum_of_squares_nonzero_modulo_prime_3mod4 (p : ℤ) (hp : Prime p) (hp3 : p % 4 = 3) :
  ∀ x y : ℤ, x % p ≠ 0 → y % p ≠ 0 → (x^2 + y^2) % p ≠ 0 :=
sorry

end sum_of_squares_modulo_prime_sum_of_squares_zero_modulo_prime_1mod4_sum_of_squares_nonzero_modulo_prime_3mod4_l91_9169


namespace root_in_interval_l91_9136

def f (x : ℝ) := x^3 + x^2 - 2*x - 2

theorem root_in_interval :
  f 1 = -2 →
  f 1.5 = 0.65 →
  f 1.25 = -0.984 →
  f 1.375 = -0.260 →
  f 1.4375 = 0.162 →
  f 1.40625 = -0.054 →
  ∃ x, x > 1.3 ∧ x < 1.5 ∧ f x = 0 :=
by sorry

end root_in_interval_l91_9136


namespace cube_opposite_face_l91_9168

-- Define a cube face
inductive Face : Type
| A | B | C | D | E | F

-- Define the property of being adjacent
def adjacent (x y : Face) : Prop := sorry

-- Define the property of sharing an edge
def sharesEdge (x y : Face) : Prop := sorry

-- Define the property of being opposite
def opposite (x y : Face) : Prop := sorry

-- Theorem statement
theorem cube_opposite_face :
  -- Conditions
  (sharesEdge Face.B Face.A) →
  (adjacent Face.C Face.B) →
  (¬ adjacent Face.C Face.A) →
  (sharesEdge Face.D Face.A) →
  (sharesEdge Face.D Face.F) →
  -- Conclusion
  (opposite Face.C Face.E) := by
sorry

end cube_opposite_face_l91_9168


namespace sock_selection_theorem_l91_9179

def total_socks : ℕ := 7
def blue_socks : ℕ := 2
def other_socks : ℕ := 5
def socks_to_choose : ℕ := 4

def valid_combinations : ℕ := 30

theorem sock_selection_theorem :
  (Nat.choose blue_socks 2 * Nat.choose other_socks 2) +
  (Nat.choose blue_socks 2 * Nat.choose other_socks 1) +
  (Nat.choose blue_socks 1 * Nat.choose other_socks 2) = valid_combinations :=
by sorry

end sock_selection_theorem_l91_9179


namespace expression_evaluation_l91_9194

theorem expression_evaluation :
  let f (x : ℝ) := 2 * x^2 + 3 * x - 4
  f 2 = 10 := by
sorry

end expression_evaluation_l91_9194


namespace proportion_theorem_l91_9165

theorem proportion_theorem (A B C p q r : ℝ) 
  (h1 : A / B = p) 
  (h2 : B / C = q) 
  (h3 : C / A = r) : 
  ∃ k : ℝ, k > 0 ∧ 
    A = k * (p^2 * q / r)^(1/3) ∧ 
    B = k * (q^2 * r / p)^(1/3) ∧ 
    C = k * (r^2 * p / q)^(1/3) := by
  sorry

end proportion_theorem_l91_9165


namespace composition_of_even_is_even_l91_9182

-- Define an even function
def EvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem composition_of_even_is_even (g : ℝ → ℝ) (h : EvenFunction g) :
  EvenFunction (g ∘ g) := by sorry

end composition_of_even_is_even_l91_9182


namespace divisibility_by_1989_l91_9109

theorem divisibility_by_1989 (n : ℕ) : ∃ k : ℤ, 
  13 * (-50)^n + 17 * 40^n - 30 = 1989 * k := by
  sorry

end divisibility_by_1989_l91_9109


namespace f_at_5_l91_9131

def f (x : ℝ) : ℝ := x^5 - 2*x^4 + x^3 + x^2 - x - 5

theorem f_at_5 : f 5 = 2015 := by
  sorry

end f_at_5_l91_9131


namespace partnership_investment_l91_9102

/-- Represents a partnership investment. -/
structure Partnership where
  a_investment : ℚ
  b_investment : ℚ
  c_investment : ℚ
  b_profit : ℚ
  a_profit : ℚ

/-- Theorem stating the relationship between investments and profits in a partnership. -/
theorem partnership_investment (p : Partnership) 
  (hb : p.b_investment = 11000)
  (hc : p.c_investment = 18000)
  (hbp : p.b_profit = 880)
  (hap : p.a_profit = 560) :
  p.a_investment = 7700 := by
  sorry


end partnership_investment_l91_9102


namespace first_month_sale_l91_9183

def average_sale : ℕ := 5500
def month2_sale : ℕ := 5927
def month3_sale : ℕ := 5855
def month4_sale : ℕ := 6230
def month5_sale : ℕ := 5562
def month6_sale : ℕ := 3991

theorem first_month_sale :
  let total_sale := 6 * average_sale
  let known_sales := month2_sale + month3_sale + month4_sale + month5_sale + month6_sale
  total_sale - known_sales = 5435 := by
sorry

end first_month_sale_l91_9183


namespace john_weight_on_bar_l91_9172

/-- The weight John can put on the bar given the weight bench capacity, safety margin, and his own weight -/
def weight_on_bar (bench_capacity : ℝ) (safety_margin : ℝ) (john_weight : ℝ) : ℝ :=
  bench_capacity * (1 - safety_margin) - john_weight

/-- Theorem stating the weight John can put on the bar -/
theorem john_weight_on_bar :
  weight_on_bar 1000 0.2 250 = 550 := by
  sorry

end john_weight_on_bar_l91_9172


namespace sum_of_roots_of_unity_l91_9121

def is_root_of_unity (z : ℂ) : Prop := ∃ n : ℕ, n > 0 ∧ z^n = 1

theorem sum_of_roots_of_unity (x y z : ℂ) :
  is_root_of_unity x ∧ is_root_of_unity y ∧ is_root_of_unity z →
  (is_root_of_unity (x + y + z) ↔ (x + y = 0 ∨ y + z = 0 ∨ z + x = 0)) :=
sorry

end sum_of_roots_of_unity_l91_9121


namespace max_diagonal_path_l91_9195

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a diagonal path in the rectangle -/
structure DiagonalPath where
  rectangle : Rectangle
  num_diagonals : ℕ

/-- Checks if a path is valid according to the problem constraints -/
def is_valid_path (path : DiagonalPath) : Prop :=
  path.num_diagonals > 0 ∧
  path.num_diagonals ≤ path.rectangle.width * path.rectangle.height / 2

/-- The main theorem stating the maximum number of diagonals in the path -/
theorem max_diagonal_path (rect : Rectangle) 
    (h1 : rect.width = 5) 
    (h2 : rect.height = 8) : 
  ∃ (path : DiagonalPath), 
    path.rectangle = rect ∧ 
    is_valid_path path ∧ 
    path.num_diagonals = 24 ∧
    ∀ (other_path : DiagonalPath), 
      other_path.rectangle = rect → 
      is_valid_path other_path → 
      other_path.num_diagonals ≤ path.num_diagonals :=
sorry

end max_diagonal_path_l91_9195


namespace quadratic_distinct_roots_l91_9143

theorem quadratic_distinct_roots (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 9 = 0 ∧ y^2 + m*y + 9 = 0) ↔ 
  (m < -6 ∨ m > 6) :=
sorry

end quadratic_distinct_roots_l91_9143


namespace min_students_for_duplicate_vote_l91_9149

theorem min_students_for_duplicate_vote (n : ℕ) (h : n = 10) :
  let combinations := n.choose 2
  ∃ k : ℕ, k > combinations ∧
    ∀ m : ℕ, m < k → ∃ f : Fin m → Fin n × Fin n,
      Function.Injective f ∧
      ∀ i : Fin m, (f i).1 < (f i).2 :=
by
  sorry

end min_students_for_duplicate_vote_l91_9149


namespace minimum_score_for_average_l91_9174

def exam_scores : List ℕ := [92, 85, 89, 93]
def desired_average : ℕ := 90
def num_exams : ℕ := 5

theorem minimum_score_for_average (scores : List ℕ) (avg : ℕ) (n : ℕ) :
  scores.length + 1 = n →
  (scores.sum + (n * avg - scores.sum)) / n = avg →
  n * avg - scores.sum = 91 :=
by sorry

#check minimum_score_for_average exam_scores desired_average num_exams

end minimum_score_for_average_l91_9174


namespace gcd_lcm_product_24_60_l91_9116

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by sorry

end gcd_lcm_product_24_60_l91_9116
