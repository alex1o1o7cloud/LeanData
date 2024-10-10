import Mathlib

namespace valid_pairs_eq_expected_l3821_382144

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

def valid_pairs : Finset (ℕ × ℕ) :=
  (divisors 660).product (divisors 72) |>.filter (λ (a, b) => a - b = 4)

theorem valid_pairs_eq_expected : valid_pairs = {(6, 2), (10, 6), (12, 8), (22, 18)} := by
  sorry

end valid_pairs_eq_expected_l3821_382144


namespace paper_mill_inspection_theorem_l3821_382176

/-- Represents the number of paper mills -/
def num_mills : ℕ := 5

/-- Probability of passing initial inspection -/
def prob_pass_initial : ℚ := 1/2

/-- Probability of passing after rectification -/
def prob_pass_rectification : ℚ := 4/5

/-- Probability of exactly two mills needing rectification -/
def prob_two_rectified : ℚ := 5/16

/-- Probability of at least one mill being shut down -/
def prob_at_least_one_shutdown : ℚ := 1 - (9/10)^5

/-- Average number of mills needing rectification -/
def avg_mills_rectified : ℚ := 5/2

theorem paper_mill_inspection_theorem :
  (prob_two_rectified = Nat.choose num_mills 2 * (1 - prob_pass_initial)^2 * prob_pass_initial^3) ∧
  (prob_at_least_one_shutdown = 1 - (1 - (1 - prob_pass_initial) * (1 - prob_pass_rectification))^num_mills) ∧
  (avg_mills_rectified = num_mills * (1 - prob_pass_initial)) :=
by sorry

end paper_mill_inspection_theorem_l3821_382176


namespace unique_function_exists_l3821_382140

-- Define the positive rationals
def PositiveRationals := {q : ℚ // q > 0}

-- Define the function type
def FunctionType := PositiveRationals → PositiveRationals

-- Define the conditions
def Condition1 (f : FunctionType) : Prop :=
  ∀ q : PositiveRationals, 0 < q.val ∧ q.val < 1/2 →
    f q = ⟨1 + (f ⟨q.val / (1 - 2*q.val), sorry⟩).val, sorry⟩

def Condition2 (f : FunctionType) : Prop :=
  ∀ q : PositiveRationals, 1 < q.val ∧ q.val ≤ 2 →
    f q = ⟨1 + (f ⟨q.val + 1, sorry⟩).val, sorry⟩

def Condition3 (f : FunctionType) : Prop :=
  ∀ q : PositiveRationals, (f q).val * (f ⟨1/q.val, sorry⟩).val = 1

-- State the theorem
theorem unique_function_exists :
  ∃! f : FunctionType, Condition1 f ∧ Condition2 f ∧ Condition3 f :=
sorry

end unique_function_exists_l3821_382140


namespace parallel_vectors_m_value_l3821_382196

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- The problem statement -/
theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2, m^2)
  parallel a b → m = 2 ∨ m = -2 := by
  sorry

end parallel_vectors_m_value_l3821_382196


namespace diophantine_equation_solutions_l3821_382110

theorem diophantine_equation_solutions :
  ∀ x y : ℕ, x^2 + (x + y)^2 = (x + 9)^2 ↔ (x = 0 ∧ y = 9) ∨ (x = 8 ∧ y = 7) ∨ (x = 20 ∧ y = 1) :=
by sorry

end diophantine_equation_solutions_l3821_382110


namespace f_bounded_by_four_l3821_382107

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| - |x - 3|

-- State the theorem
theorem f_bounded_by_four : ∀ x : ℝ, |f x| ≤ 4 := by sorry

end f_bounded_by_four_l3821_382107


namespace min_xy_l3821_382109

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 2 * x + 8 * y) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' * y' = 2 * x' + 8 * y' → x * y ≤ x' * y') →
  x = 16 ∧ y = 4 := by
sorry

end min_xy_l3821_382109


namespace chest_contents_l3821_382113

-- Define the types of coins
inductive CoinType
| Gold
| Silver
| Copper

-- Define the chests
structure Chest where
  inscription : CoinType → Prop
  content : CoinType

-- Define the problem setup
def chestProblem (c1 c2 c3 : Chest) : Prop :=
  -- All inscriptions are incorrect
  (¬c1.inscription c1.content) ∧
  (¬c2.inscription c2.content) ∧
  (¬c3.inscription c3.content) ∧
  -- Each chest contains a different type of coin
  (c1.content ≠ c2.content) ∧
  (c2.content ≠ c3.content) ∧
  (c3.content ≠ c1.content) ∧
  -- Inscriptions on the chests
  (c1.inscription = fun c => c = CoinType.Gold) ∧
  (c2.inscription = fun c => c = CoinType.Silver) ∧
  (c3.inscription = fun c => c = CoinType.Gold ∨ c = CoinType.Silver)

-- The theorem to prove
theorem chest_contents (c1 c2 c3 : Chest) 
  (h : chestProblem c1 c2 c3) : 
  c1.content = CoinType.Silver ∧ 
  c2.content = CoinType.Gold ∧ 
  c3.content = CoinType.Copper := by
  sorry

end chest_contents_l3821_382113


namespace max_distance_ellipse_point_l3821_382105

/-- The maximum distance between any point on the ellipse x²/36 + y²/27 = 1 and the point (3,0) is 9 -/
theorem max_distance_ellipse_point : 
  ∃ (M : ℝ × ℝ), 
    (M.1^2 / 36 + M.2^2 / 27 = 1) ∧ 
    (∀ (N : ℝ × ℝ), (N.1^2 / 36 + N.2^2 / 27 = 1) → 
      ((N.1 - 3)^2 + N.2^2)^(1/2) ≤ ((M.1 - 3)^2 + M.2^2)^(1/2)) ∧
    ((M.1 - 3)^2 + M.2^2)^(1/2) = 9 :=
by sorry

end max_distance_ellipse_point_l3821_382105


namespace wildflower_color_difference_l3821_382114

theorem wildflower_color_difference :
  let total_flowers : ℕ := 44
  let yellow_and_white : ℕ := 13
  let red_and_yellow : ℕ := 17
  let red_and_white : ℕ := 14
  let flowers_with_red : ℕ := red_and_yellow + red_and_white
  let flowers_with_white : ℕ := yellow_and_white + red_and_white
  flowers_with_red - flowers_with_white = 4 :=
by sorry

end wildflower_color_difference_l3821_382114


namespace smallest_number_divisible_by_5_and_24_l3821_382102

theorem smallest_number_divisible_by_5_and_24 : ∃ n : ℕ+, 
  (∀ m : ℕ+, 5 ∣ m ∧ 24 ∣ m → n ≤ m) ∧ 5 ∣ n ∧ 24 ∣ n := by
  sorry

end smallest_number_divisible_by_5_and_24_l3821_382102


namespace congruence_problem_l3821_382182

theorem congruence_problem (N : ℕ) (h1 : N > 1) 
  (h2 : 69 % N = 90 % N) (h3 : 90 % N = 125 % N) : 81 % N = 4 % N := by
  sorry

end congruence_problem_l3821_382182


namespace max_value_constraint_l3821_382130

theorem max_value_constraint (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (M : ℝ), M = 19 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 = 4 → x'^2 + 8*y' + 3 ≤ M :=
by sorry

end max_value_constraint_l3821_382130


namespace parabola_directrix_l3821_382189

/-- The directrix of the parabola y = -3x^2 + 6x - 5 is y = -23/12 -/
theorem parabola_directrix : ∀ x y : ℝ, 
  y = -3 * x^2 + 6 * x - 5 → 
  ∃ (k : ℝ), k = -23/12 ∧ (∀ p : ℝ × ℝ, p.1 = x ∧ p.2 = y → 
    (p.1 - 1)^2 + (p.2 - k)^2 = (p.2 + 2)^2 / 12) :=
by sorry

end parabola_directrix_l3821_382189


namespace fraction_simplification_l3821_382169

theorem fraction_simplification : 
  (3 + 6 - 12 + 24 + 48 - 96) / (6 + 12 - 24 + 48 + 96 - 192) = 1 / 2 := by
  sorry

end fraction_simplification_l3821_382169


namespace repeating_decimal_equals_fraction_l3821_382128

/-- The repeating decimal 0.868686... -/
def repeating_decimal : ℚ := 0.868686

/-- The fraction 86/99 -/
def fraction : ℚ := 86 / 99

/-- Theorem stating that the repeating decimal 0.868686... equals the fraction 86/99 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = fraction := by
  sorry

end repeating_decimal_equals_fraction_l3821_382128


namespace strawberry_harvest_l3821_382180

/-- Calculates the expected strawberry harvest for a rectangular garden. -/
theorem strawberry_harvest (length width plants_per_sqft avg_yield : ℕ) : 
  length = 10 →
  width = 12 →
  plants_per_sqft = 5 →
  avg_yield = 10 →
  length * width * plants_per_sqft * avg_yield = 6000 := by
  sorry

end strawberry_harvest_l3821_382180


namespace left_handed_classical_music_lovers_l3821_382179

theorem left_handed_classical_music_lovers (total : ℕ) (left_handed : ℕ) (classical_music : ℕ) (right_handed_non_classical : ℕ) 
  (h1 : total = 25)
  (h2 : left_handed = 10)
  (h3 : classical_music = 18)
  (h4 : right_handed_non_classical = 3)
  (h5 : left_handed + (total - left_handed) = total) :
  ∃ y : ℕ, y = 6 ∧ 
    y + (left_handed - y) + (classical_music - y) + right_handed_non_classical = total :=
by sorry

end left_handed_classical_music_lovers_l3821_382179


namespace cone_height_from_sphere_l3821_382121

/-- The height of a cone formed by melting and reshaping a sphere -/
theorem cone_height_from_sphere (r_sphere : ℝ) (r_cone h_cone : ℝ) : 
  r_sphere = 5 * 3^2 →
  (2 * π * r_cone * (3 * r_cone)) = 3 * (π * r_cone^2) →
  (4/3) * π * r_sphere^3 = (1/3) * π * r_cone^2 * h_cone →
  h_cone = 20 := by
  sorry

#check cone_height_from_sphere

end cone_height_from_sphere_l3821_382121


namespace purchase_total_cost_l3821_382183

/-- The cost of a single sandwich in dollars -/
def sandwich_cost : ℚ := 2.44

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 2

/-- The cost of a single soda in dollars -/
def soda_cost : ℚ := 0.87

/-- The number of sodas purchased -/
def num_sodas : ℕ := 4

/-- The total cost of the purchase in dollars -/
def total_cost : ℚ := 8.36

theorem purchase_total_cost : 
  (num_sandwiches : ℚ) * sandwich_cost + (num_sodas : ℚ) * soda_cost = total_cost := by
  sorry

end purchase_total_cost_l3821_382183


namespace problem_solution_l3821_382129

theorem problem_solution : ∃ x : ℝ, (0.65 * x = 0.20 * 747.50) ∧ (x = 230) := by
  sorry

end problem_solution_l3821_382129


namespace robot_capacity_theorem_l3821_382104

/-- Represents the material handling capacity of robots A and B --/
structure RobotCapacity where
  A : ℝ
  B : ℝ

/-- The conditions given in the problem --/
def satisfiesConditions (c : RobotCapacity) : Prop :=
  c.A = c.B + 30 ∧ 1000 / c.A = 800 / c.B

/-- The theorem to prove --/
theorem robot_capacity_theorem :
  ∃ c : RobotCapacity, satisfiesConditions c ∧ c.A = 150 ∧ c.B = 120 := by
  sorry

end robot_capacity_theorem_l3821_382104


namespace total_distance_theorem_l3821_382153

/-- Calculates the total distance covered by two cyclists in a week -/
def total_distance_in_week (
  onur_speed : ℝ
  ) (hanil_speed : ℝ
  ) (onur_hours : ℝ
  ) (onur_rest_day : ℕ
  ) (hanil_rest_day : ℕ
  ) (hanil_extra_distance : ℝ
  ) (days_in_week : ℕ
  ) : ℝ :=
  let onur_daily_distance := onur_speed * onur_hours
  let hanil_daily_distance := onur_daily_distance + hanil_extra_distance
  let onur_biking_days := days_in_week - (days_in_week / onur_rest_day)
  let hanil_biking_days := days_in_week - (days_in_week / hanil_rest_day)
  let onur_total_distance := onur_daily_distance * onur_biking_days
  let hanil_total_distance := hanil_daily_distance * hanil_biking_days
  onur_total_distance + hanil_total_distance

/-- Theorem stating the total distance covered by Onur and Hanil in a week -/
theorem total_distance_theorem :
  total_distance_in_week 35 45 7 3 4 40 7 = 2935 := by
  sorry

end total_distance_theorem_l3821_382153


namespace max_profit_allocation_l3821_382166

/-- Represents the profit function for Project A -/
def p (a : ℝ) (t : ℝ) : ℝ := a * t^3 + 21 * t

/-- Represents the profit function for Project B -/
def g (a : ℝ) (b : ℝ) (t : ℝ) : ℝ := -2 * a * (t - b)^2

/-- Represents the total profit function -/
def f (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := p a x + g a b (200 - x)

/-- Theorem stating the maximum profit and optimal investment allocation -/
theorem max_profit_allocation (a b : ℝ) :
  (∀ t, p a t = -1/60 * t^3 + 21 * t) →
  (∀ t, g a b t = 1/30 * (t - 110)^2) →
  (p a 30 = 180) →
  (g a b 170 = 120) →
  (b < 200) →
  (∃ x₀, x₀ ∈ Set.Icc 10 190 ∧ 
    f a b x₀ = 453.6 ∧
    (∀ x, x ∈ Set.Icc 10 190 → f a b x ≤ f a b x₀) ∧
    x₀ = 18) := by
  sorry


end max_profit_allocation_l3821_382166


namespace division_4863_by_97_l3821_382136

theorem division_4863_by_97 : ∃ (q r : ℤ), 4863 = 97 * q + r ∧ 0 ≤ r ∧ r < 97 ∧ q = 50 ∧ r = 40 := by
  sorry

end division_4863_by_97_l3821_382136


namespace andrew_flooring_theorem_l3821_382163

def andrew_flooring_problem (bedroom living_room kitchen guest_bedroom hallway leftover : ℕ) : Prop :=
  let total_used := bedroom + living_room + kitchen + guest_bedroom + 2 * hallway
  let total_original := total_used + leftover
  let ruined_per_bedroom := total_original - total_used
  (bedroom = 8) ∧
  (living_room = 20) ∧
  (kitchen = 11) ∧
  (guest_bedroom = bedroom - 2) ∧
  (hallway = 4) ∧
  (leftover = 6) ∧
  (ruined_per_bedroom = 6)

theorem andrew_flooring_theorem :
  ∀ bedroom living_room kitchen guest_bedroom hallway leftover,
  andrew_flooring_problem bedroom living_room kitchen guest_bedroom hallway leftover :=
by
  sorry

end andrew_flooring_theorem_l3821_382163


namespace complete_square_constant_l3821_382195

theorem complete_square_constant (x : ℝ) :
  ∃ (a h k : ℝ), x^2 - 8*x = a*(x - h)^2 + k ∧ k = -16 := by
  sorry

end complete_square_constant_l3821_382195


namespace oranges_picked_total_l3821_382158

theorem oranges_picked_total (mary_oranges jason_oranges : ℕ) 
  (h1 : mary_oranges = 122) 
  (h2 : jason_oranges = 105) : 
  mary_oranges + jason_oranges = 227 := by
  sorry

end oranges_picked_total_l3821_382158


namespace consecutive_integers_sqrt_seven_l3821_382198

theorem consecutive_integers_sqrt_seven (a b : ℤ) : 
  (b = a + 1) →  -- a and b are consecutive integers
  (a < Real.sqrt 7) →  -- a < √7
  (Real.sqrt 7 < b) →  -- √7 < b
  a + b = 5 := by
sorry

end consecutive_integers_sqrt_seven_l3821_382198


namespace power_sum_inequality_l3821_382167

theorem power_sum_inequality (a b c : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (sum_eq_three : a + b + c = 3) : 
  a^a + b^b + c^c ≥ 3 := by
  sorry

end power_sum_inequality_l3821_382167


namespace total_balls_bought_l3821_382191

/-- Represents the amount of money Mr. Li has -/
def total_money : ℚ := 1

/-- The cost of one plastic ball -/
def plastic_ball_cost : ℚ := 1 / 60

/-- The cost of one glass ball -/
def glass_ball_cost : ℚ := 1 / 36

/-- The cost of one wooden ball -/
def wooden_ball_cost : ℚ := 1 / 45

/-- The number of plastic balls Mr. Li buys -/
def plastic_balls_bought : ℕ := 10

/-- The number of glass balls Mr. Li buys -/
def glass_balls_bought : ℕ := 10

/-- Theorem stating the total number of balls Mr. Li buys -/
theorem total_balls_bought : 
  ∃ (wooden_balls : ℕ), 
    (plastic_balls_bought * plastic_ball_cost + 
     glass_balls_bought * glass_ball_cost + 
     wooden_balls * wooden_ball_cost = total_money) ∧
    (plastic_balls_bought + glass_balls_bought + wooden_balls = 45) :=
by sorry

end total_balls_bought_l3821_382191


namespace evaluate_expression_l3821_382159

/-- Given x = 4 and z = -2, prove that z(z - 4x) = 36 -/
theorem evaluate_expression (x z : ℝ) (hx : x = 4) (hz : z = -2) : z * (z - 4 * x) = 36 := by
  sorry

end evaluate_expression_l3821_382159


namespace h_not_prime_l3821_382145

def h (n : ℕ+) : ℤ := n^4 - 380 * n^2 + 600

theorem h_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (h n)) := by
  sorry

end h_not_prime_l3821_382145


namespace sequence_divisibility_l3821_382122

def u : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 2 * u (n + 1) - 3 * u n

def v (a b c : ℤ) : ℕ → ℤ
  | 0 => a
  | 1 => b
  | 2 => c
  | (n + 3) => v a b c (n + 2) - 3 * v a b c (n + 1) + 27 * v a b c n

theorem sequence_divisibility (a b c : ℤ) :
  (∃ N : ℕ, ∀ n > N, ∃ k : ℤ, v a b c n = k * u n) →
  3 * a = 2 * b + c := by
  sorry

end sequence_divisibility_l3821_382122


namespace eight_by_eight_tiling_ten_by_ten_no_tiling_l3821_382177

-- Define a chessboard
structure Chessboard :=
  (size : Nat)
  (total_squares : Nat)
  (black_squares : Nat)
  (white_squares : Nat)

-- Define a pedestal shape
structure Pedestal :=
  (squares_covered : Nat)

-- Define the tiling property
def can_tile (b : Chessboard) (p : Pedestal) : Prop :=
  b.total_squares % p.squares_covered = 0

-- Define the color coverage property for 10x10 board
def color_coverage_property (b : Chessboard) (p : Pedestal) : Prop :=
  ∃ (k : Nat), 3 * k + k = b.black_squares ∧ 3 * k + k = b.white_squares

-- Theorem for 8x8 chessboard
theorem eight_by_eight_tiling :
  ∀ (b : Chessboard) (p : Pedestal),
    b.size = 8 →
    b.total_squares = 64 →
    p.squares_covered = 4 →
    can_tile b p :=
sorry

-- Theorem for 10x10 chessboard
theorem ten_by_ten_no_tiling :
  ∀ (b : Chessboard) (p : Pedestal),
    b.size = 10 →
    b.total_squares = 100 →
    b.black_squares = 50 →
    b.white_squares = 50 →
    p.squares_covered = 4 →
    ¬(can_tile b p ∧ color_coverage_property b p) :=
sorry

end eight_by_eight_tiling_ten_by_ten_no_tiling_l3821_382177


namespace seven_digit_number_exists_l3821_382101

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem seven_digit_number_exists : ∃ n : ℕ, 
  (1000000 ≤ n ∧ n < 9000000) ∧ 
  (sum_of_digits n = 53) ∧ 
  (n % 13 = 0) ∧ 
  (n = 8999990) :=
sorry

end seven_digit_number_exists_l3821_382101


namespace sum_of_reciprocals_of_roots_l3821_382116

theorem sum_of_reciprocals_of_roots (p q : ℝ) : 
  p^2 - 20*p + 9 = 0 → q^2 - 20*q + 9 = 0 → p ≠ q → (1/p + 1/q) = 20/9 := by
  sorry

end sum_of_reciprocals_of_roots_l3821_382116


namespace triangle_angle_range_l3821_382193

theorem triangle_angle_range (a b : ℝ) (h_a : a = 2) (h_b : b = 2 * Real.sqrt 2) :
  ∃ (A : ℝ), 0 < A ∧ A ≤ π / 4 ∧
  ∀ (c : ℝ), c > 0 → a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A :=
by sorry

end triangle_angle_range_l3821_382193


namespace chess_team_arrangement_l3821_382147

/-- The number of boys on the chess team -/
def num_boys : ℕ := 3

/-- The number of girls on the chess team -/
def num_girls : ℕ := 2

/-- The total number of students on the chess team -/
def total_students : ℕ := num_boys + num_girls

/-- The number of ways to arrange the team with girls at the ends and boys in the middle -/
def num_arrangements : ℕ := (Nat.factorial num_girls) * (Nat.factorial num_boys)

theorem chess_team_arrangement :
  num_arrangements = 12 :=
sorry

end chess_team_arrangement_l3821_382147


namespace arithmetic_geometric_sequence_a4_l3821_382126

/-- An arithmetic-geometric sequence -/
def ArithmeticGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem arithmetic_geometric_sequence_a4
  (a : ℕ → ℝ)
  (h_seq : ArithmeticGeometricSequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  a 4 = 1 := by
sorry

end arithmetic_geometric_sequence_a4_l3821_382126


namespace units_digit_of_cube_minus_square_l3821_382184

def n : ℕ := 9867

theorem units_digit_of_cube_minus_square :
  (n^3 - n^2) % 10 = 4 := by sorry

end units_digit_of_cube_minus_square_l3821_382184


namespace continuous_piecewise_sum_l3821_382178

/-- A piecewise function f(x) defined on the real line. -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then a * x + 6
  else if x ≥ -3 then x - 7
  else 3 * x - b

/-- The function f is continuous on the real line. -/
def is_continuous (a b : ℝ) : Prop :=
  Continuous (f a b)

/-- If f is continuous, then a + b = -7/3. -/
theorem continuous_piecewise_sum (a b : ℝ) :
  is_continuous a b → a + b = -7/3 := by
  sorry

end continuous_piecewise_sum_l3821_382178


namespace cylinder_lateral_surface_area_l3821_382188

/-- A cylinder with a square axial cross-section of area 5 has a lateral surface area of 5π. -/
theorem cylinder_lateral_surface_area (h : ℝ) (r : ℝ) : 
  h * h = 5 → 2 * r = h → 2 * π * r * h = 5 * π := by
  sorry

end cylinder_lateral_surface_area_l3821_382188


namespace pizza_slices_remaining_l3821_382100

theorem pizza_slices_remaining (initial_slices : ℕ) 
  (breakfast_slices : ℕ) (lunch_slices : ℕ) (snack_slices : ℕ) (dinner_slices : ℕ) :
  initial_slices = 15 →
  breakfast_slices = 4 →
  lunch_slices = 2 →
  snack_slices = 2 →
  dinner_slices = 5 →
  initial_slices - (breakfast_slices + lunch_slices + snack_slices + dinner_slices) = 2 :=
by sorry

end pizza_slices_remaining_l3821_382100


namespace range_of_expression_l3821_382190

theorem range_of_expression (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2) :
  -π/6 < 2*α - β/2 ∧ 2*α - β/2 < π :=
by sorry

end range_of_expression_l3821_382190


namespace sunshine_cost_per_mile_is_correct_l3821_382170

/-- The cost per mile for Sunshine Car Rentals -/
def sunshine_cost_per_mile : ℝ := 0.18

/-- The daily rate for Sunshine Car Rentals -/
def sunshine_daily_rate : ℝ := 17.99

/-- The daily rate for City Rentals -/
def city_daily_rate : ℝ := 18.95

/-- The cost per mile for City Rentals -/
def city_cost_per_mile : ℝ := 0.16

/-- The number of miles at which the costs are equal -/
def equal_cost_miles : ℝ := 48.0

theorem sunshine_cost_per_mile_is_correct :
  sunshine_daily_rate + equal_cost_miles * sunshine_cost_per_mile =
  city_daily_rate + equal_cost_miles * city_cost_per_mile :=
by sorry

end sunshine_cost_per_mile_is_correct_l3821_382170


namespace inequality_preserved_under_subtraction_l3821_382142

theorem inequality_preserved_under_subtraction (a b c : ℝ) : 
  a < b → a - 2*c < b - 2*c := by
  sorry

end inequality_preserved_under_subtraction_l3821_382142


namespace library_seating_l3821_382168

theorem library_seating (x : ℕ) : 
  (∃ (y : ℕ), x + y = 16) →  -- Total number of chairs and stools is 16
  (4 * x + 3 * (16 - x) = 60) -- Equation representing the situation
  :=
by
  sorry

end library_seating_l3821_382168


namespace quadratic_inequality_range_l3821_382171

theorem quadratic_inequality_range :
  ∃ a : ℝ, a ∈ Set.Icc 1 3 ∧ ∀ x : ℝ, a * x^2 + (a - 2) * x - 2 > 0 →
    x < -1 ∨ x > 2 :=
sorry

end quadratic_inequality_range_l3821_382171


namespace fraction_seven_twentynine_repetend_l3821_382125

/-- The repetend of a rational number is the repeating part of its decimal representation. -/
def repetend (n d : ℕ) : ℕ := sorry

/-- A number is a valid repetend for a fraction if it repeats infinitely in the decimal representation. -/
def is_valid_repetend (r n d : ℕ) : Prop := sorry

theorem fraction_seven_twentynine_repetend :
  let r := 241379
  is_valid_repetend r 7 29 ∧ repetend 7 29 = r :=
sorry

end fraction_seven_twentynine_repetend_l3821_382125


namespace money_division_l3821_382146

theorem money_division (a b c : ℚ) : 
  a = (1/3 : ℚ) * b → 
  b = (1/4 : ℚ) * c → 
  b = 270 → 
  a + b + c = 1440 :=
by sorry

end money_division_l3821_382146


namespace imaginary_part_of_z_l3821_382181

theorem imaginary_part_of_z (z : ℂ) : z = (2 - Complex.I) * Complex.I → z.im = 2 := by
  sorry

end imaginary_part_of_z_l3821_382181


namespace solve_parking_ticket_problem_l3821_382106

def parking_ticket_problem (first_two_ticket_cost : ℚ) (third_ticket_fraction : ℚ) (james_remaining_money : ℚ) : Prop :=
  let total_cost := 2 * first_two_ticket_cost + third_ticket_fraction * first_two_ticket_cost
  let james_paid := total_cost - james_remaining_money
  let roommate_paid := total_cost - james_paid
  (roommate_paid / total_cost) = 13 / 14

theorem solve_parking_ticket_problem :
  parking_ticket_problem 150 (1/3) 325 := by
  sorry

end solve_parking_ticket_problem_l3821_382106


namespace log_expression_equals_negative_one_l3821_382154

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_negative_one :
  log10 (5 / 2) + 2 * log10 2 - (1 / 2)⁻¹ = -1 := by
  sorry

end log_expression_equals_negative_one_l3821_382154


namespace cases_needed_l3821_382152

theorem cases_needed (total_boxes : Nat) (boxes_per_case : Nat) : 
  total_boxes = 20 → boxes_per_case = 4 → total_boxes / boxes_per_case = 5 := by
  sorry

end cases_needed_l3821_382152


namespace smallest_with_144_divisors_and_10_consecutive_l3821_382162

/-- A function that returns the number of positive divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has 10 consecutive divisors -/
def has_10_consecutive_divisors (n : ℕ) : Prop := sorry

/-- The theorem stating that 110880 is the smallest number satisfying the conditions -/
theorem smallest_with_144_divisors_and_10_consecutive : 
  num_divisors 110880 = 144 ∧ 
  has_10_consecutive_divisors 110880 ∧ 
  ∀ m : ℕ, m < 110880 → (num_divisors m ≠ 144 ∨ ¬has_10_consecutive_divisors m) :=
sorry

end smallest_with_144_divisors_and_10_consecutive_l3821_382162


namespace complex_magnitude_eval_l3821_382161

theorem complex_magnitude_eval (ω : ℂ) (h : ω = 7 + 3 * I) :
  Complex.abs (ω^2 + 8*ω + 85) = Real.sqrt 30277 := by
  sorry

end complex_magnitude_eval_l3821_382161


namespace geometric_sequence_sum_l3821_382192

/-- Sum of a geometric sequence -/
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₀ * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a₀ : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 8
  geometric_sum a₀ r n = 3280/6561 := by
sorry

end geometric_sequence_sum_l3821_382192


namespace investment_comparison_l3821_382112

/-- Represents the value of an investment over time -/
structure Investment where
  initial : ℝ
  year1_change : ℝ
  year2_change : ℝ

/-- Calculates the final value of an investment after two years -/
def final_value (inv : Investment) : ℝ :=
  inv.initial * (1 + inv.year1_change) * (1 + inv.year2_change)

/-- The problem setup -/
def problem_setup : (Investment × Investment × Investment) :=
  ({ initial := 150, year1_change := 0.1, year2_change := 0.15 },
   { initial := 150, year1_change := -0.3, year2_change := 0.5 },
   { initial := 150, year1_change := 0, year2_change := -0.1 })

theorem investment_comparison :
  let (a, b, c) := problem_setup
  final_value a > final_value b ∧ final_value b > final_value c :=
by sorry

end investment_comparison_l3821_382112


namespace farmer_purchase_problem_l3821_382165

theorem farmer_purchase_problem :
  ∃ (p ch : ℕ), 
    p > 0 ∧ 
    ch > 0 ∧ 
    30 * p + 24 * ch = 1200 ∧ 
    p = 4 ∧ 
    ch = 45 := by
  sorry

end farmer_purchase_problem_l3821_382165


namespace beka_jackson_flight_difference_l3821_382103

/-- 
Given that Beka flew 873 miles and Jackson flew 563 miles,
prove that Beka flew 310 miles more than Jackson.
-/
theorem beka_jackson_flight_difference :
  let beka_miles : ℕ := 873
  let jackson_miles : ℕ := 563
  beka_miles - jackson_miles = 310 := by sorry

end beka_jackson_flight_difference_l3821_382103


namespace cube_shape_product_l3821_382127

/-- Represents a 3D shape constructed from identical cubes. -/
structure CubeShape where
  /-- The number of cubes in the shape. -/
  num_cubes : ℕ
  /-- Predicate that returns true if the shape satisfies the given views. -/
  satisfies_views : Bool

/-- The minimum number of cubes that can form the shape satisfying the given views. -/
def min_cubes : ℕ := 8

/-- The maximum number of cubes that can form the shape satisfying the given views. -/
def max_cubes : ℕ := 16

/-- Theorem stating that the product of the maximum and minimum number of cubes is 128. -/
theorem cube_shape_product :
  min_cubes * max_cubes = 128 ∧
  ∀ shape : CubeShape, shape.satisfies_views →
    min_cubes ≤ shape.num_cubes ∧ shape.num_cubes ≤ max_cubes :=
by sorry

end cube_shape_product_l3821_382127


namespace not_divisible_by_three_and_four_l3821_382156

theorem not_divisible_by_three_and_four (n : ℤ) : 
  ¬(∃ k : ℤ, n^2 + 1 = 3 * k) ∧ ¬(∃ m : ℤ, n^2 + 1 = 4 * m) := by
  sorry

end not_divisible_by_three_and_four_l3821_382156


namespace number_of_lineups_l3821_382187

/-- Represents the number of players in the team -/
def total_players : ℕ := 15

/-- Represents the number of players in the starting lineup -/
def lineup_size : ℕ := 4

/-- Represents the number of players that must be in the starting lineup -/
def fixed_players : ℕ := 3

/-- Calculates the number of possible starting lineups -/
def possible_lineups : ℕ := Nat.choose (total_players - fixed_players) (lineup_size - fixed_players)

/-- Theorem stating that the number of possible starting lineups is 12 -/
theorem number_of_lineups : possible_lineups = 12 := by sorry

end number_of_lineups_l3821_382187


namespace trig_equation_solution_l3821_382186

open Real

theorem trig_equation_solution (x : ℝ) : 
  (sin (x + 15 * π / 180) + sin (x + 45 * π / 180) + sin (x + 75 * π / 180) = 
   sin (15 * π / 180) + sin (45 * π / 180) + sin (75 * π / 180)) ↔ 
  (∃ k : ℤ, x = k * 2 * π ∨ x = π / 2 + k * 2 * π) :=
sorry

end trig_equation_solution_l3821_382186


namespace seedling_problem_l3821_382123

/-- Represents the unit price and quantity of seedlings --/
structure Seedling where
  price : ℚ
  quantity : ℚ

/-- Represents the total cost of a purchase --/
def totalCost (a b : Seedling) : ℚ :=
  a.price * a.quantity + b.price * b.quantity

/-- Represents the discounted price of a seedling --/
def discountedPrice (s : Seedling) (discount : ℚ) : ℚ :=
  s.price * (1 - discount)

theorem seedling_problem :
  ∃ (a b : Seedling),
    (totalCost ⟨a.price, 15⟩ ⟨b.price, 5⟩ = 190) ∧
    (totalCost ⟨a.price, 25⟩ ⟨b.price, 15⟩ = 370) ∧
    (a.price = 10) ∧
    (b.price = 8) ∧
    (∀ m : ℚ,
      m ≤ 100 ∧
      (discountedPrice a 0.1) * m + (discountedPrice b 0.1) * (100 - m) ≤ 828 →
      m ≤ 60) ∧
    (∃ m : ℚ,
      m = 60 ∧
      (discountedPrice a 0.1) * m + (discountedPrice b 0.1) * (100 - m) ≤ 828) :=
by
  sorry


end seedling_problem_l3821_382123


namespace unique_solution_for_equation_l3821_382173

theorem unique_solution_for_equation : 
  ∃! (p n : ℕ), 
    n > 0 ∧ 
    Nat.Prime p ∧ 
    17^n * 2^(n^2) - p = (2^(n^2 + 3) + 2^(n^2) - 1) * n^2 ∧ 
    p = 17 ∧ 
    n = 1 := by
  sorry

end unique_solution_for_equation_l3821_382173


namespace probability_five_blue_marbles_l3821_382117

def total_marbles : ℕ := 12
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 4
def total_draws : ℕ := 8
def blue_draws : ℕ := 5

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

def binomial_coefficient (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_five_blue_marbles :
  (binomial_coefficient total_draws blue_draws : ℚ) * 
  (probability_blue ^ blue_draws) * 
  (probability_red ^ (total_draws - blue_draws)) = 1792 / 6561 := by
sorry

end probability_five_blue_marbles_l3821_382117


namespace min_value_sum_of_roots_l3821_382175

theorem min_value_sum_of_roots (x : ℝ) :
  let y := Real.sqrt (x^2 - 2*x + 2) + Real.sqrt (x^2 - 10*x + 34)
  y ≥ 4 * Real.sqrt 2 ∧ ∃ x₀ : ℝ, Real.sqrt (x₀^2 - 2*x₀ + 2) + Real.sqrt (x₀^2 - 10*x₀ + 34) = 4 * Real.sqrt 2 :=
by sorry

end min_value_sum_of_roots_l3821_382175


namespace f_properties_l3821_382149

-- Define the function f
def f (x : ℝ) : ℝ := -x - x^3

-- State the theorem
theorem f_properties (x₁ x₂ : ℝ) (h : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧ (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) := by
  sorry

end f_properties_l3821_382149


namespace smaller_integer_problem_l3821_382119

theorem smaller_integer_problem (x y : ℤ) 
  (sum_eq : x + y = 30)
  (relation : 2 * y = 5 * x - 10) :
  x = 10 ∧ x ≤ y :=
by sorry

end smaller_integer_problem_l3821_382119


namespace quadratic_equation_solution_l3821_382151

theorem quadratic_equation_solution (a b c x₁ x₂ y₁ y₂ : ℝ) 
  (hb : b ≠ 0)
  (h1 : x₁^2 + a*x₂^2 = b)
  (h2 : x₂*y₁ - x₁*y₂ = a)
  (h3 : x₁*y₁ + a*x₂*y₂ = c) :
  y₁^2 + a*y₂^2 = (a^3 + c^2) / b := by
  sorry

end quadratic_equation_solution_l3821_382151


namespace c_range_l3821_382174

theorem c_range (c : ℝ) (h_c_pos : c > 0) : 
  (((∀ x y : ℝ, x < y → c^x > c^y) ↔ ¬(∀ x : ℝ, x + c > 0)) ∧ 
   ((∀ x : ℝ, x + c > 0) ↔ ¬(∀ x y : ℝ, x < y → c^x > c^y))) → 
  (c > 0 ∧ c ≠ 1) :=
by sorry

end c_range_l3821_382174


namespace geometric_sequence_17th_term_l3821_382164

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_17th_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_5th : a 5 = 9)
  (h_13th : a 13 = 1152) :
  a 17 = 36864 := by
sorry

end geometric_sequence_17th_term_l3821_382164


namespace completing_square_equivalence_l3821_382118

theorem completing_square_equivalence :
  ∀ x : ℝ, (x^2 - 2*x - 3 = 0) ↔ ((x - 1)^2 = 4) := by
sorry

end completing_square_equivalence_l3821_382118


namespace cow_spots_l3821_382199

theorem cow_spots (left_spots : ℕ) : 
  (left_spots + (3 * left_spots + 7) = 71) → left_spots = 16 := by
  sorry

end cow_spots_l3821_382199


namespace cylinder_volume_from_lateral_surface_l3821_382160

/-- Given a cylinder whose lateral surface unfolds to a square with side length 2,
    prove that its volume is 2/π. -/
theorem cylinder_volume_from_lateral_surface (r h : ℝ) : 
  (2 * π * r = 2) → (h = 2) → (π * r^2 * h = 2/π) := by sorry

end cylinder_volume_from_lateral_surface_l3821_382160


namespace hilt_family_fitness_l3821_382150

-- Define conversion rates
def yards_per_mile : ℝ := 1760
def miles_per_km : ℝ := 0.621371

-- Define Mrs. Hilt's activities
def mrs_hilt_running : List ℝ := [3, 2, 7]
def mrs_hilt_swimming : List ℝ := [1760, 0, 1000]
def mrs_hilt_biking : List ℝ := [0, 6, 3, 10]

-- Define Mr. Hilt's activities
def mr_hilt_biking : List ℝ := [5, 8]
def mr_hilt_running : List ℝ := [4]
def mr_hilt_swimming : List ℝ := [2000]

-- Theorem statement
theorem hilt_family_fitness :
  (mrs_hilt_running.sum = 12) ∧
  (mrs_hilt_swimming.sum / yards_per_mile + 1000 / yards_per_mile * miles_per_km = 2854 / yards_per_mile) ∧
  (mrs_hilt_biking.sum = 19) ∧
  (mr_hilt_biking.sum = 13) ∧
  (mr_hilt_running.sum = 4) ∧
  (mr_hilt_swimming.sum = 2000) :=
by sorry

end hilt_family_fitness_l3821_382150


namespace jorge_corn_yield_ratio_l3821_382197

/-- Represents the yield ratio problem for Jorge's corn fields --/
theorem jorge_corn_yield_ratio :
  let total_acres : ℚ := 60
  let good_soil_yield : ℚ := 400
  let clay_rich_proportion : ℚ := 1/3
  let total_yield : ℚ := 20000
  let clay_rich_acres : ℚ := total_acres * clay_rich_proportion
  let good_soil_acres : ℚ := total_acres - clay_rich_acres
  let good_soil_total_yield : ℚ := good_soil_acres * good_soil_yield
  let clay_rich_total_yield : ℚ := total_yield - good_soil_total_yield
  let clay_rich_yield : ℚ := clay_rich_total_yield / clay_rich_acres
  clay_rich_yield / good_soil_yield = 1/2 :=
by sorry


end jorge_corn_yield_ratio_l3821_382197


namespace triangle_perimeter_l3821_382120

/-- Given a right-angled triangle with hypotenuse 5000 km and one other side 4000 km,
    the sum of all sides is 12000 km. -/
theorem triangle_perimeter (a b c : ℝ) (h1 : a = 5000) (h2 : b = 4000) 
    (h3 : a^2 = b^2 + c^2) : a + b + c = 12000 := by
  sorry

end triangle_perimeter_l3821_382120


namespace ball_reaches_top_left_corner_l3821_382157

/-- Represents a rectangular billiard table -/
structure BilliardTable where
  width : ℕ
  length : ℕ

/-- Represents the path of a ball on the billiard table -/
def ball_path (table : BilliardTable) : ℕ :=
  Nat.lcm table.width table.length

/-- Theorem: A ball launched at 45° from the bottom-left corner of a 26x1965 table
    will reach the top-left corner after traveling the LCM of 26 and 1965 in both directions -/
theorem ball_reaches_top_left_corner (table : BilliardTable) 
    (h1 : table.width = 26) (h2 : table.length = 1965) :
    ball_path table = 50990 ∧ 
    50990 % table.width = 0 ∧ 
    50990 % table.length = 0 := by
  sorry

#eval ball_path { width := 26, length := 1965 }

end ball_reaches_top_left_corner_l3821_382157


namespace largest_t_value_for_60_degrees_l3821_382133

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 12*t + 50

-- Define the theorem
theorem largest_t_value_for_60_degrees :
  let t := 6 + Real.sqrt 46
  (∀ s ≥ 0, temperature s = 60 → s ≤ t) ∧ temperature t = 60 := by
  sorry

end largest_t_value_for_60_degrees_l3821_382133


namespace m_divided_by_8_l3821_382148

theorem m_divided_by_8 (m : ℕ) (h : m = 16^500) : m / 8 = 2^1997 := by
  sorry

end m_divided_by_8_l3821_382148


namespace student_count_l3821_382135

theorem student_count (cost_per_student : ℕ) (total_cost : ℕ) (h1 : cost_per_student = 8) (h2 : total_cost = 184) :
  total_cost / cost_per_student = 23 :=
by
  sorry

end student_count_l3821_382135


namespace network_connections_l3821_382143

/-- Given a network of switches where each switch is connected to exactly
    four others, this function calculates the total number of connections. -/
def calculate_connections (num_switches : ℕ) : ℕ :=
  (num_switches * 4) / 2

/-- Theorem stating that in a network of 30 switches, where each switch
    is directly connected to exactly 4 other switches, the total number
    of connections is 60. -/
theorem network_connections :
  calculate_connections 30 = 60 := by
  sorry

#eval calculate_connections 30

end network_connections_l3821_382143


namespace smallest_4digit_base7_divisible_by_7_l3821_382124

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 4-digit base 7 number --/
def is4DigitBase7 (n : ℕ) : Prop := sorry

/-- The smallest 4-digit base 7 number --/
def smallestBase7_4Digit : ℕ := 1000

theorem smallest_4digit_base7_divisible_by_7 :
  (is4DigitBase7 smallestBase7_4Digit) ∧
  (base7ToDecimal smallestBase7_4Digit % 7 = 0) ∧
  (∀ n : ℕ, is4DigitBase7 n ∧ n < smallestBase7_4Digit → base7ToDecimal n % 7 ≠ 0) :=
sorry

end smallest_4digit_base7_divisible_by_7_l3821_382124


namespace willie_had_48_bananas_l3821_382138

/-- Given the total number of bananas and Charles' initial bananas, 
    calculate Willie's initial bananas. -/
def willies_bananas (total : ℝ) (charles_initial : ℝ) : ℝ :=
  total - charles_initial

/-- Theorem stating that Willie had 48.0 bananas given the problem conditions. -/
theorem willie_had_48_bananas : 
  willies_bananas 83 35 = 48 := by
  sorry

#eval willies_bananas 83 35

end willie_had_48_bananas_l3821_382138


namespace cake_shop_problem_l3821_382194

theorem cake_shop_problem :
  ∃ (N n K : ℕ+), 
    (N - n * K = 6) ∧ 
    (N = (n - 1) * 8 + 1) ∧ 
    (N = 97) := by
  sorry

end cake_shop_problem_l3821_382194


namespace slope_is_constant_l3821_382137

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define a point in the first quadrant
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the line l
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the condition for the areas
def area_condition (x₁ y₁ x₂ y₂ m k : ℝ) : Prop :=
  (y₁^2 + y₂^2) / (y₁ * y₂) = (x₁^2 + x₂^2) / (x₁ * x₂)

-- Main theorem
theorem slope_is_constant
  (k m x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : ellipse x₁ y₁)
  (h₂ : ellipse x₂ y₂)
  (h₃ : in_first_quadrant x₁ y₁)
  (h₄ : in_first_quadrant x₂ y₂)
  (h₅ : line k m x₁ y₁)
  (h₆ : line k m x₂ y₂)
  (h₇ : m ≠ 0)
  (h₈ : area_condition x₁ y₁ x₂ y₂ m k) :
  k = -1/2 := by sorry

end slope_is_constant_l3821_382137


namespace team_size_l3821_382134

theorem team_size (best_score : ℕ) (hypothetical_score : ℕ) (hypothetical_average : ℕ) (total_score : ℕ) :
  best_score = 85 →
  hypothetical_score = 92 →
  hypothetical_average = 84 →
  total_score = 497 →
  ∃ n : ℕ, n = 6 ∧ n * hypothetical_average - (hypothetical_score - best_score) = total_score :=
by sorry

end team_size_l3821_382134


namespace parallelogram_angle_measure_l3821_382132

/-- In a parallelogram, if one angle exceeds the other by 50 degrees,
    and the smaller angle is 65 degrees, then the larger angle is 115 degrees. -/
theorem parallelogram_angle_measure (smaller_angle larger_angle : ℝ) : 
  smaller_angle = 65 →
  larger_angle = smaller_angle + 50 →
  larger_angle = 115 := by
  sorry

end parallelogram_angle_measure_l3821_382132


namespace min_correct_answers_for_score_l3821_382111

/-- Represents the scoring system and conditions of the math competition --/
structure MathCompetition where
  total_questions : Nat
  attempted_questions : Nat
  correct_points : Nat
  incorrect_deduction : Nat
  unanswered_points : Nat
  min_required_score : Nat

/-- Calculates the score based on the number of correct answers --/
def calculate_score (comp : MathCompetition) (correct_answers : Nat) : Int :=
  let incorrect_answers := comp.attempted_questions - correct_answers
  let unanswered := comp.total_questions - comp.attempted_questions
  (correct_answers * comp.correct_points : Int) -
  (incorrect_answers * comp.incorrect_deduction) +
  (unanswered * comp.unanswered_points)

/-- Theorem stating the minimum number of correct answers needed to achieve the required score --/
theorem min_correct_answers_for_score (comp : MathCompetition)
  (h1 : comp.total_questions = 25)
  (h2 : comp.attempted_questions = 20)
  (h3 : comp.correct_points = 8)
  (h4 : comp.incorrect_deduction = 2)
  (h5 : comp.unanswered_points = 2)
  (h6 : comp.min_required_score = 150) :
  ∃ n : Nat, (∀ m : Nat, calculate_score comp m ≥ comp.min_required_score → m ≥ n) ∧
             calculate_score comp n ≥ comp.min_required_score ∧
             n = 18 := by
  sorry


end min_correct_answers_for_score_l3821_382111


namespace max_sum_given_constraints_l3821_382131

theorem max_sum_given_constraints (x y : ℝ) 
  (h1 : x^2 + y^2 = 130) 
  (h2 : x * y = 36) : 
  x + y ≤ Real.sqrt 202 := by
  sorry

end max_sum_given_constraints_l3821_382131


namespace current_at_6_seconds_l3821_382185

/-- The charge function Q(t) representing the amount of electricity flowing through a conductor. -/
def Q (t : ℝ) : ℝ := 3 * t^2 - 3 * t + 4

/-- The current function I(t) derived from Q(t). -/
def I (t : ℝ) : ℝ := 6 * t - 3

/-- Theorem stating that the current at t = 6 seconds is 33 amperes. -/
theorem current_at_6_seconds :
  I 6 = 33 := by sorry

end current_at_6_seconds_l3821_382185


namespace simplify_expression_l3821_382172

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := by
  sorry

end simplify_expression_l3821_382172


namespace mitch_weekday_hours_l3821_382115

/-- Represents the weekly work schedule and earnings of Mitch, a freelancer -/
structure MitchSchedule where
  weekdayHours : ℕ
  weekendHours : ℕ
  weekdayRate : ℕ
  weekendRate : ℕ
  totalEarnings : ℕ

/-- Theorem stating that Mitch works 25 hours from Monday to Friday -/
theorem mitch_weekday_hours (schedule : MitchSchedule) :
  schedule.weekendHours = 6 ∧
  schedule.weekdayRate = 3 ∧
  schedule.weekendRate = 6 ∧
  schedule.totalEarnings = 111 →
  schedule.weekdayHours = 25 := by
  sorry

end mitch_weekday_hours_l3821_382115


namespace seokjin_drank_least_l3821_382141

/-- Represents the amount of milk drunk by each person in liters -/
structure MilkConsumption where
  jungkook : ℝ
  seokjin : ℝ
  yoongi : ℝ

/-- Given the milk consumption of Jungkook, Seokjin, and Yoongi, 
    proves that Seokjin drank the least amount of milk -/
theorem seokjin_drank_least (m : MilkConsumption) 
  (h1 : m.jungkook = 1.3)
  (h2 : m.seokjin = 11/10)
  (h3 : m.yoongi = 7/5) : 
  m.seokjin < m.jungkook ∧ m.seokjin < m.yoongi := by
  sorry

#check seokjin_drank_least

end seokjin_drank_least_l3821_382141


namespace sons_age_l3821_382139

/-- Proves that given the conditions, the son's age is 26 years -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 28 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 26 := by
sorry


end sons_age_l3821_382139


namespace complement_of_M_in_U_l3821_382108

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x | (x - 1) * (x - 4) = 0}

theorem complement_of_M_in_U : U \ M = {2, 3} := by sorry

end complement_of_M_in_U_l3821_382108


namespace conic_section_types_l3821_382155

/-- The equation y^4 - 6x^4 = 3y^2 - 4 represents the union of a hyperbola and an ellipse -/
theorem conic_section_types (x y : ℝ) : 
  y^4 - 6*x^4 = 3*y^2 - 4 → 
  (∃ (a b : ℝ), y^2 - a*x^2 = b ∧ a > 0 ∧ b > 0) ∧ 
  (∃ (c d : ℝ), y^2 + c*x^2 = d ∧ c > 0 ∧ d > 0) := by
sorry

end conic_section_types_l3821_382155
