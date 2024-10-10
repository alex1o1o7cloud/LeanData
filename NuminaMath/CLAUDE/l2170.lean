import Mathlib

namespace solution_x_is_three_fourths_l2170_217060

-- Define the * operation
def star (a b : ℝ) : ℝ := 4 * a - 2 * b

-- State the theorem
theorem solution_x_is_three_fourths :
  ∃ x : ℝ, star 7 (star 3 (x - 1)) = 3 ∧ x = 3/4 := by
  sorry

end solution_x_is_three_fourths_l2170_217060


namespace sum_product_remainder_l2170_217037

theorem sum_product_remainder : (1789 * 1861 * 1945 + 1533 * 1607 * 1688) % 7 = 3 := by
  sorry

end sum_product_remainder_l2170_217037


namespace tangent_parallel_to_4x_l2170_217006

/-- The curve function f(x) = x^3 + x - 2 -/
def f (x : ℝ) : ℝ := x^3 + x - 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3*x^2 + 1

theorem tangent_parallel_to_4x :
  ∃ x : ℝ, f x = 0 ∧ f' x = 4 := by
  sorry

end tangent_parallel_to_4x_l2170_217006


namespace haley_car_distance_l2170_217057

/-- Calculates the distance covered given a fuel-to-distance ratio and fuel consumption -/
def distance_covered (fuel_ratio : ℕ) (distance_ratio : ℕ) (fuel_used : ℕ) : ℕ :=
  (fuel_used / fuel_ratio) * distance_ratio

/-- Theorem stating that for a 4:7 fuel-to-distance ratio and 44 gallons of fuel, the distance covered is 77 miles -/
theorem haley_car_distance :
  distance_covered 4 7 44 = 77 := by
  sorry

end haley_car_distance_l2170_217057


namespace no_upper_bound_for_a_l2170_217008

/-- The number of different representations of n as the sum of different divisors -/
def a (n : ℕ) : ℕ := sorry

/-- There is no upper bound M for a(n) that holds for all n -/
theorem no_upper_bound_for_a : ∀ M : ℕ, ∃ n : ℕ, a n > M := by sorry

end no_upper_bound_for_a_l2170_217008


namespace article_price_l2170_217038

theorem article_price (P : ℝ) : 
  P * (1 - 0.1) * (1 - 0.2) = 72 → P = 100 := by
  sorry

end article_price_l2170_217038


namespace fractional_equation_solution_l2170_217061

theorem fractional_equation_solution :
  ∀ x : ℚ, x ≠ 1 → 3*x - 3 ≠ 0 →
  (2*x / (x - 1) = x / (3*x - 3) + 1) ↔ (x = -3/2) :=
by sorry

end fractional_equation_solution_l2170_217061


namespace rational_function_value_l2170_217070

structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c

def has_asymptotes (f : RationalFunction) (x₁ x₂ : ℝ) : Prop :=
  f.q x₁ = 0 ∧ f.q x₂ = 0

def passes_through (f : RationalFunction) (x y : ℝ) : Prop :=
  f.q x ≠ 0 ∧ f.p x / f.q x = y

theorem rational_function_value (f : RationalFunction) :
  has_asymptotes f (-4) 1 →
  passes_through f 0 0 →
  passes_through f 2 (-2) →
  f.p 3 / f.q 3 = -9/7 := by
    sorry

end rational_function_value_l2170_217070


namespace embankment_completion_time_l2170_217001

/-- The time required for a group of workers to complete an embankment -/
def embankment_time (workers : ℕ) (portion : ℚ) (days : ℚ) : Prop :=
  ∃ (rate : ℚ), rate > 0 ∧ portion = (workers : ℚ) * rate * days

theorem embankment_completion_time :
  embankment_time 60 (1/2) 5 →
  embankment_time 80 1 (15/2) :=
by sorry

end embankment_completion_time_l2170_217001


namespace money_division_l2170_217009

theorem money_division (a b c : ℚ) 
  (h1 : a = (2/3) * (b + c))
  (h2 : b = (6/9) * (a + c))
  (h3 : a = 280) : 
  a + b + c = 700 := by
sorry

end money_division_l2170_217009


namespace equation_roots_imply_m_range_l2170_217076

theorem equation_roots_imply_m_range (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   4^x₁ - m * 2^(x₁+1) + 2 - m = 0 ∧
   4^x₂ - m * 2^(x₂+1) + 2 - m = 0) →
  1 < m ∧ m < 2 :=
by sorry

end equation_roots_imply_m_range_l2170_217076


namespace extra_interest_proof_l2170_217018

/-- Calculates simple interest -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem extra_interest_proof (principal : ℝ) (rate1 : ℝ) (rate2 : ℝ) (time : ℝ) :
  principal = 5000 ∧ rate1 = 0.18 ∧ rate2 = 0.12 ∧ time = 2 →
  simpleInterest principal rate1 time - simpleInterest principal rate2 time = 600 := by
  sorry

end extra_interest_proof_l2170_217018


namespace volume_pyramid_section_l2170_217041

/-- The volume of a section of a regular triangular pyramid -/
theorem volume_pyramid_section (H α β : Real) 
  (h_positive : H > 0)
  (h_angle_α : 0 < α ∧ α < π / 2)
  (h_angle_β : 0 < β ∧ β < π / 2 - α) :
  ∃ V : Real, V = (3 * Real.sqrt 3 * H^3 * Real.sin α * Real.tan α^2 * Real.cos (α - β)) / (8 * Real.sin β) :=
sorry

end volume_pyramid_section_l2170_217041


namespace intersection_and_subsets_l2170_217016

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {x | x ≥ 0}

theorem intersection_and_subsets :
  (A ∩ B = {1, 2}) ∧
  (Set.powerset (A ∩ B) = {{1}, {2}, ∅, {1, 2}}) := by
  sorry

end intersection_and_subsets_l2170_217016


namespace sin_480_degrees_l2170_217017

theorem sin_480_degrees : Real.sin (480 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_480_degrees_l2170_217017


namespace regular_hexagon_interior_angle_regular_hexagon_interior_angle_is_120_l2170_217050

/-- The measure of one interior angle of a regular hexagon is 120 degrees. -/
theorem regular_hexagon_interior_angle : ℝ :=
  let n : ℕ := 6  -- number of sides in a hexagon
  let sum_interior_angles : ℝ := 180 * (n - 2)
  let num_angles : ℕ := n
  sum_interior_angles / num_angles

/-- The result of regular_hexagon_interior_angle is equal to 120. -/
theorem regular_hexagon_interior_angle_is_120 : 
  regular_hexagon_interior_angle = 120 := by sorry

end regular_hexagon_interior_angle_regular_hexagon_interior_angle_is_120_l2170_217050


namespace new_student_weight_l2170_217094

/-- Given a group of students and their weights, calculates the weight of a new student
    that changes the average weight of the group. -/
theorem new_student_weight
  (n : ℕ) -- number of students before new admission
  (w : ℝ) -- average weight before new admission
  (new_w : ℝ) -- new average weight after admission
  (h1 : n = 29) -- there are 29 students initially
  (h2 : w = 28) -- the initial average weight is 28 kg
  (h3 : new_w = 27.4) -- the new average weight is 27.4 kg
  : (n + 1) * new_w - n * w = 10 := by
  sorry

end new_student_weight_l2170_217094


namespace common_ratio_of_geometric_sequence_l2170_217098

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Define the arithmetic sequence property
def arithmetic_sequence_property (a : ℕ → ℝ) (q : ℝ) :=
  (3 / 2) * (a 1 * q) = (2 * a 0 + a 0 * q^2) / 2

-- Theorem statement
theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : arithmetic_sequence_property a q) :
  q = 1 ∨ q = 2 := by
  sorry

end common_ratio_of_geometric_sequence_l2170_217098


namespace new_number_properties_l2170_217049

def new_number (a b : ℕ) : ℕ := a * b + a + b

def is_new_number (n : ℕ) : Prop :=
  ∃ a b, new_number a b = n

theorem new_number_properties :
  (¬ is_new_number 2008) ∧
  (∀ a b : ℕ, 2 ∣ (new_number a b + 1)) ∧
  (∀ a b : ℕ, 10 ∣ (new_number a b + 1)) :=
sorry

end new_number_properties_l2170_217049


namespace unique_solution_power_equation_l2170_217067

theorem unique_solution_power_equation :
  ∀ x y : ℕ, x ≥ 1 → y ≥ 1 → (2^x : ℕ) + 3 = 11^y → x = 3 ∧ y = 1 := by
  sorry

end unique_solution_power_equation_l2170_217067


namespace complex_modulus_problem_l2170_217095

theorem complex_modulus_problem (z : ℂ) (i : ℂ) : 
  i * i = -1 → z = (3 - i) / (1 + i) → Complex.abs (z + i) = Real.sqrt 2 := by
  sorry

end complex_modulus_problem_l2170_217095


namespace prove_b_equals_one_l2170_217023

theorem prove_b_equals_one (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 49 * 45 * b) : b = 1 := by
  sorry

end prove_b_equals_one_l2170_217023


namespace unripe_oranges_per_day_is_65_l2170_217051

/-- The number of days of harvest -/
def harvest_days : ℕ := 6

/-- The total number of sacks of unripe oranges after the harvest period -/
def total_unripe_oranges : ℕ := 390

/-- The number of sacks of unripe oranges harvested per day -/
def unripe_oranges_per_day : ℕ := total_unripe_oranges / harvest_days

/-- Theorem stating that the number of sacks of unripe oranges harvested per day is 65 -/
theorem unripe_oranges_per_day_is_65 : unripe_oranges_per_day = 65 := by
  sorry

end unripe_oranges_per_day_is_65_l2170_217051


namespace geometric_sequence_product_l2170_217062

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 1 →
  a 10 = 3 →
  a 2 * a 3 * a 4 * a 5 * a 6 * a 7 * a 8 * a 9 = 81 := by
  sorry

end geometric_sequence_product_l2170_217062


namespace chess_tournament_matches_prime_l2170_217090

/-- Represents a single-elimination chess tournament --/
structure ChessTournament where
  totalPlayers : ℕ
  byePlayers : ℕ
  initialPlayers : ℕ

/-- Calculates the number of matches in a single-elimination tournament --/
def matchesPlayed (t : ChessTournament) : ℕ := t.totalPlayers - 1

/-- Theorem: In the given chess tournament, 119 matches are played and this number is prime --/
theorem chess_tournament_matches_prime (t : ChessTournament) 
  (h1 : t.totalPlayers = 120) 
  (h2 : t.byePlayers = 40) 
  (h3 : t.initialPlayers = 80) : 
  matchesPlayed t = 119 ∧ Nat.Prime 119 := by
  sorry

#eval Nat.Prime 119  -- To verify that 119 is indeed prime

end chess_tournament_matches_prime_l2170_217090


namespace circle_distance_inequality_l2170_217064

theorem circle_distance_inequality (r s AB : ℝ) (h1 : r > s) (h2 : AB > 0) : ¬(r - s > AB) := by
  sorry

end circle_distance_inequality_l2170_217064


namespace wardrobe_cost_is_180_l2170_217068

/-- Calculates the total cost of Marcia's wardrobe given the following conditions:
  - 3 skirts at $20.00 each
  - 5 blouses at $15.00 each
  - 2 pairs of pants at $30.00 each, with a sale: buy 1 pair, get 1 pair 1/2 off
-/
def wardrobeCost (skirtPrice blousePrice pantPrice : ℚ) : ℚ :=
  let skirtCost := 3 * skirtPrice
  let blouseCost := 5 * blousePrice
  let pantCost := pantPrice + (pantPrice / 2)
  skirtCost + blouseCost + pantCost

/-- Proves that the total cost of Marcia's wardrobe is $180.00 -/
theorem wardrobe_cost_is_180 :
  wardrobeCost 20 15 30 = 180 := by
  sorry

#eval wardrobeCost 20 15 30

end wardrobe_cost_is_180_l2170_217068


namespace intersection_and_union_range_of_a_l2170_217014

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 4}
def B : Set ℝ := {x | x^2 - x - 6 ≤ 0}

-- Part 1
theorem intersection_and_union (a : ℝ) (h : a = 0) :
  (A a ∩ B = {x | 0 ≤ x ∧ x ≤ 3}) ∧
  (A a ∪ (Set.univ \ B) = {x | x < -2 ∨ x ≥ 0}) := by sorry

-- Part 2
theorem range_of_a :
  {a : ℝ | A a ∪ B = B} = {a : ℝ | -2 ≤ a ∧ a ≤ -1} := by sorry

end intersection_and_union_range_of_a_l2170_217014


namespace reciprocal_of_one_l2170_217005

-- Define the concept of reciprocal
def is_reciprocal (a b : ℝ) : Prop := a * b = 1

-- Theorem statement
theorem reciprocal_of_one : is_reciprocal 1 1 := by
  sorry

end reciprocal_of_one_l2170_217005


namespace inequality_solution_l2170_217027

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 5/x + 21/10) ↔ -2 < x ∧ x < 0 := by
  sorry

end inequality_solution_l2170_217027


namespace abs_z_equals_2_sqrt_2_l2170_217053

def z : ℂ := Complex.I - 2 * Complex.I^2 + 3 * Complex.I^3

theorem abs_z_equals_2_sqrt_2 : Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end abs_z_equals_2_sqrt_2_l2170_217053


namespace square_side_length_l2170_217091

/-- Given two identical overlapping squares where the upper square is moved 3 cm right and 5 cm down,
    resulting in a shaded area of 57 square centimeters, prove that the side length of each square is 9 cm. -/
theorem square_side_length (a : ℝ) 
  (h1 : 3 * a + 5 * (a - 3) = 57) : 
  a = 9 := by
  sorry

end square_side_length_l2170_217091


namespace final_value_of_A_l2170_217002

-- Define the initial value of A
def A_initial : ℤ := 15

-- Define the operation as a function
def operation (x : ℤ) : ℤ := -x + 5

-- Theorem stating the final value of A after the operation
theorem final_value_of_A : operation A_initial = -10 := by
  sorry

end final_value_of_A_l2170_217002


namespace polygon_diagonals_l2170_217048

theorem polygon_diagonals (n : ℕ) (h : n = 150) : (n * (n - 3)) / 2 = 11025 := by
  sorry

end polygon_diagonals_l2170_217048


namespace polynomial_root_implies_h_value_l2170_217079

theorem polynomial_root_implies_h_value : 
  ∀ h : ℚ, (3 : ℚ)^3 + h * 3 + 14 = 0 → h = -41/3 :=
by sorry

end polynomial_root_implies_h_value_l2170_217079


namespace complex_number_problem_l2170_217046

theorem complex_number_problem (m : ℝ) (z : ℂ) :
  let z₁ : ℂ := m * (m - 1) + (m - 1) * Complex.I
  (z₁.re = 0 ∧ z₁.im ≠ 0) →
  (3 + z₁) * z = 4 + 2 * Complex.I →
  m = 0 ∧ z = 1 + Complex.I := by
sorry

end complex_number_problem_l2170_217046


namespace only_vegetarian_count_l2170_217034

/-- Represents the number of people in different dietary categories in a family --/
structure FamilyDiet where
  only_nonveg : ℕ
  both_veg_and_nonveg : ℕ
  total_veg : ℕ

/-- Theorem stating the number of people who eat only vegetarian --/
theorem only_vegetarian_count (f : FamilyDiet) 
  (h1 : f.only_nonveg = 8)
  (h2 : f.both_veg_and_nonveg = 6)
  (h3 : f.total_veg = 19) :
  f.total_veg - f.both_veg_and_nonveg = 13 := by
  sorry

#check only_vegetarian_count

end only_vegetarian_count_l2170_217034


namespace monotonic_quadratic_function_l2170_217087

/-- A function f is monotonic on an interval [a, b] if it is either
    non-decreasing or non-increasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The theorem states that for the function f(x) = x^2 - kx + 1,
    f is monotonic on the interval [1, 2] if and only if
    k is in the set (-∞, 2] ∪ [4, +∞). -/
theorem monotonic_quadratic_function (k : ℝ) :
  IsMonotonic (fun x => x^2 - k*x + 1) 1 2 ↔ k ≤ 2 ∨ k ≥ 4 :=
sorry

end monotonic_quadratic_function_l2170_217087


namespace polynomial_division_remainder_l2170_217045

/-- The polynomial division theorem for z^2023 - 1 divided by z^2 + z + 1 -/
theorem polynomial_division_remainder (z : ℂ) : ∃ (Q R : ℂ → ℂ),
  z^2023 - 1 = (z^2 + z + 1) * Q z + R z ∧ 
  (∀ x, R x = -x - 1) ∧
  (∃ a b, ∀ x, R x = a * x + b) := by
  sorry

end polynomial_division_remainder_l2170_217045


namespace maria_cookie_baggies_l2170_217015

/-- The number of baggies Maria can make with her cookies -/
def num_baggies (cookies_per_baggie : ℕ) (chocolate_chip_cookies : ℕ) (oatmeal_cookies : ℕ) : ℕ :=
  (chocolate_chip_cookies + oatmeal_cookies) / cookies_per_baggie

/-- Theorem stating that Maria can make 7 baggies of cookies -/
theorem maria_cookie_baggies :
  num_baggies 5 33 2 = 7 := by
  sorry

end maria_cookie_baggies_l2170_217015


namespace total_seashells_l2170_217030

theorem total_seashells (red_shells green_shells other_shells : ℕ) 
  (h1 : red_shells = 76)
  (h2 : green_shells = 49)
  (h3 : other_shells = 166) :
  red_shells + green_shells + other_shells = 291 := by
  sorry

end total_seashells_l2170_217030


namespace prob_b_greater_a_value_l2170_217000

/-- The number of possible choices for each person -/
def n : ℕ := 1000

/-- The probability of B picking a number greater than A -/
def prob_b_greater_a : ℚ :=
  (n * (n - 1) / 2) / (n * n)

/-- Theorem: The probability of B picking a number greater than A is 499500/1000000 -/
theorem prob_b_greater_a_value : prob_b_greater_a = 499500 / 1000000 := by
  sorry

end prob_b_greater_a_value_l2170_217000


namespace vector_operations_l2170_217029

def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (-1, 3)

theorem vector_operations :
  (a.1 + b.1, a.2 + b.2) = (1, 3) ∧
  (a.1 - b.1, a.2 - b.2) = (3, -3) := by
  sorry

end vector_operations_l2170_217029


namespace ball_path_length_l2170_217032

theorem ball_path_length (A B C M : ℝ × ℝ) : 
  -- Triangle ABC is a right triangle with ∠ABC = 90° and ∠BAC = 60°
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0 →
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = (B.1 - A.1)^2 + (B.2 - A.2)^2 →
  -- AB = 6
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 36 →
  -- M is the midpoint of BC
  M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) →
  -- The length of the path from M to AB to AC and back to M is 3√21
  ∃ (P Q : ℝ × ℝ), 
    P.1 = B.1 ∧ 
    (Q.1 - A.1) * (C.1 - A.1) + (Q.2 - A.2) * (C.2 - A.2) = 0 ∧
    (P.1 - M.1)^2 + (P.2 - M.2)^2 + 
    (Q.1 - P.1)^2 + (Q.2 - P.2)^2 + 
    (M.1 - Q.1)^2 + (M.2 - Q.2)^2 = 189 := by
  sorry

end ball_path_length_l2170_217032


namespace empty_solution_set_range_l2170_217063

-- Define the inequality
def inequality (x m : ℝ) : Prop := |x - 1| + |x - m| < 2 * m

-- Define the theorem
theorem empty_solution_set_range :
  (∀ m : ℝ, (0 < m ∧ m < 1/3) ↔ ∀ x : ℝ, ¬(inequality x m)) :=
sorry

end empty_solution_set_range_l2170_217063


namespace smallest_with_ten_divisors_l2170_217022

/-- A function that counts the number of positive divisors of a natural number -/
def countDivisors (n : ℕ) : ℕ := sorry

/-- A predicate that checks if a natural number has exactly 10 positive divisors -/
def hasTenDivisors (n : ℕ) : Prop := countDivisors n = 10

/-- The theorem stating that 48 is the smallest natural number with exactly 10 positive divisors -/
theorem smallest_with_ten_divisors :
  (∀ m : ℕ, m < 48 → ¬(hasTenDivisors m)) ∧ hasTenDivisors 48 := by sorry

end smallest_with_ten_divisors_l2170_217022


namespace mom_talia_age_ratio_l2170_217056

-- Define Talia's current age
def talia_current_age : ℕ := 20 - 7

-- Define Talia's father's current age
def father_current_age : ℕ := 36

-- Define Talia's mother's current age
def mother_current_age : ℕ := father_current_age + 3

-- Theorem stating the ratio of Talia's mom's age to Talia's age
theorem mom_talia_age_ratio :
  mother_current_age / talia_current_age = 3 := by
  sorry

end mom_talia_age_ratio_l2170_217056


namespace candies_per_packet_candies_per_packet_proof_l2170_217082

/-- The number of candies in a packet given Bobby's eating habits and the time it takes to finish the packets. -/
theorem candies_per_packet : ℕ :=
  let packets : ℕ := 2
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2
  let candies_per_weekday : ℕ := 2
  let candies_per_weekend_day : ℕ := 1
  let weeks_to_finish : ℕ := 3
  18

/-- Proof that the number of candies in a packet is 18. -/
theorem candies_per_packet_proof :
  let packets : ℕ := 2
  let weekdays : ℕ := 5
  let weekend_days : ℕ := 2
  let candies_per_weekday : ℕ := 2
  let candies_per_weekend_day : ℕ := 1
  let weeks_to_finish : ℕ := 3
  candies_per_packet = 18 := by
  sorry

end candies_per_packet_candies_per_packet_proof_l2170_217082


namespace sum_with_radical_conjugate_l2170_217021

-- Define the radical conjugate
def radical_conjugate (a b : ℝ) : ℝ := a + b

-- State the theorem
theorem sum_with_radical_conjugate :
  let x := 8 - Real.sqrt 1369
  x + radical_conjugate 8 (Real.sqrt 1369) = 16 := by
  sorry

end sum_with_radical_conjugate_l2170_217021


namespace douglas_vote_percentage_l2170_217085

theorem douglas_vote_percentage (total_voters : ℕ) (x_voters y_voters : ℕ) 
  (douglas_total_votes douglas_x_votes douglas_y_votes : ℕ) :
  x_voters = 2 * y_voters →
  douglas_total_votes = (66 * (x_voters + y_voters)) / 100 →
  douglas_x_votes = (74 * x_voters) / 100 →
  douglas_total_votes = douglas_x_votes + douglas_y_votes →
  (douglas_y_votes * 100) / y_voters = 50 :=
by sorry

end douglas_vote_percentage_l2170_217085


namespace parabola_chord_length_l2170_217077

/-- The length of a chord AB of a parabola y^2 = 8x intersected by a line y = kx - 2 -/
theorem parabola_chord_length (k : ℝ) : 
  (∃ A B : ℝ × ℝ, 
    (A.2^2 = 8 * A.1 ∧ A.2 = k * A.1 - 2) ∧ 
    (B.2^2 = 8 * B.1 ∧ B.2 = k * B.1 - 2) ∧
    A ≠ B ∧
    (A.1 + B.1) / 2 = 2) →
  ∃ A B : ℝ × ℝ, 
    (A.2^2 = 8 * A.1 ∧ A.2 = k * A.1 - 2) ∧ 
    (B.2^2 = 8 * B.1 ∧ B.2 = k * B.1 - 2) ∧
    A ≠ B ∧
    (A.1 + B.1) / 2 = 2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 15 :=
by sorry


end parabola_chord_length_l2170_217077


namespace variance_and_shifted_average_l2170_217059

theorem variance_and_shifted_average
  (x₁ x₂ x₃ x₄ : ℝ)
  (pos_x : x₁ > 0 ∧ x₂ > 0 ∧ x₃ > 0 ∧ x₄ > 0)
  (variance : (1/4) * (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16) = (1/4) * ((x₁ - (x₁ + x₂ + x₃ + x₄)/4)^2 +
                                                                  (x₂ - (x₁ + x₂ + x₃ + x₄)/4)^2 +
                                                                  (x₃ - (x₁ + x₂ + x₃ + x₄)/4)^2 +
                                                                  (x₄ - (x₁ + x₂ + x₃ + x₄)/4)^2)) :
  ((x₁ + 3) + (x₂ + 3) + (x₃ + 3) + (x₄ + 3)) / 4 = 5 := by
  sorry

end variance_and_shifted_average_l2170_217059


namespace cd_product_value_l2170_217055

/-- An equilateral triangle with vertices at (0,0), (c,17), and (d,43) -/
structure EquilateralTriangle where
  c : ℝ
  d : ℝ

/-- The product of c and d in the equilateral triangle -/
def cd_product (triangle : EquilateralTriangle) : ℝ := triangle.c * triangle.d

/-- Theorem stating that the product cd equals -1689/24 for the given equilateral triangle -/
theorem cd_product_value (triangle : EquilateralTriangle) :
  cd_product triangle = -1689 / 24 := by sorry

end cd_product_value_l2170_217055


namespace plan_a_is_lowest_l2170_217069

/-- Represents a payment plan with monthly payment, duration, and interest rate -/
structure PaymentPlan where
  monthly_payment : ℝ
  duration : ℕ
  interest_rate : ℝ

/-- Calculates the total repayment amount for a given payment plan -/
def total_repayment (plan : PaymentPlan) : ℝ :=
  let principal := plan.monthly_payment * plan.duration
  principal + principal * plan.interest_rate

/-- The three payment plans available to Aaron -/
def plan_a : PaymentPlan := ⟨100, 12, 0.1⟩
def plan_b : PaymentPlan := ⟨90, 15, 0.08⟩
def plan_c : PaymentPlan := ⟨80, 18, 0.06⟩

/-- Theorem stating that Plan A has the lowest total repayment amount -/
theorem plan_a_is_lowest :
  total_repayment plan_a < total_repayment plan_b ∧
  total_repayment plan_a < total_repayment plan_c :=
by sorry

end plan_a_is_lowest_l2170_217069


namespace f_non_monotonic_iff_a_in_range_l2170_217031

-- Define the piecewise function f(x)
noncomputable def f (a t x : ℝ) : ℝ :=
  if x ≤ t then (4*a - 3)*x + 2*a - 4 else 2*x^3 - 6*x

-- Define the property of being non-monotonic on ℝ
def is_non_monotonic (f : ℝ → ℝ) : Prop :=
  ∃ x y z : ℝ, x < y ∧ y < z ∧ (f x < f y ∧ f y > f z ∨ f x > f y ∧ f y < f z)

-- State the theorem
theorem f_non_monotonic_iff_a_in_range :
  (∀ t : ℝ, is_non_monotonic (f a t)) ↔ a ∈ Set.Iic (3/4) :=
sorry

end f_non_monotonic_iff_a_in_range_l2170_217031


namespace no_four_digit_reverse_diff_1008_l2170_217093

theorem no_four_digit_reverse_diff_1008 : 
  ¬ ∃ (a b c d : ℕ), 
    (1000 ≤ 1000 * a + 100 * b + 10 * c + d) ∧ 
    (1000 * a + 100 * b + 10 * c + d < 10000) ∧
    ((1000 * a + 100 * b + 10 * c + d) - (1000 * d + 100 * c + 10 * b + a) = 1008) :=
by sorry

end no_four_digit_reverse_diff_1008_l2170_217093


namespace spherical_to_rectangular_conversion_l2170_217084

/-- Proves that the conversion from spherical coordinates (10, 3π/4, π/4) to 
    rectangular coordinates results in (-5, 5, 5√2) -/
theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 10
  let θ : ℝ := 3 * Real.pi / 4
  let φ : ℝ := Real.pi / 4
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x = -5) ∧ (y = 5) ∧ (z = 5 * Real.sqrt 2) :=
by sorry

end spherical_to_rectangular_conversion_l2170_217084


namespace complex_number_quadrant_l2170_217026

theorem complex_number_quadrant : ∃ (z : ℂ), z = (Complex.I : ℂ) / (1 + Complex.I) ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end complex_number_quadrant_l2170_217026


namespace perpendicular_condition_l2170_217058

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The first line: 2x - y - 1 = 0 -/
def line1 : Line := { a := 2, b := -1, c := -1 }

/-- The second line: mx + y + 1 = 0 -/
def line2 (m : ℝ) : Line := { a := m, b := 1, c := 1 }

/-- Theorem: The necessary and sufficient condition for perpendicularity -/
theorem perpendicular_condition :
  ∀ m : ℝ, perpendicular line1 (line2 m) ↔ m = 1/2 := by sorry

end perpendicular_condition_l2170_217058


namespace sqrt_identity_l2170_217078

theorem sqrt_identity : (Real.sqrt 3 + Real.sqrt 2) * (Real.sqrt 3 - Real.sqrt 2)^2 = Real.sqrt 3 - Real.sqrt 2 := by
  sorry

end sqrt_identity_l2170_217078


namespace quadratic_monotone_increasing_l2170_217097

/-- A quadratic function f(x) = x^2 + 2ax + b is monotonically increasing
    on the interval [-1, +∞) if and only if a ≥ 1 -/
theorem quadratic_monotone_increasing (a b : ℝ) :
  (∀ x₁ x₂ : ℝ, -1 ≤ x₁ ∧ x₁ < x₂ → x₁^2 + 2*a*x₁ + b < x₂^2 + 2*a*x₂ + b) ↔ a ≥ 1 := by
  sorry

end quadratic_monotone_increasing_l2170_217097


namespace lighthouse_min_fuel_l2170_217028

/-- Represents the lighthouse generator's operation parameters -/
structure LighthouseGenerator where
  fuel_per_hour : ℝ
  startup_fuel : ℝ
  total_hours : ℝ
  max_stop_time : ℝ
  min_run_time : ℝ

/-- Calculates the minimum fuel required for the lighthouse generator -/
def min_fuel_required (g : LighthouseGenerator) : ℝ :=
  -- The actual calculation would go here
  sorry

/-- Theorem stating the minimum fuel required for the given parameters -/
theorem lighthouse_min_fuel :
  let g : LighthouseGenerator := {
    fuel_per_hour := 6,
    startup_fuel := 0.5,
    total_hours := 10,
    max_stop_time := 1/6,  -- 10 minutes in hours
    min_run_time := 1/4    -- 15 minutes in hours
  }
  min_fuel_required g = 47.5 := by
  sorry

end lighthouse_min_fuel_l2170_217028


namespace letter_at_unknown_position_l2170_217044

/-- Represents the letters that can be used in the grid -/
inductive Letter : Type
| A | B | C | D | E

/-- Represents a position in the 5x5 grid -/
structure Position :=
  (row : Fin 5)
  (col : Fin 5)

/-- Represents the 5x5 grid -/
def Grid := Position → Letter

/-- Check if each letter appears exactly once in each row -/
def valid_rows (g : Grid) : Prop :=
  ∀ r : Fin 5, ∀ l : Letter, ∃! c : Fin 5, g ⟨r, c⟩ = l

/-- Check if each letter appears exactly once in each column -/
def valid_columns (g : Grid) : Prop :=
  ∀ c : Fin 5, ∀ l : Letter, ∃! r : Fin 5, g ⟨r, c⟩ = l

/-- Check if each letter appears exactly once in the main diagonal -/
def valid_main_diagonal (g : Grid) : Prop :=
  ∀ l : Letter, ∃! i : Fin 5, g ⟨i, i⟩ = l

/-- Check if each letter appears exactly once in the anti-diagonal -/
def valid_anti_diagonal (g : Grid) : Prop :=
  ∀ l : Letter, ∃! i : Fin 5, g ⟨i, 4 - i⟩ = l

/-- Check if the grid satisfies all constraints -/
def valid_grid (g : Grid) : Prop :=
  valid_rows g ∧ valid_columns g ∧ valid_main_diagonal g ∧ valid_anti_diagonal g

/-- The theorem to prove -/
theorem letter_at_unknown_position (g : Grid) 
  (h_valid : valid_grid g)
  (h_A : g ⟨0, 0⟩ = Letter.A)
  (h_D : g ⟨3, 0⟩ = Letter.D)
  (h_E : g ⟨4, 0⟩ = Letter.E) :
  ∃ p : Position, g p = Letter.B :=
by sorry

end letter_at_unknown_position_l2170_217044


namespace remainder_problem_l2170_217052

theorem remainder_problem (a : ℤ) : ∃ (n : ℕ), n > 1 ∧
  (1108 + a) % n = 4 ∧
  1453 % n = 4 ∧
  (1844 + 2*a) % n = 4 ∧
  2281 % n = 4 := by
  sorry

end remainder_problem_l2170_217052


namespace rounding_estimate_l2170_217043

theorem rounding_estimate (a b c d : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (a' : ℕ) (ha' : a' ≥ a)
  (b' : ℕ) (hb' : b' ≤ b)
  (c' : ℕ) (hc' : c' ≥ c)
  (d' : ℕ) (hd' : d' ≥ d) :
  (a' * d' : ℚ) / b' + c' > (a * d : ℚ) / b + c :=
sorry

end rounding_estimate_l2170_217043


namespace smallest_equal_flock_size_l2170_217019

theorem smallest_equal_flock_size (duck_flock_size crane_flock_size : ℕ) 
  (duck_flock_size_pos : duck_flock_size > 0)
  (crane_flock_size_pos : crane_flock_size > 0)
  (duck_flock_size_eq : duck_flock_size = 13)
  (crane_flock_size_eq : crane_flock_size = 17) :
  ∃ n : ℕ, n > 0 ∧ 
    n % duck_flock_size = 0 ∧ 
    n % crane_flock_size = 0 ∧
    (∀ m : ℕ, m > 0 ∧ m % duck_flock_size = 0 ∧ m % crane_flock_size = 0 → m ≥ n) ∧
    n = 221 :=
by sorry

end smallest_equal_flock_size_l2170_217019


namespace johns_remaining_money_l2170_217088

def remaining_money (initial amount_spent_on_sweets amount_given_to_each_friend : ℚ) : ℚ :=
  initial - (amount_spent_on_sweets + 2 * amount_given_to_each_friend)

theorem johns_remaining_money :
  remaining_money 10.50 2.25 2.20 = 3.85 := by
  sorry

end johns_remaining_money_l2170_217088


namespace no_polynomial_exists_l2170_217013

theorem no_polynomial_exists : ¬∃ (P : ℤ → ℤ) (a b c d : ℤ),
  (∀ n : ℕ, ∃ k : ℤ, P n = k) ∧  -- P has integer coefficients
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- a, b, c, d are distinct
  P a = 3 ∧ P b = 3 ∧ P c = 3 ∧ P d = 4 := by
sorry

end no_polynomial_exists_l2170_217013


namespace polynomial_factorization_l2170_217012

theorem polynomial_factorization (a b : ℝ) : 
  a^2 - b^2 + 2*a + 1 = (a-b+1)*(a+b+1) := by sorry

end polynomial_factorization_l2170_217012


namespace percentage_problem_l2170_217007

theorem percentage_problem (P : ℝ) : 
  P * 140 = (4/5) * 140 - 21 → P = 0.65 := by
  sorry

end percentage_problem_l2170_217007


namespace selection_schemes_count_l2170_217086

def total_people : ℕ := 6
def cities : ℕ := 4
def excluded_from_paris : ℕ := 2

theorem selection_schemes_count :
  (total_people.choose cities) *
  (cities.factorial) -
  (excluded_from_paris * (total_people - 1).choose (cities - 1) * (cities - 1).factorial) = 240 :=
sorry

end selection_schemes_count_l2170_217086


namespace negation_equivalence_l2170_217072

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x > 0 ∧ x^2 - 1 ≤ 0) ↔ (∀ x : ℝ, x > 0 → x^2 - 1 > 0) := by
  sorry

end negation_equivalence_l2170_217072


namespace closed_set_properties_l2170_217073

-- Define what it means for a set to be closed
def is_closed (M : Set ℤ) : Prop :=
  ∀ a b, a ∈ M → b ∈ M → (a + b) ∈ M ∧ (a - b) ∈ M

-- Define the set M = {-2, -1, 0, 1, 2}
def M : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the set of positive integers
def positive_integers : Set ℤ := {n : ℤ | n > 0}

-- Define the set M = {n | n = 3k, k ∈ Z}
def M_3k : Set ℤ := {n : ℤ | ∃ k : ℤ, n = 3 * k}

theorem closed_set_properties :
  (¬ is_closed M) ∧
  (¬ is_closed positive_integers) ∧
  (is_closed M_3k) ∧
  (∃ A₁ A₂ : Set ℤ, is_closed A₁ ∧ is_closed A₂ ∧ ¬ is_closed (A₁ ∪ A₂)) := by
  sorry

end closed_set_properties_l2170_217073


namespace cubic_with_infinite_equal_pairs_has_integer_root_l2170_217083

/-- A cubic polynomial with integer coefficients -/
structure CubicPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_nonzero : a ≠ 0

/-- Evaluation of a cubic polynomial at a given integer -/
def eval (P : CubicPolynomial) (x : ℤ) : ℤ :=
  P.a * x^3 + P.b * x^2 + P.c * x + P.d

/-- The property that there are infinitely many pairs of distinct integers (x, y) such that xP(x) = yP(y) -/
def has_infinite_equal_pairs (P : CubicPolynomial) : Prop :=
  ∀ n : ℕ, ∃ x y : ℤ, x ≠ y ∧ x.natAbs > n ∧ y.natAbs > n ∧ x * eval P x = y * eval P y

/-- The main theorem: if a cubic polynomial has infinite equal pairs, then it has an integer root -/
theorem cubic_with_infinite_equal_pairs_has_integer_root (P : CubicPolynomial) 
  (h : has_infinite_equal_pairs P) : ∃ k : ℤ, eval P k = 0 := by
  sorry

end cubic_with_infinite_equal_pairs_has_integer_root_l2170_217083


namespace smallest_y_value_l2170_217020

theorem smallest_y_value (y : ℝ) : 
  (2 * y^2 + 7 * y + 3 = 5) → (y ≥ -2) :=
by sorry

end smallest_y_value_l2170_217020


namespace expression_simplification_l2170_217003

theorem expression_simplification (x : ℝ) (h : x^2 + x - 6 = 0) :
  (x - 1) / ((2 / (x - 1)) - 1) = 8 / 3 := by
  sorry

end expression_simplification_l2170_217003


namespace plane_Q_satisfies_conditions_l2170_217035

/-- Plane represented by its normal vector and constant term -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Line represented by a point and a direction vector -/
structure Line where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

def plane_intersection (p1 p2 : Plane) : Line := sorry

def distance_point_to_plane (point : ℝ × ℝ × ℝ) (plane : Plane) : ℝ := sorry

def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem plane_Q_satisfies_conditions : 
  let π₁ : Plane := ⟨2, -3, 4, -5⟩
  let π₂ : Plane := ⟨3, 1, -2, -1⟩
  let Q : Plane := ⟨6, -1, 10, -11⟩
  let intersection := plane_intersection π₁ π₂
  let point := (1, 2, 3)
  line_in_plane intersection Q ∧ 
  distance_point_to_plane point Q = 3 / Real.sqrt 5 ∧
  Q ≠ π₁ ∧ 
  Q ≠ π₂ := by
  sorry


end plane_Q_satisfies_conditions_l2170_217035


namespace power_inequality_l2170_217011

theorem power_inequality (a b x y : ℝ) 
  (ha : 0 ≤ a) (hb : 0 ≤ b) (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hab : a^5 + b^5 ≤ 1) (hxy : x^5 + y^5 ≤ 1) : 
  a^2 * x^3 + b^2 * y^3 ≤ 1 := by sorry

end power_inequality_l2170_217011


namespace same_prime_divisors_same_outcome_l2170_217099

/-- The number game as described in the problem -/
def NumberGame (k : ℕ) (n : ℕ) : Prop :=
  k > 2 ∧ n ≥ k

/-- A number is good if Banana has a winning strategy -/
def IsGood (k : ℕ) (n : ℕ) : Prop :=
  NumberGame k n ∧ sorry -- Definition of good number

/-- Two numbers have the same prime divisors up to k -/
def SamePrimeDivisorsUpTo (k : ℕ) (n n' : ℕ) : Prop :=
  ∀ p : ℕ, p ≤ k → Prime p → (p ∣ n ↔ p ∣ n')

/-- Main theorem: numbers with same prime divisors up to k have the same game outcome -/
theorem same_prime_divisors_same_outcome (k : ℕ) (n n' : ℕ) :
  NumberGame k n → NumberGame k n' → SamePrimeDivisorsUpTo k n n' →
  (IsGood k n ↔ IsGood k n') :=
sorry

end same_prime_divisors_same_outcome_l2170_217099


namespace smallest_number_with_remainders_l2170_217092

theorem smallest_number_with_remainders : ∃! x : ℕ,
  (x % 5 = 2) ∧ (x % 6 = 2) ∧ (x % 7 = 3) ∧
  (∀ y : ℕ, (y % 5 = 2) ∧ (y % 6 = 2) ∧ (y % 7 = 3) → x ≤ y) ∧
  x = 122 := by
sorry

end smallest_number_with_remainders_l2170_217092


namespace last_digit_2_power_2010_l2170_217040

/-- The last digit of 2^n for n ≥ 1 -/
def lastDigitPowerOfTwo (n : ℕ) : ℕ :=
  match n % 4 with
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | 0 => 6
  | _ => 0  -- This case should never occur

theorem last_digit_2_power_2010 : lastDigitPowerOfTwo 2010 = 4 := by
  sorry

#eval lastDigitPowerOfTwo 2010

end last_digit_2_power_2010_l2170_217040


namespace compute_expression_l2170_217025

theorem compute_expression : 25 * (216 / 3 + 36 / 6 + 16 / 25 + 2) = 2016 := by
  sorry

end compute_expression_l2170_217025


namespace rectangles_in_4x5_grid_l2170_217096

/-- The number of rectangles in a horizontal strip of width n -/
def horizontalRectangles (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of rectangles in a vertical strip of height m -/
def verticalRectangles (m : ℕ) : ℕ := m * (m + 1) / 2

/-- The total number of rectangles in an m×n grid -/
def totalRectangles (m n : ℕ) : ℕ :=
  m * horizontalRectangles n + n * verticalRectangles m - m * n

theorem rectangles_in_4x5_grid :
  totalRectangles 4 5 = 24 := by
  sorry

end rectangles_in_4x5_grid_l2170_217096


namespace expand_product_l2170_217074

theorem expand_product (x : ℝ) : (4 * x + 2) * (3 * x - 1) * (x + 6) = 12 * x^3 + 74 * x^2 + 10 * x - 12 := by
  sorry

end expand_product_l2170_217074


namespace arithmetic_calculation_l2170_217080

theorem arithmetic_calculation : 2 + 7 * 3 - 4 + 8 * 2 / 4 = 23 := by
  sorry

end arithmetic_calculation_l2170_217080


namespace number_subtraction_problem_l2170_217036

theorem number_subtraction_problem (x : ℝ) : 
  0.30 * x - 70 = 20 → x = 300 := by
  sorry

end number_subtraction_problem_l2170_217036


namespace contest_probability_l2170_217004

/-- The probability of correctly answering a single question -/
def p : ℝ := 0.8

/-- The number of preset questions -/
def n : ℕ := 5

/-- The probability of answering exactly 4 questions before advancing -/
def prob_four_questions : ℝ := 2 * p^3 * (1 - p)

theorem contest_probability :
  prob_four_questions = 0.128 :=
sorry

end contest_probability_l2170_217004


namespace number_of_women_in_first_group_l2170_217075

/-- The number of women in the first group -/
def W : ℕ := sorry

/-- The work rate of the first group in units per hour -/
def work_rate_1 : ℚ := 75 / (8 * 5)

/-- The work rate of the second group in units per hour -/
def work_rate_2 : ℚ := 30 / (3 * 8)

/-- Theorem stating that the number of women in the first group is 5 -/
theorem number_of_women_in_first_group : W = 5 := by sorry

end number_of_women_in_first_group_l2170_217075


namespace cube_sum_implies_sum_bound_l2170_217065

theorem cube_sum_implies_sum_bound (a b : ℝ) :
  a > 0 → b > 0 → a^3 + b^3 = 2 → a + b ≤ 2 := by
  sorry

end cube_sum_implies_sum_bound_l2170_217065


namespace bob_winning_strategy_l2170_217066

/-- Represents the state of the game with the number of beads -/
structure GameState where
  beads : Nat
  deriving Repr

/-- Represents a player in the game -/
inductive Player
  | Alice
  | Bob
  deriving Repr

/-- Defines a valid move in the game -/
def validMove (s : GameState) : Prop :=
  s.beads > 1

/-- Defines the next player's turn -/
def nextPlayer : Player → Player
  | Player.Alice => Player.Bob
  | Player.Bob => Player.Alice

/-- Theorem stating that Bob has a winning strategy -/
theorem bob_winning_strategy :
  ∀ (initialBeads : Nat),
    initialBeads % 2 = 1 →
    ∃ (strategy : Player → GameState → Nat),
      ∀ (game : GameState),
        game.beads = initialBeads →
        ¬(∃ (aliceStrategy : Player → GameState → Nat),
          ∀ (state : GameState),
            validMove state →
            (state.beads % 2 = 1 → 
              validMove {beads := state.beads - strategy Player.Bob state} ∧
              validMove {beads := strategy Player.Bob state}) ∧
            (state.beads % 2 = 0 →
              validMove {beads := state.beads - aliceStrategy Player.Alice state} ∧
              validMove {beads := aliceStrategy Player.Alice state})) :=
sorry

#check bob_winning_strategy

end bob_winning_strategy_l2170_217066


namespace triangle_tangent_circles_l2170_217039

/-- Given a triangle with side lengths a, b, and c, there exist radii r₁, r₂, and r₃ for circles
    centered at the triangle's vertices that satisfy both external and internal tangency conditions. -/
theorem triangle_tangent_circles
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) :
  ∃ (r₁ r₂ r₃ : ℝ),
    (r₁ > 0 ∧ r₂ > 0 ∧ r₃ > 0) ∧
    (r₁ + r₂ = c ∧ r₂ + r₃ = a ∧ r₃ + r₁ = b) ∧
    ∃ (r₁' r₂' r₃' : ℝ),
      (r₁' > 0 ∧ r₂' > 0 ∧ r₃' > 0) ∧
      (r₃' - r₂' = a ∧ r₃' - r₁' = b ∧ r₁' + r₂' = c) :=
by sorry

end triangle_tangent_circles_l2170_217039


namespace pencils_to_library_l2170_217054

theorem pencils_to_library (total_pencils : Nat) (num_classrooms : Nat) 
    (h1 : total_pencils = 935) 
    (h2 : num_classrooms = 9) : 
  total_pencils % num_classrooms = 8 := by
  sorry

end pencils_to_library_l2170_217054


namespace coeff_x_cubed_in_expansion_coeff_x_cubed_proof_l2170_217089

/-- The coefficient of x^3 in the expansion of (1-x)^10 is -120 -/
theorem coeff_x_cubed_in_expansion : Int :=
  -120

/-- The binomial coefficient (10 choose 3) -/
def binomial_coeff : Int :=
  120

theorem coeff_x_cubed_proof : coeff_x_cubed_in_expansion = -binomial_coeff := by
  sorry

end coeff_x_cubed_in_expansion_coeff_x_cubed_proof_l2170_217089


namespace range_of_sin_plus_cos_l2170_217010

theorem range_of_sin_plus_cos :
  ∀ y : ℝ, (∃ x : ℝ, y = Real.sin x + Real.cos x) ↔ -Real.sqrt 2 ≤ y ∧ y ≤ Real.sqrt 2 := by
  sorry

end range_of_sin_plus_cos_l2170_217010


namespace thermometer_distribution_count_l2170_217047

/-- The number of senior classes -/
def num_classes : ℕ := 10

/-- The total number of thermometers to distribute -/
def total_thermometers : ℕ := 23

/-- The minimum number of thermometers each class must receive -/
def min_thermometers_per_class : ℕ := 2

/-- The number of remaining thermometers after initial distribution -/
def remaining_thermometers : ℕ := total_thermometers - num_classes * min_thermometers_per_class

/-- The number of spaces between items for divider placement -/
def spaces_for_dividers : ℕ := remaining_thermometers - 1

/-- The number of dividers needed -/
def num_dividers : ℕ := num_classes - 1

theorem thermometer_distribution_count :
  (spaces_for_dividers.choose num_dividers) = 220 := by
  sorry

end thermometer_distribution_count_l2170_217047


namespace min_value_floor_sum_l2170_217024

theorem min_value_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ⌊(2*x+y)/z⌋ + ⌊(2*y+z)/x⌋ + ⌊(2*z+x)/y⌋ ≥ 6 :=
by sorry

end min_value_floor_sum_l2170_217024


namespace negative_of_negative_is_positive_l2170_217033

theorem negative_of_negative_is_positive (y : ℝ) (h : y < 0) : -y > 0 := by
  sorry

end negative_of_negative_is_positive_l2170_217033


namespace boy_scout_percentage_l2170_217081

/-- Represents the composition of a group of scouts -/
structure ScoutGroup where
  total : ℝ
  boys : ℝ
  girls : ℝ
  total_is_sum : total = boys + girls

/-- Represents the percentage of scouts with signed permission slips -/
structure PermissionSlips where
  total_percent : ℝ
  boys_percent : ℝ
  girls_percent : ℝ
  total_is_70_percent : total_percent = 0.7
  boys_is_75_percent : boys_percent = 0.75
  girls_is_62_5_percent : girls_percent = 0.625

theorem boy_scout_percentage 
  (group : ScoutGroup) 
  (slips : PermissionSlips) : 
  group.boys / group.total = 0.6 := by
  sorry

end boy_scout_percentage_l2170_217081


namespace smallest_number_divisible_plus_one_l2170_217071

theorem smallest_number_divisible_plus_one (n : ℕ) : n = 1038239 ↔ 
  (∀ m : ℕ, m < n → ¬((m + 1) % 618 = 0 ∧ (m + 1) % 3648 = 0 ∧ (m + 1) % 60 = 0)) ∧
  ((n + 1) % 618 = 0 ∧ (n + 1) % 3648 = 0 ∧ (n + 1) % 60 = 0) :=
by sorry

end smallest_number_divisible_plus_one_l2170_217071


namespace fibonacci_identity_l2170_217042

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_identity (n : ℕ) (h : n ≥ 1) :
  (fib (2 * n - 1))^2 + (fib (2 * n + 1))^2 + 1 = 3 * (fib (2 * n - 1)) * (fib (2 * n + 1)) :=
by sorry

end fibonacci_identity_l2170_217042
