import Mathlib

namespace set_intersection_empty_implies_m_geq_one_l2144_214413

theorem set_intersection_empty_implies_m_geq_one (m : ℝ) : 
  let M : Set ℝ := {x | x ≤ 1}
  let P : Set ℝ := {x | x ≤ m}
  M ∩ (Set.univ \ P) = ∅ → m ≥ 1 := by
sorry

end set_intersection_empty_implies_m_geq_one_l2144_214413


namespace cat_catches_rat_l2144_214476

/-- The time it takes for a cat to catch a rat given their speeds and a head start. -/
theorem cat_catches_rat (cat_speed rat_speed : ℝ) (head_start : ℝ) (catch_time : ℝ) : 
  cat_speed = 90 →
  rat_speed = 36 →
  head_start = 6 →
  catch_time * (cat_speed - rat_speed) = head_start * rat_speed →
  catch_time = 4 :=
by sorry

end cat_catches_rat_l2144_214476


namespace geometric_sequence_sum_l2144_214497

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 3 →
  a 1 + a 3 + a 5 = 21 →
  a 3 + a 5 + a 7 = 42 := by
  sorry

end geometric_sequence_sum_l2144_214497


namespace permutation_order_l2144_214405

/-- The alphabet in its natural order -/
def T₀ : String := "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

/-- The result of applying the permutation to T₀ -/
def T₁ : String := "JQOWIPANTZRCVMYEGSHUFDKBLX"

/-- The result of applying the permutation to T₁ -/
def T₂ : String := "ZGYKTEJMUXSODVLIAHNFPWRQCB"

/-- The permutation function -/
def permutation (s : String) : String :=
  if s = T₀ then T₁
  else if s = T₁ then T₂
  else T₀  -- This else case is not explicitly given in the problem, but needed for completeness

/-- The theorem stating that the order of the permutation is 24 -/
theorem permutation_order :
  ∃ (n : ℕ), n > 0 ∧ (∀ (k : ℕ), k > 0 ∧ k < n → (permutation^[k] T₀ ≠ T₀)) ∧ permutation^[n] T₀ = T₀ ∧ n = 24 := by
  sorry

#check permutation_order

end permutation_order_l2144_214405


namespace sales_theorem_l2144_214416

def sales_problem (sale1 sale2 sale4 sale5 average : ℕ) : Prop :=
  let total := average * 5
  let known_sales := sale1 + sale2 + sale4 + sale5
  let sale3 := total - known_sales
  sale3 = 9455

theorem sales_theorem :
  sales_problem 5700 8550 3850 14045 7800 :=
by
  sorry

end sales_theorem_l2144_214416


namespace max_difference_of_primes_l2144_214402

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem max_difference_of_primes (a b c : ℕ) : 
  (is_prime a ∧ is_prime b ∧ is_prime c ∧
   is_prime (a + b - c) ∧ is_prime (a + c - b) ∧ is_prime (b + c - a) ∧ is_prime (a + b + c) ∧
   (a + b = 800 ∨ a + c = 800 ∨ b + c = 800) ∧
   a ≠ b ∧ a ≠ c ∧ b ≠ c ∧
   a ≠ (a + b - c) ∧ a ≠ (a + c - b) ∧ a ≠ (b + c - a) ∧ a ≠ (a + b + c) ∧
   b ≠ (a + b - c) ∧ b ≠ (a + c - b) ∧ b ≠ (b + c - a) ∧ b ≠ (a + b + c) ∧
   c ≠ (a + b - c) ∧ c ≠ (a + c - b) ∧ c ≠ (b + c - a) ∧ c ≠ (a + b + c) ∧
   (a + b - c) ≠ (a + c - b) ∧ (a + b - c) ≠ (b + c - a) ∧ (a + b - c) ≠ (a + b + c) ∧
   (a + c - b) ≠ (b + c - a) ∧ (a + c - b) ≠ (a + b + c) ∧
   (b + c - a) ≠ (a + b + c)) →
  (∃ d : ℕ, d ≤ 1594 ∧ 
   d = max (a + b + c) (max (a + c - b) (max (b + c - a) (max a (max b c)))) - 
       min (a + b - c) (min a (min b c)) ∧
   ∀ d' : ℕ, d' ≤ d) :=
by sorry

end max_difference_of_primes_l2144_214402


namespace inequality_system_solution_l2144_214454

theorem inequality_system_solution :
  let S := {x : ℝ | 3*x - 2 < 2*(x + 1) ∧ (x - 1)/2 > 1}
  S = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end inequality_system_solution_l2144_214454


namespace max_player_salary_l2144_214426

theorem max_player_salary (n : ℕ) (min_salary max_total : ℝ) :
  n = 25 →
  min_salary = 20000 →
  max_total = 900000 →
  let max_single_salary := max_total - (n - 1) * min_salary
  max_single_salary = 420000 :=
by
  sorry

end max_player_salary_l2144_214426


namespace complex_fraction_magnitude_l2144_214449

theorem complex_fraction_magnitude : Complex.abs ((5 + Complex.I) / (1 - Complex.I)) = Real.sqrt 13 := by
  sorry

end complex_fraction_magnitude_l2144_214449


namespace correlation_coefficient_comparison_l2144_214480

def X : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def Y : List ℝ := [1, 2, 3, 4, 5]
def U : List ℝ := [10, 11.3, 11.8, 12.5, 13]
def V : List ℝ := [5, 4, 3, 2, 1]

def correlation_coefficient (x y : List ℝ) : ℝ := sorry

def r₁ : ℝ := correlation_coefficient X Y
def r₂ : ℝ := correlation_coefficient U V

theorem correlation_coefficient_comparison : r₂ < 0 ∧ r₁ > 0 := by sorry

end correlation_coefficient_comparison_l2144_214480


namespace van_capacity_l2144_214409

theorem van_capacity (students : ℕ) (adults : ℕ) (vans : ℕ) 
  (h1 : students = 40) 
  (h2 : adults = 14) 
  (h3 : vans = 6) :
  (students + adults) / vans = 9 :=
by
  sorry

end van_capacity_l2144_214409


namespace no_c_k_exist_l2144_214410

def S (n : ℕ) : ℚ := 4 * (1 - 1 / (2^n : ℚ))

theorem no_c_k_exist : ¬∃ (c k : ℕ), (S k + 1 - c) / (S k - c) > 2 := by
  sorry

end no_c_k_exist_l2144_214410


namespace sufficient_not_necessary_l2144_214483

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 0 → a^2 + a ≥ 0) ∧ 
  (∃ a, a^2 + a ≥ 0 ∧ a ≤ 0) := by
  sorry

end sufficient_not_necessary_l2144_214483


namespace cheesecake_calories_l2144_214412

/-- Represents a cheesecake with its properties -/
structure Cheesecake where
  calories_per_slice : ℕ
  quarter_slices : ℕ

/-- Calculates the total calories in a cheesecake -/
def total_calories (c : Cheesecake) : ℕ :=
  c.calories_per_slice * (4 * c.quarter_slices)

/-- Proves that the total calories in the given cheesecake is 2800 -/
theorem cheesecake_calories (c : Cheesecake)
    (h1 : c.calories_per_slice = 350)
    (h2 : c.quarter_slices = 2) :
    total_calories c = 2800 := by
  sorry

#eval total_calories { calories_per_slice := 350, quarter_slices := 2 }

end cheesecake_calories_l2144_214412


namespace ball_distribution_probability_ratio_l2144_214456

def numBalls : ℕ := 15
def numBins : ℕ := 5

def probability_equal_distribution : ℚ :=
  (Nat.factorial numBalls) / ((Nat.factorial 3)^5 * numBins^numBalls)

def probability_unequal_distribution : ℚ :=
  (Nat.factorial numBalls) / 
  (Nat.factorial 5 * Nat.factorial 4 * (Nat.factorial 2)^3 * numBins^numBalls)

theorem ball_distribution_probability_ratio :
  (probability_equal_distribution / probability_unequal_distribution) = 48 := by
  sorry

end ball_distribution_probability_ratio_l2144_214456


namespace distance_between_4th_and_28th_red_lights_l2144_214479

/-- Represents the color of a light -/
inductive LightColor
| Blue
| Red

/-- Calculates the position of the nth red light in the sequence -/
def redLightPosition (n : ℕ) : ℕ :=
  let groupSize := 5
  let redLightsPerGroup := 3
  let completeGroups := (n - 1) / redLightsPerGroup
  let positionInGroup := (n - 1) % redLightsPerGroup + 1
  completeGroups * groupSize + positionInGroup + 2

/-- The distance between lights in inches -/
def lightDistance : ℕ := 8

/-- The number of inches in a foot -/
def inchesPerFoot : ℕ := 12

/-- The main theorem stating the distance between the 4th and 28th red lights -/
theorem distance_between_4th_and_28th_red_lights :
  (redLightPosition 28 - redLightPosition 4) * lightDistance / inchesPerFoot = 26 := by
  sorry


end distance_between_4th_and_28th_red_lights_l2144_214479


namespace square_area_error_percentage_l2144_214469

theorem square_area_error_percentage (x : ℝ) (h : x > 0) : 
  let measured_side := 1.18 * x
  let actual_area := x ^ 2
  let calculated_area := measured_side ^ 2
  let area_error_percentage := (calculated_area - actual_area) / actual_area * 100
  area_error_percentage = 39.24 := by
sorry

end square_area_error_percentage_l2144_214469


namespace number_percentage_equality_l2144_214429

theorem number_percentage_equality (x : ℚ) : 
  (30 / 100 : ℚ) * x = (40 / 100 : ℚ) * 40 → x = 160 / 3 := by
  sorry

end number_percentage_equality_l2144_214429


namespace inscribed_sphere_volume_l2144_214452

/-- The volume of a sphere inscribed in a right circular cone -/
theorem inscribed_sphere_volume (d : ℝ) (α : ℝ) (h_d : d = 12) (h_α : α = π/4) :
  let r := d / (2 * Real.tan (α/2))
  (4/3) * π * r^3 = 72 * Real.sqrt 2 * π := by
  sorry

end inscribed_sphere_volume_l2144_214452


namespace central_cell_value_l2144_214495

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

theorem central_cell_value (t : Table) :
  satisfies_conditions t → t.e = 0.00081 := by
  sorry

end central_cell_value_l2144_214495


namespace unique_cube_number_l2144_214424

theorem unique_cube_number : ∃! y : ℕ, 
  (∃ n : ℕ, y = n^3) ∧ 
  (y % 6 = 0) ∧ 
  (50 < y) ∧ 
  (y < 350) :=
by
  -- The proof would go here
  sorry

end unique_cube_number_l2144_214424


namespace selection_theorem_l2144_214443

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

/-- The number of male students --/
def num_males : ℕ := 5

/-- The number of female students --/
def num_females : ℕ := 4

/-- The total number of representatives to be selected --/
def total_representatives : ℕ := 4

/-- The number of ways to select representatives satisfying the given conditions --/
def num_ways : ℕ := 
  choose num_males 2 * choose num_females 2 + 
  choose num_males 3 * choose num_females 1

theorem selection_theorem : num_ways = 100 := by sorry

end selection_theorem_l2144_214443


namespace six_digit_square_last_three_root_l2144_214489

-- Define a function to check if a number is a six-digit square number
def isSixDigitSquare (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ ∃ k : ℕ, n = k^2

-- Define a function to get the last three digits of a number
def lastThreeDigits (n : ℕ) : ℕ :=
  n % 1000

-- Define the main theorem
theorem six_digit_square_last_three_root : 
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, isSixDigitSquare n ∧ lastThreeDigits n = n.sqrt) ∧ 
    s.card = 2 :=
by
  sorry

end six_digit_square_last_three_root_l2144_214489


namespace pocket_probability_change_l2144_214486

-- Define the initial state of the pocket
def initial_red_balls : ℕ := 4
def initial_white_balls : ℕ := 8

-- Define the number of balls removed/added
def balls_changed : ℕ := 6

-- Define the final probability of drawing a red ball
def final_red_probability : ℚ := 5/6

-- Theorem statement
theorem pocket_probability_change :
  let total_balls : ℕ := initial_red_balls + initial_white_balls
  let new_red_balls : ℕ := initial_red_balls + balls_changed
  let new_total_balls : ℕ := total_balls
  (new_red_balls : ℚ) / new_total_balls = final_red_probability := by
  sorry

end pocket_probability_change_l2144_214486


namespace correct_calculation_l2144_214446

theorem correct_calculation (x : ℝ) (h : 6 * x = 42) : 3 * x = 21 := by
  sorry

end correct_calculation_l2144_214446


namespace fruit_ratio_l2144_214408

theorem fruit_ratio (initial_fruits : ℕ) (oranges_left : ℕ) : 
  initial_fruits = 150 → 
  oranges_left = 50 → 
  (oranges_left : ℚ) / (initial_fruits / 2 - oranges_left : ℚ) = 2 := by
  sorry

end fruit_ratio_l2144_214408


namespace reach_50_from_49_l2144_214440

def double (n : ℕ) : ℕ := n * 2

def erase_last_digit (n : ℕ) : ℕ := n / 10

def can_reach (start target : ℕ) : Prop :=
  ∃ (sequence : List (ℕ → ℕ)), 
    (∀ f ∈ sequence, f = double ∨ f = erase_last_digit) ∧
    (sequence.foldl (λ acc f => f acc) start = target)

theorem reach_50_from_49 : can_reach 49 50 := by sorry

end reach_50_from_49_l2144_214440


namespace yellow_ball_fraction_l2144_214457

theorem yellow_ball_fraction (total : ℝ) (h : total > 0) :
  let initial_green := (4/7) * total
  let initial_yellow := total - initial_green
  let new_yellow := 3 * initial_yellow
  let new_green := initial_green * (3/2)
  let new_total := new_yellow + new_green
  new_yellow / new_total = 3/5 := by
sorry

end yellow_ball_fraction_l2144_214457


namespace james_total_socks_l2144_214418

/-- Calculates the total number of socks James has -/
def total_socks (red_pairs : ℕ) : ℕ :=
  let red := red_pairs * 2
  let black := red / 2
  let white := (red + black) * 2
  red + black + white

/-- Proves that James has 180 socks in total -/
theorem james_total_socks : total_socks 20 = 180 := by
  sorry

end james_total_socks_l2144_214418


namespace root_existence_quadratic_root_existence_l2144_214490

theorem root_existence (f : ℝ → ℝ) (h1 : f 1.1 < 0) (h2 : f 1.2 > 0) 
  (h3 : Continuous f) :
  ∃ x : ℝ, x > 1.1 ∧ x < 1.2 ∧ f x = 0 :=
sorry

def f (x : ℝ) : ℝ := x^2 + 12*x - 15

theorem quadratic_root_existence (h1 : f 1.1 < 0) (h2 : f 1.2 > 0) :
  ∃ x : ℝ, x > 1.1 ∧ x < 1.2 ∧ f x = 0 :=
sorry

end root_existence_quadratic_root_existence_l2144_214490


namespace equation_solution_l2144_214464

theorem equation_solution : ∃ x : ℝ, 
  3.5 * ((3.6 * x * 2.50) / (0.12 * 0.09 * 0.5)) = 2800.0000000000005 ∧ 
  abs (x - 0.4799999999999999) < 1e-15 := by
  sorry

end equation_solution_l2144_214464


namespace remainder_3m_mod_5_l2144_214401

theorem remainder_3m_mod_5 (m : ℤ) (h : m % 5 = 2) : (3 * m) % 5 = 1 := by
  sorry

end remainder_3m_mod_5_l2144_214401


namespace bill_money_left_l2144_214428

/-- The amount of fool's gold Bill sells in ounces -/
def foolsGoldSold : ℕ := 8

/-- The price per ounce of fool's gold in dollars -/
def pricePerOunce : ℕ := 9

/-- The fine Bill has to pay in dollars -/
def fine : ℕ := 50

/-- The amount of money Bill is left with after selling fool's gold and paying the fine -/
def moneyLeft : ℕ := foolsGoldSold * pricePerOunce - fine

theorem bill_money_left : moneyLeft = 22 := by
  sorry

end bill_money_left_l2144_214428


namespace shooter_hit_rate_l2144_214411

theorem shooter_hit_rate (shots : ℕ) (prob_hit_at_least_once : ℚ) (hit_rate : ℚ) :
  shots = 4 →
  prob_hit_at_least_once = 80 / 81 →
  (1 - (1 - hit_rate) ^ shots) = prob_hit_at_least_once →
  hit_rate = 2 / 3 := by
  sorry

end shooter_hit_rate_l2144_214411


namespace total_wrappers_eq_49_l2144_214467

/-- The number of wrappers gathered by Andy -/
def andy_wrappers : ℕ := 34

/-- The number of wrappers gathered by Max -/
def max_wrappers : ℕ := 15

/-- The total number of wrappers gathered by Andy and Max -/
def total_wrappers : ℕ := andy_wrappers + max_wrappers

theorem total_wrappers_eq_49 : total_wrappers = 49 := by
  sorry

end total_wrappers_eq_49_l2144_214467


namespace seating_solution_l2144_214487

/-- Represents the seating arrangement problem with changing conditions --/
structure SeatingProblem where
  total_seats : ℕ
  initial_gap : ℕ
  final_gap : ℕ

/-- Calculates the minimum number of occupied seats required --/
def min_occupied_seats (problem : SeatingProblem) : ℕ :=
  sorry

/-- The theorem stating the solution to the specific problem --/
theorem seating_solution :
  let problem : SeatingProblem := {
    total_seats := 150,
    initial_gap := 2,
    final_gap := 1
  }
  min_occupied_seats problem = 57 := by sorry

end seating_solution_l2144_214487


namespace sphere_radius_equal_cylinder_surface_l2144_214488

/-- The radius of a sphere with surface area equal to the curved surface area of a right circular cylinder with height and diameter both 12 cm. -/
theorem sphere_radius_equal_cylinder_surface (h : ℝ) (d : ℝ) (r : ℝ) :
  h = 12 →
  d = 12 →
  4 * Real.pi * r^2 = 2 * Real.pi * (d / 2) * h →
  r = 6 :=
by sorry

end sphere_radius_equal_cylinder_surface_l2144_214488


namespace absolute_value_inequality_l2144_214403

theorem absolute_value_inequality (a b c : ℝ) (h : |a + c| < b) : |a| > |c| - |b| := by
  sorry

end absolute_value_inequality_l2144_214403


namespace toothpicks_in_specific_grid_l2144_214492

/-- The number of toothpicks in a grid with a gap -/
def toothpicks_in_grid_with_gap (length width gap_length gap_width : ℕ) : ℕ :=
  let vertical_toothpicks := (length + 1) * width
  let horizontal_toothpicks := (width + 1) * length
  let gap_vertical_toothpicks := (gap_length + 1) * gap_width
  let gap_horizontal_toothpicks := (gap_width + 1) * gap_length
  vertical_toothpicks + horizontal_toothpicks - gap_vertical_toothpicks - gap_horizontal_toothpicks

/-- Theorem stating the number of toothpicks in the specific grid described in the problem -/
theorem toothpicks_in_specific_grid :
  toothpicks_in_grid_with_gap 70 40 10 5 = 5595 := by
  sorry

end toothpicks_in_specific_grid_l2144_214492


namespace triangle_five_sixths_nine_fourths_l2144_214417

/-- The triangle operation ∆ defined for fractions -/
def triangle (m n p q : ℚ) : ℚ := m^2 * p * (q / n)

/-- Theorem stating that (5/6) ∆ (9/4) = 150 -/
theorem triangle_five_sixths_nine_fourths : 
  triangle (5/6) (5/6) (9/4) (9/4) = 150 := by sorry

end triangle_five_sixths_nine_fourths_l2144_214417


namespace hayley_stickers_l2144_214400

theorem hayley_stickers (num_friends : ℕ) (stickers_per_friend : ℕ) 
  (h1 : num_friends = 9) (h2 : stickers_per_friend = 8) : 
  num_friends * stickers_per_friend = 72 := by
  sorry

end hayley_stickers_l2144_214400


namespace division_by_less_than_one_multiplication_by_greater_than_one_multiply_by_100_equals_divide_by_001_l2144_214499

-- Statement 1
theorem division_by_less_than_one (x y : ℝ) (hx : x > 0) (hy : 0 < y) (hy1 : y < 1) :
  x / y > x :=
sorry

-- Statement 2
theorem multiplication_by_greater_than_one (x y : ℝ) (hy : y > 1) :
  x * y > x :=
sorry

-- Statement 3
theorem multiply_by_100_equals_divide_by_001 (x : ℝ) :
  x * 100 = x / 0.01 :=
sorry

end division_by_less_than_one_multiplication_by_greater_than_one_multiply_by_100_equals_divide_by_001_l2144_214499


namespace sin_cos_ratio_simplification_l2144_214435

theorem sin_cos_ratio_simplification :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) = Real.sqrt 3 :=
by sorry

end sin_cos_ratio_simplification_l2144_214435


namespace unique_five_step_palindrome_l2144_214406

/-- Reverses the digits of a natural number -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := n = reverseDigits n

/-- The transformation step: reversing and adding -/
def transformStep (n : ℕ) : ℕ := n + reverseDigits n

/-- Counts the number of steps needed to reach a palindrome -/
def stepsToParalindrome (n : ℕ) : ℕ := sorry

theorem unique_five_step_palindrome :
  ∃! n : ℕ, 200 ≤ n ∧ n < 300 ∧ ¬isPalindrome n ∧ stepsToParalindrome n = 5 ∧ n = 237 := by
  sorry

end unique_five_step_palindrome_l2144_214406


namespace probability_white_then_red_l2144_214484

def total_marbles : ℕ := 14
def red_marbles : ℕ := 6
def white_marbles : ℕ := 8

theorem probability_white_then_red :
  (white_marbles / total_marbles) * (red_marbles / (total_marbles - 1)) = 24 / 91 := by
sorry

end probability_white_then_red_l2144_214484


namespace regular_polygon_with_18_degree_exterior_angle_has_20_sides_l2144_214491

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides. -/
theorem regular_polygon_with_18_degree_exterior_angle_has_20_sides :
  ∀ n : ℕ, 
  n > 2 → 
  (360 : ℝ) / n = 18 → 
  n = 20 := by
sorry

end regular_polygon_with_18_degree_exterior_angle_has_20_sides_l2144_214491


namespace prize_orders_count_l2144_214462

/-- Represents a bowling tournament with 6 players -/
structure BowlingTournament where
  players : Fin 6

/-- Represents the number of possible outcomes for a single match -/
def match_outcomes : Nat := 2

/-- Represents the number of matches in the tournament -/
def num_matches : Nat := 5

/-- Calculates the total number of possible prize orders -/
def total_outcomes (t : BowlingTournament) : Nat :=
  match_outcomes ^ num_matches

/-- Theorem: The number of possible prize orders is 32 -/
theorem prize_orders_count (t : BowlingTournament) : 
  total_outcomes t = 32 := by
  sorry

end prize_orders_count_l2144_214462


namespace power_equation_solution_l2144_214475

theorem power_equation_solution : ∃ x : ℕ, 
  8 * (32 ^ 10) = 2 ^ x ∧ x = 53 := by
  sorry

end power_equation_solution_l2144_214475


namespace inequality_implications_l2144_214478

theorem inequality_implications (a b : ℝ) (h : a > b) : 
  (a + 2 > b + 2) ∧ 
  (a / 4 > b / 4) ∧ 
  ¬(-3 * a > -3 * b) ∧ 
  ¬(a - 1 < b - 1) := by
sorry

end inequality_implications_l2144_214478


namespace exists_congruent_polygons_l2144_214437

/-- A regular n-gon with colored vertices -/
structure ColoredRegularNGon (n : ℕ) (p : ℕ) where
  (n_ge_6 : n ≥ 6)
  (p_bounds : 3 ≤ p ∧ p < n - p)

/-- The set of red vertices -/
def R (n : ℕ) (p : ℕ) : Set (Fin n) :=
  {i | i.val < p}

/-- The set of black vertices -/
def B (n : ℕ) (p : ℕ) : Set (Fin n) :=
  {i | i.val ≥ p}

/-- Rotation of a vertex by i positions -/
def rotate (n : ℕ) (i : Fin n) (v : Fin n) : Fin n :=
  ⟨(v.val + i.val) % n, by sorry⟩

/-- The theorem to be proved -/
theorem exists_congruent_polygons (n p : ℕ) (h : ColoredRegularNGon n p) :
  ∃ (i : Fin n), (Set.image (rotate n i) (R n p) ∩ B n p).ncard > p / 2 := by
  sorry

end exists_congruent_polygons_l2144_214437


namespace product_equals_nine_twentieths_l2144_214455

theorem product_equals_nine_twentieths : 6 * 0.5 * (3/4) * 0.2 = 9/20 := by
  sorry

end product_equals_nine_twentieths_l2144_214455


namespace factorial_division_l2144_214466

theorem factorial_division :
  (9 * 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 15120 := by
  sorry

end factorial_division_l2144_214466


namespace sector_central_angle_l2144_214477

-- Define the sector
structure Sector where
  perimeter : ℝ
  area : ℝ

-- Define the theorem
theorem sector_central_angle (s : Sector) (h1 : s.perimeter = 12) (h2 : s.area = 8) :
  ∃ (r l : ℝ), r > 0 ∧ l > 0 ∧ 2 * r + l = s.perimeter ∧ 1/2 * r * l = s.area ∧
  (l / r = 1 ∨ l / r = 4) := by
  sorry

#check sector_central_angle

end sector_central_angle_l2144_214477


namespace chess_game_probability_l2144_214461

theorem chess_game_probability (p_draw p_B_win : ℝ) 
  (h_draw : p_draw = 1/2) 
  (h_B_win : p_B_win = 1/3) : 
  1 - p_draw - p_B_win = 1/6 := by
  sorry

end chess_game_probability_l2144_214461


namespace power_of_two_triplets_l2144_214473

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def valid_triplet (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 2, 2), (3, 2, 2), (2, 6, 11), (3, 5, 7),
   (2, 2, 2), (2, 3, 2), (6, 2, 11), (5, 3, 7),
   (2, 2, 2), (2, 2, 3), (11, 6, 2), (7, 5, 3)}

theorem power_of_two_triplets :
  ∀ a b c : ℕ, valid_triplet a b c ↔ (a, b, c) ∈ solution_set :=
sorry

end power_of_two_triplets_l2144_214473


namespace freken_bok_candies_l2144_214433

def initial_candies : ℕ := 111

-- n represents the number of candies before lunch
def candies_before_lunch (n : ℕ) : Prop :=
  n ≤ initial_candies ∧ 
  ∃ (k : ℕ), k * 20 = 11 * n ∧ 
  k ≤ initial_candies

def candies_found_by_freken_bok (n : ℕ) : ℕ :=
  (11 * n) / 60

theorem freken_bok_candies :
  ∃ (n : ℕ), candies_before_lunch n ∧ 
  candies_found_by_freken_bok n = 11 :=
sorry

end freken_bok_candies_l2144_214433


namespace appetizer_price_l2144_214460

theorem appetizer_price (total_spent : ℝ) (entree_percentage : ℝ) (num_appetizers : ℕ) 
  (h_total : total_spent = 50)
  (h_entree : entree_percentage = 0.8)
  (h_appetizers : num_appetizers = 2) : 
  (1 - entree_percentage) * total_spent / num_appetizers = 5 := by
  sorry

end appetizer_price_l2144_214460


namespace catering_budget_theorem_l2144_214442

def total_guests : ℕ := 80
def steak_cost : ℕ := 25
def chicken_cost : ℕ := 18

def catering_budget (chicken_guests : ℕ) : ℕ :=
  chicken_guests * chicken_cost + (3 * chicken_guests) * steak_cost

theorem catering_budget_theorem :
  ∃ (chicken_guests : ℕ),
    chicken_guests + 3 * chicken_guests = total_guests ∧
    catering_budget chicken_guests = 1860 :=
by
  sorry

end catering_budget_theorem_l2144_214442


namespace minimum_balls_for_16_of_one_color_l2144_214420

theorem minimum_balls_for_16_of_one_color : 
  let total_balls : ℕ := 21 + 17 + 24 + 10 + 14 + 14
  let red_balls : ℕ := 21
  let green_balls : ℕ := 17
  let yellow_balls : ℕ := 24
  let blue_balls : ℕ := 10
  let white_balls : ℕ := 14
  let black_balls : ℕ := 14
  ∃ (n : ℕ), n = 84 ∧ 
    (∀ (m : ℕ), m < n → 
      ∃ (r g y b w bl : ℕ), 
        r ≤ red_balls ∧ 
        g ≤ green_balls ∧ 
        y ≤ yellow_balls ∧ 
        b ≤ blue_balls ∧ 
        w ≤ white_balls ∧ 
        bl ≤ black_balls ∧ 
        r + g + y + b + w + bl = m ∧ 
        r < 16 ∧ g < 16 ∧ y < 16 ∧ b < 16 ∧ w < 16 ∧ bl < 16) ∧
    (∀ (k : ℕ), k ≥ n → 
      ∃ (color : ℕ), color ≥ 16 ∧ 
        (color ≤ red_balls ∨ 
         color ≤ green_balls ∨ 
         color ≤ yellow_balls ∨ 
         color ≤ blue_balls ∨ 
         color ≤ white_balls ∨ 
         color ≤ black_balls))
:= by sorry

end minimum_balls_for_16_of_one_color_l2144_214420


namespace school_children_count_l2144_214438

/-- The actual number of children in the school -/
def actual_children : ℕ := 840

/-- The number of absent children -/
def absent_children : ℕ := 420

/-- The number of bananas each child gets initially -/
def initial_bananas : ℕ := 2

/-- The number of extra bananas each child gets due to absences -/
def extra_bananas : ℕ := 2

theorem school_children_count :
  ∀ (total_bananas : ℕ),
  total_bananas = initial_bananas * actual_children ∧
  total_bananas = (initial_bananas + extra_bananas) * (actual_children - absent_children) →
  actual_children = 840 := by
sorry

end school_children_count_l2144_214438


namespace cost_price_is_100_l2144_214468

/-- The cost price of a clothing item, given specific price changes and profit. -/
def cost_price : ℝ → Prop :=
  fun x => 
    let price_after_increase := x * 1.2
    let final_price := price_after_increase * 0.9
    final_price - x = 8

/-- Theorem stating that the cost price is 100 yuan. -/
theorem cost_price_is_100 : cost_price 100 := by
  sorry

end cost_price_is_100_l2144_214468


namespace area_of_rotated_rectangle_curve_l2144_214471

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle defined by its width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Rotation of a point around a center point -/
def rotate90Clockwise (p : Point) (center : Point) : Point :=
  { x := center.x + (p.y - center.y),
    y := center.y - (p.x - center.x) }

/-- The area enclosed by the curve traced by a point on a rectangle under rotations -/
def areaEnclosedByCurve (rect : Rectangle) (initialPoint : Point) (rotationCenters : List Point) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem area_of_rotated_rectangle_curve (rect : Rectangle) (initialPoint : Point) 
  (rotationCenters : List Point) : 
  rect.width = 2 ∧ rect.height = 3 ∧
  initialPoint = { x := 1, y := 1 } ∧
  rotationCenters = [{ x := 2, y := 0 }, { x := 5, y := 0 }, { x := 7, y := 0 }, { x := 10, y := 0 }] →
  areaEnclosedByCurve rect initialPoint rotationCenters = 6 + 7 * Real.pi / 2 := by
  sorry

end area_of_rotated_rectangle_curve_l2144_214471


namespace log_product_eq_two_implies_x_eq_49_l2144_214427

theorem log_product_eq_two_implies_x_eq_49 
  (k x : ℝ) 
  (h : k > 0) 
  (h' : x > 0) 
  (h'' : (Real.log x / Real.log k) * (Real.log k / Real.log 7) = 2) : 
  x = 49 := by
sorry

end log_product_eq_two_implies_x_eq_49_l2144_214427


namespace algebraic_expression_equality_l2144_214451

theorem algebraic_expression_equality (x : ℝ) :
  3 * x^2 - 2 * x - 1 = 2 → -9 * x^2 + 6 * x - 1 = -10 := by
  sorry

end algebraic_expression_equality_l2144_214451


namespace find_x_l2144_214419

theorem find_x (x y : ℝ) (h1 : x + 2*y = 10) (h2 : y = 4) : x = 2 := by
  sorry

end find_x_l2144_214419


namespace optimal_bus_price_with_competition_optimal_bus_price_without_competition_decrease_in_passengers_l2144_214430

/-- Represents the demand function for transportation -/
def demand (p : ℝ) : ℝ := 3000 - 20 * p

/-- Represents the cost function for the bus company -/
def busCompanyCost (y : ℝ) : ℝ := y + 5

/-- Represents the train fare -/
def trainFare : ℝ := 10

/-- Represents the train capacity -/
def trainCapacity : ℝ := 1000

/-- Theorem stating the optimal bus price with train competition -/
theorem optimal_bus_price_with_competition :
  ∃ (p : ℝ), p = 50.5 ∧
  ∀ (p' : ℝ), p' ≠ p →
    let q := demand (min p' trainFare) - trainCapacity
    let revenue := p' * q
    let cost := busCompanyCost q
    revenue - cost ≤ p * (demand (min p trainFare) - trainCapacity) - busCompanyCost (demand (min p trainFare) - trainCapacity) :=
sorry

/-- Theorem stating the optimal bus price without train competition -/
theorem optimal_bus_price_without_competition :
  ∃ (p : ℝ), p = 75.5 ∧
  ∀ (p' : ℝ), p' ≠ p →
    let q := demand p'
    let revenue := p' * q
    let cost := busCompanyCost q
    revenue - cost ≤ p * demand p - busCompanyCost (demand p) :=
sorry

/-- Theorem stating the decrease in total passengers when train service is removed -/
theorem decrease_in_passengers :
  demand trainFare < demand 75.5 :=
sorry

end optimal_bus_price_with_competition_optimal_bus_price_without_competition_decrease_in_passengers_l2144_214430


namespace triangle_perimeter_l2144_214445

/-- Proves that a triangle with inradius 2.5 cm and area 40 cm² has a perimeter of 32 cm -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) :
  r = 2.5 →
  A = 40 →
  A = r * (p / 2) →
  p = 32 := by
sorry

end triangle_perimeter_l2144_214445


namespace quadratic_always_positive_implies_m_greater_than_one_l2144_214474

theorem quadratic_always_positive_implies_m_greater_than_one (m : ℝ) :
  (∀ x : ℝ, x^2 + 2*x + m > 0) → m > 1 := by
  sorry

end quadratic_always_positive_implies_m_greater_than_one_l2144_214474


namespace obtuse_angle_is_in_second_quadrant_l2144_214496

/-- Definition of an obtuse angle -/
def is_obtuse_angle (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

/-- Definition of an angle in the second quadrant -/
def is_in_second_quadrant (θ : ℝ) : Prop := 90 < θ ∧ θ < 180

/-- Theorem: An obtuse angle is an angle in the second quadrant -/
theorem obtuse_angle_is_in_second_quadrant (θ : ℝ) :
  is_obtuse_angle θ ↔ is_in_second_quadrant θ :=
sorry

end obtuse_angle_is_in_second_quadrant_l2144_214496


namespace tangent_lines_to_circle_l2144_214415

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents a line in 2D space using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def is_tangent_to_circle (l : Line) (c : Circle) : Prop :=
  ∃ (p : Point), (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2 ∧
    l.a * p.x + l.b * p.y + l.c = 0

def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem tangent_lines_to_circle (c : Circle) (p : Point) :
  let l1 : Line := { a := 8, b := 15, c := -37 }
  let l2 : Line := { a := 1, b := 0, c := 1 }
  c.center = { x := 2, y := -1 } ∧ c.radius = 3 ∧ p = { x := -1, y := 3 } →
  (is_tangent_to_circle l1 c ∧ point_on_line p l1) ∧
  (is_tangent_to_circle l2 c ∧ point_on_line p l2) :=
by sorry

end tangent_lines_to_circle_l2144_214415


namespace equation_solution_l2144_214463

theorem equation_solution : 
  ∃ x : ℝ, (2 * x + 16 ≥ 0) ∧ 
  ((Real.sqrt (2 * x + 16) - 8 / Real.sqrt (2 * x + 16)) = 4) ∧ 
  (x = 4 * Real.sqrt 3) := by
  sorry

end equation_solution_l2144_214463


namespace divide_fractions_example_l2144_214453

theorem divide_fractions_example : (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 := by
  sorry

end divide_fractions_example_l2144_214453


namespace simple_interest_rate_calculation_l2144_214448

/-- Simple interest rate calculation --/
theorem simple_interest_rate_calculation
  (principal : ℝ)
  (amount : ℝ)
  (time : ℝ)
  (h_principal : principal = 8000)
  (h_amount : amount = 12500)
  (h_time : time = 7)
  (h_simple_interest : amount - principal = principal * (rate / 100) * time) :
  ∃ rate : ℝ, abs (rate - 8.04) < 0.01 :=
sorry

end simple_interest_rate_calculation_l2144_214448


namespace trapezoid_area_and_perimeter_l2144_214407

/-- Represents a trapezoid EFGH with parallel sides EF and GH -/
structure Trapezoid where
  EH : ℝ
  EF : ℝ
  FG : ℝ
  altitude : ℝ

/-- Calculates the area of a trapezoid -/
def area (t : Trapezoid) : ℝ := sorry

/-- Calculates the perimeter of a trapezoid -/
def perimeter (t : Trapezoid) : ℝ := sorry

/-- Theorem stating the area and perimeter of the specific trapezoid -/
theorem trapezoid_area_and_perimeter :
  let t : Trapezoid := { EH := 25, EF := 65, FG := 30, altitude := 18 }
  area t = 1386 ∧ perimeter t = 209 := by sorry

end trapezoid_area_and_perimeter_l2144_214407


namespace green_jelly_bean_probability_l2144_214465

/-- Represents the count of jelly beans for each color -/
structure JellyBeanCount where
  red : ℕ
  green : ℕ
  yellow : ℕ
  blue : ℕ
  black : ℕ

/-- Calculates the total number of jelly beans -/
def totalJellyBeans (count : JellyBeanCount) : ℕ :=
  count.red + count.green + count.yellow + count.blue + count.black

/-- Calculates the probability of selecting a green jelly bean -/
def probabilityGreen (count : JellyBeanCount) : ℚ :=
  count.green / (totalJellyBeans count)

/-- Theorem: The probability of selecting a green jelly bean from the given bag is 5/22 -/
theorem green_jelly_bean_probability :
  let bag := JellyBeanCount.mk 8 10 9 12 5
  probabilityGreen bag = 5 / 22 := by
  sorry

end green_jelly_bean_probability_l2144_214465


namespace function_properties_l2144_214439

/-- Given a function f and a positive real number ω, proves properties about f -/
theorem function_properties (f : ℝ → ℝ) (ω : ℝ) (h_ω : ω > 0) 
  (h_f : ∀ x, f x = Real.sqrt 3 * Real.sin (2 * ω * x) - Real.cos (2 * ω * x))
  (h_dist : ∀ x, f (x + π / (4 * ω)) = f x) : 
  (∀ x, f x = 2 * Real.sin (2 * x - π / 6)) ∧ 
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ 2) := by
sorry

end function_properties_l2144_214439


namespace girls_in_class_l2144_214458

theorem girls_in_class (total_students : ℕ) (girls_ratio : ℕ) (boys_ratio : ℕ)
  (h1 : total_students = 35)
  (h2 : girls_ratio = 3)
  (h3 : boys_ratio = 4) :
  (girls_ratio * total_students) / (girls_ratio + boys_ratio) = 15 := by
  sorry

end girls_in_class_l2144_214458


namespace students_in_class_l2144_214421

theorem students_in_class (n : ℕ) : 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n + 2 = 4 * k) ∧
  (∃ l : ℕ, n + 3 = 5 * l) ∧
  (∃ m : ℕ, n + 4 = 6 * m) ↔ 
  (n = 122 ∨ n = 182) :=
by sorry

end students_in_class_l2144_214421


namespace min_value_sum_reciprocals_l2144_214498

theorem min_value_sum_reciprocals (n : ℕ+) (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) ≥ 1 ∧
  ((1 / (1 + a ^ n.val)) + (1 / (1 + b ^ n.val)) = 1 ↔ a = 1 ∧ b = 1) :=
by sorry

end min_value_sum_reciprocals_l2144_214498


namespace max_trailing_zeros_l2144_214425

theorem max_trailing_zeros (a b c : ℕ) (sum_condition : a + b + c = 1003) :
  ∀ n : ℕ, (∃ k : ℕ, a * b * c = k * 10^n) → n ≤ 7 :=
sorry

end max_trailing_zeros_l2144_214425


namespace square_area_from_points_l2144_214441

/-- The area of a square with adjacent points (4,3) and (5,7) is 17 -/
theorem square_area_from_points : 
  let p1 : ℝ × ℝ := (4, 3)
  let p2 : ℝ × ℝ := (5, 7)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 17 := by sorry

end square_area_from_points_l2144_214441


namespace intersection_points_count_l2144_214444

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x

def g (x : ℝ) : ℝ := x^2 - 4*x + 5

theorem intersection_points_count :
  ∃ (a b : ℝ), a ≠ b ∧ f a = g a ∧ f b = g b ∧
  ∀ (x : ℝ), f x = g x → x = a ∨ x = b :=
sorry

end intersection_points_count_l2144_214444


namespace problem_statement_l2144_214459

def f (m x : ℝ) : ℝ := m * x^2 + (1 - m) * x + m - 2

theorem problem_statement :
  (∀ m : ℝ, (∀ x : ℝ, f m x + 2 ≥ 0) ↔ m ≥ 1/3) ∧
  (∀ m : ℝ, m < 0 →
    (∀ x : ℝ, f m x < m - 1 ↔
      (m ≤ -1 ∧ (x < -1/m ∨ x > 1)) ∨
      (-1 < m ∧ m < 0 ∧ (x < 1 ∨ x > -1/m)))) :=
by sorry

end problem_statement_l2144_214459


namespace system_of_equations_solution_l2144_214432

theorem system_of_equations_solution (a b c : ℝ) : 
  (a - b = 3) → 
  (a^2 + b^2 = 31) → 
  (a + 2*b - c = 5) → 
  (a*b - c = 37/2) := by
sorry

end system_of_equations_solution_l2144_214432


namespace set_intersection_empty_implies_m_range_l2144_214436

-- Define the sets M and N
def M : Set ℝ := {x | x^2 + 5*x - 14 < 0}
def N (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 3}

-- State the theorem
theorem set_intersection_empty_implies_m_range :
  ∀ m : ℝ, (M ∩ N m = ∅) → (m ≤ -10 ∨ m ≥ 2) :=
by sorry

end set_intersection_empty_implies_m_range_l2144_214436


namespace tracis_road_trip_l2144_214422

/-- Traci's road trip problem -/
theorem tracis_road_trip (total_distance : ℝ) (remaining_distance : ℝ) (x : ℝ) : 
  total_distance = 600 →
  remaining_distance = 300 →
  remaining_distance = total_distance - x * total_distance - (1/4) * (total_distance - x * total_distance) →
  x = 1/3 := by
  sorry

end tracis_road_trip_l2144_214422


namespace alcohol_concentration_after_addition_l2144_214414

/-- Proves that adding 2.4 liters of pure alcohol to a 6-liter solution that is 30% alcohol 
    results in a solution that is 50% alcohol. -/
theorem alcohol_concentration_after_addition 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (added_alcohol : ℝ) 
  (target_concentration : ℝ) : 
  initial_volume = 6 →
  initial_concentration = 0.3 →
  added_alcohol = 2.4 →
  target_concentration = 0.5 →
  (initial_volume * initial_concentration + added_alcohol) / (initial_volume + added_alcohol) = target_concentration := by
  sorry

#check alcohol_concentration_after_addition

end alcohol_concentration_after_addition_l2144_214414


namespace video_votes_l2144_214494

theorem video_votes (total_votes : ℕ) (score : ℤ) (like_percent : ℚ) (dislike_percent : ℚ) : 
  score = 120 ∧ 
  like_percent = 58 / 100 ∧ 
  dislike_percent = 30 / 100 ∧ 
  (like_percent - dislike_percent) * total_votes = score →
  total_votes = 429 := by
sorry

end video_votes_l2144_214494


namespace smallest_number_with_special_sums_l2144_214485

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a list of natural numbers all have the same digit sum -/
def same_digit_sum (list : List ℕ) : Prop := sorry

/-- Theorem stating that 10010 is the smallest natural number satisfying the given conditions -/
theorem smallest_number_with_special_sums : 
  ∀ n : ℕ, n < 10010 → 
  ¬(∃ (list1 : List ℕ) (list2 : List ℕ), 
    list1.length = 2002 ∧ 
    list2.length = 2003 ∧ 
    same_digit_sum list1 ∧ 
    same_digit_sum list2 ∧ 
    list1.sum = n ∧ 
    list2.sum = n) ∧ 
  ∃ (list1 : List ℕ) (list2 : List ℕ), 
    list1.length = 2002 ∧ 
    list2.length = 2003 ∧ 
    same_digit_sum list1 ∧ 
    same_digit_sum list2 ∧ 
    list1.sum = 10010 ∧ 
    list2.sum = 10010 := by
  sorry

end smallest_number_with_special_sums_l2144_214485


namespace geometric_sum_example_l2144_214470

/-- The sum of the first n terms of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Proof that the sum of the first 8 terms of a geometric sequence
    with first term 1/4 and common ratio 2 is equal to 255/4 -/
theorem geometric_sum_example :
  geometric_sum (1/4) 2 8 = 255/4 := by
  sorry

end geometric_sum_example_l2144_214470


namespace range_of_a_l2144_214481

def P : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def M (a : ℝ) : Set ℝ := {-a, a}

theorem range_of_a (a : ℝ) : P ∪ M a = P → a ∈ P := by
  sorry

end range_of_a_l2144_214481


namespace domino_distribution_l2144_214472

theorem domino_distribution (total_dominoes : Nat) (num_players : Nat) 
  (h1 : total_dominoes = 28) (h2 : num_players = 4) :
  total_dominoes / num_players = 7 := by
  sorry

end domino_distribution_l2144_214472


namespace puppy_weight_is_2_5_l2144_214431

/-- The weight of the puppy in pounds -/
def puppy_weight : ℝ := 2.5

/-- The weight of the smaller cat in pounds -/
def smaller_cat_weight : ℝ := 7.5

/-- The weight of the larger cat in pounds -/
def larger_cat_weight : ℝ := 20

/-- Theorem stating that the weight of the puppy is 2.5 pounds given the conditions -/
theorem puppy_weight_is_2_5 :
  (puppy_weight + smaller_cat_weight + larger_cat_weight = 30) ∧
  (puppy_weight + larger_cat_weight = 3 * smaller_cat_weight) ∧
  (puppy_weight + smaller_cat_weight = larger_cat_weight - 10) →
  puppy_weight = 2.5 := by
  sorry

end puppy_weight_is_2_5_l2144_214431


namespace jerome_solution_l2144_214482

def jerome_problem (initial_money : ℕ) (given_to_meg : ℕ) (given_to_bianca : ℕ) (money_left : ℕ) : Prop :=
  initial_money / 2 = 43 ∧
  given_to_meg = 8 ∧
  money_left = 54 ∧
  initial_money = given_to_meg + given_to_bianca + money_left ∧
  given_to_bianca / given_to_meg = 3

theorem jerome_solution :
  ∃ (initial_money given_to_meg given_to_bianca money_left : ℕ),
    jerome_problem initial_money given_to_meg given_to_bianca money_left :=
by
  sorry

end jerome_solution_l2144_214482


namespace polynomial_identity_sum_l2144_214450

theorem polynomial_identity_sum (b₁ b₂ b₃ b₄ c₁ c₂ c₃ c₄ : ℝ) :
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃) * (x^2 + b₄*x + c₄)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ + b₄*c₄ = -1 := by
  sorry

end polynomial_identity_sum_l2144_214450


namespace max_salary_theorem_l2144_214493

/-- Represents a baseball team with salary constraints -/
structure BaseballTeam where
  players : ℕ
  minSalary : ℕ
  maxTotalSalary : ℕ

/-- Calculates the maximum possible salary for a single player -/
def maxSinglePlayerSalary (team : BaseballTeam) : ℕ :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player in the given conditions -/
theorem max_salary_theorem (team : BaseballTeam) 
  (h1 : team.players = 18)
  (h2 : team.minSalary = 20000)
  (h3 : team.maxTotalSalary = 800000) :
  maxSinglePlayerSalary team = 460000 := by
  sorry

#eval maxSinglePlayerSalary ⟨18, 20000, 800000⟩

end max_salary_theorem_l2144_214493


namespace truck_tire_usage_l2144_214434

/-- Calculates the number of miles each tire is used on a truck -/
def miles_per_tire (total_miles : ℕ) (total_tires : ℕ) (active_tires : ℕ) : ℚ :=
  (total_miles * active_tires : ℚ) / total_tires

theorem truck_tire_usage :
  let total_miles : ℕ := 36000
  let total_tires : ℕ := 6
  let active_tires : ℕ := 5
  miles_per_tire total_miles total_tires active_tires = 30000 := by
sorry

#eval miles_per_tire 36000 6 5

end truck_tire_usage_l2144_214434


namespace competition_result_l2144_214423

-- Define the participants
inductive Participant
| Olya
| Oleg
| Polya
| Pasha

-- Define the places
inductive Place
| First
| Second
| Third
| Fourth

def is_odd_place (p : Place) : Bool :=
  match p with
  | Place.First => true
  | Place.Third => true
  | _ => false

def is_consecutive (p1 p2 : Place) : Bool :=
  match p1, p2 with
  | Place.First, Place.Second => true
  | Place.Second, Place.Third => true
  | Place.Third, Place.Fourth => true
  | Place.Second, Place.First => true
  | Place.Third, Place.Second => true
  | Place.Fourth, Place.Third => true
  | _, _ => false

def starts_with_O (p : Participant) : Bool :=
  match p with
  | Participant.Olya => true
  | Participant.Oleg => true
  | _ => false

def is_boy (p : Participant) : Bool :=
  match p with
  | Participant.Oleg => true
  | Participant.Pasha => true
  | _ => false

-- The main theorem
theorem competition_result :
  ∃! (result : Participant → Place),
    (∀ p, ∃! place, result p = place) ∧
    (∀ place, ∃! p, result p = place) ∧
    (∃! p, (result p = Place.First ∧ starts_with_O p) ∨
           (result p = Place.Second ∧ ¬starts_with_O p) ∨
           (result p = Place.Third ∧ ¬starts_with_O p) ∨
           (result p = Place.Fourth ∧ ¬starts_with_O p)) ∧
    (result Participant.Oleg = Place.First ∧
     result Participant.Olya = Place.Second ∧
     result Participant.Polya = Place.Third ∧
     result Participant.Pasha = Place.Fourth) :=
by sorry


end competition_result_l2144_214423


namespace side_length_6_sufficient_not_necessary_l2144_214404

structure IsoscelesTriangle where
  x : ℝ
  y : ℝ
  perimeter_eq : 2 * x + y = 16
  base_eq : y = x + 1

def has_side_length_6 (t : IsoscelesTriangle) : Prop :=
  t.x = 6 ∨ t.y = 6

theorem side_length_6_sufficient_not_necessary (t : IsoscelesTriangle) :
  (∃ (t' : IsoscelesTriangle), has_side_length_6 t') ∧
  ¬(∀ (t' : IsoscelesTriangle), has_side_length_6 t') :=
sorry

end side_length_6_sufficient_not_necessary_l2144_214404


namespace johns_total_expenses_l2144_214447

/-- The number of days in John's original tour program -/
def original_days : ℕ := 20

/-- The number of additional days if John extends his trip -/
def additional_days : ℕ := 4

/-- The amount by which John must reduce his daily expenses if he extends his trip -/
def expense_reduction : ℕ := 3

/-- John's total expenses remain the same whether he stays for the original duration or extends his trip -/
axiom expense_equality (daily_expense : ℕ) :
  original_days * daily_expense = (original_days + additional_days) * (daily_expense - expense_reduction)

theorem johns_total_expenses : ∃ (total_expense : ℕ), total_expense = 360 := by
  sorry

end johns_total_expenses_l2144_214447
