import Mathlib

namespace NUMINAMATH_CALUDE_number_equation_solution_l3347_334770

theorem number_equation_solution : 
  ∃ x : ℝ, (3 * x = 2 * x - 7) ∧ (x = -7) := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3347_334770


namespace NUMINAMATH_CALUDE_jimin_remaining_distance_l3347_334703

/-- Calculates the remaining distance to travel given initial conditions. -/
def remaining_distance (speed : ℝ) (time : ℝ) (total_distance : ℝ) : ℝ :=
  total_distance - speed * time

/-- Proves that given the initial conditions, the remaining distance is 180 km. -/
theorem jimin_remaining_distance :
  remaining_distance 60 2 300 = 180 := by
  sorry

end NUMINAMATH_CALUDE_jimin_remaining_distance_l3347_334703


namespace NUMINAMATH_CALUDE_square_fraction_count_l3347_334769

theorem square_fraction_count : 
  ∃! (s : Finset Int), 
    (∀ n ∈ s, ∃ k : Int, (n : ℚ) / (25 - n) = k^2) ∧ 
    (∀ n ∉ s, ¬∃ k : Int, (n : ℚ) / (25 - n) = k^2) ∧ 
    s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_count_l3347_334769


namespace NUMINAMATH_CALUDE_jakes_weight_l3347_334750

theorem jakes_weight (jake_weight sister_weight : ℝ) : 
  jake_weight - 12 = 2 * sister_weight →
  jake_weight + sister_weight = 156 →
  jake_weight = 108 := by
sorry

end NUMINAMATH_CALUDE_jakes_weight_l3347_334750


namespace NUMINAMATH_CALUDE_fraction_sum_equals_point_three_l3347_334715

theorem fraction_sum_equals_point_three :
  5 / 50 + 4 / 40 + 6 / 60 = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equals_point_three_l3347_334715


namespace NUMINAMATH_CALUDE_teams_of_four_from_seven_l3347_334714

theorem teams_of_four_from_seven (n : ℕ) (k : ℕ) : n = 7 → k = 4 → Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_teams_of_four_from_seven_l3347_334714


namespace NUMINAMATH_CALUDE_division_problem_l3347_334772

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 122 →
  divisor = 20 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  quotient = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3347_334772


namespace NUMINAMATH_CALUDE_solution_of_equation_l3347_334779

theorem solution_of_equation (x : ℝ) : (2 / x = 1 / (x + 1)) ↔ (x = -2) := by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l3347_334779


namespace NUMINAMATH_CALUDE_sea_turtle_shell_age_l3347_334739

/-- Converts an octal digit to decimal --/
def octal_to_decimal (digit : Nat) : Nat :=
  if digit < 8 then digit else 0

/-- Converts an octal number to decimal --/
def octal_to_decimal_full (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + octal_to_decimal digit * 8^i) 0

theorem sea_turtle_shell_age :
  octal_to_decimal_full [4, 5, 7, 3] = 2028 := by
  sorry

end NUMINAMATH_CALUDE_sea_turtle_shell_age_l3347_334739


namespace NUMINAMATH_CALUDE_divisors_of_210_l3347_334738

theorem divisors_of_210 : Finset.card (Nat.divisors 210) = 16 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_210_l3347_334738


namespace NUMINAMATH_CALUDE_salmon_migration_multiple_l3347_334734

/-- 
Given an initial number of salmons and the current number of salmons in a river,
calculate the multiple of the initial number that migrated to the river.
-/
theorem salmon_migration_multiple (initial : ℕ) (current : ℕ) : 
  initial = 500 → current = 5500 → (current - initial) / initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_salmon_migration_multiple_l3347_334734


namespace NUMINAMATH_CALUDE_dice_coloring_probability_l3347_334748

/-- Represents the number of faces on a die -/
def numFaces : ℕ := 6

/-- Represents the number of color options for each face -/
def numColors : ℕ := 3

/-- Represents a coloring of a die -/
def DieColoring := Fin numFaces → Fin numColors

/-- Represents whether two die colorings are equivalent under rotation -/
def areEquivalentUnderRotation (d1 d2 : DieColoring) : Prop :=
  ∃ (rotation : Equiv.Perm (Fin numFaces)), ∀ i, d1 i = d2 (rotation i)

/-- The total number of ways to color two dice -/
def totalColorings : ℕ := numColors^numFaces * numColors^numFaces

/-- The number of ways to color two dice that are equivalent under rotation -/
def equivalentColorings : ℕ := 8425

theorem dice_coloring_probability :
  (equivalentColorings : ℚ) / totalColorings = 8425 / 531441 := by sorry

end NUMINAMATH_CALUDE_dice_coloring_probability_l3347_334748


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_digit_difference_l3347_334794

/-- Given two different digits C and D where C > D, prove that the smallest prime factor
    of the difference between the two-digit number CD and its reverse DC is 3. -/
theorem smallest_prime_factor_of_digit_difference (C D : ℕ) : 
  C ≠ D → C > D → C < 10 → D < 10 → Nat.minFac (10 * C + D - (10 * D + C)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_digit_difference_l3347_334794


namespace NUMINAMATH_CALUDE_green_tea_cost_july_l3347_334744

/-- The cost of green tea in July given initial prices and price changes -/
theorem green_tea_cost_july (initial_cost : ℝ) 
  (h1 : initial_cost > 0) 
  (h2 : 3 * (0.1 * initial_cost + 2 * initial_cost) / 2 = 3.15) : 
  0.1 * initial_cost = 0.1 := by sorry

end NUMINAMATH_CALUDE_green_tea_cost_july_l3347_334744


namespace NUMINAMATH_CALUDE_min_running_time_l3347_334706

/-- Proves the minimum running time to cover a given distance within a time limit -/
theorem min_running_time 
  (total_distance : ℝ) 
  (time_limit : ℝ) 
  (walking_speed : ℝ) 
  (running_speed : ℝ) 
  (h1 : total_distance = 2.1) 
  (h2 : time_limit = 18) 
  (h3 : walking_speed = 90) 
  (h4 : running_speed = 210) :
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ time_limit ∧ 
  running_speed * x + walking_speed * (time_limit - x) ≥ total_distance * 1000 :=
sorry

end NUMINAMATH_CALUDE_min_running_time_l3347_334706


namespace NUMINAMATH_CALUDE_alan_market_expenditure_l3347_334781

/-- The total amount Alan spent at the market --/
def total_spent (egg_count : ℕ) (egg_price : ℕ) (chicken_count : ℕ) (chicken_price : ℕ) : ℕ :=
  egg_count * egg_price + chicken_count * chicken_price

/-- Theorem stating that Alan spent $88 at the market --/
theorem alan_market_expenditure :
  total_spent 20 2 6 8 = 88 := by
  sorry

end NUMINAMATH_CALUDE_alan_market_expenditure_l3347_334781


namespace NUMINAMATH_CALUDE_family_egg_count_l3347_334704

/-- Calculates the final number of eggs a family has after various events --/
theorem family_egg_count (initial_eggs : ℚ) 
                          (mother_used : ℚ) 
                          (father_used : ℚ) 
                          (chicken1_laid : ℚ) 
                          (chicken2_laid : ℚ) 
                          (chicken3_laid : ℚ) 
                          (chicken4_laid : ℚ) 
                          (oldest_child_took : ℚ) 
                          (youngest_child_broke : ℚ) : 
  initial_eggs = 25 ∧ 
  mother_used = 7.5 ∧ 
  father_used = 2.5 ∧ 
  chicken1_laid = 2.5 ∧ 
  chicken2_laid = 3 ∧ 
  chicken3_laid = 4.5 ∧ 
  chicken4_laid = 1 ∧ 
  oldest_child_took = 1.5 ∧ 
  youngest_child_broke = 0.5 → 
  initial_eggs - (mother_used + father_used) + 
  (chicken1_laid + chicken2_laid + chicken3_laid + chicken4_laid) - 
  (oldest_child_took + youngest_child_broke) = 24 := by
  sorry


end NUMINAMATH_CALUDE_family_egg_count_l3347_334704


namespace NUMINAMATH_CALUDE_trig_equation_solution_l3347_334712

theorem trig_equation_solution (x : ℝ) : 
  12 * Real.sin x - 5 * Real.cos x = 13 ↔ 
  ∃ k : ℤ, x = π / 2 + Real.arctan (5 / 12) + 2 * k * π :=
sorry

end NUMINAMATH_CALUDE_trig_equation_solution_l3347_334712


namespace NUMINAMATH_CALUDE_courtyard_path_ratio_l3347_334701

theorem courtyard_path_ratio :
  ∀ (t p : ℝ),
  t > 0 →
  p > 0 →
  (400 * t^2) / (400 * (t + 2*p)^2) = 1/4 →
  p/t = 1/2 := by
sorry

end NUMINAMATH_CALUDE_courtyard_path_ratio_l3347_334701


namespace NUMINAMATH_CALUDE_candies_per_friend_l3347_334759

/-- Given 36 candies shared equally among 9 friends, prove that each friend receives 4 candies. -/
theorem candies_per_friend (total_candies : ℕ) (num_friends : ℕ) (candies_per_friend : ℕ) :
  total_candies = 36 →
  num_friends = 9 →
  candies_per_friend = total_candies / num_friends →
  candies_per_friend = 4 := by
  sorry

end NUMINAMATH_CALUDE_candies_per_friend_l3347_334759


namespace NUMINAMATH_CALUDE_value_of_A_l3347_334710

-- Define the letter values as variables
variable (M A T H E : ℤ)

-- State the theorem
theorem value_of_A 
  (h_H : H = 8)
  (h_MATH : M + A + T + H = 32)
  (h_TEAM : T + E + A + M = 40)
  (h_MEET : M + E + E + T = 36) :
  A = 20 := by
sorry

end NUMINAMATH_CALUDE_value_of_A_l3347_334710


namespace NUMINAMATH_CALUDE_point_not_in_region_l3347_334743

def plane_region (x y : ℝ) : Prop := 3*x + 2*y > 3

theorem point_not_in_region :
  ¬(plane_region 0 0) ∧
  (plane_region 1 1) ∧
  (plane_region 0 2) ∧
  (plane_region 2 0) := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_region_l3347_334743


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3347_334700

theorem fraction_to_decimal : (3 : ℚ) / 50 = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3347_334700


namespace NUMINAMATH_CALUDE_bread_rise_time_l3347_334765

/-- The time (in minutes) Mark lets the bread rise each time -/
def rise_time : ℕ := sorry

/-- The total time (in minutes) to make bread -/
def total_time : ℕ := 280

/-- The time (in minutes) spent kneading -/
def kneading_time : ℕ := 10

/-- The time (in minutes) spent baking -/
def baking_time : ℕ := 30

/-- Theorem stating that the rise time is 120 minutes -/
theorem bread_rise_time : rise_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_bread_rise_time_l3347_334765


namespace NUMINAMATH_CALUDE_sqrt_n_squared_minus_np_integer_l3347_334721

theorem sqrt_n_squared_minus_np_integer (p : ℕ) (hp : Prime p) (hodd : Odd p) :
  ∃! n : ℕ, n > 0 ∧ ∃ k : ℕ, k > 0 ∧ n^2 - n*p = k^2 ∧ n = ((p + 1)^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_n_squared_minus_np_integer_l3347_334721


namespace NUMINAMATH_CALUDE_garden_rectangle_length_l3347_334731

theorem garden_rectangle_length :
  ∀ (perimeter width length base_triangle height_triangle : ℝ),
    perimeter = 480 →
    width = 2 * base_triangle →
    base_triangle = 50 →
    height_triangle = 100 →
    perimeter = 2 * (length + width) →
    length = 140 := by
  sorry

end NUMINAMATH_CALUDE_garden_rectangle_length_l3347_334731


namespace NUMINAMATH_CALUDE_jans_cable_sections_l3347_334705

theorem jans_cable_sections (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hxy : y < x) :
  z = (51 * x) / (160 * y) → z = ((51 : ℕ) / 160) * (x / y) := by
sorry

end NUMINAMATH_CALUDE_jans_cable_sections_l3347_334705


namespace NUMINAMATH_CALUDE_words_with_b_count_l3347_334787

/-- The number of letters in the alphabet -/
def n : ℕ := 5

/-- The length of the words -/
def k : ℕ := 4

/-- The total number of possible words -/
def total_words : ℕ := n ^ k

/-- The number of words without the letter B -/
def words_without_b : ℕ := (n - 1) ^ k

/-- The number of words with at least one B -/
def words_with_b : ℕ := total_words - words_without_b

theorem words_with_b_count : words_with_b = 369 := by
  sorry

end NUMINAMATH_CALUDE_words_with_b_count_l3347_334787


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3347_334725

-- Define the sets P and Q
def P : Set ℝ := {y | ∃ x, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x, y = -x + 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {y | y ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l3347_334725


namespace NUMINAMATH_CALUDE_negative_four_less_than_negative_sqrt_fourteen_l3347_334747

theorem negative_four_less_than_negative_sqrt_fourteen : -4 < -Real.sqrt 14 := by
  sorry

end NUMINAMATH_CALUDE_negative_four_less_than_negative_sqrt_fourteen_l3347_334747


namespace NUMINAMATH_CALUDE_exactly_two_even_dice_probability_l3347_334707

def numDice : ℕ := 5
def numFaces : ℕ := 12

def probEven : ℚ := 1 / 2

def probExactlyTwoEven : ℚ := (numDice.choose 2 : ℚ) * probEven ^ 2 * (1 - probEven) ^ (numDice - 2)

theorem exactly_two_even_dice_probability :
  probExactlyTwoEven = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_even_dice_probability_l3347_334707


namespace NUMINAMATH_CALUDE_f_solution_sets_l3347_334728

/-- The function f(x) = ax^2 - (a+c)x + c -/
def f (a c x : ℝ) : ℝ := a * x^2 - (a + c) * x + c

theorem f_solution_sets :
  /- Part 1 -/
  (∀ a c : ℝ, a > 0 → (∀ x : ℝ, f a c x = f a c (-2 - x)) →
    {x : ℝ | f a c x > 0} = {x : ℝ | x < -3 ∨ x > 1}) ∧
  /- Part 2 -/
  (∀ a : ℝ, a ≥ 0 → f a 1 0 = 1 →
    {x : ℝ | f a 1 x > 0} =
      if a = 0 then {x : ℝ | x > 1}
      else if 0 < a ∧ a < 1 then {x : ℝ | 1 < x ∧ x < 1/a}
      else if a > 1 then {x : ℝ | 1/a < x ∧ x < 1}
      else ∅) :=
by sorry

end NUMINAMATH_CALUDE_f_solution_sets_l3347_334728


namespace NUMINAMATH_CALUDE_different_color_chips_probability_l3347_334741

/-- The probability of drawing two chips of different colors from a bag containing
    7 blue chips, 5 red chips, 4 yellow chips, and 3 green chips, when drawing
    with replacement. -/
theorem different_color_chips_probability
  (blue : ℕ) (red : ℕ) (yellow : ℕ) (green : ℕ)
  (h_blue : blue = 7)
  (h_red : red = 5)
  (h_yellow : yellow = 4)
  (h_green : green = 3) :
  let total := blue + red + yellow + green
  (blue * (total - blue) + red * (total - red) + yellow * (total - yellow) + green * (total - green)) / (total * total) = 262 / 361 :=
by sorry

end NUMINAMATH_CALUDE_different_color_chips_probability_l3347_334741


namespace NUMINAMATH_CALUDE_range_of_a_l3347_334793

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (-1) 1, a * x + 1 > 0) → a ∈ Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3347_334793


namespace NUMINAMATH_CALUDE_parallel_vectors_tan_2alpha_l3347_334784

theorem parallel_vectors_tan_2alpha (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi)
  (h2 : (Real.cos α - 5) * Real.cos α + Real.sin α * (Real.sin α - 5) = 0) :
  Real.tan (2 * α) = 24 / 7 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_tan_2alpha_l3347_334784


namespace NUMINAMATH_CALUDE_a_greater_than_b_squared_l3347_334717

theorem a_greater_than_b_squared (a b : ℝ) (ha : a > 1) (hb1 : b > -1) (hb2 : 1 > b) : a > b^2 := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_squared_l3347_334717


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3347_334758

def M : Set ℝ := {x | 1 + x > 0}
def N : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}

theorem intersection_of_M_and_N : M ∩ N = {x | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3347_334758


namespace NUMINAMATH_CALUDE_system_solution_l3347_334777

theorem system_solution (x y z : ℝ) 
  (eq1 : x + y + z = 10)
  (eq2 : x * z = y^2)
  (eq3 : z^2 + y^2 = x^2) :
  z = 5 - Real.sqrt (Real.sqrt 3125 - 50) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l3347_334777


namespace NUMINAMATH_CALUDE_complex_sixth_power_equation_simplified_polynomial_system_l3347_334795

/-- The complex number z satisfying z^6 = -8 - 8i can be characterized by a system of polynomial equations. -/
theorem complex_sixth_power_equation (z : ℂ) : 
  z^6 = -8 - 8*I ↔ 
  ∃ (x y : ℝ), z = x + y*I ∧ 
    (x^6 - 15*x^4*y^2 + 15*x^2*y^4 - y^6 = -8) ∧
    (6*x^5*y - 20*x^3*y^3 + 6*x*y^5 = -8) :=
by sorry

/-- The system of polynomial equations characterizing the solutions can be further simplified. -/
theorem simplified_polynomial_system (x y : ℝ) :
  (x^6 - 15*x^4*y^2 + 15*x^2*y^4 - y^6 = -8 ∧ 
   6*x^5*y - 20*x^3*y^3 + 6*x*y^5 = -8) ↔
  (x^6 - 15*x^4*y^2 + 15*x^2*y^4 - y^6 = -8 ∧
   x^4 - 10*x^2*y^2 + y^4 = -4/3) :=
by sorry

end NUMINAMATH_CALUDE_complex_sixth_power_equation_simplified_polynomial_system_l3347_334795


namespace NUMINAMATH_CALUDE_smallest_largest_product_l3347_334733

def digits : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_three_digit (n : Nat) : Prop := 100 ≤ n ∧ n ≤ 999

def uses_all_digits (a b c : Nat) : Prop :=
  (digits.card = 9) ∧
  (Finset.card (Finset.image (λ d => d % 10) {a, b, c, a / 10, b / 10, c / 10, a / 100, b / 100, c / 100}) = 9)

theorem smallest_largest_product :
  ∀ a b c : Nat,
  is_three_digit a ∧ is_three_digit b ∧ is_three_digit c →
  uses_all_digits a b c →
  (∀ x y z : Nat, is_three_digit x ∧ is_three_digit y ∧ is_three_digit z → uses_all_digits x y z → a * b * c ≤ x * y * z) ∧
  (∀ x y z : Nat, is_three_digit x ∧ is_three_digit y ∧ is_three_digit z → uses_all_digits x y z → x * y * z ≤ 941 * 852 * 763) :=
by sorry

end NUMINAMATH_CALUDE_smallest_largest_product_l3347_334733


namespace NUMINAMATH_CALUDE_rectangular_to_spherical_conversion_l3347_334782

def rectangular_to_spherical (x y z : ℝ) : ℝ × ℝ × ℝ := sorry

theorem rectangular_to_spherical_conversion :
  let (ρ, θ, φ) := rectangular_to_spherical (4 * Real.sqrt 2) (-4) 4
  ρ = 8 ∧ θ = 7 * Real.pi / 4 ∧ φ = Real.pi / 3 ∧
  ρ > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ 0 ≤ φ ∧ φ ≤ Real.pi :=
by sorry

end NUMINAMATH_CALUDE_rectangular_to_spherical_conversion_l3347_334782


namespace NUMINAMATH_CALUDE_oplus_properties_l3347_334789

def oplus (a b : ℚ) : ℚ := a * b + 2 * a

theorem oplus_properties :
  (oplus 2 (-1) = 2) ∧
  (oplus (-3) (oplus (-4) (1/2)) = 24) := by
  sorry

end NUMINAMATH_CALUDE_oplus_properties_l3347_334789


namespace NUMINAMATH_CALUDE_path_width_calculation_l3347_334760

theorem path_width_calculation (field_length field_width path_area : ℝ) 
  (h1 : field_length = 20)
  (h2 : field_width = 15)
  (h3 : path_area = 246)
  (h4 : field_length > 0)
  (h5 : field_width > 0)
  (h6 : path_area > 0) :
  ∃ (path_width : ℝ),
    path_width > 0 ∧
    (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width = path_area ∧
    path_width = 3 := by
  sorry

end NUMINAMATH_CALUDE_path_width_calculation_l3347_334760


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3347_334724

/-- A hyperbola with focal length 2√5 and asymptote x - 2y = 0 has equation x^2/4 - y^2 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Given hyperbola equation
  (a^2 + b^2 = 5) →                         -- Focal length condition
  (a = 2 * b) →                             -- Asymptote condition
  (∀ x y : ℝ, x^2 / 4 - y^2 = 1) :=         -- Conclusion: specific hyperbola equation
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3347_334724


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l3347_334768

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoid (a b : ℝ) :=
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_lt_b : a < b)

/-- Properties of the isosceles trapezoid -/
def trapezoid_properties (a b : ℝ) (t : IsoscelesTrapezoid a b) :=
  let AB := (a + b) / 2
  let BH := Real.sqrt (a * b)
  let BP := 2 * a * b / (a + b)
  let DF := Real.sqrt ((a^2 + b^2) / 2)
  (AB = (a + b) / 2) ∧
  (BH = Real.sqrt (a * b)) ∧
  (BP = 2 * a * b / (a + b)) ∧
  (DF = Real.sqrt ((a^2 + b^2) / 2)) ∧
  (BP < BH) ∧ (BH < AB) ∧ (AB < DF)

/-- Theorem stating the properties of the isosceles trapezoid -/
theorem isosceles_trapezoid_theorem (a b : ℝ) (t : IsoscelesTrapezoid a b) :
  trapezoid_properties a b t := by
  sorry

end NUMINAMATH_CALUDE_isosceles_trapezoid_theorem_l3347_334768


namespace NUMINAMATH_CALUDE_alligator_growth_rate_l3347_334774

def alligator_population (initial_population : ℕ) (rate : ℕ) (periods : ℕ) : ℕ :=
  initial_population + rate * periods

theorem alligator_growth_rate :
  ∀ (rate : ℕ),
    alligator_population 4 rate 2 = 16 →
    rate = 6 := by
  sorry

end NUMINAMATH_CALUDE_alligator_growth_rate_l3347_334774


namespace NUMINAMATH_CALUDE_system_solution_l3347_334773

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x / (a * b) + y / (b * c) + z / (a * c) = 3)
  (eq2 : x / a + y / b + z / c = a + b + c)
  (eq3 : c^2 * x + a^2 * y + b^2 * z = a * b * c * (a + b + c)) :
  x = a * b ∧ y = b * c ∧ z = a * c := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3347_334773


namespace NUMINAMATH_CALUDE_student_calculation_greater_than_true_average_l3347_334778

theorem student_calculation_greater_than_true_average : 
  ((2 + 4 + 6) / 2 + (8 + 10) / 2) / 2 > (2 + 4 + 6 + 8 + 10) / 5 := by
  sorry

end NUMINAMATH_CALUDE_student_calculation_greater_than_true_average_l3347_334778


namespace NUMINAMATH_CALUDE_square_side_length_l3347_334771

/-- The area enclosed between the circumferences of four circles described about the corners of a square -/
def enclosed_area : ℝ := 42.06195997410015

/-- Theorem: Given four equal circles described about the four corners of a square, 
    each touching two others, with the area enclosed between the circumferences 
    of the circles being 42.06195997410015 cm², the length of a side of the square is 14 cm. -/
theorem square_side_length (r : ℝ) (h1 : r > 0) 
  (h2 : 4 * r^2 - Real.pi * r^2 = enclosed_area) : 
  2 * r = 14 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3347_334771


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_over_i_plus_one_l3347_334732

theorem imaginary_part_of_i_over_i_plus_one :
  Complex.im (Complex.I / (Complex.I + 1)) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_over_i_plus_one_l3347_334732


namespace NUMINAMATH_CALUDE_relationship_abc_l3347_334776

-- Define the constants
noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := 2^(-1/3 : ℝ)
noncomputable def c : ℝ := Real.log 30 / Real.log 3

-- State the theorem
theorem relationship_abc : c > a ∧ a > b := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l3347_334776


namespace NUMINAMATH_CALUDE_ladder_distance_l3347_334762

theorem ladder_distance (ladder_length : Real) (wall_height : Real) 
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12) :
  ∃ (base_distance : Real), 
    base_distance^2 + wall_height^2 = ladder_length^2 ∧ 
    base_distance = 5 := by
  sorry

end NUMINAMATH_CALUDE_ladder_distance_l3347_334762


namespace NUMINAMATH_CALUDE_tuesday_most_available_l3347_334720

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

-- Define the people
inductive Person
  | Anna
  | Bill
  | Carl
  | Dave
  | Eve

-- Define the availability function
def isAvailable (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Tuesday => true
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Thursday => true
  | Person.Anna, Day.Friday => true
  | Person.Bill, Day.Monday => true
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Wednesday => true
  | Person.Bill, Day.Thursday => false
  | Person.Bill, Day.Friday => false
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Wednesday => true
  | Person.Carl, Day.Thursday => false
  | Person.Carl, Day.Friday => false
  | Person.Dave, Day.Monday => true
  | Person.Dave, Day.Tuesday => true
  | Person.Dave, Day.Wednesday => false
  | Person.Dave, Day.Thursday => true
  | Person.Dave, Day.Friday => false
  | Person.Eve, Day.Monday => false
  | Person.Eve, Day.Tuesday => true
  | Person.Eve, Day.Wednesday => true
  | Person.Eve, Day.Thursday => false
  | Person.Eve, Day.Friday => true

-- Count available people for a given day
def countAvailable (d : Day) : Nat :=
  (List.filter (fun p => isAvailable p d) [Person.Anna, Person.Bill, Person.Carl, Person.Dave, Person.Eve]).length

-- Define the theorem
theorem tuesday_most_available :
  ∀ d : Day, d ≠ Day.Tuesday → countAvailable Day.Tuesday ≥ countAvailable d :=
sorry

end NUMINAMATH_CALUDE_tuesday_most_available_l3347_334720


namespace NUMINAMATH_CALUDE_live_streaming_fee_strategy2_revenue_total_profit_l3347_334783

-- Define the problem parameters
def total_items : ℕ := 600
def strategy1_items : ℕ := 200
def strategy2_items : ℕ := 400
def strategy2_phase1_items : ℕ := 100
def strategy2_phase2_items : ℕ := 300

-- Define the strategies
def strategy1_price (m : ℝ) : ℝ := 2 * m - 5
def strategy1_fee_rate : ℝ := 0.01
def strategy2_base_price (m : ℝ) : ℝ := 2.5 * m
def strategy2_discount1 : ℝ := 0.8
def strategy2_discount2 : ℝ := 0.8

-- Theorem statements
theorem live_streaming_fee (m : ℝ) :
  strategy1_items * strategy1_price m * strategy1_fee_rate = 4 * m - 10 := by sorry

theorem strategy2_revenue (m : ℝ) :
  strategy2_phase1_items * strategy2_base_price m * strategy2_discount1 +
  strategy2_phase2_items * strategy2_base_price m * strategy2_discount1 * strategy2_discount2 = 680 * m := by sorry

theorem total_profit (m : ℝ) :
  strategy1_items * strategy1_price m +
  (strategy2_phase1_items * strategy2_base_price m * strategy2_discount1 +
   strategy2_phase2_items * strategy2_base_price m * strategy2_discount1 * strategy2_discount2) -
  (strategy1_items * strategy1_price m * strategy1_fee_rate) -
  (total_items * m) = 476 * m - 990 := by sorry

end NUMINAMATH_CALUDE_live_streaming_fee_strategy2_revenue_total_profit_l3347_334783


namespace NUMINAMATH_CALUDE_cone_base_radius_l3347_334799

theorem cone_base_radius (surface_area : ℝ) (r : ℝ) : 
  surface_area = 12 * Real.pi ∧ 
  (∃ l : ℝ, l = 2 * r ∧ surface_area = Real.pi * r^2 + Real.pi * r * l) → 
  r = 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_base_radius_l3347_334799


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_l3347_334764

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- The condition for z = (1+mi)(1+i) to be purely imaginary, where m is a real number. -/
theorem purely_imaginary_condition (m : ℝ) : 
  IsPurelyImaginary ((1 + m * Complex.I) * (1 + Complex.I)) ↔ m = 1 := by
  sorry

#check purely_imaginary_condition

end NUMINAMATH_CALUDE_purely_imaginary_condition_l3347_334764


namespace NUMINAMATH_CALUDE_zeros_product_bound_l3347_334708

/-- Given a > e and f(x) = e^x - a((ln x + x)/x) has two distinct zeros, prove x₁x₂ > e^(2-x₁-x₂) -/
theorem zeros_product_bound (a : ℝ) (x₁ x₂ : ℝ) 
  (ha : a > Real.exp 1)
  (hf : ∀ x : ℝ, x > 0 → Real.exp x - a * ((Real.log x + x) / x) = 0 ↔ x = x₁ ∨ x = x₂)
  (hx : x₁ ≠ x₂) :
  x₁ * x₂ > Real.exp (2 - x₁ - x₂) :=
by sorry

end NUMINAMATH_CALUDE_zeros_product_bound_l3347_334708


namespace NUMINAMATH_CALUDE_function_range_iff_a_ge_one_l3347_334718

/-- Given a real number a, the function f(x) = √((a-1)x² + ax + 1) has range [0, +∞) if and only if a ≥ 1 -/
theorem function_range_iff_a_ge_one (a : ℝ) :
  (Set.range (fun x => Real.sqrt ((a - 1) * x^2 + a * x + 1)) = Set.Ici 0) ↔ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_function_range_iff_a_ge_one_l3347_334718


namespace NUMINAMATH_CALUDE_product_equals_square_l3347_334711

theorem product_equals_square : 50 * 24.96 * 2.496 * 500 = (1248 : ℝ)^2 := by sorry

end NUMINAMATH_CALUDE_product_equals_square_l3347_334711


namespace NUMINAMATH_CALUDE_dilin_gave_sword_l3347_334798

-- Define the types for individuals and gifts
inductive Individual : Type
| Ilse : Individual
| Elsa : Individual
| Bilin : Individual
| Dilin : Individual

inductive Gift : Type
| Sword : Gift
| Necklace : Gift

-- Define the type for statements
inductive Statement : Type
| GiftWasSword : Statement
| IDidNotGive : Statement
| IlseGaveNecklace : Statement
| BilinGaveSword : Statement

-- Define a function to determine if an individual is an elf
def isElf (i : Individual) : Prop :=
  i = Individual.Ilse ∨ i = Individual.Elsa

-- Define a function to determine if an individual is a dwarf
def isDwarf (i : Individual) : Prop :=
  i = Individual.Bilin ∨ i = Individual.Dilin

-- Define the truth value of a statement given who made it and who gave the gift
def isTruthful (speaker : Individual) (giver : Individual) (gift : Gift) (s : Statement) : Prop :=
  match s with
  | Statement.GiftWasSword => gift = Gift.Sword
  | Statement.IDidNotGive => speaker ≠ giver
  | Statement.IlseGaveNecklace => giver = Individual.Ilse ∧ gift = Gift.Necklace
  | Statement.BilinGaveSword => giver = Individual.Bilin ∧ gift = Gift.Sword

-- Define the conditions of truthfulness based on the problem statement
def meetsConditions (speaker : Individual) (giver : Individual) (gift : Gift) (s : Statement) : Prop :=
  (isElf speaker ∧ isDwarf giver → ¬isTruthful speaker giver gift s) ∧
  (isDwarf speaker ∧ (s = Statement.IDidNotGive ∨ s = Statement.IlseGaveNecklace) → ¬isTruthful speaker giver gift s) ∧
  (¬(isElf speaker ∧ isDwarf giver) ∧ ¬(isDwarf speaker ∧ (s = Statement.IDidNotGive ∨ s = Statement.IlseGaveNecklace)) → isTruthful speaker giver gift s)

-- The theorem to be proved
theorem dilin_gave_sword :
  ∃ (speakers : Fin 4 → Individual),
    (∃ (statements : Fin 4 → Statement),
      (∀ i : Fin 4, meetsConditions (speakers i) Individual.Dilin Gift.Sword (statements i)) ∧
      (∃ i : Fin 4, statements i = Statement.GiftWasSword) ∧
      (∃ i : Fin 4, statements i = Statement.IDidNotGive) ∧
      (∃ i : Fin 4, statements i = Statement.IlseGaveNecklace) ∧
      (∃ i : Fin 4, statements i = Statement.BilinGaveSword)) :=
sorry

end NUMINAMATH_CALUDE_dilin_gave_sword_l3347_334798


namespace NUMINAMATH_CALUDE_root_existence_l3347_334722

theorem root_existence : ∃ x : ℝ, x ∈ (Set.Ioo (-1) (-1/2)) ∧ 2^x + x = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_existence_l3347_334722


namespace NUMINAMATH_CALUDE_opposite_implies_sum_l3347_334756

theorem opposite_implies_sum (x : ℝ) : 
  (3 - x) = -2 → x + 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_implies_sum_l3347_334756


namespace NUMINAMATH_CALUDE_union_equals_A_l3347_334729

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x * (x - 1) ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x > 0 ∧ Real.log x ≤ a}

-- State the theorem
theorem union_equals_A (a : ℝ) : A ∪ B a = A ↔ a = 0 := by
  sorry

end NUMINAMATH_CALUDE_union_equals_A_l3347_334729


namespace NUMINAMATH_CALUDE_problem_solution_l3347_334796

theorem problem_solution (t : ℝ) :
  let x := 3 - t
  let y := 2*t + 11
  x = 1 → y = 15 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3347_334796


namespace NUMINAMATH_CALUDE_angle_DAB_depends_on_triangle_l3347_334757

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the rectangle BCDE
structure Rectangle :=
  (B C D E : ℝ × ℝ)

-- Define the angle β (DAB)
def angle_DAB (tri : Triangle) (rect : Rectangle) : ℝ := sorry

-- State the theorem
theorem angle_DAB_depends_on_triangle (tri : Triangle) (rect : Rectangle) :
  tri.A ≠ tri.B ∧ tri.B ≠ tri.C ∧ tri.C ≠ tri.A →  -- Triangle inequality
  (tri.A.1 - tri.C.1)^2 + (tri.A.2 - tri.C.2)^2 = (tri.B.1 - tri.C.1)^2 + (tri.B.2 - tri.C.2)^2 →  -- CA = CB
  (rect.B = tri.B ∧ rect.C = tri.C) →  -- Rectangle is constructed on CB
  (rect.B.1 - rect.C.1)^2 + (rect.B.2 - rect.C.2)^2 > (rect.C.1 - rect.D.1)^2 + (rect.C.2 - rect.D.2)^2 →  -- BC > CD
  ∃ (f : Triangle → ℝ), angle_DAB tri rect = f tri :=
sorry

end NUMINAMATH_CALUDE_angle_DAB_depends_on_triangle_l3347_334757


namespace NUMINAMATH_CALUDE_video_game_discount_savings_l3347_334730

theorem video_game_discount_savings (original_price : ℚ) 
  (flat_discount : ℚ) (percentage_discount : ℚ) : 
  original_price = 60 →
  flat_discount = 10 →
  percentage_discount = 0.25 →
  (original_price - flat_discount) * (1 - percentage_discount) - 
  (original_price * (1 - percentage_discount) - flat_discount) = 
  250 / 100 := by
  sorry

end NUMINAMATH_CALUDE_video_game_discount_savings_l3347_334730


namespace NUMINAMATH_CALUDE_kaleb_video_games_l3347_334719

def video_game_problem (non_working_games : ℕ) (total_earnings : ℕ) (price_per_game : ℕ) : Prop :=
  let working_games := total_earnings / price_per_game
  working_games + non_working_games = 10

theorem kaleb_video_games :
  video_game_problem 8 12 6 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_video_games_l3347_334719


namespace NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l3347_334737

theorem arithmetic_and_geometric_sequence (a : ℕ → ℝ) : 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) →  -- arithmetic sequence
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) →  -- geometric sequence
  (∃ q : ℝ, q = 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n) := by
sorry


end NUMINAMATH_CALUDE_arithmetic_and_geometric_sequence_l3347_334737


namespace NUMINAMATH_CALUDE_initial_lives_calculation_l3347_334790

/-- Proves that the initial number of lives equals the current number of lives plus the number of lives lost -/
theorem initial_lives_calculation (current_lives lost_lives : ℕ) 
  (h1 : current_lives = 70) 
  (h2 : lost_lives = 13) : 
  current_lives + lost_lives = 83 := by
  sorry

end NUMINAMATH_CALUDE_initial_lives_calculation_l3347_334790


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3347_334761

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 3 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 3 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3347_334761


namespace NUMINAMATH_CALUDE_roger_lawn_mowing_earnings_l3347_334702

theorem roger_lawn_mowing_earnings :
  ∀ (total_lawns : ℕ) (forgotten_lawns : ℕ) (total_earnings : ℕ),
    total_lawns = 14 →
    forgotten_lawns = 8 →
    total_earnings = 54 →
    (total_earnings : ℚ) / ((total_lawns - forgotten_lawns) : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_roger_lawn_mowing_earnings_l3347_334702


namespace NUMINAMATH_CALUDE_stream_speed_l3347_334751

/-- Given a boat traveling downstream and upstream, prove the speed of the stream. -/
theorem stream_speed (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ) 
  (h1 : downstream_distance = 60) 
  (h2 : upstream_distance = 30) 
  (h3 : time = 3) :
  ∃ (boat_speed stream_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    stream_speed = 5 := by
  sorry

#check stream_speed

end NUMINAMATH_CALUDE_stream_speed_l3347_334751


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l3347_334755

/-- Proves that the tax rate is 30% given the specified conditions --/
theorem tax_rate_calculation (total_cost tax_free_cost : ℝ) 
  (h1 : total_cost = 20)
  (h2 : tax_free_cost = 14.7)
  (h3 : (total_cost - tax_free_cost) * 0.3 = (total_cost - tax_free_cost) * (30 / 100)) : 
  (((total_cost - tax_free_cost) * 0.3) / (total_cost - tax_free_cost)) * 100 = 30 := by
  sorry

#check tax_rate_calculation

end NUMINAMATH_CALUDE_tax_rate_calculation_l3347_334755


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l3347_334767

/-- Right triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_triangle : a^2 + b^2 = c^2
  side_a : a = 5
  side_b : b = 12
  side_c : c = 13

/-- Square inscribed with one vertex at the right angle -/
def inscribed_square_x (t : RightTriangle) (x : ℝ) : Prop :=
  x > 0 ∧ x < t.a ∧ x < t.b ∧ x / t.a = x / t.b

/-- Square inscribed with one side along the hypotenuse -/
def inscribed_square_y (t : RightTriangle) (y : ℝ) : Prop :=
  y > 0 ∧ y < t.c ∧ (t.a / t.c) * y / t.a = y / t.c

/-- The main theorem -/
theorem inscribed_squares_ratio (t : RightTriangle) 
  (x y : ℝ) (hx : inscribed_square_x t x) (hy : inscribed_square_y t y) : 
  x / y = 4320 / 2873 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l3347_334767


namespace NUMINAMATH_CALUDE_bob_remaining_corn_l3347_334742

theorem bob_remaining_corn (initial_bushels : ℕ) (ears_per_bushel : ℕ) 
  (terry_bushels jerry_bushels linda_bushels : ℕ) (stacy_ears : ℕ) : 
  initial_bushels = 50 →
  ears_per_bushel = 14 →
  terry_bushels = 8 →
  jerry_bushels = 3 →
  linda_bushels = 12 →
  stacy_ears = 21 →
  initial_bushels * ears_per_bushel - 
  (terry_bushels * ears_per_bushel + jerry_bushels * ears_per_bushel + 
   linda_bushels * ears_per_bushel + stacy_ears) = 357 :=
by sorry

end NUMINAMATH_CALUDE_bob_remaining_corn_l3347_334742


namespace NUMINAMATH_CALUDE_range_of_a_l3347_334786

theorem range_of_a (p q : ℝ → Prop) (a : ℝ) : 
  (∀ x, p x ↔ x^2 - 5*x - 6 ≤ 0) →
  (∀ x, q x ↔ x^2 - 2*x + 1 - 4*a^2 ≤ 0) →
  a ≥ 0 →
  (∀ x, ¬(p x) → ¬(q x)) ∧ (∃ x, ¬(p x) ∧ q x) →
  a ≥ 5/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3347_334786


namespace NUMINAMATH_CALUDE_heartsuit_not_commutative_l3347_334735

-- Define the ♥ operation
def heartsuit (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem statement
theorem heartsuit_not_commutative : ¬ ∀ (x y : ℝ), heartsuit x y = heartsuit y x := by
  sorry

end NUMINAMATH_CALUDE_heartsuit_not_commutative_l3347_334735


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l3347_334749

theorem decimal_to_percentage (x : ℝ) : x = 1.20 → (x * 100 : ℝ) = 120 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l3347_334749


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_7200_eq_12_l3347_334723

/-- The number of factors of 7200 that are perfect squares -/
def perfect_square_factors_of_7200 : ℕ :=
  let n := 7200
  let factorization := [(2, 4), (3, 2), (5, 2)]
  (List.map (fun (p : ℕ × ℕ) => (p.2 / 2 + 1)) factorization).prod

/-- Theorem stating that the number of factors of 7200 that are perfect squares is 12 -/
theorem perfect_square_factors_of_7200_eq_12 :
  perfect_square_factors_of_7200 = 12 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_7200_eq_12_l3347_334723


namespace NUMINAMATH_CALUDE_polynomial_equality_l3347_334745

theorem polynomial_equality (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a - a₁ + a₂ - a₃ + a₄ - a₅ = -243 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3347_334745


namespace NUMINAMATH_CALUDE_max_value_of_function_l3347_334726

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1/2) :
  ∃ (y : ℝ), y = 1/27 ∧ ∀ (z : ℝ), 0 < z ∧ z < 1/2 → x^2 * (1 - 2*x) ≤ y := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3347_334726


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3347_334788

theorem complex_number_in_first_quadrant : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ (1 / (1 + Complex.I) + Complex.I) = Complex.mk a b := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3347_334788


namespace NUMINAMATH_CALUDE_tug_of_war_competition_l3347_334766

/-- Calculates the number of matches in a tug-of-war competition -/
def number_of_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Calculates the number of matches for each class in a tug-of-war competition -/
def matches_per_class (n : ℕ) : ℕ := n - 1

theorem tug_of_war_competition (n : ℕ) (h : n = 7) :
  number_of_matches n = 21 ∧ matches_per_class n = 6 := by
  sorry

#eval number_of_matches 7
#eval matches_per_class 7

end NUMINAMATH_CALUDE_tug_of_war_competition_l3347_334766


namespace NUMINAMATH_CALUDE_carSalesmanFebruarySales_l3347_334780

/-- Represents the earnings and sales of a car salesman -/
structure CarSalesman where
  baseSalary : ℕ
  commission : ℕ
  januaryEarnings : ℕ

/-- Calculates the number of cars needed to be sold to reach a target earnings -/
def carsNeededForEarnings (s : CarSalesman) (targetEarnings : ℕ) : ℕ :=
  ((targetEarnings - s.baseSalary) + s.commission - 1) / s.commission

/-- Theorem: The car salesman needs to sell 13 cars in February to double January earnings -/
theorem carSalesmanFebruarySales (s : CarSalesman)
    (h1 : s.baseSalary = 1000)
    (h2 : s.commission = 200)
    (h3 : s.januaryEarnings = 1800) :
    carsNeededForEarnings s (2 * s.januaryEarnings) = 13 := by
  sorry


end NUMINAMATH_CALUDE_carSalesmanFebruarySales_l3347_334780


namespace NUMINAMATH_CALUDE_tan_function_property_l3347_334713

/-- Given a function y = a * tan(b * x) where a and b are positive constants,
    if the function passes through (π/4, 3) and has a period of 3π/2,
    then a * b = 2 * √3 -/
theorem tan_function_property (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a * Real.tan (b * (π / 4)) = 3) →
  (π / b = 3 * π / 2) →
  a * b = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_function_property_l3347_334713


namespace NUMINAMATH_CALUDE_combined_tax_rate_l3347_334752

/-- The combined tax rate problem -/
theorem combined_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (julie_rate : ℝ) 
  (mindy_income : ℝ → ℝ) 
  (julie_income : ℝ → ℝ) 
  (h1 : mork_rate = 0.45)
  (h2 : mindy_rate = 0.25)
  (h3 : julie_rate = 0.35)
  (h4 : ∀ m, mindy_income m = 4 * m)
  (h5 : ∀ m, julie_income m = 2 * m)
  (h6 : ∀ m, julie_income m = (mindy_income m) / 2) :
  ∀ m : ℝ, m > 0 → 
    (mork_rate * m + mindy_rate * (mindy_income m) + julie_rate * (julie_income m)) / 
    (m + mindy_income m + julie_income m) = 2.15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_combined_tax_rate_l3347_334752


namespace NUMINAMATH_CALUDE_perpendicular_planes_l3347_334785

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relationship between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relationship between lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relationship between planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define non-overlapping relationship for lines
variable (non_overlapping_lines : Line → Line → Prop)

-- Define non-overlapping relationship for planes
variable (non_overlapping_planes : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (m n : Line) (α β : Plane)
  (h1 : non_overlapping_lines m n)
  (h2 : non_overlapping_planes α β)
  (h3 : perp_line_plane m α)
  (h4 : perp_line_plane n β)
  (h5 : perp_line_line m n) :
  perp_plane_plane α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_planes_l3347_334785


namespace NUMINAMATH_CALUDE_completing_square_sum_l3347_334791

theorem completing_square_sum (d e f : ℤ) : 
  (100 : ℤ) * (x : ℚ)^2 + 60 * x - 90 = 0 ↔ (d * x + e)^2 = f →
  d > 0 →
  d + e + f = 112 :=
by sorry

end NUMINAMATH_CALUDE_completing_square_sum_l3347_334791


namespace NUMINAMATH_CALUDE_probability_special_arrangement_l3347_334746

/-- The number of ways to arrange n distinct items -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n distinct items with one specific item in the first two positions
    and two other specific items not adjacent -/
def specialArrangements (n : ℕ) : ℕ :=
  2 * (arrangements (n - 1) - 2 * arrangements (n - 3))

theorem probability_special_arrangement :
  specialArrangements 6 / arrangements 6 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_special_arrangement_l3347_334746


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3347_334754

theorem tan_alpha_value (α : ℝ) 
  (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1/4) : 
  Real.tan α = -2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3347_334754


namespace NUMINAMATH_CALUDE_impossible_to_flip_all_l3347_334727

/-- Represents the color of a button's face -/
inductive ButtonColor
| White
| Black

/-- Represents a configuration of buttons in a circle -/
def ButtonConfiguration := List ButtonColor

/-- Represents a move in the game -/
inductive Move
| FlipAdjacent (i : Nat)  -- Flip two adjacent buttons at position i and i+1
| FlipSeparated (i : Nat) -- Flip two buttons at position i and i+2

/-- The initial configuration of buttons -/
def initial_config : ButtonConfiguration :=
  [ButtonColor.Black] ++ List.replicate 2021 ButtonColor.White

/-- Applies a move to a button configuration -/
def apply_move (config : ButtonConfiguration) (move : Move) : ButtonConfiguration :=
  sorry

/-- Checks if all buttons have been flipped from their initial state -/
def all_flipped (config : ButtonConfiguration) : Prop :=
  sorry

/-- The main theorem stating it's impossible to flip all buttons -/
theorem impossible_to_flip_all (moves : List Move) :
  ¬(all_flipped (moves.foldl apply_move initial_config)) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_flip_all_l3347_334727


namespace NUMINAMATH_CALUDE_alice_forest_walk_l3347_334736

def morning_walk : ℕ := 10
def days_per_week : ℕ := 5
def total_distance : ℕ := 110

theorem alice_forest_walk :
  let morning_total := morning_walk * days_per_week
  let forest_total := total_distance - morning_total
  forest_total / days_per_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_alice_forest_walk_l3347_334736


namespace NUMINAMATH_CALUDE_bus_arrival_probability_l3347_334740

/-- The probability of a bus arriving on time for a single ride -/
def p : ℝ := 0.9

/-- The number of total rides -/
def n : ℕ := 5

/-- The number of on-time arrivals we're interested in -/
def k : ℕ := 4

/-- The binomial probability of k successes in n trials with probability p -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p ^ k * (1 - p) ^ (n - k)

theorem bus_arrival_probability :
  binomial_probability n k p = 0.328 := by
  sorry

end NUMINAMATH_CALUDE_bus_arrival_probability_l3347_334740


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l3347_334716

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (original_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) :
  original_players = 7 →
  original_average = 76 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  let total_weight := original_players * original_average + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  let new_average := total_weight / new_total_players
  new_average = 78 := by
sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l3347_334716


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l3347_334792

theorem pizza_slices_per_person (num_people : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_people = 6)
  (h2 : num_pizzas = 3)
  (h3 : slices_per_pizza = 8) :
  (num_pizzas * slices_per_pizza) / num_people = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l3347_334792


namespace NUMINAMATH_CALUDE_volleyball_scores_l3347_334797

/-- Volleyball competition scores -/
theorem volleyball_scores (lizzie_score : ℕ) (nathalie_score : ℕ) (aimee_score : ℕ) (team_score : ℕ) :
  lizzie_score = 4 →
  nathalie_score = lizzie_score + 3 →
  aimee_score = 2 * (lizzie_score + nathalie_score) →
  team_score = 50 →
  team_score - (lizzie_score + nathalie_score + aimee_score) = 17 := by
sorry


end NUMINAMATH_CALUDE_volleyball_scores_l3347_334797


namespace NUMINAMATH_CALUDE_same_solution_for_k_17_l3347_334775

theorem same_solution_for_k_17 :
  ∃ x : ℝ, (2 * x + 4 = 4 * (x - 2)) ∧ (17 * x - 91 = 2 * x - 1) := by
  sorry

#check same_solution_for_k_17

end NUMINAMATH_CALUDE_same_solution_for_k_17_l3347_334775


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l3347_334753

-- Define the two line equations
def line1 (x y : ℝ) : Prop := 3 * y = -2 * x + 6
def line2 (x y : ℝ) : Prop := 4 * y = 7 * x - 8

-- Define the intersection point
def intersection_point : ℝ × ℝ := (0, 2)

-- Theorem statement
theorem intersection_point_is_unique :
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) ∧
  (∀ x y : ℝ, line1 x y ∧ line2 x y → (x, y) = intersection_point) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l3347_334753


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3347_334709

theorem polynomial_simplification (x : ℝ) : 
  (6 * x^10 + 8 * x^9 + 3 * x^7) + (2 * x^12 + 3 * x^10 + x^9 + 5 * x^7 + 4 * x^4 + 7 * x + 6) = 
  2 * x^12 + 9 * x^10 + 9 * x^9 + 8 * x^7 + 4 * x^4 + 7 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3347_334709


namespace NUMINAMATH_CALUDE_nancy_widget_production_l3347_334763

/-- Nancy's widget production problem -/
theorem nancy_widget_production (t : ℝ) (h : t > 0) : 
  let w := 2 * t
  let monday_production := w * t
  let tuesday_production := (w + 5) * (t - 3)
  monday_production - tuesday_production = t + 15 := by
sorry

end NUMINAMATH_CALUDE_nancy_widget_production_l3347_334763
