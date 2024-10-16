import Mathlib

namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l1036_103643

theorem real_part_of_complex_product : ∃ (z : ℂ), z = (1 + Complex.I) * (2 - Complex.I) ∧ z.re = 3 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l1036_103643


namespace NUMINAMATH_CALUDE_hat_shoppe_pricing_l1036_103690

theorem hat_shoppe_pricing (x : ℝ) (h : x > 0) : 
  0.75 * (1.3 * x) = 0.975 * x := by
  sorry

end NUMINAMATH_CALUDE_hat_shoppe_pricing_l1036_103690


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l1036_103679

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l1036_103679


namespace NUMINAMATH_CALUDE_prime_factor_count_l1036_103626

/-- Given an expression 4^11 * x^5 * 11^2 with a total of 29 prime factors, x must be a prime number -/
theorem prime_factor_count (x : ℕ) : 
  (∀ (p : ℕ), Prime p → (Nat.factorization (4^11 * x^5 * 11^2)).sum (λ _ e => e) = 29) → 
  Prime x := by
sorry

end NUMINAMATH_CALUDE_prime_factor_count_l1036_103626


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1036_103623

/-- Given vectors a and b, if the angle between them is π/6, then the second component of b is √3. -/
theorem angle_between_vectors (a b : ℝ × ℝ) :
  a = (1, Real.sqrt 3) →
  b.1 = 3 →
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = Real.cos (π / 6) →
  b.2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1036_103623


namespace NUMINAMATH_CALUDE_ceiling_sqrt_156_l1036_103650

theorem ceiling_sqrt_156 : ⌈Real.sqrt 156⌉ = 13 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sqrt_156_l1036_103650


namespace NUMINAMATH_CALUDE_min_arg_z_l1036_103674

/-- Given a complex number z satisfying |z+3-√3i| = √3, 
    the minimum value of arg z is 5π/6 -/
theorem min_arg_z (z : ℂ) (h : Complex.abs (z + 3 - Complex.I * Real.sqrt 3) = Real.sqrt 3) :
  ∃ (min_arg : ℝ), min_arg = 5 * Real.pi / 6 ∧ 
    ∀ (θ : ℝ), Complex.arg z = θ → θ ≥ min_arg :=
by sorry

end NUMINAMATH_CALUDE_min_arg_z_l1036_103674


namespace NUMINAMATH_CALUDE_boat_downstream_time_l1036_103648

/-- Proves that the time taken for a boat to travel downstream is 4 hours -/
theorem boat_downstream_time
  (distance : ℝ)
  (upstream_time : ℝ)
  (current_speed : ℝ)
  (h1 : distance = 24)
  (h2 : upstream_time = 6)
  (h3 : current_speed = 1)
  : ∃ (boat_speed : ℝ),
    (distance / (boat_speed + current_speed) = 4 ∧
     distance / (boat_speed - current_speed) = upstream_time) :=
by sorry

end NUMINAMATH_CALUDE_boat_downstream_time_l1036_103648


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1036_103636

theorem quadratic_equations_solutions :
  (∀ x : ℝ, (x + 4)^2 - 5*(x + 4) = 0 ↔ x = -4 ∨ x = 1) ∧
  (∀ x : ℝ, x^2 - 2*x - 15 = 0 ↔ x = -3 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1036_103636


namespace NUMINAMATH_CALUDE_electronic_shop_price_l1036_103627

def smartphone_price : ℝ := 300

def personal_computer_price (smartphone_price : ℝ) : ℝ :=
  smartphone_price + 500

def advanced_tablet_price (smartphone_price personal_computer_price : ℝ) : ℝ :=
  smartphone_price + personal_computer_price

def total_price (smartphone_price personal_computer_price advanced_tablet_price : ℝ) : ℝ :=
  smartphone_price + personal_computer_price + advanced_tablet_price

def discounted_price (total_price : ℝ) : ℝ :=
  total_price * 0.9

def final_price (discounted_price : ℝ) : ℝ :=
  discounted_price * 1.05

theorem electronic_shop_price :
  final_price (discounted_price (total_price smartphone_price 
    (personal_computer_price smartphone_price) 
    (advanced_tablet_price smartphone_price (personal_computer_price smartphone_price)))) = 2079 := by
  sorry

end NUMINAMATH_CALUDE_electronic_shop_price_l1036_103627


namespace NUMINAMATH_CALUDE_fraction_of_c_grades_l1036_103653

theorem fraction_of_c_grades 
  (total_students : ℕ) 
  (a_fraction : ℚ) 
  (b_fraction : ℚ) 
  (d_count : ℕ) 
  (h_total : total_students = 800)
  (h_a : a_fraction = 1 / 5)
  (h_b : b_fraction = 1 / 4)
  (h_d : d_count = 40) :
  (total_students - (a_fraction * total_students + b_fraction * total_students + d_count)) / total_students = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_c_grades_l1036_103653


namespace NUMINAMATH_CALUDE_triangle_trig_identity_l1036_103624

theorem triangle_trig_identity (A B C : ℝ) 
  (h1 : A + B + C = Real.pi)
  (h2 : (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) = 1) :
  (Real.cos (2*A) + Real.cos (2*B) + Real.cos (2*C)) / (Real.cos A + Real.cos B + Real.cos C) = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_identity_l1036_103624


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1036_103669

theorem not_sufficient_not_necessary (a b : ℝ) : 
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_l1036_103669


namespace NUMINAMATH_CALUDE_tan_11_25_decomposition_l1036_103640

theorem tan_11_25_decomposition :
  ∃ (a b c d : ℕ+), 
    (Real.tan (11.25 * Real.pi / 180) = Real.sqrt a - Real.sqrt b + Real.sqrt c - d) ∧
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (a + b + c + d = 4) := by
  sorry

end NUMINAMATH_CALUDE_tan_11_25_decomposition_l1036_103640


namespace NUMINAMATH_CALUDE_friend_meeting_point_l1036_103688

theorem friend_meeting_point (trail_length : ℝ) (speed_difference : ℝ) 
  (h1 : trail_length = 60)
  (h2 : speed_difference = 0.4) : 
  let faster_friend_distance := trail_length * (1 + speed_difference) / (2 + speed_difference)
  faster_friend_distance = 35 := by
  sorry

end NUMINAMATH_CALUDE_friend_meeting_point_l1036_103688


namespace NUMINAMATH_CALUDE_robins_water_consumption_l1036_103614

theorem robins_water_consumption (bottles_bought : ℕ) (extra_bottles : ℕ) :
  bottles_bought = 617 →
  extra_bottles = 4 →
  ∃ (days : ℕ) (daily_consumption : ℕ) (last_day_consumption : ℕ),
    days = bottles_bought ∧
    daily_consumption = 1 ∧
    last_day_consumption = daily_consumption + extra_bottles ∧
    bottles_bought + extra_bottles = days * daily_consumption + extra_bottles :=
by
  sorry

#check robins_water_consumption

end NUMINAMATH_CALUDE_robins_water_consumption_l1036_103614


namespace NUMINAMATH_CALUDE_classroom_ratio_l1036_103639

theorem classroom_ratio : 
  ∀ (boys girls : ℕ),
  boys = girls →
  boys + girls = 32 →
  (boys : ℚ) / (girls - 8 : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_classroom_ratio_l1036_103639


namespace NUMINAMATH_CALUDE_only_set_D_is_right_triangle_l1036_103649

-- Define a function to check if three numbers form a right triangle
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Define the sets of line segments
def set_A : (ℝ × ℝ × ℝ) := (3, 5, 7)
def set_B : (ℝ × ℝ × ℝ) := (4, 6, 8)
def set_C : (ℝ × ℝ × ℝ) := (5, 7, 9)
def set_D : (ℝ × ℝ × ℝ) := (6, 8, 10)

-- Theorem stating that only set D forms a right triangle
theorem only_set_D_is_right_triangle :
  ¬(is_right_triangle set_A.1 set_A.2.1 set_A.2.2) ∧
  ¬(is_right_triangle set_B.1 set_B.2.1 set_B.2.2) ∧
  ¬(is_right_triangle set_C.1 set_C.2.1 set_C.2.2) ∧
  (is_right_triangle set_D.1 set_D.2.1 set_D.2.2) :=
by sorry


end NUMINAMATH_CALUDE_only_set_D_is_right_triangle_l1036_103649


namespace NUMINAMATH_CALUDE_evaluate_expression_l1036_103634

theorem evaluate_expression : (3^3)^2 * 3^2 = 6561 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1036_103634


namespace NUMINAMATH_CALUDE_sphere_volume_hexagonal_prism_l1036_103617

/-- The volume of a sphere circumscribing a hexagonal prism -/
theorem sphere_volume_hexagonal_prism (h : ℝ) (p : ℝ) : 
  h = Real.sqrt 3 →
  p = 3 →
  (4 / 3 * Real.pi : ℝ) = (4 / 3 * Real.pi * (((h^2 + (p / 6)^2) / 4)^(3/2))) :=
by sorry

end NUMINAMATH_CALUDE_sphere_volume_hexagonal_prism_l1036_103617


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1036_103644

/-- Given that a² varies inversely with b⁴, and a = 7 when b = 2, 
    prove that a² = 3.0625 when b = 4 -/
theorem inverse_variation_problem (a b : ℝ) (k : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → a^2 * x^4 = k) →  -- a² varies inversely with b⁴
  (7^2 * 2^4 = k) →                             -- a = 7 when b = 2
  (a^2 * 4^4 = k) →                             -- condition for b = 4
  a^2 = 3.0625                                  -- conclusion
:= by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1036_103644


namespace NUMINAMATH_CALUDE_vector_decomposition_l1036_103645

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![3, 3, -1]
def p : Fin 3 → ℝ := ![3, 1, 0]
def q : Fin 3 → ℝ := ![-1, 2, 1]
def r : Fin 3 → ℝ := ![-1, 0, 2]

/-- Theorem: x can be decomposed as p + q - r -/
theorem vector_decomposition : x = p + q - r := by
  sorry

end NUMINAMATH_CALUDE_vector_decomposition_l1036_103645


namespace NUMINAMATH_CALUDE_triangle_relationships_l1036_103695

/-- Given a triangle with sides a, b, c, circumradius R, inradius r, and semiperimeter p,
    prove the following relationships. -/
theorem triangle_relationships 
  (a b c R r p : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0 ∧ p > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_circumradius : R = (a * b * c) / (4 * (p * (p - a) * (p - b) * (p - c))^(1/2)))
  (h_inradius : r = (p * (p - a) * (p - b) * (p - c))^(1/2) / p) :
  (a * b * c = 4 * p * r * R) ∧ 
  (a * b + b * c + c * a = r^2 + p^2 + 4 * r * R) := by
  sorry


end NUMINAMATH_CALUDE_triangle_relationships_l1036_103695


namespace NUMINAMATH_CALUDE_five_digit_number_puzzle_l1036_103661

theorem five_digit_number_puzzle :
  ∃! N : ℕ,
    10000 ≤ N ∧ N < 100000 ∧
    ∃ (x y : ℕ),
      0 ≤ x ∧ x < 10 ∧
      1000 ≤ y ∧ y < 10000 ∧
      N = 10 * y + x ∧
      N + y = 54321 :=
by sorry

end NUMINAMATH_CALUDE_five_digit_number_puzzle_l1036_103661


namespace NUMINAMATH_CALUDE_fewer_heads_probability_l1036_103609

/-- The probability of getting fewer heads than tails when flipping 12 fair coins -/
def fewer_heads_prob : ℚ := 1586 / 4096

/-- The number of coins being flipped -/
def num_coins : ℕ := 12

theorem fewer_heads_probability :
  fewer_heads_prob = (2^num_coins - (num_coins.choose (num_coins / 2))) / (2 * 2^num_coins) :=
sorry

end NUMINAMATH_CALUDE_fewer_heads_probability_l1036_103609


namespace NUMINAMATH_CALUDE_book_cost_theorem_l1036_103663

theorem book_cost_theorem (selling_price_1 selling_price_2 : ℝ) 
  (h1 : selling_price_1 = 340)
  (h2 : selling_price_2 = 350)
  (h3 : ∃ (profit : ℝ), selling_price_1 = cost + profit ∧ 
                         selling_price_2 = cost + (1.05 * profit)) :
  cost = 140 := by
  sorry

end NUMINAMATH_CALUDE_book_cost_theorem_l1036_103663


namespace NUMINAMATH_CALUDE_maggies_total_earnings_l1036_103654

/-- Maggie's earnings from selling magazine subscriptions --/
def maggies_earnings (price_per_subscription : ℕ) 
  (parents_subscriptions : ℕ) 
  (grandfather_subscriptions : ℕ) 
  (next_door_neighbor_subscriptions : ℕ) : ℕ :=
  let total_subscriptions := 
    parents_subscriptions + 
    grandfather_subscriptions + 
    next_door_neighbor_subscriptions + 
    (2 * next_door_neighbor_subscriptions)
  total_subscriptions * price_per_subscription

/-- Theorem stating Maggie's earnings --/
theorem maggies_total_earnings : 
  maggies_earnings 5 4 1 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_maggies_total_earnings_l1036_103654


namespace NUMINAMATH_CALUDE_some_number_value_l1036_103625

theorem some_number_value (a x : ℚ) : 
  a = 105 → 
  a^3 = x * 25 * 45 * 49 → 
  x = 7/3 := by
sorry

end NUMINAMATH_CALUDE_some_number_value_l1036_103625


namespace NUMINAMATH_CALUDE_total_pencils_specific_pencil_case_l1036_103693

/-- Given an initial number of pencils and a number of pencils added, 
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

/-- In the specific case of 2 initial pencils and 3 added pencils, the total is 5. -/
theorem specific_pencil_case : 
  2 + 3 = 5 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_specific_pencil_case_l1036_103693


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1036_103656

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_common_difference 
  (a₁ d : ℝ) 
  (h1 : arithmetic_sequence a₁ d 5 = 8)
  (h2 : arithmetic_sequence a₁ d 1 + arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 3 = 6) :
  d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l1036_103656


namespace NUMINAMATH_CALUDE_sheep_wandered_off_percentage_l1036_103683

theorem sheep_wandered_off_percentage 
  (total_sheep : ℕ) 
  (rounded_up_percentage : ℚ) 
  (sheep_in_pen : ℕ) 
  (sheep_in_wilderness : ℕ) 
  (h1 : rounded_up_percentage = 90 / 100) 
  (h2 : sheep_in_pen = 81) 
  (h3 : sheep_in_wilderness = 9) 
  (h4 : ↑sheep_in_pen = rounded_up_percentage * ↑total_sheep) 
  (h5 : total_sheep = sheep_in_pen + sheep_in_wilderness) : 
  (↑sheep_in_wilderness / ↑total_sheep) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_sheep_wandered_off_percentage_l1036_103683


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1036_103616

theorem inequality_system_solution (x : ℝ) : 
  (3 * x + 1) / 2 > x ∧ 4 * (x - 2) ≤ x - 5 → -1 < x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1036_103616


namespace NUMINAMATH_CALUDE_toms_calculation_l1036_103641

theorem toms_calculation (x y z : ℝ) 
  (h1 : (x + y) - z = 8) 
  (h2 : (x + y) + z = 20) : 
  x + y = 14 := by
sorry

end NUMINAMATH_CALUDE_toms_calculation_l1036_103641


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_4th_power_l1036_103697

theorem nearest_integer_to_3_plus_sqrt5_4th_power :
  ∃ n : ℤ, n = 752 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 5) ^ 4 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 5) ^ 4 - (m : ℝ)| :=
by sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt5_4th_power_l1036_103697


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l1036_103606

/-- The number of distinct diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def is_nonagon (n : ℕ) : Prop := n = 9

theorem nonagon_diagonals :
  ∀ n : ℕ, is_nonagon n → num_diagonals n = 27 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l1036_103606


namespace NUMINAMATH_CALUDE_labor_cost_calculation_l1036_103601

def cost_of_seeds : ℝ := 50
def cost_of_fertilizers_and_pesticides : ℝ := 35
def number_of_bags : ℕ := 10
def price_per_bag : ℝ := 11
def profit_percentage : ℝ := 0.1

theorem labor_cost_calculation (labor_cost : ℝ) : 
  (cost_of_seeds + cost_of_fertilizers_and_pesticides + labor_cost) * (1 + profit_percentage) = 
  (number_of_bags : ℝ) * price_per_bag → 
  labor_cost = 15 := by
sorry

end NUMINAMATH_CALUDE_labor_cost_calculation_l1036_103601


namespace NUMINAMATH_CALUDE_banana_arrangement_count_l1036_103605

/-- The number of unique arrangements of the letters in BANANA -/
def banana_arrangements : ℕ := 60

/-- The total number of letters in BANANA -/
def total_letters : ℕ := 6

/-- The number of A's in BANANA -/
def num_a : ℕ := 3

/-- The number of N's in BANANA -/
def num_n : ℕ := 2

/-- The number of B's in BANANA -/
def num_b : ℕ := 1

theorem banana_arrangement_count :
  banana_arrangements = Nat.factorial total_letters / (Nat.factorial num_a * Nat.factorial num_n) :=
sorry

end NUMINAMATH_CALUDE_banana_arrangement_count_l1036_103605


namespace NUMINAMATH_CALUDE_tree_spacing_l1036_103666

theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (h1 : yard_length = 350) (h2 : num_trees = 26) :
  let num_segments := num_trees - 1
  let spacing := yard_length / num_segments
  spacing = 14 := by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l1036_103666


namespace NUMINAMATH_CALUDE_solution_set_M_range_of_a_l1036_103687

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

-- Define the solution set M
def M : Set ℝ := {x | -2/3 ≤ x ∧ x ≤ 6}

-- Define the property for part (2)
def property (a : ℝ) : Prop := ∀ x, x ≥ a → f x ≤ x - a

-- Theorem for part (1)
theorem solution_set_M : {x : ℝ | f x ≥ -2} = M := by sorry

-- Theorem for part (2)
theorem range_of_a : {a : ℝ | property a} = {a | a ≤ -2 ∨ a ≥ 4} := by sorry

end NUMINAMATH_CALUDE_solution_set_M_range_of_a_l1036_103687


namespace NUMINAMATH_CALUDE_binary_1101_equals_13_l1036_103676

/-- Converts a list of binary digits to its decimal representation -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1101₂ -/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_equals_13 : binary_to_decimal binary_1101 = 13 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101_equals_13_l1036_103676


namespace NUMINAMATH_CALUDE_total_votes_is_1375_l1036_103629

/-- Represents the election results with given conditions -/
structure ElectionResults where
  winners_votes : ℕ  -- Combined majority of winners
  spoiled_votes : ℕ  -- Number of spoiled votes
  final_percentages : List ℚ  -- Final round percentages for top three candidates

/-- Calculates the total number of votes cast in the election -/
def total_votes (results : ElectionResults) : ℕ :=
  results.winners_votes + results.spoiled_votes

/-- Theorem stating that the total number of votes is 1375 given the conditions -/
theorem total_votes_is_1375 (results : ElectionResults) 
  (h1 : results.winners_votes = 1050)
  (h2 : results.spoiled_votes = 325)
  (h3 : results.final_percentages = [41/100, 34/100, 25/100]) :
  total_votes results = 1375 := by
  sorry

#eval total_votes { winners_votes := 1050, spoiled_votes := 325, final_percentages := [41/100, 34/100, 25/100] }

end NUMINAMATH_CALUDE_total_votes_is_1375_l1036_103629


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1036_103603

theorem largest_integer_satisfying_inequality : 
  ∀ n : ℕ+, n^200 < 3^500 ↔ n ≤ 15 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l1036_103603


namespace NUMINAMATH_CALUDE_percent_decrease_l1036_103652

theorem percent_decrease (original_price sale_price : ℝ) :
  original_price = 100 ∧ sale_price = 20 →
  (original_price - sale_price) / original_price * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_percent_decrease_l1036_103652


namespace NUMINAMATH_CALUDE_acute_angled_triangle_with_acute_pedals_l1036_103668

/-- Represents an angle in degrees, minutes, and seconds -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)
  (seconds : ℕ)

/-- Converts an Angle to seconds -/
def Angle.toSeconds (a : Angle) : ℕ :=
  a.degrees * 3600 + a.minutes * 60 + a.seconds

/-- Checks if an angle is acute (less than 90 degrees) -/
def Angle.isAcute (a : Angle) : Prop :=
  a.toSeconds < 90 * 3600

/-- Calculates the i-th pedal angle given an original angle -/
def pedalAngle (a : Angle) (i : ℕ) : Angle :=
  sorry -- Implementation not required for the statement

/-- Theorem statement for the acute-angled triangle problem -/
theorem acute_angled_triangle_with_acute_pedals :
  ∃ (α β γ : Angle),
    α.toSeconds < β.toSeconds ∧
    β.toSeconds < γ.toSeconds ∧
    Angle.isAcute α ∧
    Angle.isAcute β ∧
    Angle.isAcute γ ∧
    α.toSeconds + β.toSeconds + γ.toSeconds = 180 * 3600 ∧
    (∀ i : ℕ, i > 0 → i ≤ 15 →
      Angle.isAcute (pedalAngle α i) ∧
      Angle.isAcute (pedalAngle β i) ∧
      Angle.isAcute (pedalAngle γ i)) :=
by sorry

end NUMINAMATH_CALUDE_acute_angled_triangle_with_acute_pedals_l1036_103668


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l1036_103651

/-- The side length of square EFGH -/
def side_length : ℕ := 7

/-- The area of square EFGH -/
def total_area : ℕ := side_length ^ 2

/-- The area of the first shaded region (2x2 square) -/
def shaded_area_1 : ℕ := 2 ^ 2

/-- The area of the second shaded region (5x5 square minus 3x3 square) -/
def shaded_area_2 : ℕ := 5 ^ 2 - 3 ^ 2

/-- The area of the third shaded region (7x1 rectangle) -/
def shaded_area_3 : ℕ := 7 * 1

/-- The total shaded area -/
def total_shaded_area : ℕ := shaded_area_1 + shaded_area_2 + shaded_area_3

/-- Theorem: The ratio of shaded area to total area is 33/49 -/
theorem shaded_area_ratio :
  (total_shaded_area : ℚ) / total_area = 33 / 49 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l1036_103651


namespace NUMINAMATH_CALUDE_library_problem_l1036_103608

theorem library_problem (total_books : ℕ) (books_per_student : ℕ) 
  (students_day1 : ℕ) (students_day2 : ℕ) (students_day3 : ℕ) : ℕ :=
  by
  have h1 : total_books = 120 := by sorry
  have h2 : books_per_student = 5 := by sorry
  have h3 : students_day1 = 4 := by sorry
  have h4 : students_day2 = 5 := by sorry
  have h5 : students_day3 = 6 := by sorry
  
  have remaining_books : ℕ := total_books - (students_day1 + students_day2 + students_day3) * books_per_student
  
  exact remaining_books / books_per_student

end NUMINAMATH_CALUDE_library_problem_l1036_103608


namespace NUMINAMATH_CALUDE_triangles_in_decagon_count_l1036_103678

/-- The number of triangles formed from vertices of a regular decagon -/
def triangles_in_decagon : ℕ := Nat.choose 10 3

/-- Theorem stating that the number of triangles in a regular decagon is 120 -/
theorem triangles_in_decagon_count : triangles_in_decagon = 120 := by
  sorry

end NUMINAMATH_CALUDE_triangles_in_decagon_count_l1036_103678


namespace NUMINAMATH_CALUDE_union_equals_interval_l1036_103602

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 4}

-- Define the interval [-1, 4]
def interval : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4}

-- Theorem statement
theorem union_equals_interval : A ∪ B = interval := by
  sorry

end NUMINAMATH_CALUDE_union_equals_interval_l1036_103602


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1036_103622

theorem sphere_volume_from_surface_area :
  ∀ r : ℝ, 
    r > 0 →
    4 * π * r^2 = 12 * π →
    (4 / 3) * π * r^3 = 4 * Real.sqrt 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1036_103622


namespace NUMINAMATH_CALUDE_f_sum_value_l1036_103696

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem f_sum_value : f Real.pi + (deriv f) (Real.pi / 2) = -3 / Real.pi := by
  sorry

end NUMINAMATH_CALUDE_f_sum_value_l1036_103696


namespace NUMINAMATH_CALUDE_expression_evaluation_l1036_103673

theorem expression_evaluation (b : ℚ) (h : b = -3) :
  (3 * b⁻¹ + b⁻¹ / 3) / b = 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1036_103673


namespace NUMINAMATH_CALUDE_smallest_abs_z_l1036_103682

/-- Given a complex number z satisfying |z - 8| + |z - 7i| = 15,
    the smallest possible value of |z| is 56/15 -/
theorem smallest_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - 7*I) = 15) :
  ∃ (w : ℂ), Complex.abs w = 56/15 ∧ ∀ (v : ℂ), Complex.abs (v - 8) + Complex.abs (v - 7*I) = 15 → Complex.abs v ≥ Complex.abs w :=
sorry

end NUMINAMATH_CALUDE_smallest_abs_z_l1036_103682


namespace NUMINAMATH_CALUDE_no_three_points_property_H_l1036_103642

/-- Definition of the ellipse C -/
def C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

/-- Definition of property H for a line intersecting the ellipse C -/
def property_H (A B M : ℝ × ℝ) : Prop :=
  C A.1 A.2 ∧ C B.1 B.2 ∧ C M.1 M.2 ∧
  M.1 = 3/5 * A.1 + 4/5 * B.1 ∧
  M.2 = 3/5 * A.2 + 4/5 * B.2

/-- Main theorem: No three distinct points on C form lines all having property H -/
theorem no_three_points_property_H :
  ¬ ∃ (P Q R : ℝ × ℝ),
    C P.1 P.2 ∧ C Q.1 Q.2 ∧ C R.1 R.2 ∧
    P ≠ Q ∧ Q ≠ R ∧ R ≠ P ∧
    (∃ M₁, property_H P Q M₁) ∧
    (∃ M₂, property_H Q R M₂) ∧
    (∃ M₃, property_H R P M₃) :=
sorry

end NUMINAMATH_CALUDE_no_three_points_property_H_l1036_103642


namespace NUMINAMATH_CALUDE_smallest_maximal_arrangement_l1036_103675

/-- Represents a chessboard -/
structure Chessboard :=
  (size : ℕ)
  (total_squares : ℕ := size * size)

/-- Represents a Γ piece -/
structure GammaPiece :=
  (squares_covered : ℕ := 3)

/-- Represents an arrangement of Γ pieces on a chessboard -/
structure Arrangement (board : Chessboard) :=
  (pieces : ℕ)
  (is_maximal : Bool)

/-- The theorem stating the smallest number of Γ pieces in a maximal arrangement -/
theorem smallest_maximal_arrangement (board : Chessboard) (piece : GammaPiece) :
  board.size = 8 →
  ∃ (arr : Arrangement board), 
    arr.pieces = 16 ∧ 
    arr.is_maximal = true ∧
    ∀ (arr' : Arrangement board), arr'.is_maximal = true → arr'.pieces ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_smallest_maximal_arrangement_l1036_103675


namespace NUMINAMATH_CALUDE_head_start_is_90_meters_l1036_103635

/-- The head start distance in a race between Cristina and Nicky -/
def head_start_distance (cristina_speed nicky_speed : ℝ) (catch_up_time : ℝ) : ℝ :=
  nicky_speed * catch_up_time

/-- Theorem: Given Cristina's speed of 5 m/s, Nicky's speed of 3 m/s, 
    and a catch-up time of 30 seconds, the head start distance is 90 meters -/
theorem head_start_is_90_meters :
  head_start_distance 5 3 30 = 90 := by sorry

end NUMINAMATH_CALUDE_head_start_is_90_meters_l1036_103635


namespace NUMINAMATH_CALUDE_rice_cost_problem_l1036_103691

/-- Proves that the cost of the first type of rice is 16 rupees per kg -/
theorem rice_cost_problem (rice1_weight : ℝ) (rice2_weight : ℝ) (rice2_cost : ℝ) (avg_cost : ℝ) 
  (h1 : rice1_weight = 8)
  (h2 : rice2_weight = 4)
  (h3 : rice2_cost = 22)
  (h4 : avg_cost = 18)
  (h5 : (rice1_weight * rice1_cost + rice2_weight * rice2_cost) / (rice1_weight + rice2_weight) = avg_cost) :
  rice1_cost = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_rice_cost_problem_l1036_103691


namespace NUMINAMATH_CALUDE_carly_backstroke_practice_days_l1036_103628

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the number of weeks in a month -/
def weeksInMonth : ℕ := 4

/-- Represents the total hours Carly practices swimming in a month -/
def totalPracticeHours : ℕ := 96

/-- Represents the hours Carly practices butterfly stroke per day -/
def butterflyHoursPerDay : ℕ := 3

/-- Represents the days Carly practices butterfly stroke per week -/
def butterflyDaysPerWeek : ℕ := 4

/-- Represents the hours Carly practices backstroke per day -/
def backstrokeHoursPerDay : ℕ := 2

/-- Theorem stating that Carly practices backstroke 6 days a week -/
theorem carly_backstroke_practice_days :
  ∃ (backstrokeDaysPerWeek : ℕ),
    backstrokeDaysPerWeek * backstrokeHoursPerDay * weeksInMonth +
    butterflyDaysPerWeek * butterflyHoursPerDay * weeksInMonth = totalPracticeHours ∧
    backstrokeDaysPerWeek = 6 :=
by sorry

end NUMINAMATH_CALUDE_carly_backstroke_practice_days_l1036_103628


namespace NUMINAMATH_CALUDE_correct_commutative_transformation_l1036_103684

-- Define the commutative property of addition
axiom commutative_add (a b : ℝ) : a + b = b + a

-- Define the associative property of addition
axiom associative_add (a b c : ℝ) : (a + b) + c = a + (b + c)

-- State the theorem
theorem correct_commutative_transformation :
  4 + (-6) + 3 = (-6) + 4 + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_commutative_transformation_l1036_103684


namespace NUMINAMATH_CALUDE_largest_x_and_ratio_l1036_103694

theorem largest_x_and_ratio (x : ℝ) (a b c d : ℤ) : 
  (5 * x / 7 + 1 = 3 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (x ≤ (-7 + 21 * Real.sqrt 22) / 10) ∧
  (a * c * d / b = -70) :=
by sorry

end NUMINAMATH_CALUDE_largest_x_and_ratio_l1036_103694


namespace NUMINAMATH_CALUDE_polynomial_not_equal_77_l1036_103647

theorem polynomial_not_equal_77 (x y : ℤ) : 
  x^5 - 4*x^4*y - 5*y^2*x^3 + 20*y^3*x^2 + 4*y^4*x - 16*y^5 ≠ 77 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_equal_77_l1036_103647


namespace NUMINAMATH_CALUDE_final_number_bound_l1036_103662

/-- A function that represents the process of replacing two numbers with their arithmetic mean. -/
def replace (numbers : List ℝ) : List ℝ :=
  sorry

/-- The theorem stating that the final number is not less than 1/n. -/
theorem final_number_bound (n : ℕ) (h : n > 0) :
  ∃ (process : ℕ → List ℝ), 
    (process 0 = List.replicate n 1) ∧ 
    (∀ k, process (k + 1) = replace (process k)) ∧
    (∃ m, (process m).length = 1 ∧ 
      ∀ x ∈ process m, x ≥ 1 / n) :=
  sorry

end NUMINAMATH_CALUDE_final_number_bound_l1036_103662


namespace NUMINAMATH_CALUDE_rebecca_eggs_marbles_difference_l1036_103659

theorem rebecca_eggs_marbles_difference : 
  ∀ (eggs marbles : ℕ), 
  eggs = 20 → marbles = 6 → eggs - marbles = 14 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_eggs_marbles_difference_l1036_103659


namespace NUMINAMATH_CALUDE_shelter_cats_l1036_103672

theorem shelter_cats (total : ℕ) (tuna : ℕ) (chicken : ℕ) (both : ℕ) 
  (h1 : total = 75)
  (h2 : tuna = 18)
  (h3 : chicken = 55)
  (h4 : both = 10) :
  total - (tuna + chicken - both) = 12 :=
by sorry

end NUMINAMATH_CALUDE_shelter_cats_l1036_103672


namespace NUMINAMATH_CALUDE_system_solution_l1036_103621

theorem system_solution (x y : ℝ) : 
  (2 * x + 5 * y = 26 ∧ 4 * x - 2 * y = 4) ↔ (x = 3 ∧ y = 4) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1036_103621


namespace NUMINAMATH_CALUDE_article_supports_statements_l1036_103689

/-- Represents the content of the given article about Chinese literature and Mo Yan's Nobel Prize -/
def ArticleContent : Type := sorry

/-- Represents the manifestations of the marginalization of literature since the 1990s -/
def LiteratureMarginalization (content : ArticleContent) : Prop := sorry

/-- Represents the effects of mentioning Mo Yan's award multiple times -/
def MoYanAwardEffects (content : ArticleContent) : Prop := sorry

/-- Represents ways to better develop pure literature -/
def PureLiteratureDevelopment (content : ArticleContent) : Prop := sorry

/-- The main theorem stating that the article supports the given statements -/
theorem article_supports_statements (content : ArticleContent) :
  LiteratureMarginalization content ∧
  MoYanAwardEffects content ∧
  PureLiteratureDevelopment content :=
sorry

end NUMINAMATH_CALUDE_article_supports_statements_l1036_103689


namespace NUMINAMATH_CALUDE_problem_solution_l1036_103692

-- Define the functions f and g
def f (a b x : ℝ) := |x - a| - |x + b|
def g (a b x : ℝ) := -x^2 - a*x - b

-- State the theorem
theorem problem_solution (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hf_max : ∃ x, f a b x = 3) : 
  (a + b = 3) ∧ 
  (∀ x ≥ a, g a b x < f a b x) → 
  (1/2 < a ∧ a < 3) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1036_103692


namespace NUMINAMATH_CALUDE_carter_cards_l1036_103660

/-- Given that Marcus has 210 baseball cards and 58 more than Carter, 
    prove that Carter has 152 baseball cards. -/
theorem carter_cards (marcus_cards : ℕ) (difference : ℕ) (carter_cards : ℕ) 
  (h1 : marcus_cards = 210)
  (h2 : marcus_cards = carter_cards + difference)
  (h3 : difference = 58) : 
  carter_cards = 152 := by
  sorry

end NUMINAMATH_CALUDE_carter_cards_l1036_103660


namespace NUMINAMATH_CALUDE_aperture_radius_ratio_l1036_103686

theorem aperture_radius_ratio (r : ℝ) (h : r > 0) : 
  ∃ (r_new : ℝ), (π * r_new^2 = 2 * π * r^2) ∧ (r_new / r = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_aperture_radius_ratio_l1036_103686


namespace NUMINAMATH_CALUDE_sum_of_critical_slopes_l1036_103631

/-- Parabola defined by y = 2x^2 -/
def parabola (x : ℝ) : ℝ := 2 * x^2

/-- Point Q -/
def Q : ℝ × ℝ := (10, -6)

/-- Line through Q with slope m -/
def line (m : ℝ) (x : ℝ) : ℝ := m * (x - Q.1) + Q.2

/-- Condition for line not intersecting parabola -/
def no_intersection (m : ℝ) : Prop :=
  ∀ x, line m x ≠ parabola x

/-- Theorem stating the sum of critical slopes -/
theorem sum_of_critical_slopes :
  ∃ r s, (∀ m, r < m ∧ m < s ↔ no_intersection m) ∧ r + s = 40 := by sorry

end NUMINAMATH_CALUDE_sum_of_critical_slopes_l1036_103631


namespace NUMINAMATH_CALUDE_gcd_228_1995_l1036_103667

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l1036_103667


namespace NUMINAMATH_CALUDE_nadia_mistakes_l1036_103664

/-- Calculates the number of mistakes made by a piano player given their error rate, playing speed, and duration of play. -/
def calculate_mistakes (mistakes_per_block : ℕ) (notes_per_block : ℕ) (notes_per_minute : ℕ) (minutes_played : ℕ) : ℕ :=
  let total_notes := notes_per_minute * minutes_played
  let num_blocks := total_notes / notes_per_block
  num_blocks * mistakes_per_block

/-- Theorem stating that under the given conditions, Nadia will make 36 mistakes on average when playing for 8 minutes. -/
theorem nadia_mistakes :
  calculate_mistakes 3 40 60 8 = 36 := by
  sorry

end NUMINAMATH_CALUDE_nadia_mistakes_l1036_103664


namespace NUMINAMATH_CALUDE_greatest_integer_no_substring_divisible_by_9_all_substrings_of_88888888_not_divisible_by_9_l1036_103632

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

end NUMINAMATH_CALUDE_greatest_integer_no_substring_divisible_by_9_all_substrings_of_88888888_not_divisible_by_9_l1036_103632


namespace NUMINAMATH_CALUDE_mortgage_food_ratio_is_three_to_one_l1036_103658

/-- Esperanza's monthly finances -/
structure MonthlyFinances where
  rent : ℕ
  food_ratio : ℚ
  savings : ℕ
  tax_ratio : ℚ
  gross_salary : ℕ

/-- Calculate the ratio of mortgage bill to food expenses -/
def mortgage_to_food_ratio (finances : MonthlyFinances) : ℚ :=
  let food_expense := finances.food_ratio * finances.rent
  let taxes := finances.tax_ratio * finances.savings
  let total_expenses := finances.rent + food_expense + finances.savings + taxes
  let mortgage := finances.gross_salary - total_expenses
  mortgage / food_expense

/-- Theorem stating the ratio of mortgage bill to food expenses -/
theorem mortgage_food_ratio_is_three_to_one :
  let esperanza_finances : MonthlyFinances := {
    rent := 600,
    food_ratio := 3/5,
    savings := 2000,
    tax_ratio := 2/5,
    gross_salary := 4840
  }
  mortgage_to_food_ratio esperanza_finances = 3 := by
  sorry


end NUMINAMATH_CALUDE_mortgage_food_ratio_is_three_to_one_l1036_103658


namespace NUMINAMATH_CALUDE_basketball_expected_score_l1036_103685

/-- The expected score of a basketball player making two independent free throws -/
def expected_score (p : ℝ) : ℝ :=
  2 * p

theorem basketball_expected_score :
  let p : ℝ := 0.7  -- Probability of making a free throw
  expected_score p = 1.4 := by
  sorry

end NUMINAMATH_CALUDE_basketball_expected_score_l1036_103685


namespace NUMINAMATH_CALUDE_geometric_series_sum_l1036_103638

/-- Given a sequence a_n and its partial sum S_n, prove that S_20 = 400 -/
theorem geometric_series_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (a n + 1)^2 / 4) →
  (∀ n, a n = 2 * n - 1) →
  S 20 = 400 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l1036_103638


namespace NUMINAMATH_CALUDE_tetrahedron_volume_bound_l1036_103680

/-- A tetrahedron is represented by its six edge lengths -/
structure Tetrahedron where
  edge1 : ℝ
  edge2 : ℝ
  edge3 : ℝ
  edge4 : ℝ
  edge5 : ℝ
  edge6 : ℝ

/-- The volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- A tetrahedron with five edges not exceeding 1 -/
def FiveEdgesLimitedTetrahedron : Type :=
  { t : Tetrahedron // t.edge1 ≤ 1 ∧ t.edge2 ≤ 1 ∧ t.edge3 ≤ 1 ∧ t.edge4 ≤ 1 ∧ t.edge5 ≤ 1 }

theorem tetrahedron_volume_bound (t : FiveEdgesLimitedTetrahedron) :
  volume t.val ≤ 1/8 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_bound_l1036_103680


namespace NUMINAMATH_CALUDE_min_decimal_digits_l1036_103604

def fraction : ℚ := 987654321 / (2^30 * 5^6)

theorem min_decimal_digits (n : ℕ) : n = 30 ↔ 
  (∀ m : ℕ, m < n → ∃ k : ℕ, fraction * 10^m ≠ k) ∧ 
  (∃ k : ℕ, fraction * 10^n = k) :=
sorry

end NUMINAMATH_CALUDE_min_decimal_digits_l1036_103604


namespace NUMINAMATH_CALUDE_fraction_comparisons_l1036_103611

theorem fraction_comparisons :
  ∀ (a b c : ℚ),
  (0 < b) → (b < 1) → (a * b < a) ∧
  (0 < c) → (c < b) → (a * b > a * c) ∧
  (0 < b) → (b < 1) → (a < a / b) :=
by sorry

end NUMINAMATH_CALUDE_fraction_comparisons_l1036_103611


namespace NUMINAMATH_CALUDE_function_property_l1036_103677

/-- Given a function f: ℝ → ℝ defined as f(x) = (x+a)^3 where a is a real constant,
    if f(1+x) = -f(1-x) for all x ∈ ℝ, then f(2) + f(-2) = -26 -/
theorem function_property (a : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = (x + a)^3)
    (h2 : ∀ x, f (1 + x) = -f (1 - x)) :
  f 2 + f (-2) = -26 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l1036_103677


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1036_103610

def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}
def B : Set ℤ := {2, 4, 6, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1036_103610


namespace NUMINAMATH_CALUDE_circle_properties_l1036_103665

/-- Given a circle with equation x^2 + y^2 - 4x + 2y + 4 = 0, 
    its radius is 1 and its center coordinates are (2, -1) -/
theorem circle_properties : 
  ∃ (r : ℝ) (x₀ y₀ : ℝ), 
    (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 4 = 0 ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
    r = 1 ∧ x₀ = 2 ∧ y₀ = -1 :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l1036_103665


namespace NUMINAMATH_CALUDE_distance_to_focus_l1036_103618

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus (x : ℝ) : 
  let P : ℝ × ℝ := (x, (1/4) * x^2)
  let parabola := {(x, y) : ℝ × ℝ | y = (1/4) * x^2}
  P ∈ parabola → P.2 = 4 → ∃ F : ℝ × ℝ, F.2 = 1/4 ∧ dist P F = 5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_focus_l1036_103618


namespace NUMINAMATH_CALUDE_henry_twice_jill_age_l1036_103681

/-- Represents the ages of Henry and Jill and the time when Henry was twice Jill's age. -/
structure AgeRelation where
  henry_age : ℕ
  jill_age : ℕ
  years_ago : ℕ

/-- Theorem stating the conditions and the result to be proved. -/
theorem henry_twice_jill_age (ar : AgeRelation) : 
  ar.henry_age = 29 →
  ar.jill_age = 19 →
  ar.henry_age + ar.jill_age = 48 →
  ar.henry_age - ar.years_ago = 2 * (ar.jill_age - ar.years_ago) →
  ar.years_ago = 9 := by
  sorry

#check henry_twice_jill_age

end NUMINAMATH_CALUDE_henry_twice_jill_age_l1036_103681


namespace NUMINAMATH_CALUDE_parallelogram_side_sum_l1036_103699

/-- A parallelogram with side lengths 5, 10y-2, 3x+5, and 12 has x+y equal to 91/30 -/
theorem parallelogram_side_sum (x y : ℚ) : 
  (3 * x + 5 = 12) → (10 * y - 2 = 5) → x + y = 91 / 30 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_side_sum_l1036_103699


namespace NUMINAMATH_CALUDE_no_solutions_for_equation_l1036_103637

theorem no_solutions_for_equation : ¬∃ (x : Fin 8 → ℝ), 
  (2 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + 
  (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + (x 7)^2 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_equation_l1036_103637


namespace NUMINAMATH_CALUDE_collinear_vectors_x_value_l1036_103612

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (-1, x) (1, 2) → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_x_value_l1036_103612


namespace NUMINAMATH_CALUDE_unique_m_value_l1036_103607

-- Define the universe set U
def U : Set ℕ := {0, 1, 2, 3}

-- Define the set A as a function of m
def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m*x = 0}

-- Define the complement of A in U
def C_UA : Set ℕ := {1, 2}

-- Theorem statement
theorem unique_m_value : ∃! m : ℝ, A m = U \ C_UA := by sorry

end NUMINAMATH_CALUDE_unique_m_value_l1036_103607


namespace NUMINAMATH_CALUDE_det_special_matrix_l1036_103698

/-- The determinant of the matrix [[1, x, x^2], [1, x+1, (x+1)^2], [1, x, (x+1)^2]] is equal to x + 1 -/
theorem det_special_matrix (x : ℝ) : 
  Matrix.det !![1, x, x^2; 1, x+1, (x+1)^2; 1, x, (x+1)^2] = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l1036_103698


namespace NUMINAMATH_CALUDE_ollie_fewer_than_angus_l1036_103670

/-- The number of fish caught by Patrick -/
def patrick_fish : ℕ := 8

/-- The number of fish caught by Ollie -/
def ollie_fish : ℕ := 5

/-- The number of fish caught by Angus -/
def angus_fish : ℕ := patrick_fish + 4

/-- The difference between Angus's and Ollie's fish catch -/
def fish_difference : ℕ := angus_fish - ollie_fish

theorem ollie_fewer_than_angus : fish_difference = 7 := by sorry

end NUMINAMATH_CALUDE_ollie_fewer_than_angus_l1036_103670


namespace NUMINAMATH_CALUDE_thomson_incentive_spending_l1036_103646

theorem thomson_incentive_spending (incentive : ℝ) (savings : ℝ) (f : ℝ) : 
  incentive = 240 →
  savings = 84 →
  savings = (3/4) * (incentive - f * incentive - (1/5) * incentive) →
  f = 1/3 := by
sorry

end NUMINAMATH_CALUDE_thomson_incentive_spending_l1036_103646


namespace NUMINAMATH_CALUDE_least_multiple_17_above_500_l1036_103671

theorem least_multiple_17_above_500 : ∃ (n : ℕ), n * 17 = 510 ∧ 
  510 > 500 ∧ 
  (∀ m : ℕ, m * 17 > 500 → m * 17 ≥ 510) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_17_above_500_l1036_103671


namespace NUMINAMATH_CALUDE_min_p_minus_q_equals_zero_l1036_103615

theorem min_p_minus_q_equals_zero
  (x y p q : ℤ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (eq1 : (3 : ℚ) / (x * p) = 8)
  (eq2 : (5 : ℚ) / (y * q) = 18)
  (hmin : ∀ x' y' p' q' : ℤ,
    x' ≠ 0 → y' ≠ 0 → p' ≠ 0 → q' ≠ 0 →
    (3 : ℚ) / (x' * p') = 8 →
    (5 : ℚ) / (y' * q') = 18 →
    (x' ≤ x ∧ y' ≤ y)) :
  p - q = 0 := by
sorry

end NUMINAMATH_CALUDE_min_p_minus_q_equals_zero_l1036_103615


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_l1036_103655

theorem imaginary_part_of_complex (z : ℂ) (h : z = -4 * Complex.I + 3) : 
  z.im = -4 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_l1036_103655


namespace NUMINAMATH_CALUDE_table_length_is_77_l1036_103620

/-- The length of a rectangular table covered by overlapping paper sheets. -/
def table_length : ℕ :=
  let table_width : ℕ := 80
  let sheet_width : ℕ := 8
  let sheet_height : ℕ := 5
  let offset : ℕ := 1
  let sheets_needed : ℕ := table_width - sheet_width
  sheet_height + sheets_needed

theorem table_length_is_77 : table_length = 77 := by
  sorry

end NUMINAMATH_CALUDE_table_length_is_77_l1036_103620


namespace NUMINAMATH_CALUDE_time_difference_per_mile_l1036_103613

-- Define the given conditions
def young_girl_distance : ℝ := 18  -- miles
def young_girl_time : ℝ := 135     -- minutes (2 hours and 15 minutes)
def current_distance : ℝ := 12     -- miles
def current_time : ℝ := 300        -- minutes (5 hours)

-- Define the theorem
theorem time_difference_per_mile : 
  (current_time / current_distance) - (young_girl_time / young_girl_distance) = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_time_difference_per_mile_l1036_103613


namespace NUMINAMATH_CALUDE_complex_determinant_equation_l1036_103600

-- Define the determinant operation
def det (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the theorem
theorem complex_determinant_equation :
  ∀ z : ℂ, det 1 (-1) z (z * Complex.I) = 2 → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_determinant_equation_l1036_103600


namespace NUMINAMATH_CALUDE_abc_fraction_value_l1036_103633

theorem abc_fraction_value (a b c : ℝ) 
  (eq1 : a * b / (a + b) = 3)
  (eq2 : b * c / (b + c) = 5)
  (eq3 : c * a / (c + a) = 8) :
  a * b * c / (a * b + b * c + c * a) = 240 / 79 := by
  sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l1036_103633


namespace NUMINAMATH_CALUDE_football_players_count_l1036_103630

theorem football_players_count (cricket_players hockey_players softball_players total_players : ℕ) 
  (h1 : cricket_players = 10)
  (h2 : hockey_players = 12)
  (h3 : softball_players = 13)
  (h4 : total_players = 51) :
  total_players - (cricket_players + hockey_players + softball_players) = 16 := by
  sorry

end NUMINAMATH_CALUDE_football_players_count_l1036_103630


namespace NUMINAMATH_CALUDE_mac_loss_calculation_l1036_103657

-- Define exchange rates
def canadian_dime_usd : ℝ := 0.075
def canadian_penny_usd : ℝ := 0.0075
def mexican_centavo_usd : ℝ := 0.0045
def cuban_centavo_usd : ℝ := 0.0036
def euro_cent_usd : ℝ := 0.011
def uk_pence_usd : ℝ := 0.013
def canadian_nickel_usd : ℝ := 0.038
def us_half_dollar_usd : ℝ := 0.5
def brazilian_centavo_usd : ℝ := 0.0019
def australian_cent_usd : ℝ := 0.0072
def indian_paisa_usd : ℝ := 0.0013
def mexican_peso_usd : ℝ := 0.045
def japanese_yen_usd : ℝ := 0.0089

-- Define daily trades
def day1_trade : ℝ := 6 * canadian_dime_usd + 2 * canadian_penny_usd
def day2_trade : ℝ := 10 * mexican_centavo_usd + 5 * cuban_centavo_usd
def day3_trade : ℝ := 4 * 0.1 + 1 * euro_cent_usd
def day4_trade : ℝ := 7 * uk_pence_usd + 5 * canadian_nickel_usd
def day5_trade : ℝ := 3 * us_half_dollar_usd + 2 * brazilian_centavo_usd
def day6_trade : ℝ := 12 * australian_cent_usd + 3 * indian_paisa_usd
def day7_trade : ℝ := 8 * mexican_peso_usd + 6 * japanese_yen_usd

-- Define quarter value
def quarter_value : ℝ := 0.25

-- Theorem statement
theorem mac_loss_calculation :
  (day1_trade - quarter_value) +
  (quarter_value - day2_trade) +
  (day3_trade - quarter_value) +
  (day4_trade - quarter_value) +
  (day5_trade - quarter_value) +
  (quarter_value - day6_trade) +
  (day7_trade - quarter_value) = 2.1619 :=
by sorry

end NUMINAMATH_CALUDE_mac_loss_calculation_l1036_103657


namespace NUMINAMATH_CALUDE_min_value_sum_of_roots_l1036_103619

theorem min_value_sum_of_roots (x : ℝ) :
  ∃ (y : ℝ), ∀ (x : ℝ),
    Real.sqrt (x^2 + (1 - x)^2) + Real.sqrt ((1 - x)^2 + (1 - x)^2) ≥ y ∧
    (∃ (z : ℝ), Real.sqrt (z^2 + (1 - z)^2) + Real.sqrt ((1 - z)^2 + (1 - z)^2) = y) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_roots_l1036_103619
