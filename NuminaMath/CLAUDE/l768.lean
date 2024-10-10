import Mathlib

namespace partnership_profit_distribution_l768_76805

/-- Partnership profit distribution problem -/
theorem partnership_profit_distribution (total_profit : ℚ) 
  (hA : ℚ) (hB : ℚ) (hC : ℚ) (hD : ℚ) :
  hA = 1/3 →
  hB = 1/4 →
  hC = 1/5 →
  hD = 1 - (hA + hB + hC) →
  total_profit = 2415 →
  hA * total_profit = 805 :=
by sorry

end partnership_profit_distribution_l768_76805


namespace point_in_first_or_third_quadrant_l768_76851

/-- A point is in the first or third quadrant if the product of its coordinates is positive -/
theorem point_in_first_or_third_quadrant (x y : ℝ) :
  x * y > 0 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
by sorry

end point_in_first_or_third_quadrant_l768_76851


namespace min_distance_between_points_l768_76804

/-- The minimum distance between points A(x, √2-x) and B(√2/2, 0) is 1/2 -/
theorem min_distance_between_points :
  let A : ℝ → ℝ × ℝ := λ x ↦ (x, Real.sqrt 2 - x)
  let B : ℝ × ℝ := (Real.sqrt 2 / 2, 0)
  ∃ (min_dist : ℝ), min_dist = 1/2 ∧
    ∀ x, Real.sqrt ((A x).1 - B.1)^2 + ((A x).2 - B.2)^2 ≥ min_dist :=
by sorry

end min_distance_between_points_l768_76804


namespace integral_one_plus_sin_l768_76844

theorem integral_one_plus_sin : ∫ x in -Real.pi..Real.pi, (1 + Real.sin x) = 2 * Real.pi := by sorry

end integral_one_plus_sin_l768_76844


namespace rationalize_and_simplify_l768_76855

theorem rationalize_and_simplify : 
  3 / (Real.sqrt 75 + Real.sqrt 3) = Real.sqrt 3 / 6 := by
  sorry

end rationalize_and_simplify_l768_76855


namespace sum_remainder_mod_nine_l768_76868

theorem sum_remainder_mod_nine : (88000 + 88002 + 87999 + 88001 + 88003 + 87998) % 9 = 0 := by
  sorry

end sum_remainder_mod_nine_l768_76868


namespace additional_grazing_area_l768_76870

theorem additional_grazing_area (π : ℝ) (h : π > 0) : 
  π * 23^2 - π * 16^2 = 273 * π := by
  sorry

end additional_grazing_area_l768_76870


namespace sally_coins_theorem_l768_76835

def initial_pennies : ℕ := 8
def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def mom_nickels : ℕ := 2
def penny_value : ℕ := 1
def nickel_value : ℕ := 5

theorem sally_coins_theorem :
  let total_nickels := initial_nickels + dad_nickels + mom_nickels
  let total_value := initial_pennies * penny_value + total_nickels * nickel_value
  total_nickels = 18 ∧ total_value = 98 := by sorry

end sally_coins_theorem_l768_76835


namespace blue_balls_count_l768_76830

/-- Given a jar with white and blue balls in a 5:3 ratio, 
    prove that 15 white balls implies 9 blue balls -/
theorem blue_balls_count (white_balls blue_balls : ℕ) : 
  (white_balls : ℚ) / blue_balls = 5 / 3 → 
  white_balls = 15 → 
  blue_balls = 9 := by
sorry

end blue_balls_count_l768_76830


namespace range_of_a_l768_76817

/-- The set A defined by the quadratic inequality -/
def A : Set ℝ := {x | x^2 + 2*x - 8 > 0}

/-- The set B defined by the distance from a point a -/
def B (a : ℝ) : Set ℝ := {x | |x - a| < 5}

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (h : A ∪ B a = Set.univ) : a ∈ Set.Icc (-3) 1 := by
  sorry

end range_of_a_l768_76817


namespace triangle_properties_l768_76856

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Main theorem about the triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c * (1 + Real.cos t.A) = Real.sqrt 3 * t.a * Real.sin t.C)
  (h2 : t.a = Real.sqrt 7)
  (h3 : t.b = 1) :
  t.A = π / 3 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = (3 : ℝ) * Real.sqrt 3 / 4 := by
  sorry

end triangle_properties_l768_76856


namespace always_even_l768_76872

def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

def change_sign (n : ℕ) (k : ℕ) : ℤ :=
  (sum_to_n n : ℤ) - 2 * k

theorem always_even (n : ℕ) (k : ℕ) (h1 : n = 1995) (h2 : k ≤ n) :
  Even (change_sign n k) := by
  sorry

end always_even_l768_76872


namespace convention_handshakes_l768_76897

/-- The number of companies at the convention -/
def num_companies : ℕ := 5

/-- The number of representatives from each company -/
def reps_per_company : ℕ := 5

/-- The total number of people at the convention -/
def total_people : ℕ := num_companies * reps_per_company

/-- The number of people each person shakes hands with -/
def handshakes_per_person : ℕ := total_people - reps_per_company - 1

/-- The total number of handshakes at the convention -/
def total_handshakes : ℕ := (total_people * handshakes_per_person) / 2

theorem convention_handshakes :
  total_handshakes = 250 :=
by sorry

end convention_handshakes_l768_76897


namespace min_draws_for_target_color_l768_76879

/- Define the number of balls for each color -/
def red_balls : ℕ := 34
def green_balls : ℕ := 25
def yellow_balls : ℕ := 23
def blue_balls : ℕ := 18
def white_balls : ℕ := 14
def black_balls : ℕ := 10

/- Define the target number of balls of a single color -/
def target : ℕ := 20

/- Define the total number of balls -/
def total_balls : ℕ := red_balls + green_balls + yellow_balls + blue_balls + white_balls + black_balls

/- Theorem statement -/
theorem min_draws_for_target_color :
  ∃ (n : ℕ), n = 100 ∧
  (∀ (m : ℕ), m < n → 
    ∃ (r g y b w k : ℕ), 
      r ≤ red_balls ∧ 
      g ≤ green_balls ∧ 
      y ≤ yellow_balls ∧ 
      b ≤ blue_balls ∧ 
      w ≤ white_balls ∧ 
      k ≤ black_balls ∧
      r + g + y + b + w + k = m ∧
      r < target ∧ g < target ∧ y < target ∧ b < target ∧ w < target ∧ k < target) ∧
  (∀ (r g y b w k : ℕ),
    r ≤ red_balls →
    g ≤ green_balls →
    y ≤ yellow_balls →
    b ≤ blue_balls →
    w ≤ white_balls →
    k ≤ black_balls →
    r + g + y + b + w + k = n →
    r ≥ target ∨ g ≥ target ∨ y ≥ target ∨ b ≥ target ∨ w ≥ target ∨ k ≥ target) :=
by sorry

end min_draws_for_target_color_l768_76879


namespace tulip_area_l768_76802

/-- Given a flower bed with roses and tulips, calculate the area occupied by tulips -/
theorem tulip_area (total_area : Real) (rose_fraction : Real) (tulip_fraction : Real) 
  (h1 : total_area = 2.4)
  (h2 : rose_fraction = 1/3)
  (h3 : tulip_fraction = 1/4) :
  tulip_fraction * (total_area - rose_fraction * total_area) = 0.4 := by
  sorry

#check tulip_area

end tulip_area_l768_76802


namespace hyperbola_eccentricity_l768_76812

/-- The eccentricity of a hyperbola with the given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (A B F : ℝ × ℝ) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  -- F is the right focus of C
  (F.1 = c ∧ F.2 = 0) →
  -- A and B are on the asymptotes of C
  (A.2 = (b / a) * A.1 ∧ B.2 = -(b / a) * B.1) →
  -- AF is perpendicular to the x-axis
  (A.1 = c ∧ A.2 = (b * c) / a) →
  -- AB is perpendicular to OB
  ((A.2 - B.2) / (A.1 - B.1) = a / b) →
  -- BF is parallel to OA
  ((F.2 - B.2) / (F.1 - B.1) = A.2 / A.1) →
  -- The eccentricity of the hyperbola
  c / a = 2 * Real.sqrt 3 / 3 :=
by sorry

end hyperbola_eccentricity_l768_76812


namespace xyz_value_l768_76816

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9)
  (h3 : x + y + z = 3) : 
  x * y * z = 5 := by
  sorry

end xyz_value_l768_76816


namespace most_cost_effective_plan_l768_76865

-- Define the prices of A and B type devices
def price_A : ℕ := 12000
def price_B : ℕ := 10000

-- Define the production capacities
def capacity_A : ℕ := 240
def capacity_B : ℕ := 180

-- Define the total number of devices to purchase
def total_devices : ℕ := 10

-- Define the budget constraint
def budget : ℕ := 110000

-- Define the minimum required production capacity
def min_capacity : ℕ := 2040

-- Theorem statement
theorem most_cost_effective_plan :
  ∃ (num_A num_B : ℕ),
    -- The total number of devices is 10
    num_A + num_B = total_devices ∧
    -- The total cost is within budget
    num_A * price_A + num_B * price_B ≤ budget ∧
    -- The total production capacity meets the minimum requirement
    num_A * capacity_A + num_B * capacity_B ≥ min_capacity ∧
    -- This is the most cost-effective plan
    ∀ (other_A other_B : ℕ),
      other_A + other_B = total_devices →
      other_A * capacity_A + other_B * capacity_B ≥ min_capacity →
      other_A * price_A + other_B * price_B ≥ num_A * price_A + num_B * price_B :=
by
  -- The proof goes here
  sorry

#check most_cost_effective_plan

end most_cost_effective_plan_l768_76865


namespace bruce_triple_age_l768_76818

/-- Bruce's current age -/
def bruce_age : ℕ := 36

/-- Bruce's son's current age -/
def son_age : ℕ := 8

/-- The number of years it will take for Bruce to be three times as old as his son -/
def years_until_triple : ℕ := 6

/-- Theorem stating that in 6 years, Bruce will be three times as old as his son -/
theorem bruce_triple_age :
  bruce_age + years_until_triple = 3 * (son_age + years_until_triple) :=
sorry

end bruce_triple_age_l768_76818


namespace fraction_simplification_l768_76875

theorem fraction_simplification (x y : ℝ) (h : x / y = 2 / 5) :
  (3 * y - 2 * x) / (3 * y + 2 * x) = 11 / 19 := by
  sorry

end fraction_simplification_l768_76875


namespace school_trip_theorem_l768_76849

/-- Represents the number of people initially planned per bus -/
def initial_people_per_bus : ℕ := 28

/-- Represents the number of students who couldn't get on the buses -/
def students_left_behind : ℕ := 13

/-- Represents the final number of people per bus -/
def final_people_per_bus : ℕ := 32

/-- Represents the number of empty seats per bus after redistribution -/
def empty_seats_per_bus : ℕ := 3

/-- Proves that the number of third-grade students is 125 and the number of buses rented is 4 -/
theorem school_trip_theorem :
  ∃ (num_students num_buses : ℕ),
    num_students = 125 ∧
    num_buses = 4 ∧
    num_students = initial_people_per_bus * num_buses + students_left_behind ∧
    num_students = final_people_per_bus * num_buses - empty_seats_per_bus * num_buses :=
by
  sorry

end school_trip_theorem_l768_76849


namespace solution_completeness_l768_76899

def is_integer (q : ℚ) : Prop := ∃ n : ℤ, q = n

def satisfies_conditions (x y z : ℚ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  x ≤ y ∧ y ≤ z ∧
  is_integer (x + y + z) ∧
  is_integer (1/x + 1/y + 1/z) ∧
  is_integer (x * y * z)

def solution_set : Set (ℚ × ℚ × ℚ) :=
  {(1, 1, 1), (1, 2, 2), (2, 3, 6), (2, 4, 4), (3, 3, 3)}

theorem solution_completeness :
  ∀ x y z : ℚ, satisfies_conditions x y z ↔ (x, y, z) ∈ solution_set :=
sorry

end solution_completeness_l768_76899


namespace parallel_lines_corresponding_angles_l768_76820

-- Define the types for lines and angles
variable (Line : Type) (Angle : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (corresponding_angles : Line → Line → Angle → Angle → Prop)
variable (equal_angles : Angle → Angle → Prop)

-- State the theorem
theorem parallel_lines_corresponding_angles 
  (l1 l2 : Line) (a1 a2 : Angle) : 
  (parallel l1 l2 → corresponding_angles l1 l2 a1 a2 → equal_angles a1 a2) ∧
  (corresponding_angles l1 l2 a1 a2 → equal_angles a1 a2 → parallel l1 l2) ∧
  (¬parallel l1 l2 → corresponding_angles l1 l2 a1 a2 → ¬equal_angles a1 a2) ∧
  (corresponding_angles l1 l2 a1 a2 → ¬equal_angles a1 a2 → ¬parallel l1 l2) :=
by sorry

end parallel_lines_corresponding_angles_l768_76820


namespace count_integers_with_repeated_digits_is_1192_l768_76889

/-- A function that counts the number of positive four-digit integers less than 5000 
    with at least two identical digits -/
def count_integers_with_repeated_digits : ℕ :=
  let lower_bound := 1000
  let upper_bound := 4999
  sorry

/-- Theorem stating that the count of positive four-digit integers less than 5000 
    with at least two identical digits is 1192 -/
theorem count_integers_with_repeated_digits_is_1192 : 
  count_integers_with_repeated_digits = 1192 := by sorry

end count_integers_with_repeated_digits_is_1192_l768_76889


namespace divisors_product_prime_factors_l768_76848

theorem divisors_product_prime_factors :
  let divisors : List Nat := [1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60]
  let A : Nat := divisors.prod
  (Nat.factors A).toFinset.card = 3 := by
  sorry

end divisors_product_prime_factors_l768_76848


namespace complex_abs_3_minus_10i_l768_76854

theorem complex_abs_3_minus_10i :
  let z : ℂ := 3 - 10*I
  Complex.abs z = Real.sqrt 109 := by
  sorry

end complex_abs_3_minus_10i_l768_76854


namespace square_area_tripled_side_l768_76800

theorem square_area_tripled_side (s : ℝ) (h : s > 0) :
  (3 * s)^2 = 9 * s^2 :=
by sorry

end square_area_tripled_side_l768_76800


namespace all_fruits_fallen_on_day_10_l768_76858

/-- Represents the number of fruits that fall on a given day -/
def fruitsFallingOnDay (day : ℕ) : ℕ :=
  if day % 9 = 0 then 9 else day % 9

/-- Represents the total number of fruits that have fallen up to and including a given day -/
def totalFruitsFallen (day : ℕ) : ℕ :=
  (day / 9) * 45 + (day % 9) * (day % 9 + 1) / 2

/-- The theorem stating that all fruits will have fallen after 10 days -/
theorem all_fruits_fallen_on_day_10 (initial_fruits : ℕ) (h : initial_fruits = 46) :
  totalFruitsFallen 10 = initial_fruits :=
sorry

end all_fruits_fallen_on_day_10_l768_76858


namespace combined_average_age_l768_76803

theorem combined_average_age (people_a : ℕ) (people_b : ℕ) (avg_age_a : ℝ) (avg_age_b : ℝ)
  (h1 : people_a = 8)
  (h2 : people_b = 2)
  (h3 : avg_age_a = 38)
  (h4 : avg_age_b = 30) :
  (people_a * avg_age_a + people_b * avg_age_b) / (people_a + people_b) = 36.4 := by
  sorry

end combined_average_age_l768_76803


namespace factor_equality_l768_76862

theorem factor_equality (x y : ℝ) : 9*x^2 - y^2 - 4*y - 4 = (3*x + y + 2)*(3*x - y - 2) := by
  sorry

end factor_equality_l768_76862


namespace no_solution_for_equation_l768_76845

theorem no_solution_for_equation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 0 → (6 / (x - 1) - (x + 5) / (x^2 - x) ≠ 0) := by
  sorry

end no_solution_for_equation_l768_76845


namespace black_area_after_seven_changes_l768_76837

/-- Represents the fraction of black area remaining after a number of changes -/
def blackFraction (changes : ℕ) : ℚ :=
  (8/9) ^ changes

/-- The number of changes applied to the triangle -/
def numChanges : ℕ := 7

/-- Theorem stating the fraction of black area after seven changes -/
theorem black_area_after_seven_changes :
  blackFraction numChanges = 2097152/4782969 := by
  sorry

#eval blackFraction numChanges

end black_area_after_seven_changes_l768_76837


namespace opposite_of_negative_half_l768_76859

theorem opposite_of_negative_half :
  -(-(1/2 : ℚ)) = 1/2 := by
  sorry

end opposite_of_negative_half_l768_76859


namespace three_color_theorem_l768_76864

theorem three_color_theorem : ∃ f : ℕ → Fin 3,
  (∀ n : ℕ, ∀ x y : ℕ, 2^n ≤ x ∧ x < 2^(n+1) ∧ 2^n ≤ y ∧ y < 2^(n+1) → f x = f y) ∧
  (∀ x y z : ℕ, f x = f y ∧ f y = f z ∧ x + y = z^2 → x = 2 ∧ y = 2 ∧ z = 2) :=
sorry

end three_color_theorem_l768_76864


namespace special_function_property_l768_76839

/-- A function satisfying the given property for all real x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^3 + y^3) = (x + y) * (f x^2 - f x * f y + f y^2)

/-- Theorem stating that if f is a special function, then f(1996x) = 1996f(x) for all real x -/
theorem special_function_property (f : ℝ → ℝ) (h : special_function f) :
  ∀ x : ℝ, f (1996 * x) = 1996 * f x := by
  sorry


end special_function_property_l768_76839


namespace f_one_half_equals_sixteen_l768_76852

-- Define the function f
noncomputable def f : ℝ → ℝ := fun t => 1 / ((1 - t) / 2)^2

-- State the theorem
theorem f_one_half_equals_sixteen :
  (∀ x, f (1 - 2 * x) = 1 / x^2) → f (1/2) = 16 :=
by
  sorry

end f_one_half_equals_sixteen_l768_76852


namespace luke_coin_count_l768_76828

theorem luke_coin_count : 
  ∀ (quarter_piles dime_piles coins_per_pile : ℕ),
    quarter_piles = 5 →
    dime_piles = 5 →
    coins_per_pile = 3 →
    quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 30 :=
by
  sorry

end luke_coin_count_l768_76828


namespace quintuplet_babies_count_l768_76808

/-- Represents the number of sets of a given multiple birth type -/
structure MultipleBirthSets where
  twins : ℕ
  triplets : ℕ
  quintuplets : ℕ

/-- Calculates the total number of babies from multiple birth sets -/
def totalBabies (s : MultipleBirthSets) : ℕ :=
  2 * s.twins + 3 * s.triplets + 5 * s.quintuplets

theorem quintuplet_babies_count (s : MultipleBirthSets) :
  s.triplets = 6 * s.quintuplets →
  s.twins = 2 * s.triplets →
  totalBabies s = 1500 →
  5 * s.quintuplets = 160 := by
sorry

end quintuplet_babies_count_l768_76808


namespace initial_women_count_l768_76888

theorem initial_women_count (x y : ℕ) : 
  (y / (x - 15) = 2) → 
  ((y - 45) / (x - 15) = 1 / 5) → 
  x = 40 := by
  sorry

end initial_women_count_l768_76888


namespace min_value_expression_l768_76807

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min : ℝ), min = 3 ∧ (∀ x y : ℝ, x > 0 → y > 0 → 
    (Real.sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x*y) ≥ min) ∧
    ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
      (Real.sqrt ((x^2 + y^2) * (4*x^2 + y^2))) / (x*y) = min :=
sorry

end min_value_expression_l768_76807


namespace batman_game_cost_batman_game_cost_proof_l768_76850

def total_spent : ℝ := 35.52
def football_cost : ℝ := 14.02
def strategy_cost : ℝ := 9.46

theorem batman_game_cost : ℝ := by
  sorry

theorem batman_game_cost_proof : batman_game_cost = 12.04 := by
  sorry

end batman_game_cost_batman_game_cost_proof_l768_76850


namespace fraction_to_decimal_l768_76896

theorem fraction_to_decimal : (7 : ℚ) / 125 = 0.056 := by
  sorry

end fraction_to_decimal_l768_76896


namespace code_cracking_probabilities_l768_76832

/-- The probability of person A succeeding -/
def prob_A : ℚ := 1/2

/-- The probability of person B succeeding -/
def prob_B : ℚ := 3/5

/-- The probability of person C succeeding -/
def prob_C : ℚ := 3/4

/-- The probability of exactly one person succeeding -/
def prob_exactly_one : ℚ := 
  prob_A * (1 - prob_B) * (1 - prob_C) + 
  (1 - prob_A) * prob_B * (1 - prob_C) + 
  (1 - prob_A) * (1 - prob_B) * prob_C

/-- The probability of the code being successfully cracked -/
def prob_success : ℚ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- The minimum number of people like C needed for at least 95% success rate -/
def min_people_C : ℕ := 3

theorem code_cracking_probabilities :
  prob_exactly_one = 11/40 ∧ 
  prob_success = 19/20 ∧
  (∀ n : ℕ, n ≥ min_people_C → 1 - (1 - prob_C)^n ≥ 95/100) ∧
  (∀ n : ℕ, n < min_people_C → 1 - (1 - prob_C)^n < 95/100) :=
sorry

end code_cracking_probabilities_l768_76832


namespace inequality_solution_set_inequality_proof_l768_76846

-- Define the function f(x) = |x-1|
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for part (1)
theorem inequality_solution_set (x : ℝ) :
  f x + f (x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3 := by
  sorry

-- Theorem for part (2)
theorem inequality_proof (a b : ℝ) (ha : a ≠ 0) (h1 : f (a + 1) < 1) (h2 : f (b + 1) < 1) :
  f (a * b) / |a| > f (b / a) := by
  sorry

end inequality_solution_set_inequality_proof_l768_76846


namespace problem_1_problem_2_problem_3_problem_4_l768_76867

-- Problem 1
theorem problem_1 (a : ℝ) : a * a^3 - 5 * a^4 + (2 * a^2)^2 = 0 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : (2 * a + 3 * b) * (a - 2 * b) - 1/8 * a * (4 * a - 3 * b) = 3/2 * a^2 - 5/8 * a * b - 6 * b^2 := by sorry

-- Problem 3
theorem problem_3 : (-0.125)^2023 * 2^2024 * 4^2024 = -8 := by sorry

-- Problem 4
theorem problem_4 : (2 * (1/2 : ℝ) - (-1))^2 + ((1/2 : ℝ) - (-1)) * ((1/2 : ℝ) + (-1)) - 5 * (1/2 : ℝ) * ((1/2 : ℝ) - 2 * (-1)) = -3 := by sorry

end problem_1_problem_2_problem_3_problem_4_l768_76867


namespace hcf_problem_l768_76843

theorem hcf_problem (a b hcf : ℕ) (h1 : a = 391) (h2 : a ≥ b) 
  (h3 : ∃ (lcm : ℕ), lcm = hcf * 16 * 17 ∧ lcm = a * b / hcf) : hcf = 23 := by
  sorry

end hcf_problem_l768_76843


namespace runner_problem_l768_76842

theorem runner_problem (v : ℝ) (h : v > 0) : 
  (40 / v = 8) → (8 - 20 / v = 4) := by sorry

end runner_problem_l768_76842


namespace gumball_machine_total_l768_76847

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- Defines the properties of the gumball machine -/
def validGumballMachine (m : GumballMachine) : Prop :=
  m.red = 16 ∧
  m.blue = m.red / 2 ∧
  m.green = 4 * m.blue

/-- Calculates the total number of gumballs in the machine -/
def totalGumballs (m : GumballMachine) : ℕ :=
  m.red + m.blue + m.green

/-- Theorem stating that a valid gumball machine has 56 gumballs in total -/
theorem gumball_machine_total (m : GumballMachine) 
  (h : validGumballMachine m) : totalGumballs m = 56 := by
  sorry

end gumball_machine_total_l768_76847


namespace shaded_fraction_of_semicircle_l768_76893

/-- Given a larger semicircle with diameter 4 and a smaller semicircle removed from it,
    where the two semicircles touch at exactly three points, prove that the fraction
    of the larger semicircle that remains shaded is 1/2. -/
theorem shaded_fraction_of_semicircle (R : ℝ) (r : ℝ) : 
  R = 2 →  -- Radius of larger semicircle
  r^2 + r^2 = (R - r)^2 →  -- Condition for touching at three points
  (π * R^2 / 2 - π * r^2 / 2) / (π * R^2 / 2) = 1 / 2 := by
  sorry

end shaded_fraction_of_semicircle_l768_76893


namespace ball_cost_l768_76822

/-- Proves that if Kyoko buys 3 balls for a total cost of $4.62, then each ball costs $1.54. -/
theorem ball_cost (total_cost : ℝ) (num_balls : ℕ) (cost_per_ball : ℝ) 
  (h1 : total_cost = 4.62)
  (h2 : num_balls = 3)
  (h3 : cost_per_ball = total_cost / num_balls) : 
  cost_per_ball = 1.54 := by
  sorry

end ball_cost_l768_76822


namespace point_on_x_axis_l768_76857

/-- If a point P(a-1, a+2) lies on the x-axis, then P = (-3, 0) -/
theorem point_on_x_axis (a : ℝ) : 
  (∃ P : ℝ × ℝ, P.1 = a - 1 ∧ P.2 = a + 2 ∧ P.2 = 0) → 
  ∃ P : ℝ × ℝ, P = (-3, 0) :=
by sorry

end point_on_x_axis_l768_76857


namespace probability_need_change_is_six_sevenths_l768_76876

/-- Represents the cost of a toy in cents -/
def ToyCost := Fin 8 → Nat

/-- The machine with 8 toys -/
structure ToyMachine where
  toys : Fin 8
  costs : ToyCost
  favorite_toy_cost : costs 3 = 175  -- $1.75 is the 4th most expensive toy (index 3)

/-- Sam's initial money in quarters -/
def initial_quarters : Nat := 8

/-- Probability of needing to get change -/
def probability_need_change (m : ToyMachine) : Rat :=
  1 - (1 : Rat) / 7

/-- Main theorem: The probability of needing change is 6/7 -/
theorem probability_need_change_is_six_sevenths (m : ToyMachine) :
  probability_need_change m = 6 / 7 := by
  sorry

/-- All costs are between 25 cents and 2 dollars, decreasing by 25 cents each time -/
axiom cost_constraint (m : ToyMachine) :
  ∀ i : Fin 8, m.costs i = 200 - 25 * i.val

/-- The machine randomly selects one of the remaining toys each time -/
axiom random_selection (m : ToyMachine) : True

/-- The machine only accepts quarters -/
axiom quarters_only (m : ToyMachine) : True

end probability_need_change_is_six_sevenths_l768_76876


namespace sum_real_coefficients_binomial_expansion_l768_76836

theorem sum_real_coefficients_binomial_expansion (i : ℂ) :
  let x : ℂ := Complex.I
  let n : ℕ := 1010
  let T : ℝ := (Finset.range (n + 1)).sum (λ k => if k % 2 = 0 then (n.choose k : ℝ) else 0)
  T = 2^(n - 1) :=
sorry

end sum_real_coefficients_binomial_expansion_l768_76836


namespace carmichael_561_l768_76831

theorem carmichael_561 (a : ℤ) : a ^ 561 ≡ a [ZMOD 561] := by
  sorry

end carmichael_561_l768_76831


namespace rhind_papyrus_bread_division_l768_76869

theorem rhind_papyrus_bread_division :
  ∀ (a d : ℚ),
    d > 0 →
    5 * a = 100 →
    (1 / 7) * (a + (a + d) + (a + 2 * d)) = (a - 2 * d) + (a - d) →
    a - 2 * d = 5 / 3 :=
by sorry

end rhind_papyrus_bread_division_l768_76869


namespace product_of_specific_roots_l768_76834

/-- Given distinct real numbers a, b, c, d satisfying specific equations, their product is 11 -/
theorem product_of_specific_roots (a b c d : ℝ) 
  (ha : a = Real.sqrt (4 + Real.sqrt (5 + a)))
  (hb : b = Real.sqrt (4 - Real.sqrt (5 + b)))
  (hc : c = Real.sqrt (4 + Real.sqrt (5 - c)))
  (hd : d = Real.sqrt (4 - Real.sqrt (5 - d)))
  (hdistinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) :
  a * b * c * d = 11 := by
  sorry

end product_of_specific_roots_l768_76834


namespace min_monkeys_correct_l768_76833

/-- Represents the problem of transporting weapons with monkeys --/
structure WeaponTransport where
  total_weight : ℕ
  max_weapon_weight : ℕ
  max_monkey_capacity : ℕ

/-- Calculates the minimum number of monkeys needed to transport all weapons --/
def min_monkeys_needed (wt : WeaponTransport) : ℕ :=
  23

/-- Theorem stating that the minimum number of monkeys needed is correct --/
theorem min_monkeys_correct (wt : WeaponTransport) 
  (h1 : wt.total_weight = 600)
  (h2 : wt.max_weapon_weight = 30)
  (h3 : wt.max_monkey_capacity = 50) :
  min_monkeys_needed wt = 23 ∧ 
  ∀ n : ℕ, n < 23 → ¬ (n * wt.max_monkey_capacity ≥ wt.total_weight) :=
sorry

end min_monkeys_correct_l768_76833


namespace total_cost_price_l768_76810

/-- Represents the cost and selling information for a fruit --/
structure Fruit where
  sellingPrice : ℚ
  lossRatio : ℚ

/-- Calculates the cost price of a fruit given its selling price and loss ratio --/
def costPrice (fruit : Fruit) : ℚ :=
  fruit.sellingPrice / (1 - fruit.lossRatio)

/-- The apple sold in the shop --/
def apple : Fruit := { sellingPrice := 30, lossRatio := 1/5 }

/-- The orange sold in the shop --/
def orange : Fruit := { sellingPrice := 45, lossRatio := 1/4 }

/-- The banana sold in the shop --/
def banana : Fruit := { sellingPrice := 15, lossRatio := 1/6 }

/-- Theorem stating the total cost price of all three fruits --/
theorem total_cost_price :
  costPrice apple + costPrice orange + costPrice banana = 115.5 := by
  sorry

end total_cost_price_l768_76810


namespace arithmetic_geometric_sequence_product_l768_76866

theorem arithmetic_geometric_sequence_product (a₁ a₂ b₁ b₂ b₃ : ℝ) : 
  ((-9 : ℝ) < a₁ ∧ a₁ < a₂ ∧ a₂ < (-1 : ℝ)) →  -- Arithmetic sequence condition
  ((-9 : ℝ) < b₁ ∧ b₁ < b₂ ∧ b₂ < b₃ ∧ b₃ < (-1 : ℝ)) →  -- Geometric sequence condition
  (a₂ - a₁ = a₁ - (-9 : ℝ)) →  -- Arithmetic sequence property
  (b₂ * b₂ = b₁ * b₃) →  -- Geometric sequence property
  (b₂ * (a₂ - a₁) = (-8 : ℝ)) := by
  sorry

end arithmetic_geometric_sequence_product_l768_76866


namespace equal_products_l768_76826

theorem equal_products : 2 * 20212021 * 1011 * 202320232023 = 43 * 47 * 20232023 * 202220222022 := by
  sorry

end equal_products_l768_76826


namespace cubic_roots_sum_l768_76892

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 4*a^2 + 6*a - 3 = 0) →
  (b^3 - 4*b^2 + 6*b - 3 = 0) →
  (c^3 - 4*c^2 + 6*c - 3 = 0) →
  (a/(b*c + 2) + b/(a*c + 2) + c/(a*b + 2) = 4/5) :=
by sorry

end cubic_roots_sum_l768_76892


namespace cheese_wedge_volume_l768_76821

/-- The volume of a wedge of cheese that represents one-third of a cylindrical log --/
theorem cheese_wedge_volume (d h r : ℝ) : 
  d = 12 →  -- diameter is 12 cm
  h = d →   -- height is equal to diameter
  r = d / 2 →  -- radius is half the diameter
  (1 / 3) * (π * r^2 * h) = 144 * π := by
  sorry

end cheese_wedge_volume_l768_76821


namespace sugar_for_muffins_sugar_for_muffins_proof_l768_76882

/-- Given that 45 muffins require 3 cups of sugar, 
    prove that 135 muffins require 9 cups of sugar. -/
theorem sugar_for_muffins : ℝ → ℝ → ℝ → Prop :=
  fun muffins_base sugar_base muffins_target =>
    (muffins_base = 45 ∧ sugar_base = 3) →
    (muffins_target = 135) →
    (muffins_target * sugar_base / muffins_base = 9)

/-- Proof of the theorem -/
theorem sugar_for_muffins_proof : sugar_for_muffins 45 3 135 := by
  sorry

end sugar_for_muffins_sugar_for_muffins_proof_l768_76882


namespace prob_three_non_defective_pencils_l768_76825

/-- The probability of selecting 3 non-defective pencils from a box of 8 pencils, where 2 are defective. -/
theorem prob_three_non_defective_pencils :
  let total_pencils : ℕ := 8
  let defective_pencils : ℕ := 2
  let selected_pencils : ℕ := 3
  let non_defective_pencils := total_pencils - defective_pencils
  Nat.choose non_defective_pencils selected_pencils / Nat.choose total_pencils selected_pencils = 5 / 14 :=
by sorry

end prob_three_non_defective_pencils_l768_76825


namespace inequality_of_roots_l768_76887

theorem inequality_of_roots (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end inequality_of_roots_l768_76887


namespace square_area_ratio_l768_76898

theorem square_area_ratio : 
  let a : ℝ := 36
  let b : ℝ := 42
  let c : ℝ := 54
  (a^2 + b^2) / c^2 = 255 / 243 := by
  sorry

end square_area_ratio_l768_76898


namespace no_functions_exist_l768_76827

theorem no_functions_exist : ¬ ∃ (f g : ℝ → ℝ), ∀ (x y : ℝ), f x * g y = x + y + 1 := by
  sorry

end no_functions_exist_l768_76827


namespace power_of_two_divisibility_l768_76877

theorem power_of_two_divisibility (n : ℕ+) :
  (∃ k : ℕ, 2^n.val - 1 = 7 * k) ↔ (∃ m : ℕ, n.val = 3 * m) ∧
  ¬(∃ k : ℕ, 2^n.val + 1 = 7 * k) :=
by sorry

end power_of_two_divisibility_l768_76877


namespace exercise_book_price_l768_76811

/-- The price of an exercise book in yuan -/
def price_per_book : ℚ := 0.55

/-- The number of books Xiaoming took -/
def xiaoming_books : ℕ := 8

/-- The number of books Xiaohong took -/
def xiaohong_books : ℕ := 12

/-- The amount Xiaohong gave to Xiaoming in yuan -/
def amount_given : ℚ := 1.1

theorem exercise_book_price :
  (xiaoming_books + xiaohong_books : ℚ) * price_per_book / 2 =
    (xiaoming_books : ℚ) * price_per_book + amount_given / 2 ∧
  (xiaoming_books + xiaohong_books : ℚ) * price_per_book / 2 =
    (xiaohong_books : ℚ) * price_per_book - amount_given / 2 :=
sorry

end exercise_book_price_l768_76811


namespace original_number_proof_l768_76824

theorem original_number_proof (N : ℕ) : N = 28 ↔ 
  (∃ k : ℕ, N - 11 = 17 * k) ∧ 
  (∀ x : ℕ, x < 11 → ¬(∃ m : ℕ, N - x = 17 * m)) ∧
  (∀ M : ℕ, M < N → ¬(∃ k : ℕ, M - 11 = 17 * k) ∨ 
    (∃ x : ℕ, x < 11 ∧ ∃ m : ℕ, M - x = 17 * m)) :=
by sorry

end original_number_proof_l768_76824


namespace expression_evaluation_l768_76885

theorem expression_evaluation (a b c : ℝ) (ha : a = 14) (hb : b = 19) (hc : c = 13) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = a + b + c := by
  sorry

end expression_evaluation_l768_76885


namespace sandy_molly_age_ratio_l768_76890

/-- The ratio of Sandy's age to Molly's age -/
def age_ratio (sandy_age molly_age : ℕ) : ℚ :=
  sandy_age / molly_age

/-- Theorem stating that the ratio of Sandy's age to Molly's age is 7/9 -/
theorem sandy_molly_age_ratio :
  let sandy_age : ℕ := 63
  let molly_age : ℕ := sandy_age + 18
  age_ratio sandy_age molly_age = 7/9 := by
sorry

end sandy_molly_age_ratio_l768_76890


namespace intersection_point_inside_circle_l768_76860

theorem intersection_point_inside_circle (a : ℝ) : 
  let line1 : ℝ → ℝ := λ x => x + 2 * a
  let line2 : ℝ → ℝ := λ x => 2 * x + a + 1
  let P : ℝ × ℝ := (a - 1, 3 * a - 1)
  (∀ x y, y = line1 x ∧ y = line2 x → (x, y) = P) →
  P.1^2 + P.2^2 < 4 →
  -1/5 < a ∧ a < 1 :=
by sorry

end intersection_point_inside_circle_l768_76860


namespace seashell_collection_l768_76861

/-- Theorem: Given an initial collection of 19 seashells and adding 6 more,
    the total number of seashells is 25. -/
theorem seashell_collection (initial : Nat) (added : Nat) (total : Nat) : 
  initial = 19 → added = 6 → total = initial + added → total = 25 := by
  sorry

end seashell_collection_l768_76861


namespace inscribed_cube_volume_in_equilateral_pyramid_l768_76814

/-- A pyramid with an equilateral triangular base and equilateral triangular lateral faces -/
structure EquilateralPyramid where
  base_side_length : ℝ
  lateral_face_is_equilateral : Bool

/-- A cube inscribed in a pyramid -/
structure InscribedCube where
  side_length : ℝ
  base_on_pyramid_base : Bool
  top_edges_on_lateral_faces : Bool

/-- The volume of the inscribed cube in the given pyramid -/
def inscribed_cube_volume (p : EquilateralPyramid) (c : InscribedCube) : ℝ :=
  c.side_length ^ 3

theorem inscribed_cube_volume_in_equilateral_pyramid 
  (p : EquilateralPyramid) 
  (c : InscribedCube) 
  (h1 : p.base_side_length = 2)
  (h2 : p.lateral_face_is_equilateral = true)
  (h3 : c.base_on_pyramid_base = true)
  (h4 : c.top_edges_on_lateral_faces = true) :
  inscribed_cube_volume p c = 3 * Real.sqrt 3 / 8 :=
sorry

end inscribed_cube_volume_in_equilateral_pyramid_l768_76814


namespace min_value_product_min_value_achieved_l768_76841

theorem min_value_product (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 6) : 
  x^3 * y^2 * z ≥ 1/108 := by
sorry

theorem min_value_achieved (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : 1/x + 1/y + 1/z = 6) :
  ∃ (x₀ y₀ z₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 1/x₀ + 1/y₀ + 1/z₀ = 6 ∧ x₀^3 * y₀^2 * z₀ = 1/108 := by
sorry

end min_value_product_min_value_achieved_l768_76841


namespace x_squared_eq_one_is_quadratic_l768_76894

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 1 -/
def f (x : ℝ) : ℝ := x^2 - 1

/-- Theorem: x^2 = 1 is a quadratic equation -/
theorem x_squared_eq_one_is_quadratic : is_quadratic_equation f :=
sorry

end x_squared_eq_one_is_quadratic_l768_76894


namespace football_yards_gained_l768_76829

theorem football_yards_gained (initial_loss : ℤ) (final_progress : ℤ) (yards_gained : ℤ) : 
  initial_loss = -5 → final_progress = 2 → yards_gained = initial_loss + final_progress →
  yards_gained = 7 := by
sorry

end football_yards_gained_l768_76829


namespace egg_roll_ratio_l768_76883

def matthew_egg_rolls : ℕ := 6
def alvin_egg_rolls : ℕ := 4
def patrick_egg_rolls : ℕ := alvin_egg_rolls / 2

theorem egg_roll_ratio :
  matthew_egg_rolls / patrick_egg_rolls = 3 := by
  sorry

end egg_roll_ratio_l768_76883


namespace work_completion_time_l768_76895

/-- Given two workers A and B, where A can complete a work in 10 days and B can complete the same work in 7 days, 
    this theorem proves that A and B working together can complete the work in 70/17 days. -/
theorem work_completion_time 
  (work : ℝ) -- Total amount of work
  (a_rate : ℝ) -- A's work rate
  (b_rate : ℝ) -- B's work rate
  (ha : a_rate = work / 10) -- A completes the work in 10 days
  (hb : b_rate = work / 7)  -- B completes the work in 7 days
  : work / (a_rate + b_rate) = 70 / 17 := by
sorry


end work_completion_time_l768_76895


namespace complex_calculation_proof_l768_76863

theorem complex_calculation_proof :
  let expr1 := (1) - (3^3) * ((-1/3)^2) - 24 * (3/4 - 1/6 + 3/8)
  let expr2 := (2) - (1^100) - (3/4) / (((-2)^2) * ((-1/4)^2) - 1/2)
  (expr1 = -26) ∧ (expr2 = 2) := by
sorry

end complex_calculation_proof_l768_76863


namespace russian_chess_championship_games_l768_76871

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 18 players, 153 games are played -/
theorem russian_chess_championship_games : 
  num_games 18 = 153 := by
  sorry

end russian_chess_championship_games_l768_76871


namespace fence_cost_l768_76838

/-- The cost of building a fence around a circular plot -/
theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 289 →
  price_per_foot = 58 →
  cost = 2 * (Real.sqrt (289 * Real.pi)) * price_per_foot →
  cost = 1972 :=
by
  sorry

#check fence_cost

end fence_cost_l768_76838


namespace gcd_product_l768_76881

theorem gcd_product (a b A B : ℕ) (d D : ℕ+) :
  d = Nat.gcd a b →
  D = Nat.gcd A B →
  (d * D : ℕ) = Nat.gcd (a * A) (Nat.gcd (a * B) (Nat.gcd (b * A) (b * B))) :=
by sorry

end gcd_product_l768_76881


namespace unique_k_exists_l768_76874

-- Define the sequence sum function
def S (n : ℕ) : ℤ := n^2 - 9*n

-- Define the k-th term of the sequence
def a (k : ℕ) : ℤ := S k - S (k-1)

-- State the theorem
theorem unique_k_exists (k : ℕ) :
  (∃ k, 5 < a k ∧ a k < 8) → k = 8 :=
sorry

end unique_k_exists_l768_76874


namespace gcd_765432_654321_l768_76815

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 111111 := by sorry

end gcd_765432_654321_l768_76815


namespace imaginary_power_difference_l768_76813

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_difference : i^23 - i^210 = -i + 1 := by sorry

end imaginary_power_difference_l768_76813


namespace polynomial_division_remainder_l768_76809

theorem polynomial_division_remainder (x : ℂ) : 
  (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end polynomial_division_remainder_l768_76809


namespace tangent_slope_angle_sin_plus_cos_l768_76853

theorem tangent_slope_angle_sin_plus_cos (x : Real) : 
  let f : Real → Real := λ x => Real.sin x + Real.cos x
  let f' : Real → Real := λ x => -Real.sin x + Real.cos x
  let slope : Real := f' (π/4)
  let slope_angle : Real := Real.arctan slope
  x = π/4 → slope_angle = 0 := by
sorry

end tangent_slope_angle_sin_plus_cos_l768_76853


namespace vertical_angles_equal_l768_76884

-- Define a line as a type
def Line : Type := ℝ → ℝ → Prop

-- Define a point as a pair of real numbers
def Point : Type := ℝ × ℝ

-- Define the notion of two lines intersecting at a point
def intersect (l1 l2 : Line) (p : Point) : Prop :=
  l1 p.1 p.2 ∧ l2 p.1 p.2

-- Define vertical angles
def vertical_angles (l1 l2 : Line) (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (i : Point), intersect l1 l2 i ∧
  (p1 ≠ i ∧ p2 ≠ i ∧ p3 ≠ i ∧ p4 ≠ i) ∧
  (l1 p1.1 p1.2 ∧ l1 p3.1 p3.2) ∧
  (l2 p2.1 p2.2 ∧ l2 p4.1 p4.2)

-- Define angle measure
def angle_measure (p1 p2 p3 : Point) : ℝ := sorry

-- Theorem: Vertical angles are equal
theorem vertical_angles_equal (l1 l2 : Line) (p1 p2 p3 p4 : Point) :
  vertical_angles l1 l2 p1 p2 p3 p4 →
  angle_measure p1 i p2 = angle_measure p3 i p4 :=
sorry

end vertical_angles_equal_l768_76884


namespace oil_price_reduction_l768_76806

/-- Given a price reduction that allows buying 3 kg more for Rs. 700,
    and a reduced price of Rs. 70 per kg, prove that the percentage
    reduction in the price of oil is 30%. -/
theorem oil_price_reduction (original_price : ℝ) :
  (∃ (reduced_price : ℝ),
    reduced_price = 70 ∧
    700 / original_price + 3 = 700 / reduced_price) →
  (original_price - 70) / original_price * 100 = 30 :=
by sorry

end oil_price_reduction_l768_76806


namespace set_operations_l768_76878

theorem set_operations (M N P : Set ℕ) 
  (hM : M = {1})
  (hN : N = {1, 2})
  (hP : P = {1, 2, 3}) :
  (M ∪ N) ∩ P = {1, 2} := by
  sorry

end set_operations_l768_76878


namespace parabola_points_relation_l768_76823

/-- Prove that for points A(1, y₁) and B(2, y₂) lying on the parabola y = a(x+1)² + 2 where a < 0, 
    the relationship 2 > y₁ > y₂ holds. -/
theorem parabola_points_relation (a y₁ y₂ : ℝ) : 
  a < 0 → 
  y₁ = a * (1 + 1)^2 + 2 → 
  y₂ = a * (2 + 1)^2 + 2 → 
  2 > y₁ ∧ y₁ > y₂ := by
  sorry

end parabola_points_relation_l768_76823


namespace fruit_problem_equations_l768_76801

/-- Represents the ancient Chinese fruit problem --/
structure FruitProblem where
  totalFruits : ℕ
  totalCost : ℕ
  bitterFruitCount : ℕ
  bitterFruitCost : ℕ
  sweetFruitCount : ℕ
  sweetFruitCost : ℕ

/-- The system of equations for the fruit problem --/
def fruitEquations (p : FruitProblem) (x y : ℚ) : Prop :=
  x + y = p.totalFruits ∧
  (4 / 7 : ℚ) * x + (11 / 9 : ℚ) * y = p.totalCost

/-- Theorem stating that the given system of equations correctly represents the fruit problem --/
theorem fruit_problem_equations (p : FruitProblem) 
  (h1 : p.totalFruits = 1000)
  (h2 : p.totalCost = 999)
  (h3 : p.bitterFruitCount = 7)
  (h4 : p.bitterFruitCost = 4)
  (h5 : p.sweetFruitCount = 9)
  (h6 : p.sweetFruitCost = 11) :
  ∃ x y : ℚ, fruitEquations p x y :=
sorry

end fruit_problem_equations_l768_76801


namespace quadratic_roots_inequality_l768_76886

theorem quadratic_roots_inequality (t : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 - t*x₁ + t = 0) → 
  (x₂^2 - t*x₂ + t = 0) → 
  (x₁^2 + x₂^2 ≥ 2*(x₁ + x₂)) := by
  sorry

end quadratic_roots_inequality_l768_76886


namespace piggy_bank_ratio_l768_76880

theorem piggy_bank_ratio (T A S X Y : ℝ) (hT : T = 450) (hA : A = 30) 
  (hS : S > A) (hX : X > S) (hY : Y > X) (hTotal : A + S + X + Y = T) :
  ∃ (r : ℝ), r = (T - A - X - Y) / A ∧ r = S / A :=
sorry

end piggy_bank_ratio_l768_76880


namespace symmetric_circle_equation_l768_76873

/-- Given a circle with center (a, b) and radius r, returns the equation of the circle symmetric to it with respect to the line y = x -/
def symmetricCircle (a b r : ℝ) : (ℝ × ℝ → Prop) :=
  fun p => (p.1 - b)^2 + (p.2 - a)^2 = r^2

/-- The original circle (x-1)^2 + (y-2)^2 = 1 -/
def originalCircle : (ℝ × ℝ → Prop) :=
  fun p => (p.1 - 1)^2 + (p.2 - 2)^2 = 1

theorem symmetric_circle_equation :
  symmetricCircle 1 2 1 = fun p => (p.1 - 2)^2 + (p.2 - 1)^2 = 1 := by
  sorry

end symmetric_circle_equation_l768_76873


namespace line_parallel_to_intersection_l768_76819

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the intersection operation between two planes
variable (intersect_planes : Plane → Plane → Line)

-- Define the parallel relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Theorem statement
theorem line_parallel_to_intersection
  (m n : Line) (α β : Plane)
  (h1 : parallel_line_plane m α)
  (h2 : subset_line_plane m β)
  (h3 : intersect_planes α β = n) :
  parallel_lines m n :=
sorry

end line_parallel_to_intersection_l768_76819


namespace circle_line_intersection_range_l768_76891

theorem circle_line_intersection_range (r : ℝ) (h_r_pos : r > 0) :
  (∀ m : ℝ, ∃ A B : ℝ × ℝ,
    (A.1^2 + A.2^2 = r^2) ∧
    (B.1^2 + B.2^2 = r^2) ∧
    (m * A.1 - A.2 + 1 = 0) ∧
    (m * B.1 - B.2 + 1 = 0) ∧
    A ≠ B) ∧
  (∃ m : ℝ, ∀ A B : ℝ × ℝ,
    (A.1^2 + A.2^2 = r^2) ∧
    (B.1^2 + B.2^2 = r^2) ∧
    (m * A.1 - A.2 + 1 = 0) ∧
    (m * B.1 - B.2 + 1 = 0) →
    (A.1 + B.1)^2 + (A.2 + B.2)^2 ≥ (B.1 - A.1)^2 + (B.2 - A.2)^2) →
  1 < r ∧ r ≤ Real.sqrt 2 := by
sorry

end circle_line_intersection_range_l768_76891


namespace binomial_divisibility_l768_76840

theorem binomial_divisibility (k : ℤ) : k ≠ 1 ↔ ∃ f : ℕ → ℕ, StrictMono f ∧ ∀ i : ℕ, (f i + k : ℤ) ∣ Nat.choose (2 * f i) (f i) → False :=
sorry

end binomial_divisibility_l768_76840
