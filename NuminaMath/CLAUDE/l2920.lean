import Mathlib

namespace chorus_selection_probability_equal_l2920_292005

/-- Represents a two-stage sampling process in a high school chorus selection -/
structure ChorusSelection where
  total_students : ℕ
  eliminated_students : ℕ
  selected_students : ℕ

/-- The probability of a student being selected for the chorus -/
def selection_probability (cs : ChorusSelection) : ℚ :=
  cs.selected_students / cs.total_students

/-- Theorem stating that the selection probability is equal for all students -/
theorem chorus_selection_probability_equal
  (cs : ChorusSelection)
  (h1 : cs.total_students = 1815)
  (h2 : cs.eliminated_students = 15)
  (h3 : cs.selected_students = 30)
  (h4 : cs.total_students = cs.eliminated_students + (cs.total_students - cs.eliminated_students))
  (h5 : cs.selected_students ≤ cs.total_students - cs.eliminated_students) :
  selection_probability cs = 30 / 1815 := by
  sorry

#check chorus_selection_probability_equal

end chorus_selection_probability_equal_l2920_292005


namespace grants_age_fraction_l2920_292015

theorem grants_age_fraction (grant_current_age hospital_current_age : ℕ) 
  (h1 : grant_current_age = 25) (h2 : hospital_current_age = 40) :
  (grant_current_age + 5 : ℚ) / (hospital_current_age + 5 : ℚ) = 2 / 3 := by
  sorry

end grants_age_fraction_l2920_292015


namespace hurricane_damage_conversion_l2920_292083

theorem hurricane_damage_conversion (damage_aud : ℝ) (exchange_rate : ℝ) : 
  damage_aud = 45000000 → 
  exchange_rate = 2 → 
  damage_aud / exchange_rate = 22500000 :=
by sorry

end hurricane_damage_conversion_l2920_292083


namespace normal_distribution_std_dev_l2920_292011

/-- Represents a normal distribution -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ

/-- The value that is exactly 2 standard deviations less than the mean -/
def two_std_dev_below (d : NormalDistribution) : ℝ :=
  d.mean - 2 * d.std_dev

/-- Theorem: If the mean is 14.0 and the value 2 standard deviations below the mean is 11,
    then the standard deviation is 1.5 -/
theorem normal_distribution_std_dev
  (d : NormalDistribution)
  (h_mean : d.mean = 14.0)
  (h_two_below : two_std_dev_below d = 11) :
  d.std_dev = 1.5 := by
  sorry

end normal_distribution_std_dev_l2920_292011


namespace g_zero_at_three_l2920_292091

def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + x^3 - 4 * x^2 + 5 * x + s

theorem g_zero_at_three (s : ℝ) : g 3 s = 0 ↔ s = -573 := by sorry

end g_zero_at_three_l2920_292091


namespace quadratic_sum_l2920_292010

/-- Given a quadratic function f(x) = -3x^2 + 24x + 144, prove that when written
    in the form a(x+b)^2 + c, the sum of a, b, and c is 185. -/
theorem quadratic_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = -3 * x^2 + 24 * x + 144) →
  (∀ x, f x = a * (x + b)^2 + c) →
  a + b + c = 185 := by
  sorry

end quadratic_sum_l2920_292010


namespace selection_schemes_count_l2920_292055

/-- The number of people in the group -/
def totalPeople : ℕ := 5

/-- The number of cities to be visited -/
def totalCities : ℕ := 4

/-- The number of people who can visit Paris (excluding A) -/
def parisVisitors : ℕ := totalPeople - 1

/-- Calculate the number of selection schemes -/
def selectionSchemes : ℕ :=
  parisVisitors * (totalPeople - 1) * (totalPeople - 2) * (totalPeople - 3)

/-- Theorem stating the number of selection schemes is 96 -/
theorem selection_schemes_count :
  selectionSchemes = 96 := by sorry

end selection_schemes_count_l2920_292055


namespace equation_equality_l2920_292061

theorem equation_equality : 2 * 18 * 14 = 6 * 12 * 7 := by
  sorry

end equation_equality_l2920_292061


namespace seconds_in_minutes_l2920_292017

theorem seconds_in_minutes (minutes : ℚ) (seconds_per_minute : ℕ) :
  minutes = 11 / 3 →
  seconds_per_minute = 60 →
  (minutes * seconds_per_minute : ℚ) = 220 := by
sorry

end seconds_in_minutes_l2920_292017


namespace rectangle_probability_l2920_292073

/-- A rectangle in the 2D plane --/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The probability that a randomly selected point from a rectangle is closer to one point than another --/
def closerProbability (r : Rectangle) (p1 : ℝ × ℝ) (p2 : ℝ × ℝ) : ℝ :=
  sorry

/-- The theorem to be proved --/
theorem rectangle_probability : 
  let r := Rectangle.mk 0 0 3 2
  closerProbability r (0, 0) (4, 0) = 1/2 := by
  sorry

end rectangle_probability_l2920_292073


namespace smallest_dual_palindrome_proof_l2920_292026

/-- Checks if a natural number is a palindrome when represented in the given base. -/
def is_palindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a natural number to its representation in the given base. -/
def to_base (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- The smallest 5-digit palindrome in base 2 that is also a 3-digit palindrome in base 5 -/
def smallest_dual_palindrome : ℕ := 27

theorem smallest_dual_palindrome_proof :
  (is_palindrome smallest_dual_palindrome 2) ∧
  (is_palindrome smallest_dual_palindrome 5) ∧
  (to_base smallest_dual_palindrome 2).length = 5 ∧
  (to_base smallest_dual_palindrome 5).length = 3 ∧
  (∀ m : ℕ, m < smallest_dual_palindrome →
    ¬(is_palindrome m 2 ∧ is_palindrome m 5 ∧
      (to_base m 2).length = 5 ∧ (to_base m 5).length = 3)) :=
by
  sorry

#eval smallest_dual_palindrome

end smallest_dual_palindrome_proof_l2920_292026


namespace geometric_sequence_sum_l2920_292042

theorem geometric_sequence_sum (a r : ℚ) (n : ℕ) (h1 : a = 1/2) (h2 : r = 1/3) (h3 : n = 8) :
  (a * (1 - r^n)) / (1 - r) = 9840/6561 := by
sorry

end geometric_sequence_sum_l2920_292042


namespace cosine_sum_constant_l2920_292060

theorem cosine_sum_constant (A B C : ℝ) 
  (h1 : Real.cos A + Real.cos B + Real.cos C = 0)
  (h2 : Real.sin A + Real.sin B + Real.sin C = 0) : 
  Real.cos A ^ 2 + Real.cos B ^ 2 + Real.cos C ^ 2 = 3/2 := by
  sorry

end cosine_sum_constant_l2920_292060


namespace cereal_spending_l2920_292041

/-- The amount spent by Pop on cereal -/
def pop_spend : ℝ := 15

/-- The amount spent by Crackle on cereal -/
def crackle_spend : ℝ := 3 * pop_spend

/-- The amount spent by Snap on cereal -/
def snap_spend : ℝ := 2 * crackle_spend

/-- The total amount spent by Snap, Crackle, and Pop on cereal -/
def total_spend : ℝ := snap_spend + crackle_spend + pop_spend

theorem cereal_spending :
  total_spend = 150 := by sorry

end cereal_spending_l2920_292041


namespace william_land_percentage_l2920_292086

-- Define the tax amounts
def total_village_tax : ℝ := 3840
def william_tax : ℝ := 480

-- Define the theorem
theorem william_land_percentage :
  william_tax / total_village_tax * 100 = 12.5 := by
  sorry

end william_land_percentage_l2920_292086


namespace largest_prime_divisor_to_test_l2920_292078

theorem largest_prime_divisor_to_test (n : ℕ) : 
  1000 ≤ n ∧ n ≤ 1050 → 
  (∀ p : ℕ, Prime p ∧ p ≤ 31 → ¬(p ∣ n)) → 
  Prime n ∨ n = 1 := by
sorry

end largest_prime_divisor_to_test_l2920_292078


namespace cos_five_pi_thirds_equals_one_half_l2920_292012

theorem cos_five_pi_thirds_equals_one_half : 
  Real.cos (5 * π / 3) = 1 / 2 := by sorry

end cos_five_pi_thirds_equals_one_half_l2920_292012


namespace tangent_line_at_x_1_l2920_292004

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3 * x^2

-- Theorem statement
theorem tangent_line_at_x_1 :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 3*x - 3 :=
sorry

end tangent_line_at_x_1_l2920_292004


namespace cos_B_value_angle_A_value_projection_BC_BA_l2920_292081

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Sides

-- Define the conditions
axiom triangle_condition : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi
axiom side_angle_correspondence : a = 2 * Real.sin (A / 2) ∧ 
                                  b = 2 * Real.sin (B / 2) ∧ 
                                  c = 2 * Real.sin (C / 2)
axiom line_condition : 2 * a * Real.cos B - b * Real.cos C = c * Real.cos B

-- Define the specific values for a and b
axiom a_value : a = 2 * Real.sqrt 3 / 3
axiom b_value : b = 2

-- Theorem statements
theorem cos_B_value : Real.cos B = 1 / 2 := by sorry

theorem angle_A_value : A = Real.arccos (Real.sqrt 3 / 3) := by sorry

theorem projection_BC_BA : a * Real.cos B = Real.sqrt 3 / 3 := by sorry

end cos_B_value_angle_A_value_projection_BC_BA_l2920_292081


namespace polynomial_square_condition_l2920_292046

theorem polynomial_square_condition (P : Polynomial ℤ) : 
  (∃ R : Polynomial ℤ, (X^2 + 6*X + 10) * P^2 - 1 = R^2) → P = 0 :=
by
  sorry

end polynomial_square_condition_l2920_292046


namespace square_side_length_l2920_292036

theorem square_side_length (rectangle_width rectangle_length : ℝ) 
  (h1 : rectangle_width = 6)
  (h2 : rectangle_length = 24)
  (h3 : rectangle_width > 0)
  (h4 : rectangle_length > 0) :
  ∃ (square_side : ℝ), 
    square_side ^ 2 = rectangle_width * rectangle_length ∧ 
    square_side = 12 := by
  sorry

end square_side_length_l2920_292036


namespace veg_eaters_count_l2920_292089

/-- Represents the number of people in a family with different eating habits -/
structure FamilyEatingHabits where
  only_veg : ℕ
  only_non_veg : ℕ
  both_veg_and_non_veg : ℕ

/-- Theorem stating that the number of people who eat veg in the family is 20 -/
theorem veg_eaters_count (family : FamilyEatingHabits)
  (h1 : family.only_veg = 11)
  (h2 : family.only_non_veg = 6)
  (h3 : family.both_veg_and_non_veg = 9) :
  family.only_veg + family.both_veg_and_non_veg = 20 := by
  sorry

end veg_eaters_count_l2920_292089


namespace min_value_fraction_l2920_292048

theorem min_value_fraction (x y z w : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) (pos_w : 0 < w)
  (sum_eq_one : x + y + z + w = 1) :
  ∀ a b c d : ℝ, 0 < a → 0 < b → 0 < c → 0 < d → a + b + c + d = 1 →
  (x + y + z) / (x * y * z * w) ≤ (a + b + c) / (a * b * c * d) ∧
  (x + y + z) / (x * y * z * w) = 144 := by
sorry

end min_value_fraction_l2920_292048


namespace min_radius_for_area_l2920_292065

/-- The minimum radius of a circle with an area of at least 314 square feet is 10 feet. -/
theorem min_radius_for_area (π : ℝ) (h : π > 0) : 
  (∀ r : ℝ, π * r^2 ≥ 314 → r ≥ 10) ∧ (∃ r : ℝ, π * r^2 = 314 ∧ r = 10) := by
  sorry

end min_radius_for_area_l2920_292065


namespace function_inequality_l2920_292071

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def is_monotone_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

theorem function_inequality (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : is_periodic f 2)
  (h_monotone : is_monotone_decreasing f (-1) 0)
  (a b c : ℝ)
  (h_a : a = f (-2.8))
  (h_b : b = f (-1.6))
  (h_c : c = f 0.5) :
  a > c ∧ c > b := by
  sorry

end function_inequality_l2920_292071


namespace frog_probability_l2920_292062

-- Define the number of lily pads
def num_pads : ℕ := 9

-- Define the set of predator positions
def predator_positions : Set ℕ := {2, 5, 6}

-- Define the target position
def target_position : ℕ := 7

-- Define the probability of moving 1 or 2 positions
def move_probability : ℚ := 1/2

-- Define the function to calculate the probability of reaching the target
def reach_probability (start : ℕ) (target : ℕ) (predators : Set ℕ) (p : ℚ) : ℚ :=
  sorry

-- Theorem statement
theorem frog_probability :
  reach_probability 0 target_position predator_positions move_probability = 1/16 :=
sorry

end frog_probability_l2920_292062


namespace original_painting_width_l2920_292064

/-- Given a painting and its enlarged print, calculate the width of the original painting. -/
theorem original_painting_width
  (original_height : ℝ)
  (print_height : ℝ)
  (print_width : ℝ)
  (h1 : original_height = 10)
  (h2 : print_height = 25)
  (h3 : print_width = 37.5) :
  print_width / (print_height / original_height) = 15 :=
by sorry

#check original_painting_width

end original_painting_width_l2920_292064


namespace limit_polynomial_at_2_l2920_292018

theorem limit_polynomial_at_2 :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |3*x^2 - 2*x + 7 - 15| < ε := by
  sorry

end limit_polynomial_at_2_l2920_292018


namespace bells_lcm_l2920_292075

/-- The time interval (in minutes) between consecutive rings of the library bell -/
def library_interval : ℕ := 18

/-- The time interval (in minutes) between consecutive rings of the community center bell -/
def community_interval : ℕ := 24

/-- The time interval (in minutes) between consecutive rings of the restaurant bell -/
def restaurant_interval : ℕ := 30

/-- The theorem states that the least common multiple of the three bell intervals is 360 minutes -/
theorem bells_lcm :
  lcm (lcm library_interval community_interval) restaurant_interval = 360 :=
by sorry

end bells_lcm_l2920_292075


namespace johns_dog_walking_earnings_l2920_292097

/-- Proves that John earns $10 per day for walking the dog -/
theorem johns_dog_walking_earnings :
  ∀ (days_in_april : ℕ) (sundays : ℕ) (total_spent : ℕ) (money_left : ℕ),
    days_in_april = 30 →
    sundays = 4 →
    total_spent = 100 →
    money_left = 160 →
    (days_in_april - sundays) * 10 = total_spent + money_left :=
by
  sorry

end johns_dog_walking_earnings_l2920_292097


namespace calculate_expression_l2920_292067

theorem calculate_expression : 150 * (150 - 5) + (150 * 150 + 5) = 44255 := by
  sorry

end calculate_expression_l2920_292067


namespace equation_solution_l2920_292080

theorem equation_solution (x : ℝ) (h : x + 2 ≠ 0) :
  (x / (x + 2) + 1 = 1 / (x + 2)) ↔ (x = -1 / 2) :=
by sorry

end equation_solution_l2920_292080


namespace hexagon_side_length_l2920_292052

/-- The side length of a regular hexagon given the distance between opposite sides -/
theorem hexagon_side_length (opposite_distance : ℝ) : 
  opposite_distance = 18 → 
  ∃ (side_length : ℝ), side_length = 12 * Real.sqrt 3 ∧ 
    opposite_distance = (Real.sqrt 3 / 2) * side_length :=
by sorry

end hexagon_side_length_l2920_292052


namespace max_distance_is_1375_l2920_292098

/-- Represents the boat trip scenario -/
structure BoatTrip where
  totalTime : Real
  rowingTime : Real
  restTime : Real
  boatSpeed : Real
  currentSpeed : Real

/-- Calculates the maximum distance the boat can travel from the starting point -/
def maxDistance (trip : BoatTrip) : Real :=
  sorry

/-- Theorem stating that the maximum distance is 1.375 km for the given conditions -/
theorem max_distance_is_1375 :
  let trip : BoatTrip := {
    totalTime := 120,
    rowingTime := 30,
    restTime := 10,
    boatSpeed := 3,
    currentSpeed := 1.5
  }
  maxDistance trip = 1.375 := by
  sorry

end max_distance_is_1375_l2920_292098


namespace factorial_a_ratio_l2920_292044

/-- Definition of n_a! for positive n and a -/
def factorial_a (n a : ℕ) : ℕ :=
  (List.range ((n / a) + 1)).foldl (fun acc k => acc * (n - k * a)) n

/-- Theorem stating that 96_4! / 48_3! = 2^8 -/
theorem factorial_a_ratio : (factorial_a 96 4) / (factorial_a 48 3) = 2^8 := by
  sorry

end factorial_a_ratio_l2920_292044


namespace s_range_for_composites_l2920_292049

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def s (n : ℕ) : ℕ := sorry

theorem s_range_for_composites :
  (∀ n : ℕ, is_composite n → s n ≥ 12) ∧
  (∀ m : ℕ, m ≥ 12 → ∃ n : ℕ, is_composite n ∧ s n = m) :=
sorry

end s_range_for_composites_l2920_292049


namespace vector_sum_zero_l2920_292053

variable {V : Type*} [AddCommGroup V]

def closed_polygon (a b c f : V) : Prop :=
  a + (c - b) + (f - c) + (b - f) = 0

theorem vector_sum_zero (a b c f : V) (h : closed_polygon a b c f) :
  (b - a) + (f - c) + (c - b) + (a - f) = 0 := by sorry

end vector_sum_zero_l2920_292053


namespace bus_stop_problem_l2920_292027

/-- The number of children who got on the bus at a stop -/
def children_got_on (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

theorem bus_stop_problem :
  let initial_children : ℕ := 64
  let final_children : ℕ := 78
  children_got_on initial_children final_children = 14 := by
  sorry

end bus_stop_problem_l2920_292027


namespace two_numbers_satisfy_conditions_l2920_292030

/-- A function that checks if a number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- A function that checks if a two-digit number is a square --/
def isTwoDigitSquare (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ isPerfectSquare n

/-- A function that checks if a number is a single-digit square (1, 4, or 9) --/
def isSingleDigitSquare (n : ℕ) : Prop :=
  n = 1 ∨ n = 4 ∨ n = 9

/-- A function that returns the first two digits of a five-digit number --/
def firstTwoDigits (n : ℕ) : ℕ :=
  n / 1000

/-- A function that returns the sum of the third and fourth digits of a five-digit number --/
def sumMiddleTwoDigits (n : ℕ) : ℕ :=
  (n / 100 % 10) + (n / 10 % 10)

/-- A function that checks if a number satisfies all the given conditions --/
def satisfiesAllConditions (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999 ∧  -- five-digit number
  (∀ d, d ∈ [n / 10000, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] → d ≠ 0) ∧  -- no digit is zero
  isPerfectSquare n ∧  -- perfect square
  isTwoDigitSquare (firstTwoDigits n) ∧  -- first two digits form a square
  isSingleDigitSquare (sumMiddleTwoDigits n) ∧  -- sum of middle two digits is a single-digit square
  n % 7 = 0  -- divisible by 7

/-- The main theorem stating that exactly two numbers satisfy all conditions --/
theorem two_numbers_satisfy_conditions : 
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfiesAllConditions n) ∧ s.card = 2 :=
sorry

end two_numbers_satisfy_conditions_l2920_292030


namespace no_x_satisfies_conditions_l2920_292014

theorem no_x_satisfies_conditions : ¬ ∃ x : ℝ, 
  400 ≤ x ∧ x ≤ 600 ∧ 
  Int.floor (Real.sqrt x) = 23 ∧ 
  Int.floor (Real.sqrt (100 * x)) = 480 := by
sorry

end no_x_satisfies_conditions_l2920_292014


namespace hiker_first_pack_weight_l2920_292021

/-- Calculates the weight of the first pack for a hiker given specific conditions --/
theorem hiker_first_pack_weight
  (supplies_per_mile : Real)
  (hiking_rate : Real)
  (hours_per_day : Real)
  (days : Real)
  (first_resupply_ratio : Real)
  (second_resupply_ratio : Real)
  (h1 : supplies_per_mile = 0.6)
  (h2 : hiking_rate = 2.5)
  (h3 : hours_per_day = 9)
  (h4 : days = 7)
  (h5 : first_resupply_ratio = 0.3)
  (h6 : second_resupply_ratio = 0.2) :
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := supplies_per_mile * total_distance
  let first_resupply := first_resupply_ratio * total_supplies
  let second_resupply := second_resupply_ratio * total_supplies
  let first_pack_weight := total_supplies - (first_resupply + second_resupply)
  first_pack_weight = 47.25 := by sorry

end hiker_first_pack_weight_l2920_292021


namespace square_sum_reciprocal_l2920_292095

theorem square_sum_reciprocal (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x)^2 = 23 := by
  sorry

end square_sum_reciprocal_l2920_292095


namespace square_of_rational_l2920_292093

theorem square_of_rational (x y : ℚ) (h : x^5 + y^5 = 2*x^2*y^2) :
  ∃ z : ℚ, 1 - x*y = z^2 := by
sorry

end square_of_rational_l2920_292093


namespace root_sum_theorem_l2920_292068

-- Define the polynomial
def P (k x : ℝ) : ℝ := k * (x^2 - x) + x + 7

-- Define the condition for k1 and k2
def K_condition (k : ℝ) : Prop :=
  ∃ a b : ℝ, P k a = 0 ∧ P k b = 0 ∧ a/b + b/a = 3/7

-- State the theorem
theorem root_sum_theorem (k1 k2 : ℝ) :
  K_condition k1 ∧ K_condition k2 →
  k1/k2 + k2/k1 = 322 :=
sorry

end root_sum_theorem_l2920_292068


namespace bear_buns_l2920_292032

theorem bear_buns (x : ℚ) : 
  (x / 8 - 7 / 8 = 0) → x = 7 := by sorry

end bear_buns_l2920_292032


namespace expression_evaluation_l2920_292072

theorem expression_evaluation : (3 * 10^9) / (6 * 10^5) = 5000 := by
  sorry

end expression_evaluation_l2920_292072


namespace solution_set_a_gt_1_solution_set_a_eq_1_solution_set_a_lt_1_a_range_subset_l2920_292059

-- Define the inequality function
def f (a x : ℝ) : ℝ := (a * x - (a - 2)) * (x + 1)

-- Define the solution set P
def P (a : ℝ) : Set ℝ := {x | f a x > 0}

-- Theorem for the solution set when a > 1
theorem solution_set_a_gt_1 (a : ℝ) (h : a > 1) :
  P a = {x | x < -1 ∨ x > (a - 2) / a} := by sorry

-- Theorem for the solution set when a = 1
theorem solution_set_a_eq_1 :
  P 1 = {x | x ≠ -1} := by sorry

-- Theorem for the solution set when 0 < a < 1
theorem solution_set_a_lt_1 (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  P a = {x | x < (a - 2) / a ∨ x > -1} := by sorry

-- Theorem for the range of a when {x | -3 < x < -1} ⊆ P
theorem a_range_subset (a : ℝ) (h : {x : ℝ | -3 < x ∧ x < -1} ⊆ P a) :
  a ∈ Set.Ici 1 := by sorry

end solution_set_a_gt_1_solution_set_a_eq_1_solution_set_a_lt_1_a_range_subset_l2920_292059


namespace hyperbola_rational_parameterization_l2920_292069

theorem hyperbola_rational_parameterization
  (x p q : ℚ) 
  (h : p^2 - x*q^2 = 1) :
  ∃ (a b : ℤ), 
    p = (a^2 + x*b^2) / (a^2 - x*b^2) ∧
    q = 2*a*b / (a^2 - x*b^2) := by
  sorry

end hyperbola_rational_parameterization_l2920_292069


namespace always_positive_l2920_292013

theorem always_positive (x : ℝ) : 3 * x^2 - 6 * x + 3.5 > 0 := by
  sorry

end always_positive_l2920_292013


namespace intersection_M_N_l2920_292063

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l2920_292063


namespace twelve_mile_ride_cost_l2920_292088

/-- Calculates the cost of a taxi ride given the specified conditions -/
def taxiRideCost (baseFare mileRate discountThreshold discountRate miles : ℚ) : ℚ :=
  let totalBeforeDiscount := baseFare + mileRate * miles
  if miles > discountThreshold then
    totalBeforeDiscount * (1 - discountRate)
  else
    totalBeforeDiscount

theorem twelve_mile_ride_cost :
  taxiRideCost 2 (30/100) 10 (10/100) 12 = 504/100 := by
  sorry

#eval taxiRideCost 2 (30/100) 10 (10/100) 12

end twelve_mile_ride_cost_l2920_292088


namespace cyclists_speed_l2920_292031

/-- Cyclist's trip problem -/
theorem cyclists_speed (v : ℝ) : 
  v > 0 → -- The speed is positive
  (9 / v + 12 / 9 : ℝ) = 21 / 10.08 → -- Total time equation
  v = 12 := by
sorry

end cyclists_speed_l2920_292031


namespace simplify_trig_expression_l2920_292024

theorem simplify_trig_expression (α : Real) (h : α ∈ Set.Ioo (π / 2) π) :
  (Real.sqrt (1 - 2 * Real.sin α * Real.cos α)) / (Real.sin α + Real.sqrt (1 - Real.sin α ^ 2)) = 1 := by
  sorry

end simplify_trig_expression_l2920_292024


namespace log_stack_count_15_5_l2920_292066

/-- The number of logs in a stack with a given bottom and top row count -/
def logStackCount (bottom top : ℕ) : ℕ :=
  let n := bottom - top + 1
  n * (bottom + top) / 2

/-- Theorem: A stack of logs with 15 on the bottom row and 5 on the top row has 110 logs -/
theorem log_stack_count_15_5 : logStackCount 15 5 = 110 := by
  sorry

end log_stack_count_15_5_l2920_292066


namespace brick_length_is_20_l2920_292045

/-- The length of a brick in centimeters -/
def brick_length : ℝ := 20

/-- The width of a brick in centimeters -/
def brick_width : ℝ := 10

/-- The height of a brick in centimeters -/
def brick_height : ℝ := 7.5

/-- The length of the wall in meters -/
def wall_length : ℝ := 26

/-- The width of the wall in meters -/
def wall_width : ℝ := 2

/-- The height of the wall in meters -/
def wall_height : ℝ := 0.75

/-- The number of bricks required to build the wall -/
def num_bricks : ℕ := 26000

/-- Theorem stating that the length of the brick is 20 cm given the conditions -/
theorem brick_length_is_20 :
  brick_length = 20 ∧
  brick_width * brick_height * brick_length * num_bricks = 
  wall_length * wall_width * wall_height * 1000000 :=
by sorry

end brick_length_is_20_l2920_292045


namespace smallest_digit_sum_of_sum_l2920_292092

/-- Two different two-digit positive integers -/
def is_valid_pair (x y : ℕ) : Prop :=
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ x ≠ y

/-- All four digits in the two numbers are unique -/
def has_unique_digits (x y : ℕ) : Prop :=
  let digits := [x / 10, x % 10, y / 10, y % 10]
  List.Nodup digits

/-- The sum is a two-digit number -/
def is_two_digit_sum (x y : ℕ) : Prop :=
  10 ≤ x + y ∧ x + y < 100

/-- The sum of digits of a number -/
def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

/-- The main theorem -/
theorem smallest_digit_sum_of_sum :
  ∃ (x y : ℕ), 
    is_valid_pair x y ∧ 
    has_unique_digits x y ∧ 
    is_two_digit_sum x y ∧
    ∀ (a b : ℕ), 
      is_valid_pair a b → 
      has_unique_digits a b → 
      is_two_digit_sum a b → 
      digit_sum (x + y) ≤ digit_sum (a + b) ∧
      digit_sum (x + y) = 10 :=
sorry

end smallest_digit_sum_of_sum_l2920_292092


namespace smallest_n_satisfying_conditions_l2920_292084

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (9 ∣ (n + 7)) ∧ 
  (6 ∣ (n - 9)) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (9 ∣ (m + 7)) ∧ (6 ∣ (m - 9))) → False) ∧
  n = 101 := by
  sorry

end smallest_n_satisfying_conditions_l2920_292084


namespace handshake_count_total_handshakes_l2920_292090

theorem handshake_count : ℕ :=
  let twin_sets : ℕ := 12
  let triplet_sets : ℕ := 8
  let twins : ℕ := twin_sets * 2
  let triplets : ℕ := triplet_sets * 3
  let twin_handshakes : ℕ := (twins * (twins - 2)) / 2
  let triplet_handshakes : ℕ := (triplets * (triplets - 3)) / 2
  let cross_handshakes : ℕ := twins * (2 * triplets / 3)
  twin_handshakes + triplet_handshakes + cross_handshakes

theorem total_handshakes : handshake_count = 900 := by
  sorry

end handshake_count_total_handshakes_l2920_292090


namespace delegate_seating_probability_l2920_292085

/-- Represents the number of delegates -/
def num_delegates : ℕ := 8

/-- Represents the number of countries -/
def num_countries : ℕ := 4

/-- Represents the number of delegates per country -/
def delegates_per_country : ℕ := 2

/-- Represents the number of seats at the round table -/
def num_seats : ℕ := 8

/-- Calculates the total number of possible seating arrangements -/
def total_arrangements : ℕ := num_delegates.factorial / (delegates_per_country.factorial ^ num_countries)

/-- Calculates the number of favorable seating arrangements -/
def favorable_arrangements : ℕ := total_arrangements - 324

/-- The probability that each delegate sits next to at least one delegate from another country -/
def probability : ℚ := favorable_arrangements / total_arrangements

theorem delegate_seating_probability :
  probability = 131 / 140 := by sorry

end delegate_seating_probability_l2920_292085


namespace min_value_of_rounded_sum_l2920_292070

-- Define the rounding functions
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

noncomputable def roundToNearestTenth (x : ℝ) : ℝ :=
  (roundToNearest (x * 10)) / 10

-- Define the main theorem
theorem min_value_of_rounded_sum (a b : ℝ) 
  (h1 : roundToNearestTenth a + roundToNearest b = 98.6)
  (h2 : roundToNearest a + roundToNearestTenth b = 99.3) :
  roundToNearest (10 * (a + b)) ≥ 988 :=
sorry

end min_value_of_rounded_sum_l2920_292070


namespace expression_equals_one_l2920_292007

theorem expression_equals_one :
  (150^2 - 13^2) / (90^2 - 17^2) * ((90 - 17) * (90 + 17)) / ((150 - 13) * (150 + 13)) = 1 := by
  sorry

end expression_equals_one_l2920_292007


namespace min_value_sum_reciprocals_l2920_292037

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) : 
  1/a + 4/b + 9/c + 16/d + 25/e + 36/f ≥ 441/10 ∧
  ∃ a' b' c' d' e' f' : ℝ, 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 0 < d' ∧ 0 < e' ∧ 0 < f' ∧
    a' + b' + c' + d' + e' + f' = 10 ∧
    1/a' + 4/b' + 9/c' + 16/d' + 25/e' + 36/f' = 441/10 :=
by sorry

end min_value_sum_reciprocals_l2920_292037


namespace reinforcement_arrival_time_l2920_292096

/-- Calculates the number of days that passed before reinforcement arrived --/
def days_before_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) 
  (reinforcement : ℕ) (remaining_provisions : ℕ) : ℕ :=
  (initial_garrison * initial_provisions - (initial_garrison + reinforcement) * remaining_provisions) / initial_garrison

/-- Theorem stating that 21 days passed before reinforcement arrived --/
theorem reinforcement_arrival_time : 
  days_before_reinforcement 2000 54 1300 20 = 21 := by
  sorry


end reinforcement_arrival_time_l2920_292096


namespace sequence_value_l2920_292022

theorem sequence_value (a : ℕ → ℝ) (h : ∀ n, (3 - a (n + 1)) * (6 + a n) = 18) (h0 : a 0 ≠ 3) :
  ∀ n, a n = 2^(n + 2) - n - 3 :=
sorry

end sequence_value_l2920_292022


namespace horner_method_proof_polynomial_value_at_2_l2920_292050

def horner_polynomial (x : ℝ) : ℝ :=
  ((((2 * x + 4) * x - 2) * x - 3) * x + 1) * x

theorem horner_method_proof :
  horner_polynomial 2 = 2 * 2^5 - 3 * 2^2 + 4 * 2^4 - 2 * 2^3 + 2 :=
by sorry

theorem polynomial_value_at_2 :
  horner_polynomial 2 = 102 :=
by sorry

end horner_method_proof_polynomial_value_at_2_l2920_292050


namespace polynomial_remainder_l2920_292058

def f (a b x : ℚ) : ℚ := a * x^3 - 7 * x^2 + b * x - 6

theorem polynomial_remainder (a b : ℚ) :
  (f a b 2 = -8) ∧ (f a b (-1) = -18) → a = 2/3 ∧ b = 13/3 := by
  sorry

end polynomial_remainder_l2920_292058


namespace function_extrema_condition_l2920_292016

def f (a x : ℝ) : ℝ := x^3 + (a+1)*x^2 + (a+1)*x + a

theorem function_extrema_condition (a : ℝ) :
  (∃ (max min : ℝ), ∀ x, f a x ≤ max ∧ f a x ≥ min) ↔ (a < -1 ∨ a > 2) :=
by sorry

end function_extrema_condition_l2920_292016


namespace plane_equation_from_point_and_normal_specific_plane_equation_l2920_292008

/-- Given a point M and a normal vector N, this theorem states that
    the equation Ax + By + Cz + D = 0 represents a plane passing through M
    and perpendicular to N, where (A, B, C) are the components of N. -/
theorem plane_equation_from_point_and_normal (M : ℝ × ℝ × ℝ) (N : ℝ × ℝ × ℝ) :
  let (x₀, y₀, z₀) := M
  let (A, B, C) := N
  let D := -(A * x₀ + B * y₀ + C * z₀)
  ∀ (x y z : ℝ), A * x + B * y + C * z + D = 0 ↔
    ((x - x₀) * A + (y - y₀) * B + (z - z₀) * C = 0 ∧
     ∃ (t : ℝ), x - x₀ = t * A ∧ y - y₀ = t * B ∧ z - z₀ = t * C) :=
by sorry

/-- The equation 4x + 3y + 2z - 27 = 0 represents a plane that passes through
    the point (2, 3, 5) and is perpendicular to the vector (4, 3, 2). -/
theorem specific_plane_equation :
  let M : ℝ × ℝ × ℝ := (2, 3, 5)
  let N : ℝ × ℝ × ℝ := (4, 3, 2)
  ∀ (x y z : ℝ), 4 * x + 3 * y + 2 * z - 27 = 0 ↔
    ((x - 2) * 4 + (y - 3) * 3 + (z - 5) * 2 = 0 ∧
     ∃ (t : ℝ), x - 2 = t * 4 ∧ y - 3 = t * 3 ∧ z - 5 = t * 2) :=
by sorry

end plane_equation_from_point_and_normal_specific_plane_equation_l2920_292008


namespace decagon_diagonals_from_vertex_l2920_292002

/-- The number of sides in a regular decagon -/
def decagon_sides : ℕ := 10

/-- The number of diagonals from one vertex of a polygon with n sides -/
def diagonals_from_vertex (n : ℕ) : ℕ := n - 3

/-- Theorem: The number of diagonals from one vertex of a regular decagon is 7 -/
theorem decagon_diagonals_from_vertex : 
  diagonals_from_vertex decagon_sides = 7 := by sorry

end decagon_diagonals_from_vertex_l2920_292002


namespace shelter_adoption_rate_l2920_292033

def puppies_adopted_per_day (initial_puppies : ℕ) (additional_puppies : ℕ) (total_days : ℕ) : ℕ :=
  (initial_puppies + additional_puppies) / total_days

theorem shelter_adoption_rate :
  puppies_adopted_per_day 9 12 7 = 3 := by
  sorry

end shelter_adoption_rate_l2920_292033


namespace equation_solution_l2920_292040

theorem equation_solution : ∃ x : ℝ, (45 / 75 = Real.sqrt (x / 75) + 1 / 5) ∧ x = 12 := by
  sorry

end equation_solution_l2920_292040


namespace cube_has_six_faces_l2920_292079

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the specifics of a cube for this problem

/-- The number of faces of a cube -/
def num_faces (c : Cube) : ℕ := 6

/-- Theorem: A cube has 6 faces -/
theorem cube_has_six_faces (c : Cube) : num_faces c = 6 := by
  sorry

end cube_has_six_faces_l2920_292079


namespace solution_of_equation_l2920_292001

theorem solution_of_equation (x : ℝ) : (5 / (x + 1) - 4 / x = 0) ↔ (x = 4) :=
by sorry

end solution_of_equation_l2920_292001


namespace arbor_day_planting_l2920_292077

theorem arbor_day_planting (class_average : ℝ) (girls_trees : ℝ) (boys_trees : ℝ) :
  class_average = 6 →
  girls_trees = 15 →
  (1 / boys_trees + 1 / girls_trees = 1 / class_average) →
  boys_trees = 10 := by
  sorry

end arbor_day_planting_l2920_292077


namespace range_when_p_true_range_when_p_false_and_q_true_l2920_292054

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - a*x + a > 0

def q (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / (a^2 + 12) - y^2 / (4 - a^2) = 1

-- Theorem for the first part
theorem range_when_p_true :
  {a : ℝ | p a} = {a : ℝ | 0 < a ∧ a < 4} :=
sorry

-- Theorem for the second part
theorem range_when_p_false_and_q_true :
  {a : ℝ | ¬(p a) ∧ q a} = {a : ℝ | -2 < a ∧ a ≤ 0} :=
sorry

end range_when_p_true_range_when_p_false_and_q_true_l2920_292054


namespace october_price_reduction_november_profit_impossible_l2920_292000

def initial_profit_per_box : ℝ := 50
def initial_monthly_sales : ℝ := 500
def sales_increase_per_dollar : ℝ := 20

def profit_function (x : ℝ) : ℝ :=
  (initial_profit_per_box - x) * (initial_monthly_sales + sales_increase_per_dollar * x)

theorem october_price_reduction :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  profit_function x₁ = 28000 ∧
  profit_function x₂ = 28000 ∧
  (x₁ = 10 ∨ x₁ = 15) ∧
  (x₂ = 10 ∨ x₂ = 15) :=
sorry

theorem november_profit_impossible :
  ¬ ∃ x : ℝ, profit_function x = 30000 :=
sorry

end october_price_reduction_november_profit_impossible_l2920_292000


namespace circle_sequence_periodic_l2920_292006

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the sequence of circles
def circleSequence (ABC : Triangle) : Fin 7 → Circle
  | 1 => sorry  -- S₁ inscribed in angle A of ABC
  | 2 => sorry  -- S₂ inscribed in triangle formed by tangent from C to S₁
  | 3 => sorry  -- S₃ inscribed in triangle formed by tangent from A to S₂
  | 4 => sorry  -- S₄
  | 5 => sorry  -- S₅
  | 6 => sorry  -- S₆
  | 7 => sorry  -- S₇

-- Theorem statement
theorem circle_sequence_periodic (ABC : Triangle) :
  circleSequence ABC 7 = circleSequence ABC 1 := by
  sorry

end circle_sequence_periodic_l2920_292006


namespace candy_distribution_l2920_292003

theorem candy_distribution (total_candies : ℕ) (num_friends : ℕ) 
  (h1 : total_candies = 27) (h2 : num_friends = 5) :
  total_candies % num_friends = 
    (total_candies - (total_candies / num_friends) * num_friends) := by
  sorry

end candy_distribution_l2920_292003


namespace sum_of_five_consecutive_odds_l2920_292035

def is_sum_of_five_consecutive_odds (n : ℤ) : Prop :=
  ∃ k : ℤ, n = (2*k-3) + (2*k-1) + (2*k+1) + (2*k+3) + (2*k+5)

theorem sum_of_five_consecutive_odds :
  is_sum_of_five_consecutive_odds 25 ∧
  is_sum_of_five_consecutive_odds 55 ∧
  is_sum_of_five_consecutive_odds 85 ∧
  is_sum_of_five_consecutive_odds 105 ∧
  ¬ is_sum_of_five_consecutive_odds 150 :=
by sorry

end sum_of_five_consecutive_odds_l2920_292035


namespace problem_solution_l2920_292038

theorem problem_solution : 
  ((-0.125 : ℝ)^2023 * 8^2024 = -8) ∧ 
  (((-27 : ℝ)^(1/3 : ℝ) + (5^2 : ℝ)^(1/2 : ℝ) - 2/3 * ((9/4 : ℝ)^(1/2 : ℝ))) = 1) := by
  sorry

end problem_solution_l2920_292038


namespace original_class_size_l2920_292043

theorem original_class_size (N : ℕ) : 
  (N > 0) →
  (40 * N + 8 * 32 = 36 * (N + 8)) →
  N = 8 := by
sorry

end original_class_size_l2920_292043


namespace rectangleB_is_top_leftmost_l2920_292099

-- Define a structure for rectangles
structure Rectangle where
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

-- Define the six rectangles
def rectangleA : Rectangle := ⟨2, 7, 4, 7⟩
def rectangleB : Rectangle := ⟨0, 6, 8, 5⟩
def rectangleC : Rectangle := ⟨6, 3, 1, 1⟩
def rectangleD : Rectangle := ⟨8, 4, 0, 2⟩
def rectangleE : Rectangle := ⟨5, 9, 3, 6⟩
def rectangleF : Rectangle := ⟨7, 5, 9, 0⟩

-- Define a function to check if a rectangle is leftmost
def isLeftmost (r : Rectangle) : Prop :=
  ∀ other : Rectangle, r.w ≤ other.w

-- Define a function to check if a rectangle is topmost among leftmost rectangles
def isTopmostLeftmost (r : Rectangle) : Prop :=
  isLeftmost r ∧ ∀ other : Rectangle, isLeftmost other → r.y ≥ other.y

-- Theorem stating that Rectangle B is the top leftmost rectangle
theorem rectangleB_is_top_leftmost :
  isTopmostLeftmost rectangleB :=
sorry


end rectangleB_is_top_leftmost_l2920_292099


namespace mrs_hilt_marbles_l2920_292020

/-- Calculates the final number of marbles Mrs. Hilt has -/
def final_marbles (initial lost given_away found : ℕ) : ℕ :=
  initial - lost - given_away + found

/-- Theorem stating that Mrs. Hilt's final number of marbles is correct -/
theorem mrs_hilt_marbles :
  final_marbles 38 15 6 8 = 25 := by
  sorry

end mrs_hilt_marbles_l2920_292020


namespace unique_square_sum_pair_l2920_292074

theorem unique_square_sum_pair : 
  ∃! (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    (∃ (m n : ℕ), 100 * a + b = m^2 ∧ 201 * a + b = n^2) ∧
    a = 17 ∧ b = 64 := by
  sorry

end unique_square_sum_pair_l2920_292074


namespace min_value_expression_l2920_292056

theorem min_value_expression (a b : ℝ) (ha : a ≠ 0) (hb : b > 0) (hsum : 2 * a + b = 1) :
  1 / a + 2 / b ≥ 8 ∧ ∃ (a₀ b₀ : ℝ), a₀ ≠ 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 8 :=
by sorry

end min_value_expression_l2920_292056


namespace stratified_sample_female_count_l2920_292087

/-- Calculates the number of female students in a stratified sample -/
def femaleInSample (totalPopulation malePopulation sampleSize : ℕ) : ℕ :=
  let femalePopulation := totalPopulation - malePopulation
  (femalePopulation * sampleSize) / totalPopulation

theorem stratified_sample_female_count :
  femaleInSample 900 500 45 = 20 := by
  sorry

end stratified_sample_female_count_l2920_292087


namespace collinear_vectors_m_value_l2920_292076

/-- Two vectors in ℝ² are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem collinear_vectors_m_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  collinear a b → m = -4 := by
sorry

end collinear_vectors_m_value_l2920_292076


namespace only_proposition3_is_true_l2920_292023

-- Define the propositions
def proposition1 : Prop := ∀ m : ℝ, m > 0 → ∃ x : ℝ, x^2 - x + m = 0

def proposition2 : Prop := ∀ x y : ℝ, x + y > 2 → (x > 1 ∧ y > 1)

def proposition3 : Prop := ∃ x : ℝ, -2 < x ∧ x < 4 ∧ |x - 2| ≥ 3

def proposition4 : Prop := ∀ a b c : ℝ, a ≠ 0 →
  (b^2 - 4*a*c > 0 ↔ ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0)

-- The main theorem
theorem only_proposition3_is_true :
  ¬proposition1 ∧ ¬proposition2 ∧ proposition3 ∧ ¬proposition4 :=
sorry

end only_proposition3_is_true_l2920_292023


namespace truthful_dwarfs_l2920_292034

theorem truthful_dwarfs (n : ℕ) (h_n : n = 10) 
  (raised_vanilla raised_chocolate raised_fruit : ℕ)
  (h_vanilla : raised_vanilla = n)
  (h_chocolate : raised_chocolate = n / 2)
  (h_fruit : raised_fruit = 1) :
  ∃ (truthful liars : ℕ),
    truthful + liars = n ∧
    truthful + 2 * liars = raised_vanilla + raised_chocolate + raised_fruit ∧
    truthful = 4 ∧
    liars = 6 := by
  sorry

end truthful_dwarfs_l2920_292034


namespace bridge_length_calculation_l2920_292028

theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 72 →
  crossing_time = 20 →
  ∃ bridge_length : ℝ,
    bridge_length = 150 ∧
    train_length + bridge_length = (train_speed_kmh * 1000 / 3600) * crossing_time :=
by sorry

end bridge_length_calculation_l2920_292028


namespace color_film_fraction_l2920_292025

theorem color_film_fraction (x y : ℝ) (h : x ≠ 0) : 
  let total_bw := 30 * x
  let total_color := 6 * y
  let selected_bw := (y / x) * (1 / 100) * total_bw
  let selected_color := total_color
  (selected_color / (selected_bw + selected_color)) = 20 / 21 := by
sorry

end color_film_fraction_l2920_292025


namespace smallest_a_value_l2920_292082

theorem smallest_a_value (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0)
  (h3 : ∀ x : ℤ, Real.sin (a * x + b + π / 4) = Real.sin (15 * x + π / 4)) :
  a ≥ 15 ∧ ∃ a₀ : ℝ, a₀ = 15 ∧ a₀ ≥ 0 ∧
    ∀ x : ℤ, Real.sin (a₀ * x + π / 4) = Real.sin (15 * x + π / 4) :=
by sorry

end smallest_a_value_l2920_292082


namespace remainder_problem_l2920_292009

theorem remainder_problem : (7^6 + 8^7 + 9^8) % 7 = 5 := by
  sorry

end remainder_problem_l2920_292009


namespace perpendicular_slope_l2920_292039

/-- Given a line with equation 4x - 5y = 20, the slope of the perpendicular line is -5/4 -/
theorem perpendicular_slope (x y : ℝ) :
  (4 * x - 5 * y = 20) → 
  ∃ (m : ℝ), m = -5/4 ∧ m * (4/5) = -1 :=
by sorry

end perpendicular_slope_l2920_292039


namespace batsman_final_average_l2920_292029

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  lastInningRuns : Nat
  averageIncrease : Nat

/-- Calculates the average runs of a batsman after their last inning -/
def finalAverage (b : Batsman) : Nat :=
  (b.totalRuns + b.lastInningRuns) / (b.innings + 1)

/-- Theorem stating the final average of the batsman -/
theorem batsman_final_average (b : Batsman) 
  (h1 : b.innings = 10)
  (h2 : b.lastInningRuns = 80)
  (h3 : b.averageIncrease = 5)
  (h4 : finalAverage b = (b.totalRuns / b.innings) + b.averageIncrease) :
  finalAverage b = 30 := by
  sorry

#check batsman_final_average

end batsman_final_average_l2920_292029


namespace angle_inequality_l2920_292047

theorem angle_inequality (x : ℝ) :
  x ∈ Set.Ioo 0 (2 * Real.pi) →
  (2^x * (2 * Real.sin x - Real.sqrt 3) ≥ 0) ↔
  x ∈ Set.Icc (Real.pi / 3) (2 * Real.pi / 3) :=
by sorry

end angle_inequality_l2920_292047


namespace base_eight_1263_equals_691_l2920_292057

def base_eight_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_eight_1263_equals_691 :
  base_eight_to_ten [3, 6, 2, 1] = 691 := by
  sorry

end base_eight_1263_equals_691_l2920_292057


namespace hyperbola_eccentricity_l2920_292094

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > b
  k : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: Eccentricity of a special hyperbola -/
theorem hyperbola_eccentricity (h : Hyperbola) 
  (hexagon_condition : ∃ (c : ℝ), c > 0 ∧ 
    (∃ (x y : ℝ), x^2 / h.a^2 - y^2 / h.b^2 = 1 ∧ 
      x^2 + y^2 = c^2 ∧ 
      -- The following condition represents that the intersections form a regular hexagon
      2 * h.a = (Real.sqrt 3 - 1) * c)) :
  eccentricity h = Real.sqrt 3 + 1 := by
  sorry

end hyperbola_eccentricity_l2920_292094


namespace x_value_and_upper_bound_l2920_292051

theorem x_value_and_upper_bound :
  ∀ (x : ℤ) (u : ℚ),
    0 < x ∧ x < 7 ∧
    0 < x ∧ x < 15 ∧
    -1 < x ∧ x < 5 ∧
    0 < x ∧ x < u ∧
    x + 2 < 4 →
    x = 1 ∧ 1 < u ∧ u < 2 :=
by sorry

end x_value_and_upper_bound_l2920_292051


namespace hotel_room_charges_l2920_292019

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * (1 - 0.25))
  (h2 : P = G * (1 - 0.10)) :
  R = G * 1.20 := by
  sorry

end hotel_room_charges_l2920_292019
