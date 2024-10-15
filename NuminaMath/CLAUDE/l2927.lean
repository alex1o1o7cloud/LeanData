import Mathlib

namespace NUMINAMATH_CALUDE_zoey_holidays_per_month_l2927_292773

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The total number of holidays Zoey took in a year -/
def total_holidays : ℕ := 24

/-- Zoey took holidays every month for an entire year -/
axiom holidays_every_month : ∀ (month : ℕ), month ≤ months_in_year → ∃ (holidays : ℕ), holidays > 0

/-- The number of holidays Zoey took each month -/
def holidays_per_month : ℚ := total_holidays / months_in_year

theorem zoey_holidays_per_month : holidays_per_month = 2 := by sorry

end NUMINAMATH_CALUDE_zoey_holidays_per_month_l2927_292773


namespace NUMINAMATH_CALUDE_find_FC_l2927_292730

/-- Given a triangle ABC with point D on AC and point E on AD, prove the length of FC. -/
theorem find_FC (DC CB : ℝ) (h1 : DC = 10) (h2 : CB = 12)
  (AB AD ED : ℝ) (h3 : AB = 1/3 * AD) (h4 : ED = 2/3 * AD) : 
  ∃ (FC : ℝ), FC = 506/33 := by
  sorry

end NUMINAMATH_CALUDE_find_FC_l2927_292730


namespace NUMINAMATH_CALUDE_power_function_k_values_l2927_292734

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x ^ b

theorem power_function_k_values (k : ℝ) :
  is_power_function (λ x => (k^2 - k - 5) * x^3) → k = 3 ∨ k = -2 := by
  sorry

end NUMINAMATH_CALUDE_power_function_k_values_l2927_292734


namespace NUMINAMATH_CALUDE_residue_mod_13_l2927_292795

theorem residue_mod_13 : (156 + 3 * 52 + 4 * 182 + 6 * 26) % 13 = 0 := by
  sorry

end NUMINAMATH_CALUDE_residue_mod_13_l2927_292795


namespace NUMINAMATH_CALUDE_game_result_l2927_292705

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 ∧ n % 4 = 0 then 8
  else if n % 3 = 0 then 3
  else if n % 4 = 0 then 1
  else 0

def allie_rolls : List ℕ := [6, 3, 4, 1]
def betty_rolls : List ℕ := [12, 9, 4, 2]

def total_points (rolls : List ℕ) : ℕ :=
  (rolls.map g).sum

theorem game_result : 
  total_points allie_rolls * total_points betty_rolls = 84 := by
  sorry

end NUMINAMATH_CALUDE_game_result_l2927_292705


namespace NUMINAMATH_CALUDE_oil_in_peanut_butter_l2927_292757

/-- Given a ratio of oil to peanuts and the total weight of peanut butter,
    calculate the amount of oil used. -/
def oil_amount (oil_ratio : ℚ) (peanut_ratio : ℚ) (total_weight : ℚ) : ℚ :=
  (oil_ratio / (oil_ratio + peanut_ratio)) * total_weight

/-- Theorem stating that for the given ratios and total weight,
    the amount of oil used is 4 ounces. -/
theorem oil_in_peanut_butter :
  oil_amount 2 8 20 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oil_in_peanut_butter_l2927_292757


namespace NUMINAMATH_CALUDE_clara_loses_prob_l2927_292727

/-- The probability of Clara's coin landing heads -/
def clara_heads_prob : ℚ := 2/3

/-- The probability of Ethan's coin landing heads -/
def ethan_heads_prob : ℚ := 1/4

/-- The probability of both Clara and Ethan getting tails in one round -/
def both_tails_prob : ℚ := (1 - clara_heads_prob) * (1 - ethan_heads_prob)

/-- The game where Clara and Ethan alternately toss coins until one gets a head and loses -/
def coin_toss_game : Prop :=
  ∃ (p : ℚ), p = clara_heads_prob * (1 / (1 - both_tails_prob))

/-- The theorem stating that the probability of Clara losing is 8/9 -/
theorem clara_loses_prob : 
  coin_toss_game → (∃ (p : ℚ), p = 8/9 ∧ p = clara_heads_prob * (1 / (1 - both_tails_prob))) :=
by sorry

end NUMINAMATH_CALUDE_clara_loses_prob_l2927_292727


namespace NUMINAMATH_CALUDE_multiples_of_12_between_15_and_250_l2927_292761

theorem multiples_of_12_between_15_and_250 : 
  (Finset.filter (λ x => x > 15 ∧ x < 250 ∧ x % 12 = 0) (Finset.range 251)).card = 19 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_12_between_15_and_250_l2927_292761


namespace NUMINAMATH_CALUDE_max_area_right_triangle_l2927_292731

theorem max_area_right_triangle (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a^2 + b^2 = 8^2 → (1/2) * a * b ≤ 16 := by
sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l2927_292731


namespace NUMINAMATH_CALUDE_food_allocation_difference_l2927_292788

/-- Proves that the difference in food allocation between soldiers on the first and second sides is 2 pounds -/
theorem food_allocation_difference (
  soldiers_first : ℕ)
  (soldiers_second : ℕ)
  (food_per_soldier_first : ℝ)
  (total_food : ℝ)
  (h1 : soldiers_first = 4000)
  (h2 : soldiers_second = soldiers_first - 500)
  (h3 : food_per_soldier_first = 10)
  (h4 : total_food = 68000)
  (h5 : total_food = soldiers_first * food_per_soldier_first + 
    soldiers_second * (food_per_soldier_first - (food_per_soldier_first - food_per_soldier_second)))
  : food_per_soldier_first - food_per_soldier_second = 2 := by
  sorry

end NUMINAMATH_CALUDE_food_allocation_difference_l2927_292788


namespace NUMINAMATH_CALUDE_function_monotonicity_l2927_292725

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def periodic_two (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_monotonicity (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_periodic : periodic_two f)
  (h_decreasing : decreasing_on f 1 2) :
  increasing_on f (-2) (-1) ∧ decreasing_on f 3 4 :=
by sorry

end NUMINAMATH_CALUDE_function_monotonicity_l2927_292725


namespace NUMINAMATH_CALUDE_equation_solution_l2927_292770

theorem equation_solution : ∃ x : ℝ, (x - 1) / 2 = 1 - (x + 2) / 3 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2927_292770


namespace NUMINAMATH_CALUDE_distance_to_origin_l2927_292766

theorem distance_to_origin (x y : ℝ) (h1 : y = 15) (h2 : x = 2 + Real.sqrt 105) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (334 + 4 * Real.sqrt 105) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l2927_292766


namespace NUMINAMATH_CALUDE_marbles_lost_l2927_292780

theorem marbles_lost (initial : ℝ) (remaining : ℝ) (lost : ℝ) : 
  initial = 9.5 → remaining = 4.25 → lost = initial - remaining → lost = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_marbles_lost_l2927_292780


namespace NUMINAMATH_CALUDE_real_solutions_condition_l2927_292758

theorem real_solutions_condition (a : ℝ) :
  (∃ x : ℝ, |x| + x^2 = a) ↔ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_real_solutions_condition_l2927_292758


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2927_292799

-- Define propositions p and q
def p (x : ℝ) : Prop := 1 < x ∧ x < 2
def q (x : ℝ) : Prop := x > 0

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l2927_292799


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l2927_292747

theorem cubic_root_sum_squares (a b c : ℂ) : 
  (a^3 + 3*a^2 - 10*a + 5 = 0) →
  (b^3 + 3*b^2 - 10*b + 5 = 0) →
  (c^3 + 3*c^2 - 10*c + 5 = 0) →
  a^2*b^2 + b^2*c^2 + c^2*a^2 = 70 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l2927_292747


namespace NUMINAMATH_CALUDE_min_sum_with_constraint_l2927_292749

theorem min_sum_with_constraint (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z)
  (h4 : x^2 + y^2 + z^2 + x + 2*y + 3*z = 13/4) :
  x + y + z ≥ (-3 + Real.sqrt 22) / 2 ∧
  ∃ (x' y' z' : ℝ), 0 ≤ x' ∧ 0 ≤ y' ∧ 0 ≤ z' ∧
    x'^2 + y'^2 + z'^2 + x' + 2*y' + 3*z' = 13/4 ∧
    x' + y' + z' = (-3 + Real.sqrt 22) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_with_constraint_l2927_292749


namespace NUMINAMATH_CALUDE_system_solution_l2927_292713

theorem system_solution :
  ∃ a b c d e : ℤ,
    (ab + a + 2*b = 78 ∧
     bc + 3*b + c = 101 ∧
     cd + 5*c + 3*d = 232 ∧
     de + 4*d + 5*e = 360 ∧
     ea + 2*e + 4*a = 192) →
    ((a = 8 ∧ b = 7 ∧ c = 10 ∧ d = 14 ∧ e = 16) ∨
     (a = -12 ∧ b = -9 ∧ c = -16 ∧ d = -24 ∧ e = -24)) :=
by sorry

#check system_solution

end NUMINAMATH_CALUDE_system_solution_l2927_292713


namespace NUMINAMATH_CALUDE_hyperbola_k_range_l2927_292703

-- Define the equation of the hyperbola
def hyperbola_eq (x y k : ℝ) : Prop :=
  x^2 / (3 - k) - y^2 / (k - 1) = 1

-- Define the condition for k to represent a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, hyperbola_eq x y k

-- Theorem statement
theorem hyperbola_k_range :
  ∀ k : ℝ, is_hyperbola k ↔ (1 < k ∧ k < 3) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_k_range_l2927_292703


namespace NUMINAMATH_CALUDE_bicycling_problem_l2927_292765

/-- The number of days after which the condition is satisfied -/
def days : ℕ := 12

/-- The total distance between points A and B in kilometers -/
def total_distance : ℕ := 600

/-- The distance person A travels per day in kilometers -/
def person_a_speed : ℕ := 40

/-- The effective daily distance person B travels in kilometers -/
def person_b_speed : ℕ := 30

/-- The remaining distance for person A after the given number of days -/
def remaining_distance_a : ℕ := total_distance - person_a_speed * days

/-- The remaining distance for person B after the given number of days -/
def remaining_distance_b : ℕ := total_distance - person_b_speed * days

theorem bicycling_problem :
  remaining_distance_b = 2 * remaining_distance_a :=
sorry

end NUMINAMATH_CALUDE_bicycling_problem_l2927_292765


namespace NUMINAMATH_CALUDE_unique_max_divisor_number_l2927_292767

/-- A positive integer N satisfies the special divisor property if all of its divisors
    can be written as p-2 for some prime number p -/
def has_special_divisor_property (N : ℕ+) : Prop :=
  ∀ d : ℕ, d ∣ N.val → ∃ p : ℕ, Nat.Prime p ∧ d = p - 2

/-- The maximum number of divisors for any N satisfying the special divisor property -/
def max_divisors : ℕ := 8

/-- The theorem stating that 135 is the only number with the maximum number of divisors
    satisfying the special divisor property -/
theorem unique_max_divisor_number :
  ∃! N : ℕ+, has_special_divisor_property N ∧
  (Nat.card {d : ℕ | d ∣ N.val} = max_divisors) ∧
  N.val = 135 := by sorry

#check unique_max_divisor_number

end NUMINAMATH_CALUDE_unique_max_divisor_number_l2927_292767


namespace NUMINAMATH_CALUDE_share_calculation_l2927_292768

theorem share_calculation (total : ℚ) (a b c : ℚ) 
  (h_total : total = 578)
  (h_a : a = (2/3) * b)
  (h_b : b = (1/4) * c)
  (h_sum : a + b + c = total) :
  b = 102 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l2927_292768


namespace NUMINAMATH_CALUDE_smallest_base_representation_l2927_292785

/-- Given two bases a and b greater than 2, this function returns the base-10 
    representation of 21 in base a and 12 in base b. -/
def baseRepresentation (a b : ℕ) : ℕ := 2 * a + 1

/-- The smallest base-10 integer that can be represented as 21₍ₐ₎ in one base 
    and 12₍ᵦ₎ in another base, where a and b are any bases larger than 2. -/
def smallestInteger : ℕ := 7

theorem smallest_base_representation :
  ∀ a b : ℕ, a > 2 → b > 2 → 
  (baseRepresentation a b = baseRepresentation b a) → 
  (baseRepresentation a b ≥ smallestInteger) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_representation_l2927_292785


namespace NUMINAMATH_CALUDE_circle_area_tripled_l2927_292786

theorem circle_area_tripled (n : ℝ) (r : ℝ) (h_pos : r > 0) : 
  π * (r + n)^2 = 3 * π * r^2 → r = n * (Real.sqrt 3 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l2927_292786


namespace NUMINAMATH_CALUDE_school_population_l2927_292764

theorem school_population (total_students : ℕ) : total_students = 400 :=
  sorry

end NUMINAMATH_CALUDE_school_population_l2927_292764


namespace NUMINAMATH_CALUDE_alyssa_final_money_l2927_292787

def weekly_allowance : ℕ := 8
def movie_spending : ℕ := weekly_allowance / 2
def car_wash_earnings : ℕ := 8

theorem alyssa_final_money :
  weekly_allowance - movie_spending + car_wash_earnings = 12 :=
by sorry

end NUMINAMATH_CALUDE_alyssa_final_money_l2927_292787


namespace NUMINAMATH_CALUDE_kayla_total_is_15_l2927_292762

def theresa_chocolate : ℕ := 12
def theresa_soda : ℕ := 18

def kayla_chocolate : ℕ := theresa_chocolate / 2
def kayla_soda : ℕ := theresa_soda / 2

def kayla_total : ℕ := kayla_chocolate + kayla_soda

theorem kayla_total_is_15 : kayla_total = 15 := by
  sorry

end NUMINAMATH_CALUDE_kayla_total_is_15_l2927_292762


namespace NUMINAMATH_CALUDE_minji_clothes_combinations_l2927_292726

theorem minji_clothes_combinations (tops : ℕ) (bottoms : ℕ) 
  (h1 : tops = 3) (h2 : bottoms = 5) : tops * bottoms = 15 := by
  sorry

end NUMINAMATH_CALUDE_minji_clothes_combinations_l2927_292726


namespace NUMINAMATH_CALUDE_water_polo_team_selection_l2927_292755

/-- The number of ways to select a starting team in a water polo club. -/
theorem water_polo_team_selection (total_members : Nat) (team_size : Nat) (h1 : total_members = 20) (h2 : team_size = 9) :
  (total_members * Nat.choose (total_members - 1) (team_size - 1) * (team_size - 1)) = 12093120 := by
  sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_l2927_292755


namespace NUMINAMATH_CALUDE_longest_side_of_special_triangle_l2927_292700

-- Define a triangle with sides in arithmetic progression
structure ArithmeticTriangle where
  a : ℝ
  d : ℝ
  angle : ℝ

-- Theorem statement
theorem longest_side_of_special_triangle (t : ArithmeticTriangle) 
  (h1 : t.d = 2)
  (h2 : t.angle = 2 * π / 3) -- 120° in radians
  (h3 : (t.a + t.d)^2 = (t.a - t.d)^2 + t.a^2 - 2*(t.a - t.d)*t.a*(- 1/2)) -- Law of Cosines for 120°
  : t.a + t.d = 7 := by
  sorry

end NUMINAMATH_CALUDE_longest_side_of_special_triangle_l2927_292700


namespace NUMINAMATH_CALUDE_expression_equality_l2927_292748

theorem expression_equality : 12 * 171 + 29 * 9 + 171 * 13 + 29 * 16 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2927_292748


namespace NUMINAMATH_CALUDE_reflection_coordinate_sum_l2927_292754

/-- Given a point A with coordinates (3, y) and its reflection B over the x-axis,
    the sum of all four coordinate values is 6. -/
theorem reflection_coordinate_sum (y : ℝ) : 
  let A : ℝ × ℝ := (3, y)
  let B : ℝ × ℝ := (3, -y)
  (A.1 + A.2 + B.1 + B.2) = 6 := by
  sorry

end NUMINAMATH_CALUDE_reflection_coordinate_sum_l2927_292754


namespace NUMINAMATH_CALUDE_tangent_lines_theorem_l2927_292733

noncomputable def f (x : ℝ) : ℝ := x^2 + x - 2

theorem tangent_lines_theorem :
  let l1 : ℝ → ℝ → Prop := λ x y => x - y - 2 = 0
  let l2 : ℝ → ℝ → Prop := λ x y => x + y + 3 = 0
  (∀ x, deriv f x = 2*x + 1) ∧
  (l1 0 (-2)) ∧
  (∃ a b, f a = b ∧ l2 a b) ∧
  (∀ x y, l1 x y → ∀ x' y', l2 x' y' → (y - (-2)) / (x - 0) * (y' - y) / (x' - x) = -1) →
  (∀ x y, l1 x y ↔ (x - y - 2 = 0)) ∧
  (∀ x y, l2 x y ↔ (x + y + 3 = 0))
:= by sorry

end NUMINAMATH_CALUDE_tangent_lines_theorem_l2927_292733


namespace NUMINAMATH_CALUDE_f_odd_and_decreasing_l2927_292789

-- Define the function f(x) = -x^3
def f (x : ℝ) : ℝ := -x^3

-- State the theorem
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_odd_and_decreasing_l2927_292789


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l2927_292712

def a : ℕ := 999
def b : ℕ := 666

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_9ab : sum_of_digits (9 * a * b) = 36 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l2927_292712


namespace NUMINAMATH_CALUDE_root_value_theorem_l2927_292797

theorem root_value_theorem (a : ℝ) : 
  (2 * a^2 + 3 * a - 4 = 0) → (2 * a^2 + 3 * a = 4) := by
  sorry

end NUMINAMATH_CALUDE_root_value_theorem_l2927_292797


namespace NUMINAMATH_CALUDE_train_length_calculation_l2927_292728

/-- The length of two trains given their speeds and overtaking time -/
theorem train_length_calculation (v1 v2 t : ℝ) (h1 : v1 = 46) (h2 : v2 = 36) (h3 : t = 27) :
  let relative_speed := (v1 - v2) * (5 / 18)
  let distance := relative_speed * t
  let train_length := distance / 2
  train_length = 37.5 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l2927_292728


namespace NUMINAMATH_CALUDE_slip_4_5_in_R_l2927_292753

-- Define the set of slips
def slips : List ℝ := [1, 1.5, 1.5, 2, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4.5, 5, 5.5]

-- Define the boxes
inductive Box
| P | Q | R | S | T | U

-- Define a distribution of slips to boxes
def Distribution := Box → List ℝ

-- Define the constraint that the sum in each box is an integer
def sumIsInteger (d : Distribution) : Prop :=
  ∀ b : Box, ∃ n : ℤ, (d b).sum = n

-- Define the constraint that the sums are consecutive integers
def consecutiveSums (d : Distribution) : Prop :=
  ∃ n : ℤ, (d Box.P).sum = n ∧
           (d Box.Q).sum = n + 1 ∧
           (d Box.R).sum = n + 2 ∧
           (d Box.S).sum = n + 3 ∧
           (d Box.T).sum = n + 4 ∧
           (d Box.U).sum = n + 5

-- Define the constraint that 1 is in box U and 2 is in box Q
def fixedSlips (d : Distribution) : Prop :=
  1 ∈ d Box.U ∧ 2 ∈ d Box.Q

-- Main theorem
theorem slip_4_5_in_R (d : Distribution) 
  (h1 : d Box.P ++ d Box.Q ++ d Box.R ++ d Box.S ++ d Box.T ++ d Box.U = slips)
  (h2 : sumIsInteger d)
  (h3 : consecutiveSums d)
  (h4 : fixedSlips d) :
  4.5 ∈ d Box.R :=
sorry

end NUMINAMATH_CALUDE_slip_4_5_in_R_l2927_292753


namespace NUMINAMATH_CALUDE_maximum_assignment_x_plus_v_l2927_292763

def Values : Finset ℕ := {2, 3, 4, 5}

structure Assignment where
  V : ℕ
  W : ℕ
  X : ℕ
  Y : ℕ
  h1 : V ∈ Values
  h2 : W ∈ Values
  h3 : X ∈ Values
  h4 : Y ∈ Values
  h5 : V ≠ W ∧ V ≠ X ∧ V ≠ Y ∧ W ≠ X ∧ W ≠ Y ∧ X ≠ Y

def ExpressionValue (a : Assignment) : ℕ := a.Y^a.X - a.W^a.V

def MaximumAssignment : Assignment → Prop := λ a => 
  ∀ b : Assignment, ExpressionValue a ≥ ExpressionValue b

theorem maximum_assignment_x_plus_v (a : Assignment) 
  (h : MaximumAssignment a) : a.X + a.V = 8 := by
  sorry

end NUMINAMATH_CALUDE_maximum_assignment_x_plus_v_l2927_292763


namespace NUMINAMATH_CALUDE_complex_product_of_three_l2927_292790

theorem complex_product_of_three (α₁ α₂ α₃ : ℝ) (z₁ z₂ z₃ : ℂ) :
  z₁ = Complex.exp (Complex.I * α₁) →
  z₂ = Complex.exp (Complex.I * α₂) →
  z₃ = Complex.exp (Complex.I * α₃) →
  z₁ * z₂ = Complex.exp (Complex.I * (α₁ + α₂)) →
  z₂ * z₃ = Complex.exp (Complex.I * (α₂ + α₃)) →
  z₁ * z₂ * z₃ = Complex.exp (Complex.I * (α₁ + α₂ + α₃)) :=
by sorry

end NUMINAMATH_CALUDE_complex_product_of_three_l2927_292790


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2927_292708

/-- Given a hyperbola with center (1, -1), one focus at (1, 5), one vertex at (1, 2),
    and equation ((y-k)^2 / a^2) - ((x-h)^2 / b^2) = 1,
    prove that h + k + a + b = 3√3 + 3 -/
theorem hyperbola_sum (h k a b : ℝ) : 
  h = 1 ∧ k = -1 ∧  -- center at (1, -1)
  ∃ (x y : ℝ), x = 1 ∧ y = 5 ∧  -- one focus at (1, 5)
    (y - k)^2 = (x - h)^2 + a^2 ∧  -- relationship between focus, center, and a
  ∃ (x y : ℝ), x = 1 ∧ y = 2 ∧  -- one vertex at (1, 2)
    (y - k)^2 = a^2 ∧  -- relationship between vertex, center, and a
  ∀ (x y : ℝ), ((y - k)^2 / a^2) - ((x - h)^2 / b^2) = 1  -- equation of hyperbola
  →
  h + k + a + b = 3 * Real.sqrt 3 + 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2927_292708


namespace NUMINAMATH_CALUDE_bus_stop_problem_l2927_292732

theorem bus_stop_problem (boys girls : ℕ) : 
  (boys = 2 * (girls - 15)) →
  (girls - 15 = 5 * (boys - 45)) →
  (boys = 50 ∧ girls = 40) := by
  sorry

end NUMINAMATH_CALUDE_bus_stop_problem_l2927_292732


namespace NUMINAMATH_CALUDE_product_evaluation_l2927_292740

def product_term (n : ℕ) : ℚ := (n * (n + 2) + n) / ((n + 1)^2 : ℚ)

def product_series : ℕ → ℚ
  | 0 => 1
  | n + 1 => product_series n * product_term (n + 1)

theorem product_evaluation : 
  product_series 98 = 9800 / 9801 := by sorry

end NUMINAMATH_CALUDE_product_evaluation_l2927_292740


namespace NUMINAMATH_CALUDE_pass_percentage_l2927_292735

theorem pass_percentage 
  (passed_english : Real) 
  (passed_math : Real) 
  (failed_both : Real) 
  (h1 : passed_english = 63) 
  (h2 : passed_math = 65) 
  (h3 : failed_both = 27) : 
  100 - failed_both = 73 := by
  sorry

end NUMINAMATH_CALUDE_pass_percentage_l2927_292735


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l2927_292750

theorem inequality_of_positive_reals (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  b * c / a + a * c / b + a * b / c ≥ a + b + c := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l2927_292750


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l2927_292778

theorem quadratic_distinct_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*x₁ + 4*c = 0 ∧ x₂^2 + 2*x₂ + 4*c = 0) ↔ c < 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l2927_292778


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2927_292777

theorem gcd_lcm_product (a b : ℕ) (ha : a = 180) (hb : b = 225) :
  (Nat.gcd a b) * (Nat.lcm a b) = 40500 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2927_292777


namespace NUMINAMATH_CALUDE_trajectory_of_M_l2927_292798

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 16

-- Define point A
def point_A : ℝ × ℝ := (1, 0)

-- Define a point Q on the circle
def point_Q (x y : ℝ) : Prop := circle_C x y

-- Define point M as the intersection of the perpendicular bisector of AQ and CQ
def point_M (x y : ℝ) : Prop :=
  ∃ (qx qy : ℝ), point_Q qx qy ∧
  (x - 1)^2 + y^2 = (x - qx)^2 + (y - qy)^2 ∧
  (x + qx = 1) ∧ (y + qy = 0)

-- Theorem statement
theorem trajectory_of_M :
  ∀ (x y : ℝ), point_M x y → x^2/4 + y^2/3 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_of_M_l2927_292798


namespace NUMINAMATH_CALUDE_problem_statement_l2927_292759

def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 ≤ 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | (x - 1) * (x - a) ≤ 0}

theorem problem_statement :
  (∀ a : ℝ, B a ⊆ A → a ∈ Set.Icc 1 2) ∧
  (∀ a : ℝ, A ∩ B a = {1} → a ∈ Set.Iic 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2927_292759


namespace NUMINAMATH_CALUDE_altitudes_constructible_l2927_292739

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle in a 2D plane -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents a circle in a 2D plane -/
structure Circle :=
  (center : Point)
  (radius : ℝ)

/-- Represents construction tools -/
inductive ConstructionTool
  | Straightedge
  | Protractor

/-- Represents an altitude of a triangle -/
structure Altitude :=
  (base : Point)
  (apex : Point)

/-- Function to construct altitudes of a triangle -/
def constructAltitudes (t : Triangle) (c : Circle) (tools : List ConstructionTool) : 
  List Altitude :=
  sorry

/-- Theorem stating that altitudes can be constructed -/
theorem altitudes_constructible (t : Triangle) (c : Circle) : 
  ∃ (tools : List ConstructionTool), 
    (ConstructionTool.Straightedge ∈ tools) ∧ 
    (ConstructionTool.Protractor ∈ tools) ∧ 
    (constructAltitudes t c tools).length = 3 :=
  sorry

end NUMINAMATH_CALUDE_altitudes_constructible_l2927_292739


namespace NUMINAMATH_CALUDE_age_problem_l2927_292794

theorem age_problem (age_older age_younger : ℕ) : 
  age_older = age_younger + 2 →
  age_older + age_younger = 74 →
  age_older = 38 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2927_292794


namespace NUMINAMATH_CALUDE_team_selection_ways_eq_103950_l2927_292711

/-- The number of ways to select a team of 8 people, consisting of 4 boys from a group of 10 boys
    and 4 girls from a group of 12 girls. -/
def team_selection_ways : ℕ :=
  Nat.choose 10 4 * Nat.choose 12 4

/-- Theorem stating that the number of ways to select the team is 103950. -/
theorem team_selection_ways_eq_103950 : team_selection_ways = 103950 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_ways_eq_103950_l2927_292711


namespace NUMINAMATH_CALUDE_one_eighth_of_2_36_l2927_292792

theorem one_eighth_of_2_36 (y : ℤ) : (1 / 8 : ℚ) * (2 ^ 36) = 2 ^ y → y = 33 := by
  sorry

end NUMINAMATH_CALUDE_one_eighth_of_2_36_l2927_292792


namespace NUMINAMATH_CALUDE_fifth_dog_weight_l2927_292704

def dog_weights : List ℝ := [25, 31, 35, 33]

theorem fifth_dog_weight (w : ℝ) :
  (dog_weights.sum + w) / 5 = dog_weights.sum / 4 →
  w = 31 :=
by sorry

end NUMINAMATH_CALUDE_fifth_dog_weight_l2927_292704


namespace NUMINAMATH_CALUDE_shopkeeper_profit_percent_l2927_292783

theorem shopkeeper_profit_percent (initial_value : ℝ) (theft_percent : ℝ) (overall_loss_percent : ℝ) :
  theft_percent = 20 →
  overall_loss_percent = 12 →
  initial_value > 0 →
  let remaining_value := initial_value * (1 - theft_percent / 100)
  let selling_price := initial_value * (1 - overall_loss_percent / 100)
  let profit := selling_price - remaining_value
  let profit_percent := (profit / remaining_value) * 100
  profit_percent = 10 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_percent_l2927_292783


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2927_292781

/-- Given a quadratic function y = x^2 - 2px - p with two distinct roots, 
    prove properties about p and the roots. -/
theorem quadratic_function_properties (p : ℝ) 
  (h_distinct : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 - 2*p*x₁ - p = 0 ∧ x₂^2 - 2*p*x₂ - p = 0) :
  (∃ (x₁ x₂ : ℝ), 2*p*x₁ + x₂^2 + 3*p > 0) ∧
  (∃ (max_p : ℝ), max_p = 9/16 ∧ 
    ∀ (q : ℝ), (∃ (x₁ x₂ : ℝ), x₁^2 - 2*q*x₁ - q = 0 ∧ x₂^2 - 2*q*x₂ - q = 0 ∧ |x₁ - x₂| ≤ |2*q - 3|) 
    → q ≤ max_p) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2927_292781


namespace NUMINAMATH_CALUDE_no_statement_implies_p_and_not_q_l2927_292772

theorem no_statement_implies_p_and_not_q (p q : Prop) : 
  ¬((p → q) → (p ∧ ¬q)) ∧ 
  ¬((p ∨ ¬q) → (p ∧ ¬q)) ∧ 
  ¬((¬p ∧ q) → (p ∧ ¬q)) ∧ 
  ¬((¬p ∨ q) → (p ∧ ¬q)) := by
  sorry

end NUMINAMATH_CALUDE_no_statement_implies_p_and_not_q_l2927_292772


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_symmetry_l2927_292741

-- Define the points
variable (A B C D A₁ B₁ C₁ D₁ P : Point)

-- Define the property of being cyclic
def is_cyclic (A B C D : Point) : Prop := sorry

-- Define symmetry with respect to a point
def symmetrical_wrt (A B : Point) (P : Point) : Prop := sorry

-- State the theorem
theorem cyclic_quadrilateral_symmetry 
  (h1 : symmetrical_wrt A A₁ P) 
  (h2 : symmetrical_wrt B B₁ P) 
  (h3 : symmetrical_wrt C C₁ P) 
  (h4 : symmetrical_wrt D D₁ P)
  (h5 : is_cyclic A₁ B C D)
  (h6 : is_cyclic A B₁ C D)
  (h7 : is_cyclic A B C₁ D) :
  is_cyclic A B C D₁ := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_symmetry_l2927_292741


namespace NUMINAMATH_CALUDE_emily_chairs_l2927_292742

/-- The number of chairs Emily bought -/
def num_chairs : ℕ := sorry

/-- The number of tables Emily bought -/
def num_tables : ℕ := 2

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_furniture : ℕ := 8

/-- The total time spent (in minutes) -/
def total_time : ℕ := 48

theorem emily_chairs : 
  num_chairs = 4 ∧ 
  time_per_furniture * (num_chairs + num_tables) = total_time :=
sorry

end NUMINAMATH_CALUDE_emily_chairs_l2927_292742


namespace NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l2927_292719

def S : Finset ℕ := Finset.range 11

def difference_sum (S : Finset ℕ) : ℕ :=
  S.sum (λ i => S.sum (λ j => if i < j then 3^j - 3^i else 0))

theorem difference_sum_of_powers_of_three : difference_sum S = 787484 := by
  sorry

end NUMINAMATH_CALUDE_difference_sum_of_powers_of_three_l2927_292719


namespace NUMINAMATH_CALUDE_probability_of_three_successes_l2927_292775

def n : ℕ := 7
def k : ℕ := 3
def p : ℚ := 1/3

theorem probability_of_three_successes :
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_successes_l2927_292775


namespace NUMINAMATH_CALUDE_transistors_in_2010_l2927_292743

/-- Moore's law doubling period in years -/
def moores_law_period : ℕ := 2

/-- Initial year for calculation -/
def initial_year : ℕ := 1992

/-- Target year for calculation -/
def target_year : ℕ := 2010

/-- Initial number of transistors in 1992 -/
def initial_transistors : ℕ := 500000

/-- Calculate the number of transistors in a given year according to Moore's law -/
def transistors_in_year (year : ℕ) : ℕ :=
  initial_transistors * 2^((year - initial_year) / moores_law_period)

/-- Theorem stating the number of transistors in 2010 -/
theorem transistors_in_2010 :
  transistors_in_year target_year = 256000000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_in_2010_l2927_292743


namespace NUMINAMATH_CALUDE_quiche_theorem_l2927_292760

/-- Quiche ingredients and their properties --/
structure QuicheIngredients where
  spinach_initial : ℝ
  mushrooms_initial : ℝ
  onions_initial : ℝ
  spinach_reduction : ℝ
  mushrooms_reduction : ℝ
  onions_reduction : ℝ
  cream_cheese_volume : ℝ
  cream_cheese_calories : ℝ
  eggs_volume : ℝ
  eggs_calories : ℝ
  oz_to_cup_conversion : ℝ

/-- Calculate the total volume and calorie content of the quiche --/
def quiche_properties (ingredients : QuicheIngredients) : ℝ × ℝ :=
  let cooked_spinach := ingredients.spinach_initial * ingredients.spinach_reduction
  let cooked_mushrooms := ingredients.mushrooms_initial * ingredients.mushrooms_reduction
  let cooked_onions := ingredients.onions_initial * ingredients.onions_reduction
  let total_volume_oz := cooked_spinach + cooked_mushrooms + cooked_onions + 
                         ingredients.cream_cheese_volume + ingredients.eggs_volume
  let total_volume_cups := total_volume_oz * ingredients.oz_to_cup_conversion
  let total_calories := ingredients.cream_cheese_volume * ingredients.cream_cheese_calories + 
                        ingredients.eggs_volume * ingredients.eggs_calories
  (total_volume_cups, total_calories)

/-- Theorem stating the properties of the quiche --/
theorem quiche_theorem (ingredients : QuicheIngredients) 
  (h1 : ingredients.spinach_initial = 40)
  (h2 : ingredients.mushrooms_initial = 25)
  (h3 : ingredients.onions_initial = 15)
  (h4 : ingredients.spinach_reduction = 0.2)
  (h5 : ingredients.mushrooms_reduction = 0.65)
  (h6 : ingredients.onions_reduction = 0.5)
  (h7 : ingredients.cream_cheese_volume = 6)
  (h8 : ingredients.cream_cheese_calories = 80)
  (h9 : ingredients.eggs_volume = 4)
  (h10 : ingredients.eggs_calories = 70)
  (h11 : ingredients.oz_to_cup_conversion = 0.125) :
  quiche_properties ingredients = (5.21875, 760) := by
  sorry

#eval quiche_properties {
  spinach_initial := 40,
  mushrooms_initial := 25,
  onions_initial := 15,
  spinach_reduction := 0.2,
  mushrooms_reduction := 0.65,
  onions_reduction := 0.5,
  cream_cheese_volume := 6,
  cream_cheese_calories := 80,
  eggs_volume := 4,
  eggs_calories := 70,
  oz_to_cup_conversion := 0.125
}

end NUMINAMATH_CALUDE_quiche_theorem_l2927_292760


namespace NUMINAMATH_CALUDE_polynomial_properties_l2927_292701

def f (x : ℝ) : ℝ := x^3 - 2*x

theorem polynomial_properties :
  (∀ x y : ℚ, f x = f y → x = y) ∧
  (∃ a b : ℝ, a ≠ b ∧ f a = f b) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l2927_292701


namespace NUMINAMATH_CALUDE_sum_of_squares_is_312_l2927_292791

/-- Represents the rates and distances for biking, jogging, and swimming activities. -/
structure ActivityRates where
  bike_rate : ℕ
  jog_rate : ℕ
  swim_rate : ℕ

/-- Calculates the total distance covered given rates and times. -/
def total_distance (rates : ActivityRates) (bike_time jog_time swim_time : ℕ) : ℕ :=
  rates.bike_rate * bike_time + rates.jog_rate * jog_time + rates.swim_rate * swim_time

/-- Theorem stating that given the conditions, the sum of squares of rates is 312. -/
theorem sum_of_squares_is_312 (rates : ActivityRates) : 
  total_distance rates 1 4 3 = 66 ∧ 
  total_distance rates 3 3 2 = 76 → 
  rates.bike_rate ^ 2 + rates.jog_rate ^ 2 + rates.swim_rate ^ 2 = 312 := by
  sorry

#check sum_of_squares_is_312

end NUMINAMATH_CALUDE_sum_of_squares_is_312_l2927_292791


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2927_292737

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x + 1/y) ≥ 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2927_292737


namespace NUMINAMATH_CALUDE_special_sequence_characterization_l2927_292746

/-- A sequence of real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, a n ≤ a (n + 1)) ∧ 
  (∀ m n : ℕ, a (m^2 + n^2) = (a m)^2 + (a n)^2)

/-- The theorem stating the only possible sequences satisfying the conditions -/
theorem special_sequence_characterization (a : ℕ → ℝ) :
  SpecialSequence a →
  ((∀ n, a n = 0) ∨ (∀ n, a n = 1/2) ∨ (∀ n, a n = n)) :=
by sorry

end NUMINAMATH_CALUDE_special_sequence_characterization_l2927_292746


namespace NUMINAMATH_CALUDE_not_hyperbola_equation_l2927_292724

/-- A hyperbola with given properties -/
structure Hyperbola where
  center_at_origin : Bool
  symmetric_about_axes : Bool
  eccentricity : ℝ
  focus_to_asymptote_distance : ℝ

/-- The equation of a hyperbola -/
def hyperbola_equation (a b : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 / a - y^2 / b = 1

/-- Theorem stating that the given equation cannot be the equation of the hyperbola with the specified properties -/
theorem not_hyperbola_equation (M : Hyperbola) 
  (h1 : M.center_at_origin = true)
  (h2 : M.symmetric_about_axes = true)
  (h3 : M.eccentricity = Real.sqrt 3)
  (h4 : M.focus_to_asymptote_distance = 2) :
  ¬(hyperbola_equation 4 2 = fun x y => x^2 / 4 - y^2 / 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_not_hyperbola_equation_l2927_292724


namespace NUMINAMATH_CALUDE_at_least_two_primes_of_form_l2927_292721

theorem at_least_two_primes_of_form (n : ℕ) : ∃ (a b : ℕ), 2 ≤ a ∧ 2 ≤ b ∧ a ≠ b ∧ 
  Nat.Prime (a^3 + a^2 + 1) ∧ Nat.Prime (b^3 + b^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_primes_of_form_l2927_292721


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2927_292738

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - 2*x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2927_292738


namespace NUMINAMATH_CALUDE_triangle_square_apothem_equality_l2927_292752

/-- Theorem: Value of k for a specific right triangle and square configuration -/
theorem triangle_square_apothem_equality (x : ℝ) (k : ℝ) : 
  x > 0 →  -- Ensure positive side lengths
  (3*x)^2 + (4*x)^2 = (5*x)^2 →  -- Pythagorean theorem for right triangle
  12*x = k * (6*x^2) →  -- Perimeter = k * Area for triangle
  4*x = 5 →  -- Apothem equality
  100 = 3 * 40 →  -- Square area = 3 * Square perimeter
  k = 8/5 := by sorry

end NUMINAMATH_CALUDE_triangle_square_apothem_equality_l2927_292752


namespace NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l2927_292782

theorem no_prime_roots_for_quadratic : 
  ¬ ∃ (k : ℤ), ∃ (p q : ℕ), 
    Prime p ∧ Prime q ∧ 
    p ≠ q ∧
    (p : ℤ) + q = 107 ∧ 
    (p : ℤ) * q = k ∧
    p^2 - 107*p + k = 0 ∧ 
    q^2 - 107*q + k = 0 :=
sorry

end NUMINAMATH_CALUDE_no_prime_roots_for_quadratic_l2927_292782


namespace NUMINAMATH_CALUDE_mechanics_billing_problem_l2927_292769

/-- A mechanic's billing problem -/
theorem mechanics_billing_problem 
  (total_bill : ℝ) 
  (parts_cost : ℝ) 
  (job_duration : ℝ) 
  (h1 : total_bill = 450)
  (h2 : parts_cost = 225)
  (h3 : job_duration = 5) :
  (total_bill - parts_cost) / job_duration = 45 := by
sorry

end NUMINAMATH_CALUDE_mechanics_billing_problem_l2927_292769


namespace NUMINAMATH_CALUDE_intersection_proof_l2927_292745

def S : Set Nat := {0, 1, 3, 5, 7, 9}

theorem intersection_proof (A B : Set Nat) 
  (h1 : S = {0, 1, 3, 5, 7, 9})
  (h2 : (S \ A) = {0, 5, 9})
  (h3 : B = {3, 5, 7}) :
  A ∩ B = {3, 7} := by
sorry

end NUMINAMATH_CALUDE_intersection_proof_l2927_292745


namespace NUMINAMATH_CALUDE_solution_set_when_a_neg_one_range_of_a_l2927_292729

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 1|
def g (a : ℝ) (x : ℝ) : ℝ := 2 * |x| + a

-- Theorem for part (1)
theorem solution_set_when_a_neg_one :
  {x : ℝ | f x ≤ g (-1) x} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} := by sorry

-- Theorem for part (2)
theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ ≥ (1/2) * g a x₀) → a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_neg_one_range_of_a_l2927_292729


namespace NUMINAMATH_CALUDE_hydrochloric_acid_mixture_l2927_292717

def total_mass : ℝ := 600
def final_concentration : ℝ := 0.15
def concentration_1 : ℝ := 0.3
def concentration_2 : ℝ := 0.1
def mass_1 : ℝ := 150
def mass_2 : ℝ := 450

theorem hydrochloric_acid_mixture :
  mass_1 + mass_2 = total_mass ∧
  (concentration_1 * mass_1 + concentration_2 * mass_2) / total_mass = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_hydrochloric_acid_mixture_l2927_292717


namespace NUMINAMATH_CALUDE_reflection_of_C_l2927_292716

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point C -/
def C : ℝ × ℝ := (3, 1)

theorem reflection_of_C :
  (reflect_x ∘ reflect_y) C = (-3, -1) := by sorry

end NUMINAMATH_CALUDE_reflection_of_C_l2927_292716


namespace NUMINAMATH_CALUDE_cycling_average_speed_l2927_292714

theorem cycling_average_speed 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (rest_duration : ℝ) 
  (num_rests : ℕ) 
  (h1 : total_distance = 56) 
  (h2 : total_time = 8) 
  (h3 : rest_duration = 0.5) 
  (h4 : num_rests = 2) : 
  total_distance / (total_time - num_rests * rest_duration) = 8 := by
sorry

end NUMINAMATH_CALUDE_cycling_average_speed_l2927_292714


namespace NUMINAMATH_CALUDE_sams_new_nickels_l2927_292718

/-- The number of nickels Sam's dad gave him -/
def nickels_from_dad (initial_nickels final_nickels : ℕ) : ℕ :=
  final_nickels - initial_nickels

/-- Proof that Sam's dad gave him 39 nickels -/
theorem sams_new_nickels :
  let initial_nickels : ℕ := 24
  let final_nickels : ℕ := 63
  nickels_from_dad initial_nickels final_nickels = 39 := by
sorry

end NUMINAMATH_CALUDE_sams_new_nickels_l2927_292718


namespace NUMINAMATH_CALUDE_age_ratio_proof_l2927_292751

/-- Given three people a, b, and c, with the following conditions:
  1. a is two years older than b
  2. The total of the ages of a, b, and c is 72
  3. b is 28 years old
Prove that the ratio of b's age to c's age is 2:1 -/
theorem age_ratio_proof (a b c : ℕ) : 
  a = b + 2 →
  a + b + c = 72 →
  b = 28 →
  b / c = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l2927_292751


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_l2927_292702

theorem cube_roots_of_unity (α β : ℂ) 
  (h1 : Complex.abs α = 1) 
  (h2 : Complex.abs β = 1) 
  (h3 : α + β + 1 = 0) : 
  α^3 = 1 ∧ β^3 = 1 := by
sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_l2927_292702


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2927_292707

theorem quadratic_equations_solutions :
  (∃ x1 x2 : ℝ, x1 = 4 + 3 * Real.sqrt 2 ∧ x2 = 4 - 3 * Real.sqrt 2 ∧ 
    x1^2 - 8*x1 - 2 = 0 ∧ x2^2 - 8*x2 - 2 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = 3/2 ∧ x2 = -1 ∧ 
    2*x1^2 - x1 - 3 = 0 ∧ 2*x2^2 - x2 - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2927_292707


namespace NUMINAMATH_CALUDE_grocery_store_distance_l2927_292774

theorem grocery_store_distance (distance_house_to_park : ℝ) 
                               (distance_park_to_store : ℝ) 
                               (total_distance : ℝ) : ℝ := by
  have h1 : distance_house_to_park = 5 := by sorry
  have h2 : distance_park_to_store = 3 := by sorry
  have h3 : total_distance = 16 := by sorry
  
  let distance_store_to_house := total_distance - distance_house_to_park - distance_park_to_store
  
  have h4 : distance_store_to_house = 
            total_distance - distance_house_to_park - distance_park_to_store := by rfl
  
  exact distance_store_to_house

end NUMINAMATH_CALUDE_grocery_store_distance_l2927_292774


namespace NUMINAMATH_CALUDE_health_risk_factors_l2927_292793

theorem health_risk_factors (total_population : ℝ) 
  (prob_single : ℝ) (prob_pair : ℝ) (prob_all_given_two : ℝ) :
  prob_single = 0.08 →
  prob_pair = 0.15 →
  prob_all_given_two = 1/4 →
  ∃ (prob_none_given_not_one : ℝ),
    prob_none_given_not_one = 26/57 := by
  sorry

end NUMINAMATH_CALUDE_health_risk_factors_l2927_292793


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l2927_292771

structure Rectangle where
  width : ℝ
  height : ℝ

def similar (r1 r2 : Rectangle) : Prop :=
  r1.width / r2.width = r1.height / r2.height

theorem rectangle_area_ratio 
  (ABCD EFGH : Rectangle) 
  (h1 : similar ABCD EFGH) 
  (h2 : ∃ (K : ℝ), K > 0 ∧ K < ABCD.width ∧ (ABCD.width - K) / K = 2 / 3) : 
  (ABCD.width * ABCD.height) / (EFGH.width * EFGH.height) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l2927_292771


namespace NUMINAMATH_CALUDE_donna_episodes_per_weekday_l2927_292776

theorem donna_episodes_per_weekday : 
  ∀ (weekday_episodes : ℕ),
  weekday_episodes > 0 →
  5 * weekday_episodes + 2 * (3 * weekday_episodes) = 88 →
  weekday_episodes = 8 := by
sorry

end NUMINAMATH_CALUDE_donna_episodes_per_weekday_l2927_292776


namespace NUMINAMATH_CALUDE_new_game_cost_new_game_cost_is_8_l2927_292720

def initial_money : ℕ := 57
def toy_cost : ℕ := 4
def num_toys : ℕ := 2

theorem new_game_cost : ℕ :=
  initial_money - (toy_cost * num_toys)

#check new_game_cost

theorem new_game_cost_is_8 : new_game_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_new_game_cost_new_game_cost_is_8_l2927_292720


namespace NUMINAMATH_CALUDE_unique_square_with_special_property_l2927_292723

/-- Checks if a number uses exactly 5 different non-zero digits in base 6 --/
def hasFiveDifferentNonZeroDigitsBase6 (n : ℕ) : Prop := sorry

/-- Converts a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : List ℕ := sorry

/-- Moves the last digit of a number to the front --/
def moveLastToFront (n : ℕ) : ℕ := sorry

/-- Reverses the digits of a number --/
def reverseDigits (n : ℕ) : ℕ := sorry

theorem unique_square_with_special_property :
  ∃! n : ℕ,
    n ^ 2 ≤ 54321 ∧
    n ^ 2 ≥ 12345 ∧
    hasFiveDifferentNonZeroDigitsBase6 (n ^ 2) ∧
    (∃ m : ℕ, m ^ 2 = moveLastToFront (n ^ 2) ∧
              m = reverseDigits n) ∧
    n = 221 := by sorry

end NUMINAMATH_CALUDE_unique_square_with_special_property_l2927_292723


namespace NUMINAMATH_CALUDE_min_value_of_f_on_interval_l2927_292709

-- Define the function f(x)
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the interval [0, 3]
def interval : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem min_value_of_f_on_interval :
  ∃ (min : ℝ), min = 0 ∧ ∀ x ∈ interval, f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_on_interval_l2927_292709


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2927_292744

theorem inequality_system_solution (x : ℝ) :
  (3 * (x - 2) ≤ x - 4 ∧ (1 + 2 * x) / 3 > x - 1) ↔ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2927_292744


namespace NUMINAMATH_CALUDE_f_two_l2927_292706

/-- A linear function satisfying certain conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The inverse function of f -/
def f_inv (x : ℝ) : ℝ := sorry

/-- f is a linear function -/
axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- f satisfies the equation f(x) = 3f^(-1)(x) + 5 -/
axiom f_equation : ∀ x, f x = 3 * f_inv x + 5

/-- f(1) = 5 -/
axiom f_one : f 1 = 5

/-- The main theorem: f(2) = 3 -/
theorem f_two : f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_f_two_l2927_292706


namespace NUMINAMATH_CALUDE_james_streaming_income_l2927_292784

/-- James' streaming income calculation --/
theorem james_streaming_income 
  (initial_subscribers : ℕ) 
  (gifted_subscribers : ℕ) 
  (income_per_subscriber : ℕ) : ℕ :=
  by
  have total_subscribers : ℕ := initial_subscribers + gifted_subscribers
  have monthly_income : ℕ := total_subscribers * income_per_subscriber
  exact monthly_income

#check james_streaming_income 150 50 9

end NUMINAMATH_CALUDE_james_streaming_income_l2927_292784


namespace NUMINAMATH_CALUDE_larger_number_problem_l2927_292715

theorem larger_number_problem (x y : ℝ) : 4 * y = 5 * x → x + y = 54 → y = 30 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l2927_292715


namespace NUMINAMATH_CALUDE_cindys_age_l2927_292779

/-- Given the ages of siblings, prove Cindy's age -/
theorem cindys_age (cindy jan marcia greg : ℕ) 
  (h1 : jan = cindy + 2)
  (h2 : marcia = 2 * jan)
  (h3 : greg = marcia + 2)
  (h4 : greg = 16) :
  cindy = 5 := by
  sorry

end NUMINAMATH_CALUDE_cindys_age_l2927_292779


namespace NUMINAMATH_CALUDE_count_four_digit_divisible_by_5_ending_0_is_900_l2927_292736

/-- A function that counts the number of positive four-digit integers divisible by 5 and ending in 0 -/
def count_four_digit_divisible_by_5_ending_0 : ℕ :=
  let first_digit := Finset.range 9  -- 1 to 9
  let second_digit := Finset.range 10  -- 0 to 9
  let third_digit := Finset.range 10  -- 0 to 9
  (first_digit.card * second_digit.card * third_digit.card : ℕ)

/-- Theorem stating that the count of positive four-digit integers divisible by 5 and ending in 0 is 900 -/
theorem count_four_digit_divisible_by_5_ending_0_is_900 :
  count_four_digit_divisible_by_5_ending_0 = 900 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_divisible_by_5_ending_0_is_900_l2927_292736


namespace NUMINAMATH_CALUDE_f_eval_at_one_l2927_292722

-- Define the polynomials g and f
def g (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 2*x + 15
def f (b c : ℝ) (x : ℝ) : ℝ := x^4 + 2*x^3 + b*x^2 + 150*x + c

-- State the theorem
theorem f_eval_at_one (a b c : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    g a x = 0 ∧ g a y = 0 ∧ g a z = 0 ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0) →
  f b c 1 = -15640 :=
by sorry

end NUMINAMATH_CALUDE_f_eval_at_one_l2927_292722


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l2927_292756

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l2927_292756


namespace NUMINAMATH_CALUDE_swimming_contest_proof_l2927_292796

def kelly_time : ℕ := 3 * 60  -- Kelly's time in seconds

def brittany_time (kelly : ℕ) : ℕ := kelly - 20

def buffy_time (brittany : ℕ) : ℕ := brittany - 40

def carmen_time (kelly : ℕ) : ℕ := kelly + 15

def denise_time (carmen : ℕ) : ℕ := carmen - 35

def total_time (kelly brittany buffy carmen denise : ℕ) : ℕ :=
  kelly + brittany + buffy + carmen + denise

def average_time (total : ℕ) (count : ℕ) : ℕ := total / count

theorem swimming_contest_proof :
  let kelly := kelly_time
  let brittany := brittany_time kelly
  let buffy := buffy_time brittany
  let carmen := carmen_time kelly
  let denise := denise_time carmen
  let total := total_time kelly brittany buffy carmen denise
  let avg := average_time total 5
  total = 815 ∧ avg = 163 := by
  sorry

end NUMINAMATH_CALUDE_swimming_contest_proof_l2927_292796


namespace NUMINAMATH_CALUDE_arithmetic_mean_odd_eq_n_l2927_292710

/-- The sum of the first n odd positive integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n^2

/-- The arithmetic mean of the first n odd positive integers -/
def arithmetic_mean_odd (n : ℕ) : ℚ := (sum_first_n_odd n : ℚ) / n

/-- Theorem: The arithmetic mean of the first n odd positive integers is equal to n -/
theorem arithmetic_mean_odd_eq_n (n : ℕ) (h : n > 0) : 
  arithmetic_mean_odd n = n := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_odd_eq_n_l2927_292710
