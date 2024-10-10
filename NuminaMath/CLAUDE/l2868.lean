import Mathlib

namespace no_three_numbers_exist_l2868_286819

theorem no_three_numbers_exist : ¬∃ (a b c : ℕ), 
  a > 1 ∧ b > 1 ∧ c > 1 ∧ 
  (a * a - 1) % b = 0 ∧ (a * a - 1) % c = 0 ∧
  (b * b - 1) % a = 0 ∧ (b * b - 1) % c = 0 ∧
  (c * c - 1) % a = 0 ∧ (c * c - 1) % b = 0 :=
by sorry


end no_three_numbers_exist_l2868_286819


namespace arithmetic_sequence_product_l2868_286849

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  is_arithmetic_sequence b →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 4 * b 6 = 17 →
  b 3 * b 7 = -175 :=
by sorry

end arithmetic_sequence_product_l2868_286849


namespace simplify_expression_l2868_286856

theorem simplify_expression (x : ℝ) : (2*x)^3 + (3*x)*(x^2) = 11*x^3 := by
  sorry

end simplify_expression_l2868_286856


namespace tallest_person_position_l2868_286868

/-- Represents a person with a height -/
structure Person where
  height : ℝ

/-- A line of people sorted by height -/
def SortedLine (n : ℕ) := Fin n → Person

theorem tallest_person_position
  (n : ℕ)
  (line : SortedLine n)
  (h_sorted : ∀ i j : Fin n, i < j → (line i).height ≤ (line j).height)
  (tallest : Fin n)
  (h_tallest : ∀ i : Fin n, (line i).height ≤ (line tallest).height) :
  tallest.val + 1 = n :=
sorry

end tallest_person_position_l2868_286868


namespace correct_contribution_l2868_286829

/-- Represents the amount spent by each person -/
structure Expenses where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Represents the contribution from Person C to others -/
structure Contribution where
  to_a : ℚ
  to_b : ℚ

def calculate_contribution (e : Expenses) : Contribution :=
  { to_a := 6,
    to_b := 3 }

theorem correct_contribution (e : Expenses) :
  e.b = 12/13 * e.a ∧ 
  e.c = 2/3 * e.b ∧ 
  calculate_contribution e = { to_a := 6, to_b := 3 } :=
by sorry

#check correct_contribution

end correct_contribution_l2868_286829


namespace p_shape_points_for_10cm_square_l2868_286889

/-- Calculates the number of points on a "P" shape formed from a square --/
def count_points_on_p_shape (square_side_length : ℕ) : ℕ :=
  let points_per_side := square_side_length + 1
  let total_sides := 3
  let overlapping_vertices := 2
  points_per_side * total_sides - overlapping_vertices

/-- Theorem stating that a "P" shape formed from a 10 cm square has 31 points --/
theorem p_shape_points_for_10cm_square :
  count_points_on_p_shape 10 = 31 := by
  sorry

#eval count_points_on_p_shape 10  -- Should output 31

end p_shape_points_for_10cm_square_l2868_286889


namespace right_triangle_hypotenuse_l2868_286893

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (hypotenuse : ℝ) : 
  leg = 15 →
  angle = 45 →
  hypotenuse = leg * Real.sqrt 2 →
  hypotenuse = 15 * Real.sqrt 2 :=
by
  sorry

#check right_triangle_hypotenuse

end right_triangle_hypotenuse_l2868_286893


namespace jenny_friends_count_l2868_286874

theorem jenny_friends_count (cost_per_night : ℕ) (nights : ℕ) (total_cost : ℕ) : 
  cost_per_night = 40 →
  nights = 3 →
  total_cost = 360 →
  (1 + 2) * (cost_per_night * nights) = total_cost :=
by
  sorry

#check jenny_friends_count

end jenny_friends_count_l2868_286874


namespace chess_team_arrangement_count_l2868_286869

def chess_team_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  if num_boys + num_girls ≠ 7 then 0
  else if num_boys ≠ 3 then 0
  else if num_girls ≠ 4 then 0
  else Nat.factorial num_boys * Nat.factorial num_girls

theorem chess_team_arrangement_count :
  chess_team_arrangements 3 4 = 144 := by
  sorry

end chess_team_arrangement_count_l2868_286869


namespace geometric_sequence_increasing_condition_l2868_286814

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_increasing_condition
  (a : ℕ → ℝ)
  (h_geometric : is_geometric_sequence a)
  (h_positive : a 1 > 0) :
  (is_increasing_sequence a → a 1^2 < a 2^2) ∧
  (a 1^2 < a 2^2 → ¬(is_increasing_sequence a → False)) :=
sorry

end geometric_sequence_increasing_condition_l2868_286814


namespace units_digit_of_product_is_two_l2868_286832

def first_composite : ℕ := 4
def second_composite : ℕ := 6
def third_composite : ℕ := 8

def product_of_first_three_composites : ℕ := first_composite * second_composite * third_composite

theorem units_digit_of_product_is_two :
  product_of_first_three_composites % 10 = 2 := by sorry

end units_digit_of_product_is_two_l2868_286832


namespace mcdonald_farm_production_l2868_286890

/-- Calculates the total number of eggs needed in a month for Mcdonald's farm --/
def monthly_egg_production (saly_weekly : ℕ) (ben_weekly : ℕ) (weeks_in_month : ℕ) : ℕ :=
  let ked_weekly := ben_weekly / 2
  let total_weekly := saly_weekly + ben_weekly + ked_weekly
  total_weekly * weeks_in_month

/-- Proves that Mcdonald's farm should produce 124 eggs in a month --/
theorem mcdonald_farm_production : monthly_egg_production 10 14 4 = 124 := by
  sorry

#eval monthly_egg_production 10 14 4

end mcdonald_farm_production_l2868_286890


namespace smallest_common_multiple_of_6_and_15_l2868_286817

theorem smallest_common_multiple_of_6_and_15 :
  ∃ a : ℕ+, (∀ b : ℕ+, (6 ∣ b) ∧ (15 ∣ b) → a ≤ b) ∧ (6 ∣ a) ∧ (15 ∣ a) ∧ a = 30 :=
sorry

end smallest_common_multiple_of_6_and_15_l2868_286817


namespace circle_area_proof_l2868_286800

/-- The area of a circle with center at (-5, 3) and touching the point (7, -4) is 193π. -/
theorem circle_area_proof : 
  let center : ℝ × ℝ := (-5, 3)
  let point : ℝ × ℝ := (7, -4)
  let radius : ℝ := Real.sqrt ((center.1 - point.1)^2 + (center.2 - point.2)^2)
  let area : ℝ := π * radius^2
  area = 193 * π := by sorry

end circle_area_proof_l2868_286800


namespace sum_ten_smallest_multiples_of_eight_l2868_286863

theorem sum_ten_smallest_multiples_of_eight : 
  (Finset.range 10).sum (fun i => 8 * (i + 1)) = 440 := by
  sorry

end sum_ten_smallest_multiples_of_eight_l2868_286863


namespace circle_distance_problem_l2868_286809

theorem circle_distance_problem (r₁ r₂ d : ℝ) (A B C : ℝ × ℝ) :
  r₁ = 13 →
  r₂ = 30 →
  d = 41 →
  let O₁ : ℝ × ℝ := (0, 0)
  let O₂ : ℝ × ℝ := (d, 0)
  (A.1 - O₂.1)^2 + A.2^2 = r₁^2 →
  A.1 > r₂ →
  (B.1 - O₂.1)^2 + B.2^2 = r₁^2 →
  (C.1 - O₁.1)^2 + C.2^2 = r₂^2 →
  B = ((A.1 + C.1) / 2, (A.2 + C.2) / 2) →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = 12^2 * 13 :=
by sorry

end circle_distance_problem_l2868_286809


namespace arithmetic_sequence_seventh_term_l2868_286871

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_first : a 1 = 3) 
  (h_third : a 3 = 5) : 
  a 7 = 9 := by
sorry

end arithmetic_sequence_seventh_term_l2868_286871


namespace impossible_table_fill_l2868_286803

/-- A type representing a 6x6 table of integers -/
def Table : Type := Fin 6 → Fin 6 → ℤ

/-- Predicate to check if all numbers in the table are distinct -/
def all_distinct (t : Table) : Prop :=
  ∀ i j i' j', (i ≠ i' ∨ j ≠ j') → t i j ≠ t i' j'

/-- Predicate to check if the sum of numbers in a 1x5 rectangle is valid -/
def valid_sum (t : Table) (i j : Fin 6) (horizontal : Bool) : Prop :=
  let sum := if horizontal then
               (Finset.range 5).sum (fun k => t i (j + k))
             else
               (Finset.range 5).sum (fun k => t (i + k) j)
  sum = 2022 ∨ sum = 2023

/-- Predicate to check if all 1x5 rectangles have valid sums -/
def all_valid_sums (t : Table) : Prop :=
  ∀ i j, (j.val + 5 ≤ 6 → valid_sum t i j true) ∧
         (i.val + 5 ≤ 6 → valid_sum t i j false)

/-- Theorem stating that it's impossible to fill the table satisfying all conditions -/
theorem impossible_table_fill : ¬ ∃ t : Table, all_distinct t ∧ all_valid_sums t := by
  sorry

end impossible_table_fill_l2868_286803


namespace unique_max_f_and_sum_of_digits_l2868_286884

/-- Number of positive integer divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- Function f(n) = d(n) / n^(1/4) -/
noncomputable def f (n : ℕ+) : ℝ := (d n : ℝ) / n.val ^ (1/4 : ℝ)

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem unique_max_f_and_sum_of_digits :
  ∃! N : ℕ+, (∀ n : ℕ+, n ≠ N → f N > f n) ∧ sum_of_digits N.val = 18 := by sorry

end unique_max_f_and_sum_of_digits_l2868_286884


namespace quadratic_inequality_condition_l2868_286877

theorem quadratic_inequality_condition (a : ℝ) :
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
by sorry

end quadratic_inequality_condition_l2868_286877


namespace sophie_donut_purchase_l2868_286846

/-- Calculates the total cost and remaining donuts for Sophie's purchase --/
theorem sophie_donut_purchase (budget : ℕ) (box_cost : ℕ) (discount_rate : ℚ) 
  (boxes_bought : ℕ) (donuts_per_box : ℕ) (boxes_given : ℕ) (donuts_given : ℕ) :
  budget = 50 ∧ 
  box_cost = 12 ∧ 
  discount_rate = 1/10 ∧ 
  boxes_bought = 4 ∧ 
  donuts_per_box = 12 ∧ 
  boxes_given = 1 ∧ 
  donuts_given = 6 →
  ∃ (total_cost : ℚ) (donuts_left : ℕ),
    total_cost = 43.2 ∧ 
    donuts_left = 30 :=
by sorry

end sophie_donut_purchase_l2868_286846


namespace polynomial_division_remainder_l2868_286811

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  X^4 + 1 = (X^2 - 3*X + 5) * q + r ∧
  r.degree < (X^2 - 3*X + 5).degree ∧
  r = -3*X - 19 := by
  sorry

end polynomial_division_remainder_l2868_286811


namespace max_sum_of_distances_squared_l2868_286897

def A : ℝ × ℝ := (-2, -2)
def B : ℝ × ℝ := (-2, 6)
def C : ℝ × ℝ := (4, -2)

def distance_squared (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def sum_of_distances_squared (P : ℝ × ℝ) : ℝ :=
  distance_squared P A + distance_squared P B + distance_squared P C

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 4

theorem max_sum_of_distances_squared :
  ∃ (max : ℝ), max = 88 ∧
  ∀ (P : ℝ × ℝ), on_circle P → sum_of_distances_squared P ≤ max :=
sorry

end max_sum_of_distances_squared_l2868_286897


namespace sqrt_product_quotient_equals_six_sqrt_three_l2868_286880

theorem sqrt_product_quotient_equals_six_sqrt_three :
  (Real.sqrt 12 * Real.sqrt 27) / Real.sqrt 3 = 6 * Real.sqrt 3 := by
  sorry

end sqrt_product_quotient_equals_six_sqrt_three_l2868_286880


namespace min_keys_for_scenario_min_keys_sufficiency_l2868_286851

/-- Represents the minimum number of keys required for the given scenario -/
def min_keys (n_drivers : ℕ) (n_cars : ℕ) : ℕ :=
  n_cars + (n_drivers - n_cars) * n_cars

/-- Theorem stating the minimum number of keys required for the given scenario -/
theorem min_keys_for_scenario :
  min_keys 50 40 = 440 :=
sorry

/-- Theorem proving that the minimum number of keys allows any subset of drivers to operate all cars -/
theorem min_keys_sufficiency (n_drivers : ℕ) (n_cars : ℕ) 
  (h1 : n_drivers ≥ n_cars) (h2 : n_cars > 0) :
  ∀ (subset : Finset (Fin n_drivers)), 
    subset.card = n_cars → 
    ∃ (key_distribution : Fin n_drivers → Finset (Fin n_cars)),
      (∀ i, (key_distribution i).card ≤ min_keys n_drivers n_cars) ∧
      (∀ i ∈ subset, (key_distribution i).card = n_cars) :=
sorry

end min_keys_for_scenario_min_keys_sufficiency_l2868_286851


namespace product_of_solutions_l2868_286896

theorem product_of_solutions (x₁ x₂ : ℝ) (h₁ : x₁ * Real.exp x₁ = Real.exp 2) (h₂ : x₂ * Real.log x₂ = Real.exp 2) :
  x₁ * x₂ = Real.exp 2 := by
  sorry

end product_of_solutions_l2868_286896


namespace remainder_theorem_l2868_286841

theorem remainder_theorem (n : ℤ) : n % 5 = 3 → (4 * n - 9) % 5 = 3 := by
  sorry

end remainder_theorem_l2868_286841


namespace cheese_slices_left_l2868_286807

/-- Represents the number of slices in each pizza -/
def slices_per_pizza : ℕ := 16

/-- Represents the total number of people -/
def total_people : ℕ := 4

/-- Represents the number of pepperoni slices left -/
def pepperoni_left : ℕ := 1

/-- Represents the number of people who eat both types of pizza -/
def people_eating_both : ℕ := 3

/-- Calculates the total number of slices eaten by the person who only eats pepperoni -/
def pepperoni_only_eater_slices : ℕ := slices_per_pizza - (pepperoni_left + 1)

/-- Calculates the number of pepperoni slices eaten by people who eat both types -/
def pepperoni_eaten_by_both : ℕ := slices_per_pizza - pepperoni_only_eater_slices - pepperoni_left

/-- Theorem stating that the number of cheese slices left is 7 -/
theorem cheese_slices_left : 
  slices_per_pizza - (pepperoni_eaten_by_both / people_eating_both * people_eating_both) = 7 := by
  sorry

end cheese_slices_left_l2868_286807


namespace yellow_peaches_count_l2868_286847

/-- The number of yellow peaches in a basket, given the number of green peaches
    and the difference between green and yellow peaches. -/
def yellow_peaches (green : ℕ) (difference : ℕ) : ℕ :=
  green - difference

/-- Theorem stating that the number of yellow peaches is 6, given the conditions. -/
theorem yellow_peaches_count : yellow_peaches 14 8 = 6 := by
  sorry

end yellow_peaches_count_l2868_286847


namespace same_color_probability_l2868_286891

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 5

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 6

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 7

/-- The number of green marbles in the bag -/
def green_marbles : ℕ := 2

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles + green_marbles

/-- The number of marbles drawn -/
def drawn_marbles : ℕ := 4

/-- The probability of drawing four marbles of the same color without replacement -/
theorem same_color_probability : 
  (red_marbles * (red_marbles - 1) * (red_marbles - 2) * (red_marbles - 3) +
   white_marbles * (white_marbles - 1) * (white_marbles - 2) * (white_marbles - 3) +
   blue_marbles * (blue_marbles - 1) * (blue_marbles - 2) * (blue_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)) =
  55 / 4855 := by
  sorry

end same_color_probability_l2868_286891


namespace henry_workout_convergence_l2868_286875

theorem henry_workout_convergence (gym_distance : ℝ) (walk_fraction : ℝ) : 
  gym_distance = 3 →
  walk_fraction = 2/3 →
  ∃ (A B : ℝ), 
    (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
      |A - (gym_distance - walk_fraction^n * gym_distance)| < ε ∧
      |B - (walk_fraction * gym_distance - walk_fraction^n * (walk_fraction * gym_distance - gym_distance))| < ε) ∧
    |A - B| = 3/2 :=
by sorry

end henry_workout_convergence_l2868_286875


namespace last_digits_l2868_286825

theorem last_digits (n : ℕ) : 
  (6^811 : ℕ) % 10 = 6 ∧ 
  (2^1000 : ℕ) % 10 = 6 ∧ 
  (3^999 : ℕ) % 10 = 7 := by
  sorry

end last_digits_l2868_286825


namespace key_arrangement_theorem_l2868_286842

/-- The number of permutations of n elements -/
def totalPermutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of permutations of n elements with exactly one cycle -/
def onePermutation (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of permutations of 10 elements where at least two cycles are present -/
def atLeastTwoCycles : ℕ := totalPermutations 10 - onePermutation 10

/-- The number of permutations of 10 elements with exactly two cycles -/
def exactlyTwoCycles : ℕ :=
  choose 10 1 * Nat.factorial 8 +
  choose 10 2 * Nat.factorial 7 +
  choose 10 3 * Nat.factorial 2 * Nat.factorial 6 +
  choose 10 4 * Nat.factorial 3 * Nat.factorial 5 +
  (choose 10 5 * Nat.factorial 4 * Nat.factorial 4) / 2

theorem key_arrangement_theorem :
  atLeastTwoCycles = 9 * Nat.factorial 9 ∧ exactlyTwoCycles = 1024576 := by sorry

end key_arrangement_theorem_l2868_286842


namespace advisory_panel_combinations_l2868_286840

theorem advisory_panel_combinations (n : ℕ) (k : ℕ) : n = 30 → k = 5 → Nat.choose n k = 142506 := by
  sorry

end advisory_panel_combinations_l2868_286840


namespace chemical_plant_max_profit_l2868_286808

/-- Represents the annual profit function for a chemical plant. -/
def L (x a : ℝ) : ℝ := (x - 3 - a) * (11 - x)^2

/-- Proves the maximum annual profit for the chemical plant under given conditions. -/
theorem chemical_plant_max_profit :
  ∀ (a : ℝ), 1 ≤ a → a ≤ 3 →
    (∀ (x : ℝ), 7 ≤ x → x ≤ 10 →
      (1 ≤ a ∧ a ≤ 2 →
        L x a ≤ 16 * (4 - a) ∧
        L 7 a = 16 * (4 - a)) ∧
      (2 < a →
        L x a ≤ (8 - a)^3 ∧
        L ((17 + 2*a)/3) a = (8 - a)^3)) :=
by sorry

end chemical_plant_max_profit_l2868_286808


namespace sqrt_x_minus_2_real_range_l2868_286887

theorem sqrt_x_minus_2_real_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) → x ≥ 2 := by
sorry

end sqrt_x_minus_2_real_range_l2868_286887


namespace right_triangle_arctan_sum_l2868_286857

theorem right_triangle_arctan_sum (d e f : ℝ) (h : d^2 + e^2 = f^2) :
  Real.arctan (d / (e + 2*f)) + Real.arctan (e / (d + 2*f)) = π/4 := by
  sorry

end right_triangle_arctan_sum_l2868_286857


namespace problem_statement_l2868_286853

theorem problem_statement (x : ℝ) 
  (h : (4:ℝ)^(2*x) + (2:ℝ)^(-x) + 1 = (129 + 8*Real.sqrt 2) * ((4:ℝ)^x + (2:ℝ)^(-x) - (2:ℝ)^x)) :
  10 * x = 35 := by
  sorry

end problem_statement_l2868_286853


namespace initial_ratio_of_partners_to_associates_l2868_286872

theorem initial_ratio_of_partners_to_associates 
  (partners : ℕ) 
  (associates : ℕ) 
  (h1 : partners = 18) 
  (h2 : associates + 45 = 34 * partners) : 
  (2 : ℕ) / (63 : ℕ) = partners / associates :=
sorry

end initial_ratio_of_partners_to_associates_l2868_286872


namespace complex_magnitude_l2868_286898

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I) = 2) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l2868_286898


namespace three_digit_numbers_34_times_sum_of_digits_l2868_286886

theorem three_digit_numbers_34_times_sum_of_digits : 
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ n = 34 * (n / 100 + (n / 10 % 10) + (n % 10))} = 
  {102, 204, 306, 408} := by
sorry

end three_digit_numbers_34_times_sum_of_digits_l2868_286886


namespace tangent_slope_at_point_A_l2868_286838

-- Define the function f(x) = x^2 + 3x
def f (x : ℝ) : ℝ := x^2 + 3*x

-- State the theorem
theorem tangent_slope_at_point_A :
  -- The derivative of f at x = 1 is equal to 5
  (deriv f) 1 = 5 :=
sorry

end tangent_slope_at_point_A_l2868_286838


namespace min_value_expression_min_value_achieved_l2868_286820

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (5 * z) / (3 * x + y) + (5 * x) / (y + 3 * z) + (2 * y) / (x + z) ≥ 2 :=
by sorry

theorem min_value_achieved (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (5 * c) / (3 * a + b) + (5 * a) / (b + 3 * c) + (2 * b) / (a + c) = 2) :=
by sorry

end min_value_expression_min_value_achieved_l2868_286820


namespace cos_three_halves_pi_l2868_286881

theorem cos_three_halves_pi : Real.cos (3 * π / 2) = 0 := by
  sorry

end cos_three_halves_pi_l2868_286881


namespace power_of_two_l2868_286824

theorem power_of_two (k : ℕ) (h : 2^k = 4) : 2^(3*k) = 64 := by
  sorry

end power_of_two_l2868_286824


namespace fred_cards_after_purchase_l2868_286826

/-- The number of baseball cards Fred has after Melanie's purchase -/
def fred_remaining_cards (initial : ℕ) (bought : ℕ) : ℕ :=
  initial - bought

/-- Theorem: Fred has 2 baseball cards left after Melanie's purchase -/
theorem fred_cards_after_purchase :
  fred_remaining_cards 5 3 = 2 := by
  sorry

end fred_cards_after_purchase_l2868_286826


namespace socorro_training_time_l2868_286899

/-- Calculates the total training time in hours given daily training times and number of days -/
def total_training_time (mult_time : ℕ) (div_time : ℕ) (days : ℕ) (mins_per_hour : ℕ) : ℚ :=
  (mult_time + div_time) * days / mins_per_hour

/-- Proves that Socorro's total training time is 5 hours -/
theorem socorro_training_time :
  total_training_time 10 20 10 60 = 5 := by
  sorry

end socorro_training_time_l2868_286899


namespace shaded_cubes_count_total_cubes_count_face_size_edge_size_l2868_286894

/-- Represents a large cube constructed from smaller cubes -/
structure LargeCube where
  size : Nat
  total_cubes : Nat
  shaded_cubes : Nat

/-- Defines the properties of our specific large cube -/
def our_cube : LargeCube :=
  { size := 4
  , total_cubes := 64
  , shaded_cubes := 28 }

/-- Calculates the number of cubes on one face of the large cube -/
def face_cubes (c : LargeCube) : Nat :=
  c.size * c.size

/-- Calculates the number of cubes along one edge of the large cube -/
def edge_cubes (c : LargeCube) : Nat :=
  c.size

/-- Calculates the number of corner cubes in the large cube -/
def corner_cubes : Nat := 8

/-- Theorem stating that the number of shaded cubes in our specific cube is 28 -/
theorem shaded_cubes_count (c : LargeCube) (h1 : c = our_cube) :
  c.shaded_cubes = 28 := by
  sorry

/-- Theorem stating that the total number of smaller cubes is 64 -/
theorem total_cubes_count (c : LargeCube) (h1 : c = our_cube) :
  c.total_cubes = 64 := by
  sorry

/-- Theorem stating that the size of each face is 4x4 -/
theorem face_size (c : LargeCube) (h1 : c = our_cube) :
  face_cubes c = 16 := by
  sorry

/-- Theorem stating that each edge has 4 cubes -/
theorem edge_size (c : LargeCube) (h1 : c = our_cube) :
  edge_cubes c = 4 := by
  sorry

end shaded_cubes_count_total_cubes_count_face_size_edge_size_l2868_286894


namespace stream_speed_l2868_286866

/-- Given Julie's rowing distances and times, prove that the speed of the stream is 5 km/h -/
theorem stream_speed (v_j v_s : ℝ) 
  (h1 : 32 / (v_j - v_s) = 4)  -- Upstream equation
  (h2 : 72 / (v_j + v_s) = 4)  -- Downstream equation
  : v_s = 5 := by
  sorry

end stream_speed_l2868_286866


namespace x_eq_one_sufficient_not_necessary_l2868_286888

theorem x_eq_one_sufficient_not_necessary :
  (∃ x : ℝ, x ^ 2 = 1 ∧ x ≠ 1) ∧
  (∀ x : ℝ, x = 1 → x ^ 2 = 1) :=
by sorry

end x_eq_one_sufficient_not_necessary_l2868_286888


namespace circle_radii_order_l2868_286845

theorem circle_radii_order (rA rB rC : ℝ) : 
  rA = Real.sqrt 16 →
  π * rB^2 = 16 * π →
  2 * π * rC = 10 * π →
  rA ≤ rB ∧ rB ≤ rC :=
by sorry

end circle_radii_order_l2868_286845


namespace cos_squared_plus_sin_double_l2868_286878

open Real

theorem cos_squared_plus_sin_double (α : ℝ) (h : tan α = 2) : 
  cos α ^ 2 + sin (2 * α) = 1 := by
  sorry

end cos_squared_plus_sin_double_l2868_286878


namespace tangent_parallel_point_l2868_286821

theorem tangent_parallel_point (x y : ℝ) : 
  y = x^4 - x →                           -- Curve equation
  (4 * x^3 - 1 : ℝ) = 3 →                 -- Tangent slope equals 3
  (x = 1 ∧ y = 0) :=                      -- Coordinates of point P
by sorry

end tangent_parallel_point_l2868_286821


namespace hyperbola_eccentricity_l2868_286830

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (A B D : ℝ × ℝ) (c : ℝ),
    -- Right focus of the hyperbola
    c > 0 ∧
    -- Equation of the hyperbola
    (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ (x, y) ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}) ∧
    -- A and B are on the hyperbola and on a line perpendicular to x-axis through the right focus
    A ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1} ∧
    B ∈ {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1} ∧
    A.1 = c ∧ B.1 = c ∧
    -- D is on the imaginary axis
    D = (0, b) ∧
    -- ABD is a right-angled triangle
    (A.1 - D.1) * (B.1 - D.1) + (A.2 - D.2) * (B.2 - D.2) = 0 →
    -- The eccentricity is either √2 or √(2 + √2)
    c / a = Real.sqrt 2 ∨ c / a = Real.sqrt (2 + Real.sqrt 2) :=
by sorry

end hyperbola_eccentricity_l2868_286830


namespace cube_volume_in_pyramid_l2868_286892

/-- A tetrahedral pyramid with an equilateral triangular base -/
structure TetrahedralPyramid where
  base_side_length : ℝ
  lateral_faces_equilateral : Prop

/-- A cube placed inside a tetrahedral pyramid -/
structure CubeInPyramid where
  pyramid : TetrahedralPyramid
  vertex_at_centroid : Prop
  edges_touch_midpoints : Prop

/-- The volume of a cube -/
def cube_volume (side_length : ℝ) : ℝ := side_length ^ 3

theorem cube_volume_in_pyramid (c : CubeInPyramid) : 
  c.pyramid.base_side_length = 3 → cube_volume (c.pyramid.base_side_length / 3) = 1 := by
  sorry

end cube_volume_in_pyramid_l2868_286892


namespace ceiling_sum_sqrt_l2868_286810

theorem ceiling_sum_sqrt : ⌈Real.sqrt 50⌉ + ⌈Real.sqrt 150⌉ + ⌈Real.sqrt 250⌉ = 37 := by
  sorry

end ceiling_sum_sqrt_l2868_286810


namespace angle_460_in_second_quadrant_l2868_286843

/-- An angle is in the second quadrant if it's between 90° and 180° in its standard position -/
def in_second_quadrant (angle : ℝ) : Prop :=
  let standard_angle := angle % 360
  90 < standard_angle ∧ standard_angle ≤ 180

/-- 460° is in the second quadrant -/
theorem angle_460_in_second_quadrant : in_second_quadrant 460 := by
  sorry

end angle_460_in_second_quadrant_l2868_286843


namespace bottle_production_l2868_286839

/-- Given that 6 identical machines produce 24 bottles per minute at a constant rate,
    prove that 10 such machines will produce 160 bottles in 4 minutes. -/
theorem bottle_production 
  (rate : ℕ) -- Production rate per machine per minute
  (h1 : 6 * rate = 24) -- 6 machines produce 24 bottles per minute
  : 10 * rate * 4 = 160 := by
  sorry

end bottle_production_l2868_286839


namespace parabola_coefficient_b_l2868_286802

/-- Given a parabola y = ax^2 + bx + c with vertex (q, q+1) and y-intercept (0, -2q-1),
    where q ≠ -1/2, prove that b = 6 + 4/q -/
theorem parabola_coefficient_b (a b c q : ℝ) (h : q ≠ -1/2) :
  (∀ x y, y = a * x^2 + b * x + c) →
  (q + 1 = a * q^2 + b * q + c) →
  (-2 * q - 1 = c) →
  b = 6 + 4 / q := by
sorry

end parabola_coefficient_b_l2868_286802


namespace troy_straw_distribution_l2868_286879

/-- Given the conditions of Troy's straw distribution problem, prove that
    the number of straws fed to adult pigs is 120. -/
theorem troy_straw_distribution
  (total_straws : ℕ)
  (num_piglets : ℕ)
  (straws_per_piglet : ℕ)
  (h1 : total_straws = 300)
  (h2 : num_piglets = 20)
  (h3 : straws_per_piglet = 6)
  (h4 : ∃ (x : ℕ), x + x ≤ total_straws ∧ x = num_piglets * straws_per_piglet) :
  ∃ (x : ℕ), x = 120 ∧ x + x ≤ total_straws ∧ x = num_piglets * straws_per_piglet :=
sorry

end troy_straw_distribution_l2868_286879


namespace notebook_and_pen_prices_l2868_286834

def notebook_price : ℝ := 12
def pen_price : ℝ := 6

theorem notebook_and_pen_prices :
  (2 * notebook_price + pen_price = 30) ∧
  (notebook_price = 2 * pen_price) :=
by sorry

end notebook_and_pen_prices_l2868_286834


namespace distance_between_circle_centers_l2868_286860

theorem distance_between_circle_centers (a b c : ℝ) (ha : a = 7) (hb : b = 8) (hc : c = 9) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let R := (a * b * c) / (4 * A)
  let r := A / s
  let cos_C := (a^2 + b^2 - c^2) / (2 * a * b)
  let sin_C := Real.sqrt (1 - cos_C^2)
  let O₁O₂ := Real.sqrt (R^2 + 2 * R * r * cos_C + r^2)
  ∃ ε > 0, abs (O₁O₂ - 5.75) < ε :=
sorry

end distance_between_circle_centers_l2868_286860


namespace minimum_guests_with_both_l2868_286827

theorem minimum_guests_with_both (total : ℕ) 
  (sunglasses : ℕ) (wristbands : ℕ) (both : ℕ) : 
  (3 : ℚ) / 7 * total = sunglasses →
  (4 : ℚ) / 9 * total = wristbands →
  total = sunglasses + wristbands - both →
  total ≥ 63 →
  both ≥ 8 :=
sorry

end minimum_guests_with_both_l2868_286827


namespace triangle_center_distance_inequality_l2868_286835

/-- Given a triangle with circumradius R, inradius r, and distance d between
    its circumcenter and centroid, prove that d^2 ≤ R(R - 2r) -/
theorem triangle_center_distance_inequality 
  (R r d : ℝ) 
  (h_R : R > 0) 
  (h_r : r > 0) 
  (h_d : d ≥ 0) 
  (h_circumradius : R = circumradius_of_triangle) 
  (h_inradius : r = inradius_of_triangle) 
  (h_distance : d = distance_between_circumcenter_and_centroid) : 
  d^2 ≤ R * (R - 2*r) := by
sorry

end triangle_center_distance_inequality_l2868_286835


namespace race_head_start_l2868_286837

theorem race_head_start (v_a v_b L : ℝ) (h : v_a = (16 / 15) * v_b) :
  (L / v_a = (L - (L / 16)) / v_b) → (L / 16 : ℝ) = L - (L - (L / 16)) := by
  sorry

end race_head_start_l2868_286837


namespace cube_root_of_square_64_l2868_286865

theorem cube_root_of_square_64 (x : ℝ) (h : x^2 = 64) :
  ∃ y, y^3 = x ∧ (y = 2 ∨ y = -2) := by sorry

end cube_root_of_square_64_l2868_286865


namespace students_per_row_l2868_286862

theorem students_per_row (total_students : ℕ) (rows : ℕ) (leftover : ℕ) 
  (h1 : total_students = 45)
  (h2 : rows = 11)
  (h3 : leftover = 1)
  (h4 : total_students = rows * (total_students / rows) + leftover) :
  total_students / rows = 4 := by
  sorry

end students_per_row_l2868_286862


namespace right_triangle_side_length_l2868_286818

theorem right_triangle_side_length (Q R S : ℝ) (cosR : ℝ) (RS : ℝ) :
  cosR = 3 / 5 →
  RS = 10 →
  (Q - R) * (S - R) = 0 →  -- This represents the right angle at R
  (Q - S) * (Q - S) = 8 * 8 :=
by sorry

end right_triangle_side_length_l2868_286818


namespace husband_saves_225_monthly_l2868_286844

/-- Represents the savings and investment scenario of a married couple -/
structure SavingsScenario where
  wife_weekly_savings : ℕ
  savings_period_months : ℕ
  stock_price : ℕ
  stocks_bought : ℕ

/-- Calculates the husband's monthly savings based on the given scenario -/
def husband_monthly_savings (scenario : SavingsScenario) : ℕ :=
  let total_savings := 2 * scenario.stock_price * scenario.stocks_bought
  let wife_total_savings := scenario.wife_weekly_savings * 4 * scenario.savings_period_months
  let husband_total_savings := total_savings - wife_total_savings
  husband_total_savings / scenario.savings_period_months

/-- Theorem stating that given the specific scenario, the husband's monthly savings is $225 -/
theorem husband_saves_225_monthly (scenario : SavingsScenario) 
  (h1 : scenario.wife_weekly_savings = 100)
  (h2 : scenario.savings_period_months = 4)
  (h3 : scenario.stock_price = 50)
  (h4 : scenario.stocks_bought = 25) :
  husband_monthly_savings scenario = 225 := by
  sorry

end husband_saves_225_monthly_l2868_286844


namespace smallest_three_digit_congruence_l2868_286885

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    (n ≥ 100 ∧ n < 1000) ∧ 
    (75 * n) % 345 = 225 ∧
    (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ (75 * m) % 345 = 225 → m ≥ n) ∧
    n = 118 := by
  sorry

end smallest_three_digit_congruence_l2868_286885


namespace train_platform_ratio_l2868_286848

/-- Given a train of length L traveling at constant velocity v,
    if it passes a pole in time t and a platform in time 4t,
    then the ratio of the platform length P to the train length L is 3:1 -/
theorem train_platform_ratio
  (L : ℝ) -- Length of the train
  (v : ℝ) -- Velocity of the train
  (t : ℝ) -- Time to pass the pole
  (P : ℝ) -- Length of the platform
  (h1 : v > 0) -- Velocity is positive
  (h2 : L > 0) -- Train length is positive
  (h3 : t > 0) -- Time is positive
  (h4 : v = L / t) -- Velocity equation for passing the pole
  (h5 : v = (L + P) / (4 * t)) -- Velocity equation for passing the platform
  : P / L = 3 := by
  sorry

end train_platform_ratio_l2868_286848


namespace equation_equivalence_l2868_286876

theorem equation_equivalence (x : ℝ) : 
  (4 * x^2 + 1 = (2*x + 1)^2) ∨ (4 * x^2 + 1 = (2*x - 1)^2) ↔ (4*x = 0 ∨ -4*x = 0) :=
sorry

end equation_equivalence_l2868_286876


namespace triangle_with_sum_of_two_angles_less_than_third_is_obtuse_l2868_286805

theorem triangle_with_sum_of_two_angles_less_than_third_is_obtuse 
  (α β γ : Real) 
  (triangle_angles : α + β + γ = 180) 
  (angle_sum_condition : α + β < γ) : 
  γ > 90 := by
sorry

end triangle_with_sum_of_two_angles_less_than_third_is_obtuse_l2868_286805


namespace triarc_area_theorem_l2868_286850

/-- Represents a region enclosed by three circular arcs --/
structure TriarcRegion where
  radius : ℝ
  centralAngle : ℝ

/-- Calculates the area of the triarc region --/
def triarcArea (region : TriarcRegion) : ℝ := sorry

/-- Theorem stating the area of the specific triarc region --/
theorem triarc_area_theorem (region : TriarcRegion) 
  (h_radius : region.radius = 6)
  (h_angle : region.centralAngle = π / 2) :
  ∃ (p q r : ℝ), 
    triarcArea region = p * Real.sqrt q + r * π ∧ 
    p + q + r = 7.5 := by sorry

end triarc_area_theorem_l2868_286850


namespace y_intercept_of_parallel_line_l2868_286882

/-- A line in the xy-plane represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Checks if two lines are parallel. -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Checks if a point lies on a line. -/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.y_intercept

/-- The given line y = -3x + 7 -/
def given_line : Line :=
  { slope := -3, y_intercept := 7 }

theorem y_intercept_of_parallel_line :
  ∀ (b : Line),
    are_parallel b given_line →
    point_on_line b 5 (-2) →
    b.y_intercept = 13 := by
  sorry

end y_intercept_of_parallel_line_l2868_286882


namespace simplify_and_evaluate_expression_l2868_286861

theorem simplify_and_evaluate_expression :
  let x := Real.cos (30 * π / 180)
  (x - (2 * x - 1) / x) / (x / (x - 1)) = Real.sqrt 3 / 2 - 1 := by
  sorry

end simplify_and_evaluate_expression_l2868_286861


namespace train_speed_calculation_l2868_286852

/-- Represents the speed and travel time of a train -/
structure Train where
  speed : ℝ
  time_after_meeting : ℝ

/-- Theorem stating the relationship between two trains meeting and their speeds -/
theorem train_speed_calculation (train_a train_b : Train) 
  (h1 : train_a.speed = 60)
  (h2 : train_a.time_after_meeting = 9)
  (h3 : train_b.time_after_meeting = 4) :
  train_b.speed = 135 := by
  sorry

end train_speed_calculation_l2868_286852


namespace abs_less_implies_sum_positive_l2868_286822

theorem abs_less_implies_sum_positive (a b : ℝ) : |a| < b → a + b > 0 := by
  sorry

end abs_less_implies_sum_positive_l2868_286822


namespace balance_rearrangements_l2868_286831

def word : String := "BALANCE"

def is_vowel (c : Char) : Bool :=
  c ∈ ['A', 'E', 'I', 'O', 'U']

def vowels : List Char :=
  word.toList.filter is_vowel

def consonants : List Char :=
  word.toList.filter (fun c => !is_vowel c)

def vowel_arrangements : ℕ :=
  Nat.factorial vowels.length / (Nat.factorial 2)  -- 2 is the count of repeated 'A's

def consonant_arrangements : ℕ :=
  Nat.factorial consonants.length

theorem balance_rearrangements :
  vowel_arrangements * consonant_arrangements = 72 :=
sorry

end balance_rearrangements_l2868_286831


namespace least_sum_of_primes_l2868_286855

theorem least_sum_of_primes (p q : ℕ) : 
  Prime p → Prime q → 
  (∀ n : ℕ, n > 0 → (n^(3*p*q) - n) % (3*p*q) = 0) → 
  (∀ p' q' : ℕ, Prime p' → Prime q' → 
    (∀ n : ℕ, n > 0 → (n^(3*p'*q') - n) % (3*p'*q') = 0) → 
    p' + q' ≥ p + q) →
  p + q = 28 := by
sorry

end least_sum_of_primes_l2868_286855


namespace inverse_of_10_mod_1001_l2868_286867

theorem inverse_of_10_mod_1001 : ∃ x : ℕ, x ∈ Finset.range 1001 ∧ (10 * x) % 1001 = 1 :=
by
  use 901
  sorry

end inverse_of_10_mod_1001_l2868_286867


namespace eight_row_triangle_pieces_l2868_286823

/-- Calculates the sum of the first n positive integers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculates the number of rods in an n-row triangle -/
def total_rods (n : ℕ) : ℕ := 3 * triangular_number n

/-- Calculates the number of connectors in an n-row triangle -/
def total_connectors (n : ℕ) : ℕ := triangular_number (n + 1)

/-- Calculates the total number of pieces in an n-row triangle -/
def total_pieces (n : ℕ) : ℕ := total_rods n + total_connectors n

/-- Theorem: The total number of pieces in an eight-row triangle is 153 -/
theorem eight_row_triangle_pieces : total_pieces 8 = 153 := by
  sorry

end eight_row_triangle_pieces_l2868_286823


namespace deck_size_l2868_286804

theorem deck_size (r b : ℕ) : 
  r > 0 ∧ b > 0 → -- Ensure positive number of cards
  r / (r + b : ℚ) = 1 / 4 → -- Initial probability
  r / (r + b + 6 : ℚ) = 1 / 6 → -- Probability after adding 6 black cards
  r + b = 12 := by
sorry

end deck_size_l2868_286804


namespace ellipse_line_intersection_range_l2868_286816

/-- The range of b for which the ellipse C: x²/4 + y²/b = 1 always intersects with any line l: y = mx + 1 -/
theorem ellipse_line_intersection_range :
  ∀ (b : ℝ),
  (∀ (m : ℝ), ∃ (x y : ℝ), x^2/4 + y^2/b = 1 ∧ y = m*x + 1) →
  (b ∈ Set.Icc 1 4 ∪ Set.Ioi 4) :=
by sorry

end ellipse_line_intersection_range_l2868_286816


namespace ellipse_eccentricity_range_l2868_286828

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    where a line passing through its left vertex with slope k 
    intersects the ellipse at a point whose x-coordinate is the 
    distance from the center to the focus, prove that the 
    eccentricity e of the ellipse is between 1/2 and 2/3 
    when k is between 1/3 and 1/2. -/
theorem ellipse_eccentricity_range (a b : ℝ) (k : ℝ) 
  (h1 : a > b) (h2 : b > 0) 
  (h3 : 1/3 < k) (h4 : k < 1/2) :
  let e := Real.sqrt (1 - b^2 / a^2)
  ∃ (x y : ℝ), 
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    y = k * (x + a) ∧
    x = a * e ∧
    1/2 < e ∧ e < 2/3 :=
by sorry

end ellipse_eccentricity_range_l2868_286828


namespace division_of_25_by_4_l2868_286836

theorem division_of_25_by_4 : ∃ (q r : ℕ), 25 = 4 * q + r ∧ r < 4 ∧ q = 6 := by
  sorry

end division_of_25_by_4_l2868_286836


namespace harry_uses_whole_bag_l2868_286859

/-- The number of batches of cookies -/
def num_batches : ℕ := 3

/-- The number of chocolate chips per cookie -/
def chips_per_cookie : ℕ := 9

/-- The number of chips in a bag -/
def chips_per_bag : ℕ := 81

/-- The number of cookies in a batch -/
def cookies_per_batch : ℕ := 3

/-- The portion of the bag used for making the dough -/
def portion_used : ℚ := (num_batches * cookies_per_batch * chips_per_cookie) / chips_per_bag

theorem harry_uses_whole_bag : portion_used = 1 := by
  sorry

end harry_uses_whole_bag_l2868_286859


namespace no_real_m_for_reciprocal_sum_l2868_286870

theorem no_real_m_for_reciprocal_sum (m : ℝ) : ¬ (∃ x₁ x₂ : ℝ,
  (m * x₁^2 - 2*x₁ + m*(m^2 + 1) = 0) ∧
  (m * x₂^2 - 2*x₂ + m*(m^2 + 1) = 0) ∧
  (x₁ ≠ x₂) ∧
  (1/x₁ + 1/x₂ = m)) := by
  sorry

#check no_real_m_for_reciprocal_sum

end no_real_m_for_reciprocal_sum_l2868_286870


namespace transformation_composition_dilation_property_rotation_property_transformation_is_dilation_then_rotation_l2868_286864

def dilation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![2, 0; 0, 2]
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![0, -2; 2, 0]

theorem transformation_composition :
  rotation_matrix * dilation_matrix = transformation_matrix :=
by sorry

theorem dilation_property (v : Fin 2 → ℝ) :
  dilation_matrix.mulVec v = 2 • v :=
by sorry

theorem rotation_property (v : Fin 2 → ℝ) :
  rotation_matrix.mulVec v = ![- v 1, v 0] :=
by sorry

theorem transformation_is_dilation_then_rotation :
  ∀ v : Fin 2 → ℝ,
  transformation_matrix.mulVec v = rotation_matrix.mulVec (dilation_matrix.mulVec v) :=
by sorry

end transformation_composition_dilation_property_rotation_property_transformation_is_dilation_then_rotation_l2868_286864


namespace consecutive_odd_numbers_l2868_286873

theorem consecutive_odd_numbers (a b c d e : ℕ) : 
  (∃ k : ℕ, a = 2*k + 1) ∧ 
  b = a + 2 ∧ 
  c = b + 2 ∧ 
  d = c + 2 ∧ 
  e = d + 2 ∧ 
  a + c = 146 ∧ 
  e = 79 →
  a = 71 := by
sorry

end consecutive_odd_numbers_l2868_286873


namespace exists_valid_grid_l2868_286815

def is_valid_grid (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  (∀ i j, grid i j ≤ 25) ∧
  (∀ i j, grid i j > 0) ∧
  (∀ i₁ j₁ i₂ j₂, i₁ ≠ i₂ ∨ j₁ ≠ j₂ → grid i₁ j₁ ≠ grid i₂ j₂) ∧
  (∀ i j, i < 2 → (grid i j ∣ grid (i+1) j) ∨ (grid (i+1) j ∣ grid i j)) ∧
  (∀ i j, j < 2 → (grid i j ∣ grid i (j+1)) ∨ (grid i (j+1) ∣ grid i j))

theorem exists_valid_grid : ∃ (grid : Matrix (Fin 3) (Fin 3) ℕ), is_valid_grid grid := by
  sorry

end exists_valid_grid_l2868_286815


namespace triangle_vector_relation_l2868_286801

/-- Given a triangle ABC with point D such that BD = 2DC, prove that AD = (1/3)AB + (2/3)AC -/
theorem triangle_vector_relation (A B C D : EuclideanSpace ℝ (Fin 3)) :
  (B - D) = 2 • (D - C) →
  (A - D) = (1 / 3) • (A - B) + (2 / 3) • (A - C) := by
  sorry

end triangle_vector_relation_l2868_286801


namespace simplify_expression_1_simplify_expression_2_simplify_expression_3_l2868_286883

-- Define variables
variable (a b x y : ℝ)

-- Theorem 1
theorem simplify_expression_1 : 2*a - 3*b + a - 5*b = 3*a - 8*b := by sorry

-- Theorem 2
theorem simplify_expression_2 : (a^2 - 6*a) - 3*(a^2 - 2*a + 1) + 3 = -2*a^2 := by sorry

-- Theorem 3
theorem simplify_expression_3 : 4*(x^2*y - 2*x*y^2) - 3*(-x*y^2 + 2*x^2*y) = -2*x^2*y - 5*x*y^2 := by sorry

end simplify_expression_1_simplify_expression_2_simplify_expression_3_l2868_286883


namespace nickels_per_stack_l2868_286833

theorem nickels_per_stack (total_nickels : ℕ) (num_stacks : ℕ) 
  (h1 : total_nickels = 72) 
  (h2 : num_stacks = 9) : 
  total_nickels / num_stacks = 8 := by
  sorry

end nickels_per_stack_l2868_286833


namespace little_league_games_l2868_286854

theorem little_league_games (games_won : ℕ) (games_lost_difference : ℕ) : 
  games_won = 18 → games_lost_difference = 21 → games_won + (games_won + games_lost_difference) = 57 := by
  sorry

end little_league_games_l2868_286854


namespace mixed_oil_rate_l2868_286858

/-- Given two oils mixed together, calculate the rate of the mixed oil per litre -/
theorem mixed_oil_rate (volume1 volume2 rate1 rate2 : ℚ) 
  (h1 : volume1 = 10)
  (h2 : volume2 = 5)
  (h3 : rate1 = 40)
  (h4 : rate2 = 66) :
  (volume1 * rate1 + volume2 * rate2) / (volume1 + volume2) = 730 / 15 := by
  sorry

end mixed_oil_rate_l2868_286858


namespace faster_train_speed_l2868_286813

/-- The speed of the faster train when two trains cross each other --/
theorem faster_train_speed
  (train_length : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 100)
  (h2 : crossing_time = 8)
  (h3 : crossing_time > 0) :
  ∃ (v : ℝ), v > 0 ∧ 2 * v * crossing_time = 2 * train_length ∧ v = 25 / 3 :=
by sorry

end faster_train_speed_l2868_286813


namespace solution_value_l2868_286895

theorem solution_value (x a : ℝ) : x = 2 ∧ 2*x + 3*a = 10 → a = 2 := by
  sorry

end solution_value_l2868_286895


namespace investment_ratio_is_one_to_one_l2868_286806

-- Define the interest rates
def interest_rate_1 : ℝ := 0.05
def interest_rate_2 : ℝ := 0.06

-- Define the total interest earned
def total_interest : ℝ := 520

-- Define the investment amounts
def investment_1 : ℝ := 2000
def investment_2 : ℝ := 2000

-- Theorem statement
theorem investment_ratio_is_one_to_one :
  (investment_1 * interest_rate_1 + investment_2 * interest_rate_2 = total_interest) →
  (investment_1 / investment_2 = 1) :=
by
  sorry


end investment_ratio_is_one_to_one_l2868_286806


namespace plot_breadth_is_8_l2868_286812

/-- A rectangular plot with the given properties. -/
structure RectangularPlot where
  breadth : ℝ
  length : ℝ
  area_is_18_times_breadth : length * breadth = 18 * breadth
  length_breadth_difference : length - breadth = 10

/-- The breadth of the rectangular plot is 8 meters. -/
theorem plot_breadth_is_8 (plot : RectangularPlot) : plot.breadth = 8 := by
  sorry

end plot_breadth_is_8_l2868_286812
