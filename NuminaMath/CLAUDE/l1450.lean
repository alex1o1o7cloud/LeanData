import Mathlib

namespace oprah_car_collection_reduction_l1450_145091

def reduce_car_collection (initial_cars : ℕ) (target_cars : ℕ) (cars_given_per_year : ℕ) : ℕ :=
  (initial_cars - target_cars) / cars_given_per_year

theorem oprah_car_collection_reduction :
  reduce_car_collection 3500 500 50 = 60 := by
  sorry

end oprah_car_collection_reduction_l1450_145091


namespace remainder_of_Q_l1450_145070

-- Define the polynomial Q
variable (Q : ℝ → ℝ)

-- Define the conditions
axiom Q_div_21 : ∃ P₁ : ℝ → ℝ, ∀ x, Q x = (x - 21) * (P₁ x) + 105
axiom Q_div_105 : ∃ P₂ : ℝ → ℝ, ∀ x, Q x = (x - 105) * (P₂ x) + 21

-- Theorem statement
theorem remainder_of_Q : 
  ∃ P : ℝ → ℝ, ∀ x, Q x = (x - 21) * (x - 105) * (P x) + (-x + 126) := by
  sorry

end remainder_of_Q_l1450_145070


namespace soccer_points_for_win_l1450_145097

theorem soccer_points_for_win (total_games : ℕ) (wins : ℕ) (losses : ℕ) (total_points : ℕ)
  (h_total_games : total_games = 20)
  (h_wins : wins = 14)
  (h_losses : losses = 2)
  (h_total_points : total_points = 46)
  (h_games_balance : total_games = wins + losses + (total_games - wins - losses)) :
  ∃ (points_for_win : ℕ),
    points_for_win * wins + (total_games - wins - losses) = total_points ∧ 
    points_for_win = 3 := by
sorry

end soccer_points_for_win_l1450_145097


namespace set_operations_and_subset_condition_l1450_145014

def A : Set ℝ := {x | 2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | 3 * x - 7 ≥ 8 - 2 * x}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem set_operations_and_subset_condition (a : ℝ) :
  (A ∩ B = {x | 3 ≤ x ∧ x < 4}) ∧
  (A ∪ (C a ∪ B) = {x | x < 4}) ∧
  (A ⊆ C a → a ≥ 4) := by sorry

end set_operations_and_subset_condition_l1450_145014


namespace ab_plus_cd_eq_zero_l1450_145049

theorem ab_plus_cd_eq_zero 
  (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1) 
  (h3 : a*c + b*d = 0) : 
  a*b + c*d = 0 := by
sorry

end ab_plus_cd_eq_zero_l1450_145049


namespace mathematics_letter_probability_l1450_145016

theorem mathematics_letter_probability : 
  let alphabet_size : ℕ := 26
  let unique_letters : ℕ := 8
  let probability : ℚ := unique_letters / alphabet_size
  probability = 4 / 13 := by
sorry

end mathematics_letter_probability_l1450_145016


namespace peach_expense_l1450_145069

theorem peach_expense (total berries apples : ℝ) 
  (h_total : total = 34.72)
  (h_berries : berries = 11.08)
  (h_apples : apples = 14.33) :
  total - (berries + apples) = 9.31 := by sorry

end peach_expense_l1450_145069


namespace glass_bowls_problem_l1450_145089

/-- The number of glass bowls initially bought -/
def initial_bowls : ℕ := 139

/-- The cost per bowl in Rupees -/
def cost_per_bowl : ℚ := 13

/-- The selling price per bowl in Rupees -/
def selling_price : ℚ := 17

/-- The number of bowls sold -/
def bowls_sold : ℕ := 108

/-- The percentage gain -/
def percentage_gain : ℚ := 23.88663967611336

theorem glass_bowls_problem :
  (percentage_gain / 100 * (initial_bowls * cost_per_bowl) = 
   bowls_sold * selling_price - bowls_sold * cost_per_bowl) ∧
  (initial_bowls ≥ bowls_sold) := by
  sorry

end glass_bowls_problem_l1450_145089


namespace inequality_system_solution_l1450_145078

theorem inequality_system_solution (x : ℝ) : 
  (2 / (x - 1) - 3 / (x - 2) + 5 / (x - 3) - 2 / (x - 4) < 1 / 20) →
  (1 / (x - 2) > 1 / 5) →
  (x ∈ Set.Ioo 2 3) ∨ (x ∈ Set.Ioo 4 6) := by
  sorry

end inequality_system_solution_l1450_145078


namespace integer_between_bounds_l1450_145030

theorem integer_between_bounds (x : ℤ) :
  (-4.5 : ℝ) < (x : ℝ) ∧ (x : ℝ) < (-4 : ℝ) / 3 →
  x = -4 ∨ x = -3 ∨ x = -2 := by
sorry

end integer_between_bounds_l1450_145030


namespace beads_per_necklace_l1450_145021

/-- Given that Emily made 6 necklaces and used a total of 18 beads,
    prove that each necklace needs 3 beads. -/
theorem beads_per_necklace :
  let total_necklaces : ℕ := 6
  let total_beads : ℕ := 18
  total_beads / total_necklaces = 3 := by sorry

end beads_per_necklace_l1450_145021


namespace intersection_when_m_neg_one_subset_iff_m_leq_neg_two_l1450_145064

-- Define sets A and B
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m < x ∧ x < 1 - m}

-- Theorem 1: When m = -1, A ∩ B = {x | 1 < x < 2}
theorem intersection_when_m_neg_one :
  A ∩ B (-1) = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 2: A ⊆ B if and only if m ≤ -2
theorem subset_iff_m_leq_neg_two :
  ∀ m : ℝ, A ⊆ B m ↔ m ≤ -2 := by sorry

end intersection_when_m_neg_one_subset_iff_m_leq_neg_two_l1450_145064


namespace min_distance_line_ellipse_l1450_145024

noncomputable def minDistance : ℝ := (24 - 2 * Real.sqrt 41) / 5

/-- The minimum distance between a point on the line 4x + 3y = 24
    and a point on the ellipse (x²/8) + (y²/4) = 1 is (24 - 2√41) / 5 -/
theorem min_distance_line_ellipse :
  let line := {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 = 24}
  let ellipse := {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}
  ∃ (p₁ : ℝ × ℝ) (p₂ : ℝ × ℝ),
    p₁ ∈ line ∧ p₂ ∈ ellipse ∧
    ∀ (q₁ : ℝ × ℝ) (q₂ : ℝ × ℝ),
      q₁ ∈ line → q₂ ∈ ellipse →
      Real.sqrt ((q₁.1 - q₂.1)^2 + (q₁.2 - q₂.2)^2) ≥ minDistance :=
by sorry

end min_distance_line_ellipse_l1450_145024


namespace greatest_int_with_gcd_18_6_l1450_145063

theorem greatest_int_with_gcd_18_6 : 
  (∀ n : ℕ, n < 200 ∧ n > 174 → Nat.gcd n 18 ≠ 6) ∧ 
  Nat.gcd 174 18 = 6 := by
  sorry

end greatest_int_with_gcd_18_6_l1450_145063


namespace triangle_distance_inequality_l1450_145093

/-- Given a triangle ABC with an internal point P, prove the inequality involving distances from P to vertices and sides. -/
theorem triangle_distance_inequality 
  (x y z p q r : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) 
  (hp : p ≥ 0) (hq : q ≥ 0) (hr : r ≥ 0) 
  (h_internal : x + y > z ∧ y + z > x ∧ z + x > y) : 
  x * y * z ≥ (q + r) * (r + p) * (p + q) := by
  sorry

end triangle_distance_inequality_l1450_145093


namespace scale_multiplication_l1450_145002

theorem scale_multiplication (a b c : ℝ) (h : a * b = c) :
  (a / 100) * (b / 100) = c / 10000 := by
  sorry

end scale_multiplication_l1450_145002


namespace train_speed_calculation_l1450_145071

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 150 →
  crossing_time = 12.499 →
  ∃ (speed : ℝ), abs (speed - 72) < 0.1 ∧ speed = (train_length + bridge_length) / crossing_time * 3.6 := by
  sorry

#check train_speed_calculation

end train_speed_calculation_l1450_145071


namespace four_digit_number_with_two_schemes_l1450_145086

/-- Represents a division scheme for a four-digit number -/
structure DivisionScheme where
  divisor : Nat
  quotient : Nat
  remainder : Nat

/-- Checks if a number satisfies a given division scheme -/
def satisfiesScheme (n : Nat) (scheme : DivisionScheme) : Prop :=
  n / scheme.divisor = scheme.quotient ∧ n % scheme.divisor = scheme.remainder

/-- Theorem stating the existence of a four-digit number satisfying two division schemes -/
theorem four_digit_number_with_two_schemes :
  ∃ (n : Nat) (scheme1 scheme2 : DivisionScheme),
    1000 ≤ n ∧ n < 10000 ∧
    scheme1.divisor ≠ scheme2.divisor ∧
    scheme1.divisor < 10 ∧ scheme2.divisor < 10 ∧
    satisfiesScheme n scheme1 ∧
    satisfiesScheme n scheme2 := by
  sorry

#check four_digit_number_with_two_schemes

end four_digit_number_with_two_schemes_l1450_145086


namespace D_72_l1450_145041

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where the order of factors matters. -/
def D (n : ℕ+) : ℕ := sorry

/-- Theorem stating that D(72) = 35 -/
theorem D_72 : D 72 = 35 := by sorry

end D_72_l1450_145041


namespace twelfth_term_is_fifteen_l1450_145045

/-- An arithmetic sequence {a_n} with the given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 + a 4 + a 5 = 3 ∧
  a 8 = 8

/-- Theorem stating that for an arithmetic sequence satisfying the given conditions, 
    the 12th term is equal to 15 -/
theorem twelfth_term_is_fifteen (a : ℕ → ℚ) 
  (h : arithmetic_sequence a) : a 12 = 15 := by
  sorry

end twelfth_term_is_fifteen_l1450_145045


namespace range_of_m_for_two_distinct_zeros_l1450_145048

/-- A quadratic function with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + (m + 3)

/-- The discriminant of the quadratic function -/
def discriminant (m : ℝ) : ℝ := m^2 - 4*(m + 3)

/-- The theorem stating the range of m for which the quadratic function has two distinct zeros -/
theorem range_of_m_for_two_distinct_zeros :
  ∀ m : ℝ, (∃ x y : ℝ, x ≠ y ∧ f m x = 0 ∧ f m y = 0) ↔ m ∈ Set.Ioi 6 ∪ Set.Iio (-2) :=
sorry

end range_of_m_for_two_distinct_zeros_l1450_145048


namespace unique_lowest_degree_polynomial_l1450_145023

def f (n : ℕ) : ℕ := n^3 + 2*n^2 + n + 3

theorem unique_lowest_degree_polynomial :
  (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 21 ∧ f 3 = 51) ∧
  (∀ g : ℕ → ℕ, (g 0 = 3 ∧ g 1 = 7 ∧ g 2 = 21 ∧ g 3 = 51) →
    (∃ a b c d : ℕ, ∀ n, g n = a*n^3 + b*n^2 + c*n + d) →
    (∀ n, f n = g n)) :=
by sorry

end unique_lowest_degree_polynomial_l1450_145023


namespace units_digit_of_n_l1450_145033

/-- Given two natural numbers m and n, returns true if their product has a units digit of 1 -/
def product_has_units_digit_one (m n : ℕ) : Prop :=
  (m * n) % 10 = 1

/-- Given a natural number m, returns true if it has a units digit of 9 -/
def has_units_digit_nine (m : ℕ) : Prop :=
  m % 10 = 9

theorem units_digit_of_n (m n : ℕ) 
  (h1 : m * n = 11^4)
  (h2 : has_units_digit_nine m) :
  n % 10 = 9 := by
  sorry

end units_digit_of_n_l1450_145033


namespace inverse_function_condition_l1450_145079

noncomputable def g (a b c d x : ℝ) : ℝ := (2*a*x + b) / (2*c*x - d)

theorem inverse_function_condition (a b c d : ℝ) 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : ∀ x, x ∈ {x | 2*c*x - d ≠ 0} → g a b c d (g a b c d x) = x) : 
  2*a - d = 0 := by
  sorry

end inverse_function_condition_l1450_145079


namespace russian_tennis_pairing_probability_l1450_145054

theorem russian_tennis_pairing_probability :
  let total_players : ℕ := 10
  let russian_players : ℕ := 4
  let total_pairs : ℕ := total_players / 2
  let russian_pairs : ℕ := russian_players / 2
  let favorable_outcomes : ℕ := Nat.choose total_pairs russian_pairs
  let total_outcomes : ℕ := Nat.choose total_players russian_players
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 21 :=
by sorry

end russian_tennis_pairing_probability_l1450_145054


namespace circle_configuration_exists_l1450_145098

/-- A configuration of numbers in circles -/
structure CircleConfiguration where
  numbers : Fin 9 → ℕ
  consecutive : ∀ i j : Fin 9, i.val < j.val → numbers i < numbers j
  contains_six : ∃ i : Fin 9, numbers i = 6

/-- The lines connecting the circles -/
inductive Line
  | Line1 : Line
  | Line2 : Line
  | Line3 : Line
  | Line4 : Line
  | Line5 : Line
  | Line6 : Line

/-- The endpoints of each line -/
def lineEndpoints : Line → Fin 9 × Fin 9
  | Line.Line1 => (⟨0, by norm_num⟩, ⟨1, by norm_num⟩)
  | Line.Line2 => (⟨1, by norm_num⟩, ⟨2, by norm_num⟩)
  | Line.Line3 => (⟨2, by norm_num⟩, ⟨3, by norm_num⟩)
  | Line.Line4 => (⟨3, by norm_num⟩, ⟨4, by norm_num⟩)
  | Line.Line5 => (⟨4, by norm_num⟩, ⟨5, by norm_num⟩)
  | Line.Line6 => (⟨5, by norm_num⟩, ⟨0, by norm_num⟩)

/-- The sum of numbers on a line -/
def lineSum (config : CircleConfiguration) (line : Line) : ℕ :=
  let (a, b) := lineEndpoints line
  config.numbers a + config.numbers b

/-- The theorem statement -/
theorem circle_configuration_exists :
  ∃ config : CircleConfiguration, ∀ line : Line, lineSum config line = 23 := by
  sorry


end circle_configuration_exists_l1450_145098


namespace special_quadrilateral_is_kite_l1450_145026

/-- A quadrilateral with perpendicular diagonals, two adjacent equal sides, and one pair of equal opposite angles -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perp_diagonals : Bool
  /-- Two adjacent sides of the quadrilateral are equal -/
  adj_sides_equal : Bool
  /-- One pair of opposite angles are equal -/
  opp_angles_equal : Bool

/-- Definition of a kite -/
def is_kite (q : SpecialQuadrilateral) : Prop :=
  q.perp_diagonals ∧ q.adj_sides_equal

/-- Theorem stating that a quadrilateral with the given properties is a kite -/
theorem special_quadrilateral_is_kite (q : SpecialQuadrilateral) 
  (h1 : q.perp_diagonals = true) 
  (h2 : q.adj_sides_equal = true) 
  (h3 : q.opp_angles_equal = true) : 
  is_kite q :=
sorry

end special_quadrilateral_is_kite_l1450_145026


namespace max_x_minus_y_l1450_145025

theorem max_x_minus_y (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (prod_eq : x*y + y*z + z*x = 1) : 
  ∃ (max : ℝ), max = 2 / Real.sqrt 3 ∧ 
  ∀ (a b c : ℝ), a + b + c = 2 → a*b + b*c + c*a = 1 → 
  |a - b| ≤ max := by
sorry

end max_x_minus_y_l1450_145025


namespace farm_animals_l1450_145019

/-- Given a farm with chickens and pigs, prove the number of chickens. -/
theorem farm_animals (total_legs : ℕ) (num_pigs : ℕ) (num_chickens : ℕ) : 
  total_legs = 48 → num_pigs = 9 → 2 * num_chickens + 4 * num_pigs = total_legs → num_chickens = 6 := by
  sorry

end farm_animals_l1450_145019


namespace sum_220_is_5500_div_3_l1450_145009

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  /-- The first term of the progression -/
  a : ℚ
  /-- The common difference of the progression -/
  d : ℚ
  /-- The sum of the first 20 terms is 500 -/
  sum_20 : (20 : ℚ) / 2 * (2 * a + (19 : ℚ) * d) = 500
  /-- The sum of the first 200 terms is 2000 -/
  sum_200 : (200 : ℚ) / 2 * (2 * a + (199 : ℚ) * d) = 2000

/-- The sum of the first n terms of an arithmetic progression -/
def sum_n (ap : ArithmeticProgression) (n : ℚ) : ℚ :=
  n / 2 * (2 * ap.a + (n - 1) * ap.d)

/-- Theorem: The sum of the first 220 terms is 5500/3 -/
theorem sum_220_is_5500_div_3 (ap : ArithmeticProgression) :
  sum_n ap 220 = 5500 / 3 := by
  sorry

end sum_220_is_5500_div_3_l1450_145009


namespace percentage_of_percentage_l1450_145067

theorem percentage_of_percentage (y : ℝ) (h : y ≠ 0) :
  (30 / 100) * (80 / 100) * y = (24 / 100) * y := by
  sorry

end percentage_of_percentage_l1450_145067


namespace steak_weight_l1450_145035

/-- Given 15 pounds of beef cut into 20 equal steaks, prove that each steak weighs 12 ounces. -/
theorem steak_weight (total_pounds : ℕ) (num_steaks : ℕ) (ounces_per_pound : ℕ) : 
  total_pounds = 15 → 
  num_steaks = 20 → 
  ounces_per_pound = 16 → 
  (total_pounds * ounces_per_pound) / num_steaks = 12 := by
  sorry

end steak_weight_l1450_145035


namespace tyler_meal_combinations_correct_l1450_145012

/-- The number of different meal combinations Tyler can choose at a buffet. -/
def tyler_meal_combinations : ℕ := 150

/-- The number of meat options available. -/
def meat_options : ℕ := 3

/-- The number of vegetable options available. -/
def vegetable_options : ℕ := 5

/-- The number of vegetables Tyler must choose. -/
def vegetables_to_choose : ℕ := 3

/-- The number of dessert options available. -/
def dessert_options : ℕ := 5

/-- Theorem stating that the number of meal combinations Tyler can choose is correct. -/
theorem tyler_meal_combinations_correct :
  tyler_meal_combinations = meat_options * (Nat.choose vegetable_options vegetables_to_choose) * dessert_options :=
by sorry

end tyler_meal_combinations_correct_l1450_145012


namespace class_mean_calculation_l1450_145065

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_mean : ℚ)
  (group2_students : ℕ) (group2_mean : ℚ) :
  total_students = 28 →
  group1_students = 24 →
  group2_students = 4 →
  group1_mean = 68 / 100 →
  group2_mean = 82 / 100 →
  (group1_students * group1_mean + group2_students * group2_mean) / total_students = 70 / 100 := by
sorry

end class_mean_calculation_l1450_145065


namespace points_always_odd_l1450_145092

/-- The number of points after k operations of adding a point between every two neighboring points. -/
def num_points (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then n
  else 2 * (num_points n (k - 1)) - 1

/-- Theorem: The number of points is always odd after each operation. -/
theorem points_always_odd (n : ℕ) (k : ℕ) (h : n ≥ 2) :
  Odd (num_points n k) :=
sorry

end points_always_odd_l1450_145092


namespace father_son_ages_new_age_ratio_l1450_145094

/-- Given the ratio of a father's age to his son's age and their age product, 
    prove the father's age, son's age, and their combined income. -/
theorem father_son_ages (father_son_ratio : ℚ) (age_product : ℕ) (income_percentage : ℚ) :
  father_son_ratio = 7/3 →
  age_product = 756 →
  income_percentage = 2/5 →
  ∃ (father_age son_age : ℕ) (combined_income : ℚ),
    father_age = 42 ∧
    son_age = 18 ∧
    combined_income = 105 ∧
    (father_age : ℚ) / son_age = father_son_ratio ∧
    father_age * son_age = age_product ∧
    (father_age : ℚ) = income_percentage * combined_income :=
by sorry

/-- Given the father's and son's ages, prove their new age ratio after 6 years. -/
theorem new_age_ratio (father_age son_age : ℕ) (years : ℕ) :
  father_age = 42 →
  son_age = 18 →
  years = 6 →
  ∃ (new_ratio : ℚ),
    new_ratio = 2/1 ∧
    new_ratio = (father_age + years : ℚ) / (son_age + years) :=
by sorry

end father_son_ages_new_age_ratio_l1450_145094


namespace sum_of_digits_of_B_is_seven_l1450_145072

/-- The sum of digits of a natural number in base 10 -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digitSum (n / 10)

/-- The number 4444^444 -/
def bigNumber : ℕ := 4444^444

/-- A is the sum of digits of bigNumber -/
def A : ℕ := digitSum bigNumber

/-- B is the sum of digits of A -/
def B : ℕ := digitSum A

theorem sum_of_digits_of_B_is_seven : digitSum B = 7 := by
  sorry

end sum_of_digits_of_B_is_seven_l1450_145072


namespace perfect_square_count_l1450_145010

theorem perfect_square_count : ∃ (S : Finset Nat), 
  (∀ n ∈ S, n > 0 ∧ n ≤ 2000 ∧ ∃ k : Nat, 21 * n = k * k) ∧ 
  S.card = 9 ∧
  (∀ n : Nat, n > 0 ∧ n ≤ 2000 ∧ (∃ k : Nat, 21 * n = k * k) → n ∈ S) := by
  sorry

end perfect_square_count_l1450_145010


namespace DE_DB_ratio_l1450_145020

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the conditions
variable (right_angle_ABC : (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = 0)
variable (right_angle_ABD : (D.1 - A.1) * (B.1 - A.1) + (D.2 - A.2) * (B.2 - A.2) = 0)
variable (AC_length : Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 2)
variable (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 3)
variable (AD_length : Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 7)
variable (C_D_opposite : (B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1) * (B.1 - A.1) * (D.2 - A.2) - (B.2 - A.2) * (D.1 - A.1) < 0)
variable (D_parallel_AC : (D.2 - A.2) * (C.1 - A.1) = (D.1 - A.1) * (C.2 - A.2))
variable (E_on_CB_extended : ∃ t : ℝ, E = (B.1 + t * (C.1 - B.1), B.2 + t * (C.2 - B.2)) ∧ t > 1)

-- Theorem statement
theorem DE_DB_ratio :
  Real.sqrt ((E.1 - D.1)^2 + (E.2 - D.2)^2) / Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 5 / 9 :=
sorry

end DE_DB_ratio_l1450_145020


namespace no_integer_roots_l1450_145059

theorem no_integer_roots (a b c : ℤ) (ha : a ≠ 0)
  (h0 : Odd (c))
  (h1 : Odd (a + b + c)) :
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
sorry

end no_integer_roots_l1450_145059


namespace sum_of_special_integers_l1450_145036

theorem sum_of_special_integers (x y : ℕ) 
  (h1 : x > y) 
  (h2 : x - y = 8) 
  (h3 : x * y = 168) : 
  x + y = 32 := by
sorry

end sum_of_special_integers_l1450_145036


namespace distance_one_fourth_from_perigee_l1450_145095

/-- Represents an elliptical orbit -/
structure EllipticalOrbit where
  perigee : ℝ
  apogee : ℝ

/-- Calculates the distance from the focus to a point on the major axis of an elliptical orbit -/
def distanceFromFocus (orbit : EllipticalOrbit) (fraction : ℝ) : ℝ :=
  let majorAxis := orbit.apogee + orbit.perigee
  let centerToFocus := Real.sqrt ((majorAxis / 2) ^ 2 - orbit.perigee ^ 2)
  let distanceFromPerigee := fraction * majorAxis
  distanceFromPerigee

/-- Theorem: For an elliptical orbit with perigee 3 AU and apogee 15 AU,
    the distance from the focus to a point 1/4 of the way from perigee to apogee
    along the major axis is 4.5 AU -/
theorem distance_one_fourth_from_perigee (orbit : EllipticalOrbit)
    (h1 : orbit.perigee = 3)
    (h2 : orbit.apogee = 15) :
    distanceFromFocus orbit (1/4) = 4.5 := by
  sorry

end distance_one_fourth_from_perigee_l1450_145095


namespace line_plane_relationship_l1450_145060

-- Define the necessary structures
structure Line :=
  (id : String)

structure Plane :=
  (id : String)

-- Define the relationships
def parallel (l : Line) (p : Plane) : Prop :=
  sorry

def incident (l : Line) (p : Plane) : Prop :=
  sorry

def parallel_lines (l1 l2 : Line) : Prop :=
  sorry

def skew_lines (l1 l2 : Line) : Prop :=
  sorry

-- Theorem statement
theorem line_plane_relationship (a b : Line) (α : Plane) 
  (h1 : parallel a α) (h2 : incident b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end line_plane_relationship_l1450_145060


namespace work_rate_ratio_l1450_145031

/-- 
Theorem: Given a job that P can complete in 4 days, and P and Q together can complete in 3 days, 
the ratio of Q's work rate to P's work rate is 1/3.
-/
theorem work_rate_ratio (p q : ℝ) : 
  p > 0 ∧ q > 0 →  -- Ensure positive work rates
  (1 / p = 1 / 4) →  -- P completes the job in 4 days
  (1 / (p + q) = 1 / 3) →  -- P and Q together complete the job in 3 days
  q / p = 1 / 3 :=
by
  sorry


end work_rate_ratio_l1450_145031


namespace multiply_and_add_l1450_145017

theorem multiply_and_add : 12 * 25 + 16 * 15 = 540 := by sorry

end multiply_and_add_l1450_145017


namespace segment_length_l1450_145042

/-- Given two points P and Q on a line segment AB, prove that AB has length 336/11 -/
theorem segment_length (A B P Q : ℝ) : 
  (0 < A ∧ A < P ∧ P < Q ∧ Q < B) →  -- P and Q are on AB and on the same side of midpoint
  (P - A) / (B - P) = 3 / 4 →        -- P divides AB in ratio 3:4
  (Q - A) / (B - Q) = 5 / 7 →        -- Q divides AB in ratio 5:7
  Q - P = 4 →                        -- PQ = 4
  B - A = 336 / 11 := by             -- AB has length 336/11
sorry


end segment_length_l1450_145042


namespace fraction_product_l1450_145044

theorem fraction_product : (2 : ℚ) / 3 * 5 / 7 * 9 / 11 * 4 / 13 = 360 / 3003 := by
  sorry

end fraction_product_l1450_145044


namespace dima_places_more_berries_l1450_145084

/-- The total number of berries on the bush -/
def total_berries : ℕ := 450

/-- Dima's picking pattern: fraction of berries that go into the basket -/
def dima_basket_ratio : ℚ := 1/2

/-- Sergei's picking pattern: fraction of berries that go into the basket -/
def sergei_basket_ratio : ℚ := 2/3

/-- Dima's picking speed relative to Sergei -/
def dima_speed_ratio : ℕ := 2

/-- The number of berries Dima puts in the basket -/
def dima_basket_berries : ℕ := 150

/-- The number of berries Sergei puts in the basket -/
def sergei_basket_berries : ℕ := 100

/-- Theorem stating that Dima places 50 more berries into the basket than Sergei -/
theorem dima_places_more_berries :
  dima_basket_berries - sergei_basket_berries = 50 :=
sorry

end dima_places_more_berries_l1450_145084


namespace pin_sequence_solution_l1450_145055

def pin_sequence (k : ℕ) (n : ℕ) : ℕ := 2 + k * (n - 1)

theorem pin_sequence_solution :
  ∀ k : ℕ, (pin_sequence k 10 > 45 ∧ pin_sequence k 15 < 90) ↔ (k = 5 ∨ k = 6) :=
by sorry

end pin_sequence_solution_l1450_145055


namespace liquid_film_radius_l1450_145001

/-- Given a box with dimensions and a liquid that partially fills it, 
    calculate the radius of the circular film formed when poured on water. -/
theorem liquid_film_radius 
  (box_length : ℝ) 
  (box_width : ℝ) 
  (box_height : ℝ) 
  (fill_percentage : ℝ) 
  (film_thickness : ℝ) : 
  box_length = 5 → 
  box_width = 4 → 
  box_height = 10 → 
  fill_percentage = 0.8 → 
  film_thickness = 0.05 → 
  ∃ (r : ℝ), r = Real.sqrt (3200 / Real.pi) ∧ 
  r^2 * Real.pi * film_thickness = box_length * box_width * box_height * fill_percentage :=
by sorry

end liquid_film_radius_l1450_145001


namespace arithmetic_sequence_sum_l1450_145006

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- Last term of an arithmetic sequence -/
def last_term (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem arithmetic_sequence_sum :
  ∃ n : ℕ, n > 0 ∧ last_term 1 2 n = 21 ∧ arithmetic_sum 1 2 n = 121 := by
  sorry

end arithmetic_sequence_sum_l1450_145006


namespace curve_transformation_l1450_145075

/-- Given a curve y = sin(2x) and a scaling transformation x' = 2x, y' = 3y,
    prove that the resulting curve has the equation y' = 3sin(x'). -/
theorem curve_transformation (x x' y y' : ℝ) : 
  y = Real.sin (2 * x) → 
  x' = 2 * x → 
  y' = 3 * y → 
  y' = 3 * Real.sin x' :=
by sorry

end curve_transformation_l1450_145075


namespace line_parallel_plane_necessary_not_sufficient_l1450_145076

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallelism relation for planes and lines
variable (planeParallel : Plane → Plane → Prop)
variable (lineParallelPlane : Line → Plane → Prop)

-- Define the containment relation for lines in planes
variable (lineInPlane : Line → Plane → Prop)

theorem line_parallel_plane_necessary_not_sufficient
  (α β : Plane) (m : Line)
  (distinct : α ≠ β)
  (m_in_α : lineInPlane m α) :
  (∀ (α β : Plane) (m : Line), planeParallel α β → lineInPlane m α → lineParallelPlane m β) ∧
  (∃ (α β : Plane) (m : Line), lineParallelPlane m β ∧ lineInPlane m α ∧ ¬planeParallel α β) :=
by sorry

end line_parallel_plane_necessary_not_sufficient_l1450_145076


namespace root_shift_theorem_l1450_145032

theorem root_shift_theorem (a b c : ℂ) : 
  (∀ x : ℂ, x^3 - 6*x^2 + 11*x - 6 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x : ℂ, x^3 - 15*x^2 + 74*x - 120 = 0 ↔ x = a + 3 ∨ x = b + 3 ∨ x = c + 3) :=
by sorry

end root_shift_theorem_l1450_145032


namespace trig_identities_l1450_145087

theorem trig_identities (α : ℝ) (h : Real.tan α = 7) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8/13 ∧
  Real.sin α * Real.cos α = 7/50 := by
  sorry

end trig_identities_l1450_145087


namespace boys_playing_basketball_l1450_145028

/-- Given a class with the following properties:
  * There are 30 students in total
  * One-third of the students are girls
  * Three-quarters of the boys play basketball
  Prove that the number of boys who play basketball is 15 -/
theorem boys_playing_basketball (total_students : ℕ) (girls : ℕ) (boys : ℕ) (boys_playing : ℕ) : 
  total_students = 30 →
  girls = total_students / 3 →
  boys = total_students - girls →
  boys_playing = (3 * boys) / 4 →
  boys_playing = 15 := by
sorry

end boys_playing_basketball_l1450_145028


namespace original_lettuce_price_l1450_145040

/-- Grocery order with item substitutions -/
def grocery_order (original_total delivery_tip new_total original_tomatoes new_tomatoes
                   original_celery new_celery new_lettuce : ℚ) : Prop :=
  -- Original order total before changes
  original_total = 25 ∧
  -- Delivery and tip
  delivery_tip = 8 ∧
  -- New total after changes and delivery/tip
  new_total = 35 ∧
  -- Original and new prices for tomatoes
  original_tomatoes = 0.99 ∧
  new_tomatoes = 2.20 ∧
  -- Original and new prices for celery
  original_celery = 1.96 ∧
  new_celery = 2 ∧
  -- New price for lettuce
  new_lettuce = 1.75

/-- The cost of the original lettuce -/
def original_lettuce_cost (original_total delivery_tip new_total original_tomatoes new_tomatoes
                           original_celery new_celery new_lettuce : ℚ) : ℚ :=
  new_lettuce - ((new_total - delivery_tip) - (original_total + (new_tomatoes - original_tomatoes) + (new_celery - original_celery)))

theorem original_lettuce_price
  (original_total delivery_tip new_total original_tomatoes new_tomatoes
   original_celery new_celery new_lettuce : ℚ)
  (h : grocery_order original_total delivery_tip new_total original_tomatoes new_tomatoes
                     original_celery new_celery new_lettuce) :
  original_lettuce_cost original_total delivery_tip new_total original_tomatoes new_tomatoes
                        original_celery new_celery new_lettuce = 1 := by
  sorry

end original_lettuce_price_l1450_145040


namespace ceiling_negative_seven_fourths_squared_l1450_145058

theorem ceiling_negative_seven_fourths_squared : ⌈(-(7/4))^2⌉ = 4 := by sorry

end ceiling_negative_seven_fourths_squared_l1450_145058


namespace sqrt_x_minus_2_meaningful_l1450_145004

theorem sqrt_x_minus_2_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 2) ↔ x ≥ 2 := by sorry

end sqrt_x_minus_2_meaningful_l1450_145004


namespace division_problem_l1450_145062

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 22 →
  divisor = 3 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  quotient = 7 := by
sorry

end division_problem_l1450_145062


namespace toy_car_speed_l1450_145022

theorem toy_car_speed (t s : ℝ) (h1 : t = 15 * s^2) (h2 : t = 3) : s = Real.sqrt 2 / 5 := by
  sorry

end toy_car_speed_l1450_145022


namespace melissa_points_per_game_l1450_145047

/-- The number of points Melissa scored in total -/
def total_points : ℕ := 91

/-- The number of games Melissa played -/
def num_games : ℕ := 13

/-- The number of points Melissa scored in each game -/
def points_per_game : ℕ := total_points / num_games

/-- Theorem stating that Melissa scored 7 points in each game -/
theorem melissa_points_per_game : points_per_game = 7 := by
  sorry

end melissa_points_per_game_l1450_145047


namespace red_books_probability_l1450_145082

-- Define the number of red books and total books
def num_red_books : ℕ := 4
def total_books : ℕ := 8

-- Define the number of books to be selected
def books_selected : ℕ := 2

-- Define the probability function
def probability (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ :=
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ)

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem red_books_probability :
  probability (combination num_red_books books_selected) (combination total_books books_selected) = 3 / 14 := by
  sorry

end red_books_probability_l1450_145082


namespace wage_payment_period_l1450_145073

/-- Given a sum of money that can pay three workers' wages for different periods,
    prove that it can pay their combined wages for a specific period when working together. -/
theorem wage_payment_period (M : ℝ) (p q r : ℝ) : 
  M = 24 * p ∧ M = 40 * q ∧ M = 30 * r → M = 10 * (p + q + r) := by
sorry

end wage_payment_period_l1450_145073


namespace additive_function_negative_on_positive_properties_l1450_145000

/-- A function satisfying f(x+y) = f(x) + f(y) for all x, y and f(x) < 0 for x > 0 -/
def AdditiveFunctionNegativeOnPositive (f : ℝ → ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y) ∧ (∀ x, x > 0 → f x < 0)

/-- Theorem stating that such a function is odd and monotonically decreasing -/
theorem additive_function_negative_on_positive_properties
    (f : ℝ → ℝ) (h : AdditiveFunctionNegativeOnPositive f) :
    (∀ x, f (-x) = -f x) ∧ (∀ x₁ x₂, x₁ > x₂ → f x₁ < f x₂) := by
  sorry


end additive_function_negative_on_positive_properties_l1450_145000


namespace peach_pie_customers_l1450_145027

/-- Represents the number of slices in an apple pie -/
def apple_slices : ℕ := 8

/-- Represents the number of slices in a peach pie -/
def peach_slices : ℕ := 6

/-- Represents the number of customers who ordered apple pie slices -/
def apple_customers : ℕ := 56

/-- Represents the total number of pies sold -/
def total_pies : ℕ := 15

/-- Theorem stating that the number of customers who ordered peach pie slices is 48 -/
theorem peach_pie_customers : 
  (total_pies * peach_slices) - (apple_customers / apple_slices * peach_slices) = 48 := by
  sorry

end peach_pie_customers_l1450_145027


namespace min_value_theorem_l1450_145099

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_seq : x * Real.log 2 + y * Real.log 2 = 2 * Real.log (Real.sqrt 2)) :
  ∀ a b : ℝ, a > 0 ∧ b > 0 ∧ a * Real.log 2 + b * Real.log 2 = 2 * Real.log (Real.sqrt 2) →
  1 / x + 9 / y ≤ 1 / a + 9 / b :=
by sorry

end min_value_theorem_l1450_145099


namespace michaels_initial_money_l1450_145081

def total_cost : ℕ := 61
def additional_needed : ℕ := 11

theorem michaels_initial_money :
  total_cost - additional_needed = 50 := by
  sorry

end michaels_initial_money_l1450_145081


namespace sum_of_solutions_l1450_145003

-- Define the equation
def equation (x : ℝ) : Prop :=
  x / 3 + x / Real.sqrt (x^2 - 9) = 35 / 12

-- Define the set of solutions
def solution_set : Set ℝ :=
  {x | equation x ∧ x^2 > 9}

-- Theorem statement
theorem sum_of_solutions :
  ∃ (s : Finset ℝ), s.toSet = solution_set ∧ s.sum id = 35 / 4 := by
  sorry

end sum_of_solutions_l1450_145003


namespace min_value_of_g_l1450_145066

-- Define the function g(x)
def g (x : ℝ) : ℝ := 4 * x - x^3

-- State the theorem
theorem min_value_of_g :
  ∃ (min : ℝ), min = 16 * Real.sqrt 3 / 9 ∧
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → g x ≥ min :=
sorry

end min_value_of_g_l1450_145066


namespace pencil_cost_is_0_602_l1450_145011

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := sorry

/-- The cost of a ruler in dollars -/
def ruler_cost : ℝ := sorry

/-- The total cost of six notebooks and four pencils is $7.44 -/
axiom six_notebooks_four_pencils : 6 * notebook_cost + 4 * pencil_cost = 7.44

/-- The total cost of three notebooks and seven pencils is $6.73 -/
axiom three_notebooks_seven_pencils : 3 * notebook_cost + 7 * pencil_cost = 6.73

/-- The total cost of one notebook, two pencils, and a ruler is $3.36 -/
axiom one_notebook_two_pencils_ruler : notebook_cost + 2 * pencil_cost + ruler_cost = 3.36

/-- The cost of each pencil is $0.602 -/
theorem pencil_cost_is_0_602 : pencil_cost = 0.602 := by sorry

end pencil_cost_is_0_602_l1450_145011


namespace cage_cost_proof_l1450_145088

def cat_toy_cost : ℝ := 10.22
def total_cost : ℝ := 21.95

theorem cage_cost_proof : total_cost - cat_toy_cost = 11.73 := by
  sorry

end cage_cost_proof_l1450_145088


namespace certain_number_minus_fifteen_l1450_145096

theorem certain_number_minus_fifteen (x : ℝ) : x / 10 = 6 → x - 15 = 45 := by
  sorry

end certain_number_minus_fifteen_l1450_145096


namespace parabola_translation_leftward_shift_by_2_l1450_145029

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x + 2)^2

-- Theorem stating the translation
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) :=
by
  sorry

-- Theorem stating the leftward shift by 2 units
theorem leftward_shift_by_2 :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) :=
by
  sorry

end parabola_translation_leftward_shift_by_2_l1450_145029


namespace ceo_dividends_calculation_l1450_145007

/-- Calculates the CEO's dividends based on company financial data -/
theorem ceo_dividends_calculation (revenue : ℝ) (expenses : ℝ) (tax_rate : ℝ) 
  (monthly_loan_payment : ℝ) (months_in_year : ℕ) (total_shares : ℕ) (ceo_ownership : ℝ) 
  (h1 : revenue = 2500000)
  (h2 : expenses = 1576250)
  (h3 : tax_rate = 0.2)
  (h4 : monthly_loan_payment = 25000)
  (h5 : months_in_year = 12)
  (h6 : total_shares = 1600)
  (h7 : ceo_ownership = 0.35) :
  ∃ (ceo_dividends : ℝ),
    ceo_dividends = 153440 ∧
    ceo_dividends = 
      ((revenue - expenses - (revenue - expenses) * tax_rate - 
        (monthly_loan_payment * months_in_year)) / total_shares) * 
      ceo_ownership * total_shares :=
by
  sorry

end ceo_dividends_calculation_l1450_145007


namespace sum_of_eleventh_powers_l1450_145005

theorem sum_of_eleventh_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^11 + b^11 = 199 := by
  sorry

end sum_of_eleventh_powers_l1450_145005


namespace sequence_inequality_l1450_145050

theorem sequence_inequality (a : ℕ → ℕ) (h1 : a 1 < a 2) 
  (h2 : ∀ k ≥ 3, a k = 4 * a (k - 1) - 3 * a (k - 2)) : 
  a 45 > 3^43 := by
sorry

end sequence_inequality_l1450_145050


namespace library_visit_equation_l1450_145077

/-- Represents the growth of library visits over three months -/
def library_visit_growth (initial_visits : ℕ) (final_visits : ℕ) (growth_rate : ℝ) : Prop :=
  initial_visits * (1 + growth_rate)^2 = final_visits

/-- Theorem stating that the given equation accurately represents the library visit growth -/
theorem library_visit_equation : 
  ∃ (x : ℝ), library_visit_growth 560 830 x :=
sorry

end library_visit_equation_l1450_145077


namespace a_minus_b_values_l1450_145051

theorem a_minus_b_values (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a + b > 0) :
  a - b = -1 ∨ a - b = -7 := by
  sorry

end a_minus_b_values_l1450_145051


namespace flagpole_height_l1450_145068

/-- Given a flagpole and a building under similar shadow-casting conditions,
    prove that the flagpole's height is 18 meters. -/
theorem flagpole_height 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (building_shadow : ℝ)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_height : building_height = 28)
  (h_building_shadow : building_shadow = 70)
  (h_similar_conditions : True)  -- This represents the similar conditions
  : ∃ (flagpole_height : ℝ), flagpole_height = 18 := by
  sorry

end flagpole_height_l1450_145068


namespace parabola_intersection_condition_l1450_145052

-- Define the parabola function
def parabola (m : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(m-1)*x + m + 1

-- Theorem statement
theorem parabola_intersection_condition (m : ℝ) :
  (∃ a b : ℝ, a > 0 ∧ b < 0 ∧ parabola m a = 0 ∧ parabola m b = 0) ↔ m > -1 := by
  sorry

end parabola_intersection_condition_l1450_145052


namespace partial_fraction_decomposition_l1450_145034

theorem partial_fraction_decomposition (x : ℝ) (h1 : x ≠ -4) (h2 : x ≠ 12) :
  (7 * x - 3) / (x^2 - 8 * x - 48) = 11 / (x + 4) + 0 / (x - 12) := by
  sorry

end partial_fraction_decomposition_l1450_145034


namespace percentage_invalid_votes_l1450_145053

/-- The percentage of invalid votes in an election --/
theorem percentage_invalid_votes 
  (total_votes : ℕ) 
  (candidate_a_percentage : ℚ) 
  (candidate_a_valid_votes : ℕ) 
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 75 / 100)
  (h3 : candidate_a_valid_votes = 357000) :
  (1 - (candidate_a_valid_votes : ℚ) / (candidate_a_percentage * total_votes)) * 100 = 15 := by
sorry

end percentage_invalid_votes_l1450_145053


namespace quartic_roots_arithmetic_sequence_l1450_145037

theorem quartic_roots_arithmetic_sequence (m n : ℚ) : 
  (∃ a b c d : ℚ, 
    (a^2 - 2*a + m) * (a^2 - 2*a + n) = 0 ∧
    (b^2 - 2*b + m) * (b^2 - 2*b + n) = 0 ∧
    (c^2 - 2*c + m) * (c^2 - 2*c + n) = 0 ∧
    (d^2 - 2*d + m) * (d^2 - 2*d + n) = 0 ∧
    a = 1/4 ∧
    b - a = c - b ∧
    c - b = d - c) →
  |m - n| = 1/2 := by
sorry

end quartic_roots_arithmetic_sequence_l1450_145037


namespace vector_collinearity_l1450_145038

/-- Given vectors a, b, and c in ℝ², prove that if k*a + b is collinear with c, then k = -26/15 -/
theorem vector_collinearity (a b c : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (2, 3)) (h3 : c = (4, -7)) :
  (∃ (k : ℝ), ∃ (t : ℝ), t • c = k • a + b) → 
  (∃ (k : ℝ), k • a + b = (-26/15) • a + b) :=
by sorry

end vector_collinearity_l1450_145038


namespace pizza_dough_flour_calculation_l1450_145043

theorem pizza_dough_flour_calculation 
  (original_doughs : ℕ) 
  (original_flour_per_dough : ℚ) 
  (new_doughs : ℕ) 
  (total_flour : ℚ) 
  (h1 : original_doughs = 45)
  (h2 : original_flour_per_dough = 1/9)
  (h3 : new_doughs = 15)
  (h4 : total_flour = original_doughs * original_flour_per_dough)
  (h5 : total_flour = new_doughs * (total_flour / new_doughs)) :
  total_flour / new_doughs = 1/3 := by
sorry

end pizza_dough_flour_calculation_l1450_145043


namespace total_spent_equals_sum_l1450_145080

/-- The total amount Jason spent on clothing -/
def total_spent : ℚ := 19.02

/-- The amount Jason spent on shorts -/
def shorts_cost : ℚ := 14.28

/-- The amount Jason spent on a jacket -/
def jacket_cost : ℚ := 4.74

/-- Theorem stating that the total amount spent equals the sum of shorts and jacket costs -/
theorem total_spent_equals_sum : total_spent = shorts_cost + jacket_cost := by
  sorry

end total_spent_equals_sum_l1450_145080


namespace evening_temperature_l1450_145046

def initial_temp : Int := -7
def temp_rise : Int := 11
def temp_drop : Int := 9

theorem evening_temperature :
  initial_temp + temp_rise - temp_drop = -5 :=
by sorry

end evening_temperature_l1450_145046


namespace sum_mod_five_zero_l1450_145074

theorem sum_mod_five_zero : (4283 + 4284 + 4285 + 4286 + 4287) % 5 = 0 := by
  sorry

end sum_mod_five_zero_l1450_145074


namespace greatest_common_divisor_and_digit_sum_l1450_145090

def a : ℕ := 1305
def b : ℕ := 4665
def c : ℕ := 6905

def diff1 : ℕ := b - a
def diff2 : ℕ := c - b
def diff3 : ℕ := c - a

def n : ℕ := Nat.gcd diff1 (Nat.gcd diff2 diff3)

def sum_of_digits (k : ℕ) : ℕ :=
  if k < 10 then k else (k % 10) + sum_of_digits (k / 10)

theorem greatest_common_divisor_and_digit_sum :
  n = 1120 ∧ sum_of_digits n = 4 := by sorry

end greatest_common_divisor_and_digit_sum_l1450_145090


namespace total_blue_balloons_l1450_145039

theorem total_blue_balloons (joan_balloons melanie_balloons : ℕ) 
  (h1 : joan_balloons = 40) 
  (h2 : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := by
  sorry

end total_blue_balloons_l1450_145039


namespace hyperbola_condition_l1450_145056

-- Define the condition for a hyperbola
def is_hyperbola (k : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 / (k - 3) - y^2 / (k + 3) = 1

-- Define the sufficient condition
def sufficient_condition (k : ℝ) : Prop :=
  k > 3 → is_hyperbola k

-- Define the necessary condition
def necessary_condition (k : ℝ) : Prop :=
  is_hyperbola k → k > 3

-- Theorem statement
theorem hyperbola_condition :
  (∀ k : ℝ, sufficient_condition k) ∧ ¬(∀ k : ℝ, necessary_condition k) :=
sorry

end hyperbola_condition_l1450_145056


namespace duck_percentage_among_non_herons_l1450_145083

theorem duck_percentage_among_non_herons (total : ℝ) (geese swan heron duck : ℝ) :
  geese = 0.28 * total →
  swan = 0.20 * total →
  heron = 0.15 * total →
  duck = 0.32 * total →
  (duck / (total - heron)) * 100 = 37.6 :=
by
  sorry

end duck_percentage_among_non_herons_l1450_145083


namespace painter_week_total_l1450_145018

/-- Represents the painter's work schedule and productivity --/
structure PainterSchedule where
  monday_speed : ℝ
  normal_speed : ℝ
  friday_speed : ℝ
  normal_hours : ℝ
  friday_hours : ℝ
  friday_monday_diff : ℝ

/-- Calculates the total length of fence painted over the week --/
def total_painted (schedule : PainterSchedule) : ℝ :=
  let monday_length := schedule.monday_speed * schedule.normal_hours
  let normal_day_length := schedule.normal_speed * schedule.normal_hours
  let friday_length := schedule.friday_speed * schedule.friday_hours
  monday_length + 3 * normal_day_length + friday_length

/-- Theorem stating the total length of fence painted over the week --/
theorem painter_week_total (schedule : PainterSchedule)
  (h1 : schedule.monday_speed = 0.5 * schedule.normal_speed)
  (h2 : schedule.friday_speed = 2 * schedule.normal_speed)
  (h3 : schedule.friday_hours = 6)
  (h4 : schedule.normal_hours = 8)
  (h5 : schedule.friday_speed * schedule.friday_hours - 
        schedule.monday_speed * schedule.normal_hours = schedule.friday_monday_diff)
  (h6 : schedule.friday_monday_diff = 300) :
  total_painted schedule = 1500 := by
  sorry


end painter_week_total_l1450_145018


namespace gcd_83_power_plus_one_l1450_145015

theorem gcd_83_power_plus_one (h : Prime 83) : 
  Nat.gcd (83^9 + 1) (83^9 + 83^2 + 1) = 1 := by
  sorry

end gcd_83_power_plus_one_l1450_145015


namespace min_pigs_on_farm_l1450_145085

theorem min_pigs_on_farm (P : ℕ) (T : ℕ) : 
  (P > 0) → 
  (T > 0) → 
  (P ≤ T) → 
  (54 * T ≤ 100 * P) → 
  (100 * P ≤ 57 * T) → 
  (∀ Q : ℕ, Q > 0 ∧ Q < P → ¬(54 * T ≤ 100 * Q ∧ 100 * Q ≤ 57 * T)) →
  P = 5 :=
by sorry

#check min_pigs_on_farm

end min_pigs_on_farm_l1450_145085


namespace pants_price_problem_l1450_145008

theorem pants_price_problem (total_cost belt_price pants_price : ℝ) :
  total_cost = 70.93 →
  pants_price = belt_price - 2.93 →
  total_cost = pants_price + belt_price →
  pants_price = 34.00 := by
sorry

end pants_price_problem_l1450_145008


namespace meaningful_expression_l1450_145057

theorem meaningful_expression (x : ℝ) : 
  (10 - x ≥ 0 ∧ x ≠ 4) ↔ x = 8 := by sorry

end meaningful_expression_l1450_145057


namespace zipline_configurations_count_l1450_145013

/-- The number of stories in each building -/
def n : ℕ := 5

/-- The total number of steps (right + up) -/
def total_steps : ℕ := n + n

/-- The number of ways to string ziplines between two n-story buildings
    satisfying the given conditions -/
def num_zipline_configurations : ℕ := Nat.choose total_steps n

/-- Theorem stating that the number of zipline configurations
    is equal to 252 -/
theorem zipline_configurations_count :
  num_zipline_configurations = 252 := by sorry

end zipline_configurations_count_l1450_145013


namespace sin_150_degrees_l1450_145061

theorem sin_150_degrees : Real.sin (150 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_150_degrees_l1450_145061
