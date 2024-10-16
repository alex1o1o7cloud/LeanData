import Mathlib

namespace NUMINAMATH_CALUDE_red_cell_remains_l4139_413902

theorem red_cell_remains (n : ℕ) :
  ∀ (black_rows black_cols : Finset (Fin (2*n))),
  black_rows.card = n ∧ black_cols.card = n →
  ∃ (red_cells : Finset (Fin (2*n) × Fin (2*n))),
  red_cells.card = 2*n^2 + 1 ∧
  ∃ (cell : Fin (2*n) × Fin (2*n)),
  cell ∈ red_cells ∧ cell.1 ∉ black_rows ∧ cell.2 ∉ black_cols :=
sorry

end NUMINAMATH_CALUDE_red_cell_remains_l4139_413902


namespace NUMINAMATH_CALUDE_apple_cost_l4139_413990

theorem apple_cost (total_cost : ℝ) (initial_dozen : ℕ) (target_dozen : ℕ) :
  total_cost = 62.40 ∧ initial_dozen = 8 ∧ target_dozen = 5 →
  (target_dozen : ℝ) * (total_cost / initial_dozen) = 39.00 :=
by sorry

end NUMINAMATH_CALUDE_apple_cost_l4139_413990


namespace NUMINAMATH_CALUDE_cos_equality_angle_l4139_413925

theorem cos_equality_angle (n : ℤ) : 0 ≤ n ∧ n ≤ 180 → Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_angle_l4139_413925


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l4139_413910

theorem relay_race_arrangements (total_students : Nat) (boys : Nat) (girls : Nat) 
  (selected_students : Nat) (selected_boys : Nat) (selected_girls : Nat) : 
  total_students = 8 →
  boys = 6 →
  girls = 2 →
  selected_students = 4 →
  selected_boys = 3 →
  selected_girls = 1 →
  (Nat.choose girls selected_girls) * 
  (Nat.choose boys selected_boys) * 
  selected_boys * 
  (Nat.factorial (selected_students - 1)) = 720 := by
sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l4139_413910


namespace NUMINAMATH_CALUDE_trisha_walk_distance_l4139_413933

theorem trisha_walk_distance (total_distance : ℝ) (tshirt_to_hotel : ℝ) (hotel_to_postcard : ℝ) :
  total_distance = 0.89 →
  tshirt_to_hotel = 0.67 →
  total_distance = hotel_to_postcard + hotel_to_postcard + tshirt_to_hotel →
  hotel_to_postcard = 0.11 := by
sorry

end NUMINAMATH_CALUDE_trisha_walk_distance_l4139_413933


namespace NUMINAMATH_CALUDE_max_k_value_l4139_413975

theorem max_k_value (k : ℝ) : 
  (k > 0 ∧ ∀ x > 0, k * Real.log (k * x) - Real.exp x ≤ 0) →
  k ≤ Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l4139_413975


namespace NUMINAMATH_CALUDE_restaurant_bill_total_l4139_413979

theorem restaurant_bill_total (number_of_people : ℕ) (individual_payment : ℕ) : 
  number_of_people = 3 → individual_payment = 45 → number_of_people * individual_payment = 135 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_total_l4139_413979


namespace NUMINAMATH_CALUDE_simplify_fraction_a_l4139_413937

theorem simplify_fraction_a (a b c d : ℝ) :
  (3 * a^4 * c + 2 * a^4 * d - 3 * b^4 * c - 2 * b^4 * d) /
  ((9 * c^2 * (a - b) - 4 * d^2 * (a - b)) * ((a + b)^2 - 2 * a * b)) =
  (a + b) / (3 * c - 2 * d) :=
sorry


end NUMINAMATH_CALUDE_simplify_fraction_a_l4139_413937


namespace NUMINAMATH_CALUDE_quadratic_bound_l4139_413929

/-- Given a quadratic function f(x) = a x^2 + b x + c, if for any |u| ≤ 10/11 there exists a v 
    such that |u-v| ≤ 1/11 and |f(v)| ≤ 1, then for all x in [-1, 1], |f(x)| ≤ 2. -/
theorem quadratic_bound (a b c : ℝ) : 
  (∀ u : ℝ, |u| ≤ 10/11 → ∃ v : ℝ, |u - v| ≤ 1/11 ∧ |a * v^2 + b * v + c| ≤ 1) →
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_bound_l4139_413929


namespace NUMINAMATH_CALUDE_f_composition_equals_sqrt2_over_2_l4139_413957

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 - 2^x else Real.sqrt x

theorem f_composition_equals_sqrt2_over_2 : f (f (-1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_equals_sqrt2_over_2_l4139_413957


namespace NUMINAMATH_CALUDE_additional_days_to_double_earnings_l4139_413951

/-- Represents the number of days John has worked so far -/
def days_worked : ℕ := 10

/-- Represents the amount of money John has earned so far in dollars -/
def current_earnings : ℕ := 250

/-- Calculates John's daily rate in dollars -/
def daily_rate : ℚ := current_earnings / days_worked

/-- Calculates the total amount John needs to earn to double his current earnings -/
def target_earnings : ℕ := 2 * current_earnings

/-- Calculates the additional amount John needs to earn -/
def additional_earnings : ℕ := target_earnings - current_earnings

/-- Theorem stating the number of additional days John needs to work -/
theorem additional_days_to_double_earnings : 
  (additional_earnings : ℚ) / daily_rate = 10 := by sorry

end NUMINAMATH_CALUDE_additional_days_to_double_earnings_l4139_413951


namespace NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l4139_413980

theorem complex_fraction_equals_negative_two :
  let z : ℂ := 1 + I
  (z^2) / (1 - z) = -2 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_negative_two_l4139_413980


namespace NUMINAMATH_CALUDE_range_of_m_solution_set_l4139_413904

-- Define the functions f and g
def f (x : ℝ) : ℝ := -abs (x - 2)
def g (x m : ℝ) : ℝ := -abs (x - 3) + m

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∀ x : ℝ, f x > g x m) ↔ m < 1 :=
sorry

-- Theorem for the solution set of f(x) + a - 1 > 0
theorem solution_set (a : ℝ) :
  (∀ x : ℝ, f x + a - 1 > 0) ↔
    (a = 1 ∧ (∀ x : ℝ, x ≠ 2 → x ∈ Set.univ)) ∨
    (a > 1 ∧ (∀ x : ℝ, x ∈ Set.univ)) ∨
    (a < 1 ∧ (∀ x : ℝ, x < 1 + a ∨ x > 3 - a)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_solution_set_l4139_413904


namespace NUMINAMATH_CALUDE_total_weight_of_balls_l4139_413996

theorem total_weight_of_balls (blue_weight brown_weight green_weight : ℝ) 
  (h1 : blue_weight = 6)
  (h2 : brown_weight = 3.12)
  (h3 : green_weight = 4.5) :
  blue_weight + brown_weight + green_weight = 13.62 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_balls_l4139_413996


namespace NUMINAMATH_CALUDE_binomial_permutation_equality_l4139_413913

theorem binomial_permutation_equality (n : ℕ+) :
  3 * (Nat.choose (n.val - 1) (n.val - 5)) = 5 * (Nat.factorial (n.val - 2) / Nat.factorial (n.val - 4)) →
  n.val = 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_permutation_equality_l4139_413913


namespace NUMINAMATH_CALUDE_central_angle_A_B_l4139_413956

noncomputable def earthRadius : ℝ := 1 -- Normalized Earth radius

/-- Represents a point on the Earth's surface using latitude and longitude -/
structure EarthPoint where
  latitude : ℝ
  longitude : ℝ

/-- Calculates the angle at the Earth's center between two points on the surface -/
noncomputable def centralAngle (p1 p2 : EarthPoint) : ℝ := sorry

/-- Point A on Earth's surface -/
def pointA : EarthPoint := { latitude := 0, longitude := 90 }

/-- Point B on Earth's surface -/
def pointB : EarthPoint := { latitude := 30, longitude := -80 }

/-- Theorem stating that the central angle between points A and B is 140 degrees -/
theorem central_angle_A_B :
  centralAngle pointA pointB = 140 * (π / 180) := by sorry

end NUMINAMATH_CALUDE_central_angle_A_B_l4139_413956


namespace NUMINAMATH_CALUDE_triangle_proof_l4139_413919

theorem triangle_proof (a b c A B C : ℝ) : 
  0 < A ∧ A < π → 
  0 < B ∧ B < π → 
  0 < C ∧ C < π → 
  b * Real.cos A - a * Real.sin B = 0 → 
  b = Real.sqrt 2 → 
  (1 / 2) * b * c * Real.sin A = 1 → 
  A = π / 4 ∧ a = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_proof_l4139_413919


namespace NUMINAMATH_CALUDE_alcohol_concentration_l4139_413908

/-- Prove that the concentration of alcohol in the final mixture is 30% --/
theorem alcohol_concentration (vessel1_capacity : ℝ) (vessel1_alcohol_percent : ℝ)
  (vessel2_capacity : ℝ) (vessel2_alcohol_percent : ℝ)
  (total_liquid : ℝ) (final_vessel_capacity : ℝ) :
  vessel1_capacity = 2 →
  vessel1_alcohol_percent = 30 →
  vessel2_capacity = 6 →
  vessel2_alcohol_percent = 40 →
  total_liquid = 8 →
  final_vessel_capacity = 10 →
  let total_alcohol := (vessel1_capacity * vessel1_alcohol_percent / 100) +
                       (vessel2_capacity * vessel2_alcohol_percent / 100)
  (total_alcohol / final_vessel_capacity) * 100 = 30 := by
  sorry

#check alcohol_concentration

end NUMINAMATH_CALUDE_alcohol_concentration_l4139_413908


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l4139_413984

theorem part_to_whole_ratio (N A : ℕ) (h1 : N = 48) (h2 : A = 15) : 
  ∃ P : ℕ, P + A = 27 → P * 4 = N := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l4139_413984


namespace NUMINAMATH_CALUDE_farmer_cows_l4139_413940

theorem farmer_cows (initial_cows : ℕ) (added_cows : ℕ) (sold_fraction : ℚ) 
  (h1 : initial_cows = 51)
  (h2 : added_cows = 5)
  (h3 : sold_fraction = 1 / 4) :
  initial_cows + added_cows - ⌊(initial_cows + added_cows : ℚ) * sold_fraction⌋ = 42 := by
  sorry

end NUMINAMATH_CALUDE_farmer_cows_l4139_413940


namespace NUMINAMATH_CALUDE_smaller_number_proof_l4139_413952

theorem smaller_number_proof (x y : ℝ) : 
  x + y = 84 ∧ y = x + 12 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l4139_413952


namespace NUMINAMATH_CALUDE_power_equality_implies_y_equals_two_l4139_413978

theorem power_equality_implies_y_equals_two : 
  ∀ y : ℝ, (3 : ℝ)^6 = 27^y → y = 2 := by
sorry

end NUMINAMATH_CALUDE_power_equality_implies_y_equals_two_l4139_413978


namespace NUMINAMATH_CALUDE_olympic_volunteer_allocation_l4139_413997

-- Define the number of volunteers and projects
def num_volunteers : ℕ := 5
def num_projects : ℕ := 4

-- Define a function to calculate combinations
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define a function to calculate permutations
def permutation (n r : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - r))

-- Theorem statement
theorem olympic_volunteer_allocation :
  (combination num_volunteers 2) * (permutation num_projects num_projects) = 240 :=
by sorry

end NUMINAMATH_CALUDE_olympic_volunteer_allocation_l4139_413997


namespace NUMINAMATH_CALUDE_reciprocal_sum_theorem_l4139_413970

theorem reciprocal_sum_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 8 * x * y) : 1 / x + 1 / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_sum_theorem_l4139_413970


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_specific_numbers_l4139_413944

theorem arithmetic_mean_of_specific_numbers :
  let numbers := [17, 29, 45, 64]
  (numbers.sum / numbers.length : ℚ) = 38.75 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_specific_numbers_l4139_413944


namespace NUMINAMATH_CALUDE_sphere_volume_l4139_413936

theorem sphere_volume (r : ℝ) (h : 4 * Real.pi * r^2 = 36 * Real.pi) :
  (4 / 3) * Real.pi * r^3 = 36 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_volume_l4139_413936


namespace NUMINAMATH_CALUDE_parallel_line_equation_l4139_413969

-- Define a line by its slope and y-intercept
def Line (m b : ℝ) := {(x, y) : ℝ × ℝ | y = m * x + b}

-- Define parallel lines
def Parallel (l₁ l₂ : ℝ × ℝ → Prop) :=
  ∃ m b₁ b₂, l₁ = Line m b₁ ∧ l₂ = Line m b₂

theorem parallel_line_equation :
  let l₁ := Line (-4) 1  -- y = -4x + 1
  let l₂ := {(x, y) : ℝ × ℝ | 4 * x + y - 3 = 0}  -- 4x + y - 3 = 0
  Parallel l₁ l₂ ∧ (0, 3) ∈ l₂ := by sorry

end NUMINAMATH_CALUDE_parallel_line_equation_l4139_413969


namespace NUMINAMATH_CALUDE_right_triangle_side_ratio_range_l4139_413971

theorem right_triangle_side_ratio_range (a b c x : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b = c * x →
  1 < x ∧ x ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_ratio_range_l4139_413971


namespace NUMINAMATH_CALUDE_prob_at_most_one_even_is_three_fourths_l4139_413934

/-- A die is fair if each number has an equal probability of 1/6 -/
def FairDie (d : Fin 6 → ℝ) : Prop :=
  ∀ n : Fin 6, d n = 1 / 6

/-- The probability of getting an even number on a fair die -/
def ProbEven (d : Fin 6 → ℝ) : ℝ :=
  d 1 + d 3 + d 5

/-- The probability of getting an odd number on a fair die -/
def ProbOdd (d : Fin 6 → ℝ) : ℝ :=
  d 0 + d 2 + d 4

/-- The probability of at most one die showing an even number when throwing two fair dice -/
def ProbAtMostOneEven (d1 d2 : Fin 6 → ℝ) : ℝ :=
  ProbOdd d1 * ProbOdd d2 + ProbOdd d1 * ProbEven d2 + ProbEven d1 * ProbOdd d2

theorem prob_at_most_one_even_is_three_fourths 
  (red blue : Fin 6 → ℝ) 
  (hred : FairDie red) 
  (hblue : FairDie blue) : 
  ProbAtMostOneEven red blue = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_most_one_even_is_three_fourths_l4139_413934


namespace NUMINAMATH_CALUDE_odd_cube_minus_odd_divisible_by_24_l4139_413909

theorem odd_cube_minus_odd_divisible_by_24 (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^3 - (2*n + 1) = 24 * k := by
sorry

end NUMINAMATH_CALUDE_odd_cube_minus_odd_divisible_by_24_l4139_413909


namespace NUMINAMATH_CALUDE_minimum_savings_for_contribution_l4139_413917

def savings_september : ℕ := 50
def savings_october : ℕ := 37
def savings_november : ℕ := 11
def mom_contribution : ℕ := 25
def video_game_cost : ℕ := 87
def amount_left : ℕ := 36

def total_savings : ℕ := savings_september + savings_october + savings_november

theorem minimum_savings_for_contribution :
  total_savings = (amount_left + video_game_cost) - mom_contribution :=
by sorry

end NUMINAMATH_CALUDE_minimum_savings_for_contribution_l4139_413917


namespace NUMINAMATH_CALUDE_inequality_system_solution_l4139_413955

theorem inequality_system_solution (x : ℝ) :
  (1 - x > 3) ∧ (2 * x + 5 ≥ 0) → -2.5 ≤ x ∧ x < -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l4139_413955


namespace NUMINAMATH_CALUDE_gcd_108_450_l4139_413943

theorem gcd_108_450 : Nat.gcd 108 450 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_108_450_l4139_413943


namespace NUMINAMATH_CALUDE_football_count_proof_l4139_413964

/-- The cost of one football -/
def football_cost : ℝ := 35

/-- The cost of one soccer ball -/
def soccer_cost : ℝ := 50

/-- The number of footballs in the first set -/
def num_footballs : ℕ := 3

/-- The total cost of the first set -/
def first_set_cost : ℝ := 155

/-- The total cost of the second set -/
def second_set_cost : ℝ := 220

theorem football_count_proof :
  (football_cost * num_footballs + soccer_cost = first_set_cost) ∧
  (2 * football_cost + 3 * soccer_cost = second_set_cost) :=
by sorry

end NUMINAMATH_CALUDE_football_count_proof_l4139_413964


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l4139_413914

theorem trigonometric_equation_solution (x : ℝ) :
  8.483 * Real.tan x - Real.sin (2 * x) - Real.cos (2 * x) + 2 * (2 * Real.cos x - 1 / Real.cos x) = 0 ↔
  ∃ k : ℤ, x = π / 4 * (2 * k + 1) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l4139_413914


namespace NUMINAMATH_CALUDE_jerrys_breakfast_l4139_413985

theorem jerrys_breakfast (pancake_calories : ℕ) (bacon_calories : ℕ) (cereal_calories : ℕ) 
  (total_calories : ℕ) (bacon_strips : ℕ) :
  pancake_calories = 120 →
  bacon_calories = 100 →
  cereal_calories = 200 →
  total_calories = 1120 →
  bacon_strips = 2 →
  ∃ (num_pancakes : ℕ), 
    num_pancakes * pancake_calories + 
    bacon_strips * bacon_calories + 
    cereal_calories = total_calories ∧
    num_pancakes = 6 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_breakfast_l4139_413985


namespace NUMINAMATH_CALUDE_sequence_limit_l4139_413912

def a (n : ℕ) : ℚ := (7 * n + 4) / (2 * n + 1)

theorem sequence_limit : ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a n - 7/2| < ε := by
  sorry

end NUMINAMATH_CALUDE_sequence_limit_l4139_413912


namespace NUMINAMATH_CALUDE_possible_ordering_l4139_413903

theorem possible_ordering (a b c : ℝ) 
  (distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (positive : a > 0 ∧ b > 0 ∧ c > 0)
  (eq : a^2 + c^2 = 2*b*c) :
  b > a ∧ a > c :=
sorry

end NUMINAMATH_CALUDE_possible_ordering_l4139_413903


namespace NUMINAMATH_CALUDE_x_value_l4139_413993

theorem x_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l4139_413993


namespace NUMINAMATH_CALUDE_least_valid_number_l4139_413977

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Check if a number is the least positive integer divisible by 17 with digit sum 17 -/
def is_least_valid (m : ℕ) : Prop :=
  m > 0 ∧ m % 17 = 0 ∧ digit_sum m = 17 ∧
  ∀ k : ℕ, 0 < k ∧ k < m → ¬(k % 17 = 0 ∧ digit_sum k = 17)

theorem least_valid_number : is_least_valid 476 := by sorry

end NUMINAMATH_CALUDE_least_valid_number_l4139_413977


namespace NUMINAMATH_CALUDE_correct_calculation_l4139_413976

theorem correct_calculation (a b : ℝ) : 9 * a^2 * b - 9 * a^2 * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l4139_413976


namespace NUMINAMATH_CALUDE_unique_five_digit_number_l4139_413900

theorem unique_five_digit_number : ∃! n : ℕ,
  (10000 ≤ n ∧ n < 100000) ∧ 
  (∀ i j, i ≠ j → (n / 10^i) % 10 ≠ (n / 10^j) % 10) ∧
  (∀ i, 0 ≤ i ∧ i < 5 → (n / 10^i) % 10 ≠ 0) ∧
  ((n % 1000) = 7 * (n / 100)) ∧
  n = 12946 := by
sorry

end NUMINAMATH_CALUDE_unique_five_digit_number_l4139_413900


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l4139_413927

theorem quadratic_rewrite (b : ℝ) (h1 : b > 0) : 
  (∃ n : ℝ, ∀ x : ℝ, x^2 + b*x + 60 = (x + n)^2 + 16) → b = 4 * Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l4139_413927


namespace NUMINAMATH_CALUDE_max_daily_revenue_l4139_413938

def price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def daily_revenue (t : ℕ) : ℝ :=
  price t * sales_volume t

theorem max_daily_revenue :
  ∃ t : ℕ, 0 < t ∧ t ≤ 30 ∧ daily_revenue t = 1125 ∧
  ∀ s : ℕ, 0 < s ∧ s ≤ 30 → daily_revenue s ≤ daily_revenue t :=
sorry

end NUMINAMATH_CALUDE_max_daily_revenue_l4139_413938


namespace NUMINAMATH_CALUDE_log_inequality_implies_a_range_l4139_413981

theorem log_inequality_implies_a_range (a : ℝ) : 
  (∃ (loga : ℝ → ℝ → ℝ), loga a 3 < 1) → (a > 3 ∨ (0 < a ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_implies_a_range_l4139_413981


namespace NUMINAMATH_CALUDE_exists_more_kites_than_points_l4139_413987

/-- A point on a grid --/
structure GridPoint where
  x : ℤ
  y : ℤ

/-- A kite shape formed by four points --/
structure Kite where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  p4 : GridPoint

/-- A configuration of points on a grid --/
structure GridConfiguration where
  points : List GridPoint
  kites : List Kite

/-- Function to count the number of kites in a configuration --/
def countKites (config : GridConfiguration) : ℕ :=
  config.kites.length

/-- Function to count the number of points in a configuration --/
def countPoints (config : GridConfiguration) : ℕ :=
  config.points.length

/-- Theorem stating that there exists a configuration with more kites than points --/
theorem exists_more_kites_than_points :
  ∃ (config : GridConfiguration), countKites config > countPoints config := by
  sorry

end NUMINAMATH_CALUDE_exists_more_kites_than_points_l4139_413987


namespace NUMINAMATH_CALUDE_three_equidistant_lines_l4139_413972

/-- Two points in a plane -/
structure TwoPoints (α : Type*) [NormedAddCommGroup α] where
  C : α
  D : α
  dist_CD : ‖C - D‖ = 7

/-- A line that is equidistant from two points -/
structure EquidistantLine (α : Type*) [NormedAddCommGroup α] (p : TwoPoints α) where
  line : Set α
  dist_C : ∀ x ∈ line, ‖x - p.C‖ = 3
  dist_D : ∀ x ∈ line, ‖x - p.D‖ = 4

/-- The theorem stating that there are exactly 3 equidistant lines -/
theorem three_equidistant_lines (α : Type*) [NormedAddCommGroup α] (p : TwoPoints α) :
  ∃! (s : Finset (EquidistantLine α p)), s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_three_equidistant_lines_l4139_413972


namespace NUMINAMATH_CALUDE_cos_alpha_value_l4139_413928

theorem cos_alpha_value (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.sin (α / 2) + Real.cos (α / 2) = Real.sqrt 6 / 2) : 
  Real.cos α = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l4139_413928


namespace NUMINAMATH_CALUDE_bob_has_31_pennies_l4139_413992

/-- The number of pennies Alex currently has -/
def alexPennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bobPennies : ℕ := sorry

/-- If Alex gives Bob a penny, Bob will have four times as many pennies as Alex has -/
axiom condition1 : bobPennies + 1 = 4 * (alexPennies - 1)

/-- If Bob gives Alex a penny, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bobPennies - 1 = 3 * (alexPennies + 1)

/-- Bob currently has 31 pennies -/
theorem bob_has_31_pennies : bobPennies = 31 := by sorry

end NUMINAMATH_CALUDE_bob_has_31_pennies_l4139_413992


namespace NUMINAMATH_CALUDE_units_digit_17_pow_2023_l4139_413945

theorem units_digit_17_pow_2023 : ∃ k : ℕ, 17^2023 ≡ 3 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_17_pow_2023_l4139_413945


namespace NUMINAMATH_CALUDE_sum_coordinates_of_D_l4139_413982

/-- Given a point M that is the midpoint of line segment CD, 
    prove that the sum of coordinates of D is 12 -/
theorem sum_coordinates_of_D (M C D : ℝ × ℝ) : 
  M = (2, 5) → 
  C = (1/2, 3/2) → 
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 12 := by
sorry

end NUMINAMATH_CALUDE_sum_coordinates_of_D_l4139_413982


namespace NUMINAMATH_CALUDE_artists_contemporary_probability_l4139_413953

/-- Represents the birth year of an artist, measured in years ago --/
def BirthYear := Fin 301

/-- Represents the lifetime of an artist --/
structure Lifetime where
  birth : BirthYear
  death : BirthYear
  age_constraint : death.val = birth.val + 80

/-- Two artists are contemporaries if their lifetimes overlap --/
def are_contemporaries (a b : Lifetime) : Prop :=
  (a.birth.val ≤ b.death.val ∧ b.birth.val ≤ a.death.val) ∨
  (b.birth.val ≤ a.death.val ∧ a.birth.val ≤ b.death.val)

/-- The probability of two artists being contemporaries --/
def probability_contemporaries : ℚ :=
  209 / 225

theorem artists_contemporary_probability :
  probability_contemporaries = 209 / 225 := by sorry


end NUMINAMATH_CALUDE_artists_contemporary_probability_l4139_413953


namespace NUMINAMATH_CALUDE_one_true_statement_l4139_413935

theorem one_true_statement (a b c : ℝ) : 
  (∃! n : Nat, n = 1 ∧ 
    (((a ≤ b → a * c^2 ≤ b * c^2) ∨ 
      (a > b → a * c^2 > b * c^2) ∨ 
      (a * c^2 ≤ b * c^2 → a ≤ b)))) := by sorry

end NUMINAMATH_CALUDE_one_true_statement_l4139_413935


namespace NUMINAMATH_CALUDE_distance_ratio_car_a_to_b_l4139_413949

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled by a car -/
def distance (c : Car) : ℝ := c.speed * c.time

/-- Theorem: The ratio of distances covered by Car A and Car B is 3:1 -/
theorem distance_ratio_car_a_to_b (car_a car_b : Car)
    (h_speed_a : car_a.speed = 50)
    (h_time_a : car_a.time = 6)
    (h_speed_b : car_b.speed = 100)
    (h_time_b : car_b.time = 1) :
    distance car_a / distance car_b = 3 := by
  sorry

#check distance_ratio_car_a_to_b

end NUMINAMATH_CALUDE_distance_ratio_car_a_to_b_l4139_413949


namespace NUMINAMATH_CALUDE_valid_outfits_count_l4139_413973

/-- The number of shirts available -/
def num_shirts : ℕ := 8

/-- The number of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 8

/-- The number of colors shared by shirts, pants, and hats -/
def num_shared_colors : ℕ := 5

/-- The number of additional colors for shirts and hats -/
def num_additional_colors : ℕ := 2

/-- The total number of outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of combinations where shirt and hat have the same color -/
def same_color_combinations : ℕ := num_shared_colors * num_pants

/-- The number of valid outfit combinations -/
def valid_combinations : ℕ := total_combinations - same_color_combinations

theorem valid_outfits_count :
  valid_combinations = 295 := by sorry

end NUMINAMATH_CALUDE_valid_outfits_count_l4139_413973


namespace NUMINAMATH_CALUDE_recess_time_calculation_l4139_413930

/-- Calculates the total recess time based on the number of each grade received -/
def total_recess_time (normal_recess : ℕ) (extra_a : ℕ) (extra_b : ℕ) (extra_c : ℕ) (minus_d : ℕ) 
  (num_a : ℕ) (num_b : ℕ) (num_c : ℕ) (num_d : ℕ) : ℕ :=
  normal_recess + extra_a * num_a + extra_b * num_b + extra_c * num_c - minus_d * num_d

theorem recess_time_calculation : 
  total_recess_time 20 3 2 1 1 10 12 14 5 = 83 := by
  sorry

end NUMINAMATH_CALUDE_recess_time_calculation_l4139_413930


namespace NUMINAMATH_CALUDE_inequality_preservation_l4139_413946

theorem inequality_preservation (a b : ℝ) (h : a < b) : a / 3 < b / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l4139_413946


namespace NUMINAMATH_CALUDE_probability_two_teachers_in_A_proof_l4139_413923

/-- The probability of exactly two out of three teachers being assigned to place A -/
def probability_two_teachers_in_A : ℚ := 3/8

/-- The number of teachers -/
def num_teachers : ℕ := 3

/-- The number of places -/
def num_places : ℕ := 2

theorem probability_two_teachers_in_A_proof :
  probability_two_teachers_in_A = 
    (Nat.choose num_teachers 2 : ℚ) / (num_places ^ num_teachers : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_probability_two_teachers_in_A_proof_l4139_413923


namespace NUMINAMATH_CALUDE_cost_effectiveness_theorem_l4139_413941

/-- Represents the cost of a plan based on the number of students -/
def plan_cost (students : ℕ) (teacher_free : Bool) (discount : ℚ) : ℚ :=
  if teacher_free then
    25 * students
  else
    25 * discount * (students + 1)

/-- Determines which plan is more cost-effective based on the number of students -/
def cost_effective_plan (students : ℕ) : String :=
  let plan1_cost := plan_cost students true 1
  let plan2_cost := plan_cost students false (4/5)
  if plan1_cost < plan2_cost then "Plan 1"
  else if plan1_cost > plan2_cost then "Plan 2"
  else "Both plans are equally cost-effective"

theorem cost_effectiveness_theorem (students : ℕ) :
  cost_effective_plan students =
    if students < 4 then "Plan 1"
    else if students > 4 then "Plan 2"
    else "Both plans are equally cost-effective" :=
  sorry

end NUMINAMATH_CALUDE_cost_effectiveness_theorem_l4139_413941


namespace NUMINAMATH_CALUDE_fraction_of_fifteen_l4139_413967

theorem fraction_of_fifteen (x : ℚ) : 
  (x * 15 = 0.8 * 40 - 20) → x = 4/5 := by
sorry

end NUMINAMATH_CALUDE_fraction_of_fifteen_l4139_413967


namespace NUMINAMATH_CALUDE_vector_on_line_from_projection_l4139_413918

/-- Given a vector u = (x, y) in R², prove that if its projection onto (4, 3) 
    equals (4, 3), then u lies on the line y = -4/3x + 25/3 -/
theorem vector_on_line_from_projection (x y : ℝ) : 
  let u : Fin 2 → ℝ := ![x, y]
  let v : Fin 2 → ℝ := ![4, 3]
  (v • u / (v • v)) • v = v → y = -4/3 * x + 25/3 := by
  sorry

end NUMINAMATH_CALUDE_vector_on_line_from_projection_l4139_413918


namespace NUMINAMATH_CALUDE_sum_of_digits_implies_even_l4139_413926

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of digits of N is 100 and the sum of digits of 5N is 50, then N is even -/
theorem sum_of_digits_implies_even (N : ℕ) 
  (h1 : sumOfDigits N = 100) 
  (h2 : sumOfDigits (5 * N) = 50) : 
  Even N := by
  sorry


end NUMINAMATH_CALUDE_sum_of_digits_implies_even_l4139_413926


namespace NUMINAMATH_CALUDE_calculate_hourly_wage_l4139_413906

/-- Calculates the hourly wage of a worker given their work conditions and pay --/
theorem calculate_hourly_wage (hours_per_week : ℕ) (deduction_per_lateness : ℕ) 
  (lateness_count : ℕ) (pay_after_deductions : ℕ) : 
  hours_per_week = 18 → 
  deduction_per_lateness = 5 → 
  lateness_count = 3 → 
  pay_after_deductions = 525 → 
  (pay_after_deductions + lateness_count * deduction_per_lateness) / hours_per_week = 30 := by
  sorry

end NUMINAMATH_CALUDE_calculate_hourly_wage_l4139_413906


namespace NUMINAMATH_CALUDE_alma_carrot_distribution_l4139_413999

/-- Given a number of carrots and goats, calculate the number of carrots left over
    when distributing carrots equally among goats. -/
def carrots_left_over (total_carrots : ℕ) (num_goats : ℕ) : ℕ :=
  total_carrots % num_goats

theorem alma_carrot_distribution :
  carrots_left_over 47 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_alma_carrot_distribution_l4139_413999


namespace NUMINAMATH_CALUDE_pet_store_black_cats_l4139_413924

/-- Given a pet store with white, black, and gray cats, prove the number of black cats. -/
theorem pet_store_black_cats 
  (total_cats : ℕ) 
  (white_cats : ℕ) 
  (gray_cats : ℕ) 
  (h_total : total_cats = 15) 
  (h_white : white_cats = 2) 
  (h_gray : gray_cats = 3) :
  total_cats - white_cats - gray_cats = 10 :=
by
  sorry

#check pet_store_black_cats

end NUMINAMATH_CALUDE_pet_store_black_cats_l4139_413924


namespace NUMINAMATH_CALUDE_greatest_number_of_teams_l4139_413994

theorem greatest_number_of_teams (num_girls num_boys : ℕ) 
  (h_girls : num_girls = 40)
  (h_boys : num_boys = 32) :
  (∃ k : ℕ, k > 0 ∧ k ∣ num_girls ∧ k ∣ num_boys ∧ 
    ∀ m : ℕ, m > 0 → m ∣ num_girls → m ∣ num_boys → m ≤ k) ↔ 
  Nat.gcd num_girls num_boys = 8 :=
by sorry

end NUMINAMATH_CALUDE_greatest_number_of_teams_l4139_413994


namespace NUMINAMATH_CALUDE_tangency_implies_n_equals_two_l4139_413961

/-- The value of n for which the ellipse x^2 + 9y^2 = 9 is tangent to the hyperbola x^2 - n(y - 1)^2 = 1 -/
def tangency_value : ℝ := 2

/-- The equation of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- The equation of the hyperbola -/
def is_on_hyperbola (x y n : ℝ) : Prop := x^2 - n*(y - 1)^2 = 1

/-- The ellipse and hyperbola are tangent -/
def are_tangent (n : ℝ) : Prop :=
  ∃ x y : ℝ, is_on_ellipse x y ∧ is_on_hyperbola x y n ∧
  ∀ x' y' : ℝ, is_on_ellipse x' y' ∧ is_on_hyperbola x' y' n → (x', y') = (x, y)

theorem tangency_implies_n_equals_two :
  are_tangent tangency_value := by sorry

end NUMINAMATH_CALUDE_tangency_implies_n_equals_two_l4139_413961


namespace NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l4139_413942

theorem negation_of_quadratic_inequality :
  (∃ x : ℝ, x^2 - x + 3 ≤ 0) ↔ ¬(∀ x : ℝ, x^2 - x + 3 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_quadratic_inequality_l4139_413942


namespace NUMINAMATH_CALUDE_ab_dot_bc_equals_two_l4139_413931

/-- Given two vectors AB and AC in R², and the magnitude of BC is 1, 
    prove that the dot product of AB and BC is 2. -/
theorem ab_dot_bc_equals_two 
  (AB : ℝ × ℝ) 
  (AC : ℝ × ℝ) 
  (h1 : AB = (2, 3)) 
  (h2 : ∃ t, AC = (3, t)) 
  (h3 : ‖AC - AB‖ = 1) : 
  AB • (AC - AB) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_dot_bc_equals_two_l4139_413931


namespace NUMINAMATH_CALUDE_owen_sleep_time_l4139_413921

theorem owen_sleep_time (total_hours work_hours chore_hours sleep_hours : ℕ) :
  total_hours = 24 ∧ work_hours = 6 ∧ chore_hours = 7 ∧ sleep_hours = total_hours - (work_hours + chore_hours) →
  sleep_hours = 11 := by
  sorry

end NUMINAMATH_CALUDE_owen_sleep_time_l4139_413921


namespace NUMINAMATH_CALUDE_triangle_area_is_four_l4139_413911

/-- The area of the triangle formed by the intersection of lines y = x + 2, y = -x + 8, and y = 3 -/
def triangleArea : ℝ := 4

/-- The first line equation: y = x + 2 -/
def line1 (x y : ℝ) : Prop := y = x + 2

/-- The second line equation: y = -x + 8 -/
def line2 (x y : ℝ) : Prop := y = -x + 8

/-- The third line equation: y = 3 -/
def line3 (x y : ℝ) : Prop := y = 3

theorem triangle_area_is_four :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    line1 x₁ y₁ ∧ line3 x₁ y₁ ∧
    line2 x₂ y₂ ∧ line3 x₂ y₂ ∧
    line1 x₃ y₃ ∧ line2 x₃ y₃ ∧
    triangleArea = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_is_four_l4139_413911


namespace NUMINAMATH_CALUDE_speaker_discount_savings_l4139_413988

/-- Calculates the savings from a discount given the initial price and discounted price. -/
def savings (initial_price discounted_price : ℝ) : ℝ :=
  initial_price - discounted_price

/-- Theorem stating that the savings from a discount on speakers priced at $475.00 and sold for $199.00 is equal to $276.00. -/
theorem speaker_discount_savings :
  savings 475 199 = 276 := by
  sorry

end NUMINAMATH_CALUDE_speaker_discount_savings_l4139_413988


namespace NUMINAMATH_CALUDE_bakery_tart_flour_calculation_l4139_413983

theorem bakery_tart_flour_calculation 
  (initial_tarts : ℕ) 
  (new_tarts : ℕ) 
  (initial_flour_per_tart : ℚ) 
  (h1 : initial_tarts = 36)
  (h2 : new_tarts = 18)
  (h3 : initial_flour_per_tart = 1 / 12)
  : (initial_tarts : ℚ) * initial_flour_per_tart / new_tarts = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_bakery_tart_flour_calculation_l4139_413983


namespace NUMINAMATH_CALUDE_sheet_reduction_percentage_l4139_413932

def original_sheets : ℕ := 20
def original_lines_per_sheet : ℕ := 55
def original_chars_per_line : ℕ := 65

def new_lines_per_sheet : ℕ := 65
def new_chars_per_line : ℕ := 70

def total_chars : ℕ := original_sheets * original_lines_per_sheet * original_chars_per_line
def new_chars_per_sheet : ℕ := new_lines_per_sheet * new_chars_per_line
def new_sheets : ℕ := (total_chars + new_chars_per_sheet - 1) / new_chars_per_sheet

theorem sheet_reduction_percentage : 
  (original_sheets - new_sheets : ℚ) / original_sheets * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_sheet_reduction_percentage_l4139_413932


namespace NUMINAMATH_CALUDE_simplify_expression_calculate_expression_l4139_413948

-- Part 1
theorem simplify_expression (x y : ℝ) :
  3 * (2 * x - y) - 2 * (4 * x + 1/2 * y) = -2 * x - 4 * y := by sorry

-- Part 2
theorem calculate_expression (x y : ℝ) (h1 : x * y = 4) (h2 : x - y = -7.5) :
  3 * (x * y - 2/3 * y) - 1/2 * (2 * x + 4 * x * y) - (-2 * x - y) = -7/2 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_calculate_expression_l4139_413948


namespace NUMINAMATH_CALUDE_negation_of_existential_quantifier_l4139_413989

theorem negation_of_existential_quantifier :
  (¬ ∃ x : ℝ, x^2 ≤ |x|) ↔ (∀ x : ℝ, x^2 > |x|) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_quantifier_l4139_413989


namespace NUMINAMATH_CALUDE_ball_distribution_count_l4139_413995

/-- Represents a valid distribution of balls into boxes -/
structure BallDistribution where
  x : ℕ
  y : ℕ
  z : ℕ
  sum_eq_7 : x + y + z = 7
  ordered : x ≥ y ∧ y ≥ z

/-- The number of ways to distribute 7 indistinguishable balls into 3 indistinguishable boxes -/
def distributionCount : ℕ := sorry

theorem ball_distribution_count : distributionCount = 8 := by sorry

end NUMINAMATH_CALUDE_ball_distribution_count_l4139_413995


namespace NUMINAMATH_CALUDE_sqrt_negative_undefined_l4139_413963

theorem sqrt_negative_undefined : ¬ ∃ (x : ℝ), x^2 = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_negative_undefined_l4139_413963


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l4139_413920

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧
  ∃ y, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l4139_413920


namespace NUMINAMATH_CALUDE_binomial_10_choose_3_l4139_413947

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_choose_3_l4139_413947


namespace NUMINAMATH_CALUDE_tan_five_pi_fourths_l4139_413922

theorem tan_five_pi_fourths : Real.tan (5 * π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_five_pi_fourths_l4139_413922


namespace NUMINAMATH_CALUDE_test_probability_l4139_413915

/-- The probability of answering exactly k questions correctly out of n questions,
    where the probability of answering each question correctly is p. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

/-- The probability of answering exactly 2 questions correctly out of 6 questions,
    where the probability of answering each question correctly is 1/3, is 240/729. -/
theorem test_probability : binomial_probability 6 2 (1/3) = 240/729 := by
  sorry

end NUMINAMATH_CALUDE_test_probability_l4139_413915


namespace NUMINAMATH_CALUDE_modified_cube_surface_area_l4139_413950

/-- Calculate the entire surface area of a modified cube -/
theorem modified_cube_surface_area :
  let cube_edge : ℝ := 5
  let large_hole_side : ℝ := 2
  let small_hole_side : ℝ := 0.5
  let original_surface_area : ℝ := 6 * cube_edge^2
  let large_holes_area : ℝ := 6 * large_hole_side^2
  let exposed_inner_area : ℝ := 6 * 4 * large_hole_side^2
  let small_holes_area : ℝ := 6 * 4 * small_hole_side^2
  original_surface_area - large_holes_area + exposed_inner_area - small_holes_area = 228 := by
  sorry

end NUMINAMATH_CALUDE_modified_cube_surface_area_l4139_413950


namespace NUMINAMATH_CALUDE_keaton_yearly_earnings_l4139_413916

/-- Represents Keaton's farm earnings --/
def farm_earnings (orange_harvest_interval : ℕ) (orange_harvest_value : ℕ) 
                  (apple_harvest_interval : ℕ) (apple_harvest_value : ℕ) : ℕ :=
  let orange_harvests_per_year := 12 / orange_harvest_interval
  let apple_harvests_per_year := 12 / apple_harvest_interval
  orange_harvests_per_year * orange_harvest_value + apple_harvests_per_year * apple_harvest_value

/-- Theorem stating Keaton's yearly earnings --/
theorem keaton_yearly_earnings : farm_earnings 2 50 3 30 = 420 := by
  sorry

end NUMINAMATH_CALUDE_keaton_yearly_earnings_l4139_413916


namespace NUMINAMATH_CALUDE_function_and_tangent_line_l4139_413968

/-- Given a function f(x) = (ax-6) / (x^2 + b) and its tangent line at (-1, f(-1)) 
    with equation x + 2y + 5 = 0, prove that f(x) = (2x-6) / (x^2 + 3) -/
theorem function_and_tangent_line (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => (a * x - 6) / (x^2 + b)
  let tangent_line : ℝ → ℝ := λ x => -(1/2) * x - 5/2
  (f (-1) = tangent_line (-1)) ∧ 
  (deriv f (-1) = deriv tangent_line (-1)) →
  f = λ x => (2 * x - 6) / (x^2 + 3) := by
sorry

end NUMINAMATH_CALUDE_function_and_tangent_line_l4139_413968


namespace NUMINAMATH_CALUDE_ceiling_of_negative_real_l4139_413901

theorem ceiling_of_negative_real : ⌈(-3.67 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_real_l4139_413901


namespace NUMINAMATH_CALUDE_max_profit_week_is_5_l4139_413974

/-- Price function based on week number -/
def price (x : ℕ) : ℚ :=
  if x ≤ 4 then 10 + 2 * x
  else if x ≤ 10 then 20
  else 20 - 2 * (x - 10)

/-- Cost function based on week number -/
def cost (x : ℕ) : ℚ :=
  -0.125 * (x - 8)^2 + 12

/-- Profit function based on week number -/
def profit (x : ℕ) : ℚ :=
  price x - cost x

/-- The week with maximum profit is the 5th week -/
theorem max_profit_week_is_5 :
  ∀ x : ℕ, x ≤ 16 → profit 5 ≥ profit x :=
sorry

end NUMINAMATH_CALUDE_max_profit_week_is_5_l4139_413974


namespace NUMINAMATH_CALUDE_same_city_probability_l4139_413905

/-- The probability that two specific students are assigned to the same city
    given the total number of students and the number of spots in each city. -/
theorem same_city_probability
  (total_students : ℕ)
  (spots_moscow : ℕ)
  (spots_tula : ℕ)
  (spots_voronezh : ℕ)
  (h1 : total_students = 30)
  (h2 : spots_moscow = 15)
  (h3 : spots_tula = 8)
  (h4 : spots_voronezh = 7)
  (h5 : total_students = spots_moscow + spots_tula + spots_voronezh) :
  (spots_moscow.choose 2 + spots_tula.choose 2 + spots_voronezh.choose 2) / total_students.choose 2 = 154 / 435 :=
by sorry

end NUMINAMATH_CALUDE_same_city_probability_l4139_413905


namespace NUMINAMATH_CALUDE_bird_count_after_changes_l4139_413958

/-- Represents the number of birds of each type on the fence -/
structure BirdCount where
  sparrows : ℕ
  storks : ℕ
  pigeons : ℕ
  swallows : ℕ

/-- Calculates the total number of birds -/
def totalBirds (birds : BirdCount) : ℕ :=
  birds.sparrows + birds.storks + birds.pigeons + birds.swallows

/-- Represents the changes in bird population -/
structure BirdChanges where
  sparrowsJoined : ℕ
  swallowsJoined : ℕ
  pigeonsLeft : ℕ

/-- Applies changes to the bird population -/
def applyChanges (initial : BirdCount) (changes : BirdChanges) : BirdCount :=
  { sparrows := initial.sparrows + changes.sparrowsJoined,
    storks := initial.storks,
    pigeons := initial.pigeons - changes.pigeonsLeft,
    swallows := initial.swallows + changes.swallowsJoined }

theorem bird_count_after_changes 
  (initial : BirdCount)
  (changes : BirdChanges)
  (h_initial : initial = { sparrows := 3, storks := 2, pigeons := 4, swallows := 0 })
  (h_changes : changes = { sparrowsJoined := 3, swallowsJoined := 5, pigeonsLeft := 2 }) :
  totalBirds (applyChanges initial changes) = 15 := by
  sorry


end NUMINAMATH_CALUDE_bird_count_after_changes_l4139_413958


namespace NUMINAMATH_CALUDE_trapezoid_longer_side_length_l4139_413962

-- Define the square
def square_side_length : ℝ := 2

-- Define the number of regions the square is divided into
def num_regions : ℕ := 3

-- Define the theorem
theorem trapezoid_longer_side_length :
  ∀ (trapezoid_area pentagon_area : ℝ),
  trapezoid_area > 0 →
  pentagon_area > 0 →
  trapezoid_area = pentagon_area →
  trapezoid_area = (square_side_length ^ 2) / num_regions →
  ∃ (y : ℝ),
    y = 5 / 3 ∧
    trapezoid_area = (y + 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_trapezoid_longer_side_length_l4139_413962


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l4139_413966

theorem arithmetic_simplification :
  (100 - 25 * 4 = 0) ∧
  (20 / 5 * 2 = 8) ∧
  (360 - 200 / 4 = 310) ∧
  (36 / 3 + 27 = 39) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l4139_413966


namespace NUMINAMATH_CALUDE_equation_solution_l4139_413907

theorem equation_solution : 
  ∀ x : ℝ, (x - 1) * (x + 1) = x - 1 ↔ x = 0 ∨ x = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4139_413907


namespace NUMINAMATH_CALUDE_min_operations_for_square_l4139_413991

-- Define the points
variable (A B C D : Point)

-- Define the operations
def measure_distance (P Q : Point) : ℝ := sorry
def compare_numbers (x y : ℝ) : Bool := sorry

-- Define what it means for ABCD to be a square
def is_square (A B C D : Point) : Prop :=
  let AB := measure_distance A B
  let BC := measure_distance B C
  let CD := measure_distance C D
  let DA := measure_distance D A
  let AC := measure_distance A C
  let BD := measure_distance B D
  (AB = BC) ∧ (BC = CD) ∧ (CD = DA) ∧ (AC = BD)

-- The theorem to prove
theorem min_operations_for_square (A B C D : Point) :
  ∃ (n : ℕ), n = 7 ∧ 
  (∀ (m : ℕ), m < n → ¬∃ (algorithm : Unit → Bool), 
    (algorithm () = true ↔ is_square A B C D)) :=
sorry

end NUMINAMATH_CALUDE_min_operations_for_square_l4139_413991


namespace NUMINAMATH_CALUDE_gangster_undetected_speed_l4139_413965

/-- Represents the speed of a moving object -/
structure Speed :=
  (value : ℝ)

/-- Represents the distance between two points -/
structure Distance :=
  (value : ℝ)

/-- Represents a moving police officer -/
structure PoliceOfficer :=
  (speed : Speed)
  (spacing : Distance)

/-- Represents a moving gangster -/
structure Gangster :=
  (speed : Speed)

/-- Determines if a gangster is undetected by police officers -/
def is_undetected (g : Gangster) (p : PoliceOfficer) : Prop :=
  (g.speed.value = 2 * p.speed.value) ∨ (g.speed.value = p.speed.value / 2)

/-- Theorem stating the conditions for a gangster to remain undetected -/
theorem gangster_undetected_speed (v : ℝ) (a : ℝ) :
  ∀ (g : Gangster) (p : PoliceOfficer),
  p.speed.value = v →
  p.spacing.value = 9 * a →
  is_undetected g p :=
sorry

end NUMINAMATH_CALUDE_gangster_undetected_speed_l4139_413965


namespace NUMINAMATH_CALUDE_no_solution_to_exponential_equation_l4139_413954

theorem no_solution_to_exponential_equation :
  ¬∃ (x y : ℝ), (9 : ℝ) ^ (x^3 + y) + (9 : ℝ) ^ (x + y^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_exponential_equation_l4139_413954


namespace NUMINAMATH_CALUDE_two_bedroom_units_l4139_413939

theorem two_bedroom_units (total_units : ℕ) (one_bedroom_cost two_bedroom_cost : ℕ) (total_cost : ℕ) :
  total_units = 12 →
  one_bedroom_cost = 360 →
  two_bedroom_cost = 450 →
  total_cost = 4950 →
  ∃ (one_bedroom_count two_bedroom_count : ℕ),
    one_bedroom_count + two_bedroom_count = total_units ∧
    one_bedroom_count * one_bedroom_cost + two_bedroom_count * two_bedroom_cost = total_cost ∧
    two_bedroom_count = 7 :=
by sorry

end NUMINAMATH_CALUDE_two_bedroom_units_l4139_413939


namespace NUMINAMATH_CALUDE_concert_attendance_theorem_l4139_413960

/-- Represents the relationship between number of attendees and ticket price -/
structure ConcertAttendance where
  n : ℕ  -- number of attendees
  t : ℕ  -- ticket price in dollars
  k : ℕ  -- constant of proportionality
  h : n * t = k  -- inverse proportionality relationship

/-- Given initial conditions and final ticket price, calculates the final number of attendees -/
def calculate_attendance (initial : ConcertAttendance) (final_price : ℕ) : ℕ :=
  initial.k / final_price

theorem concert_attendance_theorem (initial : ConcertAttendance) 
    (h1 : initial.n = 300) 
    (h2 : initial.t = 50) 
    (h3 : calculate_attendance initial 75 = 200) : 
  calculate_attendance initial 75 = 200 := by
  sorry

#check concert_attendance_theorem

end NUMINAMATH_CALUDE_concert_attendance_theorem_l4139_413960


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l4139_413986

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - 2*p^2 - p + 3 = 0) →
  (q^3 - 2*q^2 - q + 3 = 0) →
  (r^3 - 2*r^2 - r + 3 = 0) →
  (p ≠ q) → (q ≠ r) → (p ≠ r) →
  1/(p-2) + 1/(q-2) + 1/(r-2) = -3 := by
sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l4139_413986


namespace NUMINAMATH_CALUDE_upstream_distance_l4139_413998

/-- Calculates the upstream distance swam by a man given his speed in still water,
    downstream distance, and time spent swimming both downstream and upstream. -/
theorem upstream_distance
  (still_speed : ℝ)
  (downstream_distance : ℝ)
  (time : ℝ)
  (h1 : still_speed = 5.5)
  (h2 : downstream_distance = 35)
  (h3 : time = 5) :
  let stream_speed := downstream_distance / time - still_speed
  let upstream_speed := still_speed - stream_speed
  upstream_speed * time = 20 := by
sorry

end NUMINAMATH_CALUDE_upstream_distance_l4139_413998


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l4139_413959

/-- Given a triangle ABC with angles B = 60°, C = 75°, and side a = 4,
    prove that side b = 2√6 -/
theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) : 
  B = π / 3 →  -- 60° in radians
  C = 5 * π / 12 →  -- 75° in radians
  a = 4 →
  A + B + C = π →  -- Sum of angles in a triangle
  a / Real.sin A = b / Real.sin B →  -- Law of Sines
  b = 2 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l4139_413959
