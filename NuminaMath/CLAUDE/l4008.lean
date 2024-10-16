import Mathlib

namespace NUMINAMATH_CALUDE_student_in_first_vehicle_probability_l4008_400806

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of vehicles -/
def num_vehicles : ℕ := 2

/-- The number of seats in each vehicle -/
def seats_per_vehicle : ℕ := 2

/-- The probability that a specific student is in the first vehicle -/
def prob_student_in_first_vehicle : ℚ := 1/2

theorem student_in_first_vehicle_probability :
  prob_student_in_first_vehicle = 1/2 := by sorry

end NUMINAMATH_CALUDE_student_in_first_vehicle_probability_l4008_400806


namespace NUMINAMATH_CALUDE_book_to_bookmark_ratio_l4008_400805

def books : ℕ := 72
def bookmarks : ℕ := 16

theorem book_to_bookmark_ratio : 
  (books / (Nat.gcd books bookmarks)) / (bookmarks / (Nat.gcd books bookmarks)) = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_book_to_bookmark_ratio_l4008_400805


namespace NUMINAMATH_CALUDE_square_sum_of_roots_l4008_400855

theorem square_sum_of_roots (r s : ℝ) (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_roots_l4008_400855


namespace NUMINAMATH_CALUDE_range_of_k_l4008_400895

theorem range_of_k (k : ℝ) : 
  (∀ a b : ℝ, a^2 + b^2 ≥ 2*k*a*b) → k ∈ Set.Icc (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_k_l4008_400895


namespace NUMINAMATH_CALUDE_stock_worth_l4008_400891

/-- The total worth of a stock given specific sales conditions and overall loss -/
theorem stock_worth (stock : ℝ) : 
  (0.2 * stock * 1.1 + 0.8 * stock * 0.95 = stock - 450) → 
  stock = 22500 := by
sorry

end NUMINAMATH_CALUDE_stock_worth_l4008_400891


namespace NUMINAMATH_CALUDE_scientific_calculator_cost_l4008_400825

theorem scientific_calculator_cost
  (total_cost : ℕ)
  (num_scientific : ℕ)
  (num_graphing : ℕ)
  (graphing_cost : ℕ)
  (h1 : total_cost = 1625)
  (h2 : num_scientific = 20)
  (h3 : num_graphing = 25)
  (h4 : graphing_cost = 57)
  (h5 : num_scientific + num_graphing = 45) :
  ∃ (scientific_cost : ℕ),
    scientific_cost * num_scientific + graphing_cost * num_graphing = total_cost ∧
    scientific_cost = 10 :=
by sorry

end NUMINAMATH_CALUDE_scientific_calculator_cost_l4008_400825


namespace NUMINAMATH_CALUDE_quadratic_function_value_l4008_400883

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_value (a b c : ℝ) :
  f a b c (Real.sqrt 2) = 3 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, |f a b c x| ≤ 1) →
  f a b c (Real.sqrt 2013) = 1343.67 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_value_l4008_400883


namespace NUMINAMATH_CALUDE_max_value_of_p_l4008_400835

theorem max_value_of_p (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x * y * z + x + z = y) :
  ∃ (p : ℝ), p = 2 / (x^2 + 1) - 2 / (y^2 + 1) + 3 / (z^2 + 1) ∧
  p ≤ 10 / 3 ∧
  ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧
    x' * y' * z' + x' + z' = y' ∧
    2 / (x'^2 + 1) - 2 / (y'^2 + 1) + 3 / (z'^2 + 1) = 10 / 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_p_l4008_400835


namespace NUMINAMATH_CALUDE_ellipse_focus_k_value_l4008_400862

/-- An ellipse with equation 5x^2 + ky^2 = 5 and one focus at (0, 2) has k = 1 -/
theorem ellipse_focus_k_value (k : ℝ) :
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    ∀ (x y : ℝ), 5 * x^2 + k * y^2 = 5 ↔
      (x^2 / a^2 + y^2 / b^2 = 1 ∧
       c^2 = a^2 - b^2 ∧
       2^2 = c^2)) →
  k = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_k_value_l4008_400862


namespace NUMINAMATH_CALUDE_bookcase_weight_excess_l4008_400847

/-- Proves that the total weight of books and knick-knacks exceeds the bookcase weight limit by 33 pounds -/
theorem bookcase_weight_excess :
  let bookcase_limit : ℕ := 80
  let hardcover_count : ℕ := 70
  let hardcover_weight : ℚ := 1/2
  let textbook_count : ℕ := 30
  let textbook_weight : ℕ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℕ := 6
  let total_weight := (hardcover_count : ℚ) * hardcover_weight + 
                      (textbook_count * textbook_weight : ℚ) + 
                      (knickknack_count * knickknack_weight : ℚ)
  total_weight - bookcase_limit = 33
  := by sorry

end NUMINAMATH_CALUDE_bookcase_weight_excess_l4008_400847


namespace NUMINAMATH_CALUDE_union_of_A_and_B_intersection_of_complements_l4008_400860

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - x - 2 ≥ 0}
def B : Set ℝ := {x | |2*x + 1| ≤ 1}

-- Define the union of A and B
def AUnionB : Set ℝ := {x | x ≤ 0 ∨ x ≥ 2}

-- Define the intersection of complements of A and B
def ACompIntBComp : Set ℝ := {x | 0 < x ∧ x < 2}

-- Theorem statements
theorem union_of_A_and_B : A ∪ B = AUnionB := by sorry

theorem intersection_of_complements : Aᶜ ∩ Bᶜ = ACompIntBComp := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_intersection_of_complements_l4008_400860


namespace NUMINAMATH_CALUDE_cake_slices_kept_l4008_400800

theorem cake_slices_kept (total_slices : ℕ) (eaten_fraction : ℚ) (extra_eaten : ℕ) : 
  total_slices = 35 →
  eaten_fraction = 2/5 →
  extra_eaten = 3 →
  total_slices - (eaten_fraction * total_slices + extra_eaten) = 18 :=
by sorry

end NUMINAMATH_CALUDE_cake_slices_kept_l4008_400800


namespace NUMINAMATH_CALUDE_largest_number_l4008_400851

theorem largest_number (a b c d e : ℝ) 
  (ha : a = 0.998) 
  (hb : b = 0.989) 
  (hc : c = 0.999) 
  (hd : d = 0.990) 
  (he : e = 0.980) : 
  c ≥ a ∧ c ≥ b ∧ c ≥ d ∧ c ≥ e := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l4008_400851


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l4008_400845

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.tan (60 * π / 180) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l4008_400845


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_and_average_l4008_400829

theorem consecutive_integers_sum_and_average (n : ℤ) :
  let consecutive_integers := [n+1, n+2, n+3, n+4, n+5, n+6]
  (consecutive_integers.sum = 6*n + 21) ∧ 
  (consecutive_integers.sum / 6 : ℚ) = n + (21 : ℚ) / 6 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_and_average_l4008_400829


namespace NUMINAMATH_CALUDE_heaviest_tv_weight_difference_l4008_400850

-- Define the dimensions and weight ratios of the TVs
def bill_width : ℝ := 48
def bill_height : ℝ := 100
def bill_weight_ratio : ℝ := 4

def bob_width : ℝ := 70
def bob_height : ℝ := 60
def bob_weight_ratio : ℝ := 3.5

def steve_width : ℝ := 84
def steve_height : ℝ := 92
def steve_weight_ratio : ℝ := 4.5

-- Define the conversion factor from ounces to pounds
def oz_to_lb : ℝ := 16

-- Theorem to prove
theorem heaviest_tv_weight_difference : 
  let bill_area := bill_width * bill_height
  let bob_area := bob_width * bob_height
  let steve_area := steve_width * steve_height
  
  let bill_weight := bill_area * bill_weight_ratio / oz_to_lb
  let bob_weight := bob_area * bob_weight_ratio / oz_to_lb
  let steve_weight := steve_area * steve_weight_ratio / oz_to_lb
  
  let heaviest_weight := max bill_weight (max bob_weight steve_weight)
  let combined_weight := bill_weight + bob_weight
  
  heaviest_weight - combined_weight = 54.75 := by
  sorry

end NUMINAMATH_CALUDE_heaviest_tv_weight_difference_l4008_400850


namespace NUMINAMATH_CALUDE_books_remaining_l4008_400890

theorem books_remaining (initial_books given_away : ℝ) 
  (h1 : initial_books = 54.0)
  (h2 : given_away = 23.0) : 
  initial_books - given_away = 31.0 := by
sorry

end NUMINAMATH_CALUDE_books_remaining_l4008_400890


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l4008_400840

theorem cube_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq : a + b = c + d) (square_sum_gt : a^2 + b^2 > c^2 + d^2) :
  a^3 + b^3 > c^3 + d^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l4008_400840


namespace NUMINAMATH_CALUDE_total_snakes_count_l4008_400869

/-- Represents the total population in the neighborhood -/
def total_population : ℕ := 200

/-- Represents the percentage of people who own only snakes -/
def only_snakes_percent : ℚ := 5 / 100

/-- Represents the percentage of people who own both cats and snakes, but no other pets -/
def cats_and_snakes_percent : ℚ := 4 / 100

/-- Represents the percentage of people who own both snakes and rabbits, but no other pets -/
def snakes_and_rabbits_percent : ℚ := 5 / 100

/-- Represents the percentage of people who own both snakes and birds, but no other pets -/
def snakes_and_birds_percent : ℚ := 3 / 100

/-- Represents the percentage of exotic pet owners who also own snakes -/
def exotic_and_snakes_percent : ℚ := 25 / 100

/-- Represents the total percentage of exotic pet owners -/
def total_exotic_percent : ℚ := 34 / 100

/-- Calculates the total percentage of snake owners in the neighborhood -/
def total_snake_owners_percent : ℚ :=
  only_snakes_percent + cats_and_snakes_percent + snakes_and_rabbits_percent + 
  snakes_and_birds_percent + (exotic_and_snakes_percent * total_exotic_percent)

/-- Theorem stating that the total number of snakes in the neighborhood is 51 -/
theorem total_snakes_count : ⌊(total_snake_owners_percent * total_population : ℚ)⌋ = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_snakes_count_l4008_400869


namespace NUMINAMATH_CALUDE_complement_of_union_l4008_400865

def U : Set ℤ := {x | 0 < x ∧ x ≤ 8}
def M : Set ℤ := {1, 3, 5, 7}
def N : Set ℤ := {5, 6, 7}

theorem complement_of_union :
  (U \ (M ∪ N)) = {2, 4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l4008_400865


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_small_prime_factors_l4008_400897

/-- The greatest prime factor of a positive integer n > 1 -/
noncomputable def greatestPrimeFactor (n : ℕ) : ℕ := sorry

/-- Check if three numbers form an arithmetic progression -/
def isArithmeticProgression (x y z : ℕ) : Prop :=
  y - x = z - y

/-- Main theorem -/
theorem arithmetic_progression_with_small_prime_factors
  (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (hap : isArithmeticProgression x y z)
  (hprime : greatestPrimeFactor (x * y * z) ≤ 3) :
  ∃ (l a b : ℕ), (a ≥ 0 ∧ b ≥ 0) ∧ l = 2^a * 3^b ∧
    ((x, y, z) = (l, 2*l, 3*l) ∨
     (x, y, z) = (2*l, 3*l, 4*l) ∨
     (x, y, z) = (2*l, 9*l, 16*l)) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_small_prime_factors_l4008_400897


namespace NUMINAMATH_CALUDE_not_divides_power_minus_one_l4008_400888

theorem not_divides_power_minus_one (n : ℕ) (hn : n > 1) : ¬(n ∣ 2^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_minus_one_l4008_400888


namespace NUMINAMATH_CALUDE_b_invests_after_six_months_l4008_400893

/-- A partnership with three investors A, B, and C -/
structure Partnership where
  x : ℝ  -- A's investment
  m : ℝ  -- Months after which B invests
  total_gain : ℝ  -- Total annual gain
  a_share : ℝ  -- A's share of the gain

/-- The conditions of the partnership -/
def partnership_conditions (p : Partnership) : Prop :=
  p.total_gain = 24000 ∧ 
  p.a_share = 8000 ∧ 
  0 < p.x ∧ 
  0 < p.m ∧ 
  p.m < 12

/-- The theorem stating that B invests after 6 months -/
theorem b_invests_after_six_months (p : Partnership) 
  (h : partnership_conditions p) : p.m = 6 := by
  sorry


end NUMINAMATH_CALUDE_b_invests_after_six_months_l4008_400893


namespace NUMINAMATH_CALUDE_eighteenth_term_of_sequence_l4008_400873

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem eighteenth_term_of_sequence : arithmetic_sequence 3 4 18 = 71 := by
  sorry

end NUMINAMATH_CALUDE_eighteenth_term_of_sequence_l4008_400873


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l4008_400816

theorem cubic_root_ratio (a b c d : ℝ) (h : ∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = -2 ∨ x = -3) : 
  c / d = -11 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l4008_400816


namespace NUMINAMATH_CALUDE_no_real_solutions_l4008_400870

theorem no_real_solutions (x : ℝ) :
  x ≠ -1 → (x^2 + x + 1) / (x + 1) ≠ x^2 + 5*x + 6 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l4008_400870


namespace NUMINAMATH_CALUDE_cereal_box_capacity_l4008_400814

theorem cereal_box_capacity (cups_per_serving : ℕ) (total_servings : ℕ) : 
  cups_per_serving = 2 → total_servings = 9 → cups_per_serving * total_servings = 18 := by
  sorry

end NUMINAMATH_CALUDE_cereal_box_capacity_l4008_400814


namespace NUMINAMATH_CALUDE_equally_spaced_number_line_l4008_400838

theorem equally_spaced_number_line (total_distance : ℝ) (num_steps : ℕ) (step_to_z : ℕ) : 
  total_distance = 16 → num_steps = 4 → step_to_z = 2 →
  let step_length := total_distance / num_steps
  let z := step_to_z * step_length
  z = 8 := by
  sorry

end NUMINAMATH_CALUDE_equally_spaced_number_line_l4008_400838


namespace NUMINAMATH_CALUDE_binary_10101_equals_21_l4008_400836

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_10101_equals_21 :
  binary_to_decimal [true, false, true, false, true] = 21 := by
  sorry

end NUMINAMATH_CALUDE_binary_10101_equals_21_l4008_400836


namespace NUMINAMATH_CALUDE_range_of_a_given_decreasing_function_l4008_400842

-- Define a decreasing function on the real line
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- State the theorem
theorem range_of_a_given_decreasing_function (f : ℝ → ℝ) (h : DecreasingFunction f) :
  ∀ a : ℝ, a ∈ Set.univ :=
by
  sorry

end NUMINAMATH_CALUDE_range_of_a_given_decreasing_function_l4008_400842


namespace NUMINAMATH_CALUDE_sum_of_three_nines_power_twenty_l4008_400834

theorem sum_of_three_nines_power_twenty (n : ℕ) : 9^20 + 9^20 + 9^20 = 3^41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_nines_power_twenty_l4008_400834


namespace NUMINAMATH_CALUDE_our_circle_center_and_radius_l4008_400827

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def Circle.center (c : Circle) : ℝ × ℝ := sorry

/-- The radius of a circle -/
def Circle.radius (c : Circle) : ℝ := sorry

/-- Our specific circle -/
def our_circle : Circle :=
  { equation := λ x y => x^2 + y^2 - 2*x - 3 = 0 }

theorem our_circle_center_and_radius :
  Circle.center our_circle = (1, 0) ∧ Circle.radius our_circle = 2 := by
  sorry

end NUMINAMATH_CALUDE_our_circle_center_and_radius_l4008_400827


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l4008_400882

-- Define a geometric sequence
def is_geometric_sequence (b₁ b₂ b₃ : ℝ) : Prop :=
  ∃ q : ℝ, b₂ = b₁ * q ∧ b₃ = b₂ * q

theorem geometric_sequence_properties :
  -- There exist real numbers b₁, b₂, b₃ forming a geometric sequence such that b₁ < b₂ and b₂ > b₃
  (∃ b₁ b₂ b₃ : ℝ, is_geometric_sequence b₁ b₂ b₃ ∧ b₁ < b₂ ∧ b₂ > b₃) ∧
  -- If b₁ * b₂ < 0, then b₂ * b₃ < 0 for any geometric sequence b₁, b₂, b₃
  (∀ b₁ b₂ b₃ : ℝ, is_geometric_sequence b₁ b₂ b₃ → b₁ * b₂ < 0 → b₂ * b₃ < 0) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l4008_400882


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l4008_400876

-- Define the propositions p and q
def p (x y : ℝ) : Prop := x + y > 2 ∧ x * y > 1
def q (x y : ℝ) : Prop := x > 1 ∧ y > 1

-- Theorem statement
theorem p_necessary_not_sufficient :
  (∀ x y : ℝ, q x y → p x y) ∧
  ¬(∀ x y : ℝ, p x y → q x y) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l4008_400876


namespace NUMINAMATH_CALUDE_lose_sector_area_l4008_400856

/-- Given a circular spinner with radius 15 cm and a probability of winning of 1/3,
    the area of the LOSE sector is 150π sq cm. -/
theorem lose_sector_area (radius : ℝ) (win_prob : ℝ) (lose_area : ℝ) : 
  radius = 15 → 
  win_prob = 1/3 → 
  lose_area = 150 * Real.pi → 
  lose_area = (1 - win_prob) * Real.pi * radius^2 := by
sorry

end NUMINAMATH_CALUDE_lose_sector_area_l4008_400856


namespace NUMINAMATH_CALUDE_angle_C_value_max_area_l4008_400808

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions of the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (2 * t.a + t.b) / t.c = cos (t.A + t.C) / cos t.C

-- Theorem 1: If the condition is satisfied, then C = 2π/3
theorem angle_C_value (t : Triangle) (h : satisfiesCondition t) : t.C = 2 * π / 3 := by
  sorry

-- Theorem 2: Maximum area when c = 2 and C = 2π/3
theorem max_area (t : Triangle) (h1 : t.c = 2) (h2 : t.C = 2 * π / 3) :
  ∃ (maxArea : ℝ), maxArea = Real.sqrt 3 / 3 ∧
  ∀ (s : ℝ), s = (1 / 2) * t.a * t.b * sin t.C → s ≤ maxArea := by
  sorry

end NUMINAMATH_CALUDE_angle_C_value_max_area_l4008_400808


namespace NUMINAMATH_CALUDE_total_weight_of_fruits_l4008_400846

-- Define the weight of oranges and apples
def orange_weight : ℚ := 24 / 12
def apple_weight : ℚ := 30 / 8

-- Define the number of bags for each fruit
def orange_bags : ℕ := 5
def apple_bags : ℕ := 4

-- Theorem to prove
theorem total_weight_of_fruits :
  orange_bags * orange_weight + apple_bags * apple_weight = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_fruits_l4008_400846


namespace NUMINAMATH_CALUDE_stream_speed_l4008_400844

/-- The speed of a stream given downstream and upstream speeds -/
theorem stream_speed (downstream_speed upstream_speed : ℝ) :
  downstream_speed = 15 →
  upstream_speed = 8 →
  (downstream_speed - upstream_speed) / 2 = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l4008_400844


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l4008_400810

theorem quadratic_inequalities (a : ℝ) :
  ((∀ x : ℝ, x^2 + a*x + 3 ≥ a) ↔ a ∈ Set.Icc (-6) 2) ∧
  ((∃ x : ℝ, x < 1 ∧ x^2 + a*x + 3 ≤ a) ↔ a ∈ Set.Ici 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l4008_400810


namespace NUMINAMATH_CALUDE_program_output_equals_b_l4008_400863

def program (a b : ℕ) : ℕ :=
  if a > b then a else b

theorem program_output_equals_b :
  let a : ℕ := 2
  let b : ℕ := 3
  program a b = b := by sorry

end NUMINAMATH_CALUDE_program_output_equals_b_l4008_400863


namespace NUMINAMATH_CALUDE_new_person_weight_l4008_400867

/-- Given a group of 10 people, if replacing one person weighing 65 kg
    with a new person increases the average weight by 7.2 kg,
    then the weight of the new person is 137 kg. -/
theorem new_person_weight
  (n : ℕ)
  (initial_weight : ℝ)
  (weight_increase : ℝ)
  (replaced_weight : ℝ)
  (h1 : n = 10)
  (h2 : weight_increase = 7.2)
  (h3 : replaced_weight = 65)
  : initial_weight + n * weight_increase = initial_weight - replaced_weight + 137 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l4008_400867


namespace NUMINAMATH_CALUDE_paula_candy_distribution_l4008_400861

def minimum_candies (initial_candies : ℕ) (num_friends : ℕ) : ℕ :=
  let total := initial_candies + (num_friends - initial_candies % num_friends) % num_friends
  total

theorem paula_candy_distribution (initial_candies : ℕ) (num_friends : ℕ) 
  (h1 : initial_candies = 20) (h2 : num_friends = 10) :
  minimum_candies initial_candies num_friends = 30 ∧
  minimum_candies initial_candies num_friends / num_friends = 3 :=
by sorry

end NUMINAMATH_CALUDE_paula_candy_distribution_l4008_400861


namespace NUMINAMATH_CALUDE_thank_you_cards_percentage_l4008_400884

/-- The percentage of students who gave thank you cards to Ms. Jones -/
def percentage_thank_you_cards (
  total_students : ℕ)
  (gift_card_value : ℚ)
  (total_gift_card_amount : ℚ)
  (gift_card_fraction : ℚ) : ℚ :=
  (total_gift_card_amount / gift_card_value / gift_card_fraction) / total_students * 100

/-- Theorem stating that 30% of Ms. Jones' class gave her thank you cards -/
theorem thank_you_cards_percentage :
  percentage_thank_you_cards 50 10 50 (1/3) = 30 := by
  sorry

end NUMINAMATH_CALUDE_thank_you_cards_percentage_l4008_400884


namespace NUMINAMATH_CALUDE_remainder_proof_l4008_400858

/-- The largest integer n such that 5^n divides 12^2015 + 13^2015 -/
def n : ℕ := 3

/-- The theorem statement -/
theorem remainder_proof :
  (12^2015 + 13^2015) / 5^n % 1000 = 625 :=
sorry

end NUMINAMATH_CALUDE_remainder_proof_l4008_400858


namespace NUMINAMATH_CALUDE_student_grouping_l4008_400857

theorem student_grouping (total_students : ℕ) (students_per_group : ℕ) (h1 : total_students = 30) (h2 : students_per_group = 5) :
  total_students / students_per_group = 6 := by
  sorry

end NUMINAMATH_CALUDE_student_grouping_l4008_400857


namespace NUMINAMATH_CALUDE_unique_solution_l4008_400826

theorem unique_solution (a b c : ℕ+) 
  (eq1 : b = a^2 - a)
  (eq2 : c = b^2 - b)
  (eq3 : a = c^2 - c) : 
  a = 2 ∧ b = 2 ∧ c = 2 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_l4008_400826


namespace NUMINAMATH_CALUDE_disjunction_true_false_l4008_400866

theorem disjunction_true_false (p q : Prop) (hp : p) (hq : ¬q) : p ∨ q := by
  sorry

end NUMINAMATH_CALUDE_disjunction_true_false_l4008_400866


namespace NUMINAMATH_CALUDE_books_remaining_l4008_400811

theorem books_remaining (initial_books yard_sale_books day1_books day2_books day3_books : ℕ) :
  initial_books = 75 →
  yard_sale_books = 33 →
  day1_books = 15 →
  day2_books = 8 →
  day3_books = 12 →
  initial_books - (yard_sale_books + day1_books + day2_books + day3_books) = 7 :=
by sorry

end NUMINAMATH_CALUDE_books_remaining_l4008_400811


namespace NUMINAMATH_CALUDE_same_color_marble_probability_same_color_marble_probability_value_l4008_400898

/-- The probability of drawing three marbles of the same color from a bag containing
    5 red marbles, 7 white marbles, and 4 green marbles, without replacement. -/
theorem same_color_marble_probability : ℚ :=
  let total_marbles : ℕ := 5 + 7 + 4
  let red_marbles : ℕ := 5
  let white_marbles : ℕ := 7
  let green_marbles : ℕ := 4
  let prob_all_red : ℚ := (red_marbles * (red_marbles - 1) * (red_marbles - 2)) /
    (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  let prob_all_white : ℚ := (white_marbles * (white_marbles - 1) * (white_marbles - 2)) /
    (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  let prob_all_green : ℚ := (green_marbles * (green_marbles - 1) * (green_marbles - 2)) /
    (total_marbles * (total_marbles - 1) * (total_marbles - 2))
  prob_all_red + prob_all_white + prob_all_green

theorem same_color_marble_probability_value :
  same_color_marble_probability = 43 / 280 := by
  sorry

end NUMINAMATH_CALUDE_same_color_marble_probability_same_color_marble_probability_value_l4008_400898


namespace NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l4008_400837

theorem largest_prime_divisor_to_test (n : ℕ) (h : 500 ≤ n ∧ n ≤ 550) :
  (∀ p : ℕ, p.Prime → p ≤ 23 → ¬(p ∣ n)) → n.Prime ∨ n = 1 :=
sorry

end NUMINAMATH_CALUDE_largest_prime_divisor_to_test_l4008_400837


namespace NUMINAMATH_CALUDE_divisible_by_five_l4008_400820

theorem divisible_by_five (n : ℕ) : 5 ∣ (2^(4*n+1) + 3^(4*n+1)) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l4008_400820


namespace NUMINAMATH_CALUDE_circle_area_ratio_after_tripling_diameter_l4008_400887

theorem circle_area_ratio_after_tripling_diameter :
  ∀ (r : ℝ), r > 0 →
  (π * r^2) / (π * (3*r)^2) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_after_tripling_diameter_l4008_400887


namespace NUMINAMATH_CALUDE_prove_strawberry_basket_price_l4008_400853

def strawberry_plants : ℕ := 5
def tomato_plants : ℕ := 7
def strawberries_per_plant : ℕ := 14
def tomatoes_per_plant : ℕ := 16
def fruits_per_basket : ℕ := 7
def tomato_basket_price : ℕ := 6
def total_revenue : ℕ := 186

def strawberry_basket_price : ℕ := 9

theorem prove_strawberry_basket_price :
  let total_strawberries := strawberry_plants * strawberries_per_plant
  let total_tomatoes := tomato_plants * tomatoes_per_plant
  let strawberry_baskets := total_strawberries / fruits_per_basket
  let tomato_baskets := total_tomatoes / fruits_per_basket
  let tomato_revenue := tomato_baskets * tomato_basket_price
  let strawberry_revenue := total_revenue - tomato_revenue
  strawberry_basket_price = strawberry_revenue / strawberry_baskets := by
  sorry

#eval strawberry_basket_price

end NUMINAMATH_CALUDE_prove_strawberry_basket_price_l4008_400853


namespace NUMINAMATH_CALUDE_baba_yaga_journey_l4008_400896

/-- The problem of Baba Yaga's journey to Bald Mountain -/
theorem baba_yaga_journey 
  (arrival_time : ℕ) 
  (slow_speed : ℕ) 
  (fast_speed : ℕ) 
  (late_hours : ℕ) 
  (early_hours : ℕ) 
  (h : arrival_time = 24) -- Midnight is represented as 24
  (h_slow : slow_speed = 50)
  (h_fast : fast_speed = 150)
  (h_late : late_hours = 2)
  (h_early : early_hours = 2)
  : ∃ (departure_time speed : ℕ),
    departure_time = 20 ∧ 
    speed = 75 ∧
    (arrival_time - departure_time) * speed = 
      (arrival_time - departure_time + late_hours) * slow_speed ∧
    (arrival_time - departure_time) * speed = 
      (arrival_time - departure_time - early_hours) * fast_speed :=
sorry

end NUMINAMATH_CALUDE_baba_yaga_journey_l4008_400896


namespace NUMINAMATH_CALUDE_line_through_points_l4008_400824

-- Define a structure for points
structure Point where
  x : ℝ
  y : ℝ

-- Define the line passing through the given points
def line_equation (x : ℝ) : ℝ := 3 * x + 2

-- Define the given points
def p1 : Point := ⟨2, 8⟩
def p2 : Point := ⟨4, 14⟩
def p3 : Point := ⟨6, 20⟩
def p4 : Point := ⟨35, line_equation 35⟩

-- Theorem statement
theorem line_through_points :
  p1.y = line_equation p1.x ∧
  p2.y = line_equation p2.x ∧
  p3.y = line_equation p3.x ∧
  p4.y = 107 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l4008_400824


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2023_l4008_400872

theorem smallest_prime_factor_of_2023 : Nat.minFac 2023 = 7 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2023_l4008_400872


namespace NUMINAMATH_CALUDE_borrowed_sum_calculation_l4008_400859

/-- Proves that given a sum of money borrowed at 6% per annum simple interest, 
    if the interest after 6 years is Rs. 672 less than the borrowed sum, 
    then the borrowed sum is Rs. 1050. -/
theorem borrowed_sum_calculation (P : ℝ) : 
  (P * 0.06 * 6 = P - 672) → P = 1050 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sum_calculation_l4008_400859


namespace NUMINAMATH_CALUDE_cat_shelter_ratio_l4008_400892

theorem cat_shelter_ratio : 
  ∀ (initial_cats replacement_cats adopted_cats dogs : ℕ),
    initial_cats = 15 →
    adopted_cats = initial_cats / 3 →
    initial_cats = initial_cats - adopted_cats + replacement_cats →
    dogs = 2 * initial_cats →
    initial_cats + dogs + replacement_cats = 60 →
    replacement_cats / adopted_cats = 3 ∧ adopted_cats ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_cat_shelter_ratio_l4008_400892


namespace NUMINAMATH_CALUDE_largest_two_digit_product_l4008_400881

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_single_digit (x : ℕ) : Prop := 1 ≤ x ∧ x ≤ 9

theorem largest_two_digit_product :
  ∃ (n x : ℕ), 
    is_two_digit n ∧
    is_single_digit x ∧
    n = x * (10 * x + 2 * x) ∧
    ∀ (m y : ℕ), 
      is_two_digit m → 
      is_single_digit y → 
      m = y * (10 * y + 2 * y) → 
      m ≤ n ∧
    n = 48 :=
sorry

end NUMINAMATH_CALUDE_largest_two_digit_product_l4008_400881


namespace NUMINAMATH_CALUDE_rectangle_not_cuttable_from_square_cannot_cut_rectangle_from_square_l4008_400809

/-- Proves that a rectangle with area 30 and length-to-width ratio 2:1 cannot be cut from a square with area 36 -/
theorem rectangle_not_cuttable_from_square : 
  ∀ (rect_length rect_width square_side : ℝ),
  rect_length > 0 → rect_width > 0 → square_side > 0 →
  rect_length * rect_width = 30 →
  rect_length = 2 * rect_width →
  square_side * square_side = 36 →
  rect_length > square_side :=
by sorry

/-- Concludes that the rectangular piece cannot be cut from the square piece -/
theorem cannot_cut_rectangle_from_square : 
  ∃ (rect_length rect_width square_side : ℝ),
  rect_length > 0 ∧ rect_width > 0 ∧ square_side > 0 ∧
  rect_length * rect_width = 30 ∧
  rect_length = 2 * rect_width ∧
  square_side * square_side = 36 ∧
  rect_length > square_side :=
by sorry

end NUMINAMATH_CALUDE_rectangle_not_cuttable_from_square_cannot_cut_rectangle_from_square_l4008_400809


namespace NUMINAMATH_CALUDE_sqrt_expression_defined_l4008_400843

theorem sqrt_expression_defined (a : ℝ) : 
  (∀ x : ℝ, (a + 1) * x^2 - 2 * (a - 1) * x + 3 * a - 3 ≥ 0) ↔ a ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sqrt_expression_defined_l4008_400843


namespace NUMINAMATH_CALUDE_hotel_air_conditioning_l4008_400830

theorem hotel_air_conditioning (total_rooms : ℝ) (total_rooms_pos : 0 < total_rooms) : 
  let rented_rooms := (3/4 : ℝ) * total_rooms
  let air_conditioned_rooms := (3/5 : ℝ) * total_rooms
  let rented_air_conditioned := (2/3 : ℝ) * air_conditioned_rooms
  let not_rented_rooms := total_rooms - rented_rooms
  let not_rented_air_conditioned := air_conditioned_rooms - rented_air_conditioned
  (not_rented_air_conditioned / not_rented_rooms) * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_hotel_air_conditioning_l4008_400830


namespace NUMINAMATH_CALUDE_length_breadth_difference_is_ten_l4008_400877

/-- Represents a rectangular plot with given dimensions and fencing costs. -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fenceCostPerMeter : ℝ
  totalFenceCost : ℝ

/-- Calculates the difference between length and breadth of the plot. -/
def lengthBreadthDifference (plot : RectangularPlot) : ℝ :=
  plot.length - plot.breadth

/-- Theorem stating that for a rectangular plot with length 55 meters,
    where the cost of fencing at Rs. 26.50 per meter totals Rs. 5300,
    the length is 10 meters more than the breadth. -/
theorem length_breadth_difference_is_ten
  (plot : RectangularPlot)
  (h1 : plot.length = 55)
  (h2 : plot.fenceCostPerMeter = 26.5)
  (h3 : plot.totalFenceCost = 5300)
  (h4 : plot.totalFenceCost = plot.fenceCostPerMeter * (2 * (plot.length + plot.breadth))) :
  lengthBreadthDifference plot = 10 := by
  sorry

#eval lengthBreadthDifference { length := 55, breadth := 45, fenceCostPerMeter := 26.5, totalFenceCost := 5300 }

end NUMINAMATH_CALUDE_length_breadth_difference_is_ten_l4008_400877


namespace NUMINAMATH_CALUDE_problem_solution_l4008_400812

theorem problem_solution (x y : ℚ) : 
  x / y = 12 / 5 → y = 25 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4008_400812


namespace NUMINAMATH_CALUDE_number_puzzle_l4008_400875

theorem number_puzzle : ∃ x : ℝ, x + (1/5) * x + 1 = 10 ∧ x = 7.5 := by sorry

end NUMINAMATH_CALUDE_number_puzzle_l4008_400875


namespace NUMINAMATH_CALUDE_least_possible_a_2000_l4008_400894

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ m n, m ∣ n → m < n → a m ∣ a n ∧ a m < a n

theorem least_possible_a_2000 (a : ℕ → ℕ) (h : sequence_property a) : a 2000 ≥ 128 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_a_2000_l4008_400894


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l4008_400804

theorem fraction_to_decimal (n : ℕ) (d : ℕ) (h : d = 2^3 * 5^7) :
  (n : ℚ) / d = 0.0006625 ↔ n = 53 :=
sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l4008_400804


namespace NUMINAMATH_CALUDE_calculation_proof_l4008_400839

theorem calculation_proof : 0.2 * 63 + 1.9 * 126 + 196 * 9 = 2016 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l4008_400839


namespace NUMINAMATH_CALUDE_no_common_real_solution_l4008_400832

theorem no_common_real_solution :
  ¬ ∃ (x y : ℝ), (x^2 + y^2 + 16 = 0) ∧ (x^2 - 3*y + 12 = 0) := by
  sorry

end NUMINAMATH_CALUDE_no_common_real_solution_l4008_400832


namespace NUMINAMATH_CALUDE_flower_bed_fraction_l4008_400813

theorem flower_bed_fraction (yard_length yard_width remainder_length_1 remainder_length_2 : ℝ)
  (h1 : yard_length = 30)
  (h2 : yard_width = 10)
  (h3 : remainder_length_1 = 30)
  (h4 : remainder_length_2 = 22)
  (h5 : yard_length = remainder_length_1) :
  let triangle_leg := (remainder_length_1 - remainder_length_2) / 2
  let triangle_area := triangle_leg ^ 2 / 2
  let total_triangle_area := 2 * triangle_area
  let yard_area := yard_length * yard_width
  total_triangle_area / yard_area = 4 / 75 := by
  sorry

end NUMINAMATH_CALUDE_flower_bed_fraction_l4008_400813


namespace NUMINAMATH_CALUDE_noemi_initial_amount_l4008_400864

/-- Calculates the initial amount of money Noemi had before gambling --/
def initial_amount (roulette_loss blackjack_loss poker_loss baccarat_loss purse_left : ℕ) : ℕ :=
  roulette_loss + blackjack_loss + poker_loss + baccarat_loss + purse_left

/-- Proves that Noemi's initial amount is correct given her losses and remaining money --/
theorem noemi_initial_amount : 
  initial_amount 600 800 400 700 1500 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_noemi_initial_amount_l4008_400864


namespace NUMINAMATH_CALUDE_shoes_cost_is_74_l4008_400868

-- Define the discount rate
def discount_rate : ℚ := 0.1

-- Define the cost of socks and bag
def socks_cost : ℚ := 2 * 2
def bag_cost : ℚ := 42

-- Define the discount threshold
def discount_threshold : ℚ := 100

-- Define the final payment amount
def final_payment : ℚ := 118

-- Theorem to prove
theorem shoes_cost_is_74 :
  ∃ (shoes_cost : ℚ),
    let total_cost := shoes_cost + socks_cost + bag_cost
    let discount := max (discount_rate * (total_cost - discount_threshold)) 0
    total_cost - discount = final_payment ∧ shoes_cost = 74 := by
  sorry

end NUMINAMATH_CALUDE_shoes_cost_is_74_l4008_400868


namespace NUMINAMATH_CALUDE_apple_profit_percentage_l4008_400828

/-- Calculates the total profit percentage for a stock of apples -/
theorem apple_profit_percentage 
  (total_stock : ℝ)
  (first_portion : ℝ)
  (second_portion : ℝ)
  (profit_rate : ℝ)
  (h1 : total_stock = 280)
  (h2 : first_portion = 0.4)
  (h3 : second_portion = 0.6)
  (h4 : profit_rate = 0.3)
  (h5 : first_portion + second_portion = 1) :
  let total_sp := (first_portion * total_stock * (1 + profit_rate)) + 
                  (second_portion * total_stock * (1 + profit_rate))
  let total_profit := total_sp - total_stock
  let total_profit_percentage := (total_profit / total_stock) * 100
  total_profit_percentage = 30 := by
sorry


end NUMINAMATH_CALUDE_apple_profit_percentage_l4008_400828


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l4008_400880

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ ∀ x, ¬ p x :=
by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l4008_400880


namespace NUMINAMATH_CALUDE_probability_of_dime_l4008_400879

theorem probability_of_dime (quarter_value nickel_value penny_value dime_value : ℚ)
  (total_quarter_value total_nickel_value total_penny_value total_dime_value : ℚ)
  (h1 : quarter_value = 25/100)
  (h2 : nickel_value = 5/100)
  (h3 : penny_value = 1/100)
  (h4 : dime_value = 10/100)
  (h5 : total_quarter_value = 15)
  (h6 : total_nickel_value = 5)
  (h7 : total_penny_value = 2)
  (h8 : total_dime_value = 12) :
  (total_dime_value / dime_value) / 
  ((total_quarter_value / quarter_value) + 
   (total_nickel_value / nickel_value) + 
   (total_penny_value / penny_value) + 
   (total_dime_value / dime_value)) = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_dime_l4008_400879


namespace NUMINAMATH_CALUDE_number_problem_l4008_400802

theorem number_problem (x : ℝ) : (0.7 * x - 40 = 30) → x = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l4008_400802


namespace NUMINAMATH_CALUDE_corrected_mean_calculation_l4008_400874

def original_mean : ℝ := 45
def num_observations : ℕ := 100
def incorrect_observations : List ℝ := [32, 12, 25]
def correct_observations : List ℝ := [67, 52, 85]

theorem corrected_mean_calculation :
  let original_sum := original_mean * num_observations
  let incorrect_sum := incorrect_observations.sum
  let correct_sum := correct_observations.sum
  let adjustment := correct_sum - incorrect_sum
  let corrected_sum := original_sum + adjustment
  let corrected_mean := corrected_sum / num_observations
  corrected_mean = 46.35 := by sorry

end NUMINAMATH_CALUDE_corrected_mean_calculation_l4008_400874


namespace NUMINAMATH_CALUDE_g_12_equals_191_l4008_400849

def g (n : ℕ) : ℕ := n^2 + 2*n + 23

theorem g_12_equals_191 : g 12 = 191 := by
  sorry

end NUMINAMATH_CALUDE_g_12_equals_191_l4008_400849


namespace NUMINAMATH_CALUDE_natural_roots_equation_l4008_400815

theorem natural_roots_equation :
  ∃ (x y z t : ℕ),
    17 * (x * y * z * t + x * y + x * t + z * t + 1) - 54 * (y * z * t + y + t) = 0 ∧
    x = 3 ∧ y = 5 ∧ z = 1 ∧ t = 2 :=
by sorry

end NUMINAMATH_CALUDE_natural_roots_equation_l4008_400815


namespace NUMINAMATH_CALUDE_circle_radius_eight_l4008_400854

/-- The equation of a potential circle -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + 14*x + y^2 + 8*y - k = 0

/-- The center of the potential circle -/
def circle_center : ℝ × ℝ := (-7, -4)

/-- Theorem stating that the equation represents a circle of radius 8 iff k = 1 -/
theorem circle_radius_eight (k : ℝ) :
  (∀ x y : ℝ, circle_equation x y k ↔ 
    ((x - circle_center.1)^2 + (y - circle_center.2)^2 = 64)) ↔ 
  k = 1 :=
sorry

end NUMINAMATH_CALUDE_circle_radius_eight_l4008_400854


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4008_400841

/-- Given a geometric sequence {aₙ} with a₁ + a₂ = -1 and a₁ - a₃ = -3, prove that a₄ = -8 -/
theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∃ (q : ℝ), ∀ n, a (n + 1) = a n * q)  -- Geometric sequence condition
  (h_sum : a 1 + a 2 = -1)  -- First condition
  (h_diff : a 1 - a 3 = -3)  -- Second condition
  : a 4 = -8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l4008_400841


namespace NUMINAMATH_CALUDE_tank_length_proof_l4008_400886

/-- Proves that the length of a rectangular tank is 3 feet given specific conditions -/
theorem tank_length_proof (l : ℝ) : 
  let w : ℝ := 6
  let h : ℝ := 2
  let cost_per_sqft : ℝ := 20
  let total_cost : ℝ := 1440
  let surface_area : ℝ := 2 * l * w + 2 * l * h + 2 * w * h
  total_cost = cost_per_sqft * surface_area → l = 3 := by
  sorry

end NUMINAMATH_CALUDE_tank_length_proof_l4008_400886


namespace NUMINAMATH_CALUDE_largest_power_of_three_dividing_A_l4008_400848

/-- Given that A is the largest product of natural numbers whose sum is 2011,
    this theorem states that the largest power of three that divides A is 3^669. -/
theorem largest_power_of_three_dividing_A : ∃ A : ℕ,
  (∀ (factors : List ℕ), (factors.sum = 2011 ∧ factors.prod ≤ A) → 
    ∃ (k : ℕ), A = 3^669 * k ∧ ¬(∃ m : ℕ, A = 3^(669 + 1) * m)) := by
  sorry

end NUMINAMATH_CALUDE_largest_power_of_three_dividing_A_l4008_400848


namespace NUMINAMATH_CALUDE_inequality_equivalence_l4008_400817

def inequality_solution (x : ℝ) : Prop :=
  (x - 1) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0

def solution_set (x : ℝ) : Prop :=
  x < 1 ∨ (1 < x ∧ x < 2) ∨ (2 < x ∧ x < 4) ∨ (4 < x ∧ x < 5) ∨ 7 < x

theorem inequality_equivalence :
  ∀ x : ℝ, inequality_solution x ↔ solution_set x := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l4008_400817


namespace NUMINAMATH_CALUDE_growth_percentage_calculation_l4008_400871

def previous_height : Real := 139.65
def current_height : Real := 147.0

theorem growth_percentage_calculation :
  let difference := current_height - previous_height
  let growth_rate := difference / previous_height
  let growth_percentage := growth_rate * 100
  ∃ ε > 0, abs (growth_percentage - 5.26) < ε :=
sorry

end NUMINAMATH_CALUDE_growth_percentage_calculation_l4008_400871


namespace NUMINAMATH_CALUDE_katies_cupcakes_l4008_400899

/-- Proves that Katie initially made 26 cupcakes given the conditions of the problem -/
theorem katies_cupcakes (X : ℕ) : X = 26 :=
  by
    -- Define the conditions
    have sold : ℕ := 20
    have made_more : ℕ := 20
    have current : ℕ := 26

    -- State the relationship between initial, sold, made more, and current cupcakes
    have h : X - sold + made_more = current := by sorry

    -- Prove that X = 26
    sorry

end NUMINAMATH_CALUDE_katies_cupcakes_l4008_400899


namespace NUMINAMATH_CALUDE_sum_of_ratio_terms_l4008_400822

-- Define the points
variable (A B C D O P X Y : ℝ × ℝ)

-- Define the lengths
def length_AD : ℝ := 10
def length_AO : ℝ := 10
def length_OB : ℝ := 10
def length_BC : ℝ := 10
def length_AB : ℝ := 12
def length_DO : ℝ := 12
def length_OC : ℝ := 12

-- Define the conditions
axiom isosceles_DAO : length_AD = length_AO
axiom isosceles_AOB : length_AO = length_OB
axiom isosceles_OBC : length_OB = length_BC
axiom P_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B
axiom OP_perpendicular_AB : (O.1 - P.1) * (B.1 - A.1) + (O.2 - P.2) * (B.2 - A.2) = 0
axiom X_midpoint_AD : X = ((A.1 + D.1) / 2, (A.2 + D.2) / 2)
axiom Y_midpoint_BC : Y = ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Define the areas of trapezoids
def area_ABYX : ℝ := sorry
def area_XYCD : ℝ := sorry

-- Define the ratio of areas
def ratio_areas : ℚ := sorry

-- Theorem to prove
theorem sum_of_ratio_terms : 
  ∃ (p q : ℕ), ratio_areas = p / q ∧ p + q = 12 :=
sorry

end NUMINAMATH_CALUDE_sum_of_ratio_terms_l4008_400822


namespace NUMINAMATH_CALUDE_light_bulb_probabilities_l4008_400823

/-- Market share of Factory A -/
def market_share_A : ℝ := 0.6

/-- Market share of Factory B -/
def market_share_B : ℝ := 0.4

/-- Qualification rate of Factory A products -/
def qual_rate_A : ℝ := 0.9

/-- Qualification rate of Factory B products -/
def qual_rate_B : ℝ := 0.8

/-- Probability of exactly one qualified light bulb out of two from Factory A -/
def prob_one_qualified_A : ℝ := 2 * qual_rate_A * (1 - qual_rate_A)

/-- Probability of a randomly purchased light bulb being qualified -/
def prob_random_qualified : ℝ := market_share_A * qual_rate_A + market_share_B * qual_rate_B

theorem light_bulb_probabilities :
  prob_one_qualified_A = 0.18 ∧ prob_random_qualified = 0.86 := by
  sorry

#check light_bulb_probabilities

end NUMINAMATH_CALUDE_light_bulb_probabilities_l4008_400823


namespace NUMINAMATH_CALUDE_probability_of_letter_in_probability_l4008_400889

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of unique letters in the word 'PROBABILITY' -/
def unique_letters : ℕ := 9

/-- The probability of randomly selecting a letter from the alphabet
    that appears in the word 'PROBABILITY' -/
def probability : ℚ := unique_letters / alphabet_size

theorem probability_of_letter_in_probability :
  probability = 9 / 26 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_letter_in_probability_l4008_400889


namespace NUMINAMATH_CALUDE_inequality_solution_l4008_400885

theorem inequality_solution (x : ℝ) (h : x ≠ 2) :
  |((3 * x - 2) / (x - 2))| > 3 ↔ x < 4/3 ∨ x > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4008_400885


namespace NUMINAMATH_CALUDE_perfect_square_implies_zero_l4008_400878

theorem perfect_square_implies_zero (a b : ℤ) :
  (∀ n : ℕ, ∃ k : ℤ, a * 2013^n + b = k^2) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_implies_zero_l4008_400878


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l4008_400821

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 6 ∧ 
  (x₁^2 - 7*x₁ + 6 = 0) ∧ (x₂^2 - 7*x₂ + 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l4008_400821


namespace NUMINAMATH_CALUDE_yellow_ball_count_l4008_400801

theorem yellow_ball_count (red yellow green : ℕ) : 
  red + yellow + green = 68 →
  yellow = 2 * red →
  3 * green = 4 * yellow →
  yellow = 24 := by
sorry

end NUMINAMATH_CALUDE_yellow_ball_count_l4008_400801


namespace NUMINAMATH_CALUDE_tangent_implies_t_equals_4e_l4008_400803

-- Define the curves C₁ and C₂
def C₁ (t : ℝ) (x y : ℝ) : Prop := y^2 = t*x ∧ y > 0 ∧ t > 0

def C₂ (x y : ℝ) : Prop := y = Real.exp (x + 1) - 1

-- Define the tangent line condition
def tangent_condition (t : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ), C₁ t (4/t) 2 ∧ C₁ t x₀ y₀ ∧ C₂ x₀ y₀ ∧
  (∀ (x y : ℝ), y - 2 = (t/4)*(x - 4/t) → (C₁ t x y ∨ C₂ x y))

-- State the theorem
theorem tangent_implies_t_equals_4e :
  ∀ t : ℝ, tangent_condition t → t = 4 * Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_implies_t_equals_4e_l4008_400803


namespace NUMINAMATH_CALUDE_unique_four_digit_number_with_geometric_property_l4008_400818

def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def first_digit (n : ℕ) : ℕ :=
  n / 1000

def second_digit (n : ℕ) : ℕ :=
  (n / 100) % 10

def third_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def fourth_digit (n : ℕ) : ℕ :=
  n % 10

def ab (n : ℕ) : ℕ :=
  10 * (first_digit n) + (second_digit n)

def bc (n : ℕ) : ℕ :=
  10 * (second_digit n) + (third_digit n)

def cd (n : ℕ) : ℕ :=
  10 * (third_digit n) + (fourth_digit n)

def is_increasing_geometric_sequence (x y z : ℕ) : Prop :=
  x < y ∧ y < z ∧ y * y = x * z

theorem unique_four_digit_number_with_geometric_property :
  ∃! n : ℕ, is_valid_four_digit_number n ∧
             first_digit n ≠ 0 ∧
             is_increasing_geometric_sequence (ab n) (bc n) (cd n) :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_with_geometric_property_l4008_400818


namespace NUMINAMATH_CALUDE_y1_greater_y2_l4008_400852

/-- A linear function f(x) = -3x + 1 -/
def f (x : ℝ) : ℝ := -3 * x + 1

/-- Theorem: For the linear function f(x) = -3x + 1, 
    if (2, y₁) and (3, y₂) are points on its graph, then y₁ > y₂ -/
theorem y1_greater_y2 (y₁ y₂ : ℝ) 
  (h1 : f 2 = y₁) 
  (h2 : f 3 = y₂) : 
  y₁ > y₂ := by
  sorry


end NUMINAMATH_CALUDE_y1_greater_y2_l4008_400852


namespace NUMINAMATH_CALUDE_perimeter_diagonal_ratio_bounds_l4008_400807

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Add appropriate convexity condition

/-- The perimeter of a convex quadrilateral -/
def perimeter (q : ConvexQuadrilateral) : ℝ := sorry

/-- The sum of diagonal lengths of a convex quadrilateral -/
def diagonalSum (q : ConvexQuadrilateral) : ℝ := sorry

/-- Theorem: The ratio of perimeter to diagonal sum is strictly between 1 and 2 -/
theorem perimeter_diagonal_ratio_bounds (q : ConvexQuadrilateral) :
  1 < perimeter q / diagonalSum q ∧ perimeter q / diagonalSum q < 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_diagonal_ratio_bounds_l4008_400807


namespace NUMINAMATH_CALUDE_total_canoes_by_april_l4008_400819

def canoes_built (month : Nat) : Nat :=
  match month with
  | 0 => 5  -- February (0-indexed)
  | n + 1 => 3 * canoes_built n

theorem total_canoes_by_april : 
  canoes_built 0 + canoes_built 1 + canoes_built 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_canoes_by_april_l4008_400819


namespace NUMINAMATH_CALUDE_students_taking_algebra_or_drafting_but_not_both_l4008_400831

-- Define the sets of students
def algebra : Finset ℕ := sorry
def drafting : Finset ℕ := sorry
def geometry : Finset ℕ := sorry

-- State the theorem
theorem students_taking_algebra_or_drafting_but_not_both : 
  (algebra.card + drafting.card - (algebra ∩ drafting).card) - ((geometry ∩ drafting).card - (algebra ∩ geometry ∩ drafting).card) = 42 :=
by
  -- Given conditions
  have h1 : (algebra ∩ drafting).card = 15 := sorry
  have h2 : algebra.card = 30 := sorry
  have h3 : (drafting \ algebra).card = 14 := sorry
  have h4 : (geometry \ (algebra ∪ drafting)).card = 8 := sorry
  have h5 : ((geometry ∩ drafting) \ algebra).card = 5 := sorry
  
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_students_taking_algebra_or_drafting_but_not_both_l4008_400831


namespace NUMINAMATH_CALUDE_linear_system_solution_l4008_400833

theorem linear_system_solution (x y : ℚ) 
  (eq1 : 3 * x - y = 9) 
  (eq2 : 2 * y - x = 1) : 
  5 * x + 4 * y = 39 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l4008_400833
