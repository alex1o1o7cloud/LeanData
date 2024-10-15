import Mathlib

namespace NUMINAMATH_CALUDE_max_value_under_constraint_l1467_146729

/-- The objective function to be maximized -/
def f (x y : ℝ) : ℝ := 8 * x^2 + 9 * x * y + 18 * y^2 + 2 * x + 3 * y

/-- The constraint function -/
def g (x y : ℝ) : ℝ := 4 * x^2 + 9 * y^2 - 8

/-- Theorem stating that the maximum value of f subject to the constraint g = 0 is 26 -/
theorem max_value_under_constraint : 
  ∃ (x y : ℝ), g x y = 0 ∧ f x y = 26 ∧ ∀ (x' y' : ℝ), g x' y' = 0 → f x' y' ≤ 26 := by
  sorry

end NUMINAMATH_CALUDE_max_value_under_constraint_l1467_146729


namespace NUMINAMATH_CALUDE_marbles_distribution_l1467_146759

/-- Given a total number of marbles and a number of groups, 
    calculates the number of marbles in each group -/
def marbles_per_group (total_marbles : ℕ) (num_groups : ℕ) : ℕ :=
  total_marbles / num_groups

/-- Proves that given 20 marbles and 5 groups, there are 4 marbles in each group -/
theorem marbles_distribution :
  marbles_per_group 20 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_marbles_distribution_l1467_146759


namespace NUMINAMATH_CALUDE_range_of_s_squared_minus_c_squared_l1467_146708

theorem range_of_s_squared_minus_c_squared (k : ℝ) (x y : ℝ) :
  k > 0 →
  x = k * y →
  let r := Real.sqrt (x^2 + y^2)
  let s := y / r
  let c := x / r
  (∀ z, s^2 - c^2 = z → -1 ≤ z ∧ z ≤ 1) ∧
  (∃ z, s^2 - c^2 = z ∧ z = -1) ∧
  (∃ z, s^2 - c^2 = z ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_s_squared_minus_c_squared_l1467_146708


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1467_146769

theorem min_value_reciprocal_sum (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h_sum : 2 * m + n = 1) :
  (1 / m + 2 / n) ≥ 8 ∧ ∃ m₀ n₀ : ℝ, 0 < m₀ ∧ 0 < n₀ ∧ 2 * m₀ + n₀ = 1 ∧ 1 / m₀ + 2 / n₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1467_146769


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l1467_146721

/-- Represents a position on the grid -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the grid -/
def Grid := List Position

/-- The initially shaded squares -/
def initial_shaded : Grid := 
  [⟨1, 2⟩, ⟨3, 1⟩, ⟨4, 4⟩, ⟨6, 1⟩]

/-- Function to check if a grid has both horizontal and vertical symmetry -/
def has_symmetry (g : Grid) : Bool := sorry

/-- Function to count the number of additional squares needed for symmetry -/
def additional_squares_needed (g : Grid) : Nat := sorry

/-- Theorem stating that 8 additional squares are needed for symmetry -/
theorem min_additional_squares_for_symmetry :
  additional_squares_needed initial_shaded = 8 := by sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l1467_146721


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1467_146756

theorem trigonometric_identity :
  Real.sin (20 * π / 180) * Real.sin (80 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1467_146756


namespace NUMINAMATH_CALUDE_stating_nth_smallest_d₀_is_correct_l1467_146740

/-- 
Given a non-negative integer d₀ and a positive integer v,
this function returns true if v² = 8d₀, false otherwise.
-/
def is_valid_pair (d₀ v : ℕ) : Prop :=
  v^2 = 8 * d₀

/-- 
This function returns the nth smallest non-negative integer d₀
such that there exists a positive integer v where v² = 8d₀.
-/
def nth_smallest_d₀ (n : ℕ) : ℕ :=
  4^(n-1)

/-- 
Theorem stating that nth_smallest_d₀ correctly computes
the nth smallest d₀ satisfying the required property.
-/
theorem nth_smallest_d₀_is_correct (n : ℕ) :
  n > 0 →
  (∃ v : ℕ, is_valid_pair (nth_smallest_d₀ n) v) ∧
  (∀ d : ℕ, d < nth_smallest_d₀ n →
    (∃ v : ℕ, is_valid_pair d v) →
    (∃ k < n, d = nth_smallest_d₀ k)) :=
by sorry

end NUMINAMATH_CALUDE_stating_nth_smallest_d₀_is_correct_l1467_146740


namespace NUMINAMATH_CALUDE_emery_shoe_alteration_cost_l1467_146732

theorem emery_shoe_alteration_cost :
  let num_pairs : ℕ := 17
  let cost_per_shoe : ℕ := 29
  let total_shoes : ℕ := num_pairs * 2
  let total_cost : ℕ := total_shoes * cost_per_shoe
  total_cost = 986 := by sorry

end NUMINAMATH_CALUDE_emery_shoe_alteration_cost_l1467_146732


namespace NUMINAMATH_CALUDE_foreign_language_teachers_l1467_146700

/-- The number of teachers who do not teach English, Japanese, or French -/
theorem foreign_language_teachers (total : ℕ) (english : ℕ) (japanese : ℕ) (french : ℕ)
  (eng_jap : ℕ) (eng_fre : ℕ) (jap_fre : ℕ) (all_three : ℕ) :
  total = 120 →
  english = 50 →
  japanese = 45 →
  french = 40 →
  eng_jap = 15 →
  eng_fre = 10 →
  jap_fre = 8 →
  all_three = 4 →
  total - (english + japanese + french - eng_jap - eng_fre - jap_fre + all_three) = 14 :=
by sorry

end NUMINAMATH_CALUDE_foreign_language_teachers_l1467_146700


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1467_146770

theorem fraction_multiplication : (1 : ℚ) / 2 * 3 / 5 * 7 / 11 = 21 / 110 := by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1467_146770


namespace NUMINAMATH_CALUDE_wiper_line_to_surface_l1467_146762

/-- A car wiper blade modeled as a line -/
structure WiperBlade :=
  (length : ℝ)

/-- A windshield modeled as a surface -/
structure Windshield :=
  (width : ℝ)
  (height : ℝ)

/-- The area swept by a wiper blade on a windshield -/
def swept_area (blade : WiperBlade) (shield : Windshield) : ℝ :=
  blade.length * shield.width

/-- Theorem stating that a car wiper on a windshield represents a line moving into a surface -/
theorem wiper_line_to_surface (blade : WiperBlade) (shield : Windshield) :
  ∃ (area : ℝ), area = swept_area blade shield ∧ area > 0 :=
sorry

end NUMINAMATH_CALUDE_wiper_line_to_surface_l1467_146762


namespace NUMINAMATH_CALUDE_factorial_not_prime_l1467_146781

theorem factorial_not_prime (n : ℕ) (h : n > 1) : ¬ Nat.Prime (n!) := by
  sorry

end NUMINAMATH_CALUDE_factorial_not_prime_l1467_146781


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_fraction_simplification_l1467_146760

-- Problem 1
theorem sqrt_expression_equality : 
  Real.sqrt 12 + Real.sqrt 3 * (Real.sqrt 2 - 1) = Real.sqrt 3 + Real.sqrt 6 := by
  sorry

-- Problem 2
theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) :
  (a + (2 * a * b + b^2) / a) / ((a + b) / a) = a + b := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_fraction_simplification_l1467_146760


namespace NUMINAMATH_CALUDE_inequality_proof_l1467_146753

theorem inequality_proof (x y : ℝ) (h : x^12 + y^12 ≤ 2) :
  x^2 + y^2 + x^2*y^2 ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1467_146753


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1467_146735

/-- A linear function f(x) = mx + b passes through a quadrant if there exists a point (x, y) in that quadrant such that y = f(x) -/
def passes_through_quadrant (m b : ℝ) (q : Nat) : Prop :=
  match q with
  | 1 => ∃ x y, x > 0 ∧ y > 0 ∧ y = m * x + b
  | 2 => ∃ x y, x < 0 ∧ y > 0 ∧ y = m * x + b
  | 3 => ∃ x y, x < 0 ∧ y < 0 ∧ y = m * x + b
  | 4 => ∃ x y, x > 0 ∧ y < 0 ∧ y = m * x + b
  | _ => False

/-- The linear function y = -2x + 1 passes through Quadrants I, II, and IV -/
theorem linear_function_quadrants :
  passes_through_quadrant (-2) 1 1 ∧
  passes_through_quadrant (-2) 1 2 ∧
  passes_through_quadrant (-2) 1 4 :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1467_146735


namespace NUMINAMATH_CALUDE_unique_solution_square_sum_l1467_146796

theorem unique_solution_square_sum (x y : ℝ) : 
  (x - 2*y)^2 + (y - 1)^2 = 0 ↔ x = 2 ∧ y = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_square_sum_l1467_146796


namespace NUMINAMATH_CALUDE_rectangle_to_circle_area_l1467_146791

/-- Given a rectangle with area 200 and length twice its width, 
    the area of the largest circle that can be formed from a string 
    equal to the rectangle's perimeter is 900/π. -/
theorem rectangle_to_circle_area (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let area_rect := w * l
  let perimeter := 2 * (w + l)
  area_rect = 200 → (perimeter^2) / (4 * π) = 900 / π := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_circle_area_l1467_146791


namespace NUMINAMATH_CALUDE_inequality_preservation_l1467_146758

theorem inequality_preservation (x y : ℝ) (h : x > y) : x / 2 > y / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l1467_146758


namespace NUMINAMATH_CALUDE_commute_days_calculation_l1467_146794

theorem commute_days_calculation (morning_bus afternoon_bus train_commute : ℕ) 
  (h1 : morning_bus = 8)
  (h2 : afternoon_bus = 15)
  (h3 : train_commute = 9) :
  ∃ (morning_train afternoon_train both_bus : ℕ),
    morning_train + afternoon_train = train_commute ∧
    morning_bus = afternoon_train + both_bus ∧
    afternoon_bus = morning_train + both_bus ∧
    morning_train + afternoon_train + both_bus = 16 :=
by sorry

end NUMINAMATH_CALUDE_commute_days_calculation_l1467_146794


namespace NUMINAMATH_CALUDE_union_of_sets_l1467_146713

theorem union_of_sets : 
  let A : Set ℕ := {1, 2}
  let B : Set ℕ := {1, 3, 5}
  A ∪ B = {1, 2, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l1467_146713


namespace NUMINAMATH_CALUDE_vector_subtraction_l1467_146739

def a : Fin 3 → ℝ := ![-3, 4, 2]
def b : Fin 3 → ℝ := ![5, -1, 3]

theorem vector_subtraction :
  (fun i => a i - 2 * b i) = ![-13, 6, -4] := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l1467_146739


namespace NUMINAMATH_CALUDE_distance_center_to_secant_l1467_146793

/-- Given a circle O with center (0, 0) and radius 5, a tangent line AD of length 4,
    and a secant line ABC where AC = 8, the distance from the center O to the line AC is 4. -/
theorem distance_center_to_secant (O A B C D : ℝ × ℝ) : 
  let r := 5
  let circle := {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}
  (A ∉ circle) →
  (B ∈ circle) →
  (C ∈ circle) →
  (D ∈ circle) →
  (∀ p ∈ circle, (p.1 - A.1) * (D.1 - A.1) + (p.2 - A.2) * (D.2 - A.2) = 0) →
  (Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2) = 4) →
  (Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 8) →
  (abs ((O.2 - A.2) * (C.1 - A.1) - (O.1 - A.1) * (C.2 - A.2)) / 
   Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = 4) :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_secant_l1467_146793


namespace NUMINAMATH_CALUDE_geometric_mean_of_square_sides_l1467_146704

theorem geometric_mean_of_square_sides (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 64) (h₂ : a₂ = 81) (h₃ : a₃ = 144) :
  (((a₁.sqrt * a₂.sqrt * a₃.sqrt) ^ (1/3 : ℝ)) : ℝ) = 6 * (4 ^ (1/3 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_of_square_sides_l1467_146704


namespace NUMINAMATH_CALUDE_emily_garden_seeds_l1467_146783

theorem emily_garden_seeds (total_seeds : ℕ) (big_garden_seeds : ℕ) (small_gardens : ℕ) 
  (h1 : total_seeds = 41)
  (h2 : big_garden_seeds = 29)
  (h3 : small_gardens = 3)
  (h4 : small_gardens > 0) :
  (total_seeds - big_garden_seeds) / small_gardens = 4 := by
sorry

end NUMINAMATH_CALUDE_emily_garden_seeds_l1467_146783


namespace NUMINAMATH_CALUDE_alloy_mixing_theorem_l1467_146763

/-- Represents an alloy with two metals -/
structure Alloy where
  ratio1 : ℚ
  ratio2 : ℚ

/-- Creates a new alloy by mixing two existing alloys -/
def mixAlloys (a1 : Alloy) (p1 : ℚ) (a2 : Alloy) (p2 : ℚ) : Alloy :=
  { ratio1 := (a1.ratio1 * p1 + a2.ratio1 * p2) / (p1 + p2),
    ratio2 := (a1.ratio2 * p1 + a2.ratio2 * p2) / (p1 + p2) }

theorem alloy_mixing_theorem :
  let alloy1 : Alloy := { ratio1 := 1, ratio2 := 2 }
  let alloy2 : Alloy := { ratio1 := 2, ratio2 := 3 }
  let mixedAlloy := mixAlloys alloy1 9 alloy2 35
  mixedAlloy.ratio1 / mixedAlloy.ratio2 = 17 / 27 := by
  sorry

end NUMINAMATH_CALUDE_alloy_mixing_theorem_l1467_146763


namespace NUMINAMATH_CALUDE_b_share_is_correct_l1467_146798

/-- Represents the rental information for a person -/
structure RentalInfo where
  horses : ℕ
  months : ℕ

/-- Calculates the total horse-months for a given rental information -/
def horseMonths (info : RentalInfo) : ℕ :=
  info.horses * info.months

/-- Represents the pasture rental problem -/
structure PastureRental where
  totalRent : ℚ
  a : RentalInfo
  b : RentalInfo
  c : RentalInfo

/-- Calculates the total horse-months for all renters -/
def totalHorseMonths (rental : PastureRental) : ℕ :=
  horseMonths rental.a + horseMonths rental.b + horseMonths rental.c

/-- Calculates the rent per horse-month -/
def rentPerHorseMonth (rental : PastureRental) : ℚ :=
  rental.totalRent / totalHorseMonths rental

/-- Calculates the rent for a specific renter -/
def renterShare (rental : PastureRental) (renter : RentalInfo) : ℚ :=
  (rentPerHorseMonth rental) * (horseMonths renter)

/-- The main theorem stating b's share of the rent -/
theorem b_share_is_correct (rental : PastureRental) 
  (h1 : rental.totalRent = 841)
  (h2 : rental.a = ⟨12, 8⟩)
  (h3 : rental.b = ⟨16, 9⟩)
  (h4 : rental.c = ⟨18, 6⟩) :
  renterShare rental rental.b = 348.48 := by
  sorry

end NUMINAMATH_CALUDE_b_share_is_correct_l1467_146798


namespace NUMINAMATH_CALUDE_increasing_magnitude_l1467_146723

theorem increasing_magnitude (x : ℝ) (h : 0.85 < x ∧ x < 1.1) :
  x ≤ x + Real.sin x ∧ x + Real.sin x < x^(x^x) := by
  sorry

end NUMINAMATH_CALUDE_increasing_magnitude_l1467_146723


namespace NUMINAMATH_CALUDE_removed_number_for_mean_l1467_146743

theorem removed_number_for_mean (n : ℕ) (h : n ≥ 9) :
  ∃ x : ℕ, x ≤ n ∧ 
    (((n * (n + 1)) / 2 - x) / (n - 1) : ℚ) = 19/4 →
    x = 7 :=
  sorry

end NUMINAMATH_CALUDE_removed_number_for_mean_l1467_146743


namespace NUMINAMATH_CALUDE_total_stickers_count_l1467_146792

/-- The number of stickers on each page -/
def stickers_per_page : ℕ := 10

/-- The number of pages -/
def number_of_pages : ℕ := 22

/-- The total number of stickers -/
def total_stickers : ℕ := stickers_per_page * number_of_pages

theorem total_stickers_count : total_stickers = 220 := by
  sorry

end NUMINAMATH_CALUDE_total_stickers_count_l1467_146792


namespace NUMINAMATH_CALUDE_time_interval_is_20_minutes_l1467_146785

/-- The time interval between cars given total time and number of cars -/
def time_interval (total_time_hours : ℕ) (num_cars : ℕ) : ℚ :=
  (total_time_hours * 60 : ℚ) / num_cars

/-- Theorem: The time interval between cars is 20 minutes -/
theorem time_interval_is_20_minutes :
  time_interval 10 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_time_interval_is_20_minutes_l1467_146785


namespace NUMINAMATH_CALUDE_incorrect_value_calculation_l1467_146716

theorem incorrect_value_calculation (n : ℕ) (initial_mean correct_mean correct_value : ℝ) 
  (h1 : n = 30)
  (h2 : initial_mean = 150)
  (h3 : correct_mean = 151)
  (h4 : correct_value = 165) :
  let initial_sum := n * initial_mean
  let correct_sum := n * correct_mean
  let difference := correct_sum - initial_sum
  initial_sum + correct_value - difference = n * correct_mean := by sorry

end NUMINAMATH_CALUDE_incorrect_value_calculation_l1467_146716


namespace NUMINAMATH_CALUDE_divide_negative_four_by_two_l1467_146703

theorem divide_negative_four_by_two : -4 / 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_divide_negative_four_by_two_l1467_146703


namespace NUMINAMATH_CALUDE_euclidean_algorithm_fibonacci_bound_l1467_146701

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define the Euclidean algorithm
def euclidean_algorithm (m₀ m₁ : ℕ) : ℕ → Prop
  | 0 => m₁ = 0
  | k + 1 => ∃ q r, m₀ = q * m₁ + r ∧ r < m₁ ∧ euclidean_algorithm m₁ r k

-- Theorem statement
theorem euclidean_algorithm_fibonacci_bound {m₀ m₁ k : ℕ} 
  (h : euclidean_algorithm m₀ m₁ k) : 
  m₁ ≥ fib (k + 1) ∧ m₀ ≥ fib (k + 2) := by
  sorry

end NUMINAMATH_CALUDE_euclidean_algorithm_fibonacci_bound_l1467_146701


namespace NUMINAMATH_CALUDE_john_restringing_problem_l1467_146767

/-- The number of basses John needs to restring -/
def num_basses : ℕ := 3

/-- The number of guitars John needs to restring -/
def num_guitars : ℕ := 2 * num_basses

/-- The number of 8-string guitars John needs to restring -/
def num_8string_guitars : ℕ := num_guitars - 3

/-- The total number of strings needed -/
def total_strings : ℕ := 72

theorem john_restringing_problem :
  4 * num_basses + 6 * num_guitars + 8 * num_8string_guitars = total_strings :=
by sorry

end NUMINAMATH_CALUDE_john_restringing_problem_l1467_146767


namespace NUMINAMATH_CALUDE_yellow_shirt_pairs_l1467_146730

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) :
  blue_students = 75 →
  yellow_students = 105 →
  total_students = blue_students + yellow_students →
  total_pairs = 90 →
  blue_blue_pairs = 30 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 45 ∧ 
    yellow_yellow_pairs = (yellow_students - (total_students - 2 * blue_blue_pairs)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_yellow_shirt_pairs_l1467_146730


namespace NUMINAMATH_CALUDE_beverage_distribution_l1467_146727

/-- Represents the number of cans of beverage -/
def total_cans : ℚ := 5

/-- Represents the number of children -/
def num_children : ℚ := 8

/-- Represents each child's share of the total beverage -/
def share_of_total : ℚ := 1 / num_children

/-- Represents each child's share in terms of cans -/
def share_in_cans : ℚ := total_cans / num_children

theorem beverage_distribution :
  share_of_total = 1 / 8 ∧ share_in_cans = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_beverage_distribution_l1467_146727


namespace NUMINAMATH_CALUDE_f_monotone_and_bounded_l1467_146711

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - (1/2) * x^2 - a * Real.sin x - 1

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - x - a * Real.cos x

theorem f_monotone_and_bounded (a : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  (∀ x y : ℝ, x < y → f a x < f a y) ∧
  (∀ M : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-π/3) (π/3) → |f_deriv a x| ≤ M) →
    ∀ x : ℝ, x ∈ Set.Icc (-π/3) (π/3) → |f a x| ≤ M) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_and_bounded_l1467_146711


namespace NUMINAMATH_CALUDE_existence_of_three_quadratic_polynomials_l1467_146797

theorem existence_of_three_quadratic_polynomials :
  ∃ (p₁ p₂ p₃ : ℝ → ℝ),
    (∃ x₁, p₁ x₁ = 0) ∧
    (∃ x₂, p₂ x₂ = 0) ∧
    (∃ x₃, p₃ x₃ = 0) ∧
    (∀ x, p₁ x + p₂ x ≠ 0) ∧
    (∀ x, p₁ x + p₃ x ≠ 0) ∧
    (∀ x, p₂ x + p₃ x ≠ 0) ∧
    (∀ x, p₁ x = (x^2 : ℝ)) ∧
    (∀ x, p₂ x = ((x - 1)^2 : ℝ)) ∧
    (∀ x, p₃ x = ((x - 2)^2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_three_quadratic_polynomials_l1467_146797


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l1467_146766

/-- Proves that arctan(tan 70° - 2 tan 45°) = 135° --/
theorem arctan_tan_difference (θ : Real) : 
  θ ∈ Set.Icc 0 180 ∧ 
  θ * (π / 180) = Real.arctan (Real.tan (70 * π / 180) - 2 * Real.tan (45 * π / 180)) → 
  θ = 135 := by
  sorry

#check arctan_tan_difference

end NUMINAMATH_CALUDE_arctan_tan_difference_l1467_146766


namespace NUMINAMATH_CALUDE_jason_arm_tattoos_count_l1467_146772

-- Define the number of tattoos Jason has on each arm
def jason_arm_tattoos : ℕ := sorry

-- Define the number of tattoos Jason has on each leg
def jason_leg_tattoos : ℕ := 3

-- Define the total number of tattoos Jason has
def jason_total_tattoos : ℕ := 2 * jason_arm_tattoos + 2 * jason_leg_tattoos

-- Define the number of tattoos Adam has
def adam_tattoos : ℕ := 23

-- Theorem to prove
theorem jason_arm_tattoos_count :
  jason_arm_tattoos = 2 ∧
  adam_tattoos = 2 * jason_total_tattoos + 3 :=
by sorry

end NUMINAMATH_CALUDE_jason_arm_tattoos_count_l1467_146772


namespace NUMINAMATH_CALUDE_good_games_count_l1467_146748

def games_from_friend : ℕ := 41
def games_from_garage_sale : ℕ := 14
def non_working_games : ℕ := 31

theorem good_games_count : 
  games_from_friend + games_from_garage_sale - non_working_games = 24 := by
  sorry

end NUMINAMATH_CALUDE_good_games_count_l1467_146748


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l1467_146750

theorem divisibility_by_twelve (m : Nat) : m ≤ 9 → (365 * 10 + m) % 12 = 0 ↔ m = 0 := by sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l1467_146750


namespace NUMINAMATH_CALUDE_line_l_passes_through_A_and_B_l1467_146744

/-- The line l passes through points A(-1, 0) and B(1, 4) -/
def line_l (x y : ℝ) : Prop := y = 2 * x + 2

/-- Point A has coordinates (-1, 0) -/
def point_A : ℝ × ℝ := (-1, 0)

/-- Point B has coordinates (1, 4) -/
def point_B : ℝ × ℝ := (1, 4)

/-- The line l passes through points A and B -/
theorem line_l_passes_through_A_and_B : 
  line_l point_A.1 point_A.2 ∧ line_l point_B.1 point_B.2 := by sorry

end NUMINAMATH_CALUDE_line_l_passes_through_A_and_B_l1467_146744


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1467_146733

def data : List ℝ := [2, 4, 6, 8]

def median (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem min_value_reciprocal_sum 
  (m : ℝ) 
  (n : ℝ) 
  (hm : m = median data) 
  (hn : n = variance data) 
  (a : ℝ) 
  (b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (heq : m * a + n * b = 1) : 
  (1 / a + 1 / b) ≥ 20 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1467_146733


namespace NUMINAMATH_CALUDE_min_a4_value_l1467_146702

theorem min_a4_value (a : Fin 10 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_a2 : a 2 = a 1 + a 5)
  (h_a3 : a 3 = a 2 + a 6)
  (h_a4 : a 4 = a 3 + a 7)
  (h_a6 : a 6 = a 5 + a 8)
  (h_a7 : a 7 = a 6 + a 9)
  (h_a9 : a 9 = a 8 + a 10) :
  ∀ b : Fin 10 → ℕ+, 
    (∀ i j, i ≠ j → b i ≠ b j) →
    (b 2 = b 1 + b 5) →
    (b 3 = b 2 + b 6) →
    (b 4 = b 3 + b 7) →
    (b 6 = b 5 + b 8) →
    (b 7 = b 6 + b 9) →
    (b 9 = b 8 + b 10) →
    a 4 ≤ b 4 :=
by sorry

#check min_a4_value

end NUMINAMATH_CALUDE_min_a4_value_l1467_146702


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1467_146728

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (-x^4 + 4 * x^3 - 2 * x^2 + 8 * x - 7) =
  x^4 + x^3 + 3 * x^2 + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1467_146728


namespace NUMINAMATH_CALUDE_train_crossing_time_l1467_146764

/-- Given a train crossing a platform, calculate the time it takes to cross a signal pole. -/
theorem train_crossing_time (train_length platform_length platform_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_length = 1162.5)
  (h3 : platform_crossing_time = 39)
  : (train_length / ((train_length + platform_length) / platform_crossing_time)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l1467_146764


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l1467_146777

-- Define the distance function
def S (t : ℝ) : ℝ := t^3 - 2

-- State the theorem
theorem instantaneous_velocity_at_one (t : ℝ) : 
  (deriv S) 1 = 3 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l1467_146777


namespace NUMINAMATH_CALUDE_f_maximum_l1467_146745

/-- The quadratic function f(x) = -3x^2 + 9x + 24 -/
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 24

/-- The point where f attains its maximum -/
def x_max : ℝ := 1.5

theorem f_maximum :
  ∀ x : ℝ, f x ≤ f x_max := by sorry

end NUMINAMATH_CALUDE_f_maximum_l1467_146745


namespace NUMINAMATH_CALUDE_amelia_weekly_goal_l1467_146746

/-- Amelia's weekly Jet Bar sales goal -/
def weekly_goal (monday_sales tuesday_sales remaining : ℕ) : ℕ :=
  monday_sales + tuesday_sales + remaining

/-- Theorem: Amelia's weekly Jet Bar sales goal is 90 -/
theorem amelia_weekly_goal :
  ∀ (monday_sales tuesday_sales remaining : ℕ),
  monday_sales = 45 →
  tuesday_sales = monday_sales - 16 →
  remaining = 16 →
  weekly_goal monday_sales tuesday_sales remaining = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_amelia_weekly_goal_l1467_146746


namespace NUMINAMATH_CALUDE_minimal_ratio_S₁_S₂_l1467_146782

noncomputable def S₁ (α : Real) : Real :=
  4 - (2 * Real.sqrt 2 / Real.cos α)

noncomputable def S₂ (α : Real) : Real :=
  ((Real.sqrt 2 * (Real.sin α + Real.cos α) - 1)^2) / (2 * Real.sin α * Real.cos α)

theorem minimal_ratio_S₁_S₂ :
  ∃ (α₁ α₂ : Real), 
    0 ≤ α₁ ∧ α₁ ≤ Real.pi/12 ∧
    Real.pi/12 ≤ α₂ ∧ α₂ ≤ 5*Real.pi/12 ∧
    S₁ α₁ / (8 - S₁ α₁) = 1/7 ∧
    S₂ α₂ / (8 - S₂ α₂) = 1/7 ∧
    ∀ (β γ : Real), 
      (0 ≤ β ∧ β ≤ Real.pi/12 → S₁ β / (8 - S₁ β) ≥ 1/7) ∧
      (Real.pi/12 ≤ γ ∧ γ ≤ 5*Real.pi/12 → S₂ γ / (8 - S₂ γ) ≥ 1/7) :=
by sorry

end NUMINAMATH_CALUDE_minimal_ratio_S₁_S₂_l1467_146782


namespace NUMINAMATH_CALUDE_work_completion_time_l1467_146799

theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- A's completion time is positive
  (2 * (1/x + 1/10) + 10 * (1/x) = 1) →  -- Work completion equation
  (x = 15) :=  -- A's solo completion time is 15 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1467_146799


namespace NUMINAMATH_CALUDE_unique_six_digit_reverse_when_multiplied_by_nine_l1467_146754

/-- A function that returns the digits of a natural number in reverse order -/
def reverseDigits (n : ℕ) : List ℕ :=
  sorry

/-- A function that checks if a number is a six-digit number -/
def isSixDigitNumber (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

/-- The main theorem stating that 109989 is the only six-digit number
    that, when multiplied by 9, has its digits arranged in reverse order -/
theorem unique_six_digit_reverse_when_multiplied_by_nine :
  ∀ n : ℕ, isSixDigitNumber n →
    (reverseDigits n = reverseDigits (9 * n) → n = 109989) ∧
    (n = 109989 → reverseDigits n = reverseDigits (9 * n)) :=
by sorry

end NUMINAMATH_CALUDE_unique_six_digit_reverse_when_multiplied_by_nine_l1467_146754


namespace NUMINAMATH_CALUDE_connie_marbles_theorem_l1467_146726

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 593

/-- The initial number of marbles Connie had -/
def initial_marbles : ℕ := marbles_given + marbles_left

theorem connie_marbles_theorem : initial_marbles = 776 := by sorry

end NUMINAMATH_CALUDE_connie_marbles_theorem_l1467_146726


namespace NUMINAMATH_CALUDE_remainder_sum_powers_mod_5_l1467_146747

theorem remainder_sum_powers_mod_5 : (9^5 + 11^6 + 12^7) % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_powers_mod_5_l1467_146747


namespace NUMINAMATH_CALUDE_initial_gasoline_percentage_l1467_146722

/-- Proves that the initial gasoline percentage is 95% given the problem conditions -/
theorem initial_gasoline_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (optimal_ethanol_percentage : ℝ)
  (added_ethanol : ℝ)
  (h1 : initial_volume = 36)
  (h2 : initial_ethanol_percentage = 0.05)
  (h3 : optimal_ethanol_percentage = 0.10)
  (h4 : added_ethanol = 2)
  (h5 : optimal_ethanol_percentage * (initial_volume + added_ethanol) =
        initial_ethanol_percentage * initial_volume + added_ethanol) :
  initial_volume * (1 - initial_ethanol_percentage) / initial_volume = 0.95 := by
  sorry

#check initial_gasoline_percentage

end NUMINAMATH_CALUDE_initial_gasoline_percentage_l1467_146722


namespace NUMINAMATH_CALUDE_max_value_of_N_l1467_146786

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def last_two_digits (n : ℕ) : ℕ := n % 100

def remove_last_two_digits (n : ℕ) : ℕ := n / 100

theorem max_value_of_N :
  ∃ N : ℕ,
    is_perfect_square N ∧
    N ≥ 100 ∧
    last_two_digits N ≠ 0 ∧
    is_perfect_square (remove_last_two_digits N) ∧
    (∀ M : ℕ, 
      (is_perfect_square M ∧
       M ≥ 100 ∧
       last_two_digits M ≠ 0 ∧
       is_perfect_square (remove_last_two_digits M)) →
      M ≤ N) ∧
    N = 1681 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_N_l1467_146786


namespace NUMINAMATH_CALUDE_volume_Q_3_l1467_146720

/-- Recursive definition of polyhedron volumes -/
def Q : ℕ → ℚ
  | 0 => 8
  | (n + 1) => Q n + 4 * (1 / 27)^n

/-- The volume of Q₃ is 5972/729 -/
theorem volume_Q_3 : Q 3 = 5972 / 729 := by sorry

end NUMINAMATH_CALUDE_volume_Q_3_l1467_146720


namespace NUMINAMATH_CALUDE_total_cost_of_pen_and_pencil_l1467_146731

theorem total_cost_of_pen_and_pencil (pencil_cost : ℝ) (h1 : pencil_cost = 8) :
  let pen_cost := pencil_cost / 2
  pencil_cost + pen_cost = 12 := by
sorry

end NUMINAMATH_CALUDE_total_cost_of_pen_and_pencil_l1467_146731


namespace NUMINAMATH_CALUDE_circle_distance_l1467_146705

theorem circle_distance (R r : ℝ) : 
  R^2 - 4*R + 2 = 0 → 
  r^2 - 4*r + 2 = 0 → 
  R ≠ r → 
  (∃ d : ℝ, d = abs (R - r) ∧ (d = 4 ∨ d = 2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_distance_l1467_146705


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l1467_146789

/-- The perimeter of an equilateral triangle, given an isosceles triangle with specific properties -/
theorem equilateral_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun perimeter_isosceles base_isosceles perimeter_equilateral =>
    perimeter_isosceles = 40 ∧
    base_isosceles = 10 ∧
    ∃ (side : ℝ), 
      2 * side + base_isosceles = perimeter_isosceles ∧
      3 * side = perimeter_equilateral ∧
      perimeter_equilateral = 45

/-- Proof of the theorem -/
theorem equilateral_triangle_perimeter_proof :
  ∃ (perimeter_equilateral : ℝ),
    equilateral_triangle_perimeter 40 10 perimeter_equilateral :=
by
  sorry

#check equilateral_triangle_perimeter
#check equilateral_triangle_perimeter_proof

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_equilateral_triangle_perimeter_proof_l1467_146789


namespace NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1467_146707

-- Equation 1
theorem solve_equation_one (x : ℝ) : 2 * x - 7 = 5 * x - 1 → x = -2 := by
  sorry

-- Equation 2
theorem solve_equation_two (x : ℝ) : (x - 2) / 2 - (x - 1) / 6 = 1 → x = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_one_solve_equation_two_l1467_146707


namespace NUMINAMATH_CALUDE_alternating_ball_probability_l1467_146771

def num_black_balls : ℕ := 5
def num_white_balls : ℕ := 4
def total_balls : ℕ := num_black_balls + num_white_balls

def alternating_sequence (n : ℕ) : List Bool :=
  List.map (fun i => i % 2 = 0) (List.range n)

def is_valid_sequence (seq : List Bool) : Prop :=
  seq.length = total_balls ∧
  seq.head? = some true ∧
  seq = alternating_sequence total_balls

def num_valid_sequences : ℕ := 1

def total_outcomes : ℕ := Nat.choose total_balls num_black_balls

theorem alternating_ball_probability :
  (num_valid_sequences : ℚ) / total_outcomes = 1 / 126 :=
sorry

end NUMINAMATH_CALUDE_alternating_ball_probability_l1467_146771


namespace NUMINAMATH_CALUDE_number_puzzle_l1467_146719

theorem number_puzzle : ∃ x : ℝ, ((x - 50) / 4) * 3 + 28 = 73 ∧ x = 110 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1467_146719


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1467_146795

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x : ℕ | ∃ k ∈ A, x = 2 * k}

theorem complement_union_theorem :
  (U \ A) ∪ B = {0, 2, 4, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1467_146795


namespace NUMINAMATH_CALUDE_sum_of_integers_l1467_146737

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1467_146737


namespace NUMINAMATH_CALUDE_product_inequality_l1467_146712

theorem product_inequality (n : ℕ) (x : ℕ → ℝ) 
  (h_n : n ≥ 3) 
  (h_x_pos : ∀ i ∈ Finset.range (n - 1), x (i + 2) > 0)
  (h_x_prod : (Finset.range (n - 1)).prod (λ i => x (i + 2)) = 1) :
  (Finset.range (n - 1)).prod (λ i => (1 + x (i + 2)) ^ (i + 2)) > n ^ n := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1467_146712


namespace NUMINAMATH_CALUDE_power_multiplication_l1467_146709

theorem power_multiplication (a : ℝ) : a^3 * a^2 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l1467_146709


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1467_146742

theorem arithmetic_calculation : 5 * 7 + 9 * 4 - 36 / 3 = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1467_146742


namespace NUMINAMATH_CALUDE_bean_garden_rows_l1467_146761

/-- Given a garden with bean plants arranged in rows and columns,
    prove that with 15 columns and 780 total plants, there are 52 rows. -/
theorem bean_garden_rows (total_plants : ℕ) (columns : ℕ) (rows : ℕ) : 
  total_plants = 780 → columns = 15 → total_plants = rows * columns → rows = 52 := by
  sorry

end NUMINAMATH_CALUDE_bean_garden_rows_l1467_146761


namespace NUMINAMATH_CALUDE_g_has_four_zeros_l1467_146773

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x + 1/x else Real.log x

noncomputable def g (x : ℝ) : ℝ :=
  f (f x + 2) + 2

theorem g_has_four_zeros :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧
    (∀ x : ℝ, g x = 0 → x = a ∨ x = b ∨ x = c ∨ x = d) :=
sorry

end NUMINAMATH_CALUDE_g_has_four_zeros_l1467_146773


namespace NUMINAMATH_CALUDE_height_to_ad_l1467_146752

/-- Represents a parallelogram ABCD with specific properties -/
structure Parallelogram where
  -- AB length
  ab : ℝ
  -- BC length
  bc : ℝ
  -- Height dropped to CD
  height_cd : ℝ
  -- Parallelogram property
  is_parallelogram : ab > 0 ∧ bc > 0 ∧ height_cd > 0

/-- Theorem: In a parallelogram ABCD where AB = 6, BC = 8, and the height dropped to CD is 4,
    the height dropped to AD is 3 -/
theorem height_to_ad (p : Parallelogram) 
    (h_ab : p.ab = 6)
    (h_bc : p.bc = 8)
    (h_height_cd : p.height_cd = 4) :
  ∃ (height_ad : ℝ), height_ad = 3 ∧ p.ab * p.height_cd = p.bc * height_ad :=
by sorry

end NUMINAMATH_CALUDE_height_to_ad_l1467_146752


namespace NUMINAMATH_CALUDE_car_rental_hours_per_day_l1467_146755

/-- Proves that given the rental conditions, the number of hours rented per day is 8 --/
theorem car_rental_hours_per_day 
  (hourly_rate : ℝ)
  (days_per_week : ℕ)
  (weekly_income : ℝ)
  (h : hourly_rate = 20)
  (d : days_per_week = 4)
  (w : weekly_income = 640) :
  (weekly_income / (hourly_rate * days_per_week : ℝ)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_car_rental_hours_per_day_l1467_146755


namespace NUMINAMATH_CALUDE_wire_length_from_sphere_l1467_146774

/-- The length of a wire formed by melting a sphere -/
theorem wire_length_from_sphere (r : ℝ) (h : r > 0) : 
  (4 / 3 * π * 12^3) = (π * r^2 * ((4 * 12^3) / (3 * r^2))) := by
  sorry

#check wire_length_from_sphere

end NUMINAMATH_CALUDE_wire_length_from_sphere_l1467_146774


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l1467_146775

theorem p_sufficient_not_necessary_q :
  (∀ x : ℝ, 0 < x ∧ x < 2 → -1 < x ∧ x < 3) ∧
  (∃ x : ℝ, -1 < x ∧ x < 3 ∧ ¬(0 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_q_l1467_146775


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1467_146749

/-- Given a point P, return its symmetric point with respect to the y-axis -/
def symmetric_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Given a point P, return its symmetric point with respect to the x-axis -/
def symmetric_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem symmetric_point_coordinates :
  let P : ℝ × ℝ := (-10, -1)
  let P₁ : ℝ × ℝ := symmetric_y P
  let P₂ : ℝ × ℝ := symmetric_x P₁
  P₂ = (10, 1) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1467_146749


namespace NUMINAMATH_CALUDE_cyclist_heartbeats_l1467_146724

/-- The number of heartbeats during a cycling race -/
def heartbeats_during_race (heart_rate : ℕ) (pace : ℕ) (distance : ℕ) : ℕ :=
  heart_rate * pace * distance

/-- Theorem: The cyclist's heart beats 57600 times during the race -/
theorem cyclist_heartbeats :
  heartbeats_during_race 120 4 120 = 57600 := by
  sorry

#eval heartbeats_during_race 120 4 120

end NUMINAMATH_CALUDE_cyclist_heartbeats_l1467_146724


namespace NUMINAMATH_CALUDE_snow_removal_volume_l1467_146717

/-- The volume of snow to be removed from a rectangular driveway -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Proof that the volume of snow to be removed is 67.5 cubic feet -/
theorem snow_removal_volume :
  let length : ℝ := 30
  let width : ℝ := 3
  let depth : ℝ := 0.75
  snow_volume length width depth = 67.5 := by
sorry

end NUMINAMATH_CALUDE_snow_removal_volume_l1467_146717


namespace NUMINAMATH_CALUDE_sock_drawing_probability_l1467_146778

/-- The number of colors of socks --/
def num_colors : ℕ := 5

/-- The number of socks per color --/
def socks_per_color : ℕ := 2

/-- The total number of socks --/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn --/
def socks_drawn : ℕ := 5

/-- The probability of drawing exactly one pair of socks with the same color
    and the rest all different colors --/
theorem sock_drawing_probability : 
  (num_colors * (Nat.choose (num_colors - 1) (socks_drawn - 2)) * 
   (socks_per_color ^ 2) * (socks_per_color ^ (socks_drawn - 2))) /
  (Nat.choose total_socks socks_drawn) = 40 / 63 :=
by sorry

end NUMINAMATH_CALUDE_sock_drawing_probability_l1467_146778


namespace NUMINAMATH_CALUDE_prob_adjacent_is_half_l1467_146788

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange four students with two specific students adjacent. -/
def adjacent_arrangements : ℕ := 2 * (permutations 3)

/-- The total number of ways to arrange four students. -/
def total_arrangements : ℕ := permutations 4

/-- The probability of two specific students being adjacent in a line of four students. -/
def prob_adjacent : ℚ := adjacent_arrangements / total_arrangements

theorem prob_adjacent_is_half : prob_adjacent = 1/2 := by sorry

end NUMINAMATH_CALUDE_prob_adjacent_is_half_l1467_146788


namespace NUMINAMATH_CALUDE_roberta_initial_records_l1467_146725

/-- The number of records Roberta initially had -/
def initial_records : ℕ := sorry

/-- The number of records Roberta received as gifts -/
def gifted_records : ℕ := 12

/-- The number of records Roberta bought at a garage sale -/
def bought_records : ℕ := 30

/-- The number of days it takes Roberta to listen to one record -/
def days_per_record : ℕ := 2

/-- The total number of days it will take Roberta to listen to her entire collection -/
def total_listening_days : ℕ := 100

theorem roberta_initial_records :
  initial_records = 8 :=
by sorry

end NUMINAMATH_CALUDE_roberta_initial_records_l1467_146725


namespace NUMINAMATH_CALUDE_andy_total_distance_l1467_146780

/-- The total distance Andy walks given his trips to school and market -/
def total_distance (house_to_school market_to_house : ℕ) : ℕ :=
  2 * house_to_school + market_to_house

/-- Theorem stating the total distance Andy walks -/
theorem andy_total_distance :
  let house_to_school := 50
  let house_to_market := 40
  total_distance house_to_school house_to_market = 140 := by
  sorry

end NUMINAMATH_CALUDE_andy_total_distance_l1467_146780


namespace NUMINAMATH_CALUDE_backyard_sod_coverage_l1467_146741

/-- Represents the dimensions of a rectangular section -/
structure Section where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular section -/
def sectionArea (s : Section) : ℕ := s.length * s.width

/-- Represents the dimensions of a sod square -/
structure SodSquare where
  side : ℕ

/-- Calculates the area of a sod square -/
def sodSquareArea (s : SodSquare) : ℕ := s.side * s.side

/-- Calculates the number of sod squares needed to cover a given area -/
def sodSquaresNeeded (totalArea : ℕ) (sodSquare : SodSquare) : ℕ :=
  totalArea / sodSquareArea sodSquare

theorem backyard_sod_coverage (section1 : Section) (section2 : Section) (sodSquare : SodSquare) :
  section1.length = 30 →
  section1.width = 40 →
  section2.length = 60 →
  section2.width = 80 →
  sodSquare.side = 2 →
  sodSquaresNeeded (sectionArea section1 + sectionArea section2) sodSquare = 1500 := by
  sorry

end NUMINAMATH_CALUDE_backyard_sod_coverage_l1467_146741


namespace NUMINAMATH_CALUDE_crayons_per_friend_l1467_146706

def total_crayons : ℕ := 210
def num_friends : ℕ := 30

theorem crayons_per_friend :
  total_crayons / num_friends = 7 :=
by sorry

end NUMINAMATH_CALUDE_crayons_per_friend_l1467_146706


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1467_146738

theorem regular_polygon_sides (n : ℕ) (h_exterior : (360 : ℝ) / n = 30) 
  (h_interior : (180 : ℝ) - 30 = 150) : n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1467_146738


namespace NUMINAMATH_CALUDE_inequality_proof_l1467_146715

theorem inequality_proof (x : ℝ) (h1 : 3/2 ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1467_146715


namespace NUMINAMATH_CALUDE_tree_height_after_three_years_l1467_146765

/-- The height of a tree after n years, given its initial height and growth factors -/
def tree_height (initial_height : ℝ) (n : ℕ) : ℝ :=
  if n ≤ 4 then
    initial_height * 3^n
  else
    initial_height * 3^4 * 2^(n - 4)

/-- Theorem: If a tree reaches 648 feet after 7 years with the given growth pattern,
    its height after 3 years was 27 feet -/
theorem tree_height_after_three_years
  (h : tree_height (tree_height 1 3) 4 = 648) :
  tree_height 1 3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_three_years_l1467_146765


namespace NUMINAMATH_CALUDE_factorial_calculation_l1467_146736

theorem factorial_calculation : (4 * Nat.factorial 6 + 24 * Nat.factorial 5) / Nat.factorial 7 = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l1467_146736


namespace NUMINAMATH_CALUDE_student_ticket_cost_l1467_146710

/-- Proves that the cost of each student ticket is 2 dollars given the conditions of the ticket sales -/
theorem student_ticket_cost (total_tickets : ℕ) (total_revenue : ℕ) 
  (nonstudent_price : ℕ) (student_tickets : ℕ) :
  total_tickets = 821 →
  total_revenue = 1933 →
  nonstudent_price = 3 →
  student_tickets = 530 →
  ∃ (student_price : ℕ),
    student_price * student_tickets + 
    nonstudent_price * (total_tickets - student_tickets) = total_revenue ∧
    student_price = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_student_ticket_cost_l1467_146710


namespace NUMINAMATH_CALUDE_geometric_series_equality_l1467_146787

def C (n : ℕ) : ℚ := 512 * (1 - (1/2)^n) / (1 - 1/2)

def D (n : ℕ) : ℚ := 1536 * (1 - (1/(-2))^n) / (1 + 1/2)

theorem geometric_series_equality :
  ∃ (n : ℕ), n > 0 ∧ C n = D n ∧ ∀ (m : ℕ), 0 < m ∧ m < n → C m ≠ D m :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l1467_146787


namespace NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l1467_146757

theorem power_of_product_equals_product_of_powers (a : ℝ) :
  (3 * a^3)^2 = 9 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_equals_product_of_powers_l1467_146757


namespace NUMINAMATH_CALUDE_intersection_exists_l1467_146790

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def B : Set ℝ := {y | ∃ x, y = 2^x}

-- State the theorem
theorem intersection_exists : ∃ z, z ∈ A ∩ B := by
  sorry

end NUMINAMATH_CALUDE_intersection_exists_l1467_146790


namespace NUMINAMATH_CALUDE_cone_radius_l1467_146776

/-- Given a cone with surface area 6 and lateral surface that unfolds into a semicircle,
    prove that the radius of its base is √(2/π) -/
theorem cone_radius (r : ℝ) (l : ℝ) : 
  r > 0 →  -- radius is positive
  l > 0 →  -- slant height is positive
  2 * π * r = π * l →  -- lateral surface unfolds into a semicircle
  π * r^2 + π * r * l = 6 →  -- surface area is 6
  r = Real.sqrt (2 / π) := by
sorry

end NUMINAMATH_CALUDE_cone_radius_l1467_146776


namespace NUMINAMATH_CALUDE_final_women_count_l1467_146768

/-- Represents the number of people in each category --/
structure Population :=
  (men : ℕ)
  (women : ℕ)
  (children : ℕ)
  (elderly : ℕ)

/-- Theorem stating the final number of women in the room --/
theorem final_women_count (initial : Population) 
  (h1 : initial.men + initial.women + initial.children + initial.elderly > 0)
  (h2 : initial.men = 4 * initial.elderly / 2)
  (h3 : initial.women = 5 * initial.elderly / 2)
  (h4 : initial.children = 3 * initial.elderly / 2)
  (h5 : initial.men + 2 = 14)
  (h6 : initial.children - 5 = 7)
  (h7 : initial.elderly - 3 = 6) :
  2 * (initial.women - 3) = 24 := by
  sorry

#check final_women_count

end NUMINAMATH_CALUDE_final_women_count_l1467_146768


namespace NUMINAMATH_CALUDE_exactly_one_incorrect_statement_l1467_146714

/-- Represents a statement about regression analysis -/
inductive RegressionStatement
  | residualBand
  | scatterPlotCorrelation
  | regressionLineInterpretation
  | sumSquaredResiduals

/-- Determines if a given statement about regression analysis is correct -/
def isCorrect (statement : RegressionStatement) : Prop :=
  match statement with
  | .residualBand => True
  | .scatterPlotCorrelation => False
  | .regressionLineInterpretation => True
  | .sumSquaredResiduals => True

theorem exactly_one_incorrect_statement :
  ∃! (s : RegressionStatement), ¬(isCorrect s) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_incorrect_statement_l1467_146714


namespace NUMINAMATH_CALUDE_graphing_calculator_count_l1467_146718

theorem graphing_calculator_count :
  ∀ (S G : ℕ),
    S + G = 45 →
    10 * S + 57 * G = 1625 →
    G = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_graphing_calculator_count_l1467_146718


namespace NUMINAMATH_CALUDE_highest_power_equals_carries_l1467_146779

/-- The number of carries when adding two natural numbers in a given base. -/
def num_carries (m n p : ℕ) : ℕ := sorry

/-- The highest power of p that divides the binomial coefficient (n+m choose m). -/
def highest_power_dividing_binom (n m p : ℕ) : ℕ := sorry

/-- Theorem stating the relationship between the highest power of p dividing
    (n+m choose m) and the number of carries when adding m and n in base p. -/
theorem highest_power_equals_carries (p m n : ℕ) (hp : Nat.Prime p) :
  highest_power_dividing_binom n m p = num_carries m n p :=
sorry

end NUMINAMATH_CALUDE_highest_power_equals_carries_l1467_146779


namespace NUMINAMATH_CALUDE_magic_square_solution_l1467_146751

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  f : ℕ
  g : ℕ
  h : ℕ
  i : ℕ
  magic_sum : ℕ
  row_sum : a + b + c = magic_sum ∧ d + e + f = magic_sum ∧ g + h + i = magic_sum
  col_sum : a + d + g = magic_sum ∧ b + e + h = magic_sum ∧ c + f + i = magic_sum
  diag_sum : a + e + i = magic_sum ∧ c + e + g = magic_sum

/-- Theorem: In a 3x3 magic square with top row entries x, 23, 102 and middle-left entry 5, x must equal 208 -/
theorem magic_square_solution (ms : MagicSquare) (h1 : ms.b = 23) (h2 : ms.c = 102) (h3 : ms.d = 5) : ms.a = 208 := by
  sorry


end NUMINAMATH_CALUDE_magic_square_solution_l1467_146751


namespace NUMINAMATH_CALUDE_smallest_with_ten_divisors_l1467_146734

/-- A function that returns the number of positive integer divisors of a given natural number. -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 10 positive integer divisors. -/
def has_ten_divisors (n : ℕ) : Prop := num_divisors n = 10

/-- Theorem stating that 48 is the smallest positive integer with exactly 10 positive integer divisors. -/
theorem smallest_with_ten_divisors : 
  has_ten_divisors 48 ∧ ∀ m : ℕ, m < 48 → ¬(has_ten_divisors m) :=
sorry

end NUMINAMATH_CALUDE_smallest_with_ten_divisors_l1467_146734


namespace NUMINAMATH_CALUDE_spheres_theorem_l1467_146784

/-- The configuration of four spheres -/
structure SpheresConfiguration where
  r : ℝ  -- radius of the three smaller spheres
  R : ℝ  -- radius of the larger sphere
  h : R > r  -- condition that R is greater than r

/-- The condition for the configuration to be possible -/
def configuration_possible (c : SpheresConfiguration) : Prop :=
  c.R ≥ (2 / Real.sqrt 3 - 1) * c.r

/-- The radius of the sphere tangent to all four spheres -/
noncomputable def tangent_sphere_radius (c : SpheresConfiguration) : ℝ :=
  let numerator := c.R * (c.R + c.r - Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3))
  let denominator := c.r + Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3) - c.R
  numerator / denominator

/-- The main theorem stating the conditions and the radius of the tangent sphere -/
theorem spheres_theorem (c : SpheresConfiguration) :
  configuration_possible c ∧
  tangent_sphere_radius c = (c.R * (c.R + c.r - Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3))) /
                            (c.r + Real.sqrt (c.R^2 + 2*c.R*c.r - c.r^2/3) - c.R) := by
  sorry

end NUMINAMATH_CALUDE_spheres_theorem_l1467_146784
