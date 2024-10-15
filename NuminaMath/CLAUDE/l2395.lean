import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_l2395_239539

theorem reciprocal_of_negative_one :
  (∃ x : ℝ, x * (-1) = 1) ∧ (∀ y : ℝ, y * (-1) = 1 → y = -1) :=
by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_l2395_239539


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l2395_239505

theorem smallest_angle_in_triangle (a b c : ℝ) (h1 : a = 4) (h2 : b = 3) (h3 : c = 2) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))
  C = Real.arccos (7/8) ∧ C ≤ Real.arccos ((b^2 + c^2 - a^2) / (2*b*c)) ∧ C ≤ Real.arccos ((a^2 + c^2 - b^2) / (2*a*c)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l2395_239505


namespace NUMINAMATH_CALUDE_tree_height_after_two_years_l2395_239514

/-- The height of a tree that triples its height every year -/
def tree_height (initial_height : ℝ) (years : ℕ) : ℝ :=
  initial_height * (3 ^ years)

/-- Theorem: A tree that triples its height every year and reaches 81 feet after 4 years
    will be 9 feet tall after 2 years -/
theorem tree_height_after_two_years
  (h : ∃ (initial_height : ℝ), tree_height initial_height 4 = 81) :
  ∃ (initial_height : ℝ), tree_height initial_height 2 = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_tree_height_after_two_years_l2395_239514


namespace NUMINAMATH_CALUDE_no_solution_for_f_l2395_239583

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Function f as defined in the problem -/
def f (n : ℕ+) : ℕ := (sumOfDigits n.val) * n.val

/-- 3-adic valuation of a natural number -/
def threeAdicVal (n : ℕ) : ℕ := sorry

/-- Main theorem: There is no positive integer n such that f(n) = 19091997 -/
theorem no_solution_for_f :
  ∀ n : ℕ+, f n ≠ 19091997 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_f_l2395_239583


namespace NUMINAMATH_CALUDE_new_person_weight_l2395_239554

/-- Proves that the weight of a new person is 77 kg when they replace a 65 kg person in a group of 8,
    causing the average weight to increase by 1.5 kg -/
theorem new_person_weight (n : ℕ) (old_weight new_weight avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : old_weight = 65)
  (h3 : avg_increase = 1.5) :
  new_weight = old_weight + n * avg_increase :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2395_239554


namespace NUMINAMATH_CALUDE_square_equation_solution_l2395_239597

theorem square_equation_solution (x : ℝ) (h : x^2 - 100 = -75) : x = -5 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_equation_solution_l2395_239597


namespace NUMINAMATH_CALUDE_new_average_production_l2395_239555

def past_days : ℕ := 14
def past_average : ℝ := 60
def today_production : ℝ := 90

theorem new_average_production :
  let total_past_production : ℝ := past_days * past_average
  let total_production : ℝ := total_past_production + today_production
  let new_average : ℝ := total_production / (past_days + 1)
  new_average = 62 := by sorry

end NUMINAMATH_CALUDE_new_average_production_l2395_239555


namespace NUMINAMATH_CALUDE_point_on_inverse_proportion_graph_l2395_239520

/-- Proves that the point (-2, 2) lies on the graph of the inverse proportion function y = -4/x -/
theorem point_on_inverse_proportion_graph :
  let f : ℝ → ℝ := λ x => -4 / x
  f (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_point_on_inverse_proportion_graph_l2395_239520


namespace NUMINAMATH_CALUDE_unique_pair_divisibility_l2395_239592

theorem unique_pair_divisibility (a b : ℕ) :
  a > 0 ∧ b > 0 ∧ (b^a ∣ a^b - 1) ↔ a = 3 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_divisibility_l2395_239592


namespace NUMINAMATH_CALUDE_line_through_point_with_triangle_area_l2395_239525

theorem line_through_point_with_triangle_area (x y : ℝ) :
  let P : ℝ × ℝ := (4/3, 2)
  let l : ℝ → ℝ → Prop := λ x y ↦ 6*x + 3*y - 14 = 0
  let A : ℝ × ℝ := (7/3, 0)
  let B : ℝ × ℝ := (0, 14/3)
  let O : ℝ × ℝ := (0, 0)
  l P.1 P.2 ∧
  l A.1 A.2 ∧
  l B.1 B.2 ∧
  A.1 > 0 ∧
  B.2 > 0 ∧
  (1/2 * A.1 * B.2 = 6) →
  l x y :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_triangle_area_l2395_239525


namespace NUMINAMATH_CALUDE_gcd_lcm_sum_l2395_239501

theorem gcd_lcm_sum : Nat.gcd 54 24 + Nat.lcm 48 18 = 150 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_sum_l2395_239501


namespace NUMINAMATH_CALUDE_simplify_complex_square_root_l2395_239559

theorem simplify_complex_square_root : 
  Real.sqrt ((9^8 + 3^14) / (9^6 + 3^15)) = Real.sqrt (15/14) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_square_root_l2395_239559


namespace NUMINAMATH_CALUDE_min_tokens_99x99_grid_l2395_239587

/-- Represents a grid of cells -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a square subgrid -/
structure Subgrid :=
  (size : ℕ)

/-- Calculates the minimum number of tokens required for a grid -/
def min_tokens (g : Grid) (sg : Subgrid) (tokens_per_subgrid : ℕ) : ℕ :=
  g.rows * g.cols - (g.rows / sg.size) * (g.cols / sg.size) * tokens_per_subgrid

/-- The main theorem stating the minimum number of tokens required -/
theorem min_tokens_99x99_grid : 
  let g : Grid := ⟨99, 99⟩
  let sg : Subgrid := ⟨4⟩
  let tokens_per_subgrid : ℕ := 8
  min_tokens g sg tokens_per_subgrid = 4801 := by
  sorry

#check min_tokens_99x99_grid

end NUMINAMATH_CALUDE_min_tokens_99x99_grid_l2395_239587


namespace NUMINAMATH_CALUDE_cylinder_diameter_l2395_239541

/-- The diameter of a cylinder given its height and volume -/
theorem cylinder_diameter (h : ℝ) (v : ℝ) (h_pos : h > 0) (v_pos : v > 0) :
  let d := 2 * Real.sqrt (9 / Real.pi)
  h = 5 ∧ v = 45 → d * d * Real.pi * h / 4 = v := by
  sorry

end NUMINAMATH_CALUDE_cylinder_diameter_l2395_239541


namespace NUMINAMATH_CALUDE_find_true_product_l2395_239519

theorem find_true_product (a b : ℕ) : 
  b = 2 * a →
  136 * (10 * b + a) = 136 * (10 * a + b) + 1224 →
  136 * (10 * a + b) = 1632 := by
sorry

end NUMINAMATH_CALUDE_find_true_product_l2395_239519


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2395_239560

-- Define the sets P and Q
def P : Set ℝ := {x : ℝ | x > 1}
def Q : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem intersection_of_P_and_Q : P ∩ Q = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2395_239560


namespace NUMINAMATH_CALUDE_train_crossing_time_l2395_239515

/-- Given a train crossing two platforms, calculate the time to cross the second platform -/
theorem train_crossing_time 
  (train_length : ℝ) 
  (platform1_length : ℝ) 
  (platform2_length : ℝ) 
  (time1 : ℝ) 
  (h1 : train_length = 270) 
  (h2 : platform1_length = 120) 
  (h3 : platform2_length = 250) 
  (h4 : time1 = 15) : 
  (train_length + platform2_length) / ((train_length + platform1_length) / time1) = 20 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_time_l2395_239515


namespace NUMINAMATH_CALUDE_overlapping_area_area_covered_by_both_strips_l2395_239562

/-- The area covered by both strips -/
def S : ℝ := 13.5

/-- The length of the original rectangular strip -/
def total_length : ℝ := 16

/-- The length of the left strip -/
def left_length : ℝ := 9

/-- The length of the right strip -/
def right_length : ℝ := 7

/-- The area covered only by the left strip -/
def left_area : ℝ := 27

/-- The area covered only by the right strip -/
def right_area : ℝ := 18

theorem overlapping_area :
  (left_area + S) / (right_area + S) = left_length / right_length :=
by sorry

theorem area_covered_by_both_strips : S = 13.5 :=
by sorry

end NUMINAMATH_CALUDE_overlapping_area_area_covered_by_both_strips_l2395_239562


namespace NUMINAMATH_CALUDE_seating_position_indeterminable_l2395_239585

/-- Represents a seat number as a pair of integers -/
def SeatNumber := ℤ × ℤ

/-- Represents a seating position as a row and column -/
structure SeatingPosition where
  row : ℤ
  column : ℤ

/-- Function that attempts to determine the seating position from a seat number -/
noncomputable def determineSeatingPosition (seatNumber : SeatNumber) : Option SeatingPosition :=
  sorry

/-- Theorem stating that it's not possible to determine the seating position
    from the seat number (2, 4) without additional information -/
theorem seating_position_indeterminable :
  ∀ (f : SeatNumber → Option SeatingPosition),
    ∃ (p1 p2 : SeatingPosition), p1 ≠ p2 ∧
      (f (2, 4) = some p1 ∨ f (2, 4) = some p2 ∨ f (2, 4) = none) :=
by
  sorry

end NUMINAMATH_CALUDE_seating_position_indeterminable_l2395_239585


namespace NUMINAMATH_CALUDE_secret_spread_theorem_l2395_239588

/-- The number of students who know the secret on a given day -/
def students_knowing_secret (day : ℕ) : ℕ :=
  (3^(day + 1) - 1) / 2

/-- The day of the week when 3280 students know the secret -/
def secret_spread_day : ℕ := 7

/-- Theorem stating that on the 7th day (Sunday), 3280 students know the secret -/
theorem secret_spread_theorem : 
  students_knowing_secret secret_spread_day = 3280 := by
  sorry

end NUMINAMATH_CALUDE_secret_spread_theorem_l2395_239588


namespace NUMINAMATH_CALUDE_largest_n_for_factorization_l2395_239572

theorem largest_n_for_factorization : 
  ∀ n : ℤ, 
  (∃ A B : ℤ, ∀ x : ℤ, 3 * x^2 + n * x + 72 = (3 * x + A) * (x + B)) →
  n ≤ 217 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_factorization_l2395_239572


namespace NUMINAMATH_CALUDE_base5_division_theorem_l2395_239528

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc
    else aux (m / 5) ((m % 5) :: acc)
  aux n []

theorem base5_division_theorem :
  let dividend := [4, 3, 0, 2]  -- 2034₅ in reverse order
  let divisor := [3, 2]         -- 23₅ in reverse order
  let quotient := [0, 4]        -- 40₅ in reverse order
  (base5ToBase10 dividend) / (base5ToBase10 divisor) = base5ToBase10 quotient :=
by sorry

end NUMINAMATH_CALUDE_base5_division_theorem_l2395_239528


namespace NUMINAMATH_CALUDE_probability_at_least_seven_three_times_l2395_239517

/-- The probability of rolling at least a seven on a single roll of an 8-sided die -/
def p : ℚ := 1/4

/-- The number of rolls -/
def n : ℕ := 4

/-- The probability of rolling at least a seven at least three times in four rolls of an 8-sided die -/
theorem probability_at_least_seven_three_times : 
  (Finset.sum (Finset.range 2) (λ k => (n.choose (n - k)) * p^(n - k) * (1 - p)^k)) = 13/256 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_seven_three_times_l2395_239517


namespace NUMINAMATH_CALUDE_theater_seats_l2395_239538

theorem theater_seats (people_watching : ℕ) (empty_seats : ℕ) : 
  people_watching = 532 → empty_seats = 218 → people_watching + empty_seats = 750 := by
  sorry

end NUMINAMATH_CALUDE_theater_seats_l2395_239538


namespace NUMINAMATH_CALUDE_division_problem_l2395_239530

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 171 →
  quotient = 8 →
  remainder = 3 →
  dividend = divisor * quotient + remainder →
  divisor = 21 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l2395_239530


namespace NUMINAMATH_CALUDE_garden_length_ratio_l2395_239531

/-- Given a rectangular property and a rectangular garden, this theorem proves
    the ratio of the garden's length to the property's length. -/
theorem garden_length_ratio
  (property_length : ℝ)
  (property_width : ℝ)
  (garden_area : ℝ)
  (h_property_length : property_length = 2250)
  (h_property_width : property_width = 1000)
  (h_garden_area : garden_area = 28125)
  (garden_width : ℝ)
  (h_garden_width_pos : garden_width > 0) :
  garden_area / garden_width / property_length = 12.5 / garden_width :=
by sorry

end NUMINAMATH_CALUDE_garden_length_ratio_l2395_239531


namespace NUMINAMATH_CALUDE_range_of_m_for_inverse_proposition_l2395_239598

theorem range_of_m_for_inverse_proposition : 
  ∀ m : ℝ, 
  (∀ x : ℝ, (1 < x ∧ x < 3) → (m < x ∧ x < m + 3)) → 
  (0 ≤ m ∧ m ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_for_inverse_proposition_l2395_239598


namespace NUMINAMATH_CALUDE_gcd_of_nine_digit_numbers_l2395_239595

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n ≤ 999999999) ∧
  ∃ (d₁ d₂ d₃ d₄ d₅ d₆ d₇ d₈ d₉ : ℕ),
    d₁ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₂ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₃ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₄ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₅ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₆ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₇ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₈ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₉ ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ) ∧
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₁ ≠ d₅ ∧ d₁ ≠ d₆ ∧ d₁ ≠ d₇ ∧ d₁ ≠ d₈ ∧ d₁ ≠ d₉ ∧
    d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₂ ≠ d₅ ∧ d₂ ≠ d₆ ∧ d₂ ≠ d₇ ∧ d₂ ≠ d₈ ∧ d₂ ≠ d₉ ∧
    d₃ ≠ d₄ ∧ d₃ ≠ d₅ ∧ d₃ ≠ d₆ ∧ d₃ ≠ d₇ ∧ d₃ ≠ d₈ ∧ d₃ ≠ d₉ ∧
    d₄ ≠ d₅ ∧ d₄ ≠ d₆ ∧ d₄ ≠ d₇ ∧ d₄ ≠ d₈ ∧ d₄ ≠ d₉ ∧
    d₅ ≠ d₆ ∧ d₅ ≠ d₇ ∧ d₅ ≠ d₈ ∧ d₅ ≠ d₉ ∧
    d₆ ≠ d₇ ∧ d₆ ≠ d₈ ∧ d₆ ≠ d₉ ∧
    d₇ ≠ d₈ ∧ d₇ ≠ d₉ ∧
    d₈ ≠ d₉ ∧
    n = d₁ * 100000000 + d₂ * 10000000 + d₃ * 1000000 + d₄ * 100000 + d₅ * 10000 + d₆ * 1000 + d₇ * 100 + d₈ * 10 + d₉

theorem gcd_of_nine_digit_numbers :
  ∃ (g : ℕ), g > 0 ∧ (∀ (n : ℕ), is_valid_number n → g ∣ n) ∧
  (∀ (d : ℕ), d > 0 → (∀ (n : ℕ), is_valid_number n → d ∣ n) → d ≤ g) ∧
  g = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_nine_digit_numbers_l2395_239595


namespace NUMINAMATH_CALUDE_strawberry_distribution_l2395_239547

/-- Represents the distribution of strawberries in buckets -/
structure StrawberryDistribution where
  buckets : Fin 5 → ℕ

/-- The initial distribution of strawberries -/
def initial_distribution : StrawberryDistribution :=
  { buckets := λ _ => 60 }

/-- Removes a specified number of strawberries from each bucket -/
def remove_from_each (d : StrawberryDistribution) (amount : ℕ) : StrawberryDistribution :=
  { buckets := λ i => d.buckets i - amount }

/-- Adds strawberries to specific buckets -/
def add_to_buckets (d : StrawberryDistribution) (additions : Fin 5 → ℕ) : StrawberryDistribution :=
  { buckets := λ i => d.buckets i + additions i }

/-- The final distribution of strawberries after all adjustments -/
def final_distribution : StrawberryDistribution :=
  add_to_buckets
    (remove_from_each initial_distribution 20)
    (λ i => match i with
      | 0 => 15
      | 1 => 15
      | 2 => 25
      | _ => 0)

/-- Theorem stating the final distribution of strawberries -/
theorem strawberry_distribution :
  final_distribution.buckets = λ i => match i with
    | 0 => 55
    | 1 => 55
    | 2 => 65
    | _ => 40 := by sorry

end NUMINAMATH_CALUDE_strawberry_distribution_l2395_239547


namespace NUMINAMATH_CALUDE_last_four_average_l2395_239590

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  (list.sum / list.length) = 65 →
  ((list.take 3).sum / 3) = 60 →
  ((list.drop 3).sum / 4) = 68.75 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l2395_239590


namespace NUMINAMATH_CALUDE_probability_same_group_l2395_239574

def total_items : ℕ := 6
def group_size : ℕ := 2
def num_groups : ℕ := 3
def items_to_choose : ℕ := 2

theorem probability_same_group :
  (num_groups * (group_size.choose items_to_choose)) / (total_items.choose items_to_choose) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_group_l2395_239574


namespace NUMINAMATH_CALUDE_min_value_quadratic_roots_l2395_239551

theorem min_value_quadratic_roots (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + a*x + b = 0) →
  (∃ x : ℝ, x^2 + b*x + a = 0) →
  3*a + 2*b ≥ 20 := by
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_roots_l2395_239551


namespace NUMINAMATH_CALUDE_complex_fraction_value_l2395_239594

theorem complex_fraction_value (a : ℝ) (z : ℂ) : 
  z = (a^2 - 1) + (a - 1) * Complex.I → z.re = 0 → (a^2 + Complex.I) / (1 + a * Complex.I) = Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_value_l2395_239594


namespace NUMINAMATH_CALUDE_undefined_expression_l2395_239516

theorem undefined_expression (y : ℝ) : 
  y^2 - 16*y + 64 = 0 → y = 8 :=
by sorry

end NUMINAMATH_CALUDE_undefined_expression_l2395_239516


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2395_239593

theorem arithmetic_calculation : 14 - (-12) + (-25) - 17 = -16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2395_239593


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l2395_239546

theorem perfect_square_quadratic (m : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, x^2 + (m + 2) * x + 36 = y^2) →
  m = 10 ∨ m = -14 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l2395_239546


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2395_239584

/-- Given an equilateral triangle where the area is numerically twice the length of one of its sides,
    the perimeter of the triangle is 8√3 units. -/
theorem equilateral_triangle_perimeter : ∀ s : ℝ,
  s > 0 →  -- side length is positive
  (s^2 * Real.sqrt 3) / 4 = 2 * s →  -- area is twice the side length
  3 * s = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l2395_239584


namespace NUMINAMATH_CALUDE_count_numeric_hex_500_l2395_239506

/-- Converts a positive integer to its hexadecimal representation --/
def to_hex (n : ℕ+) : List (Fin 16) :=
  sorry

/-- Checks if a hexadecimal digit is numeric (0-9) --/
def is_numeric_hex_digit (d : Fin 16) : Bool :=
  d.val < 10

/-- Checks if a hexadecimal number contains only numeric digits --/
def has_only_numeric_digits (h : List (Fin 16)) : Bool :=
  h.all is_numeric_hex_digit

/-- Counts numbers with only numeric hexadecimal digits up to n --/
def count_numeric_hex (n : ℕ+) : ℕ :=
  (List.range n).filter (fun i => has_only_numeric_digits (to_hex ⟨i + 1, by sorry⟩)) |>.length

theorem count_numeric_hex_500 : count_numeric_hex 500 = 199 :=
  sorry

end NUMINAMATH_CALUDE_count_numeric_hex_500_l2395_239506


namespace NUMINAMATH_CALUDE_semicircle_radius_in_trapezoid_l2395_239563

/-- A trapezoid with specific measurements and an inscribed semicircle. -/
structure TrapezoidWithSemicircle where
  -- Define the trapezoid
  AB : ℝ
  CD : ℝ
  side1 : ℝ
  side2 : ℝ
  -- Conditions
  AB_eq : AB = 27
  CD_eq : CD = 45
  side1_eq : side1 = 13
  side2_eq : side2 = 15
  -- Semicircle properties
  semicircle_diameter : ℝ
  semicircle_diameter_eq : semicircle_diameter = AB
  tangential_to_CD : Bool -- represents that the semicircle is tangential to CD

/-- The radius of the semicircle in the trapezoid is 13.5. -/
theorem semicircle_radius_in_trapezoid (t : TrapezoidWithSemicircle) :
  t.semicircle_diameter / 2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_radius_in_trapezoid_l2395_239563


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l2395_239567

theorem fraction_sum_equality : (3 : ℚ) / 5 - 1 / 10 + 2 / 15 = 19 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l2395_239567


namespace NUMINAMATH_CALUDE_fourth_root_simplification_l2395_239535

theorem fourth_root_simplification :
  (3^7 * 5^3 : ℝ)^(1/4) = 3 * (135 : ℝ)^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_simplification_l2395_239535


namespace NUMINAMATH_CALUDE_jqk_base14_to_binary_digits_l2395_239553

def base14_to_decimal (j k q : ℕ) : ℕ := j * 14^2 + q * 14 + k

def count_binary_digits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

theorem jqk_base14_to_binary_digits : 
  count_binary_digits (base14_to_decimal 11 13 12) = 11 := by
  sorry

end NUMINAMATH_CALUDE_jqk_base14_to_binary_digits_l2395_239553


namespace NUMINAMATH_CALUDE_juan_peter_speed_difference_l2395_239504

/-- The speed difference between Juan and Peter -/
def speed_difference (juan_speed peter_speed : ℝ) : ℝ :=
  juan_speed - peter_speed

/-- The total distance traveled by Juan and Peter -/
def total_distance (juan_speed peter_speed : ℝ) (time : ℝ) : ℝ :=
  (juan_speed + peter_speed) * time

theorem juan_peter_speed_difference :
  ∃ (juan_speed : ℝ),
    speed_difference juan_speed 5.0 = 3 ∧
    total_distance juan_speed 5.0 1.5 = 19.5 := by
  sorry

end NUMINAMATH_CALUDE_juan_peter_speed_difference_l2395_239504


namespace NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l2395_239569

def grid_size : Nat := 4

-- Define the number of ways to choose 2 items from n items
def choose_two (n : Nat) : Nat :=
  n * (n - 1) / 2

-- Define the number of rectangles in a square grid
def num_rectangles (n : Nat) : Nat :=
  (choose_two n) * (choose_two n)

-- Theorem statement
theorem rectangles_in_4x4_grid :
  num_rectangles grid_size = 36 := by
  sorry

end NUMINAMATH_CALUDE_rectangles_in_4x4_grid_l2395_239569


namespace NUMINAMATH_CALUDE_solution_of_equation_l2395_239508

theorem solution_of_equation (x : ℝ) : 
  (Real.sqrt (3 * x + 1) + Real.sqrt (3 * x + 6) = Real.sqrt (4 * x - 2) + Real.sqrt (4 * x + 3)) → x = 3 :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l2395_239508


namespace NUMINAMATH_CALUDE_cube_difference_l2395_239566

theorem cube_difference (a b : ℝ) (h1 : a - b = 7) (h2 : a^2 + b^2 = 40) : 
  a^3 - b^3 = 248.5 := by sorry

end NUMINAMATH_CALUDE_cube_difference_l2395_239566


namespace NUMINAMATH_CALUDE_max_handshakes_l2395_239564

theorem max_handshakes (n : ℕ) (h : n = 60) : n * (n - 1) / 2 = 1770 := by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_l2395_239564


namespace NUMINAMATH_CALUDE_intersection_equals_open_interval_l2395_239576

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the open interval (0, 1)
def open_interval_zero_one : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem statement
theorem intersection_equals_open_interval : M ∩ N = open_interval_zero_one := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_open_interval_l2395_239576


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l2395_239543

def num_black_balls : ℕ := 6
def num_white_balls : ℕ := 5

def total_balls : ℕ := num_black_balls + num_white_balls

theorem probability_of_white_ball :
  (num_white_balls : ℚ) / (total_balls : ℚ) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l2395_239543


namespace NUMINAMATH_CALUDE_smallest_x_value_l2395_239540

theorem smallest_x_value (y : ℕ+) (x : ℕ) 
  (h : (857 : ℚ) / 1000 = (y : ℚ) / ((210 : ℚ) + x)) : 
  ∀ x' : ℕ, x' ≥ x → x = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2395_239540


namespace NUMINAMATH_CALUDE_cos_sum_24_144_264_l2395_239550

theorem cos_sum_24_144_264 :
  Real.cos (24 * π / 180) + Real.cos (144 * π / 180) + Real.cos (264 * π / 180) =
    (3 - Real.sqrt 5) / 4 - Real.sin (3 * π / 180) * Real.sqrt (10 + 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_24_144_264_l2395_239550


namespace NUMINAMATH_CALUDE_zero_is_self_opposite_l2395_239533

/-- Two real numbers are opposite if they have the same magnitude but opposite signs, or both are zero. -/
def are_opposite (a b : ℝ) : Prop := (a = -b) ∨ (a = 0 ∧ b = 0)

/-- Zero is its own opposite number. -/
theorem zero_is_self_opposite : are_opposite 0 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_is_self_opposite_l2395_239533


namespace NUMINAMATH_CALUDE_triangle_side_length_l2395_239579

theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  c = 2 → b = 2 * a → Real.cos C = (1 : ℝ) / 4 → 
  (a^2 + b^2 - c^2) / (2 * a * b) = Real.cos C → a = 1 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2395_239579


namespace NUMINAMATH_CALUDE_train_distance_problem_l2395_239510

theorem train_distance_problem (v1 v2 d : ℝ) (h1 : v1 = 20) (h2 : v2 = 25) (h3 : d = 70) :
  let t := d / (v2 - v1)
  let x := v1 * t
  (x + (x + d)) = 630 := by sorry

end NUMINAMATH_CALUDE_train_distance_problem_l2395_239510


namespace NUMINAMATH_CALUDE_melody_civics_pages_l2395_239561

/-- The number of pages Melody needs to read for her English class -/
def english_pages : ℕ := 20

/-- The number of pages Melody needs to read for her Science class -/
def science_pages : ℕ := 16

/-- The number of pages Melody needs to read for her Chinese class -/
def chinese_pages : ℕ := 12

/-- The fraction of pages Melody will read tomorrow for each class -/
def read_fraction : ℚ := 1/4

/-- The total number of pages Melody will read tomorrow -/
def total_pages_tomorrow : ℕ := 14

/-- The number of pages Melody needs to read for her Civics class -/
def civics_pages : ℕ := 8

theorem melody_civics_pages :
  (english_pages : ℚ) * read_fraction +
  (science_pages : ℚ) * read_fraction +
  (chinese_pages : ℚ) * read_fraction +
  (civics_pages : ℚ) * read_fraction = total_pages_tomorrow :=
sorry

end NUMINAMATH_CALUDE_melody_civics_pages_l2395_239561


namespace NUMINAMATH_CALUDE_extended_quadrilateral_area_l2395_239512

/-- Represents a quadrilateral with extended sides -/
structure ExtendedQuadrilateral where
  -- The area of the original quadrilateral
  area : ℝ
  -- The lengths of the sides and their extensions
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ

/-- Theorem stating the area of the extended quadrilateral -/
theorem extended_quadrilateral_area (q : ExtendedQuadrilateral)
  (h_area : q.area = 25)
  (h_ef : q.ef = 5)
  (h_fg : q.fg = 7)
  (h_gh : q.gh = 9)
  (h_he : q.he = 8) :
  q.area + 2 * q.area = 75 := by
  sorry

end NUMINAMATH_CALUDE_extended_quadrilateral_area_l2395_239512


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l2395_239596

theorem ellipse_eccentricity (a b x₀ y₀ : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : x₀^2 / a^2 + y₀^2 / b^2 = 1) 
  (h₄ : y₀^2 / ((x₀ + a) * (a - x₀)) = 1/3) : 
  Real.sqrt (1 - b^2 / a^2) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l2395_239596


namespace NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_12_l2395_239578

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem no_two_digit_primes_with_digit_sum_12 :
  ∀ n : ℕ, is_two_digit n → Nat.Prime n → digit_sum n = 12 → False :=
by
  sorry

end NUMINAMATH_CALUDE_no_two_digit_primes_with_digit_sum_12_l2395_239578


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l2395_239537

theorem arithmetic_sequence_term_count 
  (a₁ aₙ d : ℤ) 
  (h₁ : a₁ = -25)
  (h₂ : aₙ = 96)
  (h₃ : d = 7)
  (h₄ : aₙ = a₁ + (n - 1) * d)
  : n = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_count_l2395_239537


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l2395_239580

/-- The minimum distance between two points on different curves with the same y-coordinate -/
theorem min_distance_between_curves : ∃ (min_dist : ℝ),
  (∀ (a x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 →
    a = 2 * (x₁ + 1) →
    a = x₂ + Real.log x₂ →
    |x₂ - x₁| ≥ min_dist) ∧
  (∃ (a x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧
    a = 2 * (x₁ + 1) ∧
    a = x₂ + Real.log x₂ ∧
    |x₂ - x₁| = min_dist) ∧
  min_dist = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l2395_239580


namespace NUMINAMATH_CALUDE_sticker_problem_l2395_239545

theorem sticker_problem (initial_stickers : ℚ) : 
  let lost_stickers := (1 : ℚ) / 3 * initial_stickers
  let found_stickers := (3 : ℚ) / 4 * lost_stickers
  let remaining_stickers := initial_stickers - lost_stickers + found_stickers
  initial_stickers - remaining_stickers = (1 : ℚ) / 12 * initial_stickers :=
by sorry

end NUMINAMATH_CALUDE_sticker_problem_l2395_239545


namespace NUMINAMATH_CALUDE_min_value_theorem_l2395_239502

theorem min_value_theorem (x : ℝ) (h : x > 2) :
  x + 4 / (x - 2) ≥ 6 ∧ (x + 4 / (x - 2) = 6 ↔ x = 4) := by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2395_239502


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2395_239518

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (10 + Real.sqrt x) = 4 → x = 36 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2395_239518


namespace NUMINAMATH_CALUDE_order_relationship_l2395_239582

theorem order_relationship (a b c : ℝ) : 
  a = Real.exp (1/2) - 1 → 
  b = Real.log (3/2) → 
  c = 5/12 → 
  a > c ∧ c > b := by
sorry

end NUMINAMATH_CALUDE_order_relationship_l2395_239582


namespace NUMINAMATH_CALUDE_no_real_solutions_implies_a_less_than_one_l2395_239503

theorem no_real_solutions_implies_a_less_than_one :
  (∀ x : ℝ, ¬∃ (y : ℝ), y^2 = x + 4 ∧ y = a - 1) → a < 1 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_implies_a_less_than_one_l2395_239503


namespace NUMINAMATH_CALUDE_expression_simplification_l2395_239532

theorem expression_simplification (n : ℝ) (h : n = Real.sqrt 2 + 1) :
  ((n + 3) / (n^2 - 1) - 1 / (n + 1)) / (2 / (n + 1)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2395_239532


namespace NUMINAMATH_CALUDE_fraction_simplification_l2395_239526

theorem fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hsum : a + b + c ≠ 0) :
  (a^2 + a*b - b^2 + a*c) / (b^2 + b*c - c^2 + b*a) = (a - b) / (b - c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2395_239526


namespace NUMINAMATH_CALUDE_arithmetic_mean_sqrt3_sqrt2_l2395_239536

theorem arithmetic_mean_sqrt3_sqrt2 :
  let a := Real.sqrt 3 + Real.sqrt 2
  let b := Real.sqrt 3 - Real.sqrt 2
  (a + b) / 2 = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_sqrt3_sqrt2_l2395_239536


namespace NUMINAMATH_CALUDE_three_digit_base_problem_l2395_239511

theorem three_digit_base_problem :
  ∃! (x y z b : ℕ),
    x * b^2 + y * b + z = 1993 ∧
    x + y + z = 22 ∧
    x < b ∧ y < b ∧ z < b ∧
    b > 10 ∧
    x = 2 ∧ y = 15 ∧ z = 5 ∧ b = 28 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_base_problem_l2395_239511


namespace NUMINAMATH_CALUDE_existence_of_sum_of_cubes_l2395_239544

theorem existence_of_sum_of_cubes :
  ∃ (a b c d : ℕ), a^3 + b^3 + c^3 + d^3 = 100^100 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_sum_of_cubes_l2395_239544


namespace NUMINAMATH_CALUDE_answer_determines_sanity_not_species_l2395_239524

-- Define the species of the interlocutor
inductive Species
| Human
| Ghoul

-- Define the mental state of the interlocutor
inductive MentalState
| Sane
| Insane

-- Define the possible answers
inductive Answer
| Yes
| No

-- Define a function that determines the answer based on species and mental state
def getAnswer (s : Species) (m : MentalState) : Answer :=
  match m with
  | MentalState.Sane => Answer.Yes
  | MentalState.Insane => Answer.No

-- Theorem stating that the answer determines sanity but not species
theorem answer_determines_sanity_not_species :
  ∀ (s1 s2 : Species) (m1 m2 : MentalState),
    getAnswer s1 m1 = getAnswer s2 m2 →
    m1 = m2 ∧ (s1 = s2 ∨ s1 ≠ s2) :=
by sorry

end NUMINAMATH_CALUDE_answer_determines_sanity_not_species_l2395_239524


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l2395_239565

theorem max_value_sum_of_roots (a b c : ℝ) 
  (sum_eq : a + b + c = 3)
  (a_ge : a ≥ -1)
  (b_ge : b ≥ -2)
  (c_ge : c ≥ -3) :
  ∃ (max : ℝ), max = 3 * Real.sqrt 34 ∧
    Real.sqrt (2 * a + 2) + Real.sqrt (4 * b + 8) + Real.sqrt (6 * c + 18) ≤ max ∧
    ∃ (a' b' c' : ℝ), a' + b' + c' = 3 ∧ a' ≥ -1 ∧ b' ≥ -2 ∧ c' ≥ -3 ∧
      Real.sqrt (2 * a' + 2) + Real.sqrt (4 * b' + 8) + Real.sqrt (6 * c' + 18) = max :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l2395_239565


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_third_l2395_239577

theorem opposite_of_negative_one_third :
  let x : ℚ := -1/3
  let opposite (y : ℚ) : ℚ := -y
  opposite x = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_third_l2395_239577


namespace NUMINAMATH_CALUDE_fourth_side_length_is_correct_l2395_239581

/-- A quadrilateral inscribed in a circle with radius 150√3, where three sides have length 300 --/
structure InscribedQuadrilateral where
  radius : ℝ
  three_side_length : ℝ
  h_radius : radius = 150 * Real.sqrt 3
  h_three_sides : three_side_length = 300

/-- The length of the fourth side of the inscribed quadrilateral --/
def fourth_side_length (q : InscribedQuadrilateral) : ℝ := 562.5

/-- Theorem stating that the fourth side length is correct --/
theorem fourth_side_length_is_correct (q : InscribedQuadrilateral) :
  fourth_side_length q = 562.5 := by
  sorry

end NUMINAMATH_CALUDE_fourth_side_length_is_correct_l2395_239581


namespace NUMINAMATH_CALUDE_factor_implies_p_value_l2395_239507

theorem factor_implies_p_value (m p : ℤ) : 
  (∃ k : ℤ, m^2 - p*m - 24 = (m - 8) * k) → p = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_p_value_l2395_239507


namespace NUMINAMATH_CALUDE_total_roses_l2395_239513

/-- The total number of roses in a special n-gon garden -/
def roseCount (n : ℕ) : ℕ :=
  Nat.choose n 4 + Nat.choose (n - 1) 2

/-- Properties of the rose garden -/
structure RoseGarden (n : ℕ) where
  convex : n ≥ 4
  redRoses : Fin n → Unit  -- One red rose at each vertex
  paths : Fin n → Fin n → Unit  -- Path between each pair of vertices
  noTripleIntersection : Unit  -- No three paths intersect at a single point
  whiteRoses : Unit  -- One white/black rose in each region

/-- Theorem: The total number of roses in the garden is given by roseCount -/
theorem total_roses (n : ℕ) (garden : RoseGarden n) : 
  (Fin n → Unit) × Unit → ℕ :=
by sorry

end NUMINAMATH_CALUDE_total_roses_l2395_239513


namespace NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l2395_239527

theorem factor_81_minus_27x_cubed (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3*x + x^2) := by
  sorry

end NUMINAMATH_CALUDE_factor_81_minus_27x_cubed_l2395_239527


namespace NUMINAMATH_CALUDE_dave_initial_apps_l2395_239568

/-- The number of apps Dave had initially -/
def initial_apps : ℕ := sorry

/-- The number of apps Dave had after adding one -/
def apps_after_adding : ℕ := 18

theorem dave_initial_apps : 
  initial_apps = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_dave_initial_apps_l2395_239568


namespace NUMINAMATH_CALUDE_count_odd_coefficients_l2395_239509

/-- The number of odd coefficients in (x^2 + x + 1)^n -/
def odd_coefficients (n : ℕ+) : ℕ :=
  (2^n.val - 1) / 3 * 4 + 1

/-- Theorem stating the number of odd coefficients in (x^2 + x + 1)^n -/
theorem count_odd_coefficients (n : ℕ+) :
  odd_coefficients n = (2^n.val - 1) / 3 * 4 + 1 :=
by sorry

end NUMINAMATH_CALUDE_count_odd_coefficients_l2395_239509


namespace NUMINAMATH_CALUDE_correct_average_marks_l2395_239548

/-- Proves that the correct average marks for a class of 50 students is 82.8,
    given an initial average of 85 and three incorrectly recorded marks. -/
theorem correct_average_marks
  (num_students : ℕ)
  (initial_average : ℚ)
  (incorrect_mark1 incorrect_mark2 incorrect_mark3 : ℕ)
  (correct_mark1 correct_mark2 correct_mark3 : ℕ)
  (h_num_students : num_students = 50)
  (h_initial_average : initial_average = 85)
  (h_incorrect1 : incorrect_mark1 = 95)
  (h_incorrect2 : incorrect_mark2 = 78)
  (h_incorrect3 : incorrect_mark3 = 120)
  (h_correct1 : correct_mark1 = 45)
  (h_correct2 : correct_mark2 = 58)
  (h_correct3 : correct_mark3 = 80) :
  (num_students : ℚ) * initial_average - (incorrect_mark1 - correct_mark1 + incorrect_mark2 - correct_mark2 + incorrect_mark3 - correct_mark3 : ℚ) / num_students = 82.8 :=
by
  sorry

end NUMINAMATH_CALUDE_correct_average_marks_l2395_239548


namespace NUMINAMATH_CALUDE_ians_money_left_l2395_239542

/-- Calculates Ian's remaining money after expenses and taxes -/
def ians_remaining_money (total_hours : ℕ) (first_rate second_rate : ℚ) 
  (spending_ratio tax_rate : ℚ) (monthly_expense : ℚ) : ℚ :=
  let total_earnings := (first_rate * (total_hours / 2 : ℚ)) + (second_rate * (total_hours / 2 : ℚ))
  let spending := total_earnings * spending_ratio
  let taxes := total_earnings * tax_rate
  let total_deductions := spending + taxes + monthly_expense
  total_earnings - total_deductions

theorem ians_money_left :
  ians_remaining_money 8 18 22 (1/2) (1/10) 50 = 14 := by
  sorry

end NUMINAMATH_CALUDE_ians_money_left_l2395_239542


namespace NUMINAMATH_CALUDE_prob_ace_king_queen_same_suit_l2395_239523

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of cards dealt -/
def CardsDealt : ℕ := 3

/-- Represents the probability of drawing a specific Ace from a standard deck -/
def ProbFirstAce : ℚ := 1 / StandardDeck

/-- Represents the probability of drawing a specific King after an Ace is drawn -/
def ProbSecondKing : ℚ := 1 / (StandardDeck - 1)

/-- Represents the probability of drawing a specific Queen after an Ace and a King are drawn -/
def ProbThirdQueen : ℚ := 1 / (StandardDeck - 2)

/-- The probability of dealing an Ace, King, and Queen of the same suit in that order -/
def ProbAceKingQueen : ℚ := ProbFirstAce * ProbSecondKing * ProbThirdQueen

theorem prob_ace_king_queen_same_suit :
  ProbAceKingQueen = 1 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_queen_same_suit_l2395_239523


namespace NUMINAMATH_CALUDE_prob_snow_at_least_one_day_l2395_239557

-- Define the probabilities
def prob_snow_friday : ℝ := 0.30
def prob_snow_monday : ℝ := 0.45

-- Theorem statement
theorem prob_snow_at_least_one_day : 
  let prob_no_snow_friday := 1 - prob_snow_friday
  let prob_no_snow_monday := 1 - prob_snow_monday
  let prob_no_snow_both := prob_no_snow_friday * prob_no_snow_monday
  1 - prob_no_snow_both = 0.615 := by
  sorry

end NUMINAMATH_CALUDE_prob_snow_at_least_one_day_l2395_239557


namespace NUMINAMATH_CALUDE_misread_number_calculation_l2395_239529

theorem misread_number_calculation (n : ℕ) (initial_avg correct_avg wrong_num : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 23)
  (h3 : correct_avg = 24)
  (h4 : wrong_num = 26) : 
  ∃ (actual_num : ℚ), 
    (n : ℚ) * correct_avg - (n : ℚ) * initial_avg = actual_num - wrong_num ∧ 
    actual_num = 36 := by
  sorry

end NUMINAMATH_CALUDE_misread_number_calculation_l2395_239529


namespace NUMINAMATH_CALUDE_irregular_shape_area_l2395_239500

/-- The area of an irregular shape formed by removing a smaller rectangle and a right triangle from a larger rectangle --/
theorem irregular_shape_area (large_length large_width small_length small_width triangle_base triangle_height : ℝ)
  (h1 : large_length = 10)
  (h2 : large_width = 6)
  (h3 : small_length = 4)
  (h4 : small_width = 3)
  (h5 : triangle_base = small_length)
  (h6 : triangle_height = 3) :
  large_length * large_width - (small_length * small_width + 1/2 * triangle_base * triangle_height) = 42 := by
  sorry

end NUMINAMATH_CALUDE_irregular_shape_area_l2395_239500


namespace NUMINAMATH_CALUDE_unique_solution_l2395_239573

/-- Represents a single-digit integer (0 to 9) -/
def SingleDigit : Type := {n : ℕ // n ≤ 9}

/-- The equation 38A - B1 = 364 holds for single-digit integers A and B -/
def EquationHolds (A B : SingleDigit) : Prop :=
  380 + A.val - 10 * B.val - 1 = 364

theorem unique_solution :
  ∃! (A B : SingleDigit), EquationHolds A B ∧ A.val = 5 ∧ B.val = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2395_239573


namespace NUMINAMATH_CALUDE_red_blue_difference_after_border_l2395_239534

/-- Represents a hexagonal figure with blue and red tiles -/
structure HexFigure where
  blue_tiles : ℕ
  red_tiles : ℕ

/-- Adds a border to a hexagonal figure, alternating between blue and red tiles -/
def add_border (fig : HexFigure) : HexFigure :=
  { blue_tiles := fig.blue_tiles + 12,
    red_tiles := fig.red_tiles + 12 }

/-- The initial hexagonal figure -/
def initial_figure : HexFigure :=
  { blue_tiles := 10,
    red_tiles := 20 }

theorem red_blue_difference_after_border :
  (add_border initial_figure).red_tiles - (add_border initial_figure).blue_tiles = 10 := by
  sorry

end NUMINAMATH_CALUDE_red_blue_difference_after_border_l2395_239534


namespace NUMINAMATH_CALUDE_laticia_socks_l2395_239586

def sock_problem (nephew_socks week1_socks week2_extra week3_fraction week4_decrease : ℕ) : Prop :=
  let week2_socks := week1_socks + week2_extra
  let week3_socks := (week1_socks + week2_socks) / 2
  let week4_socks := week3_socks - week4_decrease
  nephew_socks + week1_socks + week2_socks + week3_socks + week4_socks = 57

theorem laticia_socks : 
  sock_problem 4 12 4 2 3 := by sorry

end NUMINAMATH_CALUDE_laticia_socks_l2395_239586


namespace NUMINAMATH_CALUDE_coefficient_of_monomial_l2395_239522

theorem coefficient_of_monomial (a b : ℝ) :
  let expression := (4 * Real.pi * a^2 * b) / 5
  let coefficient := -(4 / 5) * Real.pi
  coefficient = expression / (a^2 * b) := by sorry

end NUMINAMATH_CALUDE_coefficient_of_monomial_l2395_239522


namespace NUMINAMATH_CALUDE_segment_length_product_l2395_239570

theorem segment_length_product (a : ℝ) : 
  (∃ a₁ a₂ : ℝ, 
    (∀ a, ((3*a - 5)^2 + (2*a - 4)^2 = 34) ↔ (a = a₁ ∨ a = a₂)) ∧
    (a₁ * a₂ = -722/169)) :=
by sorry

end NUMINAMATH_CALUDE_segment_length_product_l2395_239570


namespace NUMINAMATH_CALUDE_expected_value_of_sum_is_twelve_l2395_239591

def marbles : Finset ℕ := Finset.range 7

def choose_three (s : Finset ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ subset => subset.card = 3)

def sum_of_subset (subset : Finset ℕ) : ℕ :=
  subset.sum id

theorem expected_value_of_sum_is_twelve :
  let all_choices := choose_three marbles
  let sum_of_sums := all_choices.sum sum_of_subset
  let num_choices := all_choices.card
  (sum_of_sums : ℚ) / num_choices = 12 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_sum_is_twelve_l2395_239591


namespace NUMINAMATH_CALUDE_amc8_participants_l2395_239521

/-- The number of mathematics students at Euclid Middle School taking the AMC 8 contest -/
def total_students (germain newton young gauss : ℕ) : ℕ :=
  germain + newton + young + gauss

/-- Theorem stating that the total number of students taking the AMC 8 contest is 38 -/
theorem amc8_participants : total_students 12 10 9 7 = 38 := by
  sorry

end NUMINAMATH_CALUDE_amc8_participants_l2395_239521


namespace NUMINAMATH_CALUDE_total_discount_calculation_l2395_239552

theorem total_discount_calculation (cost_price_A cost_price_B cost_price_C : ℝ)
  (markup_percentage : ℝ) (loss_percentage_A loss_percentage_B loss_percentage_C : ℝ)
  (h1 : cost_price_A = 200)
  (h2 : cost_price_B = 150)
  (h3 : cost_price_C = 100)
  (h4 : markup_percentage = 0.5)
  (h5 : loss_percentage_A = 0.01)
  (h6 : loss_percentage_B = 0.03)
  (h7 : loss_percentage_C = 0.02) :
  let marked_price (cp : ℝ) := cp * (1 + markup_percentage)
  let selling_price (cp : ℝ) (loss : ℝ) := cp * (1 - loss)
  let discount (cp : ℝ) (loss : ℝ) := marked_price cp - selling_price cp loss
  discount cost_price_A loss_percentage_A +
  discount cost_price_B loss_percentage_B +
  discount cost_price_C loss_percentage_C = 233.5 :=
by sorry


end NUMINAMATH_CALUDE_total_discount_calculation_l2395_239552


namespace NUMINAMATH_CALUDE_smallest_integer_power_l2395_239589

theorem smallest_integer_power (x : ℕ) : (∀ y : ℕ, 27^y ≤ 3^24 → y < x) ∧ 27^x > 3^24 ↔ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_power_l2395_239589


namespace NUMINAMATH_CALUDE_remainder_theorem_l2395_239571

theorem remainder_theorem (P D D' Q Q' R R' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + R')
  (h3 : R < D)
  (h4 : R' < D') :
  P % (2 * D * D') = D * R' + R :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2395_239571


namespace NUMINAMATH_CALUDE_opposite_numbers_l2395_239549

theorem opposite_numbers : -(-(3 : ℤ)) = -(-3) :=
by sorry

end NUMINAMATH_CALUDE_opposite_numbers_l2395_239549


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_2005_20_l2395_239599

/-- Given a population size and a desired sample size, calculate the sampling interval for systematic sampling -/
def systematicSamplingInterval (populationSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (populationSize / sampleSize : ℕ)

/-- Theorem: For a population of 2005 numbers and a sample size of 20, the systematic sampling interval is 100 -/
theorem systematic_sampling_interval_2005_20 :
  systematicSamplingInterval 2005 20 = 100 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_2005_20_l2395_239599


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_side_lengths_l2395_239558

/-- A quadrilateral inscribed in a circle with given properties has specific side lengths -/
theorem inscribed_quadrilateral_side_lengths (R : ℝ) (d₁ d₂ : ℝ) :
  R = 25 →
  d₁ = 48 →
  d₂ = 40 →
  ∃ (a b c d : ℝ),
    a = 5 * Real.sqrt 10 ∧
    b = 9 * Real.sqrt 10 ∧
    c = 13 * Real.sqrt 10 ∧
    d = 15 * Real.sqrt 10 ∧
    a^2 + c^2 = d₁^2 ∧
    b^2 + d^2 = d₂^2 ∧
    a * c + b * d = d₁ * d₂ :=
by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_side_lengths_l2395_239558


namespace NUMINAMATH_CALUDE_scientific_notation_of_12000_l2395_239556

theorem scientific_notation_of_12000 :
  (12000 : ℝ) = 1.2 * (10 ^ 4) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_12000_l2395_239556


namespace NUMINAMATH_CALUDE_count_valid_arrangements_l2395_239575

/-- The number of valid 18-letter arrangements of 6 D's, 6 E's, and 6 F's -/
def valid_arrangements : ℕ :=
  Finset.sum (Finset.range 7) (fun m => (Nat.choose 6 m) ^ 3)

/-- Theorem stating the number of valid arrangements -/
theorem count_valid_arrangements :
  valid_arrangements =
    (Finset.sum (Finset.range 7) (fun m => (Nat.choose 6 m) ^ 3)) := by
  sorry

end NUMINAMATH_CALUDE_count_valid_arrangements_l2395_239575
