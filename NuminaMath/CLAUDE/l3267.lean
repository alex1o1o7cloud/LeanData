import Mathlib

namespace NUMINAMATH_CALUDE_function_monotonicity_l3267_326727

open Set
open Function

theorem function_monotonicity (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 0, f x > -x * deriv f x) :
  Monotone (fun x => x * f x) := by
sorry

end NUMINAMATH_CALUDE_function_monotonicity_l3267_326727


namespace NUMINAMATH_CALUDE_sqrt_product_quotient_l3267_326771

theorem sqrt_product_quotient :
  3 * Real.sqrt 5 * (2 * Real.sqrt 15) / Real.sqrt 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_quotient_l3267_326771


namespace NUMINAMATH_CALUDE_min_value_of_f_range_of_a_l3267_326784

-- Define the quadratic function f(x) = ax^2 + x
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x

-- Theorem 1
theorem min_value_of_f (a : ℝ) (h1 : 0 < a) (h2 : a < 1/2) 
  (h3 : ∀ x : ℝ, f a (Real.sin x) ≤ 5/4) :
  ∃ x : ℝ, f a x = -1 ∧ ∀ y : ℝ, f a y ≥ -1 := by sorry

-- Theorem 2
theorem range_of_a (a : ℝ) 
  (h : ∀ x : ℝ, x ∈ Set.Icc (-π/2) 0 → 
    a/2 * Real.sin x * Real.cos x + 1/2 * Real.sin x + 1/2 * Real.cos x + a/4 ≤ 1) :
  a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_range_of_a_l3267_326784


namespace NUMINAMATH_CALUDE_sum_xy_equals_three_l3267_326744

theorem sum_xy_equals_three (x y : ℝ) (h : Real.sqrt (1 - x) + abs (2 - y) = 0) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_xy_equals_three_l3267_326744


namespace NUMINAMATH_CALUDE_problem_statement_l3267_326729

theorem problem_statement (a b c : ℝ) 
  (eq_condition : a - 2*b + c = 0) 
  (ineq_condition : a + 2*b + c < 0) : 
  b < 0 ∧ b^2 - a*c ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3267_326729


namespace NUMINAMATH_CALUDE_smallest_perimeter_consecutive_sides_l3267_326749

theorem smallest_perimeter_consecutive_sides (a b c : ℕ) : 
  a > 2 →
  b = a + 1 →
  c = a + 2 →
  (∀ x y z : ℕ, x > 2 ∧ y = x + 1 ∧ z = x + 2 → a + b + c ≤ x + y + z) →
  a + b + c = 12 :=
sorry

end NUMINAMATH_CALUDE_smallest_perimeter_consecutive_sides_l3267_326749


namespace NUMINAMATH_CALUDE_decimal_expansion_415th_digit_l3267_326751

/-- The decimal expansion of 17/29 -/
def decimal_expansion : ℚ := 17 / 29

/-- The length of the repeating cycle in the decimal expansion of 17/29 -/
def cycle_length : ℕ := 87

/-- The position of the 415th digit within the repeating cycle -/
def position_in_cycle : ℕ := 415 % cycle_length

/-- The 415th digit in the decimal expansion of 17/29 -/
def digit_415 : ℕ := 8

/-- Theorem stating that the 415th digit to the right of the decimal point
    in the decimal expansion of 17/29 is 8 -/
theorem decimal_expansion_415th_digit :
  digit_415 = 8 :=
sorry

end NUMINAMATH_CALUDE_decimal_expansion_415th_digit_l3267_326751


namespace NUMINAMATH_CALUDE_product_digits_sum_base9_l3267_326700

/-- Converts a base 9 number to decimal --/
def base9ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 9 --/
def decimalToBase9 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a base 9 number --/
def sumOfDigitsBase9 (n : ℕ) : ℕ := sorry

/-- The main theorem --/
theorem product_digits_sum_base9 :
  let a := 36
  let b := 21
  let product := (base9ToDecimal a) * (base9ToDecimal b)
  sumOfDigitsBase9 (decimalToBase9 product) = 19 := by sorry

end NUMINAMATH_CALUDE_product_digits_sum_base9_l3267_326700


namespace NUMINAMATH_CALUDE_total_ways_from_A_to_C_l3267_326790

/-- The number of roads from village A to village B -/
def roads_A_to_B : ℕ := 3

/-- The number of roads from village B to village C -/
def roads_B_to_C : ℕ := 2

/-- The total number of different ways to go from village A to village C via village B -/
def total_ways : ℕ := roads_A_to_B * roads_B_to_C

theorem total_ways_from_A_to_C : total_ways = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_from_A_to_C_l3267_326790


namespace NUMINAMATH_CALUDE_betty_daughter_age_difference_l3267_326796

/-- Proves that Betty's daughter is 40% younger than Betty given the specified conditions -/
theorem betty_daughter_age_difference (betty_age : ℕ) (granddaughter_age : ℕ) : 
  betty_age = 60 →
  granddaughter_age = 12 →
  granddaughter_age = (betty_age - (betty_age - granddaughter_age * 3)) / 3 →
  (betty_age - granddaughter_age * 3) / betty_age * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_betty_daughter_age_difference_l3267_326796


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3267_326715

theorem geometric_series_sum (a r : ℝ) (n : ℕ) (h1 : r ≠ 1) (h2 : n > 0) :
  let last_term := a * r^(n - 1)
  let series_sum := a * (r^n - 1) / (r - 1)
  (a = 2 ∧ r = 3 ∧ last_term = 4374) → series_sum = 6560 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3267_326715


namespace NUMINAMATH_CALUDE_ferry_time_difference_l3267_326757

/-- Represents the properties of a ferry --/
structure Ferry where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The problem setup for the two ferries --/
def ferryProblem : Prop :=
  ∃ (P Q : Ferry),
    P.speed = 6 ∧
    P.time = 3 ∧
    P.distance = P.speed * P.time ∧
    Q.distance = 3 * P.distance ∧
    Q.speed = P.speed + 3 ∧
    Q.time = Q.distance / Q.speed ∧
    Q.time - P.time = 3

/-- The main theorem to be proved --/
theorem ferry_time_difference : ferryProblem :=
sorry

end NUMINAMATH_CALUDE_ferry_time_difference_l3267_326757


namespace NUMINAMATH_CALUDE_night_crew_ratio_l3267_326745

theorem night_crew_ratio (day_workers : ℝ) (night_workers : ℝ) (boxes_per_day_worker : ℝ) 
  (h1 : day_workers > 0)
  (h2 : night_workers > 0)
  (h3 : boxes_per_day_worker > 0)
  (h4 : day_workers * boxes_per_day_worker = 0.7 * (day_workers * boxes_per_day_worker + night_workers * (3/4 * boxes_per_day_worker))) :
  night_workers / day_workers = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_night_crew_ratio_l3267_326745


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3267_326760

theorem binomial_coefficient_equality (m : ℕ) : 
  (Nat.choose 15 m = Nat.choose 15 (m - 3)) ↔ m = 9 := by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3267_326760


namespace NUMINAMATH_CALUDE_correct_ways_to_leave_shop_l3267_326731

/-- The number of different flavors of oreos --/
def num_oreo_flavors : ℕ := 6

/-- The number of different flavors of milk --/
def num_milk_flavors : ℕ := 4

/-- The total number of product types (oreos + milk) --/
def total_product_types : ℕ := num_oreo_flavors + num_milk_flavors

/-- The number of products they leave the shop with --/
def num_products : ℕ := 4

/-- Function to calculate the number of ways Alpha and Beta can leave the shop --/
def ways_to_leave_shop : ℕ := sorry

/-- Theorem stating the correct number of ways to leave the shop --/
theorem correct_ways_to_leave_shop : ways_to_leave_shop = 2546 := by sorry

end NUMINAMATH_CALUDE_correct_ways_to_leave_shop_l3267_326731


namespace NUMINAMATH_CALUDE_point_D_coordinates_l3267_326799

def P : ℝ × ℝ := (2, -2)
def Q : ℝ × ℝ := (6, 4)

def is_on_segment (D P Q : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • P + t • Q

def twice_distance (D P Q : ℝ × ℝ) : Prop :=
  ‖D - P‖ = 2 * ‖D - Q‖

theorem point_D_coordinates :
  ∃ D : ℝ × ℝ, is_on_segment D P Q ∧ twice_distance D P Q ∧ D = (3, -0.5) := by
sorry

end NUMINAMATH_CALUDE_point_D_coordinates_l3267_326799


namespace NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l3267_326746

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

theorem binary_11010_equals_octal_32 :
  decimal_to_octal (binary_to_decimal [false, true, false, true, true]) = [3, 2] := by
  sorry

end NUMINAMATH_CALUDE_binary_11010_equals_octal_32_l3267_326746


namespace NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3267_326766

theorem smallest_sum_of_sequence (X Y Z W : ℤ) : 
  X > 0 → Y > 0 → Z > 0 →  -- X, Y, Z are positive integers
  (∃ d : ℤ, Y - X = d ∧ Z - Y = d) →  -- X, Y, Z form an arithmetic sequence
  (∃ r : ℚ, Z = Y * r ∧ W = Z * r) →  -- Y, Z, W form a geometric sequence
  Z = (7 * Y) / 4 →  -- Z/Y = 7/4
  (∀ X' Y' Z' W' : ℤ, 
    X' > 0 → Y' > 0 → Z' > 0 →
    (∃ d : ℤ, Y' - X' = d ∧ Z' - Y' = d) →
    (∃ r : ℚ, Z' = Y' * r ∧ W' = Z' * r) →
    Z' = (7 * Y') / 4 →
    X + Y + Z + W ≤ X' + Y' + Z' + W') →
  X + Y + Z + W = 97 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_sequence_l3267_326766


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l3267_326783

theorem at_least_one_non_negative (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  max (a₁*a₃ + a₂*a₄) (max (a₁*a₅ + a₂*a₆) (max (a₁*a₇ + a₂*a₈) 
    (max (a₃*a₅ + a₄*a₆) (max (a₃*a₇ + a₄*a₈) (a₅*a₇ + a₆*a₈))))) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l3267_326783


namespace NUMINAMATH_CALUDE_water_source_distance_l3267_326756

-- Define the actual distance to the water source
def d : ℝ := sorry

-- Alice's statement is false
axiom alice_false : ¬(d ≥ 8)

-- Bob's statement is false
axiom bob_false : ¬(d ≤ 6)

-- Charlie's statement is false
axiom charlie_false : ¬(d = 7)

-- Theorem to prove
theorem water_source_distance :
  d ∈ Set.union (Set.Ioo 6 7) (Set.Ioo 7 8) :=
sorry

end NUMINAMATH_CALUDE_water_source_distance_l3267_326756


namespace NUMINAMATH_CALUDE_plant_arrangements_eq_1271040_l3267_326788

/-- The number of ways to arrange 5 basil plants and 5 tomato plants with given conditions -/
def plant_arrangements : ℕ :=
  let basil_count : ℕ := 5
  let tomato_count : ℕ := 5
  let tomato_group1_size : ℕ := 2
  let tomato_group2_size : ℕ := 3
  let total_groups : ℕ := basil_count + 2  -- 5 basil plants + 2 tomato groups

  Nat.factorial total_groups *
  (Nat.choose total_groups basil_count * Nat.choose 2 1) *
  Nat.factorial tomato_group1_size *
  Nat.factorial tomato_group2_size

theorem plant_arrangements_eq_1271040 : plant_arrangements = 1271040 := by
  sorry

end NUMINAMATH_CALUDE_plant_arrangements_eq_1271040_l3267_326788


namespace NUMINAMATH_CALUDE_no_preimage_condition_l3267_326720

-- Define the function f: ℝ → ℝ
def f (x : ℝ) : ℝ := -x^2 + 2*x

-- Theorem statement
theorem no_preimage_condition (k : ℝ) :
  (∀ x, f x ≠ k) ↔ k > 1 := by
  sorry

end NUMINAMATH_CALUDE_no_preimage_condition_l3267_326720


namespace NUMINAMATH_CALUDE_polynomial_expansion_theorem_l3267_326791

theorem polynomial_expansion_theorem (a k n : ℤ) : 
  (∀ x : ℚ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) → 
  a - n + k = 7 := by
sorry

end NUMINAMATH_CALUDE_polynomial_expansion_theorem_l3267_326791


namespace NUMINAMATH_CALUDE_koi_added_per_day_proof_l3267_326717

/-- The number of koi fish added per day to the tank -/
def koi_added_per_day : ℕ := 2

/-- The initial total number of fish in the tank -/
def initial_total_fish : ℕ := 280

/-- The number of goldfish added per day -/
def goldfish_added_per_day : ℕ := 5

/-- The number of days in 3 weeks -/
def days_in_three_weeks : ℕ := 21

/-- The final number of goldfish in the tank -/
def final_goldfish : ℕ := 200

/-- The final number of koi fish in the tank -/
def final_koi : ℕ := 227

theorem koi_added_per_day_proof :
  koi_added_per_day = 2 :=
by sorry

end NUMINAMATH_CALUDE_koi_added_per_day_proof_l3267_326717


namespace NUMINAMATH_CALUDE_circle_equation_with_PQ_diameter_l3267_326728

/-- Given circle equation -/
def given_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + x - 6*y + 3 = 0

/-- Given line equation -/
def given_line (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

/-- Intersection points P and Q -/
def intersection_points (P Q : ℝ × ℝ) : Prop :=
  given_circle P.1 P.2 ∧ given_line P.1 P.2 ∧
  given_circle Q.1 Q.2 ∧ given_line Q.1 Q.2 ∧
  P ≠ Q

/-- Circle equation with PQ as diameter -/
def circle_PQ_diameter (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 4*y = 0

/-- Theorem statement -/
theorem circle_equation_with_PQ_diameter
  (P Q : ℝ × ℝ) (h : intersection_points P Q) :
  ∀ x y : ℝ, circle_PQ_diameter x y ↔
    (x - P.1)^2 + (y - P.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2 / 4 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_with_PQ_diameter_l3267_326728


namespace NUMINAMATH_CALUDE_geometric_sequence_angle_l3267_326721

theorem geometric_sequence_angle (a : ℕ → ℝ) (α : ℝ) : 
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- geometric sequence condition
  (a 1 * a 8 = -Real.sqrt 3 * Real.sin α) →  -- root product condition
  (a 1 + a 8 = 2 * Real.sin α) →  -- root sum condition
  ((a 1 + a 8)^2 = 2 * a 3 * a 6 + 6) →  -- given equation
  (0 < α ∧ α < π / 2) →  -- acute angle condition
  α = π / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_angle_l3267_326721


namespace NUMINAMATH_CALUDE_star_sharing_problem_l3267_326735

theorem star_sharing_problem (stars : ℝ) (students_per_star : ℝ) 
  (h1 : stars = 3.0) 
  (h2 : students_per_star = 41.33333333) : 
  ⌊stars * students_per_star⌋ = 124 := by
  sorry

end NUMINAMATH_CALUDE_star_sharing_problem_l3267_326735


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3267_326772

theorem rectangular_prism_diagonal (l w h : ℝ) (hl : l = 3) (hw : w = 4) (hh : h = 5) :
  Real.sqrt (l^2 + w^2 + h^2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l3267_326772


namespace NUMINAMATH_CALUDE_pentagon_diagonals_through_vertex_l3267_326748

/-- The number of diagonals passing through a vertex in a pentagon -/
def diagonals_through_vertex_in_pentagon : ℕ :=
  (5 : ℕ) - 3

theorem pentagon_diagonals_through_vertex :
  diagonals_through_vertex_in_pentagon = 2 :=
by sorry

end NUMINAMATH_CALUDE_pentagon_diagonals_through_vertex_l3267_326748


namespace NUMINAMATH_CALUDE_equation_solutions_l3267_326714

theorem equation_solutions (n : ℕ+) :
  (∃ (s : Finset (ℕ+ × ℕ+ × ℕ+)), 
    s.card = 15 ∧ 
    (∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ 3*x + 3*y + z = n)) →
  n = 19 := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3267_326714


namespace NUMINAMATH_CALUDE_quadratic_inequality_ordering_l3267_326701

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_ordering (a b c : ℝ) :
  (∀ x, f a b c x > 0 ↔ x < -2 ∨ x > 4) →
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_ordering_l3267_326701


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3267_326763

theorem no_integer_solutions : ¬ ∃ (a b : ℤ), 3 * a^2 = b^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3267_326763


namespace NUMINAMATH_CALUDE_olympic_medal_awards_l3267_326716

/-- The number of ways to award medals in the Olympic 100-meter finals -/
def medal_awards (total_sprinters : ℕ) (american_sprinters : ℕ) (medals : ℕ) : ℕ :=
  let non_american_sprinters := total_sprinters - american_sprinters
  let no_american_medals := non_american_sprinters.descFactorial medals
  let one_american_medal := american_sprinters * medals * (non_american_sprinters.descFactorial (medals - 1))
  no_american_medals + one_american_medal

/-- Theorem stating the number of ways to award medals in the given scenario -/
theorem olympic_medal_awards :
  medal_awards 10 4 3 = 480 :=
by sorry

end NUMINAMATH_CALUDE_olympic_medal_awards_l3267_326716


namespace NUMINAMATH_CALUDE_greatest_integer_radius_l3267_326733

theorem greatest_integer_radius (r : ℕ) : (∀ n : ℕ, n > r → (n : ℝ)^2 * Real.pi ≥ 75 * Real.pi) ∧ r^2 * Real.pi < 75 * Real.pi → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_radius_l3267_326733


namespace NUMINAMATH_CALUDE_no_solution_exists_l3267_326741

theorem no_solution_exists : ¬∃ x : ℝ, 2 * ((x - 3) / 2 + 3) = x + 6 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l3267_326741


namespace NUMINAMATH_CALUDE_regular_tetrahedron_inradius_l3267_326781

/-- A regular tetrahedron -/
structure RegularTetrahedron where
  /-- The altitude of the regular tetrahedron -/
  altitude : ℝ
  /-- The inradius of the regular tetrahedron -/
  inradius : ℝ

/-- The inradius of a regular tetrahedron is one fourth of its altitude -/
theorem regular_tetrahedron_inradius (t : RegularTetrahedron) :
  t.inradius = (1 / 4) * t.altitude := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_inradius_l3267_326781


namespace NUMINAMATH_CALUDE_pyramid_x_value_l3267_326722

/-- Pyramid represents a numerical pyramid where each number below the top row
    is the product of the number to the right and the number to the left in the row immediately above it. -/
structure Pyramid where
  top_left : ℕ
  middle : ℕ
  bottom_left : ℕ
  x : ℕ

/-- Given a Pyramid, this theorem proves that x must be 4 -/
theorem pyramid_x_value (p : Pyramid) (h1 : p.top_left = 35) (h2 : p.middle = 700) (h3 : p.bottom_left = 5)
  (h4 : p.middle = p.top_left * (p.middle / p.top_left))
  (h5 : p.middle / p.top_left = p.bottom_left * p.x) :
  p.x = 4 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_x_value_l3267_326722


namespace NUMINAMATH_CALUDE_tempo_original_value_l3267_326792

/-- Given a tempo insured to 5/7 of its original value, with a 3% premium rate
resulting in a $300 premium, prove that the original value of the tempo is $14,000. -/
theorem tempo_original_value (insurance_ratio : ℚ) (premium_rate : ℚ) (premium_amount : ℚ) :
  insurance_ratio = 5 / 7 →
  premium_rate = 3 / 100 →
  premium_amount = 300 →
  premium_rate * (insurance_ratio * 14000) = premium_amount :=
by sorry

end NUMINAMATH_CALUDE_tempo_original_value_l3267_326792


namespace NUMINAMATH_CALUDE_seashells_given_to_sam_l3267_326795

theorem seashells_given_to_sam (initial_seashells : ℕ) (remaining_seashells : ℕ) 
  (h1 : initial_seashells = 70) 
  (h2 : remaining_seashells = 27) : 
  initial_seashells - remaining_seashells = 43 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_to_sam_l3267_326795


namespace NUMINAMATH_CALUDE_find_c_l3267_326779

/-- Given two functions p and q, prove that c = 7 -/
theorem find_c (p q : ℝ → ℝ) (c : ℝ) : 
  (∀ x, p x = 3 * x - 9) → 
  (∀ x, q x = 4 * x - c) → 
  p (q 3) = 6 → 
  c = 7 := by
sorry

end NUMINAMATH_CALUDE_find_c_l3267_326779


namespace NUMINAMATH_CALUDE_M_mod_1000_l3267_326778

/-- The number of 8-digit positive integers with strictly increasing digits -/
def M : ℕ := Nat.choose 9 8

/-- Theorem stating that M modulo 1000 equals 9 -/
theorem M_mod_1000 : M % 1000 = 9 := by
  sorry

end NUMINAMATH_CALUDE_M_mod_1000_l3267_326778


namespace NUMINAMATH_CALUDE_problem_solution_l3267_326789

theorem problem_solution :
  (∀ n : ℕ, 2 * 8^n * 32^n = 2^17 → n = 2) ∧
  (∀ n : ℕ, ∀ x : ℝ, n > 0 → x^(2*n) = 2 → (2*x^(3*n))^2 - 3*(x^2)^(2*n) = 20) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3267_326789


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l3267_326768

-- Define the two-digit number
def two_digit_number : ℕ → Prop :=
  λ n => 10 ≤ n ∧ n ≤ 99

-- Define the property of the number
def satisfies_equation (x : ℕ) : Prop :=
  500 + x = 9 * x - 12

-- Theorem statement
theorem two_digit_number_problem :
  ∃ x : ℕ, two_digit_number x ∧ satisfies_equation x ∧ x = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l3267_326768


namespace NUMINAMATH_CALUDE_linda_egg_ratio_l3267_326758

theorem linda_egg_ratio : 
  ∀ (total_eggs : ℕ) (brown_eggs : ℕ) (white_eggs : ℕ),
  total_eggs = 12 →
  brown_eggs = 5 →
  total_eggs = brown_eggs + white_eggs →
  (white_eggs : ℚ) / (brown_eggs : ℚ) = 7 / 5 := by
sorry

end NUMINAMATH_CALUDE_linda_egg_ratio_l3267_326758


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l3267_326770

theorem imaginary_part_of_z_is_zero (z : ℂ) (h : z * (Complex.I + 1) = 2 / (Complex.I - 1)) :
  z.im = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_is_zero_l3267_326770


namespace NUMINAMATH_CALUDE_equation_solution_l3267_326703

theorem equation_solution (y : ℚ) : 
  (40 : ℚ) / 60 = Real.sqrt (y / 60) → y = 110 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3267_326703


namespace NUMINAMATH_CALUDE_unique_prime_with_next_square_is_three_l3267_326753

theorem unique_prime_with_next_square_is_three :
  ∀ p : ℕ, Prime p → (∃ n : ℕ, p + 1 = n^2) → p = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_next_square_is_three_l3267_326753


namespace NUMINAMATH_CALUDE_exponential_inequality_and_unique_a_l3267_326750

open Real

theorem exponential_inequality_and_unique_a :
  (∀ x > -1, exp x > (x + 1)^2 / 2) ∧
  (∃! a : ℝ, a > 0 ∧ ∀ x > 0, exp (1 - x) + 2 * log x ≤ a * (x - 1) + 1) ∧
  (∃ a : ℝ, a > 0 ∧ ∀ x > 0, exp (1 - x) + 2 * log x ≤ a * (x - 1) + 1 ∧ a = 1) :=
by sorry

end NUMINAMATH_CALUDE_exponential_inequality_and_unique_a_l3267_326750


namespace NUMINAMATH_CALUDE_unique_solutions_l3267_326755

def is_solution (m n p : ℕ) : Prop :=
  p.Prime ∧ m > 0 ∧ n > 0 ∧ p^n + 3600 = m^2

theorem unique_solutions :
  ∀ m n p : ℕ,
    is_solution m n p ↔
      (m = 61 ∧ n = 2 ∧ p = 11) ∨
      (m = 65 ∧ n = 4 ∧ p = 5) ∨
      (m = 68 ∧ n = 10 ∧ p = 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l3267_326755


namespace NUMINAMATH_CALUDE_probability_at_least_one_defective_l3267_326711

/-- The probability of selecting at least one defective item from a set of products -/
theorem probability_at_least_one_defective 
  (total : ℕ) 
  (defective : ℕ) 
  (selected : ℕ) 
  (h1 : total = 10) 
  (h2 : defective = 3) 
  (h3 : selected = 3) :
  (1 : ℚ) - (Nat.choose (total - defective) selected : ℚ) / (Nat.choose total selected : ℚ) = 17/24 := by
  sorry

#check probability_at_least_one_defective

end NUMINAMATH_CALUDE_probability_at_least_one_defective_l3267_326711


namespace NUMINAMATH_CALUDE_circle_point_distance_sum_l3267_326782

/-- Given a circle with diameter AB and radius R, and a tangent AT at point A,
    prove that a point M on the circle satisfying the condition that the sum of
    its distances to lines AB and AT is l exists if and only if l ≤ R(√2 + 1). -/
theorem circle_point_distance_sum (R l : ℝ) : 
  ∃ (M : ℝ × ℝ), 
    (M.1^2 + M.2^2 = R^2) ∧ 
    (M.1 + M.2 = l) ↔ 
    l ≤ R * (Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_point_distance_sum_l3267_326782


namespace NUMINAMATH_CALUDE_count_non_adjacent_arrangements_l3267_326739

/-- The number of arrangements of 5 letters where two specific letters are not adjacent to a third specific letter -/
def non_adjacent_arrangements : ℕ :=
  let total_letters := 5
  let non_adjacent_three := 12  -- arrangements where a, b, c are not adjacent
  let adjacent_pair_not_third := 24  -- arrangements where a and b are adjacent, but not to c
  non_adjacent_three + adjacent_pair_not_third

/-- Theorem stating that the number of arrangements of a, b, c, d, e where both a and b are not adjacent to c is 36 -/
theorem count_non_adjacent_arrangements :
  non_adjacent_arrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_count_non_adjacent_arrangements_l3267_326739


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l3267_326785

-- Define a type for lines in 3D space
structure Line3D where
  -- We don't need to specify the exact representation of a line here
  -- as we're only interested in their relationships

-- Define the relationships between lines
def perpendicular (l1 l2 : Line3D) : Prop := sorry
def parallel (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem perpendicular_parallel_implies_perpendicular 
  (l1 l2 l3 : Line3D) 
  (h1 : perpendicular l1 l2) 
  (h2 : parallel l2 l3) : 
  perpendicular l1 l3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_l3267_326785


namespace NUMINAMATH_CALUDE_admission_cost_proof_l3267_326734

/-- Calculates the total cost of admission tickets for a group -/
def total_cost (adult_price child_price : ℕ) (num_children : ℕ) : ℕ :=
  let num_adults := num_children + 25
  let adult_cost := num_adults * adult_price
  let child_cost := num_children * child_price
  adult_cost + child_cost

/-- Proves that the total cost for the given group is $720 -/
theorem admission_cost_proof :
  total_cost 15 8 15 = 720 := by
  sorry

end NUMINAMATH_CALUDE_admission_cost_proof_l3267_326734


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3267_326740

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a ≥ 5) →
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) ∧
  ¬(∀ a : ℝ, (∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0) → a ≥ 5) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l3267_326740


namespace NUMINAMATH_CALUDE_sum_of_areas_decomposition_l3267_326787

/-- Represents a 1 by 1 by 1 cube -/
structure UnitCube where
  side : ℝ
  is_unit : side = 1

/-- Represents a triangle with vertices on the cube -/
structure CubeTriangle where
  vertices : Fin 3 → Fin 8

/-- The area of a triangle on the cube -/
noncomputable def triangle_area (t : CubeTriangle) : ℝ := sorry

/-- The sum of areas of all triangles on the cube -/
noncomputable def sum_of_triangle_areas (cube : UnitCube) : ℝ := sorry

/-- The theorem to be proved -/
theorem sum_of_areas_decomposition (cube : UnitCube) :
  ∃ (m n p : ℕ), sum_of_triangle_areas cube = m + Real.sqrt n + Real.sqrt p ∧ m + n + p = 348 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_areas_decomposition_l3267_326787


namespace NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_parallel_lines_implies_perpendicular_planes_l3267_326761

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the parallel relation between two lines
variable (parallel_lines : Line → Line → Prop)

-- Define the perpendicular relation between two planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the relation of a line being contained in a plane
variable (contained_in : Line → Plane → Prop)

-- Theorem 1
theorem perpendicular_parallel_implies_perpendicular
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : parallel_planes α β) :
  perpendicular_lines l m :=
sorry

-- Theorem 2
theorem parallel_lines_implies_perpendicular_planes
  (l m : Line) (α β : Plane)
  (h1 : perpendicular l α)
  (h2 : contained_in m β)
  (h3 : parallel_lines l m) :
  perpendicular_planes α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_implies_perpendicular_parallel_lines_implies_perpendicular_planes_l3267_326761


namespace NUMINAMATH_CALUDE_tammys_climbing_speed_l3267_326752

/-- Tammy's mountain climbing problem -/
theorem tammys_climbing_speed 
  (total_time : ℝ) 
  (total_distance : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (h1 : total_time = 14) 
  (h2 : total_distance = 52) 
  (h3 : speed_difference = 0.5) 
  (h4 : time_difference = 2) :
  ∃ (speed_day1 speed_day2 time_day1 time_day2 : ℝ),
    speed_day2 = speed_day1 + speed_difference ∧
    time_day2 = time_day1 - time_difference ∧
    time_day1 + time_day2 = total_time ∧
    speed_day1 * time_day1 + speed_day2 * time_day2 = total_distance ∧
    speed_day2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_tammys_climbing_speed_l3267_326752


namespace NUMINAMATH_CALUDE_fundraiser_proof_l3267_326742

def fundraiser (total_promised : ℕ) (amount_received : ℕ) (amy_owes : ℕ) : Prop :=
  let total_owed : ℕ := total_promised - amount_received
  let derek_owes : ℕ := amy_owes / 2
  let sally_carl_owe : ℕ := total_owed - (amy_owes + derek_owes)
  sally_carl_owe / 2 = 35

theorem fundraiser_proof : fundraiser 400 285 30 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_proof_l3267_326742


namespace NUMINAMATH_CALUDE_factorization_validity_l3267_326769

theorem factorization_validity (x : ℝ) : x^2 - 4*x + 4 = (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_validity_l3267_326769


namespace NUMINAMATH_CALUDE_max_salary_in_semipro_league_l3267_326794

/-- Represents a baseball team -/
structure Team where
  players : Nat
  minSalary : Nat
  maxTotalSalary : Nat

/-- Calculates the maximum possible salary for a single player in a team -/
def maxSinglePlayerSalary (team : Team) : Nat :=
  team.maxTotalSalary - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player in the given conditions -/
theorem max_salary_in_semipro_league :
  let team : Team := {
    players := 21,
    minSalary := 15000,
    maxTotalSalary := 700000
  }
  maxSinglePlayerSalary team = 400000 := by sorry

end NUMINAMATH_CALUDE_max_salary_in_semipro_league_l3267_326794


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_3_199_4_l3267_326712

/-- Calculates the number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (first last commonDiff : ℕ) : ℕ :=
  (last - first) / commonDiff + 1

/-- Theorem: The arithmetic sequence starting with 3, ending with 199,
    and having a common difference of 4 contains exactly 50 terms -/
theorem arithmetic_sequence_length_3_199_4 :
  arithmeticSequenceLength 3 199 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_3_199_4_l3267_326712


namespace NUMINAMATH_CALUDE_max_consecutive_set_size_l3267_326793

/-- Sum of digits of a positive integer -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Property: sum of digits is not a multiple of 11 -/
def validNumber (n : ℕ) : Prop :=
  sumOfDigits n % 11 ≠ 0

/-- A set of consecutive positive integers with the given property -/
structure ConsecutiveSet :=
  (start : ℕ)
  (size : ℕ)
  (property : ∀ k, k ∈ Finset.range size → validNumber (start + k))

/-- The theorem to be proved -/
theorem max_consecutive_set_size :
  (∃ S : ConsecutiveSet, S.size = 38) ∧
  (∀ S : ConsecutiveSet, S.size ≤ 38) :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_set_size_l3267_326793


namespace NUMINAMATH_CALUDE_base9_calculation_l3267_326719

/-- Converts a base 10 number to its base 9 representation -/
def toBase9 (n : ℕ) : ℕ := sorry

/-- Converts a base 9 number to its base 10 representation -/
def fromBase9 (n : ℕ) : ℕ := sorry

/-- Addition in base 9 -/
def addBase9 (a b : ℕ) : ℕ := toBase9 (fromBase9 a + fromBase9 b)

/-- Subtraction in base 9 -/
def subBase9 (a b : ℕ) : ℕ := toBase9 (fromBase9 a - fromBase9 b)

theorem base9_calculation :
  subBase9 (addBase9 (addBase9 2365 1484) 782) 671 = 4170 := by sorry

end NUMINAMATH_CALUDE_base9_calculation_l3267_326719


namespace NUMINAMATH_CALUDE_unique_solution_xy_equation_l3267_326765

theorem unique_solution_xy_equation :
  ∃! (x y : ℕ), x < y ∧ x^y = y^x :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_xy_equation_l3267_326765


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3267_326718

theorem arithmetic_expression_equality : 2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = -14 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3267_326718


namespace NUMINAMATH_CALUDE_parabola_directrix_l3267_326736

theorem parabola_directrix (x y : ℝ) :
  y = 4 * x^2 → (∃ (k : ℝ), y = -1/(4*k) ∧ k = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3267_326736


namespace NUMINAMATH_CALUDE_principal_is_7000_l3267_326702

/-- Calculates the simple interest for a given principal, rate, and time -/
def simpleInterest (principal : ℚ) (rate : ℚ) (time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Represents the financial transaction described in the problem -/
structure Transaction where
  principal : ℚ
  borrowRate : ℚ
  lendRate : ℚ
  time : ℚ
  gainPerYear : ℚ

/-- Theorem stating that given the conditions, the principal is 7000 -/
theorem principal_is_7000 (t : Transaction) 
  (h1 : t.time = 2)
  (h2 : t.borrowRate = 4)
  (h3 : t.lendRate = 6)
  (h4 : t.gainPerYear = 140)
  (h5 : t.gainPerYear = (simpleInterest t.principal t.lendRate t.time - 
                         simpleInterest t.principal t.borrowRate t.time) / t.time) :
  t.principal = 7000 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_7000_l3267_326702


namespace NUMINAMATH_CALUDE_given_equation_is_quadratic_l3267_326730

/-- An equation is quadratic in one variable if it can be expressed in the form ax² + bx + c = 0, where a ≠ 0 and x is the variable. --/
def is_quadratic_one_var (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation (x - 1)(x + 2) = 1 --/
def given_equation (x : ℝ) : ℝ :=
  (x - 1) * (x + 2) - 1

theorem given_equation_is_quadratic :
  is_quadratic_one_var given_equation :=
sorry

end NUMINAMATH_CALUDE_given_equation_is_quadratic_l3267_326730


namespace NUMINAMATH_CALUDE_triangle_exterior_angle_theorem_l3267_326732

/-- 
Given a triangle where one side is extended:
- ext_angle is the exterior angle
- int_angle1 is one of the non-adjacent interior angles
- int_angle2 is the other non-adjacent interior angle
-/
theorem triangle_exterior_angle_theorem 
  (ext_angle int_angle1 int_angle2 : ℝ) : 
  ext_angle = 154 ∧ int_angle1 = 58 → int_angle2 = 96 := by
sorry

end NUMINAMATH_CALUDE_triangle_exterior_angle_theorem_l3267_326732


namespace NUMINAMATH_CALUDE_complex_number_sum_l3267_326724

theorem complex_number_sum (z : ℂ) : z = (1 + I) / (1 - I) → ∃ a b : ℝ, z = a + b * I ∧ a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_l3267_326724


namespace NUMINAMATH_CALUDE_anika_pencils_excess_l3267_326767

theorem anika_pencils_excess (reeta_pencils : ℕ) (total_pencils : ℕ) (anika_pencils : ℕ) : 
  reeta_pencils = 20 →
  anika_pencils + reeta_pencils = total_pencils →
  total_pencils = 64 →
  ∃ m : ℕ, anika_pencils = 2 * reeta_pencils + m →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_anika_pencils_excess_l3267_326767


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l3267_326705

/-- Proves that a train 150 metres long running at 54 km/hr takes 10 seconds to pass a pole. -/
theorem train_passing_pole_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (h1 : train_length = 150) 
  (h2 : train_speed_kmh = 54) : 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 10 := by
sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l3267_326705


namespace NUMINAMATH_CALUDE_trapezoid_area_is_15_l3267_326773

/-- A trapezoid bounded by y = 2x, y = 8, y = 2, and the y-axis -/
structure Trapezoid where
  /-- The line y = 2x -/
  line_1 : ℝ → ℝ := λ x => 2 * x
  /-- The line y = 8 -/
  line_2 : ℝ → ℝ := λ _ => 8
  /-- The line y = 2 -/
  line_3 : ℝ → ℝ := λ _ => 2
  /-- The y-axis (x = 0) -/
  y_axis : ℝ → ℝ := λ y => 0

/-- The area of the trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ :=
  15

/-- Theorem stating that the area of the given trapezoid is 15 square units -/
theorem trapezoid_area_is_15 (t : Trapezoid) : trapezoidArea t = 15 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_15_l3267_326773


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l3267_326710

theorem junk_mail_distribution (blocks : ℕ) (pieces_per_block : ℕ) (h1 : blocks = 4) (h2 : pieces_per_block = 48) :
  blocks * pieces_per_block = 192 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l3267_326710


namespace NUMINAMATH_CALUDE_vectors_coplanar_iff_x_eq_five_l3267_326764

/-- Given vectors a, b, and c in ℝ³, prove that they are coplanar if and only if x = 5 -/
theorem vectors_coplanar_iff_x_eq_five (a b c : ℝ × ℝ × ℝ) :
  a = (1, -1, 3) →
  b = (-1, 4, -2) →
  c = (1, 5, x) →
  (∃ (m n : ℝ), c = m • a + n • b) ↔ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_vectors_coplanar_iff_x_eq_five_l3267_326764


namespace NUMINAMATH_CALUDE_machine_production_in_10_seconds_l3267_326759

/-- A machine that produces items at a constant rate -/
structure Machine where
  items_per_minute : ℕ

/-- Calculate the number of items produced in a given number of seconds -/
def items_produced (m : Machine) (seconds : ℕ) : ℚ :=
  (m.items_per_minute : ℚ) * (seconds : ℚ) / 60

theorem machine_production_in_10_seconds (m : Machine) 
  (h : m.items_per_minute = 150) : 
  items_produced m 10 = 25 := by
  sorry

#eval items_produced ⟨150⟩ 10

end NUMINAMATH_CALUDE_machine_production_in_10_seconds_l3267_326759


namespace NUMINAMATH_CALUDE_nigels_money_ratio_l3267_326707

/-- Represents Nigel's money transactions and proves the final ratio --/
theorem nigels_money_ratio :
  ∀ (original : ℝ) (given_away : ℝ),
  original > 0 →
  given_away > 0 →
  original + 45 - given_away + 80 - 25 = 2 * original + 25 →
  (original + 45 + 80 - 25) / original = 3 :=
by sorry

end NUMINAMATH_CALUDE_nigels_money_ratio_l3267_326707


namespace NUMINAMATH_CALUDE_small_tile_position_l3267_326704

/-- Represents a tile on the grid -/
inductive Tile
| Small : Tile  -- 1x1 tile
| Large : Tile  -- 1x3 tile

/-- Represents a position on the 7x7 grid -/
structure Position where
  row : Fin 7
  col : Fin 7

/-- Represents the state of the grid -/
structure GridState where
  smallTilePos : Position
  largeTiles : Finset (Position × Position × Position)

/-- Checks if a position is at the center or adjacent to the border -/
def isCenterOrBorder (pos : Position) : Prop :=
  pos.row = 0 ∨ pos.row = 3 ∨ pos.row = 6 ∨
  pos.col = 0 ∨ pos.col = 3 ∨ pos.col = 6

/-- Main theorem -/
theorem small_tile_position (grid : GridState) 
  (h1 : grid.largeTiles.card = 16) : 
  isCenterOrBorder grid.smallTilePos :=
sorry

end NUMINAMATH_CALUDE_small_tile_position_l3267_326704


namespace NUMINAMATH_CALUDE_infinitely_many_primes_of_form_l3267_326762

theorem infinitely_many_primes_of_form (p : Nat) (hp : Nat.Prime p) (hp_odd : Odd p) :
  ∃ (S : Set Nat), (∀ n ∈ S, Nat.Prime n ∧ ∃ x, n = 2 * p * x + 1) ∧ Set.Infinite S :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_of_form_l3267_326762


namespace NUMINAMATH_CALUDE_multiples_of_2_are_even_is_universal_l3267_326775

/-- A predicate representing a property of natural numbers -/
def P (n : ℕ) : Prop := Even n

/-- Definition of a universal proposition -/
def UniversalProposition (P : α → Prop) : Prop :=
  ∀ x, P x

/-- The statement "All multiples of 2 are even" -/
def AllMultiplesOf2AreEven : Prop :=
  ∀ n : ℕ, 2 ∣ n → Even n

/-- Theorem stating that "All multiples of 2 are even" is a universal proposition -/
theorem multiples_of_2_are_even_is_universal :
  UniversalProposition (λ n => 2 ∣ n → Even n) :=
sorry

end NUMINAMATH_CALUDE_multiples_of_2_are_even_is_universal_l3267_326775


namespace NUMINAMATH_CALUDE_sales_tax_percentage_l3267_326743

theorem sales_tax_percentage 
  (total_bill : ℝ)
  (food_price : ℝ)
  (tip_percentage : ℝ)
  (h1 : total_bill = 158.40)
  (h2 : food_price = 120)
  (h3 : tip_percentage = 0.20)
  : ∃ (tax_percentage : ℝ), 
    tax_percentage = 0.10 ∧ 
    total_bill = (food_price * (1 + tax_percentage) * (1 + tip_percentage)) :=
by sorry

end NUMINAMATH_CALUDE_sales_tax_percentage_l3267_326743


namespace NUMINAMATH_CALUDE_equation_solution_l3267_326797

theorem equation_solution : ∃ x : ℝ, 4 * x - 7 = 5 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3267_326797


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3267_326798

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 5*x + 6 > 0 ∧ x ≠ 3) ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3267_326798


namespace NUMINAMATH_CALUDE_skirt_ratio_is_two_thirds_l3267_326708

-- Define the number of skirts in each valley
def purple_skirts : ℕ := 10
def azure_skirts : ℕ := 60

-- Define the relationship between Purple and Seafoam Valley skirts
def seafoam_skirts : ℕ := 4 * purple_skirts

-- Define the ratio of Seafoam to Azure Valley skirts
def skirt_ratio : Rat := seafoam_skirts / azure_skirts

-- Theorem to prove
theorem skirt_ratio_is_two_thirds : skirt_ratio = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_skirt_ratio_is_two_thirds_l3267_326708


namespace NUMINAMATH_CALUDE_stream_speed_l3267_326726

/-- The speed of the stream given rowing distances and times -/
theorem stream_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 78) 
  (h2 : upstream_distance = 50) (h3 : downstream_time = 2) (h4 : upstream_time = 2) : 
  ∃ (boat_speed stream_speed : ℝ), 
    downstream_distance = (boat_speed + stream_speed) * downstream_time ∧ 
    upstream_distance = (boat_speed - stream_speed) * upstream_time ∧ 
    stream_speed = 7 :=
by sorry

end NUMINAMATH_CALUDE_stream_speed_l3267_326726


namespace NUMINAMATH_CALUDE_simultaneous_equations_solution_l3267_326777

theorem simultaneous_equations_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 2 ∧ y = (3 * m - 2) * x + 5 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_solution_l3267_326777


namespace NUMINAMATH_CALUDE_unique_k_value_l3267_326754

theorem unique_k_value : ∃! k : ℝ, ∀ x : ℝ, 
  (x * (2 * x + 3) < k ↔ -5/2 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_unique_k_value_l3267_326754


namespace NUMINAMATH_CALUDE_power_of_power_l3267_326780

theorem power_of_power (x : ℝ) : (x^2)^3 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3267_326780


namespace NUMINAMATH_CALUDE_rectangle_area_l3267_326774

theorem rectangle_area (length width area : ℝ) : 
  length = 24 →
  width = 0.875 * length →
  area = length * width →
  area = 504 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3267_326774


namespace NUMINAMATH_CALUDE_eggs_per_friend_l3267_326786

/-- Proves that sharing 16 eggs equally among 8 friends results in 2 eggs per friend -/
theorem eggs_per_friend (total_eggs : ℕ) (num_friends : ℕ) (eggs_per_friend : ℕ) 
  (h1 : total_eggs = 16) 
  (h2 : num_friends = 8) 
  (h3 : eggs_per_friend * num_friends = total_eggs) : 
  eggs_per_friend = 2 := by
  sorry

end NUMINAMATH_CALUDE_eggs_per_friend_l3267_326786


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_unique_pair_l3267_326713

theorem sqrt_equality_implies_unique_pair :
  ∀ a b : ℕ,
  0 < a → 0 < b → a < b →
  (Real.sqrt (4 + Real.sqrt (36 + 24 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b) →
  a = 1 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_unique_pair_l3267_326713


namespace NUMINAMATH_CALUDE_triangle_cosine_A_l3267_326706

theorem triangle_cosine_A (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 - b^2 = Real.sqrt 3 * b * c →
  Real.sin C = 2 * Real.sqrt 3 * Real.sin B →
  Real.cos A = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_A_l3267_326706


namespace NUMINAMATH_CALUDE_matchstick_20th_term_l3267_326738

/-- Arithmetic sequence with first term 4 and common difference 3 -/
def matchstick_sequence (n : ℕ) : ℕ := 4 + 3 * (n - 1)

/-- The 20th term of the matchstick sequence is 61 -/
theorem matchstick_20th_term : matchstick_sequence 20 = 61 := by
  sorry

end NUMINAMATH_CALUDE_matchstick_20th_term_l3267_326738


namespace NUMINAMATH_CALUDE_watch_loss_percentage_loss_percentage_is_ten_percent_l3267_326747

/-- Proves that the loss percentage is 10% for a watch sale scenario --/
theorem watch_loss_percentage : ℝ → Prop :=
  λ L : ℝ =>
    let cost_price : ℝ := 2000
    let selling_price : ℝ := cost_price - (L / 100 * cost_price)
    let new_selling_price : ℝ := cost_price + (4 / 100 * cost_price)
    new_selling_price = selling_price + 280 →
    L = 10

/-- The loss percentage is indeed 10% --/
theorem loss_percentage_is_ten_percent : watch_loss_percentage 10 := by
  sorry

end NUMINAMATH_CALUDE_watch_loss_percentage_loss_percentage_is_ten_percent_l3267_326747


namespace NUMINAMATH_CALUDE_equal_first_two_numbers_l3267_326723

theorem equal_first_two_numbers (a : Fin 17 → ℕ) 
  (h : ∀ i : Fin 17, (a i) ^ (a (i + 1)) = (a ((i + 1) % 17)) ^ (a ((i + 2) % 17))) : 
  a 0 = a 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_first_two_numbers_l3267_326723


namespace NUMINAMATH_CALUDE_distribute_five_items_four_bags_l3267_326776

/-- The number of ways to distribute n different items into k identical bags, allowing empty bags. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- Theorem: There are 36 ways to distribute 5 different items into 4 identical bags, allowing empty bags. -/
theorem distribute_five_items_four_bags : distribute 5 4 = 36 := by sorry

end NUMINAMATH_CALUDE_distribute_five_items_four_bags_l3267_326776


namespace NUMINAMATH_CALUDE_michael_monica_ratio_l3267_326737

/-- The ages of three people satisfy certain conditions -/
structure AgesProblem where
  /-- Patrick's age -/
  p : ℕ
  /-- Michael's age -/
  m : ℕ
  /-- Monica's age -/
  mo : ℕ
  /-- The ages of Patrick and Michael are in the ratio of 3:5 -/
  patrick_michael_ratio : 3 * m = 5 * p
  /-- The sum of their ages is 88 -/
  sum_of_ages : p + m + mo = 88
  /-- The difference between Monica and Patrick's ages is 22 -/
  monica_patrick_diff : mo - p = 22

/-- The ratio of Michael's age to Monica's age is 3:4 -/
theorem michael_monica_ratio (prob : AgesProblem) : 3 * prob.mo = 4 * prob.m := by
  sorry

end NUMINAMATH_CALUDE_michael_monica_ratio_l3267_326737


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l3267_326709

theorem largest_multiple_of_15_under_400 : ∃ n : ℕ, n * 15 = 390 ∧ 
  390 < 400 ∧ 
  (∀ m : ℕ, m * 15 < 400 → m * 15 ≤ 390) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_400_l3267_326709


namespace NUMINAMATH_CALUDE_determinant_equation_one_root_l3267_326725

/-- Given nonzero real numbers a, b, c, and k with k > 0, the determinant equation has exactly one real root at x = 0 -/
theorem determinant_equation_one_root (a b c k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k > 0) :
  ∃! x : ℝ, x * (x^2 + k^2 * a^2 + k^2 * b^2 + k^2 * c^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_determinant_equation_one_root_l3267_326725
