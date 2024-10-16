import Mathlib

namespace NUMINAMATH_CALUDE_minimum_buses_required_l2374_237477

theorem minimum_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 535) (h2 : bus_capacity = 45) :
  ∃ (num_buses : ℕ), num_buses * bus_capacity ≥ total_students ∧ 
  ∀ (m : ℕ), m * bus_capacity ≥ total_students → m ≥ num_buses ∧
  num_buses = 12 :=
sorry

end NUMINAMATH_CALUDE_minimum_buses_required_l2374_237477


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2374_237450

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 3 * x^2 - 6 * x
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2374_237450


namespace NUMINAMATH_CALUDE_plot_width_l2374_237456

/-- 
Given a rectangular plot with length 90 meters, if 60 poles placed 5 meters apart 
are needed to enclose the plot, then the width of the plot is 60 meters.
-/
theorem plot_width (poles : ℕ) (pole_distance : ℝ) (length width : ℝ) : 
  poles = 60 → 
  pole_distance = 5 → 
  length = 90 → 
  poles * pole_distance = 2 * (length + width) → 
  width = 60 := by sorry

end NUMINAMATH_CALUDE_plot_width_l2374_237456


namespace NUMINAMATH_CALUDE_rational_coefficient_terms_count_rational_coefficient_terms_count_is_126_l2374_237460

theorem rational_coefficient_terms_count : ℕ :=
  let expansion := (λ x y : ℝ => (x * (2 ^ (1/4 : ℝ)) + y * (5 ^ (1/2 : ℝ))) ^ 500)
  let total_terms := 501
  let is_rational_coeff := λ k : ℕ => (k % 4 = 0) ∧ ((500 - k) % 2 = 0)
  (Finset.range total_terms).filter is_rational_coeff |>.card

/-- The number of terms with rational coefficients in the expansion of (x∗∜2+y∗√5)^500 is 126 -/
theorem rational_coefficient_terms_count_is_126 : 
  rational_coefficient_terms_count = 126 := by sorry

end NUMINAMATH_CALUDE_rational_coefficient_terms_count_rational_coefficient_terms_count_is_126_l2374_237460


namespace NUMINAMATH_CALUDE_angle_b_range_in_geometric_progression_triangle_l2374_237432

/-- In a triangle ABC, if sides a, b, c form a geometric progression,
    then angle B is in the range (0, π/3] -/
theorem angle_b_range_in_geometric_progression_triangle
  (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a + b > c → b + c > a → c + a > b →
  b^2 = a * c →
  0 < B ∧ B ≤ π/3 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_range_in_geometric_progression_triangle_l2374_237432


namespace NUMINAMATH_CALUDE_eight_lines_form_784_parallelograms_intersecting_parallel_lines_theorem_l2374_237480

/-- The number of parallelograms formed by two sets of intersecting parallel lines -/
def parallelogramsCount (n m : ℕ) : ℕ := (n.choose 2) * (m.choose 2)

/-- Theorem stating that 8 lines in each set form 784 parallelograms -/
theorem eight_lines_form_784_parallelograms (n : ℕ) :
  parallelogramsCount n 8 = 784 → n = 8 := by
  sorry

/-- Main theorem proving that given 8 lines in one set and 784 parallelograms, 
    the other set must have 8 lines -/
theorem intersecting_parallel_lines_theorem :
  ∃ (n : ℕ), parallelogramsCount n 8 = 784 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_lines_form_784_parallelograms_intersecting_parallel_lines_theorem_l2374_237480


namespace NUMINAMATH_CALUDE_left_seats_count_l2374_237481

/-- Represents the seating configuration of a bus -/
structure BusSeating where
  leftSeats : ℕ
  rightSeats : ℕ
  backSeatCapacity : ℕ
  seatCapacity : ℕ
  totalCapacity : ℕ

/-- The bus seating configuration satisfies the given conditions -/
def validBusSeating (bus : BusSeating) : Prop :=
  bus.rightSeats = bus.leftSeats - 3 ∧
  bus.backSeatCapacity = 11 ∧
  bus.seatCapacity = 3 ∧
  bus.totalCapacity = 92 ∧
  bus.totalCapacity = bus.seatCapacity * (bus.leftSeats + bus.rightSeats) + bus.backSeatCapacity

/-- The number of seats on the left side of the bus is 15 -/
theorem left_seats_count (bus : BusSeating) (h : validBusSeating bus) : bus.leftSeats = 15 := by
  sorry

end NUMINAMATH_CALUDE_left_seats_count_l2374_237481


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2374_237428

theorem fraction_equivalence : 
  let original_numerator : ℚ := 4
  let original_denominator : ℚ := 7
  let target_numerator : ℚ := 7
  let target_denominator : ℚ := 9
  let n : ℚ := 13/2
  (original_numerator + n) / (original_denominator + n) = target_numerator / target_denominator :=
by sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2374_237428


namespace NUMINAMATH_CALUDE_ochos_friends_ratio_l2374_237473

/-- Given that Ocho has 8 friends, all boys play theater with him, and 4 boys play theater with him,
    prove that the ratio of girls to boys among Ocho's friends is 1:1 -/
theorem ochos_friends_ratio (total_friends : ℕ) (boys_theater : ℕ) 
    (h1 : total_friends = 8)
    (h2 : boys_theater = 4) :
    (total_friends - boys_theater) / boys_theater = 1 := by
  sorry

end NUMINAMATH_CALUDE_ochos_friends_ratio_l2374_237473


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2374_237463

theorem complex_number_in_first_quadrant : 
  let z : ℂ := 1 / (1 - Complex.I)
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2374_237463


namespace NUMINAMATH_CALUDE_john_received_120_l2374_237484

/-- The amount of money John received from his grandpa -/
def grandpa_amount : ℕ := 30

/-- The amount of money John received from his grandma -/
def grandma_amount : ℕ := 3 * grandpa_amount

/-- The total amount of money John received from both grandparents -/
def total_amount : ℕ := grandpa_amount + grandma_amount

theorem john_received_120 : total_amount = 120 := by
  sorry

end NUMINAMATH_CALUDE_john_received_120_l2374_237484


namespace NUMINAMATH_CALUDE_stating_jasons_game_attendance_l2374_237465

/-- Represents the number of football games Jason attends in a month. -/
structure MonthlyGames where
  games : ℕ

/-- Represents Jason's football game attendance over three months. -/
structure ThreeMonthAttendance where
  lastMonth : MonthlyGames
  thisMonth : MonthlyGames
  nextMonth : MonthlyGames
  total : ℕ

/-- 
Theorem stating that given Jason's attendance for last month, 
next month, and the total for three months, we can determine 
the number of games he attended this month.
-/
theorem jasons_game_attendance 
  (attendance : ThreeMonthAttendance) 
  (h1 : attendance.lastMonth.games = 17) 
  (h2 : attendance.nextMonth.games = 16) 
  (h3 : attendance.total = 44) : 
  attendance.thisMonth.games = 11 := by
sorry

end NUMINAMATH_CALUDE_stating_jasons_game_attendance_l2374_237465


namespace NUMINAMATH_CALUDE_product_of_solutions_l2374_237482

theorem product_of_solutions (x : ℝ) : 
  (12 = 3 * x^2 + 18 * x) → 
  (∃ x₁ x₂ : ℝ, (12 = 3 * x₁^2 + 18 * x₁) ∧ (12 = 3 * x₂^2 + 18 * x₂) ∧ (x₁ * x₂ = -4)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2374_237482


namespace NUMINAMATH_CALUDE_two_digit_reverse_sum_l2374_237495

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

theorem two_digit_reverse_sum (n : ℕ) :
  is_two_digit n →
  (n : ℤ) - (reverse_digits n : ℤ) = 7 * ((n / 10 : ℤ) + (n % 10 : ℤ)) →
  (n : ℕ) + reverse_digits n = 99 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_reverse_sum_l2374_237495


namespace NUMINAMATH_CALUDE_trig_equation_solution_range_l2374_237442

theorem trig_equation_solution_range :
  ∀ m : ℝ, 
  (∀ x : ℝ, ∃ y : ℝ, 4 * Real.cos y + Real.sin y ^ 2 + m - 4 = 0) ↔ 
  (0 ≤ m ∧ m ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_trig_equation_solution_range_l2374_237442


namespace NUMINAMATH_CALUDE_parallel_from_equation_basis_transformation_l2374_237499

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Two non-zero vectors are parallel if one is a scalar multiple of the other -/
def Parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

theorem parallel_from_equation (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h : 2 • a = -3 • b) :
  Parallel a b := by sorry

theorem basis_transformation {e₁ e₂ : V} (h : LinearIndependent ℝ ![e₁, e₂]) :
  LinearIndependent ℝ ![e₁ + 2 • e₂, e₁ - 2 • e₂] := by sorry

end NUMINAMATH_CALUDE_parallel_from_equation_basis_transformation_l2374_237499


namespace NUMINAMATH_CALUDE_min_value_of_z_l2374_237444

theorem min_value_of_z (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x - 2*y + 3 = 0) :
  ∀ z : ℝ, z = y^2 / x → z ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_z_l2374_237444


namespace NUMINAMATH_CALUDE_sum_of_digits_for_special_triangle_l2374_237459

/-- Given a positive integer n, returns the sum of its digits -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The sum of the first n natural numbers -/
def triangle_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem sum_of_digits_for_special_triangle : 
  ∃ (N : ℕ), (triangle_sum N = 2145) ∧ (sum_of_digits N = 11) :=
sorry

end NUMINAMATH_CALUDE_sum_of_digits_for_special_triangle_l2374_237459


namespace NUMINAMATH_CALUDE_base6_addition_problem_l2374_237472

/-- Represents a digit in base 6 -/
def Base6Digit := Fin 6

/-- Checks if three Base6Digits are distinct -/
def are_distinct (s h e : Base6Digit) : Prop :=
  s ≠ h ∧ s ≠ e ∧ h ≠ e

/-- Converts a natural number to its base 6 representation -/
def to_base6 (n : ℕ) : ℕ :=
  sorry

/-- Adds two base 6 numbers -/
def base6_add (a b : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem base6_addition_problem :
  ∃ (s h e : Base6Digit),
    are_distinct s h e ∧
    0 < s.val ∧ 0 < h.val ∧ 0 < e.val ∧
    base6_add (s.val * 36 + h.val * 6 + e.val) (e.val * 36 + s.val * 6 + h.val) = s.val * 36 + h.val * 6 + s.val ∧
    s.val = 4 ∧ h.val = 2 ∧ e.val = 3 ∧
    to_base6 (s.val + h.val + e.val) = 13 :=
  sorry

end NUMINAMATH_CALUDE_base6_addition_problem_l2374_237472


namespace NUMINAMATH_CALUDE_reciprocal_of_three_halves_l2374_237404

theorem reciprocal_of_three_halves (x : ℚ) : x = 3 / 2 → 1 / x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_three_halves_l2374_237404


namespace NUMINAMATH_CALUDE_negative_three_and_half_equality_l2374_237468

theorem negative_three_and_half_equality : -4 + (1/2 : ℚ) = -(7/2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_negative_three_and_half_equality_l2374_237468


namespace NUMINAMATH_CALUDE_william_bottle_caps_l2374_237436

/-- Given that William initially had 2 bottle caps and now has 43 bottle caps in total,
    prove that he bought 41 bottle caps. -/
theorem william_bottle_caps :
  let initial_caps : ℕ := 2
  let total_caps : ℕ := 43
  let bought_caps : ℕ := total_caps - initial_caps
  bought_caps = 41 := by sorry

end NUMINAMATH_CALUDE_william_bottle_caps_l2374_237436


namespace NUMINAMATH_CALUDE_angle_with_special_supplement_and_complement_l2374_237496

theorem angle_with_special_supplement_and_complement :
  ∀ x : ℝ,
  (0 < x) →
  (x < 180) →
  (180 - x = 4 * (90 - x)) →
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_with_special_supplement_and_complement_l2374_237496


namespace NUMINAMATH_CALUDE_root_of_fifth_unity_l2374_237488

theorem root_of_fifth_unity (p q r s t k : ℂ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) (ht : t ≠ 0)
  (h1 : p * k^4 + q * k^3 + r * k^2 + s * k + t = 0)
  (h2 : q * k^4 + r * k^3 + s * k^2 + t * k + p = 0) :
  k^5 = 1 := by
sorry

end NUMINAMATH_CALUDE_root_of_fifth_unity_l2374_237488


namespace NUMINAMATH_CALUDE_archer_fish_catch_l2374_237464

def fish_problem (first_round : ℕ) (second_round_increase : ℕ) (third_round_percentage : ℕ) : Prop :=
  let second_round := first_round + second_round_increase
  let third_round := second_round + (second_round * third_round_percentage) / 100
  let total_fish := first_round + second_round + third_round
  total_fish = 60

theorem archer_fish_catch :
  fish_problem 8 12 60 :=
sorry

end NUMINAMATH_CALUDE_archer_fish_catch_l2374_237464


namespace NUMINAMATH_CALUDE_arctan_sum_eq_pi_over_four_l2374_237498

theorem arctan_sum_eq_pi_over_four :
  ∃! (n : ℕ), n > 0 ∧ Real.arctan (1/2) + Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/n) = π/4 :=
by sorry

end NUMINAMATH_CALUDE_arctan_sum_eq_pi_over_four_l2374_237498


namespace NUMINAMATH_CALUDE_phone_number_probability_l2374_237451

/-- Represents the possible prefixes for the phone number -/
def prefixes : Finset String := {"296", "299", "298"}

/-- Represents the digits for the remaining part of the phone number -/
def remainingDigits : Finset Char := {'0', '1', '6', '7', '9'}

/-- The total number of digits in the phone number -/
def totalDigits : Nat := 8

theorem phone_number_probability :
  (Finset.card prefixes * (Finset.card remainingDigits).factorial : ℚ)⁻¹ = 1 / 360 := by
  sorry

end NUMINAMATH_CALUDE_phone_number_probability_l2374_237451


namespace NUMINAMATH_CALUDE_box_volume_formula_l2374_237433

/-- The volume of an open box formed from a rectangular cardboard sheet. -/
def box_volume (y : ℝ) : ℝ :=
  (20 - 2*y) * (12 - 2*y) * y

theorem box_volume_formula (y : ℝ) 
  (h : 0 < y ∧ y < 6) : -- y is positive and less than half the smaller dimension
  box_volume y = 4*y^3 - 64*y^2 + 240*y := by
  sorry

end NUMINAMATH_CALUDE_box_volume_formula_l2374_237433


namespace NUMINAMATH_CALUDE_class_size_is_ten_l2374_237457

/-- The number of students who scored 92 -/
def high_scorers : ℕ := 5

/-- The number of students who scored 80 -/
def mid_scorers : ℕ := 4

/-- The score of the last student -/
def last_score : ℕ := 70

/-- The minimum required average score -/
def min_average : ℕ := 85

/-- The total number of students in the class -/
def total_students : ℕ := high_scorers + mid_scorers + 1

theorem class_size_is_ten :
  total_students = 10 ∧
  (high_scorers * 92 + mid_scorers * 80 + last_score) / total_students ≥ min_average := by
  sorry

end NUMINAMATH_CALUDE_class_size_is_ten_l2374_237457


namespace NUMINAMATH_CALUDE_function_value_2008_l2374_237462

theorem function_value_2008 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (4 - x)) 
  (h2 : ∀ x, f (2 - x) + f (x - 2) = 0) : 
  f 2008 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_value_2008_l2374_237462


namespace NUMINAMATH_CALUDE_system_solution_l2374_237447

/-- Given a system of linear equations and an additional equation,
    prove that k must equal 2 for the equations to have a common solution. -/
theorem system_solution (k : ℝ) : 
  (∃ x y : ℝ, x + y = 5*k ∧ x - y = k ∧ 2*x + 3*y = 24) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2374_237447


namespace NUMINAMATH_CALUDE_rent_share_ratio_l2374_237469

theorem rent_share_ratio (purity_share : ℚ) (rose_share : ℚ) (total_rent : ℚ) :
  rose_share = 1800 →
  total_rent = 5400 →
  total_rent = 5 * purity_share + purity_share + rose_share →
  rose_share / purity_share = 3 := by
  sorry

end NUMINAMATH_CALUDE_rent_share_ratio_l2374_237469


namespace NUMINAMATH_CALUDE_blithe_toy_collection_l2374_237493

/-- Given Blithe's toy collection changes, prove the initial number of toys. -/
theorem blithe_toy_collection (X : ℕ) : 
  X - 6 + 9 + 5 - 3 = 43 → X = 38 := by
  sorry

end NUMINAMATH_CALUDE_blithe_toy_collection_l2374_237493


namespace NUMINAMATH_CALUDE_even_function_decreasing_l2374_237479

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function f is decreasing on an interval (a, b) if
    for all x, y in (a, b), x < y implies f(x) > f(y) -/
def IsDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x > f y

theorem even_function_decreasing :
  let f := fun x => (m - 1) * x^2 + 2 * m * x + 3
  IsEven f →
  IsDecreasing f 2 5 :=
by
  sorry

#check even_function_decreasing

end NUMINAMATH_CALUDE_even_function_decreasing_l2374_237479


namespace NUMINAMATH_CALUDE_beaker_capacity_ratio_l2374_237476

theorem beaker_capacity_ratio : 
  ∀ (S L : ℝ), 
  S > 0 → L > 0 →
  (∃ k : ℝ, L = k * S) →
  (1/2 * S + 1/5 * L = 3/10 * L) →
  L / S = 5 := by
sorry

end NUMINAMATH_CALUDE_beaker_capacity_ratio_l2374_237476


namespace NUMINAMATH_CALUDE_third_number_value_l2374_237435

theorem third_number_value (a b c : ℕ) : 
  a + b + c = 500 → 
  a = 200 → 
  b = 2 * c → 
  c = 100 := by
sorry

end NUMINAMATH_CALUDE_third_number_value_l2374_237435


namespace NUMINAMATH_CALUDE_max_m_proof_l2374_237410

/-- The maximum value of m given the condition -/
def max_m : ℝ := -2

/-- The condition function -/
def condition (x : ℝ) : Prop := x^2 - 2*x - 8 > 0

/-- The main theorem -/
theorem max_m_proof :
  (∀ x : ℝ, x < max_m → condition x) ∧
  (∃ x : ℝ, x < max_m ∧ ¬condition x) ∧
  (∀ m : ℝ, m > max_m → ∃ x : ℝ, x < m ∧ ¬condition x) :=
sorry

end NUMINAMATH_CALUDE_max_m_proof_l2374_237410


namespace NUMINAMATH_CALUDE_purchase_ways_count_l2374_237424

/-- Represents the number of oreo flavors --/
def oreo_flavors : ℕ := 6

/-- Represents the number of milk flavors --/
def milk_flavors : ℕ := 4

/-- Represents the total number of products they purchase collectively --/
def total_products : ℕ := 4

/-- Represents the maximum number of same flavor items Alpha can order --/
def alpha_max_same_flavor : ℕ := 2

/-- Function to calculate the number of ways Alpha and Beta can purchase products --/
def purchase_ways : ℕ := sorry

/-- Theorem stating the correct number of ways to purchase products --/
theorem purchase_ways_count : purchase_ways = 2143 := by sorry

end NUMINAMATH_CALUDE_purchase_ways_count_l2374_237424


namespace NUMINAMATH_CALUDE_triangle_radii_inequality_l2374_237470

/-- For any triangle with circumradius R, inradius r, and exradii r_a, r_b, r_c,
    the inequality (r * r_a * r_b * r_c) / R^4 ≤ 27/16 holds. -/
theorem triangle_radii_inequality (R r r_a r_b r_c : ℝ) 
    (h_R : R > 0) 
    (h_r : r > 0) 
    (h_ra : r_a > 0) 
    (h_rb : r_b > 0) 
    (h_rc : r_c > 0) 
    (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
      R = (a * b * c) / (4 * (a + b + c) * (a + b - c) * (b + c - a) * (c + a - b))^(1/2) ∧
      r = (a + b - c) * (b + c - a) * (c + a - b) / (4 * (a + b + c)) ∧
      r_a = (b + c - a) / 2 ∧
      r_b = (c + a - b) / 2 ∧
      r_c = (a + b - c) / 2) :
  (r * r_a * r_b * r_c) / R^4 ≤ 27/16 := by
sorry

end NUMINAMATH_CALUDE_triangle_radii_inequality_l2374_237470


namespace NUMINAMATH_CALUDE_mean_of_two_numbers_l2374_237408

theorem mean_of_two_numbers (a b c : ℝ) : 
  (a + b + c + 100) / 4 = 90 →
  a = 70 →
  a ≤ b ∧ b ≤ c ∧ c ≤ 100 →
  (b + c) / 2 = 95 := by
sorry

end NUMINAMATH_CALUDE_mean_of_two_numbers_l2374_237408


namespace NUMINAMATH_CALUDE_extremum_of_f_l2374_237490

/-- The function f(x) = (k-1)x^2 - 2(k-1)x - k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x^2 - 2 * (k - 1) * x - k

/-- Theorem: Extremum of f(x) when k ≠ 1 -/
theorem extremum_of_f (k : ℝ) (h : k ≠ 1) :
  (k > 1 → ∀ x, f k x ≥ -2 * k + 1) ∧
  (k < 1 → ∀ x, f k x ≤ -2 * k + 1) := by
  sorry

end NUMINAMATH_CALUDE_extremum_of_f_l2374_237490


namespace NUMINAMATH_CALUDE_math_club_members_l2374_237411

/-- 
Given a Math club where:
- There are two times as many males as females
- There are 6 female members
Prove that the total number of members in the Math club is 18.
-/
theorem math_club_members :
  ∀ (female_members male_members total_members : ℕ),
    female_members = 6 →
    male_members = 2 * female_members →
    total_members = female_members + male_members →
    total_members = 18 := by
  sorry

end NUMINAMATH_CALUDE_math_club_members_l2374_237411


namespace NUMINAMATH_CALUDE_triangle_max_area_l2374_237494

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the maximum area of the triangle is 2 + √3 when b²-2√3bc*sin(A)+c²=4 and a=2 -/
theorem triangle_max_area (a b c : ℝ) (A B C : ℝ) :
  a = 2 →
  b^2 - 2 * Real.sqrt 3 * b * c * Real.sin A + c^2 = 4 →
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  (∃ (S : ℝ), S = (1/2) * b * c * Real.sin A ∧
    ∀ (S' : ℝ), S' = (1/2) * b * c * Real.sin A → S' ≤ 2 + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_area_l2374_237494


namespace NUMINAMATH_CALUDE_chairs_produced_in_six_hours_l2374_237418

/-- The number of chairs produced by a group of workers over a given time period -/
def chairs_produced (num_workers : ℕ) (chairs_per_worker_per_hour : ℕ) (additional_chairs : ℕ) (hours : ℕ) : ℕ :=
  num_workers * chairs_per_worker_per_hour * hours + additional_chairs

/-- Theorem stating that 3 workers producing 4 chairs per hour, with an additional chair every 6 hours, produce 73 chairs in 6 hours -/
theorem chairs_produced_in_six_hours :
  chairs_produced 3 4 1 6 = 73 := by
  sorry

end NUMINAMATH_CALUDE_chairs_produced_in_six_hours_l2374_237418


namespace NUMINAMATH_CALUDE_ten_percent_of_point_one_l2374_237446

theorem ten_percent_of_point_one (x : ℝ) : x = 0.1 * 0.10 → x = 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ten_percent_of_point_one_l2374_237446


namespace NUMINAMATH_CALUDE_alia_markers_l2374_237429

theorem alia_markers (steve_markers : ℕ) (austin_markers : ℕ) (alia_markers : ℕ)
  (h1 : steve_markers = 60)
  (h2 : austin_markers = steve_markers / 3)
  (h3 : alia_markers = 2 * austin_markers) :
  alia_markers = 40 := by
sorry

end NUMINAMATH_CALUDE_alia_markers_l2374_237429


namespace NUMINAMATH_CALUDE_odd_number_probability_l2374_237434

/-- The set of digits used to form the number -/
def digits : Finset Nat := {1, 4, 6, 9}

/-- The set of odd digits from the given set -/
def oddDigits : Finset Nat := {1, 9}

/-- The probability of forming an odd four-digit number -/
def probabilityOdd : ℚ := (oddDigits.card : ℚ) / (digits.card : ℚ)

/-- Theorem stating that the probability of forming an odd four-digit number is 1/2 -/
theorem odd_number_probability : probabilityOdd = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_odd_number_probability_l2374_237434


namespace NUMINAMATH_CALUDE_chicken_problem_l2374_237453

/-- The number of chickens Colten has -/
def colten : ℕ := 37

/-- The number of chickens Skylar has -/
def skylar : ℕ := 3 * colten - 4

/-- The number of chickens Quentin has -/
def quentin : ℕ := 2 * skylar + 25

/-- The total number of chickens -/
def total : ℕ := 383

theorem chicken_problem :
  colten + skylar + quentin = total :=
sorry

end NUMINAMATH_CALUDE_chicken_problem_l2374_237453


namespace NUMINAMATH_CALUDE_robotics_club_age_problem_l2374_237412

theorem robotics_club_age_problem (total_members : ℕ) (girls : ℕ) (boys : ℕ) (adults : ℕ)
  (overall_avg : ℚ) (girls_avg : ℚ) (boys_avg : ℚ) :
  total_members = 30 →
  girls = 10 →
  boys = 10 →
  adults = 10 →
  overall_avg = 22 →
  girls_avg = 18 →
  boys_avg = 20 →
  (total_members * overall_avg - girls * girls_avg - boys * boys_avg) / adults = 28 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_age_problem_l2374_237412


namespace NUMINAMATH_CALUDE_function_domain_real_iff_m_in_range_l2374_237497

/-- The function y = (mx - 1) / (mx² + 4mx + 3) has domain ℝ if and only if m ∈ [0, 3/4) -/
theorem function_domain_real_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, mx^2 + 4*m*x + 3 ≠ 0) ↔ m ∈ Set.Ici 0 ∩ Set.Iio (3/4) :=
by sorry

end NUMINAMATH_CALUDE_function_domain_real_iff_m_in_range_l2374_237497


namespace NUMINAMATH_CALUDE_fourth_sphere_radius_l2374_237419

/-- A cone with four spheres inside, where three spheres have radius 3 and touch the base. -/
structure ConeFourSpheres where
  /-- Radius of the three identical spheres -/
  r₁ : ℝ
  /-- Radius of the fourth sphere -/
  r₂ : ℝ
  /-- Angle between the slant height and the base of the cone -/
  θ : ℝ
  /-- The three identical spheres touch the base of the cone -/
  touch_base : True
  /-- All spheres touch each other externally -/
  touch_externally : True
  /-- All spheres touch the lateral surface of the cone -/
  touch_lateral : True
  /-- The radius of the three identical spheres is 3 -/
  r₁_eq_3 : r₁ = 3
  /-- The angle between the slant height and the base of the cone is π/3 -/
  θ_eq_pi_div_3 : θ = π / 3

/-- The radius of the fourth sphere in the cone arrangement is 9 - 4√2 -/
theorem fourth_sphere_radius (c : ConeFourSpheres) : c.r₂ = 9 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_fourth_sphere_radius_l2374_237419


namespace NUMINAMATH_CALUDE_max_toys_purchasable_l2374_237417

theorem max_toys_purchasable (initial_amount : ℚ) (game_cost : ℚ) (book_cost : ℚ) (toy_cost : ℚ) :
  initial_amount = 57.45 →
  game_cost = 26.89 →
  book_cost = 12.37 →
  toy_cost = 6 →
  ⌊(initial_amount - game_cost - book_cost) / toy_cost⌋ = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_toys_purchasable_l2374_237417


namespace NUMINAMATH_CALUDE_theater_sales_proof_l2374_237439

/-- Calculates the total ticket sales for a theater performance. -/
def theater_sales (adult_price child_price total_attendance children_attendance : ℕ) : ℕ :=
  let adults := total_attendance - children_attendance
  let adult_sales := adults * adult_price
  let child_sales := children_attendance * child_price
  adult_sales + child_sales

/-- Theorem stating that given the specific conditions, the theater collects $50 from ticket sales. -/
theorem theater_sales_proof :
  theater_sales 8 1 22 18 = 50 := by
  sorry

end NUMINAMATH_CALUDE_theater_sales_proof_l2374_237439


namespace NUMINAMATH_CALUDE_maria_trip_expenses_l2374_237486

def initial_amount : ℕ := 760
def ticket_cost : ℕ := 300
def hotel_cost : ℕ := ticket_cost / 2
def total_spent : ℕ := ticket_cost + hotel_cost
def remaining_amount : ℕ := initial_amount - total_spent

theorem maria_trip_expenses :
  remaining_amount = 310 :=
sorry

end NUMINAMATH_CALUDE_maria_trip_expenses_l2374_237486


namespace NUMINAMATH_CALUDE_stocking_discount_percentage_l2374_237427

-- Define the given conditions
def num_grandchildren : ℕ := 5
def num_children : ℕ := 4
def stockings_per_person : ℕ := 5
def stocking_price : ℚ := 20
def monogram_price : ℚ := 5
def total_cost_after_discount : ℚ := 1035

-- Define the theorem
theorem stocking_discount_percentage :
  let total_people := num_grandchildren + num_children
  let total_stockings := total_people * stockings_per_person
  let stocking_cost := total_stockings * stocking_price
  let monogram_cost := total_stockings * monogram_price
  let total_cost_before_discount := stocking_cost + monogram_cost
  let discount_amount := total_cost_before_discount - total_cost_after_discount
  let discount_percentage := (discount_amount / total_cost_before_discount) * 100
  discount_percentage = 8 := by
sorry

end NUMINAMATH_CALUDE_stocking_discount_percentage_l2374_237427


namespace NUMINAMATH_CALUDE_subtraction_equation_solution_l2374_237489

def is_valid_subtraction (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
  1000 * a + 100 * b + 82 - (900 + 10 * c + 9) = 4000 + 90 * d + 3

theorem subtraction_equation_solution :
  ∀ a b c d : Nat, is_valid_subtraction a b c d → a = 5 :=
sorry

end NUMINAMATH_CALUDE_subtraction_equation_solution_l2374_237489


namespace NUMINAMATH_CALUDE_wire_service_reporters_l2374_237416

theorem wire_service_reporters (total : ℝ) (local_politics : ℝ) (non_local_politics : ℝ) 
  (h1 : local_politics = 0.18 * total)
  (h2 : non_local_politics = 0.4 * (local_politics + non_local_politics)) :
  (total - (local_politics + non_local_politics)) / total = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l2374_237416


namespace NUMINAMATH_CALUDE_taxi_fare_proof_l2374_237415

/-- Proves that given an initial fare of $2.00 for the first 1/5 mile and a total fare of $25.40 for an 8-mile ride, the fare for each 1/5 mile after the first 1/5 mile is $0.60. -/
theorem taxi_fare_proof (initial_fare : ℝ) (total_fare : ℝ) (ride_distance : ℝ) 
  (h1 : initial_fare = 2)
  (h2 : total_fare = 25.4)
  (h3 : ride_distance = 8) :
  let increments : ℝ := ride_distance * 5
  let remaining_fare : ℝ := total_fare - initial_fare
  let remaining_increments : ℝ := increments - 1
  remaining_fare / remaining_increments = 0.6 := by
sorry

end NUMINAMATH_CALUDE_taxi_fare_proof_l2374_237415


namespace NUMINAMATH_CALUDE_marias_additional_cupcakes_l2374_237400

/-- Given that Maria initially made 19 cupcakes, sold 5, and ended up with 24 cupcakes,
    prove that she made 10 additional cupcakes. -/
theorem marias_additional_cupcakes :
  let initial_cupcakes : ℕ := 19
  let sold_cupcakes : ℕ := 5
  let final_cupcakes : ℕ := 24
  let additional_cupcakes := final_cupcakes - (initial_cupcakes - sold_cupcakes)
  additional_cupcakes = 10 := by sorry

end NUMINAMATH_CALUDE_marias_additional_cupcakes_l2374_237400


namespace NUMINAMATH_CALUDE_expression_evaluation_l2374_237422

theorem expression_evaluation (x : ℕ) (h : x = 3) : x^2 + x * (x^(x^2)) = 59058 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2374_237422


namespace NUMINAMATH_CALUDE_inequality_proof_l2374_237438

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 / (1 + Real.sqrt x)^2) + (1 / (1 + Real.sqrt y)^2) ≥ 2 / (x + y + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2374_237438


namespace NUMINAMATH_CALUDE_chord_length_squared_in_sector_l2374_237445

/-- Given a circular sector with central angle 60° and radius 6 cm, 
    the square of the chord length subtending the central angle is 36 cm^2 -/
theorem chord_length_squared_in_sector (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = π/3) :
  (2 * r * Real.sin (θ/2))^2 = 36 :=
sorry

end NUMINAMATH_CALUDE_chord_length_squared_in_sector_l2374_237445


namespace NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l2374_237401

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem least_three_digit_with_digit_product_12 :
  ∀ n : ℕ, is_three_digit n → digit_product n = 12 → 134 ≤ n :=
sorry

end NUMINAMATH_CALUDE_least_three_digit_with_digit_product_12_l2374_237401


namespace NUMINAMATH_CALUDE_colleen_pencils_colleen_pencils_proof_l2374_237414

theorem colleen_pencils (joy_pencils : ℕ) (pencil_cost : ℕ) (colleen_extra : ℕ) : ℕ :=
  let joy_total := joy_pencils * pencil_cost
  let colleen_total := joy_total + colleen_extra
  colleen_total / pencil_cost

#check colleen_pencils 30 4 80 = 50

theorem colleen_pencils_proof :
  colleen_pencils 30 4 80 = 50 := by
  sorry

end NUMINAMATH_CALUDE_colleen_pencils_colleen_pencils_proof_l2374_237414


namespace NUMINAMATH_CALUDE_baby_grab_theorem_l2374_237426

/-- Represents the number of possible outcomes when a baby grabs one item from a set of items -/
def possible_outcomes (educational living entertainment : ℕ) : ℕ :=
  educational + living + entertainment

/-- Theorem: The number of possible outcomes when a baby grabs one item
    is equal to the sum of educational, living, and entertainment items -/
theorem baby_grab_theorem (educational living entertainment : ℕ) :
  possible_outcomes educational living entertainment =
  educational + living + entertainment := by
  sorry

end NUMINAMATH_CALUDE_baby_grab_theorem_l2374_237426


namespace NUMINAMATH_CALUDE_opposite_reciprocal_problem_l2374_237413

theorem opposite_reciprocal_problem (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 4) : 
  (a + b = 0) ∧ 
  (c * d = 1) ∧ 
  ((a + b) / 3 + m^2 - 5 * c * d = 11) := by
sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_problem_l2374_237413


namespace NUMINAMATH_CALUDE_function_equals_identity_l2374_237466

theorem function_equals_identity (f : ℝ → ℝ) :
  (ContinuousOn f (Set.Icc 0 1)) →
  (f 0 = 0) →
  (f 1 = 1) →
  (∀ x ∈ Set.Ioo 0 1, ∃ h > 0, 
    0 ≤ x - h ∧ x + h ≤ 1 ∧ 
    f x = (f (x - h) + f (x + h)) / 2) →
  (∀ x ∈ Set.Icc 0 1, f x = x) := by
sorry

end NUMINAMATH_CALUDE_function_equals_identity_l2374_237466


namespace NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l2374_237441

/-- The total surface area of a hemisphere (excluding its base) and cylinder side surface -/
theorem hemisphere_cylinder_surface_area (r h : ℝ) (hr : r = 5) (hh : h = 10) :
  2 * π * r * h + 2 * π * r^2 = 150 * π := by sorry

end NUMINAMATH_CALUDE_hemisphere_cylinder_surface_area_l2374_237441


namespace NUMINAMATH_CALUDE_solution_to_system_l2374_237478

theorem solution_to_system (x y : ℝ) : 
  x = (3^(1/5) + 1) / 2 ∧ y = (3^(1/5) - 1) / 2 →
  (1 / x - 1 / (2 * y) = 2 * y^4 - 2 * x^4) ∧
  (1 / x + 1 / (2 * y) = (3 * x^2 + y^2) * (x^2 + 3 * y^2)) := by
sorry

end NUMINAMATH_CALUDE_solution_to_system_l2374_237478


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2374_237423

theorem sum_of_reciprocals_of_roots (x : ℝ) : 
  x^2 - 17*x + 6 = 0 → 
  ∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 17*x + 6 = (x - r₁) * (x - r₂) ∧ 
  (1 / r₁ + 1 / r₂ = 17 / 6) := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l2374_237423


namespace NUMINAMATH_CALUDE_ellipse_condition_l2374_237406

/-- A curve represented by the equation x²/(7-m) + y²/(m-3) = 1 is an ellipse -/
def is_ellipse (m : ℝ) : Prop :=
  7 - m > 0 ∧ m - 3 > 0 ∧ 7 - m ≠ m - 3

/-- The condition 3 < m < 7 is necessary but not sufficient for the curve to be an ellipse -/
theorem ellipse_condition (m : ℝ) :
  (is_ellipse m → 3 < m ∧ m < 7) ∧
  ¬(3 < m ∧ m < 7 → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2374_237406


namespace NUMINAMATH_CALUDE_fraction_equality_l2374_237474

theorem fraction_equality : (1 + 3 + 5) / (10 + 6 + 2) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2374_237474


namespace NUMINAMATH_CALUDE_prism_with_seven_faces_has_fifteen_edges_l2374_237420

/-- A prism is a polyhedron with two congruent and parallel faces (bases) 
    and all other faces are parallelograms (lateral faces). -/
structure Prism where
  faces : ℕ
  bases : ℕ
  lateral_faces : ℕ
  edges_per_base : ℕ
  lateral_edges : ℕ
  total_edges : ℕ

/-- The number of edges in a prism with 7 faces is 15. -/
theorem prism_with_seven_faces_has_fifteen_edges :
  ∀ (p : Prism), p.faces = 7 → p.total_edges = 15 := by
  sorry


end NUMINAMATH_CALUDE_prism_with_seven_faces_has_fifteen_edges_l2374_237420


namespace NUMINAMATH_CALUDE_recipe_ratio_l2374_237431

/-- Given a recipe with 5 cups of flour and 1 cup of shortening,
    if 2/3 cup of shortening is used, then 3 1/3 cups of flour
    should be used to maintain the same ratio. -/
theorem recipe_ratio (original_flour : ℚ) (original_shortening : ℚ)
                     (available_shortening : ℚ) (needed_flour : ℚ) :
  original_flour = 5 →
  original_shortening = 1 →
  available_shortening = 2/3 →
  needed_flour = 10/3 →
  needed_flour / available_shortening = original_flour / original_shortening :=
by sorry

end NUMINAMATH_CALUDE_recipe_ratio_l2374_237431


namespace NUMINAMATH_CALUDE_inscribed_box_radius_l2374_237402

/-- A rectangular box inscribed in a sphere --/
structure InscribedBox where
  x : ℝ
  y : ℝ
  z : ℝ
  r : ℝ
  h_surface_area : 2 * (x*y + x*z + y*z) = 432
  h_edge_sum : 4 * (x + y + z) = 104
  h_inscribed : (2*r)^2 = x^2 + y^2 + z^2

/-- Theorem: If a rectangular box Q is inscribed in a sphere, with surface area 432,
    sum of edge lengths 104, and one dimension 8, then the radius of the sphere is 7 --/
theorem inscribed_box_radius (Q : InscribedBox) (h_x : Q.x = 8) : Q.r = 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_box_radius_l2374_237402


namespace NUMINAMATH_CALUDE_circle_packing_problem_l2374_237430

theorem circle_packing_problem (n : ℕ) :
  (n^2 = ((n + 14) * (n + 15)) / 2) → n^2 = 1225 := by
  sorry

end NUMINAMATH_CALUDE_circle_packing_problem_l2374_237430


namespace NUMINAMATH_CALUDE_initial_manager_percentage_l2374_237440

/-- The initial percentage of managers in a room with 200 employees,
    given that 99.99999999999991 managers leave and the resulting
    percentage is 98%, is approximately 99%. -/
theorem initial_manager_percentage :
  let total_employees : ℕ := 200
  let managers_who_left : ℝ := 99.99999999999991
  let final_percentage : ℝ := 98
  let initial_percentage : ℝ := 
    ((managers_who_left + (final_percentage / 100) * (total_employees - managers_who_left)) / total_employees) * 100
  ∀ ε > 0, |initial_percentage - 99| < ε :=
by sorry


end NUMINAMATH_CALUDE_initial_manager_percentage_l2374_237440


namespace NUMINAMATH_CALUDE_sum_of_digits_of_b_is_nine_l2374_237471

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A function that checks if a natural number has exactly 1995 digits -/
def has1995Digits (n : ℕ) : Prop := sorry

theorem sum_of_digits_of_b_is_nine (N : ℕ) 
  (h1 : has1995Digits N) 
  (h2 : N % 9 = 0) : 
  let a := sumOfDigits N
  let b := sumOfDigits a
  sumOfDigits b = 9 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_b_is_nine_l2374_237471


namespace NUMINAMATH_CALUDE_max_value_sum_cubes_fourth_powers_l2374_237449

theorem max_value_sum_cubes_fourth_powers (a b c : ℕ+) 
  (h : a + b + c = 2) : 
  (∀ x y z : ℕ+, x + y + z = 2 → a + b^3 + c^4 ≥ x + y^3 + z^4) ∧ 
  (∃ x y z : ℕ+, x + y + z = 2 ∧ a + b^3 + c^4 = x + y^3 + z^4) :=
sorry

end NUMINAMATH_CALUDE_max_value_sum_cubes_fourth_powers_l2374_237449


namespace NUMINAMATH_CALUDE_chess_swimming_percentage_l2374_237421

theorem chess_swimming_percentage (total_students : ℕ) 
  (chess_percentage : ℚ) (swimming_students : ℕ) :
  total_students = 1000 →
  chess_percentage = 1/5 →
  swimming_students = 20 →
  (swimming_students : ℚ) / (chess_percentage * total_students) * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_chess_swimming_percentage_l2374_237421


namespace NUMINAMATH_CALUDE_factorization_a_squared_minus_four_a_l2374_237461

theorem factorization_a_squared_minus_four_a (a : ℝ) : a^2 - 4*a = a*(a - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_a_squared_minus_four_a_l2374_237461


namespace NUMINAMATH_CALUDE_correct_algebraic_equation_l2374_237492

theorem correct_algebraic_equation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_algebraic_equation_l2374_237492


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l2374_237425

/-- Given three points A, B, and C in a plane, where C divides AB in a 1:2 ratio,
    and the coordinates of B and C are known, prove that the sum of A's coordinates is 9. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/3 →
  (B.1 - C.1) / (B.1 - A.1) = 2/3 →
  B = (2, 8) →
  C = (5, 2) →
  A.1 + A.2 = 9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l2374_237425


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_N_l2374_237405

def M : Set Int := {-1, 0, 1}

def N : Set Int := {x | ∃ a b, a ∈ M ∧ b ∈ M ∧ a ≠ b ∧ x = a * b}

theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_N_l2374_237405


namespace NUMINAMATH_CALUDE_average_of_first_two_l2374_237483

theorem average_of_first_two (total_avg : ℝ) (second_set_avg : ℝ) (third_set_avg : ℝ)
  (h1 : total_avg = 2.5)
  (h2 : second_set_avg = 1.4)
  (h3 : third_set_avg = 5) :
  let total_sum := 6 * total_avg
  let second_set_sum := 2 * second_set_avg
  let third_set_sum := 2 * third_set_avg
  let first_set_sum := total_sum - second_set_sum - third_set_sum
  first_set_sum / 2 = 1.1 := by
sorry

end NUMINAMATH_CALUDE_average_of_first_two_l2374_237483


namespace NUMINAMATH_CALUDE_sum_squared_l2374_237485

theorem sum_squared (x y : ℝ) (h1 : x * (x + y) = 27) (h2 : y * (x + y) = 54) : 
  (x + y)^2 = 81 := by
sorry

end NUMINAMATH_CALUDE_sum_squared_l2374_237485


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2374_237487

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (x + y > 2 → (x > 1 ∨ y > 1)) ∧
  ¬((x > 1 ∨ y > 1) → (x + y > 2)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2374_237487


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l2374_237455

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x - 3| = 5 - 2*x ↔ x = 8/3 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l2374_237455


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l2374_237437

theorem perfect_square_divisibility (a p q : ℕ+) (h1 : ∃ k : ℕ+, a = k ^ 2) 
  (h2 : a = p * q) (h3 : (2021 : ℕ) ∣ p ^ 3 + q ^ 3 + p ^ 2 * q + p * q ^ 2) :
  (2021 : ℕ) ∣ Nat.sqrt a.val := by sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l2374_237437


namespace NUMINAMATH_CALUDE_total_badges_sum_l2374_237407

/-- The total number of spelling badges for Hermione, Luna, and Celestia -/
def total_badges (hermione_badges luna_badges celestia_badges : ℕ) : ℕ :=
  hermione_badges + luna_badges + celestia_badges

/-- Theorem stating that the total number of spelling badges is 83 -/
theorem total_badges_sum : total_badges 14 17 52 = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_badges_sum_l2374_237407


namespace NUMINAMATH_CALUDE_total_ants_l2374_237452

theorem total_ants (abe beth cece duke : ℕ) : 
  abe = 4 →
  beth = abe + abe / 2 →
  cece = 2 * abe →
  duke = abe / 2 →
  abe + beth + cece + duke = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_ants_l2374_237452


namespace NUMINAMATH_CALUDE_block_weight_difference_l2374_237491

theorem block_weight_difference :
  let yellow_weight : ℝ := 0.6
  let green_weight : ℝ := 0.4
  yellow_weight - green_weight = 0.2 := by
sorry

end NUMINAMATH_CALUDE_block_weight_difference_l2374_237491


namespace NUMINAMATH_CALUDE_pyramid_volume_in_cube_l2374_237448

theorem pyramid_volume_in_cube (s : ℝ) (h : s > 0) :
  let cube_volume := s^3
  let pyramid_volume := (1/3) * (s^2/2) * s
  pyramid_volume = (1/6) * cube_volume := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_in_cube_l2374_237448


namespace NUMINAMATH_CALUDE_common_chord_equation_l2374_237409

/-- Given two circles with equations x^2 + y^2 - 4x = 0 and x^2 + y^2 - 4y = 0,
    the equation of the line where their common chord lies is x - y = 0. -/
theorem common_chord_equation (x y : ℝ) : 
  (x^2 + y^2 - 4*x = 0 ∧ x^2 + y^2 - 4*y = 0) → x - y = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l2374_237409


namespace NUMINAMATH_CALUDE_scaled_circle_area_l2374_237454

/-- Given a circle with center P(-5, 3) passing through Q(7, -4), 
    when uniformly scaled by a factor of 2 from its center, 
    the area of the resulting circle is 772π. -/
theorem scaled_circle_area : 
  let P : ℝ × ℝ := (-5, 3)
  let Q : ℝ × ℝ := (7, -4)
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  let scale_factor : ℝ := 2
  let scaled_area := π * (scale_factor * r)^2
  scaled_area = 772 * π :=
by sorry

end NUMINAMATH_CALUDE_scaled_circle_area_l2374_237454


namespace NUMINAMATH_CALUDE_arithmetic_sequence_term_number_l2374_237443

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_term_number
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a5 : a 5 = 33)
  (h_a45 : a 45 = 153)
  : (∃ n : ℕ, a n = 201) ∧ (∀ n : ℕ, a n = 201 → n = 61) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_term_number_l2374_237443


namespace NUMINAMATH_CALUDE_separate_amount_possible_l2374_237467

/-- Represents the result of a weighing operation -/
inductive WeighingResult
  | EqualGroup (value : ℚ)
  | UnequalGroups (value1 value2 : ℚ)

/-- Represents a weighing operation -/
def Weighing := List ℚ → WeighingResult

/-- The total amount of money in rubles -/
def total_amount : ℚ := 80

/-- The value of a single coin in rubles -/
def coin_value : ℚ := 1/20

/-- The target amount to be separated -/
def target_amount : ℚ := 25

/-- The maximum number of weighings allowed -/
def max_weighings : ℕ := 4

/-- 
  Proves that it's possible to separate the target amount from the total amount 
  using coins of the given value with only a balance scale in the specified number of weighings
-/
theorem separate_amount_possible : 
  ∃ (weighings : List Weighing), 
    weighings.length ≤ max_weighings ∧ 
    ∃ (result : List ℚ), 
      result.sum = target_amount ∧ 
      result.all (λ x => x ≤ total_amount) :=
sorry

end NUMINAMATH_CALUDE_separate_amount_possible_l2374_237467


namespace NUMINAMATH_CALUDE_exists_special_set_l2374_237458

/-- A function that checks if a natural number is a perfect power -/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ (b k : ℕ), k > 1 ∧ n = b^k

/-- The existence of a set of 1992 positive integers with the required property -/
theorem exists_special_set : ∃ (S : Finset ℕ), 
  (S.card = 1992) ∧ 
  (∀ (T : Finset ℕ), T ⊆ S → isPerfectPower (T.sum id)) :=
sorry

end NUMINAMATH_CALUDE_exists_special_set_l2374_237458


namespace NUMINAMATH_CALUDE_running_race_participants_l2374_237403

theorem running_race_participants (first_grade : ℕ) (second_grade : ℕ) : 
  first_grade = 8 →
  second_grade = 5 * first_grade →
  first_grade + second_grade = 48 := by
  sorry

end NUMINAMATH_CALUDE_running_race_participants_l2374_237403


namespace NUMINAMATH_CALUDE_correct_calculation_l2374_237475

theorem correct_calculation (a b : ℝ) : 2 * a^2 * b - 3 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2374_237475
