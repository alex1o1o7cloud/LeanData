import Mathlib

namespace NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l1329_132922

theorem distinct_prime_factors_of_30_factorial (n : ℕ) :
  n = 30 →
  (Finset.filter Nat.Prime (Finset.range (n + 1))).card =
  (Finset.filter (λ p => p.Prime ∧ p ∣ n!) (Finset.range (n + 1))).card :=
sorry

end NUMINAMATH_CALUDE_distinct_prime_factors_of_30_factorial_l1329_132922


namespace NUMINAMATH_CALUDE_special_item_identification_l1329_132943

/-- Represents the result of a yes/no question -/
inductive Answer
| Yes
| No

/-- Converts an Answer to a natural number (0 for Yes, 1 for No) -/
def answerToNat (a : Answer) : Nat :=
  match a with
  | Answer.Yes => 0
  | Answer.No => 1

/-- Represents the set of items -/
def Items : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7}

/-- The function to determine the special item based on three answers -/
def determineSpecialItem (a₁ a₂ a₃ : Answer) : Nat :=
  answerToNat a₁ + 2 * answerToNat a₂ + 4 * answerToNat a₃

theorem special_item_identification :
  ∀ (special : Nat),
  special ∈ Items →
  ∃ (a₁ a₂ a₃ : Answer),
  determineSpecialItem a₁ a₂ a₃ = special ∧
  ∀ (other : Nat),
  other ∈ Items →
  other ≠ special →
  determineSpecialItem a₁ a₂ a₃ ≠ other :=
sorry

end NUMINAMATH_CALUDE_special_item_identification_l1329_132943


namespace NUMINAMATH_CALUDE_classroom_desks_l1329_132919

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Calculates the number of desks needed given the number of students and students per desk -/
def calculateDesks (students : ℕ) (studentsPerDesk : ℕ) : ℕ := sorry

theorem classroom_desks :
  let studentsBase6 : ℕ := 305
  let studentsPerDesk : ℕ := 3
  let studentsBase10 : ℕ := base6ToBase10 studentsBase6
  calculateDesks studentsBase10 studentsPerDesk = 38 := by sorry

end NUMINAMATH_CALUDE_classroom_desks_l1329_132919


namespace NUMINAMATH_CALUDE_coordinates_of_N_l1329_132987

/-- Given a point M and a line segment MN parallel to the x-axis, 
    this function returns the possible coordinates of point N -/
def possible_coordinates_of_N (M : ℝ × ℝ) (length_MN : ℝ) : Set (ℝ × ℝ) :=
  let (x, y) := M
  { (x - length_MN, y), (x + length_MN, y) }

/-- Theorem stating that given M(2, -4) and MN of length 5 parallel to x-axis,
    N has coordinates either (-3, -4) or (7, -4) -/
theorem coordinates_of_N : 
  possible_coordinates_of_N (2, -4) 5 = {(-3, -4), (7, -4)} := by
  sorry


end NUMINAMATH_CALUDE_coordinates_of_N_l1329_132987


namespace NUMINAMATH_CALUDE_sin_30_degrees_l1329_132970

theorem sin_30_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_degrees_l1329_132970


namespace NUMINAMATH_CALUDE_minimum_in_interval_implies_a_range_l1329_132927

open Real

/-- The function f(x) = x³ - 2ax + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*a*x + a

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a

theorem minimum_in_interval_implies_a_range (a : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, IsLocalMin (f a) x) →
  (∀ x ∈ Set.Ioo 0 1, ¬IsLocalMax (f a) x) →
  a ∈ Set.Ioo 0 (3/2) :=
by sorry

end NUMINAMATH_CALUDE_minimum_in_interval_implies_a_range_l1329_132927


namespace NUMINAMATH_CALUDE_sheets_per_day_l1329_132950

def sheets_per_pad : ℕ := 60
def working_days_per_week : ℕ := 5

theorem sheets_per_day :
  sheets_per_pad / working_days_per_week = 12 := by
  sorry

end NUMINAMATH_CALUDE_sheets_per_day_l1329_132950


namespace NUMINAMATH_CALUDE_log_sum_exists_base_l1329_132959

theorem log_sum_exists_base : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, x > 0 → Real.log x / Real.log 2 + Real.log x / Real.log 3 = Real.log x / Real.log a := by
  sorry

end NUMINAMATH_CALUDE_log_sum_exists_base_l1329_132959


namespace NUMINAMATH_CALUDE_multiple_with_ones_and_zeros_multiple_with_only_ones_l1329_132914

def a (k : ℕ) : ℕ := (10^k - 1) / 9

theorem multiple_with_ones_and_zeros (n : ℤ) :
  ∃ k l : ℕ, k < l ∧ n ∣ (a l - a k) :=
sorry

theorem multiple_with_only_ones (n : ℤ) (h_odd : Odd n) (h_not_div_5 : ¬(5 ∣ n)) :
  ∃ d : ℕ, n ∣ (10^d - 1) :=
sorry

end NUMINAMATH_CALUDE_multiple_with_ones_and_zeros_multiple_with_only_ones_l1329_132914


namespace NUMINAMATH_CALUDE_equation_solution_l1329_132900

theorem equation_solution (x y z w : ℝ) (h : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
  (eq : 1/x + 1/y = 1/z + w) : 
  z = x*y / (x + y - w*x*y) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1329_132900


namespace NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l1329_132915

/-- Given an angle α = 2012°, this theorem states that the smallest positive angle 
    with the same terminal side as α is 212°. -/
theorem smallest_positive_angle_same_terminal_side (α : Real) : 
  α = 2012 → ∃ (θ : Real), θ = 212 ∧ 
  θ > 0 ∧ 
  θ < 360 ∧
  ∃ (k : ℤ), α = θ + 360 * k := by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l1329_132915


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1329_132946

theorem condition_necessary_not_sufficient :
  (∀ a b : ℝ, a + b ≠ 3 → (a ≠ 1 ∨ b ≠ 2)) ∧
  (∃ a b : ℝ, (a ≠ 1 ∨ b ≠ 2) ∧ a + b = 3) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l1329_132946


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1329_132995

/-- Given a parabola y = ax^2 + bx + c with vertex at (h, h) and y-intercept at (0, -2h),
    where h ≠ 0, prove that b = 6 -/
theorem parabola_coefficient (a b c h : ℝ) : 
  h ≠ 0 →
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + h) →
  a * h^2 + h = -2 * h →
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1329_132995


namespace NUMINAMATH_CALUDE_buses_passed_count_l1329_132967

/-- Represents the frequency of bus departures in hours -/
structure BusSchedule where
  austin_to_san_antonio : ℕ
  san_antonio_to_austin : ℕ

/-- Represents the journey details -/
structure JourneyDetails where
  trip_duration : ℕ
  same_highway : Bool

/-- Calculates the number of buses passed during the journey -/
def buses_passed (schedule : BusSchedule) (journey : JourneyDetails) : ℕ :=
  sorry

theorem buses_passed_count 
  (schedule : BusSchedule)
  (journey : JourneyDetails)
  (h1 : schedule.austin_to_san_antonio = 2)
  (h2 : schedule.san_antonio_to_austin = 3)
  (h3 : journey.trip_duration = 8)
  (h4 : journey.same_highway = true) :
  buses_passed schedule journey = 4 :=
sorry

end NUMINAMATH_CALUDE_buses_passed_count_l1329_132967


namespace NUMINAMATH_CALUDE_good_number_characterization_l1329_132937

def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), Function.Bijective a ∧
    ∀ k : Fin n, ∃ m : ℕ, (k.val + 1 + (a k).val + 1 : ℕ) = m ^ 2

def notGoodNumbers : Set ℕ := {1, 2, 4, 6, 7, 9, 11}

theorem good_number_characterization (n : ℕ) :
  n ≠ 0 → (isGoodNumber n ↔ n ∉ notGoodNumbers) :=
by sorry

end NUMINAMATH_CALUDE_good_number_characterization_l1329_132937


namespace NUMINAMATH_CALUDE_mango_count_proof_l1329_132997

/-- Calculates the total number of mangoes in multiple boxes -/
def total_mangoes (mangoes_per_dozen : ℕ) (dozens_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  mangoes_per_dozen * dozens_per_box * num_boxes

/-- Proves that 36 boxes of 10 dozen mangoes each contain 4,320 mangoes in total -/
theorem mango_count_proof : total_mangoes 12 10 36 = 4320 := by
  sorry

#eval total_mangoes 12 10 36

end NUMINAMATH_CALUDE_mango_count_proof_l1329_132997


namespace NUMINAMATH_CALUDE_only_five_regular_polyhedra_five_platonic_solids_l1329_132988

/-- A regular polyhedron with n-gon faces and m faces meeting at each vertex -/
structure RegularPolyhedron where
  n : ℕ  -- number of sides of each face
  m : ℕ  -- number of faces meeting at each vertex
  n_ge_3 : n ≥ 3
  m_ge_3 : m ≥ 3

/-- The set of all possible (m, n) pairs for regular polyhedra -/
def valid_regular_polyhedra : Set (ℕ × ℕ) :=
  {(3, 3), (3, 4), (4, 3), (3, 5), (5, 3)}

/-- Theorem stating that only five regular polyhedra exist -/
theorem only_five_regular_polyhedra :
  ∀ p : RegularPolyhedron, (p.m, p.n) ∈ valid_regular_polyhedra := by
  sorry

/-- Corollary: There are exactly five types of regular polyhedra -/
theorem five_platonic_solids :
  ∃! (s : Set (ℕ × ℕ)), s = valid_regular_polyhedra ∧ (∀ p : RegularPolyhedron, (p.m, p.n) ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_only_five_regular_polyhedra_five_platonic_solids_l1329_132988


namespace NUMINAMATH_CALUDE_rainbow_bead_arrangement_probability_l1329_132941

def num_beads : ℕ := 7

def num_permutations (n : ℕ) : ℕ := Nat.factorial n

def probability_specific_arrangement (n : ℕ) : ℚ :=
  1 / (num_permutations n)

theorem rainbow_bead_arrangement_probability :
  probability_specific_arrangement num_beads = 1 / 5040 := by
  sorry

end NUMINAMATH_CALUDE_rainbow_bead_arrangement_probability_l1329_132941


namespace NUMINAMATH_CALUDE_salary_problem_l1329_132996

theorem salary_problem (a b : ℝ) 
  (h1 : a + b = 3000)
  (h2 : a * 0.05 = b * 0.15) : 
  a = 2250 := by
sorry

end NUMINAMATH_CALUDE_salary_problem_l1329_132996


namespace NUMINAMATH_CALUDE_angle_sum_l1329_132926

theorem angle_sum (θ φ : Real) (h1 : 4 * (Real.cos θ)^2 + 3 * (Real.cos φ)^2 = 1)
  (h2 : 4 * Real.cos (2 * θ) + 3 * Real.sin (2 * φ) = 0)
  (h3 : 0 < θ ∧ θ < Real.pi / 2) (h4 : 0 < φ ∧ φ < Real.pi / 2) :
  θ + 3 * φ = Real.pi / 2 :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_l1329_132926


namespace NUMINAMATH_CALUDE_decagon_diagonals_l1329_132993

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals :
  num_diagonals decagon_sides = 35 := by sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l1329_132993


namespace NUMINAMATH_CALUDE_larger_number_problem_l1329_132963

theorem larger_number_problem (L S : ℕ) (hL : L > S) :
  L - S = 1390 → L = 6 * S + 15 → L = 1665 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1329_132963


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l1329_132905

theorem perpendicular_vectors_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -1]
  let b : Fin 2 → ℝ := ![x, 2]
  (a 0 * b 0 + a 1 * b 1 = 0) →  -- perpendicular condition
  ‖a + 2 • b‖ = Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l1329_132905


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l1329_132965

/-- Two lines are parallel if their slopes are equal or if they are both vertical -/
def parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 = 0 ∧ a2 = 0) ∨ (a1 ≠ 0 ∧ a2 ≠ 0 ∧ b1 / a1 = b2 / a2)

/-- The statement of the problem -/
theorem parallel_lines_k_value (k : ℝ) :
  parallel (k - 3) (4 - k) 1 (k - 3) (-1) 1 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l1329_132965


namespace NUMINAMATH_CALUDE_inequality_proof_l1329_132977

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1329_132977


namespace NUMINAMATH_CALUDE_sams_friend_points_l1329_132990

theorem sams_friend_points (sam_points friend_points total_points : ℕ) :
  sam_points = 75 →
  total_points = 87 →
  total_points = sam_points + friend_points →
  friend_points = 12 := by
sorry

end NUMINAMATH_CALUDE_sams_friend_points_l1329_132990


namespace NUMINAMATH_CALUDE_average_price_of_rackets_l1329_132933

/-- The average price of a pair of rackets given total sales and number of pairs sold -/
theorem average_price_of_rackets (total_sales : ℝ) (num_pairs : ℕ) (h1 : total_sales = 490) (h2 : num_pairs = 50) :
  total_sales / num_pairs = 9.80 := by
  sorry

end NUMINAMATH_CALUDE_average_price_of_rackets_l1329_132933


namespace NUMINAMATH_CALUDE_selene_and_tanya_spending_l1329_132920

/-- Represents the prices of items in the school canteen -/
structure CanteenPrices where
  sandwich : ℕ
  hamburger : ℕ
  hotdog : ℕ
  fruitJuice : ℕ

/-- Represents Selene's purchase -/
structure SelenePurchase where
  sandwiches : ℕ
  fruitJuice : ℕ

/-- Represents Tanya's purchase -/
structure TanyaPurchase where
  hamburgers : ℕ
  fruitJuice : ℕ

/-- Calculates the total spending of Selene and Tanya -/
def totalSpending (prices : CanteenPrices) (selene : SelenePurchase) (tanya : TanyaPurchase) : ℕ :=
  prices.sandwich * selene.sandwiches + prices.fruitJuice * selene.fruitJuice +
  prices.hamburger * tanya.hamburgers + prices.fruitJuice * tanya.fruitJuice

/-- Theorem stating that Selene and Tanya spend $16 in total -/
theorem selene_and_tanya_spending :
  ∀ (prices : CanteenPrices) (selene : SelenePurchase) (tanya : TanyaPurchase),
    prices.sandwich = 2 →
    prices.hamburger = 2 →
    prices.hotdog = 1 →
    prices.fruitJuice = 2 →
    selene.sandwiches = 3 →
    selene.fruitJuice = 1 →
    tanya.hamburgers = 2 →
    tanya.fruitJuice = 2 →
    totalSpending prices selene tanya = 16 := by
  sorry

end NUMINAMATH_CALUDE_selene_and_tanya_spending_l1329_132920


namespace NUMINAMATH_CALUDE_remaining_distance_to_cave_end_l1329_132981

theorem remaining_distance_to_cave_end (total_depth : ℕ) (traveled_distance : ℕ) 
  (h1 : total_depth = 1218)
  (h2 : traveled_distance = 849) :
  total_depth - traveled_distance = 369 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_to_cave_end_l1329_132981


namespace NUMINAMATH_CALUDE_division_problem_l1329_132966

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 725 →
  divisor = 36 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  quotient = 20 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1329_132966


namespace NUMINAMATH_CALUDE_triangle_area_arithmetic_angles_l1329_132994

/-- Given a triangle ABC where angles A, B, and C form an arithmetic sequence,
    and sides a = 1 and b = √3, the area of the triangle is √3/2. -/
theorem triangle_area_arithmetic_angles (A B C : ℝ) (a b c : ℝ) : 
  A + C = 2 * B → -- angles form arithmetic sequence
  A + B + C = π → -- sum of angles in a triangle
  a = 1 → -- given side length
  b = Real.sqrt 3 → -- given side length
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_arithmetic_angles_l1329_132994


namespace NUMINAMATH_CALUDE_teddy_bear_cost_teddy_bear_cost_proof_l1329_132902

theorem teddy_bear_cost (initial_toys : ℕ) (initial_toy_cost : ℕ) 
  (teddy_bears : ℕ) (total_cost : ℕ) : ℕ :=
  let remaining_cost := total_cost - initial_toys * initial_toy_cost
  remaining_cost / teddy_bears

theorem teddy_bear_cost_proof :
  teddy_bear_cost 28 10 20 580 = 15 := by
  sorry

end NUMINAMATH_CALUDE_teddy_bear_cost_teddy_bear_cost_proof_l1329_132902


namespace NUMINAMATH_CALUDE_adult_office_visit_cost_l1329_132906

/-- Represents the cost of an adult's office visit -/
def adult_cost : ℝ := sorry

/-- Represents the number of adult patients seen per hour -/
def adults_per_hour : ℕ := 4

/-- Represents the number of child patients seen per hour -/
def children_per_hour : ℕ := 3

/-- Represents the cost of a child's office visit -/
def child_cost : ℝ := 25

/-- Represents the number of hours worked in a day -/
def hours_per_day : ℕ := 8

/-- Represents the total income for a day -/
def daily_income : ℝ := 2200

theorem adult_office_visit_cost :
  adult_cost * (adults_per_hour * hours_per_day : ℝ) +
  child_cost * (children_per_hour * hours_per_day : ℝ) =
  daily_income ∧ adult_cost = 50 := by sorry

end NUMINAMATH_CALUDE_adult_office_visit_cost_l1329_132906


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l1329_132992

/-- Represents a point on the grid --/
structure Point where
  x : Nat
  y : Nat

/-- Represents the grid and its shaded squares --/
structure Grid where
  size : Nat
  shaded : List Point

/-- Checks if a grid has horizontal, vertical, and diagonal symmetry --/
def hasSymmetry (g : Grid) : Bool := sorry

/-- Counts the number of additional squares needed for symmetry --/
def additionalSquaresForSymmetry (g : Grid) : Nat := sorry

/-- The initial grid configuration --/
def initialGrid : Grid := {
  size := 6,
  shaded := [⟨2, 5⟩, ⟨3, 3⟩, ⟨4, 2⟩, ⟨6, 1⟩]
}

theorem min_additional_squares_for_symmetry :
  additionalSquaresForSymmetry initialGrid = 9 := by sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l1329_132992


namespace NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l1329_132999

theorem divisibility_implies_multiple_of_three (n : ℕ) (h1 : n ≥ 2) (h2 : n ∣ 2^n + 1) : 
  3 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_multiple_of_three_l1329_132999


namespace NUMINAMATH_CALUDE_solve_consecutive_integer_sets_l1329_132952

/-- A set of consecutive integers -/
structure ConsecutiveIntegerSet where
  start : ℤ
  size : ℕ

/-- The sum of elements in a ConsecutiveIntegerSet -/
def sum_of_set (s : ConsecutiveIntegerSet) : ℤ :=
  (s.size : ℤ) * (2 * s.start + s.size - 1) / 2

/-- The greatest element in a ConsecutiveIntegerSet -/
def greatest_element (s : ConsecutiveIntegerSet) : ℤ :=
  s.start + s.size - 1

theorem solve_consecutive_integer_sets :
  ∃ (m : ℕ) (a b : ConsecutiveIntegerSet),
    m > 0 ∧
    a.size = m ∧
    b.size = 2 * m ∧
    sum_of_set a = 2 * m ∧
    sum_of_set b = m ∧
    |greatest_element a - greatest_element b| = 99 →
    m = 201 := by
  sorry

end NUMINAMATH_CALUDE_solve_consecutive_integer_sets_l1329_132952


namespace NUMINAMATH_CALUDE_acute_triangle_properties_l1329_132974

/-- Properties of an acute triangle ABC -/
structure AcuteTriangle where
  -- Sides
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles
  A : ℝ
  B : ℝ
  C : ℝ
  -- Conditions
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  angle_sum : A + B + C = π
  side_angle_relation : Real.sqrt 3 * Real.sin C - Real.cos B = Real.cos (A - C)
  side_a : a = 2 * Real.sqrt 3
  area : 1/2 * b * c * Real.sin A = 3 * Real.sqrt 3

/-- Theorem about the properties of the specified acute triangle -/
theorem acute_triangle_properties (t : AcuteTriangle) : 
  t.A = π/3 ∧ t.b + t.c = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_acute_triangle_properties_l1329_132974


namespace NUMINAMATH_CALUDE_unique_integer_root_difference_l1329_132982

/-- Given an integer n, A = √(n² + 24), and B = √(n² - 9),
    prove that n = 5 is the only value for which A - B is an integer. -/
theorem unique_integer_root_difference (n : ℤ) : 
  (∃ m : ℤ, Real.sqrt (n^2 + 24) - Real.sqrt (n^2 - 9) = m) ↔ n = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_root_difference_l1329_132982


namespace NUMINAMATH_CALUDE_round_robin_tournament_teams_l1329_132969

/-- The number of games in a round-robin tournament -/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 36 games, there are 9 teams -/
theorem round_robin_tournament_teams :
  ∃ (n : ℕ), n > 0 ∧ num_games n = 36 → n = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_round_robin_tournament_teams_l1329_132969


namespace NUMINAMATH_CALUDE_vector_projection_l1329_132958

theorem vector_projection (a b : ℝ × ℝ) 
  (h1 : (a.1 + b.1, a.2 + b.2) • (2*a.1 - b.1, 2*a.2 - b.2) = -12)
  (h2 : Real.sqrt (a.1^2 + a.2^2) = 2)
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 4) :
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2) = -2 := by
sorry

end NUMINAMATH_CALUDE_vector_projection_l1329_132958


namespace NUMINAMATH_CALUDE_train_speed_l1329_132980

/-- The speed of a train crossing a bridge -/
theorem train_speed (train_length : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 240 ∧ 
  bridge_length = 150 ∧ 
  crossing_time = 20 →
  (train_length + bridge_length) / crossing_time * 3.6 = 70.2 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1329_132980


namespace NUMINAMATH_CALUDE_cheapest_plan_b_l1329_132903

/-- Represents the cost of a cell phone plan in cents -/
def PlanCost (flatFee minutes : ℕ) (perMinute : ℚ) : ℚ :=
  (flatFee : ℚ) * 100 + perMinute * minutes

theorem cheapest_plan_b (minutes : ℕ) : 
  (minutes ≥ 834) ↔ 
  (PlanCost 25 minutes 6 < PlanCost 0 minutes 12 ∧ 
   PlanCost 25 minutes 6 < PlanCost 0 minutes 9) :=
sorry

end NUMINAMATH_CALUDE_cheapest_plan_b_l1329_132903


namespace NUMINAMATH_CALUDE_stars_count_theorem_l1329_132961

theorem stars_count_theorem (east : ℕ) (west_percent : ℕ) : 
  east = 120 → west_percent = 473 → 
  east + (east * (west_percent : ℚ) / 100).ceil = 688 := by
sorry

end NUMINAMATH_CALUDE_stars_count_theorem_l1329_132961


namespace NUMINAMATH_CALUDE_fraction_product_equality_l1329_132975

theorem fraction_product_equality : (2 : ℚ) / 8 * (6 : ℚ) / 9 = (1 : ℚ) / 6 := by sorry

end NUMINAMATH_CALUDE_fraction_product_equality_l1329_132975


namespace NUMINAMATH_CALUDE_min_value_abc_l1329_132929

theorem min_value_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 27) :
  54 ≤ 3 * a + 6 * b + 9 * c ∧ ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ * b₀ * c₀ = 27 ∧ 3 * a₀ + 6 * b₀ + 9 * c₀ = 54 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l1329_132929


namespace NUMINAMATH_CALUDE_reflection_line_l1329_132904

-- Define a Point type
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define the triangle vertices
def P : Point := ⟨-2, 3⟩
def Q : Point := ⟨3, 7⟩
def R : Point := ⟨5, 1⟩

-- Define the reflected triangle vertices
def P' : Point := ⟨-6, 3⟩
def Q' : Point := ⟨-9, 7⟩
def R' : Point := ⟨-11, 1⟩

-- Define the line of reflection
def line_of_reflection (x : ℝ) : Prop :=
  (P.x + P'.x) / 2 = x ∧
  (Q.x + Q'.x) / 2 = x ∧
  (R.x + R'.x) / 2 = x

-- Theorem statement
theorem reflection_line : line_of_reflection (-3) := by
  sorry

end NUMINAMATH_CALUDE_reflection_line_l1329_132904


namespace NUMINAMATH_CALUDE_crates_delivered_is_twelve_l1329_132983

/-- The number of crates of apples delivered to a factory --/
def crates_delivered (apples_per_crate : ℕ) (rotten_apples : ℕ) (apples_per_box : ℕ) (boxes_filled : ℕ) : ℕ :=
  (boxes_filled * apples_per_box + rotten_apples) / apples_per_crate

/-- Theorem stating that the number of crates delivered is 12 --/
theorem crates_delivered_is_twelve :
  crates_delivered 42 4 10 50 = 12 := by
  sorry

end NUMINAMATH_CALUDE_crates_delivered_is_twelve_l1329_132983


namespace NUMINAMATH_CALUDE_sum_of_constants_l1329_132910

theorem sum_of_constants (a b : ℝ) : 
  (∀ x : ℝ, (x - a) / (x + b) = (x^2 - 50*x + 621) / (x^2 + 75*x - 3400)) → 
  a + b = 112 := by
sorry

end NUMINAMATH_CALUDE_sum_of_constants_l1329_132910


namespace NUMINAMATH_CALUDE_area_bounded_by_curves_l1329_132989

/-- The area between the parabola y = x^2 - x and the line y = mx from x = 0 to their intersection point. -/
def area_under_curve (m : ℤ) : ℚ :=
  (m + 1)^3 / 6

/-- The theorem statement -/
theorem area_bounded_by_curves (m n : ℤ) (h1 : m > n) (h2 : n > 0) :
  area_under_curve m - area_under_curve n = 37 / 6 → m = 3 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_area_bounded_by_curves_l1329_132989


namespace NUMINAMATH_CALUDE_team_formation_ways_l1329_132916

/-- The number of ways to choose 2 players from a group of 5 players -/
def choose_teams (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- There are 5 friends in total -/
def total_players : ℕ := 5

/-- The size of the smaller team -/
def team_size : ℕ := 2

theorem team_formation_ways :
  choose_teams total_players team_size = 10 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_ways_l1329_132916


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1329_132945

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - 2*x₁ - 8 = 0) ∧ 
  (x₂^2 - 2*x₂ - 8 = 0) ∧ 
  x₁ = 4 ∧ 
  x₂ = -2 :=
by
  sorry

#check quadratic_equation_solution

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1329_132945


namespace NUMINAMATH_CALUDE_log_powers_sum_l1329_132957

theorem log_powers_sum (a b : ℝ) (ha : a = Real.log 25) (hb : b = Real.log 49) :
  (5 : ℝ) ^ (a / b) + (7 : ℝ) ^ (b / a) = 12 := by
  sorry

end NUMINAMATH_CALUDE_log_powers_sum_l1329_132957


namespace NUMINAMATH_CALUDE_koschei_coins_l1329_132908

theorem koschei_coins : ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 10 = 7 ∧ n % 12 = 9 ∧ n = 357 := by
  sorry

end NUMINAMATH_CALUDE_koschei_coins_l1329_132908


namespace NUMINAMATH_CALUDE_function_properties_l1329_132912

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.sqrt 3 * sin (ω * x) + cos (ω * x + π / 3) + cos (ω * x - π / 3) - 1

noncomputable def g (x : ℝ) : ℝ :=
  2 * sin (2 * x - π / 6) - 1

theorem function_properties (ω : ℝ) (h_ω : ω > 0) :
  (∀ x, f ω x = 2 * sin (2 * x + π / 6) - 1) ∧
  (∀ x, g x = 2 * sin (2 * x - π / 6) - 1) ∧
  (Set.Icc 0 (π / 2)).image g = Set.Icc (-2) 1 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1329_132912


namespace NUMINAMATH_CALUDE_inheritance_sum_l1329_132935

theorem inheritance_sum (n : ℕ) : n = 36 → (n * (n + 1)) / 2 = 666 := by
  sorry

end NUMINAMATH_CALUDE_inheritance_sum_l1329_132935


namespace NUMINAMATH_CALUDE_shelf_books_count_l1329_132985

/-- The number of books on a shelf after adding more books -/
def total_books (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of books on the shelf is 48 -/
theorem shelf_books_count : total_books 38 10 = 48 := by
  sorry

end NUMINAMATH_CALUDE_shelf_books_count_l1329_132985


namespace NUMINAMATH_CALUDE_abs_diff_eq_one_point_one_l1329_132938

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ :=
  x - Int.floor x

/-- Theorem: Given the conditions, |x - y| = 1.1 -/
theorem abs_diff_eq_one_point_one (x y : ℝ) 
  (h1 : floor x + frac y = 3.7)
  (h2 : frac x + floor y = 4.6) : 
  |x - y| = 1.1 := by
sorry

end NUMINAMATH_CALUDE_abs_diff_eq_one_point_one_l1329_132938


namespace NUMINAMATH_CALUDE_max_value_of_f_l1329_132942

open Real

theorem max_value_of_f (φ : ℝ) :
  (⨆ x, cos (x + 2*φ) + 2*sin φ * sin (x + φ)) = 1 := by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1329_132942


namespace NUMINAMATH_CALUDE_ad_bc_ratio_l1329_132953

-- Define the triangle ABC
structure EquilateralTriangle :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define the triangle BCD
structure IsoscelesTriangle :=
  (side : ℝ)
  (angle : ℝ)
  (side_positive : side > 0)
  (angle_value : angle = 2 * Real.pi / 3)  -- 120° in radians

-- Define the configuration
structure TriangleConfiguration :=
  (abc : EquilateralTriangle)
  (bcd : IsoscelesTriangle)
  (shared_side : abc.side = bcd.side)

-- State the theorem
theorem ad_bc_ratio (config : TriangleConfiguration) :
  ∃ (ad bc : ℝ), ad / bc = 1 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ad_bc_ratio_l1329_132953


namespace NUMINAMATH_CALUDE_fractional_equation_solution_range_l1329_132978

theorem fractional_equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ (m / (x - 2) + 1 = x / (2 - x))) → 
  (m ≤ 2 ∧ m ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_range_l1329_132978


namespace NUMINAMATH_CALUDE_positive_real_inequality_l1329_132918

theorem positive_real_inequality (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_prod : a * b * c * d = 1) : 
  a^5 + b^5 + c^5 + d^5 ≥ a + b + c + d := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequality_l1329_132918


namespace NUMINAMATH_CALUDE_exitCell_l1329_132948

/-- Represents a cell on the 4x4 grid --/
structure Cell :=
  (row : Fin 4)
  (col : Fin 4)

/-- Represents the four possible directions --/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- The state of the game at any point --/
structure GameState :=
  (position : Cell)
  (arrows : Cell → Direction)

/-- Applies a single move to the game state --/
def move (state : GameState) : GameState :=
  sorry

/-- Checks if a cell is on the boundary of the grid --/
def isBoundary (cell : Cell) : Bool :=
  sorry

/-- Plays the game until the piece exits the grid --/
def playUntilExit (initialState : GameState) : Cell :=
  sorry

theorem exitCell :
  let initialArrows : Cell → Direction := sorry
  let initialState : GameState := {
    position := ⟨2, 1⟩,  -- C2 in 0-based indexing
    arrows := initialArrows
  }
  playUntilExit initialState = ⟨0, 1⟩  -- A2 in 0-based indexing
:= by sorry

end NUMINAMATH_CALUDE_exitCell_l1329_132948


namespace NUMINAMATH_CALUDE_divisibility_conditions_l1329_132913

theorem divisibility_conditions :
  (∀ n : ℕ, n ≥ 1 → (n ∣ 2^n - 1) → n = 1) ∧
  (∀ n : ℕ, n ≥ 1 → Odd n → (n ∣ 3^n + 1) → n = 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_conditions_l1329_132913


namespace NUMINAMATH_CALUDE_average_age_of_extreme_new_employees_is_30_l1329_132972

/-- Represents a company with employees and their ages -/
structure Company where
  initialEmployees : ℕ
  group1Size : ℕ
  group1AvgAge : ℕ
  group2Size : ℕ
  group2AvgAge : ℕ
  group3Size : ℕ
  group3AvgAge : ℕ
  newEmployees : ℕ
  newEmployeesTotalAge : ℕ
  ageDifference : ℕ

/-- Calculates the average age of the youngest and oldest new employees -/
def averageAgeOfExtremeNewEmployees (c : Company) : ℚ :=
  let totalAge := c.group1Size * c.group1AvgAge + c.group2Size * c.group2AvgAge + c.group3Size * c.group3AvgAge
  let totalEmployees := c.initialEmployees + c.newEmployees
  let x := (c.newEmployeesTotalAge - (c.newEmployees - 1) * (c.ageDifference / 2)) / c.newEmployees
  (x + x + c.ageDifference) / 2

/-- Theorem stating that for the given company configuration, 
    the average age of the youngest and oldest new employees is 30 -/
theorem average_age_of_extreme_new_employees_is_30 :
  let c : Company := {
    initialEmployees := 50,
    group1Size := 20,
    group1AvgAge := 30,
    group2Size := 20,
    group2AvgAge := 40,
    group3Size := 10,
    group3AvgAge := 50,
    newEmployees := 5,
    newEmployeesTotalAge := 150,
    ageDifference := 20
  }
  averageAgeOfExtremeNewEmployees c = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_extreme_new_employees_is_30_l1329_132972


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1329_132954

theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  let e := (2 * Real.sqrt 3) / 3
  let line_distance := Real.sqrt 3 / 2
  let eccentricity_eq := e^2 = 1 + b^2 / a^2
  let distance_eq := (a * b)^2 / (a^2 + b^2) = line_distance^2
  eccentricity_eq ∧ distance_eq →
  a^2 = 3 ∧ b^2 = 1 :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l1329_132954


namespace NUMINAMATH_CALUDE_inverse_of_f_l1329_132911

noncomputable def f (x : ℝ) := Real.log x + 1

theorem inverse_of_f (x : ℝ) :
  x > 0 → f (Real.exp (x - 1)) = x ∧ Real.exp (f x - 1) = x := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_f_l1329_132911


namespace NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l1329_132964

theorem largest_of_seven_consecutive_integers (a : ℕ) 
  (h1 : a > 0)
  (h2 : (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) = 77) :
  a + 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_seven_consecutive_integers_l1329_132964


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1329_132923

theorem cubic_equation_solution :
  ∃ (x : ℝ), x + x^3 = 10 ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1329_132923


namespace NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_6_over_sqrt_2_between_1_and_2_l1329_132932

theorem sqrt_18_minus_sqrt_6_over_sqrt_2_between_1_and_2 :
  1 < (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 ∧
  (Real.sqrt 18 - Real.sqrt 6) / Real.sqrt 2 < 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_minus_sqrt_6_over_sqrt_2_between_1_and_2_l1329_132932


namespace NUMINAMATH_CALUDE_count_positive_area_triangles_l1329_132951

/-- A point in the 4x4 grid -/
structure GridPoint where
  x : Fin 4
  y : Fin 4

/-- A triangle formed by three points in the grid -/
structure GridTriangle where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint

/-- Predicate to check if three points are collinear -/
def collinear (p1 p2 p3 : GridPoint) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Predicate to check if a triangle has positive area -/
def positiveArea (t : GridTriangle) : Prop :=
  ¬collinear t.p1 t.p2 t.p3

/-- The set of all triangles with positive area in the 4x4 grid -/
def PositiveAreaTriangles : Finset GridTriangle :=
  sorry

theorem count_positive_area_triangles :
  Finset.card PositiveAreaTriangles = 520 :=
sorry

end NUMINAMATH_CALUDE_count_positive_area_triangles_l1329_132951


namespace NUMINAMATH_CALUDE_markeesha_cracker_sales_l1329_132947

theorem markeesha_cracker_sales :
  ∀ (friday saturday sunday : ℕ),
    friday = 30 →
    saturday = 2 * friday →
    friday + saturday + sunday = 135 →
    saturday - sunday = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_markeesha_cracker_sales_l1329_132947


namespace NUMINAMATH_CALUDE_janes_number_exists_and_unique_l1329_132955

theorem janes_number_exists_and_unique :
  ∃! n : ℕ,
    200 ∣ n ∧
    45 ∣ n ∧
    500 < n ∧
    n < 2500 ∧
    Even n :=
by
  sorry

end NUMINAMATH_CALUDE_janes_number_exists_and_unique_l1329_132955


namespace NUMINAMATH_CALUDE_exterior_angle_is_60_l1329_132925

/-- An isosceles triangle with one angle opposite an equal side being 30 degrees -/
structure IsoscelesTriangle30 where
  /-- The measure of one of the angles opposite an equal side -/
  angle_opposite_equal_side : ℝ
  /-- The measure of the largest angle in the triangle -/
  largest_angle : ℝ
  /-- The fact that the triangle is isosceles with one angle being 30 degrees -/
  is_isosceles_30 : angle_opposite_equal_side = 30

/-- The measure of the exterior angle adjacent to the largest angle in the triangle -/
def exterior_angle (t : IsoscelesTriangle30) : ℝ := 180 - t.largest_angle

/-- Theorem: The measure of the exterior angle adjacent to the largest angle is 60 degrees -/
theorem exterior_angle_is_60 (t : IsoscelesTriangle30) : exterior_angle t = 60 := by
  sorry

end NUMINAMATH_CALUDE_exterior_angle_is_60_l1329_132925


namespace NUMINAMATH_CALUDE_hiking_rate_ratio_l1329_132928

/-- Prove the ratio of hiking rates for a mountain trip -/
theorem hiking_rate_ratio 
  (rate_up : ℝ) 
  (time_up : ℝ) 
  (distance_down : ℝ) 
  (rate_up_is_4 : rate_up = 4)
  (time_up_is_2 : time_up = 2)
  (distance_down_is_12 : distance_down = 12)
  (time_equal : time_up = distance_down / (distance_down / time_up * rate_up)) :
  distance_down / (time_up * rate_up) / rate_up = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_hiking_rate_ratio_l1329_132928


namespace NUMINAMATH_CALUDE_roots_of_equation_l1329_132917

def f (x : ℝ) : ℝ := x * (x - 3)^2 * (5 + x) * (x^2 - 1)

theorem roots_of_equation :
  {x : ℝ | f x = 0} = {0, 3, -5, 1, -1} := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1329_132917


namespace NUMINAMATH_CALUDE_share_of_y_l1329_132998

theorem share_of_y (total : ℝ) (x y z : ℝ) : 
  total = 273 →
  y = (45/100) * x →
  z = (50/100) * x →
  total = x + y + z →
  y = 63 := by
sorry

end NUMINAMATH_CALUDE_share_of_y_l1329_132998


namespace NUMINAMATH_CALUDE_solution_triples_l1329_132984

theorem solution_triples (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a + b + c = 1/a + 1/b + 1/c) ∧ (a^2 + b^2 + c^2 = 1/a^2 + 1/b^2 + 1/c^2) →
  ((a = 1 ∧ c = 1/b) ∨ (b = 1/a ∧ c = 1) ∨ (b = 1 ∧ c = 1/a) ∨
   (a = -1 ∧ c = 1/b) ∨ (b = -1 ∧ c = 1/a) ∨ (b = 1/a ∧ c = -1)) :=
by sorry

end NUMINAMATH_CALUDE_solution_triples_l1329_132984


namespace NUMINAMATH_CALUDE_asymptotic_necessary_not_sufficient_l1329_132936

-- Define the hyperbola C
def hyperbola (a b x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the asymptotic line equation
def asymptotic_line (a b x y : ℝ) : Prop := y = (b/a) * x ∨ y = -(b/a) * x

-- Theorem stating that the asymptotic line is a necessary but not sufficient condition for the hyperbola
theorem asymptotic_necessary_not_sufficient (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) :
  (∀ x y, hyperbola a b x y → asymptotic_line a b x y) ∧
  (∃ x y, asymptotic_line a b x y ∧ ¬hyperbola a b x y) :=
sorry

end NUMINAMATH_CALUDE_asymptotic_necessary_not_sufficient_l1329_132936


namespace NUMINAMATH_CALUDE_garden_perimeter_l1329_132939

/-- The perimeter of a rectangular garden with length 205 m and breadth 95 m is 600 m. -/
theorem garden_perimeter : 
  ∀ (perimeter length breadth : ℕ), 
    length = 205 → 
    breadth = 95 → 
    perimeter = 2 * (length + breadth) → 
    perimeter = 600 := by
  sorry

end NUMINAMATH_CALUDE_garden_perimeter_l1329_132939


namespace NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l1329_132960

theorem square_sum_of_difference_and_product (a b : ℝ) 
  (h1 : a - b = 6) 
  (h2 : a * b = 6) : 
  a^2 + b^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_of_difference_and_product_l1329_132960


namespace NUMINAMATH_CALUDE_sqrt_point_three_six_equals_point_six_l1329_132949

theorem sqrt_point_three_six_equals_point_six : Real.sqrt 0.36 = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_point_three_six_equals_point_six_l1329_132949


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1329_132956

theorem polynomial_factorization (a : ℝ) : a^2 - a = a * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1329_132956


namespace NUMINAMATH_CALUDE_stick_cutting_l1329_132968

theorem stick_cutting (n : ℕ) : (1 : ℝ) / 2^n = 1 / 64 → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_stick_cutting_l1329_132968


namespace NUMINAMATH_CALUDE_periodic_function_proof_l1329_132940

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a * f b

theorem periodic_function_proof (f : ℝ → ℝ) (c : ℝ) 
    (h1 : FunctionalEquation f) 
    (h2 : c > 0) 
    (h3 : f (c / 2) = 0) :
    ∀ x : ℝ, f (x + 2 * c) = f x := by
  sorry

end NUMINAMATH_CALUDE_periodic_function_proof_l1329_132940


namespace NUMINAMATH_CALUDE_inequality_proof_l1329_132921

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2) / (x^2 + y*z) + 
  (y^2 + 2*z^2 + 2*x^2) / (y^2 + z*x) + 
  (z^2 + 2*x^2 + 2*y^2) / (z^2 + x*y) > 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1329_132921


namespace NUMINAMATH_CALUDE_cone_cross_section_area_l1329_132907

/-- Represents a cone with height and base radius -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents a cross-section of a cone -/
structure CrossSection where
  distanceFromCenter : ℝ

/-- Calculates the area of a cross-section passing through the vertex of a cone -/
def crossSectionArea (c : Cone) (cs : CrossSection) : ℝ :=
  sorry

theorem cone_cross_section_area 
  (c : Cone) 
  (cs : CrossSection) 
  (h1 : c.height = 20) 
  (h2 : c.baseRadius = 25) 
  (h3 : cs.distanceFromCenter = 12) : 
  crossSectionArea c cs = 500 := by
  sorry

end NUMINAMATH_CALUDE_cone_cross_section_area_l1329_132907


namespace NUMINAMATH_CALUDE_three_fourths_cubed_l1329_132930

theorem three_fourths_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end NUMINAMATH_CALUDE_three_fourths_cubed_l1329_132930


namespace NUMINAMATH_CALUDE_initial_water_amount_l1329_132934

/-- Proves that the initial amount of water in the tank was 100 L given the conditions of the rainstorm. -/
theorem initial_water_amount (flow_rate : ℝ) (duration : ℝ) (total_after : ℝ) : 
  flow_rate = 2 → duration = 90 → total_after = 280 → 
  total_after - (flow_rate * duration) = 100 := by
sorry

end NUMINAMATH_CALUDE_initial_water_amount_l1329_132934


namespace NUMINAMATH_CALUDE_evaluate_expression_l1329_132971

theorem evaluate_expression (x y z : ℝ) (hx : x = 5) (hy : y = 10) (hz : z = 3) :
  z * (y - 2 * x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1329_132971


namespace NUMINAMATH_CALUDE_cards_distribution_l1329_132909

/-- Given 60 cards dealt to 7 people as evenly as possible, 
    exactly 3 people will have fewer than 9 cards. -/
theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) (h2 : num_people = 7) :
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l1329_132909


namespace NUMINAMATH_CALUDE_solve_equation_l1329_132976

theorem solve_equation (x : ℝ) : 3 * x + 12 = (1/3) * (6 * x + 36) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1329_132976


namespace NUMINAMATH_CALUDE_tan_75_degrees_l1329_132944

theorem tan_75_degrees : Real.tan (75 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_75_degrees_l1329_132944


namespace NUMINAMATH_CALUDE_scientific_notation_3120000_l1329_132924

theorem scientific_notation_3120000 :
  3120000 = 3.12 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_3120000_l1329_132924


namespace NUMINAMATH_CALUDE_gcf_of_2550_and_7140_l1329_132931

theorem gcf_of_2550_and_7140 : Nat.gcd 2550 7140 = 510 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_2550_and_7140_l1329_132931


namespace NUMINAMATH_CALUDE_problem_statement_l1329_132986

theorem problem_statement (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : (3 : ℝ)^x = (4 : ℝ)^y) (h2 : 2 * x = a * y) : 
  a = 4 * (Real.log 2 / Real.log 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1329_132986


namespace NUMINAMATH_CALUDE_initial_oak_trees_l1329_132991

theorem initial_oak_trees (initial : ℕ) (planted : ℕ) (total : ℕ) : 
  planted = 2 → total = 11 → initial + planted = total → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_oak_trees_l1329_132991


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1329_132979

theorem inequality_equivalence (x : ℝ) : 5 * x - 12 ≤ 2 * (4 * x - 3) ↔ x ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1329_132979


namespace NUMINAMATH_CALUDE_symmetric_origin_correct_symmetric_point_correct_l1329_132973

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry with respect to the origin
def symmetricOrigin (p : Point2D) : Point2D :=
  { x := -p.x, y := -p.y }

-- Define symmetry with respect to another point
def symmetricPoint (p : Point2D) (k : Point2D) : Point2D :=
  { x := 2 * k.x - p.x, y := 2 * k.y - p.y }

-- Theorem for symmetry with respect to the origin
theorem symmetric_origin_correct (m : Point2D) :
  symmetricOrigin m = { x := -m.x, y := -m.y } := by sorry

-- Theorem for symmetry with respect to another point
theorem symmetric_point_correct (m k : Point2D) :
  symmetricPoint m k = { x := 2 * k.x - m.x, y := 2 * k.y - m.y } := by sorry

end NUMINAMATH_CALUDE_symmetric_origin_correct_symmetric_point_correct_l1329_132973


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l1329_132962

/-- The area of a square with adjacent vertices at (1,3) and (-2,7) is 25 -/
theorem square_area_from_vertices :
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (-2, 7)
  let distance := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
  let area := distance^2
  area = 25 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l1329_132962


namespace NUMINAMATH_CALUDE_only_108_117_207_satisfy_l1329_132901

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Calculates the sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- Increases a three-digit number by 3 -/
def increaseByThree (n : ThreeDigitNumber) : ThreeDigitNumber :=
  let newOnes := (n.ones + 3) % 10
  let carryTens := (n.ones + 3) / 10
  let newTens := (n.tens + carryTens) % 10
  let carryHundreds := (n.tens + carryTens) / 10
  let newHundreds := n.hundreds + carryHundreds
  ⟨newHundreds, newTens, newOnes, sorry, sorry, sorry⟩

/-- Checks if a three-digit number satisfies the condition -/
def satisfiesCondition (n : ThreeDigitNumber) : Prop :=
  digitSum (increaseByThree n) = 3 * digitSum n

/-- The main theorem stating that only 108, 117, and 207 satisfy the condition -/
theorem only_108_117_207_satisfy :
  ∀ n : ThreeDigitNumber, satisfiesCondition n ↔ 
    (n.hundreds = 1 ∧ n.tens = 0 ∧ n.ones = 8) ∨
    (n.hundreds = 1 ∧ n.tens = 1 ∧ n.ones = 7) ∨
    (n.hundreds = 2 ∧ n.tens = 0 ∧ n.ones = 7) :=
  sorry

end NUMINAMATH_CALUDE_only_108_117_207_satisfy_l1329_132901
