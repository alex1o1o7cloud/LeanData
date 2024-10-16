import Mathlib

namespace NUMINAMATH_CALUDE_circle_radius_from_area_l3449_344939

/-- Given a circle with area 49π, prove its radius is 7 -/
theorem circle_radius_from_area (A : ℝ) (r : ℝ) : 
  A = 49 * Real.pi → A = Real.pi * r^2 → r = 7 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l3449_344939


namespace NUMINAMATH_CALUDE_banana_permutations_eq_60_l3449_344906

/-- The number of letters in the word "BANANA" -/
def total_letters : Nat := 6

/-- The number of occurrences of 'A' in "BANANA" -/
def count_A : Nat := 3

/-- The number of occurrences of 'N' in "BANANA" -/
def count_N : Nat := 2

/-- The number of occurrences of 'B' in "BANANA" -/
def count_B : Nat := 1

/-- The number of distinct permutations of the letters in "BANANA" -/
def banana_permutations : Nat := Nat.factorial total_letters / (Nat.factorial count_A * Nat.factorial count_N)

theorem banana_permutations_eq_60 : banana_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_permutations_eq_60_l3449_344906


namespace NUMINAMATH_CALUDE_time_is_48_seconds_l3449_344995

-- Define the constants
def building_diameter : ℝ := 150
def path_distance : ℝ := 300
def kenny_speed : ℝ := 4
def jenny_speed : ℝ := 2
def initial_distance : ℝ := 300

-- Define the function to calculate the time
def time_to_see_again (building_diameter path_distance kenny_speed jenny_speed initial_distance : ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem time_is_48_seconds :
  time_to_see_again building_diameter path_distance kenny_speed jenny_speed initial_distance = 48 :=
sorry

end NUMINAMATH_CALUDE_time_is_48_seconds_l3449_344995


namespace NUMINAMATH_CALUDE_right_triangle_common_factor_l3449_344971

theorem right_triangle_common_factor (d : ℝ) (h_pos : d > 0) : 
  (2 * d = 45 ∨ 4 * d = 45 ∨ 5 * d = 45) ∧ 
  (2 * d)^2 + (4 * d)^2 = (5 * d)^2 → 
  d = 9 := by sorry

end NUMINAMATH_CALUDE_right_triangle_common_factor_l3449_344971


namespace NUMINAMATH_CALUDE_class_size_problem_l3449_344931

/-- Given a class where:
    - The average mark of all students is 80
    - If 5 students with an average mark of 60 are excluded, the remaining students' average is 90
    Prove that the total number of students is 15 -/
theorem class_size_problem (total_average : ℝ) (excluded_average : ℝ) (remaining_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 60) 
  (h3 : remaining_average = 90) (h4 : excluded_count = 5) : 
  ∃ (N : ℕ), N = 15 ∧ 
  N * total_average = (N - excluded_count) * remaining_average + excluded_count * excluded_average :=
sorry

end NUMINAMATH_CALUDE_class_size_problem_l3449_344931


namespace NUMINAMATH_CALUDE_three_propositions_l3449_344964

theorem three_propositions :
  (∀ a b : ℝ, |a - b| < 1 → |a| < |b| + 1) ∧
  (∀ a b : ℝ, |a + b| - 2*|a| ≤ |a - b|) ∧
  (∀ x y : ℝ, |x| < 2 ∧ |y| > 3 → |x / y| < 2/3) := by
  sorry

end NUMINAMATH_CALUDE_three_propositions_l3449_344964


namespace NUMINAMATH_CALUDE_mary_sugar_needed_l3449_344928

/-- Given a recipe that requires a certain amount of sugar and an amount already added,
    calculate the remaining amount of sugar needed. -/
def sugar_needed (recipe_requirement : ℕ) (already_added : ℕ) : ℕ :=
  recipe_requirement - already_added

/-- Prove that Mary needs to add 3 more cups of sugar. -/
theorem mary_sugar_needed : sugar_needed 7 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mary_sugar_needed_l3449_344928


namespace NUMINAMATH_CALUDE_candy_average_l3449_344980

theorem candy_average (eunji_candies : ℕ) (jimin_diff : ℕ) (jihyun_diff : ℕ) : 
  eunji_candies = 35 →
  jimin_diff = 6 →
  jihyun_diff = 3 →
  (eunji_candies + (eunji_candies + jimin_diff) + (eunji_candies - jihyun_diff)) / 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_candy_average_l3449_344980


namespace NUMINAMATH_CALUDE_gcd_15378_21333_48906_l3449_344938

theorem gcd_15378_21333_48906 : Nat.gcd 15378 (Nat.gcd 21333 48906) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_15378_21333_48906_l3449_344938


namespace NUMINAMATH_CALUDE_roses_picked_l3449_344986

theorem roses_picked (initial : ℕ) (sold : ℕ) (final : ℕ) : initial = 37 → sold = 16 → final = 40 → final - (initial - sold) = 19 := by
  sorry

end NUMINAMATH_CALUDE_roses_picked_l3449_344986


namespace NUMINAMATH_CALUDE_not_necessary_condition_l3449_344925

theorem not_necessary_condition : ¬(∀ x y : ℝ, x * y = 0 → x^2 + y^2 = 0) := by sorry

end NUMINAMATH_CALUDE_not_necessary_condition_l3449_344925


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l3449_344940

theorem semicircle_area_with_inscribed_rectangle (r : ℝ) : 
  r > 0 → 
  (∃ (w h : ℝ), w > 0 ∧ h > 0 ∧ w = 1 ∧ h = 3 ∧ h = r) → 
  (π * r^2) / 2 = 9 * π / 2 := by
sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l3449_344940


namespace NUMINAMATH_CALUDE_expression_evaluation_l3449_344994

theorem expression_evaluation : (3^2 - 3 + 1) - (4^2 - 4 + 1) + (5^2 - 5 + 1) - (6^2 - 6 + 1) = -16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3449_344994


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3449_344992

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the third term is 20 and the sixth term is 26,
    prove that the ninth term is 32. -/
theorem arithmetic_sequence_ninth_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_third_term : a 3 = 20)
  (h_sixth_term : a 6 = 26) :
  a 9 = 32 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3449_344992


namespace NUMINAMATH_CALUDE_largest_n_value_l3449_344912

/-- A function that checks if for any group of at least 145 candies,
    there is a type of candy which appears exactly 10 times -/
def has_type_with_10_occurrences (candies : List Nat) : Prop :=
  ∀ (group : List Nat), group.length ≥ 145 → group ⊆ candies →
    ∃ (type : Nat), (group.filter (· = type)).length = 10

/-- The theorem stating the largest possible value of n -/
theorem largest_n_value :
  ∀ (n : Nat),
    n > 145 →
    (∀ (candies : List Nat), candies.length = n →
      has_type_with_10_occurrences candies) →
    n ≤ 160 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_value_l3449_344912


namespace NUMINAMATH_CALUDE_toms_weekly_income_l3449_344958

/-- Tom's crab fishing business --/
def crab_business (num_buckets : ℕ) (crabs_per_bucket : ℕ) (price_per_crab : ℕ) (days_per_week : ℕ) : ℕ :=
  num_buckets * crabs_per_bucket * price_per_crab * days_per_week

/-- Tom's weekly income from selling crabs --/
theorem toms_weekly_income :
  crab_business 8 12 5 7 = 3360 := by
  sorry

#eval crab_business 8 12 5 7

end NUMINAMATH_CALUDE_toms_weekly_income_l3449_344958


namespace NUMINAMATH_CALUDE_half_equals_fifty_percent_l3449_344913

theorem half_equals_fifty_percent (muffin : ℝ) (h : muffin > 0) :
  (1 / 2 : ℝ) * muffin = (50 / 100 : ℝ) * muffin := by sorry

end NUMINAMATH_CALUDE_half_equals_fifty_percent_l3449_344913


namespace NUMINAMATH_CALUDE_geometric_sequence_a11_l3449_344984

/-- A geometric sequence with a_3 = 3 and a_7 = 6 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = a n * q) ∧
  a 3 = 3 ∧ 
  a 7 = 6

theorem geometric_sequence_a11 (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 11 = 12 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a11_l3449_344984


namespace NUMINAMATH_CALUDE_fifth_term_is_eight_l3449_344902

/-- Represents a geometric sequence with positive terms -/
structure GeometricSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  ratio : ∀ n, a (n + 1) = 2 * a n

/-- Theorem: In a geometric sequence with common ratio 2 and a₂a₆ = 16, a₅ = 8 -/
theorem fifth_term_is_eight (seq : GeometricSequence) 
    (h : seq.a 2 * seq.a 6 = 16) : seq.a 5 = 8 := by
  sorry

#check fifth_term_is_eight

end NUMINAMATH_CALUDE_fifth_term_is_eight_l3449_344902


namespace NUMINAMATH_CALUDE_clouddale_rainfall_2008_l3449_344904

def average_monthly_rainfall_2007 : ℝ := 45.2
def rainfall_increase_2008 : ℝ := 3.5
def months_in_year : ℕ := 12

theorem clouddale_rainfall_2008 :
  let average_monthly_rainfall_2008 := average_monthly_rainfall_2007 + rainfall_increase_2008
  let total_rainfall_2008 := average_monthly_rainfall_2008 * months_in_year
  total_rainfall_2008 = 584.4 := by
sorry

end NUMINAMATH_CALUDE_clouddale_rainfall_2008_l3449_344904


namespace NUMINAMATH_CALUDE_two_numbers_problem_l3449_344900

theorem two_numbers_problem : ∃ (A B : ℕ+), 
  A + B = 581 ∧ 
  Nat.lcm A B / Nat.gcd A B = 240 ∧ 
  ((A = 560 ∧ B = 21) ∨ (A = 21 ∧ B = 560)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_problem_l3449_344900


namespace NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l3449_344974

/-- A quadrilateral with perpendicular diagonals, an inscribed circle, and a circumscribed circle -/
structure SpecialQuadrilateral where
  /-- The quadrilateral has perpendicular diagonals -/
  perp_diagonals : Bool
  /-- A circle can be inscribed within the quadrilateral -/
  has_inscribed_circle : Bool
  /-- A circle can be circumscribed around the quadrilateral -/
  has_circumscribed_circle : Bool

/-- Definition of a square -/
def is_square (q : SpecialQuadrilateral) : Prop :=
  -- A square has all sides equal and all angles right angles
  sorry

/-- Theorem: A quadrilateral with perpendicular diagonals, an inscribed circle, 
    and a circumscribed circle is not necessarily a square -/
theorem special_quadrilateral_not_necessarily_square :
  ∃ q : SpecialQuadrilateral, q.perp_diagonals ∧ q.has_inscribed_circle ∧ q.has_circumscribed_circle ∧ ¬is_square q :=
by
  sorry


end NUMINAMATH_CALUDE_special_quadrilateral_not_necessarily_square_l3449_344974


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l3449_344920

theorem quadratic_inequality_condition (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + m > 0) ↔ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l3449_344920


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3449_344990

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Statement for the first expression
theorem simplify_expression_1 : 
  2 * Real.sqrt 3 * (1.5 : ℝ) ^ (1/3) * 12 ^ (1/6) = 6 := by sorry

-- Statement for the second expression
theorem simplify_expression_2 : 
  log10 25 + (2/3) * log10 8 + log10 5 * log10 20 + (log10 2)^2 = 3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l3449_344990


namespace NUMINAMATH_CALUDE_complement_of_A_l3449_344934

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}

theorem complement_of_A (A B : Set ℕ) 
  (h1 : A ∪ B = {1, 2, 3, 4, 5})
  (h2 : A ∩ B = {3, 4, 5}) :
  (U \ A) = {6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_l3449_344934


namespace NUMINAMATH_CALUDE_product_remainder_divisible_by_eight_l3449_344937

theorem product_remainder_divisible_by_eight :
  (1502 * 1786 * 1822 * 2026) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_divisible_by_eight_l3449_344937


namespace NUMINAMATH_CALUDE_max_value_of_a_l3449_344978

-- Define the condition function
def condition (x : ℝ) : Prop := x^2 - 2*x - 3 > 0

-- Define the theorem
theorem max_value_of_a :
  (∃ a : ℝ, ∀ x : ℝ, x < a → condition x) ∧
  (∀ a : ℝ, ∃ x : ℝ, condition x ∧ x ≥ a) →
  (∀ a : ℝ, (∀ x : ℝ, x < a → condition x) → a ≤ -1) ∧
  (∀ x : ℝ, x < -1 → condition x) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3449_344978


namespace NUMINAMATH_CALUDE_some_number_value_l3449_344948

theorem some_number_value (x : ℝ) : (45 + 23 / x) * x = 4028 → x = 89 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l3449_344948


namespace NUMINAMATH_CALUDE_spotted_animals_with_horns_l3449_344959

/-- Represents the classification of Hagrid's animals -/
structure AnimalCounts where
  total : Nat
  spotted : Nat
  striped_with_wings : Nat
  with_horns : Nat

/-- Theorem stating the number of spotted animals with horns -/
theorem spotted_animals_with_horns (h : AnimalCounts) 
  (h_total : h.total = 100)
  (h_spotted : h.spotted = 62)
  (h_striped_wings : h.striped_with_wings = 28)
  (h_horns : h.with_horns = 36)
  (h_spotted_striped : h.spotted + (h.striped_with_wings + (h.total - h.spotted - h.striped_with_wings)) = h.total)
  (h_wings_horns : h.striped_with_wings + h.with_horns + (h.total - h.striped_with_wings - h.with_horns) = h.total) :
  h.with_horns - (h.total - h.spotted - h.striped_with_wings) = 26 := by
  sorry


end NUMINAMATH_CALUDE_spotted_animals_with_horns_l3449_344959


namespace NUMINAMATH_CALUDE_fraction_simplification_l3449_344983

theorem fraction_simplification (x y : ℚ) (hx : x = 4) (hy : y = 5) :
  (1 / (x + y)) / (1 / (x - y)) = -1/9 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3449_344983


namespace NUMINAMATH_CALUDE_floor_equation_solution_l3449_344932

theorem floor_equation_solution (x : ℝ) : 
  ⌊⌊3 * x⌋ + 1/3⌋ = ⌊x + 3⌋ ↔ 4/3 ≤ x ∧ x < 5/3 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l3449_344932


namespace NUMINAMATH_CALUDE_four_digit_numbers_count_l3449_344985

theorem four_digit_numbers_count : 
  (Finset.range 4001).card = (Finset.Icc 1000 5000).card := by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_count_l3449_344985


namespace NUMINAMATH_CALUDE_students_allowance_l3449_344960

theorem students_allowance (allowance : ℚ) : 
  (3 / 5 * allowance + 1 / 3 * (2 / 5 * allowance) + 1 = allowance) → 
  allowance = 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_students_allowance_l3449_344960


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l3449_344908

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 30 ∧ x - y = 6 → x * y = 216 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l3449_344908


namespace NUMINAMATH_CALUDE_min_value_of_function_l3449_344903

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  x^2 + 6*x + 36/x^2 ≥ 31 ∧
  (x^2 + 6*x + 36/x^2 = 31 ↔ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3449_344903


namespace NUMINAMATH_CALUDE_f_minimum_f_has_root_l3449_344977

noncomputable section

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := Real.exp (x - m) - x

-- Statement for the extremum of f(x)
theorem f_minimum (m : ℝ) : 
  (∀ x : ℝ, f m x ≥ f m m) ∧ f m m = 1 - m :=
sorry

-- Statement for the existence of a root in (m, 2m) when m > 1
theorem f_has_root (m : ℝ) (h : m > 1) : 
  ∃ x : ℝ, m < x ∧ x < 2*m ∧ f m x = 0 :=
sorry

end

end NUMINAMATH_CALUDE_f_minimum_f_has_root_l3449_344977


namespace NUMINAMATH_CALUDE_gold_tetrahedron_volume_l3449_344915

/-- Represents a cube with alternately colored vertices -/
structure ColoredCube where
  sideLength : ℝ
  vertexColors : Fin 8 → Bool  -- True for gold, False for red

/-- Calculates the volume of a tetrahedron formed by selected vertices of a cube -/
def tetrahedronVolume (cube : ColoredCube) (selectVertex : Fin 8 → Bool) : ℝ :=
  sorry

/-- The main theorem stating the volume of the gold-colored tetrahedron -/
theorem gold_tetrahedron_volume (cube : ColoredCube) 
  (h1 : cube.sideLength = 8)
  (h2 : ∀ i : Fin 8, cube.vertexColors i = (i.val % 2 == 0))  -- Alternating colors
  : tetrahedronVolume cube cube.vertexColors = 170.67 := by
  sorry

end NUMINAMATH_CALUDE_gold_tetrahedron_volume_l3449_344915


namespace NUMINAMATH_CALUDE_increasing_function_a_range_l3449_344923

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 3*x

-- State the theorem
theorem increasing_function_a_range (a : ℝ) :
  (∀ x : ℝ, Monotone (f a)) → -3 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_increasing_function_a_range_l3449_344923


namespace NUMINAMATH_CALUDE_christmas_monday_implies_jan25_thursday_l3449_344918

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in a year -/
structure Date where
  month : Nat
  day : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (d : Date) : DayOfWeek :=
  sorry

/-- Function to get the date of the next year -/
def nextYearDate (d : Date) : Date :=
  sorry

theorem christmas_monday_implies_jan25_thursday
  (h : dayOfWeek ⟨12, 25⟩ = DayOfWeek.Monday) :
  dayOfWeek (nextYearDate ⟨1, 25⟩) = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_christmas_monday_implies_jan25_thursday_l3449_344918


namespace NUMINAMATH_CALUDE_simplify_expression_l3449_344961

theorem simplify_expression (x : ℝ) : (3*x)^5 + (4*x^2)*(3*x^2) = 243*x^5 + 12*x^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3449_344961


namespace NUMINAMATH_CALUDE_M_necessary_not_sufficient_for_N_l3449_344936

def M : Set ℝ := {x | |x + 1| < 4}
def N : Set ℝ := {x | x / (x - 3) < 0}

theorem M_necessary_not_sufficient_for_N :
  (∀ a : ℝ, a ∈ N → a ∈ M) ∧ (∃ b : ℝ, b ∈ M ∧ b ∉ N) := by
  sorry

end NUMINAMATH_CALUDE_M_necessary_not_sufficient_for_N_l3449_344936


namespace NUMINAMATH_CALUDE_oplus_calculation_l3449_344981

def oplus (x y : ℚ) : ℚ := 1 / (x - y) + y

theorem oplus_calculation :
  (oplus 2 (-3) = -2 - 4/5) ∧
  (oplus (oplus (-4) (-1)) (-5) = -4 - 8/11) := by
  sorry

end NUMINAMATH_CALUDE_oplus_calculation_l3449_344981


namespace NUMINAMATH_CALUDE_square_area_with_five_equal_rectangles_l3449_344989

theorem square_area_with_five_equal_rectangles (s : ℝ) (x : ℝ) (y : ℝ) : 
  s > 0 →  -- side length of square is positive
  x > 0 →  -- width of central rectangle is positive
  y > 0 →  -- height of bottom rectangle is positive
  s = 5 + 2 * y →  -- relationship between side length and rectangles
  x * (s / 2) = 5 * y →  -- equal area condition
  s^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_five_equal_rectangles_l3449_344989


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_plane_l3449_344972

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_perpendicular_plane 
  (m n : Line) (α : Plane) 
  (h1 : parallel_line_plane m α) 
  (h2 : perpendicular_line_plane n α) : 
  perpendicular_lines m n := by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_plane_l3449_344972


namespace NUMINAMATH_CALUDE_tangent_sum_identity_l3449_344909

theorem tangent_sum_identity (α β γ : Real) (h : α + β + γ = Real.pi / 2) :
  Real.tan α * Real.tan β + Real.tan β * Real.tan γ + Real.tan γ * Real.tan α = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_identity_l3449_344909


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l3449_344951

/-- The jumping contest between a grasshopper and a frog -/
theorem grasshopper_jump_distance 
  (frog_jump : ℕ) 
  (frog_grasshopper_difference : ℕ) 
  (h1 : frog_jump = 12)
  (h2 : frog_jump = frog_grasshopper_difference + grasshopper_jump) :
  grasshopper_jump = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l3449_344951


namespace NUMINAMATH_CALUDE_packs_per_box_is_40_l3449_344970

/-- Represents Meadow's diaper business --/
structure DiaperBusiness where
  boxes_per_week : ℕ
  diapers_per_pack : ℕ
  price_per_diaper : ℕ
  total_revenue : ℕ

/-- Calculates the number of packs in each box --/
def packs_per_box (business : DiaperBusiness) : ℕ :=
  (business.total_revenue / business.price_per_diaper) / 
  (business.diapers_per_pack * business.boxes_per_week)

/-- Theorem stating that the number of packs in each box is 40 --/
theorem packs_per_box_is_40 (business : DiaperBusiness) 
  (h1 : business.boxes_per_week = 30)
  (h2 : business.diapers_per_pack = 160)
  (h3 : business.price_per_diaper = 5)
  (h4 : business.total_revenue = 960000) :
  packs_per_box business = 40 := by
  sorry

end NUMINAMATH_CALUDE_packs_per_box_is_40_l3449_344970


namespace NUMINAMATH_CALUDE_sunzi_deer_problem_l3449_344943

/-- The number of deer that enter the city -/
def total_deer : ℕ := 100

/-- The number of families in the city -/
def num_families : ℕ := 75

theorem sunzi_deer_problem :
  (num_families : ℚ) + (1 / 3 : ℚ) * num_families = total_deer :=
by sorry

end NUMINAMATH_CALUDE_sunzi_deer_problem_l3449_344943


namespace NUMINAMATH_CALUDE_sum_of_even_coefficients_l3449_344997

theorem sum_of_even_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x + 1)^4 + (x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₀ + a₂ + a₄ = -8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_even_coefficients_l3449_344997


namespace NUMINAMATH_CALUDE_rulers_equation_initial_rulers_count_l3449_344911

/-- The number of rulers initially in the drawer -/
def initial_rulers : ℕ := sorry

/-- The number of rulers added to the drawer -/
def added_rulers : ℕ := 14

/-- The final number of rulers in the drawer -/
def final_rulers : ℕ := 25

/-- Theorem stating that the initial number of rulers plus the added rulers equals the final number of rulers -/
theorem rulers_equation : initial_rulers + added_rulers = final_rulers := by sorry

/-- Theorem proving that the initial number of rulers is 11 -/
theorem initial_rulers_count : initial_rulers = 11 := by sorry

end NUMINAMATH_CALUDE_rulers_equation_initial_rulers_count_l3449_344911


namespace NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l3449_344949

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (a + 2, a - 3)

-- Define the property of being in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem range_of_a_in_fourth_quadrant :
  ∀ a : ℝ, in_fourth_quadrant (P a) ↔ -2 < a ∧ a < 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_in_fourth_quadrant_l3449_344949


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3449_344982

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {y | ∃ x, y = 2^x}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = Real.log (3 - x)}

-- Theorem statement
theorem complement_M_intersect_N : 
  (U \ M) ∩ N = {y | y ≤ 0} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3449_344982


namespace NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l3449_344999

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 4

-- Define the hyperbola equation
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y - 2)^2 = 4

-- Define the tangency condition
def are_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), ellipse x y ∧ hyperbola x y m ∧
  ∀ (x' y' : ℝ), ellipse x' y' ∧ hyperbola x' y' m → (x = x' ∧ y = y')

-- Theorem statement
theorem tangent_ellipse_hyperbola :
  ∀ m : ℝ, are_tangent m → m = 1/3 := by sorry

end NUMINAMATH_CALUDE_tangent_ellipse_hyperbola_l3449_344999


namespace NUMINAMATH_CALUDE_f_2015_equals_negative_5_l3449_344953

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem f_2015_equals_negative_5
  (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) + f x = 0)
  (h2 : is_periodic f 4)
  (h3 : f 1 = 5) :
  f 2015 = -5 := by
  sorry

end NUMINAMATH_CALUDE_f_2015_equals_negative_5_l3449_344953


namespace NUMINAMATH_CALUDE_max_fertilizer_a_six_tons_achievable_l3449_344929

/-- Represents the price and quantity of fertilizers A and B --/
structure Fertilizer where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℝ
  quantity_b : ℝ

/-- The conditions of the fertilizer purchase problem --/
def fertilizer_conditions (f : Fertilizer) : Prop :=
  f.price_a = f.price_b + 100 ∧
  2 * f.price_a + f.price_b = 1700 ∧
  f.quantity_a + f.quantity_b = 10 ∧
  f.quantity_a * f.price_a + f.quantity_b * f.price_b ≤ 5600

/-- The theorem stating the maximum quantity of fertilizer A that can be purchased --/
theorem max_fertilizer_a (f : Fertilizer) :
  fertilizer_conditions f → f.quantity_a ≤ 6 := by
  sorry

/-- The theorem stating that 6 tons of fertilizer A is achievable --/
theorem six_tons_achievable :
  ∃ f : Fertilizer, fertilizer_conditions f ∧ f.quantity_a = 6 := by
  sorry

end NUMINAMATH_CALUDE_max_fertilizer_a_six_tons_achievable_l3449_344929


namespace NUMINAMATH_CALUDE_polyhedron_parity_l3449_344956

-- Define a polyhedron structure
structure Polyhedron where
  vertices : Set (ℕ × ℕ × ℕ)
  edges : Set (Set (ℕ × ℕ × ℕ))
  faces : Set (Set (ℕ × ℕ × ℕ))
  -- Add necessary conditions for a valid polyhedron

-- Function to count faces with odd number of sides
def count_odd_faces (p : Polyhedron) : ℕ := sorry

-- Function to count vertices with odd degree
def count_odd_degree_vertices (p : Polyhedron) : ℕ := sorry

-- Theorem statement
theorem polyhedron_parity (p : Polyhedron) : 
  Even (count_odd_faces p) ∧ Even (count_odd_degree_vertices p) := by sorry

end NUMINAMATH_CALUDE_polyhedron_parity_l3449_344956


namespace NUMINAMATH_CALUDE_circle_center_quadrant_l3449_344914

theorem circle_center_quadrant (α : Real) :
  (∃ x y : Real, x^2 * Real.cos α - y^2 * Real.sin α + 2 = 0) →  -- hyperbola condition
  let center := (- Real.cos α, Real.sin α)
  (center.1 < 0 ∧ center.2 > 0) ∨ (center.1 > 0 ∧ center.2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_quadrant_l3449_344914


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3449_344988

/-- The number of different math books -/
def num_math_books : ℕ := 4

/-- The number of different history books -/
def num_history_books : ℕ := 6

/-- The number of ways to arrange the books under the given conditions -/
def arrangement_count : ℕ := num_math_books * (num_math_books - 1) * Nat.factorial (num_math_books + num_history_books - 3)

theorem book_arrangement_count :
  arrangement_count = 60480 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3449_344988


namespace NUMINAMATH_CALUDE_unique_solution_floor_equation_l3449_344998

theorem unique_solution_floor_equation :
  ∃! (x : ℝ), x > 0 ∧ x * ↑(⌊x⌋) = 72 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_floor_equation_l3449_344998


namespace NUMINAMATH_CALUDE_only_vehicle_green_light_is_random_l3449_344905

-- Define the type for events
inductive Event
  | TriangleInequality
  | SunRise
  | VehicleGreenLight
  | NegativeAbsoluteValue

-- Define a predicate for random events
def isRandomEvent : Event → Prop :=
  fun e => match e with
    | Event.TriangleInequality => false
    | Event.SunRise => false
    | Event.VehicleGreenLight => true
    | Event.NegativeAbsoluteValue => false

-- Theorem statement
theorem only_vehicle_green_light_is_random :
  ∀ e : Event, isRandomEvent e ↔ e = Event.VehicleGreenLight :=
by sorry

end NUMINAMATH_CALUDE_only_vehicle_green_light_is_random_l3449_344905


namespace NUMINAMATH_CALUDE_boat_journey_distance_l3449_344901

def boat_journey (total_time : ℝ) (stream_velocity : ℝ) (boat_speed : ℝ) : Prop :=
  let downstream_speed : ℝ := boat_speed + stream_velocity
  let upstream_speed : ℝ := boat_speed - stream_velocity
  let distance : ℝ := 180
  (distance / downstream_speed + (distance / 2) / upstream_speed = total_time) ∧
  (downstream_speed > 0) ∧
  (upstream_speed > 0)

theorem boat_journey_distance :
  boat_journey 19 4 14 := by sorry

end NUMINAMATH_CALUDE_boat_journey_distance_l3449_344901


namespace NUMINAMATH_CALUDE_cost_effective_flower_purchase_l3449_344927

/-- Represents the cost-effective flower purchasing problem --/
theorem cost_effective_flower_purchase
  (total_flowers : ℕ)
  (carnation_price lily_price : ℚ)
  (h_total : total_flowers = 300)
  (h_carnation_price : carnation_price = 5)
  (h_lily_price : lily_price = 10)
  : ∃ (carnations lilies : ℕ),
    carnations + lilies = total_flowers ∧
    carnations ≤ 2 * lilies ∧
    ∀ (c l : ℕ),
      c + l = total_flowers →
      c ≤ 2 * l →
      carnation_price * carnations + lily_price * lilies ≤
      carnation_price * c + lily_price * l ∧
    carnations = 200 ∧
    lilies = 100 := by
  sorry

end NUMINAMATH_CALUDE_cost_effective_flower_purchase_l3449_344927


namespace NUMINAMATH_CALUDE_arithmetic_progression_tenth_term_zero_l3449_344907

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.a + (n - 1 : ℝ) * ap.d

theorem arithmetic_progression_tenth_term_zero
  (ap : ArithmeticProgression)
  (h : ap.nthTerm 5 + ap.nthTerm 21 = ap.nthTerm 8 + ap.nthTerm 15 + ap.nthTerm 13) :
  ap.nthTerm 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_tenth_term_zero_l3449_344907


namespace NUMINAMATH_CALUDE_cos_shift_equals_sin_shift_l3449_344963

theorem cos_shift_equals_sin_shift (x : ℝ) : 
  Real.cos (2 * x - π / 4) = Real.sin (2 * (x + π / 8)) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_equals_sin_shift_l3449_344963


namespace NUMINAMATH_CALUDE_event_attendance_l3449_344941

theorem event_attendance (total : ℕ) (movie picnic gaming : ℕ) 
  (movie_picnic movie_gaming picnic_gaming : ℕ) (all_three : ℕ) 
  (h1 : total = 200)
  (h2 : movie = 50)
  (h3 : picnic = 80)
  (h4 : gaming = 60)
  (h5 : movie_picnic = 35)
  (h6 : movie_gaming = 10)
  (h7 : picnic_gaming = 20)
  (h8 : all_three = 8) :
  movie + picnic + gaming - (movie_picnic + movie_gaming + picnic_gaming) + all_three = 133 := by
sorry

end NUMINAMATH_CALUDE_event_attendance_l3449_344941


namespace NUMINAMATH_CALUDE_intersection_complement_proof_l3449_344965

def U : Set Nat := {1, 2, 3, 4}

theorem intersection_complement_proof
  (A B : Set Nat)
  (h1 : A ⊆ U)
  (h2 : B ⊆ U)
  (h3 : (U \ (A ∪ B)) = {4})
  (h4 : B = {1, 2}) :
  A ∩ (U \ B) = {3} :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_proof_l3449_344965


namespace NUMINAMATH_CALUDE_mud_efficacy_ratio_l3449_344917

/-- Represents the number of sprigs of mint in the original mud mixture -/
def original_mint_sprigs : ℕ := 3

/-- Represents the number of green tea leaves per sprig of mint -/
def tea_leaves_per_sprig : ℕ := 2

/-- Represents the number of green tea leaves needed in the new mud for the same efficacy -/
def new_mud_tea_leaves : ℕ := 12

/-- Calculates the ratio of efficacy of new mud to original mud -/
def efficacy_ratio : ℚ := 1 / 2

theorem mud_efficacy_ratio :
  efficacy_ratio = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_mud_efficacy_ratio_l3449_344917


namespace NUMINAMATH_CALUDE_f_as_difference_of_increasing_functions_l3449_344967

def f (x : ℝ) : ℝ := 2 * x^2 + 5 * x - 3

theorem f_as_difference_of_increasing_functions :
  ∃ (g h : ℝ → ℝ), 
    (∀ x y, x < y → g x < g y) ∧ 
    (∀ x y, x < y → h x < h y) ∧ 
    (∀ x, f x = g x - h x) :=
sorry

end NUMINAMATH_CALUDE_f_as_difference_of_increasing_functions_l3449_344967


namespace NUMINAMATH_CALUDE_line_intercepts_l3449_344945

/-- Given a line with equation x/4 - y/3 = 1, prove that its x-intercept is 4 and y-intercept is -3 -/
theorem line_intercepts :
  let line := (fun (x y : ℝ) => x/4 - y/3 = 1)
  (∃ x : ℝ, line x 0 ∧ x = 4) ∧
  (∃ y : ℝ, line 0 y ∧ y = -3) := by
sorry

end NUMINAMATH_CALUDE_line_intercepts_l3449_344945


namespace NUMINAMATH_CALUDE_license_plate_count_l3449_344942

/-- The number of possible digits (0-9) -/
def num_digits : ℕ := 10

/-- The number of possible letters (A-Z) -/
def num_letters : ℕ := 26

/-- The number of digits in a license plate -/
def digits_count : ℕ := 5

/-- The number of letters in a license plate -/
def letters_count : ℕ := 3

/-- The number of possible positions for the letter block -/
def block_positions : ℕ := digits_count + 1

/-- The total number of distinct license plates -/
def total_license_plates : ℕ := 
  block_positions * (num_digits ^ digits_count) * (num_letters ^ letters_count)

theorem license_plate_count : total_license_plates = 105456000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l3449_344942


namespace NUMINAMATH_CALUDE_canoe_downstream_speed_l3449_344973

/-- Represents the speed of a canoe in different conditions -/
structure CanoeSpeed where
  stillWater : ℝ
  upstream : ℝ

/-- Calculates the downstream speed of a canoe given its speed in still water and upstream -/
def downstreamSpeed (c : CanoeSpeed) : ℝ :=
  2 * c.stillWater - c.upstream

/-- Theorem stating that for a canoe with 12.5 km/hr speed in still water and 9 km/hr upstream speed, 
    the downstream speed is 16 km/hr -/
theorem canoe_downstream_speed :
  let c : CanoeSpeed := { stillWater := 12.5, upstream := 9 }
  downstreamSpeed c = 16 := by
  sorry


end NUMINAMATH_CALUDE_canoe_downstream_speed_l3449_344973


namespace NUMINAMATH_CALUDE_triangle_problem_l3449_344969

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The law of sines for a triangle -/
def lawOfSines (t : Triangle) : Prop :=
  t.a / (Real.sin t.A) = t.b / (Real.sin t.B) ∧ 
  t.b / (Real.sin t.B) = t.c / (Real.sin t.C)

/-- The law of cosines for a triangle -/
def lawOfCosines (t : Triangle) : Prop :=
  t.a^2 = t.b^2 + t.c^2 - 2*t.b*t.c*(Real.cos t.A) ∧
  t.b^2 = t.a^2 + t.c^2 - 2*t.a*t.c*(Real.cos t.B) ∧
  t.c^2 = t.a^2 + t.b^2 - 2*t.a*t.b*(Real.cos t.C)

theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 7)
  (h2 : t.c = 3)
  (h3 : Real.sin t.C / Real.sin t.B = 3/5)
  (h4 : lawOfSines t)
  (h5 : lawOfCosines t) :
  t.b = 5 ∧ Real.cos t.A = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3449_344969


namespace NUMINAMATH_CALUDE_vector_equation_l3449_344946

/-- Given vectors a, b, c, and e in a vector space, 
    prove that 2a - 3b + c = 23e under certain conditions. -/
theorem vector_equation (V : Type*) [AddCommGroup V] [Module ℝ V] 
  (e : V) (a b c : V) 
  (ha : a = 5 • e) 
  (hb : b = -3 • e) 
  (hc : c = 4 • e) : 
  2 • a - 3 • b + c = 23 • e := by sorry

end NUMINAMATH_CALUDE_vector_equation_l3449_344946


namespace NUMINAMATH_CALUDE_pet_store_cages_l3449_344966

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l3449_344966


namespace NUMINAMATH_CALUDE_loan_duration_for_b_l3449_344921

/-- Calculates the simple interest given principal, rate, and time -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem loan_duration_for_b (principal_b principal_c duration_c total_interest : ℝ) 
  (rate : ℝ) (h1 : principal_b = 5000)
  (h2 : principal_c = 3000)
  (h3 : duration_c = 4)
  (h4 : rate = 0.1)
  (h5 : simple_interest principal_b rate (duration_b) + 
        simple_interest principal_c rate duration_c = total_interest)
  (h6 : total_interest = 2200) :
  duration_b = 2 := by
  sorry

#check loan_duration_for_b

end NUMINAMATH_CALUDE_loan_duration_for_b_l3449_344921


namespace NUMINAMATH_CALUDE_direct_proportion_decreasing_l3449_344954

theorem direct_proportion_decreasing (k x₁ x₂ y₁ y₂ : ℝ) :
  k < 0 →
  x₁ < x₂ →
  y₁ = k * x₁ →
  y₂ = k * x₂ →
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_decreasing_l3449_344954


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3449_344947

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  a 3 + a 9 + a 15 + a 21 = 8 →
  a 1 + a 23 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3449_344947


namespace NUMINAMATH_CALUDE_remainder_of_difference_l3449_344935

theorem remainder_of_difference (s t : ℕ) (hs : s > 0) (ht : t > 0) 
  (h_s_mod : s % 6 = 2) (h_t_mod : t % 6 = 3) (h_s_gt_t : s > t) : 
  (s - t) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_difference_l3449_344935


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3449_344933

theorem inequality_solution_set (x : ℝ) : 2 ≤ x / (2 * x - 1) ∧ x / (2 * x - 1) < 5 ↔ x ∈ Set.Ioo (5/9 : ℝ) (2/3 : ℝ) ∪ {2/3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3449_344933


namespace NUMINAMATH_CALUDE_trig_function_amplitude_l3449_344987

theorem trig_function_amplitude 
  (y : ℝ → ℝ) 
  (a b c d : ℝ) 
  (h1 : ∀ x, y x = a * Real.cos (b * x + c) + d) 
  (h2 : ∃ x, y x = 4) 
  (h3 : ∃ x, y x = 0) 
  (h4 : ∀ x, y x ≤ 4) 
  (h5 : ∀ x, y x ≥ 0) : 
  a = 2 := by sorry

end NUMINAMATH_CALUDE_trig_function_amplitude_l3449_344987


namespace NUMINAMATH_CALUDE_point_on_y_axis_m_zero_l3449_344924

/-- A point P with coordinates (x, y) lies on the y-axis if and only if x = 0 -/
def lies_on_y_axis (P : ℝ × ℝ) : Prop := P.1 = 0

/-- The theorem states that if a point P(m,2) lies on the y-axis, then m = 0 -/
theorem point_on_y_axis_m_zero (m : ℝ) :
  lies_on_y_axis (m, 2) → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_y_axis_m_zero_l3449_344924


namespace NUMINAMATH_CALUDE_binary_addition_subtraction_l3449_344955

/-- Converts a binary number (represented as a list of bits) to a natural number. -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents the binary number 1101₂ -/
def b1101 : List Bool := [true, true, false, true]

/-- Represents the binary number 111₂ -/
def b111 : List Bool := [true, true, true]

/-- Represents the binary number 1010₂ -/
def b1010 : List Bool := [true, false, true, false]

/-- Represents the binary number 1011₂ -/
def b1011 : List Bool := [true, false, true, true]

/-- Represents the binary number 11001₂ (the expected result) -/
def b11001 : List Bool := [true, true, false, false, true]

/-- The main theorem to prove -/
theorem binary_addition_subtraction :
  binary_to_nat b1101 + binary_to_nat b111 - binary_to_nat b1010 + binary_to_nat b1011 =
  binary_to_nat b11001 := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_subtraction_l3449_344955


namespace NUMINAMATH_CALUDE_range_of_n_l3449_344993

theorem range_of_n (m n : ℝ) : (m^2 - 2*m)^2 + 4*m^2 - 8*m + 6 - n = 0 → n ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_n_l3449_344993


namespace NUMINAMATH_CALUDE_right_to_left_equiv_standard_not_equiv_l3449_344952

/-- Evaluates an expression in a right-to-left order -/
noncomputable def evaluateRightToLeft (a b c d : ℝ) : ℝ :=
  a / (b - c - d)

/-- Standard algebraic evaluation -/
noncomputable def evaluateStandard (a b c d : ℝ) : ℝ :=
  a / b - c + d

/-- Theorem stating the equivalence of right-to-left evaluation and the correct standard algebraic form -/
theorem right_to_left_equiv (a b c d : ℝ) :
  evaluateRightToLeft a b c d = a / (b - c - d) :=
by sorry

/-- Theorem stating that the standard algebraic evaluation is not equivalent to the right-to-left evaluation -/
theorem standard_not_equiv (a b c d : ℝ) :
  evaluateStandard a b c d ≠ evaluateRightToLeft a b c d :=
by sorry

end NUMINAMATH_CALUDE_right_to_left_equiv_standard_not_equiv_l3449_344952


namespace NUMINAMATH_CALUDE_largest_decimal_l3449_344910

theorem largest_decimal : 
  let a := 0.9123
  let b := 0.9912
  let c := 0.9191
  let d := 0.9301
  let e := 0.9091
  b > a ∧ b > c ∧ b > d ∧ b > e := by
  sorry

end NUMINAMATH_CALUDE_largest_decimal_l3449_344910


namespace NUMINAMATH_CALUDE_car_speed_comparison_l3449_344976

theorem car_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0) :
  3 / (1/u + 1/v + 1/w) ≤ (u + v + w) / 3 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_comparison_l3449_344976


namespace NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3449_344968

/-- Represents a tetrahedron with vertices P, Q, R, and S -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the specific tetrahedron is 20/3 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 6,
    PR := 5,
    PS := 5,
    QR := 5,
    QS := 4,
    RS := (10/3) * Real.sqrt 3
  }
  tetrahedronVolume t = 20/3 := by sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_volume_l3449_344968


namespace NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l3449_344930

theorem shaded_fraction_of_rectangle (length width : ℕ) 
  (h_length : length = 15)
  (h_width : width = 20)
  (section_fraction : ℚ)
  (h_section : section_fraction = 1 / 5)
  (shaded_fraction : ℚ)
  (h_shaded : shaded_fraction = 1 / 4) :
  (shaded_fraction * section_fraction : ℚ) = 1 / 20 := by
sorry

end NUMINAMATH_CALUDE_shaded_fraction_of_rectangle_l3449_344930


namespace NUMINAMATH_CALUDE_ratio_equals_2021_l3449_344916

def numerator_sum : ℕ → ℚ
  | 0 => 0
  | n + 1 => numerator_sum n + (2021 - n) / (n + 1)

def denominator_sum : ℕ → ℚ
  | 0 => 0
  | n + 1 => denominator_sum n + 1 / (n + 3)

theorem ratio_equals_2021 : 
  (numerator_sum 2016) / (denominator_sum 2016) = 2021 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equals_2021_l3449_344916


namespace NUMINAMATH_CALUDE_dave_initial_apps_l3449_344950

/-- The number of apps Dave had on his phone initially -/
def initial_apps : ℕ := sorry

/-- The number of apps Dave deleted -/
def deleted_apps : ℕ := 18

/-- The number of apps remaining after deletion -/
def remaining_apps : ℕ := 5

/-- Theorem stating the initial number of apps -/
theorem dave_initial_apps : initial_apps = 23 := by
  sorry

end NUMINAMATH_CALUDE_dave_initial_apps_l3449_344950


namespace NUMINAMATH_CALUDE_decreasing_interval_of_quadratic_l3449_344979

def f (x : ℝ) := x^2 - 2*x - 3

theorem decreasing_interval_of_quadratic :
  ∀ x : ℝ, (∀ y : ℝ, y < x → f y < f x) ↔ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_decreasing_interval_of_quadratic_l3449_344979


namespace NUMINAMATH_CALUDE_honey_servings_l3449_344944

/-- Proves that a container with 47 1/3 cups of honey contains 40 12/21 servings when each serving is 1 1/6 cups -/
theorem honey_servings (container : ℚ) (serving : ℚ) :
  container = 47 + 1 / 3 →
  serving = 1 + 1 / 6 →
  container / serving = 40 + 12 / 21 := by
sorry

end NUMINAMATH_CALUDE_honey_servings_l3449_344944


namespace NUMINAMATH_CALUDE_stratified_sample_female_count_l3449_344991

/-- Calculates the number of female athletes in a stratified sample -/
theorem stratified_sample_female_count 
  (total_athletes : ℕ) 
  (female_athletes : ℕ) 
  (male_athletes : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = female_athletes + male_athletes)
  (h2 : total_athletes = 98)
  (h3 : female_athletes = 42)
  (h4 : male_athletes = 56)
  (h5 : sample_size = 28) :
  (female_athletes : ℚ) * (sample_size : ℚ) / (total_athletes : ℚ) = 12 := by
  sorry

#check stratified_sample_female_count

end NUMINAMATH_CALUDE_stratified_sample_female_count_l3449_344991


namespace NUMINAMATH_CALUDE_course_selection_combinations_l3449_344957

/-- The number of available courses -/
def num_courses : ℕ := 4

/-- The number of courses student A chooses -/
def courses_A : ℕ := 2

/-- The number of courses students B and C each choose -/
def courses_BC : ℕ := 3

/-- The total number of different possible combinations -/
def total_combinations : ℕ := Nat.choose num_courses courses_A * (Nat.choose num_courses courses_BC)^2

theorem course_selection_combinations :
  total_combinations = 96 :=
by sorry

end NUMINAMATH_CALUDE_course_selection_combinations_l3449_344957


namespace NUMINAMATH_CALUDE_range_of_a_l3449_344922

/-- Proposition p: x^2 + 2ax + 4 > 0 holds for all x ∈ ℝ -/
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

/-- Proposition q: x^2 - (a+1)x + 1 ≤ 0 has an empty solution set -/
def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 - (a+1)*x + 1 > 0

/-- The disjunction p ∨ q is true -/
axiom h1 (a : ℝ) : p a ∨ q a

/-- The conjunction p ∧ q is false -/
axiom h2 (a : ℝ) : ¬(p a ∧ q a)

/-- The range of values for a is (-3, -2] ∪ [1, 2) -/
theorem range_of_a : 
  {a : ℝ | (a > -3 ∧ a ≤ -2) ∨ (a ≥ 1 ∧ a < 2)} = {a : ℝ | p a ∨ q a ∧ ¬(p a ∧ q a)} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3449_344922


namespace NUMINAMATH_CALUDE_abc_inequality_l3449_344975

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  a + b + c + a * b + b * c + c * a ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3449_344975


namespace NUMINAMATH_CALUDE_total_oranges_picked_l3449_344996

theorem total_oranges_picked (del_per_day : ℕ) (del_days : ℕ) (juan_oranges : ℕ) : 
  del_per_day = 23 → del_days = 2 → juan_oranges = 61 → 
  del_per_day * del_days + juan_oranges = 107 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l3449_344996


namespace NUMINAMATH_CALUDE_remaining_money_theorem_l3449_344919

def calculate_remaining_money (initial_amount : ℚ) : ℚ :=
  let day1_remaining := initial_amount * (1 - 3/5)
  let day2_remaining := day1_remaining * (1 - 7/12)
  let day3_remaining := day2_remaining * (1 - 2/3)
  let day4_remaining := day3_remaining * (1 - 1/6)
  let day5_remaining := day4_remaining * (1 - 5/8)
  let day6_remaining := day5_remaining * (1 - 3/5)
  day6_remaining

theorem remaining_money_theorem :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  abs (calculate_remaining_money 500 - 347/100) < ε := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_theorem_l3449_344919


namespace NUMINAMATH_CALUDE_min_value_theorem_l3449_344962

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : Real.log 2 * x + Real.log 8 * y = Real.log 2) :
  ∃ (min : ℝ), min = 4 ∧ ∀ z, z = 1/x + 1/(3*y) → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3449_344962


namespace NUMINAMATH_CALUDE_ebook_reader_difference_l3449_344926

theorem ebook_reader_difference (anna_count john_original_count : ℕ) : 
  anna_count = 50 →
  john_original_count < anna_count →
  john_original_count + anna_count = 82 + 3 →
  anna_count - john_original_count = 15 := by
sorry

end NUMINAMATH_CALUDE_ebook_reader_difference_l3449_344926
