import Mathlib

namespace NUMINAMATH_CALUDE_toy_store_optimization_l323_32368

/-- Toy store profit optimization problem --/
theorem toy_store_optimization :
  let initial_price : ℝ := 120
  let initial_cost : ℝ := 80
  let initial_sales : ℝ := 20
  let price_reduction (x : ℝ) := x
  let sales_increase (x : ℝ) := 2 * x
  let new_price (x : ℝ) := initial_price - price_reduction x
  let new_sales (x : ℝ) := initial_sales + sales_increase x
  let profit (x : ℝ) := (new_price x - initial_cost) * new_sales x

  -- Daily sales function
  ∀ x, new_sales x = 20 + 2*x ∧

  -- Profit function and domain
  (∀ x, profit x = -2*x^2 + 60*x + 800) ∧
  (∀ x, 0 < x → x ≤ 40 → new_price x ≥ initial_cost) ∧

  -- Maximum profit
  ∃ x, 0 < x ∧ x ≤ 40 ∧ 
    profit x = 1250 ∧
    (∀ y, 0 < y → y ≤ 40 → profit y ≤ profit x) ∧
    new_price x = 105 :=
by sorry

end NUMINAMATH_CALUDE_toy_store_optimization_l323_32368


namespace NUMINAMATH_CALUDE_range_of_a_l323_32329

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x - a ≥ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ (Set.univ \ B a) → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l323_32329


namespace NUMINAMATH_CALUDE_abs_neg_three_equals_three_l323_32350

theorem abs_neg_three_equals_three : abs (-3 : ℝ) = 3 := by sorry

end NUMINAMATH_CALUDE_abs_neg_three_equals_three_l323_32350


namespace NUMINAMATH_CALUDE_harold_finances_theorem_l323_32371

/-- Harold's monthly finances --/
def harold_finances (income rent car_payment groceries : ℚ) : Prop :=
  let utilities := car_payment / 2
  let total_expenses := rent + car_payment + utilities + groceries
  let remaining := income - total_expenses
  let retirement_savings := remaining / 2
  let final_remaining := remaining - retirement_savings
  income = 2500 ∧ 
  rent = 700 ∧ 
  car_payment = 300 ∧ 
  groceries = 50 ∧ 
  final_remaining = 650

theorem harold_finances_theorem :
  ∀ income rent car_payment groceries : ℚ,
  harold_finances income rent car_payment groceries :=
by
  sorry

end NUMINAMATH_CALUDE_harold_finances_theorem_l323_32371


namespace NUMINAMATH_CALUDE_coat_price_calculation_l323_32348

/-- Calculates the final price of a coat after discounts and tax -/
def finalPrice (originalPrice : ℝ) (initialDiscount : ℝ) (additionalDiscount : ℝ) (salesTax : ℝ) : ℝ :=
  let priceAfterInitialDiscount := originalPrice * (1 - initialDiscount)
  let priceAfterAdditionalDiscount := priceAfterInitialDiscount - additionalDiscount
  priceAfterAdditionalDiscount * (1 + salesTax)

/-- Theorem stating that the final price of the coat is $112.75 -/
theorem coat_price_calculation :
  finalPrice 150 0.25 10 0.1 = 112.75 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_calculation_l323_32348


namespace NUMINAMATH_CALUDE_painting_time_calculation_l323_32312

/-- Given an artist's weekly painting hours and production rate over four weeks,
    calculate the time needed to complete one painting. -/
theorem painting_time_calculation (weekly_hours : ℕ) (paintings_in_four_weeks : ℕ) :
  weekly_hours = 30 →
  paintings_in_four_weeks = 40 →
  (4 * weekly_hours) / paintings_in_four_weeks = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_painting_time_calculation_l323_32312


namespace NUMINAMATH_CALUDE_difference_ones_zeros_253_l323_32309

def binary_representation (n : ℕ) : List Bool :=
  sorry

def count_ones (binary : List Bool) : ℕ :=
  sorry

def count_zeros (binary : List Bool) : ℕ :=
  sorry

theorem difference_ones_zeros_253 :
  let binary := binary_representation 253
  let ones := count_ones binary
  let zeros := count_zeros binary
  ones - zeros = 6 :=
sorry

end NUMINAMATH_CALUDE_difference_ones_zeros_253_l323_32309


namespace NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l323_32308

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem stating the distance between foci for a specific ellipse -/
theorem specific_ellipse_foci_distance :
  ∃ (e : ParallelAxisEllipse),
    e.x_tangent = (3, 0) ∧
    e.y_tangent = (0, 2) ∧
    foci_distance e = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_specific_ellipse_foci_distance_l323_32308


namespace NUMINAMATH_CALUDE_integral_equals_ln5_over_8_l323_32399

/-- The definite integral of the given function from 0 to 1 is equal to (1/8) * ln(5) -/
theorem integral_equals_ln5_over_8 :
  ∫ x in (0 : ℝ)..1, (4 * Real.sqrt (1 - x) - Real.sqrt (x + 1)) /
    ((Real.sqrt (x + 1) + 4 * Real.sqrt (1 - x)) * (x + 1)^2) = (1/8) * Real.log 5 := by
  sorry

end NUMINAMATH_CALUDE_integral_equals_ln5_over_8_l323_32399


namespace NUMINAMATH_CALUDE_fish_in_tank_l323_32332

theorem fish_in_tank (total : ℕ) (blue : ℕ) (spotted : ℕ) : 
  3 * blue = total → 
  2 * spotted = blue → 
  spotted = 10 → 
  total = 60 := by
sorry

end NUMINAMATH_CALUDE_fish_in_tank_l323_32332


namespace NUMINAMATH_CALUDE_functional_equation_solution_l323_32307

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y z : ℝ, g (x^2 + y^2 + y * g z) = x * g x + z^2 * g y

/-- The theorem stating that g must be either the zero function or the identity function -/
theorem functional_equation_solution (g : ℝ → ℝ) (h : FunctionalEquation g) :
    (∀ x, g x = 0) ∨ (∀ x, g x = x) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l323_32307


namespace NUMINAMATH_CALUDE_jessie_points_l323_32382

def total_points : ℕ := 311
def some_players_points : ℕ := 188
def num_equal_scorers : ℕ := 3

theorem jessie_points : 
  (total_points - some_players_points) / num_equal_scorers = 41 := by
  sorry

end NUMINAMATH_CALUDE_jessie_points_l323_32382


namespace NUMINAMATH_CALUDE_sun_division_problem_l323_32384

/-- Prove that the total amount is 105 given the conditions of the sun division problem -/
theorem sun_division_problem (x y z : ℝ) : 
  (y = 0.45 * x) →  -- For each rupee x gets, y gets 45 paisa
  (z = 0.30 * x) →  -- For each rupee x gets, z gets 30 paisa
  (y = 27) →        -- y's share is Rs. 27
  (x + y + z = 105) -- The total amount is 105
  := by sorry

end NUMINAMATH_CALUDE_sun_division_problem_l323_32384


namespace NUMINAMATH_CALUDE_system_solution_l323_32372

theorem system_solution :
  ∃ (x y : ℚ), (7 * x = -5 - 3 * y) ∧ (4 * x = 5 * y - 34) :=
by
  use (-127/47), (218/47)
  sorry

end NUMINAMATH_CALUDE_system_solution_l323_32372


namespace NUMINAMATH_CALUDE_tangent_line_condition_l323_32386

/-- Given a function f(x) = e^x + a*cos(x), if its tangent line at x = 0 passes through (1, 6), then a = 4 -/
theorem tangent_line_condition (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ Real.exp x + a * Real.cos x
  let f' : ℝ → ℝ := λ x ↦ Real.exp x - a * Real.sin x
  (f 0 = 6) ∧ (f' 0 = 1) → a = 4 := by sorry

end NUMINAMATH_CALUDE_tangent_line_condition_l323_32386


namespace NUMINAMATH_CALUDE_x_fourth_power_zero_l323_32304

theorem x_fourth_power_zero (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (1 - x^2) + Real.sqrt (1 + x^2) = 2) : x^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_power_zero_l323_32304


namespace NUMINAMATH_CALUDE_carey_gumballs_difference_l323_32392

/-- The number of gumballs Carolyn bought -/
def carolyn_gumballs : ℕ := 17

/-- The number of gumballs Lew bought -/
def lew_gumballs : ℕ := 12

/-- The average number of gumballs bought by the three people -/
def average_gumballs : ℚ → ℚ := λ c => (carolyn_gumballs + lew_gumballs + c) / 3

/-- The theorem stating the difference between max and min gumballs Carey could have bought -/
theorem carey_gumballs_difference :
  ∃ (min_c max_c : ℕ),
    (∀ c : ℚ, 19 ≤ average_gumballs c → average_gumballs c ≤ 25 → ↑min_c ≤ c ∧ c ≤ ↑max_c) ∧
    max_c - min_c = 18 := by
  sorry

end NUMINAMATH_CALUDE_carey_gumballs_difference_l323_32392


namespace NUMINAMATH_CALUDE_rogers_first_bag_l323_32353

/-- Represents the number of candy bags a person has -/
def num_bags : ℕ := 2

/-- Represents the number of pieces in each of Sandra's bags -/
def sandra_pieces_per_bag : ℕ := 6

/-- Represents the number of pieces in Roger's second bag -/
def roger_second_bag : ℕ := 3

/-- Represents the difference in total pieces between Roger and Sandra -/
def roger_sandra_diff : ℕ := 2

/-- Represents the number of pieces in one of Roger's bags -/
def roger_one_bag : ℕ := 11

/-- Calculates the total number of candy pieces Sandra has -/
def sandra_total : ℕ := num_bags * sandra_pieces_per_bag

/-- Calculates the total number of candy pieces Roger has -/
def roger_total : ℕ := sandra_total + roger_sandra_diff

/-- Theorem: The number of pieces in Roger's first bag is 11 -/
theorem rogers_first_bag : roger_total - roger_second_bag = roger_one_bag :=
by sorry

end NUMINAMATH_CALUDE_rogers_first_bag_l323_32353


namespace NUMINAMATH_CALUDE_probability_multiple_2_3_or_5_l323_32358

def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

def count_multiples (max : ℕ) (divisor : ℕ) : ℕ :=
  (max / divisor : ℕ)

def count_multiples_of_2_3_or_5 (max : ℕ) : ℕ :=
  count_multiples max 2 + count_multiples max 3 + count_multiples max 5 -
  (count_multiples max 6 + count_multiples max 10 + count_multiples max 15) +
  count_multiples max 30

theorem probability_multiple_2_3_or_5 :
  (count_multiples_of_2_3_or_5 120 : ℚ) / 120 = 11 / 15 := by
  sorry

#eval count_multiples_of_2_3_or_5 120

end NUMINAMATH_CALUDE_probability_multiple_2_3_or_5_l323_32358


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l323_32330

/-- The parabola function -/
def f (y : ℝ) : ℝ := -3 * y^2 + 2 * y + 4

/-- The line function -/
def g (k : ℝ) : ℝ → ℝ := λ _ ↦ k

/-- The condition for a single intersection point -/
def has_single_intersection (k : ℝ) : Prop :=
  ∃! y, f y = g k y

theorem parabola_line_intersection :
  ∀ k, has_single_intersection k ↔ k = 13/3 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l323_32330


namespace NUMINAMATH_CALUDE_max_triangles_is_28_l323_32343

/-- The number of points on the hypotenuse of a right triangle with legs of length 7 -/
def hypotenuse_points : ℕ := 8

/-- The maximum number of triangles that can be formed within the right triangle -/
def max_triangles : ℕ := Nat.choose hypotenuse_points 2

/-- Theorem stating the maximum number of triangles is 28 -/
theorem max_triangles_is_28 : max_triangles = 28 := by sorry

end NUMINAMATH_CALUDE_max_triangles_is_28_l323_32343


namespace NUMINAMATH_CALUDE_modulus_of_Z_l323_32355

theorem modulus_of_Z (Z : ℂ) (h : (1 + Complex.I) * Z = Complex.I) : 
  Complex.abs Z = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_Z_l323_32355


namespace NUMINAMATH_CALUDE_number_of_papers_l323_32344

/-- Represents the marks obtained in each paper -/
structure PaperMarks where
  fullMarks : ℝ
  proportions : List ℝ
  totalPapers : ℕ

/-- Checks if the given PaperMarks satisfies the problem conditions -/
def satisfiesConditions (pm : PaperMarks) : Prop :=
  pm.proportions = [5, 6, 7, 8, 9] ∧
  pm.totalPapers = pm.proportions.length ∧
  (pm.proportions.sum * pm.fullMarks * 0.6 = pm.proportions.sum * pm.fullMarks) ∧
  (List.filter (fun p => p * pm.fullMarks > 0.5 * pm.fullMarks) pm.proportions).length = 5

/-- Theorem stating that if the conditions are satisfied, the number of papers is 5 -/
theorem number_of_papers (pm : PaperMarks) (h : satisfiesConditions pm) : pm.totalPapers = 5 := by
  sorry

end NUMINAMATH_CALUDE_number_of_papers_l323_32344


namespace NUMINAMATH_CALUDE_sons_age_l323_32391

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 22 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l323_32391


namespace NUMINAMATH_CALUDE_function_identity_l323_32345

theorem function_identity (f : ℕ → ℕ) (h : ∀ x y : ℕ, f (f x + f y) = x + y) :
  ∀ n : ℕ, f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l323_32345


namespace NUMINAMATH_CALUDE_handshakes_and_highfives_l323_32397

/-- The number of unique pairings in a group of n people -/
def uniquePairings (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of people at the gathering -/
def numberOfPeople : ℕ := 12

theorem handshakes_and_highfives :
  uniquePairings numberOfPeople = 66 ∧
  uniquePairings numberOfPeople = 66 := by
  sorry

#eval uniquePairings numberOfPeople

end NUMINAMATH_CALUDE_handshakes_and_highfives_l323_32397


namespace NUMINAMATH_CALUDE_squeak_interval_is_nine_seconds_l323_32327

/-- Represents a gear mechanism with two gears -/
structure GearMechanism where
  small_gear_teeth : ℕ
  large_gear_teeth : ℕ
  large_gear_revolution_time : ℝ

/-- Calculates the time interval between squeaks for a gear mechanism -/
def squeak_interval (gm : GearMechanism) : ℝ :=
  let lcm := Nat.lcm gm.small_gear_teeth gm.large_gear_teeth
  let large_gear_revolutions := lcm / gm.large_gear_teeth
  large_gear_revolutions * gm.large_gear_revolution_time

/-- Theorem stating that for the given gear mechanism, the squeak interval is 9 seconds -/
theorem squeak_interval_is_nine_seconds (gm : GearMechanism) 
  (h1 : gm.small_gear_teeth = 12) 
  (h2 : gm.large_gear_teeth = 32) 
  (h3 : gm.large_gear_revolution_time = 3) : 
  squeak_interval gm = 9 := by
  sorry

#eval squeak_interval { small_gear_teeth := 12, large_gear_teeth := 32, large_gear_revolution_time := 3 }

end NUMINAMATH_CALUDE_squeak_interval_is_nine_seconds_l323_32327


namespace NUMINAMATH_CALUDE_negative_number_identification_l323_32319

theorem negative_number_identification :
  (0 ≥ 0) ∧ ((1/2 : ℝ) > 0) ∧ (-(-5) > 0) ∧ (-Real.sqrt 5 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negative_number_identification_l323_32319


namespace NUMINAMATH_CALUDE_maya_books_last_week_l323_32346

/-- The number of pages in each book Maya reads. -/
def pages_per_book : ℕ := 300

/-- The total number of pages Maya read over two weeks. -/
def total_pages : ℕ := 4500

/-- The ratio of pages read this week compared to last week. -/
def week_ratio : ℕ := 2

/-- The number of books Maya read last week. -/
def books_last_week : ℕ := 5

theorem maya_books_last_week :
  books_last_week * pages_per_book * (week_ratio + 1) = total_pages :=
sorry

end NUMINAMATH_CALUDE_maya_books_last_week_l323_32346


namespace NUMINAMATH_CALUDE_cube_volume_problem_l323_32351

theorem cube_volume_problem (a : ℝ) : 
  (a > 0) → 
  (a^3 - ((a - 2) * a * (a + 2)) = 16) → 
  (a^3 = 64) :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l323_32351


namespace NUMINAMATH_CALUDE_fallen_sheets_l323_32323

def is_permutation (a b : Nat) : Prop :=
  (a.digits 10).sum = (b.digits 10).sum ∧
  (a.digits 10).prod = (b.digits 10).prod

theorem fallen_sheets (n : Nat) 
  (h1 : is_permutation 387 n)
  (h2 : n > 387)
  (h3 : Even n) :
  (n - 387 + 1) / 2 = 176 :=
sorry

end NUMINAMATH_CALUDE_fallen_sheets_l323_32323


namespace NUMINAMATH_CALUDE_chromium_percentage_in_cast_iron_l323_32337

theorem chromium_percentage_in_cast_iron 
  (x y : ℝ) 
  (h1 : 5 * x + y = 6 * min x y) 
  (h2 : x + y = 0.16) : 
  (x = 0.11 ∧ y = 0.05) ∨ (x = 0.05 ∧ y = 0.11) :=
sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_cast_iron_l323_32337


namespace NUMINAMATH_CALUDE_half_angle_quadrants_l323_32396

theorem half_angle_quadrants (α : Real) : 
  (∃ k : ℤ, 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) →
  (∃ n : ℤ, 2 * n * π < α / 2 ∧ α / 2 < 2 * n * π + π / 2) ∨
  (∃ n : ℤ, (2 * n + 1) * π < α / 2 ∧ α / 2 < (2 * n + 1) * π + π / 2) := by
sorry


end NUMINAMATH_CALUDE_half_angle_quadrants_l323_32396


namespace NUMINAMATH_CALUDE_duck_cow_problem_l323_32339

/-- Proves that in a group of ducks and cows, if the total number of legs is 32 more than twice the number of heads, then the number of cows is 16 -/
theorem duck_cow_problem (ducks cows : ℕ) : 
  2 * (ducks + cows) + 32 = 2 * ducks + 4 * cows → cows = 16 := by
  sorry

end NUMINAMATH_CALUDE_duck_cow_problem_l323_32339


namespace NUMINAMATH_CALUDE_shoe_ratio_proof_l323_32366

theorem shoe_ratio_proof (total_shoes brown_shoes : ℕ) 
  (h1 : total_shoes = 66) 
  (h2 : brown_shoes = 22) : 
  (total_shoes - brown_shoes) / brown_shoes = 2 := by
sorry

end NUMINAMATH_CALUDE_shoe_ratio_proof_l323_32366


namespace NUMINAMATH_CALUDE_rectangle_area_l323_32385

/-- The area of a rectangle with width 10 meters and length 2 meters is 20 square meters. -/
theorem rectangle_area : 
  ∀ (width length area : ℝ), 
  width = 10 → 
  length = 2 → 
  area = width * length → 
  area = 20 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l323_32385


namespace NUMINAMATH_CALUDE_floor_times_self_eq_100_l323_32378

theorem floor_times_self_eq_100 :
  ∃ (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 100 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_100_l323_32378


namespace NUMINAMATH_CALUDE_time_after_1456_minutes_l323_32383

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

theorem time_after_1456_minutes :
  let start_time : Time := ⟨6, 0, by sorry⟩
  let elapsed_minutes : Nat := 1456
  let end_time : Time := addMinutes start_time elapsed_minutes
  end_time = ⟨6, 16, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_time_after_1456_minutes_l323_32383


namespace NUMINAMATH_CALUDE_sum_always_positive_l323_32347

def is_monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (h1 : is_monotone_increasing f)
  (h2 : is_odd_function f)
  (h3 : arithmetic_sequence a)
  (h4 : a 1 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l323_32347


namespace NUMINAMATH_CALUDE_angle_C_value_max_area_l323_32331

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

end NUMINAMATH_CALUDE_angle_C_value_max_area_l323_32331


namespace NUMINAMATH_CALUDE_school_wall_stars_l323_32379

theorem school_wall_stars (num_students : ℕ) (stars_per_student : ℕ) (total_stars : ℕ) :
  num_students = 210 →
  stars_per_student = 6 →
  total_stars = num_students * stars_per_student →
  total_stars = 1260 :=
by sorry

end NUMINAMATH_CALUDE_school_wall_stars_l323_32379


namespace NUMINAMATH_CALUDE_raft_existence_l323_32333

-- Define the river shape
def RiverShape : Type := sorry

-- Define the path of the chip
def ChipPath (river : RiverShape) : Type := sorry

-- Define the raft shape
def RaftShape : Type := sorry

-- Function to check if a raft touches both banks at all points
def touchesBothBanks (river : RiverShape) (raft : RaftShape) (path : ChipPath river) : Prop := sorry

-- Theorem statement
theorem raft_existence (river : RiverShape) (chip_path : ChipPath river) :
  ∃ (raft : RaftShape), touchesBothBanks river raft chip_path := by
  sorry

end NUMINAMATH_CALUDE_raft_existence_l323_32333


namespace NUMINAMATH_CALUDE_cone_sphere_equal_volume_l323_32334

theorem cone_sphere_equal_volume (r : ℝ) (h : ℝ) :
  r = 1 →
  (1/3 * π * r^2 * h) = (4/3 * π) →
  Real.sqrt (r^2 + h^2) = Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_cone_sphere_equal_volume_l323_32334


namespace NUMINAMATH_CALUDE_inequality_solution_l323_32365

theorem inequality_solution (x : ℝ) : 
  (8 * x^2 + 16 * x - 51) / ((2 * x - 3) * (x + 4)) < 3 ↔ 
  (x > -4 ∧ x < -3) ∨ (x > 3/2 ∧ x < 5/2) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l323_32365


namespace NUMINAMATH_CALUDE_poison_frog_count_l323_32354

theorem poison_frog_count (total : ℕ) (tree : ℕ) (wood : ℕ) (poison : ℕ) :
  total = 78 →
  tree = 55 →
  wood = 13 →
  poison = total - (tree + wood) →
  poison = 10 := by
  sorry

end NUMINAMATH_CALUDE_poison_frog_count_l323_32354


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l323_32324

/-- Given that x² varies inversely with y⁴, prove that if x = 10 when y = 2, then x = 5 when y = √8 -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) (h1 : x^2 * y^4 = k) (h2 : x = 10 ∧ y = 2) :
  x = 5 ∧ y = Real.sqrt 8 :=
by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l323_32324


namespace NUMINAMATH_CALUDE_election_votes_l323_32326

theorem election_votes (winning_percentage : ℝ) (majority : ℕ) (total_votes : ℕ) : 
  winning_percentage = 0.6 →
  majority = 1380 →
  total_votes * winning_percentage - total_votes * (1 - winning_percentage) = majority →
  total_votes = 6900 := by
sorry

end NUMINAMATH_CALUDE_election_votes_l323_32326


namespace NUMINAMATH_CALUDE_function_inequality_l323_32380

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 1, (x - 1) * (deriv f x) - f x > 0) :
  (1 / (Real.sqrt 2 - 1)) * f (Real.sqrt 2) < f 2 ∧ f 2 < (1 / 2) * f 3 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l323_32380


namespace NUMINAMATH_CALUDE_sticker_difference_l323_32340

theorem sticker_difference (karl_stickers : ℕ) (ryan_more_than_karl : ℕ) (total_stickers : ℕ)
  (h1 : karl_stickers = 25)
  (h2 : ryan_more_than_karl = 20)
  (h3 : total_stickers = 105) :
  let ryan_stickers := karl_stickers + ryan_more_than_karl
  let ben_stickers := total_stickers - karl_stickers - ryan_stickers
  ryan_stickers - ben_stickers = 10 := by
sorry

end NUMINAMATH_CALUDE_sticker_difference_l323_32340


namespace NUMINAMATH_CALUDE_inequality_range_l323_32398

theorem inequality_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ π/2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) 
  ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l323_32398


namespace NUMINAMATH_CALUDE_peters_walked_distance_l323_32364

/-- Calculates the distance Peter has already walked given the total distance,
    his walking speed, and the remaining time to reach the store. -/
theorem peters_walked_distance
  (total_distance : ℝ)
  (walking_speed : ℝ)
  (remaining_time : ℝ)
  (h1 : total_distance = 2.5)
  (h2 : walking_speed = 1 / 20)
  (h3 : remaining_time = 30) :
  total_distance - (walking_speed * remaining_time) = 1 := by
  sorry

#check peters_walked_distance

end NUMINAMATH_CALUDE_peters_walked_distance_l323_32364


namespace NUMINAMATH_CALUDE_scaled_box_capacity_l323_32393

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.height * d.width * d.length

/-- Theorem: A box with 3 times the height, 2 times the width, and 1/2 times the length of a box
    that can hold 60 grams of clay can hold 180 grams of clay -/
theorem scaled_box_capacity
  (first_box : BoxDimensions)
  (first_box_capacity : ℝ)
  (h_first_box_capacity : first_box_capacity = 60)
  (second_box : BoxDimensions)
  (h_second_box_height : second_box.height = 3 * first_box.height)
  (h_second_box_width : second_box.width = 2 * first_box.width)
  (h_second_box_length : second_box.length = 1/2 * first_box.length) :
  (boxVolume second_box / boxVolume first_box) * first_box_capacity = 180 := by
  sorry

end NUMINAMATH_CALUDE_scaled_box_capacity_l323_32393


namespace NUMINAMATH_CALUDE_square_land_perimeter_l323_32315

theorem square_land_perimeter (a : ℝ) (h : 5 * a = 10 * (4 * Real.sqrt a) + 45) :
  4 * Real.sqrt a = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_land_perimeter_l323_32315


namespace NUMINAMATH_CALUDE_expression_is_integer_l323_32321

theorem expression_is_integer (x y z : ℤ) (n : ℕ) 
  (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
  ∃ k : ℤ, k = (x^n / ((x-y)*(x-z))) + (y^n / ((y-x)*(y-z))) + (z^n / ((z-x)*(z-y))) :=
by sorry

end NUMINAMATH_CALUDE_expression_is_integer_l323_32321


namespace NUMINAMATH_CALUDE_product_equals_29_l323_32387

theorem product_equals_29 (a : ℝ) (h : a^2 + a + 1 = 2) : (5 - a) * (6 + a) = 29 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_29_l323_32387


namespace NUMINAMATH_CALUDE_parabola_and_intersection_properties_l323_32335

/-- Parabola C with directrix x = -1/4 -/
def ParabolaC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = p.1}

/-- Line l passing through P(t, 0) -/
def LineL (t : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ m : ℝ, p.1 = m * p.2 + t}

/-- Points A and B are the intersections of ParabolaC and LineL -/
def IntersectionPoints (t : ℝ) : Set (ℝ × ℝ) :=
  ParabolaC ∩ LineL t

/-- Circle with diameter AB passes through the origin -/
def CircleThroughOrigin (t : ℝ) : Prop :=
  ∀ A B : ℝ × ℝ, A ∈ IntersectionPoints t → B ∈ IntersectionPoints t →
    A.1 * B.1 + A.2 * B.2 = 0

theorem parabola_and_intersection_properties :
  (∀ p : ℝ × ℝ, p ∈ ParabolaC ↔ p.2^2 = p.1) ∧
  (∀ t : ℝ, CircleThroughOrigin t → t = 0 ∨ t = 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_and_intersection_properties_l323_32335


namespace NUMINAMATH_CALUDE_sum_of_factors_30_l323_32314

def factors (n : ℕ) : Finset ℕ :=
  Finset.filter (λ x => n % x = 0) (Finset.range (n + 1))

theorem sum_of_factors_30 : (factors 30).sum id = 72 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_30_l323_32314


namespace NUMINAMATH_CALUDE_two_std_dev_below_mean_l323_32389

def normal_distribution (μ σ : ℝ) : Type := sorry

theorem two_std_dev_below_mean 
  (μ σ : ℝ) 
  (dist : normal_distribution μ σ) 
  (h_μ : μ = 14.5) 
  (h_σ : σ = 1.5) : 
  μ - 2 * σ = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_below_mean_l323_32389


namespace NUMINAMATH_CALUDE_car_profit_theorem_l323_32316

/-- Calculates the profit percentage on the original price of a car
    given the discount percentage on purchase and markup percentage on sale. -/
def profit_percentage (discount : ℝ) (markup : ℝ) : ℝ :=
  let purchase_price := 1 - discount
  let sale_price := purchase_price * (1 + markup)
  (sale_price - 1) * 100

/-- Theorem stating that buying a car at 5% discount and selling at 60% markup
    results in a 52% profit on the original price. -/
theorem car_profit_theorem :
  profit_percentage 0.05 0.60 = 52 := by sorry

end NUMINAMATH_CALUDE_car_profit_theorem_l323_32316


namespace NUMINAMATH_CALUDE_solution_set_inequality_l323_32322

theorem solution_set_inequality (x : ℝ) :
  x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l323_32322


namespace NUMINAMATH_CALUDE_sets_inclusion_l323_32306

-- Define the sets M, N, and P
def M : Set (ℝ × ℝ) := {p | abs p.1 + abs p.2 < 1}

def N : Set (ℝ × ℝ) := {p | Real.sqrt ((p.1 - 1/2)^2 + (p.2 + 1/2)^2) + 
                             Real.sqrt ((p.1 + 1/2)^2 + (p.2 - 1/2)^2) < 2 * Real.sqrt 2}

def P : Set (ℝ × ℝ) := {p | abs (p.1 + p.2) < 1 ∧ abs p.1 < 1 ∧ abs p.2 < 1}

-- State the theorem
theorem sets_inclusion : M ⊆ P ∧ P ⊆ N := by sorry

end NUMINAMATH_CALUDE_sets_inclusion_l323_32306


namespace NUMINAMATH_CALUDE_equation_solution_l323_32357

theorem equation_solution : ∃ x : ℝ, 2*x + 5 = 3*x - 2 ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l323_32357


namespace NUMINAMATH_CALUDE_garden_ratio_l323_32301

theorem garden_ratio (area width length : ℝ) : 
  area = 588 → width = 14 → length * width = area → (length / width = 3) := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l323_32301


namespace NUMINAMATH_CALUDE_inequality_implies_not_equal_l323_32381

theorem inequality_implies_not_equal (a b : ℝ) :
  (a / b + b / a > 2) → (a ≠ b) ∧ ¬(∀ a b : ℝ, a ≠ b → a / b + b / a > 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_implies_not_equal_l323_32381


namespace NUMINAMATH_CALUDE_product_of_powers_equals_fifty_l323_32367

theorem product_of_powers_equals_fifty :
  (5^(2/10)) * (10^(4/10)) * (10^(1/10)) * (10^(5/10)) * (5^(8/10)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_equals_fifty_l323_32367


namespace NUMINAMATH_CALUDE_remainder_after_division_l323_32361

theorem remainder_after_division (n : ℕ) : 
  (n / 7 = 12 ∧ n % 7 = 5) → n % 8 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_after_division_l323_32361


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l323_32325

theorem pentagon_largest_angle (a b c d e : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 → -- All angles are positive
  b / a = 3 / 2 ∧ c / a = 2 ∧ d / a = 5 / 2 ∧ e / a = 3 → -- Angles are in ratio 2:3:4:5:6
  a + b + c + d + e = 540 → -- Sum of angles in a pentagon
  e = 162 := by
sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l323_32325


namespace NUMINAMATH_CALUDE_floor_painting_rate_l323_32363

/-- Proves that the painting rate is 3 Rs. per square meter for a rectangular floor with given conditions --/
theorem floor_painting_rate (length breadth area cost : ℝ) : 
  length = 3 * breadth →
  length = 15.491933384829668 →
  area = length * breadth →
  cost = 240 →
  cost / area = 3 := by sorry

end NUMINAMATH_CALUDE_floor_painting_rate_l323_32363


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l323_32373

theorem complex_modulus_problem (a b : ℝ) :
  (Complex.mk a 1) * (Complex.mk 1 (-1)) = Complex.mk 3 b →
  Complex.abs (Complex.mk a b) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l323_32373


namespace NUMINAMATH_CALUDE_regions_for_99_lines_l323_32300

/-- The number of regions formed by a given number of lines in a plane -/
def num_regions (num_lines : ℕ) : Set ℕ :=
  {n | ∃ (configuration : Type) (f : configuration → ℕ), 
       (∀ c, f c ≤ (num_lines * (num_lines - 1)) / 2 + num_lines + 1) ∧
       (∃ c, f c = n)}

/-- Theorem stating that for 99 lines, the only possible numbers of regions less than 199 are 100 and 198 -/
theorem regions_for_99_lines :
  num_regions 99 ∩ {n | n < 199} = {100, 198} :=
by sorry

end NUMINAMATH_CALUDE_regions_for_99_lines_l323_32300


namespace NUMINAMATH_CALUDE_shortest_side_is_15_l323_32370

/-- Represents a triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 72
  side_eq : a = 30

/-- Calculates the semiperimeter of a triangle -/
def semiperimeter (t : IntTriangle) : ℚ :=
  (t.a + t.b + t.c) / 2

/-- Calculates the area of a triangle using Heron's formula -/
def area (t : IntTriangle) : ℚ :=
  let s := semiperimeter t
  (s * (s - t.a) * (s - t.b) * (s - t.c)).sqrt

/-- Main theorem: The shortest side of the triangle is 15 -/
theorem shortest_side_is_15 (t : IntTriangle) (area_int : ∃ n : ℕ, area t = n) :
  min t.a (min t.b t.c) = 15 := by
  sorry

#check shortest_side_is_15

end NUMINAMATH_CALUDE_shortest_side_is_15_l323_32370


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l323_32313

/-- Given a circle with equation x^2 + y^2 - 4x + 2y - 4 = 0, 
    prove that its center is at (2, -1) and its radius is 3. -/
theorem circle_center_and_radius : 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
    (C = (2, -1) ∧ r = 3) ∧ 
    ∀ (x y : ℝ), x^2 + y^2 - 4*x + 2*y - 4 = 0 ↔ (x - C.1)^2 + (y - C.2)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_l323_32313


namespace NUMINAMATH_CALUDE_smallest_k_for_f_iteration_zero_l323_32302

def f (a b M : ℕ) (n : ℤ) : ℤ :=
  if n < M then n + a else n - b

def iterateF (a b M : ℕ) : ℕ → ℤ → ℤ
  | 0, n => n
  | k+1, n => f a b M (iterateF a b M k n)

theorem smallest_k_for_f_iteration_zero (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ b) :
  let M := (a + b) / 2
  ∃ k : ℕ, k = (a + b) / Nat.gcd a b ∧ 
    iterateF a b M k 0 = 0 ∧ 
    ∀ j : ℕ, j < k → iterateF a b M j 0 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_f_iteration_zero_l323_32302


namespace NUMINAMATH_CALUDE_freshman_class_size_l323_32375

theorem freshman_class_size :
  ∃! n : ℕ, n < 700 ∧
    n % 20 = 19 ∧
    n % 25 = 24 ∧
    n % 9 = 3 ∧
    n = 399 := by sorry

end NUMINAMATH_CALUDE_freshman_class_size_l323_32375


namespace NUMINAMATH_CALUDE_stock_worth_l323_32388

theorem stock_worth (X : ℝ) : 
  (0.1 * X * 1.2 + 0.9 * X * 0.95 = X - 400) → X = 16000 := by sorry

end NUMINAMATH_CALUDE_stock_worth_l323_32388


namespace NUMINAMATH_CALUDE_prime_power_plus_three_l323_32341

theorem prime_power_plus_three (p : ℕ) : 
  Prime p → Prime (p^4 + 3) → p^5 + 3 = 35 := by sorry

end NUMINAMATH_CALUDE_prime_power_plus_three_l323_32341


namespace NUMINAMATH_CALUDE_ball_box_probabilities_l323_32303

/-- The number of different balls -/
def num_balls : ℕ := 4

/-- The number of different boxes -/
def num_boxes : ℕ := 4

/-- The total number of possible outcomes when placing balls into boxes -/
def total_outcomes : ℕ := num_boxes ^ num_balls

/-- The probability of no empty boxes when placing balls into boxes -/
def prob_no_empty_boxes : ℚ := 3 / 32

/-- The probability of exactly one empty box when placing balls into boxes -/
def prob_one_empty_box : ℚ := 9 / 16

/-- Theorem stating the probabilities for different scenarios when placing balls into boxes -/
theorem ball_box_probabilities :
  (prob_no_empty_boxes = 3 / 32) ∧ (prob_one_empty_box = 9 / 16) := by
  sorry

end NUMINAMATH_CALUDE_ball_box_probabilities_l323_32303


namespace NUMINAMATH_CALUDE_gcd_of_f_over_primes_ge_11_l323_32395

-- Define the function f(p)
def f (p : ℕ) : ℕ := p^6 - 7*p^2 + 6

-- Define the set of prime numbers greater than or equal to 11
def P : Set ℕ := {p : ℕ | Nat.Prime p ∧ p ≥ 11}

-- Theorem statement
theorem gcd_of_f_over_primes_ge_11 : 
  ∃ (d : ℕ), d > 0 ∧ (∀ (p : ℕ), p ∈ P → (f p).gcd d = d) ∧ 
  (∀ (m : ℕ), (∀ (p : ℕ), p ∈ P → (f p).gcd m = m) → m ≤ d) ∧ d = 16 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_f_over_primes_ge_11_l323_32395


namespace NUMINAMATH_CALUDE_teresa_total_score_l323_32311

def teresa_scores (science music social_studies : ℕ) : Prop :=
  ∃ (physics total : ℕ),
    physics = music / 2 ∧
    total = science + music + social_studies + physics

theorem teresa_total_score :
  teresa_scores 70 80 85 → ∃ total : ℕ, total = 275 :=
by sorry

end NUMINAMATH_CALUDE_teresa_total_score_l323_32311


namespace NUMINAMATH_CALUDE_fraction_of_decimals_cubed_and_squared_l323_32320

theorem fraction_of_decimals_cubed_and_squared :
  (0.3 ^ 3) / (0.03 ^ 2) = 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_decimals_cubed_and_squared_l323_32320


namespace NUMINAMATH_CALUDE_unique_sums_count_l323_32338

def bag_A : Finset ℕ := {1, 4, 9}
def bag_B : Finset ℕ := {16, 25, 36}

def possible_sums : Finset ℕ := (bag_A.product bag_B).image (fun p => p.1 + p.2)

theorem unique_sums_count : possible_sums.card = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_sums_count_l323_32338


namespace NUMINAMATH_CALUDE_apple_basket_problem_l323_32390

/-- The number of baskets in the apple-picking problem -/
def number_of_baskets : ℕ := 11

/-- The total number of apples initially -/
def total_apples : ℕ := 1000

/-- The number of apples left after picking -/
def apples_left : ℕ := 340

/-- The number of children picking apples -/
def number_of_children : ℕ := 10

/-- The sum of the first n natural numbers -/
def sum_of_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem apple_basket_problem :
  (number_of_children * sum_of_first_n number_of_baskets = total_apples - apples_left) ∧
  (number_of_baskets > 0) :=
sorry

end NUMINAMATH_CALUDE_apple_basket_problem_l323_32390


namespace NUMINAMATH_CALUDE_total_spent_is_211_20_l323_32328

/-- Calculates the total amount spent on a meal given the food price, sales tax rate, and tip rate. -/
def total_amount_spent (food_price : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) : ℝ :=
  let price_with_tax := food_price * (1 + sales_tax_rate)
  price_with_tax * (1 + tip_rate)

/-- Theorem stating that the total amount spent is $211.20 given the specified conditions. -/
theorem total_spent_is_211_20 :
  total_amount_spent 160 0.1 0.2 = 211.20 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_211_20_l323_32328


namespace NUMINAMATH_CALUDE_ellipse_and_line_properties_l323_32310

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- Definition of the line l -/
def line_l (x y m : ℝ) : Prop :=
  y = x + m

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_and_line_properties :
  ∃ (a b : ℝ),
    -- Foci conditions
    ((-1 : ℝ)^2 + 0^2 = a^2 - b^2) ∧
    ((1 : ℝ)^2 + 0^2 = a^2 - b^2) ∧
    -- Point P on the ellipse
    ellipse_C 1 (Real.sqrt 2 / 2) ∧
    -- Standard equation of the ellipse
    (∀ x y, ellipse_C x y ↔ x^2 / 2 + y^2 = 1) ∧
    -- Maximum intersection distance occurs when m = 0
    (∀ m, ∃ x₁ y₁ x₂ y₂,
      line_l x₁ y₁ m ∧ line_l x₂ y₂ m ∧
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 ≤ (2 : ℝ)^2 + (2 : ℝ)^2) ∧
    -- The line y = x achieves this maximum
    (∃ x₁ y₁ x₂ y₂,
      line_l x₁ y₁ 0 ∧ line_l x₂ y₂ 0 ∧
      ellipse_C x₁ y₁ ∧ ellipse_C x₂ y₂ ∧
      (x₂ - x₁)^2 + (y₂ - y₁)^2 = (2 : ℝ)^2 + (2 : ℝ)^2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_properties_l323_32310


namespace NUMINAMATH_CALUDE_max_value_sqrt_xy_1_minus_x_2y_l323_32359

theorem max_value_sqrt_xy_1_minus_x_2y :
  ∀ x y : ℝ, x > 0 → y > 0 →
  Real.sqrt (x * y) * (1 - x - 2 * y) ≤ Real.sqrt 2 / 16 ∧
  (Real.sqrt (x * y) * (1 - x - 2 * y) = Real.sqrt 2 / 16 ↔ x = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_xy_1_minus_x_2y_l323_32359


namespace NUMINAMATH_CALUDE_systematic_sampling_l323_32318

/-- Systematic sampling problem -/
theorem systematic_sampling
  (population_size : ℕ)
  (sample_size : ℕ)
  (last_sample : ℕ)
  (h1 : population_size = 2000)
  (h2 : sample_size = 100)
  (h3 : last_sample = 1994)
  : ∃ (first_sample : ℕ), first_sample = 14 ∧
    last_sample = (sample_size - 1) * (population_size / sample_size) + first_sample :=
sorry

end NUMINAMATH_CALUDE_systematic_sampling_l323_32318


namespace NUMINAMATH_CALUDE_ten_percent_increase_l323_32349

theorem ten_percent_increase (original : ℝ) (increased : ℝ) : 
  original = 600 → increased = original * 1.1 → increased = 660 := by
  sorry

end NUMINAMATH_CALUDE_ten_percent_increase_l323_32349


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l323_32377

theorem product_of_sum_and_sum_of_cubes (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (sum_cubes_eq : x^3 + y^3 = 370) : 
  x * y = 21 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l323_32377


namespace NUMINAMATH_CALUDE_milk_dilution_l323_32360

/-- Represents the milk dilution problem -/
theorem milk_dilution (initial_capacity : ℝ) (removal_amount : ℝ) : 
  initial_capacity = 45 →
  removal_amount = 9 →
  let first_milk_remaining := initial_capacity - removal_amount
  let first_mixture_milk_ratio := first_milk_remaining / initial_capacity
  let second_milk_remaining := first_milk_remaining - (first_mixture_milk_ratio * removal_amount)
  second_milk_remaining = 28.8 := by
  sorry

end NUMINAMATH_CALUDE_milk_dilution_l323_32360


namespace NUMINAMATH_CALUDE_area_of_triangle_ABF_l323_32305

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- A square defined by four points -/
structure Square where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Check if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Check if a point is inside a square -/
def isInside (p : Point) (s : Square) : Prop := sorry

/-- Find the intersection point of two line segments -/
def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

theorem area_of_triangle_ABF 
  (A B C D E F : Point)
  (square : Square)
  (triangle : Triangle)
  (h1 : square = Square.mk A B C D)
  (h2 : triangle = Triangle.mk A B E)
  (h3 : isEquilateral triangle)
  (h4 : isInside E square)
  (h5 : F = intersectionPoint B D A E)
  (h6 : (B.x - A.x)^2 + (B.y - A.y)^2 = 1 + Real.sqrt 3) :
  triangleArea (Triangle.mk A B F) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_ABF_l323_32305


namespace NUMINAMATH_CALUDE_anna_basketball_score_product_l323_32369

def first_10_games : List ℕ := [5, 7, 9, 2, 6, 10, 5, 7, 8, 4]

theorem anna_basketball_score_product :
  ∀ (game11 game12 : ℕ),
  game11 < 15 ∧ game12 < 15 →
  (List.sum first_10_games + game11) % 11 = 0 →
  (List.sum first_10_games + game11 + game12) % 12 = 0 →
  game11 * game12 = 18 := by
  sorry

end NUMINAMATH_CALUDE_anna_basketball_score_product_l323_32369


namespace NUMINAMATH_CALUDE_det_E_equals_25_l323_32336

/-- A 2x2 matrix representing a dilation by factor 5 centered at the origin -/
def D : Matrix (Fin 2) (Fin 2) ℝ := !![5, 0; 0, 5]

/-- A 2x2 matrix representing a 90-degree counter-clockwise rotation -/
def R : Matrix (Fin 2) (Fin 2) ℝ := !![0, -1; 1, 0]

/-- The combined transformation matrix E -/
def E : Matrix (Fin 2) (Fin 2) ℝ := R * D

theorem det_E_equals_25 : Matrix.det E = 25 := by
  sorry

end NUMINAMATH_CALUDE_det_E_equals_25_l323_32336


namespace NUMINAMATH_CALUDE_common_terms_count_l323_32362

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  start : ℝ
  diff : ℝ
  length : ℕ

/-- Returns the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.start + (n - 1 : ℝ) * seq.diff

/-- Counts the number of common terms between two arithmetic sequences -/
def countCommonTerms (seq1 seq2 : ArithmeticSequence) : ℕ :=
  (seq1.length).min seq2.length

theorem common_terms_count (seq1 seq2 : ArithmeticSequence) 
  (h1 : seq1.start = 5 ∧ seq1.diff = 3 ∧ seq1.length = 100)
  (h2 : seq2.start = 3 ∧ seq2.diff = 5 ∧ seq2.length = 100) :
  countCommonTerms seq1 seq2 = 20 := by
  sorry

#check common_terms_count

end NUMINAMATH_CALUDE_common_terms_count_l323_32362


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l323_32317

/-- Represents the amount of pizza eaten by each sibling -/
structure PizzaConsumption where
  alex : ℚ
  beth : ℚ
  cyril : ℚ
  dan : ℚ
  eve : ℚ

/-- Calculates the pizza consumption for each sibling based on the given conditions -/
def calculate_consumption : PizzaConsumption := {
  alex := 1/6,
  beth := 2/7,
  cyril := 1/3,
  dan := 1 - (1/6 + 2/7 + 1/3 + 1/8) - 2/168,
  eve := 1/8 + 2/168
}

/-- Represents the correct order of siblings based on pizza consumption -/
def correct_order := ["Cyril", "Beth", "Eve", "Alex", "Dan"]

/-- Theorem stating that the calculated consumption leads to the correct order -/
theorem pizza_consumption_order : 
  let c := calculate_consumption
  (c.cyril > c.beth) ∧ (c.beth > c.eve) ∧ (c.eve > c.alex) ∧ (c.alex > c.dan) := by
  sorry

#check pizza_consumption_order

end NUMINAMATH_CALUDE_pizza_consumption_order_l323_32317


namespace NUMINAMATH_CALUDE_complex_equation_solution_l323_32374

theorem complex_equation_solution :
  ∀ a : ℂ, (1 - I)^3 / (1 + I) = a + 3*I → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l323_32374


namespace NUMINAMATH_CALUDE_inequality_equivalence_l323_32342

theorem inequality_equivalence (x : ℝ) :
  (5 * x^2 + 20 * x - 34) / ((3 * x - 2) * (x - 5) * (x + 1)) < 2 ↔
  (-6 * x^3 + 27 * x^2 + 33 * x - 44) / ((3 * x - 2) * (x - 5) * (x + 1)) < 0 :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l323_32342


namespace NUMINAMATH_CALUDE_min_value_theorem_l323_32376

theorem min_value_theorem (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y ≤ 2) :
  ∃ (min_val : ℝ), min_val = (3 + 2 * Real.sqrt 2) / 4 ∧
  ∀ (z : ℝ), z = 2 / (x + 3 * y) + 1 / (x - y) → z ≥ min_val := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l323_32376


namespace NUMINAMATH_CALUDE_cereal_eating_time_l323_32394

/-- The time taken for two people to eat a certain amount of cereal together -/
def time_to_eat_together (fat_rate : ℚ) (thin_rate : ℚ) (amount : ℚ) : ℚ :=
  amount / (fat_rate + thin_rate)

/-- Theorem: Given the eating rates and amount of cereal, prove that it takes 45 minutes to finish -/
theorem cereal_eating_time :
  let fat_rate : ℚ := 1 / 15  -- Mr. Fat's eating rate in pounds per minute
  let thin_rate : ℚ := 1 / 45  -- Mr. Thin's eating rate in pounds per minute
  let amount : ℚ := 4  -- Amount of cereal in pounds
  time_to_eat_together fat_rate thin_rate amount = 45 := by
  sorry

#eval time_to_eat_together (1/15 : ℚ) (1/45 : ℚ) 4

end NUMINAMATH_CALUDE_cereal_eating_time_l323_32394


namespace NUMINAMATH_CALUDE_dog_care_time_ratio_l323_32356

/-- Proves the ratio of blow-drying time to bathing time for Marcus and his dog --/
theorem dog_care_time_ratio 
  (total_time : ℕ) 
  (bath_time : ℕ) 
  (walk_speed : ℚ) 
  (walk_distance : ℚ) 
  (h1 : total_time = 60) 
  (h2 : bath_time = 20) 
  (h3 : walk_speed = 6) 
  (h4 : walk_distance = 3) : 
  (total_time - bath_time - (walk_distance / walk_speed * 60).floor) * 2 = bath_time := by
sorry


end NUMINAMATH_CALUDE_dog_care_time_ratio_l323_32356


namespace NUMINAMATH_CALUDE_max_value_of_function_l323_32352

theorem max_value_of_function (x : ℝ) : 
  1 / (x^2 + 2) ≤ 1 / 2 ∧ ∃ y : ℝ, 1 / (y^2 + 2) = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l323_32352
