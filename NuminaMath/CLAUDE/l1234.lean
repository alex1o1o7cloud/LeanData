import Mathlib

namespace NUMINAMATH_CALUDE_apple_crate_weight_l1234_123433

/-- The weight of one original box of apples in kilograms. -/
def original_box_weight : ℝ := 35

/-- The number of crates in the original set. -/
def num_crates : ℕ := 7

/-- The amount of apples removed from each crate in kilograms. -/
def removed_weight : ℝ := 20

/-- The number of original crates that equal the weight of all crates after removal. -/
def equivalent_crates : ℕ := 3

theorem apple_crate_weight :
  num_crates * (original_box_weight - removed_weight) = equivalent_crates * original_box_weight :=
sorry

end NUMINAMATH_CALUDE_apple_crate_weight_l1234_123433


namespace NUMINAMATH_CALUDE_company_daily_production_l1234_123458

/-- Proves that a company producing enough bottles to fill 2000 cases, 
    where each case holds 25 bottles, produces 50000 bottles daily. -/
theorem company_daily_production 
  (bottles_per_case : ℕ) 
  (cases_per_day : ℕ) 
  (h1 : bottles_per_case = 25)
  (h2 : cases_per_day = 2000) :
  bottles_per_case * cases_per_day = 50000 := by
  sorry

#check company_daily_production

end NUMINAMATH_CALUDE_company_daily_production_l1234_123458


namespace NUMINAMATH_CALUDE_outfit_count_l1234_123402

/-- Represents the number of shirts of each color -/
def shirts_per_color : ℕ := 4

/-- Represents the number of pants -/
def pants : ℕ := 6

/-- Represents the number of hats of each color -/
def hats_per_color : ℕ := 8

/-- Represents the number of colors -/
def colors : ℕ := 3

/-- Theorem: The number of outfits with one shirt, one pair of pants, and one hat,
    where the shirt and hat are not the same color, is 1152 -/
theorem outfit_count : 
  shirts_per_color * (colors - 1) * hats_per_color * pants = 1152 := by
  sorry


end NUMINAMATH_CALUDE_outfit_count_l1234_123402


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l1234_123483

/-- A parabola y = ax^2 + 4x + 3 is tangent to the line y = 2x + 1 if and only if a = 1/2 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃ x : ℝ, ax^2 + 4*x + 3 = 2*x + 1 ∧ 
   ∀ y : ℝ, y ≠ x → ax^2 + 4*x + 3 ≠ 2*y + 1) ↔ 
  a = (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l1234_123483


namespace NUMINAMATH_CALUDE_congruence_and_divisibility_solutions_l1234_123452

theorem congruence_and_divisibility_solutions : 
  {x : ℤ | x^3 ≡ -1 [ZMOD 7] ∧ (7 : ℤ) ∣ (x^2 - x + 1)} = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_congruence_and_divisibility_solutions_l1234_123452


namespace NUMINAMATH_CALUDE_max_area_rectangular_pen_l1234_123437

/-- The maximum area of a rectangular pen with a perimeter of 60 feet -/
theorem max_area_rectangular_pen :
  let perimeter : ℝ := 60
  let area (x : ℝ) : ℝ := x * (perimeter / 2 - x)
  ∀ x, 0 < x → x < perimeter / 2 → area x ≤ 225 :=
by sorry

end NUMINAMATH_CALUDE_max_area_rectangular_pen_l1234_123437


namespace NUMINAMATH_CALUDE_cylinder_height_given_cone_volume_ratio_l1234_123460

theorem cylinder_height_given_cone_volume_ratio (base_area : ℝ) (cone_height : ℝ) :
  cone_height = 4.5 →
  (1 / 3 * base_area * cone_height) / (base_area * cylinder_height) = 1 / 6 →
  cylinder_height = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_height_given_cone_volume_ratio_l1234_123460


namespace NUMINAMATH_CALUDE_female_employees_with_advanced_degrees_l1234_123432

theorem female_employees_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_advanced_degrees : ℕ)
  (males_college_only : ℕ)
  (h1 : total_employees = 200)
  (h2 : total_females = 120)
  (h3 : total_advanced_degrees = 100)
  (h4 : males_college_only = 40) :
  total_advanced_degrees - (total_employees - total_females - males_college_only) = 60 :=
by sorry

end NUMINAMATH_CALUDE_female_employees_with_advanced_degrees_l1234_123432


namespace NUMINAMATH_CALUDE_point_line_range_l1234_123455

theorem point_line_range (a : ℝ) : 
  (∀ x y : ℝ, (x = -3 ∧ y = -1) ∨ (x = 4 ∧ y = -6) → 
    (3*(-3) - 2*(-1) - a) * (3*4 - 2*(-6) - a) < 0) ↔ 
  -7 < a ∧ a < 24 :=
sorry

end NUMINAMATH_CALUDE_point_line_range_l1234_123455


namespace NUMINAMATH_CALUDE_symmetric_function_intersection_l1234_123472

/-- Definition of a symmetric function -/
def symmetricFunction (m n : ℝ) : ℝ → ℝ := λ x ↦ n * x + m

/-- The given function -/
def givenFunction : ℝ → ℝ := λ x ↦ -6 * x + 4

/-- Theorem: The intersection point of the symmetric function of y=-6x+4 with the y-axis is (0, -6) -/
theorem symmetric_function_intersection :
  let f := symmetricFunction (-6) 4
  (0, f 0) = (0, -6) := by sorry

end NUMINAMATH_CALUDE_symmetric_function_intersection_l1234_123472


namespace NUMINAMATH_CALUDE_price_change_theorem_l1234_123415

/-- Proves that a price of $100 after a 10% increase followed by a 10% decrease results in $99 -/
theorem price_change_theorem (initial_price : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) : 
  initial_price = 100 ∧ 
  increase_rate = 0.1 ∧ 
  decrease_rate = 0.1 → 
  initial_price * (1 + increase_rate) * (1 - decrease_rate) = 99 :=
by
  sorry

#check price_change_theorem

end NUMINAMATH_CALUDE_price_change_theorem_l1234_123415


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1234_123434

/-- Given that the solution set of ax² - bx + c < 0 is (-2, 3), 
    prove that the solution set of bx² + ax + c < 0 is (-3, 2) -/
theorem solution_set_inequality (a b c : ℝ) : 
  (∀ x, ax^2 - b*x + c < 0 ↔ -2 < x ∧ x < 3) →
  (∀ x, b*x^2 + a*x + c < 0 ↔ -3 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1234_123434


namespace NUMINAMATH_CALUDE_sin_10_50_70_product_l1234_123426

theorem sin_10_50_70_product : Real.sin (10 * π / 180) * Real.sin (50 * π / 180) * Real.sin (70 * π / 180) = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_10_50_70_product_l1234_123426


namespace NUMINAMATH_CALUDE_unique_solution_xyz_l1234_123490

theorem unique_solution_xyz (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (h : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 12 ∧ y = 10 ∧ z = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_xyz_l1234_123490


namespace NUMINAMATH_CALUDE_diplomat_languages_l1234_123476

theorem diplomat_languages (total : ℕ) (french : ℕ) (not_russian : ℕ) (both_percent : ℚ) 
  (h_total : total = 180)
  (h_french : french = 14)
  (h_not_russian : not_russian = 32)
  (h_both_percent : both_percent = 1/10) : 
  (total - (french + (total - not_russian) - (both_percent * total))) / total = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_diplomat_languages_l1234_123476


namespace NUMINAMATH_CALUDE_octagon_non_intersecting_diagonals_l1234_123485

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The number of non-intersecting diagonals in a star pattern for a regular polygon -/
def nonIntersectingDiagonals (p : RegularPolygon n) : ℕ := n

/-- Theorem: For an octagon, the number of non-intersecting diagonals in a star pattern
    is equal to the number of sides -/
theorem octagon_non_intersecting_diagonals :
  ∀ (p : RegularPolygon 8), nonIntersectingDiagonals p = 8 := by
  sorry

end NUMINAMATH_CALUDE_octagon_non_intersecting_diagonals_l1234_123485


namespace NUMINAMATH_CALUDE_geq_three_necessary_not_sufficient_for_gt_three_l1234_123424

theorem geq_three_necessary_not_sufficient_for_gt_three :
  (∀ x : ℝ, x > 3 → x ≥ 3) ∧
  (∃ x : ℝ, x ≥ 3 ∧ ¬(x > 3)) := by
  sorry

end NUMINAMATH_CALUDE_geq_three_necessary_not_sufficient_for_gt_three_l1234_123424


namespace NUMINAMATH_CALUDE_identify_tasty_candies_l1234_123488

/-- Represents a candy on the table. -/
structure Candy where
  tasty : Bool

/-- Represents the state of the game. -/
structure GameState where
  candies : Finset Candy
  moves_left : Nat

/-- Represents a query about a subset of candies. -/
def Query := Finset Candy → Nat

/-- The main theorem stating that all tasty candies can be identified within the given number of moves. -/
theorem identify_tasty_candies 
  (n : Nat) 
  (candies : Finset Candy) 
  (h1 : candies.card = 28) 
  (query : Query) : 
  (∃ (strategy : GameState → Finset Candy), 
    (∀ (gs : GameState), 
      gs.candies = candies → 
      gs.moves_left ≥ 21 → 
      strategy gs = {c ∈ candies | c.tasty})) ∧ 
    (∃ (strategy : GameState → Finset Candy), 
      (∀ (gs : GameState), 
        gs.candies = candies → 
        gs.moves_left ≥ 20 → 
        strategy gs = {c ∈ candies | c.tasty})) :=
by sorry

end NUMINAMATH_CALUDE_identify_tasty_candies_l1234_123488


namespace NUMINAMATH_CALUDE_reflect_point_3_2_l1234_123435

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis in a Cartesian coordinate system -/
def reflect_x_axis (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

/-- Theorem: Reflecting the point (3,2) across the x-axis results in (3,-2) -/
theorem reflect_point_3_2 :
  reflect_x_axis ⟨3, 2⟩ = ⟨3, -2⟩ := by
  sorry

end NUMINAMATH_CALUDE_reflect_point_3_2_l1234_123435


namespace NUMINAMATH_CALUDE_complex_vector_difference_l1234_123481

theorem complex_vector_difference (z : ℂ) (h : z = 1 - I) :
  z^2 - z = -1 - I := by sorry

end NUMINAMATH_CALUDE_complex_vector_difference_l1234_123481


namespace NUMINAMATH_CALUDE_sequence_length_l1234_123499

/-- Calculates the number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- The number of terms in the arithmetic sequence from 5 to 200 with common difference 3 -/
theorem sequence_length : arithmeticSequenceLength 5 200 3 = 66 := by
  sorry

#eval arithmeticSequenceLength 5 200 3

end NUMINAMATH_CALUDE_sequence_length_l1234_123499


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l1234_123421

/-- If the solution set of the inequality |ax+2| < 6 is (-1,2), then a = -4 -/
theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |a*x + 2| < 6 ↔ -1 < x ∧ x < 2) →
  a = -4 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l1234_123421


namespace NUMINAMATH_CALUDE_horner_method_proof_l1234_123431

def f (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 6*x^3 - 2*x^2 - 5*x - 2

theorem horner_method_proof : f 5 = 7548 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l1234_123431


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1234_123454

theorem arithmetic_calculations : 
  ((-23) + 13 - 12 = -22) ∧ 
  ((-2)^3 / 4 + 3 * (-5) = -17) ∧ 
  ((-24) * (1/2 - 3/4 - 1/8) = 9) ∧ 
  ((2-7) / 5^2 + (-1)^2023 * (1/10) = -3/10) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1234_123454


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l1234_123491

theorem fraction_sum_simplification :
  3 / 462 + 17 / 42 = 95 / 231 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l1234_123491


namespace NUMINAMATH_CALUDE_existence_of_solution_l1234_123439

theorem existence_of_solution : ∃ (x y : ℤ), 2 * x^2 + 8 * y = 26 ∧ x - y = 26 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l1234_123439


namespace NUMINAMATH_CALUDE_office_population_l1234_123477

theorem office_population (men women : ℕ) : 
  men = women →
  6 = women / 5 →
  men + women = 60 := by
sorry

end NUMINAMATH_CALUDE_office_population_l1234_123477


namespace NUMINAMATH_CALUDE_infinitely_many_powers_of_two_l1234_123443

def lastDigit (n : ℕ) : ℕ := n % 10

def sequenceA : ℕ → ℕ
  | 0 => 0  -- This is a placeholder, as a₁ is actually the first term
  | n + 1 => sequenceA n + lastDigit (sequenceA n)

theorem infinitely_many_powers_of_two 
  (h₁ : sequenceA 1 % 5 ≠ 0)  -- a₁ is not divisible by 5
  (h₂ : ∀ n, sequenceA (n + 1) = sequenceA n + lastDigit (sequenceA n)) :
  ∀ k, ∃ n, ∃ m, sequenceA n = 2^m ∧ m ≥ k :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_powers_of_two_l1234_123443


namespace NUMINAMATH_CALUDE_b_completion_time_l1234_123422

/-- Represents the time (in days) it takes for a worker to complete a job alone -/
structure WorkerTime where
  days : ℝ
  days_pos : days > 0

/-- Represents the share of earnings for a worker -/
structure EarningShare where
  amount : ℝ
  amount_pos : amount > 0

/-- Represents a job with multiple workers -/
structure Job where
  a : WorkerTime
  c : WorkerTime
  total_earnings : ℝ
  b_share : EarningShare
  total_earnings_pos : total_earnings > 0

theorem b_completion_time (job : Job) 
  (ha : job.a.days = 6)
  (hc : job.c.days = 12)
  (htotal : job.total_earnings = 1170)
  (hb_share : job.b_share.amount = 390) :
  ∃ (b : WorkerTime), b.days = 8 := by
  sorry

end NUMINAMATH_CALUDE_b_completion_time_l1234_123422


namespace NUMINAMATH_CALUDE_vector_operation_result_unique_linear_combination_l1234_123487

-- Define the vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (5, 6)

-- Theorem for part 1
theorem vector_operation_result : 
  (3 • a) + b - (2 • c) = (-2, -4) := by sorry

-- Theorem for part 2
theorem unique_linear_combination :
  ∃! (m n : ℝ), c = m • a + n • b ∧ m = 2 ∧ n = 1 := by sorry

-- Note: • is used for scalar multiplication in Lean

end NUMINAMATH_CALUDE_vector_operation_result_unique_linear_combination_l1234_123487


namespace NUMINAMATH_CALUDE_division_remainder_l1234_123482

theorem division_remainder (k : ℕ) : 
  k > 0 ∧ k < 38 ∧ 
  k % 5 = 2 ∧ 
  (∃ n : ℕ, n > 5 ∧ k % n = 5) →
  k % 7 = 5 :=
by sorry

end NUMINAMATH_CALUDE_division_remainder_l1234_123482


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1234_123494

/-- Given a triangle with inradius 2.5 cm and area 35 cm², its perimeter is 28 cm. -/
theorem triangle_perimeter (r : ℝ) (A : ℝ) (p : ℝ) 
  (h1 : r = 2.5)
  (h2 : A = 35)
  (h3 : A = r * (p / 2)) :
  p = 28 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1234_123494


namespace NUMINAMATH_CALUDE_program_requires_eight_sessions_l1234_123438

/-- Calculates the number of seating sessions required for a group -/
def sessionsRequired (groupSize : ℕ) (capacity : ℕ) : ℕ :=
  (groupSize + capacity - 1) / capacity

/-- Represents the seating program -/
structure SeatingProgram where
  totalParents : ℕ
  totalPupils : ℕ
  capacity : ℕ
  parentsMorning : ℕ
  parentsAfternoon : ℕ
  pupilsMorning : ℕ
  pupilsMidDay : ℕ
  pupilsEvening : ℕ

/-- Calculates the total number of seating sessions required -/
def totalSessions (program : SeatingProgram) : ℕ :=
  sessionsRequired program.parentsMorning program.capacity +
  sessionsRequired program.parentsAfternoon program.capacity +
  sessionsRequired program.pupilsMorning program.capacity +
  sessionsRequired program.pupilsMidDay program.capacity +
  sessionsRequired program.pupilsEvening program.capacity

/-- Theorem stating that the given program requires 8 seating sessions -/
theorem program_requires_eight_sessions (program : SeatingProgram)
  (h1 : program.totalParents = 61)
  (h2 : program.totalPupils = 177)
  (h3 : program.capacity = 44)
  (h4 : program.parentsMorning = 35)
  (h5 : program.parentsAfternoon = 26)
  (h6 : program.pupilsMorning = 65)
  (h7 : program.pupilsMidDay = 57)
  (h8 : program.pupilsEvening = 55)
  : totalSessions program = 8 := by
  sorry


end NUMINAMATH_CALUDE_program_requires_eight_sessions_l1234_123438


namespace NUMINAMATH_CALUDE_boys_percentage_in_specific_classroom_l1234_123449

/-- Represents the composition of a classroom -/
structure Classroom where
  total_people : ℕ
  boy_girl_ratio : ℚ
  student_teacher_ratio : ℕ

/-- Calculates the percentage of boys in the classroom -/
def boys_percentage (c : Classroom) : ℚ :=
  sorry

/-- Theorem stating the percentage of boys in the specific classroom scenario -/
theorem boys_percentage_in_specific_classroom :
  let c : Classroom := {
    total_people := 36,
    boy_girl_ratio := 2 / 3,
    student_teacher_ratio := 6
  }
  boys_percentage c = 400 / 7 := by
  sorry

end NUMINAMATH_CALUDE_boys_percentage_in_specific_classroom_l1234_123449


namespace NUMINAMATH_CALUDE_degree_of_example_monomial_l1234_123417

/-- Represents a monomial with coefficient and exponents for x and y -/
structure Monomial where
  coeff : ℤ
  x_exp : ℕ
  y_exp : ℕ

/-- Calculates the degree of a monomial -/
def degree (m : Monomial) : ℕ := m.x_exp + m.y_exp

/-- The specific monomial -5x^2y -/
def example_monomial : Monomial := ⟨-5, 2, 1⟩

theorem degree_of_example_monomial :
  degree example_monomial = 3 := by sorry

end NUMINAMATH_CALUDE_degree_of_example_monomial_l1234_123417


namespace NUMINAMATH_CALUDE_smallest_n_with_common_divisors_l1234_123419

def M : ℕ := 30030

theorem smallest_n_with_common_divisors (n : ℕ) : n = 9440 ↔ 
  (∀ k : ℕ, k ≤ 20 → ∃ d : ℕ, d > 1 ∧ d ∣ (n + k) ∧ d ∣ M) ∧
  (∀ m : ℕ, m < n → ∃ k : ℕ, k ≤ 20 ∧ ∀ d : ℕ, d > 1 → d ∣ (m + k) → ¬(d ∣ M)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_common_divisors_l1234_123419


namespace NUMINAMATH_CALUDE_inequality_proof_l1234_123444

theorem inequality_proof (x y : ℝ) (n k : ℕ) 
  (h1 : x > y) (h2 : y > 0) (h3 : n > k) : 
  (x^k - y^k)^n < (x^n - y^n)^k := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1234_123444


namespace NUMINAMATH_CALUDE_cubic_projection_equality_l1234_123405

/-- A cubic function -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- Theorem: For a cubic function and two horizontal lines intersecting it, 
    the difference between the middle x-coordinates equals the sum of the 
    differences between the outer x-coordinates. -/
theorem cubic_projection_equality 
  (a b c d : ℝ) 
  (x₁ x₂ x₃ X₁ X₂ X₃ : ℝ) 
  (y₁ y₂ Y₁ Y₂ : ℝ) 
  (h₁ : x₁ < x₂) (h₂ : x₂ < x₃) 
  (h₃ : X₁ < X₂) (h₄ : X₂ < X₃) 
  (h₅ : cubic_function a b c d x₁ = y₁) 
  (h₆ : cubic_function a b c d x₂ = y₁) 
  (h₇ : cubic_function a b c d x₃ = y₁) 
  (h₈ : cubic_function a b c d X₁ = Y₁) 
  (h₉ : cubic_function a b c d X₂ = Y₁) 
  (h₁₀ : cubic_function a b c d X₃ = Y₁) :
  x₂ - X₂ = (X₁ - x₁) + (X₃ - x₃) := by sorry

end NUMINAMATH_CALUDE_cubic_projection_equality_l1234_123405


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1234_123464

def U : Set ℕ := {2011, 2012, 2013, 2014, 2015}
def M : Set ℕ := {2011, 2012, 2013}

theorem complement_of_M_in_U : U \ M = {2014, 2015} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1234_123464


namespace NUMINAMATH_CALUDE_tommy_savings_needed_l1234_123496

def number_of_books : ℕ := 8
def cost_per_book : ℕ := 5
def current_savings : ℕ := 13

theorem tommy_savings_needed : 
  number_of_books * cost_per_book - current_savings = 27 := by
  sorry

end NUMINAMATH_CALUDE_tommy_savings_needed_l1234_123496


namespace NUMINAMATH_CALUDE_sum_of_five_unit_fractions_l1234_123465

theorem sum_of_five_unit_fractions :
  ∃ (a b c d e : ℕ+), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                       b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                       c ≠ d ∧ c ≠ e ∧ 
                       d ≠ e ∧
                       (1 : ℚ) = 1 / a + 1 / b + 1 / c + 1 / d + 1 / e :=
sorry

end NUMINAMATH_CALUDE_sum_of_five_unit_fractions_l1234_123465


namespace NUMINAMATH_CALUDE_lap_time_six_minutes_l1234_123400

/-- Represents a circular track with two photographers -/
structure CircularTrack :=
  (length : ℝ)
  (photographer1_position : ℝ)
  (photographer2_position : ℝ)

/-- Represents a runner on the circular track -/
structure Runner :=
  (speed : ℝ)
  (start_position : ℝ)

/-- Calculates the time spent closer to each photographer -/
def time_closer_to_photographer (track : CircularTrack) (runner : Runner) : ℝ × ℝ := sorry

/-- The main theorem to prove -/
theorem lap_time_six_minutes 
  (track : CircularTrack) 
  (runner : Runner) 
  (h1 : (time_closer_to_photographer track runner).1 = 2)
  (h2 : (time_closer_to_photographer track runner).2 = 3) :
  runner.speed * track.length = 6 * runner.speed := by sorry

end NUMINAMATH_CALUDE_lap_time_six_minutes_l1234_123400


namespace NUMINAMATH_CALUDE_standing_men_ratio_l1234_123410

theorem standing_men_ratio (total_passengers : ℕ) (seated_men : ℕ) : 
  total_passengers = 48 →
  seated_men = 14 →
  (standing_men : ℚ) / (total_men : ℚ) = 1 / 8 :=
by
  intros h_total h_seated
  sorry
where
  women := (2 : ℚ) / 3 * total_passengers
  total_men := total_passengers - women
  standing_men := total_men - seated_men

end NUMINAMATH_CALUDE_standing_men_ratio_l1234_123410


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1234_123453

def f (a : ℝ) (x : ℝ) := x^2 - 2*a*x

theorem sufficient_not_necessary (a : ℝ) :
  (a < 0 → ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∃ a, 0 ≤ a ∧ a ≤ 1 ∧ ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → f a x₁ < f a x₂) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1234_123453


namespace NUMINAMATH_CALUDE_sum_of_cubes_l1234_123468

theorem sum_of_cubes (p q r : ℝ) 
  (sum_eq : p + q + r = 4)
  (sum_prod_eq : p * q + p * r + q * r = 3)
  (prod_eq : p * q * r = -6) :
  p^3 + q^3 + r^3 = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l1234_123468


namespace NUMINAMATH_CALUDE_parabola_intercept_sum_l1234_123447

/-- Parabola equation -/
def parabola (y : ℝ) : ℝ := y^2 - 4*y + 4

/-- X-intercept of the parabola -/
def a : ℝ := parabola 0

/-- Y-intercepts of the parabola -/
def b_and_c : Set ℝ := {y | parabola y = 0}

theorem parabola_intercept_sum :
  ∃ (b c : ℝ), b ∈ b_and_c ∧ c ∈ b_and_c ∧ a + b + c = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_intercept_sum_l1234_123447


namespace NUMINAMATH_CALUDE_quadratic_equation_sum_product_l1234_123480

theorem quadratic_equation_sum_product (m p : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + p = 0 ∧ 3 * y^2 - m * y + p = 0 ∧ x + y = 9 ∧ x * y = 14) →
  m + p = 69 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_sum_product_l1234_123480


namespace NUMINAMATH_CALUDE_pigeon_problem_l1234_123401

theorem pigeon_problem (x y : ℕ) : 
  (y + 1 = (1/6) * (x + y + 1)) → 
  (x - 1 = y + 1) → 
  (x = 4 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_pigeon_problem_l1234_123401


namespace NUMINAMATH_CALUDE_line_parallel_to_countless_lines_l1234_123469

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines
variable (parallel : Line → Line → Prop)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- Define the parallelism relation between a line and a plane
variable (parallelToPlane : Line → Plane → Prop)

-- Define a function that checks if a line is parallel to countless lines in a plane
variable (parallelToCountlessLines : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_countless_lines
  (l b : Line) (α : Plane)
  (h1 : parallel l b)
  (h2 : subset b α) :
  parallelToCountlessLines l α :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_countless_lines_l1234_123469


namespace NUMINAMATH_CALUDE_sphere_ratio_surface_to_volume_l1234_123493

theorem sphere_ratio_surface_to_volume 
  (s₁ s₂ : ℝ) (v₁ v₂ : ℝ) (h : s₁ / s₂ = 1 / 3) :
  v₁ / v₂ = 1 / (3 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_sphere_ratio_surface_to_volume_l1234_123493


namespace NUMINAMATH_CALUDE_first_two_valid_numbers_l1234_123492

/-- Represents a sequence of digits -/
def DigitSequence := List Nat

/-- Checks if a number is within the valid range for bag numbering -/
def isValidBagNumber (n : Nat) : Prop := 1 ≤ n ∧ n ≤ 850

/-- Extracts the first valid bag number from a digit sequence -/
def extractFirstValidNumber (seq : DigitSequence) : Option Nat :=
  sorry

/-- The sequence of digits starting from the given point in the random number table -/
def randomSequence : DigitSequence :=
  [7, 8, 5, 9, 1, 6, 9, 5, 5, 5, 6, 7, 1, 9, 9, 8, 1, 0, 5, 0, 7, 1, 7, 5]

theorem first_two_valid_numbers :
  let firstNumber := extractFirstValidNumber randomSequence
  let remainingSequence := randomSequence.drop 3  -- Drop the first 3 digits (785)
  let secondNumber := extractFirstValidNumber remainingSequence
  firstNumber = some 785 ∧ secondNumber = some 567 :=
sorry

end NUMINAMATH_CALUDE_first_two_valid_numbers_l1234_123492


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1234_123446

/-- The curve function -/
def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 - 18 * x + 7

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 12 * x - 18

/-- The point of tangency -/
def p : ℝ × ℝ := (-2, 3)

/-- The slope of the tangent line at the point of tangency -/
def m : ℝ := f' p.1

theorem tangent_line_equation :
  ∀ x y : ℝ, y = f p.1 → (y - p.2 = m * (x - p.1)) ↔ (30 * x - y + 63 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1234_123446


namespace NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l1234_123440

theorem fraction_division (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) (h3 : c ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by sorry

theorem simplify_fraction_division :
  (5 : ℚ) / 6 / ((9 : ℚ) / 10) = 25 / 27 :=
by sorry

end NUMINAMATH_CALUDE_fraction_division_simplify_fraction_division_l1234_123440


namespace NUMINAMATH_CALUDE_removed_number_is_1011_l1234_123436

/-- The sum of the first n natural numbers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem stating that removing 1011 from the set {1, 2, ..., 2021}
    makes the sum of the remaining numbers divisible by 2022 -/
theorem removed_number_is_1011 :
  ∀ x : ℕ, x ≤ 2021 →
    (sum_first_n 2021 - x) % 2022 = 0 → x = 1011 := by
  sorry

#check removed_number_is_1011

end NUMINAMATH_CALUDE_removed_number_is_1011_l1234_123436


namespace NUMINAMATH_CALUDE_max_annual_profit_l1234_123463

/-- Additional investment function R -/
noncomputable def R (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then 10 * x^2 + 300 * x
  else (901 * x^2 - 9450 * x + 10000) / x

/-- Annual profit function W -/
noncomputable def W (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 40 then -10 * x^2 + 600 * x - 260
  else -x - 10000 / x + 9190

/-- Theorem stating the maximum annual profit and corresponding production volume -/
theorem max_annual_profit :
  ∃ (x : ℝ), x = 100 ∧ W x = 8990 ∧ ∀ y, W y ≤ W x :=
sorry

end NUMINAMATH_CALUDE_max_annual_profit_l1234_123463


namespace NUMINAMATH_CALUDE_cos_two_thirds_pi_plus_two_alpha_l1234_123412

theorem cos_two_thirds_pi_plus_two_alpha (α : Real) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  Real.cos (2 * π / 3 + 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_two_thirds_pi_plus_two_alpha_l1234_123412


namespace NUMINAMATH_CALUDE_sandy_change_is_13_5_l1234_123425

/-- Represents the prices and quantities of drinks in Sandy's order -/
structure DrinkOrder where
  cappuccino_price : ℝ
  iced_tea_price : ℝ
  cafe_latte_price : ℝ
  espresso_price : ℝ
  mocha_price : ℝ
  hot_chocolate_price : ℝ
  cappuccino_qty : ℕ
  iced_tea_qty : ℕ
  cafe_latte_qty : ℕ
  espresso_qty : ℕ
  mocha_qty : ℕ
  hot_chocolate_qty : ℕ

/-- Calculates the total cost of the drink order -/
def total_cost (order : DrinkOrder) : ℝ :=
  order.cappuccino_price * order.cappuccino_qty +
  order.iced_tea_price * order.iced_tea_qty +
  order.cafe_latte_price * order.cafe_latte_qty +
  order.espresso_price * order.espresso_qty +
  order.mocha_price * order.mocha_qty +
  order.hot_chocolate_price * order.hot_chocolate_qty

/-- Calculates the change from a given payment amount -/
def calculate_change (payment : ℝ) (order : DrinkOrder) : ℝ :=
  payment - total_cost order

/-- Theorem stating that Sandy's change is $13.5 -/
theorem sandy_change_is_13_5 :
  let sandy_order : DrinkOrder := {
    cappuccino_price := 2,
    iced_tea_price := 3,
    cafe_latte_price := 1.5,
    espresso_price := 1,
    mocha_price := 2.5,
    hot_chocolate_price := 2,
    cappuccino_qty := 4,
    iced_tea_qty := 3,
    cafe_latte_qty := 5,
    espresso_qty := 3,
    mocha_qty := 2,
    hot_chocolate_qty := 2
  }
  calculate_change 50 sandy_order = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_is_13_5_l1234_123425


namespace NUMINAMATH_CALUDE_inequality_solution_l1234_123470

-- Define the parameter a
variable (a : ℝ)

-- Define the condition a < -1
variable (h : a < -1)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x ∈ (Set.Iio (-1) ∪ Set.Ioi (1/a))

-- State the theorem
theorem inequality_solution :
  ∀ x, (a * x - 1) / (x + 1) < 0 ↔ solution_set a x :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1234_123470


namespace NUMINAMATH_CALUDE_max_distance_line_theorem_l1234_123414

/-- The line equation that passes through point A(1, 2) and is at the maximum distance from the origin -/
def max_distance_line : ℝ → ℝ → Prop :=
  fun x y => x + 2 * y - 5 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (1, 2)

/-- The origin -/
def origin : ℝ × ℝ := (0, 0)

theorem max_distance_line_theorem :
  (max_distance_line (point_A.1) (point_A.2)) ∧
  (∀ x y, max_distance_line x y → 
    ∀ a b, (a, b) ≠ origin → 
      (a - origin.1)^2 + (b - origin.2)^2 ≤ (x - origin.1)^2 + (y - origin.2)^2) :=
sorry

end NUMINAMATH_CALUDE_max_distance_line_theorem_l1234_123414


namespace NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1234_123430

theorem product_of_repeating_decimal_and_eight :
  ∃ (x : ℚ), (∃ (n : ℕ), x = (456 : ℚ) / (10^n - 1)) ∧ 8 * x = 1216 / 333 := by
  sorry

end NUMINAMATH_CALUDE_product_of_repeating_decimal_and_eight_l1234_123430


namespace NUMINAMATH_CALUDE_meeting_percentage_is_37_5_percent_l1234_123411

def total_work_hours : ℕ := 8
def first_meeting_duration : ℕ := 30
def minutes_per_hour : ℕ := 60

def total_work_minutes : ℕ := total_work_hours * minutes_per_hour
def second_meeting_duration : ℕ := 2 * first_meeting_duration
def third_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration
def total_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration + third_meeting_duration

theorem meeting_percentage_is_37_5_percent :
  (total_meeting_duration : ℚ) / (total_work_minutes : ℚ) * 100 = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_37_5_percent_l1234_123411


namespace NUMINAMATH_CALUDE_ball_distribution_problem_l1234_123498

/-- Represents the number of ways to distribute balls among boxes -/
def distribute_balls (total_balls : ℕ) (num_boxes : ℕ) (min_balls : ℕ → ℕ) : ℕ :=
  sorry

/-- The specific problem setup -/
def problem_setup : ℕ × ℕ × (ℕ → ℕ) :=
  (9, 3, fun i => i)

theorem ball_distribution_problem :
  let (total_balls, num_boxes, min_balls) := problem_setup
  distribute_balls total_balls num_boxes min_balls = 10 := by
  sorry

end NUMINAMATH_CALUDE_ball_distribution_problem_l1234_123498


namespace NUMINAMATH_CALUDE_line_through_midpoint_l1234_123448

-- Define the points
def P : ℝ × ℝ := (1, 3)
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 6)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 3 * x + y - 6 = 0

-- Theorem statement
theorem line_through_midpoint :
  (P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →  -- P is midpoint of AB
  (A.2 = 0) →  -- A is on x-axis
  (B.1 = 0) →  -- B is on y-axis
  (line_equation P.1 P.2) →  -- Line passes through P
  (∀ x y, line_equation x y ↔ 3 * x + y - 6 = 0) :=  -- Prove the line equation
by sorry

end NUMINAMATH_CALUDE_line_through_midpoint_l1234_123448


namespace NUMINAMATH_CALUDE_fraction_sum_proof_l1234_123461

theorem fraction_sum_proof (fractions : Finset ℚ) 
  (h1 : fractions.card = 9)
  (h2 : ∀ f ∈ fractions, ∃ n : ℕ+, f = 1 / n)
  (h3 : (fractions.sum id) = 1)
  (h4 : (1 / 3) ∈ fractions ∧ (1 / 7) ∈ fractions ∧ (1 / 9) ∈ fractions ∧ 
        (1 / 11) ∈ fractions ∧ (1 / 33) ∈ fractions)
  (h5 : ∃ f1 f2 f3 f4 : ℚ, f1 ∈ fractions ∧ f2 ∈ fractions ∧ f3 ∈ fractions ∧ f4 ∈ fractions ∧
        ∃ n1 n2 n3 n4 : ℕ, f1 = 1 / n1 ∧ f2 = 1 / n2 ∧ f3 = 1 / n3 ∧ f4 = 1 / n4 ∧
        n1 % 10 = 5 ∧ n2 % 10 = 5 ∧ n3 % 10 = 5 ∧ n4 % 10 = 5) :
  ∃ f1 f2 f3 f4 : ℚ, f1 ∈ fractions ∧ f2 ∈ fractions ∧ f3 ∈ fractions ∧ f4 ∈ fractions ∧
  f1 = 1 / 5 ∧ f2 = 1 / 15 ∧ f3 = 1 / 45 ∧ f4 = 1 / 385 :=
by sorry

end NUMINAMATH_CALUDE_fraction_sum_proof_l1234_123461


namespace NUMINAMATH_CALUDE_min_vertical_distance_l1234_123486

-- Define the two functions
def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := -x^2 - 4*x - 3

-- Define the vertical distance between the two functions
def vertical_distance (x : ℝ) : ℝ := f x - g x

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x : ℝ), vertical_distance x = 8 ∧ 
  ∀ (y : ℝ), vertical_distance y ≥ 8 := by
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l1234_123486


namespace NUMINAMATH_CALUDE_x_squared_less_than_abs_x_l1234_123457

theorem x_squared_less_than_abs_x (x : ℝ) :
  x^2 < |x| ↔ (-1 < x ∧ x < 0) ∨ (0 < x ∧ x < 1) := by
  sorry

end NUMINAMATH_CALUDE_x_squared_less_than_abs_x_l1234_123457


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1234_123459

theorem geometric_sequence_ratio_sum 
  (m x y : ℝ) 
  (h_m : m ≠ 0) 
  (h_x_ne_y : x ≠ y) 
  (h_nonconstant : x ≠ 1 ∧ y ≠ 1) 
  (h_eq : m * x^2 - m * y^2 = 3 * (m * x - m * y)) : 
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_sum_l1234_123459


namespace NUMINAMATH_CALUDE_number_125_with_digit_sum_5_l1234_123495

/-- A function that calculates the sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- A function that returns the nth number in the sequence of natural numbers with digit sum 5 -/
def nthNumberWithDigitSum5 (n : ℕ) : ℕ := sorry

/-- The theorem stating that the 125th number in the sequence is 41000 -/
theorem number_125_with_digit_sum_5 : nthNumberWithDigitSum5 125 = 41000 := by sorry

end NUMINAMATH_CALUDE_number_125_with_digit_sum_5_l1234_123495


namespace NUMINAMATH_CALUDE_sequence_sum_property_l1234_123403

theorem sequence_sum_property (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → S n + (1 + 2 / n) * a n = 4) →
  (∀ n : ℕ, n > 0 → a n = n / (2^(n-1))) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_property_l1234_123403


namespace NUMINAMATH_CALUDE_line_slope_is_one_l1234_123474

theorem line_slope_is_one : 
  let line_eq := fun (x y : ℝ) => x - y + 1 = 0
  ∃ m : ℝ, (∀ x₁ y₁ x₂ y₂ : ℝ, 
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧ x₁ ≠ x₂ → 
    m = (y₂ - y₁) / (x₂ - x₁)) ∧ m = 1 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_is_one_l1234_123474


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l1234_123479

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 101 ways to distribute 7 indistinguishable balls into 4 distinguishable boxes -/
theorem seven_balls_four_boxes : distribute_balls 7 4 = 101 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l1234_123479


namespace NUMINAMATH_CALUDE_simplify_expressions_l1234_123467

theorem simplify_expressions (a x : ℝ) :
  (-a^3 + (-4*a^2)*a = -5*a^3) ∧
  (-x^2 * (-x)^2 * (-x^2)^3 - 2*x^10 = -x^10) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l1234_123467


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1234_123413

theorem quadratic_inequality (x : ℝ) : x^2 - 42*x + 400 ≤ 10 ↔ 13 ≤ x ∧ x ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1234_123413


namespace NUMINAMATH_CALUDE_helens_hotdogs_count_l1234_123427

/-- The number of hotdogs Dylan's mother brought -/
def dylans_hotdogs : ℕ := 379

/-- The total number of hotdogs -/
def total_hotdogs : ℕ := 480

/-- The number of hotdogs Helen's mother brought -/
def helens_hotdogs : ℕ := total_hotdogs - dylans_hotdogs

theorem helens_hotdogs_count : helens_hotdogs = 101 := by
  sorry

end NUMINAMATH_CALUDE_helens_hotdogs_count_l1234_123427


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1234_123462

theorem quadratic_inequality_solution_set (m : ℝ) :
  m > 2 → ∀ x : ℝ, x^2 - 2*x + m > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1234_123462


namespace NUMINAMATH_CALUDE_average_of_solutions_is_zero_l1234_123478

theorem average_of_solutions_is_zero : 
  let solutions := {x : ℝ | Real.sqrt (3 * x^2 + 4) = Real.sqrt 28}
  ∃ (a b : ℝ), a ∈ solutions ∧ b ∈ solutions ∧ a ≠ b ∧
  (∀ x, x ∈ solutions → x = a ∨ x = b) ∧
  (a + b) / 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_average_of_solutions_is_zero_l1234_123478


namespace NUMINAMATH_CALUDE_M_minus_N_equals_closed_open_l1234_123409

-- Definition of set difference
def set_difference (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∉ B}

-- Definition of set M
def M : Set ℝ := {x | |x + 1| ≤ 2}

-- Definition of set N
def N : Set ℝ := {x | ∃ α : ℝ, x = |Real.sin α|}

-- Theorem statement
theorem M_minus_N_equals_closed_open :
  set_difference M N = Set.Ico (-3) 0 := by sorry

end NUMINAMATH_CALUDE_M_minus_N_equals_closed_open_l1234_123409


namespace NUMINAMATH_CALUDE_sum_first_23_equals_11_l1234_123416

def repeatingSequence : List Int := [4, -3, 2, -1, 0]

def sumFirstN (n : Nat) : Int :=
  let fullCycles := n / repeatingSequence.length
  let remainder := n % repeatingSequence.length
  fullCycles * repeatingSequence.sum +
    (repeatingSequence.take remainder).sum

theorem sum_first_23_equals_11 : sumFirstN 23 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_23_equals_11_l1234_123416


namespace NUMINAMATH_CALUDE_tan_x_minus_pi_fourth_l1234_123484

theorem tan_x_minus_pi_fourth (x : ℝ) 
  (h1 : x ∈ Set.Ioo 0 π) 
  (h2 : Real.cos (2 * x - π / 2) = Real.sin x ^ 2) : 
  Real.tan (x - π / 4) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_minus_pi_fourth_l1234_123484


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_four_l1234_123406

/-- Two-digit number represented as a pair of digits -/
def TwoDigitNumber := Nat × Nat

/-- Sum of digits of two two-digit numbers -/
def sumOfDigits (n1 n2 : TwoDigitNumber) : Nat :=
  n1.1 + n1.2 + n2.1 + n2.2

/-- Result of adding two two-digit numbers -/
def addTwoDigitNumbers (n1 n2 : TwoDigitNumber) : Nat × Nat × Nat :=
  let sum := n1.1 * 10 + n1.2 + n2.1 * 10 + n2.2
  (sum / 100, (sum / 10) % 10, sum % 10)

theorem sum_of_x_and_y_is_four (n1 n2 : TwoDigitNumber) :
  sumOfDigits n1 n2 = 22 →
  (addTwoDigitNumbers n1 n2).2.2 = 9 →
  (addTwoDigitNumbers n1 n2).1 + (addTwoDigitNumbers n1 n2).2.1 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_four_l1234_123406


namespace NUMINAMATH_CALUDE_inscribed_circumscribed_ratio_l1234_123442

/-- An equilateral triangle with inscribed and circumscribed circles -/
structure EquilateralTriangleWithCircles where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The radius of the circumscribed circle -/
  R : ℝ
  /-- The radius of the inscribed circle is positive -/
  r_pos : r > 0
  /-- The radius of the circumscribed circle is positive -/
  R_pos : R > 0

/-- The ratio of the inscribed circle radius to the circumscribed circle radius in an equilateral triangle is 1:2 -/
theorem inscribed_circumscribed_ratio (t : EquilateralTriangleWithCircles) : t.r / t.R = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_inscribed_circumscribed_ratio_l1234_123442


namespace NUMINAMATH_CALUDE_fair_tickets_sold_l1234_123408

theorem fair_tickets_sold (baseball_tickets : ℕ) (fair_tickets : ℕ) : 
  baseball_tickets = 56 → 
  fair_tickets = 2 * baseball_tickets + 6 →
  fair_tickets = 118 := by
sorry

end NUMINAMATH_CALUDE_fair_tickets_sold_l1234_123408


namespace NUMINAMATH_CALUDE_seventh_term_ratio_l1234_123404

/-- Two arithmetic sequences with their sum ratios -/
structure ArithmeticSequences where
  a : ℕ → ℚ  -- First sequence
  b : ℕ → ℚ  -- Second sequence
  S : ℕ → ℚ  -- Sum of first n terms of sequence a
  T : ℕ → ℚ  -- Sum of first n terms of sequence b
  sum_ratio : ∀ n, S n / T n = (7 * n + 2) / (n + 3)
  arithmetic_a : ∀ n m, a (n + m) - a n = m * (a 2 - a 1)
  arithmetic_b : ∀ n m, b (n + m) - b n = m * (b 2 - b 1)
  sum_formula_a : ∀ n, S n = n * (a 1 + a n) / 2
  sum_formula_b : ∀ n, T n = n * (b 1 + b n) / 2

/-- The ratio of the 7th terms equals 93/16 -/
theorem seventh_term_ratio (seq : ArithmeticSequences) : seq.a 7 / seq.b 7 = 93 / 16 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_ratio_l1234_123404


namespace NUMINAMATH_CALUDE_x_twelfth_power_l1234_123423

theorem x_twelfth_power (x : ℂ) : x + 1/x = 2 * Real.sqrt 2 → x^12 = -4096 := by
  sorry

end NUMINAMATH_CALUDE_x_twelfth_power_l1234_123423


namespace NUMINAMATH_CALUDE_square_area_error_l1234_123456

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.02)
  let actual_area := s^2
  let calculated_area := measured_side^2
  (calculated_area - actual_area) / actual_area * 100 = 4.04 := by
  sorry

end NUMINAMATH_CALUDE_square_area_error_l1234_123456


namespace NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l1234_123473

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : Type := Unit

/-- The number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1s in the first n rows of Pascal's Triangle -/
def numberOfOnes (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def probabilityOfOne (n : ℕ) : ℚ := (numberOfOnes n : ℚ) / (totalElements n : ℚ)

theorem probability_of_one_in_20_rows :
  probabilityOfOne 20 = 13 / 70 := by sorry

end NUMINAMATH_CALUDE_probability_of_one_in_20_rows_l1234_123473


namespace NUMINAMATH_CALUDE_appliance_final_cost_l1234_123497

theorem appliance_final_cost (initial_price : ℝ) : 
  initial_price * 1.4 = 1680 →
  (1680 * 0.8) * 0.9 = 1209.6 :=
by sorry

end NUMINAMATH_CALUDE_appliance_final_cost_l1234_123497


namespace NUMINAMATH_CALUDE_company_sampling_methods_l1234_123428

/-- Enumeration of regions --/
inductive Region
| A
| B
| C
| D

/-- Enumeration of sampling methods --/
inductive SamplingMethod
| StratifiedSampling
| SimpleRandomSampling

/-- Structure representing the sales points distribution --/
structure SalesDistribution where
  total_points : ℕ
  region_points : Region → ℕ
  large_points_C : ℕ

/-- Structure representing an investigation --/
structure Investigation where
  sample_size : ℕ
  population_size : ℕ

/-- Function to determine the appropriate sampling method --/
def appropriate_sampling_method (dist : SalesDistribution) (inv : Investigation) : SamplingMethod :=
  sorry

/-- Theorem stating the appropriate sampling methods for the given scenario --/
theorem company_sampling_methods 
  (dist : SalesDistribution)
  (inv1 inv2 : Investigation)
  (h1 : dist.total_points = 600)
  (h2 : dist.region_points Region.A = 150)
  (h3 : dist.region_points Region.B = 120)
  (h4 : dist.region_points Region.C = 180)
  (h5 : dist.region_points Region.D = 150)
  (h6 : dist.large_points_C = 20)
  (h7 : inv1.sample_size = 100)
  (h8 : inv1.population_size = 600)
  (h9 : inv2.sample_size = 7)
  (h10 : inv2.population_size = 20) :
  (appropriate_sampling_method dist inv1 = SamplingMethod.StratifiedSampling) ∧
  (appropriate_sampling_method dist inv2 = SamplingMethod.SimpleRandomSampling) :=
sorry

end NUMINAMATH_CALUDE_company_sampling_methods_l1234_123428


namespace NUMINAMATH_CALUDE_max_score_for_successful_teams_l1234_123450

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  num_teams : Nat
  num_successful_teams : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- The maximum score that can be achieved by the successful teams -/
def max_total_score (t : FootballTournament) : Nat :=
  let internal_matches := t.num_successful_teams * (t.num_successful_teams - 1) / 2
  let external_matches := t.num_successful_teams * (t.num_teams - t.num_successful_teams)
  (internal_matches + external_matches) * t.points_for_win

/-- The theorem stating the maximum integer N for which at least 6 teams can score N points -/
theorem max_score_for_successful_teams (t : FootballTournament) 
    (h1 : t.num_teams = 15)
    (h2 : t.num_successful_teams = 6)
    (h3 : t.points_for_win = 3)
    (h4 : t.points_for_draw = 1)
    (h5 : t.points_for_loss = 0) :
    ∃ (N : Nat), N = 34 ∧ 
    (∀ (M : Nat), (M > N → ¬(t.num_successful_teams * M ≤ max_total_score t))) ∧
    (t.num_successful_teams * N ≤ max_total_score t) := by
  sorry

end NUMINAMATH_CALUDE_max_score_for_successful_teams_l1234_123450


namespace NUMINAMATH_CALUDE_base9_734_equals_base3_211110_l1234_123451

/-- Converts a digit from base 9 to base 3 --/
def base9_to_base3 (d : ℕ) : ℕ := sorry

/-- Converts a number from base 9 to base 3 --/
def convert_base9_to_base3 (n : ℕ) : ℕ := sorry

/-- The main theorem stating that 734 in base 9 is equal to 211110 in base 3 --/
theorem base9_734_equals_base3_211110 :
  convert_base9_to_base3 734 = 211110 := by sorry

end NUMINAMATH_CALUDE_base9_734_equals_base3_211110_l1234_123451


namespace NUMINAMATH_CALUDE_farm_theorem_l1234_123418

def farm_problem (num_pigs num_hens : ℕ) : Prop :=
  let num_heads := num_pigs + num_hens
  let num_legs := 4 * num_pigs + 2 * num_hens
  (num_pigs = 11) ∧ (∃ k : ℕ, num_legs = 2 * num_heads + k) ∧ (num_legs - 2 * num_heads = 22)

theorem farm_theorem : ∃ num_hens : ℕ, farm_problem 11 num_hens :=
by sorry

end NUMINAMATH_CALUDE_farm_theorem_l1234_123418


namespace NUMINAMATH_CALUDE_complex_expression_simplification_l1234_123489

theorem complex_expression_simplification (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1.22 * (((Real.sqrt a + Real.sqrt b)^2 - 4*b) / ((a - b) / (Real.sqrt (1/b) + 3 * Real.sqrt (1/a)))) / 
  ((a + 9*b + 6 * Real.sqrt (a*b)) / (1 / Real.sqrt a + 1 / Real.sqrt b))) = 1 / (a * b) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_simplification_l1234_123489


namespace NUMINAMATH_CALUDE_roots_of_equation_l1234_123420

theorem roots_of_equation : 
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x * (x - 1) = x ↔ x = x₁ ∨ x = x₂) ∧ x₁ = 2 ∧ x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1234_123420


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l1234_123429

/-- The number of y-intercepts for the parabola x = 3y^2 - 4y + 5 -/
theorem parabola_y_intercepts :
  let f (y : ℝ) := 3 * y^2 - 4 * y + 5
  ∃! x : ℝ, (∀ y : ℝ, f y = x) ∧ (¬ ∃ y : ℝ, f y = 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l1234_123429


namespace NUMINAMATH_CALUDE_right_triangle_area_l1234_123471

theorem right_triangle_area (a b c : ℝ) (h1 : a = 40) (h2 : c = 41) (h3 : a^2 + b^2 = c^2) : 
  (1/2) * a * b = 180 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1234_123471


namespace NUMINAMATH_CALUDE_james_earnings_difference_l1234_123441

theorem james_earnings_difference (january_earnings : ℕ) 
  (february_earnings : ℕ) (march_earnings : ℕ) (total_earnings : ℕ) :
  january_earnings = 4000 →
  february_earnings = 2 * january_earnings →
  march_earnings < february_earnings →
  total_earnings = january_earnings + february_earnings + march_earnings →
  total_earnings = 18000 →
  february_earnings - march_earnings = 2000 := by
sorry

end NUMINAMATH_CALUDE_james_earnings_difference_l1234_123441


namespace NUMINAMATH_CALUDE_williams_riding_time_l1234_123466

def max_riding_time : ℝ := 6

theorem williams_riding_time (x : ℝ) : 
  (2 * max_riding_time) + (2 * x) + (2 * (max_riding_time / 2)) = 21 → x = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_williams_riding_time_l1234_123466


namespace NUMINAMATH_CALUDE_field_area_minus_ponds_l1234_123407

/-- The area of a square field with sides of 10 meters, minus the area of three non-overlapping 
    circular ponds each with a radius of 3 meters, is equal to 100 - 27π square meters. -/
theorem field_area_minus_ponds (π : ℝ) : ℝ := by
  -- Define the side length of the square field
  let square_side : ℝ := 10
  -- Define the radius of each circular pond
  let pond_radius : ℝ := 3
  -- Define the number of ponds
  let num_ponds : ℕ := 3
  -- Calculate the area of the square field
  let square_area : ℝ := square_side ^ 2
  -- Calculate the area of one circular pond
  let pond_area : ℝ := π * pond_radius ^ 2
  -- Calculate the total area of all ponds
  let total_pond_area : ℝ := num_ponds * pond_area
  -- Calculate the remaining area (field area minus pond area)
  let remaining_area : ℝ := square_area - total_pond_area
  -- Prove that the remaining area is equal to 100 - 27π
  sorry

#check field_area_minus_ponds

end NUMINAMATH_CALUDE_field_area_minus_ponds_l1234_123407


namespace NUMINAMATH_CALUDE_tan_double_angle_l1234_123445

theorem tan_double_angle (α : Real) (h : Real.tan α = 1/3) : Real.tan (2 * α) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_double_angle_l1234_123445


namespace NUMINAMATH_CALUDE_route_length_is_200_l1234_123475

/-- Two trains traveling on a route --/
structure TrainRoute where
  length : ℝ
  train_a_time : ℝ
  train_b_time : ℝ
  meeting_distance : ℝ

/-- The specific train route from the problem --/
def problem_route : TrainRoute where
  length := 200
  train_a_time := 10
  train_b_time := 6
  meeting_distance := 75

/-- Theorem stating that the given conditions imply the route length is 200 miles --/
theorem route_length_is_200 (route : TrainRoute) :
  route.train_a_time = 10 ∧
  route.train_b_time = 6 ∧
  route.meeting_distance = 75 →
  route.length = 200 := by
  sorry

#check route_length_is_200

end NUMINAMATH_CALUDE_route_length_is_200_l1234_123475
