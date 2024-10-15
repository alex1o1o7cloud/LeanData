import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_factorization_l714_71478

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 10 * a * x + 25 * a = a * (x - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l714_71478


namespace NUMINAMATH_CALUDE_system_solution_l714_71495

theorem system_solution :
  ∃! (x y : ℝ), (3 * x = 2 * y) ∧ (x - 2 * y = -4) :=
by
  sorry

end NUMINAMATH_CALUDE_system_solution_l714_71495


namespace NUMINAMATH_CALUDE_only_negative_five_smaller_than_negative_three_l714_71434

theorem only_negative_five_smaller_than_negative_three :
  let numbers : List ℚ := [0, -1, -5, -1/2]
  ∀ x ∈ numbers, x < -3 ↔ x = -5 := by
sorry

end NUMINAMATH_CALUDE_only_negative_five_smaller_than_negative_three_l714_71434


namespace NUMINAMATH_CALUDE_cloth_coloring_problem_l714_71469

/-- The length of cloth that can be colored by a given number of men in a given number of days -/
def clothLength (men : ℕ) (days : ℚ) : ℚ :=
  sorry

theorem cloth_coloring_problem :
  let men₁ : ℕ := 6
  let days₁ : ℚ := 2
  let men₂ : ℕ := 2
  let days₂ : ℚ := 4.5
  let length₂ : ℚ := 36

  clothLength men₂ days₂ = length₂ →
  clothLength men₁ days₁ = 48 :=
by sorry

end NUMINAMATH_CALUDE_cloth_coloring_problem_l714_71469


namespace NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l714_71453

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem fourth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_first : a 1 = 13)
  (h_last : a 6 = 49) :
  a 4 = 31 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_arithmetic_sequence_l714_71453


namespace NUMINAMATH_CALUDE_books_removed_l714_71445

theorem books_removed (damaged_books : ℕ) (obsolete_books : ℕ) : 
  damaged_books = 11 →
  obsolete_books = 6 * damaged_books - 8 →
  damaged_books + obsolete_books = 69 :=
by
  sorry

end NUMINAMATH_CALUDE_books_removed_l714_71445


namespace NUMINAMATH_CALUDE_total_silk_dyed_l714_71494

theorem total_silk_dyed (green_silk : ℕ) (pink_silk : ℕ) 
  (h1 : green_silk = 61921) (h2 : pink_silk = 49500) : 
  green_silk + pink_silk = 111421 := by
  sorry

end NUMINAMATH_CALUDE_total_silk_dyed_l714_71494


namespace NUMINAMATH_CALUDE_angle_sum_bound_l714_71483

theorem angle_sum_bound (A B : Real) (h_triangle : 0 < A ∧ 0 < B ∧ A + B < π) 
  (h_inequality : ∀ x > 0, (Real.sin B / Real.cos A)^x + (Real.sin A / Real.cos B)^x < 2) :
  0 < A + B ∧ A + B < π/2 := by sorry

end NUMINAMATH_CALUDE_angle_sum_bound_l714_71483


namespace NUMINAMATH_CALUDE_area_triangle_on_hyperbola_l714_71405

/-- The area of a triangle formed by three points on the curve xy = 1 -/
theorem area_triangle_on_hyperbola (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) 
  (h₁ : x₁ * y₁ = 1) 
  (h₂ : x₂ * y₂ = 1) 
  (h₃ : x₃ * y₃ = 1) 
  (h₄ : x₁ ≠ 0) 
  (h₅ : x₂ ≠ 0) 
  (h₆ : x₃ ≠ 0) :
  let t := abs ((x₁ - x₂) * (x₂ - x₃) * (x₃ - x₁)) / (2 * x₁ * x₂ * x₃)
  t = abs (1/2 * (x₁ * y₂ + x₂ * y₃ + x₃ * y₁ - x₂ * y₁ - x₃ * y₂ - x₁ * y₃)) := by
  sorry


end NUMINAMATH_CALUDE_area_triangle_on_hyperbola_l714_71405


namespace NUMINAMATH_CALUDE_alec_class_size_l714_71436

theorem alec_class_size :
  ∀ S : ℕ,
  (3 * S / 4 : ℚ) = S / 2 + 5 + ((S / 2 - 5) / 5 : ℚ) + 5 →
  S = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_alec_class_size_l714_71436


namespace NUMINAMATH_CALUDE_arithmetic_sequence_specific_values_l714_71412

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  S : ℕ → ℤ  -- The sum function
  sum_property : ∀ n, S n = n * (a 1 + a n) / 2  -- Property of sum of arithmetic sequence
  arithmetic_property : ∀ n, a (n + 1) - a n = a 2 - a 1  -- Property of arithmetic sequence

/-- Main theorem about specific values in an arithmetic sequence -/
theorem arithmetic_sequence_specific_values (seq : ArithmeticSequence) 
    (h1 : seq.S 9 = -36) (h2 : seq.S 13 = -104) : 
    seq.a 5 = -4 ∧ seq.S 11 = -66 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_specific_values_l714_71412


namespace NUMINAMATH_CALUDE_building_houses_200_people_l714_71418

/-- Calculates the number of people housed in a building given the number of stories,
    apartments per floor, and people per apartment. -/
def people_in_building (stories : ℕ) (apartments_per_floor : ℕ) (people_per_apartment : ℕ) : ℕ :=
  stories * apartments_per_floor * people_per_apartment

/-- Theorem stating that a 25-story building with 4 apartments per floor and 2 people
    per apartment houses 200 people. -/
theorem building_houses_200_people :
  people_in_building 25 4 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_building_houses_200_people_l714_71418


namespace NUMINAMATH_CALUDE_youngest_child_age_l714_71437

/-- Represents a family with its members and ages -/
structure Family where
  memberCount : ℕ
  totalAge : ℕ

/-- Calculates the average age of a family -/
def averageAge (f : Family) : ℚ :=
  f.totalAge / f.memberCount

theorem youngest_child_age (initialFamily : Family) 
  (finalFamily : Family) (yearsPassed : ℕ) :
  initialFamily.memberCount = 4 →
  averageAge initialFamily = 24 →
  yearsPassed = 10 →
  finalFamily.memberCount = initialFamily.memberCount + 2 →
  averageAge finalFamily = 24 →
  ∃ (youngestAge olderAge : ℕ), 
    olderAge = youngestAge + 2 ∧
    youngestAge + olderAge = finalFamily.totalAge - (initialFamily.totalAge + yearsPassed * initialFamily.memberCount) ∧
    youngestAge = 3 := by
  sorry


end NUMINAMATH_CALUDE_youngest_child_age_l714_71437


namespace NUMINAMATH_CALUDE_garden_problem_l714_71422

/-- Represents the gardening problem with eggplants and sunflowers. -/
theorem garden_problem (eggplants_per_packet : ℕ) (sunflowers_per_packet : ℕ) 
  (eggplant_packets : ℕ) (total_plants : ℕ) :
  eggplants_per_packet = 14 →
  sunflowers_per_packet = 10 →
  eggplant_packets = 4 →
  total_plants = 116 →
  ∃ sunflower_packets : ℕ, 
    sunflower_packets = 6 ∧ 
    total_plants = eggplants_per_packet * eggplant_packets + sunflowers_per_packet * sunflower_packets :=
by sorry

end NUMINAMATH_CALUDE_garden_problem_l714_71422


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l714_71487

theorem min_perimeter_triangle (a b x : ℕ) (h1 : a = 40) (h2 : b = 50) : 
  (a + b + x > a + b ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, y ≠ x → a + b + y > a + b + x) → 
  a + b + x = 101 := by
sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l714_71487


namespace NUMINAMATH_CALUDE_fred_earnings_l714_71467

/-- Fred's initial amount of money in dollars -/
def initial_amount : ℕ := 23

/-- Fred's final amount of money in dollars after washing cars -/
def final_amount : ℕ := 86

/-- The amount Fred made washing cars -/
def earnings : ℕ := final_amount - initial_amount

theorem fred_earnings : earnings = 63 := by
  sorry

end NUMINAMATH_CALUDE_fred_earnings_l714_71467


namespace NUMINAMATH_CALUDE_solution_in_quadrant_I_l714_71414

theorem solution_in_quadrant_I (c : ℝ) :
  (∃ x y : ℝ, x - y = 4 ∧ c * x + y = 7 ∧ x > 0 ∧ y > 0) ↔ -1 < c ∧ c < 7/4 :=
by sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_I_l714_71414


namespace NUMINAMATH_CALUDE_binomial_expansion_constant_term_l714_71409

/-- The constant term in the binomial expansion of (ax - 1/√x)^6 -/
def constant_term (a : ℝ) : ℝ := 15 * a^2

theorem binomial_expansion_constant_term (a : ℝ) (h1 : a > 0) (h2 : constant_term a = 120) : 
  a = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_constant_term_l714_71409


namespace NUMINAMATH_CALUDE_solution_set_equality_l714_71462

/-- The solution set of the inequality (a^2 - 1)x^2 - (a - 1)x - 1 < 0 is equal to ℝ if and only if -3/5 < a < 1 -/
theorem solution_set_equality (a : ℝ) : 
  (∀ x : ℝ, (a^2 - 1) * x^2 - (a - 1) * x - 1 < 0) ↔ (-3/5 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l714_71462


namespace NUMINAMATH_CALUDE_men_on_bus_l714_71425

theorem men_on_bus (total : ℕ) (women : ℕ) (children : ℕ) 
  (h1 : total = 54)
  (h2 : women = 26)
  (h3 : children = 10) :
  total - women - children = 18 := by
sorry

end NUMINAMATH_CALUDE_men_on_bus_l714_71425


namespace NUMINAMATH_CALUDE_total_paintings_after_five_weeks_l714_71444

/-- Represents a painter's weekly schedule and initial paintings -/
structure Painter where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat
  saturday : Nat
  sunday : Nat
  initial : Nat

/-- Calculates the total number of paintings after a given number of weeks -/
def total_paintings (p : Painter) (weeks : Nat) : Nat :=
  p.initial + weeks * (p.monday + p.tuesday + p.wednesday + p.thursday + p.friday + p.saturday + p.sunday)

/-- Philip's painting schedule -/
def philip : Painter :=
  { monday := 3, tuesday := 3, wednesday := 2, thursday := 5, friday := 5, saturday := 0, sunday := 0, initial := 20 }

/-- Amelia's painting schedule -/
def amelia : Painter :=
  { monday := 2, tuesday := 2, wednesday := 2, thursday := 2, friday := 2, saturday := 2, sunday := 2, initial := 45 }

theorem total_paintings_after_five_weeks :
  total_paintings philip 5 + total_paintings amelia 5 = 225 := by
  sorry

end NUMINAMATH_CALUDE_total_paintings_after_five_weeks_l714_71444


namespace NUMINAMATH_CALUDE_highway_extension_l714_71451

theorem highway_extension (current_length : ℕ) (target_length : ℕ) (first_day : ℕ) : 
  current_length = 200 →
  target_length = 650 →
  first_day = 50 →
  target_length - current_length - (first_day + 3 * first_day) = 250 := by
sorry

end NUMINAMATH_CALUDE_highway_extension_l714_71451


namespace NUMINAMATH_CALUDE_original_cube_volume_l714_71482

/-- Given two similar cubes where one has twice the side length of the other,
    if the larger cube has a volume of 216 cubic feet,
    then the smaller cube has a volume of 27 cubic feet. -/
theorem original_cube_volume
  (s : ℝ)  -- side length of the original cube
  (h1 : (2 * s) ^ 3 = 216)  -- volume of the larger cube is 216 cubic feet
  : s ^ 3 = 27 :=
by
  sorry

end NUMINAMATH_CALUDE_original_cube_volume_l714_71482


namespace NUMINAMATH_CALUDE_paperback_copies_sold_l714_71401

theorem paperback_copies_sold (hardback_copies : ℕ) (total_copies : ℕ) :
  hardback_copies = 36000 →
  total_copies = 440000 →
  ∃ (paperback_copies : ℕ), 
    paperback_copies = 9 * hardback_copies ∧
    total_copies = hardback_copies + paperback_copies ∧
    paperback_copies = 324000 :=
by sorry

end NUMINAMATH_CALUDE_paperback_copies_sold_l714_71401


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l714_71486

theorem complex_fraction_equality : (5 * Complex.I) / (2 + Complex.I) = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l714_71486


namespace NUMINAMATH_CALUDE_remainder_theorem_l714_71470

/-- The polynomial being divided -/
def p (x : ℝ) : ℝ := x^6 - x^5 - x^4 + x^3 + x^2

/-- The divisor -/
def d (x : ℝ) : ℝ := (x^2 - 4) * (x + 1)

/-- The remainder -/
def r (x : ℝ) : ℝ := 15 * x^2 - 12 * x - 24

/-- Theorem stating that r is the remainder when p is divided by d -/
theorem remainder_theorem : ∃ q : ℝ → ℝ, ∀ x : ℝ, p x = d x * q x + r x := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l714_71470


namespace NUMINAMATH_CALUDE_shifted_quadratic_sum_l714_71491

/-- The sum of coefficients after shifting a quadratic function -/
theorem shifted_quadratic_sum (a b c : ℝ) : 
  (∀ x, 3 * (x + 2)^2 + 2 * (x + 2) + 4 = a * x^2 + b * x + c) → 
  a + b + c = 37 := by
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_sum_l714_71491


namespace NUMINAMATH_CALUDE_division_problem_l714_71473

theorem division_problem (a b c : ℝ) 
  (h1 : a / b = 3) 
  (h2 : b / c = 2 / 5) : 
  c / a = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l714_71473


namespace NUMINAMATH_CALUDE_image_of_A_under_f_l714_71499

def A : Set ℕ := {1, 2}

def f (x : ℕ) : ℕ := x^2

theorem image_of_A_under_f : Set.image f A = {1, 4} := by sorry

end NUMINAMATH_CALUDE_image_of_A_under_f_l714_71499


namespace NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l714_71477

theorem probability_four_twos_in_five_rolls : 
  let n_rolls : ℕ := 5
  let n_sides : ℕ := 6
  let n_twos : ℕ := 4
  let p_two : ℚ := 1 / n_sides
  let p_not_two : ℚ := 1 - p_two
  Nat.choose n_rolls n_twos * p_two ^ n_twos * p_not_two ^ (n_rolls - n_twos) = 3125 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_probability_four_twos_in_five_rolls_l714_71477


namespace NUMINAMATH_CALUDE_points_for_tie_l714_71498

/-- The number of teams in the tournament -/
def num_teams : ℕ := 6

/-- The number of points awarded for a win -/
def points_for_win : ℕ := 3

/-- The number of points awarded for a loss -/
def points_for_loss : ℕ := 0

/-- The difference between max and min total points -/
def max_min_difference : ℕ := 15

/-- The number of games played in the tournament -/
def num_games : ℕ := (num_teams * (num_teams - 1)) / 2

theorem points_for_tie (T : ℕ) : 
  (num_games * points_for_win) - (num_games * T) = max_min_difference → T = 2 := by
  sorry

end NUMINAMATH_CALUDE_points_for_tie_l714_71498


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_l714_71497

theorem smallest_n_for_candy (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ 15 * m % 10 = 0 ∧ 15 * m % 18 = 0 ∧ 15 * m % 25 = 0 → m ≥ n) ∧
  n > 0 ∧ 15 * n % 10 = 0 ∧ 15 * n % 18 = 0 ∧ 15 * n % 25 = 0 →
  n = 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_l714_71497


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l714_71457

theorem at_least_one_greater_than_one (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) : 
  x > 1 ∨ y > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l714_71457


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l714_71408

-- Define the displacement function
def h (t : ℝ) : ℝ := 15 * t - t^2

-- Define the velocity function as the derivative of the displacement function
def v (t : ℝ) : ℝ := 15 - 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 9 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l714_71408


namespace NUMINAMATH_CALUDE_power_function_above_identity_l714_71413

theorem power_function_above_identity {x α : ℝ} (hx : x ∈ Set.Ioo 0 1) :
  x^α > x ↔ α ∈ Set.Iio 1 := by sorry

end NUMINAMATH_CALUDE_power_function_above_identity_l714_71413


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l714_71490

theorem unique_solution_for_equation (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 = 2020 * x) ↔
  x = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l714_71490


namespace NUMINAMATH_CALUDE_tin_in_new_alloy_l714_71463

/-- Calculate the amount of tin in a new alloy formed by mixing two alloys -/
theorem tin_in_new_alloy (alloy_a_mass : ℝ) (alloy_b_mass : ℝ)
  (lead_tin_ratio_a : ℚ) (tin_copper_ratio_b : ℚ) :
  alloy_a_mass = 135 →
  alloy_b_mass = 145 →
  lead_tin_ratio_a = 3 / 5 →
  tin_copper_ratio_b = 2 / 3 →
  let tin_in_a := alloy_a_mass * (5 / 8 : ℝ)
  let tin_in_b := alloy_b_mass * (2 / 5 : ℝ)
  tin_in_a + tin_in_b = 142.375 := by
  sorry

end NUMINAMATH_CALUDE_tin_in_new_alloy_l714_71463


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l714_71448

/-- The eccentricity of a hyperbola with equation (y^2 / a^2) - (x^2 / b^2) = 1 and asymptote y = 2x is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : ∃ (k : ℝ), k = a / b ∧ k = 2) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l714_71448


namespace NUMINAMATH_CALUDE_rationalize_denominator_l714_71475

theorem rationalize_denominator :
  ∃ (A B C D E : ℤ),
    (B < D) ∧
    (5 : ℝ) / (4 * Real.sqrt 7 + 3 * Real.sqrt 13) = (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    A = 4 ∧ B = 7 ∧ C = -3 ∧ D = 13 ∧ E = 1 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l714_71475


namespace NUMINAMATH_CALUDE_parallel_vectors_subtraction_l714_71446

/-- Given vectors a and b where a is parallel to b, prove that 2a - b = (4, -8) -/
theorem parallel_vectors_subtraction (m : ℝ) : 
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![m, 4]
  (∃ (k : ℝ), a = k • b) →  -- parallel condition
  (2 • a - b) = ![4, -8] := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_subtraction_l714_71446


namespace NUMINAMATH_CALUDE_inverse_function_point_l714_71430

-- Define a function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x ∧ f (f_inv x) = x

-- State that the graph of f passes through (0, 1)
axiom f_point : f 0 = 1

-- Theorem to prove
theorem inverse_function_point :
  (f_inv 1) + 1 = 1 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_point_l714_71430


namespace NUMINAMATH_CALUDE_logarithm_simplification_l714_71417

theorem logarithm_simplification (a b c d x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (hx : x > 0) (hy : y > 0) :
  Real.log (a^2 / b^3) + Real.log (b^2 / c) + Real.log (c^3 / d^2) - Real.log (a^2 * y^2 / (d^3 * x)) 
  = Real.log (c^2 * d * x / y^2) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l714_71417


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l714_71433

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l714_71433


namespace NUMINAMATH_CALUDE_jenny_easter_eggs_l714_71450

theorem jenny_easter_eggs :
  ∃ (n : ℕ), n > 0 ∧ n ≥ 5 ∧ 30 % n = 0 ∧ 45 % n = 0 ∧
  ∀ (m : ℕ), m > 0 ∧ m ≥ 5 ∧ 30 % m = 0 ∧ 45 % m = 0 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_jenny_easter_eggs_l714_71450


namespace NUMINAMATH_CALUDE_apple_distribution_l714_71465

theorem apple_distribution (total_apples : ℕ) (apples_per_student : ℕ) : 
  total_apples = 120 →
  apples_per_student = 2 →
  (∃ (num_students : ℕ), 
    num_students * apples_per_student = total_apples - 1 ∧
    num_students > 0) →
  ∃ (num_students : ℕ), num_students = 59 := by
sorry

end NUMINAMATH_CALUDE_apple_distribution_l714_71465


namespace NUMINAMATH_CALUDE_clock_hands_angle_at_3_15_clock_hands_angle_at_3_15_is_7_5_l714_71429

/-- The angle between clock hands at 3:15 -/
theorem clock_hands_angle_at_3_15 : ℝ :=
  let hours_on_clock : ℕ := 12
  let degrees_per_hour : ℝ := 360 / hours_on_clock
  let minutes_per_hour : ℕ := 60
  let degrees_per_minute : ℝ := 360 / minutes_per_hour
  let minutes_past_3 : ℕ := 15
  let minute_hand_angle : ℝ := degrees_per_minute * minutes_past_3
  let hour_hand_angle : ℝ := 3 * degrees_per_hour + (degrees_per_hour / 4)
  let angle_difference : ℝ := hour_hand_angle - minute_hand_angle
  angle_difference

theorem clock_hands_angle_at_3_15_is_7_5 :
  clock_hands_angle_at_3_15 = 7.5 := by sorry

end NUMINAMATH_CALUDE_clock_hands_angle_at_3_15_clock_hands_angle_at_3_15_is_7_5_l714_71429


namespace NUMINAMATH_CALUDE_oil_measurement_l714_71484

/-- The amount of oil currently in Scarlett's measuring cup -/
def current_oil : ℝ := 0.16666666666666674

/-- The amount of oil Scarlett adds to the measuring cup -/
def added_oil : ℝ := 0.6666666666666666

/-- The total amount of oil after Scarlett adds more -/
def total_oil : ℝ := 0.8333333333333334

/-- Theorem stating that the current amount of oil plus the added amount equals the total amount -/
theorem oil_measurement :
  current_oil + added_oil = total_oil :=
by sorry

end NUMINAMATH_CALUDE_oil_measurement_l714_71484


namespace NUMINAMATH_CALUDE_bob_muffins_l714_71404

theorem bob_muffins (total : ℕ) (days : ℕ) (increment : ℕ) (second_day : ℚ) : 
  total = 55 → 
  days = 4 → 
  increment = 2 → 
  (∃ (first_day : ℚ), 
    first_day + (first_day + ↑increment) + (first_day + 2 * ↑increment) + (first_day + 3 * ↑increment) = total ∧
    second_day = first_day + ↑increment) →
  second_day = 12.75 := by sorry

end NUMINAMATH_CALUDE_bob_muffins_l714_71404


namespace NUMINAMATH_CALUDE_smallest_sum_of_factors_l714_71474

theorem smallest_sum_of_factors (a b : ℕ+) (h : a * b = 240) :
  a + b ≥ 31 ∧ ∃ (x y : ℕ+), x * y = 240 ∧ x + y = 31 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_factors_l714_71474


namespace NUMINAMATH_CALUDE_rectangle_side_length_l714_71461

/-- Given a rectangle with perimeter 8 and one side length -a-2, 
    the length of the other side is 6+a. -/
theorem rectangle_side_length (a : ℝ) : 
  let perimeter : ℝ := 8
  let side1 : ℝ := -a - 2
  let side2 : ℝ := 6 + a
  perimeter = 2 * (side1 + side2) := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_length_l714_71461


namespace NUMINAMATH_CALUDE_odd_primes_cube_sum_l714_71415

theorem odd_primes_cube_sum (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  Odd p → Odd q → Odd r → 
  p^3 + q^3 + 3*p*q*r ≠ r^3 := by
  sorry

end NUMINAMATH_CALUDE_odd_primes_cube_sum_l714_71415


namespace NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l714_71435

-- Define the universe set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define set M
def M : Set ℕ := {1, 2, 3}

-- Define set N
def N : Set ℕ := {2, 3, 4}

-- State the theorem
theorem complement_intersection_equals_specific_set :
  (M ∩ N)ᶜ = {1, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equals_specific_set_l714_71435


namespace NUMINAMATH_CALUDE_five_letter_words_count_l714_71428

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The length of the word --/
def word_length : ℕ := 5

/-- The number of positions that can vary --/
def variable_positions : ℕ := word_length - 2

theorem five_letter_words_count :
  (alphabet_size ^ variable_positions : ℕ) = 17576 := by
  sorry

end NUMINAMATH_CALUDE_five_letter_words_count_l714_71428


namespace NUMINAMATH_CALUDE_billy_spits_30_inches_l714_71407

/-- The distance Billy can spit a watermelon seed -/
def billy_distance : ℝ := sorry

/-- The distance Madison can spit a watermelon seed -/
def madison_distance : ℝ := sorry

/-- The distance Ryan can spit a watermelon seed -/
def ryan_distance : ℝ := 18

/-- Madison spits 20% farther than Billy -/
axiom madison_farther : madison_distance = billy_distance * 1.2

/-- Ryan spits 50% shorter than Madison -/
axiom ryan_shorter : ryan_distance = madison_distance * 0.5

theorem billy_spits_30_inches : billy_distance = 30 := by sorry

end NUMINAMATH_CALUDE_billy_spits_30_inches_l714_71407


namespace NUMINAMATH_CALUDE_smallest_possible_a_l714_71410

theorem smallest_possible_a (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) 
  (h3 : ∀ x : ℤ, Real.sin (a * ↑x + b) = Real.sin (17 * ↑x)) : 
  (∀ a' : ℝ, (0 ≤ a' ∧ ∀ x : ℤ, Real.sin (a' * ↑x + b) = Real.sin (17 * ↑x)) → a' ≥ 17) ∧ a ≥ 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l714_71410


namespace NUMINAMATH_CALUDE_probability_largest_smaller_theorem_l714_71485

/-- The probability that the largest number in each row is smaller than the largest number in each row with more numbers, given n rows arranged as described. -/
def probability_largest_smaller (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / (n + 1).factorial

/-- Theorem stating the probability for the arrangement of numbers in rows. -/
theorem probability_largest_smaller_theorem (n : ℕ) :
  let total_numbers := n * (n + 1) / 2
  let row_sizes := List.range n.succ
  probability_largest_smaller n =
    (2 ^ n : ℚ) / (n + 1).factorial :=
by
  sorry

end NUMINAMATH_CALUDE_probability_largest_smaller_theorem_l714_71485


namespace NUMINAMATH_CALUDE_thirty_three_million_equals_33000000_l714_71400

-- Define million
def million : ℕ := 1000000

-- Define 33 million
def thirty_three_million : ℕ := 33 * million

-- Theorem to prove
theorem thirty_three_million_equals_33000000 : 
  thirty_three_million = 33000000 := by
  sorry

end NUMINAMATH_CALUDE_thirty_three_million_equals_33000000_l714_71400


namespace NUMINAMATH_CALUDE_lines_coplanar_iff_m_eq_zero_l714_71427

/-- Definition of the first line -/
def line1 (m s : ℝ) : ℝ × ℝ × ℝ := (1 + 2*s, 2 + 2*s, 3 - m*s)

/-- Definition of the second line -/
def line2 (m v : ℝ) : ℝ × ℝ × ℝ := (m*v, 5 + 3*v, 6 + 2*v)

/-- Two vectors are coplanar if their cross product is zero -/
def coplanar (u v w : ℝ × ℝ × ℝ) : Prop :=
  let (u₁, u₂, u₃) := u
  let (v₁, v₂, v₃) := v
  let (w₁, w₂, w₃) := w
  (v₁ - u₁) * (w₂ - u₂) * (u₃ - u₃) +
  (v₂ - u₂) * (w₃ - u₃) * (u₁ - u₁) +
  (v₃ - u₃) * (w₁ - u₁) * (u₂ - u₂) -
  (v₃ - u₃) * (w₂ - u₂) * (u₁ - u₁) -
  (v₁ - u₁) * (w₃ - u₃) * (u₂ - u₂) -
  (v₂ - u₂) * (w₁ - u₁) * (u₃ - u₃) = 0

/-- Theorem: The lines are coplanar if and only if m = 0 -/
theorem lines_coplanar_iff_m_eq_zero :
  ∀ s v : ℝ, coplanar (1, 2, 3) (line1 m s) (line2 m v) ↔ m = 0 :=
sorry

end NUMINAMATH_CALUDE_lines_coplanar_iff_m_eq_zero_l714_71427


namespace NUMINAMATH_CALUDE_cube_sum_divided_l714_71411

theorem cube_sum_divided (x y : ℝ) (hx : x = 3) (hy : y = 4) : 
  (x^3 + 3*y^3) / 9 = 73/3 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_divided_l714_71411


namespace NUMINAMATH_CALUDE_cone_volume_over_pi_l714_71481

/-- Given a cone formed from a 240-degree sector of a circle with radius 24,
    the volume of the cone divided by π is equal to 2048√5/3 -/
theorem cone_volume_over_pi (r : ℝ) (θ : ℝ) :
  r = 24 →
  θ = 240 * π / 180 →
  let base_radius := r * θ / (2 * π)
  let height := Real.sqrt (r^2 - base_radius^2)
  let volume := (1/3) * π * base_radius^2 * height
  volume / π = 2048 * Real.sqrt 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_volume_over_pi_l714_71481


namespace NUMINAMATH_CALUDE_rogers_trays_l714_71464

/-- Roger's tray-carrying problem -/
theorem rogers_trays (trays_per_trip : ℕ) (trips : ℕ) (trays_second_table : ℕ) : 
  trays_per_trip = 4 → trips = 3 → trays_second_table = 2 →
  trays_per_trip * trips - trays_second_table = 10 := by
  sorry

end NUMINAMATH_CALUDE_rogers_trays_l714_71464


namespace NUMINAMATH_CALUDE_system_solution_l714_71443

-- Define the system of equations
def system_equations (x₁ x₂ x₃ x₄ : ℝ) : Prop :=
  (x₁ + x₂*x₃*x₄ = 2) ∧
  (x₂ + x₁*x₃*x₄ = 2) ∧
  (x₃ + x₁*x₂*x₄ = 2) ∧
  (x₄ + x₁*x₂*x₃ = 2)

-- Define the set of solutions
def solution_set : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(1, 1, 1, 1), (-1, -1, -1, 3), (-1, -1, 3, -1), (-1, 3, -1, -1), (3, -1, -1, -1)}

-- Theorem statement
theorem system_solution :
  ∀ x₁ x₂ x₃ x₄ : ℝ, system_equations x₁ x₂ x₃ x₄ ↔ (x₁, x₂, x₃, x₄) ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_system_solution_l714_71443


namespace NUMINAMATH_CALUDE_litter_patrol_cans_l714_71420

theorem litter_patrol_cans (total_litter : ℕ) (glass_bottles : ℕ) (aluminum_cans : ℕ) : 
  total_litter = 18 → glass_bottles = 10 → aluminum_cans = total_litter - glass_bottles → 
  aluminum_cans = 8 := by sorry

end NUMINAMATH_CALUDE_litter_patrol_cans_l714_71420


namespace NUMINAMATH_CALUDE_f_value_at_2_l714_71442

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l714_71442


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l714_71432

-- Define a geometric sequence with positive terms
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∃ r : ℝ, r > 0 ∧ ∀ n, a (n + 1) = r * a n)

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1) * (a 19) = 16 →
  (a 1) + (a 19) = 10 →
  (a 8) * (a 10) * (a 12) = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l714_71432


namespace NUMINAMATH_CALUDE_generated_number_is_square_l714_71466

/-- Generates a number with n threes followed by 34 -/
def generateNumber (n : ℕ) : ℕ :=
  3 * (10^n - 1) / 9 * 10 + 34

/-- Theorem stating that the generated number is always a perfect square -/
theorem generated_number_is_square (n : ℕ) :
  ∃ k : ℕ, (generateNumber n) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_generated_number_is_square_l714_71466


namespace NUMINAMATH_CALUDE_min_value_expression_l714_71426

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^2 + y^2 + 16 / (x + y)^2 ≥ 8 ∧
  (x^2 + y^2 + 16 / (x + y)^2 = 8 ↔ x = y ∧ x = 2^(1/4)) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l714_71426


namespace NUMINAMATH_CALUDE_complement_of_angle1_l714_71476

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the given angle
def angle1 : Angle := ⟨38, 15⟩

-- Define the complement of an angle
def complement (a : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (a.degrees * 60 + a.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- Theorem statement
theorem complement_of_angle1 :
  complement angle1 = ⟨51, 45⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_angle1_l714_71476


namespace NUMINAMATH_CALUDE_percentage_difference_l714_71488

theorem percentage_difference (x y : ℝ) (h : x = 0.65 * y) : y = (1 + 0.35) * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l714_71488


namespace NUMINAMATH_CALUDE_equation_proof_l714_71424

theorem equation_proof (h : Real.sqrt 27 = 3 * Real.sqrt 3) :
  -2 * Real.sqrt 3 + Real.sqrt 27 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l714_71424


namespace NUMINAMATH_CALUDE_distance_between_vertices_l714_71493

-- Define the equation
def equation (x y : ℝ) : Prop := Real.sqrt (x^2 + y^2) + |y - 2| = 4

-- Define the two parabolas
def parabola1 (x y : ℝ) : Prop := y = -(1/12) * x^2 + 3
def parabola2 (x y : ℝ) : Prop := y = (1/4) * x^2 - 1

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 3)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices : 
  ∀ x y : ℝ, equation x y → 
  (parabola1 x y ∨ parabola2 x y) → 
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = 4 := by
sorry

end NUMINAMATH_CALUDE_distance_between_vertices_l714_71493


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l714_71496

/-- Given vectors a and b, where a is parallel to b, prove that tan(α + π/4) = 3 -/
theorem tan_alpha_plus_pi_fourth (α : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (-2, Real.cos α))
  (hb : b = (-1, Real.sin α))
  (parallel : ∃ (k : ℝ), a = k • b) :
  Real.tan (α + π/4) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_fourth_l714_71496


namespace NUMINAMATH_CALUDE_morning_rowers_count_l714_71447

def afternoon_rowers : ℕ := 17
def total_rowers : ℕ := 32

theorem morning_rowers_count : 
  total_rowers - afternoon_rowers = 15 := by sorry

end NUMINAMATH_CALUDE_morning_rowers_count_l714_71447


namespace NUMINAMATH_CALUDE_sum_digits_12_4_less_than_32_l714_71459

/-- The sum of digits of a number n in base b -/
def sum_of_digits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Theorem stating that for all bases greater than 10, the sum of digits of 12^4 is less than 2^5 -/
theorem sum_digits_12_4_less_than_32 (b : ℕ) (h : b > 10) : 
  sum_of_digits (12^4) b < 2^5 := by sorry

end NUMINAMATH_CALUDE_sum_digits_12_4_less_than_32_l714_71459


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l714_71421

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by
  sorry

theorem negation_of_specific_proposition : 
  (¬ ∀ x : ℝ, x^2 + x > 2) ↔ (∃ x : ℝ, x^2 + x ≤ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_specific_proposition_l714_71421


namespace NUMINAMATH_CALUDE_factorization_of_quadratic_l714_71406

theorem factorization_of_quadratic (x : ℝ) : x^2 - 3*x = x*(x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_quadratic_l714_71406


namespace NUMINAMATH_CALUDE_discount_percentage_l714_71479

theorem discount_percentage (cupcake_price cookie_price : ℝ) 
  (cupcakes_sold cookies_sold : ℕ) (total_revenue : ℝ) :
  cupcake_price = 3 →
  cookie_price = 2 →
  cupcakes_sold = 16 →
  cookies_sold = 8 →
  total_revenue = 32 →
  ∃ (x : ℝ), 
    (cupcakes_sold : ℝ) * (cupcake_price * (100 - x) / 100) + 
    (cookies_sold : ℝ) * (cookie_price * (100 - x) / 100) = total_revenue ∧
    x = 50 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l714_71479


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l714_71439

def purchase_price : ℕ := 9000
def transportation_charges : ℕ := 1000
def profit_percentage : ℚ := 50 / 100
def selling_price : ℕ := 22500

theorem repair_cost_calculation :
  ∃ (repair_cost : ℕ),
    (purchase_price + repair_cost + transportation_charges) * (1 + profit_percentage) = selling_price ∧
    repair_cost = 5000 := by
  sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l714_71439


namespace NUMINAMATH_CALUDE_z_percentage_of_x_l714_71403

theorem z_percentage_of_x (x y z : ℝ) 
  (h1 : 0.45 * z = 1.2 * y) 
  (h2 : y = 0.75 * x) : 
  z = 2 * x := by
sorry

end NUMINAMATH_CALUDE_z_percentage_of_x_l714_71403


namespace NUMINAMATH_CALUDE_u_5_value_l714_71423

def sequence_u (u : ℕ → ℝ) : Prop :=
  ∀ n, u (n + 2) = 3 * u (n + 1) + 2 * u n

theorem u_5_value (u : ℕ → ℝ) (h : sequence_u u) (h3 : u 3 = 10) (h6 : u 6 = 256) :
  u 5 = 808 / 11 := by
  sorry

end NUMINAMATH_CALUDE_u_5_value_l714_71423


namespace NUMINAMATH_CALUDE_problem_polygon_area_l714_71492

/-- Represents a polygon with right angles at each corner -/
structure RightAnglePolygon where
  -- Define the lengths of the segments
  left_height : ℝ
  bottom_width : ℝ
  middle_height : ℝ
  middle_width : ℝ
  top_right_height : ℝ
  top_right_width : ℝ
  top_left_height : ℝ
  top_left_width : ℝ

/-- Calculates the area of the RightAnglePolygon -/
def area (p : RightAnglePolygon) : ℝ :=
  p.left_height * p.bottom_width +
  p.middle_height * p.middle_width +
  p.top_right_height * p.top_right_width +
  p.top_left_height * p.top_left_width

/-- The specific polygon from the problem -/
def problem_polygon : RightAnglePolygon :=
  { left_height := 7
  , bottom_width := 6
  , middle_height := 5
  , middle_width := 4
  , top_right_height := 6
  , top_right_width := 5
  , top_left_height := 1
  , top_left_width := 2
  }

/-- Theorem stating that the area of the problem_polygon is 94 -/
theorem problem_polygon_area :
  area problem_polygon = 94 := by
  sorry


end NUMINAMATH_CALUDE_problem_polygon_area_l714_71492


namespace NUMINAMATH_CALUDE_ages_sum_l714_71468

theorem ages_sum (a b c : ℕ) : 
  a = 20 + b + c → 
  a^2 = 2120 + (b + c)^2 → 
  a + b + c = 82 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l714_71468


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_negation_set_equivalence_l714_71416

theorem necessary_not_sufficient_condition (a b : ℝ) :
  (a > b → a + 1 > b) ∧ ¬(a + 1 > b → a > b) := by sorry

theorem negation_set_equivalence :
  {x : ℝ | ¬(1 / (x - 2) > 0)} = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_negation_set_equivalence_l714_71416


namespace NUMINAMATH_CALUDE_tim_travel_distance_l714_71471

/-- Represents the problem of Tim and Élan moving towards each other with increasing speeds -/
structure MeetingProblem where
  initialDistance : ℝ
  timInitialSpeed : ℝ
  elanInitialSpeed : ℝ

/-- Calculates the distance Tim travels before meeting Élan -/
def distanceTraveled (p : MeetingProblem) : ℝ :=
  sorry

/-- Theorem stating that Tim travels 20 miles before meeting Élan -/
theorem tim_travel_distance (p : MeetingProblem) 
  (h1 : p.initialDistance = 30)
  (h2 : p.timInitialSpeed = 10)
  (h3 : p.elanInitialSpeed = 5) :
  distanceTraveled p = 20 :=
sorry

end NUMINAMATH_CALUDE_tim_travel_distance_l714_71471


namespace NUMINAMATH_CALUDE_category_selection_probability_l714_71441

def total_items : ℕ := 8
def swimming_items : ℕ := 1
def ball_games_items : ℕ := 3
def track_field_items : ℕ := 4
def items_to_select : ℕ := 4

theorem category_selection_probability :
  (Nat.choose swimming_items 1 * Nat.choose ball_games_items 1 * Nat.choose track_field_items 2 +
   Nat.choose swimming_items 1 * Nat.choose ball_games_items 2 * Nat.choose track_field_items 1) /
  Nat.choose total_items items_to_select = 3 / 7 := by sorry

end NUMINAMATH_CALUDE_category_selection_probability_l714_71441


namespace NUMINAMATH_CALUDE_randys_pig_feed_l714_71454

/-- Calculates the total amount of pig feed for a month -/
def total_pig_feed_per_month (feed_per_pig_per_day : ℕ) (num_pigs : ℕ) (days_in_month : ℕ) : ℕ :=
  feed_per_pig_per_day * num_pigs * days_in_month

/-- Proves that Randy's pigs will be fed 1800 pounds of pig feed per month -/
theorem randys_pig_feed :
  total_pig_feed_per_month 15 4 30 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_randys_pig_feed_l714_71454


namespace NUMINAMATH_CALUDE_percentage_females_with_glasses_l714_71431

def total_population : ℕ := 5000
def male_population : ℕ := 2000
def females_with_glasses : ℕ := 900

theorem percentage_females_with_glasses :
  (females_with_glasses : ℚ) / ((total_population - male_population) : ℚ) * 100 = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_females_with_glasses_l714_71431


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l714_71456

/-- The acute angle formed by the asymptotes of a hyperbola with eccentricity 2 is 60°. -/
theorem hyperbola_asymptote_angle (e : ℝ) (h : e = 2) :
  let a : ℝ := 1  -- Arbitrary choice for a, as the angle is independent of a's value
  let b : ℝ := Real.sqrt 3 * a
  let asymptote_angle : ℝ := 2 * Real.arctan (b / a)
  asymptote_angle = π / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l714_71456


namespace NUMINAMATH_CALUDE_projection_equality_non_right_triangle_l714_71460

/-- Theorem: Projection equality in non-right triangles -/
theorem projection_equality_non_right_triangle 
  (a b c : ℝ) 
  (h_non_right : ¬(a^2 = b^2 + c^2)) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b) :
  ∃ (c'_b c'_c : ℝ),
    (c'_b = c * (b • c) / (b • b)) ∧ 
    (c'_c = b * (b • c) / (c • c)) ∧
    (a^2 = b^2 + c^2 + 2 * b * c'_b) ∧
    (a^2 = b^2 + c^2 + 2 * c * c'_c) :=
sorry

end NUMINAMATH_CALUDE_projection_equality_non_right_triangle_l714_71460


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l714_71480

theorem probability_of_white_ball
  (P_red P_black P_yellow P_white : ℝ)
  (h_red : P_red = 1/3)
  (h_black_yellow : P_black + P_yellow = 5/12)
  (h_yellow_white : P_yellow + P_white = 5/12)
  (h_sum : P_red + P_black + P_yellow + P_white = 1)
  (h_nonneg : P_red ≥ 0 ∧ P_black ≥ 0 ∧ P_yellow ≥ 0 ∧ P_white ≥ 0) :
  P_white = 1/4 :=
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l714_71480


namespace NUMINAMATH_CALUDE_specialSquaresTheorem_l714_71419

/-- Function to check if a number contains the digits 0 or 5 -/
def containsZeroOrFive (n : ℕ) : Bool :=
  sorry

/-- Function to delete the second digit of a number -/
def deleteSecondDigit (n : ℕ) : ℕ :=
  sorry

/-- The set of perfect squares satisfying the given conditions -/
def specialSquares : Finset ℕ :=
  sorry

theorem specialSquaresTheorem : specialSquares = {16, 36, 121, 484} := by
  sorry

end NUMINAMATH_CALUDE_specialSquaresTheorem_l714_71419


namespace NUMINAMATH_CALUDE_complex_product_real_l714_71455

theorem complex_product_real (b : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := 2 + b * I
  (z₁ * z₂).im = 0 → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l714_71455


namespace NUMINAMATH_CALUDE_cats_total_l714_71489

theorem cats_total (initial_cats bought_cats : Float) : 
  initial_cats = 11.0 → bought_cats = 43.0 → initial_cats + bought_cats = 54.0 := by
  sorry

end NUMINAMATH_CALUDE_cats_total_l714_71489


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l714_71472

theorem geometric_series_common_ratio : 
  let a₁ : ℚ := 7/8
  let a₂ : ℚ := -5/12
  let a₃ : ℚ := 35/144
  let r : ℚ := a₂ / a₁
  (∀ n : ℕ, n ≥ 2 → a₂ / a₁ = a₃ / a₂) →
  r = -10/21 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l714_71472


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l714_71449

theorem prime_square_mod_twelve (p : Nat) (h_prime : Nat.Prime p) (h_gt_three : p > 3) :
  p^2 % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l714_71449


namespace NUMINAMATH_CALUDE_coin_in_corner_l714_71402

/-- Represents a 2×n rectangle with coins --/
structure Rectangle (n : ℕ) where
  coins : Fin 2 → Fin n → ℕ

/-- Represents an operation of moving coins --/
inductive Operation
  | MoveRight : Fin 2 → Fin n → Operation
  | MoveUp : Fin 2 → Fin n → Operation

/-- Applies an operation to a rectangle --/
def applyOperation (rect : Rectangle n) (op : Operation) : Rectangle n :=
  sorry

/-- Checks if a sequence of operations results in a coin in (1,n) --/
def validSequence (rect : Rectangle n) (ops : List Operation) : Prop :=
  sorry

/-- Main theorem: There exists a sequence of operations to put a coin in (1,n) --/
theorem coin_in_corner (n : ℕ) (rect : Rectangle n) : 
  ∃ (ops : List Operation), validSequence rect ops :=
sorry

end NUMINAMATH_CALUDE_coin_in_corner_l714_71402


namespace NUMINAMATH_CALUDE_sin_cos_330_degrees_l714_71452

theorem sin_cos_330_degrees :
  Real.sin (330 * π / 180) = -1/2 ∧ Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_330_degrees_l714_71452


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l714_71440

theorem complex_simplification_and_multiplication :
  -2 * ((5 - 3 * Complex.I) - (2 + 5 * Complex.I)) = -6 + 16 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l714_71440


namespace NUMINAMATH_CALUDE_cubic_quadratic_inequality_l714_71438

theorem cubic_quadratic_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a < b) :
  a^3 * b^2 < a^2 * b^3 := by
sorry

end NUMINAMATH_CALUDE_cubic_quadratic_inequality_l714_71438


namespace NUMINAMATH_CALUDE_line_equation_proof_l714_71458

-- Define the given line
def given_line (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x - 2

-- Define the point through which the desired line passes
def point : ℝ × ℝ := (-1, 1)

-- Define the slope of the desired line
def desired_slope (m : ℝ) : Prop := m = 2 * (Real.sqrt 2 / 2)

-- Define the equation of the desired line
def desired_line (x y : ℝ) : Prop := y - 1 = Real.sqrt 2 * (x + 1)

-- Theorem statement
theorem line_equation_proof :
  ∀ (x y : ℝ),
  given_line x y →
  desired_slope (Real.sqrt 2) →
  desired_line point.1 point.2 →
  desired_line x y :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l714_71458
