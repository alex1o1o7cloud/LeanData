import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_cos_c_l1374_137417

theorem right_triangle_cos_c (A B C : ℝ) (h1 : A + B + C = Real.pi) 
  (h2 : A = Real.pi / 2) (h3 : Real.sin B = 3 / 5) : Real.cos C = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cos_c_l1374_137417


namespace NUMINAMATH_CALUDE_sam_total_pennies_l1374_137446

def initial_pennies : ℕ := 98
def found_pennies : ℕ := 93

theorem sam_total_pennies :
  initial_pennies + found_pennies = 191 := by
  sorry

end NUMINAMATH_CALUDE_sam_total_pennies_l1374_137446


namespace NUMINAMATH_CALUDE_max_n_theorem_l1374_137411

/-- Represents a convex polygon with interior points -/
structure ConvexPolygonWithInteriorPoints where
  n : ℕ  -- number of vertices in the polygon
  interior_points : ℕ  -- number of interior points
  no_collinear : Bool  -- no three points are collinear

/-- Calculates the number of triangles formed in a polygon with interior points -/
def num_triangles (p : ConvexPolygonWithInteriorPoints) : ℕ :=
  p.n + p.interior_points + 198

/-- The maximum value of n for which no more than 300 triangles are formed -/
def max_n_for_300_triangles : ℕ := 102

/-- Theorem stating the maximum value of n for which no more than 300 triangles are formed -/
theorem max_n_theorem (p : ConvexPolygonWithInteriorPoints) 
    (h1 : p.interior_points = 100)
    (h2 : p.no_collinear = true) :
    (∀ m : ℕ, m > max_n_for_300_triangles → num_triangles { n := m, interior_points := 100, no_collinear := true } > 300) ∧
    num_triangles { n := max_n_for_300_triangles, interior_points := 100, no_collinear := true } ≤ 300 :=
  sorry

end NUMINAMATH_CALUDE_max_n_theorem_l1374_137411


namespace NUMINAMATH_CALUDE_smallest_solution_quadratic_l1374_137495

theorem smallest_solution_quadratic (x : ℝ) : 
  (3 * x^2 + 36 * x - 60 = x * (x + 17)) → x ≥ -12 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_quadratic_l1374_137495


namespace NUMINAMATH_CALUDE_class_election_is_survey_conduction_l1374_137443

/-- Represents the steps in a survey process -/
inductive SurveyStep
  | DetermineObject
  | SelectMethod
  | Conduct
  | DrawConclusions

/-- Represents a voting process in a class election -/
structure ClassElection where
  students : Set Student
  candidates : Set Candidate
  ballot_box : Set Vote

/-- Definition of conducting a survey -/
def conducSurvey (process : ClassElection) : SurveyStep :=
  SurveyStep.Conduct

theorem class_election_is_survey_conduction (election : ClassElection) :
  conducSurvey election = SurveyStep.Conduct := by
  sorry

#check class_election_is_survey_conduction

end NUMINAMATH_CALUDE_class_election_is_survey_conduction_l1374_137443


namespace NUMINAMATH_CALUDE_race_speed_ratio_l1374_137431

theorem race_speed_ratio (L : ℝ) (h_L : L > 0) : 
  let head_start := 0.35 * L
  let winning_distance := 0.25 * L
  let a_distance := L + head_start
  let b_distance := L + winning_distance
  ∃ R : ℝ, R * (L / b_distance) = a_distance / b_distance ∧ R = 1.08 :=
by sorry

end NUMINAMATH_CALUDE_race_speed_ratio_l1374_137431


namespace NUMINAMATH_CALUDE_largest_positive_integer_solution_l1374_137499

theorem largest_positive_integer_solution :
  ∀ x : ℕ+, 2 * (x + 1) ≥ 5 * x - 3 ↔ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_largest_positive_integer_solution_l1374_137499


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sums_l1374_137440

theorem polynomial_coefficient_sums (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x : ℝ, (x + 2)^8 = a₀ + a₁*(x + 1) + a₂*(x + 1)^2 + a₃*(x + 1)^3 + 
                        a₄*(x + 1)^4 + a₅*(x + 1)^5 + a₆*(x + 1)^6 + 
                        a₇*(x + 1)^7 + a₈*(x + 1)^8) →
  (a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 255 ∧
   a₁ + a₃ + a₅ + a₇ = 128) := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sums_l1374_137440


namespace NUMINAMATH_CALUDE_tom_ran_median_distance_l1374_137429

def runners : Finset String := {"Phil", "Tom", "Pete", "Amal", "Sanjay"}

def distance : String → ℝ
| "Phil" => 4
| "Tom" => 6
| "Pete" => 2
| "Amal" => 8
| "Sanjay" => 7
| _ => 0

def isMedian (x : ℝ) (s : Finset ℝ) : Prop :=
  2 * (s.filter (· ≤ x)).card ≥ s.card ∧
  2 * (s.filter (· ≥ x)).card ≥ s.card

theorem tom_ran_median_distance :
  isMedian (distance "Tom") (runners.image distance) :=
sorry

end NUMINAMATH_CALUDE_tom_ran_median_distance_l1374_137429


namespace NUMINAMATH_CALUDE_cube_volume_from_side_area_l1374_137491

theorem cube_volume_from_side_area (side_area : ℝ) (h : side_area = 64) :
  let side_length := Real.sqrt side_area
  side_length ^ 3 = 512 := by sorry

end NUMINAMATH_CALUDE_cube_volume_from_side_area_l1374_137491


namespace NUMINAMATH_CALUDE_sum_of_i_powers_2021_to_2024_l1374_137484

/-- The imaginary unit i -/
def i : ℂ := Complex.I

/-- Theorem: The sum of i^2021, i^2022, i^2023, and i^2024 is equal to 0 -/
theorem sum_of_i_powers_2021_to_2024 : i^2021 + i^2022 + i^2023 + i^2024 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_2021_to_2024_l1374_137484


namespace NUMINAMATH_CALUDE_arithmetic_mean_fractions_l1374_137498

theorem arithmetic_mean_fractions (x a : ℝ) (hx : x ≠ 0) :
  (1 / 2) * ((x + 2 * a) / x + (x - 3 * a) / x) = 1 - a / (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fractions_l1374_137498


namespace NUMINAMATH_CALUDE_f_simplification_inverse_sum_value_l1374_137455

noncomputable def f (α : Real) : Real :=
  (Real.sin (2 * Real.pi - α) * Real.cos (Real.pi + α) * Real.cos (Real.pi / 2 - α) * Real.cos (11 * Real.pi / 2 - α)) /
  (Real.sin (3 * Real.pi - α) * Real.cos (Real.pi / 2 + α) * Real.sin (9 * Real.pi / 2 + α)) +
  Real.cos (2 * Real.pi - α)

theorem f_simplification (α : Real) : f α = Real.sin α + Real.cos α := by sorry

theorem inverse_sum_value (α : Real) (h : f α = Real.sqrt 10 / 5) :
  1 / Real.sin α + 1 / Real.cos α = -4 * Real.sqrt 10 / 3 := by sorry

end NUMINAMATH_CALUDE_f_simplification_inverse_sum_value_l1374_137455


namespace NUMINAMATH_CALUDE_puzzles_sum_is_five_l1374_137419

def alphabet_value (n : ℕ) : ℤ :=
  match n % 8 with
  | 0 => 0
  | 1 => -1
  | 2 => 2
  | 3 => -1
  | 4 => 0
  | 5 => 1
  | 6 => -2
  | 7 => 1
  | _ => 0 -- This case should never occur, but Lean requires it for completeness

def letter_position (c : Char) : ℕ :=
  match c with
  | 'p' => 16
  | 'u' => 21
  | 'z' => 26
  | 'l' => 12
  | 'e' => 5
  | 's' => 19
  | _ => 0 -- Default case for other characters

theorem puzzles_sum_is_five :
  (alphabet_value (letter_position 'p') +
   alphabet_value (letter_position 'u') +
   alphabet_value (letter_position 'z') +
   alphabet_value (letter_position 'z') +
   alphabet_value (letter_position 'l') +
   alphabet_value (letter_position 'e') +
   alphabet_value (letter_position 's')) = 5 := by
  sorry

end NUMINAMATH_CALUDE_puzzles_sum_is_five_l1374_137419


namespace NUMINAMATH_CALUDE_inverse_of_A_cubed_l1374_137453

variable (A : Matrix (Fin 2) (Fin 2) ℝ)

theorem inverse_of_A_cubed 
  (h : A⁻¹ = ![![-3, 2], ![-1, 3]]) : 
  (A^3)⁻¹ = ![![-21, 14], ![-7, 21]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_cubed_l1374_137453


namespace NUMINAMATH_CALUDE_pizza_order_total_l1374_137485

theorem pizza_order_total (m : ℕ) (total_pizzas : ℚ) : 
  m > 17 →
  (10 : ℚ) / m + 17 * ((10 : ℚ) / m) / 2 = total_pizzas →
  total_pizzas = 11 := by
  sorry

end NUMINAMATH_CALUDE_pizza_order_total_l1374_137485


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l1374_137402

theorem no_solution_implies_a_leq_two (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |3 - x| ≥ 2*a + 1) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l1374_137402


namespace NUMINAMATH_CALUDE_xyz_sum_l1374_137470

theorem xyz_sum (x y z : ℕ+) 
  (h1 : x * y + z = 47)
  (h2 : y * z + x = 47)
  (h3 : z * x + y = 47) :
  x + y + z = 48 := by
  sorry

end NUMINAMATH_CALUDE_xyz_sum_l1374_137470


namespace NUMINAMATH_CALUDE_derivative_f_at_zero_l1374_137449

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.log (1 - Real.sin (x^3 * Real.sin (1/x)))
  else 0

theorem derivative_f_at_zero :
  deriv f 0 = 0 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_zero_l1374_137449


namespace NUMINAMATH_CALUDE_part_one_part_two_l1374_137410

-- Definitions for propositions q and p
def prop_q (a : ℝ) : Prop := ∃ x : ℝ, -2 < x ∧ x < 2 ∧ x^2 - 2*x - a = 0

def prop_p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + (a-1)*x + 4 < 0

-- Part 1
theorem part_one (a : ℝ) : prop_q a ∨ prop_p a → a < -3 ∨ a ≥ -1 := by sorry

-- Definitions for part 2
def prop_p_part2 (a : ℝ) : Prop := ∃ x : ℝ, 2*a < x ∧ x < a + 1

-- Part 2
theorem part_two (a : ℝ) : 
  (∀ x : ℝ, prop_p_part2 a → prop_q a) ∧ 
  (∃ x : ℝ, prop_q a ∧ ¬prop_p_part2 a) → 
  a ≥ -1/2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1374_137410


namespace NUMINAMATH_CALUDE_subset_transitive_and_complement_subset_l1374_137460

variable {α : Type*}
variable (U : Set α)

theorem subset_transitive_and_complement_subset : 
  (∀ A B C : Set α, A ⊆ B → B ⊆ C → A ⊆ C) ∧ 
  (∀ A B : Set α, A ⊆ B → (U \ B) ⊆ (U \ A)) :=
sorry

end NUMINAMATH_CALUDE_subset_transitive_and_complement_subset_l1374_137460


namespace NUMINAMATH_CALUDE_hawks_score_l1374_137409

/-- 
Given the total points scored and the winning margin in a basketball game,
this theorem proves the score of the losing team.
-/
theorem hawks_score (total_points winning_margin : ℕ) 
  (h1 : total_points = 42)
  (h2 : winning_margin = 6) : 
  (total_points - winning_margin) / 2 = 18 := by
  sorry

end NUMINAMATH_CALUDE_hawks_score_l1374_137409


namespace NUMINAMATH_CALUDE_unique_intersection_point_l1374_137481

/-- Two equations y = x^2 and y = 2x + k intersect at exactly one point if and only if k = 0 -/
theorem unique_intersection_point (k : ℝ) : 
  (∃! x : ℝ, x^2 = 2*x + k) ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l1374_137481


namespace NUMINAMATH_CALUDE_linear_function_slope_l1374_137487

theorem linear_function_slope (x₁ x₂ y₁ y₂ m : ℝ) :
  x₁ > x₂ →
  y₁ > y₂ →
  y₁ = (m - 3) * x₁ - 4 →
  y₂ = (m - 3) * x₂ - 4 →
  m > 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_slope_l1374_137487


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1374_137448

theorem sufficient_not_necessary : 
  (∃ m : ℝ, m = 9 → m > 8) ∧ 
  (∃ m : ℝ, m > 8 ∧ m ≠ 9) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1374_137448


namespace NUMINAMATH_CALUDE_batsman_matches_l1374_137482

theorem batsman_matches (total_matches : ℕ) (first_set_matches : ℕ) (first_set_avg : ℝ) 
                         (second_set_avg : ℝ) (total_avg : ℝ) :
  total_matches = 30 →
  first_set_matches = 20 →
  first_set_avg = 30 →
  second_set_avg = 15 →
  total_avg = 25 →
  (total_matches - first_set_matches : ℝ) = 10 := by
  sorry


end NUMINAMATH_CALUDE_batsman_matches_l1374_137482


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_set_l1374_137420

def number_set : List ℝ := [16, 23, 38, 11.5]

theorem arithmetic_mean_of_set : 
  (number_set.sum / number_set.length : ℝ) = 22.125 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_set_l1374_137420


namespace NUMINAMATH_CALUDE_M_inter_N_eq_l1374_137457

def M : Set ℤ := {x | -3 < x ∧ x < 3}
def N : Set ℤ := {x | x < 1}

theorem M_inter_N_eq : M ∩ N = {-2, -1, 0} := by
  sorry

end NUMINAMATH_CALUDE_M_inter_N_eq_l1374_137457


namespace NUMINAMATH_CALUDE_number_problem_l1374_137496

theorem number_problem (x : ℝ) : 0.20 * x - 4 = 6 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1374_137496


namespace NUMINAMATH_CALUDE_max_remainder_is_456_l1374_137478

/-- The maximum number on the board initially -/
def max_initial : ℕ := 2012

/-- The sum of numbers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of operations performed -/
def num_operations : ℕ := max_initial - 1

/-- The final number N after all operations -/
def final_number : ℕ := sum_to_n max_initial * 2^num_operations

theorem max_remainder_is_456 : final_number % 1000 = 456 := by
  sorry

end NUMINAMATH_CALUDE_max_remainder_is_456_l1374_137478


namespace NUMINAMATH_CALUDE_pure_imaginary_z_l1374_137477

theorem pure_imaginary_z (a : ℝ) : 
  (∃ (b : ℝ), (1 - a * Complex.I) / (1 + a * Complex.I) = Complex.I * b) → 
  (a = 1 ∨ a = -1) := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_z_l1374_137477


namespace NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1374_137468

/-- 
Given an equilateral triangle where one of its sides is also a side of an isosceles triangle,
this theorem proves that if the isosceles triangle has a perimeter of 65 and a base of 25,
then the perimeter of the equilateral triangle is 60.
-/
theorem equilateral_triangle_perimeter 
  (s : ℝ) 
  (h_isosceles_perimeter : s + s + 25 = 65) 
  (h_equilateral_side : s > 0) : 
  3 * s = 60 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_perimeter_l1374_137468


namespace NUMINAMATH_CALUDE_derivative_at_x0_l1374_137442

theorem derivative_at_x0 (f : ℝ → ℝ) (x₀ : ℝ) (h : Differentiable ℝ f) :
  (∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |(f (x₀ - 2*Δx) - f x₀) / Δx - 2| < ε) →
  deriv f x₀ = -1 := by
sorry

end NUMINAMATH_CALUDE_derivative_at_x0_l1374_137442


namespace NUMINAMATH_CALUDE_two_invariant_lines_l1374_137421

/-- The transformation f: ℝ² → ℝ² defined by f(x,y) = (3y,2x) -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (3 * p.2, 2 * p.1)

/-- A line in ℝ² represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A line is invariant under f if for all points on the line, 
    their images under f also lie on the same line -/
def is_invariant (l : Line) : Prop :=
  ∀ x y : ℝ, y = l.slope * x + l.intercept → 
    (f (x, y)).2 = l.slope * (f (x, y)).1 + l.intercept

/-- There are exactly two distinct lines that are invariant under f -/
theorem two_invariant_lines : 
  ∃! (l1 l2 : Line), l1 ≠ l2 ∧ is_invariant l1 ∧ is_invariant l2 ∧
    (∀ l : Line, is_invariant l → l = l1 ∨ l = l2) :=
sorry

end NUMINAMATH_CALUDE_two_invariant_lines_l1374_137421


namespace NUMINAMATH_CALUDE_inequality_proof_l1374_137488

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a + b + c = 1) : 
  (a * b + b * c + c * a ≤ 1/3) ∧ 
  (b^2 / a + c^2 / b + a^2 / c ≥ 1) := by
sorry


end NUMINAMATH_CALUDE_inequality_proof_l1374_137488


namespace NUMINAMATH_CALUDE_vance_family_stamp_cost_difference_l1374_137458

theorem vance_family_stamp_cost_difference :
  let mr_rooster_count : ℕ := 3
  let mr_rooster_price : ℚ := 3/2
  let mr_daffodil_count : ℕ := 5
  let mr_daffodil_price : ℚ := 3/4
  let mrs_rooster_count : ℕ := 2
  let mrs_rooster_price : ℚ := 5/4
  let mrs_daffodil_count : ℕ := 7
  let mrs_daffodil_price : ℚ := 4/5
  let john_rooster_count : ℕ := 4
  let john_rooster_price : ℚ := 7/5
  let john_daffodil_count : ℕ := 3
  let john_daffodil_price : ℚ := 7/10

  let total_rooster_cost : ℚ := 
    mr_rooster_count * mr_rooster_price + 
    mrs_rooster_count * mrs_rooster_price + 
    john_rooster_count * john_rooster_price

  let total_daffodil_cost : ℚ := 
    mr_daffodil_count * mr_daffodil_price + 
    mrs_daffodil_count * mrs_daffodil_price + 
    john_daffodil_count * john_daffodil_price

  total_rooster_cost - total_daffodil_cost = 23/20
  := by sorry

end NUMINAMATH_CALUDE_vance_family_stamp_cost_difference_l1374_137458


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1374_137405

theorem quadratic_inequality (x : ℝ) : 
  3 * x^2 + 2 * x - 3 > 10 - 2 * x ↔ x < (-2 - Real.sqrt 43) / 3 ∨ x > (-2 + Real.sqrt 43) / 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1374_137405


namespace NUMINAMATH_CALUDE_chili_beans_cans_l1374_137425

/-- Given a ratio of tomato soup cans to chili beans cans and the total number of cans,
    calculate the number of chili beans cans. -/
theorem chili_beans_cans (tomato_ratio chili_ratio total_cans : ℕ) :
  tomato_ratio ≠ 0 →
  chili_ratio = 2 * tomato_ratio →
  total_cans = tomato_ratio + chili_ratio →
  chili_ratio = 8 := by
  sorry

end NUMINAMATH_CALUDE_chili_beans_cans_l1374_137425


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l1374_137406

theorem largest_angle_in_triangle (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  a = 45 →           -- One angle is 45°
  5 * b = 4 * c →    -- The other two angles are in the ratio 4:5
  max a (max b c) = 75 -- The largest angle is 75°
  := by sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l1374_137406


namespace NUMINAMATH_CALUDE_derricks_yard_length_l1374_137464

theorem derricks_yard_length (derrick_length alex_length brianne_length : ℝ) : 
  alex_length = derrick_length / 2 →
  brianne_length = 6 * alex_length →
  brianne_length = 30 →
  derrick_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_derricks_yard_length_l1374_137464


namespace NUMINAMATH_CALUDE_radical_product_simplification_l1374_137459

theorem radical_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (14 * q) = 14 * q * Real.sqrt (21 * q) :=
by sorry

end NUMINAMATH_CALUDE_radical_product_simplification_l1374_137459


namespace NUMINAMATH_CALUDE_infinite_solutions_exponential_equation_l1374_137452

theorem infinite_solutions_exponential_equation :
  ∀ x : ℝ, (2 : ℝ) ^ (6 * x + 3) * (4 : ℝ) ^ (3 * x + 6) = (8 : ℝ) ^ (4 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_exponential_equation_l1374_137452


namespace NUMINAMATH_CALUDE_cage_cost_proof_l1374_137432

def snake_toy_cost : ℝ := 11.76
def dollar_found : ℝ := 1
def total_cost : ℝ := 26.3

theorem cage_cost_proof :
  total_cost - (snake_toy_cost + dollar_found) = 13.54 := by
  sorry

end NUMINAMATH_CALUDE_cage_cost_proof_l1374_137432


namespace NUMINAMATH_CALUDE_power_function_through_point_and_value_l1374_137450

/-- A power function that passes through the point (2,8) -/
def f (x : ℝ) : ℝ := x^3

theorem power_function_through_point_and_value : 
  f 2 = 8 ∧ ∃ x : ℝ, f x = 27 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_power_function_through_point_and_value_l1374_137450


namespace NUMINAMATH_CALUDE_seminar_discount_percentage_l1374_137403

/-- Calculates the discount percentage for early registration of a seminar --/
theorem seminar_discount_percentage
  (regular_fee : ℝ)
  (num_teachers : ℕ)
  (food_allowance : ℝ)
  (total_spent : ℝ)
  (h1 : regular_fee = 150)
  (h2 : num_teachers = 10)
  (h3 : food_allowance = 10)
  (h4 : total_spent = 1525)
  : (1 - (total_spent - num_teachers * food_allowance) / (num_teachers * regular_fee)) * 100 = 5 := by
  sorry

#check seminar_discount_percentage

end NUMINAMATH_CALUDE_seminar_discount_percentage_l1374_137403


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_five_l1374_137461

theorem sum_of_solutions_eq_five :
  let f : ℝ → ℝ := λ M => M * (M - 5) + 9
  ∃ M₁ M₂ : ℝ, (f M₁ = 0 ∧ f M₂ = 0 ∧ M₁ ≠ M₂) ∧ M₁ + M₂ = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_five_l1374_137461


namespace NUMINAMATH_CALUDE_teacher_assignment_theorem_l1374_137492

/-- The number of ways to assign 4 teachers to 3 schools, with each school having at least 1 teacher -/
def teacher_assignment_count : ℕ := 36

/-- The number of teachers -/
def num_teachers : ℕ := 4

/-- The number of schools -/
def num_schools : ℕ := 3

theorem teacher_assignment_theorem :
  (∀ assignment : Fin num_teachers → Fin num_schools,
    (∀ s : Fin num_schools, ∃ t : Fin num_teachers, assignment t = s) →
    (∃ s : Fin num_schools, ∃ t₁ t₂ : Fin num_teachers, t₁ ≠ t₂ ∧ assignment t₁ = s ∧ assignment t₂ = s)) →
  (Fintype.card {assignment : Fin num_teachers → Fin num_schools |
    ∀ s : Fin num_schools, ∃ t : Fin num_teachers, assignment t = s}) = teacher_assignment_count :=
by sorry

#check teacher_assignment_theorem

end NUMINAMATH_CALUDE_teacher_assignment_theorem_l1374_137492


namespace NUMINAMATH_CALUDE_vector_simplification_l1374_137430

variable {V : Type*} [AddCommGroup V]
variable (A B C D F : V)

theorem vector_simplification :
  (C - D) + (B - C) + (A - B) = A - D ∧
  (A - B) + (D - F) + (C - D) + (B - C) + (F - A) = 0 := by
  sorry

end NUMINAMATH_CALUDE_vector_simplification_l1374_137430


namespace NUMINAMATH_CALUDE_max_value_a_l1374_137415

theorem max_value_a (a b c d : ℕ+) 
  (h1 : a < 3 * b)
  (h2 : b < 4 * c)
  (h3 : c < 5 * d)
  (h4 : d < 50) :
  a ≤ 2924 ∧ ∃ (a' b' c' d' : ℕ+), 
    a' = 2924 ∧ 
    a' < 3 * b' ∧ 
    b' < 4 * c' ∧ 
    c' < 5 * d' ∧ 
    d' < 50 :=
sorry

end NUMINAMATH_CALUDE_max_value_a_l1374_137415


namespace NUMINAMATH_CALUDE_daily_savings_l1374_137423

def original_coffees : ℕ := 4
def original_price : ℚ := 2
def price_increase_percentage : ℚ := 50
def new_coffees_ratio : ℚ := 1/2

def original_spending : ℚ := original_coffees * original_price

def new_price : ℚ := original_price * (1 + price_increase_percentage / 100)
def new_coffees : ℚ := original_coffees * new_coffees_ratio
def new_spending : ℚ := new_coffees * new_price

theorem daily_savings : original_spending - new_spending = 2 := by sorry

end NUMINAMATH_CALUDE_daily_savings_l1374_137423


namespace NUMINAMATH_CALUDE_runs_ratio_l1374_137483

/-- Represents the runs scored by each player -/
structure Runs where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The conditions of the cricket match -/
def cricket_match (r : Runs) : Prop :=
  r.a + r.b + r.c = 95 ∧
  r.c = 75 ∧
  r.a * 3 = r.b * 1

/-- The theorem to prove -/
theorem runs_ratio (r : Runs) (h : cricket_match r) : 
  r.b * 5 = r.c * 1 := by
sorry


end NUMINAMATH_CALUDE_runs_ratio_l1374_137483


namespace NUMINAMATH_CALUDE_training_end_time_l1374_137467

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  hValid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

theorem training_end_time :
  let startTime : Time := ⟨8, 0, sorry⟩
  let sessionDuration : ℕ := 40
  let breakDuration : ℕ := 15
  let numSessions : ℕ := 4
  let totalDuration := numSessions * sessionDuration + (numSessions - 1) * breakDuration
  let endTime := addMinutes startTime totalDuration
  endTime = ⟨11, 25, sorry⟩ := by sorry

end NUMINAMATH_CALUDE_training_end_time_l1374_137467


namespace NUMINAMATH_CALUDE_additional_decorations_to_buy_l1374_137463

def halloween_decorations (skulls broomsticks spiderwebs pumpkins cauldron left_to_put_up total : ℕ) : Prop :=
  skulls = 12 ∧
  broomsticks = 4 ∧
  spiderwebs = 12 ∧
  pumpkins = 2 * spiderwebs ∧
  cauldron = 1 ∧
  left_to_put_up = 10 ∧
  total = 83

theorem additional_decorations_to_buy 
  (skulls broomsticks spiderwebs pumpkins cauldron left_to_put_up total : ℕ)
  (h : halloween_decorations skulls broomsticks spiderwebs pumpkins cauldron left_to_put_up total) :
  total - (skulls + broomsticks + spiderwebs + pumpkins + cauldron) - left_to_put_up = 20 :=
sorry

end NUMINAMATH_CALUDE_additional_decorations_to_buy_l1374_137463


namespace NUMINAMATH_CALUDE_distance_between_points_l1374_137497

/-- The distance between points A and B -/
def distance : ℝ := sorry

/-- The speed of the first pedestrian -/
def speed1 : ℝ := sorry

/-- The speed of the second pedestrian -/
def speed2 : ℝ := sorry

theorem distance_between_points (h1 : distance / (2 * speed1) = 15 / speed2)
                                (h2 : 24 / speed1 = distance / (2 * speed2))
                                (h3 : distance / speed1 = distance / speed2) :
  distance = 40 := by sorry

end NUMINAMATH_CALUDE_distance_between_points_l1374_137497


namespace NUMINAMATH_CALUDE_brothers_multiple_l1374_137472

/-- Given that Aaron has 4 brothers and Bennett has 6 brothers, 
    prove that the multiple relating their number of brothers is 2. -/
theorem brothers_multiple (aaron_brothers : ℕ) (bennett_brothers : ℕ) : 
  aaron_brothers = 4 → bennett_brothers = 6 → ∃ x : ℕ, x * aaron_brothers - 2 = bennett_brothers ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_brothers_multiple_l1374_137472


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l1374_137436

theorem slower_speed_calculation (actual_distance : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) :
  actual_distance = 40 →
  faster_speed = 12 →
  additional_distance = 20 →
  ∃ slower_speed : ℝ, 
    slower_speed > 0 ∧
    actual_distance / slower_speed = (actual_distance + additional_distance) / faster_speed ∧
    slower_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l1374_137436


namespace NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l1374_137437

/-- Represents a point in spherical coordinates -/
structure SphericalPoint where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Checks if a SphericalPoint is in standard representation -/
def isStandardRepresentation (p : SphericalPoint) : Prop :=
  p.ρ > 0 ∧ 0 ≤ p.θ ∧ p.θ < 2 * Real.pi ∧ 0 ≤ p.φ ∧ p.φ ≤ Real.pi

/-- Theorem stating the equivalence of the given spherical coordinates -/
theorem spherical_coordinate_equivalence :
  let p1 := SphericalPoint.mk 4 (5 * Real.pi / 6) (9 * Real.pi / 4)
  let p2 := SphericalPoint.mk 4 (11 * Real.pi / 6) (Real.pi / 4)
  (p1.ρ = p2.ρ) ∧ 
  (p1.θ % (2 * Real.pi) = p2.θ % (2 * Real.pi)) ∧ 
  (p1.φ % (2 * Real.pi) = p2.φ % (2 * Real.pi)) ∧
  isStandardRepresentation p2 :=
by sorry

end NUMINAMATH_CALUDE_spherical_coordinate_equivalence_l1374_137437


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1374_137490

theorem min_value_quadratic (x : ℝ) : x^2 + 4*x + 5 ≥ 1 ∧ (x^2 + 4*x + 5 = 1 ↔ x = -2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1374_137490


namespace NUMINAMATH_CALUDE_milk_distribution_l1374_137454

theorem milk_distribution (boxes : Nat) (bottles_per_box : Nat) (eaten : Nat) (people : Nat) :
  boxes = 7 →
  bottles_per_box = 9 →
  eaten = 7 →
  people = 8 →
  (boxes * bottles_per_box - eaten) / people = 7 := by
  sorry

end NUMINAMATH_CALUDE_milk_distribution_l1374_137454


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l1374_137462

/-- An isosceles triangle with given leg and base lengths -/
structure IsoscelesTriangle where
  leg : ℝ
  base : ℝ

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.leg + t.base

/-- The perimeter of a quadrilateral formed by joining two isosceles triangles along their legs -/
def perimeterQuadLeg (t : IsoscelesTriangle) : ℝ := 2 * t.base + 2 * t.leg

/-- The perimeter of a quadrilateral formed by joining two isosceles triangles along their bases -/
def perimeterQuadBase (t : IsoscelesTriangle) : ℝ := 4 * t.leg

theorem isosceles_triangle_sides (t : IsoscelesTriangle) :
  perimeter t = 100 ∧
  perimeterQuadLeg t + 4 = perimeterQuadBase t →
  t.leg = 34 ∧ t.base = 32 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l1374_137462


namespace NUMINAMATH_CALUDE_minimize_sum_of_distances_l1374_137469

/-- Given points P, Q, and R in ℝ², if R is chosen to minimize the sum of distances |PR| + |RQ|, then R lies on the line segment PQ. -/
theorem minimize_sum_of_distances (P Q R : ℝ × ℝ) :
  P = (-2, -2) →
  Q = (0, -1) →
  R.1 = 2 →
  (∀ S : ℝ × ℝ, dist P R + dist R Q ≤ dist P S + dist S Q) →
  R.2 = 0 := by sorry


end NUMINAMATH_CALUDE_minimize_sum_of_distances_l1374_137469


namespace NUMINAMATH_CALUDE_similar_triangles_problem_l1374_137444

theorem similar_triangles_problem (A₁ A₂ : ℕ) (k : ℕ) (s : ℝ) :
  A₁ > A₂ →
  A₁ - A₂ = 18 →
  A₁ = k^2 * A₂ →
  s = 3 →
  (∃ (a b c : ℝ), A₂ = (a * b) / 2 ∧ c^2 = a^2 + b^2 ∧ s = c) →
  (∃ (a' b' c' : ℝ), A₁ = (a' * b') / 2 ∧ c'^2 = a'^2 + b'^2 ∧ 6 = c') :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_problem_l1374_137444


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_10_l1374_137426

/-- The displacement function of a moving object -/
def s (t : ℝ) : ℝ := 3 * t^2 - 2 * t + 1

/-- The velocity function derived from the displacement function -/
def v (t : ℝ) : ℝ := 6 * t - 2

theorem instantaneous_velocity_at_10 : v 10 = 58 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_10_l1374_137426


namespace NUMINAMATH_CALUDE_sum_of_sides_equals_two_point_five_l1374_137434

/-- Represents a polygon ABCDEFGH with given properties -/
structure Polygon where
  area : ℝ
  AB : ℝ
  BC : ℝ
  HA : ℝ

/-- The sum of lengths DE, EF, FG, and GH in the polygon -/
def sum_of_sides (p : Polygon) : ℝ := sorry

/-- Theorem stating that for a polygon with given properties, the sum of certain sides equals 2.5 -/
theorem sum_of_sides_equals_two_point_five (p : Polygon) 
  (h1 : p.area = 85)
  (h2 : p.AB = 7)
  (h3 : p.BC = 10)
  (h4 : p.HA = 6) :
  sum_of_sides p = 2.5 := by sorry

end NUMINAMATH_CALUDE_sum_of_sides_equals_two_point_five_l1374_137434


namespace NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l1374_137418

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio 
  (r : ℝ) -- radius of the sphere
  (k : ℝ) -- material density coefficient of the hemisphere
  (h : k = 2/3) -- given condition for k
  : (4/3 * Real.pi * r^3) / (k * 1/2 * 4/3 * Real.pi * (3*r)^3) = 2/27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l1374_137418


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l1374_137447

def banana_arrangements : ℕ :=
  Nat.factorial 6 / Nat.factorial 3

theorem banana_arrangements_count : banana_arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l1374_137447


namespace NUMINAMATH_CALUDE_tan_fifteen_identity_l1374_137489

theorem tan_fifteen_identity : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_identity_l1374_137489


namespace NUMINAMATH_CALUDE_f_7_equals_neg_2_l1374_137438

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic_4 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 4) = f x

theorem f_7_equals_neg_2 (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_periodic : is_periodic_4 f) 
  (h_interval : ∀ x ∈ Set.Ioo 0 2, f x = 2 * x^2) : 
  f 7 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_7_equals_neg_2_l1374_137438


namespace NUMINAMATH_CALUDE_max_ratio_two_digit_mean_50_l1374_137414

/-- Two-digit positive integer -/
def TwoDigitPositiveInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The mean of two numbers is 50 -/
def MeanIs50 (x y : ℕ) : Prop := (x + y) / 2 = 50

theorem max_ratio_two_digit_mean_50 :
  ∃ (x y : ℕ), TwoDigitPositiveInt x ∧ TwoDigitPositiveInt y ∧ MeanIs50 x y ∧
    ∀ (a b : ℕ), TwoDigitPositiveInt a → TwoDigitPositiveInt b → MeanIs50 a b →
      (a : ℚ) / b ≤ (x : ℚ) / y ∧ (x : ℚ) / y = 99 := by
  sorry

end NUMINAMATH_CALUDE_max_ratio_two_digit_mean_50_l1374_137414


namespace NUMINAMATH_CALUDE_soda_cost_lucille_soda_cost_l1374_137493

/-- The cost of Lucille's soda given her weeding earnings and remaining money -/
theorem soda_cost (cents_per_weed : ℕ) (flower_bed_weeds : ℕ) (vegetable_patch_weeds : ℕ) 
  (grass_weeds : ℕ) (remaining_cents : ℕ) : ℕ :=
  let total_weeds := flower_bed_weeds + vegetable_patch_weeds + grass_weeds / 2
  let total_earnings := total_weeds * cents_per_weed
  total_earnings - remaining_cents

/-- Proof that Lucille's soda cost 99 cents -/
theorem lucille_soda_cost : soda_cost 6 11 14 32 147 = 99 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_lucille_soda_cost_l1374_137493


namespace NUMINAMATH_CALUDE_min_value_is_11_l1374_137435

-- Define the variables and constraints
def is_feasible (x y : ℝ) : Prop :=
  x * y - 3 ≥ 0 ∧ x - y ≥ 1 ∧ y ≥ 5

-- Define the objective function
def objective_function (x y : ℝ) : ℝ :=
  3 * x + 4 * y

-- Theorem statement
theorem min_value_is_11 :
  ∀ x y : ℝ, is_feasible x y →
  objective_function x y ≥ 11 ∧
  ∃ x₀ y₀ : ℝ, is_feasible x₀ y₀ ∧ objective_function x₀ y₀ = 11 :=
sorry

end NUMINAMATH_CALUDE_min_value_is_11_l1374_137435


namespace NUMINAMATH_CALUDE_max_area_AEBF_l1374_137445

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- A line passing through the origin with positive slope -/
def line_through_origin (m : ℝ) (x y : ℝ) : Prop := y = m * x ∧ m > 0

/-- Point A -/
def point_A : ℝ × ℝ := (2, 0)

/-- Point B -/
def point_B : ℝ × ℝ := (0, 1)

/-- The intersection points E and F -/
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ ellipse x y ∧ line_through_origin m x y}

/-- The area of quadrilateral AEBF -/
noncomputable def area_AEBF (E F : ℝ × ℝ) : ℝ := sorry

/-- The theorem stating the maximum area of quadrilateral AEBF -/
theorem max_area_AEBF :
  ∃ m : ℝ, ∃ E F : ℝ × ℝ,
    E ∈ intersection_points m ∧
    F ∈ intersection_points m ∧
    E ≠ F ∧
    (∀ m' : ℝ, ∀ E' F' : ℝ × ℝ,
      E' ∈ intersection_points m' ∧
      F' ∈ intersection_points m' ∧
      E' ≠ F' →
      area_AEBF E F ≥ area_AEBF E' F') ∧
    area_AEBF E F = 2 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_max_area_AEBF_l1374_137445


namespace NUMINAMATH_CALUDE_sports_equipment_problem_l1374_137427

/-- Represents the purchase and selling prices of sports equipment -/
structure SportsPrices where
  tabletennis_purchase : ℝ
  badminton_purchase : ℝ
  tabletennis_sell : ℝ
  badminton_sell : ℝ

/-- Represents the number of sets and profit -/
structure SalesData where
  tabletennis_sets : ℝ
  profit : ℝ

/-- Theorem stating the conditions and results of the sports equipment problem -/
theorem sports_equipment_problem 
  (prices : SportsPrices)
  (sales : SalesData) :
  -- Conditions
  2 * prices.tabletennis_purchase + prices.badminton_purchase = 110 ∧
  4 * prices.tabletennis_purchase + 3 * prices.badminton_purchase = 260 ∧
  prices.tabletennis_sell = 50 ∧
  prices.badminton_sell = 60 ∧
  sales.tabletennis_sets ≤ 150 ∧
  sales.tabletennis_sets ≥ (300 - sales.tabletennis_sets) / 2 →
  -- Results
  prices.tabletennis_purchase = 35 ∧
  prices.badminton_purchase = 40 ∧
  sales.profit = -5 * sales.tabletennis_sets + 6000 ∧
  100 ≤ sales.tabletennis_sets ∧ sales.tabletennis_sets ≤ 150 ∧
  (∀ a : ℝ, 0 < a ∧ a < 10 →
    (a < 5 → sales.tabletennis_sets = 100) ∧
    (a > 5 → sales.tabletennis_sets = 150) ∧
    (a = 5 → sales.profit = 6000)) :=
by sorry

end NUMINAMATH_CALUDE_sports_equipment_problem_l1374_137427


namespace NUMINAMATH_CALUDE_certain_number_proof_l1374_137494

theorem certain_number_proof (x : ℚ) : 
  (5 / 6 : ℚ) * x = (5 / 16 : ℚ) * x + 300 → x = 576 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1374_137494


namespace NUMINAMATH_CALUDE_distance_between_cities_l1374_137451

/-- The distance between two cities given two trains traveling towards each other -/
theorem distance_between_cities (t : ℝ) (v₁ v₂ : ℝ) (h₁ : t = 4) (h₂ : v₁ = 115) (h₃ : v₂ = 85) :
  (v₁ + v₂) * t = 800 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_cities_l1374_137451


namespace NUMINAMATH_CALUDE_max_university_students_l1374_137400

theorem max_university_students (j m : ℕ) : 
  m = 2 * j + 100 →  -- Max's university has twice Julie's students plus 100
  m + j = 5400 →     -- Total students in both universities
  m = 3632           -- Number of students in Max's university
  := by sorry

end NUMINAMATH_CALUDE_max_university_students_l1374_137400


namespace NUMINAMATH_CALUDE_binary_representation_of_21_l1374_137439

theorem binary_representation_of_21 :
  (21 : ℕ).digits 2 = [1, 0, 1, 0, 1] :=
sorry

end NUMINAMATH_CALUDE_binary_representation_of_21_l1374_137439


namespace NUMINAMATH_CALUDE_calculator_decimal_correction_l1374_137428

theorem calculator_decimal_correction (x y : ℚ) (z : ℕ) :
  x = 0.065 →
  y = 3.25 →
  z = 21125 →
  (x * y : ℚ) = 0.21125 :=
by
  sorry

end NUMINAMATH_CALUDE_calculator_decimal_correction_l1374_137428


namespace NUMINAMATH_CALUDE_ball_distribution_theorem_l1374_137424

/-- The number of ways to distribute indistinguishable balls into boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ :=
  Nat.choose (num_balls + num_boxes - 1) num_balls

/-- The number of ways to distribute balls of three colors into boxes -/
def distribute_three_colors (num_balls_per_color : ℕ) (num_boxes : ℕ) : ℕ :=
  (distribute_balls num_balls_per_color num_boxes) ^ 3

theorem ball_distribution_theorem (num_balls_per_color num_boxes : ℕ) 
  (h1 : num_balls_per_color = 4) 
  (h2 : num_boxes = 6) : 
  distribute_three_colors num_balls_per_color num_boxes = (Nat.choose 9 4) ^ 3 := by
  sorry

#eval distribute_three_colors 4 6

end NUMINAMATH_CALUDE_ball_distribution_theorem_l1374_137424


namespace NUMINAMATH_CALUDE_stone_game_termination_and_uniqueness_l1374_137422

/-- Represents the state of stones on the infinite strip --/
def StoneConfiguration := Int → ℕ

/-- Represents a move on the strip --/
inductive Move
  | typeA (n : Int) : Move
  | typeB (n : Int) : Move

/-- Applies a move to a configuration --/
def applyMove (config : StoneConfiguration) (move : Move) : StoneConfiguration :=
  match move with
  | Move.typeA n => fun i =>
      if i = n - 1 || i = n then config i - 1
      else if i = n + 1 then config i + 1
      else config i
  | Move.typeB n => fun i =>
      if i = n then config i - 2
      else if i = n + 1 || i = n - 2 then config i + 1
      else config i

/-- Checks if a move is valid for a given configuration --/
def isValidMove (config : StoneConfiguration) (move : Move) : Prop :=
  match move with
  | Move.typeA n => config (n - 1) > 0 ∧ config n > 0
  | Move.typeB n => config n ≥ 2

/-- Checks if any move is possible for a given configuration --/
def canMove (config : StoneConfiguration) : Prop :=
  ∃ (move : Move), isValidMove config move

/-- The theorem to be proved --/
theorem stone_game_termination_and_uniqueness 
  (initial : StoneConfiguration) : 
  ∃! (final : StoneConfiguration), 
    (∃ (moves : List Move), (moves.foldl applyMove initial = final)) ∧ 
    ¬(canMove final) := by
  sorry

end NUMINAMATH_CALUDE_stone_game_termination_and_uniqueness_l1374_137422


namespace NUMINAMATH_CALUDE_circle_center_l1374_137416

/-- Given a circle with equation (x-2)^2 + (y+1)^2 = 3, prove that its center is at (2, -1) -/
theorem circle_center (x y : ℝ) : (x - 2)^2 + (y + 1)^2 = 3 → (2, -1) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l1374_137416


namespace NUMINAMATH_CALUDE_candy_packaging_remainder_l1374_137473

theorem candy_packaging_remainder : 38759863 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_candy_packaging_remainder_l1374_137473


namespace NUMINAMATH_CALUDE_line_intersects_y_axis_l1374_137404

/-- A line in 2D space defined by two points -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- The y-axis, represented as a vertical line with x-coordinate 0 -/
def yAxis : Line := { point1 := (0, 0), point2 := (0, 1) }

/-- Function to determine if a point lies on a given line -/
def pointOnLine (l : Line) (p : ℝ × ℝ) : Prop :=
  let (x1, y1) := l.point1
  let (x2, y2) := l.point2
  let (x, y) := p
  (y - y1) * (x2 - x1) = (x - x1) * (y2 - y1)

/-- Function to determine if a point lies on the y-axis -/
def pointOnYAxis (p : ℝ × ℝ) : Prop :=
  p.1 = 0

/-- The main theorem to be proved -/
theorem line_intersects_y_axis :
  let l : Line := { point1 := (2, 3), point2 := (6, -9) }
  pointOnLine l (0, 9) ∧ pointOnYAxis (0, 9) := by sorry

end NUMINAMATH_CALUDE_line_intersects_y_axis_l1374_137404


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l1374_137475

theorem absolute_value_equation_solution :
  ∀ x : ℚ, (|x - 3| = 2*x + 4) ↔ (x = -1/3) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l1374_137475


namespace NUMINAMATH_CALUDE_stock_yield_percentage_l1374_137466

/-- Calculates the yield percentage of a stock given its dividend rate, par value, and market value. -/
def yield_percentage (dividend_rate : ℚ) (par_value : ℚ) (market_value : ℚ) : ℚ :=
  (dividend_rate * par_value) / market_value

/-- Proves that a 6% stock with a market value of $75 and an assumed par value of $100 has a yield percentage of 8%. -/
theorem stock_yield_percentage :
  let dividend_rate : ℚ := 6 / 100
  let par_value : ℚ := 100
  let market_value : ℚ := 75
  yield_percentage dividend_rate par_value market_value = 8 / 100 := by
sorry

#eval yield_percentage (6/100) 100 75

end NUMINAMATH_CALUDE_stock_yield_percentage_l1374_137466


namespace NUMINAMATH_CALUDE_like_terms_sum_l1374_137408

theorem like_terms_sum (a b : ℤ) : 
  (a + 1 = 2) ∧ (b - 2 = 3) → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_sum_l1374_137408


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1374_137480

theorem arithmetic_geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : d ≠ 0)
  (h2 : ∀ n, a (n + 1) = a n + d)  -- arithmetic sequence definition
  (h3 : (a 5)^2 = a 1 * a 17)  -- geometric sequence property
  : (a 5) / (a 1) = 3 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1374_137480


namespace NUMINAMATH_CALUDE_students_like_both_sports_l1374_137413

/-- The number of students who like basketball -/
def B : ℕ := 10

/-- The number of students who like cricket -/
def C : ℕ := 8

/-- The number of students who like either basketball or cricket or both -/
def B_union_C : ℕ := 14

/-- The number of students who like both basketball and cricket -/
def B_intersect_C : ℕ := B + C - B_union_C

theorem students_like_both_sports : B_intersect_C = 4 := by
  sorry

end NUMINAMATH_CALUDE_students_like_both_sports_l1374_137413


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1374_137474

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def geometric_sequence (b c d : ℝ) :=
  c * c = b * d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  d ≠ 0 →
  a 4 = 10 →
  arithmetic_sequence a d →
  geometric_sequence (a 3) (a 6) (a 10) →
  ∀ n, a n = n + 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1374_137474


namespace NUMINAMATH_CALUDE_quadratic_real_solutions_range_l1374_137407

theorem quadratic_real_solutions_range (m : ℝ) :
  (∃ x : ℝ, m * x^2 + 2 * x + 1 = 0) ∧ (m ≠ 0) ↔ m ≤ 1 ∧ m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_solutions_range_l1374_137407


namespace NUMINAMATH_CALUDE_boat_license_count_l1374_137401

/-- Represents the set of possible letters for a boat license -/
def BoatLicenseLetter : Finset Char := {'A', 'M'}

/-- Represents the set of possible digits for a boat license -/
def BoatLicenseDigit : Finset Nat := Finset.range 10

/-- The number of digits in a boat license -/
def BoatLicenseDigitCount : Nat := 5

/-- Calculates the total number of possible boat licenses -/
def TotalBoatLicenses : Nat :=
  BoatLicenseLetter.card * (BoatLicenseDigit.card ^ BoatLicenseDigitCount)

/-- Theorem stating that the total number of boat licenses is 200,000 -/
theorem boat_license_count :
  TotalBoatLicenses = 200000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_count_l1374_137401


namespace NUMINAMATH_CALUDE_A_intersect_B_equals_open_interval_l1374_137441

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | x - 1 > 0}

-- State the theorem
theorem A_intersect_B_equals_open_interval : A ∩ B = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_equals_open_interval_l1374_137441


namespace NUMINAMATH_CALUDE_association_properties_l1374_137465

/-- A function f is associated with a set S if for any x₂-x₁ ∈ S, f(x₂)-f(x₁) ∈ S -/
def associated (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x₁ x₂, x₂ - x₁ ∈ S → f x₂ - f x₁ ∈ S

theorem association_properties :
  let f₁ : ℝ → ℝ := λ x ↦ 2*x - 1
  let f₂ : ℝ → ℝ := λ x ↦ if x < 3 then x^2 - 2*x else x^2 - 2*x + 3
  (associated f₁ (Set.Ici 0) ∧ ¬ associated f₁ (Set.Icc 0 1)) ∧
  (associated f₂ {3} → Set.Icc (Real.sqrt 3 + 1) 5 = {x | 2 ≤ f₂ x ∧ f₂ x ≤ 3}) ∧
  (∀ f : ℝ → ℝ, (associated f {1} ∧ associated f (Set.Ici 0)) ↔ associated f (Set.Icc 1 2)) :=
by sorry

#check association_properties

end NUMINAMATH_CALUDE_association_properties_l1374_137465


namespace NUMINAMATH_CALUDE_expression_evaluation_l1374_137433

theorem expression_evaluation (a b : ℤ) (h1 : a = -1) (h2 : b = 2) :
  (2*a + b) - 2*(3*a - 2*b) = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1374_137433


namespace NUMINAMATH_CALUDE_shmacks_in_shneids_l1374_137479

-- Define the conversion rates
def shmacks_per_shick : ℚ := 5 / 2
def shicks_per_shure : ℚ := 3 / 5
def shures_per_shneid : ℚ := 2 / 9

-- Define the problem
def shneids_to_convert : ℚ := 6

-- Theorem to prove
theorem shmacks_in_shneids : 
  shneids_to_convert * shures_per_shneid * shicks_per_shure * shmacks_per_shick = 2 := by
  sorry

end NUMINAMATH_CALUDE_shmacks_in_shneids_l1374_137479


namespace NUMINAMATH_CALUDE_geometric_sequence_a11_l1374_137456

-- Define the geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = q * a n

-- Define the theorem
theorem geometric_sequence_a11
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_a2a5 : a 2 * a 5 = 20)
  (h_a1a6 : a 1 + a 6 = 9) :
  a 11 = 25 / 4 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a11_l1374_137456


namespace NUMINAMATH_CALUDE_max_min_difference_2a_minus_b_l1374_137471

theorem max_min_difference_2a_minus_b : 
  ∃ (max min : ℝ), 
    (∀ a b : ℝ, a^2 + b^2 - 2*a - 4 = 0 → 2*a - b ≤ max) ∧
    (∀ a b : ℝ, a^2 + b^2 - 2*a - 4 = 0 → 2*a - b ≥ min) ∧
    (∃ a1 b1 a2 b2 : ℝ, 
      a1^2 + b1^2 - 2*a1 - 4 = 0 ∧
      a2^2 + b2^2 - 2*a2 - 4 = 0 ∧
      2*a1 - b1 = max ∧
      2*a2 - b2 = min) ∧
    max - min = 10 :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_2a_minus_b_l1374_137471


namespace NUMINAMATH_CALUDE_weight_difference_l1374_137486

/-- The weights of individuals A, B, C, D, and E --/
structure Weights where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ

/-- The conditions of the problem --/
def WeightConditions (w : Weights) : Prop :=
  (w.A + w.B + w.C) / 3 = 84 ∧
  (w.A + w.B + w.C + w.D) / 4 = 80 ∧
  (w.B + w.C + w.D + w.E) / 4 = 79 ∧
  w.A = 77 ∧
  w.E > w.D

/-- The theorem to prove --/
theorem weight_difference (w : Weights) (h : WeightConditions w) : w.E - w.D = 5 := by
  sorry

end NUMINAMATH_CALUDE_weight_difference_l1374_137486


namespace NUMINAMATH_CALUDE_max_true_statements_l1374_137476

theorem max_true_statements (a b : ℝ) : 
  ¬(∃ (p1 p2 p3 p4 : Prop), 
    (p1 ∧ p2 ∧ p3 ∧ p4) ∧
    (p1 → a < b) ∧
    (p2 → b < 0) ∧
    (p3 → a < 0) ∧
    (p4 → 1 / a < 1 / b) ∧
    (p1 ∨ p2 ∨ p3 ∨ p4 → a^2 < b^2)) :=
by sorry

end NUMINAMATH_CALUDE_max_true_statements_l1374_137476


namespace NUMINAMATH_CALUDE_systematic_sampling_fourth_element_l1374_137412

/-- Systematic sampling function -/
def systematicSample (total : ℕ) (sampleSize : ℕ) (start : ℕ) : ℕ → ℕ :=
  fun n => start + (n - 1) * (total / sampleSize)

theorem systematic_sampling_fourth_element :
  let total := 800
  let sampleSize := 50
  let interval := total / sampleSize
  let start := 7
  (systematicSample total sampleSize start 4 = 55) ∧ 
  (49 ≤ systematicSample total sampleSize start 4) ∧ 
  (systematicSample total sampleSize start 4 ≤ 64) := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_fourth_element_l1374_137412
