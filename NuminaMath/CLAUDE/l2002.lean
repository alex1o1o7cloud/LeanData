import Mathlib

namespace NUMINAMATH_CALUDE_distance_to_origin_l2002_200262

theorem distance_to_origin (x y : ℝ) (h1 : y = 15) (h2 : x = 2 + Real.sqrt 105) :
  Real.sqrt (x^2 + y^2) = Real.sqrt (334 + 4 * Real.sqrt 105) := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l2002_200262


namespace NUMINAMATH_CALUDE_hyperbola_parametric_equation_l2002_200230

theorem hyperbola_parametric_equation :
  ∀ (x : ℝ), x ≠ 0 →
  ∃ (t : ℝ), x = Real.tan t ∧ (1 / x) = (1 / Real.tan t) ∧ x * (1 / x) = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_parametric_equation_l2002_200230


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l2002_200232

/-- Represents a participant's score in a single day of the competition -/
structure DailyScore where
  scored : ℕ
  attempted : ℕ
  success_ratio : scored ≤ attempted

/-- Represents a participant's total score across two days -/
structure TotalScore where
  day1 : DailyScore
  day2 : DailyScore
  total_attempted : day1.attempted + day2.attempted = 500

/-- Gamma's scores for the two days -/
def gamma : TotalScore := {
  day1 := { scored := 180, attempted := 280, success_ratio := by sorry },
  day2 := { scored := 120, attempted := 220, success_ratio := by sorry },
  total_attempted := by sorry
}

/-- Delta's scores for the two days -/
structure DeltaScore extends TotalScore where
  day1_ratio_less : (day1.scored : ℚ) / day1.attempted < (gamma.day1.scored : ℚ) / gamma.day1.attempted
  day2_ratio_less : (day2.scored : ℚ) / day2.attempted < (gamma.day2.scored : ℚ) / gamma.day2.attempted

theorem delta_max_success_ratio :
  ∀ delta : DeltaScore,
    (delta.day1.scored + delta.day2.scored : ℚ) / 500 ≤ 409 / 500 := by sorry

end NUMINAMATH_CALUDE_delta_max_success_ratio_l2002_200232


namespace NUMINAMATH_CALUDE_solve_equation_l2002_200258

theorem solve_equation (Q : ℝ) : (Q^3)^(1/2) = 9 * 729^(1/6) → Q = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2002_200258


namespace NUMINAMATH_CALUDE_ratio_and_mean_determine_a_l2002_200287

theorem ratio_and_mean_determine_a (a b c : ℕ+) : 
  (a : ℚ) / b = 2 / 3 →
  (a : ℚ) / c = 2 / 4 →
  (b : ℚ) / c = 3 / 4 →
  (a + b + c : ℚ) / 3 = 42 →
  a = 28 := by
sorry

end NUMINAMATH_CALUDE_ratio_and_mean_determine_a_l2002_200287


namespace NUMINAMATH_CALUDE_profit_percentage_l2002_200200

/-- If selling an article at 2/3 of price P results in a 10% loss,
    then selling it at price P results in a 35% profit. -/
theorem profit_percentage (P : ℝ) (P_pos : P > 0) : 
  (∃ C : ℝ, C > 0 ∧ (2/3 * P) = (0.9 * C)) →
  ((P - ((2/3 * P) / 0.9)) / ((2/3 * P) / 0.9)) * 100 = 35 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l2002_200200


namespace NUMINAMATH_CALUDE_difference_of_cubes_factorization_l2002_200276

theorem difference_of_cubes_factorization (t : ℝ) : t^3 - 125 = (t - 5) * (t^2 + 5*t + 25) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_cubes_factorization_l2002_200276


namespace NUMINAMATH_CALUDE_root_transformation_l2002_200249

theorem root_transformation (r₁ r₂ r₃ : ℂ) : 
  (r₁^3 - 3 * r₁^2 + 13 = 0) ∧ 
  (r₂^3 - 3 * r₂^2 + 13 = 0) ∧ 
  (r₃^3 - 3 * r₃^2 + 13 = 0) →
  ((3 * r₁)^3 - 9 * (3 * r₁)^2 + 351 = 0) ∧
  ((3 * r₂)^3 - 9 * (3 * r₂)^2 + 351 = 0) ∧
  ((3 * r₃)^3 - 9 * (3 * r₃)^2 + 351 = 0) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l2002_200249


namespace NUMINAMATH_CALUDE_min_days_for_progress_ratio_l2002_200238

theorem min_days_for_progress_ratio : ∃ n : ℕ, n = 23 ∧ 
  (∀ x : ℕ, (1.2 : ℝ)^x / (0.8 : ℝ)^x ≥ 10000 → x ≥ n) ∧
  (1.2 : ℝ)^n / (0.8 : ℝ)^n ≥ 10000 :=
by sorry

end NUMINAMATH_CALUDE_min_days_for_progress_ratio_l2002_200238


namespace NUMINAMATH_CALUDE_exists_special_function_l2002_200274

open Function Set

/-- A function f: ℝ → ℝ satisfying specific properties --/
structure SpecialFunction where
  f : ℝ → ℝ
  increasing : Monotone f
  composite_increasing : Monotone (f ∘ f)
  not_fixed_point : ∀ a : ℝ, f a ≠ a
  involutive : ∀ x : ℝ, f (f x) = x

/-- Theorem stating the existence of a function satisfying the required properties --/
theorem exists_special_function : ∃ sf : SpecialFunction, True := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l2002_200274


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2002_200286

/-- A quadratic function passing through (-3,0) and (5,0) with maximum value 76 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_minus_three : a * (-3)^2 + b * (-3) + c = 0
  passes_through_five : a * 5^2 + b * 5 + c = 0
  max_value : ∃ x, a * x^2 + b * x + c = 76
  is_max : ∀ x, a * x^2 + b * x + c ≤ 76

/-- The sum of coefficients of the quadratic function is 76 -/
theorem sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = 76 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_coefficients_l2002_200286


namespace NUMINAMATH_CALUDE_unique_solution_logarithmic_system_l2002_200239

theorem unique_solution_logarithmic_system :
  ∃! (x y : ℝ), 
    Real.log (x^2 + y^2) / Real.log 10 = 1 + Real.log 8 / Real.log 10 ∧
    Real.log (x + y) / Real.log 10 - Real.log (x - y) / Real.log 10 = Real.log 3 / Real.log 10 ∧
    x + y > 0 ∧
    x - y > 0 ∧
    x = 8 ∧
    y = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_logarithmic_system_l2002_200239


namespace NUMINAMATH_CALUDE_paul_books_left_l2002_200243

/-- The number of books Paul had left after the garage sale -/
def books_left (initial_books : ℕ) (books_sold : ℕ) : ℕ :=
  initial_books - books_sold

/-- Theorem stating that Paul had 66 books left after the garage sale -/
theorem paul_books_left : books_left 108 42 = 66 := by
  sorry

end NUMINAMATH_CALUDE_paul_books_left_l2002_200243


namespace NUMINAMATH_CALUDE_xyz_inequality_l2002_200235

theorem xyz_inequality (x y z : ℝ) 
  (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z)
  (h_sum : x + y + z = 1) : 
  0 ≤ x*y + y*z + z*x - 2*x*y*z ∧ x*y + y*z + z*x - 2*x*y*z ≤ 7/27 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequality_l2002_200235


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_is_417_l2002_200269

/-- The sum of areas of six rectangles with width 3 and lengths (2², 3², 4², 5², 6², 7²) -/
def sum_of_rectangle_areas : ℕ :=
  let width := 3
  let lengths := [2, 3, 4, 5, 6, 7].map (λ x => x^2)
  (lengths.map (λ l => width * l)).sum

/-- Theorem stating that the sum of the areas is 417 -/
theorem sum_of_rectangle_areas_is_417 : sum_of_rectangle_areas = 417 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_is_417_l2002_200269


namespace NUMINAMATH_CALUDE_total_veranda_area_l2002_200224

/-- Calculates the total area of verandas in a multi-story building. -/
theorem total_veranda_area (floors : ℕ) (room_length room_width veranda_width : ℝ) :
  floors = 4 →
  room_length = 21 →
  room_width = 12 →
  veranda_width = 2 →
  (floors * ((room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) - room_length * room_width)) = 592 :=
by sorry

end NUMINAMATH_CALUDE_total_veranda_area_l2002_200224


namespace NUMINAMATH_CALUDE_harmonic_mean_of_three_fourths_and_five_sixths_l2002_200229

theorem harmonic_mean_of_three_fourths_and_five_sixths :
  let a : ℚ := 3/4
  let b : ℚ := 5/6
  let harmonic_mean (x y : ℚ) := 2 * x * y / (x + y)
  harmonic_mean a b = 15/19 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_of_three_fourths_and_five_sixths_l2002_200229


namespace NUMINAMATH_CALUDE_calculation_proof_l2002_200253

theorem calculation_proof : 
  (5 : ℚ) / 19 * ((3 + 4 / 5) * (5 + 1 / 3) + (4 + 2 / 3) * (19 / 5)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2002_200253


namespace NUMINAMATH_CALUDE_find_k_value_l2002_200211

theorem find_k_value (x y z k : ℝ) 
  (eq1 : 9 / (x + y) = k / (x + 2*z))
  (eq2 : k / (x + 2*z) = 14 / (z - y))
  (cond1 : y = 2*x)
  (cond2 : x + z = 10) :
  k = 46 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l2002_200211


namespace NUMINAMATH_CALUDE_fifth_number_ninth_row_is_61_l2002_200254

/-- Represents a lattice pattern with a given number of columns per row -/
structure LatticePattern where
  columnsPerRow : ℕ

/-- Calculates the last number in a given row of the lattice pattern -/
def lastNumberInRow (pattern : LatticePattern) (row : ℕ) : ℕ :=
  pattern.columnsPerRow * row

/-- Calculates the nth number from the end in a given row -/
def nthNumberFromEnd (pattern : LatticePattern) (row : ℕ) (n : ℕ) : ℕ :=
  lastNumberInRow pattern row - (n - 1)

/-- The theorem to be proved -/
theorem fifth_number_ninth_row_is_61 :
  let pattern : LatticePattern := ⟨7⟩
  nthNumberFromEnd pattern 9 5 = 61 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_ninth_row_is_61_l2002_200254


namespace NUMINAMATH_CALUDE_shirts_per_minute_l2002_200242

/-- A machine that makes shirts -/
structure ShirtMachine where
  yesterday_production : ℕ
  today_production : ℕ
  total_working_time : ℕ

/-- Theorem: The machine can make 8 shirts per minute -/
theorem shirts_per_minute (m : ShirtMachine)
  (h1 : m.yesterday_production = 13)
  (h2 : m.today_production = 3)
  (h3 : m.total_working_time = 2) :
  (m.yesterday_production + m.today_production) / m.total_working_time = 8 := by
  sorry

end NUMINAMATH_CALUDE_shirts_per_minute_l2002_200242


namespace NUMINAMATH_CALUDE_cookie_store_spending_l2002_200275

theorem cookie_store_spending (B D : ℝ) 
  (h1 : D = 0.7 * B)  -- David spent 30 cents less for each dollar Ben spent
  (h2 : B = D + 15)   -- Ben paid $15 more than David
  : B + D = 85 := by
  sorry

end NUMINAMATH_CALUDE_cookie_store_spending_l2002_200275


namespace NUMINAMATH_CALUDE_problem_statement_l2002_200240

theorem problem_statement (a b c A B C : ℝ) 
  (eq1 : a + b + c = 0)
  (eq2 : A + B + C = 0)
  (eq3 : a / A + b / B + c / C = 0)
  (hA : A ≠ 0)
  (hB : B ≠ 0)
  (hC : C ≠ 0) :
  a * A^2 + b * B^2 + c * C^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2002_200240


namespace NUMINAMATH_CALUDE_budget_allocation_l2002_200278

theorem budget_allocation (research_dev : ℝ) (utilities : ℝ) (equipment : ℝ) (supplies : ℝ) 
  (transportation_degrees : ℝ) (total_degrees : ℝ) :
  research_dev = 9 →
  utilities = 5 →
  equipment = 4 →
  supplies = 2 →
  transportation_degrees = 72 →
  total_degrees = 360 →
  let transportation := (transportation_degrees / total_degrees) * 100
  let other_categories := research_dev + utilities + equipment + supplies + transportation
  let salaries := 100 - other_categories
  salaries = 60 := by
sorry

end NUMINAMATH_CALUDE_budget_allocation_l2002_200278


namespace NUMINAMATH_CALUDE_age_difference_l2002_200220

/-- Given that the total age of A and B is 16 years more than the total age of B and C,
    prove that C is 16 years younger than A. -/
theorem age_difference (A B C : ℕ) (h : A + B = B + C + 16) : A = C + 16 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l2002_200220


namespace NUMINAMATH_CALUDE_vector_collinearity_l2002_200204

/-- Two vectors in ℝ² -/
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (-1, 2)

/-- Collinearity of two vectors in ℝ² -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

/-- The main theorem -/
theorem vector_collinearity (m : ℝ) :
  collinear ((m * a.1 + 4 * b.1, m * a.2 + 4 * b.2)) (a.1 - 2 * b.1, a.2 - 2 * b.2) →
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2002_200204


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l2002_200296

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l2002_200296


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l2002_200273

/-- Given two rectangles A and B with sides proportional to a constant k, 
    prove that their area ratio is 4:1 -/
theorem rectangle_area_ratio 
  (k : ℝ) 
  (a b c d : ℝ) 
  (h_pos : k > 0) 
  (h_ka : a = k * a) 
  (h_kb : b = k * b) 
  (h_kc : c = k * c) 
  (h_kd : d = k * d) 
  (h_ratio : a / c = b / d) 
  (h_val : a / c = 2 / 5) : 
  (a * b) / (c * d) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l2002_200273


namespace NUMINAMATH_CALUDE_systematic_sampling_theorem_l2002_200236

/-- Represents a systematic sampling of students -/
structure SystematicSampling where
  totalStudents : ℕ
  sampleSize : ℕ
  startNumber : ℕ
  hTotalPositive : 0 < totalStudents
  hSamplePositive : 0 < sampleSize
  hStartValid : startNumber ≤ totalStudents

/-- Generates the sequence of selected student numbers -/
def generateSequence (s : SystematicSampling) : List ℕ :=
  List.range s.sampleSize |>.map (fun i => s.startNumber + i * (s.totalStudents / s.sampleSize))

/-- Theorem stating that the systematic sampling generates the expected sequence -/
theorem systematic_sampling_theorem (s : SystematicSampling)
  (h1 : s.totalStudents = 50)
  (h2 : s.sampleSize = 5)
  (h3 : s.startNumber = 3) :
  generateSequence s = [3, 13, 23, 33, 43] := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_theorem_l2002_200236


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2002_200218

/-- The eccentricity of a hyperbola with specific properties -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≥ 1) 
  (h4 : ∃ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ∧ 
    (Real.sqrt ((x + c)^2 + y^2) - Real.sqrt ((x - c)^2 + y^2))^2 = b^2 - 3*a*b) 
  (h5 : c^2 = a^2 + b^2) : 
  Real.sqrt (a^2 + b^2) / a = Real.sqrt 17 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2002_200218


namespace NUMINAMATH_CALUDE_water_polo_team_selection_l2002_200295

/-- The number of ways to select a starting team in a water polo club. -/
theorem water_polo_team_selection (total_members : Nat) (team_size : Nat) (h1 : total_members = 20) (h2 : team_size = 9) :
  (total_members * Nat.choose (total_members - 1) (team_size - 1) * (team_size - 1)) = 12093120 := by
  sorry

end NUMINAMATH_CALUDE_water_polo_team_selection_l2002_200295


namespace NUMINAMATH_CALUDE_number_of_kids_l2002_200284

theorem number_of_kids (total_money : ℕ) (apple_cost : ℕ) (apples_per_kid : ℕ) : 
  total_money = 360 → 
  apple_cost = 4 → 
  apples_per_kid = 5 → 
  (total_money / apple_cost) / apples_per_kid = 18 := by
  sorry

end NUMINAMATH_CALUDE_number_of_kids_l2002_200284


namespace NUMINAMATH_CALUDE_power_sum_theorem_l2002_200206

theorem power_sum_theorem (m n : ℕ) (h1 : 2^m = 3) (h2 : 2^n = 2) : 2^(m+2*n) = 12 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l2002_200206


namespace NUMINAMATH_CALUDE_expression_value_l2002_200207

theorem expression_value :
  let x : ℤ := 2
  let y : ℤ := -3
  let z : ℤ := 1
  x^2 + y^2 - z^2 + 2*x*y + 10 = 10 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2002_200207


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2002_200227

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 7 * p - 9 = 0) →
  (3 * q^3 - 4 * q^2 + 7 * q - 9 = 0) →
  (3 * r^3 - 4 * r^2 + 7 * r - 9 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l2002_200227


namespace NUMINAMATH_CALUDE_family_children_count_l2002_200265

theorem family_children_count :
  ∀ (num_children : ℕ),
    (5 * (num_children + 3) + 2 * num_children + 4 * 3 = 55) →
    num_children = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_family_children_count_l2002_200265


namespace NUMINAMATH_CALUDE_sequence_sum_lower_bound_l2002_200212

theorem sequence_sum_lower_bound (n : ℕ) (a : ℕ → ℝ) 
  (h1 : a 1 = 0)
  (h2 : ∀ i ∈ Finset.range n, i ≥ 2 → |a i| = |a (i-1) + 1|) :
  (Finset.range n).sum a ≥ -n / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_lower_bound_l2002_200212


namespace NUMINAMATH_CALUDE_rectangle_measurement_error_l2002_200203

theorem rectangle_measurement_error (L W : ℝ) (x : ℝ) (h_positive : L > 0 ∧ W > 0) :
  (1.10 * L) * (W * (1 - x / 100)) = L * W * 1.045 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_measurement_error_l2002_200203


namespace NUMINAMATH_CALUDE_sarah_apples_to_teachers_l2002_200217

def apples_given_to_teachers (initial_apples : ℕ) (locker_apples : ℕ) (friend_apples : ℕ) 
  (classmate_apples : ℕ) (traded_apples : ℕ) (close_friends : ℕ) (eaten_apples : ℕ) 
  (final_apples : ℕ) : ℕ :=
  initial_apples - locker_apples - friend_apples - classmate_apples - traded_apples - 
  close_friends - eaten_apples - final_apples

theorem sarah_apples_to_teachers :
  apples_given_to_teachers 50 10 3 8 4 5 1 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_sarah_apples_to_teachers_l2002_200217


namespace NUMINAMATH_CALUDE_parallel_vectors_l2002_200261

def a (m : ℝ) : Fin 2 → ℝ := ![2, m]
def b : Fin 2 → ℝ := ![1, -2]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, u i = k * v i)

theorem parallel_vectors (m : ℝ) :
  parallel (a m) (λ i => a m i + 2 * b i) → m = -4 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l2002_200261


namespace NUMINAMATH_CALUDE_fred_balloons_l2002_200205

theorem fred_balloons (initial : ℕ) (to_sandy : ℕ) (to_bob : ℕ) :
  initial = 709 →
  to_sandy = 221 →
  to_bob = 153 →
  initial - to_sandy - to_bob = 335 := by
  sorry

end NUMINAMATH_CALUDE_fred_balloons_l2002_200205


namespace NUMINAMATH_CALUDE_rational_numbers_equivalence_l2002_200222

-- Define the set of integers
def Integers : Set ℚ := {q : ℚ | ∃ (n : ℤ), q = n}

-- Define the set of fractions
def Fractions : Set ℚ := {q : ℚ | ∃ (a b : ℤ), b ≠ 0 ∧ q = a / b}

-- Theorem statement
theorem rational_numbers_equivalence :
  Set.univ = Integers ∪ Fractions :=
sorry

end NUMINAMATH_CALUDE_rational_numbers_equivalence_l2002_200222


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2002_200248

theorem complex_number_in_first_quadrant : 
  let z : ℂ := 1 - (1 / Complex.I)
  (z.re > 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l2002_200248


namespace NUMINAMATH_CALUDE_no_solution_for_sock_problem_l2002_200299

theorem no_solution_for_sock_problem :
  ¬∃ (n m : ℕ), n + m = 2009 ∧ 
  (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1) : ℚ) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_no_solution_for_sock_problem_l2002_200299


namespace NUMINAMATH_CALUDE_b_lending_rate_to_c_l2002_200290

/-- Given the lending scenario between A, B, and C, prove that B's lending rate to C is 12.5% --/
theorem b_lending_rate_to_c (principal : ℝ) (a_rate : ℝ) (b_gain : ℝ) (time : ℝ) :
  principal = 3150 →
  a_rate = 8 →
  b_gain = 283.5 →
  time = 2 →
  ∃ (b_rate : ℝ),
    b_rate = 12.5 ∧
    b_gain = (principal * b_rate / 100 * time) - (principal * a_rate / 100 * time) :=
by sorry

end NUMINAMATH_CALUDE_b_lending_rate_to_c_l2002_200290


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_21_over_10_l2002_200291

theorem sqrt_difference_equals_21_over_10 :
  Real.sqrt (25 / 4) - Real.sqrt (4 / 25) = 21 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_21_over_10_l2002_200291


namespace NUMINAMATH_CALUDE_tshirt_cost_l2002_200221

/-- The Razorback t-shirt Shop problem -/
theorem tshirt_cost (total_sales : ℝ) (num_shirts : ℕ) (cost_per_shirt : ℝ)
  (h1 : total_sales = 720)
  (h2 : num_shirts = 45)
  (h3 : cost_per_shirt = total_sales / num_shirts) :
  cost_per_shirt = 16 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_l2002_200221


namespace NUMINAMATH_CALUDE_alice_weekly_distance_l2002_200282

/-- The distance Alice walks to school each day -/
def distance_to_school : ℕ := 10

/-- The distance Alice walks back home each day -/
def distance_from_school : ℕ := 12

/-- The number of days Alice walks to and from school in a week -/
def days_per_week : ℕ := 5

/-- Theorem: Alice's total walking distance for the week is 110 miles -/
theorem alice_weekly_distance :
  (distance_to_school + distance_from_school) * days_per_week = 110 := by
  sorry

end NUMINAMATH_CALUDE_alice_weekly_distance_l2002_200282


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l2002_200259

/-- The area of a rectangular plot with length thrice its breadth and breadth of 11 meters is 363 square meters. -/
theorem rectangular_plot_area (breadth : ℝ) (length : ℝ) (area : ℝ) : 
  breadth = 11 →
  length = 3 * breadth →
  area = length * breadth →
  area = 363 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l2002_200259


namespace NUMINAMATH_CALUDE_mathematician_daily_questions_l2002_200219

theorem mathematician_daily_questions 
  (project1_questions : ℕ) 
  (project2_questions : ℕ) 
  (days_in_week : ℕ) 
  (h1 : project1_questions = 518) 
  (h2 : project2_questions = 476) 
  (h3 : days_in_week = 7) :
  (project1_questions + project2_questions) / days_in_week = 142 :=
by sorry

end NUMINAMATH_CALUDE_mathematician_daily_questions_l2002_200219


namespace NUMINAMATH_CALUDE_increase_by_percentage_l2002_200247

/-- Theorem: Increasing 350 by 50% results in 525. -/
theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 350 → percentage = 50 → result = initial * (1 + percentage / 100) → result = 525 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l2002_200247


namespace NUMINAMATH_CALUDE_star_equality_implies_x_equals_four_l2002_200231

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b - d)

-- Theorem statement
theorem star_equality_implies_x_equals_four :
  ∀ x y : ℤ, star 5 5 2 1 = star x y 1 4 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_star_equality_implies_x_equals_four_l2002_200231


namespace NUMINAMATH_CALUDE_remaining_quadrilateral_perimeter_l2002_200208

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a quadrilateral with side lengths a, b, c, and d -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The perimeter of a quadrilateral -/
def Quadrilateral.perimeter (q : Quadrilateral) : ℝ :=
  q.a + q.b + q.c + q.d

/-- Given an equilateral triangle ABC with side length 5 and a triangular section DBE cut from it
    with DB = EB = 2, the perimeter of the remaining quadrilateral ACED is 13 -/
theorem remaining_quadrilateral_perimeter :
  ∀ (abc : Triangle) (dbe : Triangle) (aced : Quadrilateral),
    abc.a = 5 ∧ abc.b = 5 ∧ abc.c = 5 →  -- ABC is equilateral with side length 5
    dbe.a = 2 ∧ dbe.b = 2 ∧ dbe.c = 2 →  -- DBE is equilateral with side length 2
    aced.a = 5 ∧                         -- AC remains untouched
    aced.b = abc.b - dbe.b ∧             -- CE = AB - DB
    aced.c = dbe.c ∧                     -- ED is a side of DBE
    aced.d = abc.c - dbe.c →             -- DA = BC - BE
    aced.perimeter = 13 :=
by sorry

end NUMINAMATH_CALUDE_remaining_quadrilateral_perimeter_l2002_200208


namespace NUMINAMATH_CALUDE_min_dot_product_plane_vectors_l2002_200210

theorem min_dot_product_plane_vectors (a b : ℝ × ℝ) :
  ‖(2 : ℝ) • a - b‖ ≤ 3 →
  (∀ a' b' : ℝ × ℝ, ‖(2 : ℝ) • a' - b'‖ ≤ 3 → a' • b' ≥ a • b) →
  a • b = -(9 / 8) := by
  sorry

end NUMINAMATH_CALUDE_min_dot_product_plane_vectors_l2002_200210


namespace NUMINAMATH_CALUDE_complex_equation_solutions_l2002_200245

theorem complex_equation_solutions (c p q r s : ℂ) : 
  (p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) →
  (∀ z : ℂ, (z - p) * (z - q) * (z - r) * (z - s) = 
             (z - c*p) * (z - c*q) * (z - c*r) * (z - c*s)) →
  (∃ (solutions : Finset ℂ), solutions.card = 4 ∧ c ∈ solutions ∧
    ∀ x ∈ solutions, x^4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solutions_l2002_200245


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l2002_200256

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type -/
structure Line where
  passes_through : ℝ × ℝ → Prop

/-- Intersection point of a line and a parabola -/
structure IntersectionPoint where
  point : ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_intersection_distance 
  (p : Parabola)
  (l : Line)
  (A B : IntersectionPoint)
  (h1 : p.equation = fun x y => y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : l.passes_through p.focus)
  (h4 : l.passes_through A.point ∧ l.passes_through B.point)
  (h5 : distance A.point p.focus = 4)
  : distance B.point p.focus = 4/3 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l2002_200256


namespace NUMINAMATH_CALUDE_max_xy_on_line_segment_l2002_200270

/-- The maximum value of xy for a point P(x,y) on the line segment between A(3,0) and B(0,4) is 3 -/
theorem max_xy_on_line_segment : 
  let A : ℝ × ℝ := (3, 0)
  let B : ℝ × ℝ := (0, 4)
  ∃ (M : ℝ), M = 3 ∧ 
    ∀ (P : ℝ × ℝ), (∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • A + t • B) → 
      P.1 * P.2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_xy_on_line_segment_l2002_200270


namespace NUMINAMATH_CALUDE_unique_square_cube_factor_of_1800_l2002_200214

/-- A number is a perfect square if it can be expressed as the product of an integer with itself. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

/-- A number is a perfect cube if it can be expressed as the product of an integer with itself three times. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k * k

/-- A number is a factor of another number if it divides the latter without a remainder. -/
def IsFactor (a n : ℕ) : Prop :=
  n % a = 0

theorem unique_square_cube_factor_of_1800 :
  ∃! x : ℕ, x > 0 ∧ IsFactor x 1800 ∧ IsPerfectSquare x ∧ IsPerfectCube x :=
sorry

end NUMINAMATH_CALUDE_unique_square_cube_factor_of_1800_l2002_200214


namespace NUMINAMATH_CALUDE_cloth_sale_theorem_l2002_200234

/-- Represents the sale of cloth with given parameters. -/
structure ClothSale where
  totalSellingPrice : ℕ
  lossPerMetre : ℕ
  costPricePerMetre : ℕ

/-- Calculates the number of metres of cloth sold. -/
def metresSold (sale : ClothSale) : ℕ :=
  sale.totalSellingPrice / (sale.costPricePerMetre - sale.lossPerMetre)

/-- Theorem stating that for the given parameters, 600 metres of cloth were sold. -/
theorem cloth_sale_theorem (sale : ClothSale) 
  (h1 : sale.totalSellingPrice = 18000)
  (h2 : sale.lossPerMetre = 5)
  (h3 : sale.costPricePerMetre = 35) :
  metresSold sale = 600 := by
  sorry

#eval metresSold { totalSellingPrice := 18000, lossPerMetre := 5, costPricePerMetre := 35 }

end NUMINAMATH_CALUDE_cloth_sale_theorem_l2002_200234


namespace NUMINAMATH_CALUDE_triangle_third_side_length_l2002_200226

theorem triangle_third_side_length (a b : ℝ) (θ : ℝ) (h1 : a = 9) (h2 : b = 10) (h3 : θ = 135 * π / 180) :
  ∃ c : ℝ, c^2 = a^2 + b^2 - 2 * a * b * Real.cos θ ∧ c = Real.sqrt (181 + 90 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_third_side_length_l2002_200226


namespace NUMINAMATH_CALUDE_combined_work_time_l2002_200288

def team_A_time : ℝ := 15
def team_B_time : ℝ := 30

theorem combined_work_time :
  1 / (1 / team_A_time + 1 / team_B_time) = 10 := by
  sorry

end NUMINAMATH_CALUDE_combined_work_time_l2002_200288


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2002_200283

theorem complex_equation_solution (z : ℂ) (h : (2 - Complex.I) * z = 5) : z = 2 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2002_200283


namespace NUMINAMATH_CALUDE_star_value_l2002_200268

-- Define the sequence type
def Sequence := Fin 12 → ℕ

-- Define the property that the sum of any four adjacent numbers is 11
def SumProperty (s : Sequence) : Prop :=
  ∀ i : Fin 9, s i + s (i + 1) + s (i + 2) + s (i + 3) = 11

-- Define the repeating pattern property
def PatternProperty (s : Sequence) : Prop :=
  ∀ i : Fin 3, 
    s (4 * i) = 2 ∧ 
    s (4 * i + 1) = 0 ∧ 
    s (4 * i + 2) = 1

-- Main theorem
theorem star_value (s : Sequence) 
  (h1 : SumProperty s) 
  (h2 : PatternProperty s) : 
  ∀ i : Fin 3, s (4 * i + 3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_star_value_l2002_200268


namespace NUMINAMATH_CALUDE_calculate_expression_l2002_200271

theorem calculate_expression : 15 * 25 + 35 * 15 + 16 * 28 + 32 * 16 = 1860 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2002_200271


namespace NUMINAMATH_CALUDE_beth_and_jan_total_money_l2002_200280

def beth_money : ℕ := 70
def jan_money : ℕ := 80

theorem beth_and_jan_total_money :
  (beth_money + 35 = 105) ∧
  (jan_money - 10 = beth_money) →
  beth_money + jan_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_beth_and_jan_total_money_l2002_200280


namespace NUMINAMATH_CALUDE_set_relations_theorem_l2002_200285

universe u

theorem set_relations_theorem (U : Type u) (A B : Set U) : 
  (A ∩ B = ∅ → (Set.compl A ∪ Set.compl B) = Set.univ) ∧
  (A ∪ B = Set.univ → (Set.compl A ∩ Set.compl B) = ∅) ∧
  (A ∪ B = ∅ → A = ∅ ∧ B = ∅) := by
  sorry

end NUMINAMATH_CALUDE_set_relations_theorem_l2002_200285


namespace NUMINAMATH_CALUDE_fraction_sum_difference_l2002_200225

theorem fraction_sum_difference : 1/2 + 3/4 - 5/8 = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_difference_l2002_200225


namespace NUMINAMATH_CALUDE_manuscript_cost_calculation_l2002_200257

def manuscript_typing_cost (first_typing_rate : ℕ) (revision_rate : ℕ) 
  (total_pages : ℕ) (pages_revised_once : ℕ) (pages_revised_twice : ℕ) : ℕ :=
  let pages_not_revised := total_pages - pages_revised_once - pages_revised_twice
  let first_typing_cost := total_pages * first_typing_rate
  let revision_cost_once := pages_revised_once * revision_rate
  let revision_cost_twice := pages_revised_twice * (2 * revision_rate)
  first_typing_cost + revision_cost_once + revision_cost_twice

theorem manuscript_cost_calculation :
  manuscript_typing_cost 5 4 100 30 20 = 780 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_cost_calculation_l2002_200257


namespace NUMINAMATH_CALUDE_unique_max_divisor_number_l2002_200263

/-- A positive integer N satisfies the special divisor property if all of its divisors
    can be written as p-2 for some prime number p -/
def has_special_divisor_property (N : ℕ+) : Prop :=
  ∀ d : ℕ, d ∣ N.val → ∃ p : ℕ, Nat.Prime p ∧ d = p - 2

/-- The maximum number of divisors for any N satisfying the special divisor property -/
def max_divisors : ℕ := 8

/-- The theorem stating that 135 is the only number with the maximum number of divisors
    satisfying the special divisor property -/
theorem unique_max_divisor_number :
  ∃! N : ℕ+, has_special_divisor_property N ∧
  (Nat.card {d : ℕ | d ∣ N.val} = max_divisors) ∧
  N.val = 135 := by sorry

#check unique_max_divisor_number

end NUMINAMATH_CALUDE_unique_max_divisor_number_l2002_200263


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2002_200298

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digits_nonzero (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def product_divisible_by_1000 (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  (n * (10 * a + b) * a) % 1000 = 0

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_three_digit n ∧
             digits_nonzero n ∧
             product_divisible_by_1000 n ∧
             n = 875 := by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2002_200298


namespace NUMINAMATH_CALUDE_expected_sixes_two_dice_l2002_200223

/-- The probability of rolling a 6 on a standard die -/
def prob_six : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 2

/-- The expected number of 6's when rolling two standard dice -/
def expected_sixes : ℚ := 1 / 3

/-- Theorem: The expected number of 6's when rolling two standard dice is 1/3 -/
theorem expected_sixes_two_dice : 
  expected_sixes = num_dice * prob_six := by sorry

end NUMINAMATH_CALUDE_expected_sixes_two_dice_l2002_200223


namespace NUMINAMATH_CALUDE_probability_rain_all_three_days_l2002_200267

theorem probability_rain_all_three_days 
  (prob_friday : ℝ) 
  (prob_saturday : ℝ) 
  (prob_sunday : ℝ) 
  (h1 : prob_friday = 0.4)
  (h2 : prob_saturday = 0.5)
  (h3 : prob_sunday = 0.3)
  (h4 : 0 ≤ prob_friday ∧ prob_friday ≤ 1)
  (h5 : 0 ≤ prob_saturday ∧ prob_saturday ≤ 1)
  (h6 : 0 ≤ prob_sunday ∧ prob_sunday ≤ 1) :
  prob_friday * prob_saturday * prob_sunday = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_probability_rain_all_three_days_l2002_200267


namespace NUMINAMATH_CALUDE_white_sox_wins_l2002_200246

theorem white_sox_wins (total_games : ℕ) (games_lost : ℕ) (win_difference : ℕ) : 
  total_games = 162 →
  games_lost = 63 →
  win_difference = 36 →
  total_games = games_lost + (games_lost + win_difference) →
  games_lost + win_difference = 99 := by
sorry

end NUMINAMATH_CALUDE_white_sox_wins_l2002_200246


namespace NUMINAMATH_CALUDE_divisibility_condition_l2002_200213

theorem divisibility_condition (n : ℕ) : 
  (∃ m : ℕ, (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → k ∣ m) ∧ 
            ¬((n + 1) ∣ m) ∧ ¬((n + 2) ∣ m) ∧ ¬((n + 3) ∣ m)) ↔ 
  n = 1 ∨ n = 2 ∨ n = 6 :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2002_200213


namespace NUMINAMATH_CALUDE_james_coat_cost_l2002_200241

def total_cost : ℕ := 110
def shoe_cost : ℕ := 30
def jeans_cost : ℕ := 20

def james_items : ℕ := 3  -- 1 coat and 2 pairs of jeans
def jamie_items : ℕ := 1  -- 1 pair of shoes

theorem james_coat_cost : 
  total_cost - (shoe_cost + 2 * jeans_cost) = 40 := by
  sorry

end NUMINAMATH_CALUDE_james_coat_cost_l2002_200241


namespace NUMINAMATH_CALUDE_sqrt_29_between_consecutive_integers_product_l2002_200209

theorem sqrt_29_between_consecutive_integers_product (a b : ℤ) : 
  (a : ℝ) < Real.sqrt 29 ∧ Real.sqrt 29 < (b : ℝ) ∧ b = a + 1 → a * b = 30 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_29_between_consecutive_integers_product_l2002_200209


namespace NUMINAMATH_CALUDE_treasure_chest_rubies_l2002_200281

theorem treasure_chest_rubies (total_gems diamonds : ℕ) 
  (h1 : total_gems = 5155)
  (h2 : diamonds = 45) :
  total_gems - diamonds = 5110 := by
  sorry

end NUMINAMATH_CALUDE_treasure_chest_rubies_l2002_200281


namespace NUMINAMATH_CALUDE_remaining_ripe_mangoes_l2002_200216

theorem remaining_ripe_mangoes (total_mangoes : ℕ) (ripe_fraction : ℚ) (eaten_fraction : ℚ) : 
  total_mangoes = 400 →
  ripe_fraction = 3/5 →
  eaten_fraction = 3/5 →
  (total_mangoes : ℚ) * ripe_fraction * (1 - eaten_fraction) = 96 := by
  sorry

end NUMINAMATH_CALUDE_remaining_ripe_mangoes_l2002_200216


namespace NUMINAMATH_CALUDE_rachel_essay_time_l2002_200252

/-- Rachel's essay writing process -/
def essay_time (pages_per_30min : ℕ) (research_time : ℕ) (total_pages : ℕ) (editing_time : ℕ) : ℕ :=
  let writing_time := 30 * total_pages / pages_per_30min
  (research_time + writing_time + editing_time) / 60

/-- Theorem: Rachel spends 5 hours completing her essay -/
theorem rachel_essay_time :
  essay_time 1 45 6 75 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rachel_essay_time_l2002_200252


namespace NUMINAMATH_CALUDE_lcm_hcf_product_l2002_200244

theorem lcm_hcf_product (a b : ℕ+) (h1 : Nat.lcm a b = 72) (h2 : a * b = 432) :
  Nat.gcd a b = 6 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_product_l2002_200244


namespace NUMINAMATH_CALUDE_amusement_park_tickets_l2002_200297

theorem amusement_park_tickets (adult_price child_price total_paid total_tickets : ℕ)
  (h1 : adult_price = 8)
  (h2 : child_price = 5)
  (h3 : total_paid = 201)
  (h4 : total_tickets = 33)
  : ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_paid ∧
    child_tickets = 21 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_tickets_l2002_200297


namespace NUMINAMATH_CALUDE_calories_in_one_bar_l2002_200201

/-- The number of calories in 3 candy bars -/
def total_calories : ℕ := 24

/-- The number of candy bars -/
def num_bars : ℕ := 3

/-- The number of calories in one candy bar -/
def calories_per_bar : ℕ := total_calories / num_bars

theorem calories_in_one_bar : calories_per_bar = 8 := by
  sorry

end NUMINAMATH_CALUDE_calories_in_one_bar_l2002_200201


namespace NUMINAMATH_CALUDE_parallelogram_area_l2002_200250

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : θ = π/4) :
  a * b * Real.sin θ = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2002_200250


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2002_200237

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_3 : a 3 = -3)
  (h_4 : a 4 = 6) :
  a 6 = 24 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2002_200237


namespace NUMINAMATH_CALUDE_mcgregor_books_finished_l2002_200293

theorem mcgregor_books_finished 
  (total_books : ℕ) 
  (floyd_finished : ℕ) 
  (books_left : ℕ) 
  (h1 : total_books = 89) 
  (h2 : floyd_finished = 32) 
  (h3 : books_left = 23) : 
  total_books - floyd_finished - books_left = 34 := by
sorry

end NUMINAMATH_CALUDE_mcgregor_books_finished_l2002_200293


namespace NUMINAMATH_CALUDE_min_value_theorem_l2002_200272

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∀ z : ℝ, z = x^2 / (x + 2) + y^2 / (y + 1) → z ≥ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2002_200272


namespace NUMINAMATH_CALUDE_banana_production_ratio_l2002_200289

/-- The ratio of banana production between Jakies Island and a nearby island -/
theorem banana_production_ratio :
  ∀ (jakies_multiple : ℕ) (nearby_production : ℕ) (total_production : ℕ),
  nearby_production = 9000 →
  total_production = 99000 →
  total_production = nearby_production + jakies_multiple * nearby_production →
  (jakies_multiple * nearby_production) / nearby_production = 10 :=
by
  sorry

#check banana_production_ratio

end NUMINAMATH_CALUDE_banana_production_ratio_l2002_200289


namespace NUMINAMATH_CALUDE_sum_from_difference_and_squares_l2002_200228

theorem sum_from_difference_and_squares (m n : ℤ) 
  (h1 : m^2 - n^2 = 18) 
  (h2 : m - n = 9) : 
  m + n = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_from_difference_and_squares_l2002_200228


namespace NUMINAMATH_CALUDE_max_area_rectangle_142_perimeter_l2002_200277

/-- Represents a rectangle with integer side lengths. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle. -/
def Rectangle.perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle. -/
def Rectangle.area (r : Rectangle) : ℕ := r.length * r.width

/-- The theorem stating the maximum area of a rectangle with a perimeter of 142 feet. -/
theorem max_area_rectangle_142_perimeter :
  ∃ (r : Rectangle), r.perimeter = 142 ∧
    ∀ (s : Rectangle), s.perimeter = 142 → s.area ≤ r.area ∧
    r.area = 1260 := by
  sorry


end NUMINAMATH_CALUDE_max_area_rectangle_142_perimeter_l2002_200277


namespace NUMINAMATH_CALUDE_final_amount_is_correct_l2002_200233

-- Define the quantities and prices of fruits
def grapes_quantity : ℝ := 15
def grapes_price : ℝ := 98
def mangoes_quantity : ℝ := 8
def mangoes_price : ℝ := 120
def pineapples_quantity : ℝ := 5
def pineapples_price : ℝ := 75
def oranges_quantity : ℝ := 10
def oranges_price : ℝ := 60

-- Define the discount and tax rates
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.08

-- Define the function to calculate the final amount
def calculate_final_amount : ℝ :=
  let total_cost := grapes_quantity * grapes_price + 
                    mangoes_quantity * mangoes_price + 
                    pineapples_quantity * pineapples_price + 
                    oranges_quantity * oranges_price
  let discounted_total := total_cost * (1 - discount_rate)
  let final_amount := discounted_total * (1 + tax_rate)
  final_amount

-- Theorem statement
theorem final_amount_is_correct : 
  calculate_final_amount = 3309.66 := by sorry

end NUMINAMATH_CALUDE_final_amount_is_correct_l2002_200233


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2002_200215

/-- A quadratic function with roots at 2 and -4, and a minimum value of 32 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  root1 : a * 2^2 + b * 2 + c = 0
  root2 : a * (-4)^2 + b * (-4) + c = 0
  min_value : ∀ x, a * x^2 + b * x + c ≥ 32

theorem sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = 160 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2002_200215


namespace NUMINAMATH_CALUDE_conference_tables_theorem_l2002_200264

/-- Represents the available table sizes -/
inductive TableSize
  | Four
  | Six
  | Eight

/-- Calculates the minimum number of tables needed -/
def minTablesNeeded (totalInvited : ℕ) (noShows : ℕ) (tableSizes : List TableSize) : ℕ :=
  sorry

/-- Theorem stating the minimum number of tables needed for the given problem -/
theorem conference_tables_theorem (totalInvited noShows : ℕ) (tableSizes : List TableSize) :
  totalInvited = 75 →
  noShows = 33 →
  tableSizes = [TableSize.Four, TableSize.Six, TableSize.Eight] →
  minTablesNeeded totalInvited noShows tableSizes = 6 :=
sorry

end NUMINAMATH_CALUDE_conference_tables_theorem_l2002_200264


namespace NUMINAMATH_CALUDE_units_digit_characteristic_l2002_200294

theorem units_digit_characteristic (p : ℕ) : 
  p > 0 ∧ 
  Even p ∧ 
  (p^3 % 10 - p^2 % 10 = 0) ∧ 
  ((p + 1) % 10 = 7) → 
  p % 10 = 6 := by
sorry

end NUMINAMATH_CALUDE_units_digit_characteristic_l2002_200294


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2002_200251

-- Define the inverse relationship between a^3 and b^4
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^3 * b^4 = k

-- Theorem statement
theorem inverse_variation_problem (a₀ b₀ a₁ b₁ : ℝ) 
  (h_inverse : inverse_relation a₀ b₀ ∧ inverse_relation a₁ b₁)
  (h_initial : a₀ = 2 ∧ b₀ = 4)
  (h_final_a : a₁ = 8) :
  b₁ = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2002_200251


namespace NUMINAMATH_CALUDE_function_minimum_l2002_200255

theorem function_minimum (a : ℝ) :
  (∀ x : ℝ, x^2 + a*x - 3*a - 9 ≥ 0) →
  (1^2 + a*1 - 3*a - 9 = 4) :=
by
  sorry

end NUMINAMATH_CALUDE_function_minimum_l2002_200255


namespace NUMINAMATH_CALUDE_fraction_simplification_l2002_200292

theorem fraction_simplification (x y : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hsum : x + 1/y ≠ 0) :
  (x + 1/y) / (y + 1/x) = x/y := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2002_200292


namespace NUMINAMATH_CALUDE_mike_video_game_days_l2002_200279

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The number of hours Mike watches TV per day -/
def tv_hours_per_day : ℕ := 4

/-- The total hours Mike spends on TV and video games in a week -/
def total_hours_per_week : ℕ := 34

/-- The number of days Mike plays video games in a week -/
def video_game_days : ℕ := 3

theorem mike_video_game_days :
  ∃ (video_game_hours_per_day : ℕ),
    video_game_hours_per_day = tv_hours_per_day / 2 ∧
    video_game_days * video_game_hours_per_day =
      total_hours_per_week - (days_in_week * tv_hours_per_day) :=
by
  sorry

end NUMINAMATH_CALUDE_mike_video_game_days_l2002_200279


namespace NUMINAMATH_CALUDE_journey_problem_solution_exists_l2002_200260

/-- Proves the existence of a solution for the journey problem -/
theorem journey_problem_solution_exists :
  ∃ (x y T : ℝ),
    x > 0 ∧ y > 0 ∧ T > 0 ∧
    x < 150 ∧ y < x ∧
    (x / 30 + (150 - x) / 3 = T) ∧
    (x / 30 + y / 30 + (150 - (x - y)) / 30 = T) ∧
    ((x - y) / 10 + (150 - (x - y)) / 30 = T) :=
by sorry

#check journey_problem_solution_exists

end NUMINAMATH_CALUDE_journey_problem_solution_exists_l2002_200260


namespace NUMINAMATH_CALUDE_minimum_choir_size_l2002_200266

def is_valid_choir_size (n : ℕ) : Prop :=
  n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0

theorem minimum_choir_size :
  ∃ (n : ℕ), is_valid_choir_size n ∧ ∀ (m : ℕ), m < n → ¬ is_valid_choir_size m :=
by
  use 360
  sorry

end NUMINAMATH_CALUDE_minimum_choir_size_l2002_200266


namespace NUMINAMATH_CALUDE_circle_radius_theorem_l2002_200202

/-- The radius of a circle concentric with and outside a regular octagon -/
def circle_radius (octagon_side_length : ℝ) (probability_four_sides : ℝ) : ℝ :=
  sorry

/-- The theorem stating the relationship between the circle radius, octagon side length, and probability of seeing four sides -/
theorem circle_radius_theorem (octagon_side_length : ℝ) (probability_four_sides : ℝ) :
  circle_radius octagon_side_length probability_four_sides = 6 * Real.sqrt 2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_theorem_l2002_200202
