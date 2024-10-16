import Mathlib

namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_plus_one_less_than_zero_l1409_140949

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) := by sorry

theorem negation_of_squared_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_squared_plus_one_less_than_zero_l1409_140949


namespace NUMINAMATH_CALUDE_selected_student_in_range_l1409_140985

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) (n : ℕ) : ℕ :=
  firstSelected + (n - 1) * (totalStudents / sampleSize)

/-- Theorem: The selected student number in the range 33 to 48 is 39 -/
theorem selected_student_in_range (totalStudents : ℕ) (sampleSize : ℕ) (firstSelected : ℕ) :
  totalStudents = 800 →
  sampleSize = 50 →
  firstSelected = 7 →
  ∃ n : ℕ, systematicSample totalStudents sampleSize firstSelected n ∈ Set.Icc 33 48 ∧
           systematicSample totalStudents sampleSize firstSelected n = 39 :=
by
  sorry


end NUMINAMATH_CALUDE_selected_student_in_range_l1409_140985


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1409_140939

theorem complex_expression_equality : 
  (Real.sqrt 3 + 5) * (5 - Real.sqrt 3) - 
  (Real.sqrt 8 + 2 * Real.sqrt (1/2)) / Real.sqrt 2 + 
  Real.sqrt ((Real.sqrt 5 - 3)^2) = 22 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1409_140939


namespace NUMINAMATH_CALUDE_highest_points_is_38_l1409_140999

/-- The TRISQUARE game awards points for triangles and squares --/
structure TRISQUARE where
  small_triangles : ℕ
  large_triangles : ℕ
  small_squares : ℕ
  large_squares : ℕ
  triangle_points : ℕ
  square_points : ℕ

/-- Calculate the total points for a TRISQUARE game --/
def total_points (game : TRISQUARE) : ℕ :=
  (game.small_triangles + game.large_triangles) * game.triangle_points +
  (game.small_squares + game.large_squares) * game.square_points

/-- Theorem: The highest number of points achievable in the given TRISQUARE game is 38 --/
theorem highest_points_is_38 (game : TRISQUARE) 
  (h1 : game.small_triangles = 4)
  (h2 : game.large_triangles = 2)
  (h3 : game.small_squares = 4)
  (h4 : game.large_squares = 1)
  (h5 : game.triangle_points = 3)
  (h6 : game.square_points = 4) :
  total_points game = 38 := by
  sorry

#check highest_points_is_38

end NUMINAMATH_CALUDE_highest_points_is_38_l1409_140999


namespace NUMINAMATH_CALUDE_wrong_calculation_correction_l1409_140937

theorem wrong_calculation_correction (x : ℝ) : 
  x / 5 + 16 = 58 → x / 15 + 74 = 88 := by
  sorry

end NUMINAMATH_CALUDE_wrong_calculation_correction_l1409_140937


namespace NUMINAMATH_CALUDE_product_of_solutions_l1409_140978

theorem product_of_solutions (x : ℝ) : 
  (|18 / x + 4| = 3) → (∃ y : ℝ, (|18 / y + 4| = 3) ∧ x * y = 324 / 7) := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_l1409_140978


namespace NUMINAMATH_CALUDE_intersection_M_N_l1409_140962

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 4 > 0}
def N : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | x < -2} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1409_140962


namespace NUMINAMATH_CALUDE_divisibility_of_245245_by_35_l1409_140948

theorem divisibility_of_245245_by_35 : 35 ∣ 245245 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_245245_by_35_l1409_140948


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1409_140903

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def team_size : ℕ := 7
def quadruplets_in_team : ℕ := 2

theorem basketball_team_selection :
  (Nat.choose quadruplets quadruplets_in_team) *
  (Nat.choose (total_players - quadruplets) (team_size - quadruplets_in_team)) = 12012 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1409_140903


namespace NUMINAMATH_CALUDE_complex_equation_modulus_l1409_140958

theorem complex_equation_modulus : ∃ (x y : ℝ), 
  (Complex.I + 1) * x + Complex.I * y = (Complex.I + 3 * Complex.I) * Complex.I ∧ 
  Complex.abs (x + Complex.I * y) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_modulus_l1409_140958


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l1409_140960

def f (x : ℝ) := x^2 - 1

theorem inequality_holds_iff_m_in_range :
  ∀ m : ℝ, (∀ x ≥ 3, f (x / m) - 4 * m^2 * f x ≤ f (x - 1) + 4 * f m) ↔
    m ≤ -Real.sqrt 2 / 2 ∨ m ≥ Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l1409_140960


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1409_140983

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (6 * y^12 + 3 * y^11 + 6 * y^10 + 3 * y^9) =
  18 * y^13 - 3 * y^12 + 12 * y^11 - 3 * y^10 - 6 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1409_140983


namespace NUMINAMATH_CALUDE_no_valid_base_for_450_l1409_140911

def is_four_digit (n : ℕ) (b : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

def last_digit (n : ℕ) (b : ℕ) : ℕ :=
  n % b

theorem no_valid_base_for_450 :
  ¬ ∃ (b : ℕ), b > 1 ∧ is_four_digit 450 b ∧ Odd (last_digit 450 b) :=
sorry

end NUMINAMATH_CALUDE_no_valid_base_for_450_l1409_140911


namespace NUMINAMATH_CALUDE_lea_purchases_cost_l1409_140990

/-- The cost of Léa's purchases -/
def total_cost (book_price : ℕ) (binder_price : ℕ) (notebook_price : ℕ) 
  (num_binders : ℕ) (num_notebooks : ℕ) : ℕ :=
  book_price + (binder_price * num_binders) + (notebook_price * num_notebooks)

/-- Theorem stating that the total cost of Léa's purchases is $28 -/
theorem lea_purchases_cost :
  total_cost 16 2 1 3 6 = 28 := by
  sorry

end NUMINAMATH_CALUDE_lea_purchases_cost_l1409_140990


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1409_140991

/-- Represents a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The first line from the problem -/
def line1 (a : ℝ) : Line :=
  { a := a, b := -1, c := 3 }

/-- The second line from the problem -/
def line2 (a : ℝ) : Line :=
  { a := 2, b := -(a+1), c := 4 }

/-- The condition a=-2 is sufficient for the lines to be parallel -/
theorem sufficient_condition :
  parallel (line1 (-2)) (line2 (-2)) := by sorry

/-- The condition a=-2 is not necessary for the lines to be parallel -/
theorem not_necessary_condition :
  ∃ a : ℝ, a ≠ -2 ∧ parallel (line1 a) (line2 a) := by sorry

/-- The main theorem stating that a=-2 is a sufficient but not necessary condition -/
theorem sufficient_but_not_necessary :
  (parallel (line1 (-2)) (line2 (-2))) ∧
  (∃ a : ℝ, a ≠ -2 ∧ parallel (line1 a) (line2 a)) := by sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l1409_140991


namespace NUMINAMATH_CALUDE_total_people_in_line_l1409_140918

/-- Given a line of people at an amusement park ride, this theorem proves
    the total number of people in line based on Eunji's position. -/
theorem total_people_in_line (eunji_position : ℕ) (people_behind_eunji : ℕ) :
  eunji_position = 6 →
  people_behind_eunji = 7 →
  eunji_position + people_behind_eunji = 13 := by
  sorry

#check total_people_in_line

end NUMINAMATH_CALUDE_total_people_in_line_l1409_140918


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_l1409_140912

/-- Tetrahedron with specific properties -/
structure Tetrahedron where
  -- Base is a square with side length 3
  baseSideLength : ℝ := 3
  -- PD length is 4
  pdLength : ℝ := 4
  -- Lateral faces PAD and PCD are perpendicular to the base
  lateralFacesPerpendicular : Prop

/-- Calculate the surface area of the tetrahedron -/
def surfaceArea (t : Tetrahedron) : ℝ :=
  -- We don't implement the actual calculation here
  sorry

/-- Theorem stating the surface area of the tetrahedron -/
theorem tetrahedron_surface_area (t : Tetrahedron) : 
  surfaceArea t = 9 + 6 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_surface_area_l1409_140912


namespace NUMINAMATH_CALUDE_prob_both_paper_is_one_ninth_l1409_140923

/-- Represents the possible choices in rock-paper-scissors -/
inductive Choice
| Rock
| Paper
| Scissors

/-- Represents the outcome of a rock-paper-scissors game -/
structure GameOutcome :=
  (player1 : Choice)
  (player2 : Choice)

/-- The set of all possible game outcomes -/
def allOutcomes : Finset GameOutcome :=
  sorry

/-- The set of outcomes where both players choose paper -/
def bothPaperOutcomes : Finset GameOutcome :=
  sorry

/-- The probability of both players choosing paper -/
def probBothPaper : ℚ :=
  (bothPaperOutcomes.card : ℚ) / (allOutcomes.card : ℚ)

theorem prob_both_paper_is_one_ninth :
  probBothPaper = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_paper_is_one_ninth_l1409_140923


namespace NUMINAMATH_CALUDE_circle_area_equality_l1409_140941

theorem circle_area_equality (r₁ r₂ r₃ : ℝ) : 
  r₁ = 13 → r₂ = 23 → π * r₃^2 = π * (r₂^2 - r₁^2) → r₃ = 6 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_equality_l1409_140941


namespace NUMINAMATH_CALUDE_simplify_fraction_expression_l1409_140914

theorem simplify_fraction_expression (x y : ℝ) (hx : x ≠ 0) (hxy : x + y ≠ 0) :
  1 / (2 * x) - 1 / (x + y) * ((x + y) / (2 * x) - x - y) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_expression_l1409_140914


namespace NUMINAMATH_CALUDE_rahul_twice_mary_age_l1409_140963

/-- Proves that Rahul will be twice as old as Mary after 20 years -/
theorem rahul_twice_mary_age : ∀ (x : ℕ),
  let mary_age : ℕ := 10
  let rahul_age : ℕ := mary_age + 30
  x = 20 ↔ rahul_age + x = 2 * (mary_age + x) :=
by sorry

end NUMINAMATH_CALUDE_rahul_twice_mary_age_l1409_140963


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l1409_140959

theorem cos_thirty_degrees : Real.cos (30 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l1409_140959


namespace NUMINAMATH_CALUDE_count_negative_numbers_l1409_140971

def given_set : Finset Int := {-3, -2, 0, 5}

theorem count_negative_numbers : 
  (given_set.filter (λ x => x < 0)).card = 2 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l1409_140971


namespace NUMINAMATH_CALUDE_b_investment_value_l1409_140988

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit_share : ℕ

/-- Theorem stating that given the conditions, B's investment is 32,000 --/
theorem b_investment_value (p : Partnership) 
  (h1 : p.a_investment = 24000)
  (h2 : p.c_investment = 36000)
  (h3 : p.total_profit = 92000)
  (h4 : p.c_profit_share = 36000)
  (h5 : p.c_investment / p.c_profit_share = (p.a_investment + p.b_investment + p.c_investment) / p.total_profit) : 
  p.b_investment = 32000 := by
  sorry

#check b_investment_value

end NUMINAMATH_CALUDE_b_investment_value_l1409_140988


namespace NUMINAMATH_CALUDE_largest_n_perfect_cube_l1409_140932

theorem largest_n_perfect_cube (n : ℕ) : n = 497 ↔ 
  (n < 500 ∧ 
   ∃ m : ℕ, 6048 * 28^n = m^3 ∧ 
   ∀ k : ℕ, k < 500 ∧ k > n → ¬∃ l : ℕ, 6048 * 28^k = l^3) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_perfect_cube_l1409_140932


namespace NUMINAMATH_CALUDE_calculation_proof_l1409_140910

theorem calculation_proof : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1409_140910


namespace NUMINAMATH_CALUDE_perpendicular_equivalence_l1409_140972

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the theorem
theorem perpendicular_equivalence 
  (α β : Plane) (m n : Line) 
  (h_diff_planes : α ≠ β) 
  (h_diff_lines : m ≠ n) 
  (h_n_perp_α : perp n α) 
  (h_n_perp_β : perp n β) :
  perp m α ↔ perp m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_equivalence_l1409_140972


namespace NUMINAMATH_CALUDE_typing_time_proportional_l1409_140916

/-- Given that 450 characters can be typed in 9 minutes, 
    prove that 1800 characters can be typed in 36 minutes. -/
theorem typing_time_proportional 
  (chars_per_9min : ℕ) 
  (h_chars : chars_per_9min = 450) :
  (1800 : ℝ) / (36 : ℝ) = (chars_per_9min : ℝ) / 9 :=
by sorry

end NUMINAMATH_CALUDE_typing_time_proportional_l1409_140916


namespace NUMINAMATH_CALUDE_problem_statement_l1409_140926

theorem problem_statement (a b : ℝ) (h1 : a^2 + b^2 = 1) :
  (|a - b| / |1 - a*b| ≤ 1) ∧
  (a*b > 0 → (a + b)*(a^3 + b^3) ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1409_140926


namespace NUMINAMATH_CALUDE_modulus_of_z_l1409_140943

theorem modulus_of_z (z : ℂ) (h : (z + Complex.I) * (1 + Complex.I) = 1 - Complex.I) :
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1409_140943


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l1409_140980

theorem smallest_base_perfect_square : 
  ∃ (b : ℕ), b > 4 ∧ ∃ (n : ℕ), 4 * b + 5 = n^2 ∧ 
  ∀ (x : ℕ), x > 4 ∧ x < b → ¬∃ (m : ℕ), 4 * x + 5 = m^2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l1409_140980


namespace NUMINAMATH_CALUDE_minimum_order_amount_correct_l1409_140924

/-- The minimum order amount to get a discount at Silvia's bakery -/
def minimum_order_amount : ℝ := 60

/-- The discount percentage offered by the bakery -/
def discount_percentage : ℝ := 0.10

/-- The total cost of Silvia's order before discount -/
def order_cost : ℝ := 2 * 15 + 6 * 3 + 6 * 2

/-- The total cost of Silvia's order after discount -/
def discounted_cost : ℝ := 54

/-- Theorem stating that the minimum order amount to get the discount is correct -/
theorem minimum_order_amount_correct :
  minimum_order_amount = order_cost ∧
  discounted_cost = order_cost * (1 - discount_percentage) :=
sorry

end NUMINAMATH_CALUDE_minimum_order_amount_correct_l1409_140924


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l1409_140998

theorem quadratic_one_solution (n : ℝ) : 
  (∃! x : ℝ, 16 * x^2 + n * x + 4 = 0) ↔ (n = 16 ∨ n = -16) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l1409_140998


namespace NUMINAMATH_CALUDE_negation_of_exists_le_zero_is_forall_gt_zero_l1409_140973

theorem negation_of_exists_le_zero_is_forall_gt_zero :
  (¬ ∃ x : ℝ, (2 : ℝ) ^ x ≤ 0) ↔ (∀ x : ℝ, (2 : ℝ) ^ x > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_exists_le_zero_is_forall_gt_zero_l1409_140973


namespace NUMINAMATH_CALUDE_astronomy_club_committee_probability_l1409_140945

/-- The probability of selecting a committee with more boys than girls -/
theorem astronomy_club_committee_probability :
  let total_members : ℕ := 24
  let boys : ℕ := 14
  let girls : ℕ := 10
  let committee_size : ℕ := 5
  let total_committees : ℕ := Nat.choose total_members committee_size
  let committees_more_boys : ℕ := 
    Nat.choose boys 3 * Nat.choose girls 2 +
    Nat.choose boys 4 * Nat.choose girls 1 +
    Nat.choose boys 5
  (committees_more_boys : ℚ) / total_committees = 7098 / 10626 := by
sorry

end NUMINAMATH_CALUDE_astronomy_club_committee_probability_l1409_140945


namespace NUMINAMATH_CALUDE_eggs_per_set_l1409_140964

theorem eggs_per_set (total_eggs : ℕ) (num_sets : ℕ) (h1 : total_eggs = 108) (h2 : num_sets = 9) :
  total_eggs / num_sets = 12 := by
sorry

end NUMINAMATH_CALUDE_eggs_per_set_l1409_140964


namespace NUMINAMATH_CALUDE_wall_building_time_l1409_140928

theorem wall_building_time
  (workers_initial : ℕ)
  (days_initial : ℕ)
  (workers_new : ℕ)
  (h1 : workers_initial = 60)
  (h2 : days_initial = 3)
  (h3 : workers_new = 30)
  (h4 : workers_initial > 0)
  (h5 : workers_new > 0)
  (h6 : days_initial > 0) :
  let days_new := workers_initial * days_initial / workers_new
  days_new = 6 :=
by sorry

end NUMINAMATH_CALUDE_wall_building_time_l1409_140928


namespace NUMINAMATH_CALUDE_linear_function_value_l1409_140906

/-- A linear function in three variables -/
def LinearFunction (f : ℝ → ℝ → ℝ → ℝ) : Prop :=
  ∀ a b c x y z : ℝ, f (a + x) (b + y) (c + z) = f a b c + f x y z

theorem linear_function_value (f : ℝ → ℝ → ℝ → ℝ) 
  (h_linear : LinearFunction f)
  (h_value_3 : f 3 3 3 = 1 / (3 * 3 * 3))
  (h_value_4 : f 4 4 4 = 1 / (4 * 4 * 4)) :
  f 5 5 5 = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_value_l1409_140906


namespace NUMINAMATH_CALUDE_log_abs_eq_sin_roots_l1409_140995

noncomputable def log_abs (x : ℝ) : ℝ := Real.log (abs x)

theorem log_abs_eq_sin_roots :
  let f (x : ℝ) := log_abs x - Real.sin x
  ∃ (S : Finset ℝ), (∀ x ∈ S, f x = 0) ∧ S.card = 10 ∧ 
    (∀ y : ℝ, f y = 0 → y ∈ S) := by sorry

end NUMINAMATH_CALUDE_log_abs_eq_sin_roots_l1409_140995


namespace NUMINAMATH_CALUDE_peter_age_approx_l1409_140929

def cindy_age : ℕ := 5

def jan_age : ℕ := cindy_age + 2

def marcia_age : ℕ := 2 * jan_age

def greg_age : ℕ := marcia_age + 2

def bobby_age : ℕ := (3 * greg_age) / 2

noncomputable def peter_age : ℝ := 2 * Real.sqrt (bobby_age : ℝ)

theorem peter_age_approx : 
  ∀ ε > 0, |peter_age - 10| < ε := by sorry

end NUMINAMATH_CALUDE_peter_age_approx_l1409_140929


namespace NUMINAMATH_CALUDE_zeros_of_f_l1409_140967

def f (x : ℝ) : ℝ := x^2 - 3*x + 2

theorem zeros_of_f : 
  {x : ℝ | f x = 0} = {1, 2} := by sorry

end NUMINAMATH_CALUDE_zeros_of_f_l1409_140967


namespace NUMINAMATH_CALUDE_binomial_60_3_l1409_140989

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l1409_140989


namespace NUMINAMATH_CALUDE_correct_stratified_sampling_l1409_140996

/-- Represents the number of students in each year --/
structure StudentCounts where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ

/-- Calculates the stratified sample size for a given year --/
def stratifiedSampleSize (yearCount : ℕ) (totalCount : ℕ) (sampleSize : ℕ) : ℕ :=
  (yearCount * sampleSize + totalCount - 1) / totalCount

/-- Theorem stating the correct stratified sampling for the given problem --/
theorem correct_stratified_sampling (totalStudents : StudentCounts) 
    (h1 : totalStudents.firstYear = 540)
    (h2 : totalStudents.secondYear = 440)
    (h3 : totalStudents.thirdYear = 420)
    (totalSampleSize : ℕ) 
    (h4 : totalSampleSize = 70) :
  let totalCount := totalStudents.firstYear + totalStudents.secondYear + totalStudents.thirdYear
  (stratifiedSampleSize totalStudents.firstYear totalCount totalSampleSize,
   stratifiedSampleSize totalStudents.secondYear totalCount totalSampleSize,
   stratifiedSampleSize totalStudents.thirdYear totalCount totalSampleSize) = (27, 22, 21) := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sampling_l1409_140996


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l1409_140976

/-- Given a point A with coordinates (3,2), prove that the point symmetric 
    to A' with respect to the y-axis has coordinates (1,2), where A' is obtained 
    by translating A 4 units left along the x-axis. -/
theorem symmetric_point_coordinates : 
  let A : ℝ × ℝ := (3, 2)
  let A' : ℝ × ℝ := (A.1 - 4, A.2)
  let symmetric_point : ℝ × ℝ := (-A'.1, A'.2)
  symmetric_point = (1, 2) := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l1409_140976


namespace NUMINAMATH_CALUDE_restaurant_sales_restaurant_sales_proof_l1409_140994

/-- Calculates the total sales of a restaurant given the number of meals sold at different price points. -/
theorem restaurant_sales (meals_at_8 meals_at_10 meals_at_4 : ℕ) 
  (price_8 price_10 price_4 : ℕ) : ℕ :=
  let total_sales := meals_at_8 * price_8 + meals_at_10 * price_10 + meals_at_4 * price_4
  total_sales

/-- Proves that the restaurant's total sales for the day is $210. -/
theorem restaurant_sales_proof :
  restaurant_sales 10 5 20 8 10 4 = 210 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_sales_restaurant_sales_proof_l1409_140994


namespace NUMINAMATH_CALUDE_sum_of_roots_l1409_140920

theorem sum_of_roots (a b c d : ℝ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (∀ x : ℝ, x^2 - 8*a*x - 9*b = 0 ↔ (x = c ∨ x = d)) →
  (∀ x : ℝ, x^2 - 8*c*x - 9*d = 0 ↔ (x = a ∨ x = b)) →
  a + b + c + d = 648 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1409_140920


namespace NUMINAMATH_CALUDE_farm_animals_l1409_140965

theorem farm_animals (chickens buffalos : ℕ) : 
  chickens + buffalos = 9 →
  2 * chickens + 4 * buffalos = 26 →
  chickens = 5 := by
sorry

end NUMINAMATH_CALUDE_farm_animals_l1409_140965


namespace NUMINAMATH_CALUDE_dow_decrease_l1409_140900

def initial_dow : ℝ := 8900
def percentage_decrease : ℝ := 0.02

theorem dow_decrease (initial : ℝ) (decrease : ℝ) :
  initial = initial_dow →
  decrease = percentage_decrease →
  initial * (1 - decrease) = 8722 :=
by sorry

end NUMINAMATH_CALUDE_dow_decrease_l1409_140900


namespace NUMINAMATH_CALUDE_final_value_after_percentage_changes_l1409_140930

theorem final_value_after_percentage_changes (initial_value : ℝ) 
  (increase_percent : ℝ) (decrease_percent : ℝ) : 
  initial_value = 1500 → 
  increase_percent = 20 → 
  decrease_percent = 40 → 
  let increased_value := initial_value * (1 + increase_percent / 100)
  let final_value := increased_value * (1 - decrease_percent / 100)
  final_value = 1080 := by
  sorry

end NUMINAMATH_CALUDE_final_value_after_percentage_changes_l1409_140930


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1409_140947

/-- A rectangle with given diagonal and area has a specific perimeter -/
theorem rectangle_perimeter (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  a^2 + b^2 = 25^2 → a * b = 168 → 2 * (a + b) = 62 := by
  sorry

#check rectangle_perimeter

end NUMINAMATH_CALUDE_rectangle_perimeter_l1409_140947


namespace NUMINAMATH_CALUDE_certain_number_proof_l1409_140969

theorem certain_number_proof (y : ℝ) : 
  (0.25 * 680 = 0.20 * y - 30) → y = 1000 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1409_140969


namespace NUMINAMATH_CALUDE_abs_neg_one_sixth_gt_neg_one_seventh_l1409_140913

theorem abs_neg_one_sixth_gt_neg_one_seventh : |-(1/6)| > -(1/7) := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_one_sixth_gt_neg_one_seventh_l1409_140913


namespace NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l1409_140921

-- Define the number of erasers for each person
def jungkook_erasers : ℕ := 6
def jimin_erasers : ℕ := jungkook_erasers + 4
def seokjin_erasers : ℕ := jimin_erasers - 3

-- Theorem to prove Jungkook has the fewest erasers
theorem jungkook_has_fewest_erasers :
  jungkook_erasers < jimin_erasers ∧ jungkook_erasers < seokjin_erasers :=
by sorry

end NUMINAMATH_CALUDE_jungkook_has_fewest_erasers_l1409_140921


namespace NUMINAMATH_CALUDE_least_positive_integer_with_specific_remainders_l1409_140953

theorem least_positive_integer_with_specific_remainders : ∃ n : ℕ, 
  (n > 0) ∧ 
  (n % 4 = 3) ∧ 
  (n % 5 = 4) ∧ 
  (n % 6 = 5) ∧ 
  (n % 7 = 6) ∧ 
  (n % 11 = 10) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 ∧ m % 11 = 10 → m ≥ n) ∧
  n = 4619 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_specific_remainders_l1409_140953


namespace NUMINAMATH_CALUDE_triangle_problem_l1409_140936

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement for the given triangle problem -/
theorem triangle_problem (t : Triangle) 
  (h_area : (1/2) * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 15)
  (h_diff : t.b - t.c = 2)
  (h_cosA : Real.cos t.A = -1/4) : 
  t.a = 8 ∧ Real.sin t.C = Real.sqrt 15 / 8 ∧ 
  Real.cos (2 * t.A + π/6) = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1409_140936


namespace NUMINAMATH_CALUDE_three_digit_number_proof_l1409_140986

theorem three_digit_number_proof :
  ∃! (a b c : ℕ),
    0 < a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    a + b + c = 16 ∧
    100 * b + 10 * a + c = 100 * a + 10 * b + c - 360 ∧
    100 * a + 10 * c + b = 100 * a + 10 * b + c + 54 ∧
    100 * a + 10 * b + c = 628 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_proof_l1409_140986


namespace NUMINAMATH_CALUDE_no_prime_in_first_15_cumulative_sums_l1409_140907

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def nthPrime (n : ℕ) : ℕ := sorry

def cumulativePrimeSum (n : ℕ) : ℕ := 
  if n = 0 then 0 else cumulativePrimeSum (n-1) + nthPrime (n+1)

theorem no_prime_in_first_15_cumulative_sums : 
  ∀ n : ℕ, n > 0 → n ≤ 15 → ¬(isPrime (cumulativePrimeSum n)) :=
sorry

end NUMINAMATH_CALUDE_no_prime_in_first_15_cumulative_sums_l1409_140907


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l1409_140922

theorem arithmetic_mean_problem (a₁ a₂ a₃ a₄ a₅ a₆ A : ℝ) 
  (h1 : (a₁ + a₂ + a₃ + a₄ + a₅ + a₆) / 6 = A)
  (h2 : (a₁ + a₂ + a₃ + a₄) / 4 = A + 10)
  (h3 : (a₃ + a₄ + a₅ + a₆) / 4 = A - 7) :
  (a₁ + a₂ + a₅ + a₆) / 4 = A - 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l1409_140922


namespace NUMINAMATH_CALUDE_collin_savings_l1409_140974

/-- Represents the number of cans Collin collected from various sources --/
structure CanCollection where
  home : ℕ
  grandparents : ℕ
  neighbor : ℕ
  dad : ℕ

/-- Calculates the total number of cans collected --/
def total_cans (c : CanCollection) : ℕ :=
  c.home + c.grandparents + c.neighbor + c.dad

/-- Represents the recycling scenario for Collin --/
structure RecyclingScenario where
  collection : CanCollection
  price_per_can : ℚ
  savings_ratio : ℚ

/-- Calculates the amount Collin will put into savings --/
def savings_amount (s : RecyclingScenario) : ℚ :=
  s.savings_ratio * s.price_per_can * (total_cans s.collection)

/-- Theorem stating that Collin will put $43.00 into savings --/
theorem collin_savings (s : RecyclingScenario) 
  (h1 : s.collection.home = 12)
  (h2 : s.collection.grandparents = 3 * s.collection.home)
  (h3 : s.collection.neighbor = 46)
  (h4 : s.collection.dad = 250)
  (h5 : s.price_per_can = 1/4)
  (h6 : s.savings_ratio = 1/2) :
  savings_amount s = 43 := by
  sorry

end NUMINAMATH_CALUDE_collin_savings_l1409_140974


namespace NUMINAMATH_CALUDE_folded_rectangle_long_side_l1409_140942

/-- A rectangular sheet of paper with a special folding property -/
structure FoldedRectangle where
  short_side : ℝ
  long_side : ℝ
  is_folded_to_midpoint : Bool
  triangles_congruent : Bool

/-- The folded rectangle satisfies the problem conditions -/
def satisfies_conditions (r : FoldedRectangle) : Prop :=
  r.short_side = 8 ∧ r.is_folded_to_midpoint ∧ r.triangles_congruent

/-- The theorem stating that under the given conditions, the long side must be 12 units -/
theorem folded_rectangle_long_side
  (r : FoldedRectangle)
  (h : satisfies_conditions r) :
  r.long_side = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_folded_rectangle_long_side_l1409_140942


namespace NUMINAMATH_CALUDE_triangle_properties_l1409_140975

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  b = 2 * Real.sqrt 3 →
  (Real.cos B) / (Real.cos C) = b / (2 * a - c) →
  1 / (Real.tan A) + 1 / (Real.tan B) = (Real.sin C) / (Real.sqrt 3 * Real.sin A * Real.cos B) →
  4 * Real.sqrt 3 * S + 3 * (b^2 - a^2) = 3 * c^2 →
  S = Real.sqrt 3 / 3 ∧
  (0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2 →
    (Real.sqrt 3 + 1) / 2 < (b + c) / a ∧ (b + c) / a < Real.sqrt 3 + 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1409_140975


namespace NUMINAMATH_CALUDE_pyramid_top_value_l1409_140981

/-- Represents a pyramid structure where each number is the product of the two numbers above it -/
structure Pyramid where
  bottom_row : Fin 3 → ℕ
  x : ℕ
  y : ℕ

/-- The conditions of the pyramid problem -/
def pyramid_conditions (p : Pyramid) : Prop :=
  p.bottom_row 0 = 240 ∧
  p.bottom_row 1 = 720 ∧
  p.bottom_row 2 = 1440 ∧
  p.x * 6 = 720

/-- The theorem stating that given the conditions, y must be 120 -/
theorem pyramid_top_value (p : Pyramid) (h : pyramid_conditions p) : p.y = 120 := by
  sorry

#check pyramid_top_value

end NUMINAMATH_CALUDE_pyramid_top_value_l1409_140981


namespace NUMINAMATH_CALUDE_five_double_prime_l1409_140957

-- Define the prime operation
def prime (q : ℝ) : ℝ := 3 * q - 3

-- Theorem statement
theorem five_double_prime : prime (prime 5) = 33 := by
  sorry

end NUMINAMATH_CALUDE_five_double_prime_l1409_140957


namespace NUMINAMATH_CALUDE_sqrt_fourth_power_eq_256_l1409_140993

theorem sqrt_fourth_power_eq_256 (x : ℝ) (h : (Real.sqrt x)^4 = 256) : x = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fourth_power_eq_256_l1409_140993


namespace NUMINAMATH_CALUDE_b_work_rate_l1409_140931

/-- Given work rates for individuals and groups, prove B's work rate -/
theorem b_work_rate 
  (a_rate : ℚ)
  (b_rate : ℚ)
  (c_rate : ℚ)
  (d_rate : ℚ)
  (h1 : a_rate = 1/4)
  (h2 : b_rate + c_rate = 1/2)
  (h3 : a_rate + c_rate = 1/2)
  (h4 : d_rate = 1/8)
  (h5 : a_rate + b_rate + d_rate = 1/(8/5)) :
  b_rate = 1/4 := by
sorry

end NUMINAMATH_CALUDE_b_work_rate_l1409_140931


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l1409_140966

def a : ℝ × ℝ := (1, 3)
def b (m : ℝ) : ℝ × ℝ := (m, 2*m - 1)

def collinear (u v : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * u.1 ∧ v.2 = k * u.2

theorem collinear_vectors_m_value :
  collinear a (b m) → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l1409_140966


namespace NUMINAMATH_CALUDE_union_and_intersection_when_m_is_3_intersection_empty_iff_l1409_140992

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | m + 1 < x ∧ x < 2*m}

-- Theorem for part 1
theorem union_and_intersection_when_m_is_3 :
  (A ∪ B 3 = {x | -2 ≤ x ∧ x < 6}) ∧ (A ∩ B 3 = ∅) := by sorry

-- Theorem for part 2
theorem intersection_empty_iff :
  ∀ m : ℝ, A ∩ B m = ∅ ↔ m ≤ 1 ∨ m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_union_and_intersection_when_m_is_3_intersection_empty_iff_l1409_140992


namespace NUMINAMATH_CALUDE_total_games_in_season_l1409_140927

/-- Calculate the number of games in a round-robin tournament -/
def num_games (n : ℕ) (r : ℕ) : ℕ :=
  (n * (n - 1) / 2) * r

/-- The number of teams in the league -/
def num_teams : ℕ := 14

/-- The number of times each team plays every other team -/
def num_rounds : ℕ := 5

theorem total_games_in_season :
  num_games num_teams num_rounds = 455 := by sorry

end NUMINAMATH_CALUDE_total_games_in_season_l1409_140927


namespace NUMINAMATH_CALUDE_rectangle_area_l1409_140950

theorem rectangle_area (length width : ℝ) :
  (2 * (length + width) = 48) →
  (length = width + 2) →
  (length * width = 143) :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1409_140950


namespace NUMINAMATH_CALUDE_k_range_for_two_solutions_l1409_140905

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x

noncomputable def g (x : ℝ) : ℝ := (log x) / x

theorem k_range_for_two_solutions (k : ℝ) :
  (∃ x y, x ∈ Set.Icc (1/ℯ) ℯ ∧ y ∈ Set.Icc (1/ℯ) ℯ ∧ x ≠ y ∧ f k x = g x ∧ f k y = g y) →
  k ∈ Set.Ioo (1/ℯ^2) (1/(2*ℯ)) :=
sorry

end NUMINAMATH_CALUDE_k_range_for_two_solutions_l1409_140905


namespace NUMINAMATH_CALUDE_b_is_geometric_sequence_l1409_140940

-- Define the geometric sequence a_n
def a (n : ℕ) (a₁ q : ℝ) : ℝ := a₁ * q^(n - 1)

-- Define the sequence b_n
def b (n : ℕ) (a₁ q : ℝ) : ℝ := a (3*n - 2) a₁ q + a (3*n - 1) a₁ q + a (3*n) a₁ q

-- Theorem statement
theorem b_is_geometric_sequence (a₁ q : ℝ) (hq : q ≠ 1) :
  ∀ n : ℕ, b (n + 1) a₁ q = (b n a₁ q) * q^3 :=
sorry

end NUMINAMATH_CALUDE_b_is_geometric_sequence_l1409_140940


namespace NUMINAMATH_CALUDE_intersection_sum_l1409_140908

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x + 1
def g (x y : ℝ) : Prop := x + 3*y = 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | f p.1 = p.2 ∧ g p.1 p.2}

-- State the theorem
theorem intersection_sum :
  ∃ (p₁ p₂ p₃ : ℝ × ℝ),
    p₁ ∈ intersection_points ∧
    p₂ ∈ intersection_points ∧
    p₃ ∈ intersection_points ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    p₁.1 + p₂.1 + p₃.1 = 3 ∧
    p₁.2 + p₂.2 + p₃.2 = 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1409_140908


namespace NUMINAMATH_CALUDE_outfit_combinations_l1409_140919

theorem outfit_combinations (shirts : ℕ) (ties : ℕ) (belts : ℕ) : 
  shirts = 8 → ties = 6 → belts = 4 → shirts * ties * belts = 192 := by
  sorry

end NUMINAMATH_CALUDE_outfit_combinations_l1409_140919


namespace NUMINAMATH_CALUDE_nadia_bought_20_roses_l1409_140984

/-- Represents the number of roses Nadia bought -/
def roses : ℕ := 20

/-- Represents the number of lilies Nadia bought -/
def lilies : ℚ := (3 / 4) * roses

/-- Cost of a single rose in dollars -/
def rose_cost : ℚ := 5

/-- Cost of a single lily in dollars -/
def lily_cost : ℚ := 2 * rose_cost

/-- Total amount spent on flowers in dollars -/
def total_spent : ℚ := 250

theorem nadia_bought_20_roses :
  roses * rose_cost + lilies * lily_cost = total_spent := by sorry

end NUMINAMATH_CALUDE_nadia_bought_20_roses_l1409_140984


namespace NUMINAMATH_CALUDE_simplify_power_expression_l1409_140951

theorem simplify_power_expression (x : ℝ) : (3 * x^4)^4 = 81 * x^16 := by sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l1409_140951


namespace NUMINAMATH_CALUDE_leak_drops_per_minute_l1409_140977

/-- Proves that the leak drips 3 drops per minute given the conditions -/
theorem leak_drops_per_minute 
  (drop_volume : ℝ) 
  (pot_capacity : ℝ) 
  (fill_time : ℝ) 
  (h1 : drop_volume = 20) 
  (h2 : pot_capacity = 3000) 
  (h3 : fill_time = 50) : 
  (pot_capacity / drop_volume) / fill_time = 3 := by
  sorry

#check leak_drops_per_minute

end NUMINAMATH_CALUDE_leak_drops_per_minute_l1409_140977


namespace NUMINAMATH_CALUDE_ellipse_equation_line_equation_l1409_140955

-- Define the ellipse
structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ
  foci_on_x_axis : Bool

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define the problem
def ellipse_problem (e : Ellipse) (B : Point) (Q : Point) (F : Point) : Prop :=
  e.center = (0, 0) ∧
  e.foci_on_x_axis = true ∧
  B = ⟨0, 1⟩ ∧
  Q = ⟨0, 3/2⟩ ∧
  F.y = 0 ∧
  F.x > 0 ∧
  (F.x - 0 + 2 * Real.sqrt 2) / Real.sqrt 2 = 3

-- Theorem for the ellipse equation
theorem ellipse_equation (e : Ellipse) (B : Point) (Q : Point) (F : Point) :
  ellipse_problem e B Q F →
  ∀ x y : ℝ, (x^2 / 3 + y^2 = 1) ↔ (x^2 / e.a^2 + y^2 / e.b^2 = 1) :=
sorry

-- Theorem for the line equation
theorem line_equation (e : Ellipse) (B : Point) (Q : Point) (F : Point) (l : Line) :
  ellipse_problem e B Q F →
  (∃ M N : Point,
    M ≠ N ∧
    (M.x^2 / 3 + M.y^2 = 1) ∧
    (N.x^2 / 3 + N.y^2 = 1) ∧
    M.y = l.slope * M.x + l.intercept ∧
    N.y = l.slope * N.x + l.intercept ∧
    (M.x - B.x)^2 + (M.y - B.y)^2 = (N.x - B.x)^2 + (N.y - B.y)^2) →
  (l.slope = Real.sqrt 6 / 3 ∧ l.intercept = 3/2) ∨
  (l.slope = -Real.sqrt 6 / 3 ∧ l.intercept = 3/2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_line_equation_l1409_140955


namespace NUMINAMATH_CALUDE_max_value_of_fraction_l1409_140925

theorem max_value_of_fraction (x : ℝ) : (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 9) ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_fraction_l1409_140925


namespace NUMINAMATH_CALUDE_cantaloupes_left_l1409_140968

/-- The number of cantaloupes left after growing and losing some due to bad weather -/
theorem cantaloupes_left (fred tim maria lost : ℕ) (h1 : fred = 38) (h2 : tim = 44) (h3 : maria = 57) (h4 : lost = 12) :
  fred + tim + maria - lost = 127 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_left_l1409_140968


namespace NUMINAMATH_CALUDE_f_value_at_one_l1409_140935

/-- A quadratic function f(x) with a specific behavior -/
def f (m : ℝ) (x : ℝ) : ℝ := 4 * x^2 - m * x + 5

/-- The function is increasing on [-2, +∞) -/
def increasing_on_right (m : ℝ) : Prop :=
  ∀ x y, -2 ≤ x ∧ x < y → f m x < f m y

/-- The function is decreasing on (-∞, -2] -/
def decreasing_on_left (m : ℝ) : Prop :=
  ∀ x y, x < y ∧ y ≤ -2 → f m x > f m y

theorem f_value_at_one (m : ℝ) 
  (h1 : increasing_on_right m) 
  (h2 : decreasing_on_left m) : 
  f m 1 = 25 := by sorry

end NUMINAMATH_CALUDE_f_value_at_one_l1409_140935


namespace NUMINAMATH_CALUDE_yoongis_rank_l1409_140970

theorem yoongis_rank (namjoons_rank yoongis_rank : ℕ) : 
  namjoons_rank = 2 →
  yoongis_rank = namjoons_rank + 10 →
  yoongis_rank = 12 :=
by sorry

end NUMINAMATH_CALUDE_yoongis_rank_l1409_140970


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1409_140901

/-- Represents a hyperbola with center (h, k), focus (h, f), and vertex (h, v) -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  f : ℝ
  v : ℝ

/-- The equation of the hyperbola is (y - k)²/a² - (x - h)²/b² = 1 -/
def hyperbola_equation (hyp : Hyperbola) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (y - hyp.k)^2 / a^2 - (x - hyp.h)^2 / b^2 = 1

/-- The theorem to be proved -/
theorem hyperbola_sum (hyp : Hyperbola) (a b : ℝ) :
  hyp.h = 1 ∧ hyp.k = 1 ∧ hyp.f = 7 ∧ hyp.v = -2 ∧ 
  hyperbola_equation hyp a b →
  hyp.h + hyp.k + a + b = 5 + 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1409_140901


namespace NUMINAMATH_CALUDE_expression_value_l1409_140956

theorem expression_value (x y : ℝ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1409_140956


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l1409_140946

/-- The area of a stripe wrapping around a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h_diameter : diameter = 40) 
  (h_stripe_width : stripe_width = 4) 
  (h_revolutions : revolutions = 3) : 
  stripe_width * revolutions * π * diameter = 480 * π := by
  sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l1409_140946


namespace NUMINAMATH_CALUDE_line_ratio_sum_l1409_140944

/-- Given two lines l₁ and l₂, and points P₁ and P₂ on these lines respectively,
    prove that the sum of certain ratios of the line coefficients equals 3. -/
theorem line_ratio_sum (a₁ b₁ c₁ a₂ b₂ c₂ x₁ y₁ x₂ y₂ : ℝ) : 
  a₁ * x₁ + b₁ * y₁ = c₁ →
  a₂ * x₂ + b₂ * y₂ = c₂ →
  a₁ + b₁ = c₁ →
  a₂ + b₂ = 2 * c₂ →
  (∀ x₁ y₁ x₂ y₂, (x₁ - x₂)^2 + (y₁ - y₂)^2 ≥ 1/2) →
  c₁ / a₁ + a₂ / c₂ = 3 := by
sorry

end NUMINAMATH_CALUDE_line_ratio_sum_l1409_140944


namespace NUMINAMATH_CALUDE_floor_sum_example_l1409_140934

theorem floor_sum_example : ⌊(2.7 : ℝ) + 1.5⌋ = 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l1409_140934


namespace NUMINAMATH_CALUDE_product_equation_solution_l1409_140917

theorem product_equation_solution : ∃! (B : ℕ), 
  B < 10 ∧ (10 * B + 2) * (90 + B) = 8016 := by sorry

end NUMINAMATH_CALUDE_product_equation_solution_l1409_140917


namespace NUMINAMATH_CALUDE_divisibility_by_101_l1409_140987

theorem divisibility_by_101 : ∃! (x y : Nat), x < 10 ∧ y < 10 ∧ (201300 + 10 * x + y) % 101 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_101_l1409_140987


namespace NUMINAMATH_CALUDE_complex_expression_equals_minus_half_minus_half_i_l1409_140933

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- The complex number (1+i)^2 / (1-i)^3 -/
noncomputable def complex_expression : ℂ := (1 + i)^2 / (1 - i)^3

/-- Theorem stating that the complex expression equals -1/2 - 1/2i -/
theorem complex_expression_equals_minus_half_minus_half_i :
  complex_expression = -1/2 - 1/2 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equals_minus_half_minus_half_i_l1409_140933


namespace NUMINAMATH_CALUDE_sequence_identity_l1409_140954

def IsIncreasing (a : ℕ → ℕ) : Prop :=
  ∀ i j, i ≤ j → a i ≤ a j

def DivisorCountEqual (a : ℕ → ℕ) : Prop :=
  ∀ i j, (Nat.divisors (i + j)).card = (Nat.divisors (a i + a j)).card

theorem sequence_identity (a : ℕ → ℕ) 
    (h1 : IsIncreasing a) 
    (h2 : DivisorCountEqual a) : 
    ∀ n : ℕ, a n = n := by
  sorry

end NUMINAMATH_CALUDE_sequence_identity_l1409_140954


namespace NUMINAMATH_CALUDE_existence_of_cube_sum_equal_100_power_100_l1409_140979

theorem existence_of_cube_sum_equal_100_power_100 : 
  ∃ (a b c d : ℕ), a^3 + b^3 + c^3 + d^3 = 100^100 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_cube_sum_equal_100_power_100_l1409_140979


namespace NUMINAMATH_CALUDE_remaining_money_l1409_140938

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def john_savings : Nat := base_to_decimal [5, 3, 2, 5] 9
def ticket_cost : Nat := base_to_decimal [0, 5, 2, 1] 8

theorem remaining_money :
  john_savings - ticket_cost = 3159 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l1409_140938


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1409_140902

-- Define the set A
def A : Set ℝ := {y | ∃ x ∈ Set.Icc (-1/2 : ℝ) 2, y = x^2 - (3/2)*x + 1}

-- Define the set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | |x - m| ≥ 1}

-- Define the range of m
def m_range : Set ℝ := Set.Iic (-9/16) ∪ Set.Ici 3

-- Theorem statement
theorem necessary_not_sufficient_condition (m : ℝ) :
  (∀ t, t ∈ A → t ∈ B m) ∧ (∃ t, t ∈ B m ∧ t ∉ A) ↔ m ∈ m_range :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1409_140902


namespace NUMINAMATH_CALUDE_photo_reactions_l1409_140997

/-- 
Proves that given a photo with a starting score of 0, where "thumbs up" increases 
the score by 1 and "thumbs down" decreases it by 1, if the current score is 50 
and 75% of reactions are "thumbs up", then the total number of reactions is 100.
-/
theorem photo_reactions 
  (score : ℤ) 
  (total_reactions : ℕ) 
  (thumbs_up_ratio : ℚ) :
  score = 0 + total_reactions * thumbs_up_ratio - total_reactions * (1 - thumbs_up_ratio) →
  score = 50 →
  thumbs_up_ratio = 3/4 →
  total_reactions = 100 := by
  sorry

#check photo_reactions

end NUMINAMATH_CALUDE_photo_reactions_l1409_140997


namespace NUMINAMATH_CALUDE_cartons_used_is_38_l1409_140952

/-- Represents the packing of tennis rackets into cartons. -/
structure RacketPacking where
  total_rackets : ℕ
  cartons_of_three : ℕ
  cartons_of_two : ℕ

/-- Calculates the total number of cartons used. -/
def total_cartons (packing : RacketPacking) : ℕ :=
  packing.cartons_of_two + packing.cartons_of_three

/-- Theorem stating that for the given packing scenario, 38 cartons are used in total. -/
theorem cartons_used_is_38 (packing : RacketPacking) 
  (h1 : packing.total_rackets = 100)
  (h2 : packing.cartons_of_three = 24)
  (h3 : 2 * packing.cartons_of_two + 3 * packing.cartons_of_three = packing.total_rackets) :
  total_cartons packing = 38 := by
  sorry

#check cartons_used_is_38

end NUMINAMATH_CALUDE_cartons_used_is_38_l1409_140952


namespace NUMINAMATH_CALUDE_garden_area_l1409_140915

/-- The area of a rectangular garden with dimensions 90 cm and 4.5 meters is 4.05 square meters. -/
theorem garden_area : 
  let length_cm : ℝ := 90
  let width_m : ℝ := 4.5
  let length_m : ℝ := length_cm / 100
  let area_m2 : ℝ := length_m * width_m
  area_m2 = 4.05 := by
sorry

end NUMINAMATH_CALUDE_garden_area_l1409_140915


namespace NUMINAMATH_CALUDE_yarn_balls_per_sweater_l1409_140909

/-- The number of balls of yarn needed for each sweater -/
def balls_per_sweater : ℕ := sorry

/-- The cost of each ball of yarn in dollars -/
def yarn_cost : ℕ := 6

/-- The selling price of each sweater in dollars -/
def sweater_price : ℕ := 35

/-- The number of sweaters sold -/
def sweaters_sold : ℕ := 28

/-- The total profit from selling all sweaters in dollars -/
def total_profit : ℕ := 308

theorem yarn_balls_per_sweater :
  (sweaters_sold * (sweater_price - yarn_cost * balls_per_sweater) = total_profit) →
  balls_per_sweater = 4 := by sorry

end NUMINAMATH_CALUDE_yarn_balls_per_sweater_l1409_140909


namespace NUMINAMATH_CALUDE_supplement_congruence_l1409_140961

/-- Two angles are congruent if they have the same measure -/
def congruent_angles (α β : Real) : Prop := α = β

/-- The supplement of an angle is another angle that, when added to it, equals 180° -/
def supplement (α : Real) : Real := 180 - α

theorem supplement_congruence (α β : Real) :
  congruent_angles (supplement α) (supplement β) → congruent_angles α β := by
  sorry

end NUMINAMATH_CALUDE_supplement_congruence_l1409_140961


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l1409_140904

theorem geometric_sequence_first_term (a : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ a * r = 2 * a ∧ (2 * a) * r = 8) →
  a = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_first_term_l1409_140904


namespace NUMINAMATH_CALUDE_range_of_a_f_lower_bound_l1409_140982

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a - 1| + |x - 2*a|

-- Theorem 1: Range of a when f(1) < 3
theorem range_of_a (a : ℝ) : f 1 a < 3 → a ∈ Set.Ioo (-2/3) (4/3) :=
sorry

-- Theorem 2: f(x) ≥ 2 when a ≥ 1 and x ∈ ℝ
theorem f_lower_bound (a x : ℝ) : a ≥ 1 → f x a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_f_lower_bound_l1409_140982
