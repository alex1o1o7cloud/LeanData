import Mathlib

namespace NUMINAMATH_CALUDE_ellipse_focus_distance_l2724_272444

/-- An ellipse with equation x²/25 + y²/9 = 1 -/
structure Ellipse :=
  (x y : ℝ)
  (eq : x^2/25 + y^2/9 = 1)

/-- The distance from a point to a focus of the ellipse -/
def distance_to_focus (P : Ellipse) (focus : ℝ × ℝ) : ℝ :=
  sorry

theorem ellipse_focus_distance (P : Ellipse) (F1 F2 : ℝ × ℝ) :
  distance_to_focus P F1 = 3 →
  distance_to_focus P F2 = 7 :=
sorry

end NUMINAMATH_CALUDE_ellipse_focus_distance_l2724_272444


namespace NUMINAMATH_CALUDE_baseball_team_score_l2724_272415

theorem baseball_team_score :
  let total_players : ℕ := 9
  let high_scorers : ℕ := 5
  let high_scorer_average : ℕ := 50
  let low_scorer_average : ℕ := 5
  let low_scorers : ℕ := total_players - high_scorers
  let total_score : ℕ := high_scorers * high_scorer_average + low_scorers * low_scorer_average
  total_score = 270 := by sorry

end NUMINAMATH_CALUDE_baseball_team_score_l2724_272415


namespace NUMINAMATH_CALUDE_oxygen_mass_percentage_l2724_272453

/-- Given a compound with a mass percentage of oxygen, prove that the ratio of oxygen mass to total mass equals the mass percentage expressed as a decimal. -/
theorem oxygen_mass_percentage (compound_mass total_mass oxygen_mass : ℝ) 
  (h1 : compound_mass > 0)
  (h2 : total_mass = compound_mass)
  (h3 : oxygen_mass > 0)
  (h4 : oxygen_mass ≤ total_mass)
  (h5 : (oxygen_mass / total_mass) * 100 = 58.33) :
  oxygen_mass / total_mass = 0.5833 := by
sorry

end NUMINAMATH_CALUDE_oxygen_mass_percentage_l2724_272453


namespace NUMINAMATH_CALUDE_sum_of_distances_is_36_root_3_l2724_272420

/-- A regular hexagon with side length 12 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 12)

/-- A point on one side of the hexagon -/
structure PointOnSide (h : RegularHexagon) :=
  (point : ℝ × ℝ)
  (on_side : point.1 ≥ 0 ∧ point.1 ≤ h.side_length)

/-- The sum of distances from a point on one side to lines containing other sides -/
def sum_of_distances (h : RegularHexagon) (p : PointOnSide h) : ℝ :=
  sorry

/-- Theorem stating the sum of distances is 36√3 -/
theorem sum_of_distances_is_36_root_3 (h : RegularHexagon) (p : PointOnSide h) :
  sum_of_distances h p = 36 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_distances_is_36_root_3_l2724_272420


namespace NUMINAMATH_CALUDE_upstream_speed_calculation_l2724_272412

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  stillWater : ℝ
  downstream : ℝ

/-- Calculates the speed of the man rowing upstream -/
def upstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.downstream

/-- Theorem stating that given the man's speed in still water and downstream,
    his upstream speed is 20 kmph -/
theorem upstream_speed_calculation (s : RowingSpeed)
  (h1 : s.stillWater = 24)
  (h2 : s.downstream = 28) :
  upstreamSpeed s = 20 := by
sorry

#eval upstreamSpeed { stillWater := 24, downstream := 28 }

end NUMINAMATH_CALUDE_upstream_speed_calculation_l2724_272412


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2724_272425

theorem complex_fraction_simplification (x y : ℚ) 
  (hx : x = 3) 
  (hy : y = 4) : 
  (1 / (y + 1)) / (1 / (x - 1)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2724_272425


namespace NUMINAMATH_CALUDE_event_attendees_l2724_272480

/-- Represents the number of men at the event -/
def num_men : ℕ := 15

/-- Represents the number of women each man danced with -/
def dances_per_man : ℕ := 4

/-- Represents the number of men each woman danced with -/
def dances_per_woman : ℕ := 3

/-- Calculates the number of women at the event -/
def num_women : ℕ := (num_men * dances_per_man) / dances_per_woman

theorem event_attendees :
  num_women = 20 := by
  sorry

end NUMINAMATH_CALUDE_event_attendees_l2724_272480


namespace NUMINAMATH_CALUDE_unique_solution_l2724_272434

/-- Represents a 3x3 grid with some fixed numbers and variables A, B, C, D --/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if two cells are adjacent in the grid --/
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ (j.val + 1 = l.val ∨ l.val + 1 = j.val)) ∨
  (j = l ∧ (i.val + 1 = k.val ∨ k.val + 1 = i.val))

/-- The sum of any two adjacent numbers is less than 12 --/
def valid_sum (g : Grid) : Prop :=
  ∀ i j k l, adjacent i j k l → g i j + g k l < 12

/-- The grid contains all numbers from 1 to 9 --/
def contains_all_numbers (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → ∃ i j, g i j = n

/-- The given arrangement of known numbers in the grid --/
def given_arrangement (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 2 = 9 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7

/-- The theorem stating the unique solution for A, B, C, D --/
theorem unique_solution (g : Grid) 
  (h1 : valid_sum g) 
  (h2 : contains_all_numbers g) 
  (h3 : given_arrangement g) :
  g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by sorry

end NUMINAMATH_CALUDE_unique_solution_l2724_272434


namespace NUMINAMATH_CALUDE_compound_proposition_truth_l2724_272471

theorem compound_proposition_truth (p q : Prop) 
  (h : ¬(¬p ∨ ¬q)) : (p ∧ q) ∧ (p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_compound_proposition_truth_l2724_272471


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2724_272452

theorem sqrt_product_equality : Real.sqrt (4 / 75) * Real.sqrt 3 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2724_272452


namespace NUMINAMATH_CALUDE_sin_product_equality_l2724_272493

theorem sin_product_equality : 
  Real.sin (3 * π / 180) * Real.sin (39 * π / 180) * Real.sin (63 * π / 180) * Real.sin (75 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_equality_l2724_272493


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l2724_272438

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) → m = 49 / 12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l2724_272438


namespace NUMINAMATH_CALUDE_axis_of_symmetry_compare_points_range_of_t_max_t_value_l2724_272463

-- Define the parabola
def parabola (t x y : ℝ) : Prop := y = x^2 - 2*t*x + 1

-- Theorem 1: Axis of symmetry
theorem axis_of_symmetry (t : ℝ) :
  ∀ x y : ℝ, parabola t x y → (∀ ε > 0, ∃ y₁ y₂ : ℝ, 
    parabola t (t - ε) y₁ ∧ parabola t (t + ε) y₂ ∧ y₁ = y₂) :=
sorry

-- Theorem 2: Comparing points
theorem compare_points (t m n : ℝ) :
  parabola t (t-2) m → parabola t (t+3) n → n > m :=
sorry

-- Theorem 3: Range of t
theorem range_of_t (t : ℝ) :
  (∀ x₁ y₁ x₂ y₂ : ℝ, -1 ≤ x₁ → x₁ < 3 → x₂ = 3 →
    parabola t x₁ y₁ → parabola t x₂ y₂ → y₁ ≤ y₂) → t ≤ 1 :=
sorry

-- Theorem 4: Maximum value of t
theorem max_t_value :
  ∃ t_max : ℝ, t_max = 5 ∧
  ∀ t y₁ y₂ : ℝ, parabola t (t+1) y₁ → parabola t (2*t-4) y₂ → y₁ ≥ y₂ → t ≤ t_max :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_compare_points_range_of_t_max_t_value_l2724_272463


namespace NUMINAMATH_CALUDE_people_to_lift_car_l2724_272476

theorem people_to_lift_car : ℕ :=
  let people_for_car : ℕ := sorry
  let people_for_truck : ℕ := 2 * people_for_car
  have h1 : 6 * people_for_car + 3 * people_for_truck = 60 := by sorry
  have h2 : people_for_car = 5 := by sorry
  5

#check people_to_lift_car

end NUMINAMATH_CALUDE_people_to_lift_car_l2724_272476


namespace NUMINAMATH_CALUDE_daves_initial_apps_daves_initial_apps_proof_l2724_272488

theorem daves_initial_apps : ℕ :=
  let initial_files : ℕ := 9
  let final_files : ℕ := 5
  let final_apps : ℕ := 12
  let app_file_difference : ℕ := 7

  have h1 : final_apps = final_files + app_file_difference := by sorry
  have h2 : ∃ (initial_apps : ℕ), initial_apps - final_apps = initial_files - final_files := by sorry

  16

theorem daves_initial_apps_proof : daves_initial_apps = 16 := by sorry

end NUMINAMATH_CALUDE_daves_initial_apps_daves_initial_apps_proof_l2724_272488


namespace NUMINAMATH_CALUDE_nancy_crayon_packs_l2724_272429

theorem nancy_crayon_packs (total_crayons : ℕ) (crayons_per_pack : ℕ) (h1 : total_crayons = 615) (h2 : crayons_per_pack = 15) :
  total_crayons / crayons_per_pack = 41 := by
  sorry

end NUMINAMATH_CALUDE_nancy_crayon_packs_l2724_272429


namespace NUMINAMATH_CALUDE_expand_and_simplify_polynomial_l2724_272419

theorem expand_and_simplify_polynomial (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_polynomial_l2724_272419


namespace NUMINAMATH_CALUDE_investment_income_is_660_l2724_272487

/-- Calculates the total annual income from an investment split between a savings account and bonds. -/
def totalAnnualIncome (totalInvestment : ℝ) (savingsAmount : ℝ) (savingsRate : ℝ) (bondRate : ℝ) : ℝ :=
  let bondAmount := totalInvestment - savingsAmount
  savingsAmount * savingsRate + bondAmount * bondRate

/-- Proves that the total annual income from the given investment scenario is $660. -/
theorem investment_income_is_660 :
  totalAnnualIncome 10000 6000 0.05 0.09 = 660 := by
  sorry

#eval totalAnnualIncome 10000 6000 0.05 0.09

end NUMINAMATH_CALUDE_investment_income_is_660_l2724_272487


namespace NUMINAMATH_CALUDE_min_value_expression_l2724_272409

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 1 → 4 * x^2 + y^2 + 1 / (x * y) ≥ 17 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2724_272409


namespace NUMINAMATH_CALUDE_defective_product_arrangements_l2724_272418

theorem defective_product_arrangements :
  let total_products : ℕ := 7
  let defective_products : ℕ := 4
  let non_defective_products : ℕ := 3
  let third_defective_position : ℕ := 4

  (Nat.choose non_defective_products 1) *
  (Nat.choose defective_products 1) *
  (Nat.choose (defective_products - 1) 1) *
  1 *
  (Nat.choose 2 1) *
  ((total_products - third_defective_position) - (defective_products - 3)) = 288 :=
by sorry

end NUMINAMATH_CALUDE_defective_product_arrangements_l2724_272418


namespace NUMINAMATH_CALUDE_movie_ticket_distribution_l2724_272490

theorem movie_ticket_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (n.descFactorial k) = 720 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_distribution_l2724_272490


namespace NUMINAMATH_CALUDE_interest_rate_difference_l2724_272454

theorem interest_rate_difference 
  (principal : ℝ) 
  (time : ℝ) 
  (interest_diff : ℝ) 
  (rate1 : ℝ) 
  (rate2 : ℝ) :
  principal = 6000 →
  time = 3 →
  interest_diff = 360 →
  (principal * rate2 * time) / 100 = (principal * rate1 * time) / 100 + interest_diff →
  rate2 - rate1 = 2 := by
sorry

end NUMINAMATH_CALUDE_interest_rate_difference_l2724_272454


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l2724_272411

theorem quadratic_form_ratio (k : ℝ) :
  ∃ (c r s : ℝ), 10 * k^2 - 6 * k + 20 = c * (k + r)^2 + s ∧ s / r = -191 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l2724_272411


namespace NUMINAMATH_CALUDE_cosine_inequality_l2724_272470

theorem cosine_inequality (x y : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 (Real.pi / 2) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.cos (x + y) ≥ Real.cos x * Real.cos y) ↔
  y = 0 := by
sorry

end NUMINAMATH_CALUDE_cosine_inequality_l2724_272470


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_times_three_l2724_272439

theorem arithmetic_sequence_sum_times_three : 
  let a := 75  -- first term
  let d := 2   -- common difference
  let n := 5   -- number of terms
  3 * (a + (a + d) + (a + 2*d) + (a + 3*d) + (a + 4*d)) = 1185 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_times_three_l2724_272439


namespace NUMINAMATH_CALUDE_max_value_expression_l2724_272437

def S : Set ℕ := {0, 1, 2, 3}

theorem max_value_expression (a b c d : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) (hd : d ∈ S) :
  (∀ x y z w : ℕ, x ∈ S → y ∈ S → z ∈ S → w ∈ S →
    z * (x^y + 1) - w ≤ c * (a^b + 1) - d) →
  c * (a^b + 1) - d = 30 :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l2724_272437


namespace NUMINAMATH_CALUDE_range_of_a_l2724_272401

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2724_272401


namespace NUMINAMATH_CALUDE_student_d_score_l2724_272496

/-- Represents a student's answers and score -/
structure StudentAnswers :=
  (answers : List Bool)
  (score : Nat)

/-- The problem setup -/
def mathTestProblem :=
  let numQuestions : Nat := 8
  let pointsPerQuestion : Nat := 5
  let totalPossibleScore : Nat := 40
  let studentA : StudentAnswers := ⟨[false, true, false, true, false, false, true, false], 30⟩
  let studentB : StudentAnswers := ⟨[false, false, true, true, true, false, false, true], 25⟩
  let studentC : StudentAnswers := ⟨[true, false, false, false, true, true, true, false], 25⟩
  let studentD : StudentAnswers := ⟨[false, true, false, true, true, false, true, true], 0⟩ -- score unknown
  (numQuestions, pointsPerQuestion, totalPossibleScore, studentA, studentB, studentC, studentD)

/-- The theorem to prove -/
theorem student_d_score :
  let (numQuestions, pointsPerQuestion, totalPossibleScore, studentA, studentB, studentC, studentD) := mathTestProblem
  studentD.score = 30 := by
  sorry


end NUMINAMATH_CALUDE_student_d_score_l2724_272496


namespace NUMINAMATH_CALUDE_rectangle_perimeter_problem_l2724_272430

/-- The perimeter of a rectangle given its width and height -/
def perimeter (width : ℝ) (height : ℝ) : ℝ := 2 * (width + height)

/-- The theorem statement -/
theorem rectangle_perimeter_problem :
  ∀ (x y : ℝ),
  (perimeter (6 * x) y = 56) →
  (perimeter (4 * x) (3 * y) = 56) →
  (perimeter (2 * x) (3 * y) = 40) :=
by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_problem_l2724_272430


namespace NUMINAMATH_CALUDE_money_distribution_l2724_272481

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 500)
  (AC : A + C = 200)
  (BC : B + C = 360) :
  C = 60 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l2724_272481


namespace NUMINAMATH_CALUDE_neil_initial_games_l2724_272483

theorem neil_initial_games (henry_initial : ℕ) (henry_gave : ℕ) (henry_neil_ratio : ℕ) :
  henry_initial = 33 →
  henry_gave = 5 →
  henry_neil_ratio = 4 →
  henry_initial - henry_gave = henry_neil_ratio * (2 + henry_gave) :=
by
  sorry

end NUMINAMATH_CALUDE_neil_initial_games_l2724_272483


namespace NUMINAMATH_CALUDE_mod_thirteen_four_eleven_l2724_272461

theorem mod_thirteen_four_eleven (m : ℕ) : 
  13^4 % 11 = m ∧ 0 ≤ m ∧ m < 11 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_mod_thirteen_four_eleven_l2724_272461


namespace NUMINAMATH_CALUDE_one_student_passes_probability_l2724_272464

/-- The probability that exactly one out of three students passes, given their individual passing probabilities -/
theorem one_student_passes_probability
  (p_jia p_yi p_bing : ℚ)
  (h_jia : p_jia = 4 / 5)
  (h_yi : p_yi = 3 / 5)
  (h_bing : p_bing = 7 / 10) :
  (p_jia * (1 - p_yi) * (1 - p_bing)) +
  ((1 - p_jia) * p_yi * (1 - p_bing)) +
  ((1 - p_jia) * (1 - p_yi) * p_bing) =
  47 / 250 := by
  sorry

end NUMINAMATH_CALUDE_one_student_passes_probability_l2724_272464


namespace NUMINAMATH_CALUDE_farmer_purchase_problem_l2724_272491

theorem farmer_purchase_problem :
  ∃ (p ch : ℕ), 
    p > 0 ∧ 
    ch > 0 ∧ 
    30 * p + 24 * ch = 1200 ∧ 
    p = 4 ∧ 
    ch = 45 := by
  sorry

end NUMINAMATH_CALUDE_farmer_purchase_problem_l2724_272491


namespace NUMINAMATH_CALUDE_cube_root_simplification_l2724_272482

theorem cube_root_simplification :
  (2^9 * 3^3 * 5^3 * 11^3 : ℝ)^(1/3) = 1320 := by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l2724_272482


namespace NUMINAMATH_CALUDE_lollipop_sequence_l2724_272440

theorem lollipop_sequence (a b c d e : ℕ) : 
  a + b + c + d + e = 100 →
  b = a + 6 →
  c = b + 6 →
  d = c + 6 →
  e = d + 6 →
  c = 20 := by sorry

end NUMINAMATH_CALUDE_lollipop_sequence_l2724_272440


namespace NUMINAMATH_CALUDE_fraction_equality_l2724_272459

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + 2 * y) / (2 * x - 8 * y) = 3) : 
  (x + 4 * y) / (4 * x - y) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l2724_272459


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2724_272485

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a - 1)*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2724_272485


namespace NUMINAMATH_CALUDE_complex_magnitude_l2724_272469

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = 1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l2724_272469


namespace NUMINAMATH_CALUDE_aero_tees_count_l2724_272489

/-- The number of people golfing -/
def num_people : ℕ := 4

/-- The number of tees in a package of generic tees -/
def generic_package_size : ℕ := 12

/-- The maximum number of generic packages Bill will buy -/
def max_generic_packages : ℕ := 2

/-- The minimum number of tees needed per person -/
def min_tees_per_person : ℕ := 20

/-- The number of aero flight tee packages Bill must purchase -/
def aero_packages : ℕ := 28

/-- The number of aero flight tees in one package -/
def aero_tees_per_package : ℕ := 2

theorem aero_tees_count : 
  num_people * min_tees_per_person ≤ 
  max_generic_packages * generic_package_size + 
  aero_packages * aero_tees_per_package ∧
  num_people * min_tees_per_person > 
  max_generic_packages * generic_package_size + 
  aero_packages * (aero_tees_per_package - 1) :=
by sorry

end NUMINAMATH_CALUDE_aero_tees_count_l2724_272489


namespace NUMINAMATH_CALUDE_tangent_sum_l2724_272477

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_tangent_sum_l2724_272477


namespace NUMINAMATH_CALUDE_product_three_consecutive_integers_div_by_6_l2724_272467

theorem product_three_consecutive_integers_div_by_6 (n : ℤ) :
  ∃ k : ℤ, n * (n + 1) * (n + 2) = 6 * k := by sorry

end NUMINAMATH_CALUDE_product_three_consecutive_integers_div_by_6_l2724_272467


namespace NUMINAMATH_CALUDE_monomial_degree_5_l2724_272400

/-- The degree of a monomial of the form 3a^2b^n -/
def monomialDegree (n : ℕ) : ℕ := 2 + n

theorem monomial_degree_5 (n : ℕ) : monomialDegree n = 5 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_monomial_degree_5_l2724_272400


namespace NUMINAMATH_CALUDE_num_cows_is_24_l2724_272479

/-- Represents the number of ducks in the group -/
def num_ducks : ℕ := sorry

/-- Represents the number of cows in the group -/
def num_cows : ℕ := sorry

/-- The total number of legs in the group -/
def total_legs : ℕ := 2 * num_ducks + 4 * num_cows

/-- The total number of heads in the group -/
def total_heads : ℕ := num_ducks + num_cows

/-- Theorem stating that the number of cows is 24 given the conditions -/
theorem num_cows_is_24 : 
  (total_legs = 2 * total_heads + 48) → num_cows = 24 := by
  sorry

end NUMINAMATH_CALUDE_num_cows_is_24_l2724_272479


namespace NUMINAMATH_CALUDE_snow_probability_l2724_272447

/-- The probability of no snow on each of the first five days -/
def no_snow_prob (n : ℕ) : ℚ :=
  if n ≤ 5 then (n + 1) / (n + 2) else 7/8

/-- The probability of snow on at least one day out of seven -/
def snow_prob : ℚ :=
  1 - (no_snow_prob 1 * no_snow_prob 2 * no_snow_prob 3 * no_snow_prob 4 * no_snow_prob 5 * no_snow_prob 6 * no_snow_prob 7)

theorem snow_probability : snow_prob = 139/384 := by
  sorry

end NUMINAMATH_CALUDE_snow_probability_l2724_272447


namespace NUMINAMATH_CALUDE_vector_sum_parallel_l2724_272446

def a : ℝ × ℝ := (-1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, 1)

theorem vector_sum_parallel (m : ℝ) : 
  (∃ k : ℝ, k ≠ 0 ∧ (a + 2 • b m) = k • (2 • a - b m)) → 
  (a + b m) = (-3/2, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_parallel_l2724_272446


namespace NUMINAMATH_CALUDE_distinct_reciprocals_inequality_l2724_272478

theorem distinct_reciprocals_inequality (a b c : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c)
  (h_sum : 2 * b = a + c) : 
  2 / b ≠ 1 / a + 1 / c := by
sorry

end NUMINAMATH_CALUDE_distinct_reciprocals_inequality_l2724_272478


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l2724_272417

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, -1, 2}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l2724_272417


namespace NUMINAMATH_CALUDE_g_512_minus_g_256_l2724_272494

def sumOfDivisors (n : ℕ+) : ℕ := sorry

def f (n : ℕ+) : ℚ := (sumOfDivisors n : ℚ) / n

def g (n : ℕ+) : ℚ := f n + 1 / n

theorem g_512_minus_g_256 : g 512 - g 256 = 0 := by sorry

end NUMINAMATH_CALUDE_g_512_minus_g_256_l2724_272494


namespace NUMINAMATH_CALUDE_toy_store_revenue_ratio_l2724_272431

theorem toy_store_revenue_ratio : 
  ∀ (N D J : ℝ),
  J = (1 / 2) * N →
  D = (10 / 3) * ((N + J) / 2) →
  N / D = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_ratio_l2724_272431


namespace NUMINAMATH_CALUDE_cos_160_sin_10_minus_sin_20_cos_10_l2724_272427

theorem cos_160_sin_10_minus_sin_20_cos_10 :
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) -
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_160_sin_10_minus_sin_20_cos_10_l2724_272427


namespace NUMINAMATH_CALUDE_total_worth_of_toys_l2724_272410

theorem total_worth_of_toys (total_toys : Nat) (special_toy_value : Nat) (regular_toy_value : Nat) :
  total_toys = 9 →
  special_toy_value = 12 →
  regular_toy_value = 5 →
  (total_toys - 1) * regular_toy_value + special_toy_value = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_worth_of_toys_l2724_272410


namespace NUMINAMATH_CALUDE_max_a_fourth_quadrant_l2724_272406

theorem max_a_fourth_quadrant (a : ℤ) : 
  let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  (0 < z.re ∧ z.im < 0) → a ≤ 3 ∧ ∃ (b : ℤ), b ≤ 3 ∧ 
    let w : ℂ := (2 + b * Complex.I) / (1 + 2 * Complex.I)
    0 < w.re ∧ w.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_max_a_fourth_quadrant_l2724_272406


namespace NUMINAMATH_CALUDE_perimeter_ABCDEHG_l2724_272451

-- Define the points
variable (A B C D E F G H : ℝ × ℝ)

-- Define the conditions
def is_equilateral (X Y Z : ℝ × ℝ) : Prop := sorry
def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry
def distance (X Y : ℝ × ℝ) : ℝ := sorry

-- State the theorem
theorem perimeter_ABCDEHG :
  is_equilateral A B C →
  is_equilateral A D E →
  is_equilateral E F G →
  is_midpoint D A C →
  is_midpoint H A E →
  distance A B = 6 →
  distance A B + distance B C + distance C D + distance D E +
  distance E F + distance F G + distance G H + distance H A = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_ABCDEHG_l2724_272451


namespace NUMINAMATH_CALUDE_exists_valid_superchess_configuration_l2724_272422

/-- Represents a chess piece in the game of superchess -/
structure Piece where
  id : Fin 20

/-- Represents a position on the superchess board -/
structure Position where
  x : Fin 100
  y : Fin 100

/-- Represents the superchess board -/
def Board := Fin 100 → Fin 100 → Option Piece

/-- Predicate to check if a piece attacks a position -/
def attacks (p : Piece) (pos : Position) (board : Board) : Prop :=
  ∃ (attacked : Finset Position), attacked.card ≤ 20 ∧ pos ∈ attacked

/-- Predicate to check if a board configuration is valid (no piece attacks another) -/
def valid_board (board : Board) : Prop :=
  ∀ (p₁ p₂ : Piece) (pos₁ pos₂ : Position),
    board pos₁.x pos₁.y = some p₁ →
    board pos₂.x pos₂.y = some p₂ →
    p₁ ≠ p₂ →
    ¬(attacks p₁ pos₂ board ∨ attacks p₂ pos₁ board)

/-- Theorem stating that there exists a valid board configuration -/
theorem exists_valid_superchess_configuration :
  ∃ (board : Board), (∀ p : Piece, ∃ pos : Position, board pos.x pos.y = some p) ∧ valid_board board :=
sorry

end NUMINAMATH_CALUDE_exists_valid_superchess_configuration_l2724_272422


namespace NUMINAMATH_CALUDE_correct_system_of_equations_l2724_272448

/-- Represents the number of students in a grade -/
def total_students : ℕ := 246

/-- Theorem: The system of equations {x + y = 246, y = 2x + 2} correctly represents
    the scenario where the total number of students is 246, and the number of boys (y)
    is 2 more than twice the number of girls (x). -/
theorem correct_system_of_equations (x y : ℕ) :
  x + y = total_students ∧ y = 2 * x + 2 →
  x + y = total_students ∧ y = 2 * x + 2 :=
by sorry

end NUMINAMATH_CALUDE_correct_system_of_equations_l2724_272448


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2724_272402

theorem quadratic_roots_sum_and_product (m n : ℝ) : 
  (m^2 - 4*m = 12) → (n^2 - 4*n = 12) → m + n + m*n = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_and_product_l2724_272402


namespace NUMINAMATH_CALUDE_line_equation_problem_l2724_272403

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 3)^2 + (y - 4)^2 = 16

-- Define the unit circle
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the line of symmetry
def symmetry_line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 1)

-- Define a line
def line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

-- Define the tangency condition
def is_tangent (m b : ℝ) : Prop := ∃ x y : ℝ, unit_circle x y ∧ line m b x y

-- State the theorem
theorem line_equation_problem (M N : ℝ × ℝ) (k : ℝ) :
  (∃ k, symmetry_line k M.1 M.2 ∧ symmetry_line k N.1 N.2) →
  circle_C M.1 M.2 →
  circle_C N.1 N.2 →
  (∃ m b, is_tangent m b ∧ line m b M.1 M.2 ∧ line m b N.1 N.2) →
  ∃ m b, m = 1 ∧ b = 2 ∧ ∀ x y, line m b x y ↔ y = x + 2 :=
sorry

end NUMINAMATH_CALUDE_line_equation_problem_l2724_272403


namespace NUMINAMATH_CALUDE_train_acceleration_equation_l2724_272474

theorem train_acceleration_equation 
  (v : ℝ) (s : ℝ) (x : ℝ) 
  (h1 : v > 0) 
  (h2 : s > 0) 
  (h3 : x > v) :
  s / (x - v) = (s + 50) / x :=
by sorry

end NUMINAMATH_CALUDE_train_acceleration_equation_l2724_272474


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l2724_272428

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- State that f is differentiable
variable (hf : Differentiable ℝ f)

-- Define the limit condition
variable (h_limit : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
  |((f 1 - f (1 + 2 * Δx)) / Δx) - 2| < ε)

-- Theorem statement
theorem tangent_slope_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_limit : ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f 1 - f (1 + 2 * Δx)) / Δx) - 2| < ε) : 
  deriv f 1 = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l2724_272428


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2724_272457

theorem x_plus_y_value (x y : ℝ) (h1 : 1/x = 2) (h2 : 1/x + 3/y = 3) : x + y = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2724_272457


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l2724_272468

/-- Given a line segment AB of length 3r with point C on AB such that AC = r and CB = 2r,
    and semi-circles constructed on AB, AC, and CB, prove that the ratio of the shaded area
    to the area of a circle with radius equal to the radius of the semi-circle on CB is 2:1. -/
theorem shaded_area_ratio (r : ℝ) (h : r > 0) : 
  let total_area := π * (3 * r)^2 / 2
  let small_semicircle_area := π * r^2 / 2
  let medium_semicircle_area := π * (2 * r)^2 / 2
  let shaded_area := total_area - (small_semicircle_area + medium_semicircle_area)
  let circle_area := π * r^2
  shaded_area / circle_area = 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l2724_272468


namespace NUMINAMATH_CALUDE_jungkook_item_sum_l2724_272472

theorem jungkook_item_sum : ∀ (a b : ℕ),
  a = 585 →
  a = b + 249 →
  a + b = 921 :=
by
  sorry

end NUMINAMATH_CALUDE_jungkook_item_sum_l2724_272472


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2724_272435

def a : Fin 2 → ℝ := ![1, 2]
def b (y : ℝ) : Fin 2 → ℝ := ![-2, y]

theorem parallel_vectors_magnitude (y : ℝ) 
  (h : a 0 * (b y 1) = a 1 * (b y 0)) : 
  Real.sqrt ((3 * a 0 + b y 0)^2 + (3 * a 1 + b y 1)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l2724_272435


namespace NUMINAMATH_CALUDE_number_of_tables_bought_l2724_272498

/-- Proves that the number of tables bought at the cost price is 15, given the conditions -/
theorem number_of_tables_bought (C S : ℝ) (N : ℕ) : 
  N * C = 20 * S → -- The cost price of N tables equals the selling price of 20 tables
  S = 0.75 * C →   -- The selling price is 75% of the cost price (due to 25% loss)
  N = 15 :=
by sorry

end NUMINAMATH_CALUDE_number_of_tables_bought_l2724_272498


namespace NUMINAMATH_CALUDE_max_value_of_f_l2724_272432

def S : Finset ℕ := {0, 1, 2, 3, 4}

def f (a b c d e : ℕ) : ℕ := e * c^a + b - d

theorem max_value_of_f :
  ∃ (a b c d e : ℕ),
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    f a b c d e = 39 ∧
    ∀ (a' b' c' d' e' : ℕ),
      a' ∈ S → b' ∈ S → c' ∈ S → d' ∈ S → e' ∈ S →
      a' ≠ b' → a' ≠ c' → a' ≠ d' → a' ≠ e' →
      b' ≠ c' → b' ≠ d' → b' ≠ e' →
      c' ≠ d' → c' ≠ e' →
      d' ≠ e' →
      f a' b' c' d' e' ≤ 39 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2724_272432


namespace NUMINAMATH_CALUDE_power_negative_product_l2724_272462

theorem power_negative_product (a : ℝ) : (-a)^3 * (-a)^5 = a^8 := by
  sorry

end NUMINAMATH_CALUDE_power_negative_product_l2724_272462


namespace NUMINAMATH_CALUDE_min_colours_for_cube_l2724_272407

/-- Represents a colouring of a cube's faces -/
def CubeColouring := Fin 6 → ℕ

/-- Checks if two face indices are adjacent on a cube -/
def are_adjacent (i j : Fin 6) : Prop :=
  (i.val + j.val) % 2 = 1 ∧ i ≠ j

/-- A valid colouring has different colours for adjacent faces -/
def is_valid_colouring (c : CubeColouring) : Prop :=
  ∀ i j : Fin 6, are_adjacent i j → c i ≠ c j

/-- The number of colours used in a colouring -/
def num_colours (c : CubeColouring) : ℕ :=
  Finset.card (Finset.image c Finset.univ)

/-- There exists a valid 3-colouring of a cube -/
axiom exists_valid_3_colouring : ∃ c : CubeColouring, is_valid_colouring c ∧ num_colours c = 3

/-- Any valid colouring of a cube uses at least 3 colours -/
axiom valid_colouring_needs_at_least_3 : ∀ c : CubeColouring, is_valid_colouring c → num_colours c ≥ 3

theorem min_colours_for_cube : ∃ n : ℕ, n = 3 ∧
  (∃ c : CubeColouring, is_valid_colouring c ∧ num_colours c = n) ∧
  (∀ c : CubeColouring, is_valid_colouring c → num_colours c ≥ n) :=
sorry

end NUMINAMATH_CALUDE_min_colours_for_cube_l2724_272407


namespace NUMINAMATH_CALUDE_polynomial_multiplication_equality_l2724_272424

-- Define the polynomials
def p (y : ℝ) : ℝ := 2*y - 1
def q (y : ℝ) : ℝ := 5*y^12 - 3*y^11 + y^9 - 4*y^8
def r (y : ℝ) : ℝ := 10*y^13 - 11*y^12 + 3*y^11 + y^10 - 9*y^9 + 4*y^8

-- Theorem statement
theorem polynomial_multiplication_equality :
  ∀ y : ℝ, p y * q y = r y :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_equality_l2724_272424


namespace NUMINAMATH_CALUDE_a_explicit_form_l2724_272460

def a : ℕ → ℤ
  | 0 => -1
  | 1 => 1
  | n + 2 => 2 * a (n + 1) + 3 * a n + 3^(n + 2)

theorem a_explicit_form (n : ℕ) :
  a n = (1 / 16) * ((4 * n - 3) * 3^(n + 1) - 7 * (-1)^n) :=
by sorry

end NUMINAMATH_CALUDE_a_explicit_form_l2724_272460


namespace NUMINAMATH_CALUDE_num_large_beds_is_two_l2724_272441

/-- The number of seeds that can be planted in a large bed -/
def large_bed_capacity : ℕ := 100

/-- The number of seeds that can be planted in a medium bed -/
def medium_bed_capacity : ℕ := 60

/-- The number of medium beds -/
def num_medium_beds : ℕ := 2

/-- The total number of seeds that can be planted -/
def total_seeds : ℕ := 320

/-- Theorem stating that the number of large beds is 2 -/
theorem num_large_beds_is_two :
  ∃ (n : ℕ), n * large_bed_capacity + num_medium_beds * medium_bed_capacity = total_seeds ∧ n = 2 :=
sorry

end NUMINAMATH_CALUDE_num_large_beds_is_two_l2724_272441


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_implies_a_range_l2724_272484

/-- A function f is monotonic on an interval [a, b] if it is either
    nondecreasing or nonincreasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

theorem quadratic_monotonicity_implies_a_range (a : ℝ) :
  IsMonotonic (fun x => x^2 - 2*a*x - 3) 1 2 → a ≤ 1 ∨ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_implies_a_range_l2724_272484


namespace NUMINAMATH_CALUDE_runner_time_difference_l2724_272436

theorem runner_time_difference (total_distance : ℝ) (half_distance : ℝ) (second_half_time : ℝ) :
  total_distance = 40 →
  half_distance = total_distance / 2 →
  second_half_time = 10 →
  ∃ (initial_speed : ℝ),
    initial_speed > 0 ∧
    half_distance / initial_speed + half_distance / (initial_speed / 2) = second_half_time + half_distance / initial_speed ∧
    second_half_time - half_distance / initial_speed = 5 :=
by sorry

end NUMINAMATH_CALUDE_runner_time_difference_l2724_272436


namespace NUMINAMATH_CALUDE_simplify_expression_l2724_272445

theorem simplify_expression (x : ℝ) : 3*x + 4 - x + 8 = 2*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2724_272445


namespace NUMINAMATH_CALUDE_josh_initial_money_l2724_272449

/-- The cost of the candy bar in dollars -/
def candy_cost : ℚ := 45 / 100

/-- The change Josh received in dollars -/
def change_received : ℚ := 135 / 100

/-- The initial amount of money Josh had -/
def initial_money : ℚ := candy_cost + change_received

theorem josh_initial_money : initial_money = 180 / 100 := by
  sorry

end NUMINAMATH_CALUDE_josh_initial_money_l2724_272449


namespace NUMINAMATH_CALUDE_hardcover_book_weight_l2724_272433

/-- Proves that the weight of each hardcover book is 1/2 pound given the problem conditions -/
theorem hardcover_book_weight :
  let bookcase_limit : ℚ := 80
  let hardcover_count : ℕ := 70
  let textbook_count : ℕ := 30
  let textbook_weight : ℚ := 2
  let knickknack_count : ℕ := 3
  let knickknack_weight : ℚ := 6
  let overweight : ℚ := 33
  let hardcover_weight : ℚ := 1/2

  hardcover_count * hardcover_weight + 
  textbook_count * textbook_weight + 
  knickknack_count * knickknack_weight = 
  bookcase_limit + overweight :=
by
  sorry

#check hardcover_book_weight

end NUMINAMATH_CALUDE_hardcover_book_weight_l2724_272433


namespace NUMINAMATH_CALUDE_baking_time_per_batch_l2724_272443

/-- Proves that the time to bake one batch of cupcakes is 20 minutes -/
theorem baking_time_per_batch (
  num_batches : ℕ)
  (icing_time_per_batch : ℕ)
  (total_time : ℕ)
  (h1 : num_batches = 4)
  (h2 : icing_time_per_batch = 30)
  (h3 : total_time = 200)
  : ∃ (baking_time_per_batch : ℕ),
    baking_time_per_batch * num_batches + icing_time_per_batch * num_batches = total_time ∧
    baking_time_per_batch = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_baking_time_per_batch_l2724_272443


namespace NUMINAMATH_CALUDE_nine_digit_increasing_integers_mod_1000_l2724_272499

/-- The number of ways to select 9 items from 10 items with replacement and order matters -/
def M : ℕ := Nat.choose 18 9

/-- The theorem to prove -/
theorem nine_digit_increasing_integers_mod_1000 :
  M % 1000 = 620 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_increasing_integers_mod_1000_l2724_272499


namespace NUMINAMATH_CALUDE_last_period_production_theorem_l2724_272414

/-- Represents the TV production scenario in a factory --/
structure TVProduction where
  total_days : ℕ
  first_period_days : ℕ
  first_period_avg : ℕ
  monthly_avg : ℕ

/-- Calculates the average daily production for the last period --/
def last_period_avg (prod : TVProduction) : ℚ :=
  let total_production := prod.total_days * prod.monthly_avg
  let first_period_production := prod.first_period_days * prod.first_period_avg
  let last_period_days := prod.total_days - prod.first_period_days
  (total_production - first_period_production) / last_period_days

/-- Theorem stating the average production for the last 5 days --/
theorem last_period_production_theorem (prod : TVProduction) 
  (h1 : prod.total_days = 30)
  (h2 : prod.first_period_days = 25)
  (h3 : prod.first_period_avg = 63)
  (h4 : prod.monthly_avg = 58) :
  last_period_avg prod = 33 := by
  sorry

end NUMINAMATH_CALUDE_last_period_production_theorem_l2724_272414


namespace NUMINAMATH_CALUDE_max_profit_allocation_l2724_272492

/-- Represents the profit function for Project A -/
def p (a : ℝ) (t : ℝ) : ℝ := a * t^3 + 21 * t

/-- Represents the profit function for Project B -/
def g (a : ℝ) (b : ℝ) (t : ℝ) : ℝ := -2 * a * (t - b)^2

/-- Represents the total profit function -/
def f (a : ℝ) (b : ℝ) (x : ℝ) : ℝ := p a x + g a b (200 - x)

/-- Theorem stating the maximum profit and optimal investment allocation -/
theorem max_profit_allocation (a b : ℝ) :
  (∀ t, p a t = -1/60 * t^3 + 21 * t) →
  (∀ t, g a b t = 1/30 * (t - 110)^2) →
  (p a 30 = 180) →
  (g a b 170 = 120) →
  (b < 200) →
  (∃ x₀, x₀ ∈ Set.Icc 10 190 ∧ 
    f a b x₀ = 453.6 ∧
    (∀ x, x ∈ Set.Icc 10 190 → f a b x ≤ f a b x₀) ∧
    x₀ = 18) := by
  sorry


end NUMINAMATH_CALUDE_max_profit_allocation_l2724_272492


namespace NUMINAMATH_CALUDE_sequence_equation_l2724_272458

theorem sequence_equation (n : ℕ+) : 9 * n + (n - 1) = 10 * n - 1 := by
  sorry

#check sequence_equation

end NUMINAMATH_CALUDE_sequence_equation_l2724_272458


namespace NUMINAMATH_CALUDE_correction_is_subtract_30x_l2724_272404

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "half-dollar" => 50
  | "dollar" => 100
  | "quarter" => 25
  | "nickel" => 5
  | _ => 0

/-- Calculates the correction needed for mistaken coin counts -/
def correction_amount (x : ℕ) : ℤ :=
  (coin_value "dollar" - coin_value "half-dollar") * x -
  (coin_value "quarter" - coin_value "nickel") * x

theorem correction_is_subtract_30x (x : ℕ) :
  correction_amount x = -30 * x :=
sorry

end NUMINAMATH_CALUDE_correction_is_subtract_30x_l2724_272404


namespace NUMINAMATH_CALUDE_least_common_period_is_36_l2724_272455

/-- A function satisfying the given condition -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) + f (x - 6) = f x

/-- A function is periodic with period p -/
def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

/-- The least common positive period for all functions satisfying the condition -/
def LeastCommonPeriod : ℝ := 36

/-- Main theorem: The least common positive period for all functions satisfying the condition is 36 -/
theorem least_common_period_is_36 :
  ∀ f : ℝ → ℝ, SatisfiesCondition f →
  (∀ p : ℝ, p > 0 → IsPeriodic f p → p ≥ LeastCommonPeriod) ∧
  (∃ f : ℝ → ℝ, SatisfiesCondition f ∧ IsPeriodic f LeastCommonPeriod) :=
sorry

end NUMINAMATH_CALUDE_least_common_period_is_36_l2724_272455


namespace NUMINAMATH_CALUDE_total_amount_is_175_l2724_272486

/-- Represents the share distribution among x, y, and z -/
structure ShareDistribution where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the total amount from a given share distribution -/
def totalAmount (s : ShareDistribution) : ℝ :=
  s.x + s.y + s.z

/-- Theorem stating that given the conditions, the total amount is 175 -/
theorem total_amount_is_175 :
  ∀ (s : ShareDistribution),
    s.y = 45 →                -- y's share is 45
    s.y = 0.45 * s.x →        -- y gets 45 paisa for each rupee x gets
    s.z = 0.30 * s.x →        -- z gets 30 paisa for each rupee x gets
    totalAmount s = 175 :=
by
  sorry

#check total_amount_is_175

end NUMINAMATH_CALUDE_total_amount_is_175_l2724_272486


namespace NUMINAMATH_CALUDE_phone_number_fraction_l2724_272408

def is_valid_phone_number (n : ℕ) : Prop :=
  1000000 ≤ n ∧ n < 10000000 ∧ n / 1000000 ≥ 3

def starts_with_3_to_9_ends_even (n : ℕ) : Prop :=
  is_valid_phone_number n ∧ n % 2 = 0

def count_valid_numbers : ℕ := 7 * 10^6

def count_start_3_to_9_end_even : ℕ := 7 * 10^5 * 5

theorem phone_number_fraction :
  (count_start_3_to_9_end_even : ℚ) / count_valid_numbers = 1 / 2 :=
sorry

end NUMINAMATH_CALUDE_phone_number_fraction_l2724_272408


namespace NUMINAMATH_CALUDE_function_equality_implies_m_zero_l2724_272405

/-- Given two functions f and g, prove that m = 0 when 3f(3) = 2g(3) -/
theorem function_equality_implies_m_zero (m : ℝ) : 
  let f := fun (x : ℝ) => x^2 - 3*x + 2*m
  let g := fun (x : ℝ) => 2*x^2 - 6*x + 5*m
  3 * f 3 = 2 * g 3 → m = 0 := by
sorry

end NUMINAMATH_CALUDE_function_equality_implies_m_zero_l2724_272405


namespace NUMINAMATH_CALUDE_other_endpoint_coordinates_l2724_272465

/-- Given a line segment with midpoint (3, 1) and one endpoint at (7, -3),
    prove that the other endpoint is at (-1, 5). -/
theorem other_endpoint_coordinates :
  ∀ (x y : ℝ),
  (3 = (7 + x) / 2) →
  (1 = (-3 + y) / 2) →
  x = -1 ∧ y = 5 := by
sorry

end NUMINAMATH_CALUDE_other_endpoint_coordinates_l2724_272465


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l2724_272421

/-- The number of unique arrangements of the letters in "BANANA" -/
def banana_arrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of the letters in "BANANA" is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_count_l2724_272421


namespace NUMINAMATH_CALUDE_expression_equality_l2724_272497

theorem expression_equality : -15 + 9 * (6 / 3) = 3 := by sorry

end NUMINAMATH_CALUDE_expression_equality_l2724_272497


namespace NUMINAMATH_CALUDE_sin_330_degrees_l2724_272475

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l2724_272475


namespace NUMINAMATH_CALUDE_semicircle_problem_l2724_272416

/-- Given a large semicircle with diameter D and N congruent small semicircles
    fitting exactly on its diameter, if the ratio of the combined area of the
    small semicircles to the area of the large semicircle not covered by the
    small semicircles is 1:10, then N = 11. -/
theorem semicircle_problem (D : ℝ) (N : ℕ) (h : N > 0) :
  let r := D / (2 * N)
  let A := N * π * r^2 / 2
  let B := π * (N * r)^2 / 2 - A
  A / B = 1 / 10 → N = 11 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_problem_l2724_272416


namespace NUMINAMATH_CALUDE_efficient_coin_labeling_theorem_l2724_272473

/-- A coin labeling is a list of 8 positive integers representing coin values in cents -/
def CoinLabeling := List Nat

/-- Checks if a given coin labeling is n-efficient -/
def is_n_efficient (labeling : CoinLabeling) (n : Nat) : Prop :=
  (labeling.length = 8) ∧
  (∀ (k : Nat), 1 ≤ k ∧ k ≤ n → ∃ (buyer_coins seller_coins : List Nat),
    buyer_coins ⊆ labeling.take 4 ∧
    seller_coins ⊆ labeling.drop 4 ∧
    buyer_coins.sum - seller_coins.sum = k)

/-- The maximum n for which an n-efficient labeling exists -/
def max_efficient_n : Nat := 240

/-- Theorem stating the existence of a 240-efficient labeling and that it's the maximum -/
theorem efficient_coin_labeling_theorem :
  (∃ (labeling : CoinLabeling), is_n_efficient labeling max_efficient_n) ∧
  (∀ (n : Nat), n > max_efficient_n → ¬∃ (labeling : CoinLabeling), is_n_efficient labeling n) :=
sorry

end NUMINAMATH_CALUDE_efficient_coin_labeling_theorem_l2724_272473


namespace NUMINAMATH_CALUDE_intersection_points_count_l2724_272442

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem intersection_points_count 
  (f : ℝ → ℝ) 
  (h_periodic : is_periodic f 2)
  (h_def : ∀ x ∈ Set.Icc 0 2, f x = x^3 - x) :
  (Set.Icc 0 6 ∩ {x | f x = 0}).ncard = 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_count_l2724_272442


namespace NUMINAMATH_CALUDE_edward_good_games_l2724_272466

def games_from_friend : ℕ := 41
def games_from_garage_sale : ℕ := 14
def non_working_games : ℕ := 31

theorem edward_good_games :
  games_from_friend + games_from_garage_sale - non_working_games = 24 := by
  sorry

end NUMINAMATH_CALUDE_edward_good_games_l2724_272466


namespace NUMINAMATH_CALUDE_sin_cos_equality_solution_l2724_272495

theorem sin_cos_equality_solution :
  ∃ x : ℝ, x * (180 / π) = 9 ∧ Real.sin (4 * x) * Real.sin (6 * x) = Real.cos (4 * x) * Real.cos (6 * x) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equality_solution_l2724_272495


namespace NUMINAMATH_CALUDE_data_set_range_l2724_272413

/-- The range of a data set with maximum value 78 and minimum value 21 is 57. -/
theorem data_set_range : ℝ → ℝ → ℝ → Prop :=
  fun (max min range : ℝ) =>
    max = 78 ∧ min = 21 → range = max - min → range = 57

/-- Proof of the theorem -/
lemma prove_data_set_range : data_set_range 78 21 57 := by
  sorry

end NUMINAMATH_CALUDE_data_set_range_l2724_272413


namespace NUMINAMATH_CALUDE_mariela_get_well_cards_l2724_272456

theorem mariela_get_well_cards (cards_from_home : ℝ) (cards_from_country : ℕ) 
  (h1 : cards_from_home = 287.0) 
  (h2 : cards_from_country = 116) : 
  ↑cards_from_country + cards_from_home = 403 := by
  sorry

end NUMINAMATH_CALUDE_mariela_get_well_cards_l2724_272456


namespace NUMINAMATH_CALUDE_bucket_fill_time_l2724_272423

/-- Given that two-thirds of a bucket is filled in 90 seconds, 
    prove that the time taken to fill the bucket completely is 135 seconds. -/
theorem bucket_fill_time (fill_rate : ℝ) (h : 2/3 * fill_rate = 90) : fill_rate = 135 := by
  sorry

end NUMINAMATH_CALUDE_bucket_fill_time_l2724_272423


namespace NUMINAMATH_CALUDE_equation_solutions_l2724_272450

theorem equation_solutions : 
  (∃ (S₁ S₂ : Set ℝ), 
    S₁ = {x : ℝ | x * (x + 4) = -5 * (x + 4)} ∧ 
    S₂ = {x : ℝ | (x + 2)^2 = (2*x - 1)^2} ∧
    S₁ = {-4, -5} ∧
    S₂ = {3, -1/3}) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2724_272450


namespace NUMINAMATH_CALUDE_max_value_implies_sum_l2724_272426

/-- The function f(x) = x^3 + ax^2 + bx - a^2 - 7a -/
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - a^2 - 7*a

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem max_value_implies_sum (a b : ℝ) :
  (∀ x, f a b x ≤ f a b 1) ∧ 
  (f a b 1 = 10) ∧ 
  (f' a b 1 = 0) →
  a + b = 3 := by sorry

end NUMINAMATH_CALUDE_max_value_implies_sum_l2724_272426
