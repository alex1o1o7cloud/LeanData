import Mathlib

namespace NUMINAMATH_CALUDE_school_student_count_l309_30909

theorem school_student_count 
  (total : ℕ) 
  (transferred : ℕ) 
  (difference : ℕ) 
  (h1 : total = 432) 
  (h2 : transferred = 16) 
  (h3 : difference = 24) :
  ∃ (a b : ℕ), 
    a + b = total ∧ 
    (a - transferred) = (b + transferred + difference) ∧
    a = 244 ∧ 
    b = 188 := by
  sorry

end NUMINAMATH_CALUDE_school_student_count_l309_30909


namespace NUMINAMATH_CALUDE_reciprocal_of_sum_l309_30996

theorem reciprocal_of_sum : (1 / (1/3 + 1/4) : ℚ) = 12/7 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_sum_l309_30996


namespace NUMINAMATH_CALUDE_car_distance_in_13_hours_l309_30914

/-- Represents the driving characteristics of the car -/
structure Car where
  speed : ℕ            -- Speed in miles per hour
  drive_time : ℕ       -- Continuous driving time in hours
  cool_time : ℕ        -- Cooling time in hours

/-- Calculates the total distance a car can travel in a given time -/
def total_distance (c : Car) (total_time : ℕ) : ℕ :=
  sorry

/-- Theorem stating the total distance the car can travel in 13 hours -/
theorem car_distance_in_13_hours (c : Car) 
  (h1 : c.speed = 8)
  (h2 : c.drive_time = 5)
  (h3 : c.cool_time = 1) :
  total_distance c 13 = 88 :=
sorry

end NUMINAMATH_CALUDE_car_distance_in_13_hours_l309_30914


namespace NUMINAMATH_CALUDE_average_price_reduction_l309_30920

theorem average_price_reduction (original_price final_price : ℝ) 
  (h1 : original_price = 60) 
  (h2 : final_price = 48.6) : 
  ∃ (x : ℝ), x = 0.1 ∧ original_price * (1 - x)^2 = final_price :=
sorry

end NUMINAMATH_CALUDE_average_price_reduction_l309_30920


namespace NUMINAMATH_CALUDE_triangle_area_l309_30903

theorem triangle_area (p A B : Real) (h_positive : p > 0) (h_angles : 0 < A ∧ 0 < B ∧ A + B < π) : 
  let C := π - A - B
  let S := (2 * p^2 * Real.sin A * Real.sin B * Real.sin C) / (Real.sin A + Real.sin B + Real.sin C)^2
  S > 0 ∧ S < p^2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l309_30903


namespace NUMINAMATH_CALUDE_paper_folding_perimeter_ratio_l309_30936

/-- Given a square piece of paper with side length 4 inches that is folded in half vertically
    and then cut in half parallel to the fold, the ratio of the perimeter of one of the resulting
    small rectangles to the perimeter of the large rectangle is 5/6. -/
theorem paper_folding_perimeter_ratio :
  let initial_side_length : ℝ := 4
  let small_rectangle_length : ℝ := initial_side_length
  let small_rectangle_width : ℝ := initial_side_length / 4
  let large_rectangle_length : ℝ := initial_side_length
  let large_rectangle_width : ℝ := initial_side_length / 2
  let small_perimeter : ℝ := 2 * (small_rectangle_length + small_rectangle_width)
  let large_perimeter : ℝ := 2 * (large_rectangle_length + large_rectangle_width)
  small_perimeter / large_perimeter = 5 / 6 := by
sorry


end NUMINAMATH_CALUDE_paper_folding_perimeter_ratio_l309_30936


namespace NUMINAMATH_CALUDE_curve_cartesian_to_polar_l309_30983

/-- Given a curve C in the Cartesian coordinate system described by the parametric equations
    x = cos α and y = sin α + 1, prove that its polar equation is ρ = 2 sin θ. -/
theorem curve_cartesian_to_polar (α θ : Real) (ρ : Real) (x y : Real) :
  (x = Real.cos α ∧ y = Real.sin α + 1) →
  (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ρ = 2 * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_curve_cartesian_to_polar_l309_30983


namespace NUMINAMATH_CALUDE_correct_verbs_for_sentence_l309_30910

-- Define the structure of a sentence with two blanks
structure SentenceWithBlanks where
  first_blank : String
  second_blank : String

-- Define the concept of subject-verb agreement
def subjectVerbAgrees (subject : String) (verb : String) : Prop := sorry

-- Define the specific sentence structure
def remoteAreasNeed : String := "Remote areas need"
def childrenNeed : String := "children need"

-- Theorem to prove
theorem correct_verbs_for_sentence :
  ∃ (s : SentenceWithBlanks),
    subjectVerbAgrees remoteAreasNeed s.first_blank ∧
    subjectVerbAgrees childrenNeed s.second_blank ∧
    s.first_blank = "is" ∧
    s.second_blank = "are" := by sorry

end NUMINAMATH_CALUDE_correct_verbs_for_sentence_l309_30910


namespace NUMINAMATH_CALUDE_train_length_l309_30954

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (speed_kmph : ℝ) (time_seconds : ℝ) (bridge_length : ℝ) : 
  speed_kmph = 36 → 
  time_seconds = 23.998080153587715 → 
  bridge_length = 140 → 
  (speed_kmph * 1000 / 3600) * time_seconds - bridge_length = 99.98080153587715 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l309_30954


namespace NUMINAMATH_CALUDE_max_value_of_roots_sum_l309_30902

/-- Given a quadratic polynomial x^2 - sx + q with roots r₁ and r₂ satisfying
    certain conditions, the maximum value of 1/r₁¹¹ + 1/r₂¹¹ is 2. -/
theorem max_value_of_roots_sum (s q r₁ r₂ : ℝ) : 
  r₁ + r₂ = s ∧ r₁ * r₂ = q ∧ 
  r₁ + r₂ = r₁^2 + r₂^2 ∧ 
  r₁ + r₂ = r₁^10 + r₂^10 →
  ∃ (M : ℝ), M = 2 ∧ ∀ (s' q' r₁' r₂' : ℝ), 
    (r₁' + r₂' = s' ∧ r₁' * r₂' = q' ∧ 
     r₁' + r₂' = r₁'^2 + r₂'^2 ∧ 
     r₁' + r₂' = r₁'^10 + r₂'^10) →
    1 / r₁'^11 + 1 / r₂'^11 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_roots_sum_l309_30902


namespace NUMINAMATH_CALUDE_fly_ceiling_distance_l309_30937

theorem fly_ceiling_distance (z : ℝ) :
  (3 : ℝ)^2 + 2^2 + z^2 = 6^2 → z = Real.sqrt 23 := by
  sorry

end NUMINAMATH_CALUDE_fly_ceiling_distance_l309_30937


namespace NUMINAMATH_CALUDE_same_color_probability_l309_30931

def total_balls : ℕ := 5 + 8 + 4 + 3

def green_balls : ℕ := 5
def white_balls : ℕ := 8
def blue_balls : ℕ := 4
def red_balls : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem same_color_probability :
  (choose green_balls 4 + choose white_balls 4 + choose blue_balls 4) / choose total_balls 4 = 76 / 4845 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l309_30931


namespace NUMINAMATH_CALUDE_relay_race_arrangements_eq_12_l309_30940

/-- The number of ways to arrange 5 runners in a relay race with specific constraints -/
def relay_race_arrangements : ℕ :=
  let total_runners : ℕ := 5
  let specific_runners : ℕ := 2
  let other_runners : ℕ := total_runners - specific_runners
  let ways_to_arrange_specific_runners : ℕ := 2
  let ways_to_arrange_other_runners : ℕ := Nat.factorial other_runners
  ways_to_arrange_specific_runners * ways_to_arrange_other_runners

/-- Theorem stating that the number of arrangements is 12 -/
theorem relay_race_arrangements_eq_12 : relay_race_arrangements = 12 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_eq_12_l309_30940


namespace NUMINAMATH_CALUDE_land_plot_side_length_l309_30926

theorem land_plot_side_length (area : ℝ) (h : area = Real.sqrt 1600) :
  Real.sqrt area = 40 := by
  sorry

end NUMINAMATH_CALUDE_land_plot_side_length_l309_30926


namespace NUMINAMATH_CALUDE_f_value_plus_derivative_at_pi_half_l309_30988

noncomputable def f (x : ℝ) : ℝ := (1 / x) * Real.cos x

theorem f_value_plus_derivative_at_pi_half (π : ℝ) (h : π > 0) :
  f π + (deriv f) (π / 2) = -3 / π :=
sorry

end NUMINAMATH_CALUDE_f_value_plus_derivative_at_pi_half_l309_30988


namespace NUMINAMATH_CALUDE_sin_symmetry_l309_30952

theorem sin_symmetry (t : ℝ) : 
  Real.sin ((π / 6 + t) + π / 3) = Real.sin ((π / 6 - t) + π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_symmetry_l309_30952


namespace NUMINAMATH_CALUDE_runner_time_difference_l309_30998

theorem runner_time_difference (danny_time steve_time : ℝ) (h1 : danny_time = 27) 
  (h2 : danny_time = steve_time / 2) : steve_time / 4 - danny_time / 2 = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_runner_time_difference_l309_30998


namespace NUMINAMATH_CALUDE_nails_to_buy_l309_30959

theorem nails_to_buy (tom_nails : ℝ) (toolshed_nails : ℝ) (drawer_nail : ℝ) (neighbor_nails : ℝ) (total_needed : ℝ) :
  tom_nails = 247 →
  toolshed_nails = 144 →
  drawer_nail = 0.5 →
  neighbor_nails = 58.75 →
  total_needed = 625.25 →
  total_needed - (tom_nails + toolshed_nails + drawer_nail + neighbor_nails) = 175 := by
  sorry

end NUMINAMATH_CALUDE_nails_to_buy_l309_30959


namespace NUMINAMATH_CALUDE_continuity_at_two_l309_30900

noncomputable def f (x : ℝ) : ℝ := (x^4 - 16) / (x^3 - 2*x^2)

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → |f x - 8| < ε :=
sorry

end NUMINAMATH_CALUDE_continuity_at_two_l309_30900


namespace NUMINAMATH_CALUDE_triangle_inequality_l309_30995

theorem triangle_inequality (R r a b c p : ℝ) 
  (h_R : R > 0) 
  (h_r : r > 0) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_c : c > 0) 
  (h_p : p = (a + b + c) / 2) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_circumradius : R = (a * b * c) / (4 * p * r)) 
  (h_inradius : r = p * (p - a) * (p - b) * (p - c) / (a * b * c)) :
  20 * R * r - 4 * r^2 ≤ a * b + b * c + c * a ∧ 
  a * b + b * c + c * a ≤ 4 * (R + r)^2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l309_30995


namespace NUMINAMATH_CALUDE_factorial_ratio_l309_30941

theorem factorial_ratio : (Nat.factorial 15) / ((Nat.factorial 6) * (Nat.factorial 9)) = 5005 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l309_30941


namespace NUMINAMATH_CALUDE_number_thought_of_l309_30992

theorem number_thought_of (x : ℝ) : (x / 4) + 9 = 15 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_thought_of_l309_30992


namespace NUMINAMATH_CALUDE_max_ratio_ab_l309_30907

theorem max_ratio_ab (a b : ℕ+) (h : (a : ℚ) / ((a : ℚ) - 2) = ((b : ℚ) + 2021) / ((b : ℚ) + 2008)) :
  (a : ℚ) / (b : ℚ) ≤ 312 / 7 := by
sorry

end NUMINAMATH_CALUDE_max_ratio_ab_l309_30907


namespace NUMINAMATH_CALUDE_ruble_exchange_impossible_l309_30919

theorem ruble_exchange_impossible : ¬ ∃ (x y z : ℕ), 
  x + y + z = 10 ∧ x + 3*y + 5*z = 25 := by sorry

end NUMINAMATH_CALUDE_ruble_exchange_impossible_l309_30919


namespace NUMINAMATH_CALUDE_pie_shop_revenue_l309_30913

/-- Represents the price of a single slice of pie in dollars -/
def slice_price : ℕ := 5

/-- Represents the number of slices in a whole pie -/
def slices_per_pie : ℕ := 4

/-- Represents the number of pies sold -/
def pies_sold : ℕ := 9

/-- Calculates the total revenue from selling pies -/
def total_revenue : ℕ := pies_sold * slices_per_pie * slice_price

theorem pie_shop_revenue :
  total_revenue = 180 :=
by sorry

end NUMINAMATH_CALUDE_pie_shop_revenue_l309_30913


namespace NUMINAMATH_CALUDE_remaining_balloons_l309_30977

theorem remaining_balloons (fred_balloons sam_balloons destroyed_balloons : ℝ)
  (h1 : fred_balloons = 10.0)
  (h2 : sam_balloons = 46.0)
  (h3 : destroyed_balloons = 16.0) :
  fred_balloons + sam_balloons - destroyed_balloons = 40.0 :=
by
  sorry

end NUMINAMATH_CALUDE_remaining_balloons_l309_30977


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l309_30921

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- Theorem: If S_7/7 - S_4/4 = 3 for an arithmetic sequence, then its common difference is 2 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h : seq.S 7 / 7 - seq.S 4 / 4 = 3) :
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l309_30921


namespace NUMINAMATH_CALUDE_fourth_person_height_l309_30915

/-- Given four people with heights in increasing order, prove that the fourth person is 82 inches tall -/
theorem fourth_person_height (h₁ h₂ h₃ h₄ : ℝ) : 
  h₁ < h₂ ∧ h₂ < h₃ ∧ h₃ < h₄ →  -- Heights are in increasing order
  h₂ - h₁ = 2 →                 -- Difference between 1st and 2nd person
  h₃ - h₂ = 2 →                 -- Difference between 2nd and 3rd person
  h₄ - h₃ = 6 →                 -- Difference between 3rd and 4th person
  (h₁ + h₂ + h₃ + h₄) / 4 = 76  -- Average height is 76 inches
  → h₄ = 82 := by               -- The fourth person is 82 inches tall
sorry

end NUMINAMATH_CALUDE_fourth_person_height_l309_30915


namespace NUMINAMATH_CALUDE_left_handed_rock_lovers_l309_30957

theorem left_handed_rock_lovers (total : ℕ) (left_handed : ℕ) (rock_lovers : ℕ) (right_handed_rock_dislikers : ℕ) :
  total = 30 →
  left_handed = 14 →
  rock_lovers = 20 →
  right_handed_rock_dislikers = 5 →
  ∃ (x : ℕ),
    x = left_handed + rock_lovers - total + right_handed_rock_dislikers ∧
    x = 9 :=
by sorry

end NUMINAMATH_CALUDE_left_handed_rock_lovers_l309_30957


namespace NUMINAMATH_CALUDE_grid_whitening_l309_30986

/-- Represents the color of a cell -/
inductive Color
| Black
| White

/-- Represents a 9x9 grid of cells -/
def Grid := Fin 9 → Fin 9 → Color

/-- Represents a corner shape operation -/
structure CornerOperation where
  row : Fin 9
  col : Fin 9
  orientation : Fin 4

/-- Applies a corner operation to a grid -/
def applyOperation (g : Grid) (op : CornerOperation) : Grid :=
  sorry

/-- Checks if all cells in the grid are white -/
def allWhite (g : Grid) : Prop :=
  ∀ (i j : Fin 9), g i j = Color.White

/-- Main theorem: Any grid can be made all white with finite operations -/
theorem grid_whitening (g : Grid) :
  ∃ (ops : List CornerOperation), allWhite (ops.foldl applyOperation g) :=
  sorry

end NUMINAMATH_CALUDE_grid_whitening_l309_30986


namespace NUMINAMATH_CALUDE_mr_li_age_is_25_l309_30982

-- Define Xiaofang's age this year
def xiaofang_age : ℕ := 5

-- Define the number of years in the future
def years_in_future : ℕ := 3

-- Define the age difference between Mr. Li and Xiaofang in the future
def future_age_difference : ℕ := 20

-- Define Mr. Li's age this year
def mr_li_age : ℕ := xiaofang_age + future_age_difference

-- Theorem to prove
theorem mr_li_age_is_25 : mr_li_age = 25 := by
  sorry

end NUMINAMATH_CALUDE_mr_li_age_is_25_l309_30982


namespace NUMINAMATH_CALUDE_family_ages_l309_30964

/-- Problem statement about the ages of family members -/
theorem family_ages (mark_age john_age emma_age parents_current_age : ℕ)
  (h1 : mark_age = 18)
  (h2 : john_age = mark_age - 10)
  (h3 : emma_age = mark_age - 4)
  (h4 : parents_current_age = 7 * john_age)
  (h5 : parents_current_age = 25 + emma_age) :
  parents_current_age - mark_age = 38 := by
  sorry

end NUMINAMATH_CALUDE_family_ages_l309_30964


namespace NUMINAMATH_CALUDE_special_form_not_perfect_square_l309_30911

/-- A function that returns true if the input number has at least three digits,
    all digits except the first and last are zeros, and the first and last digits are non-zeros -/
def has_special_form (n : ℕ) : Prop :=
  n ≥ 100 ∧
  ∃ (d b : ℕ) (k : ℕ), 
    d ≠ 0 ∧ b ≠ 0 ∧ 
    n = d * 10^k + b ∧
    k ≥ 1

theorem special_form_not_perfect_square (n : ℕ) :
  has_special_form n → ¬ ∃ (m : ℕ), n = m^2 :=
by sorry

end NUMINAMATH_CALUDE_special_form_not_perfect_square_l309_30911


namespace NUMINAMATH_CALUDE_set_operations_l309_30978

def A : Set ℕ := {x | x > 0 ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem set_operations :
  (A ∩ C = {3, 4, 5, 6, 7}) ∧
  ((A \ B) = {5, 6, 7, 8, 9, 10}) ∧
  ((A \ (B ∪ C)) = {8, 9, 10}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) := by
sorry

end NUMINAMATH_CALUDE_set_operations_l309_30978


namespace NUMINAMATH_CALUDE_problem_statement_l309_30924

open Real

theorem problem_statement : 
  let p := ∃ x₀ : ℝ, Real.exp x₀ ≤ 0
  let q := ∀ x : ℝ, 2^x > x^2
  (¬p) ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_statement_l309_30924


namespace NUMINAMATH_CALUDE_quadratic_max_min_difference_l309_30917

/-- Given a quadratic function f(x) = -x^2 + 10x + 9 defined on the interval [2, a/9],
    where a/9 ≥ 8, the difference between its maximum and minimum values is 9. -/
theorem quadratic_max_min_difference (a : ℝ) (h : a / 9 ≥ 8) :
  let f : ℝ → ℝ := λ x ↦ -x^2 + 10*x + 9
  let max_val := (⨆ x ∈ Set.Icc 2 (a / 9), f x)
  let min_val := (⨅ x ∈ Set.Icc 2 (a / 9), f x)
  max_val - min_val = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_max_min_difference_l309_30917


namespace NUMINAMATH_CALUDE_is_hyperbola_center_l309_30942

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 288 * y - 576 = 0

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 4)

/-- Theorem stating that the given point is the center of the hyperbola -/
theorem is_hyperbola_center : 
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    ((x - hyperbola_center.1)^2 / 9 - (y - hyperbola_center.2)^2 / 4 = 1) :=
by sorry

end NUMINAMATH_CALUDE_is_hyperbola_center_l309_30942


namespace NUMINAMATH_CALUDE_problem_solution_l309_30946

theorem problem_solution (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = -3) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 5 + (2829/27) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l309_30946


namespace NUMINAMATH_CALUDE_triangle_problem_l309_30967

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if (b-2a)cos C + c cos B = 0, c = √7, and b = 3a, then the measure of angle C is π/3
    and the area of the triangle is 3√3/4. -/
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (b - 2*a) * Real.cos C + c * Real.cos B = 0 →
  c = Real.sqrt 7 →
  b = 3*a →
  C = π/3 ∧ (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l309_30967


namespace NUMINAMATH_CALUDE_nonagon_diagonals_l309_30949

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- A regular nine-sided polygon contains 27 diagonals -/
theorem nonagon_diagonals : num_diagonals 9 = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonals_l309_30949


namespace NUMINAMATH_CALUDE_polynomial_integral_theorem_l309_30905

/-- A polynomial of degree at most 2 -/
def Polynomial2 := ℝ → ℝ

/-- The definite integral of a polynomial from a to b -/
noncomputable def integral (f : Polynomial2) (a b : ℝ) : ℝ := sorry

/-- The condition that the integrals sum to zero -/
def integralCondition (f : Polynomial2) (p q r : ℝ) : Prop :=
  integral f (-1) p - integral f p q + integral f q r - integral f r 1 = 0

theorem polynomial_integral_theorem :
  ∃! (p q r : ℝ), 
    -1 < p ∧ p < q ∧ q < r ∧ r < 1 ∧
    (∀ f : Polynomial2, integralCondition f p q r) ∧
    p = 1 / Real.sqrt 2 ∧ q = 0 ∧ r = -1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_polynomial_integral_theorem_l309_30905


namespace NUMINAMATH_CALUDE_paving_rate_per_sq_meter_l309_30963

/-- Given a rectangular room with length 5.5 m and width 4 m, 
    and a total paving cost of Rs. 16500, 
    prove that the paving rate per square meter is Rs. 750. -/
theorem paving_rate_per_sq_meter 
  (length : ℝ) 
  (width : ℝ) 
  (total_cost : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 4)
  (h3 : total_cost = 16500) :
  total_cost / (length * width) = 750 := by
  sorry

end NUMINAMATH_CALUDE_paving_rate_per_sq_meter_l309_30963


namespace NUMINAMATH_CALUDE_sum_of_xyz_l309_30991

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 40) (h2 : x * z = 80) (h3 : y * z = 160) :
  x + y + z = 14 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l309_30991


namespace NUMINAMATH_CALUDE_fill_time_three_pipes_l309_30971

/-- Represents a pipe that can fill or empty a tank -/
structure Pipe where
  rate : ℚ  -- Rate at which the pipe fills (positive) or empties (negative) the tank per hour

/-- Represents a system of pipes filling a tank -/
def PipeSystem (pipes : List Pipe) : ℚ :=
  pipes.map (·.rate) |> List.sum

theorem fill_time_three_pipes (a b c : Pipe) 
  (ha : a.rate = 1/3)
  (hb : b.rate = 1/4)
  (hc : c.rate = -1/4) :
  (PipeSystem [a, b, c])⁻¹ = 3 := by
  sorry

#check fill_time_three_pipes

end NUMINAMATH_CALUDE_fill_time_three_pipes_l309_30971


namespace NUMINAMATH_CALUDE_rainfall_problem_l309_30962

/-- Rainfall problem -/
theorem rainfall_problem (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 35 →
  ratio = 1.5 →
  ∃ (first_week : ℝ),
    first_week + ratio * first_week = total_rainfall ∧
    ratio * first_week = 21 := by
  sorry


end NUMINAMATH_CALUDE_rainfall_problem_l309_30962


namespace NUMINAMATH_CALUDE_distance_to_market_is_40_l309_30997

/-- The distance between Andy's house and the market -/
def distance_to_market (distance_to_school : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance - 2 * distance_to_school

/-- Theorem: The distance between Andy's house and the market is 40 meters -/
theorem distance_to_market_is_40 :
  distance_to_market 50 140 = 40 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_market_is_40_l309_30997


namespace NUMINAMATH_CALUDE_unique_solution_l309_30979

def U : Set ℤ := {-2, 3, 4, 5}

def M (p q : ℝ) : Set ℤ := {x ∈ U | (x : ℝ)^2 + p * x + q = 0}

theorem unique_solution :
  ∃! (p q : ℝ), (U \ M p q : Set ℤ) = {3, 5} := by sorry

end NUMINAMATH_CALUDE_unique_solution_l309_30979


namespace NUMINAMATH_CALUDE_tea_mixture_price_l309_30994

/-- Given three varieties of tea with prices and mixing ratios, calculate the price of the mixture --/
theorem tea_mixture_price (price1 price2 price3 : ℚ) (ratio1 ratio2 ratio3 : ℚ) :
  price1 = 126 →
  price2 = 135 →
  price3 = 175.5 →
  ratio1 = 1 →
  ratio2 = 1 →
  ratio3 = 2 →
  (price1 * ratio1 + price2 * ratio2 + price3 * ratio3) / (ratio1 + ratio2 + ratio3) = 153 := by
  sorry

end NUMINAMATH_CALUDE_tea_mixture_price_l309_30994


namespace NUMINAMATH_CALUDE_equation_solution_l309_30948

theorem equation_solution : ∃ x : ℝ, (225 - 4209520 / ((1000795 + (250 + x) * 50) / 27) = 113) ∧ x = 40 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l309_30948


namespace NUMINAMATH_CALUDE_coin_flip_probability_l309_30955

/-- The probability of a coin landing heads. -/
def p_heads : ℚ := 3/5

/-- The probability of a coin landing tails. -/
def p_tails : ℚ := 1 - p_heads

/-- The number of times the coin is flipped. -/
def num_flips : ℕ := 8

/-- The number of initial flips that should be heads. -/
def num_heads : ℕ := 3

/-- The number of final flips that should be tails. -/
def num_tails : ℕ := num_flips - num_heads

/-- The probability of getting heads on the first 3 flips and tails on the last 5 flips. -/
def prob_specific_sequence : ℚ := p_heads^num_heads * p_tails^num_tails

theorem coin_flip_probability : prob_specific_sequence = 864/390625 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l309_30955


namespace NUMINAMATH_CALUDE_probability_two_twos_in_five_rolls_probability_two_twos_in_five_rolls_proof_l309_30984

/-- The probability of rolling a 2 exactly two times in five rolls of a fair eight-sided die -/
theorem probability_two_twos_in_five_rolls : ℝ :=
let p : ℝ := 1 / 8  -- probability of rolling a 2
let q : ℝ := 1 - p  -- probability of not rolling a 2
let n : ℕ := 5      -- number of rolls
let k : ℕ := 2      -- number of desired successes
3430 / 32768

/-- Proof that the probability is correct -/
theorem probability_two_twos_in_five_rolls_proof :
  probability_two_twos_in_five_rolls = 3430 / 32768 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_twos_in_five_rolls_probability_two_twos_in_five_rolls_proof_l309_30984


namespace NUMINAMATH_CALUDE_four_thirds_of_nine_halves_l309_30956

theorem four_thirds_of_nine_halves : (4 / 3 : ℚ) * (9 / 2 : ℚ) = 6 := by
  sorry

end NUMINAMATH_CALUDE_four_thirds_of_nine_halves_l309_30956


namespace NUMINAMATH_CALUDE_green_balls_count_l309_30922

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 ∧
  white = 50 ∧
  yellow = 10 ∧
  red = 7 ∧
  purple = 3 ∧
  prob_not_red_purple = 9/10 →
  ∃ green : ℕ, green = 30 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_green_balls_count_l309_30922


namespace NUMINAMATH_CALUDE_shooting_scenario_outcomes_l309_30973

/-- Represents the number of shots fired -/
def total_shots : ℕ := 8

/-- Represents the number of successful hits -/
def total_hits : ℕ := 4

/-- Represents the number of consecutive hits required -/
def consecutive_hits : ℕ := 3

/-- Calculates the number of different outcomes for the shooting scenario -/
def shooting_outcomes : ℕ := total_shots + 1 - total_hits

/-- Theorem stating that the number of different outcomes is 20 -/
theorem shooting_scenario_outcomes : 
  shooting_outcomes = 20 := by sorry

end NUMINAMATH_CALUDE_shooting_scenario_outcomes_l309_30973


namespace NUMINAMATH_CALUDE_system_solution_l309_30923

theorem system_solution (x y z w : ℝ) : 
  (x - y + z - w = 2 ∧
   x^2 - y^2 + z^2 - w^2 = 6 ∧
   x^3 - y^3 + z^3 - w^3 = 20 ∧
   x^4 - y^4 + z^4 - w^4 = 60) ↔ 
  ((x = 1 ∧ y = 2 ∧ z = 3 ∧ w = 0) ∨
   (x = 1 ∧ y = 0 ∧ z = 3 ∧ w = 2) ∨
   (x = 3 ∧ y = 2 ∧ z = 1 ∧ w = 0) ∨
   (x = 3 ∧ y = 0 ∧ z = 1 ∧ w = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l309_30923


namespace NUMINAMATH_CALUDE_art_interest_group_end_time_l309_30927

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.minutes + m
  let newHours := (t.hours + totalMinutes / 60) % 24
  let newMinutes := totalMinutes % 60
  ⟨newHours, newMinutes, by sorry, by sorry⟩

theorem art_interest_group_end_time :
  let start_time : Time := ⟨15, 20, by sorry, by sorry⟩
  let duration : Nat := 50
  addMinutes start_time duration = ⟨16, 10, by sorry, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_art_interest_group_end_time_l309_30927


namespace NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l309_30947

/-- For the quadratic x^2 - 20x + 49, when written as (x+b)^2+c, b+c equals -61 -/
theorem quadratic_sum_of_b_and_c : ∃ b c : ℝ, 
  (∀ x : ℝ, x^2 - 20*x + 49 = (x+b)^2 + c) ∧ b + c = -61 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_b_and_c_l309_30947


namespace NUMINAMATH_CALUDE_average_work_difference_l309_30976

def daily_differences : List ℤ := [2, -1, 3, 1, -2, 2, 1]

def days_in_week : ℕ := 7

theorem average_work_difference :
  (daily_differences.sum : ℚ) / days_in_week = 0.857 := by
  sorry

end NUMINAMATH_CALUDE_average_work_difference_l309_30976


namespace NUMINAMATH_CALUDE_correct_quadratic_equation_l309_30958

theorem correct_quadratic_equation :
  ∀ (b c : ℝ),
  (∃ (b' c' : ℝ), (5 : ℝ) * (1 : ℝ) = c' ∧ 5 + 1 = -b) →
  (∃ (b'' : ℝ), (-7 : ℝ) * (-2 : ℝ) = c) →
  (x^2 + b*x + c = 0) = (x^2 - 6*x + 14 = 0) :=
by sorry

end NUMINAMATH_CALUDE_correct_quadratic_equation_l309_30958


namespace NUMINAMATH_CALUDE_new_girl_weight_l309_30908

/-- The weight of the new girl given the conditions of the problem -/
def weight_of_new_girl (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) : ℝ :=
  replaced_weight + initial_count * weight_increase

/-- Theorem stating the weight of the new girl under the given conditions -/
theorem new_girl_weight :
  weight_of_new_girl 8 3 70 = 94 := by
  sorry

end NUMINAMATH_CALUDE_new_girl_weight_l309_30908


namespace NUMINAMATH_CALUDE_ceiling_equation_solution_l309_30974

theorem ceiling_equation_solution :
  ∃! b : ℝ, b + ⌈b⌉ = 21.5 ∧ b = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_equation_solution_l309_30974


namespace NUMINAMATH_CALUDE_overlapping_sectors_area_l309_30975

theorem overlapping_sectors_area (r : ℝ) (h : r = 12) :
  let sector_angle : ℝ := 60
  let sector_area := (sector_angle / 360) * Real.pi * r^2
  let triangle_area := (Real.sqrt 3 / 4) * r^2
  let shaded_area := 2 * (sector_area - triangle_area)
  shaded_area = 48 * Real.pi - 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_sectors_area_l309_30975


namespace NUMINAMATH_CALUDE_election_loss_calculation_l309_30928

theorem election_loss_calculation (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 8000 →
  candidate_percentage = 1/4 →
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 4000 := by
  sorry

end NUMINAMATH_CALUDE_election_loss_calculation_l309_30928


namespace NUMINAMATH_CALUDE_intersection_point_x_coordinate_l309_30933

theorem intersection_point_x_coordinate :
  let line1 : ℝ → ℝ := λ x => 3 * x - 22
  let line2 : ℝ → ℝ := λ x => 100 - 3 * x
  ∃ x : ℝ, line1 x = line2 x ∧ x = 61 / 3 :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_x_coordinate_l309_30933


namespace NUMINAMATH_CALUDE_correct_sticker_count_l309_30930

/-- Represents the number of stickers per page for each type -/
def stickers_per_page : Fin 4 → ℕ
  | 0 => 5  -- Type A
  | 1 => 3  -- Type B
  | 2 => 2  -- Type C
  | 3 => 1  -- Type D

/-- The total number of pages -/
def total_pages : ℕ := 22

/-- Calculates the total number of stickers for a given type -/
def total_stickers (type : Fin 4) : ℕ :=
  (stickers_per_page type) * total_pages

/-- Theorem stating the correct total number of stickers for each type -/
theorem correct_sticker_count :
  (total_stickers 0 = 110) ∧
  (total_stickers 1 = 66) ∧
  (total_stickers 2 = 44) ∧
  (total_stickers 3 = 22) := by
  sorry

end NUMINAMATH_CALUDE_correct_sticker_count_l309_30930


namespace NUMINAMATH_CALUDE_train_length_calculation_l309_30951

/-- Calculate the length of a train given its speed, the speed of a person moving in the opposite direction, and the time it takes for the train to pass the person. -/
theorem train_length_calculation (train_speed : ℝ) (person_speed : ℝ) (passing_time : ℝ) :
  train_speed = 60 →
  person_speed = 6 →
  passing_time = 6 →
  (train_speed + person_speed) * passing_time * (1000 / 3600) = 110.04 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l309_30951


namespace NUMINAMATH_CALUDE_anna_candy_per_house_proof_l309_30935

/-- The number of candy pieces Anna gets per house -/
def anna_candy_per_house : ℕ := 14

/-- The number of candy pieces Billy gets per house -/
def billy_candy_per_house : ℕ := 11

/-- The number of houses Anna visits -/
def anna_houses : ℕ := 60

/-- The number of houses Billy visits -/
def billy_houses : ℕ := 75

/-- The difference in total candy pieces between Anna and Billy -/
def candy_difference : ℕ := 15

theorem anna_candy_per_house_proof :
  anna_candy_per_house * anna_houses = billy_candy_per_house * billy_houses + candy_difference :=
by
  sorry

#eval anna_candy_per_house

end NUMINAMATH_CALUDE_anna_candy_per_house_proof_l309_30935


namespace NUMINAMATH_CALUDE_cone_surface_area_l309_30993

/-- The surface area of a cone with slant height 2 and base radius 1 is 3π -/
theorem cone_surface_area :
  let slant_height : ℝ := 2
  let base_radius : ℝ := 1
  let lateral_area := π * base_radius * slant_height
  let base_area := π * base_radius^2
  lateral_area + base_area = 3 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_surface_area_l309_30993


namespace NUMINAMATH_CALUDE_largest_non_sum_36_composite_l309_30961

def is_composite (n : ℕ) : Prop := ∃ m k, 1 < m ∧ 1 < k ∧ n = m * k

def is_sum_of_multiple_36_and_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 0 < a ∧ is_composite b ∧ n = 36 * a + b

theorem largest_non_sum_36_composite : 
  (∀ n > 209, is_sum_of_multiple_36_and_composite n) ∧
  ¬is_sum_of_multiple_36_and_composite 209 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_36_composite_l309_30961


namespace NUMINAMATH_CALUDE_unit_intervals_have_continuum_cardinality_l309_30999

-- Define the cardinality of the continuum
def continuum_cardinality := Cardinal.mk ℝ

-- Define the open interval (0,1)
def open_unit_interval := Set.Ioo (0 : ℝ) 1

-- Define the closed interval [0,1]
def closed_unit_interval := Set.Icc (0 : ℝ) 1

-- Theorem statement
theorem unit_intervals_have_continuum_cardinality :
  (Cardinal.mk open_unit_interval = continuum_cardinality) ∧
  (Cardinal.mk closed_unit_interval = continuum_cardinality) := by
  sorry

end NUMINAMATH_CALUDE_unit_intervals_have_continuum_cardinality_l309_30999


namespace NUMINAMATH_CALUDE_calculate_expression_l309_30990

theorem calculate_expression : 3 * 301 + 4 * 301 + 5 * 301 + 300 = 3912 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l309_30990


namespace NUMINAMATH_CALUDE_water_leaked_calculation_l309_30980

/-- The amount of water that leaked out of a bucket -/
def water_leaked (initial : ℝ) (remaining : ℝ) : ℝ :=
  initial - remaining

theorem water_leaked_calculation (initial : ℝ) (remaining : ℝ) 
  (h1 : initial = 0.75)
  (h2 : remaining = 0.5) : 
  water_leaked initial remaining = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_water_leaked_calculation_l309_30980


namespace NUMINAMATH_CALUDE_polynomial_real_root_exists_l309_30968

theorem polynomial_real_root_exists (b : ℝ) : ∃ x : ℝ, x^3 + b*x^2 - 4*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_real_root_exists_l309_30968


namespace NUMINAMATH_CALUDE_coal_relationship_warehouse_b_coal_amount_l309_30960

/-- The amount of coal in warehouse A in tons -/
def warehouse_a_coal : ℝ := 130

/-- The amount of coal in warehouse B in tons -/
def warehouse_b_coal : ℝ := 150

/-- Theorem stating the relationship between coal in warehouses A and B -/
theorem coal_relationship : warehouse_a_coal = 0.8 * warehouse_b_coal + 10 := by
  sorry

/-- Theorem proving the amount of coal in warehouse B -/
theorem warehouse_b_coal_amount : warehouse_b_coal = 150 := by
  sorry

end NUMINAMATH_CALUDE_coal_relationship_warehouse_b_coal_amount_l309_30960


namespace NUMINAMATH_CALUDE_units_digit_13_times_41_l309_30969

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The theorem stating that the units digit of 13 · 41 is 3 -/
theorem units_digit_13_times_41 : unitsDigit (13 * 41) = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_13_times_41_l309_30969


namespace NUMINAMATH_CALUDE_complex_magnitude_l309_30965

theorem complex_magnitude (z : ℂ) (h : (2 + Complex.I) * z = 4 - (1 + Complex.I)^2) : 
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l309_30965


namespace NUMINAMATH_CALUDE_max_min_product_l309_30989

theorem max_min_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y + z = 12) (hprod : x * y + y * z + z * x = 30) :
  ∃ (n : ℝ), n = min (x * y) (min (y * z) (z * x)) ∧ n ≤ 2 ∧
  ∀ (m : ℝ), m = min (x * y) (min (y * z) (z * x)) → m ≤ n :=
sorry

end NUMINAMATH_CALUDE_max_min_product_l309_30989


namespace NUMINAMATH_CALUDE_no_roots_in_interval_l309_30987

-- Define the function f(x) = x^3 + x^2 - 2x - 1
def f (x : ℝ) : ℝ := x^3 + x^2 - 2*x - 1

-- State the theorem
theorem no_roots_in_interval :
  (Continuous f) →
  (f 0 < 0) →
  (f 1 < 0) →
  ∀ x ∈ Set.Ioo 0 1, f x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_roots_in_interval_l309_30987


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l309_30932

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 8*a + 8 = 0) → (b^2 - 8*b + 8 = 0) → (a^2 + b^2 = 48) := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l309_30932


namespace NUMINAMATH_CALUDE_plan_y_cheaper_at_min_mb_l309_30981

/-- Represents the cost of a data plan in cents -/
def PlanCost (initialFee : ℕ) (ratePerMB : ℕ) (dataUsage : ℕ) : ℕ :=
  initialFee * 100 + ratePerMB * dataUsage

/-- The minimum whole number of MBs for Plan Y to be cheaper than Plan X -/
def minMBForPlanYCheaper : ℕ := 501

theorem plan_y_cheaper_at_min_mb :
  PlanCost 25 10 minMBForPlanYCheaper < PlanCost 0 15 minMBForPlanYCheaper ∧
  ∀ m : ℕ, m < minMBForPlanYCheaper →
    PlanCost 0 15 m ≤ PlanCost 25 10 m :=
by sorry

end NUMINAMATH_CALUDE_plan_y_cheaper_at_min_mb_l309_30981


namespace NUMINAMATH_CALUDE_consecutive_non_prime_powers_l309_30934

theorem consecutive_non_prime_powers (k : ℕ+) :
  ∃ (n : ℕ), ∀ (i : ℕ), i ∈ Finset.range k →
    ¬∃ (p : ℕ) (e : ℕ), Nat.Prime p ∧ (n + i = p ^ e) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_non_prime_powers_l309_30934


namespace NUMINAMATH_CALUDE_infinitely_many_common_terms_l309_30970

/-- Sequence a_n defined by the recurrence relation -/
def a : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 14 * a (n + 1) + a n

/-- Sequence b_n defined by the recurrence relation -/
def b : ℕ → ℤ
  | 0 => 2
  | 1 => 14
  | (n + 2) => 6 * b (n + 1) - b n

/-- There exist infinitely many pairs of natural numbers (n, m) such that a_n = b_m -/
theorem infinitely_many_common_terms : ∀ k : ℕ, ∃ n m : ℕ, n > k ∧ m > k ∧ a n = b m := by
  sorry


end NUMINAMATH_CALUDE_infinitely_many_common_terms_l309_30970


namespace NUMINAMATH_CALUDE_coastline_scientific_notation_l309_30901

theorem coastline_scientific_notation : 
  37515000 = 3.7515 * (10 : ℝ)^7 := by
  sorry

end NUMINAMATH_CALUDE_coastline_scientific_notation_l309_30901


namespace NUMINAMATH_CALUDE_inequality_range_proof_l309_30966

theorem inequality_range_proof (a : ℝ) : 
  (∀ x : ℝ, x > -2/a → a * Real.exp (a * x) - Real.log (x + 2/a) - 2 ≥ 0) ↔ 
  (a ≥ Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_proof_l309_30966


namespace NUMINAMATH_CALUDE_solve_system_l309_30929

theorem solve_system (x y : ℤ) (h1 : x + y = 14) (h2 : x - y = 60) : x = 37 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l309_30929


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l309_30953

theorem nested_fraction_evaluation : 
  2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l309_30953


namespace NUMINAMATH_CALUDE_odd_sum_even_equivalence_l309_30918

theorem odd_sum_even_equivalence (x y : ℤ) :
  (Odd x ∧ Odd y → Even (x + y)) ↔ (¬Even (x + y) → ¬(Odd x ∧ Odd y)) := by sorry

end NUMINAMATH_CALUDE_odd_sum_even_equivalence_l309_30918


namespace NUMINAMATH_CALUDE_max_points_top_three_l309_30939

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (games_per_pair : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculate the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  (t.num_teams.choose 2) * t.games_per_pair

/-- Calculate the maximum points a team can achieve -/
def max_points_per_team (t : Tournament) : ℕ :=
  (t.num_teams - 1) * t.games_per_pair * t.points_for_win

/-- The main theorem to prove -/
theorem max_points_top_three (t : Tournament) 
  (h1 : t.num_teams = 9)
  (h2 : t.games_per_pair = 2)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1)
  (h5 : t.points_for_loss = 0) :
  ∃ (max_points : ℕ), max_points = 42 ∧ 
  (∀ (top_three_points : ℕ), top_three_points ≤ max_points) ∧
  (∃ (strategy : Tournament → ℕ), strategy t = max_points) :=
sorry

end NUMINAMATH_CALUDE_max_points_top_three_l309_30939


namespace NUMINAMATH_CALUDE_sine_equality_l309_30938

theorem sine_equality (n : ℤ) : 0 ≤ n ∧ n ≤ 180 ∧ n = 55 → Real.sin (n * π / 180) = Real.sin (845 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sine_equality_l309_30938


namespace NUMINAMATH_CALUDE_elberta_has_45_dollars_l309_30904

-- Define the amounts for each person
def granny_smith_amount : ℕ := 100
def anjou_amount : ℕ := (2 * granny_smith_amount) / 5
def elberta_amount : ℕ := anjou_amount + 5

-- Theorem to prove
theorem elberta_has_45_dollars : elberta_amount = 45 := by
  sorry

end NUMINAMATH_CALUDE_elberta_has_45_dollars_l309_30904


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l309_30985

def quadratic_function (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_properties (f : ℝ → ℝ) 
  (h_quadratic : quadratic_function f)
  (h_f_0 : f 0 = 1)
  (h_f_diff : ∀ x, f (x + 1) - f x = 2 * x) :
  (∀ x, f x = x^2 - x + 1) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ 3) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≥ 3/4) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3) ∧
  (∃ x ∈ Set.Icc (-1) 1, f x = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l309_30985


namespace NUMINAMATH_CALUDE_percentage_problem_l309_30972

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 600 = (40 / 100) * 1050 → P = 70 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l309_30972


namespace NUMINAMATH_CALUDE_walking_problem_l309_30925

/-- The correct system of equations for the walking problem -/
theorem walking_problem (x y : ℝ) : 
  (∃ (total_distance : ℝ) (meeting_time : ℝ) (additional_time : ℝ),
    total_distance = 3 ∧ 
    meeting_time = 20/60 ∧ 
    additional_time = 10/60 ∧ 
    meeting_time * (x + y) = total_distance ∧
    (total_distance - (meeting_time + additional_time) * x) = 2 * (total_distance - (meeting_time + additional_time) * y)) ↔ 
  ((20/60 * x + 20/60 * y = 3) ∧ (3 - 30/60 * x = 2 * (3 - 30/60 * y))) :=
by sorry

end NUMINAMATH_CALUDE_walking_problem_l309_30925


namespace NUMINAMATH_CALUDE_circle_and_tangents_l309_30943

-- Define the circles and points
def circle_C (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = (a + 1)^2 + 1}

def circle_D : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 4)^2 + p.2^2 = 4}

-- Define the conditions
def passes_through (C : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ C

def tangent_line (P A : ℝ × ℝ) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (y - A.2 = m * (x - A.1)) →
    ((x, y) ∈ C → (x, y) = A ∨ (x, y) = P)

-- State the theorem
theorem circle_and_tangents :
  ∀ (a : ℝ),
    passes_through (circle_C a) (0, 0) →
    passes_through (circle_C a) (-1, 1) →
    (∀ (P : ℝ × ℝ), P ∈ circle_D →
      ∃ (A B : ℝ × ℝ),
        A.1 = 0 ∧ B.1 = 0 ∧
        tangent_line P A (circle_C a) ∧
        tangent_line P B (circle_C a)) →
  (∀ (x y : ℝ), (x, y) ∈ circle_C a ↔ (x + 1)^2 + y^2 = 1) ∧
  (∃ (min max : ℝ),
    min = 5 * Real.sqrt 2 / 4 ∧
    max = Real.sqrt 2 ∧
    (∀ (P : ℝ × ℝ), P ∈ circle_D →
      ∃ (A B : ℝ × ℝ),
        A.1 = 0 ∧ B.1 = 0 ∧
        tangent_line P A (circle_C a) ∧
        tangent_line P B (circle_C a) ∧
        min ≤ |A.2 - B.2| ∧ |A.2 - B.2| ≤ max)) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangents_l309_30943


namespace NUMINAMATH_CALUDE_white_balls_count_l309_30944

/-- The number of balls in the bag -/
def total_balls : ℕ := 7

/-- The expected number of white balls when drawing 2 balls -/
def expected_white : ℚ := 6/7

/-- Calculates the expected number of white balls drawn -/
def calculate_expected (white_balls : ℕ) : ℚ :=
  (Nat.choose white_balls 2 * 2 + Nat.choose white_balls 1 * Nat.choose (total_balls - white_balls) 1) / Nat.choose total_balls 2

/-- Theorem stating that the number of white balls is 3 -/
theorem white_balls_count :
  ∃ (n : ℕ), n < total_balls ∧ calculate_expected n = expected_white ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_white_balls_count_l309_30944


namespace NUMINAMATH_CALUDE_reciprocal_sum_inequality_quadratic_inequality_range_l309_30945

variable (a b c : ℝ)

-- Define the conditions
def sum_condition (a b c : ℝ) : Prop := a + b + c = 3
def positive_condition (a b c : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0

-- Theorem 1
theorem reciprocal_sum_inequality (h1 : sum_condition a b c) (h2 : positive_condition a b c) :
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 3 / 2 := by sorry

-- Theorem 2
theorem quadratic_inequality_range (h1 : sum_condition a b c) (h2 : positive_condition a b c) :
  ∀ m : ℝ, (∀ x : ℝ, -x^2 + m*x + 2 ≤ a^2 + b^2 + c^2) ↔ -2 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_reciprocal_sum_inequality_quadratic_inequality_range_l309_30945


namespace NUMINAMATH_CALUDE_min_value_of_expression_l309_30916

theorem min_value_of_expression (x y z : ℝ) 
  (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) 
  (h4 : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z) ≥ 36 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 1 ∧ (1 / a + 4 / b + 9 / c = 36) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l309_30916


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l309_30950

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) :
  total_distance = 560 →
  total_time = 25 →
  first_half_speed = 21 →
  ∃ second_half_speed : ℝ,
    second_half_speed = 24 ∧
    (total_distance / 2) / first_half_speed + (total_distance / 2) / second_half_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l309_30950


namespace NUMINAMATH_CALUDE_sine_of_intersection_angle_l309_30906

/-- The sine of the angle formed by a point on y = 3x and x^2 + y^2 = 1 in the first quadrant -/
theorem sine_of_intersection_angle (x y : ℝ) (h1 : y = 3 * x) (h2 : x^2 + y^2 = 1) 
  (h3 : x > 0) (h4 : y > 0) : 
  Real.sin (Real.arctan (y / x)) = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_sine_of_intersection_angle_l309_30906


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l309_30912

/-- Given a 2x2 matrix M, prove that its inverse is correct. -/
theorem matrix_inverse_proof (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : M = ![![1, 0], ![1, 1]]) : 
  M⁻¹ = ![![1, 0], ![-1, 1]] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l309_30912
