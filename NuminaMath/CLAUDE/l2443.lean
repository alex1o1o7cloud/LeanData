import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_range_l2443_244377

theorem complex_modulus_range (a : ℝ) : 
  (∀ θ : ℝ, Complex.abs ((a + Real.cos θ) + (2 * a - Real.sin θ) * Complex.I) ≤ 2) ↔ 
  a ∈ Set.Icc (-(Real.sqrt 5 / 5)) (Real.sqrt 5 / 5) := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l2443_244377


namespace NUMINAMATH_CALUDE_line_through_point_with_triangle_area_l2443_244331

theorem line_through_point_with_triangle_area (x y : ℝ) :
  let P : ℝ × ℝ := (4/3, 2)
  let l : ℝ → ℝ → Prop := λ x y ↦ 6*x + 3*y - 14 = 0
  let A : ℝ × ℝ := (7/3, 0)
  let B : ℝ × ℝ := (0, 14/3)
  let O : ℝ × ℝ := (0, 0)
  l P.1 P.2 ∧
  l A.1 A.2 ∧
  l B.1 B.2 ∧
  A.1 > 0 ∧
  B.2 > 0 ∧
  (1/2 * A.1 * B.2 = 6) →
  l x y :=
by sorry

end NUMINAMATH_CALUDE_line_through_point_with_triangle_area_l2443_244331


namespace NUMINAMATH_CALUDE_chessboard_decomposition_l2443_244343

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  white_squares : Nat
  black_squares : Nat

/-- Represents a decomposition of the chessboard -/
def Decomposition := List Rectangle

/-- Checks if a decomposition is valid according to the given conditions -/
def is_valid_decomposition (d : Decomposition) : Prop :=
  d.all (λ r => r.white_squares = r.black_squares) ∧
  d.length > 0 ∧
  (List.zip d (List.tail d)).all (λ (r1, r2) => r1.white_squares < r2.white_squares) ∧
  (d.map (λ r => r.white_squares + r.black_squares)).sum = 64

/-- The main theorem to be proved -/
theorem chessboard_decomposition :
  (∃ (d : Decomposition), is_valid_decomposition d ∧ d.length = 7) ∧
  (∀ (d : Decomposition), is_valid_decomposition d → d.length ≤ 7) :=
sorry

end NUMINAMATH_CALUDE_chessboard_decomposition_l2443_244343


namespace NUMINAMATH_CALUDE_vehicle_distance_after_three_minutes_l2443_244364

/-- The distance between two vehicles after a given time -/
def distance_between_vehicles (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v2 - v1) * t

theorem vehicle_distance_after_three_minutes :
  let truck_speed : ℝ := 65
  let car_speed : ℝ := 85
  let time_minutes : ℝ := 3
  let time_hours : ℝ := time_minutes / 60
  distance_between_vehicles truck_speed car_speed time_hours = 1 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_distance_after_three_minutes_l2443_244364


namespace NUMINAMATH_CALUDE_tile_coverage_l2443_244365

/-- Represents the dimensions of a rectangle in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

theorem tile_coverage (tile : Dimensions) (region : Dimensions) : 
  tile.length = 2 ∧ tile.width = 6 ∧ 
  region.length = feetToInches 3 ∧ region.width = feetToInches 4 → 
  (area region / area tile : ℕ) = 144 := by
  sorry

#check tile_coverage

end NUMINAMATH_CALUDE_tile_coverage_l2443_244365


namespace NUMINAMATH_CALUDE_equilateral_triangle_product_l2443_244370

/-- Given an equilateral triangle with vertices at (0,0), (a,19), and (b,61), 
    the product ab equals 7760/9 -/
theorem equilateral_triangle_product (a b : ℝ) : 
  (∃ (z : ℂ), z ^ 3 = 1 ∧ z ≠ 1 ∧ a + 19 * I = (b + 61 * I) * z) →
  a * b = 7760 / 9 := by
sorry


end NUMINAMATH_CALUDE_equilateral_triangle_product_l2443_244370


namespace NUMINAMATH_CALUDE_toy_cost_price_l2443_244398

/-- The cost price of a toy -/
def cost_price : ℕ := sorry

/-- The number of toys sold -/
def toys_sold : ℕ := 18

/-- The total selling price of all toys -/
def total_selling_price : ℕ := 23100

/-- The number of toys whose cost price equals the gain -/
def gain_equivalent_toys : ℕ := 3

theorem toy_cost_price : 
  (toys_sold + gain_equivalent_toys) * cost_price = total_selling_price ∧ 
  cost_price = 1100 := by
  sorry

end NUMINAMATH_CALUDE_toy_cost_price_l2443_244398


namespace NUMINAMATH_CALUDE_trig_identity_proof_l2443_244386

theorem trig_identity_proof : 
  (Real.sin (47 * π / 180) - Real.sin (17 * π / 180) * Real.cos (30 * π / 180)) / 
  Real.sin (73 * π / 180) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_proof_l2443_244386


namespace NUMINAMATH_CALUDE_special_divisor_form_l2443_244303

/-- A function that checks if a number is of the form a^r + 1 --/
def isOfForm (d : ℕ) : Prop :=
  ∃ (a r : ℕ), a > 0 ∧ r > 1 ∧ d = a^r + 1

/-- The main theorem --/
theorem special_divisor_form (n : ℕ) :
  n > 1 ∧ (∀ d : ℕ, 1 < d ∧ d ∣ n → isOfForm d) →
  n = 10 ∨ ∃ a : ℕ, n = a^2 + 1 :=
sorry

end NUMINAMATH_CALUDE_special_divisor_form_l2443_244303


namespace NUMINAMATH_CALUDE_sally_total_spent_l2443_244337

-- Define the amounts spent on peaches and cherries
def peaches_cost : ℚ := 12.32
def cherries_cost : ℚ := 11.54

-- Define the total cost
def total_cost : ℚ := peaches_cost + cherries_cost

-- Theorem statement
theorem sally_total_spent : total_cost = 23.86 := by
  sorry

end NUMINAMATH_CALUDE_sally_total_spent_l2443_244337


namespace NUMINAMATH_CALUDE_number_greater_than_one_sixth_l2443_244389

theorem number_greater_than_one_sixth (x : ℝ) : x = 1/6 + 0.33333333333333337 → x = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_number_greater_than_one_sixth_l2443_244389


namespace NUMINAMATH_CALUDE_intersection_point_is_solution_l2443_244381

/-- Given two linear functions that intersect at a specific point,
    prove that this point is the solution to the system of equations. -/
theorem intersection_point_is_solution (b : ℝ) :
  (∃ (x y : ℝ), y = 3 * x - 5 ∧ y = 2 * x + b) →
  (1 : ℝ) = 3 * (1 : ℝ) - 5 →
  (-2 : ℝ) = 2 * (1 : ℝ) + b →
  (∀ (x y : ℝ), y = 3 * x - 5 ∧ y = 2 * x + b → x = 1 ∧ y = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_solution_l2443_244381


namespace NUMINAMATH_CALUDE_sum_of_central_angles_is_360_l2443_244357

/-- A circle with an inscribed pentagon -/
structure PentagonInCircle where
  /-- The circle -/
  circle : Set ℝ × Set ℝ
  /-- The inscribed pentagon -/
  pentagon : Set (ℝ × ℝ)
  /-- The center of the circle -/
  center : ℝ × ℝ
  /-- The vertices of the pentagon -/
  vertices : Fin 5 → ℝ × ℝ
  /-- The lines from vertices to center -/
  lines : Fin 5 → Set (ℝ × ℝ)

/-- The sum of angles at the center formed by lines from pentagon vertices to circle center -/
def sumOfCentralAngles (p : PentagonInCircle) : ℝ := sorry

/-- Theorem: The sum of central angles in a pentagon inscribed in a circle is 360° -/
theorem sum_of_central_angles_is_360 (p : PentagonInCircle) : 
  sumOfCentralAngles p = 360 := by sorry

end NUMINAMATH_CALUDE_sum_of_central_angles_is_360_l2443_244357


namespace NUMINAMATH_CALUDE_solution_set_eq_singleton_l2443_244350

/-- The solution set of the system of equations x + y = 1 and x^2 - y^2 = 9 -/
def solution_set : Set (ℝ × ℝ) :=
  {p | p.1 + p.2 = 1 ∧ p.1^2 - p.2^2 = 9}

/-- Theorem stating that the solution set contains only the point (5, -4) -/
theorem solution_set_eq_singleton :
  solution_set = {(5, -4)} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_eq_singleton_l2443_244350


namespace NUMINAMATH_CALUDE_smallest_b_value_l2443_244335

theorem smallest_b_value (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : b * b = a * c)  -- Geometric progression condition
  (h4 : a + b = 2 * c)  -- Arithmetic progression condition
  : b ≥ 2 ∧ ∃ (a' b' c' : ℤ), a' < b' ∧ b' < c' ∧ b' * b' = a' * c' ∧ a' + b' = 2 * c' ∧ b' = 2 :=
by sorry

#check smallest_b_value

end NUMINAMATH_CALUDE_smallest_b_value_l2443_244335


namespace NUMINAMATH_CALUDE_log_8_problem_l2443_244393

theorem log_8_problem (x : ℝ) :
  Real.log x / Real.log 8 = 3.5 → x = 1024 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_log_8_problem_l2443_244393


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2443_244317

theorem cos_alpha_value (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : 1 - 2 * Real.sin (2 * α) = Real.cos (2 * α)) : 
  Real.cos α = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2443_244317


namespace NUMINAMATH_CALUDE_range_of_x_l2443_244383

theorem range_of_x (x : ℝ) : 
  (x^2 - 2*x - 3 ≤ 0) → (1/(x-2) ≤ 0) → (-1 ≤ x ∧ x < 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2443_244383


namespace NUMINAMATH_CALUDE_polygon_angles_l2443_244352

theorem polygon_angles (n : ℕ) (h : n > 2) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_angles_l2443_244352


namespace NUMINAMATH_CALUDE_school_student_count_l2443_244314

/-- The number of teachers in the school -/
def num_teachers : ℕ := 9

/-- The number of additional students needed for equal distribution -/
def additional_students : ℕ := 4

/-- The current number of students in the school -/
def current_students : ℕ := 23

/-- Theorem stating that the current number of students is correct -/
theorem school_student_count :
  ∃ (k : ℕ), (current_students + additional_students) = num_teachers * k ∧
             ∀ (m : ℕ), m < current_students →
               ¬(∃ (j : ℕ), (m + additional_students) = num_teachers * j) :=
by sorry

end NUMINAMATH_CALUDE_school_student_count_l2443_244314


namespace NUMINAMATH_CALUDE_tom_flashlight_batteries_l2443_244363

/-- The number of batteries Tom used on his flashlights -/
def flashlight_batteries : ℕ := 19 - 15 - 2

/-- Proof that Tom used 4 batteries on his flashlights -/
theorem tom_flashlight_batteries :
  flashlight_batteries = 4 :=
by sorry

end NUMINAMATH_CALUDE_tom_flashlight_batteries_l2443_244363


namespace NUMINAMATH_CALUDE_lcm_problem_l2443_244300

theorem lcm_problem (a b : ℕ+) (h1 : Nat.gcd a b = 20) (h2 : a * b = 2560) :
  Nat.lcm a b = 128 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l2443_244300


namespace NUMINAMATH_CALUDE_original_line_length_l2443_244373

/-- Proves that the original length of a line is 1 meter -/
theorem original_line_length
  (erased_length : ℝ)
  (remaining_length : ℝ)
  (h1 : erased_length = 33)
  (h2 : remaining_length = 67)
  (h3 : (100 : ℝ) = (1 : ℝ) * 100) :
  erased_length + remaining_length = 100 := by
sorry

end NUMINAMATH_CALUDE_original_line_length_l2443_244373


namespace NUMINAMATH_CALUDE_smallest_common_factor_l2443_244341

theorem smallest_common_factor (n : ℕ) : n = 85 ↔ 
  (n > 0 ∧ 
   ∃ (k : ℕ), k > 1 ∧ k ∣ (11*n - 4) ∧ k ∣ (8*n + 6) ∧
   ∀ (m : ℕ), m < n → 
     (∀ (j : ℕ), j > 1 → ¬(j ∣ (11*m - 4) ∧ j ∣ (8*m + 6)))) := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_factor_l2443_244341


namespace NUMINAMATH_CALUDE_cannot_visit_all_friends_l2443_244309

-- Define the building structure
structure Building where
  num_floors : ℕ
  start_floor : ℕ
  friend_floors : List ℕ
  elevator_moves : List ℕ

-- Define the problem specifics
def problem : Building :=
  { num_floors := 14
  , start_floor := 1
  , friend_floors := [12, 14]
  , elevator_moves := [3, 7]
  }

-- Define a single elevator trip
def elevator_trip (current : ℤ) (move : ℤ) : ℤ :=
  current + move

-- Define if a floor is reachable within given moves
def is_reachable (building : Building) (target : ℕ) (max_moves : ℕ) : Prop :=
  ∃ (moves : List ℤ),
    moves.length ≤ max_moves ∧
    moves.all (λ m => m.natAbs ∈ building.elevator_moves) ∧
    (moves.foldl elevator_trip building.start_floor : ℤ) = target

-- Theorem statement
theorem cannot_visit_all_friends :
  ¬∃ (moves : List ℤ),
    moves.length ≤ 6 ∧
    moves.all (λ m => m.natAbs ∈ problem.elevator_moves) ∧
    (∀ floor ∈ problem.friend_floors,
      ∃ (submoves : List ℤ),
        submoves ⊆ moves ∧
        (submoves.foldl elevator_trip problem.start_floor : ℤ) = floor) :=
sorry

end NUMINAMATH_CALUDE_cannot_visit_all_friends_l2443_244309


namespace NUMINAMATH_CALUDE_paper_products_distribution_l2443_244392

theorem paper_products_distribution (total : ℕ) 
  (h1 : total = 20)
  (h2 : total / 2 + total / 4 + total / 5 + paper_cups = total) : 
  paper_cups = 1 := by
  sorry

end NUMINAMATH_CALUDE_paper_products_distribution_l2443_244392


namespace NUMINAMATH_CALUDE_angle_between_diagonals_l2443_244342

/-- 
Given a quadrilateral with area A, and diagonals d₁ and d₂, 
the angle α between the diagonals satisfies the equation:
A = (1/2) * d₁ * d₂ * sin(α)
-/
def quadrilateral_area_diagonals (A d₁ d₂ α : ℝ) : Prop :=
  A = (1/2) * d₁ * d₂ * Real.sin α

theorem angle_between_diagonals (A d₁ d₂ α : ℝ) 
  (h_area : A = 3)
  (h_diag1 : d₁ = 6)
  (h_diag2 : d₂ = 2)
  (h_quad : quadrilateral_area_diagonals A d₁ d₂ α) :
  α = π / 6 := by
  sorry

#check angle_between_diagonals

end NUMINAMATH_CALUDE_angle_between_diagonals_l2443_244342


namespace NUMINAMATH_CALUDE_min_value_of_function_l2443_244378

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  2 * x + 1 / x^6 ≥ 3 ∧ ∃ y > 0, 2 * y + 1 / y^6 = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2443_244378


namespace NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l2443_244305

theorem sin_cos_sum_equals_one : 
  Real.sin (15 * π / 180) * Real.cos (75 * π / 180) + 
  Real.cos (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_equals_one_l2443_244305


namespace NUMINAMATH_CALUDE_volleyball_points_proof_l2443_244327

theorem volleyball_points_proof (x : ℚ) : 
  x > 0 → 
  (1 / 3 : ℚ) * x + (3 / 8 : ℚ) * x + 21 + 18 = x → 
  ∀ y : ℚ, y ≤ 24 → 
  (1 / 3 : ℚ) * x + (3 / 8 : ℚ) * x + 21 + y = x → 
  y = 18 := by
sorry

end NUMINAMATH_CALUDE_volleyball_points_proof_l2443_244327


namespace NUMINAMATH_CALUDE_y_over_z_equals_negative_five_l2443_244348

theorem y_over_z_equals_negative_five 
  (x y z : ℚ) 
  (eq1 : x + y = 2*x + z) 
  (eq2 : x - 2*y = 4*z) 
  (eq3 : x + y + z = 21) : 
  y / z = -5 := by sorry

end NUMINAMATH_CALUDE_y_over_z_equals_negative_five_l2443_244348


namespace NUMINAMATH_CALUDE_quadratic_roots_existence_l2443_244332

theorem quadratic_roots_existence : ∃ (p q : ℝ), 
  ((p - 1)^2 - 4*q > 0) ∧ 
  ((p + 1)^2 - 4*q > 0) ∧ 
  (p^2 - 4*q < 0) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_existence_l2443_244332


namespace NUMINAMATH_CALUDE_sum_of_squares_and_cubes_l2443_244321

theorem sum_of_squares_and_cubes (a b : ℤ) (h : ∃ k : ℤ, a^2 - 4*b = k^2) :
  (∃ x y : ℤ, a^2 - 2*b = x^2 + y^2) ∧
  (∃ u v : ℤ, 3*a*b - a^3 = u^3 + v^3) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_cubes_l2443_244321


namespace NUMINAMATH_CALUDE_frequency_histogram_interval_length_l2443_244351

/-- Given a frequency histogram interval [a,b), prove that its length |a-b| equals m/h,
    where m is the frequency and h is the histogram height for this interval. -/
theorem frequency_histogram_interval_length
  (a b m h : ℝ)
  (h_interval : a < b)
  (h_frequency : m > 0)
  (h_height : h > 0)
  (h_histogram : h = m / (b - a)) :
  b - a = m / h :=
sorry

end NUMINAMATH_CALUDE_frequency_histogram_interval_length_l2443_244351


namespace NUMINAMATH_CALUDE_log_calculation_l2443_244355

-- Define the common logarithm
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem log_calculation :
  (lg 2)^2 + lg 2 * lg 50 + lg 25 = 2 :=
by
  -- Properties of logarithms
  have h1 : lg 50 = lg 2 + lg 25 := by sorry
  have h2 : lg 25 = 2 * lg 5 := by sorry
  have h3 : lg 10 = 1 := by sorry
  
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_log_calculation_l2443_244355


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_2000_l2443_244302

theorem greatest_multiple_of_four_under_cube_root_2000 :
  ∃ (x : ℕ), x > 0 ∧ 4 ∣ x ∧ x^3 < 2000 ∧
  ∀ (y : ℕ), y > 0 → 4 ∣ y → y^3 < 2000 → y ≤ x :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_under_cube_root_2000_l2443_244302


namespace NUMINAMATH_CALUDE_total_cost_is_2200_l2443_244371

/-- The total cost of buying one smartphone, one personal computer, and one advanced tablet -/
def total_cost (smartphone_price : ℕ) (pc_price_difference : ℕ) : ℕ :=
  let pc_price := smartphone_price + pc_price_difference
  let tablet_price := smartphone_price + pc_price
  smartphone_price + pc_price + tablet_price

/-- Proof that the total cost is $2200 given the specified prices -/
theorem total_cost_is_2200 :
  total_cost 300 500 = 2200 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_2200_l2443_244371


namespace NUMINAMATH_CALUDE_greatest_root_of_f_l2443_244395

def f (x : ℝ) := 12 * x^4 - 8 * x^2 + 1

theorem greatest_root_of_f :
  ∃ (r : ℝ), r = Real.sqrt 2 / 2 ∧ 
  f r = 0 ∧ 
  ∀ (x : ℝ), f x = 0 → x ≤ r :=
sorry

end NUMINAMATH_CALUDE_greatest_root_of_f_l2443_244395


namespace NUMINAMATH_CALUDE_x_neq_one_necessary_not_sufficient_l2443_244336

theorem x_neq_one_necessary_not_sufficient :
  (∃ x : ℝ, x ≠ 1 ∧ x^2 - 3*x + 2 = 0) ∧
  (∀ x : ℝ, x^2 - 3*x + 2 ≠ 0 → x ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_x_neq_one_necessary_not_sufficient_l2443_244336


namespace NUMINAMATH_CALUDE_essay_time_theorem_l2443_244345

/-- Represents the time spent on various activities during essay writing -/
structure EssayWritingTime where
  wordsPerPage : ℕ
  timePerPageFirstDraft : ℕ
  researchTime : ℕ
  outlineTime : ℕ
  brainstormTime : ℕ
  firstDraftPages : ℕ
  timePerPageSecondDraft : ℕ
  breakTimePerPage : ℕ
  editingTime : ℕ
  proofreadingTime : ℕ

/-- Calculates the total time spent on writing the essay -/
def totalEssayTime (t : EssayWritingTime) : ℕ :=
  t.researchTime +
  t.outlineTime * 60 +
  t.brainstormTime +
  t.firstDraftPages * t.timePerPageFirstDraft +
  (t.firstDraftPages - 1) * t.breakTimePerPage +
  t.firstDraftPages * t.timePerPageSecondDraft +
  t.editingTime +
  t.proofreadingTime

/-- Theorem stating that the total time spent on the essay is 34900 seconds -/
theorem essay_time_theorem (t : EssayWritingTime)
  (h1 : t.wordsPerPage = 500)
  (h2 : t.timePerPageFirstDraft = 1800)
  (h3 : t.researchTime = 2700)
  (h4 : t.outlineTime = 15)
  (h5 : t.brainstormTime = 1200)
  (h6 : t.firstDraftPages = 6)
  (h7 : t.timePerPageSecondDraft = 1500)
  (h8 : t.breakTimePerPage = 600)
  (h9 : t.editingTime = 4500)
  (h10 : t.proofreadingTime = 1800) :
  totalEssayTime t = 34900 := by
  sorry

#eval totalEssayTime {
  wordsPerPage := 500,
  timePerPageFirstDraft := 1800,
  researchTime := 2700,
  outlineTime := 15,
  brainstormTime := 1200,
  firstDraftPages := 6,
  timePerPageSecondDraft := 1500,
  breakTimePerPage := 600,
  editingTime := 4500,
  proofreadingTime := 1800
}

end NUMINAMATH_CALUDE_essay_time_theorem_l2443_244345


namespace NUMINAMATH_CALUDE_only_negative_one_squared_is_negative_l2443_244304

theorem only_negative_one_squared_is_negative : 
  ((-1 : ℝ)^0 < 0 ∨ |(-1 : ℝ)| < 0 ∨ Real.sqrt 1 < 0 ∨ -(1 : ℝ)^2 < 0) ∧
  ((-1 : ℝ)^0 ≥ 0 ∧ |(-1 : ℝ)| ≥ 0 ∧ Real.sqrt 1 ≥ 0) ∧
  (-(1 : ℝ)^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_only_negative_one_squared_is_negative_l2443_244304


namespace NUMINAMATH_CALUDE_man_speed_against_current_l2443_244334

/-- Given a man's speed with the current and the speed of the current, 
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current speed_of_current : ℝ) : ℝ :=
  speed_with_current - 2 * speed_of_current

/-- Theorem stating that for the given conditions, 
    the man's speed against the current is 14 km/h. -/
theorem man_speed_against_current :
  speed_against_current 20 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_against_current_l2443_244334


namespace NUMINAMATH_CALUDE_root_difference_square_range_l2443_244307

/-- Given a quadratic equation with two distinct real roots, 
    prove that the square of the difference of the roots has a specific range -/
theorem root_difference_square_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 - 2*a*x₁ + 2*a^2 - 3*a + 2 = 0 ∧
   x₂^2 - 2*a*x₂ + 2*a^2 - 3*a + 2 = 0) →
  ∃ y : ℝ, 0 < y ∧ y ≤ 1 ∧
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ →
    x₁^2 - 2*a*x₁ + 2*a^2 - 3*a + 2 = 0 →
    x₂^2 - 2*a*x₂ + 2*a^2 - 3*a + 2 = 0 →
    (x₁ - x₂)^2 = y :=
by sorry

end NUMINAMATH_CALUDE_root_difference_square_range_l2443_244307


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l2443_244319

theorem solution_implies_a_value (x a : ℝ) : x = 1 ∧ 2 * x - a = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l2443_244319


namespace NUMINAMATH_CALUDE_cubic_root_transformation_l2443_244326

theorem cubic_root_transformation (p : ℝ) : 
  p^3 + p - 3 = 0 → (p^2)^3 + 2*(p^2)^2 + p^2 - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_transformation_l2443_244326


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l2443_244358

theorem right_triangle_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
  (pythagorean : a^2 + b^2 = c^2) : (a + b) / Real.sqrt 2 ≤ c := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l2443_244358


namespace NUMINAMATH_CALUDE_inequality_proof_l2443_244344

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c) (non_neg_d : 0 ≤ d)
  (sum_condition : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1/3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2443_244344


namespace NUMINAMATH_CALUDE_mixture_concentration_l2443_244396

/-- Represents a vessel with spirit -/
structure Vessel where
  concentration : Rat
  volume : Rat

/-- Calculates the concentration of spirit in a mixture of vessels -/
def mixConcentration (vessels : List Vessel) : Rat :=
  let totalSpirit := vessels.map (λ v => v.concentration * v.volume) |>.sum
  let totalVolume := vessels.map (λ v => v.volume) |>.sum
  totalSpirit / totalVolume

/-- The main theorem stating that the mixture of given vessels results in 26% concentration -/
theorem mixture_concentration : 
  let vessels := [
    Vessel.mk (45/100) 4,
    Vessel.mk (30/100) 5,
    Vessel.mk (10/100) 6
  ]
  mixConcentration vessels = 26/100 := by
  sorry


end NUMINAMATH_CALUDE_mixture_concentration_l2443_244396


namespace NUMINAMATH_CALUDE_average_selling_price_l2443_244366

def initial_stock : ℝ := 100
def morning_sale_weight : ℝ := 50
def morning_sale_price : ℝ := 1.2
def noon_sale_weight : ℝ := 30
def noon_sale_price : ℝ := 1
def afternoon_sale_weight : ℝ := 20
def afternoon_sale_price : ℝ := 0.8

theorem average_selling_price :
  let total_revenue := morning_sale_weight * morning_sale_price +
                       noon_sale_weight * noon_sale_price +
                       afternoon_sale_weight * afternoon_sale_price
  let total_weight := morning_sale_weight + noon_sale_weight + afternoon_sale_weight
  total_revenue / total_weight = 1.06 := by
  sorry

end NUMINAMATH_CALUDE_average_selling_price_l2443_244366


namespace NUMINAMATH_CALUDE_unique_two_digit_integer_l2443_244323

theorem unique_two_digit_integer (t : ℕ) : 
  (10 ≤ t ∧ t < 100) ∧ (11 * t) % 100 = 36 ↔ t = 76 := by sorry

end NUMINAMATH_CALUDE_unique_two_digit_integer_l2443_244323


namespace NUMINAMATH_CALUDE_survey_response_rate_change_l2443_244340

theorem survey_response_rate_change 
  (original_customers : Nat) 
  (original_responses : Nat)
  (final_customers : Nat)
  (final_responses : Nat)
  (h1 : original_customers = 100)
  (h2 : original_responses = 10)
  (h3 : final_customers = 90)
  (h4 : final_responses = 27) :
  ((final_responses : ℝ) / final_customers - (original_responses : ℝ) / original_customers) / 
  ((original_responses : ℝ) / original_customers) * 100 = 200 := by
sorry

end NUMINAMATH_CALUDE_survey_response_rate_change_l2443_244340


namespace NUMINAMATH_CALUDE_billy_coins_problem_l2443_244380

theorem billy_coins_problem (total_coins : Nat) (quarter_piles : Nat) (dime_piles : Nat) 
  (h1 : total_coins = 20)
  (h2 : quarter_piles = 2)
  (h3 : dime_piles = 3) :
  ∃! coins_per_pile : Nat, 
    coins_per_pile > 0 ∧ 
    quarter_piles * coins_per_pile + dime_piles * coins_per_pile = total_coins ∧
    coins_per_pile = 4 := by
  sorry

end NUMINAMATH_CALUDE_billy_coins_problem_l2443_244380


namespace NUMINAMATH_CALUDE_moles_of_MgO_formed_l2443_244390

-- Define the chemical elements and compounds
inductive Chemical
| Mg
| CO2
| MgO
| C

-- Define a structure to represent a chemical equation
structure ChemicalEquation :=
  (reactants : List (Chemical × ℕ))
  (products : List (Chemical × ℕ))

-- Define the balanced chemical equation
def balancedEquation : ChemicalEquation :=
  { reactants := [(Chemical.Mg, 2), (Chemical.CO2, 1)]
  , products := [(Chemical.MgO, 2), (Chemical.C, 1)] }

-- Define the available moles of reactants
def availableMg : ℕ := 2
def availableCO2 : ℕ := 1

-- Theorem to prove
theorem moles_of_MgO_formed :
  availableMg = 2 →
  availableCO2 = 1 →
  (balancedEquation.reactants.map (λ (c, n) => (c, n)) = [(Chemical.Mg, 2), (Chemical.CO2, 1)]) →
  (balancedEquation.products.map (λ (c, n) => (c, n)) = [(Chemical.MgO, 2), (Chemical.C, 1)]) →
  ∃ (molesOfMgO : ℕ), molesOfMgO = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_moles_of_MgO_formed_l2443_244390


namespace NUMINAMATH_CALUDE_clock_cost_price_l2443_244369

/-- Represents the cost price of a single clock -/
def cost_price : ℝ := 125

/-- The total number of clocks -/
def total_clocks : ℕ := 150

/-- The number of clocks sold at 12% profit -/
def clocks_at_12_percent : ℕ := 60

/-- The number of clocks sold at 18% profit -/
def clocks_at_18_percent : ℕ := 90

/-- The profit percentage for the first group of clocks -/
def profit_12_percent : ℝ := 0.12

/-- The profit percentage for the second group of clocks -/
def profit_18_percent : ℝ := 0.18

/-- The uniform profit percentage if all clocks were sold at the same profit -/
def uniform_profit_percent : ℝ := 0.16

/-- The difference in profit between the actual sales and the hypothetical uniform profit sales -/
def profit_difference : ℝ := 75

theorem clock_cost_price :
  (clocks_at_12_percent : ℝ) * cost_price * (1 + profit_12_percent) +
  (clocks_at_18_percent : ℝ) * cost_price * (1 + profit_18_percent) =
  (total_clocks : ℝ) * cost_price * (1 + uniform_profit_percent) + profit_difference :=
by sorry

end NUMINAMATH_CALUDE_clock_cost_price_l2443_244369


namespace NUMINAMATH_CALUDE_liters_conversion_hours_conversion_cubic_meters_conversion_l2443_244311

-- Define conversion factors
def liters_to_milliliters : ℝ := 1000
def hours_per_day : ℝ := 24
def cubic_meters_to_cubic_centimeters : ℝ := 1000000

-- Theorem for 9.12 liters conversion
theorem liters_conversion (x : ℝ) (h : x = 9.12) :
  ∃ (l m : ℝ), x * liters_to_milliliters = l * liters_to_milliliters + m ∧ l = 9 ∧ m = 120 :=
sorry

-- Theorem for 4 hours conversion
theorem hours_conversion (x : ℝ) (h : x = 4) :
  x / hours_per_day = 1 / 6 :=
sorry

-- Theorem for 0.25 cubic meters conversion
theorem cubic_meters_conversion (x : ℝ) (h : x = 0.25) :
  x * cubic_meters_to_cubic_centimeters = 250000 :=
sorry

end NUMINAMATH_CALUDE_liters_conversion_hours_conversion_cubic_meters_conversion_l2443_244311


namespace NUMINAMATH_CALUDE_saltwater_volume_proof_l2443_244353

theorem saltwater_volume_proof (x : ℝ) : 
  (0.20 * x + 12) / (0.75 * x + 18) = 1/3 → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_saltwater_volume_proof_l2443_244353


namespace NUMINAMATH_CALUDE_range_of_a_part1_range_of_a_part2_l2443_244318

def proposition_p (a : ℝ) : Prop := a^2 - 5*a - 6 > 0

def proposition_q (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧
  x₁^2 + a*x₁ + 1 = 0 ∧ x₂^2 + a*x₂ + 1 = 0

theorem range_of_a_part1 :
  {a : ℝ | proposition_p a} = {a | a < -1 ∨ a > 6} :=
sorry

theorem range_of_a_part2 :
  {a : ℝ | (proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a)} =
  {a | a < -1 ∨ (2 < a ∧ a ≤ 6)} :=
sorry

end NUMINAMATH_CALUDE_range_of_a_part1_range_of_a_part2_l2443_244318


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sufficient_not_necessary_l2443_244324

/-- Definition of an arithmetic sequence -/
def is_arithmetic_sequence (s : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, s (n + 1) - s n = d

/-- Given sequences a and b with the relation b_n = a_n + a_{n+1} -/
def sequence_relation (a b : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, b n = a n + a (n + 1)

/-- Theorem stating that {a_n} being arithmetic is sufficient but not necessary for {b_n} to be arithmetic -/
theorem arithmetic_sequence_sufficient_not_necessary
  (a b : ℕ → ℝ) (h : sequence_relation a b) :
  (is_arithmetic_sequence a → is_arithmetic_sequence b) ∧
  ¬(is_arithmetic_sequence b → is_arithmetic_sequence a) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_sufficient_not_necessary_l2443_244324


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_l2443_244356

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem smallest_sum_of_digits :
  (∀ n : ℕ, sum_of_digits (3 * n^2 + n + 1) ≥ 3) ∧
  (∃ n : ℕ, sum_of_digits (3 * n^2 + n + 1) = 3) := by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_l2443_244356


namespace NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l2443_244308

theorem remainder_444_power_444_mod_13 : 444^444 % 13 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_444_mod_13_l2443_244308


namespace NUMINAMATH_CALUDE_julia_watch_collection_l2443_244375

theorem julia_watch_collection (silver_watches : ℕ) (bronze_watches : ℕ) (gold_watches : ℕ) : 
  silver_watches = 20 →
  bronze_watches = 3 * silver_watches →
  gold_watches = (silver_watches + bronze_watches) / 10 →
  silver_watches + bronze_watches + gold_watches = 88 := by
  sorry

end NUMINAMATH_CALUDE_julia_watch_collection_l2443_244375


namespace NUMINAMATH_CALUDE_simplify_polynomial_subtraction_l2443_244382

theorem simplify_polynomial_subtraction (r : ℝ) : 
  (r^2 + 3*r - 2) - (r^2 + 7*r - 5) = -4*r + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_subtraction_l2443_244382


namespace NUMINAMATH_CALUDE_pet_shop_dogs_count_l2443_244399

/-- Given a pet shop with dogs, cats, and bunnies, where the ratio of dogs to cats to bunnies
    is 7 : 7 : 8, and the total number of dogs and bunnies is 330, prove that there are 154 dogs. -/
theorem pet_shop_dogs_count : ℕ → ℕ → ℕ → Prop :=
  fun dogs cats bunnies =>
    (dogs : ℚ) / cats = 1 →
    (dogs : ℚ) / bunnies = 7 / 8 →
    dogs + bunnies = 330 →
    dogs = 154

/-- Proof of the pet_shop_dogs_count theorem -/
lemma prove_pet_shop_dogs_count : ∃ dogs cats bunnies, pet_shop_dogs_count dogs cats bunnies :=
  sorry

end NUMINAMATH_CALUDE_pet_shop_dogs_count_l2443_244399


namespace NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l2443_244322

/-- Proves that the total amount of ethanol in a fuel tank is 30 gallons given specific conditions -/
theorem ethanol_in_fuel_tank (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) 
  (h1 : tank_capacity = 214)
  (h2 : fuel_a_volume = 106)
  (h3 : fuel_a_ethanol_percent = 0.12)
  (h4 : fuel_b_ethanol_percent = 0.16) :
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let total_ethanol := fuel_a_volume * fuel_a_ethanol_percent + fuel_b_volume * fuel_b_ethanol_percent
  total_ethanol = 30 := by
sorry


end NUMINAMATH_CALUDE_ethanol_in_fuel_tank_l2443_244322


namespace NUMINAMATH_CALUDE_unique_solution_l2443_244339

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : Nat
  h1 : value ≥ 100
  h2 : value ≤ 999

/-- Check if a number has distinct digits in ascending order -/
def hasDistinctAscendingDigits (n : ThreeDigitNumber) : Prop :=
  let d1 := n.value / 100
  let d2 := (n.value / 10) % 10
  let d3 := n.value % 10
  d1 < d2 ∧ d2 < d3

/-- Check if all words in the name of a number start with the same letter -/
def allWordsSameInitial (n : ThreeDigitNumber) : Prop :=
  sorry

/-- Check if a number has identical digits -/
def hasIdenticalDigits (n : ThreeDigitNumber) : Prop :=
  let d1 := n.value / 100
  let d2 := (n.value / 10) % 10
  let d3 := n.value % 10
  d1 = d2 ∧ d2 = d3

/-- Check if all words in the name of a number start with different letters -/
def allWordsDifferentInitials (n : ThreeDigitNumber) : Prop :=
  sorry

/-- The main theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! (n1 n2 : ThreeDigitNumber),
    (hasDistinctAscendingDigits n1 ∧ allWordsSameInitial n1) ∧
    (hasIdenticalDigits n2 ∧ allWordsDifferentInitials n2) ∧
    n1.value = 147 ∧ n2.value = 111 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2443_244339


namespace NUMINAMATH_CALUDE_zoes_purchase_cost_l2443_244349

/-- The total cost of Zoe's purchase for herself and her family -/
def total_cost (num_people : ℕ) (soda_cost pizza_cost icecream_cost topping_cost : ℚ) 
  (num_toppings icecream_per_person : ℕ) : ℚ :=
  let soda_total := num_people * soda_cost
  let pizza_total := num_people * (pizza_cost + num_toppings * topping_cost)
  let icecream_total := num_people * icecream_per_person * icecream_cost
  soda_total + pizza_total + icecream_total

/-- Theorem stating that Zoe's total purchase cost is $54.00 -/
theorem zoes_purchase_cost :
  total_cost 6 0.5 1 3 0.75 2 2 = 54 := by
  sorry

end NUMINAMATH_CALUDE_zoes_purchase_cost_l2443_244349


namespace NUMINAMATH_CALUDE_wheel_radii_theorem_l2443_244329

/-- The ratio of revolutions per minute of wheel A to wheel B -/
def revolution_ratio : ℚ := 1200 / 1500

/-- The total length from the outer radius of wheel A to the outer radius of wheel B in cm -/
def total_length : ℝ := 9

/-- The radius of wheel A in cm -/
def radius_A : ℝ := 2.5

/-- The radius of wheel B in cm -/
def radius_B : ℝ := 2

theorem wheel_radii_theorem :
  revolution_ratio = 4 / 5 ∧
  2 * (radius_A + radius_B) = total_length ∧
  radius_A * 4 = radius_B * 5 := by
  sorry

end NUMINAMATH_CALUDE_wheel_radii_theorem_l2443_244329


namespace NUMINAMATH_CALUDE_contrapositive_inequality_l2443_244330

theorem contrapositive_inequality (a b c : ℝ) :
  (¬(a + c < b + c) → ¬(a < b)) ↔ (a < b → a + c < b + c) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_inequality_l2443_244330


namespace NUMINAMATH_CALUDE_exists_number_plus_digit_sum_equals_2014_l2443_244379

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a natural number κ such that κ plus the sum of its digits equals 2014 -/
theorem exists_number_plus_digit_sum_equals_2014 : ∃ κ : ℕ, κ + sum_of_digits κ = 2014 := by
  sorry

end NUMINAMATH_CALUDE_exists_number_plus_digit_sum_equals_2014_l2443_244379


namespace NUMINAMATH_CALUDE_room_ratios_l2443_244312

/-- Represents a rectangular room with given length and width. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Represents a ratio as a pair of natural numbers. -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

theorem room_ratios (room : Rectangle) 
    (h1 : room.length = 24) 
    (h2 : room.width = 14) : 
    (∃ r1 : Ratio, r1.numerator = 6 ∧ r1.denominator = 19 ∧ 
      r1.numerator * perimeter room = r1.denominator * room.length) ∧
    (∃ r2 : Ratio, r2.numerator = 7 ∧ r2.denominator = 38 ∧ 
      r2.numerator * perimeter room = r2.denominator * room.width) := by
  sorry


end NUMINAMATH_CALUDE_room_ratios_l2443_244312


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_diagonal_length_l2443_244384

-- Define the cyclic quadrilateral ABCD and point K
variable (A B C D K : Point)

-- Define the property of being a cyclic quadrilateral
def is_cyclic_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the property of K being the intersection of diagonals
def is_diagonal_intersection (A B C D K : Point) : Prop := sorry

-- Define the distance between two points
def distance (P Q : Point) : ℝ := sorry

-- State the theorem
theorem cyclic_quadrilateral_diagonal_length
  (h_cyclic : is_cyclic_quadrilateral A B C D)
  (h_diagonal : is_diagonal_intersection A B C D K)
  (h_equal_sides : distance A B = distance B C)
  (h_BK : distance B K = b)
  (h_DK : distance D K = d)
  : distance A B = Real.sqrt (b^2 + b*d) := by sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_diagonal_length_l2443_244384


namespace NUMINAMATH_CALUDE_air_quality_probability_l2443_244388

theorem air_quality_probability (p_one_day p_two_days : ℝ) 
  (h1 : p_one_day = 0.8)
  (h2 : p_two_days = 0.6) :
  p_two_days / p_one_day = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_air_quality_probability_l2443_244388


namespace NUMINAMATH_CALUDE_abc_remainder_mod_9_l2443_244346

theorem abc_remainder_mod_9 (a b c : ℕ) 
  (ha : a < 9) (hb : b < 9) (hc : c < 9)
  (cong1 : a + 2*b + 3*c ≡ 0 [ZMOD 9])
  (cong2 : 2*a + 3*b + c ≡ 5 [ZMOD 9])
  (cong3 : 3*a + b + 2*c ≡ 5 [ZMOD 9]) :
  a * b * c ≡ 0 [ZMOD 9] := by
sorry

end NUMINAMATH_CALUDE_abc_remainder_mod_9_l2443_244346


namespace NUMINAMATH_CALUDE_adult_ticket_price_l2443_244301

/-- Given information about ticket sales, prove the price of an adult ticket --/
theorem adult_ticket_price
  (student_price : ℝ)
  (total_tickets : ℕ)
  (total_revenue : ℝ)
  (student_tickets : ℕ)
  (h1 : student_price = 2.5)
  (h2 : total_tickets = 59)
  (h3 : total_revenue = 222.5)
  (h4 : student_tickets = 9) :
  (total_revenue - student_price * student_tickets) / (total_tickets - student_tickets) = 4 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_price_l2443_244301


namespace NUMINAMATH_CALUDE_g_of_2_equals_6_l2443_244316

/-- The function g defined as g(x) = x³ - 2 for all real x -/
def g (x : ℝ) : ℝ := x^3 - 2

/-- Theorem stating that g(2) = 6 -/
theorem g_of_2_equals_6 : g 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_g_of_2_equals_6_l2443_244316


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2443_244374

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, x^2 + a*x + b < 0 ↔ 2 < x ∧ x < 4) → b - a = 14 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2443_244374


namespace NUMINAMATH_CALUDE_expression_simplification_l2443_244391

theorem expression_simplification (x : ℝ) : (x + 1)^2 + 2*(x + 1)*(5 - x) + (5 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2443_244391


namespace NUMINAMATH_CALUDE_even_number_less_than_square_l2443_244325

theorem even_number_less_than_square (m : ℕ) (h1 : m > 1) (h2 : Even m) : m < m^2 := by
  sorry

end NUMINAMATH_CALUDE_even_number_less_than_square_l2443_244325


namespace NUMINAMATH_CALUDE_janets_class_size_l2443_244376

/-- The number of children in Janet's class -/
def num_children : ℕ := 35

/-- The number of chaperones -/
def num_chaperones : ℕ := 5

/-- The number of additional lunches -/
def additional_lunches : ℕ := 3

/-- The cost of each lunch in dollars -/
def lunch_cost : ℕ := 7

/-- The total cost of all lunches in dollars -/
def total_cost : ℕ := 308

theorem janets_class_size :
  num_children + num_chaperones + 1 + additional_lunches = total_cost / lunch_cost :=
sorry

end NUMINAMATH_CALUDE_janets_class_size_l2443_244376


namespace NUMINAMATH_CALUDE_min_sort_steps_l2443_244333

/-- Represents the color of a cow -/
inductive Color
| Purple
| White

/-- A configuration of cows -/
def Configuration (n : ℕ) := Fin (2 * n) → Color

/-- A valid swap operation on a configuration -/
def ValidSwap (n : ℕ) (c : Configuration n) (i j : ℕ) : Prop :=
  i < j ∧ j ≤ 2 * n ∧ j - i = 2 * n - j

/-- The number of steps required to sort a configuration -/
def SortSteps (n : ℕ) (c : Configuration n) : ℕ := sorry

/-- The theorem stating that n steps are always sufficient and sometimes necessary -/
theorem min_sort_steps (n : ℕ) :
  (∀ c : Configuration n, SortSteps n c ≤ n) ∧
  (∃ c : Configuration n, SortSteps n c = n) := by sorry

end NUMINAMATH_CALUDE_min_sort_steps_l2443_244333


namespace NUMINAMATH_CALUDE_abs_a_minus_b_equals_eight_l2443_244315

theorem abs_a_minus_b_equals_eight (a b : ℝ) (h1 : a * b = 9) (h2 : a + b = 10) : 
  |a - b| = 8 := by
sorry

end NUMINAMATH_CALUDE_abs_a_minus_b_equals_eight_l2443_244315


namespace NUMINAMATH_CALUDE_susan_chairs_l2443_244368

theorem susan_chairs (total : ℕ) (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 43 →
  red = 5 →
  yellow = 4 * red →
  total = red + yellow + blue →
  blue = 18 := by
sorry

end NUMINAMATH_CALUDE_susan_chairs_l2443_244368


namespace NUMINAMATH_CALUDE_third_month_sale_l2443_244394

def average_sale : ℕ := 7500
def num_months : ℕ := 6
def sale_month1 : ℕ := 7435
def sale_month2 : ℕ := 7927
def sale_month4 : ℕ := 8230
def sale_month5 : ℕ := 7562
def sale_month6 : ℕ := 5991

theorem third_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := sale_month1 + sale_month2 + sale_month4 + sale_month5 + sale_month6
  total_sales - known_sales = 7855 := by
sorry

end NUMINAMATH_CALUDE_third_month_sale_l2443_244394


namespace NUMINAMATH_CALUDE_book_rearrangement_combinations_l2443_244385

/-- The number of options for each day of the week --/
def daily_options : List Nat := [1, 2, 3, 3, 2]

/-- The total number of combinations --/
def total_combinations : Nat := daily_options.prod

/-- Theorem stating that the total number of combinations is 36 --/
theorem book_rearrangement_combinations :
  total_combinations = 36 := by
  sorry

end NUMINAMATH_CALUDE_book_rearrangement_combinations_l2443_244385


namespace NUMINAMATH_CALUDE_not_perfect_square_l2443_244313

theorem not_perfect_square (a b : ℕ+) : ¬∃ k : ℤ, (a : ℤ)^2 + Int.ceil ((4 * (a : ℤ)^2) / (b : ℤ)) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l2443_244313


namespace NUMINAMATH_CALUDE_angle_bisector_length_bound_l2443_244367

theorem angle_bisector_length_bound (a b : ℝ) (θ : ℝ) (h1 : a = 10) (h2 : b = 15) (h3 : 0 < θ) (h4 : θ < π) :
  (2 * a * b * Real.cos (θ / 2)) / (a + b) < 12 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_length_bound_l2443_244367


namespace NUMINAMATH_CALUDE_running_speed_calculation_l2443_244338

/-- Proves that given the specified conditions, the running speed must be 6 mph -/
theorem running_speed_calculation (total_distance : ℝ) (running_time : ℝ) (walking_speed : ℝ) (walking_time : ℝ)
  (h1 : total_distance = 3)
  (h2 : running_time = 20 / 60)
  (h3 : walking_speed = 2)
  (h4 : walking_time = 30 / 60) :
  ∃ (running_speed : ℝ), running_speed * running_time + walking_speed * walking_time = total_distance ∧ running_speed = 6 := by
  sorry

end NUMINAMATH_CALUDE_running_speed_calculation_l2443_244338


namespace NUMINAMATH_CALUDE_fenced_square_cost_l2443_244347

/-- A square with fenced sides -/
structure FencedSquare where
  side_cost : ℕ
  sides : ℕ

/-- The total cost of fencing a square -/
def total_fencing_cost (s : FencedSquare) : ℕ :=
  s.side_cost * s.sides

/-- Theorem: The total cost of fencing a square with 4 sides at $69 per side is $276 -/
theorem fenced_square_cost :
  ∀ (s : FencedSquare), s.side_cost = 69 → s.sides = 4 → total_fencing_cost s = 276 :=
by
  sorry

end NUMINAMATH_CALUDE_fenced_square_cost_l2443_244347


namespace NUMINAMATH_CALUDE_birth_year_property_l2443_244362

def current_year : Nat := 2023
def birth_year : Nat := 1957

def sum_of_digits (n : Nat) : Nat :=
  let digits := n.repr.data.map (λ c => c.toNat - '0'.toNat)
  digits.sum

theorem birth_year_property : 
  current_year - birth_year = sum_of_digits birth_year := by
  sorry

end NUMINAMATH_CALUDE_birth_year_property_l2443_244362


namespace NUMINAMATH_CALUDE_time_to_find_two_artifacts_l2443_244306

/-- The time it takes to find two artifacts given research and expedition times for the first, 
    and a multiplier for the second. -/
def time_to_find_artifacts (research_time : ℝ) (expedition_time : ℝ) (multiplier : ℝ) : ℝ :=
  let first_artifact_time := research_time + expedition_time
  let second_artifact_time := multiplier * first_artifact_time
  first_artifact_time + second_artifact_time

/-- Theorem stating that under the given conditions, it takes 10 years to find both artifacts. -/
theorem time_to_find_two_artifacts : 
  time_to_find_artifacts 0.5 2 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_time_to_find_two_artifacts_l2443_244306


namespace NUMINAMATH_CALUDE_power_relationship_l2443_244360

theorem power_relationship (a b c : ℝ) 
  (ha : a = Real.rpow 0.8 5.2)
  (hb : b = Real.rpow 0.8 5.5)
  (hc : c = Real.rpow 5.2 0.1) : 
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_power_relationship_l2443_244360


namespace NUMINAMATH_CALUDE_percentage_problem_l2443_244397

theorem percentage_problem (x : ℝ) (h : 45 = 25 / 100 * x) : x = 180 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2443_244397


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l2443_244328

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 + 4*y + m = 0 → y = x) → 
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l2443_244328


namespace NUMINAMATH_CALUDE_intersection_A_B_l2443_244354

-- Define set A
def A : Set ℝ := {x | -2 < x ∧ x < 4}

-- Define set B
def B : Set ℝ := {2, 3, 4, 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2443_244354


namespace NUMINAMATH_CALUDE_calculate_expression_l2443_244372

def A (n k : ℕ) : ℕ := n * (n - 1) * (n - 2)

def C (n k : ℕ) : ℕ := n * (n - 1) * (n - 2) / (3 * 2 * 1)

theorem calculate_expression : (3 * A 5 3 + 4 * C 6 3) / (3 * 2 * 1) = 130 / 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2443_244372


namespace NUMINAMATH_CALUDE_cubic_equation_solvable_l2443_244359

theorem cubic_equation_solvable (a b c d : ℝ) (ha : a ≠ 0) :
  ∃ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ∧
  (∃ (e f g h i j k l m n : ℝ),
    x = (e * (f^(1/3)) + g * (h^(1/3)) + i * (j^(1/3)) + k) /
        (l * (m^(1/2)) + n)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solvable_l2443_244359


namespace NUMINAMATH_CALUDE_eleven_distinct_points_l2443_244361

/-- Represents a circular track with a cyclist and pedestrian -/
structure Track where
  length : ℝ
  pedestrian_speed : ℝ
  cyclist_speed : ℝ
  (cyclist_faster : cyclist_speed = pedestrian_speed * 1.55)
  (positive_speed : pedestrian_speed > 0)

/-- Calculates the number of distinct overtaking points on the track -/
def distinct_overtaking_points (track : Track) : ℕ :=
  sorry

/-- Theorem stating that there are 11 distinct overtaking points -/
theorem eleven_distinct_points (track : Track) :
  distinct_overtaking_points track = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_distinct_points_l2443_244361


namespace NUMINAMATH_CALUDE_zoo_penguins_l2443_244320

theorem zoo_penguins (penguins : ℕ) (polar_bears : ℕ) : 
  polar_bears = 2 * penguins → 
  penguins + polar_bears = 63 → 
  penguins = 21 := by
sorry

end NUMINAMATH_CALUDE_zoo_penguins_l2443_244320


namespace NUMINAMATH_CALUDE_least_months_to_triple_l2443_244310

/-- The initial borrowed amount in dollars -/
def initial_amount : ℝ := 2000

/-- The monthly interest rate as a decimal -/
def monthly_rate : ℝ := 0.04

/-- The function that calculates the owed amount after t months -/
def owed_amount (t : ℕ) : ℝ := initial_amount * (1 + monthly_rate) ^ t

/-- Theorem stating that 30 is the least integer number of months 
    after which the owed amount exceeds three times the initial amount -/
theorem least_months_to_triple : 
  (∀ k : ℕ, k < 30 → owed_amount k ≤ 3 * initial_amount) ∧ 
  (owed_amount 30 > 3 * initial_amount) := by
  sorry

#check least_months_to_triple

end NUMINAMATH_CALUDE_least_months_to_triple_l2443_244310


namespace NUMINAMATH_CALUDE_fraction_simplification_l2443_244387

theorem fraction_simplification : 1000^2 / (252^2 - 248^2) = 500 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2443_244387
