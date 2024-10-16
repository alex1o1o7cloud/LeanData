import Mathlib

namespace NUMINAMATH_CALUDE_max_area_rectangle_l2625_262516

/-- Represents a rectangle with integer dimensions and perimeter 40 -/
structure Rectangle where
  length : ℕ
  width : ℕ
  perimeter_constraint : length + width = 20

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem: The maximum area of a rectangle with perimeter 40 and integer dimensions is 100 -/
theorem max_area_rectangle :
  ∀ r : Rectangle, area r ≤ 100 :=
sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l2625_262516


namespace NUMINAMATH_CALUDE_price_comparison_l2625_262537

theorem price_comparison (x : ℝ) (h : x > 0) : x * 1.1 * 0.9 < x := by
  sorry

end NUMINAMATH_CALUDE_price_comparison_l2625_262537


namespace NUMINAMATH_CALUDE_series_convergence_l2625_262504

noncomputable def x : ℕ → ℝ
  | 0 => 1
  | n + 1 => Real.log (Real.exp (x n) - x n)

theorem series_convergence :
  (∑' n, x n) = Real.exp 1 - 1 := by sorry

end NUMINAMATH_CALUDE_series_convergence_l2625_262504


namespace NUMINAMATH_CALUDE_train_passing_platform_l2625_262594

/-- Calculates the time for a train to pass a platform -/
theorem train_passing_platform 
  (train_length : ℝ) 
  (tree_crossing_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 1500) 
  (h2 : tree_crossing_time = 120) 
  (h3 : platform_length = 500) : 
  (train_length + platform_length) / (train_length / tree_crossing_time) = 160 := by
  sorry

#check train_passing_platform

end NUMINAMATH_CALUDE_train_passing_platform_l2625_262594


namespace NUMINAMATH_CALUDE_jenny_stamps_last_page_l2625_262528

/-- Represents the stamp collection system -/
structure StampCollection where
  initialBooks : ℕ
  pagesPerBook : ℕ
  initialStampsPerPage : ℕ
  newStampsPerPage : ℕ
  filledBooks : ℕ
  filledPagesInLastBook : ℕ

/-- Calculates the number of stamps on the last page after reorganization -/
def stampsOnLastPage (sc : StampCollection) : ℕ :=
  let totalStamps := sc.initialBooks * sc.pagesPerBook * sc.initialStampsPerPage
  let filledPages := sc.filledBooks * sc.pagesPerBook + sc.filledPagesInLastBook
  totalStamps - (filledPages * sc.newStampsPerPage)

/-- Theorem: Given Jenny's stamp collection details, the last page contains 8 stamps -/
theorem jenny_stamps_last_page :
  let sc : StampCollection := {
    initialBooks := 10,
    pagesPerBook := 50,
    initialStampsPerPage := 6,
    newStampsPerPage := 8,
    filledBooks := 6,
    filledPagesInLastBook := 45
  }
  stampsOnLastPage sc = 8 := by
  sorry

end NUMINAMATH_CALUDE_jenny_stamps_last_page_l2625_262528


namespace NUMINAMATH_CALUDE_expression_evaluation_l2625_262570

theorem expression_evaluation (x y z : ℝ) (hx : x = -6) (hy : y = -3) (hz : z = 1/2) :
  4 * z * (x - y)^2 - (x * z) / y + 3 * Real.sin (y * z) = 17 + 3 * Real.sin (-3/2) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2625_262570


namespace NUMINAMATH_CALUDE_system_solution_l2625_262599

theorem system_solution (x y : ℝ) : 
  (16 * x^3 + 4*x = 16*y + 5) ∧ 
  (16 * y^3 + 4*y = 16*x + 5) → 
  (x = y) ∧ (16 * x^3 - 12*x - 5 = 0) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2625_262599


namespace NUMINAMATH_CALUDE_roses_ratio_l2625_262543

theorem roses_ratio (roses_day1 : ℕ) (roses_day2 : ℕ) (roses_day3 : ℕ) 
  (h1 : roses_day1 = 50)
  (h2 : roses_day2 = roses_day1 + 20)
  (h3 : roses_day1 + roses_day2 + roses_day3 = 220) :
  roses_day3 / roses_day1 = 2 := by
sorry

end NUMINAMATH_CALUDE_roses_ratio_l2625_262543


namespace NUMINAMATH_CALUDE_triangle_side_a_triangle_angle_B_l2625_262517

-- Part I
theorem triangle_side_a (A B C : ℝ) (a b c : ℝ) : 
  b = Real.sqrt 3 → A = π / 4 → C = 5 * π / 12 → a = Real.sqrt 2 := by sorry

-- Part II
theorem triangle_angle_B (A B C : ℝ) (a b c : ℝ) :
  b^2 = a^2 + c^2 + Real.sqrt 2 * a * c → B = 3 * π / 4 := by sorry

end NUMINAMATH_CALUDE_triangle_side_a_triangle_angle_B_l2625_262517


namespace NUMINAMATH_CALUDE_sum_of_base6_series_l2625_262571

/-- Represents a number in base 6 -/
def Base6 := Nat

/-- Converts a base 6 number to decimal -/
def to_decimal (n : Base6) : Nat :=
  sorry

/-- Converts a decimal number to base 6 -/
def to_base6 (n : Nat) : Base6 :=
  sorry

/-- The sum of an arithmetic series in base 6 -/
def arithmetic_sum_base6 (first last : Base6) (common_diff : Base6) : Base6 :=
  sorry

/-- Theorem: The sum of the series 2₆ + 4₆ + 6₆ + ⋯ + 100₆ in base 6 is 1330₆ -/
theorem sum_of_base6_series : 
  arithmetic_sum_base6 (to_base6 2) (to_base6 36) (to_base6 2) = to_base6 342 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_base6_series_l2625_262571


namespace NUMINAMATH_CALUDE_min_value_and_inequality_l2625_262511

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

-- Theorem statement
theorem min_value_and_inequality :
  (∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a ∧ ∃ (x₀ : ℝ), f x₀ = a) ∧
  (a = 3) ∧
  (∀ (p q r : ℝ), p > 0 → q > 0 → r > 0 → p + q + r = 3 → p^2 + q^2 + r^2 ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_l2625_262511


namespace NUMINAMATH_CALUDE_zoo_camels_l2625_262557

theorem zoo_camels (a : ℕ) 
  (h1 : ∃ x y : ℕ, x = y + 10 ∧ x + y = a)
  (h2 : ∃ x y : ℕ, x + 2*y = 55 ∧ x + y = a) : 
  a = 40 := by
sorry

end NUMINAMATH_CALUDE_zoo_camels_l2625_262557


namespace NUMINAMATH_CALUDE_isabel_pop_albums_l2625_262522

/-- The number of country albums Isabel bought -/
def country_albums : ℕ := 6

/-- The number of songs per album -/
def songs_per_album : ℕ := 9

/-- The total number of songs Isabel bought -/
def total_songs : ℕ := 72

/-- The number of pop albums Isabel bought -/
def pop_albums : ℕ := (total_songs - country_albums * songs_per_album) / songs_per_album

theorem isabel_pop_albums : pop_albums = 2 := by
  sorry

end NUMINAMATH_CALUDE_isabel_pop_albums_l2625_262522


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2625_262590

-- Define the set of real numbers satisfying the inequality
def S : Set ℝ := {x : ℝ | |x - 2| - |2*x - 1| > 0}

-- State the theorem
theorem inequality_solution_set : S = Set.Ioo (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2625_262590


namespace NUMINAMATH_CALUDE_subsets_with_adjacent_chairs_12_l2625_262500

/-- The number of subsets with at least three adjacent chairs in a circular arrangement of 12 chairs -/
def subsets_with_adjacent_chairs (n : ℕ) : ℕ :=
  let adjacent_3_to_6 := 4 * n
  let adjacent_7_plus := (Finset.range 6).sum (fun k => Nat.choose n (n - k))
  adjacent_3_to_6 + adjacent_7_plus

/-- Theorem stating that the number of subsets with at least three adjacent chairs
    in a circular arrangement of 12 chairs is 1634 -/
theorem subsets_with_adjacent_chairs_12 :
  subsets_with_adjacent_chairs 12 = 1634 := by
  sorry

end NUMINAMATH_CALUDE_subsets_with_adjacent_chairs_12_l2625_262500


namespace NUMINAMATH_CALUDE_square_field_area_l2625_262505

theorem square_field_area (side_length : ℝ) (h : side_length = 16) : 
  side_length * side_length = 256 := by
  sorry

end NUMINAMATH_CALUDE_square_field_area_l2625_262505


namespace NUMINAMATH_CALUDE_total_holiday_savings_l2625_262561

def holiday_savings (sam_savings victory_savings : ℕ) : ℕ :=
  sam_savings + victory_savings

theorem total_holiday_savings : 
  ∀ (sam_savings victory_savings : ℕ),
    sam_savings = 1000 →
    victory_savings = sam_savings - 100 →
    holiday_savings sam_savings victory_savings = 1900 :=
by
  sorry

end NUMINAMATH_CALUDE_total_holiday_savings_l2625_262561


namespace NUMINAMATH_CALUDE_inscribed_decagon_area_proof_l2625_262591

/-- The area of a decagon inscribed in a square with perimeter 150 cm, 
    where the vertices of the decagon divide each side of the square into five equal segments. -/
def inscribed_decagon_area : ℝ := 1181.25

/-- The perimeter of the square. -/
def square_perimeter : ℝ := 150

/-- The number of equal segments each side of the square is divided into. -/
def num_segments : ℕ := 5

/-- The number of triangles removed from the square to form the decagon. -/
def num_triangles : ℕ := 8

theorem inscribed_decagon_area_proof :
  let side_length := square_perimeter / 4
  let segment_length := side_length / num_segments
  let triangle_area := (1 / 2) * segment_length * segment_length
  let total_triangle_area := num_triangles * triangle_area
  let square_area := side_length * side_length
  square_area - total_triangle_area = inscribed_decagon_area := by sorry

end NUMINAMATH_CALUDE_inscribed_decagon_area_proof_l2625_262591


namespace NUMINAMATH_CALUDE_hotel_room_charges_l2625_262544

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * (1 - 0.4))
  (h2 : P = G * (1 - 0.1)) :
  R = G * 1.5 := by
  sorry

end NUMINAMATH_CALUDE_hotel_room_charges_l2625_262544


namespace NUMINAMATH_CALUDE_square_difference_equals_hundred_l2625_262534

theorem square_difference_equals_hundred :
  ∃ (x y : ℕ), x^2 - y^2 = 100 ∧ x = 26 ∧ y = 24 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equals_hundred_l2625_262534


namespace NUMINAMATH_CALUDE_min_value_fraction_l2625_262563

theorem min_value_fraction (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 3 + 27 / n ≥ 6 ∧ ((n : ℝ) / 3 + 27 / n = 6 ↔ n = 9) :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l2625_262563


namespace NUMINAMATH_CALUDE_first_grade_enrollment_proof_l2625_262583

theorem first_grade_enrollment_proof :
  ∃! a : ℕ,
    200 ≤ a ∧ a ≤ 300 ∧
    (∃ R : ℕ, a = 25 * R + 10) ∧
    (∃ L : ℕ, a = 30 * L - 15) ∧
    a = 285 := by
  sorry

end NUMINAMATH_CALUDE_first_grade_enrollment_proof_l2625_262583


namespace NUMINAMATH_CALUDE_no_integer_solution_l2625_262588

theorem no_integer_solution :
  ¬ ∃ (a b c : ℤ), a^2 + b^2 - 8*c = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2625_262588


namespace NUMINAMATH_CALUDE_melies_money_left_l2625_262596

/-- The amount of money Méliès has left after buying meat -/
def money_left (initial_money meat_quantity meat_price : ℝ) : ℝ :=
  initial_money - meat_quantity * meat_price

/-- Theorem: Méliès has $16 left after buying meat -/
theorem melies_money_left :
  let initial_money : ℝ := 180
  let meat_quantity : ℝ := 2
  let meat_price : ℝ := 82
  money_left initial_money meat_quantity meat_price = 16 := by
  sorry

end NUMINAMATH_CALUDE_melies_money_left_l2625_262596


namespace NUMINAMATH_CALUDE_mean_of_three_numbers_l2625_262538

theorem mean_of_three_numbers (a b c : ℝ) : 
  (a + b + c + 105) / 4 = 90 →
  (a + b + c) / 3 = 85 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_three_numbers_l2625_262538


namespace NUMINAMATH_CALUDE_rhombus_area_scaling_l2625_262587

theorem rhombus_area_scaling (d1 d2 : ℝ) :
  d1 > 0 → d2 > 0 → (d1 * d2) / 2 = 3 → ((5 * d1) * (5 * d2)) / 2 = 75 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_scaling_l2625_262587


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l2625_262573

/-- A six-digit number is between 100000 and 999999 -/
def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

/-- Function to reduce the first digit of a number by 3 and append 3 at the end -/
def transform (n : ℕ) : ℕ := (n - 300000) * 10 + 3

theorem unique_six_digit_number : 
  ∃! n : ℕ, is_six_digit n ∧ 3 * n = transform n ∧ n = 428571 := by sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l2625_262573


namespace NUMINAMATH_CALUDE_ball_probability_l2625_262527

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ)
  (h_total : total = 60)
  (h_white : white = 22)
  (h_green : green = 10)
  (h_yellow : yellow = 7)
  (h_red : red = 15)
  (h_purple : purple = 6)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 13 / 20 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l2625_262527


namespace NUMINAMATH_CALUDE_sum_of_triple_products_of_roots_l2625_262578

theorem sum_of_triple_products_of_roots (p q r s : ℂ) : 
  (4 * p^4 - 8 * p^3 + 18 * p^2 - 14 * p + 7 = 0) →
  (4 * q^4 - 8 * q^3 + 18 * q^2 - 14 * q + 7 = 0) →
  (4 * r^4 - 8 * r^3 + 18 * r^2 - 14 * r + 7 = 0) →
  (4 * s^4 - 8 * s^3 + 18 * s^2 - 14 * s + 7 = 0) →
  p * q * r + p * q * s + p * r * s + q * r * s = 7 / 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_triple_products_of_roots_l2625_262578


namespace NUMINAMATH_CALUDE_conference_games_l2625_262532

theorem conference_games (total_teams : Nat) (divisions : Nat) (teams_per_division : Nat)
  (h1 : total_teams = 12)
  (h2 : divisions = 3)
  (h3 : teams_per_division = 4)
  (h4 : total_teams = divisions * teams_per_division) :
  (total_teams * (3 * (teams_per_division - 1) + 2 * (total_teams - teams_per_division))) / 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_l2625_262532


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2625_262524

theorem negative_fraction_comparison : -2/3 < -3/5 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2625_262524


namespace NUMINAMATH_CALUDE_computer_price_after_15_years_l2625_262546

/-- The price of a computer after a certain number of 5-year periods, given an initial price and a price decrease rate. -/
def computer_price (initial_price : ℝ) (decrease_rate : ℝ) (periods : ℕ) : ℝ :=
  initial_price * (1 - decrease_rate) ^ periods

/-- Theorem stating that a computer with an initial price of 8100 yuan and a price decrease of 1/3 every 5 years will cost 2400 yuan after 15 years. -/
theorem computer_price_after_15_years :
  computer_price 8100 (1/3) 3 = 2400 := by
  sorry

end NUMINAMATH_CALUDE_computer_price_after_15_years_l2625_262546


namespace NUMINAMATH_CALUDE_max_value_abc_l2625_262533

theorem max_value_abc (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :
  (a * b + 2 * b * c) / (a^2 + b^2 + c^2) ≤ Real.sqrt 5 / 2 ∧
  ∃ a' b' c' : ℝ, (a' ≠ 0 ∨ b' ≠ 0 ∨ c' ≠ 0) ∧
    (a' * b' + 2 * b' * c') / (a'^2 + b'^2 + c'^2) = Real.sqrt 5 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_abc_l2625_262533


namespace NUMINAMATH_CALUDE_perfect_square_condition_l2625_262598

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, n^4 - n^3 + 3*n^2 + 5 = m^2) ↔ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l2625_262598


namespace NUMINAMATH_CALUDE_quadruple_equation_solutions_l2625_262562

def is_solution (x y z n : ℕ) : Prop :=
  x^2 + y^2 + z^2 + 1 = 2^n

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(1, 1, 1, 2), (0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 0, 1), (0, 0, 0, 0)}

theorem quadruple_equation_solutions :
  ∀ x y z n : ℕ, is_solution x y z n ↔ (x, y, z, n) ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_quadruple_equation_solutions_l2625_262562


namespace NUMINAMATH_CALUDE_distance_circle_center_to_line_l2625_262521

/-- The distance from the center of the circle ρ = 4cos θ to the line tan θ = 1 is √2 -/
theorem distance_circle_center_to_line : 
  ∀ (θ : ℝ) (ρ : ℝ → ℝ) (x y : ℝ),
  (ρ θ = 4 * Real.cos θ) →  -- Circle equation
  (Real.tan θ = 1) →        -- Line equation
  (x - 2)^2 + y^2 = 4 →     -- Standard form of circle equation
  x - y = 0 →               -- Line equation in rectangular coordinates
  Real.sqrt 2 = |x - 2| / Real.sqrt ((1:ℝ)^2 + (-1:ℝ)^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_line_l2625_262521


namespace NUMINAMATH_CALUDE_windows_preference_l2625_262518

theorem windows_preference (total : ℕ) (mac : ℕ) (no_pref : ℕ) 
  (h1 : total = 210)
  (h2 : mac = 60)
  (h3 : no_pref = 90) :
  total - mac - (mac / 3) - no_pref = 40 := by
  sorry

end NUMINAMATH_CALUDE_windows_preference_l2625_262518


namespace NUMINAMATH_CALUDE_price_change_theorem_l2625_262506

theorem price_change_theorem (initial_price : ℝ) (price_increase : ℝ) 
  (discount1 : ℝ) (discount2 : ℝ) :
  price_increase = 32 ∧ discount1 = 10 ∧ discount2 = 15 →
  let increased_price := initial_price * (1 + price_increase / 100)
  let after_discount1 := increased_price * (1 - discount1 / 100)
  let final_price := after_discount1 * (1 - discount2 / 100)
  (final_price - initial_price) / initial_price * 100 = 0.98 := by
sorry

end NUMINAMATH_CALUDE_price_change_theorem_l2625_262506


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l2625_262592

/-- Given that 3/4 of 16 bananas are worth 10 oranges, 
    prove that 3/5 of 15 bananas are worth 7.5 oranges -/
theorem banana_orange_equivalence :
  (3 / 4 : ℚ) * 16 * (1 / 10 : ℚ) = 1 →
  (3 / 5 : ℚ) * 15 * (1 / 10 : ℚ) = (15 / 2 : ℚ) * (1 / 10 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l2625_262592


namespace NUMINAMATH_CALUDE_problem_statement_l2625_262510

theorem problem_statement :
  (∀ (a b m : ℝ), (a * m^2 < b * m^2 → a < b) ∧ ¬(a < b → a * m^2 < b * m^2)) ∧
  (¬(∀ x : ℝ, x^3 - x^2 - 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 - 1 > 0)) ∧
  (∀ (p q : Prop), ¬p → ¬q → ¬(p ∧ q)) ∧
  ¬(∀ x : ℝ, (x ≠ 1 ∨ x ≠ -1 → x^2 ≠ 1) ↔ (x^2 = 1 → x = 1 ∨ x = -1)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2625_262510


namespace NUMINAMATH_CALUDE_inequality_implies_a_bounds_l2625_262565

-- Define the operation ⊕
def circleplus (x y : ℝ) : ℝ := (x + 3) * (y - 1)

-- State the theorem
theorem inequality_implies_a_bounds :
  (∀ x : ℝ, circleplus (x - a) (x + a) > -16) → -2 < a ∧ a < 6 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_a_bounds_l2625_262565


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2625_262542

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | |x| ≥ 1}
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.univ \ A) ∩ (Set.univ \ B) = {x : ℝ | -1 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2625_262542


namespace NUMINAMATH_CALUDE_least_positive_y_l2625_262581

-- Define variables
variable (c d : ℝ)
variable (y : ℝ)

-- Define the conditions
def condition1 : Prop := Real.tan y = (2 * c) / (3 * d)
def condition2 : Prop := Real.tan (2 * y) = (3 * d) / (2 * c + 3 * d)

-- State the theorem
theorem least_positive_y (h1 : condition1 c d y) (h2 : condition2 c d y) :
  y = Real.arctan (1 / 3) ∧ ∀ z, 0 < z ∧ z < y → ¬(condition1 c d z ∧ condition2 c d z) :=
sorry

end NUMINAMATH_CALUDE_least_positive_y_l2625_262581


namespace NUMINAMATH_CALUDE_fifteenth_student_age_l2625_262584

theorem fifteenth_student_age 
  (total_students : Nat) 
  (avg_age_all : ℝ) 
  (group1_students : Nat) 
  (avg_age_group1 : ℝ) 
  (group2_students : Nat) 
  (avg_age_group2 : ℝ)
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_students = 5)
  (h4 : avg_age_group1 = 14)
  (h5 : group2_students = 9)
  (h6 : avg_age_group2 = 16) :
  (total_students : ℝ) * avg_age_all - 
  ((group1_students : ℝ) * avg_age_group1 + (group2_students : ℝ) * avg_age_group2) = 11 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_student_age_l2625_262584


namespace NUMINAMATH_CALUDE_smallest_x_value_l2625_262559

theorem smallest_x_value (x y : ℝ) 
  (hx : 4 < x ∧ x < 8) 
  (hy : 8 < y ∧ y < 12) 
  (h_diff : ∃ (n : ℕ), n = 7 ∧ n = ⌊y - x⌋) : 
  4 < x :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l2625_262559


namespace NUMINAMATH_CALUDE_lending_interest_rate_l2625_262574

/-- Calculates the simple interest --/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem lending_interest_rate 
  (principal : ℝ)
  (b_to_c_rate : ℝ)
  (time : ℝ)
  (b_gain : ℝ)
  (h1 : principal = 3200)
  (h2 : b_to_c_rate = 0.145)
  (h3 : time = 5)
  (h4 : b_gain = 400)
  : ∃ (a_to_b_rate : ℝ), 
    simpleInterest principal a_to_b_rate time = 
    simpleInterest principal b_to_c_rate time - b_gain ∧ 
    a_to_b_rate = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_lending_interest_rate_l2625_262574


namespace NUMINAMATH_CALUDE_square_midpoint_dot_product_l2625_262508

-- Define the square ABCD
def Square (A B C D : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let CD := (D.1 - C.1, D.2 - C.2)
  let DA := (A.1 - D.1, A.2 - D.2)
  (AB.1 * AB.1 + AB.2 * AB.2 = 4) ∧
  (BC.1 * BC.1 + BC.2 * BC.2 = 4) ∧
  (CD.1 * CD.1 + CD.2 * CD.2 = 4) ∧
  (DA.1 * DA.1 + DA.2 * DA.2 = 4) ∧
  (AB.1 * BC.1 + AB.2 * BC.2 = 0) ∧
  (BC.1 * CD.1 + BC.2 * CD.2 = 0) ∧
  (CD.1 * DA.1 + CD.2 * DA.2 = 0) ∧
  (DA.1 * AB.1 + DA.2 * AB.2 = 0)

-- Define the midpoint E of CD
def Midpoint (C D E : ℝ × ℝ) : Prop :=
  E.1 = (C.1 + D.1) / 2 ∧ E.2 = (C.2 + D.2) / 2

-- Define the dot product of two vectors
def DotProduct (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Theorem statement
theorem square_midpoint_dot_product 
  (A B C D E : ℝ × ℝ) 
  (h1 : Square A B C D) 
  (h2 : Midpoint C D E) : 
  DotProduct (E.1 - A.1, E.2 - A.2) (D.1 - B.1, D.2 - B.2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_midpoint_dot_product_l2625_262508


namespace NUMINAMATH_CALUDE_expression_value_l2625_262575

theorem expression_value (x y : ℝ) (h1 : x ≠ y) 
  (h2 : 1 / (x^2 + 1) + 1 / (y^2 + 1) = 2 / (x * y + 1)) : 
  1 / (x^2 + 1) + 1 / (y^2 + 1) + 2 / (x * y + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2625_262575


namespace NUMINAMATH_CALUDE_minimum_planting_cost_l2625_262501

/-- Represents the dimensions of a rectangular region -/
structure Region where
  width : ℝ
  height : ℝ

/-- Represents a type of flower with its cost -/
structure Flower where
  name : String
  cost : ℝ

/-- Calculates the area of a region -/
def area (r : Region) : ℝ := r.width * r.height

/-- Calculates the cost of planting a flower in a region -/
def plantingCost (f : Flower) (r : Region) : ℝ := f.cost * area r

/-- The flower bed configuration -/
def flowerBed : Region := { width := 11, height := 6 }

/-- The vertical strip -/
def verticalStrip : Region := { width := 3, height := 6 }

/-- The horizontal strip -/
def horizontalStrip : Region := { width := 11, height := 2 }

/-- The overlap region between vertical and horizontal strips -/
def overlapRegion : Region := { width := 3, height := 2 }

/-- The remaining region -/
def remainingRegion : Region :=
  { width := flowerBed.width - verticalStrip.width,
    height := flowerBed.height - horizontalStrip.height }

/-- The available flower types -/
def flowers : List Flower :=
  [{ name := "Easter Lily", cost := 3 },
   { name := "Dahlia", cost := 2.5 },
   { name := "Canna", cost := 2 }]

/-- Theorem: The minimum cost for planting the flowers is $157 -/
theorem minimum_planting_cost :
  plantingCost (flowers[2]) remainingRegion +
  plantingCost (flowers[1]) verticalStrip +
  plantingCost (flowers[0]) { width := horizontalStrip.width - verticalStrip.width,
                              height := horizontalStrip.height } = 157 := by
  sorry


end NUMINAMATH_CALUDE_minimum_planting_cost_l2625_262501


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2625_262536

theorem polynomial_factorization (a b : ℝ) : a^2 + 2*b - b^2 - 1 = (a-b+1)*(a+b-1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2625_262536


namespace NUMINAMATH_CALUDE_max_integer_difference_l2625_262539

theorem max_integer_difference (x y : ℤ) (hx : -6 < x ∧ x < -2) (hy : 4 < y ∧ y < 10) :
  (∀ (a b : ℤ), -6 < a ∧ a < -2 ∧ 4 < b ∧ b < 10 → b - a ≤ y - x) →
  y - x = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_difference_l2625_262539


namespace NUMINAMATH_CALUDE_unique_solution_l2625_262520

/-- Define the function f as specified in the problem -/
def f (x y z : ℕ+) : ℤ :=
  (((x + y - 2) * (x + y - 1)) / 2) - z

/-- Theorem stating the unique solution to the problem -/
theorem unique_solution :
  ∃! (a b c d : ℕ+), f a b c = 1993 ∧ f c d a = 1993 ∧ a = 23 ∧ b = 42 ∧ c = 23 ∧ d = 42 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2625_262520


namespace NUMINAMATH_CALUDE_peters_pizza_fraction_l2625_262545

-- Define the number of slices in the pizza
def total_slices : ℕ := 16

-- Define the number of whole slices Peter ate
def whole_slices_eaten : ℕ := 1

-- Define the number of slices shared
def shared_slices : ℕ := 2

-- Theorem statement
theorem peters_pizza_fraction :
  (whole_slices_eaten : ℚ) / total_slices + 
  (shared_slices : ℚ) / total_slices / 2 = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_peters_pizza_fraction_l2625_262545


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l2625_262529

/-- Three vectors in ℝ³ -/
def a : Fin 3 → ℝ := ![3, 7, 2]
def b : Fin 3 → ℝ := ![-2, 0, -1]
def c : Fin 3 → ℝ := ![2, 2, 1]

/-- Scalar triple product of three vectors in ℝ³ -/
def scalarTripleProduct (u v w : Fin 3 → ℝ) : ℝ :=
  Matrix.det !![u 0, u 1, u 2; v 0, v 1, v 2; w 0, w 1, w 2]

/-- Theorem: The vectors a, b, and c are not coplanar -/
theorem vectors_not_coplanar : scalarTripleProduct a b c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l2625_262529


namespace NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2625_262515

theorem quadrilateral_diagonal_length 
  (offset1 offset2 total_area : ℝ) 
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : total_area = 180)
  (h4 : total_area = (offset1 + offset2) * diagonal / 2) :
  diagonal = 24 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonal_length_l2625_262515


namespace NUMINAMATH_CALUDE_difference_of_squares_l2625_262558

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l2625_262558


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2625_262507

theorem necessary_not_sufficient (a b : ℝ) : 
  (∀ a b : ℝ, b ≥ 0 → a^2 + b ≥ 0) ∧ 
  (∃ a b : ℝ, a^2 + b ≥ 0 ∧ b < 0) := by
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2625_262507


namespace NUMINAMATH_CALUDE_cakes_dinner_today_l2625_262519

def cakes_lunch_today : ℕ := 5
def cakes_yesterday : ℕ := 3
def total_cakes : ℕ := 14

theorem cakes_dinner_today : ∃ x : ℕ, x = total_cakes - cakes_lunch_today - cakes_yesterday ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cakes_dinner_today_l2625_262519


namespace NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l2625_262541

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Define a function to get the last two nonzero digits
def lastTwoNonzeroDigits (n : ℕ) : ℕ :=
  n % 100

-- Theorem statement
theorem last_two_nonzero_digits_80_factorial :
  lastTwoNonzeroDigits (factorial 80) = 76 := by
  sorry


end NUMINAMATH_CALUDE_last_two_nonzero_digits_80_factorial_l2625_262541


namespace NUMINAMATH_CALUDE_quadratic_inequality_condition_l2625_262576

theorem quadratic_inequality_condition (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + x + m > 0) → m > (1 / 4 : ℝ) ∧ 
  ∃ m₀ : ℝ, m₀ > (1 / 4 : ℝ) ∧ ∃ x₀ : ℝ, m₀ * x₀^2 + x₀ + m₀ ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_condition_l2625_262576


namespace NUMINAMATH_CALUDE_correct_calculation_l2625_262552

theorem correct_calculation (x : ℤ) : 
  (713 + x = 928) → (713 - x = 498) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2625_262552


namespace NUMINAMATH_CALUDE_binomial_square_constant_l2625_262526

theorem binomial_square_constant (c : ℚ) : 
  (∃ a b : ℚ, ∀ x, 9 * x^2 + 27 * x + c = (a * x + b)^2) → c = 81/4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l2625_262526


namespace NUMINAMATH_CALUDE_arc_length_radius_l2625_262503

/-- Given an arc length and central angle, calculate the radius of the circle. -/
theorem arc_length_radius (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = 2) :
  s / θ = 2 := by sorry

end NUMINAMATH_CALUDE_arc_length_radius_l2625_262503


namespace NUMINAMATH_CALUDE_alice_monthly_increase_l2625_262550

/-- Represents Alice's savings pattern over three months -/
def aliceSavings (initialSavings : ℝ) (monthlyIncrease : ℝ) : ℝ :=
  initialSavings + (initialSavings + monthlyIncrease) + (initialSavings + 2 * monthlyIncrease)

/-- Theorem stating Alice's monthly savings increase -/
theorem alice_monthly_increase (initialSavings totalSavings : ℝ) 
  (h1 : initialSavings = 10)
  (h2 : totalSavings = 70)
  (h3 : ∃ x : ℝ, aliceSavings initialSavings x = totalSavings) :
  ∃ x : ℝ, x = 40 / 3 ∧ aliceSavings initialSavings x = totalSavings :=
sorry

end NUMINAMATH_CALUDE_alice_monthly_increase_l2625_262550


namespace NUMINAMATH_CALUDE_complex_arithmetic_expression_l2625_262564

theorem complex_arithmetic_expression : -1^2009 * (-3) + 1 - 2^2 * 3 + (1 - 2^2) / 3 + (1 - 2 * 3)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_expression_l2625_262564


namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2625_262597

theorem smallest_k_no_real_roots : 
  let f (k : ℤ) (x : ℝ) := 3 * x * (k * x - 5) - x^2 + 4
  ∀ k : ℤ, (∀ x : ℝ, f k x ≠ 0) → k ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2625_262597


namespace NUMINAMATH_CALUDE_percentage_equality_l2625_262548

theorem percentage_equality : (0.75 * 40 : ℝ) = (4/5 : ℝ) * 25 + 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equality_l2625_262548


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l2625_262555

theorem algebraic_expression_equality (x : ℝ) (h : x^2 + 3*x + 5 = 7) : 
  3*x^2 + 9*x - 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l2625_262555


namespace NUMINAMATH_CALUDE_sum_of_exponents_eight_l2625_262514

/-- Sum of geometric series from 1 to x^n -/
def geometricSum (x n : ℕ) : ℕ := (x^(n+1) - 1) / (x - 1)

/-- Sum of divisors of 2^i * 3^j * 5^k -/
def sumDivisors (i j k : ℕ) : ℕ :=
  (geometricSum 2 i) * (geometricSum 3 j) * (geometricSum 5 k)

/-- Theorem: If the sum of divisors of 2^i * 3^j * 5^k is 1800, then i + j + k = 8 -/
theorem sum_of_exponents_eight (i j k : ℕ) :
  sumDivisors i j k = 1800 → i + j + k = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_exponents_eight_l2625_262514


namespace NUMINAMATH_CALUDE_christmas_day_is_saturday_l2625_262579

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in November or December -/
structure Date where
  month : Nat
  day : Nat

/-- Function to determine the day of the week for a given date -/
def dayOfWeek (date : Date) : DayOfWeek := sorry

/-- Function to add days to a given date -/
def addDays (date : Date) (days : Nat) : Date := sorry

theorem christmas_day_is_saturday 
  (thanksgiving : Date)
  (h1 : thanksgiving.month = 11)
  (h2 : thanksgiving.day = 25)
  (h3 : dayOfWeek thanksgiving = DayOfWeek.Thursday) :
  dayOfWeek (Date.mk 12 25) = DayOfWeek.Saturday := by sorry

end NUMINAMATH_CALUDE_christmas_day_is_saturday_l2625_262579


namespace NUMINAMATH_CALUDE_no_singleton_set_with_conditions_l2625_262551

theorem no_singleton_set_with_conditions :
  ¬ ∃ (A : Set ℝ), (∃ (a : ℝ), A = {a}) ∧
    (∀ a : ℝ, a ∈ A → (1 / (1 - a)) ∈ A) ∧
    (1 ∈ A) := by
  sorry

end NUMINAMATH_CALUDE_no_singleton_set_with_conditions_l2625_262551


namespace NUMINAMATH_CALUDE_base6_to_base10_conversion_l2625_262595

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (6 ^ i)) 0

/-- The base 6 representation of the number -/
def base6Number : List Nat := [1, 2, 5, 4, 3]

theorem base6_to_base10_conversion :
  base6ToBase10 base6Number = 4945 := by
  sorry

end NUMINAMATH_CALUDE_base6_to_base10_conversion_l2625_262595


namespace NUMINAMATH_CALUDE_min_quotient_is_53_5_l2625_262554

/-- A three-digit number with distinct non-zero digits -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  a_nonzero : a ≠ 0
  b_nonzero : b ≠ 0
  c_nonzero : c ≠ 0
  a_lt_ten : a < 10
  b_lt_ten : b < 10
  c_lt_ten : c < 10
  distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- The sum of digits of a three-digit number -/
def digitSum (n : ThreeDigitNumber) : Nat :=
  n.a + n.b + n.c

/-- The quotient of a three-digit number divided by the sum of its digits -/
def quotient (n : ThreeDigitNumber) : Rat :=
  (value n : Rat) / (digitSum n : Rat)

theorem min_quotient_is_53_5 :
  ∃ (min : Rat), ∀ (n : ThreeDigitNumber), quotient n ≥ min ∧ (∃ (m : ThreeDigitNumber), quotient m = min) ∧ min = 53.5 := by
  sorry

end NUMINAMATH_CALUDE_min_quotient_is_53_5_l2625_262554


namespace NUMINAMATH_CALUDE_perpendicular_condition_l2625_262556

theorem perpendicular_condition (x : ℝ) :
  let a : ℝ × ℝ := (1, 2*x)
  let b : ℝ × ℝ := (4, -x)
  (x = Real.sqrt 2 → a.1 * b.1 + a.2 * b.2 = 0) ∧
  ¬(a.1 * b.1 + a.2 * b.2 = 0 → x = Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l2625_262556


namespace NUMINAMATH_CALUDE_product_pricing_and_purchase_l2625_262531

-- Define variables
variable (x : ℝ) -- Price of product A
variable (y : ℝ) -- Price of product B
variable (m : ℝ) -- Number of units of product A to be purchased

-- Define the conditions
def condition1 : Prop := 2 * x + 3 * y = 690
def condition2 : Prop := x + 4 * y = 720
def condition3 : Prop := m * x + (40 - m) * y ≤ 5400
def condition4 : Prop := m ≤ 3 * (40 - m)

-- State the theorem
theorem product_pricing_and_purchase (h1 : condition1 x y) (h2 : condition2 x y) 
  (h3 : condition3 x y m) (h4 : condition4 m) : 
  x = 120 ∧ y = 150 ∧ 20 ≤ m ∧ m ≤ 30 := by
  sorry

end NUMINAMATH_CALUDE_product_pricing_and_purchase_l2625_262531


namespace NUMINAMATH_CALUDE_fraction_equality_l2625_262512

theorem fraction_equality (x y : ℚ) (hx : x = 4/7) (hy : y = 8/11) : 
  (7*x + 11*y) / (49*x*y) = 231/56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2625_262512


namespace NUMINAMATH_CALUDE_salary_increase_proof_l2625_262567

/-- Calculates the increase in average salary when adding a manager to a group of employees -/
def salary_increase (num_employees : ℕ) (initial_avg : ℚ) (manager_salary : ℚ) : ℚ :=
  let new_total := num_employees * initial_avg + manager_salary
  let new_avg := new_total / (num_employees + 1)
  new_avg - initial_avg

/-- The increase in average salary when adding a manager's salary of 3300 to a group of 20 employees with an initial average salary of 1200 is equal to 100 -/
theorem salary_increase_proof :
  salary_increase 20 1200 3300 = 100 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l2625_262567


namespace NUMINAMATH_CALUDE_fraction_equality_l2625_262525

theorem fraction_equality (m n : ℚ) (h : m / n = 3 / 4) : 
  (m + n) / n = 7 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2625_262525


namespace NUMINAMATH_CALUDE_bill_division_l2625_262513

/-- The total bill amount when three people divide it evenly -/
def total_bill (individual_payment : ℕ) : ℕ := 3 * individual_payment

/-- Theorem: If three people divide a bill evenly and each pays $33, then the total bill is $99 -/
theorem bill_division (individual_payment : ℕ) 
  (h : individual_payment = 33) : 
  total_bill individual_payment = 99 := by
  sorry

end NUMINAMATH_CALUDE_bill_division_l2625_262513


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l2625_262577

theorem perfect_square_binomial (x : ℝ) (k : ℝ) : 
  (∃ b : ℝ, ∀ x, x^2 + 24*x + k = (x + b)^2) ↔ k = 144 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l2625_262577


namespace NUMINAMATH_CALUDE_min_side_in_triangle_l2625_262566

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if c = 2b and the area of the triangle is 1, then the minimum value of a is √3.
-/
theorem min_side_in_triangle (a b c : ℝ) (A B C : ℝ) :
  c = 2 * b →
  (1 / 2) * b * c * Real.sin A = 1 →
  ∃ (a_min : ℝ), a_min = Real.sqrt 3 ∧ ∀ a', a' ≥ a_min := by
  sorry

end NUMINAMATH_CALUDE_min_side_in_triangle_l2625_262566


namespace NUMINAMATH_CALUDE_f_increasing_l2625_262589

-- Define the function
def f (x : ℝ) : ℝ := x^3 + x

-- Theorem statement
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_l2625_262589


namespace NUMINAMATH_CALUDE_coplanar_condition_l2625_262572

open Vector

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V] [CompleteSpace V]

-- Define the points
variable (O E F G H : V)

-- Define the condition for coplanarity
def are_coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), (B - A) + b • (C - A) + c • (D - A) = 0

-- Define the theorem
theorem coplanar_condition (m : ℝ) :
  (4 • (E - O) - 3 • (F - O) + 6 • (G - O) + m • (H - O) = 0) →
  (are_coplanar E F G H ↔ m = -7) :=
by sorry

end NUMINAMATH_CALUDE_coplanar_condition_l2625_262572


namespace NUMINAMATH_CALUDE_original_profit_percentage_l2625_262530

theorem original_profit_percentage (cost selling_price : ℝ) 
  (h1 : cost > 0) 
  (h2 : selling_price > cost) 
  (h3 : selling_price - (1.12 * cost) = 0.552 * selling_price) : 
  (selling_price - cost) / cost = 1.5 := by
sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l2625_262530


namespace NUMINAMATH_CALUDE_negation_equivalence_l2625_262553

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 < 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2625_262553


namespace NUMINAMATH_CALUDE_product_modulo_seven_l2625_262582

theorem product_modulo_seven : (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_modulo_seven_l2625_262582


namespace NUMINAMATH_CALUDE_nested_expression_equals_one_l2625_262586

theorem nested_expression_equals_one :
  (3 * (3 * (3 * (3 * (3 * (3 - 2) - 2) - 2) - 2) - 2) - 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_equals_one_l2625_262586


namespace NUMINAMATH_CALUDE_election_votes_total_l2625_262523

/-- Proves that the total number of votes in an election is 180, given that Emma received 4/15 of the total votes and 48 votes in total. -/
theorem election_votes_total (emma_fraction : Rat) (emma_votes : ℕ) (total_votes : ℕ) 
  (h1 : emma_fraction = 4 / 15)
  (h2 : emma_votes = 48)
  (h3 : emma_fraction * total_votes = emma_votes) :
  total_votes = 180 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_total_l2625_262523


namespace NUMINAMATH_CALUDE_M_eq_roster_l2625_262540

def M : Set ℚ := {x | ∃ (m : ℤ) (n : ℕ), n > 0 ∧ n ≤ 3 ∧ abs m < 2 ∧ x = m / n}

theorem M_eq_roster : M = {-1, -1/2, -1/3, 0, 1/3, 1/2, 1} := by sorry

end NUMINAMATH_CALUDE_M_eq_roster_l2625_262540


namespace NUMINAMATH_CALUDE_quadratic_common_roots_l2625_262535

theorem quadratic_common_roots (p : ℚ) (x : ℚ) : 
  (x^2 - (p+1)*x + (p+1) = 0 ∧ 2*x^2 + (p-2)*x - p - 7 = 0) ↔ 
  ((p = 3 ∧ x = 2) ∨ (p = -3/2 ∧ x = -1)) := by
sorry

end NUMINAMATH_CALUDE_quadratic_common_roots_l2625_262535


namespace NUMINAMATH_CALUDE_khali_snow_volume_l2625_262580

/-- Calculates the total volume of snow on a rectangular sidewalk with two layers -/
def total_snow_volume (length width depth1 depth2 : ℝ) : ℝ :=
  length * width * (depth1 + depth2)

/-- Theorem: The total volume of snow on Khali's sidewalk is 90 cubic feet -/
theorem khali_snow_volume :
  let length : ℝ := 30
  let width : ℝ := 3
  let depth1 : ℝ := 0.6
  let depth2 : ℝ := 0.4
  total_snow_volume length width depth1 depth2 = 90 := by
  sorry

#eval total_snow_volume 30 3 0.6 0.4

end NUMINAMATH_CALUDE_khali_snow_volume_l2625_262580


namespace NUMINAMATH_CALUDE_smallest_divisible_k_l2625_262547

/-- The polynomial p(z) = z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1 -/
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

/-- The function f(k) = z^k - 1 -/
def f (k : ℕ) (z : ℂ) : ℂ := z^k - 1

/-- Theorem: The smallest positive integer k such that p(z) divides f(k)(z) is 112 -/
theorem smallest_divisible_k : (∀ z : ℂ, p z ∣ f 112 z) ∧
  (∀ k : ℕ, k < 112 → ∃ z : ℂ, ¬(p z ∣ f k z)) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_k_l2625_262547


namespace NUMINAMATH_CALUDE_newer_model_travels_200_miles_l2625_262568

/-- The distance traveled by the older model car -/
def older_model_distance : ℝ := 160

/-- The percentage increase in distance for the newer model -/
def newer_model_percentage : ℝ := 0.25

/-- The distance traveled by the newer model car -/
def newer_model_distance : ℝ := older_model_distance * (1 + newer_model_percentage)

/-- Theorem stating that the newer model travels 200 miles -/
theorem newer_model_travels_200_miles :
  newer_model_distance = 200 := by sorry

end NUMINAMATH_CALUDE_newer_model_travels_200_miles_l2625_262568


namespace NUMINAMATH_CALUDE_same_points_iff_odd_participants_l2625_262585

/-- Represents a round-robin chess tournament -/
structure Tournament where
  participants : ℕ
  no_draws : Bool

/-- The number of games played in a round-robin tournament -/
def games_played (t : Tournament) : ℕ :=
  t.participants * (t.participants - 1) / 2

/-- The total points scored in the tournament -/
def total_points (t : Tournament) : ℕ := games_played t

/-- Whether all participants have the same number of points -/
def all_same_points (t : Tournament) : Prop :=
  ∃ (p : ℕ), p * t.participants = total_points t

theorem same_points_iff_odd_participants (t : Tournament) (h : t.no_draws = true) :
  all_same_points t ↔ Odd t.participants :=
sorry

end NUMINAMATH_CALUDE_same_points_iff_odd_participants_l2625_262585


namespace NUMINAMATH_CALUDE_ball_probabilities_l2625_262560

/-- Represents the box of balls -/
structure BallBox where
  red_balls : ℕ
  white_balls : ℕ

/-- The probability of drawing exactly one red ball and one white ball without replacement -/
def prob_one_red_one_white (box : BallBox) : ℚ :=
  let total := box.red_balls + box.white_balls
  (box.red_balls : ℚ) / total * (box.white_balls : ℚ) / (total - 1) +
  (box.white_balls : ℚ) / total * (box.red_balls : ℚ) / (total - 1)

/-- The probability of getting at least one red ball in three draws with replacement -/
def prob_at_least_one_red (box : BallBox) : ℚ :=
  let p_red := (box.red_balls : ℚ) / (box.red_balls + box.white_balls)
  1 - (1 - p_red) ^ 3

theorem ball_probabilities (box : BallBox) (h1 : box.red_balls = 2) (h2 : box.white_balls = 4) :
  prob_one_red_one_white box = 8/15 ∧ prob_at_least_one_red box = 19/27 := by
  sorry


end NUMINAMATH_CALUDE_ball_probabilities_l2625_262560


namespace NUMINAMATH_CALUDE_train_speed_l2625_262502

/-- The speed of a train passing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (time : ℝ) (h1 : train_length = 240) 
  (h2 : bridge_length = 130) (h3 : time = 26.64) : 
  ∃ (speed : ℝ), abs (speed - 50.004) < 0.001 ∧ 
  speed = (train_length + bridge_length) / time * 3.6 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2625_262502


namespace NUMINAMATH_CALUDE_eyes_saw_airplane_l2625_262509

/-- Given 200 students and 3/4 of them looking up, prove that 300 eyes saw the airplane. -/
theorem eyes_saw_airplane (total_students : ℕ) (fraction_looked_up : ℚ) (h1 : total_students = 200) (h2 : fraction_looked_up = 3/4) :
  (fraction_looked_up * total_students : ℚ).num * 2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_eyes_saw_airplane_l2625_262509


namespace NUMINAMATH_CALUDE_volume_surface_area_ratio_l2625_262593

/-- A structure formed by connecting eight unit cubes -/
structure CubeStructure where
  /-- The number of unit cubes in the structure -/
  num_cubes : ℕ
  /-- The volume of the structure in cubic units -/
  volume : ℕ
  /-- The surface area of the structure in square units -/
  surface_area : ℕ
  /-- The number of cubes is 8 -/
  cube_count : num_cubes = 8
  /-- The volume is equal to the number of cubes -/
  volume_def : volume = num_cubes
  /-- The surface area is 24 square units -/
  surface_area_def : surface_area = 24

/-- Theorem: The ratio of volume to surface area is 1/3 -/
theorem volume_surface_area_ratio (c : CubeStructure) :
  (c.volume : ℚ) / c.surface_area = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_volume_surface_area_ratio_l2625_262593


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_3_6_l2625_262549

theorem gcf_lcm_sum_3_6 : Nat.gcd 3 6 + Nat.lcm 3 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_3_6_l2625_262549


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l2625_262569

theorem consecutive_integers_product_812_sum_57 :
  ∀ x y : ℕ,
    x > 0 →
    y = x + 1 →
    x * y = 812 →
    x + y = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l2625_262569
