import Mathlib

namespace NUMINAMATH_CALUDE_power_of_two_representation_l2628_262868

theorem power_of_two_representation (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℤ), 2^n = 7*x^2 + y^2 ∧ Odd x ∧ Odd y :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_representation_l2628_262868


namespace NUMINAMATH_CALUDE_parabola_focus_coordinates_l2628_262824

/-- Given a quadratic function f(x) = ax^2 + bx + 2 where a ≠ 0 and |f(x)| ≥ 2 for all real x,
    the focus of the parabola has coordinates (0, 1/(4a) + 2). -/
theorem parabola_focus_coordinates (a b : ℝ) (ha : a ≠ 0) 
    (hf : ∀ x : ℝ, |a * x^2 + b * x + 2| ≥ 2) :
    ∃ (focus : ℝ × ℝ), focus = (0, 1 / (4 * a) + 2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_coordinates_l2628_262824


namespace NUMINAMATH_CALUDE_books_per_shelf_l2628_262889

theorem books_per_shelf (total_books : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (h1 : total_books = 72) 
  (h2 : mystery_shelves = 5) 
  (h3 : picture_shelves = 4) :
  ∃ (books_per_shelf : ℕ), 
    books_per_shelf * (mystery_shelves + picture_shelves) = total_books ∧ 
    books_per_shelf = 8 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l2628_262889


namespace NUMINAMATH_CALUDE_johns_number_l2628_262878

theorem johns_number : ∃ x : ℝ, (2 * (3 * x - 6) + 20 = 122) ∧ x = 19 := by
  sorry

end NUMINAMATH_CALUDE_johns_number_l2628_262878


namespace NUMINAMATH_CALUDE_fixed_root_quadratic_l2628_262877

theorem fixed_root_quadratic (k : ℝ) : 
  ∃ x : ℝ, x^2 + (k + 3) * x + k + 2 = 0 ∧ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_root_quadratic_l2628_262877


namespace NUMINAMATH_CALUDE_mean_score_problem_l2628_262841

theorem mean_score_problem (m_mean a_mean : ℝ) (m a : ℕ) 
  (h1 : m_mean = 75)
  (h2 : a_mean = 65)
  (h3 : m = 2 * a / 3) :
  (m_mean * m + a_mean * a) / (m + a) = 69 := by
sorry

end NUMINAMATH_CALUDE_mean_score_problem_l2628_262841


namespace NUMINAMATH_CALUDE_penny_to_nickel_ratio_l2628_262855

/-- Represents the number of coins of each type -/
structure CoinCounts where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents -/
def totalValue (coins : CoinCounts) : ℕ :=
  coins.pennies + 5 * coins.nickels + 10 * coins.dimes + 25 * coins.quarters

/-- The main theorem stating the ratio of pennies to nickels -/
theorem penny_to_nickel_ratio (coins : CoinCounts) :
  coins.pennies = 120 ∧
  coins.nickels = 5 * coins.dimes ∧
  coins.quarters = 2 * coins.dimes ∧
  totalValue coins = 800 →
  coins.pennies / coins.nickels = 3 :=
by sorry

end NUMINAMATH_CALUDE_penny_to_nickel_ratio_l2628_262855


namespace NUMINAMATH_CALUDE_no_common_solution_l2628_262876

theorem no_common_solution : ¬∃ (x y : ℝ), x^2 + y^2 = 16 ∧ x^2 + 3*y + 30 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l2628_262876


namespace NUMINAMATH_CALUDE_linear_system_solution_l2628_262822

theorem linear_system_solution (m : ℚ) :
  let x : ℚ → ℚ := λ m => 6 * m + 1
  let y : ℚ → ℚ := λ m => -10 * m - 1
  (∀ m, x m + y m = -4 * m ∧ 2 * x m + y m = 2 * m + 1) ∧
  (x (1/2) - y (1/2) = 10) :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l2628_262822


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2628_262827

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 3*x + 1

-- Define the point on the tangent line
def point : ℝ × ℝ := (2, 5)

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 7*x - y - 9 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (k : ℝ), 
    (∀ x, (deriv f) x = 2*x + 3) ∧ 
    (deriv f) point.1 = k ∧
    ∀ x y, tangent_line x y ↔ y - point.2 = k * (x - point.1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2628_262827


namespace NUMINAMATH_CALUDE_square_difference_540_460_l2628_262831

theorem square_difference_540_460 : 540^2 - 460^2 = 80000 := by sorry

end NUMINAMATH_CALUDE_square_difference_540_460_l2628_262831


namespace NUMINAMATH_CALUDE_infinite_powers_of_two_l2628_262885

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- The sequence a_n -/
noncomputable def a (n : ℕ) : ℤ :=
  floor (n * Real.sqrt 2)

/-- Statement: There are infinitely many n such that a_n is a power of 2 -/
theorem infinite_powers_of_two : ∀ k : ℕ, ∃ n > k, ∃ m : ℕ, a n = 2^m :=
sorry

end NUMINAMATH_CALUDE_infinite_powers_of_two_l2628_262885


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2628_262854

/-- Given a hyperbola with equation x²/a² - y² = 1 and an asymptote √3x + y = 0,
    prove that a = √3/3 -/
theorem hyperbola_asymptote (a : ℝ) : 
  (∃ x y : ℝ, x^2/a^2 - y^2 = 1) ∧ 
  (∃ x y : ℝ, Real.sqrt 3 * x + y = 0) → 
  a = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2628_262854


namespace NUMINAMATH_CALUDE_wire_weight_proportional_l2628_262863

/-- Given that a 25 m roll of wire weighs 5 kg, prove that a 75 m roll of wire weighs 15 kg. -/
theorem wire_weight_proportional (length_short : ℝ) (weight_short : ℝ) (length_long : ℝ) :
  length_short = 25 →
  weight_short = 5 →
  length_long = 75 →
  (length_long / length_short) * weight_short = 15 := by
  sorry

end NUMINAMATH_CALUDE_wire_weight_proportional_l2628_262863


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2628_262849

theorem unique_solution_for_equation (x y : ℝ) :
  (x - 6)^2 + (y - 7)^2 + (x - y)^2 = 1/3 ↔ x = 19/3 ∧ y = 20/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2628_262849


namespace NUMINAMATH_CALUDE_probability_of_E_l2628_262828

def vowels : Finset Char := {'A', 'E', 'I', 'O', 'U'}

def count (c : Char) : ℕ :=
  match c with
  | 'A' => 5
  | 'E' => 3
  | 'I' => 4
  | 'O' => 2
  | 'U' => 6
  | _ => 0

def total_count : ℕ := (vowels.sum count)

theorem probability_of_E : 
  (count 'E' : ℚ) / total_count = 3 / 20 := by sorry

end NUMINAMATH_CALUDE_probability_of_E_l2628_262828


namespace NUMINAMATH_CALUDE_thalassa_population_estimate_l2628_262874

-- Define the initial population in 2020
def initial_population : ℕ := 500

-- Define the doubling period in years
def doubling_period : ℕ := 30

-- Define the target year
def target_year : ℕ := 2075

-- Define the base year
def base_year : ℕ := 2020

-- Function to calculate the number of complete doubling periods
def complete_doubling_periods (start_year end_year doubling_period : ℕ) : ℕ :=
  (end_year - start_year) / doubling_period

-- Function to estimate population after a number of complete doubling periods
def population_after_doubling (initial_pop doubling_periods : ℕ) : ℕ :=
  initial_pop * (2 ^ doubling_periods)

-- Theorem statement
theorem thalassa_population_estimate :
  let complete_periods := complete_doubling_periods base_year target_year doubling_period
  let pop_at_last_complete_period := population_after_doubling initial_population complete_periods
  let pop_at_next_complete_period := pop_at_last_complete_period * 2
  (pop_at_last_complete_period + pop_at_next_complete_period) / 2 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_thalassa_population_estimate_l2628_262874


namespace NUMINAMATH_CALUDE_problem_rectangle_triangle_count_l2628_262897

/-- A rectangle subdivided into smaller rectangles with diagonals -/
structure SubdividedRectangle where
  height : ℕ
  width : ℕ
  vertical_divisions : ℕ
  horizontal_divisions : ℕ

/-- Count the number of triangles in a subdivided rectangle -/
def count_triangles (r : SubdividedRectangle) : ℕ :=
  sorry

/-- The specific rectangle from the problem -/
def problem_rectangle : SubdividedRectangle := {
  height := 30,
  width := 40,
  vertical_divisions := 3,
  horizontal_divisions := 2
}

/-- Theorem stating that the number of triangles in the problem rectangle is 46 -/
theorem problem_rectangle_triangle_count :
  count_triangles problem_rectangle = 46 := by sorry

end NUMINAMATH_CALUDE_problem_rectangle_triangle_count_l2628_262897


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l2628_262893

noncomputable def f (a b x : ℝ) : ℝ := (a * x + b) * (Real.exp x + x + 2)

theorem tangent_line_and_inequality 
  (a b : ℝ) 
  (h1 : f a b 0 = 0) 
  (h2 : (deriv (f a b)) 0 = 6) :
  (a = 2 ∧ b = 0) ∧ 
  ∀ x > 0, f 2 0 x > 2 * Real.log x + 2 * x + 3 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l2628_262893


namespace NUMINAMATH_CALUDE_smallest_c_in_special_progression_l2628_262879

theorem smallest_c_in_special_progression (a b c : ℕ) : 
  a > b ∧ b > c ∧ c > 0 →  -- a, b, c are positive integers with a > b > c
  (b * b = a * c) →        -- a, b, c form a geometric progression
  (a + b = 2 * c) →        -- a, c, b form an arithmetic progression
  c ≥ 1 ∧                  -- c is at least 1
  (∀ k : ℕ, k > 0 ∧ k < c →
    ¬∃ x y : ℕ, x > y ∧ y > k ∧ 
    (y * y = x * k) ∧ 
    (x + y = 2 * k)) →     -- c is the smallest value satisfying the conditions
  c = 1                    -- The smallest possible value of c is 1
:= by sorry

end NUMINAMATH_CALUDE_smallest_c_in_special_progression_l2628_262879


namespace NUMINAMATH_CALUDE_rectangle_circle_ratio_l2628_262848

theorem rectangle_circle_ratio (r : ℝ) (h : r > 0) : 
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧ 
    (x + 2*y)^2 = 16 * π * r^2 ∧ 
    y = r * Real.sqrt π ∧
    x / y = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_ratio_l2628_262848


namespace NUMINAMATH_CALUDE_trig_identity_l2628_262860

theorem trig_identity (α β : Real) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2628_262860


namespace NUMINAMATH_CALUDE_joanne_coin_collection_l2628_262807

/-- Represents the coin collection problem for Joanne at the mall fountain -/
def coin_collection_problem (first_hour : ℕ) (next_two_hours : ℕ) (coins_given_away : ℕ) (total_after_four_hours : ℕ) : Prop :=
  let total_three_hours := first_hour + 2 * next_two_hours
  let total_before_giving := total_after_four_hours + coins_given_away
  let fourth_hour_collection := total_before_giving - total_three_hours
  fourth_hour_collection = 50

/-- Theorem stating the solution to Joanne's coin collection problem -/
theorem joanne_coin_collection :
  coin_collection_problem 15 35 15 120 := by
  sorry

#check joanne_coin_collection

end NUMINAMATH_CALUDE_joanne_coin_collection_l2628_262807


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2628_262844

/-- Given an arithmetic sequence {aₙ} with common difference d = 2 and a₄ = 3,
    prove that a₂ + a₈ = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = 2) →  -- arithmetic sequence with common difference 2
  a 4 = 3 →                    -- a₄ = 3
  a 2 + a 8 = 10 :=             -- prove a₂ + a₈ = 10
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2628_262844


namespace NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l2628_262862

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5}

-- Define the region where x + y ≤ 6
def region : Set (ℝ × ℝ) := {p ∈ rectangle | p.1 + p.2 ≤ 6}

-- Define the probability measure on the rectangle
noncomputable def prob : MeasureTheory.Measure (ℝ × ℝ) := sorry

-- State the theorem
theorem probability_x_plus_y_leq_6 : 
  prob region / prob rectangle = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_leq_6_l2628_262862


namespace NUMINAMATH_CALUDE_expand_and_simplify_solve_equation_l2628_262894

-- Problem 1
theorem expand_and_simplify (x y : ℝ) : 
  (x + 3*y)^2 - (x + 3*y)*(x - 3*y) = 6*x*y + 18*y^2 := by sorry

-- Problem 2
theorem solve_equation : 
  ∃ (x : ℝ), x / (2*x - 1) = 2 - 3 / (1 - 2*x) ∧ x = -1/3 := by sorry

end NUMINAMATH_CALUDE_expand_and_simplify_solve_equation_l2628_262894


namespace NUMINAMATH_CALUDE_dodo_is_sane_l2628_262847

-- Define the characters
inductive Character : Type
| Dodo : Character
| Lori : Character
| Eagle : Character

-- Define the "thinks" relation
def thinks (x y : Character) (p : Prop) : Prop := sorry

-- Define sanity
def is_sane (x : Character) : Prop := sorry

-- State the theorem
theorem dodo_is_sane :
  (thinks Dodo Lori (¬ is_sane Eagle)) →
  (thinks Lori Dodo (¬ is_sane Dodo)) →
  (thinks Eagle Dodo (is_sane Dodo)) →
  is_sane Dodo := by sorry

end NUMINAMATH_CALUDE_dodo_is_sane_l2628_262847


namespace NUMINAMATH_CALUDE_pencil_cost_l2628_262836

theorem pencil_cost (pen_price pencil_price : ℚ) : 
  3 * pen_price + 2 * pencil_price = 165/100 →
  4 * pen_price + 7 * pencil_price = 303/100 →
  pencil_price = 19155/100000 :=
by sorry

end NUMINAMATH_CALUDE_pencil_cost_l2628_262836


namespace NUMINAMATH_CALUDE_equal_selection_probability_l2628_262853

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

/-- Represents the population and sample characteristics -/
structure Population where
  total_items : ℕ
  first_grade : ℕ
  second_grade : ℕ
  third_grade : ℕ
  fourth_grade : ℕ
  sample_size : ℕ

/-- Calculates the probability of an item being selected for a given sampling method -/
def selection_probability (pop : Population) (method : SamplingMethod) : ℚ :=
  pop.sample_size / pop.total_items

/-- The main theorem stating that all sampling methods have the same selection probability -/
theorem equal_selection_probability (pop : Population) 
  (h1 : pop.total_items = 160)
  (h2 : pop.first_grade = 48)
  (h3 : pop.second_grade = 64)
  (h4 : pop.third_grade = 32)
  (h5 : pop.fourth_grade = 16)
  (h6 : pop.sample_size = 20)
  (h7 : pop.total_items = pop.first_grade + pop.second_grade + pop.third_grade + pop.fourth_grade) :
  ∀ m : SamplingMethod, selection_probability pop m = 1/8 := by
  sorry

#check equal_selection_probability

end NUMINAMATH_CALUDE_equal_selection_probability_l2628_262853


namespace NUMINAMATH_CALUDE_stratified_sampling_11th_grade_l2628_262887

theorem stratified_sampling_11th_grade (total_students : ℕ) (eleventh_grade_students : ℕ) (sample_size : ℕ) :
  total_students = 5000 →
  eleventh_grade_students = 1500 →
  sample_size = 30 →
  (eleventh_grade_students : ℚ) / (total_students : ℚ) * (sample_size : ℚ) = 9 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_11th_grade_l2628_262887


namespace NUMINAMATH_CALUDE_b_value_determination_l2628_262809

/-- The function f(x) defined as 4x³ + bx + 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := 4 * x^3 + b * x + 1

/-- Theorem stating that if f(x) ≥ 0 for all x in [-1, 1], then b = -3 -/
theorem b_value_determination (b : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f b x ≥ 0) → b = -3 := by
  sorry

end NUMINAMATH_CALUDE_b_value_determination_l2628_262809


namespace NUMINAMATH_CALUDE_sandwich_slices_count_l2628_262842

/-- Given the total number of sandwiches and the total number of bread slices,
    calculate the number of slices per sandwich. -/
def slices_per_sandwich (total_sandwiches : ℕ) (total_slices : ℕ) : ℚ :=
  total_slices / total_sandwiches

/-- Theorem stating that for 5 sandwiches and 15 slices, each sandwich consists of 3 slices. -/
theorem sandwich_slices_count :
  slices_per_sandwich 5 15 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_slices_count_l2628_262842


namespace NUMINAMATH_CALUDE_cl2_moles_required_l2628_262813

/-- Represents the stoichiometric ratio of Cl2 to CH4 in the reaction -/
def cl2_ch4_ratio : ℚ := 4

/-- Represents the number of moles of CH4 given -/
def ch4_moles : ℚ := 3

/-- Represents the number of moles of CCl4 produced -/
def ccl4_moles : ℚ := 3

/-- Theorem stating that the number of moles of Cl2 required is 12 -/
theorem cl2_moles_required : cl2_ch4_ratio * ch4_moles = 12 := by
  sorry

end NUMINAMATH_CALUDE_cl2_moles_required_l2628_262813


namespace NUMINAMATH_CALUDE_quadratic_monotonicity_l2628_262899

/-- A function f: ℝ → ℝ is monotonic on an interval [a, b] if it is either
    monotonically increasing or monotonically decreasing on that interval. -/
def IsMonotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f x ≤ f y) ∨
  (∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x)

/-- The quadratic function f(x) = 2x^2 - kx + 1 is monotonic on [1, 3]
    if and only if k ≤ 4 or k ≥ 12. -/
theorem quadratic_monotonicity (k : ℝ) :
  IsMonotonic (fun x => 2 * x^2 - k * x + 1) 1 3 ↔ k ≤ 4 ∨ k ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotonicity_l2628_262899


namespace NUMINAMATH_CALUDE_proper_subsets_of_m_n_l2628_262872

def S : Set (Set Char) := {{}, {'m'}, {'n'}}

theorem proper_subsets_of_m_n :
  {A : Set Char | A ⊂ {'m', 'n'}} = S := by sorry

end NUMINAMATH_CALUDE_proper_subsets_of_m_n_l2628_262872


namespace NUMINAMATH_CALUDE_double_discount_l2628_262825

/-- Calculates the final price as a percentage of the original price after applying two consecutive discounts -/
theorem double_discount (initial_discount coupon_discount : ℝ) :
  initial_discount = 0.4 →
  coupon_discount = 0.25 →
  (1 - initial_discount) * (1 - coupon_discount) = 0.45 :=
by
  sorry

#check double_discount

end NUMINAMATH_CALUDE_double_discount_l2628_262825


namespace NUMINAMATH_CALUDE_equidistant_centers_l2628_262805

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the structure for a triangle
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

-- Define the structure for a circle
structure Circle where
  center : Point2D
  radius : ℝ

def is_right_triangle (t : Triangle) : Prop := sorry

def altitude_to_hypotenuse (t : Triangle) : Point2D := sorry

def inscribed_circle (t : Triangle) : Circle := sorry

def touch_point_on_hypotenuse (c : Circle) (t : Triangle) : Point2D := sorry

def distance (p1 p2 : Point2D) : ℝ := sorry

theorem equidistant_centers (ABC : Triangle) (H₃ : Point2D) :
  is_right_triangle ABC →
  H₃ = altitude_to_hypotenuse ABC →
  let O := (inscribed_circle ABC).center
  let O₁ := (inscribed_circle ⟨ABC.A, ABC.C, H₃⟩).center
  let O₂ := (inscribed_circle ⟨ABC.B, ABC.C, H₃⟩).center
  let T := touch_point_on_hypotenuse (inscribed_circle ABC) ABC
  distance O T = distance O₁ T ∧ distance O T = distance O₂ T :=
by sorry

end NUMINAMATH_CALUDE_equidistant_centers_l2628_262805


namespace NUMINAMATH_CALUDE_power_of_product_l2628_262870

theorem power_of_product (a b : ℝ) : (a * b) ^ 3 = a ^ 3 * b ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l2628_262870


namespace NUMINAMATH_CALUDE_percentage_green_shirts_l2628_262869

theorem percentage_green_shirts (total_students : ℕ) (blue_percent red_percent : ℚ) (other_students : ℕ) :
  total_students = 600 →
  blue_percent = 45/100 →
  red_percent = 23/100 →
  other_students = 102 →
  (total_students - (blue_percent * total_students + red_percent * total_students + other_students)) / total_students = 15/100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_green_shirts_l2628_262869


namespace NUMINAMATH_CALUDE_initial_apples_count_l2628_262803

/-- The number of apples Sarah initially had in her bag -/
def initial_apples : ℕ := 25

/-- The number of apples Sarah gave to teachers -/
def apples_to_teachers : ℕ := 16

/-- The number of apples Sarah gave to friends -/
def apples_to_friends : ℕ := 5

/-- The number of apples Sarah ate -/
def apples_eaten : ℕ := 1

/-- The number of apples left in Sarah's bag when she got home -/
def apples_left : ℕ := 3

/-- Theorem stating that the initial number of apples equals the sum of apples given away, eaten, and left -/
theorem initial_apples_count : 
  initial_apples = apples_to_teachers + apples_to_friends + apples_eaten + apples_left :=
by sorry

end NUMINAMATH_CALUDE_initial_apples_count_l2628_262803


namespace NUMINAMATH_CALUDE_no_real_solutions_l2628_262864

theorem no_real_solutions : ¬∃ x : ℝ, (2*x - 3*x + 7)^2 + 2 = -|2*x| := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l2628_262864


namespace NUMINAMATH_CALUDE_jim_reading_speed_increase_l2628_262840

-- Define Jim's reading parameters
def original_rate : ℝ := 40 -- pages per hour
def original_total : ℝ := 600 -- pages per week
def time_reduction : ℝ := 4 -- hours
def new_total : ℝ := 660 -- pages per week

-- Theorem statement
theorem jim_reading_speed_increase :
  let original_time := original_total / original_rate
  let new_time := original_time - time_reduction
  let new_rate := new_total / new_time
  new_rate / original_rate = 1.5
  := by sorry

end NUMINAMATH_CALUDE_jim_reading_speed_increase_l2628_262840


namespace NUMINAMATH_CALUDE_min_rounds_for_sole_winner_l2628_262843

/-- Represents a chess tournament -/
structure ChessTournament where
  num_players : ℕ
  num_rounds : ℕ
  points_per_win : ℚ
  points_per_draw : ℚ
  points_per_loss : ℚ

/-- Checks if a tournament configuration allows for a sole winner -/
def has_sole_winner (t : ChessTournament) : Prop :=
  ∃ (leader_score : ℚ) (max_other_score : ℚ),
    leader_score > max_other_score ∧
    leader_score ≤ t.num_rounds * t.points_per_win ∧
    max_other_score ≤ (t.num_rounds - 1) * t.points_per_win + t.points_per_draw

/-- The main theorem stating the minimum number of rounds for a sole winner -/
theorem min_rounds_for_sole_winner :
  ∀ (t : ChessTournament),
    t.num_players = 10 →
    t.points_per_win = 1 →
    t.points_per_draw = 1/2 →
    t.points_per_loss = 0 →
    (∀ n : ℕ, n < 7 → ¬(has_sole_winner {num_players := t.num_players,
                                         num_rounds := n,
                                         points_per_win := t.points_per_win,
                                         points_per_draw := t.points_per_draw,
                                         points_per_loss := t.points_per_loss})) ∧
    (has_sole_winner {num_players := t.num_players,
                      num_rounds := 7,
                      points_per_win := t.points_per_win,
                      points_per_draw := t.points_per_draw,
                      points_per_loss := t.points_per_loss}) :=
by
  sorry

end NUMINAMATH_CALUDE_min_rounds_for_sole_winner_l2628_262843


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2628_262898

theorem fraction_to_decimal : (47 : ℚ) / (2^3 * 5^7) = 0.0000752 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2628_262898


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2628_262871

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- First term of the sequence
  a : ℚ
  -- Common difference of the sequence
  d : ℚ
  -- Sum of first 5 terms is 10
  sum_5 : (5 : ℚ) / 2 * (2 * a + 4 * d) = 10
  -- Sum of first 50 terms is 150
  sum_50 : (50 : ℚ) / 2 * (2 * a + 49 * d) = 150

/-- Properties of the 55th term and sum of first 55 terms -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  let sum_55 := (55 : ℚ) / 2 * (2 * seq.a + 54 * seq.d)
  let term_55 := seq.a + 54 * seq.d
  sum_55 = 171 ∧ term_55 = 4.31 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2628_262871


namespace NUMINAMATH_CALUDE_divisors_of_1800_power_l2628_262875

theorem divisors_of_1800_power (n : Nat) : 
  (∃ (a b c : Nat), (a + 1) * (b + 1) * (c + 1) = 180 ∧
   n = 2^a * 3^b * 5^c ∧ n ∣ 1800^1800) ↔ n ∈ Finset.range 109 :=
by sorry

#check divisors_of_1800_power

end NUMINAMATH_CALUDE_divisors_of_1800_power_l2628_262875


namespace NUMINAMATH_CALUDE_smallest_period_scaled_function_l2628_262856

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem smallest_period_scaled_function
  (f : ℝ → ℝ) (h : is_periodic f 10) :
  ∃ b : ℝ, b > 0 ∧ (∀ x, f ((x - b) / 2) = f (x / 2)) ∧
    ∀ b' : ℝ, 0 < b' → (∀ x, f ((x - b') / 2) = f (x / 2)) → b ≤ b' :=
sorry

end NUMINAMATH_CALUDE_smallest_period_scaled_function_l2628_262856


namespace NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2016_l2628_262819

/-- The last two digits of 7^n, for n ≥ 1 -/
def lastTwoDigits (n : ℕ) : ℕ :=
  (7^n) % 100

/-- The period of the last two digits of powers of 7 -/
def period : ℕ := 4

theorem last_two_digits_of_7_pow_2016 :
  lastTwoDigits 2016 = 01 :=
by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_7_pow_2016_l2628_262819


namespace NUMINAMATH_CALUDE_final_payment_calculation_l2628_262873

/-- Calculates the final amount John will pay for three articles with given costs and discounts, including sales tax. -/
theorem final_payment_calculation (cost_A cost_B cost_C : ℝ)
  (discount_A discount_B discount_C : ℝ) (sales_tax_rate : ℝ)
  (h_cost_A : cost_A = 200)
  (h_cost_B : cost_B = 300)
  (h_cost_C : cost_C = 400)
  (h_discount_A : discount_A = 0.5)
  (h_discount_B : discount_B = 0.3)
  (h_discount_C : discount_C = 0.4)
  (h_sales_tax : sales_tax_rate = 0.05) :
  let discounted_A := cost_A * (1 - discount_A)
  let discounted_B := cost_B * (1 - discount_B)
  let discounted_C := cost_C * (1 - discount_C)
  let total_discounted := discounted_A + discounted_B + discounted_C
  let final_amount := total_discounted * (1 + sales_tax_rate)
  final_amount = 577.5 := by sorry


end NUMINAMATH_CALUDE_final_payment_calculation_l2628_262873


namespace NUMINAMATH_CALUDE_find_divisor_l2628_262865

theorem find_divisor (n : ℕ) (d : ℕ) : 
  n % 44 = 0 ∧ n / 44 = 432 ∧ n % d = 8 → d = 50 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l2628_262865


namespace NUMINAMATH_CALUDE_ones_digit_of_large_power_l2628_262891

theorem ones_digit_of_large_power : ∃ n : ℕ, n < 10 ∧ 34^(34*(17^17)) ≡ n [ZMOD 10] :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ones_digit_of_large_power_l2628_262891


namespace NUMINAMATH_CALUDE_outstanding_consumer_credit_l2628_262846

/-- The total outstanding consumer installment credit in billions of dollars -/
def total_credit : ℝ := 855

/-- The automobile installment credit in billions of dollars -/
def auto_credit : ℝ := total_credit * 0.2

/-- The credit extended by automobile finance companies in billions of dollars -/
def finance_company_credit : ℝ := 57

theorem outstanding_consumer_credit :
  (auto_credit = total_credit * 0.2) ∧
  (finance_company_credit = 57) ∧
  (finance_company_credit = auto_credit / 3) →
  total_credit = 855 :=
by sorry

end NUMINAMATH_CALUDE_outstanding_consumer_credit_l2628_262846


namespace NUMINAMATH_CALUDE_smallest_winning_number_l2628_262880

def bernardo_wins (N : ℕ) : Prop :=
  2 * N < 1000 ∧
  2 * N + 60 < 1000 ∧
  4 * N + 120 < 1000 ∧
  4 * N + 180 < 1000 ∧
  8 * N + 360 < 1000 ∧
  8 * N + 420 < 1000 ∧
  16 * N + 840 < 1000 ∧
  16 * N + 900 ≥ 1000

theorem smallest_winning_number :
  bernardo_wins 5 ∧ ∀ k : ℕ, k < 5 → ¬bernardo_wins k :=
sorry

end NUMINAMATH_CALUDE_smallest_winning_number_l2628_262880


namespace NUMINAMATH_CALUDE_x_plus_y_value_l2628_262884

theorem x_plus_y_value (x y : ℝ) (h1 : |x| = 3) (h2 : |y| = 2) (h3 : x * y < 0) :
  x + y = 1 ∨ x + y = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l2628_262884


namespace NUMINAMATH_CALUDE_function_monotonicity_l2628_262845

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a + a^x else 3 + (a - 1) * x

-- State the theorem
theorem function_monotonicity (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) > 0) →
  a ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_function_monotonicity_l2628_262845


namespace NUMINAMATH_CALUDE_max_value_of_function_sum_of_powers_greater_than_one_l2628_262838

-- Part 1
theorem max_value_of_function (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  ∃ M : ℝ, M = 1 ∧ ∀ x > -1, (1 + x)^a - a * x ≤ M :=
sorry

-- Part 2
theorem sum_of_powers_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^b + b^a > 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_function_sum_of_powers_greater_than_one_l2628_262838


namespace NUMINAMATH_CALUDE_inequality_solution_l2628_262861

theorem inequality_solution (x : ℝ) : 
  (4 * x + 2 < (x - 1)^2 ∧ (x - 1)^2 < 5 * x + 5) ↔ 
  (x > 3 + Real.sqrt 10 ∧ x < (7 + Real.sqrt 65) / 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2628_262861


namespace NUMINAMATH_CALUDE_similar_triangles_height_l2628_262895

/-- Given two similar triangles with an area ratio of 1:9 and the height of the smaller triangle is 5 cm,
    prove that the corresponding height of the larger triangle is 15 cm. -/
theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 5 →
  area_ratio = 9 →
  ∃ h_large : ℝ, h_large = 15 ∧ h_large / h_small = Real.sqrt area_ratio :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_height_l2628_262895


namespace NUMINAMATH_CALUDE_edge_touching_sphere_radius_is_geometric_mean_l2628_262882

/-- Regular tetrahedron with inscribed, circumscribed, and edge-touching spheres -/
structure RegularTetrahedronWithSpheres where
  /-- Radius of the inscribed sphere -/
  r : ℝ
  /-- Radius of the circumscribed sphere -/
  R : ℝ
  /-- Radius of the sphere touching the edges -/
  ρ : ℝ
  /-- The circumscribed sphere's radius is three times the inscribed sphere's radius -/
  h_R_eq_3r : R = 3 * r
  /-- All radii are positive -/
  h_r_pos : r > 0
  h_R_pos : R > 0
  h_ρ_pos : ρ > 0

/-- 
The radius of the sphere touching the edges of a regular tetrahedron 
is the geometric mean of the radii of the inscribed and circumscribed spheres
-/
theorem edge_touching_sphere_radius_is_geometric_mean 
  (t : RegularTetrahedronWithSpheres) : t.ρ ^ 2 = t.R * t.r := by
  sorry

end NUMINAMATH_CALUDE_edge_touching_sphere_radius_is_geometric_mean_l2628_262882


namespace NUMINAMATH_CALUDE_shortest_side_l2628_262804

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

-- Theorem statement
theorem shortest_side (t : Triangle) (h : t.a^2 + t.b^2 > 5 * t.c^2) : 
  t.c < t.a ∧ t.c < t.b := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_l2628_262804


namespace NUMINAMATH_CALUDE_square_difference_l2628_262821

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 64) 
  (h2 : x * y = 15) : 
  (x - y)^2 = 4 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l2628_262821


namespace NUMINAMATH_CALUDE_car_catch_up_time_l2628_262866

/-- The time it takes for a car to catch up with a truck, given their speeds and the truck's head start -/
theorem car_catch_up_time (truck_speed car_speed : ℝ) (head_start : ℝ) : 
  truck_speed = 45 →
  car_speed = 60 →
  head_start = 1 →
  (car_speed * t - truck_speed * t = truck_speed * head_start) →
  t = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_car_catch_up_time_l2628_262866


namespace NUMINAMATH_CALUDE_subtracted_amount_l2628_262835

theorem subtracted_amount (n : ℚ) (x : ℚ) : 
  n = 25 / 3 → 3 * n + 15 = 6 * n - x → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l2628_262835


namespace NUMINAMATH_CALUDE_madeline_flowers_l2628_262896

theorem madeline_flowers (red : ℕ) (white : ℕ) (blue_percentage : ℚ) (total : ℕ) :
  red = 4 →
  white = 2 →
  blue_percentage = 40 / 100 →
  (red + white : ℚ) + blue_percentage * total = total →
  total = 10 :=
by sorry

end NUMINAMATH_CALUDE_madeline_flowers_l2628_262896


namespace NUMINAMATH_CALUDE_smallest_m_exceeds_15_l2628_262832

def sum_digits_after_decimal (n : ℚ) : ℕ :=
  sorry

def exceeds_15 (m : ℕ) : Prop :=
  sum_digits_after_decimal (1 / 3^m) > 15

theorem smallest_m_exceeds_15 : 
  (∀ k < 7, ¬(exceeds_15 k)) ∧ exceeds_15 7 :=
sorry

end NUMINAMATH_CALUDE_smallest_m_exceeds_15_l2628_262832


namespace NUMINAMATH_CALUDE_power_value_from_equation_l2628_262817

theorem power_value_from_equation (a b : ℝ) : 
  |a - 2| + (b + 3)^2 = 0 → b^a = 9 := by
sorry

end NUMINAMATH_CALUDE_power_value_from_equation_l2628_262817


namespace NUMINAMATH_CALUDE_lettuce_calories_l2628_262839

/-- Calculates the calories in lettuce given the conditions of Jackson's meal -/
theorem lettuce_calories (
  pizza_crust : ℝ)
  (pizza_cheese : ℝ)
  (salad_dressing : ℝ)
  (total_calories_consumed : ℝ)
  (h1 : pizza_crust = 600)
  (h2 : pizza_cheese = 400)
  (h3 : salad_dressing = 210)
  (h4 : total_calories_consumed = 330) :
  let pizza_pepperoni := pizza_crust / 3
  let total_pizza := pizza_crust + pizza_pepperoni + pizza_cheese
  let pizza_consumed := total_pizza / 5
  let salad_consumed := total_calories_consumed - pizza_consumed
  let total_salad := salad_consumed * 4
  let lettuce := (total_salad - salad_dressing) / 3
  lettuce = 50 := by sorry

end NUMINAMATH_CALUDE_lettuce_calories_l2628_262839


namespace NUMINAMATH_CALUDE_correct_verb_form_surround_is_correct_verb_l2628_262806

/-- Represents the grammatical form of a verb --/
inductive VerbForm
| Base
| PresentParticiple
| PastParticiple
| PresentPerfect

/-- Represents the structure of a sentence --/
structure Sentence :=
  (verb : VerbForm)
  (isImperative : Bool)
  (hasFutureTense : Bool)

/-- Determines if a given sentence structure is correct --/
def isCorrectSentenceStructure (s : Sentence) : Prop :=
  s.isImperative ∧ s.hasFutureTense ∧ s.verb = VerbForm.Base

/-- The specific sentence structure in the problem --/
def givenSentence : Sentence :=
  { verb := VerbForm.Base,
    isImperative := true,
    hasFutureTense := true }

/-- Theorem stating that the given sentence structure is correct --/
theorem correct_verb_form :
  isCorrectSentenceStructure givenSentence :=
sorry

/-- Theorem stating that "Surround" is the correct verb to use --/
theorem surround_is_correct_verb :
  givenSentence.verb = VerbForm.Base →
  isCorrectSentenceStructure givenSentence →
  "Surround" = "Surround" :=
sorry

end NUMINAMATH_CALUDE_correct_verb_form_surround_is_correct_verb_l2628_262806


namespace NUMINAMATH_CALUDE_matrix_N_computation_l2628_262851

theorem matrix_N_computation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec (![3, -2]) = ![5, 1])
  (h2 : N.mulVec (![(-2), 4]) = ![0, -2]) :
  N.mulVec (![7, 0]) = ![17.5, 0] := by
sorry

end NUMINAMATH_CALUDE_matrix_N_computation_l2628_262851


namespace NUMINAMATH_CALUDE_gcd_108_45_l2628_262890

theorem gcd_108_45 : Nat.gcd 108 45 = 9 := by
  sorry

end NUMINAMATH_CALUDE_gcd_108_45_l2628_262890


namespace NUMINAMATH_CALUDE_shoppers_equal_share_l2628_262808

theorem shoppers_equal_share (isabella sam giselle : ℕ) : 
  isabella = sam + 45 →
  isabella = giselle + 15 →
  giselle = 120 →
  (isabella + sam + giselle) / 3 = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_shoppers_equal_share_l2628_262808


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2628_262892

theorem quadratic_roots_properties (a b c x₁ x₂ : ℝ) (ha : a ≠ 0)
  (hx₁ : a * x₁^2 + b * x₁ + c = 0) (hx₂ : a * x₂^2 + b * x₂ + c = 0) :
  x₁^2 + x₂^2 = (b^2 - 2*a*c) / a^2 ∧ x₁^3 + x₂^3 = (3*a*b*c - b^3) / a^3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2628_262892


namespace NUMINAMATH_CALUDE_bike_distance_proof_l2628_262811

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a bike traveling at 8 m/s for 6 seconds covers 48 meters -/
theorem bike_distance_proof :
  let speed : ℝ := 8
  let time : ℝ := 6
  distance speed time = 48 := by
  sorry

end NUMINAMATH_CALUDE_bike_distance_proof_l2628_262811


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2628_262888

theorem solution_set_quadratic_inequality (x : ℝ) :
  x * (x - 1) > 0 ↔ x < 0 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2628_262888


namespace NUMINAMATH_CALUDE_smallest_natural_with_congruences_l2628_262815

theorem smallest_natural_with_congruences (m : ℕ) : 
  (∀ k : ℕ, k < m → (k % 3 ≠ 1 ∨ k % 7 ≠ 5 ∨ k % 11 ≠ 4)) → 
  m % 3 = 1 → 
  m % 7 = 5 → 
  m % 11 = 4 → 
  m % 4 = 3 := by
sorry

end NUMINAMATH_CALUDE_smallest_natural_with_congruences_l2628_262815


namespace NUMINAMATH_CALUDE_cos_135_degrees_l2628_262800

theorem cos_135_degrees : Real.cos (135 * π / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_135_degrees_l2628_262800


namespace NUMINAMATH_CALUDE_arithmetic_seq_fifth_term_l2628_262829

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

theorem arithmetic_seq_fifth_term
  (a : ℕ → ℤ)
  (h_arith : arithmetic_seq a)
  (h_eq : 2 * a 9 = a 12 + 6) :
  a 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_seq_fifth_term_l2628_262829


namespace NUMINAMATH_CALUDE_widget_production_difference_l2628_262883

/-- Given David's widget production scenario, this theorem proves the difference
    in production between two consecutive days. -/
theorem widget_production_difference
  (t : ℕ) -- Number of hours worked on the first day
  (w : ℕ) -- Number of widgets produced per hour on the first day
  (h1 : w = 2 * t^2) -- Relation between w and t
  : w * t - (w + 3) * (t - 3) = 6 * t^2 - 3 * t + 9 :=
by sorry

end NUMINAMATH_CALUDE_widget_production_difference_l2628_262883


namespace NUMINAMATH_CALUDE_diagonal_intersection_fixed_point_l2628_262820

/-- An ellipse with equation x^2/4 + y^2/3 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- A point is on the ellipse C -/
def on_ellipse_C (p : ℝ × ℝ) : Prop := ellipse_C p.1 p.2

/-- Quadrilateral MNPQ with vertices on ellipse C -/
structure Quadrilateral_MNPQ where
  M : ℝ × ℝ
  N : ℝ × ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  hM : on_ellipse_C M
  hN : on_ellipse_C N
  hP : on_ellipse_C P
  hQ : on_ellipse_C Q
  hMQ_NP : M.2 + Q.2 = 0 ∧ N.2 + P.2 = 0  -- MQ || NP and MQ ⊥ x-axis
  hS : ∃ t : ℝ, (M.2 - N.2) * (4 - P.1) = (M.1 - 4) * (P.2 - N.2) ∧
                (Q.2 - P.2) * (4 - N.1) = (Q.1 - 4) * (N.2 - P.2)  -- MN and QP intersect at S(4,0)

/-- The theorem to be proved -/
theorem diagonal_intersection_fixed_point (q : Quadrilateral_MNPQ) :
  ∃ (I : ℝ × ℝ), I = (1, 0) ∧
  (q.M.2 - q.P.2) * (I.1 - q.N.1) = (q.M.1 - I.1) * (I.2 - q.N.2) ∧
  (q.N.2 - q.Q.2) * (I.1 - q.M.1) = (q.N.1 - I.1) * (I.2 - q.M.2) := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersection_fixed_point_l2628_262820


namespace NUMINAMATH_CALUDE_scarves_per_box_l2628_262837

theorem scarves_per_box (num_boxes : ℕ) (total_pieces : ℕ) : 
  num_boxes = 6 → 
  total_pieces = 60 → 
  ∃ (scarves_per_box : ℕ), 
    scarves_per_box * num_boxes * 2 = total_pieces ∧ 
    scarves_per_box = 5 :=
by sorry

end NUMINAMATH_CALUDE_scarves_per_box_l2628_262837


namespace NUMINAMATH_CALUDE_line_perp_from_plane_perp_l2628_262850

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (perp_plane : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line : Line → Line → Prop)

-- State the theorem
theorem line_perp_from_plane_perp
  (m n : Line) (α β : Plane)
  (h1 : perp_line_plane m α)
  (h2 : perp_line_plane n β)
  (h3 : perp_plane α β) :
  perp_line m n :=
sorry

end NUMINAMATH_CALUDE_line_perp_from_plane_perp_l2628_262850


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2628_262812

open Complex

theorem complex_modulus_problem (m : ℝ) : 
  (↑(1 + m * I) * (3 + I) * I).im ≠ 0 →
  (↑(1 + m * I) * (3 + I) * I).re = 0 →
  abs ((m + 3 * I) / (1 - I)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2628_262812


namespace NUMINAMATH_CALUDE_raffle_ticket_sales_difference_l2628_262816

theorem raffle_ticket_sales_difference :
  ∀ (friday_sales saturday_sales sunday_sales : ℕ),
    friday_sales = 181 →
    saturday_sales = 2 * friday_sales →
    sunday_sales = 78 →
    saturday_sales - sunday_sales = 284 :=
by
  sorry

end NUMINAMATH_CALUDE_raffle_ticket_sales_difference_l2628_262816


namespace NUMINAMATH_CALUDE_simplify_expressions_l2628_262867

theorem simplify_expressions :
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 →
    (Real.sqrt (2 / 3) + 3 * Real.sqrt (1 / 6) - (1 / 2) * Real.sqrt 54 = -(2 * Real.sqrt 6) / 3) ∧
    (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 12 + Real.sqrt 24 = 4 + Real.sqrt 6)) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2628_262867


namespace NUMINAMATH_CALUDE_tylers_puppies_l2628_262833

theorem tylers_puppies (num_dogs : ℕ) (puppies_per_dog : ℕ) (total_puppies : ℕ) : 
  num_dogs = 15 → puppies_per_dog = 5 → total_puppies = num_dogs * puppies_per_dog → total_puppies = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_tylers_puppies_l2628_262833


namespace NUMINAMATH_CALUDE_y_value_at_27_l2628_262810

-- Define the relation between y and x
def y (k : ℝ) (x : ℝ) : ℝ := k * x^(1/3)

-- State the theorem
theorem y_value_at_27 (k : ℝ) :
  y k 8 = 4 → y k 27 = 6 := by
  sorry

end NUMINAMATH_CALUDE_y_value_at_27_l2628_262810


namespace NUMINAMATH_CALUDE_certain_number_calculation_l2628_262886

theorem certain_number_calculation : 5 * 3 + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_calculation_l2628_262886


namespace NUMINAMATH_CALUDE_fraction_simplification_l2628_262802

theorem fraction_simplification (x y : ℝ) (h : y = x / (1 - 2*x)) :
  (2*x - 3*x*y - 2*y) / (y + x*y - x) = -7/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2628_262802


namespace NUMINAMATH_CALUDE_sin_double_angle_plus_5pi_6_l2628_262858

theorem sin_double_angle_plus_5pi_6 (α : Real) 
  (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.sin (2 * α + 5 * π / 6) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_plus_5pi_6_l2628_262858


namespace NUMINAMATH_CALUDE_lana_extra_flowers_l2628_262857

/-- The number of extra flowers Lana picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Theorem stating that Lana picked 280 extra flowers -/
theorem lana_extra_flowers :
  extra_flowers 860 920 1500 = 280 := by
  sorry

end NUMINAMATH_CALUDE_lana_extra_flowers_l2628_262857


namespace NUMINAMATH_CALUDE_fewer_sevens_100_l2628_262834

/-- A function that represents an arithmetic expression using sevens -/
def SevenExpression : Type := ℕ → ℚ

/-- Count the number of sevens used in an expression -/
def count_sevens : SevenExpression → ℕ := sorry

/-- Evaluate a SevenExpression -/
def evaluate : SevenExpression → ℚ := sorry

/-- Theorem: There exists an expression using fewer than 10 sevens that evaluates to 100 -/
theorem fewer_sevens_100 : ∃ e : SevenExpression, count_sevens e < 10 ∧ evaluate e = 100 := by sorry

end NUMINAMATH_CALUDE_fewer_sevens_100_l2628_262834


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2628_262859

/-- Given vectors a and b, if (k*a + b) is parallel to (a - 3*b), then k = -1/3 --/
theorem parallel_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (ha : a = (1, 2))
    (hb : b = (-3, 2))
    (h_parallel : ∃ (t : ℝ), t • (k • a + b) = (a - 3 • b)) :
  k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2628_262859


namespace NUMINAMATH_CALUDE_unique_solutions_l2628_262801

/-- A pair of positive integers (m, n) satisfies the given conditions if
    m^2 - 4n and n^2 - 4m are both perfect squares. -/
def satisfies_conditions (m n : ℕ+) : Prop :=
  ∃ k l : ℕ, (m : ℤ)^2 - 4*(n : ℤ) = (k : ℤ)^2 ∧ (n : ℤ)^2 - 4*(m : ℤ) = (l : ℤ)^2

/-- The theorem stating that the only pairs of positive integers (m, n) satisfying
    the conditions are (4, 4), (5, 6), and (6, 5). -/
theorem unique_solutions :
  ∀ m n : ℕ+, satisfies_conditions m n ↔ 
    ((m = 4 ∧ n = 4) ∨ (m = 5 ∧ n = 6) ∨ (m = 6 ∧ n = 5)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l2628_262801


namespace NUMINAMATH_CALUDE_mode_of_data_set_l2628_262814

def data_set : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_data_set :
  mode data_set = 3 := by
  sorry

end NUMINAMATH_CALUDE_mode_of_data_set_l2628_262814


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2628_262826

theorem sum_of_coefficients (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 15625 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2628_262826


namespace NUMINAMATH_CALUDE_fuel_used_fraction_l2628_262818

def car_speed : ℝ := 50
def fuel_efficiency : ℝ := 30
def tank_capacity : ℝ := 15
def travel_time : ℝ := 5

theorem fuel_used_fraction (speed : ℝ) (efficiency : ℝ) (capacity : ℝ) (time : ℝ)
  (h1 : speed = car_speed)
  (h2 : efficiency = fuel_efficiency)
  (h3 : capacity = tank_capacity)
  (h4 : time = travel_time) :
  (speed * time / efficiency) / capacity = 5 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fuel_used_fraction_l2628_262818


namespace NUMINAMATH_CALUDE_correct_average_weight_l2628_262830

/-- Given a class of boys with an initially miscalculated average weight and a single misread weight, 
    calculate the correct average weight. -/
theorem correct_average_weight 
  (n : ℕ) 
  (initial_avg : ℝ) 
  (misread_weight : ℝ) 
  (correct_weight : ℝ) 
  (h1 : n = 20) 
  (h2 : initial_avg = 58.4) 
  (h3 : misread_weight = 56) 
  (h4 : correct_weight = 61) : 
  (n * initial_avg + (correct_weight - misread_weight)) / n = 58.65 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l2628_262830


namespace NUMINAMATH_CALUDE_special_circle_equation_l2628_262881

/-- A circle with center (a, b) and radius r satisfying specific conditions -/
structure SpecialCircle where
  a : ℝ
  b : ℝ
  r : ℝ
  center_on_line : a - 3 * b = 0
  tangent_to_y_axis : r = |a|
  chord_length : ((a - b) ^ 2 / 2) + 7 = r ^ 2

/-- The equation of a circle given its center (a, b) and radius r -/
def circle_equation (c : SpecialCircle) (x y : ℝ) : Prop :=
  (x - c.a) ^ 2 + (y - c.b) ^ 2 = c.r ^ 2

/-- The main theorem stating that a SpecialCircle satisfies one of two specific equations -/
theorem special_circle_equation (c : SpecialCircle) :
  (∀ x y, circle_equation c x y ↔ (x - 3) ^ 2 + (y - 1) ^ 2 = 9) ∨
  (∀ x y, circle_equation c x y ↔ (x + 3) ^ 2 + (y + 1) ^ 2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_special_circle_equation_l2628_262881


namespace NUMINAMATH_CALUDE_order_relation_l2628_262823

noncomputable def a (e : ℝ) : ℝ := 5 * Real.log (2^e)
noncomputable def b (e : ℝ) : ℝ := 2 * Real.log (5^e)
def c : ℝ := 10

theorem order_relation (e : ℝ) (h : e > 0) : c > a e ∧ a e > b e := by
  sorry

end NUMINAMATH_CALUDE_order_relation_l2628_262823


namespace NUMINAMATH_CALUDE_B_equals_set_l2628_262852

def A : Set ℤ := {-1, 2, 3, 4}

def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2*x + 2}

theorem B_equals_set : B = {2, 5, 10} := by sorry

end NUMINAMATH_CALUDE_B_equals_set_l2628_262852
