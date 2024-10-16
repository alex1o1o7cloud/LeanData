import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l3699_369985

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3

theorem tangent_line_and_monotonicity (a : ℝ) :
  (a = 1 → ∃ (m b : ℝ), m = 9 ∧ b = 8 ∧ 
    ∀ x y, y = f 1 x → (x = -1 → y = m*x + b)) ∧
  (a = 0 → ∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ∧
  (a < 0 → (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 2*a → f a x₁ > f a x₂) ∧
           (∀ x₁ x₂, 2*a < x₁ ∧ x₁ < x₂ ∧ x₂ < 0 → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
  (a > 0 → (∀ x₁ x₂, x₁ < x₂ ∧ x₂ < 0 → f a x₁ < f a x₂) ∧
           (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 2*a → f a x₁ > f a x₂) ∧
           (∀ x₁ x₂, 2*a < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_monotonicity_l3699_369985


namespace NUMINAMATH_CALUDE_average_age_combined_group_l3699_369971

/-- Calculate the average age of a combined group of sixth-graders, parents, and teachers -/
theorem average_age_combined_group
  (n_students : ℕ) (avg_age_students : ℝ)
  (n_parents : ℕ) (avg_age_parents : ℝ)
  (n_teachers : ℕ) (avg_age_teachers : ℝ)
  (h_students : n_students = 40 ∧ avg_age_students = 12)
  (h_parents : n_parents = 50 ∧ avg_age_parents = 35)
  (h_teachers : n_teachers = 10 ∧ avg_age_teachers = 45) :
  let total_people := n_students + n_parents + n_teachers
  let total_age := n_students * avg_age_students + n_parents * avg_age_parents + n_teachers * avg_age_teachers
  total_age / total_people = 26.8 :=
by sorry

end NUMINAMATH_CALUDE_average_age_combined_group_l3699_369971


namespace NUMINAMATH_CALUDE_radius_is_ten_l3699_369929

/-- A square with a circle tangent to two adjacent sides -/
structure TangentSquare where
  /-- Side length of the square -/
  side : ℝ
  /-- Radius of the circle -/
  radius : ℝ
  /-- Length of segment cut off from vertices B and D -/
  tangent_segment : ℝ
  /-- Length of segment cut off from one non-tangent side -/
  intersect_segment1 : ℝ
  /-- Length of segment cut off from the other non-tangent side -/
  intersect_segment2 : ℝ
  /-- The circle is tangent to two adjacent sides -/
  tangent_condition : side = radius + tangent_segment
  /-- The circle intersects the other two sides -/
  intersect_condition : side = radius + intersect_segment1 + intersect_segment2

/-- The radius of the circle is 10 given the specific measurements -/
theorem radius_is_ten (ts : TangentSquare) 
  (h1 : ts.tangent_segment = 8)
  (h2 : ts.intersect_segment1 = 4)
  (h3 : ts.intersect_segment2 = 2) : 
  ts.radius = 10 := by
  sorry

end NUMINAMATH_CALUDE_radius_is_ten_l3699_369929


namespace NUMINAMATH_CALUDE_years_before_aziz_birth_l3699_369973

def current_year : ℕ := 2021
def aziz_age : ℕ := 36
def parents_move_year : ℕ := 1982

theorem years_before_aziz_birth : 
  current_year - aziz_age - parents_move_year = 3 := by sorry

end NUMINAMATH_CALUDE_years_before_aziz_birth_l3699_369973


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3699_369909

theorem diophantine_equation_solution : ∃ (a b c : ℕ), a^3 + b^4 = c^5 ∧ a = 4 ∧ b = 16 ∧ c = 18 := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3699_369909


namespace NUMINAMATH_CALUDE_asteroid_speed_comparison_l3699_369923

/-- Asteroid observation and speed comparison -/
theorem asteroid_speed_comparison 
  (distance_X13 : ℝ) 
  (time_X13 : ℝ) 
  (speed_X13 : ℝ) 
  (speed_Y14 : ℝ) 
  (h1 : distance_X13 = 2000) 
  (h2 : speed_X13 = distance_X13 / time_X13) 
  (h3 : speed_Y14 = 3 * speed_X13) : 
  speed_Y14 - speed_X13 = speed_X13 := by
  sorry

end NUMINAMATH_CALUDE_asteroid_speed_comparison_l3699_369923


namespace NUMINAMATH_CALUDE_problem_statement_l3699_369925

theorem problem_statement (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 5) :
  b / (a + b) + c / (b + c) + a / (c + a) = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3699_369925


namespace NUMINAMATH_CALUDE_max_leftover_stickers_l3699_369933

theorem max_leftover_stickers (y : ℕ+) : 
  ∃ (q r : ℕ), y = 12 * q + r ∧ r ≤ 11 ∧ 
  ∀ (q' r' : ℕ), y = 12 * q' + r' → r' ≤ r :=
sorry

end NUMINAMATH_CALUDE_max_leftover_stickers_l3699_369933


namespace NUMINAMATH_CALUDE_kishore_rent_expense_l3699_369947

def monthly_salary (savings : ℕ) : ℕ := savings * 10

def total_expenses_excluding_rent : ℕ := 1500 + 4500 + 2500 + 2000 + 5200

def rent_expense (salary savings : ℕ) : ℕ :=
  salary - (total_expenses_excluding_rent + savings)

theorem kishore_rent_expense :
  rent_expense (monthly_salary 2300) 2300 = 5000 := by
  sorry

end NUMINAMATH_CALUDE_kishore_rent_expense_l3699_369947


namespace NUMINAMATH_CALUDE_direct_proportion_iff_m_eq_neg_one_l3699_369997

/-- A function f(x) is a direct proportion function if there exists a non-zero constant k such that f(x) = kx for all x. -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x, f x = k * x

/-- The function y = (m-1)x + m^2 - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x + m^2 - 1

theorem direct_proportion_iff_m_eq_neg_one (m : ℝ) :
  is_direct_proportion (f m) ↔ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_iff_m_eq_neg_one_l3699_369997


namespace NUMINAMATH_CALUDE_taras_rowing_speed_l3699_369976

/-- Tara's rowing problem -/
theorem taras_rowing_speed 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (time : ℝ) 
  (current_speed : ℝ) 
  (h1 : downstream_distance = 20) 
  (h2 : upstream_distance = 4) 
  (h3 : time = 2) 
  (h4 : current_speed = 2) :
  ∃ v : ℝ, 
    v + current_speed = downstream_distance / time ∧ 
    v - current_speed = upstream_distance / time ∧ 
    v = 8 := by
sorry

end NUMINAMATH_CALUDE_taras_rowing_speed_l3699_369976


namespace NUMINAMATH_CALUDE_gift_exchange_probability_l3699_369930

theorem gift_exchange_probability :
  let num_boys : ℕ := 4
  let num_girls : ℕ := 4
  let total_people : ℕ := num_boys + num_girls
  let total_configurations : ℕ := num_boys ^ total_people
  let valid_configurations : ℕ := 288

  (valid_configurations : ℚ) / total_configurations = 9 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_gift_exchange_probability_l3699_369930


namespace NUMINAMATH_CALUDE_eating_relationship_l3699_369920

def A : Set ℝ := {-1, 1/2, 1}

def B (a : ℝ) : Set ℝ := {x | a * x^2 = 1}

def full_eating (X Y : Set ℝ) : Prop := X ⊆ Y ∨ Y ⊆ X

def partial_eating (X Y : Set ℝ) : Prop := 
  (∃ x, x ∈ X ∩ Y) ∧ ¬(X ⊆ Y) ∧ ¬(Y ⊆ X)

theorem eating_relationship (a : ℝ) : 
  (a ≥ 0) → (full_eating A (B a) ∨ partial_eating A (B a)) ↔ a ∈ ({0, 1, 4} : Set ℝ) :=
sorry

end NUMINAMATH_CALUDE_eating_relationship_l3699_369920


namespace NUMINAMATH_CALUDE_match_problem_solution_l3699_369914

/-- Represents the number of matches in each pile -/
structure MatchPiles :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Performs the described operations on the piles -/
def performOperations (piles : MatchPiles) : MatchPiles :=
  let step1 := MatchPiles.mk
    (piles.first - piles.second)
    (piles.second + piles.second)
    piles.third
  let step2 := MatchPiles.mk
    step1.first
    (step1.second - step1.third)
    (step1.third + step1.third)
  MatchPiles.mk
    (step2.first + step2.third)
    step2.second
    (step2.third - step2.first)

/-- Theorem stating the solution to the match problem -/
theorem match_problem_solution (piles : MatchPiles) :
  piles.first + piles.second + piles.third = 96 →
  let final := performOperations piles
  final.first = final.second ∧ final.second = final.third →
  piles = MatchPiles.mk 44 28 24 := by
  sorry

end NUMINAMATH_CALUDE_match_problem_solution_l3699_369914


namespace NUMINAMATH_CALUDE_grandson_age_l3699_369943

theorem grandson_age (grandmother_age grandson_age : ℕ) : 
  grandmother_age = grandson_age * 12 →
  grandmother_age + grandson_age = 65 →
  grandson_age = 5 := by
sorry

end NUMINAMATH_CALUDE_grandson_age_l3699_369943


namespace NUMINAMATH_CALUDE_circle_square_area_difference_l3699_369928

/-- The difference between the areas of the non-overlapping portions of a circle and a square -/
theorem circle_square_area_difference (r c s : ℝ) (h1 : r = 3) (h2 : s = 2) : 
  (π * r^2 - s^2) = 9 * π - 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_square_area_difference_l3699_369928


namespace NUMINAMATH_CALUDE_solve_equation_l3699_369924

theorem solve_equation : ∀ x : ℝ, x + 1 = 2 → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3699_369924


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3699_369952

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 6*x > 20} = {x : ℝ | x < -2 ∨ x > 10} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3699_369952


namespace NUMINAMATH_CALUDE_mineral_water_case_price_l3699_369988

/-- The price of a case of mineral water -/
def case_price (daily_consumption : ℚ) (days : ℕ) (bottles_per_case : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / ((daily_consumption * days) / bottles_per_case)

/-- Theorem stating the price of a case of mineral water is $12 -/
theorem mineral_water_case_price :
  case_price (1/2) 240 24 60 = 12 := by
  sorry

end NUMINAMATH_CALUDE_mineral_water_case_price_l3699_369988


namespace NUMINAMATH_CALUDE_january_salary_l3699_369961

/-- Represents the salary structure for five months --/
structure SalaryStructure where
  jan : ℕ
  feb : ℕ
  mar : ℕ
  apr : ℕ
  may : ℕ

/-- Theorem stating the salary for January given the conditions --/
theorem january_salary (s : SalaryStructure) : s.jan = 4000 :=
  sorry

/-- The average salary for the first four months is 8000 --/
axiom avg_first_four (s : SalaryStructure) : 
  (s.jan + s.feb + s.mar + s.apr) / 4 = 8000

/-- The average salary for the last four months (including bonus) is 8800 --/
axiom avg_last_four (s : SalaryStructure) : 
  (s.feb + s.mar + s.apr + s.may + 1500) / 4 = 8800

/-- The salary for May (excluding bonus) is 6500 --/
axiom may_salary (s : SalaryStructure) : s.may = 6500

/-- February had a deduction of 700 --/
axiom feb_deduction (s : SalaryStructure) (feb_original : ℕ) : 
  s.feb = feb_original - 700

/-- No deductions in other months --/
axiom no_other_deductions (s : SalaryStructure) 
  (jan_original mar_original apr_original : ℕ) : 
  s.jan = jan_original ∧ s.mar = mar_original ∧ s.apr = apr_original

end NUMINAMATH_CALUDE_january_salary_l3699_369961


namespace NUMINAMATH_CALUDE_circumcircle_area_l3699_369907

/-- An isosceles triangle with two sides of length 6 and a base of length 4 -/
structure IsoscelesTriangle where
  side : ℝ
  base : ℝ
  is_isosceles : side = 6 ∧ base = 4

/-- A circle passing through the vertices of an isosceles triangle -/
def CircumCircle (t : IsoscelesTriangle) : ℝ → Prop :=
  fun area => area = 16 * Real.pi

/-- The theorem stating that the area of the circumcircle of the given isosceles triangle is 16π -/
theorem circumcircle_area (t : IsoscelesTriangle) : 
  ∃ area, CircumCircle t area :=
sorry

end NUMINAMATH_CALUDE_circumcircle_area_l3699_369907


namespace NUMINAMATH_CALUDE_kite_area_is_18_l3699_369982

/-- The area of a kite with width 6 units and height 7 units, where each unit is one inch. -/
def kite_area : ℝ := 18

/-- The width of the kite in units. -/
def kite_width : ℕ := 6

/-- The height of the kite in units. -/
def kite_height : ℕ := 7

/-- Theorem stating that the area of the kite is 18 square inches. -/
theorem kite_area_is_18 : kite_area = 18 := by sorry

end NUMINAMATH_CALUDE_kite_area_is_18_l3699_369982


namespace NUMINAMATH_CALUDE_boys_ratio_in_class_l3699_369941

theorem boys_ratio_in_class (p : ℝ) 
  (h1 : p ≥ 0 ∧ p ≤ 1) -- Probability is between 0 and 1
  (h2 : p = 3/4 * (1 - p)) -- Condition from the problem
  : p = 3/7 := by
  sorry

end NUMINAMATH_CALUDE_boys_ratio_in_class_l3699_369941


namespace NUMINAMATH_CALUDE_age_difference_l3699_369994

/-- Given four individuals with ages a, b, c, and d, prove that c is 10 years younger than a. -/
theorem age_difference (a b c d : ℝ) 
  (sum_ab_bc : a + b = b + c + 10)
  (sum_cd_ad : c + d = a + d - 15)
  (ratio_ad : a / d = 7 / 4) :
  a - c = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3699_369994


namespace NUMINAMATH_CALUDE_min_value_theorem_l3699_369977

theorem min_value_theorem (c a b : ℝ) (hc : c > 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (heq : 4 * a^2 - 2 * a * b + b^2 - c = 0)
  (hmax : ∀ (a' b' : ℝ), a' ≠ 0 → b' ≠ 0 → 4 * a'^2 - 2 * a' * b' + b'^2 - c = 0 →
    |2 * a + b| ≥ |2 * a' + b'|) :
  (1 / a + 2 / b + 4 / c) ≥ -1 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3699_369977


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3699_369998

theorem circle_area_ratio : 
  ∀ (r1 r2 : ℝ), r1 > 0 → r2 = 3 * r1 → 
  (π * r2^2) / (π * r1^2) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3699_369998


namespace NUMINAMATH_CALUDE_hidden_dots_count_l3699_369948

/-- The sum of numbers on a single die -/
def dieDots : ℕ := 21

/-- The number of dice stacked -/
def numDice : ℕ := 4

/-- The visible numbers on the stacked dice -/
def visibleNumbers : List ℕ := [1, 2, 2, 3, 3, 5, 6]

/-- The number of visible faces -/
def visibleFaces : ℕ := 7

/-- The number of hidden faces -/
def hiddenFaces : ℕ := 17

theorem hidden_dots_count : 
  numDice * dieDots - visibleNumbers.sum = 62 :=
sorry

end NUMINAMATH_CALUDE_hidden_dots_count_l3699_369948


namespace NUMINAMATH_CALUDE_runner_parade_time_l3699_369911

/-- Calculates the time taken for a runner to travel from the front to the end of a moving parade. -/
theorem runner_parade_time (parade_length : ℝ) (parade_speed : ℝ) (runner_speed : ℝ) 
  (h1 : parade_length = 2)
  (h2 : parade_speed = 3)
  (h3 : runner_speed = 6) :
  (parade_length / (runner_speed - parade_speed)) * 60 = 40 := by
  sorry

end NUMINAMATH_CALUDE_runner_parade_time_l3699_369911


namespace NUMINAMATH_CALUDE_stratified_sample_properties_l3699_369919

/-- Represents the grades of parts in a batch -/
inductive Grade
  | First
  | Second
  | Third

/-- Structure representing a batch of parts -/
structure Batch :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Structure representing a sample drawn from a batch -/
structure Sample :=
  (first : ℕ)
  (second : ℕ)
  (third : ℕ)

/-- Function to check if a sample is valid for a given batch -/
def isValidSample (b : Batch) (s : Sample) : Prop :=
  s.first + s.second + s.third = 20 ∧
  s.first ≤ b.first ∧
  s.second ≤ b.second ∧
  s.third ≤ b.third

/-- Theorem stating the properties of the stratified sample -/
theorem stratified_sample_properties (b : Batch) (s : Sample) :
  b.first = 24 →
  b.second = 36 →
  s.third = 10 →
  isValidSample b s →
  b.third = 60 ∧ s.second = 6 := by sorry

end NUMINAMATH_CALUDE_stratified_sample_properties_l3699_369919


namespace NUMINAMATH_CALUDE_total_songs_bought_l3699_369956

/-- The number of country albums Megan bought -/
def country_albums : ℕ := 2

/-- The number of pop albums Megan bought -/
def pop_albums : ℕ := 8

/-- The number of songs in each album -/
def songs_per_album : ℕ := 7

/-- The total number of albums Megan bought -/
def total_albums : ℕ := country_albums + pop_albums

/-- Theorem: The total number of songs Megan bought is 70 -/
theorem total_songs_bought : total_albums * songs_per_album = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_songs_bought_l3699_369956


namespace NUMINAMATH_CALUDE_place_mat_length_l3699_369901

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) : 
  r = 6 ∧ n = 8 ∧ w = 1 ∧ 
  (∀ i : Fin n, ∃ p₁ p₂ : ℝ × ℝ, 
    (p₁.1 - r)^2 + p₁.2^2 = r^2 ∧
    (p₂.1 - r)^2 + p₂.2^2 = r^2 ∧
    (p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2 = w^2) ∧
  (∀ i : Fin n, ∃ q₁ q₂ : ℝ × ℝ,
    (q₁.1 - r)^2 + q₁.2^2 < r^2 ∧
    (q₂.1 - r)^2 + q₂.2^2 < r^2 ∧
    (q₂.1 - q₁.1)^2 + (q₂.2 - q₁.2)^2 = y^2 ∧
    (∃ j : Fin n, j ≠ i ∧ (q₂.1 = q₁.1 ∨ q₂.2 = q₁.2))) →
  y = 3 * Real.sqrt (2 - Real.sqrt 2) := by
sorry

end NUMINAMATH_CALUDE_place_mat_length_l3699_369901


namespace NUMINAMATH_CALUDE_equation_transformation_l3699_369992

theorem equation_transformation :
  ∀ x : ℝ, x^2 - 6*x = 0 ↔ (x - 3)^2 = 9 := by sorry

end NUMINAMATH_CALUDE_equation_transformation_l3699_369992


namespace NUMINAMATH_CALUDE_equation_solution_l3699_369951

theorem equation_solution (t : ℝ) : 
  (5 * 3^t + Real.sqrt (25 * 9^t) = 50) ↔ (t = Real.log 5 / Real.log 3) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3699_369951


namespace NUMINAMATH_CALUDE_f_not_prime_l3699_369908

def f (n : ℕ+) : ℤ := n.val^6 - 550 * n.val^3 + 324

theorem f_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (f n)) := by
  sorry

end NUMINAMATH_CALUDE_f_not_prime_l3699_369908


namespace NUMINAMATH_CALUDE_inverse_sum_equals_one_l3699_369995

theorem inverse_sum_equals_one (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  1 / x + 1 / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_equals_one_l3699_369995


namespace NUMINAMATH_CALUDE_diagonal_length_after_triangle_removal_l3699_369975

/-- The diagonal length of a quadrilateral formed by removing two equal-area right triangles from opposite corners of a square --/
theorem diagonal_length_after_triangle_removal (s : ℝ) (A : ℝ) (h1 : s = 20) (h2 : A = 50) :
  let x := Real.sqrt (2 * A)
  let diagonal := Real.sqrt ((s - x)^2 + (s - x)^2)
  diagonal = 10 * Real.sqrt 2 := by
  sorry

#check diagonal_length_after_triangle_removal

end NUMINAMATH_CALUDE_diagonal_length_after_triangle_removal_l3699_369975


namespace NUMINAMATH_CALUDE_expression_equals_negative_two_over_tan_l3699_369980

theorem expression_equals_negative_two_over_tan (α : Real) 
  (h : α ∈ Set.Ioo π (3 * π / 2)) : 
  Real.sqrt ((1 + Real.cos α) / (1 - Real.cos α)) - 
  Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) = 
  -2 / Real.tan α := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_two_over_tan_l3699_369980


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3699_369990

/-- Given a quadratic function f(x) = ax^2 + bx - 3 where a ≠ 0,
    if f(2) = f(4), then f(6) = -3 -/
theorem quadratic_function_property (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x - 3
  f 2 = f 4 → f 6 = -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3699_369990


namespace NUMINAMATH_CALUDE_harry_book_pages_l3699_369972

/-- Given that Selena's book has x pages and Harry's book has y fewer pages than half of Selena's book,
    prove that the number of pages in Harry's book is (x/2) - y. -/
theorem harry_book_pages (x y : ℕ) (selena_pages : ℕ) (harry_pages : ℕ) 
    (h1 : selena_pages = x)
    (h2 : harry_pages = selena_pages / 2 - y) :
  harry_pages = x / 2 - y := by
  sorry

end NUMINAMATH_CALUDE_harry_book_pages_l3699_369972


namespace NUMINAMATH_CALUDE_exists_five_digit_not_sum_of_beautiful_l3699_369999

/-- A beautiful number is a number consisting of identical digits. -/
def is_beautiful (n : ℕ) : Prop :=
  ∃ d : ℕ, d ≤ 9 ∧ ∃ k : ℕ, k > 0 ∧ n = d * (10^k - 1) / 9

/-- The sum of beautiful numbers with pairwise different lengths. -/
def sum_of_beautiful (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), 
    n = a * 11111 + b * 1111 + c * 111 + d * 11 + e * 1 ∧
    is_beautiful (a * 11111) ∧ 
    is_beautiful (b * 1111) ∧ 
    is_beautiful (c * 111) ∧ 
    is_beautiful (d * 11) ∧ 
    is_beautiful e

/-- Theorem: There exists a five-digit number that cannot be represented as a sum of beautiful numbers with pairwise different lengths. -/
theorem exists_five_digit_not_sum_of_beautiful : 
  ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ ¬(sum_of_beautiful n) := by
  sorry

end NUMINAMATH_CALUDE_exists_five_digit_not_sum_of_beautiful_l3699_369999


namespace NUMINAMATH_CALUDE_loss_percentage_is_29_percent_l3699_369987

-- Define the markup percentage
def markup : ℝ := 0.40

-- Define the discount percentage
def discount : ℝ := 0.07857142857142857

-- Define the loss percentage we want to prove
def target_loss_percentage : ℝ := 0.29

-- Theorem statement
theorem loss_percentage_is_29_percent (cost_price : ℝ) (cost_price_positive : cost_price > 0) :
  let marked_price := cost_price * (1 + markup)
  let selling_price := marked_price * (1 - discount)
  let loss := cost_price - selling_price
  let loss_percentage := loss / cost_price
  loss_percentage = target_loss_percentage :=
by sorry

end NUMINAMATH_CALUDE_loss_percentage_is_29_percent_l3699_369987


namespace NUMINAMATH_CALUDE_han_xin_counting_l3699_369954

theorem han_xin_counting (n : ℕ) : n ≥ 53 ∧ n % 3 = 2 ∧ n % 5 = 3 ∧ n % 7 = 4 →
  ∀ m : ℕ, m < 53 → ¬(m % 3 = 2 ∧ m % 5 = 3 ∧ m % 7 = 4) := by
  sorry

end NUMINAMATH_CALUDE_han_xin_counting_l3699_369954


namespace NUMINAMATH_CALUDE_percentage_calculation_l3699_369917

theorem percentage_calculation (N : ℝ) (P : ℝ) : 
  N = 150 → 
  (3 / 5) * N = 90 → 
  (P / 100) * 90 = 36 → 
  P = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_calculation_l3699_369917


namespace NUMINAMATH_CALUDE_distance_difference_l3699_369935

theorem distance_difference (john_distance nina_distance : ℝ) 
  (h1 : john_distance = 0.7)
  (h2 : nina_distance = 0.4) :
  john_distance - nina_distance = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l3699_369935


namespace NUMINAMATH_CALUDE_number_pair_theorem_l3699_369970

theorem number_pair_theorem (S P x y : ℝ) (h1 : x + y = S) (h2 : x * y = P) :
  ((x = (S + Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4*P)) / 2) ∨
   (x = (S - Real.sqrt (S^2 - 4*P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4*P)) / 2)) ∧
  S^2 ≥ 4*P := by
  sorry

end NUMINAMATH_CALUDE_number_pair_theorem_l3699_369970


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3699_369921

theorem rectangle_dimension_change (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_L := L * (1 - 0.25)
  let new_W := W * (1 + 1/3)
  new_L * new_W = L * W := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3699_369921


namespace NUMINAMATH_CALUDE_car_tank_size_l3699_369993

/-- Calculates the size of a car's gas tank given the advertised mileage, actual miles driven, and the difference between advertised and actual mileage. -/
theorem car_tank_size 
  (advertised_mileage : ℝ) 
  (miles_driven : ℝ) 
  (mileage_difference : ℝ) : 
  advertised_mileage = 35 →
  miles_driven = 372 →
  mileage_difference = 4 →
  miles_driven / (advertised_mileage - mileage_difference) = 12 := by
    sorry

#check car_tank_size

end NUMINAMATH_CALUDE_car_tank_size_l3699_369993


namespace NUMINAMATH_CALUDE_range_of_a_l3699_369962

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 4}
def B (a : ℝ) : Set ℝ := {x | 2*a ≤ x ∧ x ≤ a+3}

theorem range_of_a (a : ℝ) : A ∪ B a = A ↔ a ≤ -4 ∨ (2 ≤ a ∧ a ≤ 3) ∨ a > 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3699_369962


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3699_369939

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of Hydrogen atoms in the compound -/
def num_H : ℕ := 1

/-- The number of Bromine atoms in the compound -/
def num_Br : ℕ := 1

/-- The number of Oxygen atoms in the compound -/
def num_O : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 
  num_H * atomic_weight_H + num_Br * atomic_weight_Br + num_O * atomic_weight_O

/-- Theorem stating that the molecular weight of the compound is 128.91 g/mol -/
theorem compound_molecular_weight : molecular_weight = 128.91 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3699_369939


namespace NUMINAMATH_CALUDE_factorization_x_squared_plus_2x_l3699_369918

theorem factorization_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x+2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_plus_2x_l3699_369918


namespace NUMINAMATH_CALUDE_cube_spheres_diagonal_outside_length_l3699_369989

/-- Given a cube with edge length 1 and identical spheres centered at each vertex,
    where each sphere touches three neighboring spheres, the length of the part of
    the space diagonal of the cube that lies outside the spheres is √3 - 1. -/
theorem cube_spheres_diagonal_outside_length :
  let cube_edge_length : ℝ := 1
  let sphere_radius : ℝ := cube_edge_length / 2
  let cube_diagonal : ℝ := Real.sqrt 3
  let diagonal_inside_spheres : ℝ := 2 * sphere_radius
  cube_diagonal - diagonal_inside_spheres = Real.sqrt 3 - 1 := by
  sorry

#check cube_spheres_diagonal_outside_length

end NUMINAMATH_CALUDE_cube_spheres_diagonal_outside_length_l3699_369989


namespace NUMINAMATH_CALUDE_frustum_volume_l3699_369957

/-- The volume of a frustum obtained from a cone with specific properties -/
theorem frustum_volume (r h l : ℝ) : 
  r > 0 ∧ h > 0 ∧ l > 0 ∧  -- Positive dimensions
  π * r * l = 2 * π ∧     -- Lateral surface area condition
  l = 2 * r ∧             -- Relationship between slant height and radius
  h = Real.sqrt 3 ∧       -- Height of the cone
  r = 1 →                 -- Radius of the cone
  (7 * Real.sqrt 3 * π) / 24 = 
    (7 / 8) * ((π * r^2 * h) / 3) := by
  sorry

end NUMINAMATH_CALUDE_frustum_volume_l3699_369957


namespace NUMINAMATH_CALUDE_new_average_is_65_l3699_369960

/-- Calculates the new average marks per paper after additional marks are added. -/
def new_average_marks (num_papers : ℕ) (original_average : ℚ) (additional_marks_geo : ℕ) (additional_marks_hist : ℕ) : ℚ :=
  (num_papers * original_average + additional_marks_geo + additional_marks_hist) / num_papers

/-- Proves that the new average marks per paper is 65 given the specified conditions. -/
theorem new_average_is_65 :
  new_average_marks 11 63 20 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_new_average_is_65_l3699_369960


namespace NUMINAMATH_CALUDE_sixth_root_of_two_squared_equals_cube_root_of_two_l3699_369936

theorem sixth_root_of_two_squared_equals_cube_root_of_two : 
  (2^2)^(1/6) = 2^(1/3) := by sorry

end NUMINAMATH_CALUDE_sixth_root_of_two_squared_equals_cube_root_of_two_l3699_369936


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l3699_369932

def C : Set Nat := {33, 35, 36, 39, 41}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ (∀ (m : Nat), m ∈ C → ∀ (p q : Nat), Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q) ∧ n = 36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l3699_369932


namespace NUMINAMATH_CALUDE_trout_ratio_is_three_to_one_l3699_369902

/-- The ratio of trouts caught by Caleb's dad to those caught by Caleb -/
def trout_ratio (caleb_trouts : ℕ) (dad_extra_trouts : ℕ) : ℚ :=
  (caleb_trouts + dad_extra_trouts) / caleb_trouts

theorem trout_ratio_is_three_to_one :
  trout_ratio 2 4 = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_trout_ratio_is_three_to_one_l3699_369902


namespace NUMINAMATH_CALUDE_expansion_terms_count_l3699_369938

theorem expansion_terms_count (N : ℕ) : 
  (Nat.choose N 5 = 3003) ↔ (N = 15) := by sorry

end NUMINAMATH_CALUDE_expansion_terms_count_l3699_369938


namespace NUMINAMATH_CALUDE_circle_lines_theorem_l3699_369927

/-- The number of points on the circle -/
def n : ℕ := 5

/-- The total number of lines between any two points -/
def total_lines (m : ℕ) : ℕ := m * (m - 1) / 2

/-- The number of lines between immediate neighbors -/
def neighbor_lines (m : ℕ) : ℕ := m

/-- The number of valid lines (excluding immediate neighbors) -/
def valid_lines (m : ℕ) : ℕ := total_lines m - neighbor_lines m

theorem circle_lines_theorem : valid_lines n = 5 := by sorry

end NUMINAMATH_CALUDE_circle_lines_theorem_l3699_369927


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_l3699_369905

/-- A quadratic polynomial q(x) satisfying specific conditions -/
def q (x : ℚ) : ℚ := (16/7) * x^2 + (32/7) * x - 240/7

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions : 
  q (-5) = 0 ∧ q 3 = 0 ∧ q 2 = -16 := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_l3699_369905


namespace NUMINAMATH_CALUDE_prime_square_mod_360_l3699_369966

theorem prime_square_mod_360 (p : Nat) (h_prime : Prime p) (h_gt_5 : p > 5) :
  (p^2 : Nat) % 360 = 1 ∨ (p^2 : Nat) % 360 = 289 := by
  sorry

#check prime_square_mod_360

end NUMINAMATH_CALUDE_prime_square_mod_360_l3699_369966


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3699_369942

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a > b - 1) ∧
  (∃ a b, a > b - 1 ∧ ¬(a > b)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l3699_369942


namespace NUMINAMATH_CALUDE_reflection_coordinates_l3699_369944

/-- Given points P and M in a 2D plane, this function returns the coordinates of Q, 
    which is the reflection of P about M. -/
def reflection_point (P M : ℝ × ℝ) : ℝ × ℝ :=
  (2 * M.1 - P.1, 2 * M.2 - P.2)

theorem reflection_coordinates :
  let P : ℝ × ℝ := (1, -2)
  let M : ℝ × ℝ := (3, 0)
  reflection_point P M = (5, 2) := by sorry

end NUMINAMATH_CALUDE_reflection_coordinates_l3699_369944


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l3699_369978

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {(x, y) | y = x^2}

/-- The point Q -/
def Q : ℝ × ℝ := (20, 14)

/-- The line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | y - Q.2 = m * (x - Q.1)}

/-- The condition for non-intersection -/
def no_intersection (m : ℝ) : Prop :=
  line_through_Q m ∩ P = ∅

/-- The theorem statement -/
theorem parabola_line_intersection :
  ∃ r s : ℝ, (∀ m : ℝ, no_intersection m ↔ r < m ∧ m < s) ∧ r + s = 80 := by sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l3699_369978


namespace NUMINAMATH_CALUDE_min_cost_2009_proof_l3699_369931

/-- Represents the available coin denominations in rubles -/
inductive Coin : Type
  | one : Coin
  | two : Coin
  | five : Coin
  | ten : Coin

/-- The value of a coin in rubles -/
def coin_value : Coin → ℕ
  | Coin.one => 1
  | Coin.two => 2
  | Coin.five => 5
  | Coin.ten => 10

/-- An arithmetic expression using coins and operations -/
inductive Expr : Type
  | coin : Coin → Expr
  | add : Expr → Expr → Expr
  | sub : Expr → Expr → Expr
  | mul : Expr → Expr → Expr
  | div : Expr → Expr → Expr

/-- Evaluates an expression to its numeric value -/
def eval : Expr → ℕ
  | Expr.coin c => coin_value c
  | Expr.add e1 e2 => eval e1 + eval e2
  | Expr.sub e1 e2 => eval e1 - eval e2
  | Expr.mul e1 e2 => eval e1 * eval e2
  | Expr.div e1 e2 => eval e1 / eval e2

/-- Calculates the total cost of an expression in rubles -/
def cost : Expr → ℕ
  | Expr.coin c => coin_value c
  | Expr.add e1 e2 => cost e1 + cost e2
  | Expr.sub e1 e2 => cost e1 + cost e2
  | Expr.mul e1 e2 => cost e1 + cost e2
  | Expr.div e1 e2 => cost e1 + cost e2

/-- The minimum cost to create an expression equal to 2009 -/
def min_cost_2009 : ℕ := 23

theorem min_cost_2009_proof :
  ∀ e : Expr, eval e = 2009 → cost e ≥ min_cost_2009 :=
by sorry

end NUMINAMATH_CALUDE_min_cost_2009_proof_l3699_369931


namespace NUMINAMATH_CALUDE_max_A_value_l3699_369958

-- Define the structure of our number configuration
structure NumberConfig where
  A : ℕ
  B : ℕ
  C : ℕ
  D : ℕ
  E : ℕ
  F : ℕ
  G : ℕ
  H : ℕ
  I : ℕ
  J : ℕ

-- Define the properties of the number configuration
def validConfig (n : NumberConfig) : Prop :=
  n.A > n.B ∧ n.B > n.C ∧
  n.D > n.E ∧ n.E > n.F ∧
  n.G > n.H ∧ n.H > n.I ∧ n.I > n.J ∧
  ∃ k : ℕ, n.D = 3 * k + 3 ∧ n.E = 3 * k ∧ n.F = 3 * k - 3 ∧
  ∃ m : ℕ, n.G = 2 * m + 1 ∧ n.H = 2 * m - 1 ∧ n.I = 2 * m - 3 ∧ n.J = 2 * m - 5 ∧
  n.A + n.B + n.C = 9

-- Theorem statement
theorem max_A_value (n : NumberConfig) (h : validConfig n) :
  n.A ≤ 8 ∧ ∃ m : NumberConfig, validConfig m ∧ m.A = 8 :=
sorry

end NUMINAMATH_CALUDE_max_A_value_l3699_369958


namespace NUMINAMATH_CALUDE_base6_120_to_base2_l3699_369910

/-- Converts a number from base 6 to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ :=
  (n / 100) * 36 + ((n / 10) % 10) * 6 + (n % 10)

/-- Converts a number from base 10 to base 2 --/
def base10ToBase2 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec toBinary (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else toBinary (m / 2) ((m % 2) :: acc)
    toBinary n []

theorem base6_120_to_base2 :
  base10ToBase2 (base6ToBase10 120) = [1, 1, 0, 0, 0, 0] := by
  sorry

end NUMINAMATH_CALUDE_base6_120_to_base2_l3699_369910


namespace NUMINAMATH_CALUDE_person_speed_l3699_369940

/-- Given a person crossing a street, calculate their speed in km/hr -/
theorem person_speed (distance : ℝ) (time : ℝ) (h1 : distance = 720) (h2 : time = 12) :
  distance / 1000 / (time / 60) = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_person_speed_l3699_369940


namespace NUMINAMATH_CALUDE_m_range_l3699_369974

def p (m : ℝ) : Prop := ∀ x y : ℝ, (x + y - m = 0) → ((x - 1)^2 + y^2 = 1) → False

def q (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, (x₁^2 - x₁ + m - 4 = 0) ∧ (x₂^2 - x₂ + m - 4 = 0) ∧ (x₁ * x₂ < 0)

theorem m_range : 
  ∀ m : ℝ, (p m ∨ q m) → ¬(p m) → m ∈ Set.Icc (1 - Real.sqrt 2) (1 + Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3699_369974


namespace NUMINAMATH_CALUDE_wrong_to_correct_ratio_l3699_369979

theorem wrong_to_correct_ratio (total_sums correct_sums : ℕ) 
  (h1 : total_sums = 36)
  (h2 : correct_sums = 12) : 
  (total_sums - correct_sums) / correct_sums = 2 := by
  sorry

end NUMINAMATH_CALUDE_wrong_to_correct_ratio_l3699_369979


namespace NUMINAMATH_CALUDE_bank_checks_total_amount_l3699_369926

theorem bank_checks_total_amount : 
  let million_won_checks : ℕ := 25
  let hundred_thousand_won_checks : ℕ := 8
  let million_won_value : ℕ := 1000000
  let hundred_thousand_won_value : ℕ := 100000
  (million_won_checks * million_won_value + hundred_thousand_won_checks * hundred_thousand_won_value : ℕ) = 25800000 :=
by
  sorry

end NUMINAMATH_CALUDE_bank_checks_total_amount_l3699_369926


namespace NUMINAMATH_CALUDE_cylinder_cone_sphere_volume_l3699_369959

/-- Given a cylinder with volume 150π cm³, prove that the sum of the volumes of a cone 
    with the same base radius and height as the cylinder, and a sphere with the same 
    radius as the cylinder, is equal to 50π + (4/3)π * (∛150)² cm³. -/
theorem cylinder_cone_sphere_volume 
  (r h : ℝ) 
  (h_cylinder_volume : π * r^2 * h = 150 * π) :
  (1/3 * π * r^2 * h) + (4/3 * π * r^3) = 50 * π + 4/3 * π * (150^(2/3)) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cone_sphere_volume_l3699_369959


namespace NUMINAMATH_CALUDE_seokjin_math_score_l3699_369955

/-- Given Seokjin's scores and average, prove his math score -/
theorem seokjin_math_score 
  (korean_score : ℕ) 
  (english_score : ℕ) 
  (average_score : ℕ) 
  (h1 : korean_score = 93)
  (h2 : english_score = 91)
  (h3 : average_score = 89)
  (h4 : (korean_score + english_score + math_score) / 3 = average_score) :
  math_score = 83 :=
by
  sorry

end NUMINAMATH_CALUDE_seokjin_math_score_l3699_369955


namespace NUMINAMATH_CALUDE_factorization_equality_l3699_369950

theorem factorization_equality (x y : ℝ) : 3 * x^2 * y - 6 * x = 3 * x * (x * y - 2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3699_369950


namespace NUMINAMATH_CALUDE_seventeen_in_sample_l3699_369969

/-- Represents a systematic sampling with equal intervals -/
structure SystematicSampling where
  start : ℕ
  interval : ℕ
  size : ℕ

/-- Checks if a number is included in the systematic sampling -/
def isInSample (s : SystematicSampling) (n : ℕ) : Prop :=
  ∃ k : ℕ, k < s.size ∧ n = s.start + k * s.interval

/-- Theorem: Given a systematic sampling that includes 5, 23, and 29, it also includes 17 -/
theorem seventeen_in_sample (s : SystematicSampling) 
  (h5 : isInSample s 5) 
  (h23 : isInSample s 23) 
  (h29 : isInSample s 29) : 
  isInSample s 17 := by
  sorry

end NUMINAMATH_CALUDE_seventeen_in_sample_l3699_369969


namespace NUMINAMATH_CALUDE_negation_relationship_l3699_369915

theorem negation_relationship (a : ℝ) : 
  ¬(∀ a, a > 0 → a^2 > a) ∧ ¬(∀ a, a^2 ≤ a → a ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_relationship_l3699_369915


namespace NUMINAMATH_CALUDE_peach_basket_problem_l3699_369983

theorem peach_basket_problem (baskets : Nat) (total_peaches : Nat) (green_excess : Nat) :
  baskets = 2 →
  green_excess = 2 →
  total_peaches = 12 →
  ∃ red_peaches : Nat, red_peaches * baskets + (red_peaches + green_excess) * baskets = total_peaches ∧ red_peaches = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_peach_basket_problem_l3699_369983


namespace NUMINAMATH_CALUDE_total_distance_travelled_l3699_369900

theorem total_distance_travelled (distance_by_land distance_by_sea : ℕ) 
  (h1 : distance_by_land = 451)
  (h2 : distance_by_sea = 150) : 
  distance_by_land + distance_by_sea = 601 := by
sorry

end NUMINAMATH_CALUDE_total_distance_travelled_l3699_369900


namespace NUMINAMATH_CALUDE_club_size_after_five_years_l3699_369967

def club_growth (initial_members : ℕ) (executives : ℕ) (years : ℕ) : ℕ :=
  let regular_members := initial_members - executives
  let final_regular_members := regular_members * (2 ^ years)
  final_regular_members + executives

theorem club_size_after_five_years :
  club_growth 18 6 5 = 390 := by sorry

end NUMINAMATH_CALUDE_club_size_after_five_years_l3699_369967


namespace NUMINAMATH_CALUDE_ethans_net_income_l3699_369965

/-- Calculates Ethan's net income after deductions for a 5-week period -/
def calculate_net_income (hourly_wage : ℚ) (hours_per_day : ℕ) (days_per_week : ℕ) 
  (total_weeks : ℕ) (tax_rate : ℚ) (health_insurance_per_week : ℚ) (retirement_rate : ℚ) : ℚ :=
  let gross_income := hourly_wage * hours_per_day * days_per_week * total_weeks
  let income_tax := tax_rate * gross_income
  let health_insurance := health_insurance_per_week * total_weeks
  let retirement_contribution := retirement_rate * gross_income
  let total_deductions := income_tax + health_insurance + retirement_contribution
  gross_income - total_deductions

/-- Theorem stating that Ethan's net income after deductions for a 5-week period is $2447 -/
theorem ethans_net_income : 
  calculate_net_income 18 8 5 5 (15/100) 65 (8/100) = 2447 := by
  sorry

end NUMINAMATH_CALUDE_ethans_net_income_l3699_369965


namespace NUMINAMATH_CALUDE_power_fraction_equals_two_l3699_369981

theorem power_fraction_equals_two : (2^4 - 2) / (2^3 - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_fraction_equals_two_l3699_369981


namespace NUMINAMATH_CALUDE_sandy_potatoes_l3699_369913

theorem sandy_potatoes (nancy_potatoes : ℕ) (total_potatoes : ℕ) (sandy_potatoes : ℕ) : 
  nancy_potatoes = 6 → 
  total_potatoes = 13 → 
  total_potatoes = nancy_potatoes + sandy_potatoes → 
  sandy_potatoes = 7 := by
sorry

end NUMINAMATH_CALUDE_sandy_potatoes_l3699_369913


namespace NUMINAMATH_CALUDE_no_real_solutions_l3699_369912

theorem no_real_solutions :
  ¬ ∃ x : ℝ, x + Real.sqrt (x + 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3699_369912


namespace NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3699_369906

/-- Given a bus that travels at 43 kmph including stoppages and stops for 8.4 minutes per hour,
    its speed excluding stoppages is 50 kmph. -/
theorem bus_speed_excluding_stoppages (speed_with_stops : ℝ) (stoppage_time : ℝ) :
  speed_with_stops = 43 →
  stoppage_time = 8.4 →
  (60 - stoppage_time) / 60 * speed_with_stops = 50 := by
  sorry

#check bus_speed_excluding_stoppages

end NUMINAMATH_CALUDE_bus_speed_excluding_stoppages_l3699_369906


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3699_369903

theorem absolute_value_inequality (y : ℝ) : 
  |((7 - y) / 4)| ≤ 3 ↔ -5 ≤ y ∧ y ≤ 19 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3699_369903


namespace NUMINAMATH_CALUDE_quadratic_solution_range_l3699_369934

/-- The range of t for which the quadratic equation x^2 - 4x + 1 - t = 0 has solutions in (0, 7/2) -/
theorem quadratic_solution_range (t : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ x < 7/2 ∧ x^2 - 4*x + 1 - t = 0) ↔ -3 ≤ t ∧ t < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_range_l3699_369934


namespace NUMINAMATH_CALUDE_toms_remaining_candy_l3699_369996

theorem toms_remaining_candy (initial_boxes : ℕ) (boxes_given_away : ℕ) (pieces_per_box : ℕ) : 
  initial_boxes = 14 → boxes_given_away = 8 → pieces_per_box = 3 →
  (initial_boxes - boxes_given_away) * pieces_per_box = 18 := by
  sorry

end NUMINAMATH_CALUDE_toms_remaining_candy_l3699_369996


namespace NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_l3699_369984

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17

theorem sum_first_six_primes_mod_seventh : (first_six_primes.sum % seventh_prime) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_six_primes_mod_seventh_l3699_369984


namespace NUMINAMATH_CALUDE_tree_height_average_l3699_369968

def tree_heights : List ℕ → Prop
  | [h1, h2, h3, h4, h5] =>
    h2 = 6 ∧
    (h1 = 2 * h2 ∨ 2 * h1 = h2) ∧
    (h2 = 2 * h3 ∨ 2 * h2 = h3) ∧
    (h3 = 2 * h4 ∨ 2 * h3 = h4) ∧
    (h4 = 2 * h5 ∨ 2 * h4 = h5)
  | _ => False

theorem tree_height_average :
  ∀ (heights : List ℕ),
    tree_heights heights →
    (heights.sum : ℚ) / heights.length = 66 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tree_height_average_l3699_369968


namespace NUMINAMATH_CALUDE_point_outside_ellipse_l3699_369946

theorem point_outside_ellipse (m n : ℝ) 
  (h_intersect : ∃ x y : ℝ, m * x + n * y = 4 ∧ x^2 + y^2 = 4) :
  m^2 / 4 + n^2 / 3 > 1 := by
  sorry

end NUMINAMATH_CALUDE_point_outside_ellipse_l3699_369946


namespace NUMINAMATH_CALUDE_tangent_circles_count_l3699_369904

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency relation between two circles
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2 ∨
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius - c2.radius)^2

theorem tangent_circles_count (c1 c2 : Circle) : 
  c1.radius = 1 →
  c2.radius = 1 →
  are_tangent c1 c2 →
  ∃ (s : Finset Circle), 
    s.card = 6 ∧ 
    (∀ c ∈ s, c.radius = 3 ∧ are_tangent c c1 ∧ are_tangent c c2) ∧
    (∀ c : Circle, c.radius = 3 ∧ are_tangent c c1 ∧ are_tangent c c2 → c ∈ s) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_count_l3699_369904


namespace NUMINAMATH_CALUDE_range_of_m_l3699_369945

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x ∈ Set.Icc 0 1, x^2 - m*x - 2 = 0

def q (m : ℝ) : Prop := ∀ x ≥ 1, 
  (∀ y ≥ x, (y^2 - 2*m*y + 1/2) / (x^2 - 2*m*x + 1/2) ≥ 1) ∧ 
  (x^2 - 2*m*x + 1/2 > 0)

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (¬(p m) ∧ (p m ∨ q m)) → (m > -1 ∧ m < 3/4) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3699_369945


namespace NUMINAMATH_CALUDE_full_price_tickets_count_l3699_369916

/-- Represents the number of tickets sold at each price point -/
structure TicketSales where
  full : ℕ
  half : ℕ
  double : ℕ

/-- Represents the price of a full-price ticket -/
def FullPrice : ℕ := 30

/-- The total number of tickets sold -/
def TotalTickets : ℕ := 200

/-- The total revenue from all ticket sales -/
def TotalRevenue : ℕ := 3600

/-- Calculates the total number of tickets sold -/
def totalTicketCount (sales : TicketSales) : ℕ :=
  sales.full + sales.half + sales.double

/-- Calculates the total revenue from all ticket sales -/
def totalRevenue (sales : TicketSales) : ℕ :=
  sales.full * FullPrice + sales.half * (FullPrice / 2) + sales.double * (2 * FullPrice)

/-- Theorem stating that the number of full-price tickets sold is 80 -/
theorem full_price_tickets_count :
  ∃ (sales : TicketSales),
    totalTicketCount sales = TotalTickets ∧
    totalRevenue sales = TotalRevenue ∧
    sales.full = 80 :=
by sorry

end NUMINAMATH_CALUDE_full_price_tickets_count_l3699_369916


namespace NUMINAMATH_CALUDE_jump_rope_record_time_l3699_369964

theorem jump_rope_record_time (record : ℕ) (jumps_per_second : ℕ) : 
  record = 54000 → jumps_per_second = 3 → 
  ∃ (hours : ℕ), hours = 5 ∧ hours * (jumps_per_second * 3600) > record :=
by sorry

end NUMINAMATH_CALUDE_jump_rope_record_time_l3699_369964


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l3699_369986

-- Define the quadratic polynomial f(x) = ax² + bx + c
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for f(x) to have exactly one root
def has_one_root (a b c : ℝ) : Prop :=
  ∃! x, f a b c x = 0

-- Define g(x) = f(3x + 2) - 2f(2x - 1)
def g (a b c x : ℝ) : ℝ := f a b c (3*x + 2) - 2 * f a b c (2*x - 1)

-- Theorem statement
theorem quadratic_root_theorem (a b c : ℝ) :
  a ≠ 0 →
  has_one_root a b c →
  has_one_root 1 (20 - b) (2 + 4*b - b^2/4) →
  ∃ x, f a b c x = 0 ∧ x = -7 :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l3699_369986


namespace NUMINAMATH_CALUDE_sum_equals_target_l3699_369922

theorem sum_equals_target : 2.75 + 0.003 + 0.158 = 2.911 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_target_l3699_369922


namespace NUMINAMATH_CALUDE_zero_subset_M_l3699_369991

def M : Set ℤ := {x : ℤ | |x| < 5}

theorem zero_subset_M : {0} ⊆ M := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_M_l3699_369991


namespace NUMINAMATH_CALUDE_circle_tangent_range_l3699_369963

/-- The range of k values for which two tangent lines can be drawn from (1, 2) to the circle x^2 + y^2 + kx + 2y + k^2 - 15 = 0 -/
theorem circle_tangent_range (k : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + k*x + 2*y + k^2 - 15 = 0) ∧ 
  (∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    (∃ (x₁ y₁ x₂ y₂ : ℝ), 
      x₁^2 + y₁^2 + k*x₁ + 2*y₁ + k^2 - 15 = 0 ∧
      x₂^2 + y₂^2 + k*x₂ + 2*y₂ + k^2 - 15 = 0 ∧
      (y₁ - 2) = t₁ * (x₁ - 1) ∧
      (y₂ - 2) = t₂ * (x₂ - 1))) ↔ 
  (k > -8*Real.sqrt 3/3 ∧ k < -3) ∨ (k > 2 ∧ k < 8*Real.sqrt 3/3) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_range_l3699_369963


namespace NUMINAMATH_CALUDE_common_point_of_alternating_ap_lines_l3699_369949

/-- Represents a line in 2D space with equation ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point (x, y) lies on a given line --/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Defines an alternating arithmetic progression for a, b, c --/
def is_alternating_ap (a b c : ℝ) : Prop :=
  ∃ d : ℝ, b = a - d ∧ c = a + d

theorem common_point_of_alternating_ap_lines :
  ∀ l : Line, is_alternating_ap l.a l.b l.c → l.contains 1 (-1) :=
sorry

end NUMINAMATH_CALUDE_common_point_of_alternating_ap_lines_l3699_369949


namespace NUMINAMATH_CALUDE_left_handed_fraction_is_four_ninths_l3699_369953

/-- Represents the ratio of red to blue participants -/
def red_to_blue_ratio : ℚ := 2

/-- Fraction of left-handed red participants -/
def left_handed_red_fraction : ℚ := 1/3

/-- Fraction of left-handed blue participants -/
def left_handed_blue_fraction : ℚ := 2/3

/-- Theorem stating the fraction of left-handed participants -/
theorem left_handed_fraction_is_four_ninths 
  (h1 : red_to_blue_ratio = 2)
  (h2 : left_handed_red_fraction = 1/3)
  (h3 : left_handed_blue_fraction = 2/3) :
  (red_to_blue_ratio * left_handed_red_fraction + left_handed_blue_fraction) / 
  (red_to_blue_ratio + 1) = 4/9 := by
  sorry

#check left_handed_fraction_is_four_ninths

end NUMINAMATH_CALUDE_left_handed_fraction_is_four_ninths_l3699_369953


namespace NUMINAMATH_CALUDE_coefficients_of_given_equation_l3699_369937

/-- Represents a quadratic equation in the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The quadratic equation x^2 - x + 3 = 0 -/
def given_equation : QuadraticEquation := { a := 1, b := -1, c := 3 }

theorem coefficients_of_given_equation :
  given_equation.a = 1 ∧ given_equation.b = -1 ∧ given_equation.c = 3 := by
  sorry

end NUMINAMATH_CALUDE_coefficients_of_given_equation_l3699_369937
