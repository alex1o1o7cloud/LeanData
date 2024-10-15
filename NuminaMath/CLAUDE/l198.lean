import Mathlib

namespace NUMINAMATH_CALUDE_modular_inverse_three_mod_seventeen_l198_19833

theorem modular_inverse_three_mod_seventeen :
  ∃! x : ℕ, x ≤ 16 ∧ (3 * x) % 17 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_modular_inverse_three_mod_seventeen_l198_19833


namespace NUMINAMATH_CALUDE_planes_perpendicular_parallel_l198_19894

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_parallel 
  (a b : Line) (α β γ : Plane) 
  (h1 : perpendicular α γ) 
  (h2 : parallel β γ) : 
  perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_parallel_l198_19894


namespace NUMINAMATH_CALUDE_min_value_expression_l198_19842

theorem min_value_expression (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) :
  1 / (x + y)^2 + 1 / (x - y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l198_19842


namespace NUMINAMATH_CALUDE_fraction_evaluation_l198_19824

theorem fraction_evaluation : (3 : ℚ) / (2 - 5 / 4) = 4 := by sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l198_19824


namespace NUMINAMATH_CALUDE_linda_travel_distance_l198_19897

/-- Represents the travel data for one day -/
structure DayTravel where
  minutes_per_mile : ℕ
  distance : ℕ

/-- Calculates the distance traveled in one hour given the minutes per mile -/
def distance_traveled (minutes_per_mile : ℕ) : ℕ :=
  60 / minutes_per_mile

/-- Generates the travel data for four days -/
def generate_four_days (initial_minutes_per_mile : ℕ) : List DayTravel :=
  [0, 1, 2, 3].map (λ i =>
    { minutes_per_mile := initial_minutes_per_mile + i * 5,
      distance := distance_traveled (initial_minutes_per_mile + i * 5) })

theorem linda_travel_distance :
  ∃ (initial_minutes_per_mile : ℕ),
    let four_days := generate_four_days initial_minutes_per_mile
    four_days.length = 4 ∧
    (∀ day ∈ four_days, day.minutes_per_mile > 0 ∧ day.minutes_per_mile ≤ 60) ∧
    (∀ day ∈ four_days, day.distance > 0) ∧
    (List.sum (four_days.map (λ day => day.distance)) = 25) := by
  sorry

end NUMINAMATH_CALUDE_linda_travel_distance_l198_19897


namespace NUMINAMATH_CALUDE_average_price_approx_1_70_l198_19852

/-- The average price per bottle given the purchase of large and small bottles -/
def average_price_per_bottle (large_bottles : ℕ) (small_bottles : ℕ) 
  (large_price : ℚ) (small_price : ℚ) : ℚ :=
  ((large_bottles : ℚ) * large_price + (small_bottles : ℚ) * small_price) / 
  ((large_bottles : ℚ) + (small_bottles : ℚ))

/-- Theorem stating that the average price per bottle is approximately $1.70 -/
theorem average_price_approx_1_70 :
  let large_bottles : ℕ := 1300
  let small_bottles : ℕ := 750
  let large_price : ℚ := 189/100  -- $1.89
  let small_price : ℚ := 138/100  -- $1.38
  abs (average_price_per_bottle large_bottles small_bottles large_price small_price - 17/10) < 1/100
  := by sorry

end NUMINAMATH_CALUDE_average_price_approx_1_70_l198_19852


namespace NUMINAMATH_CALUDE_smallest_period_sin_polar_l198_19870

theorem smallest_period_sin_polar (t : ℝ) : 
  (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → 
    ∃ r : ℝ, r = Real.sin θ ∧ 
    (∃ x y : ℝ, x = r * Real.cos θ ∧ y = r * Real.sin θ)) → 
  (∀ x y : ℝ, x^2 + y^2 ≤ 1 → 
    ∃ θ : ℝ, 0 ≤ θ ∧ θ ≤ t ∧ 
    x = (Real.sin θ) * (Real.cos θ) ∧ 
    y = (Real.sin θ) * (Real.sin θ)) →
  t ≥ π :=
sorry

end NUMINAMATH_CALUDE_smallest_period_sin_polar_l198_19870


namespace NUMINAMATH_CALUDE_circle_properties_l198_19848

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9
def C₂ (x y : ℝ) : Prop := (x-1)^2 + (y+1)^2 = 16

-- Define the line y = x
def line_y_eq_x (x y : ℝ) : Prop := y = x

theorem circle_properties :
  -- 1. The equation of the line passing through the centers of C₁ and C₂ is y = -x
  (∃ m b : ℝ, ∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = -1) → y = m * x + b) ∧
  (∀ x y : ℝ, (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = -1) → y = -x) ∧
  -- 2. The circles intersect and the length of their common chord is √94/2
  (∃ x₁ y₁ x₂ y₂ : ℝ, C₁ x₁ y₁ ∧ C₁ x₂ y₂ ∧ C₂ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
    ((x₂ - x₁)^2 + (y₂ - y₁)^2)^(1/2) = (94)^(1/2)/2) ∧
  -- 3. There exist exactly 4 points on C₂ that are at a distance of 2 from the line y = x
  (∃! (a b c d : ℝ × ℝ), 
    C₂ a.1 a.2 ∧ C₂ b.1 b.2 ∧ C₂ c.1 c.2 ∧ C₂ d.1 d.2 ∧
    (∀ x y : ℝ, line_y_eq_x x y → 
      ((a.1 - x)^2 + (a.2 - y)^2)^(1/2) = 2 ∧
      ((b.1 - x)^2 + (b.2 - y)^2)^(1/2) = 2 ∧
      ((c.1 - x)^2 + (c.2 - y)^2)^(1/2) = 2 ∧
      ((d.1 - x)^2 + (d.2 - y)^2)^(1/2) = 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l198_19848


namespace NUMINAMATH_CALUDE_apple_count_l198_19844

theorem apple_count (apples oranges : ℕ) 
  (h1 : apples = oranges + 27)
  (h2 : apples + oranges = 301) : 
  apples = 164 := by
sorry

end NUMINAMATH_CALUDE_apple_count_l198_19844


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l198_19812

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 3| = |x - 5| :=
by
  use 4
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l198_19812


namespace NUMINAMATH_CALUDE_right_triangle_side_length_l198_19820

theorem right_triangle_side_length (D E F : ℝ) : 
  -- DEF is a right triangle with angle E being right
  (D^2 + E^2 = F^2) →
  -- cos(D) = (8√85)/85
  (Real.cos D = (8 * Real.sqrt 85) / 85) →
  -- EF:DF = 1:2
  (E / F = 1 / 2) →
  -- The length of DF is 2√85
  F = 2 * Real.sqrt 85 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_side_length_l198_19820


namespace NUMINAMATH_CALUDE_numerator_increase_percentage_l198_19884

theorem numerator_increase_percentage (P : ℚ) : 
  (1 + P / 100) / ((3 / 4) * 12) = 2 / 15 → P = 20 := by
  sorry

end NUMINAMATH_CALUDE_numerator_increase_percentage_l198_19884


namespace NUMINAMATH_CALUDE_third_valid_number_is_105_l198_19881

def is_valid_number (n : ℕ) : Bool :=
  n < 600

def find_third_valid_number (sequence : List ℕ) : Option ℕ :=
  let valid_numbers := sequence.filter is_valid_number
  valid_numbers.get? 2

theorem third_valid_number_is_105 (sequence : List ℕ) :
  sequence = [59, 16, 95, 55, 67, 19, 98, 10, 50, 71] →
  find_third_valid_number sequence = some 105 := by
  sorry

end NUMINAMATH_CALUDE_third_valid_number_is_105_l198_19881


namespace NUMINAMATH_CALUDE_gcf_of_270_108_150_l198_19862

theorem gcf_of_270_108_150 : Nat.gcd 270 (Nat.gcd 108 150) = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_270_108_150_l198_19862


namespace NUMINAMATH_CALUDE_choose_positions_count_l198_19810

def num_people : ℕ := 6
def num_positions : ℕ := 3

theorem choose_positions_count :
  (num_people.factorial) / ((num_people - num_positions).factorial) = 120 :=
sorry

end NUMINAMATH_CALUDE_choose_positions_count_l198_19810


namespace NUMINAMATH_CALUDE_tangent_line_equation_l198_19892

open Real

noncomputable def f (x : ℝ) : ℝ := 2 * log x

theorem tangent_line_equation :
  let p : ℝ × ℝ := (2, f 2)
  let m : ℝ := (deriv f) 2
  let tangent_eq (x y : ℝ) : Prop := x - y + 2 * log 2 - 2 = 0
  tangent_eq p.1 p.2 ∧ ∀ x y, tangent_eq x y ↔ y - p.2 = m * (x - p.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l198_19892


namespace NUMINAMATH_CALUDE_smallest_k_satisfies_condition_no_smaller_k_satisfies_condition_l198_19851

/-- The smallest positive real number k satisfying the given condition -/
def smallest_k : ℝ := 4

/-- Predicate to check if a quadratic equation has two distinct real roots -/
def has_distinct_real_roots (p q : ℝ) : Prop :=
  p^2 - 4*q > 0

/-- Predicate to check if four real numbers are distinct -/
def are_distinct (a b c d : ℝ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Main theorem stating that smallest_k satisfies the required condition -/
theorem smallest_k_satisfies_condition :
  ∀ a b c d : ℝ,
  are_distinct a b c d →
  a ≥ smallest_k → b ≥ smallest_k → c ≥ smallest_k → d ≥ smallest_k →
  ∃ p q r s : ℝ,
    ({p, q, r, s} : Set ℝ) = {a, b, c, d} ∧
    has_distinct_real_roots p q ∧
    has_distinct_real_roots r s ∧
    (∀ x : ℝ, (x^2 + p*x + q = 0 ∨ x^2 + r*x + s = 0) →
      (∀ y : ℝ, y ≠ x → (y^2 + p*y + q ≠ 0 ∧ y^2 + r*y + s ≠ 0))) :=
by sorry

/-- Theorem stating that no smaller positive real number than smallest_k satisfies the condition -/
theorem no_smaller_k_satisfies_condition :
  ∀ k : ℝ, 0 < k → k < smallest_k →
  ∃ a b c d : ℝ,
    are_distinct a b c d ∧
    a ≥ k ∧ b ≥ k ∧ c ≥ k ∧ d ≥ k ∧
    (∀ p q r s : ℝ,
      ({p, q, r, s} : Set ℝ) = {a, b, c, d} →
      ¬(has_distinct_real_roots p q ∧
        has_distinct_real_roots r s ∧
        (∀ x : ℝ, (x^2 + p*x + q = 0 ∨ x^2 + r*x + s = 0) →
          (∀ y : ℝ, y ≠ x → (y^2 + p*y + q ≠ 0 ∧ y^2 + r*y + s ≠ 0))))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_satisfies_condition_no_smaller_k_satisfies_condition_l198_19851


namespace NUMINAMATH_CALUDE_row_6_seat_16_notation_l198_19847

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (number : ℕ)

/-- The format for denoting a seat -/
def seatNotation (s : Seat) : ℕ × ℕ := (s.row, s.number)

/-- Given condition: "row 10, seat 3" is denoted as (10,3) -/
axiom example_seat : seatNotation { row := 10, number := 3 } = (10, 3)

/-- Theorem: "row 6, seat 16" is denoted as (6,16) -/
theorem row_6_seat_16_notation :
  seatNotation { row := 6, number := 16 } = (6, 16) := by
  sorry


end NUMINAMATH_CALUDE_row_6_seat_16_notation_l198_19847


namespace NUMINAMATH_CALUDE_sum_of_roots_l198_19849

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (y + 3) * (y - 4) = 22 ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l198_19849


namespace NUMINAMATH_CALUDE_thomas_weekly_wage_l198_19893

/-- Calculates the weekly wage given the monthly wage and number of weeks in a month. -/
def weekly_wage (monthly_wage : ℕ) (weeks_per_month : ℕ) : ℕ :=
  monthly_wage / weeks_per_month

/-- Proves that given a monthly wage of 19500 and 4 weeks in a month, the weekly wage is 4875. -/
theorem thomas_weekly_wage :
  weekly_wage 19500 4 = 4875 := by
  sorry

#eval weekly_wage 19500 4

end NUMINAMATH_CALUDE_thomas_weekly_wage_l198_19893


namespace NUMINAMATH_CALUDE_abc_is_cube_l198_19879

theorem abc_is_cube (a b c : ℤ) (h : (a / b) + (b / c) + (c / a) = 3) : 
  ∃ n : ℤ, a * b * c = n^3 := by
sorry

end NUMINAMATH_CALUDE_abc_is_cube_l198_19879


namespace NUMINAMATH_CALUDE_square_is_rectangle_and_rhombus_l198_19831

-- Define a quadrilateral
structure Quadrilateral :=
  (sides : Fin 4 → ℝ)
  (angles : Fin 4 → ℝ)

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, q.angles i = 90 ∧ q.sides i = q.sides ((i + 2) % 4)

-- Define a rhombus
def is_rhombus (q : Quadrilateral) : Prop :=
  ∀ i j : Fin 4, q.sides i = q.sides j

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  is_rectangle q ∧ is_rhombus q

-- Theorem statement
theorem square_is_rectangle_and_rhombus (q : Quadrilateral) :
  is_square q → is_rectangle q ∧ is_rhombus q :=
sorry

end NUMINAMATH_CALUDE_square_is_rectangle_and_rhombus_l198_19831


namespace NUMINAMATH_CALUDE_trees_per_sharpening_l198_19856

def cost_per_sharpening : ℕ := 5
def total_sharpening_cost : ℕ := 35
def min_trees_chopped : ℕ := 91

theorem trees_per_sharpening :
  ∃ (x : ℕ), x > 0 ∧ 
    x * (total_sharpening_cost / cost_per_sharpening) ≥ min_trees_chopped ∧
    ∀ (y : ℕ), y > 0 → y * (total_sharpening_cost / cost_per_sharpening) ≥ min_trees_chopped → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_trees_per_sharpening_l198_19856


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l198_19871

/-- An isosceles triangle with congruent sides of length 7 cm and perimeter 23 cm has a base of length 9 cm. -/
theorem isosceles_triangle_base_length : ∀ (base : ℝ), 
  base > 0 → -- The base length is positive
  7 > 0 → -- The congruent side length is positive
  2 * 7 + base = 23 → -- The perimeter is 23 cm
  base = 9 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l198_19871


namespace NUMINAMATH_CALUDE_rotation_matrix_correct_l198_19882

def A : Fin 2 → ℝ := ![1, 1]
def B : Fin 2 → ℝ := ![-1, 1]
def C : Fin 2 → ℝ := ![-1, -1]
def D : Fin 2 → ℝ := ![1, -1]

def N : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; -1, 0]

theorem rotation_matrix_correct :
  N.mulVec A = D ∧
  N.mulVec B = A ∧
  N.mulVec C = B ∧
  N.mulVec D = C := by
  sorry

end NUMINAMATH_CALUDE_rotation_matrix_correct_l198_19882


namespace NUMINAMATH_CALUDE_share_calculation_l198_19861

/-- Given a total amount divided among three parties with specific ratios, 
    prove that the first party's share is a certain value. -/
theorem share_calculation (total : ℚ) (a b c : ℚ) : 
  total = 700 →
  a + b + c = total →
  a = (2/3) * (b + c) →
  b = (6/9) * (a + c) →
  a = 280 := by
  sorry

end NUMINAMATH_CALUDE_share_calculation_l198_19861


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l198_19809

/-- Represents an ellipse equation with parameter m -/
def is_ellipse (m : ℝ) : Prop :=
  (m > 1) ∧ (m < 3) ∧ (m ≠ 2)

/-- The condition given in the problem -/
def given_condition (m : ℝ) : Prop :=
  (m > 1) ∧ (m < 3)

/-- Theorem stating that the given condition is necessary but not sufficient -/
theorem necessary_not_sufficient :
  (∀ m : ℝ, is_ellipse m → given_condition m) ∧
  ¬(∀ m : ℝ, given_condition m → is_ellipse m) :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l198_19809


namespace NUMINAMATH_CALUDE_min_units_for_nonnegative_profit_l198_19874

/-- Represents the profit function for ice powder sales -/
def profit : ℕ → ℤ
| 0 => -120
| 10 => -80
| 20 => -40
| 30 => 0
| 40 => 40
| 50 => 80
| _ => 0  -- Default case, not used in the proof

/-- Theorem: The minimum number of units to be sold for non-negative profit is 30 -/
theorem min_units_for_nonnegative_profit :
  (∀ x : ℕ, x < 30 → profit x < 0) ∧
  profit 30 = 0 ∧
  (∀ x : ℕ, x > 30 → profit x > 0) :=
by sorry


end NUMINAMATH_CALUDE_min_units_for_nonnegative_profit_l198_19874


namespace NUMINAMATH_CALUDE_sum_of_h_at_x_values_l198_19875

def f (x : ℝ) : ℝ := |x| - 3

def g (x : ℝ) : ℝ := -x

def h (x : ℝ) : ℝ := f (g (f x))

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_h_at_x_values :
  (x_values.map h).sum = -17 := by sorry

end NUMINAMATH_CALUDE_sum_of_h_at_x_values_l198_19875


namespace NUMINAMATH_CALUDE_alley_width_l198_19840

/-- Given a ladder of length l placed in an alley, touching one wall at a 60° angle
    and the other wall at a 30° angle with the ground, the width w of the alley
    is equal to l(√3 + 1)/2. -/
theorem alley_width (l : ℝ) (h : l > 0) :
  let w := l * (Real.sqrt 3 + 1) / 2
  let angle_A := 60 * π / 180
  let angle_B := 30 * π / 180
  ∃ (m : ℝ), m > 0 ∧ w = l * Real.sin angle_A + l * Real.sin angle_B :=
sorry

end NUMINAMATH_CALUDE_alley_width_l198_19840


namespace NUMINAMATH_CALUDE_incorrect_transformation_l198_19860

theorem incorrect_transformation (x y m : ℝ) :
  ¬(∀ (x y m : ℝ), x = y → x / m = y / m) :=
sorry

end NUMINAMATH_CALUDE_incorrect_transformation_l198_19860


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l198_19823

/-- Given a cylinder with original volume of 15 cubic feet, proves that tripling its radius and halving its height results in a new volume of 67.5 cubic feet. -/
theorem cylinder_volume_change (r h : ℝ) (h1 : r > 0) (h2 : h > 0) (h3 : π * r^2 * h = 15) :
  π * (3*r)^2 * (h/2) = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l198_19823


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l198_19876

def U : Finset Nat := {1, 2, 3, 4, 5, 6}
def M : Finset Nat := {1, 3, 5}

theorem complement_of_M_in_U :
  (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l198_19876


namespace NUMINAMATH_CALUDE_number_of_hens_l198_19899

/-- Given a farm with hens and cows, prove that the number of hens is 24 -/
theorem number_of_hens (hens cows : ℕ) : 
  hens + cows = 44 →  -- Total number of heads
  2 * hens + 4 * cows = 128 →  -- Total number of feet
  hens = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_hens_l198_19899


namespace NUMINAMATH_CALUDE_possible_perimeters_only_possible_perimeters_l198_19837

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Represents the possible ways to cut the original rectangle -/
inductive Cut
  | Vertical
  | Horizontal
  | Mixed

/-- The original rectangle -/
def originalRect : Rectangle := { length := 6, width := 3 }

/-- Theorem stating the possible perimeters of the resulting rectangles -/
theorem possible_perimeters :
  ∃ (c : Cut) (r : Rectangle),
    (c = Cut.Vertical ∧ perimeter r = 14) ∨
    (c = Cut.Horizontal ∧ perimeter r = 10) ∨
    (c = Cut.Mixed ∧ perimeter r = 10.5) :=
  sorry

/-- Theorem stating that these are the only possible perimeters -/
theorem only_possible_perimeters :
  ∀ (c : Cut) (r : Rectangle),
    (perimeter r ≠ 14 ∧ perimeter r ≠ 10 ∧ perimeter r ≠ 10.5) →
    ¬(∃ (r1 r2 : Rectangle), 
      perimeter r = perimeter r1 ∧
      perimeter r = perimeter r2 ∧
      r.length + r1.length + r2.length = originalRect.length ∧
      r.width = r1.width ∧ r.width = r2.width ∧ r.width = originalRect.width) :=
  sorry

end NUMINAMATH_CALUDE_possible_perimeters_only_possible_perimeters_l198_19837


namespace NUMINAMATH_CALUDE_cos_45_cos_15_plus_sin_45_sin_15_l198_19838

theorem cos_45_cos_15_plus_sin_45_sin_15 :
  Real.cos (45 * π / 180) * Real.cos (15 * π / 180) +
  Real.sin (45 * π / 180) * Real.sin (15 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_45_cos_15_plus_sin_45_sin_15_l198_19838


namespace NUMINAMATH_CALUDE_integral_x_plus_inverse_x_l198_19843

open Real MeasureTheory

theorem integral_x_plus_inverse_x : ∫ x in (1 : ℝ)..2, (x + 1/x) = 3/2 + Real.log 2 := by sorry

end NUMINAMATH_CALUDE_integral_x_plus_inverse_x_l198_19843


namespace NUMINAMATH_CALUDE_car_lot_total_l198_19887

theorem car_lot_total (air_bags : ℕ) (power_windows : ℕ) (both : ℕ) (neither : ℕ)
  (h1 : air_bags = 45)
  (h2 : power_windows = 30)
  (h3 : both = 12)
  (h4 : neither = 2) :
  air_bags + power_windows - both + neither = 65 := by
  sorry

end NUMINAMATH_CALUDE_car_lot_total_l198_19887


namespace NUMINAMATH_CALUDE_range_of_a_l198_19864

open Set

def A : Set ℝ := {x | -5 < x ∧ x < 1}
def B : Set ℝ := {x | -2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x | x < a}

theorem range_of_a (a : ℝ) (h : A ∩ B ⊆ C a) : a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l198_19864


namespace NUMINAMATH_CALUDE_compare_logarithmic_expressions_l198_19811

open Real

theorem compare_logarithmic_expressions :
  let e := exp 1
  1/e > log (3^(1/3)) ∧ 
  log (3^(1/3)) > log π / π ∧ 
  log π / π > sqrt 15 * log 15 / 30 :=
by
  sorry

end NUMINAMATH_CALUDE_compare_logarithmic_expressions_l198_19811


namespace NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l198_19815

theorem sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds :
  7 < Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) ∧
  Real.sqrt 2 * (2 * Real.sqrt 2 + Real.sqrt 5) < 8 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_2_times_2sqrt_2_plus_sqrt_5_bounds_l198_19815


namespace NUMINAMATH_CALUDE_unclaimed_candy_fraction_l198_19846

/-- Represents the order of arrival -/
inductive Participant
| Charlie
| Alice
| Bob

/-- The fraction of candy each participant should receive based on the 4:3:2 ratio -/
def intended_share (p : Participant) : ℚ :=
  match p with
  | Participant.Charlie => 2/9
  | Participant.Alice => 4/9
  | Participant.Bob => 1/3

/-- The actual amount of candy taken by each participant -/
def actual_take (p : Participant) : ℚ :=
  match p with
  | Participant.Charlie => 2/9
  | Participant.Alice => 28/81
  | Participant.Bob => 17/81

theorem unclaimed_candy_fraction :
  1 - (actual_take Participant.Charlie + actual_take Participant.Alice + actual_take Participant.Bob) = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_unclaimed_candy_fraction_l198_19846


namespace NUMINAMATH_CALUDE_weight_problem_l198_19801

theorem weight_problem (a b c d e f g h : ℝ) 
  (h1 : (a + b + c + f) / 4 = 80)
  (h2 : (a + b + c + d + e + f) / 6 = 82)
  (h3 : g = d + 5)
  (h4 : h = e - 4)
  (h5 : (c + d + e + f + g + h) / 6 = 83) :
  a + b = 167 := by
sorry

end NUMINAMATH_CALUDE_weight_problem_l198_19801


namespace NUMINAMATH_CALUDE_station_entry_problem_l198_19854

/-- The number of ways for n people to enter through k gates, where each gate must have at least one person -/
def enterWays (n k : ℕ) : ℕ :=
  sorry

/-- The condition that the number of people is greater than the number of gates -/
def validInput (n k : ℕ) : Prop :=
  n > k ∧ k > 0

theorem station_entry_problem :
  ∀ n k : ℕ, validInput n k → (n = 5 ∧ k = 3) → enterWays n k = 720 :=
sorry

end NUMINAMATH_CALUDE_station_entry_problem_l198_19854


namespace NUMINAMATH_CALUDE_rachel_math_homework_l198_19895

/-- The number of pages of reading homework Rachel had to complete -/
def reading_homework : ℕ := 3

/-- The additional pages of math homework compared to reading homework -/
def additional_math_pages : ℕ := 4

/-- The total number of pages of math homework Rachel had to complete -/
def math_homework : ℕ := reading_homework + additional_math_pages

theorem rachel_math_homework :
  math_homework = 7 :=
by sorry

end NUMINAMATH_CALUDE_rachel_math_homework_l198_19895


namespace NUMINAMATH_CALUDE_min_value_fraction_l198_19868

theorem min_value_fraction (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 2) : 
  (x + y) / (x * y * z) ≥ 4 ∧ ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a + b + c = 2 ∧ (a + b) / (a * b * c) = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_fraction_l198_19868


namespace NUMINAMATH_CALUDE_border_length_is_even_l198_19878

/-- Represents a domino on the board -/
inductive Domino
| Horizontal
| Vertical

/-- Represents the board -/
def Board := Fin 2010 → Fin 2011 → Domino

/-- The border length between horizontal and vertical dominoes -/
def borderLength (board : Board) : ℕ := sorry

/-- Theorem stating that the border length is even -/
theorem border_length_is_even (board : Board) : 
  Even (borderLength board) := by sorry

end NUMINAMATH_CALUDE_border_length_is_even_l198_19878


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l198_19896

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  is_in_second_quadrant (-1 : ℝ) (3 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l198_19896


namespace NUMINAMATH_CALUDE_three_intersection_points_k_value_l198_19891

/-- Curve C1 -/
def C1 (k : ℝ) (x y : ℝ) : Prop :=
  y = k * abs x + 2

/-- Curve C2 -/
def C2 (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4

/-- Number of intersection points between C1 and C2 -/
def intersectionPoints (k : ℝ) : ℕ :=
  sorry -- This would require a complex implementation to count intersection points

theorem three_intersection_points_k_value :
  ∀ k : ℝ, intersectionPoints k = 3 → k = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_three_intersection_points_k_value_l198_19891


namespace NUMINAMATH_CALUDE_other_bill_value_l198_19888

/-- Represents the class fund with two types of bills -/
structure ClassFund where
  total_amount : ℕ
  num_other_bills : ℕ
  value_ten_dollar_bill : ℕ

/-- Theorem stating the value of the other type of bills -/
theorem other_bill_value (fund : ClassFund)
  (h1 : fund.total_amount = 120)
  (h2 : fund.num_other_bills = 3)
  (h3 : fund.value_ten_dollar_bill = 10)
  (h4 : 2 * fund.num_other_bills = (fund.total_amount - fund.num_other_bills * (fund.total_amount / fund.num_other_bills)) / fund.value_ten_dollar_bill) :
  fund.total_amount / fund.num_other_bills = 40 := by
sorry

end NUMINAMATH_CALUDE_other_bill_value_l198_19888


namespace NUMINAMATH_CALUDE_range_of_m_l198_19867

/-- The statement p: The equation x^2 + mx + 1 = 0 has two distinct negative real roots -/
def p (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

/-- The statement q: The equation 4x^2 + 4(m-2)x + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

theorem range_of_m :
  (∃ m : ℝ, ¬(p m) ∧ q m) →
  (∃ m : ℝ, 1 < m ∧ m ≤ 2) ∧ (∀ m : ℝ, (1 < m ∧ m ≤ 2) → (¬(p m) ∧ q m)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l198_19867


namespace NUMINAMATH_CALUDE_union_of_complements_is_certain_l198_19890

-- Define the sample space
variable {Ω : Type}

-- Define events as sets of outcomes
variable (A B C D : Set Ω)

-- Define the properties of events
variable (h1 : A ∩ B = ∅)  -- A and B are mutually exclusive
variable (h2 : C = Aᶜ)     -- C is the complement of A
variable (h3 : D = Bᶜ)     -- D is the complement of B

-- Theorem statement
theorem union_of_complements_is_certain : C ∪ D = univ := by
  sorry

end NUMINAMATH_CALUDE_union_of_complements_is_certain_l198_19890


namespace NUMINAMATH_CALUDE_plane_equation_correct_l198_19832

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space represented by the equation ax + by + cz + d = 0 -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Check if a point lies on a plane -/
def Point3D.liesOn (p : Point3D) (plane : Plane) : Prop :=
  plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

/-- The origin point (0,0,0) -/
def origin : Point3D := ⟨0, 0, 0⟩

/-- The point where the perpendicular meets the plane -/
def perpendicularPoint : Point3D := ⟨10, -2, 5⟩

/-- The plane in question -/
def targetPlane : Plane := ⟨10, -2, 5, -129⟩

/-- Vector from origin to perpendicularPoint -/
def normalVector : Point3D := perpendicularPoint

theorem plane_equation_correct :
  (∀ (p : Point3D), p.liesOn targetPlane ↔ 
    (p.x - perpendicularPoint.x) * normalVector.x + 
    (p.y - perpendicularPoint.y) * normalVector.y + 
    (p.z - perpendicularPoint.z) * normalVector.z = 0) ∧
  perpendicularPoint.liesOn targetPlane :=
sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l198_19832


namespace NUMINAMATH_CALUDE_triangle_theorem_l198_19829

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c

/-- The given condition relating sides and angles -/
def triangle_condition (t : Triangle) : Prop :=
  (2 * t.a + t.b) / t.c = (Real.cos (t.A + t.C)) / (Real.cos t.C)

theorem triangle_theorem (t : Triangle) (h : triangle_condition t) :
  t.C = 2 * π / 3 ∧ 1 < (t.a + t.b) / t.c ∧ (t.a + t.b) / t.c ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l198_19829


namespace NUMINAMATH_CALUDE_area_ratio_of_squares_l198_19818

/-- Given four square regions with perimeters p₁, p₂, p₃, and p₄, 
    this theorem proves that the ratio of the area of the second square 
    to the area of the fourth square is 9/16 when p₁ = 16, p₂ = 36, p₃ = p₄ = 48. -/
theorem area_ratio_of_squares (p₁ p₂ p₃ p₄ : ℝ) 
    (h₁ : p₁ = 16) (h₂ : p₂ = 36) (h₃ : p₃ = 48) (h₄ : p₄ = 48) :
    (p₂ / 4)^2 / (p₄ / 4)^2 = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_area_ratio_of_squares_l198_19818


namespace NUMINAMATH_CALUDE_tan_660_degrees_l198_19889

theorem tan_660_degrees : Real.tan (660 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_660_degrees_l198_19889


namespace NUMINAMATH_CALUDE_colored_ngon_at_most_two_colors_l198_19866

/-- A regular n-gon with colored sides and diagonals -/
structure ColoredNGon where
  n : ℕ
  vertices : Fin n → Point
  colors : ℕ
  coloring : (Fin n × Fin n) → Fin colors

/-- The coloring satisfies the first condition -/
def satisfies_condition1 (R : ColoredNGon) : Prop :=
  ∀ c : Fin R.colors, ∀ A B : Fin R.n,
    (R.coloring (A, B) = c) ∨
    (∃ C : Fin R.n, R.coloring (A, C) = c ∧ R.coloring (B, C) = c)

/-- The coloring satisfies the second condition -/
def satisfies_condition2 (R : ColoredNGon) : Prop :=
  ∀ A B C : Fin R.n,
    (R.coloring (A, B) ≠ R.coloring (B, C)) →
    (R.coloring (A, C) = R.coloring (A, B) ∨ R.coloring (A, C) = R.coloring (B, C))

/-- Main theorem: If a ColoredNGon satisfies both conditions, then it has at most 2 colors -/
theorem colored_ngon_at_most_two_colors (R : ColoredNGon)
  (h1 : satisfies_condition1 R) (h2 : satisfies_condition2 R) :
  R.colors ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_colored_ngon_at_most_two_colors_l198_19866


namespace NUMINAMATH_CALUDE_problem_statement_l198_19822

theorem problem_statement :
  (∀ a : ℝ, a < (3/2) → 2*a + 4/(2*a - 3) + 3 ≤ 2) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → x + 3*y = 2*x*y → x + 3*y ≥ 6) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l198_19822


namespace NUMINAMATH_CALUDE_albert_has_two_snakes_l198_19813

/-- Represents the number of snakes Albert has -/
def num_snakes : ℕ := 2

/-- Length of the garden snake in inches -/
def garden_snake_length : ℝ := 10.0

/-- Ratio of garden snake length to boa constrictor length -/
def snake_length_ratio : ℝ := 7.0

/-- Length of the boa constrictor in inches -/
def boa_constrictor_length : ℝ := 1.428571429

/-- Theorem stating that Albert has exactly 2 snakes given the conditions -/
theorem albert_has_two_snakes :
  num_snakes = 2 ∧
  garden_snake_length = 10.0 ∧
  boa_constrictor_length = garden_snake_length / snake_length_ratio ∧
  boa_constrictor_length = 1.428571429 :=
by sorry

end NUMINAMATH_CALUDE_albert_has_two_snakes_l198_19813


namespace NUMINAMATH_CALUDE_polynomial_equality_implies_b_value_l198_19827

theorem polynomial_equality_implies_b_value 
  (a b c : ℝ) 
  (h : ∀ x : ℝ, (4*x^2 - 2*x + 5/2)*(a*x^2 + b*x + c) = 
                 12*x^4 - 8*x^3 + 15*x^2 - 5*x + 5/2) : 
  b = -1/2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_implies_b_value_l198_19827


namespace NUMINAMATH_CALUDE_power_division_equality_l198_19826

theorem power_division_equality (a : ℝ) (h : a ≠ 0) : a^10 / a^9 = a := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l198_19826


namespace NUMINAMATH_CALUDE_product_of_fractions_l198_19808

theorem product_of_fractions : 
  (7 / 5 : ℚ) * (8 / 16 : ℚ) * (21 / 15 : ℚ) * (14 / 28 : ℚ) * 
  (35 / 25 : ℚ) * (20 / 40 : ℚ) * (49 / 35 : ℚ) * (32 / 64 : ℚ) = 2401 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_l198_19808


namespace NUMINAMATH_CALUDE_vasyas_numbers_l198_19873

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  x = (1 : ℝ) / 2 ∧ y = -(1 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_vasyas_numbers_l198_19873


namespace NUMINAMATH_CALUDE_triangle_square_side_ratio_l198_19880

theorem triangle_square_side_ratio : 
  ∀ (triangle_side square_side : ℚ),
    triangle_side * 3 = 60 →
    square_side * 4 = 60 →
    triangle_side / square_side = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_square_side_ratio_l198_19880


namespace NUMINAMATH_CALUDE_triangle_minimum_value_l198_19865

theorem triangle_minimum_value (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (0 < B) → (B < Real.pi / 2) →
  (Real.cos B)^2 + (1/2) * Real.sin (2 * B) = 1 →
  -- |BC + AB| = 3
  b = 3 →
  -- Minimum value of 16b/(ac)
  (∀ x y z : Real, x > 0 → y > 0 → z > 0 →
    (Real.cos x)^2 + (1/2) * Real.sin (2 * x) = 1 →
    y = 3 →
    16 * y / (z * x) ≥ 16 * (2 - Real.sqrt 2) / 3) ∧
  (∃ x y z : Real, x > 0 ∧ y > 0 ∧ z > 0 ∧
    (Real.cos x)^2 + (1/2) * Real.sin (2 * x) = 1 ∧
    y = 3 ∧
    16 * y / (z * x) = 16 * (2 - Real.sqrt 2) / 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_minimum_value_l198_19865


namespace NUMINAMATH_CALUDE_not_divisible_power_ten_plus_one_l198_19819

theorem not_divisible_power_ten_plus_one (m n : ℕ) :
  ¬ ∃ (k : ℕ), (10^m + 1) = k * (10^n - 1) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_power_ten_plus_one_l198_19819


namespace NUMINAMATH_CALUDE_product_plus_one_is_square_l198_19825

theorem product_plus_one_is_square (n : ℕ) : 
  ∃ m : ℕ, n * (n + 1) * (n + 2) * (n + 3) + 1 = m ^ 2 := by
  sorry

#check product_plus_one_is_square 7321

end NUMINAMATH_CALUDE_product_plus_one_is_square_l198_19825


namespace NUMINAMATH_CALUDE_range_of_m_l198_19836

-- Define the sets A and B
def A : Set ℝ := {x | (x + 1) * (x - 1) < 0}
def B (m : ℝ) : Set ℝ := {x | m < x ∧ x < 1}

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, (∅ ≠ B m) ∧ 
  (∀ x : ℝ, x ∈ B m → x ∈ A) ∧ 
  (∃ y : ℝ, y ∈ A ∧ y ∉ B m) →
  -1 < m ∧ m < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l198_19836


namespace NUMINAMATH_CALUDE_sum_of_fractions_l198_19830

theorem sum_of_fractions : 
  (3 / 15 : ℚ) + (6 / 15 : ℚ) + (9 / 15 : ℚ) + (12 / 15 : ℚ) + (1 : ℚ) + 
  (18 / 15 : ℚ) + (21 / 15 : ℚ) + (24 / 15 : ℚ) + (27 / 15 : ℚ) + (5 : ℚ) = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l198_19830


namespace NUMINAMATH_CALUDE_sin_210_degrees_l198_19814

theorem sin_210_degrees : Real.sin (210 * π / 180) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l198_19814


namespace NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l198_19834

-- Define the equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + a = 0

-- Define what it means for the equation to represent a circle
def represents_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem a_equals_one_sufficient_not_necessary :
  (represents_circle 1) ∧
  (∃ (a : ℝ), a ≠ 1 ∧ represents_circle a) :=
sorry

end NUMINAMATH_CALUDE_a_equals_one_sufficient_not_necessary_l198_19834


namespace NUMINAMATH_CALUDE_my_circle_center_l198_19802

/-- A circle in the 2D plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- The center of a circle -/
def center (c : Circle) : ℝ × ℝ := sorry

/-- Our specific circle -/
def my_circle : Circle :=
  { equation := fun x y => (x + 2)^2 + y^2 = 5 }

/-- Theorem: The center of our specific circle is (-2, 0) -/
theorem my_circle_center :
  center my_circle = (-2, 0) := by sorry

end NUMINAMATH_CALUDE_my_circle_center_l198_19802


namespace NUMINAMATH_CALUDE_investment_sum_l198_19872

/-- Given a sum invested at different interest rates, prove the sum's value --/
theorem investment_sum (P : ℝ) : 
  (P * 0.15 * 2 - P * 0.12 * 2 = 720) → P = 12000 := by
  sorry

end NUMINAMATH_CALUDE_investment_sum_l198_19872


namespace NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_41_l198_19800

theorem right_triangle_with_hypotenuse_41 :
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →
  c = 41 →
  a < b →
  a = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_41_l198_19800


namespace NUMINAMATH_CALUDE_confetti_area_sum_l198_19817

/-- The sum of the areas of two square-shaped pieces of confetti, one with side length 11 cm and the other with side length 5 cm, is equal to 146 cm². -/
theorem confetti_area_sum : 
  let red_side : ℝ := 11
  let blue_side : ℝ := 5
  red_side ^ 2 + blue_side ^ 2 = 146 :=
by sorry

end NUMINAMATH_CALUDE_confetti_area_sum_l198_19817


namespace NUMINAMATH_CALUDE_karen_total_distance_l198_19853

/-- The number of shelves in the library. -/
def num_shelves : ℕ := 4

/-- The number of books on each shelf. -/
def books_per_shelf : ℕ := 400

/-- The total number of books in the library. -/
def total_books : ℕ := num_shelves * books_per_shelf

/-- The distance in miles from the library to Karen's home. -/
def distance_to_home : ℕ := total_books

/-- The total distance Karen bikes from home to library and back. -/
def total_distance : ℕ := 2 * distance_to_home

/-- Theorem stating that the total distance Karen bikes is 3200 miles. -/
theorem karen_total_distance : total_distance = 3200 := by
  sorry

end NUMINAMATH_CALUDE_karen_total_distance_l198_19853


namespace NUMINAMATH_CALUDE_simultaneous_equations_imply_quadratic_l198_19839

theorem simultaneous_equations_imply_quadratic (x y : ℝ) :
  (2 * x^2 + 6 * x + 5 * y + 1 = 0) →
  (2 * x + y + 3 = 0) →
  (y^2 + 10 * y - 7 = 0) :=
by
  sorry

end NUMINAMATH_CALUDE_simultaneous_equations_imply_quadratic_l198_19839


namespace NUMINAMATH_CALUDE_tangent_lines_count_l198_19805

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

-- Define the arithmetic sequence condition
def arithmetic_sequence (a : ℝ) : Prop :=
  ∃ d : ℝ, d ≠ 0 ∧ f a (-a) + f a (3*a) = 2 * f a a

-- Define a tangent line from the origin
def tangent_from_origin (a : ℝ) (x₀ : ℝ) : Prop :=
  ∃ y₀ : ℝ, f a x₀ = y₀ ∧ y₀ = (3 * a * x₀^2 - 6 * x₀) * x₀

-- Main theorem
theorem tangent_lines_count (a : ℝ) (ha : a ≠ 0) :
  arithmetic_sequence a →
  ∃! (count : ℕ), count = 2 ∧ 
    ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
      tangent_from_origin a x₁ ∧ 
      tangent_from_origin a x₂ ∧
      ∀ (x : ℝ), tangent_from_origin a x → (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_count_l198_19805


namespace NUMINAMATH_CALUDE_sum_difference_odd_even_l198_19885

theorem sum_difference_odd_even : 
  let range := Finset.Icc 372 506
  let odd_sum := (range.filter (λ n => n % 2 = 1)).sum id
  let even_sum := (range.filter (λ n => n % 2 = 0)).sum id
  odd_sum - even_sum = 439 := by sorry

end NUMINAMATH_CALUDE_sum_difference_odd_even_l198_19885


namespace NUMINAMATH_CALUDE_xy_value_l198_19807

theorem xy_value (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : x + 2 * y = 5) : 2 * x * y / 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l198_19807


namespace NUMINAMATH_CALUDE_rain_duration_l198_19883

/-- Given a 9-hour period where it did not rain for 5 hours, prove that it rained for 4 hours. -/
theorem rain_duration (total_hours : ℕ) (no_rain_hours : ℕ) (h1 : total_hours = 9) (h2 : no_rain_hours = 5) :
  total_hours - no_rain_hours = 4 := by
  sorry

end NUMINAMATH_CALUDE_rain_duration_l198_19883


namespace NUMINAMATH_CALUDE_min_intersection_size_l198_19821

theorem min_intersection_size (total students_with_brown_eyes students_with_lunch_box : ℕ) 
  (h1 : total = 25)
  (h2 : students_with_brown_eyes = 15)
  (h3 : students_with_lunch_box = 18) :
  ∃ (intersection : ℕ), 
    intersection ≤ students_with_brown_eyes ∧ 
    intersection ≤ students_with_lunch_box ∧
    intersection ≥ students_with_brown_eyes + students_with_lunch_box - total ∧
    intersection = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_intersection_size_l198_19821


namespace NUMINAMATH_CALUDE_kaleb_tickets_proof_l198_19869

/-- The number of tickets Kaleb initially bought at the fair -/
def initial_tickets : ℕ := 6

/-- The cost of each ticket in dollars -/
def ticket_cost : ℕ := 9

/-- The amount Kaleb spent on the ferris wheel in dollars -/
def ferris_wheel_cost : ℕ := 27

/-- The number of tickets Kaleb had left after riding the ferris wheel -/
def remaining_tickets : ℕ := 3

/-- Theorem stating that the initial number of tickets is correct given the conditions -/
theorem kaleb_tickets_proof :
  initial_tickets = (ferris_wheel_cost / ticket_cost) + remaining_tickets :=
by sorry

end NUMINAMATH_CALUDE_kaleb_tickets_proof_l198_19869


namespace NUMINAMATH_CALUDE_meal_combinations_eq_sixty_l198_19877

/-- The number of menu items in the restaurant -/
def total_menu_items : ℕ := 12

/-- The number of vegetarian dishes available -/
def vegetarian_dishes : ℕ := 5

/-- The number of different meal combinations for Elena and Nasir -/
def meal_combinations : ℕ := total_menu_items * vegetarian_dishes

/-- Theorem stating that the number of meal combinations is 60 -/
theorem meal_combinations_eq_sixty :
  meal_combinations = 60 := by sorry

end NUMINAMATH_CALUDE_meal_combinations_eq_sixty_l198_19877


namespace NUMINAMATH_CALUDE_largest_number_l198_19886

/-- Represents a number with a repeating decimal expansion -/
structure RepeatingDecimal where
  integerPart : ℕ
  nonRepeatingPart : List ℕ
  repeatingPart : List ℕ

/-- Convert a RepeatingDecimal to a real number -/
noncomputable def toReal (x : RepeatingDecimal) : ℝ := sorry

/-- The numbers given in the problem -/
def a : ℝ := 9.12344
def b : RepeatingDecimal := ⟨9, [1, 2, 3], [4]⟩
def c : RepeatingDecimal := ⟨9, [1, 2], [3, 4]⟩
def d : RepeatingDecimal := ⟨9, [1], [2, 3, 4]⟩
def e : RepeatingDecimal := ⟨9, [], [1, 2, 3, 4]⟩

/-- Theorem stating that 9.123̄4 is the largest among the given numbers -/
theorem largest_number : 
  toReal b > a ∧ 
  toReal b > toReal c ∧ 
  toReal b > toReal d ∧ 
  toReal b > toReal e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l198_19886


namespace NUMINAMATH_CALUDE_second_number_proof_l198_19835

theorem second_number_proof (h1 : 268 * x = 19832) (h2 : 2.68 * 0.74 = 1.9832) : x = 74 := by
  sorry

end NUMINAMATH_CALUDE_second_number_proof_l198_19835


namespace NUMINAMATH_CALUDE_ten_thousand_one_hundred_one_l198_19806

theorem ten_thousand_one_hundred_one (n : ℕ) : n = 10101 → n = 10000 + 100 + 1 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousand_one_hundred_one_l198_19806


namespace NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l198_19804

/-- Proves that the ratio of time taken to row upstream to downstream is 2:1 -/
theorem upstream_downstream_time_ratio 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (h1 : boat_speed = 36) 
  (h2 : stream_speed = 12) : 
  (boat_speed - stream_speed) / (boat_speed + stream_speed) = 1 / 2 := by
  sorry

#check upstream_downstream_time_ratio

end NUMINAMATH_CALUDE_upstream_downstream_time_ratio_l198_19804


namespace NUMINAMATH_CALUDE_total_reimbursement_is_correct_l198_19898

/-- Reimbursement rate for industrial clients on weekdays -/
def industrial_rate : ℚ := 36 / 100

/-- Reimbursement rate for commercial clients on weekdays -/
def commercial_rate : ℚ := 42 / 100

/-- Reimbursement rate for any clients on weekends -/
def weekend_rate : ℚ := 45 / 100

/-- Mileage for industrial clients on Monday -/
def monday_industrial : ℕ := 10

/-- Mileage for commercial clients on Monday -/
def monday_commercial : ℕ := 8

/-- Mileage for industrial clients on Tuesday -/
def tuesday_industrial : ℕ := 12

/-- Mileage for commercial clients on Tuesday -/
def tuesday_commercial : ℕ := 14

/-- Mileage for industrial clients on Wednesday -/
def wednesday_industrial : ℕ := 15

/-- Mileage for commercial clients on Wednesday -/
def wednesday_commercial : ℕ := 5

/-- Mileage for commercial clients on Thursday -/
def thursday_commercial : ℕ := 20

/-- Mileage for industrial clients on Friday -/
def friday_industrial : ℕ := 8

/-- Mileage for commercial clients on Friday -/
def friday_commercial : ℕ := 8

/-- Mileage for commercial clients on Saturday -/
def saturday_commercial : ℕ := 12

/-- Calculate the total reimbursement for the week -/
def total_reimbursement : ℚ :=
  industrial_rate * (monday_industrial + tuesday_industrial + wednesday_industrial + friday_industrial) +
  commercial_rate * (monday_commercial + tuesday_commercial + wednesday_commercial + thursday_commercial + friday_commercial) +
  weekend_rate * saturday_commercial

/-- Theorem stating that the total reimbursement is equal to $44.70 -/
theorem total_reimbursement_is_correct : total_reimbursement = 4470 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_reimbursement_is_correct_l198_19898


namespace NUMINAMATH_CALUDE_sum_of_xyz_l198_19850

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : x*z/(x+y) + y*x/(y+z) + z*y/(z+x) = -5)
  (h2 : y*z/(x+y) + z*x/(y+z) + x*y/(z+x) = 7) :
  x + y + z = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l198_19850


namespace NUMINAMATH_CALUDE_ratio_approximation_l198_19859

def geometric_sum (n : ℕ) : ℚ :=
  (10^n - 1) / 9

def ratio (n : ℕ) : ℚ :=
  (10^n * 9) / (10^n - 1)

theorem ratio_approximation :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ |ratio 8 - 9| < ε :=
sorry

end NUMINAMATH_CALUDE_ratio_approximation_l198_19859


namespace NUMINAMATH_CALUDE_library_problem_l198_19858

theorem library_problem (total_books : ℕ) (books_per_student : ℕ) 
  (day1_students : ℕ) (day2_students : ℕ) (day4_students : ℕ) :
  total_books = 120 →
  books_per_student = 5 →
  day1_students = 4 →
  day2_students = 5 →
  day4_students = 9 →
  ∃ (day3_students : ℕ),
    day3_students = 6 ∧
    total_books = (day1_students + day2_students + day3_students + day4_students) * books_per_student :=
by sorry

end NUMINAMATH_CALUDE_library_problem_l198_19858


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l198_19803

/-- Given an equilateral cone with an inscribed sphere of volume 100 cm³,
    the lateral surface area of the cone is 6π * ∛(5625/π²) cm² -/
theorem cone_lateral_surface_area (v : ℝ) (r : ℝ) (l : ℝ) (P : ℝ) :
  v = 100 →  -- volume of the sphere
  v = (4/3) * π * r^3 →  -- volume formula of a sphere
  l = 2 * Real.sqrt 3 * (75/π)^(1/3) →  -- side length of the cone
  P = 6 * π * ((5625:ℝ)/π^2)^(1/3) →  -- lateral surface area of the cone
  P = 6 * π * ((75:ℝ)^2/π^2)^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l198_19803


namespace NUMINAMATH_CALUDE_speed_ratio_with_head_start_l198_19855

/-- The ratio of speeds between two runners in a race with a head start --/
theorem speed_ratio_with_head_start (va vb : ℝ) (h : va > 0 ∧ vb > 0) :
  (∃ k : ℝ, va = k * vb) →
  (va * (1 - 0.15625) = vb) →
  va / vb = 32 / 27 := by
sorry

end NUMINAMATH_CALUDE_speed_ratio_with_head_start_l198_19855


namespace NUMINAMATH_CALUDE_eliminate_x_from_system_l198_19857

theorem eliminate_x_from_system (x y : ℝ) :
  (5 * x - 3 * y = -5) ∧ (5 * x + 4 * y = -1) → 7 * y = 4 := by
sorry

end NUMINAMATH_CALUDE_eliminate_x_from_system_l198_19857


namespace NUMINAMATH_CALUDE_circle_equation_l198_19845

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition for a circle to be tangent to the y-axis
def tangentToYAxis (c : Circle) : Prop :=
  c.center.1 = c.radius

-- Define the condition for the center to be on the line 3x - y = 0
def centerOnLine (c : Circle) : Prop :=
  c.center.2 = 3 * c.center.1

-- Define the condition for the circle to pass through point (2,3)
def passesThrough (c : Circle) : Prop :=
  (c.center.1 - 2)^2 + (c.center.2 - 3)^2 = c.radius^2

-- Define the equation of the circle
def circleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- State the theorem
theorem circle_equation :
  ∀ c : Circle,
  tangentToYAxis c → centerOnLine c → passesThrough c →
  (∀ x y : ℝ, circleEquation c x y ↔ 
    ((x - 1)^2 + (y - 3)^2 = 1) ∨ 
    ((x - 13/9)^2 + (y - 13/3)^2 = 169/81)) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l198_19845


namespace NUMINAMATH_CALUDE_passenger_catches_train_l198_19863

/-- Represents the problem of a passenger trying to catch a train --/
theorem passenger_catches_train 
  (train_delay : ℝ) 
  (train_speed : ℝ) 
  (distance : ℝ) 
  (train_stop_time : ℝ) 
  (passenger_delay : ℝ) 
  (passenger_speed : ℝ) 
  (h1 : train_delay = 11) 
  (h2 : train_speed = 10) 
  (h3 : distance = 1.5) 
  (h4 : train_stop_time = 14.5) 
  (h5 : passenger_delay = 12) 
  (h6 : passenger_speed = 4) :
  passenger_delay + distance / passenger_speed * 60 ≤ 
  train_delay + distance / train_speed * 60 + train_stop_time := by
  sorry

#check passenger_catches_train

end NUMINAMATH_CALUDE_passenger_catches_train_l198_19863


namespace NUMINAMATH_CALUDE_population_change_l198_19841

/-- Represents the population changes over 5 years -/
structure PopulationChange where
  year1 : Real
  year2 : Real
  year3 : Real
  year4 : Real
  year5 : Real

/-- Calculates the final population given an initial population and population changes -/
def finalPopulation (initialPop : Real) (changes : PopulationChange) : Real :=
  initialPop * (1 + changes.year1) * (1 + changes.year2) * (1 + changes.year3) * (1 + changes.year4) * (1 + changes.year5)

/-- The theorem to be proved -/
theorem population_change (changes : PopulationChange) 
  (h1 : changes.year1 = 0.10)
  (h2 : changes.year2 = -0.08)
  (h3 : changes.year3 = 0.15)
  (h4 : changes.year4 = -0.06)
  (h5 : changes.year5 = 0.12)
  (h6 : finalPopulation 13440 changes = 16875) : 
  ∃ (initialPop : Real), abs (initialPop - 13440) < 1 ∧ finalPopulation initialPop changes = 16875 := by
  sorry

end NUMINAMATH_CALUDE_population_change_l198_19841


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l198_19816

/-- Given a geometric sequence where the 5th term is 2 and the 8th term is 16,
    prove that the 11th term is 128. -/
theorem geometric_sequence_11th_term
  (a : ℕ → ℝ)  -- The sequence
  (h_geom : ∀ n m, a (n + 1) / a n = a (m + 1) / a m)  -- Geometric sequence condition
  (h_5th : a 5 = 2)  -- 5th term is 2
  (h_8th : a 8 = 16)  -- 8th term is 16
  : a 11 = 128 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l198_19816


namespace NUMINAMATH_CALUDE_right_triangle_area_l198_19828

theorem right_triangle_area (a b c : ℝ) (h1 : a^2 + b^2 = c^2) (h2 : c = 10) (h3 : a = 6) :
  (1/2) * a * b = 24 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_area_l198_19828
