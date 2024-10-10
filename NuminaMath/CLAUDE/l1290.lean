import Mathlib

namespace p_necessary_not_sufficient_l1290_129043

theorem p_necessary_not_sufficient (p q : Prop) : 
  (∀ (h : p ∧ q), p) ∧ 
  (∃ (h : p), ¬(p ∧ q)) := by
sorry

end p_necessary_not_sufficient_l1290_129043


namespace reflections_on_circumcircle_l1290_129025

-- Define the triangle ABC
variable (A B C : EuclideanSpace ℝ (Fin 2))

-- Define the orthocenter H
variable (H : EuclideanSpace ℝ (Fin 2))

-- Define the circumcircle
variable (circumcircle : Sphere (EuclideanSpace ℝ (Fin 2)))

-- Assumptions
variable (h_acute : IsAcute A B C)
variable (h_orthocenter : IsOrthocenter H A B C)
variable (h_circumcircle : IsCircumcircle circumcircle A B C)

-- Define the reflections of H with respect to the sides
def reflect_H_BC : EuclideanSpace ℝ (Fin 2) := sorry
def reflect_H_CA : EuclideanSpace ℝ (Fin 2) := sorry
def reflect_H_AB : EuclideanSpace ℝ (Fin 2) := sorry

-- Theorem statement
theorem reflections_on_circumcircle :
  circumcircle.mem reflect_H_BC ∧
  circumcircle.mem reflect_H_CA ∧
  circumcircle.mem reflect_H_AB :=
sorry

end reflections_on_circumcircle_l1290_129025


namespace largest_integer_with_3_digit_square_base7_l1290_129040

/-- The largest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to base 7 -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) : List ℕ :=
      if m = 0 then [] else (m % 7) :: aux (m / 7)
    aux n |>.reverse

theorem largest_integer_with_3_digit_square_base7 :
  (M ^ 2 ≥ 7^2) ∧ 
  (M ^ 2 < 7^3) ∧ 
  (∀ n : ℕ, n > M → n ^ 2 ≥ 7^3) ∧
  (toBase7 M = [6, 6]) := by
  sorry

#eval M
#eval toBase7 M

end largest_integer_with_3_digit_square_base7_l1290_129040


namespace ending_number_proof_l1290_129060

theorem ending_number_proof (n : ℕ) : 
  (∃ (evens : Finset ℕ), evens.card = 35 ∧ 
    (∀ x ∈ evens, 25 < x ∧ x ≤ n ∧ Even x) ∧
    (∀ y, 25 < y ∧ y ≤ n ∧ Even y → y ∈ evens)) ↔ 
  n = 94 :=
sorry

end ending_number_proof_l1290_129060


namespace system_solution_l1290_129096

theorem system_solution (a b : ℝ) (h : a ≠ b) :
  ∃! (x y : ℝ), (a + 1) * x + (a - 1) * y = a ∧ (b + 1) * x + (b - 1) * y = b ∧ x = (1 : ℝ) / 2 ∧ y = (1 : ℝ) / 2 := by
  sorry

end system_solution_l1290_129096


namespace smallest_prime_dividing_expression_l1290_129071

theorem smallest_prime_dividing_expression : 
  ∃ (a : ℕ), a > 1 ∧ 179 ∣ (a^89 - 1) / (a - 1) ∧
  ∀ (p : ℕ), p > 100 → p < 179 → Prime p → 
    ¬(∃ (b : ℕ), b > 1 ∧ p ∣ (b^89 - 1) / (b - 1)) :=
by sorry

end smallest_prime_dividing_expression_l1290_129071


namespace least_number_divisible_by_53_and_71_l1290_129015

theorem least_number_divisible_by_53_and_71 (x : ℕ) : 
  (∀ y : ℕ, y < x → ¬((1357 + y) % 53 = 0 ∧ (1357 + y) % 71 = 0)) ∧ 
  (1357 + x) % 53 = 0 ∧ (1357 + x) % 71 = 0 → 
  x = 2406 := by sorry

end least_number_divisible_by_53_and_71_l1290_129015


namespace joseph_running_distance_l1290_129088

/-- Calculates the daily running distance given the total distance and number of days -/
def dailyDistance (totalDistance : ℕ) (days : ℕ) : ℕ :=
  totalDistance / days

theorem joseph_running_distance :
  let totalDistance : ℕ := 2700
  let days : ℕ := 3
  dailyDistance totalDistance days = 900 := by
  sorry

end joseph_running_distance_l1290_129088


namespace parallel_lines_set_l1290_129014

-- Define the plane
variable (Plane : Type)

-- Define points on the plane
variable (D E P : Plane)

-- Define the distance between two points
variable (distance : Plane → Plane → ℝ)

-- Define the area of a triangle given three points
variable (triangle_area : Plane → Plane → Plane → ℝ)

-- Define a set of points
variable (T : Set Plane)

-- State the theorem
theorem parallel_lines_set (h_distinct : D ≠ E) :
  T = {P | triangle_area D E P = 0.5} →
  ∃ (l₁ l₂ : Set Plane), 
    (∀ X ∈ l₁, ∀ Y ∈ l₂, distance X Y = 2 / distance D E) ∧
    T = l₁ ∪ l₂ :=
sorry

end parallel_lines_set_l1290_129014


namespace triangle_properties_l1290_129000

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the circumradius
def circumradius (t : Triangle) : ℝ := sorry

-- Define the length of a side
def side_length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define an angle in a triangle
def angle (t : Triangle) (vertex : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem triangle_properties (t : Triangle) :
  side_length t.A t.B = Real.sqrt 10 →
  side_length t.A t.C = Real.sqrt 2 →
  circumradius t = Real.sqrt 5 →
  angle t t.C < Real.pi / 2 →
  side_length t.B t.C = 4 ∧ angle t t.C = Real.pi / 4 := by
  sorry


end triangle_properties_l1290_129000


namespace intersection_of_A_and_B_l1290_129078

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 2^x - 1}

-- Define set B
def B : Set ℝ := {x | |2*x - 3| ≤ 3}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x | 0 < x ∧ x ≤ 3} := by
  sorry

end intersection_of_A_and_B_l1290_129078


namespace middle_digit_zero_l1290_129082

theorem middle_digit_zero (N : ℕ) (a b c : ℕ) : 
  (N = 49*a + 7*b + c) →  -- N in base 7
  (N = 81*c + 9*b + a) →  -- N in base 9
  (0 ≤ a ∧ a < 7) →       -- a is a valid digit in base 7
  (0 ≤ b ∧ b < 7) →       -- b is a valid digit in base 7
  (0 ≤ c ∧ c < 7) →       -- c is a valid digit in base 7
  (b = 0) :=              -- middle digit is 0
by sorry

end middle_digit_zero_l1290_129082


namespace pet_store_birds_l1290_129007

/-- Calculates the total number of birds in a pet store with the given conditions -/
def totalBirds (totalCages : Nat) (emptyCages : Nat) (initialParrots : Nat) (initialParakeets : Nat) : Nat :=
  let nonEmptyCages := totalCages - emptyCages
  let parrotSum := nonEmptyCages * (2 * initialParrots + (nonEmptyCages - 1)) / 2
  let parakeetSum := nonEmptyCages * (2 * initialParakeets + 2 * (nonEmptyCages - 1)) / 2
  parrotSum + parakeetSum

/-- Theorem stating that the total number of birds in the pet store is 399 -/
theorem pet_store_birds : totalBirds 17 3 2 7 = 399 := by
  sorry

end pet_store_birds_l1290_129007


namespace bahs_equal_to_yahs_l1290_129072

/-- The number of bahs equal to 30 rahs -/
def bahs_to_30_rahs : ℕ := 20

/-- The number of rahs equal to 20 yahs -/
def rahs_to_20_yahs : ℕ := 12

/-- The number of yahs we want to convert to bahs -/
def yahs_to_convert : ℕ := 1200

/-- The theorem stating the equivalence between bahs and yahs -/
theorem bahs_equal_to_yahs : ∃ (n : ℕ), n * bahs_to_30_rahs * rahs_to_20_yahs = yahs_to_convert * 30 * 20 :=
sorry

end bahs_equal_to_yahs_l1290_129072


namespace smallest_five_digit_divisible_by_53_l1290_129075

theorem smallest_five_digit_divisible_by_53 : 
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 → n ≥ 10017 := by
  sorry

end smallest_five_digit_divisible_by_53_l1290_129075


namespace football_field_area_l1290_129023

theorem football_field_area (A : ℝ) 
  (h1 : 500 / 3500 = 1200 / A) : A = 8400 := by
  sorry

end football_field_area_l1290_129023


namespace trigonometric_identity_l1290_129008

theorem trigonometric_identity : 
  Real.sin (30 * π / 180) + Real.cos (120 * π / 180) + 2 * Real.cos (45 * π / 180) - Real.sqrt 3 * Real.tan (30 * π / 180) = Real.sqrt 2 - 1 := by
  sorry

end trigonometric_identity_l1290_129008


namespace swimmer_speed_in_still_water_l1290_129016

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeed where
  swimmer : ℝ  -- Speed of the swimmer in still water
  stream : ℝ   -- Speed of the stream

/-- Calculates the effective speed given swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeed) (downstream : Bool) : ℝ :=
  if downstream then s.swimmer + s.stream else s.swimmer - s.stream

/-- Theorem: Given the conditions, the swimmer's speed in still water is 4 km/h. -/
theorem swimmer_speed_in_still_water :
  ∀ (s : SwimmerSpeed),
  (effectiveSpeed s true * 6 = 30) →   -- Downstream condition
  (effectiveSpeed s false * 6 = 18) →  -- Upstream condition
  s.swimmer = 4 := by
sorry

end swimmer_speed_in_still_water_l1290_129016


namespace girls_in_class_correct_number_of_girls_l1290_129041

theorem girls_in_class (total_books : ℕ) (boys : ℕ) (girls_books : ℕ) : ℕ :=
  let boys_books := total_books - girls_books
  let books_per_student := boys_books / boys
  girls_books / books_per_student

theorem correct_number_of_girls :
  girls_in_class 375 10 225 = 15 := by
  sorry

end girls_in_class_correct_number_of_girls_l1290_129041


namespace square_field_division_l1290_129034

/-- Represents a square field with side length and division properties -/
structure SquareField where
  side_length : ℝ
  division_fence_length : ℝ

/-- Theorem: A square field of side 33m can be divided into three equal areas with at most 54m of fencing -/
theorem square_field_division (field : SquareField) 
  (h1 : field.side_length = 33) 
  (h2 : field.division_fence_length ≤ 54) : 
  ∃ (area_1 area_2 area_3 : ℝ), 
    area_1 = area_2 ∧ 
    area_2 = area_3 ∧ 
    area_1 + area_2 + area_3 = field.side_length * field.side_length := by
  sorry

end square_field_division_l1290_129034


namespace pattern_1005th_row_l1290_129099

/-- Represents the number of items in the nth row of the pattern -/
def num_items (n : ℕ) : ℕ := n

/-- Represents the sum of items in the nth row of the pattern -/
def sum_items (n : ℕ) : ℕ := n * (n + 1) / 2 + (n - 1) * n / 2

/-- Theorem stating that the 1005th row is the one where the number of items
    and their sum equals 20092 -/
theorem pattern_1005th_row :
  num_items 1005 + sum_items 1005 = 20092 := by sorry

end pattern_1005th_row_l1290_129099


namespace product_inequality_l1290_129044

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y = 1) :
  (1 + x + y^2) * (1 + y + x^2) ≥ 9 := by
  sorry

end product_inequality_l1290_129044


namespace spider_plant_theorem_l1290_129090

def spider_plant_problem (baby_plants_per_time : ℕ) (times_per_year : ℕ) (total_baby_plants : ℕ) : Prop :=
  let baby_plants_per_year := baby_plants_per_time * times_per_year
  let years_passed := total_baby_plants / baby_plants_per_year
  years_passed = 4

theorem spider_plant_theorem :
  spider_plant_problem 2 2 16 := by
  sorry

end spider_plant_theorem_l1290_129090


namespace average_of_remaining_quantities_l1290_129047

theorem average_of_remaining_quantities
  (total_count : ℕ)
  (subset_count : ℕ)
  (total_average : ℚ)
  (subset_average : ℚ)
  (h1 : total_count = 6)
  (h2 : subset_count = 4)
  (h3 : total_average = 8)
  (h4 : subset_average = 5) :
  let remaining_count := total_count - subset_count
  let remaining_sum := total_count * total_average - subset_count * subset_average
  remaining_sum / remaining_count = 14 := by
sorry

end average_of_remaining_quantities_l1290_129047


namespace larger_number_problem_l1290_129055

theorem larger_number_problem (s l : ℝ) : 
  s = 48 → l - s = (1 / 3) * l → l = 72 := by
  sorry

end larger_number_problem_l1290_129055


namespace f_composition_one_ninth_l1290_129019

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 3
  else 2^x

theorem f_composition_one_ninth : f (f (1/9)) = 1/4 := by
  sorry

end f_composition_one_ninth_l1290_129019


namespace max_product_constrained_max_value_is_three_l1290_129062

theorem max_product_constrained (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a/3 + b/4 = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → x/3 + y/4 = 1 → x*y ≤ a*b := by
  sorry

theorem max_value_is_three (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a/3 + b/4 = 1) :
  a*b = 3 := by
  sorry

end max_product_constrained_max_value_is_three_l1290_129062


namespace five_balls_four_boxes_l1290_129021

/-- Number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem five_balls_four_boxes : distribute_balls 5 4 = 60 := by sorry

end five_balls_four_boxes_l1290_129021


namespace perpendicular_lines_intersection_l1290_129038

/-- Given two perpendicular lines and the foot of their perpendicular, prove m - n + p = 20 -/
theorem perpendicular_lines_intersection (m n p : ℝ) : 
  (∀ x y, m * x + 4 * y - 2 = 0 ∨ 2 * x - 5 * y + n = 0) →  -- Two lines
  (m * 2 = -4 * 5) →  -- Perpendicularity condition
  (m * 1 + 4 * p - 2 = 0) →  -- Foot satisfies first line equation
  (2 * 1 - 5 * p + n = 0) →  -- Foot satisfies second line equation
  m - n + p = 20 := by
sorry

end perpendicular_lines_intersection_l1290_129038


namespace rational_inequality_solution_l1290_129037

theorem rational_inequality_solution (x : ℝ) :
  (x - 3) / (x^2 + 4*x + 13) ≤ 0 ↔ x ≤ 3 :=
by sorry

end rational_inequality_solution_l1290_129037


namespace line_slope_problem_l1290_129081

theorem line_slope_problem (k : ℚ) : 
  (∃ line : ℝ → ℝ, 
    (line (-1) = -4) ∧ 
    (line 3 = k) ∧ 
    (∀ x y : ℝ, x ≠ -1 → (line y - line x) / (y - x) = k)) → 
  k = 4/3 := by
sorry

end line_slope_problem_l1290_129081


namespace linear_system_solution_l1290_129094

/-- A system of linear equations with a parameter m -/
structure LinearSystem (m : ℝ) where
  eq1 : ∀ x y z : ℝ, x + m*y + 5*z = 0
  eq2 : ∀ x y z : ℝ, 4*x + m*y - 3*z = 0
  eq3 : ∀ x y z : ℝ, 3*x + 6*y - 4*z = 0

/-- The solution to the system exists and is nontrivial -/
def has_nontrivial_solution (m : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    x + m*y + 5*z = 0 ∧
    4*x + m*y - 3*z = 0 ∧
    3*x + 6*y - 4*z = 0

theorem linear_system_solution :
  ∃ m : ℝ, has_nontrivial_solution m ∧ m = 11.5 ∧
    ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → z ≠ 0 →
      x + m*y + 5*z = 0 →
      4*x + m*y - 3*z = 0 →
      3*x + 6*y - 4*z = 0 →
      x*z / (y^2) = -108/169 := by
  sorry

end linear_system_solution_l1290_129094


namespace triangle_area_l1290_129054

theorem triangle_area (a b : ℝ) (h1 : a = 8) (h2 : b = 7) : (1/2) * a * b = 28 := by
  sorry

#check triangle_area

end triangle_area_l1290_129054


namespace students_in_canteen_l1290_129084

theorem students_in_canteen (total : ℕ) (absent_fraction : ℚ) (classroom_fraction : ℚ) :
  total = 40 →
  absent_fraction = 1 / 10 →
  classroom_fraction = 3 / 4 →
  (total : ℚ) * (1 - absent_fraction) * (1 - classroom_fraction) = 9 :=
by
  sorry

end students_in_canteen_l1290_129084


namespace soccer_games_played_l1290_129093

theorem soccer_games_played (total_players : ℕ) (total_goals : ℕ) (goals_by_others : ℕ) :
  total_players = 24 →
  total_goals = 150 →
  goals_by_others = 30 →
  ∃ (games_played : ℕ),
    games_played = 15 ∧
    games_played * (total_players / 3) + goals_by_others = total_goals :=
by sorry

end soccer_games_played_l1290_129093


namespace sixth_score_for_mean_90_l1290_129048

def quiz_scores : List ℕ := [85, 90, 88, 92, 95]

def arithmetic_mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem sixth_score_for_mean_90 (x : ℕ) :
  arithmetic_mean (quiz_scores ++ [x]) = 90 → x = 90 := by
  sorry

end sixth_score_for_mean_90_l1290_129048


namespace area_of_triangle_l1290_129001

/-- Two externally tangent circles with centers O and O' and radii 1 and 2 -/
structure TangentCircles where
  O : ℝ × ℝ
  O' : ℝ × ℝ
  radius_C : ℝ
  radius_C' : ℝ
  tangent_externally : (O.1 - O'.1)^2 + (O.2 - O'.2)^2 = (radius_C + radius_C')^2
  radius_C_eq_1 : radius_C = 1
  radius_C'_eq_2 : radius_C' = 2

/-- Point P is on circle C, and P' is on circle C' -/
def TangentPoints (tc : TangentCircles) :=
  {P : ℝ × ℝ | (P.1 - tc.O.1)^2 + (P.2 - tc.O.2)^2 = tc.radius_C^2} ×
  {P' : ℝ × ℝ | (P'.1 - tc.O'.1)^2 + (P'.2 - tc.O'.2)^2 = tc.radius_C'^2}

/-- X is the intersection point of O'P and OP' -/
def IntersectionPoint (tc : TangentCircles) (tp : TangentPoints tc) : ℝ × ℝ :=
  sorry -- Definition of X as the intersection point

/-- The area of triangle OXO' -/
def TriangleArea (tc : TangentCircles) (tp : TangentPoints tc) : ℝ :=
  let X := IntersectionPoint tc tp
  sorry -- Definition of the area of triangle OXO'

/-- Main theorem: The area of triangle OXO' is (4√2 - √5) / 3 -/
theorem area_of_triangle (tc : TangentCircles) (tp : TangentPoints tc) :
  TriangleArea tc tp = (4 * Real.sqrt 2 - Real.sqrt 5) / 3 := by
  sorry

end area_of_triangle_l1290_129001


namespace ellipse_hyperbola_ab_value_l1290_129032

theorem ellipse_hyperbola_ab_value (a b : ℝ) : 
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → (x = 7 ∧ y = 0) ∨ (x = -7 ∧ y = 0)) →
  |a*b| = 2 * Real.sqrt 111 := by
sorry

end ellipse_hyperbola_ab_value_l1290_129032


namespace equation_solutions_l1290_129074

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 - 9 = 0 ↔ x = 4 ∨ x = -2) ∧
  (∀ x : ℝ, 2*x*(x - 3) + (x - 3) = 0 ↔ x = 3 ∨ x = -1/2) ∧
  (∀ x : ℝ, 2*x^2 - x - 1 = 0 ↔ x = 1 ∨ x = -1/2) ∧
  (∀ x : ℝ, x^2 - 6*x - 16 = 0 ↔ x = 8 ∨ x = -2) :=
by sorry

end equation_solutions_l1290_129074


namespace fraction_equivalence_l1290_129089

theorem fraction_equivalence (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  (∀ x y : ℝ, (x + a) / (y + c) = a / c) ↔ (∀ x y : ℝ, x = (a / c) * y) :=
sorry

end fraction_equivalence_l1290_129089


namespace discount_calculation_l1290_129003

/-- Proves the true discount and the difference between claimed and true discount for a given discount scenario. -/
theorem discount_calculation (initial_discount : ℝ) (additional_discount : ℝ) (claimed_discount : ℝ) :
  initial_discount = 0.25 →
  additional_discount = 0.1 →
  claimed_discount = 0.4 →
  let remaining_after_initial := 1 - initial_discount
  let remaining_after_additional := remaining_after_initial * (1 - additional_discount)
  let true_discount := 1 - remaining_after_additional
  true_discount = 0.325 ∧ claimed_discount - true_discount = 0.075 := by
  sorry

end discount_calculation_l1290_129003


namespace ralph_tv_hours_l1290_129080

/-- The number of hours Ralph watches TV each day from Monday to Friday -/
def weekday_hours : ℝ := sorry

/-- The number of hours Ralph watches TV each day on Saturday and Sunday -/
def weekend_hours : ℝ := 6

/-- The total number of hours Ralph watches TV in one week -/
def total_weekly_hours : ℝ := 32

/-- Theorem stating that Ralph watches TV for 4 hours each day from Monday to Friday -/
theorem ralph_tv_hours : weekday_hours = 4 := by
  have h1 : 5 * weekday_hours + 2 * weekend_hours = total_weekly_hours := sorry
  sorry

end ralph_tv_hours_l1290_129080


namespace complex_number_properties_l1290_129070

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Theorem to prove the four statements
theorem complex_number_properties :
  (i^2017 = i) ∧
  ((i + 1) * i = -1 + i) ∧
  ((1 - i) / (1 + i) = -i) ∧
  (Complex.abs (2 + i) = Real.sqrt 5) := by
  sorry

end complex_number_properties_l1290_129070


namespace cricket_bat_selling_price_l1290_129052

/-- Calculates the selling price of a cricket bat given the profit and profit percentage -/
theorem cricket_bat_selling_price (profit : ℝ) (profit_percentage : ℝ) :
  profit = 225 →
  profit_percentage = 36 →
  ∃ (cost_price selling_price : ℝ),
    cost_price = profit * 100 / profit_percentage ∧
    selling_price = cost_price + profit ∧
    selling_price = 850 := by
  sorry

end cricket_bat_selling_price_l1290_129052


namespace sum_of_a_and_b_l1290_129012

theorem sum_of_a_and_b (a b : ℝ) (h1 : |a| = 10) (h2 : |b| = 7) (h3 : a > b) :
  a + b = 17 ∨ a + b = 3 := by
  sorry

end sum_of_a_and_b_l1290_129012


namespace smallest_three_digit_square_base_seven_l1290_129056

/-- The smallest integer whose square has exactly 3 digits in base 7 -/
def M : ℕ := 7

/-- Converts a natural number to its base 7 representation -/
def to_base_seven (n : ℕ) : List ℕ := sorry

/-- Checks if a number has exactly 3 digits when written in base 7 -/
def has_three_digits_base_seven (n : ℕ) : Prop :=
  (to_base_seven n).length = 3

theorem smallest_three_digit_square_base_seven :
  (M ^ 2 ≥ 7^2) ∧
  (M ^ 2 < 7^3) ∧
  (∀ k : ℕ, k < M → ¬(has_three_digits_base_seven (k^2))) ∧
  (to_base_seven M = [1, 0]) := by sorry

end smallest_three_digit_square_base_seven_l1290_129056


namespace basketball_team_grouping_probability_l1290_129011

theorem basketball_team_grouping_probability :
  let total_teams : ℕ := 7
  let group_size_1 : ℕ := 3
  let group_size_2 : ℕ := 4
  let specific_teams : ℕ := 2
  
  let total_arrangements : ℕ := (Nat.choose total_teams group_size_1) * (Nat.choose group_size_1 group_size_1) +
                                (Nat.choose total_teams group_size_2) * (Nat.choose group_size_2 group_size_2)
  
  let favorable_arrangements : ℕ := (Nat.choose specific_teams specific_teams) *
                                    ((Nat.choose (total_teams - specific_teams) (group_size_1 - specific_teams)) +
                                     (Nat.choose (total_teams - specific_teams) (group_size_2 - specific_teams))) *
                                    (Nat.factorial specific_teams)
  
  (favorable_arrangements : ℚ) / total_arrangements = 3 / 7 :=
sorry

end basketball_team_grouping_probability_l1290_129011


namespace box_sum_remainder_l1290_129073

theorem box_sum_remainder (n : ℕ) (a : Fin (2 * n) → ℕ) :
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ (a i + i.val) % (2 * n) = (a j + j.val) % (2 * n) := by
  sorry

end box_sum_remainder_l1290_129073


namespace chess_club_problem_l1290_129069

theorem chess_club_problem (total_members chess_players checkers_players both_players : ℕ) 
  (h1 : total_members = 70)
  (h2 : chess_players = 45)
  (h3 : checkers_players = 38)
  (h4 : both_players = 25) :
  total_members - (chess_players + checkers_players - both_players) = 12 := by
  sorry

end chess_club_problem_l1290_129069


namespace chord_length_l1290_129076

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end chord_length_l1290_129076


namespace alices_favorite_number_l1290_129009

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem alices_favorite_number :
  ∃! n : ℕ, 100 < n ∧ n < 150 ∧ 
  13 ∣ n ∧ ¬(2 ∣ n) ∧ 
  4 ∣ sum_of_digits n ∧
  n = 143 :=
sorry

end alices_favorite_number_l1290_129009


namespace vertical_shift_theorem_l1290_129051

theorem vertical_shift_theorem (f : ℝ → ℝ) :
  ∀ x y : ℝ, y = f x + 3 ↔ ∃ y₀ : ℝ, y₀ = f x ∧ y = y₀ + 3 := by sorry

end vertical_shift_theorem_l1290_129051


namespace equation_solution_l1290_129049

theorem equation_solution : ∃! x : ℝ, (81 : ℝ)^(x - 2) / (9 : ℝ)^(x - 2) = (27 : ℝ)^(3*x + 2) ∧ x = -10/7 := by
  sorry

end equation_solution_l1290_129049


namespace triangle_angle_measurement_l1290_129095

theorem triangle_angle_measurement (D E F : ℝ) : 
  D = 85 →                  -- Measure of ∠D is 85 degrees
  E = 4 * F + 15 →          -- Measure of ∠E is 15 degrees more than four times the measure of ∠F
  D + E + F = 180 →         -- Sum of angles in a triangle is 180 degrees
  F = 16                    -- Measure of ∠F is 16 degrees
:= by sorry

end triangle_angle_measurement_l1290_129095


namespace range_of_m_l1290_129006

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x, x ∈ Set.Icc (-2 : ℝ) 2 → f x ∈ Set.range f) →  -- domain of f is [-2, 2]
  (∀ x y, x ∈ Set.Icc (-2 : ℝ) 2 → y ∈ Set.Icc (-2 : ℝ) 2 → x < y → f x < f y) →  -- f is increasing on [-2, 2]
  f (1 - m) < f m →  -- given condition
  m ∈ Set.Ioo (1/2 : ℝ) 2 :=  -- conclusion: m is in the open interval (1/2, 2]
by sorry

end range_of_m_l1290_129006


namespace total_cost_calculation_l1290_129077

def vegetable_price : ℝ := 2
def beef_price_multiplier : ℝ := 3
def vegetable_weight : ℝ := 6
def beef_weight : ℝ := 4

theorem total_cost_calculation : 
  (vegetable_price * vegetable_weight) + (vegetable_price * beef_price_multiplier * beef_weight) = 36 := by
  sorry

end total_cost_calculation_l1290_129077


namespace reflect_A_x_axis_l1290_129042

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point A -/
def A : ℝ × ℝ := (-3, 2)

theorem reflect_A_x_axis : reflect_x A = (-3, -2) := by
  sorry

end reflect_A_x_axis_l1290_129042


namespace clock_angle_at_3_25_l1290_129028

/-- The angle of the minute hand on a clock face at a given number of minutes past the hour -/
def minute_hand_angle (minutes : ℕ) : ℝ :=
  minutes * 6

/-- The angle of the hour hand on a clock face at a given hour and minute -/
def hour_hand_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  hour * 30 + minute * 0.5

/-- The angle between the hour hand and minute hand on a clock face -/
def clock_angle (hour : ℕ) (minute : ℕ) : ℝ :=
  |hour_hand_angle hour minute - minute_hand_angle minute|

theorem clock_angle_at_3_25 :
  clock_angle 3 25 = 47.5 := by sorry

end clock_angle_at_3_25_l1290_129028


namespace parallel_lines_same_black_cells_l1290_129020

/-- Represents a cell in the grid -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a line in the grid (horizontal, vertical, or diagonal) -/
inductive Line
  | Horizontal : Nat → Line
  | Vertical : Nat → Line
  | LeftDiagonal : Nat → Line
  | RightDiagonal : Nat → Line

/-- The grid configuration -/
structure GridConfig where
  n : Nat
  blackCells : Set Cell

/-- Two lines are parallel -/
def areLinesParallel (l1 l2 : Line) : Prop :=
  match l1, l2 with
  | Line.Horizontal _, Line.Horizontal _ => true
  | Line.Vertical _, Line.Vertical _ => true
  | Line.LeftDiagonal _, Line.LeftDiagonal _ => true
  | Line.RightDiagonal _, Line.RightDiagonal _ => true
  | _, _ => false

/-- Count black cells in a line -/
def countBlackCells (g : GridConfig) (l : Line) : Nat :=
  sorry

/-- Main theorem -/
theorem parallel_lines_same_black_cells 
  (g : GridConfig) 
  (h : g.n ≥ 3) :
  ∃ l1 l2 : Line, 
    areLinesParallel l1 l2 ∧ 
    l1 ≠ l2 ∧ 
    countBlackCells g l1 = countBlackCells g l2 := by
  sorry

end parallel_lines_same_black_cells_l1290_129020


namespace function_properties_l1290_129017

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x) + Real.cos (ω * x)

theorem function_properties (ω : ℝ) (h_ω_pos : ω > 0) 
  (h_eq : f ω (π/8) = f ω (5*π/8)) :
  (∃! (min max : ℝ), min ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    max ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    (∀ x ∈ Set.Ioo (π/8) (5*π/8), f ω x ≥ f ω min ∧ f ω x ≤ f ω max) →
    ω = 4) ∧
  (∃! (z₁ z₂ : ℝ), z₁ ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    z₂ ∈ Set.Ioo (π/8) (5*π/8) ∧ 
    f ω z₁ = 0 ∧ f ω z₂ = 0 ∧ 
    (∀ x ∈ Set.Ioo (π/8) (5*π/8), f ω x = 0 → x = z₁ ∨ x = z₂) →
    ω = 10/3 ∨ ω = 4 ∨ ω = 6) :=
by sorry

end function_properties_l1290_129017


namespace snake_length_ratio_l1290_129067

/-- The length of the garden snake in inches -/
def garden_snake_length : ℕ := 10

/-- The length of the boa constrictor in inches -/
def boa_constrictor_length : ℕ := 70

/-- The ratio of the boa constrictor's length to the garden snake's length -/
def length_ratio : ℚ := boa_constrictor_length / garden_snake_length

theorem snake_length_ratio :
  length_ratio = 7 := by sorry

end snake_length_ratio_l1290_129067


namespace fraction_equality_l1290_129033

theorem fraction_equality (x y : ℚ) (hx : x = 4/6) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 9/20 := by
  sorry

end fraction_equality_l1290_129033


namespace prism_problem_l1290_129031

theorem prism_problem (x : ℝ) (d : ℝ) : 
  x > 0 → 
  let a := Real.log x / Real.log 5
  let b := Real.log x / Real.log 7
  let c := Real.log x / Real.log 9
  let surface_area := 2 * (a * b + b * c + c * a)
  let volume := a * b * c
  surface_area * (1/3 * volume) = 54 →
  d = Real.sqrt (a^2 + b^2 + c^2) →
  x = 216 ∧ d = 7 := by
  sorry

#check prism_problem

end prism_problem_l1290_129031


namespace stone_pile_total_l1290_129063

/-- Represents the number of stones in each pile -/
structure StonePiles where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

/-- The conditions of the stone pile problem -/
def stone_pile_conditions (piles : StonePiles) : Prop :=
  piles.fifth = 6 * piles.third ∧
  piles.second = 2 * (piles.third + piles.fifth) ∧
  piles.first * 3 = piles.fifth ∧
  piles.first + 10 = piles.fourth ∧
  2 * piles.fourth = piles.second

/-- The theorem stating that under the given conditions, the total number of stones is 60 -/
theorem stone_pile_total (piles : StonePiles) 
  (h : stone_pile_conditions piles) : 
  piles.first + piles.second + piles.third + piles.fourth + piles.fifth = 60 := by
  sorry


end stone_pile_total_l1290_129063


namespace ellipse_t_squared_range_l1290_129050

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the condition for points A, B, and P
def condition (A B P : ℝ × ℝ) (t : ℝ) : Prop :=
  (A.1, A.2) + (B.1, B.2) = t • (P.1, P.2)

-- Define the inequality condition
def inequality (A B P : ℝ × ℝ) : Prop :=
  ‖(P.1 - A.1, P.2 - A.2) - (P.1 - B.1, P.2 - B.2)‖ < Real.sqrt 3

-- Theorem statement
theorem ellipse_t_squared_range :
  ∀ (A B P : ℝ × ℝ) (t : ℝ),
    ellipse A.1 A.2 → ellipse B.1 B.2 → ellipse P.1 P.2 →
    condition A B P t → inequality A B P →
    20 - Real.sqrt 283 < t^2 ∧ t^2 < 4 :=
sorry

end ellipse_t_squared_range_l1290_129050


namespace dennis_purchase_cost_l1290_129066

/-- Calculates the total cost after discount for Dennis's purchase -/
def total_cost_after_discount (pants_price : ℝ) (socks_price : ℝ) (pants_quantity : ℕ) (socks_quantity : ℕ) (discount_rate : ℝ) : ℝ :=
  let total_before_discount := pants_price * pants_quantity + socks_price * socks_quantity
  total_before_discount * (1 - discount_rate)

/-- Proves that the total cost after discount for Dennis's purchase is $392 -/
theorem dennis_purchase_cost :
  total_cost_after_discount 110 60 4 2 0.3 = 392 := by
  sorry

end dennis_purchase_cost_l1290_129066


namespace sqrt_six_div_sqrt_two_eq_sqrt_three_l1290_129039

theorem sqrt_six_div_sqrt_two_eq_sqrt_three :
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end sqrt_six_div_sqrt_two_eq_sqrt_three_l1290_129039


namespace alcohol_mixture_proof_l1290_129059

/-- Given two solutions with different alcohol concentrations, prove that mixing them in specific quantities results in a desired alcohol concentration. -/
theorem alcohol_mixture_proof (x_volume : ℝ) (y_volume : ℝ) (x_concentration : ℝ) (y_concentration : ℝ) (target_concentration : ℝ) :
  x_volume = 300 →
  y_volume = 200 →
  x_concentration = 0.1 →
  y_concentration = 0.3 →
  target_concentration = 0.18 →
  (x_volume * x_concentration + y_volume * y_concentration) / (x_volume + y_volume) = target_concentration :=
by sorry


end alcohol_mixture_proof_l1290_129059


namespace sum_of_coefficients_l1290_129045

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
  sorry

end sum_of_coefficients_l1290_129045


namespace insects_in_lab_l1290_129036

/-- The number of insects in a laboratory given the total number of legs --/
def number_of_insects (total_legs : ℕ) (legs_per_insect : ℕ) : ℕ :=
  total_legs / legs_per_insect

/-- Theorem: The number of insects in the laboratory is 5 --/
theorem insects_in_lab : number_of_insects 30 6 = 5 := by
  sorry

end insects_in_lab_l1290_129036


namespace AR_equals_six_l1290_129061

-- Define the triangle and points
variable (A B C R P Q : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (acute_triangle : AcuteTriangle A B C)
variable (R_on_perpendicular_bisector : OnPerpendicularBisector R A C)
variable (CA_bisects_BAR : AngleBisector C A (B, R))
variable (Q_intersection : OnLine Q A C ∧ OnLine Q B R)
variable (P_on_circumcircle : OnCircumcircle P A R C)
variable (P_on_AB : SegmentND P A B)
variable (AP_length : dist A P = 1)
variable (PB_length : dist P B = 5)
variable (AQ_length : dist A Q = 2)

-- State the theorem
theorem AR_equals_six : dist A R = 6 := by sorry

end AR_equals_six_l1290_129061


namespace waiter_tip_calculation_l1290_129002

/-- Waiter's tip calculation problem -/
theorem waiter_tip_calculation
  (total_customers : ℕ)
  (non_tipping_customers : ℕ)
  (total_tip_amount : ℕ)
  (h1 : total_customers = 9)
  (h2 : non_tipping_customers = 5)
  (h3 : total_tip_amount = 32) :
  total_tip_amount / (total_customers - non_tipping_customers) = 8 := by
  sorry

#check waiter_tip_calculation

end waiter_tip_calculation_l1290_129002


namespace apple_production_formula_l1290_129091

/-- Represents an apple orchard with additional trees planted -/
structure Orchard where
  initial_trees : ℕ
  initial_avg_apples : ℕ
  decrease_per_tree : ℕ
  additional_trees : ℕ

/-- Calculates the total number of apples produced in an orchard -/
def total_apples (o : Orchard) : ℕ :=
  (o.initial_trees + o.additional_trees) * (o.initial_avg_apples - o.decrease_per_tree * o.additional_trees)

/-- Theorem stating the relationship between additional trees and total apples -/
theorem apple_production_formula (x : ℕ) :
  let o : Orchard := {
    initial_trees := 10,
    initial_avg_apples := 200,
    decrease_per_tree := 5,
    additional_trees := x
  }
  total_apples o = (10 + x) * (200 - 5 * x) := by
  sorry

end apple_production_formula_l1290_129091


namespace binary_110101_equals_53_l1290_129030

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enum b).foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110101_equals_53 :
  binary_to_decimal [true, false, true, false, true, true] = 53 := by
  sorry

end binary_110101_equals_53_l1290_129030


namespace equation_solutions_l1290_129087

theorem equation_solutions :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁^2 - 9*y₁^2 = 18 ∧
    x₂^2 - 9*y₂^2 = 18 ∧
    x₁ = 19/2 ∧ y₁ = 17/6 ∧
    x₂ = 11/2 ∧ y₂ = 7/6 :=
by sorry

end equation_solutions_l1290_129087


namespace square_field_area_l1290_129068

/-- Proves that a square field with given conditions has an area of 27889 square meters -/
theorem square_field_area (wire_cost_per_meter : ℝ) (total_cost : ℝ) (gate_width : ℝ) (num_gates : ℕ) :
  wire_cost_per_meter = 1.30 →
  total_cost = 865.80 →
  gate_width = 1 →
  num_gates = 2 →
  ∃ (side_length : ℝ),
    (4 * side_length - gate_width * num_gates) * wire_cost_per_meter = total_cost ∧
    side_length^2 = 27889 :=
by sorry

end square_field_area_l1290_129068


namespace distance_to_city_l1290_129004

theorem distance_to_city (D : ℝ) 
  (h1 : D / 2 + D / 4 + 6 = D) : D = 24 := by
  sorry

end distance_to_city_l1290_129004


namespace abs_eq_sqrt_square_l1290_129092

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end abs_eq_sqrt_square_l1290_129092


namespace pentagon_from_equal_segments_l1290_129013

theorem pentagon_from_equal_segments (segment_length : Real) 
  (h1 : segment_length = 2 / 5)
  (h2 : 5 * segment_length = 2) : 
  4 * segment_length > segment_length := by
  sorry

end pentagon_from_equal_segments_l1290_129013


namespace min_value_of_product_l1290_129065

-- Define the quadratic function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem min_value_of_product (a b c : ℝ) (x₁ x₂ x₃ : ℝ) :
  a ≠ 0 →
  f a b c (-1) = 0 →
  (∀ x : ℝ, f a b c x ≥ x) →
  (∀ x ∈ Set.Ioo 0 2, f a b c x ≤ (x + 1)^2 / 4) →
  x₁ ∈ Set.Ioo 0 2 →
  x₂ ∈ Set.Ioo 0 2 →
  x₃ ∈ Set.Ioo 0 2 →
  1 / x₁ + 1 / x₂ + 1 / x₃ = 3 →
  (f a b c x₁) * (f a b c x₂) * (f a b c x₃) ≥ 1 :=
by sorry

#check min_value_of_product

end min_value_of_product_l1290_129065


namespace prob_sum_le_5_is_correct_l1290_129086

/-- The probability of the sum of two fair six-sided dice being less than or equal to 5 -/
def prob_sum_le_5 : ℚ :=
  5 / 18

/-- The set of possible outcomes when rolling two dice -/
def dice_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range 6) (Finset.range 6)

/-- The set of favorable outcomes (sum ≤ 5) when rolling two dice -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  dice_outcomes.filter (fun p => p.1 + p.2 + 2 ≤ 5)

/-- Theorem stating that the probability of the sum of two fair six-sided dice
    being less than or equal to 5 is 5/18 -/
theorem prob_sum_le_5_is_correct :
  (favorable_outcomes.card : ℚ) / dice_outcomes.card = prob_sum_le_5 :=
sorry

end prob_sum_le_5_is_correct_l1290_129086


namespace acid_dilution_l1290_129079

/-- Given n ounces of n% acid solution, to obtain a (n-20)% solution by adding y ounces of water, 
    where n > 30, y must equal 20n / (n-20). -/
theorem acid_dilution (n : ℝ) (y : ℝ) (h : n > 30) :
  (n * n / 100 = (n - 20) * (n + y) / 100) → y = 20 * n / (n - 20) := by
  sorry

end acid_dilution_l1290_129079


namespace calculation_proof_l1290_129029

theorem calculation_proof : 
  41 * ((2 + 2/7) - (3 + 3/5)) / ((3 + 1/5) + (2 + 1/4)) = -10 := by
  sorry

end calculation_proof_l1290_129029


namespace binomial_9_choose_5_l1290_129035

theorem binomial_9_choose_5 : Nat.choose 9 5 = 126 := by
  sorry

end binomial_9_choose_5_l1290_129035


namespace smaller_integer_problem_l1290_129097

theorem smaller_integer_problem (x y : ℤ) : 
  y = 2 * x → x + y = 96 → x = 32 := by
  sorry

end smaller_integer_problem_l1290_129097


namespace quadratic_equation_has_solution_l1290_129058

theorem quadratic_equation_has_solution (a b : ℝ) :
  ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 := by
  sorry

end quadratic_equation_has_solution_l1290_129058


namespace race_champion_is_C_l1290_129053

-- Define the participants
inductive Participant : Type
| A : Participant
| B : Participant
| C : Participant
| D : Participant

-- Define the opinions
def xiaozhangs_opinion (champion : Participant) : Prop :=
  champion = Participant.A ∨ champion = Participant.B

def xiaowangs_opinion (champion : Participant) : Prop :=
  champion ≠ Participant.C

def xiaolis_opinion (champion : Participant) : Prop :=
  champion ≠ Participant.A ∧ champion ≠ Participant.B

-- Theorem statement
theorem race_champion_is_C :
  ∀ (champion : Participant),
    (xiaozhangs_opinion champion ∨ xiaowangs_opinion champion ∨ xiaolis_opinion champion) ∧
    (¬(xiaozhangs_opinion champion ∧ xiaowangs_opinion champion) ∧
     ¬(xiaozhangs_opinion champion ∧ xiaolis_opinion champion) ∧
     ¬(xiaowangs_opinion champion ∧ xiaolis_opinion champion)) →
    champion = Participant.C :=
by sorry

end race_champion_is_C_l1290_129053


namespace mod_eight_difference_l1290_129046

theorem mod_eight_difference (n : ℕ) : (47^1824 - 25^1824) % 8 = 0 := by
  sorry

end mod_eight_difference_l1290_129046


namespace fifth_bowler_score_l1290_129026

/-- A bowling team with 5 members and their scores -/
structure BowlingTeam where
  total_points : ℕ
  p1 : ℕ
  p2 : ℕ
  p3 : ℕ
  p4 : ℕ
  p5 : ℕ

/-- The conditions of the bowling team's scores -/
def validBowlingTeam (team : BowlingTeam) : Prop :=
  team.total_points = 2000 ∧
  team.p1 = team.p2 / 4 ∧
  team.p2 = team.p3 * 5 / 3 ∧
  team.p3 ≤ 500 ∧
  team.p3 = team.p4 * 3 / 5 ∧
  team.p4 = team.p5 * 9 / 10 ∧
  team.p1 + team.p2 + team.p3 + team.p4 + team.p5 = team.total_points

theorem fifth_bowler_score (team : BowlingTeam) :
  validBowlingTeam team → team.p5 = 561 := by
  sorry

end fifth_bowler_score_l1290_129026


namespace geometric_arithmetic_sequence_l1290_129085

/-- Given three positive real numbers that form a geometric sequence,
    their sum is 21, and subtracting 9 from the third number results
    in an arithmetic sequence, prove that the numbers are either
    (1, 4, 16) or (16, 4, 1). -/
theorem geometric_arithmetic_sequence (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive numbers
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- geometric sequence
  a + b + c = 21 →  -- sum is 21
  ∃ d : ℝ, b - a = d ∧ (c - 9) - b = d →  -- arithmetic sequence after subtracting 9
  ((a = 1 ∧ b = 4 ∧ c = 16) ∨ (a = 16 ∧ b = 4 ∧ c = 1)) :=
by sorry

end geometric_arithmetic_sequence_l1290_129085


namespace toothpicks_43_10_l1290_129098

/-- The number of toothpicks used in a 1 × 10 grid -/
def toothpicks_1_10 : ℕ := 31

/-- The number of toothpicks used in an n × 10 grid -/
def toothpicks_n_10 (n : ℕ) : ℕ := 21 * n + 10

/-- Theorem: The number of toothpicks in a 43 × 10 grid is 913 -/
theorem toothpicks_43_10 :
  toothpicks_n_10 43 = 913 :=
sorry

end toothpicks_43_10_l1290_129098


namespace ellipse_origin_inside_l1290_129018

theorem ellipse_origin_inside (k : ℝ) : 
  (∀ x y : ℝ, k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1 = 0 → x^2 + y^2 > 0) →
  (k^2 * 0^2 + 0^2 - 4*k*0 + 2*k*0 + k^2 - 1 < 0) →
  0 < |k| ∧ |k| < 1 := by
sorry

end ellipse_origin_inside_l1290_129018


namespace pandas_minus_lions_l1290_129083

/-- The number of animals in John's zoo --/
structure ZooAnimals where
  snakes : ℕ
  monkeys : ℕ
  lions : ℕ
  pandas : ℕ
  dogs : ℕ

/-- The conditions of John's zoo --/
def validZoo (zoo : ZooAnimals) : Prop :=
  zoo.snakes = 15 ∧
  zoo.monkeys = 2 * zoo.snakes ∧
  zoo.lions = zoo.monkeys - 5 ∧
  zoo.dogs = zoo.pandas / 3 ∧
  zoo.snakes + zoo.monkeys + zoo.lions + zoo.pandas + zoo.dogs = 114

/-- The theorem to prove --/
theorem pandas_minus_lions (zoo : ZooAnimals) (h : validZoo zoo) : 
  zoo.pandas - zoo.lions = 8 := by
  sorry


end pandas_minus_lions_l1290_129083


namespace cube_sum_from_sum_and_square_sum_l1290_129027

theorem cube_sum_from_sum_and_square_sum (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : x^2 + y^2 = 13) : 
  x^3 + y^3 = 35 := by
sorry

end cube_sum_from_sum_and_square_sum_l1290_129027


namespace hyperbola_foci_l1290_129010

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 / 5 = 1

/-- The coordinates of a focus of the hyperbola -/
def focus_coordinate : ℝ × ℝ := (3, 0)

/-- Theorem: The coordinates of the foci of the hyperbola x^2/4 - y^2/5 = 1 are (±3, 0) -/
theorem hyperbola_foci :
  (∀ x y, hyperbola_equation x y → 
    (x = focus_coordinate.1 ∧ y = focus_coordinate.2) ∨ 
    (x = -focus_coordinate.1 ∧ y = focus_coordinate.2)) :=
by sorry

end hyperbola_foci_l1290_129010


namespace hexagon_enclosure_l1290_129024

theorem hexagon_enclosure (m n : ℕ) (h1 : m = 6) (h2 : m + 1 = 7) : 
  (3 * (360 / n) = 2 * (180 - (m - 2) * 180 / m)) → n = 6 := by
  sorry

end hexagon_enclosure_l1290_129024


namespace lars_baking_hours_l1290_129022

/-- The number of loaves of bread Lars can bake per hour -/
def loaves_per_hour : ℕ := 10

/-- The number of baguettes Lars can bake per hour -/
def baguettes_per_hour : ℕ := 15

/-- The total number of breads Lars makes -/
def total_breads : ℕ := 150

/-- The number of hours Lars bakes each day -/
def baking_hours : ℕ := 6

theorem lars_baking_hours :
  loaves_per_hour * baking_hours + baguettes_per_hour * baking_hours = total_breads :=
sorry

end lars_baking_hours_l1290_129022


namespace expression_evaluation_l1290_129057

theorem expression_evaluation (x y z : ℝ) :
  3 * (x - (2 * y - 3 * z)) - 2 * ((3 * x - 2 * y) - 4 * z) = -3 * x - 2 * y + 17 * z :=
by sorry

end expression_evaluation_l1290_129057


namespace product_of_primes_with_square_sum_l1290_129064

theorem product_of_primes_with_square_sum (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ →
  p₁^2 + p₂^2 + p₃^2 + p₄^2 = 476 →
  p₁ * p₂ * p₃ * p₄ = 1989 := by
sorry

end product_of_primes_with_square_sum_l1290_129064


namespace square_perimeter_from_area_l1290_129005

theorem square_perimeter_from_area (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 450 → 
  side * side = area → 
  perimeter = 4 * side → 
  perimeter = 60 * Real.sqrt 2 := by
sorry

end square_perimeter_from_area_l1290_129005
