import Mathlib

namespace NUMINAMATH_CALUDE_monotonic_increasing_condition_monotonic_decreasing_interval_condition_not_always_above_line_l1140_114034

-- Define the function f(x) = x^3 - ax - 1
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x - 1

-- Statement 1
theorem monotonic_increasing_condition (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → a ≤ 0 :=
sorry

-- Statement 2
theorem monotonic_decreasing_interval_condition (a : ℝ) :
  (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x > f a y) → a ≥ 3 :=
sorry

-- Statement 3
theorem not_always_above_line (a : ℝ) :
  ∃ x : ℝ, f a x < a :=
sorry

end NUMINAMATH_CALUDE_monotonic_increasing_condition_monotonic_decreasing_interval_condition_not_always_above_line_l1140_114034


namespace NUMINAMATH_CALUDE_unreasonable_milk_volume_l1140_114019

/-- Represents the volume of milk in liters --/
def milk_volume : ℝ := 250

/-- Represents a reasonable maximum volume of milk a person can drink in a day (in liters) --/
def max_reasonable_volume : ℝ := 10

/-- Theorem stating that the given milk volume is unreasonable for a person to drink in a day --/
theorem unreasonable_milk_volume : milk_volume > max_reasonable_volume := by
  sorry

end NUMINAMATH_CALUDE_unreasonable_milk_volume_l1140_114019


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1140_114030

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {y | ∃ x, y = 2^x + 1}

-- Define set B
def B : Set ℝ := {x | Real.log x < 0}

-- Statement to prove
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x | x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1140_114030


namespace NUMINAMATH_CALUDE_cylinder_radius_with_prisms_l1140_114057

theorem cylinder_radius_with_prisms (h₁ h₂ d : ℝ) (h₁_pos : h₁ > 0) (h₂_pos : h₂ > 0) (d_pos : d > 0) 
  (h₁_eq : h₁ = 9) (h₂_eq : h₂ = 2) (d_eq : d = 23) : ∃ R : ℝ, 
  R > 0 ∧ 
  R^2 = (R - h₁)^2 + (d - x)^2 ∧ 
  R^2 = (R - h₂)^2 + x^2 ∧ 
  R = 17 :=
sorry

end NUMINAMATH_CALUDE_cylinder_radius_with_prisms_l1140_114057


namespace NUMINAMATH_CALUDE_even_function_implies_cubic_l1140_114062

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = (m+1)x^2 + (m-2)x -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m + 1) * x^2 + (m - 2) * x

theorem even_function_implies_cubic (m : ℝ) :
  IsEven (f m) → f m = fun x ↦ 3 * x^2 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_cubic_l1140_114062


namespace NUMINAMATH_CALUDE_men_with_ac_at_least_12_l1140_114038

theorem men_with_ac_at_least_12 (total : ℕ) (married : ℕ) (tv : ℕ) (radio : ℕ) (all_four : ℕ) (ac : ℕ) :
  total = 100 →
  married = 82 →
  tv = 75 →
  radio = 85 →
  all_four = 12 →
  all_four ≤ ac →
  ac ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_men_with_ac_at_least_12_l1140_114038


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l1140_114086

theorem butterflies_in_garden (initial : ℕ) (fraction : ℚ) (remaining : ℕ) : 
  initial = 9 → fraction = 1/3 → remaining = initial - (initial * fraction).floor → remaining = 6 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l1140_114086


namespace NUMINAMATH_CALUDE_scientific_notation_36000_l1140_114091

/-- Proves that 36000 is equal to 3.6 * 10^4 in scientific notation -/
theorem scientific_notation_36000 :
  36000 = 3.6 * (10 : ℝ)^4 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_36000_l1140_114091


namespace NUMINAMATH_CALUDE_inequality_proof_l1140_114043

theorem inequality_proof (a b : ℝ) (h1 : 0 < b) (h2 : b < 1) (h3 : 1 < a) :
  a * b^2 < a * b ∧ a * b < a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1140_114043


namespace NUMINAMATH_CALUDE_parallelogram_angle_equality_l1140_114018

-- Define the points
variable (A B C D P : Point)

-- Define the parallelogram property
def is_parallelogram (A B C D : Point) : Prop := sorry

-- Define the angle equality
def angle_eq (P Q R S T U : Point) : Prop := sorry

-- State the theorem
theorem parallelogram_angle_equality 
  (h_parallelogram : is_parallelogram A B C D)
  (h_angle_eq : angle_eq P A D P C D) :
  angle_eq P B C P D C := by sorry

end NUMINAMATH_CALUDE_parallelogram_angle_equality_l1140_114018


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1140_114061

theorem geometric_sequence_property (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, a n = 4 * q^(n-1)) →  -- a_n is a geometric sequence with a_1 = 4 and common ratio q
  (∀ n, S n = 4 * (1 - q^n) / (1 - q)) →  -- S_n is the sum of the first n terms
  (∃ r, ∀ n, (S (n+1) + 2) = r * (S n + 2)) →  -- {S_n + 2} is a geometric sequence
  q = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1140_114061


namespace NUMINAMATH_CALUDE_logical_equivalence_l1140_114089

theorem logical_equivalence (R S T : Prop) :
  (R → ¬S ∧ ¬T) ↔ (S ∨ T → ¬R) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l1140_114089


namespace NUMINAMATH_CALUDE_escalator_speed_l1140_114010

/-- Proves that the escalator speed is 12 feet per second given the conditions -/
theorem escalator_speed (escalator_length : ℝ) (person_speed : ℝ) (time_taken : ℝ) :
  escalator_length = 210 →
  person_speed = 2 →
  time_taken = 15 →
  (person_speed + (escalator_length / time_taken)) * time_taken = escalator_length →
  escalator_length / time_taken = 12 := by
  sorry


end NUMINAMATH_CALUDE_escalator_speed_l1140_114010


namespace NUMINAMATH_CALUDE_rectangle_longer_side_length_l1140_114022

/-- Given a circle with radius 6 cm tangent to three sides of a rectangle,
    and the area of the rectangle being three times the area of the circle,
    the length of the longer side of the rectangle is 9π cm. -/
theorem rectangle_longer_side_length (circle_radius : ℝ) (rectangle_area : ℝ) (circle_area : ℝ)
  (h1 : circle_radius = 6)
  (h2 : rectangle_area = 3 * circle_area)
  (h3 : circle_area = Real.pi * circle_radius ^ 2)
  (h4 : rectangle_area = 12 * longer_side) :
  longer_side = 9 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_rectangle_longer_side_length_l1140_114022


namespace NUMINAMATH_CALUDE_largest_n_for_perfect_square_l1140_114016

theorem largest_n_for_perfect_square (n : ℕ) : 
  (∃ k : ℕ, 4^27 + 4^500 + 4^n = k^2) → n ≤ 972 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_perfect_square_l1140_114016


namespace NUMINAMATH_CALUDE_journey_time_change_l1140_114093

/-- Given a journey that takes 5 hours at 80 miles per hour, prove that the same journey at 50 miles per hour will take 8 hours. -/
theorem journey_time_change (initial_time initial_speed new_speed : ℝ) 
  (h1 : initial_time = 5)
  (h2 : initial_speed = 80)
  (h3 : new_speed = 50) :
  (initial_time * initial_speed) / new_speed = 8 := by
  sorry

end NUMINAMATH_CALUDE_journey_time_change_l1140_114093


namespace NUMINAMATH_CALUDE_sum_of_cubes_roots_l1140_114096

theorem sum_of_cubes_roots (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - a*x₁ + a + 2 = 0 ∧ 
                x₂^2 - a*x₂ + a + 2 = 0 ∧ 
                x₁^3 + x₂^3 = -8) ↔ 
  a = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_cubes_roots_l1140_114096


namespace NUMINAMATH_CALUDE_brookes_initial_balloons_l1140_114080

theorem brookes_initial_balloons :
  ∀ (b : ℕ), -- Brooke's initial number of balloons
  let brooke_final := b + 8 -- Brooke's final number of balloons
  let tracy_initial := 6 -- Tracy's initial number of balloons
  let tracy_added := 24 -- Number of balloons Tracy adds
  let tracy_final := (tracy_initial + tracy_added) / 2 -- Tracy's final number of balloons after popping half
  brooke_final + tracy_final = 35 → b = 12 :=
by
  sorry

#check brookes_initial_balloons

end NUMINAMATH_CALUDE_brookes_initial_balloons_l1140_114080


namespace NUMINAMATH_CALUDE_cubeRoot_of_negative_27_l1140_114081

-- Define the cube root function
noncomputable def cubeRoot (x : ℝ) : ℝ := Real.rpow x (1/3)

-- State the theorem
theorem cubeRoot_of_negative_27 : cubeRoot (-27) = -3 := by
  sorry

end NUMINAMATH_CALUDE_cubeRoot_of_negative_27_l1140_114081


namespace NUMINAMATH_CALUDE_sum_lent_proof_l1140_114092

/-- Proves that given a sum P lent at 4% per annum simple interest,
    if the interest after 4 years is Rs. 1260 less than P, then P = 1500. -/
theorem sum_lent_proof (P : ℝ) : 
  (P * (4 / 100) * 4 = P - 1260) → P = 1500 := by sorry

end NUMINAMATH_CALUDE_sum_lent_proof_l1140_114092


namespace NUMINAMATH_CALUDE_fraction_multiplication_l1140_114008

theorem fraction_multiplication (x : ℚ) : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5020 = 753 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l1140_114008


namespace NUMINAMATH_CALUDE_expression_equals_negative_one_l1140_114026

theorem expression_equals_negative_one :
  -5 * (2/3) + 6 * (2/7) + (1/3) * (-5) - (2/7) * (-8) = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_negative_one_l1140_114026


namespace NUMINAMATH_CALUDE_min_y_difference_parabola_l1140_114079

/-- Parabola structure -/
structure Parabola where
  a : ℝ
  vertex : ℝ × ℝ

/-- Point on a parabola -/
structure ParabolaPoint (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4 * p.a * (x - p.vertex.1)

/-- Line passing through two points -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem: Minimum value of |y₁ - 4y₂| for points on parabola y² = 4x -/
theorem min_y_difference_parabola (p : Parabola) 
  (h_p : p.a = 1 ∧ p.vertex = (0, 0))
  (l : Line) 
  (h_l : l.a * 1 + l.b * 0 + l.c = 0)  -- Line passes through focus (1, 0)
  (A B : ParabolaPoint p)
  (h_A : A.x ≥ 0 ∧ A.y ≥ 0)  -- A is in the first quadrant
  (h_line : l.a * A.x + l.b * A.y + l.c = 0 ∧ 
            l.a * B.x + l.b * B.y + l.c = 0)  -- A and B are on the line
  : ∃ (y₁ y₂ : ℝ), |y₁ - 4*y₂| ≥ 8 ∧ 
    (∃ (A' B' : ParabolaPoint p), 
      |A'.y - 4*B'.y| = 8 ∧ 
      l.a * A'.x + l.b * A'.y + l.c = 0 ∧ 
      l.a * B'.x + l.b * B'.y + l.c = 0) := by
  sorry

end NUMINAMATH_CALUDE_min_y_difference_parabola_l1140_114079


namespace NUMINAMATH_CALUDE_percentage_calculation_l1140_114047

theorem percentage_calculation : 
  (0.60 * 4500 * 0.40 * 2800) - (0.80 * 1750 + 0.35 * 3000) = 3021550 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1140_114047


namespace NUMINAMATH_CALUDE_ladas_isosceles_triangle_l1140_114031

theorem ladas_isosceles_triangle 
  (α β γ : ℝ) 
  (triangle_sum : α + β + γ = 180)
  (positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ)
  (sum_angles_exist : ∃ δ ε : ℝ, 0 < δ ∧ 0 < ε ∧ δ + ε ≤ 180 ∧ δ = α + β ∧ ε = α + γ) :
  β = γ := by
sorry

end NUMINAMATH_CALUDE_ladas_isosceles_triangle_l1140_114031


namespace NUMINAMATH_CALUDE_divisibility_by_1956_l1140_114014

theorem divisibility_by_1956 (n : ℕ) (h : Odd n) :
  ∃ k : ℤ, 24 * 80^n + 1992 * 83^(n-1) = 1956 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1956_l1140_114014


namespace NUMINAMATH_CALUDE_expression_equality_l1140_114044

theorem expression_equality (y b : ℝ) (h1 : y > 0) 
  (h2 : (4 * y) / b + (3 * y) / 10 = 0.5 * y) : b = 20 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1140_114044


namespace NUMINAMATH_CALUDE_range_of_a_l1140_114035

theorem range_of_a (p q : ℝ → Prop) :
  (∀ x, q x → p x) ∧
  (∃ x, p x ∧ ¬q x) ∧
  (∀ x, q x ↔ -x^2 + 5*x - 6 > 0) ∧
  (∀ x a, p x ↔ |x - a| < 4) →
  ∃ a_min a_max, a_min = -1 ∧ a_max = 6 ∧ ∀ a, (a_min < a ∧ a < a_max) → 
    (∃ x, p x) ∧ (∃ x, ¬p x) := by sorry


end NUMINAMATH_CALUDE_range_of_a_l1140_114035


namespace NUMINAMATH_CALUDE_least_integer_with_deletion_property_l1140_114020

theorem least_integer_with_deletion_property : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x = 950) ∧ 
  (∀ y : ℕ, y > 0 ∧ y < x → ¬(y / 10 = y / 19)) ∧
  (x / 10 = x / 19) := by
  sorry

end NUMINAMATH_CALUDE_least_integer_with_deletion_property_l1140_114020


namespace NUMINAMATH_CALUDE_percent_of_x_is_z_l1140_114005

theorem percent_of_x_is_z (x y z : ℝ) 
  (h1 : 0.45 * z = 1.20 * y) 
  (h2 : y = 0.75 * x) : 
  z = 2 * x := by
sorry

end NUMINAMATH_CALUDE_percent_of_x_is_z_l1140_114005


namespace NUMINAMATH_CALUDE_dvd_book_capacity_l1140_114098

theorem dvd_book_capacity (total_capacity : ℕ) (current_dvds : ℕ) (h1 : total_capacity = 126) (h2 : current_dvds = 81) :
  total_capacity - current_dvds = 45 := by
  sorry

end NUMINAMATH_CALUDE_dvd_book_capacity_l1140_114098


namespace NUMINAMATH_CALUDE_correct_problem_percentage_l1140_114028

/-- Given a total number of problems and the number of missed problems,
    calculate the percentage of correctly solved problems. -/
theorem correct_problem_percentage
  (x : ℕ) -- x represents the number of missed problems
  (h : x > 0) -- ensure x is positive to avoid division by zero
  : (((7 : ℚ) * x - x) / (7 * x)) * 100 = (6 : ℚ) / 7 * 100 := by
  sorry

#eval (6 : ℚ) / 7 * 100 -- To show the approximate result

end NUMINAMATH_CALUDE_correct_problem_percentage_l1140_114028


namespace NUMINAMATH_CALUDE_initial_shoe_pairs_l1140_114078

theorem initial_shoe_pairs (remaining_pairs : ℕ) (lost_shoes : ℕ) : 
  remaining_pairs = 19 → lost_shoes = 9 → 
  (2 * remaining_pairs + lost_shoes + 1) / 2 = 23 := by
sorry

end NUMINAMATH_CALUDE_initial_shoe_pairs_l1140_114078


namespace NUMINAMATH_CALUDE_linear_function_quadrants_l1140_114076

/-- A linear function f(x) = mx + b passes through a quadrant if there exists a point (x, f(x)) in that quadrant. -/
def passes_through_quadrant (m b : ℝ) (quad : ℕ) : Prop :=
  ∃ x y : ℝ, y = m * x + b ∧
  match quad with
  | 1 => x > 0 ∧ y > 0
  | 2 => x < 0 ∧ y > 0
  | 3 => x < 0 ∧ y < 0
  | 4 => x > 0 ∧ y < 0
  | _ => False

/-- The slope of the linear function -/
def m : ℝ := -5

/-- The y-intercept of the linear function -/
def b : ℝ := 3

/-- Theorem stating that the linear function f(x) = -5x + 3 passes through Quadrants I, II, and IV -/
theorem linear_function_quadrants :
  passes_through_quadrant m b 1 ∧
  passes_through_quadrant m b 2 ∧
  passes_through_quadrant m b 4 ∧
  ¬passes_through_quadrant m b 3 :=
sorry

end NUMINAMATH_CALUDE_linear_function_quadrants_l1140_114076


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1140_114024

theorem expand_and_simplify (a b : ℝ) : (3*a + b) * (a - b) = 3*a^2 - 2*a*b - b^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1140_114024


namespace NUMINAMATH_CALUDE_pauls_weekend_homework_l1140_114037

/-- Represents Paul's homework schedule for a week -/
structure HomeworkSchedule where
  weeknight_hours : ℕ  -- Hours of homework on a regular weeknight
  practice_nights : ℕ  -- Number of nights with practice (no homework)
  total_weeknights : ℕ -- Total number of weeknights
  average_hours : ℕ   -- Required average hours on non-practice nights

/-- Calculates the weekend homework hours based on Paul's schedule -/
def weekend_homework (schedule : HomeworkSchedule) : ℕ :=
  let non_practice_nights := schedule.total_weeknights - schedule.practice_nights
  let required_hours := non_practice_nights * schedule.average_hours
  let available_weeknight_hours := (schedule.total_weeknights - schedule.practice_nights) * schedule.weeknight_hours
  required_hours - available_weeknight_hours

/-- Theorem stating that Paul's weekend homework is 3 hours -/
theorem pauls_weekend_homework :
  let pauls_schedule : HomeworkSchedule := {
    weeknight_hours := 2,
    practice_nights := 2,
    total_weeknights := 5,
    average_hours := 3
  }
  weekend_homework pauls_schedule = 3 := by sorry

end NUMINAMATH_CALUDE_pauls_weekend_homework_l1140_114037


namespace NUMINAMATH_CALUDE_min_values_theorem_l1140_114067

theorem min_values_theorem (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m + n = 2 * m * n) :
  (m + n ≥ 2) ∧ (Real.sqrt (m * n) ≥ 1) ∧ (n^2 / m + m^2 / n ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_min_values_theorem_l1140_114067


namespace NUMINAMATH_CALUDE_small_pizza_price_is_two_l1140_114095

/-- The price of a small pizza given the conditions of the problem -/
def small_pizza_price (large_pizza_price : ℕ) (total_sales : ℕ) (small_pizzas_sold : ℕ) (large_pizzas_sold : ℕ) : ℕ :=
  (total_sales - large_pizza_price * large_pizzas_sold) / small_pizzas_sold

/-- Theorem stating that the price of a small pizza is $2 under the given conditions -/
theorem small_pizza_price_is_two :
  small_pizza_price 8 40 8 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_small_pizza_price_is_two_l1140_114095


namespace NUMINAMATH_CALUDE_drop_notation_l1140_114011

/-- Represents a temperature change in Celsius -/
structure TempChange where
  value : ℤ

/-- Notation for temperature changes -/
def temp_notation (change : TempChange) : ℤ :=
  change.value

/-- Given condition: A temperature rise of 3℃ is denoted as +3℃ -/
axiom rise_notation : temp_notation ⟨3⟩ = 3

/-- Theorem: A temperature drop of 8℃ is denoted as -8℃ -/
theorem drop_notation : temp_notation ⟨-8⟩ = -8 := by
  sorry

end NUMINAMATH_CALUDE_drop_notation_l1140_114011


namespace NUMINAMATH_CALUDE_percentage_difference_l1140_114040

theorem percentage_difference (x y : ℝ) (h : x = 11 * y) :
  (x - y) / x * 100 = 10 / 11 * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1140_114040


namespace NUMINAMATH_CALUDE_cindy_hit_nine_l1140_114054

/-- Represents a player in the dart-throwing contest -/
inductive Player
| Alice
| Ben
| Cindy
| Dave
| Ellen

/-- Represents the score of a single dart throw -/
def DartScore := Fin 15

/-- Represents the scores of three dart throws for a player -/
def PlayerScores := Fin 3 → DartScore

/-- The total score for a player is the sum of their three dart scores -/
def totalScore (scores : PlayerScores) : Nat :=
  (scores 0).val + (scores 1).val + (scores 2).val

/-- The scores for each player -/
def playerTotalScores : Player → Nat
| Player.Alice => 24
| Player.Ben => 13
| Player.Cindy => 19
| Player.Dave => 28
| Player.Ellen => 30

/-- Predicate to check if a player's scores contain a specific value -/
def containsScore (scores : PlayerScores) (n : DartScore) : Prop :=
  ∃ i, scores i = n

/-- Statement: Cindy is the only player who hit the region worth 9 points -/
theorem cindy_hit_nine :
  ∃! p : Player, ∃ scores : PlayerScores,
    totalScore scores = playerTotalScores p ∧
    containsScore scores ⟨9, by norm_num⟩ ∧
    p = Player.Cindy :=
by
  sorry

end NUMINAMATH_CALUDE_cindy_hit_nine_l1140_114054


namespace NUMINAMATH_CALUDE_eldest_boy_age_l1140_114023

/-- Given three boys whose ages are in proportion 3 : 5 : 7 and have an average age of 15 years,
    the age of the eldest boy is 21 years. -/
theorem eldest_boy_age (age1 age2 age3 : ℕ) : 
  age1 + age2 + age3 = 45 →  -- average age is 15
  ∃ (k : ℕ), age1 = 3 * k ∧ age2 = 5 * k ∧ age3 = 7 * k →  -- ages are in proportion 3 : 5 : 7
  age3 = 21 :=
by sorry

end NUMINAMATH_CALUDE_eldest_boy_age_l1140_114023


namespace NUMINAMATH_CALUDE_sum_digits_of_valid_hex_count_l1140_114071

/-- Represents a hexadecimal digit -/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Represents a hexadecimal number -/
def HexNumber := List HexDigit

/-- Converts a natural number to hexadecimal representation -/
def toHex (n : ℕ) : HexNumber :=
  sorry

/-- Checks if a hexadecimal number contains only numeric digits and doesn't start with 0 -/
def isValidHex (h : HexNumber) : Bool :=
  sorry

/-- Counts valid hexadecimal numbers in the first n positive integers -/
def countValidHex (n : ℕ) : ℕ :=
  sorry

/-- Sums the digits of a natural number -/
def sumDigits (n : ℕ) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem sum_digits_of_valid_hex_count :
  sumDigits (countValidHex 2000) = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_digits_of_valid_hex_count_l1140_114071


namespace NUMINAMATH_CALUDE_max_thursday_hours_l1140_114009

def max_video_game_hours (wednesday : ℝ) (friday : ℝ) (average : ℝ) : Prop :=
  ∃ thursday : ℝ,
    wednesday = 2 ∧
    friday > wednesday + 3 ∧
    average = 3 ∧
    (wednesday + thursday + friday) / 3 = average ∧
    thursday = 2

theorem max_thursday_hours :
  max_video_game_hours 2 5 3 :=
sorry

end NUMINAMATH_CALUDE_max_thursday_hours_l1140_114009


namespace NUMINAMATH_CALUDE_wrong_value_correction_l1140_114015

theorem wrong_value_correction (n : ℕ) (initial_mean correct_mean wrong_value : ℝ) 
  (h1 : n = 20)
  (h2 : initial_mean = 150)
  (h3 : wrong_value = 135)
  (h4 : correct_mean = 151.25) :
  (n : ℝ) * correct_mean - ((n : ℝ) * initial_mean - wrong_value) = 160 := by
  sorry

end NUMINAMATH_CALUDE_wrong_value_correction_l1140_114015


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1140_114094

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  1 / x + 4 / y ≥ 9 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 4 / y = 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1140_114094


namespace NUMINAMATH_CALUDE_largest_n_multiple_of_5_l1140_114039

def is_multiple_of_5 (n : ℕ) : Prop :=
  ∃ k : ℤ, 7 * (n - 3)^7 - 2 * n^3 + 21 * n - 36 = 5 * k

theorem largest_n_multiple_of_5 :
  ∀ n : ℕ, n < 100000 → is_multiple_of_5 n → n ≤ 99998 ∧
  is_multiple_of_5 99998 ∧
  99998 < 100000 :=
sorry

end NUMINAMATH_CALUDE_largest_n_multiple_of_5_l1140_114039


namespace NUMINAMATH_CALUDE_minimum_value_of_x_l1140_114033

theorem minimum_value_of_x (x : ℝ) 
  (h_pos : x > 0) 
  (h_log : Real.log x ≥ Real.log 3 + Real.log (Real.sqrt x)) : 
  x ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_x_l1140_114033


namespace NUMINAMATH_CALUDE_sundae_cost_theorem_l1140_114045

def sundae_cost (monday_sundaes tuesday_sundaes : ℕ)
  (monday_mms monday_gummies monday_marshmallows : ℕ)
  (tuesday_mms tuesday_gummies tuesday_marshmallows : ℕ)
  (mms_per_pack gummies_per_pack marshmallows_per_pack : ℕ)
  (mms_pack_cost gummies_pack_cost marshmallows_pack_cost : ℚ) : ℚ :=
  let total_mms := monday_sundaes * monday_mms + tuesday_sundaes * tuesday_mms
  let total_gummies := monday_sundaes * monday_gummies + tuesday_sundaes * tuesday_gummies
  let total_marshmallows := monday_sundaes * monday_marshmallows + tuesday_sundaes * tuesday_marshmallows
  let mms_packs := (total_mms + mms_per_pack - 1) / mms_per_pack
  let gummies_packs := (total_gummies + gummies_per_pack - 1) / gummies_per_pack
  let marshmallows_packs := (total_marshmallows + marshmallows_per_pack - 1) / marshmallows_per_pack
  mms_packs * mms_pack_cost + gummies_packs * gummies_pack_cost + marshmallows_packs * marshmallows_pack_cost

theorem sundae_cost_theorem :
  sundae_cost 40 20 6 4 8 10 5 12 40 30 50 2 (3/2) 1 = 95/2 :=
by sorry

end NUMINAMATH_CALUDE_sundae_cost_theorem_l1140_114045


namespace NUMINAMATH_CALUDE_complement_A_union_B_l1140_114084

def A : Set Int := {x | ∃ k, x = 3*k + 1}
def B : Set Int := {x | ∃ k, x = 3*k + 2}
def U : Set Int := Set.univ

theorem complement_A_union_B : 
  (U \ (A ∪ B)) = {x : Int | ∃ k, x = 3*k} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_l1140_114084


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l1140_114052

/-- Given a > 0 and the terminal side of angle α passes through point P(-3a, 4a),
    prove that sin α + 2cos α = -2/5 -/
theorem sin_plus_two_cos_alpha (a : ℝ) (α : ℝ) (h1 : a > 0) 
    (h2 : ∃ (t : ℝ), t > 0 ∧ -3 * a = t * Real.cos α ∧ 4 * a = t * Real.sin α) : 
    Real.sin α + 2 * Real.cos α = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l1140_114052


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1140_114082

-- Define sets A and B
def A : Set ℝ := {y | ∃ x, 0 ≤ x ∧ x ≤ 3 ∧ y = (x - 1)^2 + 1}
def B : Set ℝ := {y | ∃ x, 1 ≤ x ∧ x ≤ 3 ∧ y = x^2 + 1}

-- Define set difference
def setDifference (X Y : Set ℝ) : Set ℝ := {x | x ∈ X ∧ x ∉ Y}

-- Define symmetric difference
def symmetricDifference (X Y : Set ℝ) : Set ℝ := 
  (setDifference X Y) ∪ (setDifference Y X)

-- Theorem statement
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {y | (1 ≤ y ∧ y < 2) ∨ (5 < y ∧ y ≤ 10)} := by
  sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1140_114082


namespace NUMINAMATH_CALUDE_fraction_sum_inequality_l1140_114046

theorem fraction_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^3 / (x^3 + 2*y^2*z)) + (y^3 / (y^3 + 2*z^2*x)) + (z^3 / (z^3 + 2*x^2*y)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_inequality_l1140_114046


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1140_114064

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n ∧ a n > 0

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_positive_geometric_sequence a)
  (h_sum : a 2 + a 6 = 10)
  (h_prod : a 4 * a 8 = 64) :
  a 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1140_114064


namespace NUMINAMATH_CALUDE_min_abc_value_l1140_114001

-- Define the quadratic function P(x)
def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for P(x) having exactly one real root
def has_one_root (a b c : ℝ) : Prop := ∃! x : ℝ, P a b c x = 0

-- Define the condition for P(P(P(x))) having exactly three different real roots
def triple_P_has_three_roots (a b c : ℝ) : Prop :=
  ∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    P a b c (P a b c (P a b c x)) = 0 ∧
    P a b c (P a b c (P a b c y)) = 0 ∧
    P a b c (P a b c (P a b c z)) = 0

-- State the theorem
theorem min_abc_value (a b c : ℝ) :
  has_one_root a b c →
  triple_P_has_three_roots a b c →
  ∀ a' b' c' : ℝ, has_one_root a' b' c' → triple_P_has_three_roots a' b' c' →
    a * b * c ≤ a' * b' * c' →
    a * b * c = -2 :=
sorry

end NUMINAMATH_CALUDE_min_abc_value_l1140_114001


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l1140_114073

theorem geometric_progression_first_term 
  (S : ℝ) 
  (sum_first_two : ℝ) 
  (h1 : S = 6) 
  (h2 : sum_first_two = 9/2) : 
  ∃ a : ℝ, (a = 9 ∨ a = 3) ∧ 
  ∃ r : ℝ, (r = 1/2 ∨ r = -1/2) ∧ 
  S = a / (1 - r) ∧ 
  sum_first_two = a + a * r :=
sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l1140_114073


namespace NUMINAMATH_CALUDE_division_remainder_3005_95_l1140_114088

theorem division_remainder_3005_95 : 3005 % 95 = 60 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_3005_95_l1140_114088


namespace NUMINAMATH_CALUDE_at_least_one_less_than_two_l1140_114066

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  min ((1 + x) / y) ((1 + y) / x) < 2 := by
sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_two_l1140_114066


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l1140_114074

/-- The radius of the inscribed circle in a triangle with sides 9, 10, and 11 is 2√2 -/
theorem inscribed_circle_radius (a b c : ℝ) (h_a : a = 9) (h_b : b = 10) (h_c : c = 11) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l1140_114074


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1140_114060

theorem polynomial_simplification (s : ℝ) : 
  (2 * s^2 + 5 * s - 3) - (2 * s^2 + 9 * s - 6) = -4 * s + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1140_114060


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l1140_114083

theorem nested_fraction_evaluation :
  1 / (2 - 1 / (2 - 1 / (2 - 1 / 3))) = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l1140_114083


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1140_114017

theorem solve_exponential_equation :
  ∃ n : ℕ, (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n = (729 : ℝ)^4 ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1140_114017


namespace NUMINAMATH_CALUDE_u_2002_equals_2_l1140_114063

-- Define the function g
def g : ℕ → ℕ
| 1 => 5
| 2 => 3
| 3 => 4
| 4 => 2
| 5 => 1
| _ => 0  -- For completeness, though not used in the problem

-- Define the sequence u
def u : ℕ → ℕ
| 0 => 3
| (n + 1) => g (u n)

-- State the theorem
theorem u_2002_equals_2 : u 2002 = 2 := by
  sorry

end NUMINAMATH_CALUDE_u_2002_equals_2_l1140_114063


namespace NUMINAMATH_CALUDE_election_winning_margin_l1140_114027

/-- Represents an election with two candidates -/
structure Election :=
  (total_votes : ℕ)
  (winner_votes : ℕ)
  (winner_percentage : ℚ)

/-- Calculates the number of votes the winner won by -/
def winning_margin (e : Election) : ℕ :=
  e.winner_votes - (e.total_votes - e.winner_votes)

/-- Theorem stating the winning margin for the given election scenario -/
theorem election_winning_margin :
  ∃ (e : Election),
    e.winner_percentage = 62 / 100 ∧
    e.winner_votes = 899 ∧
    winning_margin e = 348 := by
  sorry

end NUMINAMATH_CALUDE_election_winning_margin_l1140_114027


namespace NUMINAMATH_CALUDE_september_reading_goal_l1140_114003

def total_pages_read (total_days : ℕ) (non_reading_days : ℕ) (special_day_pages : ℕ) (regular_daily_pages : ℕ) : ℕ :=
  let reading_days := total_days - non_reading_days
  let regular_reading_days := reading_days - 1
  regular_reading_days * regular_daily_pages + special_day_pages

theorem september_reading_goal :
  total_pages_read 30 4 100 20 = 600 := by
  sorry

end NUMINAMATH_CALUDE_september_reading_goal_l1140_114003


namespace NUMINAMATH_CALUDE_same_terminal_side_as_60_degrees_l1140_114075

def has_same_terminal_side (α : ℤ) : Prop :=
  ∃ k : ℤ, α = k * 360 + 60

theorem same_terminal_side_as_60_degrees :
  has_same_terminal_side (-300) ∧
  ¬has_same_terminal_side (-60) ∧
  ¬has_same_terminal_side 600 ∧
  ¬has_same_terminal_side 1380 :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_60_degrees_l1140_114075


namespace NUMINAMATH_CALUDE_pencils_removed_l1140_114069

/-- Given a jar of pencils, prove that the number of pencils removed is correct. -/
theorem pencils_removed (original : ℕ) (remaining : ℕ) (removed : ℕ) : 
  original = 87 → remaining = 83 → removed = original - remaining → removed = 4 := by
  sorry

end NUMINAMATH_CALUDE_pencils_removed_l1140_114069


namespace NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l1140_114068

-- Define a function to convert octal to decimal
def octal_to_decimal (octal : List Nat) : Nat :=
  octal.enum.foldl (fun acc (i, digit) => acc + digit * (8 ^ i)) 0

-- Theorem statement
theorem octal_123_equals_decimal_83 :
  octal_to_decimal [3, 2, 1] = 83 := by
  sorry

end NUMINAMATH_CALUDE_octal_123_equals_decimal_83_l1140_114068


namespace NUMINAMATH_CALUDE_final_score_eq_initial_minus_one_l1140_114021

/-- A scoring system where 1 point is deducted for a missing answer -/
structure ScoringSystem where
  initial_score : ℝ
  missing_answer_deduction : ℝ := 1

/-- The final score after deducting for a missing answer -/
def final_score (s : ScoringSystem) : ℝ :=
  s.initial_score - s.missing_answer_deduction

/-- Theorem stating that the final score is equal to the initial score minus 1 -/
theorem final_score_eq_initial_minus_one (s : ScoringSystem) :
  final_score s = s.initial_score - 1 := by
  sorry

#check final_score_eq_initial_minus_one

end NUMINAMATH_CALUDE_final_score_eq_initial_minus_one_l1140_114021


namespace NUMINAMATH_CALUDE_constant_value_l1140_114099

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define points
def P : ℝ × ℝ := (-2, 0)
def Q : ℝ × ℝ := (-2, -1)

-- Define the line l
def line_l (n : ℝ) (x y : ℝ) : Prop := x = n * (y + 1) - 2

-- Define the intersection points A and B
def A (n : ℝ) : ℝ × ℝ := sorry
def B (n : ℝ) : ℝ × ℝ := sorry

-- Define points C and D
def C (n : ℝ) : ℝ × ℝ := sorry
def D (n : ℝ) : ℝ × ℝ := sorry

-- Define the distances |QC| and |QD|
def QC (n : ℝ) : ℝ := sorry
def QD (n : ℝ) : ℝ := sorry

-- The main theorem
theorem constant_value (n : ℝ) :
  ellipse (A n).1 (A n).2 ∧ 
  ellipse (B n).1 (B n).2 ∧ 
  (A n).2 < 0 ∧ 
  (B n).2 < 0 ∧
  line_l n (A n).1 (A n).2 ∧
  line_l n (B n).1 (B n).2 →
  QC n + QD n - QC n * QD n = 0 :=
sorry

end

end NUMINAMATH_CALUDE_constant_value_l1140_114099


namespace NUMINAMATH_CALUDE_khali_snow_volume_l1140_114050

/-- The volume of snow on a rectangular sidewalk -/
def snow_volume (length width depth : ℝ) : ℝ := length * width * depth

/-- Theorem: The volume of snow on Khali's sidewalk is 20 cubic feet -/
theorem khali_snow_volume :
  snow_volume 20 2 (1/2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_khali_snow_volume_l1140_114050


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1140_114059

theorem solution_set_inequality (x : ℝ) : 
  (Set.Icc (-3 : ℝ) 6 : Set ℝ) = {x | (x + 3) * (6 - x) ≥ 0} := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1140_114059


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l1140_114006

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line L
def line_L (x y : ℝ) : Prop := x - y = 2

-- Theorem statement
theorem circle_and_line_properties :
  -- The radius of circle C is 1
  (∃ (a b : ℝ), ∀ (x y : ℝ), circle_C x y ↔ (x - a)^2 + (y - b)^2 = 1) ∧
  -- The distance from the center of C to line L is √2
  (∃ (a b : ℝ), (a - b - 2) / Real.sqrt 2 = Real.sqrt 2) ∧
  -- The minimum distance from a point on C to line L is √2 - 1
  (∃ (x y : ℝ), circle_C x y ∧ 
    (∀ (x' y' : ℝ), circle_C x' y' →
      |x' - y' - 2| / Real.sqrt 2 ≥ Real.sqrt 2 - 1) ∧
    |x - y - 2| / Real.sqrt 2 = Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_circle_and_line_properties_l1140_114006


namespace NUMINAMATH_CALUDE_coffee_buyers_fraction_l1140_114029

theorem coffee_buyers_fraction (total : ℕ) (non_coffee : ℕ) 
  (h1 : total = 25) (h2 : non_coffee = 10) : 
  (total - non_coffee : ℚ) / total = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_coffee_buyers_fraction_l1140_114029


namespace NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1140_114056

theorem consecutive_odd_integers_sum (a b c : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  b = a + 2 →               -- b is the next consecutive odd integer
  c = b + 2 →               -- c is the next consecutive odd integer after b
  a + c = 150 →             -- sum of first and third is 150
  a + b + c = 225 :=        -- sum of all three is 225
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_integers_sum_l1140_114056


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1140_114042

/-- Given a > 0 and a ≠ 1, prove that (-2, 2) is a fixed point of f(x) = a^(x+2) + 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(x + 2) + 1
  f (-2) = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1140_114042


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1140_114012

/-- Given a hyperbola with real axis length 16 and imaginary axis length 12, its eccentricity is 5/4 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a = 8) (h2 : b = 6) : 
  let c := Real.sqrt (a^2 + b^2)
  c / a = 5/4 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1140_114012


namespace NUMINAMATH_CALUDE_stock_price_change_l1140_114085

theorem stock_price_change (initial_price : ℝ) (h_pos : initial_price > 0) : 
  let price_after_decrease := initial_price * (1 - 0.08)
  let final_price := price_after_decrease * (1 + 0.10)
  let net_change_percentage := (final_price - initial_price) / initial_price * 100
  net_change_percentage = 1.2 := by
sorry

end NUMINAMATH_CALUDE_stock_price_change_l1140_114085


namespace NUMINAMATH_CALUDE_coin_distribution_l1140_114072

theorem coin_distribution (total : ℕ) (ways : ℕ) 
  (h_total : total = 1512)
  (h_ways : ways = 1512)
  (h_denominations : ∃ (c₂ c₅ c₁₀ c₂₀ c₅₀ c₁₀₀ c₂₀₀ : ℕ),
    (c₂ ≥ 1 ∧ c₅ ≥ 1 ∧ c₁₀ ≥ 1 ∧ c₂₀ ≥ 1 ∧ c₅₀ ≥ 1 ∧ c₁₀₀ ≥ 1 ∧ c₂₀₀ ≥ 1) ∧
    (2 * c₂ + 5 * c₅ + 10 * c₁₀ + 20 * c₂₀ + 50 * c₅₀ + 100 * c₁₀₀ + 200 * c₂₀₀ = total) ∧
    ((c₂ + 1) * (c₅ + 1) * (c₁₀ + 1) * (c₂₀ + 1) * (c₅₀ + 1) * (c₁₀₀ + 1) * (c₂₀₀ + 1) = ways)) :
  ∃! (c₂ c₅ c₁₀ c₂₀ c₅₀ c₁₀₀ c₂₀₀ : ℕ),
    (c₂ = 1 ∧ c₅ = 2 ∧ c₁₀ = 1 ∧ c₂₀ = 2 ∧ c₅₀ = 1 ∧ c₁₀₀ = 2 ∧ c₂₀₀ = 6) ∧
    (2 * c₂ + 5 * c₅ + 10 * c₁₀ + 20 * c₂₀ + 50 * c₅₀ + 100 * c₁₀₀ + 200 * c₂₀₀ = total) ∧
    ((c₂ + 1) * (c₅ + 1) * (c₁₀ + 1) * (c₂₀ + 1) * (c₅₀ + 1) * (c₁₀₀ + 1) * (c₂₀₀ + 1) = ways) :=
by sorry

end NUMINAMATH_CALUDE_coin_distribution_l1140_114072


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_negative_one_l1140_114036

theorem sum_of_a_and_b_is_negative_one :
  ∀ (a b : ℝ) (S T : ℕ → ℝ),
  (∀ n, S n = 2^n + a) →  -- Sum of geometric sequence
  (∀ n, T n = n^2 - 2*n + b) →  -- Sum of arithmetic sequence
  a + b = -1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_negative_one_l1140_114036


namespace NUMINAMATH_CALUDE_ab_equals_one_l1140_114097

theorem ab_equals_one (θ : ℝ) (a b : ℝ) 
  (h1 : a * Real.sin θ + Real.cos θ = 1)
  (h2 : b * Real.sin θ - Real.cos θ = 1) :
  a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_ab_equals_one_l1140_114097


namespace NUMINAMATH_CALUDE_trapezoid_area_in_isosceles_triangle_l1140_114004

/-- Represents a triangle in a plane -/
structure Triangle where
  area : ℝ

/-- Represents a trapezoid in a plane -/
structure Trapezoid where
  area : ℝ

/-- The main theorem statement -/
theorem trapezoid_area_in_isosceles_triangle 
  (PQR : Triangle) 
  (smallest : Triangle)
  (RSQT : Trapezoid) :
  PQR.area = 72 ∧ 
  smallest.area = 2 ∧ 
  (∃ n : ℕ, n = 9 ∧ n * smallest.area = PQR.area) ∧
  (∃ m : ℕ, m = 3 ∧ m * smallest.area ≤ RSQT.area) →
  RSQT.area = 39 := by
sorry

end NUMINAMATH_CALUDE_trapezoid_area_in_isosceles_triangle_l1140_114004


namespace NUMINAMATH_CALUDE_matrix_sum_proof_l1140_114051

theorem matrix_sum_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 5]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![-6, 8; -3, 7]
  A + B = !![-2, 5; -1, 12] := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_proof_l1140_114051


namespace NUMINAMATH_CALUDE_correct_ranking_l1140_114055

-- Define the cities
inductive City
| Dover
| Eden
| Fairview

-- Define the growth rate comparison relation
def higherGrowthRate : City → City → Prop := sorry

-- Define the statements
def statement1 : Prop := higherGrowthRate City.Dover City.Eden ∧ higherGrowthRate City.Dover City.Fairview
def statement2 : Prop := ¬(higherGrowthRate City.Eden City.Dover ∧ higherGrowthRate City.Eden City.Fairview)
def statement3 : Prop := ¬(higherGrowthRate City.Dover City.Fairview ∧ higherGrowthRate City.Eden City.Fairview)

-- Theorem stating the correct ranking
theorem correct_ranking :
  (statement1 ∨ statement2 ∨ statement3) ∧
  (statement1 → ¬statement2 ∧ ¬statement3) ∧
  (statement2 → ¬statement1 ∧ ¬statement3) ∧
  (statement3 → ¬statement1 ∧ ¬statement2) →
  higherGrowthRate City.Eden City.Dover ∧
  higherGrowthRate City.Dover City.Fairview :=
sorry

end NUMINAMATH_CALUDE_correct_ranking_l1140_114055


namespace NUMINAMATH_CALUDE_initial_velocity_is_three_l1140_114007

/-- The displacement function of an object moving in a straight line -/
def displacement (t : ℝ) : ℝ := 3 * t - t^2

/-- The velocity function of the object -/
def velocity (t : ℝ) : ℝ := 3 - 2 * t

/-- The theorem stating that the initial velocity is 3 -/
theorem initial_velocity_is_three : velocity 0 = 3 := by sorry

end NUMINAMATH_CALUDE_initial_velocity_is_three_l1140_114007


namespace NUMINAMATH_CALUDE_min_value_expression_l1140_114025

theorem min_value_expression (a b c d : ℝ) (h1 : b > d) (h2 : d > c) (h3 : c > a) (h4 : b ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (d - c)^2 + (c - a)^2) / b^2 ≥ 9 ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ), b₀ > d₀ ∧ d₀ > c₀ ∧ c₀ > a₀ ∧ b₀ ≠ 0 ∧
    ((a₀ + b₀)^2 + (b₀ - c₀)^2 + (d₀ - c₀)^2 + (c₀ - a₀)^2) / b₀^2 = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1140_114025


namespace NUMINAMATH_CALUDE_sixth_term_of_sequence_l1140_114087

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℝ) (d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

-- State the theorem
theorem sixth_term_of_sequence (a d : ℝ) :
  arithmetic_sequence a d 2 = 14 ∧
  arithmetic_sequence a d 4 = 32 →
  arithmetic_sequence a d 6 = 50 := by
  sorry


end NUMINAMATH_CALUDE_sixth_term_of_sequence_l1140_114087


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1140_114058

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) ∧ 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1140_114058


namespace NUMINAMATH_CALUDE_circles_are_disjoint_l1140_114065

-- Define the circles
def C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
def C₂ : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + (p.2 + 2)^2 = 1}

-- Define the centers and radii
def center₁ : ℝ × ℝ := (0, 0)
def center₂ : ℝ × ℝ := (2, -2)
def radius₁ : ℝ := 1
def radius₂ : ℝ := 1

-- Theorem statement
theorem circles_are_disjoint : 
  Real.sqrt ((center₂.1 - center₁.1)^2 + (center₂.2 - center₁.2)^2) > radius₁ + radius₂ := by
  sorry

end NUMINAMATH_CALUDE_circles_are_disjoint_l1140_114065


namespace NUMINAMATH_CALUDE_two_zeros_cubic_l1140_114090

def f (c : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + c

theorem two_zeros_cubic (c : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f c x = 0 ∧ f c y = 0 ∧ ∀ z : ℝ, f c z = 0 → z = x ∨ z = y) ↔ c = -2 ∨ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_zeros_cubic_l1140_114090


namespace NUMINAMATH_CALUDE_theater_audience_l1140_114070

/-- Proves the number of children in the audience given theater conditions -/
theorem theater_audience (total_seats : ℕ) (adult_price child_price : ℚ) (total_income : ℚ) 
  (h_seats : total_seats = 200)
  (h_adult_price : adult_price = 3)
  (h_child_price : child_price = (3/2))
  (h_total_income : total_income = 510) :
  ∃ (adults children : ℕ), 
    adults + children = total_seats ∧ 
    adult_price * adults + child_price * children = total_income ∧
    children = 60 := by
  sorry

end NUMINAMATH_CALUDE_theater_audience_l1140_114070


namespace NUMINAMATH_CALUDE_striped_shirts_difference_l1140_114049

theorem striped_shirts_difference (total : ℕ) (striped_ratio : ℚ) (checkered_ratio : ℚ) 
  (h_total : total = 120)
  (h_striped : striped_ratio = 3/5)
  (h_checkered : checkered_ratio = 1/4)
  (h_shorts_plain : ∃ (plain : ℕ) (shorts : ℕ), 
    plain = total - (striped_ratio * total).num - (checkered_ratio * total).num ∧
    shorts + 10 = plain) :
  ∃ (striped : ℕ) (shorts : ℕ),
    striped = (striped_ratio * total).num ∧
    striped - shorts = 44 :=
sorry

end NUMINAMATH_CALUDE_striped_shirts_difference_l1140_114049


namespace NUMINAMATH_CALUDE_daily_sales_volume_selling_price_for_profit_daily_sales_profit_and_max_l1140_114013

-- Define the variables and constants
variable (x : ℝ) -- Selling price in yuan
variable (y : ℝ) -- Daily sales volume in items
variable (w : ℝ) -- Daily sales profit in yuan

-- Define the given conditions
def cost_price : ℝ := 6
def min_price : ℝ := 6
def max_price : ℝ := 12
def base_price : ℝ := 8
def base_volume : ℝ := 200
def volume_change_rate : ℝ := 10

-- Theorem 1: Daily sales volume function
theorem daily_sales_volume : 
  ∀ x, min_price ≤ x ∧ x ≤ max_price → y = -volume_change_rate * x + (base_volume + volume_change_rate * base_price) :=
sorry

-- Theorem 2: Selling price for specific profit
theorem selling_price_for_profit (target_profit : ℝ) : 
  ∃ x, min_price ≤ x ∧ x ≤ max_price ∧ 
  (x - cost_price) * (-volume_change_rate * x + (base_volume + volume_change_rate * base_price)) = target_profit :=
sorry

-- Theorem 3: Daily sales profit function and maximum profit
theorem daily_sales_profit_and_max : 
  ∃ w_max : ℝ,
  (∀ x, min_price ≤ x ∧ x ≤ max_price → 
    w = -volume_change_rate * (x - 11)^2 + 1210) ∧
  (w_max = -volume_change_rate * (max_price - 11)^2 + 1210) ∧
  (∀ x, min_price ≤ x ∧ x ≤ max_price → w ≤ w_max) :=
sorry

end NUMINAMATH_CALUDE_daily_sales_volume_selling_price_for_profit_daily_sales_profit_and_max_l1140_114013


namespace NUMINAMATH_CALUDE_area_of_closed_region_l1140_114002

-- Define the functions
def f₀ (x : ℝ) := |x|
def f₁ (x : ℝ) := |f₀ x - 1|
def f₂ (x : ℝ) := |f₁ x - 2|

-- Define the area function
noncomputable def area_under_curve (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, f x

-- Theorem statement
theorem area_of_closed_region :
  area_under_curve f₂ (-3) 3 = 7 := by
  sorry

end NUMINAMATH_CALUDE_area_of_closed_region_l1140_114002


namespace NUMINAMATH_CALUDE_twelfth_even_multiple_of_5_l1140_114048

-- Define a function that represents the nth positive integer that is both even and a multiple of 5
def evenMultipleOf5 (n : ℕ) : ℕ := 10 * n

-- State the theorem
theorem twelfth_even_multiple_of_5 : evenMultipleOf5 12 = 120 := by sorry

end NUMINAMATH_CALUDE_twelfth_even_multiple_of_5_l1140_114048


namespace NUMINAMATH_CALUDE_expected_value_of_strategic_die_rolling_l1140_114053

/-- Represents a 6-sided die -/
def Die := Fin 6

/-- The strategy for re-rolling -/
def rerollStrategy (roll : Die) : Bool :=
  roll.val < 4

/-- The expected value of a single roll of a 6-sided die -/
def singleRollExpectedValue : ℚ := 7/2

/-- The expected value after applying the re-roll strategy once -/
def strategicRollExpectedValue : ℚ := 17/4

/-- The final expected value after up to two re-rolls -/
def finalExpectedValue : ℚ := 17/4

theorem expected_value_of_strategic_die_rolling :
  finalExpectedValue = 17/4 := by sorry

end NUMINAMATH_CALUDE_expected_value_of_strategic_die_rolling_l1140_114053


namespace NUMINAMATH_CALUDE_steve_markers_l1140_114077

/-- Given the number of markers for Alia, Austin, and Steve, 
    where Alia has 2 times as many markers as Austin, 
    Austin has one-third as many markers as Steve, 
    and Alia has 40 markers, prove that Steve has 60 markers. -/
theorem steve_markers (alia austin steve : ℕ) 
  (h1 : alia = 2 * austin) 
  (h2 : austin = steve / 3)
  (h3 : alia = 40) : 
  steve = 60 := by sorry

end NUMINAMATH_CALUDE_steve_markers_l1140_114077


namespace NUMINAMATH_CALUDE_travel_time_calculation_l1140_114032

/-- Given a person traveling at a constant speed for a certain distance,
    prove that the time taken is equal to the distance divided by the speed. -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (h1 : speed > 0) :
  let time := distance / speed
  speed = 20 ∧ distance = 50 → time = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l1140_114032


namespace NUMINAMATH_CALUDE_original_professors_count_l1140_114000

/-- The original number of professors in the DVEU Department of Mathematical Modeling. -/
def original_professors : ℕ := 5

/-- The number of failing grades given in the first academic year. -/
def first_year_grades : ℕ := 6480

/-- The number of failing grades given in the second academic year. -/
def second_year_grades : ℕ := 11200

/-- The increase in the number of professors in the second year. -/
def professor_increase : ℕ := 3

theorem original_professors_count :
  (first_year_grades % original_professors = 0) ∧
  (second_year_grades % (original_professors + professor_increase) = 0) ∧
  (first_year_grades / original_professors < second_year_grades / (original_professors + professor_increase)) ∧
  (∀ p : ℕ, p < original_professors →
    (first_year_grades % p = 0 ∧ 
     second_year_grades % (p + professor_increase) = 0) → 
    (first_year_grades / p ≥ second_year_grades / (p + professor_increase))) :=
by sorry

end NUMINAMATH_CALUDE_original_professors_count_l1140_114000


namespace NUMINAMATH_CALUDE_equation_solution_l1140_114041

theorem equation_solution : 
  ∀ x : ℝ, x ≠ -3 → 
  ((7 * x^2 - 3) / (x + 3) - 3 / (x + 3) = 1 / (x + 3)) ↔ 
  (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1140_114041
