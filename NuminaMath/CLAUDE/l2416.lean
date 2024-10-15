import Mathlib

namespace NUMINAMATH_CALUDE_f_shifted_is_even_f_monotonicity_f_satisfies_properties_l2416_241621

-- Define the function f(x) = (x-2)^2
def f (x : ℝ) : ℝ := (x - 2)^2

-- Property 1: f(x+2) is an even function
theorem f_shifted_is_even : ∀ x : ℝ, f (x + 2) = f (-x + 2) := by sorry

-- Property 2: f(x) is decreasing on (-∞, 2) and increasing on (2, +∞)
theorem f_monotonicity :
  (∀ x y : ℝ, x < y → y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x → x < y → f x < f y) := by sorry

-- Theorem combining both properties
theorem f_satisfies_properties : 
  (∀ x : ℝ, f (x + 2) = f (-x + 2)) ∧
  (∀ x y : ℝ, x < y → y < 2 → f y < f x) ∧
  (∀ x y : ℝ, 2 < x → x < y → f x < f y) := by sorry

end NUMINAMATH_CALUDE_f_shifted_is_even_f_monotonicity_f_satisfies_properties_l2416_241621


namespace NUMINAMATH_CALUDE_one_fourth_of_8_4_l2416_241620

theorem one_fourth_of_8_4 : 
  ∃ (n d : ℕ), n ≠ 0 ∧ d ≠ 0 ∧ (8.4 / 4 : ℚ) = n / d ∧ Nat.gcd n d = 1 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_8_4_l2416_241620


namespace NUMINAMATH_CALUDE_f_of_x_minus_one_l2416_241690

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem f_of_x_minus_one (x : ℝ) : f (x - 1) = x^2 - 4*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_f_of_x_minus_one_l2416_241690


namespace NUMINAMATH_CALUDE_private_pilot_course_cost_l2416_241648

/-- The cost of a private pilot course -/
theorem private_pilot_course_cost :
  ∀ (flight_cost ground_cost total_cost : ℕ),
    flight_cost = 950 →
    ground_cost = 325 →
    flight_cost = ground_cost + 625 →
    total_cost = flight_cost + ground_cost →
    total_cost = 1275 := by
  sorry

end NUMINAMATH_CALUDE_private_pilot_course_cost_l2416_241648


namespace NUMINAMATH_CALUDE_new_person_weight_l2416_241667

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 10 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 100 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l2416_241667


namespace NUMINAMATH_CALUDE_product_of_solutions_l2416_241616

theorem product_of_solutions (x : ℝ) : 
  (∀ x, -49 = -2*x^2 + 6*x) → 
  (∃ α β : ℝ, (α * β = -24.5) ∧ (-49 = -2*α^2 + 6*α) ∧ (-49 = -2*β^2 + 6*β)) :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_l2416_241616


namespace NUMINAMATH_CALUDE_average_sale_l2416_241614

def sales : List ℕ := [5420, 5660, 6200, 6350, 6500]
def projected_sale : ℕ := 6470

theorem average_sale :
  (sales.sum + projected_sale) / (sales.length + 1) = 6100 := by
  sorry

end NUMINAMATH_CALUDE_average_sale_l2416_241614


namespace NUMINAMATH_CALUDE_divisibility_implication_l2416_241642

theorem divisibility_implication (m : ℕ+) (h : 39 ∣ m^2) : 39 ∣ m := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implication_l2416_241642


namespace NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l2416_241672

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem neither_sufficient_nor_necessary (a b : V) : 
  ¬(∀ a b : V, ‖a‖ = ‖b‖ → ‖a + b‖ = ‖a - b‖) ∧ 
  ¬(∀ a b : V, ‖a + b‖ = ‖a - b‖ → ‖a‖ = ‖b‖) := by
  sorry

end NUMINAMATH_CALUDE_neither_sufficient_nor_necessary_l2416_241672


namespace NUMINAMATH_CALUDE_triangle_area_isosceles_l2416_241684

/-- The area of a triangle with two sides of length 30 and one side of length 40 -/
theorem triangle_area_isosceles (a b c : ℝ) (h1 : a = 30) (h2 : b = 30) (h3 : c = 40) : 
  ∃ area : ℝ, abs (area - Real.sqrt (50 * (50 - a) * (50 - b) * (50 - c))) < 0.01 ∧ 
  446.99 < area ∧ area < 447.01 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_isosceles_l2416_241684


namespace NUMINAMATH_CALUDE_same_solution_equations_l2416_241634

theorem same_solution_equations (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 6 = 0 ∧ c * x - 15 = -3) → c = -6 := by
sorry

end NUMINAMATH_CALUDE_same_solution_equations_l2416_241634


namespace NUMINAMATH_CALUDE_shelter_cat_count_l2416_241678

/-- Calculates the total number of cats and kittens in an animal shelter --/
theorem shelter_cat_count (total_adults : ℕ) (female_ratio : ℚ) (litter_ratio : ℚ) (avg_kittens : ℕ) : 
  total_adults = 100 →
  female_ratio = 1/2 →
  litter_ratio = 1/2 →
  avg_kittens = 4 →
  total_adults + (total_adults * female_ratio * litter_ratio * avg_kittens) = 200 := by
sorry

end NUMINAMATH_CALUDE_shelter_cat_count_l2416_241678


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l2416_241651

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 120 → 
  a * b + b * c + c * d ≤ 3600 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l2416_241651


namespace NUMINAMATH_CALUDE_ellipse_properties_l2416_241658

-- Define the ellipse
def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

-- Define the foci
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

-- Define eccentricity
def e : ℝ := 0.8

-- Define dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector from a point to another
def vector_to (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

theorem ellipse_properties :
  ∃ (a b : ℝ),
    -- Standard equation of the ellipse
    (a = 5 ∧ b = 3) ∧
    -- Existence of point P
    ∃ (P : ℝ × ℝ),
      P ∈ Ellipse a b ∧
      dot_product (vector_to F₁ P) (vector_to F₂ P) = 0 ∧
      -- Coordinates of point P
      ((P.1 = 5 * Real.sqrt 7 / 4 ∧ P.2 = 9 / 4) ∨
       (P.1 = -5 * Real.sqrt 7 / 4 ∧ P.2 = 9 / 4) ∨
       (P.1 = 5 * Real.sqrt 7 / 4 ∧ P.2 = -9 / 4) ∨
       (P.1 = -5 * Real.sqrt 7 / 4 ∧ P.2 = -9 / 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l2416_241658


namespace NUMINAMATH_CALUDE_mean_goals_is_6_l2416_241695

/-- The number of players who scored 5 goals -/
def players_5 : ℕ := 4

/-- The number of players who scored 6 goals -/
def players_6 : ℕ := 3

/-- The number of players who scored 7 goals -/
def players_7 : ℕ := 2

/-- The number of players who scored 8 goals -/
def players_8 : ℕ := 1

/-- The total number of goals scored -/
def total_goals : ℕ := 5 * players_5 + 6 * players_6 + 7 * players_7 + 8 * players_8

/-- The total number of players -/
def total_players : ℕ := players_5 + players_6 + players_7 + players_8

/-- The mean number of goals scored -/
def mean_goals : ℚ := total_goals / total_players

theorem mean_goals_is_6 : mean_goals = 6 := by sorry

end NUMINAMATH_CALUDE_mean_goals_is_6_l2416_241695


namespace NUMINAMATH_CALUDE_stating_dodgeball_tournament_teams_l2416_241601

/-- Represents the total points scored in a dodgeball tournament. -/
def total_points : ℕ := 1151

/-- Points awarded for a win in the tournament. -/
def win_points : ℕ := 15

/-- Points awarded for a tie in the tournament. -/
def tie_points : ℕ := 11

/-- Points awarded for a loss in the tournament. -/
def loss_points : ℕ := 0

/-- The number of teams in the tournament. -/
def num_teams : ℕ := 12

/-- 
Theorem stating that given the tournament conditions, 
the number of teams must be 12.
-/
theorem dodgeball_tournament_teams : 
  ∀ n : ℕ, 
    (n * (n - 1) / 2) * win_points ≤ total_points ∧ 
    total_points ≤ (n * (n - 1) / 2) * (win_points + tie_points) / 2 →
    n = num_teams :=
by sorry

end NUMINAMATH_CALUDE_stating_dodgeball_tournament_teams_l2416_241601


namespace NUMINAMATH_CALUDE_unique_n_reaching_three_l2416_241698

def g (n : ℕ) : ℕ :=
  if n % 2 = 1 then n^2 + 3 else n / 2

def iterateG (n : ℕ) : ℕ → ℕ
  | 0 => n
  | k + 1 => g (iterateG n k)

theorem unique_n_reaching_three :
  ∃! n : ℕ, n ∈ Finset.range 100 ∧ ∃ k : ℕ, iterateG n k = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_n_reaching_three_l2416_241698


namespace NUMINAMATH_CALUDE_largest_three_digit_base_7_is_342_l2416_241615

/-- The largest decimal number represented by a three-digit base-7 number -/
def largest_three_digit_base_7 : ℕ := 342

/-- The base of the number system -/
def base : ℕ := 7

/-- The number of digits -/
def num_digits : ℕ := 3

/-- Theorem: The largest decimal number represented by a three-digit base-7 number is 342 -/
theorem largest_three_digit_base_7_is_342 :
  largest_three_digit_base_7 = (base ^ num_digits - 1) := by sorry

end NUMINAMATH_CALUDE_largest_three_digit_base_7_is_342_l2416_241615


namespace NUMINAMATH_CALUDE_factorization_equality_l2416_241641

theorem factorization_equality (a b : ℝ) : 6 * a^2 * b - 3 * a * b = 3 * a * b * (2 * a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2416_241641


namespace NUMINAMATH_CALUDE_present_difference_l2416_241604

/-- The number of presents Santana buys for her brothers in a year -/
def presentCount : ℕ → ℕ
| 1 => 3  -- March (first half)
| 2 => 1  -- October (second half)
| 3 => 1  -- November (second half)
| 4 => 2  -- December (second half)
| _ => 0

/-- The total number of brothers Santana has -/
def totalBrothers : ℕ := 7

/-- The number of presents bought in the first half of the year -/
def firstHalfPresents : ℕ := presentCount 1

/-- The number of presents bought in the second half of the year -/
def secondHalfPresents : ℕ := presentCount 2 + presentCount 3 + presentCount 4 + totalBrothers

theorem present_difference : secondHalfPresents - firstHalfPresents = 8 := by
  sorry

end NUMINAMATH_CALUDE_present_difference_l2416_241604


namespace NUMINAMATH_CALUDE_binomial_coefficient_two_l2416_241637

theorem binomial_coefficient_two (n : ℕ+) : Nat.choose n.val 2 = n.val * (n.val - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_two_l2416_241637


namespace NUMINAMATH_CALUDE_increasing_cubic_function_a_range_l2416_241696

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x

-- State the theorem
theorem increasing_cubic_function_a_range :
  (∀ x y : ℝ, x < y → f a x < f a y) → -Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_increasing_cubic_function_a_range_l2416_241696


namespace NUMINAMATH_CALUDE_triangle_area_qin_jiushao_l2416_241645

theorem triangle_area_qin_jiushao (a b c : ℝ) (h₁ : a = Real.sqrt 2) (h₂ : b = Real.sqrt 3) (h₃ : c = 2) :
  let S := Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))
  S = Real.sqrt 23 / 4 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_qin_jiushao_l2416_241645


namespace NUMINAMATH_CALUDE_max_handshakes_l2416_241636

theorem max_handshakes (N : ℕ) (h1 : N > 4) : ∃ (max_shaken : ℕ),
  (∃ (not_shaken : Fin N → Prop),
    (∃ (a b : Fin N), a ≠ b ∧ not_shaken a ∧ not_shaken b ∧
      ∀ (x : Fin N), not_shaken x → (x = a ∨ x = b)) ∧
    (∀ (x : Fin N), ¬(not_shaken x) →
      ∀ (y : Fin N), y ≠ x → ∃ (shaken : Prop), shaken)) ∧
  max_shaken = N - 2 ∧
  ∀ (k : ℕ), k > max_shaken →
    ¬(∃ (not_shaken : Fin N → Prop),
      (∃ (a b : Fin N), a ≠ b ∧ not_shaken a ∧ not_shaken b ∧
        ∀ (x : Fin N), not_shaken x → (x = a ∨ x = b)) ∧
      (∀ (x : Fin N), ¬(not_shaken x) →
        ∀ (y : Fin N), y ≠ x → ∃ (shaken : Prop), shaken))
  := by sorry

end NUMINAMATH_CALUDE_max_handshakes_l2416_241636


namespace NUMINAMATH_CALUDE_markus_family_ages_l2416_241633

theorem markus_family_ages (grandson_age : ℕ) : 
  grandson_age > 0 →
  let son_age := 2 * grandson_age
  let markus_age := 2 * son_age
  grandson_age + son_age + markus_age = 140 →
  grandson_age = 20 := by
sorry

end NUMINAMATH_CALUDE_markus_family_ages_l2416_241633


namespace NUMINAMATH_CALUDE_min_time_circular_chain_no_faster_solution_l2416_241600

/-- Represents a chain piece with a certain number of links -/
structure ChainPiece where
  links : ℕ

/-- Represents the time required for chain operations -/
structure ChainOperations where
  cutTime : ℕ
  joinTime : ℕ

/-- Calculates the minimum time required to form a circular chain -/
def minTimeToCircularChain (pieces : List ChainPiece) (ops : ChainOperations) : ℕ :=
  sorry

/-- Theorem stating the minimum time to form a circular chain from given pieces -/
theorem min_time_circular_chain :
  let pieces := [
    ChainPiece.mk 10,
    ChainPiece.mk 10,
    ChainPiece.mk 8,
    ChainPiece.mk 8,
    ChainPiece.mk 5,
    ChainPiece.mk 2
  ]
  let ops := ChainOperations.mk 1 2
  minTimeToCircularChain pieces ops = 15 := by
  sorry

/-- Theorem stating that it's impossible to form the circular chain in less than 15 minutes -/
theorem no_faster_solution (t : ℕ) :
  let pieces := [
    ChainPiece.mk 10,
    ChainPiece.mk 10,
    ChainPiece.mk 8,
    ChainPiece.mk 8,
    ChainPiece.mk 5,
    ChainPiece.mk 2
  ]
  let ops := ChainOperations.mk 1 2
  t < 15 → minTimeToCircularChain pieces ops ≠ t := by
  sorry

end NUMINAMATH_CALUDE_min_time_circular_chain_no_faster_solution_l2416_241600


namespace NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2416_241640

open Complex

theorem fourteenth_root_of_unity : ∃ (n : ℕ) (h : n ≤ 13),
  (Complex.tan (Real.pi / 7) + Complex.I) / (Complex.tan (Real.pi / 7) - Complex.I) =
  Complex.exp (2 * Real.pi * (n : ℝ) * Complex.I / 14) :=
by sorry

end NUMINAMATH_CALUDE_fourteenth_root_of_unity_l2416_241640


namespace NUMINAMATH_CALUDE_triangle_side_length_l2416_241668

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the relationship between side lengths and angles in the given triangle -/
theorem triangle_side_length (t : Triangle) (h1 : t.a = 2) (h2 : t.B = 135 * π / 180)
    (h3 : (1/2) * t.a * t.c * Real.sin t.B = 4) : t.b = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2416_241668


namespace NUMINAMATH_CALUDE_transformed_sin_equals_cos_l2416_241675

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin x

theorem transformed_sin_equals_cos :
  ∀ x : ℝ, f x = g (2 * (x + π / 4)) :=
by
  sorry

end NUMINAMATH_CALUDE_transformed_sin_equals_cos_l2416_241675


namespace NUMINAMATH_CALUDE_students_taking_history_l2416_241627

theorem students_taking_history 
  (total_students : ℕ) 
  (statistics_students : ℕ)
  (history_or_statistics : ℕ)
  (history_not_statistics : ℕ)
  (h_total : total_students = 89)
  (h_statistics : statistics_students = 32)
  (h_history_or_stats : history_or_statistics = 59)
  (h_history_not_stats : history_not_statistics = 27) :
  ∃ history_students : ℕ, history_students = 54 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_history_l2416_241627


namespace NUMINAMATH_CALUDE_xiao_ming_distance_l2416_241619

/-- The distance between Xiao Ming's house and school -/
def distance : ℝ := 1500

/-- The original planned speed in meters per minute -/
def original_speed : ℝ := 200

/-- The reduced speed due to rain in meters per minute -/
def reduced_speed : ℝ := 120

/-- The additional time taken due to reduced speed in minutes -/
def additional_time : ℝ := 5

theorem xiao_ming_distance :
  distance = original_speed * (distance / reduced_speed - additional_time) :=
sorry

end NUMINAMATH_CALUDE_xiao_ming_distance_l2416_241619


namespace NUMINAMATH_CALUDE_range_of_a_l2416_241646

/-- The condition for two distinct real roots -/
def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + a = 0 ∧ y^2 - 2*y + a = 0

/-- The condition for a hyperbola -/
def is_hyperbola (a : ℝ) : Prop :=
  (a - 3) * (a + 1) < 0

/-- The main theorem -/
theorem range_of_a (a : ℝ) : 
  ¬(has_two_distinct_real_roots a ∨ is_hyperbola a) → a ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2416_241646


namespace NUMINAMATH_CALUDE_cookfire_logs_added_l2416_241639

/-- The number of logs added to a cookfire each hour, given the initial number of logs,
    burn rate, duration, and final number of logs. -/
def logsAddedPerHour (initialLogs burnRate duration finalLogs : ℕ) : ℕ :=
  let logsAfterBurning := initialLogs - burnRate * duration
  (finalLogs - logsAfterBurning + burnRate * (duration - 1)) / duration

theorem cookfire_logs_added (x : ℕ) :
  logsAddedPerHour 6 3 3 3 = 2 :=
sorry

end NUMINAMATH_CALUDE_cookfire_logs_added_l2416_241639


namespace NUMINAMATH_CALUDE_counterexample_exists_l2416_241659

theorem counterexample_exists (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) :
  ∃ b : ℝ, c * b^2 ≥ a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2416_241659


namespace NUMINAMATH_CALUDE_shaded_area_ratio_l2416_241655

/-- Given two squares ABCD and CEFG with the same side length sharing a common vertex C,
    the ratio of the shaded area to the area of square ABCD is 2 - √2 -/
theorem shaded_area_ratio (l : ℝ) (h : l > 0) : 
  let diagonal := l * Real.sqrt 2
  let small_side := diagonal - l
  let shaded_area := l^2 - 2 * (1/2 * small_side * l)
  shaded_area / l^2 = 2 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_ratio_l2416_241655


namespace NUMINAMATH_CALUDE_inequality_range_l2416_241670

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_range_l2416_241670


namespace NUMINAMATH_CALUDE_no_intersection_points_l2416_241653

/-- The number of intersection points between r = 3 cos θ and r = 6 sin θ is 0 -/
theorem no_intersection_points : ∀ θ : ℝ, 
  ¬∃ r : ℝ, (r = 3 * Real.cos θ ∧ r = 6 * Real.sin θ) :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_points_l2416_241653


namespace NUMINAMATH_CALUDE_subset_ratio_for_ten_elements_l2416_241665

theorem subset_ratio_for_ten_elements : 
  let n : ℕ := 10
  let k : ℕ := 3
  let total_subsets : ℕ := 2^n
  let three_element_subsets : ℕ := n.choose k
  (three_element_subsets : ℚ) / total_subsets = 15 / 128 := by
  sorry

end NUMINAMATH_CALUDE_subset_ratio_for_ten_elements_l2416_241665


namespace NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l2416_241603

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)

-- Define the perpendicular relation between planes
variable (planePerp : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicular_parallel 
  (m n : Line) (α β γ : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : α ≠ γ)
  (h4 : β ≠ γ)
  (h5 : perpendicular m β)
  (h6 : parallel m α) :
  planePerp α β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicular_parallel_l2416_241603


namespace NUMINAMATH_CALUDE_thelmas_tomato_slices_l2416_241630

def slices_per_meal : ℕ := 20
def people_to_feed : ℕ := 8
def tomatoes_needed : ℕ := 20

def slices_per_tomato : ℕ := (slices_per_meal * people_to_feed) / tomatoes_needed

theorem thelmas_tomato_slices : slices_per_tomato = 8 := by
  sorry

end NUMINAMATH_CALUDE_thelmas_tomato_slices_l2416_241630


namespace NUMINAMATH_CALUDE_dormitory_to_city_distance_l2416_241626

theorem dormitory_to_city_distance :
  ∀ (D : ℝ),
  (1/3 : ℝ) * D + (3/5 : ℝ) * D + 2 = D →
  D = 30 := by
sorry

end NUMINAMATH_CALUDE_dormitory_to_city_distance_l2416_241626


namespace NUMINAMATH_CALUDE_circle_condition_implies_m_range_necessary_but_not_sufficient_condition_implies_a_range_l2416_241644

-- Define the equation
def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 4*m*x + 5*m^2 + m - 2 = 0

-- Define the condition for being a circle
def is_circle (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y m

-- Define the inequality condition
def inequality_condition (m a : ℝ) : Prop :=
  (m - a) * (m - a - 4) < 0

theorem circle_condition_implies_m_range :
  (∀ m : ℝ, is_circle m → m > -2 ∧ m < 1) :=
sorry

theorem necessary_but_not_sufficient_condition_implies_a_range :
  (∀ a : ℝ, (∀ m : ℝ, inequality_condition m a → is_circle m) ∧
            (∃ m : ℝ, is_circle m ∧ ¬inequality_condition m a) →
   a ≥ -3 ∧ a ≤ -2) :=
sorry

end NUMINAMATH_CALUDE_circle_condition_implies_m_range_necessary_but_not_sufficient_condition_implies_a_range_l2416_241644


namespace NUMINAMATH_CALUDE_square_difference_l2416_241628

theorem square_difference : (50 : ℕ)^2 - (49 : ℕ)^2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2416_241628


namespace NUMINAMATH_CALUDE_terminating_decimal_thirteen_over_sixtwentyfive_l2416_241629

theorem terminating_decimal_thirteen_over_sixtwentyfive :
  (13 : ℚ) / 625 = (208 : ℚ) / 10000 :=
by sorry

end NUMINAMATH_CALUDE_terminating_decimal_thirteen_over_sixtwentyfive_l2416_241629


namespace NUMINAMATH_CALUDE_quadrilateral_formation_count_l2416_241654

theorem quadrilateral_formation_count :
  let rod_lengths : Finset ℕ := Finset.range 25
  let chosen_rods : Finset ℕ := {4, 9, 12}
  let remaining_rods := rod_lengths \ chosen_rods
  (remaining_rods.filter (fun d => 
    d + 4 + 9 > 12 ∧ d + 4 + 12 > 9 ∧ d + 9 + 12 > 4 ∧ 4 + 9 + 12 > d
  )).card = 22 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_formation_count_l2416_241654


namespace NUMINAMATH_CALUDE_binomial_expansion_sum_zero_l2416_241647

theorem binomial_expansion_sum_zero (n : ℕ) (b : ℕ) (h1 : n ≥ 2) (h2 : b > 0) :
  let a := 3 * b
  (n.choose 1 * (a - 2 * b) ^ (n - 1) + n.choose 2 * (a - 2 * b) ^ (n - 2) = 0) ↔ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_sum_zero_l2416_241647


namespace NUMINAMATH_CALUDE_cos_210_degrees_l2416_241638

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l2416_241638


namespace NUMINAMATH_CALUDE_thirty_six_in_binary_l2416_241623

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- The binary representation of 36 -/
def binary_36 : List Bool := [false, false, true, false, false, true]

/-- Theorem stating that the binary representation of 36 is 100100₂ -/
theorem thirty_six_in_binary :
  to_binary 36 = binary_36 := by sorry

end NUMINAMATH_CALUDE_thirty_six_in_binary_l2416_241623


namespace NUMINAMATH_CALUDE_average_age_combined_l2416_241681

/-- The average age of a combined group of fifth-graders and parents -/
theorem average_age_combined (num_fifth_graders : ℕ) (num_parents : ℕ) 
  (avg_age_fifth_graders : ℚ) (avg_age_parents : ℚ) :
  num_fifth_graders = 40 →
  num_parents = 50 →
  avg_age_fifth_graders = 10 →
  avg_age_parents = 35 →
  (num_fifth_graders * avg_age_fifth_graders + num_parents * avg_age_parents) / 
  (num_fifth_graders + num_parents : ℚ) = 215 / 9 := by
sorry

end NUMINAMATH_CALUDE_average_age_combined_l2416_241681


namespace NUMINAMATH_CALUDE_final_amount_after_bets_l2416_241625

/-- Calculates the final amount after a series of bets -/
def finalAmount (initialAmount : ℚ) (numBets numWins numLosses : ℕ) : ℚ :=
  initialAmount * (3/2)^numWins * (1/2)^numLosses

/-- Theorem stating the final amount after 7 bets with 4 wins and 3 losses -/
theorem final_amount_after_bets :
  finalAmount 128 7 4 3 = 81 := by
  sorry

end NUMINAMATH_CALUDE_final_amount_after_bets_l2416_241625


namespace NUMINAMATH_CALUDE_upstream_speed_l2416_241610

theorem upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 45) 
  (h2 : speed_downstream = 53) : 
  speed_still - (speed_downstream - speed_still) = 37 := by
  sorry

end NUMINAMATH_CALUDE_upstream_speed_l2416_241610


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l2416_241679

def a : Fin 2 → ℝ := ![2, -1]
def b : Fin 2 → ℝ := ![1, 1]
def c : Fin 2 → ℝ := ![-5, 1]

theorem parallel_vectors_k_value (k : ℝ) :
  (∀ i : Fin 2, ∃ t : ℝ, a i + k * b i = t * c i) →
  k = 1/2 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l2416_241679


namespace NUMINAMATH_CALUDE_inverse_proportion_points_order_l2416_241671

theorem inverse_proportion_points_order (x₁ x₂ x₃ : ℝ) :
  x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ x₃ ≠ 0 →
  -4 / x₁ = -1 →
  -4 / x₂ = 3 →
  -4 / x₃ = 5 →
  x₂ < x₃ ∧ x₃ < x₁ :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_points_order_l2416_241671


namespace NUMINAMATH_CALUDE_obtuse_angle_line_range_l2416_241673

/-- The slope of a line forming an obtuse angle with the x-axis is negative -/
def obtuse_angle_slope (a : ℝ) : Prop := a^2 + 2*a < 0

/-- The range of a for a line (a^2 + 2a)x - y + 1 = 0 forming an obtuse angle -/
theorem obtuse_angle_line_range (a : ℝ) : 
  obtuse_angle_slope a ↔ -2 < a ∧ a < 0 := by sorry

end NUMINAMATH_CALUDE_obtuse_angle_line_range_l2416_241673


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l2416_241674

/-- The length of the diagonal of a rectangle with length 40 and width 40√2 is 40√3 -/
theorem rectangle_diagonal : 
  ∀ (l w d : ℝ), 
  l = 40 → 
  w = 40 * Real.sqrt 2 → 
  d = Real.sqrt (l^2 + w^2) → 
  d = 40 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l2416_241674


namespace NUMINAMATH_CALUDE_craig_seashells_l2416_241624

theorem craig_seashells : ∃ (c : ℕ), c = 54 ∧ c > 0 ∧ ∃ (b : ℕ), 
  (c : ℚ) / (b : ℚ) = 9 / 7 ∧ b = c - 12 := by
  sorry

end NUMINAMATH_CALUDE_craig_seashells_l2416_241624


namespace NUMINAMATH_CALUDE_position_after_2004_seconds_l2416_241617

/-- Represents the position of the particle -/
structure Position :=
  (x : ℕ) (y : ℕ)

/-- Defines the movement pattern of the particle -/
def nextPosition (p : Position) : Position :=
  if p.x = p.y then Position.mk (p.x + 1) p.y
  else if p.x > p.y then Position.mk p.x (p.y + 1)
  else Position.mk p.x (p.y - 1)

/-- Calculates the position after n seconds -/
def positionAfterSeconds (n : ℕ) : Position :=
  match n with
  | 0 => Position.mk 0 0
  | 1 => Position.mk 0 1
  | n + 2 => nextPosition (positionAfterSeconds (n + 1))

/-- The main theorem stating the position after 2004 seconds -/
theorem position_after_2004_seconds :
  positionAfterSeconds 2004 = Position.mk 20 44 := by
  sorry


end NUMINAMATH_CALUDE_position_after_2004_seconds_l2416_241617


namespace NUMINAMATH_CALUDE_square_min_rotation_l2416_241609

/-- A square is a geometric shape with four equal sides and four right angles. -/
structure Square where
  side : ℝ
  side_positive : side > 0

/-- The minimum rotation angle for a square to coincide with itself. -/
def minRotationAngle (s : Square) : ℝ := 90

/-- Theorem stating that the minimum rotation angle for a square to coincide with itself is 90 degrees. -/
theorem square_min_rotation (s : Square) : minRotationAngle s = 90 := by
  sorry

end NUMINAMATH_CALUDE_square_min_rotation_l2416_241609


namespace NUMINAMATH_CALUDE_angle_sum_inequality_l2416_241694

theorem angle_sum_inequality (θ₁ θ₂ θ₃ θ₄ : Real)
  (h₁ : 0 < θ₁ ∧ θ₁ < π/2)
  (h₂ : 0 < θ₂ ∧ θ₂ < π/2)
  (h₃ : 0 < θ₃ ∧ θ₃ < π/2)
  (h₄ : 0 < θ₄ ∧ θ₄ < π/2)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) :
  (Real.sqrt 2 * Real.sin θ₁ - 1) / Real.cos θ₁ +
  (Real.sqrt 2 * Real.sin θ₂ - 1) / Real.cos θ₂ +
  (Real.sqrt 2 * Real.sin θ₃ - 1) / Real.cos θ₃ +
  (Real.sqrt 2 * Real.sin θ₄ - 1) / Real.cos θ₄ ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_angle_sum_inequality_l2416_241694


namespace NUMINAMATH_CALUDE_average_weight_increase_l2416_241607

/-- Proves that replacing a person weighing 45 kg with a person weighing 93 kg
    in a group of 8 people increases the average weight by 6 kg. -/
theorem average_weight_increase (initial_average : ℝ) :
  let group_size : ℕ := 8
  let old_weight : ℝ := 45
  let new_weight : ℝ := 93
  let weight_difference : ℝ := new_weight - old_weight
  let average_increase : ℝ := weight_difference / group_size
  average_increase = 6 := by
  sorry

#check average_weight_increase

end NUMINAMATH_CALUDE_average_weight_increase_l2416_241607


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2416_241649

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_sum : x + y + z = 3) :
  1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 1 ∧
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) = 1 ↔ x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l2416_241649


namespace NUMINAMATH_CALUDE_special_permutations_count_l2416_241664

/-- The number of permutations of 5 distinct elements where 2 specific elements are not placed at the ends -/
def special_permutations : ℕ :=
  -- Number of ways to choose 2 positions out of 3 for A and E
  (3 * 2) *
  -- Number of ways to arrange the remaining 3 elements
  (3 * 2 * 1)

theorem special_permutations_count : special_permutations = 36 := by
  sorry

end NUMINAMATH_CALUDE_special_permutations_count_l2416_241664


namespace NUMINAMATH_CALUDE_composition_of_even_is_even_l2416_241657

-- Define a type for real-valued functions
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be even
def IsEven (f : RealFunction) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (g : RealFunction) (h_even : IsEven g) :
  IsEven (g ∘ g) := by
  sorry

end NUMINAMATH_CALUDE_composition_of_even_is_even_l2416_241657


namespace NUMINAMATH_CALUDE_a_2_value_a_n_formula_l2416_241656

def sequence_a (n : ℕ) : ℝ := sorry

def S (n : ℕ) : ℝ := sorry

axiom a_1 : sequence_a 1 = 1

axiom relation (n : ℕ) (hn : n > 0) : 
  2 * S n / n = sequence_a (n + 1) - (1/3) * n^2 - n - 2/3

theorem a_2_value : sequence_a 2 = 4 := by sorry

theorem a_n_formula (n : ℕ) (hn : n > 0) : sequence_a n = n^2 := by sorry

end NUMINAMATH_CALUDE_a_2_value_a_n_formula_l2416_241656


namespace NUMINAMATH_CALUDE_train_speed_equation_l2416_241643

theorem train_speed_equation (x : ℝ) (h : x > 0) : 
  700 / x - 700 / (2.8 * x) = 3.6 ↔ 
  (∃ (t_express t_highspeed : ℝ),
    t_express = 700 / x ∧
    t_highspeed = 700 / (2.8 * x) ∧
    t_express - t_highspeed = 3.6) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_equation_l2416_241643


namespace NUMINAMATH_CALUDE_treasure_hunt_probability_l2416_241677

def num_islands : ℕ := 7
def num_treasure_islands : ℕ := 4

def prob_treasure : ℚ := 1/3
def prob_traps : ℚ := 1/6
def prob_neither : ℚ := 1/2

theorem treasure_hunt_probability :
  (Nat.choose num_islands num_treasure_islands : ℚ) *
  prob_treasure ^ num_treasure_islands *
  prob_neither ^ (num_islands - num_treasure_islands) =
  35/648 := by
  sorry

end NUMINAMATH_CALUDE_treasure_hunt_probability_l2416_241677


namespace NUMINAMATH_CALUDE_survey_selection_theorem_l2416_241608

-- Define the number of boys and girls
def num_boys : ℕ := 4
def num_girls : ℕ := 2

-- Define the total number of students to be selected
def num_selected : ℕ := 4

-- Define the function to calculate the number of ways to select students
def num_ways_to_select : ℕ := (num_boys + num_girls).choose num_selected - num_boys.choose num_selected

-- Theorem statement
theorem survey_selection_theorem : num_ways_to_select = 14 := by
  sorry

end NUMINAMATH_CALUDE_survey_selection_theorem_l2416_241608


namespace NUMINAMATH_CALUDE_diego_martha_can_ratio_l2416_241699

theorem diego_martha_can_ratio :
  let martha_cans : ℕ := 90
  let total_needed : ℕ := 150
  let more_needed : ℕ := 5
  let total_collected : ℕ := total_needed - more_needed
  let diego_cans : ℕ := total_collected - martha_cans
  (diego_cans : ℚ) / martha_cans = 11 / 18 := by
  sorry

end NUMINAMATH_CALUDE_diego_martha_can_ratio_l2416_241699


namespace NUMINAMATH_CALUDE_negation_square_nonnegative_l2416_241605

theorem negation_square_nonnegative :
  ¬(∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by sorry

end NUMINAMATH_CALUDE_negation_square_nonnegative_l2416_241605


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_equality_l2416_241682

theorem min_value_theorem (x : ℝ) (h : x > 0) :
  3 * x + 5 + 2 / x^5 ≥ 10 + 3 * (2/5)^(1/5) :=
by sorry

theorem min_value_equality :
  let x := (2/5)^(1/5)
  3 * x + 5 + 2 / x^5 = 10 + 3 * (2/5)^(1/5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_equality_l2416_241682


namespace NUMINAMATH_CALUDE_g_of_3_equals_135_l2416_241697

/-- Given that g(x) = 3x^4 - 5x^3 + 2x^2 + x + 6, prove that g(3) = 135 -/
theorem g_of_3_equals_135 : 
  let g : ℝ → ℝ := λ x ↦ 3*x^4 - 5*x^3 + 2*x^2 + x + 6
  g 3 = 135 := by sorry

end NUMINAMATH_CALUDE_g_of_3_equals_135_l2416_241697


namespace NUMINAMATH_CALUDE_household_size_proof_l2416_241689

/-- The number of slices of bread consumed by each member daily. -/
def daily_consumption : ℕ := 5

/-- The number of slices in a loaf of bread. -/
def slices_per_loaf : ℕ := 12

/-- The number of loaves that last for 3 days. -/
def loaves_for_three_days : ℕ := 5

/-- The number of days the loaves last. -/
def days : ℕ := 3

/-- The number of members in the household. -/
def household_members : ℕ := 4

theorem household_size_proof :
  household_members * daily_consumption * days = loaves_for_three_days * slices_per_loaf :=
by sorry

end NUMINAMATH_CALUDE_household_size_proof_l2416_241689


namespace NUMINAMATH_CALUDE_remainder_approximation_l2416_241666

/-- Given two positive real numbers satisfying certain conditions, 
    prove that the remainder of their division is approximately 15. -/
theorem remainder_approximation (L S : ℝ) (hL : L > 0) (hS : S > 0) 
    (h_diff : L - S = 1365)
    (h_approx : |L - 1542.857| < 0.001)
    (h_div : ∃ R : ℝ, R ≥ 0 ∧ L = 8 * S + R) : 
  ∃ R : ℝ, R ≥ 0 ∧ L = 8 * S + R ∧ |R - 15| < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_approximation_l2416_241666


namespace NUMINAMATH_CALUDE_equal_number_of_boys_and_girls_l2416_241631

/-- Represents a school with boys and girls -/
structure School where
  boys : ℕ
  girls : ℕ
  boys_age_sum : ℕ
  girls_age_sum : ℕ

/-- The average age of boys -/
def boys_avg (s : School) : ℚ := s.boys_age_sum / s.boys

/-- The average age of girls -/
def girls_avg (s : School) : ℚ := s.girls_age_sum / s.girls

/-- The average age of all students -/
def total_avg (s : School) : ℚ := (s.boys_age_sum + s.girls_age_sum) / (s.boys + s.girls)

/-- The theorem stating that the number of boys equals the number of girls -/
theorem equal_number_of_boys_and_girls (s : School) 
  (h1 : boys_avg s ≠ girls_avg s) 
  (h2 : (boys_avg s + girls_avg s) / 2 = total_avg s) : 
  s.boys = s.girls := by sorry

end NUMINAMATH_CALUDE_equal_number_of_boys_and_girls_l2416_241631


namespace NUMINAMATH_CALUDE_overall_profit_percentage_l2416_241660

def book_a_cost : ℚ := 50
def book_b_cost : ℚ := 75
def book_c_cost : ℚ := 100
def book_a_sell : ℚ := 60
def book_b_sell : ℚ := 90
def book_c_sell : ℚ := 120

def total_investment_cost : ℚ := book_a_cost + book_b_cost + book_c_cost
def total_revenue : ℚ := book_a_sell + book_b_sell + book_c_sell
def total_profit : ℚ := total_revenue - total_investment_cost
def profit_percentage : ℚ := (total_profit / total_investment_cost) * 100

theorem overall_profit_percentage :
  profit_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_overall_profit_percentage_l2416_241660


namespace NUMINAMATH_CALUDE_festival_attendance_l2416_241688

theorem festival_attendance (total_students : ℕ) (festival_attendees : ℕ) 
  (h1 : total_students = 1500)
  (h2 : festival_attendees = 900) : ℕ :=
by
  let girls : ℕ := sorry
  let boys : ℕ := sorry
  have h3 : girls + boys = total_students := sorry
  have h4 : (3 * girls / 4 : ℚ) + (2 * boys / 3 : ℚ) = festival_attendees := sorry
  have h5 : (3 * girls / 4 : ℕ) = 900 := sorry
  exact 900

#check festival_attendance

end NUMINAMATH_CALUDE_festival_attendance_l2416_241688


namespace NUMINAMATH_CALUDE_annual_income_calculation_l2416_241612

theorem annual_income_calculation (total : ℝ) (p1 : ℝ) (rate1 : ℝ) (rate2 : ℝ)
  (h1 : total = 2500)
  (h2 : p1 = 500.0000000000002)
  (h3 : rate1 = 0.05)
  (h4 : rate2 = 0.06) :
  let p2 := total - p1
  let income1 := p1 * rate1
  let income2 := p2 * rate2
  income1 + income2 = 145 := by sorry

end NUMINAMATH_CALUDE_annual_income_calculation_l2416_241612


namespace NUMINAMATH_CALUDE_prob_live_to_25_given_20_l2416_241652

/-- The probability of an animal living to 25 years given it has lived to 20 years -/
theorem prob_live_to_25_given_20 (p_20 p_25 : ℝ) 
  (h1 : p_20 = 0.8) 
  (h2 : p_25 = 0.4) 
  (h3 : 0 ≤ p_20 ∧ p_20 ≤ 1) 
  (h4 : 0 ≤ p_25 ∧ p_25 ≤ 1) 
  (h5 : p_25 ≤ p_20) : 
  p_25 / p_20 = 0.5 := by sorry

end NUMINAMATH_CALUDE_prob_live_to_25_given_20_l2416_241652


namespace NUMINAMATH_CALUDE_line_point_sum_l2416_241622

/-- The line equation y = -1/2x + 8 -/
def line_equation (x y : ℝ) : Prop := y = -1/2 * x + 8

/-- Point P is where the line crosses the x-axis -/
def point_P : ℝ × ℝ := (16, 0)

/-- Point Q is where the line crosses the y-axis -/
def point_Q : ℝ × ℝ := (0, 8)

/-- Point T is on the line segment PQ -/
def point_T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
  r = t * point_P.1 + (1 - t) * point_Q.1 ∧
  s = t * point_P.2 + (1 - t) * point_Q.2

/-- The area of triangle POQ is four times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((1/2) * point_P.1 * point_Q.2) = 4 * abs ((1/2) * r * s)

theorem line_point_sum (r s : ℝ) :
  line_equation r s →
  point_T_on_PQ r s →
  area_condition r s →
  r + s = 14 :=
by sorry

end NUMINAMATH_CALUDE_line_point_sum_l2416_241622


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l2416_241606

def geometric_sequence (a₀ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₀ * r^n

theorem fifth_term_of_sequence (x : ℝ) :
  let a₀ := 3
  let r := 3 * x^2
  geometric_sequence a₀ r 4 = 243 * x^8 :=
by sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l2416_241606


namespace NUMINAMATH_CALUDE_candy_bar_difference_l2416_241602

theorem candy_bar_difference (lena kevin nicole : ℕ) : 
  lena = 16 → 
  lena + 5 = 3 * kevin → 
  nicole = kevin + 4 → 
  lena - nicole = 5 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_difference_l2416_241602


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l2416_241691

theorem max_value_trig_expression (x y z : ℝ) :
  (Real.sin (2 * x) + Real.sin y + Real.sin (3 * z)) *
  (Real.cos (2 * x) + Real.cos y + Real.cos (3 * z)) ≤ 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l2416_241691


namespace NUMINAMATH_CALUDE_sum_243_81_base3_l2416_241687

/-- Converts a natural number to its base 3 representation as a list of digits -/
def toBase3 (n : ℕ) : List ℕ := sorry

/-- Adds two numbers represented in base 3 -/
def addBase3 (a b : List ℕ) : List ℕ := sorry

/-- Checks if a list of digits is a valid base 3 representation -/
def isValidBase3 (l : List ℕ) : Prop := sorry

theorem sum_243_81_base3 :
  let a := toBase3 243
  let b := toBase3 81
  let sum := addBase3 a b
  isValidBase3 a ∧ isValidBase3 b ∧ isValidBase3 sum ∧ sum = [0, 0, 0, 0, 1, 1] := by sorry

end NUMINAMATH_CALUDE_sum_243_81_base3_l2416_241687


namespace NUMINAMATH_CALUDE_paving_cost_l2416_241686

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 400) :
  length * width * rate = 8250 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l2416_241686


namespace NUMINAMATH_CALUDE_sum_of_roots_l2416_241676

theorem sum_of_roots (x y : ℝ) 
  (hx : x^3 + 6*x^2 + 16*x = -15) 
  (hy : y^3 + 6*y^2 + 16*y = -17) : 
  x + y = -4 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l2416_241676


namespace NUMINAMATH_CALUDE_cubic_factorization_l2416_241680

theorem cubic_factorization (x : ℝ) : 
  x^3 + x^2 - 2*x - 2 = (x + 1) * (x - Real.sqrt 2) * (x + Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2416_241680


namespace NUMINAMATH_CALUDE_apple_capacity_l2416_241683

def bookbag_capacity : ℕ := 20
def other_fruit_weight : ℕ := 3

theorem apple_capacity : bookbag_capacity - other_fruit_weight = 17 := by
  sorry

end NUMINAMATH_CALUDE_apple_capacity_l2416_241683


namespace NUMINAMATH_CALUDE_roberts_birthday_l2416_241693

/-- The number of years until Robert turns 30 -/
def years_until_30 (patrick_age : ℕ) (robert_age : ℕ) : ℕ :=
  30 - robert_age

/-- Robert's current age is twice Patrick's age -/
def robert_age (patrick_age : ℕ) : ℕ :=
  2 * patrick_age

theorem roberts_birthday (patrick_age : ℕ) (h1 : patrick_age = 14) :
  years_until_30 patrick_age (robert_age patrick_age) = 2 := by
  sorry

end NUMINAMATH_CALUDE_roberts_birthday_l2416_241693


namespace NUMINAMATH_CALUDE_expression_simplification_l2416_241632

theorem expression_simplification (x y : ℝ) (h : 2 * x + y - 3 = 0) :
  ((3 * x) / (x - y) + x / (x + y)) / (x / (x^2 - y^2)) = 6 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2416_241632


namespace NUMINAMATH_CALUDE_vacation_cost_from_dog_walking_vacation_cost_proof_l2416_241669

/-- Calculates the total cost of a vacation based on dog walking earnings --/
theorem vacation_cost_from_dog_walking 
  (start_charge : ℚ)
  (per_block_charge : ℚ)
  (num_dogs : ℕ)
  (total_blocks : ℕ)
  (family_members : ℕ)
  (h1 : start_charge = 2)
  (h2 : per_block_charge = 5/4)
  (h3 : num_dogs = 20)
  (h4 : total_blocks = 128)
  (h5 : family_members = 5)
  : ℚ
  :=
  let total_earnings := start_charge * num_dogs + per_block_charge * total_blocks
  total_earnings

theorem vacation_cost_proof
  (start_charge : ℚ)
  (per_block_charge : ℚ)
  (num_dogs : ℕ)
  (total_blocks : ℕ)
  (family_members : ℕ)
  (h1 : start_charge = 2)
  (h2 : per_block_charge = 5/4)
  (h3 : num_dogs = 20)
  (h4 : total_blocks = 128)
  (h5 : family_members = 5)
  : vacation_cost_from_dog_walking start_charge per_block_charge num_dogs total_blocks family_members h1 h2 h3 h4 h5 = 200 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_from_dog_walking_vacation_cost_proof_l2416_241669


namespace NUMINAMATH_CALUDE_three_digit_number_theorem_l2416_241650

theorem three_digit_number_theorem (x y z : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ x ≠ 0 →
  let n := 100 * x + 10 * y + z
  let sum_digits := x + y + z
  n / sum_digits = 13 ∧ n % sum_digits = 15 →
  n = 106 ∨ n = 145 ∨ n = 184 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_theorem_l2416_241650


namespace NUMINAMATH_CALUDE_veg_eaters_count_l2416_241685

/-- Represents the number of people in different dietary categories in a family -/
structure FamilyDiet where
  onlyVeg : ℕ
  bothVegNonVeg : ℕ

/-- Calculates the total number of people who eat vegetarian food in the family -/
def totalVegEaters (fd : FamilyDiet) : ℕ :=
  fd.onlyVeg + fd.bothVegNonVeg

/-- Theorem: The number of people who eat veg in the family is 21 -/
theorem veg_eaters_count (fd : FamilyDiet) 
  (h1 : fd.onlyVeg = 13)
  (h2 : fd.bothVegNonVeg = 8) : 
  totalVegEaters fd = 21 := by
  sorry

end NUMINAMATH_CALUDE_veg_eaters_count_l2416_241685


namespace NUMINAMATH_CALUDE_al2s3_weight_l2416_241635

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight (al_weight s_weight : ℝ) : ℝ :=
  2 * al_weight + 3 * s_weight

/-- The total weight of a given number of moles of a compound -/
def total_weight (moles : ℝ) (mol_weight : ℝ) : ℝ :=
  moles * mol_weight

/-- Theorem: The molecular weight of 3 moles of Al2S3 is 450.51 grams -/
theorem al2s3_weight : 
  let al_weight := 26.98
  let s_weight := 32.07
  let mol_weight := molecular_weight al_weight s_weight
  total_weight 3 mol_weight = 450.51 := by
sorry


end NUMINAMATH_CALUDE_al2s3_weight_l2416_241635


namespace NUMINAMATH_CALUDE_equation_implies_fraction_value_l2416_241661

theorem equation_implies_fraction_value (a x y : ℝ) :
  x * Real.sqrt (a * (x - a)) + y * Real.sqrt (a * (y - a)) = Real.sqrt (Real.log (x - a) - Real.log (a - y)) →
  (3 * x^2 + x * y - y^2) / (x^2 - x * y + y^2) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_implies_fraction_value_l2416_241661


namespace NUMINAMATH_CALUDE_distance_to_line_mn_l2416_241692

/-- The distance from the origin to the line MN, where M is on the hyperbola 2x² - y² = 1
    and N is on the ellipse 4x² + y² = 1, with OM perpendicular to ON. -/
theorem distance_to_line_mn (M N : ℝ × ℝ) : 
  (2 * M.1^2 - M.2^2 = 1) →  -- M is on the hyperbola
  (4 * N.1^2 + N.2^2 = 1) →  -- N is on the ellipse
  (M.1 * N.1 + M.2 * N.2 = 0) →  -- OM ⟂ ON
  let d := Real.sqrt 3 / 3
  ∃ (t : ℝ), t * M.1 + (1 - t) * N.1 = d * (N.2 - M.2) ∧
             t * M.2 + (1 - t) * N.2 = d * (M.1 - N.1) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_line_mn_l2416_241692


namespace NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l2416_241613

/-- A circle is tangent to two parallel lines and its center lies on a third line -/
theorem circle_tangent_to_parallel_lines (x y : ℚ) :
  (3 * x + 4 * y = 40) ∧ 
  (3 * x + 4 * y = -20) ∧ 
  (x - 3 * y = 0) →
  x = 30 / 13 ∧ y = 10 / 13 := by
  sorry

#check circle_tangent_to_parallel_lines

end NUMINAMATH_CALUDE_circle_tangent_to_parallel_lines_l2416_241613


namespace NUMINAMATH_CALUDE_book_pricing_loss_percentage_l2416_241618

theorem book_pricing_loss_percentage 
  (cost_price selling_price : ℝ) 
  (h1 : cost_price > 0) 
  (h2 : 5 * cost_price = 20 * selling_price) : 
  (cost_price - selling_price) / cost_price = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_book_pricing_loss_percentage_l2416_241618


namespace NUMINAMATH_CALUDE_train_speed_l2416_241611

theorem train_speed (length time : ℝ) (h1 : length = 300) (h2 : time = 15) :
  length / time = 20 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2416_241611


namespace NUMINAMATH_CALUDE_four_digit_number_theorem_l2416_241663

def is_valid_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def first_two_digits (n : ℕ) : ℕ :=
  n / 100

def last_two_digits (n : ℕ) : ℕ :=
  n % 100

def is_permutation (a b : ℕ) : Prop :=
  a / 10 = b % 10 ∧ a % 10 = b / 10

def sum_of_digits (n : ℕ) : ℕ :=
  n / 10 + n % 10

theorem four_digit_number_theorem :
  ∃! n : ℕ, 
    is_valid_four_digit_number n ∧
    is_permutation (first_two_digits n) (last_two_digits n) ∧
    first_two_digits n - last_two_digits n = sum_of_digits (first_two_digits n) ∧
    n = 5445 :=
by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_theorem_l2416_241663


namespace NUMINAMATH_CALUDE_sin_seven_pi_sixths_l2416_241662

theorem sin_seven_pi_sixths : Real.sin (7 * π / 6) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_seven_pi_sixths_l2416_241662
