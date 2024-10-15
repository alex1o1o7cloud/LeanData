import Mathlib

namespace NUMINAMATH_CALUDE_min_sum_x_y_l1599_159916

theorem min_sum_x_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + x*y - 7 = 0) :
  ∃ (m : ℝ), m = 3 ∧ x + y ≥ m ∧ ∀ (z : ℝ), x + y > z → z < m :=
sorry

end NUMINAMATH_CALUDE_min_sum_x_y_l1599_159916


namespace NUMINAMATH_CALUDE_open_box_volume_l1599_159900

/-- The volume of an open box formed by cutting squares from the corners of a rectangular sheet -/
theorem open_box_volume 
  (sheet_length : ℝ) 
  (sheet_width : ℝ) 
  (cut_square_side : ℝ) 
  (h1 : sheet_length = 50) 
  (h2 : sheet_width = 36) 
  (h3 : cut_square_side = 8) : 
  (sheet_length - 2 * cut_square_side) * (sheet_width - 2 * cut_square_side) * cut_square_side = 5440 := by
  sorry

#check open_box_volume

end NUMINAMATH_CALUDE_open_box_volume_l1599_159900


namespace NUMINAMATH_CALUDE_reinforcement_size_l1599_159993

/-- Calculates the size of reinforcement given initial garrison size, initial provisions duration,
    time passed before reinforcement, and remaining provisions duration after reinforcement. -/
def calculate_reinforcement (initial_garrison : ℕ) (initial_duration : ℕ) 
                            (time_passed : ℕ) (remaining_duration : ℕ) : ℕ :=
  let total_provisions := initial_garrison * initial_duration
  let provisions_left := total_provisions - (initial_garrison * time_passed)
  let reinforcement := (provisions_left / remaining_duration) - initial_garrison
  reinforcement

/-- Theorem stating that given the specific conditions of the problem,
    the calculated reinforcement size is 3000. -/
theorem reinforcement_size :
  calculate_reinforcement 2000 65 15 20 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_l1599_159993


namespace NUMINAMATH_CALUDE_contest_probability_l1599_159999

theorem contest_probability (p q : ℝ) (h_p : p = 2/3) (h_q : q = 1/3) :
  ∃ n : ℕ, n > 0 ∧ (p ^ n < 0.05) ∧ ∀ m : ℕ, m > 0 → m < n → p ^ m ≥ 0.05 :=
sorry

end NUMINAMATH_CALUDE_contest_probability_l1599_159999


namespace NUMINAMATH_CALUDE_a_equals_permutation_l1599_159907

-- Define a as the product n(n-1)(n-2)...(n-50)
def a (n : ℕ) : ℕ := (List.range 51).foldl (λ acc i => acc * (n - i)) n

-- Define the permutation function A_n^k
def permutation (n k : ℕ) : ℕ := (List.range k).foldl (λ acc i => acc * (n - i)) 1

-- Theorem statement
theorem a_equals_permutation (n : ℕ) : a n = permutation n 51 := by sorry

end NUMINAMATH_CALUDE_a_equals_permutation_l1599_159907


namespace NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l1599_159940

/-- The perimeter of a rectangular garden with a width of 12 meters and an area equal to a 16x12 meter playground is 56 meters. -/
theorem garden_perimeter : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun garden_width garden_length playground_length playground_width =>
    garden_width = 12 ∧
    garden_length * garden_width = playground_length * playground_width ∧
    playground_length = 16 ∧
    playground_width = 12 →
    2 * (garden_length + garden_width) = 56

-- The proof is omitted
theorem garden_perimeter_proof : garden_perimeter 12 16 16 12 := by sorry

end NUMINAMATH_CALUDE_garden_perimeter_garden_perimeter_proof_l1599_159940


namespace NUMINAMATH_CALUDE_arccos_cos_ten_equals_two_l1599_159977

theorem arccos_cos_ten_equals_two : Real.arccos (Real.cos 10) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arccos_cos_ten_equals_two_l1599_159977


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1599_159972

theorem polynomial_simplification (x : ℝ) :
  4 * x^3 + 5 * x^2 + 2 * x + 8 - (3 * x^3 - 7 * x^2 + 4 * x - 6) =
  x^3 + 12 * x^2 - 2 * x + 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1599_159972


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l1599_159994

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 5|

-- Theorem for part I
theorem solution_set_of_inequality (x : ℝ) :
  (f x ≤ x + 10) ↔ (x ∈ Set.Icc (-2) 14) :=
sorry

-- Theorem for part II
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ a - (x - 2)^2) ↔ (a ∈ Set.Iic 6) :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_range_of_a_l1599_159994


namespace NUMINAMATH_CALUDE_inequality_proof_l1599_159911

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  b * c * d / (1 - a)^2 + c * d * a / (1 - b)^2 + 
  d * a * b / (1 - c)^2 + a * b * c / (1 - d)^2 ≤ 1 / 9 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1599_159911


namespace NUMINAMATH_CALUDE_quadratic_equation_from_roots_l1599_159945

theorem quadratic_equation_from_roots (x₁ x₂ : ℝ) (hx₁ : x₁ = 1) (hx₂ : x₂ = 2) :
  ∃ a b c : ℝ, a ≠ 0 ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧
  a * x^2 + b * x + c = x^2 - 3*x + 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_roots_l1599_159945


namespace NUMINAMATH_CALUDE_unique_prime_ending_l1599_159959

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def number (A : ℕ) : ℕ := 130400 + A

theorem unique_prime_ending :
  ∃! A : ℕ, A < 10 ∧ is_prime (number A) :=
sorry

end NUMINAMATH_CALUDE_unique_prime_ending_l1599_159959


namespace NUMINAMATH_CALUDE_absolute_value_square_inequality_l1599_159996

theorem absolute_value_square_inequality {a b : ℝ} (h : |a| < b) : a^2 < b^2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_inequality_l1599_159996


namespace NUMINAMATH_CALUDE_soccer_team_points_l1599_159930

/-- Calculates the total points for a soccer team given their game results -/
def calculate_points (total_games : ℕ) (wins : ℕ) (losses : ℕ) (win_points : ℕ) (draw_points : ℕ) (loss_points : ℕ) : ℕ :=
  let draws := total_games - wins - losses
  wins * win_points + draws * draw_points + losses * loss_points

/-- Theorem stating that the soccer team's total points is 46 -/
theorem soccer_team_points :
  calculate_points 20 14 2 3 1 0 = 46 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_points_l1599_159930


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l1599_159901

theorem two_digit_number_sum (n : ℕ) : 
  10 ≤ n ∧ n < 100 →  -- n is a two-digit number
  (n : ℚ) / 2 = n / 4 + 3 →  -- one half of n exceeds its one fourth by 3
  (n / 10 + n % 10 : ℕ) = 12  -- sum of digits is 12
  := by sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l1599_159901


namespace NUMINAMATH_CALUDE_equation_solvability_l1599_159955

theorem equation_solvability (a : ℝ) : 
  (∃ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x = 2 * a - 1) ↔ 
  (-1/2 : ℝ) ≤ a ∧ a ≤ (3/2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_equation_solvability_l1599_159955


namespace NUMINAMATH_CALUDE_single_color_subgraph_exists_l1599_159981

/-- A graph where each pair of vertices is connected by exactly one of two types of edges -/
structure TwoColorGraph (α : Type*) where
  vertices : Set α
  edge_type1 : α → α → Prop
  edge_type2 : α → α → Prop
  edge_exists : ∀ (v w : α), v ∈ vertices → w ∈ vertices → v ≠ w → 
    (edge_type1 v w ∧ ¬edge_type2 v w) ∨ (edge_type2 v w ∧ ¬edge_type1 v w)

/-- A subgraph that includes all vertices and uses only one type of edge -/
def SingleColorSubgraph {α : Type*} (G : TwoColorGraph α) :=
  {H : Set (α × α) // 
    (∀ v ∈ G.vertices, ∃ w, (v, w) ∈ H ∨ (w, v) ∈ H) ∧
    (∀ (v w : α), (v, w) ∈ H → G.edge_type1 v w) ∨
    (∀ (v w : α), (v, w) ∈ H → G.edge_type2 v w)}

/-- The main theorem: there always exists a single-color subgraph -/
theorem single_color_subgraph_exists {α : Type*} (G : TwoColorGraph α) :
  Nonempty (SingleColorSubgraph G) := by
  sorry

end NUMINAMATH_CALUDE_single_color_subgraph_exists_l1599_159981


namespace NUMINAMATH_CALUDE_equation_solution_l1599_159995

theorem equation_solution :
  let f (x : ℝ) := x + 3 = 4 / (x - 2)
  ∀ x : ℝ, x ≠ 2 → (f x ↔ (x = (-1 + Real.sqrt 41) / 2 ∨ x = (-1 - Real.sqrt 41) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1599_159995


namespace NUMINAMATH_CALUDE_circle_line_tangent_problem_l1599_159947

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y + 1 = 0

-- Define the line l
def line_l (x y : ℝ) (m : ℝ) : Prop :=
  x + m*y + 1 = 0

-- Define the point M
def point_M (m : ℝ) : ℝ × ℝ :=
  (m, m)

-- Define the symmetric property
def symmetric_points_exist (m : ℝ) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle_C x1 y1 ∧ circle_C x2 y2 ∧
    line_l ((x1 + x2)/2) ((y1 + y2)/2) m

-- Define the tangent property
def tangent_exists (m : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_C x y ∧
    ∃ (k : ℝ), ∀ (t : ℝ),
      ¬(circle_C (m + k*t) (m + t))

-- Main theorem
theorem circle_line_tangent_problem (m : ℝ) :
  symmetric_points_exist m → tangent_exists m →
  m = -1 ∧ Real.sqrt ((m - 1)^2 + (m - 2)^2 - 4) = 3 :=
sorry

end NUMINAMATH_CALUDE_circle_line_tangent_problem_l1599_159947


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1599_159944

theorem two_numbers_difference (x y : ℚ) 
  (sum_eq : x + y = 40)
  (triple_minus_quadruple : 3 * y - 4 * x = 20) :
  |y - x| = 80 / 7 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1599_159944


namespace NUMINAMATH_CALUDE_lesser_number_l1599_159968

theorem lesser_number (x y : ℤ) (sum_eq : x + y = 58) (diff_eq : x - y = 6) : 
  min x y = 26 := by
sorry

end NUMINAMATH_CALUDE_lesser_number_l1599_159968


namespace NUMINAMATH_CALUDE_binomial_expansion_theorem_l1599_159979

theorem binomial_expansion_theorem (n k : ℕ) (a b : ℝ) : 
  n ≥ 2 → 
  a * b ≠ 0 → 
  k ≥ 1 → 
  a = 2 * k * b → 
  (Nat.choose n 2 * (2 * b)^(n - 2) * (k - 1)^2 + 
   Nat.choose n 3 * (2 * b)^(n - 3) * (k - 1)^3 = 0) → 
  n = 3 * k - 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_theorem_l1599_159979


namespace NUMINAMATH_CALUDE_bathtub_fill_time_l1599_159913

/-- Represents the filling and draining rates of a bathtub -/
structure BathtubRates where
  cold_fill_time : ℚ
  hot_fill_time : ℚ
  drain_time : ℚ

/-- Calculates the time to fill the bathtub with both taps open and drain unplugged -/
def fill_time (rates : BathtubRates) : ℚ :=
  1 / ((1 / rates.cold_fill_time) + (1 / rates.hot_fill_time) - (1 / rates.drain_time))

/-- Theorem: Given the specified filling and draining rates, the bathtub will fill in 5 minutes -/
theorem bathtub_fill_time (rates : BathtubRates) 
  (h1 : rates.cold_fill_time = 20 / 3)
  (h2 : rates.hot_fill_time = 8)
  (h3 : rates.drain_time = 40 / 3) :
  fill_time rates = 5 := by
  sorry

#eval fill_time { cold_fill_time := 20 / 3, hot_fill_time := 8, drain_time := 40 / 3 }

end NUMINAMATH_CALUDE_bathtub_fill_time_l1599_159913


namespace NUMINAMATH_CALUDE_prob_different_topics_correct_l1599_159991

/-- The number of essay topics -/
def num_topics : ℕ := 6

/-- The probability that two students select different topics -/
def prob_different_topics : ℚ := 5/6

/-- Theorem stating that the probability of two students selecting different topics
    out of num_topics is equal to prob_different_topics -/
theorem prob_different_topics_correct :
  (num_topics : ℚ) * (num_topics - 1) / (num_topics * num_topics) = prob_different_topics := by
  sorry

end NUMINAMATH_CALUDE_prob_different_topics_correct_l1599_159991


namespace NUMINAMATH_CALUDE_cos_negative_thirteen_pi_over_four_l1599_159914

theorem cos_negative_thirteen_pi_over_four :
  Real.cos (-13 * π / 4) = -Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_cos_negative_thirteen_pi_over_four_l1599_159914


namespace NUMINAMATH_CALUDE_bike_trip_distance_difference_l1599_159998

-- Define the parameters of the problem
def total_time : ℝ := 6
def alberto_speed : ℝ := 12
def bjorn_speed : ℝ := 10
def bjorn_rest_time : ℝ := 1

-- Define the distances traveled by Alberto and Bjorn
def alberto_distance : ℝ := alberto_speed * total_time
def bjorn_distance : ℝ := bjorn_speed * (total_time - bjorn_rest_time)

-- State the theorem
theorem bike_trip_distance_difference :
  alberto_distance - bjorn_distance = 22 := by sorry

end NUMINAMATH_CALUDE_bike_trip_distance_difference_l1599_159998


namespace NUMINAMATH_CALUDE_sin_A_value_side_c_value_l1599_159934

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.C = 2 * Real.pi / 3 ∧ t.a = 6

-- Theorem 1
theorem sin_A_value (t : Triangle) (h : triangle_conditions t) (hc : t.c = 14) :
  Real.sin t.A = (3 / 14) * Real.sqrt 3 := by
  sorry

-- Theorem 2
theorem side_c_value (t : Triangle) (h : triangle_conditions t) (harea : 1/2 * t.a * t.b * Real.sin t.C = 3 * Real.sqrt 3) :
  t.c = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_sin_A_value_side_c_value_l1599_159934


namespace NUMINAMATH_CALUDE_system_solution_l1599_159974

theorem system_solution :
  ∃ (x y : ℝ), x = -1 ∧ y = -2 ∧ x - 3*y = 5 ∧ 4*x - 3*y = 2 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1599_159974


namespace NUMINAMATH_CALUDE_range_of_a_l1599_159961

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → 0 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1599_159961


namespace NUMINAMATH_CALUDE_factorization_1_factorization_2_l1599_159962

-- Factorization of -4a²x + 12ax - 9x
theorem factorization_1 (a x : ℝ) : -4 * a^2 * x + 12 * a * x - 9 * x = -x * (2*a - 3)^2 := by
  sorry

-- Factorization of (2x + y)² - (x + 2y)²
theorem factorization_2 (x y : ℝ) : (2*x + y)^2 - (x + 2*y)^2 = 3 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_factorization_2_l1599_159962


namespace NUMINAMATH_CALUDE_roots_of_equation_l1599_159927

theorem roots_of_equation (x : ℝ) : 
  (x - 3)^2 = 4 ↔ x = 5 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l1599_159927


namespace NUMINAMATH_CALUDE_empty_plane_speed_theorem_l1599_159943

/-- The speed of an empty plane given the conditions of the problem -/
def empty_plane_speed (p1 p2 p3 : ℕ) (speed_reduction : ℕ) (avg_speed : ℕ) : ℕ :=
  3 * avg_speed + p1 * speed_reduction + p2 * speed_reduction + p3 * speed_reduction

/-- Theorem stating the speed of an empty plane under the given conditions -/
theorem empty_plane_speed_theorem :
  empty_plane_speed 50 60 40 2 500 = 600 := by
  sorry

#eval empty_plane_speed 50 60 40 2 500

end NUMINAMATH_CALUDE_empty_plane_speed_theorem_l1599_159943


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1599_159904

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 + 2*i) / (3 - 4*i)) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1599_159904


namespace NUMINAMATH_CALUDE_polynomial_remainder_zero_l1599_159939

theorem polynomial_remainder_zero (x : ℝ) : 
  (x^3 - 5*x^2 + 2*x + 8) % (x - 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_zero_l1599_159939


namespace NUMINAMATH_CALUDE_negation_equivalence_l1599_159949

theorem negation_equivalence (S : Set ℕ) :
  (¬ ∀ x ∈ S, x^2 ≠ 4) ↔ (∃ x ∈ S, x^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1599_159949


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_set_l1599_159956

def A : Set ℝ := {-2, -1, 0, 1}

def B : Set ℝ := {y | ∃ x, y = 1 / (2^x - 2)}

theorem A_intersect_B_eq_set : A ∩ B = {-2, -1, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_set_l1599_159956


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l1599_159950

/-- The slope of a chord in an ellipse --/
theorem ellipse_chord_slope (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁^2 / 8 + y₁^2 / 6 = 1) →
  (x₂^2 / 8 + y₂^2 / 6 = 1) →
  ((x₁ + x₂) / 2 = 2) →
  ((y₁ + y₂) / 2 = 1) →
  (y₂ - y₁) / (x₂ - x₁) = -3/2 := by
sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l1599_159950


namespace NUMINAMATH_CALUDE_union_of_P_and_Q_l1599_159989

-- Define the sets P and Q
def P : Set ℝ := {x | -1 < x ∧ x < 1}
def Q : Set ℝ := {x | -2 < x ∧ x < 0}

-- Define the open interval (-2, 1)
def openInterval : Set ℝ := {x | -2 < x ∧ x < 1}

-- Theorem statement
theorem union_of_P_and_Q : P ∪ Q = openInterval := by sorry

end NUMINAMATH_CALUDE_union_of_P_and_Q_l1599_159989


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1599_159963

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 8) : y = 26 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1599_159963


namespace NUMINAMATH_CALUDE_circle_intersection_slope_range_l1599_159903

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 25

-- Define the line that contains the center of C
def center_line (x y : ℝ) : Prop :=
  2*x - y - 2 = 0

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y - 5 = k*(x + 2)

-- Main theorem
theorem circle_intersection_slope_range :
  ∀ k : ℝ,
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    -- C passes through M(-3,3) and N(1,-5)
    circle_C (-3) 3 ∧ circle_C 1 (-5) ∧
    -- Center of C lies on the given line
    ∃ xc yc : ℝ, circle_C xc yc ∧ center_line xc yc ∧
    -- l intersects C at two distinct points
    x₁ ≠ x₂ ∧ circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧ line_l k x₂ y₂ ∧
    -- l passes through (-2,5)
    line_l k (-2) 5 ∧
    -- k > 0
    k > 0) →
  k > 15/8 :=
by sorry


end NUMINAMATH_CALUDE_circle_intersection_slope_range_l1599_159903


namespace NUMINAMATH_CALUDE_tom_books_count_l1599_159984

/-- Given that Joan has 10 books and the total number of books is 48,
    prove that Tom has 38 books. -/
theorem tom_books_count (joan_books : ℕ) (total_books : ℕ) (tom_books : ℕ) 
    (h1 : joan_books = 10)
    (h2 : total_books = 48)
    (h3 : tom_books + joan_books = total_books) : 
  tom_books = 38 := by
sorry

end NUMINAMATH_CALUDE_tom_books_count_l1599_159984


namespace NUMINAMATH_CALUDE_assignment_schemes_proof_l1599_159953

/-- The number of ways to assign 3 out of 5 volunteers to 3 distinct tasks -/
def assignment_schemes : ℕ := 60

/-- The total number of volunteers -/
def total_volunteers : ℕ := 5

/-- The number of volunteers to be selected -/
def selected_volunteers : ℕ := 3

/-- The number of tasks -/
def num_tasks : ℕ := 3

theorem assignment_schemes_proof :
  assignment_schemes = (total_volunteers.factorial) / ((total_volunteers - selected_volunteers).factorial) :=
by sorry

end NUMINAMATH_CALUDE_assignment_schemes_proof_l1599_159953


namespace NUMINAMATH_CALUDE_area_common_triangles_circle_l1599_159931

/-- The area of the region common to two inscribed equilateral triangles and an inscribed circle in a square -/
theorem area_common_triangles_circle (square_side : ℝ) (triangle_side : ℝ) (circle_radius : ℝ) : ℝ :=
  by
  -- Given conditions
  have h1 : square_side = 4 := by sorry
  have h2 : triangle_side = square_side := by sorry
  have h3 : circle_radius = square_side / 2 := by sorry
  
  -- Approximate area calculation
  have triangle_area : ℝ := by sorry
  have circle_area : ℝ := by sorry
  have overlap_per_triangle : ℝ := by sorry
  have total_overlap : ℝ := by sorry
  
  -- Prove the approximate area is 4π
  sorry

#check area_common_triangles_circle

end NUMINAMATH_CALUDE_area_common_triangles_circle_l1599_159931


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_l1599_159980

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | |x - 4| ≤ 2}
def B : Set ℝ := {x : ℝ | (5 - x) / (x + 1) > 0}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}

-- Theorem 1: A ∩ (Uᶜ B) = [5,6]
theorem intersection_A_complement_B :
  A ∩ (Set.univ \ B) = Set.Icc 5 6 := by sorry

-- Theorem 2: If A ∩ C ≠ ∅, then a ∈ (2, +∞)
theorem range_of_a (a : ℝ) (h : (A ∩ C a).Nonempty) :
  a ∈ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_range_of_a_l1599_159980


namespace NUMINAMATH_CALUDE_find_x_l1599_159908

theorem find_x : ∃ x : ℝ, 
  (∃ y : ℝ, y = 1.5 * x ∧ 0.5 * x - 10 = 0.25 * y) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1599_159908


namespace NUMINAMATH_CALUDE_min_socks_for_ten_pairs_l1599_159920

/-- Represents the number of colors of socks in the drawer -/
def num_colors : ℕ := 4

/-- Represents the number of pairs we want to ensure -/
def required_pairs : ℕ := 10

/-- Calculates the minimum number of socks needed to ensure the required number of pairs -/
def min_socks (colors : ℕ) (pairs : ℕ) : ℕ :=
  3 + 2 * pairs

/-- Theorem stating that the minimum number of socks needed to ensure 10 pairs from 4 colors is 23 -/
theorem min_socks_for_ten_pairs : 
  min_socks num_colors required_pairs = 23 := by
  sorry

#eval min_socks num_colors required_pairs

end NUMINAMATH_CALUDE_min_socks_for_ten_pairs_l1599_159920


namespace NUMINAMATH_CALUDE_parabola_equation_l1599_159905

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define points and vectors
variable (F A B C : ℝ × ℝ)  -- Points as pairs of real numbers
variable (AF FB BA BC : ℝ × ℝ)  -- Vectors as pairs of real numbers

-- Define vector operations
def vector_equal (v w : ℝ × ℝ) : Prop := v.1 = w.1 ∧ v.2 = w.2
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- State the theorem
theorem parabola_equation (p : ℝ) :
  parabola p A.1 A.2 →  -- A is on the parabola
  vector_equal AF FB →  -- AF = FB
  dot_product BA BC = 48 →  -- BA · BC = 48
  p = 2 ∧ parabola 2 A.1 A.2  -- The parabola equation is y² = 4x
  := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1599_159905


namespace NUMINAMATH_CALUDE_gingers_size_l1599_159902

theorem gingers_size (anna_size becky_size ginger_size : ℕ) : 
  anna_size = 2 →
  becky_size = 3 * anna_size →
  ginger_size = 2 * becky_size - 4 →
  ginger_size = 8 := by
sorry

end NUMINAMATH_CALUDE_gingers_size_l1599_159902


namespace NUMINAMATH_CALUDE_negation_of_implication_l1599_159941

theorem negation_of_implication (x : ℝ) : 
  ¬(x^2 = 1 → x = 1 ∨ x = -1) ↔ (x^2 ≠ 1 → x ≠ 1 ∧ x ≠ -1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l1599_159941


namespace NUMINAMATH_CALUDE_sequence_length_problem_solution_l1599_159933

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (fun i => a₁ + d * i)

theorem sequence_length (a₁ a_n d : ℤ) (h : d ≠ 0) :
  ∃ n : ℕ, arithmetic_sequence a₁ d n = List.reverse (arithmetic_sequence a_n (-d) n) ∧
           a₁ = a_n + (n - 1) * d :=
by sorry

theorem problem_solution :
  let a₁ := 160
  let a_n := 28
  let d := -4
  ∃ n : ℕ, arithmetic_sequence a₁ d n = List.reverse (arithmetic_sequence a_n (-d) n) ∧
           n = 34 :=
by sorry

end NUMINAMATH_CALUDE_sequence_length_problem_solution_l1599_159933


namespace NUMINAMATH_CALUDE_sum_squares_inequality_l1599_159912

theorem sum_squares_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) :
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_inequality_l1599_159912


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1599_159925

/-- Calculates the compound interest earned given the initial principal, interest rate, 
    compounding frequency, time, and final amount -/
def compound_interest_earned (principal : ℝ) (rate : ℝ) (n : ℝ) (time : ℝ) (final_amount : ℝ) : ℝ :=
  final_amount - principal

/-- Theorem stating that for an investment with 8% annual interest rate compounded annually 
    for 2 years, resulting in a total of 19828.80, the interest earned is 2828.80 -/
theorem compound_interest_problem :
  ∃ (principal : ℝ),
    principal > 0 ∧
    (principal * (1 + 0.08)^2 = 19828.80) ∧
    (compound_interest_earned principal 0.08 1 2 19828.80 = 2828.80) := by
  sorry


end NUMINAMATH_CALUDE_compound_interest_problem_l1599_159925


namespace NUMINAMATH_CALUDE_g_of_3_l1599_159983

def g (x : ℝ) : ℝ := 3 * x^3 + 5 * x^2 - 2 * x - 7

theorem g_of_3 : g 3 = 113 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l1599_159983


namespace NUMINAMATH_CALUDE_coach_sunscreen_fraction_is_correct_l1599_159960

/-- The fraction of sunscreen transferred to a person's forehead when heading the ball -/
def transfer_fraction : ℚ := 1 / 10

/-- The fraction of sunscreen remaining on the ball after a header -/
def remaining_fraction : ℚ := 1 - transfer_fraction

/-- The sequence of headers -/
inductive Header
| C : Header  -- Coach
| A : Header  -- Player A
| B : Header  -- Player B

/-- The repeating sequence of headers -/
def header_sequence : List Header := [Header.C, Header.A, Header.C, Header.B]

/-- The fraction of original sunscreen on Coach C's forehead after infinite headers -/
def coach_sunscreen_fraction : ℚ := 10 / 19

/-- Theorem stating that the fraction of original sunscreen on Coach C's forehead
    after infinite headers is 10/19 -/
theorem coach_sunscreen_fraction_is_correct :
  coach_sunscreen_fraction = 
    (transfer_fraction * (1 / (1 - remaining_fraction^2))) := by sorry

end NUMINAMATH_CALUDE_coach_sunscreen_fraction_is_correct_l1599_159960


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1599_159951

theorem complex_equation_solution (x : ℂ) : 5 - 2 * Complex.I * x = 7 - 5 * Complex.I * x ↔ x = (2 * Complex.I) / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1599_159951


namespace NUMINAMATH_CALUDE_rebecca_eggs_count_l1599_159946

/-- Given that Rebecca wants to split eggs into 3 groups with 5 eggs in each group,
    prove that the total number of eggs is 15. -/
theorem rebecca_eggs_count :
  let num_groups : ℕ := 3
  let eggs_per_group : ℕ := 5
  num_groups * eggs_per_group = 15 := by sorry

end NUMINAMATH_CALUDE_rebecca_eggs_count_l1599_159946


namespace NUMINAMATH_CALUDE_count_five_or_six_base_eight_l1599_159924

/-- 
Given a positive integer n and a base b, returns true if n (when expressed in base b)
contains at least one digit that is either 5 or 6.
-/
def contains_five_or_six (n : ℕ+) (b : ℕ) : Prop := sorry

/-- 
Counts the number of positive integers up to n (inclusive) that contain
at least one 5 or 6 when expressed in base b.
-/
def count_with_five_or_six (n : ℕ) (b : ℕ) : ℕ := sorry

/-- 
Theorem: The number of integers from 1 to 256 (inclusive) in base 8
that contain at least one 5 or 6 digit is equal to 220.
-/
theorem count_five_or_six_base_eight : 
  count_with_five_or_six 256 8 = 220 := by sorry

end NUMINAMATH_CALUDE_count_five_or_six_base_eight_l1599_159924


namespace NUMINAMATH_CALUDE_problem_statement_l1599_159958

theorem problem_statement : (π - 3.14) ^ 0 + (-0.125) ^ 2008 * 8 ^ 2008 = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1599_159958


namespace NUMINAMATH_CALUDE_emu_count_correct_l1599_159935

/-- Represents the number of emus in Farmer Brown's flock -/
def num_emus : ℕ := 20

/-- Represents the total number of heads and legs in the flock -/
def total_heads_and_legs : ℕ := 60

/-- Represents the number of parts (head + legs) per emu -/
def parts_per_emu : ℕ := 3

/-- Theorem stating that the number of emus is correct given the total number of heads and legs -/
theorem emu_count_correct : num_emus * parts_per_emu = total_heads_and_legs := by
  sorry

end NUMINAMATH_CALUDE_emu_count_correct_l1599_159935


namespace NUMINAMATH_CALUDE_three_number_sum_l1599_159975

theorem three_number_sum : ∀ (a b c : ℝ),
  b = 150 →
  a = 2 * b →
  c = a / 3 →
  a + b + c = 550 := by
sorry

end NUMINAMATH_CALUDE_three_number_sum_l1599_159975


namespace NUMINAMATH_CALUDE_basement_bulbs_l1599_159967

def light_bulbs_problem (bedroom bathroom kitchen basement garage : ℕ) : Prop :=
  bedroom = 2 ∧
  bathroom = 1 ∧
  kitchen = 1 ∧
  garage = basement / 2 ∧
  bedroom + bathroom + kitchen + basement + garage = 12

theorem basement_bulbs :
  ∃ (bedroom bathroom kitchen basement garage : ℕ),
    light_bulbs_problem bedroom bathroom kitchen basement garage ∧
    basement = 5 := by
  sorry

end NUMINAMATH_CALUDE_basement_bulbs_l1599_159967


namespace NUMINAMATH_CALUDE_flowchart_output_l1599_159985

def swap_operation (a b c : ℕ) : ℕ × ℕ × ℕ := 
  let (a', c') := (c, a)
  let (b', c'') := (c', b)
  (a', b', c'')

theorem flowchart_output (a b c : ℕ) (h1 : a = 21) (h2 : b = 32) (h3 : c = 75) :
  swap_operation a b c = (75, 21, 32) := by
  sorry

end NUMINAMATH_CALUDE_flowchart_output_l1599_159985


namespace NUMINAMATH_CALUDE_race_outcomes_six_participants_l1599_159936

/-- The number of different 1st-2nd-3rd-4th place outcomes in a race with 6 participants and no ties -/
def race_outcomes (n : ℕ) : ℕ :=
  if n ≥ 4 then n * (n - 1) * (n - 2) * (n - 3) else 0

/-- Theorem: The number of different 1st-2nd-3rd-4th place outcomes in a race with 6 participants and no ties is 360 -/
theorem race_outcomes_six_participants : race_outcomes 6 = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_six_participants_l1599_159936


namespace NUMINAMATH_CALUDE_evaluate_expression_l1599_159909

theorem evaluate_expression : -25 + 5 * (4^2 / 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1599_159909


namespace NUMINAMATH_CALUDE_mosaic_tiles_l1599_159976

/-- Calculates the number of square tiles needed to cover a rectangular area -/
def tilesNeeded (height_feet width_feet tile_side_inches : ℕ) : ℕ :=
  (height_feet * 12 * width_feet * 12) / (tile_side_inches * tile_side_inches)

/-- Theorem stating the number of 1-inch square tiles needed for a 10ft by 15ft mosaic -/
theorem mosaic_tiles : tilesNeeded 10 15 1 = 21600 := by
  sorry

end NUMINAMATH_CALUDE_mosaic_tiles_l1599_159976


namespace NUMINAMATH_CALUDE_always_separable_l1599_159992

/-- Represents a cell in the square -/
structure Cell where
  x : Nat
  y : Nat

/-- Represents a square of size 2n × 2n -/
structure Square (n : Nat) where
  size : Nat := 2 * n

/-- Represents a cut in the square -/
inductive Cut
  | Vertical : Nat → Cut
  | Horizontal : Nat → Cut

/-- Checks if two cells are separated by a cut -/
def separatedByCut (c1 c2 : Cell) (cut : Cut) : Prop :=
  match cut with
  | Cut.Vertical x => (c1.x ≤ x ∧ c2.x > x) ∨ (c1.x > x ∧ c2.x ≤ x)
  | Cut.Horizontal y => (c1.y ≤ y ∧ c2.y > y) ∨ (c1.y > y ∧ c2.y ≤ y)

/-- Main theorem: There always exists a cut that separates any two colored cells -/
theorem always_separable (n : Nat) (c1 c2 : Cell) 
    (h1 : c1.x < 2 * n ∧ c1.y < 2 * n)
    (h2 : c2.x < 2 * n ∧ c2.y < 2 * n)
    (h3 : c1 ≠ c2) :
    ∃ (cut : Cut), separatedByCut c1 c2 cut :=
  sorry


end NUMINAMATH_CALUDE_always_separable_l1599_159992


namespace NUMINAMATH_CALUDE_burger_cost_proof_l1599_159937

def total_cost : ℝ := 15
def fries_cost : ℝ := 2
def fries_quantity : ℕ := 2
def salad_cost_multiplier : ℕ := 3

theorem burger_cost_proof :
  let salad_cost := salad_cost_multiplier * fries_cost
  let fries_total_cost := fries_quantity * fries_cost
  let burger_cost := total_cost - (salad_cost + fries_total_cost)
  burger_cost = 5 := by sorry

end NUMINAMATH_CALUDE_burger_cost_proof_l1599_159937


namespace NUMINAMATH_CALUDE_parabola_directrix_l1599_159957

/-- The directrix of the parabola y = (x^2 - 4x + 4) / 8 is y = -1/4 -/
theorem parabola_directrix :
  let f : ℝ → ℝ := λ x => (x^2 - 4*x + 4) / 8
  ∃ (directrix : ℝ), directrix = -1/4 ∧
    ∀ (x y : ℝ), y = f x → 
      ∃ (focus : ℝ × ℝ), (x - focus.1)^2 + (y - focus.2)^2 = (y - directrix)^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1599_159957


namespace NUMINAMATH_CALUDE_minibus_students_l1599_159917

theorem minibus_students (boys : ℕ) (girls : ℕ) : 
  boys = 8 →
  girls = boys + 2 →
  boys + girls = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_minibus_students_l1599_159917


namespace NUMINAMATH_CALUDE_inequality_solution_l1599_159982

open Set
open Function
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the condition f'(x) > f(x)
variable (h : ∀ x : ℝ, deriv f x > f x)

-- Define the solution set
def solution_set := {x : ℝ | Real.exp (f (Real.log x)) - x * f 1 < 0}

-- Theorem statement
theorem inequality_solution :
  solution_set f = Ioo 0 (Real.exp 1) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l1599_159982


namespace NUMINAMATH_CALUDE_barycentric_geometry_l1599_159986

/-- Barycentric coordinates in a triangle --/
structure BarycentricCoord where
  α : ℝ
  β : ℝ
  γ : ℝ

/-- Definition of a line in barycentric coordinates --/
def is_line (f : BarycentricCoord → ℝ) : Prop :=
  ∃ u v w : ℝ, ∀ p : BarycentricCoord, f p = u * p.α + v * p.β + w * p.γ

/-- Definition of a circle in barycentric coordinates --/
def is_circle (f : BarycentricCoord → ℝ) : Prop :=
  ∃ a b c u v w : ℝ, ∀ p : BarycentricCoord,
    f p = -a^2 * p.β * p.γ - b^2 * p.γ * p.α - c^2 * p.α * p.β + 
          (u * p.α + v * p.β + w * p.γ) * (p.α + p.β + p.γ)

theorem barycentric_geometry :
  ∀ A : BarycentricCoord,
  (∃ f : BarycentricCoord → ℝ, is_line f ∧ ∀ p : BarycentricCoord, f p = p.β * w - p.γ * v) ∧
  (∃ g : BarycentricCoord → ℝ, is_line g) ∧
  (∃ h : BarycentricCoord → ℝ, is_circle h) :=
by sorry

end NUMINAMATH_CALUDE_barycentric_geometry_l1599_159986


namespace NUMINAMATH_CALUDE_single_burger_cost_l1599_159915

/-- Proves that the cost of a single burger is $1.00 given the specified conditions -/
theorem single_burger_cost
  (total_spent : ℝ)
  (total_hamburgers : ℕ)
  (double_burgers : ℕ)
  (double_burger_cost : ℝ)
  (h1 : total_spent = 74.50)
  (h2 : total_hamburgers = 50)
  (h3 : double_burgers = 49)
  (h4 : double_burger_cost = 1.50) :
  total_spent - (double_burgers * double_burger_cost) = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_single_burger_cost_l1599_159915


namespace NUMINAMATH_CALUDE_recurring_decimal_subtraction_l1599_159918

theorem recurring_decimal_subtraction : 
  (1 : ℚ) / 3 - (2 : ℚ) / 99 = (31 : ℚ) / 99 := by sorry

end NUMINAMATH_CALUDE_recurring_decimal_subtraction_l1599_159918


namespace NUMINAMATH_CALUDE_complex_modulus_l1599_159990

theorem complex_modulus (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : 
  Complex.abs z = 2 * Real.sqrt 313 / 13 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_l1599_159990


namespace NUMINAMATH_CALUDE_fraction_order_l1599_159987

theorem fraction_order : 
  let f1 := 21 / 16
  let f2 := 25 / 19
  let f3 := 23 / 17
  let f4 := 27 / 20
  f1 < f2 ∧ f2 < f4 ∧ f4 < f3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l1599_159987


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1599_159969

theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + (a - 1)*x + 25 = (x + b)^2) → (a = 11 ∨ a = -9) :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1599_159969


namespace NUMINAMATH_CALUDE_tangent_line_product_l1599_159926

/-- A cubic function with parameters a and b -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x + b

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

theorem tangent_line_product (a b : ℝ) (h1 : a ≠ 0) :
  f_derivative a 2 = 0 ∧ f a b 2 = 8 → a * b = 128 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_product_l1599_159926


namespace NUMINAMATH_CALUDE_f_simplification_and_result_l1599_159965

noncomputable def f (α : ℝ) : ℝ :=
  (Real.tan (-α - Real.pi) * Real.sin (-α - Real.pi) ^ 2) /
  (Real.sin (α - Real.pi / 2) * Real.cos (Real.pi / 2 + α) * Real.tan (Real.pi - α))

theorem f_simplification_and_result (α : ℝ) :
  f α = Real.tan α ∧
  (f α = 2 → (3 * Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 7/3) :=
by sorry

end NUMINAMATH_CALUDE_f_simplification_and_result_l1599_159965


namespace NUMINAMATH_CALUDE_function_not_in_first_quadrant_l1599_159966

theorem function_not_in_first_quadrant (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : b < -1) :
  ∀ x y : ℝ, x > 0 → y > 0 → a^x + b < y :=
by sorry

end NUMINAMATH_CALUDE_function_not_in_first_quadrant_l1599_159966


namespace NUMINAMATH_CALUDE_min_value_on_interval_l1599_159919

def f (x : ℝ) := x^2 - x

theorem min_value_on_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 0 1 ∧ f c = -1/4 ∧ ∀ x ∈ Set.Icc 0 1, f x ≥ f c := by
  sorry

end NUMINAMATH_CALUDE_min_value_on_interval_l1599_159919


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1599_159970

theorem inequality_solution_set 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a < 1) 
  (h3 : ∀ x : ℝ, x^2 - 2*a*x + a > 0) : 
  {x : ℝ | a^(x^2 - 3) < a^(2*x) ∧ a^(2*x) < 1} = {x : ℝ | x > 3} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1599_159970


namespace NUMINAMATH_CALUDE_log_579_between_consecutive_integers_l1599_159923

theorem log_579_between_consecutive_integers : 
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 579 / Real.log 10 ∧ Real.log 579 / Real.log 10 < b ∧ a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_log_579_between_consecutive_integers_l1599_159923


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1599_159928

theorem sufficient_not_necessary :
  (∀ x : ℝ, x < Real.sqrt 2 → 2 * x < 3) ∧
  ¬(∀ x : ℝ, 2 * x < 3 → x < Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1599_159928


namespace NUMINAMATH_CALUDE_volume_of_solid_l1599_159988

/-- Volume of a solid with specific dimensions -/
theorem volume_of_solid (a : ℝ) (h1 : a = 3 * Real.sqrt 2) : 
  2 * a^3 = 108 * Real.sqrt 2 := by
  sorry

#check volume_of_solid

end NUMINAMATH_CALUDE_volume_of_solid_l1599_159988


namespace NUMINAMATH_CALUDE_six_women_four_men_arrangements_l1599_159932

/-- The number of ways to arrange n indistinguishable objects of one type
    and m indistinguishable objects of another type in a row,
    such that no two objects of the same type are adjacent -/
def alternating_arrangements (n m : ℕ) : ℕ := sorry

/-- Theorem stating that there are 6 ways to arrange 6 women and 4 men
    alternately in a row -/
theorem six_women_four_men_arrangements :
  alternating_arrangements 6 4 = 6 := by sorry

end NUMINAMATH_CALUDE_six_women_four_men_arrangements_l1599_159932


namespace NUMINAMATH_CALUDE_smallest_three_digit_candy_count_l1599_159948

theorem smallest_three_digit_candy_count (n : ℕ) : 
  (100 ≤ n ∧ n < 1000) →  -- n is a three-digit number
  ((n + 7) % 9 = 0) →     -- if Alicia gains 7 candies, she'll have a multiple of 9
  ((n - 9) % 7 = 0) →     -- if Alicia loses 9 candies, she'll have a multiple of 7
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0) → False) →  -- n is the smallest such number
  n = 101 :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_candy_count_l1599_159948


namespace NUMINAMATH_CALUDE_two_digit_numbers_satisfying_condition_l1599_159978

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def satisfies_condition (n : ℕ) : Prop :=
  (sum_of_digits n)^2 = sum_of_digits (n^2)

theorem two_digit_numbers_satisfying_condition :
  {n : ℕ | is_two_digit n ∧ satisfies_condition n} =
  {10, 11, 12, 13, 20, 21, 22, 30, 31} := by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_satisfying_condition_l1599_159978


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1599_159973

theorem cubic_equation_solution :
  ∀ x : ℝ, x^3 + (x + 1)^3 + (x + 2)^3 = (x + 3)^3 ↔ x = 3 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1599_159973


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l1599_159921

-- Define the distance-time function
def s (t : ℝ) : ℝ := 4 * t^2 - 3

-- State the theorem
theorem instantaneous_velocity_at_5 :
  (deriv s) 5 = 40 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_5_l1599_159921


namespace NUMINAMATH_CALUDE_part_one_part_two_l1599_159906

/-- The absolute value function -/
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

/-- Part I: Range of a such that f(x) ≤ 3 for all x in [-1, 3] -/
theorem part_one (a : ℝ) : 
  (∀ x : ℝ, x ∈ [-1, 3] → f a x ≤ 3) ↔ a ∈ Set.Icc 0 2 := by sorry

/-- Part II: Minimum value of a such that f(x-a) + f(x+a) ≥ 1-2a for all x -/
theorem part_two : 
  (∃ a : ℝ, (∀ x : ℝ, f a (x-a) + f a (x+a) ≥ 1-2*a) ∧ 
   (∀ b : ℝ, (∀ x : ℝ, f b (x-b) + f b (x+b) ≥ 1-2*b) → a ≤ b)) ∧
  (let a := (1/4 : ℝ); ∀ x : ℝ, f a (x-a) + f a (x+a) ≥ 1-2*a) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1599_159906


namespace NUMINAMATH_CALUDE_unique_solutions_l1599_159964

/-- A triple of strictly positive integers (a, b, p) satisfies the equation if a^p = b! + p and p is prime. -/
def SatisfiesEquation (a b p : ℕ+) : Prop :=
  a ^ p.val = Nat.factorial b.val + p.val ∧ Nat.Prime p.val

theorem unique_solutions :
  ∀ a b p : ℕ+, SatisfiesEquation a b p →
    ((a = 2 ∧ b = 2 ∧ p = 2) ∨ (a = 3 ∧ b = 4 ∧ p = 3)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l1599_159964


namespace NUMINAMATH_CALUDE_river_speed_l1599_159929

/-- The speed of a river given certain rowing conditions -/
theorem river_speed (man_speed : ℝ) (total_time : ℝ) (total_distance : ℝ) : 
  man_speed = 8 → 
  total_time = 1 → 
  total_distance = 7.5 → 
  ∃ (river_speed : ℝ), 
    river_speed = 2 ∧ 
    total_distance / (man_speed - river_speed) + total_distance / (man_speed + river_speed) = total_time :=
by sorry

end NUMINAMATH_CALUDE_river_speed_l1599_159929


namespace NUMINAMATH_CALUDE_max_min_difference_l1599_159971

theorem max_min_difference (a b : ℝ) : 
  a^2 + b^2 - 2*a - 4 = 0 → 
  (∃ (t_max t_min : ℝ), 
    (∀ t : ℝ, (∃ a' b' : ℝ, a'^2 + b'^2 - 2*a' - 4 = 0 ∧ t = 2*a' - b') → t_min ≤ t ∧ t ≤ t_max) ∧
    t_max - t_min = 10) :=
by sorry

end NUMINAMATH_CALUDE_max_min_difference_l1599_159971


namespace NUMINAMATH_CALUDE_paper_division_l1599_159938

/-- Represents the number of pieces after n divisions -/
def pieces (n : ℕ) : ℕ := 3 * n + 1

/-- The main theorem about paper division -/
theorem paper_division :
  (∀ n : ℕ, pieces n = 3 * n + 1) ∧
  (∃ n : ℕ, pieces n = 2011) :=
by sorry

end NUMINAMATH_CALUDE_paper_division_l1599_159938


namespace NUMINAMATH_CALUDE_year_2023_ad_representation_l1599_159942

/-- Represents a year in the Gregorian calendar. -/
structure Year where
  value : Int
  is_ad : Bool

/-- Converts a Year to its numerical representation. -/
def Year.to_int (y : Year) : Int :=
  if y.is_ad then y.value else -y.value

/-- The year 500 BC -/
def year_500_bc : Year := { value := 500, is_ad := false }

/-- The year 2023 AD -/
def year_2023_ad : Year := { value := 2023, is_ad := true }

/-- Theorem stating that given 500 BC is denoted as -500, 2023 AD is denoted as +2023 -/
theorem year_2023_ad_representation :
  (year_500_bc.to_int = -500) → (year_2023_ad.to_int = 2023) := by
  sorry

end NUMINAMATH_CALUDE_year_2023_ad_representation_l1599_159942


namespace NUMINAMATH_CALUDE_salary_increase_l1599_159922

/-- Regression equation for monthly salary based on labor productivity -/
def salary_equation (x : ℝ) : ℝ := 50 + 80 * x

/-- Theorem stating that an increase of 1000 yuan in labor productivity
    results in an increase of 80 yuan in salary -/
theorem salary_increase (x : ℝ) :
  salary_equation (x + 1) - salary_equation x = 80 := by
  sorry

#check salary_increase

end NUMINAMATH_CALUDE_salary_increase_l1599_159922


namespace NUMINAMATH_CALUDE_committee_formation_count_l1599_159954

def total_members : ℕ := 25
def male_members : ℕ := 15
def female_members : ℕ := 10
def committee_size : ℕ := 5
def min_females : ℕ := 2

theorem committee_formation_count : 
  (Finset.sum (Finset.range (committee_size - min_females + 1))
    (fun k => Nat.choose female_members (k + min_females) * 
              Nat.choose male_members (committee_size - k - min_females))) = 36477 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l1599_159954


namespace NUMINAMATH_CALUDE_treasure_points_l1599_159952

theorem treasure_points (total_treasures : ℕ) (total_score : ℕ) 
  (h1 : total_treasures = 7) (h2 : total_score = 35) : 
  (total_score / total_treasures : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_treasure_points_l1599_159952


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1599_159910

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℤ) :
  total_questions = 120 →
  correct_answers = 75 →
  total_marks = 180 →
  (∃ (score_per_correct : ℕ),
    score_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧
    score_per_correct = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1599_159910


namespace NUMINAMATH_CALUDE_window_width_is_four_l1599_159997

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangular surface -/
def rectangleArea (d : Dimensions) : ℝ := d.length * d.width

/-- Calculates the perimeter of a rectangular surface -/
def rectanglePerimeter (d : Dimensions) : ℝ := 2 * (d.length + d.width)

/-- Represents the properties of the room and whitewashing job -/
structure RoomProperties where
  roomDimensions : Dimensions
  doorDimensions : Dimensions
  windowHeight : ℝ
  numWindows : ℕ
  costPerSquareFoot : ℝ
  totalCost : ℝ

/-- Theorem: The width of each window is 4 feet -/
theorem window_width_is_four (props : RoomProperties) 
  (h1 : props.roomDimensions = ⟨25, 15, 12⟩)
  (h2 : props.doorDimensions = ⟨6, 3, 0⟩)
  (h3 : props.windowHeight = 3)
  (h4 : props.numWindows = 3)
  (h5 : props.costPerSquareFoot = 8)
  (h6 : props.totalCost = 7248) : 
  ∃ w : ℝ, w = 4 ∧ 
    props.totalCost = props.costPerSquareFoot * 
      (rectanglePerimeter props.roomDimensions * props.roomDimensions.height - 
       rectangleArea props.doorDimensions - 
       props.numWindows * (w * props.windowHeight)) := by
  sorry

end NUMINAMATH_CALUDE_window_width_is_four_l1599_159997
