import Mathlib

namespace NUMINAMATH_CALUDE_lcm_product_geq_lcm_square_l651_65187

theorem lcm_product_geq_lcm_square (k m n : ℕ) :
  Nat.lcm (Nat.lcm k m) n * Nat.lcm (Nat.lcm m n) k * Nat.lcm (Nat.lcm n k) m ≥ (Nat.lcm (Nat.lcm k m) n)^2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_product_geq_lcm_square_l651_65187


namespace NUMINAMATH_CALUDE_solution_set_of_f_geq_1_l651_65116

def f (x : ℝ) : ℝ := |x - 1| - |x - 2|

theorem solution_set_of_f_geq_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_f_geq_1_l651_65116


namespace NUMINAMATH_CALUDE_unique_six_digit_number_l651_65188

theorem unique_six_digit_number : ∃! n : ℕ,
  (100000 ≤ n ∧ n < 1000000) ∧  -- 6-digit number
  (n % 10 = 2 ∧ n / 100000 = 2) ∧  -- begins and ends with 2
  (∃ k : ℕ, n = (2*k - 2) * (2*k) * (2*k + 2)) ∧  -- product of three consecutive even integers
  n = 287232 :=
by sorry

end NUMINAMATH_CALUDE_unique_six_digit_number_l651_65188


namespace NUMINAMATH_CALUDE_absolute_value_equation_l651_65150

theorem absolute_value_equation (a : ℝ) : 
  |2*a + 1| = 3*|a| - 2 → a = -1 ∨ a = 3 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_equation_l651_65150


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l651_65177

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 2 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ line A.1 A.2 ∧ line B.1 B.2

-- Define the midpoint condition
def midpoint_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1/2

-- Main theorem
theorem ellipse_line_intersection :
  ∀ A B : ℝ × ℝ,
  intersection_points A B →
  midpoint_condition A B →
  (∀ x y : ℝ, line x y ↔ x + 2*y - 2 = 0) ∧
  (Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l651_65177


namespace NUMINAMATH_CALUDE_f_extrema_l651_65124

def f (x : ℝ) := x^2 - 2*x + 2

def A₁ : Set ℝ := Set.Icc (-2) 0
def A₂ : Set ℝ := Set.Icc 2 3

theorem f_extrema :
  (∀ x ∈ A₁, f x ≤ 10 ∧ f x ≥ 2) ∧
  (∃ x₁ ∈ A₁, f x₁ = 10) ∧
  (∃ x₂ ∈ A₁, f x₂ = 2) ∧
  (∀ x ∈ A₂, f x ≤ 5 ∧ f x ≥ 2) ∧
  (∃ x₃ ∈ A₂, f x₃ = 5) ∧
  (∃ x₄ ∈ A₂, f x₄ = 2) :=
sorry

end NUMINAMATH_CALUDE_f_extrema_l651_65124


namespace NUMINAMATH_CALUDE_combined_boys_avg_is_67_l651_65148

/-- Represents a high school with average scores for boys, girls, and combined --/
structure School where
  boys_avg : ℝ
  girls_avg : ℝ
  combined_avg : ℝ

/-- Represents the combined data for two schools --/
structure CombinedSchools where
  school1 : School
  school2 : School
  combined_girls_avg : ℝ

/-- Calculates the combined average score for boys given two schools --/
def combined_boys_avg (schools : CombinedSchools) : ℝ :=
  -- Implementation details are omitted
  sorry

/-- Theorem stating that the combined average score for boys is 67 --/
theorem combined_boys_avg_is_67 (schools : CombinedSchools) 
  (h1 : schools.school1 = ⟨65, 75, 68⟩)
  (h2 : schools.school2 = ⟨70, 85, 75⟩)
  (h3 : schools.combined_girls_avg = 80) :
  combined_boys_avg schools = 67 := by
  sorry

end NUMINAMATH_CALUDE_combined_boys_avg_is_67_l651_65148


namespace NUMINAMATH_CALUDE_simple_interest_problem_l651_65171

/-- Given a sum P put at simple interest rate R for 4 years, 
    if increasing the rate by 3% results in Rs. 120 more interest, 
    then P = 1000. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (r : R > 0) :
  (P * (R + 3) * 4) / 100 - (P * R * 4) / 100 = 120 →
  P = 1000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l651_65171


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l651_65189

/-- Given a cube with a sphere inscribed within it, and another cube inscribed within that sphere,
    this theorem relates the surface area of the outer cube to the surface area of the inner cube. -/
theorem inscribed_cube_surface_area (outer_surface_area : ℝ) :
  outer_surface_area = 54 →
  ∃ (inner_surface_area : ℝ),
    inner_surface_area = 18 ∧
    (∃ (outer_side_length inner_side_length : ℝ),
      outer_surface_area = 6 * outer_side_length^2 ∧
      inner_surface_area = 6 * inner_side_length^2 ∧
      inner_side_length = outer_side_length / Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l651_65189


namespace NUMINAMATH_CALUDE_hike_length_is_83_l651_65176

/-- Represents the length of a 5-day hike satisfying specific conditions -/
def HikeLength (a b c d e : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧  -- Non-negative distances
  a + b = 36 ∧                              -- First two days
  (b + c + d) / 3 = 15 ∧                    -- Average of days 2, 3, 4
  c + d + e = 45 ∧                          -- Last three days
  a + c + e = 38                            -- Days 1, 3, 5

/-- The theorem stating that the total hike length is 83 miles -/
theorem hike_length_is_83 {a b c d e : ℝ} (h : HikeLength a b c d e) :
  a + b + c + d + e = 83 := by
  sorry


end NUMINAMATH_CALUDE_hike_length_is_83_l651_65176


namespace NUMINAMATH_CALUDE_triangle_medians_area_relationship_l651_65141

/-- Represents a triangle with three medians -/
structure Triangle where
  median1 : ℝ
  median2 : ℝ
  median3 : ℝ
  area : ℝ

/-- The theorem stating the relationship between the medians and area of the triangle -/
theorem triangle_medians_area_relationship (t : Triangle) 
  (h1 : t.median1 = 5)
  (h2 : t.median2 = 7)
  (h3 : t.area = 10 * Real.sqrt 3) :
  t.median3 = 4 * Real.sqrt 3 := by
  sorry

#check triangle_medians_area_relationship

end NUMINAMATH_CALUDE_triangle_medians_area_relationship_l651_65141


namespace NUMINAMATH_CALUDE_teenage_group_size_l651_65166

theorem teenage_group_size (total_bill : ℝ) (individual_cost : ℝ) (gratuity_rate : ℝ) :
  total_bill = 840 →
  individual_cost = 100 →
  gratuity_rate = 0.2 →
  ∃ n : ℕ, n = 7 ∧ total_bill = (individual_cost * n) * (1 + gratuity_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_teenage_group_size_l651_65166


namespace NUMINAMATH_CALUDE_power_sum_equality_l651_65115

theorem power_sum_equality : (-2)^2009 + (-2)^2010 = 2^2009 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l651_65115


namespace NUMINAMATH_CALUDE_intersection_points_correct_l651_65107

/-- The number of intersection points of line segments connecting m points on the positive X-axis
    and n points on the positive Y-axis, where no three segments intersect at the same point. -/
def intersectionPoints (m n : ℕ) : ℚ :=
  (m * (m - 1) * n * (n - 1) : ℚ) / 4

/-- Theorem stating that the number of intersection points is correct. -/
theorem intersection_points_correct (m n : ℕ) :
  intersectionPoints m n = (m * (m - 1) * n * (n - 1) : ℚ) / 4 :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_correct_l651_65107


namespace NUMINAMATH_CALUDE_inequality_solution_set_l651_65108

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x - 1) - 3 / (x - 3) + 5 / (x - 5) - 2 / (x - 7) < 1 / 15) ↔ 
  (x < -8 ∨ (-7 < x ∧ x < -1) ∨ (1 < x ∧ x < 3) ∨ (5 < x ∧ x < 7) ∨ x > 8) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l651_65108


namespace NUMINAMATH_CALUDE_alcohol_mixture_percentage_l651_65112

/-- Proves that mixing 100 mL of 10% alcohol solution with 300 mL of 30% alcohol solution 
    results in a 25% alcohol solution -/
theorem alcohol_mixture_percentage :
  let x_volume : ℝ := 100
  let x_percentage : ℝ := 10
  let y_volume : ℝ := 300
  let y_percentage : ℝ := 30
  let total_volume : ℝ := x_volume + y_volume
  let total_alcohol : ℝ := (x_volume * x_percentage + y_volume * y_percentage) / 100
  total_alcohol / total_volume * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_mixture_percentage_l651_65112


namespace NUMINAMATH_CALUDE_floor_abs_negative_real_l651_65106

theorem floor_abs_negative_real : ⌊|(-58.7 : ℝ)|⌋ = 58 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_real_l651_65106


namespace NUMINAMATH_CALUDE_complex_sum_simplification_l651_65149

theorem complex_sum_simplification : 
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2
  z₁^12 + z₂^12 = 2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_simplification_l651_65149


namespace NUMINAMATH_CALUDE_inequality_property_l651_65146

theorem inequality_property (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a * c < b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l651_65146


namespace NUMINAMATH_CALUDE_complex_equation_solution_l651_65147

open Complex

theorem complex_equation_solution :
  let z : ℂ := (1 + I^2 + 3*(1-I)) / (2+I)
  ∀ (a b : ℝ), z^2 + a*z + b = 1 + I → a = -3 ∧ b = 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l651_65147


namespace NUMINAMATH_CALUDE_average_and_difference_l651_65123

theorem average_and_difference (y : ℝ) : 
  (47 + y) / 2 = 53 → |y - 47| = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_and_difference_l651_65123


namespace NUMINAMATH_CALUDE_midpoint_distance_after_movement_l651_65180

/-- Given two points A and B in a Cartesian plane, if A moves 5 units right and 6 units up,
    and B moves 12 units left and 4 units down, then the distance between the original
    midpoint M and the new midpoint M' is √53/2. -/
theorem midpoint_distance_after_movement (p q r s : ℝ) : 
  let A : ℝ × ℝ := (p, q)
  let B : ℝ × ℝ := (r, s)
  let M : ℝ × ℝ := ((p + r) / 2, (q + s) / 2)
  let A' : ℝ × ℝ := (p + 5, q + 6)
  let B' : ℝ × ℝ := (r - 12, s - 4)
  let M' : ℝ × ℝ := ((p + 5 + r - 12) / 2, (q + 6 + s - 4) / 2)
  Real.sqrt ((M.1 - M'.1)^2 + (M.2 - M'.2)^2) = Real.sqrt 53 / 2 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_distance_after_movement_l651_65180


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l651_65190

theorem min_value_squared_sum (x y : ℝ) (h : x * y = 1) :
  x^2 + 4*y^2 ≥ 4 ∧ ∃ (a b : ℝ), a * b = 1 ∧ a^2 + 4*b^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l651_65190


namespace NUMINAMATH_CALUDE_expected_heads_is_40_l651_65151

/-- A coin toss simulation with specific rules --/
def CoinTossSimulation :=
  { n : ℕ  // n = 80 }

/-- The probability of a coin showing heads after all tosses --/
def prob_heads (c : CoinTossSimulation) : ℚ :=
  1 / 2

/-- The expected number of heads in the simulation --/
def expected_heads (c : CoinTossSimulation) : ℚ :=
  c.val * prob_heads c

/-- Theorem stating that the expected number of heads is 40 --/
theorem expected_heads_is_40 (c : CoinTossSimulation) :
  expected_heads c = 40 := by
  sorry

#check expected_heads_is_40

end NUMINAMATH_CALUDE_expected_heads_is_40_l651_65151


namespace NUMINAMATH_CALUDE_expected_participants_2008_l651_65104

/-- The number of participants in the school festival after n years, given an initial number of participants and an annual increase rate. -/
def participants_after_n_years (initial : ℝ) (rate : ℝ) (n : ℕ) : ℝ :=
  initial * (1 + rate) ^ n

/-- The expected number of participants in 2008, given the initial number in 2005 and the annual increase rate. -/
theorem expected_participants_2008 :
  participants_after_n_years 1000 0.25 3 = 1953.125 := by
  sorry

#eval participants_after_n_years 1000 0.25 3

end NUMINAMATH_CALUDE_expected_participants_2008_l651_65104


namespace NUMINAMATH_CALUDE_red_ball_certain_l651_65122

/-- Represents the number of balls of each color in the box -/
structure BallCount where
  red : Nat
  yellow : Nat

/-- Represents the number of balls drawn from the box -/
def BallsDrawn : Nat := 3

/-- The initial state of the box -/
def initialBox : BallCount where
  red := 3
  yellow := 2

/-- A function to check if drawing at least one red ball is certain -/
def isRedBallCertain (box : BallCount) : Prop :=
  box.yellow < BallsDrawn

/-- Theorem stating that drawing at least one red ball is certain -/
theorem red_ball_certain :
  isRedBallCertain initialBox := by
  sorry

end NUMINAMATH_CALUDE_red_ball_certain_l651_65122


namespace NUMINAMATH_CALUDE_hexagonal_circle_selection_l651_65183

/-- Represents the number of ways to choose three consecutive circles in a direction --/
def consecutive_triples (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total number of circles in the figure --/
def total_circles : ℕ := 33

/-- The number of circles in the longest row --/
def longest_row : ℕ := 6

/-- The number of ways to choose three consecutive circles in the first direction --/
def first_direction : ℕ := consecutive_triples longest_row

/-- The number of ways to choose three consecutive circles in each of the other two directions --/
def other_directions : ℕ := 18

/-- The total number of ways to choose three consecutive circles in all directions --/
def total_ways : ℕ := first_direction + 2 * other_directions

theorem hexagonal_circle_selection :
  total_ways = 57 :=
sorry

end NUMINAMATH_CALUDE_hexagonal_circle_selection_l651_65183


namespace NUMINAMATH_CALUDE_domino_count_for_0_to_12_l651_65197

/-- The number of tiles in a standard set of dominoes -/
def standard_domino_count : ℕ := 28

/-- The lowest value on a domino tile -/
def min_value : ℕ := 0

/-- The highest value on a domino tile in the new set -/
def max_value : ℕ := 12

/-- The number of tiles in a domino set with values from min_value to max_value -/
def domino_count (min : ℕ) (max : ℕ) : ℕ :=
  let n := max - min + 1
  (n * (n + 1)) / 2

theorem domino_count_for_0_to_12 :
  domino_count min_value max_value = 91 :=
sorry

end NUMINAMATH_CALUDE_domino_count_for_0_to_12_l651_65197


namespace NUMINAMATH_CALUDE_range_of_difference_l651_65111

theorem range_of_difference (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 1) :
  -2 < a - b ∧ a - b < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_difference_l651_65111


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l651_65164

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 3 = 17 →
  a 7 = 38 →
  a 4 + a 5 + a 6 = 93 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l651_65164


namespace NUMINAMATH_CALUDE_complex_sum_fourth_powers_l651_65135

theorem complex_sum_fourth_powers : 
  let z₁ : ℂ := (-1 + Complex.I * Real.sqrt 7) / 2
  let z₂ : ℂ := (-1 - Complex.I * Real.sqrt 7) / 2
  z₁^4 + z₂^4 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_sum_fourth_powers_l651_65135


namespace NUMINAMATH_CALUDE_skyler_song_difference_l651_65137

def composer_songs (total_songs hit_songs top_100_extra : ℕ) : Prop :=
  let top_100_songs := hit_songs + top_100_extra
  let unreleased_songs := total_songs - (hit_songs + top_100_songs)
  hit_songs - unreleased_songs = 5

theorem skyler_song_difference :
  composer_songs 80 25 10 :=
by
  sorry

end NUMINAMATH_CALUDE_skyler_song_difference_l651_65137


namespace NUMINAMATH_CALUDE_stump_pulling_force_l651_65175

/-- The force required to pull a stump varies inversely with the lever length -/
def inverse_variation (force length : ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ force * length = k

theorem stump_pulling_force 
  (force_10 length_10 force_25 length_25 : ℝ)
  (h1 : force_10 = 180)
  (h2 : length_10 = 10)
  (h3 : length_25 = 25)
  (h4 : inverse_variation force_10 length_10)
  (h5 : inverse_variation force_25 length_25)
  : force_25 = 72 := by
sorry

end NUMINAMATH_CALUDE_stump_pulling_force_l651_65175


namespace NUMINAMATH_CALUDE_train_journey_duration_l651_65185

/-- Given a train journey with a distance and average speed, calculate the duration of the journey. -/
theorem train_journey_duration (distance : ℝ) (speed : ℝ) (duration : ℝ) 
  (h_distance : distance = 27) 
  (h_speed : speed = 3) 
  (h_duration : duration = distance / speed) : 
  duration = 9 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_duration_l651_65185


namespace NUMINAMATH_CALUDE_jonessa_pay_l651_65121

theorem jonessa_pay (tax_rate : ℝ) (take_home_pay : ℝ) (total_pay : ℝ) : 
  tax_rate = 0.1 →
  take_home_pay = 450 →
  take_home_pay = total_pay * (1 - tax_rate) →
  total_pay = 500 := by
sorry

end NUMINAMATH_CALUDE_jonessa_pay_l651_65121


namespace NUMINAMATH_CALUDE_max_islands_is_36_l651_65134

/-- Represents an archipelago with islands and bridges -/
structure Archipelago where
  N : Nat
  bridges : Fin N → Fin N → Bool

/-- The number of islands is at least 7 -/
def atLeastSevenIslands (a : Archipelago) : Prop :=
  a.N ≥ 7

/-- Any two islands are connected by at most one bridge -/
def atMostOneBridge (a : Archipelago) : Prop :=
  ∀ i j : Fin a.N, i ≠ j → (a.bridges i j = true → a.bridges j i = false)

/-- No more than 5 bridges lead from each island -/
def atMostFiveBridges (a : Archipelago) : Prop :=
  ∀ i : Fin a.N, (Finset.filter (fun j => a.bridges i j) (Finset.univ : Finset (Fin a.N))).card ≤ 5

/-- Among any 7 islands, there are always two that are connected by a bridge -/
def twoConnectedInSeven (a : Archipelago) : Prop :=
  ∀ s : Finset (Fin a.N), s.card = 7 →
    ∃ i j : Fin a.N, i ∈ s ∧ j ∈ s ∧ i ≠ j ∧ a.bridges i j

/-- The maximum number of islands satisfying the conditions is 36 -/
theorem max_islands_is_36 (a : Archipelago) :
    atLeastSevenIslands a →
    atMostOneBridge a →
    atMostFiveBridges a →
    twoConnectedInSeven a →
    a.N ≤ 36 := by
  sorry

end NUMINAMATH_CALUDE_max_islands_is_36_l651_65134


namespace NUMINAMATH_CALUDE_total_flowers_l651_65182

theorem total_flowers (class_a_students class_b_students flowers_per_student : ℕ) 
  (h1 : class_a_students = 48)
  (h2 : class_b_students = 48)
  (h3 : flowers_per_student = 16) :
  class_a_students * flowers_per_student + class_b_students * flowers_per_student = 1536 := by
  sorry

end NUMINAMATH_CALUDE_total_flowers_l651_65182


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l651_65170

/-- Given a mixture of milk and water with an initial ratio of 3:2, prove that
    after adding 48 liters of water to make the new ratio 3:4, the initial
    volume of the mixture was 120 liters. -/
theorem initial_mixture_volume
  (initial_milk : ℚ) (initial_water : ℚ)
  (initial_ratio : initial_milk / initial_water = 3 / 2)
  (new_ratio : initial_milk / (initial_water + 48) = 3 / 4) :
  initial_milk + initial_water = 120 := by
sorry

end NUMINAMATH_CALUDE_initial_mixture_volume_l651_65170


namespace NUMINAMATH_CALUDE_mariela_cards_total_l651_65179

/-- Calculates the total number of cards Mariela received based on the given quantities -/
def total_cards (hospital_dozens : ℕ) (hospital_hundreds : ℕ) (home_dozens : ℕ) (home_hundreds : ℕ) : ℕ :=
  (hospital_dozens * 12 + hospital_hundreds * 100) + (home_dozens * 12 + home_hundreds * 100)

/-- Proves that Mariela received 1768 cards in total -/
theorem mariela_cards_total : total_cards 25 7 39 3 = 1768 := by
  sorry

end NUMINAMATH_CALUDE_mariela_cards_total_l651_65179


namespace NUMINAMATH_CALUDE_max_y_value_l651_65119

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -1) : y ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l651_65119


namespace NUMINAMATH_CALUDE_complex_cube_equals_negative_one_l651_65193

theorem complex_cube_equals_negative_one : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (1/2 - Complex.I * (Real.sqrt 3)/2) ^ 3 = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_cube_equals_negative_one_l651_65193


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l651_65143

theorem sqrt_sum_inequality (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
  Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
  Real.sqrt (e / (a + b + c + d)) > 2 ∧
  ∀ n : ℝ, (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + Real.sqrt (b / (a + c + d + e)) +
    Real.sqrt (c / (a + b + d + e)) + Real.sqrt (d / (a + b + c + e)) +
    Real.sqrt (e / (a + b + c + d)) > n) →
  n ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l651_65143


namespace NUMINAMATH_CALUDE_parabola_intersection_points_l651_65178

/-- The parabola y = x^2 + 2x + a - 2 has exactly two intersection points with the coordinate axes if and only if a = 2 or a = 3 -/
theorem parabola_intersection_points (a : ℝ) : 
  (∃! (x y : ℝ), y = x^2 + 2*x + a - 2 ∧ (x = 0 ∨ y = 0)) ∧ 
  (∃ (x1 x2 y1 y2 : ℝ), (x1 ≠ x2 ∨ y1 ≠ y2) ∧ 
    (y1 = x1^2 + 2*x1 + a - 2) ∧ (y2 = x2^2 + 2*x2 + a - 2) ∧ 
    ((x1 = 0 ∨ y1 = 0) ∧ (x2 = 0 ∨ y2 = 0))) ↔ 
  (a = 2 ∨ a = 3) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_points_l651_65178


namespace NUMINAMATH_CALUDE_balloon_arrangements_l651_65163

def word_length : Nat := 7
def repeated_letters : Nat := 2
def repetitions_per_letter : Nat := 2

theorem balloon_arrangements :
  (word_length.factorial) / (repeated_letters.factorial * repetitions_per_letter.factorial) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_balloon_arrangements_l651_65163


namespace NUMINAMATH_CALUDE_sets_satisfying_union_condition_l651_65105

theorem sets_satisfying_union_condition :
  ∃! (S : Finset (Finset ℕ)), 
    (∀ M ∈ S, M ∪ {1} = {1, 2, 3}) ∧ 
    (∀ M, M ∪ {1} = {1, 2, 3} → M ∈ S) ∧
    Finset.card S = 3 :=
by sorry

end NUMINAMATH_CALUDE_sets_satisfying_union_condition_l651_65105


namespace NUMINAMATH_CALUDE_legoland_animals_l651_65144

theorem legoland_animals (kangaroos koalas pandas : ℕ) : 
  kangaroos = 567 →
  kangaroos = 9 * koalas →
  koalas = 7 * pandas →
  kangaroos + koalas + pandas = 639 := by
sorry

end NUMINAMATH_CALUDE_legoland_animals_l651_65144


namespace NUMINAMATH_CALUDE_restaurant_production_june_l651_65101

theorem restaurant_production_june :
  let weekday_cheese_pizzas := 60 + 40
  let weekday_pepperoni_pizzas := 2 * weekday_cheese_pizzas
  let weekday_beef_hotdogs := 30
  let weekday_chicken_hotdogs := 30
  let weekend_cheese_pizzas := 50 + 30
  let weekend_pepperoni_pizzas := 2 * weekend_cheese_pizzas
  let weekend_beef_hotdogs := 20
  let weekend_chicken_hotdogs := 30
  let weekend_bbq_chicken_pizzas := 25
  let weekend_veggie_pizzas := 15
  let weekdays_in_june := 20
  let weekends_in_june := 10
  
  (weekday_cheese_pizzas * weekdays_in_june + weekend_cheese_pizzas * weekends_in_june = 2800) ∧
  (weekday_pepperoni_pizzas * weekdays_in_june + weekend_pepperoni_pizzas * weekends_in_june = 5600) ∧
  (weekday_beef_hotdogs * weekdays_in_june + weekend_beef_hotdogs * weekends_in_june = 800) ∧
  (weekday_chicken_hotdogs * weekdays_in_june + weekend_chicken_hotdogs * weekends_in_june = 900) ∧
  (weekend_bbq_chicken_pizzas * weekends_in_june = 250) ∧
  (weekend_veggie_pizzas * weekends_in_june = 150) := by
  sorry

end NUMINAMATH_CALUDE_restaurant_production_june_l651_65101


namespace NUMINAMATH_CALUDE_tax_difference_equals_0_625_l651_65128

/-- The price of an item before tax -/
def price : ℝ := 50

/-- The higher tax rate -/
def high_rate : ℝ := 0.075

/-- The lower tax rate -/
def low_rate : ℝ := 0.0625

/-- The difference between the two tax amounts -/
def tax_difference : ℝ := price * high_rate - price * low_rate

theorem tax_difference_equals_0_625 : tax_difference = 0.625 := by
  sorry

end NUMINAMATH_CALUDE_tax_difference_equals_0_625_l651_65128


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l651_65103

theorem quadratic_roots_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + 2 * x + 1 = 0 ∧ a * y^2 + 2 * y + 1 = 0) →
  a < 1 ∧ a ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l651_65103


namespace NUMINAMATH_CALUDE_square_area_ratio_when_doubled_l651_65142

theorem square_area_ratio_when_doubled (s : ℝ) (h : s > 0) :
  (s^2) / ((2*s)^2) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_when_doubled_l651_65142


namespace NUMINAMATH_CALUDE_log_equality_implies_ratio_l651_65129

theorem log_equality_implies_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 4 = Real.log q / Real.log 18) ∧
  (Real.log p / Real.log 4 = Real.log (p + q) / Real.log 25) →
  q / p = 2 - 2/5 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_ratio_l651_65129


namespace NUMINAMATH_CALUDE_exponent_equality_l651_65173

theorem exponent_equality (x : ℕ) : 
  2010^2011 - 2010^2009 = 2010^x * 2009 * 2011 → x = 2009 := by
  sorry

end NUMINAMATH_CALUDE_exponent_equality_l651_65173


namespace NUMINAMATH_CALUDE_proportion_solution_l651_65160

theorem proportion_solution (x : ℝ) : (0.75 / x = 5 / 11) → x = 1.65 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l651_65160


namespace NUMINAMATH_CALUDE_rectangle_max_area_l651_65158

/-- Given a rectangle with perimeter 60 meters and one side three times longer than the other,
    the maximum area is 168.75 square meters. -/
theorem rectangle_max_area (perimeter : ℝ) (ratio : ℝ) (area : ℝ) :
  perimeter = 60 →
  ratio = 3 →
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x = ratio * y ∧ 2 * (x + y) = perimeter ∧ x * y = area) →
  area = 168.75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l651_65158


namespace NUMINAMATH_CALUDE_kyle_gas_and_maintenance_l651_65191

/-- Calculates the amount left for gas and maintenance given monthly income and expenses --/
def amount_left_for_gas_and_maintenance (monthly_income : ℕ) (rent utilities retirement_savings groceries insurance misc car_payment : ℕ) : ℕ :=
  monthly_income - (rent + utilities + retirement_savings + groceries + insurance + misc + car_payment)

/-- Theorem: Kyle's amount left for gas and maintenance is $350 --/
theorem kyle_gas_and_maintenance :
  amount_left_for_gas_and_maintenance 3200 1250 150 400 300 200 200 350 = 350 := by
  sorry

end NUMINAMATH_CALUDE_kyle_gas_and_maintenance_l651_65191


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_exists_l651_65172

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define the property of a quadrilateral being circumscribed around a circle
def isCircumscribed (q : Quadrilateral) (c : Circle) : Prop :=
  sorry

-- Define the property of a point being on a circle
def isOnCircle (p : ℝ × ℝ) (c : Circle) : Prop :=
  sorry

-- Define homothety between two quadrilaterals
def isHomothetic (q1 q2 : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem inscribed_quadrilateral_exists (ABCD : Quadrilateral) (c : Circle) :
  isCircumscribed ABCD c →
  isOnCircle ABCD.A c →
  isOnCircle ABCD.B c →
  isOnCircle ABCD.C c →
  ∃ (EFGH : Quadrilateral), isCircumscribed EFGH c ∧ isHomothetic ABCD EFGH :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_exists_l651_65172


namespace NUMINAMATH_CALUDE_train_b_completion_time_l651_65192

/-- Proves that Train B takes 2 hours to complete the route given the conditions -/
theorem train_b_completion_time 
  (route_length : ℝ) 
  (train_a_speed : ℝ) 
  (meeting_distance : ℝ) 
  (h1 : route_length = 75) 
  (h2 : train_a_speed = 25) 
  (h3 : meeting_distance = 30) : 
  (route_length / ((route_length - meeting_distance) / (meeting_distance / train_a_speed))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_train_b_completion_time_l651_65192


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l651_65155

theorem greatest_integer_inequality (x : ℤ) :
  (∀ y : ℤ, 3 * y^2 - 5 * y - 2 < 4 - 2 * y → y ≤ 1) ∧
  (3 * 1^2 - 5 * 1 - 2 < 4 - 2 * 1) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l651_65155


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l651_65195

theorem multiplication_table_odd_fraction :
  let n : ℕ := 16
  let total_products : ℕ := n * n
  let odd_numbers : ℕ := (n + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  (odd_products : ℚ) / total_products = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l651_65195


namespace NUMINAMATH_CALUDE_g_neg_two_equals_fifteen_l651_65120

theorem g_neg_two_equals_fifteen :
  let g : ℝ → ℝ := λ x ↦ x^2 - 4*x + 3
  g (-2) = 15 := by sorry

end NUMINAMATH_CALUDE_g_neg_two_equals_fifteen_l651_65120


namespace NUMINAMATH_CALUDE_polynomial_one_root_product_l651_65130

theorem polynomial_one_root_product (d e : ℝ) : 
  (∃! x : ℝ, x^2 + d*x + e = 0) → 
  d = 2*e - 3 → 
  ∃ e₁ e₂ : ℝ, (∀ e' : ℝ, (∃ x : ℝ, x^2 + d*x + e' = 0) → (e' = e₁ ∨ e' = e₂)) ∧ 
              e₁ * e₂ = 9/4 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_one_root_product_l651_65130


namespace NUMINAMATH_CALUDE_magic_square_b_plus_c_l651_65102

/-- Represents a 3x3 magic square with the given layout -/
structure MagicSquare where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  S : ℕ
  row1_sum : 30 + b + 18 = S
  row2_sum : 15 + c + d = S
  row3_sum : a + 33 + e = S
  col1_sum : 30 + 15 + a = S
  col2_sum : b + c + 33 = S
  col3_sum : 18 + d + e = S
  diag1_sum : 30 + c + e = S
  diag2_sum : 18 + c + a = S

/-- The sum of b and c in a magic square is 33 -/
theorem magic_square_b_plus_c (ms : MagicSquare) : ms.b + ms.c = 33 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_b_plus_c_l651_65102


namespace NUMINAMATH_CALUDE_complex_equation_roots_l651_65109

theorem complex_equation_roots : 
  let z₁ : ℂ := -1 + Real.sqrt 5 - (2 * Real.sqrt 5 / 5) * I
  let z₂ : ℂ := -1 - Real.sqrt 5 + (2 * Real.sqrt 5 / 5) * I
  (z₁^2 + 2*z₁ = 3 - 4*I) ∧ (z₂^2 + 2*z₂ = 3 - 4*I) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l651_65109


namespace NUMINAMATH_CALUDE_absolute_value_inequalities_l651_65198

theorem absolute_value_inequalities (x y : ℝ) : 
  (abs (x + y) ≤ abs x + abs y) ∧ 
  (abs (x - y) ≥ abs x - abs y) ∧ 
  (abs (x - y) ≥ abs (abs x - abs y)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequalities_l651_65198


namespace NUMINAMATH_CALUDE_intersection_of_S_and_T_l651_65139

-- Define the sets S and T
def S : Set ℝ := {x : ℝ | x^2 + 2*x = 0}
def T : Set ℝ := {x : ℝ | x^2 - 2*x = 0}

-- State the theorem
theorem intersection_of_S_and_T : S ∩ T = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_S_and_T_l651_65139


namespace NUMINAMATH_CALUDE_page_number_digit_difference_l651_65131

/-- Counts the occurrences of a digit in a range of numbers -/
def countDigit (d : Nat) (start finish : Nat) : Nat :=
  sorry

/-- The difference between the number of 3's and 7's in page numbers of a book -/
def digitDifference (pages : Nat) : Nat :=
  (countDigit 3 1 pages) - (countDigit 7 1 pages)

theorem page_number_digit_difference :
  digitDifference 350 = 56 := by sorry

end NUMINAMATH_CALUDE_page_number_digit_difference_l651_65131


namespace NUMINAMATH_CALUDE_water_bottles_fourth_game_l651_65110

/-- Represents the number of bottles in a case -/
structure CaseSize where
  water : ℕ
  sports_drink : ℕ

/-- Represents the number of cases purchased -/
structure CasesPurchased where
  water : ℕ
  sports_drink : ℕ

/-- Represents the consumption of bottles in a game -/
structure GameConsumption where
  water : ℕ
  sports_drink : ℕ

/-- Calculates the total number of bottles initially available -/
def totalInitialBottles (caseSize : CaseSize) (casesPurchased : CasesPurchased) : ℕ × ℕ :=
  (caseSize.water * casesPurchased.water, caseSize.sports_drink * casesPurchased.sports_drink)

/-- Calculates the total consumption for the first three games -/
def totalConsumptionFirstThreeGames (game1 game2 game3 : GameConsumption) : ℕ × ℕ :=
  (game1.water + game2.water + game3.water, game1.sports_drink + game2.sports_drink + game3.sports_drink)

/-- Theorem: The number of water bottles used in the fourth game is 20 -/
theorem water_bottles_fourth_game 
  (caseSize : CaseSize)
  (casesPurchased : CasesPurchased)
  (game1 game2 game3 : GameConsumption)
  (remainingBottles : ℕ × ℕ) :
  caseSize.water = 20 →
  caseSize.sports_drink = 15 →
  casesPurchased.water = 10 →
  casesPurchased.sports_drink = 5 →
  game1 = { water := 70, sports_drink := 30 } →
  game2 = { water := 40, sports_drink := 20 } →
  game3 = { water := 50, sports_drink := 25 } →
  remainingBottles = (20, 10) →
  let (initialWater, _) := totalInitialBottles caseSize casesPurchased
  let (consumedWater, _) := totalConsumptionFirstThreeGames game1 game2 game3
  initialWater - consumedWater - remainingBottles.1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_water_bottles_fourth_game_l651_65110


namespace NUMINAMATH_CALUDE_not_divides_power_diff_l651_65181

theorem not_divides_power_diff (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) 
  (hm_odd : Odd m) (hn_odd : Odd n) : 
  ¬ ((2^m - 1) ∣ (3^n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_not_divides_power_diff_l651_65181


namespace NUMINAMATH_CALUDE_mork_and_mindy_tax_rate_l651_65126

/-- Calculates the combined tax rate for Mork and Mindy given their individual tax rates and income ratio. -/
theorem mork_and_mindy_tax_rate 
  (mork_rate : ℝ) 
  (mindy_rate : ℝ) 
  (income_ratio : ℝ) 
  (h1 : mork_rate = 0.1) 
  (h2 : mindy_rate = 0.2) 
  (h3 : income_ratio = 3) : 
  (mork_rate + mindy_rate * income_ratio) / (1 + income_ratio) = 0.175 := by
sorry

#eval (0.1 + 0.2 * 3) / (1 + 3)

end NUMINAMATH_CALUDE_mork_and_mindy_tax_rate_l651_65126


namespace NUMINAMATH_CALUDE_geometric_progression_problem_l651_65136

theorem geometric_progression_problem (b₃ b₆ : ℚ) 
  (h₁ : b₃ = -1)
  (h₂ : b₆ = 27/8) :
  ∃ (b₁ q : ℚ), 
    b₁ = -4/9 ∧ 
    q = -3/2 ∧ 
    b₃ = b₁ * q^2 ∧ 
    b₆ = b₁ * q^5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_problem_l651_65136


namespace NUMINAMATH_CALUDE_rectangle_x_coordinate_l651_65118

/-- A rectangle with vertices (1, 0), (x, 0), (1, 2), and (x, 2) is divided into two identical
    quadrilaterals by a line passing through the origin with slope 0.2.
    This theorem proves that the x-coordinate of the second and fourth vertices is 9. -/
theorem rectangle_x_coordinate (x : ℝ) :
  (∃ (l : Set (ℝ × ℝ)),
    -- Line l passes through the origin
    (0, 0) ∈ l ∧
    -- Line l has slope 0.2
    (∀ (x₁ y₁ x₂ y₂ : ℝ), (x₁, y₁) ∈ l → (x₂, y₂) ∈ l → x₁ ≠ x₂ → (y₂ - y₁) / (x₂ - x₁) = 0.2) ∧
    -- Line l divides the rectangle into two identical quadrilaterals
    (∃ (m n : ℝ × ℝ), m ∈ l ∧ n ∈ l ∧
      m.1 = (1 + x) / 2 ∧ m.2 = 1 ∧
      n.1 = (1 + x) / 2 ∧ n.2 = 1)) →
  x = 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_x_coordinate_l651_65118


namespace NUMINAMATH_CALUDE_remaining_money_l651_65113

-- Define the initial amount, amount spent on sweets, and amount given to each friend
def initial_amount : ℚ := 20.10
def sweets_cost : ℚ := 1.05
def friend_gift : ℚ := 1.00
def num_friends : ℕ := 2

-- Define the theorem
theorem remaining_money :
  initial_amount - sweets_cost - (friend_gift * num_friends) = 17.05 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l651_65113


namespace NUMINAMATH_CALUDE_angle_triple_supplement_measure_l651_65140

theorem angle_triple_supplement_measure : 
  ∀ x : ℝ, (x = 3 * (180 - x)) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_measure_l651_65140


namespace NUMINAMATH_CALUDE_no_integer_solutions_l651_65127

theorem no_integer_solutions (m s : ℤ) (h : m * s = 2000^2001) :
  ¬∃ (x y : ℤ), m * x^2 - s * y^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l651_65127


namespace NUMINAMATH_CALUDE_lemonade_proportion_l651_65114

theorem lemonade_proportion (lemons_small : ℕ) (gallons_small : ℕ) (gallons_large : ℕ) :
  lemons_small = 36 →
  gallons_small = 48 →
  gallons_large = 100 →
  (lemons_small * gallons_large) / gallons_small = 75 :=
by
  sorry

end NUMINAMATH_CALUDE_lemonade_proportion_l651_65114


namespace NUMINAMATH_CALUDE_slices_per_pizza_l651_65159

theorem slices_per_pizza (total_pizzas : ℕ) (total_slices : ℕ) 
  (h1 : total_pizzas = 21) 
  (h2 : total_slices = 168) : 
  total_slices / total_pizzas = 8 := by
  sorry

end NUMINAMATH_CALUDE_slices_per_pizza_l651_65159


namespace NUMINAMATH_CALUDE_num_valid_distributions_is_180_l651_65117

/-- Represents a club -/
inductive Club
| ChunhuiLiteratureSociety
| DancersRollerSkatingClub
| BasketballHome
| GoGarden

/-- Represents a student -/
inductive Student
| A
| B
| C
| D
| E

/-- A valid distribution of students to clubs -/
def ValidDistribution := Student → Club

/-- Checks if a distribution is valid according to the problem conditions -/
def isValidDistribution (d : ValidDistribution) : Prop :=
  (∀ c : Club, ∃ s : Student, d s = c) ∧ 
  (d Student.A ≠ Club.GoGarden)

/-- The number of valid distributions -/
def numValidDistributions : ℕ := sorry

/-- The main theorem stating that the number of valid distributions is 180 -/
theorem num_valid_distributions_is_180 : numValidDistributions = 180 := by sorry

end NUMINAMATH_CALUDE_num_valid_distributions_is_180_l651_65117


namespace NUMINAMATH_CALUDE_integral_proof_l651_65184

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (1/16) * log (abs (x - 2)) + (15/16) * log (abs (x + 2)) + (33*x + 34) / (4*(x + 2)^2)

theorem integral_proof (x : ℝ) (hx2 : x ≠ 2) (hx_2 : x ≠ -2) :
  deriv f x = (x^3 - 6*x^2 + 13*x - 6) / ((x - 2)*(x + 2)^3) :=
by sorry

end NUMINAMATH_CALUDE_integral_proof_l651_65184


namespace NUMINAMATH_CALUDE_same_number_of_friends_l651_65194

theorem same_number_of_friends (n : ℕ) (h : n > 0) :
  ∃ (f : Fin n → Fin n),
    ∃ (i j : Fin n), i ≠ j ∧ f i = f j :=
by
  sorry

end NUMINAMATH_CALUDE_same_number_of_friends_l651_65194


namespace NUMINAMATH_CALUDE_jaylens_vegetables_l651_65133

theorem jaylens_vegetables (x y z g : ℚ) : 
  x = 5/3 * y → 
  z = 2 * (1/2 * y) → 
  g = (1/2 * (x/4)) - 3 → 
  20 = x/4 → 
  x + y + z + g = 183 := by
  sorry

end NUMINAMATH_CALUDE_jaylens_vegetables_l651_65133


namespace NUMINAMATH_CALUDE_triangle_perpendicular_theorem_l651_65162

structure Triangle (P Q R : ℝ × ℝ) where
  pq_length : Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 15
  pr_length : Real.sqrt ((P.1 - R.1)^2 + (P.2 - R.2)^2) = 20

def foot_of_perpendicular (P Q R S : ℝ × ℝ) : Prop :=
  (S.1 - Q.1) * (R.1 - Q.1) + (S.2 - Q.2) * (R.2 - Q.2) = 0 ∧
  (P.1 - S.1) * (R.1 - Q.1) + (P.2 - S.2) * (R.2 - Q.2) = 0

def segment_ratio (Q S R : ℝ × ℝ) : Prop :=
  Real.sqrt ((Q.1 - S.1)^2 + (Q.2 - S.2)^2) / Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) = 3 / 7

theorem triangle_perpendicular_theorem (P Q R S : ℝ × ℝ) 
  (tri : Triangle P Q R) (foot : foot_of_perpendicular P Q R S) (ratio : segment_ratio Q S R) :
  Real.sqrt ((P.1 - S.1)^2 + (P.2 - S.2)^2) = 13.625 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perpendicular_theorem_l651_65162


namespace NUMINAMATH_CALUDE_square_points_probability_l651_65125

/-- The number of points around the square -/
def num_points : ℕ := 8

/-- The number of pairs of points that are one unit apart -/
def favorable_pairs : ℕ := 8

/-- The total number of ways to choose two points from the available points -/
def total_pairs : ℕ := num_points.choose 2

/-- The probability of choosing two points that are one unit apart -/
def probability : ℚ := favorable_pairs / total_pairs

theorem square_points_probability : probability = 2/7 := by sorry

end NUMINAMATH_CALUDE_square_points_probability_l651_65125


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l651_65169

-- Problem 1
theorem problem_1 (x : ℝ) : 
  (4 / (x^2 - 1) - 1 = (1 - x) / (x + 1)) ↔ x = 5/2 :=
sorry

-- Problem 2
theorem problem_2 : 
  ¬∃ (x : ℝ), (2 / (x - 3) + 2 = (1 - x) / (3 - x)) :=
sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l651_65169


namespace NUMINAMATH_CALUDE_raisin_problem_l651_65199

theorem raisin_problem (x : ℕ) : 
  (x / 3 : ℚ) + 4 + ((2 * x / 3 - 4) / 2 : ℚ) + 16 = x → x = 54 := by
  sorry

end NUMINAMATH_CALUDE_raisin_problem_l651_65199


namespace NUMINAMATH_CALUDE_cylinder_not_triangular_front_view_l651_65165

/-- A solid geometry object --/
inductive Solid
  | Cylinder
  | Cone
  | Tetrahedron
  | TriangularPrism

/-- The shape of a view (projection) of a solid --/
inductive ViewShape
  | Triangle
  | Rectangle

/-- The front view of a solid --/
def frontView (s : Solid) : ViewShape :=
  match s with
  | Solid.Cylinder => ViewShape.Rectangle
  | _ => ViewShape.Triangle  -- We only care about the cylinder case for this problem

/-- Theorem: A cylinder cannot have a triangular front view --/
theorem cylinder_not_triangular_front_view :
  ∀ s : Solid, s = Solid.Cylinder → frontView s ≠ ViewShape.Triangle :=
by
  sorry

end NUMINAMATH_CALUDE_cylinder_not_triangular_front_view_l651_65165


namespace NUMINAMATH_CALUDE_board_length_proof_l651_65100

/-- Given a board cut into two pieces, where one piece is twice the length of the other
    and the shorter piece is 23 inches long, the total length of the board is 69 inches. -/
theorem board_length_proof (shorter_piece longer_piece total_length : ℕ) :
  shorter_piece = 23 →
  longer_piece = 2 * shorter_piece →
  total_length = shorter_piece + longer_piece →
  total_length = 69 := by
  sorry

#check board_length_proof

end NUMINAMATH_CALUDE_board_length_proof_l651_65100


namespace NUMINAMATH_CALUDE_julia_birth_year_l651_65156

/-- Given that Wayne is 37 years old in 2021, Peter is 3 years older than Wayne,
    and Julia is 2 years older than Peter, prove that Julia was born in 1979. -/
theorem julia_birth_year (wayne_age : ℕ) (peter_age_diff : ℕ) (julia_age_diff : ℕ) :
  wayne_age = 37 →
  peter_age_diff = 3 →
  julia_age_diff = 2 →
  2021 - wayne_age - peter_age_diff - julia_age_diff = 1979 := by
  sorry

end NUMINAMATH_CALUDE_julia_birth_year_l651_65156


namespace NUMINAMATH_CALUDE_N_divisible_by_2027_l651_65168

theorem N_divisible_by_2027 : ∃ k : ℤ, (7 * 9 * 13 + 2020 * 2018 * 2014) = 2027 * k := by
  sorry

end NUMINAMATH_CALUDE_N_divisible_by_2027_l651_65168


namespace NUMINAMATH_CALUDE_inequality_proof_l651_65196

theorem inequality_proof (x y : ℝ) (hx : -1 < x ∧ x < 1) (hy : -1 < y ∧ y < 1) :
  1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x*y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l651_65196


namespace NUMINAMATH_CALUDE_ones_divisibility_l651_65167

theorem ones_divisibility (d : ℕ) (h1 : d > 0) (h2 : ¬ 2 ∣ d) (h3 : ¬ 5 ∣ d) :
  ∃ n : ℕ, d ∣ ((10^n - 1) / 9) :=
sorry

end NUMINAMATH_CALUDE_ones_divisibility_l651_65167


namespace NUMINAMATH_CALUDE_solution_verification_l651_65132

theorem solution_verification (x y : ℝ) : x = 2 ∧ x + y = 3 → y = 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_verification_l651_65132


namespace NUMINAMATH_CALUDE_apartment_cost_difference_l651_65152

def apartment_cost (rent : ℕ) (utilities : ℕ) (daily_miles : ℕ) : ℕ :=
  rent + utilities + (daily_miles * 58 * 20) / 100

theorem apartment_cost_difference : 
  apartment_cost 800 260 31 - apartment_cost 900 200 21 = 76 := by sorry

end NUMINAMATH_CALUDE_apartment_cost_difference_l651_65152


namespace NUMINAMATH_CALUDE_solution_difference_l651_65157

theorem solution_difference (r s : ℝ) : 
  (r - 5) * (r + 5) = 25 * r - 125 →
  (s - 5) * (s + 5) = 25 * s - 125 →
  r ≠ s →
  r > s →
  r - s = 15 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l651_65157


namespace NUMINAMATH_CALUDE_janous_problem_l651_65138

def is_valid_triple (x y z : ℕ+) : Prop :=
  x ∣ (y + 1) ∧ y ∣ (z + 1) ∧ z ∣ (x + 1)

def solution_set : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(1, 1, 1), (1, 2, 1), (1, 1, 2), (1, 3, 2), (2, 1, 1), (2, 1, 3), (3, 1, 2), 
   (2, 2, 2), (2, 2, 3), (2, 3, 2), (3, 2, 2)}

theorem janous_problem :
  ∀ x y z : ℕ+, is_valid_triple x y z ↔ (x, y, z) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_janous_problem_l651_65138


namespace NUMINAMATH_CALUDE_divisibility_problem_solutions_l651_65145

/-- The set of solutions for the divisibility problem -/
def SolutionSet : Set (ℕ × ℕ) := {(1, 1), (1, 5), (5, 1)}

/-- The divisibility condition -/
def DivisibilityCondition (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ (m * n) ∣ ((2^(2^n) + 1) * (2^(2^m) + 1))

/-- Theorem stating that the SolutionSet contains all and only the pairs satisfying the divisibility condition -/
theorem divisibility_problem_solutions :
  ∀ m n : ℕ, DivisibilityCondition m n ↔ (m, n) ∈ SolutionSet := by
  sorry


end NUMINAMATH_CALUDE_divisibility_problem_solutions_l651_65145


namespace NUMINAMATH_CALUDE_coffee_cheesecake_set_price_l651_65153

/-- The final price of a coffee and cheesecake set with discount -/
theorem coffee_cheesecake_set_price (coffee_price : ℝ) (cheesecake_price : ℝ) (discount_rate : ℝ) :
  coffee_price = 6 →
  cheesecake_price = 10 →
  discount_rate = 0.25 →
  coffee_price + cheesecake_price - (coffee_price + cheesecake_price) * discount_rate = 12 := by
  sorry

end NUMINAMATH_CALUDE_coffee_cheesecake_set_price_l651_65153


namespace NUMINAMATH_CALUDE_ben_win_probability_l651_65161

/-- The probability of Ben winning a game, given the probability of losing and that tying is impossible -/
theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5/7) (h2 : lose_prob + win_prob = 1) : win_prob = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_ben_win_probability_l651_65161


namespace NUMINAMATH_CALUDE_inequalities_hold_l651_65174

theorem inequalities_hold (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (b + 1) / (a + 1) > b / a ∧ a + 1 / b > b + 1 / a := by
  sorry

end NUMINAMATH_CALUDE_inequalities_hold_l651_65174


namespace NUMINAMATH_CALUDE_third_term_is_four_l651_65154

/-- A geometric sequence with a₁ = 1 and a₅ = 16 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ a 5 = 16 ∧ ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)

/-- The third term of the geometric sequence is 4 -/
theorem third_term_is_four (a : ℕ → ℝ) (h : geometric_sequence a) : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_four_l651_65154


namespace NUMINAMATH_CALUDE_meeting_point_closer_to_a_l651_65186

/-- The distance between two points A and B -/
def total_distance : ℕ := 85

/-- The constant speed of the person starting from point A -/
def speed_a : ℕ := 5

/-- The initial speed of the person starting from point B -/
def initial_speed_b : ℕ := 4

/-- The hourly increase in speed for the person starting from point B -/
def speed_increase_b : ℕ := 1

/-- The number of hours until the two people meet -/
def meeting_time : ℕ := 6

/-- The distance walked by the person starting from point A -/
def distance_a : ℕ := speed_a * meeting_time

/-- The distance walked by the person starting from point B -/
def distance_b : ℕ := meeting_time * (initial_speed_b + (meeting_time - 1) / 2 * speed_increase_b)

/-- The difference in distances walked by the two people -/
def distance_difference : ℤ := distance_b - distance_a

theorem meeting_point_closer_to_a : distance_difference = 9 := by sorry

end NUMINAMATH_CALUDE_meeting_point_closer_to_a_l651_65186
