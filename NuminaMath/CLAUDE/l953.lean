import Mathlib

namespace NUMINAMATH_CALUDE_prob_both_greater_than_four_l953_95311

/-- The probability of rolling a number greater than 4 on a fair six-sided die -/
def prob_greater_than_four : ℚ := 1 / 3

/-- The number of fair six-sided dice rolled -/
def num_dice : ℕ := 2

/-- Theorem: The probability of both dice showing a number greater than 4 when rolling two fair six-sided dice is 1/9 -/
theorem prob_both_greater_than_four :
  prob_greater_than_four ^ num_dice = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_both_greater_than_four_l953_95311


namespace NUMINAMATH_CALUDE_red_star_selection_probability_l953_95381

/-- The probability of selecting a specific book from a set of books -/
def probability_of_selection (total_books : ℕ) (target_books : ℕ) : ℚ :=
  target_books / total_books

/-- Theorem: The probability of selecting "The Red Star Shines Over China" from 4 books is 1/4 -/
theorem red_star_selection_probability :
  probability_of_selection 4 1 = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_red_star_selection_probability_l953_95381


namespace NUMINAMATH_CALUDE_unique_row_with_53_in_pascal_triangle_l953_95318

theorem unique_row_with_53_in_pascal_triangle :
  ∃! n : ℕ, ∃ k : ℕ, k ≤ n ∧ Nat.choose n k = 53 :=
by sorry

end NUMINAMATH_CALUDE_unique_row_with_53_in_pascal_triangle_l953_95318


namespace NUMINAMATH_CALUDE_geometric_series_common_ratio_l953_95341

/-- The first term of the geometric series -/
def a₁ : ℚ := 7/8

/-- The second term of the geometric series -/
def a₂ : ℚ := -14/27

/-- The third term of the geometric series -/
def a₃ : ℚ := 28/81

/-- The common ratio of the geometric series -/
def r : ℚ := -2/3

/-- Theorem stating that the given series is geometric with common ratio r -/
theorem geometric_series_common_ratio :
  a₂ = a₁ * r ∧ a₃ = a₂ * r := by sorry

end NUMINAMATH_CALUDE_geometric_series_common_ratio_l953_95341


namespace NUMINAMATH_CALUDE_brick_height_calculation_l953_95355

/-- Calculates the height of a brick given the wall dimensions, mortar percentage, brick dimensions, and number of bricks --/
theorem brick_height_calculation (wall_length wall_width wall_height : ℝ)
  (mortar_percentage : ℝ) (brick_length brick_width : ℝ) (num_bricks : ℕ) :
  wall_length = 10 →
  wall_width = 4 →
  wall_height = 5 →
  mortar_percentage = 0.1 →
  brick_length = 0.25 →
  brick_width = 0.15 →
  num_bricks = 6000 →
  ∃ (brick_height : ℝ),
    brick_height = 0.8 ∧
    (1 - mortar_percentage) * wall_length * wall_width * wall_height =
    (brick_length * brick_width * brick_height) * num_bricks :=
by sorry

end NUMINAMATH_CALUDE_brick_height_calculation_l953_95355


namespace NUMINAMATH_CALUDE_parabola_vertex_l953_95398

/-- The equation of a parabola in the form 2y^2 + 8y + 3x + 7 = 0 -/
def parabola_equation (x y : ℝ) : Prop :=
  2 * y^2 + 8 * y + 3 * x + 7 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  eq x y ∧ ∀ x' y', eq x' y' → y ≤ y'

theorem parabola_vertex :
  is_vertex (1/3) (-2) parabola_equation := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l953_95398


namespace NUMINAMATH_CALUDE_kelly_initial_games_l953_95371

/-- The number of games Kelly gave away -/
def games_given_away : ℕ := 91

/-- The number of games Kelly has left -/
def games_left : ℕ := 92

/-- The initial number of games Kelly had -/
def initial_games : ℕ := games_given_away + games_left

theorem kelly_initial_games : initial_games = 183 := by
  sorry

end NUMINAMATH_CALUDE_kelly_initial_games_l953_95371


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l953_95331

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- number of days
  let k : ℕ := 4  -- number of successes (chocolate milk days)
  let p : ℚ := 2/3  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 :=
by sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l953_95331


namespace NUMINAMATH_CALUDE_town_distance_interval_l953_95309

def distance_to_town (d : ℝ) : Prop :=
  (¬ (d ≥ 8)) ∧ (¬ (d ≤ 7)) ∧ (¬ (d ≤ 6)) ∧ (d ≠ 5)

theorem town_distance_interval :
  ∀ d : ℝ, distance_to_town d → (7 < d ∧ d < 8) :=
by sorry

end NUMINAMATH_CALUDE_town_distance_interval_l953_95309


namespace NUMINAMATH_CALUDE_school_averages_l953_95316

theorem school_averages 
  (J L : ℕ) -- Number of boys at Jefferson and Lincoln
  (j l : ℕ) -- Number of girls at Jefferson and Lincoln
  (h1 : (68 * J + 73 * j) / (J + j) = 70) -- Jefferson combined average
  (h2 : (68 * J + 78 * L) / (J + L) = 76) -- Boys combined average
  (h3 : J = (3 * j) / 2) -- Derived from h1
  (h4 : J = L) -- Derived from h2
  (h5 : l = j) -- Assumption of equal girls at both schools
  : ((73 * j + 85 * l) / (j + l) = 79) ∧ 
    ((78 * L + 85 * l) / (L + l) = 808/10) :=
by sorry

end NUMINAMATH_CALUDE_school_averages_l953_95316


namespace NUMINAMATH_CALUDE_units_digit_of_7_to_1000_l953_95307

theorem units_digit_of_7_to_1000 : (7^1000 : ℕ) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_7_to_1000_l953_95307


namespace NUMINAMATH_CALUDE_least_8bit_number_proof_l953_95390

/-- The least positive base-10 number requiring 8 binary digits -/
def least_8bit_number : ℕ := 128

/-- Convert a natural number to its binary representation -/
def to_binary (n : ℕ) : List Bool := sorry

/-- Count the number of digits in a binary representation -/
def binary_digit_count (n : ℕ) : ℕ := (to_binary n).length

theorem least_8bit_number_proof :
  (∀ m : ℕ, m < least_8bit_number → binary_digit_count m < 8) ∧
  binary_digit_count least_8bit_number = 8 := by sorry

end NUMINAMATH_CALUDE_least_8bit_number_proof_l953_95390


namespace NUMINAMATH_CALUDE_find_number_l953_95338

theorem find_number : ∃ n : ℤ, 695 - 329 = n - 254 ∧ n = 620 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l953_95338


namespace NUMINAMATH_CALUDE_quadratic_factorization_l953_95348

theorem quadratic_factorization (x : ℝ) : x^2 - 2*x + 1 = (x - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l953_95348


namespace NUMINAMATH_CALUDE_ab_max_and_4a2_b2_min_l953_95378

theorem ab_max_and_4a2_b2_min (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → a * b ≥ x * y) ∧
  (∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = 1 → 4 * a^2 + b^2 ≤ 4 * x^2 + y^2) ∧
  a * b = 1/8 ∧
  4 * a^2 + b^2 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_ab_max_and_4a2_b2_min_l953_95378


namespace NUMINAMATH_CALUDE_angle_bisector_locus_l953_95317

/-- An angle with vertex A and sides AB and AC -/
structure Angle (A B C : ℝ × ℝ) : Prop where
  nondegenerate : A ≠ B ∧ A ≠ C

/-- A ray from point P through point Q -/
def Ray (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X | ∃ t : ℝ, t ≥ 0 ∧ X = P + t • (Q - P)}

/-- A line segment between two points -/
def Segment (P Q : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {X | ∃ t : ℝ, 0 < t ∧ t < 1 ∧ X = P + t • (Q - P)}

/-- Perpendicular from a point to a line -/
def Perpendicular (P Q R : ℝ × ℝ) (X : ℝ × ℝ) : Prop :=
  (X - P) • (Q - R) = 0 ∧ ∃ t : ℝ, X = P + t • (Q - R)

/-- The locus of points for the angle bisector problem -/
def AngleBisectorLocus (A O B C K L : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {M | (M ∈ Ray A O \ {A, O}) ∨ (M ∈ Segment K L \ {K, L})}

theorem angle_bisector_locus
  (A O B C K L : ℝ × ℝ)
  (h_angle : Angle A B C)
  (h_bisector : O ∈ Ray A O \ {A})
  (h_perp_K : Perpendicular O K B A)
  (h_perp_L : Perpendicular O L C A)
  (M : ℝ × ℝ) :
  M ∈ AngleBisectorLocus A O B C K L ↔
    ((M ∈ Ray A O ∧ M ≠ A ∧ M ≠ O) ∨ (M ∈ Segment K L ∧ M ≠ K ∧ M ≠ L)) :=
by sorry

end NUMINAMATH_CALUDE_angle_bisector_locus_l953_95317


namespace NUMINAMATH_CALUDE_track_meet_adults_l953_95305

theorem track_meet_adults (children : ℕ) (total_seats : ℕ) (empty_seats : ℕ) 
  (h1 : children = 52)
  (h2 : total_seats = 95)
  (h3 : empty_seats = 14) :
  total_seats - empty_seats - children = 29 := by
  sorry

end NUMINAMATH_CALUDE_track_meet_adults_l953_95305


namespace NUMINAMATH_CALUDE_apples_left_over_l953_95360

theorem apples_left_over (greg_sarah_apples susan_apples mark_apples : ℕ) : 
  greg_sarah_apples = 18 →
  susan_apples = 2 * (greg_sarah_apples / 2) →
  mark_apples = susan_apples - 5 →
  (greg_sarah_apples + susan_apples + mark_apples) - 40 = 9 :=
by sorry

end NUMINAMATH_CALUDE_apples_left_over_l953_95360


namespace NUMINAMATH_CALUDE_solve_equation_l953_95346

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 4 / 3 → x = -27 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l953_95346


namespace NUMINAMATH_CALUDE_remainder_proof_l953_95389

theorem remainder_proof : ∃ (q : ℕ), 4351 = 101 * q + 8 :=
by
  -- We define the greatest common divisor G as 101
  let G : ℕ := 101

  -- We define the condition that G divides 5161 with remainder 10
  have h1 : ∃ (q : ℕ), 5161 = G * q + 10 := by sorry

  -- We prove that 4351 divided by G has remainder 8
  sorry

end NUMINAMATH_CALUDE_remainder_proof_l953_95389


namespace NUMINAMATH_CALUDE_library_visitors_average_l953_95344

theorem library_visitors_average (monday_visitors : ℕ) (total_visitors : ℕ) :
  monday_visitors = 50 →
  total_visitors = 250 →
  (total_visitors - (monday_visitors + 2 * monday_visitors)) / 5 = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_library_visitors_average_l953_95344


namespace NUMINAMATH_CALUDE_weaving_problem_l953_95322

/-- Sum of an arithmetic sequence -/
def arithmeticSum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * a₁ + (n * (n - 1) / 2) * d

/-- The weaving problem -/
theorem weaving_problem : arithmeticSum 5 (16/29) 30 = 390 := by
  sorry

end NUMINAMATH_CALUDE_weaving_problem_l953_95322


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l953_95323

/-- The number of yellow marbles Mary has -/
def mary_marbles : ℕ := 9

/-- The number of yellow marbles Joan has -/
def joan_marbles : ℕ := 3

/-- The total number of yellow marbles Mary and Joan have together -/
def total_marbles : ℕ := mary_marbles + joan_marbles

theorem yellow_marbles_count : total_marbles = 12 := by sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l953_95323


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l953_95359

/-- Volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume
  (a b : ℝ) (h_positive : 0 < a ∧ 0 < b) (h_height_constraint : a * Real.sqrt 2 / 2 ≤ b ∧ b < a * Real.sqrt 3 / 2) :
  ∃ V : ℝ, V = (a^3 * b) / (12 * Real.sqrt (3 * a^2 - 4 * b^2)) ∧ V > 0 := by
  sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l953_95359


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_A_intersect_B_B_subset_A_iff_m_range_l953_95337

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

-- Part 1
theorem union_of_A_and_B : A ∪ B 4 = {x | -2 ≤ x ∧ x < 5} := by sorry

theorem complement_A_intersect_B : (Set.univ \ A) ∩ B 4 = {x | 4 < x ∧ x < 5} := by sorry

-- Part 2
theorem B_subset_A_iff_m_range : 
  ∀ m : ℝ, B m ⊆ A ↔ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_A_intersect_B_B_subset_A_iff_m_range_l953_95337


namespace NUMINAMATH_CALUDE_negation_of_all_cuboids_are_prisms_l953_95368

-- Define the universe of shapes
variable {Shape : Type}

-- Define properties
variable (isCuboid : Shape → Prop)
variable (isPrism : Shape → Prop)
variable (hasLateralFaces : Shape → ℕ → Prop)

-- The theorem
theorem negation_of_all_cuboids_are_prisms :
  (¬ ∀ x : Shape, isCuboid x → (isPrism x ∧ hasLateralFaces x 4)) ↔ 
  (∃ x : Shape, isCuboid x ∧ ¬(isPrism x ∧ hasLateralFaces x 4)) := by
sorry

end NUMINAMATH_CALUDE_negation_of_all_cuboids_are_prisms_l953_95368


namespace NUMINAMATH_CALUDE_albert_needs_more_money_l953_95386

-- Define the costs of items and Albert's current money
def paintbrush_cost : ℚ := 1.50
def paints_cost : ℚ := 4.35
def easel_cost : ℚ := 12.65
def canvas_cost : ℚ := 7.95
def palette_cost : ℚ := 3.75
def albert_current_money : ℚ := 10.60

-- Define the total cost of items
def total_cost : ℚ := paintbrush_cost + paints_cost + easel_cost + canvas_cost + palette_cost

-- Theorem: Albert needs $19.60 more
theorem albert_needs_more_money : total_cost - albert_current_money = 19.60 := by
  sorry

end NUMINAMATH_CALUDE_albert_needs_more_money_l953_95386


namespace NUMINAMATH_CALUDE_parallel_lines_bisect_circle_perimeter_l953_95301

-- Define the lines and circle
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x - 2 * y + 2 = 0
def line_m (a : ℝ) (x y : ℝ) : Prop := x + (a - 3) * y + 1 = 0
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, line_l a x y ↔ line_m a x y) ↔ a = 1 :=
sorry

-- Theorem for bisecting circle's perimeter
theorem bisect_circle_perimeter (a : ℝ) :
  (∃ x y : ℝ, line_l a x y ∧ x = 1 ∧ y = 0) ↔ a = -2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_bisect_circle_perimeter_l953_95301


namespace NUMINAMATH_CALUDE_parallelogram_area_is_36_l953_95349

-- Define the vectors v and w
def v : ℝ × ℝ := (4, -6)
def w : ℝ × ℝ := (8, -3)

-- Define the area of the parallelogram
def parallelogramArea (a b : ℝ × ℝ) : ℝ :=
  |a.1 * b.2 - a.2 * b.1|

-- Theorem statement
theorem parallelogram_area_is_36 :
  parallelogramArea v w = 36 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_is_36_l953_95349


namespace NUMINAMATH_CALUDE_lines_parallel_or_skew_l953_95313

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for planes
variable (parallelPlanes : Plane → Plane → Prop)

-- Define the subset relation for lines and planes
variable (subsetLinePlane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallelLines : Line → Line → Prop)

-- Define the skew relation for lines
variable (skewLines : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_or_skew
  (a b : Line) (α β : Plane)
  (h_diff_lines : a ≠ b)
  (h_diff_planes : α ≠ β)
  (h_parallel_planes : parallelPlanes α β)
  (h_a_in_α : subsetLinePlane a α)
  (h_b_in_β : subsetLinePlane b β) :
  parallelLines a b ∨ skewLines a b :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_or_skew_l953_95313


namespace NUMINAMATH_CALUDE_age_difference_l953_95354

theorem age_difference (alvin_age simon_age : ℕ) (h1 : alvin_age = 30) (h2 : simon_age = 10) :
  alvin_age / 2 - simon_age = 5 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l953_95354


namespace NUMINAMATH_CALUDE_third_roll_five_prob_l953_95343

/-- Represents a six-sided die --/
inductive Die
| Fair
| Biased

/-- Probability of rolling a five for a given die --/
def prob_five (d : Die) : ℚ :=
  match d with
  | Die.Fair => 1/6
  | Die.Biased => 3/4

/-- Probability of rolling a non-five for a given die --/
def prob_not_five (d : Die) : ℚ :=
  match d with
  | Die.Fair => 5/6
  | Die.Biased => 1/4

/-- Probability of choosing each die initially --/
def initial_prob : ℚ := 1/2

/-- Theorem stating the probability of rolling a five on the third roll --/
theorem third_roll_five_prob :
  let p_fair := initial_prob * (prob_five Die.Fair)^2
  let p_biased := initial_prob * (prob_five Die.Biased)^2
  let p_fair_given_two_fives := p_fair / (p_fair + p_biased)
  let p_biased_given_two_fives := p_biased / (p_fair + p_biased)
  p_fair_given_two_fives * (prob_five Die.Fair) + 
  p_biased_given_two_fives * (prob_five Die.Biased) = 223/74 := by
  sorry

end NUMINAMATH_CALUDE_third_roll_five_prob_l953_95343


namespace NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l953_95320

theorem triangle_with_angle_ratio_1_2_3_is_right_triangle (A B C : ℝ) :
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = 180 →
  B = 2 * A →
  C = 3 * A →
  C = 90 :=
sorry

end NUMINAMATH_CALUDE_triangle_with_angle_ratio_1_2_3_is_right_triangle_l953_95320


namespace NUMINAMATH_CALUDE_jacket_cost_calculation_l953_95387

/-- The amount spent on clothes in cents -/
def total_spent : ℕ := 1428

/-- The amount spent on shorts in cents -/
def shorts_cost : ℕ := 954

/-- The amount spent on the jacket in cents -/
def jacket_cost : ℕ := total_spent - shorts_cost

theorem jacket_cost_calculation : jacket_cost = 474 := by
  sorry

end NUMINAMATH_CALUDE_jacket_cost_calculation_l953_95387


namespace NUMINAMATH_CALUDE_somu_father_age_ratio_l953_95352

/-- Proves that the ratio of Somu's age to his father's age is 1:3 given the conditions -/
theorem somu_father_age_ratio :
  ∀ (somu_age father_age : ℕ),
  somu_age = 18 →
  somu_age - 9 = (father_age - 9) / 5 →
  ∃ (k : ℕ), k > 0 ∧ somu_age * 3 = father_age * k ∧ k = 1 :=
by sorry

end NUMINAMATH_CALUDE_somu_father_age_ratio_l953_95352


namespace NUMINAMATH_CALUDE_alice_gadget_sales_l953_95315

/-- The worth of gadgets Alice sold -/
def gadget_worth : ℝ := 2500

/-- Alice's monthly basic salary -/
def basic_salary : ℝ := 240

/-- Alice's commission rate -/
def commission_rate : ℝ := 0.02

/-- Amount Alice saves -/
def savings : ℝ := 29

/-- Percentage of total earnings Alice saves -/
def savings_rate : ℝ := 0.10

/-- Alice's total earnings -/
def total_earnings : ℝ := basic_salary + commission_rate * gadget_worth

theorem alice_gadget_sales :
  gadget_worth = 2500 ∧
  basic_salary = 240 ∧
  commission_rate = 0.02 ∧
  savings = 29 ∧
  savings_rate = 0.10 ∧
  savings = savings_rate * total_earnings :=
by sorry

end NUMINAMATH_CALUDE_alice_gadget_sales_l953_95315


namespace NUMINAMATH_CALUDE_car_overtake_distance_l953_95336

/-- Proves that the initial distance between two cars is 10 miles given their speeds and overtaking time -/
theorem car_overtake_distance (speed_a speed_b time_to_overtake : ℝ) 
  (h1 : speed_a = 58)
  (h2 : speed_b = 50)
  (h3 : time_to_overtake = 2.25)
  (h4 : (speed_a - speed_b) * time_to_overtake = initial_distance + 8) :
  initial_distance = 10 := by sorry


end NUMINAMATH_CALUDE_car_overtake_distance_l953_95336


namespace NUMINAMATH_CALUDE_even_and_mono_decreasing_implies_ordering_l953_95314

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define a monotonically decreasing function on an interval
def MonoDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x > f y

-- Main theorem
theorem even_and_mono_decreasing_implies_ordering (f : ℝ → ℝ)
  (h1 : EvenFunction f)
  (h2 : MonoDecreasing (fun x => f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 := by
  sorry

end NUMINAMATH_CALUDE_even_and_mono_decreasing_implies_ordering_l953_95314


namespace NUMINAMATH_CALUDE_solution_difference_l953_95353

theorem solution_difference (r s : ℝ) : 
  ((6 * r - 18) / (r^2 - 4*r - 21) = r - 3) →
  ((6 * s - 18) / (s^2 - 4*s - 21) = s - 3) →
  r ≠ s →
  r > s →
  r - s = 4 := by
sorry

end NUMINAMATH_CALUDE_solution_difference_l953_95353


namespace NUMINAMATH_CALUDE_product_inequality_l953_95302

theorem product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 + 1/x) * (1 + 1/y) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l953_95302


namespace NUMINAMATH_CALUDE_prob_five_shots_expected_shots_l953_95362

-- Define the probability of hitting a target
variable (p : ℝ) (hp : 0 < p) (hp1 : p < 1)

-- Define the number of targets
def num_targets : ℕ := 3

-- Theorem for part (a)
theorem prob_five_shots : 
  (6 : ℝ) * p^3 * (1 - p)^2 = 
  (num_targets.choose 2) * p^3 * (1 - p)^2 := by sorry

-- Theorem for part (b)
theorem expected_shots : 
  (3 : ℝ) / p = num_targets / p := by sorry

end NUMINAMATH_CALUDE_prob_five_shots_expected_shots_l953_95362


namespace NUMINAMATH_CALUDE_least_bench_sections_thirteen_is_least_l953_95380

theorem least_bench_sections (M : ℕ) : M > 0 ∧ 5 * M = 13 * M → M ≥ 13 := by
  sorry

theorem thirteen_is_least : ∃ M : ℕ, M > 0 ∧ 5 * M = 13 * M ∧ M = 13 := by
  sorry

end NUMINAMATH_CALUDE_least_bench_sections_thirteen_is_least_l953_95380


namespace NUMINAMATH_CALUDE_no_valid_rope_net_with_2001_knots_l953_95350

/-- A rope net is a structure where knots are connected by ropes. -/
structure RopeNet where
  knots : ℕ
  ropes_per_knot : ℕ

/-- A valid rope net has a positive number of knots and exactly 3 ropes per knot. -/
def is_valid_rope_net (net : RopeNet) : Prop :=
  net.knots > 0 ∧ net.ropes_per_knot = 3

/-- The total number of rope ends in a rope net. -/
def total_rope_ends (net : RopeNet) : ℕ :=
  net.knots * net.ropes_per_knot

/-- The number of distinct ropes in a rope net. -/
def distinct_ropes (net : RopeNet) : ℚ :=
  (total_rope_ends net : ℚ) / 2

/-- Theorem: It is impossible for a valid rope net to have exactly 2001 knots. -/
theorem no_valid_rope_net_with_2001_knots :
  ¬ ∃ (net : RopeNet), is_valid_rope_net net ∧ net.knots = 2001 :=
sorry

end NUMINAMATH_CALUDE_no_valid_rope_net_with_2001_knots_l953_95350


namespace NUMINAMATH_CALUDE_badminton_purchase_costs_l953_95312

/-- Represents the cost calculation for badminton equipment purchases --/
def BadmintonPurchase (x : ℕ) : Prop :=
  x > 16 →
  let racket_price : ℕ := 150
  let shuttlecock_price : ℕ := 40
  let num_rackets : ℕ := 16
  let store_a_cost : ℕ := num_rackets * racket_price + (x - num_rackets) * shuttlecock_price
  let store_b_cost : ℕ := ((num_rackets * racket_price + x * shuttlecock_price) * 80) / 100
  (store_a_cost = 1760 + 40 * x) ∧ (store_b_cost = 1920 + 32 * x)

theorem badminton_purchase_costs (x : ℕ) : BadmintonPurchase x := by
  sorry

end NUMINAMATH_CALUDE_badminton_purchase_costs_l953_95312


namespace NUMINAMATH_CALUDE_smallest_k_for_inequality_l953_95327

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 4 ∧ 
  (∀ n : ℕ, n > 0 → ∀ a : ℝ, 0 ≤ a ∧ a ≤ 1 → a^k * (1-a)^n < 1 / (n+1)^3) ∧
  (∀ k' : ℕ, k' < k → ∃ n : ℕ, n > 0 ∧ ∃ a : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ a^k' * (1-a)^n ≥ 1 / (n+1)^3) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_inequality_l953_95327


namespace NUMINAMATH_CALUDE_max_prime_angle_in_isosceles_triangle_l953_95394

def IsIsosceles (a b c : ℕ) : Prop := a + b + c = 180 ∧ a = b

def IsPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem max_prime_angle_in_isosceles_triangle :
  ∀ x : ℕ,
    IsIsosceles x x (180 - 2*x) →
    IsPrime x →
    x ≤ 7 :=
sorry

end NUMINAMATH_CALUDE_max_prime_angle_in_isosceles_triangle_l953_95394


namespace NUMINAMATH_CALUDE_baker_problem_l953_95365

/-- The number of cakes that can be made given the available ingredients and recipe requirements. -/
def num_cakes : ℕ := 49

/-- The number of loaves of bread that can be made given the available ingredients and recipe requirements. -/
def num_bread : ℕ := 30

/-- The amount of flour available (in cups). -/
def flour_available : ℕ := 188

/-- The amount of sugar available (in cups). -/
def sugar_available : ℕ := 113

/-- The amount of flour required for one loaf of bread (in cups). -/
def flour_per_bread : ℕ := 3

/-- The amount of sugar required for one loaf of bread (in cups). -/
def sugar_per_bread : ℚ := 1/2

/-- The amount of flour required for one cake (in cups). -/
def flour_per_cake : ℕ := 2

/-- The amount of sugar required for one cake (in cups). -/
def sugar_per_cake : ℕ := 2

theorem baker_problem :
  (num_bread * flour_per_bread + num_cakes * flour_per_cake = flour_available) ∧
  (num_bread * sugar_per_bread + num_cakes * sugar_per_cake = sugar_available) :=
by sorry

end NUMINAMATH_CALUDE_baker_problem_l953_95365


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l953_95376

theorem arithmetic_calculation : 2535 + 240 / 30 - 435 = 2108 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l953_95376


namespace NUMINAMATH_CALUDE_right_triangle_set_l953_95340

theorem right_triangle_set : ∀ (a b c : ℝ),
  ((a = 1 ∧ b = Real.sqrt 2 ∧ c = 3) ∨
   (a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 6 ∧ b = 8 ∧ c = 12) ∨
   (a = 5 ∧ b = 11 ∧ c = 13)) →
  (a^2 + b^2 = c^2 ↔ (a = 3 ∧ b = 4 ∧ c = 5)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_set_l953_95340


namespace NUMINAMATH_CALUDE_apples_picked_theorem_l953_95358

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- The total number of apples picked -/
def total_apples : ℕ := mike_apples + nancy_apples + keith_apples

theorem apples_picked_theorem : total_apples = 16 := by
  sorry

end NUMINAMATH_CALUDE_apples_picked_theorem_l953_95358


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l953_95379

/-- Proves that a person's swimming speed in still water is 4 km/h given the conditions -/
theorem swimming_speed_in_still_water 
  (water_speed : ℝ) 
  (swimming_time : ℝ) 
  (swimming_distance : ℝ) 
  (h1 : water_speed = 2)
  (h2 : swimming_time = 5)
  (h3 : swimming_distance = 10)
  : ∃ (still_water_speed : ℝ), 
    swimming_distance = (still_water_speed - water_speed) * swimming_time ∧ 
    still_water_speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l953_95379


namespace NUMINAMATH_CALUDE_triangle_problem_l953_95395

/-- Given a triangle ABC with sides a, b, c and corresponding angles A, B, C. -/
theorem triangle_problem (a b c A B C : Real) :
  -- Conditions
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (a * Real.cos B + Real.sqrt 3 * b * Real.sin A = c) →
  (a = 1) →
  (b * c * Real.cos A = 3) →
  -- Conclusions
  (A = π / 6) ∧ (b + c = Real.sqrt 3 + 2) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l953_95395


namespace NUMINAMATH_CALUDE_quadratic_rational_solution_l953_95370

/-- The quadratic equation kx^2 + 16x + k = 0 has rational solutions if and only if k = 8, where k is a positive integer. -/
theorem quadratic_rational_solution (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rational_solution_l953_95370


namespace NUMINAMATH_CALUDE_park_trees_l953_95369

theorem park_trees (pine_percentage : ℝ) (non_pine_count : ℕ) 
  (h1 : pine_percentage = 0.7)
  (h2 : non_pine_count = 105) : 
  ∃ (total_trees : ℕ), 
    (↑non_pine_count : ℝ) = (1 - pine_percentage) * (total_trees : ℝ) ∧ 
    total_trees = 350 :=
by sorry

end NUMINAMATH_CALUDE_park_trees_l953_95369


namespace NUMINAMATH_CALUDE_equivalent_discount_l953_95388

theorem equivalent_discount (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) 
  (h1 : original_price = 50)
  (h2 : discount1 = 0.3)
  (h3 : discount2 = 0.2) : 
  original_price * (1 - discount1) * (1 - discount2) = original_price * (1 - 0.44) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_discount_l953_95388


namespace NUMINAMATH_CALUDE_P_intersect_Q_equals_closed_interval_l953_95385

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 - 2*x ≤ 0}
def Q : Set ℝ := {y | ∃ x, y = x^2 - 2*x}

-- State the theorem
theorem P_intersect_Q_equals_closed_interval :
  P ∩ Q = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_P_intersect_Q_equals_closed_interval_l953_95385


namespace NUMINAMATH_CALUDE_last_three_average_l953_95304

theorem last_three_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 60 →
  (list.take 4).sum / 4 = 55 →
  (list.drop 4).sum / 3 = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_last_three_average_l953_95304


namespace NUMINAMATH_CALUDE_circle_center_height_l953_95374

/-- Represents a circle inside a parabola y = 2x^2, tangent at two points -/
structure CircleInParabola where
  /-- x-coordinate of one tangency point -/
  a : ℝ
  /-- y-coordinate of the circle's center -/
  b : ℝ
  /-- Radius of the circle -/
  r : ℝ
  /-- Condition: The circle is tangent to the parabola -/
  tangent : (a^2 + (2*a^2 - b)^2 = r^2) ∧ ((-a)^2 + (2*(-a)^2 - b)^2 = r^2)
  /-- Condition: The circle's center is on the y-axis -/
  center_on_y_axis : True

/-- Theorem: The y-coordinate of the circle's center equals the y-coordinate of the tangency points -/
theorem circle_center_height (c : CircleInParabola) : c.b = 2 * c.a^2 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_height_l953_95374


namespace NUMINAMATH_CALUDE_solve_for_C_l953_95397

theorem solve_for_C : ∃ C : ℝ, (2 * C - 3 = 11) ∧ (C = 7) := by sorry

end NUMINAMATH_CALUDE_solve_for_C_l953_95397


namespace NUMINAMATH_CALUDE_borrowed_amount_l953_95334

/-- Proves that the amount borrowed is 5000 given the specified conditions --/
theorem borrowed_amount (loan_duration : ℕ) (borrow_rate lend_rate : ℚ) (gain_per_year : ℕ) : 
  loan_duration = 2 →
  borrow_rate = 4 / 100 →
  lend_rate = 8 / 100 →
  gain_per_year = 200 →
  ∃ (amount : ℕ), amount = 5000 ∧ 
    (amount * lend_rate * loan_duration) - (amount * borrow_rate * loan_duration) = gain_per_year * loan_duration :=
by sorry

end NUMINAMATH_CALUDE_borrowed_amount_l953_95334


namespace NUMINAMATH_CALUDE_absolute_value_trigonometry_and_reciprocal_quadratic_equation_solution_l953_95351

-- Problem 1
theorem absolute_value_trigonometry_and_reciprocal :
  |(-3)| - 4 * Real.sin (π / 6) + (1 / 3)⁻¹ = 4 := by sorry

-- Problem 2
theorem quadratic_equation_solution :
  ∀ x : ℝ, 2 * x - 6 = x^2 - 9 ↔ x = -1 ∨ x = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_trigonometry_and_reciprocal_quadratic_equation_solution_l953_95351


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l953_95357

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_first : a 1 = 2)
  (h_sum : a 1 + a 2 + a 3 = 12) :
  a 6 = 12 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l953_95357


namespace NUMINAMATH_CALUDE_total_weight_calculation_l953_95335

/-- Given the weight of apples and the ratio of pears to apples, 
    calculate the total weight of apples and pears. -/
def total_weight (apple_weight : ℝ) (pear_to_apple_ratio : ℝ) : ℝ :=
  apple_weight + pear_to_apple_ratio * apple_weight

/-- Theorem stating that the total weight of apples and pears is equal to
    the weight of apples plus three times the weight of apples, 
    given that there are three times as many pears as apples. -/
theorem total_weight_calculation (apple_weight : ℝ) :
  total_weight apple_weight 3 = apple_weight + 3 * apple_weight :=
by
  sorry

#eval total_weight 240 3  -- Should output 960

end NUMINAMATH_CALUDE_total_weight_calculation_l953_95335


namespace NUMINAMATH_CALUDE_even_function_inequality_l953_95393

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

theorem even_function_inequality (f : ℝ → ℝ) (m : ℝ) :
  is_even_function f →
  (∀ x, -2 ≤ x → x ≤ 2 → f x ∈ Set.range f) →
  monotone_decreasing_on f 0 2 →
  f (1 - m) < f m →
  -1 ≤ m ∧ m < 1/2 := by sorry

end NUMINAMATH_CALUDE_even_function_inequality_l953_95393


namespace NUMINAMATH_CALUDE_sophie_dryer_sheet_savings_l953_95364

/-- Calculates the annual cost savings from not buying dryer sheets -/
def annual_dryer_sheet_savings (loads_per_week : ℕ) (sheets_per_load : ℕ) 
  (sheets_per_box : ℕ) (cost_per_box : ℚ) (weeks_per_year : ℕ) : ℚ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := (sheets_per_year + sheets_per_box - 1) / sheets_per_box
  boxes_per_year * cost_per_box

theorem sophie_dryer_sheet_savings :
  annual_dryer_sheet_savings 4 1 104 (11/2) 52 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sophie_dryer_sheet_savings_l953_95364


namespace NUMINAMATH_CALUDE_rectangular_box_surface_area_l953_95392

theorem rectangular_box_surface_area
  (a b c : ℝ)
  (edge_sum : a + b + c = 39)
  (diagonal : a^2 + b^2 + c^2 = 625) :
  2 * (a * b + b * c + c * a) = 896 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_box_surface_area_l953_95392


namespace NUMINAMATH_CALUDE_inequality_solution_set_l953_95303

theorem inequality_solution_set (x : ℝ) : 
  (Set.Iio (-1) ∪ Set.Ioi 3) = {x | (3 - x) / (x + 1) < 0} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l953_95303


namespace NUMINAMATH_CALUDE_earth_sun_max_distance_l953_95333

/-- The semi-major axis of Earth's orbit in kilometers -/
def semi_major_axis : ℝ := 1.5e8

/-- The semi-minor axis of Earth's orbit in kilometers -/
def semi_minor_axis : ℝ := 3e6

/-- The maximum distance from Earth to Sun in kilometers -/
def max_distance : ℝ := semi_major_axis + semi_minor_axis

theorem earth_sun_max_distance :
  max_distance = 1.53e8 := by sorry

end NUMINAMATH_CALUDE_earth_sun_max_distance_l953_95333


namespace NUMINAMATH_CALUDE_supremum_of_expression_is_zero_l953_95321

open Real

theorem supremum_of_expression_is_zero :
  ∀ ε > 0, ∃ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧
  x * y * z * (x + y + z) / (x + y + z)^3 < ε :=
by sorry

end NUMINAMATH_CALUDE_supremum_of_expression_is_zero_l953_95321


namespace NUMINAMATH_CALUDE_max_value_sum_of_roots_l953_95399

theorem max_value_sum_of_roots (a b c : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → a + b + c = 7 → 
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ 3 * Real.sqrt 23 ∧
  ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 7 ∧
    Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) = 3 * Real.sqrt 23 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sum_of_roots_l953_95399


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l953_95361

theorem count_integers_satisfying_inequality :
  (Finset.filter (fun n : ℤ => (n - 1) * (n + 3) * (n + 7) < 0)
    (Finset.Icc (-10 : ℤ) 12)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l953_95361


namespace NUMINAMATH_CALUDE_same_club_probability_l953_95384

theorem same_club_probability :
  let num_students : ℕ := 2
  let num_clubs : ℕ := 3
  let total_outcomes : ℕ := num_clubs ^ num_students
  let favorable_outcomes : ℕ := num_clubs
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_same_club_probability_l953_95384


namespace NUMINAMATH_CALUDE_gas_cost_per_gallon_l953_95367

/-- Proves that the cost of gas per gallon is $4, given the specified conditions. -/
theorem gas_cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) :
  miles_per_gallon = 32 →
  total_miles = 304 →
  total_cost = 38 →
  total_cost / (total_miles / miles_per_gallon) = 4 := by
  sorry

#check gas_cost_per_gallon

end NUMINAMATH_CALUDE_gas_cost_per_gallon_l953_95367


namespace NUMINAMATH_CALUDE_smallest_b_value_l953_95363

/-- The second smallest positive integer with exactly 3 factors -/
def a : ℕ := 9

/-- A function that returns the number of factors of a positive integer -/
def num_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem smallest_b_value :
  ∃ b : ℕ,
    b > 0 ∧
    num_factors b = a ∧
    a ∣ b ∧
    ∀ c : ℕ, c > 0 → num_factors c = a → a ∣ c → b ≤ c ∧
    b = 30 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l953_95363


namespace NUMINAMATH_CALUDE_remainder_preserved_l953_95382

theorem remainder_preserved (n : ℤ) (h : n % 8 = 3) : (n + 5040) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_preserved_l953_95382


namespace NUMINAMATH_CALUDE_first_valid_year_is_2049_l953_95332

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ year < 3000 ∧ sum_of_digits year = 15

theorem first_valid_year_is_2049 :
  (∀ year : ℕ, year < 2049 → ¬(is_valid_year year)) ∧ 
  is_valid_year 2049 :=
sorry

end NUMINAMATH_CALUDE_first_valid_year_is_2049_l953_95332


namespace NUMINAMATH_CALUDE_div_exp_eq_pow_reciprocal_l953_95325

/-- Division exponentiation for rational numbers -/
def div_exp (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1
  else if n = 1 then a
  else a / (div_exp a (n - 1))

/-- Theorem: Division exponentiation equals power of reciprocal -/
theorem div_exp_eq_pow_reciprocal (a : ℚ) (n : ℕ) (h1 : a ≠ 0) (h2 : n ≥ 3) :
  div_exp a n = (1 / a) ^ (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_div_exp_eq_pow_reciprocal_l953_95325


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l953_95342

theorem polynomial_remainder_theorem (p : ℝ → ℝ) (hp1 : p 1 = 5) (hp3 : p 3 = 8) :
  ∃ (t : ℝ), ∃ (q : ℝ → ℝ), 
    ∀ x, p x = q x * ((x - 1) * (x - 3) * (x - 5)) + 
              (t * x^2 + (3 - 8*t)/2 * x + (7 + 6*t)/2) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l953_95342


namespace NUMINAMATH_CALUDE_closest_to_zero_l953_95300

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem closest_to_zero (a₁ d : ℤ) (h₁ : a₁ = 81) (h₂ : d = -7) :
  ∀ n : ℕ, n ≠ 13 → |arithmetic_sequence a₁ d 13| ≤ |arithmetic_sequence a₁ d n| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_zero_l953_95300


namespace NUMINAMATH_CALUDE_sequence_arrangement_count_l953_95345

theorem sequence_arrangement_count : ℕ :=
  let n : ℕ := 40
  let k : ℕ := 31
  let m : ℕ := 20
  Nat.choose n (n - k) * Nat.factorial (k - 2) * Nat.factorial (n - k)

#check sequence_arrangement_count

end NUMINAMATH_CALUDE_sequence_arrangement_count_l953_95345


namespace NUMINAMATH_CALUDE_f_strictly_increasing_l953_95377

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + 1)

theorem f_strictly_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂ := by sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_l953_95377


namespace NUMINAMATH_CALUDE_briannes_yard_length_l953_95328

theorem briannes_yard_length (derricks_length : ℝ) (alexs_length : ℝ) (briannes_length : ℝ) : 
  derricks_length = 10 →
  alexs_length = derricks_length / 2 →
  briannes_length = 6 * alexs_length →
  briannes_length = 30 := by sorry

end NUMINAMATH_CALUDE_briannes_yard_length_l953_95328


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l953_95329

/-- Given a right triangle with medians from acute angles 6 and √30, prove the hypotenuse is 2√52.8 -/
theorem right_triangle_hypotenuse (a b : ℝ) (h_right : a^2 + b^2 = (a + b)^2 / 4)
  (h_median1 : b^2 + (a/2)^2 = 30) (h_median2 : a^2 + (b/2)^2 = 36) :
  (2*a)^2 + (2*b)^2 = 4 * 52.8 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l953_95329


namespace NUMINAMATH_CALUDE_quadratic_two_roots_l953_95373

/-- The quadratic function f(x) = x^2 - 2x - 3 has exactly two real roots -/
theorem quadratic_two_roots : ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
  (∀ x, x^2 - 2*x - 3 = 0 ↔ x = r₁ ∨ x = r₂) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_two_roots_l953_95373


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l953_95319

theorem partial_fraction_decomposition (M₁ M₂ : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 2 → (45*x - 31) / (x^2 - 3*x + 2) = M₁ / (x - 1) + M₂ / (x - 2)) →
  M₁ * M₂ = -826 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l953_95319


namespace NUMINAMATH_CALUDE_movie_length_after_cut_l953_95396

/-- The final length of a movie after cutting a scene -/
def final_movie_length (original_length scene_cut : ℕ) : ℕ :=
  original_length - scene_cut

/-- Theorem: The final length of the movie is 52 minutes -/
theorem movie_length_after_cut :
  final_movie_length 60 8 = 52 := by
  sorry

end NUMINAMATH_CALUDE_movie_length_after_cut_l953_95396


namespace NUMINAMATH_CALUDE_xiao_ming_brother_age_l953_95391

/-- Check if a year has unique digits -/
def has_unique_digits (year : Nat) : Bool := sorry

/-- Find the latest year before 2013 that is a multiple of 19 and has unique digits -/
def find_birth_year : Nat := sorry

/-- Calculate age in 2013 given a birth year -/
def calculate_age (birth_year : Nat) : Nat := 2013 - birth_year

theorem xiao_ming_brother_age :
  (∀ y : Nat, y < 2013 → ¬(has_unique_digits y)) →
  has_unique_digits 2013 →
  find_birth_year % 19 = 0 →
  has_unique_digits find_birth_year →
  calculate_age find_birth_year = 18 := by sorry

end NUMINAMATH_CALUDE_xiao_ming_brother_age_l953_95391


namespace NUMINAMATH_CALUDE_larger_number_problem_l953_95383

theorem larger_number_problem (x y : ℝ) : 
  x - y = 5 → x + y = 27 → max x y = 16 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l953_95383


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l953_95366

/-- Given the initial conditions of pie crust baking and a new number of crusts,
    calculate the amount of flour required for each new crust. -/
theorem pie_crust_flour_calculation (initial_crusts : ℕ) (initial_flour : ℚ) (new_crusts : ℕ) :
  initial_crusts > 0 →
  initial_flour > 0 →
  new_crusts > 0 →
  (initial_flour / initial_crusts) * new_crusts = initial_flour →
  initial_flour / new_crusts = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l953_95366


namespace NUMINAMATH_CALUDE_quadratic_is_perfect_square_l953_95310

theorem quadratic_is_perfect_square (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 26 * x + 9 = (r * x + s)^2) → a = 169 / 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_is_perfect_square_l953_95310


namespace NUMINAMATH_CALUDE_max_log_sum_min_reciprocal_sum_l953_95326

-- Define the conditions
variable (x y : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (h_eq : 2 * x + 5 * y = 20)

-- Theorem for the maximum value of log x + log y
theorem max_log_sum :
  ∃ (max : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 → 2 * a + 5 * b = 20 → 
    Real.log a + Real.log b ≤ max ∧ 
    (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ 2 * c + 5 * d = 20 ∧ Real.log c + Real.log d = max) ∧
    max = 1 :=
sorry

-- Theorem for the minimum value of 1/x + 1/y
theorem min_reciprocal_sum :
  ∃ (min : ℝ), ∀ (a b : ℝ), a > 0 → b > 0 → 2 * a + 5 * b = 20 → 
    1 / a + 1 / b ≥ min ∧ 
    (∃ (c d : ℝ), c > 0 ∧ d > 0 ∧ 2 * c + 5 * d = 20 ∧ 1 / c + 1 / d = min) ∧
    min = (7 + 2 * Real.sqrt 10) / 20 :=
sorry

end NUMINAMATH_CALUDE_max_log_sum_min_reciprocal_sum_l953_95326


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l953_95356

theorem coefficient_of_x_squared (x : ℝ) :
  ∃ (k n : ℝ), (3 * x + 2) * (2 * x - 7) = 6 * x^2 + k * x + n := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l953_95356


namespace NUMINAMATH_CALUDE_probability_at_least_one_six_l953_95339

theorem probability_at_least_one_six (n : ℕ) (p : ℚ) : 
  n = 3 → p = 1/6 → (1 - (1 - p)^n) = 91/216 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_six_l953_95339


namespace NUMINAMATH_CALUDE_chocolate_difference_l953_95375

/-- The number of chocolates Robert ate -/
def robert_chocolates : ℕ := 10

/-- The number of chocolates Nickel ate -/
def nickel_chocolates : ℕ := 5

/-- Theorem stating the difference in chocolate consumption -/
theorem chocolate_difference : robert_chocolates - nickel_chocolates = 5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_difference_l953_95375


namespace NUMINAMATH_CALUDE_cos_alpha_plus_pi_implies_sin_alpha_plus_three_halves_pi_l953_95324

theorem cos_alpha_plus_pi_implies_sin_alpha_plus_three_halves_pi 
  (α : Real) 
  (h : Real.cos (α + Real.pi) = -2/3) : 
  Real.sin (α + 3/2 * Real.pi) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_plus_pi_implies_sin_alpha_plus_three_halves_pi_l953_95324


namespace NUMINAMATH_CALUDE_ln_1_1_approx_fourth_root_17_approx_l953_95330

-- Define the required accuracy
def accuracy : ℝ := 0.0001

-- Theorem for ln(1.1)
theorem ln_1_1_approx : |Real.log 1.1 - 0.0953| < accuracy := by sorry

-- Theorem for ⁴√17
theorem fourth_root_17_approx : |((17 : ℝ) ^ (1/4)) - 2.0305| < accuracy := by sorry

end NUMINAMATH_CALUDE_ln_1_1_approx_fourth_root_17_approx_l953_95330


namespace NUMINAMATH_CALUDE_rabbit_count_prove_rabbit_count_l953_95308

theorem rabbit_count : ℕ → ℕ → Prop :=
  fun total_white total_gray =>
    (∃ (caged_white : ℕ), caged_white = 6 ∧ total_white = caged_white + 9) ∧
    (∃ (caged_gray : ℕ), caged_gray = 4 ∧ total_gray = caged_gray) ∧
    (∃ (caged_white : ℕ), caged_white = 9 ∧ total_white = caged_white) ∧
    (∃ (caged_gray : ℕ), caged_gray = 4 ∧ total_gray = caged_gray + 16) →
    total_white + total_gray = 159

theorem prove_rabbit_count : ∃ (total_white total_gray : ℕ), rabbit_count total_white total_gray :=
  sorry

end NUMINAMATH_CALUDE_rabbit_count_prove_rabbit_count_l953_95308


namespace NUMINAMATH_CALUDE_couscous_first_shipment_l953_95372

theorem couscous_first_shipment (total_shipments : ℕ) 
  (shipment_a shipment_b first_shipment : ℝ) 
  (num_dishes : ℕ) (couscous_per_dish : ℝ) : 
  total_shipments = 3 →
  shipment_a = 13 →
  shipment_b = 45 →
  num_dishes = 13 →
  couscous_per_dish = 5 →
  first_shipment ≠ shipment_b →
  first_shipment = num_dishes * couscous_per_dish :=
by sorry

end NUMINAMATH_CALUDE_couscous_first_shipment_l953_95372


namespace NUMINAMATH_CALUDE_min_value_implies_a_l953_95347

def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

theorem min_value_implies_a (a : ℝ) :
  (∃ x₀ : ℝ, f a x₀ = 5 ∧ ∀ x : ℝ, f a x ≥ 5) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_implies_a_l953_95347


namespace NUMINAMATH_CALUDE_x_plus_y_power_2023_l953_95306

theorem x_plus_y_power_2023 (x y : ℝ) (h : |x - 2| + (y + 3)^2 = 0) : 
  (x + y)^2023 = -1 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_power_2023_l953_95306
