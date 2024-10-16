import Mathlib

namespace NUMINAMATH_CALUDE_monochromatic_triangle_exists_l3829_382968

-- Define a type for colors
inductive Color
| Red
| Blue
| Green

-- Define a type for the graph
def Graph := Fin 17 → Fin 17 → Color

-- Statement of the theorem
theorem monochromatic_triangle_exists (g : Graph) : 
  ∃ (a b c : Fin 17), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  g a b = g b c ∧ g b c = g a c :=
sorry

end NUMINAMATH_CALUDE_monochromatic_triangle_exists_l3829_382968


namespace NUMINAMATH_CALUDE_cos_540_degrees_l3829_382970

theorem cos_540_degrees : Real.cos (540 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_cos_540_degrees_l3829_382970


namespace NUMINAMATH_CALUDE_rebus_solution_l3829_382954

theorem rebus_solution : ∃! (A B C : ℕ), 
  (A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) ∧ 
  (A ≠ B ∧ B ≠ C ∧ A ≠ C) ∧
  (100 * A + 10 * B + A) + (100 * A + 10 * B + C) + (100 * A + 10 * C + C) = 1416 ∧
  A = 4 ∧ B = 7 ∧ C = 6 := by
sorry

end NUMINAMATH_CALUDE_rebus_solution_l3829_382954


namespace NUMINAMATH_CALUDE_opposite_of_two_thirds_l3829_382962

theorem opposite_of_two_thirds :
  -(2 : ℚ) / 3 = (-2 : ℚ) / 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_two_thirds_l3829_382962


namespace NUMINAMATH_CALUDE_prob_ratio_l3829_382942

/- Define the total number of cards -/
def total_cards : ℕ := 50

/- Define the number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/- Define the number of cards for each number -/
def cards_per_number : ℕ := 5

/- Define the number of cards drawn -/
def cards_drawn : ℕ := 5

/- Function to calculate the probability of drawing 5 cards of the same number -/
def prob_same_number : ℚ :=
  (distinct_numbers : ℚ) / Nat.choose total_cards cards_drawn

/- Function to calculate the probability of drawing 4 cards of one number and 1 of another -/
def prob_four_and_one : ℚ :=
  (distinct_numbers * (distinct_numbers - 1) * cards_per_number * cards_per_number : ℚ) / 
  Nat.choose total_cards cards_drawn

/- Theorem stating the ratio of probabilities -/
theorem prob_ratio : 
  prob_four_and_one / prob_same_number = 225 := by sorry

end NUMINAMATH_CALUDE_prob_ratio_l3829_382942


namespace NUMINAMATH_CALUDE_sum_sub_fixed_points_ln_exp_zero_l3829_382974

/-- A real number t is a sub-fixed point of function f if f(t) = -t -/
def IsSubFixedPoint (f : ℝ → ℝ) (t : ℝ) : Prop := f t = -t

/-- The sum of sub-fixed points of ln and exp -/
def SumSubFixedPoints : ℝ := sorry

theorem sum_sub_fixed_points_ln_exp_zero : SumSubFixedPoints = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_sub_fixed_points_ln_exp_zero_l3829_382974


namespace NUMINAMATH_CALUDE_star_identity_l3829_382911

/-- The binary operation * on pairs of real numbers -/
def star (p q : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * q.1, p.1 * q.2 + p.2 * q.1)

/-- The identity element for the star operation -/
def identity_element : ℝ × ℝ := (1, 0)

/-- Theorem stating that (1, 0) is the unique identity element for the star operation -/
theorem star_identity :
  ∀ p : ℝ × ℝ, star p identity_element = p ∧
  (∀ q : ℝ × ℝ, (∀ p : ℝ × ℝ, star p q = p) → q = identity_element) := by
  sorry

end NUMINAMATH_CALUDE_star_identity_l3829_382911


namespace NUMINAMATH_CALUDE_paula_shirts_bought_l3829_382912

def shirts_bought (initial_amount : ℕ) (shirt_cost : ℕ) (pants_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  (initial_amount - pants_cost - remaining_amount) / shirt_cost

theorem paula_shirts_bought :
  shirts_bought 109 11 13 74 = 2 := by
  sorry

end NUMINAMATH_CALUDE_paula_shirts_bought_l3829_382912


namespace NUMINAMATH_CALUDE_soda_cost_l3829_382939

theorem soda_cost (bill : ℕ) (change : ℕ) (num_sodas : ℕ) (h1 : bill = 20) (h2 : change = 14) (h3 : num_sodas = 3) :
  (bill - change) / num_sodas = 2 := by
sorry

end NUMINAMATH_CALUDE_soda_cost_l3829_382939


namespace NUMINAMATH_CALUDE_win_sector_area_l3829_382915

theorem win_sector_area (r : ℝ) (p : ℝ) (A_win : ℝ) :
  r = 8 →
  p = 1 / 4 →
  A_win = p * π * r^2 →
  A_win = 16 * π :=
by sorry

end NUMINAMATH_CALUDE_win_sector_area_l3829_382915


namespace NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l3829_382980

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Calculates the minimum number of voters needed to win -/
def min_voters_to_win (vs : VotingStructure) : Nat :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The giraffe beauty contest voting structure -/
def giraffe_contest : VotingStructure :=
  { total_voters := 135
  , num_districts := 5
  , precincts_per_district := 9
  , voters_per_precinct := 3 }

/-- Theorem stating the minimum number of voters needed for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win :
  min_voters_to_win giraffe_contest = 30 := by
  sorry

#eval min_voters_to_win giraffe_contest

end NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l3829_382980


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3829_382935

theorem coefficient_x_squared_in_expansion (x : ℝ) : 
  (Finset.range 7).sum (λ k => (Nat.choose 6 k) * (-2)^(6 - k) * x^k) = 
  240 * x^2 + (Finset.range 7).sum (λ k => if k ≠ 2 then (Nat.choose 6 k) * (-2)^(6 - k) * x^k else 0) := by
sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l3829_382935


namespace NUMINAMATH_CALUDE_unpainted_cubes_in_64_cube_l3829_382987

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  side_count : ℕ
  total_cubes : ℕ
  inner_side_count : ℕ

/-- The number of small cubes with no painted faces in a cut cube -/
def unpainted_cubes (c : CutCube) : ℕ :=
  c.inner_side_count ^ 3

/-- Theorem: In a cube cut into 64 equal smaller cubes, 
    the number of small cubes with no painted faces is 8 -/
theorem unpainted_cubes_in_64_cube :
  ∃ c : CutCube, c.side_count = 4 ∧ c.total_cubes = 64 ∧ c.inner_side_count = 2 ∧ 
  unpainted_cubes c = 8 :=
sorry

end NUMINAMATH_CALUDE_unpainted_cubes_in_64_cube_l3829_382987


namespace NUMINAMATH_CALUDE_airport_gate_change_probability_l3829_382961

/-- The number of gates in the airport -/
def num_gates : ℕ := 15

/-- The distance between adjacent gates in feet -/
def gate_distance : ℕ := 100

/-- The maximum walking distance in feet -/
def max_walk_distance : ℕ := 300

/-- The probability of selecting two different gates that are within the maximum walking distance -/
def gate_change_probability : ℚ := 37 / 105

theorem airport_gate_change_probability :
  let total_pairs := num_gates * (num_gates - 1)
  let valid_pairs := (
    4 * 3 + -- Extremity gates
    2 * 4 + -- Gates next to extremities
    9 * 6   -- Middle gates
  )
  valid_pairs / total_pairs = gate_change_probability := by sorry

end NUMINAMATH_CALUDE_airport_gate_change_probability_l3829_382961


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3829_382929

theorem fraction_meaningful (x : ℝ) : (∃ y : ℝ, y = 1 / (x + 1)) ↔ x ≠ -1 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3829_382929


namespace NUMINAMATH_CALUDE_chromium_percent_alloy1_l3829_382960

-- Define the weights and percentages
def weight_alloy1 : ℝ := 15
def weight_alloy2 : ℝ := 30
def chromium_percent_alloy2 : ℝ := 8
def chromium_percent_new : ℝ := 9.333333333333334

-- Theorem statement
theorem chromium_percent_alloy1 :
  ∃ (x : ℝ),
    x ≥ 0 ∧ x ≤ 100 ∧
    (x / 100 * weight_alloy1 + chromium_percent_alloy2 / 100 * weight_alloy2) / (weight_alloy1 + weight_alloy2) * 100 = chromium_percent_new ∧
    x = 12 :=
by sorry

end NUMINAMATH_CALUDE_chromium_percent_alloy1_l3829_382960


namespace NUMINAMATH_CALUDE_sum_of_y_values_l3829_382950

theorem sum_of_y_values (x y : ℝ) : 
  x^2 + x^2*y^2 + x^2*y^4 = 525 ∧ x + x*y + x*y^2 = 35 →
  ∃ y₁ y₂ : ℝ, (x^2 + x^2*y₁^2 + x^2*y₁^4 = 525 ∧ x + x*y₁ + x*y₁^2 = 35) ∧
             (x^2 + x^2*y₂^2 + x^2*y₂^4 = 525 ∧ x + x*y₂ + x*y₂^2 = 35) ∧
             y₁ + y₂ = 5/2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_y_values_l3829_382950


namespace NUMINAMATH_CALUDE_census_set_is_population_l3829_382903

/-- The term for the entire set of objects to be investigated in a census -/
def census_set : String := "Population"

/-- Theorem stating that the entire set of objects to be investigated in a census is called "Population" -/
theorem census_set_is_population : census_set = "Population" := by
  sorry

end NUMINAMATH_CALUDE_census_set_is_population_l3829_382903


namespace NUMINAMATH_CALUDE_building_floor_height_l3829_382914

/-- Proves that the height of each of the first 18 floors is 3 meters -/
theorem building_floor_height
  (total_floors : ℕ)
  (last_two_extra_height : ℝ)
  (total_height : ℝ)
  (h : ℝ)
  (h_total_floors : total_floors = 20)
  (h_last_two_extra : last_two_extra_height = 0.5)
  (h_total_height : total_height = 61)
  (h_height_equation : 18 * h + 2 * (h + last_two_extra_height) = total_height) :
  h = 3 := by
sorry

end NUMINAMATH_CALUDE_building_floor_height_l3829_382914


namespace NUMINAMATH_CALUDE_dice_sum_probability_l3829_382977

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the number of dice rolled
def num_dice : ℕ := 4

-- Define the total number of possible outcomes
def total_outcomes : ℕ := die_sides ^ num_dice

-- Define a function to calculate favorable outcomes
noncomputable def favorable_outcomes : ℕ := 
  -- This function would calculate the number of favorable outcomes
  -- Based on the problem, this should be 480, but we don't assume this knowledge
  sorry

-- Theorem statement
theorem dice_sum_probability : 
  (favorable_outcomes : ℚ) / total_outcomes = 10 / 27 := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_probability_l3829_382977


namespace NUMINAMATH_CALUDE_intersection_point_l3829_382921

def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 9*x + 15

theorem intersection_point (a b : ℝ) :
  (∀ x ≠ a, f x ≠ f a) ∧ 
  f a = b ∧ 
  f b = a ∧ 
  (∀ x y, f x = y ∧ f y = x → x = a ∧ y = b) →
  a = -1 ∧ b = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_point_l3829_382921


namespace NUMINAMATH_CALUDE_amit_work_days_l3829_382927

/-- Proves that Amit worked for 3 days before leaving the work -/
theorem amit_work_days (total_days : ℕ) (amit_rate : ℚ) (ananthu_rate : ℚ) :
  total_days = 75 ∧ amit_rate = 1 / 15 ∧ ananthu_rate = 1 / 90 →
  ∃ x : ℚ, x = 3 ∧ x * amit_rate + (total_days - x) * ananthu_rate = 1 :=
by sorry

end NUMINAMATH_CALUDE_amit_work_days_l3829_382927


namespace NUMINAMATH_CALUDE_misha_grade_size_l3829_382956

/-- The number of students in Misha's grade -/
def num_students : ℕ := 149

/-- Misha's position from the top of the grade -/
def position_from_top : ℕ := 75

/-- Misha's position from the bottom of the grade -/
def position_from_bottom : ℕ := 75

/-- Theorem: Given Misha's positions from top and bottom, prove the number of students in her grade -/
theorem misha_grade_size :
  position_from_top + position_from_bottom - 1 = num_students :=
by sorry

end NUMINAMATH_CALUDE_misha_grade_size_l3829_382956


namespace NUMINAMATH_CALUDE_quadratic_value_at_three_l3829_382923

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  min_value : ℝ
  min_x : ℝ
  y_at_zero : ℝ
  h : min_value = -4
  h' : min_x = -2
  h'' : y_at_zero = 8

/-- The value of y when x = 3 for the given quadratic function -/
def y_at_three (f : QuadraticFunction) : ℝ :=
  f.a * 3^2 + f.b * 3 + f.c

/-- Theorem stating that y = 71 when x = 3 for the given quadratic function -/
theorem quadratic_value_at_three (f : QuadraticFunction) : y_at_three f = 71 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_value_at_three_l3829_382923


namespace NUMINAMATH_CALUDE_systematic_sample_validity_l3829_382955

def isValidSystematicSample (sample : List Nat) (populationSize : Nat) (sampleSize : Nat) : Prop :=
  sample.length = sampleSize ∧
  sample.all (· ≤ populationSize) ∧
  sample.all (· > 0) ∧
  ∃ k : Nat, k > 0 ∧ List.zipWith (·-·) (sample.tail) sample = List.replicate (sampleSize - 1) k

theorem systematic_sample_validity :
  isValidSystematicSample [3, 13, 23, 33, 43] 50 5 :=
sorry

end NUMINAMATH_CALUDE_systematic_sample_validity_l3829_382955


namespace NUMINAMATH_CALUDE_video_views_proof_l3829_382936

/-- Calculates the total number of views for a video given initial views,
    a multiplier for increase after 4 days, and additional views after 2 more days. -/
def totalViews (initialViews : ℕ) (increaseMultiplier : ℕ) (additionalViews : ℕ) : ℕ :=
  initialViews + (increaseMultiplier * initialViews) + additionalViews

/-- Proves that given the specific conditions from the problem,
    the total number of views is 94000. -/
theorem video_views_proof :
  totalViews 4000 10 50000 = 94000 := by
  sorry

end NUMINAMATH_CALUDE_video_views_proof_l3829_382936


namespace NUMINAMATH_CALUDE_curve_equation_relationship_l3829_382932

-- Define the curve C as a set of points in 2D space
def C : Set (ℝ × ℝ) := sorry

-- Define the function f
def f : ℝ → ℝ → ℝ := sorry

-- State the theorem
theorem curve_equation_relationship :
  (∀ x y, f x y = 0 → (x, y) ∈ C) →
  (∀ x y, (x, y) ∉ C → f x y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_curve_equation_relationship_l3829_382932


namespace NUMINAMATH_CALUDE_expression_evaluations_l3829_382902

theorem expression_evaluations :
  (3 / Real.sqrt 3 + (Real.pi + Real.sqrt 3) ^ 0 + |Real.sqrt 3 - 2| = 3) ∧
  ((3 * Real.sqrt 12 - 2 * Real.sqrt (1/3) + Real.sqrt 48) / Real.sqrt 3 = 28/3) := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluations_l3829_382902


namespace NUMINAMATH_CALUDE_simplify_expressions_l3829_382985

theorem simplify_expressions (x y : ℝ) :
  (2 * (2 * x - y) - (x + y) = 3 * x - 3 * y) ∧
  (x^2 * y + (-3 * (2 * x * y - x^2 * y) - x * y) = 4 * x^2 * y - 7 * x * y) := by
sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3829_382985


namespace NUMINAMATH_CALUDE_tan_point_zero_l3829_382978

theorem tan_point_zero (φ : ℝ) : 
  (fun x => Real.tan (x + φ)) (π / 3) = 0 → φ = -π / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_point_zero_l3829_382978


namespace NUMINAMATH_CALUDE_system_not_unique_solution_l3829_382913

/-- The system of equations does not have a unique solution when k = 9 -/
theorem system_not_unique_solution (x y z : ℝ) (k m : ℝ) :
  (3 * (3 * x^2 + 4 * y^2) = 36) →
  (k * x^2 + 12 * y^2 = 30) →
  (m * x^3 - 2 * y^3 + z^2 = 24) →
  (k = 9 → ∃ (c : ℝ), c ≠ 0 ∧ (3 * x^2 + 4 * y^2 = c * (k * x^2 + 12 * y^2))) :=
by sorry

end NUMINAMATH_CALUDE_system_not_unique_solution_l3829_382913


namespace NUMINAMATH_CALUDE_complex_quotient_real_l3829_382953

theorem complex_quotient_real (t : ℝ) : 
  let z₁ : ℂ := 2*t + Complex.I
  let z₂ : ℂ := 1 - 2*Complex.I
  (∃ (r : ℝ), z₁ / z₂ = r) → t = -1/4 := by
sorry

end NUMINAMATH_CALUDE_complex_quotient_real_l3829_382953


namespace NUMINAMATH_CALUDE_short_trees_planted_count_l3829_382949

/-- The number of short trees planted in the park -/
def short_trees_planted (initial_short_trees final_short_trees : ℕ) : ℕ :=
  final_short_trees - initial_short_trees

/-- Theorem stating that the number of short trees planted is 64 -/
theorem short_trees_planted_count : short_trees_planted 31 95 = 64 := by
  sorry

end NUMINAMATH_CALUDE_short_trees_planted_count_l3829_382949


namespace NUMINAMATH_CALUDE_evaluate_expression_l3829_382918

theorem evaluate_expression : (-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3829_382918


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_squared_l3829_382994

theorem largest_x_sqrt_3x_eq_5x_squared :
  let f : ℝ → ℝ := λ x => Real.sqrt (3 * x) - 5 * x^2
  ∃ (max_x : ℝ), max_x = (3/25)^(1/3) ∧
    (∀ x : ℝ, f x = 0 → x ≤ max_x) ∧
    f max_x = 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_5x_squared_l3829_382994


namespace NUMINAMATH_CALUDE_chord_equation_l3829_382981

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given circle, point P, prove that the line AB passing through P has the specified equation -/
theorem chord_equation (c : Circle) (p : ℝ × ℝ) : 
  c.center = (2, 0) → 
  c.radius = 4 → 
  p = (3, 1) → 
  ∃ (l : Line), l.a = 1 ∧ l.b = 1 ∧ l.c = -4 := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l3829_382981


namespace NUMINAMATH_CALUDE_two_solutions_l3829_382926

/-- Sum of digits function -/
def T (n : ℕ) : ℕ := sorry

/-- The number of solutions to the equation -/
def num_solutions : ℕ := 2

/-- Theorem stating that there are exactly 2 solutions -/
theorem two_solutions :
  (∃ (S : Finset ℕ), S.card = num_solutions ∧
    (∀ n, n ∈ S ↔ (n : ℕ) + T n + T (T n) = 2187) ∧
    (∀ n ∈ S, n > 0)) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_l3829_382926


namespace NUMINAMATH_CALUDE_perpendicular_vectors_t_value_l3829_382959

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_t_value :
  ∀ t : ℝ, perpendicular (3, 1) (t, -3) → t = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_t_value_l3829_382959


namespace NUMINAMATH_CALUDE_weighted_sum_inequality_l3829_382982

theorem weighted_sum_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (order : a ≤ b ∧ b ≤ c ∧ c ≤ d)
  (sum_cond : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_weighted_sum_inequality_l3829_382982


namespace NUMINAMATH_CALUDE_point_on_curve_iff_satisfies_equation_l3829_382984

-- Define a curve C in 2D space
def Curve (F : ℝ → ℝ → ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | F p.1 p.2 = 0}

-- Define a point P
def Point (a b : ℝ) : ℝ × ℝ := (a, b)

-- Theorem statement
theorem point_on_curve_iff_satisfies_equation (F : ℝ → ℝ → ℝ) (a b : ℝ) :
  Point a b ∈ Curve F ↔ F a b = 0 := by
  sorry

end NUMINAMATH_CALUDE_point_on_curve_iff_satisfies_equation_l3829_382984


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3829_382989

/-- Given an arithmetic sequence {a_n} where a_3 = 1 and a_4 + a_10 = 18, prove that a_1 = -3 -/
theorem arithmetic_sequence_first_term 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a3 : a 3 = 1) 
  (h_sum : a 4 + a 10 = 18) : 
  a 1 = -3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l3829_382989


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l3829_382920

theorem inverse_proportion_inequality (k : ℝ) (y₁ y₂ y₃ : ℝ)
  (h_k : k > 0)
  (h_y₁ : y₁ = k / (-3))
  (h_y₂ : y₂ = k / (-1))
  (h_y₃ : y₃ = k / 1) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l3829_382920


namespace NUMINAMATH_CALUDE_a_values_l3829_382937

def P : Set ℝ := {x | x^2 = 1}
def Q (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem a_values (a : ℝ) : Q a ⊆ P → a = 0 ∨ a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_a_values_l3829_382937


namespace NUMINAMATH_CALUDE_parallelogram_z_range_l3829_382972

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (-1, 2)
def B : ℝ × ℝ := (3, 4)
def C : ℝ × ℝ := (4, -2)

-- Define the function z
def z (x y : ℝ) : ℝ := 2 * x - 5 * y

-- Theorem statement
theorem parallelogram_z_range :
  ∀ (x y : ℝ), 
  (∃ (t₁ t₂ t₃ : ℝ), 0 ≤ t₁ ∧ 0 ≤ t₂ ∧ 0 ≤ t₃ ∧ t₁ + t₂ + t₃ ≤ 1 ∧
    (x, y) = t₁ • A + t₂ • B + t₃ • C + (1 - t₁ - t₂ - t₃) • (A + C - B)) →
  -14 ≤ z x y ∧ z x y ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_parallelogram_z_range_l3829_382972


namespace NUMINAMATH_CALUDE_one_quadrilateral_is_rhombus_l3829_382925

-- Define the properties of quadrilaterals
structure QuadrilateralProperties where
  opposite_sides_equal : Bool
  opposite_sides_parallel : Bool
  adjacent_sides_equal : Bool
  diagonals_perpendicular_and_bisected : Bool

-- Define a function to check if a quadrilateral with given properties is a rhombus
def is_rhombus (props : QuadrilateralProperties) : Bool :=
  (props.opposite_sides_equal && props.adjacent_sides_equal) ||
  (props.adjacent_sides_equal && props.diagonals_perpendicular_and_bisected) ||
  (props.opposite_sides_equal && props.diagonals_perpendicular_and_bisected)

-- Theorem statement
theorem one_quadrilateral_is_rhombus 
  (quad1 props1 quad2 props2 : QuadrilateralProperties) 
  (h1 : props1.opposite_sides_equal + props1.opposite_sides_parallel + 
        props1.adjacent_sides_equal + props1.diagonals_perpendicular_and_bisected = 2)
  (h2 : props2.opposite_sides_equal + props2.opposite_sides_parallel + 
        props2.adjacent_sides_equal + props2.diagonals_perpendicular_and_bisected = 2)
  (h3 : props1.opposite_sides_equal + props2.opposite_sides_equal = 1)
  (h4 : props1.opposite_sides_parallel + props2.opposite_sides_parallel = 1)
  (h5 : props1.adjacent_sides_equal + props2.adjacent_sides_equal = 1)
  (h6 : props1.diagonals_perpendicular_and_bisected + props2.diagonals_perpendicular_and_bisected = 1) :
  is_rhombus props1 ∨ is_rhombus props2 :=
sorry

end NUMINAMATH_CALUDE_one_quadrilateral_is_rhombus_l3829_382925


namespace NUMINAMATH_CALUDE_triangle_theorem_l3829_382905

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  h1 : 0 < a ∧ 0 < b ∧ 0 < c
  h2 : 0 < A ∧ 0 < B ∧ 0 < C
  h3 : A + B + C = π

/-- The main theorem -/
theorem triangle_theorem (t : Triangle) (h : t.a + t.a * Real.cos t.C = Real.sqrt 3 * t.c * Real.sin t.A) :
  t.C = π / 3 ∧
  ∃ (D : ℝ), (D > 0 ∧ 2 * Real.sqrt 3 = D ∧
    ∀ (a b : ℝ), (a > 0 ∧ b > 0 → 2 * a + b ≥ 6 + 4 * Real.sqrt 2)) := by
  sorry

end NUMINAMATH_CALUDE_triangle_theorem_l3829_382905


namespace NUMINAMATH_CALUDE_marble_bag_problem_l3829_382924

theorem marble_bag_problem :
  ∀ (r b : ℕ),
  r + b > 0 →
  r = (r + b) / 3 →
  (r - 3) = (r + b - 3) / 4 →
  r = (r + b - 2) / 3 →
  r + b = 19 := by
sorry

end NUMINAMATH_CALUDE_marble_bag_problem_l3829_382924


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_specific_pyramid_l3829_382976

/-- Regular hexagonal pyramid -/
structure RegularHexagonalPyramid where
  baseEdgeLength : ℝ
  volume : ℝ

/-- Lateral surface area of a regular hexagonal pyramid -/
def lateralSurfaceArea (pyramid : RegularHexagonalPyramid) : ℝ :=
  sorry

theorem lateral_surface_area_of_specific_pyramid :
  let pyramid : RegularHexagonalPyramid :=
    { baseEdgeLength := 2
    , volume := 2 * Real.sqrt 3 }
  lateralSurfaceArea pyramid = 12 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_specific_pyramid_l3829_382976


namespace NUMINAMATH_CALUDE_inequality_proof_l3829_382973

theorem inequality_proof (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  Real.sqrt (x₁ * x₂) < (x₁ - x₂) / (Real.log x₁ - Real.log x₂) ∧
  (x₁ - x₂) / (Real.log x₁ - Real.log x₂) < (x₁ + x₂) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3829_382973


namespace NUMINAMATH_CALUDE_division_problem_l3829_382930

theorem division_problem (L S q : ℕ) : 
  L - S = 1335 → 
  L = 1584 → 
  L = S * q + 15 → 
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3829_382930


namespace NUMINAMATH_CALUDE_total_balloons_l3829_382945

theorem total_balloons (joan_balloons melanie_balloons : ℕ) 
  (h1 : joan_balloons = 40)
  (h2 : melanie_balloons = 41) : 
  joan_balloons + melanie_balloons = 81 := by
  sorry

end NUMINAMATH_CALUDE_total_balloons_l3829_382945


namespace NUMINAMATH_CALUDE_apple_eating_duration_l3829_382916

/-- The number of apples Eva needs to buy -/
def total_apples : ℕ := 14

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks Eva should eat an apple every day -/
def weeks_to_eat_apples : ℚ := total_apples / days_per_week

theorem apple_eating_duration : weeks_to_eat_apples = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_eating_duration_l3829_382916


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3829_382999

theorem arithmetic_calculation : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3829_382999


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3829_382979

theorem solve_exponential_equation :
  ∃ x : ℝ, (3 : ℝ)^x * (3 : ℝ)^x * (3 : ℝ)^x * (3 : ℝ)^x = (81 : ℝ)^3 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3829_382979


namespace NUMINAMATH_CALUDE_min_difference_of_h_l3829_382967

noncomputable section

variable (a : ℝ) (x₁ x₂ : ℝ)

def h (x : ℝ) : ℝ := x - 1/x + a * Real.log x

theorem min_difference_of_h (ha : a > 0) (hx₁ : 0 < x₁ ∧ x₁ ≤ 1/Real.exp 1)
  (hroots : x₁^2 + a*x₁ + 1 = 0 ∧ x₂^2 + a*x₂ + 1 = 0) :
  ∃ (m : ℝ), m = 4/Real.exp 1 ∧ ∀ y₁ y₂, 
    (0 < y₁ ∧ y₁ ≤ 1/Real.exp 1) → 
    (y₁^2 + a*y₁ + 1 = 0 ∧ y₂^2 + a*y₂ + 1 = 0) → 
    h a y₁ - h a y₂ ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_difference_of_h_l3829_382967


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3829_382909

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  a 2 + a 3 = 4 →
  a 4 + a 5 = 6 →
  a 9 + a 10 = 11 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3829_382909


namespace NUMINAMATH_CALUDE_marathon_finishers_l3829_382995

/-- Proves that 563 people finished the marathon given the conditions --/
theorem marathon_finishers :
  ∀ (finished : ℕ),
  (finished + (finished + 124) = 1250) →
  finished = 563 := by
  sorry

end NUMINAMATH_CALUDE_marathon_finishers_l3829_382995


namespace NUMINAMATH_CALUDE_miles_driven_l3829_382996

-- Define the efficiency of the car in miles per gallon
def miles_per_gallon : ℝ := 45

-- Define the price of gas per gallon
def price_per_gallon : ℝ := 5

-- Define the amount spent on gas
def amount_spent : ℝ := 25

-- Theorem to prove
theorem miles_driven : 
  miles_per_gallon * (amount_spent / price_per_gallon) = 225 := by
sorry

end NUMINAMATH_CALUDE_miles_driven_l3829_382996


namespace NUMINAMATH_CALUDE_problem_statement_l3829_382997

theorem problem_statement (x y : ℕ) (h1 : y > 3) 
  (h2 : x^2 + y^4 = 2*((x-6)^2 + (y+1)^2)) : 
  x^2 + y^4 = 1994 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3829_382997


namespace NUMINAMATH_CALUDE_quadratic_roots_midpoint_l3829_382943

theorem quadratic_roots_midpoint (a b : ℝ) (x₁ x₂ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  (f 2014 = f 2016) →
  (x₁^2 + a*x₁ + b = 0) →
  (x₂^2 + a*x₂ + b = 0) →
  (x₁ + x₂) / 2 = 2015 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_midpoint_l3829_382943


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3829_382908

theorem sufficient_not_necessary_condition (a b : ℝ) : 
  (b < -4 → ∀ a, |a| + |b| > 4) ∧ 
  (∃ a b, |a| + |b| > 4 ∧ b ≥ -4) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3829_382908


namespace NUMINAMATH_CALUDE_inequality_proof_l3829_382919

theorem inequality_proof (a b : ℝ) (ha : -4 < a ∧ a < 0) (hb : -4 < b ∧ b < 0) :
  2 * |a - b| < |a * b + 2 * a + 2 * b| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3829_382919


namespace NUMINAMATH_CALUDE_similar_triangle_shortest_side_l3829_382904

theorem similar_triangle_shortest_side 
  (a b c : ℝ) 
  (h₁ : a^2 + b^2 = c^2) 
  (h₂ : a = 21) 
  (h₃ : c = 29) 
  (h₄ : a ≤ b) 
  (k : ℝ) 
  (h₅ : k * c = 87) : 
  k * a = 60 := by
sorry

end NUMINAMATH_CALUDE_similar_triangle_shortest_side_l3829_382904


namespace NUMINAMATH_CALUDE_julio_fishing_result_l3829_382951

/-- Calculates the number of fish Julio has after fishing for a given duration and losing some fish. -/
def fish_remaining (rate : ℕ) (duration : ℕ) (loss : ℕ) : ℕ :=
  rate * duration - loss

/-- Proves that Julio has 48 fish after fishing for 9 hours at a rate of 7 fish per hour and losing 15 fish. -/
theorem julio_fishing_result :
  fish_remaining 7 9 15 = 48 := by
  sorry

end NUMINAMATH_CALUDE_julio_fishing_result_l3829_382951


namespace NUMINAMATH_CALUDE_sidney_thursday_jumping_jacks_l3829_382963

/-- The number of jumping jacks Sidney did on Thursday -/
def thursday_jumping_jacks : ℕ := 50

/-- The number of jumping jacks Sidney did on Monday -/
def monday_jumping_jacks : ℕ := 20

/-- The number of jumping jacks Sidney did on Tuesday -/
def tuesday_jumping_jacks : ℕ := 36

/-- The number of jumping jacks Sidney did on Wednesday -/
def wednesday_jumping_jacks : ℕ := 40

/-- The total number of jumping jacks Brooke did -/
def brooke_total_jumping_jacks : ℕ := 438

theorem sidney_thursday_jumping_jacks :
  thursday_jumping_jacks = 
    brooke_total_jumping_jacks / 3 - 
    (monday_jumping_jacks + tuesday_jumping_jacks + wednesday_jumping_jacks) := by
  sorry

end NUMINAMATH_CALUDE_sidney_thursday_jumping_jacks_l3829_382963


namespace NUMINAMATH_CALUDE_largest_prime_with_square_conditions_l3829_382964

theorem largest_prime_with_square_conditions : 
  ∀ p : ℕ, 
    p.Prime → 
    (∃ x : ℕ, (p + 1) / 2 = x^2) → 
    (∃ y : ℕ, (p^2 + 1) / 2 = y^2) → 
    p ≤ 7 := by
  sorry

end NUMINAMATH_CALUDE_largest_prime_with_square_conditions_l3829_382964


namespace NUMINAMATH_CALUDE_prime_sum_squares_l3829_382990

theorem prime_sum_squares (a b c d : ℕ) : 
  Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧ Nat.Prime d ∧ 
  a > 3 ∧ b > 6 ∧ c > 12 ∧
  a^2 - b^2 + c^2 - d^2 = 1749 →
  a^2 + b^2 + c^2 + d^2 = 2143 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_squares_l3829_382990


namespace NUMINAMATH_CALUDE_xy_value_l3829_382993

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 + y^2 = 58) : x * y = 21 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3829_382993


namespace NUMINAMATH_CALUDE_solve_equation_l3829_382900

theorem solve_equation (x : ℝ) : 9 / (5 + 3 / x) = 1 → x = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3829_382900


namespace NUMINAMATH_CALUDE_marble_probability_l3829_382969

/-- The probability of drawing either a green or purple marble from a bag -/
theorem marble_probability (green purple orange : ℕ) 
  (h_green : green = 5)
  (h_purple : purple = 4)
  (h_orange : orange = 6) :
  (green + purple : ℚ) / (green + purple + orange) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l3829_382969


namespace NUMINAMATH_CALUDE_slope_through_origin_and_point_l3829_382965

/-- The slope of a line passing through (0, 0) and (5, 1) is 1/5 -/
theorem slope_through_origin_and_point :
  let x1 : ℝ := 0
  let y1 : ℝ := 0
  let x2 : ℝ := 5
  let y2 : ℝ := 1
  let slope : ℝ := (y2 - y1) / (x2 - x1)
  slope = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_slope_through_origin_and_point_l3829_382965


namespace NUMINAMATH_CALUDE_spatial_pythagorean_quadruplet_l3829_382944

theorem spatial_pythagorean_quadruplet (m n p q : ℤ) : 
  let x := 2 * m * p + 2 * n * q
  let y := |2 * m * q - 2 * n * p|
  let z := |m^2 + n^2 - p^2 - q^2|
  let u := m^2 + n^2 + p^2 + q^2
  (x^2 + y^2 + z^2 = u^2) →
  (∀ d : ℤ, d > 1 → ¬(d ∣ x ∧ d ∣ y ∧ d ∣ z ∧ d ∣ u)) →
  (∀ d : ℤ, d > 1 → ¬(d ∣ m ∧ d ∣ n ∧ d ∣ p ∧ d ∣ q)) :=
by sorry

end NUMINAMATH_CALUDE_spatial_pythagorean_quadruplet_l3829_382944


namespace NUMINAMATH_CALUDE_complex_equation_roots_l3829_382992

theorem complex_equation_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = 1 - I ∧ z₂ = -3 + I ∧ 
  (z₁^2 + 2*z₁ = 3 - 4*I) ∧ 
  (z₂^2 + 2*z₂ = 3 - 4*I) := by
sorry

end NUMINAMATH_CALUDE_complex_equation_roots_l3829_382992


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3829_382983

theorem polynomial_division_remainder : ∃ (r : ℚ),
  ∀ (z : ℚ), 4 * z^3 - 5 * z^2 - 18 * z + 4 = (4 * z + 6) * (z^2 - 4 * z + 2/3) + r :=
by
  use 10/3
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3829_382983


namespace NUMINAMATH_CALUDE_number_of_divisors_36_l3829_382910

/-- The number of positive divisors of 36 is 9. -/
theorem number_of_divisors_36 : (Finset.filter (· ∣ 36) (Finset.range 37)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_36_l3829_382910


namespace NUMINAMATH_CALUDE_total_students_l3829_382946

theorem total_students (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 16) 
  (h2 : rank_from_left = 6) : 
  rank_from_right + rank_from_left - 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l3829_382946


namespace NUMINAMATH_CALUDE_marble_problem_l3829_382966

theorem marble_problem (r b : ℕ) : 
  ((r - 3 : ℚ) / (r + b - 3) = 1 / 10) →
  ((r : ℚ) / (r + b - 3) = 1 / 4) →
  r + b = 13 := by
  sorry

end NUMINAMATH_CALUDE_marble_problem_l3829_382966


namespace NUMINAMATH_CALUDE_unique_divisible_by_36_l3829_382917

def is_divisible_by_36 (n : ℕ) : Prop := n % 36 = 0

def five_digit_number (x : ℕ) : ℕ := x * 10000 + 9500 + x

theorem unique_divisible_by_36 :
  ∃! x : ℕ, x < 10 ∧ is_divisible_by_36 (five_digit_number x) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_36_l3829_382917


namespace NUMINAMATH_CALUDE_smallest_max_sum_l3829_382907

theorem smallest_max_sum (a b c d e : ℕ+) (sum_eq : a + b + c + d + e = 2023) :
  let M := max (a + b) (max (b + c) (max (c + d) (d + e)))
  405 ≤ M ∧ ∃ (a' b' c' d' e' : ℕ+), a' + b' + c' + d' + e' = 2023 ∧
    max (a' + b') (max (b' + c') (max (c' + d') (d' + e'))) = 405 := by
  sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l3829_382907


namespace NUMINAMATH_CALUDE_base_r_square_property_l3829_382998

/-- A natural number x is representable as a two-digit number with identical digits in base r -/
def is_two_digit_identical (x r : ℕ) : Prop :=
  ∃ a : ℕ, 0 < a ∧ a < r ∧ x = a * (r + 1)

/-- A natural number y is representable as a four-digit number in base r with form b00b -/
def is_four_digit_b00b (y r : ℕ) : Prop :=
  ∃ b : ℕ, 0 < b ∧ b < r ∧ y = b * (r^3 + 1)

/-- The main theorem -/
theorem base_r_square_property (r : ℕ) (hr : r ≤ 100) :
  (∃ x : ℕ, is_two_digit_identical x r ∧ is_four_digit_b00b (x^2) r) →
  r = 2 ∨ r = 23 :=
by sorry

end NUMINAMATH_CALUDE_base_r_square_property_l3829_382998


namespace NUMINAMATH_CALUDE_three_card_selections_count_l3829_382991

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- The number of ways to select and order three different cards from a standard deck -/
def ThreeCardSelections : ℕ := StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)

/-- Theorem: The number of ways to select and order three different cards from a standard 52-card deck is 132600 -/
theorem three_card_selections_count : ThreeCardSelections = 132600 := by
  sorry

end NUMINAMATH_CALUDE_three_card_selections_count_l3829_382991


namespace NUMINAMATH_CALUDE_area_KLMQ_is_ten_l3829_382948

structure Rectangle where
  width : ℝ
  height : ℝ

def area (r : Rectangle) : ℝ := r.width * r.height

theorem area_KLMQ_is_ten (JLMR JKQR : Rectangle) 
  (h1 : JLMR.width = 2)
  (h2 : JKQR.height = 3)
  (h3 : JLMR.height = 8) :
  ∃ KLMQ : Rectangle, area KLMQ = 10 :=
sorry

end NUMINAMATH_CALUDE_area_KLMQ_is_ten_l3829_382948


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3829_382934

/-- Two vectors a and b in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

theorem perpendicular_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (-Real.sqrt 3, m)
  let b : ℝ × ℝ := (2, 1)
  perpendicular a b → m = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l3829_382934


namespace NUMINAMATH_CALUDE_set_inclusion_iff_m_range_l3829_382971

theorem set_inclusion_iff_m_range (m : ℝ) :
  ({x : ℝ | x^2 - 2*x - 3 ≤ 0} ⊆ {x : ℝ | |x - m| > 3}) ↔ 
  (m < -4 ∨ m > 6) :=
sorry

end NUMINAMATH_CALUDE_set_inclusion_iff_m_range_l3829_382971


namespace NUMINAMATH_CALUDE_total_days_2010_to_2014_l3829_382922

def days_in_year (year : ℕ) : ℕ :=
  if year = 2012 then 366 else 365

def years_range : List ℕ := [2010, 2011, 2012, 2013, 2014]

theorem total_days_2010_to_2014 :
  (years_range.map days_in_year).sum = 1826 := by sorry

end NUMINAMATH_CALUDE_total_days_2010_to_2014_l3829_382922


namespace NUMINAMATH_CALUDE_coconut_grove_average_yield_l3829_382941

/-- The yield of coconuts per year for a group of trees -/
structure CoconutYield where
  trees : ℕ
  nuts_per_year : ℕ

/-- The total yield of coconuts from multiple groups of trees -/
def total_yield (yields : List CoconutYield) : ℕ :=
  yields.map (λ y => y.trees * y.nuts_per_year) |>.sum

/-- The total number of trees from multiple groups -/
def total_trees (yields : List CoconutYield) : ℕ :=
  yields.map (λ y => y.trees) |>.sum

/-- The average yield per tree per year -/
def average_yield (yields : List CoconutYield) : ℚ :=
  (total_yield yields : ℚ) / (total_trees yields : ℚ)

theorem coconut_grove_average_yield : 
  let yields : List CoconutYield := [
    { trees := 3, nuts_per_year := 60 },
    { trees := 2, nuts_per_year := 120 },
    { trees := 1, nuts_per_year := 180 }
  ]
  average_yield yields = 100 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_average_yield_l3829_382941


namespace NUMINAMATH_CALUDE_arithmetic_sequence_angles_l3829_382947

theorem arithmetic_sequence_angles (angles : Fin 5 → ℝ) : 
  (∀ i j : Fin 5, i < j → angles i < angles j) →  -- angles are strictly increasing
  (∀ i : Fin 4, angles (i + 1) - angles i = angles (i + 2) - angles (i + 1)) →  -- arithmetic sequence
  angles 0 = 25 →  -- smallest angle
  angles 4 = 105 →  -- largest angle
  ∀ i : Fin 4, angles (i + 1) - angles i = 20 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_angles_l3829_382947


namespace NUMINAMATH_CALUDE_model_b_piano_keys_l3829_382940

theorem model_b_piano_keys : ∃ (x : ℕ), 
  (104 : ℕ) = 2 * x - 72 → x = 88 := by
  sorry

end NUMINAMATH_CALUDE_model_b_piano_keys_l3829_382940


namespace NUMINAMATH_CALUDE_square_area_equal_perimeter_l3829_382906

/-- Given a triangle with sides 7.5 cm, 9.5 cm, and 12 cm, and a square with the same perimeter as this triangle, the area of the square is 52.5625 square centimeters. -/
theorem square_area_equal_perimeter (a b c s : ℝ) : 
  a = 7.5 → b = 9.5 → c = 12 → s * 4 = a + b + c → s^2 = 52.5625 := by
  sorry

end NUMINAMATH_CALUDE_square_area_equal_perimeter_l3829_382906


namespace NUMINAMATH_CALUDE_cubic_inequality_l3829_382928

theorem cubic_inequality (x : ℝ) : x^3 - 12*x^2 > -36*x ↔ x ∈ Set.Ioo 0 6 ∪ Set.Ioi 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l3829_382928


namespace NUMINAMATH_CALUDE_positive_numbers_l3829_382958

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : b * c + c * a + a * b > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l3829_382958


namespace NUMINAMATH_CALUDE_inequality_proof_l3829_382938

theorem inequality_proof (a b c d : ℝ) : 
  (a^2 + b^2 + 1) * (c^2 + d^2 + 1) ≥ 2 * (a + c) * (b + d) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3829_382938


namespace NUMINAMATH_CALUDE_same_solutions_quadratic_l3829_382931

theorem same_solutions_quadratic (b c : ℝ) : 
  (∀ x : ℝ, |x - 5| = 2 ↔ x^2 + b*x + c = 0) → 
  b = -10 ∧ c = 21 := by
sorry

end NUMINAMATH_CALUDE_same_solutions_quadratic_l3829_382931


namespace NUMINAMATH_CALUDE_sam_found_35_seashells_l3829_382957

/-- The number of seashells Joan found -/
def joans_seashells : ℕ := 18

/-- The total number of seashells Sam and Joan found together -/
def total_seashells : ℕ := 53

/-- The number of seashells Sam found -/
def sams_seashells : ℕ := total_seashells - joans_seashells

theorem sam_found_35_seashells : sams_seashells = 35 := by
  sorry

end NUMINAMATH_CALUDE_sam_found_35_seashells_l3829_382957


namespace NUMINAMATH_CALUDE_cars_per_client_l3829_382933

theorem cars_per_client 
  (total_cars : ℕ) 
  (total_clients : ℕ) 
  (selections_per_car : ℕ) 
  (h1 : total_cars = 12) 
  (h2 : total_clients = 9) 
  (h3 : selections_per_car = 3) : 
  (total_cars * selections_per_car) / total_clients = 4 := by
  sorry

end NUMINAMATH_CALUDE_cars_per_client_l3829_382933


namespace NUMINAMATH_CALUDE_reflection_about_x_axis_l3829_382901

/-- Represents a parabola in the Cartesian coordinate system -/
structure Parabola where
  f : ℝ → ℝ

/-- Reflects a parabola about the x-axis -/
def reflect_x (p : Parabola) : Parabola :=
  { f := λ x => -(p.f x) }

/-- The original parabola y = x^2 + x - 2 -/
def original_parabola : Parabola :=
  { f := λ x => x^2 + x - 2 }

/-- The expected reflected parabola y = -x^2 - x + 2 -/
def expected_reflected_parabola : Parabola :=
  { f := λ x => -x^2 - x + 2 }

theorem reflection_about_x_axis :
  reflect_x original_parabola = expected_reflected_parabola :=
by sorry

end NUMINAMATH_CALUDE_reflection_about_x_axis_l3829_382901


namespace NUMINAMATH_CALUDE_alice_additional_spend_l3829_382986

/-- The amount Alice needs to spend for free delivery -/
def free_delivery_threshold : ℚ := 35

/-- The cost of chicken per pound -/
def chicken_price : ℚ := 6

/-- The amount of chicken in pounds -/
def chicken_amount : ℚ := 3/2

/-- The cost of lettuce -/
def lettuce_price : ℚ := 3

/-- The cost of cherry tomatoes -/
def tomatoes_price : ℚ := 5/2

/-- The cost of one sweet potato -/
def sweet_potato_price : ℚ := 3/4

/-- The number of sweet potatoes -/
def sweet_potato_count : ℕ := 4

/-- The cost of one head of broccoli -/
def broccoli_price : ℚ := 2

/-- The number of broccoli heads -/
def broccoli_count : ℕ := 2

/-- The cost of Brussel sprouts -/
def brussel_sprouts_price : ℚ := 5/2

/-- The total cost of items in Alice's cart -/
def cart_total : ℚ :=
  chicken_price * chicken_amount + lettuce_price + tomatoes_price +
  sweet_potato_price * sweet_potato_count + broccoli_price * broccoli_count +
  brussel_sprouts_price

/-- The additional amount Alice needs to spend for free delivery -/
def additional_spend : ℚ := free_delivery_threshold - cart_total

theorem alice_additional_spend :
  additional_spend = 11 := by sorry

end NUMINAMATH_CALUDE_alice_additional_spend_l3829_382986


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l3829_382975

/-- A sequence is arithmetic if the difference between consecutive terms is constant. -/
def IsArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The general term of the sequence. -/
def a (n : ℕ) : ℝ := 2 * n + 5

/-- The theorem stating that the given sequence is arithmetic with first term 7 and common difference 2. -/
theorem sequence_is_arithmetic :
  IsArithmeticSequence a ∧ a 1 = 7 ∧ ∀ n : ℕ, a (n + 1) - a n = 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l3829_382975


namespace NUMINAMATH_CALUDE_M_subset_N_l3829_382952

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def N : Set ℝ := {y | ∃ x : ℝ, y = x^2}

-- State the theorem
theorem M_subset_N : M ⊆ N := by sorry

end NUMINAMATH_CALUDE_M_subset_N_l3829_382952


namespace NUMINAMATH_CALUDE_matrix_multiplication_result_l3829_382988

theorem matrix_multiplication_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; 4, -2]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![7, -3; 2, 2]
  A * B = !![23, -7; 24, -16] := by
  sorry

end NUMINAMATH_CALUDE_matrix_multiplication_result_l3829_382988
