import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l1441_144172

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l1441_144172


namespace NUMINAMATH_CALUDE_max_card_arrangement_l1441_144118

/-- A type representing the cards with numbers from 1 to 9 -/
inductive Card : Type
| one : Card
| two : Card
| three : Card
| four : Card
| five : Card
| six : Card
| seven : Card
| eight : Card
| nine : Card

/-- Convert a Card to its corresponding natural number -/
def card_to_nat (c : Card) : Nat :=
  match c with
  | Card.one => 1
  | Card.two => 2
  | Card.three => 3
  | Card.four => 4
  | Card.five => 5
  | Card.six => 6
  | Card.seven => 7
  | Card.eight => 8
  | Card.nine => 9

/-- Check if one card is divisible by another -/
def is_divisible (a b : Card) : Prop :=
  (card_to_nat a) % (card_to_nat b) = 0 ∨ (card_to_nat b) % (card_to_nat a) = 0

/-- A valid arrangement of cards -/
def valid_arrangement (arr : List Card) : Prop :=
  ∀ i, i + 1 < arr.length → is_divisible (arr.get ⟨i, by sorry⟩) (arr.get ⟨i + 1, by sorry⟩)

/-- The main theorem -/
theorem max_card_arrangement :
  ∃ (arr : List Card), arr.length = 8 ∧ valid_arrangement arr ∧
  ∀ (arr' : List Card), valid_arrangement arr' → arr'.length ≤ 8 :=
sorry

end NUMINAMATH_CALUDE_max_card_arrangement_l1441_144118


namespace NUMINAMATH_CALUDE_min_value_expression_l1441_144109

theorem min_value_expression (a b c d : ℝ) (hb : b ≠ 0) (horder : b > c ∧ c > a ∧ a > d) :
  ((2*a + b)^2 + (b - 2*c)^2 + (c - a)^2 + 3*d^2) / b^2 ≥ 49/36 :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l1441_144109


namespace NUMINAMATH_CALUDE_not_all_data_sets_have_regression_equation_l1441_144121

-- Define a type for data sets
structure DataSet where
  -- Add necessary fields to represent a data set
  dummy : Unit

-- Define a predicate for the existence of a regression equation
def has_regression_equation (ds : DataSet) : Prop :=
  -- Add necessary conditions for a data set to have a regression equation
  sorry

-- Theorem stating that not every data set has a regression equation
theorem not_all_data_sets_have_regression_equation :
  ¬ (∀ ds : DataSet, has_regression_equation ds) := by
  sorry

end NUMINAMATH_CALUDE_not_all_data_sets_have_regression_equation_l1441_144121


namespace NUMINAMATH_CALUDE_seven_consecutive_integers_product_first_57_integers_product_l1441_144136

-- Define a function to calculate the number of trailing zeros
def trailingZeros (n : ℕ) : ℕ := sorry

-- Theorem for seven consecutive integers
theorem seven_consecutive_integers_product (k : ℕ) :
  ∃ m : ℕ, m > 0 ∧ trailingZeros ((k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6) * (k + 7)) ≥ m :=
sorry

-- Theorem for the product of first 57 positive integers
theorem first_57_integers_product :
  trailingZeros (Nat.factorial 57) = 13 :=
sorry

end NUMINAMATH_CALUDE_seven_consecutive_integers_product_first_57_integers_product_l1441_144136


namespace NUMINAMATH_CALUDE_range_of_g_bounds_achievable_l1441_144130

theorem range_of_g (x : ℝ) : ∃ (y : ℝ), y ∈ Set.Icc (3/4 : ℝ) 1 ∧ y = Real.cos x ^ 4 + Real.sin x ^ 2 :=
sorry

theorem bounds_achievable :
  (∃ (x : ℝ), Real.cos x ^ 4 + Real.sin x ^ 2 = 3/4) ∧
  (∃ (x : ℝ), Real.cos x ^ 4 + Real.sin x ^ 2 = 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_g_bounds_achievable_l1441_144130


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1441_144182

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + 2*I) = Complex.abs (-3 + 4*I)) : 
  Complex.im z = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1441_144182


namespace NUMINAMATH_CALUDE_triangle_problem_l1441_144144

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC is acute
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  -- a, b, c are sides opposite to angles A, B, C
  a = 4 ∧
  b = 5 ∧
  -- Area of triangle ABC is 5√3
  (1/2) * a * b * Real.sin C = 5 * Real.sqrt 3 →
  c = Real.sqrt 21 ∧
  Real.sin A = (2 * Real.sqrt 7) / 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1441_144144


namespace NUMINAMATH_CALUDE_total_interest_received_l1441_144145

/-- Calculate simple interest -/
def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

/-- Calculate total interest from two loans -/
def total_interest (principal1 principal2 rate time1 time2 : ℕ) : ℕ :=
  simple_interest principal1 rate time1 + simple_interest principal2 rate time2

/-- Theorem stating the total interest received by A -/
theorem total_interest_received : 
  total_interest 5000 3000 12 2 4 = 2440 := by
  sorry

end NUMINAMATH_CALUDE_total_interest_received_l1441_144145


namespace NUMINAMATH_CALUDE_line_slope_l1441_144193

theorem line_slope (x y : ℝ) : x - Real.sqrt 3 * y - Real.sqrt 3 = 0 → 
  (y - (-1)) / (x - 0) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l1441_144193


namespace NUMINAMATH_CALUDE_skips_per_meter_correct_l1441_144152

/-- Represents the number of skips in one meter given the following conditions:
    * x hops equals y skips
    * z jumps equals w hops
    * u jumps equals v meters
-/
def skips_per_meter (x y z w u v : ℚ) : ℚ :=
  u * y * w / (v * x * z)

/-- Theorem stating that under the given conditions, 
    1 meter equals (uyw / (vxz)) skips -/
theorem skips_per_meter_correct
  (x y z w u v : ℚ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) (hu : u > 0) (hv : v > 0)
  (hops_to_skips : x * 1 = y)
  (jumps_to_hops : z * 1 = w)
  (jumps_to_meters : u * 1 = v) :
  skips_per_meter x y z w u v = u * y * w / (v * x * z) :=
by sorry

end NUMINAMATH_CALUDE_skips_per_meter_correct_l1441_144152


namespace NUMINAMATH_CALUDE_pythagorean_triple_parity_l1441_144123

theorem pythagorean_triple_parity (x y z : ℤ) (h : x^2 + y^2 = z^2) :
  Even x ∨ Even y := by
  sorry

end NUMINAMATH_CALUDE_pythagorean_triple_parity_l1441_144123


namespace NUMINAMATH_CALUDE_parabola_properties_l1441_144127

def parabola (x : ℝ) : ℝ := (x - 1)^2 - 2

theorem parabola_properties :
  (∀ x, parabola x ≥ parabola 1) ∧
  (∀ x₁ x₂, x₁ > 1 ∧ x₂ > 1 ∧ x₂ > x₁ → parabola x₂ > parabola x₁) ∧
  (parabola 1 = -2) ∧
  (∀ x, parabola x = parabola (2 - x)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l1441_144127


namespace NUMINAMATH_CALUDE_not_both_squares_l1441_144183

theorem not_both_squares (a b : ℤ) : ¬(∃ (c d : ℤ), c > 0 ∧ d > 0 ∧ a * (a + 4) = c^2 ∧ b * (b + 4) = d^2) := by
  sorry

end NUMINAMATH_CALUDE_not_both_squares_l1441_144183


namespace NUMINAMATH_CALUDE_inequality_implies_a_range_l1441_144199

theorem inequality_implies_a_range (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 1, (x + Real.log a) / Real.exp x - a * Real.log x / x > 0) →
  a ∈ Set.Icc (Real.exp (-1)) 1 ∧ a ≠ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_implies_a_range_l1441_144199


namespace NUMINAMATH_CALUDE_f_sum_2009_2010_l1441_144185

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_sum_2009_2010 (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period (fun x ↦ f (3*x + 1)) 2)
  (h_f_1 : f 1 = 2010) :
  f 2009 + f 2010 = -2010 := by sorry

end NUMINAMATH_CALUDE_f_sum_2009_2010_l1441_144185


namespace NUMINAMATH_CALUDE_bingo_prize_calculation_l1441_144148

/-- The total prize money for the bingo night. -/
def total_prize_money : ℝ := 2400

/-- The amount received by each of the 10 winners after the first winner. -/
def winner_amount : ℝ := 160

theorem bingo_prize_calculation :
  let first_winner_share := total_prize_money / 3
  let remaining_after_first := total_prize_money - first_winner_share
  let each_winner_share := remaining_after_first / 10
  (each_winner_share = winner_amount) ∧ 
  (total_prize_money > 0) ∧
  (winner_amount > 0) :=
by sorry

end NUMINAMATH_CALUDE_bingo_prize_calculation_l1441_144148


namespace NUMINAMATH_CALUDE_smallest_m_correct_l1441_144176

/-- The smallest positive value of m for which 10x^2 - mx + 660 = 0 has integral solutions -/
def smallest_m : ℕ := 170

/-- A function representing the quadratic equation 10x^2 - mx + 660 = 0 -/
def quadratic (m : ℕ) (x : ℤ) : ℤ := 10 * x^2 - m * x + 660

theorem smallest_m_correct :
  (∃ x y : ℤ, quadratic smallest_m x = 0 ∧ quadratic smallest_m y = 0 ∧ x ≠ y) ∧
  (∀ m : ℕ, m < smallest_m → ¬∃ x y : ℤ, quadratic m x = 0 ∧ quadratic m y = 0 ∧ x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_correct_l1441_144176


namespace NUMINAMATH_CALUDE_prob_friends_same_group_l1441_144116

/-- The total number of students -/
def total_students : ℕ := 900

/-- The number of lunch groups -/
def num_groups : ℕ := 5

/-- The number of friends we're considering -/
def num_friends : ℕ := 4

/-- Represents a random assignment of students to lunch groups -/
def random_assignment : Type := Fin total_students → Fin num_groups

/-- The probability of a specific student being assigned to a specific group -/
def prob_single_assignment : ℚ := 1 / num_groups

/-- 
The probability that all friends are assigned to the same group
given a random assignment of students to groups
-/
def prob_all_friends_same_group (assignment : random_assignment) : ℚ :=
  prob_single_assignment ^ (num_friends - 1)

theorem prob_friends_same_group :
  ∀ (assignment : random_assignment),
    prob_all_friends_same_group assignment = 1 / 125 :=
by sorry

end NUMINAMATH_CALUDE_prob_friends_same_group_l1441_144116


namespace NUMINAMATH_CALUDE_root_transformation_l1441_144119

theorem root_transformation (a₁ a₂ a₃ b c₁ c₂ c₃ : ℝ) 
  (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃)
  (h_roots : ∀ x, (x - a₁) * (x - a₂) * (x - a₃) = b ↔ x = c₁ ∨ x = c₂ ∨ x = c₃)
  (h_distinct_roots : c₁ ≠ c₂ ∧ c₁ ≠ c₃ ∧ c₂ ≠ c₃) :
  ∀ x, (x + c₁) * (x + c₂) * (x + c₃) = b ↔ x = -a₁ ∨ x = -a₂ ∨ x = -a₃ :=
sorry

end NUMINAMATH_CALUDE_root_transformation_l1441_144119


namespace NUMINAMATH_CALUDE_exist_non_congruent_polyhedra_with_same_views_l1441_144186

/-- Represents a polyhedron --/
structure Polyhedron where
  vertices : Set (Fin 3 → ℝ)
  faces : Set (Set (Fin 3 → ℝ))

/-- Represents a 2D view of a polyhedron --/
structure View where
  points : Set (Fin 2 → ℝ)
  edges : Set (Fin 2 → ℝ) × (Fin 2 → ℝ)

/-- Checks if two polyhedra are congruent --/
def are_congruent (p1 p2 : Polyhedron) : Prop :=
  sorry

/-- Gets the front view of a polyhedron --/
def front_view (p : Polyhedron) : View :=
  sorry

/-- Gets the top view of a polyhedron --/
def top_view (p : Polyhedron) : View :=
  sorry

/-- Checks if a view has an internal intersection point at the center of the square --/
def has_center_intersection (v : View) : Prop :=
  sorry

/-- Checks if all segments of the squares in a view are visible edges --/
def all_segments_visible (v : View) : Prop :=
  sorry

/-- Checks if a view has no hidden edges --/
def no_hidden_edges (v : View) : Prop :=
  sorry

/-- The main theorem stating the existence of two non-congruent polyhedra with the given properties --/
theorem exist_non_congruent_polyhedra_with_same_views : 
  ∃ (p1 p2 : Polyhedron), 
    front_view p1 = front_view p2 ∧
    top_view p1 = top_view p2 ∧
    has_center_intersection (front_view p1) ∧
    has_center_intersection (top_view p1) ∧
    all_segments_visible (front_view p1) ∧
    all_segments_visible (top_view p1) ∧
    no_hidden_edges (front_view p1) ∧
    no_hidden_edges (top_view p1) ∧
    ¬(are_congruent p1 p2) :=
  sorry

end NUMINAMATH_CALUDE_exist_non_congruent_polyhedra_with_same_views_l1441_144186


namespace NUMINAMATH_CALUDE_opposites_sum_l1441_144166

theorem opposites_sum (x y : ℝ) : (x + 5)^2 + |y - 2| = 0 → x + 2*y = -1 := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_l1441_144166


namespace NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l1441_144170

theorem equation_represents_pair_of_lines :
  ∀ (x y : ℝ), x^2 - 9*y^2 = 0 → ∃ (m₁ m₂ : ℝ), (x = m₁*y ∨ x = m₂*y) ∧ m₁ ≠ m₂ := by
  sorry

end NUMINAMATH_CALUDE_equation_represents_pair_of_lines_l1441_144170


namespace NUMINAMATH_CALUDE_smallest_K_is_correct_l1441_144188

/-- The smallest positive integer K such that 8000 × K is a perfect square -/
def smallest_K : ℕ := 5

/-- A predicate that checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem smallest_K_is_correct :
  (∀ k : ℕ, k > 0 → k < smallest_K → ¬ is_perfect_square (8000 * k)) ∧
  is_perfect_square (8000 * smallest_K) := by
  sorry

#check smallest_K_is_correct

end NUMINAMATH_CALUDE_smallest_K_is_correct_l1441_144188


namespace NUMINAMATH_CALUDE_construction_delay_l1441_144132

/-- Represents the construction project with given parameters -/
structure ConstructionProject where
  totalDays : ℕ
  initialWorkers : ℕ
  additionalWorkers : ℕ
  additionalWorkersStartDay : ℕ

/-- Calculates the total work units completed in the project -/
def totalWorkUnits (project : ConstructionProject) : ℕ :=
  project.initialWorkers * project.totalDays +
  project.additionalWorkers * (project.totalDays - project.additionalWorkersStartDay)

/-- Calculates the number of days needed to complete the work with only initial workers -/
def daysNeededWithoutAdditionalWorkers (project : ConstructionProject) : ℕ :=
  (totalWorkUnits project) / project.initialWorkers

/-- Theorem: The project will be 90 days behind schedule without additional workers -/
theorem construction_delay (project : ConstructionProject) 
  (h1 : project.totalDays = 100)
  (h2 : project.initialWorkers = 100)
  (h3 : project.additionalWorkers = 100)
  (h4 : project.additionalWorkersStartDay = 10) :
  daysNeededWithoutAdditionalWorkers project - project.totalDays = 90 := by
  sorry

end NUMINAMATH_CALUDE_construction_delay_l1441_144132


namespace NUMINAMATH_CALUDE_imaginaria_city_population_l1441_144103

theorem imaginaria_city_population : ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a^2 + 225 = b^2 + 1 ∧
  b^2 + 76 = c^2 ∧
  5 ∣ a^2 := by
sorry

end NUMINAMATH_CALUDE_imaginaria_city_population_l1441_144103


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1441_144181

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define the set M
def M : Set Nat := {1, 3, 5}

-- Theorem statement
theorem complement_of_M_in_U : 
  (U \ M) = {2, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1441_144181


namespace NUMINAMATH_CALUDE_amount_ratio_l1441_144147

theorem amount_ratio (total : ℕ) (r_amount : ℕ) : 
  total = 4000 →
  r_amount = 1600 →
  (r_amount : ℚ) / ((total - r_amount) : ℚ) = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_amount_ratio_l1441_144147


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l1441_144153

theorem arithmetic_expression_equality : 50 + 5 * 12 / (180 / 3) = 51 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l1441_144153


namespace NUMINAMATH_CALUDE_election_votes_l1441_144128

theorem election_votes (V : ℕ) : 
  (60 * V / 100 - 40 * V / 100 = 1380) → V = 6900 := by sorry

end NUMINAMATH_CALUDE_election_votes_l1441_144128


namespace NUMINAMATH_CALUDE_smallest_norm_given_condition_l1441_144108

/-- Given a vector v in ℝ², prove that the smallest possible value of its norm,
    given that ‖v + (4, 2)‖ = 10, is 10 - 2√5. -/
theorem smallest_norm_given_condition (v : ℝ × ℝ) 
    (h : ‖v + (4, 2)‖ = 10) : 
    ∃ (w : ℝ × ℝ), ‖w‖ = 10 - 2 * Real.sqrt 5 ∧ 
    ∀ (u : ℝ × ℝ), ‖u + (4, 2)‖ = 10 → ‖w‖ ≤ ‖u‖ := by
  sorry

end NUMINAMATH_CALUDE_smallest_norm_given_condition_l1441_144108


namespace NUMINAMATH_CALUDE_sword_length_difference_main_result_l1441_144117

/-- Proves that Jameson's sword is 3 inches longer than twice Christopher's sword length -/
theorem sword_length_difference : ℕ → ℕ → ℕ → Prop :=
  fun christopher_length june_christopher_diff jameson_june_diff =>
    let christopher_length : ℕ := 15
    let june_christopher_diff : ℕ := 23
    let jameson_june_diff : ℕ := 5
    let june_length : ℕ := christopher_length + june_christopher_diff
    let jameson_length : ℕ := june_length - jameson_june_diff
    let twice_christopher_length : ℕ := 2 * christopher_length
    jameson_length - twice_christopher_length = 3

/-- Main theorem stating the result -/
theorem main_result : sword_length_difference 15 23 5 := by
  sorry

end NUMINAMATH_CALUDE_sword_length_difference_main_result_l1441_144117


namespace NUMINAMATH_CALUDE_eleventh_term_of_geometric_sequence_l1441_144140

def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem eleventh_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_fifth : a 5 = 5)
  (h_eighth : a 8 = 40) :
  a 11 = 320 := by
sorry

end NUMINAMATH_CALUDE_eleventh_term_of_geometric_sequence_l1441_144140


namespace NUMINAMATH_CALUDE_triangle_rotation_path_length_l1441_144177

/-- The total path length of a vertex of an equilateral triangle rotating inside a square -/
theorem triangle_rotation_path_length 
  (triangle_side : ℝ) 
  (square_side : ℝ) 
  (h1 : triangle_side = 3) 
  (h2 : square_side = 6) 
  (h3 : triangle_side > 0) 
  (h4 : square_side > triangle_side) : 
  (4 : ℝ) * (π / 2) * triangle_side = 6 * π := by
sorry

end NUMINAMATH_CALUDE_triangle_rotation_path_length_l1441_144177


namespace NUMINAMATH_CALUDE_mrs_a_speed_l1441_144135

/-- Proves that Mrs. A's speed is 10 kmph given the problem conditions --/
theorem mrs_a_speed (initial_distance : ℝ) (mr_a_speed : ℝ) (bee_speed : ℝ) (bee_distance : ℝ)
  (h1 : initial_distance = 120)
  (h2 : mr_a_speed = 30)
  (h3 : bee_speed = 60)
  (h4 : bee_distance = 180) :
  let time := bee_distance / bee_speed
  let mr_a_distance := mr_a_speed * time
  let mrs_a_distance := initial_distance - mr_a_distance
  mrs_a_distance / time = 10 := by
  sorry

#check mrs_a_speed

end NUMINAMATH_CALUDE_mrs_a_speed_l1441_144135


namespace NUMINAMATH_CALUDE_perimeter_is_twelve_l1441_144191

/-- An equilateral triangle -/
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  base_positive : base > 0
  leg_positive : leg > 0

/-- A quadrilateral formed by cutting a corner from an equilateral triangle -/
def CutCornerQuadrilateral (et : EquilateralTriangle) (it : IsoscelesTriangle) :=
  it.leg < et.side ∧ it.base < et.side

/-- The perimeter of the quadrilateral formed by cutting a corner from an equilateral triangle -/
def perimeter (et : EquilateralTriangle) (it : IsoscelesTriangle) 
    (h : CutCornerQuadrilateral et it) : ℝ :=
  et.side + 2 * (et.side - it.leg) + it.base

/-- The main theorem -/
theorem perimeter_is_twelve 
    (et : EquilateralTriangle)
    (it : IsoscelesTriangle)
    (h : CutCornerQuadrilateral et it)
    (h_et_side : et.side = 4)
    (h_it_leg : it.leg = 0.5)
    (h_it_base : it.base = 1) :
    perimeter et it h = 12 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_is_twelve_l1441_144191


namespace NUMINAMATH_CALUDE_math_scores_properties_l1441_144156

def scores : List ℝ := [60, 60, 60, 65, 65, 70, 70, 70, 75, 75, 75, 75, 75, 80, 80, 80, 80, 85, 85, 90]

def group_a : List ℝ := [60, 60, 60, 65, 65, 70, 70, 70]

def mode (l : List ℝ) : ℝ := sorry

def variance (l : List ℝ) : ℝ := sorry

theorem math_scores_properties :
  (mode scores = 75) ∧ (variance group_a = 75/4) := by sorry

end NUMINAMATH_CALUDE_math_scores_properties_l1441_144156


namespace NUMINAMATH_CALUDE_derivative_at_one_l1441_144129

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_at_one :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 2| < ε :=
sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1441_144129


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l1441_144175

theorem absolute_value_inequality (x y : ℝ) 
  (h1 : |x - y| < 1) 
  (h2 : |2*x + y| < 1) : 
  |y| < 1 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l1441_144175


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l1441_144141

theorem simplify_and_evaluate : 
  let x : ℝ := -3
  3 * (2 * x^2 - 5 * x) - 2 * (-3 * x - 2 + 3 * x^2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l1441_144141


namespace NUMINAMATH_CALUDE_gavin_shirts_count_l1441_144192

theorem gavin_shirts_count (blue_shirts green_shirts : ℕ) :
  blue_shirts = 6 →
  green_shirts = 17 →
  blue_shirts + green_shirts = 23 :=
by sorry

end NUMINAMATH_CALUDE_gavin_shirts_count_l1441_144192


namespace NUMINAMATH_CALUDE_band_gigs_theorem_l1441_144133

/-- Represents a band with its members and earnings -/
structure Band where
  members : ℕ
  earnings_per_member_per_gig : ℕ
  total_earnings : ℕ

/-- Calculates the number of gigs played by a band -/
def gigs_played (b : Band) : ℕ :=
  b.total_earnings / (b.members * b.earnings_per_member_per_gig)

/-- Theorem stating that for a band with 4 members, $20 earnings per member per gig,
    and $400 total earnings, the number of gigs played is 5 -/
theorem band_gigs_theorem (b : Band) 
    (h1 : b.members = 4)
    (h2 : b.earnings_per_member_per_gig = 20)
    (h3 : b.total_earnings = 400) :
    gigs_played b = 5 := by
  sorry

end NUMINAMATH_CALUDE_band_gigs_theorem_l1441_144133


namespace NUMINAMATH_CALUDE_oak_elm_difference_pine_elm_difference_l1441_144137

-- Define the heights of the trees
def elm_height : ℚ := 49/4  -- 12¼ feet
def oak_height : ℚ := 37/2  -- 18½ feet
def pine_height_inches : ℚ := 225  -- 225 inches

-- Convert pine height to feet
def pine_height : ℚ := pine_height_inches / 12

-- Define the theorems to be proved
theorem oak_elm_difference : oak_height - elm_height = 25/4 := by sorry

theorem pine_elm_difference : pine_height - elm_height = 13/2 := by sorry

end NUMINAMATH_CALUDE_oak_elm_difference_pine_elm_difference_l1441_144137


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l1441_144149

/-- The range of m for which the line y = kx + 1 and the ellipse x²/5 + y²/m = 1 always intersect -/
theorem line_ellipse_intersection_range (k : ℝ) (m : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1 → m ≥ 1 ∧ m ≠ 5) ∧
  (m ≥ 1 ∧ m ≠ 5 → ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) :=
sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l1441_144149


namespace NUMINAMATH_CALUDE_existence_of_xy_l1441_144162

theorem existence_of_xy (n : ℕ) (k : ℕ) (h : n = 4 * k + 1) : 
  ∃ (x y : ℤ), (x^n + y^n) ∈ {z : ℤ | ∃ (a b : ℤ), z = a^2 + n * b^2} ∧ 
  (x + y) ∉ {z : ℤ | ∃ (a b : ℤ), z = a^2 + n * b^2} := by
  sorry

end NUMINAMATH_CALUDE_existence_of_xy_l1441_144162


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1441_144126

open Real

/-- The sum of the infinite series ∑(n=1 to ∞) (n + 1) / (n + 2)! is equal to e - 3 -/
theorem infinite_series_sum : 
  (∑' n : ℕ, (n + 1 : ℝ) / (n + 2).factorial) = Real.exp 1 - 3 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1441_144126


namespace NUMINAMATH_CALUDE_decimal_17_to_binary_l1441_144124

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec go (m : ℕ) (acc : List Bool) : List Bool :=
      if m = 0 then acc
      else go (m / 2) ((m % 2 = 1) :: acc)
    go n []

/-- Converts a list of bits to a string representation of a binary number -/
def binaryToString (bits : List Bool) : String :=
  bits.map (fun b => if b then '1' else '0') |> String.mk

theorem decimal_17_to_binary :
  binaryToString (toBinary 17) = "10001" := by
  sorry

end NUMINAMATH_CALUDE_decimal_17_to_binary_l1441_144124


namespace NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l1441_144113

theorem min_value_x_plus_four_over_x :
  ∀ x : ℝ, x > 0 → x + 4 / x ≥ 4 ∧ ∃ y : ℝ, y > 0 ∧ y + 4 / y = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_four_over_x_l1441_144113


namespace NUMINAMATH_CALUDE_train_passing_time_l1441_144115

/-- The time taken for a train to pass a man moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) :
  train_length = 120 →
  train_speed = 50 * (1000 / 3600) →
  man_speed = 4 * (1000 / 3600) →
  (train_length / (train_speed + man_speed)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1441_144115


namespace NUMINAMATH_CALUDE_triangle_right_angle_l1441_144163

/-- If in a triangle ABC, sin(A+B) = sin(A-B), then the triangle ABC is a right triangle. -/
theorem triangle_right_angle (A B C : ℝ) (h_triangle : A + B + C = π) 
  (h_sin_eq : Real.sin (A + B) = Real.sin (A - B)) : 
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_right_angle_l1441_144163


namespace NUMINAMATH_CALUDE_sphere_triangle_distance_l1441_144196

-- Define the sphere
def sphere_radius : ℝ := 9

-- Define the triangle
def triangle_side1 : ℝ := 20
def triangle_side2 : ℝ := 20
def triangle_side3 : ℝ := 30

-- State the theorem
theorem sphere_triangle_distance :
  let s := (triangle_side1 + triangle_side2 + triangle_side3) / 2
  let area := Real.sqrt (s * (s - triangle_side1) * (s - triangle_side2) * (s - triangle_side3))
  let inradius := area / s
  Real.sqrt (sphere_radius ^ 2 - inradius ^ 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sphere_triangle_distance_l1441_144196


namespace NUMINAMATH_CALUDE_no_real_solution_l1441_144184

theorem no_real_solution :
  ¬∃ (r s : ℝ), (r - 50) / 3 = (s - 2 * r) / 4 ∧ r^2 + 3 * s = 50 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solution_l1441_144184


namespace NUMINAMATH_CALUDE_min_value_theorem_l1441_144190

theorem min_value_theorem (x y : ℝ) 
  (h : Real.exp x + x - 2023 = Real.exp 2023 / (y + 2023) - Real.log (y + 2023)) :
  (∀ x' y' : ℝ, Real.exp x' + x' - 2023 = Real.exp 2023 / (y' + 2023) - Real.log (y' + 2023) → 
    Real.exp x' + y' + 2024 ≥ Real.exp x + y + 2024) →
  Real.exp x + y + 2024 = 2 * Real.sqrt (Real.exp 2023) + 1 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1441_144190


namespace NUMINAMATH_CALUDE_log_graph_passes_through_point_l1441_144169

-- Define the logarithm function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Define our function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log a (x - 2) + 3

-- State the theorem
theorem log_graph_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 3 = 3 := by sorry

end NUMINAMATH_CALUDE_log_graph_passes_through_point_l1441_144169


namespace NUMINAMATH_CALUDE_infinite_solutions_abs_value_equation_l1441_144161

theorem infinite_solutions_abs_value_equation (a : ℝ) :
  (∀ x : ℝ, |x - 2| = a * x - 2) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_abs_value_equation_l1441_144161


namespace NUMINAMATH_CALUDE_cat_food_finished_on_sunday_l1441_144151

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the number of cans consumed up to and including a given day -/
def cans_consumed (d : Day) : ℚ :=
  match d with
  | Day.Monday => 3/4
  | Day.Tuesday => 3/2
  | Day.Wednesday => 9/4
  | Day.Thursday => 3
  | Day.Friday => 15/4
  | Day.Saturday => 9/2
  | Day.Sunday => 21/4

/-- The amount of cat food Roy starts with -/
def initial_cans : ℚ := 8

theorem cat_food_finished_on_sunday :
  ∀ d : Day, cans_consumed d ≤ initial_cans ∧
  (d = Day.Sunday → cans_consumed d > initial_cans - 3/4) :=
by sorry

end NUMINAMATH_CALUDE_cat_food_finished_on_sunday_l1441_144151


namespace NUMINAMATH_CALUDE_library_visitors_l1441_144104

theorem library_visitors (visitors_non_sunday : ℕ) (avg_visitors_per_day : ℕ) :
  visitors_non_sunday = 140 →
  avg_visitors_per_day = 200 →
  ∃ (visitors_sunday : ℕ),
    5 * visitors_sunday + 25 * visitors_non_sunday = 30 * avg_visitors_per_day ∧
    visitors_sunday = 500 := by
  sorry

end NUMINAMATH_CALUDE_library_visitors_l1441_144104


namespace NUMINAMATH_CALUDE_exponent_fraction_simplification_l1441_144173

theorem exponent_fraction_simplification :
  (3^8 + 3^6) / (3^8 - 3^6) = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_fraction_simplification_l1441_144173


namespace NUMINAMATH_CALUDE_special_triangle_area_l1441_144105

/-- Triangle with specific properties -/
structure SpecialTriangle where
  -- Three sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- One angle is 120°
  angle_120 : ∃ θ, θ = 2 * π / 3
  -- Sides form arithmetic sequence with difference 4
  arithmetic_seq : ∃ x : ℝ, a = x - 4 ∧ b = x ∧ c = x + 4

/-- The area of the special triangle is 15√3 -/
theorem special_triangle_area (t : SpecialTriangle) : 
  (1/2) * t.a * t.b * Real.sqrt 3 = 15 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_area_l1441_144105


namespace NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l1441_144159

theorem triangle_sine_sum_inequality (A B C : Real) : 
  A + B + C = π → 0 < A → 0 < B → 0 < C → 
  Real.sin A + Real.sin B + Real.sin C ≤ (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l1441_144159


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1441_144114

theorem expression_simplification_and_evaluation :
  let x : ℚ := 3
  let expr := (1 / (x - 1) + 1) / ((x^2 - 1) / (x^2 - 2*x + 1))
  expr = 3/4 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1441_144114


namespace NUMINAMATH_CALUDE_fraction_of_72_l1441_144106

theorem fraction_of_72 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 72 = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_72_l1441_144106


namespace NUMINAMATH_CALUDE_square_sum_identity_l1441_144107

theorem square_sum_identity (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l1441_144107


namespace NUMINAMATH_CALUDE_chemistry_books_count_l1441_144110

/-- The number of ways to choose 2 items from n items -/
def choose_2 (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The problem statement -/
theorem chemistry_books_count :
  ∃ (C : ℕ), C > 0 ∧ choose_2 15 * choose_2 C = 2940 ∧ C = 8 := by sorry

end NUMINAMATH_CALUDE_chemistry_books_count_l1441_144110


namespace NUMINAMATH_CALUDE_min_value_a_l1441_144171

theorem min_value_a : ∃ (a : ℝ),
  (∃ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 1 ∧ 1 + 2^x + a * 4^x ≥ 0) ∧
  (∀ (b : ℝ), (∃ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 1 ∧ 1 + 2^x + b * 4^x ≥ 0) → b ≥ a) ∧
  a = -6 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l1441_144171


namespace NUMINAMATH_CALUDE_female_officers_count_l1441_144134

theorem female_officers_count (total_on_duty : ℕ) (male_on_duty : ℕ) 
  (female_on_duty_percentage : ℚ) :
  total_on_duty = 475 →
  male_on_duty = 315 →
  female_on_duty_percentage = 65/100 →
  ∃ (total_female : ℕ), 
    (total_female : ℚ) * female_on_duty_percentage = (total_on_duty - male_on_duty : ℚ) ∧
    total_female = 246 :=
by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l1441_144134


namespace NUMINAMATH_CALUDE_mean_age_is_eleven_l1441_144111

/-- Represents the ages of children in the Euler family -/
def euler_ages : List ℕ := [10, 12, 8]

/-- Represents the ages of children in the Gauss family -/
def gauss_ages : List ℕ := [8, 8, 8, 16, 18]

/-- Calculates the mean age of all children from both families -/
def mean_age : ℚ := (euler_ages.sum + gauss_ages.sum) / (euler_ages.length + gauss_ages.length)

theorem mean_age_is_eleven : mean_age = 11 := by
  sorry

end NUMINAMATH_CALUDE_mean_age_is_eleven_l1441_144111


namespace NUMINAMATH_CALUDE_hall_covering_cost_l1441_144102

def hall_length : ℝ := 20
def hall_width : ℝ := 15
def hall_height : ℝ := 5
def cost_per_square_meter : ℝ := 50

def total_area : ℝ := 2 * (hall_length * hall_width) + 2 * (hall_length * hall_height) + 2 * (hall_width * hall_height)

def total_expenditure : ℝ := total_area * cost_per_square_meter

theorem hall_covering_cost : total_expenditure = 47500 := by
  sorry

end NUMINAMATH_CALUDE_hall_covering_cost_l1441_144102


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1441_144154

/-- Given a parabola and a hyperbola with coinciding foci, 
    prove that the eccentricity of the hyperbola is 2√3/3 -/
theorem hyperbola_eccentricity 
  (parabola : ℝ → ℝ) 
  (hyperbola : ℝ → ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x y, parabola y = (1/8) * x^2)
  (h2 : ∀ x y, hyperbola x y = y^2/a - x^2 - 1)
  (h3 : ∃ x y, parabola y = (1/8) * x^2 ∧ hyperbola x y = 0 ∧ 
              x^2 + (y - a/2)^2 = (a/2)^2) : 
  ∃ e : ℝ, e = 2 * Real.sqrt 3 / 3 ∧ 
    ∀ x y, hyperbola x y = 0 → x^2/(a/e^2) + y^2/a = 1 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1441_144154


namespace NUMINAMATH_CALUDE_benny_seashells_l1441_144158

/-- Proves that the initial number of seashells Benny found is equal to the number of seashells he has now plus the number of seashells he gave away. -/
theorem benny_seashells (seashells_now : ℕ) (seashells_given : ℕ) 
  (h1 : seashells_now = 14) 
  (h2 : seashells_given = 52) : 
  seashells_now + seashells_given = 66 := by
  sorry

#check benny_seashells

end NUMINAMATH_CALUDE_benny_seashells_l1441_144158


namespace NUMINAMATH_CALUDE_distance_between_points_l1441_144142

/-- The distance between points (3,4) and (8,17) is √194 -/
theorem distance_between_points : Real.sqrt ((8 - 3)^2 + (17 - 4)^2) = Real.sqrt 194 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l1441_144142


namespace NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_nine_squared_l1441_144168

theorem sqrt_square_abs (x : ℝ) : Real.sqrt (x ^ 2) = |x| := by sorry

theorem sqrt_neg_nine_squared : Real.sqrt ((-9) ^ 2) = 9 := by sorry

end NUMINAMATH_CALUDE_sqrt_square_abs_sqrt_neg_nine_squared_l1441_144168


namespace NUMINAMATH_CALUDE_fraction_sum_l1441_144179

theorem fraction_sum (m n : ℕ) (hcoprime : Nat.Coprime m n) 
  (heq : (2013 * 2013) / (2014 * 2014 + 2012) = n / m) : m + n = 1343 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1441_144179


namespace NUMINAMATH_CALUDE_seating_arrangement_count_l1441_144146

/-- The number of seats at the bus station -/
def total_seats : ℕ := 10

/-- The number of passengers -/
def num_passengers : ℕ := 4

/-- The number of consecutive empty seats required -/
def consecutive_empty_seats : ℕ := 5

/-- Calculate the number of ways to arrange seating -/
def seating_arrangements (total : ℕ) (passengers : ℕ) (empty_block : ℕ) : ℕ :=
  (Nat.factorial passengers) * (Nat.factorial (total - passengers - empty_block + 1) / Nat.factorial (total - passengers - empty_block - 1))

theorem seating_arrangement_count : 
  seating_arrangements total_seats num_passengers consecutive_empty_seats = 480 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangement_count_l1441_144146


namespace NUMINAMATH_CALUDE_original_line_equation_l1441_144125

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Shift a line horizontally -/
def shift_line (l : Line) (shift : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - l.slope * shift }

theorem original_line_equation (l : Line) :
  (shift_line l 2).slope = 2 ∧ (shift_line l 2).intercept = 3 →
  l.slope = 2 ∧ l.intercept = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_line_equation_l1441_144125


namespace NUMINAMATH_CALUDE_quadratic_properties_l1441_144197

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_neg : a < 0
  root_neg_one : a * (-1)^2 + b * (-1) + c = 0
  symmetry_axis : -b / (2 * a) = 1

/-- Properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (f.a - f.b + f.c = 0) ∧
  (∀ m : ℝ, f.a * m^2 + f.b * m + f.c ≤ -4 * f.a) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → 
    f.a * x₁^2 + f.b * x₁ + f.c + 1 = 0 → 
    f.a * x₂^2 + f.b * x₂ + f.c + 1 = 0 → 
    x₁ < -1 ∧ x₂ > 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1441_144197


namespace NUMINAMATH_CALUDE_exponent_calculation_l1441_144100

theorem exponent_calculation (m : ℕ) : m = 8^126 → (m * 16) / 64 = 16^94 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l1441_144100


namespace NUMINAMATH_CALUDE_article_profit_percentage_l1441_144187

theorem article_profit_percentage (cost selling_price : ℚ) : 
  cost = 70 →
  (0.8 * cost) * 1.3 = selling_price - 14.70 →
  (selling_price - cost) / cost * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_article_profit_percentage_l1441_144187


namespace NUMINAMATH_CALUDE_bills_age_l1441_144167

theorem bills_age (caroline_age bill_age : ℕ) : 
  bill_age = 2 * caroline_age - 1 →
  bill_age + caroline_age = 26 →
  bill_age = 17 := by
sorry

end NUMINAMATH_CALUDE_bills_age_l1441_144167


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1441_144160

theorem geometric_sequence_ratio (a₁ a₂ a₃ : ℝ) (h1 : a₁ = 9) (h2 : a₂ = -18) (h3 : a₃ = 36) :
  ∃ r : ℝ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1441_144160


namespace NUMINAMATH_CALUDE_smallest_max_sum_l1441_144150

theorem smallest_max_sum (a b c d e : ℕ+) (h_sum : a + b + c + d + e = 2500) :
  ∃ N : ℕ, N = max (a + b) (max (b + c) (max (c + d) (d + e))) ∧
  (∀ M : ℕ, M = max (a + b) (max (b + c) (max (c + d) (d + e))) → N ≤ M) ∧
  N = 834 :=
sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l1441_144150


namespace NUMINAMATH_CALUDE_minimum_cactus_species_l1441_144155

theorem minimum_cactus_species (n : ℕ) (h1 : n = 80) : ∃ (k : ℕ),
  (∀ (S : Finset (Finset ℕ)), S.card = n → 
    (∀ i : ℕ, i ≤ k → ∃ s ∈ S, i ∉ s) ∧ 
    (∀ T : Finset (Finset ℕ), T ⊆ S → T.card = 15 → ∃ i : ℕ, i ≤ k ∧ ∀ s ∈ T, i ∈ s)) ∧
  k = 16 ∧ 
  (∀ m : ℕ, m < k → ¬∃ (S : Finset (Finset ℕ)), S.card = n ∧ 
    (∀ i : ℕ, i ≤ m → ∃ s ∈ S, i ∉ s) ∧
    (∀ T : Finset (Finset ℕ), T ⊆ S → T.card = 15 → ∃ i : ℕ, i ≤ m ∧ ∀ s ∈ T, i ∈ s)) :=
by sorry


end NUMINAMATH_CALUDE_minimum_cactus_species_l1441_144155


namespace NUMINAMATH_CALUDE_dice_roll_probability_l1441_144174

def standard_die := Finset.range 6

def valid_roll (a b c : ℕ) : Prop :=
  (a - 1) * (b - 1) * (c - 1) * (6 - a) * (6 - b) * (6 - c) ≠ 0

def total_outcomes : ℕ := standard_die.card ^ 3

def successful_outcomes : ℕ := ({2, 3, 4, 5} : Finset ℕ).card ^ 3

theorem dice_roll_probability :
  (successful_outcomes : ℚ) / total_outcomes = 8 / 27 := by
  sorry

end NUMINAMATH_CALUDE_dice_roll_probability_l1441_144174


namespace NUMINAMATH_CALUDE_max_good_sequences_theorem_l1441_144178

/-- A necklace with blue, red, and green beads. -/
structure Necklace where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- A sequence of four consecutive beads in the necklace. -/
structure Sequence where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- A "good" sequence contains exactly 2 blue beads, 1 red bead, and 1 green bead. -/
def is_good (s : Sequence) : Prop :=
  s.blue = 2 ∧ s.red = 1 ∧ s.green = 1

/-- The maximum number of good sequences in a necklace. -/
def max_good_sequences (n : Necklace) : ℕ := sorry

/-- Theorem: The maximum number of good sequences in a necklace with 50 blue, 100 red, and 100 green beads is 99. -/
theorem max_good_sequences_theorem (n : Necklace) (h : n.blue = 50 ∧ n.red = 100 ∧ n.green = 100) :
  max_good_sequences n = 99 := by sorry

end NUMINAMATH_CALUDE_max_good_sequences_theorem_l1441_144178


namespace NUMINAMATH_CALUDE_jhons_total_pay_l1441_144122

/-- Calculates the total pay for a worker given their work schedule and pay rates. -/
def calculate_total_pay (total_days : ℕ) (present_days : ℕ) (present_rate : ℚ) (absent_rate : ℚ) : ℚ :=
  let absent_days := total_days - present_days
  present_days * present_rate + absent_days * absent_rate

/-- Proves that Jhon's total pay is $320.00 given the specified conditions. -/
theorem jhons_total_pay :
  calculate_total_pay 60 35 7 3 = 320 := by
  sorry

end NUMINAMATH_CALUDE_jhons_total_pay_l1441_144122


namespace NUMINAMATH_CALUDE_gcd_3270_594_l1441_144194

theorem gcd_3270_594 : Nat.gcd 3270 594 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_3270_594_l1441_144194


namespace NUMINAMATH_CALUDE_positive_sum_l1441_144195

theorem positive_sum (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + c := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_l1441_144195


namespace NUMINAMATH_CALUDE_negation_equivalence_l1441_144143

theorem negation_equivalence : 
  (¬ ∀ x : ℝ, x^2 - 2*x + 3 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 3 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1441_144143


namespace NUMINAMATH_CALUDE_sum_abc_equals_two_l1441_144165

theorem sum_abc_equals_two (a b c : ℝ) 
  (h : (a - 1)^2 + |b + 1| + Real.sqrt (b + c - a) = 0) : 
  a + b + c = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_abc_equals_two_l1441_144165


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l1441_144131

/-- The function f(x) = x³ - 3ax² + 3bx -/
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + 3*b*x

/-- The derivative of f(x) -/
def f_derivative (a b x : ℝ) : ℝ := 3*x^2 - 6*a*x + 3*b

theorem tangent_line_and_extrema :
  ∃ (a b : ℝ),
    /- f(x) is tangent to 12x + y - 1 = 0 at (1, -11) -/
    (f a b 1 = -11 ∧ f_derivative a b 1 = -12) ∧
    /- a = 1 and b = -3 -/
    (a = 1 ∧ b = -3) ∧
    /- Maximum value of f(x) in [-2, 4] is 5 -/
    (∀ x, x ∈ Set.Icc (-2) 4 → f a b x ≤ 5) ∧
    (∃ x, x ∈ Set.Icc (-2) 4 ∧ f a b x = 5) ∧
    /- Minimum value of f(x) in [-2, 4] is -27 -/
    (∀ x, x ∈ Set.Icc (-2) 4 → f a b x ≥ -27) ∧
    (∃ x, x ∈ Set.Icc (-2) 4 ∧ f a b x = -27) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l1441_144131


namespace NUMINAMATH_CALUDE_parabola_and_line_properties_l1441_144164

-- Define the parabola C: y² = 4x
def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus F(1, 0)
def F : ℝ × ℝ := (1, 0)

-- Define point K(-1, 0)
def K : ℝ × ℝ := (-1, 0)

-- Define the line l passing through K and intersecting C at A and B
def l (m : ℝ) (y : ℝ) : ℝ := m*y - 1

-- Define the symmetry of A and D with respect to the x-axis
def symmetric_x_axis (A D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ A.2 = -D.2

-- Define the dot product condition
def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  (A.1 - F.1) * (B.1 - F.1) + (A.2 - F.2) * (B.2 - F.2) = 8/9

-- Main theorem
theorem parabola_and_line_properties
  (A B D : ℝ × ℝ)
  (m : ℝ)
  (h1 : C A.1 A.2)
  (h2 : C B.1 B.2)
  (h3 : A.1 = l m A.2)
  (h4 : B.1 = l m B.2)
  (h5 : symmetric_x_axis A D)
  (h6 : dot_product_condition A B) :
  (∃ (t : ℝ), F = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2)) ∧
  (∃ (a r : ℝ), a = 1/9 ∧ r = 2/3 ∧ ∀ (x y : ℝ), (x - a)^2 + y^2 = r^2 ↔ 
    (x - K.1)^2 + y^2 ≤ r^2 ∧ (x - B.1)^2 + (y - B.2)^2 ≤ r^2 ∧ (x - D.1)^2 + (y - D.2)^2 ≤ r^2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_and_line_properties_l1441_144164


namespace NUMINAMATH_CALUDE_collision_count_theorem_l1441_144157

/-- Represents the physical properties and conditions of the ball collision problem -/
structure BallCollisionProblem where
  tubeLength : ℝ
  numBalls : ℕ
  ballVelocity : ℝ
  timePeriod : ℝ

/-- Calculates the number of collisions for a given BallCollisionProblem -/
def calculateCollisions (problem : BallCollisionProblem) : ℕ :=
  sorry

/-- Theorem stating that the number of collisions for the given problem is 505000 -/
theorem collision_count_theorem (problem : BallCollisionProblem) 
  (h1 : problem.tubeLength = 1)
  (h2 : problem.numBalls = 100)
  (h3 : problem.ballVelocity = 10)
  (h4 : problem.timePeriod = 10) :
  calculateCollisions problem = 505000 := by
  sorry

end NUMINAMATH_CALUDE_collision_count_theorem_l1441_144157


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l1441_144120

theorem jacket_price_reduction (initial_reduction : ℝ) (final_increase : ℝ) (special_reduction : ℝ) : 
  initial_reduction = 25 →
  final_increase = 48.148148148148145 →
  (1 - initial_reduction / 100) * (1 - special_reduction / 100) * (1 + final_increase / 100) = 1 →
  special_reduction = 10 := by
sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l1441_144120


namespace NUMINAMATH_CALUDE_square_minus_self_sum_l1441_144139

theorem square_minus_self_sum : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_self_sum_l1441_144139


namespace NUMINAMATH_CALUDE_chicken_farm_growth_l1441_144189

theorem chicken_farm_growth (initial_chickens : ℕ) (annual_increase : ℕ) (years : ℕ) 
  (h1 : initial_chickens = 550)
  (h2 : annual_increase = 150)
  (h3 : years = 9) :
  initial_chickens + years * annual_increase = 1900 :=
by sorry

end NUMINAMATH_CALUDE_chicken_farm_growth_l1441_144189


namespace NUMINAMATH_CALUDE_inequality_problem_l1441_144180

theorem inequality_problem (s r p q : ℝ) 
  (hs : s > 0) 
  (hr : r > 0) 
  (hpq : p * q ≠ 0) 
  (hineq : s * (p * r) > s * (q * r)) : 
  ¬(-p > -q) ∧ ¬(-p > q) ∧ ¬(1 > -q/p) ∧ ¬(1 < q/p) := by
  sorry

end NUMINAMATH_CALUDE_inequality_problem_l1441_144180


namespace NUMINAMATH_CALUDE_positive_numbers_l1441_144198

theorem positive_numbers (a b c : ℝ) 
  (sum_pos : a + b + c > 0)
  (sum_prod_pos : a * b + b * c + c * a > 0)
  (prod_pos : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l1441_144198


namespace NUMINAMATH_CALUDE_steve_writes_24_pages_l1441_144101

/-- Calculates the number of pages Steve writes in a month -/
def stevePages : ℕ :=
  let daysInMonth : ℕ := 30
  let letterFrequency : ℕ := 3
  let regularLetterTime : ℕ := 20
  let timePerPage : ℕ := 10
  let longLetterTimeMultiplier : ℕ := 2
  let longLetterTotalTime : ℕ := 80

  let regularLettersCount : ℕ := daysInMonth / letterFrequency
  let regularLettersTotalTime : ℕ := regularLettersCount * regularLetterTime
  let regularLettersPages : ℕ := regularLettersTotalTime / timePerPage

  let longLetterTimePerPage : ℕ := timePerPage * longLetterTimeMultiplier
  let longLetterPages : ℕ := longLetterTotalTime / longLetterTimePerPage

  regularLettersPages + longLetterPages

theorem steve_writes_24_pages : stevePages = 24 := by
  sorry

end NUMINAMATH_CALUDE_steve_writes_24_pages_l1441_144101


namespace NUMINAMATH_CALUDE_odd_function_product_negative_l1441_144112

theorem odd_function_product_negative (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_nonzero : ∀ x, f x ≠ 0) :
  ∀ x, f x * f (-x) < 0 := by
sorry

end NUMINAMATH_CALUDE_odd_function_product_negative_l1441_144112


namespace NUMINAMATH_CALUDE_quadratic_roots_in_sixth_degree_l1441_144138

theorem quadratic_roots_in_sixth_degree (p q : ℝ) : 
  (∀ x : ℝ, x^2 - x - 1 = 0 → x^6 - p*x^2 + q = 0) → 
  p = 8 ∧ q = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_in_sixth_degree_l1441_144138
