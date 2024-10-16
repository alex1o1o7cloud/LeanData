import Mathlib

namespace NUMINAMATH_CALUDE_vector_subtraction_l438_43841

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (2, 1) → b = (-3, 4) → a - b = (5, -3) := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_l438_43841


namespace NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l438_43849

theorem smallest_value_w_cube_plus_z_cube (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 1)
  (h2 : Complex.abs (w^2 + z^2) = 14) :
  ∃ (min_val : ℝ), min_val = 41/2 ∧ 
    ∀ (w' z' : ℂ), Complex.abs (w' + z') = 1 → Complex.abs (w'^2 + z'^2) = 14 → 
      Complex.abs (w'^3 + z'^3) ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_smallest_value_w_cube_plus_z_cube_l438_43849


namespace NUMINAMATH_CALUDE_solution_set_x_squared_gt_x_l438_43867

theorem solution_set_x_squared_gt_x : 
  {x : ℝ | x^2 > x} = {x : ℝ | x < 0 ∨ x > 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_gt_x_l438_43867


namespace NUMINAMATH_CALUDE_owl_money_problem_l438_43825

theorem owl_money_problem (x : ℚ) : 
  (((3 * ((3 * ((3 * ((3 * x) - 50)) - 50)) - 50)) - 50) = 0) → 
  (x = 2000 / 81) := by
sorry

end NUMINAMATH_CALUDE_owl_money_problem_l438_43825


namespace NUMINAMATH_CALUDE_percentage_of_seats_filled_l438_43855

theorem percentage_of_seats_filled (total_seats vacant_seats : ℕ) 
  (h1 : total_seats = 600) 
  (h2 : vacant_seats = 150) : 
  (((total_seats - vacant_seats : ℚ) / total_seats) * 100 : ℚ) = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_seats_filled_l438_43855


namespace NUMINAMATH_CALUDE_fathers_age_l438_43861

theorem fathers_age (sebastian_age : ℕ) (sister_age : ℕ) (father_age : ℕ) :
  sebastian_age = 40 →
  sebastian_age = sister_age + 10 →
  (sebastian_age - 5 + sister_age - 5 : ℚ) = 3/4 * (father_age - 5) →
  father_age = 85 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_l438_43861


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l438_43873

theorem triangle_angle_measure (X Y Z : ℝ) (h1 : X + Y = 90) (h2 : Y = 2 * X) : Z = 90 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l438_43873


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l438_43826

/-- If p and q are nonzero real numbers and (3 - 4i)(p + qi) is pure imaginary, then p/q = -4/3 -/
theorem pure_imaginary_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = y * Complex.I) : 
  p / q = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l438_43826


namespace NUMINAMATH_CALUDE_lawrence_county_houses_l438_43820

/-- The number of houses in Lawrence County before the housing boom -/
def houses_before : ℕ := 1426

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 574

/-- The total number of houses in Lawrence County after the housing boom -/
def total_houses : ℕ := houses_before + houses_built

theorem lawrence_county_houses : total_houses = 2000 := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_houses_l438_43820


namespace NUMINAMATH_CALUDE_min_sum_a_b_l438_43889

theorem min_sum_a_b (a b : ℕ+) (h : (20 : ℚ) / 19 = 1 + 1 / (1 + a / b)) :
  ∃ (a' b' : ℕ+), (20 : ℚ) / 19 = 1 + 1 / (1 + a' / b') ∧ a' + b' = 19 ∧ 
  ∀ (c d : ℕ+), (20 : ℚ) / 19 = 1 + 1 / (1 + c / d) → a' + b' ≤ c + d :=
sorry

end NUMINAMATH_CALUDE_min_sum_a_b_l438_43889


namespace NUMINAMATH_CALUDE_total_balls_in_box_l438_43853

/-- Given a box with blue and red balls, calculate the total number of balls -/
theorem total_balls_in_box (blue_balls : ℕ) (red_balls : ℕ) : 
  blue_balls = 3 → red_balls = 2 → blue_balls + red_balls = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_balls_in_box_l438_43853


namespace NUMINAMATH_CALUDE_milk_container_problem_l438_43813

/-- The initial quantity of milk in container A --/
def initial_quantity : ℝ := 1232

/-- The fraction of container A's capacity that goes into container B --/
def fraction_in_B : ℝ := 0.375

/-- The amount transferred from C to B to equalize the quantities --/
def transfer_amount : ℝ := 154

theorem milk_container_problem :
  -- Container A is filled to its brim
  -- All milk from A is poured into B and C
  -- Quantity in B is 62.5% less than A (which means it's 37.5% of A)
  -- If 154L is transferred from C to B, they become equal
  (fraction_in_B * initial_quantity + transfer_amount = 
   (1 - fraction_in_B) * initial_quantity - transfer_amount) →
  -- Then the initial quantity in A is 1232 liters
  initial_quantity = 1232 := by
  sorry

end NUMINAMATH_CALUDE_milk_container_problem_l438_43813


namespace NUMINAMATH_CALUDE_expression_value_l438_43807

theorem expression_value (a b c d x : ℝ) : 
  (a = b) →  -- a and -b are opposite numbers
  (c * d = -1) →  -- c and -d are reciprocals
  (abs x = 3) →  -- absolute value of x is 3
  (x^3 + c * d * x^2 - (a - b) / 2 = 18 ∨ x^3 + c * d * x^2 - (a - b) / 2 = -36) :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l438_43807


namespace NUMINAMATH_CALUDE_aluminum_atomic_weight_l438_43824

/-- The atomic weight of chlorine in atomic mass units (amu) -/
def chlorine_weight : ℝ := 35.45

/-- The molecular weight of the compound in atomic mass units (amu) -/
def compound_weight : ℝ := 132

/-- The number of chlorine atoms in the compound -/
def chlorine_count : ℕ := 3

/-- The atomic weight of aluminum in atomic mass units (amu) -/
def aluminum_weight : ℝ := compound_weight - chlorine_count * chlorine_weight

theorem aluminum_atomic_weight :
  aluminum_weight = 25.65 := by sorry

end NUMINAMATH_CALUDE_aluminum_atomic_weight_l438_43824


namespace NUMINAMATH_CALUDE_rectangle_ratio_l438_43818

/-- Proves that a rectangle with area 100 m² and length 20 m has a length-to-width ratio of 4:1 -/
theorem rectangle_ratio (area : ℝ) (length : ℝ) (width : ℝ) 
  (h_area : area = 100) 
  (h_length : length = 20) 
  (h_rect : area = length * width) : 
  length / width = 4 := by
sorry

end NUMINAMATH_CALUDE_rectangle_ratio_l438_43818


namespace NUMINAMATH_CALUDE_mary_flour_calculation_l438_43878

/-- Given a cake recipe that requires 12 cups of flour, and knowing that Mary still needs 2 more cups,
    prove that Mary has already put in 10 cups of flour. -/
theorem mary_flour_calculation (recipe_flour : ℕ) (flour_needed : ℕ) (flour_put_in : ℕ) : 
  recipe_flour = 12 → flour_needed = 2 → flour_put_in = recipe_flour - flour_needed := by
  sorry

#check mary_flour_calculation

end NUMINAMATH_CALUDE_mary_flour_calculation_l438_43878


namespace NUMINAMATH_CALUDE_largest_value_is_E_l438_43840

theorem largest_value_is_E :
  let a := 24680 + 2 / 1357
  let b := 24680 - 2 / 1357
  let c := 24680 * 2 / 1357
  let d := 24680 / (2 / 1357)
  let e := 24680 ^ 1.357
  (e > a) ∧ (e > b) ∧ (e > c) ∧ (e > d) := by
  sorry

end NUMINAMATH_CALUDE_largest_value_is_E_l438_43840


namespace NUMINAMATH_CALUDE_total_lunch_combinations_l438_43893

def meat_dishes : ℕ := 4
def vegetable_dishes : ℕ := 7

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

def case1_combinations : ℕ := (choose meat_dishes 2) * (choose vegetable_dishes 2)
def case2_combinations : ℕ := (choose meat_dishes 1) * (choose vegetable_dishes 2)

theorem total_lunch_combinations : 
  case1_combinations + case2_combinations = 210 :=
by sorry

end NUMINAMATH_CALUDE_total_lunch_combinations_l438_43893


namespace NUMINAMATH_CALUDE_nat_less_than_5_finite_int_solution_set_nonempty_l438_43848

-- Define the set of natural numbers less than 5
def nat_less_than_5 : Set ℕ := {n | n < 5}

-- Define the set of integers satisfying 2x + 1 > 7
def int_solution_set : Set ℤ := {x | 2 * x + 1 > 7}

-- Theorem 1: The set of natural numbers less than 5 is finite
theorem nat_less_than_5_finite : Finite nat_less_than_5 := by sorry

-- Theorem 2: The set of integers satisfying 2x + 1 > 7 is non-empty
theorem int_solution_set_nonempty : Set.Nonempty int_solution_set := by sorry

end NUMINAMATH_CALUDE_nat_less_than_5_finite_int_solution_set_nonempty_l438_43848


namespace NUMINAMATH_CALUDE_other_number_is_64_l438_43856

/-- Given two positive integers with specific LCM and HCF, prove that one is 64 -/
theorem other_number_is_64 (A B : ℕ+) (h1 : A = 48) 
  (h2 : Nat.lcm A B = 192) (h3 : Nat.gcd A B = 16) : B = 64 := by
  sorry

end NUMINAMATH_CALUDE_other_number_is_64_l438_43856


namespace NUMINAMATH_CALUDE_intersection_value_l438_43822

theorem intersection_value (a : ℝ) : 
  let A := {x : ℝ | x^2 - 4 ≤ 0}
  let B := {x : ℝ | 2*x + a ≤ 0}
  (A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ 1}) → a = -4 := by
sorry

end NUMINAMATH_CALUDE_intersection_value_l438_43822


namespace NUMINAMATH_CALUDE_art_supplies_cost_l438_43805

def total_spent : ℕ := 50
def num_skirts : ℕ := 2
def skirt_cost : ℕ := 15

theorem art_supplies_cost : total_spent - (num_skirts * skirt_cost) = 20 := by
  sorry

end NUMINAMATH_CALUDE_art_supplies_cost_l438_43805


namespace NUMINAMATH_CALUDE_baseball_gear_cost_l438_43865

theorem baseball_gear_cost (initial_amount : ℕ) (remaining_amount : ℕ) 
  (h1 : initial_amount = 67)
  (h2 : remaining_amount = 33) :
  initial_amount - remaining_amount = 34 := by
  sorry

end NUMINAMATH_CALUDE_baseball_gear_cost_l438_43865


namespace NUMINAMATH_CALUDE_robin_gum_total_l438_43895

theorem robin_gum_total (packages : ℕ) (pieces_per_package : ℕ) (extra_pieces : ℕ) : 
  packages = 5 → pieces_per_package = 7 → extra_pieces = 6 →
  packages * pieces_per_package + extra_pieces = 41 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_total_l438_43895


namespace NUMINAMATH_CALUDE_third_divisor_l438_43844

theorem third_divisor (x : ℕ) : 
  x - 16 = 136 →
  4 ∣ (x - 16) →
  6 ∣ (x - 16) →
  10 ∣ (x - 16) →
  19 ≠ 4 →
  19 ≠ 6 →
  19 ≠ 10 →
  19 ∣ x :=
by
  sorry

end NUMINAMATH_CALUDE_third_divisor_l438_43844


namespace NUMINAMATH_CALUDE_equation_solutions_l438_43815

theorem equation_solutions :
  (∀ x : ℝ, x * (x + 2) - (x + 2) = 0 ↔ x = -2 ∨ x = 1) ∧
  (∀ x : ℝ, x^2 - 4*x - 3 = 0 ↔ x = 2 + Real.sqrt 7 ∨ x = 2 - Real.sqrt 7) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l438_43815


namespace NUMINAMATH_CALUDE_panda_equation_l438_43823

/-- Represents a digit from 0 to 9 -/
def Digit := Fin 10

/-- Condition that all digits are distinct -/
def all_distinct (a b c d e : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

/-- Convert a two-digit number to a natural number -/
def to_nat (tens units : Digit) : ℕ :=
  10 * tens.val + units.val

/-- Convert a three-digit number to a natural number -/
def to_nat_3 (hundreds tens units : Digit) : ℕ :=
  100 * hundreds.val + 10 * tens.val + units.val

theorem panda_equation (tuan yuan da xiong mao : Digit)
  (h_distinct : all_distinct tuan yuan da xiong mao)
  (h_eq : to_nat tuan tuan * to_nat yuan yuan = to_nat_3 da xiong mao) :
  da.val + xiong.val + mao.val = 23 := by
  sorry

end NUMINAMATH_CALUDE_panda_equation_l438_43823


namespace NUMINAMATH_CALUDE_exists_unobserved_planet_l438_43894

/-- Represents a planet in the solar system -/
structure Planet where
  id : Nat

/-- The solar system with its properties -/
class SolarSystem where
  planets : Finset Planet
  distance : Planet → Planet → ℝ
  nearest_planet : Planet → Planet
  num_planets_odd : Odd planets.card
  distances_distinct : ∀ p1 p2 p3 p4 : Planet, p1 ≠ p2 ∧ p3 ≠ p4 → distance p1 p2 ≠ distance p3 p4
  nearest_is_nearest : ∀ p1 p2 : Planet, p1 ≠ p2 → distance p1 (nearest_planet p1) ≤ distance p1 p2

/-- Main theorem: There exists a planet that no astronomer is observing -/
theorem exists_unobserved_planet [s : SolarSystem] :
  ∃ p : Planet, p ∈ s.planets ∧ ∀ q : Planet, q ∈ s.planets → s.nearest_planet q ≠ p :=
sorry

end NUMINAMATH_CALUDE_exists_unobserved_planet_l438_43894


namespace NUMINAMATH_CALUDE_inequality_solution_set_l438_43811

theorem inequality_solution_set (x : ℝ) :
  (2 / (x^2 + 2*x + 1) + 4 / (x^2 + 8*x + 7) > 3/2) ↔ 
  (x < -7 ∨ (-7 < x ∧ x < -1) ∨ x > -1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l438_43811


namespace NUMINAMATH_CALUDE_complex_equation_solution_l438_43870

theorem complex_equation_solution (a b : ℝ) :
  (1 + 2*I : ℂ)*a + b = 2*I → a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l438_43870


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l438_43874

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_second : a 2 = 1)
  (h_relation : a 8 = a 6 + 2 * a 4) :
  a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l438_43874


namespace NUMINAMATH_CALUDE_prob_different_suits_modified_deck_l438_43876

/-- A deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- The probability of drawing two cards of different suits -/
def prob_different_suits (d : Deck) : ℚ :=
  (d.total_cards - d.cards_per_suit) / (d.total_cards - 1)

/-- The modified 40-card deck -/
def modified_deck : Deck :=
  { total_cards := 40
  , num_suits := 4
  , cards_per_suit := 10
  , h_total := rfl }

theorem prob_different_suits_modified_deck :
  prob_different_suits modified_deck = 10 / 13 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_modified_deck_l438_43876


namespace NUMINAMATH_CALUDE_exponent_simplification_l438_43834

theorem exponent_simplification : ((-2 : ℝ) ^ 3) ^ (1/3) - (-1 : ℝ) ^ 0 = -3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l438_43834


namespace NUMINAMATH_CALUDE_masha_meeting_time_l438_43803

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents the scenario of Masha's journey home -/
structure MashaJourney where
  usual_end_time : Time
  usual_arrival_time : Time
  early_end_time : Time
  early_arrival_time : Time
  meeting_time : Time

/-- Calculate the time difference in minutes between two Time values -/
def time_diff_minutes (t1 t2 : Time) : ℤ :=
  (t1.hours - t2.hours) * 60 + (t1.minutes - t2.minutes)

/-- The main theorem to prove -/
theorem masha_meeting_time (journey : MashaJourney) : 
  journey.usual_end_time = ⟨13, 0, by norm_num⟩ →
  journey.early_end_time = ⟨12, 0, by norm_num⟩ →
  time_diff_minutes journey.usual_arrival_time journey.early_arrival_time = 12 →
  journey.meeting_time = ⟨12, 54, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_masha_meeting_time_l438_43803


namespace NUMINAMATH_CALUDE_test_score_problem_l438_43816

theorem test_score_problem (total_questions : ℕ) (correct_points : ℚ) (incorrect_penalty : ℚ) 
  (final_score : ℚ) (h1 : total_questions = 120) (h2 : correct_points = 1) 
  (h3 : incorrect_penalty = 1/4) (h4 : final_score = 100) : 
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_points - (total_questions - correct_answers) * incorrect_penalty = final_score ∧
    correct_answers = 104 := by
  sorry

end NUMINAMATH_CALUDE_test_score_problem_l438_43816


namespace NUMINAMATH_CALUDE_anchuria_laws_theorem_l438_43831

variables (K N M : ℕ) (p : ℝ)

/-- The probability that exactly M laws are included in the Concept -/
def prob_M_laws_included : ℝ :=
  (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)

/-- The expected number of laws included in the Concept -/
def expected_laws_included : ℝ :=
  K * (1 - (1 - p)^N)

/-- Theorem stating the correctness of the probability and expectation calculations -/
theorem anchuria_laws_theorem (h1 : 0 ≤ p) (h2 : p ≤ 1) (h3 : M ≤ K) :
  (prob_M_laws_included K N M p = (Nat.choose K M : ℝ) * (1 - (1 - p)^N)^M * ((1 - p)^N)^(K - M)) ∧
  (expected_laws_included K N p = K * (1 - (1 - p)^N)) := by
  sorry

end NUMINAMATH_CALUDE_anchuria_laws_theorem_l438_43831


namespace NUMINAMATH_CALUDE_height_difference_l438_43898

/-- Prove that the difference between 3 times Kim's height and Tamara's height is 4 inches -/
theorem height_difference (kim_height tamara_height : ℕ) : 
  tamara_height + kim_height = 92 →
  tamara_height = 68 →
  ∃ x, tamara_height = 3 * kim_height - x →
  3 * kim_height - tamara_height = 4 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l438_43898


namespace NUMINAMATH_CALUDE_ordering_of_a_ab_ab_squared_l438_43804

theorem ordering_of_a_ab_ab_squared (a b : ℝ) (ha : a < 0) (hb : b < -1) :
  a * b > a ∧ a > a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_ordering_of_a_ab_ab_squared_l438_43804


namespace NUMINAMATH_CALUDE_largest_among_four_l438_43842

theorem largest_among_four : ∀ (a b c d : ℝ), 
  a = 0 → b = -1 → c = -2 → d = Real.sqrt 3 →
  d = max a (max b (max c d)) :=
by sorry

end NUMINAMATH_CALUDE_largest_among_four_l438_43842


namespace NUMINAMATH_CALUDE_tom_marble_pairs_l438_43883

/-- Represents the set of marbles Tom has -/
structure MarbleSet where
  distinct_colors : Nat  -- Number of distinct colored marbles
  yellow_marbles : Nat   -- Number of identical yellow marbles

/-- Calculates the number of ways to choose 2 marbles from a given MarbleSet -/
def count_marble_pairs (ms : MarbleSet) : Nat :=
  let yellow_pair := if ms.yellow_marbles ≥ 2 then 1 else 0
  let distinct_pairs := Nat.choose ms.distinct_colors 2
  yellow_pair + distinct_pairs

/-- Theorem: Given Tom's marble set, the number of different groups of two marbles is 7 -/
theorem tom_marble_pairs :
  let toms_marbles : MarbleSet := ⟨4, 5⟩
  count_marble_pairs toms_marbles = 7 := by
  sorry

end NUMINAMATH_CALUDE_tom_marble_pairs_l438_43883


namespace NUMINAMATH_CALUDE_hexagon_side_length_squared_l438_43888

/-- A regular hexagon inscribed in an ellipse -/
structure InscribedHexagon where
  /-- The ellipse equation is x^2 + 9y^2 = 9 -/
  ellipse : ∀ (x y : ℝ), x^2 + 9*y^2 = 9 → ∃ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y
  /-- One vertex of the hexagon is (0,1) -/
  vertex : ∃ (v : ℝ × ℝ), v = (0, 1)
  /-- One diagonal of the hexagon is aligned along the y-axis -/
  diagonal : ∃ (d : ℝ × ℝ) (e : ℝ × ℝ), d.1 = 0 ∧ e.1 = 0 ∧ d.2 = -e.2
  /-- The hexagon is regular -/
  regular : ∀ (s1 s2 : ℝ × ℝ), s1 ≠ s2 → ‖s1 - s2‖ = ‖s2 - s1‖

/-- The square of the length of each side of the hexagon is 729/98 -/
theorem hexagon_side_length_squared (h : InscribedHexagon) : 
  ∃ (s1 s2 : ℝ × ℝ), s1 ≠ s2 ∧ ‖s1 - s2‖^2 = 729/98 :=
sorry

end NUMINAMATH_CALUDE_hexagon_side_length_squared_l438_43888


namespace NUMINAMATH_CALUDE_milk_students_l438_43858

theorem milk_students (total : ℕ) (soda_percent : ℚ) (milk_percent : ℚ) (soda_count : ℕ) :
  soda_percent = 70 / 100 →
  milk_percent = 20 / 100 →
  soda_count = 84 →
  total = soda_count / soda_percent →
  ↑(total * milk_percent) = 24 := by
  sorry

end NUMINAMATH_CALUDE_milk_students_l438_43858


namespace NUMINAMATH_CALUDE_concert_ticket_price_l438_43845

theorem concert_ticket_price :
  ∀ (ticket_price : ℚ),
    (2 * ticket_price) +                    -- Cost of two tickets
    (0.15 * 2 * ticket_price) +             -- 15% processing fee
    10 +                                    -- Parking fee
    (2 * 5) =                               -- Entrance fee for two people
    135 →                                   -- Total cost
    ticket_price = 50 := by
  sorry

end NUMINAMATH_CALUDE_concert_ticket_price_l438_43845


namespace NUMINAMATH_CALUDE_wendy_bouquets_l438_43835

/-- Given the initial number of flowers, flowers per bouquet, and number of wilted flowers,
    calculate the number of bouquets that can be made. -/
def bouquets_remaining (initial_flowers : ℕ) (flowers_per_bouquet : ℕ) (wilted_flowers : ℕ) : ℕ :=
  (initial_flowers - wilted_flowers) / flowers_per_bouquet

/-- Prove that Wendy can make 2 bouquets with the remaining flowers. -/
theorem wendy_bouquets :
  bouquets_remaining 45 5 35 = 2 := by
  sorry

end NUMINAMATH_CALUDE_wendy_bouquets_l438_43835


namespace NUMINAMATH_CALUDE_cyclist_meeting_theorem_l438_43843

def cyclist_meeting_time (t1 t2 t3 : ℚ) : ℚ :=
  let n : ℕ := 9
  let m : ℕ := 5
  (35 * n) / 2

theorem cyclist_meeting_theorem (t1 t2 t3 : ℚ) 
  (h1 : t1 = 5)
  (h2 : t2 = 7)
  (h3 : t3 = 9)
  (h4 : t1 > 0 ∧ t2 > 0 ∧ t3 > 0) :
  cyclist_meeting_time t1 t2 t3 = 315/2 :=
by sorry

end NUMINAMATH_CALUDE_cyclist_meeting_theorem_l438_43843


namespace NUMINAMATH_CALUDE_prime_pair_perfect_square_sum_theorem_l438_43890

/-- A pair of prime numbers (p, q) such that p^2 + 5pq + 4q^2 is a perfect square -/
def PrimePairWithPerfectSquareSum (p q : ℕ) : Prop :=
  Nat.Prime p ∧ Nat.Prime q ∧ ∃ k : ℕ, p^2 + 5*p*q + 4*q^2 = k^2

/-- The theorem stating that only three specific pairs of prime numbers satisfy the condition -/
theorem prime_pair_perfect_square_sum_theorem :
  ∀ p q : ℕ, PrimePairWithPerfectSquareSum p q ↔ 
    ((p = 13 ∧ q = 3) ∨ (p = 5 ∧ q = 11) ∨ (p = 7 ∧ q = 5)) :=
by sorry

end NUMINAMATH_CALUDE_prime_pair_perfect_square_sum_theorem_l438_43890


namespace NUMINAMATH_CALUDE_park_trees_after_planting_l438_43879

theorem park_trees_after_planting (current_trees new_trees : ℕ) 
  (h1 : current_trees = 25)
  (h2 : new_trees = 73) :
  current_trees + new_trees = 98 :=
by sorry

end NUMINAMATH_CALUDE_park_trees_after_planting_l438_43879


namespace NUMINAMATH_CALUDE_m_range_l438_43880

def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

theorem m_range (m : ℝ) : A ⊆ B m ∧ A ≠ B m → m > 2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l438_43880


namespace NUMINAMATH_CALUDE_function_passes_through_point_l438_43814

theorem function_passes_through_point (a : ℝ) (h : 0 < a ∧ a < 1) :
  let f : ℝ → ℝ := λ x ↦ 2 * a * x - 1
  f 1 = 1 := by sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l438_43814


namespace NUMINAMATH_CALUDE_diana_tue_thu_hours_l438_43860

/-- Represents Diana's work schedule and earnings --/
structure WorkSchedule where
  hours_mon_wed_fri : ℕ
  weekly_earnings : ℕ
  hourly_rate : ℕ

/-- Calculates the number of hours Diana works on Tuesday and Thursday --/
def hours_tue_thu (schedule : WorkSchedule) : ℕ :=
  schedule.weekly_earnings / schedule.hourly_rate - 3 * schedule.hours_mon_wed_fri

/-- Theorem stating that Diana works 30 hours on Tuesday and Thursday --/
theorem diana_tue_thu_hours (schedule : WorkSchedule) 
  (h1 : schedule.hours_mon_wed_fri = 10)
  (h2 : schedule.weekly_earnings = 1800)
  (h3 : schedule.hourly_rate = 30) :
  hours_tue_thu schedule = 30 := by
  sorry

#eval hours_tue_thu { hours_mon_wed_fri := 10, weekly_earnings := 1800, hourly_rate := 30 }

end NUMINAMATH_CALUDE_diana_tue_thu_hours_l438_43860


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l438_43832

theorem fraction_equals_zero (x : ℝ) (h : x + 1 ≠ 0) :
  x = 1 → (x^2 - 1) / (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l438_43832


namespace NUMINAMATH_CALUDE_petya_has_winning_strategy_l438_43800

/-- Represents a player in the game -/
inductive Player : Type
| Petya : Player
| Vasya : Player

/-- Represents the game state -/
structure GameState :=
  (grid : Matrix (Fin 2021) (Fin 2021) Bool)
  (currentPlayer : Player)

/-- Checks if a rectangle contains a piece -/
def rectangleContainsPiece (state : GameState) (x y width height : Nat) : Bool :=
  sorry

/-- Checks if the game is over (every 3x5 and 5x3 rectangle contains a piece) -/
def isGameOver (state : GameState) : Bool :=
  sorry

/-- Represents a valid move in the game -/
structure Move :=
  (row : Fin 2021)
  (col : Fin 2021)

/-- Applies a move to the game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  sorry

/-- Determines if a move is valid (cell is empty) -/
def isValidMove (state : GameState) (move : Move) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Determines if a strategy is a winning strategy for the given player -/
def isWinningStrategy (player : Player) (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem: Petya has a winning strategy -/
theorem petya_has_winning_strategy :
  ∃ (strategy : Strategy), isWinningStrategy Player.Petya strategy :=
sorry

end NUMINAMATH_CALUDE_petya_has_winning_strategy_l438_43800


namespace NUMINAMATH_CALUDE_parabola_symmetry_l438_43808

/-- Prove that if (2, 3) lies on the parabola y = ax^2 + 2ax + c, then (-4, 3) also lies on it -/
theorem parabola_symmetry (a c : ℝ) : 
  (3 = a * 2^2 + 2 * a * 2 + c) → (3 = a * (-4)^2 + 2 * a * (-4) + c) := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l438_43808


namespace NUMINAMATH_CALUDE_real_roots_condition_specific_condition_l438_43882

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 + (2*m - 1)*x + m^2

-- Part 1: Real roots condition
theorem real_roots_condition (m : ℝ) :
  (∃ x : ℝ, quadratic m x = 0) ↔ m ≤ 1/4 :=
sorry

-- Part 2: Specific condition leading to m = -1
theorem specific_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic m x₁ = 0 ∧ quadratic m x₂ = 0 ∧ x₁*x₂ + x₁ + x₂ = 4) →
  m = -1 :=
sorry

end NUMINAMATH_CALUDE_real_roots_condition_specific_condition_l438_43882


namespace NUMINAMATH_CALUDE_pirate_treasure_sum_l438_43830

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (7 ^ i)) 0

/-- The value of silver medallions in base 7 --/
def silverValue : List Nat := [6, 2, 3, 5]

/-- The value of precious gemstones in base 7 --/
def gemstonesValue : List Nat := [1, 6, 4, 3]

/-- The value of spices in base 7 --/
def spicesValue : List Nat := [6, 5, 6]

theorem pirate_treasure_sum :
  base7ToBase10 silverValue + base7ToBase10 gemstonesValue + base7ToBase10 spicesValue = 3485 := by
  sorry


end NUMINAMATH_CALUDE_pirate_treasure_sum_l438_43830


namespace NUMINAMATH_CALUDE_sqrt_five_squared_times_four_to_sixth_l438_43851

theorem sqrt_five_squared_times_four_to_sixth (x : ℝ) : x = Real.sqrt (5^2 * 4^6) → x = 320 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_squared_times_four_to_sixth_l438_43851


namespace NUMINAMATH_CALUDE_install_time_proof_l438_43862

/-- Calculates the time needed to install remaining windows -/
def time_to_install_remaining (total : ℕ) (installed : ℕ) (time_per_window : ℕ) : ℕ :=
  (total - installed) * time_per_window

/-- Proves that the time to install the remaining windows is 48 hours -/
theorem install_time_proof (total : ℕ) (installed : ℕ) (time_per_window : ℕ)
  (h1 : total = 14)
  (h2 : installed = 8)
  (h3 : time_per_window = 8) :
  time_to_install_remaining total installed time_per_window = 48 := by
  sorry

#eval time_to_install_remaining 14 8 8

end NUMINAMATH_CALUDE_install_time_proof_l438_43862


namespace NUMINAMATH_CALUDE_smallest_valid_sum_of_cubes_l438_43891

def is_valid (n : ℕ) : Prop :=
  ∀ p, Nat.Prime p → p ∣ n → p > 18

def is_sum_of_cubes (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = a^3 + b^3

theorem smallest_valid_sum_of_cubes : 
  is_valid 1843 ∧ 
  is_sum_of_cubes 1843 ∧ 
  ∀ m : ℕ, m < 1843 → ¬(is_valid m ∧ is_sum_of_cubes m) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_sum_of_cubes_l438_43891


namespace NUMINAMATH_CALUDE_study_time_difference_l438_43847

-- Define the study times
def kwame_hours : ℝ := 2.5
def connor_hours : ℝ := 1.5
def lexia_minutes : ℝ := 97

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℝ := 60

-- Theorem to prove
theorem study_time_difference : 
  (kwame_hours * minutes_per_hour + connor_hours * minutes_per_hour) - lexia_minutes = 143 := by
  sorry

end NUMINAMATH_CALUDE_study_time_difference_l438_43847


namespace NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l438_43886

theorem odd_square_minus_one_div_eight (n : ℤ) : 
  ∃ k : ℤ, (2*n + 1)^2 - 1 = 8*k :=
by sorry

end NUMINAMATH_CALUDE_odd_square_minus_one_div_eight_l438_43886


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l438_43821

theorem cubic_polynomial_roots : ∃ (r₁ r₂ : ℝ), 
  (∀ x : ℝ, x^3 - 7*x^2 + 8*x + 16 = 0 ↔ x = r₁ ∨ x = r₂) ∧
  r₁ = -1 ∧ r₂ = 4 ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - r₂| < δ → |x^3 - 7*x^2 + 8*x + 16| < ε * |x - r₂|^2) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l438_43821


namespace NUMINAMATH_CALUDE_permutation_problem_arrangement_problem_photo_arrangement_problem_l438_43837

-- Problem 1
theorem permutation_problem (m : ℕ) : 
  (Nat.factorial 10) / (Nat.factorial (10 - m)) = (Nat.factorial 10) / (Nat.factorial 4) → m = 6 := by
sorry

-- Problem 2
theorem arrangement_problem : 
  (Nat.factorial 3) = 6 := by
sorry

-- Problem 3
theorem photo_arrangement_problem : 
  2 * 4 * (Nat.factorial 4) = 192 := by
sorry

end NUMINAMATH_CALUDE_permutation_problem_arrangement_problem_photo_arrangement_problem_l438_43837


namespace NUMINAMATH_CALUDE_cubic_polynomial_property_l438_43852

/-- A cubic polynomial with integer coefficients -/
def cubic_polynomial (a b : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + 9*a

/-- Predicate for a cubic polynomial having two coincident roots -/
def has_coincident_roots (a b : ℤ) : Prop :=
  ∃ r s : ℤ, r ≠ s ∧ 
    ∀ x : ℝ, cubic_polynomial a b x = (x - r)^2 * (x - s)

/-- Theorem stating that under given conditions, |ab| = 1344 -/
theorem cubic_polynomial_property (a b : ℤ) :
  a ≠ 0 → b ≠ 0 → has_coincident_roots a b → |a*b| = 1344 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_property_l438_43852


namespace NUMINAMATH_CALUDE_distance_between_trees_l438_43839

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : 
  yard_length = 273 ∧ num_trees = 14 → 
  (yard_length : ℚ) / (num_trees - 1 : ℚ) = 21 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l438_43839


namespace NUMINAMATH_CALUDE_f_always_positive_l438_43869

def f (x : ℝ) : ℝ := x^2 + 3*x + 4

theorem f_always_positive : ∀ x : ℝ, f x > 0 := by
  sorry

end NUMINAMATH_CALUDE_f_always_positive_l438_43869


namespace NUMINAMATH_CALUDE_point_of_tangency_parabolas_l438_43846

/-- The point of tangency for two parabolas -/
theorem point_of_tangency_parabolas :
  let f (x : ℝ) := x^2 + 10*x + 18
  let g (y : ℝ) := y^2 + 60*y + 910
  ∃! p : ℝ × ℝ, 
    (p.2 = f p.1 ∧ p.1 = g p.2) ∧ 
    (∀ x y, y = f x ∧ x = g y → (x, y) = p) :=
by
  sorry

end NUMINAMATH_CALUDE_point_of_tangency_parabolas_l438_43846


namespace NUMINAMATH_CALUDE_not_necessarily_divisible_by_44_l438_43884

theorem not_necessarily_divisible_by_44 (k : ℤ) (n : ℤ) : 
  n = k * (k + 1) * (k + 2) → 
  11 ∣ n → 
  ¬ (∀ m : ℤ, n = k * (k + 1) * (k + 2) ∧ 11 ∣ n → 44 ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_not_necessarily_divisible_by_44_l438_43884


namespace NUMINAMATH_CALUDE_simple_interest_problem_l438_43812

/-- Proves that for a principal of 1000 Rs., if increasing the interest rate by 3%
    results in 90 Rs. more interest, then the time period for which the sum was invested is 3 years. -/
theorem simple_interest_problem (R : ℝ) (T : ℝ) :
  (1000 * R * T / 100 + 90 = 1000 * (R + 3) * T / 100) →
  T = 3 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l438_43812


namespace NUMINAMATH_CALUDE_second_person_average_pages_per_day_l438_43864

/-- The average number of pages read per day by the second person -/
def average_pages_per_day (summer_days : ℕ) (books_read : ℕ) (pages_per_book : ℕ) (second_person_percentage : ℚ) : ℚ :=
  (books_read * pages_per_book : ℚ) * second_person_percentage / summer_days

/-- Theorem stating that the average number of pages read per day by the second person is 180 -/
theorem second_person_average_pages_per_day :
  average_pages_per_day 80 60 320 (3/4) = 180 := by
  sorry

end NUMINAMATH_CALUDE_second_person_average_pages_per_day_l438_43864


namespace NUMINAMATH_CALUDE_fill_time_is_100_l438_43827

/-- Represents the water filling system with three pipes and a tank -/
structure WaterSystem where
  tankCapacity : ℕ
  pipeARate : ℕ
  pipeBRate : ℕ
  pipeCRate : ℕ
  pipeATime : ℕ
  pipeBTime : ℕ
  pipeCTime : ℕ

/-- Calculates the time required to fill the tank -/
def fillTime (sys : WaterSystem) : ℕ :=
  let cycleAmount := sys.pipeARate * sys.pipeATime + sys.pipeBRate * sys.pipeBTime - sys.pipeCRate * sys.pipeCTime
  let cycles := (sys.tankCapacity + cycleAmount - 1) / cycleAmount
  cycles * (sys.pipeATime + sys.pipeBTime + sys.pipeCTime)

/-- Theorem stating that the fill time for the given system is 100 minutes -/
theorem fill_time_is_100 (sys : WaterSystem) 
  (h1 : sys.tankCapacity = 5000)
  (h2 : sys.pipeARate = 200)
  (h3 : sys.pipeBRate = 50)
  (h4 : sys.pipeCRate = 25)
  (h5 : sys.pipeATime = 1)
  (h6 : sys.pipeBTime = 2)
  (h7 : sys.pipeCTime = 2) :
  fillTime sys = 100 := by
  sorry

#eval fillTime { tankCapacity := 5000, pipeARate := 200, pipeBRate := 50, pipeCRate := 25, 
                 pipeATime := 1, pipeBTime := 2, pipeCTime := 2 }

end NUMINAMATH_CALUDE_fill_time_is_100_l438_43827


namespace NUMINAMATH_CALUDE_chinese_space_station_altitude_l438_43877

theorem chinese_space_station_altitude :
  ∃ (n : ℝ), n = 389000 ∧ n = 3.89 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_chinese_space_station_altitude_l438_43877


namespace NUMINAMATH_CALUDE_hcf_proof_l438_43892

/-- Given two positive integers with specific HCF and LCM, prove that their HCF is 20 -/
theorem hcf_proof (a b : ℕ) (h1 : Nat.gcd a b = 20) (h2 : Nat.lcm a b = 396) (h3 : a = 36) :
  Nat.gcd a b = 20 := by
  sorry

end NUMINAMATH_CALUDE_hcf_proof_l438_43892


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l438_43887

theorem defective_shipped_percentage
  (total_units : ℝ)
  (defective_rate : ℝ)
  (shipped_rate : ℝ)
  (h1 : defective_rate = 0.07)
  (h2 : shipped_rate = 0.05) :
  (defective_rate * shipped_rate) * 100 = 0.35 := by
sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l438_43887


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l438_43854

/-- Given vectors a and b, if they are parallel, then the magnitude of a is 2. -/
theorem parallel_vectors_magnitude (x : ℝ) :
  let a : Fin 2 → ℝ := ![1, x]
  let b : Fin 2 → ℝ := ![x, 3]
  (∃ (k : ℝ), a = k • b) → ‖a‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l438_43854


namespace NUMINAMATH_CALUDE_unique_solution_for_g_l438_43817

/-- Given functions f and g where g(x) = 4f⁻¹(x) and f(x) = 30 / (x + 4),
    prove that the unique value of x satisfying g(x) = 20 is 10/3 -/
theorem unique_solution_for_g (f g : ℝ → ℝ) 
    (h1 : ∀ x, g x = 4 * (f⁻¹ x)) 
    (h2 : ∀ x, f x = 30 / (x + 4)) : 
    ∃! x, g x = 20 ∧ x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_g_l438_43817


namespace NUMINAMATH_CALUDE_min_squares_for_25x25_grid_l438_43863

/-- Represents a square grid -/
structure SquareGrid where
  size : ℕ
  total_squares : ℕ

/-- Calculates the minimum number of 1x1 squares needed to create an image of a square grid -/
def min_squares_for_image (grid : SquareGrid) : ℕ :=
  let perimeter := 4 * grid.size - 4
  let interior := (grid.size - 2) * (grid.size - 2)
  let dominos := interior / 2
  perimeter + dominos

/-- Theorem stating the minimum number of squares needed for a 25x25 grid -/
theorem min_squares_for_25x25_grid :
  ∃ (grid : SquareGrid), grid.size = 25 ∧ grid.total_squares = 625 ∧ min_squares_for_image grid = 360 := by
  sorry

end NUMINAMATH_CALUDE_min_squares_for_25x25_grid_l438_43863


namespace NUMINAMATH_CALUDE_rope_cutting_probability_rope_cutting_probability_is_one_third_l438_43850

/-- The probability of cutting a rope of length 3 into two segments,
    each at least 1 unit long, when cut at a random position. -/
theorem rope_cutting_probability : ℝ :=
  let rope_length : ℝ := 3
  let min_segment_length : ℝ := 1
  let favorable_cut_length : ℝ := rope_length - 2 * min_segment_length
  favorable_cut_length / rope_length

/-- The probability of cutting a rope of length 3 into two segments,
    each at least 1 unit long, when cut at a random position, is 1/3. -/
theorem rope_cutting_probability_is_one_third :
  rope_cutting_probability = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_probability_rope_cutting_probability_is_one_third_l438_43850


namespace NUMINAMATH_CALUDE_constant_term_expansion_l438_43899

/-- The constant term in the expansion of (1/x + 2x)^6 is 160 -/
theorem constant_term_expansion : ∃ c : ℕ, c = 160 ∧ 
  ∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, (λ x => (1/x + 2*x)^6) = (λ x => c + x * f x)) := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l438_43899


namespace NUMINAMATH_CALUDE_unique_matrix_transformation_l438_43838

theorem unique_matrix_transformation (A : Matrix (Fin 2) (Fin 2) ℝ) :
  ∃! M : Matrix (Fin 2) (Fin 2) ℝ,
    (∀ i j, (M * A) i j = if j = 1 then (if i = 0 then 2 * A i j else 3 * A i j) else A i j) ∧
    M = ![![1, 0], ![0, 3]] := by
  sorry

end NUMINAMATH_CALUDE_unique_matrix_transformation_l438_43838


namespace NUMINAMATH_CALUDE_alien_energy_cells_l438_43859

/-- Converts a number from base 7 to base 10 --/
def base7ToBase10 (hundreds : ℕ) (tens : ℕ) (ones : ℕ) : ℕ :=
  hundreds * 7^2 + tens * 7^1 + ones * 7^0

/-- Theorem stating that 321 in base 7 is equal to 162 in base 10 --/
theorem alien_energy_cells : base7ToBase10 3 2 1 = 162 := by
  sorry

end NUMINAMATH_CALUDE_alien_energy_cells_l438_43859


namespace NUMINAMATH_CALUDE_wire_service_reporters_l438_43829

theorem wire_service_reporters (total : ℝ) (h_total : total > 0) :
  let local_politics := 0.28 * total
  let all_politics := local_politics / 0.7
  let non_politics := total - all_politics
  non_politics / total = 0.6 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l438_43829


namespace NUMINAMATH_CALUDE_function_identity_proof_l438_43896

theorem function_identity_proof (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), (f m)^2 + f n ∣ (m^2 + n)^2) : 
  ∀ (n : ℕ+), f n = n := by
  sorry

end NUMINAMATH_CALUDE_function_identity_proof_l438_43896


namespace NUMINAMATH_CALUDE_digit_sum_difference_l438_43801

theorem digit_sum_difference : ∃ (a b : Nat), 
  (a ≠ b) ∧ 
  (a < 10) ∧ 
  (b < 10) ∧
  (123456789 * 8 = 987654300 + a * 10 + b) ∧ 
  (987654321 = 987654300 + b * 10 + a) ∧
  (a + b = 3) := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_difference_l438_43801


namespace NUMINAMATH_CALUDE_centroid_altitude_length_l438_43868

/-- Triangle XYZ with sides a, b, c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- Foot of the altitude from a point to a line segment -/
def altitude_foot (point : ℝ × ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem centroid_altitude_length (t : Triangle) (h1 : t.a = 13) (h2 : t.b = 15) (h3 : t.c = 24) :
  let g := centroid t
  let yz := ((0, 0), (t.c, 0))  -- Assuming YZ is on the x-axis
  let q := altitude_foot g yz
  distance g q = 2.4 := by sorry

end NUMINAMATH_CALUDE_centroid_altitude_length_l438_43868


namespace NUMINAMATH_CALUDE_max_x_implies_a_value_l438_43806

/-- Given that the maximum value of x satisfying (x² - 4x + a) + |x - 3| ≤ 5 is 3, prove that a = 8 -/
theorem max_x_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + a) + |x - 3| ≤ 5 → x ≤ 3) ∧ 
  (∃ x : ℝ, x = 3 ∧ (x^2 - 4*x + a) + |x - 3| = 5) → 
  a = 8 := by
sorry

end NUMINAMATH_CALUDE_max_x_implies_a_value_l438_43806


namespace NUMINAMATH_CALUDE_sum_binary_digits_365_l438_43872

/-- Sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ :=
  (n.digits 2).sum

/-- The sum of the digits in the binary representation of 365 is 6 -/
theorem sum_binary_digits_365 : sumBinaryDigits 365 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_binary_digits_365_l438_43872


namespace NUMINAMATH_CALUDE_steven_arrangement_count_l438_43885

/-- The number of letters in "STEVEN" excluding one "E" -/
def n : ℕ := 5

/-- The number of permutations of "STEVEN" with one "E" fixed at the end -/
def steven_permutations : ℕ := n.factorial

theorem steven_arrangement_count : steven_permutations = 120 := by
  sorry

end NUMINAMATH_CALUDE_steven_arrangement_count_l438_43885


namespace NUMINAMATH_CALUDE_function_composition_problem_l438_43897

theorem function_composition_problem (a b : ℝ) : 
  (∀ x, (3 * ((a * x) + b) - 4) = 4 * x + 5) → 
  a + b = 13 / 3 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_problem_l438_43897


namespace NUMINAMATH_CALUDE_wall_height_l438_43866

-- Define the width of the wall
def wall_width : ℝ := 4

-- Define the area of the wall
def wall_area : ℝ := 16

-- Theorem: The height of the wall is 4 feet
theorem wall_height : 
  wall_area / wall_width = 4 :=
by sorry

end NUMINAMATH_CALUDE_wall_height_l438_43866


namespace NUMINAMATH_CALUDE_A_n_squared_value_l438_43857

theorem A_n_squared_value (n : ℕ) : (n.choose 2 = 15) → (n * (n - 1) = 30) := by
  sorry

end NUMINAMATH_CALUDE_A_n_squared_value_l438_43857


namespace NUMINAMATH_CALUDE_level3_available_spots_l438_43833

/-- Represents a parking level in a multi-story parking lot -/
structure ParkingLevel where
  totalSpots : ℕ
  parkedCars : ℕ
  reservedParkedCars : ℕ

/-- Calculates the available non-reserved parking spots on a given level -/
def availableNonReservedSpots (level : ParkingLevel) : ℕ :=
  level.totalSpots - (level.parkedCars - level.reservedParkedCars)

/-- Theorem stating that the available non-reserved parking spots on level 3 is 450 -/
theorem level3_available_spots :
  let level3 : ParkingLevel := {
    totalSpots := 480,
    parkedCars := 45,
    reservedParkedCars := 15
  }
  availableNonReservedSpots level3 = 450 := by
  sorry

end NUMINAMATH_CALUDE_level3_available_spots_l438_43833


namespace NUMINAMATH_CALUDE_factorial_difference_l438_43875

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: 6! - 4! = 696 -/
theorem factorial_difference : factorial 6 - factorial 4 = 696 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l438_43875


namespace NUMINAMATH_CALUDE_longest_altitudes_sum_is_31_l438_43819

/-- A right triangle with sides 7, 24, and 25 -/
structure RightTriangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (side_a : a = 7)
  (side_b : b = 24)
  (side_c : c = 25)

/-- The sum of the lengths of the two longest altitudes in the right triangle -/
def longest_altitudes_sum (t : RightTriangle) : ℝ :=
  t.a + t.b

/-- Theorem: The sum of the lengths of the two longest altitudes in the given right triangle is 31 -/
theorem longest_altitudes_sum_is_31 (t : RightTriangle) :
  longest_altitudes_sum t = 31 := by
  sorry

end NUMINAMATH_CALUDE_longest_altitudes_sum_is_31_l438_43819


namespace NUMINAMATH_CALUDE_price_difference_l438_43809

theorem price_difference (P : ℝ) (P_positive : P > 0) : 
  let new_price := P * 1.2
  let discounted_price := new_price * 0.8
  new_price - discounted_price = P * 0.24 :=
by sorry

end NUMINAMATH_CALUDE_price_difference_l438_43809


namespace NUMINAMATH_CALUDE_cycle_original_price_l438_43802

/-- Proves that the original price of a cycle is 1600 when sold at a 10% loss for 1440 --/
theorem cycle_original_price (selling_price : ℝ) (loss_percentage : ℝ) 
  (h1 : selling_price = 1440)
  (h2 : loss_percentage = 10) : 
  selling_price / (1 - loss_percentage / 100) = 1600 := by
sorry

end NUMINAMATH_CALUDE_cycle_original_price_l438_43802


namespace NUMINAMATH_CALUDE_max_factors_x8_minus_1_l438_43836

theorem max_factors_x8_minus_1 : 
  ∃ (k : ℕ), k = 5 ∧ 
  (∀ (p : List (Polynomial ℝ)), 
    (∀ q ∈ p, q.degree > 0) → -- Each factor is non-constant
    (List.prod p = Polynomial.X ^ 8 - 1) → -- The product of factors equals x^8 - 1
    List.length p ≤ k) ∧ -- The number of factors is at most k
  (∃ (p : List (Polynomial ℝ)), 
    (∀ q ∈ p, q.degree > 0) ∧ 
    (List.prod p = Polynomial.X ^ 8 - 1) ∧ 
    List.length p = k) -- There exists a factorization with exactly k factors
  := by sorry

end NUMINAMATH_CALUDE_max_factors_x8_minus_1_l438_43836


namespace NUMINAMATH_CALUDE_right_triangle_leg_sum_l438_43810

theorem right_triangle_leg_sum : ∃ (a b : ℕ), 
  (a + 1 = b) ∧                -- legs are consecutive whole numbers
  (a^2 + b^2 = 41^2) ∧         -- Pythagorean theorem with hypotenuse 41
  (a + b = 57) :=              -- sum of legs is 57
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_sum_l438_43810


namespace NUMINAMATH_CALUDE_notebook_problem_l438_43871

theorem notebook_problem (x : ℕ) (h : x^2 + 20 = (x + 1)^2 - 9) : x^2 + 20 = 216 := by
  sorry

end NUMINAMATH_CALUDE_notebook_problem_l438_43871


namespace NUMINAMATH_CALUDE_missed_bus_time_l438_43828

theorem missed_bus_time (usual_time : ℝ) (speed_ratio : ℝ) (h1 : usual_time = 16) (h2 : speed_ratio = 4/5) :
  (usual_time / speed_ratio) - usual_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_missed_bus_time_l438_43828


namespace NUMINAMATH_CALUDE_money_distribution_l438_43881

theorem money_distribution (p q r s : ℕ) : 
  p + q + r + s = 10000 →
  r = 2 * p →
  r = 3 * q →
  s = p + q →
  p = 1875 ∧ q = 1250 ∧ r = 3750 ∧ s = 3125 :=
by sorry

end NUMINAMATH_CALUDE_money_distribution_l438_43881
