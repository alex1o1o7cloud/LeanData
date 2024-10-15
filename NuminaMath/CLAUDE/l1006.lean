import Mathlib

namespace NUMINAMATH_CALUDE_incorrect_number_value_l1006_100630

theorem incorrect_number_value (n : ℕ) (initial_avg correct_avg correct_value : ℚ) 
  (h1 : n = 10)
  (h2 : initial_avg = 20)
  (h3 : correct_avg = 26)
  (h4 : correct_value = 86) :
  ∃ x : ℚ, n * correct_avg - n * initial_avg = correct_value - x ∧ x = 26 := by
sorry

end NUMINAMATH_CALUDE_incorrect_number_value_l1006_100630


namespace NUMINAMATH_CALUDE_five_objects_three_boxes_l1006_100610

/-- Number of ways to distribute n distinct objects into k distinct boxes,
    with each box containing at least one object -/
def distributionCount (n k : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to distribute 5 distinct objects into 3 distinct boxes,
    with each box containing at least one object, is equal to 150 -/
theorem five_objects_three_boxes : distributionCount 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_five_objects_three_boxes_l1006_100610


namespace NUMINAMATH_CALUDE_village_population_l1006_100624

theorem village_population (final_population : ℕ) : 
  final_population = 4860 → 
  ∃ (original_population : ℕ), 
    (original_population : ℝ) * 0.9 * 0.75 = final_population ∧ 
    original_population = 7200 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1006_100624


namespace NUMINAMATH_CALUDE_condition_for_proposition_l1006_100629

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

theorem condition_for_proposition (a : ℝ) :
  (∀ x ∈ A, x^2 - a ≤ 0) ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_condition_for_proposition_l1006_100629


namespace NUMINAMATH_CALUDE_complex_modulus_l1006_100695

theorem complex_modulus (z : ℂ) : z / (1 + Complex.I) = -3 * Complex.I → Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1006_100695


namespace NUMINAMATH_CALUDE_sqrt_difference_inequality_l1006_100618

theorem sqrt_difference_inequality (x : ℝ) (h : x ≥ 4) :
  Real.sqrt x - Real.sqrt (x - 1) ≥ 1 / x := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_inequality_l1006_100618


namespace NUMINAMATH_CALUDE_outer_circle_diameter_l1006_100617

/-- Proves that given an outer circle with diameter D and an inner circle with diameter 24,
    if 0.36 of the outer circle's surface is not covered by the inner circle,
    then the diameter of the outer circle is 30. -/
theorem outer_circle_diameter
  (D : ℝ) -- Diameter of the outer circle
  (h1 : D > 0) -- Diameter is positive
  (h2 : π * (D / 2)^2 - π * 12^2 = 0.36 * π * (D / 2)^2) -- Condition about uncovered area
  : D = 30 := by
  sorry


end NUMINAMATH_CALUDE_outer_circle_diameter_l1006_100617


namespace NUMINAMATH_CALUDE_frank_money_problem_l1006_100636

theorem frank_money_problem (initial_money : ℝ) : 
  initial_money > 0 →
  let remaining_after_groceries := initial_money - (1/5 * initial_money)
  let remaining_after_magazine := remaining_after_groceries - (1/4 * remaining_after_groceries)
  remaining_after_magazine = 360 →
  initial_money = 600 := by
sorry


end NUMINAMATH_CALUDE_frank_money_problem_l1006_100636


namespace NUMINAMATH_CALUDE_sudoku_like_puzzle_l1006_100680

/-- A 2x2 grid filled with numbers from 1 to 4 -/
def Grid := Fin 2 → Fin 2 → Fin 4

/-- Check if all numbers in a list are distinct -/
def all_distinct (l : List (Fin 4)) : Prop :=
  l.length = 4 ∧ l.Nodup

/-- Check if a grid satisfies Sudoku-like conditions -/
def valid_grid (g : Grid) : Prop :=
  (∀ i, all_distinct [g i 0, g i 1]) ∧  -- rows
  (∀ j, all_distinct [g 0 j, g 1 j]) ∧  -- columns
  all_distinct [g 0 0, g 0 1, g 1 0, g 1 1]  -- entire 2x2 grid

theorem sudoku_like_puzzle :
  ∀ g : Grid,
    valid_grid g →
    g 0 0 = 0 →  -- 1 in top-left (0-indexed)
    g 1 1 = 3 →  -- 4 in bottom-right (0-indexed)
    g 0 1 = 2    -- 3 in top-right (0-indexed)
    := by sorry

end NUMINAMATH_CALUDE_sudoku_like_puzzle_l1006_100680


namespace NUMINAMATH_CALUDE_pet_store_hamsters_l1006_100627

theorem pet_store_hamsters (rabbit_count : ℕ) (rabbit_ratio : ℕ) (hamster_ratio : ℕ) : 
  rabbit_count = 18 → 
  rabbit_ratio = 3 → 
  hamster_ratio = 4 → 
  (rabbit_count / rabbit_ratio) * hamster_ratio = 24 := by
sorry

end NUMINAMATH_CALUDE_pet_store_hamsters_l1006_100627


namespace NUMINAMATH_CALUDE_percentage_markup_proof_l1006_100644

def selling_price : ℚ := 8587
def cost_price : ℚ := 6925

theorem percentage_markup_proof :
  let markup := selling_price - cost_price
  let percentage_markup := (markup / cost_price) * 100
  ∃ ε > 0, abs (percentage_markup - 23.99) < ε := by
sorry

end NUMINAMATH_CALUDE_percentage_markup_proof_l1006_100644


namespace NUMINAMATH_CALUDE_discount_percentage_l1006_100657

theorem discount_percentage (initial_amount : ℝ) 
  (h1 : initial_amount = 500)
  (h2 : ∃ (needed_before_discount : ℝ), needed_before_discount = initial_amount + 2/5 * initial_amount)
  (h3 : ∃ (amount_still_needed : ℝ), amount_still_needed = 95) : 
  ∃ (discount_percentage : ℝ), discount_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_discount_percentage_l1006_100657


namespace NUMINAMATH_CALUDE_max_intersections_count_l1006_100631

/-- The number of points on the x-axis segment -/
def n : ℕ := 15

/-- The number of points on the y-axis segment -/
def m : ℕ := 10

/-- The maximum number of intersection points -/
def max_intersections : ℕ := n.choose 2 * m.choose 2

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersections_count :
  max_intersections = 4725 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_count_l1006_100631


namespace NUMINAMATH_CALUDE_sphere_only_circular_all_views_l1006_100643

-- Define the geometric shapes
inductive Shape
| Cuboid
| Cylinder
| Cone
| Sphere

-- Define the views
inductive View
| Front
| Left
| Top

-- Function to determine if a view of a shape is circular
def isCircularView (s : Shape) (v : View) : Prop :=
  match s, v with
  | Shape.Sphere, _ => True
  | Shape.Cylinder, View.Top => True
  | Shape.Cone, View.Top => True
  | _, _ => False

-- Theorem stating that only the Sphere has circular views in all three perspectives
theorem sphere_only_circular_all_views :
  ∀ s : Shape, (∀ v : View, isCircularView s v) ↔ s = Shape.Sphere := by
  sorry

end NUMINAMATH_CALUDE_sphere_only_circular_all_views_l1006_100643


namespace NUMINAMATH_CALUDE_scientific_notation_of_10374_billion_l1006_100615

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation with a specified number of significant figures -/
def toScientificNotation (x : ℝ) (sigFigs : ℕ) : ScientificNotation :=
  sorry

/-- The value to be converted (10,374 billion yuan) -/
def originalValue : ℝ := 10374 * 1000000000

/-- The number of significant figures to retain -/
def sigFigures : ℕ := 3

theorem scientific_notation_of_10374_billion :
  toScientificNotation originalValue sigFigures =
    ScientificNotation.mk 1.037 13 (by norm_num) :=
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_10374_billion_l1006_100615


namespace NUMINAMATH_CALUDE_exists_factorial_starting_with_2005_l1006_100619

theorem exists_factorial_starting_with_2005 : 
  ∃ (n : ℕ+), ∃ (k : ℕ), 2005 * 10^k ≤ n.val.factorial ∧ n.val.factorial < 2006 * 10^k :=
sorry

end NUMINAMATH_CALUDE_exists_factorial_starting_with_2005_l1006_100619


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l1006_100665

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℤ := n^2 - 10*n

/-- The nth term of the sequence -/
def a (n : ℕ) : ℤ := 2*n - 11

/-- Theorem stating that the given formula for a_n is correct -/
theorem sequence_formula_correct (n : ℕ) : 
  n ≥ 1 → S n - S (n-1) = a n := by sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l1006_100665


namespace NUMINAMATH_CALUDE_wine_purchase_problem_l1006_100693

theorem wine_purchase_problem :
  ∃ (x y n m : ℕ), 
    5 * x + 8 * y = n ^ 2 ∧
    n ^ 2 + 60 = m ^ 2 ∧
    x + y = m :=
by sorry

end NUMINAMATH_CALUDE_wine_purchase_problem_l1006_100693


namespace NUMINAMATH_CALUDE_delta_equation_solution_l1006_100648

-- Define the Δ operation
def delta (a b : ℝ) : ℝ := a * b + a + b

-- Theorem statement
theorem delta_equation_solution :
  ∀ p : ℝ, delta p 3 = 39 → p = 9 := by
  sorry

end NUMINAMATH_CALUDE_delta_equation_solution_l1006_100648


namespace NUMINAMATH_CALUDE_select_at_most_one_ab_l1006_100666

def students : ℕ := 5
def selected : ℕ := 3
def competitions : ℕ := 3

def ways_to_select (n k : ℕ) : ℕ := Nat.choose n k

def ways_to_assign (n : ℕ) : ℕ := Nat.factorial n

def select_with_at_most_one_specific (total specific selected : ℕ) : ℕ :=
  -- Case 1: One specific student selected
  2 * (ways_to_select (total - 2) (selected - 1) * ways_to_assign competitions) +
  -- Case 2: Neither specific student selected
  (ways_to_select (total - 2) selected * ways_to_assign competitions)

theorem select_at_most_one_ab :
  select_with_at_most_one_specific students 2 selected = 42 := by
  sorry

end NUMINAMATH_CALUDE_select_at_most_one_ab_l1006_100666


namespace NUMINAMATH_CALUDE_min_value_expression_l1006_100605

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1/a + a/(b^2) + b ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1006_100605


namespace NUMINAMATH_CALUDE_triangle_sine_inequality_l1006_100634

theorem triangle_sine_inequality (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin A ^ 2 + Real.sin B ^ 2 + Real.sin C ^ 2 ≤ 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_inequality_l1006_100634


namespace NUMINAMATH_CALUDE_gigi_banana_consumption_l1006_100668

theorem gigi_banana_consumption (total : ℕ) (days : ℕ) (increase : ℕ) (last_day : ℕ) :
  total = 150 →
  days = 7 →
  increase = 4 →
  (∃ first : ℚ, (days : ℚ) / 2 * (2 * first + (days - 1) * increase) = total) →
  last_day = (days - 1) * increase + (total * 2 / days - (days - 1) * increase) / 2 →
  last_day = 33 :=
by sorry

end NUMINAMATH_CALUDE_gigi_banana_consumption_l1006_100668


namespace NUMINAMATH_CALUDE_largest_common_measure_l1006_100622

theorem largest_common_measure (segment1 segment2 : ℕ) 
  (h1 : segment1 = 15) (h2 : segment2 = 12) : 
  ∃ (m : ℕ), m > 0 ∧ m ∣ segment1 ∧ m ∣ segment2 ∧ 
  ∀ (n : ℕ), n > m → (n ∣ segment1 ∧ n ∣ segment2) → False :=
by sorry

end NUMINAMATH_CALUDE_largest_common_measure_l1006_100622


namespace NUMINAMATH_CALUDE_rook_placement_on_colored_board_l1006_100625

theorem rook_placement_on_colored_board :
  let n : ℕ := 8  -- number of rooks and rows/columns
  let m : ℕ := 32  -- number of colors
  let total_arrangements : ℕ := n.factorial
  let problematic_arrangements : ℕ := m * (n - 2).factorial
  total_arrangements > problematic_arrangements :=
by sorry

end NUMINAMATH_CALUDE_rook_placement_on_colored_board_l1006_100625


namespace NUMINAMATH_CALUDE_average_calls_proof_l1006_100664

def average_calls (mon tue wed thu fri : ℕ) : ℚ :=
  (mon + tue + wed + thu + fri : ℚ) / 5

theorem average_calls_proof (mon tue wed thu fri : ℕ) :
  average_calls mon tue wed thu fri = (mon + tue + wed + thu + fri : ℚ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_average_calls_proof_l1006_100664


namespace NUMINAMATH_CALUDE_problem_solution_l1006_100621

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 30) :
  x^2 + Real.sqrt (x^4 - 16) + 1 / (x^2 + Real.sqrt (x^4 - 16)) = 52441/900 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1006_100621


namespace NUMINAMATH_CALUDE_inequality_proof_l1006_100672

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1006_100672


namespace NUMINAMATH_CALUDE_points_scored_in_quarter_l1006_100641

/-- Calculates the total points scored in a basketball quarter -/
def total_points_scored (two_point_shots : ℕ) (three_point_shots : ℕ) : ℕ :=
  2 * two_point_shots + 3 * three_point_shots

/-- Proves that given four 2-point shots and two 3-point shots, the total points scored is 14 -/
theorem points_scored_in_quarter : total_points_scored 4 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_points_scored_in_quarter_l1006_100641


namespace NUMINAMATH_CALUDE_initial_players_correct_l1006_100692

/-- The initial number of players in a video game -/
def initial_players : ℕ := 8

/-- The number of players who quit the game -/
def players_quit : ℕ := 3

/-- The number of lives each remaining player has -/
def lives_per_player : ℕ := 3

/-- The total number of lives after some players quit -/
def total_lives : ℕ := 15

/-- Theorem stating that the initial number of players is correct -/
theorem initial_players_correct : 
  lives_per_player * (initial_players - players_quit) = total_lives :=
by sorry

end NUMINAMATH_CALUDE_initial_players_correct_l1006_100692


namespace NUMINAMATH_CALUDE_triangle_value_l1006_100616

theorem triangle_value (p : ℚ) (triangle : ℚ) 
  (eq1 : triangle * p + p = 72)
  (eq2 : (triangle * p + p) + p = 111) :
  triangle = 11 / 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_value_l1006_100616


namespace NUMINAMATH_CALUDE_B_equals_l1006_100613

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}

-- Define sets A and B
variable (A B : Set Nat)

-- State the conditions
axiom union_eq : A ∪ B = U
axiom intersection_eq : A ∩ (U \ B) = {2, 4, 6}

-- Theorem to prove
theorem B_equals : B = {1, 3, 5, 7} := by sorry

end NUMINAMATH_CALUDE_B_equals_l1006_100613


namespace NUMINAMATH_CALUDE_homework_scenarios_count_l1006_100676

/-- The number of subjects available for homework -/
def num_subjects : ℕ := 4

/-- The number of students doing homework -/
def num_students : ℕ := 3

/-- The number of possible scenarios for homework assignment -/
def num_scenarios : ℕ := num_subjects ^ num_students

/-- Theorem stating that the number of possible scenarios is 64 -/
theorem homework_scenarios_count : num_scenarios = 64 := by
  sorry

end NUMINAMATH_CALUDE_homework_scenarios_count_l1006_100676


namespace NUMINAMATH_CALUDE_system_solution_l1006_100663

theorem system_solution (x y : ℝ) : 
  x + 2*y = 8 → 2*x + y = -5 → x + y = 1 := by sorry

end NUMINAMATH_CALUDE_system_solution_l1006_100663


namespace NUMINAMATH_CALUDE_right_triangle_angle_sum_l1006_100685

theorem right_triangle_angle_sum (A B C : Real) : 
  (A + B + C = 180) → (C = 90) → (B = 55) → (A = 35) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_sum_l1006_100685


namespace NUMINAMATH_CALUDE_eugene_pencils_l1006_100607

theorem eugene_pencils (x : ℕ) (h1 : x + 6 = 57) : x = 51 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l1006_100607


namespace NUMINAMATH_CALUDE_spiral_stripe_length_l1006_100667

/-- The length of a spiral stripe on a cylindrical water tower -/
theorem spiral_stripe_length 
  (circumference height : ℝ) 
  (h_circumference : circumference = 18) 
  (h_height : height = 24) :
  Real.sqrt (circumference^2 + height^2) = 30 := by sorry

end NUMINAMATH_CALUDE_spiral_stripe_length_l1006_100667


namespace NUMINAMATH_CALUDE_exact_pairing_l1006_100656

/-- The number of workers processing large gears to match pairs exactly -/
def workers_large_gears : ℕ := 18

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 34

/-- The number of large gears processed by one worker per day -/
def large_gears_per_worker : ℕ := 20

/-- The number of small gears processed by one worker per day -/
def small_gears_per_worker : ℕ := 15

/-- The number of large gears in a pair -/
def large_gears_per_pair : ℕ := 3

/-- The number of small gears in a pair -/
def small_gears_per_pair : ℕ := 2

theorem exact_pairing :
  workers_large_gears * large_gears_per_worker * small_gears_per_pair =
  (total_workers - workers_large_gears) * small_gears_per_worker * large_gears_per_pair :=
by sorry

end NUMINAMATH_CALUDE_exact_pairing_l1006_100656


namespace NUMINAMATH_CALUDE_complex_sum_reciprocal_magnitude_l1006_100600

theorem complex_sum_reciprocal_magnitude (z w : ℂ) :
  Complex.abs z = 2 →
  Complex.abs w = 4 →
  Complex.abs (z + w) = 3 →
  ∃ θ : ℝ, θ = Real.pi / 3 ∧ z * Complex.exp (Complex.I * θ) = w →
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_complex_sum_reciprocal_magnitude_l1006_100600


namespace NUMINAMATH_CALUDE_complement_of_A_wrt_I_l1006_100633

def I : Finset ℕ := {1, 2, 3, 4, 5, 6}
def A : Finset ℕ := {1, 3, 4}

theorem complement_of_A_wrt_I :
  I \ A = {2, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_wrt_I_l1006_100633


namespace NUMINAMATH_CALUDE_trihedral_angle_inequalities_l1006_100612

structure TrihedralAngle where
  SA : Real
  SB : Real
  SC : Real
  α : Real
  β : Real
  γ : Real
  ASB : Real
  BSC : Real
  CSA : Real

def is_acute_dihedral (t : TrihedralAngle) : Prop := sorry

theorem trihedral_angle_inequalities (t : TrihedralAngle) :
  t.α + t.β + t.γ ≤ t.ASB + t.BSC + t.CSA ∧
  (is_acute_dihedral t → t.α + t.β + t.γ ≥ (t.ASB + t.BSC + t.CSA) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trihedral_angle_inequalities_l1006_100612


namespace NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1006_100698

theorem x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1/x^4) : x^2 + 1/x^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_squared_plus_reciprocal_l1006_100698


namespace NUMINAMATH_CALUDE_product_inequality_l1006_100661

theorem product_inequality (a b m : ℕ) : 
  (a + b = 40 → a * b ≤ 20^2) ∧ 
  (a + b = m → a * b ≤ (m / 2)^2) :=
by sorry

end NUMINAMATH_CALUDE_product_inequality_l1006_100661


namespace NUMINAMATH_CALUDE_silver_coin_percentage_is_31_5_percent_l1006_100658

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  bead_percentage : ℝ
  gold_coin_percentage : ℝ

/-- Calculates the percentage of silver coins in the urn --/
def silver_coin_percentage (urn : UrnComposition) : ℝ :=
  (1 - urn.bead_percentage) * (1 - urn.gold_coin_percentage)

/-- Theorem stating that the percentage of silver coins is 31.5% --/
theorem silver_coin_percentage_is_31_5_percent :
  let urn : UrnComposition := ⟨0.3, 0.55⟩
  silver_coin_percentage urn = 0.315 := by sorry

end NUMINAMATH_CALUDE_silver_coin_percentage_is_31_5_percent_l1006_100658


namespace NUMINAMATH_CALUDE_triangle_obtuse_l1006_100637

theorem triangle_obtuse (A B C : Real) (hABC : A + B + C = π) 
  (h : Real.sin A ^ 2 + Real.sin B ^ 2 < Real.sin C ^ 2) : 
  π / 2 < C ∧ C < π :=
sorry

end NUMINAMATH_CALUDE_triangle_obtuse_l1006_100637


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1006_100609

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 - 6*x + 7 = 0 ↔ (x - 3)^2 = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1006_100609


namespace NUMINAMATH_CALUDE_digit_difference_750_150_l1006_100690

/-- The number of digits in the base-2 representation of a positive integer -/
def numDigitsBase2 (n : ℕ+) : ℕ :=
  Nat.log2 n + 1

/-- The difference in the number of digits between 750 and 150 in base 2 -/
theorem digit_difference_750_150 : numDigitsBase2 750 - numDigitsBase2 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_750_150_l1006_100690


namespace NUMINAMATH_CALUDE_combined_mixture_ratio_l1006_100653

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Combines two mixtures -/
def combine_mixtures (m1 m2 : Mixture) : Mixture :=
  { milk := m1.milk + m2.milk,
    water := m1.water + m2.water }

/-- Calculates the ratio of milk to water in a mixture -/
def milk_water_ratio (m : Mixture) : ℚ × ℚ :=
  (m.milk, m.water)

theorem combined_mixture_ratio :
  let m1 : Mixture := { milk := 4, water := 1 }
  let m2 : Mixture := { milk := 7, water := 3 }
  let combined := combine_mixtures m1 m2
  milk_water_ratio combined = (11, 4) := by
  sorry

end NUMINAMATH_CALUDE_combined_mixture_ratio_l1006_100653


namespace NUMINAMATH_CALUDE_participants_2003_l1006_100691

def initial_participants : ℕ := 500
def increase_2001 : ℚ := 1.3
def increase_2002 : ℚ := 1.4
def increase_2003 : ℚ := 1.5

theorem participants_2003 : 
  ⌊(((initial_participants : ℚ) * increase_2001) * increase_2002) * increase_2003⌋ = 1365 := by
  sorry

end NUMINAMATH_CALUDE_participants_2003_l1006_100691


namespace NUMINAMATH_CALUDE_range_of_a_l1006_100638

def p (a : ℝ) : Prop := a * (1 - a) > 0

def q (a : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 + (2*a - 3)*x₁ + 1 = 0 ∧ 
  x₂^2 + (2*a - 3)*x₂ + 1 = 0

def S : Set ℝ := {a | a ≤ 0 ∨ (1/2 ≤ a ∧ a < 1) ∨ a > 5/2}

theorem range_of_a : {a : ℝ | (p a ∨ q a) ∧ ¬(p a ∧ q a)} = S := by sorry

end NUMINAMATH_CALUDE_range_of_a_l1006_100638


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l1006_100652

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Two nested convex polygons Q1 and Q2 -/
structure NestedPolygons where
  Q1 : ConvexPolygon
  Q2 : ConvexPolygon
  m : ℕ
  h_m_ge_3 : m ≥ 3
  h_Q1_sides : Q1.sides = m
  h_Q2_sides : Q2.sides = 2 * m
  h_nested : Bool
  h_no_shared_segment : Bool
  h_both_convex : Q1.convex ∧ Q2.convex

/-- The maximum number of intersections between two nested convex polygons -/
def max_intersections (np : NestedPolygons) : ℕ := 2 * np.m^2

/-- Theorem stating the maximum number of intersections -/
theorem max_intersections_theorem (np : NestedPolygons) :
  max_intersections np = 2 * np.m^2 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l1006_100652


namespace NUMINAMATH_CALUDE_class_vision_median_l1006_100603

/-- Represents the vision data for a class of students -/
structure VisionData where
  values : List Float
  frequencies : List Nat
  total_students : Nat

/-- Calculates the median of the vision data -/
def median (data : VisionData) : Float :=
  sorry

/-- The specific vision data for the class -/
def class_vision_data : VisionData :=
  { values := [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0],
    frequencies := [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5],
    total_students := 39 }

/-- Theorem stating that the median of the class vision data is 4.6 -/
theorem class_vision_median :
  median class_vision_data = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_class_vision_median_l1006_100603


namespace NUMINAMATH_CALUDE_moon_speed_conversion_l1006_100673

/-- The speed of the moon in kilometers per second -/
def moon_speed_km_per_sec : ℝ := 1.04

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The speed of the moon in kilometers per hour -/
def moon_speed_km_per_hour : ℝ := moon_speed_km_per_sec * seconds_per_hour

theorem moon_speed_conversion : 
  moon_speed_km_per_hour = 3744 := by sorry

end NUMINAMATH_CALUDE_moon_speed_conversion_l1006_100673


namespace NUMINAMATH_CALUDE_koala_fiber_intake_l1006_100682

/-- Given that a koala absorbs 30% of the fiber it eats and absorbed 12 ounces of fiber in one day,
    prove that the total amount of fiber eaten by the koala that day was 40 ounces. -/
theorem koala_fiber_intake (absorption_rate : ℝ) (absorbed_amount : ℝ) (total_amount : ℝ) 
    (h1 : absorption_rate = 0.30)
    (h2 : absorbed_amount = 12)
    (h3 : absorbed_amount = absorption_rate * total_amount) :
  total_amount = 40 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_intake_l1006_100682


namespace NUMINAMATH_CALUDE_train_crossing_bridge_time_train_crossing_bridge_time_is_35_seconds_l1006_100659

/-- The time required for a train to cross a bridge -/
theorem train_crossing_bridge_time 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 200) 
  (h2 : bridge_length = 150) 
  (h3 : train_speed_kmph = 36) : ℝ :=
let train_speed_mps := train_speed_kmph * (5/18)
let total_distance := train_length + bridge_length
let time := total_distance / train_speed_mps
35

theorem train_crossing_bridge_time_is_35_seconds 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (h1 : train_length = 200) 
  (h2 : bridge_length = 150) 
  (h3 : train_speed_kmph = 36) : 
  train_crossing_bridge_time train_length bridge_length train_speed_kmph h1 h2 h3 = 35 := by
sorry

end NUMINAMATH_CALUDE_train_crossing_bridge_time_train_crossing_bridge_time_is_35_seconds_l1006_100659


namespace NUMINAMATH_CALUDE_work_completion_time_l1006_100604

theorem work_completion_time 
  (a_rate : ℚ) 
  (b_rate : ℚ) 
  (joint_work_days : ℕ) 
  (h1 : a_rate = 1 / 5) 
  (h2 : b_rate = 1 / 15) 
  (h3 : joint_work_days = 2) : 
  ℕ :=
by
  sorry

#check work_completion_time

end NUMINAMATH_CALUDE_work_completion_time_l1006_100604


namespace NUMINAMATH_CALUDE_equation_solutions_l1006_100669

-- Define the equation
def equation (x : ℂ) : Prop :=
  (x - 4)^4 + (x - 6)^4 = 16

-- Define the set of solutions
def solution_set : Set ℂ :=
  {5 + Complex.I * Real.sqrt 7, 5 - Complex.I * Real.sqrt 7, 6, 4}

-- Theorem statement
theorem equation_solutions :
  ∀ x : ℂ, equation x ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1006_100669


namespace NUMINAMATH_CALUDE_sin_600_degrees_l1006_100678

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l1006_100678


namespace NUMINAMATH_CALUDE_abc_inequality_l1006_100628

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1/9 ∧ a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2 * Real.sqrt (abc)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1006_100628


namespace NUMINAMATH_CALUDE_smallest_draw_for_red_apple_probability_l1006_100689

theorem smallest_draw_for_red_apple_probability (total_apples : Nat) (red_apples : Nat) 
  (h1 : total_apples = 15) (h2 : red_apples = 9) : 
  (∃ n : Nat, n = 5 ∧ 
    ∀ k : Nat, k < n → (red_apples - k : Rat) / (total_apples - k) ≥ 1/2 ∧
    (red_apples - n : Rat) / (total_apples - n) < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_draw_for_red_apple_probability_l1006_100689


namespace NUMINAMATH_CALUDE_largest_n_for_unique_k_l1006_100602

theorem largest_n_for_unique_k : 
  ∀ n : ℕ, n > 112 → 
  ¬(∃! k : ℤ, (7 : ℚ)/16 < (n : ℚ)/(n + k) ∧ (n : ℚ)/(n + k) < 8/17) ∧ 
  (∃! k : ℤ, (7 : ℚ)/16 < (112 : ℚ)/(112 + k) ∧ (112 : ℚ)/(112 + k) < 8/17) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_unique_k_l1006_100602


namespace NUMINAMATH_CALUDE_test_score_calculation_l1006_100620

theorem test_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (incorrect_penalty : ℕ) (total_score : ℕ) :
  total_questions = 30 →
  correct_answers = 19 →
  incorrect_penalty = 5 →
  total_score = 325 →
  ∃ (points_per_correct : ℕ),
    points_per_correct * correct_answers - incorrect_penalty * (total_questions - correct_answers) = total_score ∧
    points_per_correct = 20 :=
by sorry

end NUMINAMATH_CALUDE_test_score_calculation_l1006_100620


namespace NUMINAMATH_CALUDE_unsold_books_l1006_100674

theorem unsold_books (total_amount : ℝ) (price_per_book : ℝ) (fraction_sold : ℝ) :
  total_amount = 500 ∧
  price_per_book = 5 ∧
  fraction_sold = 2/3 →
  (1 - fraction_sold) * (total_amount / (price_per_book * fraction_sold)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_unsold_books_l1006_100674


namespace NUMINAMATH_CALUDE_blue_lipstick_count_l1006_100662

theorem blue_lipstick_count (total_students : ℕ) 
  (h1 : total_students = 200)
  (h2 : ∃ colored_lipstick : ℕ, colored_lipstick = total_students / 2)
  (h3 : ∃ red_lipstick : ℕ, red_lipstick = (total_students / 2) / 4)
  (h4 : ∃ blue_lipstick : ℕ, blue_lipstick = ((total_students / 2) / 4) / 5) :
  ∃ blue_lipstick : ℕ, blue_lipstick = 5 := by
  sorry

end NUMINAMATH_CALUDE_blue_lipstick_count_l1006_100662


namespace NUMINAMATH_CALUDE_A_n_is_integer_l1006_100687

theorem A_n_is_integer (a b n : ℕ) (h1 : a > b) (h2 : b > 0) 
  (θ : Real) (h3 : 0 < θ) (h4 : θ < Real.pi / 2) 
  (h5 : Real.sin θ = (2 * a * b : ℝ) / ((a^2 + b^2) : ℝ)) :
  ∃ k : ℤ, ((a^2 + b^2 : ℕ)^n : ℝ) * Real.sin (n * θ) = k := by
  sorry

#check A_n_is_integer

end NUMINAMATH_CALUDE_A_n_is_integer_l1006_100687


namespace NUMINAMATH_CALUDE_line_through_points_l1006_100679

/-- A line passing through two points (3,1) and (7,13) has equation y = ax + b. This theorem proves that a - b = 11. -/
theorem line_through_points (a b : ℝ) : 
  (1 = a * 3 + b) → (13 = a * 7 + b) → a - b = 11 := by sorry

end NUMINAMATH_CALUDE_line_through_points_l1006_100679


namespace NUMINAMATH_CALUDE_problem_solution_l1006_100649

theorem problem_solution (a b c : ℝ) : 
  b = 15 → 
  c = 3 → 
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1) → 
  a * b * c = 3 → 
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1006_100649


namespace NUMINAMATH_CALUDE_linear_function_characterization_l1006_100626

/-- A function satisfying the given property for a fixed α -/
def SatisfiesProperty (α : ℝ) (f : ℕ+ → ℝ) : Prop :=
  ∀ (k m : ℕ+), α * m.val ≤ k.val ∧ k.val ≤ (α + 1) * m.val → f (k + m) = f k + f m

/-- The main theorem stating that any function satisfying the property is linear -/
theorem linear_function_characterization (α : ℝ) (hα : α > 0) (f : ℕ+ → ℝ) 
  (hf : SatisfiesProperty α f) : 
  ∃ (D : ℝ), ∀ (n : ℕ+), f n = D * n.val := by
  sorry

end NUMINAMATH_CALUDE_linear_function_characterization_l1006_100626


namespace NUMINAMATH_CALUDE_pond_to_field_area_ratio_l1006_100660

/-- Represents a rectangular field with a square pond -/
structure FieldWithPond where
  field_length : ℝ
  field_width : ℝ
  pond_side : ℝ
  length_is_double_width : field_length = 2 * field_width
  length_is_96 : field_length = 96
  pond_side_is_8 : pond_side = 8

/-- The ratio of the pond area to the field area is 1:72 -/
theorem pond_to_field_area_ratio (f : FieldWithPond) :
  (f.pond_side^2) / (f.field_length * f.field_width) = 1 / 72 := by
  sorry

end NUMINAMATH_CALUDE_pond_to_field_area_ratio_l1006_100660


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1006_100684

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 0 = 3 →
  a 1 = 10 →
  a 2 = 17 →
  a 5 = 32 →
  a 3 + a 4 = 55 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1006_100684


namespace NUMINAMATH_CALUDE_jump_frequency_proof_l1006_100635

def jump_data : List Nat := [50, 63, 77, 83, 87, 88, 89, 91, 93, 100, 102, 111, 117, 121, 130, 133, 146, 158, 177, 188]

def in_range (n : Nat) : Bool := 90 ≤ n ∧ n ≤ 110

def count_in_range (data : List Nat) : Nat :=
  data.filter in_range |>.length

theorem jump_frequency_proof :
  (count_in_range jump_data : Rat) / jump_data.length = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_jump_frequency_proof_l1006_100635


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l1006_100601

theorem constant_ratio_problem (k : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  (∀ x y, (4 * x + 3) / (2 * y - 5) = k) →
  x₁ = 1 →
  y₁ = 5 →
  y₂ = 10 →
  x₂ = 9 / 2 :=
by sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l1006_100601


namespace NUMINAMATH_CALUDE_square_last_digit_six_implies_second_last_odd_l1006_100696

theorem square_last_digit_six_implies_second_last_odd (n : ℕ) : 
  n^2 % 100 ≥ 6 ∧ n^2 % 100 < 16 → (n^2 / 10) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_last_digit_six_implies_second_last_odd_l1006_100696


namespace NUMINAMATH_CALUDE_largest_angle_in_convex_pentagon_l1006_100681

/-- The largest angle in a convex pentagon with specific angle measures -/
theorem largest_angle_in_convex_pentagon : 
  ∀ x : ℝ,
  (x + 2) + (2*x + 3) + (3*x - 4) + (4*x + 5) + (5*x - 6) = 540 →
  max (x + 2) (max (2*x + 3) (max (3*x - 4) (max (4*x + 5) (5*x - 6)))) = 174 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_convex_pentagon_l1006_100681


namespace NUMINAMATH_CALUDE_same_solution_implies_c_value_l1006_100675

theorem same_solution_implies_c_value (y : ℝ) (c : ℝ) : 
  (3 * y - 9 = 0) ∧ (c * y + 15 = 3) → c = -4 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_implies_c_value_l1006_100675


namespace NUMINAMATH_CALUDE_triangle_inequality_l1006_100614

/-- Theorem: In a triangle with two sides of lengths 3 and 8, the third side is between 5 and 11 -/
theorem triangle_inequality (a b c : ℝ) : a = 3 ∧ b = 8 → 5 < c ∧ c < 11 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1006_100614


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l1006_100611

theorem contractor_daily_wage
  (total_days : ℕ)
  (absence_fine : ℚ)
  (total_payment : ℚ)
  (absent_days : ℕ)
  (h1 : total_days = 30)
  (h2 : absence_fine = 7.5)
  (h3 : total_payment = 425)
  (h4 : absent_days = 10)
  : ∃ (daily_wage : ℚ), daily_wage = 25 ∧
    (total_days - absent_days) * daily_wage - absent_days * absence_fine = total_payment :=
by
  sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l1006_100611


namespace NUMINAMATH_CALUDE_parabola_decreasing_condition_l1006_100608

/-- Represents a parabola of the form y = -5(x + m)² - 3 -/
def Parabola (m : ℝ) : ℝ → ℝ := λ x ↦ -5 * (x + m)^2 - 3

/-- States that the parabola is decreasing for x ≥ 2 -/
def IsDecreasingForXGeq2 (m : ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ 2 → x₂ ≥ 2 → x₁ < x₂ → Parabola m x₁ > Parabola m x₂

theorem parabola_decreasing_condition (m : ℝ) :
  IsDecreasingForXGeq2 m → m ≥ -2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_decreasing_condition_l1006_100608


namespace NUMINAMATH_CALUDE_quilt_shaded_fraction_l1006_100642

/-- Represents a square quilt block -/
structure QuiltBlock where
  size : ℕ
  diagonal_shaded : Bool

/-- Calculate the shaded fraction of a quilt block -/
def shaded_fraction (q : QuiltBlock) : ℚ :=
  if q.diagonal_shaded && q.size = 3 then 1/6 else 0

/-- Theorem: The shaded fraction of a 3x3 quilt block with half-shaded diagonal squares is 1/6 -/
theorem quilt_shaded_fraction :
  ∀ (q : QuiltBlock), q.size = 3 → q.diagonal_shaded → shaded_fraction q = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_quilt_shaded_fraction_l1006_100642


namespace NUMINAMATH_CALUDE_max_quarters_l1006_100677

theorem max_quarters (total : ℚ) (q : ℕ) : 
  total = 4.55 →
  (0.25 * q + 0.05 * q + 0.1 * (q / 2 : ℚ) = total) →
  (∀ n : ℕ, (0.25 * n + 0.05 * n + 0.1 * (n / 2 : ℚ) ≤ total)) →
  q = 13 :=
by sorry

end NUMINAMATH_CALUDE_max_quarters_l1006_100677


namespace NUMINAMATH_CALUDE_average_daily_high_temperature_l1006_100639

def daily_highs : List ℝ := [49, 62, 58, 57, 46]

theorem average_daily_high_temperature :
  (daily_highs.sum / daily_highs.length : ℝ) = 54.4 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_high_temperature_l1006_100639


namespace NUMINAMATH_CALUDE_max_a_value_l1006_100694

noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (x + 1)

noncomputable def g (a x : ℝ) : ℝ := Real.log (a * x^2 - 3 * x + 1)

theorem max_a_value :
  ∃ (a : ℝ), ∀ (a' : ℝ),
    (∀ (x₁ : ℝ), x₁ ≥ 0 → ∃ (x₂ : ℝ), f x₁ = g a' x₂) →
    a' ≤ a ∧
    (∀ (x₁ : ℝ), x₁ ≥ 0 → ∃ (x₂ : ℝ), f x₁ = g a x₂) ∧
    a = 9/4 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1006_100694


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l1006_100647

theorem largest_three_digit_congruence :
  ∃ (n : ℕ), n = 993 ∧ 
  (∀ m : ℕ, m ≤ 999 → 45 * m ≡ 270 [MOD 315] → m ≤ n) ∧
  45 * n ≡ 270 [MOD 315] :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l1006_100647


namespace NUMINAMATH_CALUDE_jenny_walking_distance_l1006_100671

theorem jenny_walking_distance (ran_distance : Real) (extra_ran_distance : Real) :
  ran_distance = 0.6 →
  extra_ran_distance = 0.2 →
  ∃ walked_distance : Real,
    walked_distance + extra_ran_distance = ran_distance ∧
    walked_distance = 0.4 :=
by sorry

end NUMINAMATH_CALUDE_jenny_walking_distance_l1006_100671


namespace NUMINAMATH_CALUDE_median_triangle_area_l1006_100686

-- Define a triangle
structure Triangle :=
  (a b c : ℝ)
  (ma mb mc : ℝ)
  (area : ℝ)

-- Define the triangle formed by medians
def MedianTriangle (t : Triangle) : Triangle :=
  { a := t.ma,
    b := t.mb,
    c := t.mc,
    ma := 0,  -- We don't need these values for the median triangle
    mb := 0,
    mc := 0,
    area := 0 }  -- We'll prove this is 3/4 * t.area

-- Theorem statement
theorem median_triangle_area (t : Triangle) :
  (MedianTriangle t).area = 3/4 * t.area :=
sorry

end NUMINAMATH_CALUDE_median_triangle_area_l1006_100686


namespace NUMINAMATH_CALUDE_min_value_theorem_l1006_100632

theorem min_value_theorem (x y : ℝ) (h : 2 * x^2 + 3 * x * y + 2 * y^2 = 1) :
  ∃ (min_val : ℝ), min_val = -9/8 ∧ ∀ (a b : ℝ), 2 * a^2 + 3 * a * b + 2 * b^2 = 1 →
    x + y + x * y ≥ min_val ∧ a + b + a * b ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1006_100632


namespace NUMINAMATH_CALUDE_distance_circle_center_to_line_l1006_100651

theorem distance_circle_center_to_line :
  let line_eq : ℝ → ℝ → Prop := λ x y => x + y = 6
  let circle_eq : ℝ → ℝ → Prop := λ x y => x^2 + (y - 2)^2 = 4
  let circle_center : ℝ × ℝ := (0, 2)
  ∃ d : ℝ, d = 2 * Real.sqrt 2 ∧
    d = (|0 + 2 - 6|) / Real.sqrt ((1:ℝ)^2 + 1^2) :=
by sorry

end NUMINAMATH_CALUDE_distance_circle_center_to_line_l1006_100651


namespace NUMINAMATH_CALUDE_overtime_pay_rate_l1006_100646

def regular_pay_rate : ℝ := 10
def regular_hours : ℝ := 40
def total_hours : ℝ := 60
def total_earnings : ℝ := 700

theorem overtime_pay_rate :
  ∃ (overtime_rate : ℝ),
    regular_pay_rate * regular_hours +
    overtime_rate * (total_hours - regular_hours) =
    total_earnings ∧
    overtime_rate = 15 :=
by sorry

end NUMINAMATH_CALUDE_overtime_pay_rate_l1006_100646


namespace NUMINAMATH_CALUDE_regular_hexagon_side_length_l1006_100655

/-- The length of a side in a regular hexagon given the distance between opposite sides -/
theorem regular_hexagon_side_length (d : ℝ) (h : d = 20) : 
  let s := d * 2 / Real.sqrt 3
  s = 40 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_side_length_l1006_100655


namespace NUMINAMATH_CALUDE_inequality_statement_not_always_true_l1006_100654

theorem inequality_statement_not_always_true :
  ¬ (∀ a b c : ℝ, a < b → a * c^2 < b * c^2) :=
sorry

end NUMINAMATH_CALUDE_inequality_statement_not_always_true_l1006_100654


namespace NUMINAMATH_CALUDE_k_range_unique_triangle_l1006_100697

/-- Represents an acute triangle ABC with specific properties -/
structure AcuteTriangle where
  /-- Side length AB -/
  k : ℝ
  /-- Angle C in radians -/
  angleC : ℝ
  /-- Angle A is 60 degrees (π/3 radians) -/
  angleA_eq : angleA = π/3
  /-- Side length BC is 6 -/
  bc_eq : bc = 6
  /-- Triangle is acute -/
  acute : 0 < angleC ∧ angleC < π/2
  /-- Sine rule holds -/
  sine_rule : k = 4 * Real.sqrt 3 * Real.sin angleC

/-- The range of k for the specific acute triangle -/
theorem k_range (t : AcuteTriangle) : 2 * Real.sqrt 3 < t.k ∧ t.k < 4 * Real.sqrt 3 := by
  sorry

/-- There exists only one such triangle -/
theorem unique_triangle : ∃! t : AcuteTriangle, True := by
  sorry

end NUMINAMATH_CALUDE_k_range_unique_triangle_l1006_100697


namespace NUMINAMATH_CALUDE_river_crossing_problem_l1006_100606

/-- The minimum number of trips required to transport a group of people across a river -/
def min_trips (total_people : ℕ) (boat_capacity : ℕ) (boatman_required : Bool) : ℕ :=
  let effective_capacity := if boatman_required then boat_capacity - 1 else boat_capacity
  (total_people + effective_capacity - 1) / effective_capacity

theorem river_crossing_problem :
  min_trips 14 5 true = 4 := by
  sorry

end NUMINAMATH_CALUDE_river_crossing_problem_l1006_100606


namespace NUMINAMATH_CALUDE_max_wrappers_l1006_100670

theorem max_wrappers (andy_wrappers : ℕ) (total_wrappers : ℕ) (max_wrappers : ℕ) : 
  andy_wrappers = 34 → total_wrappers = 49 → max_wrappers = total_wrappers - andy_wrappers →
  max_wrappers = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_max_wrappers_l1006_100670


namespace NUMINAMATH_CALUDE_valid_parameterization_l1006_100640

/-- Defines a line in 2D space --/
structure Line2D where
  slope : ℚ
  intercept : ℚ

/-- Defines a vector parameterization of a line --/
structure VectorParam where
  x₀ : ℚ
  y₀ : ℚ
  a : ℚ
  b : ℚ

/-- Checks if a vector parameterization is valid for a given line --/
def isValidParam (l : Line2D) (p : VectorParam) : Prop :=
  ∃ (k : ℚ), p.a = k * 5 ∧ p.b = k * 7 ∧ 
  p.y₀ = l.slope * p.x₀ + l.intercept

/-- The main theorem to prove --/
theorem valid_parameterization (l : Line2D) (p : VectorParam) :
  l.slope = 7/5 ∧ l.intercept = -23/5 →
  isValidParam l p ↔ 
    (p.x₀ = 5 ∧ p.y₀ = 2 ∧ p.a = -5 ∧ p.b = -7) ∨
    (p.x₀ = 23 ∧ p.y₀ = 7 ∧ p.a = 10 ∧ p.b = 14) ∨
    (p.x₀ = 3 ∧ p.y₀ = -8/5 ∧ p.a = 7/5 ∧ p.b = 1) ∨
    (p.x₀ = 0 ∧ p.y₀ = -23/5 ∧ p.a = 25 ∧ p.b = -35) :=
by sorry

end NUMINAMATH_CALUDE_valid_parameterization_l1006_100640


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1006_100650

theorem repeating_decimal_sum : 
  (1/3 : ℚ) + (4/999 : ℚ) + (5/9999 : ℚ) = (3378/9999 : ℚ) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1006_100650


namespace NUMINAMATH_CALUDE_cube_sum_and_product_l1006_100683

theorem cube_sum_and_product (a b : ℝ) (h1 : a + b = 12) (h2 : a * b = 20) :
  (a^3 + b^3 = 1008) ∧ ((a + b - (a - b)) * (a^3 + b^3) = 4032) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_and_product_l1006_100683


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_l1006_100699

/-- Triangle inequality for sides a, b, c -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

/-- The inequality to be proven -/
theorem triangle_inequality_cube (a b c : ℝ) (h : is_triangle a b c) :
  a^3 + b^3 + c^3 + 4*a*b*c ≤ 9/32 * (a + b + c)^3 :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_l1006_100699


namespace NUMINAMATH_CALUDE_pure_imaginary_solution_l1006_100623

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex-valued function z
def z (m : ℝ) : ℂ := (1 + i) * m^2 - (4 + i) * m + 3

-- Theorem statement
theorem pure_imaginary_solution (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 → m = 3 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_solution_l1006_100623


namespace NUMINAMATH_CALUDE_combined_average_correct_l1006_100645

-- Define the percentages for each city
def springfield : Fin 4 → ℚ
  | 0 => 12
  | 1 => 18
  | 2 => 25
  | 3 => 40

def shelbyville : Fin 4 → ℚ
  | 0 => 10
  | 1 => 15
  | 2 => 23
  | 3 => 35

-- Define the years
def years : Fin 4 → ℕ
  | 0 => 1990
  | 1 => 2000
  | 2 => 2010
  | 3 => 2020

-- Define the combined average function
def combinedAverage (i : Fin 4) : ℚ :=
  (springfield i + shelbyville i) / 2

-- Theorem statement
theorem combined_average_correct :
  (combinedAverage 0 = 11) ∧
  (combinedAverage 1 = 33/2) ∧
  (combinedAverage 2 = 24) ∧
  (combinedAverage 3 = 75/2) := by
  sorry

end NUMINAMATH_CALUDE_combined_average_correct_l1006_100645


namespace NUMINAMATH_CALUDE_no_integer_solution_l1006_100688

theorem no_integer_solution (a b c d : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) :
  ¬∃ x : ℤ, x^4 - a*x^3 - b*x^2 - c*x - d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l1006_100688
