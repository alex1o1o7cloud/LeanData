import Mathlib

namespace NUMINAMATH_CALUDE_initial_gasohol_volume_l3565_356576

/-- Represents the composition of a fuel mixture -/
structure FuelMixture where
  ethanol : ℝ  -- Percentage of ethanol
  gasoline : ℝ  -- Percentage of gasoline

/-- Represents the state of the fuel tank -/
structure FuelTank where
  volume : ℝ  -- Total volume in liters
  mixture : FuelMixture  -- Composition of the mixture

def initial_mixture : FuelMixture := { ethanol := 0.05, gasoline := 0.95 }
def desired_mixture : FuelMixture := { ethanol := 0.10, gasoline := 0.90 }
def ethanol_added : ℝ := 2.5

theorem initial_gasohol_volume (initial : FuelTank) (final : FuelTank) :
  initial.mixture = initial_mixture →
  final.mixture = desired_mixture →
  final.volume = initial.volume + ethanol_added →
  final.volume * final.mixture.ethanol = initial.volume * initial.mixture.ethanol + ethanol_added →
  initial.volume = 45 := by
  sorry

end NUMINAMATH_CALUDE_initial_gasohol_volume_l3565_356576


namespace NUMINAMATH_CALUDE_chromosome_variation_identification_l3565_356518

-- Define the structure for a genetic condition
structure GeneticCondition where
  name : String
  chromosomeAffected : Nat
  variationType : String

-- Define the statements
def statement1 : GeneticCondition := ⟨"cri-du-chat syndrome", 5, "partial deletion"⟩
def statement2 := "free combination of non-homologous chromosomes during meiosis"
def statement3 := "chromosomal exchange between synapsed homologous chromosomes"
def statement4 : GeneticCondition := ⟨"Down syndrome", 21, "extra chromosome"⟩

-- Define what constitutes a chromosome variation
def isChromosomeVariation (condition : GeneticCondition) : Prop :=
  condition.variationType = "partial deletion" ∨ condition.variationType = "extra chromosome"

-- Theorem to prove
theorem chromosome_variation_identification :
  (isChromosomeVariation statement1 ∧ isChromosomeVariation statement4) ∧
  (¬ isChromosomeVariation ⟨"", 0, statement2⟩ ∧ ¬ isChromosomeVariation ⟨"", 0, statement3⟩) := by
  sorry


end NUMINAMATH_CALUDE_chromosome_variation_identification_l3565_356518


namespace NUMINAMATH_CALUDE_prism_dimension_is_five_l3565_356532

/-- Represents a rectangular prism with dimensions n × n × 2n -/
structure RectangularPrism (n : ℕ) where
  length : ℕ := n
  width : ℕ := n
  height : ℕ := 2 * n

/-- The number of unit cubes obtained by cutting the prism -/
def num_unit_cubes (n : ℕ) : ℕ := 2 * n^3

/-- The total number of faces of all unit cubes -/
def total_faces (n : ℕ) : ℕ := 6 * num_unit_cubes n

/-- The number of blue faces (painted faces of the original prism) -/
def blue_faces (n : ℕ) : ℕ := 2 * n^2 + 4 * (2 * n^2)

/-- Theorem stating that if one-sixth of the total faces are blue, then n = 5 -/
theorem prism_dimension_is_five (n : ℕ) :
  (blue_faces n : ℚ) / (total_faces n : ℚ) = 1 / 6 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_prism_dimension_is_five_l3565_356532


namespace NUMINAMATH_CALUDE_blue_balls_count_l3565_356526

theorem blue_balls_count (total : ℕ) (removed : ℕ) (prob : ℚ) (initial : ℕ) : 
  total = 25 →
  removed = 5 →
  prob = 1/5 →
  (initial - removed : ℚ) / (total - removed : ℚ) = prob →
  initial = 9 := by sorry

end NUMINAMATH_CALUDE_blue_balls_count_l3565_356526


namespace NUMINAMATH_CALUDE_correct_average_l3565_356501

/-- Given 10 numbers with an initial average of 14, where one number 36 was incorrectly read as 26, prove that the correct average is 15. -/
theorem correct_average (n : ℕ) (initial_avg : ℚ) (incorrect_num correct_num : ℚ) :
  n = 10 →
  initial_avg = 14 →
  incorrect_num = 26 →
  correct_num = 36 →
  (n : ℚ) * initial_avg - incorrect_num + correct_num = n * 15 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l3565_356501


namespace NUMINAMATH_CALUDE_infinite_pairs_with_2020_diff_l3565_356567

/-- A positive integer is square-free if it is not divisible by any perfect square other than 1. -/
def IsSquareFree (n : ℕ) : Prop :=
  ∀ d : ℕ, d > 1 → d * d ∣ n → d = 1

/-- The sequence of square-free positive integers in ascending order. -/
def SquareFreeSequence : ℕ → ℕ := sorry

/-- The property that all integers between two given numbers are not square-free. -/
def AllBetweenNotSquareFree (m n : ℕ) : Prop :=
  ∀ k : ℕ, m < k → k < n → ¬(IsSquareFree k)

/-- The main theorem stating that there are infinitely many pairs of consecutive
    square-free integers in the sequence with a difference of 2020. -/
theorem infinite_pairs_with_2020_diff :
  ∀ N : ℕ, ∃ n : ℕ, n > N ∧
    IsSquareFree (SquareFreeSequence n) ∧
    IsSquareFree (SquareFreeSequence (n + 1)) ∧
    SquareFreeSequence (n + 1) - SquareFreeSequence n = 2020 ∧
    AllBetweenNotSquareFree (SquareFreeSequence n) (SquareFreeSequence (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinite_pairs_with_2020_diff_l3565_356567


namespace NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l3565_356588

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (m n : Line) (α β : Plane)
  (h1 : m ≠ n)
  (h2 : α ≠ β)
  (h3 : subset m α)
  (h4 : subset n β)
  (h5 : perp m β) :
  perpPlanes α β :=
sorry

end NUMINAMATH_CALUDE_line_perp_plane_implies_planes_perp_l3565_356588


namespace NUMINAMATH_CALUDE_parabola_directrix_l3565_356547

theorem parabola_directrix (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 → (∃ k : ℝ, y = 1 ↔ x^2 = 1 / (4 * k))) → 
  a = -1/4 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3565_356547


namespace NUMINAMATH_CALUDE_perimeter_of_8x4_formation_l3565_356593

/-- A rectangular formation of students -/
structure Formation :=
  (rows : ℕ)
  (columns : ℕ)

/-- The number of elements on the perimeter of a formation -/
def perimeter_count (f : Formation) : ℕ :=
  2 * (f.rows + f.columns) - 4

theorem perimeter_of_8x4_formation :
  let f : Formation := ⟨8, 4⟩
  perimeter_count f = 20 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_8x4_formation_l3565_356593


namespace NUMINAMATH_CALUDE_lillian_sugar_at_home_l3565_356561

/-- The number of cups of sugar needed for cupcake batter -/
def batterSugar (cupcakes : ℕ) : ℕ := cupcakes / 12

/-- The number of cups of sugar needed for cupcake frosting -/
def frostingSugar (cupcakes : ℕ) : ℕ := 2 * (cupcakes / 12)

/-- The total number of cups of sugar needed for cupcakes -/
def totalSugarNeeded (cupcakes : ℕ) : ℕ := batterSugar cupcakes + frostingSugar cupcakes

theorem lillian_sugar_at_home (cupcakes sugarBought sugarAtHome : ℕ) :
  cupcakes = 60 →
  sugarBought = 12 →
  totalSugarNeeded cupcakes = sugarBought + sugarAtHome →
  sugarAtHome = 3 := by
  sorry

#check lillian_sugar_at_home

end NUMINAMATH_CALUDE_lillian_sugar_at_home_l3565_356561


namespace NUMINAMATH_CALUDE_fathers_age_problem_l3565_356514

/-- The father's age problem -/
theorem fathers_age_problem (man_age father_age : ℕ) : 
  man_age = (2 * father_age) / 5 →
  man_age + 12 = (father_age + 12) / 2 →
  father_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_problem_l3565_356514


namespace NUMINAMATH_CALUDE_subgrids_cover_half_board_l3565_356511

/-- Represents a subgrid on the board -/
structure Subgrid where
  rows : ℕ
  cols : ℕ

/-- The board and its properties -/
structure Board where
  n : ℕ
  subgrids : List Subgrid

/-- Calculates the half-perimeter of a subgrid -/
def half_perimeter (s : Subgrid) : ℕ := s.rows + s.cols

/-- Checks if a list of subgrids covers the main diagonal -/
def covers_main_diagonal (b : Board) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ b.n → ∃ s ∈ b.subgrids, half_perimeter s ≥ b.n

/-- Calculates the number of squares covered by a list of subgrids -/
def squares_covered (b : Board) : ℕ :=
  sorry -- Implementation details omitted

/-- Main theorem -/
theorem subgrids_cover_half_board (b : Board) 
  (h_board_size : b.n * b.n = 11 * 60)
  (h_cover_diagonal : covers_main_diagonal b) :
  2 * (squares_covered b) ≥ b.n * b.n := by
  sorry

end NUMINAMATH_CALUDE_subgrids_cover_half_board_l3565_356511


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l3565_356515

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides and interior angles of 162 degrees -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (exterior_angle interior_angle : ℝ),
  exterior_angle = 18 →
  n = (360 : ℝ) / exterior_angle →
  interior_angle = 180 - exterior_angle →
  n = 20 ∧ interior_angle = 162 :=
by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l3565_356515


namespace NUMINAMATH_CALUDE_coin_flip_probability_l3565_356527

theorem coin_flip_probability (p : ℝ) : 
  (p ≥ 0 ∧ p ≤ 1) →  -- probability is between 0 and 1
  (∀ (n : ℕ), n > 0 → p = 1 - p) →  -- equal probability of heads and tails
  (3 : ℝ) * p^2 * (1 - p) = (3 / 8 : ℝ) →  -- probability of 2 heads in 3 flips is 0.375
  p = (1 / 2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l3565_356527


namespace NUMINAMATH_CALUDE_damaged_potatoes_calculation_l3565_356512

/-- Calculates the amount of damaged potatoes during transport -/
def damaged_potatoes (initial_amount : ℕ) (bag_size : ℕ) (price_per_bag : ℕ) (total_sales : ℕ) : ℕ :=
  initial_amount - (total_sales / price_per_bag * bag_size)

/-- Theorem stating the amount of damaged potatoes -/
theorem damaged_potatoes_calculation :
  damaged_potatoes 6500 50 72 9144 = 150 := by
  sorry

#eval damaged_potatoes 6500 50 72 9144

end NUMINAMATH_CALUDE_damaged_potatoes_calculation_l3565_356512


namespace NUMINAMATH_CALUDE_k_value_proof_l3565_356551

theorem k_value_proof (k : ℤ) 
  (h1 : (0.0004040404 : ℝ) * (10 : ℝ) ^ (k : ℝ) > 1000000)
  (h2 : (0.0004040404 : ℝ) * (10 : ℝ) ^ (k : ℝ) < 10000000) : 
  k = 11 := by
  sorry

end NUMINAMATH_CALUDE_k_value_proof_l3565_356551


namespace NUMINAMATH_CALUDE_ralph_tv_hours_l3565_356589

/-- The number of hours Ralph watches TV on weekdays (Monday to Friday) -/
def weekday_hours : ℕ := 4

/-- The number of hours Ralph watches TV on weekend days (Saturday and Sunday) -/
def weekend_hours : ℕ := 6

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The total number of hours Ralph watches TV in one week -/
def total_hours : ℕ := weekday_hours * weekdays + weekend_hours * weekend_days

theorem ralph_tv_hours : total_hours = 32 := by
  sorry

end NUMINAMATH_CALUDE_ralph_tv_hours_l3565_356589


namespace NUMINAMATH_CALUDE_efgh_is_parallelogram_l3565_356580

-- Define the types for points and quadrilaterals
variable (Point : Type) (Quadrilateral : Type)

-- Define the property of being a convex quadrilateral
variable (is_convex_quadrilateral : Quadrilateral → Prop)

-- Define the property of forming an equilateral triangle
variable (forms_equilateral_triangle : Point → Point → Point → Prop)

-- Define the property of a triangle being directed outward or inward
variable (is_outward : Point → Point → Point → Quadrilateral → Prop)
variable (is_inward : Point → Point → Point → Quadrilateral → Prop)

-- Define the property of being a parallelogram
variable (is_parallelogram : Point → Point → Point → Point → Prop)

-- Theorem statement
theorem efgh_is_parallelogram 
  (A B C D E F G H : Point) (Q : Quadrilateral) :
  is_convex_quadrilateral Q →
  forms_equilateral_triangle A B E →
  forms_equilateral_triangle B C F →
  forms_equilateral_triangle C D G →
  forms_equilateral_triangle D A H →
  is_outward A B E Q →
  is_outward C D G Q →
  is_inward B C F Q →
  is_inward D A H Q →
  is_parallelogram E F G H :=
by sorry

end NUMINAMATH_CALUDE_efgh_is_parallelogram_l3565_356580


namespace NUMINAMATH_CALUDE_expression_value_l3565_356583

theorem expression_value : 65 + (120 / 15) + (15 * 18) - 250 - (405 / 9) + 3^3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3565_356583


namespace NUMINAMATH_CALUDE_minimum_buses_needed_minimum_buses_for_field_trip_l3565_356578

theorem minimum_buses_needed 
  (total_students : ℕ) 
  (regular_capacity : ℕ) 
  (reduced_capacity : ℕ) 
  (reduced_buses : ℕ) : ℕ :=
  let remaining_students := total_students - (reduced_capacity * reduced_buses)
  let regular_buses_needed := (remaining_students + regular_capacity - 1) / regular_capacity
  regular_buses_needed + reduced_buses

theorem minimum_buses_for_field_trip : 
  minimum_buses_needed 1234 45 30 3 = 29 := by
  sorry

end NUMINAMATH_CALUDE_minimum_buses_needed_minimum_buses_for_field_trip_l3565_356578


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3565_356541

theorem simplify_and_evaluate (a b : ℤ) (h1 : a = -2) (h2 : b = 3) :
  2 * (a^2 - a*b) - 3 * ((2/3) * a^2 - a*b - 1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3565_356541


namespace NUMINAMATH_CALUDE_major_axis_length_is_8_l3565_356502

/-- The length of the major axis of an ellipse formed by intersecting a plane with a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_minor_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_minor_ratio

/-- Theorem: The length of the major axis is 8 when a plane intersects a right circular cylinder with radius 2, forming an ellipse where the major axis is double the minor axis -/
theorem major_axis_length_is_8 :
  major_axis_length 2 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_major_axis_length_is_8_l3565_356502


namespace NUMINAMATH_CALUDE_blue_cross_coverage_l3565_356555

/-- Represents a circular flag with overlapping crosses -/
structure CircularFlag where
  /-- The total area of the flag -/
  total_area : ℝ
  /-- The area covered by the blue cross -/
  blue_cross_area : ℝ
  /-- The area covered by the red cross -/
  red_cross_area : ℝ
  /-- The area covered by both crosses combined -/
  combined_crosses_area : ℝ
  /-- The red cross is half the width of the blue cross -/
  red_half_blue : blue_cross_area = 2 * red_cross_area
  /-- The combined area of both crosses is 50% of the flag's area -/
  combined_half_total : combined_crosses_area = 0.5 * total_area
  /-- The red cross covers 20% of the flag's area -/
  red_fifth_total : red_cross_area = 0.2 * total_area

/-- Theorem stating that the blue cross alone covers 30% of the flag's area -/
theorem blue_cross_coverage (flag : CircularFlag) : 
  flag.blue_cross_area = 0.3 * flag.total_area := by
  sorry

end NUMINAMATH_CALUDE_blue_cross_coverage_l3565_356555


namespace NUMINAMATH_CALUDE_f_derivative_at_2_l3565_356562

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) * (x - 6) * (x - 7) * (x - 8) * (x - 9) * (x - 10)

theorem f_derivative_at_2 : 
  deriv f 2 = 8 * 7 * 6 * 5 * 4 * 3 * 2 * 1 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_2_l3565_356562


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l3565_356579

theorem base_conversion_theorem : 
  ∃! n : ℕ, ∃ S : Finset ℕ, 
    (∀ c ∈ S, c ≥ 2 ∧ c^3 ≤ 250 ∧ 250 < c^4) ∧ 
    Finset.card S = n ∧ 
    n = 3 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l3565_356579


namespace NUMINAMATH_CALUDE_dans_stickers_l3565_356584

theorem dans_stickers (bob_stickers : ℕ) (tom_stickers : ℕ) (dan_stickers : ℕ)
  (h1 : bob_stickers = 12)
  (h2 : tom_stickers = 3 * bob_stickers)
  (h3 : dan_stickers = 2 * tom_stickers) :
  dan_stickers = 72 := by
sorry

end NUMINAMATH_CALUDE_dans_stickers_l3565_356584


namespace NUMINAMATH_CALUDE_linear_function_passes_through_point_l3565_356587

/-- A linear function y = kx - k (k ≠ 0) that passes through the point (-1, 4) also passes through the point (1, 0). -/
theorem linear_function_passes_through_point (k : ℝ) (hk : k ≠ 0) :
  (∃ y : ℝ, y = k * (-1) - k ∧ y = 4) →
  (∃ y : ℝ, y = k * 1 - k ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_passes_through_point_l3565_356587


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l3565_356585

theorem sum_of_squares_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) : 
  a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) ≥ 1/2 ∧
  (a^2 / (a + b) + b^2 / (b + c) + c^2 / (c + d) + d^2 / (d + a) = 1/2 ↔ 
   a = 1/4 ∧ b = 1/4 ∧ c = 1/4 ∧ d = 1/4) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l3565_356585


namespace NUMINAMATH_CALUDE_park_tree_increase_l3565_356530

/-- Represents the state of trees in the park -/
structure ParkState where
  maples : ℕ
  lindens : ℕ

/-- Calculates the total number of trees in the park -/
def total_trees (state : ParkState) : ℕ := state.maples + state.lindens

/-- Calculates the percentage of maples in the park -/
def maple_percentage (state : ParkState) : ℚ :=
  state.maples / (total_trees state)

/-- The initial state of the park -/
def initial_state : ParkState := sorry

/-- The state after planting lindens in spring -/
def spring_state : ParkState := sorry

/-- The final state after planting maples in autumn -/
def autumn_state : ParkState := sorry

theorem park_tree_increase :
  maple_percentage initial_state = 3/5 →
  maple_percentage spring_state = 1/5 →
  maple_percentage autumn_state = 3/5 →
  total_trees autumn_state = 6 * total_trees initial_state :=
sorry

end NUMINAMATH_CALUDE_park_tree_increase_l3565_356530


namespace NUMINAMATH_CALUDE_minimum_score_exists_l3565_356534

/-- Represents the scores of the four people who took the math test. -/
structure TestScores where
  marty : ℕ
  others : Fin 3 → ℕ

/-- The proposition that Marty's score is the minimum to conclude others scored below average. -/
def IsMinimumScore (scores : TestScores) : Prop :=
  scores.marty = 61 ∧
  (∀ i : Fin 3, scores.others i < 20) ∧
  (∀ s : TestScores, s.marty < 61 → 
    ∃ i : Fin 3, s.others i ≥ 20 ∨ (s.marty + (Finset.sum Finset.univ s.others)) / 4 ≠ 20)

/-- The theorem stating that there exists a score distribution satisfying the conditions. -/
theorem minimum_score_exists : ∃ scores : TestScores, IsMinimumScore scores ∧ 
  (scores.marty + (Finset.sum Finset.univ scores.others)) / 4 = 20 := by
  sorry

#check minimum_score_exists

end NUMINAMATH_CALUDE_minimum_score_exists_l3565_356534


namespace NUMINAMATH_CALUDE_g_sum_property_l3565_356553

def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^8 + e * x^6 - f * x^4 + 5

theorem g_sum_property (d e f : ℝ) :
  g d e f 20 = 7 → g d e f 20 + g d e f (-20) = 14 := by
  sorry

end NUMINAMATH_CALUDE_g_sum_property_l3565_356553


namespace NUMINAMATH_CALUDE_ratio_problem_l3565_356565

theorem ratio_problem (x : ℝ) : x / 10 = 17.5 / 1 → x = 175 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3565_356565


namespace NUMINAMATH_CALUDE_cubic_difference_equality_l3565_356558

theorem cubic_difference_equality (x y : ℝ) : 
  x^2 = 7 + 4 * Real.sqrt 3 ∧ 
  y^2 = 7 - 4 * Real.sqrt 3 → 
  x^3 / y - y^3 / x = 112 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_cubic_difference_equality_l3565_356558


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3565_356523

theorem arithmetic_sequence_sum (d : ℝ) (h : d ≠ 0) :
  let a : ℕ → ℝ := fun n => (n - 1 : ℝ) * d
  ∃ m : ℕ, a m = (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) ∧ m = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3565_356523


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3565_356535

-- Define set A
def A : Set ℝ := {x | x^2 - x = 0}

-- Define set B
def B : Set ℝ := {-1, 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3565_356535


namespace NUMINAMATH_CALUDE_lindas_savings_l3565_356554

theorem lindas_savings (savings : ℝ) : 
  (2 / 3 : ℝ) * savings + 250 = savings → savings = 750 := by
  sorry

end NUMINAMATH_CALUDE_lindas_savings_l3565_356554


namespace NUMINAMATH_CALUDE_m_less_than_neg_two_l3565_356586

/-- A quadratic function f(x) = x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

/-- The proposition that there exists a positive x_0 where f(x_0) < 0 -/
def exists_positive_root (m : ℝ) : Prop :=
  ∃ x_0 : ℝ, x_0 > 0 ∧ f m x_0 < 0

/-- Theorem: If there exists a positive x_0 where f(x_0) < 0, then m < -2 -/
theorem m_less_than_neg_two (m : ℝ) (h : exists_positive_root m) : m < -2 := by
  sorry

end NUMINAMATH_CALUDE_m_less_than_neg_two_l3565_356586


namespace NUMINAMATH_CALUDE_base_b_is_ten_l3565_356574

/-- Given that 1304 in base b, when squared, equals 99225 in base b, prove that b = 10 -/
theorem base_b_is_ten (b : ℕ) (h : b > 1) : 
  (b^3 + 3*b^2 + 4)^2 = 9*b^4 + 9*b^3 + 2*b^2 + 2*b + 5 → b = 10 := by
  sorry

#check base_b_is_ten

end NUMINAMATH_CALUDE_base_b_is_ten_l3565_356574


namespace NUMINAMATH_CALUDE_initial_choir_size_l3565_356539

/-- The number of girls initially in the choir is equal to the sum of blonde-haired and black-haired girls. -/
theorem initial_choir_size (blonde_girls black_girls : ℕ) 
  (h1 : blonde_girls = 30) 
  (h2 : black_girls = 50) : 
  blonde_girls + black_girls = 80 := by
  sorry

end NUMINAMATH_CALUDE_initial_choir_size_l3565_356539


namespace NUMINAMATH_CALUDE_pen_count_l3565_356594

theorem pen_count (num_pencils : ℕ) (max_students : ℕ) (h1 : num_pencils = 910) (h2 : max_students = 91) 
  (h3 : max_students ∣ num_pencils) : 
  ∃ num_pens : ℕ, num_pens = num_pencils :=
by sorry

end NUMINAMATH_CALUDE_pen_count_l3565_356594


namespace NUMINAMATH_CALUDE_original_price_proof_l3565_356529

-- Define the discount rates
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.05

-- Define the final sale price
def final_price : ℝ := 304

-- Theorem statement
theorem original_price_proof :
  ∃ (original_price : ℝ),
    original_price * (1 - first_discount) * (1 - second_discount) = final_price ∧
    original_price = 400 := by
  sorry

end NUMINAMATH_CALUDE_original_price_proof_l3565_356529


namespace NUMINAMATH_CALUDE_total_jeans_is_five_l3565_356566

/-- The number of Fox jeans purchased -/
def fox_jeans : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_jeans : ℕ := 2

/-- The total number of jeans purchased -/
def total_jeans : ℕ := fox_jeans + pony_jeans

theorem total_jeans_is_five : total_jeans = 5 := by sorry

end NUMINAMATH_CALUDE_total_jeans_is_five_l3565_356566


namespace NUMINAMATH_CALUDE_complex_location_l3565_356573

theorem complex_location (z : ℂ) (h : (1 + Complex.I * Real.sqrt 3) * z = Complex.I * (2 * Real.sqrt 3)) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end NUMINAMATH_CALUDE_complex_location_l3565_356573


namespace NUMINAMATH_CALUDE_combined_platform_length_l3565_356521

/-- The combined length of two train platforms -/
theorem combined_platform_length
  (length_train_a : ℝ)
  (time_platform_a : ℝ)
  (time_pole_a : ℝ)
  (length_train_b : ℝ)
  (time_platform_b : ℝ)
  (time_pole_b : ℝ)
  (h1 : length_train_a = 500)
  (h2 : time_platform_a = 75)
  (h3 : time_pole_a = 25)
  (h4 : length_train_b = 400)
  (h5 : time_platform_b = 60)
  (h6 : time_pole_b = 20) :
  (length_train_a + (length_train_a / time_pole_a) * time_platform_a - length_train_a) +
  (length_train_b + (length_train_b / time_pole_b) * time_platform_b - length_train_b) = 1800 :=
by sorry

end NUMINAMATH_CALUDE_combined_platform_length_l3565_356521


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3565_356595

theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x - y = 10) :
  x = 4 → y = 50 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3565_356595


namespace NUMINAMATH_CALUDE_median_in_75_79_interval_l3565_356538

/-- Represents a score interval with its frequency --/
structure ScoreInterval :=
  (lower upper : ℕ)
  (frequency : ℕ)

/-- The list of score intervals for the test --/
def scoreDistribution : List ScoreInterval :=
  [⟨85, 89, 20⟩, ⟨80, 84, 18⟩, ⟨75, 79, 15⟩, ⟨70, 74, 12⟩,
   ⟨65, 69, 10⟩, ⟨60, 64, 8⟩, ⟨55, 59, 10⟩, ⟨50, 54, 7⟩]

/-- The total number of students --/
def totalStudents : ℕ := 100

/-- Function to calculate the cumulative frequency up to a given interval --/
def cumulativeFrequency (intervals : List ScoreInterval) (targetLower : ℕ) : ℕ :=
  (intervals.filter (fun i => i.lower ≥ targetLower)).foldl (fun acc i => acc + i.frequency) 0

/-- Theorem stating that the median is in the 75-79 interval --/
theorem median_in_75_79_interval :
  ∃ (median : ℕ), 75 ≤ median ∧ median ≤ 79 ∧
  cumulativeFrequency scoreDistribution 75 > totalStudents / 2 ∧
  cumulativeFrequency scoreDistribution 80 ≤ totalStudents / 2 :=
sorry

end NUMINAMATH_CALUDE_median_in_75_79_interval_l3565_356538


namespace NUMINAMATH_CALUDE_gcf_210_286_l3565_356571

theorem gcf_210_286 : Nat.gcd 210 286 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcf_210_286_l3565_356571


namespace NUMINAMATH_CALUDE_racing_car_A_time_l3565_356568

/-- The time taken by racing car A to complete the track -/
def time_A : ℕ := 7

/-- The time taken by racing car B to complete the track -/
def time_B : ℕ := 24

/-- The time after which both cars are side by side again -/
def side_by_side_time : ℕ := 168

/-- Theorem stating that the time taken by racing car A is correct -/
theorem racing_car_A_time :
  (time_A = 7) ∧ 
  (time_B = 24) ∧
  (side_by_side_time = 168) ∧
  (Nat.lcm time_A time_B = side_by_side_time) :=
sorry

end NUMINAMATH_CALUDE_racing_car_A_time_l3565_356568


namespace NUMINAMATH_CALUDE_arithmetic_sequence_product_l3565_356503

/-- Given an arithmetic sequence {a_n}, if a_2^2 + 2a_2a_8 + a_6a_10 = 16, then a_4a_6 = 4 -/
theorem arithmetic_sequence_product (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 2^2 + 2 * a 2 * a 8 + a 6 * a 10 = 16 →
  a 4 * a 6 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_product_l3565_356503


namespace NUMINAMATH_CALUDE_square_side_length_l3565_356549

theorem square_side_length : ∃ (X : ℝ), X = 2.6 ∧ 
  (∃ (A B C D : ℝ × ℝ),
    -- Four points inside the square
    (0 < A.1 ∧ A.1 < X) ∧ (0 < A.2 ∧ A.2 < X) ∧
    (0 < B.1 ∧ B.1 < X) ∧ (0 < B.2 ∧ B.2 < X) ∧
    (0 < C.1 ∧ C.1 < X) ∧ (0 < C.2 ∧ C.2 < X) ∧
    (0 < D.1 ∧ D.1 < X) ∧ (0 < D.2 ∧ D.2 < X) ∧
    -- Nine segments of length 1
    (A.1 - 0)^2 + (A.2 - 0)^2 = 1 ∧
    (B.1 - X)^2 + (B.2 - X)^2 = 1 ∧
    (C.1 - 0)^2 + (C.2 - X)^2 = 1 ∧
    (D.1 - X)^2 + (D.2 - 0)^2 = 1 ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = 1 ∧
    (C.1 - A.1)^2 + (C.2 - A.2)^2 = 1 ∧
    (D.1 - A.1)^2 + (D.2 - A.2)^2 = 1 ∧
    (C.1 - B.1)^2 + (C.2 - B.2)^2 = 1 ∧
    (D.1 - C.1)^2 + (D.2 - C.2)^2 = 1 ∧
    -- Perpendicular segments
    A.1 = 0 ∧ B.1 = X ∧ C.2 = X ∧ D.2 = 0 ∧
    -- Distance conditions
    A.1 = (X - 1) / 2 ∧
    X - B.1 = 1) :=
by sorry

end NUMINAMATH_CALUDE_square_side_length_l3565_356549


namespace NUMINAMATH_CALUDE_tank_filling_time_l3565_356546

theorem tank_filling_time (fill_rate_A fill_rate_B : ℚ) : 
  fill_rate_A = 1 / 60 →
  15 * fill_rate_B + 15 * (fill_rate_A + fill_rate_B) = 1 →
  fill_rate_B = 1 / 40 :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l3565_356546


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_perpendicular_to_two_planes_are_parallel_l3565_356552

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Line → Plane → Prop)
variable (perpendicular_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (in_plane : Line → Plane → Prop)

-- Statement ②
theorem line_parallel_to_plane 
  (l : Line) (α : Plane) :
  parallel_plane l α → 
  ∃ (S : Set Line), (∀ m ∈ S, in_plane m α ∧ parallel l m) ∧ Set.Infinite S :=
sorry

-- Statement ④
theorem perpendicular_to_two_planes_are_parallel 
  (m : Line) (α β : Plane) :
  perpendicular_plane m α → perpendicular_plane m β → parallel_planes α β :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_perpendicular_to_two_planes_are_parallel_l3565_356552


namespace NUMINAMATH_CALUDE_rice_weight_scientific_notation_l3565_356524

theorem rice_weight_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 0.000035 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.5 ∧ n = -5 := by
  sorry

end NUMINAMATH_CALUDE_rice_weight_scientific_notation_l3565_356524


namespace NUMINAMATH_CALUDE_allan_has_one_more_balloon_l3565_356510

/-- Given the number of balloons Allan and Jake have, prove that Allan has one more balloon than Jake. -/
theorem allan_has_one_more_balloon (allan_balloons jake_initial_balloons jake_bought_balloons : ℕ) 
  (h1 : allan_balloons = 6)
  (h2 : jake_initial_balloons = 2)
  (h3 : jake_bought_balloons = 3) :
  allan_balloons - (jake_initial_balloons + jake_bought_balloons) = 1 := by
  sorry

end NUMINAMATH_CALUDE_allan_has_one_more_balloon_l3565_356510


namespace NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l3565_356598

theorem binomial_sum_divides_power_of_two (n : ℕ) : 
  n > 3 →
  (1 + Nat.choose n 1 + Nat.choose n 2 + Nat.choose n 3 ∣ 2^2000) ↔ 
  (n = 7 ∨ n = 23) := by
sorry

end NUMINAMATH_CALUDE_binomial_sum_divides_power_of_two_l3565_356598


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a15_l3565_356590

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a15
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 3 + a 13 = 20)
  (h_a2 : a 2 = -2) :
  a 15 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a15_l3565_356590


namespace NUMINAMATH_CALUDE_commute_time_difference_commute_time_difference_is_two_l3565_356520

/-- The difference in commute time between walking and taking the train -/
theorem commute_time_difference : ℝ :=
  let distance : ℝ := 1.5  -- miles
  let walking_speed : ℝ := 3  -- mph
  let train_speed : ℝ := 20  -- mph
  let additional_train_time : ℝ := 23.5  -- minutes

  let walking_time : ℝ := distance / walking_speed * 60  -- minutes
  let train_travel_time : ℝ := distance / train_speed * 60  -- minutes
  let total_train_time : ℝ := train_travel_time + additional_train_time

  walking_time - total_train_time

/-- The commute time difference is 2 minutes -/
theorem commute_time_difference_is_two : commute_time_difference = 2 := by
  sorry

end NUMINAMATH_CALUDE_commute_time_difference_commute_time_difference_is_two_l3565_356520


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3565_356559

def A : Set ℕ := {4, 5, 6, 8}
def B : Set ℕ := {3, 5, 7, 8}

theorem union_of_A_and_B : A ∪ B = {3, 4, 5, 6, 7, 8} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3565_356559


namespace NUMINAMATH_CALUDE_equation_solution_l3565_356542

theorem equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  let x : ℝ := (5 * m^2 + 18 * m * n + 18 * n^2) / (10 * m + 6 * n)
  (x + 2 * m)^2 - (x - 3 * n)^2 = 9 * (m + n)^2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3565_356542


namespace NUMINAMATH_CALUDE_banana_arrangements_count_l3565_356543

/-- The number of unique arrangements of letters in "BANANA" -/
def banana_arrangements : ℕ := 
  Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of "BANANA" is 60 -/
theorem banana_arrangements_count : banana_arrangements = 60 := by
  sorry

#eval banana_arrangements

end NUMINAMATH_CALUDE_banana_arrangements_count_l3565_356543


namespace NUMINAMATH_CALUDE_vacation_miles_theorem_l3565_356596

/-- Calculates the total miles driven during a vacation -/
def total_miles_driven (days : ℕ) (miles_per_day : ℕ) : ℕ :=
  days * miles_per_day

/-- Proves that a 5-day vacation driving 250 miles per day results in 1250 total miles -/
theorem vacation_miles_theorem :
  total_miles_driven 5 250 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_vacation_miles_theorem_l3565_356596


namespace NUMINAMATH_CALUDE_shortest_chord_length_max_triangle_area_l3565_356513

-- Define the circle and point A
def circle_radius : ℝ := 1
def distance_OA (a : ℝ) : Prop := 0 < a ∧ a < 1

-- Theorem for the shortest chord length
theorem shortest_chord_length (a : ℝ) (h : distance_OA a) :
  ∃ (chord_length : ℝ), chord_length = 2 * Real.sqrt (1 - a^2) ∧
  ∀ (other_chord : ℝ), other_chord ≥ chord_length :=
sorry

-- Theorem for the maximum area of triangle OMN
theorem max_triangle_area (a : ℝ) (h : distance_OA a) :
  ∃ (max_area : ℝ),
    (a ≥ Real.sqrt 2 / 2 → max_area = 1 / 2) ∧
    (a < Real.sqrt 2 / 2 → max_area = a * Real.sqrt (1 - a^2)) :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_length_max_triangle_area_l3565_356513


namespace NUMINAMATH_CALUDE_walk_time_proof_l3565_356548

/-- Ajay's walking speed in km/hour -/
def walking_speed : ℝ := 6

/-- The time taken to walk a certain distance in hours -/
def time_taken : ℝ := 11.666666666666666

/-- Theorem stating that the time taken to walk the distance is 11.666666666666666 hours -/
theorem walk_time_proof : time_taken = 11.666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_walk_time_proof_l3565_356548


namespace NUMINAMATH_CALUDE_course_selection_schemes_l3565_356544

theorem course_selection_schemes (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 2) :
  (Nat.choose (n - k) m) + (k * Nat.choose (n - k) (m - 1)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l3565_356544


namespace NUMINAMATH_CALUDE_lily_milk_problem_l3565_356506

theorem lily_milk_problem (initial_milk : ℚ) (milk_given : ℚ) (milk_left : ℚ) : 
  initial_milk = 5 → milk_given = 18/7 → milk_left = initial_milk - milk_given → milk_left = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_lily_milk_problem_l3565_356506


namespace NUMINAMATH_CALUDE_skier_total_time_l3565_356522

theorem skier_total_time (x : ℝ) (t₁ t₂ t₃ : ℝ) 
  (h1 : t₁ + t₂ = 40.5)
  (h2 : t₂ + t₃ = 37.5)
  (h3 : x / t₂ = (2 * x) / (t₁ + t₃))
  (h4 : x > 0) :
  t₁ + t₂ + t₃ = 58.5 := by
sorry

end NUMINAMATH_CALUDE_skier_total_time_l3565_356522


namespace NUMINAMATH_CALUDE_inequality_relationship_l3565_356577

theorem inequality_relationship :
  (∀ x : ℝ, x < -1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ x ≥ -1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_relationship_l3565_356577


namespace NUMINAMATH_CALUDE_cuboid_surface_area_example_l3565_356556

/-- The surface area of a cuboid with given dimensions. -/
def cuboidSurfaceArea (length breadth height : ℝ) : ℝ :=
  2 * (length * breadth + breadth * height + length * height)

/-- Theorem: The surface area of a cuboid with length 15, breadth 10, and height 16 is 1100. -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 15 10 16 = 1100 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_example_l3565_356556


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3565_356572

/-- A geometric sequence with a_2 = 8 and a_5 = 64 has a common ratio of 2 -/
theorem geometric_sequence_common_ratio : ∀ (a : ℕ → ℝ),
  (∀ n : ℕ, a (n + 1) = a n * a 1)  -- Definition of geometric sequence
  → a 2 = 8                         -- Given condition
  → a 5 = 64                        -- Given condition
  → a 1 = 2                         -- Common ratio q = a_1
  := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3565_356572


namespace NUMINAMATH_CALUDE_seven_successes_probability_l3565_356557

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The number of successes -/
def k : ℕ := 7

/-- The probability of k successes in n Bernoulli trials with probability p -/
def bernoulli_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- Theorem stating the probability of 7 successes in 7 Bernoulli trials with p = 2/7 -/
theorem seven_successes_probability : 
  bernoulli_probability n k p = 128/823543 := by
  sorry

end NUMINAMATH_CALUDE_seven_successes_probability_l3565_356557


namespace NUMINAMATH_CALUDE_min_triangle_area_in_cube_l3565_356582

/-- Given a cube with edge length a, the minimum area of triangles formed by 
    intersections of a plane parallel to the base with specific lines is 7a²/32 -/
theorem min_triangle_area_in_cube (a : ℝ) (ha : a > 0) : 
  ∃ (S : ℝ), S = (7 * a^2) / 32 ∧ 
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ a → 
    S ≤ (1/4) * |2*x^2 - 3*a*x + 2*a^2| := by
  sorry

end NUMINAMATH_CALUDE_min_triangle_area_in_cube_l3565_356582


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l3565_356505

/-- Fixed circle C -/
def C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 1

/-- Fixed line L -/
def L (x : ℝ) : Prop := x = 1

/-- Moving circle P -/
def P (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = r^2

/-- P is externally tangent to C -/
def externally_tangent (x y r : ℝ) : Prop :=
  (x + 2 - 1)^2 + y^2 = (r + 1)^2

/-- P is tangent to L -/
def tangent_to_L (x y r : ℝ) : Prop := x - r = 1

/-- Trajectory of the center of P -/
def trajectory (x y : ℝ) : Prop := y^2 = -8*x

theorem moving_circle_trajectory :
  ∀ x y r : ℝ,
  C x y ∧ L 1 ∧ P x y r ∧ externally_tangent x y r ∧ tangent_to_L x y r →
  trajectory x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l3565_356505


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3565_356516

theorem partial_fraction_decomposition :
  ∃! (A B C : ℝ), ∀ x : ℝ, x ≠ 0 → x^2 + 1 ≠ 0 →
    (-x^2 + 4*x - 5) / (x^3 + x) = A / x + (B*x + C) / (x^2 + 1) ∧
    A = -5 ∧ B = 4 ∧ C = 4 :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3565_356516


namespace NUMINAMATH_CALUDE_ordering_of_numbers_l3565_356569

theorem ordering_of_numbers (a b : ℝ) 
  (ha : a > 0) 
  (hb : b < 0) 
  (hab : a + b < 0) : 
  b < -a ∧ -a < 0 ∧ 0 < a ∧ a < -b :=
sorry

end NUMINAMATH_CALUDE_ordering_of_numbers_l3565_356569


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l3565_356509

theorem average_of_a_and_b (a b c : ℝ) : 
  ((b + c) / 2 = 180) → 
  (a - c = 200) → 
  ((a + b) / 2 = 280) :=
by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l3565_356509


namespace NUMINAMATH_CALUDE_wild_animal_population_estimation_l3565_356508

/-- Represents the data for a sample plot -/
structure PlotData where
  x : ℝ  -- plant coverage area
  y : ℝ  -- number of wild animals

/-- Represents the statistical data for the sample -/
structure SampleStats where
  n : ℕ              -- number of sample plots
  total_plots : ℕ    -- total number of plots in the area
  sum_x : ℝ          -- sum of x values
  sum_y : ℝ          -- sum of y values
  sum_x_squared : ℝ  -- sum of (x - x̄)²
  sum_y_squared : ℝ  -- sum of (y - ȳ)²
  sum_xy : ℝ         -- sum of (x - x̄)(y - ȳ)

/-- Theorem statement for the wild animal population estimation problem -/
theorem wild_animal_population_estimation
  (stats : SampleStats)
  (h1 : stats.n = 20)
  (h2 : stats.total_plots = 200)
  (h3 : stats.sum_x = 60)
  (h4 : stats.sum_y = 1200)
  (h5 : stats.sum_x_squared = 80)
  (h6 : stats.sum_y_squared = 9000)
  (h7 : stats.sum_xy = 800) :
  let estimated_population := (stats.sum_y / stats.n) * stats.total_plots
  let correlation_coefficient := stats.sum_xy / Real.sqrt (stats.sum_x_squared * stats.sum_y_squared)
  estimated_population = 12000 ∧ abs (correlation_coefficient - 0.94) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_wild_animal_population_estimation_l3565_356508


namespace NUMINAMATH_CALUDE_birthday_cars_count_l3565_356507

-- Define the initial number of cars
def initial_cars : ℕ := 14

-- Define the number of cars bought
def bought_cars : ℕ := 28

-- Define the number of cars given away
def given_away_cars : ℕ := 8 + 3

-- Define the final number of cars
def final_cars : ℕ := 43

-- Theorem to prove
theorem birthday_cars_count :
  ∃ (birthday_cars : ℕ), 
    initial_cars + bought_cars + birthday_cars - given_away_cars = final_cars ∧
    birthday_cars = 12 := by
  sorry

end NUMINAMATH_CALUDE_birthday_cars_count_l3565_356507


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l3565_356564

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (x + 1) / (x - 1) = 0 ∧ x ≠ 1 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l3565_356564


namespace NUMINAMATH_CALUDE_james_class_size_l3565_356531

theorem james_class_size (n : ℕ) : 
  (100 < n ∧ n < 200) ∧ 
  (∃ k : ℕ, n = 4 * k - 1) ∧
  (∃ k : ℕ, n = 5 * k - 2) ∧
  (∃ k : ℕ, n = 6 * k - 3) →
  n = 123 ∨ n = 183 := by
sorry

end NUMINAMATH_CALUDE_james_class_size_l3565_356531


namespace NUMINAMATH_CALUDE_cube_sum_inequality_l3565_356540

theorem cube_sum_inequality (a b c : ℤ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) : 
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + Real.sqrt (3 * (a * b + b * c + c * a + 1)) :=
sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_l3565_356540


namespace NUMINAMATH_CALUDE_smaller_cubes_count_l3565_356533

theorem smaller_cubes_count (large_volume : ℝ) (small_volume : ℝ) (surface_area_diff : ℝ) :
  large_volume = 125 →
  small_volume = 1 →
  surface_area_diff = 600 →
  (((6 * small_volume^(2/3)) * (large_volume / small_volume)) - (6 * large_volume^(2/3))) = surface_area_diff →
  (large_volume / small_volume) = 125 := by
  sorry

end NUMINAMATH_CALUDE_smaller_cubes_count_l3565_356533


namespace NUMINAMATH_CALUDE_no_decreasing_nat_function_exists_decreasing_int_function_l3565_356591

-- Define φ as a function from ℕ to ℕ
variable (φ : ℕ → ℕ)

-- Theorem 1: No such function f : ℕ → ℕ exists
theorem no_decreasing_nat_function : 
  ¬ ∃ f : ℕ → ℕ, ∀ x : ℕ, f x > f (φ x) := by sorry

-- Theorem 2: Such a function f : ℕ → ℤ exists
theorem exists_decreasing_int_function : 
  ∃ f : ℕ → ℤ, ∀ x : ℕ, f x > f (φ x) := by sorry

end NUMINAMATH_CALUDE_no_decreasing_nat_function_exists_decreasing_int_function_l3565_356591


namespace NUMINAMATH_CALUDE_problem_solution_l3565_356563

def X : Set ℕ := {n | 1 ≤ n ∧ n ≤ 2017}

def S : Set (ℕ × ℕ × ℕ) :=
  {t | t.1 ∈ X ∧ t.2.1 ∈ X ∧ t.2.2 ∈ X ∧
    ((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∨
     (t.2.1 < t.2.2 ∧ t.2.2 < t.1) ∨
     (t.2.2 < t.1 ∧ t.1 < t.2.1)) ∧
    ¬((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∧
      (t.2.1 < t.2.2 ∧ t.2.2 < t.1)) ∧
    ¬((t.1 < t.2.1 ∧ t.2.1 < t.2.2) ∧
      (t.2.2 < t.1 ∧ t.1 < t.2.1)) ∧
    ¬((t.2.1 < t.2.2 ∧ t.2.2 < t.1) ∧
      (t.2.2 < t.1 ∧ t.1 < t.2.1))}

theorem problem_solution (x y z w : ℕ) 
  (h1 : (x, y, z) ∈ S) (h2 : (z, w, x) ∈ S) :
  (y, z, w) ∈ S ∧ (x, y, w) ∈ S := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3565_356563


namespace NUMINAMATH_CALUDE_cube_root_negative_a_l3565_356581

theorem cube_root_negative_a (a : ℝ) : 
  ((-a : ℝ) ^ (1/3 : ℝ) = Real.sqrt 2) → (a ^ (1/3 : ℝ) = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_negative_a_l3565_356581


namespace NUMINAMATH_CALUDE_selene_sandwich_count_l3565_356500

/-- The cost of a sandwich in dollars -/
def sandwich_cost : ℕ := 2

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℕ := 2

/-- The cost of a hotdog in dollars -/
def hotdog_cost : ℕ := 1

/-- The cost of a can of fruit juice in dollars -/
def juice_cost : ℕ := 2

/-- The number of hamburgers Tanya buys -/
def tanya_hamburgers : ℕ := 2

/-- The number of cans of fruit juice Tanya buys -/
def tanya_juice : ℕ := 2

/-- The total amount spent by Selene and Tanya in dollars -/
def total_spent : ℕ := 16

/-- The number of sandwiches Selene bought -/
def selene_sandwiches : ℕ := 3

theorem selene_sandwich_count :
  ∃ (x : ℕ), x * sandwich_cost + juice_cost + 
  tanya_hamburgers * hamburger_cost + tanya_juice * juice_cost = total_spent ∧
  x = selene_sandwiches :=
by sorry

end NUMINAMATH_CALUDE_selene_sandwich_count_l3565_356500


namespace NUMINAMATH_CALUDE_inequality_preservation_l3565_356570

theorem inequality_preservation (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l3565_356570


namespace NUMINAMATH_CALUDE_average_cost_approx_1_50_l3565_356599

/-- Calculates the average cost per piece of fruit given specific quantities and prices. -/
def average_cost_per_fruit (apple_price banana_price orange_price grape_price kiwi_price : ℚ)
  (apple_qty banana_qty orange_qty grape_qty kiwi_qty : ℕ) : ℚ :=
  let apple_cost := if apple_qty ≥ 10 then (apple_qty - 2) * apple_price else apple_qty * apple_price
  let orange_cost := if orange_qty ≥ 3 then (orange_qty - (orange_qty / 3)) * orange_price else orange_qty * orange_price
  let grape_cost := if grape_qty * grape_price > 10 then grape_qty * grape_price * (1 - 0.2) else grape_qty * grape_price
  let kiwi_cost := if kiwi_qty ≥ 10 then kiwi_qty * kiwi_price * (1 - 0.15) else kiwi_qty * kiwi_price
  let banana_cost := banana_qty * banana_price
  let total_cost := apple_cost + orange_cost + grape_cost + kiwi_cost + banana_cost
  let total_pieces := apple_qty + orange_qty + grape_qty + kiwi_qty + banana_qty
  total_cost / total_pieces

/-- The average cost per piece of fruit is approximately $1.50 given the specific conditions. -/
theorem average_cost_approx_1_50 :
  ∃ ε > 0, |average_cost_per_fruit 2 1 3 (3/2) (7/4) 12 4 4 10 10 - (3/2)| < ε :=
by sorry

end NUMINAMATH_CALUDE_average_cost_approx_1_50_l3565_356599


namespace NUMINAMATH_CALUDE_inequality_holds_iff_l3565_356597

theorem inequality_holds_iff (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_l3565_356597


namespace NUMINAMATH_CALUDE_one_root_quadratic_l3565_356560

theorem one_root_quadratic (a : ℤ) : 
  (∃! x : ℝ, x ∈ Set.Icc 1 8 ∧ (x - a - 4)^2 + 2*x - 2*a - 16 = 0) ↔ 
  (a ∈ Set.Icc (-5) 0 ∨ a ∈ Set.Icc 3 8) :=
sorry

end NUMINAMATH_CALUDE_one_root_quadratic_l3565_356560


namespace NUMINAMATH_CALUDE_proportion_problem_l3565_356550

theorem proportion_problem (y : ℝ) : 0.75 / 0.9 = 5 / y → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_proportion_problem_l3565_356550


namespace NUMINAMATH_CALUDE_sequence_third_term_l3565_356545

/-- Given a sequence {a_n} with general term a_n = 3n - 5, prove that a_3 = 4 -/
theorem sequence_third_term (a : ℕ → ℤ) (h : ∀ n, a n = 3 * n - 5) : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_third_term_l3565_356545


namespace NUMINAMATH_CALUDE_geometric_sequence_quadratic_roots_l3565_356519

/-- Given that 2, b, a form a geometric sequence in order, prove that the equation ax^2 + bx + 1/3 = 0 has exactly 2 real roots -/
theorem geometric_sequence_quadratic_roots
  (b a : ℝ)
  (h_geometric : ∃ (q : ℝ), b = 2 * q ∧ a = 2 * q^2) :
  (∃! (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + 1/3 = 0 ∧ a * y^2 + b * y + 1/3 = 0) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_quadratic_roots_l3565_356519


namespace NUMINAMATH_CALUDE_oranges_per_day_l3565_356504

/-- Proves that the number of sacks harvested per day is 4, given 56 sacks over 14 days -/
theorem oranges_per_day (total_sacks : ℕ) (total_days : ℕ) 
  (h1 : total_sacks = 56) (h2 : total_days = 14) : 
  total_sacks / total_days = 4 := by
  sorry

end NUMINAMATH_CALUDE_oranges_per_day_l3565_356504


namespace NUMINAMATH_CALUDE_parabola_directrix_l3565_356575

/-- Given a parabola y = -3x^2 + 6x - 5, prove that its directrix is y = -23/12 -/
theorem parabola_directrix :
  let f : ℝ → ℝ := λ x => -3 * x^2 + 6 * x - 5
  ∃ k : ℝ, k = -23/12 ∧ ∀ x y : ℝ, f x = y →
    ∃ h : ℝ, h > 0 ∧ (x - 1)^2 + (y + 2 - k)^2 = (y + 2 - (k + h))^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3565_356575


namespace NUMINAMATH_CALUDE_cube_sum_theorem_l3565_356528

theorem cube_sum_theorem (a b : ℝ) (h1 : a + b = 7) (h2 : a * b = 5) :
  a^3 + b^3 = 238 := by
sorry

end NUMINAMATH_CALUDE_cube_sum_theorem_l3565_356528


namespace NUMINAMATH_CALUDE_positive_real_solution_of_equation_l3565_356536

theorem positive_real_solution_of_equation : 
  ∃! (x : ℝ), x > 0 ∧ (x - 6) / 11 = 6 / (x - 11) ∧ x = 17 := by
  sorry

end NUMINAMATH_CALUDE_positive_real_solution_of_equation_l3565_356536


namespace NUMINAMATH_CALUDE_equation_solution_l3565_356592

theorem equation_solution : ∃ x : ℚ, (5*x + 2*x = 450 - 10*(x - 5) + 4) ∧ (x = 504/17) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3565_356592


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l3565_356525

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 4) (h₂ : r₂ = 8) : 
  π * r₂^2 - π * r₁^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l3565_356525


namespace NUMINAMATH_CALUDE_sum_of_first_few_primes_equals_41_l3565_356517

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- The sum of the first n prime numbers -/
def sumFirstNPrimes (n : ℕ) : ℕ := sorry

/-- Theorem: There exists a unique positive integer n such that the sum of the first n prime numbers equals 41, and that n = 6 -/
theorem sum_of_first_few_primes_equals_41 :
  ∃! n : ℕ, n > 0 ∧ sumFirstNPrimes n = 41 ∧ n = 6 := by sorry

end NUMINAMATH_CALUDE_sum_of_first_few_primes_equals_41_l3565_356517


namespace NUMINAMATH_CALUDE_set_operations_and_subset_l3565_356537

def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | 2 < x ∧ x < 9}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem set_operations_and_subset :
  (A ∩ B = {x | 3 ≤ x ∧ x < 6}) ∧
  ((Bᶜ ∪ A) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ 9 ≤ x}) ∧
  (∀ a : ℝ, C a ⊆ B → (2 ≤ a ∧ a ≤ 8)) :=
sorry

end NUMINAMATH_CALUDE_set_operations_and_subset_l3565_356537
