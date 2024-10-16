import Mathlib

namespace NUMINAMATH_CALUDE_equal_volumes_l3247_324772

-- Define the tetrahedrons
structure Tetrahedron :=
  (a b c d e f : ℝ)

-- Define the volumes of the tetrahedrons
def volume (t : Tetrahedron) : ℝ := sorry

-- Define the specific tetrahedrons
def ABCD : Tetrahedron :=
  { a := 13, b := 5, c := 12, d := 13, e := 6, f := 5 }

def EFGH : Tetrahedron :=
  { a := 13, b := 13, c := 8, d := 5, e := 12, f := 5 }

-- Theorem statement
theorem equal_volumes : volume ABCD = volume EFGH := by
  sorry

end NUMINAMATH_CALUDE_equal_volumes_l3247_324772


namespace NUMINAMATH_CALUDE_roots_shifted_l3247_324751

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

-- Define the roots of the original polynomial
def roots_exist (a b c : ℝ) : Prop := 
  original_poly a = 0 ∧ original_poly b = 0 ∧ original_poly c = 0

-- Define the new polynomial
def new_poly (x : ℝ) : ℝ := x^3 + 7*x^2 + 14*x + 10

-- Theorem statement
theorem roots_shifted (a b c : ℝ) : 
  roots_exist a b c → 
  (new_poly (a - 3) = 0 ∧ new_poly (b - 3) = 0 ∧ new_poly (c - 3) = 0) :=
by sorry

end NUMINAMATH_CALUDE_roots_shifted_l3247_324751


namespace NUMINAMATH_CALUDE_substitution_remainder_l3247_324791

/-- Represents the number of ways to make substitutions in a basketball game -/
def substitution_ways (total_players : ℕ) (starting_players : ℕ) (max_substitutions : ℕ) : ℕ :=
  sorry

/-- The main theorem stating the remainder when the number of substitution ways is divided by 1000 -/
theorem substitution_remainder :
  substitution_ways 15 5 4 % 1000 = 301 := by
  sorry

end NUMINAMATH_CALUDE_substitution_remainder_l3247_324791


namespace NUMINAMATH_CALUDE_masha_comb_teeth_l3247_324745

/-- Represents a comb with teeth --/
structure Comb where
  numTeeth : ℕ
  numGaps : ℕ
  numSegments : ℕ

/-- The relationship between teeth and gaps in a comb --/
axiom comb_structure (c : Comb) : c.numGaps = c.numTeeth - 1

/-- The total number of segments in a comb --/
axiom comb_segments (c : Comb) : c.numSegments = c.numTeeth + c.numGaps

/-- Katya's comb --/
def katya_comb : Comb := { numTeeth := 11, numGaps := 10, numSegments := 21 }

/-- Masha's comb --/
def masha_comb : Comb := { numTeeth := 53, numGaps := 52, numSegments := 105 }

/-- The relationship between Katya's and Masha's combs --/
axiom comb_relationship : masha_comb.numSegments = 5 * katya_comb.numSegments

theorem masha_comb_teeth : masha_comb.numTeeth = 53 := by
  sorry

end NUMINAMATH_CALUDE_masha_comb_teeth_l3247_324745


namespace NUMINAMATH_CALUDE_four_numbers_sum_l3247_324746

theorem four_numbers_sum (a b c d T : ℝ) (h : a + b + c + d = T) :
  3 * ((a + 1) + (b + 1) + (c + 1) + (d + 1)) = 3 * T + 12 := by
  sorry

end NUMINAMATH_CALUDE_four_numbers_sum_l3247_324746


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3247_324789

theorem inequality_solution_range (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 4, x^2 + a*x - 2 < 0) ↔ a < 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3247_324789


namespace NUMINAMATH_CALUDE_nine_point_five_minutes_in_seconds_l3247_324762

/-- The number of seconds in one minute -/
def seconds_per_minute : ℕ := 60

/-- Converts minutes to seconds -/
def minutes_to_seconds (minutes : ℚ) : ℚ :=
  minutes * seconds_per_minute

/-- Theorem: 9.5 minutes is equal to 570 seconds -/
theorem nine_point_five_minutes_in_seconds :
  minutes_to_seconds (9.5 : ℚ) = 570 := by sorry

end NUMINAMATH_CALUDE_nine_point_five_minutes_in_seconds_l3247_324762


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l3247_324783

theorem triangle_angle_inequality (y : ℝ) (p q : ℝ) : 
  y > 0 →
  y + 10 > 0 →
  y + 5 > 0 →
  4*y > 0 →
  y + 10 + y + 5 > 4*y →
  y + 10 + 4*y > y + 5 →
  y + 5 + 4*y > y + 10 →
  4*y > y + 10 →
  4*y > y + 5 →
  p < y →
  y < q →
  q - p ≥ 5/3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l3247_324783


namespace NUMINAMATH_CALUDE_total_teaching_years_l3247_324788

/-- The combined total of years taught by Virginia, Adrienne, and Dennis -/
def combinedYears (adrienne virginia dennis : ℕ) : ℕ := adrienne + virginia + dennis

theorem total_teaching_years :
  ∀ (adrienne virginia dennis : ℕ),
  virginia = adrienne + 9 →
  virginia = dennis - 9 →
  dennis = 43 →
  combinedYears adrienne virginia dennis = 102 :=
by
  sorry

end NUMINAMATH_CALUDE_total_teaching_years_l3247_324788


namespace NUMINAMATH_CALUDE_book_reading_time_l3247_324776

theorem book_reading_time (pages_per_book : ℕ) (pages_per_day : ℕ) (days_to_finish : ℕ) : 
  pages_per_book = 249 → pages_per_day = 83 → days_to_finish = 3 →
  pages_per_book = days_to_finish * pages_per_day :=
by
  sorry

end NUMINAMATH_CALUDE_book_reading_time_l3247_324776


namespace NUMINAMATH_CALUDE_angle_problem_l3247_324704

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Addition of two angles -/
def Angle.add (a b : Angle) : Angle :=
  sorry

/-- Subtraction of two angles -/
def Angle.sub (a b : Angle) : Angle :=
  sorry

/-- Equality of two angles -/
def Angle.eq (a b : Angle) : Prop :=
  sorry

theorem angle_problem (x y : Angle) :
  Angle.add x y = Angle.mk 67 56 →
  Angle.sub x y = Angle.mk 12 40 →
  Angle.eq x (Angle.mk 40 18) ∧ Angle.eq y (Angle.mk 27 38) :=
by
  sorry

end NUMINAMATH_CALUDE_angle_problem_l3247_324704


namespace NUMINAMATH_CALUDE_function_inequality_solution_set_l3247_324790

open Set
open Function

theorem function_inequality_solution_set
  (f : ℝ → ℝ)
  (h_domain : ∀ x, x > 0 → DifferentiableAt ℝ f x)
  (h_ineq : ∀ x, x > 0 → x * deriv f x > f x)
  (h_f2 : f 2 = 0) :
  {x : ℝ | x > 0 ∧ f x < 0} = Ioo 0 2 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_solution_set_l3247_324790


namespace NUMINAMATH_CALUDE_body_part_count_l3247_324714

theorem body_part_count (suspension_days_per_instance : ℕ) 
                        (total_bullying_instances : ℕ) 
                        (body_part_count : ℕ) : 
  suspension_days_per_instance = 3 →
  total_bullying_instances = 20 →
  suspension_days_per_instance * total_bullying_instances = 3 * body_part_count →
  body_part_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_body_part_count_l3247_324714


namespace NUMINAMATH_CALUDE_inequality_proof_l3247_324753

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum : a + b + c = 1) :
  (1/a + 1/(b*c)) * (1/b + 1/(c*a)) * (1/c + 1/(a*b)) ≥ 1728 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3247_324753


namespace NUMINAMATH_CALUDE_fraction_calculation_l3247_324723

theorem fraction_calculation : 
  (3/7 + 2/3) / (5/12 + 1/4) = 23/14 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3247_324723


namespace NUMINAMATH_CALUDE_player_b_winning_strategy_l3247_324702

/-- Represents the game state with two players on a line -/
structure GameState where
  L : ℕ+  -- Distance between initial positions (positive integer)
  a : ℕ+  -- Move distance for player A (positive integer)
  b : ℕ+  -- Move distance for player B (positive integer)
  h : a < b  -- Condition that a is less than b

/-- Winning condition for player B -/
def winning_condition (g : GameState) : Prop :=
  g.b = 2 * g.a ∧ ∃ k : ℕ, g.L = k * g.a

/-- Theorem stating the necessary and sufficient conditions for player B to have a winning strategy -/
theorem player_b_winning_strategy (g : GameState) :
  winning_condition g ↔ ∃ (strategy : Unit), True  -- Replace True with actual strategy type when implementing
:= by sorry

end NUMINAMATH_CALUDE_player_b_winning_strategy_l3247_324702


namespace NUMINAMATH_CALUDE_lychee_sale_ratio_l3247_324747

/-- Represents the lychee harvest and sales problem -/
structure LycheeHarvest where
  total : ℕ
  remaining : ℕ
  eaten_fraction : ℚ

/-- Calculates the ratio of sold lychees to total harvested lychees -/
def sold_to_total_ratio (harvest : LycheeHarvest) : ℚ × ℚ :=
  let sold := harvest.total - (harvest.remaining / (1 - harvest.eaten_fraction))
  (sold, harvest.total)

/-- Theorem stating that the ratio of sold lychees to total harvested lychees is 1:2 -/
theorem lychee_sale_ratio :
  let harvest := LycheeHarvest.mk 500 100 (3/5)
  sold_to_total_ratio harvest = (1, 2) := by
  sorry

#eval sold_to_total_ratio (LycheeHarvest.mk 500 100 (3/5))

end NUMINAMATH_CALUDE_lychee_sale_ratio_l3247_324747


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3247_324711

theorem arithmetic_sequence_proof :
  ∀ (a : ℕ → ℤ),
    (∀ i j : ℕ, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence condition
    (a 0 = 3^2) →  -- first term is 3²
    (a 2 = 3^4) →  -- third term is 3⁴
    (a 1 = 33 ∧ a 3 = 105) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l3247_324711


namespace NUMINAMATH_CALUDE_least_multiple_21_greater_380_l3247_324724

theorem least_multiple_21_greater_380 : ∃ (n : ℕ), n * 21 = 399 ∧ 
  399 > 380 ∧ 
  (∀ m : ℕ, m * 21 > 380 → m * 21 ≥ 399) := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_21_greater_380_l3247_324724


namespace NUMINAMATH_CALUDE_chocolate_chips_per_family_member_l3247_324703

/-- Proves that each family member eats 38 chocolate chips given the problem conditions -/
theorem chocolate_chips_per_family_member :
  let family_members : ℕ := 4
  let choc_chip_batches : ℕ := 3
  let double_choc_chip_batches : ℕ := 2
  let cookies_per_choc_chip_batch : ℕ := 12
  let chips_per_choc_chip_cookie : ℕ := 2
  let cookies_per_double_choc_chip_batch : ℕ := 10
  let chips_per_double_choc_chip_cookie : ℕ := 4

  let total_choc_chip_cookies : ℕ := choc_chip_batches * cookies_per_choc_chip_batch
  let total_double_choc_chip_cookies : ℕ := double_choc_chip_batches * cookies_per_double_choc_chip_batch
  
  let total_chips_in_choc_chip_cookies : ℕ := total_choc_chip_cookies * chips_per_choc_chip_cookie
  let total_chips_in_double_choc_chip_cookies : ℕ := total_double_choc_chip_cookies * chips_per_double_choc_chip_cookie
  
  let total_chips : ℕ := total_chips_in_choc_chip_cookies + total_chips_in_double_choc_chip_cookies

  (total_chips / family_members : ℕ) = 38 := by sorry

end NUMINAMATH_CALUDE_chocolate_chips_per_family_member_l3247_324703


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l3247_324731

theorem greatest_of_three_consecutive_integers (x y z : ℤ) : 
  (y = x + 1) → (z = y + 1) → (x + y + z = 33) → (max x (max y z) = 12) := by
  sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l3247_324731


namespace NUMINAMATH_CALUDE_road_repair_groups_equivalent_l3247_324722

/-- The number of persons in the second group repairing the road -/
def second_group_size : ℕ := 30

/-- The number of persons in the first group -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours_per_day : ℕ := 10

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours_per_day : ℕ := 6

/-- The total man-hours required to complete the road repair -/
def total_man_hours : ℕ := first_group_size * first_group_days * first_group_hours_per_day

theorem road_repair_groups_equivalent :
  second_group_size * second_group_days * second_group_hours_per_day = total_man_hours :=
by sorry

end NUMINAMATH_CALUDE_road_repair_groups_equivalent_l3247_324722


namespace NUMINAMATH_CALUDE_papaya_tree_first_year_growth_l3247_324712

/-- The growth pattern of a papaya tree over 5 years -/
def PapayaTreeGrowth (first_year_growth : ℝ) : ℝ :=
  let second_year := 1.5 * first_year_growth
  let third_year := 1.5 * second_year
  let fourth_year := 2 * third_year
  let fifth_year := 0.5 * fourth_year
  first_year_growth + second_year + third_year + fourth_year + fifth_year

/-- Theorem stating that if a papaya tree grows to 23 feet in 5 years following the given pattern, 
    it must have grown 2 feet in the first year -/
theorem papaya_tree_first_year_growth :
  ∃ (x : ℝ), PapayaTreeGrowth x = 23 → x = 2 :=
sorry

end NUMINAMATH_CALUDE_papaya_tree_first_year_growth_l3247_324712


namespace NUMINAMATH_CALUDE_marbles_sum_theorem_l3247_324766

/-- The number of yellow marbles Mary and Joan have in total -/
def total_marbles (mary_marbles joan_marbles : ℕ) : ℕ :=
  mary_marbles + joan_marbles

/-- Theorem stating that Mary and Joan have 12 yellow marbles in total -/
theorem marbles_sum_theorem :
  total_marbles 9 3 = 12 := by sorry

end NUMINAMATH_CALUDE_marbles_sum_theorem_l3247_324766


namespace NUMINAMATH_CALUDE_pirate_digging_time_pirate_digging_time_proof_l3247_324725

/-- Calculates the time needed to dig up a buried treasure after natural events --/
theorem pirate_digging_time (initial_depth : ℝ) (initial_time : ℝ) 
  (storm_factor : ℝ) (tsunami_sand : ℝ) (earthquake_sand : ℝ) (mudslide_sand : ℝ) 
  (speed_change : ℝ) : ℝ :=
  let initial_speed := initial_depth / initial_time
  let new_speed := initial_speed * (1 - speed_change)
  let final_depth := initial_depth * storm_factor + tsunami_sand + earthquake_sand + mudslide_sand
  final_depth / new_speed

/-- Proves that the time to dig up the treasure is approximately 6.56 hours --/
theorem pirate_digging_time_proof :
  ∃ ε > 0, |pirate_digging_time 8 4 0.5 2 1.5 3 0.2 - 6.56| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_pirate_digging_time_pirate_digging_time_proof_l3247_324725


namespace NUMINAMATH_CALUDE_solve_strawberry_problem_l3247_324771

def strawberry_problem (betty_strawberries : ℕ) (matthew_extra : ℕ) (jar_strawberries : ℕ) (total_money : ℕ) : Prop :=
  let matthew_strawberries := betty_strawberries + matthew_extra
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let num_jars := total_strawberries / jar_strawberries
  let price_per_jar := total_money / num_jars
  price_per_jar = 4

theorem solve_strawberry_problem :
  strawberry_problem 16 20 7 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_strawberry_problem_l3247_324771


namespace NUMINAMATH_CALUDE_triangle_problem_l3247_324740

theorem triangle_problem (a b c A B C : ℝ) (h1 : c * Real.cos A - Real.sqrt 3 * a * Real.sin C - c = 0)
  (h2 : a = 2) (h3 : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  A = π/3 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3247_324740


namespace NUMINAMATH_CALUDE_NaHSO3_moles_required_l3247_324726

/-- Represents the balanced chemical equation for the reaction -/
structure ChemicalEquation :=
  (reactants : List String)
  (products : List String)

/-- Represents the stoichiometric coefficient of a substance in a reaction -/
def stoichiometricCoefficient (equation : ChemicalEquation) (substance : String) : ℕ :=
  if substance ∈ equation.reactants ∨ substance ∈ equation.products then 1 else 0

/-- The chemical equation for the given reaction -/
def reactionEquation : ChemicalEquation :=
  { reactants := ["NaHSO3", "HCl"],
    products := ["SO2", "H2O", "NaCl"] }

/-- Theorem stating the number of moles of NaHSO3 required to form 2 moles of SO2 -/
theorem NaHSO3_moles_required :
  let NaHSO3_coeff := stoichiometricCoefficient reactionEquation "NaHSO3"
  let SO2_coeff := stoichiometricCoefficient reactionEquation "SO2"
  let SO2_moles_formed := 2
  NaHSO3_coeff * SO2_moles_formed / SO2_coeff = 2 := by
  sorry

end NUMINAMATH_CALUDE_NaHSO3_moles_required_l3247_324726


namespace NUMINAMATH_CALUDE_num_sequences_equals_binomial_remainder_of_m_mod_1000_l3247_324733

/-- The number of increasing sequences of 10 positive integers satisfying given conditions -/
def num_sequences : ℕ := sorry

/-- The upper bound for each term in the sequence -/
def upper_bound : ℕ := 2007

/-- The length of the sequence -/
def sequence_length : ℕ := 10

/-- Predicate to check if a sequence satisfies the required conditions -/
def valid_sequence (a : Fin sequence_length → ℕ) : Prop :=
  (∀ i j : Fin sequence_length, i ≤ j → a i ≤ a j) ∧
  (∀ i : Fin sequence_length, a i ≤ upper_bound) ∧
  (∀ i : Fin sequence_length, Even (a i - i.val))

theorem num_sequences_equals_binomial :
  num_sequences = Nat.choose 1008 sequence_length :=
sorry

theorem remainder_of_m_mod_1000 :
  1008 % 1000 = 8 :=
sorry

end NUMINAMATH_CALUDE_num_sequences_equals_binomial_remainder_of_m_mod_1000_l3247_324733


namespace NUMINAMATH_CALUDE_janice_initial_sentences_janice_started_with_258_l3247_324797

/-- Calculates the number of sentences Janice started with today -/
theorem janice_initial_sentences 
  (typing_speed : ℕ) 
  (total_typing_time : ℕ) 
  (erased_sentences : ℕ) 
  (final_sentence_count : ℕ) : ℕ :=
let typed_sentences := typing_speed * total_typing_time
let added_sentences := typed_sentences - erased_sentences
final_sentence_count - added_sentences

/-- Proves that Janice started with 258 sentences today -/
theorem janice_started_with_258 : 
  janice_initial_sentences 6 53 40 536 = 258 := by
sorry

end NUMINAMATH_CALUDE_janice_initial_sentences_janice_started_with_258_l3247_324797


namespace NUMINAMATH_CALUDE_tile_border_ratio_l3247_324794

theorem tile_border_ratio (n : ℕ) (s d : ℝ) (h1 : n = 24) 
  (h2 : (n^2 : ℝ) * s^2 * 0.64 = 576 * s^2) : d / s = 6 / 25 := by
  sorry

end NUMINAMATH_CALUDE_tile_border_ratio_l3247_324794


namespace NUMINAMATH_CALUDE_units_digit_of_sequence_sum_l3247_324709

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sequence_term (n : ℕ) : ℕ := factorial n + 10

def sequence_sum (n : ℕ) : ℕ := (List.range n).map sequence_term |>.sum

theorem units_digit_of_sequence_sum :
  sequence_sum 10 % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sequence_sum_l3247_324709


namespace NUMINAMATH_CALUDE_sin_cos_difference_l3247_324787

theorem sin_cos_difference (x : Real) :
  (Real.sin x)^3 - (Real.cos x)^3 = -1 → Real.sin x - Real.cos x = -1 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l3247_324787


namespace NUMINAMATH_CALUDE_sets_intersection_empty_l3247_324778

-- Define set A
def A : Set ℝ := {x | x^2 + 5*x + 6 ≤ 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.sqrt (-x^2 + 2*x + 15)}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem sets_intersection_empty (a : ℝ) : (A ∪ B) ∩ C a = ∅ ↔ a ≥ 5 ∨ a ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_sets_intersection_empty_l3247_324778


namespace NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l3247_324758

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -1

-- Define a circle with center at the origin
def circle_at_origin (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Theorem statement
theorem circle_tangent_to_parabola_directrix :
  ∀ (x y r : ℝ),
  (∃ (x_d : ℝ), directrix x_d ∧ 
    (∀ (x_p y_p : ℝ), parabola x_p y_p → x_p ≥ x_d) ∧
    (∃ (x_t y_t : ℝ), parabola x_t y_t ∧ x_t = x_d ∧ 
      circle_at_origin x_t y_t r)) →
  circle_at_origin x y 1 :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_parabola_directrix_l3247_324758


namespace NUMINAMATH_CALUDE_cube_product_inequality_l3247_324780

theorem cube_product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^3 * y^3 * (x^3 + y^3) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_product_inequality_l3247_324780


namespace NUMINAMATH_CALUDE_circle_radius_difference_l3247_324749

-- Define the circles and points
def larger_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 13^2}
def smaller_circle : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9^2}
def P : ℝ × ℝ := (5, 12)
def S (k : ℝ) : ℝ × ℝ := (0, k)

-- State the theorem
theorem circle_radius_difference (k : ℝ) : 
  P ∈ larger_circle ∧ 
  S k ∈ smaller_circle ∧
  (13 : ℝ) - 9 = 4 →
  k = 9 := by sorry

end NUMINAMATH_CALUDE_circle_radius_difference_l3247_324749


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3247_324754

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B : Set ℝ := {x | x*(x-2)*(x-5) < 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | 1 < x ∧ x < 5} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3247_324754


namespace NUMINAMATH_CALUDE_parentheses_value_l3247_324781

theorem parentheses_value : (6 : ℝ) / Real.sqrt 18 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parentheses_value_l3247_324781


namespace NUMINAMATH_CALUDE_smallest_overlap_percentage_l3247_324795

theorem smallest_overlap_percentage (coffee_drinkers tea_drinkers : ℝ) 
  (h1 : coffee_drinkers = 75)
  (h2 : tea_drinkers = 80) :
  coffee_drinkers + tea_drinkers - 100 = 55 :=
by sorry

end NUMINAMATH_CALUDE_smallest_overlap_percentage_l3247_324795


namespace NUMINAMATH_CALUDE_max_power_of_five_equals_three_l3247_324735

/-- The number of divisors of a positive integer -/
noncomputable def num_divisors (n : ℕ+) : ℕ := sorry

/-- The greatest integer j such that 5^j divides n -/
noncomputable def max_power_of_five (n : ℕ+) : ℕ := sorry

theorem max_power_of_five_equals_three (n : ℕ+) 
  (h1 : num_divisors n = 72)
  (h2 : num_divisors (5 * n) = 90) :
  max_power_of_five n = 3 := by sorry

end NUMINAMATH_CALUDE_max_power_of_five_equals_three_l3247_324735


namespace NUMINAMATH_CALUDE_sqrt_6000_approx_l3247_324782

/-- Approximate value of the square root of 6 -/
def sqrt_6_approx : ℝ := 2.45

/-- Approximate value of the square root of 60 -/
def sqrt_60_approx : ℝ := 7.75

/-- Theorem stating that the square root of 6000 is approximately 77.5 -/
theorem sqrt_6000_approx : ∃ (ε : ℝ), ε > 0 ∧ |Real.sqrt 6000 - 77.5| < ε :=
sorry

end NUMINAMATH_CALUDE_sqrt_6000_approx_l3247_324782


namespace NUMINAMATH_CALUDE_parabola_tangent_sum_l3247_324755

/-- The parabola P with equation y = x^2 -/
def P : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2}

/-- The point Q -/
def Q : ℝ × ℝ := (10, 36)

/-- The line through Q with slope m -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) := 
  {p : ℝ × ℝ | p.2 - Q.2 = m * (p.1 - Q.1)}

/-- The condition for non-intersection -/
def no_intersection (m r s : ℝ) : Prop :=
  (∀ x : ℝ, (x, x^2) ∉ line_through_Q m) ↔ r < m ∧ m < s

theorem parabola_tangent_sum (r s : ℝ) 
  (h : ∀ m : ℝ, no_intersection m r s) : r + s = 40 := by
  sorry

#check parabola_tangent_sum

end NUMINAMATH_CALUDE_parabola_tangent_sum_l3247_324755


namespace NUMINAMATH_CALUDE_sheela_net_monthly_income_l3247_324752

/-- Calculates the total net monthly income for Sheela given her various income sources and tax rates. -/
theorem sheela_net_monthly_income 
  (savings_deposit : ℝ)
  (savings_deposit_percentage : ℝ)
  (freelance_income : ℝ)
  (annual_interest : ℝ)
  (freelance_tax_rate : ℝ)
  (interest_tax_rate : ℝ)
  (h1 : savings_deposit = 5000)
  (h2 : savings_deposit_percentage = 0.20)
  (h3 : freelance_income = 3000)
  (h4 : annual_interest = 2400)
  (h5 : freelance_tax_rate = 0.10)
  (h6 : interest_tax_rate = 0.05) :
  ∃ (total_net_monthly_income : ℝ), 
    total_net_monthly_income = 27890 :=
by sorry

end NUMINAMATH_CALUDE_sheela_net_monthly_income_l3247_324752


namespace NUMINAMATH_CALUDE_lena_collage_glue_drops_l3247_324764

/-- The number of closest friends Lena has -/
def num_friends : ℕ := 7

/-- The number of clippings per friend -/
def clippings_per_friend : ℕ := 3

/-- The number of glue drops needed per clipping -/
def glue_drops_per_clipping : ℕ := 6

/-- The total number of glue drops needed for Lena's collage clippings -/
def total_glue_drops : ℕ := num_friends * clippings_per_friend * glue_drops_per_clipping

theorem lena_collage_glue_drops : total_glue_drops = 126 := by
  sorry

end NUMINAMATH_CALUDE_lena_collage_glue_drops_l3247_324764


namespace NUMINAMATH_CALUDE_max_y_over_x_l3247_324750

theorem max_y_over_x (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≥ 0) (h3 : x^2 + y^2 - 4*x + 1 = 0) :
  ∃ (k : ℝ), ∀ (x' y' : ℝ), x' ≠ 0 → y' ≥ 0 → x'^2 + y'^2 - 4*x' + 1 = 0 → y'/x' ≤ k ∧ k = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_y_over_x_l3247_324750


namespace NUMINAMATH_CALUDE_inequality_proof_l3247_324738

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a^5 + b^5 + a * b) + (b * c) / (b^5 + c^5 + b * c) + (c * a) / (c^5 + a^5 + c * a) ≤ 1 ∧
  ((a * b) / (a^5 + b^5 + a * b) + (b * c) / (b^5 + c^5 + b * c) + (c * a) / (c^5 + a^5 + c * a) = 1 ↔ a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3247_324738


namespace NUMINAMATH_CALUDE_equation_solution_l3247_324774

theorem equation_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^2 + b^3/a = b^2 + a^3/b → a = b ∨ a = -b := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3247_324774


namespace NUMINAMATH_CALUDE_prob_neither_red_nor_white_l3247_324756

/-- The probability of drawing a ball that is neither red nor white from a bag containing
    2 red balls, 3 white balls, and 5 yellow balls. -/
theorem prob_neither_red_nor_white :
  let total_balls : ℕ := 2 + 3 + 5
  let yellow_balls : ℕ := 5
  (yellow_balls : ℚ) / total_balls = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_neither_red_nor_white_l3247_324756


namespace NUMINAMATH_CALUDE_greatest_multiple_of_four_with_fifth_power_less_than_2000_l3247_324757

theorem greatest_multiple_of_four_with_fifth_power_less_than_2000 :
  ∃ (x : ℕ), x > 0 ∧ 4 ∣ x ∧ x^5 < 2000 ∧ ∀ y : ℕ, y > 0 → 4 ∣ y → y^5 < 2000 → y ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_four_with_fifth_power_less_than_2000_l3247_324757


namespace NUMINAMATH_CALUDE_tree_spacing_l3247_324743

theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) :
  yard_length = 350 →
  num_trees = 26 →
  yard_length / (num_trees - 1) = 14 :=
by sorry

end NUMINAMATH_CALUDE_tree_spacing_l3247_324743


namespace NUMINAMATH_CALUDE_reflect_M_x_axis_l3247_324730

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The point M -/
def M : ℝ × ℝ := (1, 2)

theorem reflect_M_x_axis :
  reflect_x M = (1, -2) := by sorry

end NUMINAMATH_CALUDE_reflect_M_x_axis_l3247_324730


namespace NUMINAMATH_CALUDE_intersection_point_coordinates_l3247_324700

/-- Given a triangle ABC with points F on BC and G on AC, prove that the intersection Q of BG and AF
    can be expressed as a linear combination of A, B, and C. -/
theorem intersection_point_coordinates (A B C F G Q : ℝ × ℝ) : 
  (∃ t : ℝ, F = (1 - t) • B + t • C ∧ t = 1/3) →  -- F lies on BC with BF:FC = 2:1
  (∃ s : ℝ, G = (1 - s) • A + s • C ∧ s = 3/5) →  -- G lies on AC with AG:GC = 3:2
  (∃ u v : ℝ, Q = (1 - u) • B + u • G ∧ Q = (1 - v) • A + v • F) →  -- Q is intersection of BG and AF
  Q = (2/5) • A + (1/3) • B + (4/9) • C := by sorry

end NUMINAMATH_CALUDE_intersection_point_coordinates_l3247_324700


namespace NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3247_324742

def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

def leaves_remainder_1 (n : ℕ) (d : ℕ) : Prop := n % d = 1

theorem smallest_number_satisfying_conditions : 
  (∀ d : ℕ, 2 ≤ d → d ≤ 8 → leaves_remainder_1 6721 d) ∧ 
  is_divisible_by_11 6721 ∧
  (∀ m : ℕ, m < 6721 → 
    (¬(∀ d : ℕ, 2 ≤ d → d ≤ 8 → leaves_remainder_1 m d) ∨ 
     ¬(is_divisible_by_11 m))) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_satisfying_conditions_l3247_324742


namespace NUMINAMATH_CALUDE_fifteenth_digit_is_zero_l3247_324719

/-- The decimal representation of 1/8 -/
def frac_1_8 : ℚ := 1/8

/-- The decimal representation of 1/11 -/
def frac_1_11 : ℚ := 1/11

/-- The sum of the decimal representations of 1/8 and 1/11 -/
def sum_fracs : ℚ := frac_1_8 + frac_1_11

/-- The nth digit after the decimal point of a rational number -/
def nth_digit_after_decimal (q : ℚ) (n : ℕ) : ℕ :=
  sorry

theorem fifteenth_digit_is_zero :
  nth_digit_after_decimal sum_fracs 15 = 0 := by sorry

end NUMINAMATH_CALUDE_fifteenth_digit_is_zero_l3247_324719


namespace NUMINAMATH_CALUDE_intercepted_triangle_area_l3247_324708

/-- The region defined by the inequality |x - 1| + |y - 2| ≤ 2 -/
def diamond_region (x y : ℝ) : Prop :=
  abs (x - 1) + abs (y - 2) ≤ 2

/-- The line y = 3x + 1 -/
def intercepting_line (x y : ℝ) : Prop :=
  y = 3 * x + 1

/-- The triangle intercepted by the line from the diamond region -/
def intercepted_triangle (x y : ℝ) : Prop :=
  diamond_region x y ∧ intercepting_line x y

/-- The area of the intercepted triangle -/
noncomputable def triangle_area : ℝ := 2

theorem intercepted_triangle_area :
  triangle_area = 2 :=
sorry

end NUMINAMATH_CALUDE_intercepted_triangle_area_l3247_324708


namespace NUMINAMATH_CALUDE_xiaoying_journey_equations_l3247_324716

/-- Represents Xiaoying's journey to school --/
structure JourneyToSchool where
  totalDistance : ℝ
  totalTime : ℝ
  uphillSpeed : ℝ
  downhillSpeed : ℝ
  uphillTime : ℝ
  downhillTime : ℝ

/-- The system of equations representing Xiaoying's journey --/
def journeyEquations (j : JourneyToSchool) : Prop :=
  (j.uphillSpeed / 60 * j.uphillTime + j.downhillSpeed / 60 * j.downhillTime = j.totalDistance / 1000) ∧
  (j.uphillTime + j.downhillTime = j.totalTime)

/-- Theorem stating that the given conditions satisfy the journey equations --/
theorem xiaoying_journey_equations :
  ∀ (j : JourneyToSchool),
    j.totalDistance = 1200 ∧
    j.totalTime = 16 ∧
    j.uphillSpeed = 3 ∧
    j.downhillSpeed = 5 →
    journeyEquations j :=
by
  sorry

end NUMINAMATH_CALUDE_xiaoying_journey_equations_l3247_324716


namespace NUMINAMATH_CALUDE_square_rectangle_area_problem_l3247_324769

theorem square_rectangle_area_problem :
  ∃ (x₁ x₂ : ℝ),
    (∀ x : ℝ, (x - 3) * (x + 4) = 2 * (x - 2)^2 → x = x₁ ∨ x = x₂) ∧
    x₁ + x₂ = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_problem_l3247_324769


namespace NUMINAMATH_CALUDE_max_visible_sum_l3247_324737

def cube_numbers : List ℕ := [1, 3, 9, 27, 81, 243]

def is_valid_cube (c : List ℕ) : Prop :=
  c.length = 6 ∧ c.toFinset = cube_numbers.toFinset

def visible_sum (bottom middle top : List ℕ) : ℕ :=
  (bottom.take 5).sum + (middle.take 5).sum + (top.take 5).sum

def is_valid_stack (bottom middle top : List ℕ) : Prop :=
  is_valid_cube bottom ∧ is_valid_cube middle ∧ is_valid_cube top

theorem max_visible_sum :
  ∀ bottom middle top : List ℕ,
    is_valid_stack bottom middle top →
    visible_sum bottom middle top ≤ 1087 :=
sorry

end NUMINAMATH_CALUDE_max_visible_sum_l3247_324737


namespace NUMINAMATH_CALUDE_johns_donation_l3247_324784

theorem johns_donation (
  initial_contributions : ℕ) 
  (new_average : ℚ)
  (increase_percentage : ℚ) :
  initial_contributions = 3 →
  new_average = 75 →
  increase_percentage = 50 / 100 →
  ∃ (johns_donation : ℚ),
    johns_donation = 150 ∧
    new_average = (initial_contributions * (new_average / (1 + increase_percentage)) + johns_donation) / (initial_contributions + 1) :=
by sorry

end NUMINAMATH_CALUDE_johns_donation_l3247_324784


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3247_324785

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 16 :=
by
  -- The unique solution is z = -251/4
  use -251/4
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3247_324785


namespace NUMINAMATH_CALUDE_mary_minus_robert_eq_two_l3247_324773

/-- Represents the candy distribution problem -/
structure CandyDistribution where
  total : Nat
  kate : Nat
  robert : Nat
  bill : Nat
  mary : Nat
  kate_pieces : kate = 4
  robert_more_than_kate : robert = kate + 2
  bill_less_than_mary : bill + 6 = mary
  kate_more_than_bill : kate = bill + 2
  mary_more_than_robert : mary > robert

/-- Proves that Mary gets 2 more pieces of candy than Robert -/
theorem mary_minus_robert_eq_two (cd : CandyDistribution) : cd.mary - cd.robert = 2 := by
  sorry

end NUMINAMATH_CALUDE_mary_minus_robert_eq_two_l3247_324773


namespace NUMINAMATH_CALUDE_total_broken_bulbs_to_replace_l3247_324710

/-- Represents the number of broken light bulbs that need to be replaced -/
def broken_bulbs_to_replace (kitchen_bulbs foyer_broken_bulbs living_room_bulbs : ℕ) : ℕ :=
  let kitchen_broken := (3 * kitchen_bulbs) / 5
  let foyer_broken := foyer_broken_bulbs
  let living_room_broken := living_room_bulbs / 2
  kitchen_broken + foyer_broken + living_room_broken

/-- Theorem stating the total number of broken light bulbs to be replaced -/
theorem total_broken_bulbs_to_replace :
  broken_bulbs_to_replace 35 10 24 = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_total_broken_bulbs_to_replace_l3247_324710


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3247_324734

theorem complex_equation_solution (z : ℂ) : z - 1 = (z + 1) * I → z = I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3247_324734


namespace NUMINAMATH_CALUDE_sandy_jessica_marble_ratio_l3247_324777

/-- Proves that Sandy has 4 times more red marbles than Jessica -/
theorem sandy_jessica_marble_ratio :
  let jessica_marbles : ℕ := 3 * 12 -- 3 dozen
  let sandy_marbles : ℕ := 144
  (sandy_marbles : ℚ) / jessica_marbles = 4 := by
  sorry

end NUMINAMATH_CALUDE_sandy_jessica_marble_ratio_l3247_324777


namespace NUMINAMATH_CALUDE_inequality_abc_l3247_324727

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_abc_l3247_324727


namespace NUMINAMATH_CALUDE_even_operations_l3247_324701

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem even_operations (n : ℤ) (h : is_even n) :
  (is_even (n + 4)) ∧ (is_even (n - 6)) ∧ (is_even (n * 8)) := by
  sorry

end NUMINAMATH_CALUDE_even_operations_l3247_324701


namespace NUMINAMATH_CALUDE_directional_vector_for_line_l3247_324707

/-- A directional vector for a line ax + by + c = 0 is a vector (u, v) such that
    for any point (x, y) on the line, (x + u, y + v) is also on the line. -/
def IsDirectionalVector (a b c : ℝ) (u v : ℝ) : Prop :=
  ∀ x y : ℝ, a * x + b * y + c = 0 → a * (x + u) + b * (y + v) + c = 0

/-- The line 2x + 3y - 1 = 0 -/
def Line (x y : ℝ) : Prop := 2 * x + 3 * y - 1 = 0

/-- Theorem: (1, -2/3) is a directional vector for the line 2x + 3y - 1 = 0 -/
theorem directional_vector_for_line :
  IsDirectionalVector 2 3 (-1) 1 (-2/3) :=
sorry

end NUMINAMATH_CALUDE_directional_vector_for_line_l3247_324707


namespace NUMINAMATH_CALUDE_resort_flat_fee_is_40_l3247_324792

/-- Represents the pricing scheme of a resort -/
structure ResortPricing where
  flatFee : ℕ  -- Flat fee for the first night
  additionalNightFee : ℕ  -- Fee for each additional night

/-- Calculates the total cost for a stay -/
def totalCost (pricing : ResortPricing) (nights : ℕ) : ℕ :=
  pricing.flatFee + (nights - 1) * pricing.additionalNightFee

/-- Theorem stating the flat fee given the conditions -/
theorem resort_flat_fee_is_40 :
  ∀ (pricing : ResortPricing),
    totalCost pricing 5 = 320 →
    totalCost pricing 8 = 530 →
    pricing.flatFee = 40 := by
  sorry


end NUMINAMATH_CALUDE_resort_flat_fee_is_40_l3247_324792


namespace NUMINAMATH_CALUDE_trivia_team_tryouts_l3247_324720

theorem trivia_team_tryouts (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) :
  not_picked = 36 →
  groups = 4 →
  students_per_group = 7 →
  not_picked + groups * students_per_group = 64 :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_tryouts_l3247_324720


namespace NUMINAMATH_CALUDE_myrtle_eggs_count_l3247_324705

/-- The number of eggs Myrtle has after her trip -/
def myrtle_eggs : ℕ :=
  let num_hens : ℕ := 3
  let eggs_per_hen_per_day : ℕ := 3
  let days_gone : ℕ := 7
  let neighbor_taken : ℕ := 12
  let dropped : ℕ := 5
  
  let total_laid : ℕ := num_hens * eggs_per_hen_per_day * days_gone
  let remaining_after_neighbor : ℕ := total_laid - neighbor_taken
  remaining_after_neighbor - dropped

theorem myrtle_eggs_count : myrtle_eggs = 46 := by
  sorry

end NUMINAMATH_CALUDE_myrtle_eggs_count_l3247_324705


namespace NUMINAMATH_CALUDE_rabbits_ate_23_pumpkins_l3247_324759

/-- The number of pumpkins Sara initially grew -/
def initial_pumpkins : ℕ := 43

/-- The number of pumpkins Sara has left -/
def remaining_pumpkins : ℕ := 20

/-- The number of pumpkins eaten by rabbits -/
def eaten_pumpkins : ℕ := initial_pumpkins - remaining_pumpkins

theorem rabbits_ate_23_pumpkins : eaten_pumpkins = 23 := by
  sorry

end NUMINAMATH_CALUDE_rabbits_ate_23_pumpkins_l3247_324759


namespace NUMINAMATH_CALUDE_circle_equation_solution_l3247_324748

theorem circle_equation_solution :
  ∃! (x y : ℝ), (x - 13)^2 + (y - 14)^2 + (x - y)^2 = 1/3 ∧ x = 40/3 ∧ y = 41/3 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_solution_l3247_324748


namespace NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_72_l3247_324715

/-- The sum of N consecutive integers starting from a -/
def sumConsecutiveIntegers (N : ℕ) (a : ℤ) : ℤ := N * (2 * a + N - 1) / 2

/-- The proposition that 144 is the greatest number of consecutive integers summing to 72 -/
theorem greatest_consecutive_integers_sum_72 :
  ∀ N : ℕ, (∃ a : ℤ, sumConsecutiveIntegers N a = 72) → N ≤ 144 :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_integers_sum_72_l3247_324715


namespace NUMINAMATH_CALUDE_circle_diameter_l3247_324717

theorem circle_diameter (C : ℝ) (h : C = 36) : 
  (C / π) = (36 : ℝ) / π := by sorry

end NUMINAMATH_CALUDE_circle_diameter_l3247_324717


namespace NUMINAMATH_CALUDE_exact_time_proof_l3247_324721

def minutes_after_3 (h m : ℕ) : ℝ := 60 * (h - 3 : ℝ) + m

def minute_hand_position (t : ℝ) : ℝ := 6 * t

def hour_hand_position (t : ℝ) : ℝ := 90 + 0.5 * t

theorem exact_time_proof :
  ∃ (h m : ℕ), h = 3 ∧ m < 60 ∧
  let t := minutes_after_3 h m
  abs (minute_hand_position (t + 5) - hour_hand_position (t - 4)) = 178 ∧
  h = 3 ∧ m = 43 := by
  sorry

end NUMINAMATH_CALUDE_exact_time_proof_l3247_324721


namespace NUMINAMATH_CALUDE_probability_five_or_joker_l3247_324763

/-- A deck of cards with jokers -/
structure DeckWithJokers where
  standardCards : ℕ
  jokers : ℕ
  totalCards : ℕ
  total_is_sum : totalCards = standardCards + jokers

/-- The probability of drawing a specific card or a joker -/
def drawProbability (d : DeckWithJokers) (specificCards : ℕ) : ℚ :=
  (specificCards + d.jokers : ℚ) / d.totalCards

/-- The deck described in the problem -/
def problemDeck : DeckWithJokers where
  standardCards := 52
  jokers := 2
  totalCards := 54
  total_is_sum := by rfl

theorem probability_five_or_joker :
  drawProbability problemDeck 4 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_or_joker_l3247_324763


namespace NUMINAMATH_CALUDE_tangent_line_inclination_l3247_324732

theorem tangent_line_inclination (a : ℝ) : 
  (∀ x : ℝ, (fun x => a * x^3 - 2) x = a * x^3 - 2) →
  (slope_at_neg_one : ℝ) →
  slope_at_neg_one = Real.tan (π / 4) →
  slope_at_neg_one = (fun x => 3 * a * x^2) (-1) →
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_l3247_324732


namespace NUMINAMATH_CALUDE_surfer_ratio_l3247_324741

/-- Proves that the ratio of surfers on Malibu beach to Santa Monica beach is 2:1 -/
theorem surfer_ratio :
  ∀ (malibu santa_monica : ℕ),
  santa_monica = 20 →
  malibu + santa_monica = 60 →
  (malibu : ℚ) / santa_monica = 2 / 1 := by
sorry

end NUMINAMATH_CALUDE_surfer_ratio_l3247_324741


namespace NUMINAMATH_CALUDE_sum_of_fractions_l3247_324796

theorem sum_of_fractions : 
  (1 / (2 * 3 : ℚ)) + (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + 
  (1 / (5 * 6 : ℚ)) + (1 / (6 * 7 : ℚ)) = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l3247_324796


namespace NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l3247_324786

theorem binomial_coeff_not_coprime (n k l : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : 1 ≤ l) (h4 : l < n) :
  Nat.gcd (Nat.choose n k) (Nat.choose n l) > 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l3247_324786


namespace NUMINAMATH_CALUDE_inequality_proof_l3247_324799

theorem inequality_proof (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a*c + b*d)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3247_324799


namespace NUMINAMATH_CALUDE_total_pushups_count_l3247_324744

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 44

/-- The additional number of push-ups David did compared to Zachary -/
def david_extra_pushups : ℕ := 58

/-- The total number of push-ups done by Zachary and David -/
def total_pushups : ℕ := zachary_pushups + (zachary_pushups + david_extra_pushups)

/-- Theorem stating the total number of push-ups done by Zachary and David -/
theorem total_pushups_count : total_pushups = 146 := by
  sorry

end NUMINAMATH_CALUDE_total_pushups_count_l3247_324744


namespace NUMINAMATH_CALUDE_line_parallel_to_skew_line_l3247_324718

/-- Represents a line in 3D space -/
structure Line3D where
  -- Definition of a line in 3D space
  -- (We'll leave this abstract for simplicity)

/-- Two lines are skew if they are not coplanar -/
def are_skew (l1 l2 : Line3D) : Prop :=
  -- Definition of skew lines
  sorry

/-- Two lines are parallel if they have the same direction -/
def are_parallel (l1 l2 : Line3D) : Prop :=
  -- Definition of parallel lines
  sorry

/-- Two lines intersect if they have a common point -/
def intersect (l1 l2 : Line3D) : Prop :=
  -- Definition of intersecting lines
  sorry

theorem line_parallel_to_skew_line (l1 l2 l3 : Line3D) 
  (h1 : are_skew l1 l2) 
  (h2 : are_parallel l3 l1) : 
  intersect l3 l2 ∨ are_skew l3 l2 :=
by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_skew_line_l3247_324718


namespace NUMINAMATH_CALUDE_subset_of_any_set_implies_zero_l3247_324770

theorem subset_of_any_set_implies_zero (a : ℝ) :
  (∀ S : Set ℝ, {x : ℝ | a * x = 1} ⊆ S) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_subset_of_any_set_implies_zero_l3247_324770


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3247_324728

theorem arithmetic_sequence_problem (x : ℚ) :
  let a₁ := 3 * x - 4
  let a₂ := 6 * x - 14
  let a₃ := 4 * x + 2
  let d := a₂ - a₁  -- common difference
  let a_n (n : ℕ) := a₁ + (n - 1) * d  -- general term
  ∃ n : ℕ, a_n n = 4018 ∧ n = 716 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3247_324728


namespace NUMINAMATH_CALUDE_table_runner_coverage_l3247_324765

theorem table_runner_coverage (runners : Nat) 
  (area_first_three : ℝ) (area_last_two : ℝ) (table_area : ℝ) 
  (coverage_percentage : ℝ) (two_layer_area : ℝ) (one_layer_area : ℝ) :
  runners = 5 →
  area_first_three = 324 →
  area_last_two = 216 →
  table_area = 320 →
  coverage_percentage = 0.75 →
  two_layer_area = 36 →
  one_layer_area = 48 →
  ∃ (three_layer_area : ℝ),
    three_layer_area = 156 ∧
    coverage_percentage * table_area = one_layer_area + two_layer_area + three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l3247_324765


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_l3247_324793

theorem arithmetic_sequence_constant (x y z k : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x ≠ y → y ≠ z → x ≠ z →
  k ≠ 1 →
  k * x = y →
  let u := y / x
  let v := z / y
  (u - 1/v) - (v - 1/u) = (v - 1/u) - (1/u - u) →
  ∃ (k' : ℝ), k' * x = z ∧ 2 * k / k' - 2 * k + k^2 / k' - 1 / k = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_l3247_324793


namespace NUMINAMATH_CALUDE_equidistant_function_property_l3247_324736

def f (a b : ℝ) (z : ℂ) : ℂ := (Complex.mk a b) * z

theorem equidistant_function_property (a b : ℝ) :
  (∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z)) →
  Complex.abs (Complex.mk a b) = 5 →
  b^2 = 99/4 := by sorry

end NUMINAMATH_CALUDE_equidistant_function_property_l3247_324736


namespace NUMINAMATH_CALUDE_cleos_marbles_eq_15_l3247_324768

/-- The number of marbles Cleo has on the third day -/
def cleos_marbles : ℕ :=
  let initial_marbles : ℕ := 30
  let marbles_taken_day2 : ℕ := (3 * initial_marbles) / 5
  let marbles_each_day2 : ℕ := marbles_taken_day2 / 2
  let marbles_remaining_day2 : ℕ := initial_marbles - marbles_taken_day2
  let marbles_taken_day3 : ℕ := marbles_remaining_day2 / 2
  marbles_each_day2 + marbles_taken_day3

theorem cleos_marbles_eq_15 : cleos_marbles = 15 := by
  sorry

end NUMINAMATH_CALUDE_cleos_marbles_eq_15_l3247_324768


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3247_324761

theorem inequality_solution_set (x : ℝ) : 
  (2 / (x + 2) + 5 / (x + 4) ≤ 2 - x) ↔ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3247_324761


namespace NUMINAMATH_CALUDE_grassland_area_l3247_324706

theorem grassland_area (width1 : ℝ) (length : ℝ) : 
  width1 > 0 → length > 0 →
  (width1 + 10) * length = 1000 →
  (width1 - 4) * length = 650 →
  width1 * length = 750 := by
sorry

end NUMINAMATH_CALUDE_grassland_area_l3247_324706


namespace NUMINAMATH_CALUDE_max_intersection_points_l3247_324713

/-- Represents a line segment -/
structure Segment where
  id : ℕ

/-- Represents an intersection point -/
structure IntersectionPoint where
  id : ℕ

/-- The set of all segments -/
def segments : Finset Segment :=
  sorry

/-- The set of all intersection points -/
def intersectionPoints : Finset IntersectionPoint :=
  sorry

/-- Function that returns the number of intersections for a given segment -/
def intersectionsForSegment (s : Segment) : ℕ :=
  sorry

/-- Theorem stating the maximum number of intersection points -/
theorem max_intersection_points :
  (segments.card = 10) →
  (∀ s ∈ segments, intersectionsForSegment s = 3) →
  intersectionPoints.card ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_max_intersection_points_l3247_324713


namespace NUMINAMATH_CALUDE_arccos_one_half_l3247_324775

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by sorry

end NUMINAMATH_CALUDE_arccos_one_half_l3247_324775


namespace NUMINAMATH_CALUDE_multiplication_exercise_l3247_324760

theorem multiplication_exercise (a b : ℕ+) 
  (h1 : (a + 6) * b = 255)  -- Units digit changed from 1 to 7
  (h2 : (a - 10) * b = 335) -- Tens digit changed from 6 to 5
  : a * b = 285 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_exercise_l3247_324760


namespace NUMINAMATH_CALUDE_expression_evaluation_l3247_324767

theorem expression_evaluation : 
  Real.sqrt 3 * Real.cos (30 * π / 180) + (3 - π)^0 - 2 * Real.tan (45 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3247_324767


namespace NUMINAMATH_CALUDE_f_of_5_eq_19_l3247_324739

/-- Given f(x) = (7x + 3) / (x - 3), prove that f(5) = 19 -/
theorem f_of_5_eq_19 : 
  let f : ℝ → ℝ := λ x ↦ (7 * x + 3) / (x - 3)
  f 5 = 19 := by
  sorry

end NUMINAMATH_CALUDE_f_of_5_eq_19_l3247_324739


namespace NUMINAMATH_CALUDE_exists_same_acquaintance_count_exists_no_three_same_acquaintance_count_l3247_324798

/-- Represents a meeting with participants and their acquaintances -/
structure Meeting where
  participants : Finset ℕ
  acquaintances : ℕ → Finset ℕ
  valid : ∀ i ∈ participants, acquaintances i ⊆ participants ∧ i ∉ acquaintances i

/-- There exist at least two participants with the same number of acquaintances -/
theorem exists_same_acquaintance_count (m : Meeting) (h : 1 < m.participants.card) :
  ∃ i j, i ∈ m.participants ∧ j ∈ m.participants ∧ i ≠ j ∧
    (m.acquaintances i).card = (m.acquaintances j).card :=
  sorry

/-- There exists an arrangement of acquaintances such that no three participants have the same number of acquaintances -/
theorem exists_no_three_same_acquaintance_count (n : ℕ) (h : 1 < n) :
  ∃ m : Meeting, m.participants.card = n ∧
    ∀ i j k, i ∈ m.participants → j ∈ m.participants → k ∈ m.participants →
      i ≠ j → j ≠ k → i ≠ k →
        (m.acquaintances i).card ≠ (m.acquaintances j).card ∨
        (m.acquaintances j).card ≠ (m.acquaintances k).card ∨
        (m.acquaintances i).card ≠ (m.acquaintances k).card :=
  sorry

end NUMINAMATH_CALUDE_exists_same_acquaintance_count_exists_no_three_same_acquaintance_count_l3247_324798


namespace NUMINAMATH_CALUDE_john_pills_per_week_l3247_324779

/-- The number of pills John takes in a week -/
def pills_per_week (hours_between_pills : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  (hours_per_day / hours_between_pills) * days_per_week

/-- Theorem: John takes 28 pills in a week -/
theorem john_pills_per_week : 
  pills_per_week 6 24 7 = 28 := by
  sorry

end NUMINAMATH_CALUDE_john_pills_per_week_l3247_324779


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l3247_324729

theorem cubic_roots_sum (m : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2024*x + m = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 104 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l3247_324729
