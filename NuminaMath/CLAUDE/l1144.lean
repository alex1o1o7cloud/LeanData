import Mathlib

namespace NUMINAMATH_CALUDE_batsman_average_after_20th_innings_l1144_114432

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: Given the conditions, prove that the new average is 92 -/
theorem batsman_average_after_20th_innings 
  (stats : BatsmanStats)
  (h1 : stats.innings = 19)
  (h2 : newAverage stats 130 = stats.average + 2)
  : newAverage stats 130 = 92 := by
  sorry

#check batsman_average_after_20th_innings

end NUMINAMATH_CALUDE_batsman_average_after_20th_innings_l1144_114432


namespace NUMINAMATH_CALUDE_expression_value_l1144_114417

theorem expression_value (a b c d : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) : 
  (a + b + c * d) + (a + b) / (c * d) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1144_114417


namespace NUMINAMATH_CALUDE_carrie_phone_trade_in_l1144_114410

/-- The trade-in value of Carrie's old phone -/
def trade_in_value : ℕ := sorry

/-- The cost of the new iPhone -/
def iphone_cost : ℕ := 800

/-- Carrie's weekly earnings from babysitting -/
def weekly_earnings : ℕ := 80

/-- The number of weeks Carrie has to work -/
def weeks_to_work : ℕ := 7

/-- The total amount Carrie earns from babysitting -/
def total_earnings : ℕ := weekly_earnings * weeks_to_work

theorem carrie_phone_trade_in :
  trade_in_value = iphone_cost - total_earnings :=
sorry

end NUMINAMATH_CALUDE_carrie_phone_trade_in_l1144_114410


namespace NUMINAMATH_CALUDE_basketball_games_played_l1144_114437

theorem basketball_games_played (team_a_win_ratio : Rat) (team_b_win_ratio : Rat)
  (team_b_more_wins : ℕ) (team_b_more_losses : ℕ) :
  team_a_win_ratio = 3/4 →
  team_b_win_ratio = 2/3 →
  team_b_more_wins = 9 →
  team_b_more_losses = 9 →
  ∃ (team_a_games : ℕ),
    team_a_games = 36 ∧
    (team_a_games : Rat) * team_a_win_ratio + (team_a_games : Rat) * (1 - team_a_win_ratio) = team_a_games ∧
    ((team_a_games : Rat) + (team_b_more_wins + team_b_more_losses : Rat)) * team_b_win_ratio = 
      team_a_games * team_a_win_ratio + team_b_more_wins :=
by sorry

end NUMINAMATH_CALUDE_basketball_games_played_l1144_114437


namespace NUMINAMATH_CALUDE_constant_value_l1144_114484

theorem constant_value (t : ℝ) (x y : ℝ → ℝ) (c : ℝ) :
  (∀ t, x t = c - 4 * t) →
  (∀ t, y t = 2 * t - 2) →
  x 0.5 = y 0.5 →
  c = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_value_l1144_114484


namespace NUMINAMATH_CALUDE_count_valid_sequences_l1144_114499

/-- The set of digits to be used -/
def Digits : Finset Nat := {0, 1, 2, 3, 4}

/-- A function to check if a number is even -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- A function to check if a digit sequence satisfies the condition -/
def validSequence (seq : List Nat) : Bool :=
  seq.length = 5 ∧ 
  seq.toFinset = Digits ∧
  ∃ i, i ∈ [1, 2, 3] ∧ 
    isEven (seq.nthLe i sorry) ∧ 
    ¬isEven (seq.nthLe (i-1) sorry) ∧ 
    ¬isEven (seq.nthLe (i+1) sorry)

/-- The main theorem -/
theorem count_valid_sequences : 
  (Digits.toList.permutations.filter validSequence).length = 28 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_sequences_l1144_114499


namespace NUMINAMATH_CALUDE_batsman_average_after_12_innings_l1144_114431

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  lastInningsScore : Nat
  averageIncrease : Nat

/-- Calculates the average score after a given number of innings -/
def calculateAverage (stats : BatsmanStats) : Nat :=
  (stats.totalRuns) / (stats.innings)

/-- Theorem stating the batsman's average after 12 innings -/
theorem batsman_average_after_12_innings 
  (stats : BatsmanStats)
  (h1 : stats.innings = 12)
  (h2 : stats.lastInningsScore = 48)
  (h3 : stats.averageIncrease = 2)
  (h4 : calculateAverage stats = calculateAverage { stats with 
    innings := stats.innings - 1, 
    totalRuns := stats.totalRuns - stats.lastInningsScore 
  } + stats.averageIncrease) :
  calculateAverage stats = 26 := by
  sorry

#check batsman_average_after_12_innings

end NUMINAMATH_CALUDE_batsman_average_after_12_innings_l1144_114431


namespace NUMINAMATH_CALUDE_ratio_lcm_problem_l1144_114404

theorem ratio_lcm_problem (a b : ℕ+) (h1 : a.val * 4 = b.val * 3) 
  (h2 : Nat.lcm a.val b.val = 180) (h3 : a.val = 45 ∨ b.val = 45) :
  (if a.val = 45 then b.val else a.val) = 60 := by
  sorry

end NUMINAMATH_CALUDE_ratio_lcm_problem_l1144_114404


namespace NUMINAMATH_CALUDE_two_special_numbers_exist_l1144_114419

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def has_no_single_digit_prime_factors (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ n → p > 9

theorem two_special_numbers_exist : ∃ x y : ℕ,
  x + y = 173717 ∧
  is_four_digit (x - y) ∧
  has_no_single_digit_prime_factors (x - y) ∧
  (1558 ∣ x ∨ 1558 ∣ y) ∧
  x = 91143 ∧ y = 82574 := by
  sorry

end NUMINAMATH_CALUDE_two_special_numbers_exist_l1144_114419


namespace NUMINAMATH_CALUDE_no_valid_base_for_450_l1144_114452

def is_four_digit (n : ℕ) (b : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

def last_digit (n : ℕ) (b : ℕ) : ℕ :=
  n % b

theorem no_valid_base_for_450 :
  ¬ ∃ (b : ℕ), b > 1 ∧ is_four_digit 450 b ∧ Odd (last_digit 450 b) :=
sorry

end NUMINAMATH_CALUDE_no_valid_base_for_450_l1144_114452


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l1144_114448

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - x - 6 ≥ 0}
def B : Set ℝ := {x | (1 - x) / (x - 3) ≥ 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ 1 ∨ x ≤ -3/2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l1144_114448


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l1144_114436

/-- The volume of a cube inscribed in a cylinder, which is inscribed in a larger cube --/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let cylinder_radius : ℝ := outer_cube_edge / 2
  let cylinder_diameter : ℝ := outer_cube_edge
  let inscribed_cube_face_diagonal : ℝ := cylinder_diameter
  let inscribed_cube_edge : ℝ := inscribed_cube_face_diagonal / Real.sqrt 2
  let inscribed_cube_volume : ℝ := inscribed_cube_edge ^ 3
  inscribed_cube_volume = 432 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l1144_114436


namespace NUMINAMATH_CALUDE_chlorine_and_hcl_moles_l1144_114427

/-- Represents the stoichiometric coefficients of the chemical reaction:
    C2H6 + 6Cl2 → C2Cl6 + 6HCl -/
structure ReactionCoefficients where
  ethane : ℕ
  chlorine : ℕ
  hexachloroethane : ℕ
  hydrochloric_acid : ℕ

/-- The given chemical reaction -/
def reaction : ReactionCoefficients :=
  { ethane := 1
  , chlorine := 6
  , hexachloroethane := 1
  , hydrochloric_acid := 6 }

/-- The number of moles of ethane given -/
def ethane_moles : ℕ := 3

/-- Theorem stating the number of moles of chlorine required and hydrochloric acid formed -/
theorem chlorine_and_hcl_moles :
  (ethane_moles * reaction.chlorine = 18) ∧
  (ethane_moles * reaction.hydrochloric_acid = 18) := by
  sorry

end NUMINAMATH_CALUDE_chlorine_and_hcl_moles_l1144_114427


namespace NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1144_114416

theorem quadratic_equation_equivalence :
  ∃ (r : ℝ), ∀ (x : ℝ), (4 * x^2 - 8 * x - 288 = 0) ↔ ((x + r)^2 = 73) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_equivalence_l1144_114416


namespace NUMINAMATH_CALUDE_triangle_area_comparison_l1144_114444

-- Define the triangles
def Triangle := Fin 3 → ℝ × ℝ

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- Define the side length between two points of a triangle
def side_length (t : Triangle) (i k : Fin 3) : ℝ := sorry

-- Define if a triangle is obtuse-angled
def is_obtuse (t : Triangle) : Prop := sorry

-- Theorem statement
theorem triangle_area_comparison 
  (A B : Triangle) 
  (h_sides : ∀ (i k : Fin 3), side_length A i k ≥ side_length B i k) 
  (h_not_obtuse : ¬ is_obtuse A) : 
  area A ≥ area B := by sorry

end NUMINAMATH_CALUDE_triangle_area_comparison_l1144_114444


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1144_114494

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 22*x^2 + 80*x - 67

-- Define the roots
variables (p q r : ℝ)

-- Define A, B, C
variables (A B C : ℝ)

-- Axioms
axiom distinct_roots : p ≠ q ∧ q ≠ r ∧ p ≠ r
axiom roots : f p = 0 ∧ f q = 0 ∧ f r = 0

axiom partial_fraction_decomposition :
  ∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    1 / (s^3 - 22*s^2 + 80*s - 67) = A / (s - p) + B / (s - q) + C / (s - r)

-- Theorem to prove
theorem sum_of_reciprocals : 1/A + 1/B + 1/C = 244 := by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1144_114494


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1144_114430

-- Define an even function f: ℝ → ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define increasing function on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Main theorem
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_increasing : increasing_on_nonneg f)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f (x + 1) > 0} = Set.Ioo (-1 : ℝ) 1 := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1144_114430


namespace NUMINAMATH_CALUDE_fourth_largest_divisor_of_n_l1144_114450

def n : ℕ := 1000800000

def fourth_largest_divisor (m : ℕ) : ℕ := sorry

theorem fourth_largest_divisor_of_n :
  fourth_largest_divisor n = 62550000 := by sorry

end NUMINAMATH_CALUDE_fourth_largest_divisor_of_n_l1144_114450


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l1144_114445

theorem quadratic_no_real_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k - 1 ≠ 0) → k > 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l1144_114445


namespace NUMINAMATH_CALUDE_systematic_sample_fourth_member_l1144_114492

def systematic_sample (total : ℕ) (sample_size : ℕ) (known_members : List ℕ) : Prop :=
  ∃ (start : ℕ) (k : ℕ),
    k = total / sample_size ∧
    ∀ (i : ℕ), i < sample_size →
      (start + i * k) % total + 1 ∈ known_members ∪ {(start + (sample_size - 1) * k) % total + 1}

theorem systematic_sample_fourth_member 
  (total : ℕ) (sample_size : ℕ) (known_members : List ℕ) 
  (h_total : total = 52)
  (h_sample_size : sample_size = 4)
  (h_known_members : known_members = [6, 32, 45]) :
  systematic_sample total sample_size known_members →
  (19 : ℕ) ∈ known_members ∪ {19} :=
by sorry

end NUMINAMATH_CALUDE_systematic_sample_fourth_member_l1144_114492


namespace NUMINAMATH_CALUDE_cauchy_schwarz_inequality_3d_l1144_114475

theorem cauchy_schwarz_inequality_3d (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ) :
  (x₁ * x₂ + y₁ * y₂ + z₁ * z₂)^2 ≤ (x₁^2 + y₁^2 + z₁^2) * (x₂^2 + y₂^2 + z₂^2) := by
  sorry

end NUMINAMATH_CALUDE_cauchy_schwarz_inequality_3d_l1144_114475


namespace NUMINAMATH_CALUDE_cantaloupes_left_l1144_114455

/-- The number of cantaloupes left after growing and losing some due to bad weather -/
theorem cantaloupes_left (fred tim maria lost : ℕ) (h1 : fred = 38) (h2 : tim = 44) (h3 : maria = 57) (h4 : lost = 12) :
  fred + tim + maria - lost = 127 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupes_left_l1144_114455


namespace NUMINAMATH_CALUDE_sugar_purchase_efficiency_l1144_114474

/-- Proves that Xiao Li's method of buying sugar is more cost-effective than Xiao Wang's --/
theorem sugar_purchase_efficiency
  (n : ℕ) (a : ℕ → ℝ)
  (h_n : n > 1)
  (h_a : ∀ i, i ∈ Finset.range n → a i > 0) :
  (Finset.sum (Finset.range n) a) / n ≥ n / (Finset.sum (Finset.range n) (λ i => 1 / a i)) :=
by sorry

end NUMINAMATH_CALUDE_sugar_purchase_efficiency_l1144_114474


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1144_114460

/-- An isosceles triangle with sides of 4 cm and 7 cm has a perimeter of either 15 cm or 18 cm. -/
theorem isosceles_triangle_perimeter (a b c : ℝ) : 
  a = 4 ∧ b = 7 ∧ 
  ((a = b ∧ c = 7) ∨ (a = c ∧ b = 7) ∨ (b = c ∧ a = 4)) → 
  (a + b + c = 15 ∨ a + b + c = 18) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1144_114460


namespace NUMINAMATH_CALUDE_A_value_l1144_114412

noncomputable def A (m n : ℝ) : ℝ :=
  (((4 * m^2 * n^2) / (4 * m * n - m^2 - 4 * n^2) -
    (2 + n / m + m / n) / (4 / (m * n) - 1 / n^2 - 4 / m^2))^(1/2)) *
  (Real.sqrt (m * n) / (m - 2 * n))

theorem A_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  A m n = if 1 < m / n ∧ m / n < 2 then n - m else m - n := by
  sorry

end NUMINAMATH_CALUDE_A_value_l1144_114412


namespace NUMINAMATH_CALUDE_hair_cut_length_l1144_114409

/-- Given Isabella's original and current hair lengths, prove the length of hair cut off. -/
theorem hair_cut_length (original_length current_length cut_length : ℕ) : 
  original_length = 18 → current_length = 9 → cut_length = original_length - current_length :=
by sorry

end NUMINAMATH_CALUDE_hair_cut_length_l1144_114409


namespace NUMINAMATH_CALUDE_k_range_for_two_solutions_l1144_114488

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x

noncomputable def g (x : ℝ) : ℝ := (log x) / x

theorem k_range_for_two_solutions (k : ℝ) :
  (∃ x y, x ∈ Set.Icc (1/ℯ) ℯ ∧ y ∈ Set.Icc (1/ℯ) ℯ ∧ x ≠ y ∧ f k x = g x ∧ f k y = g y) →
  k ∈ Set.Ioo (1/ℯ^2) (1/(2*ℯ)) :=
sorry

end NUMINAMATH_CALUDE_k_range_for_two_solutions_l1144_114488


namespace NUMINAMATH_CALUDE_bill_donut_purchase_l1144_114487

/-- The number of ways to distribute donuts among types with constraints -/
def donut_combinations (total_donuts : ℕ) (num_types : ℕ) (min_types : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the specific case for Bill's donut purchase -/
theorem bill_donut_purchase :
  donut_combinations 8 5 4 = 425 :=
sorry

end NUMINAMATH_CALUDE_bill_donut_purchase_l1144_114487


namespace NUMINAMATH_CALUDE_selection_methods_count_l1144_114476

/-- The number of different ways to select one teacher and one student -/
def selection_methods (num_teachers : ℕ) (num_male_students : ℕ) (num_female_students : ℕ) : ℕ :=
  num_teachers * (num_male_students + num_female_students)

/-- Theorem stating that the number of selection methods for the given problem is 39 -/
theorem selection_methods_count :
  selection_methods 3 8 5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_selection_methods_count_l1144_114476


namespace NUMINAMATH_CALUDE_arman_second_week_hours_l1144_114470

/-- Calculates the number of hours worked in the second week given the conditions of Arman's work schedule and earnings. -/
def hours_worked_second_week (
  first_week_hours : ℕ)
  (first_week_rate : ℚ)
  (rate_increase : ℚ)
  (total_earnings : ℚ) : ℚ :=
  let first_week_earnings := first_week_hours * first_week_rate
  let second_week_earnings := total_earnings - first_week_earnings
  let new_rate := first_week_rate + rate_increase
  second_week_earnings / new_rate

/-- Theorem stating that given the conditions of Arman's work schedule and earnings, 
    the number of hours worked in the second week is 40. -/
theorem arman_second_week_hours :
  hours_worked_second_week 35 10 0.5 770 = 40 := by
  sorry

end NUMINAMATH_CALUDE_arman_second_week_hours_l1144_114470


namespace NUMINAMATH_CALUDE_rectangular_window_width_l1144_114426

/-- Represents the width of a rectangular window with specific pane arrangements and dimensions. -/
def window_width (pane_width : ℝ) : ℝ :=
  3 * pane_width + 4  -- 3 panes across plus 4 borders

/-- Theorem stating the width of the rectangular window under given conditions. -/
theorem rectangular_window_width :
  ∃ (pane_width : ℝ),
    pane_width > 0 ∧
    (3 : ℝ) / 4 * pane_width = 3 / 4 * pane_width ∧  -- height-to-width ratio of 3:4
    window_width pane_width = 28 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_window_width_l1144_114426


namespace NUMINAMATH_CALUDE_abs_geq_ax_implies_a_in_range_l1144_114473

theorem abs_geq_ax_implies_a_in_range (a : ℝ) :
  (∀ x : ℝ, |x| ≥ a * x) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_geq_ax_implies_a_in_range_l1144_114473


namespace NUMINAMATH_CALUDE_binomial_floor_divisibility_l1144_114443

theorem binomial_floor_divisibility (p n : ℕ) (hp : Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - n / p) := by
  sorry

end NUMINAMATH_CALUDE_binomial_floor_divisibility_l1144_114443


namespace NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l1144_114449

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1 : ℤ) * d) / 2

/-- The value of n that minimizes the sum of the first n terms -/
def minimizing_n (a₁ d : ℤ) : Set ℕ :=
  {n : ℕ | ∀ m : ℕ, arithmetic_sum a₁ d n ≤ arithmetic_sum a₁ d m}

theorem arithmetic_sequence_min_sum :
  minimizing_n (-28) 4 = {7, 8} := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_min_sum_l1144_114449


namespace NUMINAMATH_CALUDE_expression_evaluation_l1144_114463

theorem expression_evaluation : 18 * (150 / 3 + 36 / 6 + 16 / 32 + 2) = 1053 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1144_114463


namespace NUMINAMATH_CALUDE_movies_on_shelves_l1144_114439

theorem movies_on_shelves (total_movies : ℕ) (num_shelves : ℕ) (h1 : total_movies = 999) (h2 : num_shelves = 5) :
  ∃ (additional_movies : ℕ), 
    additional_movies = 1 ∧ 
    (total_movies + additional_movies) % num_shelves = 0 :=
by sorry

end NUMINAMATH_CALUDE_movies_on_shelves_l1144_114439


namespace NUMINAMATH_CALUDE_quadratic_equation_with_given_root_properties_l1144_114481

theorem quadratic_equation_with_given_root_properties :
  ∀ (a b c p q : ℝ),
    a ≠ 0 →
    (∀ x, a * x^2 + b * x + c = 0 ↔ x = p ∨ x = q) →
    p + q = 12 →
    |p - q| = 4 →
    a * x^2 + b * x + c = x^2 - 12 * x + 32 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_given_root_properties_l1144_114481


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l1144_114446

-- Define the types for points and circles
variable (Point Circle : Type)
-- Define the predicate for a point lying on a circle
variable (lies_on : Point → Circle → Prop)
-- Define the predicate for two circles intersecting
variable (intersect : Circle → Circle → Prop)
-- Define the predicate for a circle being tangent to another circle
variable (tangent : Circle → Circle → Prop)
-- Define the predicate for a point being the intersection of a line and a circle
variable (line_circle_intersection : Point → Point → Circle → Point → Prop)
-- Define the predicate for four points being concyclic
variable (concyclic : Point → Point → Point → Point → Prop)

-- State the theorem
theorem circle_intersection_theorem 
  (Γ₁ Γ₂ Γ : Circle) 
  (A B C D E F G H I : Point) :
  intersect Γ₁ Γ₂ →
  lies_on A Γ₁ ∧ lies_on A Γ₂ →
  lies_on B Γ₁ ∧ lies_on B Γ₂ →
  tangent Γ Γ₁ ∧ tangent Γ Γ₂ →
  lies_on D Γ ∧ lies_on D Γ₁ →
  lies_on E Γ ∧ lies_on E Γ₂ →
  line_circle_intersection A B Γ C →
  line_circle_intersection E C Γ₂ F →
  line_circle_intersection D C Γ₁ G →
  line_circle_intersection E D Γ₁ H →
  line_circle_intersection E D Γ₂ I →
  concyclic F G H I := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l1144_114446


namespace NUMINAMATH_CALUDE_oldest_child_age_l1144_114415

theorem oldest_child_age (average_age : ℝ) (age1 age2 age3 : ℕ) :
  average_age = 9 ∧ age1 = 5 ∧ age2 = 8 ∧ age3 = 11 →
  ∃ (age4 : ℕ), (age1 + age2 + age3 + age4 : ℝ) / 4 = average_age ∧ age4 = 12 :=
by sorry

end NUMINAMATH_CALUDE_oldest_child_age_l1144_114415


namespace NUMINAMATH_CALUDE_correct_stratified_sampling_l1144_114466

/-- Represents the number of students in each year --/
structure StudentCounts where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ

/-- Calculates the stratified sample size for a given year --/
def stratifiedSampleSize (yearCount : ℕ) (totalCount : ℕ) (sampleSize : ℕ) : ℕ :=
  (yearCount * sampleSize + totalCount - 1) / totalCount

/-- Theorem stating the correct stratified sampling for the given problem --/
theorem correct_stratified_sampling (totalStudents : StudentCounts) 
    (h1 : totalStudents.firstYear = 540)
    (h2 : totalStudents.secondYear = 440)
    (h3 : totalStudents.thirdYear = 420)
    (totalSampleSize : ℕ) 
    (h4 : totalSampleSize = 70) :
  let totalCount := totalStudents.firstYear + totalStudents.secondYear + totalStudents.thirdYear
  (stratifiedSampleSize totalStudents.firstYear totalCount totalSampleSize,
   stratifiedSampleSize totalStudents.secondYear totalCount totalSampleSize,
   stratifiedSampleSize totalStudents.thirdYear totalCount totalSampleSize) = (27, 22, 21) := by
  sorry

end NUMINAMATH_CALUDE_correct_stratified_sampling_l1144_114466


namespace NUMINAMATH_CALUDE_difference_of_expressions_l1144_114490

theorem difference_of_expressions : 
  (Real.sqrt (0.9 * 40) - (4/5 * (2/3 * 25))) = -22/3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_expressions_l1144_114490


namespace NUMINAMATH_CALUDE_jackson_decorations_given_l1144_114478

/-- The number of decorations given to the neighbor -/
def decorations_given_to_neighbor (num_boxes : ℕ) (decorations_per_box : ℕ) (decorations_used : ℕ) : ℕ :=
  num_boxes * decorations_per_box - decorations_used

/-- Theorem: Mrs. Jackson gave 92 decorations to her neighbor -/
theorem jackson_decorations_given :
  decorations_given_to_neighbor 6 25 58 = 92 := by
  sorry

end NUMINAMATH_CALUDE_jackson_decorations_given_l1144_114478


namespace NUMINAMATH_CALUDE_f_nonnegative_range_l1144_114459

def f (a x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 2

theorem f_nonnegative_range (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 3, f a x ≥ 0) →
  1/6 ≤ a ∧ a ≤ 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_f_nonnegative_range_l1144_114459


namespace NUMINAMATH_CALUDE_geometry_test_passing_l1144_114477

theorem geometry_test_passing (total_problems : ℕ) (passing_percentage : ℚ) 
  (hp : passing_percentage = 85 / 100) (ht : total_problems = 50) :
  ∃ (max_missable : ℕ), 
    (max_missable : ℚ) / total_problems ≤ 1 - passing_percentage ∧
    ∀ (n : ℕ), (n : ℚ) / total_problems ≤ 1 - passing_percentage → n ≤ max_missable :=
by sorry

end NUMINAMATH_CALUDE_geometry_test_passing_l1144_114477


namespace NUMINAMATH_CALUDE_expression_evaluation_l1144_114414

theorem expression_evaluation (x : ℝ) (h : x > 2) :
  Real.sqrt (x^2 / (1 - (x^2 - 4) / x^2)) = x^2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1144_114414


namespace NUMINAMATH_CALUDE_angle_sum_around_point_l1144_114465

/-- Given a point W with four angles around it, where one angle is 90°, 
    another is y°, a third is 3y°, and the sum of all angles is 360°, 
    prove that y = 67.5° -/
theorem angle_sum_around_point (y : ℝ) : 
  90 + y + 3*y = 360 → y = 67.5 := by sorry

end NUMINAMATH_CALUDE_angle_sum_around_point_l1144_114465


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l1144_114482

/-- The trajectory of point M satisfying the given conditions is an ellipse -/
theorem trajectory_is_ellipse (x y : ℝ) : 
  let F : ℝ × ℝ := (0, 2)
  let line_y : ℝ := 8
  let distance_to_F := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)
  let distance_to_line := |y - line_y|
  distance_to_F / distance_to_line = 1 / 2 → x^2 / 12 + y^2 / 16 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l1144_114482


namespace NUMINAMATH_CALUDE_watch_gain_percentage_l1144_114464

/-- Calculates the gain percentage when a watch is sold at a different price -/
def gainPercentage (costPrice sellPrice : ℚ) : ℚ :=
  (sellPrice - costPrice) / costPrice * 100

/-- Theorem: The gain percentage is 5% under the given conditions -/
theorem watch_gain_percentage :
  let costPrice : ℚ := 933.33
  let initialLossPercentage : ℚ := 10
  let initialSellPrice : ℚ := costPrice * (1 - initialLossPercentage / 100)
  let newSellPrice : ℚ := initialSellPrice + 140
  gainPercentage costPrice newSellPrice = 5 := by
  sorry

end NUMINAMATH_CALUDE_watch_gain_percentage_l1144_114464


namespace NUMINAMATH_CALUDE_optimal_furniture_purchase_l1144_114495

def maximize_furniture (budget chair_price table_price : ℕ) : ℕ × ℕ :=
  let (tables, chairs) := (25, 37)
  have budget_constraint : tables * table_price + chairs * chair_price ≤ budget := by sorry
  have chair_lower_bound : chairs ≥ tables := by sorry
  have chair_upper_bound : chairs ≤ (3 * tables) / 2 := by sorry
  have is_optimal : ∀ (t c : ℕ), t * table_price + c * chair_price ≤ budget → 
                    c ≥ t → c ≤ (3 * t) / 2 → t + c ≤ tables + chairs := by sorry
  (tables, chairs)

theorem optimal_furniture_purchase :
  let (tables, chairs) := maximize_furniture 2000 20 50
  tables = 25 ∧ chairs = 37 := by sorry

end NUMINAMATH_CALUDE_optimal_furniture_purchase_l1144_114495


namespace NUMINAMATH_CALUDE_slope_of_intersection_line_l1144_114496

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 15 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 48 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧ C ≠ D

-- Theorem statement
theorem slope_of_intersection_line (C D : ℝ × ℝ) (h : intersection C D) : 
  (D.2 - C.2) / (D.1 - C.1) = 11/6 := by sorry

end NUMINAMATH_CALUDE_slope_of_intersection_line_l1144_114496


namespace NUMINAMATH_CALUDE_calculation_proof_l1144_114451

theorem calculation_proof : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1144_114451


namespace NUMINAMATH_CALUDE_infinitely_many_n_not_equal_l1144_114403

/-- For any positive integers a and b greater than 1, there are infinitely many n
    such that φ(a^n - 1) ≠ b^m - b^t for any positive integers m and t. -/
theorem infinitely_many_n_not_equal (a b : ℕ) (ha : a > 1) (hb : b > 1) :
  Set.Infinite {n : ℕ | ∀ m t : ℕ, m > 0 → t > 0 → Nat.totient (a^n - 1) ≠ b^m - b^t} :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_not_equal_l1144_114403


namespace NUMINAMATH_CALUDE_even_6digit_integers_count_l1144_114468

/-- The count of even 6-digit positive integers -/
def count_even_6digit_integers : ℕ :=
  9 * 10^4 * 5

/-- Theorem: The count of even 6-digit positive integers is 450,000 -/
theorem even_6digit_integers_count : count_even_6digit_integers = 450000 := by
  sorry

end NUMINAMATH_CALUDE_even_6digit_integers_count_l1144_114468


namespace NUMINAMATH_CALUDE_minimum_points_tenth_game_l1144_114498

def first_five_games : List ℕ := [18, 15, 13, 17, 19]
def next_four_games : List ℕ := [14, 20, 12, 21]

def total_points_nine_games : ℕ := (first_five_games.sum + next_four_games.sum)

theorem minimum_points_tenth_game :
  ∀ x : ℕ, 
    (((total_points_nine_games + x) : ℚ) / 10 > 17) ∧ 
    (∀ y : ℕ, y < x → ((total_points_nine_games + y : ℚ) / 10 ≤ 17)) → 
    x = 22 :=
by sorry

end NUMINAMATH_CALUDE_minimum_points_tenth_game_l1144_114498


namespace NUMINAMATH_CALUDE_photo_reactions_l1144_114467

/-- 
Proves that given a photo with a starting score of 0, where "thumbs up" increases 
the score by 1 and "thumbs down" decreases it by 1, if the current score is 50 
and 75% of reactions are "thumbs up", then the total number of reactions is 100.
-/
theorem photo_reactions 
  (score : ℤ) 
  (total_reactions : ℕ) 
  (thumbs_up_ratio : ℚ) :
  score = 0 + total_reactions * thumbs_up_ratio - total_reactions * (1 - thumbs_up_ratio) →
  score = 50 →
  thumbs_up_ratio = 3/4 →
  total_reactions = 100 := by
  sorry

#check photo_reactions

end NUMINAMATH_CALUDE_photo_reactions_l1144_114467


namespace NUMINAMATH_CALUDE_total_toll_for_week_l1144_114471

/-- Calculate the total toll for a week for an 18-wheel truck -/
theorem total_toll_for_week (total_wheels : Nat) (front_axle_wheels : Nat) (other_axle_wheels : Nat)
  (weekday_base_toll : Real) (weekday_rate : Real) (weekend_base_toll : Real) (weekend_rate : Real) :
  total_wheels = 18 →
  front_axle_wheels = 2 →
  other_axle_wheels = 4 →
  weekday_base_toll = 2.50 →
  weekday_rate = 0.70 →
  weekend_base_toll = 3.00 →
  weekend_rate = 0.80 →
  let total_axles := (total_wheels - front_axle_wheels) / other_axle_wheels + 1
  let weekday_toll := weekday_base_toll + weekday_rate * (total_axles - 1)
  let weekend_toll := weekend_base_toll + weekend_rate * (total_axles - 1)
  let total_toll := 5 * weekday_toll + 2 * weekend_toll
  total_toll = 38.90 := by
  sorry

end NUMINAMATH_CALUDE_total_toll_for_week_l1144_114471


namespace NUMINAMATH_CALUDE_nested_sqrt_bounds_l1144_114435

theorem nested_sqrt_bounds : 
  ∃ x : ℝ, x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_bounds_l1144_114435


namespace NUMINAMATH_CALUDE_imaginary_unit_power_sum_l1144_114438

theorem imaginary_unit_power_sum : ∀ i : ℂ, i^2 = -1 → i^45 + i^205 + i^365 = 3*i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_sum_l1144_114438


namespace NUMINAMATH_CALUDE_parabola_c_value_l1144_114433

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  (p.x_coord 4 = 5) →  -- vertex at (5,4)
  (p.x_coord 6 = 1) →  -- passes through (1,6)
  (p.x_coord 0 = -27) →  -- passes through (-27,0)
  p.c = -27 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1144_114433


namespace NUMINAMATH_CALUDE_discriminant_not_necessary_nor_sufficient_l1144_114442

/-- The function f(x) = ax^2 + bx + c --/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that the graph of f is always above the x-axis --/
def always_above (a b c : ℝ) : Prop :=
  ∀ x, f a b c x > 0

/-- The discriminant condition --/
def discriminant_condition (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

theorem discriminant_not_necessary_nor_sufficient :
  ¬(∀ a b c : ℝ, discriminant_condition a b c ↔ always_above a b c) :=
sorry

end NUMINAMATH_CALUDE_discriminant_not_necessary_nor_sufficient_l1144_114442


namespace NUMINAMATH_CALUDE_alligators_in_pond_l1144_114420

/-- The number of snakes in the pond -/
def num_snakes : ℕ := 18

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := 56

/-- The number of eyes each snake has -/
def snake_eyes : ℕ := 2

/-- The number of eyes each alligator has -/
def alligator_eyes : ℕ := 2

/-- The number of alligators in the pond -/
def num_alligators : ℕ := 10

theorem alligators_in_pond :
  num_snakes * snake_eyes + num_alligators * alligator_eyes = total_eyes :=
by sorry

end NUMINAMATH_CALUDE_alligators_in_pond_l1144_114420


namespace NUMINAMATH_CALUDE_students_without_A_l1144_114408

theorem students_without_A (total : ℕ) (history : ℕ) (math : ℕ) (both : ℕ) 
  (h_total : total = 50)
  (h_history : history = 12)
  (h_math : math = 25)
  (h_both : both = 6) : 
  total - (history + math - both) = 19 := by
  sorry

end NUMINAMATH_CALUDE_students_without_A_l1144_114408


namespace NUMINAMATH_CALUDE_lines_cannot_form_triangle_iff_m_in_set_l1144_114461

/-- A line in the plane, represented by its equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three lines form a triangle -/
def form_triangle (l₁ l₂ l₃ : Line) : Prop :=
  sorry

/-- The set of m values for which the lines cannot form a triangle -/
def invalid_m_values : Set ℝ :=
  {4, -1/6, -1, 2/3}

theorem lines_cannot_form_triangle_iff_m_in_set (m : ℝ) :
  let l₁ : Line := ⟨4, 1, 4⟩
  let l₂ : Line := ⟨m, 1, 0⟩
  let l₃ : Line := ⟨2, -3*m, 4⟩
  ¬(form_triangle l₁ l₂ l₃) ↔ m ∈ invalid_m_values :=
by sorry

end NUMINAMATH_CALUDE_lines_cannot_form_triangle_iff_m_in_set_l1144_114461


namespace NUMINAMATH_CALUDE_arctan_sum_identity_l1144_114472

theorem arctan_sum_identity : 
  Real.arctan (3/4) + 2 * Real.arctan (4/3) = π - Real.arctan (3/4) := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_identity_l1144_114472


namespace NUMINAMATH_CALUDE_certain_number_proof_l1144_114456

theorem certain_number_proof (y : ℝ) : 
  (0.25 * 680 = 0.20 * y - 30) → y = 1000 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l1144_114456


namespace NUMINAMATH_CALUDE_p_or_q_is_true_l1144_114458

-- Define proposition p
def p (a b : ℝ) : Prop := a^2 + b^2 < 0

-- Define proposition q
def q (a b : ℝ) : Prop := (a - 2)^2 + |b - 3| ≥ 0

-- Theorem statement
theorem p_or_q_is_true :
  (∀ a b : ℝ, ¬(p a b)) ∧ (∀ a b : ℝ, q a b) → ∀ a b : ℝ, p a b ∨ q a b :=
by sorry

end NUMINAMATH_CALUDE_p_or_q_is_true_l1144_114458


namespace NUMINAMATH_CALUDE_custom_distance_additive_on_line_segment_l1144_114454

/-- Custom distance function between two points in 2D space -/
def custom_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₂ - x₁| + |y₂ - y₁|

/-- Theorem: For any three points A, B, and C, where C is on the line segment AB,
    the sum of the custom distances AC and CB equals the custom distance AB -/
theorem custom_distance_additive_on_line_segment 
  (x₁ y₁ x₂ y₂ x y : ℝ) 
  (h_between_x : (x₁ - x) * (x₂ - x) ≤ 0)
  (h_between_y : (y₁ - y) * (y₂ - y) ≤ 0) :
  custom_distance x₁ y₁ x y + custom_distance x y x₂ y₂ = custom_distance x₁ y₁ x₂ y₂ :=
by sorry

#check custom_distance_additive_on_line_segment

end NUMINAMATH_CALUDE_custom_distance_additive_on_line_segment_l1144_114454


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1144_114485

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_ratio : 3 * a 5 = a 6)
  (h_second : a 2 = 1) :
  a 4 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l1144_114485


namespace NUMINAMATH_CALUDE_eight_members_prefer_b_first_l1144_114425

/-- Represents the number of ballots for each permutation of candidates A, B, C -/
structure BallotCounts where
  abc : ℕ
  acb : ℕ
  cab : ℕ
  cba : ℕ
  bca : ℕ
  bac : ℕ

/-- The committee voting system with given conditions -/
def CommitteeVoting (counts : BallotCounts) : Prop :=
  -- Total number of ballots is 20
  counts.abc + counts.acb + counts.cab + counts.cba + counts.bca + counts.bac = 20 ∧
  -- Each permutation appears at least once
  counts.abc ≥ 1 ∧ counts.acb ≥ 1 ∧ counts.cab ≥ 1 ∧
  counts.cba ≥ 1 ∧ counts.bca ≥ 1 ∧ counts.bac ≥ 1 ∧
  -- 11 members prefer A to B
  counts.abc + counts.acb + counts.cab = 11 ∧
  -- 12 members prefer C to A
  counts.cab + counts.cba + counts.bca = 12 ∧
  -- 14 members prefer B to C
  counts.abc + counts.bca + counts.bac = 14

/-- The theorem stating that 8 members have B as their first choice -/
theorem eight_members_prefer_b_first (counts : BallotCounts) :
  CommitteeVoting counts → counts.bca + counts.bac = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_members_prefer_b_first_l1144_114425


namespace NUMINAMATH_CALUDE_gravel_cost_theorem_l1144_114405

/-- The cost of gravel in dollars per cubic foot -/
def gravel_cost_per_cubic_foot : ℝ := 4

/-- The conversion factor from cubic yards to cubic feet -/
def cubic_yards_to_cubic_feet : ℝ := 27

/-- The volume of gravel in cubic yards -/
def gravel_volume_cubic_yards : ℝ := 8

/-- Theorem stating the cost of 8 cubic yards of gravel -/
theorem gravel_cost_theorem :
  gravel_cost_per_cubic_foot * cubic_yards_to_cubic_feet * gravel_volume_cubic_yards = 864 := by
  sorry

end NUMINAMATH_CALUDE_gravel_cost_theorem_l1144_114405


namespace NUMINAMATH_CALUDE_arithmetic_progression_squares_l1144_114469

theorem arithmetic_progression_squares (a b c : ℝ) :
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →
  ∃ q : ℝ, (a^2 + a*c + c^2) - (a^2 + a*b + b^2) = q ∧
           (b^2 + b*c + c^2) - (a^2 + a*c + c^2) = q :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_squares_l1144_114469


namespace NUMINAMATH_CALUDE_daniel_noodles_l1144_114497

/-- The number of noodles Daniel had initially -/
def initial_noodles : ℝ := 54.0

/-- The number of noodles Daniel gave away -/
def given_away : ℝ := 12.0

/-- The number of noodles Daniel had left -/
def remaining_noodles : ℝ := initial_noodles - given_away

theorem daniel_noodles : remaining_noodles = 42.0 := by sorry

end NUMINAMATH_CALUDE_daniel_noodles_l1144_114497


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l1144_114418

theorem ratio_equation_solution (a b : ℚ) 
  (h1 : b / a = 4)
  (h2 : b = 20 - 7 * a) : 
  a = 20 / 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l1144_114418


namespace NUMINAMATH_CALUDE_carson_saw_five_octopuses_l1144_114424

/-- The number of legs an octopus has -/
def legs_per_octopus : ℕ := 8

/-- The total number of octopus legs Carson saw -/
def total_legs : ℕ := 40

/-- The number of octopuses Carson saw -/
def num_octopuses : ℕ := total_legs / legs_per_octopus

theorem carson_saw_five_octopuses : num_octopuses = 5 := by
  sorry

end NUMINAMATH_CALUDE_carson_saw_five_octopuses_l1144_114424


namespace NUMINAMATH_CALUDE_negation_equivalence_l1144_114457

-- Define the set S
variable (S : Set ℝ)

-- Define the original statement
def original_statement : Prop :=
  ∀ x ∈ S, |x| ≥ 2

-- Define the negation of the original statement
def negation_statement : Prop :=
  ∃ x ∈ S, |x| < 2

-- Theorem stating the equivalence
theorem negation_equivalence :
  ¬(original_statement S) ↔ negation_statement S :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1144_114457


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1144_114434

theorem expand_and_simplify (x : ℝ) : (1 - x^2) * (1 + x^4 + x^6) = 1 - x^2 + x^4 - x^8 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1144_114434


namespace NUMINAMATH_CALUDE_integral_sqrt_minus_square_l1144_114411

theorem integral_sqrt_minus_square : 
  ∫ x in (0:ℝ)..1, (Real.sqrt (1 - (x - 1)^2) - x^2) = π/4 - 1/3 := by sorry

end NUMINAMATH_CALUDE_integral_sqrt_minus_square_l1144_114411


namespace NUMINAMATH_CALUDE_computer_factory_earnings_l1144_114422

/-- Calculates the earnings from selling computers produced in a week -/
def weekly_earnings (daily_production : ℕ) (price_per_unit : ℕ) : ℕ :=
  daily_production * 7 * price_per_unit

/-- Proves that the weekly earnings for the given conditions equal $1,575,000 -/
theorem computer_factory_earnings :
  weekly_earnings 1500 150 = 1575000 := by
  sorry

end NUMINAMATH_CALUDE_computer_factory_earnings_l1144_114422


namespace NUMINAMATH_CALUDE_arctan_tan_difference_l1144_114406

theorem arctan_tan_difference (θ : Real) : 
  0 ≤ θ ∧ θ ≤ 180 ∧ 
  θ = Real.arctan (Real.tan (75 * π / 180) - 3 * Real.tan (15 * π / 180)) * 180 / π :=
by sorry

end NUMINAMATH_CALUDE_arctan_tan_difference_l1144_114406


namespace NUMINAMATH_CALUDE_reggie_remaining_money_l1144_114423

/-- Calculates the remaining money after a purchase --/
def remaining_money (initial_amount number_of_items cost_per_item : ℕ) : ℕ :=
  initial_amount - (number_of_items * cost_per_item)

/-- Proves that Reggie has $38 left after his purchase --/
theorem reggie_remaining_money :
  remaining_money 48 5 2 = 38 := by
  sorry

end NUMINAMATH_CALUDE_reggie_remaining_money_l1144_114423


namespace NUMINAMATH_CALUDE_gcd_power_minus_one_l1144_114413

theorem gcd_power_minus_one : Nat.gcd (2^2000 - 1) (2^1990 - 1) = 2^10 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_minus_one_l1144_114413


namespace NUMINAMATH_CALUDE_clothes_shop_discount_l1144_114486

theorem clothes_shop_discount (num_friends : ℕ) (original_price : ℝ) (discount_percent : ℝ) : 
  num_friends = 4 → 
  original_price = 20 → 
  discount_percent = 50 → 
  (num_friends : ℝ) * (original_price * (1 - discount_percent / 100)) = 40 := by
sorry

end NUMINAMATH_CALUDE_clothes_shop_discount_l1144_114486


namespace NUMINAMATH_CALUDE_lcm_gcd_product_12_9_l1144_114402

theorem lcm_gcd_product_12_9 :
  Nat.lcm 12 9 * Nat.gcd 12 9 = 108 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_12_9_l1144_114402


namespace NUMINAMATH_CALUDE_basketball_team_selection_l1144_114428

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def team_size : ℕ := 7
def quadruplets_in_team : ℕ := 2

theorem basketball_team_selection :
  (Nat.choose quadruplets quadruplets_in_team) *
  (Nat.choose (total_players - quadruplets) (team_size - quadruplets_in_team)) = 12012 :=
by sorry

end NUMINAMATH_CALUDE_basketball_team_selection_l1144_114428


namespace NUMINAMATH_CALUDE_continuous_with_property_F_is_nondecreasing_l1144_114462

-- Define property F
def has_property_F (f : ℝ → ℝ) : Prop :=
  ∀ a : ℝ, ∃ b : ℝ, b < a ∧ ∀ x ∈ Set.Ioo b a, f x ≤ f a

-- Define nondecreasing
def nondecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- Theorem statement
theorem continuous_with_property_F_is_nondecreasing (f : ℝ → ℝ) 
  (hf : Continuous f) (hF : has_property_F f) : nondecreasing f := by
  sorry


end NUMINAMATH_CALUDE_continuous_with_property_F_is_nondecreasing_l1144_114462


namespace NUMINAMATH_CALUDE_dilution_proof_l1144_114489

/-- Proves that adding 7.2 ounces of water to 12 ounces of 40% alcohol shaving lotion 
    results in a solution with 25% alcohol concentration -/
theorem dilution_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 12 ∧ 
  initial_concentration = 0.4 ∧ 
  target_concentration = 0.25 ∧
  water_added = 7.2 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration :=
by sorry

end NUMINAMATH_CALUDE_dilution_proof_l1144_114489


namespace NUMINAMATH_CALUDE_pascal_triangle_row_34_l1144_114480

theorem pascal_triangle_row_34 : 
  let row_34 := List.range 35
  let nth_elem (n : ℕ) := Nat.choose 34 n
  (nth_elem 29 = 278256) ∧ (nth_elem 30 = 46376) := by
sorry

end NUMINAMATH_CALUDE_pascal_triangle_row_34_l1144_114480


namespace NUMINAMATH_CALUDE_sum_of_numbers_l1144_114483

theorem sum_of_numbers : 1357 + 3571 + 5713 + 7135 + 1357 = 19133 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l1144_114483


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l1144_114421

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 1/y = 1 → a + 2*b ≤ x + 2*y :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l1144_114421


namespace NUMINAMATH_CALUDE_shoe_cost_difference_l1144_114407

/-- Proves that the percentage difference between the average cost per year of new shoes
    and the cost of repairing used shoes is 10.34%, given the specified conditions. -/
theorem shoe_cost_difference (used_repair_cost : ℝ) (used_repair_duration : ℝ)
    (new_shoe_cost : ℝ) (new_shoe_duration : ℝ)
    (h1 : used_repair_cost = 14.50)
    (h2 : used_repair_duration = 1)
    (h3 : new_shoe_cost = 32.00)
    (h4 : new_shoe_duration = 2) :
    let used_cost_per_year := used_repair_cost / used_repair_duration
    let new_cost_per_year := new_shoe_cost / new_shoe_duration
    let percentage_difference := (new_cost_per_year - used_cost_per_year) / used_cost_per_year * 100
    percentage_difference = 10.34 := by
  sorry

end NUMINAMATH_CALUDE_shoe_cost_difference_l1144_114407


namespace NUMINAMATH_CALUDE_R_duration_approx_l1144_114491

/-- Represents the investment and profit information for three partners -/
structure PartnershipData where
  inv_ratio_P : ℚ
  inv_ratio_Q : ℚ
  inv_ratio_R : ℚ
  profit_ratio_P : ℚ
  profit_ratio_Q : ℚ
  profit_ratio_R : ℚ
  duration_P : ℚ
  duration_Q : ℚ

/-- Calculates the investment duration for partner R given the partnership data -/
def calculate_R_duration (data : PartnershipData) : ℚ :=
  (data.profit_ratio_R * data.inv_ratio_Q * data.duration_Q) /
  (data.profit_ratio_Q * data.inv_ratio_R)

/-- Theorem stating that R's investment duration is approximately 5.185 months -/
theorem R_duration_approx (data : PartnershipData)
  (h1 : data.inv_ratio_P = 7)
  (h2 : data.inv_ratio_Q = 5)
  (h3 : data.inv_ratio_R = 3)
  (h4 : data.profit_ratio_P = 7)
  (h5 : data.profit_ratio_Q = 9)
  (h6 : data.profit_ratio_R = 4)
  (h7 : data.duration_P = 5)
  (h8 : data.duration_Q = 7) :
  abs (calculate_R_duration data - 5.185) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_R_duration_approx_l1144_114491


namespace NUMINAMATH_CALUDE_other_diagonal_length_l1144_114479

-- Define the rhombus
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  triangle_area : ℝ

-- Define the properties of the rhombus
def rhombus_properties (r : Rhombus) : Prop :=
  r.diagonal1 = 20 ∧ r.triangle_area = 75

-- Theorem statement
theorem other_diagonal_length (r : Rhombus) 
  (h : rhombus_properties r) : r.diagonal2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_other_diagonal_length_l1144_114479


namespace NUMINAMATH_CALUDE_profit_percentage_problem_l1144_114447

/-- Calculates the profit percentage given the cost price and selling price -/
def profit_percentage (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that the profit percentage is 25% for the given problem -/
theorem profit_percentage_problem : profit_percentage 96 120 = 25 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_problem_l1144_114447


namespace NUMINAMATH_CALUDE_base_conversion_1729_l1144_114400

theorem base_conversion_1729 :
  (5 * 7^3 + 0 * 7^2 + 2 * 7^1 + 0 * 7^0 : ℕ) = 1729 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_1729_l1144_114400


namespace NUMINAMATH_CALUDE_sector_arc_length_l1144_114441

theorem sector_arc_length (θ : Real) (A : Real) (l : Real) : 
  θ = 120 → A = π → l = (2 * Real.sqrt 3 * π) / 3 → 
  l = (θ * Real.sqrt (3 * A / θ) * π) / 180 := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l1144_114441


namespace NUMINAMATH_CALUDE_range_of_expression_l1144_114493

theorem range_of_expression (x y : ℝ) (h1 : x * y = 1) (h2 : 3 ≥ x) (h3 : x ≥ 4 * y) (h4 : y > 0) :
  ∃ (a b : ℝ), a = 4 ∧ b = 5 ∧
  (∀ z, (z = (x^2 + 4*y^2) / (x - 2*y)) → a ≤ z ∧ z ≤ b) ∧
  (∃ z1 z2, z1 = (x^2 + 4*y^2) / (x - 2*y) ∧ z2 = (x^2 + 4*y^2) / (x - 2*y) ∧ z1 = a ∧ z2 = b) :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l1144_114493


namespace NUMINAMATH_CALUDE_marathon_average_time_l1144_114401

def casey_time : ℝ := 6

theorem marathon_average_time (casey_time : ℝ) (zendaya_factor : ℝ) :
  casey_time = 6 →
  zendaya_factor = 1/3 →
  let zendaya_time := casey_time * (1 + zendaya_factor)
  let total_time := casey_time + zendaya_time
  let average_time := total_time / 2
  average_time = 7 := by sorry

end NUMINAMATH_CALUDE_marathon_average_time_l1144_114401


namespace NUMINAMATH_CALUDE_quadratic_polynomial_proof_l1144_114440

/-- A quadratic polynomial M in terms of x -/
def M (a : ℝ) (x : ℝ) : ℝ := (a + 4) * x^3 + 6 * x^2 - 2 * x + 5

/-- The coefficient of the quadratic term -/
def b : ℝ := 6

/-- Point A on the number line -/
def A : ℝ := -4

/-- Point B on the number line -/
def B : ℝ := 6

/-- Position of P after t seconds -/
def P (t : ℝ) : ℝ := A + 2 * t

/-- Position of Q after t seconds (starting 2 seconds after P) -/
def Q (t : ℝ) : ℝ := B - 2 * (t - 2)

/-- Distance between two points -/
def distance (x y : ℝ) : ℝ := |x - y|

theorem quadratic_polynomial_proof :
  (∀ x, M A x = 6 * x^2 - 2 * x + 5) ∧
  (∃ t, t > 0 ∧ (distance (P t) B = (1/2) * distance (P t) A)) ∧
  (∃ m, m > 2 ∧ distance (P m) (Q m) = 8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_proof_l1144_114440


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l1144_114453

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 80 →
  E = 2 * F + 24 →
  D + E + F = 180 →
  F = 76 / 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l1144_114453


namespace NUMINAMATH_CALUDE_sams_watermelons_l1144_114429

theorem sams_watermelons (grown : ℕ) (eaten : ℕ) (h1 : grown = 4) (h2 : eaten = 3) :
  grown - eaten = 1 := by
  sorry

end NUMINAMATH_CALUDE_sams_watermelons_l1144_114429
