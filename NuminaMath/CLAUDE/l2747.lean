import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_l2747_274760

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - (2 * a + 5)) > 0}
def B (a : ℝ) : Set ℝ := {x | ((a^2 + 2) - x) * (2 * a - x) < 0}

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
    a > 1/2 → 
    (∀ x : ℝ, x ∈ B a → x ∈ A a) →
    (∃ x : ℝ, x ∈ A a ∧ x ∉ B a) →
    a > 1/2 ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2747_274760


namespace NUMINAMATH_CALUDE_regular_ngon_smallest_area_and_perimeter_l2747_274761

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- An n-gon circumscribed about a circle --/
structure CircumscribedNGon where
  n : ℕ
  circle : Circle
  vertices : Fin n → ℝ × ℝ

/-- Checks if an n-gon is regular --/
def is_regular (ngon : CircumscribedNGon) : Prop :=
  sorry

/-- Calculates the area of an n-gon --/
def area (ngon : CircumscribedNGon) : ℝ :=
  sorry

/-- Calculates the perimeter of an n-gon --/
def perimeter (ngon : CircumscribedNGon) : ℝ :=
  sorry

/-- Theorem: The regular n-gon has the smallest area and perimeter among all n-gons circumscribed about a given circle --/
theorem regular_ngon_smallest_area_and_perimeter (n : ℕ) (c : Circle) :
  ∀ (ngon : CircumscribedNGon), ngon.n = n ∧ ngon.circle = c →
    ∃ (reg_ngon : CircumscribedNGon), 
      reg_ngon.n = n ∧ reg_ngon.circle = c ∧ is_regular reg_ngon ∧
      area reg_ngon ≤ area ngon ∧ perimeter reg_ngon ≤ perimeter ngon :=
  sorry

end NUMINAMATH_CALUDE_regular_ngon_smallest_area_and_perimeter_l2747_274761


namespace NUMINAMATH_CALUDE_power_inequality_l2747_274725

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^a * b^b ≥ (a*b)^((a+b)/2) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2747_274725


namespace NUMINAMATH_CALUDE_dance_move_ratio_l2747_274758

/-- Frank's dance move sequence --/
structure DanceMove where
  initial_back : ℤ
  first_forward : ℤ
  second_back : ℤ
  final_forward : ℤ

/-- The dance move Frank performs --/
def franks_move : DanceMove :=
  { initial_back := 5
  , first_forward := 10
  , second_back := 2
  , final_forward := 4 }

/-- The final position relative to the starting point --/
def final_position (move : DanceMove) : ℤ :=
  -move.initial_back + move.first_forward - move.second_back + move.final_forward

/-- The theorem stating the ratio of final forward steps to second back steps --/
theorem dance_move_ratio (move : DanceMove) : 
  final_position move = 7 → 
  (move.final_forward : ℚ) / move.second_back = 2 := by
  sorry

#eval final_position franks_move

end NUMINAMATH_CALUDE_dance_move_ratio_l2747_274758


namespace NUMINAMATH_CALUDE_x_twenty_percent_greater_than_98_l2747_274779

theorem x_twenty_percent_greater_than_98 (x : ℝ) :
  x = 98 * (1 + 20 / 100) → x = 117.6 := by
  sorry

end NUMINAMATH_CALUDE_x_twenty_percent_greater_than_98_l2747_274779


namespace NUMINAMATH_CALUDE_allowance_calculation_l2747_274785

theorem allowance_calculation (card_cost sticker_box_cost : ℚ) 
  (total_sticker_packs : ℕ) (h1 : card_cost = 10) 
  (h2 : sticker_box_cost = 2) (h3 : total_sticker_packs = 4) : 
  (card_cost + sticker_box_cost * (total_sticker_packs / 2)) / 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_allowance_calculation_l2747_274785


namespace NUMINAMATH_CALUDE_integer_root_count_theorem_l2747_274735

/-- A polynomial of degree 5 with integer coefficients -/
def IntPolynomial5 (x b c d e f : ℤ) : ℤ := x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f

/-- The set of possible numbers of integer roots for IntPolynomial5 -/
def PossibleRootCounts : Set ℕ := {0, 1, 2, 5}

/-- Theorem stating that the number of integer roots of IntPolynomial5 is in PossibleRootCounts -/
theorem integer_root_count_theorem (b c d e f : ℤ) :
  ∃ (n : ℕ), n ∈ PossibleRootCounts ∧
  (∃ (roots : List ℤ), (∀ x ∈ roots, IntPolynomial5 x b c d e f = 0) ∧
                       roots.length = n) :=
sorry

end NUMINAMATH_CALUDE_integer_root_count_theorem_l2747_274735


namespace NUMINAMATH_CALUDE_sequence_general_term_l2747_274745

def sequence_property (a : ℕ+ → ℕ+) : Prop :=
  (∀ m k : ℕ+, a (m^2) = (a m)^2) ∧
  (∀ m k : ℕ+, a (m^2 + k^2) = a m * a k)

theorem sequence_general_term (a : ℕ+ → ℕ+) (h : sequence_property a) :
  ∀ n : ℕ+, a n = 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l2747_274745


namespace NUMINAMATH_CALUDE_triangle_area_l2747_274722

/-- Given a triangle ABC where:
  * b is the length of the side opposite to angle B
  * c is the length of the side opposite to angle C
  * C is twice the measure of angle B
prove that the area of the triangle is 15√7/16 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) : 
  b = 2 → 
  c = 3 → 
  C = 2 * B → 
  (1/2) * b * c * Real.sin A = 15 * Real.sqrt 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2747_274722


namespace NUMINAMATH_CALUDE_equation_equivalence_l2747_274774

/-- An equation is homogeneous if for any solution (x, y), (rx, ry) is also a solution for any non-zero scalar r. -/
def IsHomogeneous (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ (x y r : ℝ), r ≠ 0 → f x y = 0 → f (r * x) (r * y) = 0

/-- The original equation -/
def OriginalEq (x y : ℝ) : ℝ := x^3 - 2*x^2*y + x*y^2 - 2*y^3

/-- The equivalent equation -/
def EquivalentEq (x y : ℝ) : Prop := x = 2*y

theorem equation_equivalence :
  IsHomogeneous OriginalEq →
  (∀ x y : ℝ, OriginalEq x y = 0 ↔ EquivalentEq x y) :=
sorry

end NUMINAMATH_CALUDE_equation_equivalence_l2747_274774


namespace NUMINAMATH_CALUDE_cole_fence_cost_l2747_274724

theorem cole_fence_cost (side_length : ℝ) (back_length : ℝ) (cost_per_foot : ℝ)
  (h_side : side_length = 9)
  (h_back : back_length = 18)
  (h_cost : cost_per_foot = 3)
  (h_neighbor_back : ∃ (x : ℝ), x = back_length * cost_per_foot / 2)
  (h_neighbor_left : ∃ (y : ℝ), y = side_length * cost_per_foot / 3) :
  ∃ (total_cost : ℝ), total_cost = 72 ∧
    total_cost = side_length * cost_per_foot + 
                 (2/3) * side_length * cost_per_foot + 
                 back_length * cost_per_foot / 2 :=
by sorry

end NUMINAMATH_CALUDE_cole_fence_cost_l2747_274724


namespace NUMINAMATH_CALUDE_unfactorable_expression_difference_of_squares_factorization_common_factor_factorization_perfect_square_trinomial_factorization_l2747_274791

theorem unfactorable_expression (x : ℝ) : ¬∃ (a b : ℝ), x^2 + 9 = a * b ∧ (a ≠ 1 ∨ b ≠ x^2 + 9) ∧ (a ≠ x^2 + 9 ∨ b ≠ 1) := by
  sorry

-- Helper theorems to show that other expressions can be factored
theorem difference_of_squares_factorization (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

theorem common_factor_factorization (x : ℝ) : 9*x - 9 = 9 * (x - 1) := by
  sorry

theorem perfect_square_trinomial_factorization (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_unfactorable_expression_difference_of_squares_factorization_common_factor_factorization_perfect_square_trinomial_factorization_l2747_274791


namespace NUMINAMATH_CALUDE_quadrilateral_area_l2747_274715

/-- A quadrilateral with right angles at B and D, diagonal AC of length 5,
    and two sides with distinct integer lengths has an area of 12. -/
theorem quadrilateral_area (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let DA := Real.sqrt ((D.1 - A.1)^2 + (D.2 - A.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  (B.1 - A.1) * (B.2 - D.2) = (B.2 - A.2) * (B.1 - D.1) →  -- right angle at B
  (D.1 - C.1) * (D.2 - B.2) = (D.2 - C.2) * (D.1 - B.1) →  -- right angle at D
  AC = 5 →
  (∃ (x y : ℕ), (AB = x ∨ BC = x ∨ CD = x ∨ DA = x) ∧ 
                (AB = y ∨ BC = y ∨ CD = y ∨ DA = y) ∧ x ≠ y) →
  (1/2 * AB * BC) + (1/2 * CD * DA) = 12 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l2747_274715


namespace NUMINAMATH_CALUDE_sine_cosine_unique_pair_l2747_274793

open Real

theorem sine_cosine_unique_pair :
  ∃! (c d : ℝ), 0 < c ∧ c < d ∧ d < π / 2 ∧
    sin (cos c) = c ∧ cos (sin d) = d := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_unique_pair_l2747_274793


namespace NUMINAMATH_CALUDE_derek_age_l2747_274765

/-- Given the ages of Uncle Bob, Evan, and Derek, prove Derek's age -/
theorem derek_age (uncle_bob_age : ℕ) (evan_age : ℕ) (derek_age : ℕ) : 
  uncle_bob_age = 60 →
  evan_age = 2 * uncle_bob_age / 3 →
  derek_age = evan_age - 10 →
  derek_age = 30 := by
sorry

end NUMINAMATH_CALUDE_derek_age_l2747_274765


namespace NUMINAMATH_CALUDE_proposition_truth_values_l2747_274720

theorem proposition_truth_values (p q : Prop) 
  (hp : p) 
  (hq : ¬q) : 
  (p ∨ q) ∧ ¬(p ∧ q) ∧ ¬((¬p) ∧ (¬q)) ∧ ¬(¬p) := by
  sorry

end NUMINAMATH_CALUDE_proposition_truth_values_l2747_274720


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2747_274739

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  let a : Fin 2 → ℝ := ![3, -2]
  let b : Fin 2 → ℝ := ![x, y - 1]
  (∃ (k : ℝ), a = k • b) →
  (3 / x + 2 / y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 / x₀ + 2 / y₀ = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2747_274739


namespace NUMINAMATH_CALUDE_system_solutions_l2747_274747

-- Define the system of equations
def equation1 (y z : ℚ) : Prop := y * z = 3 * y + 2 * z - 8
def equation2 (z x : ℚ) : Prop := z * x = 4 * z + 3 * x - 8
def equation3 (x y : ℚ) : Prop := x * y = 2 * x + y - 1

-- Define the solutions
def solution1 : (ℚ × ℚ × ℚ) := (2, 3, 1)
def solution2 : (ℚ × ℚ × ℚ) := (3, 5/2, -1)

-- Theorem statement
theorem system_solutions :
  (equation1 solution1.2.1 solution1.2.2 ∧ 
   equation2 solution1.2.2 solution1.1 ∧ 
   equation3 solution1.1 solution1.2.1) ∧
  (equation1 solution2.2.1 solution2.2.2 ∧ 
   equation2 solution2.2.2 solution2.1 ∧ 
   equation3 solution2.1 solution2.2.1) := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l2747_274747


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l2747_274752

theorem floor_equation_solutions : 
  (∃ (s : Finset ℕ), s.card = 110 ∧ 
    (∀ x : ℕ, x ∈ s ↔ ⌊(x : ℚ) / 10⌋ = ⌊(x : ℚ) / 11⌋ + 1)) := by
  sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l2747_274752


namespace NUMINAMATH_CALUDE_unique_root_of_abs_equation_l2747_274727

/-- The equation x|x| - 3|x| - 4 = 0 has exactly one real root -/
theorem unique_root_of_abs_equation : ∃! x : ℝ, x * |x| - 3 * |x| - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_root_of_abs_equation_l2747_274727


namespace NUMINAMATH_CALUDE_min_value_theorem_l2747_274710

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - 3 = 0) :
  (4 * y - x + 6) / (x * y) ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2747_274710


namespace NUMINAMATH_CALUDE_plot_perimeter_l2747_274709

/-- A rectangular plot with specific properties -/
structure RectangularPlot where
  width : ℝ
  length : ℝ
  fencing_rate : ℝ
  fencing_cost : ℝ
  length_width_relation : length = width + 10
  cost_relation : fencing_cost = (2 * (length + width)) * fencing_rate

/-- The perimeter of a rectangular plot -/
def perimeter (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.width)

/-- Theorem stating the perimeter of the specific plot -/
theorem plot_perimeter (plot : RectangularPlot) 
  (h1 : plot.fencing_rate = 6.5)
  (h2 : plot.fencing_cost = 910) : 
  perimeter plot = 140 := by
  sorry

end NUMINAMATH_CALUDE_plot_perimeter_l2747_274709


namespace NUMINAMATH_CALUDE_abigail_saving_period_l2747_274738

def saving_period (monthly_saving : ℕ) (total_saved : ℕ) : ℕ :=
  total_saved / monthly_saving

theorem abigail_saving_period :
  let monthly_saving : ℕ := 4000
  let total_saved : ℕ := 48000
  saving_period monthly_saving total_saved = 12 := by
  sorry

end NUMINAMATH_CALUDE_abigail_saving_period_l2747_274738


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2747_274772

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) : 
  Complex.im ((i / (1 + i)) - (1 / (2 * i))) = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2747_274772


namespace NUMINAMATH_CALUDE_digits_of_3_pow_24_times_7_pow_36_l2747_274730

theorem digits_of_3_pow_24_times_7_pow_36 : ∃ n : ℕ, 
  n > 0 ∧ n < 10^32 ∧ 10^31 ≤ 3^24 * 7^36 ∧ 3^24 * 7^36 < 10^32 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_3_pow_24_times_7_pow_36_l2747_274730


namespace NUMINAMATH_CALUDE_adams_collection_worth_80_dollars_l2747_274781

/-- The value of Adam's coin collection -/
def adams_collection_value (total_coins : ℕ) (sample_coins : ℕ) (sample_value : ℕ) : ℕ :=
  total_coins * (sample_value / sample_coins)

/-- Theorem: Adam's coin collection is worth 80 dollars -/
theorem adams_collection_worth_80_dollars :
  adams_collection_value 20 4 16 = 80 := by
  sorry

end NUMINAMATH_CALUDE_adams_collection_worth_80_dollars_l2747_274781


namespace NUMINAMATH_CALUDE_scientific_notation_of_1300000_l2747_274753

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Convert a positive real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_1300000 :
  to_scientific_notation 1300000 = ScientificNotation.mk 1.3 6 sorry :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1300000_l2747_274753


namespace NUMINAMATH_CALUDE_existence_of_special_integers_l2747_274705

theorem existence_of_special_integers (k : ℕ+) :
  ∃ x y : ℤ, x % 7 ≠ 0 ∧ y % 7 ≠ 0 ∧ x^2 + 6*y^2 = 7^(k : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_integers_l2747_274705


namespace NUMINAMATH_CALUDE_function_range_l2747_274799

theorem function_range (t : ℝ) : 
  (∃ x : ℝ, t ≤ x ∧ x ≤ t + 2 ∧ x^2 + t*x - 12 ≤ 0) → 
  -4 ≤ t ∧ t ≤ Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_function_range_l2747_274799


namespace NUMINAMATH_CALUDE_triangle_side_length_l2747_274769

theorem triangle_side_length (A B C : ℝ) (b c : ℝ) :
  A = π / 3 →
  b = 16 →
  (1 / 2) * b * c * Real.sin A = 64 * Real.sqrt 3 →
  c = 16 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2747_274769


namespace NUMINAMATH_CALUDE_integer_square_root_l2747_274794

theorem integer_square_root (x : ℤ) : 
  (∃ n : ℤ, n ≥ 0 ∧ n^2 = x^2 - x + 1) ↔ (x = 0 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_integer_square_root_l2747_274794


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2747_274797

theorem absolute_value_equation_solutions :
  {x : ℝ | |x - 2| = |x - 3| + |x - 6| + 2} = {-9, 9} := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l2747_274797


namespace NUMINAMATH_CALUDE_parabola_points_difference_l2747_274788

/-- Given a parabola x^2 = 4y with focus F, and two points A and B on it satisfying |AF| - |BF| = 2,
    prove that y₁ + x₁² - y₂ - x₂² = 10 -/
theorem parabola_points_difference (x₁ y₁ x₂ y₂ : ℝ) : 
  (x₁^2 = 4*y₁) →  -- A is on the parabola
  (x₂^2 = 4*y₂) →  -- B is on the parabola
  (y₁ + 1 - (y₂ + 1) = 2) →  -- |AF| - |BF| = 2, where F is (0, 1)
  y₁ + x₁^2 - y₂ - x₂^2 = 10 := by
sorry

end NUMINAMATH_CALUDE_parabola_points_difference_l2747_274788


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l2747_274754

/-- The probability of drawing a red ball from a bag with red and white balls -/
theorem probability_of_red_ball (red_balls white_balls : ℕ) :
  red_balls = 7 → white_balls = 3 →
  (red_balls : ℚ) / (red_balls + white_balls : ℚ) = 7 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l2747_274754


namespace NUMINAMATH_CALUDE_goose_egg_count_l2747_274790

theorem goose_egg_count (
  hatch_rate : ℚ)
  (first_month_survival : ℚ)
  (next_three_months_survival : ℚ)
  (following_six_months_survival : ℚ)
  (first_half_second_year_survival : ℚ)
  (second_year_survival : ℚ)
  (final_survivors : ℕ)
  (h1 : hatch_rate = 4 / 7)
  (h2 : first_month_survival = 3 / 5)
  (h3 : next_three_months_survival = 7 / 10)
  (h4 : following_six_months_survival = 5 / 8)
  (h5 : first_half_second_year_survival = 2 / 3)
  (h6 : second_year_survival = 4 / 5)
  (h7 : final_survivors = 200) :
  ∃ (original_eggs : ℕ), original_eggs = 2503 ∧
  (↑final_survivors : ℚ) = ↑original_eggs * hatch_rate * first_month_survival *
    next_three_months_survival * following_six_months_survival *
    first_half_second_year_survival * second_year_survival :=
by sorry

end NUMINAMATH_CALUDE_goose_egg_count_l2747_274790


namespace NUMINAMATH_CALUDE_gcf_of_180_270_450_l2747_274729

theorem gcf_of_180_270_450 : Nat.gcd 180 (Nat.gcd 270 450) = 90 := by sorry

end NUMINAMATH_CALUDE_gcf_of_180_270_450_l2747_274729


namespace NUMINAMATH_CALUDE_mike_money_total_l2747_274708

/-- Given that Mike has 9 5-dollar bills, prove that his total money is $45. -/
theorem mike_money_total : 
  let number_of_bills : ℕ := 9
  let bill_value : ℕ := 5
  number_of_bills * bill_value = 45 := by
  sorry

end NUMINAMATH_CALUDE_mike_money_total_l2747_274708


namespace NUMINAMATH_CALUDE_axis_of_symmetry_at_1_5_l2747_274771

/-- A function g is symmetric about x = 1.5 if g(x) = g(3-x) for all x -/
def IsSymmetricAbout1_5 (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (3 - x)

/-- The line x = 1.5 is an axis of symmetry for g if g is symmetric about x = 1.5 -/
theorem axis_of_symmetry_at_1_5 (g : ℝ → ℝ) (h : IsSymmetricAbout1_5 g) :
  ∀ x y, g x = y → g (3 - x) = y :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_at_1_5_l2747_274771


namespace NUMINAMATH_CALUDE_number_of_arrangements_l2747_274721

/-- Represents a step on the staircase -/
structure Step :=
  (occupants : Finset Char)
  (h : occupants.card ≤ 2)

/-- Represents an arrangement of people on the staircase -/
def Arrangement := Finset Step

/-- The set of all valid arrangements -/
def AllArrangements : Finset Arrangement :=
  sorry

/-- The number of different ways 4 people can stand on 5 steps -/
theorem number_of_arrangements :
  (AllArrangements.filter (fun arr => arr.sum (fun step => step.occupants.card) = 4)).card = 540 :=
sorry

end NUMINAMATH_CALUDE_number_of_arrangements_l2747_274721


namespace NUMINAMATH_CALUDE_sin_plus_2cos_equals_two_fifths_l2747_274786

theorem sin_plus_2cos_equals_two_fifths (a : ℝ) (α : ℝ) :
  a < 0 →
  (∃ (x y : ℝ), x = -3*a ∧ y = 4*a ∧ Real.sin α = y / Real.sqrt (x^2 + y^2) ∧ Real.cos α = x / Real.sqrt (x^2 + y^2)) →
  Real.sin α + 2 * Real.cos α = 2/5 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_2cos_equals_two_fifths_l2747_274786


namespace NUMINAMATH_CALUDE_magazine_cost_l2747_274726

/-- The cost of a magazine and pencil, given specific conditions -/
theorem magazine_cost (pencil_cost coupon_value total_spent : ℚ) :
  pencil_cost = 0.5 →
  coupon_value = 0.35 →
  total_spent = 1 →
  ∃ (magazine_cost : ℚ),
    magazine_cost + pencil_cost - coupon_value = total_spent ∧
    magazine_cost = 0.85 := by
  sorry

end NUMINAMATH_CALUDE_magazine_cost_l2747_274726


namespace NUMINAMATH_CALUDE_function_maximum_implies_inequality_l2747_274782

/-- Given a function f(x) = ln x - mx² + 2nx where m is real and n is positive,
    if f(x) ≤ f(1) for all positive x, then ln n < 8m -/
theorem function_maximum_implies_inequality (m : ℝ) (n : ℝ) (h_n_pos : n > 0) :
  (∀ x > 0, Real.log x - m * x^2 + 2 * n * x ≤ Real.log 1 - m * 1^2 + 2 * n * 1) →
  Real.log n < 8 * m :=
by sorry

end NUMINAMATH_CALUDE_function_maximum_implies_inequality_l2747_274782


namespace NUMINAMATH_CALUDE_star_difference_l2747_274706

def star (x y : ℝ) : ℝ := 2*x*y - 3*x + y

theorem star_difference : (star 6 4) - (star 4 6) = -8 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_l2747_274706


namespace NUMINAMATH_CALUDE_gcd_48_180_l2747_274717

theorem gcd_48_180 : Nat.gcd 48 180 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_48_180_l2747_274717


namespace NUMINAMATH_CALUDE_ali_age_l2747_274734

/-- Given the ages of Ali, Yusaf, and Umar, prove Ali's age -/
theorem ali_age (ali yusaf umar : ℕ) 
  (h1 : ali = yusaf + 3)
  (h2 : umar = 2 * yusaf)
  (h3 : umar = 10) : 
  ali = 8 := by
  sorry

end NUMINAMATH_CALUDE_ali_age_l2747_274734


namespace NUMINAMATH_CALUDE_tesla_ratio_proof_l2747_274711

/-- The number of Teslas owned by Chris -/
def chris_teslas : ℕ := 6

/-- The number of Teslas owned by Elon -/
def elon_teslas : ℕ := 13

/-- The number of additional Teslas Elon has compared to Sam -/
def elon_sam_difference : ℕ := 10

/-- The number of Teslas owned by Sam -/
def sam_teslas : ℕ := elon_teslas - elon_sam_difference

theorem tesla_ratio_proof :
  (sam_teslas : ℚ) / chris_teslas = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_tesla_ratio_proof_l2747_274711


namespace NUMINAMATH_CALUDE_select_four_boots_from_five_pairs_l2747_274796

/-- The number of ways to select 4 boots from 5 pairs, including exactly one pair -/
def select_boots (n : ℕ) : ℕ :=
  let total_pairs := 5
  let pairs_to_choose := 1
  let remaining_pairs := total_pairs - pairs_to_choose
  let boots_to_choose := n - 2 * pairs_to_choose
  (total_pairs.choose pairs_to_choose) * 
  (remaining_pairs.choose (boots_to_choose / 2)) * 
  2^(boots_to_choose)

/-- Theorem stating that there are 120 ways to select 4 boots from 5 pairs, including exactly one pair -/
theorem select_four_boots_from_five_pairs : select_boots 4 = 120 := by
  sorry

end NUMINAMATH_CALUDE_select_four_boots_from_five_pairs_l2747_274796


namespace NUMINAMATH_CALUDE_equation_solution_exists_l2747_274737

theorem equation_solution_exists : ∃ x : ℝ, 
  x * 3967 + 36990 - 204790 / 19852 = 322299 ∧ 
  abs (x - 71.924) < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l2747_274737


namespace NUMINAMATH_CALUDE_donna_weekly_episodes_l2747_274740

/-- The number of episodes Donna can watch on a weekday -/
def weekday_episodes : ℕ := 8

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The factor by which weekend watching increases compared to weekdays -/
def weekend_factor : ℕ := 3

/-- The total number of episodes Donna can watch in a week -/
def total_episodes : ℕ := weekday_episodes * weekdays + weekend_factor * weekday_episodes * weekend_days

theorem donna_weekly_episodes : total_episodes = 88 := by
  sorry

end NUMINAMATH_CALUDE_donna_weekly_episodes_l2747_274740


namespace NUMINAMATH_CALUDE_unique_solution_is_sqrt_2_l2747_274701

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else Real.log x / Real.log 4

theorem unique_solution_is_sqrt_2 :
  ∃! x, x > 1 ∧ f x = 1/4 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_sqrt_2_l2747_274701


namespace NUMINAMATH_CALUDE_pechkin_ate_four_tenths_l2747_274719

/-- The fraction of the cake eaten by each person -/
structure CakeFractions where
  pechkin : ℝ
  fyodor : ℝ
  matroskin : ℝ
  sharik : ℝ

/-- The conditions of the cake-eating problem -/
def cake_problem (f : CakeFractions) : Prop :=
  -- The whole cake was eaten
  f.pechkin + f.fyodor + f.matroskin + f.sharik = 1 ∧
  -- Uncle Fyodor ate half as much as Pechkin
  f.fyodor = f.pechkin / 2 ∧
  -- Cat Matroskin ate half as much as the portion of the cake that Pechkin did not eat
  f.matroskin = (1 - f.pechkin) / 2 ∧
  -- Sharik ate one-tenth of the cake
  f.sharik = 1 / 10

/-- Theorem stating that given the conditions, Pechkin ate 0.4 of the cake -/
theorem pechkin_ate_four_tenths (f : CakeFractions) :
  cake_problem f → f.pechkin = 0.4 := by sorry

end NUMINAMATH_CALUDE_pechkin_ate_four_tenths_l2747_274719


namespace NUMINAMATH_CALUDE_dividend_divisor_calculation_l2747_274783

/-- Given a dividend of 73648 and a divisor of 874, prove that the result of subtracting
    the product of the divisor and the sum of the quotient's digits from the dividend
    is equal to 63160. -/
theorem dividend_divisor_calculation : 
  let dividend : Nat := 73648
  let divisor : Nat := 874
  let quotient : Nat := dividend / divisor
  let remainder : Nat := dividend % divisor
  let sum_of_digits : Nat := (quotient / 10) + (quotient % 10)
  73648 - (sum_of_digits * 874) = 63160 := by
  sorry

#eval 73648 - ((73648 / 874 / 10 + 73648 / 874 % 10) * 874)

end NUMINAMATH_CALUDE_dividend_divisor_calculation_l2747_274783


namespace NUMINAMATH_CALUDE_intersection_height_l2747_274723

/-- Represents a line in 2D space --/
structure Line where
  m : ℚ  -- slope
  b : ℚ  -- y-intercept

/-- Creates a line from two points --/
def lineFromPoints (x1 y1 x2 y2 : ℚ) : Line :=
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  { m := m, b := b }

/-- Calculates the y-coordinate for a given x on the line --/
def Line.yAt (l : Line) (x : ℚ) : ℚ :=
  l.m * x + l.b

theorem intersection_height : 
  let line1 := lineFromPoints 0 30 120 0
  let line2 := lineFromPoints 0 0 120 50
  let x_intersect := (line2.b - line1.b) / (line1.m - line2.m)
  line1.yAt x_intersect = 75/4 := by sorry

end NUMINAMATH_CALUDE_intersection_height_l2747_274723


namespace NUMINAMATH_CALUDE_unique_solution_l2747_274750

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_range : hundreds ∈ Finset.range 10
  t_range : tens ∈ Finset.range 10
  o_range : ones ∈ Finset.range 10
  h_nonzero : hundreds ≠ 0

/-- Calculates the value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Calculates the product of digits of a number -/
def digitProduct (n : Nat) : Nat :=
  if n < 10 then n
  else if n < 100 then (n / 10) * (n % 10)
  else (n / 100) * ((n / 10) % 10) * (n % 10)

/-- Checks if a three-digit number satisfies the given conditions -/
def satisfiesConditions (n : ThreeDigitNumber) : Prop :=
  let firstProduct := digitProduct n.value
  let secondProduct := digitProduct firstProduct
  (10 ≤ firstProduct ∧ firstProduct < 100) ∧
  (0 < secondProduct ∧ secondProduct < 10) ∧
  n.hundreds = 1 ∧
  n.tens = firstProduct / 10 ∧
  n.ones = firstProduct % 10 ∧
  secondProduct = firstProduct % 10

theorem unique_solution :
  ∃! n : ThreeDigitNumber, satisfiesConditions n ∧ n.value = 144 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2747_274750


namespace NUMINAMATH_CALUDE_stratified_sampling_l2747_274766

theorem stratified_sampling (total_employees : ℕ) (employees_over_30 : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 49)
  (h2 : employees_over_30 = 14)
  (h3 : sample_size = 7) :
  ↑employees_over_30 / ↑total_employees * ↑sample_size = 2 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2747_274766


namespace NUMINAMATH_CALUDE_second_interest_rate_is_20_percent_l2747_274728

/-- Given a total amount, an amount at 10% interest, and a total profit,
    calculate the second interest rate. -/
def calculate_second_interest_rate (total_amount : ℕ) (amount_at_10_percent : ℕ) (total_profit : ℕ) : ℚ :=
  let amount_at_second_rate := total_amount - amount_at_10_percent
  let interest_from_first_part := (10 : ℚ) / 100 * amount_at_10_percent
  let interest_from_second_part := total_profit - interest_from_first_part
  (interest_from_second_part * 100) / amount_at_second_rate

/-- Theorem stating that under the given conditions, the second interest rate is 20%. -/
theorem second_interest_rate_is_20_percent :
  calculate_second_interest_rate 80000 70000 9000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_interest_rate_is_20_percent_l2747_274728


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2747_274732

theorem constant_term_expansion (α : Real) 
  (h : Real.sin (π - α) = 2 * Real.cos α) : 
  (Finset.range 7).sum (fun k => 
    (Nat.choose 6 k : Real) * 
    (Real.tan α)^k * 
    ((-1)^k * Nat.choose 6 (6-k))) = 160 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2747_274732


namespace NUMINAMATH_CALUDE_reading_competition_result_l2747_274787

/-- Represents the number of pages read by each girl --/
structure Pages where
  sasa : ℕ
  zuzka : ℕ
  ivana : ℕ
  majka : ℕ
  lucka : ℕ

/-- The conditions of the reading competition --/
def reading_conditions (p : Pages) : Prop :=
  p.lucka = 32 ∧
  p.lucka = (p.sasa + p.zuzka) / 2 ∧
  p.ivana = p.zuzka + 5 ∧
  p.majka = p.sasa - 8 ∧
  p.ivana = (p.majka + p.zuzka) / 2

/-- The theorem stating the correct number of pages read by each girl --/
theorem reading_competition_result :
  ∃ (p : Pages), reading_conditions p ∧
    p.sasa = 41 ∧ p.zuzka = 23 ∧ p.ivana = 28 ∧ p.majka = 33 ∧ p.lucka = 32 := by
  sorry

end NUMINAMATH_CALUDE_reading_competition_result_l2747_274787


namespace NUMINAMATH_CALUDE_a_5_equals_18_l2747_274789

/-- For a sequence defined by a_n = n^2 - 2n + 3, a_5 = 18 -/
theorem a_5_equals_18 :
  let a : ℕ → ℤ := λ n => n^2 - 2*n + 3
  a 5 = 18 := by sorry

end NUMINAMATH_CALUDE_a_5_equals_18_l2747_274789


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2747_274704

theorem quadratic_two_distinct_roots :
  let a : ℝ := 1
  let b : ℝ := 2
  let c : ℝ := -1
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ ∃ x y : ℝ, x ≠ y ∧ a*x^2 + b*x + c = 0 ∧ a*y^2 + b*y + c = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2747_274704


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2747_274764

theorem inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, (0 < x ∧ x < 2) ↔ ((m - 1) * x < Real.sqrt (4 * x - x^2))) → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2747_274764


namespace NUMINAMATH_CALUDE_unique_number_with_remainder_l2747_274768

theorem unique_number_with_remainder (n : ℕ) : n < 5000 ∧ 
  (∀ k ∈ Finset.range 9, n % (k + 2) = 1) ↔ n = 2521 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_with_remainder_l2747_274768


namespace NUMINAMATH_CALUDE_inequality_proof_l2747_274751

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a / (a^2 + 2)) + (b / (b^2 + 2)) + (c / (c^2 + 2)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2747_274751


namespace NUMINAMATH_CALUDE_parallelogram_count_l2747_274733

/-- Given two sets of parallel lines in a plane, prove the number of parallelograms formed -/
theorem parallelogram_count (m n : ℕ) : ℕ := by
  /- m is the number of lines in the first set -/
  /- n is the number of lines in the second set -/
  /- The two sets of lines are parallel and intersect -/
  /- The number of parallelograms formed is Combination(m,2) * Combination(n,2) -/
  sorry

#check parallelogram_count

end NUMINAMATH_CALUDE_parallelogram_count_l2747_274733


namespace NUMINAMATH_CALUDE_B_and_C_complementary_l2747_274798

open Set

-- Define the sample space for a fair cubic die
def Ω : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define event B
def B : Set Nat := {n ∈ Ω | n ≤ 3}

-- Define event C
def C : Set Nat := {n ∈ Ω | n ≥ 4}

-- Theorem statement
theorem B_and_C_complementary : B ∪ C = Ω ∧ B ∩ C = ∅ := by
  sorry

end NUMINAMATH_CALUDE_B_and_C_complementary_l2747_274798


namespace NUMINAMATH_CALUDE_range_of_m_l2747_274744

def p (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a = 8 - m ∧ b = 2 * m - 1

def q (m : ℝ) : Prop :=
  (m + 1) * (m - 2) < 0

theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Set.Ioo (-1 : ℝ) (1/2) ∪ Set.Ico 2 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2747_274744


namespace NUMINAMATH_CALUDE_usual_time_to_school_l2747_274731

/-- The usual time for a boy to reach school, given that when he walks 7/6 of his usual rate,
    he reaches school 2 minutes early. -/
theorem usual_time_to_school (usual_rate : ℝ) (usual_time : ℝ) 
    (h1 : usual_rate > 0) (h2 : usual_time > 0)
    (h3 : usual_rate * usual_time = (7/6 * usual_rate) * (usual_time - 2)) : 
  usual_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_usual_time_to_school_l2747_274731


namespace NUMINAMATH_CALUDE_topological_subgraph_is_subgraph_max_degree_3_topological_subgraph_iff_subgraph_l2747_274743

/-- A graph. -/
structure Graph (V : Type) where
  edge : V → V → Prop

/-- The maximum degree of a graph. -/
def maxDegree {V : Type} (G : Graph V) : ℕ := sorry

/-- A subgraph relation between graphs. -/
def isSubgraph {V : Type} (H G : Graph V) : Prop := sorry

/-- A topological subgraph relation between graphs. -/
def isTopologicalSubgraph {V : Type} (H G : Graph V) : Prop := sorry

theorem topological_subgraph_is_subgraph {V : Type} (G H : Graph V) :
  isTopologicalSubgraph H G → isSubgraph H G := by sorry

theorem max_degree_3_topological_subgraph_iff_subgraph {V : Type} (G H : Graph V) :
  maxDegree G ≤ 3 →
  (isTopologicalSubgraph H G ↔ isSubgraph H G) := by sorry

end NUMINAMATH_CALUDE_topological_subgraph_is_subgraph_max_degree_3_topological_subgraph_iff_subgraph_l2747_274743


namespace NUMINAMATH_CALUDE_medical_team_arrangements_l2747_274736

/-- The number of male doctors --/
def num_male_doctors : ℕ := 6

/-- The number of female nurses --/
def num_female_nurses : ℕ := 3

/-- The number of medical teams --/
def num_teams : ℕ := 3

/-- The number of male doctors per team --/
def doctors_per_team : ℕ := 2

/-- The number of female nurses per team --/
def nurses_per_team : ℕ := 1

/-- The number of distinct locations --/
def num_locations : ℕ := 3

/-- The total number of arrangements --/
def total_arrangements : ℕ := 540

theorem medical_team_arrangements :
  (num_male_doctors.choose doctors_per_team *
   (num_male_doctors - doctors_per_team).choose doctors_per_team *
   (num_male_doctors - 2 * doctors_per_team).choose doctors_per_team) /
  num_teams.factorial *
  num_teams.factorial *
  num_teams.factorial = total_arrangements :=
sorry

end NUMINAMATH_CALUDE_medical_team_arrangements_l2747_274736


namespace NUMINAMATH_CALUDE_negative_option_l2747_274713

theorem negative_option : ∀ (x : ℝ), 
  (x = -(-3) ∨ x = -|5| ∨ x = 1/2 ∨ x = 0) → 
  (x < 0 ↔ x = -|5|) := by
sorry

end NUMINAMATH_CALUDE_negative_option_l2747_274713


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l2747_274703

theorem quadratic_roots_real_and_equal : ∃ x : ℝ, 
  (∀ y : ℝ, y^2 + 4*y*Real.sqrt 2 + 8 = 0 ↔ y = x) ∧ 
  (x^2 + 4*x*Real.sqrt 2 + 8 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_real_and_equal_l2747_274703


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2747_274762

def repeating_decimal_1 : ℚ := 1/3
def repeating_decimal_2 : ℚ := 4/99
def repeating_decimal_3 : ℚ := 5/999

theorem sum_of_repeating_decimals :
  repeating_decimal_1 + repeating_decimal_2 + repeating_decimal_3 = 42/111 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l2747_274762


namespace NUMINAMATH_CALUDE_probability_is_one_third_l2747_274773

-- Define the number of hot dishes
def num_dishes : ℕ := 3

-- Define the number of dishes a student can choose
def num_choices : ℕ := 2

-- Define the function to calculate combinations
def combinations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the probability of two students choosing the same two dishes
def probability_same_choices : ℚ :=
  (combinations num_dishes num_choices) / (combinations num_dishes num_choices * combinations num_dishes num_choices)

-- Theorem to prove
theorem probability_is_one_third :
  probability_same_choices = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_probability_is_one_third_l2747_274773


namespace NUMINAMATH_CALUDE_smallest_prime_with_prime_digit_sum_l2747_274700

def digit_sum (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + digit_sum (n / 10)

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_prime_digit_sum :
  ∃ (p : Nat), is_prime p ∧ 
               is_prime (digit_sum p) ∧ 
               digit_sum p > 10 ∧ 
               (∀ q : Nat, q < p → ¬(is_prime q ∧ is_prime (digit_sum q) ∧ digit_sum q > 10)) ∧
               p = 29 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_with_prime_digit_sum_l2747_274700


namespace NUMINAMATH_CALUDE_baron_munchausen_theorem_l2747_274741

theorem baron_munchausen_theorem :
  ∀ (a b : ℕ+), ∃ (n : ℕ+), ∃ (k m : ℕ+), (a * n = k ^ 2) ∧ (b * n = m ^ 3) := by
  sorry

end NUMINAMATH_CALUDE_baron_munchausen_theorem_l2747_274741


namespace NUMINAMATH_CALUDE_cos_five_pi_fourth_plus_x_l2747_274763

theorem cos_five_pi_fourth_plus_x (x : ℝ) (h : Real.sin (π / 4 - x) = -1 / 5) :
  Real.cos (5 * π / 4 + x) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_cos_five_pi_fourth_plus_x_l2747_274763


namespace NUMINAMATH_CALUDE_correct_payment_to_C_l2747_274784

/-- The amount to be paid to worker C -/
def payment_to_C (a_rate b_rate : ℚ) (total_payment : ℕ) (days_to_complete : ℕ) : ℚ :=
  let ab_rate := a_rate + b_rate
  let ab_work := ab_rate * days_to_complete
  let c_work := 1 - ab_work
  c_work * total_payment

/-- Theorem stating the correct payment to worker C -/
theorem correct_payment_to_C :
  payment_to_C (1/6) (1/8) 2400 3 = 300 := by sorry

end NUMINAMATH_CALUDE_correct_payment_to_C_l2747_274784


namespace NUMINAMATH_CALUDE_ab_relation_to_a_over_b_l2747_274780

theorem ab_relation_to_a_over_b (a b : ℝ) (h : a * b ≠ 0) :
  ¬(∀ a b, a * b > 1 → a > 1 / b) ∧
  ¬(∀ a b, a > 1 / b → a * b > 1) := by
  sorry

end NUMINAMATH_CALUDE_ab_relation_to_a_over_b_l2747_274780


namespace NUMINAMATH_CALUDE_subtraction_of_negative_l2747_274718

theorem subtraction_of_negative : 4 - (-7) = 11 := by sorry

end NUMINAMATH_CALUDE_subtraction_of_negative_l2747_274718


namespace NUMINAMATH_CALUDE_line_slope_l2747_274712

def curve (x y : ℝ) : Prop := 5 * y = 2 * x^2 - 9 * x + 10

def line_through_origin (k x y : ℝ) : Prop := y = k * x

theorem line_slope (k : ℝ) :
  (∃ x₁ x₂ y₁ y₂ : ℝ,
    curve x₁ y₁ ∧ curve x₂ y₂ ∧
    line_through_origin k x₁ y₁ ∧
    line_through_origin k x₂ y₂ ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 77) →
  k = 29 := by
sorry

end NUMINAMATH_CALUDE_line_slope_l2747_274712


namespace NUMINAMATH_CALUDE_rachels_age_problem_l2747_274742

/-- Rachel's age problem -/
theorem rachels_age_problem 
  (rachel_age : ℕ)
  (grandfather_age : ℕ)
  (mother_age : ℕ)
  (father_age : ℕ)
  (h1 : rachel_age = 12)
  (h2 : grandfather_age = 7 * rachel_age)
  (h3 : mother_age = grandfather_age / 2)
  (h4 : father_age + (25 - rachel_age) = 60) :
  father_age - mother_age = 5 := by
sorry

end NUMINAMATH_CALUDE_rachels_age_problem_l2747_274742


namespace NUMINAMATH_CALUDE_least_sum_of_exponents_l2747_274777

def target_number : ℕ := 3124

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (fun e => 2^e)).sum ∧ exponents.Nodup

theorem least_sum_of_exponents :
  ∃ (exponents : List ℕ),
    is_sum_of_distinct_powers_of_two target_number exponents ∧
    ∀ (other_exponents : List ℕ),
      is_sum_of_distinct_powers_of_two target_number other_exponents →
      exponents.sum ≤ other_exponents.sum ∧
      exponents.sum = 32 :=
sorry

end NUMINAMATH_CALUDE_least_sum_of_exponents_l2747_274777


namespace NUMINAMATH_CALUDE_algebraic_identity_l2747_274748

theorem algebraic_identity (a b : ℝ) : 2 * a * b - a^2 - b^2 = -((a - b)^2) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_identity_l2747_274748


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_term_l2747_274755

theorem gcd_of_polynomial_and_linear_term (b : ℤ) (h : 1620 ∣ b) :
  Nat.gcd (Int.natAbs (b^2 + 11*b + 36)) (Int.natAbs (b + 6)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_linear_term_l2747_274755


namespace NUMINAMATH_CALUDE_triangular_pyramid_inequality_l2747_274716

-- Define a structure for a triangular pyramid
structure TriangularPyramid where
  -- We don't need to explicitly define vertices A, B, C, D
  -- as they are implicit in the following measurements
  R : ℝ  -- radius of circumscribed sphere
  r : ℝ  -- radius of inscribed sphere
  a : ℝ  -- length of longest edge
  h : ℝ  -- length of shortest altitude

-- State the theorem
theorem triangular_pyramid_inequality (pyramid : TriangularPyramid) :
  pyramid.R / pyramid.r > pyramid.a / pyramid.h := by
  sorry

end NUMINAMATH_CALUDE_triangular_pyramid_inequality_l2747_274716


namespace NUMINAMATH_CALUDE_geometric_series_equality_l2747_274776

theorem geometric_series_equality (n : ℕ) : n ≥ 1 → (
  let C : ℕ → ℚ := λ k => 1320 * (1 - 1 / 3^k)
  let D : ℕ → ℚ := λ k => 1008 * (1 - 1 / (-3)^k)
  (∃ k ≥ 1, C k = D k) ∧ (∀ m ≥ 1, m < n → C m ≠ D m) → n = 2
) := by sorry

end NUMINAMATH_CALUDE_geometric_series_equality_l2747_274776


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2747_274795

theorem sufficient_not_necessary_condition :
  (∀ b : ℝ, b ∈ Set.Ioo 0 4 → ∀ x : ℝ, b * x^2 - b * x + 1 > 0) ∧
  (∃ b : ℝ, b ∉ Set.Ioo 0 4 ∧ ∀ x : ℝ, b * x^2 - b * x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2747_274795


namespace NUMINAMATH_CALUDE_total_crayons_l2747_274702

/-- Given that each child has 12 crayons and there are 18 children, 
    prove that the total number of crayons is 216. -/
theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
  (h1 : crayons_per_child = 12) (h2 : num_children = 18) : 
  crayons_per_child * num_children = 216 := by
sorry

end NUMINAMATH_CALUDE_total_crayons_l2747_274702


namespace NUMINAMATH_CALUDE_horner_v2_value_l2747_274746

/-- Horner's method for a polynomial --/
def horner_step (x : ℝ) (a b : ℝ) : ℝ := a * x + b

/-- The polynomial f(x) = x^6 + 6x^4 + 9x^2 + 208 --/
def f (x : ℝ) : ℝ := x^6 + 6*x^4 + 9*x^2 + 208

/-- Theorem: The value of v₂ in Horner's method for f(x) at x = -4 is 22 --/
theorem horner_v2_value :
  let x : ℝ := -4
  let v0 : ℝ := 1
  let v1 : ℝ := horner_step x v0 0
  let v2 : ℝ := horner_step x v1 6
  v2 = 22 := by sorry

end NUMINAMATH_CALUDE_horner_v2_value_l2747_274746


namespace NUMINAMATH_CALUDE_min_quotient_four_digit_number_l2747_274778

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧
  (∃ a b c d : ℕ, n = 1000 * a + 100 * b + 10 * c + d ∧
    0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    (Even a ∨ Even b ∨ Even c ∨ Even d) ∧
    (Even a ∧ Even b ∨ Even a ∧ Even c ∨ Even a ∧ Even d ∨
     Even b ∧ Even c ∨ Even b ∧ Even d ∨ Even c ∧ Even d))

def digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum digits

theorem min_quotient_four_digit_number :
  ∀ n : ℕ, is_valid_number n → (n : ℚ) / (digit_sum n : ℚ) ≥ 87 :=
by sorry

end NUMINAMATH_CALUDE_min_quotient_four_digit_number_l2747_274778


namespace NUMINAMATH_CALUDE_decreasing_reciprocal_function_l2747_274714

theorem decreasing_reciprocal_function 
  (f : ℝ → ℝ) 
  (h_f : ∀ x, x > 0 → f x = 1 / x) :
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂ := by
sorry

end NUMINAMATH_CALUDE_decreasing_reciprocal_function_l2747_274714


namespace NUMINAMATH_CALUDE_sam_tutoring_hours_l2747_274757

/-- Sam's hourly rate for Math tutoring -/
def hourly_rate : ℕ := 10

/-- Sam's earnings for the first month -/
def first_month_earnings : ℕ := 200

/-- The additional amount Sam earned in the second month compared to the first -/
def second_month_increase : ℕ := 150

/-- The total number of hours Sam spent tutoring for two months -/
def total_hours : ℕ := 55

/-- Theorem stating that given the conditions, Sam spent 55 hours tutoring over two months -/
theorem sam_tutoring_hours :
  hourly_rate * total_hours = first_month_earnings + (first_month_earnings + second_month_increase) :=
by sorry

end NUMINAMATH_CALUDE_sam_tutoring_hours_l2747_274757


namespace NUMINAMATH_CALUDE_two_sector_area_l2747_274756

/-- The area of a figure formed by two sectors of a circle -/
theorem two_sector_area (r : ℝ) (angle1 angle2 : ℝ) (h1 : r = 15) (h2 : angle1 = 90) (h3 : angle2 = 45) :
  (angle1 / 360) * π * r^2 + (angle2 / 360) * π * r^2 = 84.375 * π := by
  sorry

#check two_sector_area

end NUMINAMATH_CALUDE_two_sector_area_l2747_274756


namespace NUMINAMATH_CALUDE_refrigerator_installment_l2747_274792

/-- Calculates the monthly installment amount for a purchase --/
def monthly_installment (cash_price deposit num_installments cash_savings : ℕ) : ℕ :=
  ((cash_price + cash_savings - deposit) / num_installments)

theorem refrigerator_installment :
  monthly_installment 8000 3000 30 4000 = 300 := by
  sorry

end NUMINAMATH_CALUDE_refrigerator_installment_l2747_274792


namespace NUMINAMATH_CALUDE_smallest_valid_assembly_is_four_l2747_274759

/-- Represents a modified cube with snaps and receptacles -/
structure ModifiedCube :=
  (snaps : Fin 2)
  (receptacles : Fin 4)

/-- Represents an assembly of modified cubes -/
structure CubeAssembly :=
  (cubes : List ModifiedCube)
  (all_snaps_covered : Bool)
  (only_receptacles_visible : Bool)

/-- Returns true if the assembly is valid according to the problem constraints -/
def is_valid_assembly (assembly : CubeAssembly) : Prop :=
  assembly.all_snaps_covered ∧ assembly.only_receptacles_visible

/-- The smallest number of cubes needed for a valid assembly -/
def smallest_valid_assembly : ℕ := 4

/-- Theorem stating that the smallest valid assembly consists of 4 cubes -/
theorem smallest_valid_assembly_is_four :
  ∀ (assembly : CubeAssembly),
    is_valid_assembly assembly →
    assembly.cubes.length ≥ smallest_valid_assembly :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_assembly_is_four_l2747_274759


namespace NUMINAMATH_CALUDE_angle_negative_1445_quadrant_l2747_274749

theorem angle_negative_1445_quadrant : 
  ∃ (k : ℤ) (θ : ℝ), -1445 = 360 * k + θ ∧ 270 < θ ∧ θ ≤ 360 :=
sorry

end NUMINAMATH_CALUDE_angle_negative_1445_quadrant_l2747_274749


namespace NUMINAMATH_CALUDE_product_expansion_l2747_274770

theorem product_expansion (x : ℝ) : 
  (7 * x^2 + 3) * (5 * x^3 + 2 * x + 1) = 35 * x^5 + 29 * x^3 + 7 * x^2 + 6 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_product_expansion_l2747_274770


namespace NUMINAMATH_CALUDE_complement_of_union_l2747_274767

def S : Finset Nat := {1,2,3,4,5,6,7,8,9,10}
def A : Finset Nat := {2,4,6,8,10}
def B : Finset Nat := {3,6,9}

theorem complement_of_union (S A B : Finset Nat) :
  S = {1,2,3,4,5,6,7,8,9,10} →
  A = {2,4,6,8,10} →
  B = {3,6,9} →
  S \ (A ∪ B) = {1,5,7} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_union_l2747_274767


namespace NUMINAMATH_CALUDE_intersection_points_theorem_l2747_274707

/-- The maximum number of intersection points in the first quadrant
    given 15 points on the x-axis and 10 points on the y-axis -/
def max_intersection_points (x_points y_points : ℕ) : ℕ :=
  (x_points.choose 2) * (y_points.choose 2)

/-- Theorem stating the maximum number of intersection points -/
theorem intersection_points_theorem :
  max_intersection_points 15 10 = 4725 := by
  sorry

end NUMINAMATH_CALUDE_intersection_points_theorem_l2747_274707


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l2747_274775

theorem trigonometric_expression_equality : 
  (Real.tan (150 * π / 180)) * (Real.cos (-210 * π / 180)) * (Real.sin (-420 * π / 180)) / 
  ((Real.sin (1050 * π / 180)) * (Real.cos (-600 * π / 180))) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l2747_274775
