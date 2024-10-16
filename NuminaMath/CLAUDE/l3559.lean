import Mathlib

namespace NUMINAMATH_CALUDE_ratio_independence_l3559_355979

/-- Two infinite increasing arithmetic progressions of positive numbers -/
def ArithmeticProgression (a : ℕ → ℚ) : Prop :=
  ∃ (first d : ℚ), first > 0 ∧ d > 0 ∧ ∀ k, a k = first + k * d

/-- The theorem statement -/
theorem ratio_independence
  (a b : ℕ → ℚ)
  (ha : ArithmeticProgression a)
  (hb : ArithmeticProgression b)
  (h_int_ratio : ∀ k, ∃ m : ℤ, a k = m * b k) :
  ∃ c : ℚ, ∀ k, a k = c * b k :=
sorry

end NUMINAMATH_CALUDE_ratio_independence_l3559_355979


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3559_355959

theorem right_triangle_hypotenuse (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) 
  (h3 : c^2 = a^2 + b^2) : c = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3559_355959


namespace NUMINAMATH_CALUDE_complement_A_complement_A_inter_B_l3559_355961

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def B : Set ℝ := {x | x < 3 ∨ x ≥ 15}

-- State the theorems
theorem complement_A : Set.compl A = {x | x ≤ -1 ∨ x > 5} := by sorry

theorem complement_A_inter_B : Set.compl (A ∩ B) = {x | x ≤ -1 ∨ x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_complement_A_inter_B_l3559_355961


namespace NUMINAMATH_CALUDE_square_ratio_proof_l3559_355978

theorem square_ratio_proof (a b : ℝ) (h : a > 0 ∧ b > 0) (h_ratio : a^2 / b^2 = 75 / 98) :
  ∃ (x y z : ℕ), 
    (Real.sqrt (a / b) = x * Real.sqrt 6 / (y : ℝ)) ∧ 
    (x + 6 + y = z) ∧
    x = 5 ∧ y = 14 ∧ z = 25 := by
  sorry


end NUMINAMATH_CALUDE_square_ratio_proof_l3559_355978


namespace NUMINAMATH_CALUDE_barbara_has_winning_strategy_l3559_355936

/-- A game played on a matrix where two players alternately fill entries --/
structure MatrixGame where
  n : ℕ
  entries : Fin n → Fin n → ℝ

/-- A strategy for the second player in the matrix game --/
def SecondPlayerStrategy (n : ℕ) := 
  (Fin n → Fin n → ℝ) → Fin n → Fin n → ℝ

/-- The determinant of a matrix is zero if two of its rows are identical --/
axiom det_zero_if_identical_rows {n : ℕ} (M : Fin n → Fin n → ℝ) :
  (∃ i j, i ≠ j ∧ (∀ k, M i k = M j k)) → Matrix.det M = 0

/-- The second player can always make two rows identical --/
axiom second_player_can_make_identical_rows (n : ℕ) :
  ∃ (strategy : SecondPlayerStrategy n),
    ∀ (game : MatrixGame),
    game.n = n →
    ∃ i j, i ≠ j ∧ (∀ k, game.entries i k = game.entries j k)

theorem barbara_has_winning_strategy :
  ∃ (strategy : SecondPlayerStrategy 2008),
    ∀ (game : MatrixGame),
    game.n = 2008 →
    Matrix.det game.entries = 0 := by
  sorry

end NUMINAMATH_CALUDE_barbara_has_winning_strategy_l3559_355936


namespace NUMINAMATH_CALUDE_tangent_line_equations_l3559_355943

/-- The function f(x) = x³ + 2 -/
def f (x : ℝ) : ℝ := x^3 + 2

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2

theorem tangent_line_equations (x : ℝ) :
  /- Part 1: Tangent line equation at x = 1 -/
  (∀ y : ℝ, (y - f 1) = f' 1 * (x - 1) ↔ 3 * x - y = 0) ∧
  /- Part 2: Tangent line equation passing through (0, 4) -/
  (∃ t : ℝ, t^3 + 2 = f t ∧
            4 - (t^3 + 2) = f' t * (0 - t) ∧
            (∀ y : ℝ, (y - f t) = f' t * (x - t) ↔ 3 * x - y + 4 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equations_l3559_355943


namespace NUMINAMATH_CALUDE_inscribed_circle_angle_theorem_l3559_355948

/-- A triangle with an inscribed circle --/
structure InscribedCircleTriangle where
  /-- The angle at the tangent point on side BC --/
  angle_bc : ℝ
  /-- The angle at the tangent point on side CA --/
  angle_ca : ℝ
  /-- The angle at the tangent point on side AB --/
  angle_ab : ℝ
  /-- The sum of angles at tangent points is 360° --/
  sum_angles : angle_bc + angle_ca + angle_ab = 360

/-- Theorem: If the angles at tangent points are 120°, 130°, and θ°, then θ = 110° --/
theorem inscribed_circle_angle_theorem (t : InscribedCircleTriangle) 
    (h1 : t.angle_bc = 120) (h2 : t.angle_ca = 130) : t.angle_ab = 110 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_angle_theorem_l3559_355948


namespace NUMINAMATH_CALUDE_syrup_volume_proof_l3559_355971

def final_syrup_volume (original_volume : ℚ) (reduction_factor : ℚ) (sugar_added : ℚ) (cups_per_quart : ℚ) : ℚ :=
  original_volume * cups_per_quart * reduction_factor + sugar_added

theorem syrup_volume_proof (original_volume : ℚ) (reduction_factor : ℚ) (sugar_added : ℚ) (cups_per_quart : ℚ)
  (h1 : original_volume = 6)
  (h2 : reduction_factor = 1 / 12)
  (h3 : sugar_added = 1)
  (h4 : cups_per_quart = 4) :
  final_syrup_volume original_volume reduction_factor sugar_added cups_per_quart = 3 := by
  sorry

end NUMINAMATH_CALUDE_syrup_volume_proof_l3559_355971


namespace NUMINAMATH_CALUDE_cupcake_milk_calculation_l3559_355940

/-- The number of cupcakes in a full recipe -/
def full_recipe_cupcakes : ℕ := 24

/-- The number of quarts of milk needed for a full recipe -/
def full_recipe_quarts : ℕ := 3

/-- The number of pints in a quart -/
def pints_per_quart : ℕ := 2

/-- The number of cupcakes we want to make -/
def target_cupcakes : ℕ := 6

/-- The amount of milk in pints needed for the target number of cupcakes -/
def milk_needed : ℚ := 1.5

theorem cupcake_milk_calculation :
  (target_cupcakes : ℚ) * (full_recipe_quarts * pints_per_quart : ℚ) / full_recipe_cupcakes = milk_needed :=
sorry

end NUMINAMATH_CALUDE_cupcake_milk_calculation_l3559_355940


namespace NUMINAMATH_CALUDE_share_difference_l3559_355941

theorem share_difference (total : ℚ) (p m s : ℚ) : 
  total = 730 →
  p + m + s = total →
  4 * p = 3 * m →
  3 * m = 3.5 * s →
  m - s = 36.5 := by
  sorry

end NUMINAMATH_CALUDE_share_difference_l3559_355941


namespace NUMINAMATH_CALUDE_compacted_cans_space_l3559_355909

/-- The space occupied by compacted cans -/
def space_occupied (num_cans : ℕ) (initial_space : ℝ) (compaction_ratio : ℝ) : ℝ :=
  (num_cans : ℝ) * initial_space * compaction_ratio

/-- Theorem: 60 cans, each initially 30 sq inches, compacted to 20%, occupy 360 sq inches -/
theorem compacted_cans_space :
  space_occupied 60 30 0.2 = 360 := by
  sorry

end NUMINAMATH_CALUDE_compacted_cans_space_l3559_355909


namespace NUMINAMATH_CALUDE_eraser_boxes_donated_l3559_355934

theorem eraser_boxes_donated (erasers_per_box : ℕ) (price_per_eraser : ℚ) (total_money : ℚ) :
  erasers_per_box = 24 →
  price_per_eraser = 3/4 →
  total_money = 864 →
  (total_money / price_per_eraser) / erasers_per_box = 48 :=
by sorry

end NUMINAMATH_CALUDE_eraser_boxes_donated_l3559_355934


namespace NUMINAMATH_CALUDE_apartments_per_floor_l3559_355985

theorem apartments_per_floor (num_buildings : ℕ) (floors_per_building : ℕ) 
  (doors_per_apartment : ℕ) (total_doors : ℕ) :
  num_buildings = 2 →
  floors_per_building = 12 →
  doors_per_apartment = 7 →
  total_doors = 1008 →
  (total_doors / doors_per_apartment) / (num_buildings * floors_per_building) = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_apartments_per_floor_l3559_355985


namespace NUMINAMATH_CALUDE_carina_coffee_amount_l3559_355905

/-- Calculates the total amount of coffee Carina has given the number of 10-ounce packages -/
def total_coffee (num_ten_oz_packages : ℕ) : ℕ :=
  let num_five_oz_packages := num_ten_oz_packages + 2
  let oz_from_ten := 10 * num_ten_oz_packages
  let oz_from_five := 5 * num_five_oz_packages
  oz_from_ten + oz_from_five

/-- Proves that Carina has 115 ounces of coffee in total -/
theorem carina_coffee_amount : total_coffee 7 = 115 := by
  sorry

end NUMINAMATH_CALUDE_carina_coffee_amount_l3559_355905


namespace NUMINAMATH_CALUDE_opposite_of_sqrt_4_l3559_355995

theorem opposite_of_sqrt_4 : -(Real.sqrt 4) = -2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_sqrt_4_l3559_355995


namespace NUMINAMATH_CALUDE_expected_value_5X_plus_4_l3559_355984

/-- Distribution of random variable X -/
structure Distribution where
  p0 : ℝ
  p2 : ℝ
  p4 : ℝ
  sum_to_one : p0 + p2 + p4 = 1
  non_negative : p0 ≥ 0 ∧ p2 ≥ 0 ∧ p4 ≥ 0

/-- Expected value of a random variable -/
def expected_value (d : Distribution) : ℝ := 0 * d.p0 + 2 * d.p2 + 4 * d.p4

/-- Theorem: Expected value of 5X+4 equals 16 -/
theorem expected_value_5X_plus_4 (d : Distribution) 
  (h1 : d.p0 = 0.3) 
  (h2 : d.p4 = 0.5) : 
  5 * expected_value d + 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_5X_plus_4_l3559_355984


namespace NUMINAMATH_CALUDE_jessica_cut_thirteen_roses_l3559_355910

/-- The number of roses Jessica cut from her garden -/
def roses_cut (initial_roses vase_roses garden_roses : ℕ) : ℕ :=
  vase_roses - initial_roses

/-- Theorem stating that Jessica cut 13 roses -/
theorem jessica_cut_thirteen_roses :
  ∃ (initial_roses vase_roses garden_roses : ℕ),
    initial_roses = 7 ∧
    garden_roses = 59 ∧
    vase_roses = 20 ∧
    roses_cut initial_roses vase_roses garden_roses = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_cut_thirteen_roses_l3559_355910


namespace NUMINAMATH_CALUDE_savings_calculation_l3559_355957

/-- Calculates the total savings of Thomas and Joseph after 6 years -/
def total_savings (thomas_monthly_savings : ℚ) (years : ℕ) : ℚ :=
  let months : ℕ := years * 12
  let thomas_total : ℚ := thomas_monthly_savings * months
  let joseph_monthly_savings : ℚ := thomas_monthly_savings - (2 / 5) * thomas_monthly_savings
  let joseph_total : ℚ := joseph_monthly_savings * months
  thomas_total + joseph_total

/-- Proves that Thomas and Joseph's combined savings after 6 years equals $4608 -/
theorem savings_calculation : total_savings 40 6 = 4608 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l3559_355957


namespace NUMINAMATH_CALUDE_trig_identities_l3559_355908

theorem trig_identities (α : Real) (h : Real.tan α = 2) : 
  ((Real.sin (Real.pi - α) + Real.cos (α - Real.pi/2) - Real.cos (3*Real.pi + α)) / 
   (Real.cos (Real.pi/2 + α) - Real.sin (2*Real.pi + α) + 2*Real.sin (α - Real.pi/2)) = -5/6) ∧ 
  (Real.cos (2*α) + Real.sin α * Real.cos α = -1/5) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l3559_355908


namespace NUMINAMATH_CALUDE_nested_fraction_simplification_l3559_355907

theorem nested_fraction_simplification :
  2 + 3 / (4 + 5 / 6) = 76 / 29 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_simplification_l3559_355907


namespace NUMINAMATH_CALUDE_chairs_per_table_l3559_355902

theorem chairs_per_table (indoor_tables outdoor_tables total_chairs : ℕ) 
  (h1 : indoor_tables = 8)
  (h2 : outdoor_tables = 12)
  (h3 : total_chairs = 60) :
  ∃ (chairs_per_table : ℕ), 
    chairs_per_table * (indoor_tables + outdoor_tables) = total_chairs ∧ 
    chairs_per_table = 3 := by
  sorry

end NUMINAMATH_CALUDE_chairs_per_table_l3559_355902


namespace NUMINAMATH_CALUDE_set_difference_equiv_l3559_355962

def A : Set ℝ := {x | ∃ y, y = Real.sqrt (1 - x) ∧ -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {y | ∃ x, y = x^2 ∧ -1 ≤ x ∧ x ≤ 1}

theorem set_difference_equiv : A \ B = {x | x < 0} := by sorry

end NUMINAMATH_CALUDE_set_difference_equiv_l3559_355962


namespace NUMINAMATH_CALUDE_remaining_money_l3559_355987

def money_spent_on_books : ℝ := 76.8
def money_spent_on_apples : ℝ := 12
def total_money_brought : ℝ := 100

theorem remaining_money :
  total_money_brought - money_spent_on_books - money_spent_on_apples = 11.2 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l3559_355987


namespace NUMINAMATH_CALUDE_eliminate_denominators_l3559_355966

theorem eliminate_denominators (x : ℝ) : 
  ((x + 1) / 2 + 1 = x / 3) ↔ (3 * (x + 1) + 6 = 2 * x) := by sorry

end NUMINAMATH_CALUDE_eliminate_denominators_l3559_355966


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_l3559_355988

/-- A geometric sequence with a_1 = 1/5 and a_3 = 5 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  a 1 = 1/5 ∧ a 3 = 5 ∧ ∀ n : ℕ, a (n + 1) = a n * (a 2 / a 1)

theorem geometric_sequence_a2 (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 2 = 1 ∨ a 2 = -1 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_l3559_355988


namespace NUMINAMATH_CALUDE_root_product_expression_l3559_355919

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 - 2*p*α + 1 = 0) → 
  (β^2 - 2*p*β + 1 = 0) → 
  (γ^2 + q*γ + 2 = 0) → 
  (δ^2 + q*δ + 2 = 0) → 
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = 2*(p - q)^2 :=
by sorry

end NUMINAMATH_CALUDE_root_product_expression_l3559_355919


namespace NUMINAMATH_CALUDE_food_waste_scientific_notation_l3559_355958

/-- The amount of food wasted in China annually in kilograms -/
def food_waste : ℕ := 500000000000

/-- Prove that the food waste in China is equivalent to 5 × 10^10 kg -/
theorem food_waste_scientific_notation : food_waste = 5 * (10 ^ 10) := by
  sorry

end NUMINAMATH_CALUDE_food_waste_scientific_notation_l3559_355958


namespace NUMINAMATH_CALUDE_inequality_proof_l3559_355901

theorem inequality_proof (a b c : ℝ) : a^2 + 4*b^2 + 9*c^2 ≥ 2*a*b + 3*a*c + 6*b*c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3559_355901


namespace NUMINAMATH_CALUDE_payroll_tax_threshold_l3559_355974

/-- The payroll tax problem -/
theorem payroll_tax_threshold (tax_rate : ℝ) (tax_paid : ℝ) (total_payroll : ℝ) (T : ℝ) : 
  tax_rate = 0.002 →
  tax_paid = 200 →
  total_payroll = 300000 →
  tax_paid = (total_payroll - T) * tax_rate →
  T = 200000 := by
  sorry


end NUMINAMATH_CALUDE_payroll_tax_threshold_l3559_355974


namespace NUMINAMATH_CALUDE_race_track_circumference_difference_l3559_355998

/-- The difference in circumferences of two concentric circles, where the outer circle's radius is 8 feet more than the inner circle's radius of 15 feet, is equal to 16π feet. -/
theorem race_track_circumference_difference : 
  let inner_radius : ℝ := 15
  let outer_radius : ℝ := inner_radius + 8
  let inner_circumference : ℝ := 2 * Real.pi * inner_radius
  let outer_circumference : ℝ := 2 * Real.pi * outer_radius
  outer_circumference - inner_circumference = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_race_track_circumference_difference_l3559_355998


namespace NUMINAMATH_CALUDE_puzzle_solving_time_l3559_355956

/-- The total time spent solving puzzles given a warm-up puzzle and two longer puzzles -/
theorem puzzle_solving_time (warm_up_time : ℕ) (num_long_puzzles : ℕ) (long_puzzle_factor : ℕ) : 
  warm_up_time = 10 → 
  num_long_puzzles = 2 → 
  long_puzzle_factor = 3 → 
  warm_up_time + num_long_puzzles * (long_puzzle_factor * warm_up_time) = 70 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solving_time_l3559_355956


namespace NUMINAMATH_CALUDE_combined_salaries_l3559_355931

/-- Given the salary of A and the average salary of A, B, C, D, and E,
    prove the combined salaries of B, C, D, and E. -/
theorem combined_salaries
  (salary_A : ℕ)
  (average_salary : ℕ)
  (h1 : salary_A = 10000)
  (h2 : average_salary = 8400) :
  salary_A + (4 * ((5 * average_salary) - salary_A)) = 42000 :=
by sorry

end NUMINAMATH_CALUDE_combined_salaries_l3559_355931


namespace NUMINAMATH_CALUDE_wire_cutting_l3559_355920

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 21 →
  ratio = 2 / 5 →
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 6 := by
sorry

end NUMINAMATH_CALUDE_wire_cutting_l3559_355920


namespace NUMINAMATH_CALUDE_parabola_directrix_l3559_355915

/-- Represents a parabola with equation x² = 4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = 4*y

/-- The directrix of a parabola -/
def directrix (p : Parabola) : ℝ → Prop :=
  fun y => y = -1

theorem parabola_directrix (p : Parabola) : 
  directrix p = fun y => y = -1 := by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3559_355915


namespace NUMINAMATH_CALUDE_sin_2a_minus_pi_6_l3559_355975

theorem sin_2a_minus_pi_6 (a : ℝ) (h : Real.sin (π / 3 - a) = 1 / 4) :
  Real.sin (2 * a - π / 6) = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2a_minus_pi_6_l3559_355975


namespace NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l3559_355990

/-- A function that returns true if a natural number is prime, false otherwise -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that returns the nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A function that returns true if all numbers in a list are non-prime, false otherwise -/
def allNonPrime (list : List ℕ) : Prop := sorry

theorem smallest_prime_after_seven_nonprimes :
  ∃ (n : ℕ), 
    (isPrime (nthPrime n)) ∧ 
    (allNonPrime [nthPrime (n-1) + 1, nthPrime (n-1) + 2, nthPrime (n-1) + 3, 
                  nthPrime (n-1) + 4, nthPrime (n-1) + 5, nthPrime (n-1) + 6, 
                  nthPrime (n-1) + 7]) ∧
    (nthPrime n = 67) ∧
    (∀ (m : ℕ), m < n → 
      ¬(isPrime (nthPrime m) ∧ 
        allNonPrime [nthPrime (m-1) + 1, nthPrime (m-1) + 2, nthPrime (m-1) + 3, 
                     nthPrime (m-1) + 4, nthPrime (m-1) + 5, nthPrime (m-1) + 6, 
                     nthPrime (m-1) + 7])) :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_after_seven_nonprimes_l3559_355990


namespace NUMINAMATH_CALUDE_value_range_of_f_l3559_355968

def f (x : ℝ) := x^2 - 4*x

theorem value_range_of_f :
  ∀ x ∈ Set.Icc 0 5, -4 ≤ f x ∧ f x ≤ 5 ∧
  (∃ x₁ ∈ Set.Icc 0 5, f x₁ = -4) ∧
  (∃ x₂ ∈ Set.Icc 0 5, f x₂ = 5) :=
sorry

end NUMINAMATH_CALUDE_value_range_of_f_l3559_355968


namespace NUMINAMATH_CALUDE_charity_raffle_winnings_l3559_355992

theorem charity_raffle_winnings (X : ℝ) : 
  let remaining_after_donation := 0.75 * X
  let remaining_after_lunch := remaining_after_donation * 0.9
  let remaining_after_gift := remaining_after_lunch * 0.85
  let amount_for_investment := remaining_after_gift * 0.3
  let investment_return := amount_for_investment * 0.5
  let final_amount := remaining_after_gift - amount_for_investment + investment_return
  final_amount = 320 → X = 485 :=
by sorry

end NUMINAMATH_CALUDE_charity_raffle_winnings_l3559_355992


namespace NUMINAMATH_CALUDE_abc_inequality_l3559_355913

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  (a * b * c) / (a + b + c) ≤ 4/3 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3559_355913


namespace NUMINAMATH_CALUDE_no_universal_divisibility_l3559_355972

/-- Represents a nonzero digit (1-9) -/
def NonzeroDigit := {d : Nat // d ≥ 1 ∧ d ≤ 9}

/-- Concatenates three numbers to form a new number -/
def concat3 (a : NonzeroDigit) (n : Nat) (b : NonzeroDigit) : Nat :=
  100 * a.val + 10 * n + b.val

/-- Concatenates two numbers to form a new number -/
def concat2 (a b : NonzeroDigit) : Nat :=
  10 * a.val + b.val

/-- Statement: There does not exist a natural number n such that
    for all nonzero digits a and b, concat3 a n b is divisible by concat2 a b -/
theorem no_universal_divisibility :
  ¬ ∃ n : Nat, ∀ (a b : NonzeroDigit), (concat3 a n b) % (concat2 a b) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_universal_divisibility_l3559_355972


namespace NUMINAMATH_CALUDE_debugging_time_l3559_355921

theorem debugging_time (total_hours : ℝ) (flow_chart_fraction : ℝ) (coding_fraction : ℝ)
  (h1 : total_hours = 48)
  (h2 : flow_chart_fraction = 1/4)
  (h3 : coding_fraction = 3/8)
  (h4 : flow_chart_fraction + coding_fraction < 1) :
  total_hours * (1 - flow_chart_fraction - coding_fraction) = 18 :=
by sorry

end NUMINAMATH_CALUDE_debugging_time_l3559_355921


namespace NUMINAMATH_CALUDE_danny_bottle_caps_l3559_355973

/-- Proves the number of bottle caps Danny found at the park -/
theorem danny_bottle_caps 
  (thrown_away : ℕ) 
  (current_total : ℕ) 
  (found_more_than_thrown : ℕ) : 
  thrown_away = 35 → 
  current_total = 22 → 
  found_more_than_thrown = 1 → 
  ∃ (previous_total : ℕ) (found : ℕ), 
    found = thrown_away + found_more_than_thrown ∧ 
    current_total = previous_total - thrown_away + found ∧
    found = 36 := by
  sorry

end NUMINAMATH_CALUDE_danny_bottle_caps_l3559_355973


namespace NUMINAMATH_CALUDE_range_of_a_l3559_355930

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x| ≤ 4) → -4 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3559_355930


namespace NUMINAMATH_CALUDE_chocolate_division_l3559_355954

theorem chocolate_division (total : ℚ) (piles : ℕ) (keep_fraction : ℚ) :
  total = 72 / 7 →
  piles = 6 →
  keep_fraction = 1 / 3 →
  (total / piles) * (1 - keep_fraction) = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_division_l3559_355954


namespace NUMINAMATH_CALUDE_max_value_reciprocal_sum_l3559_355993

/-- Given a quadratic polynomial x^2 - tx + q with roots α and β,
    where α + β = α^2 + β^2 = α^3 + β^3 = ... = α^2010 + β^2010,
    the maximum possible value of 1/α^2011 + 1/β^2011 is 2. -/
theorem max_value_reciprocal_sum (t q α β : ℝ) : 
  (∀ k : ℕ, k ≥ 1 ∧ k ≤ 2010 → α^k + β^k = α + β) →
  α^2 - t*α + q = 0 →
  β^2 - t*β + q = 0 →
  α ≠ 0 →
  β ≠ 0 →
  (1/α^2011 + 1/β^2011 : ℝ) ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_reciprocal_sum_l3559_355993


namespace NUMINAMATH_CALUDE_book_reading_ratio_l3559_355986

/-- The number of books William read last month -/
def william_last_month : ℕ := 6

/-- The number of books Brad read last month -/
def brad_last_month : ℕ := 3 * william_last_month

/-- The number of books Brad read this month -/
def brad_this_month : ℕ := 8

/-- The difference in total books read between William and Brad over two months -/
def difference_total : ℕ := 4

/-- The number of books William read this month -/
def william_this_month : ℕ := william_last_month + brad_last_month + brad_this_month + difference_total - (brad_last_month + brad_this_month)

theorem book_reading_ratio : 
  william_this_month / brad_this_month = 3 ∧ william_this_month % brad_this_month = 0 := by
  sorry

end NUMINAMATH_CALUDE_book_reading_ratio_l3559_355986


namespace NUMINAMATH_CALUDE_average_weight_abc_l3559_355928

theorem average_weight_abc (a b c : ℝ) : 
  (a + b) / 2 = 40 →
  (b + c) / 2 = 45 →
  b = 35 →
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_abc_l3559_355928


namespace NUMINAMATH_CALUDE_clock_strike_time_l3559_355939

/-- If a clock strikes 12 in 33 seconds, it will strike 6 in 15 seconds -/
theorem clock_strike_time (strike_12_time : ℕ) (strike_6_time : ℕ) : 
  strike_12_time = 33 → strike_6_time = 15 := by
  sorry

#check clock_strike_time

end NUMINAMATH_CALUDE_clock_strike_time_l3559_355939


namespace NUMINAMATH_CALUDE_parallelepipeds_in_4x4x4_cube_l3559_355989

/-- The number of distinct rectangular parallelepipeds in a cube of size n --/
def count_parallelepipeds (n : ℕ) : ℕ :=
  (n + 1).choose 2 ^ 3

/-- Theorem stating that in a 4 × 4 × 4 cube, there are 1000 distinct rectangular parallelepipeds --/
theorem parallelepipeds_in_4x4x4_cube :
  count_parallelepipeds 4 = 1000 := by sorry

end NUMINAMATH_CALUDE_parallelepipeds_in_4x4x4_cube_l3559_355989


namespace NUMINAMATH_CALUDE_medical_staff_distribution_l3559_355945

/-- Represents the number of medical staff members -/
def num_staff : ℕ := 4

/-- Represents the number of communities -/
def num_communities : ℕ := 3

/-- Represents the constraint that A and B must be together -/
def a_and_b_together : Prop := True

/-- Represents the constraint that each community must have at least one person -/
def each_community_nonempty : Prop := True

/-- The number of ways to distribute the medical staff among communities -/
def distribution_count : ℕ := 6

/-- Theorem stating that the number of ways to distribute the medical staff
    among communities, given the constraints, is equal to 6 -/
theorem medical_staff_distribution :
  (num_staff = 4) →
  (num_communities = 3) →
  a_and_b_together →
  each_community_nonempty →
  distribution_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_medical_staff_distribution_l3559_355945


namespace NUMINAMATH_CALUDE_workers_in_first_group_l3559_355953

/-- The number of workers in the first group -/
def W : ℕ := 360

/-- The time taken by the first group to build the wall -/
def T1 : ℕ := 48

/-- The number of workers in the second group -/
def T2 : ℕ := 24

/-- The time taken by the second group to build the wall -/
def W2 : ℕ := 30

/-- Theorem stating that W is the correct number of workers in the first group -/
theorem workers_in_first_group :
  W * T1 = T2 * W2 := by sorry

end NUMINAMATH_CALUDE_workers_in_first_group_l3559_355953


namespace NUMINAMATH_CALUDE_barbara_wins_2023_barbara_wins_2024_l3559_355911

/-- Represents the players in the coin removal game -/
inductive Player
| Barbara
| Jenna

/-- Represents the state of the game -/
structure GameState where
  coins : ℕ
  currentPlayer : Player

/-- Defines a valid move for a player -/
def validMove (player : Player) (coins : ℕ) : Set ℕ :=
  match player with
  | Player.Barbara => {2, 4, 5}
  | Player.Jenna => {1, 3, 5}

/-- Determines if a game state is winning for the current player -/
def isWinningState : GameState → Prop :=
  sorry

/-- Theorem stating that Barbara wins with 2023 coins -/
theorem barbara_wins_2023 :
  isWinningState ⟨2023, Player.Barbara⟩ :=
  sorry

/-- Theorem stating that Barbara wins with 2024 coins -/
theorem barbara_wins_2024 :
  isWinningState ⟨2024, Player.Barbara⟩ :=
  sorry

end NUMINAMATH_CALUDE_barbara_wins_2023_barbara_wins_2024_l3559_355911


namespace NUMINAMATH_CALUDE_class_average_weight_l3559_355925

theorem class_average_weight (students_A students_B : ℕ) (avg_weight_A avg_weight_B : ℝ) :
  students_A = 36 →
  students_B = 24 →
  avg_weight_A = 30 →
  avg_weight_B = 30 →
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l3559_355925


namespace NUMINAMATH_CALUDE_dot_only_count_l3559_355963

/-- Represents the number of letters in an alphabet with specific characteristics. -/
structure Alphabet where
  total : ℕ
  dot_and_line : ℕ
  line_only : ℕ
  dot_only : ℕ

/-- Theorem stating that in an alphabet with given properties, 
    the number of letters containing only a dot is 3. -/
theorem dot_only_count (α : Alphabet) 
  (h_total : α.total = 40)
  (h_dot_and_line : α.dot_and_line = 13)
  (h_line_only : α.line_only = 24)
  (h_all_covered : α.total = α.dot_and_line + α.line_only + α.dot_only) :
  α.dot_only = 3 := by
  sorry

end NUMINAMATH_CALUDE_dot_only_count_l3559_355963


namespace NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_l3559_355944

theorem smallest_positive_integer_3003m_55555n :
  ∃ (k : ℕ), k > 0 ∧ (∃ (m n : ℤ), k = 3003 * m + 55555 * n) ∧
  ∀ (j : ℕ), j > 0 → (∃ (x y : ℤ), j = 3003 * x + 55555 * y) → k ≤ j :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_3003m_55555n_l3559_355944


namespace NUMINAMATH_CALUDE_laundry_ratio_l3559_355922

def wednesday_loads : ℕ := 6
def thursday_loads : ℕ := 2 * wednesday_loads
def saturday_loads : ℕ := wednesday_loads / 3
def total_loads : ℕ := 26

def friday_loads : ℕ := total_loads - (wednesday_loads + thursday_loads + saturday_loads)

theorem laundry_ratio :
  (friday_loads : ℚ) / thursday_loads = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_laundry_ratio_l3559_355922


namespace NUMINAMATH_CALUDE_gcd_g_y_equals_one_l3559_355942

theorem gcd_g_y_equals_one (y : ℤ) (h : ∃ k : ℤ, y = 34567 * k) :
  let g : ℤ → ℤ := λ y => (3*y+4)*(8*y+3)*(14*y+5)*(y+14)
  Int.gcd (g y) y = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_g_y_equals_one_l3559_355942


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3559_355991

theorem repeating_decimal_to_fraction :
  ∀ (x : ℚ), (∃ (n : ℕ), x = 0.56 + 0.0056 * (1 - (1/100)^n) / (1 - 1/100)) →
  x = 56/99 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3559_355991


namespace NUMINAMATH_CALUDE_cubic_function_sign_properties_l3559_355926

/-- Given a cubic function with three real roots, prove specific sign properties -/
theorem cubic_function_sign_properties 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = x^3 - 6*x^2 + 9*x - a*b*c)
  (h2 : a < b ∧ b < c)
  (h3 : f a = 0 ∧ f b = 0 ∧ f c = 0) :
  f 0 * f 1 < 0 ∧ f 0 * f 3 > 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_function_sign_properties_l3559_355926


namespace NUMINAMATH_CALUDE_student_allowance_proof_l3559_355906

def weekly_allowance : ℝ := 3.00

theorem student_allowance_proof :
  let arcade_spend := (2 : ℝ) / 5 * weekly_allowance
  let remaining_after_arcade := weekly_allowance - arcade_spend
  let toy_store_spend := (1 : ℝ) / 3 * remaining_after_arcade
  let remaining_after_toy_store := remaining_after_arcade - toy_store_spend
  remaining_after_toy_store = 1.20
  →
  weekly_allowance = 3.00 := by
  sorry

end NUMINAMATH_CALUDE_student_allowance_proof_l3559_355906


namespace NUMINAMATH_CALUDE_sector_radius_range_l3559_355981

theorem sector_radius_range (a : ℝ) (m : ℝ) (h1 : a > 0) (h2 : 0 < m) (h3 : m < 360) :
  ∃ R : ℝ, a / (2 * (1 + π)) < R ∧ R < a / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_radius_range_l3559_355981


namespace NUMINAMATH_CALUDE_cube_sum_magnitude_l3559_355997

theorem cube_sum_magnitude (w z : ℂ) 
  (h1 : Complex.abs (w + z) = 2) 
  (h2 : Complex.abs (w^2 + z^2) = 8) : 
  Complex.abs (w^3 + z^3) = 20 := by sorry

end NUMINAMATH_CALUDE_cube_sum_magnitude_l3559_355997


namespace NUMINAMATH_CALUDE_number_categorization_l3559_355969

def numbers : List ℚ := [7, -3.14, -5, 1/8, 0, -7/4, -4/5]

def is_positive_rational (x : ℚ) : Prop := x > 0

def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x ≠ ↑(⌊x⌋)

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = ↑n

theorem number_categorization :
  (∀ x ∈ numbers, is_positive_rational x ↔ x ∈ [7, 1/8]) ∧
  (∀ x ∈ numbers, is_negative_fraction x ↔ x ∈ [-3.14, -7/4, -4/5]) ∧
  (∀ x ∈ numbers, is_integer x ↔ x ∈ [7, -5, 0]) :=
sorry

end NUMINAMATH_CALUDE_number_categorization_l3559_355969


namespace NUMINAMATH_CALUDE_johns_commute_distance_l3559_355927

theorem johns_commute_distance :
  let usual_time : ℝ := 200
  let fog_day_time : ℝ := 320
  let speed_reduction : ℝ := 15
  let distance : ℝ := 92

  let usual_speed : ℝ := distance / usual_time
  let fog_speed : ℝ := usual_speed - speed_reduction / 60

  (distance / 2) / usual_speed + (distance / 2) / fog_speed = fog_day_time :=
by sorry

end NUMINAMATH_CALUDE_johns_commute_distance_l3559_355927


namespace NUMINAMATH_CALUDE_diagonal_length_in_special_quadrilateral_l3559_355964

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a quadrilateral ABCD with diagonals intersecting at E -/
structure Quadrilateral :=
  (A B C D E : Point)

/-- The length of a line segment between two points -/
def distance (p q : Point) : ℝ := sorry

/-- The area of a triangle given three points -/
def triangleArea (p q r : Point) : ℝ := sorry

/-- Main theorem -/
theorem diagonal_length_in_special_quadrilateral 
  (ABCD : Quadrilateral) 
  (h1 : distance ABCD.A ABCD.B = 10)
  (h2 : distance ABCD.C ABCD.D = 15)
  (h3 : distance ABCD.A ABCD.C = 18)
  (h4 : triangleArea ABCD.A ABCD.E ABCD.D = triangleArea ABCD.B ABCD.E ABCD.C) :
  distance ABCD.A ABCD.E = 7.2 := by sorry

end NUMINAMATH_CALUDE_diagonal_length_in_special_quadrilateral_l3559_355964


namespace NUMINAMATH_CALUDE_log_sum_equals_three_main_theorem_l3559_355960

theorem log_sum_equals_three : Real.log 8 + 3 * Real.log 5 = 3 * Real.log 10 := by
  sorry

theorem main_theorem : Real.log 8 + 3 * Real.log 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equals_three_main_theorem_l3559_355960


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3559_355929

theorem trigonometric_simplification (θ : Real) : 
  (Real.sin (2 * Real.pi - θ) * Real.cos (Real.pi + θ) * Real.cos (Real.pi / 2 + θ) * Real.cos (11 * Real.pi / 2 - θ)) / 
  (Real.cos (Real.pi - θ) * Real.sin (3 * Real.pi - θ) * Real.sin (-Real.pi - θ) * Real.sin (9 * Real.pi / 2 + θ)) = 
  -Real.tan θ := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3559_355929


namespace NUMINAMATH_CALUDE_eight_digit_even_integers_count_l3559_355923

/-- The set of even digits -/
def EvenDigits : Finset Nat := {0, 2, 4, 6, 8}

/-- The set of non-zero even digits -/
def NonZeroEvenDigits : Finset Nat := {2, 4, 6, 8}

/-- The number of 8-digit positive integers with all even digits -/
def EightDigitEvenIntegers : Nat :=
  Finset.card NonZeroEvenDigits * (Finset.card EvenDigits ^ 7)

theorem eight_digit_even_integers_count :
  EightDigitEvenIntegers = 312500 := by
sorry

end NUMINAMATH_CALUDE_eight_digit_even_integers_count_l3559_355923


namespace NUMINAMATH_CALUDE_point_on_curve_limit_at_one_l3559_355935

/-- The curve y = x² + 1 -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- The point (1, 2) lies on the curve -/
theorem point_on_curve : f 1 = 2 := by sorry

/-- The limit of Δy/Δx as Δx approaches 0 at x = 1 is 2 -/
theorem limit_at_one : 
  ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 
    0 < |h| → |h| < δ → |(f (1 + h) - f 1) / h - 2| < ε := by sorry

end NUMINAMATH_CALUDE_point_on_curve_limit_at_one_l3559_355935


namespace NUMINAMATH_CALUDE_compound_interest_problem_l3559_355903

theorem compound_interest_problem :
  ∃ (P r : ℝ), P > 0 ∧ r > 0 ∧ 
  P * (1 + r)^2 = 8000 ∧
  P * (1 + r)^3 = 9261 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l3559_355903


namespace NUMINAMATH_CALUDE_kittens_problem_l3559_355970

/-- The number of kittens Tim gave to Jessica -/
def kittens_to_jessica (initial : ℕ) (to_sara : ℕ) (remaining : ℕ) : ℕ :=
  initial - to_sara - remaining

theorem kittens_problem (initial : ℕ) (to_sara : ℕ) (remaining : ℕ) 
  (h1 : initial = 18) 
  (h2 : to_sara = 6) 
  (h3 : remaining = 9) : 
  kittens_to_jessica initial to_sara remaining = 3 := by
  sorry

end NUMINAMATH_CALUDE_kittens_problem_l3559_355970


namespace NUMINAMATH_CALUDE_plan_d_cost_effective_l3559_355946

/-- The cost in cents for Plan C given the number of minutes used -/
def plan_c_cost (minutes : ℕ) : ℕ := 15 * minutes

/-- The cost in cents for Plan D given the number of minutes used -/
def plan_d_cost (minutes : ℕ) : ℕ := 2500 + 12 * minutes

/-- The minimum number of whole minutes for Plan D to be cost-effective -/
def min_minutes_for_plan_d : ℕ := 834

theorem plan_d_cost_effective :
  (∀ m : ℕ, m ≥ min_minutes_for_plan_d → plan_d_cost m < plan_c_cost m) ∧
  (∀ m : ℕ, m < min_minutes_for_plan_d → plan_d_cost m ≥ plan_c_cost m) :=
sorry

end NUMINAMATH_CALUDE_plan_d_cost_effective_l3559_355946


namespace NUMINAMATH_CALUDE_right_triangle_from_sine_condition_l3559_355916

theorem right_triangle_from_sine_condition (A B C : Real) (h1 : 0 < A) (h2 : A < π/2) 
  (h3 : 0 < B) (h4 : B < π/2) (h5 : A + B + C = π) 
  (h6 : Real.sin A ^ 2 + Real.sin B ^ 2 = Real.sin (A + B)) : 
  C = π/2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_from_sine_condition_l3559_355916


namespace NUMINAMATH_CALUDE_regular_hexagon_area_l3559_355938

/-- The area of a regular hexagon with vertices A at (0,0) and C at (8,2) is 34√3 -/
theorem regular_hexagon_area : 
  let A : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (8, 2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * AC^2
  let hexagon_area : ℝ := 2 * triangle_area
  hexagon_area = 34 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_regular_hexagon_area_l3559_355938


namespace NUMINAMATH_CALUDE_ganzhi_2019_l3559_355924

-- Define the Heavenly Stems
inductive HeavenlyStem
| Jia | Yi | Bing | Ding | Wu | Ji | Geng | Xin | Ren | Gui

-- Define the Earthly Branches
inductive EarthlyBranch
| Zi | Chou | Yin | Mao | Chen | Si | Wu | Wei | Shen | You | Xu | Hai

-- Define the Ganzhi combination
structure Ganzhi where
  stem : HeavenlyStem
  branch : EarthlyBranch

-- Define a function to get the next Heavenly Stem
def nextStem (s : HeavenlyStem) : HeavenlyStem :=
  match s with
  | HeavenlyStem.Jia => HeavenlyStem.Yi
  | HeavenlyStem.Yi => HeavenlyStem.Bing
  | HeavenlyStem.Bing => HeavenlyStem.Ding
  | HeavenlyStem.Ding => HeavenlyStem.Wu
  | HeavenlyStem.Wu => HeavenlyStem.Ji
  | HeavenlyStem.Ji => HeavenlyStem.Geng
  | HeavenlyStem.Geng => HeavenlyStem.Xin
  | HeavenlyStem.Xin => HeavenlyStem.Ren
  | HeavenlyStem.Ren => HeavenlyStem.Gui
  | HeavenlyStem.Gui => HeavenlyStem.Jia

-- Define a function to get the next Earthly Branch
def nextBranch (b : EarthlyBranch) : EarthlyBranch :=
  match b with
  | EarthlyBranch.Zi => EarthlyBranch.Chou
  | EarthlyBranch.Chou => EarthlyBranch.Yin
  | EarthlyBranch.Yin => EarthlyBranch.Mao
  | EarthlyBranch.Mao => EarthlyBranch.Chen
  | EarthlyBranch.Chen => EarthlyBranch.Si
  | EarthlyBranch.Si => EarthlyBranch.Wu
  | EarthlyBranch.Wu => EarthlyBranch.Wei
  | EarthlyBranch.Wei => EarthlyBranch.Shen
  | EarthlyBranch.Shen => EarthlyBranch.You
  | EarthlyBranch.You => EarthlyBranch.Xu
  | EarthlyBranch.Xu => EarthlyBranch.Hai
  | EarthlyBranch.Hai => EarthlyBranch.Zi

-- Define a function to get the next Ganzhi combination
def nextGanzhi (g : Ganzhi) : Ganzhi :=
  { stem := nextStem g.stem, branch := nextBranch g.branch }

-- Define a function to advance Ganzhi by n years
def advanceGanzhi (g : Ganzhi) (n : Nat) : Ganzhi :=
  match n with
  | 0 => g
  | n + 1 => advanceGanzhi (nextGanzhi g) n

-- Theorem statement
theorem ganzhi_2019 (ganzhi_2010 : Ganzhi)
  (h2010 : ganzhi_2010 = { stem := HeavenlyStem.Geng, branch := EarthlyBranch.Yin }) :
  advanceGanzhi ganzhi_2010 9 = { stem := HeavenlyStem.Ji, branch := EarthlyBranch.You } :=
by sorry


end NUMINAMATH_CALUDE_ganzhi_2019_l3559_355924


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3559_355983

theorem inequality_system_solution (x : ℝ) :
  x - 4 ≤ 0 ∧ 2 * (x + 1) < 3 * x → 2 < x ∧ x ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3559_355983


namespace NUMINAMATH_CALUDE_debt_payment_average_l3559_355918

theorem debt_payment_average : 
  let total_payments : ℕ := 52
  let first_payment_count : ℕ := 25
  let first_payment_amount : ℚ := 500
  let additional_amount : ℚ := 100
  let second_payment_count : ℕ := total_payments - first_payment_count
  let second_payment_amount : ℚ := first_payment_amount + additional_amount
  let total_amount : ℚ := first_payment_count * first_payment_amount + 
                          second_payment_count * second_payment_amount
  let average_payment : ℚ := total_amount / total_payments
  average_payment = 551.92 := by
sorry

end NUMINAMATH_CALUDE_debt_payment_average_l3559_355918


namespace NUMINAMATH_CALUDE_part_one_part_two_l3559_355976

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x + a - 1 = 0}
def C (m : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*(m+1)*x + m^2 - 5 = 0}

-- Theorem for part (1)
theorem part_one (a : ℝ) : A ∪ B a = A → a = 2 ∨ a = 3 := by sorry

-- Theorem for part (2)
theorem part_two (m : ℝ) : A ∩ C m = C m → m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3559_355976


namespace NUMINAMATH_CALUDE_clock_angle_at_2pm_l3559_355914

/-- The number of hours on a standard clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a complete rotation -/
def full_rotation : ℕ := 360

/-- The number of degrees the hour hand moves per hour -/
def hour_hand_degrees_per_hour : ℚ := full_rotation / clock_hours

/-- The position of the hour hand at 2:00 -/
def hour_hand_position_at_2 : ℚ := 2 * hour_hand_degrees_per_hour

/-- The position of the minute hand at 2:00 -/
def minute_hand_position_at_2 : ℚ := 0

/-- The smaller angle between the hour hand and minute hand at 2:00 -/
def smaller_angle_at_2 : ℚ := hour_hand_position_at_2 - minute_hand_position_at_2

theorem clock_angle_at_2pm :
  smaller_angle_at_2 = 60 := by sorry

end NUMINAMATH_CALUDE_clock_angle_at_2pm_l3559_355914


namespace NUMINAMATH_CALUDE_quadratic_root_l3559_355949

theorem quadratic_root (k : ℚ) : 
  let x : ℝ := (-25 - Real.sqrt 369) / 12
  k = 32 / 3 ↔ 6 * x^2 + 25 * x + k = 0 := by sorry

end NUMINAMATH_CALUDE_quadratic_root_l3559_355949


namespace NUMINAMATH_CALUDE_divisibility_equivalence_l3559_355982

theorem divisibility_equivalence (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a + b + c ∣ a^3 * b + b^3 * c + c^3 * a) ↔ (a + b + c ∣ a * b^3 + b * c^3 + c * a^3) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_equivalence_l3559_355982


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3559_355900

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3559_355900


namespace NUMINAMATH_CALUDE_power_of_power_l3559_355952

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3559_355952


namespace NUMINAMATH_CALUDE_interest_calculation_l3559_355994

/-- Calculates the total interest earned from two investments -/
def total_interest (total_investment : ℚ) (rate1 rate2 : ℚ) (amount1 : ℚ) : ℚ :=
  let amount2 := total_investment - amount1
  amount1 * rate1 + amount2 * rate2

/-- Proves that the total interest earned is $490 given the specified conditions -/
theorem interest_calculation :
  let total_investment : ℚ := 8000
  let rate1 : ℚ := 8 / 100
  let rate2 : ℚ := 5 / 100
  let amount1 : ℚ := 3000
  total_interest total_investment rate1 rate2 amount1 = 490 := by
  sorry

end NUMINAMATH_CALUDE_interest_calculation_l3559_355994


namespace NUMINAMATH_CALUDE_work_completion_time_l3559_355965

/-- Given that:
    - A can do the work in 3 days
    - A and B together can do the work in 2 days
    Prove that B can do the work alone in 6 days -/
theorem work_completion_time (a_time b_time ab_time : ℝ) 
    (ha : a_time = 3)
    (hab : ab_time = 2) :
    b_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3559_355965


namespace NUMINAMATH_CALUDE_two_positive_solutions_l3559_355932

theorem two_positive_solutions (a : ℝ) :
  (∃! x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ 
   (|2*x₁ - 1| - a = 0) ∧ (|2*x₂ - 1| - a = 0)) ↔ 
  (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_two_positive_solutions_l3559_355932


namespace NUMINAMATH_CALUDE_cars_sold_first_day_l3559_355950

theorem cars_sold_first_day (cars_second_day cars_third_day total_cars : ℕ)
  (h1 : cars_second_day = 16)
  (h2 : cars_third_day = 27)
  (h3 : total_cars = 57)
  (h4 : total_cars = cars_second_day + cars_third_day + (total_cars - cars_second_day - cars_third_day)) :
  total_cars - cars_second_day - cars_third_day = 14 := by
  sorry

end NUMINAMATH_CALUDE_cars_sold_first_day_l3559_355950


namespace NUMINAMATH_CALUDE_inequality_proof_l3559_355955

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a * b^5 + b * c^5 + c * a^5 ≥ a * b * c * (a^2 * b + b^2 * c + c^2 * a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3559_355955


namespace NUMINAMATH_CALUDE_water_bottle_refills_l3559_355951

/-- Calculates the number of times a water bottle needs to be filled in a week -/
theorem water_bottle_refills (daily_intake : ℕ) (bottle_capacity : ℕ) : 
  daily_intake = 72 → bottle_capacity = 84 → (daily_intake * 7) / bottle_capacity = 6 := by
  sorry

#check water_bottle_refills

end NUMINAMATH_CALUDE_water_bottle_refills_l3559_355951


namespace NUMINAMATH_CALUDE_phi_value_l3559_355980

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (Real.sqrt 3 * x + φ)

noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := f x φ + (deriv (f · φ)) x

theorem phi_value (φ : ℝ) 
  (h1 : -π < φ ∧ φ < 0) 
  (h2 : ∀ x, g x φ = g (-x) φ) : 
  φ = -π/3 := by
sorry

end NUMINAMATH_CALUDE_phi_value_l3559_355980


namespace NUMINAMATH_CALUDE_investment_problem_l3559_355967

theorem investment_problem (x y : ℝ) : 
  x + y = 24000 → 
  x * 0.045 + y * 0.06 = 24000 * 0.05 → 
  x = 16000 ∧ y = 8000 :=
by sorry

end NUMINAMATH_CALUDE_investment_problem_l3559_355967


namespace NUMINAMATH_CALUDE_water_fraction_after_four_replacements_l3559_355937

/-- The fraction of water remaining in a radiator after multiple replacements with antifreeze -/
def water_fraction (total_capacity : ℚ) (replacement_volume : ℚ) (num_replacements : ℕ) : ℚ :=
  (1 - replacement_volume / total_capacity) ^ num_replacements

/-- The fraction of water remaining in a 20-quart radiator after 4 replacements of 5 quarts each -/
theorem water_fraction_after_four_replacements :
  water_fraction 20 5 4 = 81 / 256 := by
  sorry

#eval water_fraction 20 5 4

end NUMINAMATH_CALUDE_water_fraction_after_four_replacements_l3559_355937


namespace NUMINAMATH_CALUDE_new_person_age_l3559_355917

/-- Given a group of 10 persons where replacing a 45-year-old person with a new person
    decreases the average age by 3 years, the age of the new person is 15 years. -/
theorem new_person_age (initial_avg : ℝ) : 
  (10 * initial_avg - 45 + 15) / 10 = initial_avg - 3 := by
  sorry

#check new_person_age

end NUMINAMATH_CALUDE_new_person_age_l3559_355917


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3559_355977

/-- The line kx - y + 1 = 3k passes through the point (3, 1) for all real k -/
theorem line_passes_through_fixed_point :
  ∀ (k : ℝ), (k * 3 : ℝ) - 1 + 1 = 3 * k := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3559_355977


namespace NUMINAMATH_CALUDE_study_time_for_target_average_l3559_355933

/-- Calculates the number of minutes needed to study on the 12th day to achieve a given average -/
def minutes_to_study_on_last_day (days_30min : ℕ) (days_45min : ℕ) (target_average : ℕ) : ℕ :=
  let total_days := days_30min + days_45min + 1
  let total_minutes_needed := total_days * target_average
  let minutes_already_studied := days_30min * 30 + days_45min * 45
  total_minutes_needed - minutes_already_studied

/-- Theorem stating that given the specific study pattern, 90 minutes are needed on the 12th day -/
theorem study_time_for_target_average :
  minutes_to_study_on_last_day 7 4 40 = 90 := by
  sorry

end NUMINAMATH_CALUDE_study_time_for_target_average_l3559_355933


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3559_355947

theorem arithmetic_geometric_mean_inequality {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  (a + b) / 2 ≥ Real.sqrt (a * b) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3559_355947


namespace NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l3559_355999

theorem sum_of_two_squares_equivalence (n : ℕ) (hn : n > 0) :
  (∃ a b : ℤ, n = a^2 + b^2) ↔ (∃ A B : ℤ, 2 * n = A^2 + B^2) :=
sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l3559_355999


namespace NUMINAMATH_CALUDE_special_function_value_l3559_355904

/-- A function satisfying the given conditions -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x y, x > y ∧ y > 0 → f (x - y) = Real.sqrt (f (x * y) + 2))

/-- The main theorem stating that f(2010) = 2 for any function satisfying the conditions -/
theorem special_function_value (f : ℝ → ℝ) (h : special_function f) : f 2010 = 2 := by
  sorry


end NUMINAMATH_CALUDE_special_function_value_l3559_355904


namespace NUMINAMATH_CALUDE_gcd_sequence_limit_l3559_355996

theorem gcd_sequence_limit (n : ℕ) : 
  ∃ N : ℕ, ∀ m : ℕ, m ≥ N → 
    Nat.gcd (100 + 2 * m^2) (100 + 2 * (m + 1)^2) = 1 := by
  sorry

#check gcd_sequence_limit

end NUMINAMATH_CALUDE_gcd_sequence_limit_l3559_355996


namespace NUMINAMATH_CALUDE_american_flag_problem_l3559_355912

theorem american_flag_problem (total_stripes : ℕ) (total_red_stripes : ℕ) : 
  total_stripes = 13 →
  total_red_stripes = 70 →
  (total_stripes - 1) / 2 + 1 = 7 →
  total_red_stripes / ((total_stripes - 1) / 2 + 1) = 10 := by
sorry

end NUMINAMATH_CALUDE_american_flag_problem_l3559_355912
