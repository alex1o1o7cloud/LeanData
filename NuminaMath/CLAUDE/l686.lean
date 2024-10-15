import Mathlib

namespace NUMINAMATH_CALUDE_percentage_problem_l686_68601

theorem percentage_problem : ∃ P : ℝ, P * 600 = 50 / 100 * 900 ∧ P = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l686_68601


namespace NUMINAMATH_CALUDE_hyperbola_ratio_l686_68638

/-- Given a point M(x, 5/x) in the first quadrant on the hyperbola y = 5/x,
    with A(x, 0), B(0, 5/x), C(x, 3/x), and D(3/y, y) where y = 5/x,
    prove that the ratio CD:AB = 2:5 -/
theorem hyperbola_ratio (x : ℝ) (hx : x > 0) : 
  let y := 5 / x
  let m := (x, y)
  let a := (x, 0)
  let b := (0, y)
  let c := (x, 3 / x)
  let d := (3 / y, y)
  let cd := Real.sqrt ((x - 3 / y)^2 + (3 / x - y)^2)
  let ab := Real.sqrt ((x - 0)^2 + (0 - y)^2)
  cd / ab = 2 / 5 := by
sorry


end NUMINAMATH_CALUDE_hyperbola_ratio_l686_68638


namespace NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l686_68669

theorem smallest_distance_between_complex_numbers
  (z w : ℂ)
  (hz : Complex.abs (z + 2 + 4 * Complex.I) = 2)
  (hw : Complex.abs (w - 6 - 7 * Complex.I) = 4) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 185 - 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' + 2 + 4 * Complex.I) = 2 →
      Complex.abs (w' - 6 - 7 * Complex.I) = 4 →
        Complex.abs (z' - w') ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_complex_numbers_l686_68669


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l686_68611

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (n_Al n_Cl n_O : ℕ) (w_Al w_Cl w_O : ℝ) : ℝ :=
  n_Al * w_Al + n_Cl * w_Cl + n_O * w_O

/-- The molecular weight of a compound with 2 Al, 6 Cl, and 3 O atoms is 314.66 g/mol -/
theorem compound_molecular_weight :
  molecular_weight 2 6 3 26.98 35.45 16.00 = 314.66 := by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l686_68611


namespace NUMINAMATH_CALUDE_no_representation_2023_l686_68600

theorem no_representation_2023 : ¬∃ (a b c : ℕ), 
  (a + b + c = 2023) ∧ 
  (∃ k : ℕ, a = k * (b + c)) ∧ 
  (∃ m : ℕ, b + c = m * (b - c + 1)) := by
sorry

end NUMINAMATH_CALUDE_no_representation_2023_l686_68600


namespace NUMINAMATH_CALUDE_problem_solution_l686_68629

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2) :
  (∀ a : ℝ, a < 1/2 → 1/x + 1/y ≥ |a + 2| - |a - 1|) ∧
  x^2 + 2*y^2 ≥ 8/3 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l686_68629


namespace NUMINAMATH_CALUDE_circle_properties_l686_68641

/-- Given a circle with polar equation ρ²-4√2ρcos(θ-π/4)+6=0, prove its properties -/
theorem circle_properties (ρ θ : ℝ) :
  ρ^2 - 4 * Real.sqrt 2 * ρ * Real.cos (θ - π/4) + 6 = 0 →
  ∃ (x y : ℝ),
    -- Standard equation
    x^2 + y^2 - 4*x - 4*y + 6 = 0 ∧
    -- Parametric equations
    x = 2 + Real.sqrt 2 * Real.cos θ ∧
    y = 2 + Real.sqrt 2 * Real.sin θ ∧
    -- Maximum and minimum values of x⋅y
    (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 → x'*y' ≤ 9) ∧
    (∀ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 → x'*y' ≥ 1) ∧
    (∃ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 ∧ x'*y' = 9) ∧
    (∃ (x' y' : ℝ), x'^2 + y'^2 - 4*x' - 4*y' + 6 = 0 ∧ x'*y' = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_properties_l686_68641


namespace NUMINAMATH_CALUDE_inheritance_calculation_l686_68663

theorem inheritance_calculation (x : ℝ) : 
  x > 0 →
  (0.25 * x + 0.15 * (x - 0.25 * x) = 18000) →
  x = 50000 := by
sorry

end NUMINAMATH_CALUDE_inheritance_calculation_l686_68663


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l686_68606

theorem point_in_fourth_quadrant (P : ℝ × ℝ) :
  P.1 = Real.tan (2011 * π / 180) →
  P.2 = Real.cos (2011 * π / 180) →
  Real.tan (2011 * π / 180) > 0 →
  Real.cos (2011 * π / 180) < 0 →
  P.1 > 0 ∧ P.2 < 0 := by
sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l686_68606


namespace NUMINAMATH_CALUDE_additional_sugar_needed_l686_68648

/-- The amount of additional sugar needed for a cake -/
theorem additional_sugar_needed (total_required sugar_available : ℕ) : 
  total_required = 450 → sugar_available = 287 → total_required - sugar_available = 163 := by
  sorry

end NUMINAMATH_CALUDE_additional_sugar_needed_l686_68648


namespace NUMINAMATH_CALUDE_batsman_average_theorem_l686_68697

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an additional inning -/
def newAverage (stats : BatsmanStats) (newInningScore : ℕ) : ℚ :=
  (stats.totalRuns + newInningScore) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 5 after scoring 110 in the 11th inning, 
    then the new average is 60 -/
theorem batsman_average_theorem (stats : BatsmanStats) 
  (h1 : stats.innings = 10)
  (h2 : newAverage stats 110 = stats.average + 5) :
  newAverage stats 110 = 60 := by
  sorry

#check batsman_average_theorem

end NUMINAMATH_CALUDE_batsman_average_theorem_l686_68697


namespace NUMINAMATH_CALUDE_expression_value_l686_68685

theorem expression_value
  (a b x y : ℝ)
  (m : ℤ)
  (h1 : a + b = 0)
  (h2 : x * y = 1)
  (h3 : m = -1) :
  2023 * (a + b) + 3 * |m| - 2 * x * y = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l686_68685


namespace NUMINAMATH_CALUDE_pizza_portion_eaten_l686_68604

theorem pizza_portion_eaten (total_slices : ℕ) (slices_left : ℕ) :
  total_slices = 16 → slices_left = 4 →
  (total_slices - slices_left : ℚ) / total_slices = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_portion_eaten_l686_68604


namespace NUMINAMATH_CALUDE_function_composition_equality_l686_68683

theorem function_composition_equality (a : ℝ) (h_pos : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x + 1 / Real.sqrt 2
  f (f (1 / Real.sqrt 2)) = f 0 → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l686_68683


namespace NUMINAMATH_CALUDE_probability_both_presidents_selected_l686_68612

def club_sizes : List Nat := [6, 8, 9, 10]

def probability_both_presidents (n : Nat) : Rat :=
  (Nat.choose (n - 2) 2 : Rat) / (Nat.choose n 4 : Rat)

theorem probability_both_presidents_selected :
  (1 / 4 : Rat) * (club_sizes.map probability_both_presidents).sum = 119 / 700 := by
  sorry

end NUMINAMATH_CALUDE_probability_both_presidents_selected_l686_68612


namespace NUMINAMATH_CALUDE_reunion_handshakes_l686_68605

/-- Calculates the number of handshakes in a group --/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the reunion scenario --/
structure Reunion :=
  (total_boys : ℕ)
  (left_handed_boys : ℕ)
  (h_left_handed_le_total : left_handed_boys ≤ total_boys)

/-- Calculates the total number of handshakes at the reunion --/
def total_handshakes (r : Reunion) : ℕ :=
  handshakes r.left_handed_boys + handshakes (r.total_boys - r.left_handed_boys)

/-- Theorem stating that the total number of handshakes is 34 for the given scenario --/
theorem reunion_handshakes :
  ∀ (r : Reunion), r.total_boys = 12 → r.left_handed_boys = 4 → total_handshakes r = 34 :=
by
  sorry


end NUMINAMATH_CALUDE_reunion_handshakes_l686_68605


namespace NUMINAMATH_CALUDE_happy_island_parrots_l686_68691

theorem happy_island_parrots (total_birds : ℕ) (yellow_fraction : ℚ) (red_parrots : ℕ) :
  total_birds = 120 →
  yellow_fraction = 2/3 →
  red_parrots = total_birds - (yellow_fraction * total_birds).floor →
  red_parrots = 40 := by
sorry

end NUMINAMATH_CALUDE_happy_island_parrots_l686_68691


namespace NUMINAMATH_CALUDE_smallest_number_proof_l686_68654

def digits : List Nat := [0, 2, 4, 6, 8, 9]

def is_valid_number (n : Nat) : Prop :=
  let digits_used := n.digits 10
  (digits_used.toFinset = digits.toFinset) ∧ 
  (digits_used.length = digits.length) ∧
  (n ≥ 100000)

theorem smallest_number_proof :
  (is_valid_number 204689) ∧ 
  (∀ m : Nat, is_valid_number m → m ≥ 204689) :=
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l686_68654


namespace NUMINAMATH_CALUDE_inequality_range_l686_68603

theorem inequality_range (a : ℝ) : 
  (∀ x y, x ∈ Set.Icc 0 (π/6) → y ∈ Set.Ioi 0 → 
    y/4 - 2*(Real.cos x)^2 ≥ a*(Real.sin x) - 9/y) → 
  a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l686_68603


namespace NUMINAMATH_CALUDE_angela_december_sleep_l686_68628

/-- The number of hours Angela slept every night in December -/
def december_sleep_hours : ℝ := sorry

/-- The number of hours Angela slept every night in January -/
def january_sleep_hours : ℝ := 8.5

/-- The number of days in December -/
def december_days : ℕ := 31

/-- The number of days in January -/
def january_days : ℕ := 31

/-- The additional hours of sleep Angela got in January compared to December -/
def additional_sleep : ℝ := 62

theorem angela_december_sleep :
  december_sleep_hours = 6.5 :=
by
  sorry

end NUMINAMATH_CALUDE_angela_december_sleep_l686_68628


namespace NUMINAMATH_CALUDE_equation_solution_l686_68657

theorem equation_solution : ∃ x : ℝ, x ≠ 1 ∧ (x / (x - 1) + 2 / (1 - x) = 2) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l686_68657


namespace NUMINAMATH_CALUDE_max_intersections_three_lines_one_circle_l686_68608

-- Define a type for geometric figures
inductive Figure
| Line : Figure
| Circle : Figure

-- Define a function to count maximum intersections between two figures
def maxIntersections (f1 f2 : Figure) : ℕ :=
  match f1, f2 with
  | Figure.Line, Figure.Line => 1
  | Figure.Line, Figure.Circle => 2
  | Figure.Circle, Figure.Line => 2
  | Figure.Circle, Figure.Circle => 0

-- Theorem statement
theorem max_intersections_three_lines_one_circle :
  ∃ (l1 l2 l3 : Figure) (c : Figure),
    l1 = Figure.Line ∧ l2 = Figure.Line ∧ l3 = Figure.Line ∧ c = Figure.Circle ∧
    l1 ≠ l2 ∧ l2 ≠ l3 ∧ l1 ≠ l3 ∧
    (maxIntersections l1 l2 + maxIntersections l2 l3 + maxIntersections l1 l3 +
     maxIntersections l1 c + maxIntersections l2 c + maxIntersections l3 c) = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_max_intersections_three_lines_one_circle_l686_68608


namespace NUMINAMATH_CALUDE_sequence_decreasing_two_equal_max_terms_l686_68693

-- Define the sequence aₙ
def a (k : ℝ) (n : ℕ) : ℝ := n * k^n

-- Proposition ②
theorem sequence_decreasing (k : ℝ) (h1 : 0 < k) (h2 : k < 1/2) :
  ∀ n : ℕ, n > 0 → a k (n + 1) < a k n :=
sorry

-- Proposition ④
theorem two_equal_max_terms (k : ℝ) (h : ∃ m : ℕ, m > 0 ∧ k / (1 - k) = m) :
  ∃ n : ℕ, n > 0 ∧ a k n = a k (n + 1) ∧ ∀ m : ℕ, m > 0 → a k m ≤ a k n :=
sorry

end NUMINAMATH_CALUDE_sequence_decreasing_two_equal_max_terms_l686_68693


namespace NUMINAMATH_CALUDE_unequal_outcome_probability_l686_68624

theorem unequal_outcome_probability : 
  let n : ℕ := 12  -- number of grandchildren
  let p : ℝ := 1/2 -- probability of each gender
  let total_outcomes : ℕ := 2^n -- total number of possible gender combinations
  let equal_outcomes : ℕ := n.choose (n/2) -- number of combinations with equal boys and girls
  
  (total_outcomes - equal_outcomes : ℝ) / total_outcomes = 793/1024 := by
  sorry

end NUMINAMATH_CALUDE_unequal_outcome_probability_l686_68624


namespace NUMINAMATH_CALUDE_range_of_m2_plus_n2_l686_68607

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the theorem
theorem range_of_m2_plus_n2
  (f : ℝ → ℝ)
  (h_decreasing : DecreasingFunction f)
  (h_inequality : ∀ m n : ℝ, f (n^2 - 10*n - 15) ≥ f (12 - m^2 + 24*m)) :
  ∀ m n : ℝ, 0 ≤ m^2 + n^2 ∧ m^2 + n^2 ≤ 729 :=
sorry

end NUMINAMATH_CALUDE_range_of_m2_plus_n2_l686_68607


namespace NUMINAMATH_CALUDE_gcd_of_128_144_512_l686_68678

theorem gcd_of_128_144_512 : Nat.gcd 128 (Nat.gcd 144 512) = 16 := by sorry

end NUMINAMATH_CALUDE_gcd_of_128_144_512_l686_68678


namespace NUMINAMATH_CALUDE_polynomial_simplification_l686_68692

theorem polynomial_simplification (x : ℝ) : (3 * x^2 - 4 * x + 5) - (2 * x^2 - 6 * x - 8) = x^2 + 2 * x + 13 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l686_68692


namespace NUMINAMATH_CALUDE_juice_dispenser_capacity_l686_68660

/-- A cylindrical juice dispenser with capacity x cups -/
structure JuiceDispenser where
  capacity : ℝ
  cylindrical : Bool

/-- Theorem: A cylindrical juice dispenser that contains 60 cups when 48% full has a total capacity of 125 cups -/
theorem juice_dispenser_capacity (d : JuiceDispenser) 
  (h_cylindrical : d.cylindrical = true) 
  (h_partial : 0.48 * d.capacity = 60) : 
  d.capacity = 125 := by
  sorry

end NUMINAMATH_CALUDE_juice_dispenser_capacity_l686_68660


namespace NUMINAMATH_CALUDE_oranges_picked_theorem_l686_68647

/-- The total number of oranges picked over three days --/
def total_oranges (monday : ℕ) (tuesday_multiplier : ℕ) (wednesday : ℕ) : ℕ :=
  monday + tuesday_multiplier * monday + wednesday

/-- Theorem: Given the conditions, the total number of oranges picked is 470 --/
theorem oranges_picked_theorem (monday : ℕ) (tuesday_multiplier : ℕ) (wednesday : ℕ)
  (h1 : monday = 100)
  (h2 : tuesday_multiplier = 3)
  (h3 : wednesday = 70) :
  total_oranges monday tuesday_multiplier wednesday = 470 := by
  sorry

end NUMINAMATH_CALUDE_oranges_picked_theorem_l686_68647


namespace NUMINAMATH_CALUDE_consumption_increase_l686_68610

theorem consumption_increase (T C : ℝ) (h1 : T > 0) (h2 : C > 0) : 
  let new_tax := 0.7 * T
  let new_revenue := 0.77 * (T * C)
  let new_consumption := C * (1 + 10/100)
  new_tax * new_consumption = new_revenue :=
sorry

end NUMINAMATH_CALUDE_consumption_increase_l686_68610


namespace NUMINAMATH_CALUDE_salary_increase_proof_l686_68649

/-- Proves that given the conditions of the salary increase problem, the new salary is $90,000 -/
theorem salary_increase_proof (S : ℝ) 
  (h1 : S + 25000 = S * (1 + 0.3846153846153846)) : S + 25000 = 90000 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_proof_l686_68649


namespace NUMINAMATH_CALUDE_jeanine_has_more_pencils_jeanine_has_more_pencils_proof_l686_68627

/-- The number of pencils Jeanine has after giving some to Abby is 3 more than Clare's pencils. -/
theorem jeanine_has_more_pencils : ℕ → Prop :=
  fun (initial_pencils : ℕ) =>
    initial_pencils = 18 →
    let clare_pencils := initial_pencils / 2
    let jeanine_remaining := initial_pencils - (initial_pencils / 3)
    jeanine_remaining - clare_pencils = 3

/-- Proof of the theorem -/
theorem jeanine_has_more_pencils_proof : jeanine_has_more_pencils 18 := by
  sorry

#check jeanine_has_more_pencils_proof

end NUMINAMATH_CALUDE_jeanine_has_more_pencils_jeanine_has_more_pencils_proof_l686_68627


namespace NUMINAMATH_CALUDE_optimal_price_maximizes_revenue_l686_68659

/-- Revenue function for the bookstore --/
def R (p : ℝ) : ℝ := p * (150 - 6 * p)

/-- The optimal price maximizes the revenue --/
theorem optimal_price_maximizes_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 30 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 30 → R p ≥ R q ∧
  p = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_optimal_price_maximizes_revenue_l686_68659


namespace NUMINAMATH_CALUDE_change_calculation_l686_68667

def bracelet_price : ℚ := 15
def necklace_price : ℚ := 10
def mug_price : ℚ := 20
def keychain_price : ℚ := 5

def bracelet_quantity : ℕ := 3
def necklace_quantity : ℕ := 2
def mug_quantity : ℕ := 1
def keychain_quantity : ℕ := 4

def discount_rate : ℚ := 12 / 100
def payment : ℚ := 100

def total_before_discount : ℚ :=
  bracelet_price * bracelet_quantity +
  necklace_price * necklace_quantity +
  mug_price * mug_quantity +
  keychain_price * keychain_quantity

def discount_amount : ℚ := total_before_discount * discount_rate
def final_amount : ℚ := total_before_discount - discount_amount

theorem change_calculation :
  payment - final_amount = 760 / 100 := by sorry

end NUMINAMATH_CALUDE_change_calculation_l686_68667


namespace NUMINAMATH_CALUDE_positive_numbers_inequalities_l686_68645

theorem positive_numbers_inequalities 
  (a b c : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_pos_c : c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a * b * c ≤ 1/9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequalities_l686_68645


namespace NUMINAMATH_CALUDE_simplify_nested_expression_l686_68636

theorem simplify_nested_expression (x : ℝ) : 1 - (2 - (1 + (2 - (3 - x)))) = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_nested_expression_l686_68636


namespace NUMINAMATH_CALUDE_opposite_numbers_sum_l686_68625

theorem opposite_numbers_sum (a b : ℝ) : a + b = 0 → 3*a + 3*b - 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_numbers_sum_l686_68625


namespace NUMINAMATH_CALUDE_unique_valid_number_l686_68676

def is_valid_product (a b : Nat) : Prop :=
  ∃ (x y : Nat), x < 10 ∧ y < 10 ∧ a * 10 + b = x * y

def is_valid_number (n : Nat) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  (∀ i : Fin 9, (n / 10^i.val % 10) ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (∀ i : Fin 9, ∀ j : Fin 9, i ≠ j → (n / 10^i.val % 10) ≠ (n / 10^j.val % 10)) ∧
  (∀ i : Fin 8, is_valid_product (n / 10^(i+1).val % 10) (n / 10^i.val % 10))

theorem unique_valid_number : 
  ∃! n : Nat, is_valid_number n ∧ n = 728163549 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l686_68676


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l686_68694

/-- 
Given a repeating decimal 0.6̄13̄ (where 13 repeats infinitely after 6),
prove that it is equal to the fraction 362/495.
-/
theorem repeating_decimal_to_fraction : 
  (6/10 : ℚ) + (13/99 : ℚ) = 362/495 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l686_68694


namespace NUMINAMATH_CALUDE_d_magnitude_when_Q_has_five_roots_l686_68635

/-- The polynomial Q(x) -/
def Q (d : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 3) * (x^2 - d*x + 5) * (x^2 - 5*x + 10)

/-- The theorem stating that if Q has exactly 5 distinct roots, then |d| = √28 -/
theorem d_magnitude_when_Q_has_five_roots (d : ℂ) :
  (∃ (S : Finset ℂ), S.card = 5 ∧ (∀ x : ℂ, x ∈ S ↔ Q d x = 0) ∧ (∀ x y : ℂ, x ∈ S → y ∈ S → x ≠ y → Q d x = 0 → Q d y = 0 → x ≠ y)) →
  Complex.abs d = Real.sqrt 28 := by
sorry

end NUMINAMATH_CALUDE_d_magnitude_when_Q_has_five_roots_l686_68635


namespace NUMINAMATH_CALUDE_f_of_g_of_3_l686_68681

/-- Given functions f and g, prove that f(2 + g(3)) = 44 -/
theorem f_of_g_of_3 (f g : ℝ → ℝ) 
    (hf : ∀ x, f x = 3 * x - 4)
    (hg : ∀ x, g x = x^2 + 2 * x - 1) : 
  f (2 + g 3) = 44 := by
  sorry

end NUMINAMATH_CALUDE_f_of_g_of_3_l686_68681


namespace NUMINAMATH_CALUDE_complement_of_M_wrt_U_l686_68619

def U : Finset Int := {1, -2, 3, -4, 5, -6}
def M : Finset Int := {1, -2, 3, -4}

theorem complement_of_M_wrt_U :
  U \ M = {5, -6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_wrt_U_l686_68619


namespace NUMINAMATH_CALUDE_fourth_hexagon_dots_l686_68620

/-- Calculates the number of dots in the nth layer of the hexagonal pattern. -/
def layerDots (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n % 2 = 0 then 7 * (n - 1)
  else 7 * n

/-- Calculates the total number of dots in the nth hexagon of the sequence. -/
def totalDots (n : ℕ) : ℕ :=
  (List.range n).map layerDots |> List.sum

/-- The fourth hexagon in the sequence contains 50 dots. -/
theorem fourth_hexagon_dots : totalDots 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_fourth_hexagon_dots_l686_68620


namespace NUMINAMATH_CALUDE_least_months_to_triple_debt_l686_68687

theorem least_months_to_triple_debt (interest_rate : ℝ) (n : ℕ) : 
  interest_rate = 0.03 →
  n = 37 →
  (∀ m : ℕ, m < n → (1 + interest_rate)^m ≤ 3) ∧
  (1 + interest_rate)^n > 3 :=
sorry

end NUMINAMATH_CALUDE_least_months_to_triple_debt_l686_68687


namespace NUMINAMATH_CALUDE_novel_pages_per_hour_l686_68696

-- Define the reading time in hours
def total_reading_time : ℚ := 1/6 * 24

-- Define the reading time for each type of book
def reading_time_per_type : ℚ := total_reading_time / 3

-- Define the pages read per hour for comic books and graphic novels
def comic_pages_per_hour : ℕ := 45
def graphic_pages_per_hour : ℕ := 30

-- Define the total pages read
def total_pages_read : ℕ := 128

-- Theorem to prove
theorem novel_pages_per_hour : 
  ∃ (n : ℕ), 
    (n : ℚ) * reading_time_per_type + 
    (comic_pages_per_hour : ℚ) * reading_time_per_type + 
    (graphic_pages_per_hour : ℚ) * reading_time_per_type = total_pages_read ∧ 
    n = 21 := by
  sorry

end NUMINAMATH_CALUDE_novel_pages_per_hour_l686_68696


namespace NUMINAMATH_CALUDE_binomial_coefficient_16_4_l686_68614

theorem binomial_coefficient_16_4 : Nat.choose 16 4 = 1820 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_16_4_l686_68614


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l686_68658

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (9 + 3 * z) = 12 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l686_68658


namespace NUMINAMATH_CALUDE_number_exceeds_fraction_l686_68609

theorem number_exceeds_fraction (N : ℚ) (F : ℚ) : 
  N = 56 → N = F + 35 → F = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_number_exceeds_fraction_l686_68609


namespace NUMINAMATH_CALUDE_ellipse_equation_eccentricity_range_l686_68661

noncomputable section

-- Define the ellipse parameters
def m : ℝ := 1  -- We know m = 1 from the solution, but we keep it as a parameter

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the focus
def focus : ℝ × ℝ := (m, 0)

-- Define the directrices
def left_directrix (x : ℝ) : Prop := x = -m - 1
def right_directrix (x : ℝ) : Prop := x = m + 1

-- Define the line y = x
def diagonal_line (x y : ℝ) : Prop := y = x

-- Define points A and B
def point_A : ℝ × ℝ := (-m - 1, -m - 1)
def point_B : ℝ × ℝ := (m + 1, m + 1)

-- Define vectors AF and FB
def vector_AF : ℝ × ℝ := (2*m + 1, m + 1)
def vector_FB : ℝ × ℝ := (1, m + 1)

-- Define dot product of AF and FB
def dot_product_AF_FB : ℝ := (2*m + 1) * 1 + (m + 1) * (m + 1)

-- Define eccentricity
def eccentricity : ℝ := 1 / Real.sqrt (1 + 1/m)

-- Theorem 1: Prove the equation of the ellipse
theorem ellipse_equation : 
  ∀ x y : ℝ, ellipse x y ↔ x^2 / 2 + y^2 = 1 :=
sorry

-- Theorem 2: Prove the range of eccentricity
theorem eccentricity_range :
  dot_product_AF_FB < 7 → 0 < eccentricity ∧ eccentricity < Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_eccentricity_range_l686_68661


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_9_l686_68643

def is_divisible_by_range (n : ℕ) (a b : ℕ) : Prop :=
  ∀ i : ℕ, a ≤ i → i ≤ b → n % i = 0

theorem smallest_divisible_by_1_to_9 :
  ∃ (n : ℕ), n > 0 ∧ is_divisible_by_range n 1 9 ∧
  ∀ (m : ℕ), m > 0 → is_divisible_by_range m 1 9 → n ≤ m :=
by
  use 2520
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_9_l686_68643


namespace NUMINAMATH_CALUDE_decimal_numbers_less_than_one_infinite_l686_68684

theorem decimal_numbers_less_than_one_infinite :
  Set.Infinite {x : ℝ | x < 1 ∧ ∃ (n : ℕ), x = ↑n / (10 ^ n)} :=
sorry

end NUMINAMATH_CALUDE_decimal_numbers_less_than_one_infinite_l686_68684


namespace NUMINAMATH_CALUDE_function_value_at_negative_l686_68639

theorem function_value_at_negative (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = x^5 + x^3 + 1) →
  f m = 10 →
  f (-m) = -8 := by
sorry

end NUMINAMATH_CALUDE_function_value_at_negative_l686_68639


namespace NUMINAMATH_CALUDE_towel_bleaching_l686_68670

theorem towel_bleaching (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let L' := L * (1 - x)
  let B' := B * 0.85
  L' * B' = (L * B) * 0.595
  →
  x = 0.3
  := by sorry

end NUMINAMATH_CALUDE_towel_bleaching_l686_68670


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l686_68615

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- Define the theorem
theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (a 2)^2 - 34*(a 2) + 64 = 0 →
  (a 6)^2 - 34*(a 6) + 64 = 0 →
  a 4 = 8 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l686_68615


namespace NUMINAMATH_CALUDE_consecutive_blue_gumballs_probability_l686_68622

theorem consecutive_blue_gumballs_probability :
  let p_pink : ℝ := 0.1428571428571428
  let p_blue : ℝ := 1 - p_pink
  p_blue * p_blue = 0.7346938775510203 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_blue_gumballs_probability_l686_68622


namespace NUMINAMATH_CALUDE_not_heart_zero_sum_property_l686_68675

def heart (x y : ℝ) : ℝ := |x + y|

theorem not_heart_zero_sum_property : ¬ ∀ x y : ℝ, (heart x 0 + heart 0 y = heart x y) := by
  sorry

end NUMINAMATH_CALUDE_not_heart_zero_sum_property_l686_68675


namespace NUMINAMATH_CALUDE_chocolate_theorem_l686_68677

/-- The number of chocolates Nick has -/
def nick_chocolates : ℕ := 10

/-- The factor by which Alix's chocolates exceed Nick's -/
def alix_factor : ℕ := 3

/-- The number of chocolates mom took from Alix -/
def mom_took : ℕ := 5

/-- The difference in chocolates between Alix and Nick after mom took some -/
def chocolate_difference : ℕ := 15

theorem chocolate_theorem :
  (alix_factor * nick_chocolates - mom_took) - nick_chocolates = chocolate_difference := by
  sorry

end NUMINAMATH_CALUDE_chocolate_theorem_l686_68677


namespace NUMINAMATH_CALUDE_circus_crowns_l686_68630

theorem circus_crowns (total_feathers : ℕ) (feathers_per_crown : ℕ) (h1 : total_feathers = 6538) (h2 : feathers_per_crown = 7) :
  total_feathers / feathers_per_crown = 934 := by
  sorry

end NUMINAMATH_CALUDE_circus_crowns_l686_68630


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l686_68666

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 + 4*x = 5 ↔ x = 1 ∨ x = -5 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l686_68666


namespace NUMINAMATH_CALUDE_root_in_interval_l686_68679

-- Define the function f(x) = x³ - 4
def f (x : ℝ) : ℝ := x^3 - 4

-- State the theorem
theorem root_in_interval :
  (f 1 < 0) → (f 2 > 0) → ∃ x : ℝ, x ∈ (Set.Ioo 1 2) ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_l686_68679


namespace NUMINAMATH_CALUDE_multiply_fractions_l686_68626

theorem multiply_fractions : (7 : ℚ) * (1 / 17) * 34 = 14 := by sorry

end NUMINAMATH_CALUDE_multiply_fractions_l686_68626


namespace NUMINAMATH_CALUDE_cars_sold_last_three_days_l686_68617

/-- Represents the number of cars sold by a salesman over 6 days -/
structure CarSales where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ
  day5 : ℕ
  day6 : ℕ

/-- Calculates the mean of car sales over 6 days -/
def meanSales (sales : CarSales) : ℚ :=
  (sales.day1 + sales.day2 + sales.day3 + sales.day4 + sales.day5 + sales.day6 : ℚ) / 6

/-- Theorem stating the number of cars sold in the last three days -/
theorem cars_sold_last_three_days (sales : CarSales) 
  (h1 : sales.day1 = 8)
  (h2 : sales.day2 = 3)
  (h3 : sales.day3 = 10)
  (h_mean : meanSales sales = 5.5) :
  sales.day4 + sales.day5 + sales.day6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cars_sold_last_three_days_l686_68617


namespace NUMINAMATH_CALUDE_solution_of_equation_l686_68688

theorem solution_of_equation (x : ℝ) :
  x ≠ 3 →
  ((2 - x) / (x - 3) = 0) ↔ (x = 2) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_equation_l686_68688


namespace NUMINAMATH_CALUDE_box_weight_is_42_l686_68640

/-- The weight of a box of books -/
def box_weight (book_weight : ℕ) (num_books : ℕ) : ℕ :=
  book_weight * num_books

/-- Theorem: The weight of a box of books is 42 pounds -/
theorem box_weight_is_42 : box_weight 3 14 = 42 := by
  sorry

end NUMINAMATH_CALUDE_box_weight_is_42_l686_68640


namespace NUMINAMATH_CALUDE_find_number_l686_68689

theorem find_number : ∃ x : ℝ, (((x - 1.9) * 1.5 + 32) / 2.5) = 20 ∧ x = 13.9 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l686_68689


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l686_68656

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

/-- The theorem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_sum : a 4 + a 8 = -2) :
  a 6 * (a 2 + 2 * a 6 + a 10) = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l686_68656


namespace NUMINAMATH_CALUDE_sum_of_root_products_l686_68665

theorem sum_of_root_products (p q r : ℂ) : 
  (4 * p^3 - 2 * p^2 + 13 * p - 9 = 0) →
  (4 * q^3 - 2 * q^2 + 13 * q - 9 = 0) →
  (4 * r^3 - 2 * r^2 + 13 * r - 9 = 0) →
  p * q + p * r + q * r = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_root_products_l686_68665


namespace NUMINAMATH_CALUDE_line_trig_identity_l686_68690

/-- Given a line with direction vector (-1, 2) and inclination angle α, 
    prove that sin(2α) - cos²(α) - 1 = -2 -/
theorem line_trig_identity (α : Real) (h : Real.tan α = -2) : 
  Real.sin (2 * α) - Real.cos α ^ 2 - 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_line_trig_identity_l686_68690


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l686_68634

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third is determined by these two
  base_angle : Real
  vertex_angle : Real
  -- Condition that the sum of angles in a triangle is 180°
  angle_sum : base_angle * 2 + vertex_angle = 180

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.base_angle = 80 ∨ triangle.vertex_angle = 80) :
  triangle.vertex_angle = 80 ∨ triangle.vertex_angle = 20 := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l686_68634


namespace NUMINAMATH_CALUDE_complex_square_one_plus_i_l686_68664

theorem complex_square_one_plus_i (i : ℂ) : i * i = -1 → (1 + i)^2 = 2*i := by
  sorry

end NUMINAMATH_CALUDE_complex_square_one_plus_i_l686_68664


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l686_68671

/-- Given a point M(2,a) on the graph of y = k/x where k > 0, prove that the coordinates of M are both positive. -/
theorem point_in_first_quadrant (k a : ℝ) (h1 : k > 0) (h2 : a = k / 2) : 2 > 0 ∧ a > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l686_68671


namespace NUMINAMATH_CALUDE_b_52_mod_55_l686_68637

/-- Definition of b_n as the integer obtained by writing all integers from 1 to n from left to right -/
def b (n : ℕ) : ℕ := sorry

/-- Theorem stating that the remainder of b_52 divided by 55 is 2 -/
theorem b_52_mod_55 : b 52 % 55 = 2 := by sorry

end NUMINAMATH_CALUDE_b_52_mod_55_l686_68637


namespace NUMINAMATH_CALUDE_unique_divisibility_condition_l686_68662

theorem unique_divisibility_condition (n : ℕ) : n > 1 → (
  (∃! a : ℕ, 0 < a ∧ a ≤ Nat.factorial n ∧ (Nat.factorial n ∣ a^n + 1)) ↔ n = 2
) := by sorry

end NUMINAMATH_CALUDE_unique_divisibility_condition_l686_68662


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l686_68673

theorem sufficient_not_necessary (a b : ℝ) :
  (0 < a ∧ a < b → 1 / a > 1 / b) ∧
  ∃ a b : ℝ, 1 / a > 1 / b ∧ ¬(0 < a ∧ a < b) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l686_68673


namespace NUMINAMATH_CALUDE_infinitely_many_primes_in_differences_l686_68686

/-- Definition of the sequence a_n -/
def a (k : ℕ) : ℕ → ℕ
  | n => if n < k then 0  -- arbitrary value for n < k
         else if n = k then 2 * k
         else if Nat.gcd (a k (n-1)) n = 1 then a k (n-1) + 1
         else 2 * n

/-- The theorem statement -/
theorem infinitely_many_primes_in_differences (k : ℕ) (h : k ≥ 3) :
  ∀ M : ℕ, ∃ n > k, ∃ p : ℕ, p.Prime ∧ p > M ∧ p ∣ (a k n - a k (n-1)) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_primes_in_differences_l686_68686


namespace NUMINAMATH_CALUDE_shirt_price_calculation_l686_68651

def shorts_price : ℝ := 15
def jacket_price : ℝ := 14.82
def total_spent : ℝ := 42.33

theorem shirt_price_calculation : 
  ∃ (shirt_price : ℝ), shirt_price = total_spent - (shorts_price + jacket_price) ∧ shirt_price = 12.51 :=
by sorry

end NUMINAMATH_CALUDE_shirt_price_calculation_l686_68651


namespace NUMINAMATH_CALUDE_pizza_toppings_count_l686_68682

theorem pizza_toppings_count (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) + (n.choose 3) = 92 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_count_l686_68682


namespace NUMINAMATH_CALUDE_barium_chloride_weight_l686_68631

/-- The atomic weight of barium in g/mol -/
def atomic_weight_Ba : ℝ := 137.33

/-- The atomic weight of chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- The number of moles of barium chloride -/
def moles_BaCl2 : ℝ := 4

/-- The molecular weight of barium chloride in g/mol -/
def molecular_weight_BaCl2 : ℝ := atomic_weight_Ba + 2 * atomic_weight_Cl

/-- The total weight of barium chloride in grams -/
def total_weight_BaCl2 : ℝ := moles_BaCl2 * molecular_weight_BaCl2

theorem barium_chloride_weight :
  total_weight_BaCl2 = 832.92 := by sorry

end NUMINAMATH_CALUDE_barium_chloride_weight_l686_68631


namespace NUMINAMATH_CALUDE_gcd_lcm_examples_l686_68698

theorem gcd_lcm_examples : 
  (Nat.gcd 17 51 = 17) ∧ 
  (Nat.lcm 17 51 = 51) ∧ 
  (Nat.gcd 6 8 = 2) ∧ 
  (Nat.lcm 8 9 = 72) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_examples_l686_68698


namespace NUMINAMATH_CALUDE_extremum_at_negative_three_l686_68699

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 5*x^2 + a*x

-- State the theorem
theorem extremum_at_negative_three (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x ≠ -3 ∧ |x + 3| < ε → f a x ≤ f a (-3)) ∨
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x ≠ -3 ∧ |x + 3| < ε → f a x ≥ f a (-3)) →
  a = 3 := by
sorry


end NUMINAMATH_CALUDE_extremum_at_negative_three_l686_68699


namespace NUMINAMATH_CALUDE_one_integral_root_l686_68623

theorem one_integral_root :
  ∃! (x : ℤ), x - 9 / (x + 4 : ℚ) = 2 - 9 / (x + 4 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_one_integral_root_l686_68623


namespace NUMINAMATH_CALUDE_regression_line_change_l686_68672

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Calculates the y-value for a given x-value on the regression line -/
def RegressionLine.predict (line : RegressionLine) (x : ℝ) : ℝ :=
  line.intercept + line.slope * x

/-- Theorem: For the regression line y = 2 - 1.5x, 
    when x increases by 1 unit, y decreases by 1.5 units -/
theorem regression_line_change 
  (line : RegressionLine) 
  (h1 : line.intercept = 2) 
  (h2 : line.slope = -1.5) 
  (x : ℝ) : 
  line.predict (x + 1) = line.predict x - 1.5 := by
  sorry


end NUMINAMATH_CALUDE_regression_line_change_l686_68672


namespace NUMINAMATH_CALUDE_subset_intersection_bound_l686_68668

/-- Given a set S of n elements and a family of b subsets of S, each containing k elements,
    with the property that any two subsets intersect in at most one element,
    the number of subsets b is bounded above by ⌊(n/k)⌊(n-1)/(k-1)⌋⌋. -/
theorem subset_intersection_bound (n k b : ℕ) (S : Finset (Fin n)) (B : Fin b → Finset (Fin n)) 
  (h1 : ∀ i, (B i).card = k)
  (h2 : ∀ i j, i < j → (B i ∩ B j).card ≤ 1)
  (h3 : k > 0)
  (h4 : n > 0)
  : b ≤ ⌊(n : ℝ) / k * ⌊(n - 1 : ℝ) / (k - 1)⌋⌋ :=
sorry

end NUMINAMATH_CALUDE_subset_intersection_bound_l686_68668


namespace NUMINAMATH_CALUDE_sewing_time_proof_l686_68644

/-- The time it takes to sew one dress -/
def time_per_dress (num_dresses : ℕ) (weekly_sewing_time : ℕ) (total_weeks : ℕ) : ℕ :=
  (weekly_sewing_time * total_weeks) / num_dresses

/-- Theorem stating that the time to sew one dress is 12 hours -/
theorem sewing_time_proof (num_dresses : ℕ) (weekly_sewing_time : ℕ) (total_weeks : ℕ) 
  (h1 : num_dresses = 5)
  (h2 : weekly_sewing_time = 4)
  (h3 : total_weeks = 15) :
  time_per_dress num_dresses weekly_sewing_time total_weeks = 12 := by
  sorry

end NUMINAMATH_CALUDE_sewing_time_proof_l686_68644


namespace NUMINAMATH_CALUDE_total_hike_length_l686_68650

/-- The length of a hike given the distance hiked on the first day and the remaining distance. -/
def hike_length (first_day_distance : ℕ) (remaining_distance : ℕ) : ℕ :=
  first_day_distance + remaining_distance

/-- Theorem stating that the total length of the hike is 36 miles. -/
theorem total_hike_length :
  hike_length 9 27 = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_hike_length_l686_68650


namespace NUMINAMATH_CALUDE_pq_is_one_eighth_of_rs_l686_68674

-- Define the line segment RS and points P and Q on it
structure LineSegment where
  length : ℝ

structure Point where
  position : ℝ

-- Define the problem setup
def problem (RS : LineSegment) (P Q : Point) : Prop :=
  -- P and Q lie on RS
  0 ≤ P.position ∧ P.position ≤ RS.length ∧
  0 ≤ Q.position ∧ Q.position ≤ RS.length ∧
  -- RP is 3 times PS
  P.position = (3/4) * RS.length ∧
  -- RQ is 7 times QS
  Q.position = (7/8) * RS.length

-- Theorem to prove
theorem pq_is_one_eighth_of_rs (RS : LineSegment) (P Q : Point) 
  (h : problem RS P Q) : 
  abs (Q.position - P.position) = (1/8) * RS.length :=
sorry

end NUMINAMATH_CALUDE_pq_is_one_eighth_of_rs_l686_68674


namespace NUMINAMATH_CALUDE_count_numerators_T_l686_68680

/-- The set of rational numbers with repeating decimal expansion 0.overline(ab) -/
def T : Set ℚ :=
  {r | 0 < r ∧ r < 1 ∧ ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ r = (10 * a + b : ℚ) / 99}

/-- The number of different numerators required to express all elements of T in lowest terms -/
def num_different_numerators : ℕ := 53

/-- Theorem stating that the number of different numerators for T is 53 -/
theorem count_numerators_T : num_different_numerators = 53 := by
  sorry

end NUMINAMATH_CALUDE_count_numerators_T_l686_68680


namespace NUMINAMATH_CALUDE_orange_juice_distribution_l686_68602

theorem orange_juice_distribution (pitcher_capacity : ℝ) (h : pitcher_capacity > 0) :
  let juice_amount : ℝ := (5 / 8) * pitcher_capacity
  let num_cups : ℕ := 4
  let juice_per_cup : ℝ := juice_amount / num_cups
  (juice_per_cup / pitcher_capacity) * 100 = 15.625 := by
sorry

end NUMINAMATH_CALUDE_orange_juice_distribution_l686_68602


namespace NUMINAMATH_CALUDE_eighth_term_geometric_sequence_l686_68642

theorem eighth_term_geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) :
  a₁ = 27 ∧ r = 1/3 ∧ n = 8 →
  a₁ * r^(n - 1) = 1/81 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_geometric_sequence_l686_68642


namespace NUMINAMATH_CALUDE_total_fish_count_l686_68633

/-- The number of tuna in the sea -/
def num_tuna : ℕ := 5

/-- The number of spearfish in the sea -/
def num_spearfish : ℕ := 2

/-- The total number of fish in the sea -/
def total_fish : ℕ := num_tuna + num_spearfish

theorem total_fish_count : total_fish = 7 := by sorry

end NUMINAMATH_CALUDE_total_fish_count_l686_68633


namespace NUMINAMATH_CALUDE_fraction_calculation_l686_68653

theorem fraction_calculation : 
  (8 / 17) / (7 / 5) + (5 / 7) * (9 / 17) = 5 / 7 := by sorry

end NUMINAMATH_CALUDE_fraction_calculation_l686_68653


namespace NUMINAMATH_CALUDE_vector_sum_proof_l686_68695

theorem vector_sum_proof :
  let v₁ : Fin 3 → ℝ := ![3, -2, 7]
  let v₂ : Fin 3 → ℝ := ![-1, 5, -3]
  v₁ + v₂ = ![2, 3, 4] := by
  sorry

end NUMINAMATH_CALUDE_vector_sum_proof_l686_68695


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l686_68618

/-- An arithmetic sequence is defined by its first term and common difference. -/
structure ArithmeticSequence where
  first_term : ℝ
  common_difference : ℝ

/-- Get the nth term of an arithmetic sequence. -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1 : ℝ) * seq.common_difference

/-- Our specific arithmetic sequence with given conditions. -/
def our_sequence : ArithmeticSequence :=
  { first_term := 0,  -- We don't know the first term yet, so we use a placeholder
    common_difference := 0 }  -- We don't know the common difference yet, so we use a placeholder

theorem arithmetic_sequence_problem :
  our_sequence.nthTerm 3 = 10 ∧
  our_sequence.nthTerm 20 = 65 →
  our_sequence.nthTerm 32 = 103.8235294118 := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l686_68618


namespace NUMINAMATH_CALUDE_optimal_sequence_l686_68621

theorem optimal_sequence (x₁ x₂ x₃ x₄ : ℝ) 
  (h_order : 0 ≤ x₄ ∧ x₄ ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_eq : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + (x₃ - x₄)^2 + x₄^2 = 1/5) :
  x₁ = 4/5 ∧ x₂ = 3/5 ∧ x₃ = 2/5 ∧ x₄ = 1/5 := by
sorry

end NUMINAMATH_CALUDE_optimal_sequence_l686_68621


namespace NUMINAMATH_CALUDE_triangle_ratio_l686_68616

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  b * (Real.cos C) + c * (Real.cos B) = 2 * b →
  a / b = 2 := by
sorry


end NUMINAMATH_CALUDE_triangle_ratio_l686_68616


namespace NUMINAMATH_CALUDE_campers_rowing_in_morning_l686_68655

theorem campers_rowing_in_morning (afternoon_campers : ℕ) (difference : ℕ) 
  (h1 : afternoon_campers = 61)
  (h2 : afternoon_campers = difference + morning_campers) 
  (h3 : difference = 9) : 
  morning_campers = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_in_morning_l686_68655


namespace NUMINAMATH_CALUDE_not_5x_representation_l686_68613

-- Define the expressions
def expr_A (x : ℝ) : ℝ := 5 * x
def expr_B (x : ℝ) : ℝ := x^5
def expr_C (x : ℝ) : ℝ := x + x + x + x + x

-- Theorem stating that B is not equal to 5x, while A and C are
theorem not_5x_representation (x : ℝ) : 
  expr_A x = 5 * x ∧ expr_C x = 5 * x ∧ expr_B x ≠ 5 * x :=
sorry

end NUMINAMATH_CALUDE_not_5x_representation_l686_68613


namespace NUMINAMATH_CALUDE_unique_negative_solution_implies_positive_a_l686_68632

theorem unique_negative_solution_implies_positive_a (a : ℝ) : 
  (∃! x : ℝ, (abs x = 2 * x + a) ∧ (x < 0)) → a > 0 := by
sorry

end NUMINAMATH_CALUDE_unique_negative_solution_implies_positive_a_l686_68632


namespace NUMINAMATH_CALUDE_simplify_sqrt_neg_five_squared_l686_68646

theorem simplify_sqrt_neg_five_squared : Real.sqrt ((-5)^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_neg_five_squared_l686_68646


namespace NUMINAMATH_CALUDE_milly_fold_count_l686_68652

/-- Represents the croissant-making process with given time constraints. -/
structure CroissantProcess where
  fold_time : ℕ         -- Time to fold dough once (in minutes)
  rest_time : ℕ         -- Time to rest dough once (in minutes)
  mix_time : ℕ          -- Time to mix ingredients (in minutes)
  bake_time : ℕ         -- Time to bake (in minutes)
  total_time : ℕ        -- Total time for the whole process (in minutes)

/-- Calculates the number of times the dough needs to be folded. -/
def fold_count (process : CroissantProcess) : ℕ :=
  ((process.total_time - process.mix_time - process.bake_time) / 
   (process.fold_time + process.rest_time))

/-- Theorem stating that for the given process, the dough needs to be folded 4 times. -/
theorem milly_fold_count : 
  let process : CroissantProcess := {
    fold_time := 5,
    rest_time := 75,
    mix_time := 10,
    bake_time := 30,
    total_time := 6 * 60  -- 6 hours in minutes
  }
  fold_count process = 4 := by
  sorry

end NUMINAMATH_CALUDE_milly_fold_count_l686_68652
