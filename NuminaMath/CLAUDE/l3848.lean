import Mathlib

namespace NUMINAMATH_CALUDE_binary_11111_equals_31_l3848_384879

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11111_equals_31 :
  binary_to_decimal [true, true, true, true, true] = 31 := by
  sorry

end NUMINAMATH_CALUDE_binary_11111_equals_31_l3848_384879


namespace NUMINAMATH_CALUDE_no_common_integers_satisfying_condition_l3848_384822

theorem no_common_integers_satisfying_condition : 
  ¬∃ i : ℤ, 10 ≤ i ∧ i ≤ 30 ∧ i^2 - 5*i - 6 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_common_integers_satisfying_condition_l3848_384822


namespace NUMINAMATH_CALUDE_sum_after_removal_l3848_384803

theorem sum_after_removal (a b c d e f : ℚ) : 
  a = 1/3 → b = 1/6 → c = 1/9 → d = 1/12 → e = 1/15 → f = 1/18 →
  a + b + c + f = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_removal_l3848_384803


namespace NUMINAMATH_CALUDE_system_solution_equivalence_l3848_384831

-- Define the system of equations
def system (x y z : ℝ) : Prop :=
  (x * y + 2 * x * z + 3 * y * z = -6) ∧
  (x^2 * y^2 + 4 * x^2 * z^2 - 9 * y^2 * z^2 = 36) ∧
  (x^3 * y^3 + 8 * x^3 * z^3 + 27 * y^3 * z^3 = -216)

-- Define the solution set
def solution_set (x y z : ℝ) : Prop :=
  (y = 0 ∧ x * z = -3) ∨
  (z = 0 ∧ x * y = -6) ∨
  (x = 3 ∧ y = -2) ∨
  (x = -3 ∧ y = 2) ∨
  (x = 3 ∧ z = -1) ∨
  (x = -3 ∧ z = 1)

-- State the theorem
theorem system_solution_equivalence :
  ∀ x y z : ℝ, system x y z ↔ solution_set x y z :=
sorry

end NUMINAMATH_CALUDE_system_solution_equivalence_l3848_384831


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3848_384805

theorem quadratic_equation_solution (k : ℚ) : 
  (∀ x : ℚ, k * x^2 + 8 * x + 15 = 0 ↔ (x = -3 ∨ x = -5/2)) → k = 11/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3848_384805


namespace NUMINAMATH_CALUDE_students_not_in_biology_l3848_384814

theorem students_not_in_biology (total_students : ℕ) (biology_percentage : ℚ) 
  (h1 : total_students = 880) 
  (h2 : biology_percentage = 325 / 1000) : 
  total_students - (total_students * biology_percentage).floor = 594 := by
  sorry

end NUMINAMATH_CALUDE_students_not_in_biology_l3848_384814


namespace NUMINAMATH_CALUDE_simplify_trig_expression_130_degrees_simplify_trig_expression_second_quadrant_l3848_384874

-- Part 1
theorem simplify_trig_expression_130_degrees :
  (Real.sqrt (1 - 2 * Real.sin (130 * π / 180) * Real.cos (130 * π / 180))) /
  (Real.sin (130 * π / 180) + Real.sqrt (1 - Real.sin (130 * π / 180) ^ 2)) = 1 := by
sorry

-- Part 2
theorem simplify_trig_expression_second_quadrant (α : Real) 
  (h : π / 2 < α ∧ α < π) :
  Real.cos α * Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) +
  Real.sin α * Real.sqrt ((1 - Real.cos α) / (1 + Real.cos α)) =
  Real.sin α - Real.cos α := by
sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_130_degrees_simplify_trig_expression_second_quadrant_l3848_384874


namespace NUMINAMATH_CALUDE_solve_system_for_x_l3848_384896

theorem solve_system_for_x :
  ∀ x y : ℚ, 
  (2 * x - 3 * y = 18) → 
  (x + 2 * y = 8) → 
  x = 60 / 7 := by
sorry

end NUMINAMATH_CALUDE_solve_system_for_x_l3848_384896


namespace NUMINAMATH_CALUDE_triangle_formation_l3848_384801

/-- Checks if three lengths can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given sticks of length 6 and 12, proves which of the given lengths can form a triangle -/
theorem triangle_formation (l : ℝ) : 
  (l = 5 ∨ l = 6 ∨ l = 11 ∨ l = 20) → 
  (can_form_triangle 6 12 l ↔ l = 11) :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l3848_384801


namespace NUMINAMATH_CALUDE_katherine_age_when_mel_21_l3848_384832

/-- Katherine's age when Mel is 21 years old -/
def katherines_age (mels_age : ℕ) (age_difference : ℕ) : ℕ :=
  mels_age + age_difference

/-- Theorem stating Katherine's age when Mel is 21 -/
theorem katherine_age_when_mel_21 :
  katherines_age 21 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_katherine_age_when_mel_21_l3848_384832


namespace NUMINAMATH_CALUDE_no_real_a_for_unique_solution_l3848_384821

theorem no_real_a_for_unique_solution : ¬∃ a : ℝ, ∃! x : ℝ, |x^2 + 4*a*x + 5*a| ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_a_for_unique_solution_l3848_384821


namespace NUMINAMATH_CALUDE_side_c_length_l3848_384806

/-- Triangle ABC with given angles and side length -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- The Law of Sines for a triangle -/
axiom law_of_sines (t : Triangle) : t.a / Real.sin t.A = t.b / Real.sin t.B

/-- Theorem: In a triangle ABC with A = 30°, C = 45°, and a = 4, the length of side c is 4√2 -/
theorem side_c_length (t : Triangle) 
  (h1 : t.A = π/6)  -- 30° in radians
  (h2 : t.C = π/4)  -- 45° in radians
  (h3 : t.a = 4) :
  t.c = 4 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_side_c_length_l3848_384806


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_range_l3848_384807

theorem geometric_sequence_sum_range (m : ℝ) (hm : m > 0) :
  ∃ (a b c : ℝ), (a ≠ 0 ∧ b / a = c / b) ∧ (a + b + c = m) →
  b ∈ Set.Icc (-m) 0 ∪ Set.Ioc 0 (m / 3) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_range_l3848_384807


namespace NUMINAMATH_CALUDE_average_book_price_l3848_384871

/-- The average price of books bought by Rahim -/
theorem average_book_price (books1 books2 : ℕ) (price1 price2 : ℚ) 
  (h1 : books1 = 42)
  (h2 : books2 = 22)
  (h3 : price1 = 520)
  (h4 : price2 = 248) :
  (price1 + price2) / (books1 + books2 : ℚ) = 12 := by
  sorry

#check average_book_price

end NUMINAMATH_CALUDE_average_book_price_l3848_384871


namespace NUMINAMATH_CALUDE_halfway_fraction_l3848_384893

theorem halfway_fraction (a b c d : ℚ) (h1 : a = 2/3) (h2 : b = 4/5) (h3 : c = (a + b) / 2) : c = 11/15 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l3848_384893


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3848_384820

theorem algebraic_expression_value (x y : ℝ) 
  (hx : x = 1 / (Real.sqrt 3 - 2))
  (hy : y = 1 / (Real.sqrt 3 + 2)) :
  (x^2 + x*y + y^2) / (13 * (x + y)) = -(Real.sqrt 3) / 6 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3848_384820


namespace NUMINAMATH_CALUDE_min_value_x2_plus_y2_l3848_384864

theorem min_value_x2_plus_y2 (x y : ℝ) (h : x^2 + 2 * Real.sqrt 3 * x * y - y^2 = 1) :
  ∃ (m : ℝ), m = (1 : ℝ) / 2 ∧ ∀ (a b : ℝ), a^2 + 2 * Real.sqrt 3 * a * b - b^2 = 1 → a^2 + b^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_x2_plus_y2_l3848_384864


namespace NUMINAMATH_CALUDE_internet_charge_proof_l3848_384868

/-- The daily charge for internet service -/
def daily_charge : ℚ := 48/100

/-- The initial balance -/
def initial_balance : ℚ := 0

/-- The payment made -/
def payment : ℚ := 7

/-- The number of days of service -/
def service_days : ℕ := 25

/-- The debt threshold for service discontinuation -/
def debt_threshold : ℚ := 5

theorem internet_charge_proof :
  (initial_balance + payment - service_days * daily_charge = -debt_threshold) ∧
  (∀ x : ℚ, x > daily_charge → initial_balance + payment - service_days * x < -debt_threshold) :=
sorry

end NUMINAMATH_CALUDE_internet_charge_proof_l3848_384868


namespace NUMINAMATH_CALUDE_machine_A_production_rate_l3848_384894

-- Define the production rates and times for machines A, P, and Q
variable (A : ℝ) -- Production rate of Machine A (sprockets per hour)
variable (P : ℝ) -- Production rate of Machine P (sprockets per hour)
variable (Q : ℝ) -- Production rate of Machine Q (sprockets per hour)
variable (T_Q : ℝ) -- Time taken by Machine Q to produce 440 sprockets

-- State the conditions
axiom total_sprockets : 440 = Q * T_Q
axiom time_difference : 440 = P * (T_Q + 10)
axiom production_ratio : Q = 1.1 * A

-- State the theorem to be proved
theorem machine_A_production_rate : A = 4 := by
  sorry

end NUMINAMATH_CALUDE_machine_A_production_rate_l3848_384894


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3848_384865

theorem quadratic_rewrite (a b c : ℤ) :
  (∀ x : ℝ, 16 * x^2 - 48 * x + 56 = (a * x + b)^2 + c) →
  a * b = -24 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3848_384865


namespace NUMINAMATH_CALUDE_limes_remaining_l3848_384858

/-- The number of limes Mike picked -/
def limes_picked : Real := 32.0

/-- The number of limes Alyssa ate -/
def limes_eaten : Real := 25.0

/-- The number of limes left -/
def limes_left : Real := limes_picked - limes_eaten

theorem limes_remaining : limes_left = 7.0 := by
  sorry

end NUMINAMATH_CALUDE_limes_remaining_l3848_384858


namespace NUMINAMATH_CALUDE_coloring_properties_l3848_384811

/-- A coloring of natural numbers with N colors. -/
def Coloring (N : ℕ) := ℕ → Fin N

/-- Property that there are infinitely many numbers of each color. -/
def InfinitelyMany (c : Coloring N) : Prop :=
  ∀ (k : Fin N), ∀ (m : ℕ), ∃ (n : ℕ), n > m ∧ c n = k

/-- Property that the color of the half-sum of two different numbers of the same parity
    depends only on the colors of the summands. -/
def HalfSumProperty (c : Coloring N) : Prop :=
  ∀ (a b x y : ℕ), a ≠ b → x ≠ y → a % 2 = b % 2 → x % 2 = y % 2 →
    c a = c x → c b = c y → c ((a + b) / 2) = c ((x + y) / 2)

/-- Main theorem about the properties of the coloring. -/
theorem coloring_properties (N : ℕ) (c : Coloring N)
    (h1 : InfinitelyMany c) (h2 : HalfSumProperty c) :
  (∀ (a b : ℕ), a % 2 = b % 2 → c a = c b → c ((a + b) / 2) = c a) ∧
  (∃ (coloring : Coloring N), InfinitelyMany coloring ∧ HalfSumProperty coloring ↔ N % 2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_coloring_properties_l3848_384811


namespace NUMINAMATH_CALUDE_perfect_pink_paint_ratio_l3848_384869

/-- The ratio of white paint to red paint in perfect pink paint is 1:1 -/
theorem perfect_pink_paint_ratio :
  ∀ (total_paint red_paint white_paint : ℚ),
  total_paint = 30 →
  red_paint = 15 →
  total_paint = red_paint + white_paint →
  white_paint / red_paint = 1 := by
sorry

end NUMINAMATH_CALUDE_perfect_pink_paint_ratio_l3848_384869


namespace NUMINAMATH_CALUDE_fixed_point_parabola_l3848_384888

theorem fixed_point_parabola (m : ℝ) : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + m * x + 3 * m
  f (-3) = 45 := by sorry

end NUMINAMATH_CALUDE_fixed_point_parabola_l3848_384888


namespace NUMINAMATH_CALUDE_black_stones_count_l3848_384862

theorem black_stones_count (total : ℕ) (difference : ℕ) (black : ℕ) : 
  total = 950 →
  difference = 150 →
  total = black + (black + difference) →
  black = 400 :=
by sorry

end NUMINAMATH_CALUDE_black_stones_count_l3848_384862


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3848_384828

/-- Given a geometric sequence {a_n} with positive common ratio q,
    if a_3 · a_9 = (a_5)^2, then q = 1. -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  q > 0 →
  (∀ n, a (n + 1) = a n * q) →
  a 3 * a 9 = (a 5)^2 →
  q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3848_384828


namespace NUMINAMATH_CALUDE_four_letter_words_with_a_l3848_384827

theorem four_letter_words_with_a (n : ℕ) (total_letters : ℕ) (letters_without_a : ℕ) : 
  n = 4 → 
  total_letters = 5 → 
  letters_without_a = 4 → 
  (total_letters ^ n) - (letters_without_a ^ n) = 369 :=
by sorry

end NUMINAMATH_CALUDE_four_letter_words_with_a_l3848_384827


namespace NUMINAMATH_CALUDE_iron_weight_is_11_16_l3848_384812

/-- The weight of the piece of aluminum in pounds -/
def aluminum_weight : ℝ := 0.83

/-- The difference in weight between the piece of iron and the piece of aluminum in pounds -/
def weight_difference : ℝ := 10.33

/-- The weight of the piece of iron in pounds -/
def iron_weight : ℝ := aluminum_weight + weight_difference

/-- Theorem stating that the weight of the piece of iron is 11.16 pounds -/
theorem iron_weight_is_11_16 : iron_weight = 11.16 := by sorry

end NUMINAMATH_CALUDE_iron_weight_is_11_16_l3848_384812


namespace NUMINAMATH_CALUDE_intersection_locus_l3848_384835

/-- The locus of intersection points of two lines passing through fixed points on the x-axis and intersecting a parabola at four concyclic points. -/
theorem intersection_locus (a b : ℝ) (h : 0 < a ∧ a < b) :
  ∀ (l m : ℝ → ℝ → Prop) (P : ℝ × ℝ),
    (∀ y, l a y ↔ y = 0) →  -- Line l passes through (a, 0)
    (∀ y, m b y ↔ y = 0) →  -- Line m passes through (b, 0)
    (∃ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄,  -- Four distinct intersection points
      x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄ ∧
      l x₁ y₁ ∧ l x₂ y₂ ∧ m x₃ y₃ ∧ m x₄ y₄ ∧
      y₁^2 = x₁ ∧ y₂^2 = x₂ ∧ y₃^2 = x₃ ∧ y₄^2 = x₄ ∧
      ∃ (r : ℝ) (c : ℝ × ℝ), -- Points are concyclic
        (x₁ - c.1)^2 + (y₁ - c.2)^2 = r^2 ∧
        (x₂ - c.1)^2 + (y₂ - c.2)^2 = r^2 ∧
        (x₃ - c.1)^2 + (y₃ - c.2)^2 = r^2 ∧
        (x₄ - c.1)^2 + (y₄ - c.2)^2 = r^2) →
    (∀ x y, l x y ∧ m x y → P = (x, y)) →  -- P is the intersection of l and m
    P.1 = (a + b) / 2  -- The x-coordinate of P satisfies 2x - (a + b) = 0
  := by sorry

end NUMINAMATH_CALUDE_intersection_locus_l3848_384835


namespace NUMINAMATH_CALUDE_existence_of_solution_l3848_384838

theorem existence_of_solution : ∃ (a b c d : ℕ+), 
  (a^3 + b^4 + c^5 = d^11) ∧ (a * b * c < 10^5) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l3848_384838


namespace NUMINAMATH_CALUDE_line_symmetry_l3848_384817

-- Define the lines
def line_l (x y : ℝ) : Prop := 3 * x - y + 3 = 0
def line_1 (x y : ℝ) : Prop := x - y - 2 = 0
def line_2 (x y : ℝ) : Prop := 7 * x + y + 22 = 0

-- Define symmetry with respect to line_l
def symmetric_wrt_l (x y x' y' : ℝ) : Prop :=
  -- The product of the slopes of PP' and line_l is -1
  ((y' - y) / (x' - x)) * 3 = -1 ∧
  -- The midpoint of PP' lies on line_l
  3 * ((x + x') / 2) - ((y + y') / 2) + 3 = 0

-- Theorem statement
theorem line_symmetry :
  ∀ x y x' y' : ℝ,
    line_1 x y ∧ line_2 x' y' →
    symmetric_wrt_l x y x' y' :=
sorry

end NUMINAMATH_CALUDE_line_symmetry_l3848_384817


namespace NUMINAMATH_CALUDE_other_factor_of_prime_multiple_l3848_384880

theorem other_factor_of_prime_multiple (p n : ℕ) : 
  Nat.Prime p → 
  (∃ k, n = k * p) → 
  (∀ d : ℕ, d ∣ n ↔ d = 1 ∨ d = n) → 
  ∃ k : ℕ, n = k * p ∧ k = 1 :=
by sorry

end NUMINAMATH_CALUDE_other_factor_of_prime_multiple_l3848_384880


namespace NUMINAMATH_CALUDE_copper_button_percentage_is_28_percent_l3848_384833

/-- Represents the composition of items in a basket --/
structure BasketComposition where
  pin_percentage : ℝ
  brass_button_percentage : ℝ
  copper_button_percentage : ℝ

/-- The percentage of copper buttons in the basket --/
def copper_button_percentage (b : BasketComposition) : ℝ :=
  b.copper_button_percentage

/-- Theorem stating the percentage of copper buttons in the basket --/
theorem copper_button_percentage_is_28_percent 
  (b : BasketComposition)
  (h1 : b.pin_percentage = 0.3)
  (h2 : b.brass_button_percentage = 0.42)
  (h3 : b.pin_percentage + b.brass_button_percentage + b.copper_button_percentage = 1) :
  copper_button_percentage b = 0.28 := by
  sorry

#check copper_button_percentage_is_28_percent

end NUMINAMATH_CALUDE_copper_button_percentage_is_28_percent_l3848_384833


namespace NUMINAMATH_CALUDE_sum_of_a_values_l3848_384876

theorem sum_of_a_values : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, (∃! (sol : Finset ℤ), 
    (∀ x ∈ sol, (4 * x - a ≥ 1 ∧ (x + 13) / 2 ≥ x + 2)) ∧ 
    sol.card = 6)) ∧ 
  S.sum id = 54 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_values_l3848_384876


namespace NUMINAMATH_CALUDE_parabola_directrix_l3848_384856

/-- The parabola equation -/
def parabola_equation (x y : ℝ) : Prop := x = -(1/4) * y^2

/-- The directrix equation -/
def directrix_equation (x : ℝ) : Prop := x = 1

/-- Theorem: The directrix of the parabola x = -1/4 * y^2 is x = 1 -/
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola_equation x y → 
  ∃ (d : ℝ), directrix_equation d ∧
  ∀ (p q : ℝ × ℝ), 
    parabola_equation p.1 p.2 →
    (p.1 - d)^2 = (p.1 - q.1)^2 + (p.2 - q.2)^2 →
    q.1 = -1 ∧ q.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3848_384856


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l3848_384850

/-- Calculates the measured weight loss percentage at the final weigh-in -/
def measuredWeightLoss (initialLoss : ℝ) (clothesWeight : ℝ) (waterRetention : ℝ) : ℝ :=
  (1 - (1 - initialLoss) * (1 + clothesWeight) * (1 + waterRetention)) * 100

/-- Theorem stating the measured weight loss percentage for given conditions -/
theorem weight_loss_challenge (initialLoss clothesWeight waterRetention : ℝ) 
  (h1 : initialLoss = 0.11)
  (h2 : clothesWeight = 0.02)
  (h3 : waterRetention = 0.015) :
  abs (measuredWeightLoss initialLoss clothesWeight waterRetention - 7.64) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l3848_384850


namespace NUMINAMATH_CALUDE_horsemen_speeds_exist_l3848_384882

/-- Represents a set of speeds for horsemen on a circular track -/
def SpeedSet (n : ℕ) := Fin n → ℝ

/-- Predicate that checks if all speeds in a set are distinct and positive -/
def distinct_positive (s : SpeedSet n) : Prop :=
  ∀ i j, i ≠ j → s i ≠ s j ∧ s i > 0 ∧ s j > 0

/-- Predicate that checks if all overtakings occur at a single point -/
def single_overtaking_point (s : SpeedSet n) : Prop :=
  ∀ i j, i ≠ j → ∃ k : ℤ, (s i) / (s i - s j) = k

/-- Theorem stating that for any number of horsemen (≥ 3), 
    there exists a set of speeds satisfying the required conditions -/
theorem horsemen_speeds_exist (n : ℕ) (h : n ≥ 3) :
  ∃ (s : SpeedSet n), distinct_positive s ∧ single_overtaking_point s :=
sorry

end NUMINAMATH_CALUDE_horsemen_speeds_exist_l3848_384882


namespace NUMINAMATH_CALUDE_average_of_a_and_b_l3848_384861

theorem average_of_a_and_b (a b : ℝ) : 
  (4 + 5 + 9 + a + b) / 5 = 18 → (a + b) / 2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_average_of_a_and_b_l3848_384861


namespace NUMINAMATH_CALUDE_jimmy_garden_servings_l3848_384878

/-- Represents the number of servings produced by a single plant of each vegetable type -/
structure ServingsPerPlant where
  carrot : ℕ
  corn : ℕ
  greenBean : ℕ
  tomato : ℕ
  zucchini : ℕ
  bellPepper : ℕ

/-- Represents the number of plants for each vegetable type in Jimmy's garden -/
structure PlantsPerPlot where
  carrot : ℕ
  corn : ℕ
  greenBean : ℕ
  tomato : ℕ
  zucchini : ℕ
  bellPepper : ℕ

/-- Calculates the total number of servings in Jimmy's garden -/
def totalServings (s : ServingsPerPlant) (p : PlantsPerPlot) : ℕ :=
  s.carrot * p.carrot +
  s.corn * p.corn +
  s.greenBean * p.greenBean +
  s.tomato * p.tomato +
  s.zucchini * p.zucchini +
  s.bellPepper * p.bellPepper

/-- Theorem stating that Jimmy's garden produces 963 servings of vegetables -/
theorem jimmy_garden_servings 
  (s : ServingsPerPlant)
  (p : PlantsPerPlot)
  (h1 : s.carrot = 4)
  (h2 : s.corn = 5 * s.carrot)
  (h3 : s.greenBean = s.corn / 2)
  (h4 : s.tomato = s.carrot + 3)
  (h5 : s.zucchini = 4 * s.greenBean)
  (h6 : s.bellPepper = s.corn - 2)
  (h7 : p.greenBean = 10)
  (h8 : p.carrot = 8)
  (h9 : p.corn = 12)
  (h10 : p.tomato = 15)
  (h11 : p.zucchini = 9)
  (h12 : p.bellPepper = 7) :
  totalServings s p = 963 := by
  sorry


end NUMINAMATH_CALUDE_jimmy_garden_servings_l3848_384878


namespace NUMINAMATH_CALUDE_simplify_expression_l3848_384867

theorem simplify_expression (x : ℝ) : (3*x + 20) + (200*x + 45) = 203*x + 65 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3848_384867


namespace NUMINAMATH_CALUDE_m_divided_by_8_l3848_384802

theorem m_divided_by_8 (m : ℕ) (h : m = 16^500) : m / 8 = 2^1997 := by
  sorry

end NUMINAMATH_CALUDE_m_divided_by_8_l3848_384802


namespace NUMINAMATH_CALUDE_money_division_l3848_384809

theorem money_division (a b c : ℚ) : 
  a = (1/3 : ℚ) * b → 
  b = (1/4 : ℚ) * c → 
  b = 270 → 
  a + b + c = 1440 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l3848_384809


namespace NUMINAMATH_CALUDE_cube_root_of_27_l3848_384823

theorem cube_root_of_27 : 
  {z : ℂ | z^3 = 27} = {3, (-3 + 3*Complex.I*Real.sqrt 3)/2, (-3 - 3*Complex.I*Real.sqrt 3)/2} := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_27_l3848_384823


namespace NUMINAMATH_CALUDE_purely_imaginary_fraction_l3848_384840

theorem purely_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + 2 * Complex.I) / (1 + Complex.I) = b * Complex.I) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_fraction_l3848_384840


namespace NUMINAMATH_CALUDE_sum_mod_nine_l3848_384853

theorem sum_mod_nine : (7155 + 7156 + 7157 + 7158 + 7159) % 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_nine_l3848_384853


namespace NUMINAMATH_CALUDE_smallest_multiple_of_36_and_45_not_25_l3848_384849

theorem smallest_multiple_of_36_and_45_not_25 :
  ∃ (n : ℕ), n > 0 ∧ 36 ∣ n ∧ 45 ∣ n ∧ ¬(25 ∣ n) ∧
  ∀ (m : ℕ), m > 0 → 36 ∣ m → 45 ∣ m → ¬(25 ∣ m) → n ≤ m :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_36_and_45_not_25_l3848_384849


namespace NUMINAMATH_CALUDE_mittens_count_l3848_384816

theorem mittens_count (original_plugs current_plugs mittens : ℕ) : 
  mittens = original_plugs - 20 →
  current_plugs = original_plugs + 30 →
  400 = 2 * current_plugs →
  mittens = 150 := by
  sorry

end NUMINAMATH_CALUDE_mittens_count_l3848_384816


namespace NUMINAMATH_CALUDE_solid_price_is_four_l3848_384887

/-- The price of solid color gift wrap per roll -/
def solid_price : ℝ := 4

/-- The total number of rolls sold -/
def total_rolls : ℕ := 480

/-- The total amount of money collected in dollars -/
def total_money : ℝ := 2340

/-- The number of print rolls sold -/
def print_rolls : ℕ := 210

/-- The price of print gift wrap per roll in dollars -/
def print_price : ℝ := 6

/-- Theorem stating that the price of solid color gift wrap is $4.00 per roll -/
theorem solid_price_is_four :
  solid_price = (total_money - print_rolls * print_price) / (total_rolls - print_rolls) :=
by sorry

end NUMINAMATH_CALUDE_solid_price_is_four_l3848_384887


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l3848_384824

theorem quadratic_root_relation (p q : ℝ) :
  (∃ α : ℝ, (α^2 + p*α + q = 0) ∧ ((2*α)^2 + p*(2*α) + q = 0)) →
  2*p^2 = 9*q :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l3848_384824


namespace NUMINAMATH_CALUDE_craig_final_apples_l3848_384837

def craig_initial_apples : ℕ := 20
def shared_apples : ℕ := 7

theorem craig_final_apples :
  craig_initial_apples - shared_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_craig_final_apples_l3848_384837


namespace NUMINAMATH_CALUDE_probability_all_white_drawn_l3848_384891

def total_balls : ℕ := 15
def white_balls : ℕ := 8
def black_balls : ℕ := 7
def drawn_balls : ℕ := 6

def probability_all_white : ℚ := 4 / 715

theorem probability_all_white_drawn (total : ℕ) (white : ℕ) (black : ℕ) (drawn : ℕ) :
  total = white + black →
  white ≥ drawn →
  probability_all_white = (Nat.choose white drawn : ℚ) / (Nat.choose total drawn : ℚ) :=
sorry

end NUMINAMATH_CALUDE_probability_all_white_drawn_l3848_384891


namespace NUMINAMATH_CALUDE_complex_real_part_l3848_384841

theorem complex_real_part (z : ℂ) (h : (z^2 + z).im = 0) : z.re = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_real_part_l3848_384841


namespace NUMINAMATH_CALUDE_james_course_cost_l3848_384875

/-- Represents the cost per unit for James's community college courses. -/
def cost_per_unit (units_per_semester : ℕ) (total_cost : ℕ) (num_semesters : ℕ) : ℚ :=
  total_cost / (units_per_semester * num_semesters)

/-- Theorem stating that the cost per unit is $50 given the conditions. -/
theorem james_course_cost : 
  cost_per_unit 20 2000 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_james_course_cost_l3848_384875


namespace NUMINAMATH_CALUDE_q_satisfies_conditions_q_unique_l3848_384800

/-- The cubic polynomial q(x) that satisfies the given conditions -/
def q (x : ℝ) : ℝ := -4 * x^3 + 24 * x^2 - 44 * x + 24

/-- Theorem stating that q(x) satisfies the required conditions -/
theorem q_satisfies_conditions :
  q 1 = 0 ∧ q 2 = 0 ∧ q 3 = 0 ∧ q 4 = -24 := by
  sorry

/-- Theorem stating that q(x) is the unique cubic polynomial satisfying the conditions -/
theorem q_unique (p : ℝ → ℝ) (h_cubic : ∃ a b c d, ∀ x, p x = a * x^3 + b * x^2 + c * x + d) 
  (h_cond : p 1 = 0 ∧ p 2 = 0 ∧ p 3 = 0 ∧ p 4 = -24) :
  ∀ x, p x = q x := by
  sorry

end NUMINAMATH_CALUDE_q_satisfies_conditions_q_unique_l3848_384800


namespace NUMINAMATH_CALUDE_rachel_age_l3848_384863

/-- Given that Rachel is 4 years older than Leah and the sum of their ages is 34,
    prove that Rachel is 19 years old. -/
theorem rachel_age (rachel_age leah_age : ℕ) 
  (h1 : rachel_age = leah_age + 4)
  (h2 : rachel_age + leah_age = 34) : 
  rachel_age = 19 := by
  sorry

end NUMINAMATH_CALUDE_rachel_age_l3848_384863


namespace NUMINAMATH_CALUDE_round_trip_time_l3848_384884

/-- Calculates the total time for a round trip boat journey given the boat's speed in standing water,
    the stream's speed, and the distance to travel. -/
theorem round_trip_time (boat_speed : ℝ) (stream_speed : ℝ) (distance : ℝ) :
  boat_speed = 9 →
  stream_speed = 6 →
  distance = 210 →
  (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = 84 :=
by sorry

end NUMINAMATH_CALUDE_round_trip_time_l3848_384884


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3848_384842

/-- Given a right circular cylinder of radius 2 intersected by a plane forming an ellipse,
    if the major axis is 20% longer than the minor axis, then the length of the major axis is 4.8. -/
theorem ellipse_major_axis_length (cylinder_radius : ℝ) (minor_axis : ℝ) (major_axis : ℝ) : 
  cylinder_radius = 2 →
  minor_axis = 2 * cylinder_radius →
  major_axis = minor_axis * 1.2 →
  major_axis = 4.8 := by
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3848_384842


namespace NUMINAMATH_CALUDE_remainder_sum_l3848_384870

theorem remainder_sum (a b : ℤ) (ha : a % 60 = 41) (hb : b % 45 = 14) : (a + b) % 15 = 10 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_l3848_384870


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3848_384877

def num_books : ℕ := 10
def num_calculus : ℕ := 3
def num_algebra : ℕ := 4
def num_statistics : ℕ := 3

theorem book_arrangement_count :
  (num_calculus.factorial * num_statistics.factorial * (num_books - num_algebra).factorial) = 25920 :=
sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3848_384877


namespace NUMINAMATH_CALUDE_first_player_wins_6x8_l3848_384883

/-- Represents a chocolate bar game --/
structure ChocolateGame where
  rows : Nat
  cols : Nat

/-- Calculates the number of moves required to completely break the chocolate bar --/
def totalMoves (game : ChocolateGame) : Nat :=
  game.rows * game.cols - 1

/-- Determines the winner of the game --/
def firstPlayerWins (game : ChocolateGame) : Prop :=
  Odd (totalMoves game)

/-- Theorem stating that the first player wins in a 6x8 chocolate bar game --/
theorem first_player_wins_6x8 :
  firstPlayerWins { rows := 6, cols := 8 } := by
  sorry

end NUMINAMATH_CALUDE_first_player_wins_6x8_l3848_384883


namespace NUMINAMATH_CALUDE_battery_problem_l3848_384892

theorem battery_problem :
  ∀ (x y z : ℚ),
  (x > 0) → (y > 0) → (z > 0) →
  (4*x + 18*y + 16*z = 4*x + 15*y + 24*z) →
  (4*x + 18*y + 16*z = 6*x + 12*y + 20*z) →
  (∃ (W : ℚ), W * z = 4*x + 18*y + 16*z ∧ W = 48) :=
by
  sorry

end NUMINAMATH_CALUDE_battery_problem_l3848_384892


namespace NUMINAMATH_CALUDE_function_range_condition_l3848_384804

open Real

/-- Given a function f(x) = ax - ln x - 1, prove that there exists x₀ ∈ (0,e] 
    such that f(x₀) < 0 if and only if a ∈ (-∞, 1). -/
theorem function_range_condition (a : ℝ) : 
  (∃ x₀ : ℝ, 0 < x₀ ∧ x₀ ≤ ℯ ∧ a * x₀ - log x₀ - 1 < 0) ↔ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_function_range_condition_l3848_384804


namespace NUMINAMATH_CALUDE_no_valid_arrangement_l3848_384815

def is_valid_arrangement (perm : List Nat) : Prop :=
  perm.length = 8 ∧
  (∀ n, n ∈ perm → n ∈ [1, 2, 3, 4, 5, 6, 8, 9]) ∧
  (∀ i, i < 7 → (10 * perm[i]! + perm[i+1]!) % 7 = 0)

theorem no_valid_arrangement : ¬∃ perm : List Nat, is_valid_arrangement perm := by
  sorry

end NUMINAMATH_CALUDE_no_valid_arrangement_l3848_384815


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3848_384830

/-- The acute angle formed by the asymptotes of a hyperbola with eccentricity 2 is 60°. -/
theorem hyperbola_asymptote_angle (e : ℝ) (h : e = 2) :
  let a : ℝ := 1  -- Arbitrary choice for a, as the angle is independent of a's value
  let b : ℝ := Real.sqrt 3 * a
  let asymptote_angle : ℝ := 2 * Real.arctan (b / a)
  asymptote_angle = π / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l3848_384830


namespace NUMINAMATH_CALUDE_expression_value_l3848_384845

theorem expression_value (m n : ℝ) (h : m + 2*n = 1) : 3*m^2 + 6*m*n + 6*n = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3848_384845


namespace NUMINAMATH_CALUDE_carl_first_six_probability_l3848_384826

/-- The probability of rolling a 6 on a single die roll -/
def prob_six : ℚ := 1 / 6

/-- The probability of not rolling a 6 on a single die roll -/
def prob_not_six : ℚ := 1 - prob_six

/-- The sequence of probabilities for Carl rolling the first 6 on his nth turn -/
def carl_first_six (n : ℕ) : ℚ := (prob_not_six ^ (3 * n - 1)) * prob_six

/-- The sum of the geometric series representing the probability of Carl rolling the first 6 -/
def probability_carl_first_six : ℚ := (carl_first_six 1) / (1 - (prob_not_six ^ 3))

theorem carl_first_six_probability :
  probability_carl_first_six = 25 / 91 :=
sorry

end NUMINAMATH_CALUDE_carl_first_six_probability_l3848_384826


namespace NUMINAMATH_CALUDE_power_expression_simplification_l3848_384860

theorem power_expression_simplification :
  (1 : ℚ) / ((-8^2)^4) * (-8)^9 = -8 := by sorry

end NUMINAMATH_CALUDE_power_expression_simplification_l3848_384860


namespace NUMINAMATH_CALUDE_square_of_98_l3848_384854

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by sorry

end NUMINAMATH_CALUDE_square_of_98_l3848_384854


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l3848_384851

theorem smallest_x_for_perfect_cube : ∃ (x : ℕ+), 
  (∀ (y : ℕ+), ∃ (M : ℤ), 1800 * y = M^3 → x ≤ y) ∧
  (∃ (M : ℤ), 1800 * x = M^3) ∧
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l3848_384851


namespace NUMINAMATH_CALUDE_wendy_shoes_theorem_l3848_384889

/-- The number of pairs of shoes Wendy gave away -/
def shoes_given_away : ℕ := 14

/-- The number of pairs of shoes Wendy kept -/
def shoes_kept : ℕ := 19

/-- The total number of pairs of shoes Wendy had -/
def total_shoes : ℕ := shoes_given_away + shoes_kept

theorem wendy_shoes_theorem : total_shoes = 33 := by
  sorry

end NUMINAMATH_CALUDE_wendy_shoes_theorem_l3848_384889


namespace NUMINAMATH_CALUDE_two_digit_number_rounded_to_3_8_l3848_384872

/-- A number that rounds to 3.8 when rounded to one decimal place -/
def RoundsTo3_8 (n : ℝ) : Prop := 3.75 ≤ n ∧ n < 3.85

/-- The set of two-digit numbers -/
def TwoDigitNumber (n : ℝ) : Prop := 10 ≤ n ∧ n < 100

theorem two_digit_number_rounded_to_3_8 :
  ∃ (max min : ℝ), 
    (∀ n : ℝ, TwoDigitNumber n → RoundsTo3_8 n → n ≤ max) ∧
    (∀ n : ℝ, TwoDigitNumber n → RoundsTo3_8 n → min ≤ n) ∧
    max = 3.84 ∧ min = 3.75 :=
sorry

end NUMINAMATH_CALUDE_two_digit_number_rounded_to_3_8_l3848_384872


namespace NUMINAMATH_CALUDE_molecular_weight_of_Y_l3848_384881

/-- Represents a chemical compound with its molecular weight -/
structure Compound where
  molecularWeight : ℝ

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactant1 : Compound
  reactant2 : Compound
  product : Compound
  reactant2Coefficient : ℕ

/-- The law of conservation of mass in a chemical reaction -/
def conservationOfMass (r : Reaction) : Prop :=
  r.reactant1.molecularWeight + r.reactant2Coefficient * r.reactant2.molecularWeight = r.product.molecularWeight

/-- Theorem: The molecular weight of Y in the given reaction -/
theorem molecular_weight_of_Y : 
  let X : Compound := ⟨136⟩
  let C6H8O7 : Compound := ⟨192⟩
  let Y : Compound := ⟨1096⟩
  let reaction : Reaction := ⟨X, C6H8O7, Y, 5⟩
  conservationOfMass reaction := by
  sorry

end NUMINAMATH_CALUDE_molecular_weight_of_Y_l3848_384881


namespace NUMINAMATH_CALUDE_calculate_biology_marks_l3848_384890

theorem calculate_biology_marks (english math physics chemistry : ℕ) (average : ℚ) :
  english = 96 →
  math = 95 →
  physics = 82 →
  chemistry = 87 →
  average = 90.4 →
  (english + math + physics + chemistry + (5 * average - (english + math + physics + chemistry))) / 5 = average :=
by sorry

end NUMINAMATH_CALUDE_calculate_biology_marks_l3848_384890


namespace NUMINAMATH_CALUDE_remainder_theorem_l3848_384825

/-- A polynomial of the form Ax^6 + Bx^4 + Cx^2 + 5 -/
def p (A B C : ℝ) (x : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 5

/-- The remainder when p(x) is divided by x-2 is 13 -/
def remainder_condition (A B C : ℝ) : Prop := p A B C 2 = 13

theorem remainder_theorem (A B C : ℝ) (h : remainder_condition A B C) :
  p A B C (-2) = 13 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3848_384825


namespace NUMINAMATH_CALUDE_largest_divisor_of_expression_l3848_384857

theorem largest_divisor_of_expression (x : ℤ) (h : Even x) :
  (∃ (k : ℤ), (10*x + 4) * (10*x + 8) * (5*x + 2) = 32 * k) ∧
  (∀ (m : ℤ), m > 32 → ∃ (y : ℤ), Even y ∧ ¬(∃ (l : ℤ), (10*y + 4) * (10*y + 8) * (5*y + 2) = m * l)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_expression_l3848_384857


namespace NUMINAMATH_CALUDE_count_dinosaur_dolls_l3848_384899

def dinosaur_dolls : ℕ := 3
def fish_dolls : ℕ := 2
def toy_cars : ℕ := 1

theorem count_dinosaur_dolls : dinosaur_dolls = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_dinosaur_dolls_l3848_384899


namespace NUMINAMATH_CALUDE_slope_is_constant_l3848_384810

-- Define the ellipse C
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define a point in the first quadrant
def in_first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the line l
def line (k m x y : ℝ) : Prop := y = k * x + m

-- Define the condition for the areas
def area_condition (x₁ y₁ x₂ y₂ m k : ℝ) : Prop :=
  (y₁^2 + y₂^2) / (y₁ * y₂) = (x₁^2 + x₂^2) / (x₁ * x₂)

-- Main theorem
theorem slope_is_constant
  (k m x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : ellipse x₁ y₁)
  (h₂ : ellipse x₂ y₂)
  (h₃ : in_first_quadrant x₁ y₁)
  (h₄ : in_first_quadrant x₂ y₂)
  (h₅ : line k m x₁ y₁)
  (h₆ : line k m x₂ y₂)
  (h₇ : m ≠ 0)
  (h₈ : area_condition x₁ y₁ x₂ y₂ m k) :
  k = -1/2 := by sorry

end NUMINAMATH_CALUDE_slope_is_constant_l3848_384810


namespace NUMINAMATH_CALUDE_asian_games_survey_l3848_384848

theorem asian_games_survey (total students : ℕ) 
  (table_tennis badminton not_interested : ℕ) : 
  total = 50 → 
  table_tennis = 35 → 
  badminton = 30 → 
  not_interested = 5 → 
  table_tennis + badminton - (total - not_interested) = 20 := by
  sorry

end NUMINAMATH_CALUDE_asian_games_survey_l3848_384848


namespace NUMINAMATH_CALUDE_unique_triple_existence_l3848_384839

theorem unique_triple_existence : 
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
  (1 / a = b + c) ∧ (1 / b = a + c) ∧ (1 / c = a + b) := by
sorry

end NUMINAMATH_CALUDE_unique_triple_existence_l3848_384839


namespace NUMINAMATH_CALUDE_bowling_ball_weight_l3848_384885

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (5 * bowling_ball_weight = 3 * canoe_weight) →
    (3 * canoe_weight = 105) →
    bowling_ball_weight = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_l3848_384885


namespace NUMINAMATH_CALUDE_line_equation_proof_l3848_384898

-- Define the given line
def given_line (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x - 2

-- Define the point through which the desired line passes
def point : ℝ × ℝ := (-1, 1)

-- Define the slope of the desired line
def desired_slope (m : ℝ) : Prop := m = 2 * (Real.sqrt 2 / 2)

-- Define the equation of the desired line
def desired_line (x y : ℝ) : Prop := y - 1 = Real.sqrt 2 * (x + 1)

-- Theorem statement
theorem line_equation_proof :
  ∀ (x y : ℝ),
  given_line x y →
  desired_slope (Real.sqrt 2) →
  desired_line point.1 point.2 →
  desired_line x y :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l3848_384898


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3848_384897

theorem at_least_one_greater_than_one (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) : 
  x > 1 ∨ y > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l3848_384897


namespace NUMINAMATH_CALUDE_parabola_intersects_x_axis_l3848_384847

/-- For a parabola y = x^2 + 2x + m - 1 to intersect with the x-axis, m must be less than or equal to 2 -/
theorem parabola_intersects_x_axis (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m - 1 = 0) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersects_x_axis_l3848_384847


namespace NUMINAMATH_CALUDE_smallest_three_digit_perfect_square_append_l3848_384866

theorem smallest_three_digit_perfect_square_append : ∃ (n : ℕ), 
  (n = 183) ∧ 
  (∀ m : ℕ, 100 ≤ m ∧ m < n → ¬(∃ k : ℕ, 1000 * m + (m + 1) = k^2)) ∧
  (∃ k : ℕ, 1000 * n + (n + 1) = k^2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_three_digit_perfect_square_append_l3848_384866


namespace NUMINAMATH_CALUDE_problem_1_l3848_384886

theorem problem_1 : (5 / 17) * (-4) - (5 / 17) * 15 + (-5 / 17) * (-2) = -5 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3848_384886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3848_384834

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a → sum_condition a → 2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3848_384834


namespace NUMINAMATH_CALUDE_projection_property_l3848_384836

def projection (v : ℝ × ℝ) (w : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_property :
  ∀ (p : (ℝ × ℝ) → (ℝ × ℝ)),
  (p (2, -4) = (3, -3)) →
  (p = projection (1, -1)) →
  (p (-8, 2) = (-5, 5)) := by sorry

end NUMINAMATH_CALUDE_projection_property_l3848_384836


namespace NUMINAMATH_CALUDE_unique_solution_modular_equation_l3848_384846

theorem unique_solution_modular_equation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 107 ∧ (103 * n) % 107 = 56 % 107 ∧ n = 85 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_modular_equation_l3848_384846


namespace NUMINAMATH_CALUDE_value_of_m_area_of_triangle_max_y_intercept_l3848_384844

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 - x - m^2 + 6*m - 7

-- Theorem 1: If the graph passes through A(-1, 2), then m = 5
theorem value_of_m (m : ℝ) : f m (-1) = 2 → m = 5 := by sorry

-- Theorem 2: If m = 5, the area of triangle ABC is 5/3
theorem area_of_triangle : 
  let m := 5
  let x1 := (- 2/3 : ℝ)  -- x-coordinate of point C
  let x2 := (1 : ℝ)      -- x-coordinate of point B
  (1/2 : ℝ) * |x2 - x1| * 2 = 5/3 := by sorry

-- Theorem 3: The maximum y-coordinate of the y-intercept is 2
theorem max_y_intercept : 
  ∃ (m : ℝ), ∀ (m' : ℝ), f m' 0 ≤ f m 0 ∧ f m 0 = 2 := by sorry

end NUMINAMATH_CALUDE_value_of_m_area_of_triangle_max_y_intercept_l3848_384844


namespace NUMINAMATH_CALUDE_total_money_is_250_l3848_384819

/-- The amount of money James owns -/
def james_money : ℕ := 145

/-- The difference between James' and Ali's money -/
def difference : ℕ := 40

/-- The amount of money Ali owns -/
def ali_money : ℕ := james_money - difference

/-- The total amount of money owned by James and Ali -/
def total_money : ℕ := james_money + ali_money

theorem total_money_is_250 : total_money = 250 := by sorry

end NUMINAMATH_CALUDE_total_money_is_250_l3848_384819


namespace NUMINAMATH_CALUDE_circle_M_properties_l3848_384852

-- Define the circle M
def circle_M (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 1 = 0

-- Define the line L
def line_L (x y : ℝ) : Prop := x + 3*y - 2 = 0

-- Theorem statement
theorem circle_M_properties :
  -- The radius of M is √5
  (∃ (h k r : ℝ), r = Real.sqrt 5 ∧ ∀ (x y : ℝ), circle_M x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧
  -- M is symmetric with respect to the line L
  (∃ (h k : ℝ), circle_M h k ∧ line_L h k ∧
    ∀ (x y : ℝ), circle_M x y → 
      ∃ (x' y' : ℝ), circle_M x' y' ∧ line_L ((x + x')/2) ((y + y')/2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_M_properties_l3848_384852


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l3848_384818

theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + (3 : ℚ) / 99 + (4 : ℚ) / 9999 = (843 : ℚ) / 3333 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l3848_384818


namespace NUMINAMATH_CALUDE_cube_surface_area_equals_prism_volume_l3848_384813

/-- The surface area of a cube with volume equal to a rectangular prism of dimensions 12 × 3 × 18 is equal to the volume of the prism. -/
theorem cube_surface_area_equals_prism_volume :
  let prism_length : ℝ := 12
  let prism_width : ℝ := 3
  let prism_height : ℝ := 18
  let prism_volume := prism_length * prism_width * prism_height
  let cube_edge := (prism_volume) ^ (1/3 : ℝ)
  let cube_surface_area := 6 * cube_edge ^ 2
  cube_surface_area = prism_volume := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_equals_prism_volume_l3848_384813


namespace NUMINAMATH_CALUDE_f_extrema_l3848_384895

noncomputable def f (x : ℝ) : ℝ := x + 2 * Real.cos x

theorem f_extrema :
  let a : ℝ := 0
  let b : ℝ := Real.pi / 2
  (∀ x ∈ Set.Icc a b, f x ≤ f (Real.pi / 6)) ∧
  (∀ x ∈ Set.Icc a b, f (Real.pi / 2) ≤ f x) := by
  sorry

#check f_extrema

end NUMINAMATH_CALUDE_f_extrema_l3848_384895


namespace NUMINAMATH_CALUDE_marks_change_factor_l3848_384855

theorem marks_change_factor (n : ℕ) (initial_avg final_avg : ℝ) (h1 : n = 10) (h2 : initial_avg = 40) (h3 : final_avg = 80) :
  ∃ (factor : ℝ), factor * (n * initial_avg) = n * final_avg ∧ factor = 2 := by
sorry

end NUMINAMATH_CALUDE_marks_change_factor_l3848_384855


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3848_384859

def U : Set ℕ := {1,2,3,4,5,6,7,8,9}
def A : Set ℕ := {2,4,5,7}
def B : Set ℕ := {3,4,5,6,8}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {3,6,8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3848_384859


namespace NUMINAMATH_CALUDE_complex_product_real_l3848_384829

theorem complex_product_real (b : ℝ) : 
  let z₁ : ℂ := 1 + I
  let z₂ : ℂ := 2 + b * I
  (z₁ * z₂).im = 0 → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_real_l3848_384829


namespace NUMINAMATH_CALUDE_h_not_prime_l3848_384808

def h (n : ℕ+) : ℤ := n^4 - 380 * n^2 + 600

theorem h_not_prime : ∀ n : ℕ+, ¬ Nat.Prime (Int.natAbs (h n)) := by
  sorry

end NUMINAMATH_CALUDE_h_not_prime_l3848_384808


namespace NUMINAMATH_CALUDE_common_difference_is_negative_three_l3848_384873

/-- An arithmetic sequence with given terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 7
  seventh_term : a 7 = -5

/-- The common difference of an arithmetic sequence -/
def common_difference (seq : ArithmeticSequence) : ℝ :=
  seq.a 2 - seq.a 1

theorem common_difference_is_negative_three (seq : ArithmeticSequence) :
  common_difference seq = -3 := by
  sorry

end NUMINAMATH_CALUDE_common_difference_is_negative_three_l3848_384873


namespace NUMINAMATH_CALUDE_union_and_subset_l3848_384843

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | 3/2 < x ∧ x < 4}
def P (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

theorem union_and_subset :
  (A ∪ B = {x | 1 < x ∧ x < 4}) ∧
  (∀ a : ℝ, P a ⊆ A ∪ B ↔ 1 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_union_and_subset_l3848_384843
