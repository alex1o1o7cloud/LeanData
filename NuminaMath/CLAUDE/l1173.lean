import Mathlib

namespace NUMINAMATH_CALUDE_subset_implies_a_range_l1173_117395

open Set Real

-- Define set A
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a+7)*x + 5 ≤ 0}

-- Theorem statement
theorem subset_implies_a_range (a : ℝ) :
  A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l1173_117395


namespace NUMINAMATH_CALUDE_total_art_pieces_l1173_117322

theorem total_art_pieces (asian : Nat) (egyptian : Nat) (european : Nat)
  (h1 : asian = 465)
  (h2 : egyptian = 527)
  (h3 : european = 320) :
  asian + egyptian + european = 1312 := by
  sorry

end NUMINAMATH_CALUDE_total_art_pieces_l1173_117322


namespace NUMINAMATH_CALUDE_bicycle_wheels_l1173_117300

theorem bicycle_wheels (num_bicycles num_tricycles tricycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 24)
  (h2 : num_tricycles = 14)
  (h3 : tricycle_wheels = 3)
  (h4 : total_wheels = 90)
  : ∃ bicycle_wheels : ℕ, 
    bicycle_wheels * num_bicycles + tricycle_wheels * num_tricycles = total_wheels ∧ 
    bicycle_wheels = 2 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_wheels_l1173_117300


namespace NUMINAMATH_CALUDE_divisibility_condition_iff_n_le_3_l1173_117373

/-- A complete graph with n vertices -/
structure CompleteGraph (n : ℕ) where
  vertices : Fin n
  edges : Fin (n.choose 2)

/-- A labeling of edges with consecutive natural numbers -/
def EdgeLabeling (n : ℕ) := Fin (n.choose 2) → ℕ

/-- Condition for divisibility in a path of length 3 -/
def DivisibilityCondition (g : CompleteGraph n) (l : EdgeLabeling n) : Prop :=
  ∀ (a b c : Fin (n.choose 2)),
    (l b) ∣ (Nat.gcd (l a) (l c))

/-- Main theorem: The divisibility condition can be satisfied if and only if n ≤ 3 -/
theorem divisibility_condition_iff_n_le_3 (n : ℕ) :
  (∃ (g : CompleteGraph n) (l : EdgeLabeling n),
    DivisibilityCondition g l ∧
    (∀ i : Fin (n.choose 2), l i = i.val + 1)) ↔
  n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_divisibility_condition_iff_n_le_3_l1173_117373


namespace NUMINAMATH_CALUDE_point_on_segment_l1173_117309

-- Define the space we're working in (Euclidean plane)
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]

-- Define points A, B, C, and M
variable (A B C M : E)

-- Define the condition that for any M, either MA ≤ MB or MA ≤ MC
def condition (A B C : E) : Prop :=
  ∀ M : E, ‖M - A‖ ≤ ‖M - B‖ ∨ ‖M - A‖ ≤ ‖M - C‖

-- Define what it means for A to lie on the segment BC
def lies_on_segment (A B C : E) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A = (1 - t) • B + t • C

-- State the theorem
theorem point_on_segment (A B C : E) :
  condition A B C → lies_on_segment A B C :=
by
  sorry

end NUMINAMATH_CALUDE_point_on_segment_l1173_117309


namespace NUMINAMATH_CALUDE_expansion_equality_l1173_117362

theorem expansion_equality (m n : ℝ) : (m + n) * (m - 2*n) = m^2 - m*n - 2*n^2 := by
  sorry

end NUMINAMATH_CALUDE_expansion_equality_l1173_117362


namespace NUMINAMATH_CALUDE_gilda_remaining_marbles_l1173_117331

/-- The percentage of marbles Gilda has left after giving away to her friends and family -/
def gildasRemainingMarbles : ℝ :=
  let initialMarbles := 100
  let afterPedro := initialMarbles * (1 - 0.30)
  let afterEbony := afterPedro * (1 - 0.05)
  let afterJimmy := afterEbony * (1 - 0.30)
  let afterTina := afterJimmy * (1 - 0.10)
  afterTina

/-- Theorem stating that Gilda has 41.895% of her original marbles left -/
theorem gilda_remaining_marbles :
  ∀ ε > 0, |gildasRemainingMarbles - 41.895| < ε :=
sorry

end NUMINAMATH_CALUDE_gilda_remaining_marbles_l1173_117331


namespace NUMINAMATH_CALUDE_coefficient_a3_value_l1173_117396

theorem coefficient_a3_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, x^5 + 3*x^3 + 1 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₃ = 13 := by
sorry

end NUMINAMATH_CALUDE_coefficient_a3_value_l1173_117396


namespace NUMINAMATH_CALUDE_square_difference_l1173_117369

theorem square_difference (n : ℕ) (h : (n + 1)^2 = n^2 + 2*n + 1) :
  n^2 - (n - 1)^2 = 2*n - 1 := by
  sorry

#check square_difference 50

end NUMINAMATH_CALUDE_square_difference_l1173_117369


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l1173_117399

-- Define the line equation
def line_equation (a x y : ℝ) : Prop :=
  (a + 2) * x + (1 - a) * y - 3 = 0

-- Theorem statement
theorem fixed_point_on_line (a : ℝ) (h : a ≠ 0) : 
  line_equation a 1 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l1173_117399


namespace NUMINAMATH_CALUDE_green_light_probability_theorem_l1173_117325

/-- Represents the duration of traffic light colors in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of encountering a green light -/
def greenLightProbability (d : TrafficLightDuration) : ℚ :=
  d.green / (d.red + d.yellow + d.green)

/-- Theorem: The probability of encountering a green light at the given intersection is 8/15 -/
theorem green_light_probability_theorem (d : TrafficLightDuration) 
    (h1 : d.red = 30)
    (h2 : d.yellow = 5)
    (h3 : d.green = 40) : 
  greenLightProbability d = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_green_light_probability_theorem_l1173_117325


namespace NUMINAMATH_CALUDE_probability_of_sum_26_l1173_117398

-- Define the faces of the dice
def die1_faces : Finset ℕ := Finset.range 20 \ {0, 19}
def die2_faces : Finset ℕ := (Finset.range 22 \ {0, 8, 21}) ∪ {0}

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 20 * 20

-- Define the favorable outcomes
def favorable_outcomes : ℕ := 13

-- Theorem statement
theorem probability_of_sum_26 :
  (favorable_outcomes : ℚ) / total_outcomes = 13 / 400 :=
sorry

end NUMINAMATH_CALUDE_probability_of_sum_26_l1173_117398


namespace NUMINAMATH_CALUDE_expression_simplification_l1173_117346

theorem expression_simplification (a b : ℝ) 
  (ha : a = Real.sqrt 3 + 1) 
  (hb : b = Real.sqrt 3 - 1) : 
  ((a^2 / (a - b) - (2*a*b - b^2) / (a - b)) / ((a - b) / (a * b))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1173_117346


namespace NUMINAMATH_CALUDE_rabbit_carrots_l1173_117380

theorem rabbit_carrots (rabbit_carrots_per_hole fox_carrots_per_hole : ℕ)
  (hole_difference : ℕ) :
  rabbit_carrots_per_hole = 5 →
  fox_carrots_per_hole = 7 →
  hole_difference = 6 →
  ∃ (rabbit_holes fox_holes : ℕ),
    rabbit_holes = fox_holes + hole_difference ∧
    rabbit_carrots_per_hole * rabbit_holes = fox_carrots_per_hole * fox_holes ∧
    rabbit_carrots_per_hole * rabbit_holes = 105 :=
by sorry

end NUMINAMATH_CALUDE_rabbit_carrots_l1173_117380


namespace NUMINAMATH_CALUDE_shaded_area_is_ten_l1173_117311

/-- A rectangle composed of twelve 1x1 squares -/
structure Rectangle where
  width : ℕ
  height : ℕ
  area : ℕ
  h1 : width = 3
  h2 : height = 4
  h3 : area = width * height
  h4 : area = 12

/-- The unshaded triangular region in the rectangle -/
structure UnshadedTriangle where
  base : ℕ
  height : ℕ
  area : ℝ
  h1 : base = 1
  h2 : height = 4
  h3 : area = (base * height : ℝ) / 2

/-- The total shaded area in the rectangle -/
def shadedArea (r : Rectangle) (ut : UnshadedTriangle) : ℝ :=
  (r.area : ℝ) - ut.area

theorem shaded_area_is_ten (r : Rectangle) (ut : UnshadedTriangle) :
  shadedArea r ut = 10 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_ten_l1173_117311


namespace NUMINAMATH_CALUDE_parameter_a_condition_l1173_117341

theorem parameter_a_condition (a : ℝ) : 
  (∀ x y : ℝ, 2 * a * x^2 + 2 * a * y^2 + 4 * a * x * y - 2 * x * y - y^2 - 2 * x + 1 ≥ 0) → 
  a ≥ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_parameter_a_condition_l1173_117341


namespace NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1173_117307

theorem range_of_a_for_always_positive_quadratic :
  (∀ a : ℝ, (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1)) := by sorry

end NUMINAMATH_CALUDE_range_of_a_for_always_positive_quadratic_l1173_117307


namespace NUMINAMATH_CALUDE_amy_remaining_money_l1173_117349

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial_amount : ℕ) (num_items : ℕ) (item_cost : ℕ) : ℕ :=
  initial_amount - (num_items * item_cost)

/-- Proves that Amy has $97 left after her purchase -/
theorem amy_remaining_money :
  remaining_money 100 3 1 = 97 := by
  sorry

end NUMINAMATH_CALUDE_amy_remaining_money_l1173_117349


namespace NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l1173_117382

theorem cos_pi_4_plus_alpha (α : Real) 
  (h : Real.sin (π / 4 - α) = Real.sqrt 2 / 2) : 
  Real.cos (π / 4 + α) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_4_plus_alpha_l1173_117382


namespace NUMINAMATH_CALUDE_pie_chart_central_angle_l1173_117348

theorem pie_chart_central_angle 
  (total_data : ℕ) 
  (group_frequency : ℕ) 
  (h1 : total_data = 60) 
  (h2 : group_frequency = 15) : 
  (group_frequency : ℝ) / (total_data : ℝ) * 360 = 90 := by
sorry

end NUMINAMATH_CALUDE_pie_chart_central_angle_l1173_117348


namespace NUMINAMATH_CALUDE_quadratic_sum_reciprocal_l1173_117343

theorem quadratic_sum_reciprocal (t : ℝ) (h1 : t^2 - 3*t + 1 = 0) (h2 : t ≠ 0) :
  t + 1/t = 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_sum_reciprocal_l1173_117343


namespace NUMINAMATH_CALUDE_correct_amount_to_return_l1173_117351

/-- Calculates the amount to be returned in rubles given an initial deposit in USD and an exchange rate. -/
def amount_to_return (initial_deposit : ℝ) (exchange_rate : ℝ) : ℝ :=
  initial_deposit * exchange_rate

/-- Theorem stating that given the specific initial deposit and exchange rate, the amount to be returned is 581,500 rubles. -/
theorem correct_amount_to_return :
  amount_to_return 10000 58.15 = 581500 := by
  sorry

end NUMINAMATH_CALUDE_correct_amount_to_return_l1173_117351


namespace NUMINAMATH_CALUDE_correct_equation_l1173_117347

theorem correct_equation : (-3)^2 = 9 ∧ 
  (-2)^3 ≠ -6 ∧ 
  ¬(∀ x, x^2 = 4 → x = 2 ∨ x = -2) ∧ 
  (Real.sqrt 2)^2 ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l1173_117347


namespace NUMINAMATH_CALUDE_total_pencils_l1173_117386

theorem total_pencils (boxes : ℕ) (pencils_per_box : ℕ) (h1 : boxes = 162) (h2 : pencils_per_box = 4) : 
  boxes * pencils_per_box = 648 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l1173_117386


namespace NUMINAMATH_CALUDE_ratio_as_percent_l1173_117340

theorem ratio_as_percent (first_part second_part : ℕ) (h1 : first_part = 4) (h2 : second_part = 20) :
  (first_part : ℚ) / second_part * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ratio_as_percent_l1173_117340


namespace NUMINAMATH_CALUDE_octal_sum_example_l1173_117321

/-- Represents a number in base 8 --/
def OctalNumber := Nat

/-- Converts a natural number to its octal representation --/
def toOctal (n : Nat) : OctalNumber := sorry

/-- Adds two octal numbers --/
def octalAdd (a b : OctalNumber) : OctalNumber := sorry

/-- Theorem: The sum of 356₈, 672₈, and 145₈ is 1477₈ in base 8 --/
theorem octal_sum_example : 
  octalAdd (octalAdd (toOctal 356) (toOctal 672)) (toOctal 145) = toOctal 1477 := by sorry

end NUMINAMATH_CALUDE_octal_sum_example_l1173_117321


namespace NUMINAMATH_CALUDE_line_intersects_y_axis_l1173_117314

/-- A line passing through two given points intersects the y-axis at (0, 0) -/
theorem line_intersects_y_axis (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : x₁ = 3 ∧ y₁ = 9)
  (h₂ : x₂ = -7 ∧ y₂ = -21) :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) ∧
    0 = m * 0 + b :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_y_axis_l1173_117314


namespace NUMINAMATH_CALUDE_unique_solution_cubic_system_l1173_117301

theorem unique_solution_cubic_system (x y z : ℝ) :
  x^3 + y = z^2 ∧ y^3 + z = x^2 ∧ z^3 + x = y^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cubic_system_l1173_117301


namespace NUMINAMATH_CALUDE_festival_allowance_petty_cash_l1173_117345

theorem festival_allowance_petty_cash (staff_count : ℕ) (days : ℕ) (daily_rate : ℕ) (total_given : ℕ) :
  staff_count = 20 →
  days = 30 →
  daily_rate = 100 →
  total_given = 65000 →
  total_given - (staff_count * days * daily_rate) = 5000 := by
sorry

end NUMINAMATH_CALUDE_festival_allowance_petty_cash_l1173_117345


namespace NUMINAMATH_CALUDE_solution_to_equation_l1173_117323

theorem solution_to_equation (x : ℝ) : 2 * x - 8 = 0 ↔ x = 4 := by sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1173_117323


namespace NUMINAMATH_CALUDE_inheritance_tax_theorem_inheritance_uniqueness_l1173_117305

/-- The original amount of inheritance --/
def inheritance : ℝ := 41379

/-- The total amount of taxes paid --/
def total_taxes : ℝ := 15000

/-- Theorem stating that the inheritance amount satisfies the tax conditions --/
theorem inheritance_tax_theorem :
  0.25 * inheritance + 0.15 * (0.75 * inheritance) = total_taxes :=
by sorry

/-- Theorem proving that the inheritance amount is unique --/
theorem inheritance_uniqueness (x : ℝ) :
  0.25 * x + 0.15 * (0.75 * x) = total_taxes → x = inheritance :=
by sorry

end NUMINAMATH_CALUDE_inheritance_tax_theorem_inheritance_uniqueness_l1173_117305


namespace NUMINAMATH_CALUDE_no_real_solutions_for_log_equation_l1173_117352

theorem no_real_solutions_for_log_equation :
  ∀ (p q : ℝ), Real.log (p * q) = Real.log (p^2 + q^2 + 1) → False :=
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_log_equation_l1173_117352


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l1173_117319

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l1173_117319


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1173_117344

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = f (x^2 - y) + 4 * f x * y) →
  (∀ x : ℝ, f x = 0 ∨ f x = x^2) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1173_117344


namespace NUMINAMATH_CALUDE_combined_selling_price_is_3620_l1173_117379

def article1_cost : ℝ := 1200
def article2_cost : ℝ := 800
def article3_cost : ℝ := 600

def article1_profit_rate : ℝ := 0.4
def article2_profit_rate : ℝ := 0.3
def article3_profit_rate : ℝ := 0.5

def selling_price (cost : ℝ) (profit_rate : ℝ) : ℝ :=
  cost * (1 + profit_rate)

def combined_selling_price : ℝ :=
  selling_price article1_cost article1_profit_rate +
  selling_price article2_cost article2_profit_rate +
  selling_price article3_cost article3_profit_rate

theorem combined_selling_price_is_3620 :
  combined_selling_price = 3620 := by
  sorry

end NUMINAMATH_CALUDE_combined_selling_price_is_3620_l1173_117379


namespace NUMINAMATH_CALUDE_garys_to_harrys_book_ratio_l1173_117339

/-- Proves that the ratio of Gary's books to Harry's books is 1:2 given the specified conditions -/
theorem garys_to_harrys_book_ratio :
  ∀ (harry_books flora_books gary_books : ℕ),
    harry_books = 50 →
    flora_books = 2 * harry_books →
    harry_books + flora_books + gary_books = 175 →
    gary_books = (1 : ℚ) / 2 * harry_books := by
  sorry

end NUMINAMATH_CALUDE_garys_to_harrys_book_ratio_l1173_117339


namespace NUMINAMATH_CALUDE_three_digit_square_insertion_l1173_117367

theorem three_digit_square_insertion (n : ℕ) : ∃ (a b c : ℕ) (a' b' c' : ℕ),
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a' ≠ b' ∧ b' ≠ c' ∧ a' ≠ c' ∧
  0 < a ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
  0 < a' ∧ a' < 10 ∧ b' < 10 ∧ c' < 10 ∧
  (a ≠ a' ∨ b ≠ b' ∨ c ≠ c') ∧
  ∃ (k : ℕ), a * 10^(2*n+2) + b * 10^(n+1) + c = k^2 ∧
  ∃ (k' : ℕ), a' * 10^(2*n+2) + b' * 10^(n+1) + c' = k'^2 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_square_insertion_l1173_117367


namespace NUMINAMATH_CALUDE_max_subsets_with_intersection_condition_l1173_117389

/-- Given a positive integer n ≥ 2, prove that the maximum number of mutually distinct subsets
    that can be selected from an n-element set, satisfying (Aᵢ ∩ Aₖ) ⊆ Aⱼ for all 1 ≤ i < j < k ≤ m,
    is 2n. -/
theorem max_subsets_with_intersection_condition (n : ℕ) (hn : n ≥ 2) :
  (∃ (m : ℕ) (S : Finset (Finset (Fin n))),
    (∀ A ∈ S, A ⊆ Finset.univ) ∧
    (Finset.card S = m) ∧
    (∀ (A B C : Finset (Fin n)), A ∈ S → B ∈ S → C ∈ S →
      (Finset.toList S).indexOf A < (Finset.toList S).indexOf B →
      (Finset.toList S).indexOf B < (Finset.toList S).indexOf C →
      (A ∩ C) ⊆ B) ∧
    (∀ (m' : ℕ) (S' : Finset (Finset (Fin n))),
      (∀ A ∈ S', A ⊆ Finset.univ) →
      (Finset.card S' = m') →
      (∀ (A B C : Finset (Fin n)), A ∈ S' → B ∈ S' → C ∈ S' →
        (Finset.toList S').indexOf A < (Finset.toList S').indexOf B →
        (Finset.toList S').indexOf B < (Finset.toList S').indexOf C →
        (A ∩ C) ⊆ B) →
      m' ≤ m)) ∧
  (m = 2 * n) :=
by sorry

end NUMINAMATH_CALUDE_max_subsets_with_intersection_condition_l1173_117389


namespace NUMINAMATH_CALUDE_max_profit_is_200_l1173_117303

/-- Represents a neighborhood with its characteristics --/
structure Neighborhood where
  homes : ℕ
  boxesPerHome : ℕ
  pricePerBox : ℚ

/-- Calculates the total sales for a neighborhood --/
def totalSales (n : Neighborhood) : ℚ :=
  n.homes * n.boxesPerHome * n.pricePerBox

/-- The four neighborhoods with their respective characteristics --/
def neighborhoodA : Neighborhood := ⟨12, 3, 3⟩
def neighborhoodB : Neighborhood := ⟨8, 6, 4⟩
def neighborhoodC : Neighborhood := ⟨15, 2, (5/2)⟩
def neighborhoodD : Neighborhood := ⟨5, 8, 5⟩

/-- Theorem stating that the maximum profit among the four neighborhoods is $200 --/
theorem max_profit_is_200 :
  max (totalSales neighborhoodA) (max (totalSales neighborhoodB) (max (totalSales neighborhoodC) (totalSales neighborhoodD))) = 200 := by
  sorry


end NUMINAMATH_CALUDE_max_profit_is_200_l1173_117303


namespace NUMINAMATH_CALUDE_f_properties_l1173_117370

noncomputable def f (x : ℝ) : ℝ := Real.log x + 1 / x

theorem f_properties :
  (∀ x > 0, f x ≥ 1) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f x₁ = f x₂ → x₁ + x₂ > 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1173_117370


namespace NUMINAMATH_CALUDE_count_four_digit_integers_l1173_117366

/-- The number of distinct four-digit positive integers formed with digits 3, 3, 8, and 8 -/
def fourDigitIntegersCount : ℕ := 6

/-- The set of digits used to form the integers -/
def digits : Finset ℕ := {3, 8}

/-- The number of times each digit is used -/
def digitRepetitions : ℕ := 2

/-- The total number of digits used -/
def totalDigits : ℕ := 4

theorem count_four_digit_integers :
  fourDigitIntegersCount = (totalDigits.factorial) / (digitRepetitions.factorial ^ digits.card) :=
sorry

end NUMINAMATH_CALUDE_count_four_digit_integers_l1173_117366


namespace NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_and_twelve_l1173_117387

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def divisible_by_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ≠ 0 → d ≠ 10 → d < 10 → (n % 10 = d ∨ (n / 10) % 10 = d ∨ n / 100 = d) → n % d = 0

theorem largest_three_digit_divisible_by_digits_and_twelve :
  ∀ n : ℕ, is_three_digit n → divisible_by_digits n → n % 12 = 0 → n ≤ 864 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_divisible_by_digits_and_twelve_l1173_117387


namespace NUMINAMATH_CALUDE_exam_score_standard_deviations_l1173_117320

/-- Given an exam with mean score 76, where 60 is 2 standard deviations below the mean,
    prove that 100 is 3 standard deviations above the mean. -/
theorem exam_score_standard_deviations 
  (mean : ℝ) 
  (std_dev : ℝ) 
  (h1 : mean = 76) 
  (h2 : mean - 2 * std_dev = 60) 
  (h3 : mean + 3 * std_dev = 100) : 
  100 = mean + 3 * std_dev := by
  sorry

end NUMINAMATH_CALUDE_exam_score_standard_deviations_l1173_117320


namespace NUMINAMATH_CALUDE_quadratic_root_values_l1173_117364

/-- Given that 1 - i is a root of a real-coefficient quadratic equation x² + ax + b = 0,
    prove that a = -2 and b = 2 -/
theorem quadratic_root_values (a b : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (1 - Complex.I)^2 + a*(1 - Complex.I) + b = 0 →
  a = -2 ∧ b = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_values_l1173_117364


namespace NUMINAMATH_CALUDE_scientific_notation_of_60000_l1173_117310

theorem scientific_notation_of_60000 : 60000 = 6 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_60000_l1173_117310


namespace NUMINAMATH_CALUDE_conference_handshakes_l1173_117383

/-- The number of handshakes in a conference where each person shakes hands with every other person exactly once. -/
def num_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a conference of 10 people where each person shakes hands with every other person exactly once, there are 45 handshakes. -/
theorem conference_handshakes : num_handshakes 10 = 45 := by
  sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1173_117383


namespace NUMINAMATH_CALUDE_quadratic_roots_real_and_unequal_l1173_117308

theorem quadratic_roots_real_and_unequal :
  let a : ℝ := 1
  let b : ℝ := -6
  let c : ℝ := 8
  let discriminant := b^2 - 4*a*c
  discriminant > 0 ∧ ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 :=
by
  sorry

#check quadratic_roots_real_and_unequal

end NUMINAMATH_CALUDE_quadratic_roots_real_and_unequal_l1173_117308


namespace NUMINAMATH_CALUDE_set_equality_l1173_117377

def A : Set ℝ := {x | x^2 - 1 < 0}
def B : Set ℝ := {x | x > 0}

theorem set_equality : (Set.compl A) ∪ B = Set.Iic (-1) ∪ Set.Ioi 0 := by sorry

end NUMINAMATH_CALUDE_set_equality_l1173_117377


namespace NUMINAMATH_CALUDE_non_adjacent_arrangement_count_l1173_117390

/-- Represents the number of ways to arrange balls in a row -/
def arrangement_count : ℕ := 12

/-- Represents the number of white balls -/
def white_ball_count : ℕ := 1

/-- Represents the number of red balls -/
def red_ball_count : ℕ := 1

/-- Represents the number of yellow balls -/
def yellow_ball_count : ℕ := 3

/-- Theorem stating that the number of arrangements where white and red balls are not adjacent is 12 -/
theorem non_adjacent_arrangement_count :
  (white_ball_count = 1) →
  (red_ball_count = 1) →
  (yellow_ball_count = 3) →
  (arrangement_count = 12) := by
  sorry

#check non_adjacent_arrangement_count

end NUMINAMATH_CALUDE_non_adjacent_arrangement_count_l1173_117390


namespace NUMINAMATH_CALUDE_coin_landing_probability_l1173_117374

/-- Represents the specially colored square -/
structure ColoredSquare where
  side_length : ℝ
  triangle_leg : ℝ
  diamond_side : ℝ

/-- Represents the circular coin -/
structure Coin where
  diameter : ℝ

/-- Calculates the probability of the coin landing on a black region -/
def black_region_probability (square : ColoredSquare) (coin : Coin) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem coin_landing_probability 
  (square : ColoredSquare)
  (coin : Coin)
  (h_square_side : square.side_length = 8)
  (h_triangle_leg : square.triangle_leg = 2)
  (h_diamond_side : square.diamond_side = 2 * Real.sqrt 2)
  (h_coin_diameter : coin.diameter = 1) :
  ∃ (a b : ℕ), 
    black_region_probability square coin = 1 / 196 * (a + b * Real.sqrt 2 + Real.pi) ∧
    a + b = 68 :=
  sorry

end NUMINAMATH_CALUDE_coin_landing_probability_l1173_117374


namespace NUMINAMATH_CALUDE_expected_pairs_in_both_arrangements_l1173_117335

/-- Represents a 7x7 grid arrangement of numbers 1 through 49 -/
def Arrangement := Fin 49 → Fin 7 × Fin 7

/-- The number of rows in the grid -/
def num_rows : Nat := 7

/-- The number of columns in the grid -/
def num_cols : Nat := 7

/-- The total number of numbers in the grid -/
def total_numbers : Nat := num_rows * num_cols

/-- Calculates the expected number of pairs that occur in the same row or column in both arrangements -/
noncomputable def expected_pairs (a1 a2 : Arrangement) : ℝ :=
  (total_numbers.choose 2 : ℝ) * (1 / 16)

/-- The main theorem stating the expected number of pairs -/
theorem expected_pairs_in_both_arrangements :
  ∀ a1 a2 : Arrangement, expected_pairs a1 a2 = 73.5 := by
  sorry

end NUMINAMATH_CALUDE_expected_pairs_in_both_arrangements_l1173_117335


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l1173_117329

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) 
  (bicycle_speed : ℝ) (foot_distance : ℝ) 
  (h1 : total_distance = 61) 
  (h2 : total_time = 9)
  (h3 : bicycle_speed = 9)
  (h4 : foot_distance = 16) :
  ∃ (foot_speed : ℝ), 
    foot_speed = 4 ∧ 
    foot_distance / foot_speed + (total_distance - foot_distance) / bicycle_speed = total_time :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l1173_117329


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1173_117361

theorem inequality_solution_set (x : ℝ) :
  (x ∈ {x : ℝ | -6 * x^2 + 2 < x}) ↔ (x < -2/3 ∨ x > 1/2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1173_117361


namespace NUMINAMATH_CALUDE_beef_weight_before_processing_l1173_117391

/-- If a side of beef loses 50 percent of its weight in processing and weighs 750 pounds after processing, then it weighed 1500 pounds before processing. -/
theorem beef_weight_before_processing (weight_after : ℝ) (h1 : weight_after = 750) :
  ∃ weight_before : ℝ, weight_before * 0.5 = weight_after ∧ weight_before = 1500 := by
  sorry

end NUMINAMATH_CALUDE_beef_weight_before_processing_l1173_117391


namespace NUMINAMATH_CALUDE_multiply_by_seven_l1173_117356

theorem multiply_by_seven (x : ℝ) (h : 8 * x = 64) : 7 * x = 56 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_seven_l1173_117356


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l1173_117304

/-- A line y = kx + 2 intersects the left branch of x^2 - y^2 = 4 at two distinct points iff k ∈ (1, √2) -/
theorem line_hyperbola_intersection (k : ℝ) : 
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ < x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧
    y₁ = k * x₁ + 2 ∧ y₂ = k * x₂ + 2 ∧
    x₁^2 - y₁^2 = 4 ∧ x₂^2 - y₂^2 = 4) ↔ 
  (1 < k ∧ k < Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l1173_117304


namespace NUMINAMATH_CALUDE_lucas_numbers_l1173_117385

theorem lucas_numbers (a b : ℤ) : 
  3 * a + 4 * b = 140 → (a = 20 ∨ b = 20) → a = 20 ∧ b = 20 := by
  sorry

end NUMINAMATH_CALUDE_lucas_numbers_l1173_117385


namespace NUMINAMATH_CALUDE_right_triangle_existence_l1173_117360

theorem right_triangle_existence (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  c + a = p ∧ c + b = q ∧ a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_existence_l1173_117360


namespace NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1173_117393

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|
def g (a : ℝ) (x : ℝ) : ℝ := f a x - |x - 3|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x + |2*x - 5| ≥ 6} = {x : ℝ | x ≥ 4 ∨ x ≤ 0} := by sorry

-- Part II
theorem range_of_a_part_ii (h : Set.Icc (-1 : ℝ) 2 ⊆ Set.range (g a)) :
  a ≤ 1 ∨ a ≥ 5 := by sorry

end NUMINAMATH_CALUDE_solution_set_part_i_range_of_a_part_ii_l1173_117393


namespace NUMINAMATH_CALUDE_arthur_muffins_l1173_117324

theorem arthur_muffins (initial_muffins additional_muffins : ℕ) 
  (h1 : initial_muffins = 35)
  (h2 : additional_muffins = 48) :
  initial_muffins + additional_muffins = 83 :=
by sorry

end NUMINAMATH_CALUDE_arthur_muffins_l1173_117324


namespace NUMINAMATH_CALUDE_h_of_two_eq_two_l1173_117388

/-- The function h satisfying the given equation for all x -/
noncomputable def h : ℝ → ℝ :=
  fun x => ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) - 1) / (x^(2^4 - 1) - 1)

/-- Theorem stating that h(2) = 2 -/
theorem h_of_two_eq_two : h 2 = 2 := by sorry

end NUMINAMATH_CALUDE_h_of_two_eq_two_l1173_117388


namespace NUMINAMATH_CALUDE_kelly_apples_l1173_117375

/-- The number of apples Kelly initially has -/
def initial_apples : ℕ := 56

/-- The number of additional apples Kelly needs to pick -/
def apples_to_pick : ℕ := 49

/-- The total number of apples Kelly wants to have -/
def total_apples : ℕ := initial_apples + apples_to_pick

theorem kelly_apples : total_apples = 105 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l1173_117375


namespace NUMINAMATH_CALUDE_inequality_of_powers_l1173_117372

theorem inequality_of_powers (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^(2*a) * b^(2*b) * c^(2*c) ≥ a^(b+c) * b^(a+c) * c^(a+b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_powers_l1173_117372


namespace NUMINAMATH_CALUDE_AC_length_l1173_117376

-- Define the isosceles trapezoid
structure IsoscelesTrapezoid :=
  (AB : ℝ)
  (CD : ℝ)
  (AD : ℝ)
  (BC : ℝ)
  (isIsosceles : AD = BC)

-- Define our specific trapezoid
def specificTrapezoid : IsoscelesTrapezoid :=
  { AB := 30
  , CD := 12
  , AD := 15
  , BC := 15
  , isIsosceles := rfl }

-- Theorem statement
theorem AC_length (t : IsoscelesTrapezoid) (h : t = specificTrapezoid) :
  ∃ (AC : ℝ), AC = Real.sqrt (12^2 + 20^2) :=
sorry

end NUMINAMATH_CALUDE_AC_length_l1173_117376


namespace NUMINAMATH_CALUDE_negative_eighth_power_2009_times_eight_power_2009_l1173_117365

theorem negative_eighth_power_2009_times_eight_power_2009 :
  (-0.125)^2009 * 8^2009 = -1 := by
  sorry

end NUMINAMATH_CALUDE_negative_eighth_power_2009_times_eight_power_2009_l1173_117365


namespace NUMINAMATH_CALUDE_certain_number_divisor_l1173_117392

theorem certain_number_divisor (n : Nat) (h1 : n = 1020) : 
  ∃ x : Nat, x > 0 ∧ 
  (n - 12) % x = 0 ∧
  (n - 12) % 12 = 0 ∧ 
  (n - 12) % 24 = 0 ∧ 
  (n - 12) % 36 = 0 ∧ 
  (n - 12) % 48 = 0 ∧
  x = 7 ∧
  x ∉ Nat.divisors (Nat.lcm 12 (Nat.lcm 24 (Nat.lcm 36 48))) ∧
  ∀ y : Nat, y > x → (n - 12) % y ≠ 0 ∨ 
    y ∈ Nat.divisors (Nat.lcm 12 (Nat.lcm 24 (Nat.lcm 36 48))) :=
by sorry

end NUMINAMATH_CALUDE_certain_number_divisor_l1173_117392


namespace NUMINAMATH_CALUDE_isosceles_triangle_l1173_117353

theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + b = Real.tan (C / 2) * (a * Real.tan A + b * Real.tan B) →
  A = B := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l1173_117353


namespace NUMINAMATH_CALUDE_product_digit_sum_l1173_117318

/-- Represents a 101-digit number with alternating digits --/
def AlternatingDigitNumber (a b : ℕ) : ℕ := sorry

/-- The first 101-digit number: 1010101...010101 --/
def num1 : ℕ := AlternatingDigitNumber 1 0

/-- The second 101-digit number: 7070707...070707 --/
def num2 : ℕ := AlternatingDigitNumber 7 0

/-- Returns the hundreds digit of a natural number --/
def hundredsDigit (n : ℕ) : ℕ := sorry

/-- Returns the units digit of a natural number --/
def unitsDigit (n : ℕ) : ℕ := sorry

theorem product_digit_sum :
  hundredsDigit (num1 * num2) + unitsDigit (num1 * num2) = 10 := by sorry

end NUMINAMATH_CALUDE_product_digit_sum_l1173_117318


namespace NUMINAMATH_CALUDE_complement_union_problem_l1173_117333

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {2, 3}

theorem complement_union_problem : (U \ A) ∪ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_union_problem_l1173_117333


namespace NUMINAMATH_CALUDE_patrol_results_l1173_117358

def travel_records : List Int := [10, -8, 6, -13, 7, -12, 3, -1]

def fuel_consumption_rate : ℝ := 0.05

def gas_station_distance : Int := 6

def final_position (records : List Int) : Int :=
  records.sum

def total_distance (records : List Int) : Int :=
  records.map (Int.natAbs) |>.sum

def times_passed_gas_station (records : List Int) (station_dist : Int) : Nat :=
  sorry

theorem patrol_results :
  (final_position travel_records = -8) ∧
  (total_distance travel_records = 60) ∧
  (times_passed_gas_station travel_records gas_station_distance = 4) := by
  sorry

end NUMINAMATH_CALUDE_patrol_results_l1173_117358


namespace NUMINAMATH_CALUDE_salary_adjustment_l1173_117315

theorem salary_adjustment (original_salary : ℝ) (original_salary_positive : original_salary > 0) :
  let reduced_salary := original_salary * 0.9
  (reduced_salary * (1 + 100/9 * 0.01) : ℝ) = original_salary := by
  sorry

end NUMINAMATH_CALUDE_salary_adjustment_l1173_117315


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1173_117359

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 2 * n - 1

-- Define S_n as the sum of first n terms of a_n
def S (n : ℕ) : ℚ := n * (a 1 + a n) / 2

-- Define b_n
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- Define T_n as the sum of first n terms of b_n
def T (n : ℕ) : ℚ := n / (2 * n + 1)

theorem arithmetic_sequence_properties (n : ℕ) :
  (S 3 = a 4 + 2) ∧ 
  (a 3 ^ 2 = a 1 * a 13) ∧ 
  (∀ k : ℕ, a (k + 1) - a k = a 2 - a 1) ∧
  (∀ k : ℕ, b k = 1 / (a k * a (k + 1))) ∧
  (∀ k : ℕ, T k = k / (2 * k + 1)) := by
  sorry

#check arithmetic_sequence_properties

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1173_117359


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_9879_l1173_117378

theorem largest_prime_factor_of_9879 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 9879 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 9879 → q ≤ p :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_9879_l1173_117378


namespace NUMINAMATH_CALUDE_donation_distribution_l1173_117317

theorem donation_distribution (total : ℝ) (contingency : ℝ) : 
  total = 240 →
  contingency = 30 →
  (3 : ℝ) / 8 * total = total - (1 / 3 * total) - (1 / 4 * (total - 1 / 3 * total)) - contingency :=
by sorry

end NUMINAMATH_CALUDE_donation_distribution_l1173_117317


namespace NUMINAMATH_CALUDE_last_four_average_l1173_117327

theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 3).sum / 3 = 58 →
  (list.drop 3).sum / 4 = 65 := by
sorry

end NUMINAMATH_CALUDE_last_four_average_l1173_117327


namespace NUMINAMATH_CALUDE_temperature_85_at_latest_time_l1173_117326

/-- The temperature function in Denver, CO, where t is time in hours past noon -/
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

/-- The latest time when the temperature is 85 degrees -/
def latest_85_degrees : ℝ := 9

theorem temperature_85_at_latest_time :
  temperature latest_85_degrees = 85 ∧
  ∀ t > latest_85_degrees, temperature t ≠ 85 := by
sorry

end NUMINAMATH_CALUDE_temperature_85_at_latest_time_l1173_117326


namespace NUMINAMATH_CALUDE_multiplication_value_proof_l1173_117313

theorem multiplication_value_proof : 
  let initial_number : ℝ := 2.25
  let division_factor : ℝ := 3
  let multiplication_value : ℝ := 12
  let result : ℝ := 9
  (initial_number / division_factor) * multiplication_value = result := by
sorry

end NUMINAMATH_CALUDE_multiplication_value_proof_l1173_117313


namespace NUMINAMATH_CALUDE_triangle_properties_l1173_117350

/-- Given two 2D vectors a and b, proves statements about the triangle formed by 0, a, and b -/
theorem triangle_properties (a b : Fin 2 → ℝ) 
  (ha : a = ![4, -1]) 
  (hb : b = ![2, 6]) : 
  (1/2 * abs (a 0 * b 1 - a 1 * b 0) = 13) ∧ 
  ((((a 0 + b 0)/2)^2 + ((a 1 + b 1)/2)^2) = 15.25) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1173_117350


namespace NUMINAMATH_CALUDE_octagon_side_length_l1173_117316

/-- The side length of a regular octagon formed from the same wire as a regular pentagon --/
theorem octagon_side_length (pentagon_side : ℝ) (h : pentagon_side = 16) : 
  let pentagon_perimeter := 5 * pentagon_side
  let octagon_side := pentagon_perimeter / 8
  octagon_side = 10 := by
sorry

end NUMINAMATH_CALUDE_octagon_side_length_l1173_117316


namespace NUMINAMATH_CALUDE_expression_evaluation_l1173_117368

theorem expression_evaluation : 60 + 5 * 12 / (180 / 3) = 61 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1173_117368


namespace NUMINAMATH_CALUDE_race_time_calculation_l1173_117355

/-- Represents a race between two runners A and B -/
structure Race where
  length : ℝ  -- Race length in meters
  lead_distance : ℝ  -- Distance by which A beats B
  lead_time : ℝ  -- Time by which A beats B
  a_time : ℝ  -- Time taken by A to complete the race

/-- Theorem stating that for the given race conditions, A's time is 5.25 seconds -/
theorem race_time_calculation (race : Race) 
  (h1 : race.length = 80)
  (h2 : race.lead_distance = 56)
  (h3 : race.lead_time = 7) :
  race.a_time = 5.25 := by
  sorry

#check race_time_calculation

end NUMINAMATH_CALUDE_race_time_calculation_l1173_117355


namespace NUMINAMATH_CALUDE_smallest_digit_sum_of_product_l1173_117332

/-- Given two two-digit positive integers with all digits different and both less than 50,
    the smallest possible sum of digits of their product (a four-digit number) is 20. -/
theorem smallest_digit_sum_of_product (m n : ℕ) : 
  10 ≤ m ∧ m < 50 ∧ 10 ≤ n ∧ n < 50 ∧ 
  (∀ d₁ d₂ d₃ d₄, m = 10 * d₁ + d₂ ∧ n = 10 * d₃ + d₄ → d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄) →
  1000 ≤ m * n ∧ m * n < 10000 →
  20 ≤ (m * n / 1000 + (m * n / 100) % 10 + (m * n / 10) % 10 + m * n % 10) ∧
  ∀ p q : ℕ, 10 ≤ p ∧ p < 50 ∧ 10 ≤ q ∧ q < 50 →
    (∀ e₁ e₂ e₃ e₄, p = 10 * e₁ + e₂ ∧ q = 10 * e₃ + e₄ → e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₁ ≠ e₄ ∧ e₂ ≠ e₃ ∧ e₂ ≠ e₄ ∧ e₃ ≠ e₄) →
    1000 ≤ p * q ∧ p * q < 10000 →
    (p * q / 1000 + (p * q / 100) % 10 + (p * q / 10) % 10 + p * q % 10) ≥ 20 :=
by sorry

end NUMINAMATH_CALUDE_smallest_digit_sum_of_product_l1173_117332


namespace NUMINAMATH_CALUDE_lemon_fraction_is_one_seventh_l1173_117337

/-- Represents the contents of a cup --/
structure CupContents where
  tea : ℚ
  honey : ℚ
  lemon : ℚ

/-- The initial setup of the cups --/
def initial_setup : CupContents × CupContents :=
  ({tea := 6, honey := 0, lemon := 0}, {tea := 0, honey := 6, lemon := 3})

/-- Pours half of the tea from the first cup to the second --/
def pour_half_tea (cups : CupContents × CupContents) : CupContents × CupContents :=
  let (cup1, cup2) := cups
  let tea_to_pour := cup1.tea / 2
  ({tea := cup1.tea - tea_to_pour, honey := cup1.honey, lemon := cup1.lemon},
   {tea := cup2.tea + tea_to_pour, honey := cup2.honey, lemon := cup2.lemon})

/-- Pours one-third of the second cup's contents back to the first cup --/
def pour_third_back (cups : CupContents × CupContents) : CupContents × CupContents :=
  let (cup1, cup2) := cups
  let total_cup2 := cup2.tea + cup2.honey + cup2.lemon
  let fraction := 1 / 3
  let tea_to_pour := cup2.tea * fraction
  let honey_to_pour := cup2.honey * fraction
  let lemon_to_pour := cup2.lemon * fraction
  ({tea := cup1.tea + tea_to_pour, honey := cup1.honey + honey_to_pour, lemon := cup1.lemon + lemon_to_pour},
   {tea := cup2.tea - tea_to_pour, honey := cup2.honey - honey_to_pour, lemon := cup2.lemon - lemon_to_pour})

/-- Calculates the fraction of lemon juice in a cup --/
def lemon_fraction (cup : CupContents) : ℚ :=
  cup.lemon / (cup.tea + cup.honey + cup.lemon)

theorem lemon_fraction_is_one_seventh :
  let final_state := pour_third_back (pour_half_tea initial_setup)
  lemon_fraction final_state.fst = 1 / 7 := by
  sorry


end NUMINAMATH_CALUDE_lemon_fraction_is_one_seventh_l1173_117337


namespace NUMINAMATH_CALUDE_simplify_expression_l1173_117312

theorem simplify_expression (y : ℝ) :
  3 * y + 7 * y^2 + 10 - (5 - 3 * y - 7 * y^2) = 14 * y^2 + 6 * y + 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1173_117312


namespace NUMINAMATH_CALUDE_average_difference_l1173_117334

theorem average_difference (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + 70 + x) / 3 + 7 → x = 19 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l1173_117334


namespace NUMINAMATH_CALUDE_count_divisible_numbers_l1173_117394

theorem count_divisible_numbers : 
  (Finset.filter (fun n : ℕ => 
    n ≤ 10^10 ∧ 
    (∀ k : ℕ, k ∈ Finset.range 10 → k.succ ∣ n)
  ) (Finset.range (10^10 + 1))).card = 3968253 := by
  sorry

end NUMINAMATH_CALUDE_count_divisible_numbers_l1173_117394


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1173_117306

/-- The line √3x - y + m = 0 is tangent to the circle x^2 + y^2 - 2y = 0 if and only if m = -1 or m = 3 -/
theorem line_tangent_to_circle (m : ℝ) : 
  (∀ x y : ℝ, (Real.sqrt 3 * x - y + m = 0) → (x^2 + y^2 - 2*y = 0) → 
   (∀ ε > 0, ∃ x' y' : ℝ, x' ≠ x ∨ y' ≠ y ∧ 
    (Real.sqrt 3 * x' - y' + m = 0) ∧ 
    (x'^2 + y'^2 - 2*y' ≠ 0) ∧
    ((x' - x)^2 + (y' - y)^2 < ε^2))) ↔ 
  (m = -1 ∨ m = 3) :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1173_117306


namespace NUMINAMATH_CALUDE_quadratic_decreasing_condition_l1173_117336

/-- Given a quadratic function y = (x - m)^2 - 1, if it decreases as x increases
    when x ≤ 3, then m ≥ 3. -/
theorem quadratic_decreasing_condition (m : ℝ) : 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 3 → 
    ((x₁ - m)^2 - 1) > ((x₂ - m)^2 - 1)) → 
  m ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_condition_l1173_117336


namespace NUMINAMATH_CALUDE_interest_rate_problem_l1173_117330

/-- Simple interest calculation -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

/-- Problem statement -/
theorem interest_rate_problem (principal interest time : ℚ) 
  (h1 : principal = 2000)
  (h2 : interest = 500)
  (h3 : time = 2)
  (h4 : simple_interest principal (12.5 : ℚ) time = interest) :
  12.5 = (interest * 100) / (principal * time) := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l1173_117330


namespace NUMINAMATH_CALUDE_big_bottles_count_l1173_117363

/-- The number of big bottles initially in storage -/
def big_bottles : ℕ := 14000

/-- The number of small bottles initially in storage -/
def small_bottles : ℕ := 6000

/-- The percentage of small bottles sold -/
def small_bottles_sold_percent : ℚ := 20 / 100

/-- The percentage of big bottles sold -/
def big_bottles_sold_percent : ℚ := 23 / 100

/-- The total number of bottles remaining in storage -/
def total_remaining : ℕ := 15580

theorem big_bottles_count :
  (small_bottles * (1 - small_bottles_sold_percent) : ℚ).floor +
  (big_bottles * (1 - big_bottles_sold_percent) : ℚ).floor = total_remaining := by
  sorry

end NUMINAMATH_CALUDE_big_bottles_count_l1173_117363


namespace NUMINAMATH_CALUDE_savings_fraction_is_5_17_l1173_117397

/-- Represents the worker's savings scenario -/
structure WorkerSavings where
  monthly_pay : ℝ
  savings_fraction : ℝ
  savings_fraction_constant : Prop
  monthly_pay_constant : Prop
  all_savings_from_pay : Prop
  total_savings_eq_5times_unsaved : Prop

/-- Theorem stating that the savings fraction is 5/17 -/
theorem savings_fraction_is_5_17 (w : WorkerSavings) : w.savings_fraction = 5 / 17 :=
by sorry

end NUMINAMATH_CALUDE_savings_fraction_is_5_17_l1173_117397


namespace NUMINAMATH_CALUDE_expand_product_l1173_117342

theorem expand_product (x : ℝ) : -2 * (x - 3) * (x + 4) * (2*x - 1) = -4*x^3 - 2*x^2 + 50*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1173_117342


namespace NUMINAMATH_CALUDE_order_of_f_values_l1173_117384

noncomputable def f (x : ℝ) : ℝ := 2 / (4^x) - x

noncomputable def a : ℝ := 0
noncomputable def b : ℝ := Real.log 2 / Real.log 0.4
noncomputable def c : ℝ := Real.log 3 / Real.log 4

theorem order_of_f_values :
  f a < f c ∧ f c < f b := by sorry

end NUMINAMATH_CALUDE_order_of_f_values_l1173_117384


namespace NUMINAMATH_CALUDE_equation_solution_l1173_117381

theorem equation_solution : ∃ x : ℚ, (2 / 7) * (1 / 8) * x = 14 ∧ x = 392 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1173_117381


namespace NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l1173_117338

-- Define the plane α
variable (α : Set Point)

-- Define lines l and m
variable (l m : Line)

-- Define the property of being outside a plane
def OutsidePlane (line : Line) (plane : Set Point) : Prop := sorry

-- Define parallel relation between a line and a plane
def ParallelToPlane (line : Line) (plane : Set Point) : Prop := sorry

-- Define parallel relation between two lines
def ParallelLines (line1 line2 : Line) : Prop := sorry

-- Theorem statement
theorem parallel_sufficient_not_necessary
  (h1 : OutsidePlane l α)
  (h2 : OutsidePlane m α)
  (h3 : l ≠ m)
  (h4 : ParallelToPlane m α) :
  (ParallelLines l m → ParallelToPlane l α) ∧
  ¬(ParallelToPlane l α → ParallelLines l m) := by
  sorry

end NUMINAMATH_CALUDE_parallel_sufficient_not_necessary_l1173_117338


namespace NUMINAMATH_CALUDE_solution1_satisfies_system1_solution2_satisfies_system2_l1173_117357

-- Part 1
def system1 (y z : ℝ → ℝ) : Prop :=
  ∀ x, (deriv y x + 2 * y x - 4 * z x = 0) ∧
       (deriv z x + y x - 3 * z x = 3 * x^2)

def solution1 (C₁ C₂ : ℝ) (y z : ℝ → ℝ) : Prop :=
  ∀ x, y x = C₁ * Real.exp (-x) + C₂ * Real.exp (2*x) - 6*x^2 + 6*x - 9 ∧
       z x = (1/4) * C₁ * Real.exp (-x) + C₂ * Real.exp (2*x) - 3*x^2 - 3

theorem solution1_satisfies_system1 (C₁ C₂ : ℝ) :
  ∀ y z, solution1 C₁ C₂ y z → system1 y z := by sorry

-- Part 2
def system2 (u v w : ℝ → ℝ) : Prop :=
  ∀ x, (6 * deriv u x - u x - 7 * v x + 5 * w x = 10 * Real.exp x) ∧
       (2 * deriv v x + u x + v x - w x = 0) ∧
       (3 * deriv w x - u x + 2 * v x - w x = Real.exp x)

def solution2 (C₁ C₂ C₃ : ℝ) (u v w : ℝ → ℝ) : Prop :=
  ∀ x, u x = C₁ + C₂ * Real.cos x + C₃ * Real.sin x + Real.exp x ∧
       v x = 2*C₁ + (1/2)*(C₃ - C₂)*Real.cos x - (1/2)*(C₃ + C₂)*Real.sin x ∧
       w x = 3*C₁ - (1/2)*(C₂ + C₃)*Real.cos x + (1/2)*(C₂ - C₃)*Real.sin x + Real.exp x

theorem solution2_satisfies_system2 (C₁ C₂ C₃ : ℝ) :
  ∀ u v w, solution2 C₁ C₂ C₃ u v w → system2 u v w := by sorry

end NUMINAMATH_CALUDE_solution1_satisfies_system1_solution2_satisfies_system2_l1173_117357


namespace NUMINAMATH_CALUDE_unique_nonnegative_integer_solution_l1173_117302

theorem unique_nonnegative_integer_solution :
  ∃! (x y z : ℕ), 5 * x + 7 * y + 5 * z = 37 ∧ 6 * x - y - 10 * z = 3 ∧ x = 4 ∧ y = 1 ∧ z = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_nonnegative_integer_solution_l1173_117302


namespace NUMINAMATH_CALUDE_man_downstream_speed_l1173_117328

/-- Calculates the downstream speed of a man given his upstream and still water speeds -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem: Given a man's upstream speed of 25 kmph and still water speed of 45 kmph, 
    his downstream speed is 65 kmph -/
theorem man_downstream_speed :
  downstream_speed 25 45 = 65 := by
  sorry

end NUMINAMATH_CALUDE_man_downstream_speed_l1173_117328


namespace NUMINAMATH_CALUDE_fraction_simplification_l1173_117371

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1173_117371


namespace NUMINAMATH_CALUDE_ellipse_equation_l1173_117354

theorem ellipse_equation (A B C : ℝ × ℝ) (h1 : A = (-2, 0)) (h2 : B = (2, 0)) (h3 : C.1^2 + C.2^2 = 5) 
  (h4 : (C.1 - A.1) * (B.2 - A.2) = (C.2 - A.2) * (B.1 - A.1)) :
  ∃ (x y : ℝ), x^2/4 + 3*y^2/4 = 1 ∧ x^2 + y^2 = C.1^2 + C.2^2 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ellipse_equation_l1173_117354
