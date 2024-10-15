import Mathlib

namespace NUMINAMATH_CALUDE_element_14_is_si_l243_24356

/-- Represents chemical elements -/
inductive Element : Type
| helium : Element
| lithium : Element
| silicon : Element
| argon : Element

/-- Returns the atomic number of an element -/
def atomic_number (e : Element) : ℕ :=
  match e with
  | Element.helium => 2
  | Element.lithium => 3
  | Element.silicon => 14
  | Element.argon => 18

/-- Returns the symbol of an element -/
def symbol (e : Element) : String :=
  match e with
  | Element.helium => "He"
  | Element.lithium => "Li"
  | Element.silicon => "Si"
  | Element.argon => "Ar"

/-- Theorem: The symbol for the element with atomic number 14 is Si -/
theorem element_14_is_si :
  ∃ (e : Element), atomic_number e = 14 ∧ symbol e = "Si" :=
by
  sorry

end NUMINAMATH_CALUDE_element_14_is_si_l243_24356


namespace NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l243_24353

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℚ := 3 * n + 5 * n^2

/-- The rth term of the arithmetic progression -/
def a (r : ℕ) : ℚ := S r - S (r - 1)

theorem arithmetic_progression_rth_term (r : ℕ) (hr : r > 0) : a r = 10 * r - 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_rth_term_l243_24353


namespace NUMINAMATH_CALUDE_opposites_sum_l243_24391

theorem opposites_sum (a b : ℝ) : 
  (|a - 2| = -(b + 5)^2) → (a + b = -3) := by
  sorry

end NUMINAMATH_CALUDE_opposites_sum_l243_24391


namespace NUMINAMATH_CALUDE_parabola_vertex_l243_24312

/-- The vertex of the parabola y = -2(x-3)^2 - 4 is at (3, -4) -/
theorem parabola_vertex :
  let f : ℝ → ℝ := λ x => -2 * (x - 3)^2 - 4
  ∃! p : ℝ × ℝ, p.1 = 3 ∧ p.2 = -4 ∧ ∀ x : ℝ, f x ≤ f p.1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l243_24312


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l243_24333

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x^2 - 1}

-- Define set B
def B : Set ℝ := {x | |x^2 - 1| ≤ 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l243_24333


namespace NUMINAMATH_CALUDE_expression_simplification_l243_24376

theorem expression_simplification (a b : ℝ) (h1 : a = 2) (h2 : b = 3) :
  3 * a^2 * b - (a * b^2 - 2 * (2 * a^2 * b - a * b^2)) - a * b^2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l243_24376


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_17_l243_24371

theorem smallest_k_for_64_power_gt_4_power_17 : 
  (∃ k : ℕ, 64^k > 4^17 ∧ ∀ m : ℕ, m < k → 64^m ≤ 4^17) ∧ 
  (∀ k : ℕ, 64^k > 4^17 → k ≥ 6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_17_l243_24371


namespace NUMINAMATH_CALUDE_x_equation_proof_l243_24330

theorem x_equation_proof (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 5*x + 4/x + 1/x^2 = 34)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_proof_l243_24330


namespace NUMINAMATH_CALUDE_square_nonnegative_is_universal_l243_24396

/-- The proposition "The square of any real number is non-negative" -/
def square_nonnegative_prop : Prop := ∀ x : ℝ, x^2 ≥ 0

/-- Definition of a universal proposition -/
def is_universal_prop (P : Prop) : Prop := ∃ (α : Type) (Q : α → Prop), P = ∀ x : α, Q x

/-- The square_nonnegative_prop is a universal proposition -/
theorem square_nonnegative_is_universal : is_universal_prop square_nonnegative_prop := by sorry

end NUMINAMATH_CALUDE_square_nonnegative_is_universal_l243_24396


namespace NUMINAMATH_CALUDE_grocery_store_salary_l243_24309

/-- Calculates the total daily salary of all employees in a grocery store -/
def total_daily_salary (owner_salary : ℕ) (manager_salary : ℕ) (cashier_salary : ℕ) 
  (clerk_salary : ℕ) (bagger_salary : ℕ) (num_owners : ℕ) (num_managers : ℕ) 
  (num_cashiers : ℕ) (num_clerks : ℕ) (num_baggers : ℕ) : ℕ :=
  owner_salary * num_owners + manager_salary * num_managers + 
  cashier_salary * num_cashiers + clerk_salary * num_clerks + 
  bagger_salary * num_baggers

theorem grocery_store_salary : 
  total_daily_salary 20 15 10 5 3 1 3 5 7 9 = 177 := by
  sorry

end NUMINAMATH_CALUDE_grocery_store_salary_l243_24309


namespace NUMINAMATH_CALUDE_sum_of_coefficients_cubic_factorization_l243_24308

theorem sum_of_coefficients_cubic_factorization :
  ∀ (a b c d e : ℝ),
  (∀ x, 512 * x^3 + 27 = (a*x + b) * (c*x^2 + d*x + e)) →
  a + b + c + d + e = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_cubic_factorization_l243_24308


namespace NUMINAMATH_CALUDE_fred_paid_twenty_dollars_l243_24361

/-- The amount Fred paid with at the movie theater -/
def fred_payment (ticket_price : ℚ) (num_tickets : ℕ) (movie_rental : ℚ) (change : ℚ) : ℚ :=
  ticket_price * num_tickets + movie_rental + change

/-- Theorem stating the amount Fred paid with -/
theorem fred_paid_twenty_dollars :
  fred_payment (92 / 100) 2 (679 / 100) (137 / 100) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fred_paid_twenty_dollars_l243_24361


namespace NUMINAMATH_CALUDE_probability_of_two_triples_l243_24357

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (ranks : Nat)
  (cards_per_rank : Nat)
  (h_total : total_cards = ranks * cards_per_rank)

/-- Represents the specific hand we're looking for -/
structure TargetHand :=
  (total_cards : Nat)
  (sets : Nat)
  (cards_per_set : Nat)
  (h_total : total_cards = sets * cards_per_set)

def probability_of_target_hand (d : Deck) (h : TargetHand) : ℚ :=
  (d.ranks.choose h.sets) * (d.cards_per_rank.choose h.cards_per_set)^h.sets /
  d.total_cards.choose h.total_cards

theorem probability_of_two_triples (d : Deck) (h : TargetHand) :
  d.total_cards = 52 →
  d.ranks = 13 →
  d.cards_per_rank = 4 →
  h.total_cards = 6 →
  h.sets = 2 →
  h.cards_per_set = 3 →
  probability_of_target_hand d h = 13 / 106470 :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_triples_l243_24357


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l243_24320

/-- A geometric sequence with first term 1 and fourth term 1/64 has common ratio 1/4 -/
theorem geometric_sequence_common_ratio (a : ℕ → ℚ) :
  (∀ n, a (n + 1) = a n * (a 2 / a 1)) →  -- Geometric sequence condition
  a 1 = 1 →                               -- First term is 1
  a 4 = 1 / 64 →                          -- Fourth term is 1/64
  a 2 / a 1 = 1 / 4 :=                    -- Common ratio is 1/4
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l243_24320


namespace NUMINAMATH_CALUDE_circle_center_sum_l243_24398

theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 10*x - 12*y + 40) → 
  ((x - 5)^2 + (y + 6)^2 = 101) → 
  x + y = -1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l243_24398


namespace NUMINAMATH_CALUDE_min_distinct_prime_factors_l243_24344

theorem min_distinct_prime_factors (m n : ℕ) :
  ∃ (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧
  (p ∣ (m * (n + 9) * (m + 2 * n^2 + 3))) ∧
  (q ∣ (m * (n + 9) * (m + 2 * n^2 + 3))) :=
sorry

end NUMINAMATH_CALUDE_min_distinct_prime_factors_l243_24344


namespace NUMINAMATH_CALUDE_rhombus_area_fraction_l243_24315

theorem rhombus_area_fraction (grid_size : ℕ) (rhombus_side : ℝ) :
  grid_size = 7 →
  rhombus_side = Real.sqrt 2 →
  (4 * (1/2 * rhombus_side * rhombus_side)) / ((grid_size - 1)^2) = 1/18 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_area_fraction_l243_24315


namespace NUMINAMATH_CALUDE_common_factor_proof_l243_24390

theorem common_factor_proof (x y a b : ℝ) :
  ∃ (k : ℝ), 3*x*(a - b) - 9*y*(b - a) = 3*(a - b) * k :=
by sorry

end NUMINAMATH_CALUDE_common_factor_proof_l243_24390


namespace NUMINAMATH_CALUDE_binary_11010_is_26_l243_24303

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11010_is_26 :
  binary_to_decimal [false, true, false, true, true] = 26 := by
  sorry

end NUMINAMATH_CALUDE_binary_11010_is_26_l243_24303


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l243_24340

/-- Proves that a train 150 metres long running at 54 km/hr takes 10 seconds to pass a pole. -/
theorem train_passing_pole_time 
  (train_length : ℝ) 
  (train_speed_kmh : ℝ) 
  (h1 : train_length = 150) 
  (h2 : train_speed_kmh = 54) : 
  (train_length / (train_speed_kmh * 1000 / 3600)) = 10 := by
sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l243_24340


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l243_24342

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (4 + ((x^3 - 2) / x)^2) = (Real.sqrt (x^6 - 4*x^3 + 4*x^2 + 4)) / x :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l243_24342


namespace NUMINAMATH_CALUDE_f_increasing_implies_a_range_l243_24349

/-- Given a real number a, f is a function from ℝ to ℝ defined as f(x) = x^2 + 2(a - 1)x + 2 -/
def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^2 + 2*(a - 1)*x + 2

/-- The theorem states that if f is increasing on [4, +∞), then a ≥ -3 -/
theorem f_increasing_implies_a_range (a : ℝ) :
  (∀ x y, x ≥ 4 → y ≥ 4 → x ≤ y → f a x ≤ f a y) →
  a ≥ -3 :=
sorry

end NUMINAMATH_CALUDE_f_increasing_implies_a_range_l243_24349


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l243_24335

/-- Proves that mixing equal volumes of 10% and 30% alcohol solutions results in a 20% solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 200
  let y_volume : ℝ := 200
  let x_concentration : ℝ := 0.1
  let y_concentration : ℝ := 0.3
  let target_concentration : ℝ := 0.2
  x_volume * x_concentration + y_volume * y_concentration = 
    (x_volume + y_volume) * target_concentration :=
by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l243_24335


namespace NUMINAMATH_CALUDE_log_sum_equals_two_l243_24324

theorem log_sum_equals_two : Real.log 4 + Real.log 25 = 2 * Real.log 10 := by sorry

end NUMINAMATH_CALUDE_log_sum_equals_two_l243_24324


namespace NUMINAMATH_CALUDE_shannon_bought_no_gum_l243_24313

/-- Represents the purchase made by Shannon -/
structure Purchase where
  yogurt_pints : ℕ
  gum_packs : ℕ
  shrimp_trays : ℕ
  yogurt_price : ℚ
  shrimp_price : ℚ
  total_cost : ℚ

/-- The conditions of Shannon's purchase -/
def shannon_purchase : Purchase where
  yogurt_pints := 5
  gum_packs := 0  -- We'll prove this
  shrimp_trays := 5
  yogurt_price := 6  -- Derived from the total cost
  shrimp_price := 5
  total_cost := 55

/-- The price of gum is half the price of yogurt -/
def gum_price (p : Purchase) : ℚ := p.yogurt_price / 2

/-- The total cost of the purchase -/
def total_cost (p : Purchase) : ℚ :=
  p.yogurt_pints * p.yogurt_price +
  p.gum_packs * (gum_price p) +
  p.shrimp_trays * p.shrimp_price

/-- Theorem stating that Shannon bought 0 packs of gum -/
theorem shannon_bought_no_gum :
  shannon_purchase.gum_packs = 0 ∧
  total_cost shannon_purchase = shannon_purchase.total_cost := by
  sorry


end NUMINAMATH_CALUDE_shannon_bought_no_gum_l243_24313


namespace NUMINAMATH_CALUDE_square_garden_area_perimeter_relation_l243_24300

theorem square_garden_area_perimeter_relation :
  ∀ (s : ℝ), 
    s > 0 →
    4 * s = 40 →
    s^2 - 2 * (4 * s) = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_square_garden_area_perimeter_relation_l243_24300


namespace NUMINAMATH_CALUDE_cannot_determine_start_month_l243_24378

/-- Represents a month of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Represents Nolan's GRE preparation period -/
structure PreparationPeriod where
  start_month : Month
  end_month : Month
  end_day : Nat

/-- The given information about Nolan's GRE preparation -/
def nolans_preparation : PreparationPeriod :=
  { end_month := Month.August,
    end_day := 3,
    start_month := sorry }  -- We don't know the start month

/-- Theorem stating that we cannot determine Nolan's start month -/
theorem cannot_determine_start_month :
  ∀ m : Month, ∃ p : PreparationPeriod,
    p.end_month = nolans_preparation.end_month ∧
    p.end_day = nolans_preparation.end_day ∧
    p.start_month = m :=
sorry

end NUMINAMATH_CALUDE_cannot_determine_start_month_l243_24378


namespace NUMINAMATH_CALUDE_quadratic_inequality_ordering_l243_24363

-- Define the function f
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality_ordering (a b c : ℝ) :
  (∀ x, f a b c x > 0 ↔ x < -2 ∨ x > 4) →
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_ordering_l243_24363


namespace NUMINAMATH_CALUDE_cucumber_price_l243_24326

theorem cucumber_price (cucumber_price : ℝ) 
  (tomato_price_relation : cucumber_price * 0.8 = cucumber_price - cucumber_price * 0.2)
  (total_price : 2 * (cucumber_price * 0.8) + 3 * cucumber_price = 23) :
  cucumber_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_cucumber_price_l243_24326


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l243_24360

theorem quadratic_inequality_theorem (k : ℝ) : 
  (¬∃ x : ℝ, (k^2 - 1) * x^2 + 4 * (1 - k) * x + 3 ≤ 0) → 
  (k = 1 ∨ (1 < k ∧ k < 7)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l243_24360


namespace NUMINAMATH_CALUDE_function_properties_l243_24352

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (ω * x + φ)

theorem function_properties
  (ω φ : ℝ)
  (h_ω : ω > 0)
  (h_φ : -π / 2 ≤ φ ∧ φ < π / 2)
  (h_sym : ∀ x, f ω φ (2 * π / 3 - x) = f ω φ (2 * π / 3 + x))
  (h_period : ∀ x, f ω φ (x + π) = f ω φ x) :
  ω = 2 ∧
  φ = -π / 6 ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f ω φ x ≤ Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f ω φ x ≥ -Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc 0 (π / 2), f ω φ x = Real.sqrt 3) ∧
  (∃ x ∈ Set.Icc 0 (π / 2), f ω φ x = -Real.sqrt 3 / 2) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l243_24352


namespace NUMINAMATH_CALUDE_metallic_sheet_volumes_l243_24338

/-- Represents the dimensions of a rectangular sheet -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Represents the dimensions of a square -/
structure SquareDimensions where
  side : ℝ

/-- Calculates the volume of open box A -/
def volume_box_a (sheet : SheetDimensions) (corner_cut : SquareDimensions) : ℝ :=
  (sheet.length - 2 * corner_cut.side) * (sheet.width - 2 * corner_cut.side) * corner_cut.side

/-- Calculates the volume of open box B -/
def volume_box_b (sheet : SheetDimensions) (corner_cut : SquareDimensions) (middle_cut : SquareDimensions) : ℝ :=
  ((sheet.length - 2 * corner_cut.side) * (sheet.width - 2 * corner_cut.side) - middle_cut.side ^ 2) * corner_cut.side

theorem metallic_sheet_volumes
  (sheet : SheetDimensions)
  (corner_cut : SquareDimensions)
  (middle_cut : SquareDimensions)
  (h1 : sheet.length = 48)
  (h2 : sheet.width = 36)
  (h3 : corner_cut.side = 8)
  (h4 : middle_cut.side = 12) :
  volume_box_a sheet corner_cut = 5120 ∧ volume_box_b sheet corner_cut middle_cut = 3968 := by
  sorry

#eval volume_box_a ⟨48, 36⟩ ⟨8⟩
#eval volume_box_b ⟨48, 36⟩ ⟨8⟩ ⟨12⟩

end NUMINAMATH_CALUDE_metallic_sheet_volumes_l243_24338


namespace NUMINAMATH_CALUDE_complex_product_theorem_l243_24393

theorem complex_product_theorem :
  let Q : ℂ := 4 + 3 * Complex.I
  let E : ℂ := 2 * Complex.I
  let D : ℂ := 4 - 3 * Complex.I
  Q * E * D = 50 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_product_theorem_l243_24393


namespace NUMINAMATH_CALUDE_number_sum_theorem_l243_24380

theorem number_sum_theorem :
  (∀ n : ℕ, n ≥ 100 → n ≥ smallest_three_digit) ∧
  (∀ n : ℕ, n < 100 → n ≤ largest_two_digit) ∧
  (∀ n : ℕ, n < 10 ∧ n % 2 = 1 → n ≥ smallest_odd_one_digit) ∧
  (∀ n : ℕ, n < 100 ∧ n % 2 = 0 → n ≤ largest_even_two_digit) →
  smallest_three_digit + largest_two_digit = 199 ∧
  smallest_odd_one_digit + largest_even_two_digit = 99 :=
by sorry

def smallest_three_digit : ℕ := 100
def largest_two_digit : ℕ := 99
def smallest_odd_one_digit : ℕ := 1
def largest_even_two_digit : ℕ := 98

end NUMINAMATH_CALUDE_number_sum_theorem_l243_24380


namespace NUMINAMATH_CALUDE_f_formula_g_minimum_l243_24397

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 7*x + 13

-- Define the function g
def g (a x : ℝ) : ℝ := f (x + a) - 7*x

-- Theorem for part (I)
theorem f_formula (x : ℝ) : f (2*x - 3) = 4*x^2 + 2*x + 1 := by sorry

-- Theorem for part (II)
theorem g_minimum (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, g a x ≥ 
    if a ≤ -3 then a^2 + 13*a + 22
    else if a < -1 then 7*a + 13
    else a^2 + 9*a + 14) ∧
  (∃ x ∈ Set.Icc 1 3, g a x = 
    if a ≤ -3 then a^2 + 13*a + 22
    else if a < -1 then 7*a + 13
    else a^2 + 9*a + 14) := by sorry

end NUMINAMATH_CALUDE_f_formula_g_minimum_l243_24397


namespace NUMINAMATH_CALUDE_exists_multiple_with_sum_of_digits_equal_to_n_l243_24319

def sumOfDigits (m : ℕ) : ℕ :=
  if m < 10 then m else m % 10 + sumOfDigits (m / 10)

theorem exists_multiple_with_sum_of_digits_equal_to_n (n : ℕ) (hn : n > 0) :
  ∃ m : ℕ, m % n = 0 ∧ sumOfDigits m = n := by
  sorry

end NUMINAMATH_CALUDE_exists_multiple_with_sum_of_digits_equal_to_n_l243_24319


namespace NUMINAMATH_CALUDE_tangent_chord_length_l243_24354

/-- The circle with equation x^2 + y^2 - 6x - 8y + 20 = 0 -/
def Circle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + 20 = 0}

/-- The origin point (0, 0) -/
def Origin : ℝ × ℝ := (0, 0)

/-- A point is on the circle if it satisfies the circle equation -/
def IsOnCircle (p : ℝ × ℝ) : Prop := p ∈ Circle

/-- A line is tangent to the circle if it touches the circle at exactly one point -/
def IsTangentLine (l : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ), IsOnCircle p ∧ l ∩ Circle = {p}

/-- The theorem stating that the length of the chord formed by two tangent lines
    from the origin to the circle is 4√5 -/
theorem tangent_chord_length :
  ∃ (A B : ℝ × ℝ) (OA OB : Set (ℝ × ℝ)),
    IsOnCircle A ∧ IsOnCircle B ∧
    IsTangentLine OA ∧ IsTangentLine OB ∧
    Origin ∈ OA ∧ Origin ∈ OB ∧
    A ∈ OA ∧ B ∈ OB ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_chord_length_l243_24354


namespace NUMINAMATH_CALUDE_cube_root_of_sum_l243_24332

theorem cube_root_of_sum (a b : ℝ) : 
  Real.sqrt (a - 1) + Real.sqrt ((9 + b)^2) = 0 → (a + b)^(1/3 : ℝ) = -2 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_of_sum_l243_24332


namespace NUMINAMATH_CALUDE_geometric_series_second_term_l243_24317

theorem geometric_series_second_term :
  ∀ (a : ℝ) (r : ℝ) (S : ℝ),
    r = (1 : ℝ) / 4 →
    S = 40 →
    S = a / (1 - r) →
    a * r = (15 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_series_second_term_l243_24317


namespace NUMINAMATH_CALUDE_gcd_360_504_l243_24346

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_CALUDE_gcd_360_504_l243_24346


namespace NUMINAMATH_CALUDE_sum_series_equals_three_halves_l243_24310

/-- The sum of the series (4n-3)/3^n from n=1 to infinity equals 3/2 -/
theorem sum_series_equals_three_halves :
  (∑' n : ℕ, (4 * n - 3 : ℝ) / 3^n) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_series_equals_three_halves_l243_24310


namespace NUMINAMATH_CALUDE_petya_vasya_meeting_l243_24368

/-- The number of lanterns along the alley -/
def num_lanterns : ℕ := 100

/-- The position where Petya is observed -/
def petya_observed : ℕ := 22

/-- The position where Vasya is observed -/
def vasya_observed : ℕ := 88

/-- The function to calculate the meeting point of Petya and Vasya -/
def meeting_point (n l p v : ℕ) : ℕ :=
  ((n - 1) - (l - p)) + 1

theorem petya_vasya_meeting :
  meeting_point num_lanterns petya_observed vasya_observed 1 = 64 := by
  sorry

end NUMINAMATH_CALUDE_petya_vasya_meeting_l243_24368


namespace NUMINAMATH_CALUDE_percentage_of_unsold_books_l243_24374

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem percentage_of_unsold_books :
  abs (percentage_not_sold - 80.57) < 0.01 := by sorry

end NUMINAMATH_CALUDE_percentage_of_unsold_books_l243_24374


namespace NUMINAMATH_CALUDE_ratio_children_to_adults_l243_24367

def total_people : ℕ := 120
def children : ℕ := 80

theorem ratio_children_to_adults :
  (children : ℚ) / (total_people - children : ℚ) = 2 / 1 := by sorry

end NUMINAMATH_CALUDE_ratio_children_to_adults_l243_24367


namespace NUMINAMATH_CALUDE_extended_parallelepiped_volume_sum_l243_24355

/-- Represents a rectangular parallelepiped with dimensions l, w, and h -/
structure Parallelepiped where
  l : ℝ
  w : ℝ
  h : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a parallelepiped -/
def volume_extended_parallelepiped (p : Parallelepiped) : ℝ := sorry

/-- Checks if two integers are relatively prime -/
def relatively_prime (a b : ℕ) : Prop := sorry

theorem extended_parallelepiped_volume_sum (m n p : ℕ) :
  (∃ (parallelepiped : Parallelepiped),
    parallelepiped.l = 3 ∧
    parallelepiped.w = 4 ∧
    parallelepiped.h = 5 ∧
    volume_extended_parallelepiped parallelepiped = (m + n * Real.pi) / p ∧
    m > 0 ∧ n > 0 ∧ p > 0 ∧
    relatively_prime n p) →
  m + n + p = 505 := by sorry

end NUMINAMATH_CALUDE_extended_parallelepiped_volume_sum_l243_24355


namespace NUMINAMATH_CALUDE_mixture_ratio_theorem_l243_24381

/-- Represents the ratio of alcohol to water in a mixture -/
structure AlcoholRatio where
  alcohol : ℝ
  water : ℝ

/-- Calculates the ratio of alcohol to water when mixing two solutions -/
def mixSolutions (v1 v2 : ℝ) (r1 r2 : AlcoholRatio) : AlcoholRatio :=
  { alcohol := v1 * r1.alcohol + v2 * r2.alcohol,
    water := v1 * r1.water + v2 * r2.water }

theorem mixture_ratio_theorem (V p q : ℝ) (hp : p > 0) (hq : q > 0) :
  let jar1 := AlcoholRatio.mk (p / (p + 2)) (2 / (p + 2))
  let jar2 := AlcoholRatio.mk (q / (q + 1)) (1 / (q + 1))
  let mixture := mixSolutions V (2 * V) jar1 jar2
  mixture.alcohol / mixture.water = (p * (q + 1) + 4 * q * (p + 2)) / (2 * (q + 1) + 4 * (p + 2)) := by
  sorry

#check mixture_ratio_theorem

end NUMINAMATH_CALUDE_mixture_ratio_theorem_l243_24381


namespace NUMINAMATH_CALUDE_two_digit_integer_count_l243_24348

/-- A function that counts the number of three-digit integers less than 1000 with exactly two different digits. -/
def count_two_digit_integers : ℕ :=
  let case1 := 9  -- Numbers with one digit as zero
  let case2 := 9 * 9 * 3  -- Numbers with two non-zero digits
  case1 + case2

/-- Theorem stating that the count of three-digit integers less than 1000 with exactly two different digits is 252. -/
theorem two_digit_integer_count : count_two_digit_integers = 252 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_integer_count_l243_24348


namespace NUMINAMATH_CALUDE_complex_modulus_power_eight_l243_24386

theorem complex_modulus_power_eight : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3 / 2)))^8 = 1 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_power_eight_l243_24386


namespace NUMINAMATH_CALUDE_largest_solution_equation_inverse_x_12_value_l243_24318

noncomputable def largest_x : ℝ :=
  Real.exp (- (7 / 12) * Real.log 10)

theorem largest_solution_equation (x : ℝ) (h : x = largest_x) :
  (Real.log 10) / (Real.log (10 * x^2)) + (Real.log 10) / (Real.log (100 * x^3)) = -2 :=
sorry

theorem inverse_x_12_value :
  (1 : ℝ) / largest_x^12 = 10000000 :=
sorry

end NUMINAMATH_CALUDE_largest_solution_equation_inverse_x_12_value_l243_24318


namespace NUMINAMATH_CALUDE_expression_simplification_l243_24311

theorem expression_simplification (x : ℝ) : 
  ((7 * x - 3) + 3 * x * 2) * 2 + (5 + 2 * 2) * (4 * x + 6) = 62 * x + 48 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l243_24311


namespace NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l243_24373

/-- The probability of rolling an odd number on a fair 6-sided die -/
def p_odd : ℚ := 1/2

/-- The number of rolls -/
def n : ℕ := 6

/-- The number of successful outcomes (rolls with odd numbers) -/
def k : ℕ := 5

/-- The probability of getting exactly k odd numbers in n rolls of a fair 6-sided die -/
def prob_k_odd_in_n_rolls (p : ℚ) (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem prob_five_odd_in_six_rolls :
  prob_k_odd_in_n_rolls p_odd n k = 3/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_five_odd_in_six_rolls_l243_24373


namespace NUMINAMATH_CALUDE_last_digit_congruence_l243_24331

theorem last_digit_congruence (N : ℕ) : ∃ (a b : ℕ), N = 10 * a + b ∧ b < 10 →
  (N ≡ b [ZMOD 10]) ∧ (N ≡ b [ZMOD 2]) ∧ (N ≡ b [ZMOD 5]) := by
  sorry

end NUMINAMATH_CALUDE_last_digit_congruence_l243_24331


namespace NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l243_24399

/-- The volume of a sphere inscribed in a cube -/
theorem volume_of_inscribed_sphere (cube_edge : ℝ) (sphere_volume : ℝ) : 
  cube_edge = 10 →
  sphere_volume = (4 / 3) * π * (cube_edge / 2)^3 →
  sphere_volume = (500 / 3) * π :=
by sorry

end NUMINAMATH_CALUDE_volume_of_inscribed_sphere_l243_24399


namespace NUMINAMATH_CALUDE_value_of_c_l243_24375

theorem value_of_c (a b c : ℝ) (h1 : a = 6) (h2 : b = 15) (h3 : 6 * 15 * c = 3) :
  (a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) ↔ c = 3 := by
sorry

end NUMINAMATH_CALUDE_value_of_c_l243_24375


namespace NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l243_24369

/-- Calculates the alcohol percentage in a mixture after adding water --/
theorem alcohol_percentage_after_dilution
  (initial_volume : ℝ)
  (initial_alcohol_percentage : ℝ)
  (added_water : ℝ)
  (h1 : initial_volume = 11)
  (h2 : initial_alcohol_percentage = 42)
  (h3 : added_water = 3)
  : (initial_alcohol_percentage * initial_volume) / (initial_volume + added_water) = 33 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_percentage_after_dilution_l243_24369


namespace NUMINAMATH_CALUDE_f_max_min_implies_m_range_l243_24387

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem f_max_min_implies_m_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 5) ∧  -- Maximum value is 5
  (∃ x ∈ Set.Icc 0 m, f x = 5) ∧  -- Maximum value is attained
  (∀ x ∈ Set.Icc 0 m, f x ≥ 1) ∧  -- Minimum value is 1
  (∃ x ∈ Set.Icc 0 m, f x = 1) →  -- Minimum value is attained
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_f_max_min_implies_m_range_l243_24387


namespace NUMINAMATH_CALUDE_total_cows_is_570_l243_24394

/-- The number of cows owned by Matthews -/
def matthews_cows : ℕ := 60

/-- The number of cows owned by Aaron -/
def aaron_cows : ℕ := 4 * matthews_cows

/-- The number of cows owned by Marovich -/
def marovich_cows : ℕ := aaron_cows + matthews_cows - 30

/-- The total number of cows owned by all three -/
def total_cows : ℕ := aaron_cows + matthews_cows + marovich_cows

theorem total_cows_is_570 : total_cows = 570 := by
  sorry

end NUMINAMATH_CALUDE_total_cows_is_570_l243_24394


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l243_24329

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 5)
  (h2 : m*a + n*b = 5) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l243_24329


namespace NUMINAMATH_CALUDE_block_weight_difference_l243_24365

theorem block_weight_difference :
  let yellow_weight : ℝ := 0.6
  let green_weight : ℝ := 0.4
  yellow_weight - green_weight = 0.2 := by
sorry

end NUMINAMATH_CALUDE_block_weight_difference_l243_24365


namespace NUMINAMATH_CALUDE_min_point_is_correct_l243_24382

/-- The equation of the transformed graph -/
def f (x : ℝ) : ℝ := 2 * |x - 4| - 1

/-- The minimum point of the transformed graph -/
def min_point : ℝ × ℝ := (-4, -1)

/-- Theorem: The minimum point of the transformed graph is (-4, -1) -/
theorem min_point_is_correct :
  ∀ x : ℝ, f x ≥ f (min_point.1) ∧ f (min_point.1) = min_point.2 :=
by sorry

end NUMINAMATH_CALUDE_min_point_is_correct_l243_24382


namespace NUMINAMATH_CALUDE_cloud9_diving_total_money_l243_24306

/-- The total money taken by Cloud 9 Diving Company -/
theorem cloud9_diving_total_money (individual_bookings group_bookings returned : ℕ) 
  (h1 : individual_bookings = 12000)
  (h2 : group_bookings = 16000)
  (h3 : returned = 1600) :
  individual_bookings + group_bookings - returned = 26400 := by
  sorry

end NUMINAMATH_CALUDE_cloud9_diving_total_money_l243_24306


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l243_24316

theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l243_24316


namespace NUMINAMATH_CALUDE_max_value_of_function_l243_24345

theorem max_value_of_function (x : ℝ) (h : x < 0) :
  2 * x + 2 / x ≤ -4 ∧ ∃ x₀, x₀ < 0 ∧ 2 * x₀ + 2 / x₀ = -4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l243_24345


namespace NUMINAMATH_CALUDE_nice_polynomial_characterization_l243_24358

def is_nice (f : ℝ → ℝ) (A B : Finset ℝ) : Prop :=
  A.card = B.card ∧ B = A.image f

def can_produce_nice (S : ℝ → ℝ) : Prop :=
  ∀ A B : Finset ℝ, A.card = B.card → ∃ f : ℝ → ℝ, is_nice f A B

def is_polynomial (f : ℝ → ℝ) : Prop := sorry

def degree (f : ℝ → ℝ) : ℕ := sorry

def leading_coefficient (f : ℝ → ℝ) : ℝ := sorry

theorem nice_polynomial_characterization (S : ℝ → ℝ) :
  (is_polynomial S ∧ can_produce_nice S) ↔
  (is_polynomial S ∧ degree S ≥ 2 ∧
   (Even (degree S) ∨ (Odd (degree S) ∧ leading_coefficient S < 0))) :=
sorry

end NUMINAMATH_CALUDE_nice_polynomial_characterization_l243_24358


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_unique_pair_l243_24323

theorem sqrt_equality_implies_unique_pair :
  ∀ a b : ℕ,
  0 < a → 0 < b → a < b →
  (Real.sqrt (4 + Real.sqrt (36 + 24 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b) →
  a = 1 ∧ b = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_unique_pair_l243_24323


namespace NUMINAMATH_CALUDE_derivative_of_cube_root_l243_24301

theorem derivative_of_cube_root (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.sqrt (x^3)) x = (3/2) * Real.sqrt x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_cube_root_l243_24301


namespace NUMINAMATH_CALUDE_only_99th_statement_true_l243_24305

/-- Represents a statement in the notebook -/
def Statement (n : ℕ) := "There are exactly n false statements in this notebook"

/-- The total number of statements in the notebook -/
def totalStatements : ℕ := 100

/-- A function that determines if a statement is true -/
def isTrue (n : ℕ) : Prop := 
  n ≤ totalStatements ∧ (totalStatements - n) = 1

theorem only_99th_statement_true : 
  ∃! n : ℕ, n ≤ totalStatements ∧ isTrue n ∧ n = 99 := by
  sorry

#check only_99th_statement_true

end NUMINAMATH_CALUDE_only_99th_statement_true_l243_24305


namespace NUMINAMATH_CALUDE_negative_fifty_deg_same_terminal_side_as_three_hundred_ten_deg_l243_24392

-- Define the property of two angles having the same terminal side
def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

-- State the theorem
theorem negative_fifty_deg_same_terminal_side_as_three_hundred_ten_deg :
  same_terminal_side (-50) 310 := by
  sorry

end NUMINAMATH_CALUDE_negative_fifty_deg_same_terminal_side_as_three_hundred_ten_deg_l243_24392


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_l243_24383

/-- An isosceles triangle with perimeter 11 and one side length 3 -/
structure IsoscelesTriangle where
  /-- The length of two equal sides -/
  side : ℝ
  /-- The length of the base -/
  base : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : side ≥ 0 ∧ base ≥ 0
  /-- The perimeter is 11 -/
  perimeterIs11 : 2 * side + base = 11
  /-- One side length is 3 -/
  oneSideIs3 : side = 3 ∨ base = 3

/-- The base of an isosceles triangle with perimeter 11 and one side length 3 can only be 3 or 5 -/
theorem isosceles_triangle_base (t : IsoscelesTriangle) : t.base = 3 ∨ t.base = 5 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_l243_24383


namespace NUMINAMATH_CALUDE_opinion_change_difference_l243_24388

theorem opinion_change_difference (initial_like initial_dislike final_like final_dislike : ℝ) :
  initial_like = 40 →
  initial_dislike = 60 →
  final_like = 80 →
  final_dislike = 20 →
  initial_like + initial_dislike = 100 →
  final_like + final_dislike = 100 →
  let min_change := |final_like - initial_like|
  let max_change := min initial_like initial_dislike + min final_like final_dislike
  max_change - min_change = 60 := by
sorry

end NUMINAMATH_CALUDE_opinion_change_difference_l243_24388


namespace NUMINAMATH_CALUDE_cassidy_grounding_period_l243_24341

/-- Calculates the total grounding period for Cassidy based on her grades and volunteering. -/
def calculate_grounding_period (
  initial_grounding : ℕ
  ) (extra_days_per_grade : ℕ
  ) (grades_below_b : ℕ
  ) (extracurricular_below_b : ℕ
  ) (volunteering_reduction : ℕ
  ) : ℕ :=
  let subject_penalty := grades_below_b * extra_days_per_grade
  let extracurricular_penalty := extracurricular_below_b * (extra_days_per_grade / 2)
  let total_before_volunteering := initial_grounding + subject_penalty + extracurricular_penalty
  total_before_volunteering - volunteering_reduction

/-- Theorem stating that Cassidy's total grounding period is 27 days. -/
theorem cassidy_grounding_period :
  calculate_grounding_period 14 3 4 2 2 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cassidy_grounding_period_l243_24341


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l243_24304

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a ternary number represented as a list of trits to its decimal equivalent -/
def ternary_to_decimal (trits : List ℕ) : ℕ :=
  trits.foldr (fun t n => 3 * n + t) 0

theorem product_of_binary_and_ternary :
  let binary_num := [true, false, true, true]  -- 1011 in binary
  let ternary_num := [1, 1, 1]  -- 111 in ternary
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 143 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l243_24304


namespace NUMINAMATH_CALUDE_smallest_number_in_special_triple_l243_24372

theorem smallest_number_in_special_triple : 
  ∀ (a b c : ℕ), 
    (a > 0 ∧ b > 0 ∧ c > 0) →  -- Three positive integers
    ((a + b + c) / 3 : ℚ) = 30 →  -- Arithmetic mean is 30
    b = 29 →  -- Median is 29
    max a (max b c) = b + 4 →  -- Median is 4 less than the largest number
    min a (min b c) = 28 :=  -- The smallest number is 28
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_in_special_triple_l243_24372


namespace NUMINAMATH_CALUDE_square_difference_divided_by_nine_l243_24336

theorem square_difference_divided_by_nine : (121^2 - 112^2) / 9 = 233 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_nine_l243_24336


namespace NUMINAMATH_CALUDE_max_gcd_13n_plus_3_7n_plus_1_l243_24321

theorem max_gcd_13n_plus_3_7n_plus_1 :
  (∃ (n : ℕ+), Nat.gcd (13 * n + 3) (7 * n + 1) = 8) ∧
  (∀ (n : ℕ+), Nat.gcd (13 * n + 3) (7 * n + 1) ≤ 8) := by sorry

end NUMINAMATH_CALUDE_max_gcd_13n_plus_3_7n_plus_1_l243_24321


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l243_24327

theorem diophantine_equation_solutions :
  ∀ a b : ℕ, 3 * 2^a + 1 = b^2 ↔ (a = 0 ∧ b = 2) ∨ (a = 3 ∧ b = 5) ∨ (a = 4 ∧ b = 7) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l243_24327


namespace NUMINAMATH_CALUDE_complex_number_properties_l243_24395

/-- Given a complex number z = (a+i)(1-i)+bi where a and b are real, and the point
    corresponding to z in the complex plane lies on the graph of y = x - 3 -/
theorem complex_number_properties (a b : ℝ) (z : ℂ) 
  (h1 : z = (a + Complex.I) * (1 - Complex.I) + b * Complex.I)
  (h2 : z.im = z.re - 3) : 
  (2 * a > b) ∧ (Complex.abs z ≥ 3 * Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l243_24395


namespace NUMINAMATH_CALUDE_inequality_transformation_l243_24307

theorem inequality_transformation (a b c : ℝ) :
  (b / (a^2 + 1) > c / (a^2 + 1)) → b > c := by sorry

end NUMINAMATH_CALUDE_inequality_transformation_l243_24307


namespace NUMINAMATH_CALUDE_power_mean_inequality_l243_24302

theorem power_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_power_mean_inequality_l243_24302


namespace NUMINAMATH_CALUDE_translation_result_l243_24343

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define the translation operation
def translate (p : Point) (dx dy : ℝ) : Point :=
  (p.1 + dx, p.2 + dy)

-- Theorem statement
theorem translation_result :
  let A : Point := (-1, 4)
  let B : Point := translate A 5 3
  B = (4, 7) := by sorry

end NUMINAMATH_CALUDE_translation_result_l243_24343


namespace NUMINAMATH_CALUDE_intersection_y_intercept_l243_24362

/-- Given two lines that intersect at a specific x-coordinate, 
    prove that the y-intercept of the first line has a specific value. -/
theorem intersection_y_intercept (m : ℝ) : 
  (∃ y : ℝ, 3 * (-6.7) + y = m ∧ -0.5 * (-6.7) + y = 20) → 
  m = -3.45 := by
sorry

end NUMINAMATH_CALUDE_intersection_y_intercept_l243_24362


namespace NUMINAMATH_CALUDE_parallel_unit_vectors_l243_24366

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def is_unit_vector (v : E) : Prop := ‖v‖ = 1

def are_parallel (v w : E) : Prop := ∃ (k : ℝ), v = k • w

theorem parallel_unit_vectors (a b : E) 
  (ha : is_unit_vector a) (hb : is_unit_vector b) (hpar : are_parallel a b) : 
  a = b ∨ a = -b := by
  sorry

end NUMINAMATH_CALUDE_parallel_unit_vectors_l243_24366


namespace NUMINAMATH_CALUDE_line_intersects_circle_l243_24337

/-- The line (x-1)a + y = 1 always intersects the circle x^2 + y^2 = 3 for any real value of a -/
theorem line_intersects_circle (a : ℝ) : ∃ (x y : ℝ), 
  ((x - 1) * a + y = 1) ∧ (x^2 + y^2 = 3) := by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l243_24337


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_3_199_4_l243_24322

/-- Calculates the number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (first last commonDiff : ℕ) : ℕ :=
  (last - first) / commonDiff + 1

/-- Theorem: The arithmetic sequence starting with 3, ending with 199,
    and having a common difference of 4 contains exactly 50 terms -/
theorem arithmetic_sequence_length_3_199_4 :
  arithmeticSequenceLength 3 199 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_3_199_4_l243_24322


namespace NUMINAMATH_CALUDE_extremum_of_f_l243_24364

/-- The function f(x) = (k-1)x^2 - 2(k-1)x - k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k - 1) * x^2 - 2 * (k - 1) * x - k

/-- Theorem: Extremum of f(x) when k ≠ 1 -/
theorem extremum_of_f (k : ℝ) (h : k ≠ 1) :
  (k > 1 → ∀ x, f k x ≥ -2 * k + 1) ∧
  (k < 1 → ∀ x, f k x ≤ -2 * k + 1) := by
  sorry

end NUMINAMATH_CALUDE_extremum_of_f_l243_24364


namespace NUMINAMATH_CALUDE_dice_throw_probability_l243_24384

theorem dice_throw_probability (n : ℕ) : 
  (1 / 2 : ℚ) ^ n = 1 / 4 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_dice_throw_probability_l243_24384


namespace NUMINAMATH_CALUDE_floor_abs_negative_l243_24377

theorem floor_abs_negative : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end NUMINAMATH_CALUDE_floor_abs_negative_l243_24377


namespace NUMINAMATH_CALUDE_seven_digit_numbers_even_together_even_odd_together_l243_24370

/-- The number of even digits from 1 to 9 -/
def num_even_digits : ℕ := 4

/-- The number of odd digits from 1 to 9 -/
def num_odd_digits : ℕ := 5

/-- The number of even digits to be selected -/
def num_even_selected : ℕ := 3

/-- The number of odd digits to be selected -/
def num_odd_selected : ℕ := 4

/-- The total number of digits to be selected -/
def total_selected : ℕ := num_even_selected + num_odd_selected

theorem seven_digit_numbers (n : ℕ) :
  (n = Nat.choose num_even_digits num_even_selected * 
       Nat.choose num_odd_digits num_odd_selected * 
       Nat.factorial total_selected) → 
  n = 100800 := by sorry

theorem even_together (n : ℕ) :
  (n = Nat.choose num_even_digits num_even_selected * 
       Nat.choose num_odd_digits num_odd_selected * 
       Nat.factorial (total_selected - num_even_selected + 1) * 
       Nat.factorial num_even_selected) → 
  n = 14400 := by sorry

theorem even_odd_together (n : ℕ) :
  (n = Nat.choose num_even_digits num_even_selected * 
       Nat.choose num_odd_digits num_odd_selected * 
       Nat.factorial num_even_selected * 
       Nat.factorial num_odd_selected * 
       Nat.factorial 2) → 
  n = 5760 := by sorry

end NUMINAMATH_CALUDE_seven_digit_numbers_even_together_even_odd_together_l243_24370


namespace NUMINAMATH_CALUDE_competition_result_l243_24379

def math_competition (sammy_score : ℕ) (opponent_score : ℕ) : Prop :=
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let total_score := sammy_score + gab_score + cher_score
  total_score - opponent_score = 55

theorem competition_result : math_competition 20 85 := by
  sorry

end NUMINAMATH_CALUDE_competition_result_l243_24379


namespace NUMINAMATH_CALUDE_repeating_decimal_36_equals_4_11_l243_24328

/-- The decimal expansion 0.363636... (infinitely repeating 36) is equal to 4/11 -/
theorem repeating_decimal_36_equals_4_11 : ∃ (x : ℚ), x = 4/11 ∧ x = ∑' n, 36 / (100 ^ (n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_36_equals_4_11_l243_24328


namespace NUMINAMATH_CALUDE_owen_final_turtles_l243_24325

def turtles_problem (owen_initial johanna_initial owen_after_month johanna_after_month owen_final : ℕ) : Prop :=
  (johanna_initial = owen_initial - 5) ∧
  (owen_after_month = 2 * owen_initial) ∧
  (johanna_after_month = johanna_initial / 2) ∧
  (owen_final = owen_after_month + johanna_after_month)

theorem owen_final_turtles :
  ∃ (owen_initial johanna_initial owen_after_month johanna_after_month owen_final : ℕ),
    turtles_problem owen_initial johanna_initial owen_after_month johanna_after_month owen_final ∧
    owen_initial = 21 ∧
    owen_final = 50 :=
by sorry

end NUMINAMATH_CALUDE_owen_final_turtles_l243_24325


namespace NUMINAMATH_CALUDE_function_property_l243_24359

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def hasPeriod (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem function_property (f : ℝ → ℝ) 
  (h_even : isEven f)
  (h_period : hasPeriod f 2)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| := by
sorry

end NUMINAMATH_CALUDE_function_property_l243_24359


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l243_24351

/-- The surface area of a sphere circumscribing a regular square pyramid -/
theorem circumscribed_sphere_surface_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (h_base : base_edge = 2)
  (h_lateral : lateral_edge = Real.sqrt 3)
  : (4 : ℝ) * Real.pi * ((3 : ℝ) / 2) ^ 2 = 9 * Real.pi :=
sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l243_24351


namespace NUMINAMATH_CALUDE_ned_lost_lives_l243_24350

/-- Proves that Ned lost 13 lives in a video game -/
theorem ned_lost_lives (initial_lives current_lives : ℕ) 
  (h1 : initial_lives = 83) 
  (h2 : current_lives = 70) : 
  initial_lives - current_lives = 13 := by
  sorry

end NUMINAMATH_CALUDE_ned_lost_lives_l243_24350


namespace NUMINAMATH_CALUDE_gcd_7254_156_minus_10_l243_24314

theorem gcd_7254_156_minus_10 : Nat.gcd 7254 156 - 10 = 68 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7254_156_minus_10_l243_24314


namespace NUMINAMATH_CALUDE_range_of_m_l243_24339

theorem range_of_m (m : ℝ) : 
  (¬∀ (x : ℝ), m * x^2 - 2 * m * x + 1 > 0) → (m < 0 ∨ m ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l243_24339


namespace NUMINAMATH_CALUDE_group_5_frequency_l243_24385

theorem group_5_frequency (total : ℕ) (group1 group2 group3 group4 : ℕ) 
  (h_total : total = 50)
  (h_group1 : group1 = 2)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 15)
  (h_group4 : group4 = 5) :
  (total - group1 - group2 - group3 - group4 : ℚ) / total = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_group_5_frequency_l243_24385


namespace NUMINAMATH_CALUDE_hyperbola_properties_l243_24347

/-- Given a hyperbola C with equation 9y^2 - 16x^2 = 144 -/
def hyperbola_C (x y : ℝ) : Prop := 9 * y^2 - 16 * x^2 = 144

/-- Point P -/
def point_P : ℝ × ℝ := (6, 4)

/-- Theorem stating properties of hyperbola C and a related hyperbola -/
theorem hyperbola_properties :
  ∃ (a b c : ℝ),
    /- Transverse axis length -/
    2 * a = 8 ∧
    /- Conjugate axis length -/
    2 * b = 6 ∧
    /- Foci coordinates -/
    (∀ (x y : ℝ), hyperbola_C x y → (x = 0 ∧ (y = c ∨ y = -c))) ∧
    /- Eccentricity -/
    c / a = 5 / 4 ∧
    /- New hyperbola equation -/
    (∀ (x y : ℝ), x^2 / 27 - y^2 / 48 = 1 →
      /- Same asymptotes as C -/
      (∃ (k : ℝ), k ≠ 0 ∧ 9 * y^2 - 16 * x^2 = 144 * k) ∧
      /- Passes through point P -/
      (let (px, py) := point_P; x = px ∧ y = py)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l243_24347


namespace NUMINAMATH_CALUDE_function_properties_l243_24389

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

def has_period (f : ℝ → ℝ) (T : ℝ) : Prop := ∀ x, f (x + T) = f x

theorem function_properties (f : ℝ → ℝ) :
  (is_even f ∧ symmetric_about f 1 → has_period f 2) ∧
  (symmetric_about f 1 ∧ has_period f 2 → is_even f) ∧
  (is_even f ∧ has_period f 2 → symmetric_about f 1) →
  is_even f ∧ symmetric_about f 1 ∧ has_period f 2 :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l243_24389


namespace NUMINAMATH_CALUDE_total_players_l243_24334

theorem total_players (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) : 
  kabadi = 10 → kho_kho_only = 35 → both = 5 → kabadi + kho_kho_only - both = 40 := by
  sorry

end NUMINAMATH_CALUDE_total_players_l243_24334
