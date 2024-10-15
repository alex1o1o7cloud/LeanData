import Mathlib

namespace NUMINAMATH_CALUDE_no_negative_roots_l3183_318396

theorem no_negative_roots :
  ∀ x : ℝ, x < 0 → x^4 - 3*x^3 - 2*x^2 - 4*x + 1 ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_negative_roots_l3183_318396


namespace NUMINAMATH_CALUDE_total_timeout_time_is_185_l3183_318386

/-- Calculates the total time spent in time-out given the number of running time-outs and the duration of each time-out. -/
def total_timeout_time (running_timeouts : ℕ) (timeout_duration : ℕ) : ℕ :=
  let food_throwing_timeouts := 5 * running_timeouts - 1
  let swearing_timeouts := food_throwing_timeouts / 3
  let total_timeouts := running_timeouts + food_throwing_timeouts + swearing_timeouts
  total_timeouts * timeout_duration

/-- Proves that the total time spent in time-out is 185 minutes given the specified conditions. -/
theorem total_timeout_time_is_185 : 
  total_timeout_time 5 5 = 185 := by
  sorry

#eval total_timeout_time 5 5

end NUMINAMATH_CALUDE_total_timeout_time_is_185_l3183_318386


namespace NUMINAMATH_CALUDE_magnitude_of_z_l3183_318350

/-- The complex number z defined as 1 + 2i + i^3 -/
def z : ℂ := 1 + 2 * Complex.I + Complex.I ^ 3

/-- Theorem stating that the magnitude of z is √2 -/
theorem magnitude_of_z : Complex.abs z = Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l3183_318350


namespace NUMINAMATH_CALUDE_linear_function_solution_l3183_318308

/-- A linear function passing through (0,2) with negative slope -/
def linearFunction (k : ℝ) (x : ℝ) : ℝ := k * x + 2

theorem linear_function_solution :
  ∀ k : ℝ, k < 0 → linearFunction (-1) = linearFunction k := by sorry

end NUMINAMATH_CALUDE_linear_function_solution_l3183_318308


namespace NUMINAMATH_CALUDE_replacement_concentration_l3183_318387

/-- Proves that the concentration of the replacing solution is 25% given the initial and final conditions --/
theorem replacement_concentration (initial_concentration : ℝ) (final_concentration : ℝ) (replaced_fraction : ℝ) :
  initial_concentration = 0.40 →
  final_concentration = 0.35 →
  replaced_fraction = 1/3 →
  (1 - replaced_fraction) * initial_concentration + replaced_fraction * final_concentration = final_concentration →
  replaced_fraction * (final_concentration - initial_concentration) / replaced_fraction = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_replacement_concentration_l3183_318387


namespace NUMINAMATH_CALUDE_blue_marble_probability_l3183_318303

theorem blue_marble_probability
  (total_marbles : ℕ)
  (red_prob : ℚ)
  (h1 : total_marbles = 30)
  (h2 : red_prob = 32/75) :
  ∃ (x y : ℕ) (r1 r2 : ℕ),
    x + y = total_marbles ∧
    (r1 : ℚ) * r2 / (x * y) = red_prob ∧
    ((x - r1) : ℚ) * (y - r2) / (x * y) = 3/25 := by
  sorry

#eval 3 + 25  -- Expected output: 28

end NUMINAMATH_CALUDE_blue_marble_probability_l3183_318303


namespace NUMINAMATH_CALUDE_least_distinct_values_l3183_318310

theorem least_distinct_values (n : ℕ) (mode_freq : ℕ) (list_size : ℕ) 
  (h1 : n > 0)
  (h2 : mode_freq = 13)
  (h3 : list_size = 2023) :
  (∃ (list : List ℕ),
    list.length = list_size ∧
    (∃ (mode : ℕ), list.count mode = mode_freq ∧
      ∀ x : ℕ, x ≠ mode → list.count x < mode_freq) ∧
    (∀ m : ℕ, m < n → ¬∃ (list' : List ℕ),
      list'.length = list_size ∧
      (∃ (mode' : ℕ), list'.count mode' = mode_freq ∧
        ∀ x : ℕ, x ≠ mode' → list'.count x < mode_freq) ∧
      list'.toFinset.card = m)) →
  n = 169 := by
sorry

end NUMINAMATH_CALUDE_least_distinct_values_l3183_318310


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_terms_l3183_318338

/-- The sum of the first n terms of a geometric sequence -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: The number of terms in the geometric sequence with first term 1/3 and common ratio 1/2, 
    whose sum equals 80/243, is 4 -/
theorem geometric_sequence_sum_terms : 
  ∃ (n : ℕ), n = 4 ∧ geometricSum (1/3) (1/2) n = 80/243 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_terms_l3183_318338


namespace NUMINAMATH_CALUDE_circle_tangent_and_intersections_l3183_318384

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*Real.sqrt 3*y + 3 = 0

-- Define point A
def A : ℝ × ℝ := (-1, 0)

-- Define line l₁
def l₁ (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define line l₂
def l₂ (x : ℝ) : Prop := x = 1

-- Define the condition for points R, M, and N
def RMN_condition (R M N : ℝ × ℝ) : Prop :=
  C R.1 R.2 → l₂ M.1 → l₂ N.1 → 
  (R.1 - N.1)^2 + (R.2 - N.2)^2 = 3 * ((R.1 - M.1)^2 + (R.2 - M.2)^2)

-- Main theorem
theorem circle_tangent_and_intersections :
  -- Length of tangent line from A to C is √6
  (∃ T : ℝ × ℝ, C T.1 T.2 ∧ (T.1 - A.1)^2 + (T.2 - A.2)^2 = 6) ∧
  -- Slope k of l₁ satisfies k = √3/3 or k = 11√3/15
  (∃ k : ℝ, (k = Real.sqrt 3 / 3 ∨ k = 11 * Real.sqrt 3 / 15) ∧
    ∃ P Q : ℝ × ℝ, C P.1 P.2 ∧ C Q.1 Q.2 ∧ l₁ k P.1 P.2 ∧ l₁ k Q.1 Q.2 ∧
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = (Q.1 - P.1)^2 + (Q.2 - P.2)^2) ∧
  -- Coordinates of M and N
  ((∃ M N : ℝ × ℝ, M = (1, 4 * Real.sqrt 3 / 3) ∧ N = (1, 2 * Real.sqrt 3) ∧
    ∀ R : ℝ × ℝ, RMN_condition R M N) ∨
   (∃ M N : ℝ × ℝ, M = (1, 2 * Real.sqrt 3 / 3) ∧ N = (1, 0) ∧
    ∀ R : ℝ × ℝ, RMN_condition R M N)) :=
by sorry

end NUMINAMATH_CALUDE_circle_tangent_and_intersections_l3183_318384


namespace NUMINAMATH_CALUDE_matt_work_time_l3183_318388

/-- The number of minutes Matt worked on Monday -/
def monday_minutes : ℕ := 450

/-- The number of minutes Matt worked on Tuesday -/
def tuesday_minutes : ℕ := monday_minutes / 2

/-- The additional minutes Matt worked on the certain day compared to Tuesday -/
def additional_minutes : ℕ := 75

/-- The number of minutes Matt worked on the certain day -/
def certain_day_minutes : ℕ := tuesday_minutes + additional_minutes

theorem matt_work_time : certain_day_minutes = 300 := by
  sorry

end NUMINAMATH_CALUDE_matt_work_time_l3183_318388


namespace NUMINAMATH_CALUDE_mn_value_l3183_318335

theorem mn_value (M N : ℝ) 
  (h1 : (Real.log N) / (2 * Real.log M) = 2 * Real.log M / Real.log N)
  (h2 : M ≠ N)
  (h3 : M * N > 0)
  (h4 : M ≠ 1)
  (h5 : N ≠ 1) :
  M * N = Real.sqrt N :=
by sorry

end NUMINAMATH_CALUDE_mn_value_l3183_318335


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3183_318327

theorem partial_fraction_decomposition (A B C : ℚ) :
  (∀ x : ℚ, x^2 - 20 = A*(x+2)*(x-3) + B*(x-2)*(x-3) + C*(x-2)*(x+2)) →
  A * B * C = 2816 / 35 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3183_318327


namespace NUMINAMATH_CALUDE_thirty_percent_less_than_80_l3183_318363

theorem thirty_percent_less_than_80 : ∃ x : ℝ, (80 - 0.3 * 80) = x + 0.25 * x ∧ x = 45 := by
  sorry

end NUMINAMATH_CALUDE_thirty_percent_less_than_80_l3183_318363


namespace NUMINAMATH_CALUDE_product_base8_units_digit_l3183_318389

theorem product_base8_units_digit (a b : ℕ) (ha : a = 256) (hb : b = 72) :
  (a * b) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_base8_units_digit_l3183_318389


namespace NUMINAMATH_CALUDE_gravel_path_width_is_quarter_length_l3183_318381

/-- Represents a rectangular garden with a rose garden and gravel path. -/
structure RectangularGarden where
  length : ℝ
  width : ℝ
  roseGardenArea : ℝ
  gravelPathWidth : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  roseGarden_half : roseGardenArea = (length * width) / 2
  gravelPath_constant : gravelPathWidth > 0

/-- Theorem stating that the gravel path width is one-fourth of the garden length. -/
theorem gravel_path_width_is_quarter_length (garden : RectangularGarden) :
  garden.gravelPathWidth = garden.length / 4 := by
  sorry

end NUMINAMATH_CALUDE_gravel_path_width_is_quarter_length_l3183_318381


namespace NUMINAMATH_CALUDE_cannot_retile_after_replacement_l3183_318392

-- Define a type for tiles
inductive Tile
| OneByFour : Tile
| TwoByTwo : Tile

-- Define a type for a tiling of a rectangle
structure Tiling :=
  (width : ℕ)
  (height : ℕ)
  (tiles : List Tile)

-- Define a function to check if a tiling is valid
def isValidTiling (t : Tiling) : Prop :=
  -- Add conditions for a valid tiling
  sorry

-- Define a function to replace one 2×2 tile with a 1×4 tile
def replaceTile (t : Tiling) : Tiling :=
  -- Implement the replacement logic
  sorry

-- Theorem statement
theorem cannot_retile_after_replacement (t : Tiling) :
  isValidTiling t → ¬(isValidTiling (replaceTile t)) :=
by
  sorry

end NUMINAMATH_CALUDE_cannot_retile_after_replacement_l3183_318392


namespace NUMINAMATH_CALUDE_system_integer_solutions_l3183_318361

theorem system_integer_solutions (a b c d : ℤ) :
  (∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) →
  (a * d - b * c = 1 ∨ a * d - b * c = -1) :=
sorry

end NUMINAMATH_CALUDE_system_integer_solutions_l3183_318361


namespace NUMINAMATH_CALUDE_tangent_line_equation_l3183_318383

/-- The equation of the curve -/
def f (x : ℝ) : ℝ := 3*x - 2*x^3

/-- The derivative of the curve -/
def f' (x : ℝ) : ℝ := 3 - 6*x^2

/-- The x-coordinate of the point of tangency -/
def a : ℝ := -1

/-- Theorem: The equation of the tangent line to y = 3x - 2x^3 at x = -1 is 3x + y + 4 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, (y - f a) = f' a * (x - a) ↔ 3*x + y + 4 = 0 := by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l3183_318383


namespace NUMINAMATH_CALUDE_alcohol_concentration_reduction_specific_alcohol_reduction_l3183_318315

/-- Calculates the percentage reduction in alcohol concentration when water is added to an alcohol solution. -/
theorem alcohol_concentration_reduction 
  (initial_volume : ℝ) 
  (initial_concentration : ℝ) 
  (water_added : ℝ) : ℝ :=
  let initial_alcohol := initial_volume * initial_concentration
  let final_volume := initial_volume + water_added
  let final_concentration := initial_alcohol / final_volume
  let reduction := (initial_concentration - final_concentration) / initial_concentration * 100
  by
    -- Proof goes here
    sorry

/-- The specific case of adding 26 liters of water to 14 liters of 20% alcohol solution results in a 65% reduction in concentration. -/
theorem specific_alcohol_reduction : 
  alcohol_concentration_reduction 14 0.20 26 = 65 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_alcohol_concentration_reduction_specific_alcohol_reduction_l3183_318315


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3183_318343

theorem arithmetic_square_root_of_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_sqrt_16_l3183_318343


namespace NUMINAMATH_CALUDE_vieta_relation_l3183_318360

/-- The quadratic equation x^2 - x - 1 = 0 --/
def quadratic_equation (x : ℝ) : Prop := x^2 - x - 1 = 0

/-- Definition of S_n --/
def S (n : ℕ) (M N : ℝ) : ℝ := M^n + N^n

/-- Theorem: Relationship between S_n, S_{n-1}, and S_{n-2} --/
theorem vieta_relation (M N : ℝ) (h : quadratic_equation M ∧ quadratic_equation N) :
  ∀ n ≥ 3, S n M N = S (n-1) M N + S (n-2) M N :=
sorry

end NUMINAMATH_CALUDE_vieta_relation_l3183_318360


namespace NUMINAMATH_CALUDE_shopping_solution_l3183_318345

/-- Represents the shopping problem with given prices, discounts, and taxes -/
def shopping_problem (initial_amount : ℝ) 
  (milk_price bread_price detergent_price banana_price_per_pound egg_price chicken_price apple_price : ℝ)
  (detergent_discount chicken_discount loyalty_discount milk_discount bread_discount : ℝ)
  (sales_tax : ℝ) : Prop :=
  let discounted_milk := milk_price * (1 - milk_discount)
  let discounted_bread := bread_price * (1 + 0.5) -- Buy one get one 50% off
  let discounted_detergent := detergent_price - detergent_discount
  let banana_total := banana_price_per_pound * 3
  let discounted_chicken := chicken_price * (1 - chicken_discount)
  let subtotal := discounted_milk + discounted_bread + discounted_detergent + banana_total + 
                  egg_price + discounted_chicken + apple_price
  let loyalty_discounted := subtotal * (1 - loyalty_discount)
  let total_with_tax := loyalty_discounted * (1 + sales_tax)
  initial_amount - total_with_tax = 38.25

/-- Theorem stating the solution to the shopping problem -/
theorem shopping_solution : 
  shopping_problem 75 3.80 4.25 11.50 0.95 2.80 8.45 6.30 2 0.20 0.10 0.15 0.50 0.08 := by
  sorry

end NUMINAMATH_CALUDE_shopping_solution_l3183_318345


namespace NUMINAMATH_CALUDE_triangle_heights_theorem_l3183_318330

/-- A triangle with given heights -/
structure Triangle where
  ha : ℝ
  hb : ℝ
  hc : ℝ

/-- Definition of an acute triangle based on heights -/
def is_acute (t : Triangle) : Prop :=
  t.ha > 0 ∧ t.hb > 0 ∧ t.hc > 0 ∧ t.ha ≠ t.hb ∧ t.hb ≠ t.hc ∧ t.ha ≠ t.hc

/-- Definition of triangle existence based on heights -/
def triangle_exists (t : Triangle) : Prop :=
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    t.ha = (2 * (a * b * c) / (a * b + b * c + c * a)) / a ∧
    t.hb = (2 * (a * b * c) / (a * b + b * c + c * a)) / b ∧
    t.hc = (2 * (a * b * c) / (a * b + b * c + c * a)) / c

theorem triangle_heights_theorem :
  (let t1 : Triangle := ⟨4, 5, 6⟩
   is_acute t1) ∧
  (let t2 : Triangle := ⟨2, 3, 6⟩
   ¬ triangle_exists t2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_heights_theorem_l3183_318330


namespace NUMINAMATH_CALUDE_tank_capacity_proof_l3183_318370

theorem tank_capacity_proof (T : ℚ) 
  (h1 : (5/8 : ℚ) * T + 15 = (4/5 : ℚ) * T) : 
  T = 86 := by
sorry

end NUMINAMATH_CALUDE_tank_capacity_proof_l3183_318370


namespace NUMINAMATH_CALUDE_circle_properties_l3183_318342

/-- A circle described by the equation x^2 + y^2 + 2ax - 2ay = 0 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*a*p.1 - 2*a*p.2 = 0}

/-- The line x + y = 0 -/
def SymmetryLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 0}

theorem circle_properties (a : ℝ) :
  (∀ (x y : ℝ), (x, y) ∈ Circle a ↔ (-y, -x) ∈ Circle a) ∧ 
  (0, 0) ∈ Circle a := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l3183_318342


namespace NUMINAMATH_CALUDE_price_increase_percentage_l3183_318357

theorem price_increase_percentage (lower_price higher_price : ℝ) 
  (h1 : lower_price > 0)
  (h2 : higher_price > lower_price)
  (h3 : higher_price = lower_price * 1.4) :
  (higher_price - lower_price) / lower_price * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l3183_318357


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l3183_318371

/-- A parabola with its focus on the line 3x - 4y - 12 = 0 -/
structure Parabola where
  focus : ℝ × ℝ
  focus_on_line : 3 * focus.1 - 4 * focus.2 - 12 = 0

/-- The standard equation of a parabola -/
inductive StandardEquation
  | VerticalAxis (p : ℝ) : StandardEquation  -- y² = 4px
  | HorizontalAxis (p : ℝ) : StandardEquation  -- x² = 4py

theorem parabola_standard_equation (p : Parabola) :
  (∃ (eq : StandardEquation), eq = StandardEquation.VerticalAxis 4 ∨ eq = StandardEquation.HorizontalAxis (-3)) :=
sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l3183_318371


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_exists_x0_negation_is_false_l3183_318359

-- Define the necessary condition
def necessary_condition (a b : ℝ) : Prop := a + b > 4

-- Define the stronger condition
def stronger_condition (a b : ℝ) : Prop := a > 2 ∧ b > 2

-- Statement 1: Necessary but not sufficient condition
theorem necessary_not_sufficient :
  (∀ a b : ℝ, stronger_condition a b → necessary_condition a b) ∧
  (∃ a b : ℝ, necessary_condition a b ∧ ¬stronger_condition a b) := by sorry

-- Statement 2: Existence of x₀
theorem exists_x0 : ∃ x₀ : ℝ, x₀^2 - x₀ > 0 := by sorry

-- Statement 3: Negation is false
theorem negation_is_false : ¬(∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_exists_x0_negation_is_false_l3183_318359


namespace NUMINAMATH_CALUDE_scooter_initial_value_l3183_318317

/-- 
Given a scooter whose value depreciates to 3/4 of its value each year, 
if its value after one year is 30000, then its initial value was 40000.
-/
theorem scooter_initial_value (initial_value : ℝ) : 
  (3 / 4 : ℝ) * initial_value = 30000 → initial_value = 40000 := by
  sorry

end NUMINAMATH_CALUDE_scooter_initial_value_l3183_318317


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3183_318352

theorem polynomial_factorization (x : ℝ) :
  x^2 + 6*x + 9 - 64*x^4 = (-8*x^2 + x + 3) * (8*x^2 + x + 3) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3183_318352


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l3183_318304

theorem quadratic_equation_condition (a : ℝ) : 
  (∀ x, (a - 3) * x^2 - 4*x + 1 = 0 → (a - 3) ≠ 0) ↔ a ≠ 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l3183_318304


namespace NUMINAMATH_CALUDE_proportion_problem_l3183_318334

theorem proportion_problem :
  ∀ x₁ x₂ x₃ x₄ : ℤ,
    (x₁ : ℚ) / x₂ = (x₃ : ℚ) / x₄ ∧
    x₁ = x₂ + 6 ∧
    x₃ = x₄ + 5 ∧
    x₁^2 + x₂^2 + x₃^2 + x₄^2 = 793 →
    ((x₁ = -12 ∧ x₂ = -18 ∧ x₃ = -10 ∧ x₄ = -15) ∨
     (x₁ = 18 ∧ x₂ = 12 ∧ x₃ = 15 ∧ x₄ = 10)) :=
by sorry

end NUMINAMATH_CALUDE_proportion_problem_l3183_318334


namespace NUMINAMATH_CALUDE_ace_ten_jack_prob_is_16_33150_l3183_318306

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (aces : Nat)
  (tens : Nat)
  (jacks : Nat)

/-- Calculates the probability of drawing a specific card from the deck -/
def draw_probability (deck : Deck) (target_cards : Nat) : Rat :=
  target_cards / deck.total_cards

/-- Calculates the probability of drawing an Ace, then a 10, then a Jack -/
def ace_ten_jack_probability (deck : Deck) : Rat :=
  let p1 := draw_probability deck deck.aces
  let p2 := draw_probability { deck with total_cards := deck.total_cards - 1 } deck.tens
  let p3 := draw_probability { deck with total_cards := deck.total_cards - 2 } deck.jacks
  p1 * p2 * p3

/-- The main theorem to be proved -/
theorem ace_ten_jack_prob_is_16_33150 :
  let standard_deck : Deck := { total_cards := 52, aces := 4, tens := 4, jacks := 4 }
  ace_ten_jack_probability standard_deck = 16 / 33150 := by
  sorry

end NUMINAMATH_CALUDE_ace_ten_jack_prob_is_16_33150_l3183_318306


namespace NUMINAMATH_CALUDE_percentage_value_in_quarters_l3183_318337

/-- Represents the number of nickels --/
def num_nickels : ℕ := 80

/-- Represents the number of quarters --/
def num_quarters : ℕ := 40

/-- Represents the value of a nickel in cents --/
def nickel_value : ℕ := 5

/-- Represents the value of a quarter in cents --/
def quarter_value : ℕ := 25

/-- Theorem stating that the percentage of total value in quarters is 5/7 --/
theorem percentage_value_in_quarters :
  (num_quarters * quarter_value : ℚ) / (num_nickels * nickel_value + num_quarters * quarter_value) = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_percentage_value_in_quarters_l3183_318337


namespace NUMINAMATH_CALUDE_problem_solution_l3183_318380

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

-- Define an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the specific function f(x) = lg(x + √(x^2 + 1))
noncomputable def f (x : ℝ) : ℝ := lg (x + Real.sqrt (x^2 + 1))

theorem problem_solution :
  (isPowerFunction (λ _ : ℝ => 1)) ∧
  (∀ g : ℝ → ℝ, isOddFunction g → g 0 = 0) ∧
  (isOddFunction f) ∧
  (∃ a : ℝ, a < 0 ∧ (a^2)^(3/2) ≠ a^3) ∧
  (∃! x : ℝ, (λ _ : ℝ => 1) x = 0 → False) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3183_318380


namespace NUMINAMATH_CALUDE_pond_amphibians_l3183_318364

/-- Calculates the total number of amphibians observed in a pond -/
def total_amphibians (green_frogs : ℕ) (observed_tree_frogs : ℕ) (bullfrogs : ℕ) 
  (exotic_tree_frogs : ℕ) (salamanders : ℕ) (first_tadpole_group : ℕ) (baby_frogs : ℕ) 
  (newts : ℕ) (toads : ℕ) (caecilians : ℕ) : ℕ :=
  let total_tree_frogs := observed_tree_frogs * 3
  let second_tadpole_group := first_tadpole_group - (first_tadpole_group / 5)
  green_frogs + total_tree_frogs + bullfrogs + exotic_tree_frogs + salamanders + 
  first_tadpole_group + second_tadpole_group + baby_frogs + newts + toads + caecilians

/-- Theorem stating the total number of amphibians observed in the pond -/
theorem pond_amphibians : 
  total_amphibians 6 5 2 8 3 50 10 1 2 1 = 138 := by
  sorry

end NUMINAMATH_CALUDE_pond_amphibians_l3183_318364


namespace NUMINAMATH_CALUDE_calculation_proof_l3183_318378

theorem calculation_proof : (-7)^3 / 7^2 - 2^5 + 4^3 - 8 = 81 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3183_318378


namespace NUMINAMATH_CALUDE_complex_moduli_sum_l3183_318309

theorem complex_moduli_sum : 
  let z1 : ℂ := 3 - 5*I
  let z2 : ℂ := 3 + 5*I
  Complex.abs z1 + Complex.abs z2 = 2 * Real.sqrt 34 := by
sorry

end NUMINAMATH_CALUDE_complex_moduli_sum_l3183_318309


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l3183_318358

/-- The area of the circle described by the polar equation r = 4 cos θ - 3 sin θ is equal to 25π/4 -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := λ θ ↦ 4 * Real.cos θ - 3 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, (r θ * Real.cos θ, r θ * Real.sin θ) ∈ Metric.sphere center radius) ∧
    Real.pi * radius^2 = 25 * Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l3183_318358


namespace NUMINAMATH_CALUDE_larger_number_problem_l3183_318372

theorem larger_number_problem (x y : ℝ) (h1 : x + y = 52) (h2 : x = 3 * y) (h3 : x > 0) (h4 : y > 0) : x = 39 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3183_318372


namespace NUMINAMATH_CALUDE_line_through_points_l3183_318329

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in the form ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a given line -/
def point_on_line (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- The theorem states that the line x - 2y + 1 = 0 passes through points A(-1, 0) and B(3, 2) -/
theorem line_through_points :
  let A : Point2D := ⟨-1, 0⟩
  let B : Point2D := ⟨3, 2⟩
  let line : Line2D := ⟨1, -2, 1⟩
  point_on_line A line ∧ point_on_line B line :=
by sorry

end NUMINAMATH_CALUDE_line_through_points_l3183_318329


namespace NUMINAMATH_CALUDE_arrangement_counts_l3183_318313

/-- The number of ways to arrange 3 boys and 4 girls in a row under specific conditions -/
theorem arrangement_counts :
  let total_people : ℕ := 7
  let num_boys : ℕ := 3
  let num_girls : ℕ := 4
  -- (1) Person A is neither at the middle nor at the ends
  (number_of_arrangements_1 : ℕ := 2880) →
  -- (2) Persons A and B must be at the two ends
  (number_of_arrangements_2 : ℕ := 240) →
  -- (3) Boys and girls alternate
  (number_of_arrangements_3 : ℕ := 144) →
  -- Prove all three conditions are true
  (number_of_arrangements_1 = 2880 ∧
   number_of_arrangements_2 = 240 ∧
   number_of_arrangements_3 = 144) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_counts_l3183_318313


namespace NUMINAMATH_CALUDE_roots_equation_s_value_l3183_318354

theorem roots_equation_s_value (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 6 = 0) → 
  (d^2 - n*d + 6 = 0) → 
  ((c + 2/d)^2 - r*(c + 2/d) + s = 0) → 
  ((d + 2/c)^2 - r*(d + 2/c) + s = 0) → 
  (s = 32/3) := by
sorry

end NUMINAMATH_CALUDE_roots_equation_s_value_l3183_318354


namespace NUMINAMATH_CALUDE_game_winning_strategy_l3183_318324

/-- Represents the players in the game -/
inductive Player
| A
| B

/-- Represents the result of the game -/
inductive GameResult
| AWins
| BWins

/-- Represents the game state -/
structure GameState where
  n : ℕ
  k : ℕ
  grid : Fin n → Fin n → Bool
  currentPlayer : Player

/-- Defines the winning strategy for the game -/
def winningStrategy (n k : ℕ) : GameResult :=
  if n ≤ 2 * k - 1 then
    GameResult.AWins
  else if n % 2 = 1 then
    GameResult.AWins
  else
    GameResult.BWins

/-- The main theorem stating the winning strategy for the game -/
theorem game_winning_strategy (n k : ℕ) (h1 : n ≥ k) (h2 : k ≥ 2) :
  (winningStrategy n k = GameResult.AWins ∧ 
   (n ≤ 2 * k - 1 ∨ (n ≥ 2 * k ∧ n % 2 = 1))) ∨
  (winningStrategy n k = GameResult.BWins ∧ 
   n ≥ 2 * k ∧ n % 2 = 0) :=
by sorry


end NUMINAMATH_CALUDE_game_winning_strategy_l3183_318324


namespace NUMINAMATH_CALUDE_pyramid_height_proof_l3183_318397

/-- The height of a square-based pyramid with base edge length 10 units, 
    which has the same volume as a cube with edge length 5 units. -/
def pyramid_height : ℝ := 3.75

theorem pyramid_height_proof :
  let cube_edge : ℝ := 5
  let pyramid_base : ℝ := 10
  let cube_volume : ℝ := cube_edge ^ 3
  let pyramid_volume (h : ℝ) : ℝ := (1/3) * pyramid_base ^ 2 * h
  pyramid_volume pyramid_height = cube_volume :=
by sorry

end NUMINAMATH_CALUDE_pyramid_height_proof_l3183_318397


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l3183_318331

/-- Represents the interest rate calculation problem --/
theorem interest_rate_calculation 
  (principal : ℝ) 
  (amount : ℝ) 
  (time : ℝ) 
  (h1 : principal = 896) 
  (h2 : amount = 1120) 
  (h3 : time = 5) :
  (amount - principal) / (principal * time) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l3183_318331


namespace NUMINAMATH_CALUDE_f_min_at_three_l3183_318398

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem stating that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_three : ∀ x : ℝ, f x ≥ f 3 := by sorry

end NUMINAMATH_CALUDE_f_min_at_three_l3183_318398


namespace NUMINAMATH_CALUDE_prob_same_color_is_89_169_l3183_318311

def blue_balls : ℕ := 8
def yellow_balls : ℕ := 5
def total_balls : ℕ := blue_balls + yellow_balls

def prob_same_color : ℚ := (blue_balls^2 + yellow_balls^2) / total_balls^2

theorem prob_same_color_is_89_169 : prob_same_color = 89 / 169 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_is_89_169_l3183_318311


namespace NUMINAMATH_CALUDE_calculation_proof_l3183_318379

theorem calculation_proof : 
  (((15^15 / 15^10)^3 * 5^6) / 25^2) = 3^15 * 5^17 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3183_318379


namespace NUMINAMATH_CALUDE_variation_problem_l3183_318373

/-- Given that R varies directly as S and inversely as T^2, prove that when R = 50 and T = 5, S = 5000/3 -/
theorem variation_problem (c : ℝ) (R S T : ℝ → ℝ) (t : ℝ) :
  (∀ t, R t = c * S t / (T t)^2) →  -- Relationship between R, S, and T
  R 0 = 3 →                        -- Initial condition for R
  S 0 = 16 →                       -- Initial condition for S
  T 0 = 2 →                        -- Initial condition for T
  R t = 50 →                       -- New value for R
  T t = 5 →                        -- New value for T
  S t = 5000 / 3 := by             -- Prove that S equals 5000/3
sorry


end NUMINAMATH_CALUDE_variation_problem_l3183_318373


namespace NUMINAMATH_CALUDE_millions_to_scientific_l3183_318366

-- Define the number in millions
def number_in_millions : ℝ := 3.111

-- Define the number in standard form
def number_standard : ℝ := 3111000

-- Define the number in scientific notation
def number_scientific : ℝ := 3.111 * (10 ^ 6)

-- Theorem to prove
theorem millions_to_scientific : number_standard = number_scientific := by
  sorry

end NUMINAMATH_CALUDE_millions_to_scientific_l3183_318366


namespace NUMINAMATH_CALUDE_nanometer_to_meter_one_nanometer_def_l3183_318339

/-- Proves that 28 nanometers is equal to 2.8 × 10^(-8) meters. -/
theorem nanometer_to_meter : 
  (28 : ℝ) * (1e-9 : ℝ) = (2.8 : ℝ) * (1e-8 : ℝ) := by
  sorry

/-- Defines the conversion factor from nanometers to meters. -/
def nanometer_to_meter_conversion : ℝ := 1e-9

/-- Proves that 1 nanometer is equal to 10^(-9) meters. -/
theorem one_nanometer_def : 
  (1 : ℝ) * nanometer_to_meter_conversion = (1e-9 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_nanometer_to_meter_one_nanometer_def_l3183_318339


namespace NUMINAMATH_CALUDE_error_percentage_bounds_l3183_318390

theorem error_percentage_bounds (y : ℝ) (h : y > 0) :
  let error_percentage := (20 / (y + 8)) * 100
  100 < error_percentage ∧ error_percentage < 120 := by
sorry

end NUMINAMATH_CALUDE_error_percentage_bounds_l3183_318390


namespace NUMINAMATH_CALUDE_plane_Q_satisfies_conditions_l3183_318300

def plane1 (x y z : ℝ) : ℝ := 2*x - y + 2*z - 4
def plane2 (x y z : ℝ) : ℝ := 3*x + y - z - 6
def planeQ (x y z : ℝ) : ℝ := 19*x - 67*y + 109*z - 362

def point : ℝ × ℝ × ℝ := (2, 0, 3)

theorem plane_Q_satisfies_conditions :
  (∀ x y z : ℝ, plane1 x y z = 0 ∧ plane2 x y z = 0 → planeQ x y z = 0) ∧ 
  (planeQ ≠ plane1 ∧ planeQ ≠ plane2) ∧
  (let (x₀, y₀, z₀) := point
   abs (19*x₀ - 67*y₀ + 109*z₀ - 362) / Real.sqrt (19^2 + (-67)^2 + 109^2) = 3 / Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_plane_Q_satisfies_conditions_l3183_318300


namespace NUMINAMATH_CALUDE_possibly_six_l3183_318336

/-- Represents the possible outcomes of a dice throw -/
inductive DiceOutcome
  | one
  | two
  | three
  | four
  | five
  | six

/-- A fair six-sided dice -/
structure FairDice :=
  (outcomes : Finset DiceOutcome)
  (fair : outcomes.card = 6)
  (complete : ∀ o : DiceOutcome, o ∈ outcomes)

/-- The result of a single throw of a fair dice -/
def singleThrow (d : FairDice) : Set DiceOutcome :=
  d.outcomes

theorem possibly_six (d : FairDice) : 
  DiceOutcome.six ∈ singleThrow d :=
sorry

end NUMINAMATH_CALUDE_possibly_six_l3183_318336


namespace NUMINAMATH_CALUDE_min_value_on_line_l3183_318362

theorem min_value_on_line (x y : ℝ) (h : x + 2*y + 1 = 0) :
  2^x + 4^y ≥ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_on_line_l3183_318362


namespace NUMINAMATH_CALUDE_min_value_trig_expression_l3183_318365

theorem min_value_trig_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 36 ∧
  ∃ α₀ β₀ : ℝ, (3 * Real.cos α₀ + 4 * Real.sin β₀ - 7)^2 + (3 * Real.sin α₀ + 4 * Real.cos β₀ - 12)^2 = 36 :=
sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_l3183_318365


namespace NUMINAMATH_CALUDE_trent_travel_distance_l3183_318385

/-- The total distance Trent traveled -/
def total_distance (house_to_bus bus_to_library : ℕ) : ℕ :=
  2 * (house_to_bus + bus_to_library)

/-- Theorem stating that Trent's total travel distance is 22 blocks -/
theorem trent_travel_distance :
  ∃ (house_to_bus bus_to_library : ℕ),
    house_to_bus = 4 ∧
    bus_to_library = 7 ∧
    total_distance house_to_bus bus_to_library = 22 :=
by
  sorry

end NUMINAMATH_CALUDE_trent_travel_distance_l3183_318385


namespace NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincide_l3183_318322

/-- The value of p for which the focus of the parabola x^2 = 2py (p > 0) 
    coincides with the focus of the hyperbola y^2/3 - x^2 = 1 -/
theorem parabola_hyperbola_focus_coincide : 
  ∃ p : ℝ, p > 0 ∧ 
  (∀ x y : ℝ, x^2 = 2*p*y ↔ (x, y) ∈ {(x, y) | x^2 = 2*p*y}) ∧
  (∀ x y : ℝ, y^2/3 - x^2 = 1 ↔ (x, y) ∈ {(x, y) | y^2/3 - x^2 = 1}) ∧
  (0, p/2) = (0, 2) ∧
  p = 4 :=
sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_focus_coincide_l3183_318322


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3183_318394

theorem imaginary_part_of_z (z : ℂ) : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I →
  z.im = (Real.sqrt 2 + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3183_318394


namespace NUMINAMATH_CALUDE_max_fourth_term_arithmetic_sequence_l3183_318351

theorem max_fourth_term_arithmetic_sequence (a d : ℕ) (h1 : 0 < a) (h2 : 0 < d) :
  (∀ k : Fin 5, 0 < a + k * d) →
  (5 * a + 10 * d = 75) →
  (∀ a' d' : ℕ, (∀ k : Fin 5, 0 < a' + k * d') → (5 * a' + 10 * d' = 75) → a + 3 * d ≥ a' + 3 * d') →
  a + 3 * d = 22 := by
sorry

end NUMINAMATH_CALUDE_max_fourth_term_arithmetic_sequence_l3183_318351


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3183_318367

def is_geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ r : ℚ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum (a : ℕ → ℚ) :
  is_geometric_sequence a →
  (a 5 + a 6 + a 7 + a 8 = 15/8) →
  (a 6 * a 7 = -9/8) →
  (1 / a 5 + 1 / a 6 + 1 / a 7 + 1 / a 8 = -5/3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3183_318367


namespace NUMINAMATH_CALUDE_triangle_area_l3183_318377

/-- The area of a triangle with vertices at (-3, 7), (-7, 3), and (0, 0) in a coordinate plane is 50 square units. -/
theorem triangle_area : Real := by
  -- Define the vertices of the triangle
  let v1 : Prod Real Real := (-3, 7)
  let v2 : Prod Real Real := (-7, 3)
  let v3 : Prod Real Real := (0, 0)

  -- Calculate the area of the triangle
  let area : Real := sorry

  -- Prove that the calculated area is equal to 50
  have h : area = 50 := by sorry

  -- Return the area
  exact 50


end NUMINAMATH_CALUDE_triangle_area_l3183_318377


namespace NUMINAMATH_CALUDE_special_function_properties_l3183_318316

/-- A function satisfying the given functional equation -/
structure SpecialFunction where
  f : ℝ → ℝ
  eq : ∀ x y, f (x + y) * f (x - y) = f x + f y
  nonzero : f 0 ≠ 0

/-- Properties of the special function -/
theorem special_function_properties (F : SpecialFunction) :
  (F.f 0 = 2) ∧
  (∀ x, F.f x = F.f (-x)) ∧
  (∀ x, F.f (2 * x) = F.f x) := by
  sorry


end NUMINAMATH_CALUDE_special_function_properties_l3183_318316


namespace NUMINAMATH_CALUDE_pq_divides_3p_minus_1_q_minus_1_l3183_318374

theorem pq_divides_3p_minus_1_q_minus_1 (p q : ℕ+) :
  (p * q : ℕ) ∣ (3 * (p - 1) * (q - 1) : ℕ) ↔
  ((p = 6 ∧ q = 5) ∨ (p = 5 ∧ q = 6) ∨
   (p = 9 ∧ q = 4) ∨ (p = 4 ∧ q = 9) ∨
   (p = 3 ∧ q = 2) ∨ (p = 2 ∧ q = 3)) :=
by sorry

end NUMINAMATH_CALUDE_pq_divides_3p_minus_1_q_minus_1_l3183_318374


namespace NUMINAMATH_CALUDE_curve_through_center_l3183_318393

-- Define a square
structure Square where
  side : ℝ
  center : ℝ × ℝ

-- Define a curve
structure Curve where
  path : ℝ → ℝ × ℝ

-- Define the property of dividing the square into equal areas
def divides_equally (γ : Curve) (s : Square) : Prop :=
  ∃ (area1 area2 : ℝ), area1 = area2 ∧ area1 + area2 = s.side * s.side

-- Define the property of a line segment passing through a point
def passes_through (a b c : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ c = (1 - t) • a + t • b

-- The main theorem
theorem curve_through_center (s : Square) (γ : Curve) 
  (h : divides_equally γ s) :
  ∃ (a b : ℝ × ℝ), (∃ (t1 t2 : ℝ), γ.path t1 = a ∧ γ.path t2 = b) ∧ 
    passes_through a b s.center :=
sorry

end NUMINAMATH_CALUDE_curve_through_center_l3183_318393


namespace NUMINAMATH_CALUDE_susan_money_problem_l3183_318353

theorem susan_money_problem (S : ℚ) :
  S - (S / 6 + S / 8 + S * (30 / 100) + 100) = 480 →
  S = 1420 := by
sorry

end NUMINAMATH_CALUDE_susan_money_problem_l3183_318353


namespace NUMINAMATH_CALUDE_fixed_points_of_f_l3183_318348

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + (b + 1) * x + (b - 1)

theorem fixed_points_of_f (a b : ℝ) (ha : a ≠ 0) :
  -- Part 1
  (a = 1 ∧ b = 2 → ∃ x : ℝ, f 1 2 x = x ∧ x = -1) ∧
  -- Part 2
  (∀ b : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a b x₁ = x₁ ∧ f a b x₂ = x₂) ↔ 0 < a ∧ a < 1) ∧
  -- Part 3
  (0 < a ∧ a < 1 →
    ∀ x₁ x₂ : ℝ, f a b x₁ = x₁ → f a b x₂ = x₂ →
      f a b x₁ + x₂ = -a / (2 * a^2 + 1) →
        0 < b ∧ b < 1/3) :=
by sorry

end NUMINAMATH_CALUDE_fixed_points_of_f_l3183_318348


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3183_318368

theorem min_value_theorem (x : ℝ) (h : x > 0) : x + 4 / x - 1 ≥ 3 :=
by sorry

theorem equality_condition : ∃ x : ℝ, x > 0 ∧ x + 4 / x - 1 = 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_condition_l3183_318368


namespace NUMINAMATH_CALUDE_vector_relations_l3183_318344

def a : Fin 3 → ℝ
| 0 => 2
| 1 => -1
| 2 => 3
| _ => 0

def b (x : ℝ) : Fin 3 → ℝ
| 0 => -4
| 1 => 2
| 2 => x
| _ => 0

theorem vector_relations (x : ℝ) :
  (∀ i : Fin 3, (a i) * (b x i) = 0 → x = 10/3) ∧
  (∃ k : ℝ, ∀ i : Fin 3, (a i) = k * (b x i) → x = -6) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l3183_318344


namespace NUMINAMATH_CALUDE_gcf_of_180_and_270_l3183_318318

theorem gcf_of_180_and_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_180_and_270_l3183_318318


namespace NUMINAMATH_CALUDE_passing_marks_calculation_l3183_318314

theorem passing_marks_calculation (T : ℝ) (P : ℝ) : 
  (0.35 * T = P - 40) → 
  (0.60 * T = P + 25) → 
  P = 131 := by
  sorry

end NUMINAMATH_CALUDE_passing_marks_calculation_l3183_318314


namespace NUMINAMATH_CALUDE_ages_sum_l3183_318325

theorem ages_sum (j l : ℝ) : 
  j = l + 8 ∧ 
  j + 5 = 3 * (l - 6) → 
  j + l = 39 := by
  sorry

end NUMINAMATH_CALUDE_ages_sum_l3183_318325


namespace NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3183_318382

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ := sorry

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List (Fin 4) := sorry

/-- The binary representation of the number 10110010 -/
def binary_number : List Bool := [true, false, true, true, false, false, true, false]

/-- The quaternary representation of the number 2302 -/
def quaternary_number : List (Fin 4) := [2, 3, 0, 2]

theorem binary_to_quaternary_conversion :
  decimal_to_quaternary (binary_to_decimal binary_number) = quaternary_number := by sorry

end NUMINAMATH_CALUDE_binary_to_quaternary_conversion_l3183_318382


namespace NUMINAMATH_CALUDE_shortest_median_not_longer_than_longest_bisector_shortest_bisector_not_longer_than_longest_altitude_l3183_318320

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define median, angle bisector, and altitude
def median (t : Triangle) : ℝ := sorry
def angle_bisector (t : Triangle) : ℝ := sorry
def altitude (t : Triangle) : ℝ := sorry

-- Theorem 1: The shortest median is never longer than the longest angle bisector
theorem shortest_median_not_longer_than_longest_bisector (t : Triangle) :
  ∀ m b, median t ≤ m → angle_bisector t ≥ b → m ≤ b :=
sorry

-- Theorem 2: The shortest angle bisector is never longer than the longest altitude
theorem shortest_bisector_not_longer_than_longest_altitude (t : Triangle) :
  ∀ b h, angle_bisector t ≤ b → altitude t ≥ h → b ≤ h :=
sorry

end NUMINAMATH_CALUDE_shortest_median_not_longer_than_longest_bisector_shortest_bisector_not_longer_than_longest_altitude_l3183_318320


namespace NUMINAMATH_CALUDE_annual_income_difference_l3183_318332

/-- Given an 8% raise, if a person's raise is Rs. 800 and another person's raise is Rs. 840,
    then the difference between their new annual incomes is Rs. 540. -/
theorem annual_income_difference (D W : ℝ) : 
  0.08 * D = 800 → 0.08 * W = 840 → W + 840 - (D + 800) = 540 := by
  sorry

end NUMINAMATH_CALUDE_annual_income_difference_l3183_318332


namespace NUMINAMATH_CALUDE_right_triangle_identification_l3183_318305

def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

theorem right_triangle_identification :
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 5 12 13) ∧
  (¬ is_right_triangle 6 8 12) ∧
  (¬ is_right_triangle 6 12 15) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_identification_l3183_318305


namespace NUMINAMATH_CALUDE_square_root_three_expansion_l3183_318346

theorem square_root_three_expansion 
  (a b c d : ℕ+) 
  (h : (a : ℝ) + (b : ℝ) * Real.sqrt 3 = ((c : ℝ) + (d : ℝ) * Real.sqrt 3) ^ 2) : 
  a = c ^ 2 + 3 * d ^ 2 ∧ b = 2 * c * d :=
sorry

end NUMINAMATH_CALUDE_square_root_three_expansion_l3183_318346


namespace NUMINAMATH_CALUDE_quadratic_root_triple_relation_l3183_318302

theorem quadratic_root_triple_relation (a b c : ℝ) :
  (∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ 
              a * y^2 + b * y + c = 0 ∧ 
              y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_triple_relation_l3183_318302


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3183_318340

/-- A geometric sequence with positive terms and a specific condition -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n) ∧
  (2 * a 1 + a 2 = a 3)

/-- The common ratio of the geometric sequence is 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (h : GeometricSequence a) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n, a (n + 1) = q * a n := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3183_318340


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3183_318369

theorem sin_alpha_value (α : Real) : 
  (∃ (x y : Real), x = 2 * Real.sin (30 * π / 180) ∧ 
                   y = 2 * Real.cos (30 * π / 180) ∧ 
                   x = 2 * Real.sin α ∧ 
                   y = 2 * Real.cos α) → 
  Real.sin α = Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3183_318369


namespace NUMINAMATH_CALUDE_fraction_value_theorem_l3183_318307

theorem fraction_value_theorem (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 4) (h3 : a < b) :
  (a - b) / (a + b) = -7 ∨ (a - b) / (a + b) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_theorem_l3183_318307


namespace NUMINAMATH_CALUDE_unique_base_solution_l3183_318391

/-- Converts a number from base b to base 10 -/
def toBase10 (n : ℕ) (b : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * b^2 + tens * b + ones

/-- The main theorem statement -/
theorem unique_base_solution :
  ∃! b : ℕ, b > 7 ∧ (toBase10 276 b) * 2 + (toBase10 145 b) = (toBase10 697 b) :=
by sorry

end NUMINAMATH_CALUDE_unique_base_solution_l3183_318391


namespace NUMINAMATH_CALUDE_expression_simplification_l3183_318355

theorem expression_simplification (m : ℝ) (hm : m = 2) : 
  (((m + 1) / (m - 1) + 1) / ((m + m^2) / (m^2 - 2*m + 1))) - ((2 - 2*m) / (m^2 - 1)) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3183_318355


namespace NUMINAMATH_CALUDE_log_sum_property_l3183_318319

theorem log_sum_property (a b : ℝ) (ha : a > 1) (hb : b > 1) 
  (h : Real.log (a + b) = Real.log a + Real.log b) : 
  Real.log (a - 1) + Real.log (b - 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_property_l3183_318319


namespace NUMINAMATH_CALUDE_exam_results_l3183_318323

/-- Given an examination where:
  * 35% of students failed in Hindi
  * 20% of students failed in both Hindi and English
  * 40% of students passed in both subjects
  Prove that 45% of students failed in English -/
theorem exam_results (total : ℝ) (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ) 
  (h_total : total = 100)
  (h_failed_hindi : failed_hindi = 35)
  (h_failed_both : failed_both = 20)
  (h_passed_both : passed_both = 40) :
  ∃ (failed_english : ℝ), failed_english = 45 :=
sorry

end NUMINAMATH_CALUDE_exam_results_l3183_318323


namespace NUMINAMATH_CALUDE_last_four_digits_of_2_to_1965_l3183_318349

theorem last_four_digits_of_2_to_1965 : 2^1965 % 10000 = 3125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_2_to_1965_l3183_318349


namespace NUMINAMATH_CALUDE_element_correspondence_l3183_318328

-- Define the mapping f from A to B
def f (x : ℝ) : ℝ := 2 * x + 1

-- Theorem statement
theorem element_correspondence : f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_element_correspondence_l3183_318328


namespace NUMINAMATH_CALUDE_all_ingredients_good_probability_l3183_318347

/-- The probability of selecting a fresh bottle of milk -/
def prob_fresh_milk : ℝ := 0.8

/-- The probability of selecting a good egg -/
def prob_good_egg : ℝ := 0.4

/-- The probability of selecting a good canister of flour -/
def prob_good_flour : ℝ := 0.75

/-- The probability that all three ingredients (milk, egg, flour) are good when selected randomly -/
def prob_all_good : ℝ := prob_fresh_milk * prob_good_egg * prob_good_flour

theorem all_ingredients_good_probability :
  prob_all_good = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_all_ingredients_good_probability_l3183_318347


namespace NUMINAMATH_CALUDE_broadcast_methods_count_l3183_318395

/-- The number of different advertisements -/
def total_ads : ℕ := 5

/-- The number of commercial advertisements -/
def commercial_ads : ℕ := 3

/-- The number of Olympic promotional advertisements -/
def olympic_ads : ℕ := 2

/-- A function that calculates the number of ways to arrange the advertisements -/
def arrangement_count : ℕ :=
  Nat.factorial commercial_ads * Nat.choose 4 2

/-- Theorem stating that the number of different broadcasting methods is 36 -/
theorem broadcast_methods_count :
  arrangement_count = 36 :=
by sorry

end NUMINAMATH_CALUDE_broadcast_methods_count_l3183_318395


namespace NUMINAMATH_CALUDE_waiter_customers_l3183_318375

theorem waiter_customers (initial new_customers customers_left : ℕ) :
  initial ≥ customers_left →
  (initial - customers_left + new_customers : ℕ) = initial - customers_left + new_customers :=
by sorry

end NUMINAMATH_CALUDE_waiter_customers_l3183_318375


namespace NUMINAMATH_CALUDE_minimum_travel_cost_l3183_318356

-- Define the cities and distances
def X : City := sorry
def Y : City := sorry
def Z : City := sorry

-- Define the distances
def distance_XY : ℝ := 5000
def distance_XZ : ℝ := 4000

-- Define the cost functions
def bus_cost (distance : ℝ) : ℝ := 0.2 * distance
def plane_cost (distance : ℝ) : ℝ := 150 + 0.15 * distance

-- Define the theorem
theorem minimum_travel_cost :
  ∃ (cost : ℝ),
    cost = plane_cost distance_XY + 
           plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
           plane_cost distance_XZ ∧
    cost = 2250 ∧
    ∀ (alternative_cost : ℝ),
      (alternative_cost = bus_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = plane_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = bus_cost distance_XY + 
                          plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = bus_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          plane_cost distance_XZ ∨
       alternative_cost = plane_cost distance_XY + 
                          plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          bus_cost distance_XZ ∨
       alternative_cost = plane_cost distance_XY + 
                          bus_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          plane_cost distance_XZ ∨
       alternative_cost = bus_cost distance_XY + 
                          plane_cost (Real.sqrt (distance_XY^2 - distance_XZ^2)) + 
                          plane_cost distance_XZ) →
      cost ≤ alternative_cost :=
by sorry

end NUMINAMATH_CALUDE_minimum_travel_cost_l3183_318356


namespace NUMINAMATH_CALUDE_cube_surface_area_from_prism_volume_cube_surface_area_from_prism_volume_proof_l3183_318333

/-- The surface area of a cube with volume equal to a 10x10x8 inch rectangular prism is 1200 square inches. -/
theorem cube_surface_area_from_prism_volume : ℝ → ℝ → ℝ → ℝ → Prop :=
  fun prism_length prism_width prism_height cube_surface_area =>
    prism_length = 10 ∧
    prism_width = 10 ∧
    prism_height = 8 ∧
    cube_surface_area = 6 * (prism_length * prism_width * prism_height) ^ (2/3) ∧
    cube_surface_area = 1200

/-- Proof of the theorem -/
theorem cube_surface_area_from_prism_volume_proof :
  cube_surface_area_from_prism_volume 10 10 8 1200 := by
  sorry

#check cube_surface_area_from_prism_volume
#check cube_surface_area_from_prism_volume_proof

end NUMINAMATH_CALUDE_cube_surface_area_from_prism_volume_cube_surface_area_from_prism_volume_proof_l3183_318333


namespace NUMINAMATH_CALUDE_A_intersect_B_l3183_318312

def A : Set ℕ := {x | x - 4 < 0}
def B : Set ℕ := {0, 1, 3, 4}

theorem A_intersect_B : A ∩ B = {0, 1, 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l3183_318312


namespace NUMINAMATH_CALUDE_smallest_multiple_year_l3183_318341

def joey_age : ℕ := 40
def chloe_age : ℕ := 38
def father_age : ℕ := 60

theorem smallest_multiple_year : 
  ∃ (n : ℕ), n > 0 ∧ 
  (joey_age + n) % father_age = 0 ∧ 
  (chloe_age + n) % father_age = 0 ∧
  (∀ (m : ℕ), m > 0 → m < n → 
    (joey_age + m) % father_age ≠ 0 ∨ 
    (chloe_age + m) % father_age ≠ 0) ∧
  n = 180 :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_year_l3183_318341


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l3183_318399

/-- In a right triangle with hypotenuse c and legs a and b, where c = a + 2, 
    the square of b is equal to 4a + 4. -/
theorem right_triangle_leg_square (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Right triangle condition
  (h_diff : c = a + 2)         -- Hypotenuse and leg difference condition
  : b^2 = 4*a + 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l3183_318399


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l3183_318301

theorem ceiling_floor_difference : 
  ⌈(10 : ℝ) / 4 * (-17 : ℝ) / 2⌉ - ⌊(10 : ℝ) / 4 * ⌊(-17 : ℝ) / 2⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l3183_318301


namespace NUMINAMATH_CALUDE_dog_catches_rabbit_problem_l3183_318376

/-- The number of leaps required for a dog to catch a rabbit -/
def dog_catches_rabbit (initial_distance : ℕ) (dog_leap : ℕ) (rabbit_jump : ℕ) : ℕ :=
  initial_distance / (dog_leap - rabbit_jump)

theorem dog_catches_rabbit_problem :
  dog_catches_rabbit 150 9 7 = 75 := by
  sorry

end NUMINAMATH_CALUDE_dog_catches_rabbit_problem_l3183_318376


namespace NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l3183_318321

theorem quadratic_roots_to_coefficients (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b*x + c = 0 ↔ x = 2 ∨ x = -3) → 
  b = 1 ∧ c = -6 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_to_coefficients_l3183_318321


namespace NUMINAMATH_CALUDE_dubblefud_game_l3183_318326

theorem dubblefud_game (red_value blue_value green_value : ℕ)
  (total_product : ℕ) (red blue green : ℕ) :
  red_value = 3 →
  blue_value = 7 →
  green_value = 11 →
  total_product = 5764801 →
  blue = green →
  (red_value ^ red) * (blue_value ^ blue) * (green_value ^ green) = total_product →
  red = 7 := by
  sorry

end NUMINAMATH_CALUDE_dubblefud_game_l3183_318326
