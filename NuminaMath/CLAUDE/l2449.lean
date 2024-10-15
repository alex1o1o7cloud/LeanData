import Mathlib

namespace NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2449_244953

theorem cubic_polynomials_common_roots (c d : ℝ) :
  c = -5 ∧ d = -6 →
  ∃ (r s : ℝ), r ≠ s ∧
    (r^3 + c*r^2 + 12*r + 7 = 0) ∧ 
    (r^3 + d*r^2 + 15*r + 9 = 0) ∧
    (s^3 + c*s^2 + 12*s + 7 = 0) ∧ 
    (s^3 + d*s^2 + 15*s + 9 = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomials_common_roots_l2449_244953


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l2449_244932

def pizza_shares (total : ℚ) (ali : ℚ) (bea : ℚ) (chris : ℚ) : ℚ × ℚ × ℚ × ℚ :=
  let dan := total - (ali + bea + chris)
  (dan, ali, chris, bea)

theorem pizza_consumption_order (total : ℚ) :
  let (dan, ali, chris, bea) := pizza_shares total (1/6) (1/8) (1/7)
  dan > ali ∧ ali > chris ∧ chris > bea := by
  sorry

#check pizza_consumption_order

end NUMINAMATH_CALUDE_pizza_consumption_order_l2449_244932


namespace NUMINAMATH_CALUDE_greatest_n_with_perfect_square_property_l2449_244913

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

def is_perfect_square (m : ℕ) : Prop := ∃ k : ℕ, m = k^2

theorem greatest_n_with_perfect_square_property :
  ∃ (n : ℕ), n = 1921 ∧ n ≤ 2008 ∧
  (∀ m : ℕ, m ≤ 2008 → m > n →
    ¬ is_perfect_square ((sum_of_squares n) * (sum_of_squares (2 * n) - sum_of_squares n))) ∧
  is_perfect_square ((sum_of_squares n) * (sum_of_squares (2 * n) - sum_of_squares n)) :=
sorry

end NUMINAMATH_CALUDE_greatest_n_with_perfect_square_property_l2449_244913


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2449_244902

theorem sum_of_coefficients (a b c : ℕ+) : 
  (∃ (k : ℚ), k * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
    (a * Real.sqrt 6 + b * Real.sqrt 8) / c) →
  (∀ (a' b' c' : ℕ+), 
    (∃ (k' : ℚ), k' * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) = 
      (a' * Real.sqrt 6 + b' * Real.sqrt 8) / c') →
    c ≤ c') →
  a.val + b.val + c.val = 106 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2449_244902


namespace NUMINAMATH_CALUDE_batsman_110_run_inning_l2449_244930

/-- Represents a batsman's scoring history -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  deriving Repr

/-- Calculate the average score of a batsman -/
def average (b : Batsman) : ℚ :=
  b.totalRuns / b.innings

/-- The inning where the batsman scores 110 runs -/
def scoreInning (b : Batsman) : ℕ :=
  b.innings + 1

theorem batsman_110_run_inning (b : Batsman) 
  (h1 : average (⟨b.innings + 1, b.totalRuns + 110⟩ : Batsman) = 60)
  (h2 : average (⟨b.innings + 1, b.totalRuns + 110⟩ : Batsman) - average b = 5) :
  scoreInning b = 11 := by
  sorry

#eval scoreInning ⟨10, 550⟩

end NUMINAMATH_CALUDE_batsman_110_run_inning_l2449_244930


namespace NUMINAMATH_CALUDE_common_ratio_is_four_l2449_244992

/-- Geometric sequence with sum S_n of first n terms -/
def geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))

theorem common_ratio_is_four 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : geometric_sequence a S)
  (h1 : 3 * S 3 = a 4 - 2)
  (h2 : 3 * S 2 = a 3 - 2) :
  a 2 / a 1 = 4 := by sorry

end NUMINAMATH_CALUDE_common_ratio_is_four_l2449_244992


namespace NUMINAMATH_CALUDE_cube_construction_problem_l2449_244925

theorem cube_construction_problem :
  ∃! (a b c : ℕ+), a^3 + b^3 + c^3 + 648 = (a + b + c)^3 :=
sorry

end NUMINAMATH_CALUDE_cube_construction_problem_l2449_244925


namespace NUMINAMATH_CALUDE_initial_candies_count_l2449_244954

/-- The number of candies initially in the box -/
def initial_candies : ℕ := sorry

/-- The number of candies Diana took from the box -/
def candies_taken : ℕ := 6

/-- The number of candies left in the box after Diana took some -/
def candies_left : ℕ := 82

/-- Theorem stating that the initial number of candies is 88 -/
theorem initial_candies_count : initial_candies = 88 :=
  by sorry

end NUMINAMATH_CALUDE_initial_candies_count_l2449_244954


namespace NUMINAMATH_CALUDE_profit_calculation_l2449_244903

theorem profit_calculation (cost_price : ℝ) (x : ℝ) : 
  (40 * cost_price = x * (cost_price * 1.25)) → x = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_profit_calculation_l2449_244903


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l2449_244951

theorem complex_magnitude_equation (m : ℝ) (h : m > 0) : 
  Complex.abs (5 + m * Complex.I) = 5 * Real.sqrt 26 → m = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l2449_244951


namespace NUMINAMATH_CALUDE_seven_day_payment_possible_l2449_244905

/-- Represents the state of rings at any given time --/
structure RingState :=
  (single : ℕ)    -- number of single rings
  (double : ℕ)    -- number of chains with 2 rings
  (quadruple : ℕ) -- number of chains with 4 rings

/-- Represents a daily transaction --/
inductive Transaction
  | give_single
  | give_double
  | give_quadruple
  | return_single
  | return_double

/-- Applies a transaction to a RingState --/
def apply_transaction (state : RingState) (t : Transaction) : RingState :=
  match t with
  | Transaction.give_single => ⟨state.single - 1, state.double, state.quadruple⟩
  | Transaction.give_double => ⟨state.single, state.double - 1, state.quadruple⟩
  | Transaction.give_quadruple => ⟨state.single, state.double, state.quadruple - 1⟩
  | Transaction.return_single => ⟨state.single + 1, state.double, state.quadruple⟩
  | Transaction.return_double => ⟨state.single, state.double + 1, state.quadruple⟩

/-- Checks if a sequence of transactions is valid for a given initial state --/
def is_valid_sequence (initial : RingState) (transactions : List Transaction) : Prop :=
  ∀ (n : ℕ), n < transactions.length →
    let state := transactions.take n.succ
      |> List.foldl apply_transaction initial
    state.single ≥ 0 ∧ state.double ≥ 0 ∧ state.quadruple ≥ 0

/-- Checks if a sequence of transactions results in a net payment of one ring per day --/
def is_daily_payment (transactions : List Transaction) : Prop :=
  transactions.foldl (λ acc t =>
    match t with
    | Transaction.give_single => acc + 1
    | Transaction.give_double => acc + 2
    | Transaction.give_quadruple => acc + 4
    | Transaction.return_single => acc - 1
    | Transaction.return_double => acc - 2
  ) 0 = 1

/-- The main theorem: it is possible to pay for 7 days using a chain of 7 rings, cutting only one --/
theorem seven_day_payment_possible : ∃ (transactions : List Transaction),
  transactions.length = 7 ∧
  is_valid_sequence ⟨1, 1, 1⟩ transactions ∧
  (∀ (n : ℕ), n < 7 → is_daily_payment (transactions.take (n + 1))) :=
sorry

end NUMINAMATH_CALUDE_seven_day_payment_possible_l2449_244905


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l2449_244998

theorem min_value_quadratic (x : ℝ) : 3 * x^2 - 18 * x + 2048 ≥ 2021 := by sorry

theorem min_value_quadratic_achieved : ∃ x : ℝ, 3 * x^2 - 18 * x + 2048 = 2021 := by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_quadratic_achieved_l2449_244998


namespace NUMINAMATH_CALUDE_painting_discount_l2449_244910

theorem painting_discount (x : ℝ) (h1 : x / 5 = 15) : x * (1 - 1/3) = 50 := by
  sorry

end NUMINAMATH_CALUDE_painting_discount_l2449_244910


namespace NUMINAMATH_CALUDE_intersection_product_l2449_244961

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Represents a parabola -/
structure Parabola where
  focus : Point

/-- Checks if a point is on the ellipse -/
def on_ellipse (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Checks if a point is on the parabola -/
def on_parabola (p : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 4 * p.focus.x * pt.x

/-- Theorem statement -/
theorem intersection_product (e : Ellipse) (p : Parabola) 
  (A B P : Point) (h_A : on_ellipse e A ∧ on_parabola p A)
  (h_B : on_ellipse e B ∧ on_parabola p B)
  (h_P : on_ellipse e P)
  (h_quad : A.y > 0 ∧ B.y < 0)
  (h_focus : p.focus.y = 0 ∧ p.focus.x > 0)
  (h_vertex : p.focus.x = e.a^2 / (4 * e.b^2))
  (M N : ℝ) (h_M : ∃ t, A.x + t * (P.x - A.x) = M ∧ A.y + t * (P.y - A.y) = 0)
  (h_N : ∃ t, B.x + t * (P.x - B.x) = N ∧ B.y + t * (P.y - B.y) = 0) :
  M * N = e.a^2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_product_l2449_244961


namespace NUMINAMATH_CALUDE_absolute_value_quadratic_equation_solution_l2449_244997

theorem absolute_value_quadratic_equation_solution :
  let y₁ : ℝ := (-1 + Real.sqrt 241) / 6
  let y₂ : ℝ := (1 - Real.sqrt 145) / 6
  (|y₁ - 4| + 3 * y₁^2 = 16) ∧ (|y₂ - 4| + 3 * y₂^2 = 16) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_quadratic_equation_solution_l2449_244997


namespace NUMINAMATH_CALUDE_complement_A_union_B_eq_B_l2449_244963

-- Define the sets A and B
def A : Set ℝ := {x | x < -1 ∨ x > 1}
def B : Set ℝ := {x | x ≥ -1}

-- State the theorem
theorem complement_A_union_B_eq_B : (Aᶜ ∪ B) = B := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_eq_B_l2449_244963


namespace NUMINAMATH_CALUDE_bouncy_balls_shipment_l2449_244908

theorem bouncy_balls_shipment (displayed_percentage : ℚ) (warehouse_count : ℕ) : 
  displayed_percentage = 1/4 →
  warehouse_count = 90 →
  ∃ total : ℕ, total = 120 ∧ (1 - displayed_percentage) * total = warehouse_count :=
by sorry

end NUMINAMATH_CALUDE_bouncy_balls_shipment_l2449_244908


namespace NUMINAMATH_CALUDE_actual_height_is_236_l2449_244909

/-- The actual height of a boy in a class, given the following conditions:
  * There are 35 boys in the class
  * The initial average height was calculated as 185 cm
  * One boy's height was wrongly written as 166 cm
  * The actual average height is 183 cm
-/
def actual_height : ℕ :=
  let num_boys : ℕ := 35
  let initial_avg : ℕ := 185
  let wrong_height : ℕ := 166
  let actual_avg : ℕ := 183
  let initial_total : ℕ := num_boys * initial_avg
  let actual_total : ℕ := num_boys * actual_avg
  let height_difference : ℕ := initial_total - actual_total
  wrong_height + height_difference

theorem actual_height_is_236 : actual_height = 236 := by
  sorry

end NUMINAMATH_CALUDE_actual_height_is_236_l2449_244909


namespace NUMINAMATH_CALUDE_simplify_expression_l2449_244979

theorem simplify_expression (x y : ℝ) : 7*x + 3 - 2*x + 15 + y = 5*x + y + 18 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2449_244979


namespace NUMINAMATH_CALUDE_only_rational_root_l2449_244985

def polynomial (x : ℚ) : ℚ := 3 * x^4 - 7 * x^3 + 4 * x^2 + 6 * x - 8

theorem only_rational_root :
  ∀ x : ℚ, polynomial x = 0 ↔ x = -1 := by sorry

end NUMINAMATH_CALUDE_only_rational_root_l2449_244985


namespace NUMINAMATH_CALUDE_salt_mixture_proof_l2449_244919

/-- Proves that adding 50 ounces of 60% salt solution to 50 ounces of 20% salt solution results in a 40% salt solution -/
theorem salt_mixture_proof :
  let initial_volume : ℝ := 50
  let initial_concentration : ℝ := 0.20
  let added_volume : ℝ := 50
  let added_concentration : ℝ := 0.60
  let final_concentration : ℝ := 0.40
  let final_volume : ℝ := initial_volume + added_volume
  let initial_salt : ℝ := initial_volume * initial_concentration
  let added_salt : ℝ := added_volume * added_concentration
  let final_salt : ℝ := initial_salt + added_salt
  (final_salt / final_volume) = final_concentration :=
by sorry


end NUMINAMATH_CALUDE_salt_mixture_proof_l2449_244919


namespace NUMINAMATH_CALUDE_combined_shape_perimeter_l2449_244938

/-- The perimeter of a combined shape of a right triangle and rectangle -/
theorem combined_shape_perimeter (a b c d : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 10) 
  (h4 : d^2 = a^2 + b^2) : a + b + c + d = 22 := by
  sorry

end NUMINAMATH_CALUDE_combined_shape_perimeter_l2449_244938


namespace NUMINAMATH_CALUDE_inverse_sum_lower_bound_l2449_244934

theorem inverse_sum_lower_bound (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  1/x + 1/y ≥ 3 + 2*Real.sqrt 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 1 ∧ 1/x₀ + 1/y₀ = 3 + 2*Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_inverse_sum_lower_bound_l2449_244934


namespace NUMINAMATH_CALUDE_exists_sequence_mod_23_l2449_244993

/-- Fibonacci-like sequence -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of a sequence with the desired property -/
theorem exists_sequence_mod_23 : ∃ (F : ℕ → ℤ), 
  F 0 = 0 ∧ 
  F 1 = 1 ∧ 
  (∀ n : ℕ, F (n + 2) = 3 * F (n + 1) - F n) ∧
  F 12 ≡ 0 [ZMOD 23] := by
  sorry


end NUMINAMATH_CALUDE_exists_sequence_mod_23_l2449_244993


namespace NUMINAMATH_CALUDE_irrational_and_no_negative_square_l2449_244929

-- Define p: 2+√2 is irrational
def p : Prop := Irrational (2 + Real.sqrt 2)

-- Define q: ∃ x ∈ ℝ, x^2 < 0
def q : Prop := ∃ x : ℝ, x^2 < 0

-- Theorem statement
theorem irrational_and_no_negative_square : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_irrational_and_no_negative_square_l2449_244929


namespace NUMINAMATH_CALUDE_lowry_big_bonsai_sold_l2449_244952

/-- Represents the sale of bonsai trees -/
structure BonsaiSale where
  small_price : ℕ  -- Price of a small bonsai
  big_price : ℕ    -- Price of a big bonsai
  small_sold : ℕ   -- Number of small bonsai sold
  total_earnings : ℕ -- Total earnings from the sale

/-- Calculates the number of big bonsai sold -/
def big_bonsai_sold (sale : BonsaiSale) : ℕ :=
  (sale.total_earnings - sale.small_price * sale.small_sold) / sale.big_price

/-- Theorem stating the number of big bonsai sold in Lowry's sale -/
theorem lowry_big_bonsai_sold :
  let sale := BonsaiSale.mk 30 20 3 190
  big_bonsai_sold sale = 5 := by
  sorry

end NUMINAMATH_CALUDE_lowry_big_bonsai_sold_l2449_244952


namespace NUMINAMATH_CALUDE_a_over_two_plus_a_is_fraction_l2449_244950

/-- Definition of a fraction -/
def is_fraction (x y : ℝ) : Prop := ∃ (a b : ℝ), x = a ∧ y = b ∧ b ≠ 0

/-- The expression a / (2 + a) is a fraction -/
theorem a_over_two_plus_a_is_fraction (a : ℝ) : is_fraction a (2 + a) := by
  sorry

end NUMINAMATH_CALUDE_a_over_two_plus_a_is_fraction_l2449_244950


namespace NUMINAMATH_CALUDE_base_number_proof_l2449_244937

theorem base_number_proof (n : ℕ) (x : ℝ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^28) 
  (h2 : n = 27) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l2449_244937


namespace NUMINAMATH_CALUDE_specific_value_problem_l2449_244901

theorem specific_value_problem (x : ℕ) (specific_value : ℕ) 
  (h1 : 25 * x = specific_value) 
  (h2 : x = 27) : 
  specific_value = 675 := by
sorry

end NUMINAMATH_CALUDE_specific_value_problem_l2449_244901


namespace NUMINAMATH_CALUDE_proportion_third_term_l2449_244928

/-- Given a proportion 0.75 : 1.65 :: y : 11, prove that y = 5 -/
theorem proportion_third_term (y : ℝ) : 
  (0.75 : ℝ) / 1.65 = y / 11 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_proportion_third_term_l2449_244928


namespace NUMINAMATH_CALUDE_ten_thousandths_digit_of_seven_thirty_seconds_l2449_244933

theorem ten_thousandths_digit_of_seven_thirty_seconds (f : ℚ) (d : ℕ) : 
  f = 7 / 32 →
  d = (⌊f * 10000⌋ % 10) →
  d = 8 :=
by sorry

end NUMINAMATH_CALUDE_ten_thousandths_digit_of_seven_thirty_seconds_l2449_244933


namespace NUMINAMATH_CALUDE_speed_conversion_proof_l2449_244923

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℚ := 3.6

/-- Given speed in km/h -/
def given_speed_kmph : ℝ := 1.5428571428571427

/-- Speed in m/s as a fraction -/
def speed_mps : ℚ := 3/7

theorem speed_conversion_proof :
  (speed_mps : ℝ) * mps_to_kmph = given_speed_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_proof_l2449_244923


namespace NUMINAMATH_CALUDE_total_revenue_is_146475_l2449_244915

/-- The number of cookies baked by Clementine -/
def C : ℕ := 72

/-- The number of cookies baked by Jake -/
def J : ℕ := (5 * C) / 2

/-- The number of cookies baked by Tory -/
def T : ℕ := (J + C) / 2

/-- The number of cookies baked by Spencer -/
def S : ℕ := (3 * (J + T)) / 2

/-- The price of each cookie in cents -/
def price_per_cookie : ℕ := 175

/-- The total revenue in cents -/
def total_revenue : ℕ := (C + J + T + S) * price_per_cookie

theorem total_revenue_is_146475 : total_revenue = 146475 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_146475_l2449_244915


namespace NUMINAMATH_CALUDE_smallest_palindrome_base3_l2449_244995

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def convertBase (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Number of digits of a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_palindrome_base3 :
  ∀ n : ℕ,
  isPalindrome n 3 ∧ numDigits n 3 = 5 →
  (∃ b : ℕ, b ≠ 3 ∧ isPalindrome (convertBase n 3 b) b ∧ numDigits (convertBase n 3 b) b = 3) →
  n ≥ 81 := by
  sorry

end NUMINAMATH_CALUDE_smallest_palindrome_base3_l2449_244995


namespace NUMINAMATH_CALUDE_first_sampling_immediate_l2449_244900

/-- Represents the stages of the yeast population experiment -/
inductive ExperimentStage
  | Inoculation
  | Sampling
  | Counting

/-- Represents the timing of the first sampling test -/
inductive SamplingTiming
  | Immediate
  | Delayed

/-- The correct procedure for the yeast population experiment -/
def correctYeastExperimentProcedure : ExperimentStage → SamplingTiming
  | ExperimentStage.Inoculation => SamplingTiming.Immediate
  | _ => SamplingTiming.Delayed

/-- Theorem stating that the first sampling test should be conducted immediately after inoculation -/
theorem first_sampling_immediate :
  correctYeastExperimentProcedure ExperimentStage.Inoculation = SamplingTiming.Immediate :=
by sorry

end NUMINAMATH_CALUDE_first_sampling_immediate_l2449_244900


namespace NUMINAMATH_CALUDE_union_equals_reals_l2449_244936

-- Define sets A and B
def A : Set ℝ := {x | x > -1}
def B : Set ℝ := {x | x^2 - x - 2 ≥ 0}

-- Theorem statement
theorem union_equals_reals : A ∪ B = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_equals_reals_l2449_244936


namespace NUMINAMATH_CALUDE_eastward_fish_caught_fraction_l2449_244971

/-- Given the following conditions:
  - 1800 fish swim westward
  - 3200 fish swim eastward
  - 500 fish swim north
  - Fishers catch 3/4 of the fish that swam westward
  - There are 2870 fish left in the sea
Prove that the fraction of eastward-swimming fish caught by fishers is 2/5 -/
theorem eastward_fish_caught_fraction :
  let total_fish : ℕ := 1800 + 3200 + 500
  let westward_fish : ℕ := 1800
  let eastward_fish : ℕ := 3200
  let northward_fish : ℕ := 500
  let westward_caught_fraction : ℚ := 3 / 4
  let remaining_fish : ℕ := 2870
  let eastward_caught_fraction : ℚ := 2 / 5
  (total_fish : ℚ) - (westward_caught_fraction * westward_fish + eastward_caught_fraction * eastward_fish) = remaining_fish :=
by
  sorry

#check eastward_fish_caught_fraction

end NUMINAMATH_CALUDE_eastward_fish_caught_fraction_l2449_244971


namespace NUMINAMATH_CALUDE_expression_simplification_l2449_244924

theorem expression_simplification (x y b c d : ℝ) (h : c * y + d * x ≠ 0) :
  (c * y * (b * x^3 + 3 * b * x^2 * y + 3 * b * x * y^2 + b * y^3) + 
   d * x * (c * x^3 + 3 * c * x^2 * y + 3 * c * x * y^2 + c * y^3)) / 
  (c * y + d * x) = (c * x + y)^3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2449_244924


namespace NUMINAMATH_CALUDE_population_increase_rate_example_l2449_244984

/-- Given an initial population and a final population after one year,
    calculate the population increase rate as a percentage. -/
def population_increase_rate (initial : ℕ) (final : ℕ) : ℚ :=
  (final - initial : ℚ) / initial * 100

/-- Theorem stating that for an initial population of 220 and
    a final population of 242, the increase rate is 10%. -/
theorem population_increase_rate_example :
  population_increase_rate 220 242 = 10 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_example_l2449_244984


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l2449_244964

theorem complex_product_magnitude (a b : ℂ) (t : ℝ) :
  Complex.abs a = 3 →
  Complex.abs b = 5 →
  a * b = t - 3 * Complex.I →
  t = 6 * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l2449_244964


namespace NUMINAMATH_CALUDE_even_function_property_l2449_244977

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the main theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_positive : ∀ x > 0, f x = x) : 
  ∀ x < 0, f x = -x :=
by
  sorry


end NUMINAMATH_CALUDE_even_function_property_l2449_244977


namespace NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l2449_244960

theorem unique_solution_for_exponential_equation :
  ∀ a b p : ℕ+,
    p.val.Prime →
    2^(a.val) + p^(b.val) = 19^(a.val) →
    a = 1 ∧ b = 1 ∧ p = 17 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_exponential_equation_l2449_244960


namespace NUMINAMATH_CALUDE_work_break_difference_l2449_244906

/-- Calculates the difference between water breaks and sitting breaks
    given work duration and break intervals. -/
def break_difference (work_duration : ℕ) (water_interval : ℕ) (sitting_interval : ℕ) : ℕ :=
  (work_duration / water_interval) - (work_duration / sitting_interval)

/-- Proves that for 240 minutes of work, with water breaks every 20 minutes
    and sitting breaks every 120 minutes, there are 10 more water breaks than sitting breaks. -/
theorem work_break_difference :
  break_difference 240 20 120 = 10 := by
  sorry

end NUMINAMATH_CALUDE_work_break_difference_l2449_244906


namespace NUMINAMATH_CALUDE_louis_fabric_purchase_l2449_244969

/-- The cost of velvet fabric per yard -/
def fabric_cost_per_yard : ℚ := 24

/-- The cost of the pattern -/
def pattern_cost : ℚ := 15

/-- The total cost of silver thread -/
def thread_cost : ℚ := 6

/-- The total amount spent -/
def total_spent : ℚ := 141

/-- The number of yards of fabric bought -/
def yards_bought : ℚ := (total_spent - pattern_cost - thread_cost) / fabric_cost_per_yard

theorem louis_fabric_purchase : yards_bought = 5 := by
  sorry

end NUMINAMATH_CALUDE_louis_fabric_purchase_l2449_244969


namespace NUMINAMATH_CALUDE_number_of_subsets_l2449_244983

/-- For a finite set with n elements, the number of subsets is 2^n -/
theorem number_of_subsets (S : Type*) [Fintype S] : 
  Finset.card (Finset.powerset (Finset.univ : Finset S)) = 2 ^ Fintype.card S := by
  sorry

end NUMINAMATH_CALUDE_number_of_subsets_l2449_244983


namespace NUMINAMATH_CALUDE_inequality_proof_l2449_244981

theorem inequality_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a - b)^2 / (8*a) < (a + b) / 2 - Real.sqrt (a*b) ∧
  (a + b) / 2 - Real.sqrt (a*b) < (a - b)^2 / (8*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2449_244981


namespace NUMINAMATH_CALUDE_no_solution_sqrt_eq_negative_l2449_244955

theorem no_solution_sqrt_eq_negative :
  ¬∃ x : ℝ, Real.sqrt (5 - x) = -3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_sqrt_eq_negative_l2449_244955


namespace NUMINAMATH_CALUDE_lcm_gcf_problem_l2449_244980

theorem lcm_gcf_problem (n : ℕ) : 
  Nat.lcm n 16 = 48 → Nat.gcd n 16 = 18 → n = 54 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_problem_l2449_244980


namespace NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l2449_244920

/-- Given plane vectors a = (-2, k) and b = (2, 4), if a is perpendicular to b, 
    then |a - b| = 5 -/
theorem perpendicular_vectors_difference_magnitude 
  (k : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (-2, k)) 
  (hb : b = (2, 4)) 
  (hperp : a.1 * b.1 + a.2 * b.2 = 0) : 
  ‖a - b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_difference_magnitude_l2449_244920


namespace NUMINAMATH_CALUDE_first_book_length_l2449_244973

theorem first_book_length :
  ∀ (book1 book2 total_pages daily_pages days : ℕ),
    book2 = 100 →
    days = 14 →
    daily_pages = 20 →
    total_pages = daily_pages * days →
    book1 + book2 = total_pages →
    book1 = 180 := by
sorry

end NUMINAMATH_CALUDE_first_book_length_l2449_244973


namespace NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_28_l2449_244935

theorem largest_four_digit_congruent_to_17_mod_28 :
  ∃ (n : ℕ), n = 9982 ∧ n < 10000 ∧ n ≡ 17 [MOD 28] ∧
  ∀ (m : ℕ), m < 10000 ∧ m ≡ 17 [MOD 28] → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_congruent_to_17_mod_28_l2449_244935


namespace NUMINAMATH_CALUDE_max_percent_x_correct_l2449_244970

/-- The maximum percentage of liquid X in the resulting solution --/
def max_percent_x : ℝ := 1.71

/-- Percentage of liquid X in solution A --/
def percent_x_a : ℝ := 0.8

/-- Percentage of liquid X in solution B --/
def percent_x_b : ℝ := 1.8

/-- Percentage of liquid X in solution C --/
def percent_x_c : ℝ := 3

/-- Percentage of liquid Y in solution A --/
def percent_y_a : ℝ := 2

/-- Percentage of liquid Y in solution B --/
def percent_y_b : ℝ := 1

/-- Percentage of liquid Y in solution C --/
def percent_y_c : ℝ := 0.5

/-- Amount of solution A in grams --/
def amount_a : ℝ := 500

/-- Amount of solution B in grams --/
def amount_b : ℝ := 700

/-- Amount of solution C in grams --/
def amount_c : ℝ := 300

/-- Maximum combined percentage of liquids X and Y in the resulting solution --/
def max_combined_percent : ℝ := 2.5

/-- Theorem stating that the maximum percentage of liquid X in the resulting solution is correct --/
theorem max_percent_x_correct :
  let total_amount := amount_a + amount_b + amount_c
  let amount_x := percent_x_a / 100 * amount_a + percent_x_b / 100 * amount_b + percent_x_c / 100 * amount_c
  let amount_y := percent_y_a / 100 * amount_a + percent_y_b / 100 * amount_b + percent_y_c / 100 * amount_c
  (amount_x + amount_y) / total_amount * 100 ≤ max_combined_percent ∧
  amount_x / total_amount * 100 = max_percent_x :=
by sorry

end NUMINAMATH_CALUDE_max_percent_x_correct_l2449_244970


namespace NUMINAMATH_CALUDE_book_sale_amount_l2449_244986

/-- Calculates the total amount received from selling books given the following conditions:
  * A fraction of the books were sold
  * A certain number of books remained unsold
  * Each sold book was sold at a fixed price
-/
def totalAmountReceived (fractionSold : Rat) (remainingBooks : Nat) (pricePerBook : Rat) : Rat :=
  let totalBooks := remainingBooks / (1 - fractionSold)
  let soldBooks := totalBooks * fractionSold
  soldBooks * pricePerBook

/-- Proves that given the specific conditions of the book sale, 
    the total amount received is $255 -/
theorem book_sale_amount : 
  totalAmountReceived (2/3) 30 (21/5) = 255 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_amount_l2449_244986


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l2449_244916

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 2/3) (h₃ : a₃ = 5/6) :
  arithmetic_sequence a₁ (a₂ - a₁) 10 = 2 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l2449_244916


namespace NUMINAMATH_CALUDE_range_of_f_l2449_244987

noncomputable def f (x : ℝ) : ℝ := (3 * x + 1) / (x - 5)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 3 ∨ y > 3} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2449_244987


namespace NUMINAMATH_CALUDE_algebraic_equality_l2449_244945

theorem algebraic_equality (a b c k m n : ℝ) 
  (h1 : b^2 - n^2 = a^2 - k^2) 
  (h2 : a^2 - k^2 = c^2 - m^2) : 
  (b*m - c*n)/(a - k) + (c*k - a*m)/(b - n) + (a*n - b*k)/(c - m) = 0 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equality_l2449_244945


namespace NUMINAMATH_CALUDE_convex_polyhedron_inequalities_l2449_244974

/-- Represents a convex polyhedron with vertices, edges, and faces. -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler_formula : vertices - edges + faces = 2
  faces_at_least_three_edges : 2 * edges ≥ 3 * faces

/-- The inequalities for convex polyhedrons. -/
theorem convex_polyhedron_inequalities (p : ConvexPolyhedron) :
  (3 * p.vertices ≥ 6 + p.faces) ∧ (3 * p.edges ≥ 6 + p.faces) := by
  sorry

end NUMINAMATH_CALUDE_convex_polyhedron_inequalities_l2449_244974


namespace NUMINAMATH_CALUDE_percentage_of_green_caps_l2449_244956

/-- Calculates the percentage of green bottle caps -/
theorem percentage_of_green_caps 
  (total_caps : ℕ) 
  (red_caps : ℕ) 
  (h1 : total_caps = 125) 
  (h2 : red_caps = 50) 
  (h3 : red_caps ≤ total_caps) : 
  (((total_caps - red_caps) : ℚ) / total_caps) * 100 = 60 := by
  sorry

#check percentage_of_green_caps

end NUMINAMATH_CALUDE_percentage_of_green_caps_l2449_244956


namespace NUMINAMATH_CALUDE_range_of_z_l2449_244989

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  12 ≤ x^2 + 4*y^2 ∧ x^2 + 4*y^2 ≤ 20 := by
  sorry

end NUMINAMATH_CALUDE_range_of_z_l2449_244989


namespace NUMINAMATH_CALUDE_furniture_shop_pricing_l2449_244988

theorem furniture_shop_pricing (cost_price : ℝ) (markup_percentage : ℝ) (selling_price : ℝ) : 
  cost_price = 6525 →
  markup_percentage = 24 →
  selling_price = cost_price * (1 + markup_percentage / 100) →
  selling_price = 8091 := by
sorry

end NUMINAMATH_CALUDE_furniture_shop_pricing_l2449_244988


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l2449_244947

theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 24 * x + c = 0) →  -- exactly one solution
  (a + c = 31) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 9 ∧ c = 22) :=                 -- conclusion
by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l2449_244947


namespace NUMINAMATH_CALUDE_no_nontrivial_integer_solution_l2449_244962

theorem no_nontrivial_integer_solution (a b c d : ℤ) :
  6 * (6 * a ^ 2 + 3 * b ^ 2 + c ^ 2) = 5 * d ^ 2 → a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_nontrivial_integer_solution_l2449_244962


namespace NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2449_244990

def a : ℝ × ℝ := (1, -2)
def b : ℝ → ℝ × ℝ := λ m ↦ (2, m)

theorem perpendicular_vectors_magnitude (m : ℝ) :
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) →
  Real.sqrt ((b m).1^2 + (b m).2^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_magnitude_l2449_244990


namespace NUMINAMATH_CALUDE_circle_M_equation_l2449_244978

-- Define the circle M
structure CircleM where
  center : ℝ × ℝ
  radius : ℝ

-- Define the conditions
axiom center_on_line (M : CircleM) : M.center.2 = -2 * M.center.1

axiom passes_through_A (M : CircleM) :
  (2 - M.center.1)^2 + (-1 - M.center.2)^2 = M.radius^2

axiom tangent_to_line (M : CircleM) :
  |M.center.1 + M.center.2 - 1| / Real.sqrt 2 = M.radius

-- Define the theorem to be proved
theorem circle_M_equation (M : CircleM) :
  ∀ x y : ℝ, (x - 1)^2 + (y + 2)^2 = 2 ↔
    (x - M.center.1)^2 + (y - M.center.2)^2 = M.radius^2 :=
sorry

end NUMINAMATH_CALUDE_circle_M_equation_l2449_244978


namespace NUMINAMATH_CALUDE_population_increase_rate_l2449_244922

/-- If a population increases by 220 persons in 55 minutes at a constant rate,
    then the rate of population increase is 15 seconds per person. -/
theorem population_increase_rate 
  (total_increase : ℕ) 
  (time_minutes : ℕ) 
  (h1 : total_increase = 220)
  (h2 : time_minutes = 55) :
  (time_minutes * 60) / total_increase = 15 := by
  sorry

end NUMINAMATH_CALUDE_population_increase_rate_l2449_244922


namespace NUMINAMATH_CALUDE_total_fish_count_l2449_244975

/-- The number of fish owned by each person -/
def lilly_fish : ℕ := 10
def rosy_fish : ℕ := 11
def alex_fish : ℕ := 15
def jamie_fish : ℕ := 8
def sam_fish : ℕ := 20

/-- Theorem stating that the total number of fish is 64 -/
theorem total_fish_count : 
  lilly_fish + rosy_fish + alex_fish + jamie_fish + sam_fish = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_count_l2449_244975


namespace NUMINAMATH_CALUDE_total_sum_calculation_l2449_244940

/-- Given a sum to be divided among four parts in the ratio 5 : 9 : 6 : 5,
    if the sum of the first and third parts is $7022.222222222222,
    then the total sum is $15959.59595959596. -/
theorem total_sum_calculation (a b c d : ℝ) : 
  a / 5 = b / 9 ∧ a / 5 = c / 6 ∧ a / 5 = d / 5 →
  a + c = 7022.222222222222 →
  a + b + c + d = 15959.59595959596 := by
  sorry

end NUMINAMATH_CALUDE_total_sum_calculation_l2449_244940


namespace NUMINAMATH_CALUDE_average_shirts_per_person_l2449_244957

/-- Represents the average number of shirts made by each person per day -/
def S : ℕ := sorry

/-- The number of employees -/
def employees : ℕ := 20

/-- The number of hours in a shift -/
def shift_hours : ℕ := 8

/-- The hourly wage in dollars -/
def hourly_wage : ℕ := 12

/-- The bonus per shirt made in dollars -/
def bonus_per_shirt : ℕ := 5

/-- The selling price of a shirt in dollars -/
def shirt_price : ℕ := 35

/-- The daily nonemployee expenses in dollars -/
def nonemployee_expenses : ℕ := 1000

/-- The daily profit in dollars -/
def daily_profit : ℕ := 9080

theorem average_shirts_per_person (S : ℕ) :
  S * (shirt_price * employees - bonus_per_shirt * employees) = 
  daily_profit + nonemployee_expenses + employees * shift_hours * hourly_wage →
  S = 20 := by sorry

end NUMINAMATH_CALUDE_average_shirts_per_person_l2449_244957


namespace NUMINAMATH_CALUDE_right_triangle_trig_inequality_l2449_244939

theorem right_triangle_trig_inequality (A B C : ℝ) (h1 : 0 < A) (h2 : A < π/4) 
  (h3 : A + B + C = π) (h4 : C = π/2) : Real.cos B < Real.sin B := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_trig_inequality_l2449_244939


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l2449_244966

theorem gcd_of_three_numbers : Nat.gcd 10234 (Nat.gcd 14322 24570) = 18 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l2449_244966


namespace NUMINAMATH_CALUDE_paving_cost_l2449_244959

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 6.5) (h2 : width = 2.75) (h3 : rate = 600) :
  length * width * rate = 10725 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l2449_244959


namespace NUMINAMATH_CALUDE_cos_120_degrees_l2449_244941

theorem cos_120_degrees : Real.cos (2 * π / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_120_degrees_l2449_244941


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l2449_244942

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- Represents 1111 in any base -/
def digits1111 : List Nat := [1, 1, 1, 1]

/-- The main theorem -/
theorem smallest_base_perfect_square :
  (∀ b : Nat, b > 0 → b < 7 → ¬isPerfectSquare (toBase10 digits1111 b)) ∧
  isPerfectSquare (toBase10 digits1111 7) := by
  sorry

#check smallest_base_perfect_square

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l2449_244942


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l2449_244904

theorem cost_of_dozen_pens 
  (total_cost : ℕ) 
  (pencil_count : ℕ) 
  (pen_cost : ℕ) 
  (pen_pencil_ratio : ℚ) :
  total_cost = 260 →
  pencil_count = 5 →
  pen_cost = 65 →
  pen_pencil_ratio = 5 / 1 →
  (12 : ℕ) * pen_cost = 780 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l2449_244904


namespace NUMINAMATH_CALUDE_total_strawberries_l2449_244907

/-- The number of strawberries picked by Jonathan and Matthew together -/
def jonathan_matthew_total : ℕ := 350

/-- The number of strawberries picked by Matthew and Zac together -/
def matthew_zac_total : ℕ := 250

/-- The number of strawberries picked by Zac alone -/
def zac_alone : ℕ := 200

/-- Theorem stating that the total number of strawberries picked is 550 -/
theorem total_strawberries : 
  ∃ (j m z : ℕ), 
    j + m = jonathan_matthew_total ∧ 
    m + z = matthew_zac_total ∧ 
    z = zac_alone ∧ 
    j + m + z = 550 := by
  sorry

end NUMINAMATH_CALUDE_total_strawberries_l2449_244907


namespace NUMINAMATH_CALUDE_min_b_value_l2449_244958

/-- Given a parabola and a circle with specific intersection properties, 
    the minimum value of b is 2 -/
theorem min_b_value (k a b r : ℝ) : 
  k > 0 → 
  (∀ x y, y = k * x^2 → (x - a)^2 + (y - b)^2 = r^2 → 
    (x = 0 ∧ y = 0) ∨ y = k * x + b) →
  a^2 + b^2 = r^2 →
  (∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
    k * x₁^2 = (x₁ - a)^2 + (k * x₁^2 - b)^2 - r^2 ∧
    k * x₂^2 = (x₂ - a)^2 + (k * x₂^2 - b)^2 - r^2 ∧
    k * x₃^2 = (x₃ - a)^2 + (k * x₃^2 - b)^2 - r^2) →
  b ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_b_value_l2449_244958


namespace NUMINAMATH_CALUDE_prob_different_classes_correct_expected_value_class1_correct_l2449_244994

/-- Represents the number of classes in the first year -/
def num_classes : ℕ := 8

/-- Represents the total number of students selected for the community service group -/
def total_selected : ℕ := 10

/-- Represents the number of students selected from Class 1 -/
def class1_selected : ℕ := 3

/-- Represents the number of students selected from each of the other classes -/
def other_classes_selected : ℕ := 1

/-- Represents the number of students randomly selected for the activity -/
def activity_selected : ℕ := 3

/-- Probability of selecting 3 students from different classes -/
def prob_different_classes : ℚ := 49/60

/-- Expected value of the number of students selected from Class 1 -/
def expected_value_class1 : ℚ := 43/40

/-- Theorem stating the probability of selecting 3 students from different classes -/
theorem prob_different_classes_correct :
  let total_ways := Nat.choose total_selected activity_selected
  let ways_with_one_from_class1 := Nat.choose class1_selected 1 * Nat.choose (total_selected - class1_selected) 2
  let ways_with_none_from_class1 := Nat.choose class1_selected 0 * Nat.choose (total_selected - class1_selected) 3
  (ways_with_one_from_class1 + ways_with_none_from_class1) / total_ways = prob_different_classes :=
sorry

/-- Theorem stating the expected value of the number of students selected from Class 1 -/
theorem expected_value_class1_correct :
  let p0 := (7 : ℚ) / 24
  let p1 := (21 : ℚ) / 40
  let p2 := (7 : ℚ) / 40
  let p3 := (1 : ℚ) / 120
  0 * p0 + 1 * p1 + 2 * p2 + 3 * p3 = expected_value_class1 :=
sorry

end NUMINAMATH_CALUDE_prob_different_classes_correct_expected_value_class1_correct_l2449_244994


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2449_244967

theorem inequality_equivalence (x : ℝ) (h : x > 0) : 
  3/8 + |x - 14/24| < 8/12 ↔ 7/24 < x ∧ x < 7/8 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2449_244967


namespace NUMINAMATH_CALUDE_min_sum_absolute_values_l2449_244976

theorem min_sum_absolute_values : ∀ x : ℝ, 
  |x + 3| + |x + 5| + |x + 6| ≥ 5 ∧ 
  ∃ y : ℝ, |y + 3| + |y + 5| + |y + 6| = 5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_values_l2449_244976


namespace NUMINAMATH_CALUDE_intersection_points_on_line_l2449_244911

/-- The slope of the line containing all intersection points of the given parametric lines -/
def intersection_line_slope : ℚ := 10/31

/-- The first line equation: 2x + 3y = 8u + 4 -/
def line1 (u x y : ℝ) : Prop := 2*x + 3*y = 8*u + 4

/-- The second line equation: 3x - 2y = 5u - 3 -/
def line2 (u x y : ℝ) : Prop := 3*x - 2*y = 5*u - 3

/-- The theorem stating that all intersection points lie on a line with slope 10/31 -/
theorem intersection_points_on_line :
  ∀ (u x y : ℝ), line1 u x y → line2 u x y →
  ∃ (k b : ℝ), y = intersection_line_slope * x + b :=
sorry

end NUMINAMATH_CALUDE_intersection_points_on_line_l2449_244911


namespace NUMINAMATH_CALUDE_max_value_of_expression_l2449_244914

theorem max_value_of_expression (x : ℝ) : 
  (4 * x^2 + 8 * x + 19) / (4 * x^2 + 8 * x + 5) ≤ 15 ∧ 
  ∃ y : ℝ, (4 * y^2 + 8 * y + 19) / (4 * y^2 + 8 * y + 5) = 15 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l2449_244914


namespace NUMINAMATH_CALUDE_power_division_equality_l2449_244917

theorem power_division_equality (a : ℝ) : (-2 * a^2)^3 / (2 * a^2) = -4 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_power_division_equality_l2449_244917


namespace NUMINAMATH_CALUDE_existence_of_prime_and_integer_l2449_244946

theorem existence_of_prime_and_integer (p : ℕ) (hp : p.Prime) (hp_gt_5 : p > 5) :
  ∃ (q : ℕ) (n : ℕ+), q.Prime ∧ q < p ∧ p ∣ n.val^2 - q := by
  sorry

end NUMINAMATH_CALUDE_existence_of_prime_and_integer_l2449_244946


namespace NUMINAMATH_CALUDE_simplify_2A_minus_3B_specific_value_2A_minus_3B_l2449_244982

/-- Definition of A in terms of a and b -/
def A (a b : ℝ) : ℝ := 3 * b^2 - 2 * a^2 + 5 * a * b

/-- Definition of B in terms of a and b -/
def B (a b : ℝ) : ℝ := 4 * a * b + 2 * b^2 - a^2

/-- Theorem stating that 2A - 3B simplifies to -a² - 2ab for all real a and b -/
theorem simplify_2A_minus_3B (a b : ℝ) : 2 * A a b - 3 * B a b = -a^2 - 2*a*b := by
  sorry

/-- Theorem stating that when a = -1 and b = 4, 2A - 3B equals 7 -/
theorem specific_value_2A_minus_3B : 2 * A (-1) 4 - 3 * B (-1) 4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_2A_minus_3B_specific_value_2A_minus_3B_l2449_244982


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2449_244943

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

def has_four_consecutive_terms (b : ℕ → ℝ) (S : Set ℝ) : Prop :=
  ∃ k : ℕ, (b k ∈ S) ∧ (b (k + 1) ∈ S) ∧ (b (k + 2) ∈ S) ∧ (b (k + 3) ∈ S)

theorem geometric_sequence_ratio (a b : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  (∀ n : ℕ, b n = a n + 1) →
  has_four_consecutive_terms b {-53, -23, 19, 37, 82} →
  abs q > 1 →
  q = -3/2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2449_244943


namespace NUMINAMATH_CALUDE_four_roots_implies_a_in_interval_l2449_244991

-- Define the polynomial
def P (x a : ℝ) : ℝ := x^4 + 8*x^3 + 18*x^2 + 8*x + a

-- Define the property of having four distinct real roots
def has_four_distinct_real_roots (a : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (P x₁ a = 0 ∧ P x₂ a = 0 ∧ P x₃ a = 0 ∧ P x₄ a = 0)

-- Theorem statement
theorem four_roots_implies_a_in_interval :
  ∀ a : ℝ, has_four_distinct_real_roots a → a ∈ Set.Ioo (-8 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_four_roots_implies_a_in_interval_l2449_244991


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l2449_244948

theorem smaller_solution_quadratic (x : ℝ) : 
  x^2 + 9*x - 22 = 0 ∧ (∀ y : ℝ, y^2 + 9*y - 22 = 0 → x ≤ y) → x = -11 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l2449_244948


namespace NUMINAMATH_CALUDE_inequalities_proof_l2449_244996

theorem inequalities_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2*b = 3) :
  (a^2 + b^2 ≥ 9/5) ∧ (a^3*b + 4*a*b^3 ≤ 81/16) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_proof_l2449_244996


namespace NUMINAMATH_CALUDE_fourth_term_of_specific_sequence_l2449_244944

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℝ
  common_ratio : ℝ

/-- The nth term of a geometric sequence -/
def nth_term (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem fourth_term_of_specific_sequence :
  ∃ (seq : GeometricSequence),
    seq.first_term = 512 ∧
    nth_term seq 6 = 32 ∧
    nth_term seq 4 = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_specific_sequence_l2449_244944


namespace NUMINAMATH_CALUDE_ratio_fraction_value_l2449_244912

-- Define the ratio condition
def ratio_condition (X Y Z : ℚ) : Prop :=
  X / Y = 3 / 2 ∧ Y / Z = 2 / 6

-- State the theorem
theorem ratio_fraction_value (X Y Z : ℚ) (h : ratio_condition X Y Z) :
  (2 * X + 3 * Y) / (5 * Z - 2 * X) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_fraction_value_l2449_244912


namespace NUMINAMATH_CALUDE_hannah_dog_food_l2449_244918

/-- The amount of dog food Hannah needs to prepare for her three dogs in a day -/
def total_dog_food (first_dog_food : ℝ) (second_dog_multiplier : ℝ) (third_dog_extra : ℝ) : ℝ :=
  first_dog_food + 
  (first_dog_food * second_dog_multiplier) + 
  (first_dog_food * second_dog_multiplier + third_dog_extra)

/-- Theorem stating that Hannah needs to prepare 10 cups of dog food for her three dogs -/
theorem hannah_dog_food : total_dog_food 1.5 2 2.5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_hannah_dog_food_l2449_244918


namespace NUMINAMATH_CALUDE_opera_ticket_price_increase_l2449_244931

theorem opera_ticket_price_increase (old_price new_price : ℝ) 
  (h1 : old_price = 85)
  (h2 : new_price = 102) : 
  (new_price - old_price) / old_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_opera_ticket_price_increase_l2449_244931


namespace NUMINAMATH_CALUDE_factorization_of_2x2_minus_2y2_l2449_244999

theorem factorization_of_2x2_minus_2y2 (x y : ℝ) : 2 * x^2 - 2 * y^2 = 2 * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2x2_minus_2y2_l2449_244999


namespace NUMINAMATH_CALUDE_expected_abs_difference_10_days_l2449_244968

/-- Represents the wealth difference between two entities -/
def WealthDifference := ℤ

/-- Probability of each outcome -/
def p_cat_wins : ℝ := 0.25
def p_fox_wins : ℝ := 0.25
def p_both_lose : ℝ := 0.5

/-- Number of days -/
def num_days : ℕ := 10

/-- Expected value of absolute wealth difference after n days -/
def expected_abs_difference (n : ℕ) : ℝ := sorry

/-- Theorem: Expected absolute wealth difference after 10 days is 1 -/
theorem expected_abs_difference_10_days :
  expected_abs_difference num_days = 1 := by sorry

end NUMINAMATH_CALUDE_expected_abs_difference_10_days_l2449_244968


namespace NUMINAMATH_CALUDE_melanie_plums_l2449_244927

/-- The number of plums picked by different people and in total -/
structure PlumPicking where
  dan : ℕ
  sally : ℕ
  total : ℕ

/-- The theorem stating how many plums Melanie picked -/
theorem melanie_plums (p : PlumPicking) (h1 : p.dan = 9) (h2 : p.sally = 3) (h3 : p.total = 16) :
  p.total - (p.dan + p.sally) = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_plums_l2449_244927


namespace NUMINAMATH_CALUDE_polynomial_root_mean_l2449_244921

theorem polynomial_root_mean (a b c d k : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  (k - a) * (k - b) * (k - c) * (k - d) = 4 →
  k = (a + b + c + d) / 4 := by
sorry

end NUMINAMATH_CALUDE_polynomial_root_mean_l2449_244921


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2449_244949

/-- The set of complex numbers z satisfying the equation (1/5)^|z-3| = (1/5)^(|z+3|-1) 
    forms a hyperbola with foci on the x-axis, a real semi-axis length of 1/2, 
    and specifically represents the right branch. -/
theorem hyperbola_equation (z : ℂ) : 
  (1/5 : ℝ) ^ Complex.abs (z - 3) = (1/5 : ℝ) ^ (Complex.abs (z + 3) - 1) →
  ∃ (a : ℝ), a = 1/2 ∧ 
    Complex.abs (z + 3) - Complex.abs (z - 3) = 2 * a ∧
    z.re > 0 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2449_244949


namespace NUMINAMATH_CALUDE_f_is_h_function_l2449_244926

def is_h_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x₁ x₂, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁)

def f (x : ℝ) : ℝ := x * |x|

theorem f_is_h_function : is_h_function f := by sorry

end NUMINAMATH_CALUDE_f_is_h_function_l2449_244926


namespace NUMINAMATH_CALUDE_coefficient_x2y3_in_binomial_expansion_l2449_244972

theorem coefficient_x2y3_in_binomial_expansion :
  (Finset.range 6).sum (fun k => Nat.choose 5 k * (if k = 3 then 1 else 0)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y3_in_binomial_expansion_l2449_244972


namespace NUMINAMATH_CALUDE_line_moved_down_l2449_244965

/-- Given a line y = -x + 1 moved down 3 units, prove that the resulting line is y = -x - 2 -/
theorem line_moved_down (x y : ℝ) :
  (y = -x + 1) → (y - 3 = -x - 2) := by
  sorry

end NUMINAMATH_CALUDE_line_moved_down_l2449_244965
