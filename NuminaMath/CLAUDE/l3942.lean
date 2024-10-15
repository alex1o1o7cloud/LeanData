import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_inequalities_l3942_394276

theorem quadratic_inequalities (a : ℝ) :
  ((∀ x : ℝ, x^2 + a*x + 3 ≥ a) ↔ a ∈ Set.Icc (-6) 2) ∧
  ((∃ x : ℝ, x < 1 ∧ x^2 + a*x + 3 ≤ a) ↔ a ∈ Set.Ici 2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l3942_394276


namespace NUMINAMATH_CALUDE_loan_amount_to_c_l3942_394271

/-- Represents the loan details and interest calculation --/
structure LoanDetails where
  amount_b : ℝ  -- Amount lent to B
  amount_c : ℝ  -- Amount lent to C (to be determined)
  years_b : ℝ   -- Years for B's loan
  years_c : ℝ   -- Years for C's loan
  rate : ℝ      -- Annual interest rate
  total_interest : ℝ  -- Total interest received from both B and C

/-- Calculates the simple interest --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The main theorem to prove --/
theorem loan_amount_to_c 
  (loan : LoanDetails) 
  (h1 : loan.amount_b = 5000)
  (h2 : loan.years_b = 2)
  (h3 : loan.years_c = 4)
  (h4 : loan.rate = 0.09)
  (h5 : loan.total_interest = 1980)
  (h6 : simple_interest loan.amount_b loan.rate loan.years_b + 
        simple_interest loan.amount_c loan.rate loan.years_c = loan.total_interest) :
  loan.amount_c = 500 := by
  sorry


end NUMINAMATH_CALUDE_loan_amount_to_c_l3942_394271


namespace NUMINAMATH_CALUDE_rectangle_not_cuttable_from_square_cannot_cut_rectangle_from_square_l3942_394275

/-- Proves that a rectangle with area 30 and length-to-width ratio 2:1 cannot be cut from a square with area 36 -/
theorem rectangle_not_cuttable_from_square : 
  ∀ (rect_length rect_width square_side : ℝ),
  rect_length > 0 → rect_width > 0 → square_side > 0 →
  rect_length * rect_width = 30 →
  rect_length = 2 * rect_width →
  square_side * square_side = 36 →
  rect_length > square_side :=
by sorry

/-- Concludes that the rectangular piece cannot be cut from the square piece -/
theorem cannot_cut_rectangle_from_square : 
  ∃ (rect_length rect_width square_side : ℝ),
  rect_length > 0 ∧ rect_width > 0 ∧ square_side > 0 ∧
  rect_length * rect_width = 30 ∧
  rect_length = 2 * rect_width ∧
  square_side * square_side = 36 ∧
  rect_length > square_side :=
by sorry

end NUMINAMATH_CALUDE_rectangle_not_cuttable_from_square_cannot_cut_rectangle_from_square_l3942_394275


namespace NUMINAMATH_CALUDE_smallest_n_value_l3942_394289

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 3000 ∧ Even a

def factorial_product_divisibility (a b c n : ℕ) (m : ℤ) : Prop :=
  ∃ (k : ℤ), (a.factorial * b.factorial * c.factorial : ℤ) = m * 10^n ∧ ¬(10 ∣ m)

theorem smallest_n_value (a b c : ℕ) (m : ℤ) :
  is_valid_triple a b c →
  (∃ n : ℕ, factorial_product_divisibility a b c n m) →
  ∃ n : ℕ, factorial_product_divisibility a b c n m ∧
    ∀ k : ℕ, factorial_product_divisibility a b c k m → n ≤ k ∧ n = 496 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_value_l3942_394289


namespace NUMINAMATH_CALUDE_no_balloons_remain_intact_l3942_394229

/-- Represents the state of balloons in a hot air balloon --/
structure BalloonState where
  total : ℕ
  intact : ℕ
  doubleDurable : ℕ

/-- Calculates the number of intact balloons after the first 30 minutes --/
def afterFirstHalfHour (initial : ℕ) : ℕ :=
  initial - (initial / 5)

/-- Calculates the number of intact balloons after the next hour --/
def afterNextHour (intact : ℕ) : ℕ :=
  intact - (intact * 3 / 10)

/-- Calculates the number of double durable balloons --/
def doubleDurableBalloons (intact : ℕ) : ℕ :=
  intact / 10

/-- Calculates the final number of intact balloons --/
def finalIntactBalloons (state : BalloonState) : ℕ :=
  let nonDurableBlownUp := state.total - state.intact
  let toBlowUp := min (2 * (nonDurableBlownUp - state.doubleDurable)) state.intact
  state.intact - toBlowUp

/-- Main theorem: After all events, no balloons remain intact --/
theorem no_balloons_remain_intact (initialBalloons : ℕ) 
    (h1 : initialBalloons = 200) : 
    finalIntactBalloons 
      { total := initialBalloons,
        intact := afterNextHour (afterFirstHalfHour initialBalloons),
        doubleDurable := doubleDurableBalloons (afterNextHour (afterFirstHalfHour initialBalloons)) } = 0 := by
  sorry

#eval finalIntactBalloons 
  { total := 200,
    intact := afterNextHour (afterFirstHalfHour 200),
    doubleDurable := doubleDurableBalloons (afterNextHour (afterFirstHalfHour 200)) }

end NUMINAMATH_CALUDE_no_balloons_remain_intact_l3942_394229


namespace NUMINAMATH_CALUDE_square_sum_identity_l3942_394212

theorem square_sum_identity (y : ℝ) :
  (y - 2)^2 + 2*(y - 2)*(4 + y) + (4 + y)^2 = 4*(y + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_identity_l3942_394212


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l3942_394240

theorem quadratic_form_ratio (k : ℝ) :
  ∃ (c r s : ℝ), 10 * k^2 - 6 * k + 20 = c * (k + r)^2 + s ∧ s / r = -191 / 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l3942_394240


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3942_394202

theorem rectangle_circle_area_ratio :
  ∀ (w l r : ℝ),
  w > 0 → l > 0 → r > 0 →
  l = 2 * w →
  2 * l + 2 * w = 2 * π * r →
  (l * w) / (π * r^2) = 2 * π / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l3942_394202


namespace NUMINAMATH_CALUDE_pigs_joined_l3942_394268

/-- Given an initial number of pigs and a final number of pigs,
    prove that the number of pigs that joined is equal to their difference. -/
theorem pigs_joined (initial final : ℕ) (h : final ≥ initial) :
  final - initial = final - initial :=
by sorry

end NUMINAMATH_CALUDE_pigs_joined_l3942_394268


namespace NUMINAMATH_CALUDE_binary_representation_of_25_l3942_394251

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The binary representation of 25 -/
def binary25 : List Bool := [true, false, false, true, true]

/-- Theorem stating that the binary representation of 25 is [1,1,0,0,1] -/
theorem binary_representation_of_25 : toBinary 25 = binary25 := by
  sorry

end NUMINAMATH_CALUDE_binary_representation_of_25_l3942_394251


namespace NUMINAMATH_CALUDE_kiwis_equal_lemons_l3942_394286

/-- Represents the contents of a fruit basket -/
structure FruitBasket where
  mangoes : Nat
  pears : Nat
  pawpaws : Nat
  lemons : Nat
  kiwis : Nat

/-- Represents Tania's collection of fruit baskets -/
def TaniaBaskets : List FruitBasket :=
  [
    { mangoes := 18, pears := 0, pawpaws := 0, lemons := 0, kiwis := 0 },
    { mangoes := 0, pears := 10, pawpaws := 0, lemons := 0, kiwis := 0 },
    { mangoes := 0, pears := 0, pawpaws := 12, lemons := 0, kiwis := 0 },
    { mangoes := 0, pears := 0, pawpaws := 0, lemons := 0, kiwis := 0 },
    { mangoes := 0, pears := 0, pawpaws := 0, lemons := 0, kiwis := 0 }
  ]

/-- The total number of fruits in all baskets -/
def totalFruits : Nat := 58

/-- The total number of lemons in all baskets -/
def totalLemons : Nat := 9

/-- Theorem stating that the number of kiwis equals the number of lemons in the last two baskets -/
theorem kiwis_equal_lemons (h1 : List.length TaniaBaskets = 5)
    (h2 : (TaniaBaskets.map (fun b => b.mangoes + b.pears + b.pawpaws + b.lemons + b.kiwis)).sum = totalFruits)
    (h3 : (TaniaBaskets.map (fun b => b.lemons)).sum = totalLemons) :
    (List.take 2 (List.reverse TaniaBaskets)).map (fun b => b.kiwis) = 
    (List.take 2 (List.reverse TaniaBaskets)).map (fun b => b.lemons) := by
  sorry

end NUMINAMATH_CALUDE_kiwis_equal_lemons_l3942_394286


namespace NUMINAMATH_CALUDE_problem_statement_l3942_394233

theorem problem_statement (a b : ℝ) 
  (h1 : |a| = 5)
  (h2 : |b| = 7)
  (h3 : |a + b| = a + b) :
  a - b = -2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3942_394233


namespace NUMINAMATH_CALUDE_equation_solution_l3942_394241

theorem equation_solution :
  ∃ (x : ℚ), x ≠ -2 ∧ (x^2 + 2*x + 2) / (x + 2) = x + 3 ∧ x = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3942_394241


namespace NUMINAMATH_CALUDE_probability_two_girls_l3942_394281

theorem probability_two_girls (p : ℝ) (h1 : p = 1 / 2) : p * p = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_girls_l3942_394281


namespace NUMINAMATH_CALUDE_tangent_cotangent_identity_l3942_394234

theorem tangent_cotangent_identity (α : Real) 
  (h1 : 0 < α) (h2 : α < π/2) (h3 : α ≠ π/4) :
  (Real.sqrt (Real.tan α) + Real.sqrt (1 / Real.tan α)) / 
  (Real.sqrt (Real.tan α) - Real.sqrt (1 / Real.tan α)) = 
  1 / Real.tan (α - π/4) := by
  sorry

end NUMINAMATH_CALUDE_tangent_cotangent_identity_l3942_394234


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_ratio_l3942_394265

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (2, sin θ) and b = (1, cos θ), then sin²θ / (1 + cos²θ) = 2/3 -/
theorem parallel_vectors_trig_ratio 
  (θ : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (2, Real.sin θ)) 
  (hb : b = (1, Real.cos θ)) 
  (h_parallel : ∃ (k : ℝ), a = k • b) : 
  (Real.sin θ)^2 / (1 + (Real.cos θ)^2) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_ratio_l3942_394265


namespace NUMINAMATH_CALUDE_coopers_age_l3942_394294

theorem coopers_age (cooper_age dante_age maria_age : ℕ) : 
  cooper_age + dante_age + maria_age = 31 →
  dante_age = 2 * cooper_age →
  maria_age = dante_age + 1 →
  cooper_age = 6 := by
sorry

end NUMINAMATH_CALUDE_coopers_age_l3942_394294


namespace NUMINAMATH_CALUDE_complex_abs_one_plus_i_over_i_l3942_394245

theorem complex_abs_one_plus_i_over_i (i : ℂ) : i * i = -1 → Complex.abs ((1 + i) / i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_abs_one_plus_i_over_i_l3942_394245


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_z_on_y_equals_x_l3942_394236

/-- The real part of the complex number z -/
def real_part (m : ℝ) : ℝ := m^2 - 8*m + 15

/-- The imaginary part of the complex number z -/
def imag_part (m : ℝ) : ℝ := m^2 - 5*m - 14

/-- The complex number z -/
def z (m : ℝ) : ℂ := Complex.mk (real_part m) (imag_part m)

/-- Condition for z to be in the fourth quadrant -/
def in_fourth_quadrant (m : ℝ) : Prop :=
  real_part m > 0 ∧ imag_part m < 0

/-- Condition for z to be on the line y = x -/
def on_y_equals_x (m : ℝ) : Prop :=
  real_part m = imag_part m

theorem z_in_fourth_quadrant :
  ∀ m : ℝ, in_fourth_quadrant m ↔ (-2 < m ∧ m < 3) ∨ (5 < m ∧ m < 7) :=
sorry

theorem z_on_y_equals_x :
  ∀ m : ℝ, on_y_equals_x m ↔ m = 29/3 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_z_on_y_equals_x_l3942_394236


namespace NUMINAMATH_CALUDE_product_of_x_and_y_l3942_394205

theorem product_of_x_and_y (x y : ℝ) (h1 : 3 * x + 4 * y = 60) (h2 : 6 * x - 4 * y = 12) :
  x * y = 72 := by
  sorry

end NUMINAMATH_CALUDE_product_of_x_and_y_l3942_394205


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3942_394263

theorem trigonometric_identities (x : Real) 
  (h1 : -π < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  ((3 * (Real.sin (x/2))^2 - 2 * Real.sin (x/2) * Real.cos (x/2) + (Real.cos (x/2))^2) / 
   (Real.tan x + 1 / Real.tan x) = -132/125) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3942_394263


namespace NUMINAMATH_CALUDE_money_problem_l3942_394239

theorem money_problem (a b : ℝ) 
  (h1 : 5 * a + 2 * b > 100)
  (h2 : 4 * a - b = 40) : 
  a > 180 / 13 ∧ b > 200 / 13 := by
  sorry

end NUMINAMATH_CALUDE_money_problem_l3942_394239


namespace NUMINAMATH_CALUDE_power_difference_l3942_394279

theorem power_difference (a : ℕ) (h : 5^a = 3125) : 5^(a-3) = 25 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l3942_394279


namespace NUMINAMATH_CALUDE_mary_eggs_l3942_394292

/-- Given that Mary starts with 27 eggs and finds 4 more, prove that she ends up with 31 eggs. -/
theorem mary_eggs :
  let initial_eggs : ℕ := 27
  let found_eggs : ℕ := 4
  let final_eggs : ℕ := initial_eggs + found_eggs
  final_eggs = 31 := by
sorry

end NUMINAMATH_CALUDE_mary_eggs_l3942_394292


namespace NUMINAMATH_CALUDE_pipe_fill_time_l3942_394293

/-- The time it takes for the second pipe to empty the tank -/
def empty_time : ℝ := 24

/-- The time after which the second pipe is closed when both pipes are open -/
def close_time : ℝ := 48

/-- The total time it takes to fill the tank -/
def total_fill_time : ℝ := 30

/-- The time it takes for the first pipe to fill the tank -/
def fill_time : ℝ := 22

theorem pipe_fill_time : 
  (close_time * (1 / fill_time - 1 / empty_time) + (total_fill_time - close_time) * (1 / fill_time) = 1) →
  fill_time = 22 := by sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l3942_394293


namespace NUMINAMATH_CALUDE_circle_equation_l3942_394247

/-- A circle with center (2, -3) passing through the origin has the equation (x - 2)^2 + (y + 3)^2 = 13 -/
theorem circle_equation (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 
    ((x - 2)^2 + (y + 3)^2 = r^2) ∧ 
    (0 - 2)^2 + (0 + 3)^2 = r^2) ↔ 
  (x - 2)^2 + (y + 3)^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3942_394247


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l3942_394231

theorem max_value_of_trigonometric_expression :
  let f : ℝ → ℝ := λ x => Real.sin (x + 3 * Real.pi / 4) + Real.cos (x + Real.pi / 3) + Real.cos (x + Real.pi / 4)
  let max_value := 2 * Real.cos (-Real.pi / 24)
  ∀ x ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 4), f x ≤ max_value ∧ ∃ x₀ ∈ Set.Icc (-Real.pi / 2) (-Real.pi / 4), f x₀ = max_value :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_expression_l3942_394231


namespace NUMINAMATH_CALUDE_rectangle_area_l3942_394230

theorem rectangle_area (width length perimeter area : ℝ) : 
  length = 4 * width →
  perimeter = 2 * (length + width) →
  perimeter = 200 →
  area = length * width →
  area = 1600 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3942_394230


namespace NUMINAMATH_CALUDE_candy_division_ways_l3942_394214

def divide_candies (total : ℕ) (min_per_person : ℕ) : ℕ :=
  total - 2 * min_per_person + 1

theorem candy_division_ways :
  divide_candies 8 1 = 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_ways_l3942_394214


namespace NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l3942_394298

theorem square_area_to_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ > 0 ∧ s₂ > 0) :
  s₁^2 / s₂^2 = 16 / 49 → (4 * s₁) / (4 * s₂) = 4 / 7 := by
sorry

end NUMINAMATH_CALUDE_square_area_to_perimeter_ratio_l3942_394298


namespace NUMINAMATH_CALUDE_special_polynomial_B_value_l3942_394250

/-- A polynomial of degree 5 with specific properties -/
structure SpecialPolynomial where
  A : ℤ
  B : ℤ
  C : ℤ
  roots : Finset ℤ
  roots_positive : ∀ r ∈ roots, r > 0
  roots_sum : (roots.sum id) = 15
  roots_card : roots.card = 5
  is_root : ∀ r ∈ roots, r^5 - 15*r^4 + A*r^3 + B*r^2 + C*r + 24 = 0

/-- The coefficient B in the special polynomial is -90 -/
theorem special_polynomial_B_value (p : SpecialPolynomial) : p.B = -90 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_B_value_l3942_394250


namespace NUMINAMATH_CALUDE_max_a_value_l3942_394249

theorem max_a_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y + 6 = 4 * x * y) :
  ∃ (a : ℝ), ∀ (b : ℝ), (∀ (u v : ℝ), u > 0 → v > 0 → u + v + 6 = 4 * u * v →
    u^2 + 2 * u * v + v^2 - b * u - b * v + 1 ≥ 0) → b ≤ a ∧ a = 10 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l3942_394249


namespace NUMINAMATH_CALUDE_problem_solution_l3942_394282

theorem problem_solution (a : ℝ) (h : a = 1 / (Real.sqrt 2 - 1)) : 
  (a = Real.sqrt 2 + 1) ∧ 
  (a^2 - 2*a = 1) ∧ 
  (2*a^3 - 4*a^2 - 1 = 2 * Real.sqrt 2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3942_394282


namespace NUMINAMATH_CALUDE_range_of_m_l3942_394280

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + m ≤ 0

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (3-m)^x < (3-m)^y

-- Define the compound statements
def p_or_q (m : ℝ) : Prop := p m ∨ q m

def p_and_q (m : ℝ) : Prop := p m ∧ q m

-- State the theorem
theorem range_of_m : 
  ∀ m : ℝ, (p_or_q m ∧ ¬(p_and_q m)) → (1 < m ∧ m < 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3942_394280


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_13_l3942_394262

/-- A number ends in 6 if it's of the form 10n + 6 for some integer n -/
def ends_in_6 (x : ℕ) : Prop := ∃ n : ℕ, x = 10 * n + 6

/-- A number is divisible by 13 if there exists an integer k such that x = 13k -/
def divisible_by_13 (x : ℕ) : Prop := ∃ k : ℕ, x = 13 * k

theorem smallest_positive_integer_ending_in_6_divisible_by_13 :
  (ends_in_6 26 ∧ divisible_by_13 26) ∧
  ∀ x : ℕ, 0 < x ∧ x < 26 → ¬(ends_in_6 x ∧ divisible_by_13 x) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_6_divisible_by_13_l3942_394262


namespace NUMINAMATH_CALUDE_arithmetic_equality_l3942_394243

theorem arithmetic_equality : 57 * 44 + 13 * 44 = 3080 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l3942_394243


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l3942_394264

theorem parametric_to_ordinary_equation 
  (t : ℝ) (x y : ℝ) 
  (h1 : t ≥ 0) 
  (h2 : x = Real.sqrt t + 1) 
  (h3 : y = 2 * Real.sqrt t - 1) : 
  y = 2 * x - 3 ∧ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l3942_394264


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_3_l3942_394296

theorem sqrt_meaningful_iff_x_geq_3 (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_3_l3942_394296


namespace NUMINAMATH_CALUDE_flash_drive_problem_l3942_394209

/-- Represents the number of flash drives needed to store files -/
def min_flash_drives (total_files : ℕ) (drive_capacity : ℚ) 
  (file_sizes : List (ℕ × ℚ)) : ℕ :=
  sorry

/-- The problem statement -/
theorem flash_drive_problem :
  let total_files : ℕ := 40
  let drive_capacity : ℚ := 2
  let file_sizes : List (ℕ × ℚ) := [(4, 1.2), (16, 0.9), (20, 0.6)]
  min_flash_drives total_files drive_capacity file_sizes = 20 := by
  sorry

end NUMINAMATH_CALUDE_flash_drive_problem_l3942_394209


namespace NUMINAMATH_CALUDE_lcm_problem_l3942_394207

theorem lcm_problem (n : ℕ) (h1 : n > 0) (h2 : Nat.lcm 30 n = 90) (h3 : Nat.lcm n 45 = 180) : n = 36 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l3942_394207


namespace NUMINAMATH_CALUDE_chicken_count_l3942_394220

theorem chicken_count (coop run free_range : ℕ) : 
  coop = 14 →
  run = 2 * coop →
  free_range = 2 * run - 4 →
  free_range = 52 := by
sorry

end NUMINAMATH_CALUDE_chicken_count_l3942_394220


namespace NUMINAMATH_CALUDE_odd_even_properties_l3942_394291

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem odd_even_properties (f g : ℝ → ℝ) (h1 : is_odd f) (h2 : is_even g) :
  (∀ x, (|f x| + g x) = (|f (-x)| + g (-x))) ∧
  (∀ x, f x * |g x| = -(f (-x) * |g (-x)|)) :=
sorry

end NUMINAMATH_CALUDE_odd_even_properties_l3942_394291


namespace NUMINAMATH_CALUDE_delta_max_success_ratio_l3942_394255

/-- Represents a player's score in a chess competition --/
structure PlayerScore where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ

/-- Calculate the success ratio for a given day --/
def day_success_ratio (score : ℕ) (total : ℕ) : ℚ :=
  ↑score / ↑total

/-- Calculate the overall success ratio --/
def overall_success_ratio (player : PlayerScore) : ℚ :=
  ↑(player.day1_score + player.day2_score) / ↑(player.day1_total + player.day2_total)

theorem delta_max_success_ratio 
  (gamma : PlayerScore)
  (delta : PlayerScore)
  (h1 : gamma.day1_score = 180 ∧ gamma.day1_total = 360)
  (h2 : gamma.day2_score = 150 ∧ gamma.day2_total = 240)
  (h3 : delta.day1_total + delta.day2_total = 600)
  (h4 : delta.day1_total ≠ 360)
  (h5 : delta.day1_score > 0 ∧ delta.day2_score > 0)
  (h6 : day_success_ratio delta.day1_score delta.day1_total < day_success_ratio gamma.day1_score gamma.day1_total)
  (h7 : day_success_ratio delta.day2_score delta.day2_total < day_success_ratio gamma.day2_score gamma.day2_total)
  (h8 : overall_success_ratio gamma = 11/20) :
  overall_success_ratio delta ≤ 599/600 :=
sorry

end NUMINAMATH_CALUDE_delta_max_success_ratio_l3942_394255


namespace NUMINAMATH_CALUDE_fraction_say_dislike_actually_like_l3942_394253

def TotalStudents : ℝ := 100

def LikeDancing : ℝ := 0.6 * TotalStudents
def DislikeDancing : ℝ := 0.4 * TotalStudents

def SayLikeActuallyLike : ℝ := 0.8 * LikeDancing
def SayDislikeActuallyLike : ℝ := 0.2 * LikeDancing
def SayDislikeActuallyDislike : ℝ := 0.9 * DislikeDancing
def SayLikeActuallyDislike : ℝ := 0.1 * DislikeDancing

def TotalSayDislike : ℝ := SayDislikeActuallyLike + SayDislikeActuallyDislike

theorem fraction_say_dislike_actually_like : 
  SayDislikeActuallyLike / TotalSayDislike = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fraction_say_dislike_actually_like_l3942_394253


namespace NUMINAMATH_CALUDE_openai_robotics_competition_weight_l3942_394204

/-- The weight of the standard robot in the OpenAI robotics competition. -/
def standard_robot_weight : ℝ := 100

/-- The maximum weight allowed for a robot in the competition. -/
def max_weight : ℝ := 210

/-- The minimum weight of a robot in the competition. -/
def min_weight : ℝ := standard_robot_weight + 5

theorem openai_robotics_competition_weight :
  standard_robot_weight = 100 ∧
  max_weight = 210 ∧
  min_weight = standard_robot_weight + 5 ∧
  max_weight ≤ 2 * min_weight :=
by sorry

end NUMINAMATH_CALUDE_openai_robotics_competition_weight_l3942_394204


namespace NUMINAMATH_CALUDE_hot_dog_packs_l3942_394285

theorem hot_dog_packs (n : ℕ) : 
  (∃ m : ℕ, m < n ∧ 12 * m ≡ 6 [MOD 8]) → 
  12 * n ≡ 6 [MOD 8] → 
  (∀ k : ℕ, k < n → k ≠ n → 12 * k ≡ 6 [MOD 8] → 
    (∃ l : ℕ, l < k ∧ 12 * l ≡ 6 [MOD 8])) → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_hot_dog_packs_l3942_394285


namespace NUMINAMATH_CALUDE_min_q_for_three_solutions_l3942_394221

theorem min_q_for_three_solutions (p q : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℝ, |x^2 + p*x + q| = 3 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃))) →
  q ≥ -3 ∧ ∃ p₀ : ℝ, ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    (∀ x : ℝ, |x^2 + p₀*x + (-3)| = 3 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by
  sorry

end NUMINAMATH_CALUDE_min_q_for_three_solutions_l3942_394221


namespace NUMINAMATH_CALUDE_book_pages_theorem_l3942_394277

/-- Calculate the number of digits used to number pages in a book -/
def digits_used (num_pages : ℕ) : ℕ :=
  let single_digit := min num_pages 9
  let double_digit := min (num_pages - 9) 90
  let triple_digit := max (num_pages - 99) 0
  single_digit + 2 * double_digit + 3 * triple_digit

theorem book_pages_theorem :
  ∃ (num_pages : ℕ), digits_used num_pages = 636 ∧ num_pages = 248 :=
by sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l3942_394277


namespace NUMINAMATH_CALUDE_recycle_128_cans_l3942_394211

/-- The number of new cans that can be created through recycling, given an initial number of cans -/
def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 2 then 0
  else (initial_cans / 2) + recycle_cans (initial_cans / 2)

/-- Theorem stating that recycling 128 cans produces 127 new cans -/
theorem recycle_128_cans :
  recycle_cans 128 = 127 := by
  sorry

end NUMINAMATH_CALUDE_recycle_128_cans_l3942_394211


namespace NUMINAMATH_CALUDE_smallest_distance_between_circles_l3942_394246

theorem smallest_distance_between_circles (z w : ℂ) 
  (hz : Complex.abs (z - (2 - 4*I)) = 2)
  (hw : Complex.abs (w - (5 - 6*I)) = 4) :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 13 - 6 ∧
    ∀ (z' w' : ℂ), Complex.abs (z' - (2 - 4*I)) = 2 →
      Complex.abs (w' - (5 - 6*I)) = 4 →
        Complex.abs (z' - w') ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_smallest_distance_between_circles_l3942_394246


namespace NUMINAMATH_CALUDE_rolling_semicircle_path_length_l3942_394267

/-- The length of the path traveled by the center of a rolling semicircular arc -/
theorem rolling_semicircle_path_length (r : ℝ) (h : r > 0) :
  let path_length := 3 * Real.pi * r
  path_length = (Real.pi * (2 * r)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_rolling_semicircle_path_length_l3942_394267


namespace NUMINAMATH_CALUDE_separation_leads_to_growth_and_blessing_l3942_394248

/-- Represents the separation experience between a child and their mother --/
structure SeparationExperience where
  duration : ℕ
  communication_frequency : ℕ
  visits : ℕ
  child_attitude : Bool
  mother_attitude : Bool

/-- Represents the outcome of the separation experience --/
inductive Outcome
  | PersonalGrowth
  | Blessing
  | Negative

/-- Function to determine the outcome of a separation experience --/
def determine_outcome (exp : SeparationExperience) : Outcome := sorry

/-- Theorem stating that a positive separation experience leads to personal growth and can be a blessing --/
theorem separation_leads_to_growth_and_blessing 
  (exp : SeparationExperience) 
  (h1 : exp.duration ≥ 3) 
  (h2 : exp.communication_frequency ≥ 300) 
  (h3 : exp.visits ≥ 1) 
  (h4 : exp.child_attitude = true) 
  (h5 : exp.mother_attitude = true) : 
  determine_outcome exp = Outcome.PersonalGrowth ∧ 
  determine_outcome exp = Outcome.Blessing := 
sorry

end NUMINAMATH_CALUDE_separation_leads_to_growth_and_blessing_l3942_394248


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l3942_394256

theorem sum_of_roots_eq_fourteen : ∀ x₁ x₂ : ℝ, (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 → x₁ + x₂ = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l3942_394256


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_f_one_over_e_gt_f_one_half_l3942_394290

noncomputable def f (x : ℝ) : ℝ := -x / Real.exp x + Real.log 2

theorem f_decreasing_on_interval :
  ∀ x : ℝ, x < 1 → ∀ y : ℝ, x < y → f y < f x :=
sorry

theorem f_one_over_e_gt_f_one_half : f (1 / Real.exp 1) > f (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_f_one_over_e_gt_f_one_half_l3942_394290


namespace NUMINAMATH_CALUDE_diamond_three_four_l3942_394297

/-- Definition of the diamond operation -/
def diamond (x y : ℝ) : ℝ := 4 * x + 6 * y

/-- Theorem stating that 3 ◊ 4 = 36 -/
theorem diamond_three_four : diamond 3 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_diamond_three_four_l3942_394297


namespace NUMINAMATH_CALUDE_unique_solution_positive_root_l3942_394235

theorem unique_solution_positive_root (x : ℝ) :
  x ≥ 0 ∧ 2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_positive_root_l3942_394235


namespace NUMINAMATH_CALUDE_function_range_theorem_l3942_394270

/-- A function f : ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- A function f : ℝ → ℝ is decreasing on [0, +∞) if f(x) ≥ f(y) for all 0 ≤ x ≤ y -/
def IsDecreasingOnNonnegatives (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≥ f y

theorem function_range_theorem (f : ℝ → ℝ) 
  (h_even : IsEven f) (h_decreasing : IsDecreasingOnNonnegatives f) :
  {x : ℝ | f (Real.log x) > f 1} = Set.Ioo (Real.exp (-1)) (Real.exp 1) := by
  sorry

end NUMINAMATH_CALUDE_function_range_theorem_l3942_394270


namespace NUMINAMATH_CALUDE_xyz_inequalities_l3942_394225

theorem xyz_inequalities (x y z : ℝ) 
  (h1 : x < y) (h2 : y < z) 
  (h3 : x + y + z = 6) 
  (h4 : x*y + y*z + z*x = 9) : 
  0 < x ∧ x < 1 ∧ 1 < y ∧ y < 3 ∧ 3 < z ∧ z < 4 := by
  sorry

end NUMINAMATH_CALUDE_xyz_inequalities_l3942_394225


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3942_394244

theorem partial_fraction_decomposition :
  ∃ (P Q R : ℚ), 
    P = -3/5 ∧ Q = -1 ∧ R = 13/5 ∧
    ∀ (x : ℚ), x ≠ 1 → x ≠ 4 → x ≠ 6 →
      (x^2 - 10) / ((x - 1) * (x - 4) * (x - 6)) = 
      P / (x - 1) + Q / (x - 4) + R / (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3942_394244


namespace NUMINAMATH_CALUDE_households_with_bike_only_l3942_394299

theorem households_with_bike_only 
  (total : ℕ) 
  (without_car_or_bike : ℕ) 
  (with_both : ℕ) 
  (with_car : ℕ) 
  (h1 : total = 90)
  (h2 : without_car_or_bike = 11)
  (h3 : with_both = 18)
  (h4 : with_car = 44) :
  total - without_car_or_bike - (with_car - with_both) - with_both = 35 := by
  sorry

end NUMINAMATH_CALUDE_households_with_bike_only_l3942_394299


namespace NUMINAMATH_CALUDE_car_journey_time_l3942_394295

theorem car_journey_time (distance : ℝ) (new_speed : ℝ) (time_ratio : ℝ) 
  (h1 : distance = 450)
  (h2 : new_speed = 50)
  (h3 : time_ratio = 3/2)
  (h4 : distance = new_speed * (time_ratio * initial_time)) :
  initial_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_car_journey_time_l3942_394295


namespace NUMINAMATH_CALUDE_unique_factorization_1870_l3942_394215

/-- A function that returns true if a number is composed of only prime factors -/
def isPrimeComposite (n : Nat) : Bool :=
  sorry

/-- A function that returns true if a number is composed of a prime factor multiplied by a one-digit non-prime number -/
def isPrimeTimesNonPrime (n : Nat) : Bool :=
  sorry

/-- A function that counts the number of valid factorizations of n according to the given conditions -/
def countValidFactorizations (n : Nat) : Nat :=
  sorry

theorem unique_factorization_1870 :
  countValidFactorizations 1870 = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_factorization_1870_l3942_394215


namespace NUMINAMATH_CALUDE_ellipse_properties_l3942_394222

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 / 5 = 1

/-- Definition of line l -/
def line_l (x y : ℝ) : Prop :=
  y = Real.sqrt 3 / 3 * (x + 2) ∨ y = -Real.sqrt 3 / 3 * (x + 2)

/-- Point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C x y

/-- Theorem statement -/
theorem ellipse_properties :
  let a := 3
  let b := Real.sqrt 5
  let e := 2/3
  ∃ (F₁ F₂ : ℝ × ℝ) (A : ℝ × ℝ),
    -- C passes through (0, √5)
    ellipse_C 0 (Real.sqrt 5) ∧
    -- Eccentricity is 2/3
    Real.sqrt (F₁.1^2 + F₁.2^2) / a = e ∧
    -- A is on x = 4
    A.1 = 4 ∧
    -- When perpendicular bisector of F₁A passes through F₂, l has the given equation
    (∀ x y, line_l x y ↔ (y - F₁.2) / (x - F₁.1) = (A.2 - F₁.2) / (A.1 - F₁.1)) ∧
    -- Minimum length of AB
    ∃ (min_length : ℝ),
      min_length = Real.sqrt 21 ∧
      ∀ (B : PointOnEllipse),
        A.1 * B.x + A.2 * B.y = 0 →  -- OA ⊥ OB
        (A.1 - B.x)^2 + (A.2 - B.y)^2 ≥ min_length^2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3942_394222


namespace NUMINAMATH_CALUDE_book_pair_count_l3942_394219

theorem book_pair_count :
  let num_genres : ℕ := 4
  let books_per_genre : ℕ := 4
  let choose_genres : ℕ := 2
  num_genres.choose choose_genres * books_per_genre^choose_genres = 96 :=
by sorry

end NUMINAMATH_CALUDE_book_pair_count_l3942_394219


namespace NUMINAMATH_CALUDE_complex_sum_equals_negative_one_l3942_394232

theorem complex_sum_equals_negative_one (z : ℂ) (h : z = Complex.exp (2 * Real.pi * Complex.I / 9)) :
  z^2 / (1 + z^3) + z^4 / (1 + z^6) + z^6 / (1 + z^9) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_equals_negative_one_l3942_394232


namespace NUMINAMATH_CALUDE_right_triangle_division_area_ratio_l3942_394278

/-- Given a right triangle divided by a point on its hypotenuse and lines parallel to its legs,
    forming a square and two smaller right triangles, this theorem proves the relationship
    between the areas of the smaller triangles and the square. -/
theorem right_triangle_division_area_ratio
  (square_side : ℝ)
  (m : ℝ)
  (h_square_side : square_side = 2)
  (h_small_triangle_area : ∃ (small_triangle_area : ℝ), small_triangle_area = m * square_side^2)
  : ∃ (other_triangle_area : ℝ), other_triangle_area / square_side^2 = 1 / (4 * m) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_division_area_ratio_l3942_394278


namespace NUMINAMATH_CALUDE_intersection_distance_l3942_394208

theorem intersection_distance (a b : ℤ) (k : ℝ) : 
  k = a + Real.sqrt b →
  (k + 4) / k = Real.sqrt 5 →
  a + b = 6 := by sorry

end NUMINAMATH_CALUDE_intersection_distance_l3942_394208


namespace NUMINAMATH_CALUDE_only_caseD_has_two_solutions_l3942_394272

-- Define a structure for triangle cases
structure TriangleCase where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given cases
def caseA : TriangleCase := { a := 0, b := 10, c := 0, A := 45, B := 70, C := 0 }
def caseB : TriangleCase := { a := 60, b := 0, c := 48, A := 0, B := 100, C := 0 }
def caseC : TriangleCase := { a := 14, b := 16, c := 0, A := 45, B := 0, C := 0 }
def caseD : TriangleCase := { a := 7, b := 5, c := 0, A := 80, B := 0, C := 0 }

-- Function to determine if a case has two solutions
def hasTwoSolutions (t : TriangleCase) : Prop :=
  ∃ (B1 B2 : ℝ), B1 ≠ B2 ∧ 
    0 < B1 ∧ B1 < 180 ∧
    0 < B2 ∧ B2 < 180 ∧
    t.a / Real.sin t.A = t.b / Real.sin B1 ∧
    t.a / Real.sin t.A = t.b / Real.sin B2

-- Theorem stating that only case D has two solutions
theorem only_caseD_has_two_solutions :
  ¬(hasTwoSolutions caseA) ∧
  ¬(hasTwoSolutions caseB) ∧
  ¬(hasTwoSolutions caseC) ∧
  hasTwoSolutions caseD :=
sorry

end NUMINAMATH_CALUDE_only_caseD_has_two_solutions_l3942_394272


namespace NUMINAMATH_CALUDE_camp_kids_count_l3942_394287

theorem camp_kids_count : ℕ :=
  let total_kids : ℕ := 2000
  let soccer_kids : ℕ := total_kids / 2
  let morning_soccer_kids : ℕ := soccer_kids / 4
  let afternoon_soccer_kids : ℕ := 750
  have h1 : soccer_kids = total_kids / 2 := by sorry
  have h2 : morning_soccer_kids = soccer_kids / 4 := by sorry
  have h3 : afternoon_soccer_kids = 750 := by sorry
  have h4 : morning_soccer_kids + afternoon_soccer_kids = soccer_kids := by sorry
  total_kids

/- Proof
sorry
-/

end NUMINAMATH_CALUDE_camp_kids_count_l3942_394287


namespace NUMINAMATH_CALUDE_limit_exp_sin_ratio_l3942_394217

theorem limit_exp_sin_ratio : 
  ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → 
    |((Real.exp (2*x) - Real.exp x) / (Real.sin (2*x) - Real.sin x)) - 1| < ε := by
sorry

end NUMINAMATH_CALUDE_limit_exp_sin_ratio_l3942_394217


namespace NUMINAMATH_CALUDE_sphere_volume_ratio_l3942_394269

theorem sphere_volume_ratio (r₁ r₂ : ℝ) (h : 4 * Real.pi * r₁^2 = (2/3) * (4 * Real.pi * r₂^2)) :
  (4/3) * Real.pi * r₁^3 = (2 * Real.sqrt 6 / 9) * ((4/3) * Real.pi * r₂^3) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_ratio_l3942_394269


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3942_394226

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity (a b c : ℝ) (m n : ℝ) :
  a > 0 →
  b > 0 →
  c = (a^2 + b^2).sqrt →
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 ↔ ((x, y) : ℝ × ℝ) ∈ {p | p.1^2 / a^2 - p.2^2 / b^2 = 1}) →
  (c, 0) ∈ {p | p.1^2 / a^2 - p.2^2 / b^2 = 1} →
  ((m + n) * c, (m - n) * b * c / a) ∈ {p | p.1^2 / a^2 - p.2^2 / b^2 = 1} →
  m * n = 2 / 9 →
  (a^2 + b^2) / a^2 = (3 * Real.sqrt 2 / 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3942_394226


namespace NUMINAMATH_CALUDE_max_ab_internally_tangent_circles_l3942_394266

/-- Two circles C₁ and C₂ are internally tangent if the distance between their centers
    is equal to the difference of their radii. -/
def internally_tangent (a b : ℝ) : Prop :=
  (a + b)^2 = 1

/-- The equation of circle C₁ -/
def C₁ (x y a : ℝ) : Prop :=
  (x - a)^2 + (y + 2)^2 = 4

/-- The equation of circle C₂ -/
def C₂ (x y b : ℝ) : Prop :=
  (x + b)^2 + (y + 2)^2 = 1

/-- The theorem stating that the maximum value of ab is 1/4 -/
theorem max_ab_internally_tangent_circles (a b : ℝ) :
  internally_tangent a b → a * b ≤ 1/4 ∧ ∃ a b, internally_tangent a b ∧ a * b = 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_ab_internally_tangent_circles_l3942_394266


namespace NUMINAMATH_CALUDE_T_divisibility_l3942_394227

def T : Set ℕ := {s | ∃ n : ℕ, s = (n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2}

theorem T_divisibility :
  (∀ s ∈ T, ¬(9 ∣ s)) ∧ (∃ s ∈ T, 4 ∣ s) := by sorry

end NUMINAMATH_CALUDE_T_divisibility_l3942_394227


namespace NUMINAMATH_CALUDE_blue_line_length_calculation_l3942_394210

-- Define the length of the white line
def white_line_length : ℝ := 7.666666666666667

-- Define the difference between the white and blue lines
def length_difference : ℝ := 4.333333333333333

-- Define the length of the blue line
def blue_line_length : ℝ := white_line_length - length_difference

-- Theorem statement
theorem blue_line_length_calculation : 
  blue_line_length = 3.333333333333334 := by sorry

end NUMINAMATH_CALUDE_blue_line_length_calculation_l3942_394210


namespace NUMINAMATH_CALUDE_tom_age_proof_l3942_394283

theorem tom_age_proof (tom_age tim_age : ℕ) : 
  (tom_age + tim_age = 21) →
  (tom_age + 3 = 2 * (tim_age + 3)) →
  tom_age = 15 := by
sorry

end NUMINAMATH_CALUDE_tom_age_proof_l3942_394283


namespace NUMINAMATH_CALUDE_cookies_left_l3942_394258

def cookies_problem (days : ℕ) (trays_per_day : ℕ) (cookies_per_tray : ℕ) 
  (frank_eats_per_day : ℕ) (ted_eats : ℕ) : ℕ :=
  days * trays_per_day * cookies_per_tray - days * frank_eats_per_day - ted_eats

theorem cookies_left : 
  cookies_problem 6 2 12 1 4 = 134 := by
  sorry

end NUMINAMATH_CALUDE_cookies_left_l3942_394258


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3942_394274

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -5 ∧ x₂ = 1 ∧ x₁^2 + 4*x₁ - 5 = 0 ∧ x₂^2 + 4*x₂ - 5 = 0) ∧
  (∃ y₁ y₂ : ℝ, y₁ = 1/3 ∧ y₂ = -1 ∧ 3*y₁^2 + 2*y₁ = 1 ∧ 3*y₂^2 + 2*y₂ = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3942_394274


namespace NUMINAMATH_CALUDE_intersection_point_sum_squares_l3942_394254

-- Define the lines
def line1 (x y : ℝ) : Prop := 323 * x + 457 * y = 1103
def line2 (x y : ℝ) : Prop := 177 * x + 543 * y = 897

-- Define the intersection point
def intersection_point (a b : ℝ) : Prop := line1 a b ∧ line2 a b

-- Theorem statement
theorem intersection_point_sum_squares :
  ∀ a b : ℝ, intersection_point a b → a^2 + 2004 * b^2 = 2008 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_squares_l3942_394254


namespace NUMINAMATH_CALUDE_bobs_age_problem_l3942_394242

theorem bobs_age_problem :
  ∃ (n : ℕ), 
    (∃ (k : ℕ), n - 3 = k^2) ∧ 
    (∃ (j : ℕ), n + 4 = j^3) ∧ 
    n = 725 := by
  sorry

end NUMINAMATH_CALUDE_bobs_age_problem_l3942_394242


namespace NUMINAMATH_CALUDE_max_min_values_on_interval_l3942_394206

def f (x : ℝ) := 2 * x^3 - 3 * x^2 - 12 * x + 5

theorem max_min_values_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc 0 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc 0 3, f x = max) ∧
    (∀ x ∈ Set.Icc 0 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc 0 3, f x = min) ∧
    max = 5 ∧ min = -15 := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_on_interval_l3942_394206


namespace NUMINAMATH_CALUDE_square_difference_equality_l3942_394288

theorem square_difference_equality (a b : ℝ) 
  (sum_eq : a + b = 10) 
  (diff_eq : a - b = 4) : 
  a^2 - b^2 = 40 := by
sorry

end NUMINAMATH_CALUDE_square_difference_equality_l3942_394288


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3942_394201

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + k = 0 → y = x) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3942_394201


namespace NUMINAMATH_CALUDE_min_questionnaires_correct_l3942_394216

/-- The minimum number of questionnaires needed to achieve the desired responses -/
def min_questionnaires : ℕ := 513

/-- The number of desired responses -/
def desired_responses : ℕ := 750

/-- The initial response rate -/
def initial_rate : ℚ := 60 / 100

/-- The decline rate for follow-ups -/
def decline_rate : ℚ := 20 / 100

/-- Calculate the total responses given the number of questionnaires sent -/
def total_responses (n : ℕ) : ℚ :=
  n * initial_rate * (1 + (1 - decline_rate) + (1 - decline_rate)^2)

/-- Theorem stating that min_questionnaires is the minimum number needed -/
theorem min_questionnaires_correct :
  (total_responses min_questionnaires ≥ desired_responses) ∧
  (∀ m : ℕ, m < min_questionnaires → total_responses m < desired_responses) :=
by sorry


end NUMINAMATH_CALUDE_min_questionnaires_correct_l3942_394216


namespace NUMINAMATH_CALUDE_shen_win_probability_correct_l3942_394273

/-- Represents a player in the game -/
inductive Player
| Shen
| Ling
| Ru

/-- The number of slips each player puts in the bucket initially -/
def initial_slips : Nat := 4

/-- The total number of slips in the bucket -/
def total_slips : Nat := 13

/-- The number of slips Shen needs to win -/
def shen_win_condition : Nat := 4

/-- Calculates the probability of Shen winning the game -/
def shen_win_probability : Rat :=
  67 / 117

/-- Theorem stating that the calculated probability is correct -/
theorem shen_win_probability_correct :
  shen_win_probability = 67 / 117 := by sorry

end NUMINAMATH_CALUDE_shen_win_probability_correct_l3942_394273


namespace NUMINAMATH_CALUDE_production_target_is_1800_l3942_394203

/-- Calculates the yearly production target for a car manufacturing company. -/
def yearly_production_target (current_monthly_production : ℕ) (monthly_increase : ℕ) : ℕ :=
  (current_monthly_production + monthly_increase) * 12

/-- Theorem: The yearly production target is 1800 cars. -/
theorem production_target_is_1800 :
  yearly_production_target 100 50 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_production_target_is_1800_l3942_394203


namespace NUMINAMATH_CALUDE_max_product_with_sum_constraint_l3942_394200

theorem max_product_with_sum_constraint :
  ∃ (x : ℤ), 
    (∀ y : ℤ, x * (340 - x) ≥ y * (340 - y)) ∧ 
    (x * (340 - x) > 2000) ∧
    (x * (340 - x) = 28900) := by
  sorry

end NUMINAMATH_CALUDE_max_product_with_sum_constraint_l3942_394200


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3942_394228

theorem quadratic_equation_properties (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 2*m*x₁ + m^2 - 1 = 0 ∧ x₂^2 + 2*m*x₂ + m^2 - 1 = 0) ∧
  ((-2)^2 + 2*m*(-2) + m^2 - 1 = 0 → 2023 - m^2 + 4*m = 2026) :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3942_394228


namespace NUMINAMATH_CALUDE_white_square_area_l3942_394259

/-- Given a cube with edge length 10 feet and 300 square feet of paint used for borders,
    the area of the white square on each face is 50 square feet. -/
theorem white_square_area (cube_edge : ℝ) (paint_area : ℝ) : 
  cube_edge = 10 →
  paint_area = 300 →
  (6 * cube_edge^2 - paint_area) / 6 = 50 := by
  sorry

end NUMINAMATH_CALUDE_white_square_area_l3942_394259


namespace NUMINAMATH_CALUDE_money_sharing_l3942_394218

theorem money_sharing (amanda ben carlos total : ℕ) : 
  amanda + ben + carlos = total →
  amanda = 3 * (ben / 5) →
  carlos = 9 * (ben / 5) →
  ben = 50 →
  total = 170 := by
sorry

end NUMINAMATH_CALUDE_money_sharing_l3942_394218


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l3942_394223

theorem rectangle_area_increase (L W : ℝ) (h : L > 0 ∧ W > 0) : 
  let original_area := L * W
  let new_length := 1.2 * L
  let new_width := 1.2 * W
  let new_area := new_length * new_width
  (new_area - original_area) / original_area * 100 = 44 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l3942_394223


namespace NUMINAMATH_CALUDE_new_average_commission_is_550_l3942_394257

/-- Represents a salesperson's commission data -/
structure SalespersonData where
  totalSales : ℕ
  lastCommission : ℝ
  averageIncrease : ℝ

/-- Calculates the new average commission for a salesperson -/
def newAverageCommission (data : SalespersonData) : ℝ :=
  -- The actual calculation is not implemented here
  sorry

/-- Theorem stating that under given conditions, the new average commission is $550 -/
theorem new_average_commission_is_550 (data : SalespersonData) 
  (h1 : data.totalSales = 6)
  (h2 : data.lastCommission = 1300)
  (h3 : data.averageIncrease = 150) :
  newAverageCommission data = 550 := by
  sorry

end NUMINAMATH_CALUDE_new_average_commission_is_550_l3942_394257


namespace NUMINAMATH_CALUDE_coopers_savings_l3942_394261

/-- Calculates the total savings given a daily savings amount and number of days -/
def totalSavings (dailySavings : ℕ) (days : ℕ) : ℕ :=
  dailySavings * days

/-- Theorem: Cooper's savings after one year -/
theorem coopers_savings :
  totalSavings 34 365 = 12410 := by
  sorry

end NUMINAMATH_CALUDE_coopers_savings_l3942_394261


namespace NUMINAMATH_CALUDE_two_year_compound_interest_l3942_394238

/-- Calculates the final amount after two years of compound interest with variable rates -/
def final_amount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  initial * (1 + rate1) * (1 + rate2)

/-- Theorem stating that given the specific initial amount and interest rates, 
    the final amount after two years is 82432 -/
theorem two_year_compound_interest :
  final_amount 64000 0.12 0.15 = 82432 := by
  sorry

#eval final_amount 64000 0.12 0.15

end NUMINAMATH_CALUDE_two_year_compound_interest_l3942_394238


namespace NUMINAMATH_CALUDE_stating_last_seat_probability_is_reciprocal_seven_seats_probability_l3942_394260

/-- 
Represents the probability that the last passenger sits in their own seat 
in a seating arrangement problem with n seats and n passengers.
-/
def last_seat_probability (n : ℕ) : ℚ :=
  if n = 0 then 0
  else 1 / n

/-- 
Theorem stating that the probability of the last passenger sitting in their own seat
is 1/n for any number of seats n > 0.
-/
theorem last_seat_probability_is_reciprocal (n : ℕ) (h : n > 0) : 
  last_seat_probability n = 1 / n := by
  sorry

/-- 
Corollary for the specific case of 7 seats, as in the original problem.
-/
theorem seven_seats_probability : 
  last_seat_probability 7 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_stating_last_seat_probability_is_reciprocal_seven_seats_probability_l3942_394260


namespace NUMINAMATH_CALUDE_profit_margin_in_terms_of_retail_price_l3942_394213

/-- Given a profit margin P, production cost C, retail price P_R, and constants k and c,
    prove that P can be expressed in terms of P_R. -/
theorem profit_margin_in_terms_of_retail_price
  (P C P_R k c : ℝ) (hP : P = k * C) (hP_R : P_R = c * (P + C)) :
  P = (k / (c * (k + 1))) * P_R :=
sorry

end NUMINAMATH_CALUDE_profit_margin_in_terms_of_retail_price_l3942_394213


namespace NUMINAMATH_CALUDE_rohan_salary_l3942_394224

def food_expense : ℚ := 30 / 100
def rent_expense : ℚ := 20 / 100
def entertainment_expense : ℚ := 10 / 100
def conveyance_expense : ℚ := 5 / 100
def education_expense : ℚ := 10 / 100
def utilities_expense : ℚ := 10 / 100
def miscellaneous_expense : ℚ := 5 / 100
def savings_amount : ℕ := 2500

def total_expenses : ℚ :=
  food_expense + rent_expense + entertainment_expense + conveyance_expense +
  education_expense + utilities_expense + miscellaneous_expense

def savings_percentage : ℚ := 1 - total_expenses

theorem rohan_salary :
  ∃ (salary : ℕ), (↑savings_amount : ℚ) / (↑salary : ℚ) = savings_percentage ∧ salary = 25000 :=
by sorry

end NUMINAMATH_CALUDE_rohan_salary_l3942_394224


namespace NUMINAMATH_CALUDE_cubic_real_root_l3942_394284

theorem cubic_real_root (c d : ℝ) (h : c ≠ 0) :
  (∃ z : ℂ, c * z^3 + 5 * z^2 + d * z - 104 = 0 ∧ z = -3 - 4*I) →
  (∃ x : ℝ, c * x^3 + 5 * x^2 + d * x - 104 = 0 ∧ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_real_root_l3942_394284


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3942_394237

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 + y^2) = f (x^2 - y^2) + f (2*x*y)

/-- The main theorem -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x, f x ≥ 0) → SatisfiesEquation f →
  ∃ a : ℝ, a ≥ 0 ∧ ∀ x, f x = a * x^2 :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3942_394237


namespace NUMINAMATH_CALUDE_subtraction_result_l3942_394252

theorem subtraction_result : 3.57 - 2.15 = 1.42 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l3942_394252
