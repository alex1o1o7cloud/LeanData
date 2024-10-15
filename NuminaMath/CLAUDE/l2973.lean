import Mathlib

namespace NUMINAMATH_CALUDE_base9_734_equals_base3_211110_l2973_297375

/-- Converts a digit from base 9 to base 3 --/
def base9_to_base3 (d : ℕ) : ℕ := sorry

/-- Converts a number from base 9 to base 3 --/
def convert_base9_to_base3 (n : ℕ) : ℕ := sorry

/-- The main theorem stating that 734 in base 9 is equal to 211110 in base 3 --/
theorem base9_734_equals_base3_211110 :
  convert_base9_to_base3 734 = 211110 := by sorry

end NUMINAMATH_CALUDE_base9_734_equals_base3_211110_l2973_297375


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2973_297322

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (2 - x) < 0} = {x : ℝ | x > 2 ∨ x < -1} := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2973_297322


namespace NUMINAMATH_CALUDE_problem_1_l2973_297339

theorem problem_1 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (1) * ((-2*a)^3 * (-a*b^2)^3 - 4*a*b^2 * (2*a^5*b^4 + 1/2*a*b^3 - 5)) / (-2*a*b) = a*b^4 - 10*b :=
sorry

end NUMINAMATH_CALUDE_problem_1_l2973_297339


namespace NUMINAMATH_CALUDE_smaller_number_between_5_and_8_l2973_297333

theorem smaller_number_between_5_and_8 :
  (5 ≤ 8) ∧ (∀ x : ℝ, 5 ≤ x ∧ x ≤ 8 → 5 ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_smaller_number_between_5_and_8_l2973_297333


namespace NUMINAMATH_CALUDE_problem_solution_l2973_297340

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x - a

theorem problem_solution :
  -- Part 1: Find the value of a
  (∃ a : ℝ, f a 0 = -5 ∧ a = -5) ∧
  -- Part 2: Find the equation of the tangent line
  (∃ x y : ℝ,
    -- Point M(x, y) is on the curve f
    y = f (-5) x ∧
    -- Tangent line at M is parallel to 3x + 2y + 2 = 0
    f' (-5) x = -3/2 ∧
    -- Equation of the tangent line
    (24 : ℝ) * x + 16 * y - 37 = 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2973_297340


namespace NUMINAMATH_CALUDE_max_value_of_f_l2973_297388

def f (x : ℝ) : ℝ := x^2 + 2*x + 1

theorem max_value_of_f :
  ∃ (m : ℝ), m = 9 ∧ ∀ (x : ℝ), x ∈ Set.Icc (-2) 2 → f x ≤ m :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2973_297388


namespace NUMINAMATH_CALUDE_painting_cost_in_cny_l2973_297304

-- Define exchange rates
def usd_to_nad : ℝ := 7
def usd_to_cny : ℝ := 6

-- Define the cost of the painting in Namibian dollars
def painting_cost_nad : ℝ := 105

-- Theorem to prove
theorem painting_cost_in_cny :
  (painting_cost_nad / usd_to_nad) * usd_to_cny = 90 := by
  sorry

end NUMINAMATH_CALUDE_painting_cost_in_cny_l2973_297304


namespace NUMINAMATH_CALUDE_probability_of_seven_in_three_eighths_l2973_297394

theorem probability_of_seven_in_three_eighths : 
  let decimal_rep := [3, 7, 5]
  let count_sevens := (decimal_rep.filter (· = 7)).length
  let total_digits := decimal_rep.length
  (count_sevens : ℚ) / total_digits = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_probability_of_seven_in_three_eighths_l2973_297394


namespace NUMINAMATH_CALUDE_influenza_transmission_rate_l2973_297321

theorem influenza_transmission_rate (initial_infected : ℕ) (total_infected : ℕ) : 
  initial_infected = 4 →
  total_infected = 256 →
  ∃ (x : ℕ), 
    x > 0 ∧
    initial_infected + initial_infected * x + (initial_infected + initial_infected * x) * x = total_infected →
    x = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_influenza_transmission_rate_l2973_297321


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l2973_297314

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus_F : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

theorem ellipse_line_intersection :
  ∃ k : ℝ, k = 2 ∨ k = -2 ∧
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧
    ellipse_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧
    line_l k x₂ y₂ ∧
    perpendicular x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l2973_297314


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2973_297345

/-- The discriminant of a quadratic equation ax^2 + bx + c -/
def discriminant (a b c : ℚ) : ℚ := b^2 - 4*a*c

/-- The coefficients of the quadratic equation 5x^2 + (5 + 1/2)x - 2 -/
def a : ℚ := 5
def b : ℚ := 5 + 1/2
def c : ℚ := -2

theorem quadratic_discriminant : discriminant a b c = 281/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2973_297345


namespace NUMINAMATH_CALUDE_cos_equality_l2973_297307

theorem cos_equality (n : ℤ) : 0 ≤ n ∧ n ≤ 180 → n = 138 → Real.cos (n * π / 180) = Real.cos (942 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_l2973_297307


namespace NUMINAMATH_CALUDE_julia_bought_399_balls_l2973_297358

/-- The number of balls Julia bought -/
def total_balls (red_packs yellow_packs green_packs balls_per_pack : ℕ) : ℕ :=
  (red_packs + yellow_packs + green_packs) * balls_per_pack

/-- Proof that Julia bought 399 balls -/
theorem julia_bought_399_balls :
  total_balls 3 10 8 19 = 399 := by
  sorry

end NUMINAMATH_CALUDE_julia_bought_399_balls_l2973_297358


namespace NUMINAMATH_CALUDE_f_minus_g_zero_iff_k_eq_9_4_l2973_297378

/-- The function f(x) = 5x^2 - 3x + 2 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 2

/-- The function g(x) = x^3 - 2x^2 + kx - 10 -/
def g (k x : ℝ) : ℝ := x^3 - 2 * x^2 + k * x - 10

/-- Theorem stating that f(5) - g(5) = 0 if and only if k = 9.4 -/
theorem f_minus_g_zero_iff_k_eq_9_4 : 
  ∀ k : ℝ, f 5 - g k 5 = 0 ↔ k = 9.4 := by sorry

end NUMINAMATH_CALUDE_f_minus_g_zero_iff_k_eq_9_4_l2973_297378


namespace NUMINAMATH_CALUDE_two_propositions_true_l2973_297336

theorem two_propositions_true : 
  (¬(∀ x : ℝ, x^2 > 0)) ∧ 
  (∃ x : ℝ, x^2 ≤ x) ∧ 
  (∀ M N : Set α, ∀ x : α, x ∈ M ∩ N → x ∈ M ∧ x ∈ N) := by
  sorry

end NUMINAMATH_CALUDE_two_propositions_true_l2973_297336


namespace NUMINAMATH_CALUDE_roger_lawn_mowing_earnings_l2973_297390

/-- Roger's lawn mowing earnings problem -/
theorem roger_lawn_mowing_earnings : 
  ∀ (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ),
    rate = 9 →
    total_lawns = 14 →
    forgotten_lawns = 8 →
    rate * (total_lawns - forgotten_lawns) = 54 := by
  sorry

end NUMINAMATH_CALUDE_roger_lawn_mowing_earnings_l2973_297390


namespace NUMINAMATH_CALUDE_cos_alpha_value_l2973_297300

theorem cos_alpha_value (α : Real) :
  (∃ P : Real × Real, P.1 = -3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  Real.cos α = -3/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l2973_297300


namespace NUMINAMATH_CALUDE_congruence_and_divisibility_solutions_l2973_297376

theorem congruence_and_divisibility_solutions : 
  {x : ℤ | x^3 ≡ -1 [ZMOD 7] ∧ (7 : ℤ) ∣ (x^2 - x + 1)} = {3, 5} := by
  sorry

end NUMINAMATH_CALUDE_congruence_and_divisibility_solutions_l2973_297376


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l2973_297391

theorem hot_dogs_remainder : 35876119 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l2973_297391


namespace NUMINAMATH_CALUDE_puzzle_solution_l2973_297337

theorem puzzle_solution :
  ∀ (S I A L T : ℕ),
  S ≠ 0 →
  S ≠ I ∧ S ≠ A ∧ S ≠ L ∧ S ≠ T ∧
  I ≠ A ∧ I ≠ L ∧ I ≠ T ∧
  A ≠ L ∧ A ≠ T ∧
  L ≠ T →
  10 * S + I < 100 →
  1000 * S + 100 * A + 10 * L + T < 10000 →
  (10 * S + I) * (10 * S + I) = 1000 * S + 100 * A + 10 * L + T →
  S = 9 ∧ I = 8 ∧ A = 6 ∧ L = 0 ∧ T = 4 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solution_l2973_297337


namespace NUMINAMATH_CALUDE_number_problem_l2973_297396

theorem number_problem (x : ℝ) : (4/5 * x) + 16 = 0.9 * 40 → x = 25 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2973_297396


namespace NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l2973_297319

theorem cos_theta_plus_pi_fourth (θ : Real) :
  (3 : Real) = 5 * Real.cos θ ∧ (-4 : Real) = 5 * Real.sin θ →
  Real.cos (θ + π/4) = 7 * Real.sqrt 2 / 10 := by sorry

end NUMINAMATH_CALUDE_cos_theta_plus_pi_fourth_l2973_297319


namespace NUMINAMATH_CALUDE_solve_pizza_problem_l2973_297313

def pizza_problem (total_slices : ℕ) (slices_left : ℕ) (slices_per_person : ℕ) : Prop :=
  let slices_eaten := total_slices - slices_left
  slices_eaten / slices_per_person = 6

theorem solve_pizza_problem :
  pizza_problem 16 4 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_pizza_problem_l2973_297313


namespace NUMINAMATH_CALUDE_line_equation_through_midpoint_l2973_297341

/-- A line passing through point P (1, 3) intersects the coordinate axes at points A and B. 
    P is the midpoint of AB. The equation of the line is 3x + y - 6 = 0. -/
theorem line_equation_through_midpoint (A B P : ℝ × ℝ) : 
  P = (1, 3) →
  (∃ a b : ℝ, A = (a, 0) ∧ B = (0, b)) →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∀ x y : ℝ, (3 * x + y - 6 = 0) ↔ (∃ t : ℝ, (x, y) = (1 - t, 3 + t * (B.2 - 3))) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_midpoint_l2973_297341


namespace NUMINAMATH_CALUDE_a_range_l2973_297327

def A (a : ℝ) : Set ℝ := {x | x^2 - 2*x + a > 0}

theorem a_range (a : ℝ) : (1 ∉ A a) ↔ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l2973_297327


namespace NUMINAMATH_CALUDE_martian_puzzle_l2973_297353

-- Define the Martian type
inductive Martian
| Red
| Blue

-- Define the state of the Martians
structure MartianState where
  total : Nat
  initialRed : Nat
  currentRed : Nat

-- Define the properties of the Martians' answers
def validAnswerSequence (state : MartianState) : Prop :=
  state.total = 2018 ∧
  ∀ i : Nat, i < state.total → 
    (i + 1 = state.initialRed + i - state.initialRed + 1)

-- Define the theorem
theorem martian_puzzle :
  ∀ state : MartianState,
    validAnswerSequence state →
    (state.initialRed = 0 ∨ state.initialRed = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_martian_puzzle_l2973_297353


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_exists_n_binomial_sum_l2973_297343

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem exists_n_binomial_sum : 
  ∃ n : ℕ, Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧ n = 6 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_exists_n_binomial_sum_l2973_297343


namespace NUMINAMATH_CALUDE_problem_1_l2973_297355

theorem problem_1 : (-1) * (-4) + 2^2 / (7 - 5) = 6 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2973_297355


namespace NUMINAMATH_CALUDE_chemistry_textbook_weight_l2973_297363

/-- The weight of the geometry textbook in pounds -/
def geometry_weight : ℝ := 0.62

/-- The additional weight of the chemistry textbook compared to the geometry textbook in pounds -/
def additional_weight : ℝ := 6.5

/-- The weight of the chemistry textbook in pounds -/
def chemistry_weight : ℝ := geometry_weight + additional_weight

theorem chemistry_textbook_weight : chemistry_weight = 7.12 := by
  sorry

end NUMINAMATH_CALUDE_chemistry_textbook_weight_l2973_297363


namespace NUMINAMATH_CALUDE_quadratic_root_property_l2973_297320

theorem quadratic_root_property (a : ℝ) : 
  (a^2 + 3*a - 5 = 0) → (-a^2 - 3*a = -5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l2973_297320


namespace NUMINAMATH_CALUDE_sticker_count_l2973_297301

/-- The number of stickers Ryan has -/
def ryan_stickers : ℕ := 30

/-- The number of stickers Steven has -/
def steven_stickers : ℕ := 3 * ryan_stickers

/-- The number of stickers Terry has -/
def terry_stickers : ℕ := steven_stickers + 20

/-- The total number of stickers Ryan, Steven, and Terry have altogether -/
def total_stickers : ℕ := ryan_stickers + steven_stickers + terry_stickers

theorem sticker_count : total_stickers = 230 := by
  sorry

end NUMINAMATH_CALUDE_sticker_count_l2973_297301


namespace NUMINAMATH_CALUDE_tangent_line_slope_logarithm_inequality_l2973_297366

-- Define the natural logarithm function
noncomputable def f (x : ℝ) : ℝ := Real.log x

-- Theorem for the tangent line
theorem tangent_line_slope (k : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f x₀ = k * x₀ ∧ (deriv f) x₀ = k) ↔ k = 1 / Real.exp 1 :=
sorry

-- Theorem for the inequality
theorem logarithm_inequality (a x : ℝ) (ha : a ≥ 1) (hx : x > 0) :
  f x ≤ a * x + (a - 1) / x - 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_slope_logarithm_inequality_l2973_297366


namespace NUMINAMATH_CALUDE_exists_M_with_properties_l2973_297330

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The theorem stating the existence of M with the required properties -/
theorem exists_M_with_properties : 
  ∃ M : ℕ, M^2 = 36^50 * 50^36 ∧ sum_of_digits M = 36 := by sorry

end NUMINAMATH_CALUDE_exists_M_with_properties_l2973_297330


namespace NUMINAMATH_CALUDE_fibFactLastTwoDigitsSum_l2973_297384

/-- The Fibonacci Factorial Series up to 144 -/
def fibFactSeries : List ℕ := [1, 1, 2, 3, 8, 13, 21, 34, 55, 89, 144]

/-- Function to calculate the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Function to get the last two digits of a number -/
def lastTwoDigits (n : ℕ) : ℕ :=
  n % 100

/-- Theorem stating that the sum of the last two digits of the Fibonacci Factorial Series is 30 -/
theorem fibFactLastTwoDigitsSum :
  (fibFactSeries.map (λ n => lastTwoDigits (factorial n))).sum = 30 := by
  sorry

/-- Lemma stating that factorials of numbers greater than 10 end with 00 -/
lemma factorialEndsWith00 (n : ℕ) (h : n > 10) :
  lastTwoDigits (factorial n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fibFactLastTwoDigitsSum_l2973_297384


namespace NUMINAMATH_CALUDE_arrangements_with_conditions_l2973_297308

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def arrangements_of_n (n : ℕ) : ℕ := factorial n

def arrangements_with_left_end (n : ℕ) : ℕ := factorial (n - 1)

def arrangements_adjacent (n : ℕ) : ℕ := 2 * factorial (n - 1)

def arrangements_left_end_and_adjacent (n : ℕ) : ℕ := factorial (n - 2)

theorem arrangements_with_conditions (n : ℕ) (h : n = 5) : 
  arrangements_of_n n - arrangements_with_left_end n - arrangements_adjacent n + arrangements_left_end_and_adjacent n = 54 :=
sorry

end NUMINAMATH_CALUDE_arrangements_with_conditions_l2973_297308


namespace NUMINAMATH_CALUDE_a_3_value_l2973_297326

def a (n : ℕ) : ℚ := (-1)^n * (n : ℚ) / (n + 1)

theorem a_3_value : a 3 = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_a_3_value_l2973_297326


namespace NUMINAMATH_CALUDE_angle_sum_inequality_l2973_297370

theorem angle_sum_inequality (θ₁ θ₂ θ₃ θ₄ : Real) 
  (h₁ : 0 < θ₁ ∧ θ₁ < π/2)
  (h₂ : 0 < θ₂ ∧ θ₂ < π/2)
  (h₃ : 0 < θ₃ ∧ θ₃ < π/2)
  (h₄ : 0 < θ₄ ∧ θ₄ < π/2)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) :
  (Real.sqrt 2 * Real.sin θ₁ - 1) / Real.cos θ₁ +
  (Real.sqrt 2 * Real.sin θ₂ - 1) / Real.cos θ₂ +
  (Real.sqrt 2 * Real.sin θ₃ - 1) / Real.cos θ₃ +
  (Real.sqrt 2 * Real.sin θ₄ - 1) / Real.cos θ₄ ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_inequality_l2973_297370


namespace NUMINAMATH_CALUDE_bird_wings_problem_l2973_297325

theorem bird_wings_problem :
  ∃! (x y z : ℕ), 2 * x + 4 * y + 3 * z = 70 ∧ x = 2 * y := by
  sorry

end NUMINAMATH_CALUDE_bird_wings_problem_l2973_297325


namespace NUMINAMATH_CALUDE_point_outside_circle_l2973_297379

theorem point_outside_circle (m : ℝ) : 
  let P : ℝ × ℝ := (m^2, 5)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 24}
  P ∉ circle := by
sorry

end NUMINAMATH_CALUDE_point_outside_circle_l2973_297379


namespace NUMINAMATH_CALUDE_factorial_6_l2973_297317

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_6 : factorial 6 = 720 := by sorry

end NUMINAMATH_CALUDE_factorial_6_l2973_297317


namespace NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l2973_297387

theorem vector_subtraction_and_scalar_multiplication :
  (⟨2, -5⟩ : ℝ × ℝ) - 4 • (⟨-1, 7⟩ : ℝ × ℝ) = (⟨6, -33⟩ : ℝ × ℝ) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_and_scalar_multiplication_l2973_297387


namespace NUMINAMATH_CALUDE_mixed_number_calculation_l2973_297306

theorem mixed_number_calculation : 
  36 * ((5 + 1/6) - (6 + 1/7)) / ((3 + 1/6) + (2 + 1/7)) = -(6 + 156/223) :=
by sorry

end NUMINAMATH_CALUDE_mixed_number_calculation_l2973_297306


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2973_297323

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) : 
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3/2 :=
sorry

theorem equality_condition (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) : 
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) = 3/2 ↔ 
  a = 1 ∧ b = 1 ∧ c = 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l2973_297323


namespace NUMINAMATH_CALUDE_parentheses_removal_l2973_297372

theorem parentheses_removal (a b c : ℝ) : -3*a - (2*b - c) = -3*a - 2*b + c := by
  sorry

end NUMINAMATH_CALUDE_parentheses_removal_l2973_297372


namespace NUMINAMATH_CALUDE_sand_truck_loads_l2973_297315

/-- Proves that the truck-loads of sand required is equal to 0.1666666666666666,
    given the total truck-loads of material needed and the truck-loads of dirt and cement. -/
theorem sand_truck_loads (total material_needed dirt cement sand : ℚ)
    (h1 : total = 0.6666666666666666)
    (h2 : dirt = 0.3333333333333333)
    (h3 : cement = 0.16666666666666666)
    (h4 : sand = total - (dirt + cement)) :
    sand = 0.1666666666666666 := by
  sorry

end NUMINAMATH_CALUDE_sand_truck_loads_l2973_297315


namespace NUMINAMATH_CALUDE_m_range_is_open_interval_l2973_297359

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative -/
def in_fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

/-- The range of m for which z = (m+3) + (m-1)i is in the fourth quadrant -/
def m_range : Set ℝ := {m : ℝ | in_fourth_quadrant ((m + 3) + (m - 1) * Complex.I)}

theorem m_range_is_open_interval : 
  m_range = Set.Ioo (-3) 1 := by sorry

end NUMINAMATH_CALUDE_m_range_is_open_interval_l2973_297359


namespace NUMINAMATH_CALUDE_solve_property_damage_l2973_297380

def property_damage_problem (medical_bills : ℝ) (carl_payment_percentage : ℝ) (carl_payment : ℝ) : Prop :=
  let total_cost := carl_payment / carl_payment_percentage
  let property_damage := total_cost - medical_bills
  property_damage = 40000

theorem solve_property_damage :
  property_damage_problem 70000 0.2 22000 := by
  sorry

end NUMINAMATH_CALUDE_solve_property_damage_l2973_297380


namespace NUMINAMATH_CALUDE_point_on_line_value_l2973_297309

theorem point_on_line_value (a b : ℝ) (h : b = 3 * a - 2) : 2 * b - 6 * a + 2 = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_value_l2973_297309


namespace NUMINAMATH_CALUDE_arithmetic_mean_fractions_l2973_297329

theorem arithmetic_mean_fractions : 
  let a := 7 / 11
  let b := 9 / 11
  let c := 8 / 11
  c = (a + b) / 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fractions_l2973_297329


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l2973_297311

/-- The equation of the tangent line to y = x³ + 2x + 1 at x = 1 is 5x - y - 1 = 0 -/
theorem tangent_line_at_x_1 : 
  let f (x : ℝ) := x^3 + 2*x + 1
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := (3 * x₀^2 + 2)
  ∀ x y : ℝ, (y - y₀ = k * (x - x₀)) ↔ (5*x - y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l2973_297311


namespace NUMINAMATH_CALUDE_max_score_for_successful_teams_l2973_297374

/-- Represents a football tournament with the given conditions -/
structure FootballTournament where
  num_teams : Nat
  num_successful_teams : Nat
  points_for_win : Nat
  points_for_draw : Nat
  points_for_loss : Nat

/-- The maximum score that can be achieved by the successful teams -/
def max_total_score (t : FootballTournament) : Nat :=
  let internal_matches := t.num_successful_teams * (t.num_successful_teams - 1) / 2
  let external_matches := t.num_successful_teams * (t.num_teams - t.num_successful_teams)
  (internal_matches + external_matches) * t.points_for_win

/-- The theorem stating the maximum integer N for which at least 6 teams can score N points -/
theorem max_score_for_successful_teams (t : FootballTournament) 
    (h1 : t.num_teams = 15)
    (h2 : t.num_successful_teams = 6)
    (h3 : t.points_for_win = 3)
    (h4 : t.points_for_draw = 1)
    (h5 : t.points_for_loss = 0) :
    ∃ (N : Nat), N = 34 ∧ 
    (∀ (M : Nat), (M > N → ¬(t.num_successful_teams * M ≤ max_total_score t))) ∧
    (t.num_successful_teams * N ≤ max_total_score t) := by
  sorry

end NUMINAMATH_CALUDE_max_score_for_successful_teams_l2973_297374


namespace NUMINAMATH_CALUDE_intersection_M_N_l2973_297334

def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

theorem intersection_M_N : ∀ x : ℝ, x ∈ (M ∩ N) ↔ 2 < x ∧ x ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2973_297334


namespace NUMINAMATH_CALUDE_kelly_carrot_harvest_l2973_297360

/-- The weight of Kelly's harvested carrots -/
def kelly_carrot_weight (bed1 bed2 bed3 carrots_per_pound : ℕ) : ℚ :=
  (bed1 + bed2 + bed3 : ℚ) / carrots_per_pound

/-- Theorem: Kelly harvested 39 pounds of carrots -/
theorem kelly_carrot_harvest :
  kelly_carrot_weight 55 101 78 6 = 39 := by
  sorry

end NUMINAMATH_CALUDE_kelly_carrot_harvest_l2973_297360


namespace NUMINAMATH_CALUDE_min_side_b_in_special_triangle_l2973_297349

/-- 
Given a triangle ABC where:
- Angles A, B, and C form an arithmetic sequence
- Sides opposite to angles A, B, and C are a, b, and c respectively
- 3ac + b² = 25
This theorem states that the minimum value of side b is 5/2
-/
theorem min_side_b_in_special_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → c > 0 →  -- Ensuring positive side lengths
  2 * B = A + C →  -- Arithmetic sequence condition
  A + B + C = π →  -- Sum of angles in a triangle
  3 * a * c + b^2 = 25 →  -- Given condition
  b ≥ 5/2 ∧ ∃ (a₀ c₀ : ℝ), a₀ > 0 ∧ c₀ > 0 ∧ 3 * a₀ * c₀ + (5/2)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_side_b_in_special_triangle_l2973_297349


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2973_297373

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for B ⊆ A
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≤ 3 := by sorry

-- Theorem for A ∩ B = ∅
theorem disjoint_condition (m : ℝ) : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l2973_297373


namespace NUMINAMATH_CALUDE_power_of_sum_squares_and_abs_l2973_297318

theorem power_of_sum_squares_and_abs (a b : ℝ) : 
  (a - 4)^2 + |2 - b| = 0 → a^b = 16 := by
  sorry

end NUMINAMATH_CALUDE_power_of_sum_squares_and_abs_l2973_297318


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l2973_297347

theorem tan_sum_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + 
   Real.tan (40 * π / 180) + Real.tan (50 * π / 180)) / Real.sin (30 * π / 180) = 
  2 * (Real.cos (40 * π / 180) * Real.cos (50 * π / 180) + 
       Real.cos (20 * π / 180) * Real.cos (30 * π / 180)) / 
      (Real.cos (20 * π / 180) * Real.cos (30 * π / 180) * 
       Real.cos (40 * π / 180) * Real.cos (50 * π / 180)) := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l2973_297347


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l2973_297310

/-- The number of candy pieces Katie and her sister have left after eating some on Halloween night -/
theorem halloween_candy_theorem (katie_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : 
  katie_candy = 8 → sister_candy = 23 → eaten_candy = 8 →
  katie_candy + sister_candy - eaten_candy = 23 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l2973_297310


namespace NUMINAMATH_CALUDE_packages_to_deliver_l2973_297393

/-- The number of packages received yesterday -/
def packages_yesterday : ℕ := 80

/-- The number of packages received today -/
def packages_today : ℕ := 2 * packages_yesterday

/-- The total number of packages to be delivered tomorrow -/
def total_packages : ℕ := packages_yesterday + packages_today

theorem packages_to_deliver :
  total_packages = 240 :=
sorry

end NUMINAMATH_CALUDE_packages_to_deliver_l2973_297393


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2973_297342

/-- The number of sides of a regular polygon whose sum of interior angles is 1080° more than
    the sum of exterior angles of a pentagon. -/
def num_sides_regular_polygon : ℕ := 10

/-- The sum of exterior angles of any polygon is always 360°. -/
axiom sum_exterior_angles : ℕ → ℝ
axiom sum_exterior_angles_def : ∀ n : ℕ, sum_exterior_angles n = 360

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180°. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem stating that the number of sides of the regular polygon is 10. -/
theorem regular_polygon_sides :
  sum_interior_angles num_sides_regular_polygon =
  sum_exterior_angles 5 + 1080 :=
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2973_297342


namespace NUMINAMATH_CALUDE_clare_bought_four_loaves_l2973_297356

def clares_bread_purchase (initial_money : ℕ) (milk_cartons : ℕ) (bread_cost : ℕ) (milk_cost : ℕ) (money_left : ℕ) : ℕ :=
  ((initial_money - money_left) - (milk_cartons * milk_cost)) / bread_cost

theorem clare_bought_four_loaves :
  clares_bread_purchase 47 2 2 2 35 = 4 := by
  sorry

end NUMINAMATH_CALUDE_clare_bought_four_loaves_l2973_297356


namespace NUMINAMATH_CALUDE_buddy_gym_class_size_l2973_297344

/-- The number of students in Buddy's gym class -/
def total_students (group1 : ℕ) (group2 : ℕ) : ℕ := group1 + group2

/-- Theorem stating the total number of students in Buddy's gym class -/
theorem buddy_gym_class_size :
  total_students 34 37 = 71 := by
  sorry

end NUMINAMATH_CALUDE_buddy_gym_class_size_l2973_297344


namespace NUMINAMATH_CALUDE_root_difference_of_cubic_l2973_297365

theorem root_difference_of_cubic (x₁ x₂ x₃ : ℝ) :
  (81 * x₁^3 - 162 * x₁^2 + 108 * x₁ - 18 = 0) →
  (81 * x₂^3 - 162 * x₂^2 + 108 * x₂ - 18 = 0) →
  (81 * x₃^3 - 162 * x₃^2 + 108 * x₃ - 18 = 0) →
  (x₂ - x₁ = x₃ - x₂) →  -- arithmetic progression condition
  (max x₁ (max x₂ x₃) - min x₁ (min x₂ x₃) = 2/3) :=
by sorry

end NUMINAMATH_CALUDE_root_difference_of_cubic_l2973_297365


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2973_297377

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2973_297377


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_l2973_297328

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - 2*a^2 - a + 2 = 0) → 
  (b^3 - 2*b^2 - b + 2 = 0) → 
  (c^3 - 2*c^2 - c + 2 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1 / (a + 2) + 1 / (b + 2) + 1 / (c + 2) = -19 / 16) :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_l2973_297328


namespace NUMINAMATH_CALUDE_twenty_fifth_is_monday_l2973_297350

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Checks if a given number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 == 0

/-- Represents a month with its dates -/
structure Month where
  dates : List Date
  threeEvenSaturdays : ∃ (d1 d2 d3 : Date),
    d1 ∈ dates ∧ d2 ∈ dates ∧ d3 ∈ dates ∧
    d1.dayOfWeek = DayOfWeek.Saturday ∧
    d2.dayOfWeek = DayOfWeek.Saturday ∧
    d3.dayOfWeek = DayOfWeek.Saturday ∧
    isEven d1.day ∧ isEven d2.day ∧ isEven d3.day ∧
    d1.day ≠ d2.day ∧ d2.day ≠ d3.day ∧ d1.day ≠ d3.day

/-- Theorem: In a month where three Saturdays fall on even dates, 
    the 25th day of that month is a Monday -/
theorem twenty_fifth_is_monday (m : Month) : 
  ∃ (d : Date), d ∈ m.dates ∧ d.day = 25 ∧ d.dayOfWeek = DayOfWeek.Monday := by
  sorry

end NUMINAMATH_CALUDE_twenty_fifth_is_monday_l2973_297350


namespace NUMINAMATH_CALUDE_special_hexagon_area_l2973_297354

/-- An equilateral hexagon with specific interior angles -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side_length : ℝ
  -- Interior angles of the hexagon in radians
  angles : Fin 6 → ℝ
  -- The hexagon is equilateral
  equilateral : side_length = 1
  -- The interior angles are as specified
  angle_values : angles = ![π/2, 2*π/3, 5*π/6, π/2, 2*π/3, 5*π/6]

/-- The area of the special hexagon -/
def area (h : SpecialHexagon) : ℝ := sorry

/-- Theorem stating the area of the special hexagon -/
theorem special_hexagon_area (h : SpecialHexagon) : area h = (3 + Real.sqrt 3) / 2 := by sorry

end NUMINAMATH_CALUDE_special_hexagon_area_l2973_297354


namespace NUMINAMATH_CALUDE_quadratic_range_theorem_l2973_297386

/-- The quadratic function f(x) = x^2 + 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 2

/-- A point P with coordinates (m, n) -/
structure Point where
  m : ℝ
  n : ℝ

theorem quadratic_range_theorem (P : Point) 
  (h1 : P.n = f P.m)  -- P lies on the graph of f
  (h2 : |P.m| < 2)    -- distance from P to y-axis is less than 2
  : -1 ≤ P.n ∧ P.n < 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_theorem_l2973_297386


namespace NUMINAMATH_CALUDE_max_min_product_l2973_297312

theorem max_min_product (A B : ℕ) (sum_constraint : A + B = 100) :
  (∃ (X Y : ℕ), X + Y = 100 ∧ X * Y ≤ A * B) ∧
  (∃ (X Y : ℕ), X + Y = 100 ∧ A * B ≤ X * Y) ∧
  (∃ (X Y : ℕ), X + Y = 100 ∧ X * Y = 2500) ∧
  (∃ (X Y : ℕ), X + Y = 100 ∧ X * Y = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_min_product_l2973_297312


namespace NUMINAMATH_CALUDE_unique_functional_equation_l2973_297348

theorem unique_functional_equation (f : ℕ+ → ℕ+)
  (h : ∀ m n : ℕ+, f (f m + f n) = m + n) :
  f 1988 = 1988 := by
  sorry

end NUMINAMATH_CALUDE_unique_functional_equation_l2973_297348


namespace NUMINAMATH_CALUDE_remaining_payment_example_l2973_297346

/-- Given a deposit percentage and amount, calculates the remaining amount to be paid -/
def remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) : ℚ :=
  let total_cost := deposit_amount / deposit_percentage
  total_cost - deposit_amount

/-- Theorem: Given a 10% deposit of $130, the remaining amount to be paid is $1170 -/
theorem remaining_payment_example : 
  remaining_payment (1/10) 130 = 1170 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_example_l2973_297346


namespace NUMINAMATH_CALUDE_total_caps_produced_l2973_297383

def week1_production : ℕ := 320
def week2_production : ℕ := 400
def week3_production : ℕ := 300

def average_production : ℕ := (week1_production + week2_production + week3_production) / 3

def total_production : ℕ := week1_production + week2_production + week3_production + average_production

theorem total_caps_produced : total_production = 1360 := by
  sorry

end NUMINAMATH_CALUDE_total_caps_produced_l2973_297383


namespace NUMINAMATH_CALUDE_binomial_coefficient_probability_l2973_297399

theorem binomial_coefficient_probability : 
  let n : ℕ := 10
  let positive_coeff : ℕ := 6
  let negative_coeff : ℕ := 5
  let total_coeff : ℕ := positive_coeff + negative_coeff
  let ways_to_choose_opposite_signs : ℕ := positive_coeff * negative_coeff
  let total_ways_to_choose : ℕ := (total_coeff * (total_coeff - 1)) / 2
  (ways_to_choose_opposite_signs : ℚ) / total_ways_to_choose = 6 / 11 :=
by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_probability_l2973_297399


namespace NUMINAMATH_CALUDE_right_trapezoid_area_l2973_297303

/-- The area of a right trapezoid with specific base proportions -/
theorem right_trapezoid_area : ∀ (lower_base : ℝ),
  lower_base > 0 →
  let upper_base := (3 / 5) * lower_base
  let height := (lower_base - upper_base) / 2
  (lower_base - 8 = height) →
  (1 / 2) * (lower_base + upper_base) * height = 192 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_area_l2973_297303


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2973_297362

/-- Represents an ellipse with the given equation -/
structure Ellipse (k : ℝ) :=
  (eq : ∀ x y : ℝ, x^2 / (k - 4) + y^2 / (10 - k) = 1)

/-- Condition for the foci to be on the x-axis -/
def foci_on_x_axis (k : ℝ) : Prop :=
  k - 4 > 10 - k

/-- The main theorem stating that 4 < k < 10 is necessary but not sufficient -/
theorem necessary_but_not_sufficient :
  ∃ k : ℝ, 4 < k ∧ k < 10 ∧
  (∀ k' : ℝ, (∃ e : Ellipse k', foci_on_x_axis k') → 4 < k' ∧ k' < 10) ∧
  ¬(∀ k' : ℝ, 4 < k' ∧ k' < 10 → ∃ e : Ellipse k', foci_on_x_axis k') :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2973_297362


namespace NUMINAMATH_CALUDE_median_salary_is_25000_l2973_297305

/-- Represents a position in the company -/
inductive Position
  | CEO
  | SeniorVicePresident
  | Manager
  | AssistantManager
  | Clerk

/-- Represents the salary distribution in the company -/
def salary_distribution : List (Position × Nat × Nat) :=
  [(Position.CEO, 1, 135000),
   (Position.SeniorVicePresident, 4, 95000),
   (Position.Manager, 12, 80000),
   (Position.AssistantManager, 8, 55000),
   (Position.Clerk, 38, 25000)]

/-- The total number of employees in the company -/
def total_employees : Nat := 63

/-- Calculates the median salary given the salary distribution and total number of employees -/
def median_salary (dist : List (Position × Nat × Nat)) (total : Nat) : Nat :=
  sorry

/-- Theorem stating that the median salary is $25,000 -/
theorem median_salary_is_25000 :
  median_salary salary_distribution total_employees = 25000 := by
  sorry

end NUMINAMATH_CALUDE_median_salary_is_25000_l2973_297305


namespace NUMINAMATH_CALUDE_probability_of_one_in_pascal_triangle_l2973_297361

/-- Represents Pascal's Triangle up to a given number of rows -/
def PascalTriangle (n : ℕ) : List (List ℕ) :=
  sorry

/-- Counts the number of 1s in the first n rows of Pascal's Triangle -/
def countOnes (n : ℕ) : ℕ :=
  sorry

/-- Counts the total number of elements in the first n rows of Pascal's Triangle -/
def totalElements (n : ℕ) : ℕ :=
  sorry

/-- Theorem: The probability of randomly selecting a 1 from the first 15 rows of Pascal's Triangle is 29/120 -/
theorem probability_of_one_in_pascal_triangle : 
  (countOnes 15 : ℚ) / (totalElements 15 : ℚ) = 29 / 120 :=
sorry

end NUMINAMATH_CALUDE_probability_of_one_in_pascal_triangle_l2973_297361


namespace NUMINAMATH_CALUDE_square_difference_equality_l2973_297367

theorem square_difference_equality : (1 + 2)^2 - (1^2 + 2^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_equality_l2973_297367


namespace NUMINAMATH_CALUDE_seashell_collection_l2973_297395

theorem seashell_collection (current : ℕ) (target : ℕ) (additional : ℕ) : 
  current = 19 → target = 25 → current + additional = target → additional = 6 := by
  sorry

end NUMINAMATH_CALUDE_seashell_collection_l2973_297395


namespace NUMINAMATH_CALUDE_daal_consumption_reduction_l2973_297368

theorem daal_consumption_reduction (old_price new_price : ℝ) 
  (hold_price : old_price = 16) 
  (hnew_price : new_price = 20) : 
  (new_price - old_price) / old_price * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_daal_consumption_reduction_l2973_297368


namespace NUMINAMATH_CALUDE_domain_transformation_l2973_297331

/-- Given that the domain of f(x^2 - 1) is [0, 3], prove that the domain of f(2x - 1) is [0, 9/2] -/
theorem domain_transformation (f : ℝ → ℝ) :
  (∀ y, f y ≠ 0 → 0 ≤ y + 1 ∧ y + 1 ≤ 3) →
  (∀ x, f (2*x - 1) ≠ 0 → 0 ≤ x ∧ x ≤ 9/2) :=
sorry

end NUMINAMATH_CALUDE_domain_transformation_l2973_297331


namespace NUMINAMATH_CALUDE_circumscribed_circle_twice_inscribed_l2973_297351

/-- Given a square, the area of its circumscribed circle is twice the area of its inscribed circle -/
theorem circumscribed_circle_twice_inscribed (a : ℝ) (ha : a > 0) :
  let square_side := 2 * a
  let inscribed_radius := a
  let circumscribed_radius := a * Real.sqrt 2
  (π * circumscribed_radius ^ 2) = 2 * (π * inscribed_radius ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_twice_inscribed_l2973_297351


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l2973_297398

/-- A point in a 2D plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the y-axis. -/
def symmetricToYAxis (a b : Point) : Prop :=
  b.x = -a.x ∧ b.y = a.y

/-- The theorem stating that if A(2, -5) is symmetric to B with respect to the y-axis,
    then B has coordinates (-2, -5). -/
theorem symmetry_coordinates :
  let a : Point := ⟨2, -5⟩
  let b : Point := ⟨-2, -5⟩
  symmetricToYAxis a b → b = ⟨-2, -5⟩ := by
  sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l2973_297398


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2973_297338

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) ↔ 
  ((a^2 + b^2 = 1 → ∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) ∧
   (∃ a b : ℝ, a^2 + b^2 ≠ 1 ∧ ∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2973_297338


namespace NUMINAMATH_CALUDE_profit_fluctuation_l2973_297324

theorem profit_fluctuation (march_profit : ℝ) (april_may_decrease : ℝ) :
  let april_profit := march_profit * 1.5
  let may_profit := april_profit * (1 - april_may_decrease / 100)
  let june_profit := may_profit * 1.5
  june_profit = march_profit * 1.8 →
  april_may_decrease = 20 := by
sorry

end NUMINAMATH_CALUDE_profit_fluctuation_l2973_297324


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2973_297352

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) :
  b^2 + c^2 - a^2 = 1 →
  b * c = 1 →
  Real.cos B * Real.cos C = -1/8 →
  a + b + c = Real.sqrt 2 + Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2973_297352


namespace NUMINAMATH_CALUDE_min_exercise_hours_l2973_297364

/-- Represents the exercise data for a month -/
structure ExerciseData where
  days_20min : Nat
  days_40min : Nat
  days_2hours : Nat
  min_exercise_time : Nat
  max_exercise_time : Nat

/-- Calculates the minimum number of hours exercised in a month -/
def min_hours_exercised (data : ExerciseData) : Rat :=
  let hours_2hours := data.days_2hours * 2
  let hours_40min := (data.days_40min - data.days_2hours) * 2 / 3
  let hours_20min := (data.days_20min - data.days_40min) * 1 / 3
  hours_2hours + hours_40min + hours_20min

/-- Theorem stating the minimum number of hours exercised -/
theorem min_exercise_hours (data : ExerciseData) 
  (h1 : data.days_20min = 26)
  (h2 : data.days_40min = 24)
  (h3 : data.days_2hours = 4)
  (h4 : data.min_exercise_time = 20)
  (h5 : data.max_exercise_time = 120) :
  min_hours_exercised data = 22 := by
  sorry

end NUMINAMATH_CALUDE_min_exercise_hours_l2973_297364


namespace NUMINAMATH_CALUDE_school_early_arrival_l2973_297381

theorem school_early_arrival (usual_time : ℝ) (rate_ratio : ℝ) (time_saved : ℝ) : 
  usual_time = 24 →
  rate_ratio = 6 / 5 →
  time_saved = usual_time - (usual_time / rate_ratio) →
  time_saved = 4 := by
sorry

end NUMINAMATH_CALUDE_school_early_arrival_l2973_297381


namespace NUMINAMATH_CALUDE_derivative_exp_cos_l2973_297369

/-- The derivative of e^x * cos(x) is e^x * (cos(x) - sin(x)) -/
theorem derivative_exp_cos (x : ℝ) : 
  deriv (λ x => Real.exp x * Real.cos x) x = Real.exp x * (Real.cos x - Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_exp_cos_l2973_297369


namespace NUMINAMATH_CALUDE_inequality_proof_l2973_297371

theorem inequality_proof (n : ℕ) (x : ℝ) (h1 : n > 0) (h2 : x ≥ n^2) :
  n * Real.sqrt (x - n^2) ≤ x / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2973_297371


namespace NUMINAMATH_CALUDE_inverse_f_at_3_l2973_297357

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2

-- Define the domain of f
def domain (x : ℝ) : Prop := -2 ≤ x ∧ x < 0

-- State the theorem
theorem inverse_f_at_3 :
  ∃ (f_inv : ℝ → ℝ), 
    (∀ x, domain x → f_inv (f x) = x) ∧
    (∀ y, ∃ x, domain x ∧ f x = y → f_inv y = x) ∧
    f_inv 3 = -1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_at_3_l2973_297357


namespace NUMINAMATH_CALUDE_union_equality_condition_l2973_297335

open Set

theorem union_equality_condition (a : ℝ) :
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
  let B : Set ℝ := {x | a ≤ x ∧ x ≤ a + 2}
  (A ∪ B = A) ↔ a ∈ Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_union_equality_condition_l2973_297335


namespace NUMINAMATH_CALUDE_pauls_garage_sale_l2973_297316

/-- The number of books Paul sold in the garage sale -/
def books_sold (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : ℕ :=
  initial - given_away - remaining

/-- Proof that Paul sold 27 books in the garage sale -/
theorem pauls_garage_sale : books_sold 134 39 68 = 27 := by
  sorry

end NUMINAMATH_CALUDE_pauls_garage_sale_l2973_297316


namespace NUMINAMATH_CALUDE_factorial_fraction_equality_l2973_297392

theorem factorial_fraction_equality : (5 * Nat.factorial 7 + 35 * Nat.factorial 6) / Nat.factorial 8 = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_factorial_fraction_equality_l2973_297392


namespace NUMINAMATH_CALUDE_intersection_M_N_l2973_297332

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set N
def N : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Icc 1 3 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2973_297332


namespace NUMINAMATH_CALUDE_limit_two_x_sin_x_over_one_minus_cos_x_l2973_297389

/-- The limit of (2x sin x) / (1 - cos x) as x approaches 0 is equal to 4 -/
theorem limit_two_x_sin_x_over_one_minus_cos_x : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x| ∧ |x| < δ → |((2 * x * Real.sin x) / (1 - Real.cos x)) - 4| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_two_x_sin_x_over_one_minus_cos_x_l2973_297389


namespace NUMINAMATH_CALUDE_modular_inverse_of_3_mod_31_l2973_297382

theorem modular_inverse_of_3_mod_31 :
  ∃ x : ℕ, x ≤ 30 ∧ (3 * x) % 31 = 1 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_modular_inverse_of_3_mod_31_l2973_297382


namespace NUMINAMATH_CALUDE_average_age_of_ten_students_l2973_297302

theorem average_age_of_ten_students
  (total_students : ℕ)
  (average_age_all : ℚ)
  (num_group1 : ℕ)
  (average_age_group1 : ℚ)
  (age_last_student : ℕ)
  (h1 : total_students = 25)
  (h2 : average_age_all = 25)
  (h3 : num_group1 = 14)
  (h4 : average_age_group1 = 28)
  (h5 : age_last_student = 13)
  : ∃ (average_age_group2 : ℚ),
    average_age_group2 = 22 ∧
    average_age_group2 * (total_students - num_group1 - 1) =
      total_students * average_age_all - num_group1 * average_age_group1 - age_last_student :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_ten_students_l2973_297302


namespace NUMINAMATH_CALUDE_ball_count_theorem_l2973_297385

/-- Represents the number of balls of each color in a jar -/
structure BallCount where
  white : ℕ
  red : ℕ
  blue : ℕ

/-- Checks if the given ball count satisfies the ratio 4:3:2 for white:red:blue -/
def satisfiesRatio (bc : BallCount) : Prop :=
  4 * bc.red = 3 * bc.white ∧ 4 * bc.blue = 2 * bc.white

theorem ball_count_theorem (bc : BallCount) 
  (ratio_satisfied : satisfiesRatio bc) 
  (white_count : bc.white = 16) : 
  bc.red = 12 ∧ bc.blue = 8 := by
  sorry

#check ball_count_theorem

end NUMINAMATH_CALUDE_ball_count_theorem_l2973_297385


namespace NUMINAMATH_CALUDE_max_product_sum_300_l2973_297397

theorem max_product_sum_300 : 
  (∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500) ∧ 
  (∀ (x y : ℤ), x + y = 300 → x * y ≤ 22500) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_300_l2973_297397
