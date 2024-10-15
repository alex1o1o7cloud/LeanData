import Mathlib

namespace NUMINAMATH_CALUDE_kaleb_games_proof_l1662_166230

theorem kaleb_games_proof (sold : ℕ) (boxes : ℕ) (games_per_box : ℕ) 
  (h1 : sold = 46)
  (h2 : boxes = 6)
  (h3 : games_per_box = 5) :
  sold + boxes * games_per_box = 76 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_games_proof_l1662_166230


namespace NUMINAMATH_CALUDE_f_intersects_x_axis_l1662_166206

-- Define the function f(x) = x + 1
def f (x : ℝ) : ℝ := x + 1

-- Theorem stating that f intersects the x-axis
theorem f_intersects_x_axis : ∃ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_x_axis_l1662_166206


namespace NUMINAMATH_CALUDE_second_term_value_l1662_166210

/-- A geometric sequence with sum of first n terms Sn = a·3^n - 2 -/
def GeometricSequence (a : ℝ) : ℕ → ℝ := sorry

/-- Sum of first n terms of the geometric sequence -/
def Sn (a : ℝ) (n : ℕ) : ℝ := a * 3^n - 2

/-- The second term of the sequence -/
def a2 (a : ℝ) : ℝ := GeometricSequence a 2

theorem second_term_value (a : ℝ) :
  a2 a = 12 :=
sorry

end NUMINAMATH_CALUDE_second_term_value_l1662_166210


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1662_166233

/-- If x^2 + 3x + a = 0 has -1 as one of its roots, then the other root is -2 -/
theorem other_root_of_quadratic (a : ℝ) : 
  ((-1 : ℝ)^2 + 3*(-1) + a = 0) → 
  (∃ x : ℝ, x ≠ -1 ∧ x^2 + 3*x + a = 0 ∧ x = -2) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1662_166233


namespace NUMINAMATH_CALUDE_range_of_a_range_of_m_l1662_166270

-- Define the propositions
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 - 4*x + a^2 > 0

def q (a : ℝ) : Prop := ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + (a+1)*x + a - 1 = 0 ∧ y^2 + (a+1)*y + a - 1 = 0

def r (a m : ℝ) : Prop := a^2 - 2*a + 1 - m^2 ≥ 0 ∧ m > 0

-- Theorem 1
theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬(p a ∧ q a) → (-2 ≤ a ∧ a < 1) ∨ a > 2 :=
sorry

-- Theorem 2
theorem range_of_m (m : ℝ) : (∀ a : ℝ, ¬(r a m) → ¬(p a)) ∧ (∃ a : ℝ, ¬(r a m) ∧ p a) → m > 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_range_of_m_l1662_166270


namespace NUMINAMATH_CALUDE_team_average_score_l1662_166252

theorem team_average_score (player1_score : ℕ) (player2_score : ℕ) (player3_score : ℕ) :
  player1_score = 20 →
  player2_score = player1_score / 2 →
  player3_score = 6 * player2_score →
  (player1_score + player2_score + player3_score) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_team_average_score_l1662_166252


namespace NUMINAMATH_CALUDE_bugs_meeting_point_l1662_166280

/-- A quadrilateral with sides of length 5, 7, 8, and 6 -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (ab_length : dist A B = 5)
  (bc_length : dist B C = 7)
  (cd_length : dist C D = 8)
  (da_length : dist D A = 6)

/-- The point where two bugs meet when starting from A and moving in opposite directions -/
def meeting_point (q : Quadrilateral) : ℝ × ℝ := sorry

/-- The distance between point B and the meeting point E -/
def BE (q : Quadrilateral) : ℝ := dist q.B (meeting_point q)

theorem bugs_meeting_point (q : Quadrilateral) : BE q = 6 := by sorry

end NUMINAMATH_CALUDE_bugs_meeting_point_l1662_166280


namespace NUMINAMATH_CALUDE_three_digit_sum_divisible_by_eleven_l1662_166293

theorem three_digit_sum_divisible_by_eleven (a b : ℕ) : 
  (100 ≤ 400 + 10*a + 3 ∧ 400 + 10*a + 3 < 1000) →  -- 4a3 is a 3-digit number
  (400 + 10*a + 3) + 984 = 1000 + 300 + 10*b + 7 →  -- 4a3 + 984 = 13b7
  (1000 + 300 + 10*b + 7) % 11 = 0 →                -- 13b7 is divisible by 11
  a + b = 10 := by
sorry

end NUMINAMATH_CALUDE_three_digit_sum_divisible_by_eleven_l1662_166293


namespace NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l1662_166208

theorem parametric_to_ordinary_equation (α : ℝ) :
  let x := Real.sin (α / 2) + Real.cos (α / 2)
  let y := Real.sqrt (2 + Real.sin α)
  (y ^ 2 - x ^ 2 = 1) ∧
  (|x| ≤ Real.sqrt 2) ∧
  (1 ≤ y) ∧
  (y ≤ Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_parametric_to_ordinary_equation_l1662_166208


namespace NUMINAMATH_CALUDE_max_value_of_function_l1662_166213

theorem max_value_of_function (x : ℝ) (hx : x < 0) : 
  2 * x + 2 / x ≤ -4 ∧ 
  ∃ y : ℝ, y < 0 ∧ 2 * y + 2 / y = -4 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1662_166213


namespace NUMINAMATH_CALUDE_constant_k_equality_l1662_166247

theorem constant_k_equality (x : ℝ) : 
  -x^2 - (-17 + 11)*x - 8 = -(x - 2)*(x - 4) := by
  sorry

end NUMINAMATH_CALUDE_constant_k_equality_l1662_166247


namespace NUMINAMATH_CALUDE_arnel_pencil_sharing_l1662_166226

/-- Calculates the number of pencils each friend receives when Arnel shares his pencils --/
def pencils_per_friend (num_boxes : ℕ) (pencils_per_box : ℕ) (kept_pencils : ℕ) (num_friends : ℕ) : ℕ :=
  ((num_boxes * pencils_per_box) - kept_pencils) / num_friends

/-- Proves that each friend receives 8 pencils under the given conditions --/
theorem arnel_pencil_sharing :
  pencils_per_friend 10 5 10 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arnel_pencil_sharing_l1662_166226


namespace NUMINAMATH_CALUDE_boxes_per_day_calculation_l1662_166298

/-- The number of apples packed in a box -/
def apples_per_box : ℕ := 40

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The total number of apples packed in two weeks -/
def total_apples : ℕ := 24500

/-- The number of fewer apples packed per day in the second week -/
def fewer_apples_per_day : ℕ := 500

/-- The number of full boxes produced per day -/
def boxes_per_day : ℕ := 50

theorem boxes_per_day_calculation :
  boxes_per_day * apples_per_box * days_per_week +
  (boxes_per_day * apples_per_box - fewer_apples_per_day) * days_per_week = total_apples :=
by sorry

end NUMINAMATH_CALUDE_boxes_per_day_calculation_l1662_166298


namespace NUMINAMATH_CALUDE_pencil_average_price_l1662_166220

/-- Given the purchase of pens and pencils, prove the average price of a pencil -/
theorem pencil_average_price 
  (total_cost : ℝ) 
  (num_pens : ℕ) 
  (num_pencils : ℕ) 
  (pen_avg_price : ℝ) 
  (h1 : total_cost = 570)
  (h2 : num_pens = 30)
  (h3 : num_pencils = 75)
  (h4 : pen_avg_price = 14) :
  (total_cost - num_pens * pen_avg_price) / num_pencils = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencil_average_price_l1662_166220


namespace NUMINAMATH_CALUDE_equation_solutions_l1662_166258

def equation (n : ℕ) (x : ℝ) : Prop :=
  (((x + 1)^2)^(1/n : ℝ)) + (((x - 1)^2)^(1/n : ℝ)) = 4 * ((x^2 - 1)^(1/n : ℝ))

theorem equation_solutions :
  (∀ x : ℝ, equation 2 x ↔ x = 2 / Real.sqrt 3 ∨ x = -2 / Real.sqrt 3) ∧
  (∀ x : ℝ, equation 3 x ↔ x = 3 * Real.sqrt 3 / 5 ∨ x = -3 * Real.sqrt 3 / 5) ∧
  (∀ x : ℝ, equation 4 x ↔ x = 7 / (4 * Real.sqrt 3) ∨ x = -7 / (4 * Real.sqrt 3)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1662_166258


namespace NUMINAMATH_CALUDE_unique_polynomial_function_l1662_166244

/-- A polynomial function over ℝ -/
def PolynomialFunction := ℝ → ℝ

/-- Predicate to check if a function is a polynomial of degree ≥ 1 -/
def IsPolynomialDegreeGEOne (f : PolynomialFunction) : Prop := sorry

/-- The conditions that the polynomial function must satisfy -/
def SatisfiesConditions (f : PolynomialFunction) : Prop :=
  IsPolynomialDegreeGEOne f ∧
  (∀ x : ℝ, f (x^2) = (f x)^3) ∧
  (∀ x : ℝ, f (f x) = f x)

/-- Theorem stating that there exists exactly one polynomial function satisfying the conditions -/
theorem unique_polynomial_function :
  ∃! f : PolynomialFunction, SatisfiesConditions f := by sorry

end NUMINAMATH_CALUDE_unique_polynomial_function_l1662_166244


namespace NUMINAMATH_CALUDE_max_ratio_on_circle_l1662_166243

-- Define a point with integer coordinates
structure IntPoint where
  x : ℤ
  y : ℤ

-- Define a function to check if a point is on the circle x^2 + y^2 = 16
def onCircle (p : IntPoint) : Prop :=
  p.x^2 + p.y^2 = 16

-- Define a function to calculate the squared distance between two points
def squaredDistance (p1 p2 : IntPoint) : ℤ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Theorem statement
theorem max_ratio_on_circle (A B C D : IntPoint) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  onCircle A ∧ onCircle B ∧ onCircle C ∧ onCircle D →
  ¬ ∃ m n : ℤ, m^2 = squaredDistance A B ∧ n > 0 →
  ¬ ∃ m n : ℤ, m^2 = squaredDistance C D ∧ n > 0 →
  ∀ r : ℚ, r * (squaredDistance C D : ℚ) ≤ (squaredDistance A B : ℚ) → r ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_ratio_on_circle_l1662_166243


namespace NUMINAMATH_CALUDE_mod_product_equivalence_l1662_166249

theorem mod_product_equivalence : ∃ m : ℕ, 
  198 * 955 ≡ m [ZMOD 50] ∧ 0 ≤ m ∧ m < 50 ∧ m = 40 := by
  sorry

end NUMINAMATH_CALUDE_mod_product_equivalence_l1662_166249


namespace NUMINAMATH_CALUDE_correct_monk_bun_equations_l1662_166218

/-- Represents the monk and bun distribution problem -/
def monk_bun_problem (x y : ℕ) : Prop :=
  -- Total number of monks is 100
  x + y = 100 ∧
  -- Total number of buns is 100, distributed as 3 per elder monk and 1/3 per younger monk
  3 * x + y / 3 = 100

/-- The correct system of equations for the monk and bun distribution problem -/
theorem correct_monk_bun_equations :
  ∀ x y : ℕ, monk_bun_problem x y ↔ x + y = 100 ∧ 3 * x + y / 3 = 100 := by
  sorry

end NUMINAMATH_CALUDE_correct_monk_bun_equations_l1662_166218


namespace NUMINAMATH_CALUDE_peter_initial_erasers_l1662_166267

-- Define the variables
def initial_erasers : ℕ := sorry
def received_erasers : ℕ := 3
def final_erasers : ℕ := 11

-- State the theorem
theorem peter_initial_erasers : 
  initial_erasers + received_erasers = final_erasers → initial_erasers = 8 := by
  sorry

end NUMINAMATH_CALUDE_peter_initial_erasers_l1662_166267


namespace NUMINAMATH_CALUDE_golden_raisins_fraction_of_total_cost_l1662_166242

/-- Represents the cost of ingredients relative to golden raisins -/
structure IngredientCost where
  goldenRaisins : ℚ
  almonds : ℚ
  cashews : ℚ
  walnuts : ℚ

/-- Represents the weight of ingredients in pounds -/
structure IngredientWeight where
  goldenRaisins : ℚ
  almonds : ℚ
  cashews : ℚ
  walnuts : ℚ

def mixtureCost (cost : IngredientCost) (weight : IngredientWeight) : ℚ :=
  cost.goldenRaisins * weight.goldenRaisins +
  cost.almonds * weight.almonds +
  cost.cashews * weight.cashews +
  cost.walnuts * weight.walnuts

theorem golden_raisins_fraction_of_total_cost 
  (cost : IngredientCost)
  (weight : IngredientWeight)
  (h1 : cost.goldenRaisins = 1)
  (h2 : cost.almonds = 2 * cost.goldenRaisins)
  (h3 : cost.cashews = 3 * cost.goldenRaisins)
  (h4 : cost.walnuts = 4 * cost.goldenRaisins)
  (h5 : weight.goldenRaisins = 4)
  (h6 : weight.almonds = 2)
  (h7 : weight.cashews = 1)
  (h8 : weight.walnuts = 1) :
  (cost.goldenRaisins * weight.goldenRaisins) / mixtureCost cost weight = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_golden_raisins_fraction_of_total_cost_l1662_166242


namespace NUMINAMATH_CALUDE_square_difference_l1662_166260

theorem square_difference : (15 + 7)^2 - (15^2 + 7^2) = 210 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l1662_166260


namespace NUMINAMATH_CALUDE_fruit_vendor_problem_l1662_166215

-- Define the parameters
def total_boxes : ℕ := 60
def strawberry_price : ℕ := 60
def apple_price : ℕ := 40
def total_spent : ℕ := 3100
def profit_strawberry_A : ℕ := 15
def profit_apple_A : ℕ := 20
def profit_strawberry_B : ℕ := 12
def profit_apple_B : ℕ := 16
def profit_A : ℕ := 600

-- Define the theorem
theorem fruit_vendor_problem :
  ∃ (strawberry_boxes apple_boxes : ℕ),
    strawberry_boxes + apple_boxes = total_boxes ∧
    strawberry_boxes * strawberry_price + apple_boxes * apple_price = total_spent ∧
    strawberry_boxes = 35 ∧
    apple_boxes = 25 ∧
    (∃ (a b : ℕ),
      a + b ≤ total_boxes ∧
      a * profit_strawberry_A + b * profit_apple_A = profit_A ∧
      (strawberry_boxes - a) * profit_strawberry_B + (apple_boxes - b) * profit_apple_B = 340 ∧
      (a + b = 52 ∨ a + b = 53)) :=
sorry

end NUMINAMATH_CALUDE_fruit_vendor_problem_l1662_166215


namespace NUMINAMATH_CALUDE_square_area_ratio_l1662_166229

theorem square_area_ratio (y : ℝ) (hy : y > 0) :
  (y^2) / ((3*y)^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1662_166229


namespace NUMINAMATH_CALUDE_jeff_trucks_count_l1662_166264

theorem jeff_trucks_count :
  ∀ (trucks cars : ℕ),
    cars = 2 * trucks →
    trucks + cars = 60 →
    trucks = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_jeff_trucks_count_l1662_166264


namespace NUMINAMATH_CALUDE_number_difference_proof_l1662_166241

theorem number_difference_proof (L S : ℕ) (h1 : L = 1636) (h2 : L = 6 * S + 10) : 
  L - S = 1365 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_proof_l1662_166241


namespace NUMINAMATH_CALUDE_power_function_through_point_l1662_166228

def is_power_function (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

theorem power_function_through_point :
  ∀ f : ℝ → ℝ, is_power_function f →
  f 2 = (1/4 : ℝ) →
  ∃ a : ℝ, (∀ x : ℝ, f x = x^a) ∧ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l1662_166228


namespace NUMINAMATH_CALUDE_event_B_more_likely_l1662_166217

/-- Represents the number of sides on a fair die -/
def numSides : ℕ := 6

/-- Represents the number of throws -/
def numThrows : ℕ := 3

/-- Probability of event B: three different numbers appear in three throws -/
def probB : ℚ := 5 / 9

/-- Probability of event A: at least one number appears at least twice in three throws -/
def probA : ℚ := 4 / 9

/-- Theorem stating that event B is more likely than event A -/
theorem event_B_more_likely : probB > probA := by sorry

end NUMINAMATH_CALUDE_event_B_more_likely_l1662_166217


namespace NUMINAMATH_CALUDE_roots_of_g_l1662_166279

theorem roots_of_g (a b : ℝ) : 
  (∀ x : ℝ, (x^2 - a*x + b = 0) ↔ (x = 2 ∨ x = 3)) →
  (∀ x : ℝ, (b*x^2 - a*x - 1 = 0) ↔ (x = 1 ∨ x = -1/6)) := by
sorry

end NUMINAMATH_CALUDE_roots_of_g_l1662_166279


namespace NUMINAMATH_CALUDE_mary_initial_amount_l1662_166202

def marco_initial : ℕ := 24

theorem mary_initial_amount (mary_initial : ℕ) : mary_initial = 27 :=
  by
  have h1 : mary_initial + marco_initial / 2 > marco_initial / 2 := by sorry
  have h2 : mary_initial - 5 = marco_initial / 2 + 10 := by sorry
  sorry


end NUMINAMATH_CALUDE_mary_initial_amount_l1662_166202


namespace NUMINAMATH_CALUDE_absolute_difference_21st_terms_l1662_166275

-- Define arithmetic sequence
def arithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

-- Define the sequences C and D
def C (n : ℕ) : ℤ := arithmeticSequence 50 12 n
def D (n : ℕ) : ℤ := arithmeticSequence 50 (-14) n

-- State the theorem
theorem absolute_difference_21st_terms :
  |C 21 - D 21| = 520 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_21st_terms_l1662_166275


namespace NUMINAMATH_CALUDE_combination_equality_l1662_166282

theorem combination_equality (x : ℕ) : 
  (Nat.choose 10 x = Nat.choose 10 (3 * x - 2)) → (x = 1 ∨ x = 3) := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l1662_166282


namespace NUMINAMATH_CALUDE_tan_alpha_and_expression_l1662_166232

theorem tan_alpha_and_expression (α : ℝ) 
  (h : Real.tan (π / 4 + α) = 2) : 
  Real.tan α = 1 / 3 ∧ 
  (2 * Real.sin α ^ 2 + Real.sin (2 * α)) / (1 + Real.tan α) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_and_expression_l1662_166232


namespace NUMINAMATH_CALUDE_equation_has_four_real_solutions_l1662_166251

theorem equation_has_four_real_solutions :
  ∃ (s : Finset ℝ), (∀ x ∈ s, x^2 + 1/x^2 = 2006 + 1/2006) ∧ s.card = 4 ∧
  (∀ y : ℝ, y^2 + 1/y^2 = 2006 + 1/2006 → y ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_four_real_solutions_l1662_166251


namespace NUMINAMATH_CALUDE_polynomial_factor_implies_constant_value_l1662_166221

theorem polynomial_factor_implies_constant_value (a : ℝ) : 
  (∃ b : ℝ, ∀ y : ℝ, y^2 + 3*y - a = (y - 3) * (y + b)) → a = 18 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_implies_constant_value_l1662_166221


namespace NUMINAMATH_CALUDE_ellipse_inequality_l1662_166286

theorem ellipse_inequality (a b x y : ℝ) (ha : a > 0) (hb : b > 0)
  (h : x^2 / a^2 + y^2 / b^2 ≤ 1) : a^2 + b^2 ≥ (x + y)^2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_inequality_l1662_166286


namespace NUMINAMATH_CALUDE_largest_n_is_max_l1662_166263

/-- The largest value of n for which 3x^2 + nx + 108 can be factored as the product of two linear factors with integer coefficients -/
def largest_n : ℕ := 325

/-- A polynomial of the form 3x^2 + nx + 108 -/
def polynomial (n : ℕ) (x : ℝ) : ℝ := 3 * x^2 + n * x + 108

/-- Predicate to check if a polynomial can be factored as the product of two linear factors with integer coefficients -/
def can_be_factored (n : ℕ) : Prop :=
  ∃ (a b : ℤ), ∀ (x : ℝ), polynomial n x = (3 * x + a) * (x + b)

/-- Theorem stating that largest_n is the largest value of n for which the polynomial can be factored -/
theorem largest_n_is_max :
  can_be_factored largest_n ∧
  ∀ (m : ℕ), m > largest_n → ¬(can_be_factored m) :=
sorry

end NUMINAMATH_CALUDE_largest_n_is_max_l1662_166263


namespace NUMINAMATH_CALUDE_pascal_triangle_prob_one_l1662_166268

/-- The number of rows in Pascal's Triangle we're considering -/
def n : ℕ := 20

/-- The total number of elements in the first n rows of Pascal's Triangle -/
def total_elements (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of 1s in the first n rows of Pascal's Triangle -/
def ones_count (n : ℕ) : ℕ := 2 * n - 1

/-- The probability of selecting a 1 from the first n rows of Pascal's Triangle -/
def prob_one (n : ℕ) : ℚ := (ones_count n : ℚ) / (total_elements n : ℚ)

theorem pascal_triangle_prob_one : 
  prob_one n = 39 / 210 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_prob_one_l1662_166268


namespace NUMINAMATH_CALUDE_cannot_reach_123456_l1662_166292

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def sequenceElement (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => sequenceElement n + sumOfDigits (sequenceElement n)

theorem cannot_reach_123456 : ∀ n : ℕ, sequenceElement n ≠ 123456 := by
  sorry

end NUMINAMATH_CALUDE_cannot_reach_123456_l1662_166292


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1662_166211

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m * (m + 1) < 500 → m + (m + 1) ≤ n + (n + 1)) → 
  n + (n + 1) = 43 :=
sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l1662_166211


namespace NUMINAMATH_CALUDE_existence_of_special_point_l1662_166238

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Checks if a triangle is acute-angled -/
def isAcute (t : Triangle) : Prop := sorry

/-- Checks if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop := sorry

/-- The feet of perpendiculars from a point to the sides of a triangle -/
def feetOfPerpendiculars (p : Point) (t : Triangle) : Triangle := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- The main theorem -/
theorem existence_of_special_point (t : Triangle) (h : isAcute t) : 
  ∃ Q : Point, isInside Q t ∧ isEquilateral (feetOfPerpendiculars Q t) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_point_l1662_166238


namespace NUMINAMATH_CALUDE_circles_intersection_condition_l1662_166236

/-- Two circles in the xy-plane -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 10*y + 1 = 0

def circle2 (x y m : ℝ) : Prop := x^2 + y^2 - 2*x + 2*y - m = 0

/-- The circles intersect -/
def circles_intersect (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle1 x y ∧ circle2 x y m

/-- Theorem stating the condition for the circles to intersect -/
theorem circles_intersection_condition :
  ∀ m : ℝ, circles_intersect m ↔ -1 < m ∧ m < 79 :=
sorry

end NUMINAMATH_CALUDE_circles_intersection_condition_l1662_166236


namespace NUMINAMATH_CALUDE_rational_roots_quadratic_l1662_166201

theorem rational_roots_quadratic (m : ℤ) :
  (∃ x y : ℚ, m * x^2 - (m - 1) * x + 1 = 0 ∧ m * y^2 - (m - 1) * y + 1 = 0 ∧ x ≠ y) →
  m = 6 ∧ (1/2 : ℚ) * m - (m - 1) * (1/2) + 1 = 0 ∧ (1/3 : ℚ) * m - (m - 1) * (1/3) + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_quadratic_l1662_166201


namespace NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l1662_166255

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_two_planes_implies_parallel 
  (l : Line) (α β : Plane) (h1 : α ≠ β) (h2 : perpendicular l α) (h3 : perpendicular l β) :
  parallel α β := by sorry

end NUMINAMATH_CALUDE_line_perp_two_planes_implies_parallel_l1662_166255


namespace NUMINAMATH_CALUDE_sum_of_seven_consecutive_integers_l1662_166248

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_seven_consecutive_integers_l1662_166248


namespace NUMINAMATH_CALUDE_average_age_of_new_men_l1662_166289

theorem average_age_of_new_men (n : ℕ) (old_avg : ℝ) (age1 age2 : ℕ) (increase : ℝ) :
  n = 15 →
  age1 = 21 →
  age2 = 23 →
  increase = 2 →
  (n * (old_avg + increase) - n * old_avg) = ((n * increase + age1 + age2) / 2) →
  ((n * increase + age1 + age2) / 2) = 37 :=
by sorry

end NUMINAMATH_CALUDE_average_age_of_new_men_l1662_166289


namespace NUMINAMATH_CALUDE_vincent_train_books_l1662_166234

theorem vincent_train_books (animal_books : ℕ) (space_books : ℕ) (book_cost : ℕ) (total_spent : ℕ) :
  animal_books = 10 →
  space_books = 1 →
  book_cost = 16 →
  total_spent = 224 →
  ∃ (train_books : ℕ), train_books = 3 ∧ total_spent = book_cost * (animal_books + space_books + train_books) :=
by sorry

end NUMINAMATH_CALUDE_vincent_train_books_l1662_166234


namespace NUMINAMATH_CALUDE_bedevir_will_participate_l1662_166214

/-- The combat skill of the n-th opponent -/
def opponent_skill (n : ℕ) : ℚ := 1 / (2^(n+1) - 1)

/-- The probability of Sir Bedevir winning against the n-th opponent -/
def win_probability (n : ℕ) : ℚ := 1 / (1 + opponent_skill n)

/-- Theorem: Sir Bedevir's probability of winning is greater than 1/2 for any opponent -/
theorem bedevir_will_participate (k : ℕ) (h : k > 1) :
  ∀ n, n < k → win_probability n > 1/2 := by sorry

end NUMINAMATH_CALUDE_bedevir_will_participate_l1662_166214


namespace NUMINAMATH_CALUDE_unique_perpendicular_line_l1662_166212

-- Define a point type
structure Point where
  x : ℝ
  y : ℝ

-- Define a line type
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a plane type
structure Plane where
  points : Set Point

-- Define what it means for a point to be on a line
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define what it means for two lines to be perpendicular
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- State the theorem
theorem unique_perpendicular_line 
  (plane : Plane) (l : Line) (p : Point) 
  (h : ¬ p.onLine l) : 
  ∃! l_perp : Line, 
    l_perp.perpendicular l ∧ 
    p.onLine l_perp :=
  sorry

end NUMINAMATH_CALUDE_unique_perpendicular_line_l1662_166212


namespace NUMINAMATH_CALUDE_shipping_cost_per_pound_l1662_166257

/-- Shipping cost calculation -/
theorem shipping_cost_per_pound 
  (flat_fee : ℝ) 
  (weight : ℝ) 
  (total_cost : ℝ) 
  (h1 : flat_fee = 5)
  (h2 : weight = 5)
  (h3 : total_cost = 9)
  (h4 : total_cost = flat_fee + weight * (total_cost - flat_fee) / weight) :
  (total_cost - flat_fee) / weight = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_per_pound_l1662_166257


namespace NUMINAMATH_CALUDE_rationalize_and_simplify_l1662_166235

theorem rationalize_and_simplify :
  ∃ (A B C D E : ℤ),
    (B < D) ∧
    (3 : ℝ) / (4 * Real.sqrt 5 + 3 * Real.sqrt 7) = 
      (A * Real.sqrt B + C * Real.sqrt D) / E ∧
    A = 12 ∧ B = 5 ∧ C = -9 ∧ D = 7 ∧ E = 17 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_and_simplify_l1662_166235


namespace NUMINAMATH_CALUDE_max_squares_covered_2inch_card_l1662_166269

/-- Represents a square card -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard -/
structure Checkerboard where
  square_size : ℝ

/-- Calculates the maximum number of squares a card can cover on a checkerboard -/
def max_squares_covered (card : Card) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a 1-inch checkerboard -/
theorem max_squares_covered_2inch_card (card : Card) (board : Checkerboard) :
  card.side_length = 2 →
  board.square_size = 1 →
  max_squares_covered card board = 9 :=
sorry

end NUMINAMATH_CALUDE_max_squares_covered_2inch_card_l1662_166269


namespace NUMINAMATH_CALUDE_complex_locus_ellipse_l1662_166265

/-- For a complex number z with |z| = 3, the locus of points traced by z + 2/z forms an ellipse -/
theorem complex_locus_ellipse (z : ℂ) (h : Complex.abs z = 3) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
  ∀ (w : ℂ), w = z + 2 / z → (w.re / a) ^ 2 + (w.im / b) ^ 2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_locus_ellipse_l1662_166265


namespace NUMINAMATH_CALUDE_cos_is_semi_odd_tan_is_semi_odd_l1662_166256

-- Definition of a semi-odd function
def is_semi_odd (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = -f (2*a - x)

-- Statement for cos(x+1)
theorem cos_is_semi_odd :
  is_semi_odd (λ x => Real.cos (x + 1)) :=
sorry

-- Statement for tan(x)
theorem tan_is_semi_odd :
  is_semi_odd Real.tan :=
sorry

end NUMINAMATH_CALUDE_cos_is_semi_odd_tan_is_semi_odd_l1662_166256


namespace NUMINAMATH_CALUDE_relay_team_orders_l1662_166203

/-- The number of permutations of n elements -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The number of different orders for a relay team of 6 runners,
    where one specific runner is fixed to run the last lap -/
def relay_orders : ℕ := factorial 5

theorem relay_team_orders :
  relay_orders = 120 := by sorry

end NUMINAMATH_CALUDE_relay_team_orders_l1662_166203


namespace NUMINAMATH_CALUDE_inequality_proof_l1662_166284

theorem inequality_proof (a b α β θ : ℝ) (ha : a > 0) (hb : b > 0) (hα : abs α > a) :
  (α * β - Real.sqrt (a^2 * β^2 + b^2 * α^2 - a^2 * b^2)) / (α^2 - a^2) ≤ 
  (β + b * Real.sin θ) / (α + a * Real.cos θ) ∧
  (β + b * Real.sin θ) / (α + a * Real.cos θ) ≤ 
  (α * β + Real.sqrt (a^2 * β^2 + b^2 * α^2 - a^2 * b^2)) / (α^2 - a^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1662_166284


namespace NUMINAMATH_CALUDE_parabola_translation_theorem_l1662_166283

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = x^2 -/
def original_parabola : Parabola := { a := 1, b := 0, c := 0 }

/-- Translates a parabola vertically by a given amount -/
def translate_vertical (p : Parabola) (amount : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + amount }

/-- Translates a parabola horizontally by a given amount -/
def translate_horizontal (p : Parabola) (amount : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * amount + p.b, c := p.a * amount^2 - p.b * amount + p.c }

/-- The resulting parabola after translations -/
def resulting_parabola : Parabola :=
  translate_horizontal (translate_vertical original_parabola 3) 5

theorem parabola_translation_theorem :
  resulting_parabola = { a := 1, b := -10, c := 28 } :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_theorem_l1662_166283


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1662_166237

theorem complex_modulus_problem (z : ℂ) (h : z * (2 + Complex.I) = 5 * Complex.I) :
  Complex.abs (z - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1662_166237


namespace NUMINAMATH_CALUDE_pi_minus_2023_power_0_minus_one_third_power_neg_2_l1662_166200

theorem pi_minus_2023_power_0_minus_one_third_power_neg_2 :
  (π - 2023) ^ (0 : ℝ) - (1 / 3 : ℝ) ^ (-2 : ℝ) = -8 := by sorry

end NUMINAMATH_CALUDE_pi_minus_2023_power_0_minus_one_third_power_neg_2_l1662_166200


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1662_166239

theorem at_least_one_not_less_than_two (a b : ℕ) (h : a + b ≥ 3) : a ≥ 2 ∨ b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1662_166239


namespace NUMINAMATH_CALUDE_rectangle_area_l1662_166216

/-- Given a rectangle where the length is 3 times the width and the width is 5 inches,
    prove that its area is 75 square inches. -/
theorem rectangle_area (width : ℝ) (length : ℝ) (area : ℝ) : 
  width = 5 → length = 3 * width → area = length * width → area = 75 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_l1662_166216


namespace NUMINAMATH_CALUDE_range_of_m_l1662_166278

/-- For the equation m/(x-2) = 3 with positive solutions for x, 
    the range of m is {m ∈ ℝ | m > -6 and m ≠ 0} -/
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ m / (x - 2) = 3) ↔ m > -6 ∧ m ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1662_166278


namespace NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1662_166261

theorem ellipse_equation_equivalence (x y : ℝ) :
  (Real.sqrt (x^2 + (y - 3)^2) + Real.sqrt (x^2 + (y + 3)^2) = 10) ↔
  (x^2 / 25 + y^2 / 16 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_equivalence_l1662_166261


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1662_166285

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)
  product_345 : a 3 * a 4 * a 5 = 3
  product_678 : a 6 * a 7 * a 8 = 24

/-- The theorem statement -/
theorem geometric_sequence_property (seq : GeometricSequence) :
  seq.a 9 * seq.a 10 * seq.a 11 = 192 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1662_166285


namespace NUMINAMATH_CALUDE_large_cube_single_color_face_l1662_166290

/-- Represents a small cube with colored faces -/
structure SmallCube :=
  (white_faces : Fin 2)
  (blue_faces : Fin 2)
  (red_faces : Fin 2)

/-- Represents the large cube assembled from small cubes -/
def LargeCube := Fin 10 → Fin 10 → Fin 10 → SmallCube

/-- Predicate to check if two adjacent small cubes have matching colors -/
def matching_colors (c1 c2 : SmallCube) : Prop := sorry

/-- Predicate to check if a face of the large cube is a single color -/
def single_color_face (cube : LargeCube) : Prop := sorry

/-- Main theorem: The large cube has at least one face that is a single color -/
theorem large_cube_single_color_face 
  (cube : LargeCube)
  (h_matching : ∀ i j k i' j' k', 
    (i = i' ∧ j = j' ∧ (k + 1 = k' ∨ k = k' + 1)) ∨
    (i = i' ∧ (j + 1 = j' ∨ j = j' + 1) ∧ k = k') ∨
    ((i + 1 = i' ∨ i = i' + 1) ∧ j = j' ∧ k = k') →
    matching_colors (cube i j k) (cube i' j' k')) :
  single_color_face cube :=
sorry

end NUMINAMATH_CALUDE_large_cube_single_color_face_l1662_166290


namespace NUMINAMATH_CALUDE_cubic_minus_four_xy_squared_factorization_l1662_166240

theorem cubic_minus_four_xy_squared_factorization (x y : ℝ) :
  x^3 - 4*x*y^2 = x*(x+2*y)*(x-2*y) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_four_xy_squared_factorization_l1662_166240


namespace NUMINAMATH_CALUDE_range_of_square_root_set_l1662_166227

theorem range_of_square_root_set (A : Set ℝ) (a : ℝ) :
  (A.Nonempty) →
  (A = {x : ℝ | x^2 = a}) →
  (∃ (y : ℝ), ∀ (x : ℝ), x ∈ A ↔ y ≤ x ∧ x^2 = a) :=
by sorry

end NUMINAMATH_CALUDE_range_of_square_root_set_l1662_166227


namespace NUMINAMATH_CALUDE_largest_gcd_sum_1008_l1662_166204

theorem largest_gcd_sum_1008 :
  ∃ (a b : ℕ+), a + b = 1008 ∧ 
  ∀ (c d : ℕ+), c + d = 1008 → Nat.gcd a b ≥ Nat.gcd c d ∧
  Nat.gcd a b = 504 :=
sorry

end NUMINAMATH_CALUDE_largest_gcd_sum_1008_l1662_166204


namespace NUMINAMATH_CALUDE_quadratic_roots_exist_l1662_166209

theorem quadratic_roots_exist (a b c : ℝ) (h : a * c < 0) : 
  ∃ x : ℝ, a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_exist_l1662_166209


namespace NUMINAMATH_CALUDE_integer_sum_problem_l1662_166225

theorem integer_sum_problem (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 180) : x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l1662_166225


namespace NUMINAMATH_CALUDE_right_triangle_leg_length_l1662_166224

theorem right_triangle_leg_length 
  (A B C : ℝ × ℝ) 
  (is_right_triangle : (A.1 - B.1) * (A.1 - C.1) + (A.2 - B.2) * (A.2 - C.2) = 0) 
  (leg_length : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 1) 
  (hypotenuse_length : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = Real.sqrt 5) :
  Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_leg_length_l1662_166224


namespace NUMINAMATH_CALUDE_smallest_winning_N_for_berta_l1662_166294

/-- A game where two players take turns removing marbles from a table. -/
structure MarbleGame where
  initialMarbles : ℕ
  currentMarbles : ℕ
  playerTurn : Bool  -- True for Anna, False for Berta

/-- The rules for removing marbles in a turn -/
def validMove (game : MarbleGame) (k : ℕ) : Prop :=
  k ≥ 1 ∧
  ((k % 2 = 0 ∧ k ≤ game.currentMarbles / 2) ∨
   (k % 2 = 1 ∧ game.currentMarbles / 2 ≤ k ∧ k ≤ game.currentMarbles))

/-- The condition for a winning position -/
def isWinningPosition (n : ℕ) : Prop :=
  ∃ m : ℕ, m ≥ 2 ∧ n = 2^m - 2

/-- The theorem to prove -/
theorem smallest_winning_N_for_berta :
  ∃ N : ℕ,
    N ≥ 100000 ∧
    isWinningPosition N ∧
    (∀ M : ℕ, M ≥ 100000 ∧ M < N → ¬isWinningPosition M) :=
  sorry

end NUMINAMATH_CALUDE_smallest_winning_N_for_berta_l1662_166294


namespace NUMINAMATH_CALUDE_race_probability_l1662_166246

theorem race_probability (total_cars : ℕ) (prob_X prob_Y prob_total : ℝ) : 
  total_cars = 12 →
  prob_X = 1/6 →
  prob_Y = 1/10 →
  prob_total = 0.39166666666666666 →
  prob_total = prob_X + prob_Y + (0.125 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_race_probability_l1662_166246


namespace NUMINAMATH_CALUDE_angle_aoc_in_regular_octagon_l1662_166271

/-- A regular octagon -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ
  is_regular : sorry

/-- The center of a regular octagon -/
def center (octagon : RegularOctagon) : ℝ × ℝ := sorry

/-- Angle between two points and the center -/
def angle_with_center (octagon : RegularOctagon) (p1 p2 : Fin 8) : ℝ := sorry

theorem angle_aoc_in_regular_octagon (octagon : RegularOctagon) :
  angle_with_center octagon 0 2 = 45 := by sorry

end NUMINAMATH_CALUDE_angle_aoc_in_regular_octagon_l1662_166271


namespace NUMINAMATH_CALUDE_identical_roots_quadratic_l1662_166288

/-- If the quadratic equation 3x^2 - 6x + k = 0 has two identical real roots, then k = 3 -/
theorem identical_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, (3 * x^2 - 6 * x + k = 0) ∧ 
   (∀ y : ℝ, 3 * y^2 - 6 * y + k = 0 → y = x)) → 
  k = 3 := by sorry

end NUMINAMATH_CALUDE_identical_roots_quadratic_l1662_166288


namespace NUMINAMATH_CALUDE_reams_needed_l1662_166231

-- Define the constants
def stories_per_week : ℕ := 3
def pages_per_story : ℕ := 50
def novel_pages_per_year : ℕ := 1200
def pages_per_sheet : ℕ := 2
def sheets_per_ream : ℕ := 500
def weeks_in_year : ℕ := 52
def weeks_to_calculate : ℕ := 12

-- Theorem to prove
theorem reams_needed : 
  (stories_per_week * pages_per_story * weeks_in_year + novel_pages_per_year) / pages_per_sheet / sheets_per_ream = 9 := by
  sorry


end NUMINAMATH_CALUDE_reams_needed_l1662_166231


namespace NUMINAMATH_CALUDE_yellow_block_weight_l1662_166254

theorem yellow_block_weight (green_weight : ℝ) (weight_difference : ℝ) 
  (h1 : green_weight = 0.4)
  (h2 : weight_difference = 0.2) : 
  green_weight + weight_difference = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_yellow_block_weight_l1662_166254


namespace NUMINAMATH_CALUDE_circle_center_coordinates_sum_l1662_166299

/-- Given a circle with equation x² + y² = -4x + 6y - 12, 
    the sum of the x and y coordinates of its center is 1. -/
theorem circle_center_coordinates_sum : 
  ∀ (x y : ℝ), x^2 + y^2 = -4*x + 6*y - 12 → 
  ∃ (h k : ℝ), (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = 1) ∧ h + k = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinates_sum_l1662_166299


namespace NUMINAMATH_CALUDE_trigonometric_identities_l1662_166262

theorem trigonometric_identities :
  (∃ (x y : ℝ), 
    x = Real.sin (-1395 * π / 180) * Real.cos (1140 * π / 180) + 
        Real.cos (-1020 * π / 180) * Real.sin (750 * π / 180) ∧
    y = Real.sin (-11 * π / 6) + Real.cos (3 * π / 4) * Real.tan (4 * π) ∧
    x = (Real.sqrt 2 + 1) / 4 ∧
    y = 1 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l1662_166262


namespace NUMINAMATH_CALUDE_problem_statement_l1662_166273

theorem problem_statement : (-1 : ℤ) ^ 53 + 2 ^ (4^3 + 3^2 - 7^2) = 16777215 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1662_166273


namespace NUMINAMATH_CALUDE_max_horses_for_25_and_7_l1662_166207

/-- Given a total number of horses and a minimum number of races to determine the top 3 fastest,
    calculate the maximum number of horses that can race together at a time. -/
def max_horses_per_race (total_horses : ℕ) (min_races : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 25 horses and 7 minimum races, the maximum number of horses
    that can race together is 5. -/
theorem max_horses_for_25_and_7 :
  max_horses_per_race 25 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_max_horses_for_25_and_7_l1662_166207


namespace NUMINAMATH_CALUDE_tank_length_is_25_l1662_166291

/-- Given a tank with specific dimensions and plastering costs, prove its length is 25 meters -/
theorem tank_length_is_25 (width : ℝ) (depth : ℝ) (plaster_cost_per_sqm : ℝ) (total_plaster_cost : ℝ) :
  width = 12 →
  depth = 6 →
  plaster_cost_per_sqm = 0.45 →
  total_plaster_cost = 334.8 →
  (∃ (length : ℝ), 
    total_plaster_cost / plaster_cost_per_sqm = 2 * (length * depth) + 2 * (width * depth) + (length * width) ∧
    length = 25) := by
  sorry

end NUMINAMATH_CALUDE_tank_length_is_25_l1662_166291


namespace NUMINAMATH_CALUDE_journey_speed_proof_l1662_166245

/-- Proves that given a journey of 108 miles completed in 90 minutes, 
    where the average speed for the first 30 minutes was 65 mph and 
    for the second 30 minutes was 70 mph, the average speed for the 
    last 30 minutes was 81 mph. -/
theorem journey_speed_proof 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (speed_first_segment : ℝ) 
  (speed_second_segment : ℝ) 
  (h1 : total_distance = 108) 
  (h2 : total_time = 90 / 60) 
  (h3 : speed_first_segment = 65) 
  (h4 : speed_second_segment = 70) : 
  ∃ (speed_last_segment : ℝ), 
    speed_last_segment = 81 ∧ 
    (speed_first_segment + speed_second_segment + speed_last_segment) / 3 = 
      total_distance / total_time := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_proof_l1662_166245


namespace NUMINAMATH_CALUDE_daniels_age_l1662_166266

theorem daniels_age (ishaan_age : ℕ) (years_until_4x : ℕ) (daniel_age : ℕ) : 
  ishaan_age = 6 →
  years_until_4x = 15 →
  daniel_age + years_until_4x = 4 * (ishaan_age + years_until_4x) →
  daniel_age = 69 := by
sorry

end NUMINAMATH_CALUDE_daniels_age_l1662_166266


namespace NUMINAMATH_CALUDE_max_value_x_1_minus_3x_l1662_166205

theorem max_value_x_1_minus_3x (x : ℝ) (h : 0 < x ∧ x < 1/3) :
  ∃ (max : ℝ), max = 1/12 ∧ ∀ y, 0 < y ∧ y < 1/3 → x * (1 - 3*x) ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_1_minus_3x_l1662_166205


namespace NUMINAMATH_CALUDE_water_tower_shortage_l1662_166222

theorem water_tower_shortage : 
  let tower_capacity : ℝ := 2700
  let first_neighborhood : ℝ := 300
  let second_neighborhood : ℝ := 2 * first_neighborhood
  let third_neighborhood : ℝ := second_neighborhood + 100
  let fourth_neighborhood : ℝ := 3 * first_neighborhood
  let fifth_neighborhood : ℝ := third_neighborhood / 2
  let leakage_loss : ℝ := 50
  let first_increased : ℝ := first_neighborhood * 1.1
  let third_increased : ℝ := third_neighborhood * 1.1
  let second_decreased : ℝ := second_neighborhood * 0.95
  let fifth_decreased : ℝ := fifth_neighborhood * 0.95
  let total_consumption : ℝ := first_increased + second_decreased + third_increased + fourth_neighborhood + fifth_decreased + leakage_loss
  total_consumption - tower_capacity = 252.5 :=
by sorry

end NUMINAMATH_CALUDE_water_tower_shortage_l1662_166222


namespace NUMINAMATH_CALUDE_a_5_equals_10_l1662_166296

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

theorem a_5_equals_10 (a : ℕ → ℕ) (h1 : arithmetic_sequence a) (h2 : a 1 = 2) :
  a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_a_5_equals_10_l1662_166296


namespace NUMINAMATH_CALUDE_solve_equation_l1662_166219

theorem solve_equation (x : ℚ) : 3 * x + 15 = (1 / 3) * (4 * x + 28) → x = -17 / 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1662_166219


namespace NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l1662_166274

/-- Proves that the manufacturing cost of a shoe is 200, given the transportation cost,
    selling price, and profit margin. -/
theorem manufacturing_cost_of_shoe (transportation_cost : ℚ) (selling_price : ℚ) 
  (profit_margin : ℚ) (h1 : transportation_cost = 500 / 100)
  (h2 : selling_price = 246) (h3 : profit_margin = 20 / 100) :
  ∃ (manufacturing_cost : ℚ), 
    selling_price = (manufacturing_cost + transportation_cost) * (1 + profit_margin) ∧
    manufacturing_cost = 200 :=
by sorry

end NUMINAMATH_CALUDE_manufacturing_cost_of_shoe_l1662_166274


namespace NUMINAMATH_CALUDE_sylvie_turtle_weight_l1662_166250

/-- The weight of turtles Sylvie has, given the feeding conditions -/
theorem sylvie_turtle_weight :
  let food_per_half_pound : ℚ := 1 -- 1 ounce of food per 1/2 pound of body weight
  let ounces_per_jar : ℚ := 15 -- Each jar contains 15 ounces
  let cost_per_jar : ℚ := 2 -- Each jar costs $2
  let total_cost : ℚ := 8 -- It costs $8 to feed the turtles
  
  (total_cost / cost_per_jar) * ounces_per_jar / food_per_half_pound / 2 = 30 := by
  sorry

end NUMINAMATH_CALUDE_sylvie_turtle_weight_l1662_166250


namespace NUMINAMATH_CALUDE_f_difference_l1662_166259

def f (n : ℕ) : ℚ :=
  (Finset.range (n + 1)).sum (fun i => 1 / ((n + 1 + i) : ℚ))

theorem f_difference (n : ℕ) : f (n + 1) - f n = 1 / (2 * n + 3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l1662_166259


namespace NUMINAMATH_CALUDE_picture_on_wall_l1662_166297

theorem picture_on_wall (wall_width picture_width : ℝ) 
  (hw : wall_width = 26) (hp : picture_width = 4) :
  let distance := (wall_width - picture_width) / 2
  distance = 11 := by
  sorry

end NUMINAMATH_CALUDE_picture_on_wall_l1662_166297


namespace NUMINAMATH_CALUDE_jeans_cost_thirty_l1662_166272

/-- The price of socks in dollars -/
def socks_price : ℕ := 5

/-- The price difference between t-shirt and socks in dollars -/
def tshirt_socks_diff : ℕ := 10

/-- The price of a t-shirt in dollars -/
def tshirt_price : ℕ := socks_price + tshirt_socks_diff

/-- The price of jeans in dollars -/
def jeans_price : ℕ := 2 * tshirt_price

theorem jeans_cost_thirty : jeans_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_jeans_cost_thirty_l1662_166272


namespace NUMINAMATH_CALUDE_interest_equality_problem_l1662_166281

theorem interest_equality_problem (total : ℝ) (first_part : ℝ) (second_part : ℝ)
  (h1 : total = 2717)
  (h2 : total = first_part + second_part)
  (h3 : first_part * (3/100) * 8 = second_part * (5/100) * 3) :
  second_part = 2449 := by
  sorry

end NUMINAMATH_CALUDE_interest_equality_problem_l1662_166281


namespace NUMINAMATH_CALUDE_cutlery_theorem_l1662_166295

/-- Calculates the total number of cutlery pieces after purchases -/
def totalCutlery (initialKnives : ℕ) : ℕ :=
  let initialTeaspoons := 2 * initialKnives
  let additionalKnives := initialKnives / 3
  let additionalTeaspoons := (2 * initialTeaspoons) / 3
  let totalKnives := initialKnives + additionalKnives
  let totalTeaspoons := initialTeaspoons + additionalTeaspoons
  totalKnives + totalTeaspoons

/-- Theorem stating that given 24 initial knives, the total cutlery after purchases is 112 -/
theorem cutlery_theorem : totalCutlery 24 = 112 := by
  sorry

end NUMINAMATH_CALUDE_cutlery_theorem_l1662_166295


namespace NUMINAMATH_CALUDE_m_less_than_2_necessary_not_sufficient_l1662_166253

-- Define the quadratic function
def f (m x : ℝ) := x^2 + m*x + 1

-- Define the condition for the solution set to be ℝ
def solution_is_real (m : ℝ) : Prop :=
  ∀ x, f m x > 0

-- Define the necessary and sufficient condition
def necessary_and_sufficient (m : ℝ) : Prop :=
  m^2 - 4 < 0

-- Theorem: m < 2 is a necessary but not sufficient condition
theorem m_less_than_2_necessary_not_sufficient :
  (∀ m, solution_is_real m → m < 2) ∧
  ¬(∀ m, m < 2 → solution_is_real m) :=
sorry

end NUMINAMATH_CALUDE_m_less_than_2_necessary_not_sufficient_l1662_166253


namespace NUMINAMATH_CALUDE_max_profit_difference_l1662_166277

def total_records : ℕ := 300

def sammy_offer : ℕ → ℚ := λ n => 4 * n

def bryan_offer : ℕ → ℚ := λ n => 6 * (2/3 * n) + 1 * (1/3 * n)

def christine_offer : ℕ → ℚ := λ n => 10 * 30 + 3 * (n - 30)

theorem max_profit_difference (n : ℕ) (h : n = total_records) : 
  max (abs (sammy_offer n - bryan_offer n))
      (max (abs (sammy_offer n - christine_offer n))
           (abs (bryan_offer n - christine_offer n)))
  = 190 :=
sorry

end NUMINAMATH_CALUDE_max_profit_difference_l1662_166277


namespace NUMINAMATH_CALUDE_six_digit_number_property_l1662_166276

/-- Represents a six-digit number in the form 1ABCDE -/
def SixDigitNumber (a b c d e : Nat) : Nat :=
  100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e

theorem six_digit_number_property 
  (a b c d e : Nat) 
  (h1 : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) 
  (h2 : SixDigitNumber a b c d e * 3 = SixDigitNumber b c d e a) : 
  a + b + c + d + e = 26 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_number_property_l1662_166276


namespace NUMINAMATH_CALUDE_probability_multiple_of_five_l1662_166287

theorem probability_multiple_of_five (total_pages : ℕ) (h : total_pages = 300) :
  (Finset.filter (fun n => n % 5 = 0) (Finset.range total_pages)).card / total_pages = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_of_five_l1662_166287


namespace NUMINAMATH_CALUDE_score_theorem_l1662_166223

/-- Represents the bags from which balls are drawn -/
inductive Bag
| A
| B

/-- Represents the color of the balls -/
inductive Color
| Black
| White
| Red

/-- Represents the score obtained from drawing a ball -/
def score (bag : Bag) (color : Color) : ℕ :=
  match bag, color with
  | Bag.A, Color.Black => 2
  | Bag.B, Color.Black => 1
  | _, _ => 0

/-- The probability of drawing a black ball from bag B -/
def probBlackB : ℝ := 0.8

/-- The probability of getting a total score of 1 -/
def probScoreOne : ℝ := 0.24

/-- The expected value of the total score -/
def expectedScore : ℝ := 1.94

/-- Theorem stating the expected value of the total score and comparing probabilities -/
theorem score_theorem :
  ∃ (probBlackA : ℝ),
    0 ≤ probBlackA ∧ probBlackA ≤ 1 ∧
    (let pA := probBlackA * (1 - probBlackB) + (1 - probBlackA) * probBlackB
     let pB := probBlackB * probBlackB
     pB > pA) ∧
    expectedScore = 1.94 := by
  sorry

end NUMINAMATH_CALUDE_score_theorem_l1662_166223
