import Mathlib

namespace number_difference_l3003_300381

theorem number_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 12) : |x - y| = 3/2 := by
  sorry

end number_difference_l3003_300381


namespace inequality_proof_l3003_300344

theorem inequality_proof :
  (∀ m n p : ℝ, m > n ∧ n > 0 ∧ p > 0 → n / m < (n + p) / (m + p)) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b →
    c / (a + b) + a / (b + c) + b / (c + a) < 2) :=
by sorry

end inequality_proof_l3003_300344


namespace min_sum_eccentricities_l3003_300374

theorem min_sum_eccentricities (e₁ e₂ : ℝ) (h₁ : e₁ > 0) (h₂ : e₂ > 0) 
  (h : 1 / (e₁^2) + 1 / (e₂^2) = 1) : 
  e₁ + e₂ ≥ 2 * Real.sqrt 2 := by
  sorry

end min_sum_eccentricities_l3003_300374


namespace chocolate_chip_calculation_l3003_300335

/-- The number of cups of chocolate chips needed for one recipe -/
def chips_per_recipe : ℝ := 3.5

/-- The number of recipes to be made -/
def num_recipes : ℕ := 37

/-- The total number of cups of chocolate chips needed -/
def total_chips : ℝ := chips_per_recipe * num_recipes

theorem chocolate_chip_calculation : total_chips = 129.5 := by
  sorry

end chocolate_chip_calculation_l3003_300335


namespace max_value_of_f_l3003_300300

open Real

noncomputable def f (θ : ℝ) : ℝ := tan (θ / 2) * (1 - sin θ)

theorem max_value_of_f :
  ∃ (θ_max : ℝ), 
    -π/2 < θ_max ∧ θ_max < π/2 ∧
    θ_max = 2 * arctan ((-2 + Real.sqrt 7) / 3) ∧
    ∀ (θ : ℝ), -π/2 < θ ∧ θ < π/2 → f θ ≤ f θ_max :=
by sorry

end max_value_of_f_l3003_300300


namespace ellipse_eccentricity_theorem_l3003_300350

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the foci of an ellipse -/
structure Foci (a b : ℝ) where
  left : Point
  right : Point
  h_ellipse : Ellipse a b

/-- Represents a triangle formed by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Defines an equilateral triangle -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Defines a line perpendicular to the x-axis passing through a point -/
def perpendicular_to_x_axis (p : Point) (A B : Point) : Prop := sorry

/-- Defines points on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse a b) : Prop := sorry

/-- Defines the eccentricity of an ellipse -/
def eccentricity (e : Ellipse a b) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_eccentricity_theorem 
  (a b : ℝ) 
  (e : Ellipse a b) 
  (f : Foci a b) 
  (A B : Point) 
  (t : Triangle) :
  perpendicular_to_x_axis f.right A B →
  on_ellipse A e →
  on_ellipse B e →
  t = Triangle.mk A B f.left →
  is_equilateral t →
  eccentricity e = Real.sqrt 3 / 3 := by sorry

end ellipse_eccentricity_theorem_l3003_300350


namespace katie_baked_18_cupcakes_l3003_300389

/-- The number of cupcakes Todd ate -/
def todd_ate : ℕ := 8

/-- The number of packages Katie could make after Todd ate some cupcakes -/
def num_packages : ℕ := 5

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 2

/-- The initial number of cupcakes Katie baked -/
def initial_cupcakes : ℕ := todd_ate + num_packages * cupcakes_per_package

theorem katie_baked_18_cupcakes : initial_cupcakes = 18 := by
  sorry

end katie_baked_18_cupcakes_l3003_300389


namespace impossible_all_black_l3003_300358

/-- Represents the color of a square on the board -/
inductive Color
| White
| Black

/-- Represents a 4x4 board -/
def Board := Fin 4 → Fin 4 → Color

/-- Represents a 1x3 rectangle on the board -/
structure Rectangle :=
  (row : Fin 4)
  (col : Fin 4)
  (horizontal : Bool)

/-- Initial state of the board where all squares are white -/
def initialBoard : Board :=
  λ _ _ => Color.White

/-- Applies a move to the board by flipping colors in a 1x3 rectangle -/
def applyMove (b : Board) (r : Rectangle) : Board :=
  sorry

/-- Checks if all squares on the board are black -/
def allBlack (b : Board) : Prop :=
  ∀ i j, b i j = Color.Black

/-- Theorem stating that it's impossible to make all squares black -/
theorem impossible_all_black :
  ¬ ∃ (moves : List Rectangle), allBlack (moves.foldl applyMove initialBoard) :=
sorry

end impossible_all_black_l3003_300358


namespace san_diego_zoo_ticket_cost_l3003_300376

/-- Calculates the total cost of zoo tickets for a family -/
def total_cost_zoo_tickets (family_size : ℕ) (adult_price : ℕ) (child_price : ℕ) (adult_tickets : ℕ) : ℕ :=
  let child_tickets := family_size - adult_tickets
  adult_price * adult_tickets + child_price * child_tickets

/-- Theorem: The total cost of zoo tickets for a family of 7 with 4 adult tickets is $126 -/
theorem san_diego_zoo_ticket_cost :
  total_cost_zoo_tickets 7 21 14 4 = 126 := by
  sorry

end san_diego_zoo_ticket_cost_l3003_300376


namespace correct_operation_order_l3003_300384

-- Define operation levels
inductive OperationLevel
| FirstLevel
| SecondLevel

-- Define operations
inductive Operation
| Multiplication
| Division
| Subtraction

-- Define the level of each operation
def operationLevel : Operation → OperationLevel
| Operation.Multiplication => OperationLevel.SecondLevel
| Operation.Division => OperationLevel.SecondLevel
| Operation.Subtraction => OperationLevel.FirstLevel

-- Define the rule for operation order
def shouldPerformBefore (op1 op2 : Operation) : Prop :=
  operationLevel op1 = OperationLevel.SecondLevel ∧ 
  operationLevel op2 = OperationLevel.FirstLevel

-- Define the expression
def expression : List Operation :=
  [Operation.Multiplication, Operation.Subtraction, Operation.Division]

-- Theorem to prove
theorem correct_operation_order :
  shouldPerformBefore Operation.Multiplication Operation.Subtraction ∧
  shouldPerformBefore Operation.Division Operation.Subtraction ∧
  (¬ shouldPerformBefore Operation.Multiplication Operation.Division ∨
   ¬ shouldPerformBefore Operation.Division Operation.Multiplication) :=
by sorry

end correct_operation_order_l3003_300384


namespace triangle_less_than_answer_l3003_300340

theorem triangle_less_than_answer (triangle : ℝ) (answer : ℝ) 
  (h : 8.5 + triangle = 5.6 + answer) : triangle < answer := by
  sorry

end triangle_less_than_answer_l3003_300340


namespace complex_ratio_theorem_l3003_300346

theorem complex_ratio_theorem (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 3) 
  (h₂ : Complex.abs z₂ = 5) 
  (h₃ : Complex.abs (z₁ - z₂) = 7) : 
  z₁ / z₂ = (3 / 5 : ℂ) * (-1 / 2 + Complex.I * Real.sqrt 3 / 2) ∨
  z₁ / z₂ = (3 / 5 : ℂ) * (-1 / 2 - Complex.I * Real.sqrt 3 / 2) :=
sorry

end complex_ratio_theorem_l3003_300346


namespace focus_coordinates_l3003_300394

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 9 = 1 ∧ a > 0

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  y = 3 * x

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop :=
  y = 2 * a * x^2

-- State the theorem
theorem focus_coordinates (a : ℝ) :
  (∃ x y : ℝ, hyperbola a x y ∧ asymptote x y) →
  (∃ x y : ℝ, parabola a x y ∧ x = 0 ∧ y = 1/8) :=
sorry

end focus_coordinates_l3003_300394


namespace jills_peaches_l3003_300377

theorem jills_peaches (steven_peaches : ℕ) (jake_fewer : ℕ) (jake_more : ℕ) 
  (h1 : steven_peaches = 14)
  (h2 : jake_fewer = 6)
  (h3 : jake_more = 3)
  : steven_peaches - jake_fewer - jake_more = 5 := by
  sorry

end jills_peaches_l3003_300377


namespace equation_solution_l3003_300322

theorem equation_solution :
  ∃ x : ℝ, (5 + 3.4 * x = 2.8 * x - 35) ∧ (x = -200 / 3) := by
  sorry

end equation_solution_l3003_300322


namespace largest_divisor_of_n_squared_divisible_by_72_l3003_300341

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (hn : n > 0) 
  (h_divisible : 72 ∣ n^2) : 
  ∀ m : ℕ, m ∣ n → m ≤ 12 ∧ 12 ∣ n :=
by sorry

end largest_divisor_of_n_squared_divisible_by_72_l3003_300341


namespace smallest_whole_number_above_sum_l3003_300383

theorem smallest_whole_number_above_sum : ⌈(10/3 : ℚ) + (17/4 : ℚ) + (26/5 : ℚ) + (37/6 : ℚ)⌉ = 19 := by
  sorry

end smallest_whole_number_above_sum_l3003_300383


namespace sum_of_reciprocals_l3003_300339

theorem sum_of_reciprocals (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : 1/x - 1/y = -3) : 
  x + y = 5/4 := by
  sorry

end sum_of_reciprocals_l3003_300339


namespace b_completes_job_in_20_days_l3003_300318

/-- The number of days it takes A to complete the job -/
def days_A : ℝ := 15

/-- The number of days A and B work together -/
def days_together : ℝ := 5

/-- The fraction of the job left after A and B work together -/
def fraction_left : ℝ := 0.41666666666666663

/-- The number of days it takes B to complete the job -/
def days_B : ℝ := 20

theorem b_completes_job_in_20_days :
  (days_together * (1 / days_A + 1 / days_B) = 1 - fraction_left) ∧
  (days_B = 20) := by sorry

end b_completes_job_in_20_days_l3003_300318


namespace candy_calculation_l3003_300385

/-- Calculates the number of candy pieces Faye's sister gave her --/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

theorem candy_calculation (initial eaten final : ℕ) 
  (h1 : initial ≥ eaten) 
  (h2 : final ≥ initial - eaten) : 
  candy_from_sister initial eaten final = final - (initial - eaten) :=
by
  sorry

#eval candy_from_sister 47 25 62  -- Should output 40

end candy_calculation_l3003_300385


namespace expression_value_l3003_300330

theorem expression_value : 
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 5000 := by
  sorry

end expression_value_l3003_300330


namespace at_least_one_good_certain_l3003_300307

def total_products : ℕ := 12
def good_products : ℕ := 10
def defective_products : ℕ := 2
def picked_products : ℕ := 3

theorem at_least_one_good_certain :
  Fintype.card {s : Finset (Fin total_products) // s.card = picked_products ∧ ∃ x ∈ s, x.val < good_products} =
  Fintype.card {s : Finset (Fin total_products) // s.card = picked_products} :=
sorry

end at_least_one_good_certain_l3003_300307


namespace sam_recycling_cans_l3003_300362

/-- The number of bags Sam filled on Saturday -/
def saturday_bags : ℕ := 3

/-- The number of bags Sam filled on Sunday -/
def sunday_bags : ℕ := 4

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 9

/-- The total number of cans Sam picked up -/
def total_cans : ℕ := (saturday_bags + sunday_bags) * cans_per_bag

theorem sam_recycling_cans : total_cans = 63 := by
  sorry

end sam_recycling_cans_l3003_300362


namespace sin_period_l3003_300326

/-- The period of y = sin(4x + π) is π/2 -/
theorem sin_period (x : ℝ) : 
  (∀ y, y = Real.sin (4 * x + π)) → 
  (∃ p, p > 0 ∧ ∀ x, Real.sin (4 * x + π) = Real.sin (4 * (x + p) + π) ∧ p = π / 2) :=
by sorry

end sin_period_l3003_300326


namespace arithmetic_sequence_2011_unique_term_2011_l3003_300342

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmeticSequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- Theorem stating that the 671st term of the sequence is 2011 -/
theorem arithmetic_sequence_2011 : arithmeticSequence 671 = 2011 := by sorry

/-- Theorem proving that 671 is the unique natural number n for which a_n = 2011 -/
theorem unique_term_2011 : ∀ n : ℕ, arithmeticSequence n = 2011 ↔ n = 671 := by sorry

end arithmetic_sequence_2011_unique_term_2011_l3003_300342


namespace triangle_inequality_new_magnitude_min_magnitude_on_line_max_magnitude_on_circle_l3003_300361

-- Define the new magnitude
def new_magnitude (x y : ℝ) : ℝ := |x + y| + |x - y|

-- Theorem for proposition (1)
theorem triangle_inequality_new_magnitude (x₁ y₁ x₂ y₂ : ℝ) :
  new_magnitude (x₁ - x₂) (y₁ - y₂) ≤ new_magnitude x₁ y₁ + new_magnitude x₂ y₂ := by
  sorry

-- Theorem for proposition (2)
theorem min_magnitude_on_line :
  ∃ (t : ℝ), ∀ (s : ℝ), new_magnitude t (t - 1) ≤ new_magnitude s (s - 1) ∧ new_magnitude t (t - 1) = 1 := by
  sorry

-- Theorem for proposition (3)
theorem max_magnitude_on_circle :
  ∃ (x y : ℝ), x^2 + y^2 = 1 ∧ new_magnitude x y = 2 ∧ 
  ∀ (a b : ℝ), a^2 + b^2 = 1 → new_magnitude a b ≤ 2 := by
  sorry

end triangle_inequality_new_magnitude_min_magnitude_on_line_max_magnitude_on_circle_l3003_300361


namespace at_least_one_positive_negation_l3003_300308

theorem at_least_one_positive_negation (a b c : ℝ) :
  (¬ (a > 0 ∨ b > 0 ∨ c > 0)) ↔ (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0) := by
  sorry

end at_least_one_positive_negation_l3003_300308


namespace remainder_of_9876543210_div_101_l3003_300316

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 := by
  sorry

end remainder_of_9876543210_div_101_l3003_300316


namespace probability_five_or_joker_l3003_300351

/-- A deck of cards with jokers -/
structure DeckWithJokers where
  standardCards : ℕ
  jokers : ℕ
  totalCards : ℕ
  total_is_sum : totalCards = standardCards + jokers

/-- The probability of drawing a specific card or a joker -/
def drawProbability (d : DeckWithJokers) (specificCards : ℕ) : ℚ :=
  (specificCards + d.jokers : ℚ) / d.totalCards

/-- The deck described in the problem -/
def problemDeck : DeckWithJokers where
  standardCards := 52
  jokers := 2
  totalCards := 54
  total_is_sum := by rfl

theorem probability_five_or_joker :
  drawProbability problemDeck 4 = 1/9 := by
  sorry

end probability_five_or_joker_l3003_300351


namespace y_fourth_power_zero_l3003_300388

theorem y_fourth_power_zero (y : ℝ) (hy : y > 0) 
  (h : Real.sqrt (1 - y^2) + Real.sqrt (1 + y^2) = 2) : y^4 = 0 := by
  sorry

end y_fourth_power_zero_l3003_300388


namespace work_earnings_equality_l3003_300320

/-- Proves that t = 5 given the conditions of the work problem --/
theorem work_earnings_equality (t : ℝ) : 
  (t - 4 > 0) →  -- My working hours are positive
  (t - 2 > 0) →  -- Sarah's working hours are positive
  (t - 4) * (3*t - 7) = (t - 2) * (t + 1) → 
  t = 5 := by
  sorry

end work_earnings_equality_l3003_300320


namespace smallest_n_for_probability_condition_l3003_300311

theorem smallest_n_for_probability_condition : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (((m : ℝ) - 4)^3 / m^3 > 1/2) → m ≥ n) ∧
  ((n : ℝ) - 4)^3 / n^3 > 1/2 :=
by sorry

end smallest_n_for_probability_condition_l3003_300311


namespace min_likes_mozart_and_beethoven_l3003_300303

/-- Given a survey of 150 people where 120 liked Mozart and 80 liked Beethoven,
    the minimum number of people who liked both Mozart and Beethoven is 50. -/
theorem min_likes_mozart_and_beethoven
  (total : ℕ) (likes_mozart : ℕ) (likes_beethoven : ℕ)
  (h_total : total = 150)
  (h_mozart : likes_mozart = 120)
  (h_beethoven : likes_beethoven = 80) :
  (likes_mozart + likes_beethoven - total : ℤ).natAbs ≥ 50 := by
  sorry


end min_likes_mozart_and_beethoven_l3003_300303


namespace area_between_circles_l3003_300336

/-- The area of the region inside a large circle and outside eight congruent circles forming a ring --/
theorem area_between_circles (R : ℝ) (h : R = 40) : ∃ L : ℝ,
  (∃ (r : ℝ), 
    -- Eight congruent circles with radius r
    -- Each circle is externally tangent to its two adjacent circles
    -- All eight circles are internally tangent to a larger circle with radius R
    r > 0 ∧ r = R / 3 ∧
    -- L is the area of the region inside the large circle and outside all eight circles
    L = π * R^2 - 8 * π * r^2) ∧
  L = 1600 * π :=
sorry

end area_between_circles_l3003_300336


namespace at_least_one_woman_probability_l3003_300373

def num_men : ℕ := 9
def num_women : ℕ := 6
def total_people : ℕ := num_men + num_women
def num_selected : ℕ := 4

theorem at_least_one_woman_probability :
  (1 : ℚ) - (Nat.choose num_men num_selected : ℚ) / (Nat.choose total_people num_selected : ℚ) = 13/15 := by
  sorry

end at_least_one_woman_probability_l3003_300373


namespace total_cookies_l3003_300314

/-- Given 272 bags of cookies with 45 cookies in each bag, 
    prove that the total number of cookies is 12240 -/
theorem total_cookies (bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : bags = 272) (h2 : cookies_per_bag = 45) : 
  bags * cookies_per_bag = 12240 := by
  sorry

end total_cookies_l3003_300314


namespace negative_a_exponent_division_l3003_300365

theorem negative_a_exponent_division (a : ℝ) : (-a)^10 / (-a)^4 = a^6 := by
  sorry

end negative_a_exponent_division_l3003_300365


namespace least_five_digit_square_cube_l3003_300309

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n ≤ 99999) ∧
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m ≤ 99999 ∧ (∃ x : ℕ, m = x^2) ∧ (∃ y : ℕ, m = y^3) → m ≥ n) ∧
  n = 15625 :=
sorry

end least_five_digit_square_cube_l3003_300309


namespace pentagon_largest_angle_l3003_300310

/-- 
Given a pentagon where:
- Angles increase sequentially by 10 degrees
- The sum of all angles is 540 degrees
Prove that the largest angle is 128 degrees
-/
theorem pentagon_largest_angle : 
  ∀ (a₁ a₂ a₃ a₄ a₅ : ℝ),
  a₂ = a₁ + 10 →
  a₃ = a₁ + 20 →
  a₄ = a₁ + 30 →
  a₅ = a₁ + 40 →
  a₁ + a₂ + a₃ + a₄ + a₅ = 540 →
  a₅ = 128 := by
sorry

end pentagon_largest_angle_l3003_300310


namespace kay_family_age_difference_l3003_300390

/-- Given Kay's family information, prove the age difference. -/
theorem kay_family_age_difference :
  ∀ (kay_age youngest_age oldest_age : ℕ),
    kay_age = 32 →
    oldest_age = 44 →
    oldest_age = 4 * youngest_age →
    (kay_age / 2 : ℚ) - youngest_age = 5 := by
  sorry

end kay_family_age_difference_l3003_300390


namespace gcd_102_238_l3003_300366

theorem gcd_102_238 : Nat.gcd 102 238 = 34 := by
  sorry

end gcd_102_238_l3003_300366


namespace profit_percent_calculation_l3003_300323

/-- 
Theorem: Given an article where selling at half of a certain price results in a 20% loss,
the profit percent when selling at the full price is 60%.
-/
theorem profit_percent_calculation (cost_price selling_price : ℝ) : 
  (selling_price / 2 = cost_price * 0.8) →  -- Half price results in 20% loss
  (selling_price - cost_price) / cost_price = 0.6  -- Profit percent is 60%
  := by sorry

end profit_percent_calculation_l3003_300323


namespace knit_socks_together_l3003_300315

/-- The number of days it takes for two people to knit a certain number of socks together -/
def days_to_knit (a_rate b_rate : ℚ) (pairs : ℚ) : ℚ :=
  pairs / (a_rate + b_rate)

/-- Theorem: Given the rates at which A and B can knit socks individually, 
    prove that they can knit two pairs of socks in 4 days when working together -/
theorem knit_socks_together 
  (a_rate : ℚ) (b_rate : ℚ) 
  (ha : a_rate = 1 / 3) 
  (hb : b_rate = 1 / 6) : 
  days_to_knit a_rate b_rate 2 = 4 := by
  sorry

#eval days_to_knit (1/3) (1/6) 2

end knit_socks_together_l3003_300315


namespace triangle_area_l3003_300321

/-- The area of a triangle ABC with given side lengths and angle relationship -/
theorem triangle_area (a b : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : Real.cos (A - B) = 31/32) :
  (1/2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 :=
by sorry

end triangle_area_l3003_300321


namespace birthday_games_increase_l3003_300395

theorem birthday_games_increase (initial_games : ℕ) (increase_percentage : ℚ) : 
  initial_games = 7 → 
  increase_percentage = 30 / 100 → 
  initial_games + Int.floor (increase_percentage * initial_games) = 9 := by
  sorry

end birthday_games_increase_l3003_300395


namespace smallest_prime_factor_in_C_l3003_300328

def C : Finset Nat := {34, 35, 37, 41, 43}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ 
    (∀ (m : Nat), m ∈ C → (∃ (p : Nat), Nat.Prime p ∧ p ∣ n) → 
      (∃ (q : Nat), Nat.Prime q ∧ q ∣ m → p ≤ q)) ∧
    n = 34 := by
  sorry

end smallest_prime_factor_in_C_l3003_300328


namespace average_of_eight_thirteen_and_M_l3003_300396

theorem average_of_eight_thirteen_and_M (M : ℝ) (h1 : 12 < M) (h2 : M < 22) :
  (8 + 13 + M) / 3 = 13 := by
sorry

end average_of_eight_thirteen_and_M_l3003_300396


namespace vessel_width_calculation_l3003_300348

/-- Proves that the width of a rectangular vessel's base is 5 cm when a cube of edge 5 cm is
    immersed, causing a 2.5 cm rise in water level, given that the vessel's base length is 10 cm. -/
theorem vessel_width_calculation (cube_edge : ℝ) (vessel_length : ℝ) (water_rise : ℝ) :
  cube_edge = 5 →
  vessel_length = 10 →
  water_rise = 2.5 →
  ∃ (vessel_width : ℝ),
    vessel_width = 5 ∧
    cube_edge ^ 3 = vessel_length * vessel_width * water_rise :=
by sorry

end vessel_width_calculation_l3003_300348


namespace unique_solution_condition_l3003_300329

/-- The equation has exactly one real solution if and only if a < 7/4 -/
theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x^3 - a*x^2 - 3*a*x + a^2 - 2 = 0) ↔ a < 7/4 :=
by sorry

end unique_solution_condition_l3003_300329


namespace function_inequality_l3003_300371

-- Define the function f(x) = ax - x^3
def f (a : ℝ) (x : ℝ) : ℝ := a * x - x^3

-- State the theorem
theorem function_inequality (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f a x₂ - f a x₁ > x₂ - x₁) →
  a ≥ 4 := by
  sorry

end function_inequality_l3003_300371


namespace center_value_is_31_l3003_300345

/-- An arithmetic sequence -/
def ArithmeticSequence (a : Fin 5 → ℕ) : Prop :=
  ∃ d, ∀ i : Fin 4, a (i + 1) = a i + d

/-- A 5x5 array where each row and column is an arithmetic sequence -/
def ArithmeticArray (A : Fin 5 → Fin 5 → ℕ) : Prop :=
  (∀ i, ArithmeticSequence (λ j => A i j)) ∧
  (∀ j, ArithmeticSequence (λ i => A i j))

theorem center_value_is_31 (A : Fin 5 → Fin 5 → ℕ) 
  (h_array : ArithmeticArray A)
  (h_first_row : A 0 0 = 1 ∧ A 0 4 = 25)
  (h_last_row : A 4 0 = 17 ∧ A 4 4 = 81) :
  A 2 2 = 31 := by
  sorry

end center_value_is_31_l3003_300345


namespace sin_870_degrees_l3003_300355

theorem sin_870_degrees : Real.sin (870 * Real.pi / 180) = 1 / 2 := by
  sorry

end sin_870_degrees_l3003_300355


namespace exam_average_proof_l3003_300349

theorem exam_average_proof (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) (h₁ : n₁ = 15) (h₂ : n₂ = 10)
  (h₃ : avg₁ = 75/100) (h₄ : avg_total = 81/100) (h₅ : n₁ + n₂ = 25) :
  let avg₂ := (((n₁ + n₂ : ℚ) * avg_total) - (n₁ * avg₁)) / n₂
  avg₂ = 90/100 := by
sorry

end exam_average_proof_l3003_300349


namespace red_marbles_count_l3003_300319

/-- The number of red marbles Mary gave to Dan -/
def red_marbles : ℕ := 78 - 64

theorem red_marbles_count : red_marbles = 14 := by
  sorry

end red_marbles_count_l3003_300319


namespace point_B_complex_number_l3003_300359

theorem point_B_complex_number 
  (A C : ℂ) 
  (AC BC : ℂ) 
  (h1 : A = 3 + I) 
  (h2 : AC = -2 - 4*I) 
  (h3 : BC = -4 - I) 
  (h4 : C = A + AC) :
  A + AC + BC = 5 - 2*I := by
sorry

end point_B_complex_number_l3003_300359


namespace grape_juice_amount_l3003_300370

/-- Represents a fruit drink composed of orange, watermelon, and grape juice -/
structure FruitDrink where
  total : ℝ
  orange_percent : ℝ
  watermelon_percent : ℝ
  grape_percent : ℝ
  orange_watermelon_oz : ℝ

/-- The fruit drink satisfies the given conditions -/
def valid_fruit_drink (drink : FruitDrink) : Prop :=
  drink.orange_percent = 0.15 ∧
  drink.watermelon_percent = 0.60 ∧
  drink.orange_watermelon_oz = 120 ∧
  drink.orange_percent + drink.watermelon_percent + drink.grape_percent = 1

/-- Calculate the amount of grape juice in ounces -/
def grape_juice_oz (drink : FruitDrink) : ℝ :=
  drink.grape_percent * drink.total

/-- Theorem stating that the amount of grape juice is 40 ounces -/
theorem grape_juice_amount (drink : FruitDrink) 
  (h : valid_fruit_drink drink) : grape_juice_oz drink = 40 := by
  sorry

end grape_juice_amount_l3003_300370


namespace age_ratio_proof_l3003_300337

/-- Given Ronaldo's current age and the future age ratio, prove the past age ratio -/
theorem age_ratio_proof (ronaldo_current_age : ℕ) (future_ratio : ℚ) : 
  ronaldo_current_age = 36 → 
  future_ratio = 7 / 8 → 
  ∃ (roonie_current_age : ℕ), 
    (roonie_current_age + 4 : ℚ) / (ronaldo_current_age + 4 : ℚ) = future_ratio ∧ 
    (roonie_current_age - 1 : ℚ) / (ronaldo_current_age - 1 : ℚ) = 6 / 7 :=
by sorry

end age_ratio_proof_l3003_300337


namespace power_sum_equality_l3003_300306

theorem power_sum_equality : 2^345 + 3^5 * 3^3 = 2^345 + 6561 := by
  sorry

end power_sum_equality_l3003_300306


namespace smallest_solution_absolute_value_equation_l3003_300397

theorem smallest_solution_absolute_value_equation :
  let f := fun x : ℝ => x * |x| - 3 * x + 2
  ∃ x₀ : ℝ, f x₀ = 0 ∧ ∀ x : ℝ, f x = 0 → x₀ ≤ x ∧ x₀ = (-3 - Real.sqrt 17) / 2 :=
sorry

end smallest_solution_absolute_value_equation_l3003_300397


namespace sticker_distribution_theorem_l3003_300398

def distribute_stickers (total_stickers : ℕ) (num_sheets : ℕ) : ℕ :=
  sorry

theorem sticker_distribution_theorem :
  distribute_stickers 10 5 = 126 := by sorry

end sticker_distribution_theorem_l3003_300398


namespace janice_purchase_l3003_300357

theorem janice_purchase (a b c : ℕ) : 
  a + b + c = 30 →
  30 * a + 200 * b + 300 * c = 3000 →
  a = 20 :=
by sorry

end janice_purchase_l3003_300357


namespace double_average_l3003_300331

theorem double_average (n : ℕ) (original_avg : ℝ) (h1 : n = 10) (h2 : original_avg = 80) :
  let total_marks := n * original_avg
  let new_total_marks := 2 * total_marks
  let new_avg := new_total_marks / n
  new_avg = 160 := by
sorry

end double_average_l3003_300331


namespace rounded_expression_smaller_l3003_300317

theorem rounded_expression_smaller (a b c : ℕ+) :
  let exact_value := (a.val^2 : ℚ) / b.val + c.val^3
  let rounded_a := (a.val + 1 : ℚ)
  let rounded_b := (b.val + 1 : ℚ)
  let rounded_c := (c.val - 1 : ℚ)
  let rounded_value := rounded_a^2 / rounded_b + rounded_c^3
  rounded_value < exact_value :=
by sorry

end rounded_expression_smaller_l3003_300317


namespace cookie_remainder_percentage_l3003_300354

/-- Proves that given 600 initial cookies, if Nicole eats 2/5 of the total and Eduardo eats 3/5 of the remaining, then 24% of the original cookies remain. -/
theorem cookie_remainder_percentage (initial_cookies : ℕ) (nicole_fraction : ℚ) (eduardo_fraction : ℚ)
  (h_initial : initial_cookies = 600)
  (h_nicole : nicole_fraction = 2 / 5)
  (h_eduardo : eduardo_fraction = 3 / 5) :
  (initial_cookies - nicole_fraction * initial_cookies - eduardo_fraction * (initial_cookies - nicole_fraction * initial_cookies)) / initial_cookies = 24 / 100 := by
  sorry

#check cookie_remainder_percentage

end cookie_remainder_percentage_l3003_300354


namespace lena_collage_glue_drops_l3003_300352

/-- The number of closest friends Lena has -/
def num_friends : ℕ := 7

/-- The number of clippings per friend -/
def clippings_per_friend : ℕ := 3

/-- The number of glue drops needed per clipping -/
def glue_drops_per_clipping : ℕ := 6

/-- The total number of glue drops needed for Lena's collage clippings -/
def total_glue_drops : ℕ := num_friends * clippings_per_friend * glue_drops_per_clipping

theorem lena_collage_glue_drops : total_glue_drops = 126 := by
  sorry

end lena_collage_glue_drops_l3003_300352


namespace smallest_n_with_gcd_conditions_l3003_300356

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem smallest_n_with_gcd_conditions :
  ∃ (n : ℕ), n > 200 ∧ 
  Nat.gcd 70 (n + 150) = 35 ∧ 
  Nat.gcd (n + 70) 150 = 75 ∧
  ∀ (m : ℕ), m > 200 → 
    Nat.gcd 70 (m + 150) = 35 → 
    Nat.gcd (m + 70) 150 = 75 → 
    n ≤ m ∧
  n = 305 ∧
  digit_sum n = 8 := by
sorry

end smallest_n_with_gcd_conditions_l3003_300356


namespace square_root_condition_l3003_300368

theorem square_root_condition (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3*x - 5) ↔ x ≥ 5/3 := by
  sorry

end square_root_condition_l3003_300368


namespace x_squared_minus_x_greater_cube_sum_greater_l3003_300379

-- Part 1
theorem x_squared_minus_x_greater (x : ℝ) : x^2 - x > x - 2 := by sorry

-- Part 2
theorem cube_sum_greater (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by sorry

end x_squared_minus_x_greater_cube_sum_greater_l3003_300379


namespace gcd_2703_1113_l3003_300302

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := by
  sorry

end gcd_2703_1113_l3003_300302


namespace sector_area_l3003_300363

/-- Given a circular sector with a central angle of 2 radians and an arc length of 4 cm,
    the area of the sector is 4 cm². -/
theorem sector_area (θ : ℝ) (arc_length : ℝ) (h1 : θ = 2) (h2 : arc_length = 4) :
  (1/2) * arc_length * (arc_length / θ) = 4 := by
  sorry

end sector_area_l3003_300363


namespace crayons_to_mary_l3003_300305

def crayons_given_to_mary (new_pack : ℕ) (locker : ℕ) : ℕ :=
  let initial_total := new_pack + locker
  let from_bobby := locker / 2
  let final_total := initial_total + from_bobby
  final_total / 3

theorem crayons_to_mary :
  crayons_given_to_mary 21 36 = 25 := by
  sorry

end crayons_to_mary_l3003_300305


namespace digit_difference_after_reversal_l3003_300391

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_units : units < 10

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses the digits of a three-digit number -/
def reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.units
  tens := n.tens
  units := n.hundreds
  h_hundreds := n.h_units
  h_tens := n.h_tens
  h_units := n.h_hundreds

theorem digit_difference_after_reversal
  (numbers : Finset ThreeDigitNumber)
  (reversed : ThreeDigitNumber)
  (h_count : numbers.card = 10)
  (h_reversed_in : reversed ∈ numbers)
  (h_average_increase : (numbers.sum value + value (reverse reversed) - value reversed) / 10 - numbers.sum value / 10 = 198 / 10) :
  (reverse reversed).units - (reverse reversed).hundreds = 2 := by
  sorry

end digit_difference_after_reversal_l3003_300391


namespace test_scores_l3003_300393

/-- Represents the score of a test -/
structure TestScore where
  correct : Nat
  unanswered : Nat
  incorrect : Nat
  score : Nat

/-- Calculates the score for a given test result -/
def calculateScore (correct unanswered incorrect : Nat) : Nat :=
  6 * correct + unanswered

/-- Checks if a given score is achievable on the test -/
def isAchievableScore (s : Nat) : Prop :=
  ∃ (correct unanswered incorrect : Nat),
    correct + unanswered + incorrect = 25 ∧
    calculateScore correct unanswered incorrect = s

theorem test_scores :
  (isAchievableScore 130) ∧
  (isAchievableScore 131) ∧
  (isAchievableScore 133) ∧
  (isAchievableScore 138) ∧
  ¬(isAchievableScore 139) := by
  sorry

end test_scores_l3003_300393


namespace smallest_n_with_seven_in_squares_l3003_300325

/-- Returns true if the given natural number contains the digit 7 -/
def containsSeven (n : ℕ) : Prop :=
  ∃ k : ℕ, n / (10 ^ k) % 10 = 7

theorem smallest_n_with_seven_in_squares : 
  ∀ n : ℕ, n < 26 → ¬(containsSeven (n^2) ∧ containsSeven ((n+1)^2)) ∧
  (containsSeven (26^2) ∧ containsSeven (27^2)) :=
sorry

end smallest_n_with_seven_in_squares_l3003_300325


namespace man_business_ownership_l3003_300375

theorem man_business_ownership (total_value : ℝ) (sold_value : ℝ) (sold_fraction : ℝ) :
  total_value = 150000 →
  sold_value = 75000 →
  sold_fraction = 3/4 →
  ∃ original_fraction : ℝ,
    original_fraction * total_value * sold_fraction = sold_value ∧
    original_fraction = 2/3 :=
by sorry

end man_business_ownership_l3003_300375


namespace remainder_97_pow_45_mod_100_l3003_300304

theorem remainder_97_pow_45_mod_100 : 97^45 % 100 = 57 := by
  sorry

end remainder_97_pow_45_mod_100_l3003_300304


namespace computer_price_reduction_l3003_300386

/-- Given a computer price reduction of 40% resulting in a final price of 'a' yuan,
    prove that the original price was (5/3)a yuan. -/
theorem computer_price_reduction (a : ℝ) : 
  (∃ (original_price : ℝ), 
    original_price * (1 - 0.4) = a ∧ 
    original_price = (5/3) * a) :=
by sorry

end computer_price_reduction_l3003_300386


namespace square_root_power_and_increasing_l3003_300332

-- Define the function f(x) = x^(1/2) on the interval (0, +∞)
def f : ℝ → ℝ := fun x ↦ x^(1/2)

-- Define the interval (0, +∞)
def openRightHalfLine : Set ℝ := {x : ℝ | x > 0}

theorem square_root_power_and_increasing :
  (∃ r : ℝ, ∀ x ∈ openRightHalfLine, f x = x^r) ∧
  StrictMonoOn f openRightHalfLine :=
sorry

end square_root_power_and_increasing_l3003_300332


namespace hat_problem_l3003_300360

/-- Proves that given the conditions of the hat problem, the number of green hats is 30 -/
theorem hat_problem (total_hats : ℕ) (blue_price green_price : ℕ) (total_price : ℕ)
  (h1 : total_hats = 85)
  (h2 : blue_price = 6)
  (h3 : green_price = 7)
  (h4 : total_price = 540) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_price * blue_hats + green_price * green_hats = total_price ∧
    green_hats = 30 :=
by sorry

end hat_problem_l3003_300360


namespace pythagorean_number_existence_l3003_300399

theorem pythagorean_number_existence (n : ℕ) (hn : n > 12) :
  ∃ (a b c P : ℕ), a > b ∧ b > 0 ∧ c > 0 ∧
  P = a * b * (a^2 - b^2) * c^2 ∧
  n < P ∧ P < 2 * n :=
by sorry

end pythagorean_number_existence_l3003_300399


namespace min_value_x_plus_2y_l3003_300347

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 2*x*y = 8) :
  ∀ z, z = x + 2*y → z ≥ 4 ∧ ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ + 2*x₀*y₀ = 8 ∧ x₀ + 2*y₀ = 4 :=
sorry

end min_value_x_plus_2y_l3003_300347


namespace lowest_true_statement_l3003_300387

def statement201 (s203 : Bool) : Bool := s203
def statement202 (s201 : Bool) : Bool := s201
def statement203 (s206 : Bool) : Bool := ¬s206
def statement204 (s202 : Bool) : Bool := ¬s202
def statement205 (s201 s202 s203 s204 : Bool) : Bool := ¬(s201 ∨ s202 ∨ s203 ∨ s204)
def statement206 : Bool := 1 + 1 = 2

theorem lowest_true_statement :
  let s206 := statement206
  let s203 := statement203 s206
  let s201 := statement201 s203
  let s202 := statement202 s201
  let s204 := statement204 s202
  let s205 := statement205 s201 s202 s203 s204
  (¬s201 ∧ ¬s202 ∧ ¬s203 ∧ s204 ∧ ¬s205 ∧ s206) ∧
  (∀ n : Nat, n < 204 → ¬(n = 201 ∧ s201 ∨ n = 202 ∧ s202 ∨ n = 203 ∧ s203)) :=
by sorry

end lowest_true_statement_l3003_300387


namespace complex_cube_sum_ratio_l3003_300364

theorem complex_cube_sum_ratio (x y z : ℂ) 
  (hnonzero : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (hsum : x + y + z = 30)
  (hdiff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 33 := by
sorry

end complex_cube_sum_ratio_l3003_300364


namespace g_of_3_l3003_300343

def g (x : ℝ) : ℝ := -3 * x^4 + 4 * x^3 - 7 * x^2 + 5 * x - 2

theorem g_of_3 : g 3 = -185 := by
  sorry

end g_of_3_l3003_300343


namespace combined_work_time_is_14_minutes_l3003_300353

/-- Represents the time taken to complete a job when working together, given individual work rates -/
def combined_work_time (george_rate : ℚ) (abe_rate : ℚ) (carla_rate : ℚ) : ℚ :=
  1 / (george_rate + abe_rate + carla_rate)

/-- Theorem stating that given the individual work rates, the combined work time is 14 minutes -/
theorem combined_work_time_is_14_minutes :
  combined_work_time (1/70) (1/30) (1/42) = 14 := by
  sorry

#eval combined_work_time (1/70) (1/30) (1/42)

end combined_work_time_is_14_minutes_l3003_300353


namespace office_paper_shortage_l3003_300369

def paper_shortage (pack1 pack2 mon_wed_fri_usage tue_thu_usage : ℕ) (period : ℕ) : ℤ :=
  (pack1 + pack2 : ℤ) - (3 * mon_wed_fri_usage + 2 * tue_thu_usage) * period

theorem office_paper_shortage :
  paper_shortage 240 320 60 100 2 = -200 :=
by sorry

end office_paper_shortage_l3003_300369


namespace solve_for_y_l3003_300372

theorem solve_for_y (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = -6) : y = 29 := by
  sorry

end solve_for_y_l3003_300372


namespace real_estate_transaction_result_l3003_300327

/-- Represents the result of a transaction -/
inductive TransactionResult
  | Loss (amount : ℚ)
  | Gain (amount : ℚ)
  | NoChange

/-- Calculates the result of a real estate transaction -/
def calculateTransactionResult (houseSalePrice storeSalePrice : ℚ) 
                               (houseLossPercentage storeGainPercentage : ℚ) : TransactionResult :=
  let houseCost := houseSalePrice / (1 - houseLossPercentage)
  let storeCost := storeSalePrice / (1 + storeGainPercentage)
  let totalCost := houseCost + storeCost
  let totalSale := houseSalePrice + storeSalePrice
  let difference := totalCost - totalSale
  if difference > 0 then TransactionResult.Loss difference
  else if difference < 0 then TransactionResult.Gain (-difference)
  else TransactionResult.NoChange

/-- Theorem stating the result of the specific real estate transaction -/
theorem real_estate_transaction_result :
  calculateTransactionResult 15000 15000 (30/100) (25/100) = TransactionResult.Loss (3428.57/100) := by
  sorry

end real_estate_transaction_result_l3003_300327


namespace table_runner_coverage_l3003_300334

theorem table_runner_coverage (runners : Nat) 
  (area_first_three : ℝ) (area_last_two : ℝ) (table_area : ℝ) 
  (coverage_percentage : ℝ) (two_layer_area : ℝ) (one_layer_area : ℝ) :
  runners = 5 →
  area_first_three = 324 →
  area_last_two = 216 →
  table_area = 320 →
  coverage_percentage = 0.75 →
  two_layer_area = 36 →
  one_layer_area = 48 →
  ∃ (three_layer_area : ℝ),
    three_layer_area = 156 ∧
    coverage_percentage * table_area = one_layer_area + two_layer_area + three_layer_area :=
by sorry

end table_runner_coverage_l3003_300334


namespace expression_equality_l3003_300378

theorem expression_equality (x : ℝ) : x * (x * (x * (2 - x) - 4) + 10) + 1 = -x^4 + 2*x^3 - 4*x^2 + 10*x + 1 := by
  sorry

end expression_equality_l3003_300378


namespace quadratic_one_root_l3003_300380

/-- The quadratic equation x^2 + 6mx + m has exactly one real root if and only if m = 1/9 -/
theorem quadratic_one_root (m : ℝ) : 
  (∃! x, x^2 + 6*m*x + m = 0) ↔ m = 1/9 := by
sorry

end quadratic_one_root_l3003_300380


namespace mod_product_equiv_l3003_300367

theorem mod_product_equiv (m : ℕ) : 
  (264 * 391 ≡ m [ZMOD 100]) → 
  (0 ≤ m ∧ m < 100) → 
  m = 24 := by
  sorry

end mod_product_equiv_l3003_300367


namespace complex_number_with_conditions_l3003_300301

theorem complex_number_with_conditions (z : ℂ) :
  Complex.abs z = 1 →
  ∃ (y : ℝ), (3 + 4*I) * z = y*I →
  z = 4/5 - 3/5*I ∨ z = -4/5 + 3/5*I :=
by sorry

end complex_number_with_conditions_l3003_300301


namespace parabola_axis_of_symmetry_range_l3003_300392

theorem parabola_axis_of_symmetry_range 
  (a b c m n t : ℝ) 
  (h_a_pos : a > 0)
  (h_point1 : m = a + b + c)
  (h_point2 : n = 9*a + 3*b + c)
  (h_order : m < n ∧ n < c)
  (h_axis : t = -b / (2*a)) : 
  3/2 < t ∧ t < 2 := by
sorry

end parabola_axis_of_symmetry_range_l3003_300392


namespace inequality_transformations_l3003_300324

theorem inequality_transformations (a b : ℝ) (h : a < b) :
  (a + 2 < b + 2) ∧ 
  (3 * a < 3 * b) ∧ 
  ((1/2) * a < (1/2) * b) ∧ 
  (-2 * a > -2 * b) :=
by sorry

end inequality_transformations_l3003_300324


namespace equation_proof_l3003_300338

theorem equation_proof (a b : ℝ) (h : a - 2 * b = 4) : 3 - a + 2 * b = -1 := by
  sorry

end equation_proof_l3003_300338


namespace expression_value_l3003_300313

theorem expression_value (x y : ℝ) (h : 2 * x + y = 6) :
  ((x - y)^2 - (x + y)^2 + y * (2 * x - y)) / (-2 * y) = 3 := by
  sorry

end expression_value_l3003_300313


namespace polynomial_factorization_l3003_300333

theorem polynomial_factorization (x : ℤ) :
  x^12 + x^9 + 1 = (x^4 + x^3 + x^2 + x + 1) * (x^8 - x^7 + x^6 - x^5 + x^3 - x^2 + x - 1) := by
  sorry

end polynomial_factorization_l3003_300333


namespace senior_discount_percentage_l3003_300312

def original_cost : ℚ := 7.5
def coupon_discount : ℚ := 2.5
def final_payment : ℚ := 4

def cost_after_coupon : ℚ := original_cost - coupon_discount
def senior_discount_amount : ℚ := cost_after_coupon - final_payment

theorem senior_discount_percentage :
  (senior_discount_amount / cost_after_coupon) * 100 = 20 := by sorry

end senior_discount_percentage_l3003_300312


namespace min_sum_of_squares_l3003_300382

theorem min_sum_of_squares (a b c d : ℤ) : 
  a + b = 18 →
  a * b + c + d = 85 →
  a * d + b * c = 180 →
  c * d = 104 →
  ∃ (min : ℤ), min = 484 ∧ ∀ (a' b' c' d' : ℤ),
    a' + b' = 18 →
    a' * b' + c' + d' = 85 →
    a' * d' + b' * c' = 180 →
    c' * d' = 104 →
    a' ^ 2 + b' ^ 2 + c' ^ 2 + d' ^ 2 ≥ min :=
by sorry


end min_sum_of_squares_l3003_300382
