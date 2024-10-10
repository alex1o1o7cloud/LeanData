import Mathlib

namespace consecutive_integers_sum_l1125_112529

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 552) : x + (x + 1) = 47 := by
  sorry

end consecutive_integers_sum_l1125_112529


namespace system_solution_l1125_112526

theorem system_solution (x y z : ℝ) 
  (eq1 : y + z = 17 - 2*x)
  (eq2 : x + z = -11 - 2*y)
  (eq3 : x + y = 9 - 2*z) :
  3*x + 3*y + 3*z = 11.25 := by
  sorry

end system_solution_l1125_112526


namespace cosine_product_equals_quarter_l1125_112522

theorem cosine_product_equals_quarter : 
  (1 + Real.cos (π/4)) * (1 + Real.cos (3*π/4)) * (1 + Real.cos (π/2)) * (1 - Real.cos (π/4)^2) = 1/4 := by
  sorry

end cosine_product_equals_quarter_l1125_112522


namespace pizza_topping_cost_l1125_112513

/-- Represents the cost of a pizza with toppings -/
def pizza_cost (base_cost : ℚ) (first_topping_cost : ℚ) (next_two_toppings_cost : ℚ) 
  (num_slices : ℕ) (cost_per_slice : ℚ) (num_toppings : ℕ) : Prop :=
  let total_cost := cost_per_slice * num_slices
  let known_cost := base_cost + first_topping_cost + 2 * next_two_toppings_cost
  let remaining_toppings_cost := total_cost - known_cost
  let num_remaining_toppings := num_toppings - 3
  remaining_toppings_cost / num_remaining_toppings = 0.5

theorem pizza_topping_cost : 
  pizza_cost 10 2 1 8 2 7 :=
by sorry

end pizza_topping_cost_l1125_112513


namespace hours_until_visit_l1125_112565

/-- The number of hours in a day -/
def hours_per_day : ℕ := 24

/-- The number of days until Joy sees her grandma -/
def days_until_visit : ℕ := 2

/-- Theorem: The number of hours until Joy sees her grandma is 48 -/
theorem hours_until_visit : days_until_visit * hours_per_day = 48 := by
  sorry

end hours_until_visit_l1125_112565


namespace heptagon_diagonals_l1125_112501

/-- The number of diagonals in a convex n-gon --/
def numDiagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex heptagon has 14 diagonals --/
theorem heptagon_diagonals : numDiagonals 7 = 14 := by
  sorry

end heptagon_diagonals_l1125_112501


namespace alcohol_dilution_l1125_112554

/-- Proves that adding 30 ml of pure water to 50 ml of 30% alcohol solution results in 18.75% alcohol concentration -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (water_added : ℝ) (final_concentration : ℝ) : 
  initial_volume = 50 →
  initial_concentration = 0.30 →
  water_added = 30 →
  final_concentration = 0.1875 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = final_concentration :=
by
  sorry

#check alcohol_dilution

end alcohol_dilution_l1125_112554


namespace pie_difference_l1125_112553

/-- The number of pies baked per day -/
def pies_per_day : ℕ := 12

/-- The number of days apple pies are baked per week -/
def apple_pie_days : ℕ := 3

/-- The number of days cherry pies are baked per week -/
def cherry_pie_days : ℕ := 2

/-- Theorem: The difference between apple pies and cherry pies baked in one week is 12 -/
theorem pie_difference : 
  apple_pie_days * pies_per_day - cherry_pie_days * pies_per_day = 12 := by
  sorry

end pie_difference_l1125_112553


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l1125_112528

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (1 : ℕ)^k * (1 : ℕ)^(8 - k)) = 256 ∧
  Nat.choose 8 3 = 56 := by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l1125_112528


namespace y_intercept_of_parallel_line_l1125_112597

/-- A line in two-dimensional space. -/
structure Line where
  slope : ℝ
  yIntercept : ℝ

/-- Returns true if two lines are parallel. -/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Returns true if a point lies on a line. -/
def pointOnLine (l : Line) (x y : ℝ) : Prop := y = l.slope * x + l.yIntercept

/-- The given line y = -3x + 6 -/
def givenLine : Line := { slope := -3, yIntercept := 6 }

theorem y_intercept_of_parallel_line :
  ∀ b : Line,
    parallel b givenLine →
    pointOnLine b 3 (-2) →
    b.yIntercept = 7 :=
by sorry

end y_intercept_of_parallel_line_l1125_112597


namespace exponent_simplification_l1125_112570

theorem exponent_simplification :
  (10 ^ 0.4) * (10 ^ 0.6) * (10 ^ 0.3) * (10 ^ 0.2) * (10 ^ 0.5) = 100 := by
  sorry

end exponent_simplification_l1125_112570


namespace certain_to_select_genuine_l1125_112584

/-- A set of products with genuine and defective items -/
structure ProductSet where
  total : ℕ
  genuine : ℕ
  defective : ℕ
  h1 : genuine + defective = total

/-- The number of products to be selected -/
def selection_size : ℕ := 3

/-- The specific product set in the problem -/
def problem_set : ProductSet where
  total := 12
  genuine := 10
  defective := 2
  h1 := by rfl

/-- The probability of selecting at least one genuine product -/
def prob_at_least_one_genuine (ps : ProductSet) : ℚ :=
  1 - (Nat.choose ps.defective selection_size : ℚ) / (Nat.choose ps.total selection_size : ℚ)

theorem certain_to_select_genuine :
  prob_at_least_one_genuine problem_set = 1 := by
  sorry

end certain_to_select_genuine_l1125_112584


namespace experimental_fields_yield_l1125_112545

theorem experimental_fields_yield (x : ℝ) : 
  x > 0 →
  (900 : ℝ) / x = (1500 : ℝ) / (x + 300) ↔
  (∃ (area : ℝ), 
    area > 0 ∧
    area * x = 900 ∧
    area * (x + 300) = 1500) :=
by sorry

end experimental_fields_yield_l1125_112545


namespace next_perfect_cube_l1125_112500

theorem next_perfect_cube (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ y : ℕ, y > x ∧ (∃ m : ℕ, y = m^3) ∧
  y = x * (x : ℝ).sqrt + 3 * x + 3 * (x : ℝ).sqrt + 1 :=
sorry

end next_perfect_cube_l1125_112500


namespace tenRowTrianglePieces_l1125_112502

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a1 n : ℕ) : ℕ := n * (2 * a1 + (n - 1)) / 2

/-- Represents a triangle structure with rods and connectors -/
structure Triangle where
  rows : ℕ
  rodSequence : ℕ → ℕ
  connectorSequence : ℕ → ℕ

/-- Calculates the total number of pieces in the triangle -/
def totalPieces (t : Triangle) : ℕ :=
  (arithmeticSum (t.rodSequence 1) t.rows) + (arithmeticSum (t.connectorSequence 1) (t.rows + 1))

/-- The specific 10-row triangle described in the problem -/
def tenRowTriangle : Triangle :=
  { rows := 10
  , rodSequence := fun n => 3 * n
  , connectorSequence := fun n => n }

/-- Theorem stating that the total number of pieces in the 10-row triangle is 231 -/
theorem tenRowTrianglePieces : totalPieces tenRowTriangle = 231 := by
  sorry

end tenRowTrianglePieces_l1125_112502


namespace pears_in_D_l1125_112534

/-- The number of baskets --/
def num_baskets : ℕ := 5

/-- The average number of fruits per basket --/
def avg_fruits_per_basket : ℕ := 25

/-- The number of apples in basket A --/
def apples_in_A : ℕ := 15

/-- The number of mangoes in basket B --/
def mangoes_in_B : ℕ := 30

/-- The number of peaches in basket C --/
def peaches_in_C : ℕ := 20

/-- The number of bananas in basket E --/
def bananas_in_E : ℕ := 35

/-- The theorem stating the number of pears in basket D --/
theorem pears_in_D : 
  (num_baskets * avg_fruits_per_basket) - (apples_in_A + mangoes_in_B + peaches_in_C + bananas_in_E) = 25 := by
  sorry

end pears_in_D_l1125_112534


namespace fibonacci_like_invariant_l1125_112518

def fibonacci_like_sequence (u : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, u (n + 2) = u n + u (n + 1)

theorem fibonacci_like_invariant (u : ℕ → ℤ) (h : fibonacci_like_sequence u) :
  ∃ c : ℕ, ∀ n : ℕ, n ≥ 1 → |u (n - 1) * u (n + 2) - u n * u (n + 1)| = c :=
sorry

end fibonacci_like_invariant_l1125_112518


namespace election_votes_calculation_l1125_112596

theorem election_votes_calculation 
  (winning_percentage : Real) 
  (majority : Nat) 
  (total_votes : Nat) : 
  winning_percentage = 0.6 → 
  majority = 1504 → 
  (winning_percentage - (1 - winning_percentage)) * total_votes = majority → 
  total_votes = 7520 := by
sorry

end election_votes_calculation_l1125_112596


namespace cos_105_degrees_l1125_112551

theorem cos_105_degrees : 
  Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_degrees_l1125_112551


namespace lisa_caffeine_consumption_l1125_112521

/-- Represents the number of each beverage Lisa consumed --/
structure BeverageConsumption where
  coffee : ℕ
  soda : ℕ
  tea : ℕ
  energyDrink : ℕ

/-- Represents the caffeine content of each beverage in milligrams --/
structure CaffeineContent where
  coffee : ℕ
  soda : ℕ
  tea : ℕ
  energyDrink : ℕ

def totalCaffeine (consumption : BeverageConsumption) (content : CaffeineContent) : ℕ :=
  consumption.coffee * content.coffee +
  consumption.soda * content.soda +
  consumption.tea * content.tea +
  consumption.energyDrink * content.energyDrink

theorem lisa_caffeine_consumption
  (consumption : BeverageConsumption)
  (content : CaffeineContent)
  (daily_goal : ℕ)
  (h_consumption : consumption = { coffee := 3, soda := 1, tea := 2, energyDrink := 1 })
  (h_content : content = { coffee := 95, soda := 45, tea := 55, energyDrink := 120 })
  (h_goal : daily_goal = 200) :
  totalCaffeine consumption content = 560 ∧ totalCaffeine consumption content - daily_goal = 360 := by
  sorry


end lisa_caffeine_consumption_l1125_112521


namespace clear_denominators_l1125_112527

theorem clear_denominators (x : ℝ) : 
  (2*x + 1) / 3 - (10*x + 1) / 6 = 1 ↔ 4*x + 2 - 10*x - 1 = 6 := by
sorry

end clear_denominators_l1125_112527


namespace simplify_expression_l1125_112581

theorem simplify_expression (a b : ℝ) : (2*a - b) - (2*a + b) = -2*b := by
  sorry

end simplify_expression_l1125_112581


namespace two_color_no_power_of_two_sum_l1125_112587

theorem two_color_no_power_of_two_sum :
  ∃ (f : ℕ → Bool), ∀ (a b : ℕ), a ≠ b → f a = f b → ¬∃ (n : ℕ), a + b = 2^n :=
sorry

end two_color_no_power_of_two_sum_l1125_112587


namespace kylie_daisies_l1125_112562

def initial_daisies : ℕ := 5
def sister_daisies : ℕ := 9

def total_daisies : ℕ := initial_daisies + sister_daisies

def remaining_daisies : ℕ := total_daisies / 2

theorem kylie_daisies : remaining_daisies = 7 := by sorry

end kylie_daisies_l1125_112562


namespace merchant_pricing_strategy_l1125_112508

/-- Represents the pricing strategy of a merchant -/
structure MerchantPricing where
  list_price : ℝ
  purchase_discount : ℝ
  marked_price : ℝ
  selling_discount : ℝ
  profit_margin : ℝ

/-- Calculates the purchase price based on the list price and purchase discount -/
def purchase_price (m : MerchantPricing) : ℝ :=
  m.list_price * (1 - m.purchase_discount)

/-- Calculates the selling price based on the marked price and selling discount -/
def selling_price (m : MerchantPricing) : ℝ :=
  m.marked_price * (1 - m.selling_discount)

/-- Calculates the profit based on the selling price and purchase price -/
def profit (m : MerchantPricing) : ℝ :=
  selling_price m - purchase_price m

/-- Theorem: The merchant must mark the goods at 125% of the list price -/
theorem merchant_pricing_strategy (m : MerchantPricing) 
  (h1 : m.purchase_discount = 0.3)
  (h2 : m.selling_discount = 0.2)
  (h3 : m.profit_margin = 0.3)
  (h4 : profit m = m.profit_margin * selling_price m) :
  m.marked_price = 1.25 * m.list_price := by
  sorry

end merchant_pricing_strategy_l1125_112508


namespace quadratic_form_equivalence_l1125_112594

theorem quadratic_form_equivalence (d : ℕ) (h : d > 0) (h_div : d ∣ (5 + 2022^2022)) :
  (∃ x y : ℤ, d = 2*x^2 + 2*x*y + 3*y^2) ↔ (d % 20 = 3 ∨ d % 20 = 7) :=
by sorry

end quadratic_form_equivalence_l1125_112594


namespace tangent_sum_l1125_112544

theorem tangent_sum (x y : ℝ) 
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 4) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 2 := by
  sorry

end tangent_sum_l1125_112544


namespace w_squared_value_l1125_112589

theorem w_squared_value (w : ℝ) (h : 2 * (w + 15)^2 = (4 * w + 9) * (3 * w + 6)) :
  w^2 = (9 + Real.sqrt 15921) / 20 := by
  sorry

end w_squared_value_l1125_112589


namespace boat_downstream_distance_l1125_112536

/-- Calculates the distance traveled downstream by a boat given its speed in still water,
    the stream speed, and the time taken. -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Proves that a boat with a speed of 24 km/hr in still water, traveling downstream
    in a stream with a speed of 4 km/hr for 6 hours, covers a distance of 168 km. -/
theorem boat_downstream_distance :
  distance_downstream 24 4 6 = 168 := by
  sorry

end boat_downstream_distance_l1125_112536


namespace solution_set_ln_inequality_l1125_112509

theorem solution_set_ln_inequality :
  {x : ℝ | Real.log (x - Real.exp 1) < 1} = {x | Real.exp 1 < x ∧ x < 2 * Real.exp 1} := by
  sorry

end solution_set_ln_inequality_l1125_112509


namespace certain_number_exists_and_unique_l1125_112543

theorem certain_number_exists_and_unique : 
  ∃! x : ℝ, 22030 = (x + 445) * (2 * (x - 445)) + 30 := by
sorry

end certain_number_exists_and_unique_l1125_112543


namespace ratio_of_sums_eleven_l1125_112515

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (seq.a 0 + seq.a (n - 1)) / 2

theorem ratio_of_sums_eleven (a b : ArithmeticSequence)
    (h : ∀ n, a.a n / b.a n = (2 * n - 1) / (n + 1)) :
  sum_n a 11 / sum_n b 11 = 11 / 7 := by
  sorry

end ratio_of_sums_eleven_l1125_112515


namespace expression_simplification_l1125_112520

theorem expression_simplification (x : ℝ) : 
  2*x - 3*(2 - x) + 4*(2 + x) - 5*(1 - 3*x) = 24*x - 3 := by
  sorry

end expression_simplification_l1125_112520


namespace borrowed_sum_l1125_112514

/-- Given a principal P borrowed at 5% simple interest per annum,
    if after 5 years the interest is Rs. 750 less than P,
    then P must be Rs. 1000. -/
theorem borrowed_sum (P : ℝ) : 
  (P * 0.05 * 5 = P - 750) → P = 1000 := by
  sorry

end borrowed_sum_l1125_112514


namespace oplus_2_3_4_l1125_112574

-- Define the operation ⊕
def oplus (a b c : ℝ) : ℝ := a * b - 4 * a + c^2

-- Theorem statement
theorem oplus_2_3_4 : oplus 2 3 4 = 14 := by sorry

end oplus_2_3_4_l1125_112574


namespace unique_cube_fraction_l1125_112569

theorem unique_cube_fraction :
  ∃! (n : ℤ), n ≠ 30 ∧ ∃ (k : ℤ), n / (30 - n) = k^3 := by
  sorry

end unique_cube_fraction_l1125_112569


namespace basketball_team_callback_l1125_112532

/-- The number of students called back for the basketball team. -/
def students_called_back (girls boys not_called : ℕ) : ℕ :=
  girls + boys - not_called

/-- Theorem stating that 26 students were called back for the basketball team. -/
theorem basketball_team_callback : students_called_back 39 4 17 = 26 := by
  sorry

end basketball_team_callback_l1125_112532


namespace root_shrinking_method_l1125_112583

theorem root_shrinking_method (a b c p α β : ℝ) (ha : a ≠ 0) (hp : p ≠ 0) 
  (hα : a * α^2 + b * α + c = 0) (hβ : a * β^2 + b * β + c = 0) :
  (p^2 * a) * (α/p)^2 + (p * b) * (α/p) + c = 0 ∧
  (p^2 * a) * (β/p)^2 + (p * b) * (β/p) + c = 0 := by
  sorry

end root_shrinking_method_l1125_112583


namespace aquarium_fish_count_l1125_112517

theorem aquarium_fish_count (stingrays sharks eels : ℕ) : 
  stingrays = 28 →
  sharks = 2 * stingrays →
  eels = 3 * stingrays →
  stingrays + sharks + eels = 168 := by
  sorry

end aquarium_fish_count_l1125_112517


namespace johns_age_theorem_l1125_112571

theorem johns_age_theorem :
  ∀ (age : ℕ),
  (∃ (s : ℕ), (age - 2) = s^2) ∧ 
  (∃ (c : ℕ), (age + 2) = c^3) →
  age = 6 ∨ age = 123 :=
by
  sorry

end johns_age_theorem_l1125_112571


namespace triangle_sine_sides_l1125_112504

theorem triangle_sine_sides (a b c : Real) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h3 : a + b + c ≤ 2 * Real.pi) :
  Real.sin a + Real.sin b > Real.sin c ∧ 
  Real.sin b + Real.sin c > Real.sin a ∧ 
  Real.sin c + Real.sin a > Real.sin b := by
sorry

end triangle_sine_sides_l1125_112504


namespace polynomial_sum_theorem_l1125_112568

variable (x : ℝ)

def f (x : ℝ) : ℝ := x^4 - 3*x^2 + x - 1

theorem polynomial_sum_theorem (g : ℝ → ℝ) 
  (h1 : ∀ x, f x + g x = 3*x^2 - 2) :
  g = λ x => -x^4 + 6*x^2 - x - 1 := by
sorry

end polynomial_sum_theorem_l1125_112568


namespace three_correct_propositions_l1125_112537

theorem three_correct_propositions (a b c d : ℝ) : 
  (∃! n : ℕ, n = 3 ∧ 
    (((a * b > 0 ∧ b * c - a * d > 0) → (c / a - d / b > 0)) ∧
     ((a * b > 0 ∧ c / a - d / b > 0) → (b * c - a * d > 0)) ∧
     ((b * c - a * d > 0 ∧ c / a - d / b > 0) → (a * b > 0)))) := by
  sorry

end three_correct_propositions_l1125_112537


namespace system_solution_l1125_112556

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Calculates the distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Calculates the distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- Represents the system of equations -/
def satisfiesSystem (p : Point) (m : ℝ) : Prop :=
  2 * p.x - p.y = m ∧ 3 * p.x + 2 * p.y = m + 7

theorem system_solution :
  (∃ p : Point, satisfiesSystem p 0 ∧ p.x = 1 ∧ p.y = 2) ∧
  (∃ p : Point, ∃ m : ℝ,
    satisfiesSystem p m ∧
    isInSecondQuadrant p ∧
    distanceToXAxis p = 3 ∧
    distanceToYAxis p = 2 ∧
    m = -7) :=
sorry

end system_solution_l1125_112556


namespace rectangular_hall_dimensions_l1125_112563

theorem rectangular_hall_dimensions (length width : ℝ) (area : ℝ) : 
  width = length / 2 →
  area = length * width →
  area = 288 →
  length - width = 12 := by
sorry

end rectangular_hall_dimensions_l1125_112563


namespace smallest_perimeter_l1125_112541

-- Define the triangle ABC and point D
structure Triangle :=
  (A B C D : ℝ × ℝ)

-- Define the conditions
def isIsosceles (t : Triangle) : Prop :=
  ‖t.A - t.B‖ = ‖t.A - t.C‖

def DOnAC (t : Triangle) : Prop :=
  ∃ k : ℝ, 0 ≤ k ∧ k ≤ 1 ∧ t.D = k • t.A + (1 - k) • t.C

def BDPerpAC (t : Triangle) : Prop :=
  (t.B.1 - t.D.1) * (t.A.1 - t.C.1) + (t.B.2 - t.D.2) * (t.A.2 - t.C.2) = 0

def ACCDEven (t : Triangle) : Prop :=
  ∃ m n : ℕ, ‖t.A - t.C‖ = 2 * m ∧ ‖t.C - t.D‖ = 2 * n

def BDSquared36 (t : Triangle) : Prop :=
  ‖t.B - t.D‖^2 = 36

def perimeter (t : Triangle) : ℝ :=
  ‖t.A - t.B‖ + ‖t.B - t.C‖ + ‖t.C - t.A‖

theorem smallest_perimeter (t : Triangle) 
  (h1 : isIsosceles t) 
  (h2 : DOnAC t) 
  (h3 : BDPerpAC t) 
  (h4 : ACCDEven t) 
  (h5 : BDSquared36 t) : 
  ∀ t' : Triangle, 
    isIsosceles t' → DOnAC t' → BDPerpAC t' → ACCDEven t' → BDSquared36 t' → 
    perimeter t ≤ perimeter t' ∧ perimeter t = 24 :=
sorry

end smallest_perimeter_l1125_112541


namespace largest_x_floor_ratio_l1125_112505

theorem largest_x_floor_ratio : ∃ (x : ℝ), x = 63/8 ∧ 
  (∀ (y : ℝ), y > x → (⌊y⌋ : ℝ) / y ≠ 8/9) ∧ 
  (⌊x⌋ : ℝ) / x = 8/9 := by
  sorry

end largest_x_floor_ratio_l1125_112505


namespace isosceles_triangle_angles_l1125_112548

theorem isosceles_triangle_angles (a b c : ℝ) : 
  a + b + c = 180 →  -- Sum of angles in a triangle is 180°
  (a = 40 ∧ b = c) ∨ (b = 40 ∧ a = c) ∨ (c = 40 ∧ a = b) →  -- One angle is 40° and it's an isosceles triangle
  ((b = 70 ∧ c = 70) ∨ (a = 100 ∧ b = 40) ∨ (a = 100 ∧ c = 40)) :=
by sorry

end isosceles_triangle_angles_l1125_112548


namespace multiple_of_sum_and_smaller_l1125_112530

theorem multiple_of_sum_and_smaller (s l : ℕ) : 
  s + l = 84 →  -- sum of two numbers is 84
  l = s * (l / s) →  -- one number is a multiple of the other
  s = 21 →  -- the smaller number is 21
  l / s = 3 :=  -- the multiple (ratio) is 3
by
  sorry

end multiple_of_sum_and_smaller_l1125_112530


namespace average_percent_increase_per_year_l1125_112585

def initial_population : ℕ := 175000
def final_population : ℕ := 297500
def time_period : ℕ := 10

theorem average_percent_increase_per_year :
  let total_increase : ℕ := final_population - initial_population
  let average_annual_increase : ℚ := total_increase / time_period
  let percent_increase : ℚ := (average_annual_increase / initial_population) * 100
  percent_increase = 7 := by sorry

end average_percent_increase_per_year_l1125_112585


namespace min_points_guarantee_victory_min_points_is_smallest_l1125_112539

/-- Represents the possible points a racer can earn in a single race -/
inductive RacePoints
  | first  : RacePoints
  | second : RacePoints
  | third  : RacePoints

/-- Converts RacePoints to its numerical value -/
def points_value : RacePoints → Nat
  | RacePoints.first  => 6
  | RacePoints.second => 4
  | RacePoints.third  => 2

/-- The total number of races in the championship -/
def num_races : Nat := 5

/-- Calculates the total points for a list of race results -/
def total_points (results : List RacePoints) : Nat :=
  results.map points_value |>.sum

/-- Checks if a list of race results is valid (has exactly num_races races) -/
def valid_results (results : List RacePoints) : Prop :=
  results.length = num_races

/-- The minimum points needed to guarantee victory -/
def min_points_for_victory : Nat := 26

theorem min_points_guarantee_victory :
  ∀ (results : List RacePoints),
    valid_results results →
    total_points results ≥ min_points_for_victory →
    ∀ (other_results : List RacePoints),
      valid_results other_results →
      total_points results > total_points other_results :=
sorry

theorem min_points_is_smallest :
  ∀ (n : Nat),
    n < min_points_for_victory →
    ∃ (results other_results : List RacePoints),
      valid_results results ∧
      valid_results other_results ∧
      total_points results = n ∧
      total_points other_results ≥ n :=
sorry

end min_points_guarantee_victory_min_points_is_smallest_l1125_112539


namespace selling_price_with_equal_loss_l1125_112593

/-- Given an article with cost price 59 and selling price 66 resulting in a profit of 7,
    prove that the selling price resulting in the same loss as the profit is 52. -/
theorem selling_price_with_equal_loss (cost_price selling_price_profit : ℕ) 
  (h1 : cost_price = 59)
  (h2 : selling_price_profit = 66)
  (h3 : selling_price_profit - cost_price = 7) : 
  ∃ (selling_price_loss : ℕ), 
    selling_price_loss = 52 ∧ 
    cost_price - selling_price_loss = selling_price_profit - cost_price :=
by sorry

end selling_price_with_equal_loss_l1125_112593


namespace mean_reading_days_l1125_112595

def reading_data : List (Nat × Nat) := [
  (2, 1), (4, 2), (5, 3), (10, 4), (7, 5), (3, 6), (2, 7)
]

def total_days : Nat := (reading_data.map (λ (students, days) => students * days)).sum

def total_students : Nat := (reading_data.map (λ (students, _) => students)).sum

theorem mean_reading_days : 
  (total_days : ℚ) / (total_students : ℚ) = 4 := by sorry

end mean_reading_days_l1125_112595


namespace circle_area_sum_l1125_112540

/-- The sum of the areas of an infinite series of circles, where the radius of the first
    circle is 2 inches and each subsequent circle's radius is one-third of its predecessor,
    is equal to 9π/2 square inches. -/
theorem circle_area_sum : 
  let radius : ℕ → ℝ := fun n => 2 * (1/3)^(n-1)
  let area : ℕ → ℝ := fun n => π * (radius n)^2
  (∑' n, area n) = (9 * π) / 2 :=
by sorry

end circle_area_sum_l1125_112540


namespace simplify_fraction_product_l1125_112535

theorem simplify_fraction_product : (210 : ℚ) / 18 * 6 / 150 * 9 / 4 = 21 / 20 := by
  sorry

end simplify_fraction_product_l1125_112535


namespace number_difference_l1125_112524

theorem number_difference (S L : ℕ) (h1 : S = 476) (h2 : L = 6 * S + 15) :
  L - S = 2395 := by
  sorry

end number_difference_l1125_112524


namespace peter_pizza_fraction_l1125_112531

/-- Given a pizza with 16 slices, calculate the fraction eaten by Peter -/
theorem peter_pizza_fraction :
  let total_slices : ℕ := 16
  let whole_slices_eaten : ℕ := 2
  let shared_slice : ℚ := 1/2
  (whole_slices_eaten : ℚ) / total_slices + shared_slice / total_slices = 5/32 := by
  sorry

end peter_pizza_fraction_l1125_112531


namespace classroom_ratio_l1125_112573

/-- Represents a classroom with two portions of students with different GPAs -/
structure Classroom where
  portion_a : ℝ  -- Size of portion A (GPA 15)
  portion_b : ℝ  -- Size of portion B (GPA 18)
  gpa_a : ℝ      -- GPA of portion A
  gpa_b : ℝ      -- GPA of portion B
  gpa_total : ℝ  -- Total GPA of the class

/-- The ratio of portion A to the whole class is 1:3 given the conditions -/
theorem classroom_ratio (c : Classroom) 
  (h1 : c.gpa_a = 15)
  (h2 : c.gpa_b = 18)
  (h3 : c.gpa_total = 17)
  (h4 : c.gpa_a * c.portion_a + c.gpa_b * c.portion_b = c.gpa_total * (c.portion_a + c.portion_b)) :
  c.portion_a / (c.portion_a + c.portion_b) = 1 / 3 := by
  sorry

#check classroom_ratio

end classroom_ratio_l1125_112573


namespace regular_polygon_perimeter_l1125_112547

/-- A regular polygon with side length 8 units and exterior angle 45 degrees has a perimeter of 64 units. -/
theorem regular_polygon_perimeter (n : ℕ) (side_length : ℝ) (exterior_angle : ℝ) :
  n > 0 ∧ 
  side_length = 8 ∧ 
  exterior_angle = 45 ∧ 
  exterior_angle = 360 / n →
  n * side_length = 64 := by
sorry


end regular_polygon_perimeter_l1125_112547


namespace car_distance_l1125_112538

/-- Proves that a car traveling 3/4 as fast as a train going 80 miles per hour will cover 20 miles in 20 minutes -/
theorem car_distance (train_speed : ℝ) (car_speed_ratio : ℝ) (travel_time : ℝ) :
  train_speed = 80 →
  car_speed_ratio = 3 / 4 →
  travel_time = 20 / 60 →
  car_speed_ratio * train_speed * travel_time = 20 := by
  sorry

end car_distance_l1125_112538


namespace lcm_plus_hundred_l1125_112582

theorem lcm_plus_hundred (a b : ℕ) (h1 : a = 1056) (h2 : b = 792) :
  Nat.lcm a b + 100 = 3268 := by sorry

end lcm_plus_hundred_l1125_112582


namespace amanda_remaining_money_l1125_112560

/-- Calculates the remaining amount after purchases -/
def remaining_amount (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Proves that Amanda will have $7 left after her purchases -/
theorem amanda_remaining_money :
  remaining_amount 50 9 2 25 = 7 := by
  sorry

end amanda_remaining_money_l1125_112560


namespace reciprocal_location_l1125_112542

/-- A complex number is in the third quadrant if its real and imaginary parts are both negative -/
def in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

/-- A complex number is inside the unit circle if its norm is less than 1 -/
def inside_unit_circle (z : ℂ) : Prop :=
  Complex.abs z < 1

/-- A complex number is in the second quadrant if its real part is negative and imaginary part is positive -/
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

/-- A complex number is outside the unit circle if its norm is greater than 1 -/
def outside_unit_circle (z : ℂ) : Prop :=
  Complex.abs z > 1

theorem reciprocal_location (F : ℂ) :
  in_third_quadrant F ∧ inside_unit_circle F →
  in_second_quadrant (1 / F) ∧ outside_unit_circle (1 / F) :=
by sorry

end reciprocal_location_l1125_112542


namespace first_student_completion_time_l1125_112558

/-- Given a race with 4 students, prove that if the average completion time of the last 3 students
    is 35 seconds, and the average completion time of all 4 students is 30 seconds,
    then the completion time of the first student is 15 seconds. -/
theorem first_student_completion_time
  (n : ℕ)
  (avg_last_three : ℝ)
  (avg_all : ℝ)
  (h1 : n = 4)
  (h2 : avg_last_three = 35)
  (h3 : avg_all = 30)
  : (n : ℝ) * avg_all - (n - 1 : ℝ) * avg_last_three = 15 :=
by
  sorry


end first_student_completion_time_l1125_112558


namespace rotational_homothety_similarity_l1125_112598

-- Define the rotational homothety transformation
def rotationalHomothety (k : ℝ) (θ : ℝ) (O : ℝ × ℝ) : (ℝ × ℝ) → (ℝ × ℝ) := sorry

-- Define the fourth vertex of a parallelogram
def fourthVertex (O A A₁ : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define similarity of triangles
def trianglesSimilar (A B C A' B' C' : ℝ × ℝ) : Prop := sorry

theorem rotational_homothety_similarity 
  (A B C : ℝ × ℝ) -- Original triangle vertices
  (k : ℝ) (θ : ℝ) (O : ℝ × ℝ) -- Rotational homothety parameters
  (A₁ B₁ C₁ : ℝ × ℝ) -- Transformed triangle vertices
  (A₂ B₂ C₂ : ℝ × ℝ) -- Fourth vertices of parallelograms
  (h₁ : A₁ = rotationalHomothety k θ O A)
  (h₂ : B₁ = rotationalHomothety k θ O B)
  (h₃ : C₁ = rotationalHomothety k θ O C)
  (h₄ : A₂ = fourthVertex O A A₁)
  (h₅ : B₂ = fourthVertex O B B₁)
  (h₆ : C₂ = fourthVertex O C C₁) :
  trianglesSimilar A B C A₂ B₂ C₂ := by sorry

end rotational_homothety_similarity_l1125_112598


namespace solve_equation_l1125_112519

theorem solve_equation (x : ℝ) : (3 * x - 7) / 4 = 14 → x = 21 := by
  sorry

end solve_equation_l1125_112519


namespace horner_rule_operations_l1125_112557

/-- Horner's Rule evaluation for a polynomial -/
def horner_eval (coeffs : List ℤ) (x : ℤ) : ℤ × ℕ × ℕ :=
  let rec go : List ℤ → ℤ → ℕ → ℕ → ℤ × ℕ × ℕ
    | [], acc, mults, adds => (acc, mults, adds)
    | c :: cs, acc, mults, adds => go cs (c + x * acc) (mults + 1) (adds + 1)
  go (coeffs.reverse.tail) (coeffs.reverse.head!) 0 0

/-- The polynomial f(x) = 3x^6 + 4x^5 + 5x^4 + 6x^3 + 7x^2 + 8x + 1 -/
def f_coeffs : List ℤ := [1, 8, 7, 6, 5, 4, 3]

theorem horner_rule_operations :
  let (_, mults, adds) := horner_eval f_coeffs 4
  mults = 6 ∧ adds = 6 := by sorry

end horner_rule_operations_l1125_112557


namespace nougat_caramel_ratio_l1125_112506

def chocolate_problem (total caramels truffles peanut_clusters nougats : ℕ) : Prop :=
  total = 50 ∧
  caramels = 3 ∧
  truffles = caramels + 6 ∧
  peanut_clusters = (64 * total) / 100 ∧
  nougats = total - caramels - truffles - peanut_clusters ∧
  nougats = 2 * caramels

theorem nougat_caramel_ratio :
  ∀ total caramels truffles peanut_clusters nougats : ℕ,
  chocolate_problem total caramels truffles peanut_clusters nougats →
  nougats = 2 * caramels :=
by
  sorry

#check nougat_caramel_ratio

end nougat_caramel_ratio_l1125_112506


namespace find_A_value_l1125_112525

theorem find_A_value (A B : Nat) (h1 : A < 10) (h2 : B < 10) 
  (h3 : 500 + 10 * A + 8 - (100 * B + 14) = 364) : A = 7 := by
  sorry

end find_A_value_l1125_112525


namespace product_simplification_l1125_112578

theorem product_simplification : 
  (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 = 81 := by
  sorry

end product_simplification_l1125_112578


namespace min_dot_product_of_tangents_l1125_112588

-- Define a circle with radius 1
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

-- Define a point outside the circle
def PointOutside (p : ℝ × ℝ) : Prop := p.1^2 + p.2^2 > 1

-- Define tangent points
def TangentPoints (p a b : ℝ × ℝ) : Prop :=
  a ∈ Circle ∧ b ∈ Circle ∧
  ((p.1 - a.1) * a.1 + (p.2 - a.2) * a.2 = 0) ∧
  ((p.1 - b.1) * b.1 + (p.2 - b.2) * b.2 = 0)

-- Define dot product of vectors
def DotProduct (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem min_dot_product_of_tangents :
  ∀ p : ℝ × ℝ, PointOutside p →
  ∀ a b : ℝ × ℝ, TangentPoints p a b →
  ∃ m : ℝ, m = -3 + 2 * Real.sqrt 2 ∧
  ∀ x y : ℝ × ℝ, TangentPoints p x y →
  DotProduct (x.1 - p.1, x.2 - p.2) (y.1 - p.1, y.2 - p.2) ≥ m :=
sorry

end min_dot_product_of_tangents_l1125_112588


namespace lloyd_excess_rate_multiple_l1125_112576

/-- Calculates the multiple of regular rate for excess hours --/
def excessRateMultiple (regularHours : Float) (regularRate : Float) (totalHours : Float) (totalEarnings : Float) : Float :=
  let regularEarnings := regularHours * regularRate
  let excessHours := totalHours - regularHours
  let excessEarnings := totalEarnings - regularEarnings
  let excessRate := excessEarnings / excessHours
  excessRate / regularRate

/-- Proves that given Lloyd's work conditions, the multiple of his regular rate for excess hours is 2.5 --/
theorem lloyd_excess_rate_multiple :
  let regularHours : Float := 7.5
  let regularRate : Float := 4.5
  let totalHours : Float := 10.5
  let totalEarnings : Float := 67.5
  excessRateMultiple regularHours regularRate totalHours totalEarnings = 2.5 := by
  sorry

#eval excessRateMultiple 7.5 4.5 10.5 67.5

end lloyd_excess_rate_multiple_l1125_112576


namespace cos_shift_l1125_112572

theorem cos_shift (x : ℝ) : 
  2 * Real.cos (2 * (x - π / 8)) = 2 * Real.cos (2 * x - π / 4) := by
  sorry

#check cos_shift

end cos_shift_l1125_112572


namespace isosceles_60_is_equilateral_l1125_112533

-- Define an isosceles triangle with one 60° angle
def IsoscelesTriangleWith60Degree (α β γ : ℝ) : Prop :=
  (α = β ∨ β = γ ∨ γ = α) ∧ (α = 60 ∨ β = 60 ∨ γ = 60)

-- Theorem statement
theorem isosceles_60_is_equilateral (α β γ : ℝ) :
  IsoscelesTriangleWith60Degree α β γ →
  α = 60 ∧ β = 60 ∧ γ = 60 :=
by
  sorry


end isosceles_60_is_equilateral_l1125_112533


namespace class_size_l1125_112510

theorem class_size (chorus : ℕ) (band : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : chorus = 18)
  (h2 : band = 26)
  (h3 : both = 2)
  (h4 : neither = 8) :
  chorus + band - both + neither = 50 := by
  sorry

end class_size_l1125_112510


namespace miles_traveled_l1125_112566

/-- Represents the efficiency of a car in miles per gallon -/
def miles_per_gallon : ℝ := 25

/-- Represents the cost of gas in dollars per gallon -/
def dollars_per_gallon : ℝ := 5

/-- Represents the amount of money spent on gas in dollars -/
def money_spent : ℝ := 25

/-- Theorem stating that given the efficiency of the car and the cost of gas,
    $25 worth of gas will allow the car to travel 125 miles -/
theorem miles_traveled (mpg : ℝ) (dpg : ℝ) (spent : ℝ) :
  mpg = miles_per_gallon →
  dpg = dollars_per_gallon →
  spent = money_spent →
  (spent / dpg) * mpg = 125 := by
  sorry

end miles_traveled_l1125_112566


namespace swimmers_passing_theorem_l1125_112575

/-- Represents the number of times two swimmers pass each other in a pool -/
def swimmers_passing_count (pool_length : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ) : ℕ :=
  -- The actual implementation is not provided, as per the instructions
  sorry

/-- Theorem stating the number of times the swimmers pass each other under given conditions -/
theorem swimmers_passing_theorem :
  let pool_length : ℝ := 120
  let speed1 : ℝ := 4
  let speed2 : ℝ := 3
  let total_time : ℝ := 15 * 60  -- 15 minutes in seconds
  swimmers_passing_count pool_length speed1 speed2 total_time = 53 := by
  sorry

end swimmers_passing_theorem_l1125_112575


namespace concert_ticket_cost_haleys_concert_cost_l1125_112550

/-- Calculate the total amount spent on concert tickets --/
theorem concert_ticket_cost (ticket_price : ℝ) (num_tickets : ℕ) 
  (discount_rate : ℝ) (discount_threshold : ℕ) (service_fee : ℝ) : ℝ :=
  let base_cost := ticket_price * num_tickets
  let discount := if num_tickets > discount_threshold then discount_rate * base_cost else 0
  let discounted_cost := base_cost - discount
  let total_service_fee := service_fee * num_tickets
  let total_cost := discounted_cost + total_service_fee
  by
    -- Proof goes here
    sorry

/-- Haley's concert ticket purchase --/
theorem haleys_concert_cost : 
  concert_ticket_cost 4 8 0.1 5 2 = 44.8 :=
by
  -- Proof goes here
  sorry

end concert_ticket_cost_haleys_concert_cost_l1125_112550


namespace min_distance_to_line_l1125_112567

/-- Given the line x + 2y = 1, the minimum value of x^2 + y^2 is 1/5 -/
theorem min_distance_to_line (x y : ℝ) (h : x + 2*y = 1) : 
  ∃ (min : ℝ), min = 1/5 ∧ ∀ (x' y' : ℝ), x' + 2*y' = 1 → x'^2 + y'^2 ≥ min :=
sorry

end min_distance_to_line_l1125_112567


namespace cyclist_climbing_speed_l1125_112580

/-- Proves that the climbing speed is 20 m/min given the specified conditions -/
theorem cyclist_climbing_speed 
  (hill_length : ℝ) 
  (total_time : ℝ) 
  (climbing_speed : ℝ) :
  hill_length = 400 ∧ 
  total_time = 30 ∧ 
  (∃ t : ℝ, t > 0 ∧ t < 30 ∧ 
    hill_length = climbing_speed * t ∧ 
    hill_length = 2 * climbing_speed * (total_time - t)) →
  climbing_speed = 20 := by
  sorry

#check cyclist_climbing_speed

end cyclist_climbing_speed_l1125_112580


namespace fraction_zero_implies_a_neg_two_l1125_112559

theorem fraction_zero_implies_a_neg_two (a : ℝ) :
  (a^2 - 4) / (a - 2) = 0 → a = -2 := by
sorry

end fraction_zero_implies_a_neg_two_l1125_112559


namespace largest_pot_cost_largest_pot_cost_is_1_92_l1125_112579

/-- The cost of the largest pot given specific conditions -/
theorem largest_pot_cost (num_pots : ℕ) (total_cost : ℚ) (price_diff : ℚ) (smallest_pot_odd_cents : Bool) : ℚ :=
  let smallest_pot_cost : ℚ := (total_cost - price_diff * (num_pots * (num_pots - 1) / 2)) / num_pots
  let rounded_smallest_pot_cost : ℚ := if smallest_pot_odd_cents then ⌊smallest_pot_cost * 100⌋ / 100 else ⌈smallest_pot_cost * 100⌉ / 100
  rounded_smallest_pot_cost + price_diff * (num_pots - 1)

/-- The main theorem proving the cost of the largest pot -/
theorem largest_pot_cost_is_1_92 :
  largest_pot_cost 6 (39/5) (1/4) true = 96/50 := by
  sorry

end largest_pot_cost_largest_pot_cost_is_1_92_l1125_112579


namespace factorization_equality_l1125_112549

theorem factorization_equality (a x y : ℝ) :
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) := by
  sorry

end factorization_equality_l1125_112549


namespace largest_number_l1125_112561

theorem largest_number (a b c d e : ℝ) : 
  a = 24680 + 1/1357 →
  b = 24680 - 1/1357 →
  c = 24680 * (1/1357) →
  d = 24680 / (1/1357) →
  e = 24680.1357 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end largest_number_l1125_112561


namespace fraction_simplification_l1125_112507

theorem fraction_simplification :
  (5 : ℝ) / (3 * Real.sqrt 50 + Real.sqrt 18 + 4 * Real.sqrt 8) = (5 * Real.sqrt 2) / 52 := by
  sorry

end fraction_simplification_l1125_112507


namespace constant_width_interior_angle_ge_120_l1125_112552

/-- A curve of constant width. -/
class ConstantWidthCurve (α : Type*) [MetricSpace α] where
  width : ℝ
  is_constant_width : ∀ (x y : α), dist x y ≤ width

/-- The interior angle at a point on a curve. -/
def interior_angle {α : Type*} [MetricSpace α] (c : ConstantWidthCurve α) (p : α) : ℝ := sorry

/-- Theorem: The interior angle at any corner point of a curve of constant width is at least 120 degrees. -/
theorem constant_width_interior_angle_ge_120 
  {α : Type*} [MetricSpace α] (c : ConstantWidthCurve α) (p : α) :
  interior_angle c p ≥ 120 := by sorry

end constant_width_interior_angle_ge_120_l1125_112552


namespace range_of_a_l1125_112564

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (a + 6) + y^2 / (a - 7) = 1 ∧ 
  (∃ (b c : ℝ), (x = 0 ∧ y = b) ∨ (x = c ∧ y = 0))

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 4*x + a < 0

-- Define the theorem
theorem range_of_a : 
  (∀ a : ℝ, p a ∨ ¬(q a)) → 
  ∀ a : ℝ, a ∈ Set.Ioi (-6) :=
sorry

end range_of_a_l1125_112564


namespace shoe_selection_probability_l1125_112590

/-- The number of pairs of shoes -/
def num_pairs : ℕ := 5

/-- The total number of shoes -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes to be selected -/
def selected_shoes : ℕ := 2

/-- The number of ways to select 2 shoes of the same color -/
def same_color_selections : ℕ := num_pairs

/-- The total number of ways to select 2 shoes from 10 shoes -/
def total_selections : ℕ := Nat.choose total_shoes selected_shoes

theorem shoe_selection_probability :
  same_color_selections / total_selections = 1 / 9 := by sorry

end shoe_selection_probability_l1125_112590


namespace middle_terms_equal_l1125_112546

/-- Given two geometric progressions with positive terms satisfying certain conditions,
    prove that the middle terms are equal. -/
theorem middle_terms_equal (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
    (h_pos_a : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0)
    (h_pos_b : b₁ > 0 ∧ b₂ > 0 ∧ b₃ > 0)
    (h_geom_a : ∃ q : ℝ, q > 0 ∧ a₂ = a₁ * q ∧ a₃ = a₂ * q)
    (h_geom_b : ∃ r : ℝ, r > 0 ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r)
    (h_sum_eq : a₁ + a₂ + a₃ = b₁ + b₂ + b₃)
    (h_arith_prog : ∃ d : ℝ, a₂ * b₂ - a₁ * b₁ = d ∧ a₃ * b₃ - a₂ * b₂ = d) :
  a₂ = b₂ := by
  sorry

end middle_terms_equal_l1125_112546


namespace power_of_two_six_l1125_112577

theorem power_of_two_six : 2^3 * 2^3 = 2^6 := by
  sorry

end power_of_two_six_l1125_112577


namespace min_value_theorem_l1125_112511

theorem min_value_theorem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_eq : 3 * x + y = 5 * x * y) :
  4 * x + 3 * y ≥ 5 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 3 * x₀ + y₀ = 5 * x₀ * y₀ ∧ 4 * x₀ + 3 * y₀ = 5 :=
sorry

end min_value_theorem_l1125_112511


namespace cubic_root_sum_l1125_112591

theorem cubic_root_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2024*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 100 := by
  sorry

end cubic_root_sum_l1125_112591


namespace black_haired_girls_l1125_112523

/-- Represents the number of girls in the choir -/
def initial_total : ℕ := 80

/-- Represents the number of blonde-haired girls added -/
def blonde_added : ℕ := 10

/-- Represents the initial number of blonde-haired girls -/
def initial_blonde : ℕ := 30

/-- Theorem stating the number of black-haired girls in the choir -/
theorem black_haired_girls : 
  initial_total - (initial_blonde + blonde_added) = 50 := by
  sorry

end black_haired_girls_l1125_112523


namespace sufficient_not_necessary_l1125_112592

-- Define the log function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the condition for lgm < 1
def condition (m : ℝ) : Prop := log m < 1

-- Define the set {1, 2}
def set_B : Set ℝ := {1, 2}

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ m ∈ set_B, condition m) ∧
  (∃ m : ℝ, condition m ∧ m ∉ set_B) :=
sorry

end sufficient_not_necessary_l1125_112592


namespace three_four_five_pythagorean_triple_l1125_112512

/-- Definition of a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

/-- Theorem: (3,4,5) is a Pythagorean triple -/
theorem three_four_five_pythagorean_triple :
  isPythagoreanTriple 3 4 5 := by
  sorry

end three_four_five_pythagorean_triple_l1125_112512


namespace greatest_integer_fraction_inequality_l1125_112599

theorem greatest_integer_fraction_inequality :
  ∀ y : ℤ, (8 : ℚ) / 11 > (y : ℚ) / 17 ↔ y ≤ 12 :=
by sorry

end greatest_integer_fraction_inequality_l1125_112599


namespace article_sale_price_l1125_112503

/-- Given an article with cost price CP, prove that the selling price SP
    that yields the same percentage profit as the percentage loss when
    sold for 1280 is 1820, given that selling it for 1937.5 gives a 25% profit. -/
theorem article_sale_price (CP : ℝ) 
    (h1 : 1937.5 = CP * 1.25)  -- 25% profit condition
    (h2 : ∃ SP, (SP - CP) / CP = (CP - 1280) / CP)  -- Equal percentage condition
    : ∃ SP, SP = 1820 ∧ (SP - CP) / CP = (CP - 1280) / CP := by
  sorry

end article_sale_price_l1125_112503


namespace geometric_series_product_sum_limit_l1125_112555

/-- The limit of the sum of the product of corresponding terms from two geometric series --/
theorem geometric_series_product_sum_limit (a r s : ℝ) 
  (hr : |r| < 1) (hs : |s| < 1) : 
  (∑' n, a^2 * (r*s)^n) = a^2 / (1 - r*s) := by
  sorry

end geometric_series_product_sum_limit_l1125_112555


namespace rectangle_inequality_l1125_112586

/-- Represents a rectangle with side lengths 3b and b -/
structure Rectangle (b : ℝ) where
  length : ℝ := 3 * b
  width : ℝ := b

/-- Represents a point P on the longer side of the rectangle -/
structure PointP (b : ℝ) where
  x : ℝ
  y : ℝ := 0
  h1 : 0 ≤ x ∧ x ≤ 3 * b

/-- Represents a point T inside the rectangle -/
structure PointT (b : ℝ) where
  x : ℝ
  y : ℝ
  h1 : 0 < x ∧ x < 3 * b
  h2 : 0 < y ∧ y < b
  h3 : y = b / 2

/-- The theorem to be proved -/
theorem rectangle_inequality (b : ℝ) (h : b > 0) (R : Rectangle b) (P : PointP b) (T : PointT b) :
  let s := (2 * b)^2 + b^2
  let rt := (T.x - 0)^2 + (T.y - 0)^2
  s > 2 * rt := by sorry

end rectangle_inequality_l1125_112586


namespace joannas_reading_time_l1125_112516

/-- Joanna's reading problem -/
theorem joannas_reading_time (
  total_pages : ℕ)
  (pages_per_hour : ℕ)
  (monday_hours : ℕ)
  (remaining_hours : ℕ)
  (h1 : total_pages = 248)
  (h2 : pages_per_hour = 16)
  (h3 : monday_hours = 3)
  (h4 : remaining_hours = 6)
  : (total_pages - (monday_hours * pages_per_hour + remaining_hours * pages_per_hour)) / pages_per_hour = 13/2 := by
  sorry

end joannas_reading_time_l1125_112516
