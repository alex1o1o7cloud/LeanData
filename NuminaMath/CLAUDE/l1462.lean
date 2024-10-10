import Mathlib

namespace min_value_and_inequality_l1462_146277

theorem min_value_and_inequality (x y z : ℝ) (h : x + y + z = 1) :
  ((x - 1)^2 + (y + 1)^2 + (z + 1)^2 ≥ 4/3) ∧
  (∀ a : ℝ, (x - 2)^2 + (y - 1)^2 + (z - a)^2 ≥ 1/3 → a ≤ -3 ∨ a ≥ -1) := by
  sorry

end min_value_and_inequality_l1462_146277


namespace min_cards_to_verify_statement_l1462_146294

/-- Represents the side of a card -/
inductive CardSide
| Color (c : String)
| Smiley (happy : Bool)

/-- Represents a card with two sides -/
structure Card :=
  (side1 side2 : CardSide)

/-- The statement to be verified -/
def statement (c : Card) : Prop :=
  match c.side1, c.side2 with
  | CardSide.Smiley true, CardSide.Color "yellow" => True
  | CardSide.Color "yellow", CardSide.Smiley true => True
  | _, _ => False

/-- The set of cards given in the problem -/
def cards : Finset Card := sorry

/-- The minimum number of cards to turn over -/
def min_cards_to_turn : ℕ := sorry

theorem min_cards_to_verify_statement :
  min_cards_to_turn = 2 ∧
  ∃ (c1 c2 : Card), c1 ∈ cards ∧ c2 ∈ cards ∧ c1 ≠ c2 ∧
    (∀ (c : Card), c ∈ cards → statement c ↔ (c = c1 ∨ c = c2)) :=
sorry

end min_cards_to_verify_statement_l1462_146294


namespace quadratic_two_distinct_roots_l1462_146268

theorem quadratic_two_distinct_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  x₁^2 - (2*k - 1)*x₁ + k^2 - k = 0 ∧
  x₂^2 - (2*k - 1)*x₂ + k^2 - k = 0 :=
by sorry

end quadratic_two_distinct_roots_l1462_146268


namespace logarithm_equality_l1462_146252

theorem logarithm_equality (a b c x : ℝ) (p q r y : ℝ) :
  a > 0 → b > 0 → c > 0 → x > 0 → x ≠ 1 →
  (∀ (base : ℝ), base > 1 →
    (Real.log a / p = Real.log b / q) ∧
    (Real.log b / q = Real.log c / r) ∧
    (Real.log c / r = Real.log x)) →
  b^3 / (a^2 * c) = x^y →
  y = 3*q - 2*p - r :=
by sorry

end logarithm_equality_l1462_146252


namespace chess_tournament_participants_l1462_146206

theorem chess_tournament_participants (n : ℕ) : 
  (n * (n - 1)) / 2 = 171 → n = 19 := by
  sorry

end chess_tournament_participants_l1462_146206


namespace parabola_y_intercepts_l1462_146265

/-- The number of y-intercepts for the parabola x = 3y^2 - 4y + 5 -/
def num_y_intercepts : ℕ := 0

/-- The equation of the parabola -/
def parabola_equation (y : ℝ) : ℝ := 3 * y^2 - 4 * y + 5

theorem parabola_y_intercepts :
  (∀ y : ℝ, parabola_equation y ≠ 0) ∧ num_y_intercepts = 0 := by sorry

end parabola_y_intercepts_l1462_146265


namespace even_decreasing_properties_l1462_146200

/-- A function that is even and monotonically decreasing on (0, +∞) -/
def EvenDecreasingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, 0 < x ∧ x < y → f y < f x)

theorem even_decreasing_properties (f : ℝ → ℝ) (hf : EvenDecreasingFunction f) :
  (∃ a : ℝ, f (2 * a) ≥ f (-a)) ∧ 
  (f π ≤ f (-3)) ∧
  (f (-Real.sqrt 3 / 2) < f (4 / 5)) ∧
  (∃ a : ℝ, f (a^2 + 1) ≥ f 1) := by
  sorry

end even_decreasing_properties_l1462_146200


namespace no_tangent_point_largest_integer_a_l1462_146298

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (a / 2) * x^2

theorem no_tangent_point (a : ℝ) : ¬∃ x, f a x = 0 ∧ (deriv (f a)) x = 0 := sorry

theorem largest_integer_a :
  ∃ a : ℤ, (∀ x₁ x₂ : ℝ, x₂ > 0 → f a (x₁ + x₂) - f a (x₁ - x₂) > -2 * x₂) ∧
  (∀ b : ℤ, b > a → ∃ x₁ x₂ : ℝ, x₂ > 0 ∧ f b (x₁ + x₂) - f b (x₁ - x₂) ≤ -2 * x₂) ∧
  a = 3 := sorry

end no_tangent_point_largest_integer_a_l1462_146298


namespace circle_radius_increase_circle_radius_increase_is_five_over_pi_l1462_146292

/-- Represents the change in radius when a circle's circumference increases from 30 to 40 inches -/
theorem circle_radius_increase : ℝ → Prop :=
  fun Δr =>
    ∃ (r₁ r₂ : ℝ),
      (2 * Real.pi * r₁ = 30) ∧
      (2 * Real.pi * r₂ = 40) ∧
      (r₂ - r₁ = Δr) ∧
      (Δr = 5 / Real.pi)

/-- Proves that the radius increase is 5/π inches -/
theorem circle_radius_increase_is_five_over_pi :
  circle_radius_increase (5 / Real.pi) :=
by
  sorry

end circle_radius_increase_circle_radius_increase_is_five_over_pi_l1462_146292


namespace money_division_l1462_146279

theorem money_division (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : a = 80) (h2 : a = (2/3) * (b + c)) (h3 : b = (6/9) * (a + c)) : 
  a + b + c = 200 := by
  sorry

end money_division_l1462_146279


namespace kolya_mistake_l1462_146249

structure Box where
  blue : ℕ
  green : ℕ

def vasya_correct (b : Box) : Prop := b.blue ≥ 4
def kolya_correct (b : Box) : Prop := b.green ≥ 5
def petya_correct (b : Box) : Prop := b.blue ≥ 3 ∧ b.green ≥ 4
def misha_correct (b : Box) : Prop := b.blue ≥ 4 ∧ b.green ≥ 4

theorem kolya_mistake (b : Box) :
  (vasya_correct b ∧ petya_correct b ∧ misha_correct b ∧ ¬kolya_correct b) ∨
  (vasya_correct b ∧ petya_correct b ∧ misha_correct b ∧ kolya_correct b) :=
by sorry

end kolya_mistake_l1462_146249


namespace tangent_parallel_and_inequality_l1462_146253

/-- The function f(x) = x^3 - ax^2 + 3x + b -/
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 + 3*x + b

/-- The derivative of f(x) -/
def f_deriv (a x : ℝ) : ℝ := 3*x^2 - 2*a*x + 3

theorem tangent_parallel_and_inequality (a b : ℝ) :
  (f_deriv a 1 = 0) →
  (∀ x ∈ Set.Icc (-1) 4, f a b x > f_deriv a x) →
  (a = 3 ∧ b > 19) := by sorry

end tangent_parallel_and_inequality_l1462_146253


namespace daves_diner_cost_l1462_146213

/-- Represents the pricing and discount structure at Dave's Diner -/
structure DavesDiner where
  burger_price : ℕ
  fries_price : ℕ
  discount_amount : ℕ
  discount_threshold : ℕ

/-- Calculates the total cost of a purchase at Dave's Diner -/
def calculate_total_cost (d : DavesDiner) (num_burgers : ℕ) (num_fries : ℕ) : ℕ :=
  let burger_cost := if num_burgers ≥ d.discount_threshold
    then (d.burger_price - d.discount_amount) * num_burgers
    else d.burger_price * num_burgers
  let fries_cost := d.fries_price * num_fries
  burger_cost + fries_cost

/-- Theorem stating that the total cost of 6 burgers and 5 fries at Dave's Diner is 27 -/
theorem daves_diner_cost : 
  let d : DavesDiner := { 
    burger_price := 4, 
    fries_price := 3, 
    discount_amount := 2, 
    discount_threshold := 4 
  }
  calculate_total_cost d 6 5 = 27 := by
  sorry

end daves_diner_cost_l1462_146213


namespace disprove_statement_l1462_146209

theorem disprove_statement : ∃ (a b c : ℤ), a > b ∧ b > c ∧ ¬(a + b > c) :=
  sorry

end disprove_statement_l1462_146209


namespace drum_sticks_per_show_l1462_146240

/-- Proves that the number of drum stick sets used per show for playing is 5 --/
theorem drum_sticks_per_show 
  (total_shows : ℕ) 
  (tossed_per_show : ℕ) 
  (total_sets : ℕ) 
  (h1 : total_shows = 30) 
  (h2 : tossed_per_show = 6) 
  (h3 : total_sets = 330) : 
  (total_sets - total_shows * tossed_per_show) / total_shows = 5 := by
  sorry

#check drum_sticks_per_show

end drum_sticks_per_show_l1462_146240


namespace point_in_fourth_quadrant_y_negative_l1462_146272

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

theorem point_in_fourth_quadrant_y_negative (y : ℝ) :
  in_fourth_quadrant (Point2D.mk 5 y) → y < 0 := by
  sorry

end point_in_fourth_quadrant_y_negative_l1462_146272


namespace fraction_product_l1462_146218

theorem fraction_product : (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 = 27 := by
  sorry

end fraction_product_l1462_146218


namespace li_elevator_journey_l1462_146220

def floor_movements : List Int := [5, -3, 10, -8, 12, -6, -10]
def floor_height : ℝ := 2.8
def electricity_per_meter : ℝ := 0.1

theorem li_elevator_journey :
  (List.sum floor_movements = 0) ∧
  (List.sum (List.map (λ x => floor_height * electricity_per_meter * |x|) floor_movements) = 15.12) := by
  sorry

end li_elevator_journey_l1462_146220


namespace christine_money_l1462_146273

theorem christine_money (total : ℕ) (difference : ℕ) : 
  total = 50 → difference = 30 → ∃ (christine siri : ℕ), 
    christine = siri + difference ∧ 
    christine + siri = total ∧ 
    christine = 40 := by sorry

end christine_money_l1462_146273


namespace subtraction_problem_l1462_146245

theorem subtraction_problem : 
  (845.59 : ℝ) - 249.27 = 596.32 := by
  sorry

end subtraction_problem_l1462_146245


namespace min_real_roots_l1462_146285

/-- A polynomial of degree 2010 with real coefficients -/
def RealPolynomial2010 : Type := Polynomial ℝ

/-- The roots of a polynomial -/
def roots (p : RealPolynomial2010) : Multiset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinctAbsValues (p : RealPolynomial2010) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def realRootCount (p : RealPolynomial2010) : ℕ := sorry

/-- The degree of the polynomial -/
def degree (p : RealPolynomial2010) : ℕ := 2010

theorem min_real_roots (g : RealPolynomial2010) 
  (h1 : degree g = 2010)
  (h2 : distinctAbsValues g = 1006) : 
  realRootCount g ≥ 6 := sorry

end min_real_roots_l1462_146285


namespace cube_sum_reciprocal_squared_l1462_146251

theorem cube_sum_reciprocal_squared (x : ℝ) (h : 53 = x^6 + 1/x^6) : (x^3 + 1/x^3)^2 = 55 := by
  sorry

end cube_sum_reciprocal_squared_l1462_146251


namespace algebraic_identity_l1462_146293

theorem algebraic_identity (a b : ℝ) : 
  a = (Real.sqrt 5 + Real.sqrt 3) / (Real.sqrt 5 - Real.sqrt 3) →
  b = (Real.sqrt 5 - Real.sqrt 3) / (Real.sqrt 5 + Real.sqrt 3) →
  a^4 + b^4 + (a + b)^4 = 7938 := by
  sorry

end algebraic_identity_l1462_146293


namespace toms_fruit_purchase_cost_l1462_146274

/-- Calculates the total cost of fruits with applied discounts -/
def total_cost_with_discounts (apple_kg : ℝ) (apple_price : ℝ) (mango_kg : ℝ) (mango_price : ℝ)
  (orange_kg : ℝ) (orange_price : ℝ) (banana_kg : ℝ) (banana_price : ℝ)
  (apple_discount : ℝ) (orange_discount : ℝ) : ℝ :=
  let apple_cost := apple_kg * apple_price * (1 - apple_discount)
  let mango_cost := mango_kg * mango_price
  let orange_cost := orange_kg * orange_price * (1 - orange_discount)
  let banana_cost := banana_kg * banana_price
  apple_cost + mango_cost + orange_cost + banana_cost

/-- Theorem stating that the total cost of Tom's fruit purchase is $1391.5 -/
theorem toms_fruit_purchase_cost :
  total_cost_with_discounts 8 70 9 65 5 50 3 30 0.1 0.15 = 1391.5 := by
  sorry

end toms_fruit_purchase_cost_l1462_146274


namespace probability_greater_than_two_l1462_146270

/-- A standard die has 6 sides -/
def die_sides : ℕ := 6

/-- The number of outcomes greater than 2 -/
def favorable_outcomes : ℕ := 4

/-- The probability of rolling a number greater than 2 on a standard six-sided die -/
theorem probability_greater_than_two : 
  (favorable_outcomes : ℚ) / die_sides = 2 / 3 := by
  sorry

end probability_greater_than_two_l1462_146270


namespace fabian_sugar_packs_l1462_146289

/-- The number of packs of sugar Fabian wants to buy -/
def sugar_packs : ℕ := 3

/-- The price of apples in dollars per kilogram -/
def apple_price : ℚ := 2

/-- The price of walnuts in dollars per kilogram -/
def walnut_price : ℚ := 6

/-- The price of sugar in dollars per pack -/
def sugar_price : ℚ := apple_price - 1

/-- The amount of apples Fabian wants to buy in kilograms -/
def apple_amount : ℚ := 5

/-- The amount of walnuts Fabian wants to buy in kilograms -/
def walnut_amount : ℚ := 1/2

/-- The total amount Fabian needs to pay in dollars -/
def total_cost : ℚ := 16

theorem fabian_sugar_packs : 
  sugar_packs = (total_cost - apple_price * apple_amount - walnut_price * walnut_amount) / sugar_price := by
  sorry

end fabian_sugar_packs_l1462_146289


namespace square_quotient_theorem_l1462_146255

theorem square_quotient_theorem (a b : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_div : (a * b + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, (a^2 + b^2) / (a * b + 1) = k^2 := by
sorry

end square_quotient_theorem_l1462_146255


namespace median_after_removal_l1462_146212

def room_sequence : List Nat := List.range 26

def remaining_rooms (seq : List Nat) : List Nat :=
  seq.filter (fun n => n ≠ 15 ∧ n ≠ 20 ∧ n ≠ 0)

theorem median_after_removal (seq : List Nat) (h : seq = room_sequence) :
  (remaining_rooms seq).get? ((remaining_rooms seq).length / 2) = some 12 := by
  sorry

end median_after_removal_l1462_146212


namespace equation_solution_l1462_146211

theorem equation_solution : ∃ x : ℝ, 4 * x - 2 = 2 * (x + 2) ∧ x = 3 := by
  sorry

end equation_solution_l1462_146211


namespace fraction_zero_implies_x_negative_four_l1462_146239

theorem fraction_zero_implies_x_negative_four (x : ℝ) :
  (|x| - 4) / (4 - x) = 0 → x = -4 := by
  sorry

end fraction_zero_implies_x_negative_four_l1462_146239


namespace product_QED_l1462_146221

theorem product_QED (Q E D : ℂ) (hQ : Q = 6 + 3*I) (hE : E = -I) (hD : D = 6 - 3*I) :
  Q * E * D = -45 * I :=
by sorry

end product_QED_l1462_146221


namespace paths_7x8_grid_l1462_146276

/-- The number of distinct paths on a rectangular grid -/
def gridPaths (width height : ℕ) : ℕ :=
  Nat.choose (width + height) height

/-- Theorem: The number of distinct paths on a 7x8 grid is 6435 -/
theorem paths_7x8_grid :
  gridPaths 7 8 = 6435 := by
  sorry

end paths_7x8_grid_l1462_146276


namespace equation_solution_l1462_146271

theorem equation_solution : 
  let x : ℚ := -43/8
  7 * (4 * x + 3) - 5 = -3 * (2 - 8 * x) + 1/2 :=
by sorry

end equation_solution_l1462_146271


namespace kelly_games_to_give_away_l1462_146266

/-- Given that Kelly has a certain number of Nintendo games and wants to keep a specific number,
    prove that the number of games she needs to give away is the difference between these two numbers. -/
theorem kelly_games_to_give_away (initial_nintendo_games kept_nintendo_games : ℕ) :
  initial_nintendo_games ≥ kept_nintendo_games →
  initial_nintendo_games - kept_nintendo_games =
  initial_nintendo_games - kept_nintendo_games :=
by
  sorry

#check kelly_games_to_give_away 20 12

end kelly_games_to_give_away_l1462_146266


namespace total_bronze_needed_l1462_146259

/-- The weight of the first bell in pounds -/
def first_bell_weight : ℕ := 50

/-- The weight of the second bell in pounds -/
def second_bell_weight : ℕ := 2 * first_bell_weight

/-- The weight of the third bell in pounds -/
def third_bell_weight : ℕ := 4 * second_bell_weight

/-- The total weight of bronze needed for all three bells -/
def total_bronze_weight : ℕ := first_bell_weight + second_bell_weight + third_bell_weight

theorem total_bronze_needed :
  total_bronze_weight = 550 := by sorry

end total_bronze_needed_l1462_146259


namespace summer_sales_is_seven_l1462_146280

/-- The number of million hamburgers sold in each season --/
structure SeasonalSales where
  spring : ℝ
  summer : ℝ
  fall : ℝ
  winter : ℝ

/-- The total annual sales of hamburgers in millions --/
def total_sales (s : SeasonalSales) : ℝ :=
  s.spring + s.summer + s.fall + s.winter

/-- Theorem stating that the number of million hamburgers sold in the summer is 7 --/
theorem summer_sales_is_seven (s : SeasonalSales) 
  (h1 : s.fall = 0.2 * total_sales s)
  (h2 : s.fall = 3)
  (h3 : s.spring = 2)
  (h4 : s.winter = 3) : 
  s.summer = 7 := by
  sorry

end summer_sales_is_seven_l1462_146280


namespace function_range_theorem_l1462_146295

def f (x : ℝ) := x^2 - 2*x - 3

theorem function_range_theorem (m : ℝ) (h_m : m > 0) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ -3) ∧
  (∃ x ∈ Set.Icc 0 m, f x = -3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ -4) ∧
  (∃ x ∈ Set.Icc 0 m, f x = -4) →
  m ∈ Set.Icc 1 2 := by sorry

end function_range_theorem_l1462_146295


namespace inequality_solutions_l1462_146241

theorem inequality_solutions : 
  ∃! (s : Finset Int), 
    (∀ y ∈ s, (2 * y ≤ -y + 4 ∧ 5 * y ≥ -10 ∧ 3 * y ≤ -2 * y + 20)) ∧ 
    s.card = 4 :=
by sorry

end inequality_solutions_l1462_146241


namespace frisbee_price_problem_l1462_146247

/-- The price of the other frisbees in a sporting goods store -/
theorem frisbee_price_problem :
  ∀ (F₃ F_x x : ℕ),
    F₃ + F_x = 64 →
    3 * F₃ + x * F_x = 200 →
    F_x ≥ 8 →
    x = 4 := by
  sorry

end frisbee_price_problem_l1462_146247


namespace fraction_numerator_greater_than_denominator_l1462_146205

theorem fraction_numerator_greater_than_denominator
  (x : ℝ)
  (h1 : -1 ≤ x)
  (h2 : x ≤ 3)
  : 4 * x + 2 > 8 - 3 * x ↔ 6 / 7 < x ∧ x ≤ 3 :=
by sorry

end fraction_numerator_greater_than_denominator_l1462_146205


namespace min_value_theorem_l1462_146258

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 2) + 1 / (y + 2) = 1 / 3) :
  x + 2 * y ≥ 3 + 6 * Real.sqrt 2 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧
    1 / (x₀ + 2) + 1 / (y₀ + 2) = 1 / 3 ∧
    x₀ + 2 * y₀ = 3 + 6 * Real.sqrt 2 := by
sorry

end min_value_theorem_l1462_146258


namespace number_problem_l1462_146222

theorem number_problem (N : ℝ) : 
  1.15 * ((1/4) * (1/3) * (2/5) * N) = 23 → 0.5 * N = 300 := by
  sorry

end number_problem_l1462_146222


namespace park_fencing_cost_l1462_146269

/-- Proves that the cost of fencing a rectangular park with given dimensions and fencing cost is 175 rupees -/
theorem park_fencing_cost (length width area perimeter_cost : ℝ) : 
  length / width = 3 / 2 →
  length * width = 3750 →
  perimeter_cost = 0.7 →
  (2 * length + 2 * width) * perimeter_cost = 175 := by
  sorry

#check park_fencing_cost

end park_fencing_cost_l1462_146269


namespace max_value_g_range_of_a_inequality_for_f_l1462_146236

noncomputable section

def f (x : ℝ) : ℝ := Real.log x

def g (x : ℝ) : ℝ := f (x + 1) - x

theorem max_value_g :
  ∀ x > -1, g x ≤ 0 ∧ ∃ x₀ > -1, g x₀ = 0 :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≤ a * x ∧ a * x ≤ x^2 + 1) →
  (1 / Real.exp 1 ≤ a ∧ a ≤ 2) :=
sorry

theorem inequality_for_f (x₁ x₂ : ℝ) (h : x₁ > x₂ ∧ x₂ > 0) :
  (f x₁ - f x₂) / (x₁ - x₂) > (2 * x₂) / (x₁^2 + x₂^2) :=
sorry

end max_value_g_range_of_a_inequality_for_f_l1462_146236


namespace rectangle_circle_ratio_l1462_146210

theorem rectangle_circle_ratio (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 1225)
  (h2 : rectangle_area = 140)
  (h3 : rectangle_breadth = 10) :
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end rectangle_circle_ratio_l1462_146210


namespace ratio_proof_l1462_146203

theorem ratio_proof (a b c k : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c = 56) 
  (h4 : c - a = 32) (h5 : a = 3 * k) (h6 : b = 5 * k) : 
  c / b = 7 / 5 := by
  sorry

end ratio_proof_l1462_146203


namespace bert_stamp_collection_l1462_146283

theorem bert_stamp_collection (stamps_bought : ℕ) (stamps_before : ℕ) : 
  stamps_bought = 300 →
  stamps_before = stamps_bought / 2 →
  stamps_before + stamps_bought = 450 := by
sorry

end bert_stamp_collection_l1462_146283


namespace ex_factory_price_decrease_selling_price_for_profit_l1462_146215

/-- Ex-factory price in 2019 -/
def price_2019 : ℝ := 144

/-- Ex-factory price in 2021 -/
def price_2021 : ℝ := 100

/-- Current selling price -/
def current_price : ℝ := 140

/-- Current daily sales -/
def current_sales : ℝ := 20

/-- Sales increase per price reduction -/
def sales_increase : ℝ := 10

/-- Price reduction step -/
def price_reduction : ℝ := 5

/-- Target daily profit -/
def target_profit : ℝ := 1250

/-- Average yearly percentage decrease in ex-factory price -/
def avg_decrease : ℝ := 16.67

/-- Selling price for desired profit -/
def desired_price : ℝ := 125

theorem ex_factory_price_decrease :
  ∃ (x : ℝ), price_2019 * (1 - x / 100)^2 = price_2021 ∧ x = avg_decrease :=
sorry

theorem selling_price_for_profit :
  ∃ (y : ℝ),
    (y - price_2021) * (current_sales + sales_increase * (current_price - y) / price_reduction) = target_profit ∧
    y = desired_price :=
sorry

end ex_factory_price_decrease_selling_price_for_profit_l1462_146215


namespace exterior_angle_square_octagon_exterior_angle_square_octagon_proof_l1462_146219

/-- The measure of the exterior angle formed by a regular square and a regular octagon that share a common side in a coplanar configuration is 135 degrees. -/
theorem exterior_angle_square_octagon : ℝ → Prop :=
  λ angle : ℝ =>
    let square_interior_angle : ℝ := 90
    let octagon_interior_angle : ℝ := 135
    let total_angle : ℝ := 360
    angle = total_angle - (square_interior_angle + octagon_interior_angle) ∧
    angle = 135

/-- Proof of the theorem -/
theorem exterior_angle_square_octagon_proof :
  ∃ angle : ℝ, exterior_angle_square_octagon angle :=
sorry

end exterior_angle_square_octagon_exterior_angle_square_octagon_proof_l1462_146219


namespace square_root_fourth_power_l1462_146287

theorem square_root_fourth_power (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end square_root_fourth_power_l1462_146287


namespace a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l1462_146226

theorem a_gt_one_sufficient_not_necessary_for_a_squared_gt_one :
  (∀ a : ℝ, a > 1 → a^2 > 1) ∧
  (∃ a : ℝ, a^2 > 1 ∧ a ≤ 1) := by
  sorry

end a_gt_one_sufficient_not_necessary_for_a_squared_gt_one_l1462_146226


namespace simplify_complex_fraction_l1462_146291

theorem simplify_complex_fraction :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 4 + Real.sqrt 7 + Real.sqrt 9) =
  -(2 * Real.sqrt 6 - 2 * Real.sqrt 2 + 2 * Real.sqrt 14) / 5 := by
  sorry

end simplify_complex_fraction_l1462_146291


namespace inequality_holds_for_p_greater_than_three_largest_interval_l1462_146217

theorem inequality_holds_for_p_greater_than_three (p q : ℝ) (hp : p > 3) (hq : q > 0) :
  (7 * (p * q^2 + p^2 * q + 3 * q^2 + 3 * p * q)) / (p + q) > 3 * p^2 * q :=
sorry

theorem largest_interval (p q : ℝ) (hq : q > 0) :
  (∀ q > 0, (7 * (p * q^2 + p^2 * q + 3 * q^2 + 3 * p * q)) / (p + q) > 3 * p^2 * q) ↔ p > 3 :=
sorry

end inequality_holds_for_p_greater_than_three_largest_interval_l1462_146217


namespace minimize_expression_l1462_146224

theorem minimize_expression (a b : ℝ) (ha : a > 0) (hb : b > 2) (hab : a + b = 3) :
  ∃ (min_a : ℝ), min_a = 2/3 ∧
  ∀ (x : ℝ), x > 0 → x + b = 3 →
  (4/x + 1/(b-2)) ≥ (4/min_a + 1/(b-2)) :=
by sorry

end minimize_expression_l1462_146224


namespace equal_sets_implies_sum_l1462_146201

-- Define sets A and B
def A (a b : ℝ) : Set ℝ := {a, b/a, 1}
def B (a b : ℝ) : Set ℝ := {a^2, a+b, 0}

-- Theorem statement
theorem equal_sets_implies_sum (a b : ℝ) (h : A a b = B a b) :
  a^2013 + b^2014 = -1 :=
sorry

end equal_sets_implies_sum_l1462_146201


namespace coin_problem_l1462_146228

theorem coin_problem (n d h : ℕ) : 
  n + d + h = 150 →
  5*n + 10*d + 50*h = 1250 →
  ∃ (d_min d_max : ℕ), 
    (∃ (n' h' : ℕ), n' + d_min + h' = 150 ∧ 5*n' + 10*d_min + 50*h' = 1250) ∧
    (∃ (n'' h'' : ℕ), n'' + d_max + h'' = 150 ∧ 5*n'' + 10*d_max + 50*h'' = 1250) ∧
    d_max - d_min = 99 :=
by sorry

end coin_problem_l1462_146228


namespace inequality_theorem_l1462_146231

theorem inequality_theorem (a b c : ℝ) (θ : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_ineq : a * (Real.cos θ)^2 + b * (Real.sin θ)^2 < c) :
  Real.sqrt a * (Real.cos θ)^2 + Real.sqrt b * (Real.sin θ)^2 < Real.sqrt c := by
  sorry

end inequality_theorem_l1462_146231


namespace original_number_is_332_l1462_146288

/-- Given a three-digit number abc, returns the sum of abc, acb, bca, bac, cab, and cba -/
def sum_permutations (a b c : Nat) : Nat :=
  100 * a + 10 * b + c +
  100 * a + 10 * c + b +
  100 * b + 10 * c + a +
  100 * b + 10 * a + c +
  100 * c + 10 * a + b +
  100 * c + 10 * b + a

/-- The original number abc satisfies the given conditions -/
theorem original_number_is_332 : 
  ∃ (a b c : Nat), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ 
    sum_permutations a b c = 4332 ∧
    100 * a + 10 * b + c = 332 :=
by sorry

end original_number_is_332_l1462_146288


namespace cubic_root_sum_cubes_l1462_146248

theorem cubic_root_sum_cubes (r s t : ℂ) : 
  (9 * r^3 + 2023 * r + 4047 = 0) →
  (9 * s^3 + 2023 * s + 4047 = 0) →
  (9 * t^3 + 2023 * t + 4047 = 0) →
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 1349 := by
  sorry

end cubic_root_sum_cubes_l1462_146248


namespace hyperbola_eccentricity_l1462_146214

/-- Hyperbola structure -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h : c^2 = a^2 + b^2

/-- Equilateral triangle structure -/
structure EquilateralTriangle where
  side : ℝ

/-- Theorem: Eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (C : Hyperbola) (T : EquilateralTriangle)
  (h1 : T.side = 2 * C.c) -- AF₁ = AF₂ = F₁F₂ = 2c
  (h2 : ∃ (B : ℝ × ℝ), B.1^2 / C.a^2 - B.2^2 / C.b^2 = 1 ∧ 
    (B.1 + C.c)^2 + B.2^2 = (5/4 * T.side)^2) -- B is on the hyperbola and AB = 5/4 * AF₁
  : C.c / C.a = (Real.sqrt 13 + 1) / 3 := by
  sorry

end hyperbola_eccentricity_l1462_146214


namespace empty_solution_set_implies_a_range_l1462_146250

theorem empty_solution_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, ¬(|x - 1| + |x - 2| ≤ a^2 + a + 1)) → 
  -1 < a ∧ a < 0 := by
  sorry

end empty_solution_set_implies_a_range_l1462_146250


namespace intersection_of_planes_intersects_at_least_one_line_l1462_146244

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the property of two lines being skew
variable (skew : Line → Line → Prop)

-- Define the property of a line being in a plane
variable (in_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (plane_intersection : Plane → Plane → Line)

-- Define the property of a line intersecting another line
variable (intersects : Line → Line → Prop)

-- Theorem statement
theorem intersection_of_planes_intersects_at_least_one_line
  (a b l : Line) (α β : Plane)
  (h1 : skew a b)
  (h2 : in_plane a α)
  (h3 : in_plane b β)
  (h4 : plane_intersection α β = l) :
  intersects l a ∨ intersects l b :=
sorry

end intersection_of_planes_intersects_at_least_one_line_l1462_146244


namespace combined_rocket_height_l1462_146246

def first_rocket_height : ℝ := 500

theorem combined_rocket_height :
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 := by
  sorry

end combined_rocket_height_l1462_146246


namespace andy_incorrect_answers_l1462_146290

/-- Represents the number of incorrect answers for each person -/
structure TestResults where
  andy : ℕ
  beth : ℕ
  charlie : ℕ
  daniel : ℕ

/-- The theorem stating that Andy gets 14 questions wrong given the conditions -/
theorem andy_incorrect_answers (results : TestResults) : results.andy = 14 :=
  by
  have h1 : results.andy + results.beth = results.charlie + results.daniel :=
    sorry
  have h2 : results.andy + results.daniel = results.beth + results.charlie + 6 :=
    sorry
  have h3 : results.charlie = 8 :=
    sorry
  sorry

end andy_incorrect_answers_l1462_146290


namespace total_pens_l1462_146233

theorem total_pens (black_pens blue_pens : ℕ) :
  black_pens = 4 → blue_pens = 4 → black_pens + blue_pens = 8 :=
by sorry

end total_pens_l1462_146233


namespace odd_function_inequality_l1462_146261

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_inequality (f : ℝ → ℝ) (a b : ℝ) 
  (h_odd : IsOdd f) (h_ineq : f a > f b) : f (-a) < f (-b) := by
  sorry

end odd_function_inequality_l1462_146261


namespace gcd_smallest_prime_factor_subtraction_l1462_146234

theorem gcd_smallest_prime_factor_subtraction : 
  10 - (Nat.minFac (Nat.gcd 105 90)) = 7 := by
  sorry

end gcd_smallest_prime_factor_subtraction_l1462_146234


namespace hikers_meeting_point_l1462_146230

/-- Represents the distance between two hikers at any given time -/
structure HikerDistance where
  total : ℝ := 100
  from_a : ℝ
  from_b : ℝ

/-- Calculates the distance traveled by hiker A in t hours -/
def distance_a (t : ℝ) : ℝ := 5 * t

/-- Calculates the distance traveled by hiker B in t hours -/
def distance_b (t : ℝ) : ℝ := t * (4 + 0.125 * (t - 1))

/-- Represents the meeting point of the two hikers -/
def meeting_point (t : ℝ) : HikerDistance :=
  { total := 100
  , from_a := distance_a t
  , from_b := distance_b t }

/-- The time at which the hikers meet -/
def meeting_time : ℕ := 10

theorem hikers_meeting_point :
  let mp := meeting_point meeting_time
  mp.from_b - mp.from_a = 2.5 := by sorry

end hikers_meeting_point_l1462_146230


namespace snail_return_time_l1462_146257

/-- Represents the movement of a point on a plane -/
structure PointMovement where
  speed : ℝ
  turnInterval : ℝ
  turnAngle : ℝ

/-- Represents the position of the point at a given time -/
def Position := ℝ × ℝ

/-- Returns the position of the point after a given time -/
noncomputable def positionAfterTime (m : PointMovement) (t : ℝ) : Position :=
  sorry

/-- Checks if the point has returned to its starting position -/
def hasReturnedToStart (m : PointMovement) (t : ℝ) : Prop :=
  positionAfterTime m t = (0, 0)

/-- The main theorem to prove -/
theorem snail_return_time (m : PointMovement) 
    (h1 : m.speed > 0)
    (h2 : m.turnInterval = 15)
    (h3 : m.turnAngle = 90) :
    ∀ t : ℝ, hasReturnedToStart m t → ∃ n : ℕ, t = 60 * n := by
  sorry

end snail_return_time_l1462_146257


namespace hiking_team_gloves_l1462_146278

/-- The minimum number of gloves needed for a hiking team -/
def minimum_gloves (total_participants small_members medium_members large_members num_activities : ℕ) : ℕ :=
  (small_members + medium_members + large_members) * num_activities

/-- Theorem: The hiking team needs 225 gloves -/
theorem hiking_team_gloves :
  let total_participants := 75
  let small_members := 20
  let medium_members := 38
  let large_members := 17
  let num_activities := 3
  minimum_gloves total_participants small_members medium_members large_members num_activities = 225 := by
  sorry


end hiking_team_gloves_l1462_146278


namespace f_odd_and_increasing_l1462_146223

open Real

/-- The function f(x) = 3^x - 3^(-x) is odd and increasing on ℝ -/
theorem f_odd_and_increasing :
  let f : ℝ → ℝ := fun x ↦ 3^x - 3^(-x)
  (∀ x, f (-x) = -f x) ∧ StrictMono f := by sorry

end f_odd_and_increasing_l1462_146223


namespace no_solution_condition_l1462_146282

theorem no_solution_condition (m : ℝ) : 
  (∀ x : ℝ, x ≠ 1 → (m * x - 1) / (x - 1) ≠ 3) ↔ (m = 1 ∨ m = 3) := by
  sorry

end no_solution_condition_l1462_146282


namespace marriage_year_proof_l1462_146243

def year_of_marriage : ℕ := 1980
def year_child1_born : ℕ := 1982
def year_child2_born : ℕ := 1984
def reference_year : ℕ := 1986

theorem marriage_year_proof :
  (reference_year - year_child1_born) + (reference_year - year_child2_born) = reference_year - year_of_marriage :=
by sorry

end marriage_year_proof_l1462_146243


namespace product_digit_sum_base7_l1462_146254

/-- Converts a base 7 number to decimal --/
def toDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 7 --/
def toBase7 (n : ℕ) : ℕ := sorry

/-- Calculates the sum of digits of a number in base 7 --/
def sumOfDigitsBase7 (n : ℕ) : ℕ := sorry

/-- The problem statement --/
theorem product_digit_sum_base7 :
  let a := 35
  let b := 42
  sumOfDigitsBase7 (toBase7 (toDecimal a * toDecimal b)) = 18 := by sorry

end product_digit_sum_base7_l1462_146254


namespace regular_polygon_area_condition_l1462_146256

/-- A regular polygon with n sides inscribed in a circle of radius 2R has an area of 6R^2 if and only if n = 12 -/
theorem regular_polygon_area_condition (n : ℕ) (R : ℝ) (h1 : R > 0) :
  2 * n * R^2 * Real.sin (2 * Real.pi / n) = 6 * R^2 ↔ n = 12 := by
  sorry


end regular_polygon_area_condition_l1462_146256


namespace trapezoid_area_l1462_146227

theorem trapezoid_area (c : ℝ) (hc : c > 0) :
  let b := Real.sqrt c
  let shorter_base := b - 3
  let altitude := b
  let longer_base := b + 3
  let area := ((shorter_base + longer_base) / 2) * altitude
  area = c := by sorry

end trapezoid_area_l1462_146227


namespace equation_solutions_l1462_146262

theorem equation_solutions : 
  let f (x : ℝ) := (15*x - x^2) / (x + 2) * (x + (15 - x) / (x + 2))
  ∃ (s : Set ℝ), s = {12, -3, -3 + Real.sqrt 33, -3 - Real.sqrt 33} ∧ 
    ∀ x ∈ s, f x = 54 ∧ 
    ∀ y ∉ s, f y ≠ 54 := by
sorry

end equation_solutions_l1462_146262


namespace eleven_billion_scientific_notation_l1462_146263

def billion : ℕ := 10^9

theorem eleven_billion_scientific_notation : 
  11 * billion = 11 * 10^9 ∧ 11 * 10^9 = 1.1 * 10^10 :=
sorry

end eleven_billion_scientific_notation_l1462_146263


namespace intersection_sum_l1462_146202

theorem intersection_sum (a b : ℚ) : 
  (∀ x y : ℚ, x = (1/3) * y + a ↔ y = (1/3) * x + b) →
  (3 : ℚ) = (1/3) * 1 + a →
  (1 : ℚ) = (1/3) * 3 + b →
  a + b = 8/3 := by
sorry

end intersection_sum_l1462_146202


namespace chessboard_uniquely_determined_l1462_146281

/-- Represents a cell on the chessboard --/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the chessboard configuration --/
def Chessboard := Cell → Fin 64

/-- Represents a 2-cell rectangle on the chessboard --/
structure Rectangle :=
  (cell1 : Cell)
  (cell2 : Cell)

/-- Function to get the sum of numbers in a 2-cell rectangle --/
def getRectangleSum (board : Chessboard) (rect : Rectangle) : Nat :=
  (board rect.cell1).val + 1 + (board rect.cell2).val + 1

/-- Predicate to check if two cells are on the same diagonal --/
def onSameDiagonal (c1 c2 : Cell) : Prop :=
  (c1.row.val + c1.col.val = c2.row.val + c2.col.val) ∨
  (c1.row.val - c1.col.val = c2.row.val - c2.col.val)

/-- The main theorem --/
theorem chessboard_uniquely_determined
  (board : Chessboard)
  (h1 : ∃ c1 c2 : Cell, (board c1 = 0) ∧ (board c2 = 63) ∧ onSameDiagonal c1 c2)
  (h2 : ∀ rect : Rectangle, ∃ s : Nat, getRectangleSum board rect = s) :
  ∀ c : Cell, ∃! n : Fin 64, board c = n :=
sorry

end chessboard_uniquely_determined_l1462_146281


namespace square_cut_into_three_rectangles_l1462_146284

theorem square_cut_into_three_rectangles :
  ∀ (a b c d e : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 →
  a + b = 36 ∧ c + d = 36 ∧ a + c = 36 →
  a * e = b * (36 - e) ∧ c * e = d * (36 - e) →
  (∃ x y : ℝ, (x = a ∨ x = b) ∧ (y = c ∨ y = d) ∧ x + y = 36) →
  36 + e = 60 :=
by sorry

end square_cut_into_three_rectangles_l1462_146284


namespace sons_age_l1462_146216

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 25 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 23 := by
sorry

end sons_age_l1462_146216


namespace intersection_theorem_l1462_146237

def M : Set ℝ := {x | x^2 - 4 > 0}

def N : Set ℝ := {x | (1 - x) / (x - 3) > 0}

theorem intersection_theorem : N ∩ (Set.univ \ M) = {x | 1 < x ∧ x ≤ 2} := by sorry

end intersection_theorem_l1462_146237


namespace complex_equation_solution_l1462_146299

open Complex

theorem complex_equation_solution :
  let z : ℂ := (1 + I^2 + 3*(1-I)) / (2+I)
  ∀ (a b : ℝ), z^2 + a*z + b = 1 + I → a = -3 ∧ b = 4 := by
  sorry

end complex_equation_solution_l1462_146299


namespace eight_members_left_for_treasurer_l1462_146238

/-- Represents a club with members and officer positions -/
structure Club where
  totalMembers : ℕ
  presidentChosen : Bool
  secretaryChosen : Bool

/-- Function to calculate remaining members for treasurer position -/
def remainingMembersForTreasurer (club : Club) : ℕ :=
  club.totalMembers - (if club.presidentChosen then 1 else 0) - (if club.secretaryChosen then 1 else 0)

/-- Theorem stating that in a club of 10 members, after choosing president and secretary,
    there are 8 members left for treasurer position -/
theorem eight_members_left_for_treasurer (club : Club) 
    (h1 : club.totalMembers = 10)
    (h2 : club.presidentChosen = true)
    (h3 : club.secretaryChosen = true) :
  remainingMembersForTreasurer club = 8 := by
  sorry

#eval remainingMembersForTreasurer { totalMembers := 10, presidentChosen := true, secretaryChosen := true }

end eight_members_left_for_treasurer_l1462_146238


namespace sqrt_difference_approximation_l1462_146235

theorem sqrt_difference_approximation : 
  ∃ ε > 0, |Real.sqrt (49 + 81) - Real.sqrt (64 - 36) - 6.1| < ε :=
sorry

end sqrt_difference_approximation_l1462_146235


namespace repeating_decimal_ratio_l1462_146204

/-- Represents a repeating decimal with a two-digit repeating part -/
def RepeatingDecimal (a b : ℕ) : ℚ := (10 * a + b : ℚ) / 99

theorem repeating_decimal_ratio :
  (RepeatingDecimal 5 4) / (RepeatingDecimal 1 8) = 3 := by
  sorry

end repeating_decimal_ratio_l1462_146204


namespace inverse_of_congruent_area_equal_l1462_146225

-- Define the types for triangles and areas
def Triangle : Type := sorry
def Area : Type := sorry

-- Define the congruence relation for triangles
def congruent : Triangle → Triangle → Prop := sorry

-- Define the equality of areas
def area_equal : Area → Area → Prop := sorry

-- Define the area function for triangles
def triangle_area : Triangle → Area := sorry

-- Define the original proposition
def original_proposition : Prop :=
  ∀ (t1 t2 : Triangle), congruent t1 t2 → area_equal (triangle_area t1) (triangle_area t2)

-- Define the inverse proposition
def inverse_proposition : Prop :=
  ∀ (t1 t2 : Triangle), area_equal (triangle_area t1) (triangle_area t2) → congruent t1 t2

-- Theorem stating that the inverse_proposition is the correct inverse of the original_proposition
theorem inverse_of_congruent_area_equal :
  inverse_proposition = (¬original_proposition → ¬(∀ (t1 t2 : Triangle), congruent t1 t2)) := by sorry

end inverse_of_congruent_area_equal_l1462_146225


namespace exponent_division_l1462_146286

theorem exponent_division (x : ℝ) (h : x ≠ 0) : x^3 / x^2 = x := by
  sorry

end exponent_division_l1462_146286


namespace xyz_value_l1462_146229

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 35)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) : 
  x * y * z = 8 := by
sorry

end xyz_value_l1462_146229


namespace bakers_cakes_l1462_146296

/-- Baker's pastry and cake problem -/
theorem bakers_cakes (pastries_made : ℕ) (cakes_sold : ℕ) (pastries_sold : ℕ) (cakes_left : ℕ) :
  pastries_made = 61 →
  cakes_sold = 108 →
  pastries_sold = 44 →
  cakes_left = 59 →
  cakes_sold + cakes_left = 167 := by
  sorry


end bakers_cakes_l1462_146296


namespace equation_solution_l1462_146267

theorem equation_solution : ∃! x : ℝ, Real.sqrt (4 - 3 * Real.sqrt (10 - 3 * x)) = x - 2 :=
by
  -- The unique solution is x = 3
  use 3
  constructor
  · -- Prove that x = 3 satisfies the equation
    sorry
  · -- Prove that any solution must equal 3
    sorry

#check equation_solution

end equation_solution_l1462_146267


namespace smallest_of_five_consecutive_even_numbers_l1462_146208

/-- Given 5 consecutive even numbers whose sum is 240, prove that the smallest of these numbers is 44 -/
theorem smallest_of_five_consecutive_even_numbers (n : ℕ) : 
  (∃ (a b c d : ℕ), 
    a = n + 2 ∧ 
    b = n + 4 ∧ 
    c = n + 6 ∧ 
    d = n + 8 ∧ 
    n + a + b + c + d = 240 ∧ 
    Even n ∧ Even a ∧ Even b ∧ Even c ∧ Even d) → 
  n = 44 :=
by sorry

end smallest_of_five_consecutive_even_numbers_l1462_146208


namespace prob_five_odd_in_seven_rolls_prob_five_odd_in_seven_rolls_proof_l1462_146242

/-- The probability of getting exactly 5 odd numbers in 7 rolls of a fair 6-sided die -/
theorem prob_five_odd_in_seven_rolls : ℚ :=
  21 / 128

/-- A fair 6-sided die has equal probability for each outcome -/
axiom fair_die : ∀ (outcome : Fin 6), ℚ

/-- The probability of rolling an odd number on a fair 6-sided die is 1/2 -/
axiom prob_odd : (fair_die 1 + fair_die 3 + fair_die 5 : ℚ) = 1 / 2

/-- The rolls are independent -/
axiom independent_rolls : ∀ (n : ℕ), ℚ

/-- The probability of exactly k successes in n independent Bernoulli trials 
    with success probability p is given by the binomial probability formula -/
axiom binomial_probability : 
  ∀ (n k : ℕ) (p : ℚ), 
  0 ≤ p ∧ p ≤ 1 → 
  independent_rolls n = (n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem prob_five_odd_in_seven_rolls_proof : 
  prob_five_odd_in_seven_rolls = independent_rolls 7 :=
sorry

end prob_five_odd_in_seven_rolls_prob_five_odd_in_seven_rolls_proof_l1462_146242


namespace arithmetic_sequence_property_l1462_146207

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

/-- The proposition to be proved -/
theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (d : ℝ) (h_arith : arithmetic_sequence a d) (h_nonzero : ∃ n, a n ≠ 0)
  (h_eq : 2 * a 4 - (a 7)^2 + 2 * a 10 = 0) :
  a 7 = 4 * d :=
sorry

end arithmetic_sequence_property_l1462_146207


namespace rotation_result_l1462_146260

/-- Represents the four positions around a circle -/
inductive Position
| Top
| Right
| Bottom
| Left

/-- Represents the four figures on the circle -/
inductive Figure
| Triangle
| SmallerCircle
| Square
| Pentagon

/-- Initial configuration of figures on the circle -/
def initial_config : Figure → Position
| Figure.Triangle => Position.Top
| Figure.SmallerCircle => Position.Right
| Figure.Square => Position.Bottom
| Figure.Pentagon => Position.Left

/-- Rotates a position by 150 degrees clockwise -/
def rotate_150_clockwise : Position → Position
| Position.Top => Position.Left
| Position.Right => Position.Top
| Position.Bottom => Position.Right
| Position.Left => Position.Bottom

/-- Final configuration after 150 degree clockwise rotation -/
def final_config : Figure → Position :=
  λ f => rotate_150_clockwise (initial_config f)

/-- Theorem stating the final positions after rotation -/
theorem rotation_result :
  final_config Figure.Triangle = Position.Left ∧
  final_config Figure.SmallerCircle = Position.Top ∧
  final_config Figure.Square = Position.Right ∧
  final_config Figure.Pentagon = Position.Bottom :=
sorry

end rotation_result_l1462_146260


namespace not_or_implies_both_false_l1462_146232

theorem not_or_implies_both_false (p q : Prop) : 
  ¬(p ∨ q) → (¬p ∧ ¬q) := by
  sorry

end not_or_implies_both_false_l1462_146232


namespace sunglasses_profit_ratio_l1462_146275

theorem sunglasses_profit_ratio (selling_price cost_price : ℚ) (pairs_sold : ℕ) (sign_cost : ℚ) :
  selling_price = 30 →
  cost_price = 26 →
  pairs_sold = 10 →
  sign_cost = 20 →
  sign_cost / ((selling_price - cost_price) * pairs_sold) = 1 / 2 := by
sorry

end sunglasses_profit_ratio_l1462_146275


namespace sum_of_roots_is_eight_l1462_146264

/-- A function f: ℝ → ℝ that is symmetric about x = 2 and has exactly four distinct real roots -/
def SymmetricFourRootFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (2 + x) = f (2 - x)) ∧
  (∃! (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0)

/-- The sum of the four distinct real roots of a SymmetricFourRootFunction is 8 -/
theorem sum_of_roots_is_eight (f : ℝ → ℝ) (h : SymmetricFourRootFunction f) :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ f d = 0 ∧
    a + b + c + d = 8 := by
  sorry

end sum_of_roots_is_eight_l1462_146264


namespace min_distance_squared_l1462_146297

theorem min_distance_squared (a b c d : ℝ) :
  (a - 2 * Real.exp a) / b = 1 →
  (2 - c) / d = 1 →
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (x - 2 * Real.exp x) / y = 1 →
    (2 - x) / y = 1 →
    (a - x)^2 + (b - y)^2 ≥ m ∧
    m = 8 :=
sorry

end min_distance_squared_l1462_146297
