import Mathlib

namespace NUMINAMATH_CALUDE_shadow_length_proportion_l2223_222334

/-- Given two objects side by side, if one object of height 20 units casts a shadow of 10 units,
    then another object of height 40 units will cast a shadow of 20 units. -/
theorem shadow_length_proportion
  (h1 : ℝ) (s1 : ℝ) (h2 : ℝ)
  (height_shadow_1 : h1 = 20)
  (shadow_1 : s1 = 10)
  (height_2 : h2 = 40)
  (proportion : h1 / s1 = h2 / (h2 / 2)) :
  h2 / 2 = 20 := by
sorry

end NUMINAMATH_CALUDE_shadow_length_proportion_l2223_222334


namespace NUMINAMATH_CALUDE_three_positions_from_six_people_l2223_222392

/-- The number of ways to choose 3 distinct positions from a group of n people -/
def choose_three_positions (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

/-- The theorem states that choosing 3 distinct positions from 6 people results in 120 ways -/
theorem three_positions_from_six_people :
  choose_three_positions 6 = 120 := by
  sorry

end NUMINAMATH_CALUDE_three_positions_from_six_people_l2223_222392


namespace NUMINAMATH_CALUDE_quadratic_sum_l2223_222335

/-- Given a quadratic polynomial 12x^2 + 72x + 300, prove that when written
    in the form a(x+b)^2+c, where a, b, and c are constants, a + b + c = 207 -/
theorem quadratic_sum (x : ℝ) :
  ∃ (a b c : ℝ), (∀ x, 12*x^2 + 72*x + 300 = a*(x+b)^2 + c) ∧ (a + b + c = 207) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2223_222335


namespace NUMINAMATH_CALUDE_primitive_cube_root_expression_l2223_222353

/-- ω is a primitive third root of unity -/
def ω : ℂ :=
  sorry

/-- ω is a primitive third root of unity -/
axiom ω_is_primitive_cube_root : ω^3 = 1 ∧ ω ≠ 1

/-- The value of the expression (1-ω)(1-ω^2)(1-ω^4)(1-ω^8) -/
theorem primitive_cube_root_expression : (1 - ω) * (1 - ω^2) * (1 - ω^4) * (1 - ω^8) = 9 :=
  sorry

end NUMINAMATH_CALUDE_primitive_cube_root_expression_l2223_222353


namespace NUMINAMATH_CALUDE_irrational_equation_solution_l2223_222327

theorem irrational_equation_solution (a b : ℝ) : 
  Irrational a → (a * b + a - b = 1) → b = -1 := by
  sorry

end NUMINAMATH_CALUDE_irrational_equation_solution_l2223_222327


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2223_222352

theorem complex_equation_solution (a : ℝ) (i : ℂ) (h1 : i^2 = -1) (h2 : (1 + a*i)*i = 3 + i) : a = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2223_222352


namespace NUMINAMATH_CALUDE_snake_revenue_theorem_l2223_222349

/-- Calculates the total revenue from selling Jake's baby snakes --/
def calculate_snake_revenue (num_snakes : ℕ) (eggs_per_snake : ℕ) (regular_price : ℕ) (rare_multiplier : ℕ) : ℕ :=
  let total_babies := num_snakes * eggs_per_snake
  let regular_babies := total_babies - 1
  let regular_revenue := regular_babies * regular_price
  let rare_revenue := regular_price * rare_multiplier
  regular_revenue + rare_revenue

/-- Proves that the total revenue from selling Jake's baby snakes is $2250 --/
theorem snake_revenue_theorem :
  calculate_snake_revenue 3 2 250 4 = 2250 := by
  sorry

end NUMINAMATH_CALUDE_snake_revenue_theorem_l2223_222349


namespace NUMINAMATH_CALUDE_min_value_theorem_l2223_222393

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y - x*y = 0) :
  ∃ (min : ℝ), min = 5 + 2 * Real.sqrt 6 ∧ ∀ (z : ℝ), z = 3*x + 2*y → z ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2223_222393


namespace NUMINAMATH_CALUDE_one_pair_probability_l2223_222323

/-- The number of colors of socks -/
def num_colors : ℕ := 5

/-- The number of socks per color -/
def socks_per_color : ℕ := 2

/-- The total number of socks -/
def total_socks : ℕ := num_colors * socks_per_color

/-- The number of socks drawn -/
def socks_drawn : ℕ := 5

/-- The probability of drawing exactly one pair of socks with the same color -/
def prob_one_pair : ℚ := 20 / 31.5

theorem one_pair_probability : 
  (num_colors.choose 1 * socks_per_color.choose 2 * (num_colors - 1).choose 3 * (socks_per_color.choose 1)^3) / 
  (total_socks.choose socks_drawn) = prob_one_pair :=
sorry

end NUMINAMATH_CALUDE_one_pair_probability_l2223_222323


namespace NUMINAMATH_CALUDE_g_twelve_equals_thirtysix_l2223_222330

/-- The area function of a rectangle with side lengths x and x+1 -/
def f (x : ℝ) : ℝ := x * (x + 1)

/-- The function g satisfying f(g(x)) = 9x^2 + 3x -/
noncomputable def g (x : ℝ) : ℝ := 
  (- 1 + Real.sqrt (36 * x^2 + 12 * x + 1)) / 2

theorem g_twelve_equals_thirtysix : g 12 = 36 := by sorry

end NUMINAMATH_CALUDE_g_twelve_equals_thirtysix_l2223_222330


namespace NUMINAMATH_CALUDE_hcf_problem_l2223_222386

theorem hcf_problem (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  a + b = 55 →
  Nat.lcm a b = 120 →
  (1 : ℚ) / a + (1 : ℚ) / b = 11 / 120 →
  Nat.gcd a b = 5 := by
sorry

end NUMINAMATH_CALUDE_hcf_problem_l2223_222386


namespace NUMINAMATH_CALUDE_reggies_book_cost_l2223_222313

theorem reggies_book_cost (initial_amount : ℕ) (books_bought : ℕ) (amount_left : ℕ) : 
  initial_amount = 48 →
  books_bought = 5 →
  amount_left = 38 →
  (initial_amount - amount_left) / books_bought = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_reggies_book_cost_l2223_222313


namespace NUMINAMATH_CALUDE_southAmericanStampsCost_l2223_222358

/-- Represents the number of stamps for a country in a specific decade -/
structure StampCount :=
  (fifties sixties seventies eighties : ℕ)

/-- Represents a country's stamp collection -/
structure Country :=
  (name : String)
  (price : ℚ)
  (counts : StampCount)

def colombia : Country :=
  { name := "Colombia"
  , price := 3 / 100
  , counts := { fifties := 7, sixties := 6, seventies := 12, eighties := 15 } }

def argentina : Country :=
  { name := "Argentina"
  , price := 6 / 100
  , counts := { fifties := 4, sixties := 8, seventies := 10, eighties := 9 } }

def southAmericanCountries : List Country := [colombia, argentina]

def stampsBefore1980s (c : Country) : ℕ :=
  c.counts.fifties + c.counts.sixties + c.counts.seventies

def totalCost (countries : List Country) : ℚ :=
  countries.map (fun c => (stampsBefore1980s c : ℚ) * c.price) |>.sum

theorem southAmericanStampsCost :
  totalCost southAmericanCountries = 207 / 100 := by sorry

end NUMINAMATH_CALUDE_southAmericanStampsCost_l2223_222358


namespace NUMINAMATH_CALUDE_opposite_sides_range_l2223_222308

/-- Given that the origin (0, 0) and the point (1, 1) are on opposite sides of the line x + y - a = 0,
    prove that the range of values for a is (0, 2) -/
theorem opposite_sides_range (a : ℝ) : 
  (∀ (x y : ℝ), x + y - a = 0 → (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1)) →
  (0 < a ∧ a < 2) :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l2223_222308


namespace NUMINAMATH_CALUDE_red_peaches_count_l2223_222379

theorem red_peaches_count (yellow_peaches : ℕ) (red_yellow_difference : ℕ) 
  (h1 : yellow_peaches = 11)
  (h2 : red_yellow_difference = 8) :
  yellow_peaches + red_yellow_difference = 19 :=
by sorry

end NUMINAMATH_CALUDE_red_peaches_count_l2223_222379


namespace NUMINAMATH_CALUDE_sams_remaining_pennies_l2223_222338

/-- Given an initial amount of pennies and an amount spent, calculate the remaining pennies. -/
def remaining_pennies (initial : ℕ) (spent : ℕ) : ℕ := initial - spent

/-- Theorem: Sam's remaining pennies -/
theorem sams_remaining_pennies :
  let initial := 98
  let spent := 93
  remaining_pennies initial spent = 5 := by
  sorry

end NUMINAMATH_CALUDE_sams_remaining_pennies_l2223_222338


namespace NUMINAMATH_CALUDE_blackboard_erasers_l2223_222377

theorem blackboard_erasers (erasers_per_class : ℕ) 
                            (broken_erasers : ℕ) 
                            (remaining_erasers : ℕ) : 
  erasers_per_class = 3 →
  broken_erasers = 12 →
  remaining_erasers = 60 →
  (((remaining_erasers + broken_erasers) / erasers_per_class) / 3) + 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_blackboard_erasers_l2223_222377


namespace NUMINAMATH_CALUDE_intersection_distance_l2223_222305

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 36 = 1

-- Define the parabola (we don't know its exact equation, but we know it exists)
def parabola (x y : ℝ) : Prop := ∃ (a b c : ℝ), y = a * x^2 + b * x + c

-- Define the shared focus condition
def shared_focus (e p : ℝ → ℝ → Prop) : Prop :=
  ∃ (x₀ y₀ : ℝ), (∀ x y, e x y → (x - x₀)^2 + (y - y₀)^2 = 20) ∧
                 (∀ x y, p x y → (x - x₀)^2 + (y - y₀)^2 ≤ 20)

-- Define the directrix condition
def directrix_on_major_axis (p : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ x y, p x y → y = k * x^2

-- Theorem statement
theorem intersection_distance :
  ∀ (p : ℝ → ℝ → Prop),
  shared_focus ellipse p →
  directrix_on_major_axis p →
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ p x₁ y₁ ∧
    ellipse x₂ y₂ ∧ p x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4 * Real.sqrt 5 / 3)^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l2223_222305


namespace NUMINAMATH_CALUDE_system_solution_l2223_222325

theorem system_solution :
  let eq1 (x y z : ℝ) := x^3 + y^3 + z^3 = 8
  let eq2 (x y z : ℝ) := x^2 + y^2 + z^2 = 22
  let eq3 (x y z : ℝ) := 1/x + 1/y + 1/z = -z/(x*y)
  ∀ (x y z : ℝ),
    ((x = 3 ∧ y = 2 ∧ z = -3) ∨
     (x = -3 ∧ y = 2 ∧ z = 3) ∨
     (x = 2 ∧ y = 3 ∧ z = -3) ∨
     (x = 2 ∧ y = -3 ∧ z = 3)) →
    (eq1 x y z ∧ eq2 x y z ∧ eq3 x y z) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2223_222325


namespace NUMINAMATH_CALUDE_abs_inequality_equivalence_abs_sum_inequality_equivalence_l2223_222341

-- Question 1
theorem abs_inequality_equivalence (x : ℝ) :
  |x - 2| < |x + 1| ↔ x > 1/2 := by sorry

-- Question 2
theorem abs_sum_inequality_equivalence (x : ℝ) :
  |2*x + 1| + |x - 2| > 4 ↔ x < -1 ∨ x > 1 := by sorry

end NUMINAMATH_CALUDE_abs_inequality_equivalence_abs_sum_inequality_equivalence_l2223_222341


namespace NUMINAMATH_CALUDE_clock_chimes_in_day_l2223_222314

/-- Calculates the number of chimes a clock makes in a day -/
def clock_chimes : ℕ :=
  let hours_in_day : ℕ := 24
  let half_hours_in_day : ℕ := hours_in_day * 2
  let sum_of_hour_strikes : ℕ := (12 * (1 + 12)) / 2
  let total_hour_strikes : ℕ := sum_of_hour_strikes * 2
  let total_half_hour_strikes : ℕ := half_hours_in_day
  total_hour_strikes + total_half_hour_strikes

/-- Theorem stating that a clock striking hours (1 to 12) and half-hours in a 24-hour day will chime 204 times -/
theorem clock_chimes_in_day : clock_chimes = 204 := by
  sorry

end NUMINAMATH_CALUDE_clock_chimes_in_day_l2223_222314


namespace NUMINAMATH_CALUDE_no_rebus_solution_l2223_222385

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_to_nat (d : Fin 10) : ℕ := d.val

def rebus_equation (K U S Y : Fin 10) : Prop :=
  let KUSY := 1000 * (digit_to_nat K) + 100 * (digit_to_nat U) + 10 * (digit_to_nat S) + (digit_to_nat Y)
  let UKSY := 1000 * (digit_to_nat U) + 100 * (digit_to_nat K) + 10 * (digit_to_nat S) + (digit_to_nat Y)
  let UKSUS := 10000 * (digit_to_nat U) + 1000 * (digit_to_nat K) + 100 * (digit_to_nat S) + 10 * (digit_to_nat U) + (digit_to_nat S)
  is_four_digit KUSY ∧ is_four_digit UKSY ∧ is_four_digit UKSUS ∧ KUSY + UKSY = UKSUS

theorem no_rebus_solution :
  ∀ (K U S Y : Fin 10), K ≠ U ∧ K ≠ S ∧ K ≠ Y ∧ U ≠ S ∧ U ≠ Y ∧ S ≠ Y → ¬(rebus_equation K U S Y) :=
sorry

end NUMINAMATH_CALUDE_no_rebus_solution_l2223_222385


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2223_222307

theorem complex_equation_solution (z : ℂ) : z * Complex.I = -1 + (3/4) * Complex.I → z = 3/4 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2223_222307


namespace NUMINAMATH_CALUDE_annual_sales_profit_scientific_notation_l2223_222316

/-- Represents the annual sales profit in yuan -/
def annual_sales_profit : ℝ := 1.5e12

/-- Expresses the annual sales profit in scientific notation -/
def scientific_notation : ℝ := 1.5 * (10 ^ 12)

theorem annual_sales_profit_scientific_notation : 
  annual_sales_profit = scientific_notation := by sorry

end NUMINAMATH_CALUDE_annual_sales_profit_scientific_notation_l2223_222316


namespace NUMINAMATH_CALUDE_parabola_coordinate_transform_l2223_222348

/-- Given a parabola y = 2x² in the original coordinate system,
    prove that its equation in a new coordinate system where
    the x-axis is moved up by 2 units and the y-axis is moved
    right by 2 units is y = 2(x+2)² - 2 -/
theorem parabola_coordinate_transform :
  ∀ (x y : ℝ),
  (y = 2 * x^2) →
  (∃ (x' y' : ℝ),
    x' = x + 2 ∧
    y' = y - 2 ∧
    y' = 2 * (x' - 2)^2 - 2) :=
by sorry

end NUMINAMATH_CALUDE_parabola_coordinate_transform_l2223_222348


namespace NUMINAMATH_CALUDE_market_equilibrium_and_subsidy_effect_l2223_222374

/-- Supply function -/
def supply (p : ℝ) : ℝ := 2 + 8 * p

/-- Demand function (to be derived) -/
def demand (p : ℝ) : ℝ := 12 - 2 * p

/-- Equilibrium price -/
def equilibrium_price : ℝ := 1

/-- Equilibrium quantity -/
def equilibrium_quantity : ℝ := 10

/-- Subsidy amount -/
def subsidy : ℝ := 1

/-- New supply function after subsidy -/
def new_supply (p : ℝ) : ℝ := supply (p + subsidy)

/-- New equilibrium price after subsidy -/
def new_equilibrium_price : ℝ := 0.2

/-- New equilibrium quantity after subsidy -/
def new_equilibrium_quantity : ℝ := 11.6

theorem market_equilibrium_and_subsidy_effect :
  (demand 2 = 8) ∧
  (demand 3 = 6) ∧
  (supply equilibrium_price = demand equilibrium_price) ∧
  (supply equilibrium_price = equilibrium_quantity) ∧
  (new_supply new_equilibrium_price = demand new_equilibrium_price) ∧
  (new_equilibrium_quantity - equilibrium_quantity = 1.6) := by
  sorry

end NUMINAMATH_CALUDE_market_equilibrium_and_subsidy_effect_l2223_222374


namespace NUMINAMATH_CALUDE_prime_divides_square_implies_divides_l2223_222366

theorem prime_divides_square_implies_divides (p n : ℕ) : 
  Prime p → (p ∣ n^2) → (p ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_square_implies_divides_l2223_222366


namespace NUMINAMATH_CALUDE_sum_x_y_given_equations_l2223_222328

theorem sum_x_y_given_equations (x y : ℝ) 
  (eq1 : 2 * |x| + 3 * x + 3 * y = 30)
  (eq2 : 3 * x + 2 * |y| - 2 * y = 36) : 
  x + y = 8512 / 2513 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_given_equations_l2223_222328


namespace NUMINAMATH_CALUDE_total_deposit_l2223_222324

def mark_deposit : ℕ := 88
def bryan_deposit : ℕ := 5 * mark_deposit - 40

theorem total_deposit : mark_deposit + bryan_deposit = 488 := by
  sorry

end NUMINAMATH_CALUDE_total_deposit_l2223_222324


namespace NUMINAMATH_CALUDE_basketball_shots_improvement_l2223_222387

theorem basketball_shots_improvement (initial_shots : ℕ) (initial_success_rate : ℚ)
  (additional_shots : ℕ) (new_success_rate : ℚ) :
  initial_shots = 30 →
  initial_success_rate = 60 / 100 →
  additional_shots = 10 →
  new_success_rate = 62 / 100 →
  (↑(initial_shots * initial_success_rate.num / initial_success_rate.den +
    (new_success_rate * ↑(initial_shots + additional_shots)).num / (new_success_rate * ↑(initial_shots + additional_shots)).den -
    (initial_success_rate * ↑initial_shots).num / (initial_success_rate * ↑initial_shots).den) : ℚ) = 7 :=
by
  sorry

#check basketball_shots_improvement

end NUMINAMATH_CALUDE_basketball_shots_improvement_l2223_222387


namespace NUMINAMATH_CALUDE_polynomial_factorization_l2223_222310

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 4*x^2 - 1 = (x - 1)^3 * (x + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l2223_222310


namespace NUMINAMATH_CALUDE_angle_in_first_quadrant_l2223_222333

theorem angle_in_first_quadrant (θ : Real) (h : θ = -5) : 
  ∃ n : ℤ, θ + 2 * π * n ∈ Set.Ioo 0 (π / 2) :=
by sorry

end NUMINAMATH_CALUDE_angle_in_first_quadrant_l2223_222333


namespace NUMINAMATH_CALUDE_set_operations_l2223_222394

/-- Given a universal set and two of its subsets, prove various set operations -/
theorem set_operations (U A B : Set ℕ) 
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 3})
  (hB : B = {2, 5}) :
  (U \ A = {2, 4, 5}) ∧
  (A ∩ B = ∅) ∧
  (A ∪ B = {1, 2, 3, 5}) ∧
  ((U \ A) ∪ (U \ B) = {1, 2, 3, 4, 5}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l2223_222394


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2223_222395

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℤ, x^3 < 1)) ↔ (∃ x : ℤ, x^3 ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2223_222395


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l2223_222396

theorem degree_to_radian_conversion (π : Real) (h : π = Real.pi) :
  120 * (π / 180) = 2 * π / 3 := by sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l2223_222396


namespace NUMINAMATH_CALUDE_solve_complex_equation_l2223_222378

theorem solve_complex_equation :
  let z : ℂ := 10 + 180 * Complex.I
  let equation := fun (x : ℂ) ↦ 7 * x - z = 15000
  ∃ (x : ℂ), equation x ∧ x = 2144 + (2 / 7) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_solve_complex_equation_l2223_222378


namespace NUMINAMATH_CALUDE_root_magnitude_bound_l2223_222344

theorem root_magnitude_bound (p : ℝ) (r₁ r₂ : ℝ) 
  (h_distinct : r₁ ≠ r₂)
  (h_root₁ : r₁^2 + p*r₁ - 12 = 0)
  (h_root₂ : r₂^2 + p*r₂ - 12 = 0) :
  abs r₁ > 3 ∨ abs r₂ > 3 := by
sorry

end NUMINAMATH_CALUDE_root_magnitude_bound_l2223_222344


namespace NUMINAMATH_CALUDE_equation_to_general_form_l2223_222345

theorem equation_to_general_form :
  ∀ x : ℝ, 5 * x^2 - 2 * x = 3 * (x + 1) ↔ 5 * x^2 - 5 * x - 3 = 0 := by
sorry

end NUMINAMATH_CALUDE_equation_to_general_form_l2223_222345


namespace NUMINAMATH_CALUDE_duck_cow_legs_heads_l2223_222354

theorem duck_cow_legs_heads :
  ∀ (D : ℕ),
  let C : ℕ := 16
  let H : ℕ := D + C
  let L : ℕ := 2 * D + 4 * C
  L - 2 * H = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_duck_cow_legs_heads_l2223_222354


namespace NUMINAMATH_CALUDE_coin_division_theorem_l2223_222365

/-- Represents a collection of coins with their values -/
structure CoinCollection where
  num_coins : Nat
  coin_values : List Nat
  total_value : Nat

/-- Predicate to check if a coin collection can be divided into three equal groups -/
def can_divide_equally (cc : CoinCollection) : Prop :=
  ∃ (g1 g2 g3 : List Nat),
    g1 ++ g2 ++ g3 = cc.coin_values ∧
    g1.sum = g2.sum ∧ g2.sum = g3.sum

/-- Theorem stating that a specific coin collection can always be divided equally -/
theorem coin_division_theorem (cc : CoinCollection) 
    (h1 : cc.num_coins = 241)
    (h2 : cc.total_value = 360)
    (h3 : ∀ c ∈ cc.coin_values, c > 0)
    (h4 : cc.coin_values.length = cc.num_coins)
    (h5 : cc.coin_values.sum = cc.total_value) :
  can_divide_equally cc :=
sorry

end NUMINAMATH_CALUDE_coin_division_theorem_l2223_222365


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_l2223_222380

theorem rectangular_parallelepiped_surface_area 
  (x y z : ℝ) 
  (h1 : (x + 1) * (y + 1) * (z + 1) - x * y * z = 18)
  (h2 : 2 * ((x + 1) * (y + 1) + (y + 1) * (z + 1) + (z + 1) * (x + 1)) - 2 * (x * y + x * z + y * z) = 30) :
  2 * (x * y + x * z + y * z) = 22 := by
sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_surface_area_l2223_222380


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2223_222339

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 1) → x ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2223_222339


namespace NUMINAMATH_CALUDE_max_value_of_f_l2223_222367

def f (x : ℝ) := -x^2 + 6*x - 10

theorem max_value_of_f :
  ∃ (m : ℝ), m = -1 ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 4 → f x ≤ m) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 4 ∧ f x = m) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l2223_222367


namespace NUMINAMATH_CALUDE_sum_ab_equals_four_l2223_222317

theorem sum_ab_equals_four (a b c d : ℤ) 
  (h1 : b + c = 7) 
  (h2 : c + d = 5) 
  (h3 : a + d = 2) : 
  a + b = 4 := by
sorry

end NUMINAMATH_CALUDE_sum_ab_equals_four_l2223_222317


namespace NUMINAMATH_CALUDE_strawberries_problem_l2223_222364

/-- Converts kilograms to grams -/
def kg_to_g (kg : ℕ) : ℕ := kg * 1000

/-- Calculates the remaining strawberries in grams -/
def remaining_strawberries (initial_kg initial_g given_kg given_g : ℕ) : ℕ :=
  (kg_to_g initial_kg + initial_g) - (kg_to_g given_kg + given_g)

theorem strawberries_problem :
  remaining_strawberries 3 300 1 900 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_strawberries_problem_l2223_222364


namespace NUMINAMATH_CALUDE_sqrt500_approx_l2223_222306

/-- Approximate value of √5 -/
def sqrt5_approx : ℝ := 2.236

/-- Theorem stating that √500 is approximately 22.36 -/
theorem sqrt500_approx : ‖Real.sqrt 500 - 22.36‖ < 0.01 :=
  sorry

end NUMINAMATH_CALUDE_sqrt500_approx_l2223_222306


namespace NUMINAMATH_CALUDE_sector_area_l2223_222388

/-- Given a sector with central angle 135° and arc length 3π cm, its area is 6π cm² -/
theorem sector_area (θ : ℝ) (arc_length : ℝ) (area : ℝ) :
  θ = 135 ∧ arc_length = 3 * Real.pi → area = 6 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2223_222388


namespace NUMINAMATH_CALUDE_band_encore_problem_l2223_222362

def band_encore_songs (total_songs : ℕ) (first_set : ℕ) (second_set : ℕ) (avg_third_fourth : ℕ) : ℕ :=
  total_songs - (first_set + second_set + 2 * avg_third_fourth)

theorem band_encore_problem :
  band_encore_songs 30 5 7 8 = 2 := by
  sorry

end NUMINAMATH_CALUDE_band_encore_problem_l2223_222362


namespace NUMINAMATH_CALUDE_final_white_pieces_l2223_222321

/-- Recursively calculates the number of remaining white pieces after each round of removal -/
def remainingWhitePieces : ℕ → ℕ
| 0 => 1990  -- Initial number of white pieces
| (n + 1) =>
  let previous := remainingWhitePieces n
  if previous % 2 = 0 then
    previous / 2
  else
    (previous + 1) / 2

/-- Theorem stating that after the removal process, 124 white pieces remain -/
theorem final_white_pieces :
  ∃ n : ℕ, remainingWhitePieces n = 124 ∧ ∀ m > n, remainingWhitePieces m = 124 :=
sorry

end NUMINAMATH_CALUDE_final_white_pieces_l2223_222321


namespace NUMINAMATH_CALUDE_joans_remaining_apples_l2223_222361

/-- Given that Joan picked a certain number of apples and gave some away,
    this theorem proves how many apples Joan has left. -/
theorem joans_remaining_apples 
  (apples_picked : ℕ) 
  (apples_given_away : ℕ) 
  (h1 : apples_picked = 43)
  (h2 : apples_given_away = 27) :
  apples_picked - apples_given_away = 16 := by
sorry

end NUMINAMATH_CALUDE_joans_remaining_apples_l2223_222361


namespace NUMINAMATH_CALUDE_stool_height_is_30_l2223_222331

/-- The height of the stool Alice needs to reach the light bulb -/
def stool_height : ℝ :=
  let ceiling_height : ℝ := 250  -- in cm
  let light_bulb_below_ceiling : ℝ := 15  -- in cm
  let alice_height : ℝ := 155  -- in cm
  let alice_reach : ℝ := 50  -- in cm
  let light_bulb_height : ℝ := ceiling_height - light_bulb_below_ceiling
  let alice_total_reach : ℝ := alice_height + alice_reach
  light_bulb_height - alice_total_reach

theorem stool_height_is_30 : stool_height = 30 := by
  sorry

end NUMINAMATH_CALUDE_stool_height_is_30_l2223_222331


namespace NUMINAMATH_CALUDE_factorization_equality_l2223_222342

theorem factorization_equality (a : ℝ) : (a + 2) * (a - 2) - 3 * a = (a - 4) * (a + 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2223_222342


namespace NUMINAMATH_CALUDE_isabellas_hair_growth_l2223_222371

/-- Given Isabella's initial and final hair lengths, prove the amount of hair growth. -/
theorem isabellas_hair_growth 
  (initial_length : ℝ) 
  (final_length : ℝ) 
  (h1 : initial_length = 18) 
  (h2 : final_length = 24) : 
  final_length - initial_length = 6 := by
sorry

end NUMINAMATH_CALUDE_isabellas_hair_growth_l2223_222371


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2223_222309

theorem sum_of_squares_16_to_30 (sum_1_to_15 : ℕ) (sum_1_to_30 : ℕ) : 
  sum_1_to_15 = 1240 → 
  sum_1_to_30 = (30 * 31 * 61) / 6 →
  sum_1_to_30 - sum_1_to_15 = 8215 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2223_222309


namespace NUMINAMATH_CALUDE_least_four_digit_divisible_by_digits_l2223_222370

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def all_digits_different (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 4 ∧ digits.toFinset.card = 4

def divisible_by_nonzero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0 → n % d = 0

theorem least_four_digit_divisible_by_digits :
  ∃ (n : ℕ), is_four_digit n ∧
             all_digits_different n ∧
             divisible_by_nonzero_digits n ∧
             (∀ m : ℕ, is_four_digit m ∧
                       all_digits_different m ∧
                       divisible_by_nonzero_digits m →
                       n ≤ m) ∧
             n = 1240 :=
  sorry

end NUMINAMATH_CALUDE_least_four_digit_divisible_by_digits_l2223_222370


namespace NUMINAMATH_CALUDE_security_to_bag_check_ratio_l2223_222320

def total_time : ℕ := 180
def uber_to_house : ℕ := 10
def check_bag_time : ℕ := 15
def wait_for_boarding : ℕ := 20

def uber_to_airport : ℕ := 5 * uber_to_house
def wait_for_takeoff : ℕ := 2 * wait_for_boarding

def known_time : ℕ := uber_to_house + uber_to_airport + check_bag_time + wait_for_boarding + wait_for_takeoff
def security_time : ℕ := total_time - known_time

theorem security_to_bag_check_ratio :
  security_time / check_bag_time = 3 ∧ security_time % check_bag_time = 0 :=
by sorry

end NUMINAMATH_CALUDE_security_to_bag_check_ratio_l2223_222320


namespace NUMINAMATH_CALUDE_thirty_seventh_digit_of_1_17_l2223_222391

-- Define the decimal representation of 1/17
def decimal_rep_1_17 : ℕ → ℕ
| 0 => 0
| 1 => 5
| 2 => 8
| 3 => 8
| 4 => 2
| 5 => 3
| 6 => 5
| 7 => 2
| 8 => 9
| 9 => 4
| 10 => 1
| 11 => 1
| 12 => 7
| 13 => 6
| 14 => 4
| 15 => 7
| n + 16 => decimal_rep_1_17 n

-- Define the period of the decimal representation
def period : ℕ := 16

-- Theorem statement
theorem thirty_seventh_digit_of_1_17 :
  decimal_rep_1_17 ((37 - 1) % period) = 8 := by
  sorry

end NUMINAMATH_CALUDE_thirty_seventh_digit_of_1_17_l2223_222391


namespace NUMINAMATH_CALUDE_georges_walk_to_school_l2223_222336

/-- Proves that given the conditions of George's walk to school, 
    the speed required for the second mile to arrive on time is 6 mph. -/
theorem georges_walk_to_school (total_distance : ℝ) (normal_speed : ℝ) 
  (normal_time : ℝ) (first_mile_speed : ℝ) :
  total_distance = 2 →
  normal_speed = 4 →
  normal_time = 0.5 →
  first_mile_speed = 3 →
  ∃ (second_mile_speed : ℝ),
    second_mile_speed = 6 ∧
    (1 / first_mile_speed + 1 / second_mile_speed = normal_time) :=
by sorry

end NUMINAMATH_CALUDE_georges_walk_to_school_l2223_222336


namespace NUMINAMATH_CALUDE_intersections_for_12_6_l2223_222383

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections (x_points y_points : ℕ) : ℕ :=
  Nat.choose x_points 2 * Nat.choose y_points 2

/-- Theorem stating the maximum number of intersection points for 12 x-axis points and 6 y-axis points -/
theorem intersections_for_12_6 :
  max_intersections 12 6 = 990 := by
  sorry

end NUMINAMATH_CALUDE_intersections_for_12_6_l2223_222383


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2223_222340

-- Define rational numbers
def IsRational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- Define the statements p and q
def p (x : ℝ) : Prop := IsRational (x^2)
def q (x : ℝ) : Prop := IsRational x

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient :
  (∀ x : ℝ, q x → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l2223_222340


namespace NUMINAMATH_CALUDE_jacob_excess_calories_l2223_222384

def jacob_calorie_problem (calorie_goal : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) : Prop :=
  calorie_goal < 1800 ∧
  breakfast = 400 ∧
  lunch = 900 ∧
  dinner = 1100 ∧
  (breakfast + lunch + dinner) - calorie_goal = 600

theorem jacob_excess_calories :
  ∃ (calorie_goal : ℕ), jacob_calorie_problem calorie_goal 400 900 1100 :=
by
  sorry

end NUMINAMATH_CALUDE_jacob_excess_calories_l2223_222384


namespace NUMINAMATH_CALUDE_max_infected_population_l2223_222373

/-- A graph representing the CMI infection spread --/
structure InfectionGraph where
  V : Type*  -- Set of vertices (people)
  E : V → V → Prop  -- Edge relation (friendship)
  degree_bound : ∀ v : V, (∃ n : ℕ, n ≤ 3 ∧ (∃ (l : List V), l.length = n ∧ (∀ u ∈ l, E v u)))

/-- The infection state of the graph over time --/
def InfectionState (G : InfectionGraph) := ℕ → G.V → Prop

/-- The initial infection state --/
def initial_infection (G : InfectionGraph) (S : InfectionState G) : Prop :=
  ∃ (infected : Finset G.V), infected.card = 2023 ∧ 
    ∀ v, S 0 v ↔ v ∈ infected

/-- The infection spread rule --/
def infection_rule (G : InfectionGraph) (S : InfectionState G) : Prop :=
  ∀ t v, S (t + 1) v ↔ 
    S t v ∨ (∃ (u₁ u₂ : G.V), u₁ ≠ u₂ ∧ G.E v u₁ ∧ G.E v u₂ ∧ S t u₁ ∧ S t u₂)

/-- Everyone eventually gets infected --/
def all_infected (G : InfectionGraph) (S : InfectionState G) : Prop :=
  ∀ v, ∃ t, S t v

/-- The main theorem --/
theorem max_infected_population (G : InfectionGraph) (S : InfectionState G) 
  (h_initial : initial_infection G S)
  (h_rule : infection_rule G S)
  (h_all : all_infected G S) :
  ∀ n : ℕ, (∃ f : G.V → Fin n, Function.Injective f) → n ≤ 4043 := by
  sorry

end NUMINAMATH_CALUDE_max_infected_population_l2223_222373


namespace NUMINAMATH_CALUDE_quadratic_solution_l2223_222355

/-- The quadratic equation ax^2 + 10x + c = 0 has exactly one solution, a + c = 12, and a < c -/
def quadratic_equation (a c : ℝ) : Prop :=
  ∃! x, a * x^2 + 10 * x + c = 0 ∧ a + c = 12 ∧ a < c

/-- The solution to the quadratic equation is (6-√11, 6+√11) -/
theorem quadratic_solution :
  ∀ a c : ℝ, quadratic_equation a c → a = 6 - Real.sqrt 11 ∧ c = 6 + Real.sqrt 11 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_l2223_222355


namespace NUMINAMATH_CALUDE_second_caterer_cheaper_l2223_222301

/-- The pricing function for the first caterer -/
def first_caterer (x : ℕ) : ℚ := 150 + 18 * x

/-- The pricing function for the second caterer -/
def second_caterer (x : ℕ) : ℚ := 250 + 15 * x

/-- The least number of people for which the second caterer is cheaper -/
def least_people : ℕ := 34

theorem second_caterer_cheaper :
  (∀ n : ℕ, n ≥ least_people → second_caterer n < first_caterer n) ∧
  (∀ n : ℕ, n < least_people → second_caterer n ≥ first_caterer n) := by
  sorry

end NUMINAMATH_CALUDE_second_caterer_cheaper_l2223_222301


namespace NUMINAMATH_CALUDE_a_range_l2223_222304

def A (a : ℝ) : Set ℝ := {x | -3 ≤ x ∧ x ≤ a}

def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 3*x + 10}

def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = 5 - x}

theorem a_range (a : ℝ) : B a ∩ C a = C a → a ∈ Set.Icc (-2/3) 4 := by
  sorry

end NUMINAMATH_CALUDE_a_range_l2223_222304


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2223_222347

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, focal length 10, and point (2,1) on its asymptote, 
    prove that its equation is x²/20 - y²/5 = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (a^2 + b^2 = 25) → (2 * b / a = 1) → 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 20 - y^2 / 5 = 1) := by
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2223_222347


namespace NUMINAMATH_CALUDE_population_growth_inequality_l2223_222351

theorem population_growth_inequality (m n p : ℝ) 
  (h1 : (1 + p / 100)^2 = (1 + m / 100) * (1 + n / 100)) : 
  p ≤ (m + n) / 2 := by
sorry

end NUMINAMATH_CALUDE_population_growth_inequality_l2223_222351


namespace NUMINAMATH_CALUDE_inequality_count_l2223_222397

theorem inequality_count (x y a b : ℝ) (hx : |x| > a) (hy : |y| > b) :
  ∃! n : ℕ, n = (Bool.toNat (|x + y| > a + b)) +
               (Bool.toNat (|x - y| > |a - b|)) +
               (Bool.toNat (x * y > a * b)) +
               (Bool.toNat (|x / y| > |a / b|)) ∧
  n = 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_count_l2223_222397


namespace NUMINAMATH_CALUDE_ice_cream_scoop_permutations_l2223_222329

theorem ice_cream_scoop_permutations :
  (Finset.range 5).card.factorial = 120 := by sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_permutations_l2223_222329


namespace NUMINAMATH_CALUDE_seeds_in_first_plot_l2223_222343

/-- The number of seeds planted in the first plot -/
def seeds_first_plot : ℕ := sorry

/-- The number of seeds planted in the second plot -/
def seeds_second_plot : ℕ := 200

/-- The percentage of seeds that germinated in the first plot -/
def germination_rate_first : ℚ := 20 / 100

/-- The percentage of seeds that germinated in the second plot -/
def germination_rate_second : ℚ := 35 / 100

/-- The percentage of total seeds that germinated -/
def total_germination_rate : ℚ := 26 / 100

/-- Theorem stating that the number of seeds in the first plot is 300 -/
theorem seeds_in_first_plot :
  (seeds_first_plot : ℚ) * germination_rate_first + 
  (seeds_second_plot : ℚ) * germination_rate_second = 
  total_germination_rate * ((seeds_first_plot : ℚ) + seeds_second_plot) ∧
  seeds_first_plot = 300 := by sorry

end NUMINAMATH_CALUDE_seeds_in_first_plot_l2223_222343


namespace NUMINAMATH_CALUDE_negation_equivalence_l2223_222350

-- Define the predicate P
def P (k : ℝ) : Prop := ∃ x y : ℝ, y = k * x + 1 ∧ x^2 + y^2 = 2

-- State the theorem
theorem negation_equivalence :
  (¬ ∀ k : ℝ, P k) ↔ (∃ k₀ : ℝ, ¬ P k₀) :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2223_222350


namespace NUMINAMATH_CALUDE_remaining_wallpaper_time_l2223_222318

/-- Time to remove wallpaper from one wall -/
def time_per_wall : ℕ := 2

/-- Number of walls in dining room -/
def dining_walls : ℕ := 4

/-- Number of walls in living room -/
def living_walls : ℕ := 4

/-- Time already spent removing wallpaper -/
def time_spent : ℕ := 2

/-- Theorem: The remaining time to remove wallpaper is 14 hours -/
theorem remaining_wallpaper_time :
  (time_per_wall * dining_walls + time_per_wall * living_walls) - time_spent = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_wallpaper_time_l2223_222318


namespace NUMINAMATH_CALUDE_unique_solution_abc_l2223_222346

theorem unique_solution_abc : 
  ∀ a b c : ℕ+, 
    (3 * a * b * c + 11 * (a + b + c) = 6 * (a * b + b * c + a * c) + 18) → 
    (a = 1 ∧ b = 2 ∧ c = 3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_abc_l2223_222346


namespace NUMINAMATH_CALUDE_tan_alpha_eq_neg_one_third_l2223_222319

theorem tan_alpha_eq_neg_one_third (α : ℝ) 
  (h : (Real.cos (π/4 - α)) / (Real.cos (π/4 + α)) = 1/2) : 
  Real.tan α = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_eq_neg_one_third_l2223_222319


namespace NUMINAMATH_CALUDE_pen_color_theorem_l2223_222363

-- Define the universe of pens
variable (Pen : Type)

-- Define the property of being in the box
variable (inBox : Pen → Prop)

-- Define the property of being blue
variable (isBlue : Pen → Prop)

-- Theorem statement
theorem pen_color_theorem :
  (¬ ∀ p : Pen, inBox p → isBlue p) →
  ((∃ p : Pen, inBox p ∧ ¬ isBlue p) ∧
   (¬ ∀ p : Pen, inBox p → isBlue p)) :=
by sorry

end NUMINAMATH_CALUDE_pen_color_theorem_l2223_222363


namespace NUMINAMATH_CALUDE_prime_equation_solution_l2223_222359

theorem prime_equation_solution :
  ∀ p q : ℕ,
  Prime p → Prime q →
  p^2 - 6*p*q + q^2 + 3*q - 1 = 0 →
  (p = 17 ∧ q = 3) :=
by sorry

end NUMINAMATH_CALUDE_prime_equation_solution_l2223_222359


namespace NUMINAMATH_CALUDE_power_four_mod_nine_l2223_222322

theorem power_four_mod_nine : 4^3023 % 9 = 7 := by sorry

end NUMINAMATH_CALUDE_power_four_mod_nine_l2223_222322


namespace NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2223_222315

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x + 3|

-- Part 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x ≥ 6} = {x : ℝ | x ≤ -4 ∨ x ≥ 2} :=
sorry

-- Part 2
theorem range_of_a_part2 :
  {a : ℝ | ∀ x, f a x > -a} = {a : ℝ | a > -3/2} :=
sorry

end NUMINAMATH_CALUDE_solution_set_part1_range_of_a_part2_l2223_222315


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l2223_222337

theorem perfect_square_trinomial (x : ℝ) : x^2 - 6*x + 9 = (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l2223_222337


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2223_222357

theorem problem_1 : |-5| + (3 - Real.sqrt 2) ^ 0 - 2 * Real.tan (π / 4) = 4 := by sorry

theorem problem_2 (a : ℝ) (h1 : a ≠ 3) (h2 : a ≠ -3) : 
  (a / (a^2 - 9)) / (1 + 3 / (a - 3)) = 1 / (a + 3) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2223_222357


namespace NUMINAMATH_CALUDE_coefficient_x2y2_l2223_222390

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (1+x)^7(1+y)^4
def expansion : ℕ → ℕ → ℕ := sorry

-- Theorem statement
theorem coefficient_x2y2 : expansion 2 2 = 126 := by sorry

end NUMINAMATH_CALUDE_coefficient_x2y2_l2223_222390


namespace NUMINAMATH_CALUDE_axels_alphabets_l2223_222369

theorem axels_alphabets (total_alphabets : ℕ) (repetitions : ℕ) (different_alphabets : ℕ) : 
  total_alphabets = 10 ∧ repetitions = 2 → different_alphabets = 5 :=
by sorry

end NUMINAMATH_CALUDE_axels_alphabets_l2223_222369


namespace NUMINAMATH_CALUDE_perpendicular_and_parallel_properties_l2223_222332

-- Define the necessary structures
structure EuclideanPlane where
  -- Add necessary axioms for Euclidean plane

structure Line where
  -- Add necessary properties for a line

structure Point where
  -- Add necessary properties for a point

-- Define the relationships
def isOn (p : Point) (l : Line) : Prop := sorry

def isPerpendicular (l1 l2 : Line) : Prop := sorry

def isParallel (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_and_parallel_properties 
  (plane : EuclideanPlane) (l : Line) : 
  (∀ (p : Point), isOn p l → ∃ (perps : Set Line), 
    (∀ (l' : Line), l' ∈ perps ↔ isPerpendicular l' l ∧ isOn p l') ∧ 
    Set.Infinite perps) ∧
  (∀ (p : Point), ¬isOn p l → 
    ∃! (l' : Line), isPerpendicular l' l ∧ isOn p l') ∧
  (∀ (p : Point), ¬isOn p l → 
    ∃! (l' : Line), isParallel l' l ∧ isOn p l') := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_and_parallel_properties_l2223_222332


namespace NUMINAMATH_CALUDE_cell_phone_call_cost_l2223_222389

/-- Given a constant rate per minute where a 3-minute call costs $0.18, 
    prove that a 10-minute call will cost $0.60. -/
theorem cell_phone_call_cost 
  (rate : ℝ) 
  (h1 : rate * 3 = 0.18) -- Cost of 3-minute call
  : rate * 10 = 0.60 := by 
  sorry

end NUMINAMATH_CALUDE_cell_phone_call_cost_l2223_222389


namespace NUMINAMATH_CALUDE_bridget_apples_l2223_222311

theorem bridget_apples (x : ℕ) : 
  (x / 2 - (x / 2) / 3 = 5) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_bridget_apples_l2223_222311


namespace NUMINAMATH_CALUDE_dart_score_proof_l2223_222302

def bullseye_points : ℕ := 50
def missed_points : ℕ := 0
def third_dart_points : ℕ := bullseye_points / 2

def total_score : ℕ := bullseye_points + missed_points + third_dart_points

theorem dart_score_proof : total_score = 75 := by
  sorry

end NUMINAMATH_CALUDE_dart_score_proof_l2223_222302


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l2223_222326

-- Define the repeating decimals
def repeating_decimal_72 : ℚ := 8/11
def repeating_decimal_124 : ℚ := 41/33

-- State the theorem
theorem repeating_decimal_division :
  repeating_decimal_72 / repeating_decimal_124 = 264/451 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l2223_222326


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2223_222376

theorem complex_power_magnitude : Complex.abs ((2 - 2 * Complex.I) ^ 6) = 512 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2223_222376


namespace NUMINAMATH_CALUDE_simplify_expression_l2223_222399

theorem simplify_expression (m : ℝ) (h : m ≠ 3) : 
  (m^2 / (m-3)) + (9 / (3-m)) = m + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2223_222399


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2223_222368

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | Real.log (x - 2) < 1}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | -1 < x ∧ x < 12} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2223_222368


namespace NUMINAMATH_CALUDE_quadratic_trinomial_equality_l2223_222356

/-- A quadratic trinomial function -/
def quadratic_trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_trinomial_equality 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x, quadratic_trinomial a b c (3.8 * x - 1) = quadratic_trinomial a b c (-3.8 * x)) →
  (∀ x, quadratic_trinomial a b c x = quadratic_trinomial a a c x) :=
sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_equality_l2223_222356


namespace NUMINAMATH_CALUDE_constant_term_value_l2223_222300

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the expansion of (1+2x)^n
def expansion (n : ℕ) (x : ℝ) : ℝ := sorry

-- Define the coefficient of the r-th term in the expansion
def coefficient (n r : ℕ) : ℝ := sorry

-- Define the condition that only the fourth term has the maximum coefficient
def fourth_term_max (n : ℕ) : Prop :=
  ∀ r, r ≠ 3 → coefficient n r ≤ coefficient n 3

-- Main theorem
theorem constant_term_value (n : ℕ) :
  fourth_term_max n →
  (expansion n 0 + coefficient n 2) = 61 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_value_l2223_222300


namespace NUMINAMATH_CALUDE_six_lines_six_intersections_l2223_222381

/-- A point in the plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- Check if a point lies on a line -/
def Point.onLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines intersect -/
def Line.intersect (l1 l2 : Line) : Prop :=
  l1.a * l2.b ≠ l1.b * l2.a

/-- A configuration of six lines -/
structure SixLineConfig :=
  (lines : Fin 6 → Line)

/-- Count the number of intersection points in a configuration -/
def SixLineConfig.intersectionCount (config : SixLineConfig) : ℕ :=
  sorry

/-- Theorem: There exists a configuration of six lines with exactly six intersection points -/
theorem six_lines_six_intersections :
  ∃ (config : SixLineConfig), config.intersectionCount = 6 :=
sorry

end NUMINAMATH_CALUDE_six_lines_six_intersections_l2223_222381


namespace NUMINAMATH_CALUDE_root_conditions_imply_m_range_l2223_222382

/-- A quadratic function f(x) with parameter m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x^2 + m * x + (2 * m + 1)

/-- The theorem stating the range of m given the root conditions -/
theorem root_conditions_imply_m_range :
  ∀ m : ℝ,
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ f m r₁ = 0 ∧ f m r₂ = 0) →
  (∃ r₁ : ℝ, -1 < r₁ ∧ r₁ < 0 ∧ f m r₁ = 0) →
  (∃ r₂ : ℝ, 1 < r₂ ∧ r₂ < 2 ∧ f m r₂ = 0) →
  1/4 < m ∧ m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_root_conditions_imply_m_range_l2223_222382


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2223_222372

theorem trigonometric_simplification :
  (Real.sin (15 * π / 180) + Real.sin (45 * π / 180)) /
  (Real.cos (15 * π / 180) + Real.cos (45 * π / 180)) =
  Real.tan (30 * π / 180) := by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2223_222372


namespace NUMINAMATH_CALUDE_min_value_n_minus_m_l2223_222375

open Real

noncomputable def f (x : ℝ) : ℝ := log (x / 2) + 1 / 2

noncomputable def g (x : ℝ) : ℝ := exp (x - 2)

theorem min_value_n_minus_m :
  (∀ m : ℝ, ∃ n : ℝ, n > 0 ∧ g m = f n) →
  (∃ m n : ℝ, n > 0 ∧ g m = f n ∧ n - m = log 2) ∧
  (∀ m n : ℝ, n > 0 → g m = f n → n - m ≥ log 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_n_minus_m_l2223_222375


namespace NUMINAMATH_CALUDE_hillarys_reading_assignment_l2223_222303

theorem hillarys_reading_assignment 
  (total_assignment : ℕ) 
  (friday_reading : ℕ) 
  (saturday_reading : ℕ) :
  total_assignment = 60 →
  friday_reading = 16 →
  saturday_reading = 28 →
  total_assignment - (friday_reading + saturday_reading) = 16 :=
by sorry

end NUMINAMATH_CALUDE_hillarys_reading_assignment_l2223_222303


namespace NUMINAMATH_CALUDE_fifth_number_in_tenth_row_is_68_l2223_222360

/-- Represents a lattice with rows of consecutive integers -/
structure IntegerLattice where
  rowLength : ℕ
  rowCount : ℕ

/-- Gets the last number in a given row of the lattice -/
def lastNumberInRow (lattice : IntegerLattice) (row : ℕ) : ℤ :=
  (lattice.rowLength : ℤ) * row

/-- Gets the nth number in a given row of the lattice -/
def nthNumberInRow (lattice : IntegerLattice) (row : ℕ) (n : ℕ) : ℤ :=
  lastNumberInRow lattice row - (lattice.rowLength - n : ℤ)

/-- The theorem to be proved -/
theorem fifth_number_in_tenth_row_is_68 :
  let lattice : IntegerLattice := { rowLength := 7, rowCount := 10 }
  nthNumberInRow lattice 10 5 = 68 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_in_tenth_row_is_68_l2223_222360


namespace NUMINAMATH_CALUDE_sine_identity_l2223_222398

theorem sine_identity (α : Real) (h : α = π / 7) :
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := by
  sorry

end NUMINAMATH_CALUDE_sine_identity_l2223_222398


namespace NUMINAMATH_CALUDE_omega_sum_l2223_222312

theorem omega_sum (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 + ω^32 + ω^34 + ω^36 + ω^38 + ω^40 + ω^42 + 
  ω^44 + ω^46 + ω^48 + ω^50 + ω^52 + ω^54 + ω^56 + ω^58 + ω^60 + ω^62 + ω^64 + ω^66 + ω^68 + ω^70 + ω^72 = -ω^7 :=
by sorry

end NUMINAMATH_CALUDE_omega_sum_l2223_222312
