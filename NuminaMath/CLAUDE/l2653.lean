import Mathlib

namespace negation_of_proposition_negation_of_inequality_l2653_265315

theorem negation_of_proposition (P : ℝ → Prop) :
  (¬ ∀ x : ℝ, P x) ↔ (∃ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∀ x : ℝ, x^2 + 1 ≥ 2*x) ↔ (∃ x : ℝ, x^2 + 1 < 2*x) :=
by sorry

end negation_of_proposition_negation_of_inequality_l2653_265315


namespace conic_is_hyperbola_l2653_265359

/-- The equation of a conic section -/
def conic_equation (x y : ℝ) : Prop :=
  (3*x - 2)^2 - 2*(5*y + 1)^2 = 288

/-- Definition of a hyperbola -/
def is_hyperbola (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b c d e : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * b < 0 ∧
    ∀ x y, f x y ↔ a * x^2 + b * y^2 + c * x + d * y + e = 0

/-- Theorem: The given conic equation represents a hyperbola -/
theorem conic_is_hyperbola : is_hyperbola conic_equation :=
sorry

end conic_is_hyperbola_l2653_265359


namespace larger_solution_of_quadratic_l2653_265397

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 36 = 0 ∧ x ≠ 4 → x = 9 := by sorry

end larger_solution_of_quadratic_l2653_265397


namespace library_book_selection_l2653_265337

theorem library_book_selection (math_books : Nat) (literature_books : Nat) (english_books : Nat) :
  math_books = 3 →
  literature_books = 5 →
  english_books = 8 →
  math_books + literature_books + english_books = 16 :=
by
  sorry

end library_book_selection_l2653_265337


namespace simple_interest_principal_l2653_265363

/-- Proves that given specific simple interest conditions, the principal amount is 2000 --/
theorem simple_interest_principal :
  ∀ (rate : ℚ) (interest : ℚ) (time : ℚ) (principal : ℚ),
    rate = 25/2 →
    interest = 500 →
    time = 2 →
    principal * rate * time / 100 = interest →
    principal = 2000 := by
  sorry

end simple_interest_principal_l2653_265363


namespace cubes_occupy_two_thirds_l2653_265308

/-- The dimensions of the rectangular box in inches -/
def box_dimensions : Fin 3 → ℕ
| 0 => 8
| 1 => 6
| 2 => 12
| _ => 0

/-- The side length of a cube in inches -/
def cube_side_length : ℕ := 4

/-- The volume of the rectangular box -/
def box_volume : ℕ := (box_dimensions 0) * (box_dimensions 1) * (box_dimensions 2)

/-- The volume occupied by cubes -/
def cubes_volume : ℕ := 
  ((box_dimensions 0) / cube_side_length) * 
  ((box_dimensions 1) / cube_side_length) * 
  ((box_dimensions 2) / cube_side_length) * 
  (cube_side_length ^ 3)

/-- The percentage of the box volume occupied by cubes -/
def volume_percentage : ℚ := (cubes_volume : ℚ) / (box_volume : ℚ) * 100

theorem cubes_occupy_two_thirds : volume_percentage = 200 / 3 := by
  sorry

end cubes_occupy_two_thirds_l2653_265308


namespace complex_equation_l2653_265304

theorem complex_equation : (2 * Complex.I) * (1 + Complex.I)^2 = -4 := by
  sorry

end complex_equation_l2653_265304


namespace complement_of_at_most_one_hit_l2653_265394

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcome of two consecutive shots -/
def TwoShotOutcome := ShotOutcome × ShotOutcome

/-- The event "at most one shot hits the target" -/
def atMostOneHit (outcome : TwoShotOutcome) : Prop :=
  match outcome with
  | (ShotOutcome.Hit, ShotOutcome.Miss) => True
  | (ShotOutcome.Miss, ShotOutcome.Hit) => True
  | (ShotOutcome.Miss, ShotOutcome.Miss) => True
  | _ => False

/-- The event "both shots hit the target" -/
def bothHit (outcome : TwoShotOutcome) : Prop :=
  outcome = (ShotOutcome.Hit, ShotOutcome.Hit)

theorem complement_of_at_most_one_hit :
  ∀ (outcome : TwoShotOutcome), ¬(atMostOneHit outcome) ↔ bothHit outcome := by
  sorry

end complement_of_at_most_one_hit_l2653_265394


namespace tax_free_items_cost_l2653_265396

theorem tax_free_items_cost 
  (total_paid : ℝ) 
  (sales_tax : ℝ) 
  (tax_rate : ℝ) 
  (h1 : total_paid = 40) 
  (h2 : sales_tax = 1.28) 
  (h3 : tax_rate = 0.08) : 
  total_paid - (sales_tax / tax_rate + sales_tax) = 22.72 := by
sorry

end tax_free_items_cost_l2653_265396


namespace therapy_charge_theorem_l2653_265392

/-- Represents the pricing structure of a psychologist's therapy sessions. -/
structure TherapyPricing where
  first_hour : ℕ
  additional_hour : ℕ
  first_hour_premium : first_hour = additional_hour + 20

/-- Calculates the total charge for a given number of therapy hours. -/
def total_charge (pricing : TherapyPricing) (hours : ℕ) : ℕ :=
  pricing.first_hour + (hours - 1) * pricing.additional_hour

/-- Theorem stating the total charge for 3 hours of therapy given the conditions. -/
theorem therapy_charge_theorem (pricing : TherapyPricing) 
  (h1 : total_charge pricing 5 = 300) : 
  total_charge pricing 3 = 188 := by
  sorry

end therapy_charge_theorem_l2653_265392


namespace polynomial_factorization_l2653_265330

theorem polynomial_factorization (k : ℤ) :
  let N : ℕ := (4 * k^4 - 8 * k^2 + 2).toNat
  let p (x : ℝ) := x^8 + N * x^4 + 1
  let f (x : ℝ) := x^4 - 2*k*x^3 + 2*k^2*x^2 - 2*k*x + 1
  let g (x : ℝ) := x^4 + 2*k*x^3 + 2*k^2*x^2 + 2*k*x + 1
  ∀ x, p x = f x * g x :=
by sorry

end polynomial_factorization_l2653_265330


namespace lego_sales_triple_pieces_l2653_265377

/-- Represents the number of Lego pieces sold for each type --/
structure LegoSales where
  single : ℕ
  double : ℕ
  triple : ℕ
  quadruple : ℕ

/-- Calculates the total earnings in cents from Lego sales --/
def totalEarnings (sales : LegoSales) : ℕ :=
  sales.single * 1 + sales.double * 2 + sales.triple * 3 + sales.quadruple * 4

/-- The main theorem to prove --/
theorem lego_sales_triple_pieces : 
  ∃ (sales : LegoSales), 
    sales.single = 100 ∧ 
    sales.double = 45 ∧ 
    sales.quadruple = 165 ∧ 
    totalEarnings sales = 1000 ∧ 
    sales.triple = 50 := by
  sorry


end lego_sales_triple_pieces_l2653_265377


namespace correct_both_problems_l2653_265307

theorem correct_both_problems (total : ℕ) (correct_sets : ℕ) (correct_functions : ℕ) (wrong_both : ℕ)
  (h1 : total = 50)
  (h2 : correct_sets = 40)
  (h3 : correct_functions = 31)
  (h4 : wrong_both = 4) :
  correct_sets + correct_functions - (total - wrong_both) = 25 := by
sorry

end correct_both_problems_l2653_265307


namespace high_school_elite_season_games_l2653_265361

/-- The number of teams in the "High School Elite" basketball league -/
def num_teams : ℕ := 8

/-- The number of times each team plays every other team -/
def games_per_pairing : ℕ := 3

/-- The number of games each team plays against non-conference opponents -/
def non_conference_games : ℕ := 5

/-- The total number of games in a season for the "High School Elite" league -/
def total_games : ℕ := (num_teams.choose 2 * games_per_pairing) + (num_teams * non_conference_games)

theorem high_school_elite_season_games :
  total_games = 124 := by sorry

end high_school_elite_season_games_l2653_265361


namespace min_value_sum_reciprocals_l2653_265328

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_eq_two : x + y + z = 2) :
  1 / (x + 3*y) + 1 / (y + 3*z) + 1 / (z + 3*x) ≥ 27 / 8 := by
  sorry

end min_value_sum_reciprocals_l2653_265328


namespace H_perimeter_is_36_l2653_265342

/-- Calculates the perimeter of a rectangle given its length and width -/
def rectanglePerimeter (length width : ℝ) : ℝ := 2 * (length + width)

/-- Represents the H-shaped figure -/
structure HShape where
  largeRectLength : ℝ
  largeRectWidth : ℝ
  smallRectLength : ℝ
  smallRectWidth : ℝ

/-- Calculates the perimeter of the H-shaped figure -/
def HPerimeter (h : HShape) : ℝ :=
  2 * rectanglePerimeter h.largeRectLength h.largeRectWidth +
  rectanglePerimeter h.smallRectLength h.smallRectWidth -
  2 * 2 * h.smallRectLength

theorem H_perimeter_is_36 :
  let h : HShape := {
    largeRectLength := 3,
    largeRectWidth := 5,
    smallRectLength := 1,
    smallRectWidth := 3
  }
  HPerimeter h = 36 := by
  sorry

end H_perimeter_is_36_l2653_265342


namespace cos_theta_plus_5pi_over_6_l2653_265399

theorem cos_theta_plus_5pi_over_6 (θ : Real) (h1 : 0 < θ) (h2 : θ < π / 2) 
  (h3 : Real.sin (θ / 2 + π / 6) = 4 / 5) : 
  Real.cos (θ + 5 * π / 6) = -24 / 25 := by
  sorry

end cos_theta_plus_5pi_over_6_l2653_265399


namespace division_with_remainder_l2653_265300

theorem division_with_remainder (A : ℕ) : 
  (A / 7 = 5) ∧ (A % 7 = 3) → A = 38 := by
  sorry

end division_with_remainder_l2653_265300


namespace line_inclination_angle_l2653_265327

/-- The inclination angle of a line with equation √3x - y + 1 = 0 is 60° -/
theorem line_inclination_angle (x y : ℝ) :
  (Real.sqrt 3 * x - y + 1 = 0) →
  ∃ θ : ℝ, 0 ≤ θ ∧ θ < Real.pi ∧ θ = Real.pi / 3 := by
  sorry

end line_inclination_angle_l2653_265327


namespace pascal_ratio_in_row_98_l2653_265362

/-- Pascal's Triangle entry -/
def pascal (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

/-- Check if three consecutive entries in a row are in ratio 4:5:6 -/
def hasRatio456 (n : ℕ) : Prop :=
  ∃ r : ℕ, 
    (pascal n r : ℚ) / (pascal n (r + 1)) = 4 / 5 ∧
    (pascal n (r + 1) : ℚ) / (pascal n (r + 2)) = 5 / 6

theorem pascal_ratio_in_row_98 : hasRatio456 98 := by
  sorry

end pascal_ratio_in_row_98_l2653_265362


namespace conference_handshakes_l2653_265333

/-- The number of handshakes in a conference of n people where each person
    shakes hands with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := Nat.choose n 2

/-- Theorem stating that in a conference of 10 people, where each person
    shakes hands with every other person exactly once, there are 45 handshakes. -/
theorem conference_handshakes :
  handshakes 10 = 45 := by sorry

end conference_handshakes_l2653_265333


namespace fair_attendance_this_year_l2653_265317

def fair_attendance (this_year next_year last_year : ℕ) : Prop :=
  (next_year = 2 * this_year) ∧
  (last_year = next_year - 200) ∧
  (this_year + next_year + last_year = 2800)

theorem fair_attendance_this_year :
  ∃ (this_year next_year last_year : ℕ),
    fair_attendance this_year next_year last_year ∧ this_year = 600 :=
by
  sorry

end fair_attendance_this_year_l2653_265317


namespace inequality_proof_l2653_265305

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  2 * (x^2 + y^2) ≥ (x + y)^2 := by
  sorry

end inequality_proof_l2653_265305


namespace largest_number_less_than_threshold_l2653_265320

def given_numbers : List ℚ := [4, 9/10, 6/5, 1/2, 13/10]
def threshold : ℚ := 111/100

theorem largest_number_less_than_threshold :
  (given_numbers.filter (· < threshold)).maximum? = some (9/10) := by
  sorry

end largest_number_less_than_threshold_l2653_265320


namespace watch_cost_price_l2653_265371

theorem watch_cost_price (loss_percentage : ℝ) (gain_percentage : ℝ) (additional_amount : ℝ) :
  loss_percentage = 10 →
  gain_percentage = 5 →
  additional_amount = 180 →
  ∃ (cost_price : ℝ),
    cost_price * (1 - loss_percentage / 100) + additional_amount = cost_price * (1 + gain_percentage / 100) ∧
    cost_price = 1200 := by
  sorry

end watch_cost_price_l2653_265371


namespace parallel_tangents_intersection_l2653_265382

theorem parallel_tangents_intersection (x₀ : ℝ) : 
  (∃ (k : ℝ), (2 * x₀ = k) ∧ (-3 * x₀^2 = k)) → (x₀ = 0 ∨ x₀ = -2/3) :=
by sorry

end parallel_tangents_intersection_l2653_265382


namespace fixed_fee_is_ten_l2653_265334

/-- Represents the billing structure for an online service provider -/
structure BillingStructure where
  fixed_fee : ℝ
  hourly_charge : ℝ

/-- Represents the monthly usage and bill -/
structure MonthlyBill where
  connect_time : ℝ
  total_bill : ℝ

/-- The billing problem with given conditions -/
def billing_problem (b : BillingStructure) : Prop :=
  ∃ (feb_time : ℝ),
    let feb : MonthlyBill := ⟨feb_time, 20⟩
    let mar : MonthlyBill := ⟨2 * feb_time, 30⟩
    let apr : MonthlyBill := ⟨3 * feb_time, 40⟩
    (b.fixed_fee + b.hourly_charge * feb.connect_time = feb.total_bill) ∧
    (b.fixed_fee + b.hourly_charge * mar.connect_time = mar.total_bill) ∧
    (b.fixed_fee + b.hourly_charge * apr.connect_time = apr.total_bill)

/-- The theorem stating that the fixed monthly fee is $10.00 -/
theorem fixed_fee_is_ten :
  ∀ b : BillingStructure, billing_problem b → b.fixed_fee = 10 := by
  sorry


end fixed_fee_is_ten_l2653_265334


namespace parabola_intersection_theorem_l2653_265370

/-- Parabola type -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y^2 = 4*p*x

/-- Line type -/
structure Line where
  m : ℝ
  b : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y = m*x + b

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem parabola_intersection_theorem (C : Parabola) (a : ℝ) (l : Line) 
  (A B A' : Point) :
  C.p = 3 →  -- This ensures y^2 = 12x
  a < 0 →
  A.y^2 = 12*A.x →
  B.y^2 = 12*B.x →
  l.eq A.x A.y →
  l.eq B.x B.y →
  l.eq a 0 →
  A'.x = A.x →
  A'.y = -A.y →
  ∃ (l' : Line), l'.eq A'.x A'.y ∧ l'.eq B.x B.y ∧ l'.eq (-a) 0 := by
  sorry

end parabola_intersection_theorem_l2653_265370


namespace factorial_300_trailing_zeros_l2653_265335

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

theorem factorial_300_trailing_zeros :
  trailing_zeros 300 = 74 := by sorry

end factorial_300_trailing_zeros_l2653_265335


namespace cubic_fraction_inequality_l2653_265398

theorem cubic_fraction_inequality (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  1/2 ≤ (a^3 + b^3) / (a^2 + b^2) ∧ (a^3 + b^3) / (a^2 + b^2) ≤ 1 ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1/2 ↔ a = 1/2 ∧ b = 1/2) ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0)) :=
sorry

end cubic_fraction_inequality_l2653_265398


namespace cycle_sale_theorem_l2653_265313

/-- Calculates the net total amount received after selling three cycles and paying tax -/
def net_total_amount (price1 price2 price3 : ℚ) (profit1 loss2 profit3 tax_rate : ℚ) : ℚ :=
  let sell1 := price1 * (1 + profit1)
  let sell2 := price2 * (1 - loss2)
  let sell3 := price3 * (1 + profit3)
  let total_sell := sell1 + sell2 + sell3
  let tax := total_sell * tax_rate
  total_sell - tax

theorem cycle_sale_theorem :
  net_total_amount 3600 4800 6000 (20/100) (15/100) (10/100) (5/100) = 14250 := by
  sorry

#eval net_total_amount 3600 4800 6000 (20/100) (15/100) (10/100) (5/100)

end cycle_sale_theorem_l2653_265313


namespace december_gas_consumption_l2653_265378

/-- Gas fee structure and consumption for a user in December --/
structure GasConsumption where
  baseRate : ℝ  -- Rate for the first 60 cubic meters
  excessRate : ℝ  -- Rate for consumption above 60 cubic meters
  baseVolume : ℝ  -- Volume threshold for base rate
  averageCost : ℝ  -- Average cost per cubic meter for the user
  consumption : ℝ  -- Total gas consumption

/-- The gas consumption satisfies the given fee structure and average cost --/
def validConsumption (g : GasConsumption) : Prop :=
  g.baseRate * g.baseVolume + g.excessRate * (g.consumption - g.baseVolume) = g.averageCost * g.consumption

/-- Theorem stating that given the fee structure and average cost, 
    the gas consumption in December was 100 cubic meters --/
theorem december_gas_consumption :
  ∃ (g : GasConsumption), 
    g.baseRate = 1 ∧ 
    g.excessRate = 1.5 ∧ 
    g.baseVolume = 60 ∧ 
    g.averageCost = 1.2 ∧ 
    g.consumption = 100 ∧
    validConsumption g :=
  sorry

end december_gas_consumption_l2653_265378


namespace triangle_area_double_l2653_265306

theorem triangle_area_double (halved_area : ℝ) :
  halved_area = 7 → 2 * halved_area = 14 :=
by sorry

end triangle_area_double_l2653_265306


namespace initial_gasoline_percentage_l2653_265352

/-- Proves that the initial gasoline percentage is 95% given the problem conditions --/
theorem initial_gasoline_percentage
  (initial_volume : ℝ)
  (initial_ethanol_percentage : ℝ)
  (desired_ethanol_percentage : ℝ)
  (added_ethanol : ℝ)
  (h1 : initial_volume = 54)
  (h2 : initial_ethanol_percentage = 0.05)
  (h3 : desired_ethanol_percentage = 0.10)
  (h4 : added_ethanol = 3)
  (h5 : initial_volume * initial_ethanol_percentage + added_ethanol = 
        (initial_volume + added_ethanol) * desired_ethanol_percentage) :
  1 - initial_ethanol_percentage = 0.95 := by
  sorry

end initial_gasoline_percentage_l2653_265352


namespace hundredth_figure_squares_l2653_265322

def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

theorem hundredth_figure_squares :
  f 100 = 30301 := by sorry

end hundredth_figure_squares_l2653_265322


namespace power_two_mod_seven_l2653_265353

theorem power_two_mod_seven : 2^2010 ≡ 1 [MOD 7] := by sorry

end power_two_mod_seven_l2653_265353


namespace complex_sum_equality_l2653_265309

theorem complex_sum_equality : ∃ (r θ : ℝ), 
  5 * Complex.exp (2 * π * Complex.I / 13) + 5 * Complex.exp (17 * π * Complex.I / 26) = 
  r * Complex.exp (θ * Complex.I) ∧ 
  r = 5 * Real.sqrt 2 ∧ 
  θ = 21 * π / 52 := by
sorry

end complex_sum_equality_l2653_265309


namespace carbon_dioxide_formation_l2653_265345

-- Define the chemical reaction
def chemical_reaction (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ) : Prop :=
  HNO3 = 1 ∧ NaHCO3 = 1 ∧ NaNO3 = 1 ∧ CO2 = 1 ∧ H2O = 1

-- Theorem statement
theorem carbon_dioxide_formation :
  ∀ (HNO3 NaHCO3 NaNO3 CO2 H2O : ℕ),
    chemical_reaction HNO3 NaHCO3 NaNO3 CO2 H2O →
    CO2 = 1 :=
by
  sorry

end carbon_dioxide_formation_l2653_265345


namespace real_number_pure_imaginary_condition_l2653_265374

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the condition for a pure imaginary number
def isPureImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

-- State the theorem
theorem real_number_pure_imaginary_condition (m : ℝ) : 
  isPureImaginary (m^2 * (1 + i) + (m - i) - 2) → m = -2 :=
by sorry

end real_number_pure_imaginary_condition_l2653_265374


namespace quadratic_sequence_problem_l2653_265376

theorem quadratic_sequence_problem (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (eq1 : y₁ + 4*y₂ + 9*y₃ + 16*y₄ + 25*y₅ = 3)
  (eq2 : 4*y₁ + 9*y₂ + 16*y₃ + 25*y₄ + 36*y₅ = 20)
  (eq3 : 9*y₁ + 16*y₂ + 25*y₃ + 36*y₄ + 49*y₅ = 150) :
  16*y₁ + 25*y₂ + 36*y₃ + 49*y₄ + 64*y₅ = 336 := by
  sorry

end quadratic_sequence_problem_l2653_265376


namespace trajectory_equation_l2653_265355

/-- The trajectory of a point whose sum of distances to the coordinate axes is 6 -/
theorem trajectory_equation (x y : ℝ) : 
  (dist x 0 + dist y 0 = 6) → (|x| + |y| = 6) :=
by sorry

end trajectory_equation_l2653_265355


namespace translation_of_complex_plane_l2653_265380

open Complex

theorem translation_of_complex_plane (t : ℂ → ℂ) :
  (t (-3 + 3*I) = -8 - 2*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = z + w) →
  (t (-2 + 6*I) = -7 + I) :=
by sorry

end translation_of_complex_plane_l2653_265380


namespace star_calculation_l2653_265350

-- Define the ⋆ operation
def star (a b : ℚ) : ℚ := a + 2 / b

-- Theorem statement
theorem star_calculation :
  (star (star 3 4) 5) - (star 3 (star 4 5)) = 49 / 110 := by
  sorry

end star_calculation_l2653_265350


namespace min_distance_to_line_l2653_265373

/-- The minimum distance from the origin to the line 3x + 4y - 20 = 0 is 4 -/
theorem min_distance_to_line : ∃ (d : ℝ),
  (∀ (a b : ℝ), 3 * a + 4 * b - 20 = 0 → a^2 + b^2 ≥ d^2) ∧
  (∃ (a b : ℝ), 3 * a + 4 * b - 20 = 0 ∧ a^2 + b^2 = d^2) ∧
  d = 4 := by
  sorry

end min_distance_to_line_l2653_265373


namespace largest_two_digit_with_digit_product_12_l2653_265341

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem largest_two_digit_with_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
by sorry

end largest_two_digit_with_digit_product_12_l2653_265341


namespace cube_congruence_implies_sum_divisibility_l2653_265369

theorem cube_congruence_implies_sum_divisibility (p x y z : ℕ) : 
  Prime p → 
  0 < x → x < y → y < z → z < p → 
  x^3 % p = y^3 % p → y^3 % p = z^3 % p → 
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
sorry

end cube_congruence_implies_sum_divisibility_l2653_265369


namespace time_spent_playing_games_l2653_265357

/-- Calculates the time spent playing games during a flight -/
theorem time_spent_playing_games 
  (total_flight_time : ℕ) 
  (reading_time : ℕ) 
  (movie_time : ℕ) 
  (dinner_time : ℕ) 
  (radio_time : ℕ) 
  (nap_time : ℕ) : 
  total_flight_time = 11 * 60 + 20 → 
  reading_time = 2 * 60 → 
  movie_time = 4 * 60 → 
  dinner_time = 30 → 
  radio_time = 40 → 
  nap_time = 3 * 60 → 
  total_flight_time - (reading_time + movie_time + dinner_time + radio_time + nap_time) = 70 := by
sorry

end time_spent_playing_games_l2653_265357


namespace triangle_inequalities_l2653_265344

-- Define a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  p : ℝ -- semi-perimeter
  R : ℝ -- circumradius
  r : ℝ -- inradius
  S : ℝ -- area

-- State the theorem
theorem triangle_inequalities (t : Triangle) : 
  (Real.cos t.A + Real.cos t.B + Real.cos t.C ≤ 3/2) ∧
  (Real.sin (t.A/2) * Real.sin (t.B/2) * Real.sin (t.C/2) ≤ 1/8) ∧
  (t.a * t.b * t.c ≥ 8 * (t.p - t.a) * (t.p - t.b) * (t.p - t.c)) ∧
  (t.R ≥ 2 * t.r) ∧
  (t.S ≤ (1/2) * t.R * t.p) := by
  sorry

end triangle_inequalities_l2653_265344


namespace max_gcd_thirteen_numbers_sum_1988_l2653_265336

theorem max_gcd_thirteen_numbers_sum_1988 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ a₁₂ a₁₃ : ℕ) 
  (h_sum : a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ + a₁₂ + a₁₃ = 1988) :
  Nat.gcd a₁ (Nat.gcd a₂ (Nat.gcd a₃ (Nat.gcd a₄ (Nat.gcd a₅ (Nat.gcd a₆ (Nat.gcd a₇ (Nat.gcd a₈ (Nat.gcd a₉ (Nat.gcd a₁₀ (Nat.gcd a₁₁ (Nat.gcd a₁₂ a₁₃))))))))))) ≤ 142 :=
by
  sorry

end max_gcd_thirteen_numbers_sum_1988_l2653_265336


namespace blueberry_picking_relationships_l2653_265354

/-- Represents the relationship between y₁ and x for blueberry picking -/
def y₁ (x : ℝ) : ℝ := 60 + 18 * x

/-- Represents the relationship between y₂ and x for blueberry picking -/
def y₂ (x : ℝ) : ℝ := 150 + 15 * x

/-- Theorem stating the relationships between y₁, y₂, and x when blueberry picking amount exceeds 10 kg -/
theorem blueberry_picking_relationships (x : ℝ) (h : x > 10) :
  y₁ x = 60 + 18 * x ∧ y₂ x = 150 + 15 * x := by
  sorry

end blueberry_picking_relationships_l2653_265354


namespace solution_set_when_a_is_one_a_value_when_x_in_range_l2653_265381

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - 1| + |x - 2*a|

-- Part I
theorem solution_set_when_a_is_one :
  let a := 1
  ∀ x, f x a ≤ 3 ↔ x ∈ Set.Icc 0 2 :=
sorry

-- Part II
theorem a_value_when_x_in_range :
  (∀ x ∈ Set.Icc 1 2, f x a ≤ 3) → a = 1 :=
sorry

end solution_set_when_a_is_one_a_value_when_x_in_range_l2653_265381


namespace train_distance_l2653_265358

theorem train_distance (v1 v2 t : ℝ) (h1 : v1 = 11) (h2 : v2 = 31) (h3 : t = 8) :
  (v2 * t) - (v1 * t) = 160 :=
by sorry

end train_distance_l2653_265358


namespace min_disks_for_lilas_problem_l2653_265366

/-- Represents the storage problem with given file sizes and quantities --/
structure StorageProblem where
  total_files : ℕ
  disk_capacity : ℚ
  large_files : ℕ
  large_file_size : ℚ
  medium_files : ℕ
  medium_file_size : ℚ
  small_file_size : ℚ

/-- Calculates the minimum number of disks required for the given storage problem --/
def min_disks_required (problem : StorageProblem) : ℕ :=
  sorry

/-- The specific storage problem instance --/
def lilas_problem : StorageProblem :=
  { total_files := 40
  , disk_capacity := 2
  , large_files := 4
  , large_file_size := 1.2
  , medium_files := 10
  , medium_file_size := 1
  , small_file_size := 0.6 }

/-- Theorem stating that the minimum number of disks required for Lila's problem is 16 --/
theorem min_disks_for_lilas_problem :
  min_disks_required lilas_problem = 16 :=
sorry

end min_disks_for_lilas_problem_l2653_265366


namespace remainder_theorem_l2653_265325

def polynomial (x : ℝ) : ℝ := 8*x^4 - 18*x^3 + 27*x^2 - 31*x + 14

def divisor (x : ℝ) : ℝ := 4*x - 8

theorem remainder_theorem :
  ∃ (q : ℝ → ℝ), ∀ x, polynomial x = (divisor x) * (q x) + 30 := by
  sorry

end remainder_theorem_l2653_265325


namespace circle_transformation_l2653_265321

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)

theorem circle_transformation :
  let initial_point : ℝ × ℝ := (3, -4)
  let reflected_point := reflect_x initial_point
  let final_point := translate_right reflected_point 10
  final_point = (13, 4) := by sorry

end circle_transformation_l2653_265321


namespace product_remainder_l2653_265351

theorem product_remainder (a b c : ℕ) (ha : a = 2457) (hb : b = 6273) (hc : c = 91409) :
  (a * b * c) % 10 = 9 := by
  sorry

end product_remainder_l2653_265351


namespace f_composition_negative_two_l2653_265356

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then -1 / x else 2 * Real.sqrt x

theorem f_composition_negative_two : f (f (-2)) = Real.sqrt 2 := by
  sorry

end f_composition_negative_two_l2653_265356


namespace inequality_proof_l2653_265301

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x * y ≥ 1) :
  x^3 + y^3 + 4*x*y ≥ x^2 + y^2 + x + y + 2 := by
  sorry

end inequality_proof_l2653_265301


namespace three_solutions_sum_l2653_265331

theorem three_solutions_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_solutions : ∃ (x₁ x₂ x₃ : ℝ), x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ = b ∧
    (∀ x : ℝ, Real.sqrt (|x|) + Real.sqrt (|x + a|) = b ↔ x = x₁ ∨ x = x₂ ∨ x = x₃)) :
  a + b = 144 := by
sorry

end three_solutions_sum_l2653_265331


namespace principal_calculation_l2653_265302

/-- Prove that given the conditions, the principal amount is 1500 --/
theorem principal_calculation (P : ℝ) : 
  P * (1 + 0.1)^2 - P - (P * 0.1 * 2) = 15 → P = 1500 := by
  sorry

end principal_calculation_l2653_265302


namespace max_value_of_expression_l2653_265314

theorem max_value_of_expression (a b : ℝ) (h1 : 300 ≤ a ∧ a ≤ 500) 
  (h2 : 500 ≤ b ∧ b ≤ 1500) : 
  let c : ℝ := 100
  ∀ x ∈ Set.Icc 300 500, ∀ y ∈ Set.Icc 500 1500, 
    (b + c) / (a - c) ≤ 8 ∧ (y + c) / (x - c) ≤ 8 :=
by
  sorry

#check max_value_of_expression

end max_value_of_expression_l2653_265314


namespace complex_fraction_evaluation_l2653_265343

theorem complex_fraction_evaluation : 
  let f1 : ℚ := 7 / 18
  let f2 : ℚ := 9 / 2  -- 4 1/2 as improper fraction
  let f3 : ℚ := 1 / 6
  let f4 : ℚ := 40 / 3  -- 13 1/3 as improper fraction
  let f5 : ℚ := 15 / 4  -- 3 3/4 as improper fraction
  let f6 : ℚ := 5 / 16
  let f7 : ℚ := 23 / 8  -- 2 7/8 as improper fraction
  (((f1 * f2 + f3) / (f4 - f5 / f6)) * f7) = 529 / 128 := by
  sorry

end complex_fraction_evaluation_l2653_265343


namespace boys_total_toys_l2653_265312

/-- The number of toys Bill has -/
def bill_toys : ℕ := 60

/-- The number of toys Hash has -/
def hash_toys : ℕ := bill_toys / 2 + 9

/-- The total number of toys both boys have -/
def total_toys : ℕ := bill_toys + hash_toys

theorem boys_total_toys : total_toys = 99 := by
  sorry

end boys_total_toys_l2653_265312


namespace employed_females_percentage_proof_l2653_265323

/-- The percentage of employed people in town X -/
def employed_percentage : ℝ := 64

/-- The percentage of employed males in town X -/
def employed_males_percentage : ℝ := 48

/-- The percentage of employed females out of the total employed population in town X -/
def employed_females_percentage : ℝ := 25

theorem employed_females_percentage_proof :
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = employed_females_percentage :=
by sorry

end employed_females_percentage_proof_l2653_265323


namespace square_perimeter_l2653_265311

theorem square_perimeter : ∀ (x₁ x₂ : ℝ),
  x₁^2 + 4*x₁ + 3 = 7 →
  x₂^2 + 4*x₂ + 3 = 7 →
  x₁ ≠ x₂ →
  4 * |x₂ - x₁| = 16 * Real.sqrt 2 :=
by sorry

end square_perimeter_l2653_265311


namespace cd_length_is_nine_l2653_265329

/-- A tetrahedron with specific edge lengths -/
structure Tetrahedron where
  edges : Finset ℝ
  edge_count : edges.card = 6
  edge_values : edges = {9, 15, 22, 35, 40, 44}
  ab_length : 44 ∈ edges

/-- The length of CD in the tetrahedron -/
def cd_length (t : Tetrahedron) : ℝ := 9

/-- Theorem stating that CD length is 9 in the given tetrahedron -/
theorem cd_length_is_nine (t : Tetrahedron) : cd_length t = 9 := by
  sorry

end cd_length_is_nine_l2653_265329


namespace boys_height_ratio_l2653_265303

theorem boys_height_ratio (total_students : ℕ) (boys_under_6ft : ℕ) 
  (h1 : total_students = 100)
  (h2 : boys_under_6ft = 10) :
  (boys_under_6ft : ℚ) / (total_students / 2 : ℚ) = 1 / 5 := by
  sorry

end boys_height_ratio_l2653_265303


namespace probability_of_different_digits_l2653_265318

/-- The number of integers from 100 to 999 inclusive -/
def total_integers : ℕ := 999 - 100 + 1

/-- The number of integers from 100 to 999 with all different digits -/
def integers_with_different_digits : ℕ := 9 * 9 * 8

/-- The probability of selecting an integer with all different digits from 100 to 999 -/
def probability : ℚ := integers_with_different_digits / total_integers

theorem probability_of_different_digits : probability = 18 / 25 := by
  sorry

end probability_of_different_digits_l2653_265318


namespace product_of_separated_evens_l2653_265310

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def circular_arrangement (n : ℕ) : Prop := n ≥ 1 ∧ n ≤ 16

theorem product_of_separated_evens (a d : ℕ) : 
  circular_arrangement a → 
  circular_arrangement d → 
  is_even a → 
  is_even d → 
  (∃ b c, circular_arrangement b ∧ circular_arrangement c ∧ 
    ((a < b ∧ b < c ∧ c < d) ∨ (d < a ∧ a < b ∧ b < c) ∨ 
     (c < d ∧ d < a ∧ a < b) ∨ (b < c ∧ c < d ∧ d < a))) →
  a * d = 120 :=
sorry

end product_of_separated_evens_l2653_265310


namespace quadratic_function_inequality_l2653_265368

/-- Given a quadratic function f(x) = ax² + bx + c, where a, b, c are constants,
    and its derivative f'(x), if f(x) ≥ f'(x) for all x ∈ ℝ,
    then the maximum value of b²/(a² + c²) is 2√2 - 2. -/
theorem quadratic_function_inequality (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c ≥ 2 * a * x + b) → 
  a > 0 → 
  (∃ M, M = 2 * Real.sqrt 2 - 2 ∧ 
    b^2 / (a^2 + c^2) ≤ M ∧ 
    ∀ N, (∀ a' b' c', (∀ x, a' * x^2 + b' * x + c' ≥ 2 * a' * x + b') → 
      a' > 0 → b'^2 / (a'^2 + c'^2) ≤ N) → 
    M ≤ N) :=
sorry

end quadratic_function_inequality_l2653_265368


namespace fraction_equality_l2653_265395

theorem fraction_equality (a b : ℝ) : (0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b) := by
  sorry

end fraction_equality_l2653_265395


namespace consecutive_odd_numbers_l2653_265390

theorem consecutive_odd_numbers (x : ℤ) : 
  (∃ (y z : ℤ), y = x + 2 ∧ z = x + 4 ∧ 
   Odd x ∧ Odd y ∧ Odd z ∧
   11 * x = 3 * (x + 4) + 16 + 4 * (x + 2)) → 
  x = 9 := by
sorry

end consecutive_odd_numbers_l2653_265390


namespace max_value_abc_l2653_265340

theorem max_value_abc (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq_3 : a + b + c = 3) :
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 → a + b^2 + c^4 ≤ x + y^2 + z^4 ∧ a + b^2 + c^4 ≤ 3 :=
by sorry

end max_value_abc_l2653_265340


namespace sqrt_equation_solution_l2653_265364

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- The unique solution is z = -11
  use (-11 : ℚ)
  sorry

end sqrt_equation_solution_l2653_265364


namespace solve_for_x_l2653_265332

theorem solve_for_x (x y : ℝ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 := by
  sorry

end solve_for_x_l2653_265332


namespace sine_function_period_l2653_265367

theorem sine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ k : ℤ, a * Real.sin (b * x + c) + d = a * Real.sin (b * (x + 2 * π / 5) + c) + d) →
  b = 5 := by
  sorry

end sine_function_period_l2653_265367


namespace intersection_M_N_l2653_265391

def M : Set ℝ := { x | x^2 ≥ 1 }
def N : Set ℝ := { y | ∃ x, y = 3*x^2 + 1 }

theorem intersection_M_N : M ∩ N = { x | x ≥ 1 ∨ x ≤ -1 } := by sorry

end intersection_M_N_l2653_265391


namespace regular_hexagon_side_length_l2653_265389

/-- A regular hexagon with a diagonal of 18 inches has sides of 9 inches. -/
theorem regular_hexagon_side_length :
  ∀ (diagonal side : ℝ),
  diagonal = 18 →
  diagonal = 2 * side →
  side = 9 :=
by
  sorry

end regular_hexagon_side_length_l2653_265389


namespace average_of_list_l2653_265387

def number_list : List Nat := [55, 48, 507, 2, 684, 42]

theorem average_of_list (list : List Nat) : 
  (list.sum / list.length : ℚ) = 223 :=
by sorry

end average_of_list_l2653_265387


namespace absolute_value_inequality_solution_set_l2653_265347

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 5| ≤ 7} = {x : ℝ | -1 ≤ x ∧ x ≤ 6} := by
  sorry

end absolute_value_inequality_solution_set_l2653_265347


namespace arithmetic_sqrt_of_four_l2653_265319

theorem arithmetic_sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end arithmetic_sqrt_of_four_l2653_265319


namespace ninth_group_number_l2653_265349

/-- Represents a systematic sampling scheme -/
structure SystematicSampling where
  total_employees : ℕ
  sample_size : ℕ
  group_size : ℕ
  fifth_group_number : ℕ

/-- The number drawn from the nth group in a systematic sampling -/
def number_drawn (s : SystematicSampling) (n : ℕ) : ℕ :=
  s.group_size * (n - 1) + (s.fifth_group_number - s.group_size * 4)

/-- Theorem stating the relationship between the 5th and 9th group numbers -/
theorem ninth_group_number (s : SystematicSampling) 
  (h1 : s.total_employees = 200)
  (h2 : s.sample_size = 40)
  (h3 : s.group_size = 5)
  (h4 : s.fifth_group_number = 22) :
  number_drawn s 9 = 42 := by
  sorry


end ninth_group_number_l2653_265349


namespace right_triangle_sets_l2653_265316

/-- A function that checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

/-- The theorem stating that among the given sets, only (6, 8, 10) forms a right triangle --/
theorem right_triangle_sets :
  ¬(is_right_triangle 2 3 4) ∧
  (is_right_triangle 6 8 10) ∧
  ¬(is_right_triangle 5 8 13) ∧
  ¬(is_right_triangle 12 13 14) :=
by sorry

end right_triangle_sets_l2653_265316


namespace necessary_not_sufficient_l2653_265384

theorem necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → x^2 - 1 > 0) ∧
  (∃ x : ℝ, x^2 - 1 > 0 ∧ ¬(x > 1)) := by
  sorry

end necessary_not_sufficient_l2653_265384


namespace average_of_ten_numbers_l2653_265338

theorem average_of_ten_numbers (first_six_avg : ℝ) (last_six_avg : ℝ) (seventh_num : ℝ) 
  (h1 : first_six_avg = 68)
  (h2 : last_six_avg = 75)
  (h3 : seventh_num = 258) :
  (6 * first_six_avg + 6 * last_six_avg - seventh_num) / 10 = 60 := by
  sorry

end average_of_ten_numbers_l2653_265338


namespace consecutive_numbers_equation_l2653_265360

theorem consecutive_numbers_equation :
  ∃ (a b c d : ℕ), (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1) ∧ (a * c - b * d = 11) := by
  sorry

end consecutive_numbers_equation_l2653_265360


namespace cost_price_calculation_l2653_265324

theorem cost_price_calculation (markup_percentage : ℝ) (discount_percentage : ℝ) (profit : ℝ) : 
  markup_percentage = 0.2 →
  discount_percentage = 0.1 →
  profit = 40 →
  ∃ (cost_price : ℝ), 
    cost_price * (1 + markup_percentage) * (1 - discount_percentage) - cost_price = profit ∧
    cost_price = 500 :=
by sorry

end cost_price_calculation_l2653_265324


namespace pentagon_angle_sum_l2653_265339

theorem pentagon_angle_sum (a b c d : ℝ) (h1 : a = 130) (h2 : b = 95) (h3 : c = 110) (h4 : d = 104) :
  ∃ q : ℝ, a + b + c + d + q = 540 ∧ q = 101 := by sorry

end pentagon_angle_sum_l2653_265339


namespace log_equation_solution_l2653_265385

/-- Given a > 0, prove that the solution to log_√2(x - a) = 1 + log_2 x is x = a + 1 + √(2a + 1) -/
theorem log_equation_solution (a : ℝ) (ha : a > 0) :
  ∃! x : ℝ, x > a ∧ Real.log (x - a) / Real.log (Real.sqrt 2) = 1 + Real.log x / Real.log 2 ∧
  x = a + 1 + Real.sqrt (2 * a + 1) :=
by sorry

end log_equation_solution_l2653_265385


namespace multiply_mixed_number_l2653_265365

theorem multiply_mixed_number : 8 * (9 + 2/5) = 75 + 1/5 := by
  sorry

end multiply_mixed_number_l2653_265365


namespace rationalize_denominator_l2653_265372

theorem rationalize_denominator :
  ∃ (A B C D E F G H I : ℤ),
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
      (A * Real.sqrt B + C * Real.sqrt D + E * Real.sqrt F + G * Real.sqrt H) / I ∧
    I > 0 ∧
    A = 3 ∧ B = 3 ∧ C = 9 ∧ D = 5 ∧ E = -9 ∧ F = 11 ∧ G = 6 ∧ H = 33 ∧ I = 51 :=
by
  sorry

end rationalize_denominator_l2653_265372


namespace min_value_trig_expression_l2653_265386

open Real

theorem min_value_trig_expression (θ : Real) (h : 0 < θ ∧ θ < π/2) :
  ∃ (min_val : Real), min_val = (11 * Real.sqrt 2) / 2 ∧
  ∀ θ', 0 < θ' ∧ θ' < π/2 →
    3 * cos θ' + 2 / sin θ' + 2 * Real.sqrt 2 * tan θ' ≥ min_val :=
by sorry

end min_value_trig_expression_l2653_265386


namespace max_value_problem_l2653_265388

theorem max_value_problem (A M C : ℕ) (h : A + M + C = 15) :
  (∀ a m c : ℕ, a + m + c = 15 → A * M * C + A * M + M * C + C * A ≥ a * m * c + a * m + m * c + c * a) →
  A * M * C + A * M + M * C + C * A = 200 :=
by sorry

end max_value_problem_l2653_265388


namespace nuts_distribution_l2653_265379

/-- The number of ways to distribute n identical objects into k distinct groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of nuts to be distributed -/
def num_nuts : ℕ := 9

/-- The number of pockets -/
def num_pockets : ℕ := 3

theorem nuts_distribution :
  distribute num_nuts num_pockets = 55 := by sorry

end nuts_distribution_l2653_265379


namespace square_root_of_polynomial_l2653_265326

theorem square_root_of_polynomial (a b c : ℝ) :
  (2*a - 3*b + 4*c)^2 = 16*a*c + 4*a^2 - 12*a*b + 9*b^2 - 24*b*c + 16*c^2 := by
  sorry

end square_root_of_polynomial_l2653_265326


namespace equivalence_of_statements_l2653_265383

variable (P Q : Prop)

theorem equivalence_of_statements :
  (P → Q) ↔ (¬Q → ¬P) ∧ (¬P ∨ Q) :=
sorry

end equivalence_of_statements_l2653_265383


namespace movie_length_after_cut_l2653_265348

/-- Calculates the final length of a movie after cutting a scene -/
theorem movie_length_after_cut (original_length cut_length : ℕ) : 
  original_length = 60 → cut_length = 3 → original_length - cut_length = 57 := by
  sorry

end movie_length_after_cut_l2653_265348


namespace square_area_increase_l2653_265375

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.3 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.69 := by
sorry

end square_area_increase_l2653_265375


namespace factorial_ratio_2016_l2653_265393

-- Define factorial
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_ratio_2016 :
  (factorial 2016)^2 / (factorial 2015 * factorial 2017) = 2016 / 2017 :=
by sorry

end factorial_ratio_2016_l2653_265393


namespace valid_arrangements_l2653_265346

/-- The number of ways to arrange 2 black, 3 white, and 4 red balls in a row such that no black ball is next to a white ball. -/
def arrangeBalls : ℕ := 200

/-- The number of black balls -/
def blackBalls : ℕ := 2

/-- The number of white balls -/
def whiteBalls : ℕ := 3

/-- The number of red balls -/
def redBalls : ℕ := 4

/-- Theorem stating that the number of valid arrangements is equal to arrangeBalls -/
theorem valid_arrangements :
  (∃ (f : ℕ → ℕ → ℕ → ℕ), f blackBalls whiteBalls redBalls = arrangeBalls) :=
sorry

end valid_arrangements_l2653_265346
