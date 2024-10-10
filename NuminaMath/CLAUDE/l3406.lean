import Mathlib

namespace expression_simplification_l3406_340637

theorem expression_simplification (m n x : ℚ) :
  (5 * m + 3 * n - 7 * m - n = -2 * m + 2 * n) ∧
  (2 * x^2 - (3 * x - 2 * (x^2 - x + 3) + 2 * x^2) = 2 * x^2 - 5 * x + 6) :=
by sorry

end expression_simplification_l3406_340637


namespace puzzle_pieces_sum_l3406_340622

/-- The number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- The number of pieces in the second and third puzzles -/
def other_puzzle_pieces : ℕ := first_puzzle_pieces + first_puzzle_pieces / 2

/-- The total number of pieces in all puzzles -/
def total_pieces : ℕ := first_puzzle_pieces + 2 * other_puzzle_pieces

theorem puzzle_pieces_sum :
  total_pieces = 4000 :=
by sorry

end puzzle_pieces_sum_l3406_340622


namespace sqrt_eight_minus_sqrt_two_l3406_340616

theorem sqrt_eight_minus_sqrt_two : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end sqrt_eight_minus_sqrt_two_l3406_340616


namespace sum_of_complex_exponentials_l3406_340692

/-- The sum of 16 complex exponentials with angles that are multiples of 2π/17 -/
theorem sum_of_complex_exponentials (ω : ℂ) (h : ω = Complex.exp (2 * Real.pi * Complex.I / 17)) :
  (Finset.range 16).sum (fun k => ω ^ (k + 1)) = ω := by
  sorry

end sum_of_complex_exponentials_l3406_340692


namespace discount_rate_sum_l3406_340642

-- Define the normal prices and quantities
def biography_price : ℝ := 20
def mystery_price : ℝ := 12
def biography_quantity : ℕ := 5
def mystery_quantity : ℕ := 3

-- Define the total savings and mystery discount rate
def total_savings : ℝ := 19
def mystery_discount_rate : ℝ := 0.375

-- Define the function to calculate the total discount rate
def total_discount_rate (biography_discount_rate : ℝ) : ℝ :=
  biography_discount_rate + mystery_discount_rate

-- Theorem statement
theorem discount_rate_sum :
  ∃ (biography_discount_rate : ℝ),
    biography_discount_rate > 0 ∧
    biography_discount_rate < 1 ∧
    (biography_price * biography_quantity * (1 - biography_discount_rate) +
     mystery_price * mystery_quantity * (1 - mystery_discount_rate) =
     biography_price * biography_quantity + mystery_price * mystery_quantity - total_savings) ∧
    total_discount_rate biography_discount_rate = 0.43 :=
by sorry

end discount_rate_sum_l3406_340642


namespace detergent_in_altered_solution_l3406_340652

/-- Represents the ratio of bleach, detergent, and water in a solution -/
structure SolutionRatio where
  bleach : ℕ
  detergent : ℕ
  water : ℕ

/-- Calculates the new ratio after altering the solution -/
def alter_ratio (r : SolutionRatio) : SolutionRatio :=
  { bleach := 3 * r.bleach,
    detergent := r.detergent,
    water := 2 * r.water }

/-- Theorem: Given the conditions, the altered solution contains 60 liters of detergent -/
theorem detergent_in_altered_solution 
  (original_ratio : SolutionRatio)
  (h_original : original_ratio = ⟨2, 40, 100⟩)
  (h_water : (alter_ratio original_ratio).water = 300) :
  (alter_ratio original_ratio).detergent = 60 :=
sorry

end detergent_in_altered_solution_l3406_340652


namespace least_digit_sum_multiple_2003_l3406_340609

/-- Sum of decimal digits of a natural number -/
def S (n : ℕ) : ℕ := sorry

/-- The least value of S(m) where m is a multiple of 2003 -/
theorem least_digit_sum_multiple_2003 : 
  (∃ m : ℕ, m % 2003 = 0 ∧ S m = 3) ∧ 
  (∀ m : ℕ, m % 2003 = 0 → S m ≥ 3) := by sorry

end least_digit_sum_multiple_2003_l3406_340609


namespace trevor_age_ratio_l3406_340688

/-- The ratio of Trevor's brother's age to Trevor's age 20 years ago -/
def age_ratio : ℚ := 16 / 3

theorem trevor_age_ratio :
  let trevor_age_decade_ago : ℕ := 16
  let brother_current_age : ℕ := 32
  let trevor_current_age : ℕ := trevor_age_decade_ago + 10
  let trevor_age_20_years_ago : ℕ := trevor_current_age - 20
  (brother_current_age : ℚ) / trevor_age_20_years_ago = age_ratio := by
  sorry

end trevor_age_ratio_l3406_340688


namespace sum_divisibility_l3406_340603

theorem sum_divisibility (y : ℕ) : 
  y = 36 + 48 + 72 + 144 + 216 + 432 + 1296 →
  3 ∣ y ∧ 4 ∣ y ∧ 6 ∣ y ∧ 12 ∣ y :=
by sorry

end sum_divisibility_l3406_340603


namespace equal_revenue_both_options_l3406_340600

/-- Represents the fishing company's financial model -/
structure FishingCompany where
  initial_cost : ℕ
  first_year_expenses : ℕ
  annual_expense_increase : ℕ
  annual_revenue : ℕ

/-- Calculates the net profit for a given number of years -/
def net_profit (company : FishingCompany) (years : ℕ) : ℤ :=
  (company.annual_revenue * years : ℤ) -
  ((company.first_year_expenses + (years - 1) * company.annual_expense_increase / 2) * years : ℤ) -
  company.initial_cost

/-- Calculates the total revenue when selling at maximum average annual profit -/
def revenue_max_avg_profit (company : FishingCompany) (sell_price : ℕ) : ℤ :=
  net_profit company 7 + sell_price

/-- Calculates the total revenue when selling at maximum total net profit -/
def revenue_max_total_profit (company : FishingCompany) (sell_price : ℕ) : ℤ :=
  net_profit company 10 + sell_price

/-- Theorem stating that both selling options result in the same total revenue -/
theorem equal_revenue_both_options (company : FishingCompany) :
  revenue_max_avg_profit company 2600000 = revenue_max_total_profit company 800000 :=
by sorry

end equal_revenue_both_options_l3406_340600


namespace symmetric_point_x_axis_l3406_340664

-- Define a point in 2D space
def Point := ℝ × ℝ

-- Define symmetry with respect to x-axis
def symmetricXAxis (p : Point) : Point :=
  (p.1, -p.2)

-- Theorem statement
theorem symmetric_point_x_axis :
  let M : Point := (3, -4)
  let M' : Point := symmetricXAxis M
  M' = (3, 4) := by sorry

end symmetric_point_x_axis_l3406_340664


namespace milk_price_increase_day_l3406_340693

/-- The day in June when the milk price increased -/
def price_increase_day : ℕ := 19

/-- The cost of milk before the price increase -/
def initial_price : ℕ := 1500

/-- The cost of milk after the price increase -/
def new_price : ℕ := 1600

/-- The total amount spent on milk in June -/
def total_spent : ℕ := 46200

/-- The number of days in June -/
def days_in_june : ℕ := 30

theorem milk_price_increase_day :
  (price_increase_day - 1) * initial_price +
  (days_in_june - (price_increase_day - 1)) * new_price = total_spent :=
by sorry

end milk_price_increase_day_l3406_340693


namespace scaled_badge_height_l3406_340628

/-- Calculates the height of a scaled rectangle while maintaining proportionality -/
def scaledHeight (originalWidth originalHeight scaledWidth : ℚ) : ℚ :=
  (originalHeight * scaledWidth) / originalWidth

/-- Theorem stating that scaling a 4x3 rectangle to width 12 results in height 9 -/
theorem scaled_badge_height :
  let originalWidth : ℚ := 4
  let originalHeight : ℚ := 3
  let scaledWidth : ℚ := 12
  scaledHeight originalWidth originalHeight scaledWidth = 9 := by
  sorry

end scaled_badge_height_l3406_340628


namespace boxes_with_neither_l3406_340666

theorem boxes_with_neither (total : ℕ) (pencils : ℕ) (pens : ℕ) (both : ℕ) 
  (h1 : total = 12)
  (h2 : pencils = 8)
  (h3 : pens = 5)
  (h4 : both = 3) :
  total - (pencils + pens - both) = 2 :=
by sorry

end boxes_with_neither_l3406_340666


namespace sons_age_l3406_340626

/-- Proves that given the conditions, the son's age is 35 years. -/
theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 37 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 35 := by
sorry

end sons_age_l3406_340626


namespace range_of_x_l3406_340633

theorem range_of_x (x : Real) 
  (h1 : 0 ≤ x ∧ x ≤ 2 * Real.pi)
  (h2 : Real.sqrt (1 - Real.sin (2 * x)) = Real.sin x - Real.cos x) :
  π / 4 ≤ x ∧ x ≤ 5 * π / 4 := by
  sorry

end range_of_x_l3406_340633


namespace sqrt_five_lt_sqrt_two_plus_one_l3406_340629

theorem sqrt_five_lt_sqrt_two_plus_one : Real.sqrt 5 < Real.sqrt 2 + 1 := by
  sorry

end sqrt_five_lt_sqrt_two_plus_one_l3406_340629


namespace intersection_A_B_l3406_340636

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}
def B : Set ℝ := {x : ℝ | |x| ≤ 1}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end intersection_A_B_l3406_340636


namespace fractional_linear_conjugacy_l3406_340682

/-- Given a fractional linear function f(x) = (ax + b) / (cx + d) where c ≠ 0 and ad ≠ bc,
    there exist functions φ and g such that f(x) = φ⁻¹(g(φ(x))). -/
theorem fractional_linear_conjugacy 
  {a b c d : ℝ} (hc : c ≠ 0) (had : a * d ≠ b * c) :
  ∃ (φ : ℝ → ℝ) (g : ℝ → ℝ),
    Function.Bijective φ ∧
    (∀ x, (a * x + b) / (c * x + d) = φ⁻¹ (g (φ x))) :=
by sorry

end fractional_linear_conjugacy_l3406_340682


namespace tower_height_proof_l3406_340685

def sum_of_arithmetic_series (n : ℕ) : ℕ := n * (n + 1) / 2

theorem tower_height_proof :
  let initial_blocks := 35
  let additional_blocks := 65
  let initial_height := sum_of_arithmetic_series initial_blocks
  let additional_height := sum_of_arithmetic_series additional_blocks
  initial_height + additional_height = 2775 :=
by sorry

end tower_height_proof_l3406_340685


namespace distance_to_school_l3406_340646

theorem distance_to_school (walking_speed run_speed : ℝ) 
  (run_distance total_time : ℝ) : 
  walking_speed = 70 →
  run_speed = 210 →
  run_distance = 600 →
  total_time ≤ 20 →
  ∃ (walk_distance : ℝ),
    walk_distance ≥ 0 ∧
    run_distance / run_speed + walk_distance / walking_speed ≤ total_time ∧
    walk_distance + run_distance ≤ 1800 := by
  sorry

end distance_to_school_l3406_340646


namespace water_ratio_is_two_to_one_l3406_340612

/-- Represents the water usage scenario of a water tower and four neighborhoods --/
structure WaterUsage where
  total : ℕ
  first : ℕ
  fourth : ℕ
  third_excess : ℕ

/-- Calculates the ratio of water used by the second neighborhood to the first neighborhood --/
def water_ratio (w : WaterUsage) : ℚ :=
  let second := (w.total - w.first - w.fourth - (w.total - w.first - w.fourth - w.third_excess)) / 2
  second / w.first

/-- Theorem stating that given the specific conditions, the water ratio is 2:1 --/
theorem water_ratio_is_two_to_one (w : WaterUsage) 
  (h1 : w.total = 1200)
  (h2 : w.first = 150)
  (h3 : w.fourth = 350)
  (h4 : w.third_excess = 100) :
  water_ratio w = 2 := by
  sorry

#eval water_ratio { total := 1200, first := 150, fourth := 350, third_excess := 100 }

end water_ratio_is_two_to_one_l3406_340612


namespace product_difference_sum_l3406_340663

theorem product_difference_sum (P Q R S : ℕ+) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P * Q = 120 →
  R * S = 120 →
  P - Q = R + S →
  P = 30 := by
sorry

end product_difference_sum_l3406_340663


namespace fire_in_city_a_l3406_340608

-- Define the cities
inductive City
| A
| B
| C

-- Define the possible statements
inductive Statement
| Fire
| LocationC

-- Define the behavior of residents in each city
def always_truth (c : City) : Prop :=
  c = City.A

def always_lie (c : City) : Prop :=
  c = City.B

def alternate (c : City) : Prop :=
  c = City.C

-- Define the caller's statements
def caller_statements : List Statement :=
  [Statement.Fire, Statement.LocationC]

-- Define the property of the actual fire location
def is_actual_fire_location (c : City) : Prop :=
  ∀ (s : Statement), s ∈ caller_statements → 
    (always_truth c → s = Statement.Fire) ∧
    (always_lie c → s ≠ Statement.LocationC) ∧
    (alternate c → (s = Statement.Fire ↔ s ≠ Statement.LocationC))

-- Theorem: The actual fire location is City A
theorem fire_in_city_a :
  is_actual_fire_location City.A :=
sorry

end fire_in_city_a_l3406_340608


namespace stationery_store_problem_l3406_340610

/-- Represents the cost and quantity of pencils in a packet -/
structure Packet where
  cost : ℝ
  quantity : ℝ

/-- The stationery store problem -/
theorem stationery_store_problem (a : ℝ) (h_pos : a > 0) :
  let s : Packet := ⟨a, 1⟩
  let m : Packet := ⟨1.2 * a, 1.5⟩
  let l : Packet := ⟨1.6 * a, 1.875⟩
  (m.cost / m.quantity < l.cost / l.quantity) ∧
  (l.cost / l.quantity < s.cost / s.quantity) := by
  sorry

#check stationery_store_problem

end stationery_store_problem_l3406_340610


namespace prime_factors_count_l3406_340678

/-- The number of positive divisors of n -/
def d (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The main expression in the problem -/
def f (n : ℕ) : ℕ := (n^(2*n) + n^n + n + 1)^(2*n) + (n^(2*n) + n^n + n + 1)^n + 1

/-- The theorem statement -/
theorem prime_factors_count (n : ℕ) (h : ¬3 ∣ n) : 
  2 * d n ≤ (Nat.factors (f n)).card := by
  sorry

end prime_factors_count_l3406_340678


namespace baseball_gear_sale_l3406_340607

theorem baseball_gear_sale (bat_price glove_original_price glove_discount cleats_price total_amount : ℝ)
  (h1 : bat_price = 10)
  (h2 : glove_original_price = 30)
  (h3 : glove_discount = 0.2)
  (h4 : cleats_price = 10)
  (h5 : total_amount = 79) :
  let glove_sale_price := glove_original_price * (1 - glove_discount)
  let other_gear_total := bat_price + glove_sale_price + 2 * cleats_price
  total_amount - other_gear_total = 25 := by
sorry

end baseball_gear_sale_l3406_340607


namespace minimum_k_value_l3406_340617

theorem minimum_k_value (m n : ℕ+) :
  (1 : ℝ) / (m + n : ℝ)^2 ≤ (1/8) * ((1 : ℝ) / m^2 + 1 / n^2) ∧
  ∀ k : ℝ, (∀ a b : ℕ+, (1 : ℝ) / (a + b : ℝ)^2 ≤ k * ((1 : ℝ) / a^2 + 1 / b^2)) →
    k ≥ 1/8 :=
by sorry

end minimum_k_value_l3406_340617


namespace jack_walking_time_l3406_340654

/-- Represents the walking parameters and time for a person -/
structure WalkingData where
  steps_per_minute : ℕ
  step_length : ℕ
  time_to_school : ℚ

/-- Calculates the distance walked based on walking data -/
def distance_walked (data : WalkingData) : ℚ :=
  (data.steps_per_minute : ℚ) * (data.step_length : ℚ) * data.time_to_school / 100

theorem jack_walking_time 
  (dave : WalkingData)
  (jack : WalkingData)
  (h1 : dave.steps_per_minute = 80)
  (h2 : dave.step_length = 80)
  (h3 : dave.time_to_school = 20)
  (h4 : jack.steps_per_minute = 120)
  (h5 : jack.step_length = 50)
  (h6 : distance_walked dave = distance_walked jack) :
  jack.time_to_school = 64/3 := by
  sorry

end jack_walking_time_l3406_340654


namespace cadence_total_earnings_l3406_340695

/-- Calculates the total earnings of Cadence from two companies given the specified conditions. -/
theorem cadence_total_earnings :
  let old_company_years : ℚ := 3.5
  let old_company_monthly_salary : ℚ := 5000
  let old_company_bonus_rate : ℚ := 0.5
  let new_company_years : ℕ := 4
  let new_company_salary_raise : ℚ := 0.2
  let new_company_bonus_rate : ℚ := 1
  let third_year_deduction_rate : ℚ := 0.02

  let old_company_salary := old_company_years * 12 * old_company_monthly_salary
  let old_company_bonus := (old_company_years.floor * old_company_bonus_rate * old_company_monthly_salary) +
                           (old_company_years - old_company_years.floor) * old_company_bonus_rate * old_company_monthly_salary
  let new_company_monthly_salary := old_company_monthly_salary * (1 + new_company_salary_raise)
  let new_company_salary := new_company_years * 12 * new_company_monthly_salary
  let new_company_bonus := new_company_years * new_company_bonus_rate * new_company_monthly_salary
  let third_year_deduction := third_year_deduction_rate * 12 * new_company_monthly_salary

  let total_earnings := old_company_salary + old_company_bonus + new_company_salary + new_company_bonus - third_year_deduction

  total_earnings = 529310 := by
    sorry

end cadence_total_earnings_l3406_340695


namespace specific_tetrahedron_volume_l3406_340644

/-- Represents a tetrahedron PQRS with given edge lengths -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating that the volume of the specific tetrahedron is 10 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    PQ := 3,
    PR := 4,
    PS := 5,
    QR := 5,
    QS := Real.sqrt 34,
    RS := Real.sqrt 41
  }
  tetrahedronVolume t = 10 := by sorry

end specific_tetrahedron_volume_l3406_340644


namespace intersection_of_P_and_Q_l3406_340667

def P : Set ℝ := {1, 2, 3}
def Q : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_of_P_and_Q : P ∩ Q = {1, 2} := by sorry

end intersection_of_P_and_Q_l3406_340667


namespace point_c_values_l3406_340632

/-- Represents a point on a number line --/
structure Point where
  value : ℝ

/-- The distance between two points on a number line --/
def distance (p q : Point) : ℝ := |p.value - q.value|

theorem point_c_values (b c : Point) : 
  b.value = 3 → distance b c = 2 → (c.value = 1 ∨ c.value = 5) := by
  sorry

end point_c_values_l3406_340632


namespace exam_average_problem_l3406_340698

theorem exam_average_problem :
  ∀ (N : ℕ),
  (15 : ℝ) * 70 + (10 : ℝ) * 95 = (N : ℝ) * 80 →
  N = 25 :=
by
  sorry

end exam_average_problem_l3406_340698


namespace isosceles_triangle_coordinates_l3406_340631

-- Define the points
def A : ℝ × ℝ := (13, 7)
def B : ℝ × ℝ := (5, -1)
def D : ℝ × ℝ := (2, 2)

-- Define the theorem
theorem isosceles_triangle_coordinates :
  ∃ (C : ℝ × ℝ),
    -- AB = AC (isosceles triangle)
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
    -- AD ⟂ BC (altitude condition)
    (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0 ∧
    -- D is on BC
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ D.1 = t * B.1 + (1 - t) * C.1 ∧ D.2 = t * B.2 + (1 - t) * C.2 ∧
    -- C has coordinates (-1, 5)
    C = (-1, 5) := by
  sorry

end isosceles_triangle_coordinates_l3406_340631


namespace ship_length_proof_l3406_340634

/-- The length of the ship in terms of Emily's normal steps -/
def ship_length : ℕ := 120

/-- The number of steps Emily takes with wind behind her -/
def steps_with_wind : ℕ := 300

/-- The number of steps Emily takes against the wind -/
def steps_against_wind : ℕ := 75

/-- The number of extra steps the wind allows Emily to take in the direction it blows -/
def wind_effect : ℕ := 20

theorem ship_length_proof :
  ∀ (E S : ℝ),
  E > 0 ∧ S > 0 →
  (steps_with_wind + wind_effect : ℝ) * E = ship_length + (steps_with_wind + wind_effect) * S →
  (steps_against_wind - wind_effect : ℝ) * E = ship_length - (steps_against_wind - wind_effect) * S →
  ship_length = 120 := by
  sorry

end ship_length_proof_l3406_340634


namespace cost_price_of_toy_cost_price_is_1300_l3406_340670

/-- The cost price of a toy given the selling conditions -/
theorem cost_price_of_toy (num_toys : ℕ) (selling_price : ℕ) (gain_toys : ℕ) : ℕ :=
  let cost_price := selling_price / (num_toys + gain_toys)
  cost_price

/-- Proof that the cost price of a toy is 1300 under given conditions -/
theorem cost_price_is_1300 : cost_price_of_toy 18 27300 3 = 1300 := by
  sorry

end cost_price_of_toy_cost_price_is_1300_l3406_340670


namespace fifteenth_thirtyseventh_415th_digit_l3406_340694

/-- The decimal representation of 15/37 has a repeating sequence of '405'. -/
def decimal_rep : ℚ → List ℕ := sorry

/-- The nth digit after the decimal point in the decimal representation of a rational number. -/
def nth_digit (q : ℚ) (n : ℕ) : ℕ := sorry

/-- The 415th digit after the decimal point in the decimal representation of 15/37 is 4. -/
theorem fifteenth_thirtyseventh_415th_digit :
  nth_digit (15 / 37) 415 = 4 := by sorry

end fifteenth_thirtyseventh_415th_digit_l3406_340694


namespace calculation_proof_l3406_340653

theorem calculation_proof : (3127 - 2972)^3 / 343 = 125 := by
  sorry

end calculation_proof_l3406_340653


namespace exists_n_order_of_two_congruent_l3406_340647

/-- The order of 2 in n! -/
def v (n : ℕ) : ℕ := sorry

/-- For any positive integers a and m, there exists n > 1 such that v(n) ≡ a (mod m) -/
theorem exists_n_order_of_two_congruent (a m : ℕ+) : ∃ n : ℕ, n > 1 ∧ v n % m = a % m := by
  sorry

end exists_n_order_of_two_congruent_l3406_340647


namespace cos_135_degrees_l3406_340684

theorem cos_135_degrees : Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end cos_135_degrees_l3406_340684


namespace tom_weekly_fee_l3406_340677

/-- Represents Tom's car leasing scenario -/
structure CarLease where
  miles_mon_wed_fri : ℕ  -- Miles driven on Monday, Wednesday, Friday
  miles_other_days : ℕ   -- Miles driven on other days
  cost_per_mile : ℚ      -- Cost per mile in dollars
  total_annual_payment : ℚ -- Total annual payment in dollars

/-- Calculate the weekly fee given a car lease scenario -/
def weekly_fee (lease : CarLease) : ℚ :=
  let weekly_miles := 3 * lease.miles_mon_wed_fri + 4 * lease.miles_other_days
  let weekly_mileage_cost := weekly_miles * lease.cost_per_mile
  let annual_mileage_cost := 52 * weekly_mileage_cost
  (lease.total_annual_payment - annual_mileage_cost) / 52

/-- Theorem stating that the weekly fee for Tom's scenario is $95 -/
theorem tom_weekly_fee :
  let tom_lease := CarLease.mk 50 100 (1/10) 7800
  weekly_fee tom_lease = 95 := by sorry

end tom_weekly_fee_l3406_340677


namespace cupcake_price_correct_l3406_340669

/-- The original price of cupcakes before the discount -/
def original_cupcake_price : ℝ := 3

/-- The original price of cookies before the discount -/
def original_cookie_price : ℝ := 2

/-- The number of cupcakes sold -/
def cupcakes_sold : ℕ := 16

/-- The number of cookies sold -/
def cookies_sold : ℕ := 8

/-- The total revenue from the sale -/
def total_revenue : ℝ := 32

/-- Theorem stating that the original cupcake price satisfies the given conditions -/
theorem cupcake_price_correct : 
  cupcakes_sold * (original_cupcake_price / 2) + cookies_sold * (original_cookie_price / 2) = total_revenue :=
by sorry

end cupcake_price_correct_l3406_340669


namespace path_area_and_cost_calculation_l3406_340668

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path -/
def construction_cost (area cost_per_sqm : ℝ) : ℝ :=
  area * cost_per_sqm

theorem path_area_and_cost_calculation 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_sqm : ℝ) 
  (h1 : field_length = 75) 
  (h2 : field_width = 55) 
  (h3 : path_width = 2.5) 
  (h4 : cost_per_sqm = 7) : 
  path_area field_length field_width path_width = 675 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_sqm = 4725 :=
by
  sorry

end path_area_and_cost_calculation_l3406_340668


namespace game_cost_l3406_340627

def initial_money : ℕ := 12
def toy_cost : ℕ := 2
def num_toys : ℕ := 2

theorem game_cost : 
  initial_money - (toy_cost * num_toys) = 8 :=
by
  sorry

end game_cost_l3406_340627


namespace greatest_a_value_l3406_340656

theorem greatest_a_value (a : ℝ) : 
  (7 * Real.sqrt ((2 * a) ^ 2 + 1 ^ 2) - 4 * a ^ 2 - 1) / (Real.sqrt (1 + 4 * a ^ 2) + 3) = 2 →
  a ≤ Real.sqrt 2 :=
sorry

end greatest_a_value_l3406_340656


namespace range_of_m_value_of_m_l3406_340658

-- Define the quadratic equation
def quadratic_eq (m x : ℝ) : Prop :=
  x^2 - 2*(1-m)*x + m^2 = 0

-- Define the roots of the equation
def roots (m x₁ x₂ : ℝ) : Prop :=
  quadratic_eq m x₁ ∧ quadratic_eq m x₂ ∧ x₁ ≠ x₂

-- Define the additional condition
def additional_condition (m x₁ x₂ : ℝ) : Prop :=
  x₁^2 + 12*m + x₂^2 = 10

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x₁ x₂, roots m x₁ x₂) → m ≤ 1/2 :=
sorry

-- Theorem for the value of m given the additional condition
theorem value_of_m (m x₁ x₂ : ℝ) :
  roots m x₁ x₂ → additional_condition m x₁ x₂ → m = -3 :=
sorry

end range_of_m_value_of_m_l3406_340658


namespace union_condition_intersection_condition_l3406_340605

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 1}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem 1
theorem union_condition (a : ℝ) : A a ∪ B = A a ↔ 0 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem 2
theorem intersection_condition (a : ℝ) : A a ∩ B = ∅ ↔ a ≤ -1/2 ∨ a ≥ 2 := by sorry

end union_condition_intersection_condition_l3406_340605


namespace zebra_catches_tiger_l3406_340673

/-- The time it takes for a zebra to catch a tiger given their speeds and the tiger's head start -/
theorem zebra_catches_tiger (zebra_speed tiger_speed : ℝ) (head_start : ℝ) : 
  zebra_speed = 55 →
  tiger_speed = 30 →
  head_start = 5 →
  (head_start * tiger_speed) / (zebra_speed - tiger_speed) = 6 := by
  sorry

end zebra_catches_tiger_l3406_340673


namespace parabola_translation_l3406_340641

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 - 1

-- Define the translation
def left_translation : ℝ := 2
def up_translation : ℝ := 1

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x + left_translation)^2

-- Theorem statement
theorem parabola_translation :
  ∀ x y : ℝ, y = original_parabola (x + left_translation) + up_translation 
  ↔ y = translated_parabola x := by sorry

end parabola_translation_l3406_340641


namespace assignments_count_l3406_340623

/-- The number of interest groups available --/
def num_groups : ℕ := 3

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of ways to assign students to interest groups --/
def num_assignments : ℕ := num_groups ^ num_students

/-- Theorem stating that the number of assignments is 81 --/
theorem assignments_count : num_assignments = 81 := by
  sorry

end assignments_count_l3406_340623


namespace library_books_count_l3406_340675

theorem library_books_count :
  ∀ (total_books : ℕ),
    (total_books : ℝ) * 0.8 * 0.4 = 736 →
    total_books = 2300 :=
by
  sorry

end library_books_count_l3406_340675


namespace square_sum_ge_double_product_l3406_340604

theorem square_sum_ge_double_product (a b : ℝ) : (a^2 + b^2 > 2*a*b) ∨ (a^2 + b^2 = 2*a*b) := by
  sorry

end square_sum_ge_double_product_l3406_340604


namespace marks_books_count_l3406_340614

/-- Given that Mark started with $85, each book costs $5, and he is left with $35, 
    prove that the number of books he bought is 10. -/
theorem marks_books_count (initial_amount : ℕ) (book_cost : ℕ) (remaining_amount : ℕ) : ℕ :=
  sorry

end marks_books_count_l3406_340614


namespace multiply_binomial_l3406_340649

theorem multiply_binomial (x : ℝ) : (-2*x)*(x - 3) = -2*x^2 + 6*x := by
  sorry

end multiply_binomial_l3406_340649


namespace exact_blue_marbles_probability_l3406_340665

def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def trials : ℕ := 6
def blue_selections : ℕ := 3

def probability_blue : ℚ := blue_marbles / total_marbles
def probability_red : ℚ := red_marbles / total_marbles

theorem exact_blue_marbles_probability :
  (Nat.choose trials blue_selections : ℚ) *
  probability_blue ^ blue_selections *
  probability_red ^ (trials - blue_selections) =
  3512320 / 11390625 := by
sorry

end exact_blue_marbles_probability_l3406_340665


namespace min_operations_to_2187_l3406_340618

/-- Represents the possible operations on the calculator --/
inductive Operation
  | AddOne
  | TimesThree

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.TimesThree => n * 3

/-- Checks if a sequence of operations transforms 1 into the target --/
def isValidSequence (ops : List Operation) (target : ℕ) : Prop :=
  ops.foldl applyOperation 1 = target

/-- The main theorem to prove --/
theorem min_operations_to_2187 :
  ∃ (ops : List Operation), isValidSequence ops 2187 ∧ 
    ops.length = 7 ∧ 
    (∀ (other_ops : List Operation), isValidSequence other_ops 2187 → other_ops.length ≥ 7) :=
sorry

end min_operations_to_2187_l3406_340618


namespace polynomial_roots_l3406_340639

theorem polynomial_roots : 
  let p : ℝ → ℝ := λ x => x^3 + 2*x^2 - 5*x - 6
  ∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 2 ∨ x = -3 := by
sorry

end polynomial_roots_l3406_340639


namespace correct_calculation_l3406_340625

theorem correct_calculation (a : ℝ) : 8 * a^2 - 5 * a^2 = 3 * a^2 := by
  sorry

end correct_calculation_l3406_340625


namespace sin_alpha_plus_beta_equals_one_l3406_340671

theorem sin_alpha_plus_beta_equals_one 
  (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1) 
  (h2 : Real.cos α + Real.sin β = Real.sqrt 3) : 
  Real.sin (α + β) = 1 := by
sorry

end sin_alpha_plus_beta_equals_one_l3406_340671


namespace blood_concentration_reaches_target_target_time_is_correct_l3406_340611

/-- Represents the blood drug concentration at a given time -/
def blood_concentration (peak_concentration : ℝ) (time : ℕ) : ℝ :=
  if time ≤ 3 then peak_concentration
  else peak_concentration * (0.4 ^ ((time - 3) / 2))

/-- Theorem stating that the blood concentration reaches 1.024% of peak after 13 hours -/
theorem blood_concentration_reaches_target (peak_concentration : ℝ) :
  blood_concentration peak_concentration 13 = 0.01024 * peak_concentration :=
by
  sorry

/-- Time when blood concentration reaches 1.024% of peak -/
def target_time : ℕ := 13

/-- Theorem proving that target_time is correct -/
theorem target_time_is_correct (peak_concentration : ℝ) :
  blood_concentration peak_concentration target_time = 0.01024 * peak_concentration :=
by
  sorry

end blood_concentration_reaches_target_target_time_is_correct_l3406_340611


namespace one_fourth_difference_product_sum_l3406_340676

theorem one_fourth_difference_product_sum : 
  (1 / 4 : ℚ) * ((9 * 5) - (7 + 3)) = 35 / 4 := by
  sorry

end one_fourth_difference_product_sum_l3406_340676


namespace inequality_solution_l3406_340699

theorem inequality_solution (a : ℝ) : 
  (∀ x : ℝ, (x < 1 ∨ x > 3) ↔ (a * x) / (x - 1) < 1) → 
  a = 2/3 := by
sorry

end inequality_solution_l3406_340699


namespace daily_harvest_l3406_340621

/-- The number of sacks harvested from each section daily -/
def sacks_per_section : ℕ := 65

/-- The number of sections in the orchard -/
def number_of_sections : ℕ := 12

/-- The total number of sacks harvested daily -/
def total_sacks : ℕ := sacks_per_section * number_of_sections

theorem daily_harvest : total_sacks = 780 := by
  sorry

end daily_harvest_l3406_340621


namespace problem_statement_l3406_340681

theorem problem_statement :
  (∀ x : ℝ, x^2 - x ≥ x - 1) ∧
  (∃ x : ℝ, x > 1 ∧ x + 4 / (x - 1) = 6) ∧
  (∀ x : ℝ, x > 2 → Real.sqrt (x^2 + 1) + 4 / Real.sqrt (x^2 + 1) ≥ 4) := by
sorry

end problem_statement_l3406_340681


namespace area_of_U_l3406_340689

noncomputable section

/-- A regular octagon in the complex plane -/
def RegularOctagon : Set ℂ :=
  sorry

/-- The region outside the regular octagon -/
def T : Set ℂ :=
  { z : ℂ | z ∉ RegularOctagon }

/-- The region U, which is the image of T under the transformation z ↦ 1/z -/
def U : Set ℂ :=
  { w : ℂ | ∃ z ∈ T, w = 1 / z }

/-- The area of a set in the complex plane -/
def area : Set ℂ → ℝ :=
  sorry

/-- The main theorem: The area of region U is 4 + 4π -/
theorem area_of_U : area U = 4 + 4 * Real.pi :=
  sorry

end area_of_U_l3406_340689


namespace relay_race_last_year_distance_l3406_340691

/-- Represents the relay race setup and calculations -/
def RelayRace (tables : ℕ) (distance_between_1_and_3 : ℝ) (multiplier : ℝ) : Prop :=
  let segment_length := distance_between_1_and_3 / 2
  let total_segments := tables - 1
  let this_year_distance := segment_length * total_segments
  let last_year_distance := this_year_distance / multiplier
  (tables = 6) ∧
  (distance_between_1_and_3 = 400) ∧
  (multiplier = 4) ∧
  (last_year_distance = 250)

/-- Theorem stating that given the conditions, the race distance last year was 250 meters -/
theorem relay_race_last_year_distance :
  ∀ (tables : ℕ) (distance_between_1_and_3 : ℝ) (multiplier : ℝ),
  RelayRace tables distance_between_1_and_3 multiplier :=
by
  sorry

end relay_race_last_year_distance_l3406_340691


namespace employed_females_percentage_l3406_340690

theorem employed_females_percentage (total_population employed_population employed_males : ℝ) :
  employed_population / total_population = 0.7 →
  employed_males / total_population = 0.21 →
  (employed_population - employed_males) / employed_population = 0.7 := by
  sorry

end employed_females_percentage_l3406_340690


namespace percent_sum_of_x_l3406_340613

theorem percent_sum_of_x (x y z v w : ℝ) : 
  (0.45 * z = 0.39 * y) →
  (y = 0.75 * x) →
  (v = 0.80 * z) →
  (w = 0.60 * y) →
  (v + w = 0.97 * x) :=
by sorry

end percent_sum_of_x_l3406_340613


namespace road_trip_mileage_l3406_340686

/-- Calculates the final mileage of a car after a road trip -/
def final_mileage (initial_mileage : ℕ) (efficiency : ℕ) (tank_capacity : ℕ) (refills : ℕ) : ℕ :=
  initial_mileage + efficiency * tank_capacity * refills

theorem road_trip_mileage :
  final_mileage 1728 30 20 2 = 2928 := by
  sorry

end road_trip_mileage_l3406_340686


namespace complement_of_union_l3406_340659

-- Define the universe set U
def U : Finset Nat := {0, 1, 2, 3, 4}

-- Define set M
def M : Finset Nat := {0, 4}

-- Define set N
def N : Finset Nat := {2, 4}

-- Theorem statement
theorem complement_of_union :
  (U \ (M ∪ N)) = {1, 3} := by sorry

end complement_of_union_l3406_340659


namespace perpendicular_lines_exist_l3406_340651

/-- Two lines l₁ and l₂ in the plane -/
structure Lines where
  a : ℝ
  l₁ : ℝ × ℝ → Prop
  l₂ : ℝ × ℝ → Prop
  h₁ : ∀ x y, l₁ (x, y) ↔ x + a * y = 3
  h₂ : ∀ x y, l₂ (x, y) ↔ 3 * x - (a - 2) * y = 2

/-- Perpendicularity condition for two lines -/
def perpendicular (l : Lines) : Prop :=
  1 * 3 + l.a * -(l.a - 2) = 0

/-- Theorem: If the lines are perpendicular, then there exists a real number a satisfying the condition -/
theorem perpendicular_lines_exist (l : Lines) (h : perpendicular l) : 
  ∃ a : ℝ, 1 * 3 + a * -(a - 2) = 0 := by
  sorry

end perpendicular_lines_exist_l3406_340651


namespace expression_bounds_l3406_340657

theorem expression_bounds (a b c d x : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ x) (hc : 0 ≤ c ∧ c ≤ x) (hd : 0 ≤ d ∧ d ≤ x)
  (hx : 0 < x ∧ x ≤ 10) : 
  2 * x * Real.sqrt 2 ≤ 
    Real.sqrt (a^2 + (x - b)^2) + Real.sqrt (b^2 + (x - c)^2) + 
    Real.sqrt (c^2 + (x - d)^2) + Real.sqrt (d^2 + (x - a)^2) ∧
  Real.sqrt (a^2 + (x - b)^2) + Real.sqrt (b^2 + (x - c)^2) + 
    Real.sqrt (c^2 + (x - d)^2) + Real.sqrt (d^2 + (x - a)^2) ≤ 4 * x ∧
  ∃ (a' b' c' d' : ℝ), 
    0 ≤ a' ∧ a' ≤ x ∧ 0 ≤ b' ∧ b' ≤ x ∧ 0 ≤ c' ∧ c' ≤ x ∧ 0 ≤ d' ∧ d' ≤ x ∧
    Real.sqrt (a'^2 + (x - b')^2) + Real.sqrt (b'^2 + (x - c')^2) + 
    Real.sqrt (c'^2 + (x - d')^2) + Real.sqrt (d'^2 + (x - a')^2) = 2 * x * Real.sqrt 2 ∧
  ∃ (a'' b'' c'' d'' : ℝ), 
    0 ≤ a'' ∧ a'' ≤ x ∧ 0 ≤ b'' ∧ b'' ≤ x ∧ 0 ≤ c'' ∧ c'' ≤ x ∧ 0 ≤ d'' ∧ d'' ≤ x ∧
    Real.sqrt (a''^2 + (x - b''^2)) + Real.sqrt (b''^2 + (x - c''^2)) + 
    Real.sqrt (c''^2 + (x - d''^2)) + Real.sqrt (d''^2 + (x - a''^2)) = 4 * x :=
by sorry

end expression_bounds_l3406_340657


namespace base_r_transaction_l3406_340674

/-- Represents a number in base r --/
def BaseR (digits : List Nat) (r : Nat) : Nat :=
  digits.foldr (fun d acc => d + r * acc) 0

/-- The problem statement --/
theorem base_r_transaction (r : Nat) : 
  (BaseR [5, 2, 1] r) + (BaseR [1, 1, 0] r) - (BaseR [3, 7, 1] r) = (BaseR [1, 0, 0, 2] r) →
  r^3 - 3*r^2 + 4*r - 2 = 0 := by
  sorry

end base_r_transaction_l3406_340674


namespace jessie_muffins_theorem_l3406_340635

/-- The number of muffins made when Jessie and her friends each receive an equal amount -/
def total_muffins (num_friends : ℕ) (muffins_per_person : ℕ) : ℕ :=
  (num_friends + 1) * muffins_per_person

/-- Theorem stating that when Jessie has 4 friends and each person gets 4 muffins, the total is 20 -/
theorem jessie_muffins_theorem :
  total_muffins 4 4 = 20 := by
  sorry

end jessie_muffins_theorem_l3406_340635


namespace max_value_of_f_l3406_340683

/-- The domain of the function f -/
def Domain (c d e : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0 ∧ c - d * p.1 - e * p.2 > 0}

/-- The function f -/
def f (a b c d e : ℝ) (p : ℝ × ℝ) : ℝ :=
  a * p.1 * b * p.2 * (c - d * p.1 - e * p.2)

/-- Theorem stating the maximum value of f -/
theorem max_value_of_f (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  ∃ M : ℝ, M = (a / d) * (b / e) * (c / 3)^3 ∧
  ∀ p ∈ Domain c d e, f a b c d e p ≤ M :=
sorry

end max_value_of_f_l3406_340683


namespace smallest_common_flock_size_l3406_340643

theorem smallest_common_flock_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 13 = 0 ∧ 
  n % 14 = 0 ∧ 
  (∀ m : ℕ, m > 0 → m % 13 = 0 → m % 14 = 0 → m ≥ n) ∧
  n = 182 := by
sorry

end smallest_common_flock_size_l3406_340643


namespace imaginary_part_of_product_l3406_340619

theorem imaginary_part_of_product : Complex.im ((3 - 4*Complex.I) * (1 + 2*Complex.I)) = 2 := by
  sorry

end imaginary_part_of_product_l3406_340619


namespace zoo_ticket_price_l3406_340672

theorem zoo_ticket_price :
  let monday_children : ℕ := 7
  let monday_adults : ℕ := 5
  let tuesday_children : ℕ := 4
  let tuesday_adults : ℕ := 2
  let child_ticket_price : ℕ := 3
  let total_revenue : ℕ := 61
  ∃ (adult_ticket_price : ℕ),
    (monday_children * child_ticket_price + monday_adults * adult_ticket_price) +
    (tuesday_children * child_ticket_price + tuesday_adults * adult_ticket_price) = total_revenue ∧
    adult_ticket_price = 4 :=
by
  sorry

end zoo_ticket_price_l3406_340672


namespace smallest_value_complex_sum_l3406_340620

theorem smallest_value_complex_sum (p q r : ℤ) (ω : ℂ) : 
  p ≠ q → q ≠ r → r ≠ p → 
  (p = 0 ∨ q = 0 ∨ r = 0) →
  ω^3 = 1 →
  ω ≠ 1 →
  ∃ (min : ℝ), min = Real.sqrt 3 ∧ 
    (∀ (p' q' r' : ℤ), p' ≠ q' → q' ≠ r' → r' ≠ p' → 
      (p' = 0 ∨ q' = 0 ∨ r' = 0) → 
      Complex.abs (↑p' + ↑q' * ω^2 + ↑r' * ω) ≥ min) ∧
    (Complex.abs (↑p + ↑q * ω^2 + ↑r * ω) = min ∨
     Complex.abs (↑q + ↑r * ω^2 + ↑p * ω) = min ∨
     Complex.abs (↑r + ↑p * ω^2 + ↑q * ω) = min) :=
by sorry

end smallest_value_complex_sum_l3406_340620


namespace integer_tuple_solution_l3406_340697

theorem integer_tuple_solution :
  ∀ a b c x y z : ℕ,
    a + b + c = x * y * z →
    x + y + z = a * b * c →
    a ≥ b →
    b ≥ c →
    c ≥ 1 →
    x ≥ y →
    y ≥ z →
    z ≥ 1 →
    ((a = 2 ∧ b = 2 ∧ c = 2 ∧ x = 6 ∧ y = 1 ∧ z = 1) ∨
     (a = 5 ∧ b = 2 ∧ c = 1 ∧ x = 8 ∧ y = 1 ∧ z = 1) ∨
     (a = 3 ∧ b = 3 ∧ c = 1 ∧ x = 7 ∧ y = 1 ∧ z = 1) ∨
     (a = 3 ∧ b = 2 ∧ c = 1 ∧ x = 3 ∧ y = 2 ∧ z = 1)) :=
by
  sorry

end integer_tuple_solution_l3406_340697


namespace probability_different_colors_7_5_l3406_340606

/-- The probability of drawing two chips of different colors from a bag -/
def probability_different_colors (blue_chips yellow_chips : ℕ) : ℚ :=
  let total_chips := blue_chips + yellow_chips
  let prob_blue_then_yellow := (blue_chips : ℚ) / total_chips * yellow_chips / (total_chips - 1)
  let prob_yellow_then_blue := (yellow_chips : ℚ) / total_chips * blue_chips / (total_chips - 1)
  prob_blue_then_yellow + prob_yellow_then_blue

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem probability_different_colors_7_5 :
  probability_different_colors 7 5 = 35 / 66 := by
  sorry

end probability_different_colors_7_5_l3406_340606


namespace apartment_ratio_l3406_340650

theorem apartment_ratio (total_floors : ℕ) (max_residents : ℕ) 
  (h1 : total_floors = 12)
  (h2 : max_residents = 264) :
  ∃ (floors_with_6 : ℕ) (floors_with_5 : ℕ),
    floors_with_6 + floors_with_5 = total_floors ∧
    6 * floors_with_6 + 5 * floors_with_5 = max_residents / 4 ∧
    floors_with_6 * 2 = total_floors := by
  sorry

end apartment_ratio_l3406_340650


namespace quadratic_equation_solution_l3406_340687

theorem quadratic_equation_solution (x c d : ℕ) (h1 : x^2 + 14*x = 72) 
  (h2 : x = Int.sqrt c - d) (h3 : 0 < c) (h4 : 0 < d) : c + d = 128 := by
  sorry

end quadratic_equation_solution_l3406_340687


namespace winner_depends_on_n_l3406_340601

/-- Represents a player in the game -/
inductive Player
| Bela
| Jenn

/-- Represents the game state -/
structure GameState where
  n : ℕ
  choices : List ℝ

/-- Checks if a move is valid -/
def is_valid_move (state : GameState) (move : ℝ) : Prop :=
  0 ≤ move ∧ move ≤ state.n ∧ ∀ c ∈ state.choices, |move - c| > 1.5

/-- Determines if the game is over -/
def is_game_over (state : GameState) : Prop :=
  ∀ move, ¬(is_valid_move state move)

/-- Determines the winner of the game -/
def winner (state : GameState) : Player :=
  if state.choices.length % 2 = 0 then Player.Jenn else Player.Bela

/-- The main theorem stating that the winner depends on the specific value of n -/
theorem winner_depends_on_n :
  ∃ n m : ℕ,
    n > 5 ∧ m > 5 ∧
    (∃ state1 : GameState, state1.n = n ∧ is_game_over state1 ∧ winner state1 = Player.Bela) ∧
    (∃ state2 : GameState, state2.n = m ∧ is_game_over state2 ∧ winner state2 = Player.Jenn) :=
  sorry


end winner_depends_on_n_l3406_340601


namespace imaginary_part_of_f_i_over_i_l3406_340645

-- Define the complex function f(x) = x^3 - 1
def f (x : ℂ) : ℂ := x^3 - 1

-- State the theorem
theorem imaginary_part_of_f_i_over_i :
  Complex.im (f Complex.I / Complex.I) = 1 := by
  sorry

end imaginary_part_of_f_i_over_i_l3406_340645


namespace class_mean_score_l3406_340648

theorem class_mean_score 
  (n : ℕ) 
  (h1 : n > 15) 
  (overall_mean : ℝ) 
  (h2 : overall_mean = 10) 
  (group_mean : ℝ) 
  (h3 : group_mean = 16) : 
  let remaining_mean := (n * overall_mean - 15 * group_mean) / (n - 15)
  remaining_mean = (10 * n - 240) / (n - 15) := by
sorry

end class_mean_score_l3406_340648


namespace turkey_cost_per_employee_l3406_340696

/-- The cost of turkeys for employees --/
theorem turkey_cost_per_employee (num_employees : ℕ) (total_cost : ℚ) : 
  num_employees = 85 → total_cost = 2125 → (total_cost / num_employees : ℚ) = 25 := by
  sorry

end turkey_cost_per_employee_l3406_340696


namespace sin_75_times_sin_15_l3406_340662

theorem sin_75_times_sin_15 :
  Real.sin (75 * π / 180) * Real.sin (15 * π / 180) = 1 / 4 := by sorry

end sin_75_times_sin_15_l3406_340662


namespace boat_stream_speed_l3406_340615

/-- Proves that the speed of a stream is 5 km/hr given the conditions of the boat problem -/
theorem boat_stream_speed 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : boat_speed = 22) 
  (h2 : distance = 81) 
  (h3 : time = 3) : 
  ∃ stream_speed : ℝ, 
    stream_speed = 5 ∧ 
    (boat_speed + stream_speed) * time = distance := by
  sorry

end boat_stream_speed_l3406_340615


namespace greatest_integer_for_fraction_twenty_nine_satisfies_twenty_nine_is_greatest_l3406_340630

def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n

theorem greatest_integer_for_fraction : 
  ∀ x : ℤ, (is_integer ((x^2 + 3*x + 8) / (x - 3))) → x ≤ 29 :=
by sorry

theorem twenty_nine_satisfies :
  is_integer ((29^2 + 3*29 + 8) / (29 - 3)) :=
by sorry

theorem twenty_nine_is_greatest :
  ∀ x : ℤ, x > 29 → ¬(is_integer ((x^2 + 3*x + 8) / (x - 3))) :=
by sorry

end greatest_integer_for_fraction_twenty_nine_satisfies_twenty_nine_is_greatest_l3406_340630


namespace intersection_A_B_l3406_340679

def A : Set ℝ := {x | -1 < x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 1}

theorem intersection_A_B : A ∩ B = {0, 1} := by
  sorry

end intersection_A_B_l3406_340679


namespace work_completion_time_l3406_340661

/-- Represents the time it takes for worker B to complete the work alone -/
def time_B_alone : ℝ := 10

/-- Represents the time it takes for worker A to complete the work alone -/
def time_A_alone : ℝ := 4

/-- Represents the time A and B work together -/
def time_together : ℝ := 2

/-- Represents the time B works alone after A leaves -/
def time_B_after_A : ℝ := 3.0000000000000004

/-- Theorem stating that given the conditions, B can finish the work alone in 10 days -/
theorem work_completion_time :
  time_B_alone = 10 :=
sorry

end work_completion_time_l3406_340661


namespace equation_solution_l3406_340680

theorem equation_solution (M : ℚ) : 
  (5 + 6 + 7) / 3 = (2005 + 2006 + 2007) / M → M = 1003 := by
  sorry

end equation_solution_l3406_340680


namespace expression_equality_l3406_340655

theorem expression_equality : (π - 1) ^ 0 + 4 * Real.sin (π / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end expression_equality_l3406_340655


namespace equidistant_point_x_coordinate_l3406_340638

theorem equidistant_point_x_coordinate :
  ∃ (x y : ℝ),
    (abs x = abs y) ∧                             -- Equally distant from x-axis and y-axis
    (abs x = abs (x + y - 3) / Real.sqrt 2) ∧     -- Equally distant from the line x + y = 3
    (x = 3/2) := by
  sorry

end equidistant_point_x_coordinate_l3406_340638


namespace total_age_proof_l3406_340624

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 8 years old
  Prove that the total of their ages is 22 years. -/
theorem total_age_proof (a b c : ℕ) : 
  b = 8 → a = b + 2 → b = 2 * c → a + b + c = 22 := by
  sorry

end total_age_proof_l3406_340624


namespace monomial_exponent_equality_l3406_340660

theorem monomial_exponent_equality (a b : ℤ) : 
  (1 : ℤ) = a - 2 → b + 1 = 3 → (a - b)^(2023 : ℕ) = 1 := by
  sorry

end monomial_exponent_equality_l3406_340660


namespace first_candidate_percentage_l3406_340640

theorem first_candidate_percentage (P : ℝ) (total_marks : ℝ) : 
  P = 199.99999999999997 →
  0.45 * total_marks = P + 25 →
  (P - 50) / total_marks * 100 = 30 := by
  sorry

end first_candidate_percentage_l3406_340640


namespace binomial_13_11_l3406_340602

theorem binomial_13_11 : Nat.choose 13 11 = 78 := by
  sorry

end binomial_13_11_l3406_340602
