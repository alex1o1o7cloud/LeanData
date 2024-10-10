import Mathlib

namespace michelle_racks_l3070_307012

/-- The number of drying racks Michelle owns -/
def current_racks : ℕ := 3

/-- The number of pounds of pasta per drying rack -/
def pasta_per_rack : ℕ := 3

/-- The number of cups of flour needed to make one pound of pasta -/
def flour_per_pound : ℕ := 2

/-- The number of cups in each bag of flour -/
def cups_per_bag : ℕ := 8

/-- The number of bags of flour Michelle has -/
def num_bags : ℕ := 3

/-- The total number of cups of flour Michelle has -/
def total_flour : ℕ := num_bags * cups_per_bag

/-- The total pounds of pasta Michelle can make -/
def total_pasta : ℕ := total_flour / flour_per_pound

/-- The number of racks needed for all the pasta Michelle can make -/
def racks_needed : ℕ := total_pasta / pasta_per_rack

theorem michelle_racks :
  current_racks = racks_needed - 1 :=
sorry

end michelle_racks_l3070_307012


namespace total_fish_is_996_l3070_307037

/-- The number of fish each friend has -/
structure FishCount where
  max : ℕ
  sam : ℕ
  joe : ℕ
  harry : ℕ

/-- The conditions of the fish distribution among friends -/
def fish_distribution (fc : FishCount) : Prop :=
  fc.max = 6 ∧
  fc.sam = 3 * fc.max ∧
  fc.joe = 9 * fc.sam ∧
  fc.harry = 5 * fc.joe

/-- The total number of fish for all friends -/
def total_fish (fc : FishCount) : ℕ :=
  fc.max + fc.sam + fc.joe + fc.harry

/-- Theorem stating that the total number of fish is 996 -/
theorem total_fish_is_996 (fc : FishCount) (h : fish_distribution fc) : total_fish fc = 996 := by
  sorry

end total_fish_is_996_l3070_307037


namespace expression_is_integer_l3070_307094

theorem expression_is_integer (m : ℕ+) : ∃ k : ℤ, (m^4 / 24 : ℚ) + (m^3 / 4 : ℚ) + (11 * m^2 / 24 : ℚ) + (m / 4 : ℚ) = k := by
  sorry

end expression_is_integer_l3070_307094


namespace ellipse_theorem_l3070_307059

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a line y = kx + m -/
structure Line where
  k : ℝ
  m : ℝ

/-- The main theorem about the ellipse and the range of |OP| -/
theorem ellipse_theorem (C : Ellipse) (l : Line) : 
  C.a^2 = 4 ∧ C.b^2 = 2 ∧ abs l.k ≤ Real.sqrt 2 / 2 →
  (∀ x y : ℝ, x^2 / C.a^2 + y^2 / C.b^2 = 1 ↔ x^2 / 4 + y^2 / 2 = 1) ∧
  (∀ P : ℝ × ℝ, P.1^2 / 4 + P.2^2 / 2 = 1 → 
    Real.sqrt 2 ≤ Real.sqrt (P.1^2 + P.2^2) ∧ 
    Real.sqrt (P.1^2 + P.2^2) ≤ Real.sqrt 3) := by
  sorry

end ellipse_theorem_l3070_307059


namespace john_hourly_rate_is_20_l3070_307056

/-- Represents John's car repair scenario -/
structure CarRepairScenario where
  total_cars : ℕ
  standard_repair_time : ℕ  -- in minutes
  longer_repair_factor : ℚ  -- factor for longer repair time
  standard_repair_count : ℕ
  total_earnings : ℕ        -- in dollars

/-- Calculates John's hourly rate given the car repair scenario -/
def calculate_hourly_rate (scenario : CarRepairScenario) : ℚ :=
  -- Function body to be implemented
  sorry

/-- Theorem stating that John's hourly rate is $20 -/
theorem john_hourly_rate_is_20 (scenario : CarRepairScenario) 
  (h1 : scenario.total_cars = 5)
  (h2 : scenario.standard_repair_time = 40)
  (h3 : scenario.longer_repair_factor = 3/2)
  (h4 : scenario.standard_repair_count = 3)
  (h5 : scenario.total_earnings = 80) :
  calculate_hourly_rate scenario = 20 := by
  sorry

end john_hourly_rate_is_20_l3070_307056


namespace bus_capacity_problem_l3070_307085

theorem bus_capacity_problem (capacity : ℕ) (first_trip_fraction : ℚ) (total_people : ℕ) 
  (h1 : capacity = 200)
  (h2 : first_trip_fraction = 3 / 4)
  (h3 : total_people = 310) :
  ∃ (return_trip_fraction : ℚ), 
    (first_trip_fraction * capacity + return_trip_fraction * capacity = total_people) ∧
    return_trip_fraction = 4 / 5 :=
by sorry

end bus_capacity_problem_l3070_307085


namespace inverse_proportion_problem_l3070_307096

/-- Two real numbers are inversely proportional if their product is constant. -/
def InverselyProportional (x y : ℝ → ℝ) :=
  ∃ k : ℝ, ∀ t : ℝ, x t * y t = k

theorem inverse_proportion_problem (x y : ℝ → ℝ) 
  (h1 : InverselyProportional x y)
  (h2 : x 1 = 36 ∧ y 1 = 4)
  (h3 : y 2 = 12) :
  x 2 = 12 := by
  sorry

end inverse_proportion_problem_l3070_307096


namespace supermarket_profit_analysis_l3070_307011

/-- Represents the daily sales volume as a function of selling price -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 1600

/-- Represents the daily profit as a function of selling price -/
def daily_profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The cost price per box -/
def cost_price : ℝ := 40

/-- The minimum selling price per box -/
def min_selling_price : ℝ := 45

/-- The sales volume at the minimum selling price -/
def base_sales : ℝ := 700

/-- The decrease in sales for each yuan increase in price -/
def sales_decrease : ℝ := 20

theorem supermarket_profit_analysis :
  (∀ x ≥ min_selling_price, sales_volume x = -20 * x + 1600) ∧
  (∃ x ≥ min_selling_price, daily_profit x = 6000 ∧ x = 50) ∧
  (∃ x ≥ min_selling_price, ∀ y ≥ min_selling_price, daily_profit x ≥ daily_profit y ∧ x = 60 ∧ daily_profit x = 8000) := by
  sorry


end supermarket_profit_analysis_l3070_307011


namespace consecutive_square_roots_l3070_307010

theorem consecutive_square_roots (x : ℝ) (n : ℕ) :
  (∃ (m : ℕ), n = m^2 ∧ x = Real.sqrt n) →
  Real.sqrt (n + 1) = Real.sqrt (x^2 + 1) := by
sorry

end consecutive_square_roots_l3070_307010


namespace log_equation_solution_l3070_307090

theorem log_equation_solution (y : ℝ) (h : y > 0) : 
  Real.log y^3 / Real.log 3 + Real.log y / Real.log (1/3) = 6 → y = 27 := by
sorry

end log_equation_solution_l3070_307090


namespace area_between_concentric_circles_l3070_307081

/-- Given two concentric circles where a chord is tangent to the smaller circle,
    this theorem calculates the area of the region between the two circles. -/
theorem area_between_concentric_circles
  (outer_radius inner_radius chord_length : ℝ)
  (h_positive : 0 < inner_radius ∧ inner_radius < outer_radius)
  (h_tangent : inner_radius ^ 2 + (chord_length / 2) ^ 2 = outer_radius ^ 2)
  (h_chord : chord_length = 100) :
  (π * (outer_radius ^ 2 - inner_radius ^ 2) : ℝ) = 2000 * π :=
sorry

end area_between_concentric_circles_l3070_307081


namespace adrian_holidays_in_year_l3070_307058

/-- The number of holidays Adrian took in a year -/
def adriansHolidays (daysOffPerMonth : ℕ) (monthsInYear : ℕ) : ℕ :=
  daysOffPerMonth * monthsInYear

/-- Theorem: Adrian took 48 holidays in the entire year -/
theorem adrian_holidays_in_year : 
  adriansHolidays 4 12 = 48 := by
  sorry

end adrian_holidays_in_year_l3070_307058


namespace switches_in_position_A_after_process_l3070_307013

/-- Represents a switch position -/
inductive Position
| A | B | C | D | E

/-- Advances a position cyclically -/
def advance_position (p : Position) : Position :=
  match p with
  | Position.A => Position.B
  | Position.B => Position.C
  | Position.C => Position.D
  | Position.D => Position.E
  | Position.E => Position.A

/-- Represents a switch with its label and position -/
structure Switch :=
  (x y z w : Nat)
  (pos : Position)

/-- The total number of switches -/
def total_switches : Nat := 6860

/-- Creates the initial set of switches -/
def initial_switches : Finset Switch :=
  sorry

/-- Advances a switch and its divisors -/
def advance_switches (switches : Finset Switch) (i : Nat) : Finset Switch :=
  sorry

/-- Performs the entire 6860-step process -/
def process (switches : Finset Switch) : Finset Switch :=
  sorry

/-- Counts switches in position A -/
def count_position_A (switches : Finset Switch) : Nat :=
  sorry

/-- The main theorem to prove -/
theorem switches_in_position_A_after_process :
  count_position_A (process initial_switches) = 6455 :=
sorry

end switches_in_position_A_after_process_l3070_307013


namespace smallest_constant_inequality_l3070_307084

theorem smallest_constant_inequality (C : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ↔ C ≤ Real.sqrt (4/3) :=
sorry

end smallest_constant_inequality_l3070_307084


namespace binomial_distribution_not_equivalent_to_expansion_l3070_307023

-- Define the binomial distribution formula
def binomial_distribution (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1-p)^(n-k)

-- Define the general term of binomial expansion
def binomial_expansion_term (n k : ℕ) (a b : ℝ) : ℝ :=
  (n.choose k) * a^k * b^(n-k)

-- Theorem statement
theorem binomial_distribution_not_equivalent_to_expansion :
  ∃ n k : ℕ, ∃ p : ℝ, 
    binomial_distribution n k p ≠ binomial_expansion_term n k p (1-p) :=
sorry

end binomial_distribution_not_equivalent_to_expansion_l3070_307023


namespace blanket_warmth_l3070_307007

theorem blanket_warmth (total_blankets : ℕ) (used_fraction : ℚ) (total_warmth : ℕ) : 
  total_blankets = 14 →
  used_fraction = 1/2 →
  total_warmth = 21 →
  (total_warmth : ℚ) / (used_fraction * total_blankets) = 3 := by
  sorry

end blanket_warmth_l3070_307007


namespace third_grade_sample_size_l3070_307078

theorem third_grade_sample_size
  (total_students : ℕ)
  (first_grade_students : ℕ)
  (sample_size : ℕ)
  (second_grade_sample_ratio : ℚ)
  (h1 : total_students = 2800)
  (h2 : first_grade_students = 910)
  (h3 : sample_size = 40)
  (h4 : second_grade_sample_ratio = 3 / 10)
  : ℕ := by
  sorry

end third_grade_sample_size_l3070_307078


namespace journey_problem_solution_exists_l3070_307031

/-- Proves the existence of a solution for the journey problem -/
theorem journey_problem_solution_exists :
  ∃ (x y T : ℝ),
    x > 0 ∧ y > 0 ∧ T > 0 ∧
    x < 150 ∧ y < x ∧
    (x / 30 + (150 - x) / 3 = T) ∧
    (x / 30 + y / 30 + (150 - (x - y)) / 30 = T) ∧
    ((x - y) / 10 + (150 - (x - y)) / 30 = T) :=
by sorry

#check journey_problem_solution_exists

end journey_problem_solution_exists_l3070_307031


namespace taxi_ride_distance_is_8_miles_l3070_307054

/-- Calculates the distance of a taxi ride given the fare structure and total charge -/
def taxi_ride_distance (initial_charge : ℚ) (additional_charge : ℚ) (total_charge : ℚ) : ℚ :=
  let remaining_charge := total_charge - initial_charge
  let additional_increments := remaining_charge / additional_charge
  (additional_increments + 1) * (1 / 5)

/-- Proves that the taxi ride distance is 8 miles given the specified fare structure and total charge -/
theorem taxi_ride_distance_is_8_miles :
  let initial_charge : ℚ := 21/10  -- $2.10
  let additional_charge : ℚ := 4/10  -- $0.40
  let total_charge : ℚ := 177/10  -- $17.70
  taxi_ride_distance initial_charge additional_charge total_charge = 8 := by
  sorry

#eval taxi_ride_distance (21/10) (4/10) (177/10)

end taxi_ride_distance_is_8_miles_l3070_307054


namespace circle_diameter_l3070_307063

theorem circle_diameter (AE EB ED : ℝ) (h1 : AE = 2) (h2 : EB = 6) (h3 : ED = 3) :
  let AB := AE + EB
  let CE := (AE * EB) / ED
  let AM := (AB) / 2
  let OM := (AE + EB) / 2
  let OA := Real.sqrt (AM^2 + OM^2)
  let diameter := 2 * OA
  diameter = Real.sqrt 65 := by sorry

end circle_diameter_l3070_307063


namespace endpoint_coordinate_sum_l3070_307079

/-- Given a line segment with one endpoint at (3, 4) and midpoint at (5, -8),
    the sum of the coordinates of the other endpoint is -13. -/
theorem endpoint_coordinate_sum :
  let a : ℝ × ℝ := (3, 4)  -- First endpoint
  let m : ℝ × ℝ := (5, -8) -- Midpoint
  let b : ℝ × ℝ := (2 * m.1 - a.1, 2 * m.2 - a.2) -- Other endpoint
  b.1 + b.2 = -13 := by sorry

end endpoint_coordinate_sum_l3070_307079


namespace present_age_of_b_l3070_307009

theorem present_age_of_b (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10)) 
  (h2 : a = b + 9) : 
  b = 39 := by
  sorry

end present_age_of_b_l3070_307009


namespace probability_ratio_l3070_307025

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def num_numbers : ℕ := 10

/-- The number of slips for each number from 1 to 5 -/
def slips_per_low_number : ℕ := 5

/-- The number of slips for each number from 6 to 10 -/
def slips_per_high_number : ℕ := 3

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability that all four drawn slips bear the same number (only possible for numbers 1 to 5) -/
def r : ℚ := (slips_per_low_number.choose drawn_slips * 5 : ℚ) / total_slips.choose drawn_slips

/-- The probability that two slips bear a number c (1 to 5) and two slips bear a number d ≠ c (6 to 10) -/
def s : ℚ := (5 * 5 * slips_per_low_number.choose 2 * slips_per_high_number.choose 2 : ℚ) / total_slips.choose drawn_slips

theorem probability_ratio : s / r = 30 := by
  sorry

end probability_ratio_l3070_307025


namespace floor_sum_equals_negative_one_l3070_307024

theorem floor_sum_equals_negative_one : ⌊(19.7 : ℝ)⌋ + ⌊(-19.7 : ℝ)⌋ = -1 := by
  sorry

end floor_sum_equals_negative_one_l3070_307024


namespace greenSpaceAfterThreeYears_l3070_307044

/-- Calculates the green space after a given number of years with a fixed annual increase rate -/
def greenSpaceAfterYears (initialSpace : ℝ) (annualIncrease : ℝ) (years : ℕ) : ℝ :=
  initialSpace * (1 + annualIncrease) ^ years

/-- Theorem stating that the green space after 3 years with initial 1000 acres and 10% annual increase is 1331 acres -/
theorem greenSpaceAfterThreeYears :
  greenSpaceAfterYears 1000 0.1 3 = 1331 := by sorry

end greenSpaceAfterThreeYears_l3070_307044


namespace house_cost_is_480000_l3070_307065

/-- Calculates the cost of a house given the following conditions:
  - A trailer costs $120,000
  - Each loan will be paid in monthly installments over 20 years
  - The monthly payment on the house is $1500 more than the trailer
-/
def house_cost (trailer_cost : ℕ) (loan_years : ℕ) (monthly_difference : ℕ) : ℕ :=
  let months : ℕ := loan_years * 12
  let trailer_monthly : ℕ := trailer_cost / months
  let house_monthly : ℕ := trailer_monthly + monthly_difference
  house_monthly * months

/-- Theorem stating that the cost of the house is $480,000 -/
theorem house_cost_is_480000 :
  house_cost 120000 20 1500 = 480000 := by
  sorry

end house_cost_is_480000_l3070_307065


namespace fraction_equality_l3070_307091

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : a/b + (a+6*b)/(b+6*a) = 3) :
  a/b = (8 + Real.sqrt 46)/6 ∨ a/b = (8 - Real.sqrt 46)/6 := by
  sorry

end fraction_equality_l3070_307091


namespace f_analytical_expression_l3070_307095

def f : Set ℝ := {x : ℝ | x ≠ -1}

theorem f_analytical_expression :
  ∀ x : ℝ, x ∈ f ↔ x ≠ -1 :=
by sorry

end f_analytical_expression_l3070_307095


namespace gain_percent_calculation_l3070_307016

theorem gain_percent_calculation (marked_price : ℝ) (marked_price_positive : marked_price > 0) :
  let cost_price := 0.25 * marked_price
  let discount := 0.5 * marked_price
  let selling_price := marked_price - discount
  let gain := selling_price - cost_price
  let gain_percent := (gain / cost_price) * 100
  gain_percent = 100 := by
sorry

end gain_percent_calculation_l3070_307016


namespace parabola_intersection_distance_l3070_307027

/-- Parabola type representing y^2 = 4x -/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type -/
structure Line where
  passes_through : ℝ × ℝ → Prop

/-- Intersection point of a line and a parabola -/
structure IntersectionPoint where
  point : ℝ × ℝ

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

theorem parabola_intersection_distance 
  (p : Parabola)
  (l : Line)
  (A B : IntersectionPoint)
  (h1 : p.equation = fun x y => y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : l.passes_through p.focus)
  (h4 : l.passes_through A.point ∧ l.passes_through B.point)
  (h5 : distance A.point p.focus = 4)
  : distance B.point p.focus = 4/3 := by sorry

end parabola_intersection_distance_l3070_307027


namespace algebraic_expression_simplification_l3070_307071

theorem algebraic_expression_simplification (x : ℝ) (h : x = -4) :
  (x^2 - 2*x) / (x - 3) / ((1 / (x + 3) + 1 / (x - 3))) = 3 := by
  sorry

end algebraic_expression_simplification_l3070_307071


namespace function_composition_ratio_l3070_307043

def f (x : ℝ) : ℝ := 3 * x + 4

def g (x : ℝ) : ℝ := 2 * x - 1

theorem function_composition_ratio :
  f (g (f 3)) / g (f (g 3)) = 79 / 37 := by
  sorry

end function_composition_ratio_l3070_307043


namespace parallel_vectors_l3070_307032

def a (m : ℝ) : Fin 2 → ℝ := ![2, m]
def b : Fin 2 → ℝ := ![1, -2]

def parallel (u v : Fin 2 → ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ (∀ i, u i = k * v i)

theorem parallel_vectors (m : ℝ) :
  parallel (a m) (λ i => a m i + 2 * b i) → m = -4 := by
  sorry

end parallel_vectors_l3070_307032


namespace traci_flour_l3070_307014

/-- The amount of flour Harris has in his house -/
def harris_flour : ℕ := 400

/-- The amount of flour needed for each cake -/
def flour_per_cake : ℕ := 100

/-- The number of cakes Traci and Harris created each -/
def cakes_per_person : ℕ := 9

/-- The total number of cakes created -/
def total_cakes : ℕ := 2 * cakes_per_person

/-- The theorem stating the amount of flour Traci brought from her own house -/
theorem traci_flour : 
  harris_flour + (total_cakes * flour_per_cake - harris_flour) = 1400 := by
sorry

end traci_flour_l3070_307014


namespace sphere_radius_in_truncated_cone_l3070_307046

/-- The radius of a sphere tangent to a truncated cone -/
theorem sphere_radius_in_truncated_cone (r₁ r₂ : ℝ) (hr₁ : r₁ = 25) (hr₂ : r₂ = 5) :
  let h := Real.sqrt ((r₁ - r₂)^2 + (r₁ + r₂)^2)
  (h / 2 : ℝ) = 5 * Real.sqrt 2 :=
by sorry

end sphere_radius_in_truncated_cone_l3070_307046


namespace marias_stamp_collection_l3070_307088

/-- The problem of calculating Maria's stamp collection increase -/
theorem marias_stamp_collection 
  (current_stamps : ℕ) 
  (increase_percentage : ℚ) 
  (h1 : current_stamps = 40)
  (h2 : increase_percentage = 20 / 100) : 
  current_stamps + (increase_percentage * current_stamps).floor = 48 := by
  sorry

end marias_stamp_collection_l3070_307088


namespace jean_calories_eaten_l3070_307062

def pages_written : ℕ := 12
def pages_per_donut : ℕ := 2
def calories_per_donut : ℕ := 150

theorem jean_calories_eaten : 
  (pages_written / pages_per_donut) * calories_per_donut = 900 := by
  sorry

end jean_calories_eaten_l3070_307062


namespace total_points_after_perfect_games_l3070_307075

/-- A perfect score in a game -/
def perfect_score : ℕ := 21

/-- The number of perfect games played -/
def games_played : ℕ := 3

/-- Theorem: The total points after 3 perfect games is 63 -/
theorem total_points_after_perfect_games : 
  perfect_score * games_played = 63 := by
  sorry

end total_points_after_perfect_games_l3070_307075


namespace max_m_value_inequality_proof_l3070_307033

-- Define the function f
def f (x : ℝ) : ℝ := |x| + |x - 1|

-- Theorem for part I
theorem max_m_value (m : ℝ) : 
  (∀ x, f x ≥ |m - 1|) → m ≤ 2 :=
sorry

-- Theorem for part II
theorem inequality_proof (a b : ℝ) :
  a > 0 → b > 0 → a^2 + b^2 = 2 → a + b ≥ 2 * a * b :=
sorry

end max_m_value_inequality_proof_l3070_307033


namespace f_expression_l3070_307052

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_expression : 
  (∀ x : ℝ, f (x + 1) = 3 * x + 2) → 
  (∀ x : ℝ, f x = 3 * x - 1) :=
by
  sorry

end f_expression_l3070_307052


namespace prime_pairs_divisibility_l3070_307083

theorem prime_pairs_divisibility (p q : Nat) : 
  Nat.Prime p ∧ Nat.Prime q ∧ 
  (p ∣ 5^q + 1) ∧ (q ∣ 5^p + 1) ↔ 
  ((p = 2 ∧ q = 13) ∨ (p = 13 ∧ q = 2) ∨ (p = 3 ∧ q = 7) ∨ (p = 7 ∧ q = 3)) :=
by sorry

end prime_pairs_divisibility_l3070_307083


namespace max_value_AMC_l3070_307017

theorem max_value_AMC (A M C : ℕ) (sum_constraint : A + M + C = 10) :
  (∀ A' M' C' : ℕ, A' + M' + C' = 10 → 
    A' * M' * C' + A' * M' + M' * C' + C' * A' ≤ A * M * C + A * M + M * C + C * A) →
  A * M * C + A * M + M * C + C * A = 69 := by
  sorry

end max_value_AMC_l3070_307017


namespace polo_shirt_price_l3070_307092

/-- The regular price of a polo shirt -/
def regular_price : ℝ := 50

/-- The number of polo shirts purchased -/
def num_shirts : ℕ := 2

/-- The discount percentage on the shirts -/
def discount_percent : ℝ := 40

/-- The total amount paid for the shirts -/
def total_paid : ℝ := 60

/-- Theorem stating that the regular price of each polo shirt is $50 -/
theorem polo_shirt_price :
  regular_price = 50 ∧
  num_shirts * regular_price * (1 - discount_percent / 100) = total_paid :=
by sorry

end polo_shirt_price_l3070_307092


namespace quadratic_polynomials_inequalities_l3070_307004

/-- Given three quadratic polynomials with the specified properties, 
    exactly two out of three inequalities are satisfied. -/
theorem quadratic_polynomials_inequalities 
  (a b c d e f : ℝ) 
  (h1 : ∃ x : ℝ, (x^2 + a*x + b = 0 ∧ x^2 + c*x + d = 0) ∨ 
                 (x^2 + a*x + b = 0 ∧ x^2 + e*x + f = 0) ∨ 
                 (x^2 + c*x + d = 0 ∧ x^2 + e*x + f = 0))
  (h2 : ¬ ∃ x : ℝ, x^2 + a*x + b = 0 ∧ x^2 + c*x + d = 0 ∧ x^2 + e*x + f = 0) :
  (((a^2 + c^2 - e^2)/4 > b + d - f) ∧ 
   ((c^2 + e^2 - a^2)/4 > d + f - b) ∧ 
   ((e^2 + a^2 - c^2)/4 ≤ f + b - d)) ∨
  (((a^2 + c^2 - e^2)/4 > b + d - f) ∧ 
   ((c^2 + e^2 - a^2)/4 ≤ d + f - b) ∧ 
   ((e^2 + a^2 - c^2)/4 > f + b - d)) ∨
  (((a^2 + c^2 - e^2)/4 ≤ b + d - f) ∧ 
   ((c^2 + e^2 - a^2)/4 > d + f - b) ∧ 
   ((e^2 + a^2 - c^2)/4 > f + b - d)) := by
  sorry

end quadratic_polynomials_inequalities_l3070_307004


namespace A_eq_zero_two_l3070_307072

/-- The set of real numbers a for which the equation ax^2 - 4x + 2 = 0 has exactly one solution -/
def A : Set ℝ :=
  {a : ℝ | ∃! x : ℝ, a * x^2 - 4 * x + 2 = 0}

/-- Theorem stating that A is equal to the set {0, 2} -/
theorem A_eq_zero_two : A = {0, 2} := by
  sorry

end A_eq_zero_two_l3070_307072


namespace fraction_comparison_l3070_307073

theorem fraction_comparison : 
  (10^1966 + 1) / (10^1967 + 1) > (10^1967 + 1) / (10^1968 + 1) := by
  sorry

end fraction_comparison_l3070_307073


namespace sticker_ratio_l3070_307008

/-- Proves that the ratio of Dan's stickers to Tom's stickers is 2:1 -/
theorem sticker_ratio :
  let bob_stickers : ℕ := 12
  let tom_stickers : ℕ := 3 * bob_stickers
  let dan_stickers : ℕ := 72
  (dan_stickers : ℚ) / tom_stickers = 2 / 1 := by
  sorry

end sticker_ratio_l3070_307008


namespace max_distance_is_217_12_l3070_307086

-- Define the constants
def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def total_gallons : ℝ := 23

-- Define the percentages for regular and peak traffic
def regular_highway_percent : ℝ := 0.4
def regular_city_percent : ℝ := 0.6
def peak_highway_percent : ℝ := 0.25
def peak_city_percent : ℝ := 0.75

-- Calculate distances for regular and peak traffic
def regular_distance : ℝ := 
  (regular_highway_percent * total_gallons * highway_mpg) + 
  (regular_city_percent * total_gallons * city_mpg)

def peak_distance : ℝ := 
  (peak_highway_percent * total_gallons * highway_mpg) + 
  (peak_city_percent * total_gallons * city_mpg)

-- Theorem to prove
theorem max_distance_is_217_12 : 
  max regular_distance peak_distance = 217.12 := by sorry

end max_distance_is_217_12_l3070_307086


namespace part_one_part_two_l3070_307064

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

-- Part 1
theorem part_one :
  (Set.compl (B (1/2))) ∩ (A (1/2)) = {x : ℝ | 9/4 ≤ x ∧ x < 5/2} := by sorry

-- Part 2
theorem part_two :
  ∀ a : ℝ, (A a ⊆ B a) ↔ (a ≥ -1/2 ∧ a ≤ (3 - Real.sqrt 5) / 2) := by sorry

end part_one_part_two_l3070_307064


namespace min_seating_arrangement_l3070_307018

/-- Given a circular table with 60 chairs, this theorem proves that the smallest number of people
    that can be seated such that any additional person must sit next to someone is 20. -/
theorem min_seating_arrangement (n : ℕ) : n = 20 ↔ (
  n ≤ 60 ∧ 
  (∀ m : ℕ, m < n → ∃ (arrangement : Fin 60 → Bool), 
    (∃ i : Fin 60, ¬arrangement i) ∧ 
    (∀ i : Fin 60, arrangement i → 
      (arrangement (i + 1) ∨ arrangement (i + 59)))) ∧
  (∀ m : ℕ, m > n → ¬∃ (arrangement : Fin 60 → Bool),
    (∀ i : Fin 60, arrangement i → 
      (arrangement (i + 1) ∨ arrangement (i + 59)))))
  := by sorry


end min_seating_arrangement_l3070_307018


namespace chloe_wins_l3070_307045

/-- Given that the ratio of Chloe's wins to Max's wins is 8:3 and Max won 9 times,
    prove that Chloe won 24 times. -/
theorem chloe_wins (ratio_chloe : ℕ) (ratio_max : ℕ) (max_wins : ℕ) 
    (h1 : ratio_chloe = 8)
    (h2 : ratio_max = 3)
    (h3 : max_wins = 9) : 
  (ratio_chloe * max_wins) / ratio_max = 24 := by
  sorry

#check chloe_wins

end chloe_wins_l3070_307045


namespace sqrt_30_simplest_l3070_307097

/-- Predicate to check if a number is a perfect square --/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Predicate to check if a square root is in its simplest form --/
def IsSimplestSquareRoot (n : ℝ) : Prop :=
  ∃ m : ℕ, n = Real.sqrt m ∧ m > 0 ∧ ¬∃ k : ℕ, k > 1 ∧ IsPerfectSquare k ∧ k ∣ m

/-- Theorem stating that √30 is the simplest square root among the given options --/
theorem sqrt_30_simplest :
  IsSimplestSquareRoot (Real.sqrt 30) ∧
  ¬IsSimplestSquareRoot (Real.sqrt 0.1) ∧
  ¬IsSimplestSquareRoot (1/2 : ℝ) ∧
  ¬IsSimplestSquareRoot (Real.sqrt 18) :=
by sorry


end sqrt_30_simplest_l3070_307097


namespace sqrt_9801_minus_99_proof_l3070_307006

theorem sqrt_9801_minus_99_proof (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : Real.sqrt 9801 - 99 = (Real.sqrt a - b)^3) : a + b = 11 := by
  sorry

end sqrt_9801_minus_99_proof_l3070_307006


namespace hundredth_term_is_401_l3070_307035

/-- 
Represents a sequence of toothpicks where:
- The first term is 5
- Each subsequent term increases by 4
-/
def toothpick_sequence (n : ℕ) : ℕ := 5 + 4 * (n - 1)

/-- 
Theorem: The 100th term of the toothpick sequence is 401
-/
theorem hundredth_term_is_401 : toothpick_sequence 100 = 401 := by
  sorry

end hundredth_term_is_401_l3070_307035


namespace two_digit_sum_property_l3070_307050

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  ∃ (x y : ℕ),
    n = 10 * x + y ∧
    x < 10 ∧ y < 10 ∧
    (x + 1 + y + 2 - 10) / 2 = x + y ∧
    y + 2 ≥ 10

theorem two_digit_sum_property :
  ∀ n : ℕ, is_valid_number n ↔ (n = 68 ∨ n = 59) :=
sorry

end two_digit_sum_property_l3070_307050


namespace pencil_total_length_l3070_307042

-- Define the pencil sections
def purple_length : ℝ := 3
def black_length : ℝ := 2
def blue_length : ℝ := 1

-- Theorem statement
theorem pencil_total_length : 
  purple_length + black_length + blue_length = 6 := by sorry

end pencil_total_length_l3070_307042


namespace expression_is_equation_l3070_307034

-- Define what an equation is
def is_equation (e : Prop) : Prop :=
  ∃ (lhs rhs : ℝ → ℝ → ℝ), e = (∀ r y, lhs r y = rhs r y)

-- Define the expression we want to prove is an equation
def expression (r y : ℝ) : ℝ := 3 * r + y

-- Theorem statement
theorem expression_is_equation :
  is_equation (∀ r y : ℝ, expression r y = 5) :=
sorry

end expression_is_equation_l3070_307034


namespace triangle_problem_l3070_307026

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a > t.c)
  (h2 : t.b * t.a * Real.cos t.B = 2)
  (h3 : Real.cos t.B = 1/3)
  (h4 : t.b = 3) :
  (t.a = 3 ∧ t.c = 2) ∧ 
  Real.cos (t.B - t.C) = 23/27 := by
  sorry

end triangle_problem_l3070_307026


namespace smallest_non_prime_non_square_no_small_factors_l3070_307028

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ p : ℕ, is_prime p → p < k → ¬(n % p = 0)

theorem smallest_non_prime_non_square_no_small_factors : 
  (∀ m : ℕ, m < 4091 → 
    is_prime m ∨ 
    is_perfect_square m ∨ 
    ¬(has_no_prime_factor_less_than m 60)) ∧ 
  ¬(is_prime 4091) ∧ 
  ¬(is_perfect_square 4091) ∧ 
  has_no_prime_factor_less_than 4091 60 :=
sorry

end smallest_non_prime_non_square_no_small_factors_l3070_307028


namespace unique_solution_l3070_307051

theorem unique_solution : ∃! x : ℕ, 
  (∃ k : ℕ, x = 7 * k) ∧ 
  x^3 < 8000 ∧ 
  10 < x ∧ x < 30 :=
by
  sorry

end unique_solution_l3070_307051


namespace alex_class_size_l3070_307015

/-- Represents a student's ranking in a class -/
structure StudentRanking where
  best : Nat
  worst : Nat

/-- Calculates the total number of students in a class given a student's ranking -/
def totalStudents (ranking : StudentRanking) : Nat :=
  ranking.best + ranking.worst - 1

/-- Theorem: If a student is ranked 20th best and 20th worst, there are 39 students in the class -/
theorem alex_class_size (ranking : StudentRanking) 
  (h1 : ranking.best = 20) 
  (h2 : ranking.worst = 20) : 
  totalStudents ranking = 39 := by
  sorry

end alex_class_size_l3070_307015


namespace bank_account_increase_percentage_l3070_307029

def al_initial_balance : ℝ := 236.36
def eliot_initial_balance : ℝ := 200

theorem bank_account_increase_percentage :
  (al_initial_balance > eliot_initial_balance) →
  (al_initial_balance - eliot_initial_balance = (al_initial_balance + eliot_initial_balance) / 12) →
  (∃ p : ℝ, (al_initial_balance * 1.1 = eliot_initial_balance * (1 + p / 100) + 20) ∧ p = 20) :=
by sorry

end bank_account_increase_percentage_l3070_307029


namespace polynomial_expansion_l3070_307089

theorem polynomial_expansion (x : ℝ) : 
  (1 + x^4) * (1 - x^5) * (1 + x^7) = 
  1 + x^4 - x^5 + x^7 + x^11 - x^9 - x^12 - x^16 := by
sorry

end polynomial_expansion_l3070_307089


namespace carlas_order_cost_l3070_307019

/-- Calculates the final cost of Carla's order at McDonald's --/
theorem carlas_order_cost (base_cost : ℝ) (coupon_discount : ℝ) (senior_discount_rate : ℝ) (swap_charge : ℝ)
  (h1 : base_cost = 7.5)
  (h2 : coupon_discount = 2.5)
  (h3 : senior_discount_rate = 0.2)
  (h4 : swap_charge = 1.0) :
  base_cost - coupon_discount - (base_cost - coupon_discount) * senior_discount_rate + swap_charge = 5 :=
by sorry


end carlas_order_cost_l3070_307019


namespace equality_proof_l3070_307041

theorem equality_proof (x y : ℤ) : 
  (x - 1) * (x + 4) * (x - 3) - (x + 1) * (x - 4) * (x + 3) = 
  (y - 1) * (y + 4) * (y - 3) - (y + 1) * (y - 4) * (y + 3) := by
  sorry

end equality_proof_l3070_307041


namespace quadratic_roots_reciprocal_l3070_307082

theorem quadratic_roots_reciprocal (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + a
  ∀ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 → x₁ = 1 / x₂ ∧ x₂ = 1 / x₁ := by
  sorry

end quadratic_roots_reciprocal_l3070_307082


namespace clerical_employee_fraction_l3070_307074

/-- Proves that the fraction of clerical employees is 4/15 given the conditions -/
theorem clerical_employee_fraction :
  let total_employees : ℕ := 3600
  let clerical_fraction : ℚ := 4/15
  let reduction_factor : ℚ := 3/4
  let remaining_fraction : ℚ := 1/5
  (clerical_fraction * total_employees : ℚ) * reduction_factor =
    remaining_fraction * total_employees :=
by sorry

end clerical_employee_fraction_l3070_307074


namespace train_length_l3070_307061

/-- The length of a train given crossing times -/
theorem train_length (tree_time platform_time platform_length : ℝ) 
  (h1 : tree_time = 120)
  (h2 : platform_time = 180)
  (h3 : platform_length = 600)
  (h4 : tree_time > 0)
  (h5 : platform_time > 0)
  (h6 : platform_length > 0) :
  (platform_time * platform_length) / (platform_time - tree_time) = 1200 := by
  sorry

end train_length_l3070_307061


namespace not_power_of_integer_l3070_307070

theorem not_power_of_integer (m : ℕ) : ¬ ∃ (n k : ℕ), m * (m + 1) = n^k := by
  sorry

end not_power_of_integer_l3070_307070


namespace inequality_and_equality_condition_l3070_307069

theorem inequality_and_equality_condition (x : ℝ) (hx : x ≠ 0) : 
  max 0 (Real.log (abs x)) ≥ 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ∧
  (max 0 (Real.log (abs x)) = 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (abs x) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (abs (x^2 - 1)) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) ↔ 
  (x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ 
   x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2)) :=
by sorry

end inequality_and_equality_condition_l3070_307069


namespace total_amount_calculation_l3070_307055

theorem total_amount_calculation (x y z : ℝ) : 
  x > 0 → 
  y = 0.45 * x → 
  z = 0.30 * x → 
  y = 36 → 
  x + y + z = 140 := by
  sorry

end total_amount_calculation_l3070_307055


namespace min_squares_partition_l3070_307003

/-- Represents a square with an integer side length -/
structure Square where
  side : ℕ

/-- Represents a partition of a square into smaller squares -/
structure Partition where
  squares : List Square

/-- Check if a partition is valid for an 11x11 square -/
def isValidPartition (p : Partition) : Prop :=
  (p.squares.map (λ s => s.side * s.side)).sum = 11 * 11 ∧
  p.squares.all (λ s => s.side > 0 ∧ s.side < 11)

/-- The theorem stating the minimum number of squares in a valid partition -/
theorem min_squares_partition :
  ∃ (p : Partition), isValidPartition p ∧ p.squares.length = 11 ∧
  ∀ (q : Partition), isValidPartition q → p.squares.length ≤ q.squares.length :=
sorry

end min_squares_partition_l3070_307003


namespace hall_length_l3070_307049

/-- The length of a hall given its width, number of stones, and stone dimensions -/
theorem hall_length (hall_width : ℝ) (num_stones : ℕ) (stone_length stone_width : ℝ) :
  hall_width = 15 ∧ 
  num_stones = 1350 ∧ 
  stone_length = 0.8 ∧ 
  stone_width = 0.5 →
  (stone_length * stone_width * num_stones) / hall_width = 36 :=
by sorry

end hall_length_l3070_307049


namespace min_value_product_quotient_equality_condition_l3070_307093

theorem min_value_product_quotient (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2*a + 1) * (b^2 + 2*b + 1) * (c^2 + 2*c + 1) / (a * b * c) ≥ 64 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + 2*a + 1) * (b^2 + 2*b + 1) * (c^2 + 2*c + 1) / (a * b * c) = 64 ↔ a = 1 ∧ b = 1 ∧ c = 1 :=
by sorry

end min_value_product_quotient_equality_condition_l3070_307093


namespace newspaper_buying_percentage_l3070_307080

def newspapers_bought : ℕ := 500
def selling_price : ℚ := 2
def percentage_sold : ℚ := 80 / 100
def profit : ℚ := 550

theorem newspaper_buying_percentage : 
  ∀ (buying_price : ℚ),
    (newspapers_bought : ℚ) * percentage_sold * selling_price - 
    (newspapers_bought : ℚ) * buying_price = profit →
    (selling_price - buying_price) / selling_price = 75 / 100 := by
  sorry

end newspaper_buying_percentage_l3070_307080


namespace sum_of_coefficients_expansion_l3070_307000

theorem sum_of_coefficients_expansion (x y : ℝ) : 
  (fun x y => (x + 2*y - 1)^6) 1 1 = 64 := by sorry

end sum_of_coefficients_expansion_l3070_307000


namespace five_squared_sum_five_times_l3070_307030

theorem five_squared_sum_five_times : (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) + (5 * 5) = 125 := by
  sorry

end five_squared_sum_five_times_l3070_307030


namespace museum_visitor_ratio_l3070_307053

/-- Represents the number of adults and children visiting a museum. -/
structure MuseumVisitors where
  adults : ℕ
  children : ℕ

/-- Calculates the total admission fee for a given number of adults and children. -/
def admissionFee (visitors : MuseumVisitors) : ℕ :=
  30 * visitors.adults + 15 * visitors.children

/-- Checks if the number of visitors satisfies the minimum requirement. -/
def satisfiesMinimum (visitors : MuseumVisitors) : Prop :=
  visitors.adults ≥ 2 ∧ visitors.children ≥ 2

/-- Calculates the ratio of adults to children. -/
def visitorRatio (visitors : MuseumVisitors) : ℚ :=
  visitors.adults / visitors.children

theorem museum_visitor_ratio :
  ∃ (visitors : MuseumVisitors),
    satisfiesMinimum visitors ∧
    admissionFee visitors = 2700 ∧
    visitorRatio visitors = 2 ∧
    (∀ (other : MuseumVisitors),
      satisfiesMinimum other →
      admissionFee other = 2700 →
      |visitorRatio other - 2| ≥ |visitorRatio visitors - 2|) := by
  sorry

end museum_visitor_ratio_l3070_307053


namespace fraction_repeating_block_length_l3070_307001

/-- The length of the repeating block in the decimal expansion of 7/13 -/
def repeatingBlockLength : ℕ := 6

/-- The fraction we're considering -/
def fraction : ℚ := 7/13

theorem fraction_repeating_block_length :
  ∃ (d : ℕ+) (n : ℕ), 
    fraction * d.val = n ∧ 
    d = 10^repeatingBlockLength - 1 := by
  sorry

end fraction_repeating_block_length_l3070_307001


namespace paul_books_left_l3070_307039

/-- The number of books Paul had left after the garage sale -/
def books_left (initial_books : ℕ) (books_sold : ℕ) : ℕ :=
  initial_books - books_sold

/-- Theorem stating that Paul had 66 books left after the garage sale -/
theorem paul_books_left : books_left 108 42 = 66 := by
  sorry

end paul_books_left_l3070_307039


namespace pot_temperature_celsius_l3070_307005

/-- Converts temperature from Fahrenheit to Celsius -/
def fahrenheit_to_celsius (f : ℚ) : ℚ :=
  (f - 32) * (5/9)

/-- The temperature of the pot of water in Fahrenheit -/
def pot_temperature_f : ℚ := 122

theorem pot_temperature_celsius :
  fahrenheit_to_celsius pot_temperature_f = 50 := by
  sorry

end pot_temperature_celsius_l3070_307005


namespace digit_200_of_17_over_70_is_2_l3070_307087

/-- The 200th digit after the decimal point in the decimal representation of 17/70 -/
def digit_200_of_17_over_70 : ℕ := 2

/-- Theorem stating that the 200th digit after the decimal point in 17/70 is 2 -/
theorem digit_200_of_17_over_70_is_2 :
  digit_200_of_17_over_70 = 2 := by sorry

end digit_200_of_17_over_70_is_2_l3070_307087


namespace price_after_discounts_l3070_307038

def initial_price : Float := 9649.12
def discount1 : Float := 0.20
def discount2 : Float := 0.10
def discount3 : Float := 0.05

def apply_discount (price : Float) (discount : Float) : Float :=
  price * (1 - discount)

def final_price : Float :=
  apply_discount (apply_discount (apply_discount initial_price discount1) discount2) discount3

theorem price_after_discounts :
  final_price = 6600.09808 := by sorry

end price_after_discounts_l3070_307038


namespace petya_max_spend_l3070_307077

/-- Represents the cost of a book in rubles -/
def BookCost := ℕ

/-- Represents Petya's purchasing behavior -/
structure PetyaPurchase where
  initialMoney : ℕ  -- Initial amount of money Petya had
  expensiveBookThreshold : ℕ  -- Threshold for expensive books (100 rubles)
  spentHalf : Bool  -- Whether Petya spent exactly half of his money

/-- Theorem stating that Petya couldn't have spent 5000 rubles or more on books -/
theorem petya_max_spend (purchase : PetyaPurchase) : 
  purchase.spentHalf → purchase.expensiveBookThreshold = 100 →
  ∃ (maxSpend : ℕ), maxSpend < 5000 ∧ 
  ∀ (actualSpend : ℕ), actualSpend ≤ maxSpend :=
sorry

end petya_max_spend_l3070_307077


namespace container_volume_ratio_l3070_307002

theorem container_volume_ratio (V₁ V₂ V₃ : ℝ) 
  (h₁ : V₁ > 0) (h₂ : V₂ > 0) (h₃ : V₃ > 0)
  (h₄ : (3/4) * V₁ = (5/8) * V₂)
  (h₅ : (5/8) * V₂ = (1/2) * V₃) :
  V₁ / V₃ = 1/2 := by
sorry

end container_volume_ratio_l3070_307002


namespace elderly_arrangement_count_l3070_307022

/-- The number of ways to arrange n distinct objects. -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to choose k objects from n distinct objects, where order matters. -/
def arrangements (n k : ℕ) : ℕ := 
  if k ≤ n then
    Nat.factorial n / Nat.factorial (n - k)
  else
    0

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a line,
    where the elderly people must be adjacent and not at the ends. -/
def elderly_arrangement : ℕ := 
  arrangements 5 2 * permutations 4 * permutations 2

theorem elderly_arrangement_count : elderly_arrangement = 960 := by
  sorry

end elderly_arrangement_count_l3070_307022


namespace sin_360_degrees_l3070_307021

theorem sin_360_degrees : Real.sin (2 * Real.pi) = 0 := by
  sorry

end sin_360_degrees_l3070_307021


namespace two_digit_sum_theorem_l3070_307098

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  tens : Nat
  ones : Nat
  is_valid : tens ≤ 9 ∧ ones ≤ 9

/-- The sum of five identical two-digit numbers equals another two-digit number -/
def sum_property (ab mb : TwoDigitNumber) : Prop :=
  5 * (10 * ab.tens + ab.ones) = 10 * mb.tens + mb.ones

/-- Different letters represent different digits -/
def different_digits (ab mb : TwoDigitNumber) : Prop :=
  ab.tens ≠ ab.ones ∧ 
  (ab.tens ≠ mb.tens ∨ ab.ones ≠ mb.ones)

theorem two_digit_sum_theorem (ab mb : TwoDigitNumber) 
  (h_sum : sum_property ab mb) 
  (h_diff : different_digits ab mb) : 
  (ab.tens = 1 ∧ ab.ones = 0) ∨ (ab.tens = 1 ∧ ab.ones = 5) :=
sorry

end two_digit_sum_theorem_l3070_307098


namespace geometric_sequence_property_l3070_307066

/-- Given a geometric sequence with first term b₁ and common ratio q,
    T_n represents the product of the first n terms. -/
def T (b₁ q : ℝ) (n : ℕ) : ℝ :=
  b₁^n * q^(n * (n - 1) / 2)

/-- Theorem: For a geometric sequence, T_4, T_8/T_4, T_12/T_8, T_16/T_12 form a geometric sequence -/
theorem geometric_sequence_property (b₁ q : ℝ) (b₁_pos : 0 < b₁) (q_pos : 0 < q) :
  ∃ r : ℝ, r ≠ 0 ∧
    (T b₁ q 8 / T b₁ q 4) = (T b₁ q 4) * r ∧
    (T b₁ q 12 / T b₁ q 8) = (T b₁ q 8 / T b₁ q 4) * r ∧
    (T b₁ q 16 / T b₁ q 12) = (T b₁ q 12 / T b₁ q 8) * r :=
by
  sorry

end geometric_sequence_property_l3070_307066


namespace pythagorean_inequality_l3070_307067

theorem pythagorean_inequality (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h5 : a^2 + b^2 = c^2 + a*b) :
  c^2 + a*b < a*c + b*c := by
  sorry

end pythagorean_inequality_l3070_307067


namespace lcm_hcf_product_l3070_307040

theorem lcm_hcf_product (a b : ℕ+) (h1 : Nat.lcm a b = 72) (h2 : a * b = 432) :
  Nat.gcd a b = 6 := by
sorry

end lcm_hcf_product_l3070_307040


namespace saturday_to_weekday_ratio_total_weekly_time_correct_total_weekly_time_is_four_hours_l3070_307020

/-- Represents the number of minutes Elle practices piano on different days of the week. -/
structure PracticeTimes where
  weekday : Nat  -- Practice time on each weekday (Monday to Friday)
  saturday : Nat -- Practice time on Saturday
  total_weekly : Nat -- Total practice time in the week

/-- Represents the practice schedule of Elle -/
def elles_practice : PracticeTimes where
  weekday := 30
  saturday := 90
  total_weekly := 240

/-- The ratio of Saturday practice time to weekday practice time is 3:1 -/
theorem saturday_to_weekday_ratio :
  elles_practice.saturday / elles_practice.weekday = 3 := by
  sorry

/-- The total weekly practice time is correct -/
theorem total_weekly_time_correct :
  elles_practice.total_weekly = elles_practice.weekday * 5 + elles_practice.saturday := by
  sorry

/-- The total weekly practice time is 4 hours -/
theorem total_weekly_time_is_four_hours :
  elles_practice.total_weekly = 4 * 60 := by
  sorry

end saturday_to_weekday_ratio_total_weekly_time_correct_total_weekly_time_is_four_hours_l3070_307020


namespace tile_arrangements_l3070_307036

def brown_tiles : ℕ := 1
def purple_tiles : ℕ := 2
def green_tiles : ℕ := 3
def yellow_tiles : ℕ := 2

def total_tiles : ℕ := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem tile_arrangements :
  (Nat.factorial total_tiles) / 
  (Nat.factorial brown_tiles * Nat.factorial purple_tiles * 
   Nat.factorial green_tiles * Nat.factorial yellow_tiles) = 1680 := by
  sorry

end tile_arrangements_l3070_307036


namespace min_value_cubic_function_l3070_307060

/-- A cubic function f(x) = ax³ + bx² + cx + d -/
def cubic_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The function is monotonically increasing on ℝ -/
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The theorem statement -/
theorem min_value_cubic_function (a b c d : ℝ) :
  a > 0 →
  a < (2/3) * b →
  monotonically_increasing (cubic_function a b c d) →
  (c / (2 * b - 3 * a)) ≥ 1 :=
sorry

end min_value_cubic_function_l3070_307060


namespace garden_dimensions_l3070_307048

/-- Represents a rectangular garden with a surrounding walkway -/
structure GardenWithWalkway where
  garden_width : ℝ
  walkway_width : ℝ

/-- The combined area of the garden and walkway is 432 square meters -/
axiom total_area (g : GardenWithWalkway) : 
  (g.garden_width + 2 * g.walkway_width) * (3 * g.garden_width + 2 * g.walkway_width) = 432

/-- The area of the walkway alone is 108 square meters -/
axiom walkway_area (g : GardenWithWalkway) : 
  (g.garden_width + 2 * g.walkway_width) * (3 * g.garden_width + 2 * g.walkway_width) - 
  3 * g.garden_width * g.garden_width = 108

/-- The dimensions of the garden are 6√3 and 18√3 meters -/
theorem garden_dimensions (g : GardenWithWalkway) : 
  g.garden_width = 6 * Real.sqrt 3 ∧ 3 * g.garden_width = 18 * Real.sqrt 3 :=
by sorry

end garden_dimensions_l3070_307048


namespace abs_inequality_l3070_307068

theorem abs_inequality (a_n e : ℝ) (h : |a_n - e| < 1) : |a_n| < |e| + 1 := by
  sorry

end abs_inequality_l3070_307068


namespace linear_regression_center_point_l3070_307099

/-- Given a linear regression equation y = 0.2x - m with the center of sample points at (m, 1.6), prove that m = -2 -/
theorem linear_regression_center_point (m : ℝ) : 
  (∀ x y : ℝ, y = 0.2 * x - m) → -- Linear regression equation
  (m, 1.6) = (m, 0.2 * m - m) → -- Center of sample points
  m = -2 := by
sorry

end linear_regression_center_point_l3070_307099


namespace AP_coordinates_l3070_307076

-- Define the vectors OA and OB
def OA : ℝ × ℝ × ℝ := (1, -1, 1)
def OB : ℝ × ℝ × ℝ := (2, 0, -1)

-- Define point P on line segment AB
def P : ℝ × ℝ × ℝ := sorry

-- Define the condition AP = 2PB
def AP_eq_2PB : ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ P = (1 - t) • OA + t • OB ∧ t = 2/3 := sorry

-- Theorem to prove
theorem AP_coordinates : 
  let AP := P - OA
  AP = (2/3, 2/3, -4/3) :=
sorry

end AP_coordinates_l3070_307076


namespace shop_equations_correct_l3070_307057

/-- Represents a shop with rooms and guests -/
structure Shop where
  rooms : ℕ
  guests : ℕ

/-- The system of equations for the shop problem -/
def shop_equations (s : Shop) : Prop :=
  (7 * s.rooms + 7 = s.guests) ∧ (9 * (s.rooms - 1) = s.guests)

/-- Theorem stating that the shop equations correctly represent the given conditions -/
theorem shop_equations_correct (s : Shop) :
  (∀ (r : ℕ), r * s.rooms + 7 = s.guests → r = 7) ∧
  (∀ (r : ℕ), r * (s.rooms - 1) = s.guests → r = 9) →
  shop_equations s :=
sorry

end shop_equations_correct_l3070_307057


namespace historicalFictionNewReleasesFractionIs12_47_l3070_307047

/-- Represents a bookstore inventory --/
structure Inventory where
  total : ℕ
  historicalFiction : ℕ
  historicalFictionNewReleases : ℕ
  otherNewReleases : ℕ

/-- Conditions of the bookstore inventory --/
def validInventory (i : Inventory) : Prop :=
  i.historicalFiction = (30 * i.total) / 100 ∧
  i.historicalFictionNewReleases = (40 * i.historicalFiction) / 100 ∧
  i.otherNewReleases = (50 * (i.total - i.historicalFiction)) / 100

/-- The fraction of new releases that are historical fiction --/
def historicalFictionNewReleasesFraction (i : Inventory) : ℚ :=
  i.historicalFictionNewReleases / (i.historicalFictionNewReleases + i.otherNewReleases)

/-- Theorem stating the fraction of new releases that are historical fiction --/
theorem historicalFictionNewReleasesFractionIs12_47 (i : Inventory) 
  (h : validInventory i) : historicalFictionNewReleasesFraction i = 12 / 47 := by
  sorry

end historicalFictionNewReleasesFractionIs12_47_l3070_307047
