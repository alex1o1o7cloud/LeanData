import Mathlib

namespace marias_savings_l2257_225701

/-- Represents the cost of the bike in dollars -/
def bike_cost : ℕ := 600

/-- Represents the amount Maria's mother offered in dollars -/
def mother_offer : ℕ := 250

/-- Represents the amount Maria needs to earn in dollars -/
def amount_to_earn : ℕ := 230

/-- Represents Maria's initial savings in dollars -/
def initial_savings : ℕ := 120

theorem marias_savings :
  initial_savings + mother_offer + amount_to_earn = bike_cost :=
sorry

end marias_savings_l2257_225701


namespace linear_equation_solution_l2257_225799

/-- Given a linear equation y = kx + b, prove the values of k and b,
    and find x for a specific y value. -/
theorem linear_equation_solution (k b : ℝ) :
  (4 * k + b = -20 ∧ -2 * k + b = 16) →
  (k = -6 ∧ b = 4) ∧
  (∀ x : ℝ, -6 * x + 4 = -8 → x = 2) :=
by sorry

end linear_equation_solution_l2257_225799


namespace combined_molecular_weight_l2257_225713

/-- Atomic weight of Nitrogen in g/mol -/
def N_weight : ℝ := 14.01

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- Atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- Molecular weight of N2O3 in g/mol -/
def N2O3_weight : ℝ := 2 * N_weight + 3 * O_weight

/-- Molecular weight of H2O in g/mol -/
def H2O_weight : ℝ := 2 * H_weight + O_weight

/-- Molecular weight of CO2 in g/mol -/
def CO2_weight : ℝ := C_weight + 2 * O_weight

/-- Combined molecular weight of 4 moles of N2O3, 3.5 moles of H2O, and 2 moles of CO2 in grams -/
theorem combined_molecular_weight :
  4 * N2O3_weight + 3.5 * H2O_weight + 2 * CO2_weight = 455.17 := by
  sorry

end combined_molecular_weight_l2257_225713


namespace find_A_l2257_225731

theorem find_A : ∃ A : ℤ, A + 10 = 15 → A = 5 := by
  sorry

end find_A_l2257_225731


namespace investment_rate_problem_l2257_225792

/-- Calculates simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_problem (principal : ℝ) (time : ℝ) (standardRate : ℝ) (additionalInterest : ℝ) :
  principal = 2500 →
  time = 2 →
  standardRate = 0.12 →
  additionalInterest = 300 →
  ∃ (rate : ℝ),
    simpleInterest principal rate time = simpleInterest principal standardRate time + additionalInterest ∧
    rate = 0.18 := by
  sorry

end investment_rate_problem_l2257_225792


namespace correct_production_matching_equation_l2257_225768

/-- Represents a workshop producing bolts and nuts -/
structure Workshop where
  total_workers : ℕ
  bolt_production_rate : ℕ
  nut_production_rate : ℕ
  nuts_per_bolt : ℕ

/-- The equation for matching bolt and nut production in the workshop -/
def production_matching_equation (w : Workshop) (x : ℕ) : Prop :=
  2 * w.bolt_production_rate * x = w.nut_production_rate * (w.total_workers - x)

/-- Theorem stating the correct equation for matching bolt and nut production -/
theorem correct_production_matching_equation (w : Workshop) 
  (h1 : w.total_workers = 28)
  (h2 : w.bolt_production_rate = 12)
  (h3 : w.nut_production_rate = 18)
  (h4 : w.nuts_per_bolt = 2) :
  ∀ x, production_matching_equation w x ↔ 2 * 12 * x = 18 * (28 - x) :=
by
  sorry


end correct_production_matching_equation_l2257_225768


namespace sqrt_sum_equality_l2257_225718

theorem sqrt_sum_equality : 
  Real.sqrt 9 + Real.sqrt (9 + 11) + Real.sqrt (9 + 11 + 13) + 
  Real.sqrt (9 + 11 + 13 + 15) + Real.sqrt (9 + 11 + 13 + 15 + 17) = 
  3 + 2 * Real.sqrt 5 + Real.sqrt 33 + 4 * Real.sqrt 3 + Real.sqrt 65 := by
  sorry

end sqrt_sum_equality_l2257_225718


namespace amusement_park_revenue_l2257_225753

def ticket_price : ℕ := 3
def weekday_visitors : ℕ := 100
def saturday_visitors : ℕ := 200
def sunday_visitors : ℕ := 300
def days_in_week : ℕ := 7
def weekdays : ℕ := 5

def total_revenue : ℕ := ticket_price * (weekday_visitors * weekdays + saturday_visitors + sunday_visitors)

theorem amusement_park_revenue : total_revenue = 3000 := by
  sorry

end amusement_park_revenue_l2257_225753


namespace engine_capacity_proof_l2257_225772

/-- Represents the relationship between diesel volume, distance, and engine capacity -/
structure DieselEngineRelation where
  volume : ℝ  -- Volume of diesel in litres
  distance : ℝ  -- Distance in km
  capacity : ℝ  -- Engine capacity in cc

/-- The relation between diesel volume and engine capacity is directly proportional -/
axiom diesel_capacity_proportion (r1 r2 : DieselEngineRelation) :
  r1.volume / r1.capacity = r2.volume / r2.capacity

/-- Given data for the first scenario -/
def scenario1 : DieselEngineRelation :=
  { volume := 60, distance := 600, capacity := 800 }

/-- Given data for the second scenario -/
def scenario2 : DieselEngineRelation :=
  { volume := 120, distance := 800, capacity := 1600 }

/-- Theorem stating that the engine capacity for the second scenario is 1600 cc -/
theorem engine_capacity_proof :
  scenario2.capacity = 1600 :=
by sorry

end engine_capacity_proof_l2257_225772


namespace fourth_quarter_total_points_l2257_225781

/-- Represents the points scored by a team in each quarter -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- The game between Raiders and Wildcats -/
structure BasketballGame :=
  (raiders : TeamScores)
  (wildcats : TeamScores)

/-- Conditions of the game -/
def game_conditions (g : BasketballGame) : Prop :=
  let r := g.raiders
  let w := g.wildcats
  -- Game tied at halftime
  r.q1 + r.q2 = w.q1 + w.q2 ∧
  -- Raiders' points form an increasing arithmetic sequence
  ∃ (d : ℕ), r.q2 = r.q1 + d ∧ r.q3 = r.q2 + d ∧ r.q4 = r.q3 + d ∧
  -- Wildcats' points are equal in first two quarters, then decrease by same difference
  ∃ (j : ℕ), w.q1 = w.q2 ∧ w.q3 = w.q2 - j ∧ w.q4 = w.q3 - j ∧
  -- Wildcats won by exactly four points
  (w.q1 + w.q2 + w.q3 + w.q4) = (r.q1 + r.q2 + r.q3 + r.q4) + 4

theorem fourth_quarter_total_points (g : BasketballGame) :
  game_conditions g → g.raiders.q4 + g.wildcats.q4 = 28 :=
by sorry

end fourth_quarter_total_points_l2257_225781


namespace geometric_sequence_increasing_condition_l2257_225717

def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def IsIncreasingSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) :
  IsGeometricSequence a →
  (a 1 < a 2 ∧ a 2 < a 3) ↔ IsIncreasingSequence a :=
sorry

end geometric_sequence_increasing_condition_l2257_225717


namespace circular_coin_flip_probability_l2257_225783

def num_people : ℕ := 10

-- Function to calculate the number of valid arrangements
def valid_arrangements : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| n + 3 => valid_arrangements (n + 1) + valid_arrangements (n + 2)

theorem circular_coin_flip_probability :
  (valid_arrangements num_people : ℚ) / 2^num_people = 123 / 1024 := by sorry

end circular_coin_flip_probability_l2257_225783


namespace people_born_in_country_l2257_225736

theorem people_born_in_country (immigrants : ℕ) (new_residents : ℕ) 
  (h1 : immigrants = 16320) 
  (h2 : new_residents = 106491) : 
  new_residents - immigrants = 90171 := by
sorry

end people_born_in_country_l2257_225736


namespace monic_quartic_problem_l2257_225774

-- Define a monic quartic polynomial
def monicQuartic (p : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + d

-- State the theorem
theorem monic_quartic_problem (p : ℝ → ℝ) 
  (h_monic : monicQuartic p)
  (h1 : p 1 = 3)
  (h2 : p 2 = 7)
  (h3 : p 3 = 13)
  (h4 : p 4 = 21) :
  p 5 = 51 := by
  sorry

end monic_quartic_problem_l2257_225774


namespace line_length_after_erasing_l2257_225751

/-- Calculates the remaining length of a line after erasing a portion. -/
def remaining_length (initial_length : ℝ) (erased_length : ℝ) : ℝ :=
  initial_length - erased_length

/-- Proves that erasing 24 cm from a 1 m line results in a 76 cm line. -/
theorem line_length_after_erasing :
  remaining_length 100 24 = 76 := by
  sorry

#check line_length_after_erasing

end line_length_after_erasing_l2257_225751


namespace arithmetic_mean_first_n_odd_l2257_225704

/-- The sum of the first n odd positive integers -/
def sum_first_n_odd (n : ℕ) : ℕ := n^2

/-- The arithmetic mean of a list of numbers -/
def arithmetic_mean (sum : ℕ) (count : ℕ) : ℚ := sum / count

theorem arithmetic_mean_first_n_odd (n : ℕ) :
  arithmetic_mean (sum_first_n_odd n) n = n := by sorry

end arithmetic_mean_first_n_odd_l2257_225704


namespace polynomial_integrality_l2257_225779

theorem polynomial_integrality (x : ℤ) : ∃ k : ℤ, (1/5 : ℚ) * x^5 + (1/3 : ℚ) * x^3 + (7/15 : ℚ) * x = k := by
  sorry

end polynomial_integrality_l2257_225779


namespace set_equality_l2257_225797

def positive_naturals : Set ℕ := {n : ℕ | n > 0}

def set_A : Set ℕ := {x ∈ positive_naturals | x - 3 < 2}
def set_B : Set ℕ := {1, 2, 3, 4}

theorem set_equality : set_A = set_B := by sorry

end set_equality_l2257_225797


namespace prob_first_red_given_second_black_l2257_225735

/-- Represents the contents of an urn -/
structure Urn :=
  (white : ℕ)
  (red : ℕ)
  (black : ℕ)

/-- The probability of drawing a specific color from an urn -/
def prob_draw (u : Urn) (color : String) : ℚ :=
  match color with
  | "white" => u.white / (u.white + u.red + u.black)
  | "red" => u.red / (u.white + u.red + u.black)
  | "black" => u.black / (u.white + u.red + u.black)
  | _ => 0

/-- The contents of Urn A -/
def urn_A : Urn := ⟨4, 2, 0⟩

/-- The contents of Urn B -/
def urn_B : Urn := ⟨0, 3, 3⟩

/-- The probability of selecting an urn -/
def prob_select_urn : ℚ := 1/2

theorem prob_first_red_given_second_black :
  let p_red_and_black := 
    (prob_select_urn * prob_draw urn_A "red" * prob_select_urn * prob_draw urn_B "black") +
    (prob_select_urn * prob_draw urn_B "red" * prob_select_urn * (prob_draw urn_B "black" * (urn_B.black - 1) / (urn_B.red + urn_B.black - 1)))
  let p_second_black :=
    (prob_select_urn * prob_select_urn * prob_draw urn_B "black") +
    (prob_select_urn * prob_draw urn_B "red" * prob_select_urn * (prob_draw urn_B "black" * (urn_B.black) / (urn_B.red + urn_B.black - 1))) +
    (prob_select_urn * prob_draw urn_B "black" * prob_select_urn * (prob_draw urn_B "black" * (urn_B.black - 1) / (urn_B.red + urn_B.black - 1)))
  p_red_and_black / p_second_black = 7/15 := by
  sorry

end prob_first_red_given_second_black_l2257_225735


namespace dans_initial_money_l2257_225798

/-- Given that Dan bought a candy bar for $7 and a chocolate for $6,
    and spent $13 in total, prove that his initial amount was $13. -/
theorem dans_initial_money :
  ∀ (candy_price chocolate_price total_spent initial_amount : ℕ),
    candy_price = 7 →
    chocolate_price = 6 →
    total_spent = 13 →
    total_spent = candy_price + chocolate_price →
    initial_amount = total_spent →
    initial_amount = 13 :=
by
  sorry

end dans_initial_money_l2257_225798


namespace correct_operation_l2257_225706

theorem correct_operation (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end correct_operation_l2257_225706


namespace boards_nailed_proof_l2257_225784

/-- Represents the number of boards nailed by each person -/
def num_boards : ℕ := 30

/-- Represents the total number of nails used by Petrov -/
def petrov_nails : ℕ := 87

/-- Represents the total number of nails used by Vasechkin -/
def vasechkin_nails : ℕ := 94

/-- Theorem stating that the number of boards nailed by each person is 30 -/
theorem boards_nailed_proof :
  ∃ (p2 p3 v3 v5 : ℕ),
    p2 + p3 = num_boards ∧
    v3 + v5 = num_boards ∧
    2 * p2 + 3 * p3 = petrov_nails ∧
    3 * v3 + 5 * v5 = vasechkin_nails :=
by sorry


end boards_nailed_proof_l2257_225784


namespace fraction_simplification_l2257_225722

theorem fraction_simplification :
  ((2^1010)^2 - (2^1008)^2) / ((2^1009)^2 - (2^1007)^2) = 4 := by
  sorry

end fraction_simplification_l2257_225722


namespace hot_dog_buns_packages_l2257_225761

/-- Calculates the number of packages of hot dog buns needed for a school picnic --/
theorem hot_dog_buns_packages (buns_per_package : ℕ) (num_classes : ℕ) (students_per_class : ℕ) (buns_per_student : ℕ) : 
  buns_per_package = 8 →
  num_classes = 4 →
  students_per_class = 30 →
  buns_per_student = 2 →
  (num_classes * students_per_class * buns_per_student + buns_per_package - 1) / buns_per_package = 30 := by
  sorry

#eval (4 * 30 * 2 + 8 - 1) / 8  -- Should output 30

end hot_dog_buns_packages_l2257_225761


namespace parallel_implies_parallel_to_intersection_l2257_225728

/-- Represents a line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here
  mk :: -- Add constructor parameters here

/-- Checks if two lines are parallel -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Checks if a line lies on a plane -/
def lies_on (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Returns the intersection line of two planes -/
def intersection (p1 p2 : Plane3D) : Line3D :=
  sorry

theorem parallel_implies_parallel_to_intersection
  (a b c : Line3D) (M N : Plane3D)
  (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h2 : lies_on a M)
  (h3 : lies_on b N)
  (h4 : c = intersection M N)
  (h5 : parallel a b) :
  parallel a c :=
sorry

end parallel_implies_parallel_to_intersection_l2257_225728


namespace binomial_sum_unique_l2257_225724

theorem binomial_sum_unique (m : ℤ) : 
  (Nat.choose 25 m.toNat + Nat.choose 25 12 = Nat.choose 26 13) ↔ m = 13 :=
sorry

end binomial_sum_unique_l2257_225724


namespace college_student_count_l2257_225708

/-- The total number of students at the college -/
def total_students : ℕ := 880

/-- The percentage of students enrolled in biology classes -/
def biology_enrollment_percentage : ℚ := 47.5 / 100

/-- The number of students not enrolled in biology classes -/
def students_not_in_biology : ℕ := 462

/-- Theorem stating the total number of students at the college -/
theorem college_student_count :
  total_students = students_not_in_biology / (1 - biology_enrollment_percentage) := by
  sorry

end college_student_count_l2257_225708


namespace minimal_area_circle_equation_l2257_225740

/-- The standard equation of a circle with minimal area, given that its center is on the curve y² = x
    and it is tangent to the line x + 2y + 6 = 0 -/
theorem minimal_area_circle_equation (x y : ℝ) : 
  (∃ (cx cy : ℝ), cy^2 = cx ∧ (x - cx)^2 + (y - cy)^2 = ((x + 2*y + 6) / Real.sqrt 5)^2) →
  (x - 1)^2 + (y + 1)^2 = 5 := by
sorry

end minimal_area_circle_equation_l2257_225740


namespace apple_cost_price_l2257_225711

/-- Proves that the cost price of an apple is 24 rupees, given the selling price and loss ratio. -/
theorem apple_cost_price (selling_price : ℚ) (loss_ratio : ℚ) : 
  selling_price = 20 → loss_ratio = 1/6 → 
  ∃ cost_price : ℚ, cost_price = 24 ∧ selling_price = cost_price - loss_ratio * cost_price :=
by sorry

end apple_cost_price_l2257_225711


namespace expression_evaluation_l2257_225758

theorem expression_evaluation :
  let x : ℚ := -2
  let y : ℚ := 1/2
  (x + 2*y)^2 - (x + y)*(3*x - y) - 5*y^2 = -10 := by
  sorry

end expression_evaluation_l2257_225758


namespace ln_concave_l2257_225709

/-- The natural logarithm function is concave on the positive real numbers. -/
theorem ln_concave : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 →
  Real.log ((x₁ + x₂) / 2) ≥ (Real.log x₁ + Real.log x₂) / 2 := by
  sorry

end ln_concave_l2257_225709


namespace total_pints_picked_l2257_225721

def annie_pints : ℕ := 8

def kathryn_pints (annie : ℕ) : ℕ := annie + 2

def ben_pints (kathryn : ℕ) : ℕ := kathryn - 3

theorem total_pints_picked :
  annie_pints + kathryn_pints annie_pints + ben_pints (kathryn_pints annie_pints) = 25 := by
  sorry

end total_pints_picked_l2257_225721


namespace derivative_x_minus_reciprocal_l2257_225726

/-- The derivative of f(x) = x - 1/x is f'(x) = 1 + 1/x^2 -/
theorem derivative_x_minus_reciprocal (x : ℝ) (hx : x ≠ 0) :
  deriv (fun x => x - 1 / x) x = 1 + 1 / x^2 := by
  sorry

end derivative_x_minus_reciprocal_l2257_225726


namespace polynomial_simplification_l2257_225748

theorem polynomial_simplification (x : ℝ) : 
  (x - 2)^4 - 4*(x - 2)^3 + 6*(x - 2)^2 - 4*(x - 2) + 1 = (x - 3)^4 := by
  sorry

end polynomial_simplification_l2257_225748


namespace sales_price_calculation_l2257_225760

theorem sales_price_calculation (C G : ℝ) (h1 : G = 1.6 * C) (h2 : G = 56) :
  C + G = 91 := by
  sorry

end sales_price_calculation_l2257_225760


namespace compute_expression_l2257_225773

theorem compute_expression : (12 : ℚ) * (1/3 + 1/4 + 1/6)⁻¹ = 16 := by
  sorry

end compute_expression_l2257_225773


namespace geometric_sequence_sum_property_l2257_225780

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum_property
  (a : ℕ → ℝ)
  (h_geom : geometric_sequence a)
  (h_sum1 : a 1 + a 2 = 40)
  (h_sum2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 := by
sorry

end geometric_sequence_sum_property_l2257_225780


namespace cylinder_lateral_surface_area_l2257_225729

/-- A cylinder with base area S whose lateral surface unfolds into a square has lateral surface area 4πS -/
theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let h := 2 * Real.pi * r
  h = 2 * Real.pi * r →
  2 * Real.pi * r * h = 4 * Real.pi * S :=
by sorry

end cylinder_lateral_surface_area_l2257_225729


namespace alice_prob_three_turns_l2257_225763

/-- Represents the person holding the ball -/
inductive Person : Type
| Alice : Person
| Bob : Person

/-- The probability of tossing the ball for each person -/
def toss_prob (p : Person) : ℚ :=
  match p with
  | Person.Alice => 1/3
  | Person.Bob => 1/4

/-- The probability of keeping the ball for each person -/
def keep_prob (p : Person) : ℚ :=
  1 - toss_prob p

/-- The probability of Alice having the ball after n turns, given she starts with it -/
def alice_prob (n : ℕ) : ℚ :=
  sorry

theorem alice_prob_three_turns :
  alice_prob 3 = 227/432 :=
sorry

end alice_prob_three_turns_l2257_225763


namespace smallest_n_congruence_l2257_225782

theorem smallest_n_congruence :
  ∃ (n : ℕ), n > 0 ∧ (23 * n) % 11 = 5678 % 11 ∧
  ∀ (m : ℕ), m > 0 ∧ (23 * m) % 11 = 5678 % 11 → n ≤ m :=
by sorry

end smallest_n_congruence_l2257_225782


namespace triangle_area_is_six_l2257_225771

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ := 
  (1/2) * b * c * Real.sin A

theorem triangle_area_is_six (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 4)
  (h2 : Real.tan A = 3)
  (h3 : Real.cos C = Real.sqrt 5 / 5) : 
  triangle_area a b c A B C = 6 := by
  sorry

end triangle_area_is_six_l2257_225771


namespace davids_travel_expenses_l2257_225756

/-- Represents currency amounts in their respective denominations -/
structure Expenses where
  usd : ℝ
  eur : ℝ
  gbp : ℝ
  jpy : ℝ

/-- Represents exchange rates to USD -/
structure ExchangeRates where
  eur_to_usd : ℝ
  gbp_to_usd : ℝ
  jpy_to_usd : ℝ

/-- Calculates the total expenses in USD -/
def total_expenses (e : Expenses) (r : ExchangeRates) : ℝ :=
  e.usd + e.eur * r.eur_to_usd + e.gbp * r.gbp_to_usd + e.jpy * r.jpy_to_usd

/-- Theorem representing David's travel expenses problem -/
theorem davids_travel_expenses 
  (initial_amount : ℝ)
  (expenses : Expenses)
  (initial_rates : ExchangeRates)
  (final_rates : ExchangeRates)
  (loan : ℝ)
  (h1 : initial_amount = 1500)
  (h2 : expenses = { usd := 400, eur := 300, gbp := 150, jpy := 5000 })
  (h3 : initial_rates = { eur_to_usd := 1.10, gbp_to_usd := 1.35, jpy_to_usd := 0.009 })
  (h4 : final_rates = { eur_to_usd := 1.08, gbp_to_usd := 1.32, jpy_to_usd := 0.009 })
  (h5 : loan = 200)
  (h6 : initial_amount - total_expenses expenses initial_rates - loan = 
        total_expenses expenses initial_rates - 500) :
  initial_amount - total_expenses expenses initial_rates + loan = 677.5 := by
  sorry

end davids_travel_expenses_l2257_225756


namespace simplify_expression_l2257_225765

theorem simplify_expression (x : ℝ) : (3*x - 10) + (7*x + 20) - (2*x - 5) = 8*x + 15 := by
  sorry

end simplify_expression_l2257_225765


namespace store_revenue_l2257_225762

theorem store_revenue (december : ℝ) (h1 : december > 0) : 
  let november := (2 / 5 : ℝ) * december
  let january := (1 / 3 : ℝ) * november
  let average := (november + january) / 2
  december / average = 15 / 4 := by
sorry

end store_revenue_l2257_225762


namespace smallest_n_congruence_l2257_225787

theorem smallest_n_congruence (k : ℕ) (h : k > 0) :
  (7 ^ k) % 3 = (k ^ 7) % 3 → k ≥ 1 :=
by sorry

end smallest_n_congruence_l2257_225787


namespace diamond_equation_solution_l2257_225747

/-- The diamond operation defined as a ◇ b = a * sqrt(b + sqrt(b + sqrt(b + ...))) -/
noncomputable def diamond (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

/-- Theorem stating that if 2 ◇ h = 8, then h = 12 -/
theorem diamond_equation_solution (h : ℝ) (eq : diamond 2 h = 8) : h = 12 := by
  sorry

end diamond_equation_solution_l2257_225747


namespace parabola_intersection_angle_l2257_225793

/-- Parabola type -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Line type -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- Intersection point type -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- Theorem statement -/
theorem parabola_intersection_angle (C : Parabola) (F M : ℝ × ℝ) (l : Line) 
  (A B : IntersectionPoint) :
  C.equation = (fun x y => y^2 = 8*x) →
  F = (2, 0) →
  M = (-2, 2) →
  l.point = F →
  (C.equation A.x A.y ∧ C.equation B.x B.y) →
  (A.y - M.2) * (B.y - M.2) = -(A.x - M.1) * (B.x - M.1) →
  l.slope = 2 :=
sorry

end parabola_intersection_angle_l2257_225793


namespace six_digit_multiple_of_99_l2257_225723

theorem six_digit_multiple_of_99 : ∃ n : ℕ, 
  (n ≥ 978600 ∧ n < 978700) ∧  -- Six-digit number starting with 9786
  (n % 99 = 0) ∧               -- Divisible by 99
  (n / 99 = 6039) :=           -- Quotient is 6039
by sorry

end six_digit_multiple_of_99_l2257_225723


namespace floor_sum_sqrt_equals_floor_sqrt_9n_plus_8_l2257_225769

theorem floor_sum_sqrt_equals_floor_sqrt_9n_plus_8 (n : ℕ) :
  ⌊Real.sqrt n + Real.sqrt (n + 1) + Real.sqrt (n + 2)⌋ = ⌊Real.sqrt (9 * n + 8)⌋ :=
sorry

end floor_sum_sqrt_equals_floor_sqrt_9n_plus_8_l2257_225769


namespace tree_walk_properties_l2257_225714

/-- Represents a random walk on a line of trees. -/
structure TreeWalk where
  n : ℕ
  trees : Fin (2 * n + 1) → ℕ
  start : Fin (2 * n + 1)
  prob_left : ℚ
  prob_stay : ℚ
  prob_right : ℚ

/-- The probability of ending at a specific tree after the walk. -/
def end_probability (w : TreeWalk) (i : Fin (2 * w.n + 1)) : ℚ :=
  (Nat.choose (2 * w.n) (i - 1)) / (2 ^ (2 * w.n))

/-- The expected distance from the starting point after the walk. -/
def expected_distance (w : TreeWalk) : ℚ :=
  (w.n * Nat.choose (2 * w.n) w.n) / (2 ^ (2 * w.n))

/-- Theorem stating the properties of the random walk. -/
theorem tree_walk_properties (w : TreeWalk) 
  (h1 : w.n > 0)
  (h2 : w.start = ⟨w.n + 1, by sorry⟩)
  (h3 : w.prob_left = 1/4)
  (h4 : w.prob_stay = 1/2)
  (h5 : w.prob_right = 1/4) :
  (∀ i : Fin (2 * w.n + 1), end_probability w i = (Nat.choose (2 * w.n) (i - 1)) / (2 ^ (2 * w.n))) ∧
  expected_distance w = (w.n * Nat.choose (2 * w.n) w.n) / (2 ^ (2 * w.n)) := by
  sorry


end tree_walk_properties_l2257_225714


namespace new_oarsman_weight_l2257_225707

/-- Given a crew of 10 oarsmen, proves that replacing a 53 kg member with a new member
    that increases the average weight by 1.8 kg results in the new member weighing 71 kg. -/
theorem new_oarsman_weight (crew_size : Nat) (old_weight : ℝ) (avg_increase : ℝ) :
  crew_size = 10 →
  old_weight = 53 →
  avg_increase = 1.8 →
  (crew_size : ℝ) * avg_increase + old_weight = 71 :=
by sorry

end new_oarsman_weight_l2257_225707


namespace set_inequality_l2257_225794

def S : Set ℤ := {x | ∃ k : ℕ+, x = 2 * k + 1}
def A : Set ℤ := {x | ∃ k : ℕ+, x = 2 * k - 1}

theorem set_inequality : A ≠ S := by
  sorry

end set_inequality_l2257_225794


namespace david_drive_distance_david_drive_distance_proof_l2257_225741

theorem david_drive_distance : ℝ → Prop :=
  fun distance =>
    ∀ (initial_speed : ℝ) (increased_speed : ℝ) (on_time_duration : ℝ),
      initial_speed = 40 ∧
      increased_speed = initial_speed + 20 ∧
      distance = initial_speed * (on_time_duration + 1.5) ∧
      distance - initial_speed = increased_speed * (on_time_duration - 2) →
      distance = 340

-- The proof is omitted
theorem david_drive_distance_proof : david_drive_distance 340 := by sorry

end david_drive_distance_david_drive_distance_proof_l2257_225741


namespace units_digit_sum_factorials_10_l2257_225788

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def unitsDigit (n : ℕ) : ℕ := n % 10

def sumFactorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem units_digit_sum_factorials_10 :
  unitsDigit (sumFactorials 10) = 3 := by
  sorry

end units_digit_sum_factorials_10_l2257_225788


namespace x_plus_y_values_l2257_225786

theorem x_plus_y_values (x y : ℝ) 
  (eq1 : x^2 + x*y + 2*y = 10) 
  (eq2 : y^2 + x*y + 2*x = 14) : 
  x + y = 4 ∨ x + y = -6 := by
sorry

end x_plus_y_values_l2257_225786


namespace cos_difference_l2257_225737

theorem cos_difference (A B : Real) 
  (h1 : Real.sin A + Real.sin B = 1/2) 
  (h2 : Real.cos A + Real.cos B = 2) : 
  Real.cos (A - B) = 9/8 := by
  sorry

end cos_difference_l2257_225737


namespace sum_of_selected_numbers_l2257_225795

def set1 := Finset.Icc 10 19
def set2 := Finset.Icc 90 99

def is_valid_selection (s1 s2 : Finset ℕ) : Prop :=
  s1.card = 5 ∧ s2.card = 5 ∧ 
  s1 ⊆ set1 ∧ s2 ⊆ set2 ∧
  ∀ x ∈ s1, ∀ y ∈ s1, x ≠ y → (x - y) % 10 ≠ 0 ∧
  ∀ x ∈ s2, ∀ y ∈ s2, x ≠ y → (x - y) % 10 ≠ 0 ∧
  ∀ x ∈ s1, ∀ y ∈ s2, (x - y) % 10 ≠ 0

theorem sum_of_selected_numbers (s1 s2 : Finset ℕ) 
  (h : is_valid_selection s1 s2) : 
  (s1.sum id + s2.sum id) = 545 := by
  sorry

end sum_of_selected_numbers_l2257_225795


namespace circle_center_correct_l2257_225777

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 5 = 0

/-- The center of the circle -/
def CircleCenter : ℝ × ℝ := (2, 1)

/-- Theorem: The center of the circle defined by CircleEquation is CircleCenter -/
theorem circle_center_correct :
  ∀ (x y : ℝ), CircleEquation x y ↔ (x - CircleCenter.1)^2 + (y - CircleCenter.2)^2 = 10 :=
by sorry

end circle_center_correct_l2257_225777


namespace devin_age_l2257_225725

theorem devin_age (devin_age eden_age mom_age : ℕ) : 
  eden_age = 2 * devin_age →
  mom_age = 2 * eden_age →
  (devin_age + eden_age + mom_age) / 3 = 28 →
  devin_age = 12 := by
sorry

end devin_age_l2257_225725


namespace cubic_feet_to_cubic_inches_l2257_225727

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Define the volume in cubic feet
def cubic_feet : ℕ := 4

-- Theorem statement
theorem cubic_feet_to_cubic_inches :
  cubic_feet * (inches_per_foot ^ 3) = 6912 := by
  sorry


end cubic_feet_to_cubic_inches_l2257_225727


namespace winnie_lollipops_theorem_l2257_225734

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (total_lollipops friends : ℕ) : ℕ :=
  total_lollipops % friends

theorem winnie_lollipops_theorem (cherry wintergreen grape shrimp friends : ℕ) :
  let total_lollipops := cherry + wintergreen + grape + shrimp
  lollipops_kept total_lollipops friends = 
    total_lollipops - friends * (total_lollipops / friends) := by
  sorry

#eval lollipops_kept (67 + 154 + 23 + 312) 17

end winnie_lollipops_theorem_l2257_225734


namespace solve_for_m_l2257_225766

theorem solve_for_m (x y m : ℝ) 
  (h1 : x = 3 * m + 1)
  (h2 : y = 2 * m - 2)
  (h3 : 4 * x - 3 * y = 10) : 
  m = 0 := by
sorry

end solve_for_m_l2257_225766


namespace cat_dog_positions_after_365_moves_l2257_225733

/-- Represents the positions on the 3x3 grid --/
inductive GridPosition
  | TopLeft | TopCenter | TopRight
  | MiddleLeft | Center | MiddleRight
  | BottomLeft | BottomCenter | BottomRight

/-- Represents the edge positions on the 3x3 grid --/
inductive EdgePosition
  | LeftTop | LeftMiddle | LeftBottom
  | BottomLeft | BottomCenter | BottomRight
  | RightBottom | RightMiddle | RightTop
  | TopRight | TopCenter | TopLeft

/-- Calculates the cat's position after a given number of moves --/
def catPosition (moves : ℕ) : GridPosition :=
  match moves % 9 with
  | 0 => GridPosition.TopLeft
  | 1 => GridPosition.TopCenter
  | 2 => GridPosition.TopRight
  | 3 => GridPosition.MiddleRight
  | 4 => GridPosition.BottomRight
  | 5 => GridPosition.BottomCenter
  | 6 => GridPosition.BottomLeft
  | 7 => GridPosition.MiddleLeft
  | _ => GridPosition.Center

/-- Calculates the dog's position after a given number of moves --/
def dogPosition (moves : ℕ) : EdgePosition :=
  match moves % 16 with
  | 0 => EdgePosition.LeftMiddle
  | 1 => EdgePosition.LeftTop
  | 2 => EdgePosition.TopLeft
  | 3 => EdgePosition.TopCenter
  | 4 => EdgePosition.TopRight
  | 5 => EdgePosition.RightTop
  | 6 => EdgePosition.RightMiddle
  | 7 => EdgePosition.RightBottom
  | 8 => EdgePosition.BottomRight
  | 9 => EdgePosition.BottomCenter
  | 10 => EdgePosition.BottomLeft
  | 11 => EdgePosition.LeftBottom
  | 12 => EdgePosition.LeftMiddle
  | 13 => EdgePosition.LeftTop
  | 14 => EdgePosition.TopLeft
  | _ => EdgePosition.TopCenter

theorem cat_dog_positions_after_365_moves :
  catPosition 365 = GridPosition.Center ∧ dogPosition 365 = EdgePosition.LeftMiddle :=
sorry

end cat_dog_positions_after_365_moves_l2257_225733


namespace unique_quadratic_solution_l2257_225754

/-- Given a nonzero constant a for which the equation ax^2 + 16x + 9 = 0 has only one solution,
    prove that this solution is -9/8. -/
theorem unique_quadratic_solution (a : ℝ) (ha : a ≠ 0) 
    (h_unique : ∃! x : ℝ, a * x^2 + 16 * x + 9 = 0) :
  ∃ x : ℝ, a * x^2 + 16 * x + 9 = 0 ∧ x = -9/8 := by
  sorry

end unique_quadratic_solution_l2257_225754


namespace initial_barking_dogs_l2257_225742

theorem initial_barking_dogs (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  additional = 10 → total = 40 → initial + additional = total → initial = 30 := by
sorry

end initial_barking_dogs_l2257_225742


namespace rhombus60_min_rotation_l2257_225710

/-- A rhombus with a 60° angle -/
structure Rhombus60 where
  /-- The rhombus has a 60° angle -/
  angle : ℝ
  angle_eq : angle = 60

/-- The minimum rotation angle for a Rhombus60 to coincide with its original position -/
def min_rotation_angle (r : Rhombus60) : ℝ := 180

/-- Theorem stating that the minimum rotation angle for a Rhombus60 is 180° -/
theorem rhombus60_min_rotation (r : Rhombus60) :
  min_rotation_angle r = 180 := by sorry

end rhombus60_min_rotation_l2257_225710


namespace peter_bought_five_large_glasses_l2257_225719

/-- The number of large glasses Peter bought -/
def num_large_glasses (small_cost large_cost total_money num_small change : ℕ) : ℕ :=
  (total_money - change - small_cost * num_small) / large_cost

/-- Proof that Peter bought 5 large glasses -/
theorem peter_bought_five_large_glasses :
  num_large_glasses 3 5 50 8 1 = 5 := by
  sorry

end peter_bought_five_large_glasses_l2257_225719


namespace fraction_simplification_l2257_225732

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end fraction_simplification_l2257_225732


namespace chord_quadrilateral_probability_l2257_225745

/-- Given 7 points on a circle, the probability that 4 randomly selected chords
    form a convex quadrilateral is 1/171. -/
theorem chord_quadrilateral_probability :
  let n : ℕ := 7  -- number of points on the circle
  let k : ℕ := 4  -- number of chords selected
  let total_chords : ℕ := n.choose 2  -- total number of possible chords
  let total_selections : ℕ := total_chords.choose k  -- ways to select k chords
  let convex_quads : ℕ := n.choose k  -- number of convex quadrilaterals
  (convex_quads : ℚ) / total_selections = 1 / 171 := by
sorry

end chord_quadrilateral_probability_l2257_225745


namespace triangle_area_l2257_225767

open Real

/-- Given a triangle ABC where angle A is π/6 and the dot product of vectors AB and AC
    equals the tangent of angle A, prove that the area of the triangle is 1/6. -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let angle_A : ℝ := π / 6
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2 = tan angle_A →
  abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) / 2 = 1/6 := by
sorry

end triangle_area_l2257_225767


namespace bowTie_seven_eq_nine_impl_g_eq_two_l2257_225744

/-- The bow-tie operation defined as a + √(b + √(b + √(b + ...))) -/
noncomputable def bowTie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

/-- Theorem stating that if 7 ⋈ g = 9, then g = 2 -/
theorem bowTie_seven_eq_nine_impl_g_eq_two :
  ∀ g : ℝ, bowTie 7 g = 9 → g = 2 := by
  sorry

end bowTie_seven_eq_nine_impl_g_eq_two_l2257_225744


namespace base_number_proof_l2257_225700

theorem base_number_proof (n : ℕ) (x : ℝ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^26) 
  (h2 : n = 25) : 
  x = 4 := by
  sorry

end base_number_proof_l2257_225700


namespace equation_solution_l2257_225755

theorem equation_solution (x y z : ℝ) (hx : x > 1) (hy : y > 1) (hz : z > 1) 
  (h : x + y + z + 3 / (x - 1) + 3 / (y - 1) + 3 / (z - 1) = 
       2 * (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))) : 
  x = (3 + Real.sqrt 13) / 2 ∧ y = (3 + Real.sqrt 13) / 2 ∧ z = (3 + Real.sqrt 13) / 2 := by
  sorry

end equation_solution_l2257_225755


namespace max_sum_is_3972_l2257_225778

/-- A function that generates all possible permutations of 9 digits -/
def generatePermutations : List (List Nat) := sorry

/-- A function that splits a list of 9 digits into three numbers -/
def splitIntoThreeNumbers (perm : List Nat) : (Nat × Nat × Nat) := sorry

/-- A function that calculates the sum of three numbers -/
def sumThreeNumbers (nums : Nat × Nat × Nat) : Nat := sorry

/-- The maximum sum achievable using digits 1 to 9 -/
def maxSum : Nat := 3972

theorem max_sum_is_3972 :
  ∀ perm ∈ generatePermutations,
    let (n1, n2, n3) := splitIntoThreeNumbers perm
    sumThreeNumbers (n1, n2, n3) ≤ maxSum :=
by sorry

end max_sum_is_3972_l2257_225778


namespace geese_survival_theorem_l2257_225720

/-- Represents the number of geese that survived the first year given the total number of eggs laid -/
def geese_survived_first_year (total_eggs : ℕ) : ℕ :=
  let hatched_eggs := (2 * total_eggs) / 3
  let survived_first_month := (3 * hatched_eggs) / 4
  let not_survived_first_year := (3 * survived_first_month) / 5
  survived_first_month - not_survived_first_year

/-- Theorem stating that the number of geese surviving the first year is 1/5 of the total eggs laid -/
theorem geese_survival_theorem (total_eggs : ℕ) :
  geese_survived_first_year total_eggs = total_eggs / 5 := by
  sorry

#eval geese_survived_first_year 60  -- Should output 12

end geese_survival_theorem_l2257_225720


namespace smallest_m_for_multiple_factorizations_l2257_225739

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def has_multiple_factorizations (n : ℕ) : Prop :=
  ∃ (f1 f2 : List ℕ), 
    f1 ≠ f2 ∧ 
    f1.length = 16 ∧ 
    f2.length = 16 ∧ 
    f1.Nodup ∧ 
    f2.Nodup ∧ 
    f1.prod = n ∧ 
    f2.prod = n

theorem smallest_m_for_multiple_factorizations :
  (∀ m : ℕ, m > 0 ∧ m < 24 → ¬has_multiple_factorizations (factorial 15 * m)) ∧
  has_multiple_factorizations (factorial 15 * 24) :=
sorry

end smallest_m_for_multiple_factorizations_l2257_225739


namespace no_three_numbers_exist_l2257_225770

theorem no_three_numbers_exist : ¬∃ (a b c : ℕ), 
  (a > 1 ∧ b > 1 ∧ c > 1) ∧ 
  ((∃ k : ℕ, a^2 - 1 = b * k ∨ a^2 - 1 = c * k) ∧
   (∃ l : ℕ, b^2 - 1 = a * l ∨ b^2 - 1 = c * l) ∧
   (∃ m : ℕ, c^2 - 1 = a * m ∨ c^2 - 1 = b * m)) :=
by sorry

end no_three_numbers_exist_l2257_225770


namespace parabola_intersection_ratio_l2257_225776

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def focus : Point := ⟨1, 0⟩

/-- The fixed point A -/
def A : Point := ⟨0, -2⟩

/-- Point M on the parabola -/
def M : Point := sorry

/-- Point N on the directrix -/
def N : Point := sorry

/-- Distance between two points -/
def distance (p q : Point) : ℝ := sorry

/-- Theorem: The ratio |MN| : |FN| = √5 : (1 + √5) -/
theorem parabola_intersection_ratio :
  (distance M N) / (distance focus N) = Real.sqrt 5 / (1 + Real.sqrt 5) := by
  sorry

end parabola_intersection_ratio_l2257_225776


namespace min_value_of_a_plus_2b_l2257_225750

theorem min_value_of_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (2 / a + 1 / b = 1) → (∀ a' b' : ℝ, a' > 0 → b' > 0 → 2 / a' + 1 / b' = 1 → a + 2*b ≤ a' + 2*b') → a + 2*b = 4 :=
by sorry

end min_value_of_a_plus_2b_l2257_225750


namespace marble_selection_problem_l2257_225796

theorem marble_selection_problem (n : ℕ) (k : ℕ) (total : ℕ) (red : ℕ) :
  n = 10 →
  k = 4 →
  total = Nat.choose n k →
  red = Nat.choose (n - 1) k →
  total - red = 84 :=
by
  sorry

end marble_selection_problem_l2257_225796


namespace quadratic_inequality_condition_l2257_225716

theorem quadratic_inequality_condition (x : ℝ) : 
  (((x < 1) ∨ (x > 4)) → (x^2 - 3*x + 2 > 0)) ∧ 
  (∃ x, (x^2 - 3*x + 2 > 0) ∧ ¬((x < 1) ∨ (x > 4))) :=
by sorry

end quadratic_inequality_condition_l2257_225716


namespace vector_equality_l2257_225703

def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)
def c : ℝ × ℝ := (-2, 4)

theorem vector_equality : c = a - 3 • b := by sorry

end vector_equality_l2257_225703


namespace x_plus_y_value_l2257_225749

theorem x_plus_y_value (x y : ℝ) 
  (h1 : x + Real.cos y = 2023)
  (h2 : x + 2023 * Real.sin y = 2022)
  (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2022 := by
  sorry

end x_plus_y_value_l2257_225749


namespace toms_age_ratio_l2257_225743

theorem toms_age_ratio (T N : ℕ) : T > 0 → N > 0 → T = 7 * N := by
  sorry

#check toms_age_ratio

end toms_age_ratio_l2257_225743


namespace rectangle_area_after_length_decrease_l2257_225738

theorem rectangle_area_after_length_decrease (square_area : ℝ) 
  (rectangle_length_decrease_percent : ℝ) : 
  square_area = 49 →
  rectangle_length_decrease_percent = 20 →
  let square_side := Real.sqrt square_area
  let initial_rectangle_length := square_side
  let initial_rectangle_width := 2 * square_side
  let new_rectangle_length := initial_rectangle_length * (1 - rectangle_length_decrease_percent / 100)
  let new_rectangle_width := initial_rectangle_width
  new_rectangle_length * new_rectangle_width = 78.4 := by
sorry

end rectangle_area_after_length_decrease_l2257_225738


namespace problem_solution_l2257_225789

theorem problem_solution : 
  (0.064 ^ (-(1/3 : ℝ)) - (-(1/8 : ℝ))^0 + 16^(3/4 : ℝ) + 0.25^(1/2 : ℝ) = 10) ∧
  ((2 * Real.log 2 + Real.log 3) / (1 + (1/2 : ℝ) * Real.log 0.36 + (1/3 : ℝ) * Real.log 8) = 1) := by
  sorry


end problem_solution_l2257_225789


namespace euler_minus_i_pi_l2257_225759

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Complex.cos z.im + Complex.I * Complex.sin z.im)

-- State Euler's formula
axiom euler_formula (x : ℝ) : cexp (Complex.I * x) = Complex.cos x + Complex.I * Complex.sin x

-- Theorem to prove
theorem euler_minus_i_pi : cexp (-Complex.I * Real.pi) = -1 := by sorry

end euler_minus_i_pi_l2257_225759


namespace three_tangent_lines_l2257_225757

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A parabola in the 2D plane represented by its equation y^2 = ax -/
structure Parabola where
  a : ℝ

/-- Predicate to check if a line passes through a given point -/
def Line.passesThrough (l : Line) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

/-- Predicate to check if a line has only one common point with a parabola -/
def Line.hasOnlyOneCommonPoint (l : Line) (p : Parabola) : Prop :=
  ∃! x y, l.passesThrough x y ∧ y^2 = p.a * x

/-- The main theorem stating that there are exactly 3 lines passing through (0,6)
    and having only one common point with the parabola y^2 = -12x -/
theorem three_tangent_lines :
  ∃! (lines : Finset Line),
    (∀ l ∈ lines, l.passesThrough 0 6 ∧ l.hasOnlyOneCommonPoint (Parabola.mk (-12))) ∧
    lines.card = 3 := by
  sorry

end three_tangent_lines_l2257_225757


namespace width_to_perimeter_ratio_l2257_225775

/-- The ratio of width to perimeter for a rectangular room -/
theorem width_to_perimeter_ratio (length width : ℝ) (h1 : length = 15) (h2 : width = 13) :
  width / (2 * (length + width)) = 13 / 56 := by
  sorry

end width_to_perimeter_ratio_l2257_225775


namespace jessica_bank_balance_l2257_225790

/-- Calculates the final balance in Jessica's bank account after withdrawing $400 and depositing 1/4 of the remaining balance. -/
theorem jessica_bank_balance (B : ℝ) (h : 2 / 5 * B = 400) : 
  (B - 400) + (1 / 4 * (B - 400)) = 750 := by
  sorry

#check jessica_bank_balance

end jessica_bank_balance_l2257_225790


namespace smallest_solution_equation_smallest_solution_is_four_minus_sqrt_two_l2257_225785

theorem smallest_solution_equation (x : ℝ) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ↔ (x = 4 - Real.sqrt 2 ∨ x = 4 + Real.sqrt 2) :=
sorry

theorem smallest_solution_is_four_minus_sqrt_two :
  ∃ (x : ℝ), (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ∧
  (∀ (y : ℝ), (1 / (y - 3) + 1 / (y - 5) = 4 / (y - 4)) → y ≥ x) ∧
  x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_equation_smallest_solution_is_four_minus_sqrt_two_l2257_225785


namespace prob_red_third_eq_147_1000_l2257_225791

/-- A fair 10-sided die with exactly 3 red sides -/
structure RedDie :=
  (sides : Nat)
  (red_sides : Nat)
  (h_sides : sides = 10)
  (h_red : red_sides = 3)

/-- The probability of rolling a non-red side -/
def prob_non_red (d : RedDie) : ℚ :=
  (d.sides - d.red_sides : ℚ) / d.sides

/-- The probability of rolling a red side -/
def prob_red (d : RedDie) : ℚ :=
  d.red_sides / d.sides

/-- The probability of rolling a red side for the first time on the third roll -/
def prob_red_third (d : RedDie) : ℚ :=
  (prob_non_red d) * (prob_non_red d) * (prob_red d)

theorem prob_red_third_eq_147_1000 (d : RedDie) : 
  prob_red_third d = 147 / 1000 := by sorry

end prob_red_third_eq_147_1000_l2257_225791


namespace k_range_l2257_225752

/-- Piecewise function f(x) -/
noncomputable def f (k a x : ℝ) : ℝ :=
  if x ≥ 0 then k^2 * x + a^2 - k
  else x^2 + (a^2 + 4*a) * x + (2-a)^2

/-- Condition for the existence of a unique nonzero x₂ for any nonzero x₁ -/
def unique_nonzero_solution (k a : ℝ) : Prop :=
  ∀ x₁ : ℝ, x₁ ≠ 0 → ∃! x₂ : ℝ, x₂ ≠ 0 ∧ x₂ ≠ x₁ ∧ f k a x₂ = f k a x₁

theorem k_range (k a : ℝ) :
  unique_nonzero_solution k a → k ∈ Set.Icc (-20) (-4) :=
by sorry

end k_range_l2257_225752


namespace range_of_p_characterization_l2257_225730

def A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
def B (p : ℝ) : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}

def range_of_p : Set ℝ := {p | B p ⊆ A}

theorem range_of_p_characterization :
  range_of_p = 
    {p | B p = ∅} ∪ 
    {p | B p ≠ ∅ ∧ ∀ x ∈ B p, x ∈ A} :=
by sorry

end range_of_p_characterization_l2257_225730


namespace minimum_economic_loss_l2257_225746

def repair_times : List Nat := [12, 17, 8, 18, 23, 30, 14]
def num_workers : Nat := 3
def loss_per_minute : Nat := 2

def optimal_allocation (times : List Nat) (workers : Nat) : List (List Nat) :=
  sorry

def total_waiting_time (allocation : List (List Nat)) : Nat :=
  sorry

theorem minimum_economic_loss :
  let allocation := optimal_allocation repair_times num_workers
  let total_wait := total_waiting_time allocation
  total_wait * loss_per_minute = 358 := by
  sorry

end minimum_economic_loss_l2257_225746


namespace problem_statement_l2257_225764

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + a*x - 3

theorem problem_statement :
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → f x ≥ (1/2) * g a x) → a ≤ 4) ∧
  (∀ x : ℝ, x > 0 → log x > 1/exp x - 2/(exp 1 * x)) := by sorry

end problem_statement_l2257_225764


namespace jelly_bean_distribution_l2257_225702

theorem jelly_bean_distribution (n : ℕ) (h1 : 10 ≤ n) (h2 : n ≤ 20) : 
  (∃ (total : ℕ), total = n^2 ∧ total % 5 = 0) → 
  (∃ (per_bag : ℕ), per_bag = 45 ∧ 5 * per_bag = n^2) := by
sorry

end jelly_bean_distribution_l2257_225702


namespace breath_holding_improvement_l2257_225705

theorem breath_holding_improvement (initial_time : ℝ) : 
  initial_time = 10 → 
  (((initial_time * 2) * 2) * 1.5) = 60 := by
sorry

end breath_holding_improvement_l2257_225705


namespace quadratic_root_sqrt5_minus3_l2257_225712

theorem quadratic_root_sqrt5_minus3 : ∃ (a b c : ℚ), 
  a = 1 ∧ (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3) :=
by sorry

end quadratic_root_sqrt5_minus3_l2257_225712


namespace line_properties_l2257_225715

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y + 3 * a = 0
def l₂ (a x y : ℝ) : Prop := 3 * x + (a - 1) * y + 3 - a = 0

-- Define perpendicularity of two lines
def perpendicular (a : ℝ) : Prop := (-a / 2) * (-3 / (a - 1)) = -1

theorem line_properties (a : ℝ) :
  (l₂ a (-2/3) 1) ∧
  (perpendicular a → a = 2/5) := by sorry

end line_properties_l2257_225715
