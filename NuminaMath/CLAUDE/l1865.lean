import Mathlib

namespace system_solution_l1865_186534

theorem system_solution :
  let S := {(x, y) : ℝ × ℝ | x^2 - 9*y^2 = 0 ∧ x^2 + y^2 = 9}
  S = {(9/Real.sqrt 10, 3/Real.sqrt 10), (-9/Real.sqrt 10, 3/Real.sqrt 10),
       (9/Real.sqrt 10, -3/Real.sqrt 10), (-9/Real.sqrt 10, -3/Real.sqrt 10)} :=
by sorry

end system_solution_l1865_186534


namespace divisibility_by_eleven_l1865_186572

def seven_digit_number (n : ℕ) : ℕ := 854 * 10000 + n * 1000 + 526

theorem divisibility_by_eleven (n : ℕ) : 
  (seven_digit_number n) % 11 = 0 ↔ n = 5 := by
  sorry

end divisibility_by_eleven_l1865_186572


namespace expression_value_l1865_186581

theorem expression_value (x y : ℝ) (hx : x = 8) (hy : y = 3) :
  (x - 2*y) * (x + 2*y) = 28 := by
  sorry

end expression_value_l1865_186581


namespace brad_carl_weight_difference_l1865_186541

/-- Given the weights of Billy, Brad, and Carl, prove that Brad weighs 5 pounds more than Carl. -/
theorem brad_carl_weight_difference
  (billy_weight : ℕ)
  (brad_weight : ℕ)
  (carl_weight : ℕ)
  (h1 : billy_weight = brad_weight + 9)
  (h2 : brad_weight > carl_weight)
  (h3 : carl_weight = 145)
  (h4 : billy_weight = 159) :
  brad_weight - carl_weight = 5 := by
  sorry

end brad_carl_weight_difference_l1865_186541


namespace complex_sixth_power_equation_simplified_polynomial_system_l1865_186532

/-- The complex number z satisfying z^6 = -8 - 8i can be characterized by a system of polynomial equations. -/
theorem complex_sixth_power_equation (z : ℂ) : 
  z^6 = -8 - 8*I ↔ 
  ∃ (x y : ℝ), z = x + y*I ∧ 
    (x^6 - 15*x^4*y^2 + 15*x^2*y^4 - y^6 = -8) ∧
    (6*x^5*y - 20*x^3*y^3 + 6*x*y^5 = -8) :=
by sorry

/-- The system of polynomial equations characterizing the solutions can be further simplified. -/
theorem simplified_polynomial_system (x y : ℝ) :
  (x^6 - 15*x^4*y^2 + 15*x^2*y^4 - y^6 = -8 ∧ 
   6*x^5*y - 20*x^3*y^3 + 6*x*y^5 = -8) ↔
  (x^6 - 15*x^4*y^2 + 15*x^2*y^4 - y^6 = -8 ∧
   x^4 - 10*x^2*y^2 + y^4 = -4/3) :=
by sorry

end complex_sixth_power_equation_simplified_polynomial_system_l1865_186532


namespace max_average_profit_l1865_186593

def profit (t : ℕ+) : ℚ := -2 * (t : ℚ)^2 + 30 * (t : ℚ) - 98

def average_profit (t : ℕ+) : ℚ := (profit t) / (t : ℚ)

theorem max_average_profit :
  ∃ (t : ℕ+), ∀ (k : ℕ+), average_profit t ≥ average_profit k ∧ t = 7 :=
sorry

end max_average_profit_l1865_186593


namespace regular_triangular_pyramid_l1865_186584

theorem regular_triangular_pyramid (a : ℝ) : 
  (∃ h : ℝ, h = (a * Real.sqrt 3) / 3) → -- height in terms of base side
  (∃ V : ℝ, V = (1 / 3) * ((a^2 * Real.sqrt 3) / 4) * ((a * Real.sqrt 3) / 3)) → -- volume formula
  V = 18 → -- given volume
  a = 6 := by sorry

end regular_triangular_pyramid_l1865_186584


namespace polyhedron_exists_l1865_186500

-- Define a custom type for vertices
inductive Vertex : Type
  | A | B | C | D | E | F | G | H

-- Define an edge as a pair of vertices
def Edge : Type := Vertex × Vertex

-- Define the list of edges
def edgeList : List Edge :=
  [(Vertex.A, Vertex.B), (Vertex.A, Vertex.C), (Vertex.B, Vertex.C),
   (Vertex.B, Vertex.D), (Vertex.C, Vertex.D), (Vertex.D, Vertex.E),
   (Vertex.E, Vertex.F), (Vertex.E, Vertex.G), (Vertex.F, Vertex.G),
   (Vertex.F, Vertex.H), (Vertex.G, Vertex.H), (Vertex.A, Vertex.H)]

-- Define a polyhedron as a list of edges
def Polyhedron : Type := List Edge

-- Theorem: There exists a polyhedron with the given list of edges
theorem polyhedron_exists : ∃ (p : Polyhedron), p = edgeList := by
  sorry

end polyhedron_exists_l1865_186500


namespace no_integer_solutions_l1865_186502

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^4 + y^2 = 3*y + 3 := by sorry

end no_integer_solutions_l1865_186502


namespace x_gt_one_sufficient_not_necessary_for_reciprocal_lt_one_l1865_186578

theorem x_gt_one_sufficient_not_necessary_for_reciprocal_lt_one :
  (∀ x : ℝ, x > 1 → 1 / x < 1) ∧
  (∃ x : ℝ, 1 / x < 1 ∧ x ≤ 1) :=
by sorry

end x_gt_one_sufficient_not_necessary_for_reciprocal_lt_one_l1865_186578


namespace arithmetic_sequence_properties_l1865_186571

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_property : a 1 + a 2 + a 3 = 21
  product_property : a 1 * a 2 * a 3 = 231

/-- Theorem about the second term and general formula of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (seq.a 2 = 7) ∧
  ((∀ n, seq.a n = -4 * n + 15) ∨ (∀ n, seq.a n = 4 * n - 1)) :=
by sorry

end arithmetic_sequence_properties_l1865_186571


namespace car_distance_problem_l1865_186547

/-- Calculates the distance between two cars given their speeds and overtake time -/
def distance_between_cars (red_speed black_speed overtake_time : ℝ) : ℝ :=
  (black_speed - red_speed) * overtake_time

/-- Theorem stating that the distance between the cars is 30 miles -/
theorem car_distance_problem :
  let red_speed : ℝ := 40
  let black_speed : ℝ := 50
  let overtake_time : ℝ := 3
  distance_between_cars red_speed black_speed overtake_time = 30 := by
sorry


end car_distance_problem_l1865_186547


namespace circle_equation_l1865_186557

/-- Given a circle with center (2, -1) and a chord of length 2√2 intercepted by the line x - y - 1 = 0,
    prove that the equation of the circle is (x-2)² + (y+1)² = 4 -/
theorem circle_equation (x y : ℝ) : 
  let center := (2, -1)
  let line := {(x, y) : ℝ × ℝ | x - y - 1 = 0}
  let chord_length := 2 * Real.sqrt 2
  true → (x - 2)^2 + (y + 1)^2 = 4 :=
by
  sorry


end circle_equation_l1865_186557


namespace discount_calculation_l1865_186587

-- Define the number of pens bought and the equivalent marked price
def pens_bought : ℕ := 50
def marked_price_equivalent : ℕ := 46

-- Define the profit percentage
def profit_percent : ℚ := 7608695652173914 / 100000000000000000

-- Define the discount percentage (to be proven)
def discount_percent : ℚ := 1 / 100

theorem discount_calculation :
  let cost_price := marked_price_equivalent
  let selling_price := cost_price * (1 + profit_percent)
  let discount := pens_bought - selling_price
  discount / pens_bought = discount_percent := by sorry

end discount_calculation_l1865_186587


namespace paint_per_statue_l1865_186561

-- Define the total amount of paint
def total_paint : ℚ := 1/2

-- Define the number of statues that can be painted
def num_statues : ℕ := 2

-- Theorem: Each statue requires 1/4 gallon of paint
theorem paint_per_statue : total_paint / num_statues = 1/4 := by
  sorry

end paint_per_statue_l1865_186561


namespace binomial_60_3_l1865_186577

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l1865_186577


namespace arithmetic_geometric_mean_equation_l1865_186509

theorem arithmetic_geometric_mean_equation (α β : ℝ) :
  (α + β) / 2 = 8 →
  Real.sqrt (α * β) = 15 →
  (∀ x : ℝ, x^2 - 16*x + 225 = 0 ↔ x = α ∨ x = β) :=
by sorry

end arithmetic_geometric_mean_equation_l1865_186509


namespace indefinite_integral_proof_l1865_186543

theorem indefinite_integral_proof (x : ℝ) : 
  (deriv (λ x => -1/4 * (7*x - 10) * Real.cos (4*x) - 7/16 * Real.sin (4*x))) x = 
  (7*x - 10) * Real.sin (4*x) := by
  sorry

end indefinite_integral_proof_l1865_186543


namespace quadratic_sum_l1865_186563

/-- Given a quadratic polynomial 6x^2 + 36x + 216, when expressed in the form a(x + b)^2 + c,
    where a, b, and c are constants, prove that a + b + c = 171. -/
theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, 6 * x^2 + 36 * x + 216 = a * (x + b)^2 + c) → a + b + c = 171 := by
  sorry

end quadratic_sum_l1865_186563


namespace yoojung_notebooks_l1865_186579

theorem yoojung_notebooks (initial : ℕ) : 
  (initial ≥ 5) →                        -- Ensure initial is at least 5
  (((initial - 5) / 2 : ℚ) = 4) →        -- Half of remaining after giving 5 equals 4
  (initial = 13) :=                      -- Prove initial is 13
sorry

end yoojung_notebooks_l1865_186579


namespace range_of_m_for_positive_functions_l1865_186592

theorem range_of_m_for_positive_functions (m : ℝ) : 
  (∀ x : ℝ, (2 * m * x^2 - 2 * m * x - 8 * x + 9 > 0) ∨ (m * x - m > 0)) →
  (0 < m ∧ m < 8) := by
sorry

end range_of_m_for_positive_functions_l1865_186592


namespace unique_solution_l1865_186506

def A (p : ℝ) : Set ℝ := {x | x^2 - p*x - 2 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 + q*x + r = 0}

theorem unique_solution :
  ∃! (p q r : ℝ),
    (A p ∪ B q r = {-2, 1, 7}) ∧
    (A p ∩ B q r = {-2}) ∧
    p = -1 ∧ q = -5 ∧ r = -14 := by
  sorry

end unique_solution_l1865_186506


namespace sphere_surface_area_l1865_186564

theorem sphere_surface_area (V : Real) (r : Real) : 
  V = (4 / 3) * Real.pi * r^3 → 
  V = 36 * Real.pi → 
  4 * Real.pi * r^2 = 36 * Real.pi := by
sorry

end sphere_surface_area_l1865_186564


namespace school_girls_count_l1865_186576

theorem school_girls_count (boys : ℕ) (girls_boys_diff : ℕ) : 
  boys = 469 → girls_boys_diff = 228 → boys + girls_boys_diff = 697 := by
  sorry

end school_girls_count_l1865_186576


namespace christel_gave_five_dolls_l1865_186583

/-- The number of dolls Christel gave to Andrena -/
def dolls_given_by_christel : ℕ := sorry

theorem christel_gave_five_dolls :
  let debelyn_initial := 20
  let debelyn_gave := 2
  let christel_initial := 24
  let andrena_more_than_christel := 2
  let andrena_more_than_debelyn := 3
  dolls_given_by_christel = 5 := by sorry

end christel_gave_five_dolls_l1865_186583


namespace sum_of_basic_terms_divisible_by_four_l1865_186599

/-- A type representing a grid cell that can be either +1 or -1 -/
inductive GridCell
  | pos : GridCell
  | neg : GridCell

/-- A type representing an n × n grid filled with +1 or -1 -/
def Grid (n : ℕ) := Fin n → Fin n → GridCell

/-- A basic term is a product of n cells, no two of which share the same row or column -/
def BasicTerm (n : ℕ) (grid : Grid n) (perm : Equiv.Perm (Fin n)) : ℤ :=
  (Finset.univ.prod fun i => match grid i (perm i) with
    | GridCell.pos => 1
    | GridCell.neg => -1)

/-- The sum of all basic terms for a given grid -/
def SumOfBasicTerms (n : ℕ) (grid : Grid n) : ℤ :=
  (Finset.univ : Finset (Equiv.Perm (Fin n))).sum fun perm => BasicTerm n grid perm

/-- The main theorem: for any n × n grid (n ≥ 4), the sum of all basic terms is divisible by 4 -/
theorem sum_of_basic_terms_divisible_by_four {n : ℕ} (h : n ≥ 4) (grid : Grid n) :
  4 ∣ SumOfBasicTerms n grid := by
  sorry

end sum_of_basic_terms_divisible_by_four_l1865_186599


namespace rope_cutting_problem_l1865_186586

theorem rope_cutting_problem (a b c : ℕ) (ha : a = 39) (hb : b = 52) (hc : c = 65) :
  Nat.gcd a (Nat.gcd b c) = 13 := by
  sorry

end rope_cutting_problem_l1865_186586


namespace amount_distribution_l1865_186540

theorem amount_distribution (A : ℕ) : 
  (A / 14 = A / 18 + 80) → A = 5040 :=
by
  sorry

end amount_distribution_l1865_186540


namespace participation_plans_eq_48_l1865_186580

/-- The number of different participation plans for selecting 3 out of 5 students
    for math, physics, and chemistry competitions, where each student competes
    in one subject and student A cannot participate in the physics competition. -/
def participation_plans : ℕ :=
  let total_students : ℕ := 5
  let selected_students : ℕ := 3
  let competitions : ℕ := 3
  let student_a_options : ℕ := 2  -- math or chemistry

  let scenario1 : ℕ := (total_students - 1).factorial / (total_students - 1 - selected_students).factorial
  let scenario2 : ℕ := student_a_options * ((total_students - 1).factorial / (total_students - 1 - (selected_students - 1)).factorial)

  scenario1 + scenario2

theorem participation_plans_eq_48 :
  participation_plans = 48 := by
  sorry

end participation_plans_eq_48_l1865_186580


namespace decimal_multiplication_addition_l1865_186575

theorem decimal_multiplication_addition : 0.45 * 0.65 + 0.1 * 0.2 = 0.3125 := by
  sorry

end decimal_multiplication_addition_l1865_186575


namespace calories_burned_per_mile_l1865_186596

/-- Represents the calories burned per mile walked -/
def calories_per_mile : ℝ := sorry

/-- The total distance walked in miles -/
def total_distance : ℝ := 3

/-- The calories in the candy bar -/
def candy_bar_calories : ℝ := 200

/-- The net calorie deficit -/
def net_deficit : ℝ := 250

theorem calories_burned_per_mile :
  calories_per_mile * total_distance - candy_bar_calories = net_deficit ∧
  calories_per_mile = 150 := by sorry

end calories_burned_per_mile_l1865_186596


namespace equation_solution_l1865_186549

theorem equation_solution : ∃! x : ℝ, (1 / (x + 12) + 1 / (x + 10) = 1 / (x + 13) + 1 / (x + 9)) ∧ x = -11 := by
  sorry

end equation_solution_l1865_186549


namespace probability_three_ones_or_twos_in_five_rolls_l1865_186570

-- Define the probability of rolling a 1 or 2 on a fair six-sided die
def prob_one_or_two : ℚ := 1 / 3

-- Define the probability of not rolling a 1 or 2 on a fair six-sided die
def prob_not_one_or_two : ℚ := 2 / 3

-- Define the number of rolls
def num_rolls : ℕ := 5

-- Define the number of times we want to roll a 1 or 2
def target_rolls : ℕ := 3

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem probability_three_ones_or_twos_in_five_rolls :
  (binomial num_rolls target_rolls : ℚ) * prob_one_or_two ^ target_rolls * prob_not_one_or_two ^ (num_rolls - target_rolls) = 40 / 243 := by
  sorry

end probability_three_ones_or_twos_in_five_rolls_l1865_186570


namespace sphere_volume_diameter_6_l1865_186552

/-- The volume of a sphere with diameter 6 is 36π. -/
theorem sphere_volume_diameter_6 : 
  let d : ℝ := 6
  let r : ℝ := d / 2
  let V : ℝ := (4 / 3) * Real.pi * r ^ 3
  V = 36 * Real.pi := by sorry

end sphere_volume_diameter_6_l1865_186552


namespace complex_fraction_equality_l1865_186594

theorem complex_fraction_equality : (4 - 2*I) / (1 + I)^2 = -1 - 2*I := by
  sorry

end complex_fraction_equality_l1865_186594


namespace customers_who_tipped_l1865_186542

theorem customers_who_tipped (initial_customers : ℕ) (additional_customers : ℕ) (non_tipping_customers : ℕ)
  (h1 : initial_customers = 29)
  (h2 : additional_customers = 20)
  (h3 : non_tipping_customers = 34) :
  initial_customers + additional_customers - non_tipping_customers = 15 :=
by sorry

end customers_who_tipped_l1865_186542


namespace mac_total_loss_l1865_186517

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "penny" => 1
  | "nickel" => 5
  | "dime" => 10
  | "quarter" => 25
  | "half-dollar" => 50
  | _ => 0

/-- Calculates the expected loss for a single trade -/
def expected_loss (given_coins : List String) (probability : ℚ) : ℚ :=
  let given_value : ℚ := (given_coins.map coin_value).sum
  let quarter_value : ℚ := coin_value "quarter"
  (given_value - quarter_value) * probability

/-- Represents Mac's trading scenario -/
def mac_trades : List (List String × ℚ × ℕ) := [
  (["dime", "dime", "dime", "dime", "penny", "penny"], 1/20, 20),
  (["nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "nickel", "penny"], 1/10, 20),
  (["half-dollar", "penny", "penny", "penny"], 17/20, 20)
]

/-- Theorem stating the total expected loss for Mac's trades -/
theorem mac_total_loss :
  (mac_trades.map (λ (coins, prob, repeats) => expected_loss coins prob * repeats)).sum = 535/100 := by
  sorry


end mac_total_loss_l1865_186517


namespace ping_pong_ball_price_l1865_186544

theorem ping_pong_ball_price 
  (quantity : ℕ) 
  (discount_rate : ℚ) 
  (total_paid : ℚ) 
  (h1 : quantity = 10000)
  (h2 : discount_rate = 30 / 100)
  (h3 : total_paid = 700) :
  let original_price := total_paid / ((1 - discount_rate) * quantity)
  original_price = 1 / 10 := by
sorry

end ping_pong_ball_price_l1865_186544


namespace largest_integer_l1865_186508

theorem largest_integer (x y z w : ℤ) 
  (sum1 : x + y + z = 234)
  (sum2 : x + y + w = 255)
  (sum3 : x + z + w = 271)
  (sum4 : y + z + w = 198) :
  max x (max y (max z w)) = 121 := by
  sorry

end largest_integer_l1865_186508


namespace asterisk_replacement_l1865_186582

theorem asterisk_replacement : ∃! (x : ℝ), x > 0 ∧ (x / 21) * (x / 84) = 1 := by
  sorry

end asterisk_replacement_l1865_186582


namespace right_triangle_and_inverse_l1865_186556

theorem right_triangle_and_inverse (a b c : Nat) (m : Nat) : 
  a = 48 → b = 55 → c = 73 → m = 4273 →
  a * a + b * b = c * c →
  (∃ (x : Nat), x * 480 ≡ 1 [MOD m]) →
  (∃ (y : Nat), y * 480 ≡ 1643 [MOD m]) :=
by sorry

end right_triangle_and_inverse_l1865_186556


namespace choose_cooks_count_l1865_186514

def total_people : ℕ := 10
def cooks_needed : ℕ := 3

theorem choose_cooks_count : Nat.choose total_people cooks_needed = 120 := by
  sorry

end choose_cooks_count_l1865_186514


namespace triangles_in_200_sided_polygon_l1865_186510

/-- The number of sides in the regular polygon -/
def n : ℕ := 200

/-- The number of vertices to select for each triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed from a regular n-sided polygon -/
def num_triangles (n : ℕ) : ℕ := Nat.choose n k

theorem triangles_in_200_sided_polygon :
  num_triangles n = 1313400 := by
  sorry

end triangles_in_200_sided_polygon_l1865_186510


namespace alice_burger_expense_l1865_186591

/-- The amount Alice spent on burgers in June -/
def aliceSpentOnBurgers (daysInJune : ℕ) (burgersPerDay : ℕ) (costPerBurger : ℕ) : ℕ :=
  daysInJune * burgersPerDay * costPerBurger

/-- Proof that Alice spent $1560 on burgers in June -/
theorem alice_burger_expense :
  aliceSpentOnBurgers 30 4 13 = 1560 := by
  sorry

end alice_burger_expense_l1865_186591


namespace city_council_vote_change_l1865_186565

theorem city_council_vote_change :
  ∀ (x y x' y' : ℕ),
    x + y = 500 →
    y > x →
    x' + y' = 500 →
    x' - y' = (3 * (y - x)) / 2 →
    x' = (13 * y) / 12 →
    x' - x = 125 :=
by sorry

end city_council_vote_change_l1865_186565


namespace binomial_coefficient_equality_l1865_186528

theorem binomial_coefficient_equality (n : ℕ) : 
  (∃ k : ℕ, k ∈ Finset.range (n - 1) ∧ 
    2 * Nat.choose n k = Nat.choose n (k - 1) + Nat.choose n (k + 1)) ↔ 
  (∃ m : ℕ, m ≥ 3 ∧ n = m^2 - 2) :=
sorry

end binomial_coefficient_equality_l1865_186528


namespace movie_theater_tickets_l1865_186527

theorem movie_theater_tickets (matinee_price evening_price threeD_price : ℕ)
  (evening_sold threeD_sold total_revenue : ℕ) :
  matinee_price = 5 →
  evening_price = 12 →
  threeD_price = 20 →
  evening_sold = 300 →
  threeD_sold = 100 →
  total_revenue = 6600 →
  ∃ (matinee_sold : ℕ), 
    matinee_sold * matinee_price + 
    evening_sold * evening_price + 
    threeD_sold * threeD_price = total_revenue ∧
    matinee_sold = 200 :=
by sorry

end movie_theater_tickets_l1865_186527


namespace square_area_proof_l1865_186525

theorem square_area_proof (x : ℝ) :
  (5 * x - 18 = 25 - 2 * x) →
  (5 * x - 18 ≥ 0) →
  ((5 * x - 18)^2 : ℝ) = 7921 / 49 := by
  sorry

end square_area_proof_l1865_186525


namespace impossibility_of_circular_arrangement_l1865_186505

theorem impossibility_of_circular_arrangement : ¬ ∃ (arrangement : Fin 1995 → ℕ), 
  (∀ i j : Fin 1995, i ≠ j → arrangement i ≠ arrangement j) ∧ 
  (∀ i : Fin 1995, Nat.Prime ((max (arrangement i) (arrangement (i + 1))) / 
                               (min (arrangement i) (arrangement (i + 1))))) :=
sorry

end impossibility_of_circular_arrangement_l1865_186505


namespace x_value_l1865_186519

theorem x_value (w y z x : ℤ) 
  (hw : w = 95)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 7) : 
  x = 142 := by sorry

end x_value_l1865_186519


namespace largest_number_l1865_186555

def a : ℚ := 24680 + 1 / 13579
def b : ℚ := 24680 - 1 / 13579
def c : ℚ := 24680 * (1 / 13579)
def d : ℚ := 24680 / (1 / 13579)
def e : ℚ := 24680.13579

theorem largest_number : d > a ∧ d > b ∧ d > c ∧ d > e := by
  sorry

end largest_number_l1865_186555


namespace jayden_half_ernesto_age_l1865_186568

/-- 
Given:
- Ernesto is currently 11 years old
- Jayden is currently 4 years old

Prove that in 3 years, Jayden will be half of Ernesto's age
-/
theorem jayden_half_ernesto_age :
  let ernesto_age : ℕ := 11
  let jayden_age : ℕ := 4
  let years_until_half : ℕ := 3
  (jayden_age + years_until_half : ℚ) = (1/2 : ℚ) * (ernesto_age + years_until_half : ℚ) :=
by sorry

end jayden_half_ernesto_age_l1865_186568


namespace box_max_volume_l1865_186503

/-- Volume function for the box -/
def V (x : ℝ) : ℝ := (16 - 2*x) * (10 - 2*x) * x

/-- The theorem stating the maximum volume and corresponding height -/
theorem box_max_volume :
  ∃ (max_vol : ℝ) (max_height : ℝ),
    (∀ x, 0 < x → x < 5 → V x ≤ max_vol) ∧
    (0 < max_height ∧ max_height < 5) ∧
    (V max_height = max_vol) ∧
    (max_height = 2) ∧
    (max_vol = 144) := by
  sorry

end box_max_volume_l1865_186503


namespace girls_fraction_is_half_l1865_186569

/-- Represents a school with a given number of students and boy-to-girl ratio -/
structure School where
  total_students : ℕ
  boy_ratio : ℕ
  girl_ratio : ℕ

/-- Calculates the number of girls in a school -/
def girls_count (s : School) : ℕ :=
  s.total_students * s.girl_ratio /(s.boy_ratio + s.girl_ratio)

/-- The fraction of girls at a dance attended by students from two schools -/
def girls_fraction (school_a : School) (school_b : School) : ℚ :=
  (girls_count school_a + girls_count school_b : ℚ) /
  (school_a.total_students + school_b.total_students)

theorem girls_fraction_is_half :
  let school_a : School := ⟨300, 3, 2⟩
  let school_b : School := ⟨240, 3, 5⟩
  girls_fraction school_a school_b = 1/2 := by
  sorry

end girls_fraction_is_half_l1865_186569


namespace geometric_sequence_fourth_term_l1865_186504

theorem geometric_sequence_fourth_term (a b c x : ℝ) : 
  a ≠ 0 → b / a = c / b → c / b * c = x → a = 0.001 → b = 0.02 → c = 0.4 → x = 8 := by
  sorry

end geometric_sequence_fourth_term_l1865_186504


namespace flour_bag_weight_l1865_186535

/-- Calculates the weight of each bag of flour given the problem conditions --/
theorem flour_bag_weight 
  (flour_needed : ℕ) 
  (bag_cost : ℚ) 
  (salt_needed : ℕ) 
  (salt_cost_per_pound : ℚ) 
  (promotion_cost : ℕ) 
  (ticket_price : ℕ) 
  (tickets_sold : ℕ) 
  (total_profit : ℚ) 
  (h1 : flour_needed = 500) 
  (h2 : bag_cost = 20) 
  (h3 : salt_needed = 10) 
  (h4 : salt_cost_per_pound = 0.2) 
  (h5 : promotion_cost = 1000) 
  (h6 : ticket_price = 20) 
  (h7 : tickets_sold = 500) 
  (h8 : total_profit = 8798) : 
  ℕ := by
  sorry

#check flour_bag_weight

end flour_bag_weight_l1865_186535


namespace square_sum_formula_l1865_186566

theorem square_sum_formula (x y a b : ℝ) 
  (h1 : x * y = 2 * b) 
  (h2 : 1 / x^2 + 1 / y^2 = a) : 
  (x + y)^2 = 4 * a * b^2 + 4 * b := by
  sorry

end square_sum_formula_l1865_186566


namespace mountain_climb_speed_l1865_186501

theorem mountain_climb_speed 
  (total_time : ℝ) 
  (speed_difference : ℝ) 
  (time_difference : ℝ) 
  (total_distance : ℝ) 
  (h1 : total_time = 14) 
  (h2 : speed_difference = 0.5) 
  (h3 : time_difference = 2) 
  (h4 : total_distance = 52) : 
  ∃ (v : ℝ), v > 0 ∧ 
    v * (total_time / 2 + time_difference) + 
    (v + speed_difference) * (total_time / 2 - time_difference) = total_distance ∧
    v + speed_difference = 4 := by
  sorry

#check mountain_climb_speed

end mountain_climb_speed_l1865_186501


namespace president_and_committee_selection_l1865_186526

theorem president_and_committee_selection (n : ℕ) (h : n = 10) : 
  n * (Nat.choose (n - 1) 3) = 840 := by
  sorry

end president_and_committee_selection_l1865_186526


namespace hyperbola_eccentricity_l1865_186598

/-- A hyperbola with foci F₁ and F₂ -/
structure Hyperbola where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- A line passing through a point -/
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

/-- Intersection points of a line with a hyperbola -/
def intersection (h : Hyperbola) (l : Line) : Set (ℝ × ℝ) :=
  sorry

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Check if a triangle is equilateral -/
def is_equilateral (p q r : ℝ × ℝ) : Prop :=
  sorry

theorem hyperbola_eccentricity (h : Hyperbola) (l : Line) :
  l.point = h.F₂ →
  ∃ (A B : ℝ × ℝ), A ∈ intersection h l ∧ B ∈ intersection h l ∧
  is_equilateral h.F₁ A B →
  eccentricity h = Real.sqrt 3 :=
sorry

end hyperbola_eccentricity_l1865_186598


namespace cylinder_height_equals_half_cm_l1865_186537

/-- Given a cylinder and a sphere with specific dimensions, proves that the height of the cylinder is 0.5 cm when their volumes are equal. -/
theorem cylinder_height_equals_half_cm 
  (d_cylinder : ℝ) 
  (d_sphere : ℝ) 
  (h_cylinder : ℝ) :
  d_cylinder = 6 →
  d_sphere = 3 →
  π * (d_cylinder / 2)^2 * h_cylinder = (4/3) * π * (d_sphere / 2)^3 →
  h_cylinder = 0.5 := by
  sorry

#check cylinder_height_equals_half_cm

end cylinder_height_equals_half_cm_l1865_186537


namespace range_of_a_l1865_186530

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (-1) 1, a * x + 1 > 0) → a ∈ Set.Ioo (-1) 1 := by
  sorry

end range_of_a_l1865_186530


namespace grandfathers_age_l1865_186512

theorem grandfathers_age (x : ℕ) (y z : ℕ) : 
  (6 * x = 6 * x) →  -- Current year
  (6 * x + y = 5 * (x + y)) →  -- In y years
  (6 * x + y + z = 4 * (x + y + z)) →  -- In y + z years
  (x > 0) →  -- Ming's age is positive
  (y > 0) →  -- First time gap is positive
  (z > 0) →  -- Second time gap is positive
  6 * x = 72 := by
  sorry

end grandfathers_age_l1865_186512


namespace thor_jump_count_l1865_186562

def jump_distance (n : ℕ) : ℝ := 3 * (3 ^ (n - 1))

theorem thor_jump_count :
  (∀ k < 10, jump_distance k ≤ 29000) ∧
  jump_distance 10 > 29000 :=
by sorry

end thor_jump_count_l1865_186562


namespace binomial_max_at_one_l1865_186560

/-- The number of trials in the binomial distribution -/
def n : ℕ := 6

/-- The probability of success in a single trial -/
def p : ℚ := 1/6

/-- The binomial probability mass function -/
def binomial_pmf (k : ℕ) : ℚ :=
  (n.choose k) * p^k * (1-p)^(n-k)

/-- Theorem stating that the binomial probability is maximized when k = 1 -/
theorem binomial_max_at_one :
  ∀ k : ℕ, k ≤ n → binomial_pmf 1 ≥ binomial_pmf k :=
sorry

end binomial_max_at_one_l1865_186560


namespace income_percentage_l1865_186589

theorem income_percentage (juan tim mart : ℝ) 
  (h1 : tim = juan * 0.5) 
  (h2 : mart = tim * 1.6) : 
  mart = juan * 0.8 := by
  sorry

end income_percentage_l1865_186589


namespace absolute_value_inequality_l1865_186551

theorem absolute_value_inequality (x : ℝ) :
  3 ≤ |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) :=
by sorry

end absolute_value_inequality_l1865_186551


namespace calcium_oxide_molecular_weight_l1865_186513

/-- Represents an element with its atomic weight -/
structure Element where
  name : String
  atomic_weight : ℝ

/-- Represents a compound made of elements -/
structure Compound where
  name : String
  elements : List Element

/-- Calculates the molecular weight of a compound -/
def molecular_weight (c : Compound) : ℝ :=
  c.elements.map (λ e => e.atomic_weight) |>.sum

/-- Calcium element -/
def calcium : Element := ⟨"Calcium", 40⟩

/-- Oxygen element -/
def oxygen : Element := ⟨"Oxygen", 16⟩

/-- Calcium oxide compound -/
def calcium_oxide : Compound := ⟨"Calcium oxide", [calcium, oxygen]⟩

/-- Theorem: The molecular weight of Calcium oxide is 56 -/
theorem calcium_oxide_molecular_weight :
  molecular_weight calcium_oxide = 56 := by sorry

end calcium_oxide_molecular_weight_l1865_186513


namespace cyclists_meeting_time_l1865_186539

/-- Two cyclists meet on a course -/
theorem cyclists_meeting_time
  (course_length : ℝ)
  (speed1 : ℝ)
  (speed2 : ℝ)
  (h1 : course_length = 45)
  (h2 : speed1 = 14)
  (h3 : speed2 = 16) :
  ∃ t : ℝ, t * (speed1 + speed2) = course_length ∧ t = 1.5 :=
by sorry

end cyclists_meeting_time_l1865_186539


namespace specific_tetrahedron_volume_l1865_186522

/-- Represents a tetrahedron with vertices A, B, C, and D -/
structure Tetrahedron where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  BD : ℝ
  AD : ℝ
  CD : ℝ

/-- Calculates the volume of a tetrahedron given its edge lengths -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem stating that the volume of the specific tetrahedron is 14√13.75 / 9 -/
theorem specific_tetrahedron_volume :
  let t : Tetrahedron := {
    AB := 6,
    AC := 4,
    BC := 5,
    BD := 5,
    AD := 4,
    CD := 3
  }
  tetrahedronVolume t = 14 * Real.sqrt 13.75 / 9 := by
  sorry

end specific_tetrahedron_volume_l1865_186522


namespace problem_solution_l1865_186533

theorem problem_solution (t : ℝ) :
  let x := 3 - t
  let y := 2*t + 11
  x = 1 → y = 15 := by
sorry

end problem_solution_l1865_186533


namespace no_base_with_final_digit_four_l1865_186546

theorem no_base_with_final_digit_four : 
  ∀ b : ℕ, 3 ≤ b ∧ b ≤ 10 → ¬(981 % b = 4) :=
by sorry

end no_base_with_final_digit_four_l1865_186546


namespace parallel_vectors_tan_2alpha_l1865_186523

theorem parallel_vectors_tan_2alpha (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi)
  (h2 : (Real.cos α - 5) * Real.cos α + Real.sin α * (Real.sin α - 5) = 0) :
  Real.tan (2 * α) = 24 / 7 := by
  sorry

end parallel_vectors_tan_2alpha_l1865_186523


namespace sum_of_squares_inequality_l1865_186518

theorem sum_of_squares_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25/2 := by
  sorry

end sum_of_squares_inequality_l1865_186518


namespace geometric_sequence_problem_l1865_186590

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

-- Define the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = (1 : ℝ) / 4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = (1 : ℝ) / 2 := by
  sorry


end geometric_sequence_problem_l1865_186590


namespace apple_pie_pieces_l1865_186573

/-- Calculates the number of pieces each pie is cut into -/
def piecesPer (totalApples : ℕ) (numPies : ℕ) (applesPerSlice : ℕ) : ℕ :=
  (totalApples / numPies) / applesPerSlice

/-- Proves that each pie is cut into 6 pieces given the problem conditions -/
theorem apple_pie_pieces : 
  piecesPer (4 * 12) 4 2 = 6 := by
  sorry

end apple_pie_pieces_l1865_186573


namespace distribute_5_3_l1865_186588

/-- The number of ways to distribute n distinguishable objects into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := k^n

/-- There are 5 distinguishable balls -/
def num_balls : ℕ := 5

/-- There are 3 distinguishable boxes -/
def num_boxes : ℕ := 3

theorem distribute_5_3 : distribute num_balls num_boxes = 243 := by
  sorry

end distribute_5_3_l1865_186588


namespace average_side_length_of_squares_l1865_186550

theorem average_side_length_of_squares (a b c : ℝ) 
  (ha : a = 25) (hb : b = 64) (hc : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end average_side_length_of_squares_l1865_186550


namespace fishbowl_water_volume_l1865_186595

/-- Calculates the volume of water in a cuboid-shaped container. -/
def water_volume (length width water_height : ℝ) : ℝ :=
  length * width * water_height

/-- Proves that the volume of water in the given cuboid-shaped container is 600 cm³. -/
theorem fishbowl_water_volume :
  water_volume 12 10 5 = 600 := by
  sorry

end fishbowl_water_volume_l1865_186595


namespace puzzle_solutions_l1865_186548

def is_valid_solution (a b : Nat) : Prop :=
  a ≠ b ∧
  a ≥ 1 ∧ a ≤ 9 ∧
  b ≥ 1 ∧ b ≤ 9 ∧
  a^b = 10*b + a ∧
  10*b + a ≠ b*a

theorem puzzle_solutions :
  {(a, b) : Nat × Nat | is_valid_solution a b} =
  {(2, 5), (6, 2), (4, 3)} :=
sorry

end puzzle_solutions_l1865_186548


namespace matrix_is_own_inverse_l1865_186545

def A (c d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; c, d]

theorem matrix_is_own_inverse (c d : ℝ) :
  A c d * A c d = 1 ↔ c = 7.5 ∧ d = -4 := by
  sorry

end matrix_is_own_inverse_l1865_186545


namespace income_ratio_is_5_to_4_l1865_186516

-- Define the incomes and expenditures
def income_A : ℕ := 4000
def income_B : ℕ := 3200
def expenditure_A : ℕ := 2400
def expenditure_B : ℕ := 1600

-- Define the savings
def savings : ℕ := 1600

-- Theorem to prove
theorem income_ratio_is_5_to_4 :
  -- Conditions
  (expenditure_A / expenditure_B = 3 / 2) ∧
  (income_A - expenditure_A = savings) ∧
  (income_B - expenditure_B = savings) ∧
  (income_A = 4000) →
  -- Conclusion
  (income_A : ℚ) / (income_B : ℚ) = 5 / 4 :=
by sorry

end income_ratio_is_5_to_4_l1865_186516


namespace inverse_mod_53_l1865_186597

theorem inverse_mod_53 (h : (17⁻¹ : ZMod 53) = 23) : (36⁻¹ : ZMod 53) = 30 := by
  sorry

end inverse_mod_53_l1865_186597


namespace cone_from_sector_l1865_186521

theorem cone_from_sector (θ : Real) (r : Real) (base_radius : Real) (slant : Real) : 
  θ = 270 ∧ r = 12 ∧ base_radius = 9 ∧ slant = 12 →
  (θ / 360) * (2 * Real.pi * r) = 2 * Real.pi * base_radius ∧
  r = slant :=
by sorry

end cone_from_sector_l1865_186521


namespace compute_expression_l1865_186585

theorem compute_expression : 75 * 1313 - 25 * 1313 = 65650 := by
  sorry

end compute_expression_l1865_186585


namespace mod_17_graph_intercepts_sum_l1865_186536

theorem mod_17_graph_intercepts_sum :
  ∀ x_0 y_0 : ℕ,
  x_0 < 17 →
  y_0 < 17 →
  (5 * x_0) % 17 = 2 →
  (3 * y_0) % 17 = 15 →
  x_0 + y_0 = 19 := by
sorry

end mod_17_graph_intercepts_sum_l1865_186536


namespace fraction_evaluation_l1865_186554

theorem fraction_evaluation : 
  (11 - 10 + 9 - 8 + 7 - 6 + 5 - 4 + 3 - 2) / 
  (0 - 1 + 2 - 3 + 4 - 5 + 6 - 7 + 8) = 5 / 4 := by
sorry

end fraction_evaluation_l1865_186554


namespace cone_height_l1865_186558

theorem cone_height (r : ℝ) (lateral_area : ℝ) (h : ℝ) : 
  r = 1 → lateral_area = 2 * Real.pi → h = Real.sqrt 3 → 
  lateral_area = Real.pi * r * Real.sqrt (h^2 + r^2) :=
by sorry

end cone_height_l1865_186558


namespace pizza_slices_per_person_l1865_186529

theorem pizza_slices_per_person (num_people : ℕ) (num_pizzas : ℕ) (slices_per_pizza : ℕ) 
  (h1 : num_people = 6)
  (h2 : num_pizzas = 3)
  (h3 : slices_per_pizza = 8) :
  (num_pizzas * slices_per_pizza) / num_people = 4 := by
  sorry

end pizza_slices_per_person_l1865_186529


namespace fans_with_all_items_count_l1865_186511

/-- The capacity of the stadium --/
def stadium_capacity : ℕ := 5000

/-- The interval for hot dog coupons --/
def hot_dog_interval : ℕ := 60

/-- The interval for soda coupons --/
def soda_interval : ℕ := 40

/-- The interval for ice cream coupons --/
def ice_cream_interval : ℕ := 90

/-- The number of fans who received all three types of free items --/
def fans_with_all_items : ℕ := stadium_capacity / (Nat.lcm hot_dog_interval (Nat.lcm soda_interval ice_cream_interval))

theorem fans_with_all_items_count : fans_with_all_items = 13 := by
  sorry

end fans_with_all_items_count_l1865_186511


namespace functional_equation_solution_l1865_186515

/-- A polynomial satisfying the given functional equation -/
def FunctionalEquationPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, P (x^2 - 2*x) = (P (x - 2))^2

/-- The form of the polynomial satisfying the functional equation -/
def PolynomialForm (P : ℝ → ℝ) : Prop :=
  ∃ n : ℕ+, ∀ x : ℝ, P x = (x + 1)^(n : ℕ)

/-- Theorem stating that any non-zero polynomial satisfying the functional equation
    must be of the form (x + 1)^n for some positive integer n -/
theorem functional_equation_solution :
  ∀ P : ℝ → ℝ, (∃ x : ℝ, P x ≠ 0) → FunctionalEquationPolynomial P → PolynomialForm P :=
by sorry


end functional_equation_solution_l1865_186515


namespace optimal_selling_price_l1865_186553

/-- Represents the annual profit function for a clothing distributor -/
def annual_profit (x : ℝ) : ℝ := -x^2 + 1000*x - 200000

/-- Represents the annual sales volume function -/
def sales_volume (x : ℝ) : ℝ := 800 - x

theorem optimal_selling_price :
  ∃ (x : ℝ),
    x = 400 ∧
    annual_profit x = 40000 ∧
    ∀ y, y ≠ x → annual_profit y = 40000 → sales_volume x > sales_volume y :=
sorry

end optimal_selling_price_l1865_186553


namespace smallest_prime_factor_of_digit_difference_l1865_186531

/-- Given two different digits C and D where C > D, prove that the smallest prime factor
    of the difference between the two-digit number CD and its reverse DC is 3. -/
theorem smallest_prime_factor_of_digit_difference (C D : ℕ) : 
  C ≠ D → C > D → C < 10 → D < 10 → Nat.minFac (10 * C + D - (10 * D + C)) = 3 := by
  sorry

end smallest_prime_factor_of_digit_difference_l1865_186531


namespace evaluate_expression_l1865_186538

theorem evaluate_expression : -(18 / 3 * 8 - 72 + 4 * 8) = 8 := by sorry

end evaluate_expression_l1865_186538


namespace sandy_lemonade_sales_l1865_186574

theorem sandy_lemonade_sales (sunday_half_dollars : ℕ) (total_amount : ℚ) (half_dollar_value : ℚ) :
  sunday_half_dollars = 6 →
  total_amount = 11.5 →
  half_dollar_value = 0.5 →
  (total_amount - sunday_half_dollars * half_dollar_value) / half_dollar_value = 17 := by
sorry

end sandy_lemonade_sales_l1865_186574


namespace min_printers_equal_expenditure_l1865_186559

def printer_costs : List Nat := [400, 350, 500, 200]

theorem min_printers_equal_expenditure :
  let total_cost := Nat.lcm (Nat.lcm (Nat.lcm 400 350) 500) 200
  let num_printers := List.sum (List.map (λ cost => total_cost / cost) printer_costs)
  num_printers = 173 ∧
  ∀ (n : Nat), n < num_printers →
    ∃ (cost : Nat), cost ∈ printer_costs ∧ (n * cost) % total_cost ≠ 0 :=
by sorry

end min_printers_equal_expenditure_l1865_186559


namespace nested_square_root_equality_l1865_186507

theorem nested_square_root_equality : Real.sqrt (13 + Real.sqrt (7 + Real.sqrt 4)) = 4 := by
  sorry

end nested_square_root_equality_l1865_186507


namespace range_of_a_l1865_186520

theorem range_of_a (x a : ℝ) : 
  (3 * x + 2 * (3 * a + 1) = 6 * x + a) → 
  (x ≥ 0) → 
  (a ≥ -2/5) :=
by sorry

end range_of_a_l1865_186520


namespace expression_simplification_l1865_186567

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) :
  (x^2 - y^2) / (x * y) - (x * y - 2 * y^2) / (x * y - x^2) = (x^2 - 2 * y^2) / (x * y) := by
  sorry

end expression_simplification_l1865_186567


namespace perpendicular_planes_l1865_186524

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relationship between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relationship between lines
variable (perp_line_line : Line → Line → Prop)

-- Define the perpendicular relationship between planes
variable (perp_plane_plane : Plane → Plane → Prop)

-- Define non-overlapping relationship for lines
variable (non_overlapping_lines : Line → Line → Prop)

-- Define non-overlapping relationship for planes
variable (non_overlapping_planes : Plane → Plane → Prop)

-- Theorem statement
theorem perpendicular_planes 
  (m n : Line) (α β : Plane)
  (h1 : non_overlapping_lines m n)
  (h2 : non_overlapping_planes α β)
  (h3 : perp_line_plane m α)
  (h4 : perp_line_plane n β)
  (h5 : perp_line_line m n) :
  perp_plane_plane α β :=
sorry

end perpendicular_planes_l1865_186524
