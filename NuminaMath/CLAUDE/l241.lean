import Mathlib

namespace factorize_difference_of_squares_l241_24176

theorem factorize_difference_of_squares (a b : ℝ) : a^2 - 4*b^2 = (a + 2*b) * (a - 2*b) := by
  sorry

end factorize_difference_of_squares_l241_24176


namespace gcd_13013_15015_l241_24130

theorem gcd_13013_15015 : Nat.gcd 13013 15015 = 1001 := by
  sorry

end gcd_13013_15015_l241_24130


namespace fraction_value_l241_24108

theorem fraction_value (x y : ℚ) (hx : x = 7/9) (hy : y = 3/5) :
  (7*x + 5*y) / (63*x*y) = 20/69 := by sorry

end fraction_value_l241_24108


namespace problem_statement_l241_24155

theorem problem_statement (a b c m : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : a + b + c = m) (h2 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2*a)^2 + b * (m - 2*b)^2 + c * (m - 2*c)^2) / (a * b * c) = 12 := by
sorry

end problem_statement_l241_24155


namespace ball_probabilities_l241_24195

def total_balls : ℕ := 4
def red_balls : ℕ := 2

def prob_two_red : ℚ := 1 / 6
def prob_at_least_one_red : ℚ := 5 / 6

theorem ball_probabilities :
  (red_balls * (red_balls - 1)) / (total_balls * (total_balls - 1)) = prob_two_red ∧
  1 - ((total_balls - red_balls) * (total_balls - red_balls - 1)) / (total_balls * (total_balls - 1)) = prob_at_least_one_red :=
by sorry

end ball_probabilities_l241_24195


namespace total_fertilizer_needed_l241_24184

def petunia_flats : ℕ := 4
def petunias_per_flat : ℕ := 8
def rose_flats : ℕ := 3
def roses_per_flat : ℕ := 6
def venus_flytraps : ℕ := 2
def fertilizer_per_petunia : ℕ := 8
def fertilizer_per_rose : ℕ := 3
def fertilizer_per_venus_flytrap : ℕ := 2

theorem total_fertilizer_needed : 
  petunia_flats * petunias_per_flat * fertilizer_per_petunia + 
  rose_flats * roses_per_flat * fertilizer_per_rose + 
  venus_flytraps * fertilizer_per_venus_flytrap = 314 := by
  sorry

end total_fertilizer_needed_l241_24184


namespace fullPriceRevenue_l241_24199

/-- Represents the fundraising event ticket sales -/
structure FundraisingEvent where
  totalTickets : ℕ
  totalRevenue : ℕ
  fullPriceTickets : ℕ
  halfPriceTickets : ℕ
  fullPrice : ℕ

/-- The conditions of the fundraising event -/
def eventConditions (e : FundraisingEvent) : Prop :=
  e.totalTickets = 180 ∧
  e.totalRevenue = 2709 ∧
  e.totalTickets = e.fullPriceTickets + e.halfPriceTickets ∧
  e.totalRevenue = e.fullPriceTickets * e.fullPrice + e.halfPriceTickets * (e.fullPrice / 2)

/-- The theorem to prove -/
theorem fullPriceRevenue (e : FundraisingEvent) :
  eventConditions e → e.fullPriceTickets * e.fullPrice = 2142 := by
  sorry


end fullPriceRevenue_l241_24199


namespace arabella_dance_steps_l241_24197

/-- Arabella's dance step learning problem -/
theorem arabella_dance_steps (T₁ T₂ T₃ : ℚ) 
  (h1 : T₁ = 30)
  (h2 : T₃ = T₁ + T₂)
  (h3 : T₁ + T₂ + T₃ = 90) :
  T₂ / T₁ = 1/2 := by
  sorry

#check arabella_dance_steps

end arabella_dance_steps_l241_24197


namespace non_opaque_arrangements_l241_24106

/-- Represents the number of glasses in the stack -/
def num_glasses : ℕ := 5

/-- Represents the number of possible rotations for each glass -/
def num_rotations : ℕ := 3

/-- Calculates the total number of possible arrangements -/
def total_arrangements : ℕ := num_glasses.factorial * num_rotations ^ (num_glasses - 1)

/-- Calculates the number of opaque arrangements -/
def opaque_arrangements : ℕ := 50 * num_glasses.factorial

/-- Theorem stating the number of non-opaque arrangements -/
theorem non_opaque_arrangements :
  total_arrangements - opaque_arrangements = 3720 :=
sorry

end non_opaque_arrangements_l241_24106


namespace solution_valid_l241_24141

open Real

variables (t : ℝ) (x y : ℝ → ℝ) (C₁ C₂ : ℝ)

def diff_eq_system (x y : ℝ → ℝ) : Prop :=
  (∀ t, deriv x t = (y t + exp (x t)) / (y t + exp t)) ∧
  (∀ t, deriv y t = (y t ^ 2 - exp (x t + t)) / (y t + exp t))

def general_solution (x y : ℝ → ℝ) (C₁ C₂ : ℝ) : Prop :=
  (∀ t, exp (-t) * y t + x t = C₁) ∧
  (∀ t, exp (-(x t)) * y t + t = C₂)

theorem solution_valid :
  diff_eq_system x y → general_solution x y C₁ C₂ → diff_eq_system x y :=
sorry

end solution_valid_l241_24141


namespace base_ratio_in_special_isosceles_trapezoid_l241_24177

/-- Represents an isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  smaller_base : ℝ
  larger_base : ℝ
  diagonal : ℝ
  altitude : ℝ
  sum_of_bases : smaller_base + larger_base = 10
  larger_base_prop : larger_base = 2 * diagonal
  smaller_base_prop : smaller_base = 2 * altitude

/-- Theorem stating the ratio of bases in the specific isosceles trapezoid -/
theorem base_ratio_in_special_isosceles_trapezoid (t : IsoscelesTrapezoid) :
  t.smaller_base / t.larger_base = (2 * Real.sqrt 2 - 1) / 2 := by
  sorry


end base_ratio_in_special_isosceles_trapezoid_l241_24177


namespace problem_statement_l241_24118

theorem problem_statement (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 4 := by
  sorry

end problem_statement_l241_24118


namespace steve_orange_count_l241_24151

/-- The number of oranges each person has -/
structure OrangeCount where
  marcie : ℝ
  brian : ℝ
  shawn : ℝ
  steve : ℝ

/-- The conditions of the orange distribution problem -/
def orange_problem (o : OrangeCount) : Prop :=
  o.marcie = 12 ∧
  o.brian = o.marcie ∧
  o.shawn = (o.marcie + o.brian) * 1.075 ∧
  o.steve = 3 * (o.marcie + o.brian + o.shawn)

/-- The theorem stating Steve's orange count -/
theorem steve_orange_count (o : OrangeCount) (h : orange_problem o) : o.steve = 149.4 := by
  sorry

end steve_orange_count_l241_24151


namespace min_trucks_required_l241_24153

/-- Represents the total weight of boxes in tons -/
def total_weight : ℝ := 10

/-- Represents the maximum weight of a single box in tons -/
def max_box_weight : ℝ := 1

/-- Represents the capacity of each truck in tons -/
def truck_capacity : ℝ := 3

/-- Calculates the minimum number of trucks required -/
def min_trucks : ℕ := 5

theorem min_trucks_required :
  ∀ (weights : List ℝ),
    weights.sum = total_weight →
    (∀ w ∈ weights, w ≤ max_box_weight) →
    (∀ n : ℕ, n < min_trucks → 
      ∃ partition : List (List ℝ),
        partition.length = n ∧
        partition.join.sum = total_weight ∧
        (∀ part ∈ partition, part.sum > truck_capacity)) →
    ∃ partition : List (List ℝ),
      partition.length = min_trucks ∧
      partition.join.sum = total_weight ∧
      (∀ part ∈ partition, part.sum ≤ truck_capacity) :=
by sorry

#check min_trucks_required

end min_trucks_required_l241_24153


namespace sin_to_cos_shift_l241_24116

theorem sin_to_cos_shift (x : ℝ) :
  let f : ℝ → ℝ := λ t ↦ Real.sin (t - π/3)
  let g : ℝ → ℝ := λ t ↦ Real.cos t
  f (x + 5*π/6) = g x := by
sorry

end sin_to_cos_shift_l241_24116


namespace intersection_point_is_solution_l241_24127

/-- Given two linear functions that intersect at a specific point,
    prove that this point is the solution to the system of equations. -/
theorem intersection_point_is_solution (b : ℝ) :
  (∃ (x y : ℝ), y = 3 * x - 5 ∧ y = 2 * x + b) →
  (1 : ℝ) = 3 * (1 : ℝ) - 5 →
  (-2 : ℝ) = 2 * (1 : ℝ) + b →
  (∀ (x y : ℝ), y = 3 * x - 5 ∧ y = 2 * x + b → x = 1 ∧ y = -2) :=
by sorry

end intersection_point_is_solution_l241_24127


namespace min_value_of_polynomial_l241_24165

theorem min_value_of_polynomial (x : ℝ) : 
  x * (x + 4) * (x + 8) * (x + 12) ≥ -256 ∧ 
  ∃ y : ℝ, y * (y + 4) * (y + 8) * (y + 12) = -256 := by sorry

end min_value_of_polynomial_l241_24165


namespace basketball_game_probability_formula_l241_24185

/-- Basketball shooting game between Student A and Student B -/
structure BasketballGame where
  /-- Probability of Student A making a basket -/
  prob_a : ℚ
  /-- Probability of Student B making a basket -/
  prob_b : ℚ
  /-- Each shot is independent -/
  independent_shots : Bool

/-- Score of Student A after one round -/
inductive Score where
  | lose : Score  -- Student A loses (-1)
  | draw : Score  -- Draw (0)
  | win  : Score  -- Student A wins (+1)

/-- Probability distribution of Student A's score after one round -/
def score_distribution (game : BasketballGame) : Score → ℚ
  | Score.lose => (1 - game.prob_a) * game.prob_b
  | Score.draw => game.prob_a * game.prob_b + (1 - game.prob_a) * (1 - game.prob_b)
  | Score.win  => game.prob_a * (1 - game.prob_b)

/-- Expected value of Student A's score after one round -/
def expected_score (game : BasketballGame) : ℚ :=
  -1 * score_distribution game Score.lose +
   0 * score_distribution game Score.draw +
   1 * score_distribution game Score.win

/-- Probability that Student A's cumulative score is lower than Student B's after n rounds -/
def p (n : ℕ) : ℚ :=
  (1 / 5) * (1 - (1 / 6)^n)

/-- Main theorem: Probability formula for Student A's score being lower after n rounds -/
theorem basketball_game_probability_formula (game : BasketballGame) (n : ℕ) 
    (h1 : game.prob_a = 2/3) (h2 : game.prob_b = 1/2) (h3 : game.independent_shots = true) :
    p n = (1 / 5) * (1 - (1 / 6)^n) := by
  sorry

end basketball_game_probability_formula_l241_24185


namespace unique_intersection_l241_24115

/-- 
Given two functions f(x) = ax² + 2x + 3 and g(x) = -2x - 3, 
this theorem states that these functions intersect at exactly one point 
if and only if a = 2/3.
-/
theorem unique_intersection (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 2 * x + 3 = -2 * x - 3) ↔ a = 2/3 := by
  sorry

end unique_intersection_l241_24115


namespace inspector_meter_count_l241_24180

theorem inspector_meter_count : 
  ∀ (total_meters : ℕ) (defective_meters : ℕ) (rejection_rate : ℚ),
    rejection_rate = 1/10 →
    defective_meters = 15 →
    (rejection_rate * total_meters : ℚ) = defective_meters →
    total_meters = 150 := by
  sorry

end inspector_meter_count_l241_24180


namespace billy_coins_problem_l241_24126

theorem billy_coins_problem (total_coins : Nat) (quarter_piles : Nat) (dime_piles : Nat) 
  (h1 : total_coins = 20)
  (h2 : quarter_piles = 2)
  (h3 : dime_piles = 3) :
  ∃! coins_per_pile : Nat, 
    coins_per_pile > 0 ∧ 
    quarter_piles * coins_per_pile + dime_piles * coins_per_pile = total_coins ∧
    coins_per_pile = 4 := by
  sorry

end billy_coins_problem_l241_24126


namespace annual_growth_rate_l241_24146

/-- Given a monthly average growth rate, calculate the annual average growth rate -/
theorem annual_growth_rate (P : ℝ) :
  let monthly_rate := P
  let annual_rate := (1 + P)^12 - 1
  annual_rate = ((1 + monthly_rate)^12 - 1) :=
by sorry

end annual_growth_rate_l241_24146


namespace digital_root_of_2_pow_100_l241_24129

/-- The digital root of a natural number is the single digit obtained by repeatedly summing its digits. -/
def digital_root (n : ℕ) : ℕ := sorry

/-- Theorem: The digital root of 2^100 is 7. -/
theorem digital_root_of_2_pow_100 : digital_root (2^100) = 7 := by sorry

end digital_root_of_2_pow_100_l241_24129


namespace vertex_difference_hexagonal_pentagonal_prism_l241_24133

/-- The number of vertices in a regular polygon. -/
def verticesInPolygon (sides : ℕ) : ℕ := sides

/-- The number of vertices in a prism with regular polygonal bases. -/
def verticesInPrism (baseSides : ℕ) : ℕ := 2 * (verticesInPolygon baseSides)

/-- The difference between the number of vertices of a hexagonal prism and a pentagonal prism. -/
theorem vertex_difference_hexagonal_pentagonal_prism : 
  verticesInPrism 6 - verticesInPrism 5 = 2 := by
  sorry


end vertex_difference_hexagonal_pentagonal_prism_l241_24133


namespace fewer_toys_by_machine_a_l241_24114

/-- The number of toys machine A makes per minute -/
def machine_a_rate : ℕ := 8

/-- The number of toys machine B makes per minute -/
def machine_b_rate : ℕ := 10

/-- The number of toys machine B made -/
def machine_b_toys : ℕ := 100

/-- The time both machines operated, in minutes -/
def operation_time : ℕ := machine_b_toys / machine_b_rate

/-- The number of toys machine A made -/
def machine_a_toys : ℕ := machine_a_rate * operation_time

theorem fewer_toys_by_machine_a : machine_b_toys - machine_a_toys = 20 := by
  sorry

end fewer_toys_by_machine_a_l241_24114


namespace haley_trees_died_l241_24187

/-- The number of trees that died in a typhoon given the initial number of trees and the number of trees remaining. -/
def trees_died (initial_trees remaining_trees : ℕ) : ℕ :=
  initial_trees - remaining_trees

/-- Proof that 5 trees died in the typhoon given the conditions in Haley's problem. -/
theorem haley_trees_died : trees_died 17 12 = 5 := by
  sorry

end haley_trees_died_l241_24187


namespace initial_customer_count_l241_24170

/-- Represents the number of customers at different times -/
structure CustomerCount where
  initial : ℕ
  after_first_hour : ℕ
  after_second_hour : ℕ

/-- Calculates the number of customers after the first hour -/
def first_hour_change (c : CustomerCount) : ℕ := c.initial + 7 - 4

/-- Calculates the number of customers after the second hour -/
def second_hour_change (c : CustomerCount) : ℕ := c.after_first_hour + 3 - 9

/-- The main theorem stating the initial number of customers -/
theorem initial_customer_count : ∃ (c : CustomerCount), 
  c.initial = 15 ∧ 
  c.after_first_hour = first_hour_change c ∧
  c.after_second_hour = second_hour_change c ∧
  c.after_second_hour = 12 := by
  sorry

end initial_customer_count_l241_24170


namespace complex_cube_sum_ratio_l241_24101

theorem complex_cube_sum_ratio (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 30)
  (h_eq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 3*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 48 := by
sorry

end complex_cube_sum_ratio_l241_24101


namespace log_equation_solution_l241_24110

theorem log_equation_solution (x : ℝ) :
  Real.log x / Real.log 9 = 2.4 → x = (81 ^ (1/5)) ^ 6 := by
  sorry

end log_equation_solution_l241_24110


namespace right_triangle_hypotenuse_l241_24109

theorem right_triangle_hypotenuse : 
  ∀ (a b c : ℕ),
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  b = c - 1575 →
  a < 1991 →
  c = 1800 := by
sorry

end right_triangle_hypotenuse_l241_24109


namespace unique_polynomial_solution_l241_24156

/-- A polynomial P(x) that satisfies P(P(x)) = (x^2 + x + 1) P(x) -/
def P (x : ℝ) : ℝ := x^2 + x

/-- Theorem stating that P(x) = x^2 + x is the unique nonconstant polynomial solution 
    to the equation P(P(x)) = (x^2 + x + 1) P(x) -/
theorem unique_polynomial_solution :
  (∀ x, P (P x) = (x^2 + x + 1) * P x) ∧
  (∀ Q : ℝ → ℝ, (∀ x, Q (Q x) = (x^2 + x + 1) * Q x) → 
    (∃ a b c, ∀ x, Q x = a * x^2 + b * x + c) →
    (∃ x y, Q x ≠ Q y) →
    (∀ x, Q x = P x)) :=
by sorry


end unique_polynomial_solution_l241_24156


namespace first_number_in_second_set_l241_24173

theorem first_number_in_second_set (x : ℝ) : 
  (24 + 35 + 58) / 3 = ((x + 51 + 29) / 3) + 6 → x = 19 := by
  sorry

end first_number_in_second_set_l241_24173


namespace arcsin_sqrt3_over_2_l241_24103

theorem arcsin_sqrt3_over_2 : Real.arcsin (Real.sqrt 3 / 2) = π / 3 := by
  sorry

end arcsin_sqrt3_over_2_l241_24103


namespace sum_of_fractions_equals_one_l241_24194

theorem sum_of_fractions_equals_one (a b c : ℝ) (h : a * b * c = 1) :
  1 / (1 + a + a * b) + 1 / (1 + b + b * c) + 1 / (1 + c + c * a) = 1 := by
  sorry

end sum_of_fractions_equals_one_l241_24194


namespace student_professor_ratio_l241_24128

def total_people : ℕ := 40000
def num_students : ℕ := 37500

theorem student_professor_ratio :
  let num_professors := total_people - num_students
  num_students / num_professors = 15 := by
  sorry

end student_professor_ratio_l241_24128


namespace dime_difference_l241_24179

/-- Represents the content of a piggy bank --/
structure PiggyBank where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ

/-- Calculates the total number of coins in the piggy bank --/
def totalCoins (pb : PiggyBank) : ℕ :=
  pb.pennies + pb.nickels + pb.dimes

/-- Calculates the total value in cents of the coins in the piggy bank --/
def totalValue (pb : PiggyBank) : ℕ :=
  pb.pennies + 5 * pb.nickels + 10 * pb.dimes

/-- Checks if a piggy bank configuration is valid --/
def isValidPiggyBank (pb : PiggyBank) : Prop :=
  totalCoins pb = 150 ∧ totalValue pb = 500

/-- The set of all valid piggy bank configurations --/
def validPiggyBanks : Set PiggyBank :=
  {pb | isValidPiggyBank pb}

/-- The theorem to be proven --/
theorem dime_difference : 
  (⨆ (pb : PiggyBank) (h : pb ∈ validPiggyBanks), pb.dimes) -
  (⨅ (pb : PiggyBank) (h : pb ∈ validPiggyBanks), pb.dimes) = 39 := by
  sorry

end dime_difference_l241_24179


namespace complement_A_inter_B_a_range_l241_24122

open Set

-- Define the sets A and B
def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 6}
def B : Set ℝ := {x | x ≥ 3}

-- Define the complement of the intersection of A and B
def complement_intersection : Set ℝ := {x | x < 3 ∨ x > 6}

-- Define set C
def C (a : ℝ) : Set ℝ := {x | x ≤ a}

-- Theorem 1: The complement of A ∩ B is equal to the defined complement_intersection
theorem complement_A_inter_B : (A ∩ B)ᶜ = complement_intersection := by sorry

-- Theorem 2: If A is a subset of C, then a is greater than or equal to 6
theorem a_range (a : ℝ) (h : A ⊆ C a) : a ≥ 6 := by sorry

end complement_A_inter_B_a_range_l241_24122


namespace app_security_theorem_all_measures_secure_l241_24145

/-- Represents a security measure for protecting credit card data -/
inductive SecurityMeasure
  | avoidStoringCardData
  | encryptStoredData
  | encryptDataInTransit
  | codeObfuscation
  | restrictRootedDevices
  | antivirusProtection

/-- Represents an online store app with credit card payment and home delivery -/
structure OnlineStoreApp :=
  (implementedMeasures : List SecurityMeasure)

/-- Defines what it means for an app to be secure -/
def isSecure (app : OnlineStoreApp) : Prop :=
  app.implementedMeasures.length ≥ 3

/-- Theorem stating that implementing at least three security measures 
    ensures the app is secure -/
theorem app_security_theorem (app : OnlineStoreApp) :
  app.implementedMeasures.length ≥ 3 → isSecure app :=
by
  sorry

/-- Corollary: An app with all six security measures is secure -/
theorem all_measures_secure (app : OnlineStoreApp) :
  app.implementedMeasures.length = 6 → isSecure app :=
by
  sorry

end app_security_theorem_all_measures_secure_l241_24145


namespace f_monotonicity_and_g_zeros_l241_24125

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (x^3 - a*x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x^2

theorem f_monotonicity_and_g_zeros (a : ℝ) :
  (a ≤ 0 → ∀ x y : ℝ, x < y → f a x < f a y) ∧
  (a > 0 → ∀ x y : ℝ, 
    ((x < y ∧ y < -Real.sqrt (a/3)) ∨ (x > Real.sqrt (a/3) ∧ y > x)) → f a x < f a y) ∧
  (a > 0 → ∀ x y : ℝ, 
    (-Real.sqrt (a/3) < x ∧ x < y ∧ y < Real.sqrt (a/3)) → f a x > f a y) ∧
  (∃ x y : ℝ, x < y ∧ g a x = 0 ∧ g a y = 0 ∧ ∀ z : ℝ, z ≠ x ∧ z ≠ y → g a z ≠ 0) →
  a > 1 :=
sorry

end f_monotonicity_and_g_zeros_l241_24125


namespace sum_of_cubes_of_roots_l241_24192

theorem sum_of_cubes_of_roots (x₁ x₂ x₃ : ℝ) : 
  x₁^3 + x₂^3 + x₃^3 = 11 ∧ 
  x₁ + x₂ + x₃ = 2 ∧
  x₁ * x₂ + x₂ * x₃ + x₃ * x₁ = -1 ∧
  x₁ * x₂ * x₃ = -1 ∧
  x₁^3 - 2*x₁^2 - x₁ + 1 = 0 ∧
  x₂^3 - 2*x₂^2 - x₂ + 1 = 0 ∧
  x₃^3 - 2*x₃^2 - x₃ + 1 = 0 :=
by sorry

end sum_of_cubes_of_roots_l241_24192


namespace circle_radius_l241_24144

/-- A circle with equation x^2 + y^2 - 2x + my - 4 = 0 that is symmetric about the line 2x + y = 0 has a radius of 3 -/
theorem circle_radius (m : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 2*x + m*y - 4 = 0 → (∃ x' y' : ℝ, x'^2 + y'^2 - 2*x' + m*y' - 4 = 0 ∧ 
    2*x + y = 0 ∧ 2*x' + y' = 0 ∧ x + x' = 2*x ∧ y + y' = 2*y)) → 
  (∃ c_x c_y : ℝ, ∀ x y : ℝ, (x - c_x)^2 + (y - c_y)^2 = 3^2 ↔ x^2 + y^2 - 2*x + m*y - 4 = 0) :=
by sorry

end circle_radius_l241_24144


namespace least_three_digit_multiple_l241_24191

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (100 ≤ n ∧ n < 1000) ∧ 
  (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ (7 ∣ n) ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n) → ¬((2 ∣ m) ∧ (3 ∣ m) ∧ (5 ∣ m) ∧ (7 ∣ m))) :=
by
  -- Proof goes here
  sorry

end least_three_digit_multiple_l241_24191


namespace cyclic_inequality_l241_24160

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 + y^3 + z^3 ≥ x^2 * Real.sqrt (y*z) + y^2 * Real.sqrt (z*x) + z^2 * Real.sqrt (x*y) :=
sorry

end cyclic_inequality_l241_24160


namespace chicken_count_l241_24150

/-- The number of chickens in the coop -/
def coop_chickens : ℕ := 14

/-- The number of chickens in the run -/
def run_chickens : ℕ := 2 * coop_chickens

/-- The total number of chickens in the coop and run -/
def total_coop_run : ℕ := coop_chickens + run_chickens

/-- The number of free-ranging chickens -/
def free_range_chickens : ℕ := 105

theorem chicken_count : 
  (2 : ℚ) / 5 = (total_coop_run : ℚ) / free_range_chickens := by
  sorry

end chicken_count_l241_24150


namespace twenty_three_to_binary_l241_24163

-- Define a function to convert a natural number to its binary representation
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false]
  else
    let rec aux (m : ℕ) : List Bool :=
      if m = 0 then []
      else (m % 2 = 1) :: aux (m / 2)
    aux n

-- Define the decimal number we want to convert
def decimal_number : ℕ := 23

-- Define the expected binary representation
def expected_binary : List Bool := [true, true, true, false, true]

-- Theorem statement
theorem twenty_three_to_binary :
  to_binary decimal_number = expected_binary := by sorry

end twenty_three_to_binary_l241_24163


namespace horner_method_v2_l241_24189

/-- Horner's method for polynomial evaluation -/
def horner_v2 (x : ℤ) : ℤ := x^2 + 6

/-- The polynomial f(x) = x^6 + 6x^4 + 9x^2 + 208 -/
def f (x : ℤ) : ℤ := x^6 + 6*x^4 + 9*x^2 + 208

theorem horner_method_v2 :
  horner_v2 (-4) = 22 :=
by sorry

end horner_method_v2_l241_24189


namespace dropped_class_hours_l241_24117

/-- Calculates the remaining class hours after dropping a class -/
def remaining_class_hours (initial_classes : ℕ) (hours_per_class : ℕ) (dropped_classes : ℕ) : ℕ :=
  (initial_classes - dropped_classes) * hours_per_class

/-- Theorem: Given 4 classes of 2 hours each, dropping 1 class results in 6 hours of classes -/
theorem dropped_class_hours : remaining_class_hours 4 2 1 = 6 := by
  sorry

end dropped_class_hours_l241_24117


namespace tabitha_current_age_l241_24162

/-- Tabitha's hair color tradition --/
def tabitha_age : ℕ → Prop :=
  fun current_age =>
    ∃ (colors : ℕ),
      colors = current_age - 15 + 2 ∧
      colors + 3 = 8

theorem tabitha_current_age :
  ∃ (age : ℕ), tabitha_age age ∧ age = 20 := by
  sorry

end tabitha_current_age_l241_24162


namespace petes_marbles_l241_24161

theorem petes_marbles (total_initial : ℕ) (blue_percent : ℚ) (trade_ratio : ℕ) (kept_red : ℕ) :
  total_initial = 10 ∧
  blue_percent = 2/5 ∧
  trade_ratio = 2 ∧
  kept_red = 1 →
  (total_initial * blue_percent).floor +
  kept_red +
  trade_ratio * ((total_initial * (1 - blue_percent)).floor - kept_red) = 15 := by
  sorry

end petes_marbles_l241_24161


namespace lottery_prizes_approx_10_l241_24158

-- Define the number of blanks
def num_blanks : ℕ := 25

-- Define the probability of drawing a blank
def blank_probability : ℚ := 5000000000000000/7000000000000000

-- Define the function to calculate the number of prizes
def calculate_prizes (blanks : ℕ) (prob : ℚ) : ℚ :=
  (blanks : ℚ) / prob - blanks

-- Theorem statement
theorem lottery_prizes_approx_10 :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_prizes num_blanks blank_probability - 10| < ε :=
sorry

end lottery_prizes_approx_10_l241_24158


namespace text_ratio_is_five_to_one_l241_24136

/-- Represents the number of texts in each category --/
structure TextCounts where
  grocery : ℕ
  notResponding : ℕ
  police : ℕ

/-- The conditions of the problem --/
def textProblemConditions (t : TextCounts) : Prop :=
  t.grocery = 5 ∧
  t.police = (t.grocery + t.notResponding) / 10 ∧
  t.grocery + t.notResponding + t.police = 33

/-- The theorem to be proved --/
theorem text_ratio_is_five_to_one (t : TextCounts) :
  textProblemConditions t →
  t.notResponding / t.grocery = 5 := by
sorry

end text_ratio_is_five_to_one_l241_24136


namespace box_sales_ratio_l241_24190

/-- Proof of the ratio of boxes sold on Saturday to Friday -/
theorem box_sales_ratio :
  ∀ (friday saturday sunday : ℕ),
  friday = 30 →
  sunday = saturday - 15 →
  friday + saturday + sunday = 135 →
  saturday / friday = 2 := by
sorry

end box_sales_ratio_l241_24190


namespace hyperbola_ellipse_intersection_l241_24104

theorem hyperbola_ellipse_intersection (m : ℝ) : 
  (∃ e : ℝ, e > Real.sqrt 2 ∧ e^2 = (3 + m) / 3) ∧ 
  (m / 2 > m - 2 ∧ m - 2 > 0) → 
  m ∈ Set.Ioo 3 4 :=
by sorry

end hyperbola_ellipse_intersection_l241_24104


namespace larger_solution_of_quadratic_l241_24181

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 36 = 0 → x ≤ 9 :=
by sorry

end larger_solution_of_quadratic_l241_24181


namespace exponential_greater_than_trig_squared_l241_24147

theorem exponential_greater_than_trig_squared (x : ℝ) : 
  Real.exp x + Real.exp (-x) ≥ (Real.sin x + Real.cos x)^2 := by
  sorry

end exponential_greater_than_trig_squared_l241_24147


namespace inverse_of_100_mod_101_l241_24132

theorem inverse_of_100_mod_101 : ∃ x : ℕ, 0 ≤ x ∧ x ≤ 100 ∧ (100 * x) % 101 = 1 := by
  sorry

end inverse_of_100_mod_101_l241_24132


namespace problem_statement_l241_24100

def P : Set ℝ := {-1, 1}
def Q (a : ℝ) : Set ℝ := {x : ℝ | a * x = 1}

theorem problem_statement (a : ℝ) : P ∪ Q a = P → a ∈ ({-1, 0, 1} : Set ℝ) := by
  sorry

end problem_statement_l241_24100


namespace intersection_with_complement_l241_24140

def U : Set Nat := {1,2,3,4,5,6,7}
def A : Set Nat := {2,4,5}
def B : Set Nat := {1,3,5,7}

theorem intersection_with_complement : A ∩ (U \ B) = {2,4} := by
  sorry

end intersection_with_complement_l241_24140


namespace inequality_and_equality_condition_l241_24198

theorem inequality_and_equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 1) :
  (1 / (1 + a) + 1 / (1 + b) ≤ 1) ∧
  (1 / (1 + a) + 1 / (1 + b) = 1 ↔ a * b = 1) := by
  sorry

end inequality_and_equality_condition_l241_24198


namespace prob_king_queen_standard_deck_l241_24102

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Nat)
  (ranks : Nat)
  (suits : Nat)
  (cards_per_rank_suit : Nat)

/-- A standard deck has 52 cards, 13 ranks, 4 suits, and 1 card per rank per suit -/
def standard_deck : Deck :=
  { cards := 52
  , ranks := 13
  , suits := 4
  , cards_per_rank_suit := 1
  }

/-- The probability of drawing a King first and a Queen second from a standard deck -/
def prob_king_queen (d : Deck) : ℚ :=
  (d.suits : ℚ) / d.cards * (d.suits : ℚ) / (d.cards - 1)

/-- Theorem: The probability of drawing a King first and a Queen second from a standard deck is 4/663 -/
theorem prob_king_queen_standard_deck : 
  prob_king_queen standard_deck = 4 / 663 := by
  sorry

end prob_king_queen_standard_deck_l241_24102


namespace clothing_purchase_optimal_l241_24172

/-- Represents the prices and quantities of clothing types A and B -/
structure ClothingPrices where
  price_a : ℝ
  price_b : ℝ
  quantity_a : ℕ
  quantity_b : ℕ

/-- The conditions and solution for the clothing purchase problem -/
def clothing_problem (p : ClothingPrices) : Prop :=
  -- Conditions
  p.price_a + 2 * p.price_b = 110 ∧
  2 * p.price_a + 3 * p.price_b = 190 ∧
  p.quantity_a + p.quantity_b = 100 ∧
  p.quantity_a ≥ (p.quantity_b : ℝ) / 3 ∧
  -- Solution
  p.price_a = 50 ∧
  p.price_b = 30 ∧
  p.quantity_a = 25 ∧
  p.quantity_b = 75

/-- The total cost of purchasing the clothing with the discount -/
def total_cost (p : ClothingPrices) : ℝ :=
  (p.price_a - 5) * p.quantity_a + p.price_b * p.quantity_b

/-- Theorem stating that the given solution minimizes the cost -/
theorem clothing_purchase_optimal (p : ClothingPrices) :
  clothing_problem p →
  total_cost p = 3375 ∧
  (∀ q : ClothingPrices, clothing_problem q → total_cost q ≥ total_cost p) :=
sorry

end clothing_purchase_optimal_l241_24172


namespace subtraction_multiplication_equality_l241_24134

theorem subtraction_multiplication_equality : (3.456 - 1.234) * 0.5 = 1.111 := by
  sorry

end subtraction_multiplication_equality_l241_24134


namespace joe_monthly_income_correct_l241_24149

/-- Joe's monthly income in dollars -/
def monthly_income : ℝ := 2120

/-- The fraction of Joe's income that goes to taxes -/
def tax_rate : ℝ := 0.4

/-- The amount Joe pays in taxes each month in dollars -/
def tax_paid : ℝ := 848

/-- Theorem stating that Joe's monthly income is correct given the tax rate and tax paid -/
theorem joe_monthly_income_correct : 
  tax_rate * monthly_income = tax_paid :=
by sorry

end joe_monthly_income_correct_l241_24149


namespace inner_triangle_perimeter_l241_24175

/-- Given a triangle ABC with side lengths, prove that the perimeter of the inner triangle
    formed by lines parallel to each side is equal to the length of side AB. -/
theorem inner_triangle_perimeter (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_x_lt_c : x < c) (h_y_lt_a : y < a) (h_z_lt_b : z < b)
  (h_prop_x : x / c = (c - x) / a) (h_prop_y : y / a = (a - y) / b) (h_prop_z : z / b = (b - z) / c) :
  x / c * a + y / a * b + z / b * c = a := by
  sorry

end inner_triangle_perimeter_l241_24175


namespace equation_solution_l241_24168

theorem equation_solution : 
  ∃! x : ℚ, (2 * x) / (x + 3) + 1 = 7 / (2 * x + 6) := by
  sorry

end equation_solution_l241_24168


namespace f_inequality_l241_24159

/-- An odd function f: ℝ → ℝ with specific properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x - 4) = -f x) ∧
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y)

/-- Theorem stating the inequality for the given function -/
theorem f_inequality (f : ℝ → ℝ) (h : f_properties f) : 
  f (-1) < f 4 ∧ f 4 < f 3 := by
  sorry

end f_inequality_l241_24159


namespace area_not_above_y_axis_equals_total_area_l241_24131

/-- Parallelogram PQRS with given vertices -/
structure Parallelogram where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ

/-- The specific parallelogram from the problem -/
def PQRS : Parallelogram :=
  { P := (-1, 5)
    Q := (2, -3)
    R := (-5, -3)
    S := (-8, 5) }

/-- Area of a parallelogram -/
def parallelogramArea (p : Parallelogram) : ℝ :=
  sorry

/-- Area of the part of the parallelogram not above the y-axis -/
def areaNotAboveYAxis (p : Parallelogram) : ℝ :=
  sorry

/-- Theorem stating that the area not above the y-axis equals the total area -/
theorem area_not_above_y_axis_equals_total_area :
  areaNotAboveYAxis PQRS = parallelogramArea PQRS :=
sorry

end area_not_above_y_axis_equals_total_area_l241_24131


namespace expression_evaluation_l241_24186

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 - 1
  (x + 3) * (x - 3) - x * (x - 2) = 2 * Real.sqrt 2 - 11 := by
  sorry

end expression_evaluation_l241_24186


namespace divisibility_problem_l241_24152

theorem divisibility_problem :
  ∃ k : ℕ, (2^286 - 1) * (3^500 - 1) * (1978^100 - 1) = k * (2^4 * 5^7 * 2003) := by
  sorry

end divisibility_problem_l241_24152


namespace digit_sequence_sum_value_l241_24119

def is_increasing (n : ℕ) : Prop := sorry

def is_decreasing (n : ℕ) : Prop := sorry

def digit_sequence_sum : ℕ := sorry

theorem digit_sequence_sum_value : 
  digit_sequence_sum = (80 * 11^10 - 35 * 2^10) / 81 - 45 := by sorry

end digit_sequence_sum_value_l241_24119


namespace cryptarithm_multiplication_l241_24123

theorem cryptarithm_multiplication :
  ∃! n : ℕ, ∃ m : ℕ,
    100 ≤ n ∧ n < 1000 ∧
    10000 ≤ m ∧ m < 100000 ∧
    n * n = m ∧
    ∃ k : ℕ, 100 ≤ k ∧ k < 1000 ∧ m = k * 1000 + k :=
by sorry

end cryptarithm_multiplication_l241_24123


namespace concentric_circles_area_l241_24137

theorem concentric_circles_area (r : ℝ) (h : r > 0) : 
  2 * π * r + 2 * π * (2 * r) = 36 * π → 
  π * (2 * r)^2 - π * r^2 = 108 * π := by
sorry

end concentric_circles_area_l241_24137


namespace overlap_time_theorem_l241_24164

structure MovingSegment where
  length : ℝ
  initialPosition : ℝ
  speed : ℝ

def positionAt (s : MovingSegment) (t : ℝ) : ℝ :=
  s.initialPosition + s.speed * t

theorem overlap_time_theorem (ab mn : MovingSegment)
  (hab : ab.length = 100)
  (hmn : mn.length = 40)
  (hab_init : ab.initialPosition = 120)
  (hab_speed : ab.speed = -50)
  (hmn_init : mn.initialPosition = -30)
  (hmn_speed : mn.speed = 30)
  (overlap : ℝ) (hoverlap : overlap = 32) :
  ∃ t : ℝ, (t = 71/40 ∨ t = 109/40) ∧
    (positionAt ab t + ab.length - positionAt mn t = overlap ∨
     positionAt mn t + mn.length - positionAt ab t = overlap) :=
sorry

end overlap_time_theorem_l241_24164


namespace constant_area_l241_24166

-- Define the ellipses
def C₁ (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1
def C₂ (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the point P on C₂
def P : ℝ × ℝ → Prop := λ p => C₂ p.1 p.2

-- Define the line OP
def OP (p : ℝ × ℝ) : Set (ℝ × ℝ) := {q : ℝ × ℝ | ∃ t : ℝ, q.1 = t * p.1 ∧ q.2 = t * p.2}

-- Define the points A and B
def A (p : ℝ × ℝ) : ℝ × ℝ := (2 * p.1, 2 * p.2)
def B : ℝ × ℝ → ℝ × ℝ := sorry  -- We don't have enough information to define B explicitly

-- Define the tangent line l to C₂ at P
def l (p : ℝ × ℝ) : Set (ℝ × ℝ) := {q : ℝ × ℝ | p.1 * q.1 + 4 * p.2 * q.2 = 4}

-- Define the points C and D
def C : ℝ × ℝ → ℝ × ℝ := sorry  -- We don't have enough information to define C explicitly
def D : ℝ × ℝ → ℝ × ℝ := sorry  -- We don't have enough information to define D explicitly

-- Define the area of quadrilateral ACBD
def area_ACBD (p : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem constant_area (p : ℝ × ℝ) (hp : P p) :
  area_ACBD p = 8 * Real.sqrt 3 := by sorry

end constant_area_l241_24166


namespace trapezoid_side_length_l241_24178

/-- Represents a trapezoid with an inscribed circle -/
structure TrapezoidWithInscribedCircle where
  /-- Distance from the center of the inscribed circle to one end of a non-parallel side -/
  distance1 : ℝ
  /-- Distance from the center of the inscribed circle to the other end of the same non-parallel side -/
  distance2 : ℝ

/-- Theorem: If the center of the inscribed circle in a trapezoid is at distances 5 and 12
    from the ends of one non-parallel side, then the length of that side is 13. -/
theorem trapezoid_side_length (t : TrapezoidWithInscribedCircle)
    (h1 : t.distance1 = 5)
    (h2 : t.distance2 = 12) :
    Real.sqrt (t.distance1^2 + t.distance2^2) = 13 := by
  sorry

end trapezoid_side_length_l241_24178


namespace jake_weight_proof_l241_24154

/-- Jake's present weight in pounds -/
def jake_weight : ℝ := 196

/-- Jake's sister's weight in pounds -/
def sister_weight : ℝ := 290 - jake_weight

theorem jake_weight_proof :
  (jake_weight - 8 = 2 * sister_weight) ∧
  (jake_weight + sister_weight = 290) →
  jake_weight = 196 := by
  sorry

end jake_weight_proof_l241_24154


namespace square_root_calculation_l241_24135

theorem square_root_calculation : 2 * (Real.sqrt 50625)^2 = 101250 := by
  sorry

end square_root_calculation_l241_24135


namespace log_sin_in_terms_of_m_n_l241_24120

theorem log_sin_in_terms_of_m_n (α m n : Real) 
  (h1 : 0 < α) (h2 : α < π/2)
  (h3 : Real.log (1 + Real.cos α) = m)
  (h4 : Real.log (1 / (1 - Real.cos α)) = n) :
  Real.log (Real.sin α) = (1/2) * (m - n) := by
  sorry

end log_sin_in_terms_of_m_n_l241_24120


namespace polynomial_division_remainder_l241_24196

theorem polynomial_division_remainder :
  let f (x : ℝ) := x^6 - 2*x^5 + x^4 - x^2 + 3*x - 1
  let g (x : ℝ) := (x^2 - 1)*(x + 2)
  let r (x : ℝ) := 7/3*x^2 + x - 7/3
  ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x :=
by sorry

end polynomial_division_remainder_l241_24196


namespace train_length_l241_24182

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) :
  speed = 45 * 1000 / 3600 →
  time = 30 →
  bridge_length = 215 →
  speed * time - bridge_length = 160 :=
by sorry

end train_length_l241_24182


namespace smallest_number_of_eggs_l241_24124

theorem smallest_number_of_eggs : ∀ n : ℕ,
  (n > 150) →
  (∃ k : ℕ, n = 15 * k - 6) →
  (∀ m : ℕ, (m > 150 ∧ ∃ j : ℕ, m = 15 * j - 6) → m ≥ n) →
  n = 159 := by
sorry

end smallest_number_of_eggs_l241_24124


namespace perimeter_circumference_ratio_l241_24138

/-- An isosceles trapezoid with an inscribed circle -/
structure IsoscelesTrapezoidWithInscribedCircle where
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The distance from the center of the circle to the intersection of diagonals -/
  d : ℝ
  /-- Condition that d is 3/5 of r -/
  h_d_ratio : d = 3/5 * r

/-- The perimeter of the trapezoid -/
def perimeter (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ := sorry

/-- The circumference of the inscribed circle -/
def circumference (t : IsoscelesTrapezoidWithInscribedCircle) : ℝ := sorry

theorem perimeter_circumference_ratio 
  (t : IsoscelesTrapezoidWithInscribedCircle) : 
  perimeter t / circumference t = 5 / Real.pi := by sorry

end perimeter_circumference_ratio_l241_24138


namespace quilt_shaded_fraction_l241_24111

theorem quilt_shaded_fraction (total_squares : ℕ) (divided_squares : ℕ) 
  (h1 : total_squares = 16) 
  (h2 : divided_squares = 8) : 
  (divided_squares : ℚ) / (2 : ℚ) / total_squares = 1 / 4 := by
  sorry

end quilt_shaded_fraction_l241_24111


namespace decimal_to_base5_250_l241_24105

/-- Converts a base-10 number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ := sorry

theorem decimal_to_base5_250 :
  toBase5 250 = [2, 0, 0, 0] := by sorry

end decimal_to_base5_250_l241_24105


namespace monomial_2015_coeff_l241_24143

/-- The coefficient of the nth monomial in the sequence -/
def monomial_coeff (n : ℕ) : ℤ := (-1)^n * (2*n - 1)

/-- The theorem stating that the 2015th monomial coefficient is -4029 -/
theorem monomial_2015_coeff : monomial_coeff 2015 = -4029 := by
  sorry

end monomial_2015_coeff_l241_24143


namespace odd_power_congruence_l241_24174

theorem odd_power_congruence (a n : ℕ) (h_odd : Odd a) (h_pos : 0 < n) :
  (a ^ (2 ^ n)) ≡ 1 [MOD 2 ^ (n + 2)] := by
  sorry

end odd_power_congruence_l241_24174


namespace renovation_project_materials_l241_24112

theorem renovation_project_materials (sand dirt cement : ℝ) 
  (h_sand : sand = 0.17)
  (h_dirt : dirt = 0.33)
  (h_cement : cement = 0.17) :
  sand + dirt + cement = 0.67 := by
  sorry

end renovation_project_materials_l241_24112


namespace max_area_rectangle_perimeter_100_l241_24113

/-- The maximum area of a rectangle with perimeter 100 and integer side lengths --/
theorem max_area_rectangle_perimeter_100 :
  ∃ (w h : ℕ), w + h = 50 ∧ w * h = 625 ∧ 
  ∀ (x y : ℕ), x + y = 50 → x * y ≤ 625 := by
  sorry

end max_area_rectangle_perimeter_100_l241_24113


namespace hide_and_seek_players_l241_24142

-- Define the players
variable (Andrew Boris Vasya Gena Denis : Prop)

-- Define the conditions
axiom condition1 : Andrew → (Boris ∧ ¬Vasya)
axiom condition2 : Boris → (Gena ∨ Denis)
axiom condition3 : ¬Vasya → (¬Boris ∧ ¬Denis)
axiom condition4 : ¬Andrew → (Boris ∧ ¬Gena)

-- Theorem to prove
theorem hide_and_seek_players :
  (Boris ∧ Vasya ∧ Denis ∧ ¬Andrew ∧ ¬Gena) ∧
  ∀ (A B V G D : Bool),
    (A → (B ∧ ¬V)) →
    (B → (G ∨ D)) →
    (¬V → (¬B ∧ ¬D)) →
    (¬A → (B ∧ ¬G)) →
    (A, B, V, G, D) = (false, true, true, false, true) :=
by sorry

end hide_and_seek_players_l241_24142


namespace girls_in_college_l241_24107

theorem girls_in_college (total_students : ℕ) (boy_ratio girl_ratio : ℕ) : 
  total_students = 1040 →
  boy_ratio = 8 →
  girl_ratio = 5 →
  (boy_ratio + girl_ratio) * (total_students / (boy_ratio + girl_ratio)) = total_students →
  girl_ratio * (total_students / (boy_ratio + girl_ratio)) = 400 :=
by
  sorry


end girls_in_college_l241_24107


namespace rectangle_perimeter_l241_24193

/-- The perimeter of a rectangular field with area 800 square meters and width 20 meters is 120 meters. -/
theorem rectangle_perimeter (area width : ℝ) (h_area : area = 800) (h_width : width = 20) :
  2 * (area / width + width) = 120 :=
by sorry

end rectangle_perimeter_l241_24193


namespace last_k_digits_power_l241_24169

theorem last_k_digits_power (A B : ℤ) (k n : ℕ) (h : A ≡ B [ZMOD 10^k]) :
  A^n ≡ B^n [ZMOD 10^k] := by sorry

end last_k_digits_power_l241_24169


namespace ellipse_chord_slope_range_l241_24188

/-- The slope of a chord of the ellipse x^2 + y^2/4 = 1 whose midpoint lies on the line segment
    between (1/2, 1/2) and (1/2, 1) is between -4 and -2. -/
theorem ellipse_chord_slope_range :
  ∀ (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ),
  (x₁^2 + y₁^2/4 = 1) →  -- P(x₁, y₁) is on the ellipse
  (x₂^2 + y₂^2/4 = 1) →  -- Q(x₂, y₂) is on the ellipse
  (x₀ = (x₁ + x₂)/2) →   -- x-coordinate of midpoint
  (y₀ = (y₁ + y₂)/2) →   -- y-coordinate of midpoint
  (x₀ = 1/2) →           -- midpoint x-coordinate is on AB
  (1/2 ≤ y₀ ∧ y₀ ≤ 1) →  -- midpoint y-coordinate is between A and B
  (-4 ≤ -(y₁ - y₂)/(x₁ - x₂) ∧ -(y₁ - y₂)/(x₁ - x₂) ≤ -2) :=
by sorry

end ellipse_chord_slope_range_l241_24188


namespace highlighter_spend_l241_24121

def total_money : ℕ := 100
def heaven_spend : ℕ := 30
def eraser_price : ℕ := 4
def eraser_count : ℕ := 10

theorem highlighter_spend :
  total_money - heaven_spend - (eraser_price * eraser_count) = 30 := by
  sorry

end highlighter_spend_l241_24121


namespace triangle_area_l241_24157

theorem triangle_area (base height : Real) (h1 : base = 8.4) (h2 : height = 5.8) :
  (base * height) / 2 = 24.36 := by
  sorry

end triangle_area_l241_24157


namespace average_weight_after_student_left_l241_24139

theorem average_weight_after_student_left (initial_count : ℕ) (left_weight : ℝ) 
  (remaining_count : ℕ) (weight_increase : ℝ) (final_average : ℝ) : 
  initial_count = 60 →
  left_weight = 45 →
  remaining_count = 59 →
  weight_increase = 0.2 →
  final_average = 57 →
  (initial_count : ℝ) * (final_average - weight_increase) = 
    (remaining_count : ℝ) * final_average + left_weight := by
  sorry

end average_weight_after_student_left_l241_24139


namespace chocolate_bar_cost_is_two_l241_24171

/-- The cost of a chocolate bar given Frank's purchase information -/
def chocolate_bar_cost (num_bars : ℕ) (num_chips : ℕ) (chips_cost : ℕ) (total_paid : ℕ) (change : ℕ) : ℕ :=
  (total_paid - change - num_chips * chips_cost) / num_bars

/-- Theorem stating that the cost of each chocolate bar is $2 -/
theorem chocolate_bar_cost_is_two :
  chocolate_bar_cost 5 2 3 20 4 = 2 := by
  sorry

end chocolate_bar_cost_is_two_l241_24171


namespace polynomial_value_theorem_l241_24167

/-- Given a polynomial function f(x) = px³ - qx² + rx - s, 
    if f(1) = 4, then 2p + q - 3r + 2s = -8 -/
theorem polynomial_value_theorem (p q r s : ℝ) : 
  let f := fun (x : ℝ) => p * x^3 - q * x^2 + r * x - s
  (f 1 = 4) → (2*p + q - 3*r + 2*s = -8) := by
  sorry

end polynomial_value_theorem_l241_24167


namespace subtracted_amount_l241_24148

theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 100 → 0.7 * N - A = 30 → A = 40 := by sorry

end subtracted_amount_l241_24148


namespace hockey_pad_cost_calculation_l241_24183

def hockey_pad_cost (initial_amount : ℝ) (skate_fraction : ℝ) (remaining : ℝ) : ℝ :=
  initial_amount - initial_amount * skate_fraction - remaining

theorem hockey_pad_cost_calculation :
  hockey_pad_cost 150 (1/2) 25 = 50 := by
  sorry

end hockey_pad_cost_calculation_l241_24183
