import Mathlib

namespace find_p_value_l1530_153040

theorem find_p_value (a b c p : ℝ) 
  (h1 : 9 / (a + b) = p / (a + c)) 
  (h2 : p / (a + c) = 13 / (c - b)) : 
  p = 22 := by
sorry

end find_p_value_l1530_153040


namespace soaking_solution_l1530_153095

/-- Represents the time needed to soak clothes for each type of stain -/
structure SoakingTime where
  grass : ℕ
  marinara : ℕ

/-- Conditions for the soaking problem -/
def soaking_problem (t : SoakingTime) : Prop :=
  t.marinara = t.grass + 7 ∧ 
  3 * t.grass + t.marinara = 19

/-- Theorem stating the solution to the soaking problem -/
theorem soaking_solution :
  ∃ (t : SoakingTime), soaking_problem t ∧ t.grass = 3 := by
  sorry

end soaking_solution_l1530_153095


namespace range_of_f_less_than_zero_l1530_153062

-- Define the properties of the function f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def decreasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 0 → f y ≤ f x

-- State the theorem
theorem range_of_f_less_than_zero
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_decreasing : decreasing_on_nonpositive f)
  (h_f_neg_two : f (-2) = 0) :
  {x : ℝ | f x < 0} = Set.Ioo (-2) 2 :=
sorry

end range_of_f_less_than_zero_l1530_153062


namespace triangle_area_l1530_153024

/-- Given a right triangle with side lengths in the ratio 2:3:4 inscribed in a circle of radius 4,
    prove that its area is 12. -/
theorem triangle_area (a b c : ℝ) (r : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Ensure positive side lengths
  r = 4 →  -- Circle radius
  a^2 + b^2 = c^2 →  -- Right triangle condition
  c = 2 * r →  -- Hypotenuse is diameter
  b / a = 3 / 2 →  -- Side ratio condition
  c / a = 2 →  -- Side ratio condition
  (1 / 2) * a * b = 12 :=  -- Area formula
by sorry

end triangle_area_l1530_153024


namespace johnny_first_job_hours_l1530_153020

/-- Represents Johnny's work schedule and earnings --/
structure WorkSchedule where
  hourlyRate1 : ℝ
  hourlyRate2 : ℝ
  hourlyRate3 : ℝ
  hours2 : ℝ
  hours3 : ℝ
  daysWorked : ℝ
  totalEarnings : ℝ

/-- Theorem stating that given the conditions, Johnny worked 3 hours on the first job each day --/
theorem johnny_first_job_hours (schedule : WorkSchedule)
  (h1 : schedule.hourlyRate1 = 7)
  (h2 : schedule.hourlyRate2 = 10)
  (h3 : schedule.hourlyRate3 = 12)
  (h4 : schedule.hours2 = 2)
  (h5 : schedule.hours3 = 4)
  (h6 : schedule.daysWorked = 5)
  (h7 : schedule.totalEarnings = 445) :
  ∃ (x : ℝ), x = 3 ∧ 
    schedule.daysWorked * (schedule.hourlyRate1 * x + 
      schedule.hourlyRate2 * schedule.hours2 + 
      schedule.hourlyRate3 * schedule.hours3) = schedule.totalEarnings :=
by
  sorry

end johnny_first_job_hours_l1530_153020


namespace acute_angle_measure_l1530_153042

theorem acute_angle_measure (x : ℝ) : 
  0 < x → x < 90 → (90 - x = (180 - x) / 2 + 20) → x = 40 := by
  sorry

end acute_angle_measure_l1530_153042


namespace license_plate_difference_l1530_153006

/-- The number of possible digits in a license plate -/
def num_digits : ℕ := 10

/-- The number of possible letters in a license plate -/
def num_letters : ℕ := 26

/-- The number of possible license plates for Alpha (LLDDDDLL format) -/
def alpha_plates : ℕ := num_letters^4 * num_digits^4

/-- The number of possible license plates for Beta (LLLDDDD format) -/
def beta_plates : ℕ := num_letters^3 * num_digits^4

/-- The theorem stating the difference in number of license plates between Alpha and Beta -/
theorem license_plate_difference :
  alpha_plates - beta_plates = num_digits^4 * num_letters^3 * 25 := by
  sorry

#eval alpha_plates - beta_plates
#eval num_digits^4 * num_letters^3 * 25

end license_plate_difference_l1530_153006


namespace total_amount_is_2500_l1530_153052

/-- Proves that the total amount of money divided into two parts is 2500,
    given the conditions from the original problem. -/
theorem total_amount_is_2500 
  (total : ℝ) 
  (part1 : ℝ) 
  (part2 : ℝ) 
  (h1 : total = part1 + part2)
  (h2 : part1 = 1000)
  (h3 : 0.05 * part1 + 0.06 * part2 = 140) :
  total = 2500 := by
  sorry

end total_amount_is_2500_l1530_153052


namespace quadrilateral_inequality_l1530_153004

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the measure of an angle
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

theorem quadrilateral_inequality (ABCD : Quadrilateral) :
  length ABCD.A ABCD.D = length ABCD.B ABCD.C →
  angle_measure ABCD.A ABCD.D ABCD.C > angle_measure ABCD.B ABCD.C ABCD.D →
  length ABCD.A ABCD.C > length ABCD.B ABCD.D :=
by sorry

end quadrilateral_inequality_l1530_153004


namespace bhaskara_solution_l1530_153028

/-- The number of people in Bhaskara's money distribution problem -/
def bhaskara_problem (n : ℕ) : Prop :=
  let initial_sum := n * (2 * 3 + (n - 1) * 1) / 2
  let redistribution_sum := 100 * n
  initial_sum = redistribution_sum

theorem bhaskara_solution :
  ∃ n : ℕ, n > 0 ∧ bhaskara_problem n ∧ n = 195 := by
  sorry

end bhaskara_solution_l1530_153028


namespace horner_method_operations_l1530_153049

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The number of operations required by Horner's method -/
def horner_operations (n : ℕ) : ℕ × ℕ :=
  (n, n)

/-- The polynomial f(x) = 5x^5 + 4x^4 + 3x^3 + 2x^2 + x + 1 -/
def f_coeffs : List ℝ := [5, 4, 3, 2, 1, 1]

theorem horner_method_operations :
  let (mults, adds) := horner_operations (f_coeffs.length - 1)
  mults ≤ 5 ∧ adds = 5 := by sorry

end horner_method_operations_l1530_153049


namespace expression_evaluation_l1530_153068

theorem expression_evaluation :
  let a : ℝ := 40
  let c : ℝ := 4
  1891 - (1600 / a + 8040 / a) * c = 927 := by
  sorry

end expression_evaluation_l1530_153068


namespace boat_rowing_probability_l1530_153063

theorem boat_rowing_probability : 
  let p_left1 : ℚ := 3/5  -- Probability of first left oar working
  let p_left2 : ℚ := 2/5  -- Probability of second left oar working
  let p_right1 : ℚ := 4/5  -- Probability of first right oar working
  let p_right2 : ℚ := 3/5  -- Probability of second right oar working
  
  -- Probability of both left oars failing
  let p_left_fail : ℚ := (1 - p_left1) * (1 - p_left2)
  
  -- Probability of both right oars failing
  let p_right_fail : ℚ := (1 - p_right1) * (1 - p_right2)
  
  -- Probability of all four oars failing
  let p_all_fail : ℚ := p_left_fail * p_right_fail
  
  -- Probability of being able to row the boat
  let p_row : ℚ := 1 - (p_left_fail + p_right_fail - p_all_fail)
  
  p_row = 437/625 := by sorry

end boat_rowing_probability_l1530_153063


namespace sector_area_l1530_153055

theorem sector_area (r : ℝ) (θ : ℝ) (h : r = 2) (k : θ = π / 3) :
  (1 / 2) * r^2 * θ = (2 * π) / 3 := by
  sorry

end sector_area_l1530_153055


namespace inequality_proof_l1530_153070

theorem inequality_proof (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_sum : a + b + c + d = 1) : 
  (a^2 / (1 + a)) + (b^2 / (1 + b)) + (c^2 / (1 + c)) + (d^2 / (1 + d)) ≥ 1/5 := by
  sorry

end inequality_proof_l1530_153070


namespace tank_capacity_l1530_153098

theorem tank_capacity
  (bucket1_capacity bucket2_capacity : ℕ)
  (bucket1_uses bucket2_uses : ℕ)
  (h1 : bucket1_capacity = 4)
  (h2 : bucket2_capacity = 3)
  (h3 : bucket2_uses = bucket1_uses + 4)
  (h4 : bucket1_capacity * bucket1_uses = bucket2_capacity * bucket2_uses) :
  bucket1_capacity * bucket1_uses = 48 :=
by sorry

end tank_capacity_l1530_153098


namespace number_of_molecules_value_l1530_153060

/-- The number of molecules in a given substance -/
def number_of_molecules : ℕ := 3 * 10^26

/-- Theorem stating that the number of molecules is 3 · 10^26 -/
theorem number_of_molecules_value : number_of_molecules = 3 * 10^26 := by
  sorry

end number_of_molecules_value_l1530_153060


namespace pentagon_rectangle_ratio_l1530_153089

/-- The ratio of the side length of a regular pentagon to the width of a rectangle with the same perimeter -/
theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side rectangle_width : ℝ),
  pentagon_side > 0 → 
  rectangle_width > 0 →
  5 * pentagon_side = 20 →
  6 * rectangle_width = 20 →
  pentagon_side / rectangle_width = 6 / 5 := by
sorry

end pentagon_rectangle_ratio_l1530_153089


namespace diamond_equation_solution_l1530_153026

-- Define the diamond operation
def diamond (x y : ℝ) : ℝ := 5 * x - 2 * y + 2 * x * y

-- State the theorem
theorem diamond_equation_solution :
  ∃! y : ℝ, diamond 4 y = 30 ∧ y = 5/3 := by sorry

end diamond_equation_solution_l1530_153026


namespace square_brush_ratio_l1530_153030

theorem square_brush_ratio (s w : ℝ) (h_positive_s : 0 < s) (h_positive_w : 0 < w) :
  (w^2 + 2 * (s^2 / 2 - w^2) = s^2 / 3) → (s / w = Real.sqrt 3) := by
  sorry

end square_brush_ratio_l1530_153030


namespace same_remainder_mod_27_l1530_153019

/-- Given a six-digit number X, Y is formed by moving the first three digits of X after the last three digits -/
def form_Y (X : ℕ) : ℕ :=
  let a := X / 1000
  let b := X % 1000
  1000 * b + a

/-- Theorem: For any six-digit number X, X and Y (formed from X) have the same remainder when divided by 27 -/
theorem same_remainder_mod_27 (X : ℕ) (h : 100000 ≤ X ∧ X < 1000000) :
  X % 27 = form_Y X % 27 := by
  sorry


end same_remainder_mod_27_l1530_153019


namespace unique_solution_condition_l1530_153075

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3 * x + 8) * (x - 6) = -50 + k * x) ↔ 
  (k = -10 + 2 * Real.sqrt 6 ∨ k = -10 - 2 * Real.sqrt 6) :=
sorry

end unique_solution_condition_l1530_153075


namespace nested_fraction_evaluation_l1530_153074

theorem nested_fraction_evaluation : 
  2 + (1 / (2 + (1 / (2 + 2)))) = 22 / 9 := by
  sorry

end nested_fraction_evaluation_l1530_153074


namespace minimize_function_l1530_153056

theorem minimize_function (x y z : ℝ) : 
  (3 * x + 2 * y + z = 3) →
  (x^2 + y^2 + 2 * z^2 ≥ 2/3) →
  (x^2 + y^2 + 2 * z^2 = 2/3 → x * y / z = 8/3) :=
by sorry

end minimize_function_l1530_153056


namespace x1_value_l1530_153046

theorem x1_value (x1 x2 x3 x4 : Real) 
  (h_order : 0 ≤ x4 ∧ x4 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1) 
  (h_eq : (1-x1)^2 + (x1-x2)^2 + (x2-x3)^2 + (x3-x4)^2 + x4^2 = 1/5) :
  x1 = 4/5 := by
  sorry

end x1_value_l1530_153046


namespace intersection_distance_l1530_153090

/-- The distance between the points of intersection of x^2 + y = 12 and x + y = 12 is √2 -/
theorem intersection_distance : ∃ (p1 p2 : ℝ × ℝ),
  (p1.1^2 + p1.2 = 12 ∧ p1.1 + p1.2 = 12) ∧
  (p2.1^2 + p2.2 = 12 ∧ p2.1 + p2.2 = 12) ∧
  p1 ≠ p2 ∧
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = Real.sqrt 2 := by
  sorry


end intersection_distance_l1530_153090


namespace toaster_pricing_theorem_l1530_153058

/-- Represents the relationship between cost and number of purchasers for toasters -/
def toaster_relation (c p : ℝ) : Prop := c * p = 6000

theorem toaster_pricing_theorem :
  -- Given condition
  toaster_relation 300 20 →
  -- Proofs to show
  (toaster_relation 600 10 ∧ toaster_relation 400 15) :=
by
  sorry

end toaster_pricing_theorem_l1530_153058


namespace price_is_400_l1530_153072

/-- The price per phone sold by Aliyah and Vivienne -/
def price_per_phone (vivienne_phones : ℕ) (aliyah_extra_phones : ℕ) (total_revenue : ℕ) : ℚ :=
  total_revenue / (vivienne_phones + (vivienne_phones + aliyah_extra_phones))

/-- Theorem stating that the price per phone is $400 -/
theorem price_is_400 :
  price_per_phone 40 10 36000 = 400 := by
  sorry

end price_is_400_l1530_153072


namespace sqrt_inequality_sum_reciprocal_inequality_l1530_153027

-- Problem 1
theorem sqrt_inequality : Real.sqrt 3 + Real.sqrt 8 < 2 + Real.sqrt 7 := by sorry

-- Problem 2
theorem sum_reciprocal_inequality {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 1/a + 1/b + 1/c ≥ 9 := by sorry

end sqrt_inequality_sum_reciprocal_inequality_l1530_153027


namespace extremum_implies_a_equals_negative_four_l1530_153003

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 - a*x^2 - b*x + a^2 - 6*a

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 - 2*a*x - b

-- Theorem statement
theorem extremum_implies_a_equals_negative_four (a b : ℝ) :
  f' a b 2 = 0 ∧ f a b 2 = 8 → a = -4 :=
by sorry

end extremum_implies_a_equals_negative_four_l1530_153003


namespace greg_trousers_count_l1530_153051

/-- The cost of a shirt -/
def shirtCost : ℝ := sorry

/-- The cost of a pair of trousers -/
def trousersCost : ℝ := sorry

/-- The cost of a tie -/
def tieCost : ℝ := sorry

/-- The number of trousers Greg bought in the first scenario -/
def firstScenarioTrousers : ℕ := sorry

theorem greg_trousers_count :
  (6 * shirtCost + firstScenarioTrousers * trousersCost + 2 * tieCost = 80) ∧
  (4 * shirtCost + 2 * trousersCost + 2 * tieCost = 140) ∧
  (5 * shirtCost + 3 * trousersCost + 2 * tieCost = 110) →
  firstScenarioTrousers = 4 := by
  sorry

end greg_trousers_count_l1530_153051


namespace counterexample_exists_l1530_153048

theorem counterexample_exists : ∃ x : ℝ, x > 1 ∧ x + 1 / (x - 1) ≤ 3 := by
  sorry

end counterexample_exists_l1530_153048


namespace constant_term_exists_l1530_153084

/-- Represents the derivative of a function q with respect to some variable -/
def derivative (q : ℝ → ℝ) : ℝ → ℝ := sorry

/-- The equation q' = 3q - 3 -/
def equation (q : ℝ → ℝ) : Prop :=
  ∀ x, derivative q x = 3 * q x - 3

/-- The value of (4')' is 72 -/
def condition (q : ℝ → ℝ) : Prop :=
  derivative (derivative q) 4 = 72

/-- There exists a constant term in the equation -/
theorem constant_term_exists (q : ℝ → ℝ) (h1 : equation q) (h2 : condition q) :
  ∃ c : ℝ, ∀ x, derivative q x = 3 * q x + c :=
sorry

end constant_term_exists_l1530_153084


namespace max_difference_when_a_is_one_sum_geq_three_iff_abs_a_geq_one_l1530_153071

-- Define the functions f and g
def f (a x : ℝ) : ℝ := |x - 2*a|
def g (a x : ℝ) : ℝ := |x + a|

-- Part 1
theorem max_difference_when_a_is_one :
  ∃ (m : ℝ), ∀ (x : ℝ), f 1 x - g 1 x ≤ m ∧ ∃ (y : ℝ), f 1 y - g 1 y = m ∧ m = 3 :=
sorry

-- Part 2
theorem sum_geq_three_iff_abs_a_geq_one (a : ℝ) :
  (∀ x : ℝ, f a x + g a x ≥ 3) ↔ |a| ≥ 1 :=
sorry

end max_difference_when_a_is_one_sum_geq_three_iff_abs_a_geq_one_l1530_153071


namespace ratio_theorem_max_coeff_theorem_l1530_153031

open Real

/-- The ratio of the sum of all coefficients to the sum of all binomial coefficients
    in the expansion of (x^(2/3) + 3x^2)^n is 32 -/
def ratio_condition (n : ℕ) : Prop :=
  (4 : ℝ)^n / (2 : ℝ)^n = 32

/-- The value of n that satisfies the ratio condition -/
def n_value : ℕ := 5

/-- Theorem stating that n_value satisfies the ratio condition -/
theorem ratio_theorem : ratio_condition n_value := by
  sorry

/-- The terms with maximum binomial coefficient in the expansion -/
def max_coeff_terms (x : ℝ) : ℝ × ℝ :=
  (90 * x^6, 270 * x^(22/3))

/-- Theorem stating that max_coeff_terms gives the correct terms -/
theorem max_coeff_theorem (x : ℝ) :
  max_coeff_terms x = (90 * x^6, 270 * x^(22/3)) := by
  sorry

end ratio_theorem_max_coeff_theorem_l1530_153031


namespace hexagon_diagonals_l1530_153017

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: A hexagon has 9 internal diagonals -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end hexagon_diagonals_l1530_153017


namespace cricket_bat_cost_price_l1530_153076

theorem cricket_bat_cost_price 
  (profit_a_to_b : Real) 
  (profit_b_to_c : Real) 
  (price_c_pays : Real) : 
  profit_a_to_b = 0.20 →
  profit_b_to_c = 0.25 →
  price_c_pays = 234 →
  ∃ (cost_price_a : Real), 
    cost_price_a = 156 ∧ 
    price_c_pays = (1 + profit_b_to_c) * ((1 + profit_a_to_b) * cost_price_a) :=
by sorry

end cricket_bat_cost_price_l1530_153076


namespace max_square_plots_l1530_153047

/-- Represents the dimensions of the park and available fencing --/
structure ParkData where
  width : ℕ
  length : ℕ
  fencing : ℕ

/-- Represents a potential partitioning of the park --/
structure Partitioning where
  sideLength : ℕ
  numPlots : ℕ

/-- Checks if a partitioning is valid for the given park data --/
def isValidPartitioning (park : ParkData) (part : Partitioning) : Prop :=
  part.sideLength > 0 ∧
  park.width % part.sideLength = 0 ∧
  park.length % part.sideLength = 0 ∧
  part.numPlots = (park.width / part.sideLength) * (park.length / part.sideLength) ∧
  (park.width / part.sideLength - 1) * park.length + (park.length / part.sideLength - 1) * park.width ≤ park.fencing

/-- Theorem stating that the maximum number of square plots is 2 --/
theorem max_square_plots (park : ParkData) 
  (h_width : park.width = 30)
  (h_length : park.length = 60)
  (h_fencing : park.fencing = 2400) :
  (∀ p : Partitioning, isValidPartitioning park p → p.numPlots ≤ 2) ∧
  (∃ p : Partitioning, isValidPartitioning park p ∧ p.numPlots = 2) :=
sorry

end max_square_plots_l1530_153047


namespace mary_sheep_theorem_l1530_153061

def initial_sheep : ℕ := 1500

def sister_percentage : ℚ := 1/4
def brother_percentage : ℚ := 3/10
def cousin_fraction : ℚ := 1/7

def remaining_sheep : ℕ := 676

theorem mary_sheep_theorem :
  let sheep_after_sister := initial_sheep - ⌊initial_sheep * sister_percentage⌋
  let sheep_after_brother := sheep_after_sister - ⌊sheep_after_sister * brother_percentage⌋
  let sheep_after_cousin := sheep_after_brother - ⌊sheep_after_brother * cousin_fraction⌋
  sheep_after_cousin = remaining_sheep := by sorry

end mary_sheep_theorem_l1530_153061


namespace rectangle_existence_l1530_153013

theorem rectangle_existence (m : ℕ) (h : m > 12) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y > m ∧ x * (y - 1) < m ∧ x ≤ y := by
  sorry

end rectangle_existence_l1530_153013


namespace weight_change_problem_l1530_153016

/-- Represents the scenario of replacing a man in a group and the resulting weight change -/
structure WeightChangeScenario where
  initial_count : ℕ
  initial_average : ℝ
  replaced_weight : ℝ
  new_weight : ℝ
  average_increase : ℝ

/-- The theorem representing the weight change problem -/
theorem weight_change_problem (scenario : WeightChangeScenario) 
  (h1 : scenario.initial_count = 10)
  (h2 : scenario.replaced_weight = 58)
  (h3 : scenario.average_increase = 2.5) :
  scenario.new_weight = 83 ∧ 
  ∀ (x : ℝ), ∃ (scenario' : WeightChangeScenario), 
    scenario'.initial_average = x ∧
    scenario'.initial_count = scenario.initial_count ∧
    scenario'.replaced_weight = scenario.replaced_weight ∧
    scenario'.new_weight = scenario.new_weight ∧
    scenario'.average_increase = scenario.average_increase :=
by sorry

end weight_change_problem_l1530_153016


namespace cubic_function_range_l1530_153083

/-- A cubic function f(x) = ax³ + bx² + cx + d satisfying given conditions -/
structure CubicFunction where
  f : ℝ → ℝ
  cubic : ∃ (a b c d : ℝ), ∀ x, f x = a * x^3 + b * x^2 + c * x + d
  cond1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2
  cond2 : 1 ≤ f 1 ∧ f 1 ≤ 3
  cond3 : 2 ≤ f 2 ∧ f 2 ≤ 4
  cond4 : -1 ≤ f 3 ∧ f 3 ≤ 1

/-- The value of f(4) is always within the range [-21¾, 1] for any CubicFunction -/
theorem cubic_function_range (cf : CubicFunction) :
  -21.75 ≤ cf.f 4 ∧ cf.f 4 ≤ 1 := by
  sorry

end cubic_function_range_l1530_153083


namespace max_pyramid_volume_l1530_153085

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a pyramid with vertex O and base ABC -/
structure Pyramid where
  O : Point3D
  A : Point3D
  B : Point3D
  C : Point3D

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point3D) : ℝ := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Calculates the volume of a pyramid -/
def pyramidVolume (p : Pyramid) : ℝ := sorry

/-- Checks if a point is on the surface of a sphere -/
def isOnSphere (center : Point3D) (radius : ℝ) (point : Point3D) : Prop :=
  distance center point = radius

theorem max_pyramid_volume (p : Pyramid) (r : ℝ) :
  r = 3 →
  isOnSphere p.O r p.A →
  isOnSphere p.O r p.B →
  isOnSphere p.O r p.C →
  angle (p.A) (p.B) = 150 * π / 180 →
  ∀ (q : Pyramid), 
    isOnSphere p.O r q.A →
    isOnSphere p.O r q.B →
    isOnSphere p.O r q.C →
    pyramidVolume q ≤ 9/2 :=
by sorry

end max_pyramid_volume_l1530_153085


namespace third_term_geometric_sequence_l1530_153093

theorem third_term_geometric_sequence
  (q : ℝ)
  (h_q_abs : |q| < 1)
  (h_sum : (a : ℕ → ℝ) → (∀ n, a (n + 1) = q * a n) → (∑' n, a n) = 8/5)
  (h_second_term : ∃ a : ℕ → ℝ, (∀ n, a (n + 1) = q * a n) ∧ a 1 = -1/2) :
  ∃ a : ℕ → ℝ, (∀ n, a (n + 1) = q * a n) ∧ a 1 = -1/2 ∧ a 2 = 1/8 :=
sorry

end third_term_geometric_sequence_l1530_153093


namespace yamaimo_moving_problem_l1530_153014

/-- The Yamaimo family's moving problem -/
theorem yamaimo_moving_problem (initial_weight : ℝ) (initial_book_percentage : ℝ) 
  (final_book_percentage : ℝ) (new_weight : ℝ) : 
  initial_weight = 100 →
  initial_book_percentage = 99 / 100 →
  final_book_percentage = 95 / 100 →
  initial_weight * initial_book_percentage = 
    new_weight * final_book_percentage →
  initial_weight * (1 - initial_book_percentage) = 
    new_weight * (1 - final_book_percentage) →
  new_weight = 20 := by
  sorry

end yamaimo_moving_problem_l1530_153014


namespace simplify_expression_l1530_153032

theorem simplify_expression (x : ℝ) : 5*x + 6 - x + 12 = 4*x + 18 := by
  sorry

end simplify_expression_l1530_153032


namespace candy_distribution_l1530_153096

theorem candy_distribution (total_candy : ℕ) (num_students : ℕ) (pieces_per_student : ℕ) : 
  total_candy = 344 → num_students = 43 → 
  pieces_per_student * num_students = total_candy →
  pieces_per_student = 8 := by
sorry

end candy_distribution_l1530_153096


namespace subset_condition_l1530_153097

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | x > a}

theorem subset_condition (a : ℝ) : A ⊆ B a → a < -2 := by
  sorry

end subset_condition_l1530_153097


namespace matrix_is_square_iff_a_eq_zero_l1530_153011

def A (a : ℚ) : Matrix (Fin 4) (Fin 4) ℚ :=
  !![a,   -a,  -1,   0;
     a,   -a,   0,  -1;
     1,    0,   a,  -a;
     0,    1,   a,  -a]

theorem matrix_is_square_iff_a_eq_zero (a : ℚ) :
  (∃ C : Matrix (Fin 4) (Fin 4) ℚ, A a = C ^ 2) ↔ a = 0 := by sorry

end matrix_is_square_iff_a_eq_zero_l1530_153011


namespace locus_is_hexagon_l1530_153043

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a triangle in 3D space
structure Triangle3D where
  A : Point3D
  B : Point3D
  C : Point3D

-- Define a function to check if a triangle is acute-angled
def isAcuteTriangle (t : Triangle3D) : Prop :=
  sorry

-- Define a function to check if a point forms acute-angled triangles with all sides of the base triangle
def formsAcuteTriangles (P : Point3D) (base : Triangle3D) : Prop :=
  sorry

-- Define the locus of points
def locusOfPoints (base : Triangle3D) : Set Point3D :=
  {P | formsAcuteTriangles P base}

-- Theorem statement
theorem locus_is_hexagon (base : Triangle3D) 
  (h : isAcuteTriangle base) : 
  ∃ (hexagon : Set Point3D), locusOfPoints base = hexagon :=
sorry

end locus_is_hexagon_l1530_153043


namespace smallest_integer_y_minus_five_smallest_l1530_153023

theorem smallest_integer_y (y : ℤ) : (7 - 3 * y < 25) ↔ y ≥ -5 := by sorry

theorem minus_five_smallest : ∃ (y : ℤ), (7 - 3 * y < 25) ∧ (∀ (z : ℤ), z < y → (7 - 3 * z ≥ 25)) := by sorry

end smallest_integer_y_minus_five_smallest_l1530_153023


namespace set_equality_implies_values_l1530_153050

theorem set_equality_implies_values (x y : ℝ) : 
  ({1, x, y} : Set ℝ) = {x, x^2, x*y} → x = -1 ∧ y = 0 := by
  sorry

end set_equality_implies_values_l1530_153050


namespace nine_million_squared_zeros_l1530_153029

/-- For a positive integer n, represent a number composed of n nines -/
def all_nines (n : ℕ) : ℕ := 10^n - 1

/-- The number of zeros in the expansion of (all_nines n)² -/
def num_zeros (n : ℕ) : ℕ := n - 1

theorem nine_million_squared_zeros :
  ∃ (k : ℕ), all_nines 7 ^ 2 = k * 10^6 + m ∧ m < 10^6 :=
sorry

end nine_million_squared_zeros_l1530_153029


namespace unique_solution_trigonometric_equation_l1530_153092

theorem unique_solution_trigonometric_equation :
  ∃! (x : ℝ), 0 < x ∧ x < 1 ∧ Real.sin (Real.arccos (Real.tan (Real.arcsin x))) = x :=
by sorry

end unique_solution_trigonometric_equation_l1530_153092


namespace largest_prime_factor_of_expression_l1530_153021

theorem largest_prime_factor_of_expression : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (16^4 + 2 * 16^2 + 1 - 15^4) ∧ 
    ∀ q : ℕ, Nat.Prime q → q ∣ (16^4 + 2 * 16^2 + 1 - 15^4) → q ≤ p) ∧
  (Nat.Prime 241 ∧ 241 ∣ (16^4 + 2 * 16^2 + 1 - 15^4)) := by
  sorry

end largest_prime_factor_of_expression_l1530_153021


namespace max_value_of_sum_of_square_roots_l1530_153088

theorem max_value_of_sum_of_square_roots (a b c : ℝ) 
  (non_neg_a : 0 ≤ a) (non_neg_b : 0 ≤ b) (non_neg_c : 0 ≤ c)
  (sum_condition : a + b + c = 1) :
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) ≤ Real.sqrt 21 := by
sorry

end max_value_of_sum_of_square_roots_l1530_153088


namespace petes_age_proof_l1530_153091

/-- Pete's current age -/
def petes_age : ℕ := 35

/-- Pete's son's current age -/
def sons_age : ℕ := 9

/-- Years into the future when the age comparison is made -/
def years_later : ℕ := 4

theorem petes_age_proof :
  petes_age = 35 ∧
  sons_age = 9 ∧
  petes_age + years_later = 3 * (sons_age + years_later) :=
by sorry

end petes_age_proof_l1530_153091


namespace union_of_A_and_B_l1530_153038

-- Define the sets A and B
def A : Set ℕ := {2, 1, 3}
def B : Set ℕ := {2, 3, 5}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5} := by
  sorry

end union_of_A_and_B_l1530_153038


namespace smallest_q_is_31_l1530_153053

theorem smallest_q_is_31 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : q = 15 * p + 1) :
  q ≥ 31 :=
sorry

end smallest_q_is_31_l1530_153053


namespace lisas_marbles_problem_l1530_153079

/-- The minimum number of additional marbles needed -/
def min_additional_marbles (num_friends : ℕ) (initial_marbles : ℕ) : ℕ :=
  (num_friends * (num_friends + 1)) / 2 - initial_marbles

/-- Theorem stating the minimum number of additional marbles needed for Lisa's problem -/
theorem lisas_marbles_problem (num_friends : ℕ) (initial_marbles : ℕ) 
    (h1 : num_friends = 12) (h2 : initial_marbles = 34) : 
    min_additional_marbles num_friends initial_marbles = 44 := by
  sorry

#eval min_additional_marbles 12 34

end lisas_marbles_problem_l1530_153079


namespace greg_granola_bars_l1530_153099

/-- Proves that Greg set aside 1 granola bar for each day of the week --/
theorem greg_granola_bars (total : ℕ) (traded : ℕ) (sisters : ℕ) (bars_per_sister : ℕ) (days : ℕ)
  (h_total : total = 20)
  (h_traded : traded = 3)
  (h_sisters : sisters = 2)
  (h_bars_per_sister : bars_per_sister = 5)
  (h_days : days = 7) :
  (total - traded - sisters * bars_per_sister) / days = 1 := by
  sorry

end greg_granola_bars_l1530_153099


namespace smallest_divisible_by_1_to_12_l1530_153059

theorem smallest_divisible_by_1_to_12 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 12 → k ∣ m) → n ≤ m) ∧ n = 27720 := by
  sorry

end smallest_divisible_by_1_to_12_l1530_153059


namespace max_k_value_l1530_153039

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (heq : 4 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * ((x / y) + (y / x))) :
  k ≤ (3/2) ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 4 = (3/2)^2 * ((x^2 / y^2) + (y^2 / x^2)) + (3/2) * ((x / y) + (y / x)) :=
sorry

end max_k_value_l1530_153039


namespace binomial_square_constant_l1530_153009

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 60*x + c = (x + a)^2) → c = 900 := by
  sorry

end binomial_square_constant_l1530_153009


namespace rita_trust_fund_growth_l1530_153078

/-- Calculates the final amount in a trust fund after compound interest -/
def trustFundGrowth (initialInvestment : ℝ) (interestRate : ℝ) (years : ℕ) : ℝ :=
  initialInvestment * (1 + interestRate) ^ years

/-- Theorem stating the approximate value of Rita's trust fund after 25 years -/
theorem rita_trust_fund_growth :
  let initialInvestment : ℝ := 5000
  let interestRate : ℝ := 0.03
  let years : ℕ := 25
  let finalAmount := trustFundGrowth initialInvestment interestRate years
  ∃ ε > 0, abs (finalAmount - 10468.87) < ε :=
by
  sorry


end rita_trust_fund_growth_l1530_153078


namespace lex_apple_count_l1530_153005

/-- The total number of apples Lex picked -/
def total_apples : ℕ := 85

/-- The number of apples with worms -/
def wormy_apples : ℕ := total_apples / 5

/-- The number of bruised apples -/
def bruised_apples : ℕ := total_apples / 5 + 9

/-- The number of apples left to eat raw -/
def raw_apples : ℕ := 42

theorem lex_apple_count :
  wormy_apples + bruised_apples + raw_apples = total_apples :=
by sorry

end lex_apple_count_l1530_153005


namespace wall_volume_is_86436_l1530_153054

def wall_volume (width : ℝ) : ℝ :=
  let height := 6 * width
  let length := 7 * height
  width * height * length

theorem wall_volume_is_86436 :
  wall_volume 7 = 86436 :=
by sorry

end wall_volume_is_86436_l1530_153054


namespace average_of_multiples_of_10_l1530_153034

def multiples_of_10 : List ℕ := List.range 30 |>.map (fun n => 10 * (n + 1))

theorem average_of_multiples_of_10 :
  (List.sum multiples_of_10) / (List.length multiples_of_10) = 155 := by
  sorry

#eval (List.sum multiples_of_10) / (List.length multiples_of_10)

end average_of_multiples_of_10_l1530_153034


namespace average_decrease_l1530_153080

theorem average_decrease (initial_count : ℕ) (initial_avg : ℚ) (new_obs : ℚ) :
  initial_count = 6 →
  initial_avg = 13 →
  new_obs = 6 →
  let total_sum := initial_count * initial_avg
  let new_sum := total_sum + new_obs
  let new_count := initial_count + 1
  let new_avg := new_sum / new_count
  initial_avg - new_avg = 1 := by
sorry

end average_decrease_l1530_153080


namespace end_of_year_deposits_l1530_153001

/-- Accumulated capital for end-of-year deposits given beginning-of-year deposits -/
theorem end_of_year_deposits (P r : ℝ) (n : ℕ) (K : ℝ) :
  P > 0 → r > 0 → n > 0 →
  K = P * ((1 + r/100)^n - 1) / (r/100) * (1 + r/100) →
  ∃ K', K' = P * ((1 + r/100)^n - 1) / (r/100) ∧ K' = K / (1 + r/100) := by
  sorry

end end_of_year_deposits_l1530_153001


namespace no_common_solution_l1530_153018

theorem no_common_solution :
  ¬∃ x : ℚ, (6 * (x - 2/3) - (x + 7) = 11) ∧ ((2*x - 1)/3 = (2*x + 1)/6 - 2) := by
  sorry

end no_common_solution_l1530_153018


namespace optimal_profit_is_1368_l1530_153073

/-- Represents the types of apples -/
inductive AppleType
| A
| B
| C

/-- Represents the configuration of cars for each apple type -/
structure CarConfiguration where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- The total number of cars -/
def totalCars : ℕ := 40

/-- The total tons of apples -/
def totalTons : ℕ := 200

/-- Returns the tons per car for a given apple type -/
def tonsPerCar (t : AppleType) : ℕ :=
  match t with
  | AppleType.A => 6
  | AppleType.B => 5
  | AppleType.C => 4

/-- Returns the profit per ton for a given apple type -/
def profitPerTon (t : AppleType) : ℕ :=
  match t with
  | AppleType.A => 5
  | AppleType.B => 7
  | AppleType.C => 8

/-- Checks if a car configuration is valid -/
def isValidConfiguration (config : CarConfiguration) : Prop :=
  config.typeA + config.typeB + config.typeC = totalCars ∧
  config.typeA * tonsPerCar AppleType.A + 
  config.typeB * tonsPerCar AppleType.B + 
  config.typeC * tonsPerCar AppleType.C = totalTons ∧
  config.typeA ≥ 4 ∧ config.typeB ≥ 4 ∧ config.typeC ≥ 4

/-- Calculates the profit for a given car configuration -/
def calculateProfit (config : CarConfiguration) : ℕ :=
  config.typeA * tonsPerCar AppleType.A * profitPerTon AppleType.A +
  config.typeB * tonsPerCar AppleType.B * profitPerTon AppleType.B +
  config.typeC * tonsPerCar AppleType.C * profitPerTon AppleType.C

/-- The optimal car configuration -/
def optimalConfig : CarConfiguration :=
  { typeA := 4, typeB := 32, typeC := 4 }

theorem optimal_profit_is_1368 :
  isValidConfiguration optimalConfig ∧
  calculateProfit optimalConfig = 1368 ∧
  ∀ (config : CarConfiguration), 
    isValidConfiguration config → 
    calculateProfit config ≤ calculateProfit optimalConfig :=
by sorry

end optimal_profit_is_1368_l1530_153073


namespace complex_equation_proof_l1530_153025

def complex_i : ℂ := Complex.I

theorem complex_equation_proof (z : ℂ) (h : z = 1 + complex_i) : 
  2 / z + z^2 = 1 + complex_i := by
  sorry

end complex_equation_proof_l1530_153025


namespace roll_five_probability_l1530_153069

/-- A cube with six faces -/
structure Cube where
  faces : Fin 6 → ℕ

/-- The specific cube described in the problem -/
def problemCube : Cube :=
  { faces := λ i => match i with
    | ⟨0, _⟩ => 1
    | ⟨1, _⟩ => 1
    | ⟨2, _⟩ => 2
    | ⟨3, _⟩ => 4
    | ⟨4, _⟩ => 5
    | ⟨5, _⟩ => 5
    | _ => 0 }

/-- The probability of rolling a specific number on the cube -/
def rollProbability (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i = n) Finset.univ).card / 6

/-- Theorem stating that the probability of rolling a 5 on the problem cube is 1/3 -/
theorem roll_five_probability :
  rollProbability problemCube 5 = 1/3 := by
  sorry


end roll_five_probability_l1530_153069


namespace series_sum_is_zero_l1530_153066

open Real
open Topology
open Tendsto

noncomputable def series_sum : ℝ := ∑' n, (3 * n + 4) / ((n + 1) * (n + 2) * (n + 3))

theorem series_sum_is_zero : series_sum = 0 := by sorry

end series_sum_is_zero_l1530_153066


namespace decimal_point_problem_l1530_153007

theorem decimal_point_problem (x : ℝ) (h1 : x > 0) (h2 : 1000 * x = 5 * (1 / x)) : 
  x = Real.sqrt 2 / 20 := by
  sorry

end decimal_point_problem_l1530_153007


namespace find_B_l1530_153082

theorem find_B (A C B : ℤ) (h1 : A = 520) (h2 : C = A + 204) (h3 : C = B + 179) : B = 545 := by
  sorry

end find_B_l1530_153082


namespace three_non_congruent_triangles_l1530_153012

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  perimeter_eq : a + b + c = 11
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

/-- Two triangles are congruent if they have the same side lengths (up to permutation) -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all valid integer triangles with perimeter 11 -/
def valid_triangles : Set IntTriangle :=
  {t : IntTriangle | True}

/-- The theorem to be proved -/
theorem three_non_congruent_triangles :
  ∃ (t1 t2 t3 : IntTriangle),
    t1 ∈ valid_triangles ∧ t2 ∈ valid_triangles ∧ t3 ∈ valid_triangles ∧
    ¬(congruent t1 t2) ∧ ¬(congruent t2 t3) ∧ ¬(congruent t1 t3) ∧
    ∀ (t : IntTriangle), t ∈ valid_triangles →
      congruent t t1 ∨ congruent t t2 ∨ congruent t t3 :=
by sorry

end three_non_congruent_triangles_l1530_153012


namespace mice_ratio_l1530_153010

theorem mice_ratio (white_mice brown_mice : ℕ) 
  (hw : white_mice = 14) 
  (hb : brown_mice = 7) : 
  (white_mice : ℚ) / (white_mice + brown_mice) = 2 / 3 := by
  sorry

end mice_ratio_l1530_153010


namespace sqrt_sum_bounds_l1530_153035

theorem sqrt_sum_bounds : 
  let n : ℝ := Real.sqrt 4 + Real.sqrt 7
  4 < n ∧ n < 5 := by sorry

end sqrt_sum_bounds_l1530_153035


namespace triangle_side_length_l1530_153064

/-- Given a triangle ABC with side lengths a = 2, b = 1, and angle C = 60°, 
    the length of side c is √3. -/
theorem triangle_side_length (a b c : ℝ) (C : ℝ) : 
  a = 2 → b = 1 → C = Real.pi / 3 → c = Real.sqrt 3 := by
  sorry

end triangle_side_length_l1530_153064


namespace point_quadrant_l1530_153086

/-- Given that point P(-4a, 2+b) is in the third quadrant, prove that point Q(a, b) is in the fourth quadrant -/
theorem point_quadrant (a b : ℝ) :
  (-4 * a < 0 ∧ 2 + b < 0) → (a > 0 ∧ b < 0) :=
by sorry

end point_quadrant_l1530_153086


namespace quadratic_polynomials_exist_l1530_153094

/-- A quadratic polynomial ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The number of real roots of a quadratic polynomial -/
def num_real_roots (p : QuadraticPolynomial) : ℕ :=
  sorry

/-- The sum of two quadratic polynomials -/
def add (p q : QuadraticPolynomial) : QuadraticPolynomial :=
  ⟨p.a + q.a, p.b + q.b, p.c + q.c⟩

theorem quadratic_polynomials_exist : ∃ (f g h : QuadraticPolynomial),
  (num_real_roots f = 2) ∧
  (num_real_roots g = 2) ∧
  (num_real_roots h = 2) ∧
  (num_real_roots (add f g) = 1) ∧
  (num_real_roots (add f h) = 1) ∧
  (num_real_roots (add g h) = 1) ∧
  (num_real_roots (add (add f g) h) = 0) :=
sorry

end quadratic_polynomials_exist_l1530_153094


namespace arithmetic_geometric_sequence_sum_l1530_153067

theorem arithmetic_geometric_sequence_sum (a : ℕ → ℤ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a_1, a_3, a_4 form a geometric sequence
  a 2 + a 3 = -10 :=            -- conclusion to prove
by sorry

end arithmetic_geometric_sequence_sum_l1530_153067


namespace inequality_proof_l1530_153008

theorem inequality_proof (x : Real) (h : 0 < x ∧ x < Real.pi / 2) :
  Real.sin (Real.cos x) < Real.cos x ∧ Real.cos x < Real.cos (Real.sin x) := by
  sorry

end inequality_proof_l1530_153008


namespace min_production_quantity_to_break_even_l1530_153015

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the sales revenue function
def sales_revenue (x : ℝ) : ℝ := 250 * x

-- Define the break-even condition
def breaks_even (x : ℝ) : Prop := sales_revenue x ≥ total_cost x

-- Theorem statement
theorem min_production_quantity_to_break_even :
  ∃ (x : ℝ), x = 150 ∧ x ∈ Set.Ioo 0 240 ∧ breaks_even x ∧
  ∀ (y : ℝ), y ∈ Set.Ioo 0 240 → breaks_even y → y ≥ x :=
sorry

end min_production_quantity_to_break_even_l1530_153015


namespace graph_not_in_first_quadrant_l1530_153036

-- Define the function
def f (k x : ℝ) : ℝ := k * (x - k)

-- Theorem statement
theorem graph_not_in_first_quadrant (k : ℝ) (h : k < 0) :
  ∀ x y : ℝ, f k x = y → ¬(x > 0 ∧ y > 0) :=
by sorry

end graph_not_in_first_quadrant_l1530_153036


namespace circle_tangency_problem_l1530_153077

theorem circle_tangency_problem (C D : ℕ) : 
  C = 144 →
  (∃ (S : Finset ℕ), S = {s : ℕ | s < C ∧ C % s = 0 ∧ s ≠ C} ∧ S.card = 14) :=
by sorry

end circle_tangency_problem_l1530_153077


namespace albert_needs_twelve_dollars_l1530_153000

/-- The amount of additional money Albert needs to buy his art supplies -/
def additional_money_needed (paintbrush_cost paint_cost easel_cost current_money : ℚ) : ℚ :=
  paintbrush_cost + paint_cost + easel_cost - current_money

/-- Theorem stating that Albert needs $12 more -/
theorem albert_needs_twelve_dollars :
  additional_money_needed 1.50 4.35 12.65 6.50 = 12 := by
  sorry

end albert_needs_twelve_dollars_l1530_153000


namespace equation_solution_l1530_153087

theorem equation_solution (x : ℚ) : 
  (1 : ℚ) / 3 + 1 / x = (3 : ℚ) / 4 → x = 12 / 5 := by
  sorry

end equation_solution_l1530_153087


namespace updated_mean_after_decrement_l1530_153045

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 6 →
  (n * original_mean - n * decrement) / n = 194 := by
  sorry

end updated_mean_after_decrement_l1530_153045


namespace gamma_interval_for_f_l1530_153057

def f (x : ℝ) : ℝ := -x * abs x + 2 * x

def is_gamma_interval (f : ℝ → ℝ) (m n : ℝ) : Prop :=
  ∀ y ∈ Set.Icc (1 / n) (1 / m), ∃ x ∈ Set.Icc m n, f x = y

theorem gamma_interval_for_f :
  let m : ℝ := 1
  let n : ℝ := (1 + Real.sqrt 5) / 2
  m < n ∧ 
  Set.Icc m n ⊆ Set.Ioi 1 ∧ 
  is_gamma_interval f m n := by sorry

end gamma_interval_for_f_l1530_153057


namespace most_cost_effective_plan_l1530_153022

/-- Represents the capacity and rental cost of a truck type -/
structure TruckType where
  capacity : ℕ
  rentalCost : ℕ

/-- Represents a rental plan -/
structure RentalPlan where
  typeA : ℕ
  typeB : ℕ

def totalCapacity (a b : TruckType) (plan : RentalPlan) : ℕ :=
  a.capacity * plan.typeA + b.capacity * plan.typeB

def totalCost (a b : TruckType) (plan : RentalPlan) : ℕ :=
  a.rentalCost * plan.typeA + b.rentalCost * plan.typeB

/-- The main theorem stating the most cost-effective rental plan -/
theorem most_cost_effective_plan 
  (typeA typeB : TruckType)
  (h1 : 2 * typeA.capacity + typeB.capacity = 10)
  (h2 : typeA.capacity + 2 * typeB.capacity = 11)
  (h3 : typeA.rentalCost = 100)
  (h4 : typeB.rentalCost = 120) :
  ∃ (plan : RentalPlan),
    totalCapacity typeA typeB plan = 31 ∧
    (∀ (otherPlan : RentalPlan),
      totalCapacity typeA typeB otherPlan = 31 →
      totalCost typeA typeB plan ≤ totalCost typeA typeB otherPlan) ∧
    plan.typeA = 1 ∧
    plan.typeB = 7 ∧
    totalCost typeA typeB plan = 940 :=
  sorry

end most_cost_effective_plan_l1530_153022


namespace units_digit_of_quotient_l1530_153044

theorem units_digit_of_quotient (n : ℕ) (h : 5 ∣ (2^1993 + 3^1993)) :
  (2^1993 + 3^1993) / 5 % 10 = 3 := by
  sorry

end units_digit_of_quotient_l1530_153044


namespace twins_age_product_difference_l1530_153002

theorem twins_age_product_difference : 
  ∀ (current_age : ℕ), 
    current_age = 6 → 
    (current_age + 1) * (current_age + 1) - current_age * current_age = 13 := by
  sorry

end twins_age_product_difference_l1530_153002


namespace smallest_number_l1530_153065

theorem smallest_number (a b c d : ℤ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -1) (h4 : d = -3) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c :=
by sorry

end smallest_number_l1530_153065


namespace sum_of_possible_values_l1530_153041

theorem sum_of_possible_values (e f : ℚ) : 
  (2 * |2 - e| = 5 ∧ |3 * e + f| = 7) → 
  (∃ e₁ f₁ e₂ f₂ : ℚ, 
    (2 * |2 - e₁| = 5 ∧ |3 * e₁ + f₁| = 7) ∧
    (2 * |2 - e₂| = 5 ∧ |3 * e₂ + f₂| = 7) ∧
    e₁ + f₁ + e₂ + f₂ = 6) :=
by sorry

end sum_of_possible_values_l1530_153041


namespace geometric_sequence_third_term_l1530_153037

/-- Given a geometric sequence {aₙ} where a₁ = -2 and a₅ = -4, prove that a₃ = -2√2 -/
theorem geometric_sequence_third_term (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a 2 / a 1) 
  (h_a1 : a 1 = -2) (h_a5 : a 5 = -4) : a 3 = -2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_third_term_l1530_153037


namespace algebraic_expressions_proof_l1530_153081

theorem algebraic_expressions_proof (a b : ℝ) 
  (ha : a = Real.sqrt 5 + 1) 
  (hb : b = Real.sqrt 5 - 1) : 
  (a^2 * b + a * b^2 = 8 * Real.sqrt 5) ∧ 
  (a^2 - a * b + b^2 = 8) := by
sorry

end algebraic_expressions_proof_l1530_153081


namespace prob_two_even_correct_l1530_153033

/-- The total number of balls -/
def total_balls : ℕ := 17

/-- The number of even-numbered balls -/
def even_balls : ℕ := 8

/-- The probability of drawing two even-numbered balls without replacement -/
def prob_two_even : ℚ := 7 / 34

theorem prob_two_even_correct :
  (even_balls : ℚ) / total_balls * (even_balls - 1) / (total_balls - 1) = prob_two_even := by
  sorry

end prob_two_even_correct_l1530_153033
