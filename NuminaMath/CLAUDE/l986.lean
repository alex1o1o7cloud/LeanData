import Mathlib

namespace quotient_problem_l986_98626

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 158)
  (h2 : divisor = 17)
  (h3 : remainder = 5)
  (h4 : dividend = quotient * divisor + remainder) :
  quotient = 9 := by
  sorry

end quotient_problem_l986_98626


namespace regression_line_equation_l986_98663

/-- Given a regression line with slope 1.23 passing through (4,5), prove its equation is y = 1.23x + 0.08 -/
theorem regression_line_equation (slope : ℝ) (center_x center_y : ℝ) :
  slope = 1.23 →
  center_x = 4 →
  center_y = 5 →
  ∃ (b : ℝ), b = 0.08 ∧ ∀ (x y : ℝ), y = slope * x + b ↔ y - center_y = slope * (x - center_x) :=
sorry

end regression_line_equation_l986_98663


namespace fraction_sum_equals_point_two_l986_98653

theorem fraction_sum_equals_point_two :
  2 / 40 + 4 / 80 + 6 / 120 + 9 / 180 = (0.2 : ℚ) := by
  sorry

end fraction_sum_equals_point_two_l986_98653


namespace function_equation_implies_odd_l986_98616

/-- A non-zero function satisfying the given functional equation is odd -/
theorem function_equation_implies_odd (f : ℝ → ℝ) 
  (h_nonzero : ∃ x, f x ≠ 0)
  (h_eq : ∀ a b : ℝ, f (a * b) = a * f b + b * f a) :
  ∀ x : ℝ, f (-x) = -f x := by
  sorry

end function_equation_implies_odd_l986_98616


namespace gasoline_price_increase_percentage_l986_98656

def lowest_price : ℝ := 15
def highest_price : ℝ := 24

theorem gasoline_price_increase_percentage :
  (highest_price - lowest_price) / lowest_price * 100 = 60 := by
  sorry

end gasoline_price_increase_percentage_l986_98656


namespace joan_sandwiches_l986_98685

/-- Represents the number of sandwiches of each type -/
structure Sandwiches where
  ham : ℕ
  grilledCheese : ℕ

/-- Represents the amount of cheese slices used -/
structure CheeseUsed where
  cheddar : ℕ
  swiss : ℕ
  gouda : ℕ

/-- Calculates the total cheese used for a given number of sandwiches -/
def totalCheeseUsed (s : Sandwiches) : CheeseUsed :=
  { cheddar := s.ham + 2 * s.grilledCheese,
    swiss := s.ham,
    gouda := s.grilledCheese }

/-- The main theorem to prove -/
theorem joan_sandwiches :
  ∃ (s : Sandwiches),
    s.ham = 8 ∧
    totalCheeseUsed s = { cheddar := 40, swiss := 20, gouda := 30 } ∧
    s.grilledCheese = 16 := by
  sorry


end joan_sandwiches_l986_98685


namespace intersection_of_A_and_B_l986_98623

def set_A : Set ℝ := {x | 2 * x < 2 + x}
def set_B : Set ℝ := {x | 5 - x > 8 - 4 * x}

theorem intersection_of_A_and_B : set_A ∩ set_B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l986_98623


namespace smallest_integer_in_set_l986_98624

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 4 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) / 5)) → n ≥ 0 :=
by sorry

end smallest_integer_in_set_l986_98624


namespace power_sum_problem_l986_98612

theorem power_sum_problem (a b x y : ℝ) 
  (h1 : a * x + b * y = 3)
  (h2 : a * x^2 + b * y^2 = 7)
  (h3 : a * x^3 + b * y^3 = 16)
  (h4 : a * x^4 + b * y^4 = 42) :
  a * x^5 + b * y^5 = 20 := by
  sorry

end power_sum_problem_l986_98612


namespace sqrt_product_simplification_l986_98674

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (45 * q) * Real.sqrt (15 * q) * Real.sqrt (10 * q) = 30 * q * Real.sqrt (15 * q) := by
  sorry

end sqrt_product_simplification_l986_98674


namespace min_staircase_steps_l986_98668

theorem min_staircase_steps (a b : ℕ+) :
  ∃ (n : ℕ), n = a + b - Nat.gcd a b ∧
  (∀ (m : ℕ), m < n → ¬∃ (k : ℕ), k * a = m ∨ k * a = m + b) ∧
  (∃ (k l : ℕ), k * a = n ∧ l * a = n + b) :=
sorry

end min_staircase_steps_l986_98668


namespace least_five_digit_congruent_to_6_mod_19_l986_98664

theorem least_five_digit_congruent_to_6_mod_19 :
  ∃ n : ℕ, 
    n ≥ 10000 ∧ 
    n < 100000 ∧ 
    n % 19 = 6 ∧ 
    ∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 19 = 6 → m ≥ n :=
by
  use 10011
  sorry

end least_five_digit_congruent_to_6_mod_19_l986_98664


namespace c_equals_square_l986_98631

/-- The sequence of positive perfect squares -/
def perfect_squares : ℕ → ℕ := λ n => n^2

/-- The nth term of the sequence formed by arranging all positive perfect squares in ascending order -/
def c : ℕ → ℕ := perfect_squares

/-- Theorem: For all positive integers n, c(n) = n^2 -/
theorem c_equals_square (n : ℕ) : c n = n^2 := by sorry

end c_equals_square_l986_98631


namespace total_milks_taken_l986_98658

/-- The total number of milks taken is the sum of all individual milk selections. -/
theorem total_milks_taken (chocolate : ℕ) (strawberry : ℕ) (regular : ℕ) (almond : ℕ) (soy : ℕ)
  (h1 : chocolate = 120)
  (h2 : strawberry = 315)
  (h3 : regular = 230)
  (h4 : almond = 145)
  (h5 : soy = 97) :
  chocolate + strawberry + regular + almond + soy = 907 := by
  sorry

end total_milks_taken_l986_98658


namespace sum_of_coefficients_equals_one_l986_98602

theorem sum_of_coefficients_equals_one (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (2*x - 3)^10 = a + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
                           a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + 
                           a₉*(x-1)^9 + a₁₀*(x-1)^10) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = 1 :=
by
  sorry

end sum_of_coefficients_equals_one_l986_98602


namespace walnut_trees_planted_l986_98655

/-- The number of walnut trees in the park before planting -/
def trees_before : ℕ := 22

/-- The number of walnut trees in the park after planting -/
def trees_after : ℕ := 55

/-- The number of walnut trees planted -/
def trees_planted : ℕ := trees_after - trees_before

theorem walnut_trees_planted : trees_planted = 33 := by
  sorry

end walnut_trees_planted_l986_98655


namespace complex_expression_equals_23_over_150_l986_98647

theorem complex_expression_equals_23_over_150 : 
  let x := (27/8)^(2/3) - (49/9)^(1/2) + 0.008^(2/3) / 0.02^(1/2) * 0.32^(1/2)
  (x / 0.0625^0.25) = 23/150 := by
  sorry

end complex_expression_equals_23_over_150_l986_98647


namespace max_minute_hands_l986_98695

/-- Represents the number of coincidences per hour for a pair of hands moving in opposite directions -/
def coincidences_per_pair : ℕ := 120

/-- Represents the total number of coincidences observed in one hour -/
def total_coincidences : ℕ := 54

/-- Proves that the maximum number of minute hands is 28 given the conditions -/
theorem max_minute_hands : 
  ∃ (m n : ℕ), 
    m * n = total_coincidences / 2 ∧ 
    m + n ≤ 28 ∧ 
    ∀ (k l : ℕ), k * l = total_coincidences / 2 → k + l ≤ m + n :=
by sorry

end max_minute_hands_l986_98695


namespace janet_stickers_l986_98683

theorem janet_stickers (x : ℕ) : 
  x + 53 = 56 → x = 3 := by
  sorry

end janet_stickers_l986_98683


namespace negation_of_proposition_l986_98693

theorem negation_of_proposition (p : Prop) : 
  (¬(∀ a : ℝ, a > 0 ∧ a ≠ 1 → ∃ x : ℝ, a * x - x - a = 0)) ↔ 
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a * x - x - a ≠ 0) :=
by sorry

end negation_of_proposition_l986_98693


namespace root_equation_l986_98637

theorem root_equation (p : ℝ) :
  (0 ≤ p ∧ p ≤ 4/3) →
  (∃! x : ℝ, Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x ∧
             x = (4 - p) / Real.sqrt (8 * (2 - p))) ∧
  (p < 0 ∨ p > 4/3) →
  (∀ x : ℝ, Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) ≠ x) := by
sorry

end root_equation_l986_98637


namespace problem_solution_l986_98630

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a*x

theorem problem_solution (a : ℝ) : f a (f a 0) = 4*a → a = 2 := by
  sorry

end problem_solution_l986_98630


namespace find_n_l986_98635

theorem find_n (a n : ℕ) (h1 : a^2 % n = 8) (h2 : a^3 % n = 25) : n = 113 := by
  sorry

end find_n_l986_98635


namespace book_cost_calculation_l986_98699

def total_cost : ℝ := 6
def num_books : ℕ := 2

theorem book_cost_calculation :
  (total_cost / num_books : ℝ) = 3 := by sorry

end book_cost_calculation_l986_98699


namespace sufficient_not_necessary_l986_98636

theorem sufficient_not_necessary :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end sufficient_not_necessary_l986_98636


namespace square_and_sqrt_properties_l986_98649

theorem square_and_sqrt_properties : 
  let a : ℕ := 10001
  let b : ℕ := 100010001
  let c : ℕ := 1000200030004000300020001
  (a^2 = 100020001) ∧ 
  (b^2 = 10002000300020001) ∧ 
  (c.sqrt = 1000100010001) := by
  sorry

end square_and_sqrt_properties_l986_98649


namespace min_modulus_on_circle_l986_98650

theorem min_modulus_on_circle (z : ℂ) (h : Complex.abs (z - (1 + Complex.I)) = 1) :
  ∃ (w : ℂ), Complex.abs w = Real.sqrt 2 - 1 ∧ 
  ∀ (v : ℂ), Complex.abs (v - (1 + Complex.I)) = 1 → Complex.abs v ≥ Complex.abs w :=
by sorry

end min_modulus_on_circle_l986_98650


namespace money_distribution_l986_98682

theorem money_distribution (A B C : ℕ) 
  (h1 : A + B + C = 500)
  (h2 : A + C = 200)
  (h3 : B + C = 350) : 
  C = 50 := by sorry

end money_distribution_l986_98682


namespace peanut_butter_servings_l986_98661

/-- The amount of peanut butter in the jar, in tablespoons -/
def jar_amount : ℚ := 37 + 4/5

/-- The amount of peanut butter in one serving, in tablespoons -/
def serving_size : ℚ := 1 + 1/2

/-- The number of servings in the jar -/
def number_of_servings : ℚ := jar_amount / serving_size

theorem peanut_butter_servings : number_of_servings = 25 + 1/5 := by
  sorry

end peanut_butter_servings_l986_98661


namespace circular_fountain_area_l986_98697

theorem circular_fountain_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : 
  let AD := AB / 2
  let R := Real.sqrt (AD ^ 2 + DC ^ 2)
  π * R ^ 2 = 244 * π := by sorry

end circular_fountain_area_l986_98697


namespace prob_three_unused_correct_expected_hits_correct_l986_98696

-- Define the probability of hitting a target with a single shot
variable (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1)

-- Define the number of rockets and targets
def num_rockets : ℕ := 10
def num_targets_a : ℕ := 5
def num_targets_b : ℕ := 9

-- Part (a): Probability of exactly three unused rockets
def prob_three_unused : ℝ := 10 * p^3 * (1-p)^2

-- Part (b): Expected number of targets hit
def expected_hits : ℝ := 10*p - p^10

-- Theorem statements
theorem prob_three_unused_correct :
  prob_three_unused p = 10 * p^3 * (1-p)^2 :=
sorry

theorem expected_hits_correct :
  expected_hits p = 10*p - p^10 :=
sorry

end prob_three_unused_correct_expected_hits_correct_l986_98696


namespace ch4_formation_and_consumption_l986_98644

/-- Represents a chemical compound with its coefficient in a reaction --/
structure Compound where
  name : String
  coefficient : ℚ

/-- Represents a chemical reaction --/
structure Reaction where
  reactants : List Compound
  products : List Compound

/-- Represents the initial conditions of the problem --/
structure InitialConditions where
  be2c : ℚ
  h2o : ℚ
  o2 : ℚ
  temperature : ℚ
  pressure : ℚ

/-- The first reaction: Be2C + 4H2O → 2Be(OH)2 + CH4 --/
def reaction1 : Reaction := {
  reactants := [⟨"Be2C", 1⟩, ⟨"H2O", 4⟩],
  products := [⟨"Be(OH)2", 2⟩, ⟨"CH4", 1⟩]
}

/-- The second reaction: CH4 + 2O2 → CO2 + 2H2O --/
def reaction2 : Reaction := {
  reactants := [⟨"CH4", 1⟩, ⟨"O2", 2⟩],
  products := [⟨"CO2", 1⟩, ⟨"H2O", 2⟩]
}

/-- The initial conditions of the problem --/
def initialConditions : InitialConditions := {
  be2c := 3,
  h2o := 15,
  o2 := 6,
  temperature := 350,
  pressure := 2
}

/-- Theorem stating the amount of CH4 formed and remaining --/
theorem ch4_formation_and_consumption 
  (r1 : Reaction)
  (r2 : Reaction)
  (ic : InitialConditions)
  (h1 : r1 = reaction1)
  (h2 : r2 = reaction2)
  (h3 : ic = initialConditions) :
  ∃ (ch4_formed : ℚ) (ch4_remaining : ℚ),
    ch4_formed = 3 ∧ ch4_remaining = 0 :=
  sorry


end ch4_formation_and_consumption_l986_98644


namespace solve_equation_l986_98669

theorem solve_equation :
  ∃ y : ℚ, 2 * y + 3 * y = 500 - (4 * y + 5 * y) → y = 250 / 7 := by
  sorry

end solve_equation_l986_98669


namespace cubic_equation_solution_l986_98665

theorem cubic_equation_solution : 27^3 + 27^3 + 27^3 = 3^10 := by sorry

end cubic_equation_solution_l986_98665


namespace line_through_point_l986_98692

theorem line_through_point (k : ℚ) : 
  (2 * k * 3 - 5 = 4 * (-4)) → k = -11/6 := by
  sorry

end line_through_point_l986_98692


namespace zoo_trip_buses_l986_98600

/-- Given a school trip to the zoo with the following conditions:
  * There are 396 total students
  * 4 students traveled in cars
  * Each bus can hold 56 students
  * All buses were filled
  Prove that the number of buses required is 7. -/
theorem zoo_trip_buses (total_students : ℕ) (car_students : ℕ) (students_per_bus : ℕ) :
  total_students = 396 →
  car_students = 4 →
  students_per_bus = 56 →
  (total_students - car_students) % students_per_bus = 0 →
  (total_students - car_students) / students_per_bus = 7 :=
by sorry

end zoo_trip_buses_l986_98600


namespace intersection_when_m_zero_range_of_m_l986_98657

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B (m : ℝ) : Set ℝ := {x | (x - m + 1) * (x - m - 1) ≥ 0}

-- Define propositions p and q
def p (x : ℝ) : Prop := x ∈ A
def q (x m : ℝ) : Prop := x ∈ B m

-- Theorem 1: Intersection of A and B when m = 0
theorem intersection_when_m_zero : 
  A ∩ B 0 = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Range of m when q is necessary but not sufficient for p
theorem range_of_m (h : ∀ x, p x → q x 0 ∧ ¬(∀ x, q x 0 → p x)) : 
  {m : ℝ | m ≤ -2 ∨ m ≥ 4} = Set.univ := by sorry

end intersection_when_m_zero_range_of_m_l986_98657


namespace part1_part2_l986_98643

/-- Given vectors in R^2 -/
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (2, 1)

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v = (t * w.1, t * w.2)

/-- Theorem for part 1 -/
theorem part1 :
  ∃ (k : ℝ), k = -1/2 ∧ collinear ((k * a.1 - b.1, k * a.2 - b.2)) (a.1 + 2 * b.1, a.2 + 2 * b.2) :=
sorry

/-- Theorem for part 2 -/
theorem part2 :
  ∃ (m : ℝ), m = 3/2 ∧
  (∃ (t : ℝ), (2 * a.1 + 3 * b.1, 2 * a.2 + 3 * b.2) = (t * (a.1 + m * b.1), t * (a.2 + m * b.2))) :=
sorry

end part1_part2_l986_98643


namespace sales_theorem_l986_98690

def sales_problem (last_four_months : List ℕ) (sixth_month : ℕ) (average : ℕ) : Prop :=
  let total_six_months := average * 6
  let sum_last_four := last_four_months.sum
  let first_month := total_six_months - (sum_last_four + sixth_month)
  first_month = 5420

theorem sales_theorem :
  sales_problem [5660, 6200, 6350, 6500] 7070 6200 := by
  sorry

end sales_theorem_l986_98690


namespace similar_triangles_height_l986_98622

theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small > 0 →
  area_ratio = 25 →
  ∃ h_large : ℝ,
    h_large = h_small * Real.sqrt area_ratio ∧
    h_large = 20 :=
by
  sorry

end similar_triangles_height_l986_98622


namespace impossible_equal_sum_configuration_l986_98659

theorem impossible_equal_sum_configuration : ¬ ∃ (a b c d e f : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) ∧
  (a + d + e = b + d + f) ∧
  (a + d + e = c + e + f) ∧
  (b + d + f = c + e + f) :=
by
  sorry

end impossible_equal_sum_configuration_l986_98659


namespace train_length_l986_98662

/-- The length of a train given crossing times and platform length -/
theorem train_length (platform_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
  (h1 : platform_length = 350)
  (h2 : platform_time = 39)
  (h3 : pole_time = 18) :
  (platform_length * pole_time) / (platform_time - pole_time) = 300 := by
sorry

end train_length_l986_98662


namespace expansion_coefficients_theorem_l986_98620

def binomial_expansion (x y : ℤ) (n : ℕ) := (x + y)^n

def max_coefficient (x y : ℤ) (n : ℕ) : ℕ := Nat.choose n (n / 2)

def second_largest_coefficient (x y : ℤ) (n : ℕ) : ℕ :=
  max (Nat.choose n ((n + 1) / 2)) (Nat.choose n ((n - 1) / 2))

theorem expansion_coefficients_theorem :
  let x : ℤ := 2
  let y : ℤ := 8
  let n : ℕ := 8
  max_coefficient x y n = 70 ∧
  second_largest_coefficient x y n = 1792 ∧
  (second_largest_coefficient x y n : ℚ) / (max_coefficient x y n : ℚ) = 128 / 5 := by
  sorry

#check expansion_coefficients_theorem

end expansion_coefficients_theorem_l986_98620


namespace max_m_for_right_angle_l986_98667

-- Define the circle C in polar coordinates
def circle_C (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

-- Define the line l
def line_l (x y m : ℝ) : Prop := y = 2 * x + 2 * m

-- Define the rectangular coordinates of circle C
def circle_C_rect (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2^2

-- Theorem statement
theorem max_m_for_right_angle (m : ℝ) :
  (∃ x y : ℝ, circle_C_rect x y ∧ line_l x y m) →
  m ≤ Real.sqrt 5 - 2 :=
sorry

end max_m_for_right_angle_l986_98667


namespace curve_intersection_tangent_l986_98603

/-- The value of a for which the curves y = a√x and y = ln√x have a common point
    with the same tangent line. -/
theorem curve_intersection_tangent (a : ℝ) (h : a > 0) :
  (∃ x : ℝ, x > 0 ∧ a * Real.sqrt x = Real.log (Real.sqrt x) ∧
    a / (2 * Real.sqrt x) = 1 / (2 * x)) →
  a = Real.exp (-1) := by
sorry

end curve_intersection_tangent_l986_98603


namespace learning_machine_price_reduction_l986_98681

/-- Represents the price reduction scenario of a learning machine -/
def price_reduction_equation (initial_price final_price : ℝ) (num_reductions : ℕ) (x : ℝ) : Prop :=
  initial_price * (1 - x)^num_reductions = final_price

/-- The equation 2000(1-x)^2 = 1280 correctly represents the given price reduction scenario -/
theorem learning_machine_price_reduction :
  price_reduction_equation 2000 1280 2 x ↔ 2000 * (1 - x)^2 = 1280 :=
sorry

end learning_machine_price_reduction_l986_98681


namespace minimum_value_quadratic_l986_98634

theorem minimum_value_quadratic (x : ℝ) :
  (4 * x^2 + 8 * x + 3 = 5) → x ≥ -1 - Real.sqrt 6 / 2 :=
by sorry

end minimum_value_quadratic_l986_98634


namespace system_one_solution_system_two_solution_l986_98627

-- System (1)
theorem system_one_solution (x y : ℝ) : 
  x = 5 - y ∧ x - 3*y = 1 → x = 4 ∧ y = 1 := by sorry

-- System (2)
theorem system_two_solution (x y : ℝ) :
  x - 2*y = 6 ∧ 2*x + 3*y = -2 → x = 2 ∧ y = -2 := by sorry

end system_one_solution_system_two_solution_l986_98627


namespace trig_simplification_l986_98671

theorem trig_simplification :
  (Real.cos (40 * π / 180)) / (Real.cos (25 * π / 180) * Real.sqrt (1 - Real.sin (40 * π / 180))) = Real.sqrt 2 := by
  sorry

end trig_simplification_l986_98671


namespace distinct_permutations_count_l986_98678

def sequence_length : ℕ := 6
def count_of_twos : ℕ := 3
def count_of_sqrt_threes : ℕ := 2
def count_of_fives : ℕ := 1

theorem distinct_permutations_count :
  (sequence_length.factorial) / (count_of_twos.factorial * count_of_sqrt_threes.factorial) = 60 := by
  sorry

end distinct_permutations_count_l986_98678


namespace sector_central_angle_l986_98642

/-- Given a sector with area 1 cm² and perimeter 4 cm, its central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) 
  (h_area : (1/2) * θ * r^2 = 1)
  (h_perimeter : 2*r + θ*r = 4) :
  θ = 2 := by
  sorry

end sector_central_angle_l986_98642


namespace kelly_time_indeterminate_but_longest_l986_98619

/-- Represents the breath-holding contest results -/
structure BreathHoldingContest where
  kelly_time : ℝ
  brittany_time : ℝ
  buffy_time : ℝ
  brittany_kelly_diff : kelly_time - brittany_time = 20
  buffy_time_exact : buffy_time = 120

/-- Kelly's time is indeterminate but greater than Buffy's if she won -/
theorem kelly_time_indeterminate_but_longest (contest : BreathHoldingContest) :
  (∀ t : ℝ, contest.kelly_time ≠ t) ∧
  (contest.kelly_time > contest.buffy_time) :=
sorry

end kelly_time_indeterminate_but_longest_l986_98619


namespace least_comic_books_l986_98691

theorem least_comic_books (n : ℕ) : n > 0 ∧ n % 7 = 3 ∧ n % 4 = 1 → n ≥ 17 :=
by sorry

end least_comic_books_l986_98691


namespace square_perimeter_problem_l986_98611

theorem square_perimeter_problem (A B C : ℝ) : 
  (A > 0) → (B > 0) → (C > 0) →
  (4 * A = 16) → (4 * B = 32) → (C = A + B - 2) →
  (4 * C = 40) := by
sorry

end square_perimeter_problem_l986_98611


namespace sequence_appearance_l986_98654

def sequence_digit (a b c : ℕ) : ℕ :=
  (a + b + c) % 10

def appears_in_sequence (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≥ 3 ∧
  (n / 1000 = sequence_digit 2 1 9) ∧
  (n / 100 % 10 = sequence_digit 1 9 (sequence_digit 2 1 9)) ∧
  (n / 10 % 10 = sequence_digit 9 (sequence_digit 2 1 9) (sequence_digit 1 9 (sequence_digit 2 1 9))) ∧
  (n % 10 = sequence_digit (sequence_digit 2 1 9) (sequence_digit 1 9 (sequence_digit 2 1 9)) (sequence_digit 9 (sequence_digit 2 1 9) (sequence_digit 1 9 (sequence_digit 2 1 9))))

theorem sequence_appearance :
  (¬ appears_in_sequence 1113 ∧ appears_in_sequence 2226 ∧ appears_in_sequence 2125 ∧ appears_in_sequence 2215) ∨
  (appears_in_sequence 1113 ∧ ¬ appears_in_sequence 2226 ∧ appears_in_sequence 2125 ∧ appears_in_sequence 2215) ∨
  (appears_in_sequence 1113 ∧ appears_in_sequence 2226 ∧ ¬ appears_in_sequence 2125 ∧ appears_in_sequence 2215) ∨
  (appears_in_sequence 1113 ∧ appears_in_sequence 2226 ∧ appears_in_sequence 2125 ∧ ¬ appears_in_sequence 2215) :=
by sorry

end sequence_appearance_l986_98654


namespace halfway_fraction_reduced_l986_98646

theorem halfway_fraction_reduced (a b c d e f : ℚ) : 
  a = 3/4 → 
  b = 5/6 → 
  c = (a + b) / 2 → 
  d = 1/12 → 
  e = c - d → 
  f = 17/24 → 
  e = f := by sorry

end halfway_fraction_reduced_l986_98646


namespace tv_price_difference_l986_98688

theorem tv_price_difference (budget : ℝ) (initial_discount : ℝ) (percentage_discount : ℝ) : 
  budget = 1000 →
  initial_discount = 100 →
  percentage_discount = 0.2 →
  budget - (budget - initial_discount) * (1 - percentage_discount) = 280 := by
  sorry

end tv_price_difference_l986_98688


namespace collinear_vectors_x_value_l986_98673

/-- Two vectors are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- The problem statement -/
theorem collinear_vectors_x_value :
  ∀ x : ℝ, collinear (2, 4) (x, 6) → x = 3 := by
sorry

end collinear_vectors_x_value_l986_98673


namespace cone_slant_height_l986_98604

/-- The slant height of a cone with base radius 1 and lateral surface that unfolds into a semicircle -/
def slant_height : ℝ := 2

/-- The base radius of the cone -/
def base_radius : ℝ := 1

/-- Theorem: The slant height of a cone with base radius 1 and lateral surface that unfolds into a semicircle is 2 -/
theorem cone_slant_height :
  let r := base_radius
  let s := slant_height
  r = 1 ∧ 2 * π * r = π * s → s = 2 :=
by sorry

end cone_slant_height_l986_98604


namespace tangent_line_slope_intersecting_line_equation_l986_98625

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 - 4*x + y^2 + 3 = 0

-- Define points P and Q
def P : ℝ × ℝ := (0, 1)
def Q : ℝ × ℝ := (0, -2)

-- Define the condition for the slopes of OA and OB
def slope_condition (k₁ k₂ : ℝ) : Prop := k₁ * k₂ = -1/7

-- Statement for part (1)
theorem tangent_line_slope :
  ∃ m : ℝ, (m = 0 ∨ m = -4/3) ∧
  ∀ x y : ℝ, y = m * x + P.2 →
  (∃! t : ℝ, circle_C t (m * t + P.2)) :=
sorry

-- Statement for part (2)
theorem intersecting_line_equation :
  ∃ k : ℝ, (k = 1 ∨ k = 5/3) ∧
  ∀ x y : ℝ, y = k * x + Q.2 →
  (∃ A B : ℝ × ℝ, 
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧
    A.2 = k * A.1 + Q.2 ∧
    B.2 = k * B.1 + Q.2 ∧
    slope_condition (A.2 / A.1) (B.2 / B.1)) :=
sorry

end tangent_line_slope_intersecting_line_equation_l986_98625


namespace sum_x_y_equals_twenty_l986_98684

theorem sum_x_y_equals_twenty (x y : ℝ) 
  (h1 : |x| - x + y = 13) 
  (h2 : x - |y| + y = 7) : 
  x + y = 20 := by
  sorry

end sum_x_y_equals_twenty_l986_98684


namespace exprC_is_factorization_left_to_right_l986_98652

/-- Represents a polynomial expression -/
structure PolynomialExpression where
  left : ℝ → ℝ → ℝ
  right : ℝ → ℝ → ℝ

/-- Checks if an expression is in product form -/
def isProductForm (expr : ℝ → ℝ → ℝ) : Prop :=
  ∃ (f g : ℝ → ℝ → ℝ), ∀ x y, expr x y = f x y * g x y

/-- Defines factorization from left to right -/
def isFactorizationLeftToRight (expr : PolynomialExpression) : Prop :=
  ¬(isProductForm expr.left) ∧ (isProductForm expr.right)

/-- The specific expression we're examining -/
def exprC : PolynomialExpression :=
  { left := λ a b => a^2 - 4*a*b + 4*b^2,
    right := λ a b => (a - 2*b)^2 }

/-- Theorem stating that exprC represents factorization from left to right -/
theorem exprC_is_factorization_left_to_right :
  isFactorizationLeftToRight exprC :=
sorry

end exprC_is_factorization_left_to_right_l986_98652


namespace other_root_of_quadratic_l986_98677

theorem other_root_of_quadratic (m : ℝ) : 
  (1^2 + m*1 + 3 = 0) → 
  ∃ (α : ℝ), α ≠ 1 ∧ α^2 + m*α + 3 = 0 ∧ α = 3 := by
sorry

end other_root_of_quadratic_l986_98677


namespace fractional_to_linear_equation_l986_98679

/-- Given the fractional equation 2/x = 1/(x-1), prove that multiplying both sides
    by x(x-1) results in a linear equation. -/
theorem fractional_to_linear_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) :
  ∃ (a b : ℝ), x * (x - 1) * (2 / x) = a * x + b :=
sorry

end fractional_to_linear_equation_l986_98679


namespace jonny_sarah_marble_difference_l986_98698

/-- The number of marbles Jonny has -/
def jonny_marbles : ℕ := 18

/-- The number of bags Sarah initially has -/
def sarah_bags : ℕ := 4

/-- The number of marbles in each of Sarah's bags -/
def sarah_marbles_per_bag : ℕ := 6

/-- The total number of marbles Sarah initially has -/
def sarah_total_marbles : ℕ := sarah_bags * sarah_marbles_per_bag

/-- The number of marbles Sarah has after giving half to Jared -/
def sarah_remaining_marbles : ℕ := sarah_total_marbles / 2

theorem jonny_sarah_marble_difference :
  jonny_marbles - sarah_remaining_marbles = 6 := by
  sorry

end jonny_sarah_marble_difference_l986_98698


namespace sum_lent_is_1000_l986_98666

/-- Proves that the sum lent is $1000 given the specified conditions --/
theorem sum_lent_is_1000 (annual_rate : ℝ) (duration : ℝ) (interest_difference : ℝ) :
  annual_rate = 0.06 →
  duration = 8 →
  interest_difference = 520 →
  ∃ (P : ℝ), P * annual_rate * duration = P - interest_difference ∧ P = 1000 := by
  sorry

#check sum_lent_is_1000

end sum_lent_is_1000_l986_98666


namespace student_probability_problem_l986_98607

theorem student_probability_problem (p q : ℝ) 
  (h_p_pos : 0 < p) (h_q_pos : 0 < q) (h_p_le_one : p ≤ 1) (h_q_le_one : q ≤ 1)
  (h_p_gt_q : p > q)
  (h_at_least_one : 1 - (1 - p) * (1 - q) = 5/6)
  (h_both_correct : p * q = 1/3)
  : p = 2/3 ∧ q = 1/2 ∧ 
    (1 - p)^2 * 2 * (1 - q) * q + (1 - p)^2 * q^2 + 2 * (1 - p) * p * q^2 = 7/36 := by
  sorry

end student_probability_problem_l986_98607


namespace six_hundred_million_scientific_notation_l986_98694

-- Define 600 million
def six_hundred_million : ℝ := 600000000

-- Theorem statement
theorem six_hundred_million_scientific_notation :
  six_hundred_million = 6 * 10^8 := by
  sorry

end six_hundred_million_scientific_notation_l986_98694


namespace main_theorem_l986_98618

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x * y) = y * f x + x * f y

/-- The main theorem capturing the problem statements -/
theorem main_theorem (f : ℝ → ℝ) (hf : FunctionalEquation f)
    (a b c d : ℝ) (F : ℝ → ℝ) (hF : ∀ x, F x = a * f x + b * x^5 + c * x^3 + 2 * x^2 + d * x + 3)
    (hF_neg5 : F (-5) = 7) :
    f 0 = 0 ∧ (∀ x, f (-x) = -f x) ∧ F 5 = 99 := by
  sorry


end main_theorem_l986_98618


namespace susan_age_l986_98617

theorem susan_age (susan arthur tom bob : ℕ) 
  (h1 : arthur = susan + 2)
  (h2 : tom = bob - 3)
  (h3 : bob = 11)
  (h4 : susan + arthur + tom + bob = 51) :
  susan = 15 := by
sorry

end susan_age_l986_98617


namespace carol_peanuts_l986_98686

/-- Given that Carol initially collects 2 peanuts and receives 5 more from her father,
    prove that Carol has a total of 7 peanuts. -/
theorem carol_peanuts (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 2 → received = 5 → total = initial + received → total = 7 := by
sorry

end carol_peanuts_l986_98686


namespace exactly_one_and_two_black_mutually_exclusive_not_opposite_l986_98676

-- Define the bag of balls
def bag : Finset (Fin 4) := Finset.univ

-- Define the color of each ball (1 and 2 are red, 3 and 4 are black)
def color : Fin 4 → Bool
  | 1 => false
  | 2 => false
  | 3 => true
  | 4 => true

-- Define a draw as a pair of distinct balls
def Draw := {pair : Fin 4 × Fin 4 // pair.1 ≠ pair.2}

-- Event: Exactly one black ball is drawn
def exactly_one_black (draw : Draw) : Prop :=
  (color draw.val.1 ∧ ¬color draw.val.2) ∨ (¬color draw.val.1 ∧ color draw.val.2)

-- Event: Exactly two black balls are drawn
def exactly_two_black (draw : Draw) : Prop :=
  color draw.val.1 ∧ color draw.val.2

-- Theorem: The events are mutually exclusive but not opposite
theorem exactly_one_and_two_black_mutually_exclusive_not_opposite :
  (∀ draw : Draw, ¬(exactly_one_black draw ∧ exactly_two_black draw)) ∧
  (∃ draw : Draw, ¬exactly_one_black draw ∧ ¬exactly_two_black draw) :=
sorry

end exactly_one_and_two_black_mutually_exclusive_not_opposite_l986_98676


namespace min_sum_of_intercepts_equality_condition_l986_98605

theorem min_sum_of_intercepts (a b : ℝ) : 
  a > 0 → b > 0 → (4 / a + 1 / b = 1) → a + b ≥ 9 := by
  sorry

theorem equality_condition (a b : ℝ) :
  a > 0 → b > 0 → (4 / a + 1 / b = 1) → (a + b = 9) → (a = 6 ∧ b = 3) := by
  sorry

end min_sum_of_intercepts_equality_condition_l986_98605


namespace divisibility_implies_equality_l986_98609

theorem divisibility_implies_equality (a b : ℕ) :
  (∃ S : Set ℕ, Set.Infinite S ∧ ∀ n ∈ S, (a^(n+1) + b^(n+1)) % (a^n + b^n) = 0) →
  a = b :=
sorry

end divisibility_implies_equality_l986_98609


namespace expression_equals_two_l986_98615

theorem expression_equals_two (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_abc : a * b * c = 1) : 
  (1 + a) / (1 + a + a * b) + 
  (1 + b) / (1 + b + b * c) + 
  (1 + c) / (1 + c + c * a) = 2 := by
sorry

end expression_equals_two_l986_98615


namespace brick_width_calculation_l986_98687

/-- Proves that given a courtyard of 25 meters by 16 meters, to be paved with 20,000 bricks of length 20 cm, the width of each brick must be 10 cm. -/
theorem brick_width_calculation (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_length : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 16 →
  brick_length = 0.2 →
  total_bricks = 20000 →
  ∃ (brick_width : ℝ), 
    brick_width = 0.1 ∧ 
    (courtyard_length * courtyard_width * 10000) = (brick_length * brick_width * total_bricks) :=
by sorry

end brick_width_calculation_l986_98687


namespace fraction_simplification_l986_98680

theorem fraction_simplification (m : ℝ) (h : m ≠ 3 ∧ m ≠ -3) : 
  12 / (m^2 - 9) + 2 / (3 - m) = -2 / (m + 3) := by
  sorry

end fraction_simplification_l986_98680


namespace heidi_has_five_more_than_kim_l986_98610

/-- The number of nail polishes each person has -/
structure NailPolishes where
  kim : ℕ
  heidi : ℕ
  karen : ℕ

/-- The conditions of the nail polish problem -/
def nail_polish_problem (np : NailPolishes) : Prop :=
  np.kim = 12 ∧
  np.heidi > np.kim ∧
  np.karen = np.kim - 4 ∧
  np.karen + np.heidi = 25

/-- The theorem stating that Heidi has 5 more nail polishes than Kim -/
theorem heidi_has_five_more_than_kim (np : NailPolishes) 
  (h : nail_polish_problem np) : np.heidi - np.kim = 5 := by
  sorry

end heidi_has_five_more_than_kim_l986_98610


namespace inverse_variation_l986_98606

/-- Given that r and s vary inversely, and s = 0.35 when r = 1200, 
    prove that s = 0.175 when r = 2400 -/
theorem inverse_variation (r s : ℝ) (h : r * s = 1200 * 0.35) :
  r = 2400 → s = 0.175 := by
  sorry

end inverse_variation_l986_98606


namespace number_problem_l986_98672

theorem number_problem : 
  ∃ x : ℚ, x = (3/7)*x + 200 ∧ x = 350 := by
  sorry

end number_problem_l986_98672


namespace sort_table_in_99_moves_l986_98645

/-- Represents a 10x10 table of integers -/
def Table := Fin 10 → Fin 10 → ℤ

/-- Checks if a table is sorted in ascending order both row-wise and column-wise -/
def is_sorted (t : Table) : Prop :=
  ∀ i j k, i < j → t i k < t j k ∧
  ∀ i j k, i < j → t k i < t k j

/-- Represents a rectangular rotation operation on the table -/
def rotate (t : Table) (i j k l : Fin 10) : Table :=
  sorry

/-- The main theorem stating that any table can be sorted in 99 or fewer moves -/
theorem sort_table_in_99_moves (t : Table) :
  (∀ i j k l, t i j ≠ t k l) →  -- All numbers are distinct
  ∃ (moves : List (Fin 10 × Fin 10 × Fin 10 × Fin 10)),
    moves.length ≤ 99 ∧
    is_sorted (moves.foldl (λ acc m => rotate acc m.1 m.2.1 m.2.2.1 m.2.2.2) t) :=
sorry

end sort_table_in_99_moves_l986_98645


namespace jakes_allowance_l986_98639

/-- 
Given:
- An amount x (in cents)
- One-quarter of x can buy 5 items
- Each item costs 20 cents

Prove that x = 400 cents ($4.00)
-/
theorem jakes_allowance (x : ℕ) 
  (h1 : x / 4 = 5 * 20) : 
  x = 400 := by
sorry

end jakes_allowance_l986_98639


namespace six_people_eight_chairs_two_restricted_l986_98638

/-- The number of ways to arrange n people in r chairs -/
def arrangements (n r : ℕ) : ℕ := n.factorial

/-- The number of ways to choose r chairs from n chairs -/
def chair_selections (n r : ℕ) : ℕ := n.choose r

/-- The number of ways to seat people in chairs with restrictions -/
def seating_arrangements (total_chairs people : ℕ) (restricted_pairs : ℕ) : ℕ :=
  (chair_selections total_chairs people - restricted_pairs) * arrangements people people

theorem six_people_eight_chairs_two_restricted : 
  seating_arrangements 8 6 30 = 18720 := by
  sorry

end six_people_eight_chairs_two_restricted_l986_98638


namespace max_sum_reciprocal_zeros_l986_98608

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if x > 1 then k * x + 1
  else 0

theorem max_sum_reciprocal_zeros (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 ∧ 1/x₁ + 1/x₂ ≤ 9/4) ∧
  (∃ k₀ : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k₀ x₁ = 0 ∧ f k₀ x₂ = 0 ∧ 1/x₁ + 1/x₂ = 9/4) :=
by sorry

end max_sum_reciprocal_zeros_l986_98608


namespace initial_white_cookies_l986_98675

def cookie_problem (w : ℕ) : Prop :=
  let b := w + 50
  let remaining_black := b / 2
  let remaining_white := w / 4
  remaining_black + remaining_white = 85

theorem initial_white_cookies : ∃ w : ℕ, cookie_problem w ∧ w = 80 := by
  sorry

end initial_white_cookies_l986_98675


namespace vessel_mixture_theorem_main_vessel_theorem_l986_98628

/-- Represents the contents of a vessel -/
structure VesselContents where
  milk : ℚ
  water : ℚ

/-- Represents a vessel with its volume and contents -/
structure Vessel where
  volume : ℚ
  contents : VesselContents

/-- Theorem stating the relationship between vessel contents and their mixture -/
theorem vessel_mixture_theorem 
  (v1 v2 : Vessel) 
  (h1 : v1.volume / v2.volume = 3 / 5)
  (h2 : v2.contents.milk / v2.contents.water = 3 / 2)
  (h3 : (v1.contents.milk + v2.contents.milk) / (v1.contents.water + v2.contents.water) = 1) :
  v1.contents.milk / v1.contents.water = 1 / 2 := by
  sorry

/-- Main theorem proving the equivalence of the vessel mixture problem -/
theorem main_vessel_theorem :
  ∀ (v1 v2 : Vessel),
  v1.volume / v2.volume = 3 / 5 →
  v2.contents.milk / v2.contents.water = 3 / 2 →
  (v1.contents.milk + v2.contents.milk) / (v1.contents.water + v2.contents.water) = 1 ↔
  v1.contents.milk / v1.contents.water = 1 / 2 := by
  sorry

end vessel_mixture_theorem_main_vessel_theorem_l986_98628


namespace total_legs_on_farm_l986_98689

/-- The number of legs for each animal type -/
def legs_per_animal (animal : String) : ℕ :=
  match animal with
  | "chicken" => 2
  | "sheep" => 4
  | _ => 0

/-- The total number of animals on the farm -/
def total_animals : ℕ := 12

/-- The number of chickens on the farm -/
def num_chickens : ℕ := 5

/-- Theorem stating the total number of animal legs on the farm -/
theorem total_legs_on_farm : 
  (num_chickens * legs_per_animal "chicken") + 
  ((total_animals - num_chickens) * legs_per_animal "sheep") = 38 := by
  sorry

end total_legs_on_farm_l986_98689


namespace larger_number_proof_l986_98621

theorem larger_number_proof (a b : ℕ+) 
  (hcf_eq : Nat.gcd a b = 28)
  (lcm_eq : Nat.lcm a b = 28 * 12 * 15) :
  max a b = 180 := by
  sorry

end larger_number_proof_l986_98621


namespace no_solution_exists_l986_98632

/-- Given a third-degree polynomial P(x) = x^3 - mx^2 + nx + 42,
    where (x + 6) and (x - a + bi) are factors,
    with a and b being real numbers and b ≠ 0,
    prove that there are no real values for m, n, a, and b
    that satisfy all conditions simultaneously. -/
theorem no_solution_exists (m n a b : ℝ) : b ≠ 0 →
  (∀ x, x^3 - m*x^2 + n*x + 42 = (x + 6) * (x - a + b*I) * (x - a - b*I)) →
  False := by sorry

end no_solution_exists_l986_98632


namespace sum_of_x_and_y_l986_98633

theorem sum_of_x_and_y (x y : ℝ) (some_number : ℝ) 
  (h1 : x + y = some_number) 
  (h2 : x - y = 5) 
  (h3 : x = 10) 
  (h4 : y = 5) : 
  x + y = 15 := by
  sorry

end sum_of_x_and_y_l986_98633


namespace square_sum_plus_sum_squares_l986_98640

theorem square_sum_plus_sum_squares : (5 + 9)^2 + (5^2 + 9^2) = 302 := by sorry

end square_sum_plus_sum_squares_l986_98640


namespace sum_of_squares_of_roots_l986_98614

theorem sum_of_squares_of_roots (a b c : ℚ) (h : a = 6 ∧ b = 9 ∧ c = -21) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = 37/4 := by sorry

end sum_of_squares_of_roots_l986_98614


namespace unique_solution_for_equation_l986_98648

theorem unique_solution_for_equation (n : ℕ+) (p : ℕ) : 
  Nat.Prime p → (n : ℕ)^8 - (n : ℕ)^2 = p^5 + p^2 → (n = 2 ∧ p = 3) :=
by sorry

end unique_solution_for_equation_l986_98648


namespace fraction_positivity_implies_x_range_l986_98670

theorem fraction_positivity_implies_x_range (x : ℝ) : (-6 : ℝ) / (7 - x) > 0 → x > 7 := by
  sorry

end fraction_positivity_implies_x_range_l986_98670


namespace sufficient_not_necessary_l986_98651

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ ¬(a > 1)) := by
  sorry

end sufficient_not_necessary_l986_98651


namespace inequality_proof_root_mean_square_arithmetic_mean_l986_98629

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

theorem root_mean_square_arithmetic_mean (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end inequality_proof_root_mean_square_arithmetic_mean_l986_98629


namespace infinite_geometric_series_l986_98601

/-- Given an infinite geometric series with first term a and sum S,
    prove the common ratio r and the second term -/
theorem infinite_geometric_series
  (a : ℝ) (S : ℝ) (h_a : a = 540) (h_S : S = 4500) :
  ∃ (r : ℝ),
    r = 0.88 ∧
    S = a / (1 - r) ∧
    abs r < 1 ∧
    a * r = 475.2 := by
  sorry

end infinite_geometric_series_l986_98601


namespace valid_sequences_count_l986_98660

/-- A transformation on a regular hexagon -/
inductive HexagonTransform
| T1  -- 60° clockwise rotation
| T2  -- 60° counterclockwise rotation
| T3  -- reflection across x-axis
| T4  -- reflection across y-axis

/-- A sequence of transformations -/
def TransformSequence := List HexagonTransform

/-- The identity transformation -/
def identity : TransformSequence := []

/-- Applies a single transformation to a sequence -/
def applyTransform (t : HexagonTransform) (s : TransformSequence) : TransformSequence :=
  t :: s

/-- Checks if a sequence of transformations results in the identity transformation -/
def isIdentity (s : TransformSequence) : Bool :=
  sorry

/-- Counts the number of valid 18-transformation sequences -/
def countValidSequences : Nat :=
  sorry

/-- Main theorem: There are 286 valid sequences of 18 transformations -/
theorem valid_sequences_count : countValidSequences = 286 := by
  sorry

end valid_sequences_count_l986_98660


namespace middle_circle_number_l986_98641

def numbers : List ℕ := [1, 5, 6, 7, 13, 14, 17, 22, 26]

def middle_fixed : List ℕ := [13, 17]

def total_sum : ℕ := numbers.sum

def group_sum : ℕ := total_sum / 3

theorem middle_circle_number (x : ℕ) 
  (h1 : x ∈ numbers)
  (h2 : ∀ (a b c : ℕ), a ∈ numbers → b ∈ numbers → c ∈ numbers → 
       a ≠ b → b ≠ c → a ≠ c → 
       (a + b + c = group_sum) → 
       (a = 13 ∧ b = 17) ∨ (a = 13 ∧ c = 17) ∨ (b = 13 ∧ c = 17) → 
       x = c ∨ x = b) :
  x = 7 := by sorry

end middle_circle_number_l986_98641


namespace perpendicular_and_tangent_l986_98613

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x - 6 * y + 1 = 0

-- Define the curve
def curve (x y : ℝ) : Prop := y = x^3 + 3 * x^2 - 1

-- Define the perpendicular line (our answer)
def perp_line (x y : ℝ) : Prop := 3 * x + y + 2 = 0

-- State the theorem
theorem perpendicular_and_tangent :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    curve x₀ y₀ ∧
    -- The point (x₀, y₀) is on the perpendicular line
    perp_line x₀ y₀ ∧
    -- The perpendicular line is indeed perpendicular to the given line
    (3 : ℝ) * (1 / 3 : ℝ) = -1 ∧
    -- The slope of the curve at (x₀, y₀) equals the slope of the perpendicular line
    (3 * x₀^2 + 6 * x₀) = -3 :=
by
  sorry


end perpendicular_and_tangent_l986_98613
