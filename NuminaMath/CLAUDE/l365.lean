import Mathlib

namespace cube_root_eight_over_sqrt_two_equals_sqrt_two_l365_36568

theorem cube_root_eight_over_sqrt_two_equals_sqrt_two : 
  (8 : ℝ)^(1/3) / (2 : ℝ)^(1/2) = (2 : ℝ)^(1/2) := by sorry

end cube_root_eight_over_sqrt_two_equals_sqrt_two_l365_36568


namespace inequality_proof_l365_36576

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by sorry

end inequality_proof_l365_36576


namespace total_bulbs_is_469_l365_36508

/-- Represents the number of lights of each type -/
structure LightCounts where
  tiny : ℕ
  small : ℕ
  medium : ℕ
  large : ℕ
  extraLarge : ℕ

/-- Calculates the total number of bulbs needed -/
def totalBulbs (counts : LightCounts) : ℕ :=
  counts.tiny * 1 + counts.small * 2 + counts.medium * 3 + counts.large * 4 + counts.extraLarge * 5

theorem total_bulbs_is_469 (counts : LightCounts) :
  counts.large = 2 * counts.medium →
  counts.small = (5 * counts.medium) / 4 →
  counts.extraLarge = counts.small - counts.tiny →
  4 * counts.tiny = 3 * counts.medium →
  2 * counts.small + 3 * counts.medium = 4 * counts.large + 5 * counts.extraLarge →
  counts.extraLarge = 14 →
  totalBulbs counts = 469 := by
  sorry

#eval totalBulbs { tiny := 21, small := 35, medium := 28, large := 56, extraLarge := 14 }

end total_bulbs_is_469_l365_36508


namespace arithmetic_sequence_sum_l365_36541

theorem arithmetic_sequence_sum (a : ℝ) :
  (a + 6 * 2 = 20) →  -- seventh term is 20
  (a + 2 + a = 18)    -- sum of first two terms is 18
:= by sorry

end arithmetic_sequence_sum_l365_36541


namespace solve_equation_l365_36590

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
  sorry

end solve_equation_l365_36590


namespace container_capacity_l365_36517

theorem container_capacity (C : ℝ) 
  (h1 : 0.30 * C + 9 = 0.75 * C) : C = 20 := by
  sorry

end container_capacity_l365_36517


namespace fraction_simplification_l365_36599

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end fraction_simplification_l365_36599


namespace system_solution_ratio_l365_36533

theorem system_solution_ratio (k x y z : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + 2*k*y + 4*z = 0 →
  4*x + k*y - 3*z = 0 →
  3*x + 5*y - 4*z = 0 →
  x^2 * z / y^3 = 125 := by
sorry

end system_solution_ratio_l365_36533


namespace f_value_at_four_l365_36589

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then x^a else |x - 2|

theorem f_value_at_four (a : ℝ) :
  (f a (-2) = f a 2) → f a 4 = 16 := by
  sorry

end f_value_at_four_l365_36589


namespace unique_integer_satisfying_inequality_l365_36529

theorem unique_integer_satisfying_inequality : 
  ∃! (n : ℕ), n > 0 ∧ (105 * n : ℝ)^30 > (n : ℝ)^90 ∧ (n : ℝ)^90 > 3^180 :=
by sorry

end unique_integer_satisfying_inequality_l365_36529


namespace dice_sum_theorem_l365_36575

/-- Represents a single die -/
structure Die where
  opposite_sum : ℕ
  opposite_sum_is_seven : opposite_sum = 7

/-- Represents a set of 7 dice -/
structure DiceSet where
  dice : Fin 7 → Die
  all_dice_have_opposite_sum_seven : ∀ i, (dice i).opposite_sum = 7

/-- The sum of numbers on the upward faces of a set of dice -/
def upward_sum (d : DiceSet) : ℕ := sorry

/-- The sum of numbers on the downward faces of a set of dice -/
def downward_sum (d : DiceSet) : ℕ := sorry

/-- The probability of getting a specific sum on the upward faces -/
noncomputable def prob_upward_sum (d : DiceSet) (sum : ℕ) : ℝ := sorry

/-- The probability of getting a specific sum on the downward faces -/
noncomputable def prob_downward_sum (d : DiceSet) (sum : ℕ) : ℝ := sorry

theorem dice_sum_theorem (d : DiceSet) (a : ℕ) 
  (h1 : a ≠ 10)
  (h2 : prob_upward_sum d 10 = prob_downward_sum d a) :
  a = 39 := by sorry

end dice_sum_theorem_l365_36575


namespace total_popsicles_l365_36502

theorem total_popsicles (grape : ℕ) (cherry : ℕ) (banana : ℕ) 
  (h1 : grape = 2) (h2 : cherry = 13) (h3 : banana = 2) : 
  grape + cherry + banana = 17 := by
  sorry

end total_popsicles_l365_36502


namespace system_one_solution_system_two_solution_l365_36546

-- System (1)
theorem system_one_solution (x y : ℚ) :
  (4 * x - 3 * y = 1 ∧ 3 * x - 2 * y = -1) ↔ (x = -5 ∧ y = 7) :=
sorry

-- System (2)
theorem system_two_solution (x y : ℚ) :
  ((y + 1) / 4 = (x + 2) / 3 ∧ 2 * x - 3 * y = 1) ↔ (x = -3 ∧ y = -7/3) :=
sorry

end system_one_solution_system_two_solution_l365_36546


namespace max_not_joined_company_l365_36501

/-- The maximum number of people who did not join any club -/
def max_not_joined (total : ℕ) (m s z : ℕ) : ℕ :=
  total - (m + max s z)

/-- Proof that the maximum number of people who did not join any club is 26 -/
theorem max_not_joined_company : max_not_joined 60 16 18 11 = 26 := by
  sorry

end max_not_joined_company_l365_36501


namespace diophantine_fraction_equality_l365_36557

theorem diophantine_fraction_equality : ∃ (A B : ℤ), 
  A = 500 ∧ B = -501 ∧ (A : ℚ) / 999 + (B : ℚ) / 1001 = 1 / 999999 := by
  sorry

end diophantine_fraction_equality_l365_36557


namespace point_coordinates_wrt_origin_l365_36565

/-- 
Given a point P in a Cartesian coordinate system, this theorem states that 
its coordinates with respect to the origin are the negatives of its original coordinates.
-/
theorem point_coordinates_wrt_origin (x y : ℝ) : 
  let P : ℝ × ℝ := (x, y)
  let P_wrt_origin : ℝ × ℝ := (-x, -y)
  P_wrt_origin = (-(P.1), -(P.2)) :=
by sorry

end point_coordinates_wrt_origin_l365_36565


namespace theater_ticket_cost_l365_36554

/-- The cost of theater tickets for a group -/
def theater_cost (adult_price : ℚ) : ℚ :=
  let child_price := adult_price / 2
  let total_price := 10 * adult_price + 8 * child_price
  total_price * (1 - 1/10)  -- 10% discount applied

theorem theater_ticket_cost :
  ∃ (adult_price : ℚ),
    8 * adult_price + 7 * (adult_price / 2) = 42 ∧
    theater_cost adult_price = 46 :=
by sorry

end theater_ticket_cost_l365_36554


namespace product_102_104_divisible_by_8_l365_36528

theorem product_102_104_divisible_by_8 : (102 * 104) % 8 = 0 := by
  sorry

end product_102_104_divisible_by_8_l365_36528


namespace negp_sufficient_not_necessary_for_negq_l365_36597

def p (x : ℝ) : Prop := x < -1 ∨ x > 1

def q (x : ℝ) : Prop := x < -2 ∨ x > 1

theorem negp_sufficient_not_necessary_for_negq :
  (∀ x, ¬(p x) → ¬(q x)) ∧ ¬(∀ x, ¬(q x) → ¬(p x)) := by sorry

end negp_sufficient_not_necessary_for_negq_l365_36597


namespace doubled_roots_ratio_l365_36572

theorem doubled_roots_ratio (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : ∃ x₁ x₂ : ℝ, (x₁^2 + a*x₁ + b = 0 ∧ x₂^2 + a*x₂ + b = 0) ∧ 
                    ((2*x₁)^2 + b*(2*x₁) + c = 0 ∧ (2*x₂)^2 + b*(2*x₂) + c = 0)) :
  a / c = 1 / 8 := by
sorry

end doubled_roots_ratio_l365_36572


namespace bakery_sugar_amount_l365_36563

/-- Given the ratios of ingredients in a bakery storage room, 
    prove the amount of sugar stored. -/
theorem bakery_sugar_amount 
  (sugar flour baking_soda : ℕ) 
  (h1 : sugar = flour)  -- sugar to flour ratio is 5:5, which simplifies to 1:1
  (h2 : flour = 10 * baking_soda)  -- flour to baking soda ratio is 10:1
  (h3 : flour = 8 * (baking_soda + 60))  -- if 60 more pounds of baking soda, ratio would be 8:1
  : sugar = 2400 := by
  sorry

end bakery_sugar_amount_l365_36563


namespace turtle_marathon_time_l365_36514

/-- The time taken by a turtle to complete a marathon -/
theorem turtle_marathon_time (turtle_speed : ℝ) (marathon_distance : ℝ) :
  turtle_speed = 15 →
  marathon_distance = 42195 →
  ∃ (days hours minutes : ℕ),
    (days = 1 ∧ hours = 22 ∧ minutes = 53) ∧
    (days * 24 * 60 + hours * 60 + minutes : ℝ) * turtle_speed = marathon_distance :=
by sorry

end turtle_marathon_time_l365_36514


namespace distance_in_scientific_notation_l365_36578

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem distance_in_scientific_notation :
  let distance : ℝ := 38000
  let scientific_form := toScientificNotation distance
  scientific_form.coefficient = 3.8 ∧ scientific_form.exponent = 4 :=
sorry

end distance_in_scientific_notation_l365_36578


namespace p_sufficient_not_necessary_l365_36595

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x - 3| < 1
def q (x : ℝ) : Prop := x^2 + x - 6 > 0

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary :
  (∀ x, p x → q x) ∧ (∃ x, q x ∧ ¬p x) := by sorry

end p_sufficient_not_necessary_l365_36595


namespace remainder_7645_div_9_l365_36503

theorem remainder_7645_div_9 : 7645 % 9 = 4 := by
  sorry

end remainder_7645_div_9_l365_36503


namespace compound_interest_rate_problem_l365_36518

theorem compound_interest_rate_problem (P r : ℝ) 
  (h1 : P * (1 + r)^2 = 17640) 
  (h2 : P * (1 + r)^3 = 18522) : 
  (1 + r)^3 / (1 + r)^2 = 18522 / 17640 := by sorry

end compound_interest_rate_problem_l365_36518


namespace abs_eq_sqrt_square_l365_36584

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end abs_eq_sqrt_square_l365_36584


namespace new_function_not_transformation_of_original_l365_36537

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the new quadratic function
def new_function (x : ℝ) : ℝ := x^2

-- Define a general quadratic function
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Theorem statement
theorem new_function_not_transformation_of_original :
  ¬∃ (h k : ℝ), ∀ x, new_function x = original_function (x - h) + k ∧
  ¬∃ (h k : ℝ), ∀ x, new_function x = original_function (-(x - h)) + k :=
by sorry

end new_function_not_transformation_of_original_l365_36537


namespace imaginary_unit_power_l365_36564

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2016 = 1 := by sorry

end imaginary_unit_power_l365_36564


namespace quadratic_inequality_range_l365_36532

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 + 2*a*x + a > 0) ↔ a > -1/3 :=
sorry

end quadratic_inequality_range_l365_36532


namespace solve_for_m_l365_36577

theorem solve_for_m (n : ℝ) : 
  ∃ m : ℝ, 21 * (m + n) + 21 = 21 * (-m + n) + 21 ∧ m = 1 / 2 := by
  sorry

end solve_for_m_l365_36577


namespace problem_solution_l365_36548

theorem problem_solution (a b c : ℝ) 
  (h1 : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
  (h2 : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 15) :
  a / (a + b) + b / (b + c) + c / (c + a) = -12 := by
sorry

end problem_solution_l365_36548


namespace project_cans_total_l365_36580

theorem project_cans_total (martha_cans : ℕ) (diego_extra : ℕ) (additional_cans : ℕ) : 
  martha_cans = 90 →
  diego_extra = 10 →
  additional_cans = 5 →
  martha_cans + (martha_cans / 2 + diego_extra) + additional_cans = 150 :=
by sorry

end project_cans_total_l365_36580


namespace arithmetic_mean_of_scores_l365_36585

def scores : List ℝ := [90, 93.5, 87, 96, 92, 89.5]

theorem arithmetic_mean_of_scores :
  (scores.sum / scores.length : ℝ) = 91.333333333333333333 := by
  sorry

end arithmetic_mean_of_scores_l365_36585


namespace park_tree_count_l365_36558

def park_trees (initial_maple : ℕ) (initial_poplar : ℕ) (oak : ℕ) : ℕ :=
  let planted_maple := 3 * initial_poplar
  let total_maple := initial_maple + planted_maple
  let planted_poplar := 3 * initial_poplar
  let total_poplar := initial_poplar + planted_poplar
  total_maple + total_poplar + oak

theorem park_tree_count :
  park_trees 2 5 4 = 32 := by sorry

end park_tree_count_l365_36558


namespace product_327_8_and_7_8_l365_36511

/-- Convert a number from base 8 to base 10 -/
def base8To10 (n : ℕ) : ℕ := sorry

/-- Convert a number from base 10 to base 8 -/
def base10To8 (n : ℕ) : ℕ := sorry

/-- Multiply two numbers in base 8 -/
def multiplyBase8 (a b : ℕ) : ℕ :=
  base10To8 (base8To10 a * base8To10 b)

theorem product_327_8_and_7_8 :
  multiplyBase8 327 7 = 2741 := by sorry

end product_327_8_and_7_8_l365_36511


namespace inequality_proof_l365_36598

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 1 / a + 1 / b + 1 / c = 1) :
  Real.sqrt (a * b + c) + Real.sqrt (b * c + a) + Real.sqrt (c * a + b) ≥
  Real.sqrt (a * b * c) + Real.sqrt a + Real.sqrt b + Real.sqrt c := by
  sorry

end inequality_proof_l365_36598


namespace vanishing_function_l365_36530

theorem vanishing_function (g : ℝ → ℝ) (h₁ : Continuous (deriv g)) 
  (h₂ : g 0 = 0) (h₃ : ∀ x, |deriv g x| ≤ |g x|) : 
  ∀ x, g x = 0 := by
  sorry

end vanishing_function_l365_36530


namespace complex_modulus_equation_l365_36527

theorem complex_modulus_equation : ∃ (t : ℝ), t > 0 ∧ Complex.abs (3 - 3 + t * Complex.I) = 5 ∧ t = 5 := by
  sorry

end complex_modulus_equation_l365_36527


namespace tan_alpha_plus_pi_fourth_l365_36516

theorem tan_alpha_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 3/22 := by
sorry

end tan_alpha_plus_pi_fourth_l365_36516


namespace square_perimeter_l365_36531

theorem square_perimeter (a : ℝ) : 
  a > 0 → 
  let l := a * Real.sqrt 2
  let d := l / 2 + l / 4 + l / 8 + l / 16
  d = 15 * Real.sqrt 2 → 
  4 * a = 64 := by
sorry

end square_perimeter_l365_36531


namespace equation_solution_l365_36512

theorem equation_solution : 
  ∃ x : ℚ, x ≠ -4 ∧ (7 * x / (x + 4) - 4 / (x + 4) = 2 / (x + 4)) ∧ x = 6 / 7 := by
  sorry

end equation_solution_l365_36512


namespace inequality_solution_set_l365_36520

theorem inequality_solution_set (x : ℝ) : 
  x ≠ 3 → ((x^2 - 1) / ((x - 3)^2) ≥ 0 ↔ x ∈ Set.Iic (-1) ∪ Set.Ici 1) :=
by sorry

end inequality_solution_set_l365_36520


namespace license_plate_count_l365_36581

/-- The number of letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of digits available -/
def num_digits : ℕ := 10

/-- A valid license plate configuration -/
structure LicensePlate where
  first : Fin num_letters
  second : Fin (num_letters + num_digits - 2)
  third : Fin num_letters
  fourth : Fin num_digits

/-- The total number of valid license plates -/
def total_license_plates : ℕ := num_letters * (num_letters + num_digits - 2) * num_letters * num_digits

theorem license_plate_count :
  total_license_plates = 236600 := by
  sorry

#eval total_license_plates

end license_plate_count_l365_36581


namespace prime_power_sum_existence_l365_36540

theorem prime_power_sum_existence (p : Finset Nat) (h_prime : ∀ q ∈ p, Nat.Prime q) :
  ∃ x : Nat,
    (∃ a b m n : Nat, (m ∈ p) ∧ (n ∈ p) ∧ (x = a^m + b^n)) ∧
    (∀ q : Nat, Nat.Prime q → (∃ c d : Nat, x = c^q + d^q) → q ∈ p) := by
  sorry

end prime_power_sum_existence_l365_36540


namespace complement_intersection_theorem_l365_36566

def U : Set Int := {-2, -1, 0, 1, 2, 3}
def M : Set Int := {0, 1, 2}
def N : Set Int := {0, 1, 2, 3}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {3} := by sorry

end complement_intersection_theorem_l365_36566


namespace max_dimes_count_l365_36538

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.1

/-- The value of a penny in dollars -/
def penny_value : ℚ := 0.01

/-- The total amount of money Sasha has in dollars -/
def total_money : ℚ := 3.5

/-- Theorem: Given $3.50 in coins and an equal number of dimes and pennies, 
    the maximum number of dimes possible is 31 -/
theorem max_dimes_count : 
  ∃ (d : ℕ), d ≤ 31 ∧ 
  ∀ (n : ℕ), n * (dime_value + penny_value) ≤ total_money → n ≤ d :=
sorry

end max_dimes_count_l365_36538


namespace xoxoxox_probability_l365_36547

/-- The probability of arranging 4 X tiles and 3 O tiles in the specific order XOXOXOX -/
theorem xoxoxox_probability (n : ℕ) (x o : ℕ) (h1 : n = 7) (h2 : x = 4) (h3 : o = 3) :
  (1 : ℚ) / (n.choose x) = 1 / 35 := by
  sorry

end xoxoxox_probability_l365_36547


namespace union_A_complement_B_I_l365_36545

def I : Set ℤ := {x | x^2 < 9}
def A : Set ℤ := {1, 2}
def B : Set ℤ := {-2, -1, 2}

theorem union_A_complement_B_I : A ∪ (I \ B) = {0, 1, 2} := by
  sorry

end union_A_complement_B_I_l365_36545


namespace megan_homework_problems_l365_36561

/-- The total number of homework problems Megan had -/
def total_problems (finished : ℕ) (pages_left : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished + pages_left * problems_per_page

/-- Proof that Megan had 40 homework problems in total -/
theorem megan_homework_problems :
  total_problems 26 2 7 = 40 := by
  sorry

end megan_homework_problems_l365_36561


namespace gold_silver_ratio_l365_36583

/-- Proves that the ratio of gold to silver bought is 2:1 given the specified conditions --/
theorem gold_silver_ratio :
  let silver_amount : ℝ := 1.5
  let silver_price_per_ounce : ℝ := 20
  let gold_price_multiplier : ℝ := 50
  let total_spent : ℝ := 3030
  let gold_price_per_ounce := silver_price_per_ounce * gold_price_multiplier
  let silver_cost := silver_amount * silver_price_per_ounce
  let gold_cost := total_spent - silver_cost
  let gold_amount := gold_cost / gold_price_per_ounce
  gold_amount / silver_amount = 2 := by
sorry


end gold_silver_ratio_l365_36583


namespace stick_triangle_area_l365_36509

/-- Given three sticks of length 24, one of which is broken into two parts,
    if these parts form a right triangle with the other two sticks,
    then the area of this triangle is 216 square centimeters. -/
theorem stick_triangle_area : ∀ a : ℝ,
  0 < a →
  a < 24 →
  a^2 + 24^2 = (48 - a)^2 →
  (1/2) * a * 24 = 216 := by
  sorry

end stick_triangle_area_l365_36509


namespace fraction_sum_equality_l365_36582

theorem fraction_sum_equality : (18 : ℚ) / 45 - 2 / 9 + 1 / 6 = 31 / 90 := by
  sorry

end fraction_sum_equality_l365_36582


namespace possible_values_of_a_plus_b_l365_36542

theorem possible_values_of_a_plus_b (a b : ℝ) 
  (h1 : |a| = 5) 
  (h2 : |b| = 1) 
  (h3 : a - b < 0) : 
  a + b = -6 ∨ a + b = -4 := by
sorry

end possible_values_of_a_plus_b_l365_36542


namespace inequality_proof_l365_36551

theorem inequality_proof (a b c : ℝ) (h : a^2*b*c + a*b^2*c + a*b*c^2 = 1) :
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 := by
  sorry

end inequality_proof_l365_36551


namespace cube_volume_to_surface_area_l365_36591

theorem cube_volume_to_surface_area :
  ∀ (s : ℝ), s > 0 → s^3 = 729 → 6 * s^2 = 486 :=
by
  sorry

end cube_volume_to_surface_area_l365_36591


namespace league_games_count_l365_36522

/-- The number of games played in a season for a league with a given number of teams and games per pair of teams. -/
def games_in_season (num_teams : ℕ) (games_per_pair : ℕ) : ℕ :=
  (num_teams * (num_teams - 1) / 2) * games_per_pair

/-- Theorem: In a league with 20 teams where each pair of teams plays 10 games, 
    the total number of games in the season is 1900. -/
theorem league_games_count : games_in_season 20 10 = 1900 := by
  sorry

#eval games_in_season 20 10

end league_games_count_l365_36522


namespace two_digit_primes_with_digit_sum_10_l365_36593

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_two_digit n ∧ Nat.Prime n ∧ digit_sum n = 10

theorem two_digit_primes_with_digit_sum_10 :
  ∃ (S : Finset ℕ), (∀ n ∈ S, satisfies_conditions n) ∧ S.card = 4 :=
sorry

end two_digit_primes_with_digit_sum_10_l365_36593


namespace unique_solution_iff_c_equals_three_l365_36560

theorem unique_solution_iff_c_equals_three :
  ∀ c : ℝ, (∃! (x y : ℝ), (2 * |x + 7| + |y - 4| = c) ∧ (|x + 4| + 2 * |y - 7| = c)) ↔ c = 3 := by
  sorry

end unique_solution_iff_c_equals_three_l365_36560


namespace opposite_sides_of_line_l365_36535

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to evaluate the line equation at a point
def evaluateLine (l : Line2D) (p : Point2D) : ℝ :=
  l.a * p.x + l.b * p.y + l.c

-- Define the specific line and points
def line : Line2D := { a := -3, b := 1, c := 2 }
def origin : Point2D := { x := 0, y := 0 }
def point : Point2D := { x := 2, y := 1 }

-- Theorem statement
theorem opposite_sides_of_line :
  evaluateLine line origin * evaluateLine line point < 0 := by
  sorry

end opposite_sides_of_line_l365_36535


namespace polynomial_evaluation_l365_36536

theorem polynomial_evaluation : 3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 5999 := by
  sorry

end polynomial_evaluation_l365_36536


namespace bill_and_harry_combined_nuts_l365_36553

def sue_nuts : ℕ := 48

theorem bill_and_harry_combined_nuts :
  let harry_nuts := 2 * sue_nuts
  let bill_nuts := 6 * harry_nuts
  bill_nuts + harry_nuts = 672 := by
  sorry

end bill_and_harry_combined_nuts_l365_36553


namespace cubic_polynomial_property_l365_36510

theorem cubic_polynomial_property (p q r : ℝ) : 
  let Q : ℝ → ℝ := λ x => x^3 + p*x^2 + q*x + r
  let mean_zeros := -p / 3
  let product_zeros := -r
  let sum_coefficients := 1 + p + q + r
  (mean_zeros = product_zeros ∧ product_zeros = sum_coefficients ∧ r = 3) →
  q = -16 := by
sorry

end cubic_polynomial_property_l365_36510


namespace count_seven_digit_phone_numbers_l365_36524

/-- The number of different seven-digit phone numbers where the first digit cannot be zero -/
def seven_digit_phone_numbers : ℕ :=
  9 * 10^6

/-- Theorem stating that the number of different seven-digit phone numbers
    where the first digit cannot be zero is equal to 9 × 10^6 -/
theorem count_seven_digit_phone_numbers :
  seven_digit_phone_numbers = 9 * 10^6 := by
  sorry

end count_seven_digit_phone_numbers_l365_36524


namespace meal_cost_l365_36588

theorem meal_cost (adults children : ℕ) (total_bill : ℚ) :
  adults = 2 →
  children = 5 →
  total_bill = 21 →
  (total_bill / (adults + children : ℚ)) = 3 := by
sorry

end meal_cost_l365_36588


namespace union_S_T_l365_36567

-- Define the sets S and T
def S : Set ℝ := {x | x > -2}
def T : Set ℝ := {x | x^2 + 3*x - 4 ≤ 0}

-- State the theorem
theorem union_S_T : S ∪ T = {x : ℝ | x ≥ -4} := by sorry

end union_S_T_l365_36567


namespace fraction_evaluation_l365_36521

theorem fraction_evaluation : 
  (10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) / 
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9) = 1 := by
  sorry

end fraction_evaluation_l365_36521


namespace pencil_profit_problem_l365_36587

theorem pencil_profit_problem (total_pencils : ℕ) (buy_price sell_price : ℚ) (profit : ℚ) :
  total_pencils = 2000 →
  buy_price = 8/100 →
  sell_price = 20/100 →
  profit = 160 →
  ∃ (sold_pencils : ℕ), sold_pencils = 1600 ∧ 
    sell_price * sold_pencils = buy_price * total_pencils + profit :=
by sorry

end pencil_profit_problem_l365_36587


namespace helicopter_rental_hours_per_day_l365_36592

/-- Given the total cost, hourly rate, and number of days for renting a helicopter,
    calculate the number of hours rented per day. -/
theorem helicopter_rental_hours_per_day 
  (total_cost : ℝ) 
  (hourly_rate : ℝ) 
  (num_days : ℝ) 
  (h1 : total_cost = 450)
  (h2 : hourly_rate = 75)
  (h3 : num_days = 3)
  (h4 : hourly_rate > 0)
  (h5 : num_days > 0) :
  total_cost / (hourly_rate * num_days) = 2 := by
  sorry

#check helicopter_rental_hours_per_day

end helicopter_rental_hours_per_day_l365_36592


namespace ellipse_intersection_k_values_l365_36513

noncomputable section

def Ellipse (a b : ℝ) := {p : ℝ × ℝ | (p.1^2 / a^2) + (p.2^2 / b^2) = 1}

def Line (k m : ℝ) := {p : ℝ × ℝ | p.2 = k * p.1 + m}

def Eccentricity (a b : ℝ) := Real.sqrt (1 - (b^2 / a^2))

def Parallelogram (A B C D : ℝ × ℝ) :=
  (B.1 - A.1 = D.1 - C.1 ∧ B.2 - A.2 = D.2 - C.2) ∧
  (C.1 - A.1 = D.1 - B.1 ∧ C.2 - A.2 = D.2 - B.2)

theorem ellipse_intersection_k_values
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (h_A : (2, 0) ∈ Ellipse a b)
  (h_e : Eccentricity a b = Real.sqrt 3 / 2)
  (k : ℝ)
  (M N : ℝ × ℝ)
  (h_MN : M ∈ Ellipse a b ∧ N ∈ Ellipse a b)
  (h_MN_line : M ∈ Line k (Real.sqrt 3) ∧ N ∈ Line k (Real.sqrt 3))
  (P : ℝ × ℝ)
  (h_P : P.1 = 3)
  (h_parallelogram : Parallelogram (2, 0) P M N) :
  k = Real.sqrt 3 / 2 ∨ k = Real.sqrt 11 / 2 ∨ k = -Real.sqrt 11 / 2 :=
sorry

end ellipse_intersection_k_values_l365_36513


namespace exists_city_that_reaches_all_l365_36556

-- Define the type for cities
variable {City : Type}

-- Define the "can reach" relation
variable (canReach : City → City → Prop)

-- Define the properties of the "can reach" relation
variable (h_reflexive : ∀ x : City, canReach x x)
variable (h_transitive : ∀ x y z : City, canReach x y → canReach y z → canReach x z)

-- Define the condition that for any two cities, there's a city that can reach both
variable (h_common_reachable : ∀ x y : City, ∃ z : City, canReach z x ∧ canReach z y)

-- State the theorem
theorem exists_city_that_reaches_all [Finite City] :
  ∃ c : City, ∀ x : City, canReach c x :=
sorry

end exists_city_that_reaches_all_l365_36556


namespace one_third_of_seven_times_nine_minus_three_l365_36559

theorem one_third_of_seven_times_nine_minus_three (x : ℚ) : 
  x = (1 / 3 : ℚ) * (7 * 9) - 3 → x = 18 := by
  sorry

end one_third_of_seven_times_nine_minus_three_l365_36559


namespace adding_five_increases_value_l365_36570

theorem adding_five_increases_value (x : ℝ) : x + 5 > x := by
  sorry

end adding_five_increases_value_l365_36570


namespace assembly_problem_solution_l365_36550

/-- Represents a worker in the assembly line -/
structure Worker where
  assemblyRate : ℝ  -- switches per hour
  payment : ℝ       -- in Ft

/-- The problem setup -/
def assemblyProblem (totalPayment : ℝ) (overtimeHours : ℝ) (worker1Payment : ℝ) 
    (worker2Rate : ℝ) (worker3PaymentDiff : ℝ) : Prop :=
  ∃ (w1 w2 w3 : Worker),
    -- Total payment condition
    w1.payment + w2.payment + w3.payment = totalPayment
    -- First worker's payment
    ∧ w1.payment = worker1Payment
    -- Second worker's assembly rate
    ∧ w2.assemblyRate = 60 / worker2Rate
    -- Third worker's payment difference
    ∧ w3.payment = w2.payment - worker3PaymentDiff
    -- Total switches assembled
    ∧ (w1.assemblyRate + w2.assemblyRate + w3.assemblyRate) * overtimeHours = 235

/-- The theorem to be proved -/
theorem assembly_problem_solution :
  assemblyProblem 4700 5 2000 4 300 :=
by sorry

end assembly_problem_solution_l365_36550


namespace power_sum_and_division_l365_36549

theorem power_sum_and_division (x y z : ℕ) : 3^128 + 8^5 / 8^3 = 65 := by
  sorry

end power_sum_and_division_l365_36549


namespace douglas_weight_is_52_l365_36515

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- The difference in weight between Anne and Douglas in pounds -/
def weight_difference : ℕ := 15

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := anne_weight - weight_difference

/-- Theorem stating Douglas's weight -/
theorem douglas_weight_is_52 : douglas_weight = 52 := by
  sorry

end douglas_weight_is_52_l365_36515


namespace competition_results_l365_36562

def team_a_scores : List ℝ := [7, 8, 9, 7, 10, 10, 9, 10, 10, 10]
def team_b_scores : List ℝ := [10, 8, 7, 9, 8, 10, 10, 9, 10, 9]
def team_a_variance : ℝ := 1.4

def median (l : List ℝ) : ℝ := sorry
def mode (l : List ℝ) : ℝ := sorry
def average (l : List ℝ) : ℝ := sorry
def variance (l : List ℝ) : ℝ := sorry

theorem competition_results :
  (median team_a_scores = 9.5) ∧
  (mode team_b_scores = 10) ∧
  (average team_b_scores = 9) ∧
  (variance team_b_scores = 1) ∧
  (variance team_b_scores < team_a_variance) :=
by sorry

end competition_results_l365_36562


namespace infinitely_many_2024_endings_l365_36586

/-- The sequence (x_n) defined by the given recurrence relation -/
def x : ℕ → ℕ
  | 0 => 0
  | 1 => 2024
  | (n + 2) => x (n + 1) + x n

/-- The set of natural numbers n where x_n ends with 2024 -/
def ends_with_2024 : Set ℕ := {n | x n % 10000 = 2024}

/-- The main theorem stating that there are infinitely many terms in the sequence ending with 2024 -/
theorem infinitely_many_2024_endings : Set.Infinite ends_with_2024 := by
  sorry


end infinitely_many_2024_endings_l365_36586


namespace range_of_a_full_range_of_a_l365_36526

/-- Given sets A and B, prove the range of a when A ∩ B = A -/
theorem range_of_a (a : ℝ) : 
  let A := {x : ℝ | x^2 - a < 0}
  let B := {x : ℝ | x < 2}
  A ∩ B = A → a ≤ 4 :=
by
  sorry

/-- The full range of a includes all numbers less than or equal to 4 -/
theorem full_range_of_a : 
  ∃ a : ℝ, 
    let A := {x : ℝ | x^2 - a < 0}
    let B := {x : ℝ | x < 2}
    (A ∩ B = A) ∧ (a ≤ 4) :=
by
  sorry

end range_of_a_full_range_of_a_l365_36526


namespace problem_solution_l365_36519

-- Define the set M
def M : Set ℝ := {m | ∃ x ∈ Set.Icc (-1 : ℝ) 1, m = x^2 - x}

-- Define the set N
def N (a : ℝ) : Set ℝ := {x | (x - a) * (x - (2 - a)) < 0}

-- Theorem statement
theorem problem_solution :
  (M = Set.Icc (-1/4 : ℝ) 2) ∧
  (∀ a : ℝ, N a ⊆ M ↔ 0 ≤ a ∧ a ≤ 2) :=
by sorry

end problem_solution_l365_36519


namespace quadratic_sum_reciprocal_l365_36543

theorem quadratic_sum_reciprocal (x : ℝ) (h : x^2 - 4*x + 2 = 0) : x + 2/x = 4 := by
  sorry

end quadratic_sum_reciprocal_l365_36543


namespace triangle_properties_l365_36507

/-- Given a triangle ABC with angle B = 150°, side a = √3c, and side b = 2√7,
    prove that its area is √3 and if sin A + √3 sin C = √2/2, then C = 15° --/
theorem triangle_properties (A B C : Real) (a b c : Real) :
  B = 150 * π / 180 →
  a = Real.sqrt 3 * c →
  b = 2 * Real.sqrt 7 →
  (1/2) * a * c * Real.sin B = Real.sqrt 3 ∧
  (Real.sin A + Real.sqrt 3 * Real.sin C = Real.sqrt 2 / 2 → C = 15 * π / 180) :=
by sorry

end triangle_properties_l365_36507


namespace initial_ratio_new_ratio_partners_count_l365_36596

/-- Represents the number of partners in a firm -/
def partners : ℕ := 18

/-- Represents the number of associates in a firm -/
def associates : ℕ := (63 * partners) / 2

/-- The ratio of partners to associates is 2:63 -/
theorem initial_ratio : partners * 63 = associates * 2 := by sorry

/-- Adding 45 associates changes the ratio to 1:34 -/
theorem new_ratio : partners * 34 = (associates + 45) * 1 := by sorry

/-- The number of partners in the firm is 18 -/
theorem partners_count : partners = 18 := by sorry

end initial_ratio_new_ratio_partners_count_l365_36596


namespace triangle_area_from_intersecting_lines_triangle_area_from_intersecting_lines_proof_l365_36504

/-- Given two lines intersecting at P(1,6), one with slope 1 and the other with slope 2,
    the area of the triangle formed by P and the x-intercepts of these lines is 9 square units. -/
theorem triangle_area_from_intersecting_lines : ℝ → Prop :=
  fun area =>
    let P : ℝ × ℝ := (1, 6)
    let slope1 : ℝ := 1
    let slope2 : ℝ := 2
    let Q : ℝ × ℝ := (P.1 - P.2 / slope1, 0)  -- x-intercept of line with slope 1
    let R : ℝ × ℝ := (P.1 - P.2 / slope2, 0)  -- x-intercept of line with slope 2
    let base : ℝ := R.1 - Q.1
    let height : ℝ := P.2
    area = (1/2) * base * height ∧ area = 9

/-- Proof of the theorem -/
theorem triangle_area_from_intersecting_lines_proof : 
  ∃ (area : ℝ), triangle_area_from_intersecting_lines area :=
by
  sorry

#check triangle_area_from_intersecting_lines
#check triangle_area_from_intersecting_lines_proof

end triangle_area_from_intersecting_lines_triangle_area_from_intersecting_lines_proof_l365_36504


namespace gcf_of_21_and_12_l365_36506

theorem gcf_of_21_and_12 (h : Nat.lcm 21 12 = 42) : Nat.gcd 21 12 = 6 := by
  sorry

end gcf_of_21_and_12_l365_36506


namespace lance_penny_savings_l365_36594

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Lance's penny savings problem -/
theorem lance_penny_savings :
  arithmetic_sum 5 2 20 = 480 := by
  sorry

end lance_penny_savings_l365_36594


namespace davids_math_marks_l365_36534

/-- Calculates David's marks in Mathematics given his marks in other subjects and the average --/
theorem davids_math_marks (english physics chemistry biology : ℕ) (average : ℕ) (h1 : english = 86) (h2 : physics = 82) (h3 : chemistry = 87) (h4 : biology = 85) (h5 : average = 85) :
  (english + physics + chemistry + biology + (5 * average - (english + physics + chemistry + biology))) / 5 = average :=
sorry

end davids_math_marks_l365_36534


namespace square_area_error_l365_36579

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * (1 + 0.01)
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.0201 := by
  sorry

end square_area_error_l365_36579


namespace product_sum_relation_l365_36574

/-- Given single-digit integers P and Q where 39P × Q3 = 32951, prove that P + Q = 15 -/
theorem product_sum_relation (P Q : ℕ) : 
  P < 10 → Q < 10 → 39 * P * 10 + P * 3 + Q * 300 + Q * 30 + Q * 3 = 32951 → P + Q = 15 := by
  sorry

end product_sum_relation_l365_36574


namespace number_equation_l365_36505

theorem number_equation (y : ℝ) : y = (1 / y) * (-y) - 5 → y = -6 := by
  sorry

end number_equation_l365_36505


namespace linear_function_m_value_l365_36552

/-- A linear function y = mx + m^2 passing through (0, 4) with positive slope -/
def linear_function (m : ℝ) (x : ℝ) : ℝ := m * x + m^2

theorem linear_function_m_value :
  ∀ m : ℝ,
  m ≠ 0 →
  linear_function m 0 = 4 →
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → linear_function m x₁ < linear_function m x₂) →
  m = 2 :=
by sorry

end linear_function_m_value_l365_36552


namespace function_inequality_l365_36571

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h1 : ∀ x, deriv f x > f x) (a : ℝ) (ha : a > 0) : 
  Real.exp a * f 0 < f a := by
  sorry

end function_inequality_l365_36571


namespace condition_relationships_l365_36500

theorem condition_relationships (α β γ : Prop) 
  (h1 : β → α)  -- α is necessary for β
  (h2 : ¬(α → β))  -- α is not sufficient for β
  (h3 : γ ↔ β)  -- γ is necessary and sufficient for β
  : (γ → α) ∧ ¬(α → γ) := by sorry

end condition_relationships_l365_36500


namespace union_of_sets_l365_36555

theorem union_of_sets : 
  let A : Set ℕ := {1, 2, 3}
  let B : Set ℕ := {2, 3, 4}
  A ∪ B = {1, 2, 3, 4} := by
sorry

end union_of_sets_l365_36555


namespace petya_bonus_points_l365_36539

def calculate_bonus (score : ℕ) : ℕ :=
  if score < 1000 then
    (score * 20) / 100
  else if score < 2000 then
    200 + ((score - 1000) * 30) / 100
  else
    200 + 300 + ((score - 2000) * 50) / 100

theorem petya_bonus_points : calculate_bonus 2370 = 685 := by
  sorry

end petya_bonus_points_l365_36539


namespace difference_3003_l365_36569

/-- The number of terms in each sequence -/
def n : ℕ := 3003

/-- The sum of the first n odd numbers -/
def sum_odd (n : ℕ) : ℕ := n * n

/-- The sum of the first n even numbers starting from 2 -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- The difference between the sum of the first n even numbers (starting from 2) 
    and the sum of the first n odd numbers -/
def difference (n : ℕ) : ℤ := sum_even n - sum_odd n

theorem difference_3003 : difference n = 7999 := by sorry

end difference_3003_l365_36569


namespace exam_score_problem_l365_36525

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 60)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 110) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 34 := by
  sorry

end exam_score_problem_l365_36525


namespace red_in_B_equals_black_in_C_l365_36573

/-- Represents a playing card color -/
inductive CardColor
| Red
| Black

/-- Represents a box where cards are placed -/
inductive Box
| A
| B
| C

/-- Represents the state of the card distribution -/
structure CardDistribution where
  cardsInA : ℕ
  redInB : ℕ
  blackInB : ℕ
  redInC : ℕ
  blackInC : ℕ

/-- The card distribution process -/
def distributeCards : CardDistribution → CardColor → CardDistribution
| d, CardColor.Red => { d with
    cardsInA := d.cardsInA + 1,
    redInB := d.redInB + 1 }
| d, CardColor.Black => { d with
    cardsInA := d.cardsInA + 1,
    blackInC := d.blackInC + 1 }

/-- The theorem stating that the number of red cards in B equals the number of black cards in C -/
theorem red_in_B_equals_black_in_C (finalDist : CardDistribution)
  (h : finalDist.cardsInA = 52) :
  finalDist.redInB = finalDist.blackInC := by
  sorry

end red_in_B_equals_black_in_C_l365_36573


namespace points_in_small_circle_l365_36544

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A definition of a unit square -/
def UnitSquare : Set Point :=
  {p : Point | 0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1}

/-- A definition of a circle with center c and radius r -/
def Circle (c : Point) (r : ℝ) : Set Point :=
  {p : Point | (p.x - c.x)^2 + (p.y - c.y)^2 ≤ r^2}

theorem points_in_small_circle (points : Finset Point) 
  (h1 : points.card = 110) 
  (h2 : ∀ p ∈ points, p ∈ UnitSquare) :
  ∃ (c : Point) (S : Finset Point), 
    S ⊆ points ∧ 
    S.card = 4 ∧ 
    ∀ p ∈ S, p ∈ Circle c (1/8) := by
  sorry


end points_in_small_circle_l365_36544


namespace dinner_time_calculation_l365_36523

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : ℕ) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  ⟨totalMinutes / 60, totalMinutes % 60, sorry⟩

theorem dinner_time_calculation (start : Time) (commute grocery drycleaning groomer cooking : ℕ) :
  commute = 30 →
  grocery = 30 →
  drycleaning = 10 →
  groomer = 20 →
  cooking = 90 →
  start = ⟨16, 0, sorry⟩ →
  addMinutes start (commute + grocery + drycleaning + groomer + cooking) = ⟨19, 0, sorry⟩ := by
  sorry


end dinner_time_calculation_l365_36523
