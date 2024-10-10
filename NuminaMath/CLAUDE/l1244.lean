import Mathlib

namespace collision_count_is_25_l1244_124464

/-- Represents a set of identical balls moving in one direction -/
structure BallSet :=
  (count : Nat)
  (direction : Bool)  -- True for left to right, False for right to left

/-- Calculates the total number of collisions between two sets of balls -/
def totalCollisions (set1 set2 : BallSet) : Nat :=
  set1.count * set2.count

/-- Theorem stating that the total number of collisions is 25 -/
theorem collision_count_is_25 :
  ∀ (left right : BallSet),
    left.count = 5 ∧ 
    right.count = 5 ∧ 
    left.direction ≠ right.direction →
    totalCollisions left right = 25 := by
  sorry

#eval totalCollisions ⟨5, true⟩ ⟨5, false⟩

end collision_count_is_25_l1244_124464


namespace smallest_satisfying_both_properties_l1244_124425

def is_sum_of_five_fourth_powers (n : ℕ) : Prop :=
  ∃ (a b c d e : ℕ), a < b ∧ b < c ∧ c < d ∧ d < e ∧ 
    n = a^4 + b^4 + c^4 + d^4 + e^4

def is_sum_of_six_consecutive_integers (n : ℕ) : Prop :=
  ∃ (m : ℕ), n = m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5)

theorem smallest_satisfying_both_properties : 
  ∀ n : ℕ, n < 2019 → ¬(is_sum_of_five_fourth_powers n ∧ is_sum_of_six_consecutive_integers n) :=
by sorry

end smallest_satisfying_both_properties_l1244_124425


namespace tan_690_degrees_l1244_124440

theorem tan_690_degrees : Real.tan (690 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_690_degrees_l1244_124440


namespace bounds_of_W_l1244_124405

/-- Given conditions on x, y, and z, prove the bounds of W = 2x + 6y + 4z -/
theorem bounds_of_W (x y z : ℝ) 
  (sum_eq_one : x + y + z = 1)
  (ineq_one : 3 * y + z ≥ 2)
  (x_bounds : 0 ≤ x ∧ x ≤ 1)
  (y_bounds : 0 ≤ y ∧ y ≤ 2) :
  let W := 2 * x + 6 * y + 4 * z
  ∃ (W_min W_max : ℝ), W_min = 4 ∧ W_max = 6 ∧ W_min ≤ W ∧ W ≤ W_max :=
by sorry

end bounds_of_W_l1244_124405


namespace inequality_proof_l1244_124494

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 / (a^5 + b^5 + c^2)) + (1 / (b^5 + c^5 + a^2)) + (1 / (c^5 + a^5 + b^2)) ≤ 1 := by
  sorry

end inequality_proof_l1244_124494


namespace tan_equality_implies_negative_thirty_l1244_124468

theorem tan_equality_implies_negative_thirty
  (n : ℤ)
  (h1 : -90 < n ∧ n < 90)
  (h2 : Real.tan (n * π / 180) = Real.tan (1230 * π / 180)) :
  n = -30 :=
by sorry

end tan_equality_implies_negative_thirty_l1244_124468


namespace victors_initial_money_l1244_124498

/-- Victor's money problem -/
theorem victors_initial_money (initial_amount allowance total : ℕ) : 
  allowance = 8 → total = 18 → initial_amount + allowance = total → initial_amount = 10 := by
  sorry

end victors_initial_money_l1244_124498


namespace product_decreasing_implies_inequality_l1244_124420

variables {f g : ℝ → ℝ} {a b x : ℝ}

theorem product_decreasing_implies_inequality
  (h_diff_f : Differentiable ℝ f)
  (h_diff_g : Differentiable ℝ g)
  (h_deriv : ∀ x, (deriv f x) * g x + f x * (deriv g x) < 0)
  (h_x : a < x ∧ x < b) :
  f x * g x > f b * g b :=
sorry

end product_decreasing_implies_inequality_l1244_124420


namespace kevin_kangaroo_hops_l1244_124455

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem kevin_kangaroo_hops :
  let a : ℚ := 1/4
  let r : ℚ := 7/16
  let n : ℕ := 5
  geometric_sum a r n = 1031769/2359296 := by sorry

end kevin_kangaroo_hops_l1244_124455


namespace bowling_ball_weight_is_14_l1244_124451

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 14

/-- The weight of one kayak in pounds -/
def kayak_weight : ℝ := 28

/-- Theorem stating that one bowling ball weighs 14 pounds given the conditions -/
theorem bowling_ball_weight_is_14 :
  (8 * bowling_ball_weight = 4 * kayak_weight) ∧
  (3 * kayak_weight = 84) →
  bowling_ball_weight = 14 := by
  sorry

end bowling_ball_weight_is_14_l1244_124451


namespace sin_45_cos_15_plus_cos_45_sin_15_l1244_124485

theorem sin_45_cos_15_plus_cos_45_sin_15 :
  Real.sin (45 * π / 180) * Real.cos (15 * π / 180) + 
  Real.cos (45 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sin_45_cos_15_plus_cos_45_sin_15_l1244_124485


namespace square_of_digit_sum_sum_of_cube_digits_l1244_124433

-- Define the sum of digits function
def sumOfDigits (n : ℕ) : ℕ := sorry

-- Part a
theorem square_of_digit_sum (N : ℕ) : N = (sumOfDigits N)^2 ↔ N = 1 ∨ N = 81 := by sorry

-- Part b
theorem sum_of_cube_digits (N : ℕ) : N = sumOfDigits (N^3) ↔ N = 1 ∨ N = 8 ∨ N = 17 ∨ N = 18 ∨ N = 26 ∨ N = 27 := by sorry

end square_of_digit_sum_sum_of_cube_digits_l1244_124433


namespace pond_water_after_50_days_l1244_124417

/-- Calculates the remaining water in a pond after a given number of days, 
    given an initial amount and daily evaporation rate. -/
def remaining_water (initial_amount : ℝ) (evaporation_rate : ℝ) (days : ℕ) : ℝ :=
  initial_amount - evaporation_rate * days

/-- Theorem stating that given specific initial conditions, 
    the amount of water remaining after 50 days is 200 gallons. -/
theorem pond_water_after_50_days 
  (initial_amount : ℝ) 
  (evaporation_rate : ℝ) 
  (h1 : initial_amount = 250)
  (h2 : evaporation_rate = 1) :
  remaining_water initial_amount evaporation_rate 50 = 200 :=
by
  sorry

#eval remaining_water 250 1 50

end pond_water_after_50_days_l1244_124417


namespace part_one_part_two_l1244_124483

-- Define the conditions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

-- Part 1
theorem part_one : ∀ x : ℝ, (p 1 x ∨ q x) → (1 < x ∧ x < 3) := by sorry

-- Part 2
theorem part_two : 
  (∀ x : ℝ, q x → p a x) ∧ 
  (∃ x : ℝ, p a x ∧ ¬q x) ∧ 
  (a > 0) → 
  (1 ≤ a ∧ a ≤ 2) := by sorry

end part_one_part_two_l1244_124483


namespace simplify_and_evaluate_l1244_124402

theorem simplify_and_evaluate (a b : ℝ) (h : |2 - a + b| + (a * b + 1)^2 = 0) :
  (4 * a - 5 * b - a * b) - (2 * a - 3 * b + 5 * a * b) = 10 := by
  sorry

end simplify_and_evaluate_l1244_124402


namespace expression_simplification_l1244_124408

theorem expression_simplification (a b c d : ℝ) :
  (2*a + b + 2*c - d)^2 - (3*a + 2*b + 3*c - 2*d)^2 - 
  (4*a + 3*b + 4*c - 3*d)^2 + (5*a + 4*b + 5*c - 4*d)^2 = 
  (2*(a + b + c - d))^2 := by
  sorry

end expression_simplification_l1244_124408


namespace conference_handshakes_l1244_124480

/-- Represents a conference with two groups of people -/
structure Conference where
  total_people : ℕ
  group_x : ℕ
  group_y : ℕ
  known_people : ℕ
  h_total : total_people = group_x + group_y
  h_group_x : group_x = 25
  h_group_y : group_y = 15
  h_known : known_people = 5

/-- Calculates the number of handshakes in the conference -/
def handshakes (c : Conference) : ℕ :=
  let between_groups := c.group_x * c.group_y
  let within_x := (c.group_x * (c.group_x - 1 - c.known_people)) / 2
  let within_y := (c.group_y * (c.group_y - 1)) / 2
  between_groups + within_x + within_y

/-- Theorem stating that the number of handshakes in the given conference is 717 -/
theorem conference_handshakes :
    ∃ (c : Conference), handshakes c = 717 :=
  sorry

end conference_handshakes_l1244_124480


namespace equation_solutions_l1244_124487

theorem equation_solutions :
  (∃ y : ℝ, 6 - 3*y = 15 + 6*y ∧ y = -1) ∧
  (∃ x : ℝ, (1 - 2*x) / 3 = (3*x + 1) / 7 - 2 ∧ x = 2) :=
by sorry

end equation_solutions_l1244_124487


namespace intersection_A_B_range_of_a_l1244_124477

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | 2*x - 4 ≥ x - 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem 1: Intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | 2 ≤ x ∧ x < 3} := by sorry

-- Theorem 2: Range of a given B ∪ C = C
theorem range_of_a (a : ℝ) (h : B ∪ C a = C a) : a ≤ 3 := by sorry

end intersection_A_B_range_of_a_l1244_124477


namespace x_cubed_y_squared_minus_y_squared_x_cubed_eq_zero_l1244_124463

theorem x_cubed_y_squared_minus_y_squared_x_cubed_eq_zero 
  (x y : ℝ) : x^3 * y^2 - y^2 * x^3 = 0 := by
  sorry

end x_cubed_y_squared_minus_y_squared_x_cubed_eq_zero_l1244_124463


namespace investment_growth_l1244_124467

-- Define the compound interest function
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

-- Define the problem parameters
def initial_investment : ℝ := 15000
def interest_rate : ℝ := 0.04
def investment_time : ℕ := 10

-- State the theorem
theorem investment_growth :
  round (compound_interest initial_investment interest_rate investment_time) = 22204 := by
  sorry


end investment_growth_l1244_124467


namespace twelve_integers_with_properties_l1244_124473

theorem twelve_integers_with_properties : ∃ (S : Finset ℤ),
  (Finset.card S = 12) ∧
  (∃ (P : Finset ℤ), P ⊆ S ∧ Finset.card P = 6 ∧ ∀ p ∈ P, Nat.Prime p.natAbs) ∧
  (∃ (O : Finset ℤ), O ⊆ S ∧ Finset.card O = 9 ∧ ∀ n ∈ O, n % 2 ≠ 0) ∧
  (∃ (NN : Finset ℤ), NN ⊆ S ∧ Finset.card NN = 10 ∧ ∀ n ∈ NN, n ≥ 0) ∧
  (∃ (GT : Finset ℤ), GT ⊆ S ∧ Finset.card GT = 7 ∧ ∀ n ∈ GT, n > 10) :=
by
  sorry


end twelve_integers_with_properties_l1244_124473


namespace magazine_budget_cut_percentage_l1244_124441

def original_budget : ℝ := 940
def desired_reduction : ℝ := 658

theorem magazine_budget_cut_percentage :
  (original_budget - desired_reduction) / original_budget * 100 = 30 := by
  sorry

end magazine_budget_cut_percentage_l1244_124441


namespace vector_sum_magnitude_l1244_124434

/-- Given plane vectors a and b, where the angle between them is 60°,
    a = (2,0), and |b| = 1, then |a+b| = √7 -/
theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  (a = (2, 0)) →
  (Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 1) →
  (Real.cos (60 * Real.pi / 180) = a.1 * b.1 + a.2 * b.2) →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 7 := by
  sorry

end vector_sum_magnitude_l1244_124434


namespace ralphs_peanuts_l1244_124406

/-- Given Ralph's initial peanuts, lost peanuts, number of bags bought, and peanuts per bag,
    prove that Ralph ends up with the correct number of peanuts. -/
theorem ralphs_peanuts (initial : ℕ) (lost : ℕ) (bags : ℕ) (per_bag : ℕ)
    (h1 : initial = 2650)
    (h2 : lost = 1379)
    (h3 : bags = 4)
    (h4 : per_bag = 450) :
    initial - lost + bags * per_bag = 3071 := by
  sorry

end ralphs_peanuts_l1244_124406


namespace quotient_60_55_is_recurring_l1244_124478

/-- Represents a recurring decimal with an integer part and a repeating fractional part -/
structure RecurringDecimal where
  integerPart : ℤ
  repeatingPart : ℕ

/-- The quotient of 60 divided by 55 as a recurring decimal -/
def quotient_60_55 : RecurringDecimal :=
  { integerPart := 1,
    repeatingPart := 9 }

/-- Theorem stating that 60 divided by 55 is equal to the recurring decimal 1.090909... -/
theorem quotient_60_55_is_recurring : (60 : ℚ) / 55 = 1 + (9 : ℚ) / 99 := by sorry

end quotient_60_55_is_recurring_l1244_124478


namespace solve_equation_l1244_124496

theorem solve_equation (x : ℝ) (h : x + 1 = 5) : x = 4 := by
  sorry

end solve_equation_l1244_124496


namespace library_experience_problem_l1244_124488

/-- Represents the years of experience of a library employee -/
structure LibraryExperience where
  current : ℕ
  fiveYearsAgo : ℕ

/-- Represents the age and experience of a library employee -/
structure Employee where
  name : String
  age : ℕ
  experience : LibraryExperience

/-- The problem statement -/
theorem library_experience_problem 
  (bill : Employee)
  (joan : Employee)
  (h1 : bill.age = 40)
  (h2 : joan.age = 50)
  (h3 : joan.experience.fiveYearsAgo = 3 * bill.experience.fiveYearsAgo)
  (h4 : joan.experience.current = 2 * bill.experience.current)
  (h5 : bill.experience.current = bill.experience.fiveYearsAgo + 5)
  (h6 : ∃ (total_experience : ℕ), total_experience = bill.experience.current + 5) :
  bill.experience.current = 10 := by
  sorry

end library_experience_problem_l1244_124488


namespace seashells_solution_l1244_124412

/-- The number of seashells found by Sam, Mary, John, and Emily -/
def seashells_problem (sam mary john emily : ℕ) : Prop :=
  sam = 18 ∧ mary = 47 ∧ john = 32 ∧ emily = 26 →
  sam + mary + john + emily = 123

/-- Theorem stating the solution to the seashells problem -/
theorem seashells_solution : seashells_problem 18 47 32 26 := by
  sorry

end seashells_solution_l1244_124412


namespace min_value_sum_product_l1244_124475

theorem min_value_sum_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 20) :
  a + 2 * b ≥ 4 * Real.sqrt 10 := by
  sorry

end min_value_sum_product_l1244_124475


namespace spending_percentage_l1244_124474

/-- Represents Roger's entertainment budget and spending --/
structure Entertainment where
  budget : ℝ
  movie_cost : ℝ
  soda_cost : ℝ
  popcorn_cost : ℝ
  tax_rate : ℝ

/-- Calculates the total spending including tax --/
def total_spending (e : Entertainment) : ℝ :=
  (e.movie_cost + e.soda_cost + e.popcorn_cost) * (1 + e.tax_rate)

/-- Theorem stating that the total spending is approximately 28% of the budget --/
theorem spending_percentage (e : Entertainment) 
  (h1 : e.movie_cost = 0.25 * (e.budget - e.soda_cost))
  (h2 : e.soda_cost = 0.10 * (e.budget - e.movie_cost))
  (h3 : e.popcorn_cost = 5)
  (h4 : e.tax_rate = 0.10)
  (h5 : e.budget > 0) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |total_spending e / e.budget - 0.28| < ε :=
sorry

end spending_percentage_l1244_124474


namespace marie_magazines_sold_l1244_124476

/-- The number of magazines Marie sold -/
def magazines_sold : ℕ := 700 - 275

/-- The total number of reading materials Marie sold -/
def total_reading_materials : ℕ := 700

/-- The number of newspapers Marie sold -/
def newspapers_sold : ℕ := 275

theorem marie_magazines_sold :
  magazines_sold = 425 ∧
  magazines_sold + newspapers_sold = total_reading_materials :=
sorry

end marie_magazines_sold_l1244_124476


namespace log_equation_solution_l1244_124481

theorem log_equation_solution :
  ∃ y : ℝ, y > 0 ∧ 2 * Real.log y - 4 * Real.log 2 = 2 → y = 160 :=
by sorry

end log_equation_solution_l1244_124481


namespace group_size_from_shoes_l1244_124422

/-- Given a group of people where the total number of shoes is 20 and each person has 2 shoes,
    prove that the number of people in the group is 10. -/
theorem group_size_from_shoes (total_shoes : ℕ) (shoes_per_person : ℕ) 
    (h1 : total_shoes = 20) (h2 : shoes_per_person = 2) : 
    total_shoes / shoes_per_person = 10 := by
  sorry

end group_size_from_shoes_l1244_124422


namespace systematic_sampling_interval_l1244_124484

-- Define the total population size
def N : ℕ := 1200

-- Define the sample size
def n : ℕ := 30

-- Define the systematic sampling interval
def k : ℕ := N / n

-- Theorem to prove
theorem systematic_sampling_interval :
  k = 40 :=
by sorry

end systematic_sampling_interval_l1244_124484


namespace boat_speed_in_still_water_l1244_124448

/-- The speed of a boat in still water given its travel distances with and against a stream. -/
theorem boat_speed_in_still_water (along_stream : ℝ) (against_stream : ℝ) 
  (h_along : along_stream = 11) 
  (h_against : against_stream = 3) : 
  (along_stream + against_stream) / 2 = 7 := by
  sorry

end boat_speed_in_still_water_l1244_124448


namespace union_of_M_and_N_l1244_124493

def M : Set ℕ := {0, 1, 3}

def N : Set ℕ := {x | ∃ a ∈ M, x = 3 * a}

theorem union_of_M_and_N : M ∪ N = {0, 1, 3, 9} := by sorry

end union_of_M_and_N_l1244_124493


namespace parallelepiped_length_l1244_124469

theorem parallelepiped_length (n : ℕ) : 
  n > 6 →
  (n - 2) * (n - 4) * (n - 6) = (2 / 3) * n * (n - 2) * (n - 4) →
  n = 18 := by
sorry

end parallelepiped_length_l1244_124469


namespace fraction_equality_l1244_124482

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 10)
  (h2 : p / n = 2)
  (h3 : p / q = 1 / 5) :
  m / q = 1 := by
  sorry

end fraction_equality_l1244_124482


namespace imaginary_part_of_z_l1244_124465

theorem imaginary_part_of_z (i : ℂ) (z : ℂ) : 
  i^2 = -1 → z = i * (i - 1) → Complex.im z = -1 := by sorry

end imaginary_part_of_z_l1244_124465


namespace max_value_expression_l1244_124444

theorem max_value_expression (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hsum : a + b + c + d ≤ 4) : 
  (a^2 * (a + b))^(1/4) + (b^2 * (b + c))^(1/4) + 
  (c^2 * (c + d))^(1/4) + (d^2 * (d + a))^(1/4) ≤ 4 * 2^(1/4) :=
sorry

end max_value_expression_l1244_124444


namespace intersection_M_complement_N_l1244_124407

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x > 1}
def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (x - 4)}

-- State the theorem
theorem intersection_M_complement_N :
  M ∩ (Set.univ \ N) = Set.Ioo 1 4 := by sorry

end intersection_M_complement_N_l1244_124407


namespace tangent_slope_angle_l1244_124458

open Real

theorem tangent_slope_angle (f : ℝ → ℝ) (x : ℝ) :
  f = (λ x => Real.log (x^2 + 1)) →
  x = 1 →
  let slope := (deriv f) x
  let angle := Real.arctan slope
  angle = π / 4 := by
  sorry

end tangent_slope_angle_l1244_124458


namespace decimal_expansion_eight_elevenths_repeating_block_size_l1244_124418

/-- The smallest repeating block in the decimal expansion of 8/11 contains 2 digits. -/
theorem decimal_expansion_eight_elevenths_repeating_block_size :
  ∃ (a b : ℕ) (h : b ≠ 0),
    (8 : ℚ) / 11 = (a : ℚ) / (10^b - 1) ∧
    ∀ (c d : ℕ) (h' : d ≠ 0), (8 : ℚ) / 11 = (c : ℚ) / (10^d - 1) → b ≤ d :=
by sorry

end decimal_expansion_eight_elevenths_repeating_block_size_l1244_124418


namespace handshake_arrangement_remainder_l1244_124423

/-- The number of people in the group -/
def n : ℕ := 10

/-- The number of handshakes each person makes -/
def k : ℕ := 3

/-- Two arrangements are considered different if at least two people who shake hands
    in one arrangement don't in the other -/
def different_arrangement : Prop := sorry

/-- The total number of possible handshaking arrangements -/
def M : ℕ := sorry

/-- The theorem stating the main result -/
theorem handshake_arrangement_remainder :
  M % 500 = 84 := by sorry

end handshake_arrangement_remainder_l1244_124423


namespace parallel_line_k_value_l1244_124426

/-- Given a line passing through points (1, -7) and (k, 19) that is parallel to 3x + 4y = 12, prove that k = -101/3 -/
theorem parallel_line_k_value (k : ℝ) : 
  (∃ (m b : ℝ), (m * 1 + b = -7) ∧ (m * k + b = 19) ∧ (m = -3/4)) → 
  k = -101/3 := by
sorry

end parallel_line_k_value_l1244_124426


namespace min_value_of_S_l1244_124443

def S (x : ℝ) : ℝ := (x - 10)^2 + (x + 5)^2

theorem min_value_of_S :
  ∃ (min : ℝ), min = 112.5 ∧ ∀ (x : ℝ), S x ≥ min :=
sorry

end min_value_of_S_l1244_124443


namespace xyz_value_l1244_124471

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 12)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 16) :
  x * y * z = -4/3 := by
  sorry

end xyz_value_l1244_124471


namespace wicket_keeper_age_difference_l1244_124459

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  total_members : ℕ
  captain_age : ℕ
  team_average_age : ℕ
  wicket_keeper_age_difference : ℕ

/-- The age difference between the wicket keeper and the captain is correct
    if it satisfies the given conditions -/
def correct_age_difference (team : CricketTeam) : Prop :=
  let remaining_players := team.total_members - 2
  let remaining_average := team.team_average_age - 1
  let total_age := team.team_average_age * team.total_members
  let remaining_age := remaining_average * remaining_players
  total_age = remaining_age + team.captain_age + (team.captain_age + team.wicket_keeper_age_difference)

/-- The theorem stating that the wicket keeper is 3 years older than the captain -/
theorem wicket_keeper_age_difference (team : CricketTeam) 
  (h1 : team.total_members = 11)
  (h2 : team.captain_age = 26)
  (h3 : team.team_average_age = 23)
  : correct_age_difference team → team.wicket_keeper_age_difference = 3 := by
  sorry

end wicket_keeper_age_difference_l1244_124459


namespace committee_formation_count_l1244_124446

/-- The number of ways to choose a committee from a basketball team -/
def choose_committee (total_players : ℕ) (committee_size : ℕ) (total_guards : ℕ) : ℕ :=
  total_guards * (Nat.choose (total_players - total_guards) (committee_size - 1))

/-- Theorem: The number of ways to form the committee is 112 -/
theorem committee_formation_count :
  choose_committee 12 3 4 = 112 := by
  sorry

end committee_formation_count_l1244_124446


namespace x_value_from_fraction_equality_l1244_124450

theorem x_value_from_fraction_equality (x y : ℝ) :
  x / (x - 1) = (y^2 + 3*y + 2) / (y^2 + 3*y - 1) →
  x = (y^2 + 3*y + 2) / 3 := by
sorry

end x_value_from_fraction_equality_l1244_124450


namespace remainder_x_squared_mod_30_l1244_124424

theorem remainder_x_squared_mod_30 (x : ℤ) 
  (h1 : 5 * x ≡ 15 [ZMOD 30]) 
  (h2 : 7 * x ≡ 13 [ZMOD 30]) : 
  x^2 ≡ 21 [ZMOD 30] := by
  sorry

end remainder_x_squared_mod_30_l1244_124424


namespace cattle_land_is_40_l1244_124403

/-- Represents the land allocation of Job's farm in hectares -/
structure FarmLand where
  total : ℕ
  house_and_machinery : ℕ
  future_expansion : ℕ
  crop_production : ℕ

/-- Calculates the land dedicated to rearing cattle -/
def cattle_land (farm : FarmLand) : ℕ :=
  farm.total - (farm.house_and_machinery + farm.future_expansion + farm.crop_production)

/-- Theorem stating that the land dedicated to rearing cattle is 40 hectares -/
theorem cattle_land_is_40 (farm : FarmLand) 
    (h1 : farm.total = 150)
    (h2 : farm.house_and_machinery = 25)
    (h3 : farm.future_expansion = 15)
    (h4 : farm.crop_production = 70) : 
  cattle_land farm = 40 := by
  sorry

#eval cattle_land { total := 150, house_and_machinery := 25, future_expansion := 15, crop_production := 70 }

end cattle_land_is_40_l1244_124403


namespace hyperbola_eccentricity_range_l1244_124495

/-- Given a hyperbola with the following properties:
    - Equation: x²/a² - y²/b² = 1, where a > 0 and b > 0
    - O is the origin
    - F₁ and F₂ are the left and right foci
    - P is a point on the left branch
    - M is the midpoint of F₂P
    - |OM| = c/5, where c is the focal distance
    Then the eccentricity e of the hyperbola satisfies 1 < e ≤ 5/3 -/
theorem hyperbola_eccentricity_range 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0)
  (O : ℝ × ℝ) (F₁ F₂ P M : ℝ × ℝ)
  (hyperbola_eq : ∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 → (x, y) ∈ {p : ℝ × ℝ | p.1^2/a^2 - p.2^2/b^2 = 1})
  (O_origin : O = (0, 0))
  (F₁_left : F₁.1 < 0)
  (F₂_right : F₂.1 > 0)
  (P_left_branch : P.1 < 0)
  (M_midpoint : M = ((F₂.1 + P.1)/2, (F₂.2 + P.2)/2))
  (OM_length : Real.sqrt ((M.1 - O.1)^2 + (M.2 - O.2)^2) = c/5)
  (e_def : e = c/a) :
  1 < e ∧ e ≤ 5/3 := by sorry

end hyperbola_eccentricity_range_l1244_124495


namespace inheritance_problem_l1244_124404

/-- Proves the unique solution for the inheritance problem --/
theorem inheritance_problem (A B C : ℕ) : 
  (A + B + C = 30000) →  -- Total inheritance
  (A - B = B - C) →      -- B's relationship to A and C
  (A = B + C) →          -- A's relationship to B and C
  (A = 15000 ∧ B = 10000 ∧ C = 5000) := by
  sorry

end inheritance_problem_l1244_124404


namespace fraction_sum_l1244_124466

theorem fraction_sum : (1 : ℚ) / 6 + (5 : ℚ) / 12 = (7 : ℚ) / 12 := by sorry

end fraction_sum_l1244_124466


namespace fixed_point_theorem_l1244_124409

-- Define the line equation
def line_equation (a x : ℝ) : ℝ := a * x - 3 * a + 2

-- Theorem stating that the line passes through (3, 2) for any real number a
theorem fixed_point_theorem (a : ℝ) : line_equation a 3 = 2 := by
  sorry

end fixed_point_theorem_l1244_124409


namespace jake_candy_cost_l1244_124486

/-- The cost of a single candy given Jake's feeding allowance and sharing behavior -/
def candy_cost (feeding_allowance : ℚ) (share_fraction : ℚ) (candies_bought : ℕ) : ℚ :=
  (feeding_allowance * share_fraction) / candies_bought

theorem jake_candy_cost :
  candy_cost 4 (1/4) 5 = 1/5 := by sorry

end jake_candy_cost_l1244_124486


namespace cos_arcsin_three_fifths_l1244_124470

theorem cos_arcsin_three_fifths : Real.cos (Real.arcsin (3/5)) = 4/5 := by
  sorry

end cos_arcsin_three_fifths_l1244_124470


namespace right_triangle_area_l1244_124415

/-- The area of a right triangle with a 30-inch leg and a 34-inch hypotenuse is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 := by
sorry

end right_triangle_area_l1244_124415


namespace inequality_proof_l1244_124456

theorem inequality_proof (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) : 
  a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y) := by
  sorry

end inequality_proof_l1244_124456


namespace validSchedules_eq_1296_l1244_124479

/-- Represents a chess tournament between two universities -/
structure ChessTournament where
  university1 : Fin 3 → Type
  university2 : Fin 3 → Type
  rounds : Fin 6 → Fin 3 → (Fin 3 × Fin 3)
  no_immediate_repeat : ∀ (r : Fin 5) (i : Fin 3),
    (rounds r i).1 ≠ (rounds (r + 1) i).1 ∨ (rounds r i).2 ≠ (rounds (r + 1) i).2

/-- The number of valid tournament schedules -/
def validSchedules : ℕ := sorry

/-- Theorem stating the number of valid tournament schedules is 1296 -/
theorem validSchedules_eq_1296 : validSchedules = 1296 := by sorry

end validSchedules_eq_1296_l1244_124479


namespace smallest_angle_measure_l1244_124461

/-- Represents a triangle with angles in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  sum_180 : angle1 + angle2 + angle3 = 180
  all_positive : 0 < angle1 ∧ 0 < angle2 ∧ 0 < angle3

/-- An isosceles, obtuse triangle with one angle 50% larger than a right angle -/
def special_triangle : Triangle :=
  { angle1 := 135
    angle2 := 22.5
    angle3 := 22.5
    sum_180 := by sorry
    all_positive := by sorry }

theorem smallest_angle_measure :
  ∃ (t : Triangle), 
    (t.angle1 = 90 * 1.5) ∧  -- One angle is 50% larger than right angle
    (t.angle2 = t.angle3) ∧  -- Isosceles property
    (t.angle1 > 90) ∧        -- Obtuse triangle
    (t.angle2 = 22.5 ∧ t.angle3 = 22.5) -- The two smallest angles
    := by
  sorry

end smallest_angle_measure_l1244_124461


namespace cos_thirty_degrees_l1244_124411

theorem cos_thirty_degrees : Real.cos (30 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_thirty_degrees_l1244_124411


namespace hypotenuse_length_l1244_124437

/-- A right triangle with specific properties -/
structure RightTriangle where
  /-- First leg of the triangle -/
  a : ℝ
  /-- Second leg of the triangle -/
  b : ℝ
  /-- Hypotenuse of the triangle -/
  c : ℝ
  /-- The triangle is right-angled -/
  right_angle : a^2 + b^2 = c^2
  /-- The perimeter of the triangle is 40 -/
  perimeter : a + b + c = 40
  /-- The area of the triangle is 30 -/
  area : a * b / 2 = 30
  /-- The ratio of the legs is 3:4 -/
  leg_ratio : 3 * a = 4 * b

theorem hypotenuse_length (t : RightTriangle) : t.c = 5 * Real.sqrt 5 := by
  sorry

end hypotenuse_length_l1244_124437


namespace consecutive_integers_around_sqrt33_l1244_124499

theorem consecutive_integers_around_sqrt33 (a b : ℤ) :
  (b = a + 1) →  -- a and b are consecutive integers
  (a < Real.sqrt 33) →  -- a < √33
  (Real.sqrt 33 < b) →  -- √33 < b
  a + b = 11 := by
sorry

end consecutive_integers_around_sqrt33_l1244_124499


namespace production_rates_theorem_l1244_124492

-- Define the number of machines
def num_machines : ℕ := 5

-- Define the list of pairwise production numbers
def pairwise_production : List ℕ := [35, 39, 40, 49, 44, 46, 30, 41, 32, 36]

-- Define the function to check if a list of production rates is valid
def is_valid_production (rates : List ℕ) : Prop :=
  rates.length = num_machines ∧
  rates.sum = 98 ∧
  (∀ i j, i < j → i < rates.length → j < rates.length →
    (rates.get ⟨i, by sorry⟩ + rates.get ⟨j, by sorry⟩) ∈ pairwise_production)

-- Theorem statement
theorem production_rates_theorem :
  ∃ (rates : List ℕ), is_valid_production rates ∧ rates = [13, 17, 19, 22, 27] := by
  sorry

end production_rates_theorem_l1244_124492


namespace gregorian_calendar_properties_l1244_124491

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the Gregorian calendar system -/
structure GregorianCalendar where
  -- Add necessary fields and methods

/-- Counts occurrences of a specific day for January 1st in a 400-year cycle -/
def countJanuary1Occurrences (day : DayOfWeek) (calendar : GregorianCalendar) : Nat :=
  sorry

/-- Counts occurrences of a specific day for the 30th of each month in a 400-year cycle -/
def count30thOccurrences (day : DayOfWeek) (calendar : GregorianCalendar) : Nat :=
  sorry

theorem gregorian_calendar_properties (calendar : GregorianCalendar) :
  (countJanuary1Occurrences DayOfWeek.Sunday calendar > countJanuary1Occurrences DayOfWeek.Saturday calendar) ∧
  (∀ d : DayOfWeek, count30thOccurrences DayOfWeek.Friday calendar ≥ count30thOccurrences d calendar) :=
by sorry

end gregorian_calendar_properties_l1244_124491


namespace expression_simplification_l1244_124401

theorem expression_simplification (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1) :
  (a - (2 * a - 1) / a) / ((a^2 - 1) / a) = (a - 1) / (a + 1) :=
by sorry

end expression_simplification_l1244_124401


namespace reduced_price_calculation_l1244_124453

/-- Represents the price of oil in Rupees per kg -/
structure OilPrice where
  price : ℝ
  price_positive : price > 0

def reduction_percentage : ℝ := 0.30

def total_cost : ℝ := 700

def additional_quantity : ℝ := 3

theorem reduced_price_calculation (original_price : OilPrice) :
  let reduced_price := original_price.price * (1 - reduction_percentage)
  let original_quantity := total_cost / original_price.price
  let new_quantity := total_cost / reduced_price
  new_quantity = original_quantity + additional_quantity →
  reduced_price = 70 := by
  sorry

end reduced_price_calculation_l1244_124453


namespace complex_equation_modulus_l1244_124410

theorem complex_equation_modulus : ∃ (x y : ℝ), 
  (Complex.I + 1) * x + Complex.I * y = (Complex.I + 3 * Complex.I) * Complex.I ∧ 
  Complex.abs (x + Complex.I * y) = 5 := by
  sorry

end complex_equation_modulus_l1244_124410


namespace max_value_theorem_l1244_124442

theorem max_value_theorem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  ∀ x y z : ℝ, 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 3 → x + y^2 + z^4 ≤ 3 :=
by sorry

end max_value_theorem_l1244_124442


namespace intersection_theorem_l1244_124447

-- Define the sets M and N
def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define the intersection set
def intersection_set : Set ℝ := {x | -1 < x ∧ x < 2}

-- Theorem statement
theorem intersection_theorem : M ∩ N = intersection_set := by
  sorry

end intersection_theorem_l1244_124447


namespace vector_magnitude_relation_l1244_124419

variables {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem vector_magnitude_relation (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) (h : a = -3 • b) :
  ‖a‖ = 3 * ‖b‖ :=
by sorry

end vector_magnitude_relation_l1244_124419


namespace monthly_interest_advantage_l1244_124472

theorem monthly_interest_advantage (p : ℝ) (n : ℕ) (hp : p > 0) (hn : n > 0) :
  (1 + p / (12 * 100)) ^ (6 * n) > (1 + p / (2 * 100)) ^ n :=
sorry

end monthly_interest_advantage_l1244_124472


namespace smallest_alpha_is_eight_l1244_124413

/-- A quadratic polynomial P(x) = ax² + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic polynomial at a given x -/
def QuadraticPolynomial.eval (P : QuadraticPolynomial) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- The derivative of a quadratic polynomial at x = 0 -/
def QuadraticPolynomial.deriv_at_zero (P : QuadraticPolynomial) : ℝ :=
  P.b

/-- The property that |P(x)| ≤ 1 for x ∈ [0,1] -/
def bounded_on_unit_interval (P : QuadraticPolynomial) : Prop :=
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |P.eval x| ≤ 1

theorem smallest_alpha_is_eight :
  (∃ α : ℝ, ∀ P : QuadraticPolynomial, bounded_on_unit_interval P → |P.deriv_at_zero| ≤ α) ∧
  (∀ β : ℝ, (∀ P : QuadraticPolynomial, bounded_on_unit_interval P → |P.deriv_at_zero| ≤ β) → 8 ≤ β) :=
by sorry

end smallest_alpha_is_eight_l1244_124413


namespace aubree_beaver_count_l1244_124489

/-- The number of beavers Aubree initially saw -/
def initial_beavers : ℕ := 20

/-- The number of chipmunks Aubree initially saw -/
def initial_chipmunks : ℕ := 40

/-- The total number of animals Aubree saw -/
def total_animals : ℕ := 130

theorem aubree_beaver_count :
  initial_beavers = 20 ∧
  initial_chipmunks = 40 ∧
  total_animals = 130 ∧
  initial_beavers + initial_chipmunks + 2 * initial_beavers + (initial_chipmunks - 10) = total_animals :=
by sorry

end aubree_beaver_count_l1244_124489


namespace shaded_area_between_squares_l1244_124429

/-- The area of the shaded region between two squares -/
theorem shaded_area_between_squares (large_side small_side : ℝ) 
  (h1 : large_side = 10)
  (h2 : small_side = 5) :
  large_side^2 - small_side^2 = 75 := by
  sorry

end shaded_area_between_squares_l1244_124429


namespace binomial_expansion_coefficients_l1244_124454

theorem binomial_expansion_coefficients :
  ∀ (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  (a₀ = 1 ∧ a₁ + a₃ + a₅ = 122) :=
by
  sorry

end binomial_expansion_coefficients_l1244_124454


namespace cos_2theta_value_l1244_124431

theorem cos_2theta_value (θ : ℝ) 
  (h1 : 3 * Real.sin (2 * θ) = 4 * Real.tan θ) 
  (h2 : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.cos (2 * θ) = 1/3 := by
  sorry

end cos_2theta_value_l1244_124431


namespace distribute_10_4_l1244_124452

/-- The number of ways to distribute n identical objects among k distinct containers,
    where each container must have at least one object. -/
def distribute (n k : ℕ) : ℕ :=
  sorry

/-- Theorem stating that distributing 10 identical objects among 4 distinct containers,
    where each container must have at least one object, results in 34 unique arrangements. -/
theorem distribute_10_4 : distribute 10 4 = 34 := by
  sorry

end distribute_10_4_l1244_124452


namespace league_games_l1244_124428

theorem league_games (n : ℕ) (m : ℕ) (h1 : n = 20) (h2 : m = 4) :
  (n * (n - 1) / 2) * m = 760 := by
  sorry

end league_games_l1244_124428


namespace range_of_m_l1244_124490

-- Define the proposition
def P (m : ℝ) : Prop := ∀ x : ℝ, 5^x + 3 > m

-- Theorem statement
theorem range_of_m :
  (∃ m : ℝ, P m) → (∀ m : ℝ, P m → m ≤ 3) ∧ (∀ y : ℝ, y < 3 → P y) :=
sorry

end range_of_m_l1244_124490


namespace parallelogram_height_eq_two_thirds_rectangle_side_l1244_124460

/-- Given a rectangle with side length r and a parallelogram with base b = 1.5r,
    prove that the height of the parallelogram h = 2r/3 when their areas are equal. -/
theorem parallelogram_height_eq_two_thirds_rectangle_side 
  (r : ℝ) (b h : ℝ) (h_positive : r > 0) : 
  b = 1.5 * r → r * r = b * h → h = 2 * r / 3 := by
  sorry

end parallelogram_height_eq_two_thirds_rectangle_side_l1244_124460


namespace car_wash_earnings_per_car_l1244_124432

/-- Proves that a car wash company making $2000 in 5 days while cleaning 80 cars per day earns $5 per car --/
theorem car_wash_earnings_per_car 
  (cars_per_day : ℕ) 
  (total_days : ℕ) 
  (total_earnings : ℕ) 
  (h1 : cars_per_day = 80) 
  (h2 : total_days = 5) 
  (h3 : total_earnings = 2000) : 
  total_earnings / (cars_per_day * total_days) = 5 := by
sorry

end car_wash_earnings_per_car_l1244_124432


namespace largest_cards_per_page_l1244_124421

theorem largest_cards_per_page (album1 album2 album3 : ℕ) 
  (h1 : album1 = 1080) 
  (h2 : album2 = 1620) 
  (h3 : album3 = 540) : 
  Nat.gcd album1 (Nat.gcd album2 album3) = 540 := by
  sorry

end largest_cards_per_page_l1244_124421


namespace quadratic_root_relation_l1244_124430

/-- Given two quadratic equations, where the roots of one are three less than the roots of the other,
    prove that the constant term of the second equation is zero. -/
theorem quadratic_root_relation (b c : ℝ) : 
  (∃ r s : ℝ, 2 * r^2 - 8 * r + 6 = 0 ∧ 2 * s^2 - 8 * s + 6 = 0) →
  (∃ x y : ℝ, x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0) →
  (∀ r s x y : ℝ, 
    (2 * r^2 - 8 * r + 6 = 0 ∧ 2 * s^2 - 8 * s + 6 = 0) →
    (x^2 + b * x + c = 0 ∧ y^2 + b * y + c = 0) →
    x = r - 3 ∧ y = s - 3) →
  c = 0 := by
sorry

end quadratic_root_relation_l1244_124430


namespace female_officers_count_l1244_124445

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_ratio : ℚ) :
  total_on_duty = 210 →
  female_on_duty_ratio = 2/3 →
  female_ratio = 24/100 →
  ∃ (total_female : ℕ), total_female = 583 ∧ 
    (↑total_on_duty * female_on_duty_ratio : ℚ) = (↑total_female * female_ratio : ℚ) :=
by sorry

end female_officers_count_l1244_124445


namespace smallest_number_l1244_124449

theorem smallest_number (a b c d : ℤ) (ha : a = 2) (hb : b = 1) (hc : c = -1) (hd : d = -2) :
  d ≤ a ∧ d ≤ b ∧ d ≤ c := by
  sorry

end smallest_number_l1244_124449


namespace total_commission_is_42000_l1244_124414

def sale_price : ℝ := 10000
def commission_rate_first_100 : ℝ := 0.03
def commission_rate_after_100 : ℝ := 0.04
def total_machines_sold : ℕ := 130

def commission_first_100 : ℝ := 100 * sale_price * commission_rate_first_100
def commission_after_100 : ℝ := (total_machines_sold - 100) * sale_price * commission_rate_after_100

theorem total_commission_is_42000 :
  commission_first_100 + commission_after_100 = 42000 := by
  sorry

end total_commission_is_42000_l1244_124414


namespace smallest_number_from_digits_l1244_124497

def digits : List Nat := [2, 0, 1, 6]

def isValidPermutation (n : Nat) : Bool :=
  let digits_n := n.digits 10
  digits_n.length == 4 && digits_n.head? != some 0 && digits_n.toFinset == digits.toFinset

theorem smallest_number_from_digits :
  ∀ n : Nat, isValidPermutation n → 1026 ≤ n := by
  sorry

end smallest_number_from_digits_l1244_124497


namespace prob_all_same_color_is_correct_l1244_124436

def yellow_marbles : ℕ := 3
def green_marbles : ℕ := 7
def purple_marbles : ℕ := 5
def total_marbles : ℕ := yellow_marbles + green_marbles + purple_marbles
def drawn_marbles : ℕ := 4

def prob_all_same_color : ℚ :=
  (green_marbles * (green_marbles - 1) * (green_marbles - 2) * (green_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3)) +
  (purple_marbles * (purple_marbles - 1) * (purple_marbles - 2) * (purple_marbles - 3)) /
  (total_marbles * (total_marbles - 1) * (total_marbles - 2) * (total_marbles - 3))

theorem prob_all_same_color_is_correct : prob_all_same_color = 532 / 4095 := by
  sorry

end prob_all_same_color_is_correct_l1244_124436


namespace rain_probability_l1244_124439

theorem rain_probability (p_saturday p_sunday : ℝ) 
  (h_saturday : p_saturday = 0.6)
  (h_sunday : p_sunday = 0.4)
  (h_independent : True) -- Assumption of independence
  : p_saturday * (1 - p_sunday) + (1 - p_saturday) * p_sunday = 0.52 := by
  sorry

end rain_probability_l1244_124439


namespace complex_division_by_i_l1244_124457

theorem complex_division_by_i (i : ℂ) (h : i^2 = -1) : 
  (3 + 4*i) / i = 4 - 3*i := by sorry

end complex_division_by_i_l1244_124457


namespace function_symmetry_and_translation_l1244_124427

-- Define the exponential function
noncomputable def exp (x : ℝ) : ℝ := Real.exp x

-- Define a function that represents a horizontal translation
def translate (f : ℝ → ℝ) (h : ℝ) : ℝ → ℝ := λ x => f (x - h)

-- Define symmetry about the y-axis
def symmetricAboutYAxis (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g (-x)

-- State the theorem
theorem function_symmetry_and_translation :
  ∀ f : ℝ → ℝ, symmetricAboutYAxis (translate f 1) exp → f = λ x => exp (-x - 1) := by
  sorry


end function_symmetry_and_translation_l1244_124427


namespace multiplication_problem_l1244_124462

theorem multiplication_problem : ∃ x : ℕ, 72517 * x = 724807415 ∧ x = 9999 := by
  sorry

end multiplication_problem_l1244_124462


namespace exists_removable_factorial_for_perfect_square_l1244_124416

def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

def product_of_factorials (n : ℕ) : ℕ := (Finset.range n).prod (λ i => factorial (i + 1))

theorem exists_removable_factorial_for_perfect_square :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 100 ∧ 
  ∃ m : ℕ, product_of_factorials 100 / factorial k = m ^ 2 :=
sorry

end exists_removable_factorial_for_perfect_square_l1244_124416


namespace quiz_answer_key_count_l1244_124435

/-- The number of ways to arrange true and false answers for 6 questions 
    with an equal number of true and false answers -/
def true_false_arrangements : ℕ := Nat.choose 6 3

/-- The number of ways to choose answers for 4 multiple-choice questions 
    with 5 options each -/
def multiple_choice_arrangements : ℕ := 5^4

/-- The total number of ways to create an answer key for the quiz -/
def total_arrangements : ℕ := true_false_arrangements * multiple_choice_arrangements

theorem quiz_answer_key_count : total_arrangements = 12500 := by
  sorry

end quiz_answer_key_count_l1244_124435


namespace expression_evaluation_l1244_124400

theorem expression_evaluation : -30 + 12 * (8 / 4)^2 = 18 := by
  sorry

end expression_evaluation_l1244_124400


namespace harmonic_table_sum_remainder_l1244_124438

theorem harmonic_table_sum_remainder : ∃ k : ℕ, (2^2007 - 1) / 2007 ≡ 1 [MOD 2008] := by
  sorry

end harmonic_table_sum_remainder_l1244_124438
