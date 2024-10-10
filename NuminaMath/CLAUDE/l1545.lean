import Mathlib

namespace arithmetic_sequence_sum_l1545_154595

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 10 →
  a 4 = 24 →
  a 6 = 38 →
  a 3 + a 5 = 48 := by
sorry

end arithmetic_sequence_sum_l1545_154595


namespace hexagon_area_from_square_l1545_154566

theorem hexagon_area_from_square (s : ℝ) (h_square_area : s^2 = Real.sqrt 3) :
  let hexagon_area := 6 * (Real.sqrt 3 / 4 * s^2)
  hexagon_area = 9 / 2 := by
  sorry

end hexagon_area_from_square_l1545_154566


namespace angle_B_obtuse_l1545_154501

theorem angle_B_obtuse (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Triangle inequality
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  -- Given conditions
  (c / b < Real.cos A) ∧ (0 < A) ∧ (A < Real.pi) →
  -- Conclusion: B is obtuse
  Real.pi / 2 < B :=
by
  sorry

end angle_B_obtuse_l1545_154501


namespace fish_theorem_l1545_154527

def fish_problem (leo agrey sierra returned : ℕ) : Prop :=
  let total := leo + agrey + sierra
  agrey = leo + 20 ∧ 
  sierra = agrey + 15 ∧ 
  leo = 40 ∧ 
  returned = 30 ∧ 
  total - returned = 145

theorem fish_theorem : 
  ∃ (leo agrey sierra returned : ℕ), fish_problem leo agrey sierra returned :=
by
  sorry

end fish_theorem_l1545_154527


namespace chemical_mixture_problem_l1545_154545

/-- Represents the chemical mixture problem --/
theorem chemical_mixture_problem (a w : ℝ) 
  (h1 : a / (a + w + 2) = 1 / 4)
  (h2 : (a + 2) / (a + w + 2) = 3 / 8) :
  a = 4 := by
  sorry

end chemical_mixture_problem_l1545_154545


namespace f_decreasing_implies_a_geq_2_l1545_154531

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

-- Define the property of f being decreasing on (-∞, 2)
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 2 → y < 2 → f a x > f a y

-- Theorem statement
theorem f_decreasing_implies_a_geq_2 :
  ∀ a : ℝ, is_decreasing_on_interval a → a ≥ 2 :=
sorry

end f_decreasing_implies_a_geq_2_l1545_154531


namespace binary_multiplication_example_l1545_154592

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNum := List Nat

/-- Converts a decimal number to its binary representation -/
def to_binary (n : Nat) : BinaryNum :=
  sorry

/-- Converts a binary number to its decimal representation -/
def to_decimal (b : BinaryNum) : Nat :=
  sorry

/-- Multiplies two binary numbers -/
def binary_multiply (a b : BinaryNum) : BinaryNum :=
  sorry

theorem binary_multiplication_example :
  let a : BinaryNum := [1, 0, 1, 0, 1]  -- 10101₂
  let b : BinaryNum := [1, 0, 1]        -- 101₂
  let result : BinaryNum := [1, 1, 0, 1, 0, 0, 1]  -- 1101001₂
  binary_multiply a b = result :=
by sorry

end binary_multiplication_example_l1545_154592


namespace safari_creatures_l1545_154586

/-- Proves that given 150 creatures with 624 legs total, where some are two-legged ostriches
    and others are six-legged aliens, the number of ostriches is 69. -/
theorem safari_creatures (total_creatures : ℕ) (total_legs : ℕ) 
    (h1 : total_creatures = 150)
    (h2 : total_legs = 624) : 
  ∃ (ostriches aliens : ℕ),
    ostriches + aliens = total_creatures ∧
    2 * ostriches + 6 * aliens = total_legs ∧
    ostriches = 69 := by
  sorry

end safari_creatures_l1545_154586


namespace f_zero_gt_f_one_l1545_154533

/-- A quadratic function f(x) = x^2 - 4x + m, where m is a real constant. -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 4*x + m

/-- Theorem stating that f(0) > f(1) for any real m. -/
theorem f_zero_gt_f_one (m : ℝ) : f m 0 > f m 1 := by
  sorry

end f_zero_gt_f_one_l1545_154533


namespace motel_rent_theorem_l1545_154550

/-- Represents the total rent charged by a motel --/
def TotalRent (x y : ℕ) : ℕ := 40 * x + 60 * y

/-- The problem statement --/
theorem motel_rent_theorem (x y : ℕ) :
  (TotalRent (x + 10) (y - 10) = (TotalRent x y) / 2) →
  TotalRent x y = 800 :=
by sorry

end motel_rent_theorem_l1545_154550


namespace cost_of_six_pens_l1545_154536

/-- Given that 3 pens cost 7.5 yuan, prove that 6 pens cost 15 yuan. -/
theorem cost_of_six_pens (cost_three_pens : ℝ) (h : cost_three_pens = 7.5) :
  let cost_one_pen := cost_three_pens / 3
  cost_one_pen * 6 = 15 := by sorry

end cost_of_six_pens_l1545_154536


namespace geometric_arithmetic_sequence_problem_l1545_154544

theorem geometric_arithmetic_sequence_problem 
  (b q a d : ℝ) 
  (h1 : b = a + d)
  (h2 : b * q = a + 3 * d)
  (h3 : b * q^2 = a + 6 * d)
  (h4 : b * (b * q) * (b * q^2) = 64) :
  b = 8 / 3 := by
sorry

end geometric_arithmetic_sequence_problem_l1545_154544


namespace fair_coin_toss_probability_sum_l1545_154579

/-- Represents a fair coin --/
structure FairCoin where
  prob_heads : ℚ
  fair : prob_heads = 1/2

/-- Calculates the probability of getting exactly k heads in n tosses --/
def binomial_probability (c : FairCoin) (n k : ℕ) : ℚ :=
  (n.choose k) * c.prob_heads^k * (1 - c.prob_heads)^(n-k)

/-- The main theorem --/
theorem fair_coin_toss_probability_sum :
  ∀ (c : FairCoin),
  (binomial_probability c 5 1 = binomial_probability c 5 2) →
  ∃ (i j : ℕ),
    (binomial_probability c 5 3 = i / j) ∧
    (∀ (a b : ℕ), (a / b = i / j) → (a ≤ i ∧ b ≤ j)) ∧
    i + j = 283 :=
sorry

end fair_coin_toss_probability_sum_l1545_154579


namespace solve_for_a_find_b_find_c_find_d_l1545_154585

-- Part 1
def simultaneous_equations (a u : ℝ) : Prop :=
  3/a + 1/u = 7/2 ∧ 2/a - 3/u = 6

theorem solve_for_a : ∃ a u : ℝ, simultaneous_equations a u ∧ a = 3/2 :=
sorry

-- Part 2
def equation_with_solutions (p q b : ℝ) (a : ℝ) : Prop :=
  p * 0 + q * (3*a) + b * 1 = 1 ∧
  p * (9*a) + q * (-1) + b * 2 = 1 ∧
  p * 0 + q * (3*a) + b * 0 = 1

theorem find_b : ∃ p q b a : ℝ, equation_with_solutions p q b a ∧ b = 0 :=
sorry

-- Part 3
def line_through_points (m c b : ℝ) : Prop :=
  5 = m * (b + 4) + c ∧
  2 = m * (-2) + c

theorem find_c : ∃ m c b : ℝ, line_through_points m c b ∧ c = 3 :=
sorry

-- Part 4
def inequality_solution (c d : ℝ) : Prop :=
  ∀ x : ℝ, d ≤ x ∧ x ≤ 1 ↔ x^2 + 5*x - 2*c ≤ 0

theorem find_d : ∃ c d : ℝ, inequality_solution c d ∧ d = -6 :=
sorry

end solve_for_a_find_b_find_c_find_d_l1545_154585


namespace maryville_population_increase_l1545_154538

/-- Calculates the average annual population increase given initial and final populations and the time period. -/
def averageAnnualIncrease (initialPopulation finalPopulation : ℕ) (years : ℕ) : ℚ :=
  (finalPopulation - initialPopulation : ℚ) / years

/-- Theorem stating that the average annual population increase in Maryville between 2000 and 2005 is 3400. -/
theorem maryville_population_increase : averageAnnualIncrease 450000 467000 5 = 3400 := by
  sorry

end maryville_population_increase_l1545_154538


namespace selena_remaining_money_is_33_74_l1545_154503

/-- Calculates the amount Selena will be left with after paying for her meal including taxes. -/
def selena_remaining_money (tip : ℚ) (steak_price : ℚ) (burger_price : ℚ) (ice_cream_price : ℚ)
  (steak_tax : ℚ) (burger_tax : ℚ) (ice_cream_tax : ℚ) : ℚ :=
  let steak_total := 2 * steak_price * (1 + steak_tax)
  let burger_total := 2 * burger_price * (1 + burger_tax)
  let ice_cream_total := 3 * ice_cream_price * (1 + ice_cream_tax)
  tip - (steak_total + burger_total + ice_cream_total)

/-- Theorem stating that Selena will be left with $33.74 after paying for her meal including taxes. -/
theorem selena_remaining_money_is_33_74 :
  selena_remaining_money 99 24 3.5 2 0.07 0.06 0.08 = 33.74 := by
  sorry

end selena_remaining_money_is_33_74_l1545_154503


namespace rectangular_field_with_pond_l1545_154558

theorem rectangular_field_with_pond (l w : ℝ) : 
  l = 2 * w →                        -- length is double the width
  l * w = 2 * (8 * 8) →              -- pond area is half of field area
  l = 16 := by
sorry

end rectangular_field_with_pond_l1545_154558


namespace probability_of_six_consecutive_heads_l1545_154507

-- Define a coin flip sequence as a list of booleans (true for heads, false for tails)
def CoinFlipSequence := List Bool

-- Function to check if a sequence has at least n consecutive heads
def hasConsecutiveHeads (n : Nat) (seq : CoinFlipSequence) : Bool :=
  sorry

-- Function to generate all possible coin flip sequences of length n
def allSequences (n : Nat) : List CoinFlipSequence :=
  sorry

-- Count the number of sequences with at least n consecutive heads
def countSequencesWithConsecutiveHeads (n : Nat) (seqs : List CoinFlipSequence) : Nat :=
  sorry

-- Theorem to prove
theorem probability_of_six_consecutive_heads :
  let allSeqs := allSequences 9
  let favorableSeqs := countSequencesWithConsecutiveHeads 6 allSeqs
  (favorableSeqs : ℚ) / (allSeqs.length : ℚ) = 49 / 512 := by
  sorry

end probability_of_six_consecutive_heads_l1545_154507


namespace max_distance_for_given_tires_l1545_154548

/-- Represents the maximum distance a car can travel with tire swapping -/
def maxDistanceWithSwap (frontTireLife : ℕ) (rearTireLife : ℕ) : ℕ :=
  frontTireLife + (rearTireLife - frontTireLife) / 2

/-- Theorem stating the maximum distance for the given tire lives -/
theorem max_distance_for_given_tires :
  maxDistanceWithSwap 42000 56000 = 48000 := by
  sorry

#eval maxDistanceWithSwap 42000 56000

end max_distance_for_given_tires_l1545_154548


namespace quadratic_root_problem_l1545_154565

theorem quadratic_root_problem (b : ℝ) :
  (∃ x₀ : ℝ, x₀^2 - 4*x₀ + b = 0 ∧ (-x₀)^2 + 4*(-x₀) - b = 0) →
  (∃ x : ℝ, x > 0 ∧ x^2 + b*x - 4 = 0 ∧ x = 2) :=
by sorry

end quadratic_root_problem_l1545_154565


namespace range_of_m_l1545_154551

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + x^2 - 2*a*x + 1

theorem range_of_m (m : ℝ) :
  (∃ x₀ : ℝ, x₀ ∈ Set.Ioc 0 1 ∧
    ∀ a : ℝ, a ∈ Set.Icc (-2) 0 →
      2*m*(Real.exp a) + f a x₀ > a^2 + 2*a + 4) →
  m ∈ Set.Ioo 1 (Real.exp 2) :=
sorry

end range_of_m_l1545_154551


namespace employee_salary_problem_l1545_154577

/-- Proves that given 20 employees, if adding a manager's salary of 3400
    increases the average salary by 100, then the initial average salary
    of the employees is 1300. -/
theorem employee_salary_problem (n : ℕ) (manager_salary : ℕ) (salary_increase : ℕ) 
    (h1 : n = 20)
    (h2 : manager_salary = 3400)
    (h3 : salary_increase = 100) :
    ∃ (initial_avg : ℕ),
      initial_avg * n + manager_salary = (initial_avg + salary_increase) * (n + 1) ∧
      initial_avg = 1300 := by
  sorry

end employee_salary_problem_l1545_154577


namespace power_of_two_triplets_l1545_154518

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def satisfies_conditions (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

theorem power_of_two_triplets :
  ∀ a b c : ℕ,
    satisfies_conditions a b c ↔
      (a = 2 ∧ b = 2 ∧ c = 2) ∨
      (a = 2 ∧ b = 2 ∧ c = 3) ∨
      (a = 2 ∧ b = 3 ∧ c = 3) ∨
      (a = 2 ∧ b = 3 ∧ c = 6) ∨
      (a = 3 ∧ b = 5 ∧ c = 7) :=
by sorry

end power_of_two_triplets_l1545_154518


namespace sodium_chloride_formation_l1545_154584

-- Define the chemical reaction
structure Reaction where
  hcl : ℕ  -- moles of Hydrochloric acid
  nahco3 : ℕ  -- moles of Sodium bicarbonate
  nacl : ℕ  -- moles of Sodium chloride produced

-- Define the stoichiometric relationship
def stoichiometric_relationship (r : Reaction) : Prop :=
  r.nacl = min r.hcl r.nahco3

-- Theorem statement
theorem sodium_chloride_formation (r : Reaction) 
  (h1 : r.hcl = 2)  -- 2 moles of Hydrochloric acid
  (h2 : r.nahco3 = 2)  -- 2 moles of Sodium bicarbonate
  (h3 : stoichiometric_relationship r)  -- The reaction follows the stoichiometric relationship
  : r.nacl = 2 :=
by sorry

end sodium_chloride_formation_l1545_154584


namespace floor_ceiling_sum_seven_l1545_154559

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℤ) + (⌈x⌉ : ℤ) = 7 ↔ 3 < x ∧ x < 4 := by
  sorry

end floor_ceiling_sum_seven_l1545_154559


namespace point_on_line_l1545_154555

/-- Given a line passing through points (3, 6) and (-4, 0), 
    prove that if (x, 10) lies on this line, then x = 23/3 -/
theorem point_on_line (x : ℚ) : 
  (∀ (t : ℚ), (3 + t * (-4 - 3), 6 + t * (0 - 6)) = (x, 10)) → x = 23 / 3 := by
  sorry

end point_on_line_l1545_154555


namespace trumpet_cost_l1545_154510

/-- The cost of a trumpet, given the total amount spent and the cost of a song book. -/
theorem trumpet_cost (total_spent song_book_cost : ℚ) 
  (h1 : total_spent = 151)
  (h2 : song_book_cost = 5.84) :
  total_spent - song_book_cost = 145.16 := by
  sorry

end trumpet_cost_l1545_154510


namespace intersection_of_A_and_B_l1545_154582

def A : Set ℕ := {1, 2, 3, 4, 5}
def B : Set ℕ := {2, 4, 5, 7, 8}

theorem intersection_of_A_and_B : A ∩ B = {2, 4, 5} := by
  sorry

end intersection_of_A_and_B_l1545_154582


namespace smallest_part_of_proportional_division_l1545_154525

theorem smallest_part_of_proportional_division (total : ℝ) (p1 p2 p3 : ℝ) :
  total = 105 →
  p1 + p2 + p3 = total →
  p1 / 2 = p2 / (1/2) →
  p1 / 2 = p3 / (1/4) →
  min p1 (min p2 p3) = 10.5 :=
by sorry

end smallest_part_of_proportional_division_l1545_154525


namespace sum_even_implies_one_even_l1545_154570

theorem sum_even_implies_one_even (a b c : ℕ) :
  Even (a + b + c) → ¬(Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end sum_even_implies_one_even_l1545_154570


namespace fraction_simplification_l1545_154542

theorem fraction_simplification (a b : ℝ) (ha : a ≠ 0) (hab : a ≠ b) :
  (a - b) / a / (a - (2 * a * b - b^2) / a) = 1 / (a - b) := by
  sorry

end fraction_simplification_l1545_154542


namespace polynomial_remainder_l1545_154563

theorem polynomial_remainder (x : ℝ) : 
  let Q := fun (x : ℝ) => 8*x^4 - 18*x^3 - 6*x^2 + 4*x - 30
  let divisor := fun (x : ℝ) => 2*x - 8
  Q 4 = 786 ∧ (∃ P : ℝ → ℝ, ∀ x, Q x = P x * divisor x + 786) :=
by sorry

end polynomial_remainder_l1545_154563


namespace trapezium_area_with_triangle_removed_l1545_154553

/-- The area of a trapezium with a right triangle removed -/
theorem trapezium_area_with_triangle_removed
  (e f g h : ℝ)
  (e_pos : 0 < e)
  (f_pos : 0 < f)
  (g_pos : 0 < g)
  (h_pos : 0 < h) :
  let trapezium_area := (e + f) * (g + h)
  let triangle_area := h^2 / 2
  trapezium_area - triangle_area = (e + f) * (g + h) - h^2 / 2 :=
by sorry

end trapezium_area_with_triangle_removed_l1545_154553


namespace max_distance_between_circles_l1545_154574

/-- Circle C₁ with equation x² + (y+3)² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + (y+3)^2 = 1

/-- Circle C₂ with equation (x-4)² + y² = 4 -/
def C₂ (x y : ℝ) : Prop := (x-4)^2 + y^2 = 4

/-- The maximum distance between any point on C₁ and any point on C₂ is 8 -/
theorem max_distance_between_circles :
  ∃ (max_dist : ℝ),
    (∀ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ → C₂ x₂ y₂ → 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ max_dist^2) ∧
    (∃ (x₁ y₁ x₂ y₂ : ℝ), C₁ x₁ y₁ ∧ C₂ x₂ y₂ ∧ 
      (x₁ - x₂)^2 + (y₁ - y₂)^2 = max_dist^2) ∧
    max_dist = 8 :=
by sorry

end max_distance_between_circles_l1545_154574


namespace mary_fruits_left_l1545_154594

/-- Calculates the total number of fruits left after eating some. -/
def fruits_left (initial_apples initial_oranges initial_blueberries eaten : ℕ) : ℕ :=
  (initial_apples - eaten) + (initial_oranges - eaten) + (initial_blueberries - eaten)

/-- Proves that Mary has 26 fruits left after eating one of each. -/
theorem mary_fruits_left : fruits_left 14 9 6 1 = 26 := by
  sorry

end mary_fruits_left_l1545_154594


namespace lawrence_county_kids_count_l1545_154505

/-- The number of kids staying home during summer break in Lawrence county -/
def kids_staying_home : ℕ := 907611

/-- The number of kids going to camp from Lawrence county -/
def kids_going_to_camp : ℕ := 455682

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := kids_staying_home + kids_going_to_camp

/-- Theorem stating that the total number of kids in Lawrence county
    is equal to the sum of kids staying home and kids going to camp -/
theorem lawrence_county_kids_count :
  total_kids = kids_staying_home + kids_going_to_camp := by
  sorry

end lawrence_county_kids_count_l1545_154505


namespace C_is_circle_when_k_is_2_C_hyperbola_y_axis_implies_k_less_than_neg_one_l1545_154598

-- Define the curve C
def C (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (5 - k) = 1

-- Theorem 1: When k=2, curve C is a circle
theorem C_is_circle_when_k_is_2 :
  ∃ (r : ℝ), ∀ (x y : ℝ), C 2 x y ↔ x^2 + y^2 = r^2 :=
sorry

-- Theorem 2: If curve C is a hyperbola with foci on the y-axis, then k < -1
theorem C_hyperbola_y_axis_implies_k_less_than_neg_one :
  (∃ (a b : ℝ), ∀ (x y : ℝ), C k x y ↔ y^2 / a^2 - x^2 / b^2 = 1) → k < -1 :=
sorry

end C_is_circle_when_k_is_2_C_hyperbola_y_axis_implies_k_less_than_neg_one_l1545_154598


namespace smallest_positive_solution_l1545_154591

theorem smallest_positive_solution :
  ∃! (x y : ℝ), x > 0 ∧ y > 0 ∧
  x^2 - 3*x + 2.5 = Real.sin y - 0.75 ∧
  (∀ (x' y' : ℝ), x' > 0 → y' > 0 → x'^2 - 3*x' + 2.5 = Real.sin y' - 0.75 → x ≤ x' ∧ y ≤ y') ∧
  x = 3/2 ∧ y = Real.pi/2 := by
sorry

end smallest_positive_solution_l1545_154591


namespace product_zero_iff_one_zero_l1545_154568

theorem product_zero_iff_one_zero (a b c : ℝ) : a * b * c = 0 ↔ a = 0 ∨ b = 0 ∨ c = 0 := by
  sorry

end product_zero_iff_one_zero_l1545_154568


namespace soccer_games_played_l1545_154583

theorem soccer_games_played (wins losses ties total : ℕ) : 
  wins + losses + ties = total →
  4 * ties = wins →
  3 * ties = losses →
  losses = 9 →
  total = 24 := by
sorry

end soccer_games_played_l1545_154583


namespace complex_pure_imaginary_l1545_154519

theorem complex_pure_imaginary (m : ℝ) : 
  (m^2 - 4 + (m + 2)*Complex.I = 0) → m = 2 :=
sorry


end complex_pure_imaginary_l1545_154519


namespace hyperbola_foci_distance_l1545_154576

/-- The distance between the foci of a hyperbola with equation y²/25 - x²/16 = 1 is 2√41 -/
theorem hyperbola_foci_distance : 
  let a : ℝ := 5
  let b : ℝ := 4
  let c : ℝ := Real.sqrt (a^2 + b^2)
  2 * c = 2 * Real.sqrt 41 :=
by sorry

end hyperbola_foci_distance_l1545_154576


namespace special_function_properties_l1545_154517

open Real

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x y, f (x + y) = f x + f y - 1) ∧
  (∀ x, x > 0 → f x > 1) ∧
  (f 3 = 4)

/-- The main theorem stating the properties of the special function -/
theorem special_function_properties (f : ℝ → ℝ) (hf : special_function f) :
  (∀ x y, x < y → f x < f y) ∧ (f 1 = 2) := by
  sorry

end special_function_properties_l1545_154517


namespace smallest_n_divisible_by_20_l1545_154515

theorem smallest_n_divisible_by_20 :
  ∃ (n : ℕ), n ≥ 4 ∧ n = 9 ∧
  (∀ (S : Finset ℤ), S.card = n →
    ∃ (a b c d : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    20 ∣ (a + b - c - d)) ∧
  (∀ (m : ℕ), m < 9 →
    ∃ (S : Finset ℤ), S.card = m ∧
    ∀ (a b c d : ℤ), a ∈ S → b ∈ S → c ∈ S → d ∈ S →
    a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
    ¬(20 ∣ (a + b - c - d))) :=
by sorry

end smallest_n_divisible_by_20_l1545_154515


namespace min_value_theorem_l1545_154596

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 2) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → 2 * x' + y' = 2 → 1 / x' - y' ≥ 2 * Real.sqrt 2 - 2) ∧
  (1 / (Real.sqrt 2 / 2) - (2 - Real.sqrt 2) = 2 * Real.sqrt 2 - 2) := by
  sorry

end min_value_theorem_l1545_154596


namespace remaining_cube_volume_l1545_154573

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) : 
  cube_side = 8 → cylinder_radius = 2 → 
  (cube_side ^ 3 : ℝ) - π * cylinder_radius ^ 2 * cube_side = 512 - 32 * π := by
  sorry

end remaining_cube_volume_l1545_154573


namespace circle_and_quadratic_inequality_l1545_154540

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 - 2*a*x + y^2 + 2*a^2 - 5*a + 4 = 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + (a-1)*x + 1 > 0

-- Theorem statement
theorem circle_and_quadratic_inequality (a : ℝ) :
  p a ∧ q a → 1 < a ∧ a < 3 :=
by sorry

end circle_and_quadratic_inequality_l1545_154540


namespace waiter_tables_l1545_154552

/-- Calculates the number of tables given the initial number of customers,
    the number of customers who left, and the number of people at each remaining table. -/
def calculate_tables (initial_customers : ℕ) (customers_left : ℕ) (people_per_table : ℕ) : ℕ :=
  (initial_customers - customers_left) / people_per_table

/-- Theorem stating that for the given problem, the number of tables is 5. -/
theorem waiter_tables : calculate_tables 62 17 9 = 5 := by
  sorry


end waiter_tables_l1545_154552


namespace triangle_side_calculation_l1545_154581

theorem triangle_side_calculation (A B C : ℝ) (a b c : ℝ) 
  (h1 : 0 < A ∧ A < π) 
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : A + B + C = π)
  (h5 : Real.cos A = 4/5)
  (h6 : Real.cos C = 5/13)
  (h7 : a = 13)
  (h8 : a / Real.sin A = b / Real.sin B)
  (h9 : b / Real.sin B = c / Real.sin C)
  : b = 21 := by
  sorry

end triangle_side_calculation_l1545_154581


namespace susan_reading_time_l1545_154528

/-- Represents the ratio of time spent on different activities -/
structure TimeRatio where
  swimming : ℕ
  reading : ℕ
  hangingOut : ℕ

/-- Calculates the time spent on an activity given the total time of another activity -/
def calculateTime (ratio : TimeRatio) (knownActivity : ℕ) (knownTime : ℕ) (targetActivity : ℕ) : ℕ :=
  (targetActivity * knownTime) / knownActivity

theorem susan_reading_time (ratio : TimeRatio) 
    (h1 : ratio.swimming = 1)
    (h2 : ratio.reading = 4)
    (h3 : ratio.hangingOut = 10)
    (h4 : calculateTime ratio ratio.hangingOut 20 ratio.reading = 8) : 
  ∃ (readingTime : ℕ), readingTime = 8 ∧ 
    readingTime = calculateTime ratio ratio.hangingOut 20 ratio.reading :=
by sorry

end susan_reading_time_l1545_154528


namespace inheritance_calculation_l1545_154502

theorem inheritance_calculation (x : ℝ) : 
  0.25 * x + 0.15 * (0.75 * x - 5000) + 5000 = 16500 → x = 33794 :=
by sorry

end inheritance_calculation_l1545_154502


namespace polygon_diagonals_l1545_154511

theorem polygon_diagonals (n : ℕ) (h : n > 2) :
  (360 / (360 / n) : ℚ) - 3 = 9 :=
sorry

end polygon_diagonals_l1545_154511


namespace largest_in_set_l1545_154541

def S (a : ℝ) : Set ℝ := {-2*a, 3*a, 18/a, a^2, 2}

theorem largest_in_set :
  ∀ a : ℝ, a = 3 → 
  ∃ m : ℝ, m ∈ S a ∧ ∀ x ∈ S a, x ≤ m ∧ 
  m = 3*a ∧ m = a^2 := by sorry

end largest_in_set_l1545_154541


namespace parabola_transformation_l1545_154537

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a, b := -2 * p.a * h + p.b, c := p.a * h^2 - p.b * h + p.c }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a, b := p.b, c := p.c + v }

theorem parabola_transformation (x : ℝ) :
  let p₀ : Parabola := { a := 1, b := 0, c := 0 }  -- y = x²
  let p₁ := shift_horizontal p₀ 3                  -- shift 3 units right
  let p₂ := shift_vertical p₁ 4                    -- shift 4 units up
  p₂.a * x^2 + p₂.b * x + p₂.c = (x - 3)^2 + 4 := by
  sorry

end parabola_transformation_l1545_154537


namespace sum_faces_edges_vertices_num_diagonals_square_base_l1545_154530

/-- A square pyramid is a polyhedron with a square base and triangular sides. -/
structure SquarePyramid where
  /-- The number of faces in a square pyramid -/
  faces : Nat
  /-- The number of edges in a square pyramid -/
  edges : Nat
  /-- The number of vertices in a square pyramid -/
  vertices : Nat
  /-- The number of sides in the base of a square pyramid -/
  base_sides : Nat

/-- Properties of a square pyramid -/
def square_pyramid : SquarePyramid :=
  { faces := 5
  , edges := 8
  , vertices := 5
  , base_sides := 4 }

/-- The sum of faces, edges, and vertices in a square pyramid is 18 -/
theorem sum_faces_edges_vertices (sp : SquarePyramid) : 
  sp.faces + sp.edges + sp.vertices = 18 := by
  sorry

/-- The number of diagonals in a polygon -/
def num_diagonals (n : Nat) : Nat :=
  n * (n - 3) / 2

/-- The number of diagonals in the square base of a square pyramid is 2 -/
theorem num_diagonals_square_base (sp : SquarePyramid) : 
  num_diagonals sp.base_sides = 2 := by
  sorry

end sum_faces_edges_vertices_num_diagonals_square_base_l1545_154530


namespace semicircle_problem_l1545_154524

theorem semicircle_problem (N : ℕ) (r : ℝ) (h_positive : r > 0) : 
  let A := (N * π * r^2) / 2
  let B := (π * r^2 / 2) * (N^2 - N)
  (N ≥ 1) → (A / B = 1 / 24) → (N = 25) := by
sorry

end semicircle_problem_l1545_154524


namespace product_of_repeating_decimal_and_nine_l1545_154500

theorem product_of_repeating_decimal_and_nine : ∃ (s : ℚ),
  (∀ (n : ℕ), s * 10^(3*n) - s * 10^(3*n-3) = 123 * 10^(3*n-3)) ∧
  s * 9 = 41 / 37 := by
  sorry

end product_of_repeating_decimal_and_nine_l1545_154500


namespace cone_surface_area_l1545_154535

/-- The surface area of a cone with base radius 1 and height √3 is 3π. -/
theorem cone_surface_area :
  let r : ℝ := 1
  let h : ℝ := Real.sqrt 3
  let l : ℝ := Real.sqrt (r^2 + h^2)
  let surface_area : ℝ := π * r^2 + π * r * l
  surface_area = 3 * π :=
by sorry

end cone_surface_area_l1545_154535


namespace taxi_cost_formula_correct_l1545_154593

/-- Represents the total cost in dollars for a taxi ride -/
def taxiCost (T : ℕ) : ℤ :=
  10 + 5 * T - 10 * (if T > 5 then 1 else 0)

/-- Theorem stating the correctness of the taxi cost formula -/
theorem taxi_cost_formula_correct (T : ℕ) :
  taxiCost T = 10 + 5 * T - 10 * (if T > 5 then 1 else 0) := by
  sorry

#check taxi_cost_formula_correct

end taxi_cost_formula_correct_l1545_154593


namespace max_leap_years_in_period_l1545_154514

/-- The number of years in the period -/
def period : ℕ := 125

/-- The interval between leap years -/
def leap_year_interval : ℕ := 5

/-- The maximum number of leap years in the given period -/
def max_leap_years : ℕ := period / leap_year_interval

theorem max_leap_years_in_period :
  max_leap_years = 25 := by sorry

end max_leap_years_in_period_l1545_154514


namespace fraction_equality_l1545_154547

theorem fraction_equality (a b : ℚ) (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := by
  sorry

end fraction_equality_l1545_154547


namespace absolute_value_theorem_l1545_154561

theorem absolute_value_theorem (x q : ℝ) (h1 : |x - 3| = q) (h2 : x < 3) :
  x - q = 3 - 2*q := by sorry

end absolute_value_theorem_l1545_154561


namespace classroom_population_classroom_population_is_8_l1545_154587

theorem classroom_population : ℕ :=
  let student_count : ℕ := sorry
  let student_avg_age : ℚ := 8
  let total_avg_age : ℚ := 11
  let teacher_age : ℕ := 32

  have h1 : (student_count * student_avg_age + teacher_age) / (student_count + 1) = total_avg_age := by sorry

  student_count + 1

theorem classroom_population_is_8 : classroom_population = 8 := by sorry

end classroom_population_classroom_population_is_8_l1545_154587


namespace cube_third_yellow_faces_l1545_154562

/-- Represents a cube with side length n -/
structure Cube where
  n : ℕ

/-- Total number of faces of unit cubes after division -/
def totalFaces (c : Cube) : ℕ := 6 * c.n^3

/-- Number of yellow faces after division -/
def yellowFaces (c : Cube) : ℕ := 6 * c.n^2

/-- Condition for exactly one-third of faces being yellow -/
def oneThirdYellow (c : Cube) : Prop :=
  3 * yellowFaces c = totalFaces c

/-- Theorem stating that n = 3 satisfies the condition -/
theorem cube_third_yellow_faces :
  ∃ (c : Cube), c.n = 3 ∧ oneThirdYellow c :=
sorry

end cube_third_yellow_faces_l1545_154562


namespace difference_of_extreme_valid_numbers_l1545_154589

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ 
  (n.digits 10).count 2 = 3 ∧ 
  (n.digits 10).count 0 = 1

def largest_valid_number : ℕ := 2220
def smallest_valid_number : ℕ := 2022

theorem difference_of_extreme_valid_numbers :
  largest_valid_number - smallest_valid_number = 198 ∧
  is_valid_number largest_valid_number ∧
  is_valid_number smallest_valid_number ∧
  (∀ n : ℕ, is_valid_number n → n ≤ largest_valid_number) ∧
  (∀ n : ℕ, is_valid_number n → smallest_valid_number ≤ n) :=
by sorry

end difference_of_extreme_valid_numbers_l1545_154589


namespace lunks_for_apples_l1545_154599

/-- Exchange rate between lunks and kunks -/
def lunk_to_kunk_rate : ℚ := 6 / 4

/-- Exchange rate between kunks and apples -/
def kunk_to_apple_rate : ℚ := 5 / 3

/-- Number of apples to purchase -/
def apples_to_buy : ℕ := 20

/-- Calculate the number of lunks needed to buy a given number of apples -/
def lunks_needed (apples : ℕ) : ℚ :=
  (apples : ℚ) / kunk_to_apple_rate / lunk_to_kunk_rate

theorem lunks_for_apples :
  lunks_needed apples_to_buy = 8 := by
  sorry

end lunks_for_apples_l1545_154599


namespace shoe_problem_contradiction_l1545_154590

theorem shoe_problem_contradiction (becky bobby bonny : ℕ) : 
  (bonny = 2 * becky - 5) →
  (bobby = 3 * becky) →
  (bonny = bobby) →
  False :=
by sorry

end shoe_problem_contradiction_l1545_154590


namespace triangle_shape_l1545_154508

theorem triangle_shape (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (eq : Real.sin C + Real.sin (B - A) = Real.sin (2 * A)) : 
  (A = B ∨ A = π / 2) := by
  sorry

end triangle_shape_l1545_154508


namespace seating_arrangements_count_l1545_154543

/-- Represents a seating arrangement in an examination room -/
structure ExamRoom where
  rows : Nat
  columns : Nat
  total_seats : Nat

/-- Calculates the number of possible seating arrangements for two students
    who cannot sit adjacent to each other in the given exam room -/
def count_seating_arrangements (room : ExamRoom) : Nat :=
  sorry

/-- Theorem stating that the number of seating arrangements for two students
    in a 5x6 exam room with 30 seats, where they cannot sit adjacent to each other,
    is 772 -/
theorem seating_arrangements_count :
  let exam_room : ExamRoom := ⟨5, 6, 30⟩
  count_seating_arrangements exam_room = 772 := by sorry

end seating_arrangements_count_l1545_154543


namespace unique_circle_circumference_equals_area_l1545_154539

theorem unique_circle_circumference_equals_area :
  ∃! r : ℝ, r > 0 ∧ 2 * Real.pi * r = Real.pi * r^2 := by sorry

end unique_circle_circumference_equals_area_l1545_154539


namespace pipe_filling_time_l1545_154556

theorem pipe_filling_time (p q r : ℝ) (hp : p = 3) (hr : r = 18) (hall : 1/p + 1/q + 1/r = 1/2) :
  q = 9 := by
sorry

end pipe_filling_time_l1545_154556


namespace negation_of_absolute_value_geq_one_l1545_154567

theorem negation_of_absolute_value_geq_one :
  (¬ ∀ x : ℝ, |x| ≥ 1) ↔ (∃ x : ℝ, |x| < 1) :=
by sorry

end negation_of_absolute_value_geq_one_l1545_154567


namespace arithmetic_progression_squares_l1545_154554

theorem arithmetic_progression_squares (a d : ℝ) : 
  (a - d)^2 + a^2 = 100 ∧ a^2 + (a + d)^2 = 164 →
  ((a - d, a, a + d) = (6, 8, 10) ∨
   (a - d, a, a + d) = (-10, -8, -6) ∨
   (a - d, a, a + d) = (-7 * Real.sqrt 2, Real.sqrt 2, 9 * Real.sqrt 2) ∨
   (a - d, a, a + d) = (10 * Real.sqrt 2, 8 * Real.sqrt 2, Real.sqrt 2)) :=
by sorry

end arithmetic_progression_squares_l1545_154554


namespace wire_length_theorem_l1545_154529

/-- Represents the wire and pole configuration -/
structure WireConfig where
  initial_poles : ℕ
  initial_distance : ℝ
  new_distance_increase : ℝ
  total_length : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem wire_length_theorem (config : WireConfig) 
  (h1 : config.initial_poles = 26)
  (h2 : config.new_distance_increase = 5/3)
  (h3 : (config.initial_poles - 1) * (config.initial_distance + config.new_distance_increase) = config.initial_poles * config.initial_distance - config.initial_distance) :
  config.total_length = 1000 := by
  sorry


end wire_length_theorem_l1545_154529


namespace investment_principal_l1545_154516

/-- Proves that an investment with a monthly interest payment of $228 and a simple annual interest rate of 9% has a principal amount of $30,400. -/
theorem investment_principal (monthly_interest : ℝ) (annual_rate : ℝ) (principal : ℝ) : 
  monthly_interest = 228 →
  annual_rate = 0.09 →
  principal = (monthly_interest * 12) / annual_rate →
  principal = 30400 := by
  sorry


end investment_principal_l1545_154516


namespace min_balls_same_color_l1545_154580

/-- Represents the number of different colors of balls in the bag -/
def num_colors : ℕ := 2

/-- Represents the minimum number of balls to draw -/
def min_balls : ℕ := 3

/-- Theorem stating that given a bag with balls of two colors, 
    the minimum number of balls that must be drawn to ensure 
    at least two balls of the same color is 3 -/
theorem min_balls_same_color :
  ∀ (n : ℕ), n ≥ min_balls → 
  ∃ (color : Fin num_colors), (n.choose 2) > 0 := by
  sorry

end min_balls_same_color_l1545_154580


namespace price_fluctuation_l1545_154571

theorem price_fluctuation (p : ℝ) (original_price : ℝ) : 
  (original_price * (1 + p / 100) * (1 - p / 100) = 1) →
  (original_price = 10000 / (10000 - p^2)) :=
by sorry

end price_fluctuation_l1545_154571


namespace new_person_weight_l1545_154513

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℚ) (replaced_weight : ℚ) : 
  initial_count = 8 →
  weight_increase = 7/2 →
  replaced_weight = 62 →
  (initial_count : ℚ) * weight_increase + replaced_weight = 90 :=
by sorry

end new_person_weight_l1545_154513


namespace calculation_proof_l1545_154522

theorem calculation_proof : (-8 - 1/3) - 12 - (-70) - (-8 - 1/3) = 58 := by
  sorry

end calculation_proof_l1545_154522


namespace intersection_with_complement_of_B_l1545_154597

variable (U A B : Finset ℕ)

theorem intersection_with_complement_of_B (hU : U = {1, 2, 3, 4, 5, 6, 7})
  (hA : A = {3, 4, 5}) (hB : B = {1, 3, 6}) :
  A ∩ (U \ B) = {4, 5} := by sorry

end intersection_with_complement_of_B_l1545_154597


namespace evaluate_F_with_f_l1545_154523

-- Define function f
def f (a : ℝ) : ℝ := a^2 - 1

-- Define function F
def F (a b : ℝ) : ℝ := 3*b^2 + 2*a

-- Theorem statement
theorem evaluate_F_with_f : F 2 (f 3) = 196 := by
  sorry

end evaluate_F_with_f_l1545_154523


namespace regular_hexagon_perimeter_l1545_154572

/-- The perimeter of a regular hexagon given its radius -/
theorem regular_hexagon_perimeter (radius : ℝ) : 
  radius = 3 → 6 * radius = 18 := by
  sorry

end regular_hexagon_perimeter_l1545_154572


namespace sum_of_digits_653xy_divisible_by_80_l1545_154520

def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = b * k

theorem sum_of_digits_653xy_divisible_by_80 (x y : ℕ) :
  x < 10 →
  y < 10 →
  is_divisible_by (653 * 100 + x * 10 + y) 80 →
  x + y = 8 := by
  sorry

end sum_of_digits_653xy_divisible_by_80_l1545_154520


namespace probability_not_blue_l1545_154578

def odds_blue : ℚ := 5 / 6

theorem probability_not_blue (odds : ℚ) (h : odds = odds_blue) :
  1 - odds / (1 + odds) = 6 / 11 := by
  sorry

end probability_not_blue_l1545_154578


namespace decimal_equality_and_unit_l1545_154575

/-- Represents a number with its counting unit -/
structure NumberWithUnit where
  value : ℝ
  unit : ℝ

/-- The statement we want to prove false -/
def statement (a b : NumberWithUnit) : Prop :=
  a.value = b.value ∧ a.unit = b.unit

/-- The theorem to prove -/
theorem decimal_equality_and_unit (a b : NumberWithUnit) 
  (h1 : a.value = b.value)
  (h2 : a.unit = 1)
  (h3 : b.unit = 0.1) : 
  ¬(statement a b) := by
  sorry

#check decimal_equality_and_unit

end decimal_equality_and_unit_l1545_154575


namespace star_calculation_l1545_154560

/-- The ⋆ operation for real numbers -/
def star (x y : ℝ) : ℝ := (x^2 + y) * (x - y)

/-- Theorem stating that 2 ⋆ (3 ⋆ 4) = -135 -/
theorem star_calculation : star 2 (star 3 4) = -135 := by sorry

end star_calculation_l1545_154560


namespace max_abs_z_l1545_154532

theorem max_abs_z (z : ℂ) (h : Complex.abs (z - (0 : ℂ) + 2 * Complex.I) = 1) :
  ∃ (M : ℝ), M = 3 ∧ ∀ w : ℂ, Complex.abs (w - (0 : ℂ) + 2 * Complex.I) = 1 → Complex.abs w ≤ M :=
by
  sorry

end max_abs_z_l1545_154532


namespace simplify_expression_l1545_154504

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) : x^3 * (y^3 / x)^2 = x * y^6 := by
  sorry

end simplify_expression_l1545_154504


namespace divisibility_of_quotient_l1545_154549

theorem divisibility_of_quotient (a b n : ℕ) (h1 : a ≠ b) (h2 : n ∣ (a^n - b^n)) :
  n ∣ ((a^n - b^n) / (a - b)) := by
  sorry

end divisibility_of_quotient_l1545_154549


namespace incorrect_classification_l1545_154509

/-- Represents a proof method -/
inductive ProofMethod
| Synthetic
| Analytic

/-- Represents the nature of a proof -/
inductive ProofNature
| Direct
| Indirect

/-- Defines the correct classification of proof methods -/
def correct_classification (method : ProofMethod) : ProofNature :=
  match method with
  | ProofMethod.Synthetic => ProofNature.Direct
  | ProofMethod.Analytic => ProofNature.Direct

/-- Theorem stating that the given classification is incorrect -/
theorem incorrect_classification :
  ¬(correct_classification ProofMethod.Synthetic = ProofNature.Direct ∧
    correct_classification ProofMethod.Analytic = ProofNature.Indirect) :=
by sorry

end incorrect_classification_l1545_154509


namespace hikers_room_arrangements_l1545_154588

theorem hikers_room_arrangements (n : ℕ) (k : ℕ) : n = 5 → k = 3 → Nat.choose n k = 10 := by
  sorry

end hikers_room_arrangements_l1545_154588


namespace rectangular_plot_perimeter_l1545_154564

/-- Given a rectangular plot with length 10 meters more than width,
    and fencing cost of 1430 Rs at 6.5 Rs/meter,
    prove the perimeter is 220 meters. -/
theorem rectangular_plot_perimeter :
  ∀ (width length : ℝ),
  length = width + 10 →
  6.5 * (2 * (length + width)) = 1430 →
  2 * (length + width) = 220 :=
by
  sorry

end rectangular_plot_perimeter_l1545_154564


namespace raja_income_proof_l1545_154506

/-- Raja's monthly income in rupees -/
def monthly_income : ℝ := 25000

/-- The amount Raja saves in rupees -/
def savings : ℝ := 5000

/-- Percentage of income spent on household items -/
def household_percentage : ℝ := 0.60

/-- Percentage of income spent on clothes -/
def clothes_percentage : ℝ := 0.10

/-- Percentage of income spent on medicines -/
def medicine_percentage : ℝ := 0.10

theorem raja_income_proof :
  monthly_income * household_percentage +
  monthly_income * clothes_percentage +
  monthly_income * medicine_percentage +
  savings = monthly_income :=
by sorry

end raja_income_proof_l1545_154506


namespace derivative_through_point_l1545_154569

theorem derivative_through_point (a : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2 + a*x + 1
  let f' : ℝ → ℝ := λ x => 2*x + a
  f' 2 = 4 → a = 0 := by
  sorry

end derivative_through_point_l1545_154569


namespace add_3333_minutes_to_leap_day_noon_l1545_154534

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Represents the starting date and time -/
def startDateTime : DateTime :=
  { year := 2020, month := 2, day := 29, hour := 12, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3333

/-- The expected result date and time -/
def expectedDateTime : DateTime :=
  { year := 2020, month := 3, day := 2, hour := 19, minute := 33 }

/-- Function to add minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

theorem add_3333_minutes_to_leap_day_noon :
  addMinutes startDateTime minutesToAdd = expectedDateTime := by sorry

end add_3333_minutes_to_leap_day_noon_l1545_154534


namespace parabola_equation_l1545_154521

-- Define the hyperbola
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop :=
  p > 0 ∧ y^2 = 2 * p * x

-- Define the area of triangle AOB
def triangle_area (area : ℝ) : Prop := area = Real.sqrt 3

-- Theorem statement
theorem parabola_equation (a b p : ℝ) :
  (∃ x y : ℝ, hyperbola a b x y) →
  eccentricity 2 →
  (∃ x y : ℝ, parabola p x y) →
  triangle_area (Real.sqrt 3) →
  p = 2 :=
by sorry

end parabola_equation_l1545_154521


namespace fold_symmetry_l1545_154557

-- Define the fold line
def fold_line (x y : ℝ) : Prop := y = 2 * x

-- Define the symmetric point
def symmetric_point (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  fold_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2) ∧
  (x₂ - x₁) = (y₂ - y₁) / 2

-- Define the perpendicular bisector property
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  fold_line ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

-- Theorem statement
theorem fold_symmetry :
  perpendicular_bisector 10 0 (-6) 8 →
  symmetric_point (-4) 2 4 (-2) :=
sorry

end fold_symmetry_l1545_154557


namespace race_length_is_1000_l1545_154512

/-- Represents the length of the race in meters -/
def race_length : ℝ := 1000

/-- Represents the time difference between runners A and B in seconds -/
def time_difference : ℝ := 20

/-- Represents the distance difference between runners A and B in meters -/
def distance_difference : ℝ := 50

/-- Represents the time taken by runner A to complete the race in seconds -/
def time_A : ℝ := 380

/-- Theorem stating that the race length is 1000 meters given the conditions -/
theorem race_length_is_1000 :
  (distance_difference / time_difference) * (time_A + time_difference) = race_length := by
  sorry


end race_length_is_1000_l1545_154512


namespace jerry_average_study_time_difference_l1545_154526

def daily_differences : List Int := [15, -5, 25, 0, -15, 10]

def extra_study_time : Int := 20

def adjust_difference (diff : Int) : Int :=
  if diff > 0 then diff + extra_study_time else diff

theorem jerry_average_study_time_difference :
  let adjusted_differences := daily_differences.map adjust_difference
  let total_difference := adjusted_differences.sum
  let num_days := daily_differences.length
  total_difference / num_days = -15 := by sorry

end jerry_average_study_time_difference_l1545_154526


namespace minutes_worked_yesterday_l1545_154546

/-- The number of shirts made by the machine yesterday -/
def shirts_made_yesterday : ℕ := 9

/-- The number of shirts the machine can make per minute -/
def shirts_per_minute : ℕ := 3

/-- Theorem: The number of minutes the machine worked yesterday is 3 -/
theorem minutes_worked_yesterday : 
  shirts_made_yesterday / shirts_per_minute = 3 := by
  sorry

end minutes_worked_yesterday_l1545_154546
