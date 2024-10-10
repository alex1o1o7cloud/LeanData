import Mathlib

namespace jackie_apples_l1766_176676

/-- 
Given that Adam has 10 apples and 8 more apples than Jackie,
prove that Jackie has 2 apples.
-/
theorem jackie_apples (adam_apples : ℕ) (difference : ℕ) (jackie_apples : ℕ)
  (h1 : adam_apples = 10)
  (h2 : adam_apples = jackie_apples + difference)
  (h3 : difference = 8) :
  jackie_apples = 2 := by
  sorry

end jackie_apples_l1766_176676


namespace robot_trap_theorem_l1766_176673

theorem robot_trap_theorem (ε : ℝ) (hε : ε > 0) : 
  ∃ m l : ℕ+, |m.val * Real.sqrt 2 - l.val| < ε := by
sorry

end robot_trap_theorem_l1766_176673


namespace siblings_age_sum_l1766_176631

theorem siblings_age_sum (a b c : ℕ+) : 
  a < b ∧ b = c ∧ a * b * c = 144 → a + b + c = 16 := by
  sorry

end siblings_age_sum_l1766_176631


namespace june_population_calculation_l1766_176625

/-- Represents the fish population model in the reservoir --/
structure FishPopulation where
  june_population : ℕ
  tagged_fish : ℕ
  october_sample : ℕ
  tagged_in_sample : ℕ

/-- Calculates the number of fish in the reservoir on June 1 --/
def calculate_june_population (model : FishPopulation) : ℕ :=
  let remaining_tagged := model.tagged_fish * 7 / 10  -- 70% of tagged fish remain
  let october_old_fish := model.october_sample / 2    -- 50% of October fish are old
  (remaining_tagged * october_old_fish) / model.tagged_in_sample

/-- Theorem stating the correct number of fish in June based on the given model --/
theorem june_population_calculation (model : FishPopulation) :
  model.tagged_fish = 100 →
  model.october_sample = 90 →
  model.tagged_in_sample = 4 →
  calculate_june_population model = 1125 :=
by
  sorry

#eval calculate_june_population ⟨1125, 100, 90, 4⟩

end june_population_calculation_l1766_176625


namespace division_problem_l1766_176686

theorem division_problem (total : ℚ) (a b c : ℚ) : 
  total = 527 →
  a = (2/3) * b →
  b = (1/4) * c →
  a + b + c = total →
  c = 372 :=
by sorry

end division_problem_l1766_176686


namespace not_p_necessary_not_sufficient_for_not_p_or_q_l1766_176608

theorem not_p_necessary_not_sufficient_for_not_p_or_q (p q : Prop) :
  (∀ (h : ¬p ∨ q), ¬p) ∧ 
  ¬(∀ (h : ¬p), ¬(p ∨ q)) :=
sorry

end not_p_necessary_not_sufficient_for_not_p_or_q_l1766_176608


namespace stating_mans_speed_with_current_l1766_176687

/-- 
Given a man's speed against a current and the speed of the current,
this function calculates the man's speed with the current.
-/
def speed_with_current (speed_against_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- 
Theorem stating that given the specific conditions in the problem,
the man's speed with the current is 20 kmph.
-/
theorem mans_speed_with_current : 
  speed_with_current 14 3 = 20 := by
  sorry

#eval speed_with_current 14 3

end stating_mans_speed_with_current_l1766_176687


namespace optimal_rate_l1766_176696

/- Define the initial conditions -/
def totalRooms : ℕ := 100
def initialRate : ℕ := 400
def initialOccupancy : ℕ := 50
def rateReduction : ℕ := 20
def occupancyIncrease : ℕ := 5

/- Define the revenue function -/
def revenue (rate : ℕ) : ℕ :=
  let occupancy := initialOccupancy + ((initialRate - rate) / rateReduction) * occupancyIncrease
  rate * occupancy

/- Theorem statement -/
theorem optimal_rate :
  ∀ (rate : ℕ), rate ≤ initialRate → revenue 300 ≥ revenue rate :=
sorry

end optimal_rate_l1766_176696


namespace quadratic_has_minimum_l1766_176633

/-- Given a quadratic function f(x) = ax² + bx + b²/(2a) where a > 0,
    prove that the graph of f has a minimum. -/
theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  ∃ (x_min : ℝ), ∀ (x : ℝ), a * x^2 + b * x + b^2 / (2 * a) ≥ a * x_min^2 + b * x_min + b^2 / (2 * a) :=
sorry

end quadratic_has_minimum_l1766_176633


namespace deductive_reasoning_not_always_correct_l1766_176698

/-- Represents a deductive argument --/
structure DeductiveArgument where
  premises : List Prop
  conclusion : Prop

/-- Represents the form of a deductive argument --/
structure DeductiveForm where
  form : DeductiveArgument → Prop

/-- Defines when a deductive argument conforms to a deductive form --/
def conformsToForm (arg : DeductiveArgument) (form : DeductiveForm) : Prop :=
  form.form arg

/-- Defines when a deductive argument is valid --/
def isValid (arg : DeductiveArgument) : Prop :=
  ∀ (form : DeductiveForm), conformsToForm arg form → arg.conclusion

/-- Theorem: A deductive argument that conforms to a deductive form is not always valid --/
theorem deductive_reasoning_not_always_correct :
  ∃ (arg : DeductiveArgument) (form : DeductiveForm),
    conformsToForm arg form ∧ ¬isValid arg := by
  sorry

end deductive_reasoning_not_always_correct_l1766_176698


namespace special_function_inequality_l1766_176644

open Set

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  f_diff : Differentiable ℝ f
  f_domain : ∀ x, x < 0 → f x ≠ 0
  f_ineq : ∀ x, x < 0 → 2 * (f x) + x * (deriv f x) > x^2

/-- The main theorem -/
theorem special_function_inequality (sf : SpecialFunction) :
  {x : ℝ | (x + 2016)^2 * sf.f (x + 2016) - 4 * sf.f (-2) > 0} = Iio (-2018) := by
  sorry

end special_function_inequality_l1766_176644


namespace tuesday_boot_sales_l1766_176622

/-- Represents the sales data for a day -/
structure DailySales where
  shoes : ℕ
  boots : ℕ
  total : ℚ

/-- Represents the pricing and sales data for the shoe store -/
structure ShoeStore where
  shoe_price : ℚ
  boot_price : ℚ
  monday : DailySales
  tuesday : DailySales

/-- The main theorem to prove -/
theorem tuesday_boot_sales (store : ShoeStore) : store.tuesday.boots = 24 :=
  by
  have price_difference : store.boot_price = store.shoe_price + 15 := by sorry
  have monday_equation : store.shoe_price * store.monday.shoes + store.boot_price * store.monday.boots = store.monday.total := by sorry
  have tuesday_equation : store.shoe_price * store.tuesday.shoes + store.boot_price * store.tuesday.boots = store.tuesday.total := by sorry
  have monday_sales : store.monday.shoes = 22 ∧ store.monday.boots = 16 ∧ store.monday.total = 460 := by sorry
  have tuesday_partial_sales : store.tuesday.shoes = 8 ∧ store.tuesday.total = 560 := by sorry
  sorry

end tuesday_boot_sales_l1766_176622


namespace fraction_product_l1766_176602

theorem fraction_product : (2 : ℚ) / 3 * 5 / 7 * 11 / 13 * 17 / 19 = 1870 / 5187 := by
  sorry

end fraction_product_l1766_176602


namespace population_growth_l1766_176648

/-- Given an initial population and two consecutive percentage increases,
    calculate the final population after both increases. -/
def final_population (initial : ℕ) (increase1 : ℚ) (increase2 : ℚ) : ℚ :=
  initial * (1 + increase1) * (1 + increase2)

/-- Theorem stating that the population after two years of growth is 1320. -/
theorem population_growth : final_population 1000 (1/10) (1/5) = 1320 := by
  sorry

end population_growth_l1766_176648


namespace existence_of_twenty_problem_sequence_l1766_176688

theorem existence_of_twenty_problem_sequence (a : ℕ → ℕ) 
  (h1 : ∀ n, a (n + 1) ≥ a n + 1)
  (h2 : ∀ n, a (n + 7) - a n ≤ 12) :
  ∃ i j, i < j ∧ j ≤ 77 ∧ a j - a i = 20 := by
  sorry

end existence_of_twenty_problem_sequence_l1766_176688


namespace product_expansion_evaluation_l1766_176651

theorem product_expansion_evaluation :
  ∀ (a b c d : ℝ),
  (∀ x : ℝ, (4 * x^2 - 3 * x + 6) * (9 - 3 * x) = a * x^3 + b * x^2 + c * x + d) →
  8 * a + 4 * b + 2 * c + d = 48 := by
sorry

end product_expansion_evaluation_l1766_176651


namespace natalies_height_l1766_176646

/-- Prove that Natalie's height is 176 cm given the conditions -/
theorem natalies_height (h_natalie : ℝ) (h_harpreet : ℝ) (h_jiayin : ℝ) 
  (h_same_height : h_natalie = h_harpreet)
  (h_jiayin_height : h_jiayin = 161)
  (h_average : (h_natalie + h_harpreet + h_jiayin) / 3 = 171) :
  h_natalie = 176 := by
  sorry

end natalies_height_l1766_176646


namespace average_age_of_three_l1766_176660

/-- Given the ages of Omi, Kimiko, and Arlette, prove their average age is 35 --/
theorem average_age_of_three (kimiko_age : ℕ) (omi_age : ℕ) (arlette_age : ℕ) 
  (h1 : kimiko_age = 28) 
  (h2 : omi_age = 2 * kimiko_age) 
  (h3 : arlette_age = 3 * kimiko_age / 4) : 
  (kimiko_age + omi_age + arlette_age) / 3 = 35 := by
  sorry

#check average_age_of_three

end average_age_of_three_l1766_176660


namespace recurring_decimal_to_fraction_l1766_176671

theorem recurring_decimal_to_fraction :
  ∃ (x : ℚ), x = 4 + 36 / 99 ∧ x = 144 / 33 := by sorry

end recurring_decimal_to_fraction_l1766_176671


namespace inequality_proof_l1766_176642

theorem inequality_proof (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  2 * m + 1 / (m^2 - 2*m*n + n^2) ≥ 2 * n + 3 := by
  sorry

end inequality_proof_l1766_176642


namespace cannot_be_square_difference_l1766_176683

/-- The square difference formula -/
def square_difference (a b : ℝ → ℝ) : ℝ → ℝ := λ x => (a x)^2 - (b x)^2

/-- The expression that we want to prove cannot be computed using the square difference formula -/
def expression (x : ℝ) : ℝ := (x + 1) * (1 + x)

theorem cannot_be_square_difference :
  ¬∃ (a b : ℝ → ℝ), ∀ x, expression x = square_difference a b x :=
sorry

end cannot_be_square_difference_l1766_176683


namespace alice_number_theorem_l1766_176616

def smallest_prime_divisor (n : ℕ) : ℕ := sorry

def subtract_smallest_prime_divisor (n : ℕ) : ℕ := n - smallest_prime_divisor n

def iterate_subtraction (n : ℕ) (iterations : ℕ) : ℕ :=
  match iterations with
  | 0 => n
  | k + 1 => iterate_subtraction (subtract_smallest_prime_divisor n) k

theorem alice_number_theorem (n : ℕ) :
  n > 0 ∧ Nat.Prime (iterate_subtraction n 2022) →
  n = 4046 ∨ n = 4047 :=
sorry

end alice_number_theorem_l1766_176616


namespace angle_of_inclination_l1766_176695

theorem angle_of_inclination (x y : ℝ) :
  let line_equation := (Real.sqrt 3) * x + y - 3 = 0
  let angle_of_inclination := 2 * Real.pi / 3
  line_equation → angle_of_inclination = Real.arctan (-(Real.sqrt 3)) + Real.pi :=
by
  sorry

end angle_of_inclination_l1766_176695


namespace power_ratio_equals_nine_l1766_176685

/-- Given real numbers a and b satisfying the specified conditions, 
    prove that 3^a / b^3 = 9 -/
theorem power_ratio_equals_nine 
  (a b : ℝ) 
  (h1 : 3^(a-2) + a = 1/2) 
  (h2 : (1/3)*b^3 + Real.log b / Real.log 3 = -1/2) : 
  3^a / b^3 = 9 := by
  sorry

end power_ratio_equals_nine_l1766_176685


namespace real_solutions_exist_l1766_176617

theorem real_solutions_exist : ∃ x : ℝ, x^4 - 6 = 0 := by
  sorry

end real_solutions_exist_l1766_176617


namespace bakery_combinations_l1766_176662

/-- The number of ways to distribute n identical items among k groups -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of roll types -/
def num_roll_types : ℕ := 4

/-- The number of remaining rolls to distribute -/
def remaining_rolls : ℕ := 2

theorem bakery_combinations :
  distribute remaining_rolls num_roll_types = 10 := by
  sorry

end bakery_combinations_l1766_176662


namespace complex_number_proof_l1766_176643

theorem complex_number_proof : 
  ∀ (z : ℂ), (Complex.im ((1 + 2*Complex.I) * z) = 0) → (Complex.abs z = Real.sqrt 5) → 
  (z = 1 - 2*Complex.I ∨ z = -1 + 2*Complex.I) :=
by
  sorry

end complex_number_proof_l1766_176643


namespace train_length_l1766_176693

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) :
  speed = 72 → time = 12 → speed * time * (1000 / 3600) = 240 :=
by sorry

end train_length_l1766_176693


namespace remainder_proof_l1766_176675

theorem remainder_proof (y : Nat) (h1 : y > 0) (h2 : (7 * y) % 29 = 1) :
  (8 + y) % 29 = 4 := by
  sorry

end remainder_proof_l1766_176675


namespace class_size_l1766_176691

theorem class_size (boys girls : ℕ) : 
  (boys : ℚ) / girls = 4 / 3 →
  (boys - 8)^2 = girls - 14 →
  boys + girls = 42 := by
  sorry

end class_size_l1766_176691


namespace officer_selection_ways_l1766_176638

theorem officer_selection_ways (n : ℕ) (h : n = 8) : 
  (n.factorial / (n - 3).factorial) = 336 := by
  sorry

end officer_selection_ways_l1766_176638


namespace problem_solution_l1766_176678

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) : 
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 841/100 := by
  sorry

end problem_solution_l1766_176678


namespace equation_solution_range_l1766_176639

-- Define the equation
def equation (a m x : ℝ) : Prop :=
  a^(2*x) + (1 + 1/m)*a^x + 1 = 0

-- Define the conditions
def conditions (a m : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  -1/3 ≤ m ∧ m < 0

-- Theorem statement
theorem equation_solution_range (a m : ℝ) :
  conditions a m →
  (∃ x : ℝ, equation a m x ∧ a^x > 0) ↔
  m_range m :=
sorry

end equation_solution_range_l1766_176639


namespace flight_cost_A_to_C_via_B_l1766_176656

/-- Represents the cost of a flight with a given distance and number of stops -/
def flight_cost (distance : ℝ) (stops : ℕ) : ℝ :=
  120 + 0.15 * distance + 50 * stops

/-- The cities A, B, and C form a right-angled triangle -/
axiom right_triangle : ∃ (AB BC AC : ℝ), AB^2 + BC^2 = AC^2

/-- The distance between A and C is 2000 km -/
axiom AC_distance : ∃ AC : ℝ, AC = 2000

/-- The distance between A and B is 4000 km -/
axiom AB_distance : ∃ AB : ℝ, AB = 4000

/-- Theorem: The cost to fly from A to C with one stop at B is $1289.62 -/
theorem flight_cost_A_to_C_via_B : 
  ∃ (AB BC AC : ℝ), 
    AB^2 + BC^2 = AC^2 ∧ 
    AC = 2000 ∧ 
    AB = 4000 ∧ 
    flight_cost (AB + BC) 1 = 1289.62 := by
  sorry

end flight_cost_A_to_C_via_B_l1766_176656


namespace coefficient_f_nonzero_l1766_176635

-- Define the polynomial Q(x)
def Q (a b c d f : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + f

-- Define the theorem
theorem coefficient_f_nonzero 
  (a b c d f : ℝ) 
  (h1 : ∃ p q r s : ℝ, p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧ 
                       Q a b c d f p = 0 ∧ Q a b c d f q = 0 ∧ Q a b c d f r = 0 ∧ Q a b c d f s = 0)
  (h2 : Q a b c d f 1 = 0) : 
  f ≠ 0 := by
  sorry

end coefficient_f_nonzero_l1766_176635


namespace circle_condition_l1766_176629

theorem circle_condition (m : ℝ) :
  (∃ (h k r : ℝ), ∀ (x y : ℝ), x^2 + y^2 + 4*m*x - 2*y + 5*m = 0 ↔ (x - h)^2 + (y - k)^2 = r^2 ∧ r > 0) ↔
  (m < 1/4 ∨ m > 1) :=
sorry

end circle_condition_l1766_176629


namespace line_slope_l1766_176605

/-- The slope of the line represented by the equation x/4 - y/3 = 1 is 3/4 -/
theorem line_slope (x y : ℝ) : (x / 4 - y / 3 = 1) → (y = (3 / 4) * x - 3) := by
  sorry

end line_slope_l1766_176605


namespace frog_riverbank_probability_l1766_176603

/-- The probability of reaching the riverbank from stone N -/
noncomputable def P (N : ℕ) : ℝ :=
  sorry

/-- The number of stones -/
def num_stones : ℕ := 7

theorem frog_riverbank_probability :
  -- The frog starts on stone 2
  -- There are 7 stones labeled from 0 to 6
  -- For stone N (0 < N < 6), the frog jumps to N-1 with probability N/6 and to N+1 with probability 1 - N/6
  -- If the frog reaches stone 0, it falls into the water (probability 0)
  -- If the frog reaches stone 6, it safely reaches the riverbank (probability 1)
  (∀ N, 0 < N → N < 6 → P N = (N / 6 : ℝ) * P (N - 1) + (1 - N / 6 : ℝ) * P (N + 1)) →
  P 0 = 0 →
  P 6 = 1 →
  P 2 = 4/9 :=
sorry

end frog_riverbank_probability_l1766_176603


namespace sufficient_not_necessary_l1766_176697

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def areParallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The first line l₁: ax + y + 1 = 0 -/
def l1 (a : ℝ) : Line2D :=
  ⟨a, 1, 1⟩

/-- The second line l₂: 2x + (a + 1)y + 3 = 0 -/
def l2 (a : ℝ) : Line2D :=
  ⟨2, a + 1, 3⟩

/-- a = 1 is sufficient but not necessary for the lines to be parallel -/
theorem sufficient_not_necessary :
  (∀ a : ℝ, a = 1 → areParallel (l1 a) (l2 a)) ∧
  ¬(∀ a : ℝ, areParallel (l1 a) (l2 a) → a = 1) :=
sorry

end sufficient_not_necessary_l1766_176697


namespace quadratic_functions_problem_l1766_176611

/-- A quadratic function -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- The x-coordinate of the vertex of a quadratic function -/
def vertex (f : QuadraticFunction) : ℝ := sorry

/-- The x-intercepts of a quadratic function -/
def x_intercepts (f : QuadraticFunction) : Set ℝ := sorry

theorem quadratic_functions_problem 
  (f g : QuadraticFunction)
  (h1 : ∀ x, g.f x = -f.f (100 - x))
  (h2 : vertex f ∈ x_intercepts g)
  (x₁ x₂ x₃ x₄ : ℝ)
  (h3 : {x₁, x₂, x₃, x₄} ⊆ x_intercepts f ∪ x_intercepts g)
  (h4 : x₁ < x₂ ∧ x₂ < x₃ ∧ x₃ < x₄)
  (h5 : x₃ - x₂ = 150)
  : x₄ - x₁ = 450 + 300 * Real.sqrt 2 := by
  sorry

end quadratic_functions_problem_l1766_176611


namespace nested_fraction_evaluation_l1766_176679

theorem nested_fraction_evaluation :
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end nested_fraction_evaluation_l1766_176679


namespace inequality_always_true_l1766_176606

theorem inequality_always_true (x : ℝ) : (4 * x) / (x^2 + 4) ≤ 1 := by
  sorry

end inequality_always_true_l1766_176606


namespace algorithm_finite_results_l1766_176649

-- Define the properties of an algorithm
structure Algorithm where
  steps : ℕ
  inputs : ℕ
  deterministic : Bool
  unique_meaning : Bool
  definite : Bool
  finite : Bool
  orderly : Bool
  non_unique : Bool
  universal : Bool

-- Define the theorem
theorem algorithm_finite_results (a : Algorithm) : 
  a.definite → ¬(∃ (results : ℕ → Prop), (∀ n : ℕ, results n) ∧ (∀ m n : ℕ, m ≠ n → results m ≠ results n)) :=
by sorry

end algorithm_finite_results_l1766_176649


namespace cathy_cookies_l1766_176654

theorem cathy_cookies (total : ℝ) (amy_fraction : ℝ) (bob_cookies : ℝ) : 
  total = 18 → 
  amy_fraction = 1/3 → 
  bob_cookies = 2.5 → 
  total - (amy_fraction * total + bob_cookies) = 9.5 := by
sorry

end cathy_cookies_l1766_176654


namespace unique_triangle_set_l1766_176621

def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def can_form_triangle (sides : List ℝ) : Prop :=
  sides.length = 3 ∧ 
  match sides with
  | [a, b, c] => triangle_inequality a b c
  | _ => False

theorem unique_triangle_set : 
  ¬ can_form_triangle [2, 3, 6] ∧
  ¬ can_form_triangle [3, 4, 8] ∧
  can_form_triangle [5, 6, 10] ∧
  ¬ can_form_triangle [5, 6, 11] :=
sorry

end unique_triangle_set_l1766_176621


namespace compound_interest_rate_l1766_176613

theorem compound_interest_rate : ∃ (r : ℝ), 
  (1 + r)^2 = 7/6 ∧ 
  0.0800 < r ∧ 
  r < 0.0802 := by
  sorry

end compound_interest_rate_l1766_176613


namespace absolute_value_sum_range_l1766_176669

theorem absolute_value_sum_range : 
  ∃ (min_value : ℝ), 
    (∀ x : ℝ, |x - 1| + |x - 2| ≥ min_value) ∧ 
    (∃ x : ℝ, |x - 1| + |x - 2| = min_value) ∧
    (∀ a : ℝ, (∃ x : ℝ, |x - 1| + |x - 2| ≤ a) ↔ a ≥ min_value) ∧
    min_value = 1 :=
by sorry

end absolute_value_sum_range_l1766_176669


namespace probability_of_three_correct_l1766_176641

/-- Two fair dice are thrown once each -/
def dice : ℕ := 2

/-- Each die has 6 faces -/
def faces_per_die : ℕ := 6

/-- The numbers facing up are different -/
def different_numbers : Prop := true

/-- The probability that one of the dice shows a 3 -/
def probability_of_three : ℚ := 1 / 3

/-- Theorem stating that the probability of getting a 3 on one die when two fair dice are thrown with different numbers is 1/3 -/
theorem probability_of_three_correct (h : different_numbers) : probability_of_three = 1 / 3 := by
  sorry

end probability_of_three_correct_l1766_176641


namespace largest_n_for_square_sum_equality_l1766_176657

theorem largest_n_for_square_sum_equality : ∃ (n : ℕ), n > 0 ∧ 
  (∃ (x : ℤ) (y : ℕ → ℤ), ∀ (i j : ℕ), i ≤ n → j ≤ n → (x + i)^2 + (y i)^2 = (x + j)^2 + (y j)^2) ∧
  (∀ (m : ℕ), m > n → ¬∃ (x : ℤ) (y : ℕ → ℤ), ∀ (i j : ℕ), i ≤ m → j ≤ m → (x + i)^2 + (y i)^2 = (x + j)^2 + (y j)^2) ∧
  n = 3 :=
by sorry

end largest_n_for_square_sum_equality_l1766_176657


namespace hospital_bill_proof_l1766_176663

theorem hospital_bill_proof (total_bill : ℝ) (medication_percentage : ℝ) 
  (food_cost : ℝ) (ambulance_cost : ℝ) :
  total_bill = 5000 →
  medication_percentage = 50 →
  food_cost = 175 →
  ambulance_cost = 1700 →
  let remaining_bill := total_bill - (medication_percentage / 100 * total_bill)
  let overnight_cost := remaining_bill - food_cost - ambulance_cost
  overnight_cost / remaining_bill * 100 = 25 := by
  sorry

end hospital_bill_proof_l1766_176663


namespace bus_stop_walking_time_l1766_176680

/-- The time taken to walk to the bus stop at the usual speed, given that walking at 4/5 of the usual speed results in arriving 9 minutes later than normal, is 36 minutes. -/
theorem bus_stop_walking_time : ∀ T : ℝ, T > 0 → (5 / 4 = (T + 9) / T) → T = 36 := by
  sorry

end bus_stop_walking_time_l1766_176680


namespace tiles_needed_for_room_main_tiling_theorem_l1766_176614

/-- Represents the tiling pattern where n is the number of days and f(n) is the number of tiles placed on day n. -/
def tilingPattern (n : ℕ) : ℕ := n

/-- Represents the total number of tiles placed after n days. -/
def totalTiles (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The surface area of the room in square units. -/
def roomArea : ℕ := 18144

/-- The theorem stating that 2016 tiles are needed to cover the room. -/
theorem tiles_needed_for_room :
  ∃ (sideLength : ℕ), sideLength > 0 ∧ totalTiles 63 = 2016 ∧ 2016 * sideLength^2 = roomArea :=
by
  sorry

/-- The main theorem proving that 2016 tiles are needed and follow the tiling pattern. -/
theorem main_tiling_theorem :
  ∃ (n : ℕ), totalTiles n = 2016 ∧
    (∀ (k : ℕ), k ≤ n → tilingPattern k = k) ∧
    (∃ (sideLength : ℕ), sideLength > 0 ∧ 2016 * sideLength^2 = roomArea) :=
by
  sorry

end tiles_needed_for_room_main_tiling_theorem_l1766_176614


namespace total_weight_of_four_l1766_176627

theorem total_weight_of_four (jim steve stan tim : ℕ) : 
  jim = 110 →
  steve = jim - 8 →
  stan = steve + 5 →
  tim = stan + 12 →
  jim + steve + stan + tim = 438 := by
sorry

end total_weight_of_four_l1766_176627


namespace parallelogram_area_l1766_176623

def v : Fin 2 → ℝ := ![5, -3]
def w : Fin 2 → ℝ := ![11, -2]

theorem parallelogram_area : 
  abs (v 0 * w 1 - v 1 * w 0) = 23 := by sorry

end parallelogram_area_l1766_176623


namespace simplify_logarithmic_expression_l1766_176684

theorem simplify_logarithmic_expression :
  let x := 1 / (Real.log 3 / Real.log 6 + 1) +
           1 / (Real.log 7 / Real.log 15 + 1) +
           1 / (Real.log 4 / Real.log 12 + 1)
  x = -Real.log 84 / Real.log 10 := by
  sorry

end simplify_logarithmic_expression_l1766_176684


namespace scheme2_more_cost_effective_l1766_176677

/-- Represents the cost of scheme 1 -/
def scheme1_cost (x : ℝ) : ℝ := 15 * x + 40

/-- Represents the cost of scheme 2 -/
def scheme2_cost (x : ℝ) : ℝ := 15.2 * x + 32

/-- The price of a pen -/
def pen_price : ℝ := 15

/-- The price of a notebook -/
def notebook_price : ℝ := 4

/-- Theorem stating that scheme 2 is always cheaper or equal to scheme 1 -/
theorem scheme2_more_cost_effective (x : ℝ) (h : x ≥ 0) :
  scheme2_cost x ≤ scheme1_cost x :=
sorry

#check scheme2_more_cost_effective

end scheme2_more_cost_effective_l1766_176677


namespace steak_price_per_pound_l1766_176601

theorem steak_price_per_pound (steak_price : ℚ) : 
  4.5 * steak_price + 1.5 * 8 = 42 → steak_price = 20 / 3 := by
  sorry

end steak_price_per_pound_l1766_176601


namespace bens_car_cost_ratio_l1766_176604

theorem bens_car_cost_ratio :
  let old_car_cost : ℚ := 1800
  let new_car_cost : ℚ := 2000 + 1800
  (new_car_cost / old_car_cost) = 19 / 9 := by
  sorry

end bens_car_cost_ratio_l1766_176604


namespace smallest_sticker_count_l1766_176658

theorem smallest_sticker_count (N : ℕ) : 
  N > 1 → 
  (∃ x y z : ℕ, N = 3 * x + 1 ∧ N = 5 * y + 1 ∧ N = 11 * z + 1) → 
  N ≥ 166 :=
by sorry

end smallest_sticker_count_l1766_176658


namespace cuboid_breadth_proof_l1766_176637

/-- The surface area of a cuboid given its length, width, and height. -/
def cuboidSurfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The breadth of a cuboid with surface area 700 m², length 12 m, and height 7 m is 14 m. -/
theorem cuboid_breadth_proof :
  ∃ w : ℝ, cuboidSurfaceArea 12 w 7 = 700 ∧ w = 14 := by
  sorry

end cuboid_breadth_proof_l1766_176637


namespace equation_solutions_l1766_176672

theorem equation_solutions :
  (∃ x : ℝ, x^2 + 2*x = 0 ↔ x = 0 ∨ x = -2) ∧
  (∃ x : ℝ, 4*x^2 - 4*x + 1 = 0 ↔ x = 1/2) := by
  sorry

end equation_solutions_l1766_176672


namespace stating_equilateral_triangle_condition_l1766_176659

/-- 
A function that checks if a natural number n satisfies the condition
that sticks of lengths 1, 2, ..., n can form an equilateral triangle.
-/
def can_form_equilateral_triangle (n : ℕ) : Prop :=
  n ≥ 5 ∧ (n % 6 = 0 ∨ n % 6 = 2 ∨ n % 6 = 3 ∨ n % 6 = 5)

/-- 
Theorem stating that sticks of lengths 1, 2, ..., n can form an equilateral triangle
if and only if n satisfies the condition defined in can_form_equilateral_triangle.
-/
theorem equilateral_triangle_condition (n : ℕ) :
  (∃ (a b c : ℕ), a + b + c = n * (n + 1) / 2 ∧ a = b ∧ b = c) ↔
  can_form_equilateral_triangle n :=
sorry

end stating_equilateral_triangle_condition_l1766_176659


namespace highest_lowest_difference_l1766_176664

/-- Represents the scores of four participants in an exam -/
structure ExamScores where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The conditions of the exam scores -/
def validExamScores (scores : ExamScores) : Prop :=
  scores.A + scores.B = scores.C + scores.D + 17 ∧
  scores.A = scores.B - 4 ∧
  scores.C = scores.D + 5

/-- The theorem stating the difference between the highest and lowest scores -/
theorem highest_lowest_difference (scores : ExamScores) 
  (h : validExamScores scores) : 
  max scores.A (max scores.B (max scores.C scores.D)) - 
  min scores.A (min scores.B (min scores.C scores.D)) = 13 := by
  sorry

#check highest_lowest_difference

end highest_lowest_difference_l1766_176664


namespace consecutive_integers_sum_36_l1766_176653

theorem consecutive_integers_sum_36 : 
  ∃! (a : ℕ), a > 0 ∧ a + (a + 1) + (a + 2) = 36 :=
by sorry

end consecutive_integers_sum_36_l1766_176653


namespace banana_count_l1766_176618

def fruit_bowl (apples pears bananas : ℕ) : Prop :=
  (pears = apples + 2) ∧
  (bananas = pears + 3) ∧
  (apples + pears + bananas = 19)

theorem banana_count :
  ∀ (a p b : ℕ), fruit_bowl a p b → b = 9 := by
  sorry

end banana_count_l1766_176618


namespace smallest_b_for_composite_polynomial_l1766_176699

theorem smallest_b_for_composite_polynomial : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), ¬ Nat.Prime (x^4 + x^3 + b^2 + 5).natAbs) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ∃ (x : ℤ), Nat.Prime (x^4 + x^3 + b'^2 + 5).natAbs) ∧
  b = 7 := by
sorry

end smallest_b_for_composite_polynomial_l1766_176699


namespace angle_measure_problem_l1766_176628

/-- Two angles are complementary if their measures sum to 90 degrees -/
def complementary (a b : ℝ) : Prop := a + b = 90

/-- Given two complementary angles A and B, where A is 5 times B, prove A is 75 degrees -/
theorem angle_measure_problem (A B : ℝ) 
  (h1 : complementary A B) 
  (h2 : A = 5 * B) : 
  A = 75 := by
  sorry

end angle_measure_problem_l1766_176628


namespace pens_distribution_l1766_176665

/-- The number of pens in each pack -/
def pens_per_pack : ℕ := 3

/-- The number of packs Kendra has -/
def kendra_packs : ℕ := 4

/-- The number of packs Tony has -/
def tony_packs : ℕ := 2

/-- The number of pens Kendra and Tony each keep for themselves -/
def pens_kept_each : ℕ := 2

/-- The total number of friends who will receive pens -/
def friends_receiving_pens : ℕ := 
  kendra_packs * pens_per_pack + tony_packs * pens_per_pack - 2 * pens_kept_each

theorem pens_distribution :
  friends_receiving_pens = 14 := by
  sorry

end pens_distribution_l1766_176665


namespace fraction_problem_l1766_176600

theorem fraction_problem (n d : ℚ) : 
  d = 2 * n - 1 → 
  (n + 1) / (d + 1) = 3 / 5 → 
  n / d = 5 / 9 := by
sorry

end fraction_problem_l1766_176600


namespace symmetric_line_equation_l1766_176650

/-- Given a line with equation 3x - y + 2 = 0, its symmetric line with respect to the y-axis has the equation 3x + y - 2 = 0 -/
theorem symmetric_line_equation (x y : ℝ) :
  (3 * x - y + 2 = 0) → 
  ∃ (x' y' : ℝ), (3 * x' + y' - 2 = 0 ∧ x' = -x ∧ y' = y) := by
  sorry

end symmetric_line_equation_l1766_176650


namespace complement_B_intersect_A_union_A_M_equiv_M_l1766_176692

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x > 1}

-- Define set M with parameter a
def M (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < a + 6}

-- Theorem for part (1)
theorem complement_B_intersect_A :
  (Set.univ \ B) ∩ A = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem for part (2)
theorem union_A_M_equiv_M (a : ℝ) :
  A ∪ M a = M a ↔ -4 < a ∧ a < -2 := by sorry

end complement_B_intersect_A_union_A_M_equiv_M_l1766_176692


namespace bake_sale_girls_l1766_176670

theorem bake_sale_girls (initial_total : ℕ) : 
  -- Initial conditions
  (3 * initial_total / 5 : ℚ) = initial_total * (60 : ℚ) / 100 →
  -- Changes in group composition
  let new_total := initial_total - 1 + 3
  let new_girls := (3 * initial_total / 5 : ℚ) - 3
  -- Final condition
  new_girls / new_total = (1 : ℚ) / 2 →
  -- Conclusion
  (3 * initial_total / 5 : ℚ) = 24 := by
sorry

end bake_sale_girls_l1766_176670


namespace probability_is_one_half_l1766_176690

def total_balls : ℕ := 12
def white_balls : ℕ := 7
def black_balls : ℕ := 5
def drawn_balls : ℕ := 6

def probability_at_least_four_white : ℚ :=
  (Nat.choose white_balls 4 * Nat.choose black_balls 2 +
   Nat.choose white_balls 5 * Nat.choose black_balls 1 +
   Nat.choose white_balls 6 * Nat.choose black_balls 0) /
  Nat.choose total_balls drawn_balls

theorem probability_is_one_half :
  probability_at_least_four_white = 1 / 2 := by
  sorry

end probability_is_one_half_l1766_176690


namespace min_value_of_function_l1766_176647

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  (x^2 + 4) / x ≥ 4 ∧ ∃ y > 0, (y^2 + 4) / y = 4 := by
  sorry

end min_value_of_function_l1766_176647


namespace product_of_fractions_l1766_176667

theorem product_of_fractions : (2 : ℚ) / 3 * 5 / 7 * 11 / 13 = 110 / 273 := by
  sorry

end product_of_fractions_l1766_176667


namespace tangent_forms_345_triangle_l1766_176645

/-- An isosceles triangle with leg 10 cm and base 12 cm -/
structure IsoscelesTriangle where
  leg : ℝ
  base : ℝ
  leg_positive : 0 < leg
  base_positive : 0 < base
  isosceles : leg = 10
  base_length : base = 12

/-- The inscribed circle of the triangle -/
def inscribed_circle (t : IsoscelesTriangle) : ℝ := sorry

/-- Tangent line to the inscribed circle parallel to the height of the triangle -/
def tangent_line (t : IsoscelesTriangle) (c : ℝ) : ℝ → ℝ := sorry

/-- Right triangle formed by the tangent line -/
structure RightTriangle where
  hypotenuse : ℝ
  leg1 : ℝ
  leg2 : ℝ
  hypotenuse_positive : 0 < hypotenuse
  leg1_positive : 0 < leg1
  leg2_positive : 0 < leg2
  right_angle : leg1^2 + leg2^2 = hypotenuse^2

/-- The theorem to be proved -/
theorem tangent_forms_345_triangle (t : IsoscelesTriangle) (c : ℝ) :
  ∃ (rt : RightTriangle), rt.leg1 = 3 ∧ rt.leg2 = 4 ∧ rt.hypotenuse = 5 :=
sorry

end tangent_forms_345_triangle_l1766_176645


namespace path_cost_calculation_l1766_176674

/-- Represents the dimensions and cost parameters of a field with a path around it. -/
structure FieldWithPath where
  field_length : ℝ
  field_width : ℝ
  path_width : ℝ
  path_area : ℝ
  cost_per_sqm : ℝ

/-- Calculates the total cost of constructing a path around a field. -/
def total_path_cost (f : FieldWithPath) : ℝ :=
  f.path_area * f.cost_per_sqm

/-- Theorem stating that the total cost of constructing the path is Rs. 3037.44. -/
theorem path_cost_calculation (f : FieldWithPath)
  (h1 : f.field_length = 75)
  (h2 : f.field_width = 55)
  (h3 : f.path_width = 2.8)
  (h4 : f.path_area = 1518.72)
  (h5 : f.cost_per_sqm = 2) :
  total_path_cost f = 3037.44 := by
  sorry

#check path_cost_calculation

end path_cost_calculation_l1766_176674


namespace baseball_card_value_decrease_l1766_176607

theorem baseball_card_value_decrease (initial_value : ℝ) (first_year_decrease : ℝ) (total_decrease : ℝ) 
  (h1 : first_year_decrease = 60)
  (h2 : total_decrease = 64)
  (h3 : initial_value > 0) :
  let value_after_first_year := initial_value * (1 - first_year_decrease / 100)
  let final_value := initial_value * (1 - total_decrease / 100)
  let second_year_decrease := (value_after_first_year - final_value) / value_after_first_year * 100
  second_year_decrease = 10 := by sorry

end baseball_card_value_decrease_l1766_176607


namespace handshaking_arrangements_mod_1000_l1766_176620

/-- Represents a handshaking arrangement for a group of people -/
structure HandshakingArrangement (n : ℕ) :=
  (shakes : Fin n → Finset (Fin n))
  (shake_count : ∀ i, (shakes i).card = 3)
  (symmetry : ∀ i j, j ∈ shakes i ↔ i ∈ shakes j)

/-- The number of valid handshaking arrangements for 12 people -/
def M : ℕ := sorry

/-- Theorem stating that the number of handshaking arrangements M for 12 people,
    where each person shakes hands with exactly 3 others, satisfies M ≡ 50 (mod 1000) -/
theorem handshaking_arrangements_mod_1000 :
  M ≡ 50 [MOD 1000] := by sorry

end handshaking_arrangements_mod_1000_l1766_176620


namespace repeating_decimal_ratio_l1766_176636

/-- The decimal representation 0.142857142857... as a real number -/
def a : ℚ := 142857 / 999999

/-- The decimal representation 0.285714285714... as a real number -/
def b : ℚ := 285714 / 999999

/-- Theorem stating that the ratio of the two repeating decimals is 1/2 -/
theorem repeating_decimal_ratio : a / b = 1 / 2 := by
  sorry

end repeating_decimal_ratio_l1766_176636


namespace probability_two_red_balls_l1766_176632

/-- The probability of selecting 2 red balls from a bag containing 3 red, 2 blue, and 4 green balls -/
theorem probability_two_red_balls (red blue green : ℕ) 
  (h_red : red = 3) 
  (h_blue : blue = 2) 
  (h_green : green = 4) : 
  (red.choose 2 : ℚ) / ((red + blue + green).choose 2) = 1 / 12 := by
  sorry

end probability_two_red_balls_l1766_176632


namespace flag_pole_height_l1766_176634

/-- Given a tree and a flag pole casting shadows at the same time, 
    calculate the height of the flag pole. -/
theorem flag_pole_height 
  (tree_height : ℝ) 
  (tree_shadow : ℝ) 
  (flag_shadow : ℝ) 
  (h_tree_height : tree_height = 12)
  (h_tree_shadow : tree_shadow = 8)
  (h_flag_shadow : flag_shadow = 100) :
  (tree_height / tree_shadow) * flag_shadow = 150 :=
by
  sorry

#check flag_pole_height

end flag_pole_height_l1766_176634


namespace more_girls_than_boys_l1766_176666

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) :
  total_students = 42 →
  boy_ratio = 3 →
  girl_ratio = 4 →
  ∃ (boys girls : ℕ),
    boys + girls = total_students ∧
    boy_ratio * girls = girl_ratio * boys ∧
    girls - boys = 6 := by
  sorry

end more_girls_than_boys_l1766_176666


namespace third_set_candies_l1766_176682

/-- Represents a set of candies with hard candies, chocolates, and gummy candies -/
structure CandySet where
  hard : ℕ
  chocolate : ℕ
  gummy : ℕ

/-- The total number of candies across all three sets -/
def totalCandies (set1 set2 set3 : CandySet) : ℕ :=
  set1.hard + set1.chocolate + set1.gummy +
  set2.hard + set2.chocolate + set2.gummy +
  set3.hard + set3.chocolate + set3.gummy

theorem third_set_candies
  (set1 set2 set3 : CandySet)
  (h1 : set1.hard + set2.hard + set3.hard = set1.chocolate + set2.chocolate + set3.chocolate)
  (h2 : set1.hard + set2.hard + set3.hard = set1.gummy + set2.gummy + set3.gummy)
  (h3 : set1.chocolate = set1.gummy)
  (h4 : set1.hard = set1.chocolate + 7)
  (h5 : set2.hard = set2.chocolate)
  (h6 : set2.gummy = set2.hard - 15)
  (h7 : set3.hard = 0) :
  set3.chocolate + set3.gummy = 29 := by
  sorry

#check third_set_candies

end third_set_candies_l1766_176682


namespace m_range_l1766_176624

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the theorem
theorem m_range (h1 : ∀ x, x ∈ Set.Icc (-2) 2 → f x ∈ Set.Icc (-2) 2)
                (h2 : StrictMono f)
                (h3 : ∀ m, f (1 - m) < f m) :
  ∀ m, m ∈ Set.Ioo (1/2) 2 :=
sorry

end m_range_l1766_176624


namespace stock_sale_total_amount_l1766_176689

/-- Calculates the total amount including brokerage for a stock sale -/
theorem stock_sale_total_amount 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 106.25) 
  (h2 : brokerage_rate = 0.25) : 
  ∃ (total_amount : ℝ), total_amount = 106.52 ∧ 
  total_amount = cash_realized + (brokerage_rate / 100) * cash_realized :=
by sorry

end stock_sale_total_amount_l1766_176689


namespace expression_simplification_l1766_176655

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 3 + 3) :
  (1 - m / (m + 3)) / ((m^2 - 9) / (m^2 + 6*m + 9)) = Real.sqrt 3 := by
  sorry

end expression_simplification_l1766_176655


namespace mark_sprint_distance_l1766_176652

/-- The distance traveled by Mark given his sprint duration and speed -/
theorem mark_sprint_distance (duration : ℝ) (speed : ℝ) (h1 : duration = 24.0) (h2 : speed = 6.0) :
  duration * speed = 144.0 := by
  sorry

end mark_sprint_distance_l1766_176652


namespace tangent_slope_at_zero_l1766_176615

def f (x : ℝ) := x^2 - 6*x

theorem tangent_slope_at_zero : 
  (deriv f) 0 = -6 := by sorry

end tangent_slope_at_zero_l1766_176615


namespace five_lines_max_sections_l1766_176681

/-- The maximum number of sections created by drawing n line segments through a rectangle -/
def max_sections (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of sections created by drawing 5 line segments through a rectangle is 16 -/
theorem five_lines_max_sections : max_sections 5 = 16 := by
  sorry

end five_lines_max_sections_l1766_176681


namespace shaded_area_fraction_l1766_176668

/-- Given a rectangle PQRS with width w and height h, and three congruent triangles
    STU, UVW, and WXR inscribed in the rectangle such that SU = UW = WR = w/3,
    prove that the total area of the three triangles is 1/2 of the rectangle's area. -/
theorem shaded_area_fraction (w h : ℝ) (hw : w > 0) (hh : h > 0) :
  let rectangle_area := w * h
  let triangle_base := w / 3
  let triangle_height := h
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let total_shaded_area := 3 * triangle_area
  total_shaded_area = (1 / 2) * rectangle_area := by
  sorry

end shaded_area_fraction_l1766_176668


namespace purely_imaginary_complex_number_l1766_176630

theorem purely_imaginary_complex_number (x : ℝ) : 
  (Complex.I * (x + 1) : ℂ).re = 0 ∧ (Complex.I * (x + 1) : ℂ).im ≠ 0 →
  x = 1 := by
sorry

end purely_imaginary_complex_number_l1766_176630


namespace sum_reciprocals_l1766_176609

theorem sum_reciprocals (a b c d e : ℝ) (ω : ℂ) :
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 → e ≠ -1 →
  ω^4 = 1 →
  ω ≠ 1 →
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) + 1 / (e + ω) : ℂ) = 3 / ω^2 →
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) + 1 / (e + 1) : ℝ) = 3 :=
by sorry

end sum_reciprocals_l1766_176609


namespace max_surface_area_inscribed_sphere_l1766_176626

/-- The maximum surface area of an inscribed sphere in a right triangular prism --/
theorem max_surface_area_inscribed_sphere (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 25) :
  ∃ (r : ℝ), r > 0 ∧ 
    r = (5/2) * (Real.sqrt 2 - 1) ∧
    4 * π * r^2 = 25 * (3 - 3 * Real.sqrt 2) * π ∧
    ∀ (r' : ℝ), r' > 0 → r' * (a + b + 5) ≤ a * b → 4 * π * r'^2 ≤ 25 * (3 - 3 * Real.sqrt 2) * π :=
by sorry

end max_surface_area_inscribed_sphere_l1766_176626


namespace journey_length_l1766_176640

theorem journey_length :
  ∀ (total : ℚ),
  (1 / 4 : ℚ) * total +  -- First part
  30 +                   -- Second part (city)
  (1 / 7 : ℚ) * total    -- Third part
  = total                -- Sum of all parts equals total
  →
  total = 840 / 17 :=
by
  sorry

end journey_length_l1766_176640


namespace rug_area_calculation_l1766_176610

/-- Calculates the area of a rug on a rectangular floor with uncovered strips along the edges -/
def rugArea (floorLength floorWidth stripWidth : ℝ) : ℝ :=
  (floorLength - 2 * stripWidth) * (floorWidth - 2 * stripWidth)

/-- Theorem stating that the area of the rug is 204 square meters given the specific dimensions -/
theorem rug_area_calculation :
  rugArea 25 20 4 = 204 := by
  sorry

end rug_area_calculation_l1766_176610


namespace inequality_proof_l1766_176694

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b)^2 / 2 + (a + b) / 4 ≥ a * Real.sqrt b + b * Real.sqrt a := by
  sorry

end inequality_proof_l1766_176694


namespace line_slope_l1766_176661

-- Define the parametric equations of the line
def x (t : ℝ) : ℝ := 3 + 4 * t
def y (t : ℝ) : ℝ := 4 - 5 * t

-- State the theorem
theorem line_slope :
  ∃ m : ℝ, ∀ t₁ t₂ : ℝ, t₁ ≠ t₂ → (y t₂ - y t₁) / (x t₂ - x t₁) = m ∧ m = -5/4 :=
sorry

end line_slope_l1766_176661


namespace negation_of_existence_negation_of_proposition_l1766_176619

variable (a : ℝ)

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x, p x) ↔ (∀ x, ¬ p x) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) :=
by sorry

end negation_of_existence_negation_of_proposition_l1766_176619


namespace probability_isosceles_triangle_l1766_176612

def roll_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

def is_isosceles (a b : ℕ) : Bool :=
  a + b > 5

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (roll_die.product roll_die).filter (fun (a, b) => is_isosceles a b)

theorem probability_isosceles_triangle :
  (favorable_outcomes.card : ℚ) / (roll_die.card * roll_die.card) = 7 / 18 := by
  sorry

end probability_isosceles_triangle_l1766_176612
