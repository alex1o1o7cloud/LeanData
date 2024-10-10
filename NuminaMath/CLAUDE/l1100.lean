import Mathlib

namespace cos_equality_l1100_110012

theorem cos_equality (n : ℤ) : 0 ≤ n ∧ n ≤ 180 → n = 138 → Real.cos (n * π / 180) = Real.cos (942 * π / 180) := by
  sorry

end cos_equality_l1100_110012


namespace cara_catches_47_l1100_110049

/-- The number of animals Martha's cat catches -/
def martha_animals : ℕ := 10

/-- The number of animals Cara's cat catches -/
def cara_animals : ℕ := 5 * martha_animals - 3

/-- Theorem stating that Cara's cat catches 47 animals -/
theorem cara_catches_47 : cara_animals = 47 := by
  sorry

end cara_catches_47_l1100_110049


namespace distance_between_points_l1100_110045

def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (8, 9)

theorem distance_between_points :
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = 6 * Real.sqrt 5 := by
  sorry

end distance_between_points_l1100_110045


namespace roger_lawn_mowing_earnings_l1100_110047

/-- Roger's lawn mowing earnings problem -/
theorem roger_lawn_mowing_earnings : 
  ∀ (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ),
    rate = 9 →
    total_lawns = 14 →
    forgotten_lawns = 8 →
    rate * (total_lawns - forgotten_lawns) = 54 := by
  sorry

end roger_lawn_mowing_earnings_l1100_110047


namespace marys_cake_flour_l1100_110055

/-- Given a cake recipe that requires a certain amount of flour and an amount already added,
    calculate the remaining amount to be added. -/
def remaining_flour (required : ℕ) (added : ℕ) : ℕ :=
  required - added

/-- Prove that for Mary's cake, which requires 8 cups of flour and has 2 cups already added,
    the remaining amount to be added is 6 cups. -/
theorem marys_cake_flour : remaining_flour 8 2 = 6 := by
  sorry

end marys_cake_flour_l1100_110055


namespace arrangements_with_conditions_l1100_110017

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def arrangements_of_n (n : ℕ) : ℕ := factorial n

def arrangements_with_left_end (n : ℕ) : ℕ := factorial (n - 1)

def arrangements_adjacent (n : ℕ) : ℕ := 2 * factorial (n - 1)

def arrangements_left_end_and_adjacent (n : ℕ) : ℕ := factorial (n - 2)

theorem arrangements_with_conditions (n : ℕ) (h : n = 5) : 
  arrangements_of_n n - arrangements_with_left_end n - arrangements_adjacent n + arrangements_left_end_and_adjacent n = 54 :=
sorry

end arrangements_with_conditions_l1100_110017


namespace inequality_proof_l1100_110079

theorem inequality_proof (x y : ℝ) (n k : ℕ) 
  (h1 : x > y) (h2 : y > 0) (h3 : n > k) : 
  (x^k - y^k)^n < (x^n - y^n)^k := by
  sorry

end inequality_proof_l1100_110079


namespace expression_evaluation_l1100_110033

theorem expression_evaluation (x : ℝ) (h : x = 6) :
  (x^9 - 27*x^6 + 216*x^3 - 512) / (x^3 - 8) = 43264 := by
  sorry

end expression_evaluation_l1100_110033


namespace factorial_6_l1100_110003

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_6 : factorial 6 = 720 := by sorry

end factorial_6_l1100_110003


namespace constant_k_value_l1100_110085

theorem constant_k_value : ∃ k : ℝ, ∀ x : ℝ, 
  -x^2 - (k + 7)*x - 8 = -(x - 2)*(x - 4) ↔ k = -13 := by
  sorry

end constant_k_value_l1100_110085


namespace z_in_second_quadrant_l1100_110027

/-- The complex number z = 2i / (1-i) corresponds to a point in the second quadrant of the complex plane. -/
theorem z_in_second_quadrant : ∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ (2 * Complex.I) / (1 - Complex.I) = Complex.mk x y := by
  sorry

end z_in_second_quadrant_l1100_110027


namespace quadratic_root_property_l1100_110025

theorem quadratic_root_property (a : ℝ) : 
  (a^2 + 3*a - 5 = 0) → (-a^2 - 3*a = -5) := by
  sorry

end quadratic_root_property_l1100_110025


namespace ball_distribution_problem_l1100_110066

/-- Represents the number of ways to distribute balls among boxes -/
def distribute_balls (total_balls : ℕ) (num_boxes : ℕ) (min_balls : ℕ → ℕ) : ℕ :=
  sorry

/-- The specific problem setup -/
def problem_setup : ℕ × ℕ × (ℕ → ℕ) :=
  (9, 3, fun i => i)

theorem ball_distribution_problem :
  let (total_balls, num_boxes, min_balls) := problem_setup
  distribute_balls total_balls num_boxes min_balls = 10 := by
  sorry

end ball_distribution_problem_l1100_110066


namespace packages_to_deliver_l1100_110036

/-- The number of packages received yesterday -/
def packages_yesterday : ℕ := 80

/-- The number of packages received today -/
def packages_today : ℕ := 2 * packages_yesterday

/-- The total number of packages to be delivered tomorrow -/
def total_packages : ℕ := packages_yesterday + packages_today

theorem packages_to_deliver :
  total_packages = 240 :=
sorry

end packages_to_deliver_l1100_110036


namespace solution_set_for_a_equals_one_range_of_a_for_existence_condition_l1100_110068

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| + |2*x + a|

-- Part I
theorem solution_set_for_a_equals_one :
  let a := 1
  {x : ℝ | f a x ≥ 5} = Set.Ici 2 ∪ Set.Iic (-4/3) := by sorry

-- Part II
theorem range_of_a_for_existence_condition :
  (∃ x₀ : ℝ, f a x₀ + |x₀ - 2| < 3) ↔ -7 < a ∧ a < -1 := by sorry

end solution_set_for_a_equals_one_range_of_a_for_existence_condition_l1100_110068


namespace cos_theta_plus_pi_fourth_l1100_110024

theorem cos_theta_plus_pi_fourth (θ : Real) :
  (3 : Real) = 5 * Real.cos θ ∧ (-4 : Real) = 5 * Real.sin θ →
  Real.cos (θ + π/4) = 7 * Real.sqrt 2 / 10 := by sorry

end cos_theta_plus_pi_fourth_l1100_110024


namespace ellipse_line_intersection_l1100_110015

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus_F : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

theorem ellipse_line_intersection :
  ∃ k : ℝ, k = 2 ∨ k = -2 ∧
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧
    ellipse_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧
    line_l k x₂ y₂ ∧
    perpendicular x₁ y₁ x₂ y₂ :=
sorry

end ellipse_line_intersection_l1100_110015


namespace positive_integer_problem_l1100_110058

theorem positive_integer_problem (n p : ℕ) (h_p_prime : Nat.Prime p) 
  (h_division : n / (12 * p) = 2) (h_n_ge_48 : n ≥ 48) : n = 48 := by
  sorry

end positive_integer_problem_l1100_110058


namespace infinitely_many_powers_of_two_l1100_110078

def lastDigit (n : ℕ) : ℕ := n % 10

def sequenceA : ℕ → ℕ
  | 0 => 0  -- This is a placeholder, as a₁ is actually the first term
  | n + 1 => sequenceA n + lastDigit (sequenceA n)

theorem infinitely_many_powers_of_two 
  (h₁ : sequenceA 1 % 5 ≠ 0)  -- a₁ is not divisible by 5
  (h₂ : ∀ n, sequenceA (n + 1) = sequenceA n + lastDigit (sequenceA n)) :
  ∀ k, ∃ n, ∃ m, sequenceA n = 2^m ∧ m ≥ k :=
sorry

end infinitely_many_powers_of_two_l1100_110078


namespace cube_root_function_l1100_110070

theorem cube_root_function (k : ℝ) : 
  (∀ x : ℝ, x > 0 → (k * x^(1/3) = 4 * Real.sqrt 3 ↔ x = 64)) →
  k * 8^(1/3) = 2 * Real.sqrt 3 := by
  sorry

end cube_root_function_l1100_110070


namespace twin_pairs_probability_l1100_110063

/-- Represents the gender composition of a pair of twins -/
inductive TwinPair
  | BothBoys
  | BothGirls
  | Mixed

/-- The probability of each outcome for a pair of twins -/
def pairProbability : TwinPair → ℚ
  | TwinPair.BothBoys => 1/3
  | TwinPair.BothGirls => 1/3
  | TwinPair.Mixed => 1/3

/-- The probability of two pairs of twins having a specific composition -/
def twoTwinPairsProbability (pair1 pair2 : TwinPair) : ℚ :=
  pairProbability pair1 * pairProbability pair2

theorem twin_pairs_probability :
  (twoTwinPairsProbability TwinPair.BothBoys TwinPair.BothBoys +
   twoTwinPairsProbability TwinPair.BothGirls TwinPair.BothGirls) =
  (twoTwinPairsProbability TwinPair.Mixed TwinPair.BothBoys +
   twoTwinPairsProbability TwinPair.Mixed TwinPair.BothGirls) ∧
  (twoTwinPairsProbability TwinPair.BothBoys TwinPair.BothBoys +
   twoTwinPairsProbability TwinPair.BothGirls TwinPair.BothGirls) = 2/9 :=
by sorry

#check twin_pairs_probability

end twin_pairs_probability_l1100_110063


namespace inscribed_square_area_is_2210_l1100_110029

/-- Represents a triangle with an inscribed square -/
structure TriangleWithInscribedSquare where
  /-- Length of side PQ -/
  pq : ℝ
  /-- Length of side PR -/
  pr : ℝ
  /-- Side length of the inscribed square -/
  square_side : ℝ
  /-- The square is inscribed in the triangle -/
  is_inscribed : square_side > 0

/-- The area of the inscribed square in the given triangle -/
def inscribed_square_area (t : TriangleWithInscribedSquare) : ℝ :=
  t.square_side^2

/-- Theorem: The area of the inscribed square is 2210 when PQ = 34 and PR = 65 -/
theorem inscribed_square_area_is_2210
    (t : TriangleWithInscribedSquare)
    (h_pq : t.pq = 34)
    (h_pr : t.pr = 65) :
    inscribed_square_area t = 2210 := by
  sorry

end inscribed_square_area_is_2210_l1100_110029


namespace employee_savings_l1100_110076

/-- Calculates the combined savings of three employees over a given period. -/
def combinedSavings (hourlyWage : ℚ) (hoursPerDay : ℚ) (daysPerWeek : ℚ) (weeks : ℚ)
  (savingsRate1 savingsRate2 savingsRate3 : ℚ) : ℚ :=
  let weeklyWage := hourlyWage * hoursPerDay * daysPerWeek
  let totalPeriod := weeklyWage * weeks
  totalPeriod * (savingsRate1 + savingsRate2 + savingsRate3)

/-- The combined savings of three employees with given work conditions and savings rates
    over four weeks is $3000. -/
theorem employee_savings : 
  combinedSavings 10 10 5 4 (2/5) (3/5) (1/2) = 3000 := by
  sorry

end employee_savings_l1100_110076


namespace roots_sum_powers_l1100_110020

theorem roots_sum_powers (a b : ℝ) : 
  a^2 - 6*a + 8 = 0 → 
  b^2 - 6*b + 8 = 0 → 
  a^5 + a^3*b^3 + b^5 = -568 := by
sorry

end roots_sum_powers_l1100_110020


namespace red_tint_percentage_after_modification_l1100_110096

/-- Calculates the percentage of red tint in a modified paint mixture -/
theorem red_tint_percentage_after_modification
  (initial_volume : ℝ)
  (initial_red_tint_percentage : ℝ)
  (added_red_tint : ℝ)
  (h_initial_volume : initial_volume = 40)
  (h_initial_red_tint_percentage : initial_red_tint_percentage = 35)
  (h_added_red_tint : added_red_tint = 10) :
  let initial_red_tint := initial_volume * initial_red_tint_percentage / 100
  let final_red_tint := initial_red_tint + added_red_tint
  let final_volume := initial_volume + added_red_tint
  final_red_tint / final_volume * 100 = 48 := by
  sorry

end red_tint_percentage_after_modification_l1100_110096


namespace modular_inverse_of_3_mod_31_l1100_110089

theorem modular_inverse_of_3_mod_31 :
  ∃ x : ℕ, x ≤ 30 ∧ (3 * x) % 31 = 1 :=
by
  use 21
  sorry

end modular_inverse_of_3_mod_31_l1100_110089


namespace largest_n_binomial_sum_exists_n_binomial_sum_l1100_110002

theorem largest_n_binomial_sum (n : ℕ) : 
  (Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n) → n ≤ 6 :=
by sorry

theorem exists_n_binomial_sum : 
  ∃ n : ℕ, Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧ n = 6 :=
by sorry

end largest_n_binomial_sum_exists_n_binomial_sum_l1100_110002


namespace buddy_gym_class_size_l1100_110016

/-- The number of students in Buddy's gym class -/
def total_students (group1 : ℕ) (group2 : ℕ) : ℕ := group1 + group2

/-- Theorem stating the total number of students in Buddy's gym class -/
theorem buddy_gym_class_size :
  total_students 34 37 = 71 := by
  sorry

end buddy_gym_class_size_l1100_110016


namespace subtraction_result_l1100_110072

theorem subtraction_result : 
  let total : ℚ := 8000
  let fraction1 : ℚ := 1 / 10
  let fraction2 : ℚ := 1 / 20 * (1 / 100)
  (total * fraction1) - (total * fraction2) = 796 :=
by sorry

end subtraction_result_l1100_110072


namespace point_on_line_value_l1100_110018

theorem point_on_line_value (a b : ℝ) (h : b = 3 * a - 2) : 2 * b - 6 * a + 2 = -2 := by
  sorry

end point_on_line_value_l1100_110018


namespace weaving_woman_problem_l1100_110094

/-- Represents the amount of cloth woven on a given day -/
def cloth_woven (day : ℕ) (initial_amount : ℚ) : ℚ :=
  initial_amount * 2^(day - 1)

/-- The problem of the weaving woman -/
theorem weaving_woman_problem :
  ∃ (initial_amount : ℚ),
    (∀ (day : ℕ), day > 0 → cloth_woven day initial_amount = initial_amount * 2^(day - 1)) ∧
    cloth_woven 5 initial_amount = 5 ∧
    initial_amount = 5/31 := by
  sorry

end weaving_woman_problem_l1100_110094


namespace sqrt_product_sqrt_l1100_110013

theorem sqrt_product_sqrt (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  Real.sqrt a * Real.sqrt b = Real.sqrt (a * b) :=
by sorry

end sqrt_product_sqrt_l1100_110013


namespace line_equation_through_midpoint_l1100_110009

/-- A line passing through point P (1, 3) intersects the coordinate axes at points A and B. 
    P is the midpoint of AB. The equation of the line is 3x + y - 6 = 0. -/
theorem line_equation_through_midpoint (A B P : ℝ × ℝ) : 
  P = (1, 3) →
  (∃ a b : ℝ, A = (a, 0) ∧ B = (0, b)) →
  P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) →
  ∀ x y : ℝ, (3 * x + y - 6 = 0) ↔ (∃ t : ℝ, (x, y) = (1 - t, 3 + t * (B.2 - 3))) :=
by sorry

end line_equation_through_midpoint_l1100_110009


namespace f_minus_g_zero_iff_k_eq_9_4_l1100_110035

/-- The function f(x) = 5x^2 - 3x + 2 -/
def f (x : ℝ) : ℝ := 5 * x^2 - 3 * x + 2

/-- The function g(x) = x^3 - 2x^2 + kx - 10 -/
def g (k x : ℝ) : ℝ := x^3 - 2 * x^2 + k * x - 10

/-- Theorem stating that f(5) - g(5) = 0 if and only if k = 9.4 -/
theorem f_minus_g_zero_iff_k_eq_9_4 : 
  ∀ k : ℝ, f 5 - g k 5 = 0 ↔ k = 9.4 := by sorry

end f_minus_g_zero_iff_k_eq_9_4_l1100_110035


namespace justine_colored_sheets_l1100_110039

/-- Given 2450 sheets of paper evenly split into 5 binders, 
    prove that Justine colors 245 sheets when she colors 
    half the sheets in one binder. -/
theorem justine_colored_sheets : 
  let total_sheets : ℕ := 2450
  let num_binders : ℕ := 5
  let sheets_per_binder : ℕ := total_sheets / num_binders
  let justine_colored : ℕ := sheets_per_binder / 2
  justine_colored = 245 := by
  sorry

#check justine_colored_sheets

end justine_colored_sheets_l1100_110039


namespace simplify_expression_l1100_110073

theorem simplify_expression (a : ℝ) (h : a < (1/4 : ℝ)) :
  4 * (4*a - 1)^2 = Real.sqrt (1 - 4*a) := by
  sorry

end simplify_expression_l1100_110073


namespace complex_exponential_to_rectangular_l1100_110043

theorem complex_exponential_to_rectangular : 2 * Complex.exp (15 * π * I / 4) = Complex.mk (Real.sqrt 2) (- Real.sqrt 2) := by
  sorry

end complex_exponential_to_rectangular_l1100_110043


namespace largest_base7_five_digit_to_base10_l1100_110077

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ (digits.length - 1 - i))) 0

/-- The largest five-digit number in base-7 --/
def largestBase7FiveDigit : List Nat := [6, 6, 6, 6, 6]

theorem largest_base7_five_digit_to_base10 :
  base7ToBase10 largestBase7FiveDigit = 16806 := by
  sorry

end largest_base7_five_digit_to_base10_l1100_110077


namespace some_magical_beings_are_mystical_creatures_l1100_110091

-- Define the universe
variable (U : Type)

-- Define the predicates
variable (Dragon : U → Prop)
variable (MagicalBeing : U → Prop)
variable (MysticalCreature : U → Prop)

-- State the theorem
theorem some_magical_beings_are_mystical_creatures :
  (∀ x, Dragon x → MagicalBeing x) →  -- All dragons are magical beings
  (∃ x, MysticalCreature x ∧ Dragon x) →  -- Some mystical creatures are dragons
  (∃ x, MagicalBeing x ∧ MysticalCreature x)  -- Some magical beings are mystical creatures
:= by sorry

end some_magical_beings_are_mystical_creatures_l1100_110091


namespace divisor_problem_l1100_110074

theorem divisor_problem (initial_number : ℕ) (added_number : ℝ) (divisor : ℕ) : 
  initial_number = 1782452 →
  added_number = 48.00000000010186 →
  divisor = 500 →
  divisor = (Int.toNat (round (initial_number + added_number))).gcd (Int.toNat (round (initial_number + added_number))) :=
by sorry

end divisor_problem_l1100_110074


namespace count_divisors_not_divisible_by_three_of_180_l1100_110084

def divisors_not_divisible_by_three (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).filter (λ x => x ∣ n ∧ ¬(3 ∣ x))

theorem count_divisors_not_divisible_by_three_of_180 :
  (divisors_not_divisible_by_three 180).card = 6 := by
  sorry

end count_divisors_not_divisible_by_three_of_180_l1100_110084


namespace problem_1_l1100_110032

theorem problem_1 : (-1) * (-4) + 2^2 / (7 - 5) = 6 := by
  sorry

end problem_1_l1100_110032


namespace vector_subtraction_and_scalar_multiplication_l1100_110061

theorem vector_subtraction_and_scalar_multiplication :
  (⟨2, -5⟩ : ℝ × ℝ) - 4 • (⟨-1, 7⟩ : ℝ × ℝ) = (⟨6, -33⟩ : ℝ × ℝ) := by
  sorry

end vector_subtraction_and_scalar_multiplication_l1100_110061


namespace binomial_coefficient_probability_l1100_110069

theorem binomial_coefficient_probability : 
  let n : ℕ := 10
  let positive_coeff : ℕ := 6
  let negative_coeff : ℕ := 5
  let total_coeff : ℕ := positive_coeff + negative_coeff
  let ways_to_choose_opposite_signs : ℕ := positive_coeff * negative_coeff
  let total_ways_to_choose : ℕ := (total_coeff * (total_coeff - 1)) / 2
  (ways_to_choose_opposite_signs : ℚ) / total_ways_to_choose = 6 / 11 :=
by
  sorry

end binomial_coefficient_probability_l1100_110069


namespace systematic_sampling_theorem_l1100_110081

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  students_per_group : ℕ
  selected_number : ℕ
  selected_group : ℕ

/-- Theorem: In a systematic sampling of 50 students into 5 groups of 10 each,
    if the student with number 22 is selected from the third group,
    then the student with number 42 will be selected from the fifth group. -/
theorem systematic_sampling_theorem (s : SystematicSampling)
    (h1 : s.total_students = 50)
    (h2 : s.num_groups = 5)
    (h3 : s.students_per_group = 10)
    (h4 : s.selected_number = 22)
    (h5 : s.selected_group = 3) :
    s.selected_number + (s.num_groups - s.selected_group) * s.students_per_group = 42 :=
  sorry


end systematic_sampling_theorem_l1100_110081


namespace quadratic_roots_l1100_110057

theorem quadratic_roots : ∃ (x₁ x₂ : ℝ), x₁ = 0 ∧ x₂ = 2 ∧ 
  (∀ x : ℝ, x^2 - 2*x = 0 ↔ x = x₁ ∨ x = x₂) := by
  sorry

end quadratic_roots_l1100_110057


namespace no_solution_implies_a_range_l1100_110053

def otimes (x y : ℝ) : ℝ := x * (1 - y)

theorem no_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(otimes (x - a) (x + 1) ≥ 1)) → -2 < a ∧ a < 2 := by
  sorry

end no_solution_implies_a_range_l1100_110053


namespace equation_solution_l1100_110090

theorem equation_solution (x y : ℝ) (h : 17 * x + 51 * y = 102) : 9 * x + 27 * y = 54 := by
  sorry

end equation_solution_l1100_110090


namespace pauls_garage_sale_l1100_110001

/-- The number of books Paul sold in the garage sale -/
def books_sold (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : ℕ :=
  initial - given_away - remaining

/-- Proof that Paul sold 27 books in the garage sale -/
theorem pauls_garage_sale : books_sold 134 39 68 = 27 := by
  sorry

end pauls_garage_sale_l1100_110001


namespace sand_truck_loads_l1100_110000

/-- Proves that the truck-loads of sand required is equal to 0.1666666666666666,
    given the total truck-loads of material needed and the truck-loads of dirt and cement. -/
theorem sand_truck_loads (total material_needed dirt cement sand : ℚ)
    (h1 : total = 0.6666666666666666)
    (h2 : dirt = 0.3333333333333333)
    (h3 : cement = 0.16666666666666666)
    (h4 : sand = total - (dirt + cement)) :
    sand = 0.1666666666666666 := by
  sorry

end sand_truck_loads_l1100_110000


namespace festival_guests_selection_l1100_110021

theorem festival_guests_selection (n m : ℕ) (h1 : n = 10) (h2 : m = 5) : 
  Nat.choose n m = 252 := by
  sorry

end festival_guests_selection_l1100_110021


namespace road_completion_proof_l1100_110092

def road_paving (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n+1 => road_paving n + 1 / road_paving n

theorem road_completion_proof :
  ∃ n : ℕ, n ≤ 5001 ∧ road_paving n ≥ 100 := by
  sorry

end road_completion_proof_l1100_110092


namespace halloween_candy_theorem_l1100_110019

/-- The number of candy pieces Katie and her sister have left after eating some on Halloween night -/
theorem halloween_candy_theorem (katie_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : 
  katie_candy = 8 → sister_candy = 23 → eaten_candy = 8 →
  katie_candy + sister_candy - eaten_candy = 23 := by
  sorry

end halloween_candy_theorem_l1100_110019


namespace school_early_arrival_l1100_110088

theorem school_early_arrival (usual_time : ℝ) (rate_ratio : ℝ) (time_saved : ℝ) : 
  usual_time = 24 →
  rate_ratio = 6 / 5 →
  time_saved = usual_time - (usual_time / rate_ratio) →
  time_saved = 4 := by
sorry

end school_early_arrival_l1100_110088


namespace max_product_l1100_110038

theorem max_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 40) :
  x^4 * y^3 ≤ (160/7)^4 * (120/7)^3 ∧
  x^4 * y^3 = (160/7)^4 * (120/7)^3 ↔ x = 160/7 ∧ y = 120/7 := by
  sorry

end max_product_l1100_110038


namespace sufficient_not_necessary_condition_l1100_110022

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) ↔ 
  ((a^2 + b^2 = 1 → ∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1) ∧
   (∃ a b : ℝ, a^2 + b^2 ≠ 1 ∧ ∀ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≤ 1)) :=
by sorry

end sufficient_not_necessary_condition_l1100_110022


namespace special_hexagon_area_l1100_110031

/-- An equilateral hexagon with specific interior angles -/
structure SpecialHexagon where
  -- Side length of the hexagon
  side_length : ℝ
  -- Interior angles of the hexagon in radians
  angles : Fin 6 → ℝ
  -- The hexagon is equilateral
  equilateral : side_length = 1
  -- The interior angles are as specified
  angle_values : angles = ![π/2, 2*π/3, 5*π/6, π/2, 2*π/3, 5*π/6]

/-- The area of the special hexagon -/
def area (h : SpecialHexagon) : ℝ := sorry

/-- Theorem stating the area of the special hexagon -/
theorem special_hexagon_area (h : SpecialHexagon) : area h = (3 + Real.sqrt 3) / 2 := by sorry

end special_hexagon_area_l1100_110031


namespace farmer_land_ownership_l1100_110040

theorem farmer_land_ownership (T : ℝ) 
  (h1 : T > 0)
  (h2 : 0.8 * T + 0.2 * T = T)
  (h3 : 0.05 * (0.8 * T) + 0.3 * (0.2 * T) = 720) :
  0.8 * T = 5760 := by
  sorry

end farmer_land_ownership_l1100_110040


namespace proposition_correctness_l1100_110026

theorem proposition_correctness : 
  (∃ (S : Finset (Prop)), 
    S.card = 4 ∧ 
    (∃ (incorrect : Finset (Prop)), 
      incorrect ⊆ S ∧ 
      incorrect.card = 2 ∧
      (∀ p ∈ S, p ∈ incorrect ↔ ¬p) ∧
      (∃ p ∈ S, p = (∀ (p q : Prop), p ∨ q → p ∧ q)) ∧
      (∃ p ∈ S, p = (∀ x : ℝ, x > 5 → x^2 - 4*x - 5 > 0) ∧ 
                   (∃ y : ℝ, y^2 - 4*y - 5 > 0 ∧ y ≤ 5)) ∧
      (∃ p ∈ S, p = ((¬∃ x : ℝ, x^2 + x - 1 < 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≥ 0))) ∧
      (∃ p ∈ S, p = (∀ x : ℝ, (x ≠ 1 ∨ x ≠ 2) → x^2 - 3*x + 2 ≠ 0)))) := by
  sorry

end proposition_correctness_l1100_110026


namespace cars_combined_efficiency_l1100_110067

/-- Calculates the combined fuel efficiency of three cars given their individual efficiencies -/
def combinedFuelEfficiency (e1 e2 e3 : ℚ) : ℚ :=
  3 / (1 / e1 + 1 / e2 + 1 / e3)

/-- Theorem: The combined fuel efficiency of cars with 30, 15, and 20 mpg is 20 mpg -/
theorem cars_combined_efficiency :
  combinedFuelEfficiency 30 15 20 = 20 := by
  sorry

#eval combinedFuelEfficiency 30 15 20

end cars_combined_efficiency_l1100_110067


namespace regular_polygon_sides_l1100_110010

/-- The number of sides of a regular polygon whose sum of interior angles is 1080° more than
    the sum of exterior angles of a pentagon. -/
def num_sides_regular_polygon : ℕ := 10

/-- The sum of exterior angles of any polygon is always 360°. -/
axiom sum_exterior_angles : ℕ → ℝ
axiom sum_exterior_angles_def : ∀ n : ℕ, sum_exterior_angles n = 360

/-- The sum of interior angles of a polygon with n sides is (n-2) * 180°. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- Theorem stating that the number of sides of the regular polygon is 10. -/
theorem regular_polygon_sides :
  sum_interior_angles num_sides_regular_polygon =
  sum_exterior_angles 5 + 1080 :=
sorry

end regular_polygon_sides_l1100_110010


namespace solution_interval_l1100_110059

theorem solution_interval (x : ℝ) : (x / 4 ≤ 3 + x ∧ 3 + x < -3 * (1 + x)) ↔ x ∈ Set.Ici (-4) ∩ Set.Iio (-1) := by
  sorry

end solution_interval_l1100_110059


namespace mixed_number_calculation_l1100_110011

theorem mixed_number_calculation : 
  36 * ((5 + 1/6) - (6 + 1/7)) / ((3 + 1/6) + (2 + 1/7)) = -(6 + 156/223) :=
by sorry

end mixed_number_calculation_l1100_110011


namespace adult_ticket_cost_l1100_110071

/-- Proves that the cost of each adult ticket is $31.50 given the problem conditions --/
theorem adult_ticket_cost : 
  let child_ticket_cost : ℚ := 15/2
  let total_bill : ℚ := 138
  let total_tickets : ℕ := 12
  ∀ (adult_tickets : ℕ) (child_tickets : ℕ) (adult_ticket_cost : ℚ),
    child_tickets = adult_tickets + 8 →
    adult_tickets + child_tickets = total_tickets →
    adult_tickets * adult_ticket_cost + child_tickets * child_ticket_cost = total_bill →
    adult_ticket_cost = 63/2 :=
by
  sorry

end adult_ticket_cost_l1100_110071


namespace appliance_final_cost_l1100_110065

theorem appliance_final_cost (initial_price : ℝ) : 
  initial_price * 1.4 = 1680 →
  (1680 * 0.8) * 0.9 = 1209.6 :=
by sorry

end appliance_final_cost_l1100_110065


namespace quadratic_equation_value_l1100_110008

theorem quadratic_equation_value (x : ℝ) : 2*x^2 + 3*x + 7 = 8 → 9 - 4*x^2 - 6*x = 7 := by
  sorry

end quadratic_equation_value_l1100_110008


namespace road_trip_distance_ratio_l1100_110050

theorem road_trip_distance_ratio : 
  ∀ (tracy michelle katie : ℕ),
  tracy + michelle + katie = 1000 →
  tracy = 2 * michelle + 20 →
  michelle = 294 →
  (michelle : ℚ) / (katie : ℚ) = 3 / 1 := by
sorry

end road_trip_distance_ratio_l1100_110050


namespace no_solution_absolute_value_equation_l1100_110046

theorem no_solution_absolute_value_equation : ¬∃ (x : ℝ), |(-4 * x)| + 6 = 0 := by
  sorry

end no_solution_absolute_value_equation_l1100_110046


namespace smallest_three_digit_multiple_of_17_l1100_110034

theorem smallest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n :=
sorry

end smallest_three_digit_multiple_of_17_l1100_110034


namespace ratio_proof_l1100_110007

def problem (A B : ℕ) : Prop :=
  A = 45 ∧ Nat.lcm A B = 180

theorem ratio_proof (A B : ℕ) (h : problem A B) : 
  A / B = 45 / 4 := by sorry

end ratio_proof_l1100_110007


namespace polygon_angle_sums_l1100_110098

/-- For an n-sided polygon, the sum of exterior angles is 360° and the sum of interior angles is (n-2) × 180° -/
theorem polygon_angle_sums (n : ℕ) (h : n ≥ 3) :
  ∃ (exterior_sum interior_sum : ℝ),
    exterior_sum = 360 ∧
    interior_sum = (n - 2) * 180 :=
by sorry

end polygon_angle_sums_l1100_110098


namespace chocolate_box_problem_l1100_110044

theorem chocolate_box_problem (N : ℕ) (rows columns : ℕ) :
  -- Initial conditions
  N > 0 ∧ rows > 0 ∧ columns > 0 ∧
  -- After operations, one-third remains
  N / 3 > 0 ∧
  -- Three rows minus one can be filled at one point
  (3 * columns - 1 ≤ N ∧ 3 * columns > N / 3) ∧
  -- Five columns minus one can be filled at another point
  (5 * rows - 1 ≤ N ∧ 5 * rows > N / 3) →
  -- Conclusions
  N = 60 ∧ N - (3 * columns - 1) = 25 := by
  sorry

end chocolate_box_problem_l1100_110044


namespace bennys_work_hours_l1100_110086

/-- Given that Benny worked for 6 days and a total of 18 hours, 
    prove that he worked 3 hours each day. -/
theorem bennys_work_hours (days : ℕ) (total_hours : ℕ) 
    (h1 : days = 6) (h2 : total_hours = 18) : 
    total_hours / days = 3 := by
  sorry

end bennys_work_hours_l1100_110086


namespace inequality_solution_set_l1100_110004

theorem inequality_solution_set :
  {x : ℝ | (x + 1) * (2 - x) < 0} = {x : ℝ | x > 2 ∨ x < -1} := by
sorry

end inequality_solution_set_l1100_110004


namespace solve_pizza_problem_l1100_110014

def pizza_problem (total_slices : ℕ) (slices_left : ℕ) (slices_per_person : ℕ) : Prop :=
  let slices_eaten := total_slices - slices_left
  slices_eaten / slices_per_person = 6

theorem solve_pizza_problem :
  pizza_problem 16 4 2 := by
  sorry

end solve_pizza_problem_l1100_110014


namespace pen_profit_calculation_l1100_110082

/-- Calculates the profit from selling pens given the purchase quantity, cost rate, and selling rate. -/
def calculate_profit (purchase_quantity : ℕ) (cost_rate : ℚ × ℚ) (selling_rate : ℚ × ℚ) : ℚ :=
  let cost_per_pen := cost_rate.2 / cost_rate.1
  let total_cost := cost_per_pen * purchase_quantity
  let selling_price_per_pen := selling_rate.2 / selling_rate.1
  let total_revenue := selling_price_per_pen * purchase_quantity
  total_revenue - total_cost

/-- The profit from selling 1200 pens, bought at 4 for $3 and sold at 3 for $2, is -$96. -/
theorem pen_profit_calculation :
  calculate_profit 1200 (4, 3) (3, 2) = -96 := by
  sorry

end pen_profit_calculation_l1100_110082


namespace rational_sqrt_fraction_l1100_110083

theorem rational_sqrt_fraction (r q n : ℚ) 
  (h : 1 / (r + q * n) + 1 / (q + r * n) = 1 / (r + q)) :
  ∃ (m : ℚ), (n - 3) / (n + 1) = m^2 := by
sorry

end rational_sqrt_fraction_l1100_110083


namespace smallest_AAB_l1100_110097

def is_digit (n : ℕ) : Prop := 1 ≤ n ∧ n ≤ 9

def two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def AB (a b : ℕ) : ℕ := 10 * a + b

def AAB (a b : ℕ) : ℕ := 100 * a + 10 * a + b

theorem smallest_AAB :
  ∃ (a b : ℕ),
    is_digit a ∧
    is_digit b ∧
    two_digit (AB a b) ∧
    three_digit (AAB a b) ∧
    AB a b = (AAB a b) / 7 ∧
    AAB a b = 996 ∧
    (∀ (x y : ℕ),
      is_digit x ∧
      is_digit y ∧
      two_digit (AB x y) ∧
      three_digit (AAB x y) ∧
      AB x y = (AAB x y) / 7 →
      AAB x y ≥ 996) :=
by sorry

end smallest_AAB_l1100_110097


namespace sum_of_slopes_constant_l1100_110051

/-- An ellipse with eccentricity 1/2 passing through (2,0) -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_ecc : a^2 - b^2 = (a/2)^2
  h_thru_point : 4/a^2 + 0/b^2 = 1

/-- A line passing through (1,0) intersecting the ellipse -/
structure IntersectingLine (E : Ellipse) where
  k : ℝ
  h_intersect : ∃ x y, x^2/E.a^2 + y^2/E.b^2 = 1 ∧ y = k*(x-1)

/-- The point P -/
def P : ℝ × ℝ := (4, 3)

/-- Slopes of PA and PB -/
def slopes (E : Ellipse) (L : IntersectingLine E) : ℝ × ℝ :=
  sorry

/-- The theorem to be proved -/
theorem sum_of_slopes_constant (E : Ellipse) (L : IntersectingLine E) :
  let (k₁, k₂) := slopes E L
  k₁ + k₂ = 2 := by sorry

end sum_of_slopes_constant_l1100_110051


namespace point_outside_circle_l1100_110064

theorem point_outside_circle (m : ℝ) : 
  let P : ℝ × ℝ := (m^2, 5)
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 24}
  P ∉ circle := by
sorry

end point_outside_circle_l1100_110064


namespace cubic_equation_root_l1100_110042

theorem cubic_equation_root (c d : ℚ) : 
  (3 + 2 * Real.sqrt 5)^3 + c * (3 + 2 * Real.sqrt 5)^2 + d * (3 + 2 * Real.sqrt 5) + 45 = 0 →
  c = -10 := by
sorry

end cubic_equation_root_l1100_110042


namespace telescope_visual_range_l1100_110056

theorem telescope_visual_range 
  (original_range : ℝ) 
  (percentage_increase : ℝ) 
  (new_range : ℝ) : 
  original_range = 50 → 
  percentage_increase = 200 → 
  new_range = original_range + (percentage_increase / 100) * original_range → 
  new_range = 150 := by
sorry

end telescope_visual_range_l1100_110056


namespace angle_sum_inequality_l1100_110006

theorem angle_sum_inequality (θ₁ θ₂ θ₃ θ₄ : Real) 
  (h₁ : 0 < θ₁ ∧ θ₁ < π/2)
  (h₂ : 0 < θ₂ ∧ θ₂ < π/2)
  (h₃ : 0 < θ₃ ∧ θ₃ < π/2)
  (h₄ : 0 < θ₄ ∧ θ₄ < π/2)
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) :
  (Real.sqrt 2 * Real.sin θ₁ - 1) / Real.cos θ₁ +
  (Real.sqrt 2 * Real.sin θ₂ - 1) / Real.cos θ₂ +
  (Real.sqrt 2 * Real.sin θ₃ - 1) / Real.cos θ₃ +
  (Real.sqrt 2 * Real.sin θ₄ - 1) / Real.cos θ₄ ≥ 0 := by
  sorry

end angle_sum_inequality_l1100_110006


namespace tangent_line_at_x_1_l1100_110005

/-- The equation of the tangent line to y = x³ + 2x + 1 at x = 1 is 5x - y - 1 = 0 -/
theorem tangent_line_at_x_1 : 
  let f (x : ℝ) := x^3 + 2*x + 1
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := (3 * x₀^2 + 2)
  ∀ x y : ℝ, (y - y₀ = k * (x - x₀)) ↔ (5*x - y - 1 = 0) :=
by sorry

end tangent_line_at_x_1_l1100_110005


namespace power_multiplication_equality_l1100_110052

theorem power_multiplication_equality : (-0.25)^2023 * 4^2024 = -4 := by
  sorry

end power_multiplication_equality_l1100_110052


namespace max_value_constraint_max_value_attained_l1100_110099

theorem max_value_constraint (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) : 
  2 * x + y ≤ Real.sqrt 11 := by
sorry

theorem max_value_attained : ∃ (x y : ℝ), 3 * x^2 + 2 * y^2 ≤ 6 ∧ 2 * x + y = Real.sqrt 11 := by
sorry

end max_value_constraint_max_value_attained_l1100_110099


namespace hot_dogs_remainder_l1100_110048

theorem hot_dogs_remainder : 35876119 % 7 = 6 := by sorry

end hot_dogs_remainder_l1100_110048


namespace smallest_absolute_value_rational_l1100_110095

theorem smallest_absolute_value_rational : 
  ∀ q : ℚ, |0| ≤ |q| := by
  sorry

end smallest_absolute_value_rational_l1100_110095


namespace no_real_solutions_l1100_110054

theorem no_real_solutions : ¬∃ x : ℝ, (2*x^2 - 6*x + 5)^2 + 1 = -|x| := by
  sorry

end no_real_solutions_l1100_110054


namespace total_caps_produced_l1100_110062

def week1_production : ℕ := 320
def week2_production : ℕ := 400
def week3_production : ℕ := 300

def average_production : ℕ := (week1_production + week2_production + week3_production) / 3

def total_production : ℕ := week1_production + week2_production + week3_production + average_production

theorem total_caps_produced : total_production = 1360 := by
  sorry

end total_caps_produced_l1100_110062


namespace max_min_product_l1100_110030

theorem max_min_product (A B : ℕ) (sum_constraint : A + B = 100) :
  (∃ (X Y : ℕ), X + Y = 100 ∧ X * Y ≤ A * B) ∧
  (∃ (X Y : ℕ), X + Y = 100 ∧ A * B ≤ X * Y) ∧
  (∃ (X Y : ℕ), X + Y = 100 ∧ X * Y = 2500) ∧
  (∃ (X Y : ℕ), X + Y = 100 ∧ X * Y = 0) :=
by sorry

end max_min_product_l1100_110030


namespace tan_double_angle_l1100_110080

theorem tan_double_angle (α : Real) (h : Real.tan α = 1/3) : Real.tan (2 * α) = 3/4 := by
  sorry

end tan_double_angle_l1100_110080


namespace basketball_weight_proof_l1100_110041

/-- The weight of a skateboard in pounds -/
def skateboard_weight : ℝ := 32

/-- The number of skateboards that balance with the basketballs -/
def num_skateboards : ℕ := 4

/-- The number of basketballs that balance with the skateboards -/
def num_basketballs : ℕ := 8

/-- The weight of a single basketball in pounds -/
def basketball_weight : ℝ := 16

theorem basketball_weight_proof :
  num_basketballs * basketball_weight = num_skateboards * skateboard_weight :=
by sorry

end basketball_weight_proof_l1100_110041


namespace price_difference_l1100_110093

def original_price : ℚ := 150
def tax_rate : ℚ := 0.07
def discount_rate : ℚ := 0.25
def service_charge_rate : ℚ := 0.05

def ann_price : ℚ :=
  original_price * (1 + tax_rate) * (1 - discount_rate) * (1 + service_charge_rate)

def ben_price : ℚ :=
  original_price * (1 - discount_rate) * (1 + tax_rate)

theorem price_difference :
  ann_price - ben_price = 6.01875 := by sorry

end price_difference_l1100_110093


namespace root_difference_of_cubic_l1100_110023

theorem root_difference_of_cubic (x₁ x₂ x₃ : ℝ) :
  (81 * x₁^3 - 162 * x₁^2 + 108 * x₁ - 18 = 0) →
  (81 * x₂^3 - 162 * x₂^2 + 108 * x₂ - 18 = 0) →
  (81 * x₃^3 - 162 * x₃^2 + 108 * x₃ - 18 = 0) →
  (x₂ - x₁ = x₃ - x₂) →  -- arithmetic progression condition
  (max x₁ (max x₂ x₃) - min x₁ (min x₂ x₃) = 2/3) :=
by sorry

end root_difference_of_cubic_l1100_110023


namespace quadratic_range_theorem_l1100_110060

/-- The quadratic function f(x) = x^2 + 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 + 2*x + 2

/-- A point P with coordinates (m, n) -/
structure Point where
  m : ℝ
  n : ℝ

theorem quadratic_range_theorem (P : Point) 
  (h1 : P.n = f P.m)  -- P lies on the graph of f
  (h2 : |P.m| < 2)    -- distance from P to y-axis is less than 2
  : -1 ≤ P.n ∧ P.n < 10 := by
  sorry

end quadratic_range_theorem_l1100_110060


namespace solve_property_damage_l1100_110087

def property_damage_problem (medical_bills : ℝ) (carl_payment_percentage : ℝ) (carl_payment : ℝ) : Prop :=
  let total_cost := carl_payment / carl_payment_percentage
  let property_damage := total_cost - medical_bills
  property_damage = 40000

theorem solve_property_damage :
  property_damage_problem 70000 0.2 22000 := by
  sorry

end solve_property_damage_l1100_110087


namespace problem_solution_l1100_110075

theorem problem_solution (s t : ℝ) 
  (eq1 : 12 * s + 8 * t = 160)
  (eq2 : s = t^2 + 2) : 
  t = (Real.sqrt 103 - 1) / 3 := by
  sorry

end problem_solution_l1100_110075


namespace triangle_perimeter_l1100_110028

theorem triangle_perimeter (a b c : ℝ) (A B C : ℝ) :
  b^2 + c^2 - a^2 = 1 →
  b * c = 1 →
  Real.cos B * Real.cos C = -1/8 →
  a + b + c = Real.sqrt 2 + Real.sqrt 5 :=
by sorry

end triangle_perimeter_l1100_110028


namespace probability_of_seven_in_three_eighths_l1100_110037

theorem probability_of_seven_in_three_eighths : 
  let decimal_rep := [3, 7, 5]
  let count_sevens := (decimal_rep.filter (· = 7)).length
  let total_digits := decimal_rep.length
  (count_sevens : ℚ) / total_digits = 1 / 3 := by
sorry

end probability_of_seven_in_three_eighths_l1100_110037
