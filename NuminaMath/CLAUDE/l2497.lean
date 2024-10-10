import Mathlib

namespace root_product_theorem_l2497_249735

theorem root_product_theorem (n r : ℝ) (c d : ℝ) : 
  c^2 - n*c + 3 = 0 → 
  d^2 - n*d + 3 = 0 → 
  (c + 2/d)^2 - r*(c + 2/d) + s = 0 → 
  (d + 2/c)^2 - r*(d + 2/c) + s = 0 → 
  s = 25/3 := by sorry

end root_product_theorem_l2497_249735


namespace zoo_tickets_cost_l2497_249758

/-- The total cost of zoo tickets for a group of children and adults. -/
def total_cost (num_children num_adults child_ticket_price adult_ticket_price : ℕ) : ℕ :=
  num_children * child_ticket_price + num_adults * adult_ticket_price

/-- Theorem: The total cost of zoo tickets for a group of 6 children and 10 adults is $220,
    given that child tickets cost $10 each and adult tickets cost $16 each. -/
theorem zoo_tickets_cost :
  total_cost 6 10 10 16 = 220 := by
sorry

end zoo_tickets_cost_l2497_249758


namespace not_divisible_by_five_l2497_249770

theorem not_divisible_by_five (n : ℤ) : ¬ (5 ∣ (n^2 - 8)) := by
  sorry

end not_divisible_by_five_l2497_249770


namespace max_intersections_convex_ngon_l2497_249700

/-- The maximum number of intersection points of diagonals in a convex n-gon -/
def max_intersections (n : ℕ) : ℕ :=
  n * (n - 1) * (n - 2) * (n - 3) / 24

/-- Theorem: In a convex n-gon with all diagonals drawn, the maximum number of 
    intersection points of the diagonals is equal to C(n,4) = n(n-1)(n-2)(n-3)/24 -/
theorem max_intersections_convex_ngon (n : ℕ) (h : n ≥ 4) :
  max_intersections n = Nat.choose n 4 := by
  sorry

end max_intersections_convex_ngon_l2497_249700


namespace no_solution_for_equation_l2497_249785

theorem no_solution_for_equation :
  ¬ ∃ (x : ℝ), (x - 2) / (x + 2) - 1 = 16 / (x^2 - 4) :=
by sorry

end no_solution_for_equation_l2497_249785


namespace stratified_sampling_theorem_l2497_249757

theorem stratified_sampling_theorem (elementary middle high : ℕ) 
  (h1 : elementary = 126) 
  (h2 : middle = 280) 
  (h3 : high = 95) 
  (sample_size : ℕ) 
  (h4 : sample_size = 100) : 
  ∃ (adjusted_elementary : ℕ), 
    adjusted_elementary = elementary - 1 ∧ 
    (adjusted_elementary + middle + high) % sample_size = 0 ∧
    (adjusted_elementary / 5 + middle / 5 + high / 5 = sample_size) :=
sorry

end stratified_sampling_theorem_l2497_249757


namespace equal_share_theorem_l2497_249743

/-- Represents the number of candies each person has initially -/
structure CandyDistribution :=
  (mark : ℕ)
  (peter : ℕ)
  (john : ℕ)

/-- Calculates the number of candies each person gets after sharing equally -/
def share_candies (dist : CandyDistribution) : ℕ :=
  (dist.mark + dist.peter + dist.john) / 3

/-- Theorem: Given the initial candy distribution, prove that each person gets 30 candies after sharing -/
theorem equal_share_theorem (dist : CandyDistribution) 
  (h1 : dist.mark = 30)
  (h2 : dist.peter = 25)
  (h3 : dist.john = 35) :
  share_candies dist = 30 := by
  sorry

end equal_share_theorem_l2497_249743


namespace symmetric_function_equality_l2497_249767

-- Define a function that is symmetric with respect to x = 1
def SymmetricFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (2 - x) = f x

-- Define the theorem
theorem symmetric_function_equality (f : ℝ → ℝ) (h : SymmetricFunction f) :
  ∃! a : ℝ, f (a - 1) = f 5 ∧ a = 6 := by
  sorry

end symmetric_function_equality_l2497_249767


namespace dans_cards_after_purchase_l2497_249745

/-- The number of baseball cards Dan has after Sam's purchase -/
def dans_remaining_cards (initial_cards sam_bought : ℕ) : ℕ :=
  initial_cards - sam_bought

/-- Theorem: Dan's remaining cards is the difference between his initial cards and those Sam bought -/
theorem dans_cards_after_purchase (initial_cards sam_bought : ℕ) 
  (h : sam_bought ≤ initial_cards) : 
  dans_remaining_cards initial_cards sam_bought = initial_cards - sam_bought := by
  sorry

end dans_cards_after_purchase_l2497_249745


namespace polar_equivalence_l2497_249747

/-- Two points in polar coordinates are equivalent if they represent the same point in the plane. -/
def polar_equivalent (r1 : ℝ) (θ1 : ℝ) (r2 : ℝ) (θ2 : ℝ) : Prop :=
  r1 * (Real.cos θ1) = r2 * (Real.cos θ2) ∧ r1 * (Real.sin θ1) = r2 * (Real.sin θ2)

/-- The theorem stating that (-3, 7π/6) is equivalent to (3, π/6) in polar coordinates. -/
theorem polar_equivalence :
  polar_equivalent (-3) (7 * Real.pi / 6) 3 (Real.pi / 6) ∧ 
  3 > 0 ∧ 
  0 ≤ Real.pi / 6 ∧ 
  Real.pi / 6 < 2 * Real.pi :=
sorry

end polar_equivalence_l2497_249747


namespace square_roots_problem_l2497_249793

theorem square_roots_problem (m a : ℝ) (hm : m > 0) 
  (h1 : (1 - 2*a)^2 = m) (h2 : (a - 5)^2 = m) (h3 : 1 - 2*a ≠ a - 5) : 
  m = 81 := by
sorry

end square_roots_problem_l2497_249793


namespace cos_pi_sixth_plus_alpha_l2497_249717

theorem cos_pi_sixth_plus_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 6) :
  Real.cos (π / 6 + α) = 1 / 6 := by
  sorry

end cos_pi_sixth_plus_alpha_l2497_249717


namespace team_b_average_points_l2497_249776

theorem team_b_average_points (average_first_two : ℝ) : 
  (2 * average_first_two + 47 + 330 > 500) → average_first_two > 61.5 := by
  sorry

end team_b_average_points_l2497_249776


namespace divisible_by_eight_probability_l2497_249753

theorem divisible_by_eight_probability (n : ℕ) : 
  (Finset.filter (λ k => (k * (k + 1)) % 8 = 0) (Finset.range 100)).card / 100 = 1 / 8 := by
sorry

end divisible_by_eight_probability_l2497_249753


namespace centroid_trace_area_l2497_249788

-- Define the circle
def Circle : Type := {p : ℝ × ℝ // (p.1^2 + p.2^2 = 225)}

-- Define points A, B, and C
def A : ℝ × ℝ := (-15, 0)
def B : ℝ × ℝ := (15, 0)

-- Define C as a point on the circle
def C : Circle := sorry

-- Define the centroid of triangle ABC
def centroid (c : Circle) : ℝ × ℝ := sorry

-- Statement to prove
theorem centroid_trace_area :
  ∃ (area : ℝ), area = 25 * Real.pi ∧
  (∀ (c : Circle), c.1 ≠ A ∧ c.1 ≠ B →
    (centroid c).1^2 + (centroid c).2^2 = 25) :=
sorry

end centroid_trace_area_l2497_249788


namespace expand_expression_l2497_249761

theorem expand_expression (y : ℝ) : 5 * (4 * y^3 - 3 * y^2 + 2 * y - 6) = 20 * y^3 - 15 * y^2 + 10 * y - 30 := by
  sorry

end expand_expression_l2497_249761


namespace parabola_triangle_area_l2497_249731

/-- Given a parabola y² = 4x with focus F(1,0), and points A and B on the parabola
    such that FA = 2BF, the area of triangle OAB is 3√2/2. -/
theorem parabola_triangle_area (A B : ℝ × ℝ) :
  let C : ℝ × ℝ → Prop := λ p => p.2^2 = 4 * p.1
  let F : ℝ × ℝ := (1, 0)
  let O : ℝ × ℝ := (0, 0)
  C A ∧ C B ∧ 
  (∃ (t : ℝ), A = F + t • (A - F) ∧ B = F + t • (B - F)) ∧
  (A - F) = 2 • (F - B) →
  abs ((A.1 * B.2 - A.2 * B.1) / 2) = 3 * Real.sqrt 2 / 2 :=
by sorry


end parabola_triangle_area_l2497_249731


namespace system_of_equations_solution_l2497_249739

theorem system_of_equations_solution (x y : ℚ) : 
  2 * x - 3 * y = 24 ∧ x + 2 * y = 15 → y = 6/7 := by
  sorry

end system_of_equations_solution_l2497_249739


namespace parabola_equation_l2497_249748

/-- A parabola is defined by the equation y = ax^2 + bx + c where a, b, and c are real numbers and a ≠ 0 -/
def Parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- A parabola opens upwards if a > 0 -/
def OpensUpwards (a b c : ℝ) : Prop := a > 0

/-- A parabola intersects the y-axis at the point (0, y) if y = c -/
def IntersectsYAxisAt (a b c y : ℝ) : Prop := c = y

theorem parabola_equation : ∃ (a b : ℝ), 
  OpensUpwards a b 2 ∧ 
  IntersectsYAxisAt a b 2 2 ∧ 
  (∀ x, Parabola a b 2 x = x^2 + 2) := by sorry

end parabola_equation_l2497_249748


namespace quadratic_inequality_boundary_l2497_249701

theorem quadratic_inequality_boundary (c : ℝ) : 
  (∀ x : ℝ, x * (3 * x + 1) < c ↔ -5/2 < x ∧ x < 3) ↔ c = 30 := by
  sorry

end quadratic_inequality_boundary_l2497_249701


namespace parallelogram_height_l2497_249794

/-- Given a parallelogram with area 576 square cm and base 12 cm, its height is 48 cm. -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 576 ∧ base = 12 ∧ area = base * height → height = 48 := by
  sorry

end parallelogram_height_l2497_249794


namespace f_2_equals_216_l2497_249766

def f (x : ℝ) : ℝ := x^5 + 2*x^4 + 3*x^3 + 4*x^2 + 5*x + 6

theorem f_2_equals_216 : f 2 = 216 := by
  sorry

end f_2_equals_216_l2497_249766


namespace f_expression_l2497_249713

/-- A linear function f satisfying specific conditions -/
def f (x : ℝ) : ℝ := sorry

/-- f is a linear function -/
axiom f_linear : ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- f(-2) = -1 -/
axiom f_neg_two : f (-2) = -1

/-- f(0) + f(2) = 10 -/
axiom f_sum : f 0 + f 2 = 10

/-- Theorem: f(x) = 2x + 3 -/
theorem f_expression : ∀ x, f x = 2 * x + 3 := by sorry

end f_expression_l2497_249713


namespace equation_solution_l2497_249752

theorem equation_solution : 
  ∃ x : ℚ, (2 * x + 1) / 3 - (x - 1) / 6 = 2 ∧ x = 3 := by
sorry

end equation_solution_l2497_249752


namespace three_digit_geometric_progression_exists_l2497_249715

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Checks if three numbers form a geometric progression -/
def is_geometric_progression (a b c : Nat) : Prop :=
  b * b = a * c

/-- Converts a ThreeDigitNumber to its decimal representation -/
def to_decimal (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- The main theorem statement -/
theorem three_digit_geometric_progression_exists : ∃! (n : ThreeDigitNumber),
  (is_geometric_progression n.1 n.2.1 n.2.2) ∧
  (to_decimal (n.2.2, n.2.1, n.1) = to_decimal n - 594) ∧
  (10 * n.2.2 + n.2.1 = 10 * n.2.1 + n.2.2 - 18) ∧
  (to_decimal n = 842) := by
  sorry

end three_digit_geometric_progression_exists_l2497_249715


namespace initial_average_production_l2497_249714

theorem initial_average_production (n : ℕ) (A : ℝ) (today_production : ℝ) (new_average : ℝ)
  (h1 : n = 5)
  (h2 : today_production = 90)
  (h3 : new_average = 65)
  (h4 : (n * A + today_production) / (n + 1) = new_average) :
  A = 60 := by
  sorry

end initial_average_production_l2497_249714


namespace minimize_m_l2497_249722

theorem minimize_m (x y : ℝ) :
  let m := 4 * x^2 - 12 * x * y + 10 * y^2 + 4 * y + 9
  ∀ a b : ℝ, m ≤ (4 * a^2 - 12 * a * b + 10 * b^2 + 4 * b + 9) ∧
  m = 5 ∧ x = -3 ∧ y = -2 := by
  sorry

end minimize_m_l2497_249722


namespace b_squared_is_zero_matrix_l2497_249789

theorem b_squared_is_zero_matrix (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 2 = 0 := by
  sorry

end b_squared_is_zero_matrix_l2497_249789


namespace smallest_n_for_125_l2497_249702

/-- The sequence term defined as a function of n -/
def a (n : ℕ) : ℤ := 2 * n^2 - 3

/-- The proposition that 8 is the smallest positive integer n for which a(n) = 125 -/
theorem smallest_n_for_125 : ∀ n : ℕ, n > 0 → a n = 125 → n ≥ 8 :=
sorry

end smallest_n_for_125_l2497_249702


namespace fifth_odd_multiple_of_five_under_hundred_fifth_odd_multiple_of_five_under_hundred_proof_l2497_249779

theorem fifth_odd_multiple_of_five_under_hundred : ℕ → Prop :=
  fun n =>
    (∃ k, n = 5 * (2 * k + 1)) ∧  -- n is odd and a multiple of 5
    n < 100 ∧  -- n is less than 100
    (∃ m, m = 5 ∧  -- m is the count of numbers satisfying the conditions
      ∀ i, i < n →
        (∃ j, i = 5 * (2 * j + 1)) ∧ i < 100 →
        i ≤ m * 9) →  -- there are exactly 4 numbers before n satisfying the conditions
    n = 45  -- the fifth such number is 45

-- The proof of this theorem is omitted
theorem fifth_odd_multiple_of_five_under_hundred_proof : fifth_odd_multiple_of_five_under_hundred 45 := by
  sorry

end fifth_odd_multiple_of_five_under_hundred_fifth_odd_multiple_of_five_under_hundred_proof_l2497_249779


namespace bertrand_odd_conjecture_counterexample_l2497_249744

-- Define what we mean by a "large" number
def isLarge (n : ℕ) : Prop := n ≥ 100

-- Define an odd number
def isOdd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

-- Define a prime number
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ d, d ∣ p → d = 1 ∨ d = p

-- Bertrand's Odd Conjecture
def bertrandOddConjecture : Prop := 
  ∀ n, isLarge n → isOdd n → 
    ∃ p q r, isPrime p ∧ isPrime q ∧ isPrime r ∧ 
             isOdd p ∧ isOdd q ∧ isOdd r ∧
             n = p + q + r

-- Theorem: There exists a counterexample to Bertrand's Odd Conjecture
theorem bertrand_odd_conjecture_counterexample :
  ∃ n, isLarge n ∧ isOdd n ∧ 
    ¬(∃ p q r, isPrime p ∧ isPrime q ∧ isPrime r ∧ 
               isOdd p ∧ isOdd q ∧ isOdd r ∧
               n = p + q + r) :=
by sorry

end bertrand_odd_conjecture_counterexample_l2497_249744


namespace arithmetic_sequence_common_difference_l2497_249736

theorem arithmetic_sequence_common_difference 
  (n : ℕ) 
  (total_sum : ℝ) 
  (even_sum : ℝ) 
  (h1 : n = 20) 
  (h2 : total_sum = 75) 
  (h3 : even_sum = 25) : 
  (even_sum - (total_sum - even_sum)) / 10 = -2.5 := by
sorry

end arithmetic_sequence_common_difference_l2497_249736


namespace n_value_equality_l2497_249718

theorem n_value_equality (n : ℕ) : 3 * (Nat.choose (n - 3) (n - 7)) = 5 * (Nat.factorial (n - 4) / Nat.factorial (n - 6)) → n = 11 := by
  sorry

end n_value_equality_l2497_249718


namespace smaller_solution_quadratic_equation_l2497_249796

theorem smaller_solution_quadratic_equation :
  ∃ x : ℝ, x^2 + 10*x - 40 = 0 ∧ 
  (∀ y : ℝ, y^2 + 10*y - 40 = 0 → x ≤ y) ∧
  x = -8 := by
sorry

end smaller_solution_quadratic_equation_l2497_249796


namespace first_day_exceeding_target_day_exceeding_target_is_tuesday_l2497_249799

/-- Geometric sequence sum function -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

/-- First day of deposit (Sunday) -/
def initialDay : ℕ := 0

/-- Initial deposit amount in cents -/
def initialDeposit : ℚ := 3

/-- Daily deposit multiplier -/
def dailyMultiplier : ℚ := 2

/-- Target amount in cents -/
def targetAmount : ℚ := 2000

/-- Function to calculate the day of the week -/
def dayOfWeek (n : ℕ) : ℕ :=
  (initialDay + n) % 7

/-- Theorem: The 10th deposit day is the first to exceed the target amount -/
theorem first_day_exceeding_target :
  (∀ k < 10, geometricSum initialDeposit dailyMultiplier k ≤ targetAmount) ∧
  geometricSum initialDeposit dailyMultiplier 10 > targetAmount :=
sorry

/-- Corollary: The day when the total first exceeds the target is Tuesday -/
theorem day_exceeding_target_is_tuesday :
  dayOfWeek 10 = 2 :=
sorry

end first_day_exceeding_target_day_exceeding_target_is_tuesday_l2497_249799


namespace linear_equation_power_l2497_249720

/-- If $2x^{n-3}-\frac{1}{3}y^{2m+1}=0$ is a linear equation in $x$ and $y$, then $n^m = 1$. -/
theorem linear_equation_power (n m : ℕ) :
  (∀ x y : ℝ, ∃ a b c : ℝ, 2 * x^(n-3) - (1/3) * y^(2*m+1) = a * x + b * y + c) →
  n^m = 1 := by
  sorry

end linear_equation_power_l2497_249720


namespace circle_point_distance_relation_l2497_249725

/-- Given a circle with radius r and a point F constructed as described in the problem,
    prove the relationship between distances u and v from F to specific lines. -/
theorem circle_point_distance_relation (r u v : ℝ) : v^2 = u^3 / (2*r - u) := by
  sorry

end circle_point_distance_relation_l2497_249725


namespace six_digit_numbers_with_zero_count_l2497_249765

/-- The number of digits in the numbers we're considering -/
def num_digits : ℕ := 6

/-- The total number of 6-digit numbers -/
def total_six_digit_numbers : ℕ := 9 * 10^5

/-- The number of 6-digit numbers with no zeros -/
def six_digit_numbers_no_zero : ℕ := 9^6

/-- The number of 6-digit numbers with at least one zero -/
def six_digit_numbers_with_zero : ℕ := total_six_digit_numbers - six_digit_numbers_no_zero

theorem six_digit_numbers_with_zero_count :
  six_digit_numbers_with_zero = 368559 := by
  sorry

end six_digit_numbers_with_zero_count_l2497_249765


namespace f_properties_l2497_249791

-- Define the function f(x) = (x-2)(x+4)
def f (x : ℝ) : ℝ := (x - 2) * (x + 4)

-- Theorem statement
theorem f_properties :
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f x > f y) ∧ 
  (∀ x y, -1 ≤ x ∧ x < y → f x < f y) ∧
  (∀ x ∈ Set.Icc (-2) 2, f x ≥ -9) ∧
  (∀ x ∈ Set.Icc (-2) 2, f x ≤ 0) ∧
  (∃ x ∈ Set.Icc (-2) 2, f x = -9) ∧
  (∃ x ∈ Set.Icc (-2) 2, f x = 0) := by
  sorry


end f_properties_l2497_249791


namespace complex_fraction_simplification_l2497_249769

theorem complex_fraction_simplification :
  ∀ (i : ℂ), i^2 = -1 →
  (11 - 3*i) / (1 + 2*i) = 3 - 5*i := by
sorry

end complex_fraction_simplification_l2497_249769


namespace configurations_formula_l2497_249740

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def num_configurations (n : ℕ) : ℕ :=
  factorial (n * (n + 1) / 2) /
  (List.range n).foldl (λ acc i => acc * factorial (n - i)) 1

theorem configurations_formula (n : ℕ) :
  num_configurations n = factorial (n * (n + 1) / 2) /
    (List.range n).foldl (λ acc i => acc * factorial (n - i)) 1 :=
by sorry

end configurations_formula_l2497_249740


namespace joan_remaining_apples_l2497_249751

/-- Given that Joan picked 43 apples and gave 27 to Melanie, prove that she now has 16 apples. -/
theorem joan_remaining_apples : 
  ∀ (initial_apples given_apples remaining_apples : ℕ), 
    initial_apples = 43 → 
    given_apples = 27 → 
    remaining_apples = initial_apples - given_apples → 
    remaining_apples = 16 := by
  sorry

end joan_remaining_apples_l2497_249751


namespace crate_height_determination_l2497_249749

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical gas tank -/
structure GasTank where
  radius : ℝ
  height : ℝ

/-- Checks if a gas tank fits inside a crate -/
def tankFitsInCrate (tank : GasTank) (crate : CrateDimensions) : Prop :=
  2 * tank.radius ≤ min crate.length (min crate.width crate.height) ∧
  tank.height ≤ max crate.length (max crate.width crate.height)

theorem crate_height_determination
  (crate : CrateDimensions)
  (tank : GasTank)
  (h_crate_dims : crate.length = 6 ∧ crate.width = 8)
  (h_tank_radius : tank.radius = 4)
  (h_tank_fits : tankFitsInCrate tank crate)
  (h_max_volume : ∀ (other_tank : GasTank),
    tankFitsInCrate other_tank crate →
    tank.radius * tank.radius * tank.height ≥ other_tank.radius * other_tank.radius * other_tank.height) :
  crate.height = 6 :=
sorry

end crate_height_determination_l2497_249749


namespace smallest_square_side_is_14_l2497_249798

/-- The smallest side length of a square composed of equal numbers of unit squares with sides 1, 2, and 3 -/
def smallest_square_side : ℕ := 14

/-- Proposition: The smallest possible side length of a square composed of an equal number of squares with sides 1, 2, and 3 is 14 units -/
theorem smallest_square_side_is_14 :
  ∀ n : ℕ, n > 0 →
  ∃ s : ℕ, s * s = n * (1 * 1 + 2 * 2 + 3 * 3) →
  s ≥ smallest_square_side :=
sorry

end smallest_square_side_is_14_l2497_249798


namespace slopes_equal_implies_parallel_false_l2497_249707

/-- A line in a 2D plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Two lines are parallel if they have the same slope --/
def parallel (l1 l2 : Line) : Prop := l1.slope = l2.slope

/-- Statement: If two lines have the same slope, they are parallel --/
theorem slopes_equal_implies_parallel_false :
  ¬ (∀ (l1 l2 : Line), l1.slope = l2.slope → parallel l1 l2) := by
  sorry

end slopes_equal_implies_parallel_false_l2497_249707


namespace water_bottle_count_l2497_249780

theorem water_bottle_count (initial bottles_drunk bottles_bought : ℕ) 
  (h1 : initial = 42)
  (h2 : bottles_drunk = 25)
  (h3 : bottles_bought = 30) : 
  initial - bottles_drunk + bottles_bought = 47 := by
  sorry

end water_bottle_count_l2497_249780


namespace no_integer_tangent_length_l2497_249750

theorem no_integer_tangent_length (t : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ 2 * π * r = 8 * π) →  -- Circle with circumference 8π
  (t^2 = (8*π/3) * π) →                    -- Tangent-secant relationship
  ¬(∃ (n : ℤ), t = n) :=                   -- No integer solution for t
by sorry

end no_integer_tangent_length_l2497_249750


namespace triangle_properties_l2497_249728

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that under certain conditions, angle C and the sum of sines of A and B
    have specific values. -/
theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  c^2 = a^2 + b^2 + a*b →
  c = 4 * Real.sqrt 7 →
  a + b + c = 12 + 4 * Real.sqrt 7 →
  C = 2 * Real.pi / 3 ∧
  Real.sin A + Real.sin B = 3 * Real.sqrt 21 / 28 := by
  sorry

end triangle_properties_l2497_249728


namespace sector_area_l2497_249782

/-- Given a circular sector with central angle α = 60° and arc length l = 6π,
    prove that the area of the sector is 54π. -/
theorem sector_area (α : Real) (l : Real) (h1 : α = 60 * π / 180) (h2 : l = 6 * π) :
  (1 / 2) * l * (l / α) = 54 * π := by
  sorry

end sector_area_l2497_249782


namespace find_a_l2497_249797

theorem find_a (x y : ℝ) (h1 : x = 1) (h2 : y = 2) (h3 : ∃ a : ℝ, a * x - y = 3) : 
  ∃ a : ℝ, a = 5 ∧ a * x - y = 3 := by
sorry

end find_a_l2497_249797


namespace sequence_general_term_l2497_249732

theorem sequence_general_term (a : ℕ+ → ℕ) :
  (a 1 = 1) →
  (∀ n : ℕ+, a (n + 1) = a n + 2 * n) →
  (∀ n : ℕ+, a n = n^2 - n + 1) :=
by sorry

end sequence_general_term_l2497_249732


namespace G_4_g_5_equals_29_l2497_249773

-- Define the functions g and G
def g (x : ℝ) : ℝ := 2 * x - 3
def G (x y : ℝ) : ℝ := x * y + 2 * x - y

-- State the theorem
theorem G_4_g_5_equals_29 : G 4 (g 5) = 29 := by
  sorry

end G_4_g_5_equals_29_l2497_249773


namespace farm_cows_l2497_249724

/-- Represents the number of bags of husk eaten by some cows in 45 days -/
def total_bags : ℕ := 45

/-- Represents the number of bags of husk eaten by one cow in 45 days -/
def bags_per_cow : ℕ := 1

/-- Calculates the number of cows on the farm -/
def num_cows : ℕ := total_bags / bags_per_cow

/-- Proves that the number of cows on the farm is 45 -/
theorem farm_cows : num_cows = 45 := by
  sorry

end farm_cows_l2497_249724


namespace kho_kho_only_count_l2497_249742

/-- The number of people who play kho kho only -/
def kho_kho_only : ℕ := 30

/-- The total number of players -/
def total_players : ℕ := 40

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 10

/-- The number of people who play both kabadi and kho kho -/
def both_players : ℕ := 5

/-- Theorem stating that the number of people who play kho kho only is 30 -/
theorem kho_kho_only_count :
  kho_kho_only = total_players - kabadi_players + both_players :=
by sorry

end kho_kho_only_count_l2497_249742


namespace f_odd_and_decreasing_l2497_249730

-- Define the function f(x) = -x|x|
def f (x : ℝ) : ℝ := -x * abs x

-- Theorem stating that f is both odd and decreasing
theorem f_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f y < f x) :=
sorry

end f_odd_and_decreasing_l2497_249730


namespace inequality_solution_set_l2497_249719

theorem inequality_solution_set (k : ℝ) : 
  (∀ x : ℝ, (1 + k / (x - 1) ≤ 0 ↔ x ∈ Set.Ici (-2) ∩ Set.Iio 1)) → k = 3 := by
  sorry

end inequality_solution_set_l2497_249719


namespace circleplus_two_three_l2497_249712

-- Define the operation ⊕
def circleplus (a b : ℝ) : ℝ := 4 * a + 5 * b

-- State the theorem
theorem circleplus_two_three : circleplus 2 3 = 23 := by
  sorry

end circleplus_two_three_l2497_249712


namespace janessa_keeps_twenty_cards_l2497_249774

/-- The number of cards Janessa keeps for herself --/
def cards_kept_by_janessa (initial_cards : ℕ) (cards_from_father : ℕ) (cards_ordered : ℕ) 
  (bad_cards : ℕ) (cards_given_to_dexter : ℕ) : ℕ :=
  initial_cards + cards_from_father + cards_ordered - bad_cards - cards_given_to_dexter

/-- Theorem stating that Janessa keeps 20 cards for herself --/
theorem janessa_keeps_twenty_cards : 
  cards_kept_by_janessa 4 13 36 4 29 = 20 := by
  sorry

end janessa_keeps_twenty_cards_l2497_249774


namespace quadruple_equation_solution_l2497_249772

def is_valid_quadruple (a b c d : ℕ) : Prop :=
  a + b = c * d ∧ c + d = a * b

def solution_set : Set (ℕ × ℕ × ℕ × ℕ) :=
  {(2, 2, 2, 2), (1, 2, 3, 5), (2, 1, 3, 5), (1, 2, 5, 3), (2, 1, 5, 3),
   (3, 5, 1, 2), (5, 3, 1, 2), (3, 5, 2, 1), (5, 3, 2, 1)}

theorem quadruple_equation_solution :
  {q : ℕ × ℕ × ℕ × ℕ | is_valid_quadruple q.1 q.2.1 q.2.2.1 q.2.2.2} = solution_set :=
by sorry

end quadruple_equation_solution_l2497_249772


namespace samantha_birth_year_l2497_249781

-- Define the year of the first AMC 8
def first_amc_year : ℕ := 1980

-- Define the function to calculate the year of the nth AMC 8
def amc_year (n : ℕ) : ℕ := first_amc_year + n - 1

-- Define Samantha's age when she took the 9th AMC 8
def samantha_age_at_ninth_amc : ℕ := 14

-- Theorem to prove Samantha's birth year
theorem samantha_birth_year :
  amc_year 9 - samantha_age_at_ninth_amc = 1974 := by
  sorry


end samantha_birth_year_l2497_249781


namespace greatest_sum_consecutive_integers_l2497_249764

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500 ∧ 
   ∀ m : ℕ, m > n → m * (m + 1) ≥ 500) → 
  n + (n + 1) = 43 := by
sorry

end greatest_sum_consecutive_integers_l2497_249764


namespace corridor_length_is_95_meters_l2497_249723

/-- Represents the scale of a blueprint in meters per centimeter. -/
def blueprint_scale : ℝ := 10

/-- Represents the length of the corridor in the blueprint in centimeters. -/
def blueprint_corridor_length : ℝ := 9.5

/-- Calculates the real-life length of the corridor in meters. -/
def real_life_corridor_length : ℝ := blueprint_scale * blueprint_corridor_length

/-- Theorem stating that the real-life length of the corridor is 95 meters. -/
theorem corridor_length_is_95_meters : real_life_corridor_length = 95 := by
  sorry

end corridor_length_is_95_meters_l2497_249723


namespace circle_equation_solution_l2497_249708

theorem circle_equation_solution :
  ∃! (x y : ℝ), (x - 11)^2 + (y - 12)^2 + (x - y)^2 = 1/3 ∧ x = 11 + 1/3 ∧ y = 11 + 2/3 := by
  sorry

end circle_equation_solution_l2497_249708


namespace function_maximum_l2497_249729

/-- The function f(x) = x + 4/x for x < 0 has a maximum value of -4 -/
theorem function_maximum (x : ℝ) (h : x < 0) : 
  x + 4 / x ≤ -4 :=
sorry

end function_maximum_l2497_249729


namespace f_minimum_at_neg_two_l2497_249790

/-- The function f(x) = |x+1| + |x+2| + |x+3| -/
def f (x : ℝ) : ℝ := |x + 1| + |x + 2| + |x + 3|

theorem f_minimum_at_neg_two :
  f (-2) = 2 ∧ ∀ x : ℝ, f x ≥ 2 := by
  sorry

end f_minimum_at_neg_two_l2497_249790


namespace carrot_harvest_calculation_l2497_249721

/-- Calculates the expected carrot harvest from a rectangular backyard --/
theorem carrot_harvest_calculation 
  (length_paces width_paces : ℕ) 
  (pace_to_feet : ℝ) 
  (carrot_yield_per_sqft : ℝ) : 
  length_paces = 25 → 
  width_paces = 30 → 
  pace_to_feet = 2.5 → 
  carrot_yield_per_sqft = 0.5 → 
  (length_paces : ℝ) * pace_to_feet * (width_paces : ℝ) * pace_to_feet * carrot_yield_per_sqft = 2343.75 := by
  sorry

#check carrot_harvest_calculation

end carrot_harvest_calculation_l2497_249721


namespace parallel_line_through_point_l2497_249703

-- Define a line in slope-intercept form (y = mx + b)
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.m * x + l.b

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.m = l2.m

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through a point
def passes_through (l : Line) (p : Point) : Prop :=
  l.equation p.x p.y

-- The given line
def given_line : Line :=
  { m := -2, b := 3 }

-- The point (0, 1)
def point : Point :=
  { x := 0, y := 1 }

-- The theorem to prove
theorem parallel_line_through_point :
  ∃! l : Line, parallel l given_line ∧ passes_through l point ∧ l.equation 0 1 :=
sorry

end parallel_line_through_point_l2497_249703


namespace arithmetic_sequence_product_divisibility_l2497_249705

/-- Given three numbers in an arithmetic sequence with common difference d,
    where one of the numbers is divisible by d, their product is divisible by 6d³ -/
theorem arithmetic_sequence_product_divisibility
  (a b c d : ℤ) -- a, b, c are the three numbers, d is the common difference
  (h_arithmetic : b - a = d ∧ c - b = d) -- arithmetic sequence condition
  (h_divisible : a % d = 0 ∨ b % d = 0 ∨ c % d = 0) -- one number divisible by d
  : (6 * d^3) ∣ (a * b * c) := by
  sorry

end arithmetic_sequence_product_divisibility_l2497_249705


namespace cubic_one_real_root_l2497_249726

theorem cubic_one_real_root (c : ℝ) :
  ∃! x : ℝ, x^3 - 4*x^2 + 9*x + c = 0 :=
by
  sorry


end cubic_one_real_root_l2497_249726


namespace product_simplification_l2497_249733

theorem product_simplification (x : ℝ) (hx : x ≠ 0) :
  (10 * x^3) * (8 * x^2) * (1 / (4*x)^3) = (5/4) * x^2 := by
  sorry

end product_simplification_l2497_249733


namespace trigonometric_product_equals_one_l2497_249738

theorem trigonometric_product_equals_one :
  let x : Real := 40 * π / 180
  let y : Real := 50 * π / 180
  (1 - 1 / Real.cos x) * (1 + 1 / Real.sin y) * (1 - 1 / Real.sin x) * (1 + 1 / Real.cos y) = 1 := by
  sorry

end trigonometric_product_equals_one_l2497_249738


namespace poly_factorable_iff_l2497_249768

/-- A polynomial in x and y with a parameter k -/
def poly (k : ℤ) (x y : ℤ) : ℤ := x^2 + 4*x*y + x + k*y - k

/-- A linear factor with integer coefficients -/
structure LinearFactor where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Check if a polynomial can be factored into two linear factors -/
def isFactorable (k : ℤ) : Prop :=
  ∃ (f g : LinearFactor), ∀ (x y : ℤ),
    poly k x y = (f.a * x + f.b * y + f.c) * (g.a * x + g.b * y + g.c)

/-- The main theorem: the polynomial is factorable iff k = 0 or k = 16 -/
theorem poly_factorable_iff (k : ℤ) : isFactorable k ↔ k = 0 ∨ k = 16 :=
sorry

end poly_factorable_iff_l2497_249768


namespace ellipse_eccentricity_l2497_249795

/-- The eccentricity of an ellipse with equation x²/16 + y²/12 = 1 is 1/2 -/
theorem ellipse_eccentricity : ∃ e : ℝ,
  (∀ x y : ℝ, x^2/16 + y^2/12 = 1 → 
    ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧
      x^2/a^2 + y^2/b^2 = 1 ∧
      c^2 = a^2 - b^2 ∧
      e = c/a) ∧
  e = 1/2 := by
  sorry

end ellipse_eccentricity_l2497_249795


namespace first_equation_value_l2497_249711

theorem first_equation_value (x y a : ℝ) 
  (eq1 : 2 * x + y = a) 
  (eq2 : x + 2 * y = 10) 
  (eq3 : (x + y) / 3 = 4) : 
  a = 12 := by
sorry

end first_equation_value_l2497_249711


namespace triangle_area_ratio_l2497_249704

theorem triangle_area_ratio (K J : ℝ) (x : ℝ) (h_positive : 0 < x) (h_less_than_one : x < 1)
  (h_ratio : J / K = x) : x = 1 / 3 := by
  sorry

end triangle_area_ratio_l2497_249704


namespace symmetry_properties_l2497_249710

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Symmetry with respect to the x-axis. -/
def symmetryOX (p : Point2D) : Point2D :=
  ⟨p.x, -p.y⟩

/-- Symmetry with respect to the y-axis. -/
def symmetryOY (p : Point2D) : Point2D :=
  ⟨-p.x, p.y⟩

theorem symmetry_properties (p : Point2D) : 
  (symmetryOX p = ⟨p.x, -p.y⟩) ∧ (symmetryOY p = ⟨-p.x, p.y⟩) := by
  sorry

end symmetry_properties_l2497_249710


namespace number_difference_l2497_249746

theorem number_difference (L S : ℕ) (h1 : L = 1584) (h2 : L = 6 * S + 15) : L - S = 1323 := by
  sorry

end number_difference_l2497_249746


namespace goose_eggs_count_l2497_249784

theorem goose_eggs_count (
  total_eggs : ℕ
) (
  hatched_ratio : Real
) (
  first_month_survival_ratio : Real
) (
  first_year_death_ratio : Real
) (
  first_year_survivors : ℕ
) (
  h1 : hatched_ratio = 1 / 4
) (
  h2 : first_month_survival_ratio = 4 / 5
) (
  h3 : first_year_death_ratio = 2 / 5
) (
  h4 : first_year_survivors = 120
) (
  h5 : (hatched_ratio * first_month_survival_ratio * (1 - first_year_death_ratio) * total_eggs : Real) = first_year_survivors
) : total_eggs = 800 := by
  sorry

end goose_eggs_count_l2497_249784


namespace two_small_triangles_exist_l2497_249716

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of a triangle -/
def area (t : Triangle) : ℝ :=
  sorry

/-- Check if a point is inside a triangle -/
def isInside (p : Point) (t : Triangle) : Prop :=
  sorry

/-- The unit triangle -/
def unitTriangle : Triangle :=
  sorry

/-- Theorem: Given 5 points in a unit triangle, there exist at least two distinct
    triangles formed by these points, each with an area not exceeding 1/4 -/
theorem two_small_triangles_exist (points : Finset Point)
    (h1 : points.card = 5)
    (h2 : ∀ p ∈ points, isInside p unitTriangle) :
    ∃ t1 t2 : Triangle,
      t1.a ∈ points ∧ t1.b ∈ points ∧ t1.c ∈ points ∧
      t2.a ∈ points ∧ t2.b ∈ points ∧ t2.c ∈ points ∧
      t1 ≠ t2 ∧
      area t1 ≤ 1/4 ∧ area t2 ≤ 1/4 :=
  sorry

end two_small_triangles_exist_l2497_249716


namespace ticket_price_uniqueness_l2497_249787

theorem ticket_price_uniqueness : ∃! x : ℕ+, 
  (x : ℕ) ∣ 72 ∧ 
  (x : ℕ) ∣ 90 ∧ 
  1 ≤ 72 / (x : ℕ) ∧ 72 / (x : ℕ) ≤ 10 ∧
  1 ≤ 90 / (x : ℕ) ∧ 90 / (x : ℕ) ≤ 10 :=
by sorry

end ticket_price_uniqueness_l2497_249787


namespace sticker_cost_l2497_249760

theorem sticker_cost (num_packs : ℕ) (stickers_per_pack : ℕ) (james_payment : ℚ) :
  num_packs = 4 →
  stickers_per_pack = 30 →
  james_payment = 6 →
  (2 * james_payment) / (num_packs * stickers_per_pack : ℚ) = 0.1 := by
  sorry

end sticker_cost_l2497_249760


namespace root_sum_theorem_l2497_249763

-- Define the equation for which a and b are roots
def root_equation (m x : ℝ) : Prop :=
  m * (x^2 - 2*x) + 3*x + 7 = 0

-- Define the condition for m₁ and m₂
def m_condition (m : ℝ) (a b : ℝ) : Prop :=
  a / b + b / a = 7 / 10

theorem root_sum_theorem :
  ∀ (m₁ m₂ a b : ℝ),
  (∃ m, root_equation m a ∧ root_equation m b) →
  m_condition m₁ a b →
  m_condition m₂ a b →
  m₁ / m₂ + m₂ / m₁ = 253 / 36 := by
sorry

end root_sum_theorem_l2497_249763


namespace abs_inequality_solution_l2497_249754

theorem abs_inequality_solution (x : ℝ) : 
  abs (x + 3) + abs (2 * x - 1) < 7 ↔ -3 ≤ x ∧ x < 5/3 := by
  sorry

end abs_inequality_solution_l2497_249754


namespace arithmetic_sequence_sum_l2497_249775

theorem arithmetic_sequence_sum : 
  ∀ (a d n : ℕ) (last : ℕ),
    a = 2 →
    d = 2 →
    last = 20 →
    last = a + (n - 1) * d →
    (n : ℕ) * (a + last) / 2 = 110 :=
by
  sorry

end arithmetic_sequence_sum_l2497_249775


namespace bicycle_trip_average_speed_l2497_249778

/-- Proves that for a bicycle trip with two parts:
    1. 10 km at 12 km/hr
    2. 12 km at 10 km/hr
    The average speed for the entire trip is 660/61 km/hr. -/
theorem bicycle_trip_average_speed :
  let distance1 : ℝ := 10
  let speed1 : ℝ := 12
  let distance2 : ℝ := 12
  let speed2 : ℝ := 10
  let total_distance : ℝ := distance1 + distance2
  let total_time : ℝ := distance1 / speed1 + distance2 / speed2
  let average_speed : ℝ := total_distance / total_time
  average_speed = 660 / 61 := by
sorry

end bicycle_trip_average_speed_l2497_249778


namespace total_pens_after_changes_l2497_249786

-- Define the initial number of pens
def initial_red : ℕ := 65
def initial_blue : ℕ := 45
def initial_black : ℕ := 58
def initial_green : ℕ := 36
def initial_purple : ℕ := 27

-- Define the changes in pen quantities
def red_decrease : ℕ := 15
def blue_decrease : ℕ := 20
def black_increase : ℕ := 12
def green_decrease : ℕ := 10
def purple_increase : ℕ := 5

-- Define the theorem
theorem total_pens_after_changes : 
  (initial_red - red_decrease) + 
  (initial_blue - blue_decrease) + 
  (initial_black + black_increase) + 
  (initial_green - green_decrease) + 
  (initial_purple + purple_increase) = 203 := by
  sorry

end total_pens_after_changes_l2497_249786


namespace amy_chocolate_bars_l2497_249706

/-- The number of chocolate bars Amy has -/
def chocolate_bars : ℕ := sorry

/-- The number of M&Ms Amy has -/
def m_and_ms : ℕ := 7 * chocolate_bars

/-- The number of marshmallows Amy has -/
def marshmallows : ℕ := 6 * m_and_ms

/-- The total number of candies Amy has -/
def total_candies : ℕ := chocolate_bars + m_and_ms + marshmallows

/-- The number of baskets Amy fills -/
def num_baskets : ℕ := 25

/-- The number of candies in each basket -/
def candies_per_basket : ℕ := 10

theorem amy_chocolate_bars : 
  chocolate_bars = 5 ∧ 
  total_candies = num_baskets * candies_per_basket := by
  sorry

end amy_chocolate_bars_l2497_249706


namespace simplify_expression_l2497_249755

theorem simplify_expression (x y : ℝ) : (x - 3*y + 2) * (x + 3*y + 2) = x^2 + 4*x + 4 - 9*y^2 := by
  sorry

end simplify_expression_l2497_249755


namespace book_page_digit_sum_l2497_249771

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- Sum of digits for all page numbers from 1 to n -/
def total_digit_sum (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of all digits in page numbers of a 2000-page book is 28002 -/
theorem book_page_digit_sum :
  total_digit_sum 2000 = 28002 := by sorry

end book_page_digit_sum_l2497_249771


namespace sphere_packing_radius_l2497_249741

/-- A structure representing a sphere packing in a cube -/
structure SpherePacking where
  cube_side_length : ℝ
  num_spheres : ℕ
  sphere_radius : ℝ
  is_valid : Prop

/-- The theorem stating the radius of spheres in the given packing configuration -/
theorem sphere_packing_radius (packing : SpherePacking) : 
  packing.cube_side_length = 2 ∧ 
  packing.num_spheres = 10 ∧ 
  packing.is_valid →
  packing.sphere_radius = 0.5 :=
sorry

end sphere_packing_radius_l2497_249741


namespace product_of_three_numbers_l2497_249792

theorem product_of_three_numbers (a b c : ℝ) : 
  a + b + c = 300 ∧ 
  9 * a = b - 11 ∧ 
  9 * a = c + 15 ∧ 
  a ≤ b ∧ 
  a ≤ c ∧ 
  c ≤ b → 
  a * b * c = 319760 := by
sorry

end product_of_three_numbers_l2497_249792


namespace parabola_vertex_l2497_249759

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 6*y + 4*x - 7 = 0

/-- The vertex of a parabola -/
def is_vertex (x y : ℝ) : Prop :=
  ∀ x' y', parabola_equation x' y' → y' ≥ y

theorem parabola_vertex :
  is_vertex 4 (-3) :=
sorry

end parabola_vertex_l2497_249759


namespace inequality_and_equality_condition_l2497_249709

theorem inequality_and_equality_condition (x y : ℝ) 
  (hx : x > -1) (hy : y > -1) (hsum : x + y = 1) :
  (x / (y + 1) + y / (x + 1) ≥ 2 / 3) ∧
  (x / (y + 1) + y / (x + 1) = 2 / 3 ↔ x = 1 / 2 ∧ y = 1 / 2) := by
  sorry

end inequality_and_equality_condition_l2497_249709


namespace range_of_x_for_inequality_l2497_249737

-- Define the function f(x, a)
def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- State the theorem
theorem range_of_x_for_inequality (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  {x : ℝ | ∀ a ∈ Set.Icc (-1) 1, f x a > 0} = Set.Iio 1 ∪ Set.Ioi 3 :=
by sorry

end range_of_x_for_inequality_l2497_249737


namespace triangle_side_length_l2497_249734

theorem triangle_side_length 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (h_area : (1/2) * a * c * Real.sin B = Real.sqrt 3)
  (h_angle : B = Real.pi / 3)
  (h_sides : a^2 + c^2 = 3*a*c) :
  b = 2 * Real.sqrt 2 := by sorry

end triangle_side_length_l2497_249734


namespace income_calculation_l2497_249777

/-- Proves that given an income to expenditure ratio of 7:6 and savings of 3000,
    the income is 21000. -/
theorem income_calculation (income expenditure savings : ℕ) : 
  income * 6 = expenditure * 7 →
  income - expenditure = savings →
  savings = 3000 →
  income = 21000 := by
sorry

end income_calculation_l2497_249777


namespace ice_cream_shop_sales_l2497_249762

/-- Given a ratio of sugar cones to waffle cones and the number of waffle cones sold,
    calculate the number of sugar cones sold. -/
def sugar_cones_sold (sugar_ratio : ℕ) (waffle_ratio : ℕ) (waffle_cones : ℕ) : ℕ :=
  (sugar_ratio * waffle_cones) / waffle_ratio

/-- Theorem stating that given the specific ratio and number of waffle cones,
    the number of sugar cones sold is 45. -/
theorem ice_cream_shop_sales : sugar_cones_sold 5 4 36 = 45 := by
  sorry

end ice_cream_shop_sales_l2497_249762


namespace m_positive_l2497_249727

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + x^2) - x) + 2 / (2^x + 1) + 1

theorem m_positive (m : ℝ) (h : f (m - 1) + f (1 - 2*m) > 4) : m > 0 := by
  sorry

end m_positive_l2497_249727


namespace problem_solution_l2497_249756

theorem problem_solution (x y : ℝ) (h1 : x + y = 4) (h2 : x - y = 6) :
  2 * x^2 - 2 * y^2 = 48 := by
  sorry

end problem_solution_l2497_249756


namespace triangle_distance_l2497_249783

/-- Given a triangle ABC with the following properties:
  - AB = x meters
  - BC = 3 meters
  - Angle B = 150°
  - Area of triangle ABC = 3√3/4 m²
  Prove that the length of AC is √3 meters. -/
theorem triangle_distance (x : ℝ) : 
  let a := x
  let b := 3
  let c := (a^2 + b^2 - 2*a*b*Real.cos (150 * π / 180))^(1/2)
  let s := 3 * Real.sqrt 3 / 4
  s = 1/2 * a * b * Real.sin (150 * π / 180) →
  c = Real.sqrt 3 := by
  sorry

end triangle_distance_l2497_249783
