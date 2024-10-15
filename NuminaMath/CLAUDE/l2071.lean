import Mathlib

namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2071_207199

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 1 + 3 * a 8 + a 15 = 100) :
  2 * a 9 - a 10 = 20 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2071_207199


namespace NUMINAMATH_CALUDE_difference_given_sum_and_difference_of_squares_l2071_207194

theorem difference_given_sum_and_difference_of_squares (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 := by
  sorry

end NUMINAMATH_CALUDE_difference_given_sum_and_difference_of_squares_l2071_207194


namespace NUMINAMATH_CALUDE_original_number_proof_l2071_207149

theorem original_number_proof : ∃ n : ℕ, n + 1 = 30 ∧ n < 30 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2071_207149


namespace NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l2071_207165

theorem complex_fraction_pure_imaginary (a : ℝ) : 
  (Complex.I * (Complex.I * (a + 1) + (1 - a)) = Complex.I * (a + Complex.I) / (1 + Complex.I)) → 
  a = -1 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_pure_imaginary_l2071_207165


namespace NUMINAMATH_CALUDE_problem_solution_l2071_207102

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x - 1 = 0}

theorem problem_solution :
  (B (1/5) ⊂ A) ∧
  ({a : ℝ | A ∩ B a = B a} = {0, 1/3, 1/5}) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2071_207102


namespace NUMINAMATH_CALUDE_compound_interest_principal_is_5000_l2071_207190

-- Define the simple interest rate
def simple_interest_rate : ℝ := 0.10

-- Define the compound interest rate
def compound_interest_rate : ℝ := 0.12

-- Define the simple interest time period
def simple_interest_time : ℕ := 5

-- Define the compound interest time period
def compound_interest_time : ℕ := 2

-- Define the simple interest principal
def simple_interest_principal : ℝ := 1272

-- Define the function to calculate simple interest
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * (time : ℝ)

-- Define the function to calculate compound interest
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

-- Theorem to prove
theorem compound_interest_principal_is_5000 :
  ∃ (compound_principal : ℝ),
    simple_interest simple_interest_principal simple_interest_rate simple_interest_time =
    (1/2) * compound_interest compound_principal compound_interest_rate compound_interest_time ∧
    compound_principal = 5000 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_principal_is_5000_l2071_207190


namespace NUMINAMATH_CALUDE_power_of_two_problem_l2071_207179

theorem power_of_two_problem (a b : ℕ+) 
  (h1 : (2 ^ a.val) ^ b.val = 2 ^ 2)
  (h2 : 2 ^ a.val * 2 ^ b.val = 8) :
  2 ^ b.val = 4 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_problem_l2071_207179


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_triangle_area_l2071_207135

/-- Given a parallelogram with area 128 square meters, the area of a triangle formed by its diagonal is 64 square meters. -/
theorem parallelogram_diagonal_triangle_area (P : Real) (h : P = 128) : P / 2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_triangle_area_l2071_207135


namespace NUMINAMATH_CALUDE_empty_solution_set_has_solutions_l2071_207171

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |3 - x| < a

-- Theorem 1: The solution set is empty iff a ≤ 1
theorem empty_solution_set (a : ℝ) :
  (∀ x : ℝ, ¬ inequality x a) ↔ a ≤ 1 := by sorry

-- Theorem 2: The inequality has solutions iff a > 1
theorem has_solutions (a : ℝ) :
  (∃ x : ℝ, inequality x a) ↔ a > 1 := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_has_solutions_l2071_207171


namespace NUMINAMATH_CALUDE_cos_sin_expression_in_terms_of_p_q_l2071_207101

open Real

theorem cos_sin_expression_in_terms_of_p_q (x : ℝ) 
  (p : ℝ) (hp : p = (1 - cos x) * (1 + sin x))
  (q : ℝ) (hq : q = (1 + cos x) * (1 - sin x)) :
  cos x ^ 2 - cos x ^ 4 - sin (2 * x) + 2 = p * q - (p + q) := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_expression_in_terms_of_p_q_l2071_207101


namespace NUMINAMATH_CALUDE_f_is_even_iff_l2071_207172

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = ax^2 + (2a+1)x - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + (2*a + 1) * x - 1

/-- Theorem: The function f is even if and only if a = -1/2 -/
theorem f_is_even_iff (a : ℝ) : IsEven (f a) ↔ a = -1/2 := by
  sorry


end NUMINAMATH_CALUDE_f_is_even_iff_l2071_207172


namespace NUMINAMATH_CALUDE_positive_poly_nonneg_ratio_l2071_207145

/-- A polynomial with real coefficients -/
def RealPolynomial := Polynomial ℝ

/-- A polynomial with nonnegative real coefficients -/
def NonnegPolynomial := {p : RealPolynomial // ∀ i, 0 ≤ p.coeff i}

/-- The theorem statement -/
theorem positive_poly_nonneg_ratio
  (P : RealPolynomial)
  (h : ∀ x : ℝ, 0 < x → 0 < P.eval x) :
  ∃ (Q R : NonnegPolynomial), ∀ x : ℝ, 0 < x →
    P.eval x = (Q.val.eval x) / (R.val.eval x) :=
sorry

end NUMINAMATH_CALUDE_positive_poly_nonneg_ratio_l2071_207145


namespace NUMINAMATH_CALUDE_blonde_to_total_ratio_l2071_207137

/-- Given a class with a specific hair color ratio and number of students, 
    prove the ratio of blonde-haired children to total children -/
theorem blonde_to_total_ratio 
  (red_ratio : ℕ) (blonde_ratio : ℕ) (black_ratio : ℕ)
  (red_count : ℕ) (total_count : ℕ)
  (h1 : red_ratio = 3)
  (h2 : blonde_ratio = 6)
  (h3 : black_ratio = 7)
  (h4 : red_count = 9)
  (h5 : total_count = 48)
  : (blonde_ratio * red_count / red_ratio) / total_count = 3 / 8 := by
  sorry

#check blonde_to_total_ratio

end NUMINAMATH_CALUDE_blonde_to_total_ratio_l2071_207137


namespace NUMINAMATH_CALUDE_allocation_schemes_count_l2071_207156

/-- The number of ways to allocate three distinct individuals to seven laboratories,
    where each laboratory can hold at most two people. -/
def allocationSchemes : ℕ := 336

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of permutations of n distinct objects. -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem allocation_schemes_count :
  allocationSchemes = 
    choose 7 3 * factorial 3 + choose 7 2 * choose 3 2 * 2 :=
by sorry

end NUMINAMATH_CALUDE_allocation_schemes_count_l2071_207156


namespace NUMINAMATH_CALUDE_expression_simplification_l2071_207108

theorem expression_simplification (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 - a / (a + 1)) / ((a^2 - 1) / (a^2 + 2*a + 1)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2071_207108


namespace NUMINAMATH_CALUDE_fundraiser_group_composition_l2071_207196

theorem fundraiser_group_composition (p : ℕ) : 
  (∃ (initial_girls : ℕ),
    -- Initial condition: 30% of the group are girls
    initial_girls = (3 * p) / 10 ∧
    -- After changes: 25% of the group are girls
    (initial_girls - 3 : ℚ) / (p + 2) = 1 / 4 →
    -- Prove that the initial number of girls was 21
    initial_girls = 21) :=
by sorry

end NUMINAMATH_CALUDE_fundraiser_group_composition_l2071_207196


namespace NUMINAMATH_CALUDE_class_size_l2071_207105

theorem class_size (football : ℕ) (tennis : ℕ) (both : ℕ) (neither : ℕ)
  (h1 : football = 26)
  (h2 : tennis = 20)
  (h3 : both = 17)
  (h4 : neither = 7) :
  football + tennis - both + neither = 36 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l2071_207105


namespace NUMINAMATH_CALUDE_binomial_square_condition_l2071_207151

/-- If 9x^2 - 18x + a is the square of a binomial, then a = 9 -/
theorem binomial_square_condition (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 9*x^2 - 18*x + a = (3*x + b)^2) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_condition_l2071_207151


namespace NUMINAMATH_CALUDE_darnel_running_results_l2071_207125

/-- Represents Darnel's running activities --/
structure RunningActivity where
  sprint1 : Real
  sprint2 : Real
  jog1 : Real
  jog2 : Real
  walk : Real

/-- Calculates the total distance covered in all activities --/
def totalDistance (activity : RunningActivity) : Real :=
  activity.sprint1 + activity.sprint2 + activity.jog1 + activity.jog2 + activity.walk

/-- Calculates the additional distance sprinted compared to jogging and walking --/
def additionalSprint (activity : RunningActivity) : Real :=
  (activity.sprint1 + activity.sprint2) - (activity.jog1 + activity.jog2 + activity.walk)

/-- Theorem stating the total distance and additional sprint for Darnel's activities --/
theorem darnel_running_results (activity : RunningActivity)
  (h1 : activity.sprint1 = 0.88)
  (h2 : activity.sprint2 = 1.12)
  (h3 : activity.jog1 = 0.75)
  (h4 : activity.jog2 = 0.45)
  (h5 : activity.walk = 0.32) :
  totalDistance activity = 3.52 ∧ additionalSprint activity = 0.48 := by
  sorry

end NUMINAMATH_CALUDE_darnel_running_results_l2071_207125


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2071_207158

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem statement -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 4 + a 10 + a 16 = 30) : 
  a 18 - 2 * a 14 = -10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l2071_207158


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2071_207167

theorem quadratic_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 5 * x^2 + 8 * x - 24 = 0 ∧ x = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2071_207167


namespace NUMINAMATH_CALUDE_solution_set_and_inequality_l2071_207176

def f (x : ℝ) := -x + |2*x + 1|

def M : Set ℝ := {x | f x < 2}

theorem solution_set_and_inequality :
  (M = {x : ℝ | -1 < x ∧ x < 1}) ∧
  (∀ a b : ℝ, a ∈ M → b ∈ M → 2 * |a * b| + 1 > |a| + |b|) := by sorry

end NUMINAMATH_CALUDE_solution_set_and_inequality_l2071_207176


namespace NUMINAMATH_CALUDE_walmart_shelving_problem_l2071_207170

/-- Given a total number of pots and the capacity of each shelf,
    calculate the number of shelves needed to stock all pots. -/
def shelves_needed (total_pots : ℕ) (vertical_capacity : ℕ) (horizontal_capacity : ℕ) : ℕ :=
  (total_pots + vertical_capacity * horizontal_capacity - 1) / (vertical_capacity * horizontal_capacity)

/-- Proof that 4 shelves are needed to stock 60 pots when each shelf can hold 
    5 vertically stacked pots in 3 side-by-side sets. -/
theorem walmart_shelving_problem : shelves_needed 60 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_walmart_shelving_problem_l2071_207170


namespace NUMINAMATH_CALUDE_fruit_basket_count_l2071_207103

/-- The number of possible fruit baskets with at least one piece of fruit -/
def num_fruit_baskets (num_apples : Nat) (num_oranges : Nat) : Nat :=
  (num_apples + 1) * (num_oranges + 1) - 1

/-- Theorem stating the number of fruit baskets with 7 apples and 12 oranges -/
theorem fruit_basket_count :
  num_fruit_baskets 7 12 = 103 := by
  sorry

end NUMINAMATH_CALUDE_fruit_basket_count_l2071_207103


namespace NUMINAMATH_CALUDE_equation_one_solution_l2071_207106

theorem equation_one_solution (x : ℝ) : 
  (3 * x + 2)^2 = 25 ↔ x = 1 ∨ x = -7/3 := by sorry

end NUMINAMATH_CALUDE_equation_one_solution_l2071_207106


namespace NUMINAMATH_CALUDE_range_of_a_l2071_207162

def f (x : ℝ) := x^2 - 4*x

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-4) a, f x ∈ Set.Icc (-4) 32) →
  (Set.Icc (-4) a = f ⁻¹' (Set.Icc (-4) 32)) →
  a ∈ Set.Icc 2 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2071_207162


namespace NUMINAMATH_CALUDE_parallel_planes_through_two_points_l2071_207121

-- Define a plane
def Plane : Type := sorry

-- Define a point
def Point : Type := sorry

-- Define a function to check if a point is outside a plane
def isOutside (p : Point) (pl : Plane) : Prop := sorry

-- Define a function to check if a plane is parallel to another plane
def isParallel (pl1 : Plane) (pl2 : Plane) : Prop := sorry

-- Define a function to count the number of planes that can be drawn through two points and parallel to a given plane
def countParallelPlanes (p1 p2 : Point) (pl : Plane) : Nat := sorry

-- Theorem statement
theorem parallel_planes_through_two_points 
  (p1 p2 : Point) (pl : Plane) 
  (h1 : isOutside p1 pl) 
  (h2 : isOutside p2 pl) : 
  countParallelPlanes p1 p2 pl = 0 ∨ countParallelPlanes p1 p2 pl = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_planes_through_two_points_l2071_207121


namespace NUMINAMATH_CALUDE_locus_of_centers_l2071_207122

/-- Circle C1 with center (1,1) and radius 2 -/
def C1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

/-- Circle C2 with center (4,1) and radius 3 -/
def C2 (x y : ℝ) : Prop := (x - 4)^2 + (y - 1)^2 = 9

/-- A circle with center (a,b) and radius r -/
def Circle (a b r : ℝ) (x y : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

/-- External tangency condition -/
def ExternallyTangent (a b r : ℝ) : Prop := (a - 1)^2 + (b - 1)^2 = (r + 2)^2

/-- Internal tangency condition -/
def InternallyTangent (a b r : ℝ) : Prop := (a - 4)^2 + (b - 1)^2 = (3 - r)^2

/-- The locus equation -/
def LocusEquation (a b : ℝ) : Prop := 84*a^2 + 100*b^2 - 336*a - 200*b + 900 = 0

theorem locus_of_centers :
  ∀ a b : ℝ, (∃ r : ℝ, ExternallyTangent a b r ∧ InternallyTangent a b r) ↔ LocusEquation a b :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l2071_207122


namespace NUMINAMATH_CALUDE_amy_blue_balloons_l2071_207129

theorem amy_blue_balloons :
  let total_balloons : ℕ := 67
  let red_balloons : ℕ := 29
  let green_balloons : ℕ := 17
  let blue_balloons : ℕ := total_balloons - red_balloons - green_balloons
  blue_balloons = 21 := by
  sorry

end NUMINAMATH_CALUDE_amy_blue_balloons_l2071_207129


namespace NUMINAMATH_CALUDE_quadruplet_babies_l2071_207161

theorem quadruplet_babies (total : ℕ) (twins triplets quadruplets : ℕ) : 
  total = 1500 →
  2 * twins + 3 * triplets + 4 * quadruplets = total →
  triplets = 3 * quadruplets →
  twins = 2 * triplets →
  4 * quadruplets = 240 := by
sorry

end NUMINAMATH_CALUDE_quadruplet_babies_l2071_207161


namespace NUMINAMATH_CALUDE_final_face_is_four_l2071_207126

/-- Represents a standard 6-sided die where opposite faces sum to 7 -/
structure StandardDie where
  faces : Fin 6 → Nat
  opposite_sum_seven : ∀ (f : Fin 6), faces f + faces (5 - f) = 7

/-- Represents a move direction -/
inductive Move
| Left
| Forward
| Right
| Back

/-- The sequence of moves in the path -/
def path : List Move := [Move.Left, Move.Forward, Move.Right, Move.Back, Move.Forward, Move.Back]

/-- Simulates rolling the die in a given direction -/
def roll (d : StandardDie) (m : Move) (top : Fin 6) : Fin 6 :=
  sorry

/-- Simulates rolling the die along the entire path -/
def rollPath (d : StandardDie) (initial : Fin 6) : Fin 6 :=
  sorry

/-- Theorem stating that the final top face is 4 regardless of initial state -/
theorem final_face_is_four (d : StandardDie) (initial : Fin 6) :
  d.faces (rollPath d initial) = 4 := by sorry

end NUMINAMATH_CALUDE_final_face_is_four_l2071_207126


namespace NUMINAMATH_CALUDE_prob_odd_sum_is_four_ninths_l2071_207189

/-- Represents a skewed six-sided die where rolling an odd number is twice as likely as rolling an even number. -/
structure SkewedDie :=
  (prob_even : ℝ)
  (prob_odd : ℝ)
  (six_sided : Nat)
  (skew_condition : prob_odd = 2 * prob_even)
  (probability_sum : prob_even + prob_odd = 1)
  (six_sided_condition : six_sided = 6)

/-- The probability of rolling an odd sum when rolling the skewed die twice. -/
def prob_odd_sum (d : SkewedDie) : ℝ :=
  2 * d.prob_even * d.prob_odd

/-- Theorem stating that the probability of rolling an odd sum with the skewed die is 4/9. -/
theorem prob_odd_sum_is_four_ninths (d : SkewedDie) : 
  prob_odd_sum d = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_sum_is_four_ninths_l2071_207189


namespace NUMINAMATH_CALUDE_negation_of_existence_l2071_207139

theorem negation_of_existence (x : ℝ) :
  (¬ ∃ x : ℝ, |x| < 1) ↔ (∀ x : ℝ, x ≤ -1 ∨ x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_l2071_207139


namespace NUMINAMATH_CALUDE_exists_acute_triangle_l2071_207141

/-- Given five positive real numbers that can form triangles in any combination of three,
    there exists at least one acute-angled triangle among them. -/
theorem exists_acute_triangle
  (a b c d e : ℝ)
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0) (pos_e : e > 0)
  (triangle_abc : a + b > c ∧ b + c > a ∧ c + a > b)
  (triangle_abd : a + b > d ∧ b + d > a ∧ d + a > b)
  (triangle_abe : a + b > e ∧ b + e > a ∧ e + a > b)
  (triangle_acd : a + c > d ∧ c + d > a ∧ d + a > c)
  (triangle_ace : a + c > e ∧ c + e > a ∧ e + a > c)
  (triangle_ade : a + d > e ∧ d + e > a ∧ e + a > d)
  (triangle_bcd : b + c > d ∧ c + d > b ∧ d + b > c)
  (triangle_bce : b + c > e ∧ c + e > b ∧ e + b > c)
  (triangle_bde : b + d > e ∧ d + e > b ∧ e + b > d)
  (triangle_cde : c + d > e ∧ d + e > c ∧ e + c > d) :
  ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                 (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                 (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
                 x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                 x^2 + y^2 > z^2 ∧ y^2 + z^2 > x^2 ∧ z^2 + x^2 > y^2 :=
sorry

end NUMINAMATH_CALUDE_exists_acute_triangle_l2071_207141


namespace NUMINAMATH_CALUDE_shooting_stars_count_difference_l2071_207127

/-- The number of shooting stars counted by Bridget -/
def bridget_count : ℕ := 14

/-- The number of shooting stars counted by Reginald -/
def reginald_count : ℕ := 12

/-- The number of shooting stars counted by Sam -/
def sam_count : ℕ := reginald_count + 4

/-- The average number of shooting stars counted by all three -/
def average_count : ℚ := (bridget_count + reginald_count + sam_count) / 3

theorem shooting_stars_count_difference :
  sam_count = average_count + 2 →
  bridget_count - reginald_count = 2 := by
  sorry

#eval bridget_count - reginald_count

end NUMINAMATH_CALUDE_shooting_stars_count_difference_l2071_207127


namespace NUMINAMATH_CALUDE_aftershave_alcohol_percentage_l2071_207116

/-- Proves that the initial alcohol percentage in an after-shave lotion is 30% -/
theorem aftershave_alcohol_percentage
  (initial_volume : ℝ)
  (water_volume : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 50)
  (h2 : water_volume = 30)
  (h3 : final_percentage = 18.75)
  (h4 : (initial_volume * x / 100) = ((initial_volume + water_volume) * final_percentage / 100)) :
  x = 30 := by
  sorry

end NUMINAMATH_CALUDE_aftershave_alcohol_percentage_l2071_207116


namespace NUMINAMATH_CALUDE_inequality_system_solution_expression_factorization_l2071_207164

-- Part 1: System of inequalities
theorem inequality_system_solution (x : ℝ) :
  (2 * x + 1 ≤ 4 - x ∧ x - 1 < 3 * x / 2) ↔ (-2 < x ∧ x ≤ 1) := by sorry

-- Part 2: Expression factorization
theorem expression_factorization (a x y : ℝ) :
  a^2 * (x - y) + 4 * (y - x) = (x - y) * (a + 2) * (a - 2) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_expression_factorization_l2071_207164


namespace NUMINAMATH_CALUDE_original_denominator_problem_l2071_207136

theorem original_denominator_problem (d : ℚ) : 
  (4 : ℚ) / d ≠ 0 →
  (4 + 3 : ℚ) / (d + 3) = 2 / 3 →
  d = 7.5 := by
sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l2071_207136


namespace NUMINAMATH_CALUDE_pi_between_three_and_four_l2071_207104

theorem pi_between_three_and_four : 
  Irrational Real.pi ∧ 3 < Real.pi ∧ Real.pi < 4 := by sorry

end NUMINAMATH_CALUDE_pi_between_three_and_four_l2071_207104


namespace NUMINAMATH_CALUDE_range_of_f_l2071_207187

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 2)

theorem range_of_f :
  Set.range f = {y : ℝ | y < 1 ∨ y > 1} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l2071_207187


namespace NUMINAMATH_CALUDE_min_sum_given_product_minus_sum_l2071_207113

theorem min_sum_given_product_minus_sum (a b : ℝ) 
  (ha : a > 1) (hb : b > 1) (hab : a * b - (a + b) = 1) : 
  a + b ≥ 2 + 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_sum_given_product_minus_sum_l2071_207113


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2071_207112

/-- Sum of a geometric sequence -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sequence_sum :
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 1365/16384 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2071_207112


namespace NUMINAMATH_CALUDE_lcm_count_theorem_exists_valid_k_upper_bound_valid_k_l2071_207115

-- Define the function to count valid k values
def count_valid_k : ℕ :=
  -- Count the number of a values from 0 to 18 (inclusive)
  -- where k = 2^a * 3^36 satisfies the LCM condition
  (Finset.range 19).card

-- State the theorem
theorem lcm_count_theorem : 
  count_valid_k = 19 :=
sorry

-- Define the LCM condition
def is_valid_k (k : ℕ) : Prop :=
  Nat.lcm (Nat.lcm (9^9) (16^16)) k = 18^18

-- State the existence of valid k values
theorem exists_valid_k :
  ∃ k : ℕ, k > 0 ∧ is_valid_k k :=
sorry

-- State the upper bound of valid k values
theorem upper_bound_valid_k :
  ∀ k : ℕ, is_valid_k k → k ≤ 18^18 :=
sorry

end NUMINAMATH_CALUDE_lcm_count_theorem_exists_valid_k_upper_bound_valid_k_l2071_207115


namespace NUMINAMATH_CALUDE_additional_time_is_24_minutes_l2071_207192

/-- Time to fill one barrel normally (in minutes) -/
def normal_time : ℕ := 3

/-- Time to fill one barrel with leak (in minutes) -/
def leak_time : ℕ := 5

/-- Number of barrels to fill -/
def num_barrels : ℕ := 12

/-- Additional time required to fill barrels with leak -/
def additional_time : ℕ := (leak_time * num_barrels) - (normal_time * num_barrels)

theorem additional_time_is_24_minutes : additional_time = 24 := by
  sorry

end NUMINAMATH_CALUDE_additional_time_is_24_minutes_l2071_207192


namespace NUMINAMATH_CALUDE_regular_soda_count_l2071_207157

/-- The number of bottles of regular soda in a grocery store -/
def regular_soda_bottles : ℕ := 83

/-- The number of bottles of diet soda in the grocery store -/
def diet_soda_bottles : ℕ := 4

/-- The difference between the number of regular soda bottles and diet soda bottles -/
def soda_difference : ℕ := 79

theorem regular_soda_count :
  regular_soda_bottles = diet_soda_bottles + soda_difference := by
  sorry

end NUMINAMATH_CALUDE_regular_soda_count_l2071_207157


namespace NUMINAMATH_CALUDE_sqrt_11_plus_1_bounds_l2071_207119

-- Define the theorem
theorem sqrt_11_plus_1_bounds : 4 < Real.sqrt 11 + 1 ∧ Real.sqrt 11 + 1 < 5 := by
  sorry

#check sqrt_11_plus_1_bounds

end NUMINAMATH_CALUDE_sqrt_11_plus_1_bounds_l2071_207119


namespace NUMINAMATH_CALUDE_total_questions_formula_l2071_207166

/-- Represents the number of questions completed by three girls in 2 hours -/
def total_questions (fiona_questions : ℕ) (r : ℚ) : ℚ :=
  let shirley_questions := r * fiona_questions
  let kiana_questions := (fiona_questions + shirley_questions) / 2
  2 * (fiona_questions + shirley_questions + kiana_questions)

/-- Theorem stating the total number of questions completed by three girls in 2 hours -/
theorem total_questions_formula (r : ℚ) : 
  total_questions 36 r = 108 + 108 * r := by
  sorry

end NUMINAMATH_CALUDE_total_questions_formula_l2071_207166


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_value_l2071_207100

/-- A geometric sequence of real numbers. -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) * a (n - 1) = a n ^ 2

/-- Given a geometric sequence satisfying certain conditions, a_4 equals 8. -/
theorem geometric_sequence_a4_value (a : ℕ → ℝ) 
    (h_geom : geometric_sequence a)
    (h_sum : a 2 + a 6 = 34)
    (h_prod : a 3 * a 5 = 64) : 
  a 4 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_a4_value_l2071_207100


namespace NUMINAMATH_CALUDE_inequality_holds_l2071_207184

theorem inequality_holds (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^3 * (y + 1) + y^3 * (x + 1) ≥ x^2 * (y + y^2) + y^2 * (x + x^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l2071_207184


namespace NUMINAMATH_CALUDE_complex_number_calculation_l2071_207195

theorem complex_number_calculation : 
  let z : ℂ := 1 + Complex.I * Real.sqrt 2
  z^2 - 2*z = -3 := by sorry

end NUMINAMATH_CALUDE_complex_number_calculation_l2071_207195


namespace NUMINAMATH_CALUDE_sum_of_squares_representation_l2071_207120

theorem sum_of_squares_representation (n m : ℕ) :
  ∃ (x y : ℕ), (2014^2 + 2016^2) / 2 = x^2 + y^2 ∧
  ∃ (a b : ℕ), (4*n^2 + 4*m^2) / 2 = a^2 + b^2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_representation_l2071_207120


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2071_207109

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 3 > 0) ↔ (∃ x : ℝ, x^2 + x + 3 ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2071_207109


namespace NUMINAMATH_CALUDE_paths_from_A_to_C_l2071_207173

/-- The number of paths between two points -/
def num_paths (start finish : Point) : ℕ := sorry

/-- A point in the graph -/
inductive Point
| A
| B
| C

/-- The total number of paths from A to C -/
def total_paths : ℕ := sorry

theorem paths_from_A_to_C :
  (num_paths Point.A Point.B = 3) →
  (num_paths Point.B Point.C = 1) →
  (num_paths Point.A Point.C = 1) →
  total_paths = 4 := by sorry

end NUMINAMATH_CALUDE_paths_from_A_to_C_l2071_207173


namespace NUMINAMATH_CALUDE_exists_prime_with_integer_roots_l2071_207138

theorem exists_prime_with_integer_roots :
  ∃ p : ℕ, Prime p ∧ 1 < p ∧ p ≤ 11 ∧
  ∃ x y : ℤ, x^2 + p*x - 720*p = 0 ∧ y^2 + p*y - 720*p = 0 :=
by sorry

end NUMINAMATH_CALUDE_exists_prime_with_integer_roots_l2071_207138


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2071_207182

theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (x + 2 > 2*x - 6 ∧ x < m) ↔ x < 8) → m ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2071_207182


namespace NUMINAMATH_CALUDE_A_intersect_B_l2071_207114

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x | 0 < 2 - x ∧ 2 - x < 3}

theorem A_intersect_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_l2071_207114


namespace NUMINAMATH_CALUDE_point_C_coordinates_l2071_207124

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the angle bisector line
def AngleBisector : ℝ → ℝ := fun x ↦ 2 * x

-- Define the condition that y=2x is the angle bisector of ∠C
def IsAngleBisector (t : Triangle) : Prop :=
  AngleBisector (t.C.1) = t.C.2

theorem point_C_coordinates (t : Triangle) :
  t.A = (-4, 2) →
  t.B = (3, 1) →
  IsAngleBisector t →
  t.C = (2, 4) := by
  sorry


end NUMINAMATH_CALUDE_point_C_coordinates_l2071_207124


namespace NUMINAMATH_CALUDE_total_gold_value_l2071_207144

/-- Calculates the total value of gold for Legacy, Aleena, and Briana -/
theorem total_gold_value (legacy_bars : ℕ) (aleena_bars_diff : ℕ) (briana_bars : ℕ)
  (legacy_aleena_value : ℕ) (briana_value : ℕ) :
  legacy_bars = 12 →
  aleena_bars_diff = 4 →
  briana_bars = 8 →
  legacy_aleena_value = 3500 →
  briana_value = 4000 →
  (legacy_bars * legacy_aleena_value) +
  ((legacy_bars - aleena_bars_diff) * legacy_aleena_value) +
  (briana_bars * briana_value) = 102000 :=
by sorry

end NUMINAMATH_CALUDE_total_gold_value_l2071_207144


namespace NUMINAMATH_CALUDE_fran_speed_l2071_207146

/-- Given Joann's bike ride parameters and Fran's time, calculate Fran's required speed --/
theorem fran_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5) :
  (joann_speed * joann_time) / fran_time = 60 / 3.5 := by
  sorry

end NUMINAMATH_CALUDE_fran_speed_l2071_207146


namespace NUMINAMATH_CALUDE_line_translation_l2071_207155

/-- Given a line y = 2x translated by vector (m, n) to y = 2x + 5, 
    prove the relationship between m and n. -/
theorem line_translation (m n : ℝ) : 
  (∀ x y : ℝ, y = 2*x + 5 ↔ y - n = 2*(x - m)) → n = 2*m + 5 := by
sorry

end NUMINAMATH_CALUDE_line_translation_l2071_207155


namespace NUMINAMATH_CALUDE_divisibility_by_fifteen_l2071_207159

theorem divisibility_by_fifteen (a : ℤ) :
  15 ∣ ((5 * a + 1) * (3 * a + 2)) ↔ a % 15 = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_fifteen_l2071_207159


namespace NUMINAMATH_CALUDE_range_of_a_l2071_207110

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ 
  (∃ x₀ : ℝ, x₀^2 + 2*a*x₀ + 2 - a = 0) →
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2071_207110


namespace NUMINAMATH_CALUDE_locus_of_centers_l2071_207117

/-- Circle C₁ with equation x² + y² = 1 -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C₂ with equation (x - 1)² + (y - 1)² = 81 -/
def C₂ (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 81

/-- A circle is externally tangent to C₁ if the distance between their centers is the sum of their radii -/
def externally_tangent_C₁ (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C₂ if the distance between their centers is the difference of their radii -/
def internally_tangent_C₂ (a b r : ℝ) : Prop := (a - 1)^2 + (b - 1)^2 = (9 - r)^2

/-- The locus of centers (a, b) of circles externally tangent to C₁ and internally tangent to C₂ -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C₁ a b r ∧ internally_tangent_C₂ a b r) → 
  a^2 + b^2 - (2*a*b)/63 - (66*a)/63 - (66*b)/63 + 17 = 0 := by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l2071_207117


namespace NUMINAMATH_CALUDE_tea_mixture_ratio_l2071_207180

theorem tea_mixture_ratio (price_tea1 price_tea2 price_mixture : ℚ) 
  (h1 : price_tea1 = 62)
  (h2 : price_tea2 = 72)
  (h3 : price_mixture = 64.5) :
  ∃ (x y : ℚ), x > 0 ∧ y > 0 ∧ x / y = 3 ∧
  (x * price_tea1 + y * price_tea2) / (x + y) = price_mixture :=
by sorry

end NUMINAMATH_CALUDE_tea_mixture_ratio_l2071_207180


namespace NUMINAMATH_CALUDE_leahs_garden_darker_tiles_l2071_207174

/-- Represents a square garden with a symmetrical tile pattern -/
structure SymmetricalGarden where
  -- The size of the repeating block
  block_size : ℕ
  -- The size of the center square in each block
  center_size : ℕ
  -- The number of darker tiles in the center square
  darker_tiles_in_center : ℕ

/-- The fraction of darker tiles in the garden -/
def fraction_of_darker_tiles (g : SymmetricalGarden) : ℚ :=
  (g.darker_tiles_in_center * (g.block_size / g.center_size)^2 : ℚ) / g.block_size^2

/-- Theorem stating the fraction of darker tiles in Leah's garden -/
theorem leahs_garden_darker_tiles :
  ∃ (g : SymmetricalGarden), 
    g.block_size = 4 ∧ 
    g.center_size = 2 ∧ 
    g.darker_tiles_in_center = 3 ∧ 
    fraction_of_darker_tiles g = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_leahs_garden_darker_tiles_l2071_207174


namespace NUMINAMATH_CALUDE_greatest_prime_divisor_l2071_207147

def ribbon_lengths : List ℕ := [8, 16, 20, 28]

theorem greatest_prime_divisor (lengths : List ℕ) : 
  ∃ (n : ℕ), n.Prime ∧ 
  (∀ m : ℕ, m.Prime → (∀ l ∈ lengths, l % m = 0) → m ≤ n) ∧
  (∀ l ∈ lengths, l % n = 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_prime_divisor_l2071_207147


namespace NUMINAMATH_CALUDE_smallest_square_count_is_minimal_l2071_207169

/-- The smallest positive integer n such that n * (1² + 2² + 3²) is a perfect square,
    where n represents the number of squares of each size (1x1, 2x2, 3x3) needed to form a larger square. -/
def smallest_square_count : ℕ := 14

/-- Predicate to check if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- Theorem stating that smallest_square_count is the smallest positive integer satisfying the conditions -/
theorem smallest_square_count_is_minimal :
  (is_perfect_square (smallest_square_count * (1 * 1 + 2 * 2 + 3 * 3))) ∧
  (∀ m : ℕ, m > 0 ∧ m < smallest_square_count →
    ¬(is_perfect_square (m * (1 * 1 + 2 * 2 + 3 * 3)))) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_count_is_minimal_l2071_207169


namespace NUMINAMATH_CALUDE_sin_graph_transformation_l2071_207197

/-- 
Given two trigonometric functions f(x) = 3sin(2x - π/6) and g(x) = 3sin(x + π/2),
prove that the graph of g(x) can be obtained from the graph of f(x) by 
extending the x-coordinates to twice their original values and 
then shifting the resulting graph to the left by 2π/3 units.
-/
theorem sin_graph_transformation (x : ℝ) : 
  3 * Real.sin (x + π/2) = 3 * Real.sin ((2*x - π/6) / 2 + 2*π/3) := by
sorry

end NUMINAMATH_CALUDE_sin_graph_transformation_l2071_207197


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l2071_207181

/-- The radius of the inscribed circle of a triangle with sides 15, 16, and 17 is √21 -/
theorem inscribed_circle_radius (a b c : ℝ) (ha : a = 15) (hb : b = 16) (hc : c = 17) :
  let s := (a + b + c) / 2
  let r := Real.sqrt (s * (s - a) * (s - b) * (s - c)) / s
  r = Real.sqrt 21 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l2071_207181


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2071_207178

theorem greatest_divisor_with_remainders : 
  ∃ (n : ℕ), n > 0 ∧ 
  (178340 % n = 20) ∧ 
  (253785 % n = 35) ∧ 
  (375690 % n = 50) ∧ 
  (∀ m : ℕ, m > 0 → 
    (178340 % m = 20) → 
    (253785 % m = 35) → 
    (375690 % m = 50) → 
    m ≤ n) ∧
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2071_207178


namespace NUMINAMATH_CALUDE_lanas_muffin_goal_l2071_207185

/-- Lana's muffin sale problem -/
theorem lanas_muffin_goal (morning_sales afternoon_sales more_needed : ℕ) 
  (h1 : morning_sales = 12)
  (h2 : afternoon_sales = 4)
  (h3 : more_needed = 4) :
  morning_sales + afternoon_sales + more_needed = 20 := by
  sorry

end NUMINAMATH_CALUDE_lanas_muffin_goal_l2071_207185


namespace NUMINAMATH_CALUDE_root_equation_r_values_l2071_207191

theorem root_equation_r_values (r : ℤ) : 
  (∃ x y : ℤ, x > 0 ∧ y > 0 ∧ 
    r * x^2 - (2*r + 7) * x + r + 7 = 0 ∧
    r * y^2 - (2*r + 7) * y + r + 7 = 0) →
  r = 7 ∨ r = 0 ∨ r = 1 :=
by sorry

end NUMINAMATH_CALUDE_root_equation_r_values_l2071_207191


namespace NUMINAMATH_CALUDE_curve_symmetrical_y_axis_l2071_207183

-- Define a function to represent the left-hand side of the equation
def f (x y : ℝ) : ℝ := x^2 - y^2

-- Theorem stating that the curve is symmetrical with respect to the y-axis
theorem curve_symmetrical_y_axis : ∀ x y : ℝ, f x y = 1 ↔ f (-x) y = 1 := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetrical_y_axis_l2071_207183


namespace NUMINAMATH_CALUDE_base_four_of_156_l2071_207175

def base_four_representation (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_four_of_156 :
  base_four_representation 156 = [2, 1, 3, 0] := by sorry

end NUMINAMATH_CALUDE_base_four_of_156_l2071_207175


namespace NUMINAMATH_CALUDE_count_special_quadrilaterals_l2071_207150

/-- A quadrilateral with specific properties -/
structure SpecialQuadrilateral where
  ab : ℕ+
  bc : ℕ+
  cd : ℕ+
  ad : ℕ+
  right_angle_b : True  -- Represents the right angle at B
  right_angle_c : True  -- Represents the right angle at C
  ab_eq_two : ab = 2
  cd_eq_ad : cd = ad

/-- The perimeter of a SpecialQuadrilateral -/
def perimeter (q : SpecialQuadrilateral) : ℕ :=
  q.ab + q.bc + q.cd + q.ad

/-- The theorem statement -/
theorem count_special_quadrilaterals :
  (∃ (s : Finset ℕ), s.card = 31 ∧
    (∀ p ∈ s, p < 2015 ∧ ∃ q : SpecialQuadrilateral, perimeter q = p) ∧
    (∀ p < 2015, (∃ q : SpecialQuadrilateral, perimeter q = p) → p ∈ s)) :=
sorry

end NUMINAMATH_CALUDE_count_special_quadrilaterals_l2071_207150


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2071_207123

theorem polynomial_simplification (x : ℝ) : 
  (3 * x^3 + 4 * x^2 + 9 * x - 5) - (2 * x^3 + 3 * x^2 + 6 * x - 8) = x^3 + x^2 + 3 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2071_207123


namespace NUMINAMATH_CALUDE_seven_lines_regions_l2071_207177

/-- The number of regions created by n lines in a plane, where no two lines are parallel and no three lines meet at a single point -/
def num_regions (n : ℕ) : ℕ := 1 + n + n * (n - 1) / 2

/-- Seven lines in a plane with the given conditions divide the plane into 29 regions -/
theorem seven_lines_regions : num_regions 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_seven_lines_regions_l2071_207177


namespace NUMINAMATH_CALUDE_total_marbles_in_jar_l2071_207140

def ben_marbles : ℕ := 56
def leo_marbles_difference : ℕ := 20

theorem total_marbles_in_jar : 
  ben_marbles + (ben_marbles + leo_marbles_difference) = 132 := by
sorry

end NUMINAMATH_CALUDE_total_marbles_in_jar_l2071_207140


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2071_207148

theorem absolute_value_inequality_solution_set (x : ℝ) :
  (|x - 1| < 1) ↔ (x ∈ Set.Ioo 0 2) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l2071_207148


namespace NUMINAMATH_CALUDE_train_passing_station_time_l2071_207163

/-- The time taken for a train to pass a station -/
theorem train_passing_station_time
  (train_length : Real)
  (train_speed_kmh : Real)
  (station_length : Real)
  (h1 : train_length = 250)
  (h2 : train_speed_kmh = 36)
  (h3 : station_length = 200) :
  (train_length + station_length) / (train_speed_kmh * 1000 / 3600) = 45 := by
  sorry

#check train_passing_station_time

end NUMINAMATH_CALUDE_train_passing_station_time_l2071_207163


namespace NUMINAMATH_CALUDE_boys_count_l2071_207142

/-- Represents the number of boys on the chess team -/
def boys : ℕ := sorry

/-- Represents the number of girls on the chess team -/
def girls : ℕ := sorry

/-- The total number of team members is 30 -/
axiom total_members : boys + girls = 30

/-- 18 members attended the last meeting -/
axiom attendees : (2 * girls / 3 : ℚ) + boys = 18

/-- Proves that the number of boys on the chess team is 6 -/
theorem boys_count : boys = 6 := by sorry

end NUMINAMATH_CALUDE_boys_count_l2071_207142


namespace NUMINAMATH_CALUDE_scientific_notation_450_million_l2071_207143

theorem scientific_notation_450_million :
  (450000000 : ℝ) = 4.5 * (10 : ℝ)^8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_450_million_l2071_207143


namespace NUMINAMATH_CALUDE_intersection_point_l2071_207130

/-- The slope of the first line -/
def m₁ : ℚ := 3

/-- The y-intercept of the first line -/
def b₁ : ℚ := -4

/-- The x-coordinate of the given point -/
def x₀ : ℚ := 3

/-- The y-coordinate of the given point -/
def y₀ : ℚ := 2

/-- The slope of the perpendicular line -/
def m₂ : ℚ := -1 / m₁

/-- The y-intercept of the perpendicular line -/
def b₂ : ℚ := y₀ - m₂ * x₀

/-- The x-coordinate of the intersection point -/
def x_intersect : ℚ := (b₂ - b₁) / (m₁ - m₂)

/-- The y-coordinate of the intersection point -/
def y_intersect : ℚ := m₁ * x_intersect + b₁

theorem intersection_point :
  (x_intersect = 27 / 10) ∧ (y_intersect = 41 / 10) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2071_207130


namespace NUMINAMATH_CALUDE_statement_c_not_always_true_l2071_207134

theorem statement_c_not_always_true :
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 := by sorry

end NUMINAMATH_CALUDE_statement_c_not_always_true_l2071_207134


namespace NUMINAMATH_CALUDE_first_discount_percentage_l2071_207193

theorem first_discount_percentage 
  (original_price : ℝ) 
  (final_price : ℝ) 
  (second_discount : ℝ) : 
  original_price = 400 →
  final_price = 342 →
  second_discount = 5 →
  ∃ (first_discount : ℝ),
    final_price = original_price * (100 - first_discount) / 100 * (100 - second_discount) / 100 ∧
    first_discount = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l2071_207193


namespace NUMINAMATH_CALUDE_football_team_numbers_l2071_207128

theorem football_team_numbers (x : ℕ) (n : ℕ) : 
  (n * (n + 1)) / 2 - x = 100 → x = 5 ∧ n = 14 := by
  sorry

end NUMINAMATH_CALUDE_football_team_numbers_l2071_207128


namespace NUMINAMATH_CALUDE_katie_sugar_calculation_l2071_207153

/-- Given a recipe that requires a total amount of sugar and an amount already added,
    calculate the remaining amount needed. -/
def remaining_sugar (total : ℝ) (added : ℝ) : ℝ :=
  total - added

theorem katie_sugar_calculation :
  let total_required : ℝ := 3
  let already_added : ℝ := 0.5
  remaining_sugar total_required already_added = 2.5 := by
sorry

end NUMINAMATH_CALUDE_katie_sugar_calculation_l2071_207153


namespace NUMINAMATH_CALUDE_sum_of_features_l2071_207131

/-- A rectangular prism -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- The number of edges in a rectangular prism -/
def num_edges (prism : RectangularPrism) : ℕ := 12

/-- The number of corners in a rectangular prism -/
def num_corners (prism : RectangularPrism) : ℕ := 8

/-- The number of faces in a rectangular prism -/
def num_faces (prism : RectangularPrism) : ℕ := 6

/-- The theorem stating that the sum of edges, corners, and faces is 26 -/
theorem sum_of_features (prism : RectangularPrism) :
  num_edges prism + num_corners prism + num_faces prism = 26 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_features_l2071_207131


namespace NUMINAMATH_CALUDE_consecutive_multiples_of_three_l2071_207132

theorem consecutive_multiples_of_three (n : ℕ) : 
  3 * (n - 1) + 3 * (n + 1) = 150 → 3 * n = 75 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_multiples_of_three_l2071_207132


namespace NUMINAMATH_CALUDE_lcm_of_25_35_50_l2071_207168

theorem lcm_of_25_35_50 : Nat.lcm 25 (Nat.lcm 35 50) = 350 := by sorry

end NUMINAMATH_CALUDE_lcm_of_25_35_50_l2071_207168


namespace NUMINAMATH_CALUDE_solution_set_is_two_lines_l2071_207152

/-- The solution set of the equation (2x - y)^2 = 4x^2 - y^2 -/
def SolutionSet : Set (ℝ × ℝ) :=
  {(x, y) | (2*x - y)^2 = 4*x^2 - y^2}

/-- The set consisting of two lines: y = 0 and y = 2x -/
def TwoLines : Set (ℝ × ℝ) :=
  {(x, y) | y = 0 ∨ y = 2*x}

/-- Theorem stating that the solution set of the equation is equivalent to two lines -/
theorem solution_set_is_two_lines : SolutionSet = TwoLines := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_two_lines_l2071_207152


namespace NUMINAMATH_CALUDE_matildas_father_chocolates_l2071_207198

/-- Calculates the number of chocolate bars Matilda's father had left -/
def fathersRemainingChocolates (initialBars : ℕ) (people : ℕ) (givenToMother : ℕ) (eaten : ℕ) : ℕ :=
  let barsPerPerson := initialBars / people
  let givenToFather := people * (barsPerPerson / 2)
  givenToFather - givenToMother - eaten

/-- Proves that Matilda's father had 5 chocolate bars left -/
theorem matildas_father_chocolates :
  fathersRemainingChocolates 20 5 3 2 = 5 := by
  sorry

#eval fathersRemainingChocolates 20 5 3 2

end NUMINAMATH_CALUDE_matildas_father_chocolates_l2071_207198


namespace NUMINAMATH_CALUDE_scientific_notation_equals_original_l2071_207118

/-- Scientific notation representation of 470,000,000 -/
def scientific_notation : ℝ := 4.7 * (10 ^ 8)

/-- The original number -/
def original_number : ℕ := 470000000

theorem scientific_notation_equals_original : 
  (scientific_notation : ℝ) = original_number := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equals_original_l2071_207118


namespace NUMINAMATH_CALUDE_min_value_f_neg_reals_l2071_207111

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x - 8/x + 4/x^2 + 5

theorem min_value_f_neg_reals :
  ∃ (x_min : ℝ), x_min < 0 ∧
  ∀ (x : ℝ), x < 0 → f x ≥ f x_min ∧ f x_min = 9 + 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_neg_reals_l2071_207111


namespace NUMINAMATH_CALUDE_x_value_l2071_207188

theorem x_value (x y : ℚ) (h1 : x / y = 5 / 2) (h2 : y = 30) : x = 75 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2071_207188


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2071_207154

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (1 / x + 9 / y) ≥ 16 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ 1 / x + 9 / y = 16 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l2071_207154


namespace NUMINAMATH_CALUDE_factor_implies_d_value_l2071_207160

-- Define the polynomial Q(x)
def Q (d : ℝ) (x : ℝ) : ℝ := x^3 - 3*x^2 + d*x - 8

-- Theorem statement
theorem factor_implies_d_value :
  ∀ d : ℝ, (∀ x : ℝ, (x - 3) ∣ Q d x) → d = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_d_value_l2071_207160


namespace NUMINAMATH_CALUDE_no_real_a_for_single_solution_l2071_207186

theorem no_real_a_for_single_solution :
  ¬ ∃ (a : ℝ), ∃! (x : ℝ), |x^2 + 4*a*x + 5*a| ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_no_real_a_for_single_solution_l2071_207186


namespace NUMINAMATH_CALUDE_luke_needs_307_stars_l2071_207107

/-- The number of additional stars Luke needs to make -/
def additional_stars_needed (stars_per_jar : ℕ) (jars_to_fill : ℕ) (stars_already_made : ℕ) : ℕ :=
  stars_per_jar * jars_to_fill - stars_already_made

/-- Proof that Luke needs to make 307 more stars -/
theorem luke_needs_307_stars :
  additional_stars_needed 85 4 33 = 307 := by
  sorry

end NUMINAMATH_CALUDE_luke_needs_307_stars_l2071_207107


namespace NUMINAMATH_CALUDE_most_appropriate_survey_method_l2071_207133

/-- Represents different survey methods -/
inductive SurveyMethod
| Census
| Sampling

/-- Represents different survey scenarios -/
inductive SurveyScenario
| CityFloatingPopulation
| AirplaneSecurityCheck
| ShellKillingRadius
| ClassMathScores

/-- Determines if a survey method is appropriate for a given scenario -/
def is_appropriate (method : SurveyMethod) (scenario : SurveyScenario) : Prop :=
  match scenario with
  | SurveyScenario.CityFloatingPopulation => method = SurveyMethod.Sampling
  | SurveyScenario.AirplaneSecurityCheck => method = SurveyMethod.Census
  | SurveyScenario.ShellKillingRadius => method = SurveyMethod.Sampling
  | SurveyScenario.ClassMathScores => method = SurveyMethod.Census

/-- Theorem stating that using a census method for class math scores is the most appropriate -/
theorem most_appropriate_survey_method :
  is_appropriate SurveyMethod.Census SurveyScenario.ClassMathScores ∧
  ¬(is_appropriate SurveyMethod.Census SurveyScenario.CityFloatingPopulation) ∧
  ¬(is_appropriate SurveyMethod.Sampling SurveyScenario.AirplaneSecurityCheck) ∧
  ¬(is_appropriate SurveyMethod.Census SurveyScenario.ShellKillingRadius) :=
by sorry

end NUMINAMATH_CALUDE_most_appropriate_survey_method_l2071_207133
