import Mathlib

namespace NUMINAMATH_CALUDE_danny_share_l1007_100796

/-- Represents the share of money each person receives -/
structure Share :=
  (amount : ℝ)
  (removed : ℝ)

/-- The problem setup -/
def problem_setup :=
  (total : ℝ) →
  (alice : Share) →
  (bond : Share) →
  (charlie : Share) →
  (danny : Share) →
  Prop

/-- The conditions of the problem -/
def conditions (total : ℝ) (alice bond charlie danny : Share) : Prop :=
  total = 2210 ∧
  alice.removed = 30 ∧
  bond.removed = 50 ∧
  charlie.removed = 40 ∧
  danny.removed = 2 * charlie.removed ∧
  (alice.amount - alice.removed) / (bond.amount - bond.removed) = 11 / 18 ∧
  (alice.amount - alice.removed) / (charlie.amount - charlie.removed) = 11 / 24 ∧
  (alice.amount - alice.removed) / (danny.amount - danny.removed) = 11 / 32 ∧
  alice.amount + bond.amount + charlie.amount + danny.amount = total

/-- The theorem to prove -/
theorem danny_share (total : ℝ) (alice bond charlie danny : Share) :
  conditions total alice bond charlie danny →
  danny.amount = 916.80 :=
sorry

end NUMINAMATH_CALUDE_danny_share_l1007_100796


namespace NUMINAMATH_CALUDE_factor_expression_l1007_100758

theorem factor_expression (x : ℝ) : 3*x*(x-4) + 5*(x-4) - 2*(x-4) = (3*x + 3)*(x-4) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l1007_100758


namespace NUMINAMATH_CALUDE_sqrt_problem_1_sqrt_problem_2_l1007_100770

-- Problem 1
theorem sqrt_problem_1 : Real.sqrt 6 * Real.sqrt 3 - 6 * Real.sqrt (1/2) = 0 := by sorry

-- Problem 2
theorem sqrt_problem_2 : (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 = 3 := by sorry

end NUMINAMATH_CALUDE_sqrt_problem_1_sqrt_problem_2_l1007_100770


namespace NUMINAMATH_CALUDE_sin_square_sum_range_l1007_100743

open Real

theorem sin_square_sum_range (α β : ℝ) (h : 3 * (sin α)^2 - 2 * sin α + 2 * (sin β)^2 = 0) :
  ∃ (x : ℝ), x = (sin α)^2 + (sin β)^2 ∧ 0 ≤ x ∧ x ≤ 4/9 ∧
  ∀ (y : ℝ), y = (sin α)^2 + (sin β)^2 → 0 ≤ y ∧ y ≤ 4/9 :=
by sorry

end NUMINAMATH_CALUDE_sin_square_sum_range_l1007_100743


namespace NUMINAMATH_CALUDE_tan_theta_plus_pi_over_8_minus_reciprocal_l1007_100790

theorem tan_theta_plus_pi_over_8_minus_reciprocal (θ : Real) 
  (h : 3 * Real.sin θ + Real.cos θ = Real.sqrt 10) : 
  Real.tan (θ + π/8) - (1 / Real.tan (θ + π/8)) = -14 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_plus_pi_over_8_minus_reciprocal_l1007_100790


namespace NUMINAMATH_CALUDE_charles_skittles_l1007_100706

/-- The number of Skittles Charles has left after Diana takes some away. -/
def skittles_left (initial : ℕ) (taken : ℕ) : ℕ := initial - taken

/-- Theorem: If Charles has 25 Skittles initially and Diana takes 7 Skittles away,
    then Charles will have 18 Skittles left. -/
theorem charles_skittles : skittles_left 25 7 = 18 := by
  sorry

end NUMINAMATH_CALUDE_charles_skittles_l1007_100706


namespace NUMINAMATH_CALUDE_symmetric_points_y_axis_l1007_100726

/-- Given two points in R² that are symmetric about the y-axis, 
    prove that their x-coordinates are negatives of each other 
    and their y-coordinates are the same. -/
theorem symmetric_points_y_axis 
  (A B : ℝ × ℝ) 
  (h_symmetric : A.1 = -B.1 ∧ A.2 = B.2) 
  (h_A : A = (1, -2)) : 
  B = (-1, -2) := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_y_axis_l1007_100726


namespace NUMINAMATH_CALUDE_polynomial_not_equal_33_l1007_100755

theorem polynomial_not_equal_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_not_equal_33_l1007_100755


namespace NUMINAMATH_CALUDE_cost_difference_70_copies_l1007_100756

/-- Calculates the cost for color copies at print shop X -/
def costX (copies : ℕ) : ℚ :=
  if copies ≤ 50 then
    1.2 * copies
  else
    1.2 * 50 + 0.9 * (copies - 50)

/-- Calculates the cost for color copies at print shop Y -/
def costY (copies : ℕ) : ℚ :=
  10 + 1.7 * copies

/-- The difference in cost between print shop Y and X for 70 color copies is $51 -/
theorem cost_difference_70_copies : costY 70 - costX 70 = 51 := by
  sorry

end NUMINAMATH_CALUDE_cost_difference_70_copies_l1007_100756


namespace NUMINAMATH_CALUDE_printer_problem_l1007_100747

/-- Calculates the time needed to print a given number of pages at a specific rate -/
def print_time (pages : ℕ) (rate : ℕ) : ℚ :=
  (pages : ℚ) / (rate : ℚ)

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem printer_problem : 
  let pages : ℕ := 300
  let rate : ℕ := 20
  round_to_nearest (print_time pages rate) = 15 := by
  sorry

end NUMINAMATH_CALUDE_printer_problem_l1007_100747


namespace NUMINAMATH_CALUDE_roots_product_equals_squared_difference_l1007_100718

theorem roots_product_equals_squared_difference (m n : ℝ) 
  (α β γ δ : ℝ) : 
  (α^2 + m*α - 1 = 0) → 
  (β^2 + m*β - 1 = 0) → 
  (γ^2 + n*γ - 1 = 0) → 
  (δ^2 + n*δ - 1 = 0) → 
  (α - γ)*(β - γ)*(α - δ)*(β - δ) = (m - n)^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_product_equals_squared_difference_l1007_100718


namespace NUMINAMATH_CALUDE_growth_rate_inequality_l1007_100763

theorem growth_rate_inequality (p q x : ℝ) (h : p ≠ q) :
  (1 + x)^2 = (1 + p) * (1 + q) → x < (p + q) / 2 := by
  sorry

end NUMINAMATH_CALUDE_growth_rate_inequality_l1007_100763


namespace NUMINAMATH_CALUDE_melanie_dimes_l1007_100772

theorem melanie_dimes (initial : ℕ) (from_dad : ℕ) (total : ℕ) 
  (h1 : initial = 19)
  (h2 : from_dad = 39)
  (h3 : total = 83) :
  total - (initial + from_dad) = 25 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_l1007_100772


namespace NUMINAMATH_CALUDE_max_gcd_triangular_number_l1007_100710

def triangular_number (n : ℕ+) : ℕ := (n : ℕ) * (n + 1) / 2

theorem max_gcd_triangular_number :
  ∃ (n : ℕ+), Nat.gcd (6 * triangular_number n) (n + 2) = 6 ∧
  ∀ (m : ℕ+), Nat.gcd (6 * triangular_number m) (m + 2) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_max_gcd_triangular_number_l1007_100710


namespace NUMINAMATH_CALUDE_third_number_solution_l1007_100789

theorem third_number_solution (x : ℝ) : 3 + 33 + x + 33.3 = 399.6 → x = 330.3 := by
  sorry

end NUMINAMATH_CALUDE_third_number_solution_l1007_100789


namespace NUMINAMATH_CALUDE_parallel_lines_m_eq_3_l1007_100716

/-- Two lines are parallel if and only if their slopes are equal and they are not the same line -/
def parallel (m : ℝ) : Prop :=
  ((-m : ℝ) = -(4*m-3)/m) ∧ (m ≠ 1)

/-- The theorem states that if the lines l₁: mx+y-1=0 and l₂: (4m-3)x+my-1=0 are parallel, then m = 3 -/
theorem parallel_lines_m_eq_3 : ∀ m : ℝ, parallel m → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_eq_3_l1007_100716


namespace NUMINAMATH_CALUDE_christine_distance_l1007_100731

/-- Calculates the distance traveled given speed and time -/
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: Christine's distance traveled -/
theorem christine_distance :
  let speed : ℝ := 20
  let time : ℝ := 4
  distance_traveled speed time = 80 := by sorry

end NUMINAMATH_CALUDE_christine_distance_l1007_100731


namespace NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l1007_100722

def smallest_five_digit_number_divisible_by_smallest_primes : ℕ := 11550

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def is_divisible_by_smallest_primes (n : ℕ) : Prop :=
  n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ n % 7 = 0 ∧ n % 11 = 0

theorem smallest_five_digit_divisible_by_smallest_primes :
  (is_five_digit smallest_five_digit_number_divisible_by_smallest_primes) ∧
  (is_divisible_by_smallest_primes smallest_five_digit_number_divisible_by_smallest_primes) ∧
  (∀ m : ℕ, m < smallest_five_digit_number_divisible_by_smallest_primes →
    ¬(is_five_digit m ∧ is_divisible_by_smallest_primes m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_five_digit_divisible_by_smallest_primes_l1007_100722


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l1007_100724

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 3) (h₂ : a₂ = 13) (h₃ : a₃ = 23) :
  arithmetic_sequence a₁ (a₂ - a₁) 15 = 143 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l1007_100724


namespace NUMINAMATH_CALUDE_statement_b_incorrect_l1007_100707

/-- A predicate representing the conditions for a point to be on a locus -/
def LocusCondition (α : Type*) := α → Prop

/-- A predicate representing the geometric locus itself -/
def GeometricLocus (α : Type*) := α → Prop

/-- Statement B: If a point is on the locus, then it satisfies the conditions;
    however, there may be points not on the locus that also satisfy these conditions. -/
def StatementB (α : Type*) (locus : GeometricLocus α) (condition : LocusCondition α) : Prop :=
  (∀ x : α, locus x → condition x) ∧
  ∃ y : α, condition y ∧ ¬locus y

/-- Theorem stating that Statement B is an incorrect method for defining a geometric locus -/
theorem statement_b_incorrect (α : Type*) :
  ¬∀ (locus : GeometricLocus α) (condition : LocusCondition α),
    StatementB α locus condition ↔ (∀ x : α, locus x ↔ condition x) :=
sorry

end NUMINAMATH_CALUDE_statement_b_incorrect_l1007_100707


namespace NUMINAMATH_CALUDE_first_turkey_weight_is_6_l1007_100717

/-- The weight of the first turkey in kilograms -/
def first_turkey_weight : ℝ := 6

/-- The weight of the second turkey in kilograms -/
def second_turkey_weight : ℝ := 9

/-- The weight of the third turkey in kilograms -/
def third_turkey_weight : ℝ := 2 * second_turkey_weight

/-- The cost of turkey per kilogram in dollars -/
def cost_per_kg : ℝ := 2

/-- The total amount spent on turkeys in dollars -/
def total_spent : ℝ := 66

/-- Theorem stating that the weight of the first turkey is 6 kilograms -/
theorem first_turkey_weight_is_6 :
  first_turkey_weight = 6 ∧
  second_turkey_weight = 9 ∧
  third_turkey_weight = 2 * second_turkey_weight ∧
  cost_per_kg = 2 ∧
  total_spent = 66 ∧
  total_spent = cost_per_kg * (first_turkey_weight + second_turkey_weight + third_turkey_weight) :=
by
  sorry

#check first_turkey_weight_is_6

end NUMINAMATH_CALUDE_first_turkey_weight_is_6_l1007_100717


namespace NUMINAMATH_CALUDE_triangle_with_specific_properties_l1007_100787

/-- Represents a triangle with side lengths and circumradius -/
structure Triangle where
  a : ℕ
  b : ℕ
  c : ℕ
  r : ℕ

/-- Represents the distances from circumcenter to sides -/
structure CircumcenterDistances where
  d : ℕ
  e : ℕ

/-- The theorem statement -/
theorem triangle_with_specific_properties 
  (t : Triangle) 
  (dist : CircumcenterDistances) 
  (h1 : t.r = 25)
  (h2 : t.a > t.b)
  (h3 : t.a^2 + 4 * dist.d^2 = 2500)
  (h4 : t.b^2 + 4 * dist.e^2 = 2500) :
  t.a = 15 ∧ t.b = 7 ∧ t.c = 20 :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_specific_properties_l1007_100787


namespace NUMINAMATH_CALUDE_rollo_guinea_pig_food_l1007_100779

/-- The amount of food needed for Rollo's guinea pigs -/
def guinea_pig_food : ℕ → ℕ
| 1 => 2  -- First guinea pig eats 2 cups
| 2 => 2 * guinea_pig_food 1  -- Second eats twice as much as the first
| 3 => guinea_pig_food 2 + 3  -- Third eats 3 cups more than the second
| _ => 0  -- For completeness, though we only have 3 guinea pigs

/-- The total amount of food needed for all guinea pigs -/
def total_food : ℕ := guinea_pig_food 1 + guinea_pig_food 2 + guinea_pig_food 3

theorem rollo_guinea_pig_food : total_food = 13 := by
  sorry

end NUMINAMATH_CALUDE_rollo_guinea_pig_food_l1007_100779


namespace NUMINAMATH_CALUDE_sequence_general_term_l1007_100760

def S (n : ℕ+) : ℤ := n.val^2 - 2

def a : ℕ+ → ℤ
  | ⟨1, _⟩ => -1
  | ⟨n+2, _⟩ => 2*(n+2) - 1

theorem sequence_general_term (n : ℕ+) : 
  a n = if n = 1 then -1 else S n - S (n-1) := by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1007_100760


namespace NUMINAMATH_CALUDE_arithmetic_mean_implies_arithmetic_progression_geometric_mean_implies_geometric_progression_l1007_100753

/-- A sequence is an arithmetic progression if the difference between consecutive terms is constant. -/
def IsArithmeticProgression (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

/-- A sequence is a geometric progression if the ratio between consecutive terms is constant. -/
def IsGeometricProgression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) / a n = r

/-- Theorem: If each term (from the second to the second-to-last) in a sequence is the arithmetic mean
    of its neighboring terms, then the sequence is an arithmetic progression. -/
theorem arithmetic_mean_implies_arithmetic_progression (a : ℕ → ℝ) (n : ℕ) (h : n ≥ 3)
    (h_arithmetic_mean : ∀ k, 2 ≤ k ∧ k < n → a k = (a (k - 1) + a (k + 1)) / 2) :
    IsArithmeticProgression a := by sorry

/-- Theorem: If each term (from the second to the second-to-last) in a sequence is the geometric mean
    of its neighboring terms, then the sequence is a geometric progression. -/
theorem geometric_mean_implies_geometric_progression (a : ℕ → ℝ) (n : ℕ) (h : n ≥ 3)
    (h_geometric_mean : ∀ k, 2 ≤ k ∧ k < n → a k = Real.sqrt (a (k - 1) * a (k + 1))) :
    IsGeometricProgression a := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_implies_arithmetic_progression_geometric_mean_implies_geometric_progression_l1007_100753


namespace NUMINAMATH_CALUDE_larger_number_problem_l1007_100769

theorem larger_number_problem (x y : ℝ) (h1 : y > x) (h2 : 4 * y = 5 * x) (h3 : y - x = 10) : y = 50 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1007_100769


namespace NUMINAMATH_CALUDE_square_partition_theorem_l1007_100785

/-- A rectangle with side lengths a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b

/-- Predicate indicating if one rectangle can be placed inside another (possibly with rotation) -/
def can_fit_inside (r1 r2 : Rectangle) : Prop :=
  (r1.a ≤ r2.a ∧ r1.b ≤ r2.b) ∨ (r1.a ≤ r2.b ∧ r1.b ≤ r2.a)

theorem square_partition_theorem (n : ℕ) (hn : n^2 ≥ 4) :
  ∃ (rectangles : Fin (n^2) → Rectangle),
    (∀ i j, i ≠ j → rectangles i ≠ rectangles j) →
    (∃ (chosen : Fin (2*n) → Fin (n^2)),
      ∀ i j, i < j → can_fit_inside (rectangles (chosen i)) (rectangles (chosen j))) :=
  sorry

end NUMINAMATH_CALUDE_square_partition_theorem_l1007_100785


namespace NUMINAMATH_CALUDE_pirate_coin_division_l1007_100793

theorem pirate_coin_division (n m : ℕ) : 
  n % 10 = 5 → m = 2 * n → m % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_pirate_coin_division_l1007_100793


namespace NUMINAMATH_CALUDE_maximum_marks_calculation_l1007_100774

theorem maximum_marks_calculation (percentage : ℝ) (obtained_marks : ℝ) (max_marks : ℝ) : 
  percentage = 95 → obtained_marks = 285 → 
  (obtained_marks / max_marks) * 100 = percentage → 
  max_marks = 300 := by
  sorry

end NUMINAMATH_CALUDE_maximum_marks_calculation_l1007_100774


namespace NUMINAMATH_CALUDE_amaya_movie_watching_time_l1007_100714

/-- Calculates the total time spent watching a movie with interruptions and rewinds -/
def total_watching_time (segment1 segment2 segment3 rewind1 rewind2 : ℕ) : ℕ :=
  segment1 + segment2 + segment3 + rewind1 + rewind2

/-- Theorem stating that the total watching time for Amaya's movie is 120 minutes -/
theorem amaya_movie_watching_time :
  total_watching_time 35 45 20 5 15 = 120 := by
  sorry

#eval total_watching_time 35 45 20 5 15

end NUMINAMATH_CALUDE_amaya_movie_watching_time_l1007_100714


namespace NUMINAMATH_CALUDE_anne_age_when_paul_is_38_l1007_100723

/-- Given the initial ages of Paul and Anne in 2015, this theorem proves
    Anne's age when Paul is 38 years old. -/
theorem anne_age_when_paul_is_38 (paul_age_2015 anne_age_2015 : ℕ) 
    (h1 : paul_age_2015 = 11) 
    (h2 : anne_age_2015 = 14) : 
    anne_age_2015 + (38 - paul_age_2015) = 41 := by
  sorry

#check anne_age_when_paul_is_38

end NUMINAMATH_CALUDE_anne_age_when_paul_is_38_l1007_100723


namespace NUMINAMATH_CALUDE_question_one_l1007_100702

theorem question_one (x : ℝ) (h : x^2 - 3*x = 2) :
  1 + 2*x^2 - 6*x = 5 := by
  sorry


end NUMINAMATH_CALUDE_question_one_l1007_100702


namespace NUMINAMATH_CALUDE_chosen_number_l1007_100777

theorem chosen_number (x : ℝ) : (x / 6) - 15 = 5 → x = 120 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l1007_100777


namespace NUMINAMATH_CALUDE_sequence_nth_term_l1007_100751

theorem sequence_nth_term (n : ℕ+) (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) 
  (h_sum : ∀ k : ℕ+, S k = k^2) :
  a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_nth_term_l1007_100751


namespace NUMINAMATH_CALUDE_largest_number_l1007_100775

theorem largest_number (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : -1 < b ∧ b < 0) :
  (a - b) = max a (max (a * b) (max (a - b) (a + b))) := by
sorry

end NUMINAMATH_CALUDE_largest_number_l1007_100775


namespace NUMINAMATH_CALUDE_circle_area_tripled_l1007_100780

theorem circle_area_tripled (r n : ℝ) : 
  (r > 0) →
  (n > 0) →
  (π * (r + n)^2 = 3 * π * r^2) →
  (r = n * (Real.sqrt 3 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l1007_100780


namespace NUMINAMATH_CALUDE_soldier_movement_l1007_100757

theorem soldier_movement (n : ℕ) :
  (∃ (initial_config : Fin (n + 2) → Fin n → Bool)
     (final_config : Fin n → Fin (n + 2) → Bool),
   (∀ i j, initial_config i j → 
     ∃ i' j', final_config i' j' ∧ 
       ((i' = i ∧ j' = j) ∨ 
        (i'.val + 1 = i.val ∧ j' = j) ∨ 
        (i'.val = i.val + 1 ∧ j' = j) ∨ 
        (i' = i ∧ j'.val + 1 = j.val) ∨ 
        (i' = i ∧ j'.val = j.val + 1))) ∧
   (∀ i j, initial_config i j ↔ true) ∧
   (∀ i j, final_config i j ↔ true)) →
  Even n :=
by sorry

end NUMINAMATH_CALUDE_soldier_movement_l1007_100757


namespace NUMINAMATH_CALUDE_complex_power_simplification_l1007_100729

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The main theorem -/
theorem complex_power_simplification :
  ((1 + i) / (1 - i)) ^ 1002 = -1 := by sorry

end NUMINAMATH_CALUDE_complex_power_simplification_l1007_100729


namespace NUMINAMATH_CALUDE_right_triangle_inradius_l1007_100766

/-- A right triangle with side lengths 9, 12, and 15 has an inradius of 3 -/
theorem right_triangle_inradius : ∀ (a b c r : ℝ),
  a = 9 ∧ b = 12 ∧ c = 15 →  -- Given side lengths
  a^2 + b^2 = c^2 →          -- Right triangle condition
  (a + b + c) / 2 * r = (a * b) / 2 →  -- Area formula using inradius
  r = 3 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_inradius_l1007_100766


namespace NUMINAMATH_CALUDE_smallest_y_for_divisibility_by_11_l1007_100750

/-- Given a number in the form 7y86038 where y is a single digit (0-9),
    2 is the smallest whole number for y that makes the number divisible by 11. -/
theorem smallest_y_for_divisibility_by_11 :
  ∃ (y : ℕ), y ≤ 9 ∧ 
  (7 * 10^6 + y * 10^5 + 8 * 10^4 + 6 * 10^3 + 0 * 10^2 + 3 * 10 + 8) % 11 = 0 ∧
  ∀ (z : ℕ), z < y → (7 * 10^6 + z * 10^5 + 8 * 10^4 + 6 * 10^3 + 0 * 10^2 + 3 * 10 + 8) % 11 ≠ 0 ∧
  y = 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_y_for_divisibility_by_11_l1007_100750


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1007_100734

theorem greatest_of_three_consecutive_integers (x : ℤ) 
  (h : x + (x + 1) + (x + 2) = 21) : 
  max x (max (x + 1) (x + 2)) = 8 := by
sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1007_100734


namespace NUMINAMATH_CALUDE_sum_gcf_lcm_l1007_100700

/-- The greatest common factor of 15, 20, and 30 -/
def A : ℕ := Nat.gcd 15 (Nat.gcd 20 30)

/-- The least common multiple of 15, 20, and 30 -/
def B : ℕ := Nat.lcm 15 (Nat.lcm 20 30)

/-- The sum of the greatest common factor and least common multiple of 15, 20, and 30 is 65 -/
theorem sum_gcf_lcm : A + B = 65 := by sorry

end NUMINAMATH_CALUDE_sum_gcf_lcm_l1007_100700


namespace NUMINAMATH_CALUDE_rationalizing_and_comparison_l1007_100771

theorem rationalizing_and_comparison :
  -- Part 1: Rationalizing factor
  (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) = 1 ∧
  -- Part 2: Eliminating square roots in denominators
  2 / (3 * Real.sqrt 2) = Real.sqrt 2 / 3 ∧
  3 / (3 - Real.sqrt 6) = 3 + Real.sqrt 6 ∧
  -- Part 3: Comparison of square root expressions
  Real.sqrt 2023 - Real.sqrt 2022 < Real.sqrt 2022 - Real.sqrt 2021 :=
by
  sorry

end NUMINAMATH_CALUDE_rationalizing_and_comparison_l1007_100771


namespace NUMINAMATH_CALUDE_shift_direct_proportion_l1007_100762

/-- A linear function represented by its slope and y-intercept -/
structure LinearFunction where
  slope : ℝ
  intercept : ℝ

/-- Shift a linear function horizontally -/
def shift_right (f : LinearFunction) (units : ℝ) : LinearFunction :=
  { slope := f.slope, intercept := f.slope * units + f.intercept }

/-- The original direct proportion function y = -2x -/
def original_function : LinearFunction :=
  { slope := -2, intercept := 0 }

theorem shift_direct_proportion :
  shift_right original_function 3 = { slope := -2, intercept := 6 } := by
  sorry

end NUMINAMATH_CALUDE_shift_direct_proportion_l1007_100762


namespace NUMINAMATH_CALUDE_fraction_chain_l1007_100748

theorem fraction_chain (x y z w : ℚ) 
  (h1 : x / y = 5)
  (h2 : y / z = 1 / 4)
  (h3 : z / w = 7) :
  w / x = 4 / 35 := by
  sorry

end NUMINAMATH_CALUDE_fraction_chain_l1007_100748


namespace NUMINAMATH_CALUDE_total_ice_cubes_l1007_100788

/-- The number of ice cubes Dave originally had -/
def original_cubes : ℕ := 2

/-- The number of new ice cubes Dave made -/
def new_cubes : ℕ := 7

/-- Theorem: The total number of ice cubes Dave had is 9 -/
theorem total_ice_cubes : original_cubes + new_cubes = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_ice_cubes_l1007_100788


namespace NUMINAMATH_CALUDE_ring_toss_earnings_l1007_100781

/-- The number of days a ring toss game earned money, given total earnings and daily earnings. -/
def days_earned (total_earnings daily_earnings : ℕ) : ℕ :=
  total_earnings / daily_earnings

/-- Theorem stating that the ring toss game earned money for 5 days. -/
theorem ring_toss_earnings : days_earned 165 33 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_earnings_l1007_100781


namespace NUMINAMATH_CALUDE_smaugs_hoard_l1007_100738

theorem smaugs_hoard (gold_coins : ℕ) (silver_coins : ℕ) (copper_coins : ℕ) 
  (silver_to_copper : ℕ) (total_value : ℕ) :
  gold_coins = 100 →
  silver_coins = 60 →
  copper_coins = 33 →
  silver_to_copper = 8 →
  total_value = 2913 →
  total_value = gold_coins * silver_to_copper * (silver_coins / gold_coins) + 
                silver_coins * silver_to_copper + 
                copper_coins →
  silver_coins / gold_coins = 3 := by
  sorry

end NUMINAMATH_CALUDE_smaugs_hoard_l1007_100738


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l1007_100735

theorem polynomial_evaluation :
  let f (x : ℝ) := 2 * x^4 + 3 * x^3 + 5 * x^2 + x + 4
  f (-2) = 30 := by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l1007_100735


namespace NUMINAMATH_CALUDE_tank_capacity_l1007_100728

/-- Represents the tank system with its properties -/
structure TankSystem where
  capacity : ℝ
  outletA_time : ℝ
  outletB_time : ℝ
  inlet_rate : ℝ
  combined_extra_time : ℝ

/-- The tank system satisfies the given conditions -/
def satisfies_conditions (ts : TankSystem) : Prop :=
  ts.outletA_time = 5 ∧
  ts.outletB_time = 8 ∧
  ts.inlet_rate = 4 * 60 ∧
  ts.combined_extra_time = 3

/-- The theorem stating that the tank capacity is 1200 litres -/
theorem tank_capacity (ts : TankSystem) 
  (h : satisfies_conditions ts) : ts.capacity = 1200 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l1007_100728


namespace NUMINAMATH_CALUDE_inequality_implication_l1007_100711

theorem inequality_implication (a b : ℝ) (h : a - b > 0) : a + 1 > b + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1007_100711


namespace NUMINAMATH_CALUDE_edward_total_money_l1007_100784

-- Define the variables
def dollars_per_lawn : ℕ := 8
def lawns_mowed : ℕ := 5
def initial_savings : ℕ := 7

-- Define the theorem
theorem edward_total_money :
  dollars_per_lawn * lawns_mowed + initial_savings = 47 :=
by
  sorry

end NUMINAMATH_CALUDE_edward_total_money_l1007_100784


namespace NUMINAMATH_CALUDE_no_unbounded_phine_sequence_l1007_100782

/-- A phine sequence is a sequence of positive real numbers satisfying
    a_{n+2} = (a_{n+1} + a_{n-1}) / a_n for all n ≥ 2 -/
def IsPhine (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ n ≥ 2, a (n + 2) = (a (n + 1) + a (n - 1)) / a n)

/-- There does not exist an unbounded phine sequence -/
theorem no_unbounded_phine_sequence :
  ¬ ∃ a : ℕ → ℝ, IsPhine a ∧ ∀ r : ℝ, ∃ n : ℕ, a n > r :=
sorry

end NUMINAMATH_CALUDE_no_unbounded_phine_sequence_l1007_100782


namespace NUMINAMATH_CALUDE_x_value_proof_l1007_100708

theorem x_value_proof (x : ℝ) (h : 9 / (x^2) = x / 81) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l1007_100708


namespace NUMINAMATH_CALUDE_opposite_sum_zero_sum_zero_opposite_exists_opposite_not_negative_one_negative_one_ratio_opposite_l1007_100719

-- Define opposite numbers
def opposite (a b : ℝ) : Prop := a = -b

-- Statement 1
theorem opposite_sum_zero (a b : ℝ) : opposite a b → a + b = 0 := by sorry

-- Statement 2
theorem sum_zero_opposite (a b : ℝ) : a + b = 0 → opposite a b := by sorry

-- Statement 3
theorem exists_opposite_not_negative_one : ∃ a b : ℝ, opposite a b ∧ a / b ≠ -1 := by sorry

-- Statement 4
theorem negative_one_ratio_opposite (a b : ℝ) (h : b ≠ 0) : a / b = -1 → opposite a b := by sorry

end NUMINAMATH_CALUDE_opposite_sum_zero_sum_zero_opposite_exists_opposite_not_negative_one_negative_one_ratio_opposite_l1007_100719


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l1007_100765

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => validArrangements (n + 1) + validArrangements n

/-- The number of people sitting around the circular table. -/
def numPeople : ℕ := 10

/-- The probability of getting the desired outcome (no two adjacent people standing). -/
def probability : ℚ := validArrangements numPeople / 2^numPeople

/-- Theorem stating that the probability of no two adjacent people standing
    in a circular arrangement of 10 people, each flipping a fair coin, is 123/1024. -/
theorem no_adjacent_standing_probability :
  probability = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l1007_100765


namespace NUMINAMATH_CALUDE_student_number_problem_l1007_100746

theorem student_number_problem (x : ℝ) : 2 * x - 138 = 106 → x = 122 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l1007_100746


namespace NUMINAMATH_CALUDE_initial_skittles_count_l1007_100764

/-- Proves that the initial number of Skittles is equal to the product of the number of friends and the number of Skittles each friend received. -/
theorem initial_skittles_count (num_friends num_skittles_per_friend : ℕ) :
  num_friends * num_skittles_per_friend = num_friends * num_skittles_per_friend :=
by sorry

#check initial_skittles_count 5 8

end NUMINAMATH_CALUDE_initial_skittles_count_l1007_100764


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_360_l1007_100795

def sum_of_divisors (n : ℕ) : ℕ := sorry

theorem largest_prime_factor_of_sum_of_divisors_360 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ sum_of_divisors 360 ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ sum_of_divisors 360 → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_sum_of_divisors_360_l1007_100795


namespace NUMINAMATH_CALUDE_total_water_intake_l1007_100778

/-- Calculates the total water intake throughout the day given specific drinking patterns --/
theorem total_water_intake (morning : ℝ) : morning = 1.5 →
  let early_afternoon := 2 * morning
  let late_afternoon := 3 * morning
  let evening := late_afternoon * (1 - 0.25)
  let night := 2 * evening
  morning + early_afternoon + late_afternoon + evening + night = 19.125 := by
  sorry

end NUMINAMATH_CALUDE_total_water_intake_l1007_100778


namespace NUMINAMATH_CALUDE_special_sequence_eventually_periodic_l1007_100730

/-- A sequence of positive integers satisfying the given property -/
def SpecialSequence (a : ℕ → ℕ+) : Prop :=
  ∀ n : ℕ, (a n : ℕ) * (a (n + 1) : ℕ) = (a (n + 2) : ℕ) * (a (n + 3) : ℕ)

/-- Definition of eventual periodicity for a sequence -/
def EventuallyPeriodic (a : ℕ → ℕ+) : Prop :=
  ∃ N k : ℕ, k > 0 ∧ ∀ n ≥ N, a n = a (n + k)

/-- Theorem stating that a special sequence is eventually periodic -/
theorem special_sequence_eventually_periodic (a : ℕ → ℕ+) 
  (h : SpecialSequence a) : EventuallyPeriodic a :=
sorry

end NUMINAMATH_CALUDE_special_sequence_eventually_periodic_l1007_100730


namespace NUMINAMATH_CALUDE_third_graders_count_l1007_100786

theorem third_graders_count (T : ℚ) 
  (h1 : T + 2 * T + T / 2 = 70) : T = 20 := by
  sorry

end NUMINAMATH_CALUDE_third_graders_count_l1007_100786


namespace NUMINAMATH_CALUDE_four_of_a_kind_count_l1007_100701

/-- Represents a standard deck of playing cards -/
structure Deck :=
  (total_cards : ℕ)
  (num_suits : ℕ)
  (cards_per_suit : ℕ)
  (h_total : total_cards = num_suits * cards_per_suit)

/-- Represents a "Four of a Kind" combination -/
structure FourOfAKind :=
  (number : ℕ)
  (fifth_card : ℕ)
  (h_number_valid : number ≤ 13)
  (h_fifth_card_valid : fifth_card ≤ 52)
  (h_fifth_card_diff : fifth_card ≠ number)

/-- The number of different "Four of a Kind" combinations in a standard deck -/
def count_four_of_a_kind (d : Deck) : ℕ :=
  13 * (d.total_cards - d.num_suits)

/-- Theorem stating that the number of "Four of a Kind" combinations is 624 -/
theorem four_of_a_kind_count (d : Deck) 
  (h_standard : d.total_cards = 52 ∧ d.num_suits = 4 ∧ d.cards_per_suit = 13) : 
  count_four_of_a_kind d = 624 := by
  sorry

end NUMINAMATH_CALUDE_four_of_a_kind_count_l1007_100701


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1007_100742

def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x|

theorem sufficient_not_necessary_condition :
  (∀ a < 0, ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y) ∧
  (∃ a ≥ 0, ∀ x y, 0 ≤ x ∧ x < y → f a x < f a y) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1007_100742


namespace NUMINAMATH_CALUDE_inequality_proof_l1007_100754

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hab : a ≤ b) (hbc : b ≤ c) (hcd : c ≤ d)
  (hsum : a + b + c + d ≥ 1) :
  a^2 + 3*b^2 + 5*c^2 + 7*d^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1007_100754


namespace NUMINAMATH_CALUDE_max_attempts_for_ten_rooms_l1007_100749

/-- The maximum number of attempts needed to match n keys to n rooms -/
def maxAttempts (n : ℕ) : ℕ := (n * (n - 1)) / 2

/-- The number of rooms and keys -/
def numRooms : ℕ := 10

theorem max_attempts_for_ten_rooms :
  maxAttempts numRooms = 45 :=
by sorry

end NUMINAMATH_CALUDE_max_attempts_for_ten_rooms_l1007_100749


namespace NUMINAMATH_CALUDE_product_constraint_sum_l1007_100773

theorem product_constraint_sum (w x y z : ℕ) : 
  w * x * y * z = 720 → 
  0 < w → w < x → x < y → y < z → z < 20 → 
  w + z = 14 := by
sorry

end NUMINAMATH_CALUDE_product_constraint_sum_l1007_100773


namespace NUMINAMATH_CALUDE_problem_solution_l1007_100741

theorem problem_solution (a b c d : ℕ) 
  (h : 342 * (a * b * c * d + a * b + a * d + c * d + 1) = 379 * (b * c * d + b + d)) :
  a * 1000 + b * 100 + c * 10 + d = 1949 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1007_100741


namespace NUMINAMATH_CALUDE_eight_possible_pairs_l1007_100761

/-- Represents a seating arrangement at a round table -/
structure RoundTable :=
  (people : Finset (Fin 5))
  (girls : Finset (Fin 5))
  (boys : Finset (Fin 5))
  (all_seated : people = Finset.univ)
  (girls_boys_partition : girls ∪ boys = people ∧ girls ∩ boys = ∅)

/-- The number of people sitting next to at least one girl -/
def g (table : RoundTable) : ℕ := sorry

/-- The number of people sitting next to at least one boy -/
def b (table : RoundTable) : ℕ := sorry

/-- The set of all possible (g,b) pairs for a given round table -/
def possible_pairs (table : RoundTable) : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 = g table ∧ p.2 = b table) (Finset.product (Finset.range 6) (Finset.range 6))

/-- The theorem stating that there are exactly 8 possible (g,b) pairs -/
theorem eight_possible_pairs :
  ∀ table : RoundTable, Finset.card (possible_pairs table) = 8 := sorry

end NUMINAMATH_CALUDE_eight_possible_pairs_l1007_100761


namespace NUMINAMATH_CALUDE_function_properties_l1007_100721

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
def odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc a b → f (-x) = -f x

def decreasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc a b → y ∈ Set.Icc a b → x < y → f x > f y

-- State the theorem
theorem function_properties :
  odd_on_interval f (-1) 1 ∧ decreasing_on_interval f (-1) 1 →
  (∀ x₁ x₂, x₁ ∈ Set.Icc (-1) 1 → x₂ ∈ Set.Icc (-1) 1 → 
    (f x₁ + f x₂) * (x₁ + x₂) ≤ 0) ∧
  (∀ a, f (1 - a) + f ((1 - a)^2) < 0 → a ∈ Set.Ico 0 1) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l1007_100721


namespace NUMINAMATH_CALUDE_factorization_f_max_value_g_l1007_100744

-- Define the polynomials
def f (x : ℝ) : ℝ := x^2 - 4*x - 5
def g (x : ℝ) : ℝ := -2*x^2 - 4*x + 3

-- Theorem for factorization of f
theorem factorization_f : ∀ x : ℝ, f x = (x + 1) * (x - 5) := by sorry

-- Theorem for maximum value of g
theorem max_value_g : 
  (∀ x : ℝ, g x ≤ 5) ∧ g (-1) = 5 := by sorry

end NUMINAMATH_CALUDE_factorization_f_max_value_g_l1007_100744


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_condition_l1007_100797

def arithmetic_sequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem arithmetic_sequence_sum_condition
  (a₁ d : ℝ) (n : ℕ)
  (h1 : a₁ = 2)
  (h2 : d = 3)
  (h3 : arithmetic_sequence a₁ d n + arithmetic_sequence a₁ d (n + 2) = 28) :
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_condition_l1007_100797


namespace NUMINAMATH_CALUDE_range_of_2a_plus_3b_l1007_100768

theorem range_of_2a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : -1 ≤ a - b ∧ a - b ≤ 1) : 
  (∀ x, 2*a + 3*b ≤ x → 3 ≤ x) ∧ 
  (∀ y, y ≤ 2*a + 3*b → y ≤ -3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_2a_plus_3b_l1007_100768


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l1007_100767

theorem scientific_notation_equivalence : 
  274000000 = 2.74 * (10 ^ 8) := by sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l1007_100767


namespace NUMINAMATH_CALUDE_distance_from_origin_l1007_100704

theorem distance_from_origin (P : ℝ × ℝ) (h : P = (5, 12)) : 
  Real.sqrt ((P.1)^2 + (P.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_l1007_100704


namespace NUMINAMATH_CALUDE_angle_A_measure_l1007_100791

/-- In a geometric configuration with angles of 110°, 100°, and 40°, there exists an angle A that measures 30°. -/
theorem angle_A_measure (α β γ : Real) (h1 : α = 110) (h2 : β = 100) (h3 : γ = 40) :
  ∃ A : Real, A = 30 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_measure_l1007_100791


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1007_100703

theorem solution_set_of_inequality (x : ℝ) :
  (x - 2) / (x + 3) > 0 ↔ x ∈ Set.Ioi (-3) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1007_100703


namespace NUMINAMATH_CALUDE_sqrt_expression_equals_three_l1007_100759

theorem sqrt_expression_equals_three : 
  (Real.sqrt 3 - 2) * Real.sqrt 3 + Real.sqrt 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equals_three_l1007_100759


namespace NUMINAMATH_CALUDE_clothes_to_total_ratio_l1007_100792

def weekly_allowance_1 : ℕ := 5
def weeks_1 : ℕ := 8
def weekly_allowance_2 : ℕ := 6
def weeks_2 : ℕ := 6
def video_game_cost : ℕ := 35
def money_left : ℕ := 3

def total_saved : ℕ := weekly_allowance_1 * weeks_1 + weekly_allowance_2 * weeks_2
def spent_on_video_game_and_left : ℕ := video_game_cost + money_left
def spent_on_clothes : ℕ := total_saved - spent_on_video_game_and_left

theorem clothes_to_total_ratio :
  (spent_on_clothes : ℚ) / total_saved = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_clothes_to_total_ratio_l1007_100792


namespace NUMINAMATH_CALUDE_min_segment_length_in_right_angle_l1007_100752

/-- Given a point inside a 90° angle, located 8 units from one side and 1 unit from the other side,
    the minimum length of a segment passing through this point with ends on the sides of the angle is 10 units. -/
theorem min_segment_length_in_right_angle (P : ℝ × ℝ) 
  (inside_angle : P.1 > 0 ∧ P.2 > 0) 
  (dist_to_sides : P.1 = 1 ∧ P.2 = 8) : 
  Real.sqrt ((P.1 + P.1)^2 + (P.2 + P.2)^2) = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_segment_length_in_right_angle_l1007_100752


namespace NUMINAMATH_CALUDE_g_at_7_equals_neg_20_l1007_100736

/-- A polynomial function g(x) of degree 7 -/
def g (a b c : ℝ) (x : ℝ) : ℝ := a * x^7 - b * x^3 + c * x - 4

/-- Theorem stating that g(7) = -20 given g(-7) = 12 -/
theorem g_at_7_equals_neg_20 (a b c : ℝ) : g a b c (-7) = 12 → g a b c 7 = -20 := by
  sorry

end NUMINAMATH_CALUDE_g_at_7_equals_neg_20_l1007_100736


namespace NUMINAMATH_CALUDE_outside_door_cost_l1007_100713

/-- Proves that the cost of each outside door is $20 -/
theorem outside_door_cost (bedroom_doors : ℕ) (outside_doors : ℕ) (total_cost : ℚ) :
  bedroom_doors = 3 →
  outside_doors = 2 →
  total_cost = 70 →
  ∃ (outside_door_cost : ℚ),
    outside_door_cost * outside_doors + (outside_door_cost / 2) * bedroom_doors = total_cost ∧
    outside_door_cost = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_outside_door_cost_l1007_100713


namespace NUMINAMATH_CALUDE_shirts_per_minute_l1007_100740

/-- Given an industrial machine that makes 12 shirts in 6 minutes,
    prove that it makes 2 shirts per minute. -/
theorem shirts_per_minute :
  let total_shirts : ℕ := 12
  let total_minutes : ℕ := 6
  let shirts_per_minute : ℚ := total_shirts / total_minutes
  shirts_per_minute = 2 := by
  sorry

end NUMINAMATH_CALUDE_shirts_per_minute_l1007_100740


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_neg_105_l1007_100745

/-- A quadratic function with given properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  has_minimum_20 : ∃ x, a * x^2 + b * x + c = 20 ∧ ∀ y, a * y^2 + b * y + c ≥ 20
  root_at_3 : a * 3^2 + b * 3 + c = 0
  root_at_7 : a * 7^2 + b * 7 + c = 0

/-- The sum of coefficients of a quadratic function with given properties is -105 -/
theorem sum_of_coefficients_is_neg_105 (f : QuadraticFunction) : 
  f.a + f.b + f.c = -105 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_neg_105_l1007_100745


namespace NUMINAMATH_CALUDE_roots_and_inequality_l1007_100709

/-- Given the equation ln x - (2a)/(x-1) = a with two distinct real roots -/
def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ Real.log x₁ - (2*a)/(x₁-1) = a ∧ Real.log x₂ - (2*a)/(x₂-1) = a

theorem roots_and_inequality (a : ℝ) (h : has_two_distinct_roots a) :
  a > 0 ∧ ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 1/(Real.log x₁ + a) + 1/(Real.log x₂ + a) < 0 :=
sorry

end NUMINAMATH_CALUDE_roots_and_inequality_l1007_100709


namespace NUMINAMATH_CALUDE_rogers_age_multiple_rogers_age_multiple_is_two_l1007_100733

/-- Proves that the multiple of Jill's age that relates to Roger's age is 2 -/
theorem rogers_age_multiple : ℕ → Prop := fun m =>
  let jill_age : ℕ := 20
  let finley_age : ℕ := 40
  let years_passed : ℕ := 15
  let roger_age : ℕ := m * jill_age + 5
  let jill_future_age : ℕ := jill_age + years_passed
  let roger_future_age : ℕ := roger_age + years_passed
  let finley_future_age : ℕ := finley_age + years_passed
  let future_age_difference : ℕ := roger_future_age - jill_future_age
  (future_age_difference = finley_future_age - 30) → (m = 2)

/-- The theorem holds for m = 2 -/
theorem rogers_age_multiple_is_two : rogers_age_multiple 2 := by
  sorry

end NUMINAMATH_CALUDE_rogers_age_multiple_rogers_age_multiple_is_two_l1007_100733


namespace NUMINAMATH_CALUDE_notebook_cost_l1007_100727

theorem notebook_cost (total_spent : ℝ) (backpack_cost : ℝ) (pens_cost : ℝ) (pencils_cost : ℝ) (num_notebooks : ℕ) :
  total_spent = 32 →
  backpack_cost = 15 →
  pens_cost = 1 →
  pencils_cost = 1 →
  num_notebooks = 5 →
  (total_spent - (backpack_cost + pens_cost + pencils_cost)) / num_notebooks = 3 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l1007_100727


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l1007_100799

/-- The circle C -/
def circle_C (x y : ℝ) : Prop := (x + 4)^2 + y^2 = 100

/-- Point A -/
def point_A : ℝ × ℝ := (4, 0)

/-- The trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

/-- Theorem: The trajectory of the center of a moving circle that is tangent to circle C
    and passes through point A is described by the equation x²/25 + y²/9 = 1 -/
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x' y' : ℝ), (x' - x)^2 + (y' - y)^2 = r^2 → 
      (∃ (x'' y'' : ℝ), circle_C x'' y'' ∧ (x' - x'')^2 + (y' - y'')^2 = 0)) ∧
    ((x - point_A.1)^2 + (y - point_A.2)^2 = r^2)) →
  trajectory x y :=
by sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l1007_100799


namespace NUMINAMATH_CALUDE_candy_distribution_l1007_100798

def candies_for_child (n : ℕ) : ℕ := 2^(n - 1)

def total_candies (n : ℕ) : ℕ := 2^n - 1

theorem candy_distribution (total : ℕ) (h : total = 2007) :
  let n := (Nat.log 2 (total + 1)).succ
  (total_candies n - total, n) = (40, 11) := by sorry

end NUMINAMATH_CALUDE_candy_distribution_l1007_100798


namespace NUMINAMATH_CALUDE_vectors_collinear_necessary_not_sufficient_l1007_100705

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a vector in 3D space
def Vector3D (A B : Point3D) : Point3D :=
  ⟨B.x - A.x, B.y - A.y, B.z - A.z⟩

-- Define collinearity for vectors
def vectorsCollinear (v1 v2 : Point3D) : Prop :=
  ∃ k : ℝ, v1 = ⟨k * v2.x, k * v2.y, k * v2.z⟩

-- Define collinearity for points
def pointsCollinear (A B C D : Point3D) : Prop :=
  ∃ t u v : ℝ, Vector3D A B = ⟨t * (C.x - A.x), t * (C.y - A.y), t * (C.z - A.z)⟩ ∧
               Vector3D A C = ⟨u * (D.x - A.x), u * (D.y - A.y), u * (D.z - A.z)⟩ ∧
               Vector3D A D = ⟨v * (B.x - A.x), v * (B.y - A.y), v * (B.z - A.z)⟩

theorem vectors_collinear_necessary_not_sufficient (A B C D : Point3D) :
  (pointsCollinear A B C D → vectorsCollinear (Vector3D A B) (Vector3D C D)) ∧
  ¬(vectorsCollinear (Vector3D A B) (Vector3D C D) → pointsCollinear A B C D) :=
by sorry

end NUMINAMATH_CALUDE_vectors_collinear_necessary_not_sufficient_l1007_100705


namespace NUMINAMATH_CALUDE_sum_of_i_powers_l1007_100720

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Theorem: Sum of specific powers of i equals 2 -/
theorem sum_of_i_powers : i^24 + i^29 + i^34 + i^39 + i^44 + i^49 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_i_powers_l1007_100720


namespace NUMINAMATH_CALUDE_ship_elevation_change_l1007_100715

/-- The average change in elevation per hour for a ship traveling between Lake Ontario and Lake Erie -/
theorem ship_elevation_change (lake_ontario_elevation lake_erie_elevation : ℝ) (travel_time : ℝ) :
  lake_ontario_elevation = 75 ∧ 
  lake_erie_elevation = 174.28 ∧ 
  travel_time = 8 →
  (lake_erie_elevation - lake_ontario_elevation) / travel_time = 12.41 :=
by sorry

end NUMINAMATH_CALUDE_ship_elevation_change_l1007_100715


namespace NUMINAMATH_CALUDE_intersection_empty_iff_m_range_union_equals_B_iff_m_range_l1007_100725

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}
def B : Set ℝ := {x | x < -6 ∨ x > 1}

-- Theorem for part (I)
theorem intersection_empty_iff_m_range (m : ℝ) :
  A m ∩ B = ∅ ↔ -6 ≤ m ∧ m ≤ 0 :=
sorry

-- Theorem for part (II)
theorem union_equals_B_iff_m_range (m : ℝ) :
  A m ∪ B = B ↔ m < -7 ∨ m > 1 :=
sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_m_range_union_equals_B_iff_m_range_l1007_100725


namespace NUMINAMATH_CALUDE_fertilizer_growth_rate_l1007_100776

theorem fertilizer_growth_rate 
  (april_output : ℝ) 
  (may_decrease : ℝ) 
  (july_output : ℝ) 
  (h1 : april_output = 500)
  (h2 : may_decrease = 0.2)
  (h3 : july_output = 576) :
  ∃ (x : ℝ), 
    april_output * (1 - may_decrease) * (1 + x)^2 = july_output ∧ 
    x = 0.2 :=
sorry

end NUMINAMATH_CALUDE_fertilizer_growth_rate_l1007_100776


namespace NUMINAMATH_CALUDE_equal_color_polygons_l1007_100737

/-- A color type to represent different colors of vertices -/
inductive Color

/-- A structure representing a regular polygon -/
structure RegularPolygon where
  vertices : Finset ℝ × ℝ
  is_regular : Bool

/-- A structure representing a colored regular n-gon -/
structure ColoredRegularNGon where
  n : ℕ
  vertices : Finset (ℝ × ℝ)
  colors : Finset Color
  vertex_coloring : (ℝ × ℝ) → Color
  is_regular : Bool
  num_vertices : vertices.card = n

/-- A function that returns the set of regular polygons formed by vertices of each color -/
def colorPolygons (ngon : ColoredRegularNGon) : Finset RegularPolygon :=
  sorry

/-- The main theorem statement -/
theorem equal_color_polygons (ngon : ColoredRegularNGon) :
  ∃ (p q : RegularPolygon), p ∈ colorPolygons ngon ∧ q ∈ colorPolygons ngon ∧ p ≠ q ∧ p.vertices = q.vertices :=
sorry

end NUMINAMATH_CALUDE_equal_color_polygons_l1007_100737


namespace NUMINAMATH_CALUDE_initial_amount_equation_l1007_100794

/-- The initial amount Kanul had, in dollars -/
def initial_amount : ℝ := 7058.82

/-- The amount spent on raw materials, in dollars -/
def raw_materials : ℝ := 3000

/-- The amount spent on machinery, in dollars -/
def machinery : ℝ := 2000

/-- The percentage of the initial amount spent as cash -/
def cash_percentage : ℝ := 0.15

/-- The amount spent on labor costs, in dollars -/
def labor_costs : ℝ := 1000

/-- Theorem stating that the initial amount satisfies the equation -/
theorem initial_amount_equation :
  initial_amount = raw_materials + machinery + cash_percentage * initial_amount + labor_costs := by
  sorry

end NUMINAMATH_CALUDE_initial_amount_equation_l1007_100794


namespace NUMINAMATH_CALUDE_dennis_lives_on_sixth_floor_l1007_100732

def frank_floor : ℕ := 16

def charlie_floor (frank : ℕ) : ℕ := frank / 4

def dennis_floor (charlie : ℕ) : ℕ := charlie + 2

theorem dennis_lives_on_sixth_floor :
  dennis_floor (charlie_floor frank_floor) = 6 := by
  sorry

end NUMINAMATH_CALUDE_dennis_lives_on_sixth_floor_l1007_100732


namespace NUMINAMATH_CALUDE_slope_value_l1007_100783

/-- The slope of a line passing through a focus of the ellipse x^2 + 2y^2 = 3 
    and intersecting it at two points with distance 2 apart. -/
def slope_through_focus (k : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ),
    -- A and B are on the ellipse
    A.1^2 + 2*A.2^2 = 3 ∧ B.1^2 + 2*B.2^2 = 3 ∧
    -- The line passes through a focus
    ∃ (x : ℝ), x^2 = 3/2 ∧ (A.2 - 0) = k * (A.1 - x) ∧ (B.2 - 0) = k * (B.1 - x) ∧
    -- The distance between A and B is 2
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4

/-- The theorem stating the absolute value of the slope -/
theorem slope_value : ∀ k : ℝ, slope_through_focus k → k^2 = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_slope_value_l1007_100783


namespace NUMINAMATH_CALUDE_vector_parallel_solution_l1007_100712

/-- Represents a 2D vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Check if two vectors are parallel -/
def parallel (v w : Vec2D) : Prop :=
  ∃ (k : ℝ), v.x * w.y = k * v.y * w.x

theorem vector_parallel_solution (m : ℝ) : 
  let a : Vec2D := ⟨1, m⟩
  let b : Vec2D := ⟨2, 5⟩
  let c : Vec2D := ⟨m, 3⟩
  parallel (Vec2D.mk (a.x + c.x) (a.y + c.y)) (Vec2D.mk (a.x - b.x) (a.y - b.y)) →
  m = (3 + Real.sqrt 17) / 2 ∨ m = (3 - Real.sqrt 17) / 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_solution_l1007_100712


namespace NUMINAMATH_CALUDE_intersection_union_ratio_l1007_100739

/-- A rhombus with given diagonal lengths -/
structure Rhombus where
  short_diagonal : ℝ
  long_diagonal : ℝ

/-- The rotation of a rhombus by 90 degrees -/
def rotate_90 (r : Rhombus) : Rhombus := r

/-- The intersection of a rhombus and its 90 degree rotation -/
def intersection (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The union of a rhombus and its 90 degree rotation -/
def union (r : Rhombus) : Set (ℝ × ℝ) := sorry

/-- The area of a set in 2D space -/
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

/-- Theorem: The ratio of the intersection area to the union area is 1/2023 -/
theorem intersection_union_ratio (r : Rhombus) 
  (h1 : r.short_diagonal = 1) 
  (h2 : r.long_diagonal = 2023) : 
  area (intersection r) / area (union r) = 1 / 2023 := by sorry

end NUMINAMATH_CALUDE_intersection_union_ratio_l1007_100739
