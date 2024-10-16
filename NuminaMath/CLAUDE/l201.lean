import Mathlib

namespace NUMINAMATH_CALUDE_square_from_relation_l201_20139

theorem square_from_relation (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b^2 = a^2 + a*b + b) : 
  ∃ k : ℕ, k > 0 ∧ b = k^2 := by
  sorry

end NUMINAMATH_CALUDE_square_from_relation_l201_20139


namespace NUMINAMATH_CALUDE_gold_award_winners_possibly_all_freshmen_l201_20149

theorem gold_award_winners_possibly_all_freshmen 
  (total_winners : ℕ) 
  (selected_students : ℕ) 
  (selected_freshmen : ℕ) 
  (selected_gold : ℕ) 
  (h1 : total_winners = 120)
  (h2 : selected_students = 24)
  (h3 : selected_freshmen = 6)
  (h4 : selected_gold = 4) :
  ∃ (total_freshmen : ℕ) (total_gold : ℕ),
    total_freshmen ≤ total_winners ∧
    total_gold ≤ total_winners ∧
    total_gold ≤ total_freshmen :=
by sorry

end NUMINAMATH_CALUDE_gold_award_winners_possibly_all_freshmen_l201_20149


namespace NUMINAMATH_CALUDE_smallest_butterfly_count_l201_20183

theorem smallest_butterfly_count (n : ℕ) : n > 0 → (
  (∃ m k : ℕ, m > 0 ∧ k > 0 ∧ n * 44 = m * 17 ∧ n * 44 = k * 25) ∧
  (∃ t : ℕ, n * 44 + n * 17 + n * 25 = 60 * t) ∧
  (∀ x : ℕ, x > 0 ∧ x < n → 
    ¬(∃ y z : ℕ, y > 0 ∧ z > 0 ∧ x * 44 = y * 17 ∧ x * 44 = z * 25) ∨
    ¬(∃ s : ℕ, x * 44 + x * 17 + x * 25 = 60 * s))
) ↔ n = 425 := by
  sorry

end NUMINAMATH_CALUDE_smallest_butterfly_count_l201_20183


namespace NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l201_20162

theorem rectangular_parallelepiped_volume 
  (x y z : ℝ) 
  (h1 : (x^2 + y^2) * z^2 = 13) 
  (h2 : (y^2 + z^2) * x^2 = 40) 
  (h3 : (x^2 + z^2) * y^2 = 45) : 
  x * y * z = 6 := by
sorry

end NUMINAMATH_CALUDE_rectangular_parallelepiped_volume_l201_20162


namespace NUMINAMATH_CALUDE_marys_bag_check_time_l201_20175

/-- Represents the time in minutes for Mary's trip to the airport -/
structure AirportTrip where
  uberToHouse : ℕ
  uberToAirport : ℕ
  bagCheck : ℕ
  security : ℕ
  waitForBoarding : ℕ
  waitForTakeoff : ℕ

/-- The total trip time in minutes -/
def totalTripTime (trip : AirportTrip) : ℕ :=
  trip.uberToHouse + trip.uberToAirport + trip.bagCheck + trip.security + trip.waitForBoarding + trip.waitForTakeoff

/-- Mary's airport trip satisfies the given conditions -/
def marysTrip (trip : AirportTrip) : Prop :=
  trip.uberToHouse = 10 ∧
  trip.uberToAirport = 5 * trip.uberToHouse ∧
  trip.security = 3 * trip.bagCheck ∧
  trip.waitForBoarding = 20 ∧
  trip.waitForTakeoff = 2 * trip.waitForBoarding ∧
  totalTripTime trip = 180  -- 3 hours in minutes

theorem marys_bag_check_time (trip : AirportTrip) (h : marysTrip trip) : trip.bagCheck = 15 := by
  sorry

end NUMINAMATH_CALUDE_marys_bag_check_time_l201_20175


namespace NUMINAMATH_CALUDE_abs_eq_sqrt_square_l201_20194

theorem abs_eq_sqrt_square (x : ℝ) : |x| = Real.sqrt (x^2) := by sorry

end NUMINAMATH_CALUDE_abs_eq_sqrt_square_l201_20194


namespace NUMINAMATH_CALUDE_total_stamps_l201_20193

theorem total_stamps (harry_stamps : ℕ) (sister_stamps : ℕ) : 
  harry_stamps = 180 → sister_stamps = 60 → harry_stamps + sister_stamps = 240 :=
by
  sorry

end NUMINAMATH_CALUDE_total_stamps_l201_20193


namespace NUMINAMATH_CALUDE_distinct_colorings_count_l201_20176

/-- Represents the symmetries of a regular octagon -/
inductive OctagonSymmetry
| Identity
| Reflection (n : Fin 8)
| Rotation (n : Fin 4)

/-- Represents a coloring of 8 disks -/
def Coloring := Fin 8 → Fin 3

/-- The number of disks -/
def n : ℕ := 8

/-- The number of colors -/
def k : ℕ := 3

/-- The number of each color -/
def colorCounts : Fin 3 → ℕ
| 0 => 4  -- blue
| 1 => 3  -- red
| 2 => 1  -- green
| _ => 0  -- unreachable

/-- The set of all possible colorings -/
def allColorings : Finset Coloring := sorry

/-- Whether a coloring is fixed by a given symmetry -/
def isFixed (c : Coloring) (s : OctagonSymmetry) : Prop := sorry

/-- The number of colorings fixed by each symmetry -/
def fixedColorings (s : OctagonSymmetry) : ℕ := sorry

/-- The set of all symmetries -/
def symmetries : Finset OctagonSymmetry := sorry

/-- The main theorem: the number of distinct colorings is 21 -/
theorem distinct_colorings_count :
  (Finset.sum symmetries fixedColorings) / Finset.card symmetries = 21 := sorry

end NUMINAMATH_CALUDE_distinct_colorings_count_l201_20176


namespace NUMINAMATH_CALUDE_standard_equation_min_area_OPQ_l201_20154

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the right focus F
def right_focus (a b : ℝ) : Prop := a^2 - b^2 = 1

-- Define the perpendicular condition
def perpendicular_condition (a b : ℝ) : Prop := b = 1

-- Theorem for the standard equation of the ellipse
theorem standard_equation (a b : ℝ) 
  (h1 : ellipse x y a b) 
  (h2 : right_focus a b) 
  (h3 : perpendicular_condition a b) : 
  x^2 / 2 + y^2 = 1 := by sorry

-- Define the triangle OPQ
def triangle_OPQ (x y m : ℝ) : Prop := 
  x^2 / 2 + y^2 = 1 ∧ 
  ∃ (P : ℝ × ℝ), P.2 = 2 ∧ 
  (P.1 * y = 2 * x ∨ (P.1 = 0 ∧ x = Real.sqrt 2))

-- Theorem for the minimum area of triangle OPQ
theorem min_area_OPQ (x y m : ℝ) 
  (h : triangle_OPQ x y m) : 
  ∃ (S : ℝ), S ≥ 1 ∧ 
  (∀ (S' : ℝ), triangle_OPQ x y m → S' ≥ S) := by sorry

end NUMINAMATH_CALUDE_standard_equation_min_area_OPQ_l201_20154


namespace NUMINAMATH_CALUDE_advance_agency_fees_calculation_l201_20155

/-- Proof of advance agency fees calculation -/
theorem advance_agency_fees_calculation 
  (C : ℕ) -- Commission
  (I : ℕ) -- Incentive
  (G : ℕ) -- Amount given to John
  (h1 : C = 25000)
  (h2 : I = 1780)
  (h3 : G = 18500)
  : C + I - G = 8280 := by
  sorry

end NUMINAMATH_CALUDE_advance_agency_fees_calculation_l201_20155


namespace NUMINAMATH_CALUDE_no_valid_integers_l201_20179

theorem no_valid_integers : ¬∃ (n : ℤ), ∃ (y : ℤ), 
  (n^2 - 21*n + 110 = y^2) ∧ (∃ (k : ℤ), n = 4*k) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_integers_l201_20179


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_range_l201_20100

theorem quadratic_roots_imply_a_range (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ 
   x^2 + 2*(a-1)*x + 2*a + 6 = 0 ∧
   y^2 + 2*(a-1)*y + 2*a + 6 = 0) →
  a < -3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_range_l201_20100


namespace NUMINAMATH_CALUDE_special_number_fraction_l201_20137

theorem special_number_fraction (list : List ℝ) (n : ℝ) :
  list.length = 21 ∧
  n ∉ list ∧
  n = 5 * (list.sum / list.length) →
  n / (list.sum + n) = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_special_number_fraction_l201_20137


namespace NUMINAMATH_CALUDE_equidistant_sum_constant_sum_of_terms_l201_20191

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem stating the sum of equidistant terms in an arithmetic sequence is constant -/
theorem equidistant_sum_constant {a : ℕ → ℝ} (h : arithmetic_sequence a) :
  ∀ n k : ℕ, a n + a (n + k) = a (n - 1) + a (n + k + 1) :=
sorry

theorem sum_of_terms (a : ℕ → ℝ) (h : arithmetic_sequence a) 
  (h_sum : a 3 + a 7 = 37) : 
  a 2 + a 4 + a 6 + a 8 = 74 :=
sorry

end NUMINAMATH_CALUDE_equidistant_sum_constant_sum_of_terms_l201_20191


namespace NUMINAMATH_CALUDE_sequence_sum_l201_20116

theorem sequence_sum (A B C D E F G H I J : ℝ) : 
  D = 8 →
  A + B + C + D = 45 →
  B + C + D + E = 45 →
  C + D + E + F = 45 →
  D + E + F + G = 45 →
  E + F + G + H = 45 →
  F + G + H + I = 45 →
  G + H + I + J = 45 →
  A + J = 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_l201_20116


namespace NUMINAMATH_CALUDE_system_of_equations_l201_20174

theorem system_of_equations (x y k : ℝ) 
  (eq1 : 3 * x + 4 * y = k + 2)
  (eq2 : 2 * x + y = 4)
  (eq3 : x + y = 2) : k = 4 := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_l201_20174


namespace NUMINAMATH_CALUDE_quadratic_solution_correctness_l201_20109

/-- Solutions to the quadratic equation (p+1)x² - 2px + p - 2 = 0 --/
def quadratic_solutions (p : ℝ) : Set ℝ :=
  if p = -1 then
    {3/2}
  else if p > -2 then
    {(p + Real.sqrt (p+2)) / (p+1), (p - Real.sqrt (p+2)) / (p+1)}
  else if p = -2 then
    {2}
  else
    ∅

/-- The quadratic equation (p+1)x² - 2px + p - 2 = 0 --/
def quadratic_equation (p x : ℝ) : Prop :=
  (p+1) * x^2 - 2*p*x + p - 2 = 0

theorem quadratic_solution_correctness (p : ℝ) :
  ∀ x, x ∈ quadratic_solutions p ↔ quadratic_equation p x :=
sorry

end NUMINAMATH_CALUDE_quadratic_solution_correctness_l201_20109


namespace NUMINAMATH_CALUDE_school_test_questions_l201_20143

theorem school_test_questions (sections : ℕ) (correct_answers : ℕ) 
  (h_sections : sections = 5)
  (h_correct : correct_answers = 20)
  (h_percentage : ∀ x : ℕ, x > 0 → (60 : ℚ) / 100 < (correct_answers : ℚ) / x ∧ (correct_answers : ℚ) / x < 70 / 100 → x = 30) :
  ∃! total_questions : ℕ, 
    total_questions > 0 ∧
    total_questions % sections = 0 ∧
    (60 : ℚ) / 100 < (correct_answers : ℚ) / total_questions ∧
    (correct_answers : ℚ) / total_questions < 70 / 100 :=
by
  sorry

end NUMINAMATH_CALUDE_school_test_questions_l201_20143


namespace NUMINAMATH_CALUDE_obtuse_triangle_side_range_l201_20125

-- Define the triangle
def Triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

-- Define an obtuse triangle
def ObtuseTriangle (a b c : ℝ) : Prop :=
  Triangle a b c ∧ (a^2 + b^2 < c^2 ∨ b^2 + c^2 < a^2 ∨ c^2 + a^2 < b^2)

-- Theorem statement
theorem obtuse_triangle_side_range :
  ∀ c : ℝ, ObtuseTriangle 4 3 c → c ∈ Set.Ioo 1 (Real.sqrt 7) ∪ Set.Ioo 5 7 :=
by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_side_range_l201_20125


namespace NUMINAMATH_CALUDE_rectangle_shading_convergence_l201_20161

theorem rectangle_shading_convergence :
  let initial_shaded : ℚ := 1/2
  let subsequent_shading_ratio : ℚ := 1/16
  let shaded_series : ℕ → ℚ := λ n => initial_shaded * subsequent_shading_ratio^n
  let total_shaded : ℚ := ∑' n, shaded_series n
  total_shaded = 17/30 := by
sorry

end NUMINAMATH_CALUDE_rectangle_shading_convergence_l201_20161


namespace NUMINAMATH_CALUDE_a_1000_equals_divisors_of_1000_l201_20140

/-- A sequence of real numbers satisfying the given power series equality -/
def PowerSeriesSequence (a : ℕ → ℝ) : Prop :=
  ∀ x : ℝ, abs x < 1 →
    (∑' n : ℕ, x^n / (1 - x^n)) = ∑' i : ℕ, a i * x^i

/-- The number of divisors of a natural number -/
def numberOfDivisors (n : ℕ) : ℕ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem a_1000_equals_divisors_of_1000 (a : ℕ → ℝ) (h : PowerSeriesSequence a) :
    a 1000 = numberOfDivisors 1000 := by
  sorry

end NUMINAMATH_CALUDE_a_1000_equals_divisors_of_1000_l201_20140


namespace NUMINAMATH_CALUDE_inequality_proof_l201_20128

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > a*b ∧ a*b > b^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l201_20128


namespace NUMINAMATH_CALUDE_triangle_with_arithmetic_angles_and_side_ratio_l201_20178

theorem triangle_with_arithmetic_angles_and_side_ratio (α β γ : Real) (a b c : Real) :
  -- Angles form an arithmetic progression
  β - α = γ - β →
  -- Sum of angles in a triangle is 180°
  α + β + γ = 180 →
  -- Smallest side is half of the largest side
  a = c / 2 →
  -- Side lengths satisfy the sine law
  a / Real.sin α = b / Real.sin β →
  b / Real.sin β = c / Real.sin γ →
  -- Angles are positive
  0 < α ∧ 0 < β ∧ 0 < γ →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Prove that the angles are 30°, 60°, and 90°
  (α = 30 ∧ β = 60 ∧ γ = 90) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_arithmetic_angles_and_side_ratio_l201_20178


namespace NUMINAMATH_CALUDE_game_theory_proof_l201_20131

theorem game_theory_proof (x y : ℝ) : 
  (x + y + (24 - x - y) = 24) →
  (2*x - 24 = 2) →
  (4*y - 24 = 4) →
  (∀ (a b c : ℝ), (a + b + c = 24) → (a = 8 ∧ b = 8 ∧ c = 8)) →
  (x = 13 ∧ y = 7 ∧ 24 - x - y = 4) :=
by sorry

end NUMINAMATH_CALUDE_game_theory_proof_l201_20131


namespace NUMINAMATH_CALUDE_problem_statement_l201_20181

theorem problem_statement (a b : ℝ) : 
  let M := {b/a, 1}
  let N := {a, 0}
  (∃ f : ℝ → ℝ, f = id ∧ f '' M ⊆ N) →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l201_20181


namespace NUMINAMATH_CALUDE_ellipse_intersection_problem_l201_20124

-- Define the ellipse
def ellipse (b : ℝ) (x y : ℝ) : Prop := x^2/4 + y^2/b^2 = 1

-- Define the line
def line (t : ℝ) (x y : ℝ) : Prop := x + t*y - 1 = 0

-- Define the triangle area
def triangle_area (A B M : ℝ × ℝ) : ℝ := sorry

-- Define the theorem
theorem ellipse_intersection_problem (b : ℝ) (h1 : 0 < b) (h2 : b < 2) :
  -- Given M(1,1) on the ellipse
  ellipse b 1 1 →
  -- Part 1: When t = 1, area of triangle ABM is √13/4
  (∃ A B : ℝ × ℝ, ellipse b A.1 A.2 ∧ ellipse b B.1 B.2 ∧ 
   line 1 A.1 A.2 ∧ line 1 B.1 B.2 ∧ 
   triangle_area A B (1, 1) = Real.sqrt 13 / 4) ∧
  -- Part 2: When S_PQM = 5S_ABM, t = ±(3√2)/2
  (∃ t : ℝ, ∃ A B P Q : ℝ × ℝ, 
   ellipse b A.1 A.2 ∧ ellipse b B.1 B.2 ∧
   line t A.1 A.2 ∧ line t B.1 B.2 ∧
   P.1 = 4 ∧ Q.1 = 4 ∧
   triangle_area P Q (1, 1) = 5 * triangle_area A B (1, 1) →
   t = 3 * Real.sqrt 2 / 2 ∨ t = -3 * Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_ellipse_intersection_problem_l201_20124


namespace NUMINAMATH_CALUDE_rectangle_length_reduction_l201_20136

theorem rectangle_length_reduction (original_length original_width : ℝ) 
  (h : original_length > 0 ∧ original_width > 0) :
  let new_width := original_width * (1 + 0.4285714285714287)
  let new_length := original_length * 0.7
  original_length * original_width = new_length * new_width :=
by
  sorry

#check rectangle_length_reduction

end NUMINAMATH_CALUDE_rectangle_length_reduction_l201_20136


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l201_20132

theorem buckingham_palace_visitors 
  (total_visitors : ℕ) 
  (previous_day_visitors : ℕ) 
  (today_visitors : ℕ) 
  (h1 : total_visitors = 949) 
  (h2 : previous_day_visitors = 703) 
  (h3 : today_visitors > 0) 
  (h4 : total_visitors = previous_day_visitors + today_visitors) : 
  today_visitors = 246 := by
sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l201_20132


namespace NUMINAMATH_CALUDE_digit_sum_problem_l201_20106

theorem digit_sum_problem (P Q : ℕ) : 
  P < 10 → Q < 10 → 77 * P + 77 * Q = 1000 * P + 100 * P + 10 * P + 7 → P + Q = 14 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l201_20106


namespace NUMINAMATH_CALUDE_function_equality_l201_20101

theorem function_equality (f : ℝ → ℝ) 
  (h1 : ∀ x, f x ≤ x) 
  (h2 : ∀ x y, f (x + y) ≤ f x + f y) : 
  ∀ x, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l201_20101


namespace NUMINAMATH_CALUDE_distinct_terms_expansion_l201_20114

/-- The number of distinct terms in the expansion of (a+b+c)(x+y+z+w+t) -/
def distinct_terms (a b c x y z w t : ℝ) : ℕ :=
  3 * 5

theorem distinct_terms_expansion (a b c x y z w t : ℝ) 
  (h_diff : a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ t ∧ 
            y ≠ z ∧ y ≠ w ∧ y ≠ t ∧ z ≠ w ∧ z ≠ t ∧ w ≠ t) : 
  distinct_terms a b c x y z w t = 15 := by
  sorry

end NUMINAMATH_CALUDE_distinct_terms_expansion_l201_20114


namespace NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l201_20102

/-- A system of linear equations parameterized by n -/
def LinearSystem (n : ℝ) :=
  ∃ (x y z : ℝ), (n * x + y = 1) ∧ ((1/2) * n * y + z = 1) ∧ (x + (1/2) * n * z = 2)

/-- The theorem stating that the system has no solution if and only if n = -1 -/
theorem no_solution_iff_n_eq_neg_one :
  ∀ n : ℝ, ¬(LinearSystem n) ↔ n = -1 := by sorry

end NUMINAMATH_CALUDE_no_solution_iff_n_eq_neg_one_l201_20102


namespace NUMINAMATH_CALUDE_unique_a_satisfies_condition_l201_20188

/-- Converts a base-25 number to its decimal representation modulo 12 -/
def base25ToDecimalMod12 (digits : List Nat) : Nat :=
  (digits.reverse.enum.map (fun (i, d) => d * (25^i % 12)) |>.sum) % 12

/-- The given number in base 25 -/
def number : List Nat := [3, 1, 4, 2, 6, 5, 2, 3]

theorem unique_a_satisfies_condition :
  ∃! a : ℕ, 0 ≤ a ∧ a ≤ 14 ∧ (base25ToDecimalMod12 number - a) % 12 = 0 ∧ a = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_a_satisfies_condition_l201_20188


namespace NUMINAMATH_CALUDE_tangent_line_equation_l201_20113

noncomputable def f (x : ℝ) : ℝ := x - Real.cos x

theorem tangent_line_equation :
  let p : ℝ × ℝ := (π / 2, π / 2)
  let m : ℝ := 1 + Real.sin (π / 2)
  let tangent_eq (x y : ℝ) : Prop := 2 * x - y - π / 2 = 0
  tangent_eq (p.1) (p.2) ∧
  ∀ x y : ℝ, tangent_eq x y ↔ y - p.2 = m * (x - p.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l201_20113


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l201_20147

/-- Calculates the time it takes for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 10)
  (h2 : train_speed = 46)
  (h3 : train_length = 120)
  (h4 : initial_distance = 340)
  : (initial_distance + train_length) / (train_speed - jogger_speed) * (3600 / 1000) = 46 := by
  sorry

#check train_passing_jogger_time

end NUMINAMATH_CALUDE_train_passing_jogger_time_l201_20147


namespace NUMINAMATH_CALUDE_price_tags_offer_advantages_l201_20186

/-- Represents a product in a store -/
structure Product where
  name : String
  price : ℝ

/-- Represents a store with a collection of products -/
structure Store where
  products : List Product
  has_price_tags : Bool

/-- Represents the advantages of using price tags -/
structure PriceTagAdvantages where
  simplifies_purchase : Bool
  reduces_personnel_requirement : Bool
  provides_advertising : Bool
  increases_trust : Bool

/-- Theorem stating that attaching price tags to all products offers advantages -/
theorem price_tags_offer_advantages (store : Store) (h : store.has_price_tags = true) :
  ∃ (advantages : PriceTagAdvantages),
    advantages.simplifies_purchase ∧
    advantages.reduces_personnel_requirement ∧
    advantages.provides_advertising ∧
    advantages.increases_trust :=
  sorry

end NUMINAMATH_CALUDE_price_tags_offer_advantages_l201_20186


namespace NUMINAMATH_CALUDE_fifth_stack_cups_l201_20123

def cup_sequence : ℕ → ℕ
  | 0 => 17
  | 1 => 21
  | 2 => 25
  | 3 => 29
  | n + 4 => cup_sequence n + 4 * 4

theorem fifth_stack_cups : cup_sequence 4 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fifth_stack_cups_l201_20123


namespace NUMINAMATH_CALUDE_suzanna_bike_ride_l201_20199

/-- Calculates the distance traveled given a constant speed and time -/
def distance_traveled (speed : ℚ) (time : ℚ) : ℚ :=
  speed * time

/-- Represents Suzanna's bike ride -/
theorem suzanna_bike_ride (speed : ℚ) (time : ℚ) (h1 : speed = 1 / 6) (h2 : time = 40) :
  distance_traveled speed time = 6 := by
  sorry

#check suzanna_bike_ride

end NUMINAMATH_CALUDE_suzanna_bike_ride_l201_20199


namespace NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l201_20190

theorem largest_number_with_equal_quotient_and_remainder (A B C : ℕ) 
  (h1 : A = 8 * B + C) 
  (h2 : B = C) 
  (h3 : C < 8) : 
  A ≤ 63 ∧ ∃ (A' : ℕ), A' = 63 ∧ ∃ (B' C' : ℕ), A' = 8 * B' + C' ∧ B' = C' ∧ C' < 8 :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_equal_quotient_and_remainder_l201_20190


namespace NUMINAMATH_CALUDE_gcd_168_486_l201_20111

theorem gcd_168_486 : Nat.gcd 168 486 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_168_486_l201_20111


namespace NUMINAMATH_CALUDE_rock_collection_total_l201_20129

theorem rock_collection_total (igneous sedimentary metamorphic : ℕ) : 
  sedimentary = 2 * igneous →
  metamorphic = 2 * igneous →
  40 = (2 * igneous) / 3 →
  igneous + sedimentary + metamorphic = 300 :=
by
  sorry

#check rock_collection_total

end NUMINAMATH_CALUDE_rock_collection_total_l201_20129


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l201_20105

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l201_20105


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l201_20135

theorem reciprocal_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l201_20135


namespace NUMINAMATH_CALUDE_tree_spacing_l201_20151

/-- Given a yard of length 300 meters with 26 equally spaced trees, including one at each end,
    the distance between consecutive trees is 12 meters. -/
theorem tree_spacing (yard_length : ℝ) (num_trees : ℕ) (tree_spacing : ℝ) : 
  yard_length = 300 →
  num_trees = 26 →
  tree_spacing * (num_trees - 1) = yard_length →
  tree_spacing = 12 := by
  sorry

end NUMINAMATH_CALUDE_tree_spacing_l201_20151


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l201_20117

theorem fraction_equation_solution :
  ∃ x : ℚ, (x + 5) / (x - 3) = (x - 4) / (x + 2) ↔ x = 1 / 7 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l201_20117


namespace NUMINAMATH_CALUDE_lassie_bones_l201_20169

theorem lassie_bones (initial_bones : ℕ) : 
  (initial_bones / 2 + 10 = 35) → initial_bones = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_lassie_bones_l201_20169


namespace NUMINAMATH_CALUDE_polynomial_simplification_l201_20156

-- Define the left-hand side of the equation
def lhs (p : ℝ) : ℝ := (7*p^5 - 4*p^3 + 8*p^2 - 5*p + 3) + (-p^5 + 3*p^3 - 7*p^2 + 6*p + 2)

-- Define the right-hand side of the equation
def rhs (p : ℝ) : ℝ := 6*p^5 - p^3 + p^2 + p + 5

-- Theorem statement
theorem polynomial_simplification (p : ℝ) : lhs p = rhs p := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l201_20156


namespace NUMINAMATH_CALUDE_janette_breakfast_jerky_l201_20126

/-- The number of days Janette went camping -/
def camping_days : ℕ := 5

/-- The initial number of beef jerky pieces Janette brought -/
def initial_jerky : ℕ := 40

/-- The number of beef jerky pieces Janette eats for lunch each day -/
def lunch_jerky : ℕ := 1

/-- The number of beef jerky pieces Janette eats for dinner each day -/
def dinner_jerky : ℕ := 2

/-- The number of beef jerky pieces Janette has left after giving half to her brother -/
def final_jerky : ℕ := 10

/-- The number of beef jerky pieces Janette eats for breakfast each day -/
def breakfast_jerky : ℕ := 1

theorem janette_breakfast_jerky :
  breakfast_jerky = 1 ∧
  camping_days * (breakfast_jerky + lunch_jerky + dinner_jerky) = initial_jerky - 2 * final_jerky :=
by sorry

end NUMINAMATH_CALUDE_janette_breakfast_jerky_l201_20126


namespace NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l201_20115

-- Part 1
theorem inequality_solution_1 (x : ℝ) :
  (x + 1) / (x - 2) ≥ 3 ↔ 2 < x ∧ x ≤ 7/2 :=
sorry

-- Part 2
theorem inequality_solution_2 (x a : ℝ) :
  x^2 - a*x - 2*a^2 ≤ 0 ↔
    (a = 0 ∧ x = 0) ∨
    (a > 0 ∧ -a ≤ x ∧ x ≤ 2*a) ∨
    (a < 0 ∧ 2*a ≤ x ∧ x ≤ -a) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_1_inequality_solution_2_l201_20115


namespace NUMINAMATH_CALUDE_average_median_relation_l201_20119

theorem average_median_relation (a b c : ℤ) : 
  (a + b + c) / 3 = 4 * b →
  a < b →
  b < c →
  a = 0 →
  c / b = 11 := by
  sorry

end NUMINAMATH_CALUDE_average_median_relation_l201_20119


namespace NUMINAMATH_CALUDE_volleyball_prob_l201_20167

-- Define the set of sports
inductive Sport
| Soccer
| Basketball
| Volleyball

-- Define the probability space
def sportProbabilitySpace : Type := Sport

-- Define the probability measure
axiom prob : sportProbabilitySpace → ℝ

-- Axioms for probability measure
axiom prob_nonneg : ∀ s : sportProbabilitySpace, 0 ≤ prob s
axiom prob_sum_one : (prob Sport.Soccer) + (prob Sport.Basketball) + (prob Sport.Volleyball) = 1

-- Axiom for equal probability of each sport
axiom equal_prob : prob Sport.Soccer = prob Sport.Basketball ∧ 
                   prob Sport.Basketball = prob Sport.Volleyball

-- Theorem: The probability of choosing volleyball is 1/3
theorem volleyball_prob : prob Sport.Volleyball = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_prob_l201_20167


namespace NUMINAMATH_CALUDE_square_of_complex_number_l201_20198

theorem square_of_complex_number :
  let z : ℂ := 5 - 2 * Complex.I
  z * z = 21 - 20 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_square_of_complex_number_l201_20198


namespace NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l201_20146

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℕ → ℝ
  | 0 => a₁
  | k + 1 => a₁ + k * d

theorem third_term_of_arithmetic_sequence :
  let a₁ : ℝ := 11
  let a₆ : ℝ := 39
  let n : ℕ := 6
  let d : ℝ := (a₆ - a₁) / (n - 1)
  arithmetic_sequence a₁ d n 2 = 22.2 := by
  sorry

end NUMINAMATH_CALUDE_third_term_of_arithmetic_sequence_l201_20146


namespace NUMINAMATH_CALUDE_sum_of_variables_l201_20160

theorem sum_of_variables (x y z : ℝ) 
  (eq1 : y + z = 10 - 2*x)
  (eq2 : x + z = -12 - 4*y)
  (eq3 : x + y = 5 - 2*z) :
  2*x + 2*y + 2*z = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_variables_l201_20160


namespace NUMINAMATH_CALUDE_converse_not_true_l201_20127

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (perp : Line → Plane → Prop)
variable (plane_perp : Plane → Plane → Prop)

-- State the theorem
theorem converse_not_true :
  ∃ (b : Line) (α β : Plane),
    (subset b β ∧ perp b α) ∧ ¬(plane_perp β α) :=
sorry

end NUMINAMATH_CALUDE_converse_not_true_l201_20127


namespace NUMINAMATH_CALUDE_band_total_earnings_l201_20177

/-- Calculates the total earnings of a band given the number of members, 
    earnings per member per gig, and number of gigs played. -/
def bandEarnings (members : ℕ) (earningsPerMember : ℕ) (gigs : ℕ) : ℕ :=
  members * earningsPerMember * gigs

/-- Theorem stating that a band with 4 members, each earning $20 per gig, 
    and having played 5 gigs, earns a total of $400. -/
theorem band_total_earnings : 
  bandEarnings 4 20 5 = 400 := by
  sorry

end NUMINAMATH_CALUDE_band_total_earnings_l201_20177


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l201_20130

theorem arithmetic_calculations :
  (-9 + 5 * (-6) - 18 / (-3) = -33) ∧
  ((-3/4 - 5/8 + 9/12) * (-24) + (-8) / (2/3) = 3) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l201_20130


namespace NUMINAMATH_CALUDE_trailing_zeros_remainder_l201_20120

-- Define the factorial function
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

-- Define the product of factorials from 1 to 120
def productOfFactorials : ℕ := (List.range 120).foldl (λ acc i => acc * factorial (i + 1)) 1

-- Define the function to count trailing zeros
def trailingZeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + trailingZeros (n / 10)
  else 0

-- Theorem statement
theorem trailing_zeros_remainder :
  (trailingZeros productOfFactorials) % 1000 = 224 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_remainder_l201_20120


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l201_20148

/-- The area of a square with vertices P(2, 3), Q(-3, 4), R(-2, -1), and S(3, 0) is 26 square units -/
theorem square_area_from_vertices : 
  let P : ℝ × ℝ := (2, 3)
  let Q : ℝ × ℝ := (-3, 4)
  let R : ℝ × ℝ := (-2, -1)
  let S : ℝ × ℝ := (3, 0)
  let square_area := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)^2
  square_area = 26 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l201_20148


namespace NUMINAMATH_CALUDE_point_on_k_graph_l201_20159

-- Define the functions f and k
variable (f : ℝ → ℝ)
variable (k : ℝ → ℝ)

-- State the theorem
theorem point_on_k_graph (h1 : f 4 = 8) (h2 : ∀ x, k x = (f x)^3) :
  ∃ x y : ℝ, k x = y ∧ x + y = 516 := by
sorry

end NUMINAMATH_CALUDE_point_on_k_graph_l201_20159


namespace NUMINAMATH_CALUDE_parabola_vertex_l201_20189

/-- The vertex of a parabola is the point where it turns. For a parabola with equation
    y² + 8y + 2x + 11 = 0, this theorem states that the vertex is (5/2, -4). -/
theorem parabola_vertex (x y : ℝ) : 
  y^2 + 8*y + 2*x + 11 = 0 → (x = 5/2 ∧ y = -4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l201_20189


namespace NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l201_20122

theorem no_solution_implies_a_leq_two (a : ℝ) :
  (∀ x : ℝ, ¬(x ≥ a + 2 ∧ x < 3*a - 2)) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_a_leq_two_l201_20122


namespace NUMINAMATH_CALUDE_retail_price_increase_l201_20112

theorem retail_price_increase (manufacturing_cost : ℝ) (retailer_price : ℝ) (customer_price : ℝ)
  (h1 : customer_price = retailer_price * 1.3)
  (h2 : customer_price = manufacturing_cost * 1.82) :
  (retailer_price - manufacturing_cost) / manufacturing_cost = 0.4 := by
sorry

end NUMINAMATH_CALUDE_retail_price_increase_l201_20112


namespace NUMINAMATH_CALUDE_two_distinct_roots_condition_l201_20170

-- Define the quadratic equation
def quadratic_equation (k x : ℝ) : ℝ := k * x^2 - 2 * x - 3

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  quadratic_equation k x₁ = 0 ∧ 
  quadratic_equation k x₂ = 0

-- Theorem statement
theorem two_distinct_roots_condition (k : ℝ) :
  has_two_distinct_real_roots k ↔ k > -1/3 ∧ k ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_two_distinct_roots_condition_l201_20170


namespace NUMINAMATH_CALUDE_discount_rate_inequality_l201_20173

/-- Represents the maximum discount rate that can be offered while ensuring a profit margin of at least 5% -/
def max_discount_rate (cost_price selling_price min_profit_margin : ℝ) : Prop :=
  ∃ x : ℝ, 
    0 ≤ x ∧ x ≤ 1 ∧
    selling_price * ((1 : ℝ) / 10) * x - cost_price ≥ min_profit_margin * cost_price

theorem discount_rate_inequality 
  (cost_price : ℝ) 
  (selling_price : ℝ) 
  (min_profit_margin : ℝ) 
  (h1 : cost_price = 100)
  (h2 : selling_price = 150)
  (h3 : min_profit_margin = 0.05) :
  max_discount_rate cost_price selling_price min_profit_margin :=
sorry

end NUMINAMATH_CALUDE_discount_rate_inequality_l201_20173


namespace NUMINAMATH_CALUDE_fraction_simplification_l201_20192

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) :
  (2 * x^2 - x + 1) / (x^2 - 1) - x / (x - 1) = (x - 1) / (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l201_20192


namespace NUMINAMATH_CALUDE_percent_of_300_l201_20157

theorem percent_of_300 : (22 : ℝ) / 100 * 300 = 66 := by sorry

end NUMINAMATH_CALUDE_percent_of_300_l201_20157


namespace NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l201_20153

/-- 
For a regular polygon where each exterior angle is 40°, 
the sum of the interior angles is 1260°.
-/
theorem sum_interior_angles_regular_polygon (n : ℕ) : 
  (360 / n = 40) → (n - 2) * 180 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_sum_interior_angles_regular_polygon_l201_20153


namespace NUMINAMATH_CALUDE_eggs_per_basket_l201_20197

theorem eggs_per_basket : ∀ (n : ℕ),
  (30 % n = 0) →  -- Yellow eggs are evenly distributed
  (42 % n = 0) →  -- Blue eggs are evenly distributed
  (n ≥ 4) →       -- At least 4 eggs per basket
  (30 / n ≥ 3) →  -- At least 3 purple baskets
  (42 / n ≥ 3) →  -- At least 3 orange baskets
  n = 6 :=
by
  sorry

#check eggs_per_basket

end NUMINAMATH_CALUDE_eggs_per_basket_l201_20197


namespace NUMINAMATH_CALUDE_eight_students_pairing_l201_20187

theorem eight_students_pairing :
  (Nat.factorial 8) / ((Nat.factorial 4) * (2^4)) = 105 := by
  sorry

end NUMINAMATH_CALUDE_eight_students_pairing_l201_20187


namespace NUMINAMATH_CALUDE_oranges_distribution_l201_20118

theorem oranges_distribution (total : ℕ) (boxes : ℕ) (difference : ℕ) (first_box : ℕ) : 
  total = 120 →
  boxes = 7 →
  difference = 2 →
  (first_box * boxes + (boxes * (boxes - 1) * difference) / 2 = total) →
  first_box = 11 := by
sorry

end NUMINAMATH_CALUDE_oranges_distribution_l201_20118


namespace NUMINAMATH_CALUDE_f_inequality_iff_a_bound_l201_20164

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - a / x - (a + 1) * Real.log x

theorem f_inequality_iff_a_bound (a : ℝ) :
  (∀ x > 0, f a x ≤ x) ↔ a ≥ 1 / (Real.exp 1 - 1) := by sorry

end NUMINAMATH_CALUDE_f_inequality_iff_a_bound_l201_20164


namespace NUMINAMATH_CALUDE_valid_solution_l201_20196

def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem valid_solution :
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {4, 5}
  set_difference A B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_valid_solution_l201_20196


namespace NUMINAMATH_CALUDE_coat_price_l201_20172

theorem coat_price (W : ℝ) (h1 : 2*W - 1.9*W = 4) : 1.9*W = 76 := by
  sorry

end NUMINAMATH_CALUDE_coat_price_l201_20172


namespace NUMINAMATH_CALUDE_remainder_theorem_l201_20150

theorem remainder_theorem : ∃ q : ℕ, 
  2^206 + 206 = q * (2^103 + 2^53 + 1) + 205 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l201_20150


namespace NUMINAMATH_CALUDE_prime_squared_minus_five_not_divisible_by_eight_l201_20104

theorem prime_squared_minus_five_not_divisible_by_eight (p : ℕ) 
  (h_prime : Nat.Prime p) (h_gt_two : p > 2) : ¬ (8 ∣ (p^2 - 5)) := by
  sorry

end NUMINAMATH_CALUDE_prime_squared_minus_five_not_divisible_by_eight_l201_20104


namespace NUMINAMATH_CALUDE_product_difference_squares_l201_20107

theorem product_difference_squares : 101 * 102^2 - 101 * 98^2 = 80800 := by
  sorry

end NUMINAMATH_CALUDE_product_difference_squares_l201_20107


namespace NUMINAMATH_CALUDE_right_triangle_with_constraints_l201_20163

/-- A right-angled triangle with perimeter 5 and shortest altitude 1 has side lengths 5/3, 5/4, and 25/12. -/
theorem right_triangle_with_constraints (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- sides are positive
  a^2 + b^2 = c^2 →  -- right-angled triangle (Pythagorean theorem)
  a + b + c = 5 →  -- perimeter is 5
  min (a*b/c) (min (b*c/a) (c*a/b)) = 1 →  -- shortest altitude is 1
  ((a = 5/3 ∧ b = 5/4 ∧ c = 25/12) ∨ (a = 5/4 ∧ b = 5/3 ∧ c = 25/12)) := by
sorry


end NUMINAMATH_CALUDE_right_triangle_with_constraints_l201_20163


namespace NUMINAMATH_CALUDE_average_reading_time_l201_20152

/-- Given that Emery reads 5 times faster than Serena and takes 20 days to read a book,
    prove that the average number of days for both to read the book is 60 days. -/
theorem average_reading_time (emery_days : ℕ) (emery_speed : ℕ) :
  emery_days = 20 →
  emery_speed = 5 →
  (emery_days + emery_speed * emery_days) / 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_average_reading_time_l201_20152


namespace NUMINAMATH_CALUDE_consecutive_even_odd_squares_divisibility_l201_20180

theorem consecutive_even_odd_squares_divisibility :
  (∀ n : ℕ+, ∃ k : ℕ, (2*n+2)^2 - (2*n)^2 = 4*k) ∧
  (∀ m : ℕ+, ∃ k : ℕ, (2*m+1)^2 - (2*m-1)^2 = 8*k) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_odd_squares_divisibility_l201_20180


namespace NUMINAMATH_CALUDE_game_terminates_l201_20144

/-- Represents the state of the game at each step -/
structure GameState where
  x : ℕ  -- First number on the blackboard
  y : ℕ  -- Second number on the blackboard
  r : ℕ  -- Lower bound of the possible range for the unknown number
  s : ℕ  -- Upper bound of the possible range for the unknown number

/-- The game terminates when the range becomes invalid (r > s) -/
def is_terminal (state : GameState) : Prop :=
  state.r > state.s

/-- The next state of the game after a question is asked -/
def next_state (state : GameState) : GameState :=
  { x := state.x
  , y := state.y
  , r := state.y - state.s
  , s := state.x - state.r }

/-- The main theorem: the game terminates in a finite number of steps -/
theorem game_terminates (a b : ℕ) (h : a > 0 ∧ b > 0) :
  ∃ n : ℕ, is_terminal (n.iterate next_state (GameState.mk (min (a + b) (a + b + 1)) (max (a + b) (a + b + 1)) 0 (a + b))) :=
sorry

end NUMINAMATH_CALUDE_game_terminates_l201_20144


namespace NUMINAMATH_CALUDE_glass_piece_coloring_l201_20134

/-- Represents the count of glass pieces for each color -/
structure GlassPieces where
  red : ℕ
  yellow : ℕ
  blue : ℕ
  sum_is_2005 : red + yellow + blue = 2005

/-- Represents a single operation on the glass pieces -/
inductive Operation
  | RedYellowToBlue
  | RedBlueToYellow
  | YellowBlueToRed

/-- Applies an operation to the glass pieces -/
def apply_operation (gp : GlassPieces) (op : Operation) : GlassPieces :=
  match op with
  | Operation.RedYellowToBlue => 
      { red := gp.red - 1, yellow := gp.yellow - 1, blue := gp.blue + 2, 
        sum_is_2005 := by sorry }
  | Operation.RedBlueToYellow => 
      { red := gp.red - 1, yellow := gp.yellow + 2, blue := gp.blue - 1, 
        sum_is_2005 := by sorry }
  | Operation.YellowBlueToRed => 
      { red := gp.red + 2, yellow := gp.yellow - 1, blue := gp.blue - 1, 
        sum_is_2005 := by sorry }

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to the glass pieces -/
def apply_sequence (gp : GlassPieces) (seq : OperationSequence) : GlassPieces :=
  match seq with
  | [] => gp
  | op :: rest => apply_sequence (apply_operation gp op) rest

/-- Predicate to check if all pieces are the same color -/
def all_same_color (gp : GlassPieces) : Prop :=
  (gp.red = 2005 ∧ gp.yellow = 0 ∧ gp.blue = 0) ∨
  (gp.red = 0 ∧ gp.yellow = 2005 ∧ gp.blue = 0) ∨
  (gp.red = 0 ∧ gp.yellow = 0 ∧ gp.blue = 2005)

theorem glass_piece_coloring
  (gp : GlassPieces) :
  (∃ (seq : OperationSequence), all_same_color (apply_sequence gp seq)) ∧
  (∀ (seq1 seq2 : OperationSequence),
    all_same_color (apply_sequence gp seq1) →
    all_same_color (apply_sequence gp seq2) →
    apply_sequence gp seq1 = apply_sequence gp seq2) := by
  sorry

end NUMINAMATH_CALUDE_glass_piece_coloring_l201_20134


namespace NUMINAMATH_CALUDE_blue_section_damage_probability_l201_20141

/-- The number of trials -/
def n : ℕ := 7

/-- The probability of success in each trial -/
def p : ℚ := 2/7

/-- The number of successes we're interested in -/
def k : ℕ := 7

/-- The probability of exactly k successes in n Bernoulli trials with probability p -/
def bernoulli_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

theorem blue_section_damage_probability :
  bernoulli_probability n k p = 128/823543 := by
  sorry

end NUMINAMATH_CALUDE_blue_section_damage_probability_l201_20141


namespace NUMINAMATH_CALUDE_rachel_apple_picking_l201_20182

theorem rachel_apple_picking (num_trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ) : 
  num_trees = 4 → 
  apples_per_tree = 7 → 
  remaining_apples = 29 → 
  num_trees * apples_per_tree = 28 :=
by sorry

end NUMINAMATH_CALUDE_rachel_apple_picking_l201_20182


namespace NUMINAMATH_CALUDE_tan_alpha_equals_two_tan_pi_fifth_l201_20110

theorem tan_alpha_equals_two_tan_pi_fifth (α : Real) 
  (h : Real.tan α = 2 * Real.tan (π / 5)) : 
  (Real.cos (α - 3 * π / 10)) / (Real.sin (α - π / 5)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_two_tan_pi_fifth_l201_20110


namespace NUMINAMATH_CALUDE_gcd_division_remainder_l201_20138

theorem gcd_division_remainder (a b : ℕ) (h1 : a > b) (h2 : ∃ q r : ℕ, a = b * q + r ∧ 0 < r ∧ r < b) :
  Nat.gcd a b = Nat.gcd b (a % b) :=
by sorry

end NUMINAMATH_CALUDE_gcd_division_remainder_l201_20138


namespace NUMINAMATH_CALUDE_candy_distribution_l201_20158

theorem candy_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  (Nat.choose (n + k - 1) (k - 1)) = 66 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l201_20158


namespace NUMINAMATH_CALUDE_michaels_art_show_earnings_l201_20165

/-- Calculates Michael's earnings from an art show -/
def michaels_earnings (
  extra_large_price : ℝ)
  (large_price : ℝ)
  (medium_price : ℝ)
  (small_price : ℝ)
  (extra_large_sold : ℕ)
  (large_sold : ℕ)
  (medium_sold : ℕ)
  (small_sold : ℕ)
  (large_discount_rate : ℝ)
  (sales_tax_rate : ℝ)
  (material_cost : ℝ)
  (commission_rate : ℝ) : ℝ :=
  let extra_large_revenue := extra_large_price * extra_large_sold
  let large_revenue := large_price * large_sold * (1 - large_discount_rate)
  let medium_revenue := medium_price * medium_sold
  let small_revenue := small_price * small_sold
  let total_revenue := extra_large_revenue + large_revenue + medium_revenue + small_revenue
  let sales_tax := total_revenue * sales_tax_rate
  let total_collected := total_revenue + sales_tax
  let commission := total_revenue * commission_rate
  let total_deductions := material_cost + commission
  total_collected - total_deductions

/-- Theorem stating Michael's earnings from the art show -/
theorem michaels_art_show_earnings :
  michaels_earnings 150 100 80 60 3 5 8 10 0.1 0.05 300 0.1 = 1733 := by
  sorry

end NUMINAMATH_CALUDE_michaels_art_show_earnings_l201_20165


namespace NUMINAMATH_CALUDE_complement_intersection_A_B_l201_20103

def I : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 4, 5, 6}

theorem complement_intersection_A_B :
  (I \ (A ∩ B)) = {1, 2, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_A_B_l201_20103


namespace NUMINAMATH_CALUDE_equation_solution_exists_l201_20133

theorem equation_solution_exists : ∃ c : ℝ, 
  Real.sqrt (4 + Real.sqrt (12 + 6 * c)) + Real.sqrt (6 + Real.sqrt (3 + c)) = 4 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l201_20133


namespace NUMINAMATH_CALUDE_max_value_implies_a_l201_20168

def f (a x : ℝ) : ℝ := a * x^3 + 2 * a * x + 1

theorem max_value_implies_a (a : ℝ) : 
  (∀ x ∈ Set.Icc (-3) 2, f a x ≤ 4) ∧ 
  (∃ x ∈ Set.Icc (-3) 2, f a x = 4) →
  a = 1/4 ∨ a = -1/11 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l201_20168


namespace NUMINAMATH_CALUDE_prob_odd_divisor_21_factorial_l201_20142

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def primeFactorization (n : ℕ) : List (ℕ × ℕ) := sorry

def numDivisors (n : ℕ) : ℕ := sorry

def numOddDivisors (n : ℕ) : ℕ := sorry

theorem prob_odd_divisor_21_factorial :
  let n := factorial 21
  let totalDivisors := numDivisors n
  let oddDivisors := numOddDivisors n
  (oddDivisors : ℚ) / totalDivisors = 1 / 19 := by sorry

end NUMINAMATH_CALUDE_prob_odd_divisor_21_factorial_l201_20142


namespace NUMINAMATH_CALUDE_matching_polygons_l201_20145

def is_matching (n m : ℕ) : Prop :=
  2 * ((n - 2) * 180 / n) = 3 * (360 / m)

theorem matching_polygons :
  ∀ n m : ℕ, n > 2 ∧ m > 2 →
    is_matching n m ↔ ((n = 3 ∧ m = 9) ∨ (n = 4 ∧ m = 6) ∨ (n = 5 ∧ m = 5) ∨ (n = 8 ∧ m = 4)) :=
by sorry

end NUMINAMATH_CALUDE_matching_polygons_l201_20145


namespace NUMINAMATH_CALUDE_expression_value_at_three_l201_20166

theorem expression_value_at_three : 
  let x : ℝ := 3
  (x^8 + 24*x^4 + 144) / (x^4 + 12) = 93 := by
sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l201_20166


namespace NUMINAMATH_CALUDE_arithmetic_sequence_1005th_term_l201_20185

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  p : ℝ
  r : ℝ
  first_term : ℝ := p
  second_term : ℝ := 11
  third_term : ℝ := 4*p - r
  fourth_term : ℝ := 4*p + r

/-- The nth term of the arithmetic sequence -/
def nth_term (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first_term + (n - 1) * (seq.second_term - seq.first_term)

/-- Theorem stating that the 1005th term of the sequence is 6029 -/
theorem arithmetic_sequence_1005th_term (seq : ArithmeticSequence) :
  nth_term seq 1005 = 6029 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_1005th_term_l201_20185


namespace NUMINAMATH_CALUDE_equation_solution_l201_20184

theorem equation_solution :
  let f (x : ℝ) := 1 / (x + 8) + 1 / (x + 5) - 1 / (x + 11) - 1 / (x + 4)
  ∀ x : ℝ, f x = 0 ↔ x = (-3 + Real.sqrt 37) / 2 ∨ x = (-3 - Real.sqrt 37) / 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l201_20184


namespace NUMINAMATH_CALUDE_not_necessary_not_sufficient_condition_l201_20195

theorem not_necessary_not_sufficient_condition (a b : ℝ) : 
  ¬(((a > 0 ∧ b > 0) → (a * b < ((a + b) / 2)^2)) ∧
    ((a * b < ((a + b) / 2)^2) → (a > 0 ∧ b > 0))) := by
  sorry

end NUMINAMATH_CALUDE_not_necessary_not_sufficient_condition_l201_20195


namespace NUMINAMATH_CALUDE_three_lines_determine_plane_l201_20121

-- Define a type for points in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for lines in 3D space
structure Line3D where
  point : Point3D
  direction : Point3D

-- Define a type for planes in 3D space
structure Plane3D where
  point : Point3D
  normal : Point3D

-- Function to check if two lines intersect
def linesIntersect (l1 l2 : Line3D) : Prop := sorry

-- Function to check if three lines intersect at the same point
def threeLinesSameIntersection (l1 l2 l3 : Line3D) : Prop := sorry

-- Function to determine if three lines define a unique plane
def defineUniquePlane (l1 l2 l3 : Line3D) : Prop := sorry

-- Theorem stating that three lines intersecting pairwise but not at the same point determine a unique plane
theorem three_lines_determine_plane (l1 l2 l3 : Line3D) :
  linesIntersect l1 l2 ∧ linesIntersect l2 l3 ∧ linesIntersect l3 l1 ∧
  ¬threeLinesSameIntersection l1 l2 l3 →
  defineUniquePlane l1 l2 l3 := by sorry

end NUMINAMATH_CALUDE_three_lines_determine_plane_l201_20121


namespace NUMINAMATH_CALUDE_sandy_correct_sums_l201_20171

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (marks_per_correct : ℕ) 
  (marks_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 50)
  (h3 : marks_per_correct = 3)
  (h4 : marks_per_incorrect = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums ≤ total_sums ∧
    (marks_per_correct : ℤ) * correct_sums - 
    (marks_per_incorrect : ℤ) * (total_sums - correct_sums) = total_marks ∧
    correct_sums = 22 := by
  sorry

#check sandy_correct_sums

end NUMINAMATH_CALUDE_sandy_correct_sums_l201_20171


namespace NUMINAMATH_CALUDE_double_elimination_64_teams_games_range_l201_20108

/-- Represents a double-elimination tournament --/
structure DoubleEliminationTournament where
  num_teams : ℕ
  no_ties : Bool

/-- The minimum number of games required to determine a champion in a double-elimination tournament --/
def min_games (t : DoubleEliminationTournament) : ℕ := sorry

/-- The maximum number of games required to determine a champion in a double-elimination tournament --/
def max_games (t : DoubleEliminationTournament) : ℕ := sorry

/-- Theorem stating the range of games required for a 64-team double-elimination tournament --/
theorem double_elimination_64_teams_games_range (t : DoubleEliminationTournament) 
  (h1 : t.num_teams = 64) (h2 : t.no_ties = true) : 
  min_games t = 96 ∧ max_games t = 97 := by sorry

end NUMINAMATH_CALUDE_double_elimination_64_teams_games_range_l201_20108
