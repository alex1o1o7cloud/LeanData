import Mathlib

namespace NUMINAMATH_CALUDE_ab_range_l1647_164712

-- Define the function f
def f (x : ℝ) : ℝ := |2 - x^2|

-- State the theorem
theorem ab_range (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) :
  0 < a * b ∧ a * b < 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_range_l1647_164712


namespace NUMINAMATH_CALUDE_number_equality_l1647_164769

theorem number_equality (x : ℚ) : 
  (35 / 100 : ℚ) * x = (20 / 100 : ℚ) * 50 → x = 200 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l1647_164769


namespace NUMINAMATH_CALUDE_existence_of_special_sequences_l1647_164768

theorem existence_of_special_sequences : ∃ (a b : ℕ → ℕ), 
  (∀ n, a n < a (n + 1)) ∧ 
  (∀ n, b n < b (n + 1)) ∧ 
  (a 1 = 25) ∧ 
  (b 1 = 57) ∧ 
  (∀ n, (b n)^2 + 1 ≡ 0 [MOD (a n) * ((a n) + 1)]) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_sequences_l1647_164768


namespace NUMINAMATH_CALUDE_apples_added_to_pile_l1647_164781

/-- Given an initial pile of apples and a final pile of apples,
    calculate the number of apples added. -/
def applesAdded (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that 5 apples were added to the pile -/
theorem apples_added_to_pile :
  let initial := 8
  let final := 13
  applesAdded initial final = 5 := by sorry

end NUMINAMATH_CALUDE_apples_added_to_pile_l1647_164781


namespace NUMINAMATH_CALUDE_library_books_loaned_l1647_164778

theorem library_books_loaned (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 70 / 100)
  (h3 : final_books = 57) :
  ∃ (loaned_books : ℕ), loaned_books = 60 ∧ 
    initial_books - (↑loaned_books * (1 - return_rate)).floor = final_books :=
by sorry

end NUMINAMATH_CALUDE_library_books_loaned_l1647_164778


namespace NUMINAMATH_CALUDE_pentagon_stack_exists_no_pentagon_stack_for_one_no_pentagon_stack_for_three_l1647_164740

/-- A regular pentagon with numbers from 1 to 5 at its vertices -/
def Pentagon : Type := Fin 5 → Fin 5

/-- A stack of pentagons -/
def PentagonStack : Type := List Pentagon

/-- The sum of numbers at a vertex in a stack of pentagons -/
def vertexSum (stack : PentagonStack) (vertex : Fin 5) : ℕ :=
  (stack.map (λ p => p vertex)).sum

/-- A predicate that checks if all vertex sums in a stack are equal -/
def allVertexSumsEqual (stack : PentagonStack) : Prop :=
  ∀ v1 v2 : Fin 5, vertexSum stack v1 = vertexSum stack v2

/-- Main theorem: For any natural number n ≠ 1 and n ≠ 3, there exists a valid pentagon stack of size n -/
theorem pentagon_stack_exists (n : ℕ) (h1 : n ≠ 1) (h3 : n ≠ 3) :
  ∃ (stack : PentagonStack), stack.length = n ∧ allVertexSumsEqual stack :=
sorry

/-- No valid pentagon stack exists for n = 1 -/
theorem no_pentagon_stack_for_one :
  ¬∃ (stack : PentagonStack), stack.length = 1 ∧ allVertexSumsEqual stack :=
sorry

/-- No valid pentagon stack exists for n = 3 -/
theorem no_pentagon_stack_for_three :
  ¬∃ (stack : PentagonStack), stack.length = 3 ∧ allVertexSumsEqual stack :=
sorry

end NUMINAMATH_CALUDE_pentagon_stack_exists_no_pentagon_stack_for_one_no_pentagon_stack_for_three_l1647_164740


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1647_164717

theorem greatest_three_digit_multiple_of_17 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n → n ≤ 986 ∧ 17 ∣ 986 :=
by sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1647_164717


namespace NUMINAMATH_CALUDE_special_function_at_three_l1647_164709

/-- A function satisfying f(2x + 1) = 2f(x) + 1 for all real x, and f(0) = 2 -/
def special_function (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (2 * x + 1) = 2 * f x + 1) ∧ f 0 = 2

/-- The value of f(3) for a special function f -/
theorem special_function_at_three (f : ℝ → ℝ) (h : special_function f) : f 3 = 11 := by
  sorry

end NUMINAMATH_CALUDE_special_function_at_three_l1647_164709


namespace NUMINAMATH_CALUDE_baseball_card_value_decrease_l1647_164774

theorem baseball_card_value_decrease : 
  let initial_value : ℝ := 100
  let year1_decrease : ℝ := 0.60
  let year2_decrease : ℝ := 0.30
  let year3_decrease : ℝ := 0.20
  let year4_decrease : ℝ := 0.10
  
  let value_after_year1 : ℝ := initial_value * (1 - year1_decrease)
  let value_after_year2 : ℝ := value_after_year1 * (1 - year2_decrease)
  let value_after_year3 : ℝ := value_after_year2 * (1 - year3_decrease)
  let value_after_year4 : ℝ := value_after_year3 * (1 - year4_decrease)
  
  let total_decrease : ℝ := (initial_value - value_after_year4) / initial_value * 100

  total_decrease = 79.84 := by sorry

end NUMINAMATH_CALUDE_baseball_card_value_decrease_l1647_164774


namespace NUMINAMATH_CALUDE_solve_quadratic_equation_l1647_164790

theorem solve_quadratic_equation (x : ℝ) (h1 : 3 * x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_quadratic_equation_l1647_164790


namespace NUMINAMATH_CALUDE_gcd_special_numbers_l1647_164701

theorem gcd_special_numbers : Nat.gcd 333333333 555555555 = 111111111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_special_numbers_l1647_164701


namespace NUMINAMATH_CALUDE_multiple_properties_l1647_164763

/-- Given that c is a multiple of 4 and d is a multiple of 8, prove the following statements -/
theorem multiple_properties (c d : ℤ) 
  (hc : ∃ k : ℤ, c = 4 * k) 
  (hd : ∃ m : ℤ, d = 8 * m) : 
  (∃ n : ℤ, d = 4 * n) ∧ 
  (∃ p : ℤ, c - d = 4 * p) ∧ 
  (∃ q : ℤ, c - d = 2 * q) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l1647_164763


namespace NUMINAMATH_CALUDE_harmonic_sum_divisibility_l1647_164756

theorem harmonic_sum_divisibility (p : ℕ) (m n : ℕ) (h_prime : Nat.Prime p) (h_p_gt_2 : p > 2) 
  (h_sum : (m : ℚ) / n = (Finset.range (p - 1)).sum (λ i => 1 / (i + 1 : ℚ))) :
  p ∣ m := by
  sorry

end NUMINAMATH_CALUDE_harmonic_sum_divisibility_l1647_164756


namespace NUMINAMATH_CALUDE_probability_all_visible_faces_same_color_l1647_164739

/-- Represents the three possible colors for painting the cube faces -/
inductive Color
| Red
| Blue
| Green

/-- A cube with 6 faces, each painted with a color -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- The probability of a specific color arrangement on the cube -/
def colorArrangementProbability : ℚ := (1 : ℚ) / 729

/-- Predicate to check if a cube can be placed with all visible vertical faces the same color -/
def hasAllVisibleFacesSameColor (c : Cube) : Prop := sorry

/-- The number of color arrangements where all visible vertical faces can be the same color -/
def numValidArrangements : ℕ := 57

/-- Theorem stating the probability of a cube having all visible vertical faces the same color -/
theorem probability_all_visible_faces_same_color :
  (numValidArrangements : ℚ) * colorArrangementProbability = 57 / 729 := by sorry

end NUMINAMATH_CALUDE_probability_all_visible_faces_same_color_l1647_164739


namespace NUMINAMATH_CALUDE_absolute_value_comparison_l1647_164765

theorem absolute_value_comparison (m n : ℝ) : m < n → n < 0 → abs m > abs n := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_comparison_l1647_164765


namespace NUMINAMATH_CALUDE_fifteen_factorial_largest_square_exponent_sum_l1647_164796

def largest_perfect_square_exponent_sum (n : ℕ) : ℕ :=
  let prime_factors := Nat.factors n
  let max_square_exponents := prime_factors.map (fun p => (Nat.factorization n p) / 2)
  max_square_exponents.sum

theorem fifteen_factorial_largest_square_exponent_sum :
  largest_perfect_square_exponent_sum (Nat.factorial 15) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_factorial_largest_square_exponent_sum_l1647_164796


namespace NUMINAMATH_CALUDE_sum_of_squared_even_differences_l1647_164764

theorem sum_of_squared_even_differences : 
  (20^2 - 18^2) + (16^2 - 14^2) + (12^2 - 10^2) + (8^2 - 6^2) + (4^2 - 2^2) = 200 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_even_differences_l1647_164764


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1647_164729

/-- A geometric sequence with a_4 = 4 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) ∧ a 4 = 4

theorem geometric_sequence_product (a : ℕ → ℝ) (h : geometric_sequence a) :
  a 2 * a 6 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1647_164729


namespace NUMINAMATH_CALUDE_characterization_of_m_l1647_164706

-- Define a good integer
def is_good (n : ℤ) : Prop :=
  ¬ ∃ k : ℤ, n.natAbs = k^2

-- Define the property for m
def has_property (m : ℤ) : Prop :=
  ∀ N : ℕ, ∃ a b c : ℤ,
    (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
    (is_good a ∧ is_good b ∧ is_good c) ∧
    (a + b + c = m) ∧
    (∃ k : ℤ, a * b * c = (2*k + 1)^2) ∧
    (N < a.natAbs ∧ N < b.natAbs ∧ N < c.natAbs)

-- The main theorem
theorem characterization_of_m (m : ℤ) :
  has_property m ↔ m % 4 = 3 :=
sorry

end NUMINAMATH_CALUDE_characterization_of_m_l1647_164706


namespace NUMINAMATH_CALUDE_imaginary_unit_powers_l1647_164748

theorem imaginary_unit_powers (i : ℂ) : i^2 = -1 → i^50 + i^105 = -1 + i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_powers_l1647_164748


namespace NUMINAMATH_CALUDE_stuffed_animals_theorem_l1647_164744

/-- Represents the number of stuffed animals for each girl -/
structure StuffedAnimals where
  mckenna : ℕ
  kenley : ℕ
  tenly : ℕ

/-- Calculates the total number of stuffed animals -/
def total (sa : StuffedAnimals) : ℕ :=
  sa.mckenna + sa.kenley + sa.tenly

/-- Calculates the average number of stuffed animals per girl -/
def average (sa : StuffedAnimals) : ℚ :=
  (total sa : ℚ) / 3

/-- Calculates the percentage of total stuffed animals McKenna has -/
def mckennaPercentage (sa : StuffedAnimals) : ℚ :=
  (sa.mckenna : ℚ) / (total sa : ℚ) * 100

theorem stuffed_animals_theorem (sa : StuffedAnimals) 
  (h1 : sa.mckenna = 34)
  (h2 : sa.kenley = 2 * sa.mckenna)
  (h3 : sa.tenly = sa.kenley + 5) :
  total sa = 175 ∧ 
  58.32 < average sa ∧ average sa < 58.34 ∧
  19.42 < mckennaPercentage sa ∧ mckennaPercentage sa < 19.44 := by
  sorry

#eval total { mckenna := 34, kenley := 68, tenly := 73 }
#eval average { mckenna := 34, kenley := 68, tenly := 73 }
#eval mckennaPercentage { mckenna := 34, kenley := 68, tenly := 73 }

end NUMINAMATH_CALUDE_stuffed_animals_theorem_l1647_164744


namespace NUMINAMATH_CALUDE_oil_usage_l1647_164761

theorem oil_usage (rons_oil sara_usage : ℚ) 
  (h1 : rons_oil = 3/8)
  (h2 : sara_usage = 5/6 * rons_oil) : 
  sara_usage = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_oil_usage_l1647_164761


namespace NUMINAMATH_CALUDE_city_population_problem_l1647_164750

theorem city_population_problem (p : ℝ) : 
  (0.85 * (p + 1500) = p + 50) → p = 1500 := by
  sorry

end NUMINAMATH_CALUDE_city_population_problem_l1647_164750


namespace NUMINAMATH_CALUDE_sequence_growth_l1647_164741

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ i : ℕ, i ≥ 1 → Nat.gcd (a i) (a (i + 1)) > a (i - 1)

theorem sequence_growth (a : ℕ → ℕ) (h : sequence_property a) :
  ∀ n : ℕ, a n ≥ 2^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_growth_l1647_164741


namespace NUMINAMATH_CALUDE_descent_time_is_two_hours_l1647_164784

/-- Proves that the time taken to descend a hill is 2 hours, given specific conditions. -/
theorem descent_time_is_two_hours 
  (time_to_top : ℝ) 
  (avg_speed_total : ℝ) 
  (avg_speed_up : ℝ) 
  (time_to_top_is_four : time_to_top = 4)
  (avg_speed_total_is_three : avg_speed_total = 3)
  (avg_speed_up_is_two_point_two_five : avg_speed_up = 2.25) :
  let distance_to_top : ℝ := avg_speed_up * time_to_top
  let total_distance : ℝ := 2 * distance_to_top
  let total_time : ℝ := total_distance / avg_speed_total
  time_to_top - (total_time - time_to_top) = 2 := by
  sorry

#check descent_time_is_two_hours

end NUMINAMATH_CALUDE_descent_time_is_two_hours_l1647_164784


namespace NUMINAMATH_CALUDE_bah_equivalent_to_yahs_l1647_164762

/-- Conversion rate between bahs and rahs -/
def bah_to_rah : ℚ := 18 / 10

/-- Conversion rate between rahs and yahs -/
def rah_to_yah : ℚ := 10 / 6

/-- The number of yahs to convert -/
def yahs_to_convert : ℕ := 1500

theorem bah_equivalent_to_yahs : 
  ∃ (n : ℕ), n * bah_to_rah * rah_to_yah = yahs_to_convert ∧ n = 500 := by
  sorry

end NUMINAMATH_CALUDE_bah_equivalent_to_yahs_l1647_164762


namespace NUMINAMATH_CALUDE_probability_to_reach_target_is_correct_l1647_164771

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a possible move direction -/
inductive Direction
  | Left
  | Right
  | Up
  | Down

/-- The probability of each direction -/
def directionProbability : ℚ := 1 / 4

/-- The number of steps allowed -/
def numberOfSteps : ℕ := 6

/-- The starting point -/
def startPoint : Point := ⟨0, 0⟩

/-- The target point -/
def targetPoint : Point := ⟨3, 1⟩

/-- Function to calculate the probability of reaching the target point -/
def probabilityToReachTarget (start : Point) (target : Point) (steps : ℕ) : ℚ :=
  sorry

theorem probability_to_reach_target_is_correct :
  probabilityToReachTarget startPoint targetPoint numberOfSteps = 45 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_probability_to_reach_target_is_correct_l1647_164771


namespace NUMINAMATH_CALUDE_greatest_common_divisor_of_98_and_n_l1647_164751

theorem greatest_common_divisor_of_98_and_n (n : ℕ) : 
  (∃ (d1 d2 d3 : ℕ), d1 < d2 ∧ d2 < d3 ∧ 
   {d : ℕ | d ∣ 98 ∧ d ∣ n} = {d1, d2, d3}) → 
  Nat.gcd 98 n = 49 := by
sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_of_98_and_n_l1647_164751


namespace NUMINAMATH_CALUDE_probability_of_one_in_pascal_triangle_l1647_164795

def pascalTriangleElements (n : ℕ) : ℕ := n * (n + 1) / 2

def onesInPascalTriangle (n : ℕ) : ℕ := 1 + 2 * (n - 1)

theorem probability_of_one_in_pascal_triangle : 
  (onesInPascalTriangle 20 : ℚ) / (pascalTriangleElements 20 : ℚ) = 39 / 210 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_one_in_pascal_triangle_l1647_164795


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l1647_164793

/-- Parabola type representing y = ax^2 + bx --/
structure Parabola where
  a : ℝ
  b : ℝ
  a_nonzero : a ≠ 0

/-- Point type representing (x, y) coordinates --/
structure Point where
  x : ℝ
  y : ℝ

theorem parabola_point_relationship (p : Parabola) (m n t : ℝ) :
  3 * p.a + p.b > 0 →
  p.a + p.b < 0 →
  Point.mk (-3) m ∈ {pt : Point | pt.y = p.a * pt.x^2 + p.b * pt.x} →
  Point.mk 2 n ∈ {pt : Point | pt.y = p.a * pt.x^2 + p.b * pt.x} →
  Point.mk 4 t ∈ {pt : Point | pt.y = p.a * pt.x^2 + p.b * pt.x} →
  n < t ∧ t < m := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_relationship_l1647_164793


namespace NUMINAMATH_CALUDE_fenced_area_calculation_l1647_164775

theorem fenced_area_calculation (yard_length yard_width cutout_side : ℕ) 
  (h1 : yard_length = 20)
  (h2 : yard_width = 18)
  (h3 : cutout_side = 4) :
  yard_length * yard_width - cutout_side * cutout_side = 344 := by
  sorry

end NUMINAMATH_CALUDE_fenced_area_calculation_l1647_164775


namespace NUMINAMATH_CALUDE_sum_of_powers_implies_sum_power_l1647_164730

theorem sum_of_powers_implies_sum_power (a b : ℝ) :
  a^2009 + b^2009 = 0 → (a + b)^2009 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_powers_implies_sum_power_l1647_164730


namespace NUMINAMATH_CALUDE_charlie_has_largest_answer_l1647_164792

def alice_calc (start : ℕ) : ℕ := ((start - 3) * 3) + 5

def bob_calc (start : ℕ) : ℕ := ((start * 3) - 3) + 5

def charlie_calc (start : ℕ) : ℕ := ((start - 3) + 5) * 3

theorem charlie_has_largest_answer (start : ℕ) (h : start = 15) :
  charlie_calc start > alice_calc start ∧ charlie_calc start > bob_calc start := by
  sorry

end NUMINAMATH_CALUDE_charlie_has_largest_answer_l1647_164792


namespace NUMINAMATH_CALUDE_base4_arithmetic_theorem_l1647_164718

/-- Converts a number from base 4 to base 10 --/
def base4To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 4 --/
def base10To4 (n : ℕ) : ℕ := sorry

/-- Performs arithmetic operations in base 4 --/
def base4Arithmetic (a b c d : ℕ) : ℕ :=
  let a10 := base4To10 a
  let b10 := base4To10 b
  let c10 := base4To10 c
  let d10 := base4To10 d
  let result := a10 * b10 / c10 * d10
  base10To4 result

theorem base4_arithmetic_theorem :
  base4Arithmetic 231 21 3 2 = 10232 := by sorry

end NUMINAMATH_CALUDE_base4_arithmetic_theorem_l1647_164718


namespace NUMINAMATH_CALUDE_function_through_point_l1647_164727

/-- Given a function f(x) = ax^3 - 2x that passes through the point (-1, 4), prove that a = -2 -/
theorem function_through_point (a : ℝ) : (fun x : ℝ ↦ a * x^3 - 2*x) (-1) = 4 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_function_through_point_l1647_164727


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l1647_164731

open Real

theorem range_of_a_for_inequality (a : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ a * x₀ - log x₀ < 0) → a < 1 / exp 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l1647_164731


namespace NUMINAMATH_CALUDE_cubic_sum_over_product_l1647_164711

theorem cubic_sum_over_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 18)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2*x*y*z) :
  (x^3 + y^3 + z^3) / (x*y*z) = 21 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_over_product_l1647_164711


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l1647_164759

/-- The perpendicular bisector of a line segment passing through two points -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) (b : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = b

theorem perpendicular_bisector_value : 
  perpendicular_bisector 2 4 6 8 10 := by
  sorry

#check perpendicular_bisector_value

end NUMINAMATH_CALUDE_perpendicular_bisector_value_l1647_164759


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l1647_164785

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l1647_164785


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1647_164742

theorem sum_of_solutions_quadratic (x : ℝ) : 
  let a : ℝ := -32
  let b : ℝ := 84
  let c : ℝ := 135
  let eq := a * x^2 + b * x + c = 0
  let sum_of_roots := -b / a
  sum_of_roots = 21 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l1647_164742


namespace NUMINAMATH_CALUDE_triangle_side_sum_l1647_164722

theorem triangle_side_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 60) (h3 : b = 30) (h4 : c = 90) (side_c : ℝ) (h5 : side_c = 8) : 
  ∃ (side_a side_b : ℝ), 
    abs ((side_a + side_b) - 18.9) < 0.05 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l1647_164722


namespace NUMINAMATH_CALUDE_percentage_relation_l1647_164773

theorem percentage_relation (j k l m x : ℝ) 
  (h1 : 1.25 * j = 0.25 * k)
  (h2 : 1.5 * k = x / 100 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l1647_164773


namespace NUMINAMATH_CALUDE_simplified_expression_terms_l1647_164749

/-- The number of terms in the simplified expression of (x+y+z)^2006 + (x-y-z)^2006 -/
def num_terms : ℕ := 1008016

/-- The exponent used in the expression -/
def exponent : ℕ := 2006

theorem simplified_expression_terms :
  num_terms = (exponent / 2 + 1)^2 :=
sorry

end NUMINAMATH_CALUDE_simplified_expression_terms_l1647_164749


namespace NUMINAMATH_CALUDE_negative_five_is_square_root_of_twenty_five_l1647_164735

theorem negative_five_is_square_root_of_twenty_five : ∃ x : ℝ, x^2 = 25 ∧ x = -5 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_is_square_root_of_twenty_five_l1647_164735


namespace NUMINAMATH_CALUDE_college_students_fraction_l1647_164738

theorem college_students_fraction (total : ℕ) (h_total : total > 0) :
  let third_year := (total : ℚ) / 2
  let not_second_year := (total : ℚ) * 7 / 10
  let second_year := total - not_second_year
  let not_third_year := total - third_year
  second_year / not_third_year = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_college_students_fraction_l1647_164738


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1647_164719

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → a^2 > 2*a) ∧ 
  (∃ a, a^2 > 2*a ∧ ¬(a > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1647_164719


namespace NUMINAMATH_CALUDE_pie_division_l1647_164723

theorem pie_division (total_pie : ℚ) (people : ℕ) :
  total_pie = 5 / 8 ∧ people = 4 →
  total_pie / people = 5 / 32 := by
sorry

end NUMINAMATH_CALUDE_pie_division_l1647_164723


namespace NUMINAMATH_CALUDE_parabola_c_value_l1647_164746

/-- Represents a parabola with equation x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  p.x_coord 3 = 4 →  -- vertex at (4, 3)
  p.x_coord 5 = 2 →  -- passes through (2, 5)
  p.c = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1647_164746


namespace NUMINAMATH_CALUDE_range_of_g_l1647_164745

theorem range_of_g (x : ℝ) : 3/4 ≤ Real.sin x ^ 2 + Real.cos x ^ 4 ∧ Real.sin x ^ 2 + Real.cos x ^ 4 ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_g_l1647_164745


namespace NUMINAMATH_CALUDE_birdhouse_planks_l1647_164702

/-- The number of planks required to build one birdhouse -/
def planks_per_birdhouse : ℕ := sorry

/-- The number of nails required to build one birdhouse -/
def nails_per_birdhouse : ℕ := 20

/-- The cost of one nail in cents -/
def nail_cost : ℕ := 5

/-- The cost of one plank in cents -/
def plank_cost : ℕ := 300

/-- The total cost to build 4 birdhouses in cents -/
def total_cost_4_birdhouses : ℕ := 8800

theorem birdhouse_planks :
  planks_per_birdhouse = 7 ∧
  4 * (nails_per_birdhouse * nail_cost + planks_per_birdhouse * plank_cost) = total_cost_4_birdhouses :=
sorry

end NUMINAMATH_CALUDE_birdhouse_planks_l1647_164702


namespace NUMINAMATH_CALUDE_product_equals_three_eighths_l1647_164700

-- Define the fractions and mixed number
def a : ℚ := 1/2
def b : ℚ := 2/3
def c : ℚ := 3/4
def d : ℚ := 3/2  -- 1.5 as a fraction

-- State the theorem
theorem product_equals_three_eighths :
  a * b * c * d = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_three_eighths_l1647_164700


namespace NUMINAMATH_CALUDE_largest_sum_is_three_fourths_l1647_164721

theorem largest_sum_is_three_fourths : 
  let sums : List ℚ := [1/4 + 1/9, 1/4 + 1/10, 1/4 + 1/2, 1/4 + 1/12, 1/4 + 1/11]
  (∀ s ∈ sums, s ≤ 3/4) ∧ (3/4 ∈ sums) := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_is_three_fourths_l1647_164721


namespace NUMINAMATH_CALUDE_triangle_x_coordinate_sum_l1647_164799

/-- Given two triangles ABC and ADF with specific areas and coordinates,
    prove that the sum of all possible x-coordinates of A is -635.6 -/
theorem triangle_x_coordinate_sum :
  let triangle_ABC_area : ℝ := 2010
  let triangle_ADF_area : ℝ := 8020
  let B : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (226, 0)
  let D : ℝ × ℝ := (680, 380)
  let F : ℝ × ℝ := (700, 400)
  ∃ (x₁ x₂ : ℝ), 
    (∃ (y₁ : ℝ), triangle_ABC_area = (1/2) * 226 * |y₁|) ∧
    (∃ (y₂ : ℝ), triangle_ADF_area = (1/2) * 20 * |x₁ - y₂ + 300| / Real.sqrt 2) ∧
    (∃ (y₃ : ℝ), triangle_ADF_area = (1/2) * 20 * |x₂ - y₃ + 300| / Real.sqrt 2) ∧
    x₁ + x₂ = -635.6 := by
  sorry

#check triangle_x_coordinate_sum

end NUMINAMATH_CALUDE_triangle_x_coordinate_sum_l1647_164799


namespace NUMINAMATH_CALUDE_sum_of_series_equals_half_l1647_164770

theorem sum_of_series_equals_half :
  let series := λ k : ℕ => (3 ^ (2 ^ k)) / ((9 ^ (2 ^ k)) - 1)
  ∑' k, series k = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_half_l1647_164770


namespace NUMINAMATH_CALUDE_pants_price_proof_l1647_164776

-- Define the total cost
def total_cost : ℝ := 70.93

-- Define the price difference between belt and pants
def price_difference : ℝ := 2.93

-- Define the price of pants
def price_of_pants : ℝ := 34.00

-- Theorem statement
theorem pants_price_proof :
  ∃ (belt_price : ℝ),
    price_of_pants + belt_price = total_cost ∧
    price_of_pants = belt_price - price_difference :=
by sorry

end NUMINAMATH_CALUDE_pants_price_proof_l1647_164776


namespace NUMINAMATH_CALUDE_product_equals_specific_number_l1647_164786

theorem product_equals_specific_number (A B : ℕ) :
  990 * 991 * 992 * 993 = 966428000000 + A * 10000000 + 910000 + B * 100 + 40 →
  A * 10 + B = 50 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_specific_number_l1647_164786


namespace NUMINAMATH_CALUDE_product_remainder_l1647_164798

theorem product_remainder (a b c : ℕ) (ha : a = 2457) (hb : b = 7623) (hc : c = 91309) : 
  (a * b * c) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1647_164798


namespace NUMINAMATH_CALUDE_total_cost_of_sarees_l1647_164733

/-- Calculates the final price of a saree after applying discounts -/
def apply_discounts (price : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl (fun p d => p * (1 - d)) price

/-- Converts a price from one currency to INR -/
def convert_to_inr (price : ℝ) (rate : ℝ) : ℝ :=
  price * rate

/-- Applies sales tax to a price -/
def apply_sales_tax (price : ℝ) (tax_rate : ℝ) : ℝ :=
  price * (1 + tax_rate)

/-- Theorem: The total cost of purchasing three sarees is 39421.08 INR -/
theorem total_cost_of_sarees : 
  let saree1_price : ℝ := 200
  let saree1_discounts : List ℝ := [0.20, 0.15, 0.05]
  let saree1_rate : ℝ := 75

  let saree2_price : ℝ := 150
  let saree2_discounts : List ℝ := [0.10, 0.07]
  let saree2_rate : ℝ := 100

  let saree3_price : ℝ := 180
  let saree3_discounts : List ℝ := [0.12]
  let saree3_rate : ℝ := 90

  let sales_tax : ℝ := 0.08

  let saree1_final := apply_sales_tax (convert_to_inr (apply_discounts saree1_price saree1_discounts) saree1_rate) sales_tax
  let saree2_final := apply_sales_tax (convert_to_inr (apply_discounts saree2_price saree2_discounts) saree2_rate) sales_tax
  let saree3_final := apply_sales_tax (convert_to_inr (apply_discounts saree3_price saree3_discounts) saree3_rate) sales_tax

  saree1_final + saree2_final + saree3_final = 39421.08 :=
by sorry


end NUMINAMATH_CALUDE_total_cost_of_sarees_l1647_164733


namespace NUMINAMATH_CALUDE_middle_number_proof_l1647_164743

theorem middle_number_proof (a b c : ℕ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_order : a < b ∧ b < c)
  (h_sum_ab : a + b = 18)
  (h_sum_ac : a + c = 22)
  (h_sum_bc : b + c = 26)
  (h_diff : c - a = 10) : 
  b = 11 := by sorry

end NUMINAMATH_CALUDE_middle_number_proof_l1647_164743


namespace NUMINAMATH_CALUDE_units_digit_of_special_two_digit_number_l1647_164725

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def units_digit (n : ℕ) : ℕ := n % 10

def digit_product (n : ℕ) : ℕ := tens_digit n * units_digit n

def digit_sum (n : ℕ) : ℕ := tens_digit n + units_digit n

theorem units_digit_of_special_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ N = digit_product N * digit_sum N ∧ units_digit N = 8 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_special_two_digit_number_l1647_164725


namespace NUMINAMATH_CALUDE_area_of_triangle_hyperbola_triangle_area_l1647_164732

/-- A hyperbola with center at the origin, foci on the x-axis, and eccentricity √2 -/
structure Hyperbola where
  eccentricity : ℝ
  passes_through : ℝ × ℝ
  eccentricity_eq : eccentricity = Real.sqrt 2
  point_on_hyperbola : passes_through = (4, Real.sqrt 10)

/-- A point M on the hyperbola where MF₁ ⟂ MF₂ -/
structure PointM (h : Hyperbola) where
  point : ℝ × ℝ
  on_hyperbola : point ∈ {p : ℝ × ℝ | p.1^2 - p.2^2 = 6}
  perpendicular : ∃ (f₁ f₂ : ℝ × ℝ), f₁.2 = 0 ∧ f₂.2 = 0 ∧ 
    (point.1 - f₁.1) * (point.1 - f₂.1) + point.2 * point.2 = 0

/-- The theorem stating that the area of triangle F₁MF₂ is 6 -/
theorem area_of_triangle (h : Hyperbola) (m : PointM h) : ℝ :=
  6

/-- The main theorem to be proved -/
theorem hyperbola_triangle_area (h : Hyperbola) (m : PointM h) :
  area_of_triangle h m = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_hyperbola_triangle_area_l1647_164732


namespace NUMINAMATH_CALUDE_proportion_with_added_number_l1647_164704

theorem proportion_with_added_number : 
  ∃ (x : ℚ), (1 : ℚ) / 3 = 4 / x ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_proportion_with_added_number_l1647_164704


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_l1647_164705

theorem quadratic_roots_imply_m (m : ℚ) : 
  (∃ x : ℂ, 9 * x^2 + 5 * x + m = 0 ∧ 
   (x = (-5 + Complex.I * Real.sqrt 391) / 18 ∨ 
    x = (-5 - Complex.I * Real.sqrt 391) / 18)) → 
  m = 104 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_l1647_164705


namespace NUMINAMATH_CALUDE_b_33_mod_35_l1647_164787

-- Definition of b_n
def b (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem b_33_mod_35 : b 33 % 35 = 21 := by sorry

end NUMINAMATH_CALUDE_b_33_mod_35_l1647_164787


namespace NUMINAMATH_CALUDE_august_math_problems_l1647_164783

def problem (x y z : ℝ) : Prop :=
  let first_answer := x
  let second_answer := 2 * x - y
  let third_answer := 3 * x - z
  let fourth_answer := (x + (2 * x - y) + (3 * x - z)) / 3
  x = 600 ∧
  y > 0 ∧
  z = (x + (2 * x - y)) - 400 ∧
  first_answer + second_answer + third_answer + fourth_answer = 2933.33

theorem august_math_problems :
  ∃ (y z : ℝ), problem 600 y z :=
sorry

end NUMINAMATH_CALUDE_august_math_problems_l1647_164783


namespace NUMINAMATH_CALUDE_power_division_23_l1647_164747

theorem power_division_23 : (23 ^ 11 : ℕ) / (23 ^ 5) = 148035889 := by sorry

end NUMINAMATH_CALUDE_power_division_23_l1647_164747


namespace NUMINAMATH_CALUDE_sum_ages_after_ten_years_l1647_164788

/-- Given Ann's age and Tom's age relative to Ann's, calculate the sum of their ages after a certain number of years. -/
def sum_ages_after_years (ann_age : ℕ) (tom_age_multiplier : ℕ) (years_later : ℕ) : ℕ :=
  (ann_age + years_later) + (ann_age * tom_age_multiplier + years_later)

/-- Theorem stating that given Ann's age is 6 and Tom's age is twice Ann's, the sum of their ages 10 years later is 38. -/
theorem sum_ages_after_ten_years :
  sum_ages_after_years 6 2 10 = 38 := by
  sorry

end NUMINAMATH_CALUDE_sum_ages_after_ten_years_l1647_164788


namespace NUMINAMATH_CALUDE_city_fuel_efficiency_l1647_164720

/-- Represents the fuel efficiency of a car -/
structure FuelEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank_size : ℝ -- Size of the fuel tank in gallons

/-- The conditions of the problem -/
def problem_conditions (fe : FuelEfficiency) : Prop :=
  fe.highway * fe.tank_size = 420 ∧
  fe.city * fe.tank_size = 336 ∧
  fe.city = fe.highway - 6

/-- The theorem to be proved -/
theorem city_fuel_efficiency 
  (fe : FuelEfficiency) 
  (h : problem_conditions fe) : 
  fe.city = 24 := by
  sorry

end NUMINAMATH_CALUDE_city_fuel_efficiency_l1647_164720


namespace NUMINAMATH_CALUDE_completing_square_result_l1647_164757

theorem completing_square_result (x : ℝ) : 
  x^2 - 4*x - 1 = 0 ↔ (x - 2)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_result_l1647_164757


namespace NUMINAMATH_CALUDE_function_and_inequality_problem_l1647_164779

/-- Given a function f(x) = b * a^x with the specified properties, 
    prove that f(x) = 3 * 2^x and find the maximum value of m. -/
theorem function_and_inequality_problem 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = b * a^x)
  (h2 : a > 0)
  (h3 : a ≠ 1)
  (h4 : f 1 = 6)
  (h5 : f 3 = 24) :
  (∀ x, f x = 3 * 2^x) ∧ 
  (∀ m, (∀ x ≤ 1, (1/a)^x + (1/b)^x - m ≥ 0) ↔ m ≤ 5/6) :=
by sorry

end NUMINAMATH_CALUDE_function_and_inequality_problem_l1647_164779


namespace NUMINAMATH_CALUDE_judy_hits_percentage_l1647_164715

theorem judy_hits_percentage (total_hits : ℕ) (home_runs : ℕ) (triples : ℕ) (doubles : ℕ)
  (h_total : total_hits = 35)
  (h_home : home_runs = 1)
  (h_triple : triples = 1)
  (h_double : doubles = 5) :
  (total_hits - (home_runs + triples + doubles)) / total_hits = 4/5 := by
sorry

end NUMINAMATH_CALUDE_judy_hits_percentage_l1647_164715


namespace NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l1647_164755

theorem x_range_for_quadratic_inequality :
  ∀ x : ℝ,
  (∀ m : ℝ, |m| ≤ 2 → m * x^2 - 2*x - m + 1 < 0) →
  (-1 + Real.sqrt 7) / 2 < x ∧ x < (1 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_x_range_for_quadratic_inequality_l1647_164755


namespace NUMINAMATH_CALUDE_correct_calculation_l1647_164714

theorem correct_calculation (x : ℚ) : x * 9 = 153 → x * 6 = 102 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1647_164714


namespace NUMINAMATH_CALUDE_loads_to_wash_l1647_164789

theorem loads_to_wash (total : ℕ) (washed : ℕ) (h1 : total = 14) (h2 : washed = 8) :
  total - washed = 6 := by
  sorry

end NUMINAMATH_CALUDE_loads_to_wash_l1647_164789


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l1647_164707

/-- Given two rectangles with equal areas, where one rectangle measures 8 inches by 15 inches
    and the other is 4 inches long, prove that the width of the second rectangle is 30 inches. -/
theorem equal_area_rectangles_width (carol_length carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_length = 8)
    (h2 : carol_width = 15)
    (h3 : jordan_length = 4)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
    jordan_width = 30 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l1647_164707


namespace NUMINAMATH_CALUDE_system_solutions_l1647_164767

/-- The system of equations -/
def satisfies_system (a b c : ℝ) : Prop :=
  a^5 = 5*b^3 - 4*c ∧ b^5 = 5*c^3 - 4*a ∧ c^5 = 5*a^3 - 4*b

/-- The set of solutions -/
def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(0, 0, 0), (1, 1, 1), (-1, -1, -1), (2, 2, 2), (-2, -2, -2)}

/-- The main theorem -/
theorem system_solutions :
  ∀ (a b c : ℝ), satisfies_system a b c ↔ (a, b, c) ∈ solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solutions_l1647_164767


namespace NUMINAMATH_CALUDE_area_of_overlapping_squares_l1647_164772

/-- The area covered by two overlapping congruent squares -/
theorem area_of_overlapping_squares (side_length : ℝ) (h : side_length = 12) :
  let square_area := side_length ^ 2
  let overlap_area := square_area / 4
  let total_area := 2 * square_area - overlap_area
  total_area = 252 := by sorry

end NUMINAMATH_CALUDE_area_of_overlapping_squares_l1647_164772


namespace NUMINAMATH_CALUDE_segment_polynomial_l1647_164782

/-- Given a line segment AB with point T, prove that x^2 - 6√2x + 16 has roots equal to AT and TB lengths -/
theorem segment_polynomial (AB : ℝ) (T : ℝ) (h1 : 0 < T ∧ T < AB) 
  (h2 : AB - T = (1/2) * T) (h3 : T * (AB - T) = 16) :
  ∃ (AT TB : ℝ), AT = T ∧ TB = AB - T ∧ 
  (∀ x : ℝ, x^2 - 6 * Real.sqrt 2 * x + 16 = 0 ↔ (x = AT ∨ x = TB)) :=
sorry

end NUMINAMATH_CALUDE_segment_polynomial_l1647_164782


namespace NUMINAMATH_CALUDE_theater_parking_increase_l1647_164766

/-- Calculates the net increase in vehicles during a theater play --/
def net_increase_vehicles (play_duration : ℝ) 
  (car_arrival_rate car_departure_rate : ℝ)
  (motorcycle_arrival_rate motorcycle_departure_rate : ℝ)
  (van_arrival_rate van_departure_rate : ℝ) :
  (ℝ × ℝ × ℝ) :=
  let net_car_increase := (car_arrival_rate - car_departure_rate) * play_duration
  let net_motorcycle_increase := (motorcycle_arrival_rate - motorcycle_departure_rate) * play_duration
  let net_van_increase := (van_arrival_rate - van_departure_rate) * play_duration
  (net_car_increase, net_motorcycle_increase, net_van_increase)

/-- Theorem stating the net increase in vehicles during the theater play --/
theorem theater_parking_increase :
  let play_duration : ℝ := 2.5
  let car_arrival_rate : ℝ := 70
  let car_departure_rate : ℝ := 40
  let motorcycle_arrival_rate : ℝ := 120
  let motorcycle_departure_rate : ℝ := 60
  let van_arrival_rate : ℝ := 30
  let van_departure_rate : ℝ := 20
  net_increase_vehicles play_duration 
    car_arrival_rate car_departure_rate
    motorcycle_arrival_rate motorcycle_departure_rate
    van_arrival_rate van_departure_rate = (75, 150, 25) := by
  sorry

end NUMINAMATH_CALUDE_theater_parking_increase_l1647_164766


namespace NUMINAMATH_CALUDE_no_common_points_range_and_max_m_l1647_164708

open Real

noncomputable def f (x : ℝ) := log x
noncomputable def g (a : ℝ) (x : ℝ) := a * x
noncomputable def h (x : ℝ) := exp x / x

theorem no_common_points_range_and_max_m :
  (∃ a : ℝ, ∀ x : ℝ, x > 0 → f x ≠ g a x) ∧
  (∃ m : ℝ, ∀ x : ℝ, x > 1/2 → f x + m / x < h x) ∧
  (∀ m : ℝ, (∀ x : ℝ, x > 1/2 → f x + m / x < h x) → m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_no_common_points_range_and_max_m_l1647_164708


namespace NUMINAMATH_CALUDE_fourteen_segments_l1647_164794

/-- A right triangle with integer leg lengths -/
structure RightTriangle where
  de : ℕ
  ef : ℕ

/-- The number of distinct integer lengths of line segments from E to DF -/
def numIntegerSegments (t : RightTriangle) : ℕ := sorry

/-- Our specific right triangle -/
def triangle : RightTriangle := { de := 24, ef := 25 }

/-- The theorem to prove -/
theorem fourteen_segments : numIntegerSegments triangle = 14 := by sorry

end NUMINAMATH_CALUDE_fourteen_segments_l1647_164794


namespace NUMINAMATH_CALUDE_square_grid_perimeter_l1647_164710

theorem square_grid_perimeter (total_area : ℝ) (h_area : total_area = 144) :
  let side_length := Real.sqrt (total_area / 4)
  let perimeter := 4 * (2 * side_length)
  perimeter = 48 := by
sorry

end NUMINAMATH_CALUDE_square_grid_perimeter_l1647_164710


namespace NUMINAMATH_CALUDE_inequality_solution_system_of_inequalities_solution_l1647_164760

-- Part 1
theorem inequality_solution (x : ℝ) :
  (1/3 * x - (3*x + 4)/6 ≤ 2/3) ↔ (x ≥ -8) :=
sorry

-- Part 2
theorem system_of_inequalities_solution (x : ℝ) :
  (4*(x+1) ≤ 7*x + 13) ∧ ((x+2)/3 - x/2 > 1) ↔ (-3 ≤ x ∧ x < -2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_system_of_inequalities_solution_l1647_164760


namespace NUMINAMATH_CALUDE_perpendicular_bisector_trajectory_l1647_164703

theorem perpendicular_bisector_trajectory (Z₁ Z₂ : ℂ) (h : Z₁ ≠ Z₂) :
  {Z : ℂ | Complex.abs (Z - Z₁) = Complex.abs (Z - Z₂)} =
  {Z : ℂ | (Z - (Z₁ + Z₂) / 2) • (Z₂ - Z₁) = 0} :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_trajectory_l1647_164703


namespace NUMINAMATH_CALUDE_convex_n_gon_division_possible_values_l1647_164791

/-- A convex n-gon divided into three convex polygons -/
structure ConvexNGonDivision (n : ℕ) where
  (polygon1 : ℕ)  -- Number of sides of the first polygon
  (polygon2 : ℕ)  -- Number of sides of the second polygon
  (polygon3 : ℕ)  -- Number of sides of the third polygon
  (h1 : polygon1 = n)  -- First polygon has n sides
  (h2 : polygon2 > n)  -- Second polygon has more than n sides
  (h3 : polygon3 < n)  -- Third polygon has fewer than n sides

/-- The theorem stating the possible values of n -/
theorem convex_n_gon_division_possible_values :
  ∀ n : ℕ, (∃ d : ConvexNGonDivision n, True) → n = 4 ∨ n = 5 :=
sorry

end NUMINAMATH_CALUDE_convex_n_gon_division_possible_values_l1647_164791


namespace NUMINAMATH_CALUDE_consecutive_sum_equals_50_l1647_164724

/-- The sum of consecutive integers from a given start to an end -/
def sum_consecutive (start : Int) (count : Nat) : Int :=
  count * (2 * start + count.pred) / 2

/-- Proves that there are exactly 100 consecutive integers starting from -49 whose sum is 50 -/
theorem consecutive_sum_equals_50 : ∃! n : Nat, sum_consecutive (-49) n = 50 ∧ n > 0 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_sum_equals_50_l1647_164724


namespace NUMINAMATH_CALUDE_complex_square_equality_l1647_164716

theorem complex_square_equality (a b : ℝ) (i : ℂ) 
  (h1 : i^2 = -1) 
  (h2 : a + i = 2 - b*i) : 
  (a + b*i)^2 = 3 - 4*i := by
sorry

end NUMINAMATH_CALUDE_complex_square_equality_l1647_164716


namespace NUMINAMATH_CALUDE_chess_tournament_theorem_l1647_164797

/-- Represents a player's score sequence in the chess tournament -/
structure PlayerScore where
  round1 : ℕ
  round2 : ℕ
  round3 : ℕ
  round4 : ℕ

/-- Checks if a sequence is quadratic -/
def isQuadraticSequence (s : PlayerScore) : Prop :=
  ∃ a t r : ℕ, 
    s.round1 = a ∧
    s.round2 = a + t + r ∧
    s.round3 = a + 2*t + 4*r ∧
    s.round4 = a + 3*t + 9*r

/-- Checks if a sequence is arithmetic -/
def isArithmeticSequence (s : PlayerScore) : Prop :=
  ∃ b d : ℕ, 
    s.round1 = b ∧
    s.round2 = b + d ∧
    s.round3 = b + 2*d ∧
    s.round4 = b + 3*d

/-- Calculates the total score of a player -/
def totalScore (s : PlayerScore) : ℕ :=
  s.round1 + s.round2 + s.round3 + s.round4

/-- The main theorem -/
theorem chess_tournament_theorem 
  (playerA playerB : PlayerScore)
  (h1 : isQuadraticSequence playerA)
  (h2 : isArithmeticSequence playerB)
  (h3 : totalScore playerA = totalScore playerB)
  (h4 : totalScore playerA ≤ 25)
  (h5 : totalScore playerB ≤ 25) :
  playerA.round1 + playerA.round2 + playerB.round1 + playerB.round2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_theorem_l1647_164797


namespace NUMINAMATH_CALUDE_heath_age_l1647_164752

theorem heath_age (heath_age jude_age : ℕ) : 
  jude_age = 2 →
  heath_age + 5 = 3 * (jude_age + 5) →
  heath_age = 16 := by
sorry

end NUMINAMATH_CALUDE_heath_age_l1647_164752


namespace NUMINAMATH_CALUDE_evaluate_expression_l1647_164777

theorem evaluate_expression : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1647_164777


namespace NUMINAMATH_CALUDE_load_capacity_calculation_l1647_164726

theorem load_capacity_calculation (T H : ℝ) (L : ℝ) : 
  T = 3 → H = 9 → L = (35 * T^3) / H^3 → L = 35 / 27 := by
  sorry

end NUMINAMATH_CALUDE_load_capacity_calculation_l1647_164726


namespace NUMINAMATH_CALUDE_greatest_common_measure_of_segments_l1647_164713

/-- The greatest common measure of two segments of lengths 19 cm and 190 cm is 19 cm, not 1 cm -/
theorem greatest_common_measure_of_segments (segment1 : ℕ) (segment2 : ℕ) 
  (h1 : segment1 = 19) (h2 : segment2 = 190) :
  Nat.gcd segment1 segment2 = 19 ∧ Nat.gcd segment1 segment2 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_measure_of_segments_l1647_164713


namespace NUMINAMATH_CALUDE_expand_polynomial_l1647_164736

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l1647_164736


namespace NUMINAMATH_CALUDE_probability_gpa_at_least_3_5_l1647_164753

/-- Represents the possible grades a student can receive --/
inductive Grade
| A
| B
| C
| D

/-- Converts a grade to its point value --/
def gradeToPoints : Grade → ℕ
| Grade.A => 4
| Grade.B => 3
| Grade.C => 2
| Grade.D => 1

/-- Calculates the GPA given a list of grades --/
def calculateGPA (grades : List Grade) : ℚ :=
  (grades.map gradeToPoints).sum / 4

/-- Represents the probability of getting each grade in a subject --/
structure GradeProbability where
  probA : ℚ
  probB : ℚ
  probC : ℚ
  probD : ℚ

/-- The probability distribution for English grades --/
def englishProb : GradeProbability :=
  { probA := 1/6
  , probB := 1/4
  , probC := 7/12
  , probD := 0 }

/-- The probability distribution for History grades --/
def historyProb : GradeProbability :=
  { probA := 1/4
  , probB := 1/3
  , probC := 5/12
  , probD := 0 }

/-- Theorem stating the probability of getting a GPA of at least 3.5 --/
theorem probability_gpa_at_least_3_5 :
  let mathGrade := Grade.A
  let scienceGrade := Grade.A
  let probAtLeast3_5 := (englishProb.probA * historyProb.probA) +
                        (englishProb.probA * historyProb.probB) +
                        (englishProb.probB * historyProb.probA) +
                        (englishProb.probA * historyProb.probC) +
                        (englishProb.probC * historyProb.probA) +
                        (englishProb.probB * historyProb.probB)
  probAtLeast3_5 = 11/24 := by
  sorry

end NUMINAMATH_CALUDE_probability_gpa_at_least_3_5_l1647_164753


namespace NUMINAMATH_CALUDE_gratuity_calculation_l1647_164737

def dish_price_1 : ℝ := 10
def dish_price_2 : ℝ := 13
def dish_price_3 : ℝ := 17
def tip_percentage : ℝ := 0.1

theorem gratuity_calculation : 
  (dish_price_1 + dish_price_2 + dish_price_3) * tip_percentage = 4 := by
sorry

end NUMINAMATH_CALUDE_gratuity_calculation_l1647_164737


namespace NUMINAMATH_CALUDE_problem_statement_l1647_164734

theorem problem_statement (a : ℝ) (h : a = 5 - 2 * Real.sqrt 6) : a^2 - 10*a + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1647_164734


namespace NUMINAMATH_CALUDE_total_weight_of_good_fruits_l1647_164754

/-- Calculates the total weight in kilograms of fruits in good condition --/
def totalWeightOfGoodFruits (
  oranges bananas apples avocados grapes pineapples : ℕ
) (
  rottenOrangesPercent rottenBananasPercent rottenApplesPercent
  rottenAvocadosPercent rottenGrapesPercent rottenPineapplesPercent : ℚ
) (
  orangeWeight bananaWeight appleWeight avocadoWeight grapeWeight pineappleWeight : ℚ
) : ℚ :=
  let goodOranges := oranges - (oranges * rottenOrangesPercent).floor
  let goodBananas := bananas - (bananas * rottenBananasPercent).floor
  let goodApples := apples - (apples * rottenApplesPercent).floor
  let goodAvocados := avocados - (avocados * rottenAvocadosPercent).floor
  let goodGrapes := grapes - (grapes * rottenGrapesPercent).floor
  let goodPineapples := pineapples - (pineapples * rottenPineapplesPercent).floor

  (goodOranges * orangeWeight + goodBananas * bananaWeight +
   goodApples * appleWeight + goodAvocados * avocadoWeight +
   goodGrapes * grapeWeight + goodPineapples * pineappleWeight) / 1000

/-- The total weight of fruits in good condition is 204.585kg --/
theorem total_weight_of_good_fruits :
  totalWeightOfGoodFruits
    600 400 300 200 100 50
    (15/100) (5/100) (8/100) (10/100) (3/100) (20/100)
    150 120 100 80 5 1000 = 204585/1000 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_good_fruits_l1647_164754


namespace NUMINAMATH_CALUDE_friction_coefficient_inclined_plane_l1647_164758

/-- The coefficient of kinetic friction for a block sliding down an inclined plane,
    given that it reaches the bottom simultaneously with a hollow cylinder rolling without slipping -/
theorem friction_coefficient_inclined_plane (θ : Real) (g : Real) 
  (h1 : 0 < θ) (h2 : θ < π / 2) (h3 : g > 0) :
  let μ := (1 / 2) * Real.tan θ
  let a_cylinder := (1 / 2) * g * Real.sin θ
  let a_block := g * Real.sin θ - μ * g * Real.cos θ
  a_cylinder = a_block :=
by sorry

end NUMINAMATH_CALUDE_friction_coefficient_inclined_plane_l1647_164758


namespace NUMINAMATH_CALUDE_ticket_multiple_calculation_l1647_164728

/-- The multiple of fair tickets compared to baseball game tickets -/
def ticket_multiple (fair_tickets baseball_tickets : ℕ) : ℚ :=
  (fair_tickets - 6 : ℚ) / baseball_tickets

theorem ticket_multiple_calculation (fair_tickets baseball_tickets : ℕ) 
  (h1 : fair_tickets = ticket_multiple fair_tickets baseball_tickets * baseball_tickets + 6)
  (h2 : fair_tickets = 25)
  (h3 : baseball_tickets = 56) :
  ticket_multiple fair_tickets baseball_tickets = 19 / 56 := by
  sorry

#eval ticket_multiple 25 56

end NUMINAMATH_CALUDE_ticket_multiple_calculation_l1647_164728


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l1647_164780

theorem inverse_of_B_squared (B : Matrix (Fin 3) (Fin 3) ℝ) 
  (h : B⁻¹ = ![![2, -3, 0], ![0, -1, 0], ![0, 0, 5]]) : 
  (B^2)⁻¹ = ![![4, -3, 0], ![0, 1, 0], ![0, 0, 25]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l1647_164780
