import Mathlib

namespace NUMINAMATH_CALUDE_unique_c_for_quadratic_equation_l2079_207995

theorem unique_c_for_quadratic_equation :
  ∃! (c : ℝ), c ≠ 0 ∧
  (∃! (b : ℝ), b > 0 ∧
    (∃! (x : ℝ), x^2 + (b + 1/b) * x + c = 0)) ∧
  c = 1 := by
sorry

end NUMINAMATH_CALUDE_unique_c_for_quadratic_equation_l2079_207995


namespace NUMINAMATH_CALUDE_range_of_a_l2079_207950

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a-1)*x + 1 > 0) → (-1 < a ∧ a < 3) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2079_207950


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l2079_207941

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℚ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

theorem collinear_points_x_value :
  ∀ x : ℚ, collinear 2 7 10 x 25 (-2) → x = 89 / 23 :=
by
  sorry

#check collinear_points_x_value

end NUMINAMATH_CALUDE_collinear_points_x_value_l2079_207941


namespace NUMINAMATH_CALUDE_pet_store_cages_l2079_207991

def number_of_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) : ℕ :=
  (initial_puppies - sold_puppies) / puppies_per_cage

theorem pet_store_cages : number_of_cages 18 3 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2079_207991


namespace NUMINAMATH_CALUDE_prob_A_given_B_l2079_207956

-- Define the sample space
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define events A and B
def A : Finset Nat := {1, 3, 5}
def B : Finset Nat := {1, 2, 4, 5, 6}

-- Define the probability measure
def P (X : Finset Nat) : ℚ := (X.card : ℚ) / (Ω.card : ℚ)

-- Define conditional probability
def conditional_prob (X Y : Finset Nat) : ℚ := P (X ∩ Y) / P Y

-- Theorem statement
theorem prob_A_given_B : conditional_prob A B = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_given_B_l2079_207956


namespace NUMINAMATH_CALUDE_only_131_not_in_second_column_l2079_207930

def second_column (n : ℕ) : ℕ := 3 * n + 1

theorem only_131_not_in_second_column :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 400 →
    (31 = second_column n ∨
     94 = second_column n ∨
     331 = second_column n ∨
     907 = second_column n) ∧
    ¬(131 = second_column n) := by
  sorry

end NUMINAMATH_CALUDE_only_131_not_in_second_column_l2079_207930


namespace NUMINAMATH_CALUDE_complex_number_problem_l2079_207999

theorem complex_number_problem (z : ℂ) :
  (∃ (a : ℝ), z = Complex.I * a) →
  (∃ (b : ℝ), (z + 2)^2 - Complex.I * 8 = Complex.I * b) →
  z = -2 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2079_207999


namespace NUMINAMATH_CALUDE_fifteen_hundredth_day_is_wednesday_l2079_207966

/-- Days of the week represented as integers mod 7 -/
inductive DayOfWeek : Type
  | Monday : DayOfWeek
  | Tuesday : DayOfWeek
  | Wednesday : DayOfWeek
  | Thursday : DayOfWeek
  | Friday : DayOfWeek
  | Saturday : DayOfWeek
  | Sunday : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday
  | DayOfWeek.Sunday => DayOfWeek.Monday

/-- Function to get the day of the week after n days -/
def dayAfter (start : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => start
  | Nat.succ m => nextDay (dayAfter start m)

theorem fifteen_hundredth_day_is_wednesday :
  dayAfter DayOfWeek.Monday 1499 = DayOfWeek.Wednesday :=
by
  sorry


end NUMINAMATH_CALUDE_fifteen_hundredth_day_is_wednesday_l2079_207966


namespace NUMINAMATH_CALUDE_refrigerator_installation_cost_l2079_207954

theorem refrigerator_installation_cost 
  (purchase_price : ℝ) 
  (discount_percentage : ℝ)
  (transport_cost : ℝ)
  (profit_percentage : ℝ)
  (selling_price : ℝ)
  (h1 : purchase_price = 12500)
  (h2 : discount_percentage = 0.20)
  (h3 : transport_cost = 125)
  (h4 : profit_percentage = 0.12)
  (h5 : selling_price = 17920) :
  ∃ (installation_cost : ℝ),
    installation_cost = 295 ∧
    selling_price = 
      (purchase_price / (1 - discount_percentage)) * 
      (1 + profit_percentage) + 
      transport_cost + 
      installation_cost :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_installation_cost_l2079_207954


namespace NUMINAMATH_CALUDE_min_value_of_f_l2079_207970

/-- The quadratic function f(x) = 3x^2 + 6x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 2

/-- The minimum value of f(x) is -1 -/
theorem min_value_of_f : ∃ (m : ℝ), m = -1 ∧ ∀ (x : ℝ), f x ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2079_207970


namespace NUMINAMATH_CALUDE_min_value_trig_expression_min_value_achievable_l2079_207977

theorem min_value_trig_expression (θ : Real) :
  (1 / (2 - Real.cos θ ^ 2)) + (1 / (2 - Real.sin θ ^ 2)) ≥ 4 / 3 :=
sorry

theorem min_value_achievable :
  ∃ θ : Real, (1 / (2 - Real.cos θ ^ 2)) + (1 / (2 - Real.sin θ ^ 2)) = 4 / 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_trig_expression_min_value_achievable_l2079_207977


namespace NUMINAMATH_CALUDE_pin_combinations_l2079_207968

/-- The number of distinct digits in the PIN -/
def n : ℕ := 4

/-- The number of permutations of n distinct objects -/
def permutations (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem: The number of permutations of 4 distinct objects is 24 -/
theorem pin_combinations : permutations n = 24 := by
  sorry

end NUMINAMATH_CALUDE_pin_combinations_l2079_207968


namespace NUMINAMATH_CALUDE_top_square_is_one_l2079_207992

/-- Represents a 4x4 grid of squares --/
def Grid := Fin 4 → Fin 4 → ℕ

/-- Initial configuration of the grid --/
def initial_grid : Grid :=
  λ i j => 4 * i.val + j.val + 1

/-- Fold right half over left half --/
def fold_right_left (g : Grid) : Grid :=
  λ i j => g i (Fin.ofNat (3 - j.val))

/-- Fold left half over right half --/
def fold_left_right (g : Grid) : Grid :=
  λ i j => g i j

/-- Fold top half over bottom half --/
def fold_top_bottom (g : Grid) : Grid :=
  λ i j => g (Fin.ofNat (3 - i.val)) j

/-- Fold bottom half over top half --/
def fold_bottom_top (g : Grid) : Grid :=
  λ i j => g i j

/-- Perform all folds in sequence --/
def perform_folds (g : Grid) : Grid :=
  fold_bottom_top ∘ fold_top_bottom ∘ fold_left_right ∘ fold_right_left $ g

theorem top_square_is_one :
  (perform_folds initial_grid) 0 0 = 1 := by sorry

end NUMINAMATH_CALUDE_top_square_is_one_l2079_207992


namespace NUMINAMATH_CALUDE_is_vertex_of_parabola_l2079_207942

/-- The parabola equation -/
def f (x : ℝ) : ℝ := -4 * x^2 - 16 * x - 20

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (-2, -4)

/-- Theorem stating that the given point is the vertex of the parabola -/
theorem is_vertex_of_parabola :
  ∀ x : ℝ, f x ≤ f (vertex.1) ∧ f (vertex.1) = vertex.2 :=
sorry

end NUMINAMATH_CALUDE_is_vertex_of_parabola_l2079_207942


namespace NUMINAMATH_CALUDE_bridge_length_l2079_207908

/-- The length of a bridge given train parameters -/
theorem bridge_length
  (train_length : ℝ)
  (train_speed_kmh : ℝ)
  (crossing_time : ℝ)
  (h1 : train_length = 160)
  (h2 : train_speed_kmh = 45)
  (h3 : crossing_time = 30) :
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 215 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l2079_207908


namespace NUMINAMATH_CALUDE_divisibility_by_112_l2079_207959

theorem divisibility_by_112 (m : ℕ) (h1 : m > 0) (h2 : m % 2 = 1) (h3 : m % 3 ≠ 0) :
  112 ∣ ⌊4^m - (2 + Real.sqrt 2)^m⌋ := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_112_l2079_207959


namespace NUMINAMATH_CALUDE_sum_and_product_of_primes_l2079_207967

theorem sum_and_product_of_primes :
  ∀ p q : ℕ, Prime p → Prime q → p + q = 85 → p * q = 166 := by
sorry

end NUMINAMATH_CALUDE_sum_and_product_of_primes_l2079_207967


namespace NUMINAMATH_CALUDE_quadratic_equation_problem1_quadratic_equation_problem2_l2079_207933

-- Problem 1
theorem quadratic_equation_problem1 (x : ℝ) :
  (x - 5)^2 - 16 = 0 ↔ x = 9 ∨ x = 1 := by sorry

-- Problem 2
theorem quadratic_equation_problem2 (x : ℝ) :
  x^2 - 4*x + 1 = 0 ↔ x = 2 + Real.sqrt 3 ∨ x = 2 - Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_problem1_quadratic_equation_problem2_l2079_207933


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2079_207986

theorem inequality_equivalence (x : ℝ) :
  2 * |x - 2| - |x + 1| > 3 ↔ x < 0 ∨ x > 8 := by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2079_207986


namespace NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l2079_207971

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10 + sum_of_digits (n / 10))

theorem sum_of_digits_of_triangular_array_rows :
  ∃ (N : ℕ), triangular_sum N = 2080 ∧ sum_of_digits N = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l2079_207971


namespace NUMINAMATH_CALUDE_no_divisor_square_sum_l2079_207905

theorem no_divisor_square_sum (n : ℕ+) :
  ¬∃ d : ℕ+, (d ∣ 2 * n^2) ∧ ∃ x : ℕ, d^2 * n^2 + d^3 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_no_divisor_square_sum_l2079_207905


namespace NUMINAMATH_CALUDE_spheres_in_cone_radius_l2079_207951

/-- Represents a right circular cone -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ

/-- Theorem stating the radius of spheres in a cone under specific conditions -/
theorem spheres_in_cone_radius (c : Cone) (s1 s2 : Sphere) : 
  c.baseRadius = 6 ∧ 
  c.height = 15 ∧ 
  s1.radius = s2.radius ∧
  -- The spheres are tangent to each other, the side, and the base of the cone
  -- (This condition is implicitly assumed in the statement)
  True →
  s1.radius = 12 * Real.sqrt 29 / 29 :=
sorry

end NUMINAMATH_CALUDE_spheres_in_cone_radius_l2079_207951


namespace NUMINAMATH_CALUDE_unique_divisibility_by_99_l2079_207960

-- Define the structure of the number N
def N (a b : ℕ) : ℕ := a * 10^9 + 2018 * 10^5 + b * 10^4 + 2019

-- Define the divisibility condition
def is_divisible_by_99 (n : ℕ) : Prop := n % 99 = 0

-- State the theorem
theorem unique_divisibility_by_99 :
  ∃! (a b : ℕ), a < 10 ∧ b < 10 ∧ is_divisible_by_99 (N a b) :=
sorry

end NUMINAMATH_CALUDE_unique_divisibility_by_99_l2079_207960


namespace NUMINAMATH_CALUDE_middle_of_three_consecutive_sum_30_l2079_207949

/-- Given three consecutive natural numbers whose sum is 30, the middle number is 10. -/
theorem middle_of_three_consecutive_sum_30 :
  ∀ n : ℕ, n + (n + 1) + (n + 2) = 30 → n + 1 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_middle_of_three_consecutive_sum_30_l2079_207949


namespace NUMINAMATH_CALUDE_total_combinations_l2079_207929

/-- The number of students -/
def num_students : ℕ := 20

/-- The number of groups -/
def num_groups : ℕ := 4

/-- The minimum number of members in each group -/
def min_members_per_group : ℕ := 3

/-- The number of topics -/
def num_topics : ℕ := 5

/-- The number of ways to divide students into groups -/
def group_formations : ℕ := 165

/-- The number of ways to assign topics to groups -/
def topic_assignments : ℕ := 120

theorem total_combinations : 
  (group_formations * topic_assignments = 19800) ∧ 
  (num_students ≥ num_groups * min_members_per_group) ∧
  (num_topics > num_groups) :=
sorry

end NUMINAMATH_CALUDE_total_combinations_l2079_207929


namespace NUMINAMATH_CALUDE_limit_of_f_difference_quotient_l2079_207934

def f (x : ℝ) := x^2

theorem limit_of_f_difference_quotient :
  ∀ ε > 0, ∃ δ > 0, ∀ Δx ≠ 0, |Δx| < δ → 
    |((f Δx) - (f 0)) / Δx - 0| < ε := by sorry

end NUMINAMATH_CALUDE_limit_of_f_difference_quotient_l2079_207934


namespace NUMINAMATH_CALUDE_investment_strategy_optimal_l2079_207922

/-- Represents the maximum interest earned from a two-rate investment strategy --/
def max_interest (total_investment : ℝ) (rate1 rate2 : ℝ) (max_at_rate1 : ℝ) : ℝ :=
  rate1 * max_at_rate1 + rate2 * (total_investment - max_at_rate1)

/-- Theorem stating the maximum interest earned under given conditions --/
theorem investment_strategy_optimal (total_investment : ℝ) (rate1 rate2 : ℝ) (max_at_rate1 : ℝ)
    (h1 : total_investment = 25000)
    (h2 : rate1 = 0.07)
    (h3 : rate2 = 0.12)
    (h4 : max_at_rate1 = 11000) :
    max_interest total_investment rate1 rate2 max_at_rate1 = 2450 := by
  sorry

#eval max_interest 25000 0.07 0.12 11000

end NUMINAMATH_CALUDE_investment_strategy_optimal_l2079_207922


namespace NUMINAMATH_CALUDE_min_value_of_f_l2079_207932

def f (x : ℝ) := x^2 - 2*x + 3

theorem min_value_of_f :
  ∃ (min : ℝ), min = 2 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2079_207932


namespace NUMINAMATH_CALUDE_union_M_N_equals_real_l2079_207944

-- Define the sets M and N
def M : Set ℝ := {x | x > 0}
def N : Set ℝ := {x | x^2 ≥ x}

-- State the theorem
theorem union_M_N_equals_real : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_union_M_N_equals_real_l2079_207944


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l2079_207940

/-- Parabola type representing y^2 = 4x --/
structure Parabola where
  equation : ℝ → ℝ → Prop
  focus : ℝ × ℝ

/-- Line type --/
structure Line where
  passes_through : ℝ × ℝ → Prop

/-- Represents a point on the parabola --/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

theorem parabola_intersection_length 
  (p : Parabola) 
  (l : Line) 
  (A B : ParabolaPoint) 
  (h1 : p.equation = fun x y => y^2 = 4*x)
  (h2 : p.focus = (1, 0))
  (h3 : l.passes_through p.focus)
  (h4 : p.equation A.x A.y)
  (h5 : p.equation B.x B.y)
  (h6 : l.passes_through (A.x, A.y))
  (h7 : l.passes_through (B.x, B.y))
  (h8 : (A.x + B.x) / 2 = 3)
  : Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l2079_207940


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2079_207924

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | x < 1}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2079_207924


namespace NUMINAMATH_CALUDE_smallest_m_is_20_l2079_207946

/-- The set of complex numbers with real part between 1/2 and 2/3 -/
def T : Set ℂ := {z : ℂ | 1/2 ≤ z.re ∧ z.re ≤ 2/3}

/-- The property that for all n ≥ m, there exists a complex number z in T such that z^n = 1 -/
def has_nth_root_of_unity (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z ∈ T, z^n = 1

/-- 20 is the smallest positive integer satisfying the property -/
theorem smallest_m_is_20 :
  has_nth_root_of_unity 20 ∧ ∀ m : ℕ, 0 < m → m < 20 → ¬(has_nth_root_of_unity m) :=
sorry

end NUMINAMATH_CALUDE_smallest_m_is_20_l2079_207946


namespace NUMINAMATH_CALUDE_complex_absolute_value_sum_l2079_207925

theorem complex_absolute_value_sum : Complex.abs (2 - 4*Complex.I) + Complex.abs (2 + 4*Complex.I) = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_sum_l2079_207925


namespace NUMINAMATH_CALUDE_counterexample_exists_l2079_207981

theorem counterexample_exists : ∃ n : ℕ, ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 2)) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2079_207981


namespace NUMINAMATH_CALUDE_die_expected_value_l2079_207984

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- The strategy for two rolls -/
def strategy2 (d : Die) : Bool :=
  d.val ≥ 4

/-- The strategy for three rolls -/
def strategy3 (d : Die) : Bool :=
  d.val ≥ 5

/-- Expected value of a single roll -/
def E1 : ℚ := 3.5

/-- Expected value with two opportunities to roll -/
def E2 : ℚ := 4.25

/-- Expected value with three opportunities to roll -/
def E3 : ℚ := 14/3

theorem die_expected_value :
  (E2 = 4.25) ∧ (E3 = 14/3) := by
  sorry

end NUMINAMATH_CALUDE_die_expected_value_l2079_207984


namespace NUMINAMATH_CALUDE_inequality_solution_l2079_207983

theorem inequality_solution (x : ℝ) :
  (x^2 - 9) / (x^2 - 1) > 0 ↔ x > 3 ∨ x < -3 ∨ (-1 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2079_207983


namespace NUMINAMATH_CALUDE_square_area_error_l2079_207939

theorem square_area_error (s : ℝ) (h : s > 0) :
  let measured_side := s * 1.04
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error_percentage := (calculated_area - actual_area) / actual_area * 100
  area_error_percentage = 8.16 := by
    sorry

end NUMINAMATH_CALUDE_square_area_error_l2079_207939


namespace NUMINAMATH_CALUDE_angle_ratio_not_sufficient_for_right_triangle_l2079_207901

theorem angle_ratio_not_sufficient_for_right_triangle 
  (A B C : ℝ) (h_sum : A + B + C = 180) (h_ratio : A / 9 = B / 12 ∧ B / 12 = C / 15) :
  ¬ (A = 90 ∨ B = 90 ∨ C = 90) :=
sorry

end NUMINAMATH_CALUDE_angle_ratio_not_sufficient_for_right_triangle_l2079_207901


namespace NUMINAMATH_CALUDE_total_fish_l2079_207907

-- Define the number of gold fish and blue fish
def gold_fish : ℕ := 15
def blue_fish : ℕ := 7

-- State the theorem
theorem total_fish : gold_fish + blue_fish = 22 := by
  sorry

end NUMINAMATH_CALUDE_total_fish_l2079_207907


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2079_207961

def sequence_a (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => (sequence_a a₁ n) ^ 2

def monotonically_increasing (f : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, f (n + 1) > f n

theorem sufficient_but_not_necessary :
  (∀ a₁ : ℝ, a₁ = 2 → monotonically_increasing (sequence_a a₁)) ∧
  (∃ a₁ : ℝ, a₁ ≠ 2 ∧ monotonically_increasing (sequence_a a₁)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l2079_207961


namespace NUMINAMATH_CALUDE_square_property_necessary_not_sufficient_l2079_207993

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property a_{n+1}^2 = a_n * a_{n+2} for all n -/
def HasSquareProperty (a : Sequence) : Prop :=
  ∀ n : ℕ, (a (n + 1))^2 = a n * a (n + 2)

/-- Definition of a geometric sequence -/
def IsGeometric (a : Sequence) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The main theorem: HasSquareProperty is necessary but not sufficient for IsGeometric -/
theorem square_property_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometric a → HasSquareProperty a) ∧
  ¬(∀ a : Sequence, HasSquareProperty a → IsGeometric a) := by
  sorry


end NUMINAMATH_CALUDE_square_property_necessary_not_sufficient_l2079_207993


namespace NUMINAMATH_CALUDE_base3_20112_equals_176_l2079_207989

/-- Converts a base-3 digit to its base-10 equivalent --/
def base3ToBase10Digit (d : Nat) : Nat :=
  if d < 3 then d else 0

/-- Converts a base-3 number represented as a list of digits to its base-10 equivalent --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + base3ToBase10Digit d * 3^i) 0

theorem base3_20112_equals_176 :
  base3ToBase10 [2, 1, 1, 0, 2] = 176 := by
  sorry

end NUMINAMATH_CALUDE_base3_20112_equals_176_l2079_207989


namespace NUMINAMATH_CALUDE_max_bullet_speed_correct_l2079_207975

/-- Represents a ring moving on a segment --/
structure MovingRing where
  segmentLength : ℝ
  speed : ℝ

/-- Represents the game setup --/
structure GameSetup where
  ringCD : MovingRing
  ringEF : MovingRing
  ringGH : MovingRing
  AO : ℝ
  OP : ℝ
  PQ : ℝ

/-- The maximum bullet speed that allows passing through all rings --/
def maxBulletSpeed (setup : GameSetup) : ℝ :=
  4.5

/-- Theorem stating the maximum bullet speed --/
theorem max_bullet_speed_correct (setup : GameSetup) 
  (h1 : setup.ringCD.segmentLength = 20)
  (h2 : setup.ringEF.segmentLength = 20)
  (h3 : setup.ringGH.segmentLength = 20)
  (h4 : setup.ringCD.speed = 5)
  (h5 : setup.ringEF.speed = 9)
  (h6 : setup.ringGH.speed = 27)
  (h7 : setup.AO = 45)
  (h8 : setup.OP = 20)
  (h9 : setup.PQ = 20) :
  maxBulletSpeed setup = 4.5 := by
  sorry

#check max_bullet_speed_correct

end NUMINAMATH_CALUDE_max_bullet_speed_correct_l2079_207975


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2079_207913

theorem solution_set_inequality (x : ℝ) : 
  (x - 2) / (x + 1) ≤ 0 ↔ -1 < x ∧ x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2079_207913


namespace NUMINAMATH_CALUDE_environmental_policy_support_l2079_207900

theorem environmental_policy_support (men_support_rate : ℚ) (women_support_rate : ℚ)
  (men_count : ℕ) (women_count : ℕ) 
  (h1 : men_support_rate = 75 / 100)
  (h2 : women_support_rate = 70 / 100)
  (h3 : men_count = 200)
  (h4 : women_count = 800) :
  (men_support_rate * men_count + women_support_rate * women_count) / (men_count + women_count) = 71 / 100 :=
by sorry

end NUMINAMATH_CALUDE_environmental_policy_support_l2079_207900


namespace NUMINAMATH_CALUDE_expression_factorization_l2079_207985

theorem expression_factorization (x : ℝ) :
  (12 * x^4 - 27 * x^2 + 9) - (-3 * x^4 - 9 * x^2 + 6) = 3 * (5 * x^2 - 1) * (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2079_207985


namespace NUMINAMATH_CALUDE_q_investment_value_l2079_207915

/-- Represents the investment and profit division of two business partners -/
structure BusinessInvestment where
  p_investment : ℝ
  q_investment : ℝ
  profit_ratio : ℝ × ℝ

/-- Given the conditions of the problem, prove that q's investment is 45000 -/
theorem q_investment_value (b : BusinessInvestment) 
  (h1 : b.p_investment = 30000)
  (h2 : b.profit_ratio = (2, 3)) :
  b.q_investment = 45000 := by
  sorry

end NUMINAMATH_CALUDE_q_investment_value_l2079_207915


namespace NUMINAMATH_CALUDE_replacement_concentration_theorem_l2079_207987

/-- Represents a hydrochloric acid solution --/
structure HClSolution where
  total_mass : ℝ
  concentration : ℝ

/-- Calculates the mass of pure HCl in a solution --/
def pure_hcl_mass (solution : HClSolution) : ℝ :=
  solution.total_mass * solution.concentration

theorem replacement_concentration_theorem 
  (initial_solution : HClSolution)
  (drained_mass : ℝ)
  (final_solution : HClSolution)
  (replacement_solution : HClSolution)
  (h1 : initial_solution.total_mass = 300)
  (h2 : initial_solution.concentration = 0.2)
  (h3 : drained_mass = 25)
  (h4 : final_solution.total_mass = initial_solution.total_mass)
  (h5 : final_solution.concentration = 0.25)
  (h6 : replacement_solution.total_mass = drained_mass)
  (h7 : pure_hcl_mass final_solution = 
        pure_hcl_mass initial_solution - pure_hcl_mass replacement_solution + 
        pure_hcl_mass replacement_solution) :
  replacement_solution.concentration = 0.8 := by
  sorry

#check replacement_concentration_theorem

end NUMINAMATH_CALUDE_replacement_concentration_theorem_l2079_207987


namespace NUMINAMATH_CALUDE_rectangle_difference_l2079_207937

theorem rectangle_difference (x y : ℝ) : 
  y = x / 3 →
  2 * x + 2 * y = 32 →
  x^2 + y^2 = 17^2 →
  x - y = 8 := by
sorry

end NUMINAMATH_CALUDE_rectangle_difference_l2079_207937


namespace NUMINAMATH_CALUDE_lemonade_stand_solution_l2079_207931

/-- Represents the lemonade stand problem --/
def lemonade_stand_problem (G : ℚ) : Prop :=
  let glasses_per_gallon : ℚ := 16
  let cost_per_gallon : ℚ := 3.5
  let price_per_glass : ℚ := 1
  let glasses_drunk : ℚ := 5
  let glasses_unsold : ℚ := 6
  let net_profit : ℚ := 14
  let total_glasses := G * glasses_per_gallon
  let glasses_sold := total_glasses - glasses_drunk - glasses_unsold
  let revenue := glasses_sold * price_per_glass
  let cost := G * cost_per_gallon
  revenue - cost = net_profit

/-- The solution to the lemonade stand problem --/
theorem lemonade_stand_solution :
  ∃ G : ℚ, lemonade_stand_problem G ∧ G = 2 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_stand_solution_l2079_207931


namespace NUMINAMATH_CALUDE_ab_value_l2079_207980

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a + b = 3) : a * b = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2079_207980


namespace NUMINAMATH_CALUDE_sin_2theta_value_l2079_207927

theorem sin_2theta_value (θ : Real) 
  (h : Real.exp (2 * Real.log 2 * ((-2 : Real) + 2 * Real.sin θ)) + 3 = 
       Real.exp (Real.log 2 * ((1 / 2 : Real) + Real.sin θ))) : 
  Real.sin (2 * θ) = 3 * Real.sqrt 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l2079_207927


namespace NUMINAMATH_CALUDE_model_M_completion_time_l2079_207965

/-- The time (in minutes) it takes for a model M computer to complete the task -/
def model_M_time : ℝ := 24

/-- The time (in minutes) it takes for a model N computer to complete the task -/
def model_N_time : ℝ := 12

/-- The number of each model of computer used -/
def num_computers : ℕ := 8

/-- The time (in minutes) it takes for the combined computers to complete the task -/
def combined_time : ℝ := 1

theorem model_M_completion_time :
  (num_computers : ℝ) / model_M_time + (num_computers : ℝ) / model_N_time = 1 / combined_time :=
sorry

end NUMINAMATH_CALUDE_model_M_completion_time_l2079_207965


namespace NUMINAMATH_CALUDE_mary_total_time_l2079_207952

-- Define the given conditions
def mac_download_time : ℕ := 10
def windows_download_time : ℕ := 3 * mac_download_time
def audio_glitch_time : ℕ := 2 * 4
def video_glitch_time : ℕ := 6
def glitch_time : ℕ := audio_glitch_time + video_glitch_time
def non_glitch_time : ℕ := 2 * glitch_time

-- Theorem statement
theorem mary_total_time :
  mac_download_time + windows_download_time + glitch_time + non_glitch_time = 82 :=
by sorry

end NUMINAMATH_CALUDE_mary_total_time_l2079_207952


namespace NUMINAMATH_CALUDE_ball_return_to_start_l2079_207938

def circle_size : ℕ := 14
def step_size : ℕ := 3

theorem ball_return_to_start :
  ∀ (start : ℕ),
  start < circle_size →
  (∃ (n : ℕ), n > 0 ∧ (start + n * step_size) % circle_size = start) →
  (∀ (m : ℕ), 0 < m → m < circle_size → (start + m * step_size) % circle_size ≠ start) →
  (start + circle_size * step_size) % circle_size = start :=
by sorry

#check ball_return_to_start

end NUMINAMATH_CALUDE_ball_return_to_start_l2079_207938


namespace NUMINAMATH_CALUDE_missing_number_proof_l2079_207948

theorem missing_number_proof : 
  ∃ x : ℝ, 248 + x - Real.sqrt (- Real.sqrt 0) = 16 ∧ x = -232 := by sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2079_207948


namespace NUMINAMATH_CALUDE_isosceles_right_triangles_are_similar_l2079_207962

/-- An isosceles right triangle is a triangle with two equal sides and a right angle. -/
structure IsoscelesRightTriangle where
  side1 : ℝ
  side2 : ℝ
  hypotenuse : ℝ
  is_right_angle : side1^2 + side2^2 = hypotenuse^2
  is_isosceles : side1 = side2

/-- Two triangles are similar if their corresponding angles are equal and the ratios of corresponding sides are equal. -/
def are_similar (t1 t2 : IsoscelesRightTriangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ 
    t1.side1 = k * t2.side1 ∧
    t1.side2 = k * t2.side2 ∧
    t1.hypotenuse = k * t2.hypotenuse

/-- Theorem: Any two isosceles right triangles are similar. -/
theorem isosceles_right_triangles_are_similar (t1 t2 : IsoscelesRightTriangle) : 
  are_similar t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_right_triangles_are_similar_l2079_207962


namespace NUMINAMATH_CALUDE_tangent_perpendicular_theorem_l2079_207988

noncomputable def f (x : ℝ) : ℝ := x^4

def perpendicular_line (x y : ℝ) : Prop := x + 4*y - 8 = 0

def tangent_line (a b c : ℝ) (x y : ℝ) : Prop := a*x + b*y + c = 0

theorem tangent_perpendicular_theorem :
  ∃ (x₀ y₀ : ℝ), 
    y₀ = f x₀ ∧ 
    (∃ (a b c : ℝ), tangent_line a b c x₀ y₀ ∧ 
      (∀ (x y : ℝ), perpendicular_line x y → 
        (a*1 + b*4 = 0))) → 
    (∃ (x y : ℝ), tangent_line 4 (-1) (-3) x y) :=
sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_theorem_l2079_207988


namespace NUMINAMATH_CALUDE_letter_150_is_Z_l2079_207914

def repeating_sequence : ℕ → Char
  | n => let idx := n % 3
         if idx = 0 then 'Z'
         else if idx = 1 then 'X'
         else 'Y'

theorem letter_150_is_Z : repeating_sequence 150 = 'Z' := by
  sorry

end NUMINAMATH_CALUDE_letter_150_is_Z_l2079_207914


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2079_207912

theorem smaller_number_proof (x y : ℝ) (h1 : x - y = 9) (h2 : x + y = 46) :
  min x y = 18.5 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2079_207912


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l2079_207906

theorem book_arrangement_count : ℕ :=
  let math_books : ℕ := 4
  let english_books : ℕ := 5
  let math_arrangements : ℕ := Nat.factorial (math_books - 1)
  let english_arrangements : ℕ := Nat.factorial (english_books - 1)
  math_arrangements * english_arrangements

theorem book_arrangement_proof :
  book_arrangement_count = 144 :=
by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l2079_207906


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l2079_207904

theorem solution_set_implies_a_value (a : ℝ) :
  (∀ x : ℝ, |a * x + 2| < 4 ↔ -1 < x ∧ x < 3) →
  a = -2 :=
by sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l2079_207904


namespace NUMINAMATH_CALUDE_company_picnic_volleyball_teams_l2079_207947

theorem company_picnic_volleyball_teams 
  (managers : ℕ) 
  (employees : ℕ) 
  (teams : ℕ) 
  (h1 : managers = 23) 
  (h2 : employees = 7) 
  (h3 : teams = 6) : 
  (managers + employees) / teams = 5 := by
  sorry

end NUMINAMATH_CALUDE_company_picnic_volleyball_teams_l2079_207947


namespace NUMINAMATH_CALUDE_fourth_month_sale_l2079_207945

def sales_problem (sale1 sale2 sale3 sale5 sale6_target average_target : ℕ) : Prop :=
  let total_sales := 6 * average_target
  let known_sales := sale1 + sale2 + sale3 + sale5 + sale6_target
  let sale4 := total_sales - known_sales
  sale4 = 6350

theorem fourth_month_sale :
  sales_problem 5420 5660 6200 6500 7070 6200 :=
by sorry

end NUMINAMATH_CALUDE_fourth_month_sale_l2079_207945


namespace NUMINAMATH_CALUDE_train_length_proof_l2079_207972

/-- Proves that the length of a train is 250 meters given specific conditions --/
theorem train_length_proof (bridge_length : ℝ) (time_to_pass : ℝ) (train_speed_kmh : ℝ) :
  bridge_length = 150 →
  time_to_pass = 41.142857142857146 →
  train_speed_kmh = 35 →
  ∃ (train_length : ℝ), train_length = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l2079_207972


namespace NUMINAMATH_CALUDE_matrix_subtraction_result_l2079_207978

theorem matrix_subtraction_result :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![4, -3; 2, 8]
  let B : Matrix (Fin 2) (Fin 2) ℤ := !![1, 5; -3, 6]
  A - B = !![3, -8; 5, 2] := by sorry

end NUMINAMATH_CALUDE_matrix_subtraction_result_l2079_207978


namespace NUMINAMATH_CALUDE_specific_building_height_l2079_207918

/-- The height of a building with varying story heights -/
def building_height (total_stories : ℕ) (base_height : ℕ) (height_increase : ℕ) : ℕ :=
  let first_half := total_stories / 2
  let second_half := total_stories - first_half
  (first_half * base_height) + (second_half * (base_height + height_increase))

/-- Theorem stating the height of the specific building described in the problem -/
theorem specific_building_height :
  building_height 20 12 3 = 270 := by
  sorry

end NUMINAMATH_CALUDE_specific_building_height_l2079_207918


namespace NUMINAMATH_CALUDE_fifth_student_stickers_l2079_207957

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem fifth_student_stickers :
  let a₁ := 29  -- first term
  let d := 6    -- common difference
  let n := 5    -- position of the term we're looking for
  arithmetic_sequence a₁ d n = 53 := by sorry

end NUMINAMATH_CALUDE_fifth_student_stickers_l2079_207957


namespace NUMINAMATH_CALUDE_inequality_theorem_l2079_207964

theorem inequality_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2079_207964


namespace NUMINAMATH_CALUDE_corn_row_length_l2079_207973

/-- Calculates the length of a row of seeds in feet, given the space per seed and the number of seeds. -/
def row_length_in_feet (space_per_seed_inches : ℕ) (num_seeds : ℕ) : ℚ :=
  (space_per_seed_inches * num_seeds : ℚ) / 12

/-- Theorem stating that a row with 80 seeds, each requiring 18 inches of space, is 120 feet long. -/
theorem corn_row_length :
  row_length_in_feet 18 80 = 120 := by
  sorry

#eval row_length_in_feet 18 80

end NUMINAMATH_CALUDE_corn_row_length_l2079_207973


namespace NUMINAMATH_CALUDE_quadratic_minimum_l2079_207910

/-- The quadratic function f(x) = x^2 + 14x + 24 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 24

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 2*x + 14

theorem quadratic_minimum :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l2079_207910


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonals_l2079_207935

/-- A rectangular prism with different dimensions -/
structure RectangularPrism where
  length : ℝ
  width : ℝ
  height : ℝ
  length_ne_width : length ≠ width
  length_ne_height : length ≠ height
  width_ne_height : width ≠ height

/-- The number of face diagonals in a rectangular prism -/
def face_diagonals (prism : RectangularPrism) : ℕ := 12

/-- The number of space diagonals in a rectangular prism -/
def space_diagonals (prism : RectangularPrism) : ℕ := 4

/-- The total number of diagonals in a rectangular prism -/
def total_diagonals (prism : RectangularPrism) : ℕ :=
  face_diagonals prism + space_diagonals prism

theorem rectangular_prism_diagonals (prism : RectangularPrism) :
  total_diagonals prism = 16 ∧ space_diagonals prism = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonals_l2079_207935


namespace NUMINAMATH_CALUDE_course_selection_ways_l2079_207974

theorem course_selection_ways (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 6 → k = 3 → m = 1 →
  (n.choose m) * ((n - m).choose (k - m)) * ((k).choose m) = 180 :=
by sorry

end NUMINAMATH_CALUDE_course_selection_ways_l2079_207974


namespace NUMINAMATH_CALUDE_poly_expansions_general_poly_expansion_possible_m_values_l2079_207990

-- Define the polynomial expressions
def poly1 (x : ℝ) := (x + 2) * (x + 3)
def poly2 (x : ℝ) := (x + 2) * (x - 3)
def poly3 (x : ℝ) := (x - 2) * (x + 3)
def poly4 (x : ℝ) := (x - 2) * (x - 3)

-- Define the general polynomial expression
def polyGeneral (x a b : ℝ) := (x + a) * (x + b)

-- Theorem statements
theorem poly_expansions :
  (∀ x : ℝ, poly1 x = x^2 + 5*x + 6) ∧
  (∀ x : ℝ, poly2 x = x^2 - x - 6) ∧
  (∀ x : ℝ, poly3 x = x^2 + x - 6) ∧
  (∀ x : ℝ, poly4 x = x^2 - 5*x + 6) :=
sorry

theorem general_poly_expansion :
  ∀ x a b : ℝ, polyGeneral x a b = x^2 + (a + b)*x + a*b :=
sorry

theorem possible_m_values :
  ∀ a b m : ℤ, (∀ x : ℝ, polyGeneral x (a : ℝ) (b : ℝ) = x^2 + m*x + 5) →
  (m = 6 ∨ m = -6) :=
sorry

end NUMINAMATH_CALUDE_poly_expansions_general_poly_expansion_possible_m_values_l2079_207990


namespace NUMINAMATH_CALUDE_sector_perimeter_l2079_207919

theorem sector_perimeter (r : ℝ) (S : ℝ) (h1 : r = 2) (h2 : S = 8) :
  let α := 2 * S / r^2
  let L := r * α
  r + r + L = 12 := by sorry

end NUMINAMATH_CALUDE_sector_perimeter_l2079_207919


namespace NUMINAMATH_CALUDE_intersection_A_B_l2079_207928

-- Define set A
def A : Set ℝ := {x | |x| < 1}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l2079_207928


namespace NUMINAMATH_CALUDE_simplify_expression_l2079_207958

theorem simplify_expression (w : ℝ) : (5 - 2*w) - (4 + 5*w) = 1 - 7*w := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2079_207958


namespace NUMINAMATH_CALUDE_f_nonnegative_iff_a_in_range_l2079_207911

/-- The function f(x) defined in the problem -/
def f (a x : ℝ) : ℝ := x^2 - 2*x - |x - 1 - a| - |x - 2| + 4

/-- Theorem stating the equivalence between the non-negativity of f(x) and the range of a -/
theorem f_nonnegative_iff_a_in_range :
  (∀ x : ℝ, f a x ≥ 0) ↔ -2 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_f_nonnegative_iff_a_in_range_l2079_207911


namespace NUMINAMATH_CALUDE_tangent_slope_implies_abscissa_l2079_207996

noncomputable def f (x : ℝ) := Real.exp x + Real.exp (-x)

theorem tangent_slope_implies_abscissa (x : ℝ) :
  (deriv f x = 3/2) → x = Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_abscissa_l2079_207996


namespace NUMINAMATH_CALUDE_dividend_rate_calculation_l2079_207979

/-- Given a share worth 48 rupees, with a desired interest rate of 12%,
    and a market value of 36.00000000000001 rupees, the dividend rate is 16%. -/
theorem dividend_rate_calculation (share_value : ℝ) (interest_rate : ℝ) (market_value : ℝ)
    (h1 : share_value = 48)
    (h2 : interest_rate = 0.12)
    (h3 : market_value = 36.00000000000001) :
    (share_value * interest_rate / market_value) * 100 = 16 := by
  sorry

end NUMINAMATH_CALUDE_dividend_rate_calculation_l2079_207979


namespace NUMINAMATH_CALUDE_inverse_sum_reciprocals_l2079_207926

theorem inverse_sum_reciprocals (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 * a⁻¹ + 3 * b⁻¹)⁻¹ = a * b / (2 * b + 3 * a) :=
by sorry

end NUMINAMATH_CALUDE_inverse_sum_reciprocals_l2079_207926


namespace NUMINAMATH_CALUDE_complex_square_l2079_207909

theorem complex_square (a b : ℝ) (i : ℂ) (h : i * i = -1) (eq : a + i = 2 - b * i) :
  (a + b * i) ^ 2 = 3 - 4 * i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_l2079_207909


namespace NUMINAMATH_CALUDE_inequality_implications_l2079_207916

theorem inequality_implications (a b : ℝ) (h : a > b) :
  (a + 2 > b + 2) ∧
  (-a < -b) ∧
  (2 * a > 2 * b) ∧
  ∃ c : ℝ, ¬(a * c^2 > b * c^2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_implications_l2079_207916


namespace NUMINAMATH_CALUDE_min_sum_with_product_constraint_l2079_207997

theorem min_sum_with_product_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : a * b = a + b + 3) : 
  6 ≤ a + b ∧ ∀ (x y : ℝ), 0 < x → 0 < y → x * y = x + y + 3 → a + b ≤ x + y := by
  sorry

end NUMINAMATH_CALUDE_min_sum_with_product_constraint_l2079_207997


namespace NUMINAMATH_CALUDE_opposite_of_negative_seven_l2079_207923

theorem opposite_of_negative_seven : 
  (-(- 7 : ℤ)) = (7 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_seven_l2079_207923


namespace NUMINAMATH_CALUDE_ducks_park_solution_l2079_207936

def ducks_park_problem (initial_ducks : ℕ) (geese_arrive : ℕ) (ducks_arrive : ℕ) (geese_leave : ℕ) : Prop :=
  let initial_geese : ℕ := 2 * initial_ducks - 10
  let final_ducks : ℕ := initial_ducks + ducks_arrive
  let final_geese : ℕ := initial_geese - geese_leave
  final_geese - final_ducks = 1

theorem ducks_park_solution :
  ducks_park_problem 25 4 4 10 := by
  sorry

end NUMINAMATH_CALUDE_ducks_park_solution_l2079_207936


namespace NUMINAMATH_CALUDE_all_expressions_correct_l2079_207998

theorem all_expressions_correct (x y : ℝ) (h : x / y = 5 / 6) :
  (x + 2*y) / y = 17 / 6 ∧
  (2*x) / (3*y) = 5 / 9 ∧
  (y - x) / (2*y) = 1 / 12 ∧
  (x + y) / (2*y) = 11 / 12 ∧
  x / (y + x) = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_all_expressions_correct_l2079_207998


namespace NUMINAMATH_CALUDE_range_of_a_l2079_207955

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def increasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_even : even_function f)
  (h_incr : increasing_on_neg f)
  (h_cond : f a ≤ f 2) :
  a ≤ -2 ∨ a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2079_207955


namespace NUMINAMATH_CALUDE_y_value_l2079_207994

theorem y_value : ∃ y : ℝ, y ≠ 0 ∧ y = 2 * (1 / y) * (-y) - 4 → y = -6 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2079_207994


namespace NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l2079_207953

/-- The shaded area of a tiled floor with white quarter circles in each tile corner -/
theorem shaded_area_of_tiled_floor (floor_length floor_width tile_size radius : ℝ)
  (h_floor_length : floor_length = 12)
  (h_floor_width : floor_width = 16)
  (h_tile_size : tile_size = 2)
  (h_radius : radius = 1/2)
  (h_positive : floor_length > 0 ∧ floor_width > 0 ∧ tile_size > 0 ∧ radius > 0) :
  let num_tiles : ℝ := (floor_length * floor_width) / (tile_size * tile_size)
  let white_area_per_tile : ℝ := 4 * π * radius^2
  let shaded_area_per_tile : ℝ := tile_size * tile_size - white_area_per_tile
  num_tiles * shaded_area_per_tile = 192 - 48 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_of_tiled_floor_l2079_207953


namespace NUMINAMATH_CALUDE_fraction_equality_l2079_207969

theorem fraction_equality (a : ℕ+) :
  (a : ℚ) / (a + 35 : ℚ) = 7 / 10 → a = 82 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2079_207969


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_three_l2079_207982

theorem absolute_value_of_negative_three : |(-3 : ℤ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_three_l2079_207982


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2079_207921

theorem arithmetic_expression_equality : 1 + 6 * 2 - 3 + 5 * 4 / 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2079_207921


namespace NUMINAMATH_CALUDE_exists_between_elements_l2079_207963

/-- A sequence of natural numbers where each natural number appears exactly once -/
def UniqueNatSequence : Type := ℕ → ℕ

/-- The property that each natural number appears exactly once in the sequence -/
def isUniqueNatSequence (a : UniqueNatSequence) : Prop :=
  (∀ n : ℕ, ∃ k : ℕ, a k = n) ∧ 
  (∀ m n : ℕ, a m = a n → m = n)

/-- The main theorem -/
theorem exists_between_elements (a : UniqueNatSequence) (h : isUniqueNatSequence a) :
  ∀ n : ℕ, ∃ k : ℕ, k < n ∧ (a (n - k) < a n ∧ a n < a (n + k)) :=
by sorry

end NUMINAMATH_CALUDE_exists_between_elements_l2079_207963


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2079_207903

/-- An arithmetic sequence with first four terms a, x, b, and 2x -/
structure ArithmeticSequence (α : Type) [LinearOrderedField α] where
  a : α
  x : α
  b : α
  arithmetic_property : x - a = 2 * x - b

theorem arithmetic_sequence_ratio 
  {α : Type} [LinearOrderedField α] (seq : ArithmeticSequence α) :
  seq.a / seq.b = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2079_207903


namespace NUMINAMATH_CALUDE_rescue_mission_analysis_l2079_207902

def daily_distances : List Int := [14, -9, 8, -7, 13, -6, 10, -5]
def fuel_consumption : Rat := 1/2
def fuel_capacity : Nat := 29

theorem rescue_mission_analysis :
  let net_distance := daily_distances.sum
  let max_distance := daily_distances.scanl (· + ·) 0 |>.map abs |>.maximum
  let total_distance := daily_distances.map abs |>.sum
  let fuel_needed := fuel_consumption * total_distance - fuel_capacity
  (net_distance = 18 ∧ 
   max_distance = some 23 ∧ 
   fuel_needed = 7) := by sorry

end NUMINAMATH_CALUDE_rescue_mission_analysis_l2079_207902


namespace NUMINAMATH_CALUDE_missing_number_proof_l2079_207920

theorem missing_number_proof (x : ℝ) : x + Real.sqrt (-4 + 6 * 4 / 3) = 13 ↔ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l2079_207920


namespace NUMINAMATH_CALUDE_binary_110101_equals_53_l2079_207976

def binary_to_decimal (binary : List Bool) : Nat :=
  binary.enum.foldr (λ (i, b) acc => acc + if b then 2^i else 0) 0

theorem binary_110101_equals_53 :
  binary_to_decimal [true, false, true, false, true, true] = 53 := by
  sorry

end NUMINAMATH_CALUDE_binary_110101_equals_53_l2079_207976


namespace NUMINAMATH_CALUDE_total_doors_is_3600_l2079_207917

/-- Calculates the number of doors needed for a building with uniform floor plans -/
def doorsForUniformBuilding (floors : ℕ) (apartmentsPerFloor : ℕ) (doorsPerApartment : ℕ) : ℕ :=
  floors * apartmentsPerFloor * doorsPerApartment

/-- Calculates the number of doors needed for a building with alternating floor plans -/
def doorsForAlternatingBuilding (floors : ℕ) (oddApartments : ℕ) (evenApartments : ℕ) (doorsPerApartment : ℕ) : ℕ :=
  ((floors + 1) / 2 * oddApartments + (floors / 2) * evenApartments) * doorsPerApartment

/-- The total number of doors needed for all four buildings -/
def totalDoors : ℕ :=
  doorsForUniformBuilding 15 5 8 +
  doorsForUniformBuilding 25 6 10 +
  doorsForAlternatingBuilding 20 7 5 9 +
  doorsForAlternatingBuilding 10 8 4 7

theorem total_doors_is_3600 : totalDoors = 3600 := by
  sorry

end NUMINAMATH_CALUDE_total_doors_is_3600_l2079_207917


namespace NUMINAMATH_CALUDE_tissue_cost_l2079_207943

/-- Proves that the cost of each tissue is $0.05 given the problem conditions -/
theorem tissue_cost (boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (total_cost : ℚ) : 
  boxes = 10 →
  packs_per_box = 20 →
  tissues_per_pack = 100 →
  total_cost = 1000 →
  (total_cost / (boxes * packs_per_box * tissues_per_pack : ℚ)) = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_tissue_cost_l2079_207943
