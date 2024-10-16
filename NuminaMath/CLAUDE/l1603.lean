import Mathlib

namespace NUMINAMATH_CALUDE_reciprocal_of_negative_two_l1603_160339

theorem reciprocal_of_negative_two :
  ∀ x : ℚ, x * (-2) = 1 → x = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_two_l1603_160339


namespace NUMINAMATH_CALUDE_monika_beans_purchase_l1603_160395

def mall_cost : ℚ := 250
def movie_cost : ℚ := 24
def num_movies : ℕ := 3
def bean_cost : ℚ := 1.25
def total_spent : ℚ := 347

theorem monika_beans_purchase :
  (total_spent - (mall_cost + movie_cost * num_movies)) / bean_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_monika_beans_purchase_l1603_160395


namespace NUMINAMATH_CALUDE_triangle_third_vertex_l1603_160390

/-- Given an obtuse triangle with vertices at (8, 6), (0, 0), and (x, 0),
    if the area of the triangle is 48 square units, then x = 16 or x = -16 -/
theorem triangle_third_vertex (x : ℝ) : 
  let v1 : ℝ × ℝ := (8, 6)
  let v2 : ℝ × ℝ := (0, 0)
  let v3 : ℝ × ℝ := (x, 0)
  let triangle_area := (1/2 : ℝ) * |v1.1 * (v2.2 - v3.2) + v2.1 * (v3.2 - v1.2) + v3.1 * (v1.2 - v2.2)|
  (triangle_area = 48) → (x = 16 ∨ x = -16) :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_vertex_l1603_160390


namespace NUMINAMATH_CALUDE_student_count_theorem_l1603_160376

def valid_student_count (n : ℕ) : Prop :=
  n < 50 ∧ n % 6 = 5 ∧ n % 3 = 2

theorem student_count_theorem : 
  {n : ℕ | valid_student_count n} = {5, 11, 17, 23, 29, 35, 41, 47} :=
sorry

end NUMINAMATH_CALUDE_student_count_theorem_l1603_160376


namespace NUMINAMATH_CALUDE_power_of_power_three_squared_four_l1603_160330

theorem power_of_power_three_squared_four : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_three_squared_four_l1603_160330


namespace NUMINAMATH_CALUDE_min_value_theorem_l1603_160345

-- Define the function f
def f (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

-- State the theorem
theorem min_value_theorem (a b c d : ℝ) (h1 : a < (2/3) * b) 
  (h2 : ∀ x y : ℝ, x < y → f a b c d x < f a b c d y) :
  ∃ m : ℝ, m = 1 ∧ ∀ k : ℝ, k = c / (2*b - 3*a) → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1603_160345


namespace NUMINAMATH_CALUDE_eight_points_on_circle_theorem_l1603_160379

/-- A point with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- The theorem statement -/
theorem eight_points_on_circle_theorem
  (p : ℕ) (n : ℕ) (points : Finset IntPoint) :
  Nat.Prime p →
  p % 2 = 1 →
  n > 0 →
  points.card = 8 →
  (∀ pt ∈ points, ∃ (x y : ℤ), pt = ⟨x, y⟩) →
  (∃ (center : IntPoint) (r : ℤ), r^2 = (p^n)^2 / 4 ∧
    ∀ pt ∈ points, (pt.x - center.x)^2 + (pt.y - center.y)^2 = r^2) →
  ∃ (a b c : IntPoint), a ∈ points ∧ b ∈ points ∧ c ∈ points ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    ∃ (ab bc ca : ℤ),
      ab = (a.x - b.x)^2 + (a.y - b.y)^2 ∧
      bc = (b.x - c.x)^2 + (b.y - c.y)^2 ∧
      ca = (c.x - a.x)^2 + (c.y - a.y)^2 ∧
      ab % p^(n+1) = 0 ∧ bc % p^(n+1) = 0 ∧ ca % p^(n+1) = 0 :=
by sorry

end NUMINAMATH_CALUDE_eight_points_on_circle_theorem_l1603_160379


namespace NUMINAMATH_CALUDE_min_value_theorem_l1603_160319

theorem min_value_theorem (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) (h5 : a * b = 1/4) :
  ∃ (min_val : ℝ), min_val = 4 + 4 * Real.sqrt 2 / 3 ∧
    ∀ (x y : ℝ), 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ x * y = 1/4 →
      1 / (1 - x) + 2 / (1 - y) ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1603_160319


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1603_160323

theorem sufficient_not_necessary (x : ℝ) :
  (x > 2 → abs (x - 1) > 1) ∧ ¬(abs (x - 1) > 1 → x > 2) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1603_160323


namespace NUMINAMATH_CALUDE_audrey_heracles_age_ratio_l1603_160320

/-- Proves that the ratio of Audrey's age in 3 years to Heracles' current age is 2:1 -/
theorem audrey_heracles_age_ratio :
  let heracles_age : ℕ := 10
  let audrey_age : ℕ := heracles_age + 7
  let audrey_age_in_3_years : ℕ := audrey_age + 3
  (audrey_age_in_3_years : ℚ) / heracles_age = 2 := by
  sorry

end NUMINAMATH_CALUDE_audrey_heracles_age_ratio_l1603_160320


namespace NUMINAMATH_CALUDE_problem_statement_l1603_160303

theorem problem_statement :
  (∀ x : ℝ, x^2 - 8*x + 17 > 0) ∧
  (∀ x : ℝ, (x + 2)^2 - (x - 3)^2 ≥ 0 → x ≥ 1/2) ∧
  (∃ n : ℕ, 11 ∣ 6*n^2 - 7) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1603_160303


namespace NUMINAMATH_CALUDE_largest_C_for_divisibility_by_4_l1603_160350

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem largest_C_for_divisibility_by_4 :
  ∃ (B : ℕ) (h_B : B < 10),
    ∀ (C : ℕ) (h_C : C < 10),
      is_divisible_by_4 (4000000 + 600000 + B * 100000 + 41800 + C) →
      C ≤ 8 ∧
      is_divisible_by_4 (4000000 + 600000 + B * 100000 + 41800 + 8) :=
by sorry

end NUMINAMATH_CALUDE_largest_C_for_divisibility_by_4_l1603_160350


namespace NUMINAMATH_CALUDE_max_interesting_in_five_l1603_160391

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for prime numbers -/
def is_prime (n : ℕ) : Prop := sorry

/-- Predicate for interesting numbers -/
def is_interesting (n : ℕ) : Prop := is_prime (sum_of_digits n)

/-- Theorem: At most 4 out of 5 consecutive natural numbers can be interesting -/
theorem max_interesting_in_five (n : ℕ) : 
  ∃ (k : Fin 5), ¬is_interesting (n + k) :=
sorry

end NUMINAMATH_CALUDE_max_interesting_in_five_l1603_160391


namespace NUMINAMATH_CALUDE_f_composition_negative_two_l1603_160349

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then 1 - Real.sqrt x else 2^x

theorem f_composition_negative_two : f (f (-2)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_two_l1603_160349


namespace NUMINAMATH_CALUDE_unit_conversions_l1603_160300

-- Define the conversion rates
def kg_per_ton : ℝ := 1000
def sq_dm_per_sq_m : ℝ := 100

-- Define the theorem
theorem unit_conversions :
  (8 : ℝ) + 800 / kg_per_ton = 8.8 ∧
  6.32 * sq_dm_per_sq_m = 632 :=
by sorry

end NUMINAMATH_CALUDE_unit_conversions_l1603_160300


namespace NUMINAMATH_CALUDE_geometric_sequence_a4_l1603_160306

/-- Represents a geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: In a geometric sequence where a_2 = 4 and a_6 = 16, a_4 = 8 -/
theorem geometric_sequence_a4 (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a) 
    (h_a2 : a 2 = 4) 
    (h_a6 : a 6 = 16) : 
  a 4 = 8 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_a4_l1603_160306


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1603_160375

/-- A geometric sequence with a_3 = 2 and a_6 = 16 has a common ratio of 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * (a 1))
  (h_a3 : a 3 = 2)
  (h_a6 : a 6 = 16) : 
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1603_160375


namespace NUMINAMATH_CALUDE_expression_evaluation_l1603_160317

theorem expression_evaluation :
  36 + (150 / 15) + (12^2 * 5) - 300 - (270 / 9) = 436 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1603_160317


namespace NUMINAMATH_CALUDE_twins_shirts_l1603_160396

/-- The number of shirts Hazel and Razel have in total -/
def total_shirts (hazel_shirts : ℕ) (razel_shirts : ℕ) : ℕ :=
  hazel_shirts + razel_shirts

/-- Theorem: If Hazel received 6 shirts and Razel received twice the number of shirts as Hazel,
    then the total number of shirts they have is 18. -/
theorem twins_shirts :
  let hazel_shirts : ℕ := 6
  let razel_shirts : ℕ := 2 * hazel_shirts
  total_shirts hazel_shirts razel_shirts = 18 := by
sorry

end NUMINAMATH_CALUDE_twins_shirts_l1603_160396


namespace NUMINAMATH_CALUDE_smallest_r_minus_p_l1603_160332

theorem smallest_r_minus_p : ∃ (p q r : ℕ+),
  (p * q * r = 362880) ∧   -- 9! = 362880
  (p < q) ∧ (q < r) ∧
  ∀ (p' q' r' : ℕ+),
    (p' * q' * r' = 362880) →
    (p' < q') → (q' < r') →
    (r - p : ℤ) ≤ (r' - p' : ℤ) ∧
  (r - p : ℤ) = 219 := by
  sorry

end NUMINAMATH_CALUDE_smallest_r_minus_p_l1603_160332


namespace NUMINAMATH_CALUDE_ice_pop_cost_l1603_160336

theorem ice_pop_cost (ice_pop_price : ℝ) (pencil_price : ℝ) (ice_pops_sold : ℕ) (pencils_bought : ℕ) : 
  ice_pop_price = 1.50 →
  pencil_price = 1.80 →
  ice_pops_sold = 300 →
  pencils_bought = 100 →
  ice_pops_sold * ice_pop_price = pencils_bought * pencil_price →
  ice_pop_price - (ice_pops_sold * ice_pop_price - pencils_bought * pencil_price) / ice_pops_sold = 0.90 := by
sorry

end NUMINAMATH_CALUDE_ice_pop_cost_l1603_160336


namespace NUMINAMATH_CALUDE_unique_valid_number_l1603_160357

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a > b ∧ b > c ∧
    (100 * a + 10 * b + c) - (100 * c + 10 * b + a) = n

theorem unique_valid_number :
  ∃! n, is_valid_number n ∧ n = 495 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l1603_160357


namespace NUMINAMATH_CALUDE_sum_of_roots_l1603_160365

theorem sum_of_roots (k d : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) 
  (h₂ : 5 * x₁^2 - k * x₁ = d) (h₃ : 5 * x₂^2 - k * x₂ = d) : 
  x₁ + x₂ = k / 5 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1603_160365


namespace NUMINAMATH_CALUDE_intersection_nonempty_range_l1603_160333

def g (x a : ℝ) : ℝ := x^2 + (a-1)*x + a - 2*a^2

def h (x : ℝ) : ℝ := (x-1)^2

def A (a : ℝ) : Set ℝ := {x | g x a > 0}

def B : Set ℝ := {x | h x < 1}

def f (x a : ℝ) : ℝ := x * (g x a)

def C (a : ℝ) : Set ℝ := {x | f x a > 0}

theorem intersection_nonempty_range (a : ℝ) :
  (A a ∩ B).Nonempty ∧ (C a ∩ B).Nonempty ↔ 
  (1/3 < a ∧ a < 2) ∨ (-1/2 < a ∧ a < 1/3) := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_range_l1603_160333


namespace NUMINAMATH_CALUDE_circle_area_difference_l1603_160304

theorem circle_area_difference (π : ℝ) (h_π : π > 0) : 
  let R := 18 / π  -- Radius of larger circle
  let r := R / 2   -- Radius of smaller circle
  (π * R^2 - π * r^2) = 243 / π := by
sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1603_160304


namespace NUMINAMATH_CALUDE_set_operations_l1603_160385

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 2)}
def N : Set ℝ := {x | x < 1 ∨ x > 3}

-- State the theorem
theorem set_operations :
  (M ∪ N = {x | x < 1 ∨ x ≥ 2}) ∧
  (M ∩ (Nᶜ) = {x | 2 ≤ x ∧ x ≤ 3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1603_160385


namespace NUMINAMATH_CALUDE_three_times_x_not_much_different_from_two_l1603_160398

theorem three_times_x_not_much_different_from_two :
  ∃ (x : ℝ), 3 * x - 2 ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_three_times_x_not_much_different_from_two_l1603_160398


namespace NUMINAMATH_CALUDE_prime_equivalence_l1603_160331

theorem prime_equivalence (k : ℕ) (h : ℕ) (n : ℕ) 
  (h_odd : Odd h) 
  (h_bound : h < 2^k) 
  (n_def : n = 2^k * h + 1) : 
  Nat.Prime n ↔ ∃ a : ℕ, a^((n-1)/2) % n = n - 1 := by
  sorry

end NUMINAMATH_CALUDE_prime_equivalence_l1603_160331


namespace NUMINAMATH_CALUDE_dividend_proof_l1603_160312

theorem dividend_proof (divisor quotient dividend : ℕ) : 
  divisor = 12 → quotient = 999809 → dividend = 11997708 → 
  dividend / divisor = quotient ∧ dividend % divisor = 0 := by
  sorry

end NUMINAMATH_CALUDE_dividend_proof_l1603_160312


namespace NUMINAMATH_CALUDE_flea_can_reach_all_naturals_l1603_160384

def jump_length (k : ℕ) : ℕ := 2^k + 1

theorem flea_can_reach_all_naturals :
  ∀ n : ℕ, ∃ (jumps : List (ℕ × Bool)), 
    (jumps.foldl (λ acc (len, dir) => if dir then acc + len else acc - len) 0 : ℤ) = n ∧
    ∀ k, k < jumps.length → (jumps.get ⟨k, by sorry⟩).1 = jump_length k :=
by sorry

end NUMINAMATH_CALUDE_flea_can_reach_all_naturals_l1603_160384


namespace NUMINAMATH_CALUDE_max_intersecting_chords_2017_l1603_160348

/-- The maximum number of intersecting chords for a circle with n points -/
def max_intersecting_chords (n : ℕ) : ℕ :=
  let k := (n - 1) / 2
  k * (n - 1 - k) + (n - 1)

/-- Theorem stating the maximum number of intersecting chords for 2017 points -/
theorem max_intersecting_chords_2017 :
  max_intersecting_chords 2017 = 1018080 := by
  sorry

#eval max_intersecting_chords 2017

end NUMINAMATH_CALUDE_max_intersecting_chords_2017_l1603_160348


namespace NUMINAMATH_CALUDE_even_function_monotonicity_l1603_160337

-- Define an even function
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- Define monotonically decreasing in an interval
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y ≤ f x

-- Define monotonically increasing in an interval
def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- Theorem statement
theorem even_function_monotonicity 
  (f : ℝ → ℝ) 
  (h_even : even_function f) 
  (h_decreasing : monotone_decreasing_on f (-2) (-1)) :
  monotone_increasing_on f 1 2 ∧ 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f 1 ≤ f x) ∧
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f x ≤ f 2) :=
by sorry

end NUMINAMATH_CALUDE_even_function_monotonicity_l1603_160337


namespace NUMINAMATH_CALUDE_f_minimum_value_l1603_160363

def f (x : ℕ+) : ℚ := (x.val^2 + 33) / x.val

theorem f_minimum_value : ∀ x : ℕ+, f x ≥ 23/2 := by sorry

end NUMINAMATH_CALUDE_f_minimum_value_l1603_160363


namespace NUMINAMATH_CALUDE_revenue_comparison_l1603_160301

theorem revenue_comparison (last_year_revenue : ℝ) : 
  let projected_revenue := last_year_revenue * 1.20
  let actual_revenue := last_year_revenue * 0.90
  actual_revenue / projected_revenue = 0.75 := by
sorry

end NUMINAMATH_CALUDE_revenue_comparison_l1603_160301


namespace NUMINAMATH_CALUDE_complex_parts_l1603_160368

theorem complex_parts (z : ℂ) : z = (1 + Real.sqrt 3) * Complex.I →
  Complex.re z = 0 ∧ Complex.im z = 1 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_parts_l1603_160368


namespace NUMINAMATH_CALUDE_lilys_remaining_balance_l1603_160325

/-- Calculates the remaining balance in Lily's account after purchases --/
def remaining_balance (initial_balance : ℕ) (shirt_cost : ℕ) : ℕ :=
  initial_balance - shirt_cost - (3 * shirt_cost)

/-- Theorem stating that Lily's remaining balance is 27 dollars --/
theorem lilys_remaining_balance :
  remaining_balance 55 7 = 27 := by
  sorry

end NUMINAMATH_CALUDE_lilys_remaining_balance_l1603_160325


namespace NUMINAMATH_CALUDE_average_of_multiples_10_to_100_l1603_160377

def multiples_of_10 : List ℕ := [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem average_of_multiples_10_to_100 : 
  average multiples_of_10 = 55 := by
  sorry

end NUMINAMATH_CALUDE_average_of_multiples_10_to_100_l1603_160377


namespace NUMINAMATH_CALUDE_initial_ratio_is_5_to_7_l1603_160362

/-- Represents the composition of an alloy -/
structure Alloy where
  zinc : ℝ
  copper : ℝ

/-- Proves that the initial ratio of zinc to copper in the alloy is 5:7 -/
theorem initial_ratio_is_5_to_7 (initial : Alloy) 
  (h1 : initial.zinc + initial.copper = 6)  -- Initial total weight is 6 kg
  (h2 : (initial.zinc + 8) / initial.copper = 3)  -- New ratio after adding 8 kg zinc is 3:1
  : initial.zinc / initial.copper = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_ratio_is_5_to_7_l1603_160362


namespace NUMINAMATH_CALUDE_five_students_two_groups_l1603_160369

/-- The number of ways to assign n students to k groups, where each student
    must be assigned to exactly one group. -/
def assignmentWays (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to assign 5 students to 2 groups is 32. -/
theorem five_students_two_groups : assignmentWays 5 2 = 32 := by
  sorry

end NUMINAMATH_CALUDE_five_students_two_groups_l1603_160369


namespace NUMINAMATH_CALUDE_alice_paid_fifteen_per_acorn_l1603_160367

/-- The price Alice paid for each acorn -/
def alice_price_per_acorn (alice_acorn_count : ℕ) (alice_bob_price_ratio : ℕ) (bob_total_price : ℕ) : ℚ :=
  (alice_bob_price_ratio * bob_total_price : ℚ) / alice_acorn_count

/-- Theorem stating that Alice paid $15 for each acorn -/
theorem alice_paid_fifteen_per_acorn :
  alice_price_per_acorn 3600 9 6000 = 15 := by
  sorry

end NUMINAMATH_CALUDE_alice_paid_fifteen_per_acorn_l1603_160367


namespace NUMINAMATH_CALUDE_perimeter_of_square_with_semicircular_arcs_l1603_160356

/-- The perimeter of a region bounded by four semicircular arcs constructed on the sides of a square with side length 1/π is equal to 2. -/
theorem perimeter_of_square_with_semicircular_arcs (π : ℝ) (h : π > 0) : 
  let side_length : ℝ := 1 / π
  let semicircle_length : ℝ := π * side_length / 2
  let num_semicircles : ℕ := 4
  num_semicircles * semicircle_length = 2 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_square_with_semicircular_arcs_l1603_160356


namespace NUMINAMATH_CALUDE_square_x_plus_2y_l1603_160394

theorem square_x_plus_2y (x y : ℝ) 
  (h1 : x * (x + y) = 40) 
  (h2 : y * (x + y) = 90) : 
  (x + 2*y)^2 = 310 + 8100/130 := by
  sorry

end NUMINAMATH_CALUDE_square_x_plus_2y_l1603_160394


namespace NUMINAMATH_CALUDE_constant_function_theorem_l1603_160382

theorem constant_function_theorem (f : ℝ → ℝ) : 
  (∀ x y z : ℝ, f (x * y) + f (x * z) ≥ f x * f (y * z) + 1) → 
  (∀ x : ℝ, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_constant_function_theorem_l1603_160382


namespace NUMINAMATH_CALUDE_fence_perimeter_is_112_l1603_160318

-- Define the parameters of the fence
def total_posts : ℕ := 28
def posts_on_long_side : ℕ := 6
def gap_between_posts : ℕ := 4

-- Define the function to calculate the perimeter
def fence_perimeter : ℕ := 
  let posts_on_short_side := (total_posts - 2 * posts_on_long_side + 2) / 2 + 1
  let long_side_length := (posts_on_long_side - 1) * gap_between_posts
  let short_side_length := (posts_on_short_side - 1) * gap_between_posts
  2 * (long_side_length + short_side_length)

-- Theorem statement
theorem fence_perimeter_is_112 : fence_perimeter = 112 := by
  sorry

end NUMINAMATH_CALUDE_fence_perimeter_is_112_l1603_160318


namespace NUMINAMATH_CALUDE_parallel_neither_sufficient_nor_necessary_l1603_160392

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem parallel_neither_sufficient_nor_necessary 
  (a b : Line) (α : Plane) 
  (h : line_in_plane b α) :
  ¬(∀ a b α, parallel_lines a b → parallel_line_plane a α) ∧ 
  ¬(∀ a b α, parallel_line_plane a α → parallel_lines a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_neither_sufficient_nor_necessary_l1603_160392


namespace NUMINAMATH_CALUDE_octagon_side_length_l1603_160327

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  w : Point
  x : Point
  y : Point
  z : Point

/-- Represents an octagon -/
structure Octagon where
  a : Point
  b : Point
  c : Point
  d : Point
  e : Point
  f : Point
  g : Point
  h : Point

def is_on_line (p q r : Point) : Prop := sorry

def is_equilateral (oct : Octagon) : Prop := sorry

def is_convex (oct : Octagon) : Prop := sorry

def side_length (oct : Octagon) : ℝ := sorry

theorem octagon_side_length 
  (rect : Rectangle)
  (oct : Octagon)
  (h1 : rect.z.x - rect.w.x = 10)
  (h2 : rect.y.y - rect.z.y = 8)
  (h3 : is_on_line rect.w oct.a rect.z)
  (h4 : is_on_line rect.w oct.b rect.z)
  (h5 : is_on_line rect.z oct.c rect.y)
  (h6 : is_on_line rect.z oct.d rect.y)
  (h7 : is_on_line rect.y oct.e rect.w)
  (h8 : is_on_line rect.y oct.f rect.w)
  (h9 : is_on_line rect.x oct.g rect.w)
  (h10 : is_on_line rect.x oct.h rect.w)
  (h11 : oct.a.x - rect.w.x = rect.z.x - oct.b.x)
  (h12 : oct.a.x - rect.w.x ≤ 5)
  (h13 : is_equilateral oct)
  (h14 : is_convex oct) :
  side_length oct = -9 + Real.sqrt 652 := by sorry

end NUMINAMATH_CALUDE_octagon_side_length_l1603_160327


namespace NUMINAMATH_CALUDE_heaviest_tv_weight_difference_l1603_160351

/-- Represents the dimensions of a TV screen -/
structure TVDimensions where
  width : ℕ
  height : ℕ

/-- Calculates the area of a TV screen given its dimensions -/
def screenArea (d : TVDimensions) : ℕ := d.width * d.height

/-- Calculates the weight of a TV in ounces given its screen area -/
def tvWeight (area : ℕ) : ℕ := area * 4

/-- Converts weight from ounces to pounds -/
def ouncesToPounds (oz : ℕ) : ℕ := oz / 16

theorem heaviest_tv_weight_difference (bill_tv bob_tv steve_tv : TVDimensions) 
    (h1 : bill_tv = ⟨48, 100⟩)
    (h2 : bob_tv = ⟨70, 60⟩)
    (h3 : steve_tv = ⟨84, 92⟩) :
  ouncesToPounds (tvWeight (screenArea steve_tv)) - 
  (ouncesToPounds (tvWeight (screenArea bill_tv)) + ouncesToPounds (tvWeight (screenArea bob_tv))) = 318 := by
  sorry


end NUMINAMATH_CALUDE_heaviest_tv_weight_difference_l1603_160351


namespace NUMINAMATH_CALUDE_inequality_solution_l1603_160347

theorem inequality_solution (x : ℝ) (h : x ≠ 4) : (x^2 + 4) / ((x - 4)^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l1603_160347


namespace NUMINAMATH_CALUDE_triangle_sides_l1603_160352

theorem triangle_sides (a b c : ℚ) : 
  a + b + c = 24 →
  a + 2*b = 2*c →
  a = (1/2) * b →
  a = 16/3 ∧ b = 32/3 ∧ c = 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_sides_l1603_160352


namespace NUMINAMATH_CALUDE_swimming_speed_in_still_water_l1603_160305

/-- The swimming speed of a person in still water, given their performance against a current. -/
theorem swimming_speed_in_still_water 
  (current_speed : ℝ) 
  (distance : ℝ) 
  (time : ℝ) 
  (h1 : current_speed = 4) 
  (h2 : distance = 12) 
  (h3 : time = 2) : 
  ∃ (speed : ℝ), speed - current_speed = distance / time ∧ speed = 10 :=
sorry

end NUMINAMATH_CALUDE_swimming_speed_in_still_water_l1603_160305


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_values_l1603_160358

theorem perpendicular_lines_k_values (k : ℝ) : 
  (∀ x y : ℝ, (k - 1) * x + (2 * k + 3) * y - 2 = 0 ∧ 
               k * x + (1 - k) * y - 3 = 0 → 
               ((k - 1) * k + (2 * k + 3) * (1 - k) = 0)) → 
  k = 1 ∨ k = -3 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_values_l1603_160358


namespace NUMINAMATH_CALUDE_exponential_function_difference_l1603_160338

theorem exponential_function_difference (x : ℝ) : 
  let f : ℝ → ℝ := fun x => (3 : ℝ) ^ x
  f (x + 1) - f x = 2 * f x := by
sorry

end NUMINAMATH_CALUDE_exponential_function_difference_l1603_160338


namespace NUMINAMATH_CALUDE_security_compromise_l1603_160359

/-- Represents the security level of a system -/
inductive SecurityLevel
  | High
  | Medium
  | Low

/-- Represents a file type -/
inductive FileType
  | Secure
  | Suspicious

/-- Represents a website -/
structure Website where
  trusted : Bool

/-- Represents a user action -/
inductive UserAction
  | ShareInfo
  | DownloadFile (fileType : FileType)

/-- Represents the state of a system after a user action -/
structure SystemState where
  securityLevel : SecurityLevel

/-- Defines how a user action affects the system state -/
def updateSystemState (website : Website) (action : UserAction) (initialState : SystemState) : SystemState :=
  match website.trusted, action with
  | true, _ => initialState
  | false, UserAction.ShareInfo => ⟨SecurityLevel.Low⟩
  | false, UserAction.DownloadFile FileType.Suspicious => ⟨SecurityLevel.Low⟩
  | false, UserAction.DownloadFile FileType.Secure => initialState

theorem security_compromise (website : Website) (action : UserAction) (initialState : SystemState) :
  ¬website.trusted →
  (action = UserAction.ShareInfo ∨ (∃ (ft : FileType), action = UserAction.DownloadFile ft ∧ ft = FileType.Suspicious)) →
  (updateSystemState website action initialState).securityLevel = SecurityLevel.Low :=
by sorry


end NUMINAMATH_CALUDE_security_compromise_l1603_160359


namespace NUMINAMATH_CALUDE_divisibility_after_subtraction_l1603_160344

/-- The original polynomial before subtraction -/
def original_poly (x : ℝ) : ℝ := x^3 + 4*x^2 - 7*x + 12*x^3 + 4*x^2 - 7*x + 12

/-- The number to be subtracted -/
def subtrahend : ℝ := 42

/-- The resulting polynomial after subtraction -/
def result_poly (x : ℝ) : ℝ := 13*x^3 + 8*x^2 - 14*x - 30

theorem divisibility_after_subtraction :
  ∃ (p : ℝ → ℝ), ∀ (x : ℝ), 
    result_poly x = original_poly x - subtrahend ∧
    ∃ (q : ℝ → ℝ), result_poly x = p x * q x :=
sorry

end NUMINAMATH_CALUDE_divisibility_after_subtraction_l1603_160344


namespace NUMINAMATH_CALUDE_perimeter_of_eight_squares_l1603_160314

theorem perimeter_of_eight_squares (total_area : ℝ) (num_squares : ℕ) :
  total_area = 512 →
  num_squares = 8 →
  let square_area := total_area / num_squares
  let side_length := Real.sqrt square_area
  let perimeter := (2 * num_squares - 2) * side_length + 2 * side_length
  perimeter = 112 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_eight_squares_l1603_160314


namespace NUMINAMATH_CALUDE_points_cover_rectangles_l1603_160380

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A rectangle in a 2D plane -/
structure Rectangle where
  left : ℝ
  bottom : ℝ
  width : ℝ
  height : ℝ

/-- The unit square -/
def unitSquare : Rectangle := { left := 0, bottom := 0, width := 1, height := 1 }

/-- Check if a point is inside a rectangle -/
def isInside (p : Point) (r : Rectangle) : Prop :=
  r.left ≤ p.x ∧ p.x ≤ r.left + r.width ∧
  r.bottom ≤ p.y ∧ p.y ≤ r.bottom + r.height

/-- Check if a rectangle is inside another rectangle -/
def isContained (inner outer : Rectangle) : Prop :=
  outer.left ≤ inner.left ∧ inner.left + inner.width ≤ outer.left + outer.width ∧
  outer.bottom ≤ inner.bottom ∧ inner.bottom + inner.height ≤ outer.bottom + outer.height

/-- The main theorem -/
theorem points_cover_rectangles : ∃ (points : Finset Point),
  points.card ≤ 1600 ∧
  ∀ (r : Rectangle),
    isContained r unitSquare →
    r.width * r.height = 0.005 →
    ∃ (p : Point), p ∈ points ∧ isInside p r := by
  sorry

end NUMINAMATH_CALUDE_points_cover_rectangles_l1603_160380


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_6_real_l1603_160354

theorem sqrt_2x_minus_6_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = 2 * x - 6) ↔ x ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_6_real_l1603_160354


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1603_160386

theorem min_value_sum_squares (a b : ℝ) : 
  a > 0 → b > 0 → a ≠ b → a^2 - 2015*a = b^2 - 2015*b → 
  ∀ x y : ℝ, x > 0 → y > 0 → x ≠ y → x^2 - 2015*x = y^2 - 2015*y → 
  a^2 + b^2 ≤ x^2 + y^2 ∧ a^2 + b^2 = 2015^2 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1603_160386


namespace NUMINAMATH_CALUDE_min_games_correct_l1603_160313

/-- Represents a chess tournament between two schools -/
structure ChessTournament where
  white_rook_students : ℕ
  black_elephant_students : ℕ
  games_per_white_student : ℕ
  total_games : ℕ

/-- The specific tournament described in the problem -/
def tournament : ChessTournament :=
  { white_rook_students := 15
  , black_elephant_students := 20
  , games_per_white_student := 20
  , total_games := 300 }

/-- The minimum number of games after which one can guarantee
    that at least one White Rook student has played all their games -/
def min_games_for_guarantee (t : ChessTournament) : ℕ :=
  (t.white_rook_students - 1) * t.games_per_white_student

theorem min_games_correct (t : ChessTournament) :
  min_games_for_guarantee t = (t.white_rook_students - 1) * t.games_per_white_student ∧
  min_games_for_guarantee t < t.total_games ∧
  ∀ n, n < min_games_for_guarantee t → 
    ∃ i j, i < t.white_rook_students ∧ j < t.games_per_white_student ∧
           n < i * t.games_per_white_student + j :=
by sorry

#eval min_games_for_guarantee tournament  -- Should output 280

end NUMINAMATH_CALUDE_min_games_correct_l1603_160313


namespace NUMINAMATH_CALUDE_rectangular_prism_volume_l1603_160381

theorem rectangular_prism_volume 
  (l w h : ℝ) 
  (face_area_1 : l * w = 15) 
  (face_area_2 : w * h = 20) 
  (face_area_3 : l * h = 30) : 
  l * w * h = 60 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_volume_l1603_160381


namespace NUMINAMATH_CALUDE_function_inequality_l1603_160311

open Set

theorem function_inequality (f g : ℝ → ℝ) (a b x : ℝ) :
  DifferentiableOn ℝ f (Icc a b) →
  DifferentiableOn ℝ g (Icc a b) →
  (∀ y ∈ Icc a b, deriv f y > deriv g y) →
  a < x →
  x < b →
  f x + g a > g x + f a :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l1603_160311


namespace NUMINAMATH_CALUDE_novel_to_history_ratio_l1603_160397

theorem novel_to_history_ratio :
  let science_pages : ℕ := 600
  let history_pages : ℕ := 300
  let novel_pages : ℕ := science_pages / 4
  novel_pages.gcd history_pages = novel_pages →
  (novel_pages / novel_pages.gcd history_pages) = 1 ∧
  (history_pages / novel_pages.gcd history_pages) = 2 :=
by sorry

end NUMINAMATH_CALUDE_novel_to_history_ratio_l1603_160397


namespace NUMINAMATH_CALUDE_function_value_problem_l1603_160342

theorem function_value_problem (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x, f (x/2 - 1) = 2*x + 3) 
  (h2 : f m = 6) : 
  m = -1/4 := by sorry

end NUMINAMATH_CALUDE_function_value_problem_l1603_160342


namespace NUMINAMATH_CALUDE_square_diff_equality_l1603_160315

theorem square_diff_equality : 1003^2 - 997^2 - 1001^2 + 999^2 = 8000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_equality_l1603_160315


namespace NUMINAMATH_CALUDE_impurity_reduction_proof_l1603_160353

/-- The logarithm base 10 of 2 -/
def lg2 : Real := 0.3010

/-- The logarithm base 10 of 3 -/
def lg3 : Real := 0.4771

/-- The reduction factor of impurities after each filtration -/
def reduction_factor : Real := 0.8

/-- The target impurity level as a fraction of the original -/
def target_impurity : Real := 0.05

/-- The minimum number of filtrations required to reduce impurities below the target level -/
def min_filtrations : Nat := 15

theorem impurity_reduction_proof :
  (reduction_factor ^ min_filtrations : Real) < target_impurity ∧
  ∀ n : Nat, n < min_filtrations → (reduction_factor ^ n : Real) ≥ target_impurity :=
by
  sorry

#check impurity_reduction_proof

end NUMINAMATH_CALUDE_impurity_reduction_proof_l1603_160353


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l1603_160328

theorem modulo_eleven_residue : (308 + 6 * 44 + 8 * 165 + 3 * 18) % 11 = 10 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l1603_160328


namespace NUMINAMATH_CALUDE_difference_of_squares_l1603_160355

theorem difference_of_squares (x : ℝ) : x^2 - 9 = (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1603_160355


namespace NUMINAMATH_CALUDE_fraction_power_product_l1603_160324

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end NUMINAMATH_CALUDE_fraction_power_product_l1603_160324


namespace NUMINAMATH_CALUDE_software_contract_probability_l1603_160346

/-- Given probabilities for a computer company's contract scenarios, 
    prove the probability of not getting the software contract. -/
theorem software_contract_probability 
  (p_hardware : ℝ) 
  (p_at_least_one : ℝ) 
  (p_both : ℝ) 
  (h1 : p_hardware = 4/5)
  (h2 : p_at_least_one = 9/10)
  (h3 : p_both = 3/10) :
  1 - (p_at_least_one - p_hardware + p_both) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_software_contract_probability_l1603_160346


namespace NUMINAMATH_CALUDE_rhombus_count_in_divided_triangle_l1603_160393

/-- Represents an equilateral triangle --/
structure EquilateralTriangle where
  sideLength : ℝ
  sideLength_pos : sideLength > 0

/-- Represents a rhombus made of smaller equilateral triangles --/
structure Rhombus where
  smallTriangles : ℕ

/-- Counts the number of rhombuses in an equilateral triangle --/
def countRhombuses (triangle : EquilateralTriangle) (rhombus : Rhombus) : ℕ :=
  sorry

/-- Theorem statement --/
theorem rhombus_count_in_divided_triangle 
  (largeTriangle : EquilateralTriangle) 
  (smallTriangle : EquilateralTriangle) 
  (rhombus : Rhombus) :
  largeTriangle.sideLength = 10 →
  smallTriangle.sideLength = 1 →
  rhombus.smallTriangles = 8 →
  (largeTriangle.sideLength / smallTriangle.sideLength) ^ 2 = 100 →
  countRhombuses largeTriangle rhombus = 84 :=
sorry

end NUMINAMATH_CALUDE_rhombus_count_in_divided_triangle_l1603_160393


namespace NUMINAMATH_CALUDE_pirate_treasure_l1603_160343

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 := by
  sorry

end NUMINAMATH_CALUDE_pirate_treasure_l1603_160343


namespace NUMINAMATH_CALUDE_only_negative_three_halves_and_one_half_satisfy_l1603_160321

def numbers : List ℚ := [-3/2, -1, 1/2, 1, 3]

def satisfies_conditions (x : ℚ) : Prop :=
  x < x⁻¹ ∧ x > -3

theorem only_negative_three_halves_and_one_half_satisfy :
  ∀ x ∈ numbers, satisfies_conditions x ↔ (x = -3/2 ∨ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_only_negative_three_halves_and_one_half_satisfy_l1603_160321


namespace NUMINAMATH_CALUDE_direction_vector_k_l1603_160308

/-- The direction vector of the line passing through points A(0,2) and B(-1,0) is (1,k). -/
theorem direction_vector_k (k : ℝ) : 
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (-1, 0)
  let direction_vector : ℝ × ℝ := (1, k)
  (direction_vector.1 * (B.1 - A.1) = direction_vector.2 * (B.2 - A.2)) → k = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_direction_vector_k_l1603_160308


namespace NUMINAMATH_CALUDE_pb_cookie_probability_l1603_160378

/-- Represents the number of peanut butter cookies Jenny brought -/
def jenny_pb : ℕ := 40

/-- Represents the number of chocolate chip cookies Jenny brought -/
def jenny_cc : ℕ := 50

/-- Represents the number of peanut butter cookies Marcus brought -/
def marcus_pb : ℕ := 30

/-- Represents the number of lemon cookies Marcus brought -/
def marcus_lemon : ℕ := 20

/-- Represents the total number of cookies -/
def total_cookies : ℕ := jenny_pb + jenny_cc + marcus_pb + marcus_lemon

/-- Represents the total number of peanut butter cookies -/
def total_pb : ℕ := jenny_pb + marcus_pb

/-- Theorem stating that the probability of selecting a peanut butter cookie is 50% -/
theorem pb_cookie_probability : 
  (total_pb : ℚ) / total_cookies * 100 = 50 := by sorry

end NUMINAMATH_CALUDE_pb_cookie_probability_l1603_160378


namespace NUMINAMATH_CALUDE_amanda_drawer_pulls_l1603_160307

/-- Proves that the number of drawer pulls Amanda is replacing is 8 --/
theorem amanda_drawer_pulls (num_cabinet_knobs : ℕ) (cost_cabinet_knob : ℚ) 
  (cost_drawer_pull : ℚ) (total_cost : ℚ) 
  (h1 : num_cabinet_knobs = 18)
  (h2 : cost_cabinet_knob = 5/2)
  (h3 : cost_drawer_pull = 4)
  (h4 : total_cost = 77) :
  (total_cost - num_cabinet_knobs * cost_cabinet_knob) / cost_drawer_pull = 8 := by
  sorry

#eval (77 : ℚ) - 18 * (5/2 : ℚ)

end NUMINAMATH_CALUDE_amanda_drawer_pulls_l1603_160307


namespace NUMINAMATH_CALUDE_complex_square_simplification_l1603_160309

theorem complex_square_simplification : 
  let i : ℂ := Complex.I
  (4 - 3*i)^2 = 7 - 24*i :=
by sorry

end NUMINAMATH_CALUDE_complex_square_simplification_l1603_160309


namespace NUMINAMATH_CALUDE_lcm_gcd_product_9_10_l1603_160370

theorem lcm_gcd_product_9_10 : Nat.lcm 9 10 * Nat.gcd 9 10 = 90 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_9_10_l1603_160370


namespace NUMINAMATH_CALUDE_more_stable_lower_variance_l1603_160329

/-- Represents an athlete's assessment scores -/
structure AthleteScores where
  variance : ℝ
  assessmentCount : ℕ

/-- Defines the stability of an athlete's scores based on variance -/
def moreStable (a b : AthleteScores) : Prop :=
  a.variance < b.variance

/-- Theorem: Given two athletes with the same average score but different variances,
    the athlete with lower variance has more stable scores -/
theorem more_stable_lower_variance 
  (athleteA athleteB : AthleteScores)
  (hCount : athleteA.assessmentCount = athleteB.assessmentCount)
  (hCountPos : athleteA.assessmentCount > 0)
  (hVarA : athleteA.variance = 1.43)
  (hVarB : athleteB.variance = 0.82) :
  moreStable athleteB athleteA := by
  sorry

#check more_stable_lower_variance

end NUMINAMATH_CALUDE_more_stable_lower_variance_l1603_160329


namespace NUMINAMATH_CALUDE_world_population_scientific_notation_l1603_160334

/-- The number of people in the global population by the end of 2022 -/
def world_population : ℕ := 8000000000

/-- The scientific notation representation of the world population -/
def scientific_notation : ℝ := 8 * (10 : ℝ) ^ 9

/-- Theorem stating that the world population is equal to its scientific notation representation -/
theorem world_population_scientific_notation : 
  (world_population : ℝ) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_world_population_scientific_notation_l1603_160334


namespace NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l1603_160371

theorem inequality_holds_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, (x^2 - m*x + 1) / (x^2 + x + 1) ≤ 2) ↔ -4 ≤ m ∧ m ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_iff_m_in_range_l1603_160371


namespace NUMINAMATH_CALUDE_smallest_other_integer_l1603_160389

theorem smallest_other_integer (m n x : ℕ) : 
  m = 30 → 
  x > 0 → 
  Nat.gcd m n = x + 3 → 
  Nat.lcm m n = x * (x + 3) → 
  n ≥ 70 ∧ ∃ (n' : ℕ), n' = 70 ∧ 
    Nat.gcd m n' = x + 3 ∧ 
    Nat.lcm m n' = x * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_smallest_other_integer_l1603_160389


namespace NUMINAMATH_CALUDE_chocolate_count_l1603_160335

/-- Represents the total number of chocolates in the jar -/
def total_chocolates : ℕ := 50

/-- Represents the number of chocolates that are not hazelnut -/
def not_hazelnut : ℕ := 12

/-- Represents the number of chocolates that are not liquor -/
def not_liquor : ℕ := 18

/-- Represents the number of chocolates that are not milk -/
def not_milk : ℕ := 20

/-- Theorem stating that the total number of chocolates is 50 -/
theorem chocolate_count :
  total_chocolates = 50 ∧
  not_hazelnut = 12 ∧
  not_liquor = 18 ∧
  not_milk = 20 ∧
  (total_chocolates - not_hazelnut) + (total_chocolates - not_liquor) + (total_chocolates - not_milk) = 2 * total_chocolates :=
by sorry

end NUMINAMATH_CALUDE_chocolate_count_l1603_160335


namespace NUMINAMATH_CALUDE_inequality_contradiction_l1603_160387

theorem inequality_contradiction (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) 
  (h4 : a / b = c / d) : 
  ¬((a + b) / (a - b) = (c + d) / (c - d)) := by
sorry

end NUMINAMATH_CALUDE_inequality_contradiction_l1603_160387


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l1603_160374

theorem least_addition_for_divisibility (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(9 ∣ (51234 + m))) ∧ (9 ∣ (51234 + n)) → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l1603_160374


namespace NUMINAMATH_CALUDE_composite_shape_area_l1603_160310

-- Define the dimensions of the rectangles
def rect1_width : ℕ := 6
def rect1_height : ℕ := 7
def rect2_width : ℕ := 3
def rect2_height : ℕ := 5
def rect3_width : ℕ := 5
def rect3_height : ℕ := 6

-- Define the function to calculate the area of a rectangle
def rectangle_area (width height : ℕ) : ℕ := width * height

-- Theorem statement
theorem composite_shape_area :
  rectangle_area rect1_width rect1_height +
  rectangle_area rect2_width rect2_height +
  rectangle_area rect3_width rect3_height = 87 := by
  sorry


end NUMINAMATH_CALUDE_composite_shape_area_l1603_160310


namespace NUMINAMATH_CALUDE_speed_ratio_l1603_160341

def equidistant_points (vA vB : ℝ) : Prop :=
  ∃ (t : ℝ), t * vA = |(-800 + t * vB)|

theorem speed_ratio : ∃ (vA vB : ℝ),
  vA > 0 ∧ vB > 0 ∧
  equidistant_points vA vB ∧
  equidistant_points (3 * vA) (3 * vB) ∧
  vA / vB = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_speed_ratio_l1603_160341


namespace NUMINAMATH_CALUDE_bill_selling_price_l1603_160388

theorem bill_selling_price (purchase_price : ℝ) : 
  (purchase_price * 1.1 : ℝ) = 550 ∧ 
  (purchase_price * 0.9 * 1.3 : ℝ) - (purchase_price * 1.1 : ℝ) = 35 :=
by sorry

end NUMINAMATH_CALUDE_bill_selling_price_l1603_160388


namespace NUMINAMATH_CALUDE_octagon_pyramid_volume_l1603_160302

/-- A right pyramid with a regular octagon base and one equilateral triangular face --/
structure OctagonPyramid where
  /-- Side length of the equilateral triangular face --/
  side_length : ℝ
  /-- The base is a regular octagon --/
  is_regular_octagon : Bool
  /-- The pyramid is a right pyramid --/
  is_right_pyramid : Bool
  /-- One face is an equilateral triangle --/
  has_equilateral_face : Bool

/-- Calculate the volume of the octagon pyramid --/
noncomputable def volume (p : OctagonPyramid) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific octagon pyramid --/
theorem octagon_pyramid_volume :
  ∀ (p : OctagonPyramid),
    p.side_length = 10 ∧
    p.is_regular_octagon ∧
    p.is_right_pyramid ∧
    p.has_equilateral_face →
    volume p = 1000 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_octagon_pyramid_volume_l1603_160302


namespace NUMINAMATH_CALUDE_no_rational_solutions_for_quadratic_l1603_160366

theorem no_rational_solutions_for_quadratic (k : ℕ+) : 
  ¬∃ (x : ℚ), k * x^2 + 18 * x + 3 * k = 0 := by
sorry

end NUMINAMATH_CALUDE_no_rational_solutions_for_quadratic_l1603_160366


namespace NUMINAMATH_CALUDE_salary_sum_l1603_160316

theorem salary_sum (average_salary : ℕ) (num_people : ℕ) (known_salary : ℕ) :
  average_salary = 9000 →
  num_people = 5 →
  known_salary = 9000 →
  (num_people * average_salary) - known_salary = 36000 := by
  sorry

end NUMINAMATH_CALUDE_salary_sum_l1603_160316


namespace NUMINAMATH_CALUDE_triangle_altitude_area_theorem_l1603_160360

/-- Definition of a triangle with altitudes and area -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ
  area : ℝ

/-- Theorem stating the existence and non-existence of triangles with specific properties -/
theorem triangle_altitude_area_theorem :
  (∃ t : Triangle, t.ha < 1 ∧ t.hb < 1 ∧ t.hc < 1 ∧ t.area > 2) ∧
  (¬ ∃ t : Triangle, t.ha > 2 ∧ t.hb > 2 ∧ t.hc > 2 ∧ t.area < 1) :=
by sorry

end NUMINAMATH_CALUDE_triangle_altitude_area_theorem_l1603_160360


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l1603_160364

theorem quadratic_roots_difference (a b : ℝ) : 
  (∀ t : ℝ, t^2 - 12*t + 20 = 0 ↔ t = a ∨ t = b) → 
  a > b → 
  a - b = 8 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l1603_160364


namespace NUMINAMATH_CALUDE_teaching_arrangements_l1603_160322

-- Define the number of classes
def num_classes : ℕ := 4

-- Define the number of Chinese teachers
def num_chinese_teachers : ℕ := 2

-- Define the number of math teachers
def num_math_teachers : ℕ := 2

-- Define the number of classes each teacher teaches
def classes_per_teacher : ℕ := 2

-- Theorem statement
theorem teaching_arrangements :
  (Nat.choose num_classes classes_per_teacher) * (Nat.choose num_classes classes_per_teacher) = 36 := by
  sorry

end NUMINAMATH_CALUDE_teaching_arrangements_l1603_160322


namespace NUMINAMATH_CALUDE_f_at_three_l1603_160361

/-- Horner's method representation of the polynomial f(x) = x^5 - 2x^4 + 3x^3 - 4x^2 + 5x + 6 -/
def f (x : ℝ) : ℝ := (((x - 2) * x + 3) * x - 4) * x + 5 * x + 6

/-- Theorem stating that f(3) = 147 -/
theorem f_at_three : f 3 = 147 := by
  sorry

end NUMINAMATH_CALUDE_f_at_three_l1603_160361


namespace NUMINAMATH_CALUDE_approx_C_squared_minus_D_squared_for_specific_values_l1603_160372

/-- Given nonnegative real numbers x, y, z, we define C and D as follows:
C = √(x + 3) + √(y + 6) + √(z + 12)
D = √(x + 2) + √(y + 4) + √(z + 8)
This theorem states that when x = 1, y = 2, and z = 3, the value of C² - D² 
is approximately 19.483 with arbitrary precision. -/
theorem approx_C_squared_minus_D_squared_for_specific_values :
  ∀ ε > 0, ∃ C D : ℝ,
  C = Real.sqrt (1 + 3) + Real.sqrt (2 + 6) + Real.sqrt (3 + 12) ∧
  D = Real.sqrt (1 + 2) + Real.sqrt (2 + 4) + Real.sqrt (3 + 8) ∧
  |C^2 - D^2 - 19.483| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_approx_C_squared_minus_D_squared_for_specific_values_l1603_160372


namespace NUMINAMATH_CALUDE_magnitude_of_vector_sum_l1603_160340

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 1)

theorem magnitude_of_vector_sum : 
  ‖(2 • a.1 + b.1, 2 • a.2 + b.2)‖ = 5 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_magnitude_of_vector_sum_l1603_160340


namespace NUMINAMATH_CALUDE_pen_profit_percentage_retailer_profit_is_20_625_percent_l1603_160326

/-- Calculates the profit percentage for a retailer selling pens with a discount -/
theorem pen_profit_percentage 
  (num_pens_bought : ℕ) 
  (num_pens_price : ℕ) 
  (discount_percent : ℝ) : ℝ :=
  let cost_price := num_pens_price
  let selling_price_per_pen := 1 - (discount_percent / 100)
  let total_selling_price := selling_price_per_pen * num_pens_bought
  let profit := total_selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  20.625

/-- The retailer's profit percentage is 20.625% -/
theorem retailer_profit_is_20_625_percent : 
  pen_profit_percentage 75 60 3.5 = 20.625 := by
  sorry

end NUMINAMATH_CALUDE_pen_profit_percentage_retailer_profit_is_20_625_percent_l1603_160326


namespace NUMINAMATH_CALUDE_probability_at_least_three_white_is_550_715_l1603_160399

def white_balls : ℕ := 8
def black_balls : ℕ := 7
def total_balls : ℕ := white_balls + black_balls
def drawn_balls : ℕ := 6

def probability_at_least_three_white : ℚ :=
  (Nat.choose white_balls 3 * Nat.choose black_balls 3 +
   Nat.choose white_balls 4 * Nat.choose black_balls 2 +
   Nat.choose white_balls 5 * Nat.choose black_balls 1 +
   Nat.choose white_balls 6 * Nat.choose black_balls 0) /
  Nat.choose total_balls drawn_balls

theorem probability_at_least_three_white_is_550_715 :
  probability_at_least_three_white = 550 / 715 :=
by sorry

end NUMINAMATH_CALUDE_probability_at_least_three_white_is_550_715_l1603_160399


namespace NUMINAMATH_CALUDE_point_in_plane_region_l1603_160383

/-- The range of values for m such that point A(2, 3) lies within or on the boundary
    of the plane region represented by 3x - 2y + m ≥ 0 -/
theorem point_in_plane_region (m : ℝ) : 
  (3 * 2 - 2 * 3 + m ≥ 0) ↔ (m ≥ 0) := by sorry

end NUMINAMATH_CALUDE_point_in_plane_region_l1603_160383


namespace NUMINAMATH_CALUDE_acid_mixture_proof_l1603_160373

theorem acid_mixture_proof :
  let volume1 : ℝ := 4
  let concentration1 : ℝ := 0.60
  let volume2 : ℝ := 16
  let concentration2 : ℝ := 0.75
  let total_volume : ℝ := 20
  let final_concentration : ℝ := 0.72
  (volume1 * concentration1 + volume2 * concentration2) / total_volume = final_concentration ∧
  volume1 + volume2 = total_volume := by
sorry

end NUMINAMATH_CALUDE_acid_mixture_proof_l1603_160373
