import Mathlib

namespace movie_duration_l3531_353128

theorem movie_duration (tuesday_time : ℕ) (max_movies : ℕ) 
  (h1 : tuesday_time = 270)
  (h2 : max_movies = 9) : 
  ∃ (movie_length : ℕ), movie_length = 90 ∧ 
  ∃ (tuesday_movies : ℕ), 
    tuesday_movies * movie_length = tuesday_time ∧
    3 * tuesday_movies = max_movies :=
by
  sorry

end movie_duration_l3531_353128


namespace sequence_problem_l3531_353111

def arithmetic_sequence (a b c d : ℚ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def geometric_sequence (a b c d e : ℚ) : Prop :=
  b / a = c / b ∧ c / b = d / c ∧ d / c = e / d

theorem sequence_problem (a₁ a₂ b₁ b₂ b₃ : ℚ) 
  (h1 : arithmetic_sequence (-1) a₁ a₂ (-4))
  (h2 : geometric_sequence (-1) b₁ b₂ b₃ (-4)) :
  (a₂ - a₁) / b₂ = 1/2 := by sorry

end sequence_problem_l3531_353111


namespace exists_projective_map_three_points_l3531_353130

-- Define the necessary structures
structure ProjectivePlane where
  Point : Type
  Line : Type
  incidence : Point → Line → Prop

-- Define a projective map
def ProjectiveMap (π : ProjectivePlane) := π.Point → π.Point

-- State the theorem
theorem exists_projective_map_three_points 
  (π : ProjectivePlane) 
  (l₀ l : π.Line) 
  (A₀ B₀ C₀ A B C : π.Point)
  (on_l₀ : π.incidence A₀ l₀ ∧ π.incidence B₀ l₀ ∧ π.incidence C₀ l₀)
  (on_l : π.incidence A l ∧ π.incidence B l ∧ π.incidence C l) :
  ∃ (f : ProjectiveMap π), 
    f A₀ = A ∧ f B₀ = B ∧ f C₀ = C := by
  sorry

end exists_projective_map_three_points_l3531_353130


namespace pastries_cakes_difference_l3531_353185

theorem pastries_cakes_difference (pastries_sold : ℕ) (cakes_sold : ℕ) 
  (h1 : pastries_sold = 154) (h2 : cakes_sold = 78) : 
  pastries_sold - cakes_sold = 76 := by
  sorry

end pastries_cakes_difference_l3531_353185


namespace odd_power_sum_divisibility_l3531_353108

theorem odd_power_sum_divisibility (k : ℕ) (x y : ℤ) (h_odd : Odd k) (h_pos : k > 0) :
  (x^k + y^k) % (x + y) = 0 → (x^(k+2) + y^(k+2)) % (x + y) = 0 := by
  sorry

end odd_power_sum_divisibility_l3531_353108


namespace pizzas_per_person_is_30_l3531_353180

/-- The number of croissants each person eats -/
def croissants_per_person : ℕ := 7

/-- The number of cakes each person eats -/
def cakes_per_person : ℕ := 18

/-- The total number of items consumed by both people -/
def total_items : ℕ := 110

/-- The number of people -/
def num_people : ℕ := 2

theorem pizzas_per_person_is_30 :
  ∃ (pizzas_per_person : ℕ),
    pizzas_per_person = 30 ∧
    num_people * (croissants_per_person + cakes_per_person + pizzas_per_person) = total_items :=
by sorry

end pizzas_per_person_is_30_l3531_353180


namespace count_equal_S_consecutive_l3531_353123

def S (n : ℕ) : ℕ := (n % 4) + (n % 5) + (n % 6) + (n % 7) + (n % 8)

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem count_equal_S_consecutive : 
  ∃ (A B : ℕ), A ≠ B ∧ 
    is_three_digit A ∧ is_three_digit B ∧
    S A = S (A + 1) ∧ S B = S (B + 1) ∧
    ∀ (n : ℕ), is_three_digit n ∧ S n = S (n + 1) → n = A ∨ n = B :=
by sorry

end count_equal_S_consecutive_l3531_353123


namespace triangles_equality_l3531_353115

-- Define the points
variable (A K L M N G G' : ℝ × ℝ)

-- Define the angle α
variable (α : ℝ)

-- Define similarity of triangles
def similar_triangles (t1 t2 : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- Define isosceles triangle
def isosceles_triangle (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

-- Define angle at vertex
def angle_at_vertex (t : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (v : ℝ × ℝ) (θ : ℝ) : Prop := sorry

-- State the theorem
theorem triangles_equality (h1 : similar_triangles (A, K, L) (A, M, N))
                           (h2 : isosceles_triangle (A, K, L))
                           (h3 : isosceles_triangle (A, M, N))
                           (h4 : angle_at_vertex (A, K, L) A α)
                           (h5 : angle_at_vertex (A, M, N) A α)
                           (h6 : similar_triangles (G, N, K) (G', L, M))
                           (h7 : isosceles_triangle (G, N, K))
                           (h8 : isosceles_triangle (G', L, M))
                           (h9 : angle_at_vertex (G, N, K) G (π - α))
                           (h10 : angle_at_vertex (G', L, M) G' (π - α)) :
  G = G' := by sorry

end triangles_equality_l3531_353115


namespace max_consecutive_irreducible_five_digit_l3531_353112

/-- A number is irreducible if it cannot be expressed as a product of two three-digit numbers -/
def irreducible (n : ℕ) : Prop :=
  ∀ a b : ℕ, 100 ≤ a ∧ a ≤ 999 ∧ 100 ≤ b ∧ b ≤ 999 → n ≠ a * b

/-- The set of five-digit numbers -/
def five_digit_numbers : Set ℕ := {n | 10000 ≤ n ∧ n ≤ 99999}

/-- A function that returns the length of the longest sequence of consecutive irreducible numbers in a set -/
def max_consecutive_irreducible (s : Set ℕ) : ℕ :=
  sorry

/-- The theorem stating that the maximum number of consecutive irreducible five-digit numbers is 99 -/
theorem max_consecutive_irreducible_five_digit :
  max_consecutive_irreducible five_digit_numbers = 99 := by
  sorry

end max_consecutive_irreducible_five_digit_l3531_353112


namespace subset_intersection_implies_empty_complement_l3531_353170

theorem subset_intersection_implies_empty_complement
  (A B : Set ℝ) (h : A ⊆ A ∩ B) : A ∩ (Set.univ \ B) = ∅ := by
  sorry

end subset_intersection_implies_empty_complement_l3531_353170


namespace parabola_circle_tangency_l3531_353152

/-- A parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  focus : ℝ
  equation : ℝ → ℝ → Prop

/-- A circle with center (a, 0) and radius r -/
structure Circle where
  center : ℝ
  radius : ℝ
  equation : ℝ → ℝ → Prop

/-- The theorem statement -/
theorem parabola_circle_tangency 
  (C : Parabola) 
  (M : Circle)
  (h1 : C.focus > 0)
  (h2 : ∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ C.equation 1 y₁ ∧ C.equation 1 y₂)
  (h3 : ∀ (y₁ y₂ : ℝ), y₁ ≠ y₂ → C.equation 1 y₁ → C.equation 1 y₂ → y₁ * y₂ = -1)
  (h4 : M.center = 2)
  (h5 : M.radius = 1) :
  (C.equation = fun x y ↦ y^2 = x) ∧ 
  (M.equation = fun x y ↦ (x - 2)^2 + y^2 = 1) ∧ 
  (∀ (A₁ A₂ A₃ : ℝ × ℝ), 
    C.equation A₁.1 A₁.2 → 
    C.equation A₂.1 A₂.2 → 
    C.equation A₃.1 A₃.2 → 
    (∃ (k₁ k₂ : ℝ), 
      (∀ x y, y = k₁ * (x - A₁.1) + A₁.2 → 
        ((x - M.center)^2 + y^2 = M.radius^2 → x = M.center - M.radius ∨ x = M.center + M.radius)) ∧
      (∀ x y, y = k₂ * (x - A₁.1) + A₁.2 → 
        ((x - M.center)^2 + y^2 = M.radius^2 → x = M.center - M.radius ∨ x = M.center + M.radius))) →
    ∃ (k : ℝ), ∀ x y, y = k * (x - A₂.1) + A₂.2 → 
      ((x - M.center)^2 + y^2 = M.radius^2 → x = M.center - M.radius ∨ x = M.center + M.radius)) :=
by
  sorry

end parabola_circle_tangency_l3531_353152


namespace infinite_linear_combinations_l3531_353168

/-- An infinite sequence of strictly positive integers with a_k < a_{k+1} for all k -/
def StrictlyIncreasingSequence (a : ℕ → ℕ) : Prop :=
  ∀ k, 0 < a k ∧ a k < a (k + 1)

/-- The property that a_m can be written as x * a_p + y * a_q -/
def CanBeWrittenAs (a : ℕ → ℕ) (m p q x y : ℕ) : Prop :=
  a m = x * a p + y * a q ∧ 0 < x ∧ 0 < y ∧ p ≠ q

theorem infinite_linear_combinations (a : ℕ → ℕ) 
  (h : StrictlyIncreasingSequence a) :
  ∀ n : ℕ, ∃ m p q x y, m > n ∧ CanBeWrittenAs a m p q x y :=
sorry

end infinite_linear_combinations_l3531_353168


namespace quadratic_distinct_roots_condition_l3531_353199

/-- 
Given a quadratic equation kx^2 - 2x - 1 = 0, this theorem states that
for the equation to have two distinct real roots, k must be greater than -1
and not equal to 0.
-/
theorem quadratic_distinct_roots_condition (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 2*x - 1 = 0 ∧ k * y^2 - 2*y - 1 = 0) ↔ 
  (k > -1 ∧ k ≠ 0) :=
sorry

end quadratic_distinct_roots_condition_l3531_353199


namespace sinusoidal_function_properties_l3531_353135

/-- Given a sinusoidal function y = A*sin(ω*x + φ) with specific properties,
    prove its exact form and characteristics. -/
theorem sinusoidal_function_properties
  (A ω φ : ℝ)
  (h_A_pos : A > 0)
  (h_ω_pos : ω > 0)
  (h_passes_through : A * Real.sin (ω * (π / 12) + φ) = 0)
  (h_highest_point : A * Real.sin (ω * (π / 3) + φ) = 5) :
  ∃ (f : ℝ → ℝ),
    (∀ x, f x = 5 * Real.sin (2 * x - π / 6)) ∧
    (∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π + π / 3) (k * π + 5 * π / 6))) ∧
    (∀ x ∈ Set.Icc 0 π, f x ≤ 5 ∧ f x ≥ -5) ∧
    (f (π / 3) = 5 ∧ f (5 * π / 6) = -5) ∧
    (∀ k : ℤ, ∀ x, (k * π - 5 * π / 12 ≤ x ∧ x ≤ k * π + π / 12) → f x ≤ 0) :=
by sorry

end sinusoidal_function_properties_l3531_353135


namespace min_value_sqrt_and_reciprocal_equality_condition_l3531_353124

theorem min_value_sqrt_and_reciprocal (x : ℝ) (hx : x > 0) :
  3 * Real.sqrt x + 4 / x ≥ 4 * Real.sqrt 2 :=
sorry

theorem equality_condition (x : ℝ) (hx : x > 0) :
  ∃ x > 0, 3 * Real.sqrt x + 4 / x = 4 * Real.sqrt 2 :=
sorry

end min_value_sqrt_and_reciprocal_equality_condition_l3531_353124


namespace sufficient_not_necessary_condition_l3531_353183

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ x, (a < x ∧ x < a + 2) → x > 3) ∧
  (∃ x, x > 3 ∧ ¬(a < x ∧ x < a + 2)) →
  a ≥ 3 :=
sorry

end sufficient_not_necessary_condition_l3531_353183


namespace number_exceeding_percentage_l3531_353117

theorem number_exceeding_percentage (x : ℝ) : x = 0.2 * x + 40 → x = 50 := by
  sorry

end number_exceeding_percentage_l3531_353117


namespace train_length_l3531_353136

/-- The length of a train given its speed, time to cross a bridge, and bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 265 →
  train_speed * crossing_time - bridge_length = 110 := by
  sorry

end train_length_l3531_353136


namespace intersection_point_unique_l3531_353178

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (54/5, -26/5)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3*y = -2*x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 7*y = -3*x - 4

theorem intersection_point_unique :
  (∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2) ∧
  (line1 intersection_point.1 intersection_point.2) ∧
  (line2 intersection_point.1 intersection_point.2) :=
sorry

end intersection_point_unique_l3531_353178


namespace min_value_approx_l3531_353122

-- Define the function to be minimized
def f (a b c : ℕ) : ℚ :=
  (100 * a + 10 * b + c) / (a + b + c)

-- Define the conditions
def valid_digits (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ b > 3

-- Theorem statement
theorem min_value_approx (a b c : ℕ) (h : valid_digits a b c) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ f a b c ≥ 19.62 - ε :=
sorry

end min_value_approx_l3531_353122


namespace marble_ratio_l3531_353187

def marble_problem (pink : ℕ) (orange_diff : ℕ) (total : ℕ) : Prop :=
  let orange := pink - orange_diff
  let purple := total - pink - orange
  purple = 4 * orange

theorem marble_ratio :
  marble_problem 13 9 33 :=
by
  sorry

end marble_ratio_l3531_353187


namespace rectangle_folding_l3531_353163

/-- Rectangle ABCD with given side lengths -/
structure Rectangle :=
  (A B C D : ℝ × ℝ)
  (is_rectangle : sorry)
  (AD_length : dist A D = 4)
  (AB_length : dist A B = 3)

/-- Point B₁ after folding along diagonal AC -/
def B₁ (rect : Rectangle) : ℝ × ℝ := sorry

/-- Dihedral angle between two planes -/
def dihedral_angle (p₁ p₂ p₃ : ℝ × ℝ) (q₁ q₂ q₃ : ℝ × ℝ) : ℝ := sorry

/-- Distance between two skew lines -/
def skew_line_distance (p₁ p₂ q₁ q₂ : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem rectangle_folding (rect : Rectangle) :
  let b₁ := B₁ rect
  dihedral_angle b₁ rect.D rect.C rect.A rect.C rect.D = Real.arctan (15/16) ∧
  skew_line_distance rect.A b₁ rect.C rect.D = 10 * Real.sqrt 34 / 17 := by
  sorry

end rectangle_folding_l3531_353163


namespace no_solution_exists_l3531_353146

/-- S(x) represents the sum of the digits of the natural number x -/
def S (x : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are no natural numbers x satisfying the equation -/
theorem no_solution_exists : ¬ ∃ x : ℕ, x + S x + S (S x) = 2014 := by
  sorry

end no_solution_exists_l3531_353146


namespace ratio_A_to_B_in_X_l3531_353174

/-- Represents a compound with two elements -/
structure Compound where
  totalWeight : ℝ
  weightB : ℝ

/-- Calculates the ratio of element A to element B in a compound -/
def ratioAtoB (c : Compound) : ℝ × ℝ :=
  let weightA := c.totalWeight - c.weightB
  (weightA, c.weightB)

/-- Theorem: The ratio of A to B in compound X is 1:5 -/
theorem ratio_A_to_B_in_X :
  let compoundX : Compound := { totalWeight := 300, weightB := 250 }
  let (a, b) := ratioAtoB compoundX
  a / b = 1 / 5 := by sorry

end ratio_A_to_B_in_X_l3531_353174


namespace arrangement_count_correct_l3531_353186

def number_of_arrangements (men women : ℕ) : ℕ :=
  let first_group := men.choose 1 * women.choose 2
  let remaining_men := men - 1
  let remaining_women := women - 2
  let remaining_groups := remaining_men.choose 1 * remaining_women.choose 2
  first_group * remaining_groups

theorem arrangement_count_correct :
  number_of_arrangements 4 5 = 360 := by
  sorry

end arrangement_count_correct_l3531_353186


namespace min_value_f_when_a_1_range_of_a_for_inequality_l3531_353195

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x - a| + |x + a|

-- Theorem for the minimum value of f when a = 1
theorem min_value_f_when_a_1 :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x 1 ≥ f x_min 1 ∧ f x_min 1 = 3/2 :=
sorry

-- Theorem for the range of a
theorem range_of_a_for_inequality (a : ℝ) :
  (a > 0 ∧ ∃ (x : ℝ), x ∈ [1, 2] ∧ f x a < 5/x + a) ↔ 0 < a ∧ a < 6 :=
sorry

end min_value_f_when_a_1_range_of_a_for_inequality_l3531_353195


namespace car_interval_duration_l3531_353193

/-- Proves that the duration of each interval is 1/7.5 hours given the conditions of the car problem -/
theorem car_interval_duration 
  (initial_speed : ℝ) 
  (speed_decrease : ℝ) 
  (fifth_interval_distance : ℝ) 
  (h1 : initial_speed = 45)
  (h2 : speed_decrease = 3)
  (h3 : fifth_interval_distance = 4.4)
  : ∃ (t : ℝ), t = 1 / 7.5 ∧ fifth_interval_distance = (initial_speed - 4 * speed_decrease) * t :=
sorry

end car_interval_duration_l3531_353193


namespace function_passes_through_point_l3531_353103

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 3)
  f 3 = 1 := by sorry

end function_passes_through_point_l3531_353103


namespace f_satisfies_conditions_l3531_353140

noncomputable section

-- Define the function f
def f (x : ℝ) : ℝ := -Real.log x

-- State the theorem
theorem f_satisfies_conditions :
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ * x₂) = f x₁ + f x₂) :=
by sorry

end

end f_satisfies_conditions_l3531_353140


namespace quadratic_intersection_at_one_point_l3531_353154

theorem quadratic_intersection_at_one_point (b : ℝ) : 
  (∃! x : ℝ, b * x^2 + 5 * x + 3 = -2 * x - 2) ↔ b = 49 / 20 :=
by sorry

end quadratic_intersection_at_one_point_l3531_353154


namespace lg_sum_equals_one_l3531_353159

theorem lg_sum_equals_one (a b : ℝ) 
  (ha : a + Real.log a = 10) 
  (hb : b + (10 : ℝ)^b = 10) : 
  Real.log (a + b) = 1 := by
  sorry

end lg_sum_equals_one_l3531_353159


namespace triangle_similarity_theorem_l3531_353164

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the line segment MN
structure LineSegment :=
  (M N : ℝ × ℝ)

-- Define the parallel relation
def parallel (l1 l2 : LineSegment) : Prop := sorry

-- Define the length of a line segment
def length (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_similarity_theorem (XYZ : Triangle) (MN : LineSegment) :
  parallel MN (LineSegment.mk XYZ.X XYZ.Y) →
  length XYZ.X (MN.M) = 5 →
  length (MN.M) XYZ.Y = 8 →
  length (MN.N) XYZ.Z = 7 →
  length XYZ.X XYZ.Z = 18.2 := by
  sorry

end triangle_similarity_theorem_l3531_353164


namespace slurpee_purchase_l3531_353138

theorem slurpee_purchase (money_given : ℕ) (slurpee_cost : ℕ) (change : ℕ) : 
  money_given = 20 ∧ slurpee_cost = 2 ∧ change = 8 → 
  (money_given - change) / slurpee_cost = 6 := by
  sorry

end slurpee_purchase_l3531_353138


namespace simple_interest_principal_l3531_353149

/-- Simple interest calculation -/
theorem simple_interest_principal 
  (interest : ℝ) 
  (rate : ℝ) 
  (time : ℝ) 
  (h1 : interest = 4016.25)
  (h2 : rate = 9)
  (h3 : time = 5) : 
  ∃ (principal : ℝ), principal = 8925 ∧ interest = principal * rate * time / 100 :=
by sorry

end simple_interest_principal_l3531_353149


namespace ruth_math_class_hours_l3531_353104

/-- Represents Ruth's weekly school schedule and math class time --/
structure RuthSchedule where
  hours_per_day : ℝ
  days_per_week : ℝ
  math_class_percentage : ℝ

/-- Calculates the number of hours Ruth spends in math class per week --/
def math_class_hours (schedule : RuthSchedule) : ℝ :=
  schedule.hours_per_day * schedule.days_per_week * schedule.math_class_percentage

/-- Theorem stating that Ruth spends 10 hours per week in math class --/
theorem ruth_math_class_hours :
  let schedule := RuthSchedule.mk 8 5 0.25
  math_class_hours schedule = 10 := by
  sorry

end ruth_math_class_hours_l3531_353104


namespace gas_cost_equation_l3531_353118

theorem gas_cost_equation (x : ℚ) : x > 0 →
  (∃ (n m : ℕ), n = 4 ∧ m = 7 ∧ x / n - x / m = 10) ↔ x = 280 / 3 := by
  sorry

end gas_cost_equation_l3531_353118


namespace sqrt_inequality_l3531_353151

theorem sqrt_inequality (a : ℝ) (h : a > 1) : 
  Real.sqrt (a + 1) + Real.sqrt (a - 1) < 2 * Real.sqrt a := by
  sorry

end sqrt_inequality_l3531_353151


namespace larger_solution_quadratic_equation_l3531_353155

theorem larger_solution_quadratic_equation :
  ∃ (x y : ℝ), x ≠ y ∧ 
  x^2 - 13*x + 40 = 0 ∧ 
  y^2 - 13*y + 40 = 0 ∧ 
  (∀ z : ℝ, z^2 - 13*z + 40 = 0 → z = x ∨ z = y) ∧
  max x y = 8 := by
sorry

end larger_solution_quadratic_equation_l3531_353155


namespace complex_inequality_l3531_353157

theorem complex_inequality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) :
  Complex.abs (a - b) ≥ (1/2 : ℝ) * (Complex.abs a + Complex.abs b) * Complex.abs ((a / Complex.abs a) - (b / Complex.abs b)) ∧
  (Complex.abs (a - b) = (1/2 : ℝ) * (Complex.abs a + Complex.abs b) * Complex.abs ((a / Complex.abs a) - (b / Complex.abs b)) ↔ Complex.abs a = Complex.abs b) :=
by sorry

end complex_inequality_l3531_353157


namespace fourth_group_number_l3531_353188

/-- Represents a systematic sampling setup -/
structure SystematicSampling where
  total_students : Nat
  num_groups : Nat
  sample_size : Nat
  second_group_number : Nat

/-- The number drawn from a specific group in systematic sampling -/
def number_in_group (setup : SystematicSampling) (group : Nat) : Nat :=
  setup.second_group_number + (group - 2) * (setup.total_students / setup.num_groups)

/-- Theorem stating the relationship between the numbers drawn from different groups -/
theorem fourth_group_number (setup : SystematicSampling) 
  (h1 : setup.total_students = 72)
  (h2 : setup.num_groups = 6)
  (h3 : setup.sample_size = 6)
  (h4 : setup.second_group_number = 16) :
  number_in_group setup 4 = 40 := by
  sorry

end fourth_group_number_l3531_353188


namespace product_fixed_sum_squares_not_always_minimized_when_equal_l3531_353162

theorem product_fixed_sum_squares_not_always_minimized_when_equal :
  ¬ (∀ (k : ℝ), k > 0 →
    ∀ (x y : ℝ), x > 0 ∧ y > 0 ∧ x * y = k →
      ∀ (a b : ℝ), a > 0 ∧ b > 0 ∧ a * b = k →
        x^2 + y^2 ≤ a^2 + b^2 → x = y) :=
by sorry

end product_fixed_sum_squares_not_always_minimized_when_equal_l3531_353162


namespace fraction_product_one_l3531_353141

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem fraction_product_one : 
  ∃ (a b c d e f : ℕ), 
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1 ∧
    (a * c * e : ℚ) / (b * d * f : ℚ) = 1 := by
  sorry

end fraction_product_one_l3531_353141


namespace circle_radius_l3531_353172

theorem circle_radius (M N : ℝ) (h1 : M > 0) (h2 : N > 0) (h3 : M / N = 15) :
  ∃ (r : ℝ), r > 0 ∧ M = π * r^2 ∧ N = 2 * π * r ∧ r = 30 := by
  sorry

end circle_radius_l3531_353172


namespace number_satisfying_condition_l3531_353156

theorem number_satisfying_condition : ∃! x : ℝ, x / 3 + 12 = 20 ∧ x = 24 := by
  sorry

end number_satisfying_condition_l3531_353156


namespace min_product_of_reciprocal_sum_l3531_353192

theorem min_product_of_reciprocal_sum (a b : ℕ+) 
  (h : (a : ℚ)⁻¹ + (3 * b : ℚ)⁻¹ = (6 : ℚ)⁻¹) : 
  (∀ c d : ℕ+, (c : ℚ)⁻¹ + (3 * d : ℚ)⁻¹ = (6 : ℚ)⁻¹ → a * b ≤ c * d) ∧ a * b = 48 :=
sorry

end min_product_of_reciprocal_sum_l3531_353192


namespace james_vegetable_consumption_l3531_353181

/-- Calculates the final weekly vegetable consumption based on initial daily consumption and changes --/
def final_weekly_consumption (initial_daily : ℚ) (kale_addition : ℚ) : ℚ :=
  (initial_daily * 2 * 7) + kale_addition

/-- Proves that James' final weekly vegetable consumption is 10 pounds --/
theorem james_vegetable_consumption :
  let initial_daily := (1/4 : ℚ) + (1/4 : ℚ)
  let kale_addition := (3 : ℚ)
  final_weekly_consumption initial_daily kale_addition = 10 := by
  sorry

#eval final_weekly_consumption ((1/4 : ℚ) + (1/4 : ℚ)) 3

end james_vegetable_consumption_l3531_353181


namespace clock_angle_at_3_45_l3531_353167

/-- The smaller angle between the hour hand and minute hand on a 12-hour analog clock at 3:45 --/
theorem clock_angle_at_3_45 :
  let full_rotation : ℝ := 360
  let hour_marks : ℕ := 12
  let degrees_per_hour : ℝ := full_rotation / hour_marks
  let minute_hand_angle : ℝ := 270
  let hour_hand_angle : ℝ := 3 * degrees_per_hour + 3/4 * degrees_per_hour
  let angle_diff : ℝ := |minute_hand_angle - hour_hand_angle|
  min angle_diff (full_rotation - angle_diff) = 157.5 := by
  sorry

end clock_angle_at_3_45_l3531_353167


namespace multiply_binomials_l3531_353106

theorem multiply_binomials (a b : ℝ) : (3*a + 2*b) * (a - 2*b) = 3*a^2 - 4*a*b - 4*b^2 := by
  sorry

end multiply_binomials_l3531_353106


namespace range_of_a_l3531_353121

-- Define the proposition p
def p (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*x + a ≤ 0

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, p a ↔ a ≤ 1) := by sorry

end range_of_a_l3531_353121


namespace ratio_equality_l3531_353107

theorem ratio_equality (x y : ℝ) (h : 1.5 * x = 0.04 * y) :
  (y - x) / (y + x) = 73 / 77 := by
  sorry

end ratio_equality_l3531_353107


namespace cd_case_side_length_l3531_353143

/-- Given a square CD case with a circumference of 60 centimeters,
    prove that the length of one side is 15 centimeters. -/
theorem cd_case_side_length (circumference : ℝ) (side_length : ℝ) 
  (h1 : circumference = 60) 
  (h2 : circumference = 4 * side_length) : 
  side_length = 15 := by
  sorry

end cd_case_side_length_l3531_353143


namespace power_15000_mod_1000_l3531_353132

theorem power_15000_mod_1000 (h : 7^500 ≡ 1 [ZMOD 1000]) :
  7^15000 ≡ 1 [ZMOD 1000] := by
  sorry

end power_15000_mod_1000_l3531_353132


namespace second_race_outcome_l3531_353109

/-- Represents the speeds of Katie and Sarah -/
structure RunnerSpeeds where
  katie : ℝ
  sarah : ℝ

/-- The problem setup -/
def race_problem (speeds : RunnerSpeeds) : Prop :=
  speeds.katie > 0 ∧ 
  speeds.sarah > 0 ∧
  speeds.katie * 95 = speeds.sarah * 100

/-- The theorem to prove -/
theorem second_race_outcome (speeds : RunnerSpeeds) 
  (h : race_problem speeds) : 
  speeds.katie * 105 = speeds.sarah * 99.75 := by
  sorry

#check second_race_outcome

end second_race_outcome_l3531_353109


namespace triangle_inequality_l3531_353191

theorem triangle_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hab : a ≥ b) (hbc : b ≥ c) :
  Real.sqrt (a * (a + b - Real.sqrt (a * b))) +
  Real.sqrt (b * (a + c - Real.sqrt (a * c))) +
  Real.sqrt (c * (b + c - Real.sqrt (b * c))) ≥
  a + b + c := by
  sorry

end triangle_inequality_l3531_353191


namespace distance_between_red_lights_l3531_353101

/-- The distance between lights in inches -/
def light_spacing : ℕ := 8

/-- The number of lights in a complete color pattern cycle -/
def pattern_length : ℕ := 2 + 3 + 1

/-- The position of the nth red light in the sequence -/
def red_light_position (n : ℕ) : ℕ :=
  (n - 1) / 2 * pattern_length + (n - 1) % 2 + 1

/-- Convert inches to feet -/
def inches_to_feet (inches : ℕ) : ℚ :=
  inches / 12

theorem distance_between_red_lights :
  inches_to_feet (light_spacing * (red_light_position 15 - red_light_position 4)) = 19.3 := by
  sorry

end distance_between_red_lights_l3531_353101


namespace age_cube_sum_l3531_353148

theorem age_cube_sum (j a r : ℕ) (h1 : 2 * j + 3 * a = 4 * r) 
  (h2 : j^3 + a^3 = (1/2) * r^3) (h3 : j + a + r = 50) : 
  j^3 + a^3 + r^3 = 24680 := by
  sorry

end age_cube_sum_l3531_353148


namespace gina_college_cost_l3531_353161

/-- Calculates the total cost of Gina's college expenses -/
def total_college_cost (num_credits : ℕ) (cost_per_credit : ℕ) (num_textbooks : ℕ) (cost_per_textbook : ℕ) (facilities_fee : ℕ) : ℕ :=
  num_credits * cost_per_credit + num_textbooks * cost_per_textbook + facilities_fee

/-- Proves that Gina's total college expenses are $7100 -/
theorem gina_college_cost :
  total_college_cost 14 450 5 120 200 = 7100 := by
  sorry

end gina_college_cost_l3531_353161


namespace absolute_value_inequality_l3531_353190

theorem absolute_value_inequality (x : ℝ) : 
  x ≠ 2 → (|(3 * x - 2) / (x - 2)| > 3 ↔ (4/3 < x ∧ x < 2) ∨ x > 2) :=
by sorry

end absolute_value_inequality_l3531_353190


namespace decagon_triangle_probability_l3531_353184

/-- The number of vertices in a regular decagon -/
def decagon_vertices : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The total number of ways to choose 3 vertices from 10 vertices -/
def total_triangles : ℕ := Nat.choose decagon_vertices triangle_vertices

/-- The number of triangles with one side being a side of the decagon -/
def one_side_triangles : ℕ := decagon_vertices * 5

/-- The number of triangles with two sides being sides of the decagon -/
def two_side_triangles : ℕ := decagon_vertices

/-- The total number of favorable outcomes -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of selecting a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 1 / 2 := by
  sorry

end decagon_triangle_probability_l3531_353184


namespace sphere_volume_surface_area_relation_l3531_353110

theorem sphere_volume_surface_area_relation (r₁ r₂ : ℝ) (h : r₁ > 0) :
  (4 / 3 * Real.pi * r₂^3) = 8 * (4 / 3 * Real.pi * r₁^3) →
  (4 * Real.pi * r₂^2) = 4 * (4 * Real.pi * r₁^2) :=
by sorry

end sphere_volume_surface_area_relation_l3531_353110


namespace camp_acquaintances_l3531_353169

/-- Represents the number of acquaintances of a child -/
def Acquaintances : Type := ℕ

/-- Represents a child in the group -/
structure Child :=
  (name : String)
  (acquaintances : Acquaintances)

/-- The fraction of one child's acquaintances who are also acquainted with another child -/
def mutualAcquaintanceFraction (a b : Child) : ℚ := sorry

/-- Petya, one of the children in the group -/
def petya : Child := ⟨"Petya", sorry⟩

/-- Vasya, one of the children in the group -/
def vasya : Child := ⟨"Vasya", sorry⟩

/-- Timofey, one of the children in the group -/
def timofey : Child := ⟨"Timofey", sorry⟩

theorem camp_acquaintances :
  (mutualAcquaintanceFraction petya vasya = 1/2) →
  (mutualAcquaintanceFraction petya timofey = 1/7) →
  (mutualAcquaintanceFraction vasya petya = 1/3) →
  (mutualAcquaintanceFraction vasya timofey = 1/6) →
  (mutualAcquaintanceFraction timofey petya = 1/5) →
  (mutualAcquaintanceFraction timofey vasya = 7/20) :=
by sorry

end camp_acquaintances_l3531_353169


namespace max_xy_on_circle_l3531_353179

theorem max_xy_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) : 
  ∃ (max : ℝ), (∀ a b : ℝ, a^2 + b^2 = 4 → a * b ≤ max) ∧ (∃ c d : ℝ, c^2 + d^2 = 4 ∧ c * d = max) ∧ max = 2 := by
  sorry

end max_xy_on_circle_l3531_353179


namespace solve_equation_l3531_353142

theorem solve_equation (x y : ℤ) 
  (h1 : x^2 - 3*x + 6 = y + 2) 
  (h2 : x = -8) : 
  y = 92 := by
sorry

end solve_equation_l3531_353142


namespace quadratic_with_one_solution_l3531_353127

theorem quadratic_with_one_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 10 * x + c = 0) →
  a + c = 11 →
  a < c →
  (a = (11 - Real.sqrt 21) / 2 ∧ c = (11 + Real.sqrt 21) / 2) :=
by sorry

end quadratic_with_one_solution_l3531_353127


namespace x_24_value_l3531_353144

theorem x_24_value (x : ℝ) (h : x + 1/x = -Real.sqrt 3) : x^24 = 390625 := by
  sorry

end x_24_value_l3531_353144


namespace friend_contribution_is_eleven_l3531_353100

/-- The amount each friend should contribute when splitting the cost of movie tickets, popcorn, and milk tea. -/
def friend_contribution : ℚ :=
  let num_friends : ℕ := 3
  let ticket_price : ℚ := 7
  let num_tickets : ℕ := 3
  let popcorn_price : ℚ := 3/2  -- $1.5 as a rational number
  let num_popcorn : ℕ := 2
  let milk_tea_price : ℚ := 3
  let num_milk_tea : ℕ := 3
  let total_cost : ℚ := ticket_price * num_tickets + popcorn_price * num_popcorn + milk_tea_price * num_milk_tea
  total_cost / num_friends

theorem friend_contribution_is_eleven :
  friend_contribution = 11 := by
  sorry

end friend_contribution_is_eleven_l3531_353100


namespace agent_encryption_possible_l3531_353145

theorem agent_encryption_possible : ∃ (m n p q : ℕ), 
  (m > 0 ∧ n > 0 ∧ p > 0 ∧ q > 0) ∧ 
  (7 / 100 : ℚ) = 1 / m + 1 / n ∧
  (13 / 100 : ℚ) = 1 / p + 1 / q :=
sorry

end agent_encryption_possible_l3531_353145


namespace least_number_divisible_by_five_primes_l3531_353171

theorem least_number_divisible_by_five_primes :
  ∃ n : ℕ, n > 0 ∧
  (∃ p₁ p₂ p₃ p₄ p₅ : ℕ, 
    Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ Nat.Prime p₃ ∧ Nat.Prime p₄ ∧ Nat.Prime p₅ ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧
    p₄ ≠ p₅ ∧
    n % p₁ = 0 ∧ n % p₂ = 0 ∧ n % p₃ = 0 ∧ n % p₄ = 0 ∧ n % p₅ = 0) ∧
  (∀ m : ℕ, m > 0 ∧ m < n →
    ¬(∃ q₁ q₂ q₃ q₄ q₅ : ℕ,
      Nat.Prime q₁ ∧ Nat.Prime q₂ ∧ Nat.Prime q₃ ∧ Nat.Prime q₄ ∧ Nat.Prime q₅ ∧
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧
      q₄ ≠ q₅ ∧
      m % q₁ = 0 ∧ m % q₂ = 0 ∧ m % q₃ = 0 ∧ m % q₄ = 0 ∧ m % q₅ = 0)) :=
by sorry

end least_number_divisible_by_five_primes_l3531_353171


namespace third_term_of_x_plus_two_pow_five_l3531_353176

/-- The coefficient of the r-th term in the expansion of (a + b)^n -/
def binomial_coefficient (n : ℕ) (r : ℕ) : ℕ :=
  Nat.choose n r

/-- The r-th term in the expansion of (a + b)^n -/
def binomial_term (n : ℕ) (r : ℕ) (a b : ℚ) : ℚ :=
  (binomial_coefficient n r : ℚ) * a^(n - r) * b^r

/-- The third term of (x + 2)^5 is 40x^3 -/
theorem third_term_of_x_plus_two_pow_five (x : ℚ) :
  binomial_term 5 2 x 2 = 40 * x^3 := by
  sorry

end third_term_of_x_plus_two_pow_five_l3531_353176


namespace quadratic_point_ordering_l3531_353129

-- Define the quadratic function
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + c

-- Define the theorem
theorem quadratic_point_ordering (c : ℝ) (y₁ y₂ y₃ : ℝ) 
  (h1 : f c (-1) = y₁)
  (h2 : f c 2 = y₂)
  (h3 : f c (-3) = y₃) :
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end quadratic_point_ordering_l3531_353129


namespace altitude_inscribed_radius_relation_l3531_353102

-- Define a triangle type
structure Triangle where
  -- Three sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Ensure the triangle inequality holds
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the altitudes of the triangle
def altitude (t : Triangle) : ℝ × ℝ × ℝ := sorry

-- Define the inscribed circle radius
def inscribed_radius (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem altitude_inscribed_radius_relation (t : Triangle) :
  let (h₁, h₂, h₃) := altitude t
  let r := inscribed_radius t
  1 / h₁ + 1 / h₂ + 1 / h₃ = 1 / r := by sorry

end altitude_inscribed_radius_relation_l3531_353102


namespace inequality_solution_set_l3531_353139

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + Real.cos x

theorem inequality_solution_set (m : ℝ) :
  f (2 * m) > f (m - 2) ↔ m < -2 ∨ m > 2/3 := by sorry

end inequality_solution_set_l3531_353139


namespace confectioner_pastry_count_l3531_353175

theorem confectioner_pastry_count :
  ∀ (P : ℕ),
  (P / 28 : ℚ) - (P / 49 : ℚ) = 6 →
  P = 378 :=
by
  sorry

end confectioner_pastry_count_l3531_353175


namespace square_perimeter_l3531_353137

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 360 →
  area = side ^ 2 →
  perimeter = 4 * side →
  perimeter = 24 * Real.sqrt 10 := by
sorry

end square_perimeter_l3531_353137


namespace equivalent_representations_l3531_353189

theorem equivalent_representations (x y z w : ℚ) : 
  x = 1 / 8 ∧ 
  y = 2 / 16 ∧ 
  z = 3 / 24 ∧ 
  w = 125 / 1000 → 
  x = y ∧ y = z ∧ z = w := by
sorry

end equivalent_representations_l3531_353189


namespace spinner_prime_probability_l3531_353160

def spinner : List Nat := [2, 7, 9, 11, 15, 17]

def isPrime (n : Nat) : Bool :=
  n > 1 && (List.range (n - 1)).all (fun d => d <= 1 || n % (d + 2) ≠ 0)

def countPrimes (l : List Nat) : Nat :=
  (l.filter isPrime).length

theorem spinner_prime_probability :
  (countPrimes spinner : Rat) / (spinner.length : Rat) = 2/3 := by
  sorry

end spinner_prime_probability_l3531_353160


namespace candy_distribution_theorem_l3531_353119

/-- 
Represents the candy distribution function.
For a given number of students n and a position i,
it returns the number of candies given to the student at position i.
-/
def candy_distribution (n : ℕ) (i : ℕ) : ℕ :=
  sorry

/-- 
Checks if every student receives at least one candy
for a given number of students n.
-/
def every_student_gets_candy (n : ℕ) : Prop :=
  sorry

/-- 
Checks if a given natural number is a power of 2.
-/
def is_power_of_two (n : ℕ) : Prop :=
  sorry

/-- 
Theorem: For n ≥ 2, every student receives at least one candy
if and only if n is a power of 2.
-/
theorem candy_distribution_theorem (n : ℕ) (h : n ≥ 2) :
  every_student_gets_candy n ↔ is_power_of_two n :=
sorry

end candy_distribution_theorem_l3531_353119


namespace series_sum_equals_five_l3531_353150

theorem series_sum_equals_five (k : ℝ) (h1 : k > 1) 
  (h2 : ∑' n, (7 * n + 2) / k^n = 5) : k = (7 + Real.sqrt 14) / 5 := by
  sorry

end series_sum_equals_five_l3531_353150


namespace simplify_and_compare_l3531_353197

theorem simplify_and_compare : 
  1.82 * (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := by
  sorry

end simplify_and_compare_l3531_353197


namespace polynomial_decomposition_l3531_353153

/-- A polynomial with real coefficients -/
def RealPolynomial := ℝ → ℝ

/-- Predicate to check if a polynomial is nonnegative on [0,1] -/
def IsNonnegativeOn01 (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → P x ≥ 0

/-- Predicate to check if a polynomial is nonnegative on ℝ -/
def IsNonnegativeOnReals (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, P x ≥ 0

theorem polynomial_decomposition (P : RealPolynomial) (h : IsNonnegativeOn01 P) :
  ∃ (P₀ P₁ P₂ : RealPolynomial),
    (IsNonnegativeOnReals P₀) ∧
    (IsNonnegativeOnReals P₁) ∧
    (IsNonnegativeOnReals P₂) ∧
    (∀ x : ℝ, P x = P₀ x + x * P₁ x + (1 - x) * P₂ x) :=
  sorry

end polynomial_decomposition_l3531_353153


namespace problem_part1_l3531_353133

theorem problem_part1 : (-2)^2 + |Real.sqrt 2 - 1| - Real.sqrt 4 = Real.sqrt 2 + 1 := by
  sorry

end problem_part1_l3531_353133


namespace geometric_series_equality_l3531_353196

/-- Given real numbers a and b satisfying an infinite geometric series equation,
    prove that another related infinite geometric series equals 3/4. -/
theorem geometric_series_equality (a b : ℝ) 
  (h : (a / (2 * b)) / (1 - 1 / (2 * b)) = 6) :
  (a / (a + 2 * b)) / (1 - 1 / (a + 2 * b)) = 3/4 := by
  sorry

end geometric_series_equality_l3531_353196


namespace inequality_solution_set_l3531_353158

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | 56 * x^2 + a * x - a^2 < 0} = {x : ℝ | a / 8 < x ∧ x < -a / 7} :=
by sorry

end inequality_solution_set_l3531_353158


namespace charlie_has_31_pennies_l3531_353165

/-- The number of pennies Charlie has -/
def charlie_pennies : ℕ := 31

/-- The number of pennies Alex has -/
def alex_pennies : ℕ := 9

/-- Condition 1: If Alex gives Charlie a penny, Charlie will have four times as many pennies as Alex has -/
axiom condition1 : charlie_pennies + 1 = 4 * (alex_pennies - 1)

/-- Condition 2: If Charlie gives Alex a penny, Charlie will have three times as many pennies as Alex has -/
axiom condition2 : charlie_pennies - 1 = 3 * (alex_pennies + 1)

theorem charlie_has_31_pennies : charlie_pennies = 31 := by
  sorry

end charlie_has_31_pennies_l3531_353165


namespace borrowed_amount_l3531_353125

theorem borrowed_amount (P : ℝ) 
  (h1 : (P * 12 / 100 * 3) + (P * 9 / 100 * 5) + (P * 13 / 100 * 3) = 8160) : 
  P = 6800 := by
  sorry

end borrowed_amount_l3531_353125


namespace sum_of_fractions_l3531_353126

theorem sum_of_fractions : (1 : ℚ) / 3 + (5 : ℚ) / 12 = (3 : ℚ) / 4 := by
  sorry

end sum_of_fractions_l3531_353126


namespace clock_problem_l3531_353114

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Represents duration in hours, minutes, and seconds -/
structure Duration where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Converts total seconds to Time structure -/
def secondsToTime (totalSeconds : Nat) : Time :=
  let hours := totalSeconds / 3600
  let minutes := (totalSeconds % 3600) / 60
  let seconds := totalSeconds % 60
  { hours := hours % 12, minutes := minutes, seconds := seconds }

/-- Adds a Duration to a Time, wrapping around 12-hour clock -/
def addDuration (t : Time) (d : Duration) : Time :=
  let totalSeconds := 
    (t.hours * 3600 + t.minutes * 60 + t.seconds) +
    (d.hours * 3600 + d.minutes * 60 + d.seconds)
  secondsToTime totalSeconds

/-- Calculates the sum of digits in a Time -/
def sumDigits (t : Time) : Nat :=
  t.hours + t.minutes + t.seconds

theorem clock_problem (initialTime : Time) (elapsedTime : Duration) : 
  initialTime.hours = 3 ∧ 
  initialTime.minutes = 15 ∧ 
  initialTime.seconds = 20 ∧
  elapsedTime.hours = 305 ∧ 
  elapsedTime.minutes = 45 ∧ 
  elapsedTime.seconds = 56 →
  let finalTime := addDuration initialTime elapsedTime
  finalTime.hours = 9 ∧ 
  finalTime.minutes = 1 ∧ 
  finalTime.seconds = 16 ∧
  sumDigits finalTime = 26 := by
  sorry

end clock_problem_l3531_353114


namespace aquarium_species_count_l3531_353116

theorem aquarium_species_count 
  (sharks : ℕ) (eels : ℕ) (whales : ℕ) (dolphins : ℕ) (rays : ℕ) (octopuses : ℕ)
  (shark_pairs : ℕ) (eel_pairs : ℕ) (whale_pairs : ℕ) (octopus_split : ℕ)
  (h1 : sharks = 48)
  (h2 : eels = 21)
  (h3 : whales = 7)
  (h4 : dolphins = 16)
  (h5 : rays = 9)
  (h6 : octopuses = 30)
  (h7 : shark_pairs = 3)
  (h8 : eel_pairs = 2)
  (h9 : whale_pairs = 1)
  (h10 : octopus_split = 1) :
  sharks + eels + whales + dolphins + rays + octopuses 
  - (shark_pairs + eel_pairs + whale_pairs) 
  + octopus_split = 126 :=
by sorry

end aquarium_species_count_l3531_353116


namespace bela_wins_iff_m_odd_l3531_353147

/-- The game interval --/
def GameInterval (m : ℕ) := Set.Icc (0 : ℝ) m

/-- Predicate for a valid move --/
def ValidMove (m : ℕ) (prev_moves : List ℝ) (x : ℝ) : Prop :=
  x ∈ GameInterval m ∧ ∀ y ∈ prev_moves, |x - y| > 2

/-- The game result --/
inductive GameResult
  | BelaWins
  | JennWins

/-- The game outcome based on the optimal strategy --/
def GameOutcome (m : ℕ) : GameResult :=
  if m % 2 = 1 then GameResult.BelaWins else GameResult.JennWins

/-- The main theorem --/
theorem bela_wins_iff_m_odd (m : ℕ) (h : m > 2) :
  GameOutcome m = GameResult.BelaWins ↔ m % 2 = 1 := by sorry

end bela_wins_iff_m_odd_l3531_353147


namespace greatest_integer_with_gcf_five_gcf_of_145_and_30_is_145_greatest_l3531_353120

theorem greatest_integer_with_gcf_five (n : ℕ) : n < 150 ∧ Nat.gcd n 30 = 5 → n ≤ 145 := by
  sorry

theorem gcf_of_145_and_30 : Nat.gcd 145 30 = 5 := by
  sorry

theorem is_145_greatest : ∀ m : ℕ, m < 150 ∧ Nat.gcd m 30 = 5 → m ≤ 145 := by
  sorry

end greatest_integer_with_gcf_five_gcf_of_145_and_30_is_145_greatest_l3531_353120


namespace total_handshakes_l3531_353105

/-- Represents the number of people in the meeting -/
def total_people : ℕ := 40

/-- Represents the number of people who mostly know each other -/
def group1_size : ℕ := 25

/-- Represents the number of strangers within group1 -/
def strangers_in_group1 : ℕ := 5

/-- Represents the number of people who know no one -/
def group2_size : ℕ := 15

/-- Calculates the number of handshakes between strangers in group1 -/
def handshakes_in_group1 : ℕ := strangers_in_group1 * (strangers_in_group1 - 1) / 2

/-- Calculates the number of handshakes involving group2 -/
def handshakes_involving_group2 : ℕ := group2_size * (total_people - 1)

/-- The main theorem stating the total number of handshakes -/
theorem total_handshakes : 
  handshakes_in_group1 + handshakes_involving_group2 = 595 := by sorry

end total_handshakes_l3531_353105


namespace inequality_solution_l3531_353182

theorem inequality_solution (x : ℝ) : 
  let x₁ : ℝ := (-9 - Real.sqrt 21) / 2
  let x₂ : ℝ := (-9 + Real.sqrt 21) / 2
  (x - 1) / (x + 3) > (4 * x + 5) / (3 * x + 8) ↔ 
    (x > -3 ∧ x < x₁) ∨ (x > x₂) :=
by sorry

end inequality_solution_l3531_353182


namespace intersection_distance_l3531_353177

-- Define the two curves
def curve1 (x y : ℝ) : Prop := x = y^3
def curve2 (x y : ℝ) : Prop := x + y^3 = 2

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ curve1 x y ∧ curve2 x y}

-- State the theorem
theorem intersection_distance :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points ∧ p2 ∈ intersection_points ∧
  p1 ≠ p2 ∧ Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2) = 2 * Real.sqrt 2 :=
sorry

end intersection_distance_l3531_353177


namespace common_divisors_9240_13860_l3531_353198

/-- The number of positive divisors that two natural numbers have in common -/
def common_divisors_count (a b : ℕ) : ℕ := (Nat.divisors (Nat.gcd a b)).card

/-- Theorem stating that 9240 and 13860 have 48 positive divisors in common -/
theorem common_divisors_9240_13860 :
  common_divisors_count 9240 13860 = 48 := by sorry

end common_divisors_9240_13860_l3531_353198


namespace batsman_average_after_17th_inning_l3531_353166

/-- Represents a batsman's performance -/
structure Batsman where
  initialAverage : ℝ  -- Average before the 17th inning
  runsScored : ℕ      -- Runs scored in the 17th inning
  averageIncrease : ℝ -- Increase in average after the 17th inning

/-- Calculates the new average after the 17th inning -/
def newAverage (b : Batsman) : ℝ :=
  b.initialAverage + b.averageIncrease

/-- Theorem: The batsman's average after the 17th inning is 140 runs -/
theorem batsman_average_after_17th_inning (b : Batsman)
  (h1 : b.runsScored = 300)
  (h2 : b.averageIncrease = 10)
  : newAverage b = 140 := by
  sorry

#check batsman_average_after_17th_inning

end batsman_average_after_17th_inning_l3531_353166


namespace arithmetic_sequence_sum_10_l3531_353194

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

/-- Sum of first n terms of an arithmetic sequence -/
def arithmeticSum (a₁ d : ℤ) (n : ℕ) : ℤ := n * a₁ + n * (n - 1) / 2 * d

theorem arithmetic_sequence_sum_10 (a₁ a₂ a₆ : ℤ) (d : ℤ) :
  a₁ = -2 →
  a₂ + a₆ = 2 →
  (∀ n : ℕ, arithmeticSequence a₁ d n = a₁ + (n - 1) * d) →
  arithmeticSum a₁ d 10 = 25 :=
by sorry

end arithmetic_sequence_sum_10_l3531_353194


namespace fraction_of_male_fish_l3531_353173

theorem fraction_of_male_fish (total : ℕ) (female : ℕ) (h1 : total = 45) (h2 : female = 15) :
  (total - female : ℚ) / total = 2 / 3 := by
  sorry

end fraction_of_male_fish_l3531_353173


namespace initial_potatoes_l3531_353134

theorem initial_potatoes (initial_tomatoes picked_potatoes remaining_total : ℕ) : 
  initial_tomatoes = 175 →
  picked_potatoes = 172 →
  remaining_total = 80 →
  initial_tomatoes + (initial_tomatoes + picked_potatoes - remaining_total) = 175 + 77 :=
by sorry

end initial_potatoes_l3531_353134


namespace knight_freedom_guaranteed_l3531_353131

/-- Represents a pile of coins -/
structure Pile :=
  (total : ℕ)
  (magical : ℕ)

/-- Represents the state of the coins -/
structure CoinState :=
  (pile1 : Pile)
  (pile2 : Pile)

/-- Checks if the piles have equal magical or ordinary coins -/
def isEqualDistribution (state : CoinState) : Prop :=
  state.pile1.magical = state.pile2.magical ∨ 
  (state.pile1.total - state.pile1.magical) = (state.pile2.total - state.pile2.magical)

/-- Represents a division strategy -/
def DivisionStrategy := ℕ → CoinState

/-- The theorem to be proved -/
theorem knight_freedom_guaranteed :
  ∃ (strategy : DivisionStrategy),
    (∀ (n : ℕ), n ≤ 25 → 
      (strategy n).pile1.total + (strategy n).pile2.total = 100 ∧
      (strategy n).pile1.magical + (strategy n).pile2.magical = 50) →
    ∃ (day : ℕ), day ≤ 25 ∧ isEqualDistribution (strategy day) :=
sorry

end knight_freedom_guaranteed_l3531_353131


namespace smallest_solution_l3531_353113

-- Define the equation
def equation (t : ℝ) : Prop :=
  (16 * t^3 - 49 * t^2 + 35 * t - 6) / (4 * t - 3) + 7 * t = 8 * t - 2

-- Define the set of all t that satisfy the equation
def solution_set : Set ℝ := {t | equation t}

-- Theorem statement
theorem smallest_solution :
  ∃ (t_min : ℝ), t_min ∈ solution_set ∧ t_min = 3/4 ∧ ∀ (t : ℝ), t ∈ solution_set → t_min ≤ t :=
sorry

end smallest_solution_l3531_353113
