import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l526_52688

noncomputable section

-- Define the function f
def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

-- Define the function g
def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2*m*(f a 2 x)

theorem problem_solution (a : ℝ) (h_a : a > 0 ∧ a ≠ 1) :
  -- Part 1: k = 2 if f is an odd function
  (∀ x, f a 2 x = -(f a 2 (-x))) →
  -- Part 2: If f(1) < 0, then f is decreasing and the inequality holds iff -3 < t < 5
  (f a 2 1 < 0 →
    (∀ x y, x < y → f a 2 x > f a 2 y) ∧
    (∀ t, (∀ x, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0) ↔ -3 < t ∧ t < 5)) ∧
  -- Part 3: If f(1) = 3/2 and g has min value -2 on [1,+∞), then m = 2
  (f a 2 1 = 3/2 →
    (∃ m, (∀ x ≥ 1, g a m x ≥ -2) ∧
          (∃ x ≥ 1, g a m x = -2)) →
    m = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l526_52688


namespace NUMINAMATH_CALUDE_river_width_is_500_l526_52603

/-- Represents the river crossing scenario -/
structure RiverCrossing where
  velocity : ℝ  -- Boatman's velocity in m/sec
  time : ℝ      -- Time taken to cross the river in seconds
  drift : ℝ     -- Drift distance in meters

/-- Calculates the width of the river given the crossing parameters -/
def riverWidth (rc : RiverCrossing) : ℝ :=
  rc.velocity * rc.time

/-- Theorem stating that the width of the river is 500 meters 
    given the specific conditions -/
theorem river_width_is_500 (rc : RiverCrossing) 
  (h1 : rc.velocity = 10)
  (h2 : rc.time = 50)
  (h3 : rc.drift = 300) : 
  riverWidth rc = 500 := by
  sorry

#check river_width_is_500

end NUMINAMATH_CALUDE_river_width_is_500_l526_52603


namespace NUMINAMATH_CALUDE_range_of_fraction_l526_52643

-- Define the quadratic equation
def quadratic (a b x : ℝ) : Prop := x^2 + a*x + 2*b - 2 = 0

-- Define the theorem
theorem range_of_fraction (a b : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    quadratic a b x₁ ∧ quadratic a b x₂ ∧
    0 < x₁ ∧ x₁ < 1 ∧ 
    1 < x₂ ∧ x₂ < 2) →
  1/2 < (b - 4) / (a - 1) ∧ (b - 4) / (a - 1) < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_fraction_l526_52643


namespace NUMINAMATH_CALUDE_triangle_equality_l526_52606

-- Define a triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  h : A + B + C = 180  -- Sum of angles in a triangle is 180°

-- Define the theorem
theorem triangle_equality (t : Triangle) 
  (h₁ : t.A > t.B)  -- A > B
  (h₂ : ∃ (C₁ C₂ : ℝ), C₁ + C₂ = t.C ∧ C₁ = 2 * C₂)  -- C₁ + C₂ = C and C₁ = 2C₂
  : t.A = t.B := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l526_52606


namespace NUMINAMATH_CALUDE_tangent_trapezoid_ratio_l526_52663

/-- Represents a trapezoid with a circle tangent to two sides -/
structure TangentTrapezoid where
  /-- Length of side EF -/
  ef : ℝ
  /-- Length of side FG -/
  fg : ℝ
  /-- Length of side GH -/
  gh : ℝ
  /-- Length of side HE -/
  he : ℝ
  /-- EF is parallel to GH -/
  parallel : ef ≠ gh
  /-- Circle with center Q on EF is tangent to FG and HE -/
  tangent : True

/-- The ratio of EQ to QF in the trapezoid -/
def ratio (t : TangentTrapezoid) : ℚ :=
  12 / 37

theorem tangent_trapezoid_ratio (t : TangentTrapezoid) 
  (h1 : t.ef = 40)
  (h2 : t.fg = 25)
  (h3 : t.gh = 12)
  (h4 : t.he = 35) :
  ratio t = 12 / 37 := by
  sorry

end NUMINAMATH_CALUDE_tangent_trapezoid_ratio_l526_52663


namespace NUMINAMATH_CALUDE_product_of_numbers_l526_52686

theorem product_of_numbers (x y : ℝ) : 
  x - y = 7 → x^2 + y^2 = 85 → x * y = 18 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l526_52686


namespace NUMINAMATH_CALUDE_complex_modulus_range_l526_52657

theorem complex_modulus_range (z : ℂ) (a : ℝ) :
  z = 3 + a * Complex.I ∧ Complex.abs z < 4 →
  a ∈ Set.Ioo (-Real.sqrt 7) (Real.sqrt 7) := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_range_l526_52657


namespace NUMINAMATH_CALUDE_geometric_sequence_properties_l526_52659

theorem geometric_sequence_properties (a₁ q : ℝ) (h_q : -1 < q ∧ q < 0) :
  let a : ℕ → ℝ := λ n => a₁ * q^(n - 1)
  (∀ n : ℕ, a n * a (n + 1) < 0) ∧
  (∀ n : ℕ, |a n| > |a (n + 1)|) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_properties_l526_52659


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l526_52698

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the original inequality
def S := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x, f a b c x > 0 ↔ x ∈ S) :
  ∀ x, f c b a x > 0 ↔ x < -1 ∨ x > (1/2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l526_52698


namespace NUMINAMATH_CALUDE_n_divided_by_six_l526_52622

theorem n_divided_by_six (n : ℕ) (h : n = 6^2024) : n / 6 = 6^2023 := by
  sorry

end NUMINAMATH_CALUDE_n_divided_by_six_l526_52622


namespace NUMINAMATH_CALUDE_quadratic_equation_root_zero_l526_52660

theorem quadratic_equation_root_zero (a : ℝ) :
  (∃ x : ℝ, (a - 1) * x^2 + x + a^2 - 1 = 0) ∧
  ((a - 1) * 0^2 + 0 + a^2 - 1 = 0) ∧
  (a - 1 ≠ 0) →
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_zero_l526_52660


namespace NUMINAMATH_CALUDE_spelling_bee_contestants_l526_52648

theorem spelling_bee_contestants (initial_students : ℕ) : 
  (initial_students / 2 : ℚ) / 3 = 24 → initial_students = 144 := by
  sorry

end NUMINAMATH_CALUDE_spelling_bee_contestants_l526_52648


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l526_52601

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Theorem for the solution set of f(x) > 2
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x > 1} ∪ {x : ℝ | x < -5} := by sorry

-- Theorem for the range of t
theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f x ≥ t^2 - (11/2)*t) ↔ (1/2 ≤ t ∧ t ≤ 5) := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_t_l526_52601


namespace NUMINAMATH_CALUDE_quadratic_properties_l526_52605

-- Define the quadratic function
def f (x : ℝ) : ℝ := -(x + 2)^2 - 3

-- State the theorem
theorem quadratic_properties :
  (∀ x y : ℝ, f x < f y → x < y ∨ x > y) ∧  -- Opens downwards
  (∀ x : ℝ, f (x + (-2)) = f ((-2) - x)) ∧  -- Axis of symmetry is x = -2
  (∀ x : ℝ, f x < 0) ∧                      -- Does not intersect x-axis
  (∀ x y : ℝ, x > -1 → y > x → f y < f x)   -- Decreases for x > -1
  :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l526_52605


namespace NUMINAMATH_CALUDE_factors_of_6000_l526_52661

/-- The number of positive integer factors of a number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- The number of positive integer factors of a number that are perfect squares -/
def num_square_factors (n : ℕ) : ℕ := sorry

theorem factors_of_6000 :
  let n : ℕ := 6000
  let factorization : List (ℕ × ℕ) := [(2, 4), (3, 1), (5, 3)]
  (num_factors n = 40) ∧
  (num_factors n - num_square_factors n = 34) := by sorry

end NUMINAMATH_CALUDE_factors_of_6000_l526_52661


namespace NUMINAMATH_CALUDE_tetrahedron_surface_area_l526_52618

-- Define a tetrahedron with edge length 2
def Tetrahedron := {edge_length : ℝ // edge_length = 2}

-- Define the surface area of a tetrahedron
noncomputable def surfaceArea (t : Tetrahedron) : ℝ :=
  4 * Real.sqrt 3

-- Theorem statement
theorem tetrahedron_surface_area (t : Tetrahedron) :
  surfaceArea t = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_surface_area_l526_52618


namespace NUMINAMATH_CALUDE_sixth_term_value_l526_52608

def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, a (n + 1) - a n = 2

theorem sixth_term_value (a : ℕ → ℕ) (h : arithmetic_sequence a) : a 6 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sixth_term_value_l526_52608


namespace NUMINAMATH_CALUDE_sum_of_specific_T_l526_52678

/-- Definition of T_n for n ≥ 2 -/
def T (n : ℕ) : ℤ :=
  if n < 2 then 0 else
  if n % 2 = 0 then -n / 2 else (n + 1) / 2

/-- Theorem stating that T₂₀ + T₃₆ + T₄₅ = -5 -/
theorem sum_of_specific_T : T 20 + T 36 + T 45 = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_T_l526_52678


namespace NUMINAMATH_CALUDE_fred_dimes_remaining_l526_52633

theorem fred_dimes_remaining (initial_dimes borrowed_dimes : ℕ) :
  initial_dimes = 7 →
  borrowed_dimes = 3 →
  initial_dimes - borrowed_dimes = 4 := by
  sorry

end NUMINAMATH_CALUDE_fred_dimes_remaining_l526_52633


namespace NUMINAMATH_CALUDE_centrally_symmetric_implies_congruent_l526_52684

-- Define a shape
def Shape : Type := sorry

-- Define central symmetry
def centrally_symmetric (s1 s2 : Shape) : Prop := 
  ∃ p : ℝ × ℝ, ∃ rotation : Shape → Shape, 
    rotation s1 = s2 ∧ 
    (∀ x : Shape, rotation (rotation x) = x)

-- Define congruence
def congruent (s1 s2 : Shape) : Prop := sorry

-- Theorem statement
theorem centrally_symmetric_implies_congruent (s1 s2 : Shape) :
  centrally_symmetric s1 s2 → congruent s1 s2 := by sorry

end NUMINAMATH_CALUDE_centrally_symmetric_implies_congruent_l526_52684


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l526_52626

theorem cubic_root_equation_solution (x : ℝ) :
  (∃ y : ℝ, x^(1/3) + y^(1/3) = 3 ∧ x + y = 26) →
  (∃ p q : ℤ, x = p - Real.sqrt q ∧ p + q = 26) :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l526_52626


namespace NUMINAMATH_CALUDE_second_set_length_is_twenty_l526_52681

/-- The length of the first set of wood in feet -/
def first_set_length : ℝ := 4

/-- The factor by which the second set is longer than the first set -/
def length_factor : ℝ := 5

/-- The length of the second set of wood in feet -/
def second_set_length : ℝ := first_set_length * length_factor

theorem second_set_length_is_twenty : second_set_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_second_set_length_is_twenty_l526_52681


namespace NUMINAMATH_CALUDE_complement_of_union_equals_five_l526_52697

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 2}

-- Define set N
def N : Finset Nat := {3, 4}

-- Theorem statement
theorem complement_of_union_equals_five :
  (U \ (M ∪ N)) = {5} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_five_l526_52697


namespace NUMINAMATH_CALUDE_quadratic_inequality_product_l526_52619

/-- Given a quadratic inequality x^2 + bx + c < 0 with solution set {x | 2 < x < 4}, 
    prove that bc = -48 -/
theorem quadratic_inequality_product (b c : ℝ) 
  (h : ∀ x, x^2 + b*x + c < 0 ↔ 2 < x ∧ x < 4) : b*c = -48 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_product_l526_52619


namespace NUMINAMATH_CALUDE_volleyball_contributions_l526_52616

theorem volleyball_contributions :
  ∀ (x y z : ℝ),
  -- Condition 1: Third boy contributed 6.4 rubles more than the first boy
  z = x + 6.4 →
  -- Condition 2: Half of first boy's contribution equals one-third of second boy's
  (1/2) * x = (1/3) * y →
  -- Condition 3: Half of first boy's contribution equals one-fourth of third boy's
  (1/2) * x = (1/4) * z →
  -- Conclusion: The contributions are 6.4, 9.6, and 12.8 rubles
  x = 6.4 ∧ y = 9.6 ∧ z = 12.8 :=
by
  sorry


end NUMINAMATH_CALUDE_volleyball_contributions_l526_52616


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l526_52653

theorem factorization_cubic_minus_linear (a : ℝ) : a^3 - 16*a = a*(a + 4)*(a - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l526_52653


namespace NUMINAMATH_CALUDE_product_integers_exist_l526_52649

theorem product_integers_exist : ∃ (a b c : ℝ), 
  (¬ ∃ (n : ℤ), a = n) ∧ 
  (¬ ∃ (n : ℤ), b = n) ∧ 
  (¬ ∃ (n : ℤ), c = n) ∧ 
  (∃ (n : ℤ), a * b = n) ∧ 
  (∃ (n : ℤ), b * c = n) ∧ 
  (∃ (n : ℤ), c * a = n) ∧ 
  (∃ (n : ℤ), a * b * c = n) := by
sorry

end NUMINAMATH_CALUDE_product_integers_exist_l526_52649


namespace NUMINAMATH_CALUDE_largest_non_expressible_l526_52631

/-- A function that checks if a number is composite -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a number can be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
def CanBeExpressed (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ IsComposite m ∧ n = 30 * k + m

/-- Theorem stating that 211 is the largest positive integer that cannot be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
theorem largest_non_expressible : ∀ n : ℕ, n > 211 → CanBeExpressed n ∧ ¬CanBeExpressed 211 :=
sorry

end NUMINAMATH_CALUDE_largest_non_expressible_l526_52631


namespace NUMINAMATH_CALUDE_unripe_oranges_zero_l526_52652

/-- Represents the daily harvest of oranges -/
structure DailyHarvest where
  ripe : ℕ
  unripe : ℕ

/-- Represents the total harvest over a period of days -/
structure TotalHarvest where
  days : ℕ
  ripe : ℕ

/-- Proves that the number of unripe oranges harvested per day is zero -/
theorem unripe_oranges_zero 
  (daily : DailyHarvest) 
  (total : TotalHarvest) 
  (h1 : daily.ripe = 82)
  (h2 : total.days = 25)
  (h3 : total.ripe = 2050)
  (h4 : daily.ripe * total.days = total.ripe) :
  daily.unripe = 0 := by
  sorry

#check unripe_oranges_zero

end NUMINAMATH_CALUDE_unripe_oranges_zero_l526_52652


namespace NUMINAMATH_CALUDE_bike_retail_price_l526_52638

/-- The retail price of a bike, given Maria's savings, her mother's offer, and the additional amount needed. -/
theorem bike_retail_price
  (maria_savings : ℕ)
  (mother_offer : ℕ)
  (additional_needed : ℕ)
  (h1 : maria_savings = 120)
  (h2 : mother_offer = 250)
  (h3 : additional_needed = 230) :
  maria_savings + mother_offer + additional_needed = 600 :=
by sorry

end NUMINAMATH_CALUDE_bike_retail_price_l526_52638


namespace NUMINAMATH_CALUDE_josh_marbles_count_l526_52609

theorem josh_marbles_count (initial_marbles found_marbles : ℕ) 
  (h1 : initial_marbles = 21)
  (h2 : found_marbles = 7) :
  initial_marbles + found_marbles = 28 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_count_l526_52609


namespace NUMINAMATH_CALUDE_B_power_101_l526_52607

def B : Matrix (Fin 3) (Fin 3) ℚ :=
  !![1, 0, 0;
     0, 0, 1;
     0, 1, 0]

theorem B_power_101 : B^101 = B := by sorry

end NUMINAMATH_CALUDE_B_power_101_l526_52607


namespace NUMINAMATH_CALUDE_f_properties_l526_52646

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.tan x

theorem f_properties :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x, f (x + 2 * Real.pi) = f x) ∧
  (∀ x, f (Real.pi - x) = f (Real.pi + x)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l526_52646


namespace NUMINAMATH_CALUDE_runner_time_difference_l526_52611

theorem runner_time_difference (total_distance : ℝ) (second_half_time : ℝ) : 
  total_distance = 40 ∧ second_half_time = 24 →
  ∃ (initial_speed : ℝ),
    initial_speed > 0 ∧
    (total_distance / 2) / initial_speed + (total_distance / 2) / (initial_speed / 2) = second_half_time ∧
    (total_distance / 2) / (initial_speed / 2) - (total_distance / 2) / initial_speed = 12 := by
sorry

end NUMINAMATH_CALUDE_runner_time_difference_l526_52611


namespace NUMINAMATH_CALUDE_collinear_vectors_m_value_l526_52690

/-- Given two non-collinear vectors in a plane, prove that m = -2/3 when the given conditions are met. -/
theorem collinear_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a ≠ 0 ∧ b ≠ 0 ∧ ¬ ∃ (k : ℝ), a = k • b →  -- a and b are non-collinear
  ∃ (A B C : ℝ × ℝ),
    B - A = 2 • a + m • b ∧  -- AB = 2a + mb
    C - B = 3 • a - b ∧  -- BC = 3a - b
    ∃ (t : ℝ), C - A = t • (B - A) →  -- A, B, C are collinear
  m = -2/3 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_m_value_l526_52690


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l526_52635

/-- Given two vectors a and b in R², prove that if they are parallel and have the given components, then m equals either -√2 or √2. -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (m, 2)
  (∃ (k : ℝ), a = k • b) → (m = -Real.sqrt 2 ∨ m = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l526_52635


namespace NUMINAMATH_CALUDE_distance_between_points_l526_52689

/-- The distance between points (0, 12) and (5, 6) is √61 -/
theorem distance_between_points : Real.sqrt 61 = Real.sqrt ((5 - 0)^2 + (6 - 12)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l526_52689


namespace NUMINAMATH_CALUDE_multiple_of_p_plus_q_l526_52671

theorem multiple_of_p_plus_q (p q : ℚ) (h : p / q = 3 / 11) :
  ∃ m : ℤ, m * p + q = 17 ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_p_plus_q_l526_52671


namespace NUMINAMATH_CALUDE_max_congruent_spherical_triangles_l526_52610

/-- A spherical triangle on the surface of a sphere --/
structure SphericalTriangle where
  -- Add necessary fields for a spherical triangle
  is_on_sphere : Bool
  sides_are_great_circle_arcs : Bool
  sides_less_than_quarter : Bool

/-- A division of a sphere into congruent spherical triangles --/
structure SphereDivision where
  triangles : List SphericalTriangle
  are_congruent : Bool

/-- The maximum number of congruent spherical triangles that satisfy the conditions --/
def max_congruent_triangles : ℕ := 60

/-- Theorem stating that 60 is the maximum number of congruent spherical triangles --/
theorem max_congruent_spherical_triangles :
  ∀ (d : SphereDivision),
    (∀ t ∈ d.triangles, t.is_on_sphere ∧ t.sides_are_great_circle_arcs ∧ t.sides_less_than_quarter) →
    d.are_congruent →
    d.triangles.length ≤ max_congruent_triangles :=
by
  sorry

#check max_congruent_spherical_triangles

end NUMINAMATH_CALUDE_max_congruent_spherical_triangles_l526_52610


namespace NUMINAMATH_CALUDE_product_49_sum_0_l526_52651

theorem product_49_sum_0 (a b c d : ℤ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 49 → 
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_product_49_sum_0_l526_52651


namespace NUMINAMATH_CALUDE_paul_total_crayons_l526_52645

/-- The number of crayons Paul received for his birthday -/
def birthday_crayons : ℝ := 479.0

/-- The number of crayons Paul received at the end of the school year -/
def school_year_crayons : ℝ := 134.0

/-- The total number of crayons Paul has now -/
def total_crayons : ℝ := birthday_crayons + school_year_crayons

/-- Theorem stating that Paul's total number of crayons is 613.0 -/
theorem paul_total_crayons : total_crayons = 613.0 := by
  sorry

end NUMINAMATH_CALUDE_paul_total_crayons_l526_52645


namespace NUMINAMATH_CALUDE_seven_digit_divisible_by_nine_l526_52666

theorem seven_digit_divisible_by_nine (m : ℕ) : 
  m < 10 →
  (746 * 1000000 + m * 10000 + 813) % 9 = 0 →
  m = 7 := by
sorry

end NUMINAMATH_CALUDE_seven_digit_divisible_by_nine_l526_52666


namespace NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l526_52664

theorem cos_sum_of_complex_exponentials (θ φ : ℝ) :
  Complex.exp (θ * I) = 4/5 + 3/5 * I →
  Complex.exp (φ * I) = -5/13 + 12/13 * I →
  Real.cos (θ + φ) = -1/13 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_of_complex_exponentials_l526_52664


namespace NUMINAMATH_CALUDE_contest_end_time_l526_52632

def contest_start : Nat := 12 * 60  -- noon in minutes since midnight
def contest_duration : Nat := 1000  -- duration in minutes

theorem contest_end_time :
  (contest_start + contest_duration) % (24 * 60) = 4 * 60 + 40 :=
sorry

end NUMINAMATH_CALUDE_contest_end_time_l526_52632


namespace NUMINAMATH_CALUDE_toll_booth_proof_l526_52625

/-- Represents the number of vehicles passing through a toll booth on each day of the week -/
structure VehicleCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- Represents the toll rates for different vehicle types -/
structure TollRates where
  car : ℕ
  bus : ℕ
  truck : ℕ

def total_vehicles : ℕ := 450

def toll_booth_theorem (vc : VehicleCount) (tr : TollRates) : Prop :=
  vc.monday = 50 ∧
  vc.tuesday = vc.monday + (vc.monday / 10) ∧
  vc.wednesday = 2 * (vc.monday + vc.tuesday) ∧
  vc.thursday = vc.wednesday / 2 ∧
  vc.friday = vc.saturday ∧ vc.saturday = vc.sunday ∧
  vc.monday + vc.tuesday + vc.wednesday + vc.thursday + vc.friday + vc.saturday + vc.sunday = total_vehicles ∧
  tr.car = 2 ∧ tr.bus = 5 ∧ tr.truck = 10 →
  (vc.tuesday + vc.wednesday + vc.thursday = 370) ∧
  (vc.friday = 10) ∧
  ((total_vehicles / 3) * (tr.car + tr.bus + tr.truck) = 2550)

theorem toll_booth_proof (vc : VehicleCount) (tr : TollRates) : 
  toll_booth_theorem vc tr := by sorry

end NUMINAMATH_CALUDE_toll_booth_proof_l526_52625


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l526_52667

def i : ℂ := Complex.I

theorem complex_modulus_problem (a : ℝ) (z : ℂ) 
  (h1 : z = (2 - a * i) / i) 
  (h2 : z.re = 0) : 
  Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l526_52667


namespace NUMINAMATH_CALUDE_weight_of_aluminum_oxide_l526_52693

/-- The atomic weight of aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of aluminum atoms in one molecule of aluminum oxide -/
def Al_count : ℕ := 2

/-- The number of oxygen atoms in one molecule of aluminum oxide -/
def O_count : ℕ := 3

/-- The number of moles of aluminum oxide -/
def moles_Al2O3 : ℝ := 5

/-- The molecular weight of aluminum oxide in g/mol -/
def molecular_weight_Al2O3 : ℝ := Al_count * atomic_weight_Al + O_count * atomic_weight_O

/-- The total weight of the given amount of aluminum oxide in grams -/
def total_weight_Al2O3 : ℝ := moles_Al2O3 * molecular_weight_Al2O3

theorem weight_of_aluminum_oxide :
  total_weight_Al2O3 = 509.8 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_aluminum_oxide_l526_52693


namespace NUMINAMATH_CALUDE_unique_remainder_mod_nine_l526_52674

theorem unique_remainder_mod_nine : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 9 ∧ -1111 ≡ n [ZMOD 9] := by
  sorry

end NUMINAMATH_CALUDE_unique_remainder_mod_nine_l526_52674


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_negative_half_l526_52669

theorem sin_cos_difference_equals_negative_half :
  Real.sin (119 * π / 180) * Real.cos (91 * π / 180) - 
  Real.sin (91 * π / 180) * Real.sin (29 * π / 180) = -1/2 :=
by sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_negative_half_l526_52669


namespace NUMINAMATH_CALUDE_total_shells_l526_52656

/-- The amount of shells in Jovana's bucket -/
def shells_in_bucket (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating the total amount of shells in Jovana's bucket -/
theorem total_shells : shells_in_bucket 5 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_l526_52656


namespace NUMINAMATH_CALUDE_essay_writing_rate_l526_52675

/-- Proves that the writing rate for the first two hours must be 400 words per hour 
    given the conditions of the essay writing problem. -/
theorem essay_writing_rate (total_words : ℕ) (total_hours : ℕ) (later_rate : ℕ) 
    (h1 : total_words = 1200)
    (h2 : total_hours = 4)
    (h3 : later_rate = 200) : 
  ∃ (initial_rate : ℕ), 
    initial_rate * 2 + later_rate * (total_hours - 2) = total_words ∧ 
    initial_rate = 400 := by
  sorry

end NUMINAMATH_CALUDE_essay_writing_rate_l526_52675


namespace NUMINAMATH_CALUDE_unique_solution_l526_52604

/-- Prove that 7 is the only positive integer solution to the equation -/
theorem unique_solution : ∃! (x : ℕ), x > 0 ∧ (1/4 : ℚ) * (10*x + 7 - x^2) - x = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l526_52604


namespace NUMINAMATH_CALUDE_number_divided_by_six_equals_four_l526_52641

theorem number_divided_by_six_equals_four (x : ℝ) : x / 6 = 4 → x = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_six_equals_four_l526_52641


namespace NUMINAMATH_CALUDE_multiply_subtract_difference_l526_52644

theorem multiply_subtract_difference (x : ℝ) (h : x = 13) : 3 * x - (36 - x) = 16 := by
  sorry

end NUMINAMATH_CALUDE_multiply_subtract_difference_l526_52644


namespace NUMINAMATH_CALUDE_compute_M_l526_52654

def M : ℕ → ℕ 
| 0 => 0
| (n + 1) => 
  let k := 4 * n + 2
  (k + 2)^2 + k^2 - 2*((k + 1)^2) - 2*((k - 1)^2) + M n

theorem compute_M : M 12 = 75 := by
  sorry

end NUMINAMATH_CALUDE_compute_M_l526_52654


namespace NUMINAMATH_CALUDE_servant_cash_received_l526_52623

/-- Calculates the cash received by a servant after working for a partial year --/
theorem servant_cash_received
  (annual_cash : ℕ)
  (turban_price : ℕ)
  (months_worked : ℕ)
  (h1 : annual_cash = 90)
  (h2 : turban_price = 50)
  (h3 : months_worked = 9) :
  (months_worked * (annual_cash + turban_price) / 12) - turban_price = 55 :=
sorry

end NUMINAMATH_CALUDE_servant_cash_received_l526_52623


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_function_l526_52687

/-- Represents an isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  /-- Length of each of the two equal sides -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The triangle is isosceles with perimeter 20 -/
  is_isosceles : 2 * x + y = 20
  /-- The side lengths satisfy the triangle inequality -/
  triangle_inequality : x + x > y
  /-- The side lengths are positive -/
  positive_sides : 0 < x ∧ 0 < y

/-- The functional relationship between the base and equal sides of the isosceles triangle -/
theorem isosceles_triangle_base_function (t : IsoscelesTriangle) :
  t.y = 20 - 2 * t.x ∧ 5 < t.x ∧ t.x < 10 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_function_l526_52687


namespace NUMINAMATH_CALUDE_money_left_l526_52614

def salary_distribution (S : ℝ) : Prop :=
  let house_rent := (2/5) * S
  let food := (3/10) * S
  let conveyance := (1/8) * S
  let food_and_conveyance := food + conveyance
  food_and_conveyance = 3399.999999999999

theorem money_left (S : ℝ) (h : salary_distribution S) : 
  S - ((2/5 + 3/10 + 1/8) * S) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_money_left_l526_52614


namespace NUMINAMATH_CALUDE_inequality_proof_l526_52642

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  b^2 / a + a^2 / b ≥ a + b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l526_52642


namespace NUMINAMATH_CALUDE_line_increase_l526_52682

/-- Given a line where an increase of 5 units in x corresponds to an increase of 11 units in y,
    prove that an increase of 15 units in x corresponds to an increase of 33 units in y. -/
theorem line_increase (m : ℝ) (h : m = 11 / 5) : m * 15 = 33 := by
  sorry

end NUMINAMATH_CALUDE_line_increase_l526_52682


namespace NUMINAMATH_CALUDE_constant_term_expansion_l526_52679

/-- The constant term in the expansion of (3x^2 + 2/x)^8 -/
def constant_term : ℕ :=
  (Nat.choose 8 4) * (3^4) * (2^4)

/-- Theorem stating that the constant term in the expansion of (3x^2 + 2/x)^8 is 90720 -/
theorem constant_term_expansion :
  constant_term = 90720 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l526_52679


namespace NUMINAMATH_CALUDE_triangle_inequalities_l526_52672

variables {a b c p r R r_a r_b r_c S : ℝ}

-- Define triangle properties
axiom triangle_sides : a > 0 ∧ b > 0 ∧ c > 0
axiom semiperimeter : p = (a + b + c) / 2
axiom area_positive : S > 0

-- Define relationships between radii
axiom radii_sum : 4 * R + r = r_a + r_b + r_c
axiom radius_inequality : R - 2 * r ≥ 0

-- Define exradii sum
axiom exradii_sum : r_a + r_b + r_c = p * r * (1 / (p - a) + 1 / (p - b) + 1 / (p - c))

-- Define relationship between exradii and area
axiom exradii_area : 1 / (p - a) + 1 / (p - b) + 1 / (p - c) = (a * b + b * c + c * a - p^2) / S

-- Define inequality for sides and area
axiom sides_area_inequality : 2 * (a * b + b * c + c * a) - (a^2 + b^2 + c^2) ≥ 4 * Real.sqrt 3 * S

-- Theorem to prove
theorem triangle_inequalities :
  (5 * R - r ≥ Real.sqrt 3 * p) ∧
  (4 * R - r_a ≥ (p - a) * (Real.sqrt 3 + (a^2 + (b - c)^2) / (2 * S))) :=
sorry

end NUMINAMATH_CALUDE_triangle_inequalities_l526_52672


namespace NUMINAMATH_CALUDE_f_composition_of_three_l526_52634

def f (n : ℕ) : ℕ :=
  if n ≤ 3 then n^2 + 1 else 4*n + 2

theorem f_composition_of_three : f (f (f 3)) = 170 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_of_three_l526_52634


namespace NUMINAMATH_CALUDE_chess_pieces_count_l526_52683

theorem chess_pieces_count (black_pieces : ℕ) (prob_black : ℚ) (white_pieces : ℕ) : 
  black_pieces = 6 → 
  prob_black = 1 / 5 → 
  (black_pieces : ℚ) / ((black_pieces : ℚ) + (white_pieces : ℚ)) = prob_black →
  white_pieces = 24 := by
sorry

end NUMINAMATH_CALUDE_chess_pieces_count_l526_52683


namespace NUMINAMATH_CALUDE_ball_placement_theorem_l526_52613

/-- The number of ways to place 5 numbered balls into 5 numbered boxes -/
def ball_placement_count : ℕ := 20

/-- A function that returns the number of ways to place n numbered balls into n numbered boxes
    such that exactly k balls match their box numbers -/
def place_balls (n k : ℕ) : ℕ := sorry

/-- The theorem stating that the number of ways to place 5 numbered balls into 5 numbered boxes,
    where each box contains one ball and exactly two balls match their box numbers, is 20 -/
theorem ball_placement_theorem : place_balls 5 2 = ball_placement_count := by sorry

end NUMINAMATH_CALUDE_ball_placement_theorem_l526_52613


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_sixteen_l526_52655

theorem sum_of_solutions_eq_sixteen : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 8)^2 = 16 ∧ (x₂ - 8)^2 = 16 ∧ x₁ + x₂ = 16 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_sixteen_l526_52655


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l526_52647

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- For the quadratic equation 3x^2 - 2x - 1 = 0, the discriminant equals 16 -/
theorem quadratic_discriminant : discriminant 3 (-2) (-1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l526_52647


namespace NUMINAMATH_CALUDE_max_attempts_l526_52695

/-- The number of unique arrangements of a four-digit number containing one 2, one 9, and two 6s -/
def password_arrangements : ℕ := sorry

/-- The maximum number of attempts needed to find the correct password -/
theorem max_attempts : password_arrangements = 12 := by sorry

end NUMINAMATH_CALUDE_max_attempts_l526_52695


namespace NUMINAMATH_CALUDE_hyosung_mimi_distance_l526_52662

/-- Calculates the remaining distance between two people walking towards each other. -/
def remaining_distance (initial_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  initial_distance - (speed1 + speed2) * time

/-- Theorem stating the remaining distance between Hyosung and Mimi after 15 minutes. -/
theorem hyosung_mimi_distance :
  let initial_distance : ℝ := 2.5
  let mimi_speed : ℝ := 2.4
  let hyosung_speed : ℝ := 0.08 * 60
  let time : ℝ := 15 / 60
  remaining_distance initial_distance mimi_speed hyosung_speed time = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_hyosung_mimi_distance_l526_52662


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l526_52620

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l526_52620


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l526_52670

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {1, 3}

-- Theorem statement
theorem complement_intersection_theorem :
  (Set.compl A ∩ B) = {3} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l526_52670


namespace NUMINAMATH_CALUDE_total_cost_is_840_l526_52615

/-- The cost of a single movie ticket in dollars -/
def movie_ticket_cost : ℚ := 30

/-- The number of movie tickets -/
def num_movie_tickets : ℕ := 8

/-- The number of football game tickets -/
def num_football_tickets : ℕ := 5

/-- The ratio of the cost of 8 movie tickets to 1 football game ticket -/
def cost_ratio : ℚ := 2

/-- The total cost of buying movie tickets and football game tickets -/
def total_cost : ℚ :=
  (num_movie_tickets : ℚ) * movie_ticket_cost +
  (num_football_tickets : ℚ) * ((num_movie_tickets : ℚ) * movie_ticket_cost / cost_ratio)

theorem total_cost_is_840 : total_cost = 840 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_840_l526_52615


namespace NUMINAMATH_CALUDE_equation_not_quadratic_l526_52650

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def equation (y : ℝ) : ℝ := 3 * y * (y - 1) - y * (3 * y + 1)

theorem equation_not_quadratic : ¬ is_quadratic equation := by
  sorry

end NUMINAMATH_CALUDE_equation_not_quadratic_l526_52650


namespace NUMINAMATH_CALUDE_horse_pig_compensation_difference_l526_52627

-- Define the consumption of each animal type relative to a pig
def pig_consumption : ℚ := 1
def sheep_consumption : ℚ := 2 * pig_consumption
def horse_consumption : ℚ := 2 * sheep_consumption
def cow_consumption : ℚ := 2 * horse_consumption

-- Define the total compensation
def total_compensation : ℚ := 9

-- Theorem statement
theorem horse_pig_compensation_difference :
  let total_consumption := pig_consumption + sheep_consumption + horse_consumption + cow_consumption
  let unit_compensation := total_compensation / total_consumption
  let horse_compensation := horse_consumption * unit_compensation
  let pig_compensation := pig_consumption * unit_compensation
  horse_compensation - pig_compensation = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_horse_pig_compensation_difference_l526_52627


namespace NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l526_52699

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1: When a = -2, A ∩ B = {x | -5 ≤ x < -1}
theorem intersection_when_a_neg_two :
  A (-2) ∩ B = {x : ℝ | -5 ≤ x ∧ x < -1} :=
sorry

-- Theorem 2: A ⊆ B if and only if a ≤ -4 or a ≥ 3
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ a ≤ -4 ∨ a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l526_52699


namespace NUMINAMATH_CALUDE_inverse_variation_cube_l526_52621

theorem inverse_variation_cube (y : ℝ → ℝ) (k : ℝ) :
  (∀ x, x ≠ 0 → 3 * (y x) = k / (x^3)) →  -- 3y varies inversely as the cube of x
  y 3 = 27 →                              -- y = 27 when x = 3
  y 9 = 1 :=                              -- y = 1 when x = 9
by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_cube_l526_52621


namespace NUMINAMATH_CALUDE_partnership_investment_l526_52639

/-- Partnership investment problem -/
theorem partnership_investment (x : ℝ) : 
  let total_gain : ℝ := 15000
  let a_investment := x
  let b_investment := 2 * x
  let c_investment := 3 * x
  let a_time := 12
  let b_time := 6
  let c_time := 4
  let total_ratio := a_investment * a_time + b_investment * b_time + c_investment * c_time
  let a_share := (a_investment * a_time / total_ratio) * total_gain
  a_share = 5000 := by
sorry


end NUMINAMATH_CALUDE_partnership_investment_l526_52639


namespace NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l526_52628

/-- A function that checks if a fraction is a terminating decimal -/
def isTerminatingDecimal (numerator : ℕ) (denominator : ℕ) : Prop :=
  ∃ (a b : ℕ), denominator = 2^a * 5^b

/-- The smallest positive integer n such that n/(n+150) is a terminating decimal -/
theorem smallest_n_for_terminating_decimal : 
  (∀ n : ℕ, n > 0 → n < 50 → ¬(isTerminatingDecimal n (n + 150))) ∧ 
  (isTerminatingDecimal 50 200) := by
  sorry

#check smallest_n_for_terminating_decimal

end NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l526_52628


namespace NUMINAMATH_CALUDE_tomato_growth_l526_52658

theorem tomato_growth (initial_tomatoes : ℕ) (increase_factor : ℕ) 
  (h1 : initial_tomatoes = 36) 
  (h2 : increase_factor = 100) : 
  initial_tomatoes * increase_factor = 3600 := by
sorry

end NUMINAMATH_CALUDE_tomato_growth_l526_52658


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_1_plus_2i_l526_52602

theorem imaginary_part_of_i_times_1_plus_2i (i : ℂ) (h : i * i = -1) :
  Complex.im (i * (1 + 2*i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_1_plus_2i_l526_52602


namespace NUMINAMATH_CALUDE_equation_solutions_l526_52673

theorem equation_solutions :
  (∃ x : ℝ, 4 * x + 3 = 5 * x - 1 ∧ x = 4) ∧
  (∃ x : ℝ, 4 * (x - 1) = 1 - x ∧ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l526_52673


namespace NUMINAMATH_CALUDE_brian_trip_distance_l526_52612

/-- Calculates the distance traveled given car efficiency, initial tank capacity, and fuel used --/
def distanceTraveled (efficiency : ℝ) (initialTank : ℝ) (fuelUsed : ℝ) : ℝ :=
  efficiency * fuelUsed

/-- Represents Brian's trip --/
structure BrianTrip where
  efficiency : ℝ
  initialTank : ℝ
  remainingFuelFraction : ℝ
  drivingTime : ℝ

/-- Theorem stating the distance Brian traveled --/
theorem brian_trip_distance (trip : BrianTrip) 
  (h1 : trip.efficiency = 20)
  (h2 : trip.initialTank = 15)
  (h3 : trip.remainingFuelFraction = 3/7)
  (h4 : trip.drivingTime = 2) :
  ∃ (distance : ℝ), abs (distance - distanceTraveled trip.efficiency trip.initialTank (trip.initialTank * (1 - trip.remainingFuelFraction))) < 0.1 :=
by sorry

#check brian_trip_distance

end NUMINAMATH_CALUDE_brian_trip_distance_l526_52612


namespace NUMINAMATH_CALUDE_ancient_pi_approximation_l526_52677

theorem ancient_pi_approximation (V : ℝ) (r : ℝ) (d : ℝ) :
  V = (4 / 3) * Real.pi * r^3 →
  d = (16 / 9 * V)^(1/3) →
  (6 * 9) / 16 = 3.375 :=
by sorry

end NUMINAMATH_CALUDE_ancient_pi_approximation_l526_52677


namespace NUMINAMATH_CALUDE_base5_product_theorem_l526_52680

/-- Converts a base-5 number to base-10 --/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a base-10 number to base-5 --/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Multiplies two base-5 numbers --/
def multiplyBase5 (a b : List Nat) : List Nat :=
  base10ToBase5 (base5ToBase10 a * base5ToBase10 b)

theorem base5_product_theorem :
  multiplyBase5 [1, 3, 2] [3, 1] = [3, 0, 0, 1, 4] := by sorry

end NUMINAMATH_CALUDE_base5_product_theorem_l526_52680


namespace NUMINAMATH_CALUDE_problem_solution_l526_52636

def A : Set ℝ := {-2, 3, 4, 6}
def B (a : ℝ) : Set ℝ := {3, a, a^2}

theorem problem_solution (a : ℝ) :
  (B a ⊆ A → a = 2) ∧
  (A ∩ B a = {3, 4} → a = 2 ∨ a = 4) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l526_52636


namespace NUMINAMATH_CALUDE_right_triangle_angles_l526_52665

theorem right_triangle_angles (α β : Real) : 
  α > 0 → β > 0 → α + β = π / 2 →
  Real.tan α + Real.tan β + (Real.tan α)^2 + (Real.tan β)^2 + (Real.tan α)^3 + (Real.tan β)^3 = 70 →
  α = π / 2.4 ∧ β = π / 12 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_angles_l526_52665


namespace NUMINAMATH_CALUDE_number_equation_solution_l526_52696

theorem number_equation_solution :
  ∃ x : ℝ, x - (1002 / 20.04) = 1295 ∧ x = 1345 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l526_52696


namespace NUMINAMATH_CALUDE_variance_fluctuation_relationship_l526_52624

/-- Definition of variance for a list of numbers -/
def variance (data : List ℝ) : ℝ := sorry

/-- Definition of fluctuation for a list of numbers -/
def fluctuation (data : List ℝ) : ℝ := sorry

/-- Theorem: If the variance of data set A is greater than the variance of data set B,
    then the fluctuation of A is greater than the fluctuation of B -/
theorem variance_fluctuation_relationship (A B : List ℝ) :
  variance A > variance B → fluctuation A > fluctuation B := by sorry

end NUMINAMATH_CALUDE_variance_fluctuation_relationship_l526_52624


namespace NUMINAMATH_CALUDE_det_A_plus_5_l526_52600

def A : Matrix (Fin 2) (Fin 2) ℝ := !![6, -2; -3, 7]

theorem det_A_plus_5 : Matrix.det A + 5 = 41 := by
  sorry

end NUMINAMATH_CALUDE_det_A_plus_5_l526_52600


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l526_52640

theorem quadratic_inequality_solution_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 2 * x + 6 * k < 0) → k < -Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l526_52640


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l526_52668

theorem polynomial_division_remainder (k : ℝ) : 
  (∀ x : ℝ, ∃ q : ℝ, 3 * x^3 - k * x^2 + 4 = (3 * x - 1) * q + 5) → k = -8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l526_52668


namespace NUMINAMATH_CALUDE_larger_number_proof_l526_52637

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 480) 
  (h2 : y = 4 * x + 30) : 
  y = 630 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l526_52637


namespace NUMINAMATH_CALUDE_unique_solution_quartic_l526_52691

theorem unique_solution_quartic (n : ℤ) : 
  (∃! x : ℝ, 4 * x^4 + n * x^2 + 4 = 0) ↔ (n = 8 ∨ n = -8) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quartic_l526_52691


namespace NUMINAMATH_CALUDE_estate_distribution_l526_52617

/-- Mrs. K's estate distribution problem -/
theorem estate_distribution (E : ℝ) 
  (daughters_share : ℝ) 
  (husband_share : ℝ) 
  (gardener_share : ℝ) : 
  (daughters_share = 0.4 * E) →
  (husband_share = 3 * daughters_share) →
  (gardener_share = 1000) →
  (E = daughters_share + husband_share + gardener_share) →
  (E = 2500) := by
sorry

end NUMINAMATH_CALUDE_estate_distribution_l526_52617


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l526_52692

theorem inequality_and_equality_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x + y^2 / x ≥ 2 * y ∧ (x + y^2 / x = 2 * y ↔ x = y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l526_52692


namespace NUMINAMATH_CALUDE_inequality_iff_solution_set_l526_52685

def inequality (x : ℝ) : Prop :=
  (2 / (x + 2)) + (8 / (x + 6)) ≥ 2

theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ -6 < x ∧ x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_iff_solution_set_l526_52685


namespace NUMINAMATH_CALUDE_simplest_form_iff_coprime_l526_52629

/-- A fraction is a pair of integers where the denominator is non-zero -/
structure Fraction where
  numerator : Int
  denominator : Int
  denom_nonzero : denominator ≠ 0

/-- A fraction is in its simplest form if it cannot be reduced further -/
def is_simplest_form (f : Fraction) : Prop :=
  ∀ k : Int, k ≠ 0 → ¬(k ∣ f.numerator ∧ k ∣ f.denominator)

/-- Two integers are coprime if their greatest common divisor is 1 -/
def are_coprime (a b : Int) : Prop :=
  Int.gcd a b = 1

/-- Theorem: A fraction is in its simplest form if and only if its numerator and denominator are coprime -/
theorem simplest_form_iff_coprime (f : Fraction) :
  is_simplest_form f ↔ are_coprime f.numerator f.denominator := by
  sorry

end NUMINAMATH_CALUDE_simplest_form_iff_coprime_l526_52629


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_constant_l526_52694

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum_constant (a : ℕ → ℝ) (S : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (∀ n : ℕ+, S n = n^2 + 2*n + (S 1 - 3)) →
  S 1 - 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_constant_l526_52694


namespace NUMINAMATH_CALUDE_hotel_rooms_theorem_l526_52676

/-- The minimum number of rooms needed for 100 tourists with k rooms under renovation -/
def min_rooms (k : ℕ) : ℕ :=
  let m := k / 2
  if k % 2 = 0 then
    100 * (m + 1)
  else
    100 * (m + 1) + 1

/-- Theorem stating the minimum number of rooms needed for 100 tourists -/
theorem hotel_rooms_theorem (k : ℕ) :
  ∀ n : ℕ, n ≥ min_rooms k →
  ∃ strategy : (Fin 100 → Fin n → Option (Fin n)),
  (∀ i : Fin 100, ∃ room : Fin n, strategy i room = some room) ∧
  (∀ i j : Fin 100, i ≠ j →
    ∀ room : Fin n, strategy i room ≠ none → strategy j room = none) :=
by
  sorry

#check hotel_rooms_theorem

end NUMINAMATH_CALUDE_hotel_rooms_theorem_l526_52676


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l526_52630

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_domain
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_positive : ∀ x > 0, f x = x * (1 + x)) :
  ∀ x < 0, f x = x * (1 - x) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l526_52630
