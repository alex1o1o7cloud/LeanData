import Mathlib

namespace equation_roots_l3340_334053

theorem equation_roots : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 4 ∧ x₂ = -2.5) ∧ 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -2 → (18 / (x^2 - 4) - 3 / (x - 2) = 2 ↔ (x = x₁ ∨ x = x₂))) :=
by sorry

end equation_roots_l3340_334053


namespace imaginary_part_of_z_l3340_334087

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 2) : 
  Complex.im z = -1 := by
  sorry

end imaginary_part_of_z_l3340_334087


namespace hexagonal_lattice_triangles_l3340_334008

/-- Represents a point in the hexagonal lattice -/
structure LatticePoint where
  x : ℝ
  y : ℝ

/-- The hexagonal lattice with two concentric hexagons -/
structure HexagonalLattice where
  center : LatticePoint
  inner_hexagon : List LatticePoint
  outer_hexagon : List LatticePoint

/-- Checks if three points form an equilateral triangle -/
def is_equilateral_triangle (p1 p2 p3 : LatticePoint) : Prop :=
  sorry

/-- Counts the number of equilateral triangles in the lattice -/
def count_equilateral_triangles (lattice : HexagonalLattice) : ℕ :=
  sorry

/-- Main theorem: The number of equilateral triangles in the described hexagonal lattice is 18 -/
theorem hexagonal_lattice_triangles 
  (lattice : HexagonalLattice)
  (h1 : lattice.inner_hexagon.length = 6)
  (h2 : lattice.outer_hexagon.length = 6)
  (h3 : ∀ p ∈ lattice.inner_hexagon, 
    ∃ q ∈ lattice.inner_hexagon, 
    (p.x - q.x)^2 + (p.y - q.y)^2 = 1)
  (h4 : ∀ p ∈ lattice.outer_hexagon, 
    ∃ q ∈ lattice.inner_hexagon, 
    (p.x - q.x)^2 + (p.y - q.y)^2 = 4) :
  count_equilateral_triangles lattice = 18 :=
sorry

end hexagonal_lattice_triangles_l3340_334008


namespace arithmetic_sequence_property_l3340_334061

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def geometric_sequence (a b c : ℝ) : Prop :=
  b * b = a * c

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (d : ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 4)
  (h3 : arithmetic_sequence a d)
  (h4 : geometric_sequence (a 1) (a 3) (a 4)) :
  (∀ n : ℕ, a n = 5 - n) ∧
  (∃ max_sum : ℝ, max_sum = 10 ∧
    ∀ n : ℕ, (n * (2 * a 1 + (n - 1) * d)) / 2 ≤ max_sum) :=
sorry

end arithmetic_sequence_property_l3340_334061


namespace initial_girls_percentage_l3340_334026

theorem initial_girls_percentage 
  (initial_total : ℕ)
  (new_boys : ℕ)
  (new_girls_percentage : ℚ)
  (h1 : initial_total = 20)
  (h2 : new_boys = 5)
  (h3 : new_girls_percentage = 32 / 100) :
  let initial_girls := (new_girls_percentage * (initial_total + new_boys)).floor
  let initial_girls_percentage := initial_girls / initial_total
  initial_girls_percentage = 2 / 5 := by sorry

end initial_girls_percentage_l3340_334026


namespace inequality_proof_l3340_334065

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b ≥ 1) :
  (a + 2 * b + 2 / (a + 1)) * (b + 2 * a + 2 / (b + 1)) ≥ 16 := by
  sorry

end inequality_proof_l3340_334065


namespace min_value_theorem_l3340_334019

open Real

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → 1/x + 4/y ≤ 1/a + 4/b) ∧ (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = 1 ∧ 1/a + 4/b = 9) :=
by sorry

end min_value_theorem_l3340_334019


namespace range_of_f_on_interval_existence_of_a_l3340_334013

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a

-- Part 1
theorem range_of_f_on_interval :
  let f1 := f 1
  ∃ (y : ℝ), y ∈ Set.range (fun x => f1 x) ∩ Set.Icc 0 4 ∧
  ∀ (z : ℝ), z ∈ Set.range (fun x => f1 x) ∩ Set.Icc 0 3 → z ∈ Set.Icc 0 4 :=
sorry

-- Part 2
theorem existence_of_a :
  ∃ (a : ℝ), 
    (∀ x, x ∈ Set.Icc (-1) 1 → f a x ∈ Set.Icc (-2) 2) ∧
    (∀ y, y ∈ Set.Icc (-2) 2 → ∃ x ∈ Set.Icc (-1) 1, f a x = y) ∧
    a = -1 :=
sorry

end range_of_f_on_interval_existence_of_a_l3340_334013


namespace lcm_lower_bound_l3340_334089

theorem lcm_lower_bound (a : Fin 10 → ℕ) (h_order : ∀ i j, i < j → a i < a j) :
  Nat.lcm (a 0) (Nat.lcm (a 1) (Nat.lcm (a 2) (Nat.lcm (a 3) (Nat.lcm (a 4) (Nat.lcm (a 5) (Nat.lcm (a 6) (Nat.lcm (a 7) (Nat.lcm (a 8) (a 9))))))))) ≥ 10 * a 0 := by
  sorry

end lcm_lower_bound_l3340_334089


namespace bowling_ball_weight_l3340_334066

theorem bowling_ball_weight :
  ∀ (bowling_ball_weight canoe_weight : ℝ),
    (8 * bowling_ball_weight = 4 * canoe_weight) →
    (3 * canoe_weight = 108) →
    bowling_ball_weight = 18 := by
  sorry

end bowling_ball_weight_l3340_334066


namespace proposition_analysis_l3340_334045

theorem proposition_analysis :
  let converse := ∀ a b c : ℝ, a * c^2 > b * c^2 → a > b
  let negation := ∃ a b c : ℝ, a > b ∧ a * c^2 ≤ b * c^2
  let contrapositive := ∀ a b c : ℝ, a * c^2 ≤ b * c^2 → a ≤ b
  (converse ∧ negation ∧ ¬contrapositive) ∨
  (converse ∧ ¬negation ∧ contrapositive) ∨
  (¬converse ∧ negation ∧ contrapositive) :=
by sorry

#check proposition_analysis

end proposition_analysis_l3340_334045


namespace opposite_of_one_third_l3340_334063

theorem opposite_of_one_third : 
  (opposite : ℚ → ℚ) (1/3) = -(1/3) := by
  sorry

end opposite_of_one_third_l3340_334063


namespace base_conversion_theorem_l3340_334095

theorem base_conversion_theorem (n : ℕ) (C D : ℕ) : 
  n > 0 ∧ 
  C < 8 ∧ 
  D < 5 ∧ 
  n = 8 * C + D ∧ 
  n = 5 * D + C → 
  n = 0 := by sorry

end base_conversion_theorem_l3340_334095


namespace different_genre_pairs_count_l3340_334070

/-- Represents the number of books in each genre -/
structure BookCollection where
  mystery : Nat
  fantasy : Nat
  biography : Nat

/-- Calculates the number of possible pairs of books from different genres -/
def differentGenrePairs (books : BookCollection) : Nat :=
  books.mystery * books.fantasy +
  books.mystery * books.biography +
  books.fantasy * books.biography

/-- Theorem: Given 4 mystery novels, 3 fantasy novels, and 2 biographies,
    the number of possible pairs of books from different genres is 26 -/
theorem different_genre_pairs_count :
  differentGenrePairs ⟨4, 3, 2⟩ = 26 := by
  sorry

end different_genre_pairs_count_l3340_334070


namespace pauls_lost_crayons_l3340_334016

/-- Given Paul's crayon situation, prove the number of lost crayons --/
theorem pauls_lost_crayons
  (initial_crayons : ℕ)
  (given_to_friends : ℕ)
  (total_lost_or_given : ℕ)
  (h1 : initial_crayons = 65)
  (h2 : given_to_friends = 213)
  (h3 : total_lost_or_given = 229)
  : total_lost_or_given - given_to_friends = 16 := by
  sorry

end pauls_lost_crayons_l3340_334016


namespace right_triangle_third_side_square_l3340_334091

theorem right_triangle_third_side_square (a b c : ℝ) : 
  a = 6 → b = 8 → (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c^2 = 28 ∨ c^2 = 100 := by
  sorry

end right_triangle_third_side_square_l3340_334091


namespace equation_proof_l3340_334028

theorem equation_proof : 484 + 2 * 22 * 3 + 9 = 625 := by
  sorry

end equation_proof_l3340_334028


namespace even_function_implies_a_zero_l3340_334004

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

/-- The function f(x) = x^2 - |x + a| -/
def f (a : ℝ) : ℝ → ℝ := fun x ↦ x^2 - |x + a|

theorem even_function_implies_a_zero :
  ∀ a : ℝ, IsEven (f a) → a = 0 := by
  sorry

end even_function_implies_a_zero_l3340_334004


namespace cubic_function_sum_l3340_334073

-- Define the function f
def f (a b c x : ℤ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_function_sum (a b c : ℤ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧   -- a, b, c are non-zero
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧   -- a, b, c are distinct
  f a b c a = a^3 ∧         -- f(a) = a^3
  f a b c b = b^3           -- f(b) = b^3
  → a + b + c = 18 := by
  sorry

end cubic_function_sum_l3340_334073


namespace problem_statement_l3340_334090

theorem problem_statement (x : ℝ) :
  x^2 + 9 * (x / (x - 3))^2 = 90 →
  let y := ((x - 3)^2 * (x + 4)) / (2 * x - 4)
  y = 39 ∨ y = 6 := by sorry

end problem_statement_l3340_334090


namespace base5_conversion_and_modulo_l3340_334033

/-- Converts a base 5 number to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Computes the modulo of a number -/
def modulo (n m : Nat) : Nat :=
  n % m

theorem base5_conversion_and_modulo :
  let base5Num : List Nat := [4, 1, 0, 1, 2]  -- 21014 in base 5, least significant digit first
  let base10Num : Nat := base5ToBase10 base5Num
  base10Num = 1384 ∧ modulo base10Num 7 = 6 := by
  sorry

#eval base5ToBase10 [4, 1, 0, 1, 2]  -- Should output 1384
#eval modulo 1384 7  -- Should output 6

end base5_conversion_and_modulo_l3340_334033


namespace no_odd_cube_ending_668_l3340_334078

theorem no_odd_cube_ending_668 : ¬∃ (n : ℕ), 
  Odd n ∧ n > 0 ∧ n^3 % 1000 = 668 := by
  sorry

end no_odd_cube_ending_668_l3340_334078


namespace range_of_a_minus_abs_b_l3340_334038

theorem range_of_a_minus_abs_b (a b : ℝ) (ha : 1 < a ∧ a < 3) (hb : -4 < b ∧ b < 2) :
  -3 < a - |b| ∧ a - |b| < 3 := by
  sorry

end range_of_a_minus_abs_b_l3340_334038


namespace value_of_a_l3340_334083

theorem value_of_a (x a : ℝ) (hx : x ≠ 1) : 
  (8 * a) / (1 - x^32) = 2 / (1 - x) + 2 / (1 + x) + 4 / (1 + x^2) + 
                         8 / (1 + x^4) + 16 / (1 + x^8) + 32 / (1 + x^16) → 
  a = 8 := by
sorry

end value_of_a_l3340_334083


namespace bucket3_most_efficient_bucket3_count_verification_l3340_334077

-- Define the tank capacities
def tank1_capacity : ℕ := 20000
def tank2_capacity : ℕ := 25000
def tank3_capacity : ℕ := 30000

-- Define the bucket capacities
def bucket1_capacity : ℕ := 13
def bucket2_capacity : ℕ := 28
def bucket3_capacity : ℕ := 36

-- Function to calculate the number of buckets needed
def buckets_needed (tank_capacity bucket_capacity : ℕ) : ℕ :=
  (tank_capacity + bucket_capacity - 1) / bucket_capacity

-- Theorem stating that the 36-litre bucket is most efficient for all tanks
theorem bucket3_most_efficient :
  (buckets_needed tank1_capacity bucket3_capacity ≤ buckets_needed tank1_capacity bucket1_capacity) ∧
  (buckets_needed tank1_capacity bucket3_capacity ≤ buckets_needed tank1_capacity bucket2_capacity) ∧
  (buckets_needed tank2_capacity bucket3_capacity ≤ buckets_needed tank2_capacity bucket1_capacity) ∧
  (buckets_needed tank2_capacity bucket3_capacity ≤ buckets_needed tank2_capacity bucket2_capacity) ∧
  (buckets_needed tank3_capacity bucket3_capacity ≤ buckets_needed tank3_capacity bucket1_capacity) ∧
  (buckets_needed tank3_capacity bucket3_capacity ≤ buckets_needed tank3_capacity bucket2_capacity) :=
by sorry

-- Verify the exact number of 36-litre buckets needed for each tank
theorem bucket3_count_verification :
  (buckets_needed tank1_capacity bucket3_capacity = 556) ∧
  (buckets_needed tank2_capacity bucket3_capacity = 695) ∧
  (buckets_needed tank3_capacity bucket3_capacity = 834) :=
by sorry

end bucket3_most_efficient_bucket3_count_verification_l3340_334077


namespace log_equation_solution_l3340_334005

theorem log_equation_solution :
  ∃ y : ℝ, (2 * Real.log y + 3 * Real.log 2 = 1) ∧ (y = Real.sqrt 5 / 2) := by
  sorry

end log_equation_solution_l3340_334005


namespace amara_clothes_thrown_away_l3340_334082

/-- The number of clothes Amara threw away -/
def clothes_thrown_away (initial_count donated_first donated_second remaining_count : ℕ) : ℕ :=
  initial_count - donated_first - donated_second - remaining_count

/-- Proof that Amara threw away 15 pieces of clothing -/
theorem amara_clothes_thrown_away :
  clothes_thrown_away 100 5 (3 * 5) 65 = 15 := by
  sorry

end amara_clothes_thrown_away_l3340_334082


namespace cyclist_time_is_pi_over_five_l3340_334000

/-- Represents the problem of a cyclist riding on a highway strip -/
def CyclistProblem (width : ℝ) (length : ℝ) (large_semicircle_distance : ℝ) (speed : ℝ) : Prop :=
  width = 40 ∧ 
  length = 5280 ∧ 
  large_semicircle_distance = 528 ∧ 
  speed = 5

/-- Calculates the time taken for the cyclist to cover the entire strip -/
noncomputable def cycleTime (width : ℝ) (length : ℝ) (large_semicircle_distance : ℝ) (speed : ℝ) : ℝ :=
  (Real.pi * length) / (speed * width)

/-- Theorem stating that the time taken is π/5 hours -/
theorem cyclist_time_is_pi_over_five 
  (width : ℝ) (length : ℝ) (large_semicircle_distance : ℝ) (speed : ℝ) 
  (h : CyclistProblem width length large_semicircle_distance speed) : 
  cycleTime width length large_semicircle_distance speed = Real.pi / 5 := by
  sorry

end cyclist_time_is_pi_over_five_l3340_334000


namespace line_slope_is_pi_over_three_l3340_334042

theorem line_slope_is_pi_over_three (x y : ℝ) :
  2 * Real.sqrt 3 * x - 2 * y - 1 = 0 →
  ∃ (m : ℝ), (∀ x y, y = m * x - 1 / 2) ∧ m = Real.tan (π / 3) :=
sorry

end line_slope_is_pi_over_three_l3340_334042


namespace area_BCD_equals_135_l3340_334084

-- Define the triangle ABC
def triangle_ABC : Set (ℝ × ℝ) := sorry

-- Define the triangle BCD
def triangle_BCD : Set (ℝ × ℝ) := sorry

-- Define the area function
def area : Set (ℝ × ℝ) → ℝ := sorry

-- Define the length function
def length : ℝ × ℝ → ℝ × ℝ → ℝ := sorry

-- Define point A
def A : ℝ × ℝ := sorry

-- Define point C
def C : ℝ × ℝ := sorry

-- Define point D
def D : ℝ × ℝ := sorry

-- Theorem statement
theorem area_BCD_equals_135 :
  area triangle_ABC = 36 →
  length A C = 8 →
  length C D = 30 →
  area triangle_BCD = 135 := by
  sorry

end area_BCD_equals_135_l3340_334084


namespace intersection_length_l3340_334054

/-- The length of segment AB is 8 when a line y = kx - k intersects 
    the parabola y² = 4x at points A and B, and the distance from 
    the midpoint of segment AB to the y-axis is 3 -/
theorem intersection_length (k : ℝ) (A B : ℝ × ℝ) : 
  (∃ (x y : ℝ), y = k * x - k ∧ y^2 = 4 * x) →  -- line intersects parabola
  (A.1 + B.1) / 2 = 3 →                         -- midpoint x-coordinate is 3
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    A = (x₁, y₁) ∧ 
    B = (x₂, y₂) ∧ 
    y₁ = k * x₁ - k ∧ 
    y₁^2 = 4 * x₁ ∧ 
    y₂ = k * x₂ - k ∧ 
    y₂^2 = 4 * x₂ ∧
    ((x₂ - x₁)^2 + (y₂ - y₁)^2).sqrt = 8 :=
by sorry

end intersection_length_l3340_334054


namespace function_value_proof_l3340_334018

/-- Given a function f(x) = ax^5 + bx^3 + cx + 8, prove that if f(-2) = 10, then f(2) = 6 -/
theorem function_value_proof (a b c : ℝ) : 
  let f : ℝ → ℝ := λ x => a * x^5 + b * x^3 + c * x + 8
  f (-2) = 10 → f 2 = 6 := by
sorry

end function_value_proof_l3340_334018


namespace total_subscription_is_50000_l3340_334043

/-- Represents the subscription amounts and profit distribution for a business venture. -/
structure BusinessSubscription where
  /-- Subscription amount of person C -/
  c_amount : ℕ
  /-- Total profit of the business -/
  total_profit : ℕ
  /-- Profit received by person C -/
  c_profit : ℕ

/-- Calculates the total subscription amount based on the given conditions -/
def total_subscription (bs : BusinessSubscription) : ℕ :=
  3 * bs.c_amount + 14000

/-- Theorem stating that the total subscription amount is 50,000 given the problem conditions -/
theorem total_subscription_is_50000 (bs : BusinessSubscription) 
  (h1 : bs.total_profit = 35000)
  (h2 : bs.c_profit = 8400)
  (h3 : bs.c_profit * (total_subscription bs) = bs.total_profit * bs.c_amount) :
  total_subscription bs = 50000 := by
  sorry

#eval total_subscription { c_amount := 12000, total_profit := 35000, c_profit := 8400 }

end total_subscription_is_50000_l3340_334043


namespace linear_function_proof_l3340_334064

def f (x : ℝ) := -3 * x + 5

theorem linear_function_proof :
  (∀ x y : ℝ, f y - f x = -3 * (y - x)) ∧ 
  (∃ y : ℝ, f 0 = 3 * 0 + 5 ∧ f 0 = y) := by
  sorry

end linear_function_proof_l3340_334064


namespace gcd_98_63_l3340_334044

theorem gcd_98_63 : Nat.gcd 98 63 = 7 := by
  sorry

end gcd_98_63_l3340_334044


namespace solve_equation_l3340_334086

theorem solve_equation (x y z a b c : ℤ) 
  (hx : x = -2272)
  (hy : y = 10^3 + 10^2 * c + 10 * b + a)
  (hz : z = 1)
  (heq : a * x + b * y + c * z = 1)
  (ha_pos : a > 0)
  (hb_pos : b > 0)
  (hc_pos : c > 0)
  (hab : a < b)
  (hbc : b < c) :
  y = 1987 := by
  sorry

end solve_equation_l3340_334086


namespace carl_typing_speed_l3340_334021

theorem carl_typing_speed :
  ∀ (hours_per_day : ℕ) (total_words : ℕ) (total_days : ℕ),
    hours_per_day = 4 →
    total_words = 84000 →
    total_days = 7 →
    (total_words / total_days) / (hours_per_day * 60) = 50 := by
  sorry

end carl_typing_speed_l3340_334021


namespace quadratic_two_real_roots_root_less_than_two_iff_l3340_334051

/-- The quadratic equation x^2 - (k+4)x + 4k = 0 -/
def quadratic (k x : ℝ) : ℝ := x^2 - (k+4)*x + 4*k

theorem quadratic_two_real_roots (k : ℝ) : 
  ∃ x y : ℝ, x ≠ y ∧ quadratic k x = 0 ∧ quadratic k y = 0 :=
sorry

theorem root_less_than_two_iff (k : ℝ) : 
  (∃ x : ℝ, quadratic k x = 0 ∧ x < 2) ↔ k < 2 :=
sorry

end quadratic_two_real_roots_root_less_than_two_iff_l3340_334051


namespace apple_basket_count_apple_basket_theorem_l3340_334035

theorem apple_basket_count : ℕ → Prop :=
  fun total_apples =>
    (total_apples : ℚ) * (12 : ℚ) / 100 + 66 = total_apples ∧ total_apples = 75

-- Proof
theorem apple_basket_theorem : ∃ n : ℕ, apple_basket_count n := by
  sorry

end apple_basket_count_apple_basket_theorem_l3340_334035


namespace intersection_point_satisfies_equations_l3340_334029

/-- The line equation in polar coordinates -/
def line_equation (ρ θ : ℝ) : Prop :=
  ρ * (Real.sqrt 3 * Real.cos θ - Real.sin θ) = 2

/-- The circle equation in polar coordinates -/
def circle_equation (ρ θ : ℝ) : Prop :=
  ρ = 4 * Real.sin θ

/-- The theorem stating that the point (2, π/6) satisfies both equations -/
theorem intersection_point_satisfies_equations :
  line_equation 2 (Real.pi / 6) ∧ circle_equation 2 (Real.pi / 6) := by
  sorry


end intersection_point_satisfies_equations_l3340_334029


namespace village_population_equality_l3340_334010

/-- The number of years it takes for the populations of two villages to be equal -/
def years_to_equal_population (x_initial : ℕ) (x_decrease : ℕ) (y_initial : ℕ) (y_increase : ℕ) : ℕ :=
  (x_initial - y_initial) / (x_decrease + y_increase)

theorem village_population_equality :
  years_to_equal_population 68000 1200 42000 800 = 13 := by
  sorry

end village_population_equality_l3340_334010


namespace inequality_implies_a_bound_l3340_334075

theorem inequality_implies_a_bound (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ioc 0 1 → |a * x^3 - Real.log x| ≥ 1) → a ≥ Real.exp 2 / 3 := by
  sorry

end inequality_implies_a_bound_l3340_334075


namespace boot_shoe_price_difference_l3340_334067

-- Define the price of shoes and boots as real numbers
variable (S B : ℝ)

-- Monday's sales equation
axiom monday_sales : 22 * S + 16 * B = 460

-- Tuesday's sales equation
axiom tuesday_sales : 8 * S + 32 * B = 560

-- Theorem stating the price difference between boots and shoes
theorem boot_shoe_price_difference : B - S = 5 := by sorry

end boot_shoe_price_difference_l3340_334067


namespace polynomial_roots_l3340_334085

theorem polynomial_roots (r : ℝ) : 
  r^2 = r + 1 → r^5 = 5*r + 3 ∧ ∀ b c : ℤ, (∀ s : ℝ, s^2 = s + 1 → s^5 = b*s + c) → b = 5 ∧ c = 3 := by
  sorry

end polynomial_roots_l3340_334085


namespace square_garden_perimeter_l3340_334048

theorem square_garden_perimeter (side : ℝ) (area perimeter : ℝ) : 
  area = side^2 → 
  perimeter = 4 * side → 
  area = 100 → 
  area = 2 * perimeter + 20 → 
  perimeter = 40 := by sorry

end square_garden_perimeter_l3340_334048


namespace odd_prime_properties_l3340_334050

theorem odd_prime_properties (p n : ℕ) (hp : Nat.Prime p) (hodd : Odd p) (hform : p = 4 * n + 1) :
  (∃ (x : ℕ), x ^ 2 % p = n % p) ∧ (n ^ n % p = 1) := by
  sorry

end odd_prime_properties_l3340_334050


namespace probability_red_then_blue_probability_red_then_blue_proof_l3340_334096

/-- The probability of drawing a red marble first and a blue marble second from a bag containing 
    4 red marbles and 6 blue marbles, when drawing two marbles sequentially without replacement. -/
theorem probability_red_then_blue (red : ℕ) (blue : ℕ) 
    (h_red : red = 4) (h_blue : blue = 6) : ℚ :=
  4 / 15

/-- Proof of the theorem -/
theorem probability_red_then_blue_proof (red : ℕ) (blue : ℕ) 
    (h_red : red = 4) (h_blue : blue = 6) : 
    probability_red_then_blue red blue h_red h_blue = 4 / 15 := by
  sorry

end probability_red_then_blue_probability_red_then_blue_proof_l3340_334096


namespace successfully_served_pizzas_l3340_334094

def pizzas_served : ℕ := 9
def pizzas_returned : ℕ := 6

theorem successfully_served_pizzas : 
  pizzas_served - pizzas_returned = 3 := by sorry

end successfully_served_pizzas_l3340_334094


namespace hemisphere_with_spire_surface_area_l3340_334049

/-- The total surface area of a hemisphere with a conical spire -/
theorem hemisphere_with_spire_surface_area :
  let r : ℝ := 8  -- radius of hemisphere
  let h : ℝ := 10 -- height of conical spire
  let l : ℝ := Real.sqrt (r^2 + h^2)  -- slant height of cone
  let area_base : ℝ := π * r^2  -- area of circular base
  let area_hemisphere : ℝ := 2 * π * r^2  -- surface area of hemisphere
  let area_cone : ℝ := π * r * l  -- lateral surface area of cone
  area_base + area_hemisphere + area_cone = 192 * π + 8 * π * Real.sqrt 164 :=
by sorry


end hemisphere_with_spire_surface_area_l3340_334049


namespace mark_leftover_money_l3340_334022

-- Define the given conditions
def old_hourly_wage : ℝ := 40
def raise_percentage : ℝ := 0.05
def hours_per_day : ℝ := 8
def days_per_week : ℝ := 5
def old_weekly_bills : ℝ := 600
def personal_trainer_cost : ℝ := 100

-- Define the calculation steps
def new_hourly_wage : ℝ := old_hourly_wage * (1 + raise_percentage)
def weekly_hours : ℝ := hours_per_day * days_per_week
def weekly_earnings : ℝ := new_hourly_wage * weekly_hours
def new_weekly_expenses : ℝ := old_weekly_bills + personal_trainer_cost

-- Theorem to prove
theorem mark_leftover_money :
  weekly_earnings - new_weekly_expenses = 980 := by
  sorry


end mark_leftover_money_l3340_334022


namespace cube_surface_area_l3340_334036

/-- Given a cube with volume 1728 cubic centimeters, its surface area is 864 square centimeters. -/
theorem cube_surface_area (volume : ℝ) (side : ℝ) :
  volume = 1728 →
  volume = side^3 →
  6 * side^2 = 864 :=
by sorry

end cube_surface_area_l3340_334036


namespace number_count_l3340_334024

theorem number_count (total_average : ℝ) (first_six_average : ℝ) (last_six_average : ℝ) (middle_number : ℝ) :
  total_average = 9.9 →
  first_six_average = 10.5 →
  last_six_average = 11.4 →
  middle_number = 22.5 →
  ∃ (n : ℕ), n = 11 ∧ n % 2 = 1 ∧
  n * total_average = 6 * first_six_average + 6 * last_six_average - middle_number :=
by sorry


end number_count_l3340_334024


namespace smallest_benches_proof_l3340_334072

/-- The number of adults that can sit on one bench -/
def adults_per_bench : ℕ := 7

/-- The number of children that can sit on one bench -/
def children_per_bench : ℕ := 11

/-- A function that returns true if the given number of benches can seat an equal number of adults and children -/
def can_seat_equally (n : ℕ) : Prop :=
  ∃ (people : ℕ), people > 0 ∧ 
    n * adults_per_bench = people ∧
    n * children_per_bench = people

/-- The smallest number of benches that can seat an equal number of adults and children -/
def smallest_n : ℕ := 18

theorem smallest_benches_proof :
  (∀ m : ℕ, m > 0 → m < smallest_n → ¬(can_seat_equally m)) ∧
  can_seat_equally smallest_n :=
sorry

end smallest_benches_proof_l3340_334072


namespace distribute_five_to_three_l3340_334069

/-- The number of ways to distribute n distinct objects into k distinct boxes,
    with each box containing at least one object. -/
def distributeObjects (n k : ℕ) : ℕ :=
  sorry

theorem distribute_five_to_three :
  distributeObjects 5 3 = 150 := by sorry

end distribute_five_to_three_l3340_334069


namespace sum_of_three_numbers_l3340_334062

theorem sum_of_three_numbers : 6 + 8 + 11 = 25 := by
  sorry

end sum_of_three_numbers_l3340_334062


namespace perimeter_of_cut_square_perimeter_of_specific_cut_square_l3340_334076

/-- The perimeter of a figure formed by cutting a square into two equal rectangles and placing them side by side -/
theorem perimeter_of_cut_square (side_length : ℝ) : 
  side_length > 0 → 
  (3 * side_length + 4 * (side_length / 2)) = 5 * side_length := by
  sorry

/-- The perimeter of a figure formed by cutting a square with side length 100 into two equal rectangles and placing them side by side is 500 -/
theorem perimeter_of_specific_cut_square : 
  (3 * 100 + 4 * (100 / 2)) = 500 := by
  sorry

end perimeter_of_cut_square_perimeter_of_specific_cut_square_l3340_334076


namespace rain_probability_l3340_334088

theorem rain_probability (umbrellas : ℕ) (take_umbrella_prob : ℝ) :
  umbrellas = 2 →
  take_umbrella_prob = 0.2 →
  ∃ (rain_prob : ℝ),
    rain_prob + (rain_prob / (rain_prob + 1)) - (rain_prob^2 / (rain_prob + 1)) = take_umbrella_prob ∧
    rain_prob = 1/9 := by
  sorry

end rain_probability_l3340_334088


namespace complex_magnitude_equation_solution_l3340_334017

theorem complex_magnitude_equation_solution :
  ∃ x : ℝ, x > 0 ∧ 
  Complex.abs (x + Complex.I * Real.sqrt 7) * Complex.abs (3 - 2 * Complex.I * Real.sqrt 5) = 45 ∧
  x = Real.sqrt (1822 / 29) :=
sorry

end complex_magnitude_equation_solution_l3340_334017


namespace emily_coloring_books_l3340_334012

theorem emily_coloring_books (x : ℕ) : 
  x - 2 + 14 = 19 → x = 7 := by
sorry

end emily_coloring_books_l3340_334012


namespace solution_set_characterization_l3340_334037

def is_solution_set (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, x ∈ S ↔ 2^(1 + f x) + 2^(1 - f x) + 2 * f (x^2) ≤ 7

theorem solution_set_characterization
  (f : ℝ → ℝ)
  (h1 : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
  (h2 : ∀ x, x > 0 → f x > 0)
  (h3 : f 1 = 1) :
  is_solution_set f (Set.Icc (-1) 1) :=
sorry

end solution_set_characterization_l3340_334037


namespace dads_contribution_undetermined_l3340_334015

/-- Represents the number of toy cars in Olaf's collection --/
structure ToyCarCollection where
  initial : ℕ
  fromUncle : ℕ
  fromGrandpa : ℕ
  fromAuntie : ℕ
  fromMum : ℕ
  fromDad : ℕ
  final : ℕ

/-- The conditions of Olaf's toy car collection --/
def olafCollection : ToyCarCollection where
  initial := 150
  fromUncle := 5
  fromGrandpa := 10
  fromAuntie := 6
  fromMum := 0  -- Unknown value
  fromDad := 0  -- Unknown value
  final := 196

/-- Theorem stating that Dad's contribution is undetermined --/
theorem dads_contribution_undetermined (c : ToyCarCollection) 
  (h1 : c.initial = 150)
  (h2 : c.fromGrandpa = 2 * c.fromUncle)
  (h3 : c.fromAuntie = c.fromUncle + 1)
  (h4 : c.final = 196)
  (h5 : c.final = c.initial + c.fromUncle + c.fromGrandpa + c.fromAuntie + c.fromMum + c.fromDad) :
  ∃ (x y : ℕ), x ≠ y ∧ 
    (c.fromMum = x ∧ c.fromDad = 25 - x) ∧
    (c.fromMum = y ∧ c.fromDad = 25 - y) :=
sorry

#check dads_contribution_undetermined

end dads_contribution_undetermined_l3340_334015


namespace baseball_league_games_l3340_334059

/-- The number of games played in a baseball league --/
def total_games (n : ℕ) (g : ℕ) : ℕ :=
  n * (n - 1) * g / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with every other team,
    the total number of games played is 180. --/
theorem baseball_league_games :
  total_games 10 4 = 180 := by
  sorry

end baseball_league_games_l3340_334059


namespace circle_center_in_second_quadrant_l3340_334055

/-- A line passing through the second, third, and fourth quadrants -/
structure Line where
  a : ℝ
  b : ℝ
  second_quadrant : a < 0 ∧ 0 < a * 0 - b
  third_quadrant : a * (-1) - b < 0
  fourth_quadrant : 0 < a * 1 - b

/-- The center of a circle (x-a)^2 + (y-b)^2 = 1 -/
def circle_center (l : Line) : ℝ × ℝ := (l.a, l.b)

/-- A point is in the second quadrant if its x-coordinate is negative and y-coordinate is positive -/
def in_second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ 0 < p.2

theorem circle_center_in_second_quadrant (l : Line) :
  in_second_quadrant (circle_center l) := by sorry

end circle_center_in_second_quadrant_l3340_334055


namespace min_value_at_three_l3340_334047

/-- The function f(y) = 3y^2 - 18y + 7 -/
def f (y : ℝ) : ℝ := 3 * y^2 - 18 * y + 7

/-- Theorem stating that the minimum value of f occurs when y = 3 -/
theorem min_value_at_three :
  ∃ (y_min : ℝ), ∀ (y : ℝ), f y ≥ f y_min ∧ y_min = 3 :=
sorry

end min_value_at_three_l3340_334047


namespace cube_sum_one_l3340_334006

theorem cube_sum_one (a b c : ℝ) 
  (sum_one : a + b + c = 1)
  (sum_products : a * b + a * c + b * c = -5)
  (product : a * b * c = 5) :
  a^3 + b^3 + c^3 = 1 := by sorry

end cube_sum_one_l3340_334006


namespace brenda_friends_count_l3340_334071

/-- Prove that Brenda has 9 friends given the pizza ordering scenario -/
theorem brenda_friends_count :
  let slices_per_person : ℕ := 2
  let slices_per_pizza : ℕ := 4
  let pizzas_ordered : ℕ := 5
  let total_slices : ℕ := slices_per_pizza * pizzas_ordered
  let total_people : ℕ := total_slices / slices_per_person
  let brenda_friends : ℕ := total_people - 1
  brenda_friends = 9 := by
  sorry

end brenda_friends_count_l3340_334071


namespace friends_decks_count_l3340_334046

/-- The number of decks Victor's friend bought -/
def friends_decks : ℕ := 2

/-- The cost of each deck in dollars -/
def deck_cost : ℕ := 8

/-- The number of decks Victor bought -/
def victors_decks : ℕ := 6

/-- The total amount spent by both Victor and his friend in dollars -/
def total_spent : ℕ := 64

theorem friends_decks_count : 
  deck_cost * (victors_decks + friends_decks) = total_spent := by sorry

end friends_decks_count_l3340_334046


namespace complex_equation_result_l3340_334020

theorem complex_equation_result (m n : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : m / (1 + i) = 1 - n * i) : 
  m - n = 1 := by
sorry

end complex_equation_result_l3340_334020


namespace pin_purchase_cost_l3340_334025

theorem pin_purchase_cost (num_pins : ℕ) (original_price : ℚ) (discount_percent : ℚ) :
  num_pins = 10 →
  original_price = 20 →
  discount_percent = 15 / 100 →
  (num_pins : ℚ) * (original_price * (1 - discount_percent)) = 170 :=
by sorry

end pin_purchase_cost_l3340_334025


namespace equation1_solutions_equation2_solutions_l3340_334032

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x * (x + 3) = 2 * (x + 3)
def equation2 (x : ℝ) : Prop := x^2 - 4*x - 5 = 0

-- Theorem for equation 1
theorem equation1_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = -3 ∧ x₂ = 2/3 ∧ equation1 x₁ ∧ equation1 x₂ ∧
  ∀ (x : ℝ), equation1 x → x = x₁ ∨ x = x₂ := by sorry

-- Theorem for equation 2
theorem equation2_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = 5 ∧ x₂ = -1 ∧ equation2 x₁ ∧ equation2 x₂ ∧
  ∀ (x : ℝ), equation2 x → x = x₁ ∨ x = x₂ := by sorry

end equation1_solutions_equation2_solutions_l3340_334032


namespace motorboat_travel_time_l3340_334014

/-- Represents the scenario of a motorboat and kayak traveling on a river --/
structure RiverTrip where
  r : ℝ  -- River current speed (also kayak speed)
  p : ℝ  -- Motorboat speed relative to the river
  t : ℝ  -- Time for motorboat to travel from X to Y

/-- The conditions of the river trip --/
def trip_conditions (trip : RiverTrip) : Prop :=
  trip.p > 0 ∧ 
  trip.r > 0 ∧ 
  trip.t > 0 ∧ 
  (trip.p + trip.r) * trip.t + (trip.p - trip.r) * (11 - trip.t) = 12 * trip.r

/-- The theorem stating that under the given conditions, 
    the motorboat's initial travel time from X to Y is 4 hours --/
theorem motorboat_travel_time (trip : RiverTrip) : 
  trip_conditions trip → trip.t = 4 := by
  sorry

end motorboat_travel_time_l3340_334014


namespace fraction_subtraction_equality_l3340_334001

theorem fraction_subtraction_equality : 
  (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := by
  sorry

end fraction_subtraction_equality_l3340_334001


namespace sin_cos_value_l3340_334098

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem sin_cos_value (θ : ℝ) :
  determinant (Real.sin θ) 2 (Real.cos θ) 3 = 0 →
  2 * (Real.sin θ)^2 + (Real.sin θ) * (Real.cos θ) = 14/13 :=
by
  sorry

end sin_cos_value_l3340_334098


namespace charity_draw_winnings_calculation_l3340_334058

/-- Calculates the charity draw winnings given initial amount, expenses, lottery winnings, and final amount -/
def charity_draw_winnings (initial_amount expenses lottery_winnings final_amount : ℕ) : ℕ :=
  final_amount - (initial_amount - expenses + lottery_winnings)

/-- Theorem stating that given the specific values from the problem, the charity draw winnings must be 19 -/
theorem charity_draw_winnings_calculation :
  charity_draw_winnings 10 (4 + 1 + 1) 65 94 = 19 := by
  sorry

end charity_draw_winnings_calculation_l3340_334058


namespace function_inequality_l3340_334011

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, (x - 3) * deriv f x ≤ 0) : 
  f 0 + f 6 ≤ 2 * f 3 := by
  sorry

end function_inequality_l3340_334011


namespace quadratic_equation_m_value_l3340_334081

theorem quadratic_equation_m_value : 
  ∃! m : ℝ, m^2 + 1 = 2 ∧ m - 1 ≠ 0 :=
by
  sorry

end quadratic_equation_m_value_l3340_334081


namespace jose_peanuts_l3340_334030

theorem jose_peanuts (kenya_peanuts : ℕ) (difference : ℕ) (h1 : kenya_peanuts = 133) (h2 : difference = 48) :
  kenya_peanuts - difference = 85 := by
  sorry

end jose_peanuts_l3340_334030


namespace min_value_range_lower_bound_value_l3340_334057

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + a * |x - 2|

-- Theorem for part I
theorem min_value_range (a : ℝ) :
  (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) → -1 ≤ a ∧ a ≤ 1 :=
by sorry

-- Theorem for part II
theorem lower_bound_value (a : ℝ) :
  (∀ (x : ℝ), f a x ≥ 1/2) → a = 1/3 :=
by sorry

end min_value_range_lower_bound_value_l3340_334057


namespace batsman_new_average_l3340_334023

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  latestScore : Nat
  averageIncrease : Nat

/-- Calculates the average score after the latest innings -/
def calculateNewAverage (stats : BatsmanStats) : Nat :=
  (stats.totalRuns + stats.latestScore) / stats.innings

/-- Theorem: Given the conditions, the batsman's new average is 43 -/
theorem batsman_new_average (stats : BatsmanStats) 
  (h1 : stats.innings = 12)
  (h2 : stats.latestScore = 65)
  (h3 : stats.averageIncrease = 2)
  (h4 : calculateNewAverage stats = (calculateNewAverage stats - stats.averageIncrease) + stats.averageIncrease) :
  calculateNewAverage stats = 43 := by
  sorry

#eval calculateNewAverage { innings := 12, totalRuns := 451, latestScore := 65, averageIncrease := 2 }

end batsman_new_average_l3340_334023


namespace parabola_intercepts_sum_l3340_334056

/-- The parabola equation -/
def parabola (y : ℝ) : ℝ := 3 * y^2 - 9 * y + 4

/-- The x-intercept of the parabola -/
def x_intercept : ℝ := parabola 0

/-- The y-intercepts of the parabola -/
noncomputable def y_intercepts : Set ℝ := {y | parabola y = 0}

theorem parabola_intercepts_sum :
  ∃ (b c : ℝ), b ∈ y_intercepts ∧ c ∈ y_intercepts ∧ b ≠ c ∧ x_intercept + b + c = 7 := by
  sorry

end parabola_intercepts_sum_l3340_334056


namespace smallest_positive_t_l3340_334068

theorem smallest_positive_t (x₁ x₂ x₃ x₄ x₅ t : ℝ) : 
  (x₁ ≥ 0 ∧ x₂ ≥ 0 ∧ x₃ ≥ 0 ∧ x₄ ≥ 0 ∧ x₅ ≥ 0) →
  (x₁ + x₂ + x₃ + x₄ + x₅ > 0) →
  (x₁ + x₃ = 2 * t * x₂) →
  (x₂ + x₄ = 2 * t * x₃) →
  (x₃ + x₅ = 2 * t * x₄) →
  t > 0 →
  t ≥ 1 / Real.sqrt 2 :=
by sorry

end smallest_positive_t_l3340_334068


namespace ratio_of_segments_on_line_l3340_334097

/-- Given four points P, Q, R, S on a line in that order, with given distances between them,
    prove that the ratio of PR to QS is 7/12. -/
theorem ratio_of_segments_on_line (P Q R S : ℝ) (h_order : P < Q ∧ Q < R ∧ R < S)
    (h_PQ : Q - P = 4) (h_QR : R - Q = 10) (h_PS : S - P = 28) :
    (R - P) / (S - Q) = 7 / 12 := by
  sorry

end ratio_of_segments_on_line_l3340_334097


namespace greatest_possible_median_l3340_334060

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 18 →
  k < m → m < r → r < s → s < t →
  t = 40 →
  r ≤ 23 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 40) / 5 = 18 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 40 ∧
    r' = 23 :=
by sorry

end greatest_possible_median_l3340_334060


namespace product_without_x2_x3_implies_p_plus_q_eq_neg_four_l3340_334074

theorem product_without_x2_x3_implies_p_plus_q_eq_neg_four (p q : ℝ) :
  (∀ x : ℝ, (x^2 + p) * (x^2 - q*x + 4) = x^4 + (-p*q)*x + 4*p) →
  p + q = -4 := by
  sorry

end product_without_x2_x3_implies_p_plus_q_eq_neg_four_l3340_334074


namespace point_P_y_coordinate_l3340_334092

theorem point_P_y_coordinate :
  ∀ (x y : ℝ),
  (|y| = (1/2) * |x|) →  -- Distance from x-axis is half the distance from y-axis
  (|x| = 18) →           -- Point P is 18 units from y-axis
  y = 9 := by            -- The y-coordinate of point P is 9
sorry

end point_P_y_coordinate_l3340_334092


namespace base_prime_630_l3340_334039

/-- Base prime representation of a natural number -/
def BasePrime : ℕ → List ℕ := sorry

/-- Check if a list represents a valid base prime representation -/
def IsValidBasePrime : List ℕ → Prop := sorry

theorem base_prime_630 : 
  let bp := BasePrime 630
  IsValidBasePrime bp ∧ bp = [2, 1, 1, 0] := by sorry

end base_prime_630_l3340_334039


namespace min_mn_value_l3340_334079

theorem min_mn_value (m : ℝ) (n : ℝ) (h_m : m > 0) 
  (h_ineq : ∀ x : ℝ, x > -m → x + m ≤ Real.exp ((2 * x / m) + n)) :
  ∃ (min_mn : ℝ), min_mn = -2 / Real.exp 2 ∧ 
    ∀ (m' n' : ℝ), (∀ x : ℝ, x > -m' → x + m' ≤ Real.exp ((2 * x / m') + n')) → 
      m' * n' ≥ min_mn :=
sorry

end min_mn_value_l3340_334079


namespace average_equation_l3340_334040

theorem average_equation (a : ℝ) : 
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74 → a = 28 := by sorry

end average_equation_l3340_334040


namespace hexagon_angle_measure_l3340_334093

/-- A hexagon with vertices N, U, M, B, E, S -/
structure Hexagon :=
  (N U M B E S : ℝ)

/-- The property that three angles are congruent -/
def three_angles_congruent (h : Hexagon) : Prop :=
  h.N = h.M ∧ h.M = h.B

/-- The property that two angles are supplementary -/
def supplementary (a b : ℝ) : Prop :=
  a + b = 180

/-- The theorem stating that in a hexagon NUMBERS where ∠N ≅ ∠M ≅ ∠B 
    and ∠U is supplementary to ∠S, the measure of ∠B is 135° -/
theorem hexagon_angle_measure (h : Hexagon) 
  (h_congruent : three_angles_congruent h)
  (h_supplementary : supplementary h.U h.S) :
  h.B = 135 :=
sorry

end hexagon_angle_measure_l3340_334093


namespace roots_sum_of_reciprocal_cubes_l3340_334003

theorem roots_sum_of_reciprocal_cubes (a b c : ℝ) (r s : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : c ≠ 0) 
  (h3 : a * r^2 + b * r + c = 0) 
  (h4 : a * s^2 + b * s + c = 0) 
  (h5 : r ≠ 0) 
  (h6 : s ≠ 0) : 
  1 / r^3 + 1 / s^3 = (-b^3 + 3*a*b*c) / c^3 := by
  sorry

end roots_sum_of_reciprocal_cubes_l3340_334003


namespace remainder_2519_div_8_l3340_334002

theorem remainder_2519_div_8 : 2519 % 8 = 7 := by
  sorry

end remainder_2519_div_8_l3340_334002


namespace min_value_quadratic_plus_constant_l3340_334052

theorem min_value_quadratic_plus_constant :
  (∀ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 ≥ 2018) ∧
  (∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018) :=
by sorry

end min_value_quadratic_plus_constant_l3340_334052


namespace fraction_multiplication_l3340_334031

theorem fraction_multiplication : (2 : ℚ) / 5 * 5 / 7 * 7 / 3 * 3 / 8 = 1 / 4 := by
  sorry

end fraction_multiplication_l3340_334031


namespace quartic_root_sum_l3340_334041

theorem quartic_root_sum (a b c d : ℂ) : 
  (a^4 - 34*a^3 + 15*a^2 - 42*a - 8 = 0) →
  (b^4 - 34*b^3 + 15*b^2 - 42*b - 8 = 0) →
  (c^4 - 34*c^3 + 15*c^2 - 42*c - 8 = 0) →
  (d^4 - 34*d^3 + 15*d^2 - 42*d - 8 = 0) →
  (a / ((1/a) + b*c*d) + b / ((1/b) + a*c*d) + c / ((1/c) + a*b*d) + d / ((1/d) + a*b*c) = -161) :=
by sorry

end quartic_root_sum_l3340_334041


namespace polynomial_division_remainder_l3340_334034

theorem polynomial_division_remainder (x : ℝ) : 
  ∃ (Q : ℝ → ℝ), x^150 = (x^2 - 4*x + 3) * Q x + ((3^150 - 1)*x + (4 - 3^150)) / 2 :=
by sorry

end polynomial_division_remainder_l3340_334034


namespace chef_michel_pies_sold_l3340_334080

theorem chef_michel_pies_sold 
  (shepherds_pie_pieces : ℕ)
  (chicken_pot_pie_pieces : ℕ)
  (shepherds_pie_customers : ℕ)
  (chicken_pot_pie_customers : ℕ)
  (h1 : shepherds_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherds_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  shepherds_pie_customers / shepherds_pie_pieces + 
  chicken_pot_pie_customers / chicken_pot_pie_pieces = 29 :=
by sorry

end chef_michel_pies_sold_l3340_334080


namespace first_year_selection_probability_l3340_334007

/-- Represents the number of students in each year --/
structure StudentCounts where
  firstYear : ℕ
  secondYear : ℕ
  thirdYear : ℕ

/-- Represents the sampling information --/
structure SamplingInfo where
  thirdYearSample : ℕ

/-- Calculates the probability of a student being selected in stratified sampling --/
def stratifiedSamplingProbability (counts : StudentCounts) (info : SamplingInfo) : ℚ :=
  info.thirdYearSample / counts.thirdYear

/-- Theorem stating the probability of a first-year student being selected --/
theorem first_year_selection_probability 
  (counts : StudentCounts) 
  (info : SamplingInfo) 
  (h1 : counts.firstYear = 800)
  (h2 : counts.thirdYear = 500)
  (h3 : info.thirdYearSample = 25) :
  stratifiedSamplingProbability counts info = 1 / 20 := by
  sorry


end first_year_selection_probability_l3340_334007


namespace total_selling_price_calculation_l3340_334027

def calculate_total_selling_price (item1_cost item2_cost item3_cost : ℚ)
  (loss1 loss2 loss3 tax_rate : ℚ) (overhead : ℚ) : ℚ :=
  let total_purchase := item1_cost + item2_cost + item3_cost
  let tax := tax_rate * total_purchase
  let selling_price1 := item1_cost * (1 - loss1)
  let selling_price2 := item2_cost * (1 - loss2)
  let selling_price3 := item3_cost * (1 - loss3)
  let total_selling := selling_price1 + selling_price2 + selling_price3
  total_selling + overhead + tax

theorem total_selling_price_calculation :
  calculate_total_selling_price 750 1200 500 0.1 0.15 0.05 0.05 300 = 2592.5 := by
  sorry

end total_selling_price_calculation_l3340_334027


namespace dallas_pears_count_l3340_334099

/-- The number of bags of pears Dallas picked -/
def dallas_pears : ℕ := 9

/-- The number of bags of apples Dallas picked -/
def dallas_apples : ℕ := 14

/-- The total number of bags Austin picked -/
def austin_total : ℕ := 24

theorem dallas_pears_count :
  dallas_pears = 9 ∧
  dallas_apples = 14 ∧
  austin_total = 24 ∧
  austin_total = (dallas_apples + 6) + (dallas_pears - 5) :=
by sorry

end dallas_pears_count_l3340_334099


namespace packet_weight_l3340_334009

-- Define constants
def pounds_per_ton : ℚ := 2200
def ounces_per_pound : ℚ := 16
def bag_capacity_tons : ℚ := 13
def num_packets : ℚ := 1760

-- Define the theorem
theorem packet_weight :
  let total_weight := bag_capacity_tons * pounds_per_ton
  let weight_per_packet := total_weight / num_packets
  weight_per_packet = 16.25 := by
sorry

end packet_weight_l3340_334009
