import Mathlib

namespace NUMINAMATH_CALUDE_dunk_a_clown_tickets_l4061_406191

def total_tickets : ℕ := 40
def num_rides : ℕ := 3
def tickets_per_ride : ℕ := 4

theorem dunk_a_clown_tickets : 
  total_tickets - (num_rides * tickets_per_ride) = 28 := by
  sorry

end NUMINAMATH_CALUDE_dunk_a_clown_tickets_l4061_406191


namespace NUMINAMATH_CALUDE_wife_speed_l4061_406177

theorem wife_speed (track_circumference : ℝ) (meet_time : ℝ) (suresh_speed : ℝ) :
  track_circumference = 726 →
  meet_time = 5.28 →
  suresh_speed = 4.5 →
  let suresh_distance := suresh_speed * 1000 / 60 * meet_time
  let wife_distance := track_circumference - suresh_distance
  let wife_speed := wife_distance / meet_time * 60 / 1000
  wife_speed = 3.75 := by sorry

end NUMINAMATH_CALUDE_wife_speed_l4061_406177


namespace NUMINAMATH_CALUDE_right_triangle_circle_theorem_l4061_406104

/-- Given a right triangle with legs a and b, hypotenuse c, and a circle of radius ρb
    touching leg b externally and extending to the other sides, prove that b + c = a + 2ρb -/
theorem right_triangle_circle_theorem
  (a b c ρb : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ ρb > 0)
  (circle_property : ∃ (x y : ℝ), x^2 + y^2 = ρb^2 ∧ x + y = b ∧ (a - x)^2 + y^2 = c^2) :
  b + c = a + 2*ρb :=
sorry

end NUMINAMATH_CALUDE_right_triangle_circle_theorem_l4061_406104


namespace NUMINAMATH_CALUDE_expression_evaluation_l4061_406117

theorem expression_evaluation : 
  let f (x : ℚ) := (2 * x + 1) / (2 * x - 1)
  f 2 = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4061_406117


namespace NUMINAMATH_CALUDE_max_value_complex_expression_l4061_406165

theorem max_value_complex_expression (z : ℂ) (h : Complex.abs z = Real.sqrt 3) :
  Complex.abs ((z - 1)^3 * (z + 1)) ≤ 12 * Real.sqrt 3 ∧
  ∃ w : ℂ, Complex.abs w = Real.sqrt 3 ∧ Complex.abs ((w - 1)^3 * (w + 1)) = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_complex_expression_l4061_406165


namespace NUMINAMATH_CALUDE_marys_number_proof_l4061_406123

theorem marys_number_proof : ∃! x : ℕ, 
  10 ≤ x ∧ x < 100 ∧ 
  (∃ a b : ℕ, 
    3 * x + 11 = 10 * a + b ∧
    10 * b + a ≥ 71 ∧ 
    10 * b + a ≤ 75) ∧
  x = 12 := by
sorry

end NUMINAMATH_CALUDE_marys_number_proof_l4061_406123


namespace NUMINAMATH_CALUDE_periodic_sum_implies_periodic_increasing_sum_not_implies_increasing_l4061_406149

def PeriodicFunction (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def IncreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

theorem periodic_sum_implies_periodic
  (f g h : ℝ → ℝ) (T : ℝ) :
  (PeriodicFunction (fun x ↦ f x + g x) T) →
  (PeriodicFunction (fun x ↦ f x + h x) T) →
  (PeriodicFunction (fun x ↦ g x + h x) T) →
  (PeriodicFunction f T) ∧ (PeriodicFunction g T) ∧ (PeriodicFunction h T) := by
  sorry

theorem increasing_sum_not_implies_increasing :
  ∃ f g h : ℝ → ℝ,
    (IncreasingFunction (fun x ↦ f x + g x)) ∧
    (IncreasingFunction (fun x ↦ f x + h x)) ∧
    (IncreasingFunction (fun x ↦ g x + h x)) ∧
    (¬IncreasingFunction f ∨ ¬IncreasingFunction g ∨ ¬IncreasingFunction h) := by
  sorry

end NUMINAMATH_CALUDE_periodic_sum_implies_periodic_increasing_sum_not_implies_increasing_l4061_406149


namespace NUMINAMATH_CALUDE_sum_of_coordinates_reflection_l4061_406166

/-- Given a point C with coordinates (3, y) and its reflection D over the x-axis,
    the sum of all coordinates of C and D is 6. -/
theorem sum_of_coordinates_reflection (y : ℝ) : 
  let C : ℝ × ℝ := (3, y)
  let D : ℝ × ℝ := (3, -y)  -- reflection of C over x-axis
  C.1 + C.2 + D.1 + D.2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_reflection_l4061_406166


namespace NUMINAMATH_CALUDE_max_value_of_sqrt_sum_l4061_406140

theorem max_value_of_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 5 → Real.sqrt (x + 1) + Real.sqrt (y + 3) ≤ 3 * Real.sqrt 2) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 5 ∧ Real.sqrt (x + 1) + Real.sqrt (y + 3) = 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sqrt_sum_l4061_406140


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4061_406128

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, -6]

theorem parallel_vectors_x_value :
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b x) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4061_406128


namespace NUMINAMATH_CALUDE_multiple_of_numbers_l4061_406121

theorem multiple_of_numbers (s l k : ℤ) : 
  s = 18 →                  -- The smaller number is 18
  l = k * s - 3 →           -- One number is 3 less than a multiple of the other
  s + l = 51 →              -- The sum of the two numbers is 51
  k = 2 :=                  -- The multiple is 2
by sorry

end NUMINAMATH_CALUDE_multiple_of_numbers_l4061_406121


namespace NUMINAMATH_CALUDE_smallest_bob_number_l4061_406175

def alice_number : ℕ := 36

def is_twice_prime (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = 2 * p

def has_only_factors_of (n m : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → p ∣ m

theorem smallest_bob_number :
  ∃ n : ℕ, 
    n > 0 ∧
    is_twice_prime n ∧
    has_only_factors_of n alice_number ∧
    (∀ m : ℕ, m > 0 → is_twice_prime m → has_only_factors_of m alice_number → n ≤ m) ∧
    n = 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l4061_406175


namespace NUMINAMATH_CALUDE_sculpture_surface_area_l4061_406159

/-- Represents a cube sculpture with three layers --/
structure CubeSculpture where
  topLayer : ℕ
  middleLayer : ℕ
  bottomLayer : ℕ
  totalCubes : ℕ
  (total_eq : totalCubes = topLayer + middleLayer + bottomLayer)

/-- Calculates the exposed surface area of the sculpture --/
def exposedSurfaceArea (s : CubeSculpture) : ℕ :=
  5 * s.topLayer +  -- Top cube: 5 sides exposed
  (5 + 4 * 4) +     -- Middle layer: 1 cube with 5 sides, 4 cubes with 4 sides
  s.bottomLayer     -- Bottom layer: only top faces exposed

/-- The main theorem to prove --/
theorem sculpture_surface_area :
  ∃ (s : CubeSculpture),
    s.topLayer = 1 ∧
    s.middleLayer = 5 ∧
    s.bottomLayer = 11 ∧
    s.totalCubes = 17 ∧
    exposedSurfaceArea s = 37 :=
  sorry

end NUMINAMATH_CALUDE_sculpture_surface_area_l4061_406159


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l4061_406116

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l4061_406116


namespace NUMINAMATH_CALUDE_remaining_movies_to_watch_l4061_406156

theorem remaining_movies_to_watch (total_movies watched_movies : ℕ) : 
  total_movies = 8 → watched_movies = 4 → total_movies - watched_movies = 4 :=
by sorry

end NUMINAMATH_CALUDE_remaining_movies_to_watch_l4061_406156


namespace NUMINAMATH_CALUDE_sarah_apples_left_l4061_406132

def apples_left (initial : ℕ) (teachers : ℕ) (friends : ℕ) (eaten : ℕ) : ℕ :=
  initial - (teachers + friends + eaten)

theorem sarah_apples_left :
  apples_left 25 16 5 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sarah_apples_left_l4061_406132


namespace NUMINAMATH_CALUDE_remainder_of_sum_mod_11_l4061_406135

theorem remainder_of_sum_mod_11 : (100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_sum_mod_11_l4061_406135


namespace NUMINAMATH_CALUDE_intersection_equals_target_l4061_406147

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (x - 1) < 0}
def N : Set ℝ := {x | 2 * x^2 - 3 * x ≤ 0}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- Define the open-closed interval (1, 3/2]
def target_set : Set ℝ := {x | 1 < x ∧ x ≤ 3/2}

-- Theorem statement
theorem intersection_equals_target : M_intersect_N = target_set := by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_target_l4061_406147


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l4061_406107

theorem sqrt_product_equality : Real.sqrt 121 * Real.sqrt 49 * Real.sqrt 11 = 77 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l4061_406107


namespace NUMINAMATH_CALUDE_anns_age_is_30_l4061_406150

/-- Represents the ages of Ann and Barbara at different points in time. -/
structure AgeRelation where
  a : ℕ  -- Ann's current age
  b : ℕ  -- Barbara's current age

/-- The condition that the sum of their present ages is 50 years. -/
def sum_of_ages (ages : AgeRelation) : Prop :=
  ages.a + ages.b = 50

/-- The complex age relation described in the problem. -/
def age_relation (ages : AgeRelation) : Prop :=
  ∃ (y : ℕ), 
    ages.b = ages.a / 2 + 2 * y ∧
    ages.a - ages.b = y

/-- The theorem stating that given the conditions, Ann's age is 30 years. -/
theorem anns_age_is_30 (ages : AgeRelation) 
  (h1 : sum_of_ages ages) 
  (h2 : age_relation ages) : 
  ages.a = 30 := by sorry

end NUMINAMATH_CALUDE_anns_age_is_30_l4061_406150


namespace NUMINAMATH_CALUDE_power_equality_l4061_406187

theorem power_equality (m n : ℕ) (h1 : 2^m = 5) (h2 : 4^n = 3) : 4^(3*n - m) = 27/25 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l4061_406187


namespace NUMINAMATH_CALUDE_chess_game_probability_l4061_406161

theorem chess_game_probability (p_not_lose p_draw : ℝ) 
  (h1 : p_not_lose = 0.8) 
  (h2 : p_draw = 0.5) : 
  p_not_lose - p_draw = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_probability_l4061_406161


namespace NUMINAMATH_CALUDE_sum_of_possible_N_values_l4061_406183

-- Define the set of expressions
def S (x y : ℝ) : Set ℝ := {(x + y)^2, (x - y)^2, x * y, x / y}

-- Define the given set of values
def T (N : ℝ) : Set ℝ := {4, 12.8, 28.8, N}

-- Theorem statement
theorem sum_of_possible_N_values (x y N : ℝ) (hy : y ≠ 0) 
  (h_equal : S x y = T N) : 
  ∃ (N₁ N₂ N₃ : ℝ), 
    (S x y = T N₁) ∧ 
    (S x y = T N₂) ∧ 
    (S x y = T N₃) ∧ 
    N₁ + N₂ + N₃ = 85.2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_possible_N_values_l4061_406183


namespace NUMINAMATH_CALUDE_sons_age_l4061_406190

/-- Given a father and son with specific age relationships, prove the son's age -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 25 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 23 := by
  sorry

end NUMINAMATH_CALUDE_sons_age_l4061_406190


namespace NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_pow6_l4061_406196

theorem nearest_integer_to_3_plus_sqrt2_pow6 :
  ∃ n : ℤ, n = 7414 ∧ ∀ m : ℤ, |((3 : ℝ) + Real.sqrt 2)^6 - (n : ℝ)| ≤ |((3 : ℝ) + Real.sqrt 2)^6 - (m : ℝ)| :=
sorry

end NUMINAMATH_CALUDE_nearest_integer_to_3_plus_sqrt2_pow6_l4061_406196


namespace NUMINAMATH_CALUDE_power_product_equality_l4061_406174

theorem power_product_equality (a b : ℝ) : 3 * a^2 * b * (-a)^2 = 3 * a^4 * b := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l4061_406174


namespace NUMINAMATH_CALUDE_min_odd_in_A_P_l4061_406139

/-- A polynomial of degree 8 -/
def Polynomial8 : Type := ℝ → ℝ

/-- The set A_P for a polynomial P -/
def A_P (P : Polynomial8) : Set ℝ := {x : ℝ | ∃ c : ℝ, P x = c}

/-- Statement: If 8 is in A_P, then A_P contains at least one odd number -/
theorem min_odd_in_A_P (P : Polynomial8) (h : 8 ∈ A_P P) : 
  ∃ x : ℤ, x % 2 = 1 ∧ (x : ℝ) ∈ A_P P :=
sorry

end NUMINAMATH_CALUDE_min_odd_in_A_P_l4061_406139


namespace NUMINAMATH_CALUDE_dice_collinearity_probability_l4061_406102

def dice_roll := Finset.range 6

def vector_p (m n : ℕ) := (m, n)
def vector_q := (3, 6)

def is_collinear (p q : ℕ × ℕ) : Prop :=
  p.1 * q.2 = p.2 * q.1

def collinear_outcomes : Finset (ℕ × ℕ) :=
  {(1, 2), (2, 4), (3, 6)}

theorem dice_collinearity_probability :
  (collinear_outcomes.card : ℚ) / (dice_roll.card * dice_roll.card : ℚ) = 1 / 12 :=
sorry

end NUMINAMATH_CALUDE_dice_collinearity_probability_l4061_406102


namespace NUMINAMATH_CALUDE_inequality_range_l4061_406113

theorem inequality_range (m : ℝ) :
  (∀ x : ℝ, m * x^2 - m * x + 1 > 0) ↔ (0 ≤ m ∧ m < 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l4061_406113


namespace NUMINAMATH_CALUDE_probability_two_nondefective_pens_l4061_406164

/-- Given a box of 8 pens with 3 defective pens, the probability of selecting 2 non-defective pens without replacement is 5/14. -/
theorem probability_two_nondefective_pens (total_pens : Nat) (defective_pens : Nat) 
  (h1 : total_pens = 8) (h2 : defective_pens = 3) :
  let nondefective_pens := total_pens - defective_pens
  let prob_first := nondefective_pens / total_pens
  let prob_second := (nondefective_pens - 1) / (total_pens - 1)
  prob_first * prob_second = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_nondefective_pens_l4061_406164


namespace NUMINAMATH_CALUDE_mary_hospital_time_l4061_406134

/-- Given the conditions of Mary's ambulance ride and Don's drive to the hospital,
    prove that Mary reaches the hospital in 15 minutes. -/
theorem mary_hospital_time (ambulance_speed : ℝ) (don_speed : ℝ) (don_time : ℝ) :
  ambulance_speed = 60 →
  don_speed = 30 →
  don_time = 0.5 →
  (don_speed * don_time) / ambulance_speed = 0.25 := by
  sorry

#check mary_hospital_time

end NUMINAMATH_CALUDE_mary_hospital_time_l4061_406134


namespace NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l4061_406119

/-- Given functions f and g, prove that if for any x₁ in [0, 2], 
    there exists x₂ in [1, 2] such that f(x₁) ≥ g(x₂), then m ≥ 1/4 -/
theorem function_inequality_implies_m_bound 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (m : ℝ)
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x, g x = (1/2)^x - m)
  (h : ∀ x₁ ∈ Set.Icc 0 2, ∃ x₂ ∈ Set.Icc 1 2, f x₁ ≥ g x₂) :
  m ≥ 1/4 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_implies_m_bound_l4061_406119


namespace NUMINAMATH_CALUDE_factorization_theorem_l4061_406144

theorem factorization_theorem (a b : ℝ) : 6 * a * b - a^2 - 9 * b^2 = -(a - 3 * b)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l4061_406144


namespace NUMINAMATH_CALUDE_hall_area_l4061_406143

/-- The area of a rectangular hall with specific proportions -/
theorem hall_area : 
  ∀ (length width : ℝ),
  width = (1/2) * length →
  length - width = 20 →
  length * width = 800 := by
sorry

end NUMINAMATH_CALUDE_hall_area_l4061_406143


namespace NUMINAMATH_CALUDE_ceiling_product_equation_l4061_406194

theorem ceiling_product_equation : ∃ x : ℝ, ⌈x⌉ * x = 220 ∧ x = 220 / 15 := by sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_l4061_406194


namespace NUMINAMATH_CALUDE_circle_equation_k_range_l4061_406153

theorem circle_equation_k_range :
  ∀ k : ℝ, (∃ x y : ℝ, x^2 + y^2 + 2*k*x + 4*y + 3*k + 8 = 0 ∧ 
    ∃ h r : ℝ, ∀ x' y' : ℝ, (x' - h)^2 + (y' - r)^2 = (x^2 + y^2 + 2*k*x + 4*y + 3*k + 8)) 
  ↔ (k > 4 ∨ k < -1) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_k_range_l4061_406153


namespace NUMINAMATH_CALUDE_problem_solution_l4061_406122

theorem problem_solution (a r : ℝ) (h1 : a * r = 24) (h2 : a * r^4 = 3) : a = 48 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4061_406122


namespace NUMINAMATH_CALUDE_interest_groups_intersection_difference_l4061_406100

theorem interest_groups_intersection_difference (total : ℕ) (math : ℕ) (english : ℕ)
  (h_total : total = 200)
  (h_math : math = 80)
  (h_english : english = 155) :
  (min math english) - (math + english - total) = 45 :=
sorry

end NUMINAMATH_CALUDE_interest_groups_intersection_difference_l4061_406100


namespace NUMINAMATH_CALUDE_expression_factorization_l4061_406193

variable (x : ℝ)

theorem expression_factorization :
  (12 * x^3 + 27 * x^2 + 90 * x - 9) - (-3 * x^3 + 9 * x^2 - 15 * x - 9) =
  3 * x * (5 * x^2 + 6 * x + 35) := by sorry

end NUMINAMATH_CALUDE_expression_factorization_l4061_406193


namespace NUMINAMATH_CALUDE_floor_sqrt_19_squared_l4061_406130

theorem floor_sqrt_19_squared : ⌊Real.sqrt 19⌋^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_19_squared_l4061_406130


namespace NUMINAMATH_CALUDE_inscribed_pentagon_segment_lengths_l4061_406197

/-- A convex pentagon with an inscribed circle -/
structure InscribedPentagon where
  -- Side lengths
  fg : ℝ
  gh : ℝ
  hi : ℝ
  ij : ℝ
  jf : ℝ
  -- Convexity and inscribed circle conditions would be added here in a more complete model

/-- The theorem stating the existence of x, y, z satisfying the given conditions -/
theorem inscribed_pentagon_segment_lengths (p : InscribedPentagon) 
  (h_fg : p.fg = 7)
  (h_gh : p.gh = 8)
  (h_hi : p.hi = 8)
  (h_ij : p.ij = 8)
  (h_jf : p.jf = 9) :
  ∃ x y z : ℝ,
    x + y = 8 ∧
    x + z = 7 ∧
    y + z = 9 ∧
    x = 3 ∧ y = 5 ∧ z = 4 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_pentagon_segment_lengths_l4061_406197


namespace NUMINAMATH_CALUDE_cylinder_minimal_material_l4061_406157

/-- For a cylindrical beverage can with a fixed volume, the material used is minimized when the base radius is half the height -/
theorem cylinder_minimal_material (V : ℝ) (h R : ℝ) (h_pos : h > 0) (R_pos : R > 0) :
  V = π * R^2 * h → (∀ R' h', V = π * R'^2 * h' → 2 * π * R^2 + 2 * π * R * h ≤ 2 * π * R'^2 + 2 * π * R' * h') ↔ R = h / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minimal_material_l4061_406157


namespace NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l4061_406189

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end NUMINAMATH_CALUDE_ceiling_neg_sqrt_64_over_9_l4061_406189


namespace NUMINAMATH_CALUDE_trisha_chicken_expense_l4061_406170

/-- Given Trisha's shopping expenses and initial amount, prove that she spent $22 on chicken -/
theorem trisha_chicken_expense (meat_cost veggies_cost eggs_cost dog_food_cost initial_amount remaining_amount : ℕ) 
  (h1 : meat_cost = 17)
  (h2 : veggies_cost = 43)
  (h3 : eggs_cost = 5)
  (h4 : dog_food_cost = 45)
  (h5 : initial_amount = 167)
  (h6 : remaining_amount = 35) :
  initial_amount - remaining_amount - (meat_cost + veggies_cost + eggs_cost + dog_food_cost) = 22 := by
  sorry

end NUMINAMATH_CALUDE_trisha_chicken_expense_l4061_406170


namespace NUMINAMATH_CALUDE_christmas_tree_lights_l4061_406127

theorem christmas_tree_lights (total : ℕ) (red : ℕ) (yellow : ℕ) (blue : ℕ) : 
  total = 95 → red = 26 → yellow = 37 → blue = total - red - yellow → blue = 32 := by
  sorry

end NUMINAMATH_CALUDE_christmas_tree_lights_l4061_406127


namespace NUMINAMATH_CALUDE_h_nonzero_l4061_406152

/-- A polynomial of degree 4 with four distinct roots, one of which is 0 -/
structure QuarticPolynomial where
  f : ℝ
  g : ℝ
  h : ℝ
  roots : Finset ℝ
  distinct_roots : roots.card = 4
  zero_root : (0 : ℝ) ∈ roots
  is_root (x : ℝ) : x ∈ roots → x^4 + f*x^3 + g*x^2 + h*x = 0

theorem h_nonzero (Q : QuarticPolynomial) : Q.h ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_h_nonzero_l4061_406152


namespace NUMINAMATH_CALUDE_sin_transformation_l4061_406176

theorem sin_transformation (x : ℝ) : 
  Real.sin (2 * x + π / 3) = Real.sin (x - π / 3) := by
  sorry

end NUMINAMATH_CALUDE_sin_transformation_l4061_406176


namespace NUMINAMATH_CALUDE_sin_q_in_special_right_triangle_l4061_406184

/-- Given a right triangle PQR with ∠P as the right angle, PR = 40, and QR = 41, prove that sin Q = 9/41 -/
theorem sin_q_in_special_right_triangle (P Q R : ℝ × ℝ) : 
  let pq := Real.sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)
  let pr := Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2)
  let qr := Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2)
  (pq^2 + pr^2 = qr^2) →  -- Right angle at P
  pr = 40 →
  qr = 41 →
  (Q.2 - P.2) / qr = 9 / 41 :=
by sorry

end NUMINAMATH_CALUDE_sin_q_in_special_right_triangle_l4061_406184


namespace NUMINAMATH_CALUDE_price_after_two_reductions_l4061_406126

/-- Represents the relationship between the initial price, reduction percentage, and final price after two reductions. -/
theorem price_after_two_reductions 
  (initial_price : ℝ) 
  (reduction_percentage : ℝ) 
  (final_price : ℝ) 
  (h1 : initial_price = 2) 
  (h2 : 0 ≤ reduction_percentage ∧ reduction_percentage < 1) :
  final_price = initial_price * (1 - reduction_percentage)^2 :=
by sorry

#check price_after_two_reductions

end NUMINAMATH_CALUDE_price_after_two_reductions_l4061_406126


namespace NUMINAMATH_CALUDE_linear_system_solution_l4061_406163

theorem linear_system_solution (x y a : ℝ) : 
  (3 * x + y = a) → 
  (x - 2 * y = 1) → 
  (2 * x + 3 * y = 2) → 
  (a = 3) := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l4061_406163


namespace NUMINAMATH_CALUDE_triangle_formation_theorem_l4061_406169

/-- Given three positive real numbers a, b, and c, they can form a triangle
    if and only if the sum of any two is greater than the third. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The theorem states that among the given combinations, only (4, 5, 6)
    satisfies the triangle inequality and thus can form a triangle. -/
theorem triangle_formation_theorem :
  ¬ can_form_triangle 2 3 6 ∧
  ¬ can_form_triangle 3 3 6 ∧
  can_form_triangle 4 5 6 ∧
  ¬ can_form_triangle 4 10 6 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_theorem_l4061_406169


namespace NUMINAMATH_CALUDE_smallest_fraction_divides_exactly_l4061_406112

def fraction1 : Rat := 6 / 7
def fraction2 : Rat := 5 / 14
def fraction3 : Rat := 10 / 21
def smallestFraction : Rat := 1 / 42

theorem smallest_fraction_divides_exactly :
  (∃ (n1 n2 n3 : ℕ), fraction1 * n1 = smallestFraction ∧
                     fraction2 * n2 = smallestFraction ∧
                     fraction3 * n3 = smallestFraction) ∧
  (∀ (f : Rat), f > 0 ∧ (∃ (m1 m2 m3 : ℕ), fraction1 * m1 = f ∧
                                           fraction2 * m2 = f ∧
                                           fraction3 * m3 = f) →
                f ≥ smallestFraction) :=
by sorry

end NUMINAMATH_CALUDE_smallest_fraction_divides_exactly_l4061_406112


namespace NUMINAMATH_CALUDE_lilys_calculation_l4061_406131

theorem lilys_calculation (a b c : ℝ) 
  (h1 : a - 2 * (b - 3 * c) = 14) 
  (h2 : a - 2 * b - 3 * c = 2) : 
  a - 2 * b = 6 := by sorry

end NUMINAMATH_CALUDE_lilys_calculation_l4061_406131


namespace NUMINAMATH_CALUDE_max_k_value_l4061_406167

theorem max_k_value (m : ℝ) (h1 : 0 < m) (h2 : m < 1/2) :
  (∀ k : ℝ, 1/m + 2/(1-2*m) ≥ k) → (∃ k : ℝ, k = 8 ∧ ∀ k' : ℝ, (∀ m' : ℝ, 0 < m' ∧ m' < 1/2 → 1/m' + 2/(1-2*m') ≥ k') → k' ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_max_k_value_l4061_406167


namespace NUMINAMATH_CALUDE_weight_of_new_person_l4061_406148

theorem weight_of_new_person (initial_count : ℕ) (replaced_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  replaced_weight = 40 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + replaced_weight = 60 :=
by sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l4061_406148


namespace NUMINAMATH_CALUDE_f_2009_equals_3_l4061_406188

/-- Given a function f and constants a, b, α, β, prove that f(2009) = 3 -/
theorem f_2009_equals_3 
  (f : ℝ → ℝ) 
  (a b α β : ℝ) 
  (h1 : ∀ x, f x = a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 4)
  (h2 : f 2000 = 5) :
  f 2009 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_2009_equals_3_l4061_406188


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4061_406111

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

def increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a)
  (h_incr : increasing_sequence a)
  (h_first : a 1 = -2)
  (h_relation : ∀ n : ℕ, 3 * (a n + a (n + 2)) = 10 * a (n + 1)) :
  ∃ q : ℝ, q = 1/3 ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l4061_406111


namespace NUMINAMATH_CALUDE_gecko_infertile_eggs_percentage_l4061_406141

theorem gecko_infertile_eggs_percentage 
  (total_eggs : ℕ) 
  (hatched_eggs : ℕ) 
  (calcification_rate : ℚ) :
  total_eggs = 30 →
  hatched_eggs = 16 →
  calcification_rate = 1/3 →
  ∃ (infertile_percentage : ℚ),
    infertile_percentage = 20/100 ∧
    hatched_eggs = (total_eggs : ℚ) * (1 - infertile_percentage) * (1 - calcification_rate) :=
by sorry

end NUMINAMATH_CALUDE_gecko_infertile_eggs_percentage_l4061_406141


namespace NUMINAMATH_CALUDE_books_read_difference_result_l4061_406114

/-- The number of books Peter has read more than his brother and Sarah combined -/
def books_read_difference (total_books : ℕ) (peter_percent : ℚ) (brother_percent : ℚ) (sarah_percent : ℚ) : ℚ :=
  (peter_percent * total_books) - ((brother_percent + sarah_percent) * total_books)

/-- Theorem stating the difference in books read -/
theorem books_read_difference_result :
  books_read_difference 50 (60 / 100) (25 / 100) (15 / 100) = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_read_difference_result_l4061_406114


namespace NUMINAMATH_CALUDE_money_left_after_purchase_l4061_406182

/-- The amount of money left after buying a gift and cake for their mother -/
def money_left (gift_cost cake_cost erika_savings : ℚ) : ℚ :=
  let rick_savings := gift_cost / 2
  let total_savings := erika_savings + rick_savings
  let total_cost := gift_cost + cake_cost
  total_savings - total_cost

/-- Theorem stating the amount of money left after buying the gift and cake -/
theorem money_left_after_purchase : 
  money_left 250 25 155 = 5 := by sorry

end NUMINAMATH_CALUDE_money_left_after_purchase_l4061_406182


namespace NUMINAMATH_CALUDE_middle_part_value_l4061_406162

theorem middle_part_value (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 120 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ x : ℚ, x > 0 ∧ a * x + b * x + c * x = total ∧ b * x = 40 :=
by sorry

end NUMINAMATH_CALUDE_middle_part_value_l4061_406162


namespace NUMINAMATH_CALUDE_aaron_brothers_count_l4061_406180

/-- 
Given that Bennett has 6 brothers and the number of Bennett's brothers is two less than twice 
the number of Aaron's brothers, prove that Aaron has 4 brothers.
-/
theorem aaron_brothers_count :
  -- Define the number of Bennett's brothers
  let bennett_brothers : ℕ := 6
  -- Define the relationship between Aaron's and Bennett's brothers
  ∀ aaron_brothers : ℕ, bennett_brothers = 2 * aaron_brothers - 2 →
  -- Prove that Aaron has 4 brothers
  aaron_brothers = 4 := by
sorry

end NUMINAMATH_CALUDE_aaron_brothers_count_l4061_406180


namespace NUMINAMATH_CALUDE_jerry_shelf_theorem_l4061_406171

/-- The number of action figures and books on Jerry's shelf -/
def shelf_contents : ℕ × ℕ := (5, 9)

/-- The number of action figures added later -/
def added_figures : ℕ := 7

/-- The final difference between action figures and books -/
def figure_book_difference : ℤ :=
  (shelf_contents.1 + added_figures : ℤ) - shelf_contents.2

theorem jerry_shelf_theorem :
  figure_book_difference = 3 := by sorry

end NUMINAMATH_CALUDE_jerry_shelf_theorem_l4061_406171


namespace NUMINAMATH_CALUDE_special_parallelogram_perimeter_l4061_406115

/-- A parallelogram with specific properties -/
structure SpecialParallelogram where
  /-- The length of the perpendicular from one vertex to the opposite side -/
  perpendicular : ℝ
  /-- The length of one diagonal -/
  diagonal : ℝ

/-- Theorem: The perimeter of a special parallelogram is 36 -/
theorem special_parallelogram_perimeter 
  (P : SpecialParallelogram) 
  (h1 : P.perpendicular = 12) 
  (h2 : P.diagonal = 15) : 
  Real.sqrt ((P.diagonal ^ 2 - P.perpendicular ^ 2) / 4) * 4 = 36 := by
  sorry

#check special_parallelogram_perimeter

end NUMINAMATH_CALUDE_special_parallelogram_perimeter_l4061_406115


namespace NUMINAMATH_CALUDE_paper_folding_cutting_l4061_406146

-- Define a rectangular sheet of paper
structure RectangularSheet :=
  (width : ℝ)
  (height : ℝ)
  (width_pos : width > 0)
  (height_pos : height > 0)

-- Define the operation of flipping one edge 180 degrees
def flip_edge (sheet : RectangularSheet) : RectangularSheet := sheet

-- Define the operation of cutting along specific lines
def cut_along_lines (sheet : RectangularSheet) : Set ℝ × ℝ := sorry

-- Define the desired figure
def desired_figure : Set ℝ × ℝ := sorry

-- Theorem statement
theorem paper_folding_cutting 
  (sheet : RectangularSheet) : 
  cut_along_lines (flip_edge sheet) = desired_figure := by sorry

end NUMINAMATH_CALUDE_paper_folding_cutting_l4061_406146


namespace NUMINAMATH_CALUDE_pr_qs_ratio_l4061_406101

-- Define the points and distances
def P : ℝ := 0
def Q : ℝ := 3
def R : ℝ := 9
def S : ℝ := 20

-- State the theorem
theorem pr_qs_ratio :
  (R - P) / (S - Q) = 9 / 17 := by
  sorry

end NUMINAMATH_CALUDE_pr_qs_ratio_l4061_406101


namespace NUMINAMATH_CALUDE_power_of_power_l4061_406186

theorem power_of_power : (3^3)^4 = 531441 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l4061_406186


namespace NUMINAMATH_CALUDE_quadratic_square_solutions_l4061_406108

theorem quadratic_square_solutions (n : ℕ) : 
  ∃ (p q : ℤ), ∃ (S : Finset ℤ), 
    (Finset.card S = n) ∧ 
    (∀ x : ℤ, x ∈ S ↔ ∃ y : ℕ, x^2 + p * x + q = y^2) ∧
    (∀ x y : ℤ, x ∈ S → y ∈ S → x ≠ y → x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_square_solutions_l4061_406108


namespace NUMINAMATH_CALUDE_cocktail_cost_per_litre_l4061_406158

/-- Calculate the cost per litre of a superfruit juice cocktail --/
theorem cocktail_cost_per_litre 
  (mixed_fruit_cost : ℝ) 
  (acai_cost : ℝ) 
  (mixed_fruit_volume : ℝ) 
  (acai_volume : ℝ) 
  (h1 : mixed_fruit_cost = 262.85)
  (h2 : acai_cost = 3104.35)
  (h3 : mixed_fruit_volume = 32)
  (h4 : acai_volume = 21.333333333333332) : 
  ∃ (cost_per_litre : ℝ), abs (cost_per_litre - 1399.99) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_cocktail_cost_per_litre_l4061_406158


namespace NUMINAMATH_CALUDE_males_band_not_orchestra_l4061_406151

/-- Represents the school band and orchestra at West Valley High -/
structure MusicGroups where
  band_females : ℕ
  band_males : ℕ
  orchestra_females : ℕ
  orchestra_males : ℕ
  both_females : ℕ
  total_students : ℕ

/-- The specific music groups at West Valley High -/
def westValleyHigh : MusicGroups :=
  { band_females := 120
  , band_males := 100
  , orchestra_females := 90
  , orchestra_males := 110
  , both_females := 70
  , total_students := 250 }

/-- The number of males in the band who are not in the orchestra is 0 -/
theorem males_band_not_orchestra (g : MusicGroups) (h : g = westValleyHigh) :
  g.band_males - (g.band_males + g.orchestra_males - (g.total_students - (g.band_females + g.orchestra_females - g.both_females))) = 0 := by
  sorry


end NUMINAMATH_CALUDE_males_band_not_orchestra_l4061_406151


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l4061_406198

theorem simplify_sqrt_expression : Real.sqrt (68 - 28 * Real.sqrt 2) = 6 - 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l4061_406198


namespace NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l4061_406124

theorem complex_exponential_to_rectangular : 
  Complex.exp (Complex.I * (13 * Real.pi / 6)) * (Real.sqrt 3 : ℂ) = (3 / 2 : ℂ) + Complex.I * ((Real.sqrt 3) / 2 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_exponential_to_rectangular_l4061_406124


namespace NUMINAMATH_CALUDE_inequality_equivalence_f_less_than_one_l4061_406105

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Part I: Equivalence of the inequality
theorem inequality_equivalence (x : ℝ) : f x < x + 1 ↔ 0 < x ∧ x < 2 := by
  sorry

-- Part II: Prove f(x) < 1 under given conditions
theorem f_less_than_one (x y : ℝ) 
  (h1 : |x - y - 1| ≤ 1/3) (h2 : |2*y + 1| ≤ 1/6) : f x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_f_less_than_one_l4061_406105


namespace NUMINAMATH_CALUDE_coin_trick_theorem_l4061_406142

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a sequence of coins -/
def CoinSequence (n : ℕ) := Fin n → CoinState

/-- Represents the strategy for the assistant and magician -/
structure Strategy (n : ℕ) where
  encode : CoinSequence n → Fin n → Fin n
  decode : CoinSequence n → Fin n

/-- Defines when a strategy is valid -/
def is_valid_strategy (n : ℕ) (s : Strategy n) : Prop :=
  ∀ (seq : CoinSequence n) (chosen : Fin n),
    ∃ (flipped : Fin n),
      s.decode (Function.update seq flipped (CoinState.Tails)) = chosen

/-- The main theorem: the trick is possible iff n is a power of 2 -/
theorem coin_trick_theorem (n : ℕ) :
  (∃ (s : Strategy n), is_valid_strategy n s) ↔ ∃ (k : ℕ), n = 2^k :=
sorry

end NUMINAMATH_CALUDE_coin_trick_theorem_l4061_406142


namespace NUMINAMATH_CALUDE_circles_externally_tangent_l4061_406199

/-- Two circles are externally tangent if the distance between their centers
    equals the sum of their radii -/
def externally_tangent (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ) : Prop :=
  (c1.1 - c2.1)^2 + (c1.2 - c2.2)^2 = (r1 + r2)^2

/-- The first circle: x^2 + y^2 = 4 -/
def circle1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

/-- The second circle: x^2 + y^2 - 2mx + m^2 - 1 = 0 -/
def circle2 (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*m*p.1 + m^2 - 1 = 0}

theorem circles_externally_tangent :
  ∀ m : ℝ, (∃ p : ℝ × ℝ, p ∈ circle1 ∩ circle2 m) ∧
           externally_tangent (0, 0) (m, 0) 2 1 ↔ m = 3 ∨ m = -3 :=
sorry

end NUMINAMATH_CALUDE_circles_externally_tangent_l4061_406199


namespace NUMINAMATH_CALUDE_cubic_equation_product_l4061_406137

theorem cubic_equation_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) 
  (h₁ : x₁^3 - 3*x₁*y₁^2 = 2006) (h₂ : y₁^3 - 3*x₁^2*y₁ = 2007)
  (h₃ : x₂^3 - 3*x₂*y₂^2 = 2006) (h₄ : y₂^3 - 3*x₂^2*y₂ = 2007)
  (h₅ : x₃^3 - 3*x₃*y₃^2 = 2006) (h₆ : y₃^3 - 3*x₃^2*y₃ = 2007) :
  (1 - x₁/y₁) * (1 - x₂/y₂) * (1 - x₃/y₃) = 1/1003 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_product_l4061_406137


namespace NUMINAMATH_CALUDE_polynomial_equation_solutions_l4061_406106

-- Define the polynomials p and q
def p (x : ℂ) : ℂ := x^5 + x
def q (x : ℂ) : ℂ := x^5 + x^2

-- Define a primitive third root of unity
noncomputable def ε : ℂ := Complex.exp ((2 * Real.pi * Complex.I) / 3)

-- Define the set of solution pairs
def solution_pairs : Set (ℂ × ℂ) :=
  {(ε, 1 - ε), (ε^2, 1 - ε^2), 
   ((1 + Complex.I * Real.sqrt 3) / 2, (1 - Complex.I * Real.sqrt 3) / 2),
   ((1 - Complex.I * Real.sqrt 3) / 2, (1 + Complex.I * Real.sqrt 3) / 2)}

-- State the theorem
theorem polynomial_equation_solutions :
  ∀ w z : ℂ, w ≠ z → (p w = p z ∧ q w = q z) ↔ (w, z) ∈ solution_pairs :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equation_solutions_l4061_406106


namespace NUMINAMATH_CALUDE_range_of_a_l4061_406109

/-- A function f(x) = -x^2 + 2ax, where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x

/-- The theorem stating the range of a given the conditions on f -/
theorem range_of_a (a : ℝ) :
  (∀ x y, x ∈ Set.Icc 0 1 → y ∈ Set.Icc 0 1 → x < y → f a x < f a y) →
  (∀ x y, x ∈ Set.Icc 2 3 → y ∈ Set.Icc 2 3 → x < y → f a x > f a y) →
  1 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4061_406109


namespace NUMINAMATH_CALUDE_function_inequality_l4061_406120

theorem function_inequality (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 3 * x + 2) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 2| < b → |f x + 4| < a) ↔
  b ≤ a / 3 :=
by sorry

end NUMINAMATH_CALUDE_function_inequality_l4061_406120


namespace NUMINAMATH_CALUDE_sun_valley_combined_population_sun_valley_combined_population_proof_l4061_406136

/-- Proves that the combined population of Sun City and Valley City is 41550 given the conditions in the problem. -/
theorem sun_valley_combined_population : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun willowdale roseville sun x valley =>
    willowdale = 2000 ∧
    roseville = 3 * willowdale - 500 ∧
    sun = 2 * roseville + 1000 ∧
    x = (6 * sun) / 10 ∧
    valley = 4 * x + 750 →
    sun + valley = 41550

/-- Proof of the theorem -/
theorem sun_valley_combined_population_proof : 
  ∃ (willowdale roseville sun x valley : ℕ), 
    sun_valley_combined_population willowdale roseville sun x valley :=
by
  sorry

#check sun_valley_combined_population
#check sun_valley_combined_population_proof

end NUMINAMATH_CALUDE_sun_valley_combined_population_sun_valley_combined_population_proof_l4061_406136


namespace NUMINAMATH_CALUDE_max_value_of_a_l4061_406110

theorem max_value_of_a : 
  (∃ (a : ℝ), ∀ (x : ℝ), x < a → x^2 - 2*x - 3 > 0) ∧ 
  (∀ (a : ℝ), ∃ (x : ℝ), x^2 - 2*x - 3 > 0 ∧ x ≥ a) →
  (∀ (b : ℝ), (∃ (a : ℝ), ∀ (x : ℝ), x < a → x^2 - 2*x - 3 > 0) ∧ 
              (∀ (a : ℝ), ∃ (x : ℝ), x^2 - 2*x - 3 > 0 ∧ x ≥ a) → 
              b ≤ -1) ∧
  (∃ (a : ℝ), a = -1 ∧ 
              (∀ (x : ℝ), x < a → x^2 - 2*x - 3 > 0) ∧ 
              (∃ (x : ℝ), x^2 - 2*x - 3 > 0 ∧ x ≥ a)) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l4061_406110


namespace NUMINAMATH_CALUDE_inequality_proof_l4061_406195

theorem inequality_proof (x y z : ℝ) 
  (h1 : y > 2*z) 
  (h2 : 2*z > 4*x) 
  (h3 : 2*(x^3 + y^3 + z^3) + 15*(x*y^2 + y*z^2 + z*x^2) > 16*(x^2*y + y^2*z + z^2*x) + 2*x*y*z) : 
  4*x + y > 4*z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l4061_406195


namespace NUMINAMATH_CALUDE_janet_fertilizer_spread_rate_l4061_406160

theorem janet_fertilizer_spread_rate 
  (horses : ℕ) 
  (fertilizer_per_horse : ℚ) 
  (acres : ℕ) 
  (fertilizer_per_acre : ℚ) 
  (days : ℕ) 
  (h1 : horses = 80)
  (h2 : fertilizer_per_horse = 5)
  (h3 : acres = 20)
  (h4 : fertilizer_per_acre = 400)
  (h5 : days = 25)
  : (acres : ℚ) / days = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_janet_fertilizer_spread_rate_l4061_406160


namespace NUMINAMATH_CALUDE_no_xy_term_implies_k_eq_two_l4061_406181

/-- Given a polynomial x^2 + kxy + 4x - 2xy + y^2 - 1, if it does not contain the term xy, then k = 2 -/
theorem no_xy_term_implies_k_eq_two (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*x*y + 4*x - 2*x*y + y^2 - 1 = x^2 + 4*x + y^2 - 1) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_xy_term_implies_k_eq_two_l4061_406181


namespace NUMINAMATH_CALUDE_partnership_profit_l4061_406145

/-- A partnership where one partner's investment and time are multiples of the other's -/
structure Partnership where
  /-- Investment of the partner with smaller investment -/
  investment_b : ℕ
  /-- Time period of investment for the partner with smaller investment -/
  time_b : ℕ
  /-- Profit received by the partner with smaller investment -/
  profit_b : ℕ

/-- Calculate the total profit of the partnership -/
def total_profit (p : Partnership) : ℕ :=
  7 * p.profit_b

/-- Theorem stating the total profit of the partnership -/
theorem partnership_profit (p : Partnership) 
  (h1 : p.profit_b = 4000) : 
  total_profit p = 28000 := by
  sorry

end NUMINAMATH_CALUDE_partnership_profit_l4061_406145


namespace NUMINAMATH_CALUDE_parabola_focus_l4061_406103

/-- The parabola equation: x = -1/4 * y^2 + 2 -/
def parabola_equation (x y : ℝ) : Prop := x = -1/4 * y^2 + 2

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Theorem: The focus of the parabola x = -1/4 * y^2 + 2 is at the point (1, 0) -/
theorem parabola_focus :
  ∀ (x y : ℝ), parabola_equation x y →
  let (fx, fy) := focus
  (x - fx)^2 + (y - fy)^2 = (x - (fx + 2))^2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_focus_l4061_406103


namespace NUMINAMATH_CALUDE_exponential_equality_l4061_406172

theorem exponential_equality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : (3 : ℝ) ^ x = (4 : ℝ) ^ y)
  (h2 : (4 : ℝ) ^ y = (6 : ℝ) ^ z) : 
  (1 / x + 1 / z = 2 / y) ∧ 
  ((3 : ℝ) ^ x < (4 : ℝ) ^ y ∧ (4 : ℝ) ^ y < (6 : ℝ) ^ z) := by
  sorry

end NUMINAMATH_CALUDE_exponential_equality_l4061_406172


namespace NUMINAMATH_CALUDE_water_tank_solution_l4061_406129

/-- Represents the water tank problem --/
def WaterTankProblem (tankCapacity : ℝ) (initialFill : ℝ) (firstDayCollection : ℝ) (thirdDayOverflow : ℝ) : Prop :=
  let initialWater := tankCapacity * initialFill
  let afterFirstDay := initialWater + firstDayCollection
  let secondDayCollection := tankCapacity - afterFirstDay
  secondDayCollection - firstDayCollection = 30

/-- Theorem statement for the water tank problem --/
theorem water_tank_solution :
  WaterTankProblem 100 (2/5) 15 25 := by
  sorry


end NUMINAMATH_CALUDE_water_tank_solution_l4061_406129


namespace NUMINAMATH_CALUDE_parabola_standard_equation_l4061_406185

/-- A parabola with vertex at the origin and axis of symmetry along coordinate axes -/
structure Parabola where
  focus_on_line : ∃ (x y : ℝ), 2*x - y - 4 = 0

/-- The standard equation of a parabola -/
inductive StandardEquation where
  | vert : StandardEquation  -- y² = 8x
  | horz : StandardEquation  -- x² = -16y

/-- Theorem: Given a parabola with vertex at origin, axis of symmetry along coordinate axes,
    and focus on the line 2x - y - 4 = 0, its standard equation is either y² = 8x or x² = -16y -/
theorem parabola_standard_equation (p : Parabola) : 
  ∃ (eq : StandardEquation), 
    (eq = StandardEquation.vert ∨ eq = StandardEquation.horz) := by
  sorry

end NUMINAMATH_CALUDE_parabola_standard_equation_l4061_406185


namespace NUMINAMATH_CALUDE_comic_arrangement_count_l4061_406168

/-- The number of ways to arrange comics as described in the problem -/
def comic_arrangements (batman : ℕ) (xmen : ℕ) (calvin_hobbes : ℕ) : ℕ :=
  (Nat.factorial (batman + xmen)) * (Nat.factorial calvin_hobbes) * 2

/-- Theorem stating the correct number of arrangements for the given comic counts -/
theorem comic_arrangement_count :
  comic_arrangements 7 6 5 = 1494084992000 := by
  sorry

end NUMINAMATH_CALUDE_comic_arrangement_count_l4061_406168


namespace NUMINAMATH_CALUDE_even_sin_function_phi_l4061_406138

theorem even_sin_function_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x, f x = Real.sin ((x + φ) / 3)) →
  (0 ≤ φ ∧ φ ≤ 2 * Real.pi) →
  (∀ x, f x = f (-x)) →
  φ = 3 * Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_even_sin_function_phi_l4061_406138


namespace NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l4061_406178

def A : ℕ := 123456
def B : ℕ := 171428
def M : ℕ := 1000000
def N : ℕ := 863347

theorem multiplicative_inverse_modulo :
  (A + B) * N ≡ 1 [MOD M] :=
sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_modulo_l4061_406178


namespace NUMINAMATH_CALUDE_largest_prime_to_check_for_primality_l4061_406154

theorem largest_prime_to_check_for_primality (n : ℕ) :
  2500 ≤ n → n ≤ 2600 →
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q : ℕ, Nat.Prime q → q^2 ≤ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q : ℕ, Nat.Prime q → q^2 ≤ n → q ≤ p ∧ p = 47) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_to_check_for_primality_l4061_406154


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l4061_406155

/-- Given two vectors a and b in R², prove that if they are perpendicular
    and a = (1, -2) and b = (m, m+2), then m = -4. -/
theorem perpendicular_vectors_m_value
  (a b : ℝ × ℝ)
  (h1 : a = (1, -2))
  (h2 : ∃ m : ℝ, b = (m, m + 2))
  (h3 : a.1 * b.1 + a.2 * b.2 = 0) :
  ∃ m : ℝ, b = (m, m + 2) ∧ m = -4 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l4061_406155


namespace NUMINAMATH_CALUDE_undefined_values_count_l4061_406118

theorem undefined_values_count : ∃ (S : Finset ℝ), 
  (∀ x ∈ S, (x^2 + 2*x - 3) * (x - 3) * (x + 1) = 0) ∧ 
  (∀ x ∉ S, (x^2 + 2*x - 3) * (x - 3) * (x + 1) ≠ 0) ∧ 
  Finset.card S = 4 := by
sorry

end NUMINAMATH_CALUDE_undefined_values_count_l4061_406118


namespace NUMINAMATH_CALUDE_conference_games_l4061_406125

/-- Calculates the total number of games in a sports conference season -/
def total_games (total_teams : ℕ) (teams_per_division : ℕ) (intra_division_games : ℕ) (inter_division_games : ℕ) : ℕ :=
  let divisions := total_teams / teams_per_division
  let intra_division_total := divisions * teams_per_division * (teams_per_division - 1) / 2 * intra_division_games
  let inter_division_total := total_teams * (total_teams - teams_per_division) / 2 * inter_division_games
  intra_division_total + inter_division_total

/-- The theorem stating the total number of games in the specific conference setup -/
theorem conference_games : total_games 18 9 3 2 = 378 := by
  sorry

end NUMINAMATH_CALUDE_conference_games_l4061_406125


namespace NUMINAMATH_CALUDE_tangent_line_and_extrema_l4061_406179

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x - x

theorem tangent_line_and_extrema :
  let a : ℝ := 0
  let b : ℝ := Real.pi / 2
  -- Tangent line at (0, f(0)) is y = 1
  (∀ y, HasDerivAt f 0 y → y = 0) ∧
  f 0 = 1 ∧
  -- Maximum value is 1 at x = 0
  (∀ x ∈ Set.Icc a b, f x ≤ f 0) ∧
  -- Minimum value is -π/2 at x = π/2
  (∀ x ∈ Set.Icc a b, f b ≤ f x) ∧
  f b = -Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_extrema_l4061_406179


namespace NUMINAMATH_CALUDE_tea_house_payment_l4061_406192

theorem tea_house_payment (t k b : ℕ+) (h : 11 ∣ (3 * t + 4 * k + 5 * b)) :
  11 ∣ (9 * t + k + 4 * b) := by
  sorry

end NUMINAMATH_CALUDE_tea_house_payment_l4061_406192


namespace NUMINAMATH_CALUDE_triangle_solutions_l4061_406133

theorem triangle_solutions (a b : ℝ) (B : ℝ) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  B = 45 * π / 180 →
  ∃ (A C c : ℝ),
    ((A = 60 * π / 180 ∧ C = 75 * π / 180 ∧ c = (Real.sqrt 2 + Real.sqrt 6) / 2) ∨
     (A = 120 * π / 180 ∧ C = 15 * π / 180 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) ∧
    A + B + C = π ∧
    a / Real.sin A = b / Real.sin B ∧
    a / Real.sin A = c / Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_solutions_l4061_406133


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l4061_406173

theorem cubic_equation_solutions : 
  let z₁ : ℂ := -3
  let z₂ : ℂ := (3/2) + (3*I*Real.sqrt 3)/2
  let z₃ : ℂ := (3/2) - (3*I*Real.sqrt 3)/2
  (z₁^3 = -27 ∧ z₂^3 = -27 ∧ z₃^3 = -27) ∧
  (∀ z : ℂ, z^3 = -27 → z = z₁ ∨ z = z₂ ∨ z = z₃) := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l4061_406173
